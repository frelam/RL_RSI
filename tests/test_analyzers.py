from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from rsi.core.config import Config
from rsi.core.errors import ConfigError
from rsi.core.types import BenchmarkReport, CollectResult, ItemResult
from rsi.modules.analyze_coverage import MissingCapabilityAnalyzer, load_reference, reference_at
from rsi.modules.analyze_llm import LLMJudgeAnalyzer
from rsi.modules.analyze_merge import DefectMerger
from rsi.modules.analyze_rule import RuleCapabilityAnalyzer
from rsi.modules.analyze_trajectory import TrajectoryAbilityAnalyzer
from rsi.modules.metrics import capability_table, overall_metrics
from rsi.hubs.catalog import BenchmarkCatalog
from tests.helpers import REPO_ROOT, FakeRuntime


def cfg(data: dict) -> Config:
    return Config(data, config_dir=REPO_ROOT, repo_root=REPO_ROOT)


def build_collect(runtime: FakeRuntime, td: str) -> CollectResult:
    items = []
    for index in range(4):
        items.append(
            ItemResult(
                f"code-{index}",
                "code.debugging",
                score=0.2,
                passed=False,
                response="wrong",
            )
        )
    items.append(ItemResult("code-general", "general", score=0.1, passed=False, response="wrong"))
    for index in range(3):
        items.append(
            ItemResult(
                f"long-{index}",
                "long_context.retrieval",
                score=0.25,
                passed=False,
                response="near miss",
            )
        )
    items.append(ItemResult("math-0", "math.reasoning", score=0.8, passed=True, response="ok"))
    items.append(ItemResult("math-1", "math.reasoning", score=0.7, passed=True, response="ok"))
    report = BenchmarkReport(
        benchmark="demo",
        checkpoint_id="ckpt_v000",
        round=0,
        items=items,
        num_items=len(items),
    )
    table = capability_table([report], {}, runtime.catalog)
    aggregate = overall_metrics([report], table)
    report.capabilities = {name: score.score for name, score in table.items()}
    return CollectResult(
        round=0,
        checkpoint_id="ckpt_v000",
        reports=[report],
        capabilities=table,
        aggregate=aggregate,
    )


class AnalyzerTests(unittest.TestCase):
    def test_trajectory_analyzer_returns_catalog_limited_ability_priorities(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = FakeRuntime(td)
            collect = build_collect(runtime, td)
            collect.trajectories = [
                __import__("rsi.core.types", fromlist=["Trajectory"]).Trajectory(
                    id="t1", prompt="debug this code", messages=[], final_answer="failed", reward=0.0
                )
            ]
            report = TrajectoryAbilityAnalyzer(
                runtime, cfg({"prompt": "trajectory_ability_analysis", "profile": "judge"})
            ).analyze(collect, 0, {})
            self.assertEqual(report.sources, ["trajectory_agent"])
            self.assertTrue(set(report.metadata["ability_priorities"]).issubset(
                {item["name"] for item in runtime.catalog.all()}
            ))

    def test_rule_analyzer_finds_low_capabilities_and_maps_general_items(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = FakeRuntime(td)
            collect = build_collect(runtime, td)
            analyzer = RuleCapabilityAnalyzer(
                runtime,
                cfg(
                    {
                        "baseline": "global",
                        "margin": 0.05,
                        "min_gap": 0.01,
                        "min_items": 2,
                        "min_severity": 0.01,
                        "include_subcapabilities": False,
                    }
                ),
            )
            report = analyzer.analyze(collect, 0, {})
            capabilities = {defect.capability for defect in report.defects}
            self.assertIn("code.debugging", capabilities)
            self.assertIn("long_context.retrieval", capabilities)
            self.assertNotIn("math.reasoning", capabilities)
            code = next(d for d in report.defects if d.capability == "code.debugging")
            self.assertEqual(code.source, "rule")
            self.assertTrue(code.evidence)

    def test_llm_judge_analyzer_parses_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = FakeRuntime(td)
            collect = build_collect(runtime, td)
            analyzer = LLMJudgeAnalyzer(
                runtime,
                cfg(
                    {
                        "prompt": "badcase_analysis",
                        "profile": "judge",
                        "votes": 1,
                        "top_n": 5,
                        "max_score": 1.0,
                        "min_confidence": 0.0,
                    }
                ),
            )
            report = analyzer.analyze(collect, 0, {})
            self.assertEqual(len(report.defects), 1)
            defect = report.defects[0]
            self.assertEqual(defect.capability, "long_context.retrieval")
            self.assertEqual(defect.source, "llm_judge")
            self.assertAlmostEqual(defect.severity, 0.6)
            self.assertGreaterEqual(defect.confidence, 0.7)

    def test_merge_corroborates_sources(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = FakeRuntime(td)
            collect = build_collect(runtime, td)
            rule = RuleCapabilityAnalyzer(
                runtime,
                cfg({"margin": 0.05, "min_gap": 0.01, "min_items": 2, "min_severity": 0.01}),
            ).analyze(collect, 0, {})
            judge = LLMJudgeAnalyzer(
                runtime,
                cfg(
                    {
                        "prompt": "badcase_analysis",
                        "profile": "judge",
                        "top_n": 5,
                        "max_score": 1.0,
                        "min_confidence": 0.0,
                    }
                ),
            ).analyze(collect, 0, {})
            merger = DefectMerger(cfg({"min_severity": 0.0, "min_confidence": 0.0}))
            merged = merger.merge(
                [rule, judge], round_no=0, checkpoint_id="ckpt_v000", capabilities=collect.capabilities
            )
            long_defect = next(d for d in merged.defects if d.capability == "long_context.retrieval")
            self.assertIn("rule", long_defect.source)
            self.assertIn("llm_judge", long_defect.source)
            self.assertGreaterEqual(long_defect.severity, 0.6)
            self.assertGreater(long_defect.confidence, 0.8)


def _report(benchmark: str, score: float, capabilities: list[str]) -> BenchmarkReport:
    return BenchmarkReport(
        benchmark=benchmark,
        checkpoint_id="ckpt_v000",
        round=0,
        items=[ItemResult(f"{benchmark}-{i}", "general", score=score) for i in range(2)],
        aggregates={"mean_score": score},
        num_items=2,
        capability_coverage=capabilities,
    )


class MissingCapabilityAnalyzerTests(unittest.TestCase):
    REGISTRY = """
benchmarks:
  good_bench: {capabilities: [math.reasoning, logic.reasoning]}
  bad_bench: {capabilities: [code.debugging]}
  unref_bench: {capabilities: [knowledge.qa]}
"""
    REFERENCE = """
benchmarks:
  good_bench: {4: 0.55, 7: 0.60, 32: 0.80}
  bad_bench: {4: 0.55, 7: 0.60, 32: 0.80}
"""

    def _write(self, td: str, text: str, name: str) -> str:
        path = Path(td) / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def _collect(self, td: str) -> CollectResult:
        reports = [
            _report("good_bench", 0.70, ["math.reasoning", "logic.reasoning"]),
            _report("bad_bench", 0.30, ["code.debugging"]),
            _report("unref_bench", 0.90, ["knowledge.qa"]),
        ]
        table = capability_table(reports, {}, None)
        return CollectResult(
            round=0, checkpoint_id="ckpt_v000", reports=reports, capabilities=table,
            aggregate=overall_metrics(reports, table),
        )

    def _analyzer(self, runtime: FakeRuntime, td: str, **extra: Any) -> MissingCapabilityAnalyzer:
        data = {
            "registry": self._write(td, self.REGISTRY, "bench.yaml"),
            "reference": self._write(td, self.REFERENCE, "reference.yaml"),
            "params_b": 7,
        }
        data.update(extra)
        return MissingCapabilityAnalyzer(runtime, cfg(data))

    def test_marks_capabilities_of_well_performing_benchmarks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = FakeRuntime(td)
            report = self._analyzer(runtime, td).analyze(self._collect(td), 0, {})
        marked = report.metadata["marked_capabilities"]
        self.assertEqual(marked, ["logic.reasoning", "math.reasoning"])
        self.assertFalse(report.metadata["benchmarks"]["bad_bench"]["good"])
        missing = set(report.metadata["missing_capabilities"])
        self.assertEqual(missing, {c["name"] for c in runtime.catalog.all()} - set(marked))
        self.assertIn("knowledge.qa", missing)  # no reference score => cannot mark
        code = next(d for d in report.defects if d.capability == "code.debugging")
        self.assertIn("bad_bench: score 0.30", code.evidence[0])
        self.assertEqual(code.source, "missing_capability")
        self.assertLessEqual(code.severity, 0.6)

    def test_interpolates_reference_and_applies_margin(self) -> None:
        points = [(4.0, 0.55, 0.05), (7.0, 0.60, 0.06), (32.0, 0.80, 0.06)]
        self.assertEqual(reference_at(points, 1.0), (0.55, 0.05))
        self.assertAlmostEqual(reference_at(points, 20.0)[0], 0.704)
        self.assertAlmostEqual(reference_at(points, 20.0)[1], 0.06)
        self.assertEqual(reference_at(points, 100.0), (0.80, 0.06))
        with tempfile.TemporaryDirectory() as td:
            # At 32B the reference is 0.80 and good_bench scores 0.70, which is 0.10 below.
            runtime = FakeRuntime(td)
            report = self._analyzer(runtime, td, params_b=32, margin=0.05).analyze(self._collect(td), 0, {})
            self.assertFalse(report.metadata["benchmarks"]["good_bench"]["good"])
            self.assertIn("math.reasoning", report.metadata["missing_capabilities"])
            runtime = FakeRuntime(td)
            report = self._analyzer(runtime, td, params_b=32, margin=0.15).analyze(self._collect(td), 0, {})
        self.assertNotIn("math.reasoning", report.metadata["missing_capabilities"])

    def test_absolute_mode_and_skip_when_nothing_judged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = FakeRuntime(td)
            analyzer = self._analyzer(runtime, td, reference=None, unreferenced="absolute")
            report = analyzer.analyze(self._collect(td), 0, {})
            self.assertIn("knowledge.qa", report.metadata["marked_capabilities"])
            self.assertIn("math.reasoning", report.metadata["marked_capabilities"])
            analyzer = self._analyzer(runtime, td, reference=None, unreferenced="skip")
            report = analyzer.analyze(self._collect(td), 0, {})
            self.assertEqual(report.defects, [])
            self.assertEqual(report.metadata["missing_capabilities"], [])

    def test_uses_subsets_when_present_and_benchmark_otherwise(self) -> None:
        registry = """
benchmarks:
  bench:
    capabilities: [math.reasoning, logic.reasoning, knowledge.qa]
    subsets:
      math_part: [math.reasoning]
      logic_part: [logic.reasoning]
"""
        # std is 0.50 on logic_part: sigma=1 makes its tolerance 0.50, so 0.20 counts as good.
        reference = """
benchmarks:
  bench: {7: 0.60}
  bench.math_part: {7: [0.60, 0.02]}
  bench.logic_part: {7: [0.60, 0.50]}
"""
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "b.yaml").write_text(registry, encoding="utf-8")
            (Path(td) / "r.yaml").write_text(reference, encoding="utf-8")
            runtime = FakeRuntime(td)
            analyzer = self._analyzer(runtime, td, margin=0.05, sigma=1.0)
            analyzer.registry = BenchmarkCatalog(Path(td) / "b.yaml")
            analyzer.reference = load_reference(Path(td) / "r.yaml")

            def labeled(rows: list[tuple[str, float]]) -> CollectResult:
                report = BenchmarkReport(
                    benchmark="bench", checkpoint_id="ckpt_v000", round=0,
                    items=[ItemResult(f"i{i}", "general", subcapability=sub, score=score) for i, (sub, score) in enumerate(rows)],
                    num_items=len(rows), aggregates={"mean_score": 0.10},
                )
                return CollectResult(round=0, checkpoint_id="ckpt_v000", reports=[report])

            report = analyzer.analyze(labeled([("math_part", 0.70), ("logic_part", 0.20), ("other", 0.90)]), 0, {})
            # Subset rows win over the benchmark score (0.10); an unknown subset falls back to
            # the benchmark capabilities but has no reference, so it is skipped.
            self.assertEqual(report.metadata["marked_capabilities"], ["logic.reasoning", "math.reasoning"])
            self.assertIn("knowledge.qa", report.metadata["missing_capabilities"])
            self.assertEqual(report.metadata["skipped"], ["bench.other"])
            report = analyzer.analyze(labeled([]), 0, {})
            self.assertEqual(report.metadata["benchmarks"], {"bench": {"score": 0.1, "reference": 0.6, "good": False}})
            self.assertNotIn("math.reasoning", report.metadata["marked_capabilities"])

    def test_reads_params_b_from_checkpoint_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            registry = self._write(td, self.REGISTRY, "bench.yaml")
            reference = self._write(td, self.REFERENCE, "reference.yaml")
            runtime = FakeRuntime(
                td, Config({"checkpoint": {"initial": {"params_b": 32}}}, config_dir=REPO_ROOT, repo_root=REPO_ROOT)
            )
            analyzer = MissingCapabilityAnalyzer(runtime, cfg({"registry": registry, "reference": reference}))
            self.assertEqual(analyzer.params_b, 32.0)
            with self.assertRaises(ConfigError):
                MissingCapabilityAnalyzer(FakeRuntime(td), cfg({"registry": registry, "reference": reference}))


if __name__ == "__main__":
    unittest.main()
