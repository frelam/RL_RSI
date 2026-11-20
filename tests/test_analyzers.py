from __future__ import annotations

import tempfile
import unittest

from rsi.core.config import Config
from rsi.core.types import BenchmarkReport, CollectResult, ItemResult
from rsi.modules.analyze_llm import LLMJudgeAnalyzer
from rsi.modules.analyze_merge import DefectMerger
from rsi.modules.analyze_rule import RuleCapabilityAnalyzer
from rsi.modules.metrics import capability_table, overall_metrics
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


if __name__ == "__main__":
    unittest.main()
