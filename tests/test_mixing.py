from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from rsi.core.config import Config
from rsi.core.jsonio import read_json, read_jsonl
from rsi.core.types import DataPlan, DataSample
from rsi.modules.data_builder import JsonlDatasetBuilder
from rsi.modules.data_mixing import DiversitySampler, HistoryBanditMixing, StaticMixing
from tests.helpers import REPO_ROOT, FakeRuntime


def cfg(data: dict) -> Config:
    return Config(data, config_dir=REPO_ROOT, repo_root=REPO_ROOT)


def sample(sample_id: str, capability: str, difficulty: float = 0.5) -> DataSample:
    return DataSample(
        id=sample_id,
        kind="text",
        capability=capability,
        prompt=f"prompt {sample_id}",
        response="response",
        group=capability,
        difficulty=difficulty,
    )


class MixingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = FakeRuntime(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_history_bandit_explores_then_exploits(self) -> None:
        mixer = HistoryBanditMixing(
            self.runtime,
            cfg(
                {
                    "explore_rounds": 2,
                    "candidates": [
                        {"a": 1.0, "b": 1.0},
                        {"a": 3.0, "b": 1.0},
                    ],
                }
            ),
        )
        samples = [sample("a1", "a"), sample("b1", "b")]
        plan = DataPlan(round=0)
        state = SimpleNamespace(history=[])
        first = mixer.plan(samples, plan, SimpleNamespace(round=0, state=state))
        self.assertEqual(first, {"a": 1.0, "b": 1.0})
        self.assertEqual(mixer.last_info["candidate"], 0)

        state.history = [
            {"data": {"mixing_candidate": 0}, "validation": {"delta": 0.01}},
            {"data": {"mixing_candidate": 1}, "validation": {"delta": 0.10}},
        ]
        second = mixer.plan(samples, plan, SimpleNamespace(round=2, state=state))
        self.assertEqual(second, {"a": 3.0, "b": 1.0})
        self.assertEqual(mixer.last_info["candidate"], 1)

    def test_diversity_sampler_respects_weights_and_band(self) -> None:
        sampler = DiversitySampler(
            self.runtime,
            cfg(
                {
                    "target_total": 2,
                    "target_band": [0.2, 0.8],
                    "enforce_band": True,
                    "min_per_group": 0,
                    "max_per_group": 0,
                }
            ),
        )
        samples = [
            sample("a1", "a", 0.5),
            sample("a2", "a", 0.6),
            sample("a3", "a", 0.4),
            sample("b1", "b", 0.5),
            sample("easy", "a", 0.01),
            sample("hard", "b", 0.99),
        ]
        selected, stats = sampler.select(samples, {"a": 1.0, "b": 0.0}, SimpleNamespace(round=0))
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(item.group == "a" for item in selected))
        self.assertEqual(stats["band_dropped"], 2)

    def test_ratio_sampler_rejects_mix_outside_tolerance(self) -> None:
        sampler = DiversitySampler(
            self.runtime,
            cfg({
                "target_total": 4,
                "min_per_group": 1,
                "max_per_group": 2,
                "target_band": [0.0, 1.0],
                "preserve_ratio": True,
            }),
        )
        samples = [sample("a1", "a"), sample("a2", "a")] + [sample(f"b{i}", "b") for i in range(1, 8)]
        selected, stats = sampler.select(
            samples, {"a": 9.0, "b": 1.0}, SimpleNamespace(round=0)
        )
        self.assertEqual(selected, [])
        self.assertFalse(stats["ratio_feasible"])

    def test_ratio_sampler_finds_largest_exact_mix_below_cap(self) -> None:
        sampler = DiversitySampler(self.runtime, cfg({
            "target_total": 8,
            "max_per_group": 4,
            "preserve_ratio": True,
            "ratio_tolerance": 0,
        }))
        samples = [sample(f"{group}{i}", group) for group in ("a", "b") for i in range(10)]

        _, stats = sampler.select(samples, {"a": 2, "b": 1}, SimpleNamespace(round=0))

        self.assertEqual(stats["groups"], {"a": 4, "b": 2})

    def test_configured_group_remains_required_when_absent(self) -> None:
        config = cfg({"weights": {"a": 1, "b": 1}, "preserve_ratio": True})
        mixer = StaticMixing(self.runtime, config)
        sampler = DiversitySampler(self.runtime, config)
        samples = [sample("a", "a")]
        context = SimpleNamespace(round=0)
        weights = mixer.plan(samples, DataPlan(round=0), context)

        selected, stats = sampler.select(samples, weights, context)

        self.assertEqual(selected, [])
        self.assertEqual(stats["target_ratio"], {"a": 0.5, "b": 0.5})

    def test_ratio_sampler_relaxes_difficulty_band_to_preserve_groups(self) -> None:
        sampler = DiversitySampler(
            self.runtime,
            cfg({
                "target_total": 0,
                "min_per_group": 1,
                "target_band": [0.4, 0.6],
                "preserve_ratio": True,
                "relax_by": 0.2,
            }),
        )
        exact = sample("exact", "a", 0.5)
        near = sample("near", "b", 0.75)
        selected, stats = sampler.select(
            [exact, near], {"a": 1.0, "b": 1.0}, SimpleNamespace(round=0)
        )
        self.assertEqual({item.id for item in selected}, {"exact", "near"})
        self.assertEqual(stats["relaxed_selected"], 1)
        self.assertEqual(stats["band_dropped"], 0)

    def test_ratio_sampler_uses_minimum_relaxation_needed_for_largest_mix(self) -> None:
        sampler = DiversitySampler(
            self.runtime,
            cfg({
                "target_total": 0,
                "min_per_group": 1,
                "target_band": [0.4, 0.6],
                "preserve_ratio": True,
                "relax_by": 0.1,
                "relax_max_steps": 3,
            }),
        )
        samples = [sample("a", "a", 0.5), sample("b", "b", 0.85)]

        selected, stats = sampler.select(
            samples, {"a": 1.0, "b": 1.0}, SimpleNamespace(round=0)
        )

        self.assertEqual({item.id for item in selected}, {"a", "b"})
        self.assertEqual(stats["relax_steps"], 3)
        self.assertAlmostEqual(stats["relax_amount"], 0.3)
        self.assertEqual(stats["target_ratio"], {"a": 0.5, "b": 0.5})
        self.assertEqual(stats["actual_ratio"], {"a": 0.5, "b": 0.5})

    def test_relaxation_prefers_strict_samples_and_stops_at_smallest_step(self) -> None:
        sampler = DiversitySampler(self.runtime, cfg({
            "target_band": [0.4, 0.6],
            "preserve_ratio": True,
            "ratio_tolerance": 0,
            "relax_by": 0.1,
            "relax_max_steps": 3,
        }))
        samples = [
            sample("strict", "a", 0.5),
            sample("outside", "a", 0.68),
            sample("needed", "b", 0.65),
        ]

        selected, stats = sampler.select(samples, {"a": 1, "b": 1}, SimpleNamespace(round=0))

        self.assertEqual({item.id for item in selected}, {"strict", "needed"})
        self.assertEqual(stats["relax_steps"], 1)
        self.assertEqual(stats["relaxed_selected"], 1)

    def test_builder_preserves_exact_train_ratio_when_validation_rounds_to_one(self) -> None:
        context = SimpleNamespace(round=0, work_dir=Path(self.temp.name))
        plan = DataPlan(round=0)
        weights = {"a": 1.0, "b": 1.0}
        sampler = DiversitySampler(self.runtime, cfg({"preserve_ratio": True, "ratio_tolerance": 0}))
        samples = [sample(f"{group}{i}", group) for group in ("a", "b") for i in range(2)]
        selected, context.sampler_stats = sampler.select(samples, weights, context)
        builder = JsonlDatasetBuilder(self.runtime, cfg({"val_ratio": 0.25, "val_min": 1}))

        dataset = builder.build(selected, context, plan, weights)

        train = Counter(row["group"] for row in read_jsonl(Path(dataset.train_path)))
        self.assertEqual(train, {"a": 2, "b": 2})
        self.assertEqual(dataset.num_val, 0)
        manifest = read_json(Path(dataset.manifest_path))
        self.assertEqual(manifest["stats"]["train_group_ratio"], {"a": 0.5, "b": 0.5})
        self.assertTrue(manifest["stats"]["sampler"]["ratio_feasible"])

    def test_diversity_sampler_keeps_all_when_target_zero(self) -> None:
        sampler = DiversitySampler(
            self.runtime,
            cfg({"target_total": 0, "target_band": [0.0, 1.0], "enforce_band": False}),
        )
        samples = [sample("a1", "a"), sample("b1", "b")]
        selected, stats = sampler.select(samples, {"a": 1.0, "b": 1.0}, SimpleNamespace(round=0))
        self.assertEqual(len(selected), 2)
        self.assertEqual(stats["band_dropped"], 0)


if __name__ == "__main__":
    unittest.main()
