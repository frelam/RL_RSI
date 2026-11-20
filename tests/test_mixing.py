from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

from rsi.core.config import Config
from rsi.core.types import DataPlan, DataSample
from rsi.modules.data_mixing import DiversitySampler, HistoryBanditMixing
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
