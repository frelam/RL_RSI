from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

from rsi.core.config import Config
from rsi.core.jsonio import stable_hash
from rsi.core.types import CapabilityScore, CollectResult, Defect
from rsi.modules.data_curriculum import (
    FixedOrderCurriculum,
    PrerequisiteCurriculum,
    SeverityCurriculum,
)
from tests.helpers import REPO_ROOT, FakeRuntime


def cfg(data: dict) -> Config:
    return Config(data, config_dir=REPO_ROOT, repo_root=REPO_ROOT)


def defect(capability: str, severity: float, confidence: float = 0.8) -> Defect:
    return Defect(
        id=stable_hash(capability),
        capability=capability,
        description=capability,
        severity=severity,
        confidence=confidence,
        source="test",
    )


class CurriculumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = FakeRuntime(self.temp.name)
        self.context = SimpleNamespace(round=0, collect_result=None)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_severity_selects_and_limits(self) -> None:
        policy = SeverityCurriculum(self.runtime, cfg({"max_capabilities": 1}))
        decision = policy.select([defect("a", 0.4), defect("b", 0.9)], self.context)
        self.assertEqual([d.capability for d in decision.defects], ["b"])
        self.assertIn("severity", decision.notes)

    def test_carried_over_capabilities_are_planned_first(self) -> None:
        policy = SeverityCurriculum(self.runtime, cfg({"max_capabilities": 1}))
        context = SimpleNamespace(round=1, collect_result=None, state=SimpleNamespace(pending_capabilities=["a"]))
        decision = policy.select([defect("a", 0.4), defect("b", 0.9)], context)
        self.assertEqual([d.capability for d in decision.defects], ["a"])
        self.assertIn("carried over first: ['a']", decision.notes)

    def test_fixed_order_unlocks_prefix_per_round(self) -> None:
        policy = FixedOrderCurriculum(
            self.runtime,
            cfg({"order": ["a", "b", "c"], "unlock_per_round": 1, "max_capabilities": 8}),
        )
        defects = [defect("a", 0.5), defect("b", 0.6), defect("c", 0.7)]
        first = policy.select(defects, SimpleNamespace(round=0, collect_result=None))
        self.assertEqual([d.capability for d in first.defects], ["a"])
        second = policy.select(defects, SimpleNamespace(round=1, collect_result=None))
        self.assertEqual({d.capability for d in second.defects}, {"a", "b"})
        self.assertEqual(len(second.excluded), 1)

    def test_prerequisite_blocks_until_floor(self) -> None:
        policy = PrerequisiteCurriculum(
            self.runtime,
            cfg(
                {
                    "score_floor": 0.6,
                    "prerequisites": {"long_context.retrieval": ["instruction.following"]},
                }
            ),
        )
        locked_context = SimpleNamespace(
            round=0,
            collect_result=CollectResult(
                round=0,
                checkpoint_id="c",
                capabilities={
                    "instruction.following": CapabilityScore("instruction.following", 0.4),
                },
            ),
        )
        decision = policy.select([defect("long_context.retrieval", 0.8)], locked_context)
        self.assertEqual(decision.defects, [])
        self.assertEqual(len(decision.excluded), 1)

        unlocked_context = SimpleNamespace(
            round=1,
            collect_result=CollectResult(
                round=1,
                checkpoint_id="c",
                capabilities={
                    "instruction.following": CapabilityScore("instruction.following", 0.8),
                },
            ),
        )
        decision2 = policy.select([defect("long_context.retrieval", 0.8)], unlocked_context)
        self.assertEqual(len(decision2.defects), 1)


if __name__ == "__main__":
    unittest.main()
