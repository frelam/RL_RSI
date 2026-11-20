from __future__ import annotations

import tempfile
import unittest

from rsi.core.config import Config
from rsi.core.types import TrainingResult
from rsi.modules.train_monitor import MonitorHub, Supervisor
from tests.helpers import REPO_ROOT, FakeRuntime


def cfg(data: dict) -> Config:
    return Config(data, config_dir=REPO_ROOT, repo_root=REPO_ROOT)


class MonitorTests(unittest.TestCase):
    def test_entropy_collapse_stops_training(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = FakeRuntime(td)
            hub = MonitorHub(
                runtime,
                cfg(
                    {
                        "stop_on_critical": True,
                        "algorithms": [
                            {
                                "type": "entropy_collapse",
                                "threshold": 0.2,
                                "critical_threshold": 0.05,
                                "patience": 3,
                            }
                        ],
                    }
                ),
            )
            self.assertIsNone(hub.observe({"step": 1, "metrics": {"entropy": 0.04}}))
            self.assertIsNone(hub.observe({"step": 2, "metrics": {"entropy": 0.04}}))
            self.assertEqual(hub.observe({"step": 3, "metrics": {"entropy": 0.04}}), "stop")
            findings = hub.snapshot()
            self.assertEqual(findings[0]["code"], "entropy_collapse")
            self.assertEqual(findings[0]["severity"], "critical")

    def test_nan_guard(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = FakeRuntime(td)
            hub = MonitorHub(
                runtime,
                cfg({"stop_on_critical": False, "algorithms": [{"type": "nan_guard"}]}),
            )
            hub.observe({"step": 1, "metrics": {"loss": float("nan")}})
            self.assertEqual(hub.snapshot()[0]["code"], "nan_metric")

    def test_supervisor_retries_then_uses_best(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = FakeRuntime(td)
            supervisor = Supervisor(
                runtime,
                cfg(
                    {
                        "enabled": True,
                        "max_retries": 1,
                        "retrain_on_critical": True,
                        "discard_on_critical": False,
                    }
                ),
            )
            result = TrainingResult(
                job_id="j",
                status="stopped_by_monitor",
                checkpoint_path="/tmp/ckpt",
                metrics=[],
            )
            findings = [
                {
                    "code": "entropy_collapse",
                    "severity": "critical",
                    "suggestion": {"entropy_coeff": 0.002},
                }
            ]
            first = supervisor.verdict(result, findings, attempt=0)
            self.assertTrue(first.retrain)
            self.assertEqual(first.adjustments["entropy_coeff"], 0.002)
            second = supervisor.verdict(result, findings, attempt=1)
            self.assertFalse(second.retrain)
            self.assertTrue(second.use_checkpoint)

    def test_supervisor_rejects_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = FakeRuntime(td)
            supervisor = Supervisor(runtime, cfg({"max_retries": 0}))
            result = TrainingResult(job_id="j", status="failed", checkpoint_path=None)
            verdict = supervisor.verdict(result, [], attempt=0)
            self.assertFalse(verdict.use_checkpoint)
            self.assertFalse(verdict.retrain)


if __name__ == "__main__":
    unittest.main()
