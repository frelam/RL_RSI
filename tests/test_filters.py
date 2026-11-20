from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rsi.core.config import Config
from rsi.core.types import DataSample
from rsi.modules.data_difficulty import RuleDifficultyGrader
from rsi.modules.data_filters import (
    DedupFilter,
    FilterPipeline,
    FormatFilter,
    JudgeQualityFilter,
    LengthFilter,
    PIIFilter,
    VerifierFilter,
)
from tests.helpers import REPO_ROOT, FakeRuntime


def make_sample(sample_id: str = "s1", **kwargs) -> DataSample:
    values = {
        "id": sample_id,
        "kind": "text",
        "capability": "code.debugging",
        "prompt": "Explain and fix the failing test in detail.",
        "response": "The bug is an off-by-one error; the fixed code is provided.",
        "reference": None,
        "group": "code.debugging",
    }
    values.update(kwargs)
    return DataSample(**values)


def cfg(data: dict) -> Config:
    return Config(data, config_dir=REPO_ROOT, repo_root=REPO_ROOT)


class FilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = FakeRuntime(self.temp.name)
        self.ctx = SimpleNamespace(round=0)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_pii_drop_and_redact(self) -> None:
        dropper = PIIFilter(self.runtime, cfg({"action": "drop"}))
        sample = make_sample(prompt="contact me at a@b.com please")
        batch = dropper.filter([sample], self.ctx)
        self.assertEqual(batch.kept, [])
        self.assertEqual(len(batch.dropped), 1)

        redactor = PIIFilter(self.runtime, cfg({"action": "redact"}))
        sample2 = make_sample(prompt="contact me at a@b.com please")
        batch2 = redactor.filter([sample2], self.ctx)
        self.assertEqual(len(batch2.kept), 1)
        self.assertNotIn("a@b.com", sample2.prompt)

    def test_length_and_format(self) -> None:
        length = LengthFilter(self.runtime, cfg({"min_prompt_chars": 100}))
        self.assertEqual(length.filter([make_sample()], self.ctx).kept, [])
        fmt = FormatFilter(self.runtime, cfg({"require_response": True}))
        self.assertEqual(fmt.filter([make_sample(response="")], self.ctx).kept, [])
        self.assertEqual(len(fmt.filter([make_sample()], self.ctx).kept), 1)

    def test_dedup_exact_and_cross_round(self) -> None:
        dedup = DedupFilter(self.runtime, cfg({"cross_round": True, "threshold": 0.99}))
        first = dedup.filter([make_sample("a")], self.ctx)
        self.assertEqual(len(first.kept), 1)
        second = dedup.filter([make_sample("b")], self.ctx)
        self.assertEqual(len(second.kept), 0)
        third = dedup.filter([make_sample("c", prompt="A completely different prompt about matrices.")], self.ctx)
        self.assertEqual(len(third.kept), 1)

    def test_judge_quality_sets_score(self) -> None:
        judge = JudgeQualityFilter(
            self.runtime,
            cfg({"prompt": "sample_quality", "profile": "judge", "min_score": 0.5}),
        )
        sample = make_sample()
        batch = judge.filter([sample], self.ctx)
        self.assertEqual(len(batch.kept), 1)
        self.assertAlmostEqual(sample.quality, 0.9)

    def test_verifier(self) -> None:
        verifier = VerifierFilter(self.runtime, cfg({"min_score": 1.0, "require_reference": True}))
        good = make_sample(response="42", reference="42", verifier={"name": "numeric"})
        bad = make_sample("bad", response="41", reference="42", verifier={"name": "numeric"})
        no_ref = make_sample("noref", reference=None)
        batch = verifier.filter([good, bad, no_ref], self.ctx)
        self.assertEqual([s.id for s in batch.kept], ["s1"])
        self.assertEqual(len(batch.dropped), 2)

    def test_rule_difficulty_is_bounded_and_ordered(self) -> None:
        grader = RuleDifficultyGrader(self.runtime, cfg({}))
        easy = make_sample(prompt="Say hi.", response="hi")
        hard = make_sample(
            prompt="Step by step, prove the claim. 1. First lemma. 2. Second lemma. 3. Finally combine. " * 3,
            response="Because of the lemmas, therefore the claim holds. ```proof``` " * 3,
            kind="agentic",
            metadata={"tool_calls": 8, "has_code": True},
        )
        easy_score = grader.grade(easy, self.ctx)
        hard_score = grader.grade(hard, self.ctx)
        self.assertGreaterEqual(easy_score, 0.0)
        self.assertLessEqual(easy_score, 1.0)
        self.assertGreater(hard_score, easy_score)

    def test_filter_pipeline_stats(self) -> None:
        pipeline = FilterPipeline(
            self.runtime,
            cfg(
                {
                    "pipeline": [
                        {"type": "length", "min_prompt_chars": 10},
                        {"type": "dedup", "threshold": 0.99},
                    ]
                }
            ),
        )
        samples = [make_sample("a"), make_sample("b"), make_sample("c", prompt="short")]
        kept, decisions, stats = pipeline.run(samples, self.ctx)
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats["total"]["input"], 3)
        self.assertTrue(any(d.filter == "dedup" and not d.keep for d in decisions))


if __name__ == "__main__":
    unittest.main()
