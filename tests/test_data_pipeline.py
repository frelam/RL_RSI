from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rsi.core.config import Config
from rsi.core.jsonio import write_jsonl
from rsi.core.types import DataPlan, DataSample
from rsi.modules.data_builder import VerlParquetDatasetBuilder
from rsi.modules.data_pipeline import DataPipeline
from tests.helpers import REPO_ROOT, FakeRuntime


class ReplayTests(unittest.TestCase):
    def test_replay_loads_only_training_rows_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp) / "round-0"
            dataset.mkdir()
            write_jsonl(dataset / "train.jsonl", [
                {
                    "id": "train-1",
                    "kind": "text",
                    "capability": "math.reasoning",
                    "prompt": "Train question one",
                    "response": "42",
                    "reference": "42",
                    "verifier": {"type": "rule", "name": "exact"},
                    "difficulty": 0.7,
                    "metadata": {"target_difficulty": [0.5, 0.9]},
                },
                {
                    "id": "train-2",
                    "kind": "text",
                    "capability": "code.debugging",
                    "prompt": "Train question two",
                    "response": "fixed",
                    "reference": "fixed",
                    "metadata": {"adaptive_rollout": {"scored": 4, "correct": 2}},
                },
                {
                    "id": "duplicate",
                    "kind": "text",
                    "capability": "math.reasoning",
                    "prompt": "Train question one",
                    "response": "different answer",
                },
            ])
            write_jsonl(dataset / "val.jsonl", [{
                "id": "validation",
                "capability": "math.reasoning",
                "prompt": "validation-only question",
                "response": "not replayed",
            }])
            pipeline = object.__new__(DataPipeline)
            pipeline.runtime = SimpleNamespace(seed=7)
            pipeline.replay_cfg = Config(
                {"enabled": True},
                config_dir=REPO_ROOT,
                repo_root=REPO_ROOT,
            )
            pipeline.replay_fraction = 0.2
            pipeline.replay_rounds = 1
            pipeline.max_replay = 128
            pipeline.max_generated = 1

            replayed = pipeline._load_replay(
                [{"round": 0, "synthesize": {"dataset_path": str(dataset)}}],
                new_count=8,
                round_no=1,
            )

            self.assertEqual(len(replayed), 2)
            self.assertNotIn("validation-only question", {sample.prompt for sample in replayed})
            self.assertEqual(len({sample.prompt for sample in replayed}), 2)
            self.assertTrue(all(sample.metadata["replay"] for sample in replayed))
            self.assertTrue(all(sample.round == 0 for sample in replayed))
            self.assertTrue(all(sample.source.startswith("replay:") for sample in replayed))

    def test_replay_rows_support_messages_and_verl_prompt_views(self) -> None:
        pipeline = object.__new__(DataPipeline)
        sample = pipeline._sample_from_replay_row(
            {
                "prompt": [{"role": "user", "content": "What is 6 times 7?"}],
                "response": "42",
                "ability": "math.reasoning",
                "reward_model": {"style": "rule", "ground_truth": "42"},
                "extra_info": {
                    "id": "verl-sample",
                    "kind": "text",
                    "verifier": {"type": "rule", "name": "numeric"},
                },
            },
            source_round=2,
        )
        self.assertIsNotNone(sample)
        self.assertEqual(sample.id, "verl-sample")
        self.assertEqual(sample.prompt, "What is 6 times 7?")
        self.assertEqual(sample.response, "42")
        self.assertEqual(sample.verifier["name"], "numeric")
        self.assertEqual(sample.metadata["replayed_from_round"], 2)


class ParquetBuilderTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow") or importlib.util.find_spec("datasets"),
        "parquet writer is not installed",
    )
    def test_prompt_only_rows_are_written_to_parquet(self) -> None:
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp:
            runtime = FakeRuntime(temp)
            builder = VerlParquetDatasetBuilder(
                runtime,
                Config(
                    {"val_ratio": 0, "include_metadata": True},
                    config_dir=REPO_ROOT,
                    repo_root=REPO_ROOT,
                ),
            )
            sample = DataSample(
                id="s1",
                kind="text",
                capability="math.reasoning",
                prompt="What is 6 times 7?",
                reference="42",
                verifier={"type": "rule", "name": "numeric"},
                source="dataset:math_word_problems",
                group="math.reasoning",
                metadata={"prompt_only": True},
            )
            context = SimpleNamespace(
                round=0,
                work_dir=Path(temp),
                defects=None,
                data_strategy=None,
                sampler_stats={},
                mixing_info={},
                synthesis=None,
            )

            artifact = builder.build([sample], context, DataPlan(round=0), {"math.reasoning": 1.0})

            train_path = Path(artifact.train_path)
            self.assertEqual(train_path.suffix, ".parquet")
            self.assertTrue(train_path.exists())
            self.assertTrue((Path(artifact.path) / "train.jsonl").exists())
            rows = pq.read_table(train_path).to_pylist()
            self.assertEqual(rows[0]["data_source"], "dataset:math_word_problems")
            self.assertTrue(rows[0]["extra_info"]["prompt_only"])
            manifest = json.loads(
                (Path(artifact.path) / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["format"], "parquet")


if __name__ == "__main__":
    unittest.main()
