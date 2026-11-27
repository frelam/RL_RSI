from __future__ import annotations

import dataclasses
import json
import tempfile
import types
import unittest
from pathlib import Path

from rsi.adapters import verl_reasoning_rl
from rsi.core.scoring import score_text
from rsi.core.types import DataRequest
from rsi.hubs.catalog import CapabilityCatalog
from rsi.modules.data_builder import VerlJsonlDatasetBuilder
from rsi.modules.data_pipeline import DataPipeline
from rsi.modules.datasets import DatasetCatalog, fetch_dataset_rows, rows_to_samples

from tests.helpers import REPO_ROOT

DATASETS = REPO_ROOT / "examples" / "datasets"


def _load(name: str, capability: str) -> list:
    catalog = DatasetCatalog(REPO_ROOT / "configs" / "datasets.yaml", REPO_ROOT)
    spec = next(item for item in catalog.datasets if item.name == name)
    rows = fetch_dataset_rows(spec, REPO_ROOT)
    return rows_to_samples(spec, rows, DataRequest(id="r", capability=capability, kind="text", count=10), 10)


def _rows(name: str) -> list:
    return [json.loads(line) for line in (DATASETS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]


CAPABILITY_BY_SOURCE = {
    "math_bigmath": "math.reasoning",
    "math_dapo": "math.reasoning",
    "code_deepcoder": "code.problem_solving",
    "code_codecontests": "code.problem_solving",
    "code_apps": "code.problem_solving",
    "logic_synlogic": "logic.reasoning",
    "logic_enigmata": "logic.reasoning",
    "logic_reasoning_gym": "logic.reasoning",
    "stem_drsci": "stem.reasoning",
    "if_nemotron": "instruction.following",
}


class VerlReasoningRLTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self._saved = verl_reasoning_rl._compute_score
        self.addCleanup(setattr, verl_reasoning_rl, "_compute_score", self._saved)

    def test_verl_rows_map_to_capabilities_and_keep_reward_fields(self) -> None:
        by_source = {source: _load(source, capability)[0] for source, capability in CAPABILITY_BY_SOURCE.items()}
        self.assertEqual(set(by_source), set(verl_reasoning_rl.SOURCES))
        for source, sample in by_source.items():
            self.assertEqual(sample.capability, CAPABILITY_BY_SOURCE[source])
            self.assertEqual((sample.source, sample.metadata["dataset"]), (f"dataset:{source}", source))
            self.assertTrue(sample.metadata["url"].startswith("https://"), source)
            self.assertTrue(sample.prompt and sample.messages[0].role == "user")
            self.assertEqual(sample.verifier["name"], "reasoning_rl")
            self.assertEqual(sample.verifier["data_source"], source)
        self.assertEqual(json.loads(by_source["code_apps"].reference).keys(), {"inputs", "outputs"})
        self.assertEqual(json.loads(by_source["logic_enigmata"].reference)["meta"]["question"], [2, 1, 4, 1])
        self.assertEqual(by_source["logic_reasoning_gym"].verifier["extra_info"]["seed"], 0)
        self.assertEqual(by_source["if_nemotron"].id, "32858")
        self.assertEqual(by_source["math_dapo"].metadata["reward_hint"], {"type": "rule", "name": "reasoning_rl", "weight": 1.0})

    def test_rows_of_capabilities_the_dataset_does_not_cover_are_skipped(self) -> None:
        catalog = DatasetCatalog(REPO_ROOT / "configs" / "datasets.yaml", REPO_ROOT)
        request = DataRequest(id="r", capability="logic.reasoning", kind="text", count=10)
        samples = []
        for name in ("math_dapo", "code_apps", "stem_drsci"):  # no logic row in these files
            spec = dataclasses.replace(next(item for item in catalog.datasets if item.name == name), abilities={})
            samples += rows_to_samples(spec, fetch_dataset_rows(spec, REPO_ROOT), request, 10)
        self.assertEqual(samples, [])

    def test_source_keeps_only_the_rows_of_one_dataset(self) -> None:
        catalog = DatasetCatalog(REPO_ROOT / "configs" / "datasets.yaml", REPO_ROOT)
        dapo = next(item for item in catalog.datasets if item.name == "math_dapo")
        request = DataRequest(id="r", capability="math.reasoning", kind="text", count=10)
        self.assertEqual([s.subcapability for s in rows_to_samples(dapo, fetch_dataset_rows(dapo, REPO_ROOT), request, 10)], ["math_dapo"])
        other = dataclasses.replace(dapo, source="math_bigmath")  # the file holds no math_bigmath row
        self.assertEqual(rows_to_samples(other, fetch_dataset_rows(other, REPO_ROOT), request, 10), [])

    def test_unknown_ability_is_rejected(self) -> None:
        with self.assertRaises(Exception):
            verl_reasoning_rl.verl_row_to_row({"data_source": "x_y", "ability": "vision", "reward_model": {}})

    def test_parquet_rows_are_adapted_like_jsonl_rows(self) -> None:
        try:
            import pyarrow.parquet  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is optional")
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [row for source in CAPABILITY_BY_SOURCE for row in _rows(source)]
        path = Path(self._dir.name) / "train.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        spec = types.SimpleNamespace(name="pool", loader="parquet", path=str(path), preprocess=[])
        adapted = fetch_dataset_rows(spec, REPO_ROOT)
        self.assertEqual([row["subcapability"] for row in adapted], [row["data_source"] for row in rows])
        self.assertEqual(adapted[2]["reference"], rows[2]["reward_model"]["ground_truth"])

    def test_scorer_delegates_to_verl_compute_score(self) -> None:
        calls = []

        def fake(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
            calls.append((data_source, solution_str, ground_truth, extra_info))
            return {"score": 0.5}

        verl_reasoning_rl._compute_score = fake
        metadata = {"data_source": "logic_reasoning_gym", "extra_info": {"seed": 7}}
        self.assertEqual(score_text("<think>x</think>\n\n6*7", '{"answer": "42"}', "reasoning_rl", metadata), 0.5)
        self.assertEqual(calls, [("logic_reasoning_gym", "<think>x</think>\n\n6*7", '{"answer": "42"}', {"seed": 7})])
        self.assertEqual(score_text("x", None, "reasoning_rl", metadata), 0.0)
        with self.assertRaises(Exception):
            score_text("x", "y", "reasoning_rl", {})

    def test_real_reward(self) -> None:
        # Logic rows score without the optional packages (verl, langdetect, reasoning_gym).
        for source in ("logic_synlogic", "logic_enigmata", "logic_reasoning_gym"):
            sample = _load(source, "logic.reasoning")[0]
            answer = json.loads(sample.reference)["answer"]

            def score(prediction: str) -> float:
                return score_text(prediction, sample.reference, "reasoning_rl", sample.verifier)

            with self.subTest(source):
                self.assertEqual(score("<think>x</think>\nFinal Answer: " + answer), 1.0)
                self.assertEqual(score("<think>x</think>\\boxed{" + answer + "}"), 1.0)
                self.assertEqual(score("<think>never closed \\boxed{" + answer + "}"), 0.0)  # format gate
                self.assertEqual(score("<think>x</think>\\boxed{definitely_wrong_424242}"), 0.0)
                self.assertEqual(score(""), 0.0)
                # A death loop is penalised even when its last answer is right (floor -1).
                self.assertEqual(score("<think>" + "the answer is " * 300 + "</think>\n" + answer), -1.0)

    def test_verl_jsonl_export_keeps_the_reward_dispatch(self) -> None:
        sample = _load("logic_enigmata", "logic.reasoning")[0]
        builder = types.SimpleNamespace(schema="verl_jsonl", include_metadata=True)
        row = VerlJsonlDatasetBuilder._row(builder, sample)
        self.assertEqual(row["data_source"], sample.subcapability)
        self.assertEqual(row["ability"], "logic")
        self.assertEqual(row["reward_model"]["ground_truth"], sample.reference)
        self.assertEqual(row["extra_info"]["task_id"], sample.id)
        self.assertEqual(row["capability"], sample.capability)
        replayed = DataPipeline._sample_from_replay_row(None, json.loads(json.dumps(row)), 1)
        self.assertEqual((replayed.capability, replayed.subcapability, replayed.reference),
                         (sample.capability, sample.subcapability, sample.reference))
        self.assertEqual(replayed.verifier, sample.verifier)

    def test_ability_catalog_covers_every_verl_data_source(self) -> None:
        capabilities = CapabilityCatalog(REPO_ROOT / "configs" / "capabilities.yaml")
        self.assertEqual(
            {verl_reasoning_rl.ABILITY_TO_CAPABILITY[ability] for ability in ("math", "code", "logic", "stem", "if")},
            {"math.reasoning", "code.problem_solving", "logic.reasoning", "stem.reasoning",
             "instruction.following"},
        )
        for capability in ("code.problem_solving", "logic.reasoning", "stem.reasoning"):
            self.assertEqual(capabilities.verifier(capability)["name"], "reasoning_rl")
        # Each verl data_source is one registered dataset: url, the abilities of its
        # rows, the reward code and the script that builds its rows.
        datasets = {item.name: item for item in DatasetCatalog(REPO_ROOT / "configs" / "datasets.yaml").datasets}
        self.assertEqual(set(datasets) & set(verl_reasoning_rl.SOURCES), set(verl_reasoning_rl.SOURCES))
        for source, spec in datasets.items():
            if source not in verl_reasoning_rl.SOURCES:
                continue
            self.assertEqual(set(spec.abilities), {CAPABILITY_BY_SOURCE[source]}, source)
            self.assertEqual(spec.reward["name"], "reasoning_rl")
            self.assertEqual(spec.reward, {"type": "rule", "name": "reasoning_rl", "weight": 1.0})
            self.assertTrue(spec.reward_code.startswith("src/rsi/rewards/reasoning/"), source)
            self.assertTrue(spec.pipeline.startswith("examples/reasoning_rl/scripts/to_parquet_"), source)
            self.assertEqual((spec.source, spec.path, spec.loader), (source, f"examples/datasets/{source}.jsonl", "jsonl"))


if __name__ == "__main__":
    unittest.main()
