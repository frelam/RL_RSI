from __future__ import annotations

import importlib
import re
import tempfile
import unittest
from pathlib import Path

import yaml

from rsi.core.scoring import score_text
from rsi.core.types import DataRequest, TaskSpec
from rsi.modules.datasets import (
    DatasetCatalog,
    DatasetSpec,
    fetch_dataset_rows,
    merge_ability_priorities,
    preprocess_row,
    rows_to_samples,
    select_datasets,
)
from tests.helpers import REPO_ROOT


class DatasetTests(unittest.TestCase):
    def test_greedy_selection_covers_high_priority_abilities(self) -> None:
        datasets = [
            DatasetSpec("math", "synthetic", abilities={"math": 1.0}, priority=1.0),
            DatasetSpec("code", "synthetic", abilities={"code": 1.0}, priority=0.8),
            DatasetSpec("general", "synthetic", abilities={"math": 0.5, "code": 0.5}, priority=0.7),
        ]
        selection = select_datasets(datasets, {"math": 1.0, "code": 0.8}, 2)
        self.assertEqual({item.name for item in selection.datasets}, {"math", "code"})
        self.assertAlmostEqual(sum(selection.ratios.values()), 1.0)
        self.assertEqual(selection.uncovered, [])

    def test_catalog_accepts_list_coverage_and_task_request_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "datasets.yaml"
            path.write_text(
                "datasets:\n  demo:\n    loader: synthetic\n    abilities: [math.reasoning]\n    rows: [{question: q, answer: a}]\n",
                encoding="utf-8",
            )
            catalog = DatasetCatalog(path)
            self.assertEqual(catalog.datasets[0].abilities, {"math.reasoning": 1.0})
        request = DataRequest("r", "math.reasoning", "dataset", 2)
        task = TaskSpec("task-1", {"math.reasoning": 0.9}, count=2)
        self.assertEqual(request.capability, task.ability_priorities.popitem()[0])

    def test_preprocess_normalizes_chat_and_drops_empty_rows(self) -> None:
        row = preprocess_row({"messages": [{"role": "user", "content": "  question  "}]}, ["normalize_chat", "drop_empty"])
        self.assertEqual(row["prompt"], "question")
        self.assertEqual(row["response"], "")
        self.assertIsNone(preprocess_row({"prompt": "  "}, ["normalize_chat", "drop_empty"]))

    def test_example_datasets_load_and_pass_their_own_verifiers(self) -> None:
        catalog = DatasetCatalog(REPO_ROOT / "configs" / "datasets.yaml")
        self.assertTrue({"math_word_problems", "code_debugging"} <= {dataset.name for dataset in catalog.datasets})
        for dataset in catalog.datasets:
            if dataset.name not in {"math_word_problems", "code_debugging"}:
                continue  # their rewards are exercised in test_verl_reasoning_rl.py / test_verl_agentic_rl.py
            rows = fetch_dataset_rows(dataset, REPO_ROOT)
            capability = next(iter(dataset.abilities))
            request = DataRequest(
                "r", capability, "dataset", len(rows),
                constraints={"verifier": {"type": "rule", "name": dataset.reward["name"]}},
            )
            samples = rows_to_samples(dataset, rows, request, len(rows))
            self.assertEqual(len(samples), len(rows))
            for sample in samples:
                score = score_text(sample.response, sample.reference, sample.verifier["name"], sample.verifier)
                self.assertGreaterEqual(score, 0.99, msg=f"{sample.id}: {sample.response}")

    def test_registered_datasets_declare_url_abilities_reward_and_preprocess(self) -> None:
        raw = yaml.safe_load((REPO_ROOT / "configs" / "datasets.yaml").read_text(encoding="utf-8"))["datasets"]
        for name, entry in raw.items():
            required = {"abilities", "reward", "reward_code", "preprocess"}
            if entry.get("license") not in ("generated", "synthetic"):
                required.add("url")
            self.assertTrue(required <= set(entry), f"{name} lacks {sorted(required - set(entry))}")
            # Every `path::function` a reward_code points at must resolve to a callable in this repo.
            references = re.findall(r"(\S+\.py)::(\w+)", entry["reward_code"])
            self.assertTrue(references, f"{name}: reward_code is not `path::function`: {entry['reward_code']}")
            for path, function in references:
                module = importlib.import_module(path.removeprefix("src/").removesuffix(".py").replace("/", "."))
                self.assertTrue(callable(getattr(module, function, None)), f"{name}: {path} has no {function}")
        catalog = DatasetCatalog(REPO_ROOT / "configs" / "datasets.yaml")
        # A file-backed dataset without local rows is declared only: it is never selected.
        for dataset in catalog.datasets:
            if not dataset.path and dataset.loader in ("jsonl", "parquet"):
                selection = catalog.select({next(iter(dataset.abilities)): 1.0}, len(catalog.datasets))
                self.assertNotIn(dataset.name, {item.name for item in selection.datasets}, dataset.name)
        listed = next(item for item in catalog.datasets if item.name == "math_dapo")
        self.assertEqual(listed.pipeline, "examples/reasoning_rl/scripts/to_parquet_math.py")
        self.assertEqual(listed.preprocess, [])
        self.assertTrue(listed.reward_code.startswith("src/rsi/rewards/reasoning/"))
        inline = next(item for item in catalog.datasets if item.name == "math_word_problems")
        self.assertEqual((inline.pipeline, inline.preprocess), ("hand-written rows", ["normalize_chat", "strip_text", "drop_empty"]))

    def test_preprocess_accepts_a_bare_step_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "datasets.yaml"
            path.write_text("datasets:\n  a:\n    url: https://example.com/a\n    preprocess: [drop_empty]\n", encoding="utf-8")
            spec = DatasetCatalog(path).datasets[0]
            self.assertEqual((spec.pipeline, spec.preprocess, spec.reward_code), ("", ["drop_empty"], ""))
            path.write_text("datasets:\n  a:\n    preprocess: {steps: drop_empty}\n", encoding="utf-8")
            with self.assertRaises(Exception):
                DatasetCatalog(path)

    def test_priority_merge_uses_available_sources_and_configured_weights(self) -> None:
        merged = merge_ability_priorities(
            {"math": 0.8, "code": 0.5},
            {"math": 0.2, "tool": 1.0},
            rule_weight=0.6,
            agent_weight=0.4,
        )
        self.assertAlmostEqual(merged["math"], 0.56)
        self.assertEqual(merged["code"], 0.5)
        self.assertEqual(merged["tool"], 1.0)


if __name__ == "__main__":
    unittest.main()
