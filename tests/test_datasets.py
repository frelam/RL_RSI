from __future__ import annotations

import importlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from rsi.core.scoring import score_text
from rsi.core.types import DataRequest, TaskSpec
from rsi.modules.datasets import (
    DatasetCatalog,
    DatasetSpec,
    fetch_dataset_rows,
    fill_dataset_quotas,
    merge_ability_priorities,
    preprocess_row,
    register_dataset_loader,
    rows_to_samples,
    select_datasets,
)
from tests.helpers import REPO_ROOT


class DatasetTests(unittest.TestCase):
    def test_optimizer_matches_target_distribution_within_dataset_limit(self) -> None:
        datasets = [
            DatasetSpec("math", "synthetic", abilities={"math": 1.0}, priority=1.0),
            DatasetSpec("code", "synthetic", abilities={"code": 1.0}, priority=0.8),
            DatasetSpec("general", "synthetic", abilities={"math": 0.5, "code": 0.5}, priority=0.7),
        ]
        selection = select_datasets(
            datasets, {"math": 1.0, "code": 0.8}, 100, max_datasets=2
        )
        self.assertEqual({item.name for item in selection.datasets}, {"math", "code"})
        self.assertAlmostEqual(sum(selection.ratios.values()), 1.0)
        self.assertAlmostEqual(selection.ratios["math"], 1.0 / 1.8)
        self.assertAlmostEqual(selection.ratios["code"], 0.8 / 1.8)
        self.assertEqual(selection.uncovered, [])

    def test_optimizer_accounts_for_overlapping_abilities(self) -> None:
        datasets = [
            DatasetSpec("head", "synthetic", abilities={"head": 1.0}, priority=1.0),
            DatasetSpec("mixed", "synthetic", abilities={"head": 0.5, "tail": 0.5}, priority=1.0),
        ]

        selection = select_datasets(datasets, {"head": 0.8, "tail": 0.2}, 100)

        self.assertAlmostEqual(selection.ratios["head"], 0.6)
        self.assertAlmostEqual(selection.ratios["mixed"], 0.4)
        self.assertAlmostEqual(selection.stats["actual_distribution"]["head"], 0.8)
        self.assertAlmostEqual(selection.stats["actual_distribution"]["tail"], 0.2)

    def test_sparse_target_penalizes_aggregate_unspecified_coverage(self) -> None:
        unspecified = {f"extra_{index}": 0.001 for index in range(900)}
        datasets = [
            DatasetSpec("focused", "synthetic", abilities={"target": 1.0}, priority=0.5),
            DatasetSpec(
                "diluted", "synthetic", abilities={"target": 0.1, **unspecified}, priority=1.0
            ),
        ]

        selection = select_datasets(datasets, {"target": 1.0}, 100)

        self.assertEqual([item.name for item in selection.datasets], ["focused"])
        self.assertEqual(selection.stats["actual_distribution"]["target"], 1.0)
        self.assertEqual(selection.stats["actual_distribution"]["__other__"], 0.0)
        self.assertEqual(selection.uncovered, [])

    def test_longtail_shortfall_is_protected_before_head_coverage(self) -> None:
        datasets = [
            DatasetSpec("head", "synthetic", abilities={"head": 1.0}, priority=1.0),
            DatasetSpec("rare", "synthetic", abilities={"rare": 1.0}, priority=0.1),
        ]

        selection = select_datasets(
            datasets, {"head": 0.99, "rare": 0.01}, 100, max_datasets=1
        )

        self.assertEqual([item.name for item in selection.datasets], ["rare"])
        self.assertEqual(selection.stats["longtail_capabilities"], ["rare"])
        self.assertEqual(selection.stats["shortfall"]["rare"], 0.0)
        self.assertEqual(selection.stats["longtail_max_relative_shortfall"], 0.0)

    def test_uncoverable_target_is_reported(self) -> None:
        datasets = [DatasetSpec("math", "synthetic", abilities={"math": 1.0})]

        selection = select_datasets(datasets, {"math": 0.9, "logic": 0.1}, 100)

        self.assertEqual(selection.uncovered, ["logic"])
        self.assertEqual(selection.stats["actual_distribution"]["logic"], 0.0)
        self.assertAlmostEqual(selection.stats["shortfall"]["logic"], 0.1)

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

    def test_custom_dataset_loader_can_be_registered(self) -> None:
        calls = []

        @register_dataset_loader("test_custom_loader")
        def load(spec: DatasetSpec, root: Path) -> list[dict]:
            calls.append((spec.name, root))
            return [{"question": f"{spec.name} question", "answer": "a"}]

        rows = fetch_dataset_rows(DatasetSpec("custom", "test_custom_loader"), REPO_ROOT)

        self.assertEqual(rows, [{"question": "custom question", "answer": "a"}])
        self.assertEqual(calls, [("custom", REPO_ROOT)])

    def test_github_loader_reads_jsonl_and_csv_files_from_a_directory(self) -> None:
        listing = [
            {
                "type": "file",
                "path": "data/a.jsonl",
                "download_url": "https://raw.githubusercontent.com/o/r/main/data/a.jsonl",
            },
            {
                "type": "file",
                "path": "data/b.csv",
                "download_url": "https://raw.githubusercontent.com/o/r/main/data/b.csv",
            },
        ]

        def fake_get(url: str) -> bytes:
            if "api.github.com" in url:
                self.assertIn("/contents/data", url)
                return json.dumps(listing).encode("utf-8")
            if url.endswith("a.jsonl"):
                return b'{"question": "q1", "answer": "a1"}\n'
            if url.endswith("b.csv"):
                return b"question,answer\nq2,a2\n"
            raise AssertionError(url)

        with patch("rsi.modules.datasets._github_get", side_effect=fake_get):
            rows = fetch_dataset_rows(
                DatasetSpec("gh", "github", url="o/r", path="data", ref="main"), REPO_ROOT
            )

        self.assertEqual(
            rows,
            [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}],
        )

    def test_github_blob_url_maps_to_the_raw_file(self) -> None:
        spec = DatasetSpec(
            "gh", "github", url="https://github.com/o/r/blob/dev/data/a.jsonl"
        )
        with patch(
            "rsi.modules.datasets._github_get",
            return_value=b'{"question": "q", "answer": "a"}\n',
        ) as get:
            rows = fetch_dataset_rows(spec, REPO_ROOT)

        get.assert_called_once_with(
            "https://raw.githubusercontent.com/o/r/dev/data/a.jsonl"
        )
        self.assertEqual(rows, [{"question": "q", "answer": "a"}])

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

    def test_shortfall_is_filled_from_the_spare_rows_of_the_other_datasets(self) -> None:
        big = DatasetSpec("big", "synthetic", abilities={"math": 1.0}, priority=1.0,
                          rows=[{"question": f"q{i}", "answer": "a"} for i in range(10)])
        small = DatasetSpec("small", "synthetic", abilities={"math": 1.0}, priority=0.5,
                            rows=[{"question": "s", "answer": "a"}])
        request = DataRequest("r", "math", "dataset", 8)
        # The quotas ask for 6 and 2, but `small` holds a single row.
        quotas = {"big": 6, "small": 2}
        usable = {spec.name: rows_to_samples(spec, spec.rows, request, len(spec.rows))
                  for spec in (big, small)}
        samples, counts = fill_dataset_quotas(
            [(spec, usable[spec.name]) for spec in (big, small)], 8, quotas
        )
        self.assertEqual(len(samples), 8)
        self.assertEqual(counts, {"big": 7, "small": 1})

    def test_a_dataset_with_zero_quota_still_lends_its_rows(self) -> None:
        only = DatasetSpec("only", "synthetic", abilities={"math": 1.0}, priority=0.0,
                           rows=[{"question": f"q{i}", "answer": "a"} for i in range(3)])
        request = DataRequest("r", "math", "dataset", 3)
        samples, counts = fill_dataset_quotas(
            [(only, rows_to_samples(only, only.rows, request, 3))], 3, {}
        )
        self.assertEqual(len(samples), 3)
        self.assertEqual(counts, {"only": 3})


if __name__ == "__main__":
    unittest.main()
