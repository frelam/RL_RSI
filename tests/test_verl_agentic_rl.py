from __future__ import annotations

import dataclasses
import json
import tempfile
import types
import unittest
from pathlib import Path

from rsi.adapters import verl_agentic_rl
from rsi.core.errors import AdapterError
from rsi.core.scoring import score_text
from rsi.core.types import DataRequest
from rsi.hubs.catalog import CapabilityCatalog
from rsi.modules.data_builder import VerlJsonlDatasetBuilder
from rsi.modules.data_pipeline import DataPipeline
from rsi.modules.datasets import DatasetCatalog, fetch_dataset_rows, rows_to_samples

from tests.helpers import REPO_ROOT

DATASETS = REPO_ROOT / "examples" / "datasets"
SINGLE_TURN = verl_agentic_rl.SINGLE_TURN_TOOL_CALL
CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
TASK_TYPES = {"st_fc", "mt_fc", "irrelevance", "abstain", "search_qa", "math_ci", "code_exec"}
#: source -> what the dataset focuses on (configs/capabilities.yaml, `agentic.*`).
FOCUS = {
    "apigen": "agentic.single_turn_tool_call",
    "procedural_st": "agentic.single_turn_tool_call",
    "procedural_mt": "agentic.multi_turn_stateful_tool_use",
    "interleaved_mt": "agentic.interleaved_user_turns",
    "when2call": "agentic.clarify_when_under_specified",
    "hammer": "agentic.tool_irrelevance",
    "parallel_mt": "agentic.parallel_tool_calls",
    "vehicle_travel_mt": "agentic.multi_domain_state_ops",
    "social_ops_mt": "agentic.multi_domain_state_ops",
    "search_r1": "agentic.search_qa",
    "reasoning_math": "agentic.math_code_reasoning",
    "code_exec": "agentic.code_from_tests",
}


def _specs() -> list:
    catalog = DatasetCatalog(REPO_ROOT / "configs" / "datasets.yaml", REPO_ROOT)
    return [item for item in catalog.datasets if item.reward["name"] == "agentic_rl"]


def _load(capability: str = SINGLE_TURN) -> list:
    request = DataRequest(id="r", capability=capability, kind="agentic", count=20)
    return [sample for spec in _specs() if spec.path
            for sample in rows_to_samples(spec, fetch_dataset_rows(spec, REPO_ROOT), request, 20)]


class VerlAgenticRLTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self._saved = verl_agentic_rl._dispatch_reward
        self.addCleanup(setattr, verl_agentic_rl, "_dispatch_reward", self._saved)

    def test_rows_are_filed_under_the_capability_their_source_focuses_on(self) -> None:
        samples = _load()
        by_source = {sample.metadata["agentic"]["source"]: sample for sample in samples}
        self.assertEqual(len(by_source), 13)
        self.assertEqual(set(by_source), set(FOCUS) | {"stateful_abstain"})
        for sample in samples:
            self.assertEqual(sample.kind, "agentic")
            self.assertEqual(sample.verifier["name"], "agentic_rl")
            self.assertEqual(sample.subcapability, sample.metadata["agentic"]["source"])
            self.assertEqual(sample.verifier["task_type"], sample.metadata["agentic"]["task_type"])
            self.assertEqual(sample.verifier["task"]["task_type"], sample.metadata["agentic"]["task_type"])
            self.assertIn(sample.metadata["agentic"]["source"], verl_agentic_rl.SOURCES)
        for source, capability in FOCUS.items():
            self.assertEqual(by_source[source].capability, capability, source)
        self.assertEqual({sample.metadata["agentic"]["task_type"] for sample in samples}, TASK_TYPES)
        self.assertEqual(by_source["apigen"].metadata["agentic"]["tools"][0]["name"], "get_weather")
        self.assertEqual(by_source["search_r1"].reference, "Samuel Beckett")
        self.assertEqual(by_source["search_r1"].messages[0].role, "system")
        self.assertEqual(by_source["hammer"].verifier["task"]["reference"]["gt_calls"], [])
        self.assertEqual(by_source["code_exec"].metadata["agentic"]["env_type"], "code_tests")

    def test_two_task_type_sources_are_split_per_row(self) -> None:
        row = {"extra_info": {"source": "hammer", "agentic_task": {"task_type": "st_fc"}}}
        self.assertEqual(verl_agentic_rl.agentic_row_to_row(row)["capability"], SINGLE_TURN)
        row["extra_info"]["agentic_task"]["task_type"] = "irrelevance"
        self.assertEqual(verl_agentic_rl.agentic_row_to_row(row)["capability"], "agentic.tool_irrelevance")
        row["extra_info"]["source"] = "when2call"
        row["extra_info"]["agentic_task"]["task_type"] = "abstain"
        self.assertEqual(verl_agentic_rl.agentic_row_to_row(row)["capability"], "agentic.clarify_when_under_specified")

    def test_rows_of_capabilities_the_dataset_does_not_cover_are_skipped(self) -> None:
        request = DataRequest(id="r", capability="math.reasoning", kind="agentic", count=20)
        samples = [sample for spec in _specs() if spec.path for sample in rows_to_samples(dataclasses.replace(spec, abilities={}), fetch_dataset_rows(spec, REPO_ROOT), request, 20)]
        self.assertEqual(samples, [])

    def test_invalid_rows_are_rejected(self) -> None:
        with self.assertRaises(AdapterError):
            verl_agentic_rl.agentic_row_to_row({"data_source": "agentic_rl", "reward_model": {}})
        with self.assertRaises(AdapterError):
            verl_agentic_rl.agentic_row_to_row(
                {"extra_info": {"agentic_task": json.dumps({"id": "x", "task_type": "vision"})}}
            )

    def test_mixed_parquet_rows_route_to_their_adapter(self) -> None:
        try:
            import pyarrow.parquet  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is optional")
        import pyarrow as pa
        import pyarrow.parquet as pq

        agentic = [json.loads(line) for name in ("procedural_st", "procedural_mt") for line in (DATASETS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        reasoning_path = DATASETS / "math_dapo.jsonl"
        reasoning = [json.loads(line) for line in reasoning_path.read_text(encoding="utf-8").splitlines()][:1]
        path = Path(self._dir.name) / "train.parquet"
        pq.write_table(pa.Table.from_pylist([reasoning[0], agentic[0], agentic[1]]), path)
        spec = types.SimpleNamespace(name="pool", loader="parquet", path=str(path), preprocess=[])
        adapted = fetch_dataset_rows(spec, REPO_ROOT)
        self.assertEqual([row["capability"] for row in adapted], ["math.reasoning", SINGLE_TURN, "agentic.multi_turn_stateful_tool_use"])
        self.assertEqual([row["subcapability"] for row in adapted], [reasoning[0]["data_source"], "procedural_st", "procedural_mt"])

    def test_tool_call_parsing(self) -> None:
        text = '<think><tool_call>{"name": "draft"}</tool_call></think>' + CALL + (
            '<tool_call>{"name": "b", "arguments": "{\\"k\\": 1}"}</tool_call>'
            '<tool_call>not json</tool_call><tool_call>{"arguments": {}}</tool_call>'
        )
        self.assertEqual(
            verl_agentic_rl.parse_tool_calls(text),
            [{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "b", "arguments": {"k": 1}}],
        )

    def test_scorer_delegates_to_dispatch_reward(self) -> None:
        calls = []

        def fake(task, trajectory):
            calls.append((task, trajectory))
            return {"score": 0.25}

        verl_agentic_rl._dispatch_reward = fake
        sample = next(s for s in _load() if s.metadata["agentic"]["task_type"] == "st_fc")
        prediction = "<think>x</think>\n" + CALL
        self.assertEqual(score_text(prediction, "label", "agentic_rl", sample.verifier), 0.25)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], sample.verifier["task"])
        self.assertEqual(calls[0][1]["turns"][0]["assistant"]["tool_calls"], [{"name": "get_weather", "arguments": {"city": "Paris"}}])
        self.assertIsNone(calls[0][1]["env_state"])

    def test_env_state_task_types_need_a_multi_turn_trajectory(self) -> None:
        verl_agentic_rl._dispatch_reward = lambda task, trajectory: {"score": 0.75 if trajectory["env_state"] else 0.0}
        for task_type in verl_agentic_rl.ENV_STATE_TASK_TYPES:
            sample = next(s for s in _load() if s.metadata["agentic"]["task_type"] == task_type)
            with self.assertRaises(AdapterError):
                score_text("```python\nprint(2)\n```", "", "agentic_rl", sample.verifier)
            metadata = {**sample.verifier, "trajectory": {"turns": [], "env_state": {"done": True}}}
            self.assertEqual(score_text("x", "", "agentic_rl", metadata), 0.75)
        with self.assertRaises(AdapterError):
            score_text("x", "", "agentic_rl", {"task_type": "st_fc"})
        with self.assertRaises(AdapterError):
            score_text("x", "", "agentic_rl", {"task": {"task_type": "code_exec"}, "trajectory": "{"})

    def test_registry_covers_every_frozen_task_type_and_rejects_unknown_ones(self) -> None:
        from rsi.rewards.agentic import TASK_TYPES, dispatch_reward
        from rsi.rewards.agentic import schema

        self.assertEqual(TASK_TYPES, schema.TASK_TYPES)
        # A mistyped task type is a data bug: it must raise, never score 0.
        with self.assertRaises(ValueError):
            dispatch_reward({"task_type": "st_fc_typo"}, verl_agentic_rl.single_turn_trajectory("x"))

    def test_real_rewards(self) -> None:
        by_source = {sample.metadata["agentic"]["source"]: sample for sample in _load()}

        def score(source: str, prediction: str) -> float:
            return score_text(prediction, "", "agentic_rl", by_source[source].verifier)

        self.assertEqual(score("apigen", CALL), 1.0)
        self.assertEqual(score("apigen", CALL.replace("Paris", "Lyon")), 0.0)
        self.assertEqual(score("apigen", "It is sunny."), 0.0)
        self.assertEqual(score("hammer", "Falling leaves whisper."), 1.0)
        self.assertEqual(score("hammer", CALL), 0.0)
        self.assertEqual(score("when2call", "I don't have a tool to remember your name. What else can I help you with?"), 1.0)
        self.assertEqual(score("when2call", CALL), 0.0)
        self.assertEqual(score("search_r1", "<answer>Samuel Beckett</answer>"), 1.0)
        self.assertEqual(score("search_r1", "I do not know."), 0.0)
        self.assertEqual(score("reasoning_math", "48 + 24 = 72. The answer is \\boxed{72}."), 1.0)
        self.assertEqual(score("reasoning_math", "\\boxed{96}"), 0.0)
        for source in ("procedural_mt", "code_exec"):
            with self.assertRaises(AdapterError):
                score(source, "```python\nprint(2)\n```")

    def test_verl_jsonl_export_round_trips_tools_and_task(self) -> None:
        sample = next(s for s in _load() if s.metadata["agentic"]["source"] == "search_r1")
        builder = types.SimpleNamespace(schema="verl_jsonl", include_metadata=True)
        row = VerlJsonlDatasetBuilder._row(builder, sample)
        self.assertEqual(row["data_source"], "agentic_rl")
        self.assertEqual(row["prompt"], [{"role": m.role, "content": m.content} for m in sample.messages])
        self.assertEqual([tool["name"] for tool in row["tools"]], [tool["name"] for tool in sample.metadata["agentic"]["tools"]])
        self.assertEqual(row["extra_info"]["task_type"], "search_qa")
        self.assertEqual(json.loads(row["extra_info"]["agentic_task"])["id"], sample.id)
        replayed = DataPipeline._sample_from_replay_row(None, json.loads(json.dumps(row)), 1)
        self.assertEqual((replayed.capability, replayed.subcapability, replayed.reference),
                         (sample.capability, sample.subcapability, sample.reference))
        self.assertEqual(replayed.verifier, sample.verifier)
        self.assertEqual(replayed.metadata["agentic"]["tools"], sample.metadata["agentic"]["tools"])
        self.assertEqual(verl_agentic_rl.agentic_row_to_row(row)["verifier"], sample.verifier)

    def test_capability_and_dataset_catalog_match_the_adapter(self) -> None:
        capabilities = CapabilityCatalog(REPO_ROOT / "configs" / "capabilities.yaml")
        catalog = {item["name"]: item for item in capabilities.all()}
        agentic = {name: item for name, item in catalog.items() if name.startswith("agentic.")}
        self.assertEqual(set(agentic), set(verl_agentic_rl.CAPABILITY_BY_TASK_TYPE.values()) | set(verl_agentic_rl.FOCUSED_SOURCES.values()))
        mapped = {
            (source, task_type): verl_agentic_rl.capability_for(source, task_type)
            for source in verl_agentic_rl.SOURCES
            for task_type in verl_agentic_rl.SOURCES[source][1].split("|")
        }
        for name, item in agentic.items():
            self.assertEqual(item["verifier"]["name"], "agentic_rl", name)
            self.assertEqual(item["reward"]["name"], "agentic_rl", name)
            self.assertEqual(capabilities.kind(name), "agentic", name)
            # Each dataset is listed under what it focuses on; every source is listed somewhere.
            self.assertTrue(set(item["seeds"]) <= {source for (source, _), cap in mapped.items() if cap == name}, name)
        self.assertEqual(set(mapped.values()), set(agentic))
        self.assertEqual({source for source, _ in mapped}, {s for item in agentic.values() for s in item["seeds"]})
        # Every source is registered with url, the abilities of its rows, the reward code
        # and the pipeline that builds its rows; the ones already downloaded point at them.
        registered = {spec.name: spec for spec in _specs()}
        with_rows = {path.stem for path in DATASETS.glob("*.jsonl")} & set(verl_agentic_rl.SOURCES)
        self.assertTrue(with_rows <= set(registered) <= set(verl_agentic_rl.SOURCES))
        for name, spec in registered.items():
            task_types = verl_agentic_rl.SOURCES[name][1].split("|")
            self.assertEqual(set(spec.abilities), {verl_agentic_rl.capability_for(name, tt) for tt in task_types}, name)
            self.assertEqual(spec.reward, {"type": "rule", "name": "agentic_rl", "weight": 1.0})
            for task_type in task_types:
                self.assertIn(f"src/rsi/rewards/agentic/{task_type}.py::reward", spec.reward_code, name)
            self.assertTrue(spec.pipeline.startswith("agentic_rl_data/") and spec.pipeline.endswith("agentic_rl_data/scripts/export_verl_parquet.py"), name)
            local = f"examples/datasets/{name}.jsonl" if name in with_rows else ""
            self.assertEqual((spec.source, spec.path, spec.loader), (name, local, "jsonl"))
            self.assertTrue(spec.url.startswith("https://") or spec.license == "generated", name)
        self.assertEqual({tt for name in registered for tt in verl_agentic_rl.SOURCES[name][1].split("|")}, TASK_TYPES)


if __name__ == "__main__":
    unittest.main()
