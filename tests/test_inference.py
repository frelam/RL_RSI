from __future__ import annotations

import tempfile
import unittest

from rsi.adapters.inference import VLLMLocalInference
from rsi.core.config import Config
from rsi.core.types import Checkpoint
from tests.helpers import REPO_ROOT, FakeRuntime


class VllmLaunchTests(unittest.TestCase):
    def test_launch_spec_substitutes_single_brace_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = FakeRuntime(td)
            inference = VLLMLocalInference(
                runtime,
                Config(
                    {
                        "base_url": "http://127.0.0.1:8000/v1",
                        "model": "policy",
                        "launch": {
                            "enabled": True,
                            "port": 8123,
                            "command": [
                                "python",
                                "-m",
                                "vllm.entrypoints.openai.api_server",
                                "--model",
                                "{model_path}",
                                "--served-model-name",
                                "{model}",
                                "--port",
                                "{port}",
                            ],
                            "env": {"CUDA_VISIBLE_DEVICES": "0"},
                        },
                    },
                    config_dir=REPO_ROOT,
                    repo_root=REPO_ROOT,
                ),
            )
            checkpoint = Checkpoint(id="ckpt_v001", version=1, model_path="/models/base")

            argv, env = inference._launch_spec(checkpoint)

        rendered = " ".join(argv)
        self.assertIn("/models/base", argv)
        self.assertNotIn("{model_path}", rendered)
        self.assertNotIn("{port}", rendered)
        self.assertEqual(argv[argv.index("--port") + 1], "8123")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")

    def test_stop_is_safe_without_a_server(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            inference = VLLMLocalInference(
                FakeRuntime(td),
                Config(
                    {"base_url": "http://127.0.0.1:8000/v1", "launch": {"enabled": False}},
                    config_dir=REPO_ROOT,
                    repo_root=REPO_ROOT,
                ),
            )
            inference.stop()


if __name__ == "__main__":
    unittest.main()
