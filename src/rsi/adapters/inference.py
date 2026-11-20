from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from rsi.adapters.openai_http import api_key_from_config, normalize_base_url, openai_chat
from rsi.adapters.llm import _substitute
from rsi.core.errors import AdapterError
from rsi.core.interfaces import InferenceBackend
from rsi.core.registry import register
from rsi.core.types import ChatRequest, ChatResponse, Checkpoint, EndpointInfo


@register("inference", "mock")
class MockInference(InferenceBackend):
    """Deterministic policy endpoint for offline loops."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self._checkpoint: Checkpoint | None = None
        self.answer_template = str(self.cfg.get("answer_template", "MOCK[{{checkpoint}}] answer {{index}}"))

    def ensure(self, checkpoint: Checkpoint) -> EndpointInfo:
        if self._checkpoint is None or self._checkpoint.id != checkpoint.id:
            self.logger.info("mock endpoint now serving %s (%s)", checkpoint.id, checkpoint.model_path)
            self._checkpoint = checkpoint
        return EndpointInfo(
            base_url="mock://inference",
            model=f"mock-{checkpoint.version}",
            checkpoint_id=checkpoint.id,
            managed=False,
        )

    def supports_checkpoint_switch(self) -> bool:
        return True

    def chat(self, request: ChatRequest) -> ChatResponse:
        version = self._checkpoint.version if self._checkpoint else 0
        texts = []
        for index in range(max(1, request.n)):
            prompt = request.messages[-1].content if request.messages else ""
            texts.append(
                _substitute(
                    self.answer_template,
                    {
                        "checkpoint": version,
                        "index": index,
                        "prompt": prompt[:80],
                    },
                )
            )
        return ChatResponse(texts=texts, model=f"mock-{version}", usage={}, raw={"mock": True})


@register("inference", "openai")
class OpenAICompatibleInference(InferenceBackend):
    """Assumes an external OpenAI-compatible server is already serving the model."""

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.base_url = normalize_base_url(str(self.cfg.require("base_url").get("base_url")))
        self.model = str(self.cfg.get("model", "default"))
        self.api_key = api_key_from_config(self.cfg)
        self.timeout_s = self.cfg.get_float("timeout_s", 300.0)
        self.max_retries = self.cfg.get_int("max_retries", 2)
        self.follow_checkpoints = self.cfg.get_bool("follow_checkpoints", False)
        self._checkpoint: Checkpoint | None = None

    def ensure(self, checkpoint: Checkpoint) -> EndpointInfo:
        if self._checkpoint is not None and self._checkpoint.id != checkpoint.id:
            if self.follow_checkpoints:
                self.logger.warning(
                    "Checkpoint changed to %s but no launch block is configured; "
                    "the external server must reload it.",
                    checkpoint.id,
                )
            else:
                self.logger.warning(
                    "Checkpoint changed to %s; this adapter points at a fixed external server.",
                    checkpoint.id,
                )
        self._checkpoint = checkpoint
        return EndpointInfo(
            base_url=self.base_url,
            model=self.model,
            checkpoint_id=checkpoint.id,
            managed=False,
        )

    def supports_checkpoint_switch(self) -> bool:
        return self.follow_checkpoints

    def chat(self, request: ChatRequest) -> ChatResponse:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        started = time.time()
        raw = openai_chat(
            base_url=self.base_url,
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            n=request.n,
            stop=request.stop,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
        )
        choices = raw.get("choices") or []
        texts = [str((choice.get("message") or {}).get("content") or "") for choice in choices]
        if not texts:
            raise AdapterError(f"Inference endpoint returned no choices: {str(raw)[:1000]}")
        return ChatResponse(
            texts=texts,
            model=str(raw.get("model") or self.model),
            usage=dict(raw.get("usage") or {}),
            raw=raw,
            latency_s=time.time() - started,
        )


@register("inference", "vllm")
class VLLMLocalInference(OpenAICompatibleInference):
    """Optionally launches a local vLLM OpenAI server when the checkpoint changes.

    Config::

        infer:
          type: vllm
          model: served-model
          base_url: http://127.0.0.1:8000/v1
          launch:
            enabled: true
            command: [python, -m, vllm.entrypoints.openai.api_server,
                      --model, "{model_path}", --served-model-name, "{model}",
                      --port, "{port}"]
            startup_timeout_s: 600
    """

    def __init__(self, runtime: Any, cfg: Any, **params: Any) -> None:
        super().__init__(runtime, cfg, **params)
        self.launch_cfg = self.cfg.section("launch", required=False)
        self.launch_enabled = self.launch_cfg.get_bool("enabled", False)
        self.port = self.launch_cfg.get_int("port", 8000)
        self.startup_timeout_s = self.launch_cfg.get_float("startup_timeout_s", 600.0)
        self._proc = None
        self._log_handle = None
        self._served_model_path: str | None = None

    def supports_checkpoint_switch(self) -> bool:
        return self.launch_enabled or self.follow_checkpoints

    def ensure(self, checkpoint: Checkpoint) -> EndpointInfo:
        if not self.launch_enabled:
            return super().ensure(checkpoint)
        if self._served_model_path == checkpoint.model_path and self._alive():
            self._checkpoint = checkpoint
            return self._info(checkpoint)
        self._stop_server()
        self._start_server(checkpoint)
        self._checkpoint = checkpoint
        return self._info(checkpoint)

    def _info(self, checkpoint: Checkpoint) -> EndpointInfo:
        return EndpointInfo(
            base_url=self.base_url,
            model=self.model,
            checkpoint_id=checkpoint.id,
            managed=True,
            pid=self._proc.pid if self._proc else None,
        )

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start_server(self, checkpoint: Checkpoint) -> None:
        command = self.launch_cfg.get("command")
        if not command:
            raise AdapterError("vllm launch.enabled=true requires launch.command")
        variables = {
            "model_path": checkpoint.model_path,
            "model": self.model,
            "port": self.port,
            "host": "127.0.0.1",
        }
        if isinstance(command, str):
            argv: str | list[str] = _substitute(command, variables)
        elif isinstance(command, list):
            argv = [str(_substitute(str(part), variables)) for part in command]
        else:
            raise AdapterError("launch.command must be a string or list")
        log_path = self.runtime.run_dir / "vllm" / f"{checkpoint.id}.log"
        self._proc, self._log_handle = self.runtime.executor.start(
            argv,
            cwd=self.runtime.run_dir,
            log_path=log_path,
            label="vllm.launch",
        )
        deadline = time.time() + self.startup_timeout_s
        health = self.base_url.rstrip("/") + "/models"
        while time.time() < deadline:
            if self._proc.poll() is not None:
                tail = ""
                if log_path.exists():
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
                raise AdapterError(f"vLLM exited during startup:\n{tail}")
            try:
                with urllib.request.urlopen(health, timeout=5) as response:
                    if response.status == 200:
                        self._served_model_path = checkpoint.model_path
                        self.logger.info("vLLM ready at %s", self.base_url)
                        return
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(1.0)
        self._stop_server()
        raise AdapterError(f"vLLM did not become healthy within {self.startup_timeout_s}s")

    def _stop_server(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                self._proc.kill()
        self._proc = None
        self._served_model_path = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def close(self) -> None:
        self._stop_server()


# SGLang exposes the same OpenAI API surface; use the external-server adapter.
register("inference", "sglang")(OpenAICompatibleInference)
