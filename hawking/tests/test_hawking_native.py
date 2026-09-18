"""Native connector contract tests without requiring a GPU model fixture."""
from __future__ import annotations

import json
import os
import signal
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from hawking.engine import Engine, EngineError
from hawking.events import EventBus
from hawking.backends import NativeRuntimeBackend
from hawking.hawking_native import (
    HawkingNativeConfig,
    HawkingNativeConfigError,
    HawkingNativeConnector,
    HawkingNativeProtocolError,
    HawkingNativeUnavailable,
    _RenderedPrompt,
    _TokenizerRenderer,
    config_for_model_path,
    suppress_native_runtime_spawning,
)
from hawking.models import discover_models
from hawking.runtime_iface import classify_backend, model_semantics_for
from hawking.workspace import Workspace


class _Renderer:
    def render(self, messages, *, thinking_requested):
        del messages
        return _RenderedPrompt(
            text="<|im_start|>user\nfixture<|im_end|>\n<|im_start|>assistant\n",
            prompt_tokens=5,
            thinking_requested=thinking_requested,
            thinking_qualified=True,
            token_count_source="fixture",
        )


ONE_SHOT = r'''#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

def value(flag):
    i = sys.argv.index(flag)
    return sys.argv[i + 1]

out = Path(value("--out"))
if os.environ.get("FAKE_NATIVE_BAD") == "1":
    out.write_text(json.dumps({"new_token_ids": [1]}), encoding="utf-8")
else:
    out.write_text(json.dumps({
        "generated_text": "fixture answer",
        "new_token_ids": [11, 12],
        "prompt_len": 5,
        "wall_ns": 1000000,
        "decode_wall_ns": 500000,
        "decode_steps": 2,
        "fallbacks": 3,
        "dense_w_materialized": 0,
    }), encoding="utf-8")
'''


RESIDENT = r'''#!/usr/bin/env python3
import json
import os
import sys

print(json.dumps({
    "status": "ready",
    "protocol": "hawking.qwen38.resident.v1",
    "resident_identity": "fixture-resident",
    "model_open_count": 1,
    "weight_upload_count": 1,
}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({
        "id": request["id"],
        "status": "ok",
        "text": "resident answer",
        "new_token_ids": [21, 22],
        "prompt_len": 5,
        "wall_ns": 1000000,
        "decode_wall_ns": 500000,
        "decode_steps": 2,
        "fallbacks": 0,
        "dense_w_materialized": 0,
    }), flush=True)
'''


def _executable(path: Path, source: str) -> str:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def _config(root: Path, *, mode: str, binary: str, resident_binary: str | None = None):
    artifact = root / "artifact"
    artifact.mkdir()
    (artifact / "MIX_REPORT.json").write_text(
        json.dumps({"mix_id": "fixture", "catalog": "fixture", "n_tensors": 1}),
        encoding="utf-8",
    )
    (artifact / "catalog.hq38m20").write_bytes(b"fixture")
    tokenizer = artifact / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    return HawkingNativeConfig(
        artifact_root=str(artifact),
        tokenizer=str(tokenizer),
        binary=binary,
        resident_binary=resident_binary,
        mode=mode,
        max_seq_len=64,
        generation={"max_new_tokens": 4, "enable_thinking": False},
        fusion_env={},
        require_fusion_env=False,
        resident_identity="fixture-resident",
        family="Fixture",
        architecture="fixture-lm",
        param_class="1B",
        quantisation="q4",
    )


class TestNativeConnector(TestCase):
    def test_one_shot_maps_native_receipt_to_common_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = _executable(root / "one-shot", ONE_SHOT)
            config = _config(root, mode="one_shot", binary=binary)
            connector = HawkingNativeConnector(config, renderer=_Renderer())
            raw = connector.complete_payload(
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 8,
                },
                timeout=5,
            )
            self.assertEqual(raw["choices"][0]["message"]["content"], "fixture answer")
            self.assertEqual(raw["usage"]["completion_tokens"], 2)
            self.assertEqual(raw["hawking"]["fallbacks"], 3)
            self.assertEqual(raw["hawking"]["timing_unit"], "ns")
            self.assertGreaterEqual(raw["hawking"]["wall_ns"], 0)
            self.assertEqual(raw["hawking"]["generation_wall_ns"], 1_000_000)
            # An explicit request is allowed to exceed the profile's default
            # max_new_tokens; only the native context window may clamp it.
            self.assertFalse(raw["hawking"]["generation_clamped"])
            self.assertEqual(raw["hawking"]["mode"], "one_shot")

    def test_malformed_one_shot_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = _executable(root / "one-shot", ONE_SHOT)
            config = _config(root, mode="one_shot", binary=binary)
            connector = HawkingNativeConnector(config, renderer=_Renderer())
            old = os.environ.get("FAKE_NATIVE_BAD")
            os.environ["FAKE_NATIVE_BAD"] = "1"
            try:
                with self.assertRaises(HawkingNativeProtocolError):
                    connector.complete_payload(
                        {"messages": [{"role": "user", "content": "hello"}]},
                        timeout=5,
                    )
            finally:
                if old is None:
                    os.environ.pop("FAKE_NATIVE_BAD", None)
                else:
                    os.environ["FAKE_NATIVE_BAD"] = old

    def test_constrained_call_refuses_one_shot_before_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = _executable(root / "one-shot", ONE_SHOT)
            connector = HawkingNativeConnector(
                _config(root, mode="one_shot", binary=binary), renderer=_Renderer())
            with patch("hawking.hawking_native.subprocess.run") as run:
                with suppress_native_runtime_spawning():
                    with self.assertRaises(HawkingNativeUnavailable):
                        connector.complete_payload(
                            {"messages": [{"role": "user", "content": "hello"}]},
                            timeout=5,
                        )
            run.assert_not_called()

    def test_constrained_call_refuses_resident_restart_before_stop_or_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = _executable(root / "one-shot", ONE_SHOT)
            resident_binary = _executable(root / "resident", RESIDENT)
            connector = HawkingNativeConnector(
                _config(root, mode="resident", binary=binary,
                        resident_binary=resident_binary),
                renderer=_Renderer(),
            )

            class _BrokenResident:
                stopped = False
                started = False

                def request(self, _request, _timeout):
                    raise HawkingNativeProtocolError("broken pipe")

                def stop(self):
                    self.stopped = True

                def start(self, timeout):
                    del timeout
                    self.started = True

                def suppress_boundary_trace(self):
                    return nullcontext()

            broken = _BrokenResident()
            connector.resident = broken  # type: ignore[assignment]
            trace = root / "constrained-boundary-trace.jsonl"
            with patch.dict(os.environ, {"HAWKING_BOUNDARY_TRACE": str(trace)}):
                with suppress_native_runtime_spawning():
                    with self.assertRaises(HawkingNativeUnavailable):
                        connector.complete_payload(
                            {"messages": [{"role": "user", "content": "hello"}]},
                            timeout=5,
                        )
            self.assertFalse(broken.stopped)
            self.assertFalse(broken.started)
            self.assertFalse(trace.exists())

    def test_constrained_native_context_never_falls_back_to_ps(self):
        from hawking import resources

        with patch.object(resources, "_start_token_libproc", return_value=None), \
                patch.object(resources, "_start_token_procfs", return_value=None), \
                patch.object(resources.subprocess, "run") as run:
            with suppress_native_runtime_spawning():
                self.assertIsNone(resources.process_start_token(424242))
        run.assert_not_called()

    def test_resident_reader_thread_honors_request_trace_suppression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = _executable(root / "one-shot", ONE_SHOT)
            resident_binary = _executable(root / "resident", RESIDENT)
            connector = HawkingNativeConnector(
                _config(root, mode="resident", binary=binary,
                        resident_binary=resident_binary),
                renderer=_Renderer(),
            )
            connector.start(timeout=5)
            assert connector.resident is not None
            pid = connector.pid
            trace = root / "reader-boundary-trace.jsonl"
            try:
                with patch.dict(os.environ, {"HAWKING_BOUNDARY_TRACE": str(trace)}):
                    with connector.resident.suppress_boundary_trace():
                        os.kill(int(pid), signal.SIGKILL)
                        for _ in range(50):
                            if connector.process is not None and connector.process.poll() is not None:
                                break
                            time.sleep(0.01)
                        reader = connector.resident._reader
                        if reader is not None:
                            reader.join(timeout=1)
                self.assertFalse(trace.exists())
            finally:
                connector.stop()

    def test_resident_reuses_one_open_and_restarts_after_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = _executable(root / "one-shot", ONE_SHOT)
            resident = _executable(root / "resident", RESIDENT)
            config = _config(
                root,
                mode="resident",
                binary=binary,
                resident_binary=resident,
            )
            connector = HawkingNativeConnector(config, renderer=_Renderer())
            try:
                connector.start(timeout=5)
                first = connector.complete_payload(
                    {"messages": [{"role": "user", "content": "one"}]},
                    timeout=5,
                )
                first_pid = connector.pid
                second = connector.complete_payload(
                    {"messages": [{"role": "user", "content": "two"}]},
                    timeout=5,
                )
                self.assertEqual(first["choices"][0]["message"]["content"], "resident answer")
                self.assertEqual(second["choices"][0]["message"]["content"], "resident answer")
                self.assertEqual(first["hawking"]["resident_health"]["model_open_count"], 1)
                self.assertEqual(second["hawking"]["resident_health"]["weight_upload_count"], 1)
                self.assertEqual(first_pid, connector.pid)

                os.kill(int(first_pid), signal.SIGKILL)
                for _ in range(20):
                    if connector.resident is not None and not connector.resident._alive():
                        break
                    time.sleep(0.02)
                recovered = connector.complete_payload(
                    {"messages": [{"role": "user", "content": "recover"}]},
                    timeout=5,
                )
                self.assertEqual(recovered["choices"][0]["message"]["content"], "resident answer")
                self.assertEqual(recovered["hawking"]["retry_count"], 1)
                self.assertEqual(connector.restart_count, 1)
                self.assertNotEqual(first_pid, connector.pid)
            finally:
                connector.stop()

    def test_invalid_relative_profile_is_rejected(self):
        with self.assertRaises(HawkingNativeConfigError):
            HawkingNativeConfig(
                artifact_root="relative-artifact",
                tokenizer="/tmp/tokenizer.json",
                binary="/tmp/native",
            )


class TestNativeGeneralization(TestCase):
    def test_plain_cognition_surface_preserves_text_and_rejects_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(
                workspace=Workspace(tmp),
                model_client=SimpleNamespace(
                    complete=lambda **_kwargs: "HAWKING_OK",
                ),
            )
            self.assertEqual(engine.complete_text("Return exactly: HAWKING_OK"), "HAWKING_OK")

            empty = Engine(
                workspace=Workspace(tmp),
                model_client=SimpleNamespace(
                    complete=lambda **_kwargs: "",
                ),
            )
            with self.assertRaisesRegex(EngineError, "provider returned empty text"):
                empty.complete_text("Return exactly: HAWKING_OK")

    def test_profile_declares_the_current_resident_prompt_fallback(self):
        profile = Path(__file__).resolve().parents[2] / "hawking" / "hawking-native.sealed-3.14.json"
        config = config_for_model_path(str(profile))
        rendered = _TokenizerRenderer(config).render(
            [{"role": "user", "content": "Return exactly: HAWKING_OK"}],
            thinking_requested=False,
        )
        self.assertEqual(
            config.prompt_contract.get("fallback_template"),
            "qwen_closed_think",
        )
        self.assertIn("<|im_start|>user", rendered.text)
        self.assertIn("<|im_start|>assistant", rendered.text)
        self.assertIn("<think>\n\n</think>", rendered.text)
        self.assertNotIn("[role:user]", rendered.text)

    def test_profile_metadata_and_discovery_are_not_qwen_hardcoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = _executable(root / "native", ONE_SHOT)
            config = _config(root, mode="one_shot", binary=binary)
            profile = root / "fixture.hawking.json"
            profile.write_text(
                json.dumps(
                    {
                        "artifact_root": config.artifact_root,
                        "tokenizer": config.tokenizer,
                        "binary": config.binary,
                        "mode": "one_shot",
                        "family": "Acme",
                        "architecture": "acme-transformer",
                        "param_class": "3B",
                        "quantisation": "int4",
                    }
                ),
                encoding="utf-8",
            )
            found = discover_models([tmp])
            selected = next(item for item in found if item.path == str(profile.resolve()))
            self.assertEqual(selected.family, "Acme")
            self.assertEqual(selected.param_class, "3B")
            self.assertEqual(selected.quantization, "int4")
            semantics = model_semantics_for(str(profile))
            self.assertEqual(semantics.architecture, "acme-transformer")
            self.assertEqual(semantics.quantisation, "int4")
            self.assertEqual(config_for_model_path(str(profile)).family, "Acme")
            self.assertFalse(config_for_model_path(str(profile)).require_fusion_env)

    def test_controller_uses_available_native_profile_as_local_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = _executable(root / "native", ONE_SHOT)
            config = _config(root, mode="one_shot", binary=binary)
            profile = root / "fixture.hawking.json"
            profile.write_text(
                json.dumps({
                    "artifact_root": config.artifact_root,
                    "tokenizer": config.tokenizer,
                    "binary": config.binary,
                    "mode": "one_shot",
                    "require_fusion_env": False,
                    "family": "Acme",
                }),
                encoding="utf-8",
            )
            with patch("hawking.controller._default_native_profile", return_value=str(profile)):
                from hawking.controller import Controller

                controller = Controller(root)
                try:
                    self.assertEqual(controller.model, str(profile.resolve()))
                    self.assertEqual(controller.model_info.provider, "native")
                finally:
                    controller.shutdown()

    def test_engine_accepts_non_http_runtime_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = SimpleNamespace(endpoint=lambda: "hawking-native://fixture/resident")
            runtime = SimpleNamespace(
                active=True,
                index=0,
                pid=123,
                port=None,
                backend=backend,
            )
            pool = SimpleNamespace(runtimes=[runtime])
            engine = Engine(
                workspace=Workspace(tmp),
                event_bus=EventBus(),
                runtime_provider=lambda: pool,
                runtime_count=1,
                model_name="fixture",
            )
            endpoint, provenance = engine._runtime_endpoint()
            self.assertEqual(endpoint, "hawking-native://fixture/resident")
            self.assertEqual(provenance["port"], None)
            self.assertEqual(engine._endpoint_from_provenance(provenance), endpoint)

    def test_one_shot_backend_lifecycle_is_not_ready_before_spawn_or_after_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = _executable(root / "native", ONE_SHOT)
            config = _config(root, mode="one_shot", binary=binary)
            profile = root / "fixture.hawking.json"
            profile.write_text(
                json.dumps(
                    {
                        "artifact_root": config.artifact_root,
                        "tokenizer": config.tokenizer,
                        "binary": config.binary,
                        "mode": "one_shot",
                        "require_fusion_env": False,
                    }
                ),
                encoding="utf-8",
            )
            backend = NativeRuntimeBackend(model_path=str(profile))
            self.assertFalse(backend.ready(0.0))
            backend.spawn()
            self.assertTrue(backend.ready(0.0))
            backend.stop()
            self.assertFalse(backend.ready(0.0))

    def test_native_backend_alias_is_canonicalized(self):
        self.assertEqual(
            classify_backend(None, env={"HAWKING_RUNTIME_BACKEND": "native"}),
            "noetic_native",
        )

    def test_native_profile_capabilities_are_not_qwen_hardcoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = _executable(root / "native", ONE_SHOT)
            config = _config(root, mode="one_shot", binary=binary)
            profile = root / "fixture.hawking.json"
            profile.write_text(
                json.dumps({
                    "artifact_root": config.artifact_root,
                    "tokenizer": config.tokenizer,
                    "binary": config.binary,
                    "mode": "one_shot",
                    "require_fusion_env": False,
                    "capabilities": {
                        "features": {
                            "vision": {"state": "supported"},
                            "tool_calling": True,
                            "grammar": {"state": "unsupported"},
                        }
                    },
                }),
                encoding="utf-8",
            )
            backend = NativeRuntimeBackend(model_path=str(profile))
            self.assertTrue(backend.supports("vision"))
            self.assertTrue(backend.supports("tool_calling"))
            self.assertFalse(backend.supports("grammar"))
            self.assertTrue(backend.supports("chat_template_kwargs"))
