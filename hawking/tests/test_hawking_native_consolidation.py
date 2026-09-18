"""Focused regression checks for the Hawking-native consolidation boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hawking import AgentOS, Hawking
from hawking.executors import WorkUnitExecutor, select_backend_name
from hawking.models import discover_models
from hawking.remote_cognition import (
    HawkingNativeProvider,
    ProviderSelector,
    RemoteCostError,
    RemoteCostPolicy,
    provider_for_selector,
)
from hawking.resources import normalize_backend, normalize_resource_class
from hawking.runtime_iface import (
    classify_backend,
    make_backend_for_model,
)
from hawking.workunit import WorkUnit


class _FakeNativeBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def identity(self):
        return {
            "provider": "hawking",
            "model_id": "test-native",
            "backend": "hawking-native",
            "runtime": "hawking-native",
            "pid": None,
            "port": None,
        }

    def complete(self, payload, timeout=None):
        self.calls.append({"payload": payload, "timeout": timeout})
        content = json.dumps(
            {
                "observations": ["native path exercised"],
                "operations": [],
                "tests": ["native-boundary"],
                "questions": [],
                "status": "observed",
                "content": "native result",
            }
        )
        return SimpleNamespace(
            text=content,
            raw={"usage": {"total_tokens": 7}},
            total_tokens=7,
        )


def test_provider_selectors_are_hawking_owned() -> None:
    assert ProviderSelector.parse("hawking:native-profile").value == "hawking:native-profile"
    assert ProviderSelector.parse("openrouter:vendor/model-id").value == "openrouter:vendor/model-id"
    with pytest.raises(ValueError):
        ProviderSelector.parse("grok:legacy-model")
    with pytest.raises(ValueError):
        ProviderSelector.parse("mlx:legacy-model")


def test_native_provider_has_no_remote_process_identity(tmp_path: Path) -> None:
    backend = _FakeNativeBackend()
    provider = provider_for_selector(
        "hawking:test-native",
        workspace=tmp_path,
        backend=backend,
    )
    identity = provider.identity()
    assert identity["provider"] == "hawking"
    assert identity["backend"] == "hawking-native"
    assert identity["runtime"] == "hawking-native"
    assert identity["pid"] is None
    assert identity["port"] is None

    unit = WorkUnit(
        id="native-1",
        role="research",
        description="Inspect the native boundary",
        preferred_backend="hawking:test-native",
    )
    result = provider.execute_workunit(unit, {"objective": unit.description})
    assert result["provider"] == "hawking"
    assert result["backend"] == "hawking-native"
    assert result["validation"]["ok"] is False
    assert result["hawking_cognition"]["provider"] == "hawking"
    assert result["hawking_cognition"]["pid"] is None
    assert result["hawking_cognition"]["port"] is None
    assert backend.calls


def test_workunit_executor_keeps_central_validation_and_receipt(tmp_path: Path) -> None:
    backend = _FakeNativeBackend()
    provider = HawkingNativeProvider("test-native", workspace=tmp_path, backend=backend)
    unit = WorkUnit(
        id="native-2",
        role="research",
        description="Produce a bounded observation",
        preferred_backend="hawking:test-native",
    )
    result = WorkUnitExecutor(
        tmp_path,
        providers={"hawking:test-native": provider},
    ).execute(unit, {"provider_instance": provider, "objective": unit.description})
    assert result["provider"] == "hawking"
    assert result["provider_receipt"]["provider"] == "hawking"
    assert result["provider_receipt"]["request"]["unit_id"] == unit.id
    assert result["validation"]["ok"] is False


def test_mutation_cognition_requires_existing_mission_lock(tmp_path: Path) -> None:
    provider = HawkingNativeProvider(
        "test-native",
        workspace=tmp_path,
        backend=_FakeNativeBackend(),
    )
    unit = WorkUnit(
        id="native-mutation",
        role="builder",
        description="Propose a change",
        resource_class="MUTATION",
    )
    result = provider.execute_workunit(unit, {})
    assert result["status"] == "failed"
    assert result["error"]["code"] == "MUTATION_LOCK_REQUIRED"


def test_legacy_backend_and_resource_labels_normalize_to_hawking() -> None:
    unit = WorkUnit(
        id="legacy-label",
        role="research",
        description="legacy data must not select a vendor runtime",
        preferred_backend="grok",
        resource_class="GROK",
    )
    assert select_backend_name(unit) == "hawking"
    assert normalize_backend("qwen") == "hawking"
    assert normalize_backend("grok_bot") == "hawking"
    assert normalize_resource_class("GROK-BOT") == "HAWKING"


def test_legacy_model_artifacts_are_refused_and_not_discovered(tmp_path: Path) -> None:
    mlx = tmp_path / "legacy-mlx"
    mlx.mkdir()
    (mlx / "config.json").write_text("{}")
    (mlx / "weights.safetensors").write_bytes(b"legacy")
    gguf = tmp_path / "legacy.gguf"
    gguf.write_bytes(b"GGUF" + b"legacy")

    assert classify_backend(mlx) == "mlx"
    assert classify_backend(gguf) == "llamacpp"
    with pytest.raises(RuntimeError, match="retired|unsupported"):
        make_backend_for_model(str(mlx))
    with pytest.raises(RuntimeError, match="retired|unsupported"):
        make_backend_for_model(str(gguf))
    assert discover_models([str(tmp_path)]) == []


def test_hawking_is_the_public_facade_and_agentos_is_only_an_alias() -> None:
    assert AgentOS is Hawking


def test_remote_cost_policy_is_fail_closed() -> None:
    policy = RemoteCostPolicy(max_cost_usd=0.0)
    with pytest.raises(RemoteCostError, match="explicit USD limit"):
        policy.authorize(
            provider="openrouter",
            model_id="vendor/model",
            request_id="req-1",
        )
