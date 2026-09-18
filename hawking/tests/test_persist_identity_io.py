"""Canonical persistence helpers keep their former consumer contracts."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest

from hawking.persist import (
    json_compatible_copy,
    read_json_object_or_none,
    sha256_bytes,
    sha256_file_or_none,
)


JSON_OBJECT_ALIASES = (
    ("hawking.agentos.accelerator_regression", "_read_json"),
    ("hawking.autonomy_gate", "_read_json"),
    ("hawking.agentos.flash_component_body", "_read_json"),
    ("hawking.agentos.flash_component_campaign", "_read_json"),
    ("hawking.agentos.flash_executable", "_read_json"),
    ("hawking.agentos.flash_graph_component", "_read_json"),
    ("hawking.agentos.flash_loader_roundtrip", "_read_json"),
    ("hawking.agentos.flash_matrix_component_body", "_read_json"),
    ("hawking.agentos.flash_representation_experiment", "_read_json"),
    ("hawking.agentos.flash_router_graph", "_read_json"),
    ("hawking.agentos.flash_router_representation_ab", "_read_json"),
    ("hawking.agentos.flash_router_selection", "_read_json"),
    ("hawking.agentos.flash_telemetry", "_read_json"),
    ("hawking.agentos.flash_tensor_probe", "_read_json"),
    ("hawking.agentos.flash_transform_parity", "_read_json"),
    ("hawking.modellake_gate", "_read_json"),
    ("hawking.agentos.qwen27_runtime_identity", "_read_json"),
    ("hawking.agentos.qwen38_fusion_audit", "_read_json"),
    ("hawking.recovery_gate", "_read"),
    ("hawking.agentos.preboard", "_read"),
    ("hawking.agentos.modellake_supervisor", "_read"),
    ("hawking.checkpoint", "_read_object"),
)

SHA_FILE_ALIASES = (
    ("hawking.flash_next", "_sha256"),
    ("hawking.agentos.accelerator_regression", "_sha256"),
    ("hawking.agentos.charge", "_sha256"),
    ("hawking.agentos.flash_component_campaign", "_sha256"),
    ("hawking.agentos.flash_executable", "_sha256"),
    ("hawking.agentos.flash_graph_component", "_sha256"),
    ("hawking.agentos.flash_router_graph", "_sha256"),
    ("hawking.agentos.flash_telemetry", "_sha256"),
    ("hawking.modellake_gate", "_sha256"),
    ("hawking.agentos.qwen27_runtime_identity", "_sha256"),
    ("hawking.agentos.qwen38_fusion_audit", "_sha256"),
    ("hawking.agentos.science_maps", "_sha256"),
    ("hawking.agentos.flash_component_body", "_sha256_file"),
    ("hawking.agentos.flash_router_selection", "_sha256_file"),
)

JSON_COPY_ALIASES = (
    ("hawking.physical_graph", "_copy"),
    ("hawking.agentos.accelerator_regression", "_safe"),
    ("hawking.research_gate", "_safe_json"),
    ("hawking.autonomy_gate", "_json_safe"),
    ("hawking.agentos.runtime", "_json_safe"),
    ("hawking.agentos.protected_benchmark_watcher", "_safe"),
    ("hawking.agentos.qwen27_runtime_identity", "_safe"),
    ("hawking.agentos.flash_component_campaign", "_copy"),
    ("hawking.agentos.flash_executable", "_safe"),
)

SHA_BYTES_ALIASES = (
    ("hawking.tool_registry", "_sha256_bytes"),
    ("hawking.agentos.flash_tensor_probe", "_sha256_bytes"),
    ("hawking.agentos.flash_science", "_sha256_bytes"),
    ("hawking.agentos.flash_transform_parity", "_sha256_bytes"),
    ("hawking.agentos.flash_representation_experiment", "_sha256_bytes"),
    ("hawking.agentos.flash_loader_roundtrip", "_sha256_bytes"),
    ("hawking.agentos.flash_matrix_component_body", "_sha256_bytes"),
    ("hawking.agentos.flash_router_representation_ab", "_sha256_bytes"),
    ("hawking.agentos.flash_router_selection", "_sha256_bytes"),
    ("hawking.agentos.flash_vector_component_body", "_sha256_bytes"),
    ("hawking.agentos.flash_component_body", "_sha256_bytes"),
)


def test_read_json_object_or_none_accepts_only_valid_utf8_objects(tmp_path: Path):
    object_path = tmp_path / "object.json"
    object_path.write_text('{"answer": 42}', encoding="utf-8")
    list_path = tmp_path / "list.json"
    list_path.write_text("[1, 2]", encoding="utf-8")
    scalar_path = tmp_path / "scalar.json"
    scalar_path.write_text("42", encoding="utf-8")
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{", encoding="utf-8")
    invalid_utf8_path = tmp_path / "invalid-utf8.json"
    invalid_utf8_path.write_bytes(b"\xff")
    directory = tmp_path / "directory"
    directory.mkdir()

    assert read_json_object_or_none(object_path) == {"answer": 42}
    for path in (list_path, scalar_path, malformed_path, invalid_utf8_path, tmp_path / "missing.json", directory):
        assert read_json_object_or_none(path) is None


def test_read_json_object_or_none_enforces_a_bounded_read_when_requested(tmp_path: Path):
    object_path = tmp_path / "object.json"
    object_path.write_text('{"answer": 42}', encoding="utf-8")

    assert read_json_object_or_none(object_path, maximum=object_path.stat().st_size) == {
        "answer": 42
    }
    assert read_json_object_or_none(object_path, maximum=object_path.stat().st_size - 1) is None
    assert read_json_object_or_none(object_path, maximum=-1) is None


def test_sha256_file_or_none_streams_large_files_and_refuses_unreadable_paths(tmp_path: Path):
    payload = (b"hawking-persist-helper\n" * 65536) + b"tail"
    assert len(payload) > 1024 * 1024
    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(payload)
    empty_path = tmp_path / "empty.bin"
    empty_path.write_bytes(b"")
    directory = tmp_path / "directory"
    directory.mkdir()

    assert sha256_file_or_none(payload_path) == hashlib.sha256(payload).hexdigest()
    assert sha256_file_or_none(empty_path) == hashlib.sha256(b"").hexdigest()
    assert sha256_file_or_none(tmp_path / "missing.bin") is None
    assert sha256_file_or_none(directory) is None


def test_json_compatible_copy_preserves_default_string_and_fallback_behavior():
    class TextValue:
        def __str__(self) -> str:
            return "text-value"

    class UnsupportedKey:
        def __repr__(self) -> str:
            return "<unsupported-key>"

    circular: list[object] = []
    circular.append(circular)

    assert json_compatible_copy({"z": [2, {"b": True, "a": None}], "a": 1}) == {
        "a": 1,
        "z": [2, {"a": None, "b": True}],
    }
    assert json_compatible_copy(TextValue()) == "text-value"
    assert json_compatible_copy({UnsupportedKey(): "value"}) == "{<unsupported-key>: 'value'}"
    assert json_compatible_copy(circular) == "[[...]]"


def test_sha256_bytes_preserves_bytes_like_hashing():
    payload = b"hawking\x00identity\xff"
    bytearray_payload = bytearray(b"bytearray-proof")

    assert sha256_bytes(b"") == hashlib.sha256(b"").hexdigest()
    assert sha256_bytes(payload) == hashlib.sha256(payload).hexdigest()
    assert sha256_bytes(bytearray_payload) == hashlib.sha256(bytearray_payload).hexdigest()


@pytest.mark.parametrize("module_name,local_name", JSON_OBJECT_ALIASES)
def test_json_consumers_reexport_the_canonical_reader(module_name: str, local_name: str):
    module = importlib.import_module(module_name)
    assert getattr(module, local_name) is read_json_object_or_none
    assert f"def {local_name}(" not in Path(module.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("module_name,local_name", SHA_FILE_ALIASES)
def test_digest_consumers_reexport_the_canonical_helper(module_name: str, local_name: str):
    module = importlib.import_module(module_name)
    assert getattr(module, local_name) is sha256_file_or_none
    assert f"def {local_name}(" not in Path(module.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("module_name,local_name", JSON_COPY_ALIASES)
def test_json_copy_consumers_reexport_the_canonical_helper(module_name: str, local_name: str):
    module = importlib.import_module(module_name)
    assert getattr(module, local_name) is json_compatible_copy
    assert f"def {local_name}(" not in Path(module.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("module_name,local_name", SHA_BYTES_ALIASES)
def test_byte_digest_consumers_reexport_the_canonical_helper(module_name: str, local_name: str):
    module = importlib.import_module(module_name)
    assert getattr(module, local_name) is sha256_bytes
    assert f"def {local_name}(" not in Path(module.__file__).read_text(encoding="utf-8")


def test_accelerator_regression_private_safe_importers_keep_the_alias():
    owner = importlib.import_module("hawking.agentos.accelerator_regression")
    for module_name in (
        "hawking.agentos.qwen27_mlp_diagnostic",
        "hawking.agentos.protected_accelerator_benchmark",
    ):
        assert importlib.import_module(module_name)._safe is owner._safe
