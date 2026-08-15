#!/usr/bin/env python3
"""Seal a read-only identity wrapper around Qwen80's static 48-layer schedule.

The permanent schedule authority predates the common receipt seal family.  It
is deliberately not rewritten here: this tool reads one explicit regular file,
checks its full planning structure, and writes one new sealed wrapper that
binds the raw bytes, source/config identities, and the 36 DeltaNet / 12 GQA
schedule facts.  It never opens model payloads, enumerates artifact trees, or
starts any runtime, GPU, lease, server, watcher, or benchmark work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lab.receipts import seal


WRAPPER_SCHEMA = "hawking.ascension.qwen80_48_layer_schedule_sealed_wrapper.v1"
WRAPPER_STATUS = "SEALED_QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY_BOUND_NOT_EXECUTED"
RAW_SCHEMA = "hawking.ascension.qwen80_48_layer_payload_schedule_authority.v1"
RAW_STATUS = "PREPARED_QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY_NOT_EXECUTED"
CANONICAL_RAW_SCHEDULE_FILENAME = "QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY_20260809T033554Z.json"
CANONICAL_RAW_SCHEDULE_BYTES = 88_551_859
CANONICAL_RAW_SCHEDULE_SHA256 = (
    "8302deb6beece8c04773ece19ae27baea67749014552b0b946516146b5e2282e"
)

MODEL_KEY = "qwen80"
MODEL_ID = "Qwen3-Coder-Next-80B"
SOURCE_REPOSITORY = "Qwen/Qwen3-Coder-Next"
SOURCE_REVISION = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb"
SOURCE_CONFIG_SHA256 = "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8"
MANIFEST_DOCUMENT_SHA256 = "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10"
MANIFEST_SEAL_SHA256 = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b"
SOURCE_CONFIG_AUTHORITY_DOCUMENT_SHA256 = (
    "c21f5ac489d58d91ba2eb43c3daf34e2412f39925632b30e147e5de28780596b"
)
SOURCE_CONFIG_AUTHORITY_SEAL_SHA256 = (
    "3d062ca5a8acdcc3c2c018e4ded049fd6647210b8161dfcedd37e99363c8fafd"
)


class ScheduleWrapperError(ValueError):
    """The static schedule cannot safely be wrapped."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_regular(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ScheduleWrapperError(f"{label} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ScheduleWrapperError(f"cannot resolve {label}: {path}: {exc}") from exc
    try:
        info = os.lstat(resolved)
    except OSError as exc:
        raise ScheduleWrapperError(f"cannot stat {label}: {resolved}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ScheduleWrapperError(f"{label} must be a regular non-symlink file")
    return resolved


def _read_raw_schedule(path: Path) -> tuple[Path, bytes, dict[str, Any]]:
    resolved = _canonical_regular(path, "--raw-schedule")
    try:
        raw = resolved.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ScheduleWrapperError(f"cannot read JSON raw schedule: {exc}") from exc
    if not isinstance(document, dict):
        raise ScheduleWrapperError("raw schedule root must be a JSON object")
    return resolved, raw, document


def _expect(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise ScheduleWrapperError(f"{label} mismatch: expected {expected!r}, observed {observed!r}")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScheduleWrapperError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ScheduleWrapperError(f"{label} must be an array")
    return value


def _validate_source_authority(document: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(document.get("source_authority"), "raw schedule source_authority")
    expected = {
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_config_sha256": SOURCE_CONFIG_SHA256,
        "descriptor_inventory_document_sha256": MANIFEST_DOCUMENT_SHA256,
        "descriptor_inventory_seal_sha256": MANIFEST_SEAL_SHA256,
        "source_config_authority_document_sha256": SOURCE_CONFIG_AUTHORITY_DOCUMENT_SHA256,
        "source_config_authority_seal_sha256": SOURCE_CONFIG_AUTHORITY_SEAL_SHA256,
    }
    for key, value in expected.items():
        _expect(source.get(key), value, f"raw schedule source_authority.{key}")
    _expect(
        source.get("descriptor_inventory_schema"),
        "hawking.ascension.qwen80_complete_binary_gravity.v1",
        "raw schedule descriptor inventory schema",
    )
    _expect(
        source.get("source_config_authority_schema"),
        "hawking.ascension.source_admission_candidate.v1",
        "raw schedule source config authority schema",
    )
    return dict(expected)


def _validate_schedule(document: Mapping[str, Any]) -> dict[str, Any]:
    _expect(document.get("schema"), RAW_SCHEMA, "raw schedule schema")
    _expect(document.get("status"), RAW_STATUS, "raw schedule status")
    if document.get("seal_sha256") not in (None, ""):
        raise ScheduleWrapperError("raw static schedule must remain unsealed; do not replace it")
    _expect(document.get("all_48_layers_scheduled"), True, "raw schedule all_48_layers_scheduled")
    _expect(
        document.get("all_descriptors_source_artifact_bound"),
        True,
        "raw schedule descriptor binding",
    )
    _expect(document.get("resolved_tensor_binding_count"), 74391, "raw schedule tensor binding count")

    layers = _list(document.get("layers"), "raw schedule layers")
    if len(layers) != 48:
        raise ScheduleWrapperError("raw schedule must contain exactly 48 layers")
    delta_layers: list[Mapping[str, Any]] = []
    gqa_layers: list[Mapping[str, Any]] = []
    for ordinal, item in enumerate(layers):
        layer = _mapping(item, f"raw schedule layers[{ordinal}]")
        _expect(layer.get("layer"), ordinal, f"raw schedule layers[{ordinal}].layer")
        mixer = layer.get("mixer")
        if mixer == "delta_net":
            delta_layers.append(layer)
            state = _mapping(layer.get("state_slot"), f"raw schedule layers[{ordinal}].state_slot")
            _expect(state.get("layer"), ordinal, f"raw schedule delta layer {ordinal} state layer")
            _expect(state.get("domain"), "delta_net_conv_and_recurrent", f"raw schedule delta layer {ordinal} domain")
            _expect(state.get("state_materialized_by_this_plan"), False, f"raw schedule delta layer {ordinal} state materialization")
        elif mixer == "gqa":
            gqa_layers.append(layer)
            state = _mapping(layer.get("state_slot"), f"raw schedule layers[{ordinal}].state_slot")
            _expect(state.get("layer"), ordinal, f"raw schedule GQA layer {ordinal} state layer")
            _expect(state.get("domain"), "gqa_kv", f"raw schedule GQA layer {ordinal} domain")
            _expect(state.get("state_materialized_by_this_plan"), False, f"raw schedule GQA layer {ordinal} state materialization")
        else:
            raise ScheduleWrapperError(f"raw schedule layer {ordinal} mixer must be delta_net or gqa")
    if len(delta_layers) != 36 or len(gqa_layers) != 12:
        raise ScheduleWrapperError("raw schedule must retain exactly 36 DeltaNet and 12 GQA layers")
    if len(_list(document.get("deltanet_state_slots"), "raw schedule deltanet_state_slots")) != 36:
        raise ScheduleWrapperError("raw schedule must retain 36 DeltaNet state slots")
    if len(_list(document.get("gqa_state_slots"), "raw schedule gqa_state_slots")) != 12:
        raise ScheduleWrapperError("raw schedule must retain 12 GQA state slots")

    layer1 = _mapping(layers[1], "raw schedule layer 1")
    layer1_state = _mapping(layer1.get("state_slot"), "raw schedule layer 1 state_slot")
    _expect(layer1.get("mixer"), "delta_net", "raw schedule layer 1 mixer")
    _expect(layer1_state.get("slot"), 1, "raw schedule layer 1 state slot")
    _expect(layer1_state.get("domain"), "delta_net_conv_and_recurrent", "raw schedule layer 1 state domain")

    command_graph = _list(document.get("full_command_graph_order"), "raw schedule full command graph")
    expected_graph = ["embedding", *[f"layer_{layer}" for layer in range(48)], "final_rmsnorm", "all_row_lm_head", "reserved_tail_mask", "deterministic_sample", "tokenizer_feedback"]
    _expect(command_graph, expected_graph, "raw schedule full command graph order")

    boundary = _mapping(document.get("claim_boundary"), "raw schedule claim_boundary")
    for field in (
        "artifact_payload_open_or_scan_performed",
        "metal_device_or_dispatch_performed",
        "runtime_watcher_registry_server_or_hcli_changed",
        "model_execution_performed",
        "token_generation_or_feedback_performed",
        "tps_or_tg_measured",
    ):
        _expect(boundary.get(field), False, f"raw schedule claim boundary {field}")

    return {
        "all_48_layers_scheduled": True,
        "layer_count": 48,
        "delta_net_layer_count": len(delta_layers),
        "gqa_layer_count": len(gqa_layers),
        "delta_net_state_slot_count": 36,
        "gqa_state_slot_count": 12,
        "full_command_graph_item_count": len(command_graph),
        "layer_1": {
            "layer": 1,
            "mixer": "delta_net",
            "state_slot": 1,
            "state_domain": "delta_net_conv_and_recurrent",
        },
    }


def _validate_canonical_raw_identity(path: Path, raw: bytes) -> None:
    """Bind this wrapper to the one permanent static authority, not a look-alike."""
    _expect(path.name, CANONICAL_RAW_SCHEDULE_FILENAME, "raw schedule filename")
    _expect(len(raw), CANONICAL_RAW_SCHEDULE_BYTES, "raw schedule byte count")
    _expect(_sha256(raw), CANONICAL_RAW_SCHEDULE_SHA256, "raw schedule SHA-256")


def build_wrapper(raw_schedule: Path) -> dict[str, Any]:
    """Build a sealed read-only identity wrapper; does not write the raw plan."""
    path, raw, document = _read_raw_schedule(raw_schedule)
    source_identity = _validate_source_authority(document)
    schedule_facts = _validate_schedule(document)
    _validate_canonical_raw_identity(path, raw)
    wrapper = {
        "schema": WRAPPER_SCHEMA,
        "status": WRAPPER_STATUS,
        "recorded_at": _utc_now(),
        "raw_schedule_authority": {
            "path": str(path),
            "present": True,
            "bytes": len(raw),
            "sha256": _sha256(raw),
            "schema": RAW_SCHEMA,
            "status": RAW_STATUS,
            "raw_schedule_seal_sha256": None,
            "raw_schedule_is_static_and_unmodified": True,
        },
        "source_authority": source_identity,
        "schedule_facts": schedule_facts,
        "claim_boundary": {
            "wrapper_is_read_only": True,
            "raw_schedule_rewritten_or_resealed": False,
            "artifact_payload_open_or_scan_performed": False,
            "metal_device_or_dispatch_performed": False,
            "runtime_server_watcher_or_hcli_changed": False,
            "lease_issued_or_released": False,
            "token_generation_or_feedback_performed": False,
            "tps_or_tg_measured": False,
            "future_joint_l0_to_l1_capture_authorized": False,
        },
    }
    return seal(wrapper)


def write_new(path: Path, wrapper: Mapping[str, Any]) -> Path:
    if not path.is_absolute():
        raise ScheduleWrapperError("--out must be an absolute path")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ScheduleWrapperError("--out parent must be an existing non-symlink directory")
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(dict(wrapper), handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite {path}") from exc
    return path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-schedule", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        wrapper = build_wrapper(args.raw_schedule)
        output = write_new(args.out, wrapper)
    except (OSError, ScheduleWrapperError, ValueError) as exc:
        print(f"ascension_qwen80_48_layer_schedule_sealed_wrapper: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(output), "seal_sha256": wrapper["seal_sha256"], "status": wrapper["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
