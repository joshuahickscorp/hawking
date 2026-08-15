#!/usr/bin/env python3
"""Fail-closed outer-launch preparation for a future Qwen80 48-layer graph.

This module intentionally never launches a child.  It evaluates whether a
separately controlled future child *could* be authorized after it presents the
immutable raw 48-layer payload/schedule plan together with sealed full-path
evidence for every layer, state slot, and terminal operation.

It does not open a model payload, scan an artifact, allocate a device buffer,
create a Metal context, start a process, bind a port, contact HCLI, generate a
token, or measure TPS/TG.  Current component-only Q80 evidence is deliberately
hard-refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.receipts import SealIntegrityError, seal, verify


INPUT_SCHEMA = "hawking.ascension.qwen80_48_layer_command_graph_outer_launcher_input.v1"
RESULT_SCHEMA = "hawking.ascension.qwen80_48_layer_command_graph_outer_launcher_result.v1"
PREPARED_STATUS = (
    "PREPARED_QWEN80_48_LAYER_COMMAND_GRAPH_OUTER_LAUNCH_CONTRACT_NO_CHILD_STARTED"
)
REFUSED_STATUS = "REFUSED_QWEN80_48_LAYER_COMMAND_GRAPH_OUTER_LAUNCH_INCOMPLETE_NO_CHILD"

PAYLOAD_PLAN_SCHEMA = "hawking.ascension.qwen80_48_layer_payload_schedule_authority.v1"
PAYLOAD_PLAN_STATUS = "PREPARED_QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY_NOT_EXECUTED"
DECODER_SCHEMA = "hawking.ascension.qwen80_complete_decoder_readiness_result.v1"
DECODER_STATUS = "EARNED_QWEN80_COMPLETE_DECODER_READINESS_CONTRACT_ONLY"
LAYER_SCHEMA = "hawking.ascension.qwen80_48_layer_full_path_device_parity_receipt.v1"
LAYER_STATUS = "EARNED_QWEN80_ALL_48_LAYER_FULL_PATH_DEVICE_PARITY_NO_FALLBACK"
STATE_SCHEMA = "hawking.ascension.qwen80_48_layer_state_rollback_receipt.v1"
STATE_STATUS = "EARNED_QWEN80_ALL_SESSION_STATE_ROLLBACK_PARITY_NO_FALLBACK"
TERMINAL_SCHEMA = "hawking.ascension.qwen80_terminal_head_full_token_receipt.v1"
TERMINAL_STATUS = "EARNED_QWEN80_FINAL_NORM_LM_HEAD_TAIL_MASK_SAMPLE_FEEDBACK_NO_FALLBACK"

MODEL_ID = "Qwen3-Coder-Next-80B"
MODEL_KEY = "qwen80"
SOURCE_REPOSITORY = "Qwen/Qwen3-Coder-Next"
SOURCE_REVISION = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb"
SOURCE_CONFIG_SHA256 = "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8"
MANIFEST_SEAL = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b"
ADMISSION_RECEIPT_SEAL = "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628"

LAYERS = 48
DELTANET_LAYERS = 36
GQA_LAYERS = 12
HIDDEN = 2048
EXPERTS = 512
TOP_K = 10
VOCAB = 151_936
TOKENIZER_VOCAB = 151_669
TAIL_ROWS = VOCAB - TOKENIZER_VOCAB
TENSOR_COUNT = 74_391


class CommandGraphOuterLauncherError(ValueError):
    """A prelaunch input is malformed or is unsafe to promote."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, label: str, errors: list[str]) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{label}: missing object")
        return None
    return dict(value)


def _expect(document: Mapping[str, Any], key: str, expected: object, label: str, errors: list[str]) -> None:
    if document.get(key) != expected:
        errors.append(f"{label}.{key}: expected {expected!r}, observed {document.get(key)!r}")


def _expect_true(document: Mapping[str, Any], key: str, label: str, errors: list[str]) -> None:
    if document.get(key) is not True:
        errors.append(f"{label}.{key}: must be true")


def _expect_false(document: Mapping[str, Any], key: str, label: str, errors: list[str]) -> None:
    if document.get(key) is not False:
        errors.append(f"{label}.{key}: must be false")


def _expect_sha256(document: Mapping[str, Any], key: str, label: str, errors: list[str]) -> None:
    if not _is_sha256(document.get(key)):
        errors.append(f"{label}.{key}: must be a lowercase SHA-256")


def _sealed(document: Mapping[str, Any], label: str, errors: list[str]) -> None:
    try:
        verify(document, label=label)
    except SealIntegrityError as exc:
        errors.append(f"{label}: invalid sealed receipt: {exc}")


def _expected_mixer(layer: int) -> str:
    return "gqa" if layer % 4 == 3 else "delta_net"


def _expected_delta_layer(slot: int) -> int:
    return slot // 3 * 4 + slot % 3


def _canonical_regular(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise CommandGraphOuterLauncherError(f"{label} must be absolute: {path}")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CommandGraphOuterLauncherError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CommandGraphOuterLauncherError(f"{label} must be a regular non-symlink file")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise CommandGraphOuterLauncherError(f"cannot canonicalize {label}: {exc}") from exc


def _read_json(path: Path, label: str) -> tuple[Path, dict[str, Any], str]:
    clean = _canonical_regular(path, label)
    try:
        raw = clean.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise CommandGraphOuterLauncherError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CommandGraphOuterLauncherError(f"{label} must be a JSON object")
    return clean, dict(value), _sha256_bytes(raw)


def _sealed_json(path: Path, label: str) -> tuple[Path, dict[str, Any], str, str]:
    clean, document, raw_sha256 = _read_json(path, label)
    errors: list[str] = []
    _sealed(document, label, errors)
    if errors:
        raise CommandGraphOuterLauncherError("; ".join(errors))
    seal_sha256 = document.get("seal_sha256")
    if not _is_sha256(seal_sha256):
        raise CommandGraphOuterLauncherError(f"{label}.seal_sha256 must be a lowercase SHA-256")
    return clean, document, raw_sha256, str(seal_sha256)


def _static_plan_errors(plan: Mapping[str, Any], raw_sha256: str) -> list[str]:
    errors: list[str] = []
    _expect(plan, "schema", PAYLOAD_PLAN_SCHEMA, "payload_schedule_plan", errors)
    _expect(plan, "status", PAYLOAD_PLAN_STATUS, "payload_schedule_plan", errors)
    if plan.get("seal_sha256") is not None:
        errors.append("payload_schedule_plan must remain a raw unsealed static authority")
    _expect(plan, "resolved_tensor_binding_count", TENSOR_COUNT, "payload_schedule_plan", errors)
    _expect_true(plan, "all_48_layers_scheduled", "payload_schedule_plan", errors)
    _expect_true(plan, "all_descriptors_source_artifact_bound", "payload_schedule_plan", errors)

    source = _mapping(plan.get("source_authority"), "payload_schedule_plan.source_authority", errors)
    if source is not None:
        for key, expected in (
            ("model_id", MODEL_ID),
            ("model_key", MODEL_KEY),
            ("source_repository", SOURCE_REPOSITORY),
            ("source_revision", SOURCE_REVISION),
            ("source_config_sha256", SOURCE_CONFIG_SHA256),
            ("descriptor_inventory_seal_sha256", MANIFEST_SEAL),
            ("descriptor_inventory_tensor_count", TENSOR_COUNT),
        ):
            _expect(source, key, expected, "payload_schedule_plan.source_authority", errors)
        for key in (
            "descriptor_inventory_document_sha256",
            "source_config_authority_document_sha256",
            "source_config_authority_seal_sha256",
        ):
            _expect_sha256(source, key, "payload_schedule_plan.source_authority", errors)

    geometry = _mapping(plan.get("geometry"), "payload_schedule_plan.geometry", errors)
    if geometry is not None:
        for key, expected in (
            ("layer_count", LAYERS),
            ("hidden_size", HIDDEN),
            ("experts", EXPERTS),
            ("top_k", TOP_K),
            ("vocab_size", VOCAB),
            ("tokenizer_vocab_size", TOKENIZER_VOCAB),
            ("reserved_lm_head_tail_rows", TAIL_ROWS),
        ):
            _expect(geometry, key, expected, "payload_schedule_plan.geometry", errors)

    embedding = _mapping(plan.get("embedding"), "payload_schedule_plan.embedding", errors)
    if embedding is not None:
        _expect(embedding, "tensor_name", "model.embed_tokens.weight", "payload_schedule_plan.embedding", errors)
        _expect(embedding, "shape", [VOCAB, HIDDEN], "payload_schedule_plan.embedding", errors)

    layers = plan.get("layers")
    if not isinstance(layers, list) or len(layers) != LAYERS:
        errors.append("payload_schedule_plan.layers: requires exactly 48 source layers")
    else:
        for layer_index, layer_value in enumerate(layers):
            layer = _mapping(layer_value, f"payload_schedule_plan.layers[{layer_index}]", errors)
            if layer is None:
                continue
            _expect(layer, "layer", layer_index, f"payload_schedule_plan.layers[{layer_index}]", errors)
            _expect(
                layer,
                "mixer",
                _expected_mixer(layer_index),
                f"payload_schedule_plan.layers[{layer_index}]",
                errors,
            )
            state = _mapping(layer.get("state_slot"), f"payload_schedule_plan.layers[{layer_index}].state_slot", errors)
            if state is None:
                continue
            if layer_index % 4 == 3:
                expected_slot = layer_index // 4
                expected_domain = "gqa_kv"
            else:
                expected_slot = layer_index // 4 * 3 + layer_index % 4
                expected_domain = "delta_net_conv_and_recurrent"
            _expect(state, "layer", layer_index, f"payload_schedule_plan.layers[{layer_index}].state_slot", errors)
            _expect(state, "slot", expected_slot, f"payload_schedule_plan.layers[{layer_index}].state_slot", errors)
            _expect(state, "domain", expected_domain, f"payload_schedule_plan.layers[{layer_index}].state_slot", errors)
            _expect_false(
                state,
                "state_materialized_by_this_plan",
                f"payload_schedule_plan.layers[{layer_index}].state_slot",
                errors,
            )

    for field, expected_count, expected_domain in (
        ("deltanet_state_slots", DELTANET_LAYERS, "delta_net_conv_and_recurrent"),
        ("gqa_state_slots", GQA_LAYERS, "gqa_kv"),
    ):
        slots = plan.get(field)
        if not isinstance(slots, list) or len(slots) != expected_count:
            errors.append(f"payload_schedule_plan.{field}: expected {expected_count} slots")
            continue
        for slot_index, slot_value in enumerate(slots):
            slot = _mapping(slot_value, f"payload_schedule_plan.{field}[{slot_index}]", errors)
            if slot is None:
                continue
            expected_layer = _expected_delta_layer(slot_index) if field == "deltanet_state_slots" else slot_index * 4 + 3
            _expect(slot, "slot", slot_index, f"payload_schedule_plan.{field}[{slot_index}]", errors)
            _expect(slot, "layer", expected_layer, f"payload_schedule_plan.{field}[{slot_index}]", errors)
            _expect(slot, "domain", expected_domain, f"payload_schedule_plan.{field}[{slot_index}]", errors)

    terminal = _mapping(plan.get("terminal_head"), "payload_schedule_plan.terminal_head", errors)
    if terminal is not None:
        final_norm = _mapping(terminal.get("final_norm"), "payload_schedule_plan.terminal_head.final_norm", errors)
        lm_head = _mapping(terminal.get("lm_head"), "payload_schedule_plan.terminal_head.lm_head", errors)
        if final_norm is not None:
            _expect(final_norm, "tensor_name", "model.norm.weight", "payload_schedule_plan.terminal_head.final_norm", errors)
            _expect(final_norm, "shape", [HIDDEN], "payload_schedule_plan.terminal_head.final_norm", errors)
        if lm_head is not None:
            _expect(lm_head, "tensor_name", "lm_head.weight", "payload_schedule_plan.terminal_head.lm_head", errors)
            _expect(lm_head, "shape", [VOCAB, HIDDEN], "payload_schedule_plan.terminal_head.lm_head", errors)
        _expect(terminal, "all_row_lm_head_rows", VOCAB, "payload_schedule_plan.terminal_head", errors)
        _expect(terminal, "tokenizer_addressable_rows", TOKENIZER_VOCAB, "payload_schedule_plan.terminal_head", errors)
        _expect(terminal, "reserved_tail_rows", TAIL_ROWS, "payload_schedule_plan.terminal_head", errors)
        _expect(
            terminal,
            "execution_order",
            [
                "final_rmsnorm",
                "all_row_lm_head",
                "reserved_tail_mask",
                "deterministic_sample",
                "tokenizer_feedback",
            ],
            "payload_schedule_plan.terminal_head",
            errors,
        )

    boundary = _mapping(plan.get("claim_boundary"), "payload_schedule_plan.claim_boundary", errors)
    if boundary is not None:
        _expect_true(boundary, "assembly_authority_only", "payload_schedule_plan.claim_boundary", errors)
        _expect_false(boundary, "decoder_readiness_report", "payload_schedule_plan.claim_boundary", errors)
        for field in (
            "artifact_payload_open_or_scan_performed",
            "metal_device_or_dispatch_performed",
            "runtime_watcher_registry_server_or_hcli_changed",
            "model_execution_performed",
            "token_generation_or_feedback_performed",
            "tps_or_tg_measured",
        ):
            _expect_false(boundary, field, "payload_schedule_plan.claim_boundary", errors)
        _expect(boundary, "execution_status", "PREPARED_NOT_EXECUTED", "payload_schedule_plan.claim_boundary", errors)
    if not _is_sha256(raw_sha256):
        errors.append("payload_schedule_plan raw SHA-256 is malformed")
    return errors


def _expected_plan_binding(plan: Mapping[str, Any], raw_sha256: str) -> dict[str, object]:
    source = dict(plan["source_authority"])
    return {
        "payload_schedule_plan_schema": PAYLOAD_PLAN_SCHEMA,
        "payload_schedule_plan_status": PAYLOAD_PLAN_STATUS,
        "payload_schedule_plan_sha256": raw_sha256,
        "model_id": MODEL_ID,
        "model_key": MODEL_KEY,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "manifest_seal_sha256": MANIFEST_SEAL,
        "admission_receipt_seal_sha256": ADMISSION_RECEIPT_SEAL,
        "descriptor_inventory_document_sha256": source["descriptor_inventory_document_sha256"],
        "source_config_authority_document_sha256": source["source_config_authority_document_sha256"],
    }


def _validate_plan_binding(
    document: Mapping[str, Any],
    plan: Mapping[str, Any],
    raw_sha256: str,
    label: str,
    errors: list[str],
) -> None:
    binding = _mapping(document.get("payload_schedule_binding"), f"{label}.payload_schedule_binding", errors)
    if binding is None:
        return
    for key, expected in _expected_plan_binding(plan, raw_sha256).items():
        _expect(binding, key, expected, f"{label}.payload_schedule_binding", errors)


def _validate_decoder(
    value: object, plan: Mapping[str, Any], raw_sha256: str
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    document = _mapping(value, "decoder_readiness", errors)
    if document is None:
        return None, errors
    _sealed(document, "decoder_readiness", errors)
    _expect(document, "schema", DECODER_SCHEMA, "decoder_readiness", errors)
    _expect(document, "status", DECODER_STATUS, "decoder_readiness", errors)
    _validate_plan_binding(document, plan, raw_sha256, "decoder_readiness", errors)
    for field in (
        "complete_decoder_readiness_earned",
        "real_gravity_server_launch_precondition_satisfied",
        "input_schema_valid",
        "source_artifact_binding_valid",
        "exact_48_layer_schedule_valid",
        "full_command_graph_device_parity_valid",
    ):
        _expect_true(document, field, "decoder_readiness", errors)
    if document.get("missing_operator_classes_or_layers") != []:
        errors.append("decoder_readiness.missing_operator_classes_or_layers: must be empty")
    return document, errors


def _validate_layer_evidence(
    value: object, plan: Mapping[str, Any], raw_sha256: str
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    document = _mapping(value, "layer_parity_receipt", errors)
    if document is None:
        return None, errors
    _sealed(document, "layer_parity_receipt", errors)
    _expect(document, "schema", LAYER_SCHEMA, "layer_parity_receipt", errors)
    _expect(document, "status", LAYER_STATUS, "layer_parity_receipt", errors)
    _validate_plan_binding(document, plan, raw_sha256, "layer_parity_receipt", errors)
    for field in (
        "source_bound",
        "artifact_bound",
        "full_48_layer_device_parity_earned",
        "same_input_provenance_retained",
    ):
        _expect_true(document, field, "layer_parity_receipt", errors)
    for field in ("fixture_only", "component_only", "synthetic_input", "fallback_used", "shadow_model_used"):
        _expect_false(document, field, "layer_parity_receipt", errors)
    same_input = document.get("same_input_provenance_sha256")
    if not _is_sha256(same_input):
        errors.append("layer_parity_receipt.same_input_provenance_sha256: must be a lowercase SHA-256")
    layers = document.get("layers")
    if not isinstance(layers, list) or len(layers) != LAYERS:
        errors.append("layer_parity_receipt.layers: requires exactly 48 entries")
        return document, errors
    for index, entry_value in enumerate(layers):
        entry = _mapping(entry_value, f"layer_parity_receipt.layers[{index}]", errors)
        if entry is None:
            continue
        _expect(entry, "layer", index, f"layer_parity_receipt.layers[{index}]", errors)
        _expect(entry, "mixer", _expected_mixer(index), f"layer_parity_receipt.layers[{index}]", errors)
        for field in ("source_bound", "artifact_bound", "full_path", "device_parity_passed"):
            _expect_true(entry, field, f"layer_parity_receipt.layers[{index}]", errors)
        for field in ("fixture_only", "component_only", "synthetic_input", "fallback_used"):
            _expect_false(entry, field, f"layer_parity_receipt.layers[{index}]", errors)
        _expect(
            entry,
            "same_input_provenance_sha256",
            same_input,
            f"layer_parity_receipt.layers[{index}]",
            errors,
        )
        _expect_sha256(entry, "device_parity_receipt_seal_sha256", f"layer_parity_receipt.layers[{index}]", errors)
    return document, errors


def _validate_slots(
    value: object,
    field: str,
    expected_count: int,
    expected_domain: str,
    plan: Mapping[str, Any],
    raw_sha256: str,
    errors: list[str],
) -> None:
    slots = value
    if not isinstance(slots, list) or len(slots) != expected_count:
        errors.append(f"state_rollback_receipt.{field}: requires exactly {expected_count} slots")
        return
    for slot_index, slot_value in enumerate(slots):
        slot = _mapping(slot_value, f"state_rollback_receipt.{field}[{slot_index}]", errors)
        if slot is None:
            continue
        expected_layer = _expected_delta_layer(slot_index) if expected_domain.startswith("delta") else slot_index * 4 + 3
        _expect(slot, "slot", slot_index, f"state_rollback_receipt.{field}[{slot_index}]", errors)
        _expect(slot, "layer", expected_layer, f"state_rollback_receipt.{field}[{slot_index}]", errors)
        _expect(slot, "domain", expected_domain, f"state_rollback_receipt.{field}[{slot_index}]", errors)
        for required in (
            "device_allocated",
            "state_read_write_parity_passed",
            "rollback_parity_passed",
            "same_plan_bound",
        ):
            _expect_true(slot, required, f"state_rollback_receipt.{field}[{slot_index}]", errors)
        for forbidden in ("fixture_only", "fallback_used"):
            _expect_false(slot, forbidden, f"state_rollback_receipt.{field}[{slot_index}]", errors)
        _expect(slot, "payload_schedule_plan_sha256", raw_sha256, f"state_rollback_receipt.{field}[{slot_index}]", errors)
        _expect_sha256(slot, "state_receipt_seal_sha256", f"state_rollback_receipt.{field}[{slot_index}]", errors)


def _validate_state_evidence(
    value: object, plan: Mapping[str, Any], raw_sha256: str
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    document = _mapping(value, "state_rollback_receipt", errors)
    if document is None:
        return None, errors
    _sealed(document, "state_rollback_receipt", errors)
    _expect(document, "schema", STATE_SCHEMA, "state_rollback_receipt", errors)
    _expect(document, "status", STATE_STATUS, "state_rollback_receipt", errors)
    _validate_plan_binding(document, plan, raw_sha256, "state_rollback_receipt", errors)
    for field in (
        "source_bound",
        "artifact_bound",
        "real_device_resident_state",
        "all_36_deltanet_state_slots_bound",
        "all_12_gqa_kv_slots_bound",
        "rollback_parity_passed",
        "no_cross_session_state_or_kv_leakage",
    ):
        _expect_true(document, field, "state_rollback_receipt", errors)
    for field in ("fixture_only", "component_only", "synthetic_input", "fallback_used"):
        _expect_false(document, field, "state_rollback_receipt", errors)
    _validate_slots(
        document.get("deltanet_slots"),
        "deltanet_slots",
        DELTANET_LAYERS,
        "delta_net_conv_and_recurrent",
        plan,
        raw_sha256,
        errors,
    )
    _validate_slots(
        document.get("gqa_slots"),
        "gqa_slots",
        GQA_LAYERS,
        "gqa_kv",
        plan,
        raw_sha256,
        errors,
    )
    return document, errors


def _validate_terminal_evidence(
    value: object, plan: Mapping[str, Any], raw_sha256: str
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    document = _mapping(value, "terminal_receipt", errors)
    if document is None:
        return None, errors
    _sealed(document, "terminal_receipt", errors)
    _expect(document, "schema", TERMINAL_SCHEMA, "terminal_receipt", errors)
    _expect(document, "status", TERMINAL_STATUS, "terminal_receipt", errors)
    _validate_plan_binding(document, plan, raw_sha256, "terminal_receipt", errors)
    for field in (
        "source_bound",
        "artifact_bound",
        "post_48_hidden_device_parity_passed",
        "final_rmsnorm_device_parity_passed",
        "lm_head_all_rows_device_parity_passed",
        "reserved_tail_mask_applied_before_sample",
        "deterministic_sample_and_feedback_executed",
        "sampled_token_is_tokenizer_addressable",
    ):
        _expect_true(document, field, "terminal_receipt", errors)
    for field in ("fixture_only", "component_only", "synthetic_input", "fallback_used", "shadow_model_used"):
        _expect_false(document, field, "terminal_receipt", errors)
    _expect(document, "post_48_hidden_shape", [HIDDEN], "terminal_receipt", errors)
    _expect(document, "lm_head_rows", VOCAB, "terminal_receipt", errors)
    _expect(document, "tokenizer_addressable_rows", TOKENIZER_VOCAB, "terminal_receipt", errors)
    _expect(document, "reserved_tail_rows", TAIL_ROWS, "terminal_receipt", errors)
    _expect_sha256(document, "same_input_provenance_sha256", "terminal_receipt", errors)
    return document, errors


def _condition(name: str, errors: Sequence[str]) -> dict[str, object]:
    return {"name": name, "satisfied": not errors, "blockers": list(errors)}


def _claim_boundary() -> dict[str, object]:
    return {
        "outer_launcher_started_no_child": True,
        "outer_launcher_started_no_process": True,
        "outer_launcher_opened_no_model_payload": True,
        "outer_launcher_performed_no_artifact_scan": True,
        "outer_launcher_allocated_no_device_buffer": True,
        "outer_launcher_created_no_metal_context_or_dispatch": True,
        "outer_launcher_bound_no_port": True,
        "outer_launcher_contacted_no_hcli": True,
        "outer_launcher_generated_no_token": True,
        "outer_launcher_measured_no_tps_or_tg": True,
    }


def current_component_evidence() -> dict[str, object]:
    """Return the explicit non-promotable current component-only input."""
    return {
        "schema": INPUT_SCHEMA,
        "current_component_only": True,
        "partial_components": [
            "L0 DeltaNet mixer component",
            "L0 postnorm/router component",
            "L0 routed/shared/combine components",
            "layer-3 GQA component",
            "CPU terminal-head component contract",
        ],
    }


def assess_outer_launch(evidence: Mapping[str, Any]) -> dict[str, object]:
    if evidence.get("schema") != INPUT_SCHEMA:
        raise CommandGraphOuterLauncherError("outer launcher input schema drifted")
    if evidence.get("current_component_only") is True:
        partial = evidence.get("partial_components")
        return {
            "schema": RESULT_SCHEMA,
            "status": REFUSED_STATUS,
            "future_child_launch_eligible": False,
            "separate_controlled_child_required_if_future_eligible": True,
            "current_component_evidence_hard_refused": True,
            "conditions": [
                _condition(
                    "raw_static_payload_schedule_authority",
                    ["current component evidence provides no immutable 48-layer payload/schedule authority"],
                ),
                _condition(
                    "sealed_full_48_layer_device_parity",
                    ["current components are not a same-input full-path 48-layer device-parity receipt"],
                ),
                _condition(
                    "sealed_all_state_slots_and_rollback",
                    ["current components provide neither all 36 DeltaNet nor all 12 GQA state-slot witnesses"],
                ),
                _condition(
                    "sealed_terminal_head_full_token_path",
                    ["current terminal component contract has no post-48 hidden/device parity/token feedback witness"],
                ),
            ],
            "blockers": [
                "current Q80 component-only evidence is not promotable to a command graph or child launch",
                f"partial components retained only as context: {partial!r}",
            ],
            "claim_boundary": _claim_boundary(),
        }

    plan = _mapping(evidence.get("payload_schedule_plan"), "payload_schedule_plan", [])
    raw_sha256 = evidence.get("payload_schedule_plan_sha256")
    if plan is None or not _is_sha256(raw_sha256):
        raise CommandGraphOuterLauncherError("outer launcher requires a raw payload/schedule plan and SHA-256")
    plan_sha256 = str(raw_sha256)
    plan_errors = _static_plan_errors(plan, plan_sha256)
    decoder, decoder_errors = _validate_decoder(evidence.get("decoder_readiness"), plan, plan_sha256)
    layers, layer_errors = _validate_layer_evidence(evidence.get("layer_parity_receipt"), plan, plan_sha256)
    state, state_errors = _validate_state_evidence(evidence.get("state_rollback_receipt"), plan, plan_sha256)
    terminal, terminal_errors = _validate_terminal_evidence(evidence.get("terminal_receipt"), plan, plan_sha256)

    cross_errors: list[str] = []
    same_inputs = []
    for label, document in (("layer", layers), ("terminal", terminal)):
        if document is not None:
            value = document.get("same_input_provenance_sha256")
            if _is_sha256(value):
                same_inputs.append((label, str(value)))
    if len(same_inputs) != 2 or len({value for _, value in same_inputs}) != 1:
        cross_errors.append("layer and terminal receipts must retain one identical real-input provenance SHA-256")
    for label, document in (("decoder", decoder), ("layers", layers), ("state", state), ("terminal", terminal)):
        if document is not None:
            binding = document.get("payload_schedule_binding")
            if not isinstance(binding, Mapping) or binding.get("payload_schedule_plan_sha256") != plan_sha256:
                cross_errors.append(f"{label} evidence does not bind the immutable raw payload/schedule plan SHA-256")

    conditions = [
        _condition("raw_static_payload_schedule_authority", plan_errors),
        _condition("sealed_truthful_complete_decoder_readiness", decoder_errors),
        _condition("sealed_full_48_layer_device_parity", layer_errors),
        _condition("sealed_all_state_slots_and_rollback", state_errors),
        _condition("sealed_terminal_head_full_token_path", terminal_errors),
        _condition("same_plan_same_input_cross_receipt_provenance", cross_errors),
    ]
    blockers = [blocker for condition in conditions for blocker in condition["blockers"]]
    eligible = not blockers
    return {
        "schema": RESULT_SCHEMA,
        "status": PREPARED_STATUS if eligible else REFUSED_STATUS,
        "future_child_launch_eligible": eligible,
        "separate_controlled_child_required_if_future_eligible": True,
        "current_component_evidence_hard_refused": False,
        "payload_schedule_plan_sha256": plan_sha256,
        "conditions": conditions,
        "blockers": blockers,
        "claim_boundary": _claim_boundary(),
    }


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise CommandGraphOuterLauncherError("--out must be absolute")
    if path.exists():
        raise CommandGraphOuterLauncherError(f"refusing to overwrite {path}")
    if not path.parent.is_dir():
        raise CommandGraphOuterLauncherError("--out parent directory must already exist")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise CommandGraphOuterLauncherError(f"refusing to overwrite {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parse_args(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--current-component-evidence", action="store_true")
    mode.add_argument("--payload-schedule-plan", type=Path)
    parser.add_argument("--payload-schedule-plan-sha256")
    parser.add_argument("--decoder-readiness", type=Path)
    parser.add_argument("--layer-parity-receipt", type=Path)
    parser.add_argument("--state-rollback-receipt", type=Path)
    parser.add_argument("--terminal-receipt", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parsed = parser.parse_args(arguments)
    if parsed.current_component_evidence:
        for field in (
            "payload_schedule_plan_sha256",
            "decoder_readiness",
            "layer_parity_receipt",
            "state_rollback_receipt",
            "terminal_receipt",
        ):
            if getattr(parsed, field) is not None:
                parser.error(f"--current-component-evidence cannot be combined with --{field.replace('_', '-')}")
    else:
        for field in (
            "payload_schedule_plan_sha256",
            "decoder_readiness",
            "layer_parity_receipt",
            "state_rollback_receipt",
            "terminal_receipt",
        ):
            if getattr(parsed, field) is None:
                parser.error(f"--payload-schedule-plan requires --{field.replace('_', '-')}")
        if not _is_sha256(parsed.payload_schedule_plan_sha256):
            parser.error("--payload-schedule-plan-sha256 must be a lowercase SHA-256")
    return parsed


def _load_path_evidence(parsed: argparse.Namespace) -> dict[str, object]:
    if parsed.current_component_evidence:
        return current_component_evidence()
    plan_path, plan, plan_sha256 = _read_json(parsed.payload_schedule_plan, "--payload-schedule-plan")
    if plan_sha256 != parsed.payload_schedule_plan_sha256:
        raise CommandGraphOuterLauncherError("--payload-schedule-plan SHA-256 does not match --payload-schedule-plan-sha256")
    _, decoder, _, _ = _sealed_json(parsed.decoder_readiness, "--decoder-readiness")
    _, layers, _, _ = _sealed_json(parsed.layer_parity_receipt, "--layer-parity-receipt")
    _, state, _, _ = _sealed_json(parsed.state_rollback_receipt, "--state-rollback-receipt")
    _, terminal, _, _ = _sealed_json(parsed.terminal_receipt, "--terminal-receipt")
    return {
        "schema": INPUT_SCHEMA,
        "payload_schedule_plan": plan,
        "payload_schedule_plan_sha256": plan_sha256,
        "payload_schedule_plan_path": str(plan_path),
        "decoder_readiness": decoder,
        "layer_parity_receipt": layers,
        "state_rollback_receipt": state,
        "terminal_receipt": terminal,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parse_args(sys.argv[1:] if arguments is None else arguments)
    try:
        report = assess_outer_launch(_load_path_evidence(parsed))
        report = seal({"recorded_at": _utc_now(), **report})
        _atomic_json_new(parsed.out, report)
    except CommandGraphOuterLauncherError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(parsed.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
