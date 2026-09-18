#!/usr/bin/env python3
"""Evaluate Flash PLE over a sealed bank of real native-session contexts.

The prior PLE geometry screen deliberately used synthetic perturbations around
one BOS state.  This adapter is the next, stricter step: it consumes F32
layer-0 states exported from one source-bound *stateful native* Flash session,
reconstructs the exact source n-gram addresses for that same token sequence,
and advances the source PLE convolution state across the whole bank.

The current native Flash body does not itself embed the PLE formula.  Thus
these are real stateful native contexts, but not upstream PLE-inclusive model
trajectory parity.  This file establishes a sealed formula-bank input for a
future conditional/generative PLE representation experiment; it does not
establish independent PLE output parity, a direct runtime, complete EBPW,
capability, TPS, or promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:  # Support package imports and direct repository-script invocation.
    from tools.odyssey.ple_access_trace import load_contract, load_layout, sha256_bytes, trace_token_segments
    from tools.odyssey.ple_context_response_screen import (
        _canonical_sha256,
        _read_sealed_receipt,
        spectral_energy_summary,
    )
    from tools.odyssey.ple_source_output_control import (
        evaluate_ple_sequence,
        load_source_lookup_embeddings,
        load_source_support,
    )
except ModuleNotFoundError:  # pragma: no cover - direct CLI invocation.
    from ple_access_trace import load_contract, load_layout, sha256_bytes, trace_token_segments
    from ple_context_response_screen import _canonical_sha256, _read_sealed_receipt, spectral_energy_summary
    from ple_source_output_control import (
        evaluate_ple_sequence,
        load_source_lookup_embeddings,
        load_source_support,
    )


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.odyssey.ple_real_state_bank_formula.v1"
STATUS = "SOURCE_BOUND_NATIVE_CONTEXT_STATEFUL_PLE_FORMULA_BANK__NO_SOURCE_PLE_TRAJECTORY_PARITY"
STATE_BANK_SCHEMA = "hawking.flash.ple_pre_layer_state_bank.v1"
STATE_BANK_STATUS = "EXPORTED_SOURCE_BOUND_NATIVE_PRE_PLE_STATE_BANK__FORMULA_ONLY"
SESSION_SCHEMA = "hawking.flash.stateful_complete_token_session.v1"
SESSION_STATUS = "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE"
CONTROL_SCHEMA = "hawking.odyssey.ple_source_output_control.v1"
CONTROL_STATUS = "SOURCE_BOUND_PLE_OUTPUT_CONTROL__NO_INDEPENDENT_PLE_OUTPUT_PARITY"
PARITY_SCHEMA = "hawking.odyssey.ple_reference_formula_oracle.v1"
PARITY_STATUS = "REFERENCE_FRAMEWORK_PLE_FORMULA_PARITY_PASS"
DEFAULT_SPEC = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
)
DEFAULT_CONTROL = ROOT / "receipts/headless/FLASH_PLE_LAYER1_BOS_SOURCE_OUTPUT_CONTROL.json"
DEFAULT_PARITY = ROOT / "receipts/headless/FLASH_PLE_LAYER1_BOS_REFERENCE_FORMULA_PARITY.json"


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _bound_path(raw: object, *, parent: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} has no path")
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _summary(path: Path, values: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(values, dtype="<f4").tobytes(order="C")
    return {
        "path": str(path),
        "dtype": "F32_LE",
        "shape": list(values.shape),
        "elements": int(values.size),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "l2": float(np.linalg.norm(values.reshape(-1))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _write_f32(path: Path, values: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes(order="C"))
    return _summary(path, np.asarray(values, dtype=np.float32))


def load_native_pre_ple_state_bank(path: Path) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """Read a complete, hash-bound native state bank and its accepted session."""
    manifest_path = path.expanduser().resolve()
    manifest, manifest_binding = _read_sealed_receipt(
        manifest_path,
        expected_schema=STATE_BANK_SCHEMA,
        expected_status=STATE_BANK_STATUS,
        label="native pre-PLE state-bank manifest",
    )
    session_record = manifest.get("session_receipt")
    capture = manifest.get("capture")
    token_ids = manifest.get("token_ids")
    if not isinstance(session_record, dict) or not isinstance(capture, dict):
        raise ValueError("native pre-PLE state bank lacks session/capture bindings")
    if (
        capture.get("layer") != 0
        or capture.get("state_width") != 10_240
        or capture.get("upstream_PLE_inclusive_source_trajectory") is not False
    ):
        raise ValueError("native pre-PLE state bank has an invalid trajectory boundary")
    if not isinstance(token_ids, list) or len(token_ids) < 2 or not all(_is_int(token) for token in token_ids):
        raise ValueError("native pre-PLE state bank has no multi-token integer sequence")
    session_path = _bound_path(
        session_record.get("path"), parent=manifest_path.parent, label="state-bank session receipt"
    )
    session, session_binding = _read_sealed_receipt(
        session_path,
        expected_schema=SESSION_SCHEMA,
        expected_status=SESSION_STATUS,
        label="source-bound native repeated session",
    )
    if (
        session_record.get("sha256") != session_binding["sha256"]
        or session_record.get("seal_sha256") != session_binding["seal_sha256"]
        or session.get("token_ids") != token_ids
        or session.get("accepted_generation_tokens") != len(token_ids) - int(manifest.get("prompt_length", 0))
    ):
        raise ValueError("native pre-PLE state bank does not bind its accepted session exactly")
    execution = session.get("execution")
    memory = execution.get("state_memory") if isinstance(execution, dict) else None
    if (
        not isinstance(execution, dict)
        or execution.get("source_reset_or_reprefill") is not False
        or not isinstance(memory, dict)
        or int(memory.get("total_persistent_bytes", 0)) <= 0
    ):
        raise ValueError("native pre-PLE state bank's session lacks persistent-state evidence")
    records = manifest.get("states")
    if not isinstance(records, list) or len(records) != len(token_ids):
        raise ValueError("native pre-PLE state bank has incomplete state coverage")
    states: list[np.ndarray] = []
    for step, (record, token_id) in enumerate(zip(records, token_ids, strict=True)):
        if not isinstance(record, dict):
            raise ValueError("native pre-PLE state bank has a non-object state record")
        state_path = _bound_path(record.get("path"), parent=manifest_path.parent, label=f"state {step}")
        raw = state_path.read_bytes()
        if (
            record.get("step") != step
            or record.get("token_id") != token_id
            or record.get("layer") != 0
            or record.get("dtype") != "F32_LE"
            or record.get("elements") != 10_240
            or record.get("bytes") != 10_240 * 4
            or len(raw) != 10_240 * 4
            or record.get("sha256") != sha256_bytes(raw)
            or record.get("finite") is not True
        ):
            raise ValueError(f"native pre-PLE state {step} does not bind its payload")
        state = np.frombuffer(raw, dtype="<f4").copy()
        if not np.isfinite(state).all():
            raise ValueError(f"native pre-PLE state {step} contains non-finite values")
        states.append(state)
    return (
        manifest,
        {
            "manifest": manifest_binding,
            "session": session_binding,
            "session_status": session.get("status"),
            "upstream_PLE_inclusive_source_trajectory": False,
        },
        np.stack(states, axis=0),
    )


def _source_revision(spec: Path) -> str | None:
    fragments = spec.name.rsplit("@", 1)
    return fragments[1] if len(fragments) == 2 and fragments[1] else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--reference-parity", type=Path, default=DEFAULT_PARITY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-injections", type=Path)
    parser.add_argument("--output-conv-state", type=Path)
    args = parser.parse_args(argv)
    started = time.perf_counter_ns()

    control, control_binding = _read_sealed_receipt(
        args.control,
        expected_schema=CONTROL_SCHEMA,
        expected_status=CONTROL_STATUS,
        label="sealed source PLE control",
    )
    parity, parity_binding = _read_sealed_receipt(
        args.reference_parity,
        expected_schema=PARITY_SCHEMA,
        expected_status=PARITY_STATUS,
        label="sealed source PLE reference parity",
    )
    if (parity.get("source_control") or {}).get("seal_sha256") != control_binding["seal_sha256"]:
        raise ValueError("reference PLE parity does not bind the supplied source control")
    manifest, state_bank_binding, states = load_native_pre_ple_state_bank(args.state_bank)
    token_ids = [int(token) for token in manifest["token_ids"]]
    spec = args.spec.expanduser().resolve()
    if not spec.is_dir():
        raise FileNotFoundError(f"Flash source specimen is unavailable: {spec}")
    revision = _source_revision(spec)
    if not revision or manifest.get("pinned_revision") != "34567a4712bc9766c4449e2e98e4468bfa24d915":
        raise ValueError("state bank does not bind the pinned Flash source revision")
    contract, config_binding = load_contract(spec, 0)
    config = json.loads((spec / "config.json").read_text(encoding="utf-8")).get("text_config")
    if not isinstance(config, dict):
        raise ValueError("Flash source config has no text_config")
    hidden_size = config.get("hidden_size")
    hc_count = config.get("hc_count")
    split_parts = config.get("split_ngram_parts")
    eps = config.get("rms_norm_eps")
    if not all(_is_int(value) and value > 0 for value in (hidden_size, hc_count, split_parts)):
        raise ValueError("Flash source config has invalid PLE geometry")
    if not isinstance(eps, (int, float)) or not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("Flash source config has invalid PLE norm epsilon")
    if states.shape != (len(token_ids), int(hidden_size) * int(hc_count)):
        raise ValueError("state-bank geometry disagrees with the current source PLE geometry")
    shards, layout_binding = load_layout(spec, contract, int(split_parts))
    trace = trace_token_segments(contract, [token_ids])
    embeddings, lookup_records, active_lookup = load_source_lookup_embeddings(shards, trace, contract)
    support = load_source_support(spec, contract, hidden_size=int(hidden_size), hc_count=int(hc_count))
    injections, conv_state, intermediates = evaluate_ple_sequence(
        states,
        embeddings,
        support,
        hidden_size=int(hidden_size),
        hc_count=int(hc_count),
        ngram_size=contract.ngram_size,
        eps=float(eps),
    )
    if not np.isfinite(injections).all() or not np.isfinite(conv_state).all():
        raise ValueError("stateful PLE formula evaluation produced non-finite output")
    output = args.output.expanduser().resolve()
    injection_path = (
        args.output_injections.expanduser().resolve()
        if args.output_injections is not None
        else output.with_suffix(".injections.f32")
    )
    conv_path = (
        args.output_conv_state.expanduser().resolve()
        if args.output_conv_state is not None
        else output.with_suffix(".ple_conv_state.f32")
    )
    injection_summary = _write_f32(injection_path, injections)
    conv_summary = _write_f32(conv_path, conv_state)
    support_bytes = sum(int(item["bytes"]) for item in support.bindings)
    state_deltas = states[1:] - states[:1]
    injection_deltas = injections[1:] - injections[:1]
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "specimen": {
            "root": str(spec),
            "model": "Qwen/Qwen3.8-Flash-Next",
            "revision": revision,
            "config": config_binding,
        },
        "source_controls": {
            "source_ple_output_control": control_binding,
            "reference_formula_parity": parity_binding,
            "native_pre_ple_state_bank": state_bank_binding,
        },
        "state_bank": {
            "token_ids": token_ids,
            "prompt_length": manifest.get("prompt_length"),
            "state_shape": list(states.shape),
            "native_contexts": True,
            "upstream_PLE_inclusive_source_trajectory": False,
            "state_delta_spectrum": spectral_energy_summary(state_deltas),
        },
        "source_formula": {
            "evaluated_path": "exact source BF16 support tensors in F32 over persistent PLE convolution state",
            "persistent_ple_convolution": True,
            "ple_embedding_trace": trace,
            "active_lookup": active_lookup,
            "source_lookup_tensor_full_materializations": 0,
            "source_model_loads": 0,
            "intermediate_shapes": {name: list(value.shape) for name, value in intermediates.items()},
        },
        "source_support": {
            "payload_bytes_materialized": support_bytes,
            "tensors": support.bindings,
            "lookup_records": len(lookup_records),
        },
        "outputs": {
            "ple_injections": injection_summary,
            "final_persistent_ple_convolution_state": conv_summary,
            "injection_delta_spectrum": spectral_energy_summary(injection_deltas),
            "post_injection_state_is_not_materialized": True,
        },
        "execution": {
            "backend": "numpy_f32_source_formula_over_native_state_bank",
            "formula_token_count": len(token_ids),
            "source_lookup_row_range_reads": active_lookup["unique_source_rows"],
            "source_support_payload_bytes_materialized": support_bytes,
            "elapsed_ns": time.perf_counter_ns() - started,
        },
        "claim_boundary": (
            "This evaluates the exact source PLE formula, including its persistent convolution state, over a "
            "sealed sequence of real source-bound native layer-0 contexts. The native session does not yet "
            "embed PLE, so this is not upstream PLE-inclusive source-model trajectory parity or an independent "
            "PLE-output oracle. It does not establish a learned function, direct execution, complete EBPW, "
            "capability, TPS, or promotion."
        ),
        "promotion_allowed": False,
        "next": (
            "Use this sealed native-context PLE contribution bank as the minimum real-context discriminator for "
            "a context/history-conditioned PLE replacement. Before any capability or source-parity claim, embed "
            "the PLE formula in the native body and compare it with an independent upstream/source trajectory "
            "contract; bill every generator, state, exception, and runtime dependency."
        ),
    }
    document["seal_sha256"] = _canonical_sha256(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": document["status"],
        "state_count": len(token_ids),
        "state_bank_rank_90": document["state_bank"]["state_delta_spectrum"]["ranks_at_energy"]["0.9"],
        "injection_rank_90": document["outputs"]["injection_delta_spectrum"]["ranks_at_energy"]["0.9"],
        "out": str(output),
        "seal": document["seal_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
