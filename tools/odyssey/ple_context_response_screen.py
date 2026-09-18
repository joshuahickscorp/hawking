#!/usr/bin/env python3
"""Measure a source-bound local PLE response over address and context.

Flash's PLE lookup table is only one input to a stateful function.  This
screen deliberately stops treating that table as the object to be encoded and
instead asks a narrower prior question: over a small, deterministic local
neighbourhood of an exact source pre-PLE state, how does the exact PLE formula
respond at real hashed lookup addresses?

The neighbourhood is *synthetic*.  It is a cheap tangent/curvature
discriminator around one sealed source state, not a substitute for a bank of
real source trajectory states.  In particular, it does not establish a
context-conditioned generator, PLE replacement, direct runtime, complete
EBPW, stateful PLE fidelity, model capability, or physical performance.

The source-control token is checked against the sealed PLE injection.  Other
addresses are drawn from a sealed source-bound historical address trace, but
all reuse the sealed BOS pre-layer state only as a local synthetic probe.  The
historical session that originally supplied that trace has a stale inner seal,
so this screen never reasserts its acceptance claim.  A real multi-context
source-state bank is a later, separately bound gate if this geometry makes
learned function work worth attempting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # Support package imports and direct receipt invocation.
    from tools.odyssey.ple_access_trace import (
        load_contract,
        load_layout,
        sha256_bytes,
        trace_token_segments,
    )
    from tools.odyssey.ple_lookup_pq_output_screen import output_metrics
    from tools.odyssey.ple_source_output_control import (
        SCHEMA as SOURCE_CONTROL_SCHEMA,
        evaluate_ple_zero_state_context_batch,
        load_source_lookup_embeddings,
        load_source_support,
    )
except ModuleNotFoundError:  # pragma: no cover - direct receipt invocation.
    from ple_access_trace import load_contract, load_layout, sha256_bytes, trace_token_segments
    from ple_lookup_pq_output_screen import output_metrics
    from ple_source_output_control import (
        SCHEMA as SOURCE_CONTROL_SCHEMA,
        evaluate_ple_zero_state_context_batch,
        load_source_lookup_embeddings,
        load_source_support,
    )


SCHEMA = "hawking.odyssey.ple_context_response_screen.v1"
STATUS = "SOURCE_PLE_CONTEXT_RESPONSE_TANGENT_ONLY"
REFERENCE_ORACLE_SCHEMA = "hawking.odyssey.ple_reference_formula_oracle.v1"
PLE_ACCESS_TRACE_SCHEMA = "hawking.odyssey.ple_access_trace.v1"
STATEFUL_SESSION_SCHEMA = "hawking.flash.stateful_complete_token_session.v1"
STATEFUL_SESSION_STATUS = "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTROL = Path("receipts/headless/FLASH_PLE_LAYER1_BOS_SOURCE_OUTPUT_CONTROL.json")
DEFAULT_PARITY = Path("receipts/headless/FLASH_PLE_LAYER1_BOS_REFERENCE_FORMULA_PARITY.json")
DEFAULT_ADDRESS_TRACE = Path("receipts/headless/FLASH_PLE_ACCEPTED_TOKEN_ACCESS_TRACE.json")
DEFAULT_DIRECTION_COUNT = 8
DEFAULT_RELATIVE_RMS = 0.01
DEFAULT_SEED = 20_260_912
BASELINE_RELATIVE_L2_MAX = 1e-6
BASELINE_MAX_ABS_MAX = 1e-6


def _canonical_sha256(document: dict[str, Any]) -> str:
    """Legacy Python receipt seal used by the existing PLE adapters."""
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode("utf-8")).hexdigest()


def _compact_utf8_canonical_sha256(document: dict[str, Any]) -> str:
    """Compact UTF-8 canonical seal emitted by native Rust source controls.

    Rust's ``serde_json`` writes compact UTF-8 JSON, whereas the early Python
    PLE adapters used Python's space-bearing ASCII-escaped default.  Both are
    explicit, finite receipt formats; accepting neither a third ad-hoc format
    nor an unsealed body keeps the cross-language boundary fail-closed.
    """
    return hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_sealed_receipt(
    path: Path, *, expected_schema: str, expected_status: str, label: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """Read a precise scientific input instead of accepting stale prose."""
    resolved = path.expanduser().resolve()
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or document.get("schema") != expected_schema
        or document.get("status") != expected_status
    ):
        raise ValueError(f"{label} has an incompatible schema or status: {resolved}")
    seal = document.get("seal_sha256")
    unsealed = dict(document)
    unsealed.pop("seal_sha256", None)
    seal_formats = {
        "python_json_sorted_default_ascii_v1": _canonical_sha256(unsealed),
        "rust_serde_json_compact_utf8_sorted_v1": _compact_utf8_canonical_sha256(unsealed),
    }
    matched_format = next(
        (format_name for format_name, candidate in seal_formats.items() if seal == candidate),
        None,
    )
    if not isinstance(seal, str) or matched_format is None:
        raise ValueError(f"{label} has an invalid seal: {resolved}")
    return document, {
        "path": str(resolved),
        "sha256": sha256_bytes(resolved.read_bytes()),
        "seal_sha256": seal,
        "seal_format": matched_format,
    }


def _resolve_record_path(raw: object) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("a bound F32 record has no path")
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_bound_f32(record: object, *, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an exact F32 support artifact after verifying its receipt binding."""
    if not isinstance(record, dict):
        raise ValueError(f"{label} record is missing")
    path = _resolve_record_path(record.get("path"))
    raw = path.read_bytes()
    if (
        record.get("dtype") != "F32_LE"
        or record.get("bytes") != len(raw)
        or record.get("elements") != len(raw) // 4
        or record.get("sha256") != sha256_bytes(raw)
    ):
        raise ValueError(f"{label} F32 artifact disagrees with its receipt: {path}")
    return np.frombuffer(raw, dtype="<f4").copy(), {
        "path": str(path),
        "dtype": "F32_LE",
        "bytes": len(raw),
        "elements": len(raw) // 4,
        "sha256": sha256_bytes(raw),
    }


def _require_int_sequence(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list) or not value or not all(_is_int(item) for item in value):
        raise ValueError(f"{label} must be a non-empty integer token sequence")
    return [int(item) for item in value]


def build_symmetric_context_bank(
    base_state: np.ndarray,
    *,
    direction_count: int,
    relative_rms: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a deterministic, orthogonal, local synthetic context bank.

    The first row is the exact source state.  Each remaining pair is a plus
    and minus perturbation of equal RMS, with directions orthogonal to the
    source state and to earlier directions.  The vectors themselves are not
    persisted in a receipt, only their deterministic construction/hash.
    """
    base = np.asarray(base_state, dtype=np.float32).reshape(-1)
    direction_count = int(direction_count)
    relative_rms = float(relative_rms)
    if base.size < 2 or not np.isfinite(base).all():
        raise ValueError("base context must be a finite vector with at least two values")
    if direction_count < 1 or direction_count >= base.size:
        raise ValueError("direction count must be positive and smaller than context width")
    if not math.isfinite(relative_rms) or relative_rms <= 0.0 or relative_rms > 1.0:
        raise ValueError("relative context RMS must be finite in (0, 1]")
    base_l2 = float(np.linalg.norm(base))
    if base_l2 <= 1e-30:
        raise ValueError("source context has zero norm and cannot define an orthogonal bank")
    base_unit = base / np.float32(base_l2)
    generator = np.random.default_rng(int(seed))
    directions: list[np.ndarray] = []
    for _ in range(direction_count):
        direction = generator.standard_normal(base.size).astype(np.float32)
        direction -= np.dot(direction, base_unit).astype(np.float32) * base_unit
        for earlier in directions:
            direction -= np.dot(direction, earlier).astype(np.float32) * earlier
        direction_l2 = float(np.linalg.norm(direction))
        if direction_l2 <= 1e-20:
            raise ValueError("synthetic context direction lost all independent support")
        directions.append((direction / np.float32(direction_l2)).astype(np.float32))
    direction_matrix = np.stack(directions, axis=0)
    base_rms = float(math.sqrt(float(np.mean(base * base))))
    perturbation_l2 = relative_rms * base_rms * math.sqrt(base.size)
    perturbations = direction_matrix * np.float32(perturbation_l2)
    contexts = [base]
    for perturbation in perturbations:
        contexts.extend((base + perturbation, base - perturbation))
    bank = np.stack(contexts, axis=0).astype(np.float32)
    if not np.isfinite(bank).all():
        raise ValueError("synthetic context bank contains non-finite values")
    return bank, {
        "construction": "deterministic orthogonal symmetric local perturbations around sealed source state",
        "synthetic": True,
        "source_trajectory_distribution": False,
        "seed": int(seed),
        "context_width": int(base.size),
        "direction_count": direction_count,
        "context_count": int(bank.shape[0]),
        "relative_rms": relative_rms,
        "base_rms": base_rms,
        "perturbation_l2": perturbation_l2,
        "directions_orthogonal_to_base": True,
        "directions_mutually_orthogonal": True,
        "direction_matrix_sha256": sha256_bytes(direction_matrix.astype("<f4").tobytes()),
        "context_bank_sha256": sha256_bytes(bank.astype("<f4").tobytes()),
        "ordering": "row zero is base; each following pair is plus, minus for one direction",
        "ple_convolution_history": "zero independently for every context row",
    }


def spectral_energy_summary(
    responses: np.ndarray, *, thresholds: Sequence[float] = (0.90, 0.95, 0.99)
) -> dict[str, Any]:
    """Summarize a response subspace without retaining the high-dimensional rows."""
    rows = np.asarray(responses, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 1 or not np.isfinite(rows).all():
        raise ValueError("response spectrum requires a finite non-empty matrix")
    checked_thresholds = tuple(float(value) for value in thresholds)
    if not checked_thresholds or any(value <= 0.0 or value >= 1.0 for value in checked_thresholds):
        raise ValueError("energy thresholds must be strictly between zero and one")
    gram = rows @ rows.T
    eigenvalues = np.linalg.eigvalsh(np.asarray(gram, dtype=np.float64))[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total_energy = float(np.sum(eigenvalues))
    if total_energy <= 1e-30:
        return {
            "response_rows": int(rows.shape[0]),
            "response_width": int(rows.shape[1]),
            "total_energy": total_energy,
            "numerical_rank": 0,
            "ranks_at_energy": {str(value): None for value in checked_thresholds},
            "top_singular_values": [],
        }
    normalized = np.cumsum(eigenvalues) / total_energy
    numerical_floor = max(float(eigenvalues[0]) * 1e-10, 1e-20)
    return {
        "response_rows": int(rows.shape[0]),
        "response_width": int(rows.shape[1]),
        "total_energy": total_energy,
        "numerical_rank": int(np.count_nonzero(eigenvalues > numerical_floor)),
        "ranks_at_energy": {
            str(value): int(np.searchsorted(normalized, value, side="left") + 1)
            for value in checked_thresholds
        },
        "top_singular_values": [
            float(math.sqrt(value)) for value in eigenvalues[: min(16, eigenvalues.size)]
        ],
    }


def _summary_statistics(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 1 or not np.isfinite(array).all():
        raise ValueError("response statistics need at least one finite value")
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def summarize_symmetric_response(
    outputs: np.ndarray, *, direction_count: int
) -> tuple[dict[str, Any], np.ndarray]:
    """Measure local linear response and symmetric nonlinear curvature."""
    values = np.asarray(outputs, dtype=np.float32)
    expected_count = 1 + 2 * int(direction_count)
    if values.ndim != 2 or values.shape[0] != expected_count or not np.isfinite(values).all():
        raise ValueError("context outputs do not match the declared symmetric bank")
    base = values[0]
    deltas = values[1:] - base[None, :]
    curvature_ratios: list[float] = []
    curvature_l2: list[float] = []
    linear_l2: list[float] = []
    step_l2: list[float] = []
    for index in range(int(direction_count)):
        plus = values[1 + 2 * index]
        minus = values[2 + 2 * index]
        plus_step = float(np.linalg.norm(plus - base))
        minus_step = float(np.linalg.norm(minus - base))
        curvature = plus + minus - np.float32(2.0) * base
        curvature_norm = float(np.linalg.norm(curvature))
        linear_norm = float(np.linalg.norm((plus - minus) * np.float32(0.5)))
        curvature_ratios.append(curvature_norm / max((plus_step + minus_step) * 0.5, 1e-30))
        curvature_l2.append(curvature_norm)
        linear_l2.append(linear_norm)
        step_l2.extend((plus_step, minus_step))
    return {
        "base_output_l2": float(np.linalg.norm(base)),
        "direction_count": int(direction_count),
        "step_l2": _summary_statistics(step_l2),
        "linear_response_l2": _summary_statistics(linear_l2),
        "symmetric_curvature_l2": _summary_statistics(curvature_l2),
        "normalized_symmetric_curvature": _summary_statistics(curvature_ratios),
        "local_delta_spectrum": spectral_energy_summary(deltas),
    }, deltas


def _source_revision(spec: Path) -> str | None:
    fields = spec.name.rsplit("@", 1)
    return fields[1] if len(fields) == 2 and fields[1] else None


def _address_trace_signature(trace: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the deterministic address contract for cross-receipt checks."""
    result: list[dict[str, Any]] = []
    for token in trace:
        accesses = token.get("accesses") if isinstance(token, dict) else None
        if not isinstance(accesses, list):
            raise ValueError("address trace has a token without access records")
        result.append(
            {
                "token_id": token.get("token_id"),
                "accesses": [
                    {
                        "global_head": access.get("global_head"),
                        "combined_row": access.get("combined_row"),
                    }
                    for access in accesses
                ],
            }
        )
    return result


def _address_result(
    *,
    label: str,
    token_id: int,
    source_group: str,
    source_token_index: int,
    embedding: np.ndarray,
    contexts: np.ndarray,
    support: Any,
    hidden_size: int,
    hc_count: int,
    ngram_size: int,
    eps: float,
    lookup_records: list[dict[str, Any]],
    expected_lookup_rows: int,
    direction_count: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    outputs, intermediates = evaluate_ple_zero_state_context_batch(
        contexts,
        embedding,
        support,
        hidden_size=hidden_size,
        hc_count=hc_count,
        ngram_size=ngram_size,
        eps=eps,
    )
    response, deltas = summarize_symmetric_response(outputs, direction_count=direction_count)
    if len(lookup_records) != int(expected_lookup_rows):
        # Every PLE embedding has exactly one declared row per global head.
        # This binds response output to the same source-address contract.
        raise ValueError("PLE lookup records have inconsistent embedding head geometry")
    return {
        "label": label,
        "token_id": int(token_id),
        "source_group": source_group,
        "source_token_index": int(source_token_index),
        "lookup_rows": [int(record["combined_row"]) for record in lookup_records],
        "lookup_shards": [int(record["shard_ordinal"]) for record in lookup_records],
        "intermediate_shapes": {name: list(value.shape) for name, value in intermediates.items()},
        "response": response,
        "base_output_sha256": sha256_bytes(outputs[0].astype("<f4").tobytes()),
    }, deltas, outputs[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--reference-parity", type=Path, default=DEFAULT_PARITY)
    parser.add_argument("--accepted-address-trace", type=Path, default=DEFAULT_ADDRESS_TRACE)
    parser.add_argument("--direction-count", type=int, default=DEFAULT_DIRECTION_COUNT)
    parser.add_argument("--relative-rms", type=float, default=DEFAULT_RELATIVE_RMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started_ns = time.perf_counter_ns()

    control, control_binding = _read_sealed_receipt(
        args.control,
        expected_schema=SOURCE_CONTROL_SCHEMA,
        expected_status="SOURCE_BOUND_PLE_OUTPUT_CONTROL__NO_INDEPENDENT_PLE_OUTPUT_PARITY",
        label="source PLE control",
    )
    parity, parity_binding = _read_sealed_receipt(
        args.reference_parity,
        expected_schema=REFERENCE_ORACLE_SCHEMA,
        expected_status="REFERENCE_FRAMEWORK_PLE_FORMULA_PARITY_PASS",
        label="reference PLE parity",
    )
    address_trace, address_trace_binding = _read_sealed_receipt(
        args.accepted_address_trace,
        expected_schema=PLE_ACCESS_TRACE_SCHEMA,
        expected_status="SOURCE_ALGORITHM_BOUND_TOKEN_ACCESS_TRACE__NOT_PLE_OUTPUT_PARITY",
        label="source-bound historical Flash address trace",
    )
    if (parity.get("source_control") or {}).get("seal_sha256") != control_binding["seal_sha256"]:
        raise ValueError("reference formula parity does not bind the supplied source control")
    specimen = control.get("specimen")
    input_control = control.get("input_control")
    outputs = control.get("outputs")
    sequence = control.get("token_sequence")
    if not all(isinstance(value, dict) for value in (specimen, input_control, outputs, sequence)):
        raise ValueError("source PLE control lacks required specimen/input/output/sequence fields")
    source_token_ids = _require_int_sequence(sequence.get("token_ids"), label="source control token IDs")
    if len(source_token_ids) != 1:
        raise ValueError("context-response screen needs exactly one sealed source-control token")
    source_token_id = source_token_ids[0]
    address_sequence = address_trace.get("sequence")
    if not isinstance(address_sequence, dict):
        raise ValueError("source-bound address trace has no sequence binding")
    accepted_segments = address_sequence.get("segments")
    if (
        not isinstance(accepted_segments, list)
        or not accepted_segments
        or not all(isinstance(segment, list) for segment in accepted_segments)
    ):
        raise ValueError("source-bound address trace has invalid token segments")
    accepted_segments = [
        _require_int_sequence(segment, label="source-bound address segment")
        for segment in accepted_segments
    ]
    accepted_tokens = [token for segment in accepted_segments for token in segment]
    source_session_path = _resolve_record_path(address_sequence.get("path"))
    source_session, source_session_binding = _read_sealed_receipt(
        source_session_path,
        expected_schema=STATEFUL_SESSION_SCHEMA,
        expected_status=STATEFUL_SESSION_STATUS,
        label="source-bound repeated Flash stateful session",
    )
    if source_session_binding["sha256"] != address_sequence.get("sha256"):
        raise ValueError("historical address trace does not bind the supplied stateful session bytes")
    if source_session.get("pinned_revision") != address_sequence.get("pinned_revision"):
        raise ValueError("historical address trace revision disagrees with its stateful session")
    if source_session.get("token_ids") != accepted_tokens:
        raise ValueError("historical address trace token sequence disagrees with its stateful session")
    if source_session.get("accepted_generation_tokens") != len(accepted_segments[-1]):
        raise ValueError("historical stateful session did not accept the traced continuations")
    spec = Path(str(specimen.get("root", ""))).expanduser().resolve()
    if not spec.is_dir():
        raise FileNotFoundError(f"source specimen is not available: {spec}")
    revision = _source_revision(spec)
    trace_specimen = address_trace.get("specimen") or {}
    if not revision or not str(trace_specimen.get("revision_prefix", "")).startswith(revision):
        raise ValueError("source-bound address trace revision disagrees with source PLE control")
    contract, config_binding = load_contract(spec, 0)
    source_config = ((control.get("reference_contract") or {}).get("config") or {})
    if source_config.get("sha256") != config_binding.get("sha256"):
        raise ValueError("source PLE control config binding disagrees with current specimen")
    config = json.loads((spec / "config.json").read_text(encoding="utf-8")).get("text_config")
    if not isinstance(config, dict):
        raise ValueError("Flash source config has no text_config")
    hidden_size = config.get("hidden_size")
    hc_count = config.get("hc_count")
    split_parts = config.get("split_ngram_parts")
    eps = config.get("rms_norm_eps")
    if not all(_is_int(value) and value > 0 for value in (hidden_size, hc_count, split_parts)):
        raise ValueError("Flash source config has invalid PLE dimensions")
    if not isinstance(eps, (int, float)) or not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("Flash source config has invalid PLE RMSNorm epsilon")
    expected_width = int(hidden_size) * int(hc_count)
    base_state, base_binding = _read_bound_f32(input_control.get("state"), label="sealed pre-PLE state")
    source_injection, source_injection_binding = _read_bound_f32(
        outputs.get("ple_injection"), label="sealed source PLE injection"
    )
    reference_injection, reference_injection_binding = _read_bound_f32(
        (parity.get("reference_output") or {}), label="reference PLE injection"
    )
    if any(value.size != expected_width for value in (base_state, source_injection, reference_injection)):
        raise ValueError("bound source PLE vectors disagree with source hidden geometry")
    contexts, context_bank = build_symmetric_context_bank(
        base_state,
        direction_count=args.direction_count,
        relative_rms=args.relative_rms,
        seed=args.seed,
    )
    shards, layout_binding = load_layout(spec, contract, int(split_parts))
    source_trace = trace_token_segments(contract, [[source_token_id]])
    accepted_trace = trace_token_segments(contract, accepted_segments)
    stored_trace = address_trace.get("trace")
    if not isinstance(stored_trace, list) or _address_trace_signature(accepted_trace) != _address_trace_signature(stored_trace):
        raise ValueError("recomputed PLE address contract disagrees with the sealed address trace")
    source_embeddings, source_ordered, source_lookup = load_source_lookup_embeddings(
        shards, source_trace, contract
    )
    accepted_embeddings, accepted_ordered, accepted_lookup = load_source_lookup_embeddings(
        shards, accepted_trace, contract
    )
    trace_lookup = address_trace.get("active_lookup") or {}
    if accepted_lookup.get("source_range_read_sha256") != trace_lookup.get("source_range_read_sha256"):
        raise ValueError("fresh source PLE range reads disagree with the sealed address trace")
    support = load_source_support(spec, contract, hidden_size=int(hidden_size), hc_count=int(hc_count))
    support_bytes = sum(int(binding["bytes"]) for binding in support.bindings)
    source_records = [
        record for record in source_ordered if int(record["session_token_index"]) == 0
    ]
    source_result, source_deltas, source_batch_injection = _address_result(
        label="sealed_bos_source_control",
        token_id=source_token_id,
        source_group="sealed_source_control",
        source_token_index=0,
        embedding=source_embeddings[0],
        contexts=contexts,
        support=support,
        hidden_size=int(hidden_size),
        hc_count=int(hc_count),
        ngram_size=contract.ngram_size,
        eps=float(eps),
        lookup_records=source_records,
        expected_lookup_rows=contract.ngram_heads,
        direction_count=args.direction_count,
    )
    source_batch_metrics = output_metrics(source_injection, source_batch_injection)
    reference_batch_metrics = output_metrics(reference_injection, source_batch_injection)
    if (
        source_batch_metrics["relative_l2"] > BASELINE_RELATIVE_L2_MAX
        or source_batch_metrics["max_abs"] > BASELINE_MAX_ABS_MAX
        or not source_batch_metrics["finite"]
    ):
        raise ValueError("zero-history batch primitive does not reproduce sealed source control")
    address_results = [source_result]
    cross_deltas = [source_deltas]
    for index, trace_token in enumerate(accepted_trace):
        records = [
            record
            for record in accepted_ordered
            if int(record["session_token_index"]) == int(trace_token["session_token_index"])
        ]
        result, deltas, _ = _address_result(
            label=f"historical_stateful_address_{index}",
            token_id=int(trace_token["token_id"]),
            source_group="historical_stateful_token_address_only",
            source_token_index=index,
            embedding=accepted_embeddings[index],
            contexts=contexts,
            support=support,
            hidden_size=int(hidden_size),
            hc_count=int(hc_count),
            ngram_size=contract.ngram_size,
            eps=float(eps),
            lookup_records=records,
            expected_lookup_rows=contract.ngram_heads,
            direction_count=args.direction_count,
        )
        address_results.append(result)
        cross_deltas.append(deltas)
    cross_response = spectral_energy_summary(np.concatenate(cross_deltas, axis=0))
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "specimen": {
            "root": str(spec),
            "model": specimen.get("model"),
            "revision": revision,
            "config": config_binding,
        },
        "source_controls": {
            "source_ple_output_control": control_binding,
            "reference_formula_parity": parity_binding,
            "historical_stateful_address_trace": address_trace_binding,
            "historical_stateful_session": source_session_binding,
            "sealed_pre_ple_state": base_binding,
            "sealed_source_ple_injection": source_injection_binding,
            "reference_ple_injection": reference_injection_binding,
        },
        "source_formula": {
            "implementation_scope": ((control.get("reference_contract") or {}).get("implementation_scope")),
            "formula": ((control.get("reference_contract") or {}).get("formula")),
            "evaluated_path": "exact source BF16 support tensors in F32, zero-history independent one-token contexts",
            "source_lookup_tensor_full_materializations": 0,
            "source_model_loads": 0,
        },
        "source_support": {
            "payload_bytes_materialized": support_bytes,
            "tensors": support.bindings,
        },
        "address_inputs": {
            "sealed_bos_control": {
                "token_ids": [source_token_id],
                "trace": source_trace,
                "active_lookup": source_lookup,
            },
            "historical_stateful_address_sequence": {
                "token_ids": accepted_tokens,
                "token_source": "segments from a sealed source-bound address-trace receipt",
                "trace": accepted_trace,
                "active_lookup": accepted_lookup,
                "historical_session_provenance": {
                    "path": address_sequence.get("path"),
                    "sha256": address_sequence.get("sha256"),
                    "recorded_status": address_sequence.get("status"),
                    "seal_format": source_session_binding["seal_format"],
                    "integrity_boundary": (
                        "The source stateful-session receipt validates under the declared compact UTF-8 "
                        "native receipt format, and its exact bytes/token sequence bind this address trace. "
                        "This screen still uses that session for address provenance only."
                    ),
                },
                "state_boundary": (
                    "These are real source hash addresses from a seal-verified accepted session, but their "
                    "source pre-layer states are not available; every address is probed around the sealed "
                    "BOS state instead."
                ),
            },
        },
        "context_bank": context_bank,
        "source_control_batch_check": {
            "source_formula_control_metrics": source_batch_metrics,
            "reference_formula_oracle_metrics": reference_batch_metrics,
            "required_max_relative_l2": BASELINE_RELATIVE_L2_MAX,
            "required_max_abs": BASELINE_MAX_ABS_MAX,
            "passed": True,
        },
        "address_context_response": {
            "address_count": len(address_results),
            "address_results": address_results,
            "cross_address_delta_spectrum": cross_response,
            "interpretation": (
                "This is the response spectrum over one synthetic low-dimensional context plane per "
                "real address. Its rank is bounded by the declared probe directions and cannot infer "
                "the rank of the full source-state distribution."
            ),
        },
        "execution": {
            "backend": "numpy_f32_source_formula_tangent_screen",
            "context_evaluations": len(address_results) * int(contexts.shape[0]),
            "real_hashed_addresses": len(address_results),
            "source_lookup_row_range_reads": (
                int(source_lookup["unique_source_rows"]) + int(accepted_lookup["unique_source_rows"])
            ),
            "source_support_payload_materializations": len(support.bindings),
            "source_support_payload_bytes_materialized": support_bytes,
            "elapsed_ns": time.perf_counter_ns() - started_ns,
        },
        "claim_boundary": (
            "This is a source-bound local geometry screen, not a representation result. It verifies that "
            "a zero-history batch form reproduces the sealed PLE control, then measures exact source-formula "
            "response across real hashed addresses under deterministic synthetic perturbations of one sealed "
            "BOS pre-layer state. A seal-verified historical accepted session supplies addresses only; its "
            "source pre-layer states are not exported, so no source-trajectory context claim follows. "
            "It does not establish source-trajectory coverage, persistent PLE state behavior, a context-conditioned "
            "generator, direct execution, complete EBPW, capability, or TPS."
        ),
        "promotion_allowed": False,
        "next": (
            "If the local response geometry justifies it, add a narrowly scoped real source multi-context "
            "state-bank contract before training or evaluating an (address, context, persistent PLE state) to "
            "contribution function. Do not resume fixed table-codec sweeps; any later generator, latent basis, "
            "or Nova replacement must bill its complete persistent representation and pass increasing functional "
            "and capability gates."
        ),
    }
    document["seal_sha256"] = _canonical_sha256(document)
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": document["status"],
                "addresses": len(address_results),
                "contexts_per_address": int(contexts.shape[0]),
                "source_control_relative_l2": source_batch_metrics["relative_l2"],
                "out": str(output),
                "seal": document["seal_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
