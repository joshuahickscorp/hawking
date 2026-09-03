"""Prepare, but never execute, the HQ30GR2 raw-final-logit successor.

The first all-layer HQ30GR2 capture is deliberately immutable: it retained
full-vector hashes and bounded top-k witnesses, not the F32LE payloads needed
to calculate a source/control/candidate distance.  This module makes the
missing retention boundary explicit before anybody is allowed to load the
source teacher or take another native Metal lease.

It only reads sealed JSON and system/preflight evidence.  It does not open a
source weight payload, create a Metal/MPS context, run a model, start an
endpoint, or grant a resource lease.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators import ascension_qwen30_quality_repack_source_bf16_memory_lease_preflight as memory_preflight
from lab.operators import ascension_qwen30_quality_repack_source_oracle_three_way_contract as oracle_contract
from lab.operators import ascension_qwen30_streamed_teacher_resource_preflight as streamed_resource
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates"
    / "gate-up-residual-v1"
)
DEFAULT_THREE_WAY_CONTRACT = (
    DEFAULT_ROOT
    / "source-bf16-three-way-final-logit-contract/receipts"
    / "QWEN30_HQ30GR2_SOURCE_BF16_THREE_WAY_FINAL_LOGIT_CONTRACT_"
    "883c59eec0371ebb6d4a9935cdbdc6bcb486c03eebd5312db608a0415a34911f.json"
)
DEFAULT_MEMORY_PREFLIGHT = (
    DEFAULT_ROOT
    / "source-bf16-three-way-memory-preflight/receipts"
    / "QWEN30_HQ30GR2_SOURCE_BF16_MEMORY_PREFLIGHT_"
    "efdacf5952583dc03d2aee37b73a0af284f2d865a3e36b9739b16150efe3f726.json"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "raw-final-logit-retention-successor/receipts"

SCHEMA = "hawking.ascension.qwen30_hq30gr2_raw_final_logit_retention_successor.v1"
STATUS = "PREPARED_RAW_FINAL_LOGIT_RETENTION_SUCCESSOR_NOT_RUN"
THREE_WAY_SCHEMA = oracle_contract.SCHEMA
THREE_WAY_STATUS = oracle_contract.STATUS
MEMORY_SCHEMA = memory_preflight.SCHEMA
MEMORY_STATUSES = {memory_preflight.READY_STATUS, memory_preflight.BLOCKED_STATUS}

NATIVE_MODE = "metal-diagnostic-retain-raw-final-logits"
NATIVE_RAW_VECTOR_MODELS = ("scalar_control", "hq30gr2_candidate")
ENDPOINTS = oracle_contract.ENDPOINTS
SOURCE_MODEL = "source_bf16"
VOCAB_ROWS = oracle_contract.VOCAB_ROWS
F32LE_BYTES_PER_VECTOR = oracle_contract.F32_BYTES * VOCAB_ROWS


class RawFinalLogitRetentionContractError(RuntimeError):
    """The future raw-vector capture lacks an immutable, bounded contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sealed(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        checked = verify(raw, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise RawFinalLogitRetentionContractError(f"{label} is absent or invalid: {exc}") from exc
    if not isinstance(checked, Mapping):
        raise RawFinalLogitRetentionContractError(f"{label} is not an object")
    return dict(checked)


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RawFinalLogitRetentionContractError(f"{label} must be an object")
    return dict(value)


def _text(value: object, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise RawFinalLogitRetentionContractError(f"{label} must be a non-empty string")
    if sha256 and (len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
        raise RawFinalLogitRetentionContractError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RawFinalLogitRetentionContractError(f"{label} must be an integer >= {minimum}")
    return value


def _evidence(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        stat_result = path.stat()
    except OSError as exc:
        raise RawFinalLogitRetentionContractError(f"cannot stat {path}: {exc}") from exc
    return {
        "path": str(path.resolve()),
        "bytes": stat_result.st_size,
        "sha256": _sha256_file(path),
        "seal_sha256": _text(document.get("seal_sha256"), label="evidence seal", sha256=True),
    }


def _assert_schema_status(document: Mapping[str, Any], *, schema: str, statuses: set[str], label: str) -> None:
    if document.get("schema") != schema or document.get("status") not in statuses:
        expected = ", ".join(sorted(statuses))
        raise RawFinalLogitRetentionContractError(f"{label} schema/status must be {schema!r}/{expected}")


def raw_vector_plan(*, vocab_rows: int = VOCAB_ROWS) -> dict[str, Any]:
    """Return the immutable six-vector naming and retention protocol."""
    if vocab_rows <= 0:
        raise RawFinalLogitRetentionContractError("vocab_rows must be positive")
    bytes_per_vector = vocab_rows * oracle_contract.F32_BYTES
    source_payloads = [f"{SOURCE_MODEL}_{endpoint}_logits.f32le" for endpoint in ENDPOINTS]
    native_payloads = [
        f"{model}_{endpoint}_logits.f32le"
        for model in NATIVE_RAW_VECTOR_MODELS
        for endpoint in ENDPOINTS
    ]
    return {
        "root_must_be_fresh_and_owned_by_one_outer_capture": True,
        "dtype": "f32le",
        "vocab_rows": vocab_rows,
        "bytes_per_vector": bytes_per_vector,
        "source_teacher_payloads": source_payloads,
        "native_successor_payloads": native_payloads,
        "required_payloads": source_payloads + native_payloads,
        "required_payload_count": 6,
        "required_total_payload_bytes": 6 * bytes_per_vector,
        "all_values_must_be_finite": True,
        "receipt_must_be_written_after_all_six_payloads_and_fsyncs": True,
        "source_and_native_may_run_sequentially_but_payloads_must_remain_immutable": True,
    }


def _expected_native_hashes(three_way: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    future = _object(three_way.get("future_capture"), label="three-way future capture")
    native = _object(future.get("native_control_and_candidate"), label="three-way native capture")
    hashes = _object(
        native.get("control_and_candidate_raw_payload_hashes_must_match_existing_witnesses"),
        label="three-way native replay hashes",
    )
    result: dict[str, dict[str, str]] = {}
    for endpoint in ENDPOINTS:
        row = _object(hashes.get(endpoint), label=f"three-way replay hashes {endpoint}")
        result[endpoint] = {
            "scalar_control": _text(
                row.get("control_full_f32le_sha256"),
                label=f"scalar control {endpoint} expected hash",
                sha256=True,
            ),
            "hq30gr2_candidate": _text(
                row.get("candidate_full_f32le_sha256"),
                label=f"HQ30GR2 candidate {endpoint} expected hash",
                sha256=True,
            ),
        }
    return result


CO_RESIDENT_TEACHER_MODE = "CO_RESIDENT_TEACHER"
STREAMED_TEACHER_MODE = "STREAMED_TEACHER"


def _snapshot_from_memory_preflight(memory: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the sealed host counters used by both mode gates."""
    measured = memory.get("measured_system_snapshot")
    if isinstance(measured, Mapping) and measured:
        snapshot = dict(measured)
        vm = _object(snapshot.get("vm_stat"), label="memory preflight vm_stat")
        swap = _object(snapshot.get("swap"), label="memory preflight swap")
        if "reclaimable_bytes" not in vm:
            headroom = _object(memory.get("headroom_assessment"), label="memory headroom assessment")
            vm = {
                **vm,
                "reclaimable_bytes": _integer(
                    headroom.get("measured_reclaimable_bytes"),
                    label="measured reclaimable bytes",
                ),
                "swapouts_pages": _integer(vm.get("swapouts_pages", 0), label="swapouts pages"),
            }
            snapshot["vm_stat"] = vm
        if "used_bytes" not in swap:
            headroom = _object(memory.get("headroom_assessment"), label="memory headroom assessment")
            snapshot["swap"] = {
                **swap,
                "used_bytes": _integer(
                    headroom.get("measured_swap_used_bytes"),
                    label="measured swap used bytes",
                ),
            }
        if "physical_memory_bytes" not in snapshot:
            headroom = _object(memory.get("headroom_assessment"), label="memory headroom assessment")
            snapshot["physical_memory_bytes"] = _integer(
                headroom.get("physical_memory_bytes", 96 * 1024**3),
                label="physical memory bytes",
                minimum=1,
            )
        return snapshot
    headroom = _object(memory.get("headroom_assessment"), label="memory headroom assessment")
    return {
        "physical_memory_bytes": _integer(
            headroom.get("physical_memory_bytes", 96 * 1024**3),
            label="physical memory bytes",
            minimum=1,
        ),
        "vm_stat": {
            "reclaimable_bytes": _integer(
                headroom.get("measured_reclaimable_bytes"),
                label="measured reclaimable bytes",
            ),
            "swapouts_pages": _integer(headroom.get("measured_swapouts_pages", 0), label="swapouts"),
        },
        "swap": {
            "used_bytes": _integer(
                headroom.get("measured_swap_used_bytes"),
                label="measured swap used bytes",
            ),
        },
    }


def build_execution_mode_gates(
    *,
    memory: Mapping[str, Any],
    three_way: Mapping[str, Any],
    source_weight_bytes: int,
    observed_executor_sha256: str | None = None,
    expected_executor_sha256: str = streamed_resource.EXPECTED_EXECUTOR_SHA256,
    bounded_stream_window_bytes: int = streamed_resource.MAX_STREAM_WINDOW_BYTES,
    live_stream_windows: int = streamed_resource.MAX_LIVE_STREAM_WINDOWS,
    measured_or_declared_child_rss_bytes: int | None = None,
) -> dict[str, Any]:
    """Build independent per-mode resource verdicts; never cross-inherit status."""
    headroom = _object(memory.get("headroom_assessment"), label="memory headroom assessment")
    co_resident_blocked = memory.get("status") == memory_preflight.BLOCKED_STATUS
    co_resident_required = _integer(
        headroom.get("minimum_reclaimable_bytes_required_before_source_load"),
        label="co-resident required reclaimable bytes",
        minimum=1,
    )
    co_resident_measured = _integer(
        headroom.get("measured_reclaimable_bytes"),
        label="co-resident measured reclaimable bytes",
    )
    co_resident_swap = _integer(
        headroom.get("measured_swap_used_bytes"),
        label="co-resident measured swap used bytes",
    )
    source_pin = _text(three_way.get("seal_sha256"), label="source three-way seal", sha256=True)
    snapshot = _snapshot_from_memory_preflight(memory)
    streamed = streamed_resource.assess_streamed_resources(
        snapshot,
        source_weight_bytes=source_weight_bytes,
        expected_source_pin_sha256=source_pin,
        observed_source_pin_sha256=source_pin,
        expected_executor_sha256=expected_executor_sha256,
        observed_executor_sha256=observed_executor_sha256,
        bounded_stream_window_bytes=bounded_stream_window_bytes,
        live_stream_windows=live_stream_windows,
        measured_or_declared_child_rss_bytes=measured_or_declared_child_rss_bytes,
    )
    return {
        CO_RESIDENT_TEACHER_MODE: {
            "execution_mode": CO_RESIDENT_TEACHER_MODE,
            "verdict": "BLOCKED" if co_resident_blocked else "READY",
            "status": memory.get("status"),
            "teacher_weights_resident_bytes": source_weight_bytes,
            "reserve_bytes": memory_preflight.MIN_RUNTIME_RESERVE_BYTES,
            "minimum_reclaimable_bytes_required": co_resident_required,
            "measured_reclaimable_bytes": co_resident_measured,
            "measured_reclaimable_deficit_bytes": headroom.get("measured_reclaimable_deficit_bytes"),
            "measured_swap_used_bytes": co_resident_swap,
            "zero_swap_required": True,
            "exact_source_pin_sha256": source_pin,
            "source_teacher_capture_is_currently_blocked": co_resident_blocked,
            "does_not_authorize_streamed_teacher": True,
        },
        STREAMED_TEACHER_MODE: {
            "execution_mode": STREAMED_TEACHER_MODE,
            "verdict": streamed["verdict"],
            "status": streamed["status"],
            "bounded_stream_window_bytes": streamed["bounded_stream_window_bytes"],
            "bounded_stream_window_bytes_ceiling": streamed["bounded_stream_window_bytes_ceiling"],
            "live_stream_windows": streamed["live_stream_windows"],
            "maximum_live_stream_windows": streamed["maximum_live_stream_windows"],
            "runtime_working_set_bytes": streamed["runtime_working_set_bytes"],
            "minimum_reclaimable_bytes_required": streamed[
                "minimum_reclaimable_bytes_required_immediately_before_source_child"
            ],
            "measured_reclaimable_bytes": streamed["measured_reclaimable_bytes"],
            "measured_reclaimable_deficit_bytes": streamed["measured_reclaimable_deficit_bytes"],
            "measured_swap_used_bytes": streamed["measured_swap_used_bytes"],
            "measured_swapouts_pages": streamed["measured_swapouts_pages"],
            "zero_swap_required": True,
            "zero_swapouts_required": True,
            "exact_source_pin_sha256": streamed["observed_source_pin_sha256"],
            "source_pin_matches": streamed["source_pin_matches"],
            "exact_executor_child_sha256": streamed["expected_executor_child_sha256"],
            "observed_executor_child_sha256": streamed["observed_executor_child_sha256"],
            "executor_pin_matches": streamed["executor_pin_matches"],
            "maximum_child_rss_bytes_non_residency_ceiling": streamed[
                "maximum_child_rss_bytes_non_residency_ceiling"
            ],
            "measured_or_declared_child_rss_bytes": streamed[
                "measured_or_declared_child_rss_bytes"
            ],
            "non_residency_proof": streamed["non_residency_proof"],
            "blockers": streamed["blockers"],
            "does_not_inherit_co_resident_verdict": True,
            "streamed_teacher_capture_is_currently_blocked": streamed["verdict"] == "BLOCKED",
        },
        "modes_are_independent": True,
        "streamed_authority_must_read_STREAMED_TEACHER_not_co_resident_flag": True,
        "co_resident_authority_must_read_CO_RESIDENT_TEACHER": True,
    }


def build_contract(
    *,
    three_way_contract_path: Path,
    memory_preflight_path: Path,
    observed_executor_sha256: str | None = None,
    expected_executor_sha256: str = streamed_resource.EXPECTED_EXECUTOR_SHA256,
    bounded_stream_window_bytes: int = streamed_resource.MAX_STREAM_WINDOW_BYTES,
    live_stream_windows: int = streamed_resource.MAX_LIVE_STREAM_WINDOWS,
    measured_or_declared_child_rss_bytes: int | None = None,
) -> dict[str, Any]:
    """Build one unrun successor contract from sealed prerequisite evidence."""
    three_way = _sealed(three_way_contract_path, label="source-BF16 three-way contract")
    _assert_schema_status(
        three_way,
        schema=THREE_WAY_SCHEMA,
        statuses={THREE_WAY_STATUS},
        label="source-BF16 three-way contract",
    )
    memory = _sealed(memory_preflight_path, label="strict source-BF16 memory preflight")
    _assert_schema_status(
        memory,
        schema=MEMORY_SCHEMA,
        statuses=MEMORY_STATUSES,
        label="strict source-BF16 memory preflight",
    )
    contract_ref = _object(memory.get("source_bf16_three_way_contract"), label="memory preflight three-way pointer")
    if Path(_text(contract_ref.get("path"), label="memory preflight three-way path")).resolve() != three_way_contract_path.resolve():
        raise RawFinalLogitRetentionContractError("memory preflight names a different source-BF16 three-way contract")
    if contract_ref.get("seal_sha256") != three_way.get("seal_sha256"):
        raise RawFinalLogitRetentionContractError("memory preflight three-way seal differs")

    evidence = _object(three_way.get("evidence"), label="three-way evidence")
    comparison = _object(evidence.get("candidate_local_comparison"), label="three-way comparison evidence")
    old_inner = _object(evidence.get("all_layer_inner_diagnostic"), label="three-way all-layer diagnostic evidence")
    exact_input = _object(three_way.get("exact_input"), label="three-way exact input")
    expected_hashes = _expected_native_hashes(three_way)
    plan = raw_vector_plan()
    headroom = _object(memory.get("headroom_assessment"), label="memory headroom assessment")
    resource = _object(
        three_way.get("resource_and_capture_requirements"),
        label="three-way resource requirements",
    )
    source_weight_bytes = _integer(
        resource.get("source_weights_static_lower_bound_bytes"),
        label="contract source weights",
        minimum=1,
    )
    # CO_RESIDENT arithmetic is owned exclusively by the strict memory preflight.
    co_resident_blocked = memory.get("status") == memory_preflight.BLOCKED_STATUS
    execution_mode_gates = build_execution_mode_gates(
        memory=memory,
        three_way=three_way,
        source_weight_bytes=source_weight_bytes,
        observed_executor_sha256=observed_executor_sha256,
        expected_executor_sha256=expected_executor_sha256,
        bounded_stream_window_bytes=bounded_stream_window_bytes,
        live_stream_windows=live_stream_windows,
        measured_or_declared_child_rss_bytes=measured_or_declared_child_rss_bytes,
    )
    return seal(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "recorded_at": _utc_now(),
            "source_bf16_three_way_contract": _evidence(three_way_contract_path, three_way),
            "strict_source_bf16_memory_preflight": _evidence(memory_preflight_path, memory),
            "replay_binding": {
                "candidate_local_comparison": comparison,
                "immutable_prior_all_layer_inner_diagnostic": old_inner,
                "exact_trace": {
                    "probe_id": _text(exact_input.get("probe_id"), label="three-way probe id"),
                    "source_template_token_count": exact_input.get("source_template_token_count"),
                    "source_template_token_ids_u32le_sha256": _text(
                        exact_input.get("source_template_token_ids_u32le_sha256"),
                        label="three-way token hash",
                        sha256=True,
                    ),
                    "forced_identical_continuation_token_id": exact_input.get("forced_identical_continuation_token_id"),
                },
                "native_raw_hashes_must_replay_prior_98db_witness": expected_hashes,
            },
            "raw_final_logit_retention_successor": {
                "native_mode": NATIVE_MODE,
                "role": "new_non_serving_native_capture_of_four_raw_vectors_only",
                "new_executor_binary_sha256_required_before_lease": True,
                "must_not_reuse_or_mutate_prior_98db_capture": True,
                "must_execute_scalar_control_then_typed_hq30gr2_under_a_new_receipt_last_outer_capture": True,
                "must_replay_all_740_native_all_layer_forwards_and_exact_L0_E0_sparse_interception_witness": True,
                "must_refuse_before_receipt_if_any_native_raw_hash_differs_from_prior_witness": True,
                "does_not_execute_or_simulate_source_bf16": True,
                "does_not_write_the_two_source_teacher_vectors": True,
            },
            "six_vector_retention_contract": plan,
            # Per-mode authority.  STREAMED consumers must read
            # execution_mode_gates.STREAMED_TEACHER; the legacy boolean below
            # is CO_RESIDENT_TEACHER only and must not authorize the streamed
            # child path.
            "execution_mode_gates": execution_mode_gates,
            "source_memory_and_eviction_gate": {
                "current_memory_preflight_status": memory.get("status"),
                "current_reclaimable_bytes": headroom.get("measured_reclaimable_bytes"),
                "current_required_reclaimable_bytes": headroom.get("minimum_reclaimable_bytes_required_before_source_load"),
                "current_reclaimable_deficit_bytes": headroom.get("measured_reclaimable_deficit_bytes"),
                "current_swap_used_bytes": headroom.get("measured_swap_used_bytes"),
                # Legacy field retained as CO_RESIDENT_TEACHER authority only.
                # Streamed path authority is execution_mode_gates.STREAMED_TEACHER.
                "source_teacher_capture_is_currently_blocked": co_resident_blocked,
                "source_teacher_capture_is_currently_blocked_applies_only_to_co_resident_teacher": True,
                "streamed_path_must_use_execution_mode_gates_STREAMED_TEACHER": True,
                "must_repeat_strict_preflight_immediately_before_future_source_load": True,
                "must_record_backend_specific_resident_allocation_before_source_load": True,
                "must_record_pre_post_swap_and_swapouts": True,
                "must_evict_source_weights_and_confirm_release_before_native_capture": True,
                "source_and_native_model_bodies_must_not_be_resident_concurrently": True,
            },
            "future_safe_sequence": [
                "Choose an execution mode.  CO_RESIDENT_TEACHER requires the strict source-BF16 memory preflight READY; STREAMED_TEACHER requires execution_mode_gates.STREAMED_TEACHER READY and must not read the co-resident blocked boolean as authority.",
                "Do not run CO_RESIDENT_TEACHER while the strict source-BF16 preflight is BLOCKED.",
                "After a new independently approved source-memory lease (or streamed resource window), re-run the matching mode preflight and capture only the two source F32LE vectors for the sealed trace; fsync and seal a source-teacher terminal receipt.",
                "Evict the source body (or drop the streamed window cache), confirm zero swap growth and released source residency, and preserve the two source vectors immutable on disk.",
                "Under a distinct new native quiet lease, run the raw-final-logit-retention successor once; it emits four direct-packed raw F32LE vectors and refuses unless all four hashes replay 98db exactly.",
                "Only a six-vector aggregator may read the immutable payloads, calculate the preregistered F64 relative-L2 values at both endpoints, and report a numerical diagnostic. A metric result is never coherence/HCLI/TPS/capability/tournament evidence.",
            ],
            "claim_boundary": {
                "preparation_only": True,
                "does_not_open_source_weight_payloads_or_load_source_model": True,
                "does_not_execute_control_candidate_or_source_model": True,
                "does_not_create_metal_or_mps_context": True,
                "does_not_take_or_grant_gpu_or_memory_lease": True,
                "does_not_touch_live_qwen30_server_watcher_adapter_or_hcli": True,
                "does_not_emit_a_quality_coherence_oracle_result": True,
                "does_not_conflate_co_resident_and_streamed_teacher_resource_contracts": True,
            },
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three-way-contract", type=Path, default=DEFAULT_THREE_WAY_CONTRACT)
    parser.add_argument("--memory-preflight", type=Path, default=DEFAULT_MEMORY_PREFLIGHT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_contract(
            three_way_contract_path=args.three_way_contract,
            memory_preflight_path=args.memory_preflight,
        )
        if args.output is None:
            candidate = _object(result["replay_binding"]["candidate_local_comparison"], label="candidate comparison")
            candidate_seal = _text(candidate.get("seal_sha256"), label="candidate comparison seal", sha256=True)
            output = DEFAULT_OUTPUT_ROOT / f"QWEN30_HQ30GR2_RAW_FINAL_LOGIT_RETENTION_SUCCESSOR_{candidate_seal}.json"
        else:
            output = args.output
        _atomic_json(output, result)
    except RawFinalLogitRetentionContractError as exc:
        print(f"Q30 raw-final-logit retention contract refused: {exc}")
        return 2
    print(json.dumps({"output": str(output.resolve()), "status": result["status"], "seal_sha256": result["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
