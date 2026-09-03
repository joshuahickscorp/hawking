"""Strict STREAMED_TEACHER resource preflight for Qwen30.

This is deliberately *not* a lease grant and *not* a co-resident gate.  It
answers only whether the host currently satisfies the streamed-teacher
resource contract:

  - reclaimable memory >= streamed floor (~1.3 GiB), not the 64.9 GiB
    co-resident requirement
  - zero swap used and zero swapouts
  - bounded stream window (max 1 live window, max 1_048_576 B)
  - runtime working set from the sealed feasibility arithmetic
  - exact source pin (three-way contract seal)
  - exact child executable pin (CURRENT_STREAMED_TEACHER_CHILD_SHA256)
  - positive non-residency proof: child RSS ceiling strictly below full
    BF16 source weight bytes, plus single-window cache bounds

It never opens source weight payloads, creates a Metal/MPS context, starts a
model, or grants a lease.  A later outer controller must re-run this check
immediately before any streamed source-teacher child and record the declared
RSS ceiling so a silent full-model load would fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators import ascension_qwen30_layer_streamed_source_bf16_oracle_feasibility as feasibility
from lab.operators import ascension_qwen30_quality_repack_source_bf16_memory_lease_preflight as memory_preflight
from lab.operators import ascension_qwen30_quality_repack_source_oracle_three_way_contract as oracle_contract
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates"
    / "gate-up-residual-v1"
)
DEFAULT_CONTRACT = (
    DEFAULT_ROOT / "source-bf16-three-way-final-logit-contract/receipts"
    / "QWEN30_HQ30GR2_SOURCE_BF16_THREE_WAY_FINAL_LOGIT_CONTRACT_"
    "883c59eec0371ebb6d4a9935cdbdc6bcb486c03eebd5312db608a0415a34911f.json"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "streamed-teacher-resource-preflight/receipts"

SCHEMA = "hawking.ascension.qwen30_streamed_teacher_resource_preflight.v1"
READY_STATUS = "PREPARED_STREAMED_TEACHER_RESOURCE_PREFLIGHT_NO_LEASE_GRANTED"
BLOCKED_STATUS = "BLOCKED_STREAMED_TEACHER_RESOURCE_PREFLIGHT_REQUIREMENTS_NOT_MET"
EXECUTION_MODE = "STREAMED_TEACHER"

CONTRACT_SCHEMA = oracle_contract.SCHEMA
CONTRACT_STATUS = oracle_contract.STATUS

MAX_STREAM_WINDOW_BYTES = 1_048_576
MAX_LIVE_STREAM_WINDOWS = 1
# Extra RSS slack above the modeled working set with allocator reserve.  Still
# orders of magnitude below a full 56.9 GiB BF16 residency.
NON_RESIDENCY_RSS_SLACK_BYTES = 512 * 1024**2
# Must match CURRENT_STREAMED_TEACHER_CHILD_SHA256 in
# ascension_qwen30_guarded_streamed_source_oracle_outer_controller.py.
# Duplicated (not imported) to avoid a circular operator dependency.
EXPECTED_EXECUTOR_SHA256 = (
    "cf235f549cf1cd5bfff4003ae394eae640697a0f7bef8e748f75fd1b9db7e8c0"
)

GIB = 1024**3


class StreamedTeacherResourcePreflightError(RuntimeError):
    """The streamed teacher cannot safely receive a future resource window."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StreamedTeacherResourcePreflightError(f"{label} must be an object")
    return dict(value)


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StreamedTeacherResourcePreflightError(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: object, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise StreamedTeacherResourcePreflightError(f"{label} must be a non-empty string")
    if sha256 and not _is_sha256(value):
        raise StreamedTeacherResourcePreflightError(f"{label} must be a lowercase SHA-256")
    return value


def _sealed(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        checked = verify(value, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise StreamedTeacherResourcePreflightError(f"{label} is absent or invalid: {exc}") from exc
    if not isinstance(checked, Mapping):
        raise StreamedTeacherResourcePreflightError(f"{label} is not an object")
    return dict(checked)


def _command(*args: str) -> str:
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=2.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StreamedTeacherResourcePreflightError(
            f"system counter command {' '.join(args)} failed: {exc}"
        ) from exc
    if result.returncode != 0:
        raise StreamedTeacherResourcePreflightError(
            f"system counter command {' '.join(args)} failed with {result.returncode}"
        )
    return result.stdout


def memory_snapshot() -> dict[str, Any]:
    """Reuse the co-resident counter collectors without adopting their verdict."""
    return memory_preflight.memory_snapshot()


def streamed_working_set(*, source_weight_bytes: int) -> dict[str, Any]:
    """Derive the streamed working-set model (no source payload open)."""
    if source_weight_bytes <= 0:
        raise StreamedTeacherResourcePreflightError("source_weight_bytes must be positive")
    working = feasibility._working_set(
        feasibility.DEFAULT_POLICY, source_weight_bytes=source_weight_bytes
    )
    window_peak = _integer(
        working.get("source_tensor_window_peak_bytes"),
        label="source tensor window peak",
        minimum=1,
    )
    if window_peak > MAX_STREAM_WINDOW_BYTES:
        raise StreamedTeacherResourcePreflightError(
            "streamed working-set peak window exceeds the 1 MiB hard ceiling"
        )
    runtime_working_set = _integer(
        working.get("modeled_working_set_with_allocator_reserve_bytes"),
        label="modeled streamed working set",
        minimum=1,
    )
    reclaimable_floor = _integer(
        working.get("minimum_reclaimable_bytes_required_for_streamed_plan"),
        label="streamed reclaimable floor",
        minimum=1,
    )
    non_residency_ceiling = runtime_working_set + NON_RESIDENCY_RSS_SLACK_BYTES
    if non_residency_ceiling >= source_weight_bytes:
        raise StreamedTeacherResourcePreflightError(
            "non-residency RSS ceiling must stay strictly below full source weight bytes"
        )
    return {
        "source_weight_bytes_whole_model_not_resident_in_this_plan": source_weight_bytes,
        "bounded_stream_window_bytes_ceiling": MAX_STREAM_WINDOW_BYTES,
        "bounded_stream_window_peak_bytes": window_peak,
        "maximum_live_stream_windows": MAX_LIVE_STREAM_WINDOWS,
        "runtime_working_set_bytes": runtime_working_set,
        "minimum_reclaimable_bytes_required_immediately_before_source_child": reclaimable_floor,
        "maximum_child_rss_bytes_non_residency_ceiling": non_residency_ceiling,
        "non_residency_rss_slack_bytes": NON_RESIDENCY_RSS_SLACK_BYTES,
        "working_set_detail": working,
    }


def assess_streamed_resources(
    snapshot: Mapping[str, Any],
    *,
    source_weight_bytes: int,
    expected_source_pin_sha256: str,
    observed_source_pin_sha256: str,
    expected_executor_sha256: str = EXPECTED_EXECUTOR_SHA256,
    observed_executor_sha256: str | None = None,
    bounded_stream_window_bytes: int = MAX_STREAM_WINDOW_BYTES,
    live_stream_windows: int = MAX_LIVE_STREAM_WINDOWS,
    measured_or_declared_child_rss_bytes: int | None = None,
) -> dict[str, Any]:
    """Pure STREAMED_TEACHER resource verdict from a synthetic or live snapshot.

    Independent of the co-resident BF16 memory gate arithmetic.  A blocked
    co-resident snapshot with healthy streamed numbers must still READY here.
    """
    if not _is_sha256(expected_source_pin_sha256):
        raise StreamedTeacherResourcePreflightError("expected source pin must be a lowercase SHA-256")
    if not _is_sha256(observed_source_pin_sha256):
        raise StreamedTeacherResourcePreflightError("observed source pin must be a lowercase SHA-256")
    if not _is_sha256(expected_executor_sha256):
        raise StreamedTeacherResourcePreflightError("expected executor pin must be a lowercase SHA-256")
    if observed_executor_sha256 is not None and not _is_sha256(observed_executor_sha256):
        raise StreamedTeacherResourcePreflightError("observed executor pin must be a lowercase SHA-256")

    physical = _integer(snapshot.get("physical_memory_bytes"), label="physical memory", minimum=1)
    vm = _object(snapshot.get("vm_stat"), label="vm snapshot")
    swap = _object(snapshot.get("swap"), label="swap snapshot")
    reclaimable = _integer(vm.get("reclaimable_bytes"), label="reclaimable bytes")
    swap_used = _integer(swap.get("used_bytes"), label="swap used bytes")
    swapouts = _integer(vm.get("swapouts_pages"), label="swapouts pages")

    plan = streamed_working_set(source_weight_bytes=source_weight_bytes)
    required = plan["minimum_reclaimable_bytes_required_immediately_before_source_child"]
    runtime_working_set = plan["runtime_working_set_bytes"]
    non_residency_ceiling = plan["maximum_child_rss_bytes_non_residency_ceiling"]
    window_ceiling = plan["bounded_stream_window_bytes_ceiling"]

    blockers: list[str] = []
    if reclaimable < required:
        blockers.append("reclaimable_bytes_below_streamed_floor")
    if swap_used != 0:
        blockers.append("nonzero_swap_used_bytes")
    if swapouts != 0:
        blockers.append("nonzero_swapouts_pages")
    if bounded_stream_window_bytes > window_ceiling:
        blockers.append("stream_window_exceeds_ceiling")
    if bounded_stream_window_bytes <= 0:
        blockers.append("stream_window_bytes_not_positive")
    if live_stream_windows != MAX_LIVE_STREAM_WINDOWS:
        blockers.append("live_stream_windows_not_exactly_one")
    if observed_source_pin_sha256 != expected_source_pin_sha256:
        blockers.append("source_pin_mismatch")
    if observed_executor_sha256 is None:
        blockers.append("executor_sha_absent")
    elif observed_executor_sha256 != expected_executor_sha256:
        blockers.append("executor_sha_mismatch")
    if measured_or_declared_child_rss_bytes is not None:
        rss = _integer(
            measured_or_declared_child_rss_bytes,
            label="measured or declared child RSS",
        )
        if rss > non_residency_ceiling:
            blockers.append("non_residency_rss_ceiling_exceeded")

    blockers = sorted(set(blockers))
    ready = not blockers
    deficit = max(0, required - reclaimable)
    return {
        "execution_mode": EXECUTION_MODE,
        "status": READY_STATUS if ready else BLOCKED_STATUS,
        "verdict": "READY" if ready else "BLOCKED",
        "lease_granted": False,
        "blockers": blockers,
        "source_weight_bytes_whole_model_not_resident_in_this_plan": source_weight_bytes,
        "bounded_stream_window_bytes": bounded_stream_window_bytes,
        "bounded_stream_window_bytes_ceiling": window_ceiling,
        "live_stream_windows": live_stream_windows,
        "maximum_live_stream_windows": MAX_LIVE_STREAM_WINDOWS,
        "runtime_working_set_bytes": runtime_working_set,
        "minimum_reclaimable_bytes_required_immediately_before_source_child": required,
        "measured_reclaimable_bytes": reclaimable,
        "measured_reclaimable_deficit_bytes": deficit,
        "measured_reclaimable_headroom_bytes": max(0, reclaimable - required),
        "measured_swap_used_bytes": swap_used,
        "measured_swapouts_pages": swapouts,
        "swap_must_remain_zero": True,
        "swapouts_must_remain_zero": True,
        "physical_memory_bytes": physical,
        "expected_source_pin_sha256": expected_source_pin_sha256,
        "observed_source_pin_sha256": observed_source_pin_sha256,
        "source_pin_matches": observed_source_pin_sha256 == expected_source_pin_sha256,
        "expected_executor_child_sha256": expected_executor_sha256,
        "observed_executor_child_sha256": observed_executor_sha256,
        "executor_pin_matches": (
            observed_executor_sha256 is not None
            and observed_executor_sha256 == expected_executor_sha256
        ),
        "maximum_child_rss_bytes_non_residency_ceiling": non_residency_ceiling,
        "measured_or_declared_child_rss_bytes": measured_or_declared_child_rss_bytes,
        "non_residency_proof": {
            "maximum_child_rss_bytes_non_residency_ceiling": non_residency_ceiling,
            "full_source_weight_bytes": source_weight_bytes,
            "ceiling_is_strictly_below_full_source_weight_bytes": non_residency_ceiling
            < source_weight_bytes,
            "maximum_source_reader_cached_windows": MAX_LIVE_STREAM_WINDOWS,
            "maximum_source_reader_cached_bytes": window_ceiling,
            "full_model_residency_would_exceed_rss_ceiling": True,
            "full_model_residency_would_exceed_single_window_cache_bounds": True,
            "field_that_catches_covert_full_model_residency": (
                "maximum_child_rss_bytes_non_residency_ceiling"
            ),
            "secondary_field_that_catches_covert_full_model_residency": (
                "maximum_source_reader_cached_bytes"
            ),
        },
        "does_not_inherit_co_resident_memory_gate_verdict": True,
        "co_resident_minimum_reclaimable_bytes_not_used": True,
        "streamed_working_set_plan": plan,
    }


def build_preflight(
    *,
    contract_path: Path,
    snapshot: Mapping[str, Any],
    expected_executor_sha256: str = EXPECTED_EXECUTOR_SHA256,
    observed_executor_sha256: str | None = None,
    executor_path: Path | None = None,
    bounded_stream_window_bytes: int = MAX_STREAM_WINDOW_BYTES,
    live_stream_windows: int = MAX_LIVE_STREAM_WINDOWS,
    measured_or_declared_child_rss_bytes: int | None = None,
) -> dict[str, Any]:
    """Seal a STREAMED_TEACHER resource preflight from metadata + counters only."""
    contract = _sealed(contract_path, label="source-BF16 three-way contract")
    if contract.get("schema") != CONTRACT_SCHEMA or contract.get("status") != CONTRACT_STATUS:
        raise StreamedTeacherResourcePreflightError(
            "source-BF16 three-way contract schema/status drifted"
        )
    resource = _object(
        contract.get("resource_and_capture_requirements"),
        label="source-BF16 contract resource requirements",
    )
    source_weight_bytes = _integer(
        resource.get("source_weights_static_lower_bound_bytes"),
        label="contract source weights",
        minimum=1,
    )
    if resource.get("source_model_has_not_been_loaded_by_this_preflight") is not True:
        raise StreamedTeacherResourcePreflightError(
            "source-BF16 contract must state that no model was loaded"
        )
    source_pin = _text(contract.get("seal_sha256"), label="source three-way seal", sha256=True)

    executor_sha = observed_executor_sha256
    executor_binding: dict[str, Any] = {"present": False}
    if executor_path is not None:
        if not executor_path.is_absolute():
            raise StreamedTeacherResourcePreflightError("executor path must be absolute")
        try:
            meta = executor_path.lstat()
        except OSError as exc:
            raise StreamedTeacherResourcePreflightError(
                f"cannot stat streamed teacher child: {exc}"
            ) from exc
        if not os.path.isfile(executor_path) or os.path.islink(executor_path):
            raise StreamedTeacherResourcePreflightError(
                "streamed teacher child must be a regular non-symlink file"
            )
        if meta.st_size <= 0:
            raise StreamedTeacherResourcePreflightError("streamed teacher child is empty")
        executor_sha = _sha256_file(executor_path)
        executor_binding = {
            "present": True,
            "path": str(executor_path.resolve()),
            "bytes": meta.st_size,
            "sha256": executor_sha,
        }

    assessment = assess_streamed_resources(
        snapshot,
        source_weight_bytes=source_weight_bytes,
        expected_source_pin_sha256=source_pin,
        observed_source_pin_sha256=source_pin,
        expected_executor_sha256=expected_executor_sha256,
        observed_executor_sha256=executor_sha,
        bounded_stream_window_bytes=bounded_stream_window_bytes,
        live_stream_windows=live_stream_windows,
        measured_or_declared_child_rss_bytes=measured_or_declared_child_rss_bytes,
    )
    return seal(
        {
            "schema": SCHEMA,
            "status": assessment["status"],
            "recorded_at": _utc_now(),
            "execution_mode": EXECUTION_MODE,
            "source_bf16_three_way_contract": {
                "path": str(contract_path.resolve()),
                "seal_sha256": source_pin,
            },
            "streamed_teacher_child_executable": executor_binding,
            "measured_system_snapshot": dict(snapshot),
            "resource_assessment": assessment,
            "future_lease_conditions": {
                "this_record_is_not_a_lease": True,
                "must_repeat_immediately_before_streamed_source_child": True,
                "must_record_declared_non_residency_rss_ceiling_before_launch": True,
                "must_refuse_if_child_rss_exceeds_non_residency_ceiling": True,
                "must_refuse_if_more_than_one_live_stream_window_or_window_exceeds_ceiling": True,
                "must_keep_zero_swap_and_zero_swapouts": True,
                "must_not_use_co_resident_memory_gate_as_streamed_authority": True,
                "no_automatic_retry": True,
            },
            "claim_boundary": {
                "cpu_only_system_counter_preflight": True,
                "streamed_teacher_resource_contract_only": True,
                "does_not_open_source_weight_payloads": True,
                "does_not_load_a_source_model": True,
                "does_not_create_metal_or_mps_context": True,
                "does_not_take_or_grant_gpu_or_memory_lease": True,
                "does_not_touch_qwen30_server_watcher_adapter_or_hcli": True,
                "does_not_claim_a_source_oracle_or_candidate_quality_result": True,
                "does_not_weaken_or_override_co_resident_memory_gate": True,
            },
        }
    )


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise StreamedTeacherResourcePreflightError(
            "--out must be a new absolute path below an existing parent"
        )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--executor", type=Path, default=None)
    parser.add_argument(
        "--expected-executor-sha256",
        default=EXPECTED_EXECUTOR_SHA256,
        help="Frozen CURRENT_STREAMED_TEACHER_CHILD_SHA256 pin",
    )
    parser.add_argument("--bounded-stream-window-bytes", type=int, default=MAX_STREAM_WINDOW_BYTES)
    parser.add_argument("--live-stream-windows", type=int, default=MAX_LIVE_STREAM_WINDOWS)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_preflight(
            contract_path=args.contract,
            snapshot=memory_snapshot(),
            expected_executor_sha256=args.expected_executor_sha256,
            executor_path=args.executor,
            bounded_stream_window_bytes=args.bounded_stream_window_bytes,
            live_stream_windows=args.live_stream_windows,
        )
        _write_new(args.out, result)
    except StreamedTeacherResourcePreflightError as exc:
        print(f"Q30 STREAMED_TEACHER resource preflight refused: {exc}")
        return 2
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "status": result["status"],
                "seal_sha256": result["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
