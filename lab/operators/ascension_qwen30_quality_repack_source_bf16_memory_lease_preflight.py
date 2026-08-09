"""Strict no-load memory/lease preflight for Qwen30's source-BF16 oracle.

This is deliberately *not* a lease grant.  It reads system counters and the
sealed three-way source-oracle contract, then fails closed unless the present
reclaimable-memory lower bound can hold the sealed source weights plus a
predeclared runtime reserve.  It never opens source weight payloads, creates a
Metal context, or starts a model.  A later outer controller must repeat this
check immediately before any approved source-teacher load and record eviction.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "source-bf16-three-way-memory-preflight/receipts"

SCHEMA = "hawking.ascension.qwen30_hq30gr2_source_bf16_memory_lease_preflight.v1"
READY_STATUS = "PREPARED_STRICT_SOURCE_BF16_MEMORY_LEASE_PREFLIGHT_NO_LEASE_GRANTED"
BLOCKED_STATUS = "BLOCKED_STRICT_SOURCE_BF16_MEMORY_LEASE_PREFLIGHT_INSUFFICIENT_RECLAIMABLE_HEADROOM"
CONTRACT_SCHEMA = oracle_contract.SCHEMA
CONTRACT_STATUS = oracle_contract.STATUS

GIB = 1024**3
MIN_RUNTIME_RESERVE_BYTES = 8 * GIB


class SourceMemoryPreflightError(RuntimeError):
    """The source-BF16 oracle cannot safely receive a future lease."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        value = json.loads(path.read_text(encoding="utf-8"))
        checked = verify(value, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise SourceMemoryPreflightError(f"{label} is absent or invalid: {exc}") from exc
    if not isinstance(checked, Mapping):
        raise SourceMemoryPreflightError(f"{label} is not an object")
    return dict(checked)


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceMemoryPreflightError(f"{label} must be an object")
    return dict(value)


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SourceMemoryPreflightError(f"{label} must be an integer >= {minimum}")
    return value


def _command(*args: str) -> str:
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=2.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SourceMemoryPreflightError(f"system counter command {' '.join(args)} failed: {exc}") from exc
    if result.returncode != 0:
        raise SourceMemoryPreflightError(f"system counter command {' '.join(args)} failed with {result.returncode}")
    return result.stdout


def _parse_vm_stat(text: str, *, page_size: int) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        page = re.match(r"^Pages ([A-Za-z ]+):\s+(\d+)\.?$", line)
        if page:
            name = page.group(1).strip().lower()
            if name == "occupied by compressor":
                name = "compressor"
            values[name] = int(page.group(2))
            continue
        swapouts = re.match(r"^Swapouts:\s+(\d+)\.?$", line)
        if swapouts:
            values["swapouts"] = int(swapouts.group(1))
    required = ("free", "inactive", "speculative", "active", "wired down", "compressor")
    missing = [name for name in required if name not in values]
    if missing:
        raise SourceMemoryPreflightError(f"vm_stat lacks required counters: {', '.join(missing)}")
    reclaimable_pages = values["free"] + values["inactive"] + values["speculative"]
    return {
        "page_size": page_size,
        "free_pages": values["free"],
        "inactive_pages": values["inactive"],
        "speculative_pages": values["speculative"],
        "active_pages": values["active"],
        "wired_pages": values["wired down"],
        "compressor_pages": values["compressor"],
        "swapouts_pages": values.get("swapouts", 0),
        "reclaimable_pages": reclaimable_pages,
        "reclaimable_bytes": reclaimable_pages * page_size,
        "free_bytes": values["free"] * page_size,
        "active_bytes": values["active"] * page_size,
        "wired_bytes": values["wired down"] * page_size,
        "compressor_bytes": values["compressor"] * page_size,
    }


def _parse_swapusage(text: str) -> dict[str, int]:
    match = re.search(r"total\s+=\s+([0-9.]+)([MG])\s+used\s+=\s+([0-9.]+)([MG])", text)
    if not match:
        raise SourceMemoryPreflightError("vm.swapusage output is unparseable")
    multiplier = {"M": 1024**2, "G": 1024**3}
    return {
        "total_bytes": int(float(match.group(1)) * multiplier[match.group(2)]),
        "used_bytes": int(float(match.group(3)) * multiplier[match.group(4)]),
    }


def memory_snapshot() -> dict[str, Any]:
    page_size = int(_command("/usr/sbin/sysctl", "-n", "hw.pagesize").strip())
    physical_bytes = int(_command("/usr/sbin/sysctl", "-n", "hw.memsize").strip())
    vm = _parse_vm_stat(_command("/usr/bin/vm_stat"), page_size=page_size)
    swap = _parse_swapusage(_command("/usr/sbin/sysctl", "vm.swapusage"))
    return {"physical_memory_bytes": physical_bytes, "vm_stat": vm, "swap": swap}


def assess_headroom(snapshot: Mapping[str, Any], *, source_weight_bytes: int) -> dict[str, Any]:
    physical = _integer(snapshot.get("physical_memory_bytes"), label="physical memory", minimum=1)
    vm = _object(snapshot.get("vm_stat"), label="vm snapshot")
    swap = _object(snapshot.get("swap"), label="swap snapshot")
    reclaimable = _integer(vm.get("reclaimable_bytes"), label="reclaimable bytes")
    swap_used = _integer(swap.get("used_bytes"), label="swap used bytes")
    required = source_weight_bytes + MIN_RUNTIME_RESERVE_BYTES
    deficit = max(0, required - reclaimable)
    status = READY_STATUS if deficit == 0 and swap_used == 0 else BLOCKED_STATUS
    return {
        "status": status,
        "source_weight_bytes": source_weight_bytes,
        "predeclared_minimum_source_runtime_reserve_bytes": MIN_RUNTIME_RESERVE_BYTES,
        "minimum_reclaimable_bytes_required_before_source_load": required,
        "measured_reclaimable_bytes": reclaimable,
        "measured_reclaimable_deficit_bytes": deficit,
        "measured_reclaimable_headroom_bytes": max(0, reclaimable - required),
        "measured_swap_used_bytes": swap_used,
        "swap_must_remain_zero_before_and_after_future_capture": True,
        "physical_memory_bytes": physical,
        "minimum_static_requirement_fraction_of_physical_memory": required / physical,
        "active_wired_compressed_and_existing_server_or_metal_residency_not_inferred_from_reclaimable_counter": True,
        "future_outer_capture_must_add_backend_specific_allocator_kv_activation_and_live_model_residency_measurement": True,
        "lease_granted": False,
    }


def build_preflight(*, contract_path: Path, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    contract = _sealed(contract_path, label="source-BF16 three-way contract")
    if contract.get("schema") != CONTRACT_SCHEMA or contract.get("status") != CONTRACT_STATUS:
        raise SourceMemoryPreflightError("source-BF16 three-way contract schema/status drifted")
    resource = _object(contract.get("resource_and_capture_requirements"), label="source-BF16 contract resource requirements")
    source_weight_bytes = _integer(resource.get("source_weights_static_lower_bound_bytes"), label="contract source weights", minimum=1)
    if resource.get("source_model_has_not_been_loaded_by_this_preflight") is not True:
        raise SourceMemoryPreflightError("source-BF16 contract must state that no model was loaded")
    assessment = assess_headroom(snapshot, source_weight_bytes=source_weight_bytes)
    return seal(
        {
            "schema": SCHEMA,
            "status": assessment["status"],
            "recorded_at": _utc_now(),
            "source_bf16_three_way_contract": {
                "path": str(contract_path.resolve()),
                "seal_sha256": contract.get("seal_sha256"),
            },
            "measured_system_snapshot": dict(snapshot),
            "headroom_assessment": assessment,
            "future_lease_conditions": {
                "this_record_is_not_a_lease": True,
                "must_repeat_immediately_before_source_load": True,
                "must_hold_qwen80_model_stage": True,
                "must_keep_qwen30_server_idle_not_stopped": True,
                "must_retain_server_and_process_tree_snapshot": True,
                "must_record_pre_post_swapouts_and_swap_usage": True,
                "must_record_source_model_backend_and_exact_resident_allocation": True,
                "must_evict_source_weights_after_terminal_capture": True,
                "must_refuse_if_reclaimable_threshold_or_zero_swap_condition_fails": True,
                "no_automatic_retry": True,
            },
            "claim_boundary": {
                "cpu_only_system_counter_preflight": True,
                "does_not_open_source_weight_payloads": True,
                "does_not_load_a_source_model": True,
                "does_not_create_metal_or_mps_context": True,
                "does_not_take_or_grant_gpu_or_memory_lease": True,
                "does_not_touch_qwen30_server_watcher_adapter_or_hcli": True,
                "does_not_claim_a_source_oracle_or_candidate_quality_result": True,
            },
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_preflight(contract_path=args.contract, snapshot=memory_snapshot())
        if args.output is None:
            output = DEFAULT_OUTPUT_ROOT / f"QWEN30_HQ30GR2_SOURCE_BF16_MEMORY_PREFLIGHT_{result['source_bf16_three_way_contract']['seal_sha256']}.json"
        else:
            output = args.output
        _atomic_json(output, result)
    except SourceMemoryPreflightError as exc:
        print(f"Q30 source-BF16 memory preflight refused: {exc}")
        return 2
    print(json.dumps({"output": str(output.resolve()), "status": result["status"], "seal_sha256": result["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
