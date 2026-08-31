"""Emit honest accelerator telemetry from bounded Flash physical receipts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.persist import atomic_write_json


EXACT_RECEIPT_NAME = "FLASH_NOETIC_EXACT_HYPERCONNECTION_NATIVE.json"
GPU_WORK_LEDGER_NAME = "FLASH_GPU_WORK_LEDGER.json"
TOKEN_CRITICAL_PATH_NAME = "FLASH_TOKEN_CRITICAL_PATH.json"
CUDA_CAPABILITY_GRAPH_NAME = "CUDA_CAPABILITY_GRAPH.json"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _receipt_ref(path: Path, receipt: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "status": receipt.get("status") if isinstance(receipt, Mapping) else None,
        "schema": receipt.get("schema") if isinstance(receipt, Mapping) else None,
    }


def _stage_rows(timing: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []

    def add(name: str, gpu_key: str, median_key: str, *, stage_class: str = "exact_hyperconnection") -> None:
        values = timing.get(gpu_key)
        rows.append({
            "stage": name,
            "stage_class": stage_class,
            "gpu_ns": values,
            "gpu_ns_median": timing.get(median_key),
            "measured_runs": len(values) if isinstance(values, list) else None,
            "device_resident": True,
            "host_roundtrip": False,
            "bytes_touched": None,
            "complete_token_stage": False,
        })

    fixed = (
        ("hc_norm", "hc_norm_gpu_ns", "hc_norm_gpu_ns_median"),
        ("input_mix_down", "input_mix_down_gpu_ns", "input_mix_down_gpu_ns_median"),
        ("read_silu_scaled", "read_silu_scaled_gpu_ns", "read_silu_scaled_gpu_ns_median"),
        ("input_mix_up", "input_mix_up_gpu_ns", "input_mix_up_gpu_ns_median"),
        ("read_mix", "read_mix_gpu_ns", "read_mix_gpu_ns_median"),
        ("shared_gate_up_swiglu", "shared_gate_up_gpu_ns", "shared_gate_up_gpu_ns_median"),
        ("shared_down", "shared_down_gpu_ns", "shared_down_gpu_ns_median"),
        ("shared_scalar_gate", "shared_scalar_gate_gpu_ns", "shared_scalar_gate_gpu_ns_median"),
        ("shared_block_output", "shared_sigmoid_gate_gpu_ns", "shared_sigmoid_gate_gpu_ns_median"),
    )
    for name, gpu_key, median_key in fixed:
        add(name, gpu_key, median_key)
    for row in timing.get("routed_experts") or []:
        if not isinstance(row, Mapping):
            continue
        expert = row.get("expert_index")
        rows.append({
            "stage": f"routed_expert_{expert}_gate_up_swiglu",
            "stage_class": "routed_expert",
            "expert_index": expert,
            "gpu_ns": row.get("gate_up_swiglu_gpu_ns"),
            "gpu_ns_median": row.get("gate_up_swiglu_gpu_ns_median"),
            "measured_runs": len(row.get("gate_up_swiglu_gpu_ns") or []) if isinstance(row.get("gate_up_swiglu_gpu_ns"), list) else None,
            "device_resident": True,
            "host_roundtrip": False,
            "bytes_touched": None,
            "complete_token_stage": False,
        })
        rows.append({
            "stage": f"routed_expert_{expert}_down",
            "stage_class": "routed_expert",
            "expert_index": expert,
            "gpu_ns": row.get("down_projection_gpu_ns"),
            "gpu_ns_median": row.get("down_projection_gpu_ns_median"),
            "measured_runs": len(row.get("down_projection_gpu_ns") or []) if isinstance(row.get("down_projection_gpu_ns"), list) else None,
            "device_resident": True,
            "host_roundtrip": False,
            "bytes_touched": None,
            "complete_token_stage": False,
        })
    add("routed_weighted_sum", "routed_weighted_sum_gpu_ns", "routed_weighted_sum_gpu_ns_median", stage_class="moe_join")
    add("moe_add_shared", "moe_add_shared_gpu_ns", "moe_add_shared_gpu_ns_median", stage_class="moe_join")
    add("block_inject", "block_inject_gpu_ns", "block_inject_gpu_ns_median")
    add("combine", "combine_gpu_ns", "combine_gpu_ns_median")
    return rows


def emit_flash_telemetry(
    repo_root: str | os.PathLike[str],
    *,
    exact_receipt: Optional[str | os.PathLike[str]] = None,
    gpu_work_ledger: Optional[str | os.PathLike[str]] = None,
    token_critical_path: Optional[str | os.PathLike[str]] = None,
    cuda_capability_graph: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Write telemetry artifacts without promoting bounded work to token performance."""
    repo = Path(repo_root).expanduser().resolve()
    receipt_path = (
        Path(exact_receipt).expanduser().resolve()
        if exact_receipt
        else repo / "receipts" / "headless" / EXACT_RECEIPT_NAME
    )
    ledger_path = Path(gpu_work_ledger).expanduser() if gpu_work_ledger else repo / "receipts" / "headless" / GPU_WORK_LEDGER_NAME
    critical_path = Path(token_critical_path).expanduser() if token_critical_path else repo / "receipts" / "headless" / TOKEN_CRITICAL_PATH_NAME
    capability_path = Path(cuda_capability_graph).expanduser() if cuda_capability_graph else repo / "receipts" / "headless" / CUDA_CAPABILITY_GRAPH_NAME
    if not ledger_path.is_absolute():
        ledger_path = ledger_path.resolve()
    if not critical_path.is_absolute():
        critical_path = critical_path.resolve()
    if not capability_path.is_absolute():
        capability_path = capability_path.resolve()

    receipt = _read_json(receipt_path)
    execution = receipt.get("execution") if isinstance(receipt, Mapping) and isinstance(receipt.get("execution"), Mapping) else {}
    timing = receipt.get("gpu_timing") if isinstance(receipt, Mapping) and isinstance(receipt.get("gpu_timing"), Mapping) else {}
    graph = receipt.get("physical_graph") if isinstance(receipt, Mapping) and isinstance(receipt.get("physical_graph"), Mapping) else {}
    source = _receipt_ref(receipt_path, receipt)
    passed = isinstance(receipt, Mapping) and receipt.get("status") == "PASSED"
    stage_rows = _stage_rows(timing) if passed else []
    graph_gpu = timing.get("graph_gpu_ns_median") if passed else None
    graph_wall = timing.get("graph_host_wall_ns_median") if passed else None
    ledger = {
        "schema": "hcli.agentos.flash_gpu_work_ledger.v1",
        "status": "PASSED" if passed else "NOT_RUN",
        "qualification": "BOUNDED_LAYER0_GRAPH_ONLY" if passed else "NOT_RUN",
        "source_receipt": source,
        "device": timing.get("device") if passed else None,
        "scope": "layer-0 exact HyperConnection read/write around selected routed-plus-shared MoE candidate",
        "physical_graph_fingerprint": graph.get("fingerprint") if passed else None,
        "measured_runs": timing.get("measured_runs") if passed else None,
        "dispatches_per_graph": timing.get("dispatches_per_graph") if passed else None,
        "graph_gpu_ns_median": graph_gpu,
        "graph_host_wall_ns_median": graph_wall,
        "graph_wall_minus_gpu_ns_median": graph_wall - graph_gpu if isinstance(graph_wall, int) and isinstance(graph_gpu, int) else None,
        "stages": stage_rows,
        "device_intermediate_no_host_roundtrip": receipt.get("device_intermediate_no_host_roundtrip") if passed else None,
        "complete_token_runtime": False,
        "flash_tps": None,
        "complete_system_ebpw": None,
        "promotion_allowed": False,
        "claim_boundary": "Measured Metal GPU work for a bounded layer-0 candidate only; no complete-token rate or EBPW is inferred.",
    }
    critical = {
        "schema": "hcli.agentos.flash_token_critical_path.v1",
        "status": "WAITING_FOR_COMPLETE_TOKEN" if passed else "NOT_RUN",
        "source_receipt": source,
        "candidate_graph": {
            "qualification": receipt.get("qualification") if passed else None,
            "physical_graph_fingerprint": graph.get("fingerprint") if passed else None,
            "dispatches_per_graph": timing.get("dispatches_per_graph") if passed else None,
            "stages": [row.get("stage") for row in stage_rows],
        },
        "complete_token_runtime": None,
        "accepted_tokens": None,
        "complete_wall_ns_per_accepted_token": None,
        "flash_tps": None,
        "complete_system_ebpw": None,
        "promotion_allowed": False,
        "blockers": [
            "source router/top-k parity is not exact" if passed and (receipt.get("source_selection_parity") or {}).get("expert_ids_exact_match") is not True else None,
            "source BF16 activation/output parity for Q4/G64 bodies is unqualified" if passed else "exact layer-0 candidate receipt is not present",
            "attention and recurrent state organs are not in the candidate graph" if passed else None,
            "complete token runtime, accepted-token accounting, TPS, and EBPW are unmeasured",
        ],
        "claim_boundary": "This is a critical-path gate, not a token-performance result. Complete-token fields remain null until protected native execution exists.",
    }
    critical["blockers"] = [item for item in critical["blockers"] if item]
    capability = {
        "schema": "hcli.agentos.cuda_capability_graph.v1",
        "status": "PARTIAL_TRANSFER_MAP",
        "execution_device": timing.get("device") if passed else None,
        "native_backend_observed": "apple_metal" if passed else None,
        "cuda_execution_observed": False,
        "qualification": "TRANSFERABLE_ACCELERATOR_LAWS_ONLY",
        "nodes": [
            {"id": "grouped_rmsnorm", "metal_status": "FUNCTIONAL" if passed else "ABSENT", "cuda_status": "UNOBSERVED", "transfer_law": "one element per output with stream-local reduction"},
            {"id": "q4_g64_matvec", "metal_status": "FUNCTIONAL" if passed else "ABSENT", "cuda_status": "UNOBSERVED", "transfer_law": "grouped packed code/scale decode with resident activation"},
            {"id": "fused_gate_up_swiglu", "metal_status": "FUNCTIONAL" if passed else "ABSENT", "cuda_status": "UNOBSERVED", "transfer_law": "paired gate/up rows and device-local SiLU product"},
            {"id": "moe_weighted_sum", "metal_status": "FUNCTIONAL" if passed else "ABSENT", "cuda_status": "UNOBSERVED", "transfer_law": "selected-weight reduction over routed expert outputs"},
            {"id": "moe_add_shared", "metal_status": "FUNCTIONAL" if passed else "ABSENT", "cuda_status": "UNOBSERVED", "transfer_law": "elementwise routed-plus-shared join"},
            {"id": "hyperconnection_read_write", "metal_status": "FUNCTIONAL" if passed else "ABSENT", "cuda_status": "UNOBSERVED", "transfer_law": "grouped stream equations with explicit scale and broadcast"},
        ],
        "edges": [
            ["grouped_rmsnorm", "q4_g64_matvec"],
            ["q4_g64_matvec", "fused_gate_up_swiglu"],
            ["fused_gate_up_swiglu", "moe_weighted_sum"],
            ["moe_weighted_sum", "moe_add_shared"],
            ["moe_add_shared", "hyperconnection_read_write"],
        ],
        "promotion_allowed": False,
        "claim_boundary": "CUDA is not claimed on this run; this graph records accelerator laws transferable from the observed Apple Metal implementation.",
    }
    atomic_write_json(ledger_path, ledger)
    atomic_write_json(critical_path, critical)
    atomic_write_json(capability_path, capability)
    return {
        "gpu_work_ledger": {"path": str(ledger_path), "status": ledger["status"], "sha256": _sha256(ledger_path)},
        "token_critical_path": {"path": str(critical_path), "status": critical["status"], "sha256": _sha256(critical_path)},
        "cuda_capability_graph": {"path": str(capability_path), "status": capability["status"], "sha256": _sha256(capability_path)},
    }


__all__ = ["emit_flash_telemetry"]
