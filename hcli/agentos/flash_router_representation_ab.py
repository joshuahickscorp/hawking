"""Compare bounded Flash router representations against source routing.

This study is intentionally CPU-bounded and source-layout aware.  It reads
only the pinned layer-0 router matrix, derives several representations in
memory, runs the deterministic router reference vector, and records source
top-k overlap, logit error, storage bytes, and native-kernel compatibility.
It does not persist a new body, mutate ModelLake, load a model, or claim a
complete-token runtime.  The result is a decision receipt for the next
Noetic/native-kernel experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.agentos import flash_router_selection as router_selection
from hcli.agentos import flash_transform_parity as transform
from hcli.flash_next import PINNED_REVISION, REPO_ID
from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.persist import atomic_write_json
from hcli.physical_graph import compile_physical_graph


SCHEMA = "hcli.agentos.flash_noetic_router_representation_ab.v1"
DEFAULT_TENSOR = router_selection.DEFAULT_TENSOR
DEFAULT_EMIT = "FLASH_NOETIC_ROUTER_REPRESENTATION_AB.json"
GROUP_SIZES = (64, 32, 16)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_source_matrix(np: Any, root: Path, tensor_name: str) -> tuple[Any, Dict[str, Any], Dict[str, Any]]:
    tensor = transform._load_tensor_header(root, tensor_name)
    shape = [int(value) for value in tensor.get("shape") or []]
    if len(shape) != 2 or str(tensor.get("dtype") or "").upper() != "BF16":
        raise ValueError("router representation study requires a rank-2 BF16 source tensor")
    shard = Path(str(tensor["shard"])).expanduser().resolve()
    expected_bytes = shape[0] * shape[1] * 2
    before = transform._source_guard(shard)
    offset = int(tensor["data_start"]) + int(tensor["data_offsets"][0])
    with shard.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(expected_bytes)
    after = transform._source_guard(shard)
    if len(raw) != expected_bytes:
        raise ValueError("short pinned router source read")
    payload_sha256 = _sha256_bytes(raw)
    if before != after:
        raise ValueError("pinned router source changed during representation study")
    source = {
        "tensor_name": tensor_name,
        "shard": str(shard),
        "shape": shape,
        "dtype": tensor.get("dtype"),
        "bytes": len(raw),
        "payload_sha256": payload_sha256,
        "source_guard_unchanged": before == after,
        "source_mutation": False,
        "label": "[V]",
    }
    return transform._decode_bf16(np, raw).reshape(shape).astype(np.float32, copy=False), source, tensor


def _uniform_q4(np: Any, values: Any, group_size: int) -> tuple[bytes, bytes, Any]:
    grouped = np.asarray(values, dtype=np.float32).reshape(-1, int(group_size))
    peak = np.max(np.abs(grouped), axis=1)
    full_scale = np.where(peak > 0.0, peak / np.float32(7.0), np.float32(1.0)).astype(np.float32)
    codes = np.rint(grouped / full_scale[:, None]).clip(-8, 7).astype(np.int8)
    packed = transform._pack_nibbles(np, codes.astype(np.uint8) + np.uint8(8))
    scale_bytes = full_scale.astype(np.float16).astype("<f2", copy=False).tobytes()
    decoded = (codes.reshape(-1).astype(np.float32) * np.repeat(full_scale, int(group_size)))[: values.size].reshape(values.shape)
    return packed, scale_bytes, decoded.astype(np.float32, copy=False)


def _candidate(
    np: Any,
    source_values: Any,
    source_logits: Any,
    source_selection: Mapping[str, Any],
    *,
    candidate_id: str,
    family: str,
    group_size: int,
    native_kernel: Optional[str],
) -> Dict[str, Any]:
    if family == "nf4":
        _, _, packed, scale_bytes = transform._quantize_nf4(np, source_values)
        decoded = transform._decode_nf4(np, packed, scale_bytes, source_values.size).reshape(source_values.shape).astype(np.float32, copy=False)
        code_dtype = "nf4_packed"
    else:
        packed, scale_bytes, decoded = _uniform_q4(np, source_values, group_size)
        code_dtype = "uint4_packed"
    logits = np.matmul(decoded, router_selection.reference_input(np, source_values.shape[1])).astype(np.float32, copy=False)
    selection = router_selection.select_router(
        np,
        logits,
        top_k=int(source_selection["_router_top_k"]),
        norm_topk_prob=bool(source_selection["_norm_topk_prob"]),
    )
    parity = router_selection._selection_parity(np, source_logits, logits, source_selection["_selection"], selection)
    body_bytes = len(packed) + len(scale_bytes)
    return {
        "id": candidate_id,
        "family": family,
        "group_size": group_size,
        "code_dtype": code_dtype,
        "code_bytes": len(packed),
        "scale_bytes": len(scale_bytes),
        "candidate_bytes": body_bytes,
        "effective_bits_per_value": body_bytes * 8 / max(1, int(source_values.size)),
        "candidate_sha256": _sha256_bytes(packed + scale_bytes),
        "native_kernel": native_kernel,
        "native_kernel_compatible": native_kernel is not None,
        "body_persisted": False,
        "selection": selection,
        "source_selection_parity": parity,
        "weight_reconstruction_finite": bool(np.isfinite(decoded).all()),
        "logits_finite": bool(np.isfinite(logits).all()),
        "label": "[D]",
    }


def run_flash_router_representation_ab(
    *,
    repo_root: Optional[str | Path] = None,
    root: Optional[str | Path] = None,
    tensor_name: str = DEFAULT_TENSOR,
    emit: Optional[str | Path] = None,
) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    specimen = Path(root).expanduser().resolve() if root else transform._final_root(None)
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / DEFAULT_EMIT
    started = time.time()
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "semantic_type": "NoeticRepresentationStudy",
        "compiler_stage": "NoeticRepresentationStudy",
        "status": "RUNNING",
        "repo": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "whole_model_capability": "NOT_TESTED",
        "complete_token_runtime": "NOT_TESTED",
        "complete_system_ebpw": None,
        "flash_tps": None,
        "model_loaded": False,
        "body_mutated": False,
        "candidate_bodies_persisted": False,
        "promotion_allowed": False,
    }
    try:
        import numpy as np

        if not specimen.is_dir():
            raise FileNotFoundError(specimen)
        manifest = transform._pinned_manifest(specimen)
        config_path = specimen / "config.json"
        config = _read_json(config_path)
        if not isinstance(config, Mapping):
            raise FileNotFoundError(config_path)
        router_cfg = router_selection._router_config(config)
        source_values, source_block, tensor = _read_source_matrix(np, specimen, tensor_name)
        if int(router_cfg["num_experts"]) != source_values.shape[0]:
            raise ValueError("router config expert count does not match source matrix rows")
        vector = router_selection.reference_input(np, source_values.shape[1])
        source_logits = np.matmul(source_values, vector).astype(np.float32, copy=False)
        source_selection_result = router_selection.select_router(
            np,
            source_logits,
            top_k=int(router_cfg["num_experts_per_tok"]),
            norm_topk_prob=bool(router_cfg["norm_topk_prob"]),
        )
        source_context = {
            "_router_top_k": int(router_cfg["num_experts_per_tok"]),
            "_norm_topk_prob": bool(router_cfg["norm_topk_prob"]),
            "_selection": source_selection_result,
        }
        candidates: list[Dict[str, Any]] = []
        for group_size in GROUP_SIZES:
            candidates.append(_candidate(np, source_values, source_logits, source_context, candidate_id=f"independent_q4_g{group_size}", family="uniform_q4", group_size=group_size, native_kernel="qwen_uniform_q4_group64_matvec" if group_size == 64 else None))
        candidates.append(_candidate(np, source_values, source_logits, source_context, candidate_id="independent_nf4_g64", family="nf4", group_size=64, native_kernel=None))
        source_top_k = set(source_selection_result["expert_ids"])
        low_bit_candidates = [row for row in candidates if row["candidate_bytes"] < source_values.size * 2]
        best = min(low_bit_candidates, key=lambda row: (-int(row["source_selection_parity"]["top_k_overlap_count"]), float(row["effective_bits_per_value"]), str(row["id"])))
        native_baseline = next(row for row in candidates if row["id"] == "independent_q4_g64")
        recommendation = {
            "source_top_k": source_selection_result["expert_ids"],
            "best_low_bit_overlap_candidate": best["id"],
            "best_low_bit_overlap_count": best["source_selection_parity"]["top_k_overlap_count"],
            "native_compatible_baseline": native_baseline["id"],
            "native_compatible_baseline_overlap_count": native_baseline["source_selection_parity"]["top_k_overlap_count"],
            "decision": "retain independent_q4_g64 for the current native kernel; evaluate independent_nf4_g64 only with a native NF4 kernel, or accept the extra scale bytes of a tighter Q4 group if exact routing quality justifies it",
            "source_top_k_exact_for_any_low_bit_candidate": any(row["source_selection_parity"]["expert_ids_exact_match"] for row in low_bit_candidates),
            "label": "[D]",
        }
        physical = compile_physical_graph(
            {
                "model_id": REPO_ID,
                "architecture": {"component": "router_representation_ab", "tensor_name": tensor_name, "shape": list(source_values.shape), "num_experts": router_cfg["num_experts"], "num_experts_per_tok": router_cfg["num_experts_per_tok"]},
                "organs": [{"id": "router", "present": True, "tensor_count": 1, "confidence": "[V]/[D]"}],
                "evidence": [{"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(), "schema": "config.json", "status": "PINNED", "label": "[V]"}],
            },
            provider={"provider": "derived-cpu-representation-study", "kernel": "router_matvec_variants"},
            devices=("cpu",),
        )
        physical["component_scope"] = "bounded full Flash router representation comparison; no persisted candidate body or native selection execution"
        physical["graph_execution_observed"] = True
        physical["native_kernel_execution_observed"] = False
        physical["qualification"] = "BOUNDED_ROUTER_REPRESENTATION_AB"
        physical["computation"] = [
            {"id": "router_bf16_source_reference", "stage": "SourceSpecimen", "kind": "verified_source_reference", "execution_input": True, "tensor_name": tensor_name, "shape": list(source_values.shape)},
            {"id": "router_source_softmax_top_k", "stage": "NoeticRepresentationStudy", "kind": "source_router_selection_reference", "execution_input": True, "top_k": router_cfg["num_experts_per_tok"]},
            {"id": "router_representation_candidates", "stage": "NoeticRepresentationStudy", "kind": "in_memory_representation_candidates", "execution_input": True, "candidate_count": len(candidates), "body_persisted": False},
            {"id": "router_ebpw_routing_tradeoff", "stage": "NoeticRepresentationStudy", "kind": "bytes_vs_source_top_k_overlap", "execution_input": True, "best_candidate": best["id"]},
        ]
        physical["dependencies"] = [
            {"from": "router_bf16_source_reference", "to": "router_source_softmax_top_k", "kind": "source_selection_reference"},
            {"from": "router_bf16_source_reference", "to": "router_representation_candidates", "kind": "source_to_candidate_transform"},
            {"from": "router_representation_candidates", "to": "router_ebpw_routing_tradeoff", "kind": "candidate_comparison"},
        ]
        physical["representation"] = {"candidate_bodies_persisted": False, "dense_rematerialization": "in_memory_study_only", "source_independent_execution": False}
        physical["residency"] = {"source": "bounded pinned router tensor read", "candidate": "in-memory only", "whole_model": "not loaded"}
        physical["fingerprint"] = hashlib.sha256(json.dumps({k: v for k, v in physical.items() if k not in {"generated_at", "fingerprint"}}, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        report.update({
            "status": "PASSED",
            "source_identity": {"repo": REPO_ID, "revision": PINNED_REVISION, "root": str(specimen), "model_lake_manifest": manifest},
            "config": {"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(), "router": router_cfg, "label": "[V]"},
            "source_block": source_block,
            "input": {"definition": f"((index * {router_selection.REFERENCE_MULTIPLIER}) mod {router_selection.REFERENCE_MODULUS} - {router_selection.REFERENCE_OFFSET}) / {router_selection.REFERENCE_MODULUS}", "values": int(source_values.shape[1]), "deterministic_sha256": _sha256_bytes(vector.astype("<f4", copy=False).tobytes()), "label": "[V]"},
            "source_selection": source_selection_result,
            "candidates": candidates,
            "recommendation": recommendation,
            "physical_graph": physical,
            "noetic_ir": {"schema": "hcli.noetic.ir.v1", "semantic_type": "NoeticIR", "operations": ["read_pinned_router_source_reference", "derive_in_memory_router_representations", "execute_source_and_candidate_router_selection", "compare_ebpw_and_source_top_k_overlap", "retain_native_compatibility_boundary"], "complete_model": False, "candidate_bodies_persisted": False},
            "validation": {"errors": [], "source_guard_unchanged": source_block["source_guard_unchanged"], "source_mutation": False, "model_loaded": False, "all_candidates_finite": all(row["weight_reconstruction_finite"] and row["logits_finite"] for row in candidates)},
            "claim_boundary": "Bounded in-memory Flash router representation study only. It compares source BF16 routing to derived low-bit candidates and identifies native-kernel compatibility; no candidate body is persisted, no source-model runtime is loaded, and no complete-token, EBPW, or Flash TPS qualification is made.",
            "next_action": "implement and parity-test the selected alternate native kernel only if its source-routing overlap justifies the representation cost; otherwise continue the current Q4/G64 path toward expert dispatch",
        })
    except Exception as exc:  # noqa: BLE001 - persist the failure boundary
        report.update({"status": "FAILED", "validation": {"errors": [f"{type(exc).__name__}: {str(exc)[:2000]}"]}, "claim_boundary": "Router representation study failed; no Flash capability or performance claim is made."})
    report["generated_at"] = started
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    report["receipt_path"] = str(destination)
    atomic_write_json(destination, report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--root")
    parser.add_argument("--tensor-name", default=DEFAULT_TENSOR)
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = run_flash_router_representation_ab(repo_root=args.repo_root, root=args.root, tensor_name=args.tensor_name, emit=args.emit)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["DEFAULT_EMIT", "DEFAULT_TENSOR", "SCHEMA", "main", "run_flash_router_representation_ab"]
