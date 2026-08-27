"""Compile the bounded Flash router matrix into a Noetic physical graph.

The graph is deliberately smaller than a router implementation: it records
the persisted matrix body, descriptor load, and native Q4 matvec as verified
or observed boundaries, while top-k/sigmoid routing semantics remain an
explicit future edge.  It never promotes a router primitive to token or
whole-model evidence.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.flash_next import PINNED_REVISION, REPO_ID
from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.persist import atomic_write_json
from hcli.physical_graph import compile_physical_graph


SCHEMA = "hcli.agentos.flash_noetic_router_graph.v1"
BODY_SCHEMA = "hcli.agentos.flash_noetic_component_body.v1"
KERNEL_SCHEMA = "hawking.flash_noetic_q4_kernel_parity.v1"
DEFAULT_BODY = "FLASH_NOETIC_ROUTER_COMPONENT_BODY.json"
DEFAULT_KERNEL = "FLASH_NOETIC_ROUTER_COMPONENT_KERNEL_PARITY.json"
DEFAULT_EMIT = "FLASH_NOETIC_ROUTER_GRAPH.json"


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


def _resolve(repo: Path, value: Optional[str | Path], default: str) -> Path:
    path = Path(value).expanduser() if value else Path(default)
    if path.is_absolute():
        return path.resolve()
    return (repo / "receipts" / "headless" / path).resolve()


def _ref(path: Path, value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "schema": value.get("schema"),
        "status": value.get("status"),
        "label": "[V]",
    }


def _identity(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "repo": value.get("repo"),
        "revision": value.get("pinned_revision"),
        "root": value.get("root"),
    }


def _validate_body(body_path: Path, body: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if body.get("schema") != BODY_SCHEMA or body.get("status") != "PASSED":
        errors.append("router component body is not a PASSED component-body receipt")
    if body.get("component_kind") != "router":
        errors.append("component body is not explicitly typed as router")
    if body.get("source_independent") is not True or body.get("candidate_body_persisted") is not True:
        errors.append("router body does not prove source-independent persistence")
    if body.get("body_mutated") is not False or body.get("model_loaded") is not False:
        errors.append("router body mutation/model-load guards are not false")
    source = body.get("source_block") if isinstance(body.get("source_block"), Mapping) else {}
    descriptor = body.get("representation_descriptor") if isinstance(body.get("representation_descriptor"), Mapping) else {}
    source_descriptor = descriptor.get("source_tensor") if isinstance(descriptor.get("source_tensor"), Mapping) else {}
    shape = source.get("shape")
    if not isinstance(shape, list) or len(shape) != 2 or source_descriptor.get("shape") != shape:
        errors.append("router body source and descriptor are not the same rank-2 shape")
    if str(source.get("dtype") or "").upper() != "BF16" or source_descriptor.get("layout") != "row-major [row, column]":
        errors.append("router body source contract is not BF16 row-major")
    if descriptor.get("candidate_id") != "independent_q4_g64":
        errors.append("router descriptor candidate is not independent_q4_g64")
    body_record = body.get("body") if isinstance(body.get("body"), Mapping) else {}
    body_file = Path(str(body_record.get("path"))).expanduser().resolve() if body_record.get("path") else None
    if body_file is None or not body_file.is_file():
        errors.append("router persisted body file is missing")
    else:
        if body_record.get("sha256") != _sha256(body_file):
            errors.append("router body receipt hash does not match persisted body")
        if body_record.get("bytes") != body_file.stat().st_size:
            errors.append("router body receipt size does not match persisted body")
    return errors


def run_flash_router_graph(
    *,
    repo_root: Optional[str | Path] = None,
    body_receipt: Optional[str | Path] = None,
    kernel_receipt: Optional[str | Path] = None,
    emit: Optional[str | Path] = None,
) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    body_path = _resolve(repo, body_receipt, DEFAULT_BODY)
    kernel_path = _resolve(repo, kernel_receipt, DEFAULT_KERNEL)
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / DEFAULT_EMIT
    started = time.time()
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "semantic_type": "NoeticExecutableCandidate",
        "compiler_stage": "PhysicalGraphCompiler",
        "status": "RUNNING",
        "component_status": "NOT_COMPILED",
        "repo": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "whole_model_capability": "NOT_TESTED",
        "complete_token_runtime": "NOT_TESTED",
        "complete_system_ebpw": None,
        "flash_tps": None,
        "promotion_allowed": False,
    }
    errors: list[str] = []
    try:
        body = _read_json(body_path)
        kernel = _read_json(kernel_path)
        if not isinstance(body, Mapping):
            errors.append(f"router body receipt is missing or invalid: {body_path}")
        if not isinstance(kernel, Mapping):
            errors.append(f"router kernel receipt is missing or invalid: {kernel_path}")
        if isinstance(body, Mapping):
            errors.extend(_validate_body(body_path, body))
        if isinstance(kernel, Mapping):
            if kernel.get("schema") != KERNEL_SCHEMA or kernel.get("status") != "PASSED":
                errors.append("router kernel receipt is not PASSED")
            native_loader = kernel.get("native_loader") if isinstance(kernel.get("native_loader"), Mapping) else {}
            native_kernel = kernel.get("native_kernel") if isinstance(kernel.get("native_kernel"), Mapping) else {}
            parity = kernel.get("parity") if isinstance(kernel.get("parity"), Mapping) else {}
            if native_loader.get("status") != "BOUNDED_NOETIC_DESCRIPTOR_AND_BODY_LOAD":
                errors.append("router kernel does not prove a persisted body load")
            if native_loader.get("source_independent_execution") is not True or native_loader.get("candidate_body_persisted") is not True:
                errors.append("router kernel does not prove source-independent execution")
            if native_kernel.get("kernel_registered") is not True or parity.get("within_tolerance") is not True:
                errors.append("router native kernel registration/parity is not verified")
            if kernel.get("body_mutated") is not False or kernel.get("model_loaded") is not False:
                errors.append("router kernel mutation/model-load guards are not false")
        if isinstance(body, Mapping) and isinstance(kernel, Mapping):
            if _identity(body) != _identity(kernel):
                errors.append("router body and kernel identities disagree")
            body_source = body.get("source_block") if isinstance(body.get("source_block"), Mapping) else {}
            kernel_source = kernel.get("source_tensor") if isinstance(kernel.get("source_tensor"), Mapping) else {}
            for field in ("tensor_name", "shape", "dtype"):
                if body_source.get(field) != kernel_source.get(field):
                    errors.append(f"router body and kernel {field} disagree")
            if body_source.get("bytes") != kernel_source.get("selected_block_bytes"):
                errors.append("router body and kernel source-block sizes disagree")
            if body_source.get("payload_sha256") != kernel_source.get("selected_block_sha256"):
                errors.append("router body and kernel source-block hashes disagree")
            if body_source.get("row_start") != kernel_source.get("selected_row_start") or body_source.get("row_count") != kernel_source.get("selected_row_count"):
                errors.append("router body and kernel source-block windows disagree")
            candidate_body = kernel.get("candidate_body") if isinstance(kernel.get("candidate_body"), Mapping) else {}
            body_record = body.get("body") if isinstance(body.get("body"), Mapping) else {}
            if candidate_body.get("path") != body_record.get("path") or candidate_body.get("sha256") != body_record.get("sha256") or candidate_body.get("bytes") != body_record.get("bytes"):
                errors.append("router kernel candidate body does not match the persisted body")
        if not errors and isinstance(body, Mapping) and isinstance(kernel, Mapping):
            body_source = body["source_block"]
            descriptor = body["representation_descriptor"]
            kernel_native = kernel["native_kernel"]
            kernel_loader = kernel["native_loader"]
            candidate_id = str(body.get("candidate_id") or "")
            body_ref = _ref(body_path, body)
            kernel_ref = _ref(kernel_path, kernel)
            physical = compile_physical_graph(
                {
                    "model_id": REPO_ID,
                    "architecture": {
                        "component": "router_matrix",
                        "candidate_id": candidate_id,
                        "tensor_name": body_source.get("tensor_name"),
                        "shape": body_source.get("shape"),
                    },
                    "organs": [{"id": "router", "present": True, "tensor_count": 1, "confidence": "[V]/[D]"}],
                    "evidence": [body_ref, kernel_ref],
                },
                provider={"provider": "apple-metal", "kernel": kernel_native.get("kernel")},
                devices=("apple_metal",),
            )
            physical["component_scope"] = "bounded source-independent Flash router matrix row block; top-k/sigmoid semantics not executed"
            physical["computation"] = [
                {"id": "source_matrix_reference", "stage": "SourceSpecimen", "kind": "verified_source_reference", "execution_input": False, "tensor_name": body_source.get("tensor_name"), "row_start": body_source.get("row_start"), "row_count": body_source.get("row_count")},
                {"id": "noetic_router_body_load", "stage": "NoeticCompiler", "kind": "source_independent_component_body", "execution_input": True, "candidate_id": candidate_id, "body_persisted": True},
                {"id": "noetic_router_descriptor_load", "stage": "NoeticCompiler", "kind": "serialized_representation_descriptor", "execution_input": True, "descriptor_schema": descriptor.get("schema"), "descriptor_sha256": descriptor.get("descriptor_sha256")},
                {"id": "router_q4_matvec", "stage": "HawkingAccelerator", "kind": "native_kernel_dispatch", "execution_input": True, "kernel": kernel_native.get("kernel"), "registered": True, "parity_within_tolerance": True, "dispatches_per_sample": kernel_native.get("dispatches_per_sample")},
                {"id": "router_logits_boundary", "stage": "NoeticExecutableCandidate", "kind": "bounded_router_logits", "execution_input": True, "top_k_executed": False, "sigmoid_executed": False, "complete_token": False},
            ]
            physical["dependencies"] = [
                {"from": "noetic_router_body_load", "to": "noetic_router_descriptor_load", "kind": "body_to_descriptor"},
                {"from": "noetic_router_descriptor_load", "to": "router_q4_matvec", "kind": "descriptor_to_kernel"},
                {"from": "router_q4_matvec", "to": "router_logits_boundary", "kind": "kernel_to_logits"},
                {"from": "source_matrix_reference", "to": "router_logits_boundary", "kind": "parity_reference_only"},
            ]
            physical["representation"] = {
                "candidate_id": candidate_id,
                "source_layout": descriptor.get("source_tensor", {}).get("layout"),
                "component_body_bytes": (body.get("body") or {}).get("bytes"),
                "candidate_body_persisted": True,
                "source_independent_execution": True,
                "dense_rematerialization": "forbidden",
            }
            physical["residency"] = {"component_body": "persisted body is the execution input", "cold_source": "parity reference only", "whole_model": "not loaded"}
            physical["qualification"] = "BOUNDED_ROUTER_MATRIX_ONLY"
            physical["graph_execution_observed"] = False
            physical["native_kernel_execution_observed"] = True
            physical["fingerprint"] = hashlib.sha256(json.dumps({k: v for k, v in physical.items() if k not in {"generated_at", "fingerprint"}}, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
            report.update({
                "status": "PASSED",
                "component_status": "BOUNDED_ROUTER_MATRIX_COMPILED",
                "candidate_id": candidate_id,
                "source_identity": _identity(body),
                "component_window": {"row_start": body_source.get("row_start"), "row_count": body_source.get("row_count")},
                "source_independent_execution": True,
                "candidate_body_persisted": True,
                "physical_graph": physical,
                "noetic_ir": {
                    "schema": "hcli.noetic.ir.v1",
                    "semantic_type": "NoeticIR",
                    "candidate_id": candidate_id,
                    "operations": ["retain_verified_router_matrix_as_parity_reference", "load_persisted_source_independent_router_body", "load_serialized_representation_descriptor", "dispatch_qwen_uniform_q4_group64_matvec", "emit_bounded_router_logits", "defer_router_sigmoid_and_top_k"],
                    "source_independent": True,
                    "complete_model": False,
                },
                "inputs": {"body": body_ref, "kernel": kernel_ref},
                "validation": {"errors": []},
                "next_action": "implement and verify router sigmoid/top-k semantics against the source model before connecting this primitive to a complete token graph",
            })
        report["claim_boundary"] = "Bounded source-independent Flash router matrix body, serialized Noetic descriptor, and native Q4 matvec compiled; router selection semantics, complete-model capability, complete-token runtime, EBPW, and Flash TPS remain untested."
        if errors:
            report["validation"] = {"errors": errors}
            report["status"] = "FAILED"
            report["component_status"] = "NOT_COMPILED"
    except Exception as exc:  # noqa: BLE001 - preserve the failure boundary
        errors.append(f"{type(exc).__name__}: {str(exc)[:2000]}")
        report.update({"status": "FAILED", "component_status": "NOT_COMPILED", "validation": {"errors": errors}, "claim_boundary": "Router graph compilation failed; no Flash capability or performance claim is made."})
    report["generated_at"] = started
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    report["receipt_path"] = str(destination)
    atomic_write_json(destination, report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--body-receipt")
    parser.add_argument("--kernel-receipt")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = run_flash_router_graph(repo_root=args.repo_root, body_receipt=args.body_receipt, kernel_receipt=args.kernel_receipt, emit=args.emit)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["DEFAULT_BODY", "DEFAULT_EMIT", "DEFAULT_KERNEL", "SCHEMA", "main", "run_flash_router_graph"]


if __name__ == "__main__":
    raise SystemExit(main())
