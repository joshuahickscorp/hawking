"""Compose a bounded campaign of independently-owned Flash Noetic bodies.

The campaign joins already-persisted component-body and native-kernel receipts
into one namespaced PhysicalGraph/NoeticIR boundary.  It is intentionally
bounded: the cold safetensors specimen remains a parity reference, the
component bodies are the execution inputs, and no complete Flash capability,
complete-token timing, EBPW, or TPS claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from hcli.agentos import flash_graph_component
from hcli.flash_next import PINNED_REVISION, REPO_ID
from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.persist import atomic_write_json
from hcli.physical_graph import compile_physical_graph


SCHEMA = "hcli.agentos.flash_noetic_component_campaign.v1"
DEFAULT_LOADER = "FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP.json"
DEFAULT_TRANSFORM = "FLASH_FULL_TENSOR_TRANSFORM_PARITY.json"
DEFAULT_EMIT = "FLASH_NOETIC_ROUTED_EXPERT_COMPONENT_CAMPAIGN.json"
DEFAULT_COMPONENTS = (
    {
        "id": "e0_r0_128",
        "body_receipt": "FLASH_NOETIC_ROUTED_EXPERT_BODY.json",
        "kernel_receipt": "FLASH_NOETIC_Q4_BODY_KERNEL_PARITY.json",
    },
    {
        "id": "e0_r128_256",
        "body_receipt": "FLASH_NOETIC_ROUTED_EXPERT_BODY_E0_R128_256.json",
        "kernel_receipt": "FLASH_NOETIC_Q4_BODY_KERNEL_E0_R128_256_PARITY.json",
    },
    {
        "id": "e1_r0_128",
        "body_receipt": "FLASH_NOETIC_ROUTED_EXPERT_BODY_E1_R0_128.json",
        "kernel_receipt": "FLASH_NOETIC_Q4_BODY_KERNEL_E1_R0_128_PARITY.json",
    },
    {
        "id": "e32_r0_128",
        "body_receipt": "FLASH_NOETIC_ROUTED_EXPERT_BODY_E32_R0_128.json",
        "kernel_receipt": "FLASH_NOETIC_Q4_BODY_KERNEL_E32_R0_128_PARITY.json",
    },
)


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


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _resolve_receipt(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (
        (repo / "receipts" / "headless" / path).resolve(),
        (repo / path).resolve(),
        path.resolve(),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _ref(path: Path, value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "schema": value.get("schema"),
        "status": value.get("status"),
        "label": "[V]",
    }


def parse_component_specs(values: Optional[Sequence[str]]) -> list[Dict[str, str]]:
    """Parse repeated ``BODY_RECEIPT,KERNEL_RECEIPT`` CLI arguments."""
    if not values:
        return [dict(item) for item in DEFAULT_COMPONENTS]
    parsed: list[Dict[str, str]] = []
    for raw in values:
        body, separator, kernel = str(raw).partition(",")
        if not separator or not body.strip() or not kernel.strip():
            raise ValueError("--component requires BODY_RECEIPT,KERNEL_RECEIPT")
        body_name = body.strip()
        kernel_name = kernel.strip()
        parsed.append({
            "id": Path(body_name).stem,
            "body_receipt": body_name,
            "kernel_receipt": kernel_name,
        })
    return parsed


def _window(source: Mapping[str, Any], *, body: bool = False) -> Dict[str, int]:
    if body:
        return {
            "expert_index": int(source.get("expert_index", 0) or 0),
            "row_start": int(source.get("row_start", 0) or 0),
            "row_count": int(source.get("row_count", 128) or 128),
        }
    return {
        "expert_index": int(source.get("selected_expert", 0) or 0),
        "row_start": int(source.get("selected_row_start", 0) or 0),
        "row_count": int(source.get("selected_row_count", 128) or 128),
    }


def _identity(value: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = value.get("model_lake_manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    return {
        "repo": value.get("repo") or manifest.get("repo"),
        "revision": value.get("pinned_revision") or manifest.get("revision"),
        "resolved_sha": value.get("resolved_sha") or manifest.get("resolved_sha") or value.get("pinned_revision"),
        "root": value.get("root") or manifest.get("path"),
    }


def _validate_component(
    repo: Path,
    descriptor_path: Path,
    loader: Mapping[str, Any],
    transform_path: Path,
    transform: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    errors: list[str] = []
    body_path = _resolve_receipt(repo, str(spec["body_receipt"]))
    kernel_path = _resolve_receipt(repo, str(spec["kernel_receipt"]))
    body = _read_json(body_path)
    kernel = _read_json(kernel_path)
    if not isinstance(body, Mapping):
        errors.append(f"body receipt is missing or invalid: {body_path}")
    if not isinstance(kernel, Mapping):
        errors.append(f"kernel receipt is missing or invalid: {kernel_path}")
    if errors:
        return {"id": str(spec.get("id") or body_path.stem), "errors": errors, "body_path": body_path, "kernel_path": kernel_path}

    if body.get("schema") != "hcli.agentos.flash_noetic_component_body.v1":
        errors.append("body receipt schema is not the Noetic component-body schema")
    if body.get("status") != "PASSED":
        errors.append("body receipt is not PASSED")
    if body.get("source_independent") is not True or body.get("candidate_body_persisted") is not True:
        errors.append("body receipt does not prove source-independent persistence")
    if kernel.get("schema") != "hawking.flash_noetic_q4_kernel_parity.v1":
        errors.append("kernel receipt schema is not the Flash Noetic kernel schema")
    if kernel.get("status") != "PASSED":
        errors.append("kernel receipt is not PASSED")
    native_loader = kernel.get("native_loader") if isinstance(kernel.get("native_loader"), Mapping) else {}
    parity = kernel.get("parity") if isinstance(kernel.get("parity"), Mapping) else {}
    if native_loader.get("status") != "BOUNDED_NOETIC_DESCRIPTOR_AND_BODY_LOAD":
        errors.append("kernel receipt does not prove a persisted body load")
    if native_loader.get("source_independent_execution") is not True or native_loader.get("candidate_body_persisted") is not True:
        errors.append("kernel receipt does not prove source-independent execution")
    if parity.get("within_tolerance") is not True:
        errors.append("kernel parity is not within tolerance")
    if body.get("body_mutated") is not False or kernel.get("body_mutated") is not False:
        errors.append("source mutation guard is not false")
    if body.get("model_loaded") is not False or kernel.get("model_loaded") is not False:
        errors.append("whole-model load guard is not false")

    body_record = body.get("body") if isinstance(body.get("body"), Mapping) else {}
    body_file = Path(str(body_record.get("path"))).expanduser().resolve() if body_record.get("path") else None
    candidate_body = kernel.get("candidate_body") if isinstance(kernel.get("candidate_body"), Mapping) else {}
    if body_file is None or not body_file.is_file():
        errors.append("persisted body file is missing")
    else:
        actual_sha = _sha256(body_file)
        if body_record.get("sha256") != actual_sha:
            errors.append("body receipt sha256 does not match persisted body")
        if body_record.get("bytes") != body_file.stat().st_size:
            errors.append("body receipt byte count does not match persisted body")
    if body_file is not None:
        if candidate_body.get("path") != str(body_file):
            errors.append("kernel candidate body path does not match body receipt")
    elif candidate_body.get("path") is not None:
        errors.append("kernel candidate body path is present without a persisted body")
    if candidate_body.get("sha256") != body_record.get("sha256"):
        errors.append("kernel candidate body sha256 does not match body receipt")
    if candidate_body.get("bytes") != body_record.get("bytes"):
        errors.append("kernel candidate body byte count does not match body receipt")
    if candidate_body.get("receipt_path") != str(body_path):
        errors.append("kernel candidate body receipt path does not match selected body receipt")

    body_source = body.get("source_block") if isinstance(body.get("source_block"), Mapping) else {}
    kernel_source = kernel.get("source_tensor") if isinstance(kernel.get("source_tensor"), Mapping) else {}
    body_window = _window(body_source, body=True)
    kernel_window = _window(kernel_source)
    if body_window != kernel_window:
        errors.append("body and kernel component windows disagree")
    if body_source.get("tensor_name") != kernel_source.get("tensor_name"):
        errors.append("body and kernel tensor names disagree")
    if body_source.get("shape") != kernel_source.get("shape"):
        errors.append("body and kernel tensor shapes disagree")
    if body_source.get("dtype") != kernel_source.get("dtype"):
        errors.append("body and kernel tensor dtypes disagree")
    if body_source.get("bytes") != kernel_source.get("selected_block_bytes"):
        errors.append("body and kernel source-block sizes disagree")
    if body_source.get("payload_sha256") != kernel_source.get("selected_block_sha256"):
        errors.append("body and kernel source-block hashes disagree")

    identities = {
        "body": _identity(body),
        "kernel": _identity(kernel),
        "loader": _identity(loader),
        "transform": _identity(transform),
    }
    for role, identity in identities.items():
        if identity["repo"] != REPO_ID:
            errors.append(f"{role} receipt is not for {REPO_ID}")
        if identity["revision"] != PINNED_REVISION or identity["resolved_sha"] != PINNED_REVISION:
            errors.append(f"{role} receipt is not for the pinned revision")
    roots = {str(identity["root"]) for identity in identities.values() if identity["root"] is not None}
    if len(roots) > 1:
        errors.append("component receipt source roots disagree")

    graph_validation = flash_graph_component._validate(
        descriptor_path,
        loader,
        kernel_path,
        kernel,
        transform_path,
        transform,
    )
    if not graph_validation["valid"]:
        errors.extend(f"graph validation: {error}" for error in graph_validation["errors"])
    return {
        "id": str(spec.get("id") or body_path.stem),
        "errors": errors,
        "body_path": body_path,
        "kernel_path": kernel_path,
        "body": body,
        "kernel": kernel,
        "body_ref": _ref(body_path, body),
        "kernel_ref": _ref(kernel_path, kernel),
        "window": body_window,
        "body_bytes": int(body_record.get("bytes") or 0),
        "graph_validation": graph_validation,
        "graph": flash_graph_component._component_graph(graph_validation) if graph_validation["valid"] else None,
    }


def _bundle_graph(records: Sequence[Mapping[str, Any]], loader_ref: Mapping[str, Any], transform_ref: Mapping[str, Any], candidate_id: str, tensor_layout: Any) -> Dict[str, Any]:
    physical = compile_physical_graph(
        {
            "model_id": REPO_ID,
            "architecture": {
                "component": "routed_expert_component_campaign",
                "candidate_id": candidate_id,
                "tensor_layout": tensor_layout,
                "component_count": len(records),
            },
            "organs": [{
                "id": "routed_expert_component_campaign",
                "present": True,
                "tensor_count": len(records),
                "confidence": "[V]/[D]",
            }],
            "evidence": [loader_ref, transform_ref] + [item["body_ref"] for item in records] + [item["kernel_ref"] for item in records],
        },
        provider={"provider": "apple-metal", "kernel": "qwen_uniform_q4_group64_matvec"},
        devices=("apple_metal",),
    )
    nodes: list[Dict[str, Any]] = []
    edges: list[Dict[str, Any]] = []
    for record in records:
        prefix = str(record["id"])
        child = record["graph"] or {}
        for node in child.get("computation") or []:
            namespaced = dict(node)
            old_id = str(namespaced.get("id"))
            namespaced["id"] = f"{prefix}:{old_id}"
            namespaced["component_id"] = prefix
            nodes.append(namespaced)
        for edge in child.get("dependencies") or []:
            namespaced_edge = dict(edge)
            namespaced_edge["from"] = f"{prefix}:{edge.get('from')}"
            namespaced_edge["to"] = f"{prefix}:{edge.get('to')}"
            namespaced_edge["component_id"] = prefix
            edges.append(namespaced_edge)
    physical["component_scope"] = "bounded source-independent routed-expert body campaign; not complete Flash execution"
    physical["computation"] = nodes
    physical["dependencies"] = edges
    physical["representation"] = {
        "candidate_id": candidate_id,
        "component_count": len(records),
        "component_windows": [dict(item["window"]) for item in records],
        "component_body_bytes": sum(int(item["body_bytes"]) for item in records),
        "candidate_body_persisted": True,
        "source_independent_execution": True,
        "dense_rematerialization": "forbidden",
    }
    physical["residency"] = {
        "component_bodies": "persisted bounded bodies are execution inputs",
        "cold_source": "parity reference only",
        "whole_model": "not loaded",
    }
    physical["qualification"] = "BOUNDED_MULTI_COMPONENT_ONLY"
    physical["graph_execution_observed"] = False
    physical["native_kernel_execution_observed"] = True
    physical["fingerprint"] = _json_hash({key: value for key, value in physical.items() if key not in {"generated_at", "fingerprint"}})
    return physical


def run_flash_component_campaign(
    *,
    repo_root: Optional[str | Path] = None,
    loader_receipt: Optional[str | Path] = None,
    transform_receipt: Optional[str | Path] = None,
    component_specs: Optional[Sequence[str | Mapping[str, Any]]] = None,
    emit: Optional[str | Path] = None,
) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    headless = repo / "receipts" / "headless"
    descriptor_path = Path(loader_receipt).expanduser().resolve() if loader_receipt else headless / DEFAULT_LOADER
    transform_path = Path(transform_receipt).expanduser().resolve() if transform_receipt else headless / DEFAULT_TRANSFORM
    destination = Path(emit).expanduser().resolve() if emit else headless / DEFAULT_EMIT
    started = time.time()
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "semantic_type": "NoeticExecutableCandidate",
        "compiler_stage": "PhysicalGraphCompiler",
        "status": "RUNNING",
        "component_status": "NOT_COMPILED",
        "qualification": False,
        "NOT_FOR_PROMOTION": True,
        "promotion_allowed": False,
        "repo": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "whole_model_capability": "NOT_TESTED",
        "complete_token_runtime": "NOT_TESTED",
        "complete_system_ebpw": None,
        "flash_tps": None,
    }
    errors: list[str] = []
    try:
        loader = _read_json(descriptor_path)
        transform = _read_json(transform_path)
        if not isinstance(loader, Mapping) or loader.get("status") != "PASSED":
            errors.append("a PASSED Noetic loader receipt is required")
        if not isinstance(transform, Mapping) or transform.get("status") != "PASSED":
            errors.append("a PASSED full-tensor transform receipt is required")
        if not errors:
            raw_specs: Sequence[str | Mapping[str, Any]] = component_specs or DEFAULT_COMPONENTS
            specs = parse_component_specs(raw_specs) if raw_specs and isinstance(raw_specs[0], str) else [dict(item) for item in raw_specs]  # type: ignore[arg-type]
            records = [
                _validate_component(repo, descriptor_path, loader, transform_path, transform, spec)
                for spec in specs
            ]
            errors.extend(
                f"{record['id']}: {error}"
                for record in records
                for error in record.get("errors") or []
            )
            windows = [tuple(sorted(item["window"].items())) for item in records if item.get("window")]
            if len(windows) != len(set(windows)):
                errors.append("component windows overlap or duplicate")
            if not records:
                errors.append("at least one component is required")
            candidate_id = str(loader.get("candidate_id") or "")
            source = loader.get("representation_descriptor", {}).get("source_tensor", {}) if isinstance(loader.get("representation_descriptor"), Mapping) else {}
            loader_ref = _ref(descriptor_path, loader)
            transform_ref = _ref(transform_path, transform)
            report.update({
                "candidate_id": candidate_id,
                "source_identity": _identity(loader),
                "inputs": {"loader": loader_ref, "transform": transform_ref},
                "components": [
                    {
                        "id": item["id"],
                        "window": item.get("window"),
                        "body_receipt": item.get("body_ref"),
                        "kernel_receipt": item.get("kernel_ref"),
                        "body_bytes": item.get("body_bytes"),
                        "parity": (item.get("kernel") or {}).get("parity"),
                        "gpu_timing": (item.get("kernel") or {}).get("gpu_timing"),
                        "source_independent_execution": True,
                        "candidate_body_persisted": True,
                    }
                    for item in records
                ],
                "component_count": len(records),
                "component_windows": [item.get("window") for item in records],
                "source_independent_execution": all(item.get("graph_validation", {}).get("source_independent_execution") is True for item in records),
                "candidate_body_persisted": all(item.get("graph_validation", {}).get("candidate_body_persisted") is True for item in records),
                "physical_graph": _bundle_graph(records, loader_ref, transform_ref, candidate_id, source.get("layout")) if not errors else None,
                "noetic_ir": {
                    "schema": "hcli.noetic.ir.v1",
                    "semantic_type": "NoeticIR",
                    "candidate_id": candidate_id,
                    "component_count": len(records),
                    "operations": [
                        "retain_verified_source_blocks_as_parity_references",
                        *[f"load_persisted_source_independent_component_body:{item['id']}" for item in records],
                        "load_serialized_representation_descriptor",
                        *[f"dispatch_qwen_uniform_q4_group64_matvec:{item['id']}" for item in records],
                        "merge_bounded_component_outputs",
                    ],
                    "source_independent": True,
                    "complete_model": False,
                },
                "validation": {"errors": errors},
            })
            report["inputs"]["components"] = [
                {"id": item["id"], "body": item["body_ref"], "kernel": item["kernel_ref"]}
                for item in records
            ]
        report["status"] = "PASSED" if not errors else "FAILED"
        report["component_status"] = "BOUNDED_MULTI_COMPONENT_COMPILED" if not errors else "NOT_COMPILED"
        report["claim_boundary"] = "Bounded multi-component source-independent routed-expert Noetic campaign compiled from persisted bodies and native parity receipts; no complete-model loader, capability, complete-token timing, EBPW, or Flash TPS claim."
        report["next_action"] = "extend the persisted body campaign across additional routed-expert windows, then add independently-owned contracts for router, DeltaNet/state, sparse attention, n-gram, MTP, and remaining Flash organs"
    except Exception as exc:  # noqa: BLE001 - persist the failure boundary
        errors.append(f"{type(exc).__name__}: {str(exc)[:2000]}")
        report.update({
            "status": "FAILED",
            "component_status": "NOT_COMPILED",
            "validation": {"errors": errors},
            "claim_boundary": "Campaign compilation failed; no Flash capability or performance claim is made.",
        })
    report["generated_at"] = started
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    report["receipt_path"] = str(destination)
    atomic_write_json(destination, report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--loader-receipt")
    parser.add_argument("--transform-receipt")
    parser.add_argument("--component", action="append", help="BODY_RECEIPT,KERNEL_RECEIPT; repeat for additional bounded components")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = run_flash_component_campaign(
        repo_root=args.repo_root,
        loader_receipt=args.loader_receipt,
        transform_receipt=args.transform_receipt,
        component_specs=args.component,
        emit=args.emit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = [
    "DEFAULT_COMPONENTS",
    "DEFAULT_EMIT",
    "SCHEMA",
    "main",
    "parse_component_specs",
    "run_flash_component_campaign",
]


if __name__ == "__main__":
    raise SystemExit(main())
