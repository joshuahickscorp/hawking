"""Persist one exact source BF16 vector for a Flash Noetic boundary.

The Flash-Next hyperconnection norm is a checkpoint-owned rank-1 vector.  It
is deliberately kept as raw BF16 rather than routed through the Q4/G64 matrix
candidate path: this receipt proves that the native boundary can load the
exact pinned source payload without rematerialising a dense model tensor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.agentos import flash_component_body as component_body
from hcli.agentos import flash_tensor_probe
from hcli.flash_next import PINNED_REVISION, REPO_ID
from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.flash_noetic_vector_body.v1"
DEFAULT_TENSOR = "model.language_model.layers.0.mlp_hyper_connection.hc_norm.weight"
DEFAULT_CANDIDATE = "source_bf16_exact"
DEFAULT_COMPONENT_KIND = "mlp_hyperconnection_hc_norm"
DEFAULT_BODY_NAME = "flash-mlp-hyperconnection-hc-norm-l0-bf16.bin"
DEFAULT_RECEIPT = "FLASH_NOETIC_MLP_HYPER_HC_NORM_BODY_L0.json"
MAX_BODY_BYTES = 4 * 1024 * 1024


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _final_root(value: Optional[str | os.PathLike[str]]) -> Path:
    chosen = value or os.environ.get("HCLI_FLASH_NEXT_ROOT")
    if chosen:
        return Path(chosen).expanduser().resolve()
    return (component_body.transform.LAKE_ROOT / "specimens" / component_body.transform.LAKE_SLUG).resolve()


def run_flash_vector_body(
    *,
    root: Optional[str | os.PathLike[str]] = None,
    repo_root: Optional[str | os.PathLike[str]] = None,
    tensor_name: str = DEFAULT_TENSOR,
    candidate_id: str = DEFAULT_CANDIDATE,
    component_kind: str = DEFAULT_COMPONENT_KIND,
    body: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    final_root = _final_root(root)
    headless = repo / "receipts" / "headless"
    body_path = Path(body).expanduser().resolve() if body else repo / ".hcli" / "flash-residual" / DEFAULT_BODY_NAME
    receipt_path = Path(emit).expanduser().resolve() if emit else headless / DEFAULT_RECEIPT
    started = time.time()
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "status": "RUNNING",
        "repo": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "root": str(final_root),
        "tensor_name": tensor_name,
        "component_kind": component_kind,
        "candidate_id": candidate_id,
        "body_path": str(body_path),
        "body_mutated": False,
        "model_loaded": False,
        "source_independent": False,
        "candidate_body_persisted": False,
        "exact_source_payload": False,
        "whole_model_capability": "NOT_TESTED",
        "complete_token_runtime": "NOT_TESTED",
        "complete_system_ebpw": None,
        "flash_tps": None,
        "promotion_allowed": False,
    }
    try:
        if candidate_id != DEFAULT_CANDIDATE:
            raise ValueError("vector body currently supports source_bf16_exact only")
        if not component_kind.strip():
            raise ValueError("component_kind must not be empty")
        if not final_root.is_dir():
            raise FileNotFoundError(final_root)
        manifest = component_body._pinned_manifest(final_root)
        tensor = flash_tensor_probe._load_tensor_header(final_root, tensor_name)
        shape = [int(value) for value in tensor.get("shape") or []]
        if tensor.get("dtype", "").upper() != "BF16" or len(shape) != 1:
            raise ValueError("vector body requires a rank-1 BF16 source tensor")
        elements = shape[0]
        if elements <= 0:
            raise ValueError("source vector must be non-empty")
        expected_bytes = elements * 2
        if expected_bytes > MAX_BODY_BYTES:
            raise ValueError("vector body exceeds the safety limit")
        shard = Path(str(tensor["shard"]))
        guard_before = component_body.transform._source_guard(shard)
        offset = int(tensor["data_start"]) + int(tensor["data_offsets"][0])
        with shard.open("rb") as handle:
            handle.seek(offset)
            raw = handle.read(expected_bytes)
        if len(raw) != expected_bytes:
            raise ValueError(f"short vector read: {len(raw)} != {expected_bytes}")
        guard_after = component_body.transform._source_guard(shard)
        if guard_after != guard_before:
            raise RuntimeError("source shard changed while persisting the vector")
        _atomic_write_bytes(body_path, raw)
        body_sha256 = _sha256_bytes(raw)
        report.update({
            "status": "PASSED",
            "source_independent": True,
            "candidate_body_persisted": True,
            "exact_source_payload": True,
            "model_lake_manifest": manifest,
            "source_identity": {
                "repo": REPO_ID,
                "pinned_revision": PINNED_REVISION,
                "root": str(final_root),
                "manifest_path": manifest.get("manifest_path"),
                "manifest_sha256": manifest.get("manifest_sha256"),
            },
            "source_block": {
                "tensor_name": tensor_name,
                "shard": str(shard),
                "shard_name": tensor.get("shard_name"),
                "dtype": tensor.get("dtype"),
                "shape": shape,
                "data_offsets": tensor.get("data_offsets"),
                "bytes": len(raw),
                "payload_sha256": _sha256_bytes(raw),
                "label": "[V]",
            },
            "body": {
                "path": str(body_path),
                "sha256": body_sha256,
                "bytes": len(raw),
                "format": "little-endian BF16 vector payload; one value per source element",
                "dtype": "BF16",
                "elements": elements,
                "label": "[D]",
            },
            "native_loader": {
                "status": "SOURCE_INDEPENDENT_EXACT_BF16_VECTOR_PERSISTED",
                "candidate_id": candidate_id,
                "source_independent": True,
                "candidate_body_persisted": True,
                "exact_source_payload": True,
                "dense_rematerialization": "forbidden",
            },
            "source_guard": {
                "unchanged": guard_after == guard_before,
                "source_mutation": False,
            },
            "claim_boundary": "Exact pinned BF16 hc_norm source payload persisted as an independently loadable vector for the layer-0 MLP HyperConnection boundary. This qualifies source loading only; native grouped-RMSNorm execution, complete hyperconnection read/write parity, complete MoE/token runtime, Flash TPS, EBPW, and promotion remain separate gates.",
            "next_action": "load this exact BF16 vector in the native grouped HyperConnection RMSNorm, then execute the source read/write equations with explicit stream and scaling parity",
        })
    except Exception as exc:  # noqa: BLE001 - persist the failure boundary
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    report["receipt_path"] = str(receipt_path)
    atomic_write_json(receipt_path, report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--repo-root")
    parser.add_argument("--tensor-name", default=DEFAULT_TENSOR)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--component-kind", default=DEFAULT_COMPONENT_KIND)
    parser.add_argument("--body")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = run_flash_vector_body(
        root=args.root,
        repo_root=args.repo_root,
        tensor_name=args.tensor_name,
        candidate_id=args.candidate,
        component_kind=args.component_kind,
        body=args.body,
        emit=args.emit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["DEFAULT_CANDIDATE", "DEFAULT_TENSOR", "SCHEMA", "main", "run_flash_vector_body"]


if __name__ == "__main__":
    raise SystemExit(main())
