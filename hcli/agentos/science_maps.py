"""Build bounded, receipt-first precedent and two-Qwen transfer maps."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from hcli.flash_next import PINNED_REVISION, REPO_ID
from hcli.persist import atomic_write_json


TRANSFER_SCHEMA = "hcli.agentos.qwen38_accelerator_transfer_map.v1"
PRECEDENT_SCHEMA = "hcli.agentos.flash_next_precedent_map.v1"


def _sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _read_receipt(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.stat().st_size > 4 * 1024 * 1024:
            return None
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _flatten(value: Any, limit: int = 24000) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)[:limit].lower()
    except (TypeError, ValueError):
        return str(value).lower()[:limit]


def _receipt_paths(repo: Path, *, limit: int = 1800) -> Iterable[Path]:
    root = repo / "receipts" / "headless"
    if not root.is_dir():
        return []
    paths: list[Path] = []
    try:
        for path in sorted(root.rglob("*")):
            if len(paths) >= limit:
                break
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
                paths.append(path)
    except OSError:
        return paths
    return paths


def _classify(name: str, content: str) -> Optional[str]:
    text = f"{name.lower()} {content.lower()}"
    relevant = (
        "qwen", "flash", "accelerator", "kernel", "gravity", "moe", "expert",
        "sparse", "basis", "residual", "affine", "binary", "ternary", "deltanet",
        "ngram", "mtp", "fusion", "dispatch", "complete_token", "complete tps",
        "negative", "candidate", "model_lake", "modellake", "representation",
    )
    if not any(token in text for token in relevant):
        return None
    negative = ("negative", "refus", "skip", "no_candidate", "no candidate", "floor", "rejected", "hard refusal")
    if any(token in text for token in negative):
        return "NEGATIVE_PRECEDENT"
    flash_specific = ("flash-next", "flash_next", "ngram", "deltanet", "mtp", "router", "expert", "moe", "sparse")
    architecture_specific = ("qwen3.8-27b", "qwen38", "qwen3.8", "sealed-3.14", "noetic_parent", "noetic", "qwen3b", "qwen30", "qwen80")
    if any(token in text for token in flash_specific):
        return "TEST_ON_FLASH"
    if any(token in text for token in architecture_specific):
        return "ARCHITECTURE_SPECIFIC"
    return "DIRECT_TRANSFER"


def _precedent_entries(repo: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for path in _receipt_paths(repo):
        value = _read_receipt(path)
        if value is None:
            continue
        classification = _classify(path.name, _flatten(value))
        if classification is None:
            continue
        rows.append({
            "receipt": str(path),
            "relative_receipt": str(path.relative_to(repo)),
            "sha256": _sha256(path),
            "schema": value.get("schema"),
            "pass": value.get("pass", value.get("status") == "PASSED"),
            "status": value.get("status"),
            "qualification": value.get("qualification"),
            "verdict": value.get("verdict"),
            "headline": value.get("headline"),
            "classification": classification,
            "authority_rule": "receipt fields and verifier outcomes outrank filenames and prose",
        })
    rows.sort(key=lambda item: (str(item.get("classification")), str(item.get("relative_receipt"))))
    return rows


def _find(repo: Path, names: Iterable[str]) -> list[str]:
    found: list[str] = []
    for name in names:
        path = repo / "receipts" / "headless" / name
        if path.is_file():
            found.append(str(path))
    return found


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def build_precedent_map(repo_root: Optional[str | os.PathLike[str]] = None) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    entries = _precedent_entries(repo)
    counts: Dict[str, int] = {}
    for row in entries:
        key = str(row["classification"])
        counts[key] = counts.get(key, 0) + 1
    payload = {
        "schema": PRECEDENT_SCHEMA,
        "generated_at": time.time(),
        "repo_root": str(repo),
        "source_policy": "bounded receipt census; receipts outrank prose summaries",
        "flash_next": {"repo": REPO_ID, "pinned_revision": PINNED_REVISION, "status": "METADATA_PINNED_WEIGHTS_NOT_PRESENT"},
        "classifications": ["DIRECT_TRANSFER", "TEST_ON_FLASH", "ARCHITECTURE_SPECIFIC", "NEGATIVE_PRECEDENT"],
        "entries": entries,
        "summary": {"count": len(entries), "by_classification": counts},
        "claim_boundary": "A precedent map indexes evidence and transfer hypotheses; it does not convert an architecture-specific or bounded negative result into a universal law.",
    }
    payload["fingerprint"] = _canonical_hash({"entries": entries, "flash_next": payload["flash_next"]})
    return payload


def build_transfer_map(repo_root: Optional[str | os.PathLike[str]] = None) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    profile_path = repo / "hcli" / "hawking-native.sealed-3.14.json"
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        profile = {}
    flash_receipt = _read_receipt(repo / "receipts" / "headless" / "HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json") or {}
    regression = _read_receipt(repo / "receipts" / "headless" / "HCLI_ACCELERATOR_REGRESSION.json") or {}
    common = [
        {"primitive": "packed low-bit GEMV", "model_a": "DIRECT_TRANSFER", "model_b": "TEST_ON_FLASH", "reason": "representation-native arithmetic and layout need same-source parity on each architecture"},
        {"primitive": "norms and fused epilogues", "model_a": "DIRECT_TRANSFER", "model_b": "TEST_ON_FLASH", "reason": "command-boundary and complete-token effects can transfer, tensor shapes cannot be assumed"},
        {"primitive": "command-buffer scheduling/telemetry", "model_a": "DIRECT_TRANSFER", "model_b": "DIRECT_TRANSFER", "reason": "dispatch, sync, complete wall, and fallback accounting are provider/runtime-neutral"},
        {"primitive": "resident state/update", "model_a": "ARCHITECTURE_SPECIFIC", "model_b": "TEST_ON_FLASH", "reason": "DeltaNet/state-machine work is a Flash target and must be revalidated against Qwen27"},
        {"primitive": "router/top-k and selected-expert gather", "model_a": "TEST_ON_FLASH", "model_b": "TEST_ON_FLASH", "reason": "sparse routing is not present in the sealed dense-ish parent contract; transfer is a hypothesis"},
        {"primitive": "dense-vs-NF A/B protocol", "model_a": "DIRECT_TRANSFER", "model_b": "DIRECT_TRANSFER", "reason": "same source/input/device/bench/capability controls apply to both"},
    ]
    payload = {
        "schema": TRANSFER_SCHEMA,
        "generated_at": time.time(),
        "repo_root": str(repo),
        "model_a": {
            "label": "Qwen3.8-27B sealed resident / NOETIC_PARENT_A",
            "profile": str(profile_path),
            "profile_sha256": _sha256(profile_path),
            "artifact_root": profile.get("artifact_root"),
            "physical_ebpw": (profile.get("representation") or {}).get("physical_ebpw"),
            "current_complete_tps": (profile.get("current_runtime") or {}).get("complete_tps_current_measured"),
            "historical_complete_tps": (profile.get("current_runtime") or {}).get("complete_tps_historical_qualified"),
            "fallbacks": (profile.get("current_runtime") or {}).get("fallbacks"),
            "identity_receipts": _find(repo, ["HCLI_ACCELERATOR_REGRESSION.json", "NOETIC_DISPATCH_FUSION.json", "HCLI_ACCELERATOR_NATIVE_SMOKE.json"]),
        },
        "model_b": {
            "label": "Qwen/Qwen3.8-Flash-Next pinned ModelLake source",
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "expected_bytes": (flash_receipt.get("source_identity") or {}).get("expected_complete_source_bytes"),
            "expected_file_count": (flash_receipt.get("source_identity") or {}).get("expected_file_count"),
            "architecture_fingerprint": (flash_receipt.get("architecture_fingerprint") or {}).get("value"),
            "architecture": flash_receipt.get("architecture"),
            "physical_status": "metadata_only_weights_not_present",
            "identity_receipts": _find(repo, ["HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json", "HCLI_MODELLAKE_FLASH_CENSUS.json"]),
        },
        "transfer_matrix": common,
        "model_a_regression_observation": {
            "current_vs_historical": regression.get("current_vs_historical"),
            "dispatch_kernel_genome": regression.get("prior_dispatch_kernel_genome"),
            "claim_boundary": "current audit records the regression and hypotheses; contaminated benchmark state is not a performance qualification",
        },
        "loser_policy": "The model that loses a final Pareto comparison remains a control, transfer-learning source, kernel benchmark, representation precedent, and regression detector.",
        "fingerprint": None,
        "claim_boundary": "Transfer labels are hypotheses bounded by the cited receipts. Every physical primitive and whole-model claim must re-earn same-model parity, capability, and complete-token acceptance.",
    }
    payload["fingerprint"] = _canonical_hash({
        "model_a": payload["model_a"],
        "model_b": payload["model_b"],
        "transfer_matrix": payload["transfer_matrix"],
    })
    return payload


def write_science_maps(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    transfer_emit: Optional[str | os.PathLike[str]] = None,
    precedent_emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    transfer = build_transfer_map(repo)
    precedent = build_precedent_map(repo)
    transfer_path = Path(transfer_emit).expanduser().resolve() if transfer_emit else repo / "receipts" / "headless" / "QWEN38_ACCELERATOR_TRANSFER_MAP.json"
    precedent_path = Path(precedent_emit).expanduser().resolve() if precedent_emit else repo / "receipts" / "headless" / "FLASH_NEXT_PRECEDENT_MAP.json"
    atomic_write_json(transfer_path, transfer)
    atomic_write_json(precedent_path, precedent)
    return {
        "schema": "hcli.agentos.science_maps.v1",
        "status": "PASSED",
        "transfer_map": str(transfer_path),
        "precedent_map": str(precedent_path),
        "transfer_fingerprint": transfer["fingerprint"],
        "precedent_fingerprint": precedent["fingerprint"],
        "transfer_entries": len(transfer["transfer_matrix"]),
        "precedent_entries": len(precedent["entries"]),
        "claim_boundary": "Maps are receipt-first evidence indexes and transfer hypotheses; they make no new performance or capability claim.",
    }


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--transfer-emit")
    parser.add_argument("--precedent-emit")
    args = parser.parse_args(argv)
    print(json.dumps(write_science_maps(repo_root=args.repo_root, transfer_emit=args.transfer_emit, precedent_emit=args.precedent_emit), indent=2, sort_keys=True))
    return 0


__all__ = ["PRECEDENT_SCHEMA", "TRANSFER_SCHEMA", "build_precedent_map", "build_transfer_map", "main", "write_science_maps"]


if __name__ == "__main__":
    raise SystemExit(main())
