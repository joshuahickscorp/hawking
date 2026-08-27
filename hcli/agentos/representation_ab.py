"""Dense-versus-NF A/B protocol and evidence scaffold.

The scaffold is executable and strict about matched controls, but it does not
invent measurements while Flash-Next weights and a native NF executable are
absent.  A future worker can fill a row and run ``evaluate_ab`` without
changing the acceptance law.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.dense_vs_nf_ab.v1"
REQUIRED_MEASUREMENTS = (
    "representation_bytes",
    "bytes_touched",
    "gpu_ns",
    "complete_wall_ns",
    "dispatches",
    "numerical_deviation",
)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def evaluate_ab(candidate: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    body = dict(candidate or {})
    controls = body.get("controls") if isinstance(body.get("controls"), Mapping) else {}
    measurements = body.get("measurements") if isinstance(body.get("measurements"), Mapping) else {}
    checks = {
        "same_source": controls.get("same_source") is True,
        "same_input": controls.get("same_input") is True,
        "same_dimensions": controls.get("same_dimensions") is True,
        "same_device": controls.get("same_device") is True,
        "same_bench_state": controls.get("same_bench_state") is True,
        "capability_preserved": body.get("capability_preserved") is True,
        "fallbacks_disclosed_and_zero": body.get("fallback_count") == 0,
    }
    missing = [name for name in REQUIRED_MEASUREMENTS if measurements.get(name) is None]
    missing.extend(name for name, ok in checks.items() if not ok)
    if missing:
        verdict = "INCONCLUSIVE"
        status = "INCOMPLETE"
    else:
        dense = body.get("dense") if isinstance(body.get("dense"), Mapping) else {}
        nf = body.get("nf") if isinstance(body.get("nf"), Mapping) else {}
        dense_ns = dense.get("complete_wall_ns")
        nf_ns = nf.get("complete_wall_ns")
        if not isinstance(dense_ns, (int, float)) or not isinstance(nf_ns, (int, float)):
            verdict = "INCONCLUSIVE"
        elif nf_ns < dense_ns:
            verdict = "NF_KERNEL_WINS"
        elif dense_ns < nf_ns:
            verdict = "DENSE_WINS"
        else:
            verdict = "INCONCLUSIVE"
        status = "MEASURED"
    return {
        "schema": SCHEMA,
        "status": status,
        "verdict": verdict,
        "checks": checks,
        "missing_or_refused": sorted(set(str(item) for item in missing)),
        "claim_boundary": "Only matched, capability-preserving complete-token measurements can produce a system verdict; a primitive win does not imply a whole-model win.",
    }


def _model_identity(repo: Path) -> Dict[str, Any]:
    profile_path = repo / "hcli" / "hawking-native.sealed-3.14.json"
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        profile = {}
    try:
        flash = json.loads((repo / "receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        flash = {}
    return {
        "model_a": {
            "identity": profile.get("model_id", "Qwen3.8-27B sealed resident"),
            "artifact_root": profile.get("artifact_root"),
            "profile": str(profile_path),
            "runtime": profile.get("runtime"),
        },
        "model_b": {
            "identity": "Qwen/Qwen3.8-Flash-Next",
            "pinned_revision": (flash.get("source_identity") or {}).get("pinned_revision") or (flash.get("source_identity") or {}).get("resolved_revision"),
            "architecture_fingerprint": (flash.get("architecture_fingerprint") or {}).get("value"),
            "weights_status": "NOT_PRESENT_OR_NOT_NATIVE",
        },
    }


def run_ab_scaffold(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    organs = (
        "packed_low_bit_gemv",
        "expert_gemv",
        "router_topk_gather",
        "deltanet_state_update",
        "ngram_lookup",
        "sparse_attention",
        "lm_head_selection",
        "mtp_draft_verify_rollback",
    )
    rows = []
    for model_key in ("model_a", "model_b"):
        for organ in organs:
            rows.append({
                "model": model_key,
                "organ": organ,
                "dense": {"representation": "source/dense", "complete_wall_ns": None, "gpu_ns": None, "dispatches": None},
                "nf": {"representation": "Noetic/Gravity NF candidate", "complete_wall_ns": None, "gpu_ns": None, "dispatches": None},
                "measurements": {name: None for name in REQUIRED_MEASUREMENTS},
                "controls": {name: False for name in ("same_source", "same_input", "same_dimensions", "same_device", "same_bench_state")},
                "capability_preserved": None,
                "fallback_count": None,
                "verdict": "NOT_RUN",
            })
    report = {
        "schema": SCHEMA,
        "status": "READY_SCAFFOLD",
        "generated_at": time.time(),
        "repo_root": str(repo),
        "models": _model_identity(repo),
        "protocol": {
            "A": "source/dense representation plus competent dense kernel",
            "B": "NF representation plus competent native NF kernel",
            "required_controls": ["same source", "same input", "same dimensions", "same device", "same benchmark state"],
            "required_measurements": list(REQUIRED_MEASUREMENTS),
            "required_capability": "protected capability contract and whole-model reference comparison",
            "complete_token_authority": "complete wall time, not raw kernel-only TPS",
        },
        "rows": rows,
        "evaluation": evaluate_ab({}),
        "fingerprint": _hash({"models": _model_identity(repo), "organs": organs, "protocol": REQUIRED_MEASUREMENTS}),
        "claim_boundary": "No A/B performance, capability, or Flash-Next whole-model result is claimed until physical rows are filled by matched native executables.",
    }
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / "HCLI_DENSE_VS_NF_AB_SCAFFOLD.json"
    atomic_write_json(destination, report)
    report["receipt_path"] = str(destination)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    print(json.dumps(run_ab_scaffold(repo_root=args.repo_root, emit=args.emit), indent=2, sort_keys=True, default=str))
    return 0


__all__ = ["REQUIRED_MEASUREMENTS", "SCHEMA", "evaluate_ab", "main", "run_ab_scaffold"]


if __name__ == "__main__":
    raise SystemExit(main())
