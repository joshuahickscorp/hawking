"""Receipt archaeology for the current Qwen27 resident.

This module reconstructs the strongest historical dispatch-fusion run that is
still described by a checked-in receipt, then compares its runtime identity to
the current HCLI profile.  Missing historical files are recorded as UNKNOWN;
the module never fills an identity gap from a nearby run or from a filename.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from hcli.persist import atomic_write_json


IDENTITY_SCHEMA = "hcli.agentos.qwen27_historical_runtime_identity.v1"
DIFF_SCHEMA = "hcli.agentos.qwen27_runtime_diff.v1"
DEFAULT_PROFILE_NAME = "hawking-native.sealed-3.14.json"
HISTORICAL_RECEIPT_NAME = "NOETIC_DISPATCH_FUSION.json"
VERIFIED = "[V]"
DERIVED = "[D]"
UNKNOWN = "UNKNOWN"


def _repo_root(value: Optional[str | os.PathLike[str]]) -> Path:
    return Path(value).expanduser().resolve() if value else Path(__file__).resolve().parents[2]


def _profile_path(repo: Path, value: Optional[str | os.PathLike[str]]) -> Path:
    chosen = value or os.environ.get("HCLI_HAWKING_NATIVE_CONFIG")
    return Path(chosen).expanduser().resolve() if chosen else (repo / "hcli" / DEFAULT_PROFILE_NAME).resolve()


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


def _safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _git(repo: Path, args: Sequence[str], *, timeout_s: float = 10.0) -> Optional[bytes]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bytes(result.stdout) if result.returncode == 0 else None


def _git_text(repo: Path, args: Sequence[str]) -> Optional[str]:
    value = _git(repo, args)
    if value is None:
        return None
    return value.decode("utf-8", errors="replace").strip()


def _git_head(repo: Path) -> Optional[str]:
    return _git_text(repo, ["rev-parse", "HEAD"])


def _git_blob_sha256(repo: Path, revision: Optional[str], path: str) -> Optional[str]:
    if not revision:
        return None
    value = _git(repo, ["show", f"{revision}:{path}"], timeout_s=20.0)
    return hashlib.sha256(value).hexdigest() if value is not None else None


def _file_record(path: Path, *, label: str = VERIFIED) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": _sha256(path),
        "label": label,
    }


def _artifact_record(root: Optional[str]) -> Dict[str, Any]:
    path = Path(root).expanduser().resolve() if root else None
    mix = path / "MIX_REPORT.json" if path is not None else None
    manifest = path / "manifest.json" if path is not None else None
    mix_body = _read_json(mix) if mix is not None else None
    manifest_body = _read_json(manifest) if manifest is not None else None
    return {
        "root": str(path) if path is not None else None,
        "exists": path.is_dir() if path is not None else False,
        "mix_report": _file_record(mix) if mix is not None else None,
        "manifest": _file_record(manifest) if manifest is not None else None,
        "declared": {
            "mix_id": mix_body.get("mix_id") if isinstance(mix_body, Mapping) else None,
            "catalog": mix_body.get("catalog") if isinstance(mix_body, Mapping) else None,
            "n_tensors": mix_body.get("n_tensors") if isinstance(mix_body, Mapping) else None,
            "payload_bytes": next(
                (mix_body.get(key) for key in ("payload_bytes", "artifact_bytes", "total_bytes")
                 if isinstance(mix_body, Mapping) and isinstance(mix_body.get(key), (int, float))),
                None,
            ),
            "manifest_tensor_payload_bytes": manifest_body.get("tensor_payload_bytes")
            if isinstance(manifest_body, Mapping)
            else None,
        },
    }


def _current_identity(repo: Path, profile_path: Path) -> Dict[str, Any]:
    from hcli.agentos.qwen38_fusion_audit import _selected_graph
    from hcli.hawking_native import HawkingNativeConfig

    config = HawkingNativeConfig.from_file(str(profile_path))
    config.validate()
    identity = config.identity()
    source_paths = {
        "qwen38_hybrid_decode": repo / "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
        "qwen38_geometry": repo / "crates/hawking-core/src/model/qwen38_geometry.rs",
        "qwen38_schedule": repo / "crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs",
        "qwen38_token_ns_ledger": repo / "crates/hawking-core/src/model/qwen38_token_ns_ledger.rs",
        "qwen_uniform_q4_metal": repo / "crates/hawking-core/shaders/qwen_uniform_q4.metal",
        "cargo_toml": repo / "Cargo.toml",
        "cargo_lock": repo / "Cargo.lock",
    }
    selected = _selected_graph(config.fusion_env, repo)
    binary = Path(config.selected_binary())
    tokenizer = Path(config.tokenizer)
    return {
        "label": VERIFIED,
        "profile": {
            "path": str(profile_path),
            "sha256": _sha256(profile_path),
            "identity": _safe(identity),
        },
        "git": {
            "head": _git_head(repo),
            "working_tree_dirty": bool(_git_text(repo, ["status", "--porcelain"])),
        },
        "source": {
            "files": {name: _file_record(path) for name, path in source_paths.items()},
            "rust_flags": os.environ.get("RUSTFLAGS"),
        },
        "binary": {
            "path": str(binary),
            "exists": binary.is_file(),
            "sha256": _sha256(binary),
            "sha256_16": identity.get("binary_sha256_16"),
        },
        "tokenizer": {
            "path": str(tokenizer),
            "exists": tokenizer.is_file(),
            "sha256": _sha256(tokenizer),
            "sha256_16": identity.get("tokenizer_sha256_16"),
        },
        "artifact": _artifact_record(identity.get("artifact_root")),
        "compiler": _safe(identity.get("compiler")),
        "runtime": {
            "provider": identity.get("provider"),
            "runtime": identity.get("runtime"),
            "protocol": identity.get("protocol"),
            "mode": identity.get("mode"),
            "executable_profile": identity.get("executable_profile"),
            "runtime_env": identity.get("runtime_env"),
        },
        "fusion": {
            "env": _safe(config.fusion_env),
            "selected_graph": _safe(selected),
        },
        "machine": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
        },
        "capability": {
            "current_runtime": _safe(identity.get("current_runtime")),
            "fallbacks_declared": _safe(identity.get("fallbacks")),
            "weights_loaded_once": identity.get("representation", {}).get("weights_loaded_once")
            if isinstance(identity.get("representation"), Mapping)
            else None,
        },
    }


def _historical_identity(repo: Path, receipt_path: Path, receipt: Mapping[str, Any]) -> Dict[str, Any]:
    revision = str(receipt.get("git_head") or "") or None
    decode = receipt.get("decode_tok_s") if isinstance(receipt.get("decode_tok_s"), Mapping) else {}
    before = decode.get("before") if isinstance(decode.get("before"), Mapping) else {}
    best = decode.get("after_mlp_swiglu_qkv_dn") if isinstance(decode.get("after_mlp_swiglu_qkv_dn"), Mapping) else {}
    gpu = receipt.get("gpu") if isinstance(receipt.get("gpu"), Mapping) else {}
    raw = receipt.get("raw_example") if isinstance(receipt.get("raw_example"), Mapping) else {}
    raw_decode = raw.get("decode") if isinstance(raw.get("decode"), Mapping) else {}
    historical_artifact = gpu.get("artifact_root") or raw.get("artifact_root")
    historical_binary = gpu.get("binary")
    source_rel = {
        "qwen38_hybrid_decode": "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
        "qwen38_geometry": "crates/hawking-core/src/model/qwen38_geometry.rs",
        "qwen38_schedule": "crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs",
        "qwen38_token_ns_ledger": "crates/hawking-core/src/model/qwen38_token_ns_ledger.rs",
        "qwen_uniform_q4_metal": "crates/hawking-core/shaders/qwen_uniform_q4.metal",
        "cargo_toml": "Cargo.toml",
        "cargo_lock": "Cargo.lock",
    }
    arm_tokens = best.get("new_token_ids") if isinstance(best.get("new_token_ids"), list) else None
    best_dispatches = best.get("dispatches_last_step_reps")
    kernel_names = receipt.get("kernels") if isinstance(receipt.get("kernels"), list) else []
    combo = receipt.get("enable", {}).get("combo_that_measured_756") if isinstance(receipt.get("enable"), Mapping) else {}
    fusion_env = {
        "HAWKING_QWEN38_FUSE_MLP": combo.get("HAWKING_QWEN38_FUSE_MLP") if isinstance(combo, Mapping) else "swiglu",
        "HAWKING_QWEN38_FUSE_GQA_QKV": combo.get("HAWKING_QWEN38_FUSE_GQA_QKV") if isinstance(combo, Mapping) else "1",
        "HAWKING_QWEN38_FUSE_DN_INPROJ": combo.get("HAWKING_QWEN38_FUSE_DN_INPROJ") if isinstance(combo, Mapping) else "1",
        "HAWKING_QWEN38_FUSE_ADD_RMSNORM": UNKNOWN,
    }
    return {
        "label": VERIFIED,
        "receipt": {
            "path": str(receipt_path),
            "sha256": _sha256(receipt_path),
            "schema": receipt.get("schema"),
            "generated_at": receipt.get("generated_at"),
            "git_head": revision,
            "bench_state": (receipt.get("bench") or {}).get("state")
            if isinstance(receipt.get("bench"), Mapping)
            else UNKNOWN,
            "qualification": False,
        },
        "run": {
            "selected_arm": "after_mlp_swiglu_qkv_dn",
            "best_tps": best.get("tok_s_mean"),
            "anchor_arm": "before",
            "anchor_tps": before.get("tok_s_mean"),
            "generated_tokens": len(arm_tokens) if arm_tokens is not None else None,
            "prompt_tokens": len(before.get("prompt_ids")) if isinstance(before.get("prompt_ids"), list) else None,
            "dispatches_per_token": best_dispatches[0] if isinstance(best_dispatches, list) and best_dispatches else None,
            "gpu_ns_per_token": (
                raw_decode.get("mlp_swiglu_qkv_dn", {}).get("median_gpu_ns_per_token_reps", [None])[1]
                if isinstance(raw_decode.get("mlp_swiglu_qkv_dn"), Mapping)
                and isinstance(raw_decode.get("mlp_swiglu_qkv_dn", {}).get("median_gpu_ns_per_token_reps"), list)
                else None
            ),
            "coherent": best.get("coherence", {}).get("coherent") if isinstance(best.get("coherence"), Mapping) else None,
            "new_token_ids": arm_tokens,
        },
        "git": {
            "head": revision,
            "working_tree_dirty": False,
        },
        "source": {
            "files": {
                name: {
                    "path": path,
                    "revision": revision,
                    "sha256": _git_blob_sha256(repo, revision, path),
                    "label": VERIFIED,
                }
                for name, path in source_rel.items()
            },
            "rust_flags": UNKNOWN,
        },
        "binary": {
            "path": historical_binary,
            "exists": bool(historical_binary and Path(historical_binary).is_file()),
            "sha256": _sha256(Path(historical_binary)) if historical_binary else None,
            "sha256_16": None,
        },
        "tokenizer": {"path": UNKNOWN, "exists": False, "sha256": None, "sha256_16": None},
        "artifact": _artifact_record(historical_artifact),
        "compiler": {
            "language": "rust",
            "profile": "release-fast",
            "cargo_toml_sha256": _git_blob_sha256(repo, revision, "Cargo.toml"),
            "cargo_lock_sha256": _git_blob_sha256(repo, revision, "Cargo.lock"),
            "rust_flags": UNKNOWN,
        },
        "runtime": {
            "provider": "native",
            "runtime": "hawking-native",
            "protocol": "hawking.qwen38.resident.v1",
            "mode": "one_shot_or_direct_native_unknown",
            "executable_profile": "release-fast",
            "runtime_env": UNKNOWN,
        },
        "fusion": {
            "env": fusion_env,
            "selected_graph": {
                "dispatches_per_token": best_dispatches[0] if isinstance(best_dispatches, list) and best_dispatches else None,
                "command_buffers": (receipt.get("dispatches_per_token") or {}).get("command_buffers")
                if isinstance(receipt.get("dispatches_per_token"), Mapping)
                else None,
                "source_derived": False,
                "label": VERIFIED,
            },
        },
        "kernel": {
            "names": kernel_names,
            "physical_trace": True,
            "label": VERIFIED,
        },
        "machine": {
            "platform": "Apple M3 Ultra (from historical bench/machine genome)",
            "architecture": "arm64",
        },
        "capability": {
            "fallbacks": (best.get("fallbacks_reps") or [None])[0]
            if isinstance(best.get("fallbacks_reps"), list)
            else 0,
            "dense_w_materialized": best.get("dense_w_materialized"),
            "coherent": best.get("coherence", {}).get("coherent") if isinstance(best.get("coherence"), Mapping) else None,
            "complete_token_accounting": UNKNOWN,
        },
    }


def _is_unknown(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().upper() in {"", UNKNOWN, "ABSENT", "UNAVAILABLE", "NOT_MEASURED"}:
        return True
    return False


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _diff_row(
    dimension: str,
    current: Any,
    historical: Any,
    *,
    impact_rank: int,
    reason: str,
    comparable: bool = True,
) -> Dict[str, Any]:
    if not comparable:
        classification = "NOT_COMPARABLE"
    elif _is_unknown(current) or _is_unknown(historical):
        classification = UNKNOWN
    elif _canonical(current) == _canonical(historical):
        classification = "SAME_VERIFIED"
    else:
        classification = "DIFFERENT_VERIFIED"
    return {
        "dimension": dimension,
        "current": _safe(current),
        "historical": _safe(historical),
        "classification": classification,
        "impact_rank": impact_rank,
        "expected_token_ns_impact": None,
        "reason": reason,
    }


def build_runtime_diff(current: Mapping[str, Any], historical: Mapping[str, Any]) -> Dict[str, Any]:
    ci = current.get("profile", {}).get("identity", {}) if isinstance(current.get("profile"), Mapping) else {}
    hi = historical.get("receipt", {}) if isinstance(historical.get("receipt"), Mapping) else {}
    cr = current.get("runtime", {}) if isinstance(current.get("runtime"), Mapping) else {}
    hr = historical.get("runtime", {}) if isinstance(historical.get("runtime"), Mapping) else {}
    cf = current.get("fusion", {}) if isinstance(current.get("fusion"), Mapping) else {}
    hf = historical.get("fusion", {}) if isinstance(historical.get("fusion"), Mapping) else {}
    cb = current.get("binary", {}) if isinstance(current.get("binary"), Mapping) else {}
    hb = historical.get("binary", {}) if isinstance(historical.get("binary"), Mapping) else {}
    ct = current.get("tokenizer", {}) if isinstance(current.get("tokenizer"), Mapping) else {}
    ht = historical.get("tokenizer", {}) if isinstance(historical.get("tokenizer"), Mapping) else {}
    cs = current.get("source", {}).get("files", {}) if isinstance(current.get("source"), Mapping) else {}
    hs = historical.get("source", {}).get("files", {}) if isinstance(historical.get("source"), Mapping) else {}
    cc = current.get("compiler", {}) if isinstance(current.get("compiler"), Mapping) else {}
    hc = historical.get("compiler", {}) if isinstance(historical.get("compiler"), Mapping) else {}
    cg = cf.get("selected_graph", {}) if isinstance(cf.get("selected_graph"), Mapping) else {}
    hg = hf.get("selected_graph", {}) if isinstance(hf.get("selected_graph"), Mapping) else {}
    crun = current.get("capability", {}).get("current_runtime", {}) if isinstance(current.get("capability"), Mapping) else {}
    hrun = historical.get("run", {}) if isinstance(historical.get("run"), Mapping) else {}
    rows = [
        _diff_row("model_family", ci.get("family"), "qwen3.8", impact_rank=3, reason="same architecture family is required before runtime comparison"),
        _diff_row("model_id", ci.get("model_id"), "Qwen3.8-27B historical fusion artifact", impact_rank=3, reason="profile model identity is not the historical artifact identity"),
        _diff_row("architecture", ci.get("architecture"), "Qwen3.8", impact_rank=3, reason="historical receipt names the Qwen3.8 runtime but does not carry a config hash"),
        _diff_row("artifact_root", ci.get("artifact_root"), historical.get("artifact", {}).get("root") if isinstance(historical.get("artifact"), Mapping) else None, impact_rank=1, reason="weights/representation are first-order runtime identity"),
        _diff_row("artifact_payload_bytes", (ci.get("artifact_inventory") or {}).get("artifact_bytes") if isinstance(ci.get("artifact_inventory"), Mapping) else None, (historical.get("artifact", {}).get("declared") or {}).get("payload_bytes") if isinstance(historical.get("artifact"), Mapping) else None, impact_rank=1, reason="stored bytes are not comparable when one artifact lacks a surviving manifest", comparable=True),
        _diff_row("tokenizer_sha256", ct.get("sha256"), ht.get("sha256"), impact_rank=2, reason="tokenization changes prompt length and can change generated ids"),
        _diff_row("binary_sha256", cb.get("sha256"), hb.get("sha256"), impact_rank=1, reason="binary hash is the executable implementation identity"),
        _diff_row("source_git_head", (current.get("git") or {}).get("head"), (historical.get("git") or {}).get("head"), impact_rank=1, reason="source commit can change graph, kernel wiring, and accounting"),
        _diff_row("qwen38_hybrid_decode_sha256", (cs.get("qwen38_hybrid_decode") or {}).get("sha256"), (hs.get("qwen38_hybrid_decode") or {}).get("sha256"), impact_rank=1, reason="fusion parser and dispatch graph authority"),
        _diff_row("metal_source_sha256", (cs.get("qwen_uniform_q4_metal") or {}).get("sha256"), (hs.get("qwen_uniform_q4_metal") or {}).get("sha256"), impact_rank=1, reason="kernel source can change GPU work even when the host graph is unchanged"),
        _diff_row("cargo_profile", cc.get("profile") or ci.get("executable_profile"), hc.get("profile"), impact_rank=2, reason="compiler optimization profile is part of the executable identity"),
        _diff_row("rust_flags", (current.get("source") or {}).get("rust_flags"), hc.get("rust_flags"), impact_rank=2, reason="Rust codegen flags are not present in the historical receipt"),
        _diff_row("fusion_env", cf.get("env"), hf.get("env"), impact_rank=1, reason="one-control fusion differences alter the selected graph"),
        _diff_row("selected_source_or_measured_dispatches", cg.get("dispatch_consequence", {}).get("selected_source_derived") if isinstance(cg.get("dispatch_consequence"), Mapping) else cg.get("dispatches_per_token"), hg.get("dispatches_per_token"), impact_rank=1, reason="source-derived current count and measured historical count are different evidence classes", comparable=False),
        _diff_row("command_buffers", cg.get("command_buffers"), hg.get("command_buffers"), impact_rank=2, reason="command-buffer boundaries affect host synchronization"),
        _diff_row("kernel_genome", (current.get("capability") or {}).get("kernel_genome"), (historical.get("kernel") or {}).get("names") if isinstance(historical.get("kernel"), Mapping) else None, impact_rank=1, reason="kernel names/genome are missing from the current sealed response until instrumented telemetry runs", comparable=False),
        _diff_row("generated_tokens", crun.get("generated_tokens") if isinstance(crun, Mapping) else None, hrun.get("generated_tokens"), impact_rank=3, reason="token accounting must match before complete-token timings can be compared"),
        _diff_row("prompt_tokens", None, hrun.get("prompt_tokens"), impact_rank=3, reason="current profile does not encode the historical prompt"),
        _diff_row("fallbacks", crun.get("fallbacks") if isinstance(crun, Mapping) else None, (historical.get("capability") or {}).get("fallbacks") if isinstance(historical.get("capability"), Mapping) else None, impact_rank=1, reason="fallbacks change the executed function and must be zero for a capability claim"),
        _diff_row("bench_state", UNKNOWN, hi.get("bench_state"), impact_rank=1, reason="historical receipt was backfilled with UNKNOWN quiescence"),
    ]
    ranked = sorted(rows, key=lambda row: (row["impact_rank"], row["dimension"]))
    return {
        "schema": DIFF_SCHEMA,
        "status": "PASSED",
        "generated_at": time.time(),
        "classification_policy": {
            "allowed": ["SAME_VERIFIED", "DIFFERENT_VERIFIED", UNKNOWN, "NOT_COMPARABLE"],
            "unknown_is_not_equal": True,
            "source_derived_is_not_physical": True,
        },
        "ranked_differences": ranked,
        "summary": {
            "different_verified": sum(row["classification"] == "DIFFERENT_VERIFIED" for row in rows),
            "same_verified": sum(row["classification"] == "SAME_VERIFIED" for row in rows),
            "unknown": sum(row["classification"] == UNKNOWN for row in rows),
            "not_comparable": sum(row["classification"] == "NOT_COMPARABLE" for row in rows),
        },
        "claim_boundary": "Historical identity is reconstructed from receipt and source evidence. The diff identifies candidates for a correction; it does not attribute the TPS gap or qualify a benchmark.",
    }


def run_runtime_archaeology(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    profile: Optional[str | os.PathLike[str]] = None,
    identity_emit: Optional[str | os.PathLike[str]] = None,
    diff_emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    repo = _repo_root(repo_root)
    profile_path = _profile_path(repo, profile)
    receipt_path = repo / "receipts" / "headless" / HISTORICAL_RECEIPT_NAME
    receipt = _read_json(receipt_path)
    started = time.time()
    result: Dict[str, Any] = {
        "schema": IDENTITY_SCHEMA,
        "status": "RUNNING",
        "repo_root": str(repo),
        "profile_path": str(profile_path),
        "historical_receipt_path": str(receipt_path),
        "historical_receipt_present": receipt is not None,
    }
    try:
        if receipt is None:
            raise FileNotFoundError(receipt_path)
        current = _current_identity(repo, profile_path)
        historical = _historical_identity(repo, receipt_path, receipt)
        diff = build_runtime_diff(current, historical)
        result.update(
            {
                "historical_selection": {
                    "receipt": HISTORICAL_RECEIPT_NAME,
                    "selection_reason": "checked-in receipt with the closest explicit ~34 TPS control and the strongest paired fusion arm",
                    "historical_anchor_tps": historical["run"].get("anchor_tps"),
                    "best_historical_tps": historical["run"].get("best_tps"),
                    "selected_arm": historical["run"].get("selected_arm"),
                },
                "current": current,
                "historical": historical,
                "diff": diff,
                "checks": {
                    "historical_receipt_is_present": True,
                    "historical_anchor_is_recorded": historical["run"].get("anchor_tps") is not None,
                    "best_historical_arm_is_recorded": historical["run"].get("best_tps") is not None,
                    "current_binary_hash_field_present": "sha256" in current.get("binary", {}),
                    "source_and_metal_hashes_recorded": all(
                        isinstance(row, Mapping) and "sha256" in row
                        for row in (current.get("source", {}).get("files", {}) or {}).values()
                    ),
                    "unknowns_are_explicit": diff.get("classification_policy", {}).get("unknown_is_not_equal") is True,
                    "no_performance_qualification": True,
                },
            }
        )
        result["status"] = "PASSED" if all(result["checks"].values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001 - preserve archaeology failure as data
        result["status"] = "FAILED"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    result["finished_at"] = time.time()
    result["elapsed_s"] = round(result["finished_at"] - started, 3)
    identity_path = Path(identity_emit).expanduser().resolve() if identity_emit else repo / "receipts" / "headless" / "QWEN27_HISTORICAL_RUNTIME_IDENTITY.json"
    diff_path = Path(diff_emit).expanduser().resolve() if diff_emit else repo / "receipts" / "headless" / "QWEN27_RUNTIME_DIFF.json"
    if not identity_path.is_absolute():
        identity_path = repo / identity_path
    if not diff_path.is_absolute():
        diff_path = repo / diff_path
    result["identity_receipt_path"] = str(identity_path)
    result["diff_receipt_path"] = str(diff_path)
    atomic_write_json(identity_path, result)
    if isinstance(result.get("diff"), Mapping):
        diff_body = dict(result["diff"])
        diff_body.update({"repo_root": str(repo), "identity_receipt_path": str(identity_path), "receipt_path": str(diff_path)})
        atomic_write_json(diff_path, diff_body)
    return result


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--profile")
    parser.add_argument("--identity-emit")
    parser.add_argument("--diff-emit")
    args = parser.parse_args(argv)
    report = run_runtime_archaeology(
        repo_root=args.repo_root,
        profile=args.profile,
        identity_emit=args.identity_emit,
        diff_emit=args.diff_emit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = [
    "DIFF_SCHEMA",
    "IDENTITY_SCHEMA",
    "build_runtime_diff",
    "main",
    "run_runtime_archaeology",
]


if __name__ == "__main__":
    raise SystemExit(main())
