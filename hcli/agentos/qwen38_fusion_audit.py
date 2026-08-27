"""Source-authority audit for the Qwen3.8 fusion controls.

This is a CPU-only experiment. It answers the first accelerator-regression
question from the checked-in Rust source and current HCLI profile without
pretending that a source-derived dispatch count is a physical trace or a
performance result.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.qwen38_fusion_source_audit.v1"
QUALIFICATION = "QWEN38_FUSION_SOURCE_AUTHORITY_NO_PERFORMANCE_CLAIM"
DEFAULT_PROFILE_NAME = "hawking-native.sealed-3.14.json"

MLP_VALUES = {
    "off": ["unset", "", "0", "off", "false", "no"],
    "pair": ["pair", "gate_up"],
    "swiglu": ["swiglu", "gate_up_swiglu", "1", "true", "on", "yes"],
}
SIBLING_FLAG_VALUES = ["1"]
ADD_RMSNORM_VALUES = ["1", "true", "on"]


def _repo_root(value: Optional[str | os.PathLike[str]]) -> Path:
    return Path(value).expanduser().resolve() if value else Path(__file__).resolve().parents[2]


def _profile_path(repo: Path, value: Optional[str | os.PathLike[str]]) -> Path:
    chosen = value or os.environ.get("HCLI_HAWKING_NATIVE_CONFIG")
    return Path(chosen).expanduser().resolve() if chosen else (repo / "hcli" / DEFAULT_PROFILE_NAME).resolve()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


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


def _git_head(repo: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (proc.stdout or "").strip() if proc.returncode == 0 else None


def _line(text: str, needle: str) -> Optional[int]:
    match = re.search(re.escape(needle), text)
    return text[: match.start()].count("\n") + 1 if match else None


def _source_contract(repo: Path) -> Dict[str, Any]:
    qwen_path = repo / "crates" / "hawking-core" / "src" / "model" / "qwen38_hybrid_decode.rs"
    lib_path = repo / "crates" / "hawking-core" / "src" / "lib.rs"
    geometry_path = repo / "crates" / "hawking-core" / "src" / "model" / "qwen38_geometry.rs"
    schedule_path = repo / "crates" / "hawking-core" / "src" / "model" / "qwen38_64_layer_execution_schedule.rs"
    ledger_path = repo / "crates" / "hawking-core" / "src" / "model" / "qwen38_token_ns_ledger.rs"
    qwen = _read_text(qwen_path)
    lib = _read_text(lib_path)
    geometry = _read_text(geometry_path)
    schedule = _read_text(schedule_path)
    ledger = _read_text(ledger_path)

    mlp_start = qwen.find("impl Qwen38MlpFusion")
    mlp_end = qwen.find("pub fn as_str", mlp_start if mlp_start >= 0 else 0)
    mlp = qwen[mlp_start:mlp_end if mlp_end >= 0 else None]
    add_start = qwen.find("pub fn qwen38_fuse_add_rmsnorm_from_env")
    add_end = qwen.find("pub fn qwen38_fuse_add_rmsnorm_enabled", add_start if add_start >= 0 else 0)
    add = qwen[add_start:add_end if add_end >= 0 else None]

    assertions = {
        "mlp_parser_present": "pub fn from_env()" in mlp,
        "mlp_off_values_present": all(
            f'"{value}"' in mlp for value in MLP_VALUES["off"] if value not in {"unset", ""}
        ),
        "mlp_pair_values_present": all(f'"{value}"' in mlp for value in MLP_VALUES["pair"]),
        "mlp_swiglu_values_present": all(f'"{value}"' in mlp for value in MLP_VALUES["swiglu"]),
        "mlp_unknown_value_panics": "panic!(" in mlp,
        "sibling_env_on_is_exact_one": 'v == "1"' in lib,
        "add_rmsnorm_enable_values_present": all(f'"{value}"' in add for value in ADD_RMSNORM_VALUES),
        "dispatch_formula_references_mlp_savings": "mlp.saved_dispatches_per_token()" in qwen,
        "dispatch_formula_references_gqa_savings": "2 * QWEN38_GQA_LAYERS" in qwen,
        "dispatch_formula_references_deltanet_savings": "QWEN38_DELTANET_LAYERS" in qwen,
        "dispatch_formula_references_add_rmsnorm_savings": "QWEN38_ADD_RMSNORM_SAVED_PER_TOKEN" in qwen,
        "production_dispatch_formula_present": "production_dispatches_per_token()" in ledger,
        "schedule_has_nine_plus_six": "QWEN38_MIXER_PREFIX_DISPATCHES: usize = 9" in schedule
        and "QWEN38_DENSE_MLP_SUFFIX_DISPATCHES: usize = 6" in schedule,
    }
    return {
        "files": {
            str(path): {
                "exists": path.is_file(),
                "sha256": _sha256(path),
                "line_count": text.count("\n") + (1 if text else 0),
            }
            for path, text in (
                (qwen_path, qwen),
                (lib_path, lib),
                (geometry_path, geometry),
                (schedule_path, schedule),
                (ledger_path, ledger),
            )
        },
        "authority_lines": {
            "mlp_from_env": _line(qwen, "pub fn from_env()"),
            "env_on": _line(lib, "pub fn env_on"),
            "add_rmsnorm_from_env": _line(qwen, "pub fn qwen38_fuse_add_rmsnorm_from_env"),
            "dispatch_formula": _line(qwen, "pub fn qwen38_fused_dispatches_per_token_full"),
            "production_dispatches": _line(ledger, "pub fn production_dispatches_per_token"),
        },
        "assertions": assertions,
        "all_assertions_pass": all(assertions.values()),
    }


def _mlp_mode(raw: Any) -> str:
    value = "" if raw is None else str(raw).strip().lower()
    if value in {"", "0", "off", "false", "no"}:
        return "off"
    if value in {"pair", "gate_up"}:
        return "pair"
    if value in {"swiglu", "gate_up_swiglu", "1", "true", "on", "yes"}:
        return "swiglu"
    return "invalid_panics"


def _source_int(text: str, name: str) -> Optional[int]:
    match = re.search(rf"pub const {re.escape(name)}: usize = (\d+);", text)
    return int(match.group(1)) if match else None


def _selected_graph(profile_env: Mapping[str, Any], repo: Path) -> Dict[str, Any]:
    geometry = _read_text(repo / "crates" / "hawking-core" / "src" / "model" / "qwen38_geometry.rs")
    schedule = _read_text(repo / "crates" / "hawking-core" / "src" / "model" / "qwen38_64_layer_execution_schedule.rs")
    layers = _source_int(geometry, "QWEN38_LAYERS")
    deltanet_layers = _source_int(geometry, "QWEN38_DELTANET_LAYERS")
    gqa_layers = _source_int(geometry, "QWEN38_GQA_LAYERS")
    mixer_prefix = _source_int(schedule, "QWEN38_MIXER_PREFIX_DISPATCHES")
    mlp_suffix = _source_int(schedule, "QWEN38_DENSE_MLP_SUFFIX_DISPATCHES")
    terminal_heads = 3 if "QWEN38_TERMINAL_HEAD_KERNELS: [&str; 3]" in schedule else None

    mlp = _mlp_mode(profile_env.get("HAWKING_QWEN38_FUSE_MLP"))
    gqa = str(profile_env.get("HAWKING_QWEN38_FUSE_GQA_QKV") or "").strip() == "1"
    dn = str(profile_env.get("HAWKING_QWEN38_FUSE_DN_INPROJ") or "").strip() == "1"
    add = str(profile_env.get("HAWKING_QWEN38_FUSE_ADD_RMSNORM") or "").strip().lower() in {"1", "true", "on"}
    ba = str(profile_env.get("HAWKING_QWEN38_FUSE_BA_DELTA") or "").strip().lower() in {"1", "true", "on"}

    baseline = (
        1 + layers * (mixer_prefix + mlp_suffix) + terminal_heads
        if None not in {layers, mixer_prefix, mlp_suffix, terminal_heads}
        else None
    )
    saved = {
        "mlp": {"off": 0, "pair": layers, "swiglu": 2 * layers}.get(mlp) if layers is not None else None,
        "gqa_qkv": 2 * gqa_layers if gqa and gqa_layers is not None else 0,
        "dn_inproj": deltanet_layers if dn else 0,
        "add_rmsnorm": 2 * layers if add and layers is not None else 0,
        "ba_delta": deltanet_layers if ba and deltanet_layers is not None else 0,
    }
    selected = baseline - sum(value for value in saved.values() if value is not None) if baseline is not None else None
    kernel_by_mlp = {
        "off": "qwen_affine_q2_group64_matvec_geo_tpr64_tg128 (separate gate/up)",
        "pair": "qwen_affine_q2_group64_matvec_gate_up_geo_tpr64_tg128",
        "swiglu": "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
        "invalid_panics": "NO_GRAPH (source panics before session construction)",
    }
    return {
        "profile_values": {key: profile_env.get(key) for key in sorted(profile_env) if key.startswith("HAWKING_QWEN38_")},
        "effective_controls": {
            "mlp_fusion": mlp,
            "fuse_gqa_qkv": gqa,
            "fuse_dn_inproj": dn,
            "fuse_add_rmsnorm": add,
            "fuse_ba_delta": ba,
        },
        "kernel_family": kernel_by_mlp[mlp],
        "dispatch_constants": {
            "layers": layers,
            "deltanet_layers": deltanet_layers,
            "gqa_layers": gqa_layers,
            "mixer_prefix_dispatches": mixer_prefix,
            "dense_mlp_suffix_dispatches": mlp_suffix,
            "terminal_head_kernels": terminal_heads,
        },
        "dispatch_consequence": {
            "baseline_source_derived": baseline,
            "saved_by_control": saved,
            "selected_source_derived": selected,
            "physical_dispatch_trace": None,
            "physical_trace_status": "NOT_MEASURED",
        },
    }


def run_qwen38_fusion_source_audit(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    profile: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Audit fusion value semantics and dispatch consequences without GPU work."""
    repo = _repo_root(repo_root)
    profile_path = _profile_path(repo, profile)
    destination = Path(emit).expanduser() if emit else repo / "receipts" / "headless" / "HCLI_QWEN38_FUSION_SOURCE_AUDIT.json"
    if not destination.is_absolute():
        destination = repo / destination
    started = time.time()
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "qualification": QUALIFICATION,
        "started_at": started,
        "repo_root": str(repo),
        "git_head": _git_head(repo),
        "profile_path": str(profile_path),
        "experiment": {
            "id": "qwen38-fusion-source-authority",
            "class": "ACCEL-SOURCE",
            "hypothesis": "The current profile's MLP value swiglu selects the strongest gate+up+SwiGLU fusion, while sibling boolean levers use the exact-one convention.",
            "control": "Read-only inspection of checked-in Rust source, current profile, and prior fusion receipt.",
            "mutation": "NONE",
            "physical_runtime_executed": False,
            "performance_claim": False,
        },
    }
    try:
        from hcli.hawking_native import HawkingNativeConfig

        config = HawkingNativeConfig.from_file(str(profile_path))
        config.validate()
        source = _source_contract(repo)
        selected = _selected_graph(config.fusion_env, repo)
        prior_path = repo / "receipts" / "headless" / "NOETIC_DISPATCH_FUSION.json"
        prior = _read_json(prior_path)
        prior_dispatch = prior.get("dispatches_per_token") if isinstance(prior, Mapping) else None
        report["source_contract"] = source
        report["profile"] = {
            "identity": config.identity(),
            "fusion_env": dict(config.fusion_env),
            "require_fusion_env": bool(config.require_fusion_env),
        }
        report["accepted_values"] = {
            "HAWKING_QWEN38_FUSE_MLP": {
                "off": MLP_VALUES["off"],
                "gate_up_pair": MLP_VALUES["pair"],
                "gate_up_swiglu": MLP_VALUES["swiglu"],
                "unknown": "panic before graph selection; never silently Off",
            },
            "HAWKING_QWEN38_FUSE_GQA_QKV": {
                "enabled_values": SIBLING_FLAG_VALUES,
                "unset_or_other": "disabled (crate::env_on exact-one parser)",
            },
            "HAWKING_QWEN38_FUSE_DN_INPROJ": {
                "enabled_values": SIBLING_FLAG_VALUES,
                "unset_or_other": "disabled (crate::env_on exact-one parser)",
            },
            "HAWKING_QWEN38_FUSE_ADD_RMSNORM": {
                "enabled_values": ADD_RMSNORM_VALUES,
                "bad_control_values": ["bad", "plainweight"],
                "unset_or_other": "disabled",
            },
        }
        report["selected_graph"] = selected
        report["prior_fusion_receipt"] = {
            "path": str(prior_path),
            "present": prior is not None,
            "git_head": prior.get("git_head") if isinstance(prior, Mapping) else None,
            "dispatches_per_token": prior_dispatch,
            "bench_state": (prior.get("bench") or {}).get("state") if isinstance(prior, Mapping) else None,
            "performance_claim_allowed": False,
        }
        checks = {
            "profile_exists": profile_path.is_file(),
            "profile_fusion_controls_declared": all(key in config.fusion_env for key in (
                "HAWKING_QWEN38_FUSE_ADD_RMSNORM",
                "HAWKING_QWEN38_FUSE_GQA_QKV",
                "HAWKING_QWEN38_FUSE_DN_INPROJ",
                "HAWKING_QWEN38_FUSE_MLP",
            )),
            "source_contract_passes": source.get("all_assertions_pass") is True,
            "selected_graph_is_source_derived": selected["dispatch_consequence"]["selected_source_derived"] is not None,
            "prior_receipt_retained": prior is not None,
            "no_physical_performance_claim": report["experiment"]["performance_claim"] is False,
        }
        report["checks"] = checks
        report["result"] = {
            "status": "SOURCE_SEMANTICS_RESOLVED",
            "finding": "swiglu is accepted as the strongest MLP fusion; 1/true/on/yes are also accepted only by the custom MLP parser, while sibling env_on flags accept exact 1.",
            "current_profile_selected_graph": selected["dispatch_consequence"],
            "physical_graph_trace": "NOT_MEASURED",
            "next_experiment": "In one protected QUIESCED window, trace identical complete-token requests for the current 628-dispatch source-derived profile and one-control mutations; record dispatches, GPU/wall timing, capability, and fallback count.",
        }
        report["status"] = "PASSED" if all(checks.values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001 - preserve the failure boundary in the receipt
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    report["claim_boundary"] = "This receipt resolves source semantics and derives a graph count; it does not claim that the selected graph was physically traced or that any TPS changed."
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    report["receipt_path"] = str(destination.resolve())
    atomic_write_json(destination, report)
    return report


__all__ = ["QUALIFICATION", "SCHEMA", "run_qwen38_fusion_source_audit"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--profile")
    parser.add_argument("--emit")
    args = parser.parse_args()
    result = run_qwen38_fusion_source_audit(repo_root=args.repo_root, profile=args.profile, emit=args.emit)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    raise SystemExit(0 if result.get("status") == "PASSED" else 1)
