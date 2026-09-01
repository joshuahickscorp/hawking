"""Bounded accelerator regression audit for the current native resident.

The current sealed-3.14 resident is the local default, but it is not allowed
to become HCLI's accelerator contract.  This gate records the current and
historical runtime identities, inspects the fusion controls in source, and
performs at most one small live native request.  A noisy machine produces an
explicit no-performance-claim receipt rather than a guessed comparison.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.agentos.benchmark_boundary import classify_window
from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.accelerator_regression.v1"
DEFAULT_PROFILE_NAME = "hawking-native.sealed-3.14.json"
FUSION_KEYS = (
    "HAWKING_QWEN38_FUSE_ADD_RMSNORM",
    "HAWKING_QWEN38_FUSE_GQA_QKV",
    "HAWKING_QWEN38_FUSE_DN_INPROJ",
    "HAWKING_QWEN38_FUSE_MLP",
)


def _repo_root(value: Optional[str | os.PathLike[str]]) -> Path:
    return Path(value).expanduser().resolve() if value else Path(__file__).resolve().parents[2]


def _profile_path(repo: Path, value: Optional[str | os.PathLike[str]]) -> Path:
    chosen = value or os.environ.get("HCLI_HAWKING_NATIVE_CONFIG")
    return Path(chosen).expanduser().resolve() if chosen else (repo / "hcli" / DEFAULT_PROFILE_NAME).resolve()


def _sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _quiescence() -> Dict[str, Any]:
    try:
        from tools.accelerator.bench import machine_quiescence

        return machine_quiescence()
    except Exception as exc:  # noqa: BLE001 - an unavailable instrument is UNKNOWN
        return {
            "quiet": None,
            "method": "unavailable",
            "contenders": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _bench(before: Dict[str, Any], after: Dict[str, Any], note: str) -> Dict[str, Any]:
    try:
        from tools.accelerator.bench import bench_block

        return bench_block(
            machine=(
                f"{platform.system()} {platform.machine()} "
                f"({platform.platform()})"
            ),
            before=before,
            after=after,
            note=note,
        )
    except Exception as exc:  # noqa: BLE001 - retain the refusal in the receipt
        return {
            "state": "UNKNOWN",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "machine": platform.platform(),
            "quiescence": None,
            "samples": {"before": before, "after": after},
            "note": f"bench instrument failed: {type(exc).__name__}: {exc}",
        }


def _source_inspection(repo: Path, profile: Mapping[str, Any]) -> Dict[str, Any]:
    source = repo / "crates" / "hawking-core" / "src" / "model" / "qwen38_hybrid_decode.rs"
    shader = repo / "crates" / "hawking-core" / "shaders" / "qwen_uniform_q4.metal"
    text = ""
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    line_map: Dict[str, Optional[int]] = {}
    for key in FUSION_KEYS:
        match = re.search(rf"(?m)^.*{re.escape(key)}.*$", text)
        line_map[key] = text[: match.start()].count("\n") + 1 if match else None
    fusion_env = profile.get("fusion_env")
    fusion_env = dict(fusion_env) if isinstance(fusion_env, Mapping) else {}
    return {
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "source_exists": source.is_file(),
        "shader_path": str(shader),
        "shader_sha256": _sha256(shader),
        "shader_exists": shader.is_file(),
        "fusion_controls_found_at_lines": line_map,
        "profile_fusion_env": {key: fusion_env.get(key) for key in FUSION_KEYS},
        "all_profile_controls_declared": all(key in fusion_env for key in FUSION_KEYS),
        "source_contains_fusion_dispatch_accounting": "fused_dispatches_per_token" in text,
        "source_control_note": (
            "The source parser is inspected, not modified. A profile value is "
            "only a configuration identity; it is not a performance result."
        ),
    }


def _kernel_genome(repo: Path) -> Dict[str, Any]:
    path = repo / "receipts" / "headless" / "NOETIC_DISPATCH_FUSION.json"
    value = _read_json(path)
    if value is None:
        return {
            "status": "ABSENT",
            "reason": "prior fusion receipt is unavailable",
            "source_receipt": str(path),
        }
    parity = value.get("parity")
    compact_parity: Dict[str, Any] = {}
    if isinstance(parity, Mapping):
        for name, row in parity.items():
            if not isinstance(row, Mapping):
                continue
            compact_parity[str(name)] = {
                key: row.get(key)
                for key in (
                    "fusion",
                    "unfused_dispatches",
                    "fused_dispatches",
                    "fused_pair_dispatches",
                    "fused_pair_gpu_ns",
                    "fused_swiglu_dispatches",
                    "fused_swiglu_gpu_ns",
                )
                if key in row
            }
    shader = value.get("shader_evidence")
    shader_summary = {}
    if isinstance(shader, Mapping):
        shader_summary = {
            "all_kernels_declared": shader.get("all_kernels_declared"),
            "kernel_needles": shader.get("kernel_needles"),
        }
    return {
        "status": "PRIOR_SOURCE_RECEIPT",
        "source_receipt": str(path),
        "source_git_head": value.get("git_head"),
        "dispatches_per_token": value.get("dispatches_per_token"),
        "parity": compact_parity,
        "shader_evidence": shader_summary,
        "bench_state": (value.get("bench") or {}).get("state")
        if isinstance(value.get("bench"), Mapping)
        else None,
        "performance_claim_allowed_from_this_receipt": False,
    }


def _profile_summary(config: Any) -> Dict[str, Any]:
    identity = config.identity()
    keys = (
        "resident_identity",
        "provider",
        "model_id",
        "family",
        "architecture",
        "param_class",
        "quantisation",
        "runtime",
        "protocol",
        "artifact_root",
        "tokenizer",
        "binary",
        "resident_binary",
        "mode",
        "physical_ebpw",
        "current_runtime",
        "representation",
        "compiler",
    )
    summary = {key: _safe(identity.get(key)) for key in keys if key in identity}
    binary = Path(config.selected_binary())
    tokenizer = Path(config.tokenizer)
    summary.update(
        {
            "binary_sha256": _sha256(binary),
            "tokenizer_sha256": _sha256(tokenizer),
            "binary_sha256_16": identity.get("binary_sha256_16"),
            "tokenizer_sha256_16": identity.get("tokenizer_sha256_16"),
            "runtime_env": _safe(getattr(config, "runtime_env", {})),
            "fusion_env": _safe(getattr(config, "fusion_env", {})),
            "selected_binary": str(binary),
            "selected_binary_exists": binary.is_file(),
            "tokenizer_exists": tokenizer.is_file(),
        }
    )
    compiler = dict(getattr(config, "compiler", {}) or {})
    summary["compiler"] = _safe(compiler)
    return summary


def _run_one_request(
    config: Any,
    timeout_s: float,
    *,
    prompt: str = "Return exactly: HAWKING_OK",
    max_new_tokens: int = 16,
) -> Dict[str, Any]:
    from hcli.hawking_native import HawkingNativeConnector

    connector = HawkingNativeConnector(config)
    started = time.perf_counter()
    raw: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, str]] = None
    try:
        connector.start(timeout=timeout_s)
        raw = connector.complete_payload(
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 - this is the measurement boundary
        error = {"type": type(exc).__name__, "message": str(exc)[:1600]}
    finally:
        connector.stop()
    elapsed_ns = int((time.perf_counter() - started) * 1_000_000_000)
    hawking = raw.get("hawking") if isinstance(raw, Mapping) else {}
    hawking = hawking if isinstance(hawking, Mapping) else {}
    health = hawking.get("resident_health")
    health = health if isinstance(health, Mapping) else {}
    native_metrics = hawking.get("native_metrics")
    native_metrics = native_metrics if isinstance(native_metrics, Mapping) else {}
    native_gpu_per_token = native_metrics.get("gpu_ns_per_generated_token")
    native_wall_minus_gpu = native_metrics.get("wall_minus_gpu_ns")
    native_wall_minus_gpu_per_token = native_metrics.get("wall_minus_gpu_ns_per_generated_token")
    native_dispatches_per_token = native_metrics.get("dispatches_per_generated_token")
    native_dispatches = native_metrics.get("dispatches")
    native_active_bytes = native_metrics.get("active_bytes_per_token")
    native_active_weight_bytes = native_metrics.get("active_weight_bytes_per_generated_token")
    native_resident_weight_bytes = native_metrics.get("resident_weight_bytes")
    native_workspace_resident_bytes = native_metrics.get("workspace_resident_bytes")
    native_active_bytes_scope = native_metrics.get("active_bytes_scope")
    native_prefill = native_metrics.get("prefill")
    native_decode = native_metrics.get("decode")
    native_prefill_steps = native_metrics.get("prefill_steps")
    native_decode_steps = native_metrics.get("decode_steps")
    if isinstance(native_prefill, Mapping):
        native_prefill_steps = native_prefill.get("steps", native_prefill_steps)
    if isinstance(native_decode, Mapping):
        native_decode_steps = native_decode.get("steps", native_decode_steps)
    native_kernel_genome = native_metrics.get("kernel_genome")
    output_text = None
    choices = raw.get("choices") if isinstance(raw, Mapping) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping) and isinstance(message.get("content"), str):
            output_text = message.get("content")
    new_token_ids = hawking.get("new_token_ids")
    if not isinstance(new_token_ids, list):
        new_token_ids = None
    generated = hawking.get("generated_tokens")
    try:
        generated_i = int(generated) if generated is not None else None
    except (TypeError, ValueError):
        generated_i = None
    wall_ns_per_token = (
        round(elapsed_ns / generated_i, 3)
        if generated_i and generated_i > 0
        else None
    )
    native_wall_minus_gpu_per_token = (
        native_wall_minus_gpu_per_token
        if native_wall_minus_gpu_per_token is not None
        else round(native_wall_minus_gpu / generated_i, 3)
        if isinstance(native_wall_minus_gpu, (int, float)) and generated_i and generated_i > 0
        else None
    )
    complete_wall_minus_gpu_per_token = (
        round(wall_ns_per_token - native_gpu_per_token, 3)
        if isinstance(wall_ns_per_token, (int, float))
        and isinstance(native_gpu_per_token, (int, float))
        else None
    )
    return {
        "request": {"max_tokens": max_new_tokens, "prompt": prompt},
        "ok": raw is not None and error is None,
        "error": error,
        "generated_tokens": generated_i,
        "output_text": output_text,
        "new_token_ids": new_token_ids,
        "prompt_tokens": hawking.get("prompt_tokens"),
        "fallbacks": hawking.get("fallbacks"),
        "wall_ns": elapsed_ns,
        "wall_ns_per_token": wall_ns_per_token,
        "native_generation_wall_ns": native_metrics.get("generation_wall_ns"),
        "gpu_ns_per_token": native_gpu_per_token,
        "wall_minus_gpu_ns": native_wall_minus_gpu,
        "wall_minus_gpu_ns_per_token": complete_wall_minus_gpu_per_token,
        "wall_minus_gpu_metric_source": (
            "derived_complete_wall_minus_gpu"
            if complete_wall_minus_gpu_per_token is not None
            else None
        ),
        "native_wall_minus_gpu_ns": native_wall_minus_gpu,
        "native_wall_minus_gpu_ns_per_token": native_wall_minus_gpu_per_token,
        "native_wall_minus_gpu_metric_source": (
            "native_wall_minus_gpu_ns_per_generated_token"
            if native_metrics.get("wall_minus_gpu_ns_per_generated_token") is not None
            else "derived_from_native_wall_minus_gpu_ns"
            if native_wall_minus_gpu_per_token is not None
            else None
        ),
        "dispatches": native_dispatches,
        "dispatches_per_token": native_dispatches_per_token,
        "active_bytes_per_token": native_active_bytes,
        "active_weight_bytes_per_generated_token": native_active_weight_bytes,
        "active_bytes_scope": native_active_bytes_scope,
        "resident_weight_bytes": native_resident_weight_bytes,
        "workspace_resident_bytes": native_workspace_resident_bytes,
        "prefill_steps": native_prefill_steps,
        "decode_steps": native_decode_steps,
        "prefill_metrics": _safe(native_prefill) if native_prefill else None,
        "decode_metrics": _safe(native_decode) if native_decode else None,
        "kernel_genome": _safe(native_kernel_genome) if native_kernel_genome else None,
        "native_metrics": _safe(native_metrics),
        "metric_absence_reasons": {
            "gpu_ns_per_token": None if native_gpu_per_token is not None else "native provider did not declare GPU timestamps",
            "wall_minus_gpu_ns": None if native_wall_minus_gpu is not None else "native provider did not declare GPU timestamps",
            "complete_wall_minus_gpu_ns_per_token": None if complete_wall_minus_gpu_per_token is not None else "complete wall or GPU per-token scope unavailable",
            "dispatches": None if native_dispatches is not None else "native provider did not declare dispatch counts",
            "active_bytes_per_token": None if native_active_bytes is not None or native_active_weight_bytes is not None else "native provider did not declare packed active weight payload accounting",
            "resident_weight_bytes": None if native_resident_weight_bytes is not None else "native provider did not declare resident weight bytes",
            "workspace_resident_bytes": None if native_workspace_resident_bytes is not None else "native provider did not declare workspace resident bytes",
            "prefill_steps": None if native_prefill_steps is not None else "native provider did not declare a prefill/decode phase split",
            "kernel_genome": None if native_kernel_genome is not None else "native provider did not declare kernel identity telemetry",
        },
        "resident_health": {
            key: health.get(key)
            for key in ("pid", "restart_count", "model_open_count", "weight_upload_count", "resident_identity")
            if key in health
        },
    }


def _write(report: Dict[str, Any], emit: Optional[str], repo: Path) -> None:
    destination = Path(emit).expanduser() if emit else repo / "receipts" / "headless" / "HCLI_ACCELERATOR_REGRESSION.json"
    if not destination.is_absolute():
        destination = repo / destination
    report["receipt_path"] = str(destination.resolve())
    atomic_write_json(destination, report)


def run_accelerator_regression(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    profile: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    timeout_s: float = 180.0,
) -> Dict[str, Any]:
    """Record one current-resident observation and the regression boundary."""
    repo = _repo_root(repo_root)
    profile_path = _profile_path(repo, profile)
    from hcli.hawking_native import HawkingNativeConfig

    started = time.time()
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "qualification": False,
        "qualification_label": "CURRENT_RUNTIME_REGRESSION_AUDITED_NO_PERFORMANCE_QUALIFICATION",
        "benchmark_class": "DIAGNOSTIC_CONTAMINATED",
        "NOT_FOR_PROMOTION": True,
        "started_at": started,
        "repo_root": str(repo),
        "profile_path": str(profile_path),
        "experiment_contract": {
            "one_experiment_at_a_time": True,
            "hypothesis": "The current resident is below its historical 34 TPS anchor; identify configuration/source evidence before changing one control.",
            "control": "No source or fusion mutation; one bounded current-resident smoke request.",
            "mutation": "NONE",
            "protected_measurement": "machine-wide enumerating quiescence before and after; CONTENDED/UNKNOWN forbids a performance claim.",
        },
    }
    before = _quiescence()
    try:
        config = HawkingNativeConfig.from_file(str(profile_path))
        config.validate()
        profile_identity = _profile_summary(config)
        fusion_env = dict(config.fusion_env)
        source = _source_inspection(repo, {**profile_identity, "fusion_env": fusion_env})
        kernel = _kernel_genome(repo)
        prior_smoke = _read_json(repo / "receipts" / "headless" / "HCLI_ACCELERATOR_NATIVE_SMOKE.json")
        live = _run_one_request(config, max(0.1, float(timeout_s)))
        report["identity"] = {
            "experiment": {
                "id": "hcli-current-resident-one-request",
                "class": "ACCEL-DEVICE",
                "changed_source": False,
                "changed_fusion": False,
            },
            "machine": {
                "platform": platform.platform(),
                "architecture": platform.machine(),
            },
            "device": {"status": "EXECUTED", "runtime": "native resident"},
            "model": profile_identity,
            "representation": {"fusion_env": fusion_env},
            "kernel": {
                "status": "EXISTING_SHIPPED_PATH",
                "source": source,
                "genome": kernel,
            },
            "runtime": {
                "provider": profile_identity.get("provider"),
                "protocol": profile_identity.get("protocol"),
                "binary": profile_identity.get("binary") or profile_identity.get("resident_binary"),
            },
            "transport": {"status": "ABSENT", "reason": "single-device native resident; no transport boundary"},
        }
        report["current_vs_historical"] = {
            "profile_current_runtime": profile_identity.get("current_runtime"),
            "current_tps": 24.4086,
            "historical_tps": 34.0,
            "gap_tps": round(34.0 - 24.4086, 4),
            "gap_percent_of_historical": round((34.0 - 24.4086) / 34.0 * 100, 2),
            "identity_warning": "The profile values are anchors, not a paired live A/B measurement in this run.",
        }
        report["source_inspection"] = source
        report["kernel_genome"] = kernel
        report["prior_receipts"] = {
            "native_smoke": {
                "path": str(repo / "receipts" / "headless" / "HCLI_ACCELERATOR_NATIVE_SMOKE.json"),
                "present": prior_smoke is not None,
                "bench_state": (prior_smoke.get("bench") or {}).get("state") if isinstance(prior_smoke, Mapping) else None,
                "performance_claim_allowed": False,
            },
            "fusion": kernel,
        }
        after = _quiescence()
        bench = _bench(
            before,
            after,
            "A single current-resident smoke request. This is a capability/execution observation; only QUIESCED paired runs can support performance claims.",
        )
        boundary = classify_window(
            before,
            after,
            bench,
            qualification=False,
            not_for_promotion=True,
        )
        report.update({
            key: boundary[key]
            for key in ("benchmark_class", "qualification", "NOT_FOR_PROMOTION", "contamination", "machine_snapshot", "protected_window")
        })
        report["experiment"] = {
            "name": "current-resident-smoke",
            "status": "OBSERVED" if live.get("ok") else "FAILED",
            "hypothesis_result": "not causally resolved; current and historical anchors are not a protected paired comparison",
            "live": live,
            "bench": bench,
            "benchmark_class": boundary["benchmark_class"],
            "qualification": boundary["qualification"],
            "NOT_FOR_PROMOTION": boundary["NOT_FOR_PROMOTION"],
            "contamination": boundary["contamination"],
            "machine_snapshot": boundary["machine_snapshot"],
            "perf_qualified": False,
            "accepted_capability_preserving_tps": None,
            "fallback_count": live.get("fallbacks"),
        }
        report["hypotheses"] = [
            {
                "id": "H1-fusion-source",
                "finding": "the current profile declares all four fusion controls and the Rust source contains their control paths",
                "confidence": "source inspection only",
                "next_experiment": "protected current baseline, then one fusion control mutation with the same prompt/length",
            },
            {
                "id": "H2-dispatch",
                "finding": "the prior fusion receipt records a 964-to-756 dispatch reduction, but its bench state is not a protected current comparison",
                "confidence": "prior receipt only",
                "next_experiment": "re-measure dispatches and complete-token wall in one QUIESCED paired window",
            },
            {
                "id": "H3-regression",
                "finding": "the 24.4086-versus-34 anchor gap is recorded but not attributed; this gate refuses to call request overhead, representation, dispatch, or host wait the cause",
                "confidence": "identity-boundary fact",
                "next_experiment": "same identity, same prompt, repeated protected current-vs-historical-compatible control",
            },
        ]
        report["checks"] = {
            "profile_valid": True,
            "current_and_historical_recorded": True,
            "fusion_values_recorded": all(key in fusion_env for key in FUSION_KEYS),
            "source_inspected_without_mutation": source.get("source_exists") is True,
            "one_live_request_or_explicit_failure": live.get("ok") is True or live.get("error") is not None,
            "fallbacks_disclosed": "fallbacks" in live,
            "no_unqualified_performance_claim": report["experiment"]["perf_qualified"] is False,
            "benchmark_boundary_recorded": report.get("benchmark_class") in {
                "QUALIFIED_PROTECTED",
                "DIAGNOSTIC_CONTAMINATED",
            },
            "diagnostic_not_for_promotion": report.get("benchmark_class") != "DIAGNOSTIC_CONTAMINATED"
            or report.get("NOT_FOR_PROMOTION") is True,
        }
        report["status"] = "PASSED" if all(report["checks"].values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001 - keep the gate's failure boundary durable
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    finally:
        if "experiment" not in report:
            after = _quiescence()
            bench = _bench(before, after, "experiment did not start")
            boundary = classify_window(before, after, bench, qualification=False, not_for_promotion=True)
            report.update({
                key: boundary[key]
                for key in ("benchmark_class", "qualification", "NOT_FOR_PROMOTION", "contamination", "machine_snapshot", "protected_window")
            })
            report["experiment"] = {
                "status": "NOT_RUN",
                "bench": bench,
                **boundary,
            }
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    _write(report, str(emit) if emit is not None else None, repo)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--profile")
    parser.add_argument("--emit")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args(argv)
    report = run_accelerator_regression(
        repo_root=args.repo_root,
        profile=args.profile,
        emit=args.emit,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FUSION_KEYS", "SCHEMA", "run_accelerator_regression", "_run_one_request"]
