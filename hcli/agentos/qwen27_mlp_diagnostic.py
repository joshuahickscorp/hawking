"""Run the smallest useful Qwen27 MLP selector experiment.

The experiment compares the two source-approved spellings ``swiglu`` and
``1``.  It changes only the environment passed to a child resident, records
the complete native telemetry envelope, and never edits the profile or source.
Contaminated runs remain valid for parser/graph questions but are explicitly
not performance-promotion evidence.
"""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.agentos.accelerator_regression import (
    _bench,
    _profile_path,
    _quiescence,
    _repo_root,
    _run_one_request,
    _safe,
)
from hcli.agentos.benchmark_boundary import (
    DIAGNOSTIC_CONTAMINATED,
    QUALIFIED_PROTECTED,
    classify_window,
)
from hcli.agentos.qwen38_fusion_audit import _selected_graph
from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.qwen27_mlp_diagnostic_ab.v1"
DEFAULT_PROFILE_NAME = "hawking-native.sealed-3.14.json"
DEFAULT_EMIT_NAME = "QWEN27_MLP_DIAGNOSTIC_AB.json"
MLP_ARMS = ("swiglu", "1")
PROMPT = "Return exactly: HAWKING_OK"
MAX_NEW_TOKENS = 16


def _graph_identity(live: Mapping[str, Any]) -> Dict[str, Any]:
    native = live.get("native_metrics") if isinstance(live.get("native_metrics"), Mapping) else {}
    genome = native.get("kernel_genome") if isinstance(native.get("kernel_genome"), Mapping) else {}
    return {
        "dispatches": live.get("dispatches"),
        "dispatches_per_token": live.get("dispatches_per_token"),
        "kernel_histogram": _safe(genome.get("histogram")),
        "trace_enabled": genome.get("trace_enabled"),
    }


def _arm_config(base: Any, mlp_value: str, resident_binary: Optional[str]) -> Any:
    fusion = dict(base.fusion_env)
    fusion["HAWKING_QWEN38_FUSE_MLP"] = mlp_value
    runtime = dict(getattr(base, "runtime_env", {}) or {})
    runtime["HAWKING_TRACE_DISPATCH"] = "1"
    changes: Dict[str, Any] = {
        "fusion_env": fusion,
        "runtime_env": runtime,
        # The profile's production validator intentionally accepts only the
        # sealed spelling. This A/B is source-approved but experimental, so
        # the control is relaxed on the child config rather than on the repo.
        "require_fusion_env": False,
    }
    if resident_binary:
        changes["resident_binary"] = resident_binary
    return replace(base, **changes)


def _run_arm(config: Any, value: str, *, timeout_s: float) -> Dict[str, Any]:
    before = _quiescence()
    started = time.time()
    live = _run_one_request(
        config,
        timeout_s,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    after = _quiescence()
    bench = _bench(
        before,
        after,
        f"Qwen27 MLP selector A/B arm {value!r}; parser/graph diagnostic. Only a protected paired run may support a performance claim.",
    )
    boundary = classify_window(
        before,
        after,
        bench,
        qualification=bench.get("state") == "QUIESCED",
        not_for_promotion=True,
    )
    return {
        "arm": value,
        "role": "control" if value == "swiglu" else "alias_mutation",
        "started_at": started,
        "finished_at": time.time(),
        "elapsed_s": round(time.time() - started, 3),
        "fusion_env": dict(config.fusion_env),
        "runtime_env": dict(config.runtime_env),
        "config_identity": _safe(config.identity()),
        "live": live,
        "graph_identity": _graph_identity(live),
        "bench": bench,
        **boundary,
    }


def _overall_boundary(arms: list[Mapping[str, Any]]) -> Dict[str, Any]:
    if not arms:
        return classify_window(None, None, None, qualification=False, not_for_promotion=True)
    first_snapshot = arms[0].get("machine_snapshot") if isinstance(arms[0].get("machine_snapshot"), Mapping) else {}
    last_snapshot = arms[-1].get("machine_snapshot") if isinstance(arms[-1].get("machine_snapshot"), Mapping) else {}
    before = first_snapshot.get("before")
    after = last_snapshot.get("after")
    all_protected = all(arm.get("benchmark_class") == QUALIFIED_PROTECTED for arm in arms)
    synthetic_bench = {
        "state": "QUIESCED" if all_protected else "CONTENDED",
        "samples": {"before": before, "after": after},
    }
    return classify_window(
        before if isinstance(before, Mapping) else None,
        after if isinstance(after, Mapping) else None,
        synthetic_bench,
        qualification=all_protected,
        not_for_promotion=True,
    )


def _verdict(arms: list[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(arms) != 2:
        return {"experiment_verdict": "INCONCLUSIVE", "selector_verdict": "MISSING_ARM"}
    left, right = arms
    left_live = left.get("live") if isinstance(left.get("live"), Mapping) else {}
    right_live = right.get("live") if isinstance(right.get("live"), Mapping) else {}
    if not (left_live.get("ok") and right_live.get("ok")):
        return {
            "experiment_verdict": "INCONCLUSIVE",
            "selector_verdict": "RUNTIME_FAILURE",
            "reason": "one or both native A/B arms failed; no parser conclusion is drawn",
        }
    graph_same = _safe(left.get("graph_identity")) == _safe(right.get("graph_identity"))
    ids_same = left_live.get("new_token_ids") == right_live.get("new_token_ids")
    if graph_same and ids_same:
        return {
            "experiment_verdict": "ACCEPT",
            "selector_verdict": "EQUIVALENT_STRONGEST_GRAPH_AND_OUTPUT",
            "reason": "swiglu and 1 selected the same observed dispatch/kernel identity and generated ids",
        }
    if not graph_same:
        return {
            "experiment_verdict": "REJECT",
            "selector_verdict": "GRAPH_CHANGED",
            "reason": "the two source-approved spellings did not select the same observed graph",
        }
    return {
        "experiment_verdict": "INCONCLUSIVE",
        "selector_verdict": "OUTPUT_CHANGED_OR_UNAVAILABLE",
        "reason": "graph identity matched but generated output did not match",
    }


def run_qwen27_mlp_diagnostic_ab(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    profile: Optional[str | os.PathLike[str]] = None,
    resident_binary: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    timeout_s: float = 180.0,
) -> Dict[str, Any]:
    repo = _repo_root(repo_root)
    profile_path = _profile_path(repo, profile)
    destination = Path(emit).expanduser() if emit else repo / "receipts" / "headless" / DEFAULT_EMIT_NAME
    if not destination.is_absolute():
        destination = repo / destination
    started = time.time()
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "repo_root": str(repo),
        "profile_path": str(profile_path),
        "experiment": {
            "id": "qwen27-mlp-selector-swiglu-vs-1",
            "model_scope": "current profile-driven resident only",
            "hypothesis": "HAWKING_QWEN38_FUSE_MLP=1 is an alias for the strongest swiglu fusion, not the unfused graph",
            "control": "HAWKING_QWEN38_FUSE_MLP=swiglu",
            "mutation": "HAWKING_QWEN38_FUSE_MLP=1",
            "prompt": PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "one_experiment_at_a_time": True,
            "source_mutation": False,
            "profile_mutation": False,
        },
        "source_truth": {
            "accepted_off": ["unset", "", "0", "off", "false", "no"],
            "accepted_pair": ["pair", "gate_up"],
            "accepted_strongest": ["swiglu", "gate_up_swiglu", "1", "true", "on", "yes"],
            "unknown_value": "PANIC_BEFORE_GRAPH_SELECTION",
            "source_of_truth": "crates/hawking-core/src/model/qwen38_hybrid_decode.rs::Qwen38MlpFusion::from_env",
        },
        "benchmark_policy": {
            "protected_class": QUALIFIED_PROTECTED,
            "diagnostic_class": DIAGNOSTIC_CONTAMINATED,
            "diagnostic_qualification": False,
            "diagnostic_NOT_FOR_PROMOTION": True,
        },
    }
    arms: list[Dict[str, Any]] = []
    try:
        from hcli.hawking_native import HawkingNativeConfig

        base = HawkingNativeConfig.from_file(str(profile_path))
        base.validate()
        binary = str(Path(resident_binary).expanduser().resolve()) if resident_binary else None
        if binary and not Path(binary).is_file():
            raise FileNotFoundError(binary)
        report["base_identity"] = _safe(base.identity())
        report["source_graph_by_arm"] = {
            value: _safe(_selected_graph({**base.fusion_env, "HAWKING_QWEN38_FUSE_MLP": value}, repo))
            for value in MLP_ARMS
        }
        for value in MLP_ARMS:
            arm_config = _arm_config(base, value, binary)
            arm_config.validate()
            arms.append(_run_arm(arm_config, value, timeout_s=max(0.1, float(timeout_s))))
        overall = _overall_boundary(arms)
        verdict = _verdict(arms)
        report["arms"] = arms
        report.update(overall)
        report.update(verdict)
        report["machine"] = {
            "platform": platform.platform(),
            "architecture": platform.machine(),
        }
        report["checks"] = {
            "profile_valid": True,
            "exactly_two_registered_arms": [arm.get("arm") for arm in arms] == list(MLP_ARMS),
            "both_arms_attempted": len(arms) == 2,
            "source_truth_recorded": all(report["source_truth"].values()),
            "native_metrics_and_absence_reasons_recorded": all(
                isinstance(arm.get("live"), Mapping)
                and "metric_absence_reasons" in arm["live"]
                for arm in arms
            ),
            "benchmark_class_recorded_per_arm": all(
                arm.get("benchmark_class") in {QUALIFIED_PROTECTED, DIAGNOSTIC_CONTAMINATED}
                for arm in arms
            ),
            "diagnostic_runs_not_for_promotion": all(
                arm.get("benchmark_class") != DIAGNOSTIC_CONTAMINATED
                or (arm.get("qualification") is False and arm.get("NOT_FOR_PROMOTION") is True)
                for arm in arms
            ),
            "no_source_or_profile_mutation": True,
        }
        report["status"] = "PASSED" if all(report["checks"].values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001 - make the experiment failure durable
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        report["arms"] = arms
        report.update(_overall_boundary(arms))
        report.update(_verdict(arms))
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    report["receipt_path"] = str(destination.resolve())
    report["claim_boundary"] = "The A/B can resolve parser and graph identity. A DIAGNOSTIC_CONTAMINATED result is not a protected performance qualification; even a protected selector result is not Flash or Qwen promotion evidence."
    atomic_write_json(destination, report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--profile")
    parser.add_argument("--resident-binary")
    parser.add_argument("--emit")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args(argv)
    report = run_qwen27_mlp_diagnostic_ab(
        repo_root=args.repo_root,
        profile=args.profile,
        resident_binary=args.resident_binary,
        emit=args.emit,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["SCHEMA", "run_qwen27_mlp_diagnostic_ab", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
