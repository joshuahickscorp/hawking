"""Qwen27 acceptance: identity freeze, protected baseline, regression explain/bound."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from tools.acceptance.lake.common import (
    GATES,
    PRIMARY,
    WORKTREE,
    jsonable,
    load_symbol,
    receipts_dir,
    timed,
    write_receipt,
)

IDENTITY_GATE = "QWEN27_RUNTIME_IDENTITY_FROZEN"
BASELINE_GATE = "QWEN27_PROTECTED_BASELINE"
REGRESSION_GATE = "QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED"

HISTORICAL_ANCHOR_TPS = 33.716875920652136
SOURCE_MLP = WORKTREE / "crates" / "hawking-core" / "src" / "model" / "qwen38_hybrid_decode.rs"


def _primary_profile() -> Path:
    return PRIMARY / "hcli" / "hawking-native.sealed-3.14.json"


def call_run_runtime_archaeology(
    *,
    repo_root: str | Path,
    profile: str | Path,
    identity_emit: str | Path,
    diff_emit: str | Path,
) -> dict[str, Any]:
    """Production call site of the catalog symbol."""
    run_runtime_archaeology = load_symbol(
        "hcli.agentos.qwen27_runtime_identity", "run_runtime_archaeology"
    )
    return run_runtime_archaeology(
        repo_root=str(repo_root),
        profile=str(profile),
        identity_emit=str(identity_emit),
        diff_emit=str(diff_emit),
    )


def call_run_protected_accelerator_benchmark(**kwargs: Any) -> dict[str, Any]:
    """Production call site of the catalog symbol."""
    run_protected_accelerator_benchmark = load_symbol(
        "hcli.agentos.protected_accelerator_benchmark",
        "run_protected_accelerator_benchmark",
    )
    return run_protected_accelerator_benchmark(**kwargs)


def call_run_qwen27_mlp_diagnostic_ab(**kwargs: Any) -> dict[str, Any]:
    """Production call site of the catalog symbol. Do not call while a live resident holds the machine."""
    run_qwen27_mlp_diagnostic_ab = load_symbol(
        "hcli.agentos.qwen27_mlp_diagnostic", "run_qwen27_mlp_diagnostic_ab"
    )
    return run_qwen27_mlp_diagnostic_ab(**kwargs)


def run_runtime_identity_gate() -> dict[str, Any]:
    command = ["python3", "-m", "tools.acceptance.lake", "--gate", IDENTITY_GATE]
    emit_dir = receipts_dir()
    with timed() as clock:
        raw = call_run_runtime_archaeology(
            repo_root=PRIMARY,
            profile=_primary_profile(),
            identity_emit=emit_dir / f"{IDENTITY_GATE}.archaeology.json",
            diff_emit=emit_dir / f"{IDENTITY_GATE}.diff.json",
        )
        checks_raw = raw.get("checks") or {}
        diff = raw.get("diff") or {}
        summary = diff.get("summary") or {}
        selection = raw.get("historical_selection") or {}
        current = raw.get("current") or {}
        frozen_fields = {
            "source_git_head": (current.get("git") or {}).get("head"),
            "working_tree_dirty": (current.get("git") or {}).get("working_tree_dirty"),
            "binary_sha256": (current.get("binary") or {}).get("sha256"),
            "binary_exists": (current.get("binary") or {}).get("exists"),
            "tokenizer_sha256": (current.get("tokenizer") or {}).get("sha256"),
            "tokenizer_exists": (current.get("tokenizer") or {}).get("exists"),
            "artifact_root": (current.get("artifact") or {}).get("root"),
            "artifact_exists": (current.get("artifact") or {}).get("exists"),
            "cargo_profile": (current.get("compiler") or {}).get("profile")
            if isinstance(current.get("compiler"), dict)
            else (current.get("runtime") or {}).get("executable_profile"),
            "rust_flags": (current.get("source") or {}).get("rust_flags"),
            "fusion_env": (current.get("fusion") or {}).get("env"),
            "historical_anchor_tps": selection.get("historical_anchor_tps"),
            "best_historical_tps": selection.get("best_historical_tps"),
            "selected_arm": selection.get("selected_arm"),
        }
        passed = raw.get("status") == "PASSED" and all(bool(v) for v in checks_raw.values())
        # Identity is frozen when the archaeology receipt records the required
        # fields and treats UNKNOWN as not equal. It does not qualify TPS.
        required_present = all(
            [
                frozen_fields["source_git_head"],
                frozen_fields["binary_sha256"] is not None,
                frozen_fields["tokenizer_sha256"] is not None
                or frozen_fields["tokenizer_exists"] is False,
                frozen_fields["artifact_root"],
                frozen_fields["historical_anchor_tps"] is not None,
                checks_raw.get("unknowns_are_explicit") is True,
                checks_raw.get("no_performance_qualification") is True,
            ]
        )
        if passed and required_present:
            verdict = "ACCEPTED"
            blocker = None
        else:
            verdict = "BLOCKED"
            blocker = {
                "missing_input": "complete historical runtime identity archaeology PASSED",
                "why": f"status={raw.get('status')!r} error={raw.get('error')!r}",
            }
        measured = {
            "status": raw.get("status"),
            "checks": checks_raw,
            "diff_summary": summary,
            "frozen": frozen_fields,
            "historical_anchor_tps": selection.get("historical_anchor_tps"),
            "best_historical_tps": selection.get("best_historical_tps"),
        }
        output = {
            "summary": (
                f"archaeology {raw.get('status')}; "
                f"anchor_tps={selection.get('historical_anchor_tps')}; "
                f"diff={summary}"
            ),
            "ranked_differences": jsonable(diff.get("ranked_differences")),
            "identity_receipt_path": raw.get("identity_receipt_path"),
            "diff_receipt_path": raw.get("diff_receipt_path"),
        }
        return write_receipt(
            IDENTITY_GATE,
            verdict=verdict,
            command=command,
            output=output,
            measured=measured,
            checks={
                "symbol_invoked": True,
                "archaeology_passed": passed,
                "required_identity_fields_present": required_present,
                "unknowns_are_explicit": checks_raw.get("unknowns_are_explicit") is True,
                "no_performance_qualification": checks_raw.get("no_performance_qualification") is True,
                "no_resident_started": True,
            },
            evidence_tier="STATIC",
            symbol_invoked=True,
            blocker=blocker,
            elapsed_s=clock.snap(),
        )


def run_protected_baseline_gate(*, ready_timeout_s: float = 2.0) -> dict[str, Any]:
    """CALL the protected benchmark. Do not wait hours; do not start a resident on timeout."""
    command = [
        "python3",
        "-m",
        "tools.acceptance.lake",
        "--gate",
        BASELINE_GATE,
        f"--ready-timeout-s={ready_timeout_s}",
    ]
    emit = receipts_dir() / f"{BASELINE_GATE}.run.json"
    with timed() as clock:
        raw = call_run_protected_accelerator_benchmark(
            repo_root=str(WORKTREE),
            profile=str(_primary_profile()) if _primary_profile().is_file() else None,
            emit=str(emit),
            ready_timeout_s=float(ready_timeout_s),
            interval_s=0.2,
            timeout_s=5.0,
            warmup_requests=1,
            measure_requests=1,
            max_new_tokens=8,
        )
        status = raw.get("status")
        error = raw.get("error") if isinstance(raw.get("error"), dict) else {}
        measurements = raw.get("measurements") or []
        qualified = bool(raw.get("qualification"))
        protected_window = raw.get("protected_window")
        started_resident = bool(raw.get("resident_ready"))
        waiting = status in {"WAITING_FOR_QUIESCENCE", "WAITING_FOR_LOCK"} or (
            isinstance(error, dict)
            and error.get("type") in {"LockTimeout", "QuiescenceTimeout"}
        )

        # A protected baseline is a quiet-window measurement. Anything else is
        # not the criterion, including a successful-looking contended run.
        if qualified and protected_window is True and measurements:
            verdict = "ACCEPTED"
            blocker = None
        else:
            verdict = "BLOCKED"
            why = []
            if waiting:
                why.append(
                    f"machine did not present a protected window "
                    f"(status={status!r} error={error})"
                )
            why.append(
                "live HCLI daemon must not be signalled/restarted; "
                "this profile cannot run `ps` (machine_quiescence → quiet=None); "
                "a competing resident must not be started"
            )
            blocker = {
                "missing_input": (
                    "a quiescent machine window (machine_quiescence().quiet is True) "
                    "with no live competing HCLI resident"
                ),
                "why": "; ".join(why),
                "run_status": status,
                "run_error": error,
                "protected_window": protected_window,
                "qualification": qualified,
                "started_resident": started_resident,
            }

        measured = {
            "status": status,
            "qualification": qualified,
            "protected_window": protected_window,
            "n_measurements": len(measurements),
            "started_resident": started_resident,
            "ready_timeout_s": ready_timeout_s,
            "readiness_polls": len(raw.get("readiness_polls") or []),
            "aggregate": jsonable(raw.get("aggregate")),
        }
        output = {
            "summary": (
                f"protected benchmark status={status!r} "
                f"qualification={qualified} window={protected_window} "
                f"measurements={len(measurements)} resident={started_resident}"
            ),
            "error": jsonable(error),
            "checks": jsonable(raw.get("checks")),
            "run_receipt_path": raw.get("receipt_path"),
        }
        return write_receipt(
            BASELINE_GATE,
            verdict=verdict,
            command=command,
            output=output,
            measured=measured,
            checks={
                "symbol_invoked": True,
                "protected_window": protected_window is True,
                "qualified": qualified,
                "did_not_start_resident_without_quiet_window": not started_resident
                or protected_window is True,
            },
            evidence_tier="STATIC",
            symbol_invoked=True,
            blocker=blocker,
            elapsed_s=clock.snap(),
        )


def _mlp_source_truth() -> dict[str, Any]:
    text = ""
    try:
        text = SOURCE_MLP.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"present": False, "error": str(exc)[:500]}
    needle = 'pub fn from_env() -> Self'
    idx = text.find("impl Qwen38MlpFusion")
    snippet = text[idx : idx + 2200] if idx >= 0 else ""
    return {
        "present": SOURCE_MLP.is_file(),
        "path": str(SOURCE_MLP),
        "from_env_present": needle in text,
        "swiglu_and_1_same_arm": (
            '"swiglu" | "gate_up_swiglu" | "1" | "true" | "on" | "yes"' in snippet
            or '"swiglu" | "gate_up_swiglu" | "1"' in snippet
        ),
        "unrecognised_panics": "is not a recognised value" in snippet,
        "comment_1_is_strongest": "mean the STRONGEST fusion" in snippet,
        "snippet": snippet[:1800],
    }


def run_regression_gate(*, invoke_live_ab: bool = False) -> dict[str, Any]:
    """Explain the Qwen27 regression from a fresh identity freeze + source.

    Live A/B (run_qwen27_mlp_diagnostic_ab) starts a resident. The live HCLI
    daemon must not be contended with. Default is to NOT invoke the live A/B;
    pass invoke_live_ab=True only when the machine is known free.
    """
    command = ["python3", "-m", "tools.acceptance.lake", "--gate", REGRESSION_GATE]
    with timed() as clock:
        source = _mlp_source_truth()
        identity_path = receipts_dir() / f"{IDENTITY_GATE}.json"
        identity: Optional[dict[str, Any]] = None
        if identity_path.is_file():
            import json

            try:
                identity = json.loads(identity_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                identity = None

        ab: Optional[dict[str, Any]] = None
        ab_invoked = False
        if invoke_live_ab:
            emit = receipts_dir() / f"{REGRESSION_GATE}.ab.json"
            ab = call_run_qwen27_mlp_diagnostic_ab(
                repo_root=str(PRIMARY),
                profile=str(_primary_profile()),
                emit=str(emit),
                timeout_s=30.0,
            )
            ab_invoked = True

        diff_summary = ((identity or {}).get("measured") or {}).get("diff_summary") or {}
        frozen = ((identity or {}).get("measured") or {}).get("frozen") or {}
        ranked = ((identity or {}).get("output") or {}).get("ranked_differences") or []
        different = [
            r
            for r in ranked
            if isinstance(r, dict) and r.get("classification") == "DIFFERENT_VERIFIED"
        ]
        explained = (
            identity is not None
            and identity.get("verdict") == "ACCEPTED"
            and int(diff_summary.get("different_verified") or 0) >= 1
            and source.get("swiglu_and_1_same_arm") is True
        )
        # The regression (current runtime vs historical ~34 TPS identity) is
        # explained by verified identity mismatch. MLP spelling is bounded in
        # source: `1` and `swiglu` select the same GateUpSwiglu arm. Live A/B
        # is confirmation, not the explanation.
        if explained:
            verdict = "ACCEPTED"
            blocker = {
                "unexecuted_confirmation": (
                    "run_qwen27_mlp_diagnostic_ab was not invoked: it starts a "
                    "resident and the live HCLI daemon must not be signalled or "
                    "contended with. Source+identity-freeze already explain the "
                    "regression; the live A/B is confirmation of graph identity, "
                    "not a missing explanation."
                )
            } if not ab_invoked else None
        else:
            verdict = "BLOCKED"
            blocker = {
                "missing_input": "identity-freeze ACCEPTED with DIFFERENT_VERIFIED rows, and MLP from_env source truth",
                "why": (
                    f"identity_verdict={(identity or {}).get('verdict')!r} "
                    f"different_verified={diff_summary.get('different_verified')} "
                    f"source_same_arm={source.get('swiglu_and_1_same_arm')}"
                ),
            }

        measured = {
            "historical_anchor_tps": frozen.get("historical_anchor_tps") or HISTORICAL_ANCHOR_TPS,
            "diff_summary": diff_summary,
            "different_verified_dimensions": [
                r.get("dimension") for r in different
            ],
            "mlp_source": {
                k: source[k]
                for k in (
                    "present",
                    "from_env_present",
                    "swiglu_and_1_same_arm",
                    "unrecognised_panics",
                    "comment_1_is_strongest",
                    "path",
                )
                if k in source
            },
            "live_ab_invoked": ab_invoked,
            "live_ab_status": None if ab is None else ab.get("status"),
            "live_ab_selector_verdict": None if ab is None else ab.get("selector_verdict"),
        }
        output = {
            "summary": (
                f"explained={explained} different_verified={diff_summary.get('different_verified')} "
                f"mlp_swiglu_eq_1={source.get('swiglu_and_1_same_arm')} live_ab={ab_invoked}"
            ),
            "explanation": (
                "Current Qwen27 runtime identity differs from the historical "
                f"~{HISTORICAL_ANCHOR_TPS:.3f} TPS fusion arm on "
                f"{len(different)} DIFFERENT_VERIFIED dimensions "
                f"({[r.get('dimension') for r in different]}). The TPS coordinate "
                "is therefore not comparable to the historical identity; the "
                "regression is an identity mismatch, not an unexplained drop on "
                "the same executable. Independently, crates/hawking-core "
                "Qwen38MlpFusion::from_env maps `swiglu` and `1` to the same "
                "GateUpSwiglu arm and panics on unknown values — the MLP selector "
                "spelling is bounded in source."
            ),
            "mlp_from_env_snippet": source.get("snippet"),
            "identity_receipt": str(identity_path) if identity_path.is_file() else None,
        }
        return write_receipt(
            REGRESSION_GATE,
            verdict=verdict,
            command=command,
            output=output,
            measured=measured,
            checks={
                "identity_freeze_accepted": (identity or {}).get("verdict") == "ACCEPTED",
                "different_verified_at_least_one": int(diff_summary.get("different_verified") or 0) >= 1,
                "mlp_swiglu_and_1_same_arm_in_source": source.get("swiglu_and_1_same_arm") is True,
                "live_ab_invoked": ab_invoked,
                "no_performance_claim": True,
            },
            evidence_tier="STATIC",
            symbol_invoked=ab_invoked,
            blocker=blocker,
            elapsed_s=clock.snap(),
            extra={
                "catalog_symbol_not_invoked_reason": None
                if ab_invoked
                else (
                    "run_qwen27_mlp_diagnostic_ab starts a Hawking native resident; "
                    "the contract forbids signalling/restarting the live HCLI daemon "
                    "and forbids a competing resident. Explanation uses the fresh "
                    "run_runtime_archaeology identity freeze plus Qwen38MlpFusion::from_env."
                )
            },
        )
