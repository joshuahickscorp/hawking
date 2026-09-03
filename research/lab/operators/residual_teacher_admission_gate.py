#!/usr/bin/env python3.12
"""Local-only residual teacher admission gate (Kimi after GLM Proto).

Fail-closed decision gate for whether a *second* teacher (Kimi) may be added
as a residual lane after the GLM-to-DSV4F Proto stage.

This module validates supplied JSON-like evidence only.  It does **not**:

- network, call Hub/Xet, download, or write caches
- load models or launch trainers / subprocesses
- promote Proto to Final or claim causal inheritance proof

Outcomes (single string):

- ``DEFERRED`` â default; missing or incomplete evidence
- ``REJECT``   â hard requirement failed (mismatch, regression, no gain, â¦)
- ``ADMIT``    â every required check passed (residual-lane admission only)

``ADMIT`` is permission to *start* an evidence-gated residual Kimi lane, not a
claim that Kimi caused a capability, and not a duplicate of the full GLM
transfer programme.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping, Sequence

from lab.receipts import seal


ADMISSION_SCHEMA = "hawking.residual_teacher.admission_gate.v1"

VERDICT_DEFERRED = "DEFERRED"
VERDICT_REJECT = "REJECT"
VERDICT_ADMIT = "ADMIT"

PROTECTED_AXES: tuple[str, ...] = (
    "math",
    "coding",
    "tool",
    "agentic",
)

# Phrases that are too generic to justify a second teacher residual lane.
_GENERIC_HYPOTHESIS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*more\s+distillation\s*$",
        r"^\s*additional\s+distillation\s*$",
        r"^\s*extra\s+distillation\s*$",
        r"^\s*generic\s+distillation\s*$",
        r"^\s*distillation\s*$",
        r"^\s*more\s+teacher\s*$",
        r"^\s*add\s+kimi\s*$",
        r"^\s*second\s+teacher\s*$",
        r"^\s*more\s+transfer\s*$",
    )
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_READY_STATUSES = frozenset(
    {
        "READY",
        "OPEN",
        "ADMITTED",
        "PASS",
        "PASSED",
        "COMPLETE",
        "ARCHITECTURE_READY",
        "FORWARD_READY",
    }
)
_SEALED_MARKERS = frozenset(
    {
        "SEALED",
        "SEALED_BASELINE",
        "GLM_ONLY_BASELINE_SEALED",
        "BASELINE_SEALED",
        "PASS_SEALED",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _nonempty_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _looks_sha256(value: str) -> bool:
    return bool(_SHA256_RE.match(value.lower()))


def _check(
    checks: list[dict[str, Any]],
    name: str,
    status: str,
    detail: str,
) -> None:
    checks.append({"name": name, "status": status, "detail": detail})


def _is_generic_hypothesis(text: str) -> bool:
    return any(p.match(text) for p in _GENERIC_HYPOTHESIS_PATTERNS)


def _receipt_sealed(receipt: Mapping[str, Any]) -> bool:
    if receipt.get("sealed") is True:
        return True
    status = _nonempty_str(receipt.get("status"))
    if status is not None and status.upper() in _SEALED_MARKERS:
        return True
    seal_hex = _nonempty_str(receipt.get("seal_sha256"))
    if seal_hex is not None and _looks_sha256(seal_hex):
        return True
    return False


def _is_glm_only_teacher(receipt: Mapping[str, Any]) -> bool:
    teacher = _nonempty_str(receipt.get("teacher"))
    if teacher is not None:
        token = teacher.lower().replace("_", "-")
        if token in {"glm", "glm-only", "glm5.2", "glm-5.2", "glm52"}:
            return True
        if "kimi" in token:
            return False
    teachers = receipt.get("teachers")
    if isinstance(teachers, Sequence) and not isinstance(teachers, (str, bytes)):
        tokens = [
            str(t).strip().lower().replace("_", "-")
            for t in teachers
            if str(t).strip()
        ]
        if tokens and all(t in {"glm", "glm-only", "glm5.2", "glm-5.2", "glm52"} for t in tokens):
            return True
        if any("kimi" in t for t in tokens):
            return False
    baseline_kind = _nonempty_str(receipt.get("baseline_kind"))
    if baseline_kind is not None and baseline_kind.lower() in {
        "glm_only",
        "glm-only",
        "proto_glm_only",
    }:
        return True
    return False


def _membership_hash(receipt: Mapping[str, Any]) -> str | None:
    for key in (
        "held_out_membership_hash",
        "membership_hash",
        "heldout_membership_hash",
    ):
        value = _nonempty_str(receipt.get(key))
        if value is not None:
            return value.lower() if _looks_sha256(value) else value
    membership = _as_mapping(receipt.get("membership"))
    if membership is not None:
        value = _nonempty_str(membership.get("hash") or membership.get("held_out_hash"))
        if value is not None:
            return value.lower() if _looks_sha256(value) else value
    return None


def _incremental_delta(receipt: Mapping[str, Any]) -> float | None:
    """Return held-out incremental improvement if present; None if absent."""

    for key in (
        "incremental_held_out_delta",
        "held_out_delta",
        "incremental_delta",
        "delta_held_out",
    ):
        raw = receipt.get(key)
        if isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str) and raw.strip():
            try:
                return float(raw.strip())
            except ValueError:
                continue

    comparison = _as_mapping(receipt.get("comparison") or receipt.get("ab"))
    if comparison is None:
        return None

    delta = comparison.get("delta")
    if isinstance(delta, (int, float)) and not isinstance(delta, bool):
        return float(delta)
    if isinstance(delta, str) and delta.strip():
        try:
            return float(delta.strip())
        except ValueError:
            pass

    baseline = comparison.get("baseline")
    with_kimi = comparison.get("with_kimi") or comparison.get("kimi") or comparison.get("treatment")
    if isinstance(baseline, (int, float)) and isinstance(with_kimi, (int, float)):
        if not isinstance(baseline, bool) and not isinstance(with_kimi, bool):
            return float(with_kimi) - float(baseline)

    baseline_m = _as_mapping(baseline)
    kimi_m = _as_mapping(with_kimi)
    if baseline_m is not None and kimi_m is not None:
        b_score = baseline_m.get("held_out_score")
        if b_score is None:
            b_score = baseline_m.get("score")
        k_score = kimi_m.get("held_out_score")
        if k_score is None:
            k_score = kimi_m.get("score")
        if isinstance(b_score, (int, float)) and isinstance(k_score, (int, float)):
            if not isinstance(b_score, bool) and not isinstance(k_score, bool):
                return float(k_score) - float(b_score)
    return None


def _axis_regressed(row: Any) -> bool | None:
    """Return True if regressed, False if held, None if unknown/missing."""

    if row is None:
        return None
    if isinstance(row, bool):
        # bare False means "not regressed" is ambiguous; treat True as regressed.
        return True if row else False
    if isinstance(row, (int, float)) and not isinstance(row, bool):
        return float(row) < 0.0
    if not isinstance(row, Mapping):
        return None
    if row.get("regressed") is True or row.get("regression") is True:
        return True
    if row.get("regressed") is False or row.get("regression") is False:
        return False
    gate = _nonempty_str(row.get("gate") or row.get("status") or row.get("verdict"))
    if gate is not None:
        token = gate.upper()
        if token in {"FAIL", "FAILED", "REGRESSED", "REJECT"}:
            return True
        if token in {"PASS", "PASSED", "HOLD", "HELD", "OK", "NON_REGRESSED"}:
            return False
    delta = row.get("delta")
    if isinstance(delta, (int, float)) and not isinstance(delta, bool):
        return float(delta) < 0.0
    return None


def _check_glm_baseline(
    checks: list[dict[str, Any]],
    receipt: Mapping[str, Any] | None,
) -> str | None:
    """Validate GLM-only sealed baseline. Returns membership hash on PASS."""

    if receipt is None:
        _check(
            checks,
            "glm_only_baseline_receipt",
            "DEFER",
            "sealed GLM-only baseline receipt not supplied",
        )
        return None

    if not _receipt_sealed(receipt):
        _check(
            checks,
            "glm_only_baseline_receipt",
            "FAIL",
            "baseline receipt is not sealed (need sealed=true, SEALED status, or seal_sha256)",
        )
        return None

    if not _is_glm_only_teacher(receipt):
        _check(
            checks,
            "glm_only_baseline_receipt",
            "FAIL",
            "baseline must be GLM-only (teacher=glm / baseline_kind=glm_only); Kimi must not be in baseline",
        )
        return None

    membership = _membership_hash(receipt)
    if membership is None:
        _check(
            checks,
            "glm_only_baseline_receipt",
            "FAIL",
            "sealed GLM baseline missing held_out_membership_hash",
        )
        return None

    _check(
        checks,
        "glm_only_baseline_receipt",
        "PASS",
        f"sealed GLM-only baseline with membership_hash={membership[:12]}â¦",
    )
    return membership


def _check_kimi_incremental(
    checks: list[dict[str, Any]],
    receipt: Mapping[str, Any] | None,
    baseline_membership: str | None,
) -> None:
    if receipt is None:
        _check(
            checks,
            "kimi_incremental_ab_receipt",
            "DEFER",
            "Kimi incremental A/B receipt not supplied",
        )
        _check(
            checks,
            "held_out_membership_match",
            "DEFER",
            "cannot compare membership without Kimi incremental receipt",
        )
        _check(
            checks,
            "incremental_held_out_improvement",
            "DEFER",
            "no incremental held-out delta available",
        )
        return

    kimi_membership = _membership_hash(receipt)
    if baseline_membership is None:
        _check(
            checks,
            "held_out_membership_match",
            "DEFER",
            "baseline membership unavailable; cannot verify same held-out set",
        )
    elif kimi_membership is None:
        _check(
            checks,
            "held_out_membership_match",
            "FAIL",
            "Kimi incremental receipt missing held_out_membership_hash",
        )
    elif kimi_membership != baseline_membership:
        _check(
            checks,
            "held_out_membership_match",
            "FAIL",
            "evaluation membership differs between GLM baseline and Kimi A/B "
            f"(baseline={baseline_membership[:12]}â¦ kimi={kimi_membership[:12]}â¦)",
        )
    else:
        _check(
            checks,
            "held_out_membership_match",
            "PASS",
            "Kimi A/B uses the same held-out membership hash as the GLM baseline",
        )

    # Confirm this is an incremental Kimi addition claim, not a second baseline.
    teacher = _nonempty_str(receipt.get("teacher") or receipt.get("added_teacher"))
    treatment = _nonempty_str(receipt.get("treatment") or receipt.get("arm"))
    is_kimi = False
    for token in (teacher, treatment):
        if token is not None and "kimi" in token.lower():
            is_kimi = True
    if receipt.get("includes_kimi") is True or receipt.get("kimi_added") is True:
        is_kimi = True
    if _as_mapping(receipt.get("comparison") or receipt.get("ab")) is not None:
        # A/B structure is the intended incremental form.
        is_kimi = True
    if not is_kimi and receipt.get("incremental_held_out_delta") is None:
        _check(
            checks,
            "kimi_incremental_ab_receipt",
            "FAIL",
            "receipt does not identify a Kimi incremental A/B (need teacher/treatment kimi, "
            "includes_kimi, or comparison/ab block)",
        )
    else:
        _check(
            checks,
            "kimi_incremental_ab_receipt",
            "PASS",
            "Kimi incremental A/B receipt structure present",
        )

    delta = _incremental_delta(receipt)
    if delta is None:
        _check(
            checks,
            "incremental_held_out_improvement",
            "FAIL",
            "incremental held-out improvement is absent (no delta / comparison scores)",
        )
    elif delta <= 0.0:
        _check(
            checks,
            "incremental_held_out_improvement",
            "FAIL",
            f"incremental held-out improvement not positive (delta={delta})",
        )
    else:
        _check(
            checks,
            "incremental_held_out_improvement",
            "PASS",
            f"held-out incremental delta={delta} > 0 (associative evidence only; not causal proof)",
        )


def _check_residual_hypothesis(
    checks: list[dict[str, Any]],
    hypothesis: Mapping[str, Any] | None,
) -> None:
    if hypothesis is None:
        _check(
            checks,
            "named_residual_hypothesis",
            "DEFER",
            "named residual hypothesis/capability not supplied",
        )
        return

    name = _nonempty_str(
        hypothesis.get("name")
        or hypothesis.get("capability")
        or hypothesis.get("hypothesis")
        or hypothesis.get("residual_capability")
    )
    if name is None:
        _check(
            checks,
            "named_residual_hypothesis",
            "FAIL",
            "residual hypothesis present but name/capability is empty",
        )
        return

    if _is_generic_hypothesis(name):
        _check(
            checks,
            "named_residual_hypothesis",
            "FAIL",
            f"residual hypothesis {name!r} is generic distillation language; "
            "require a named residual capability (e.g. long-horizon agentic planning)",
        )
        return

    # Optional explicit flag that this is residual, not a full re-transfer.
    role = _nonempty_str(hypothesis.get("role") or hypothesis.get("lane"))
    if role is not None and role.lower() in {
        "duplicate_transfer",
        "full_transfer",
        "replace_glm",
    }:
        _check(
            checks,
            "named_residual_hypothesis",
            "FAIL",
            f"hypothesis role {role!r} is not a residual lane",
        )
        return

    _check(
        checks,
        "named_residual_hypothesis",
        "PASS",
        f"named residual capability: {name}",
    )


def _check_provenance(
    checks: list[dict[str, Any]],
    provenance: Mapping[str, Any] | None,
) -> None:
    if provenance is None:
        _check(
            checks,
            "provenance_revision_identity",
            "DEFER",
            "provenance/revision identity not supplied",
        )
        return

    required_groups = (
        (
            "student_revision",
            ("student_revision", "dsv4f_revision", "student_identity", "architecture_revision"),
        ),
        (
            "glm_baseline_revision",
            ("glm_revision", "baseline_revision", "glm_baseline_revision", "proto_revision"),
        ),
        (
            "kimi_revision",
            ("kimi_revision", "teacher_revision", "kimi_identity", "second_teacher_revision"),
        ),
    )
    missing: list[str] = []
    present: dict[str, str] = {}
    for label, keys in required_groups:
        value = None
        for key in keys:
            value = _nonempty_str(provenance.get(key))
            if value is not None:
                break
        if value is None:
            missing.append(label)
        else:
            present[label] = value

    if missing:
        _check(
            checks,
            "provenance_revision_identity",
            "FAIL",
            f"provenance incomplete; missing {missing}",
        )
        return

    if provenance.get("complete") is False:
        _check(
            checks,
            "provenance_revision_identity",
            "FAIL",
            "provenance.complete=false",
        )
        return

    _check(
        checks,
        "provenance_revision_identity",
        "PASS",
        "student/glm/kimi revision identities bound: "
        + ", ".join(f"{k}={v[:16]}" for k, v in present.items()),
    )


def _check_no_regression(
    checks: list[dict[str, Any]],
    no_regression: Mapping[str, Any] | None,
) -> None:
    if no_regression is None:
        _check(
            checks,
            "protected_no_regression",
            "DEFER",
            "explicit no-regression checks not supplied "
            f"(required axes: {', '.join(PROTECTED_AXES)})",
        )
        return

    axes_map = _as_mapping(no_regression.get("axes") or no_regression.get("protected"))
    source: Mapping[str, Any] = axes_map if axes_map is not None else no_regression

    regressed: list[str] = []
    missing: list[str] = []
    held: list[str] = []
    for axis in PROTECTED_AXES:
        row = source.get(axis)
        state = _axis_regressed(row)
        if state is None:
            missing.append(axis)
        elif state:
            regressed.append(axis)
        else:
            held.append(axis)

    if regressed:
        _check(
            checks,
            "protected_no_regression",
            "FAIL",
            f"protected axis regression: {regressed} "
            "(math/coding/tool/agentic must not regress)",
        )
        return
    if missing:
        _check(
            checks,
            "protected_no_regression",
            "FAIL" if held else "DEFER",
            f"protected axes incomplete; missing {missing}; held={held}",
        )
        return

    _check(
        checks,
        "protected_no_regression",
        "PASS",
        f"no regression on protected axes: {list(PROTECTED_AXES)}",
    )


def _check_dsv4f_forward(
    checks: list[dict[str, Any]],
    forward_gate: Mapping[str, Any] | None,
) -> None:
    if forward_gate is None:
        _check(
            checks,
            "dsv4f_architecture_forward_ready",
            "DEFER",
            "DSV4F architecture/forward gate evidence not supplied",
        )
        return

    if forward_gate.get("ready") is True:
        _check(
            checks,
            "dsv4f_architecture_forward_ready",
            "PASS",
            "DSV4F architecture/forward gate ready=true",
        )
        return

    status = _nonempty_str(
        forward_gate.get("status")
        or forward_gate.get("gate")
        or forward_gate.get("forward_status")
        or forward_gate.get("architecture_status")
    )
    if status is not None and status.upper() in _READY_STATUSES:
        _check(
            checks,
            "dsv4f_architecture_forward_ready",
            "PASS",
            f"DSV4F architecture/forward gate status={status}",
        )
        return

    if forward_gate.get("ready") is False or (
        status is not None
        and status.upper()
        in {
            "PENDING",
            "DEEPSEEK_FORWARD_PENDING",
            "NOT_READY",
            "CLOSED",
            "BLOCKED",
            "FAIL",
            "FAILED",
        }
    ):
        _check(
            checks,
            "dsv4f_architecture_forward_ready",
            "FAIL",
            f"DSV4F architecture/forward gate not ready ({status or 'ready=false'})",
        )
        return

    _check(
        checks,
        "dsv4f_architecture_forward_ready",
        "FAIL",
        "DSV4F architecture/forward gate present but not in a ready state",
    )


def evaluate_residual_teacher_admission(
    evidence: Mapping[str, Any] | None = None,
    *,
    glm_baseline_receipt: Mapping[str, Any] | None = None,
    kimi_incremental_receipt: Mapping[str, Any] | None = None,
    residual_hypothesis: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    no_regression: Mapping[str, Any] | None = None,
    dsv4f_architecture_forward: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether Kimi may be admitted as a residual second teacher.

    Parameters may be passed as a single ``evidence`` mapping or as keyword
    fragments.  Keyword arguments override keys from ``evidence``.

    Returns a sealed decision document with ``verdict`` â
    {DEFERRED, REJECT, ADMIT} and structured ``checks`` / ``reasons``.
    """

    bundle: MutableMapping[str, Any] = dict(evidence) if evidence is not None else {}
    if glm_baseline_receipt is not None:
        bundle["glm_baseline_receipt"] = glm_baseline_receipt
    if kimi_incremental_receipt is not None:
        bundle["kimi_incremental_receipt"] = kimi_incremental_receipt
    if residual_hypothesis is not None:
        bundle["residual_hypothesis"] = residual_hypothesis
    if provenance is not None:
        bundle["provenance"] = provenance
    if no_regression is not None:
        bundle["no_regression"] = no_regression
    if dsv4f_architecture_forward is not None:
        bundle["dsv4f_architecture_forward"] = dsv4f_architecture_forward

    checks: list[dict[str, Any]] = []

    if not bundle:
        _check(
            checks,
            "evidence_present",
            "DEFER",
            "no evidence supplied; residual teacher admission defaults to DEFERRED",
        )
    else:
        _check(checks, "evidence_present", "PASS", "evidence bundle supplied")

    baseline = _as_mapping(bundle.get("glm_baseline_receipt"))
    baseline_membership = _check_glm_baseline(checks, baseline)

    kimi = _as_mapping(bundle.get("kimi_incremental_receipt"))
    _check_kimi_incremental(checks, kimi, baseline_membership)

    hypothesis = _as_mapping(bundle.get("residual_hypothesis"))
    _check_residual_hypothesis(checks, hypothesis)

    prov = _as_mapping(bundle.get("provenance"))
    _check_provenance(checks, prov)

    noreg = _as_mapping(bundle.get("no_regression"))
    _check_no_regression(checks, noreg)

    forward = _as_mapping(bundle.get("dsv4f_architecture_forward"))
    _check_dsv4f_forward(checks, forward)

    statuses = {c["status"] for c in checks}
    # evidence_present DEFER alone still overall DEFERRED; FAILs win.
    substantive = [c for c in checks if c["name"] != "evidence_present"]
    sub_statuses = {c["status"] for c in substantive}

    if "FAIL" in statuses:
        verdict = VERDICT_REJECT
        reasons = [c["detail"] for c in checks if c["status"] == "FAIL"]
        reason = "hard residual-teacher admission requirement(s) failed"
    elif substantive and sub_statuses == {"PASS"}:
        verdict = VERDICT_ADMIT
        reasons = [
            "all residual-teacher admission requirements passed",
            "ADMIT is residual-lane permission only; not causal inheritance proof",
        ]
        reason = (
            "all requirements passed; admit Kimi as evidence-gated residual lane "
            "(not a claimed full duplicate transfer; not causal proof)"
        )
    else:
        verdict = VERDICT_DEFERRED
        reasons = [c["detail"] for c in checks if c["status"] == "DEFER"]
        if not reasons:
            reasons = ["evidence incomplete"]
        reason = "residual teacher admission deferred pending complete evidence"

    document = {
        "schema": ADMISSION_SCHEMA,
        "recorded_at": _utc_now(),
        "verdict": verdict,
        "reason": reason,
        "reasons": reasons,
        "checks": checks,
        "requirements": {
            "sealed_glm_only_baseline": True,
            "kimi_incremental_ab_same_membership": True,
            "named_residual_hypothesis_not_generic_distillation": True,
            "provenance_revision_identity": True,
            "protected_no_regression_math_coding_tool_agentic": True,
            "dsv4f_architecture_forward_ready": True,
            "positive_incremental_held_out_improvement": True,
        },
        "protected_axes": list(PROTECTED_AXES),
        "claim_boundary": {
            "admit_is_not_causal_proof": True,
            "admit_is_residual_lane_only": True,
            "not_duplicate_full_glm_transfer": True,
            "default_is_deferred": True,
            "reject_is_hard_fail": True,
            "fabricated_scores": False,
            "networking_or_trainer_side_effects": False,
        },
        "local_only": True,
        "evidence_keys_seen": sorted(bundle.keys()),
    }
    return seal(document)


def default_decision() -> dict[str, Any]:
    """Convenience: empty-evidence decision (always DEFERRED)."""

    return evaluate_residual_teacher_admission(None)


__all__ = [
    "ADMISSION_SCHEMA",
    "PROTECTED_AXES",
    "VERDICT_ADMIT",
    "VERDICT_DEFERRED",
    "VERDICT_REJECT",
    "default_decision",
    "evaluate_residual_teacher_admission",
]
