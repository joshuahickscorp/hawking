"""doctor6 1.5-BPW water-fill: allocator, compose floor, sealed escape, selftest."""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from lab.operators.doctor6.billing import project_complete_bpw, seal_with_ceiling
from lab.operators.doctor6.ceiling import (
    enforce_ceiling,
    escape_is_sealed,
    issue_specialization_escape,
)
from lab.operators.doctor6.compose import compose_organ_chain
from lab.operators.doctor6.selftest import (
    check_allocator,
    check_ceiling,
    check_compose_incumbent_floor,
    check_qat,
    run_selftest,
)
from lab.operators.eco_common import sealed
from lab.operators.mixed_precision_alloc import allocate_from_holdout
from lab.operators.one_bit_ceiling import CeilingViolation


@contextmanager
def _raises(exc_type: type[BaseException], match: str | None = None):
    try:
        yield
    except exc_type as exc:
        if match is not None and match not in str(exc):
            raise AssertionError(f"{exc!r} does not contain {match!r}") from exc
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def test_allocator_respects_avg_le_target() -> None:
    rec = check_allocator()
    assert rec["within_budget"], rec
    assert rec["greedy_within_budget"], rec
    assert rec["achieved_avg_eff_bpw"] <= 1.5 + 1e-9


def test_hard_organ_receives_more_bits_than_easy() -> None:
    rec = check_allocator()
    assert rec["hard_gets_more_bits"], rec
    assert rec["hard_bits"] > rec["easy_bits"]


def test_allocate_from_holdout_does_not_use_weight_space_proxy() -> None:
    organs = [
        {
            "name": "a",
            "elems": 100,
            "holdout_cosine": 0.5,
            "component": "down_proj",
        },
        {
            "name": "b",
            "elems": 100,
            "holdout_cosine": 0.99,
            "component": "gate_proj",
        },
    ]
    organs.extend(
        {
            "name": f"pad{i}",
            "elems": 100,
            "holdout_cosine": 0.99,
            "component": "gate_proj",
        }
        for i in range(14)
    )
    out = allocate_from_holdout(
        organs,
        bits_set=(1, 2, 3, 4),
        target_bpw=1.5,
        layer_target=0.9857,
    )
    assert out["not_weight_space_relrms"] is True
    assert out["sensitivity"] == "activation_holdout_output_cosine"
    assert out["allocator_invoked"] is True
    assert out["allocation"]["a"] > out["allocation"]["b"]


def test_compose_never_emits_below_incumbent() -> None:
    rec = check_compose_incumbent_floor()
    assert rec["n_prescribed_below_incumbent"] == 0, rec
    assert rec["prescribed_cosine"] + 1e-12 >= rec["incumbent_cosine"]


def test_compose_fallback_on_overfit_actsvd() -> None:
    rng = np.random.default_rng(3)
    W = rng.standard_normal((12, 20), dtype=np.float32) * 0.08
    X_fit = rng.standard_normal((5, 20), dtype=np.float32)
    X_hold = rng.standard_normal((30, 20), dtype=np.float32)
    result = compose_organ_chain(
        W=W,
        X_fit=X_fit,
        X_hold=X_hold,
        organ_key="L18.gate_proj.weight",
        component="gate_proj",
        sensitivity=0.1,
        seed=11,
        device="cpu",
        qat_steps=2,
        target_cos=0.9857,
    )
    assert result["prescribed_cosine"] + 1e-12 >= result["incumbent_cosine"]


def test_ceiling_1_0_default_still_fails_closed() -> None:
    bill = project_complete_bpw(mean_expert_payload_bytes=1e9)
    with _raises(CeilingViolation):
        seal_with_ceiling(bill, target_bpw=1.0, note="test")
    legal = project_complete_bpw(mean_expert_payload_bytes=120_000.0)
    out = seal_with_ceiling(legal, target_bpw=1.0, note="test_legal")
    assert out["receipt"]["legal"] is True
    assert not out.get("escape_applied")


def test_ceiling_escape_is_sealed_not_bypassed() -> None:
    rec = check_ceiling()
    assert rec["ok"], rec
    legal = project_complete_bpw(mean_expert_payload_bytes=120_000.0)
    out = seal_with_ceiling(legal, target_bpw=1.5, note="test_escape")
    esc = out["escape_receipt"]
    assert esc is not None
    assert sealed(esc, "sha256")
    assert escape_is_sealed(esc, target_bpw=1.5)
    assert out["escape_applied"] is True
    # Direct enforce without escape still refuses — not a silent raise of 1.0.
    from lab.operators.doctor6.selftest import _legal_slots

    slots, n = _legal_slots()
    with _raises(CeilingViolation, match="upward bracketing is REJECTED"):
        enforce_ceiling(slots, n, target_bpw=1.5)
    # Tamper is not a bypass.
    tampered = dict(esc)
    tampered["justification"] = "nope"
    with _raises(CeilingViolation, match="upward bracketing is REJECTED"):
        enforce_ceiling(slots, n, target_bpw=1.5, escape_receipt=tampered)
    # Huge bill still fails at 1.5 — escape does not waive the abs ceiling.
    huge = project_complete_bpw(mean_expert_payload_bytes=1e9)
    with _raises(CeilingViolation):
        seal_with_ceiling(huge, target_bpw=1.5, note="test_over")


def test_issue_escape_refuses_outside_band() -> None:
    with _raises(CeilingViolation):
        issue_specialization_escape(target_bpw=1.0, justification="no")
    with _raises(CeilingViolation, match="abs hard ceiling"):
        issue_specialization_escape(target_bpw=1.6, justification="no")


def test_qat_fit_ge_calib_on_synthetic_organ() -> None:
    rec = check_qat()
    assert rec.get("ok"), rec
    assert rec["qat_fit"] + 1e-4 >= rec["calib_fit"]


def test_selftest_all_pass() -> None:
    out = run_selftest()
    assert out["all_pass"], out
    assert out["ceiling"]["over_ceiling_fail_closed"]
    assert out["compose"]["n_prescribed_below_incumbent"] == 0
    assert out["allocator"]["allocator_invoked"]


if __name__ == "__main__":
    tests = [
        test_allocator_respects_avg_le_target,
        test_hard_organ_receives_more_bits_than_easy,
        test_allocate_from_holdout_does_not_use_weight_space_proxy,
        test_compose_never_emits_below_incumbent,
        test_compose_fallback_on_overfit_actsvd,
        test_ceiling_1_0_default_still_fails_closed,
        test_ceiling_escape_is_sealed_not_bypassed,
        test_issue_escape_refuses_outside_band,
        test_qat_fit_ge_calib_on_synthetic_organ,
        test_selftest_all_pass,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    raise SystemExit(1 if failed else 0)
