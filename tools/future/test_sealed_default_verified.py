"""G126 part-two tests: the verifier must reject an instrument that pins arms."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sealed_default_verified as v  # noqa: E402


def _raw_patched(monkeypatch, mutate):
    d = json.loads((v.REPO / v.RAW_REL).read_text())
    mutate(d)
    real = v._load
    monkeypatch.setattr(v, "_load", lambda r: d if r == v.RAW_REL else real(r))


def test_all_nine_checks_hold_on_the_real_run():
    cs = v.checks()
    assert len(cs) == 9
    assert [c["id"] for c in cs if not c["holds"]] == []
    assert v.verdict()["verdict"] == "PROMOTED"


def test_a_raw_that_pinned_the_state_kernel_is_refused(monkeypatch):
    """This is the whole reason a new instrument existed. An A/B that pins its
    arm cannot verify a default, and must not be accepted as if it had."""
    _raw_patched(monkeypatch,
                 lambda d: d.__setitem__("the_state_kernel_was_never_pinned", False))
    with pytest.raises(v.VerificationRefused, match="pinned the state kernel"):
        v.raw()


def test_a_raw_without_the_levers_unset_attestation_is_refused(monkeypatch):
    _raw_patched(monkeypatch, lambda d: d.__setitem__("levers_unset", False))
    with pytest.raises(v.VerificationRefused, match="levers were unset"):
        v.raw()


def test_a_baseline_state_kernel_fails_verification(monkeypatch):
    _raw_patched(monkeypatch,
                 lambda d: d.__setitem__("dn_state_kernel_at_open", "baseline"))
    out = v.verdict()
    assert out["verdict"] == "NOT_VERIFIED"
    assert "OPEN_READ_THE_PROMOTED_STATE_KERNEL_FROM_ENV" in out["failed"]
    assert "reverted" in out["consequence"]


def test_a_surviving_non_bitcast_matvec_fails_verification(monkeypatch):
    def mutate(d):
        d["last"]["kernel_histogram"].append(
            {"kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128", "count": 7})
    _raw_patched(monkeypatch, mutate)
    out = v.verdict()
    assert out["verdict"] == "NOT_VERIFIED"
    assert "NO_MATVEC_STAYED_ON_THE_NON_BITCAST_PATH" in out["failed"]


def test_the_embedding_gather_is_not_counted_as_a_missed_conversion():
    di = v.dispatch_identity()
    assert di["non_bitcast_matvec_kernels_launched"] == {}
    assert "qwen_uniform_q4_embedding_lookup" in di["excluded_as_not_a_matvec"]


def test_a_nonzero_fallback_fails_verification(monkeypatch):
    _raw_patched(monkeypatch, lambda d: d["last"].__setitem__("fallbacks", 1))
    assert v.verdict()["verdict"] == "NOT_VERIFIED"


def test_a_wrong_dispatch_count_fails_verification(monkeypatch):
    _raw_patched(monkeypatch,
                 lambda d: d["last"].__setitem__("complete_token_dispatches_last", 628))
    out = v.verdict()
    assert out["verdict"] == "NOT_VERIFIED"
    assert "DISPATCH_COUNT_MATCHES_THE_MEASURED_GRAPH" in out["failed"]


def test_a_diverging_token_stream_fails_verification(monkeypatch):
    def mutate(d):
        ids = list(d["last"]["new_token_ids"])
        ids[5] = ids[5] + 1
        d["last"]["new_token_ids"] = ids
    _raw_patched(monkeypatch, mutate)
    out = v.verdict()
    assert out["verdict"] == "NOT_VERIFIED"
    assert "TOKEN_IDENTICAL_TO_EVERY_LEASE_ARM" in out["failed"]


def test_token_identity_compares_against_both_lease_runs():
    ti = v.token_identity()
    labels = set(ti["identical_to"])
    assert any(k.startswith("lease_bitcast.") for k in labels)
    assert any(k.startswith("lease_control.") for k in labels)
    assert ti["identical_to_every_compared_arm"] is True
    assert ti["all_reps_identical_to_each_other"] is True


def test_this_runs_timings_are_explicitly_not_promoted():
    out = v.what_this_run_does_not_claim()
    assert out["evidence_class"] == "IDENTITY_ONLY_TIMINGS_ARE_CONTAMINATED"
    assert min(out["this_run_ms_per_token"]) > 22.5, (
        "this run was contaminated and must read slower than the protected "
        "lease; if it ever reads faster, the claim boundary needs rewriting")


def test_the_promoted_absolute_is_cited_not_recomputed():
    a = v.verdict()["the_protected_absolute_this_arm_carries"]
    lease = json.loads((v.REPO / "receipts/future/PROTECTED_BITCAST_ABSOLUTE.json").read_text())
    m = lease["measured"]
    assert a["wall_ms"] == float(m["bitcast_wall_ms"])
    assert a["wall_tps"] == float(m["bitcast_wall_tps"])
    assert a["gpu_ms"] == float(m["bitcast_gpu_ms"])


def test_a_missing_raw_refuses(monkeypatch):
    monkeypatch.setattr(v, "RAW_REL", "receipts/future/NO_SUCH_RAW.json")
    with pytest.raises(v.VerificationRefused, match="not on disk"):
        v.raw()
