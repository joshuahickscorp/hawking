"""Tests for the ABI verdict harness.

A guard nobody has watched fail is not a guard: the negative control registers
a FALSE_POSITIVE exemption for one specific finding, then proves a different
instance of the same check class still fires. A verdict with no evidence
reference is REFUSED.
"""
from __future__ import annotations

import json

import pytest

from tools.future import abi_verdicts as abi
from tools.future._common import RECEIPTS, HardwareClaimError


EVIDENCE = {"ref": "receipts/future/STATIC_KERNEL_PREFLIGHT.json"}


@pytest.fixture(autouse=True)
def _clean_store():
    abi.reset_store()
    yield
    abi.reset_store()
    # Leave a canonical pending receipt on disk after every test so a later
    # `--pending` run is not contaminated by an in-test ruling.
    abi.build()


def _errors():
    rows = abi.error_findings()
    assert rows, "preflight ERROR findings must be loadable from STATIC_KERNEL_PREFLIGHT.json"
    return rows


def _by_check(check: str) -> list[dict]:
    return [r for r in _errors() if r.get("check") == check]


def test_entry_point_emits_sealed_receipt():
    out = abi.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "ABI_VERDICT_HARNESS.json"
    assert doc["schema"] == "hawking.future.abi_verdicts.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["evidence_source"] in {"pinned_snapshot", "live_headless"}
    assert set(doc["verdict_class_names"]) == set(abi.VERDICT_CLASS_NAMES)
    # Not a fixed count and not necessarily pending: Codex has since adjudicated
    # every reported finding, so assert the shape, not a frozen world state.
    assert isinstance(doc["all_reported_pending"], bool)
    assert doc["reported_count"] >= 1
    assert doc["pending_count"] <= doc["reported_count"]
    # Not six: Codex adjudicated every reported finding, so pending is now 0.
    # Assert the invariant that survives adjudication, not the pre-verdict count.
    assert doc["pending_count"] == len([
        r for r in doc.get("dossiers", []) if r.get("verdict_status") == "PENDING"
    ])
    assert doc["rulings_issued_by_this_sidecar"] == 0
    assert doc["codex_files_untouched"] is True
    # Counts are derived, not a hard-coded six that would rot if Codex added one.
    # Once Codex rules, reported and pending legitimately diverge: 6 reported,
    # 0 pending. The invariant that survives adjudication is against the
    # REPORTED count, not the pending one.
    assert doc["preflight_error_count_derived"] == doc["reported_count"]


def test_six_verdict_classes_and_checker_implications():
    assert abi.VERDICT_CLASS_NAMES == (
        "CONFIRMED_BUG",
        "FALSE_POSITIVE",
        "DEAD_CODE",
        "INTENTIONAL_ALIAS",
        "GENERATED_KERNEL",
        "ABI_MISMATCH",
    )
    for name, body in abi.VERDICT_CLASSES.items():
        assert body["meaning"]
        assert body["checker_implication"]
        assert "check class" in body["checker_implication"].lower() or name in {
            "CONFIRMED_BUG",
            "DEAD_CODE",
            "INTENTIONAL_ALIAS",
            "GENERATED_KERNEL",
            "ABI_MISMATCH",
        }


def test_dossiers_cover_every_error_and_every_class_before_any_verdict():
    doss = abi.dossiers()
    errors = _errors()
    assert len(doss) == len(errors) == 6
    ids = [d["finding_id"] for d in doss]
    assert ids == sorted(ids)
    assert ids == [e["finding_id"] for e in errors]
    for d in doss:
        assert d["verdict"] is None
        assert d["verdict_status"] == "PENDING"
        assert d["host_site"]["preflight"]
        assert d["host_site"]["snippet"] is not None
        assert set(d["class_responses"]) == set(abi.VERDICT_CLASS_NAMES)
        for cls, resp in d["class_responses"].items():
            assert resp["preflight_change"], cls
            assert resp["harness_action"], cls
            assert resp["would_mean"], cls
    # Two kernel_existence, four type_width — derived from the receipt.
    assert sum(1 for d in doss if d["check"] == "kernel_existence") == 2
    assert sum(1 for d in doss if d["check"] == "type_width") == 4


def test_pending_count_is_derived_not_fixed():
    """Codex has since adjudicated every reported finding, so pending is 0 of 6.

    The harness must never assert a fixed count: a hard-coded six would have been
    wrong the moment the first verdict landed.
    """
    reported = abi.error_findings()
    assert reported, "no findings recoverable from live preflight or adjudication"
    out = abi.format_pending()
    assert f"of {len(reported)} reported" in out
    assert "records rulings, it does not make them" in out

def test_verdict_without_evidence_reference_is_refused():
    fid = _errors()[0]["finding_id"]
    with pytest.raises(abi.VerdictRefused, match="no evidence reference"):
        abi.record_verdict(fid, "FALSE_POSITIVE", None)
    with pytest.raises(abi.VerdictRefused, match="no evidence reference"):
        abi.record_verdict(fid, "FALSE_POSITIVE", {})
    with pytest.raises(abi.VerdictRefused, match="no evidence reference"):
        abi.record_verdict(fid, "FALSE_POSITIVE", {"note": "a claim without a ref"})
    with pytest.raises(abi.VerdictRefused, match="no evidence reference"):
        abi.record_verdict(fid, "FALSE_POSITIVE", "   ")
    assert abi.pending()  # still six; nothing recorded
    assert len(abi.pending()) == 6


def test_unknown_verdict_and_unknown_id_are_refused():
    fid = _errors()[0]["finding_id"]
    with pytest.raises(abi.VerdictRefused, match="unknown verdict class"):
        abi.record_verdict(fid, "LOOKS_FINE_TO_ME", EVIDENCE)
    with pytest.raises(abi.VerdictRefused, match="unknown finding_id"):
        abi.record_verdict("not-a-real-finding", "FALSE_POSITIVE", EVIDENCE)


def test_negative_control_false_positive_exemption_is_not_a_blanket_disable():
    """Register a FALSE_POSITIVE for one type_width finding. A different
    instance of type_width must still fire. A guard nobody has watched fail
    is not a guard.
    """
    type_width = _by_check("type_width")
    assert len(type_width) >= 2
    target = type_width[0]
    other = type_width[1]
    assert target["kernel"] != other["kernel"] or target["host"] != other["host"]

    rec = abi.record_verdict(target["finding_id"], "FALSE_POSITIVE", EVIDENCE)
    assert rec.verdict == "FALSE_POSITIVE"
    assert rec.evidence_refs == [EVIDENCE["ref"]]
    assert rec.action["kind"] == "exemption"
    rule = abi.ExemptionRule(**rec.action["rule"])
    abi._assert_narrow(rule)

    remaining_ids = {r["finding_id"] for r in abi.pending()}
    assert target["finding_id"] not in remaining_ids
    assert other["finding_id"] in remaining_ids, (
        "exemption for one host site must not swallow a second type_width finding"
    )

    # Genuinely different instance of the same check class: set_u32 onto a
    # device pointer, different kernel, different buffer index.
    probes = abi.type_width_probe(kernel="unrelated_probe_k", index=0)
    assert probes, (
        "NEGATIVE CONTROL FAILED: type_width did not fire on a different kernel "
        f"after exempting {target['finding_id']}. findings were empty."
    )
    assert all(p["kernel"] == "unrelated_probe_k" for p in probes)
    still = abi.apply_store_to_findings(probes)
    assert still, (
        "NEGATIVE CONTROL FAILED: the exemption blanketed type_width and swallowed "
        "an unrelated set_u32-onto-device finding."
    )

    # Same check, same kernel as the exempted finding, different buffer index.
    same_kernel_other_index = abi.type_width_probe(kernel=str(target["kernel"]), index=0)
    assert same_kernel_other_index, "type_width must still fire at a different buffer index"
    still_idx = abi.apply_store_to_findings(same_kernel_other_index)
    assert still_idx, (
        "NEGATIVE CONTROL FAILED: exemption on buffer_index of the finding disabled "
        "type_width for the same kernel at a different index."
    )

    # kernel_existence is a different check class and must be untouched.
    exist = abi.kernel_existence_probe("definitely_missing_probe_kernel")
    assert exist
    assert abi.apply_store_to_findings(exist)


def test_confirmed_bug_installs_a_regression_fixture_that_still_fires():
    exist = _by_check("kernel_existence")[0]
    abi.record_verdict(exist["finding_id"], "CONFIRMED_BUG", EVIDENCE)
    assert abi.regression_still_detectable(exist["finding_id"]) is True
    # The original finding stays visible as CONFIRMED_BUG, not dropped.
    view = abi.apply_store_to_findings(abi.error_findings())
    hit = next(v for v in view if v["finding_id"] == exist["finding_id"])
    assert hit["verdict_status"] == "CONFIRMED_BUG"
    assert hit["check"] == "kernel_existence"


def test_dead_code_downgrades_only_that_finding():
    row = _by_check("type_width")[-1]
    abi.record_verdict(row["finding_id"], "DEAD_CODE", EVIDENCE)
    view = abi.apply_store_to_findings(abi.error_findings())
    hit = next(v for v in view if v["finding_id"] == row["finding_id"])
    assert hit["severity"] == "WARNING"
    assert hit["check"] == "reachability"
    others = [v for v in view if v["finding_id"] != row["finding_id"] and v.get("check") == "type_width"]
    assert others, "DEAD_CODE must not downgrade the rest of type_width"


def test_intentional_alias_teaches_only_that_name():
    exist = _by_check("kernel_existence")[0]
    with pytest.raises(abi.VerdictRefused, match="alias_of"):
        abi.record_verdict(exist["finding_id"], "INTENTIONAL_ALIAS", EVIDENCE)
    evidence = dict(EVIDENCE)
    evidence["alias_of"] = "kv_append_f32"
    abi.record_verdict(exist["finding_id"], "INTENTIONAL_ALIAS", evidence)
    assert abi.resolve_kernel_name(exist["kernel"]) == "kv_append_f32"
    assert abi.resolve_kernel_name("some_other_missing_kernel") == "some_other_missing_kernel"
    leftover = abi.kernel_existence_probe("some_other_missing_kernel")
    assert leftover
    assert abi.apply_store_to_findings(leftover)


def test_generated_kernel_seam_does_not_define_unrelated_names():
    qwen = [r for r in _by_check("kernel_existence") if "matmul_k1" in (r.get("kernel") or "")]
    assert qwen, "qwen matmul_k1 existence finding must be in the preflight"
    abi.record_verdict(qwen[0]["finding_id"], "GENERATED_KERNEL", EVIDENCE)
    names = abi.generated_names()
    assert qwen[0]["kernel"] in names
    assert "totally_unrelated_generated_looking_name" not in names
    leftover = abi.kernel_existence_probe("totally_unrelated_generated_looking_name")
    assert leftover
    assert abi.apply_store_to_findings(leftover)


def test_abi_mismatch_reclassifies_without_disabling_type_width():
    row = _by_check("type_width")[0]
    abi.record_verdict(row["finding_id"], "ABI_MISMATCH", EVIDENCE)
    view = abi.apply_store_to_findings(abi.error_findings())
    hit = next(v for v in view if v["finding_id"] == row["finding_id"])
    assert hit["check"] == "abi_mismatch"
    probes = abi.type_width_probe(kernel="abi_probe_k", index=0)
    assert probes
    still = abi.apply_store_to_findings(probes)
    assert still, "ABI_MISMATCH reclassification must not blanket-disable type_width"


def test_dossier_prepared_response_is_immutable_but_status_is_truthful():
    """A verdict must not rewrite the PREPARED response, but must not be hidden either.

    The dossier's value is that every class's implication was written down BEFORE
    the ruling, so the response is not reverse-engineered from the answer. That
    content is immutable. The status is a statement about the world, and after
    Codex rules, reporting PENDING would simply be false.
    """
    fid = _errors()[0]["finding_id"]
    before = {d["finding_id"]: d for d in abi.dossiers()}
    prepared_before = before[fid]["per_verdict"] if "per_verdict" in before[fid] else None

    st = abi.store()
    abi.record_verdict(fid, "CONFIRMED_BUG", EVIDENCE, st=st)
    after = {d["finding_id"]: d for d in abi.dossiers(st=st)}

    assert after[fid]["verdict_status"] == "RULED"
    assert after[fid]["verdict"] == "CONFIRMED_BUG"
    # Every OTHER finding in this fresh store is untouched.
    others = [d for fid2, d in after.items() if fid2 != fid]
    assert others, "need a second finding to prove the ruling did not spread"
    assert all(d["verdict_status"] == "PENDING" for d in others)
    if prepared_before is not None:
        assert after[fid]["per_verdict"] == prepared_before, (
            "recording a verdict must not rewrite the prepared per-class response"
        )


def test_duplicate_verdict_is_refused():
    fid = _errors()[0]["finding_id"]
    abi.record_verdict(fid, "CONFIRMED_BUG", EVIDENCE)
    with pytest.raises(abi.VerdictRefused, match="already has a verdict"):
        abi.record_verdict(fid, "FALSE_POSITIVE", EVIDENCE)


def test_receipt_has_no_hardware_claims():
    out = abi.build()
    doc = json.loads(out.read_text())
    # Re-seal through write_receipt: HardwareClaimError is the tripwire.
    try:
        from tools.future._common import write_receipt

        write_receipt("_ABI_VERDICT_THROWNAWAY.json", dict(doc), "test_abi_verdicts.py")
    except HardwareClaimError as exc:
        raise AssertionError(f"harness document tripped HardwareClaimError: {exc}") from exc
    p = RECEIPTS / "_ABI_VERDICT_THROWNAWAY.json"
    if p.exists():
        p.unlink()


def test_format_pending_lists_every_error_id():
    text = abi.format_pending()
    for row in abi.pending():
        assert row["finding_id"] in text


def test_exemption_rule_without_kernel_is_refused():
    with pytest.raises(abi.VerdictRefused, match="blanket"):
        abi._assert_narrow(
            abi.ExemptionRule(rule_id="bad", check="type_width", buffer_index=3)
        )
    with pytest.raises(abi.VerdictRefused, match="buffer index"):
        abi._assert_narrow(
            abi.ExemptionRule(rule_id="bad", check="type_width", kernel="gemv_f32_attn")
        )
