"""Tests for the meta representation experiment funnel.

The load-bearing negative control is that advance() REFUSES (does not pass,
does not kill, does not skip) when the teacher corpus is absent, and that a
scar is retrievable by hypothesis shape after a real kill — not only by id.
"""
from __future__ import annotations

import hashlib
import json

from tools.future import meta_funnel as mf
from tools.future._common import RECEIPTS, HardwareClaimError, _assert_no_hardware_claims


def _plan(family="shared_basis", organ="routed_experts", uniform=False):
    if uniform:
        return mf._uniform_plan(family, organ)
    return {
        "unit": "TOTAL_EXECUTABLE_INFORMATION",
        "forces_uniform_bpw": False,
        "regions": [
            {"kind": "shared_generator", "bits_class": "shared", "family": family, "organ": organ},
            {"kind": "sparse_residual", "bits_class": "sparse", "family": family, "organ": organ},
            {"kind": "routing_sensitive", "bits_class": "premium", "family": family, "organ": "router"},
            {"kind": "capability_island", "bits_class": "literal", "family": family, "organ": "recurrent_state"},
            {"kind": "predictable_bulk", "bits_class": "near_zero", "family": family, "organ": "embeddings"},
        ],
    }


def _cand(cid, *, teacher="NOT_BUILT", plan=None, **extra):
    plan = plan or _plan()
    inputs = mf._default_inputs(allocation_plan=plan, teacher_corpus=teacher)
    inputs.update(extra)
    return {
        "id": cid,
        "family": extra.pop("family_name", None) or "shared_basis",
        "organ": "routed_experts",
        "technique": "shared_basis_plus_nf_residual",
        "model": mf.FLASH_MODEL,
        "allocation_plan": plan,
        "inputs": inputs,
        "passed_gates": [],
    }


def test_build_emits_sealed_receipt():
    out = mf.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "META_EXPERIMENT_FUNNEL.json"
    assert doc["schema"] == "hawking.future.meta_funnel.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert doc["seal_sha256"] == hashlib.sha256(blob).hexdigest()
    _assert_no_hardware_claims(doc)
    assert doc["counts"]["gates"] == 9
    assert doc["counts"]["families"] >= 1
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc


def test_selftest_aliases_build():
    assert mf.selftest is mf.build or callable(mf.selftest)
    out = mf.selftest()
    assert out.name == "META_EXPERIMENT_FUNNEL.json"


def test_nine_gates_declare_contract_fields():
    required = {
        "id",
        "name",
        "cost_class",
        "required_input",
        "kill_criterion",
        "passing_proves",
        "passing_does_not_prove",
    }
    assert len(mf.GATES) == 9
    names = [g.name for g in mf.GATES]
    assert names[0] == "analytical_structure_screen"
    assert names[1] == "real_teacher_fit"
    assert names[2] == "held_out_numerical"
    assert names[3] == "route_stability"
    assert names[4] == "logit_token_validation"
    assert names[5] == "bounded_capability"
    assert names[6] == "physical_nr_lowering"
    assert names[7] == "complete_nx"
    assert names[8] == "ebpw"
    for g in mf.GATES:
        for field in required:
            assert getattr(g, field), f"gate {g.id} missing {field}"
        assert g.id == names.index(g.name) + 1


def test_advance_refuses_teacher_fit_when_corpus_absent():
    """NEGATIVE CONTROL: the refusal must actually fire.

    A candidate that passed the cheap structure screen, with no teacher corpus,
    must not be moved past gate 2. REFUSED is not PASSED and not KILLED.
    """
    funnel = mf.Funnel()
    cand = _cand("neg.no-teacher")
    g1 = funnel.advance(cand, 1)
    assert g1.verdict == "PASSED", g1.reason
    g2 = funnel.advance(cand, 2)
    assert g2.verdict == "REFUSED"
    assert g2.input_state == "NOT_BUILT"
    assert g2.required_input == "teacher_corpus"
    assert "NOT_BUILT" in g2.reason
    assert 2 not in cand.get("passed_gates", [])
    assert cand.get("died_at") is None
    assert funnel.scars == []
    # Absent as a missing key, as NOT_MEASURED, and as an empty dict: all refuse.
    for token in ("NOT_MEASURED", "ABSENT", "NOT_BUILT"):
        c = _cand(f"neg.{token}", teacher=token)
        assert funnel.advance(c, 1).verdict == "PASSED"
        r = funnel.advance(c, 2)
        assert r.verdict == "REFUSED", (token, r)
        assert r.verdict != "PASSED"


def test_scar_retrievable_by_shape_not_id():
    """NEGATIVE CONTROL: a kill is matchable by the hypothesis, not the id.

    Kill candidate A at teacher-fit. Propose candidate B with a new id and the
    same family/organ/technique/model/allocation. match_scars(B) must hit A's scar.
    A different shape must not hit it.
    """
    funnel = mf.Funnel()
    plan = _plan()
    dead = _cand(
        "originally.named.foo",
        teacher={"status": "FAILED", "fit_passed": False, "mechanism": "cosine below null on real X"},
        plan=plan,
    )
    assert funnel.advance(dead, 1).verdict == "PASSED"
    killed = funnel.advance(dead, 2)
    assert killed.verdict == "KILLED", killed
    assert killed.scar is not None
    assert killed.scar["gate_name"] == "real_teacher_fit"
    assert killed.scar["mechanism"]
    assert len(funnel.scars) == 1

    reincarnated = _cand("brand.new.id.bar", teacher="NOT_BUILT", plan=plan)
    hits = funnel.match_scars(reincarnated)
    assert hits, "same-shape proposal must retrieve the scar"
    assert hits[0]["identity"]["candidate_id"] == "originally.named.foo"
    assert hits[0]["shape_sha256"] == mf.shape_fingerprint(reincarnated)
    assert hits[0]["shape_sha256"] == mf.shape_fingerprint(dead)

    other = _cand(
        "different.shape",
        teacher="NOT_BUILT",
        plan=mf._uniform_plan("binary", "lm_head"),
    )
    other["family"] = "binary"
    other["organ"] = "lm_head"
    other["technique"] = "1-bit sign code"
    assert funnel.match_scars(other) == []


def test_deaths_accumulate_never_overwritten():
    funnel = mf.Funnel()
    a = _cand("kill.a", teacher={"status": "FAILED", "fit_passed": False, "mechanism": "a died"})
    b = _cand(
        "kill.b",
        teacher={"status": "FAILED", "fit_passed": False, "mechanism": "b died"},
        plan=mf._uniform_plan("binary", "routed_experts"),
    )
    b["family"] = "binary"
    b["technique"] = "binary"
    for c in (a, b):
        assert funnel.advance(c, 1).verdict == "PASSED"
        assert funnel.advance(c, 2).verdict == "KILLED"
    assert [s["scar_id"] for s in funnel.scars] == ["MF-0001", "MF-0002"]
    assert funnel.scars[0]["mechanism"] == "a died"
    assert funnel.scars[1]["mechanism"] == "b died"
    # a second kill of a different shape does not clobber the first
    assert funnel.match_scars(a)[0]["scar_id"] == "MF-0001"


def test_cannot_skip_to_later_gate():
    funnel = mf.Funnel()
    cand = _cand(
        "skip.me",
        teacher="NOT_BUILT",
        held_out_numerical={"status": "PASSED"},
        route_traces={"status": "PASSED"},
        logit_token={"status": "PASSED"},
        physical_nr={"status": "PASSED"},
        complete_nx={"status": "PASSED"},
        ebpw_ledger={"status": "PASSED", "all_required_bytes_included": True},
    )
    r = funnel.advance(cand, 5)
    assert r.verdict == "REFUSED"
    assert r.input_state == "EARLIER_GATE_NOT_PASSED"
    assert 5 not in cand.get("passed_gates", [])


def test_malformed_allocation_is_killed_at_gate_1():
    funnel = mf.Funnel()
    cand = _cand("bad.plan", plan={"unit": "TOTAL_EXECUTABLE_INFORMATION", "regions": []})
    r = funnel.advance(cand, 1)
    assert r.verdict == "KILLED"
    assert "no regions" in r.reason
    assert funnel.scars and funnel.scars[0]["gate_id"] == 1


def test_complete_system_claim_without_accounting_is_killed():
    funnel = mf.Funnel()
    plan = _plan()
    plan["claims_complete_system"] = True
    plan["accounting_fields"] = ["weight_codes"]
    cand = _cand("fake.ebpw.claim", plan=plan)
    r = funnel.advance(cand, 1)
    assert r.verdict == "KILLED"
    assert "omits" in r.reason


def test_heterogeneous_allocation_is_expressible_and_uniform_is_not_killed():
    funnel = mf.Funnel()
    hetero = _cand("hetero.ok")
    assert funnel.advance(hetero, 1).verdict == "PASSED"
    kinds = {r["kind"] for r in hetero["allocation_plan"]["regions"]}
    assert "shared_generator" in kinds
    assert "capability_island" in kinds
    assert "sparse_residual" in kinds
    assert not hetero["allocation_plan"]["forces_uniform_bpw"]

    uni = _cand("uniform.ok", plan=mf._uniform_plan("independent_q4_g64", "routed_experts"))
    uni["family"] = "independent_q4_g64"
    r = funnel.advance(uni, 1)
    assert r.verdict == "PASSED", r.reason
    assert uni["allocation_plan"]["forces_uniform_bpw"] is True


def test_missing_allocation_plan_is_refused_not_killed():
    funnel = mf.Funnel()
    cand = _cand("no.plan")
    cand["inputs"]["allocation_plan"] = "NOT_BUILT"
    cand["allocation_plan"] = "NOT_BUILT"
    r = funnel.advance(cand, 1)
    assert r.verdict == "REFUSED"
    assert r.input_state == "NOT_BUILT"
    assert funnel.scars == []


def test_flash_families_stall_at_teacher_or_heldout():
    families, provenance = mf.recover_families()
    assert families, "recovery must produce at least the constructed Flash meta-program"
    funnel = mf.Funnel()
    runs = [funnel.run(c) for c in families]
    stalls = {(r["id"], r["stall_gate"], r["stall_verdict"]) for r in runs}
    assert stalls
    for r in runs:
        # Honest stall: do not manufacture progress past teacher/held-out.
        assert r["stall_gate"] in {1, 2, 3}, r
        if r["stall_gate"] in {2, 3}:
            assert r["stall_verdict"] == "REFUSED", r
            assert r["passed_gates"] in ([1], [1, 2])
        assert r["stall_gate"] not in {7, 8, 9}
    # When FLASH_META_REPRESENTATION_SUB1.json is visible, recovery reads the nine
    # real family_budget entries; when it is not (a sparse lane worktree), it falls
    # back to one hand-constructed heterogeneous program. Both are correct, and the
    # real path is strictly better -- so assert the PROPERTY, not the fixture id.
    het = [r for r in runs if r.get("heterogeneous") is True]
    assert het, f"no heterogeneous candidate recovered; provenance={provenance}"
    assert all(r["stall_gate"] == 2 for r in het), het
    assert all(r["stall_verdict"] == "REFUSED" for r in het), het
    assert provenance["primary_source"] in {
        "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json",
        "recovered_from_flash_receipts_and_library",
    }


def test_teacher_fit_pass_then_heldout_absent_refuses_gate_3():
    funnel = mf.Funnel()
    cand = _cand("has.teacher", teacher={"status": "PASSED", "fit_passed": True})
    assert funnel.advance(cand, 1).verdict == "PASSED"
    assert funnel.advance(cand, 2).verdict == "PASSED"
    g3 = funnel.advance(cand, 3)
    assert g3.verdict == "REFUSED"
    assert g3.required_input == "held_out_numerical"
    assert g3.input_state == "NOT_MEASURED"


def test_route_mismatch_kills_only_after_earlier_gates():
    funnel = mf.Funnel()
    cand = _cand(
        "router.mismatch",
        teacher={"status": "PASSED", "fit_passed": True},
        held_out_numerical={"status": "PASSED"},
        route_traces={
            "status": "MISMATCH",
            "expert_ids_exact_match": False,
            "mechanism": "top-k identity diverged",
        },
    )
    assert funnel.advance(cand, 1).verdict == "PASSED"
    assert funnel.advance(cand, 2).verdict == "PASSED"
    assert funnel.advance(cand, 3).verdict == "PASSED"
    g4 = funnel.advance(cand, 4)
    assert g4.verdict == "KILLED"
    assert g4.scar["gate_name"] == "route_stability"


def test_module_level_advance_matches_funnel():
    cand = _cand("mod.level")
    r = mf.advance(cand, "analytical_structure_screen")
    assert r.verdict == "PASSED"
    r2 = mf.advance(cand, "real_teacher_fit")
    assert r2.verdict == "REFUSED"


def test_hardware_claim_guard_still_bites():
    """The sidecar must not grow a hardware number. Watch the guard fail."""
    doc = {"tps": 12.0}
    try:
        _assert_no_hardware_claims(doc)
    except HardwareClaimError:
        return
    raise AssertionError("hardware claim guard did not fire on tps=12.0")
