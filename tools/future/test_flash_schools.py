"""Tests for Flash Gravity organ schools.

Negative controls (must actually fail if the guard is removed):
  * a bytes-halving FLOP-tripling candidate is dominated / loses on joint cost
  * a candidate matching recorded-dead expert sharing is REFUSED while a
    structurally distinct cousin still emits
  * each of the fourteen schools is schedulable ALONE
A guard nobody has watched fail is not a guard.
"""
from __future__ import annotations

import json

import pytest

from tools.future import flash_schools as fs
from tools.future import expert_bank_school as ebs
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, write_receipt


def _receipt_doc():
    out = fs.build()
    return out, json.loads(out.read_text())


def test_build_emits_sealed_receipt():
    out, doc = _receipt_doc()
    assert out.parent == RECEIPTS
    assert out.name == "FLASH_ORGAN_SCHOOLS.json"
    assert doc["schema"] == "hawking.future.flash_schools.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert len(doc["seal_sha256"]) == 64
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert "Static sidecar" in doc["claim_boundary"]
    assert doc["fit_policy"] == "NOT_FIT"
    assert doc["measurement_classes"]["this_module"] == "STATIC_ONLY"
    assert doc["measurement_classes"]["DIAGNOSTIC_RELATIVE"] == "not produced"
    assert doc["measurement_classes"]["PROTECTED_ABSOLUTE"] == "not produced"
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]["discoverable"] is True
    assert doc["era_vocabulary"]["no_era_vi"] is True
    assert doc["era_vocabulary"]["no_odyssey_iv"] is True
    assert doc["era_vocabulary"]["fpga_is_not_its_own_civilization"] is True


def test_selftest_aliases_build():
    assert fs.selftest().name == "FLASH_ORGAN_SCHOOLS.json"


def test_fourteen_named_schools_and_program_families():
    assert list(fs.SCHOOL_CATALOG) == [
        "ROUTED_EXPERTS",
        "SHARED_EXPERTS",
        "ROUTER",
        "HC_HYPERCONNECTION",
        "DELTANET_RECURRENT_STATE",
        "FULL_ATTENTION",
        "KV_STATE",
        "NGRAM",
        "EMBEDDING",
        "LM_HEAD",
        "NORMALIZATION",
        "POSITIONAL_STRUCTURE",
        "DECODING",
        "MTP_SPECULATION",
    ]
    assert list(fs.PROGRAM_FAMILIES) == [
        "LiteralTensor",
        "QuantTensor",
        "SharedBasisProgram",
        "FactorizedProgram",
        "DictionaryProgram",
        "GeneratedProgram",
        "SparseResidualProgram",
        "RecurrentStateProgram",
        "LookupProgram",
        "ConditionalProgram",
        "RoutedSubprogram",
        "FusedPhysicalProgram",
        "ZeroProgram",
        "CompositeProgram",
    ]
    # Counts are derived from the catalogs, not a rotting integer.
    assert fs.build()
    doc = json.loads((RECEIPTS / "FLASH_ORGAN_SCHOOLS.json").read_text())
    assert doc["counts"]["schools_in_catalog"] == len(fs.SCHOOL_CATALOG)
    assert doc["counts"]["program_families_in_catalog"] == len(fs.PROGRAM_FAMILIES)
    assert doc["counts"]["schools_scheduled"] == len(doc["schools"])
    assert doc["counts"]["candidates"] == sum(s["n_candidates"] for s in doc["schools"])


def test_each_school_schedulable_alone():
    """Each school emits candidates and a WorkUnit without running the others."""
    ids = []
    for school in fs.SCHOOL_CATALOG:
        result = fs.schedule_school(school)
        assert result["school"] == school
        assert result["independent"] is True
        assert result["schedulable_alone"] is True
        assert result["candidates"], f"{school} emitted no candidates"
        assert result["n_candidates"] == len(result["candidates"])
        assert result["workunit"]["id"] == f"future.flash_schools.{school}"
        assert result["workunit"]["resource_class"] == "STATIC_ANALYSIS"
        assert result["workunit"]["status"] == "pending"
        assert result["workunit"]["classification"] == "STATIC_ONLY"
        assert "--school" in result["workunit"]["command"]
        assert school in result["workunit"]["command"]
        assert result["evidence_class"] == "STATIC_ONLY"
        assert result["gpu_authority"] is False
        for c in result["candidates"]:
            assert c["school"] == school
            assert c["independently_schedulable"] is True
            assert c["cheapest_falsifier"]
            assert c["evidence_class"] == "STATIC_ONLY"
            for field in fs.CANDIDATE_FIELDS:
                assert field in c, f"{c['id']} missing {field}"
        ids.append(result["workunit"]["id"])
    assert len(ids) == len(set(ids))
    with pytest.raises(fs.UnknownSchoolError):
        fs.schedule_school("ERA_VI_DOES_NOT_EXIST")


def test_elimination_questions_and_three_zeros():
    answers = fs.elimination_answers()
    assert set(answers) == set(fs.SCHOOL_CATALOG)
    for school, row in answers.items():
        for q in fs.ELIMINATION_QUESTIONS:
            assert row[q], f"{school} missing {q}"
        zeros = row["three_zeros"]
        assert set(zeros) == set(fs.THREE_ZEROS)
        for z, text in zeros.items():
            assert text, f"{school} empty {z}"
    # Router control plane cannot be zeroed.
    r = answers["ROUTER"]["three_zeros"]
    assert "REFUSED" in r["ZERO_STORAGE"]
    assert "REFUSED" in r["ZERO_INDEPENDENT_INFORMATION"]
    assert "REFUSED" in r["ZERO_EXECUTION"]
    # Positional structure is the textbook ZERO_STORAGE organ.
    assert "OPEN" in answers["POSITIONAL_STRUCTURE"]["three_zeros"]["ZERO_STORAGE"]
    # MTP ZeroProgram (no speculation) is legal.
    assert "OPEN" in answers["MTP_SPECULATION"]["three_zeros"]["ZERO_EXECUTION"]


def test_negative_control_bytes_halving_flop_tripling_is_dominated():
    """Guard nobody has watched fail is not a guard.

    A candidate that halves bytes and triples FLOPs must be able to LOSE.
    """
    routed = fs.schedule_school("ROUTED_EXPERTS")
    control = next(c for c in routed["candidates"] if c["id"] == "ROUTED_EXPERTS:LITERAL")
    proof = fs.prove_flop_trap(control)
    assert proof["trap_storage_bytes"] == control["storage_bytes"] // 2
    assert proof["trap_flop_milli"] == control["flop_milli"] * 3
    assert proof["trap_dominates_control"] is False
    assert proof["trap_loses_on_joint_cost"] is True
    assert proof["dominator_dominates_trap"] is True
    assert proof["storage_only_by_keyword_raises"] is True
    assert proof["storage_only_axes_raises"] is True
    assert proof["naive_storage_winner"] == "TRAP-HALF-BYTES-TRIPLE-FLOPS"
    assert proof["scalar_winner"] is None
    assert proof["trap_joint_cost"] > proof["control_joint_cost"]

    trap = fs.make_trap(control)
    with pytest.raises(fs.StorageOnlyRankingError):
        fs.rank([control, trap], by="storage_bytes")
    with pytest.raises(fs.StorageOnlyRankingError):
        fs.rank([control, trap], axes=("storage_bytes",))
    result = fs.rank([control, trap], control_storage=control["storage_bytes"])
    assert result["scalar_winner"] is None
    assert result["storage_only"] == "REFUSED"
    joint_ids = [row["id"] for row in result["joint_cost_order"]]
    assert joint_ids[0] == control["id"]
    assert trap["id"] in joint_ids
    assert not fs.dominates(trap, control)
    assert fs.dominates(fs.make_dominator(trap), trap)

    # Receipt carries the watched-fail proof.
    doc = json.loads(fs.build().read_text())
    nc = doc["negative_control"]["flop_trap"]
    assert nc["dominator_dominates_trap"] is True
    assert nc["trap_loses_on_joint_cost"] is True
    assert nc["trap_dominates_control"] is False


def test_negative_control_dead_sharing_refused_distinct_emits():
    """Recorded-dead trivial expert sharing is REFUSED; a cousin still emits."""
    with pytest.raises(fs.DeadHypothesisError) as ei:
        fs.admit_candidate(fs.DEAD_PROBE, require_schema=False)
    err = ei.value
    assert "REFUSED" in str(err)
    assert err.scar.get("id") or err.scar.get("scar_id")
    assert err.scar.get("family") in {
        "cross_expert_structure",
        ebs.DEAD_FAMILY_BASIS,
        ebs.DEAD_FAMILY_RAW,
        ebs.DEAD_FAMILY_ARCHETYPE,
    }

    routed = fs.schedule_school("ROUTED_EXPERTS")
    proof = fs.prove_scar_refusal(routed["candidates"])
    assert proof["dead_probe_refused"] is True
    assert proof["structurally_distinct_emitted"]
    assert proof["structurally_distinct_family"] in {
        "FactorizedProgram",
        "GeneratedProgram",
        "ConditionalProgram",
        "SparseResidualProgram",
    }
    live = next(c for c in routed["candidates"] if c["id"] == proof["structurally_distinct_emitted"])
    admitted = fs.admit_candidate(live)
    assert admitted["id"] == live["id"]
    assert fs.match_scar(live) is None

    # Phrase-level probes for the other two dead families.
    with pytest.raises(fs.DeadHypothesisError):
        fs.admit_candidate(
            {"id": "P-RAW", "mechanism": "raw global expert similarity"},
            require_schema=False,
        )
    with pytest.raises(fs.DeadHypothesisError):
        fs.admit_candidate(
            {"id": "P-ARCH", "mechanism": "unchanged archetype"},
            require_schema=False,
        )
    with pytest.raises(fs.DeadHypothesisError):
        fs.admit_candidate(
            {
                "id": "P-UNI",
                "mechanism": "uniform bpw across control and bulk",
                "hypothesis_family": "uniform_subbit_allocation",
            },
            require_schema=False,
        )

    doc = json.loads(fs.build().read_text())
    scar = doc["negative_control"]["scar_refusal"]
    assert scar["dead_probe_refused"] is True
    assert scar["structurally_distinct_emitted"]


def test_router_control_plane_not_uniform_bpw():
    inv = fs.organ_inventory()
    proof = fs.prove_router_control_plane(inv)
    if not proof.get("reachable"):
        pytest.skip("router overlay / routed_experts bytes unreachable in this checkout")
    assert proof["heterogeneous_dominates_uniform"] is True
    assert proof["uniform_dominates_heterogeneous"] is False
    assert proof["router_bits_disproportionate"] is True
    assert proof["same_total_bits_class"] is True
    hetero = proof["heterogeneous"]
    assert hetero["router_bpw_milli"] > hetero["bulk_bpw_milli"]

    router = fs.schedule_school("ROUTER")
    ids = {c["id"] for c in router["candidates"]}
    assert "ROUTER:CONTROL-PLANE-PREMIUM" in ids
    assert "ROUTER:UNIFORM-BPW-BASELINE" in ids
    crushed = next(c for c in router["candidates"] if c["id"] == "ROUTER:UNIFORM-BPW-BASELINE")
    premium = next(c for c in router["candidates"] if c["id"] == "ROUTER:CONTROL-PLANE-PREMIUM")
    assert crushed["capability_risk_class"] == "CONTROL_CRUSHED"
    assert premium["bit_class"] == "CONTROL_FLOW_PREMIUM"
    assert crushed["capability_risk_rank"] > premium["capability_risk_rank"]
    zeros = router["elimination"]["three_zeros"]
    assert "REFUSED" in zeros["ZERO_STORAGE"]


def test_extends_landed_schools_not_duplicate():
    routed = fs.schedule_school("ROUTED_EXPERTS")
    ngram = fs.schedule_school("NGRAM")
    assert any(
        str(c.get("extends") or "").startswith("tools/future/expert_bank_school.py")
        for c in routed["candidates"]
    )
    assert any(
        str(c.get("extends") or "").startswith("tools/future/ngram_school.py")
        for c in ngram["candidates"]
    )
    assert any(c["program_family"] == "SharedBasisProgram" for c in routed["candidates"])
    shared = fs.schedule_school("SHARED_EXPERTS")
    assert any(c["program_family"] == "FactorizedProgram" for c in shared["candidates"])
    lm = fs.schedule_school("LM_HEAD")
    assert any(c["id"] == "LM_HEAD:TIE-EMBED" for c in lm["candidates"])
    # Wrapping does not re-emit the dead families.
    blobs = " ".join(c["mechanism"].lower() for c in routed["candidates"])
    assert "trivial global expert sharing" not in blobs
    assert "raw global expert similarity" not in blobs
    assert "unchanged archetype" not in blobs


def test_census_bytes_derived_when_reachable():
    inv = fs.organ_inventory()
    if inv.get("census_source") == "unavailable":
        routed = fs.schedule_school("ROUTED_EXPERTS", inventory=inv)
        assert routed["independent"] is True
        return
    families = {r["family"] for r in inv["families"]}
    assert "routed_experts" in families
    routed_bytes = inv["by_family"]["routed_experts"]["bytes"]
    specimen = inv["specimen_bytes"]
    assert routed_bytes * 2 > specimen  # ~68.5% — derived, not a baked fraction assert
    assert fs.school_source_bytes("ROUTED_EXPERTS", inv) == routed_bytes
    if inv.get("router_tensor_bytes"):
        assert fs.school_source_bytes("ROUTER", inv) == inv["router_tensor_bytes"]
    # Family budget count is derived from the receipt, not a hardcoded 9.
    if inv.get("n_budget_families"):
        assert inv["n_budget_families"] == len(inv["family_budget"])


def test_no_hardware_numeric_claims_in_receipt():
    doc = json.loads(fs.build().read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                    raise AssertionError(f"hardware field {here}={v!r}")
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)
    with pytest.raises(HardwareClaimError):
        write_receipt("_flash_schools_hw_probe.json", {"tps": 1.0}, "test")


def test_resident_callable_and_workunit_shape():
    doc = json.loads(fs.build().read_text())
    rc = doc["resident_callable"]
    assert "flash_schools.py --build" in rc["entry_point"]
    assert "--school" in rc["per_school_entry"]
    assert rc["receipt"] == "receipts/future/FLASH_ORGAN_SCHOOLS.json"
    assert rc["frontier_fed"]["name"] == "flash_gravity_organ_schools"
    assert rc["fail_closed"]["dead_hypothesis"]
    assert rc["fail_closed"]["hardware_claim"]
    assert "acquire a GPU lease" in rc["cannot"]
    units = doc["workunits"]
    assert len(units) == len(doc["schools"])
    for u in units:
        for field in (
            "id",
            "role",
            "description",
            "dependencies",
            "status",
            "resource_class",
            "verifier",
            "effect_class",
            "claim_boundary",
        ):
            assert field in u, f"{u.get('id')} missing {field}"
        assert u["resource_class"] == "STATIC_ANALYSIS"
        assert u["status"] == "pending"
        assert u["id"].startswith("future.flash_schools.")


def test_every_emitted_candidate_has_falsifier_and_static_only():
    for school in fs.SCHOOL_CATALOG:
        for c in fs.schedule_school(school)["candidates"]:
            assert c["cheapest_falsifier"]
            assert c["native_execution_concept"]
            assert c["forbids_dense_rematerialization"] is True
            assert c["status"] == "HYPOTHESIS_UNFITTED"
            assert c["evidence_class"] == "STATIC_ONLY"
            assert c["program_family"] in fs.PROGRAM_FAMILIES
            assert isinstance(c["storage_bytes"], int) and c["storage_bytes"] >= 0
            assert isinstance(c["flop_milli"], int) and c["flop_milli"] >= 0
            fs._require_axes(c)
