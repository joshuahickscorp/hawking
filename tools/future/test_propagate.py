"""Tests for the Codex ingest propagator.

Includes a negative control the admission policy must refuse (a LAW delta
cannot be admitted at GENERIC_VERIFIED or any scope above MODEL_LOCAL, no
matter what the delta claims) and a negative control the idempotence guard
must be capable of firing (a second identical propagate() applies zero new
records).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import lpc_dataset as lpc
from tools.future import odyssey2_law_store as ols
from tools.future import odyssey3_adversary as o3
from tools.future import propagate as prop
from tools.future._common import RECEIPTS, HardwareClaimError, write_receipt


SHA_A = "aa" * 32
SHA_B = "bb" * 32
SHA_C = "cc" * 32


def _law_delta(
    *,
    source: str = "receipts/headless/TEST_LAW.json",
    sha: str = SHA_A,
    proposed_scope: str = "MODEL_LOCAL",
    knowledge_level: str = "INSTANCE",
    confidence: float = 1.0,
    statement: str = "fused mlp wins on this organ",
    model: str = "Qwen3.8-27B",
    organ: str = "mlp",
    spatially_meaningful: bool = True,
    evidence_strength: str | None = None,
) -> dict:
    cand = {
        "proposed_scope": proposed_scope,
        "scope_reason": f"test fixture knowledge_level={knowledge_level}",
        "sidecar_promotion_authority": False,
        "statement_sketch": statement,
        "model": model,
        "organ": organ,
        "knowledge_level": knowledge_level,
        "evidence_class": "STATIC_ONLY",
        "action": "admit as a candidate; do not promote past MODEL_LOCAL without independent evidence",
    }
    if evidence_strength is not None:
        cand["evidence_strength"] = evidence_strength
    return {
        "source": source,
        "source_sha256": sha,
        "classification": "LAW",
        "driver": {
            "confidence": confidence,
            "field": "status",
            "label": "LAW",
            "reason": "fixture PROTECTED_PASS",
            "token": "PROTECTED_PASS",
        },
        "odyssey_ii_law_candidate": cand,
        "odyssey_iii_attack_target": {
            "target": source,
            "attack": "refute, bound, or find the contamination in the claimed result",
            "suggested_angle": "transfer to a second model",
            "action": "register as an Odyssey III attack; do not treat the candidate as settled",
        },
        "architecture_atlas_behaviour_reference": {
            "behaviour": organ,
            "action": "cite as behaviour evidence; do not rewrite the atlas",
            "atlas_path": "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json",
        },
        "physical_graph_candidate_semantic": {
            "semantic_type": "PhysicalGraphPlan",
            "qualification": "PLAN_ONLY",
            "organ": organ,
            "action": "consider as a candidate semantic; sidecar does not compile a graph",
        },
        "learned_physical_compiler_row": {
            "source": source,
            "source_sha256": sha,
            "label": "LAW",
            "organ": organ,
            "model": model,
            "technique": "fixture",
            "representation": "UNKNOWN",
            "machine": "UNKNOWN",
            "contamination": "UNKNOWN",
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
            "measured": {
                "tps": None,
                "token_ns": None,
                "gpu_ns": None,
                "joules_per_token": None,
                "bandwidth_gbps": None,
            },
            "action": "append as a dataset-row skeleton; numbers remain UNKNOWN until a protected measurement",
        },
        "hwir_projection": {
            "spatially_meaningful": spatially_meaningful,
            "reason": "fixture spatial hint" if spatially_meaningful else "no spatial organ/primitive hint; HWIR projection is a stub",
            "organ": organ,
            "backend": "UNKNOWN",
            "action": "project onto HWIR once an IR exists" if spatially_meaningful else "no spatial mapping suggested",
        },
    }


def _scar_delta(
    *,
    source: str = "receipts/headless/TEST_SCAR.json",
    sha: str = SHA_B,
    organ: str = "mlp",
    token: str = "PROTECTED_REJECT",
    kills: list[str] | None = None,
) -> dict:
    return {
        "source": source,
        "source_sha256": sha,
        "classification": "SCAR",
        "driver": {
            "confidence": 1.0,
            "field": "status",
            "label": "SCAR",
            "reason": "fixture failed gate",
            "token": token,
        },
        "invalidation": {
            "kills": kills
            or [
                "fused mlp wins on this organ",
                "any hypothesis that this artifact is promotion-grade PROTECTED_ABSOLUTE evidence",
            ],
            "makes_redundant": [
                "retrying the same hypothesis on the same model/organ/machine without a new reopen condition"
            ],
            "reopen_condition": "a coherent body faster than the fixture",
            "level": "MODEL_SPECIFIC",
            "sidecar_must_not_promote": True,
            "organ": organ,
            "action": "feed the negative index; a single model's scar never globally prunes a technique",
        },
        "consumers_notified": [
            "odyssey_ii_law_store",
            "odyssey_iii",
            "negative_index",
            "learned_physical_compiler",
        ],
    }


def _run(deltas, *, apply=True, previous=None):
    return prop.propagate(deltas, apply=apply, previous=previous, load_previous=False)


def _o2_law(rec: dict) -> ols.Law:
    payload = {k: rec[k] for k in ols.LAW_FIELDS}
    payload["evidence_refs"] = tuple(payload["evidence_refs"])
    payload["transfer_candidates"] = tuple(payload["transfer_candidates"])
    return ols.validate_law(ols.Law(**payload))


# ---------------------------------------------------------------------------
# Entry point / receipt
# ---------------------------------------------------------------------------


def test_entry_point_emits_sealed_receipt(monkeypatch):
    """The CLI writes a sealed receipt (dry-run does not persist ledger)."""
    fixture = [_law_delta(), _scar_delta()]
    monkeypatch.setattr(prop, "load_ingest_deltas", lambda path=None: fixture)
    monkeypatch.setattr(prop, "RECEIPT", "_TEST_PROPAGATION_ENTRY.json")
    out = RECEIPTS / "_TEST_PROPAGATION_ENTRY.json"
    try:
        rc = prop.main(["--dry-run"])
        assert rc == 0
        doc = json.loads(out.read_text())
        assert out.parent == RECEIPTS
        assert out.name == "_TEST_PROPAGATION_ENTRY.json"
        assert doc["schema"] == prop.SCHEMA
        assert doc["version"] == 1
        assert doc["seal_sha256"]
        assert doc["bench"]["state"] == "UNKNOWN"
        assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
        assert doc["bench"]["gpu_authority"] is False
        assert doc["dry_run"] is True
        assert doc["recovered_implementation"]
        assert doc["gaps_closed"]
        assert doc["negative_findings"]
        assert doc["evidence_source"] in {"pinned_snapshot", "live_headless"}
        assert doc["admission_policy"]["odyssey2_scope"] == "MODEL_LOCAL"
        assert doc["admission_policy"]["odyssey2_evidence_strength"] == "ANECDOTE"
    finally:
        if out.exists():
            out.unlink()


def test_build_apply_writes_sealed_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(prop, "RECEIPT", "_TEST_PROPAGATION_STATE.json")
    monkeypatch.setattr(prop, "load_previous_apply", lambda path=None: None)
    out = prop.build(apply=True, deltas=[_law_delta(sha=SHA_C)])
    try:
        doc = json.loads(out.read_text())
        assert out.name == "_TEST_PROPAGATION_STATE.json"
        assert doc["schema"] == prop.SCHEMA
        assert doc["dry_run"] is False
        assert doc["seal_sha256"]
        assert doc["bench"]["state"] == "UNKNOWN"
        assert doc["totals"]["applied"] > 0
    finally:
        if out.exists():
            out.unlink()


# ---------------------------------------------------------------------------
# Negative control: LAW cannot be admitted above MODEL_LOCAL
# ---------------------------------------------------------------------------


def test_law_cannot_be_admitted_above_model_local_even_when_delta_claims_generic_verified():
    """NEGATIVE CONTROL: a heuristic LAW delta cannot mint GENERIC_VERIFIED.

    The delta claims GENERIC_VERIFIED, PROTECTED_ABSOLUTE, confidence=1.0, and
    knowledge_level GENERAL. Admission must still be MODEL_LOCAL / ANECDOTE,
    and promote() must refuse the claimed scope. A guard nobody has watched
    fail is not a guard.
    """
    delta = _law_delta(
        proposed_scope="GENERIC_VERIFIED",
        knowledge_level="GENERAL",
        confidence=1.0,
        evidence_strength="PROTECTED_ABSOLUTE",
    )
    result = _run([delta], apply=True)
    o2 = result["consumers"]["odyssey2_law_store"]
    laws = [r for r in o2["applied_records"] if r.get("law_id")]
    assert laws, "expected a MODEL_LOCAL candidate to be admitted"
    rec = laws[0]
    assert rec["scope"] == "MODEL_LOCAL"
    assert rec["evidence_strength"] == "ANECDOTE"
    assert rec["scope"] != "GENERIC_VERIFIED"
    assert rec["evidence_strength"] != "PROTECTED_ABSOLUTE"
    assert rec["transfer_confidence"]["value"] == pytest.approx(0.10)
    assert rec["time_to_first_useful_executable_ns"] is None

    scopes = {r.get("scope") for r in o2["applied_records"]}
    assert "GENERIC_VERIFIED" not in scopes
    assert "GENERIC_CANDIDATE" not in scopes
    assert "ARCHITECTURE_FAMILY" not in scopes
    assert "BACKEND_FAMILY" not in scopes
    assert "MACHINE_LOCAL" not in scopes

    assert o2["refused"] >= 1
    reasons = " ".join(r.get("reason") or "" for r in o2["refusals"])
    claimed = [r.get("claimed_scope") for r in o2["refusals"]]
    assert "GENERIC_VERIFIED" in claimed or "GENERIC_VERIFIED" in reasons
    assert any(r.get("stage") == "promote" for r in o2["refusals"])
    assert any(r.get("exception") == "ScopeViolation" for r in o2["refusals"])

    law = _o2_law(rec)
    with pytest.raises(ols.ScopeViolation) as ei:
        ols.promote(
            law,
            "GENERIC_VERIFIED",
            {
                "evidence_strength": "PROTECTED_ABSOLUTE",
                "models": ["Qwen3.8-27B", "Qwen/Qwen3.8-Flash-Next"],
                "architecture_families": ["dense_hybrid_transformer", "qwen4_exp"],
                "backends": ["Metal", "CUDA"],
                "machines": ["APPLE_GPU_0"],
                "counterexample_discharged": True,
                "evidence_refs": list(law.evidence_refs),
            },
        )
    assert ei.value.reason == "level_skip"
    assert ei.value.to_scope == "GENERIC_VERIFIED"

    o3b = result["consumers"]["odyssey3_adversary"]
    assert o3b["applied"] == 1
    assert o3b["applied_records"][0]["scope"] == "MODEL_LOCAL"
    assert o3b["applied_records"][0]["n_attacks"] == len(o3.ATTACK_FAMILIES)


def test_second_identical_propagate_applies_zero_new_records():
    """NEGATIVE CONTROL: idempotence must be capable of firing."""
    deltas = [
        _law_delta(proposed_scope="GENERIC_VERIFIED", knowledge_level="GENERAL", confidence=1.0),
        _scar_delta(),
    ]
    first = _run(deltas, apply=True)
    assert first["totals"]["applied"] > 0
    second = _run(deltas, apply=True, previous=first)
    assert second["totals"]["applied"] == 0
    assert second["totals"]["refused"] == 0
    assert second["totals"]["skipped_as_duplicate"] > 0
    assert second["ledger"]["applied_keys"] == first["ledger"]["applied_keys"]
    for name, bucket in second["consumers"].items():
        assert bucket["applied"] == 0, name
        assert bucket["applied_records"] == [], name


def test_dry_run_does_not_extend_durable_ledger():
    delta = _law_delta()
    dry = _run([delta], apply=False)
    assert dry["dry_run"] is True
    assert dry["ledger"]["applied_keys"] == []
    assert dry["totals"]["applied"] > 0
    applied = _run([delta], apply=True, previous=dry)
    assert applied["dry_run"] is False
    assert applied["totals"]["applied"] > 0
    assert applied["totals"]["skipped_as_duplicate"] == 0


# ---------------------------------------------------------------------------
# Routing / consumer APIs
# ---------------------------------------------------------------------------


def test_law_routes_to_seven_consumers():
    result = _run([_law_delta()], apply=True)
    for name in prop.SEVEN_CONSUMERS:
        assert result["consumers"][name]["applied"] >= 1, name
    assert result["consumers"]["negative_index"]["applied"] == 0


def test_scar_lands_in_negative_index_and_does_not_become_a_law():
    result = _run([_scar_delta()], apply=True)
    ni_b = result["consumers"]["negative_index"]
    assert ni_b["applied"] == 1
    rec = ni_b["applied_records"][0]
    assert rec["parse_status"] == "PARSED"
    assert rec["verdict"] == "PROTECTED_REJECT"
    assert rec["level"] == "MODEL_SPECIFIC"
    o2_laws = [r for r in result["consumers"]["odyssey2_law_store"]["applied_records"] if r.get("law_id")]
    assert o2_laws == []
    scars = [r for r in result["consumers"]["odyssey2_law_store"]["applied_records"] if r.get("kind") == "scar_invalidation"]
    assert scars


def test_scar_invalidates_overlapping_law_candidate():
    law = _law_delta(organ="mlp", statement="fused mlp wins on this organ")
    scar = _scar_delta(organ="mlp", kills=["fused mlp wins on this organ"])
    result = _run([law, scar], apply=True)
    assert result["n_scar_invalidations"] == 1
    report = result["scar_invalidations"][0]
    consumers_hit = {h["consumer"] for h in report["hits"]}
    assert "odyssey2_law_store" in consumers_hit
    o2 = [r for r in result["consumers"]["odyssey2_law_store"]["applied_records"] if r.get("law_id")][0]
    assert o2.get("scar_invalidated_by")


def test_lpc_row_is_valid_incomplete_and_refuses_zero_imputation():
    result = _run([_law_delta()], apply=True)
    rec = result["consumers"]["lpc_dataset"]["applied_records"][0]
    assert rec["status"] == "VALID"
    assert rec["complete"] is False
    assert rec["contamination_class"] == "STATIC_ONLY"
    assert rec["latency"] is None
    row = prop._lpc_row(_law_delta())
    assert lpc.validate_row(row)["status"] == "VALID"
    with pytest.raises(lpc.ImputationError):
        lpc.forbid_zero_imputation(row, "latency")
    with pytest.raises(lpc.ImputationError):
        lpc.forbid_zero_imputation(row, "dispatches")


def test_lpc_does_not_promote_contamination_class():
    delta = _law_delta()
    delta["learned_physical_compiler_row"]["contamination"] = "PROTECTED_ABSOLUTE"
    rec = _run([delta], apply=True)["consumers"]["lpc_dataset"]["applied_records"][0]
    assert rec["contamination_class"] == "STATIC_ONLY"


def test_non_spatial_hwir_is_recorded_not_validated_as_empty_graph():
    result = _run([_law_delta(spatially_meaningful=False)], apply=True)
    rec = result["consumers"]["hwir"]["applied_records"][0]
    assert rec["kind"] == "non_spatial_stub"
    assert rec["spatially_meaningful"] is False
    assert rec["qualification"] == "STATIC_ONLY"


def test_spatial_hwir_validates():
    result = _run([_law_delta(spatially_meaningful=True, organ="mlp")], apply=True)
    rec = result["consumers"]["hwir"]["applied_records"][0]
    assert rec["spatially_meaningful"] is True
    assert rec["validate_ok"] is True
    assert rec["n_nodes"] >= 1
    assert rec["qualification"] == "STATIC_ONLY"


def test_physical_graph_is_plan_only_and_does_not_rewrite_atlas():
    result = _run([_law_delta(organ="mlp")], apply=True)
    recs = result["consumers"]["physical_graph"]["applied_records"]
    kinds = {r["kind"] for r in recs}
    assert "plan_only" in kinds
    assert "atlas_citation" in kinds
    plan = next(r for r in recs if r["kind"] == "plan_only")
    assert plan["qualification"] == "PLAN_ONLY"
    assert plan["semantic_type"] == "PhysicalGraphPlan"


def test_workunit_is_emitted_through_hcli_constructor():
    result = _run([_law_delta()], apply=True)
    rec = result["consumers"]["workunit_species"]["applied_records"][0]
    assert rec["species"] == "odyssey_ii_transfer_experiment"
    assert rec["may_promote"] is False
    assert rec["status"] == "pending"
    scar = _run([_scar_delta()], apply=True)
    srec = scar["consumers"]["workunit_species"]["applied_records"][0]
    assert srec["species"] == "odyssey_iii_adversarial_experiment"


def test_tournament_stays_unrunnable():
    result = _run([_law_delta()], apply=True)
    rec = result["consumers"]["tournament"]["applied_records"][0]
    assert rec["can_run"] is False
    assert rec["reasons"]
    guard = result["tournament_run_guard"]
    assert guard["raised"] is True
    assert guard["exception"] == "TournamentNotReady"


def test_claimed_family_scope_is_refused_as_unknown_on_the_lattice():
    """Ingest emits proposed_scope FAMILY, which is not an Odyssey II lattice step."""
    delta = _law_delta(proposed_scope="FAMILY", knowledge_level="REPRESENTATION")
    result = _run([delta], apply=True)
    o2 = result["consumers"]["odyssey2_law_store"]
    rec = [r for r in o2["applied_records"] if r.get("law_id")][0]
    assert rec["scope"] == "MODEL_LOCAL"
    assert any(r.get("claimed_scope") == "FAMILY" for r in o2["refusals"])


def test_neutral_delta_is_not_applied():
    delta = {
        "source": "receipts/headless/TEST_NEUTRAL.json",
        "source_sha256": "dd" * 32,
        "classification": "NEUTRAL",
        "driver": {"confidence": 0.6, "field": "<default>", "label": "NEUTRAL", "reason": "no verdict", "token": "no_verdict"},
    }
    result = _run([delta], apply=True)
    assert result["totals"]["applied"] == 0
    assert result["by_classification"]["NEUTRAL"] == 1


def test_odyssey3_emit_for_law_is_the_public_api():
    result = _run([_law_delta()], apply=True)
    rec = result["consumers"]["odyssey3_adversary"]["applied_records"][0]
    assert rec["selected_attack_id"]
    assert rec["selected_family"] in o3.ATTACK_FAMILIES
    assert rec["evidence_class"] == "STATIC_ONLY"
    assert rec["bench_state"] == "UNKNOWN"


def test_does_not_write_consumer_receipts(monkeypatch, tmp_path):
    """Propagator must not rewrite another module's receipt."""
    monkeypatch.setattr(prop, "RECEIPT", "_TEST_PROPAGATION_ONLY.json")
    before = {
        name: (RECEIPTS / name).read_bytes() if (RECEIPTS / name).is_file() else None
        for name in (
            "ODYSSEY2_LAW_STORE.json",
            "ODYSSEY3_ADVERSARY.json",
            "LPC_DATASET.json",
            "HWIR_V1.json",
            "HCLI_FUTURE_WORKUNITS.json",
            "TOURNAMENT_READINESS.json",
            "PHYSICAL_PRIMITIVES.json",
            "NEGATIVE_SCIENCE_INDEX.json",
        )
    }
    out = prop.build(apply=True, deltas=[_law_delta(sha="ee" * 32), _scar_delta(sha="ff" * 32)])
    try:
        for name, blob in before.items():
            path = RECEIPTS / name
            if blob is None:
                # Sparse checkout may or may not have the sibling receipt; cope either way.
                continue
            assert path.read_bytes() == blob, name
        assert out.name == "_TEST_PROPAGATION_ONLY.json"
    finally:
        if out.exists():
            out.unlink()


def test_hardware_numbers_in_a_delta_are_not_sealed(tmp_path, monkeypatch):
    delta = _law_delta()
    # A number on the ingest skeleton must refuse the LPC copy, not land in the receipt.
    delta["learned_physical_compiler_row"]["measured"]["tps"] = 33.7
    result = _run([delta], apply=True)
    assert result["consumers"]["lpc_dataset"]["applied"] == 0
    assert result["consumers"]["lpc_dataset"]["refused"] == 1
    monkeypatch.setattr("tools.future._common.RECEIPTS", tmp_path)
    out = write_receipt("_prop_hw_guard.json", {"schema": "test", "version": 1, "body": result["consumers"]["lpc_dataset"]["refusals"]}, "test")
    assert json.loads(out.read_text())["seal_sha256"]


def test_receipt_refuses_hardware_field_numbers():
    with pytest.raises(HardwareClaimError):
        write_receipt("_must_not_exist_prop.json", {"schema": "test", "tps": 1.0}, "test")
    leaked = RECEIPTS / "_must_not_exist_prop.json"
    if leaked.exists():
        leaked.unlink()


def test_sorted_iteration_is_deterministic():
    a = _law_delta(source="receipts/headless/Z.json", sha=SHA_A)
    b = _law_delta(source="receipts/headless/A.json", sha=SHA_C)
    r1 = _run([a, b], apply=True)
    r2 = _run([b, a], apply=True)
    ids1 = [r["law_id"] for r in r1["consumers"]["odyssey2_law_store"]["applied_records"] if r.get("law_id")]
    ids2 = [r["law_id"] for r in r2["consumers"]["odyssey2_law_store"]["applied_records"] if r.get("law_id")]
    assert ids1 == ids2


def test_cope_with_missing_or_present_source_receipt():
    """Sparse checkout: a source path may or may not be on disk. Record the path taken."""
    delta = _law_delta(source="receipts/headless/DOES_NOT_NEED_TO_EXIST.json")
    ev = prop.resolve_evidence(delta["source"])
    assert ev["source"] in {"pinned_snapshot", "live_headless", "unresolved"}
    result = _run([delta], apply=True)
    assert result["totals"]["applied"] > 0
    # Presence of the original bytes is not required; the delta is self-contained.
    assert result["consumers"]["odyssey2_law_store"]["applied"] >= 1


def test_real_ingest_delta_shape_routes_when_present():
    """Copes with an empty or populated ingest state; does not assert absence."""
    deltas = prop.load_ingest_deltas()
    if not deltas:
        result = _run([], apply=True)
        assert result["n_deltas"] == 0
        return
    sample = [deltas[0]]
    result = _run(sample, apply=True)
    assert result["n_deltas"] == 1
    label = sample[0]["classification"]
    if label == "LAW":
        assert result["consumers"]["odyssey2_law_store"]["applied"] >= 1
        rec = [r for r in result["consumers"]["odyssey2_law_store"]["applied_records"] if r.get("law_id")][0]
        assert rec["scope"] == "MODEL_LOCAL"
        assert rec["evidence_strength"] == "ANECDOTE"
    elif label == "SCAR":
        assert result["consumers"]["negative_index"]["applied"] >= 1
