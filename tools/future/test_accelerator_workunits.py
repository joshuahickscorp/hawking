"""Negative controls for the Accelerator-loop WorkUnit scheduler.

A GPU species emitted runnable, a missing input rounded into a pass, a
PROTECTED_AB pending under LIGHT, or next_species() returning [] are all P0.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tools.future import accelerator_workunits as aw
from tools.future import codex_behaviors as cb
from tools.future import contamination as C
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims
from tools.future import workunit_species as ws


def _handoff():
    doc, src = cb.load_handoff()
    return doc, src


def _visible(ok: bool = True) -> dict[str, bool]:
    rels = {rel for recs in aw.INPUT_RECEIPTS.values() for rel in recs}
    return {rel: ok for rel in rels}


def _queue(*rows: dict, **extra) -> dict:
    body = {
        "schema": "hawking.accelerator.physical_qualification_queue.v1",
        "candidates": list(rows),
        "_loaded_from": "injected",
    }
    body.update(extra)
    return body


def _cand(cid: str, status: str, **extra) -> dict:
    row = {
        "candidate_id": cid,
        "model": "Qwen27",
        "status": status,
        "affected_physical_region": "region-a",
        "dependencies": [],
        "blocked_reason": extra.pop("blocked_reason", None),
    }
    row.update(extra)
    return row


def test_required_species_are_exactly_the_loop_and_live_in_codex_behaviors():
    assert aw.REQUIRED_SPECIES == tuple(sid for wave in aw.LOOP_WAVES for sid in wave)
    assert len(set(aw.REQUIRED_SPECIES)) == len(aw.REQUIRED_SPECIES)
    doc, _src = _handoff()
    if doc is None:
        with pytest.raises(aw.InputRefused, match="CODEX_ACCELERATOR_HANDOFF.json"):
            aw.contracts()
        return
    cat = cb.catalog_by_id(handoff=doc)
    for sid in aw.REQUIRED_SPECIES:
        assert sid in cat
        assert sid in cb.SPECIES_IDS
    table = aw.contracts(handoff=doc)
    assert list(table) == list(aw.REQUIRED_SPECIES)
    for sid, row in table.items():
        assert row["gpu_authority"] is False
        assert row["evidence_class"] == "STATIC_ONLY"
        assert row["input_receipts"]
        assert row["output_receipt"]
        assert row["verifier"]
        assert row["fails_closed"]
        assert row["lane"]


def test_every_gpu_species_emitted_sleeping_none_runnable():
    """NEGATIVE CONTROL: a GPU species is SLEEPING, never pending, on this sidecar."""
    doc, _src = _handoff()
    if doc is None:
        with pytest.raises(aw.InputRefused, match="CODEX_ACCELERATOR_HANDOFF.json"):
            aw.emit_species("PROTECTED_AB", receipts_visible=_visible(True))
        return
    loop = aw.emit_loop(
        handoff=doc,
        receipts_visible=_visible(True),
        contamination_class="HEAVY",
        blockers=[],  # empty blockers must still sleep GPU species here
    )
    gpu = [u for u in loop["units"] if u.get("gpu_authority_required")]
    cpu = [u for u in loop["units"] if not u.get("gpu_authority_required")]
    assert gpu, "loop must emit GPU stages SLEEPING, not omit them"
    assert all(u["status"] == cb.STATUS_SLEEPING for u in gpu)
    assert all(u.get("runnable") is False for u in gpu)
    assert all(u.get("wake_condition") for u in gpu)
    assert all(u.get("gpu_authority") is False for u in gpu)
    runnable_gpu = [u["species"] for u in gpu if u.get("status") == "pending" or u.get("runnable") is True]
    assert runnable_gpu == []
    # CPU species with inputs present are the ones allowed to be pending.
    assert cpu
    assert all(u.get("gpu_authority_required") is False for u in cpu)
    assert all(u.get("status") != "failed" for u in loop["units"])
    assert all(str(u.get("status")).lower() != "skipped" for u in loop["units"])


def test_species_with_absent_input_receipt_refuses_naming_the_receipt():
    """NEGATIVE CONTROL: missing input is a named refusal, never a default success."""
    doc, _src = _handoff()
    if doc is None:
        with pytest.raises(aw.InputRefused, match="CODEX_ACCELERATOR_HANDOFF.json"):
            aw.emit_species("FIND_TALLEST_COST", receipts_visible={aw.HANDOFF_REL: False})
        return
    with pytest.raises(aw.InputRefused, match="ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"):
        aw.emit_species(
            "PROTECTED_AB",
            handoff=doc,
            receipts_visible={aw.QUEUE_REL: False},
            contamination_class="QUIESCENT",
        )
    with pytest.raises(aw.InputRefused, match="CODEX_ACCELERATOR_HANDOFF.json"):
        aw.emit_species(
            "FIND_TALLEST_COST",
            handoff=doc,
            receipts_visible={aw.HANDOFF_REL: False},
        )
    with pytest.raises(aw.InputRefused, match="STATIC_KERNEL_PREFLIGHT.json"):
        aw.emit_species(
            "HOST_SHADER_ABI_VERIFY",
            handoff=doc,
            receipts_visible={aw.PREFLIGHT_REL: False},
        )
    loop = aw.emit_loop(
        handoff=doc,
        receipts_visible=_visible(False),
        contamination_class="UNKNOWN",
    )
    assert loop["units"] == []
    assert loop["refusals"]
    named = {r["missing_receipt"] for r in loop["refusals"]}
    assert any(r and "ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json" in r for r in named)
    assert any(r and "CODEX_ACCELERATOR_HANDOFF.json" in r for r in named)


def test_protected_ab_never_runnable_under_light_or_worse():
    """NEGATIVE CONTROL: LIGHT/HEAVY/UNKNOWN cannot emit PROTECTED_AB as runnable."""
    doc, _src = _handoff()
    if doc is None:
        with pytest.raises(aw.InputRefused, match="CODEX_ACCELERATOR_HANDOFF.json"):
            aw.emit_species("PROTECTED_AB", receipts_visible=_visible(True))
        return
    for klass in ("LIGHT", "HEAVY", "UNKNOWN"):
        unit = aw.emit_species(
            "PROTECTED_AB",
            handoff=doc,
            receipts_visible=_visible(True),
            contamination_class=klass,
            blockers=[],
        )
        assert unit["status"] == cb.STATUS_SLEEPING, klass
        assert unit["runnable"] is False, klass
        assert unit["gpu_authority"] is False, klass
        assert klass in (unit.get("blocked_reason") or "") or klass in (unit.get("wake_condition") or "")
    # QUIESCENT is still not runnable: this sidecar has no GPU.
    quiet = aw.emit_species(
        "PROTECTED_AB",
        handoff=doc,
        receipts_visible=_visible(True),
        contamination_class="QUIESCENT",
        blockers=[],
    )
    assert quiet["runnable"] is False
    assert quiet["status"] == cb.STATUS_SLEEPING
    nxt = aw.next_species(
        _queue(_cand("qwen27-fast-profile", "READY_PROTECTED")),
        contamination_class="LIGHT",
        receipts_visible=_visible(True),
    )
    assert nxt["species"] == "PROTECTED_AB"
    assert nxt["runnable"] is False
    assert nxt["status"] == cb.STATUS_SLEEPING
    assert "LIGHT" in nxt["reason"]


def test_next_species_empty_queue_returns_reason_not_empty_list():
    """NEGATIVE CONTROL: empty queue is a named 'nothing runnable', never []."""
    ans = aw.next_species(_queue(), contamination_class="HEAVY", receipts_visible=_visible(True))
    assert ans != []
    assert not isinstance(ans, list)
    assert isinstance(ans, dict)
    assert ans.get("reason")
    assert ans.get("species") is None
    assert ans.get("runnable") is False
    assert "0 candidate" in ans["reason"]
    zero_counts = aw.next_species(
        {"status_counts": {"READY_PROTECTED": 0, "BLOCKED": 0, "STATIC_ONLY": 0}, "total_candidates": 0},
        contamination_class="UNKNOWN",
    )
    assert zero_counts != []
    assert isinstance(zero_counts, dict)
    assert zero_counts["reason"]
    assert zero_counts["runnable"] is False


def test_next_species_missing_queue_returns_reason_naming_the_receipt():
    ans = aw.next_species(
        None,
        contamination_class="HEAVY",
        receipts_visible={aw.QUEUE_REL: False},
    )
    # Live disk may still supply the queue via another worktree. Either path copes.
    assert isinstance(ans, dict)
    assert ans != []
    assert ans.get("reason")
    if ans.get("species") is None and "missing receipt" in ans["reason"]:
        assert aw.QUEUE_REL.split("/")[-1] in ans["reason"] or aw.QUEUE_REL in ans["reason"]
        assert ans["runnable"] is False
    else:
        assert "runnable" in ans
        assert ans.get("reason")


def test_next_species_all_blocked_returns_reason():
    q = _queue(
        _cand("flash-device-mhc-state", "BLOCKED", blocked_reason="Flash NX SCAFFOLD_ONLY"),
        _cand("flash-attention-gate-fusion", "BLOCKED", blocked_reason="Flash NX SCAFFOLD_ONLY"),
    )
    ans = aw.next_species(q, contamination_class="HEAVY", receipts_visible=_visible(True))
    assert isinstance(ans, dict)
    assert ans != []
    assert ans["runnable"] is False
    assert ans["species"] is None
    assert "BLOCKED" in ans["reason"]
    assert ans.get("reason")


def test_next_species_static_only_selects_cpu_generate():
    q = _queue(
        _cand("qwen27-gqa-qkv-fusion", "STATIC_ONLY"),
        _cand("qwen27-affine2-splitk4-vec", "STATIC_ONLY"),
    )
    ans = aw.next_species(q, contamination_class="HEAVY", receipts_visible=_visible(True))
    assert ans["species"] == "GENERATE_FUSION_CANDIDATE"
    assert ans["runnable"] is True
    assert ans["status"] == "pending"
    assert ans["gpu_authority"] is False
    layout = aw.next_species(
        _queue(_cand("qwen27-q4-vecgroup-x64", "STATIC_ONLY")),
        contamination_class="LIGHT",
        receipts_visible=_visible(True),
    )
    assert layout["species"] == "GENERATE_LAYOUT_CANDIDATE"
    assert layout["runnable"] is True


def test_next_species_ready_protected_is_protected_ab_sleeping():
    q = _queue(
        _cand("qwen27-fast-profile", "READY_PROTECTED"),
        _cand("qwen27-gqa-qkv-fusion", "READY_PROTECTED"),
        _cand("flash-compact-moe-epilogue", "BLOCKED"),
    )
    ans = aw.next_species(q, contamination_class="HEAVY", receipts_visible=_visible(True))
    assert ans["species"] == "PROTECTED_AB"
    assert ans["runnable"] is False
    assert ans["status"] == cb.STATUS_SLEEPING
    assert ans["queue"]["n_ready_protected"] == 2


def test_next_species_win_selects_reprofile_sleeping():
    q = _queue(_cand("qwen27-fast-profile", "PROTECTED_PASS"))
    ans = aw.next_species(q, contamination_class="QUIESCENT", receipts_visible=_visible(True))
    assert ans["species"] == "REPROFILE_AFTER_WIN"
    assert ans["runnable"] is False
    assert ans["status"] == cb.STATUS_SLEEPING


def test_next_species_generate_refuses_when_queue_receipt_injected_absent():
    q = _queue(_cand("qwen27-gqa-qkv-fusion", "STATIC_ONLY"))
    ans = aw.next_species(
        q,
        contamination_class="HEAVY",
        receipts_visible={aw.QUEUE_REL: False},
    )
    assert ans["runnable"] is False
    assert ans["status"] == "refused"
    assert "ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json" in ans["reason"]


def test_unknown_species_refuses():
    with pytest.raises(aw.InputRefused, match="unknown species"):
        aw.emit_species("INVENT_A_KERNEL", receipts_visible=_visible(True))


def test_unknown_contamination_class_refuses():
    doc, _src = _handoff()
    if doc is None:
        with pytest.raises(aw.InputRefused, match="CODEX_ACCELERATOR_HANDOFF.json"):
            aw.emit_species("PROTECTED_AB", receipts_visible=_visible(True), contamination_class="QUIETISH")
        return
    with pytest.raises(aw.InputRefused, match="unknown contamination_class"):
        aw.emit_species(
            "PROTECTED_AB",
            handoff=doc,
            receipts_visible=_visible(True),
            contamination_class="QUIETISH",
        )


def test_find_tallest_is_cpu_pending_when_inputs_present():
    doc, _src = _handoff()
    if doc is None:
        with pytest.raises(aw.InputRefused, match="CODEX_ACCELERATOR_HANDOFF.json"):
            aw.emit_species("FIND_TALLEST_COST", receipts_visible=_visible(True))
        return
    unit = aw.emit_species(
        "FIND_TALLEST_COST",
        handoff=doc,
        receipts_visible=_visible(True),
        contamination_class="HEAVY",
        blockers=[],
    )
    assert unit["gpu_authority_required"] is False
    assert unit["status"] == "pending"
    assert unit["runnable"] is True
    assert unit["gpu_authority"] is False
    from tools.future import qwen27_profile_schema as qps
    assert unit["profile_columns"] == list(qps.REQUIRED_METRICS)


def test_emitted_units_match_hcli_shape_and_carry_scheduler_fields():
    doc, _src = _handoff()
    if doc is None:
        with pytest.raises(aw.InputRefused, match="CODEX_ACCELERATOR_HANDOFF.json"):
            aw.emit_loop(receipts_visible=_visible(True))
        return
    loop = aw.emit_loop(handoff=doc, receipts_visible=_visible(True), contamination_class="LIGHT")
    assert loop["units"]
    for row in loop["units"]:
        ws.validate_emitted_unit(row)
        assert row["measurement_class"] == "STATIC_ONLY"
        assert row["bench_state"] == "UNKNOWN"
        assert row["gpu_authority"] is False
        assert row["input_receipts"]
        assert row["output_receipt"]
        assert row["fails_closed"]
        assert "runnable" in row
        for key in HARDWARE_FIELDS:
            assert not isinstance(row.get(key), (int, float))


def test_build_writes_sealed_static_only_receipt():
    out = aw.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "ACCELERATOR_WORKUNITS.json"
    assert doc["schema"] == aw.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["next_species"]["reason"]
    assert not isinstance(doc["next_species"], list)
    gpu_ids = set(doc["gpu_species"])
    for row in doc["emitted"]["units"]:
        if row["species"] in gpu_ids:
            assert row["status"] == cb.STATUS_SLEEPING
            assert row["runnable"] is False
    _assert_no_hardware_claims(doc)
    recovered = " ".join(doc["recovered_implementation"])
    assert "codex_behaviors.py" in recovered
    assert "workunit_species.py" in recovered
    assert "candidate_planner.py" in recovered
    assert "contamination.py" in recovered
    assert "protected_window.py" in recovered
    assert "qwen27_profile_schema.py" in recovered


def test_module_parses_and_contains_no_placeholder_tokens():
    src = Path(aw.__file__).read_text()
    ast.parse(src)
    for needle in ("raise NotImplementedError", "pytest.skip", "TODO"):
        assert needle not in src
    assert not any(line.strip() == "pass" for line in src.splitlines())


def test_contamination_classes_used_are_the_recovered_vocabulary():
    for klass in ("QUIESCENT", "LIGHT", "HEAVY", "UNKNOWN"):
        assert klass in C.CONTAMINATION_CLASSES
    assert aw.BLOCKED_CONTAMINATION == frozenset({"LIGHT", "HEAVY", "UNKNOWN"})
    assert "QUIESCENT" not in aw.BLOCKED_CONTAMINATION


# ---------------------------------------------------------------------------
# Procedure species: today's hand-composed treatment, frozen as HCLI units.
# ---------------------------------------------------------------------------


def test_seven_procedure_species_validate_through_wus():
    assert aw.PROCEDURE_SPECIES == (
        "ROOF_PROBE",
        "PRODUCTION_GAP_ATTRIBUTION",
        "ADDRESSING_AUDIT",
        "GEOMETRY_SWEEP",
        "CEREMONY_AUDIT",
        "PARITY_PROOF",
        "COMPLETE_TOKEN_REPROFILE",
    )
    catalog = aw.procedure_catalog()
    assert list(catalog) == list(aw.PROCEDURE_SPECIES)
    for sid in aw.PROCEDURE_SPECIES:
        spec = catalog[sid]
        assert spec["gpu_authority"] is False
        assert spec["evidence_class"] == "STATIC_ONLY"
        assert spec["does_not_execute"] is True
        assert spec["executed"] is False
        assert spec["input_receipts"]
        assert spec["output_receipt"]
        assert spec["acceptance"]
        assert spec["refusals"]
        assert spec["scar"]
        assert spec["verifier"]
        unit = aw.emit_procedure_unit(sid)
        ws.validate_emitted_unit(unit)
        assert unit["executed"] is False
        assert unit["does_not_execute"] is True
        assert unit["gpu_authority"] is False
        assert unit["claim_boundary"]
        assert "not execute" in unit["claim_boundary"].lower() or "EMITS" in unit["claim_boundary"]
        via_emit_species = aw.emit_species(sid)
        ws.validate_emitted_unit(via_emit_species)
        assert via_emit_species["species"] == sid


def test_procedure_gpu_species_sleeping_static_pending():
    for sid in aw.PROCEDURE_SPECIES:
        unit = aw.emit_procedure_unit(sid)
        if aw.procedure_gpu_required(sid):
            assert unit["status"] == cb.STATUS_SLEEPING, sid
            assert unit["runnable"] is False, sid
            assert unit["wake_condition"]
            assert unit["gpu_authority_required"] is True
        else:
            assert sid == "PRODUCTION_GAP_ATTRIBUTION"
            assert unit["gpu_authority_required"] is False
            assert unit["status"] == "pending"
            assert unit["runnable"] is True
            assert unit["executed"] is False


def test_roof_probe_refuses_unstated_roof():
    """SCAR: a ceiling with an unstated roof produced 595.9, then 589.73."""
    with pytest.raises(aw.UnstatedRoofRefused, match="unstated roof"):
        aw.accept_roof_probe({"ceiling": 595.9})
    with pytest.raises(aw.UnstatedRoofRefused, match="589.73"):
        aw.accept_roof_probe({"roof_id": ""})
    with pytest.raises(aw.UnstatedRoofRefused):
        aw.accept_roof_probe({})
    with pytest.raises(aw.UnstatedRoofRefused):
        aw.accept_roof_probe({"roof_id": None, "roof_gb_s": 589.73})
    ok = aw.accept_roof_probe({"roof_id": "mlp_arm_a_stripped_497p4", "roof_gb_s": 497.4})
    assert ok["accepted"] is True
    assert ok["named_roof"] is True


def test_production_gap_refuses_addr_probe_without_activation():
    """SCAR: 703.5 never loads the activation."""
    with pytest.raises(aw.ActivationNotLoadedRefused, match="never loads the activation"):
        aw.accept_production_gap_attribution(
            {
                "rungs": [
                    {
                        "id": "addr_probe",
                        "gb_s": 703.5,
                        "loads_activation": False,
                        "as_production_ceiling": True,
                    }
                ]
            }
        )
    with pytest.raises(aw.ActivationNotLoadedRefused, match="703.5"):
        aw.accept_production_gap_attribution({"treat_addr_probe_as_production": True})
    ok = aw.accept_production_gap_attribution(
        {
            "rungs": [
                {
                    "id": "addr_probe",
                    "gb_s": 703.5,
                    "loads_activation": False,
                    "as_production_ceiling": False,
                    "comparable_to_production_decode": False,
                },
                {
                    "id": "arm_a",
                    "gb_s": 497.4,
                    "loads_activation": True,
                    "as_production_ceiling": True,
                },
                {"id": "production_effective", "gb_s": 337.3, "loads_activation": True},
            ]
        }
    )
    assert ok["accepted"] is True
    assert ok["addr_probe_is_not_production_ceiling"] is True


def test_addressing_audit_refuses_stream_merge_promotion():
    """SCAR: stream count REFUTED at fixed bytes/thread; merging further HURTS."""
    with pytest.raises(aw.StreamMergePromotionRefused, match="merging further HURTS"):
        aw.accept_addressing_audit(
            {"promote_stream_merge": True, "bytes_per_thread_iteration_held": 38}
        )
    with pytest.raises(aw.StreamMergePromotionRefused, match="REFUTED"):
        aw.accept_addressing_audit({"verdict": "STREAM_COUNT_BOUND"})
    ok = aw.accept_addressing_audit(
        {
            "bytes_per_thread_iteration_held": 38,
            "verdict": "MIXED",
            "merge_hurts": True,
            "stream_count_refuted": True,
        }
    )
    assert ok["accepted"] is True
    assert ok["merging_further_hurts"] is True
    assert ok["stream_count_refuted"] is True


def test_geometry_sweep_refuses_sweep_without_discriminators():
    """SCAR: two slopes with a stall; NOT dependency, NOT register pressure, NOT occupancy."""
    with pytest.raises(aw.SweepWithoutDiscriminatorsRefused, match="discriminators"):
        aw.accept_geometry_sweep({"ladder": [1, 2, 3], "sweep_only": True})
    with pytest.raises(aw.SweepWithoutDiscriminatorsRefused, match="dependency"):
        aw.accept_geometry_sweep({"discriminators": {"register_pressure": 1.078, "occupancy": "worse"}})
    with pytest.raises(aw.SweepWithoutDiscriminatorsRefused, match="raising it is worse"):
        aw.accept_geometry_sweep({})
    ok = aw.accept_geometry_sweep(
        {
            "discriminators": {
                "dependency": 1.062,
                "register_pressure": 1.078,
                "occupancy": "raising_is_worse",
            }
        }
    )
    assert ok["accepted"] is True
    assert ok["occupancy_raising_is_worse"] is True
    assert set(ok["discriminators"]) == set(aw.GEOMETRY_DISCRIMINATORS)


def test_ceremony_audit_returns_bounded_too_small_and_refuses_continue():
    """SCAR: host class bounded at 0.989 ms; continuing to hunt it is refused."""
    with pytest.raises(aw.CeremonyContinueRefused, match="unnamed"):
        aw.accept_ceremony_audit({})
    stopped = aw.accept_ceremony_audit({"host_gap_ms": 0.989})
    assert stopped["verdict"] == aw.BOUNDED_TOO_SMALL
    assert stopped["stop"] is True
    with pytest.raises(aw.CeremonyContinueRefused, match="BOUNDED_TOO_SMALL"):
        aw.accept_ceremony_audit({"host_gap_ms": 0.9894, "continue_after_bound": True})
    with pytest.raises(aw.CeremonyContinueRefused, match="not the unlock"):
        aw.accept_ceremony_audit({"host_gap_ms": 0.9894, "unlock": "host_ceremony"})


def test_parity_proof_refuses_token_ids_alone():
    """SCAR: token ids identical, 22309 of 69632 intermediate bytes NOT."""
    with pytest.raises(aw.TokenIdOnlyParityRefused, match="necessary and not sufficient"):
        aw.accept_parity_proof({"token_ids_identical": True, "parity": True})
    with pytest.raises(aw.TokenIdOnlyParityRefused, match="22309"):
        aw.accept_parity_proof({"token_ids_identical": True, "token_ids_only": True})
    with pytest.raises(aw.TokenIdOnlyParityRefused):
        aw.accept_parity_proof({"token_ids_identical": True, "parity_basis": "token_id_equality"})
    well_formed_but_not_identical = aw.accept_parity_proof(
        {
            "token_ids_identical": True,
            "n_mismatch_bytes": 22309,
            "n_bytes_compared": 69632,
            "arithmetic_exact": False,
        }
    )
    assert well_formed_but_not_identical["accepted"] is True
    assert well_formed_but_not_identical["parity"] is False
    assert well_formed_but_not_identical["token_id_equality_is_not_sufficient"] is True
    assert well_formed_but_not_identical["blocks_promotion_until_accepted"] is True


def test_complete_token_reprofile_refuses_isolated_only():
    """SCAR: a probe is not a token (0.7046 became 1.0245; 1.745 became 3.9833)."""
    with pytest.raises(aw.IsolatedOnlyReprofileRefused, match="probe is not a token"):
        aw.accept_complete_token_reprofile({"isolated_ms": 1.745, "kind": "isolated"})
    with pytest.raises(aw.IsolatedOnlyReprofileRefused, match="1.745 became 3.9833"):
        aw.accept_complete_token_reprofile({"projection_ms": 1.745})
    with pytest.raises(aw.IsolatedOnlyReprofileRefused):
        aw.accept_complete_token_reprofile({})
    ok = aw.accept_complete_token_reprofile(
        {
            "complete_token_saving_ms": 3.9833,
            "isolated_ms": 1.745,
            "kind": "complete_token",
        }
    )
    assert ok["accepted"] is True
    assert ok["complete_token"] is True
    assert ok["probe_is_not_a_token"] is True


def test_reprofile_trigger_fires_on_live_stale_baseline():
    """fold_addqx removed 3.98 ms; PATH_TO_71 TOKEN_MS 28.722 is stale vs 26.3026."""
    live = aw.live_reprofile_trigger()
    assert live["fires"] is True
    assert live["species_to_emit"] == "COMPLETE_TOKEN_REPROFILE"
    assert live["old_decomposition_valid"] is False
    assert live["win_id"] == "fold_addqx"
    assert live["win_ms"] == aw.CITED_FOLD_ADDQX_COMPLETE_SAVING_MS
    assert live["baseline_token_ms"] == aw.CITED_PATH_TO_71_TOKEN_MS
    assert live["incumbent_token_ms"] == aw.CITED_FOLD_ADDQX_INCUMBENT_MS
    assert live["win_ms"] >= aw.LARGE_WIN_THRESHOLD_MS
    assert live["baseline_token_ms"] > live["incumbent_token_ms"]
    assert "stale" in live["reason"]
    assert live["does_not_execute"] is True
    quiet = aw.reprofile_trigger(
        win_ms=0.2,
        baseline_token_ms=28.722,
        incumbent_token_ms=26.3026,
        win_id="tiny",
    )
    assert quiet["fires"] is False
    current = aw.reprofile_trigger(
        win_ms=3.9833,
        baseline_token_ms=26.3026,
        incumbent_token_ms=26.3026,
        win_id="fold_addqx",
    )
    assert current["fires"] is False


def test_chain_emitted_for_live_frontier():
    chain = aw.emit_procedure_chain()
    assert chain["does_not_execute"] is True
    assert chain["executed"] is False
    assert [u["species"] for u in chain["units"]] == list(aw.PROCEDURE_SPECIES)
    by_species = {u["species"]: u for u in chain["units"]}
    roof = by_species["ROOF_PROBE"]
    gap = by_species["PRODUCTION_GAP_ATTRIBUTION"]
    assert roof["id"] in gap["dependencies"]
    parity = by_species["PARITY_PROOF"]
    assert parity["blocks_promotion_until_accepted"] is True
    geo = by_species["GEOMETRY_SWEEP"]
    ceremony = by_species["CEREMONY_AUDIT"]
    assert geo["id"] in parity["dependencies"]
    assert ceremony["id"] in parity["dependencies"]
    reprofile = by_species["COMPLETE_TOKEN_REPROFILE"]
    assert parity["id"] in reprofile["dependencies"]
    assert chain["reprofile_trigger"]["fires"] is True
    assert chain["live_frontier"]["win_id"] == "fold_addqx"
    for unit in chain["units"]:
        ws.validate_emitted_unit(unit)
        assert unit["executed"] is False
        assert unit["gpu_authority"] is False
        if unit["gpu_authority_required"]:
            assert unit["status"] == cb.STATUS_SLEEPING
            assert unit["runnable"] is False


def test_module_does_not_claim_to_execute():
    assert aw.DOES_NOT_EXECUTE is True
    src = Path(aw.__file__).read_text()
    assert "EMITS WorkUnits" in src or "EMITS HCLI WorkUnits" in src
    assert "does not execute them" in src.lower() or "Emitting them is not executing them" in src
    chain = aw.emit_procedure_chain()
    assert "Emitting them is not executing them" in chain["note"]
    assert chain["claim_boundary"]
    doc_boundary = aw.CLAIM_BOUNDARY.lower()
    assert "emit" in doc_boundary
    assert "does not execute" in doc_boundary


def test_build_receipt_includes_procedure_chain_and_trigger():
    out = aw.build()
    doc = json.loads(out.read_text())
    assert doc["does_not_execute"] is True
    proc = doc["procedure"]
    assert proc["does_not_execute"] is True
    assert proc["executed"] is False
    assert proc["species_ids"] == list(aw.PROCEDURE_SPECIES)
    assert proc["reprofile_trigger"]["fires"] is True
    assert proc["live_frontier"]["baseline_token_ms"] == 28.722
    assert proc["live_frontier"]["incumbent_token_ms"] == 26.3026
    assert proc["live_frontier"]["win_ms"] == 3.9833
    roof_before_gap = False
    for u in proc["chain"]["units"]:
        if u["species"] == "PRODUCTION_GAP_ATTRIBUTION":
            assert any("ROOF_PROBE" in d for d in u["dependencies"])
            roof_before_gap = True
        if u["species"] == "PARITY_PROOF":
            assert u["blocks_promotion_until_accepted"] is True
        if u["species"] == "COMPLETE_TOKEN_REPROFILE":
            assert any("PARITY_PROOF" in d for d in u["dependencies"])
        assert u["executed"] is False
    assert roof_before_gap
    _assert_no_hardware_claims(doc)
    assert "does not run them" in " ".join(doc["negative_findings"])
