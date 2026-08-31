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
