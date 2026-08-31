"""Pins for the protected-window eviction envelope.

A guard nobody has watched fail is not a guard. execute() raises without a
lease; the planner evicts when protected evidence requires it (convenience
cannot veto); an interrupted unload rolls back rather than leaving the
resident unloaded.
"""
from __future__ import annotations

import ast
import json
import pathlib

import pytest

from tools.future import protected_window as pw
from tools.future import qualification_pipeline as qp
from tools.future import repro_science as rs
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, _assert_no_hardware_claims


# ---------------------------------------------------------------------------
# Fixtures — never a live GPU, never a real HCLI lock seizure
# ---------------------------------------------------------------------------


def _batch(*, qwen: list[str] | None = None, cells: list[dict] | None = None, flash: list[str] | None = None) -> dict:
    qwen_ids = ["qwen27-fast-profile", "qwen27-affine2-splitk4"] if qwen is None else list(qwen)
    flash_ids = ["flash-p7-mhc-pre-simdgroup"] if flash is None else list(flash)
    cell_rows = [] if cells is None else list(cells)
    return {
        "source": "injected",
        "status": "WAITING_FOR_AUTHORITY",
        "qwen_singleton_order": qwen_ids,
        "qwen_cells_after": cell_rows,
        "flash_singleton_order": flash_ids,
        "flash_runnable_now": False,
        "lock_rels": list(pw.DEFAULT_LOCK_RELS),
        "n_qwen_singletons": len(qwen_ids),
        "n_qwen_cells_after": len(cell_rows),
        "n_flash_singletons": len(flash_ids),
    }


def _win(
    ident: str,
    *,
    candidate_id: str | None = "qwen27-fast-profile",
    measurement_class: str = "DIAGNOSTIC_RELATIVE",
    would_change: bool = True,
    **extra: object,
) -> dict:
    row = {
        "id": ident,
        "candidate_id": candidate_id,
        "decision_target": f"qualify:{candidate_id}" if candidate_id else ident,
        "measurement_class": measurement_class,
        "would_change_decision_if_protected": would_change,
        "apparent_outcome": "WIN",
    }
    row.update(extra)
    return row


def _lease(*, present: bool) -> dict:
    return {
        "kind": "READ",
        "present": present,
        "lock_rels": list(pw.DEFAULT_LOCK_RELS),
        "lock_file_exists": present,
        "holders": {"status": "OK" if present else "SKIPPED", "pids": [99] if present else []},
        "reason": "injected fixture lease" if present else "injected: no lease",
        "not_called": ["hcli.agentos.protected_accelerator_benchmark._try_lock", "fcntl.flock"],
        "executes_benchmark": False,
        "acquires_lease": False,
        "signals_process": False,
        "quiesces_worker": False,
        "gpu_authority": False,
        "execution_ok": present,
    }


def _qualification() -> dict:
    return {
        "composed": "tools.future.qualification_pipeline.run_pipeline",
        "n_stages": len(qp.STAGES),
        "planning_walk_complete": True,
        "execution_stop": {
            "stage_id": "identify_lease_availability",
            "stage_index": 1,
            "reason": "injected: no lease",
        },
        "lease_present": False,
        "contamination_class": "HEAVY",
        "survivor_ids": ["qwen27-fast-profile"],
        "dropped_ids": [],
        "measurement_class": "STATIC_ONLY",
        "gpu_authority": False,
        "executes_benchmark": False,
        "path_taken": "injected",
    }


def _world(**overrides: object) -> dict:
    kwargs: dict = {
        "live": False,
        "dry_run": True,
        "batch": _batch(),
        "lease": _lease(present=False),
        "dirty_wins": [],
        "dirty_units": [
            {"id": "dirty.a", "resource_class": "GPU_DIRTY_OK"},
            {"id": "plan.b", "resource_class": "STATIC_ANALYSIS"},
        ],
        "resident": {"busy": True, "convenience_weight": 10**9},
        "qualification": _qualification(),
        "mutate_occupancy": False,
        "contamination_class": "HEAVY",
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Receipt / shape
# ---------------------------------------------------------------------------


def test_build_emits_sealed_receipt():
    result = pw.run_window(**_world())
    out = pw.build(window=result)
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "PROTECTED_WINDOW_PLAN.json"
    assert doc["schema"] == pw.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]
    assert doc["resident_callable"]["entry_point"]
    assert "future.protected-window.plan" in doc["resident_callable"]["workunit_emitted"]
    assert doc["eras"] == list(pw.ERAS)
    assert doc["odysseys"] == list(pw.ODYSSEYS)
    assert doc["no_era_vi"] is True
    assert doc["no_odyssey_iv"] is True
    assert "VI" not in "".join(doc["eras"])
    assert "Odyssey IV" not in "".join(doc["odysseys"])
    assert "not its own civilization" in doc["fpga"]
    window = doc["window"]
    assert window["n_stages"] == len(pw.WINDOW_STAGES)
    assert [s["name"] for s in window["stages"]] == list(pw.WINDOW_STAGES)
    assert window["measurement_class"] == "STATIC_ONLY"
    assert window["gpu_authority"] is False
    _assert_no_hardware_claims(doc)


def test_selftest_aliases_build():
    assert pw.selftest is pw.build


def test_ten_envelope_stages_in_declared_order():
    result = pw.run_window(**_world())
    assert [s["name"] for s in result["stages"]] == list(pw.WINDOW_STAGES)
    assert [s["index"] for s in result["stages"]] == list(range(1, len(pw.WINDOW_STAGES) + 1))
    assert result["planning_walk_complete"] is True


def test_dry_run_stops_at_lease_when_no_lease():
    result = pw.run_window(**_world())
    stop = result["execution_stop"]
    assert stop["stage_id"] == "establish_protected_lease"
    assert result["lease_present"] is False
    lease_stage = result["stages"][5]
    assert lease_stage["name"] == "establish_protected_lease"
    assert lease_stage["payload"]["acquired"] is False
    assert lease_stage["payload"]["would_flock"] is False
    assert lease_stage["payload"]["kind"] == "REQUEST"
    assert lease_stage["execution_ok"] is False
    # Planning still walked restore so a refused lease cannot stall the mission.
    assert result["stages"][8]["name"] == "restore_resident"
    assert result["stages"][8]["payload"]["restored"] is True
    assert result["stages"][9]["name"] == "resume_mission"


def test_composes_qualification_pipeline_not_a_fork():
    result = pw.run_window(**_world())
    nested = result["stages"][6]
    assert nested["name"] == "run_staged_qualification"
    assert nested["payload"]["kind"] == "COMPOSE"
    assert "qualification_pipeline" in nested["payload"]["qualification"]["composed"]
    assert nested["payload"]["executes_benchmark"] is False
    source = pathlib.Path(pw.__file__).read_text()
    assert "tools.future.qualification_pipeline" in source
    assert "STAGES = (" not in source.split("WINDOW_STAGES", 1)[0]
    # Envelope does not reimplement the 13 qualification stage names as its own.
    assert "identify_lease_availability" not in pw.WINDOW_STAGES


def test_batch_order_is_codex_order_not_alphabetical():
    # Injected order matches Codex (fast-profile first), not sorted().
    order = [
        "qwen27-fast-profile",
        "qwen27-pipeline-state-elision",
        "qwen27-affine2-splitk4",
    ]
    result = pw.run_window(**_world(batch=_batch(qwen=order)))
    assert result["qwen_singleton_order"] == order
    assert result["qwen_singleton_order"] != sorted(order)
    composed = result["stages"][6]["payload"]["batch_order"]
    assert composed == order


def test_load_staged_batch_records_path_taken_and_derives_counts():
    loaded = pw.load_staged_batch()
    assert loaded["source"]
    n = loaded["n_qwen_singletons"]
    assert n == len(loaded["qwen_singleton_order"])
    assert loaded["n_flash_singletons"] == len(loaded["flash_singleton_order"])
    # Counts are derived, not a pinned convenience integer.
    assert loaded["flash_runnable_now"] is False
    if "qwen27-fast-profile" in loaded["qwen_singleton_order"]:
        assert loaded["qwen_singleton_order"][0] == "qwen27-fast-profile"


# ---------------------------------------------------------------------------
# Value estimate — decisions, not speedup
# ---------------------------------------------------------------------------


def test_value_estimate_counts_decisions_not_speedup():
    batch = _batch()
    wins = [
        _win("w1", candidate_id="qwen27-fast-profile", tps=12.0, apparent_speedup=1.4),
        _win("w2", candidate_id="qwen27-affine2-splitk4"),
        _win("w2-dup", candidate_id="qwen27-fast-profile"),  # same decision target
    ]
    est = pw.estimate_qualification_value(wins, batch, resident_convenience={"convenience_weight": 10**9})
    assert est["not_a_fabricated_speedup"] is True
    assert est["speedup_claimed"] is None
    assert est["resident_convenience_ignored"] is True
    assert est["expected_decisions_changed_if_continue_dirty"] == 0
    assert est["from_dirty_wins"] == 2  # unique decision targets
    assert "tps" not in json.dumps(est)
    assert 1.4 not in (est.get("from_dirty_wins"), est.get("expected_decisions_changed_if_protected"))
    _assert_no_hardware_claims(est)


def test_continue_dirty_changes_zero_decisions_even_with_many_wins():
    batch = _batch()
    wins = [_win(f"w{i}", candidate_id="qwen27-fast-profile") for i in range(20)]
    est = pw.estimate_qualification_value(wins, batch)
    assert est["n_dirty_wins"] == 20
    assert est["expected_decisions_changed_if_continue_dirty"] == 0
    assert est["from_dirty_wins"] == 1  # one unique target
    assert est["window_justified"] is True


def test_dirty_win_outside_batch_does_not_reorder_or_count():
    batch = _batch(qwen=["qwen27-fast-profile"])
    wins = [
        _win("in", candidate_id="qwen27-fast-profile"),
        _win("out", candidate_id="flash-p7-mhc-pre-simdgroup"),  # flash not runnable
        _win("invented", candidate_id="qwen27-brand-new-guess"),
    ]
    est = pw.estimate_qualification_value(wins, batch)
    assert est["from_dirty_wins"] == 1
    deferred_ids = {d["id"] for d in est["deferred"]}
    assert "out" in deferred_ids
    assert "invented" in deferred_ids
    result = pw.run_window(**_world(batch=batch, dirty_wins=wins))
    assert result["qwen_singleton_order"] == ["qwen27-fast-profile"]
    assert "qwen27-brand-new-guess" not in result["qwen_singleton_order"]


def test_would_not_change_decision_does_not_count():
    batch = _batch(qwen=["qwen27-fast-profile"])
    wins = [_win("noop", would_change=False)]
    est = pw.estimate_qualification_value(wins, batch)
    assert est["from_dirty_wins"] == 0
    # The undecided batch candidate still justifies a window.
    assert est["from_undecided_batch"] == 1
    assert est["window_justified_by_batch"] is True


def test_empty_batch_and_no_dirty_wins_is_not_justified():
    est = pw.estimate_qualification_value([], _batch(qwen=[], flash=[]))
    assert est["expected_decisions_changed_if_protected"] == 0
    assert est["window_justified"] is False


# ---------------------------------------------------------------------------
# No self-interest
# ---------------------------------------------------------------------------


def test_evicts_when_protected_evidence_requires_despite_huge_convenience():
    value = pw.estimate_qualification_value(
        [_win("w1")],
        _batch(),
        resident_convenience={"convenience_weight": 10**9, "busy": True},
    )
    decision = pw.decide_eviction(
        value,
        resident={
            "cost_to_unload": 10**12,
            "busy": True,
            "hot": True,
            "dirty_wins_per_hour": 10**6,
            "prefer_stay_loaded": True,
            "convenience_weight": 10**9,
            "resident_producing_dirty_wins": True,
        },
    )
    assert value["window_justified"] is True
    assert decision["evict"] is True
    assert decision["protected_evidence_requires"] is True
    assert decision["resident_convenience_vetoed"] is False
    assert decision["self_preference_path"] is False
    assert decision["resident_convenience_read"]["convenience_weight"] == 10**9


def test_no_self_preference_path_across_convenience_weights():
    value = pw.estimate_qualification_value([_win("w1")], _batch())
    weights = (0, 1, 10, 10**6, 10**9, -1, None)
    for weight in weights:
        for busy in (True, False):
            for prefer in (True, False):
                decision = pw.decide_eviction(
                    value,
                    resident={
                        "convenience_weight": weight,
                        "busy": busy,
                        "prefer_stay_loaded": prefer,
                        "hot": True,
                        "cost_to_unload": weight,
                    },
                )
                assert decision["evict"] is True, (weight, busy, prefer)
                assert decision["self_preference_path"] is False
                assert pw.resident_convenience_may_veto(decision["resident_convenience_read"]) is False
    assert pw._evict_from_protected_evidence(True) is True
    assert pw._evict_from_protected_evidence(False) is False


def test_does_not_evict_when_nothing_to_qualify_even_if_resident_is_idle():
    value = pw.estimate_qualification_value([], _batch(qwen=[], flash=[]))
    decision = pw.decide_eviction(
        value,
        resident={"convenience_weight": 0, "busy": False, "prefer_stay_loaded": False},
    )
    assert value["window_justified"] is False
    assert decision["evict"] is False
    assert decision["self_preference_path"] is False
    assert "resident preference" not in decision["reason"] or "not resident preference" in decision["reason"]
    # Adding a single dirty win on a runnable candidate flips eviction regardless of convenience.
    value2 = pw.estimate_qualification_value([_win("w1")], _batch())
    decision2 = pw.decide_eviction(value2, resident={"convenience_weight": 10**9, "prefer_stay_loaded": True})
    assert decision2["evict"] is True


def test_planner_on_window_walk_evicts_when_batch_is_waiting():
    result = pw.run_window(**_world(resident={"convenience_weight": 10**9, "busy": True}))
    assert result["evict"] is True
    assert result["self_preference_path"] is False
    assert result["stages"][1]["payload"]["evict"] is True


# ---------------------------------------------------------------------------
# No seizure
# ---------------------------------------------------------------------------


def test_every_execute_path_raises_without_an_existing_lease():
    with pytest.raises(pw.WindowRefused) as excinfo:
        pw.execute(
            explicit_execute=True,
            lease=_lease(present=False),
            contamination_class="QUIESCENT",
        )
    assert excinfo.value.condition == "existing_lease"

    with pytest.raises(pw.WindowRefused) as excinfo:
        pw.execute(
            explicit_execute=False,
            lease=_lease(present=True),
            contamination_class="QUIESCENT",
        )
    assert excinfo.value.condition == "explicit_execute"

    with pytest.raises(pw.WindowRefused) as excinfo:
        pw.acquire_lease()
    assert excinfo.value.condition == "lease_seizure"

    with pytest.raises(pw.WindowRefused):
        pw.seize_lease()

    with pytest.raises(qp.AuthorityBoundaryError, match="flock"):
        pw.refuse_flock()

    with pytest.raises(pw.WindowRefused) as excinfo:
        pw.run_window(**_world(dry_run=False, lease=_lease(present=False)))
    assert excinfo.value.condition == "existing_lease"


def test_execute_raises_on_each_named_condition_separately():
    with pytest.raises(pw.WindowRefused) as excinfo:
        pw.execute(explicit_execute=False, lease=_lease(present=True), contamination_class="QUIESCENT")
    assert excinfo.value.condition == "explicit_execute"

    with pytest.raises(pw.WindowRefused) as excinfo:
        pw.execute(explicit_execute=True, lease=_lease(present=False), contamination_class="QUIESCENT")
    assert excinfo.value.condition == "existing_lease"

    with pytest.raises(pw.WindowRefused) as excinfo:
        pw.execute(explicit_execute=True, lease=_lease(present=True), contamination_class="HEAVY")
    assert excinfo.value.condition == "machine_quiescence"

    with pytest.raises(pw.WindowRefused) as excinfo:
        pw.execute(explicit_execute=True, lease=_lease(present=True), contamination_class="QUIESCENT")
    assert excinfo.value.condition == "gpu_authority"


def test_live_lock_read_neither_creates_nor_mutates_and_never_claims_authority():
    """The load-bearing property is non-interference, not absence of the file.

    `.hcli/` may or may not be materialized in this sparse tree, and the parent
    checkout may hold Codex's lock with unproven holders. Assert what holds in
    both states.
    """
    lock = qp.REPO / qp.HCLI_LOCK_REL
    before = (
        lock.exists(),
        lock.stat().st_mtime if lock.exists() else None,
        lock.stat().st_size if lock.exists() else None,
    )
    state = pw.read_protected_locks()
    after = (
        lock.exists(),
        lock.stat().st_mtime if lock.exists() else None,
        lock.stat().st_size if lock.exists() else None,
    )
    assert after == before, "reading lock state must not create or mutate the lock"
    assert state["present"] is False
    assert state["acquires_lease"] is False
    assert state["gpu_authority"] is False
    assert state["kind"] == "READ"
    assert "never fcntl.LOCK_EX" in state["probe"]
    assert isinstance(state["primary_hcli_lock"]["lock_file_exists"], bool)


def test_refuse_functions_actually_fire():
    with pytest.raises(qp.AuthorityBoundaryError, match="start_benchmark"):
        pw.refuse_start_benchmark()
    with pytest.raises(qp.AuthorityBoundaryError, match="create_lease"):
        pw.refuse_create_lease()
    with pytest.raises(qp.AuthorityBoundaryError, match="signal_process"):
        pw.refuse_signal_process()
    with pytest.raises(qp.AuthorityBoundaryError, match="quiesce_worker"):
        pw.refuse_quiesce_worker()
    with pytest.raises(qp.AuthorityBoundaryError, match="flock"):
        pw.refuse_flock()


FORBIDDEN_CALLS = {
    "kill",
    "killpg",
    "flock",
    "lockf",
    "Popen",
    "run_protected_accelerator_benchmark",
    "_try_lock",
    "LOCK_EX",
    "SIGKILL",
    "SIGSTOP",
    "SIGTERM",
    "SingletonLease",
}
FORBIDDEN_IMPORTS = {
    "fcntl",
    "signal",
    "lab.lease",
    "hcli.agentos.protected_accelerator_benchmark",
    "hcli.agentos.native_mission_gate",
    "tools.future.codex_behaviors",
    "tools.future.resident_api",
    "tools.future.workgraph",
    "tools.future.detached",
    "tools.future.wakeup",
    "tools.future.evidence_dag",
    "tools.future.scar_scheduling",
    "tools.future.dirty_measure",
    "tools.future.sandbox",
    "tools.future.resident_identity",
    "tools.future.frontiers",
    "tools.future.succession",
    "tools.future.flash_schools",
    "tools.future.flash_nr_complete",
    "tools.future.super_resident",
    "tools.future.tabula",
    "tools.future.debugger",
    "tools.future.autonomy_trial",
    "tools.future.odyssey_launch",
}


def _imported_modules(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            for alias in node.names:
                out.add(f"{node.module}.{alias.name}")
    return out


def _called_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            out.add(func.id)
        elif isinstance(func, ast.Attribute):
            out.add(func.attr)
    return out


def test_module_does_not_import_lease_runner_or_concurrent_siblings():
    tree = ast.parse(pathlib.Path(pw.__file__).read_text())
    imported = _imported_modules(tree)
    for name in FORBIDDEN_IMPORTS:
        assert name not in imported, f"forbidden import {name}"
    assert any("qualification_pipeline" in m for m in imported)
    assert any("contamination" in m for m in imported)
    assert any("workunit_species" in m for m in imported)
    assert any("candidate_planner" in m for m in imported)
    assert any("repro_science" in m for m in imported)


def test_module_does_not_call_flock_or_signal():
    tree = ast.parse(pathlib.Path(pw.__file__).read_text())
    called = _called_names(tree)
    leaked = FORBIDDEN_CALLS & called
    assert not leaked, f"forbidden calls present: {sorted(leaked)}"
    tree_imports = _imported_modules(tree)
    assert "fcntl" not in tree_imports
    for name in (
        "acquire_lease",
        "execute",
        "run_window",
        "read_protected_locks",
        "_run_one_stage",
        "rollback",
        "main",
        "build",
    ):
        fn = getattr(pw, name)
        names = set(fn.__code__.co_names)
        hit = FORBIDDEN_CALLS & names
        assert not hit, f"{name} names forbidden {sorted(hit)}"


def test_every_stage_payload_declares_no_gpu_authority():
    result = pw.run_window(**_world())
    for rec in result["stages"]:
        payload = rec["payload"]
        assert payload["executes_benchmark"] is False, rec["name"]
        assert payload["acquires_lease"] is False, rec["name"]
        assert payload["signals_process"] is False, rec["name"]
        assert payload["quiesces_worker"] is False, rec["name"]
        assert payload["gpu_authority"] is False, rec["name"]


# ---------------------------------------------------------------------------
# Interrupted window rolls back
# ---------------------------------------------------------------------------


def test_interrupt_after_unload_rolls_back_rather_than_leaving_resident_unloaded():
    kwargs = _world(mutate_occupancy=True)
    with pytest.raises(pw.WindowInterrupted) as excinfo:
        pw.run_window(**kwargs, interrupt_after="pause_unload_resident")
    ck = excinfo.value.checkpoint
    assert rs.seal_is_valid(ck)
    occ = ck["occupancy"]
    assert occ["state"] == "UNLOADED"
    assert occ["resident_unloaded"] is True
    assert pw.resident_left_unloaded(occ) is True
    repaired = pw.rollback(ck)
    assert repaired["occupancy"]["resident_unloaded"] is False
    assert repaired["occupancy"]["state"] == "RESUMED"
    assert pw.resident_left_unloaded(repaired["occupancy"]) is False
    assert repaired["rollback"]["needed"] is True
    assert repaired["rollback"]["resident_left_unloaded"] is False
    # Idempotent.
    again = pw.rollback(repaired)
    assert again["occupancy"]["state"] == "RESUMED"
    assert again["occupancy"]["resident_unloaded"] is False


def test_resume_after_unload_interrupt_still_restores():
    kwargs = _world(mutate_occupancy=True)
    with pytest.raises(pw.WindowInterrupted) as excinfo:
        pw.run_window(**kwargs, interrupt_after="pause_unload_resident")
    ck = excinfo.value.checkpoint
    result = pw.run_window(**kwargs, resume_from=ck)
    assert result["planning_walk_complete"] is True
    assert result["stages"][4]["name"] == "pause_unload_resident"
    assert result["stages"][4]["status"] == "RESUMED"
    assert result["stages"][8]["name"] == "restore_resident"
    assert result["stages"][8]["payload"]["restored"] is True
    assert result["occupancy"]["resident_unloaded"] is False
    assert result["occupancy"]["state"] == "RESUMED"
    assert result["resident_left_unloaded"] is False


def test_interrupt_after_freeze_unfreezes_on_rollback_and_never_unloads():
    kwargs = _world(mutate_occupancy=True)
    with pytest.raises(pw.WindowInterrupted) as excinfo:
        pw.run_window(**kwargs, interrupt_after="freeze_safe_work")
    ck = excinfo.value.checkpoint
    assert ck["occupancy"]["state"] == "FROZEN"
    assert ck["occupancy"]["resident_unloaded"] is False
    assert "dirty.a" in ck["occupancy"]["frozen_work_ids"]
    repaired = pw.rollback(ck)
    assert repaired["occupancy"]["state"] == "CHECKPOINTED"
    assert repaired["occupancy"]["frozen_work_ids"] == []
    assert repaired["occupancy"]["resident_unloaded"] is False


def test_completed_mutated_walk_does_not_leave_resident_unloaded():
    result = pw.run_window(**_world(mutate_occupancy=True))
    assert result["occupancy"]["state"] == "RESUMED"
    assert result["occupancy"]["resident_unloaded"] is False
    assert result["resident_left_unloaded"] is False
    assert result["stages"][3]["payload"]["n_frozen"] == 1
    assert result["stages"][9]["payload"]["frozen_work_ids"] == []


def test_execute_restores_occupancy_if_called_while_unloaded():
    occ = pw.make_occupancy(state="UNLOADED", resident_unloaded=True, mutate=True)
    with pytest.raises(pw.WindowRefused):
        pw.execute(
            explicit_execute=True,
            lease=_lease(present=False),
            contamination_class="QUIESCENT",
            occupancy=occ,
        )
    assert occ["resident_unloaded"] is False
    assert occ["state"] == "RESUMED"


def test_corrupt_checkpoint_fails_closed():
    kwargs = _world(mutate_occupancy=True)
    with pytest.raises(pw.WindowInterrupted) as excinfo:
        pw.run_window(**kwargs, interrupt_after="checkpoint_resident")
    ck = dict(excinfo.value.checkpoint)
    ck["completed_stage_ids"] = list(ck["completed_stage_ids"]) + ["nope"]
    with pytest.raises(rs.FailClosed, match="corrupt_receipt"):
        pw.run_window(**kwargs, resume_from=ck)


def test_checkpoint_hole_fails_closed():
    kwargs = _world()
    with pytest.raises(pw.WindowInterrupted) as excinfo:
        pw.run_window(**kwargs, interrupt_after="decide_eviction")
    ck = {k: v for k, v in excinfo.value.checkpoint.items() if k != "seal_sha256"}
    ck["completed_stage_ids"] = ["estimate_qualification_value", "checkpoint_resident"]
    sealed = rs.seal_doc(ck)
    with pytest.raises(rs.FailClosed, match="stale_pipeline_cache"):
        pw.run_window(**kwargs, resume_from=sealed)


def test_partial_in_progress_both_completed_fails_closed():
    kwargs = _world()
    with pytest.raises(pw.WindowInterrupted) as excinfo:
        pw.run_window(**kwargs, interrupt_after="decide_eviction")
    body = {k: v for k, v in excinfo.value.checkpoint.items() if k != "seal_sha256"}
    body["in_progress_stage"] = "decide_eviction"
    sealed = rs.seal_doc(body)
    with pytest.raises(rs.FailClosed, match="partial_result"):
        pw.run_window(**kwargs, resume_from=sealed)


# ---------------------------------------------------------------------------
# Sleeping WorkUnits / fail closed
# ---------------------------------------------------------------------------


def test_sleeping_workunits_are_proposals_not_synthetic_results():
    result = pw.run_window(**_world())
    units = result["workunits"]
    assert units
    ids = {u["id"] for u in units}
    assert "future.protected-window.plan" in ids
    assert "future.protected-window.evict-resident" in ids
    assert "future.protected-window.staged-qualification" in ids
    assert "future.protected-window.restore-resident" in ids
    for unit in units:
        assert unit["status"] == "blocked"
        assert unit["classification"] == "SLEEPING"
        assert unit["sleeping"] is True
        assert "existing_hcli_lease" in unit["wake_when"]
        assert unit["claim_boundary"]
        assert "synthetic" not in (unit.get("blocked_reason") or "").lower() or "Not a synthetic" in (unit.get("blocked_reason") or "")


def test_receipt_cannot_carry_a_hardware_number():
    result = pw.run_window(**_world())
    _assert_no_hardware_claims(result)
    planted = {"tps": 12.0, "nested": {"bandwidth_gbps": 1}}
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims(planted)
    for field in HARDWARE_FIELDS:
        with pytest.raises(HardwareClaimError):
            _assert_no_hardware_claims({field: 1})


def test_pause_unload_never_claims_to_signal():
    result = pw.run_window(**_world(mutate_occupancy=True))
    unload = result["stages"][4]["payload"]
    assert unload["sidecar_will_signal"] is False
    assert unload["sidecar_will_unload"] is False
    assert unload["evict"] is True


def test_nested_qualification_does_not_execute_benchmark():
    result = pw.run_window(**_world())
    q = result["stages"][6]["payload"]["qualification"]
    assert q["gpu_authority"] is False
    assert q["executes_benchmark"] is False
    assert q.get("measurement_class", "STATIC_ONLY") == "STATIC_ONLY"
