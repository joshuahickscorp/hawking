"""Pins for the qualification sequencer.

A guard nobody has watched fail is not a guard. execute() is proven to raise
on each of the three conditions separately, and every stage is proven not to
start a benchmark, create a lease, or signal a process.
"""
from __future__ import annotations

import ast
import json
import pathlib

import pytest

from tools.future import contamination as C
from tools.future import qualification_pipeline as qp
from tools.future import repro_science as rs
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, _assert_no_hardware_claims


# ---------------------------------------------------------------------------
# Fixtures — never a live GPU, never a real HCLI lock
# ---------------------------------------------------------------------------


def _cand(cid: str, *, status: str = "READY_PROTECTED", src: list[str] | None = None, **extra: object) -> dict:
    row = {
        "candidate_id": cid,
        "model": "Qwen27",
        "status": status,
        "affected_physical_region": f"region-{cid}",
        "dependencies": [],
        "blocked_reason": None,
        "parity_contract": "identical tokenizer-bound output ids; any divergence rejects",
        "capability_contract": "complete accepted-token capability, zero fallback",
        "control_configuration": {"child_fusion_env": {"HAWKING_QWEN38_FAST": "1"}},
        "exact_mutation": {"child_fusion_env": {"HAWKING_QWEN38_FAST": "1", f"HAWKING_{cid.upper().replace('-', '_')}": "1"}},
        "expected_dispatch_reduction": "0",
        "expected_eliminated_work": "none",
        "expected_intermediate_byte_reduction": "0",
        "expected_active_byte_change": "unchanged",
        "expected_gpu_ns_mechanism": "geometry only",
        "source_evidence": src or [f"synthetic/{cid}.metal"],
        "protected_command": ["python3", "-m", "hcli", "agentos", "protected-accelerator-bench", "--emit", f"{cid}.json"],
        "diagnostic_command": ["python3", "-m", "hcli", "agentos", "protected-accelerator-bench", "--measure-requests", "3"],
        "measurements": {"accepted_tps": None, "gpu_ns_per_token": None, "status": "NOT_MEASURED"},
    }
    row.update(extra)
    return row


def _stub_queue(rows: list[dict]) -> dict:
    ids = [r["candidate_id"] for r in rows]
    ready = [r["candidate_id"] for r in rows if r.get("status") == "READY_PROTECTED"]
    return {
        "schema": "hawking.accelerator.physical_qualification_queue.v1",
        "version": 1,
        "fingerprint": "test",
        "candidates": rows,
        "candidate_statuses": [
            "BLOCKED",
            "DIAGNOSTIC_PASS",
            "DIAGNOSTIC_REJECT",
            "INTEGRATED",
            "PROTECTED_PASS",
            "PROTECTED_REJECT",
            "READY_DIAGNOSTIC",
            "READY_PROTECTED",
            "STATIC_ONLY",
        ],
        "funnel": {
            "static_validation": ids,
            "native_parity": [],
            "diagnostic_relative_ab": [],
            "protected_absolute_complete_wall": ready,
            "promotion": [],
            "promotion_rule": "only a protected complete-token receipt may promote",
        },
        "status_transitions": {
            "STATIC_ONLY": ["BLOCKED", "READY_DIAGNOSTIC", "READY_PROTECTED"],
            "READY_DIAGNOSTIC": ["BLOCKED", "DIAGNOSTIC_PASS", "DIAGNOSTIC_REJECT"],
            "DIAGNOSTIC_PASS": ["BLOCKED", "READY_PROTECTED"],
            "DIAGNOSTIC_REJECT": ["BLOCKED", "STATIC_ONLY"],
            "READY_PROTECTED": ["BLOCKED", "PROTECTED_PASS", "PROTECTED_REJECT"],
            "PROTECTED_PASS": ["BLOCKED", "INTEGRATED"],
            "PROTECTED_REJECT": ["BLOCKED", "STATIC_ONLY"],
            "INTEGRATED": [],
            "BLOCKED": ["STATIC_ONLY"],
        },
        "queue_policy": {
            "planning_is_side_effect_free": True,
            "protected_start_requires_existing_hcli_lease": True,
            "protected_start_requires_machine_quiescence": True,
            "diagnostic_results_do_not_promote": True,
        },
        "measurement_contract": {
            "metric_scope": "accepted complete generated token",
            "protected_pass_requires_all_fields": True,
            "null_policy": "missing physical metrics remain null",
            "required_fields": [
                "accepted_tps",
                "gpu_ns_per_token",
                "complete_wall_ns_per_accepted_token",
                "fallback_count",
            ],
        },
        "_loaded_from": "fixture",
    }


def _proc(*, pid: int = 1, name: str = "idle", cpu: float = 0.0, rss: float = 0.1) -> dict:
    return {"pid": pid, "name": name, "cpu_pct": cpu, "rss_gib": rss, "state": "S"}


def _probes(*, processes: list[dict], load: dict | None = None, memory: dict | None = None) -> dict:
    return {
        "processes": {
            "status": "OK",
            "method": "ps_enumerate",
            "cpu_pct_available": True,
            "no_name_filter": True,
            "n_enumerated": len(processes),
            "all": processes,
            "reason": None,
        },
        "load": load
        or {"status": "OK", "load_1m": 0.2, "load_5m": 0.2, "load_15m": 0.2, "ncpu": 28},
        "memory": memory
        or {"status": "OK", "pressure_level": 0, "pressure_name": "normal", "pages": {}, "bytes": {}},
        "gpu_occupancy": {
            "status": "OK",
            "device_utilization_pct": 0,
            "renderer_utilization_pct": 0,
            "tiler_utilization_pct": 0,
            "gpu_core_count": 60,
        },
        "thermal": {"status": "UNKNOWN", "reason": "test"},
        "machine_identity": {"hash": "test-identity", "fields": {"hw.model": "test"}},
    }


def _quiet_snap() -> dict:
    return C.snapshot(
        benchmark_ordinal=None,
        probes=_probes(processes=[_proc(pid=2, name="zsh", cpu=0.1, rss=0.05)]),
    )


def _heavy_snap(*, name: str = "Python", rss: float = 19.0) -> dict:
    return C.snapshot(
        benchmark_ordinal=None,
        probes=_probes(
            processes=[
                _proc(pid=36753, name=name, cpu=96.3, rss=rss),
                _proc(pid=3, name="zsh", cpu=0.1, rss=0.05),
            ]
        ),
    )


def _preflight(*, errors: list[dict] | None = None) -> dict:
    findings = list(errors or [])
    return {
        "schema": "hawking.future.static_kernel_verify.v1",
        "blocking_defect_count": len(findings),
        "would_waste_a_protected_window": bool(findings),
        "counts": {"ERROR": len(findings), "WARNING": 0, "UNVERIFIABLE": 0, "INFO": 0},
        "findings": findings,
        "static_correctness_does_not_prove_speed": True,
    }


def _lease(*, present: bool) -> dict:
    return {
        "kind": "READ",
        "present": present,
        "lock_rel": qp.HCLI_LOCK_REL.as_posix(),
        "lock_file_exists": present,
        "holders": {"status": "OK" if present else "SKIPPED", "pids": [99] if present else []},
        "reason": "injected fixture lease" if present else "injected: no lease",
        "not_called": ["hcli.agentos.protected_accelerator_benchmark._try_lock"],
        "executes_benchmark": False,
        "acquires_lease": False,
        "signals_process": False,
        "quiesces_worker": False,
        "gpu_authority": False,
        "execution_ok": present,
    }


def _world(**overrides: object) -> dict:
    keep = _cand("qwen27-keep", src=["synthetic/keep.metal"])
    drop = _cand("qwen27-drop", src=["synthetic/drop.metal"])
    queue = _stub_queue([keep, drop])
    plan = qp.load_staged_plan(queue)
    kwargs = {
        "live": False,
        "dry_run": True,
        "queue": queue,
        "plan": plan,
        "snap": _quiet_snap(),
        "lease": _lease(present=False),
        "preflight": _preflight(
            errors=[
                {
                    "severity": "ERROR",
                    "check": "binding_index",
                    "kernel": "drop_k",
                    "shader": "synthetic/drop.metal:1",
                    "host": "synthetic/drop.rs:1",
                    "message": "off-by-one buffer index",
                }
            ]
        ),
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Receipt / shape
# ---------------------------------------------------------------------------


def test_build_emits_sealed_receipt():
    out = qp.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "QUALIFICATION_PIPELINE.json"
    assert doc["schema"] == qp.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["eras"] == list(qp.ERAS)
    assert doc["odysseys"] == list(qp.ODYSSEYS)
    assert doc["no_era_vi"] is True
    assert doc["no_odyssey_iv"] is True
    assert "VI" not in "".join(doc["eras"])
    assert "Odyssey IV" not in "".join(doc["odysseys"])
    assert "not its own civilization" in doc["fpga"]
    pipe = doc["pipeline"]
    assert pipe["n_stages"] == 13
    assert [s["name"] for s in pipe["stages"]] == list(qp.STAGES)
    assert pipe["execution_stop"]["stage_id"] == "identify_lease_availability"
    assert pipe["lease_present"] is False
    _assert_no_hardware_claims(doc)


def test_selftest_aliases_build():
    assert qp.selftest is qp.build


def test_thirteen_stages_in_declared_order():
    result = qp.run_pipeline(**_world())
    assert [s["name"] for s in result["stages"]] == list(qp.STAGES)
    assert [s["index"] for s in result["stages"]] == list(range(1, 14))
    assert result["planning_walk_complete"] is True
    assert result["measurement_class"] == "STATIC_ONLY"
    assert result["gpu_authority"] is False


def test_dry_run_stops_at_lease_check_when_no_lease():
    result = qp.run_pipeline(**_world())
    stop = result["execution_stop"]
    assert stop["stage_index"] == 1
    assert stop["stage_id"] == "identify_lease_availability"
    assert result["lease_present"] is False
    assert "no" in stop["reason"].lower() or "absent" in stop["reason"].lower() or "injected" in stop["reason"].lower()
    # Planning still walked every later spec.
    assert result["stages"][9]["name"] == "protected_lease_request"
    assert result["stages"][9]["status"] == "REQUEST_EMITTED"
    assert result["stages"][9]["payload"]["acquired"] is False
    assert result["stages"][9]["payload"]["kind"] == "REQUEST"


def test_live_lease_read_neither_creates_nor_mutates_the_hcli_lock():
    """The load-bearing property is non-interference, not absence.

    Whether the lock file exists is a fact about the checkout: `.hcli/` is not
    materialized in a sparse lane worktree, and in the primary worktree Codex's
    own lock has sat there since 2026-08-27. So assert what must hold in both --
    reading the lease state neither creates the file nor touches an existing one,
    and never yields authority.
    """
    lock = qp.REPO / qp.HCLI_LOCK_REL
    before = (lock.exists(), lock.stat().st_mtime if lock.exists() else None,
              lock.stat().st_size if lock.exists() else None)

    state = qp.read_hcli_lease_state()

    after = (lock.exists(), lock.stat().st_mtime if lock.exists() else None,
             lock.stat().st_size if lock.exists() else None)
    assert after == before, "reading lease state must not create or mutate the lock"

    # A lock whose holder cannot be proven without flock is NOT a held lease:
    # taking flock to find out would itself be the seizure.
    assert state["present"] is False
    assert state["acquires_lease"] is False
    assert state["gpu_authority"] is False
    assert isinstance(state["lock_file_exists"], bool)


# ---------------------------------------------------------------------------
# Composition: drop, A/B, workunits
# ---------------------------------------------------------------------------


def test_preflight_error_drops_mapped_candidate_only():
    result = qp.run_pipeline(**_world())
    drop_stage = result["stages"][4]
    assert drop_stage["name"] == "static_preflight_drop"
    payload = drop_stage["payload"]
    assert payload["n_dropped"] == 1
    assert payload["dropped"][0]["candidate_id"] == "qwen27-drop"
    assert "qwen27-keep" in payload["survivor_ids"]
    assert "qwen27-drop" not in payload["survivor_ids"]
    assert result["survivor_ids"] == ["qwen27-keep"]
    assert result["dropped_ids"] == ["qwen27-drop"]
    ab = result["stages"][5]["payload"]
    for cell in ab["cells"]:
        assert "qwen27-drop" not in cell["candidates"]
        assert cell["executes_benchmark"] is False


def test_unmapped_error_does_not_drop_untouched_candidate():
    keep = _cand("qwen27-keep", src=["synthetic/keep.metal"])
    queue = _stub_queue([keep])
    result = qp.run_pipeline(
        **_world(
            queue=queue,
            plan=qp.load_staged_plan(queue),
            preflight=_preflight(
                errors=[
                    {
                        "severity": "ERROR",
                        "check": "kernel_existence",
                        "kernel": "kv_append_q8_0_f32",
                        "shader": None,
                        "host": "crates/hawking-core/src/kernels/mod.rs:1",
                        "message": "host references kernel missing from shaders",
                    }
                ]
            ),
        )
    )
    payload = result["stages"][4]["payload"]
    assert payload["n_dropped"] == 0
    assert payload["survivor_ids"] == ["qwen27-keep"]
    assert payload["unmapped_blocking_defects"]
    assert payload["would_waste_a_protected_window"] is True


def test_ready_diagnostic_is_selected_alongside_ready_protected():
    rows = [
        _cand("qwen27-prot", status="READY_PROTECTED", src=["synthetic/a.metal"]),
        _cand("qwen27-diag", status="READY_DIAGNOSTIC", src=["synthetic/b.metal"]),
        _cand("flash-blocked", status="BLOCKED", src=["synthetic/c.metal"], model="Flash"),
    ]
    queue = _stub_queue(rows)
    result = qp.run_pipeline(
        **_world(queue=queue, plan=qp.load_staged_plan(queue), preflight=_preflight())
    )
    selected = result["stages"][3]["payload"]
    assert set(selected["candidate_ids"]) == {"qwen27-prot", "qwen27-diag"}
    assert "flash-blocked" not in selected["candidate_ids"]


def test_pausable_worker_is_identified_but_not_quiesced():
    snap = _heavy_snap(name="python lake_filler.py --fill", rss=3.0)
    result = qp.run_pipeline(**_world(snap=snap))
    worker = result["stages"][2]["payload"]["primary"]
    assert worker["class"] == "PAUSABLE"
    assert worker["policy_would_permit_quiesce"] is True
    assert worker["sidecar_will_quiesce"] is False
    assert result["stages"][2]["payload"]["sidecar_will_quiesce_any"] is False


def test_standing_python_neighbour_is_not_pausable():
    snap = _heavy_snap()
    result = qp.run_pipeline(**_world(snap=snap))
    worker = result["stages"][2]["payload"]["primary"]
    assert worker["class"] == "STANDING"
    assert worker["policy_would_permit_quiesce"] is False
    assert worker["sidecar_will_quiesce"] is False
    assert result["contamination_class"] == "HEAVY"
    assert result["stages"][1]["execution_ok"] is False


def test_promotion_gate_is_watched_to_refuse_static_only():
    result = qp.run_pipeline(**_world())
    promo = result["stages"][8]["payload"]
    assert promo["assert_promotable_refuses_this_sidecar"]["fired"] is True
    assert "STATIC_ONLY" in promo["assert_promotable_refuses_this_sidecar"]["message"]
    assert promo["sidecar_cannot_satisfy"] is True


def test_measurement_spec_empty_window_is_not_qualified_protected():
    result = qp.run_pipeline(**_world())
    spec = result["stages"][10]["payload"]
    assert spec["kind"] == "SPEC"
    assert spec["this_sidecar_emits"] == "STATIC_ONLY"
    assert spec["hcli_boundary"]["empty_window_is_not_qualified"] is True
    assert spec["hcli_boundary"]["empty_window_class"] != "QUALIFIED_PROTECTED"
    assert spec["required_fields"]["accepted_tps"] is None
    assert spec["required_fields"]["gpu_ns_per_token"] is None
    for cmd in spec["protected_commands"]:
        assert cmd["executed"] is False


def test_scoreboard_spec_does_not_write_and_has_no_hardware_numbers():
    result = qp.run_pipeline(**_world())
    spec = result["stages"][11]["payload"]
    assert spec["would_update"] is False
    assert spec["target_receipt"] == "receipts/headless/ACCELERATOR_SCOREBOARD.json"
    _assert_no_hardware_claims(spec)
    for row in spec["proposed_rows"]:
        for field in HARDWARE_FIELDS:
            assert row[field] is None


def test_next_workunits_are_proposals_not_dispatch():
    result = qp.run_pipeline(**_world())
    wu = result["stages"][12]["payload"]
    assert wu["does_not_schedule"] is True
    ids = {u["id"] for u in wu["units"]}
    assert "accelerator.physical.qwen27-keep" in ids
    assert "accelerator.physical.qwen27-drop" in ids
    assert "future.qualification.protected-lease-request" in ids
    for unit in wu["units"]:
        assert unit["status"] in {"blocked", "pending"}
        assert unit["claim_boundary"]


# ---------------------------------------------------------------------------
# Resumability + fail closed
# ---------------------------------------------------------------------------


def test_interrupt_then_resume_skips_completed_stages():
    entered: list[str] = []
    kwargs = _world()
    with pytest.raises(qp.PipelineInterrupted) as excinfo:
        qp.run_pipeline(**kwargs, interrupt_after="select_ready_candidates", on_stage=entered.append)
    assert entered == list(qp.STAGES[:4])
    ck = excinfo.value.checkpoint
    assert rs.seal_is_valid(ck)
    assert ck["completed_stage_ids"] == list(qp.STAGES[:4])
    resumed: list[str] = []
    result = qp.run_pipeline(**kwargs, resume_from=ck, on_stage=resumed.append)
    assert resumed == list(qp.STAGES[4:])
    assert result["stages"][0]["status"] == "RESUMED"
    assert result["stages"][0]["payload"]["present"] is False
    assert result["resumed_from_stage_count"] == 4
    assert result["planning_walk_complete"] is True
    assert result["stages"][4]["name"] == "static_preflight_drop"
    assert result["stages"][4]["status"] == "COMPLETED"


def test_corrupt_checkpoint_fails_closed():
    kwargs = _world()
    with pytest.raises(qp.PipelineInterrupted) as excinfo:
        qp.run_pipeline(**kwargs, interrupt_after="assess_machine_quiescence")
    ck = dict(excinfo.value.checkpoint)
    ck["completed_stage_ids"] = list(ck["completed_stage_ids"]) + ["nope"]
    with pytest.raises(rs.FailClosed, match="corrupt_receipt"):
        qp.run_pipeline(**kwargs, resume_from=ck)


def test_partial_in_progress_is_discarded_not_treated_as_success():
    kwargs = _world()
    with pytest.raises(qp.PipelineInterrupted) as excinfo:
        qp.run_pipeline(**kwargs, interrupt_after="assess_machine_quiescence")
    ck = dict(excinfo.value.checkpoint)
    body = {k: v for k, v in ck.items() if k != "seal_sha256"}
    body["in_progress_stage"] = "select_ready_candidates"
    sealed = rs.seal_doc(body)
    result = qp.run_pipeline(**kwargs, resume_from=sealed)
    # Partial stage was not in completed, so it runs (and later stages run).
    names_run_after_resume = [s["name"] for s in result["stages"] if s["status"] != "RESUMED"]
    assert "select_ready_candidates" in names_run_after_resume
    # A stage that is both completed and in_progress is refused.
    body2 = {k: v for k, v in ck.items() if k != "seal_sha256"}
    body2["in_progress_stage"] = "assess_machine_quiescence"
    sealed2 = rs.seal_doc(body2)
    with pytest.raises(rs.FailClosed, match="partial_result"):
        qp.run_pipeline(**kwargs, resume_from=sealed2)


def test_checkpoint_hole_fails_closed():
    kwargs = _world()
    with pytest.raises(qp.PipelineInterrupted) as excinfo:
        qp.run_pipeline(**kwargs, interrupt_after="select_ready_candidates")
    ck = {k: v for k, v in excinfo.value.checkpoint.items() if k != "seal_sha256"}
    ck["completed_stage_ids"] = [
        "identify_lease_availability",
        "select_ready_candidates",
    ]
    sealed = rs.seal_doc(ck)
    with pytest.raises(rs.FailClosed, match="stale_pipeline_cache"):
        qp.run_pipeline(**kwargs, resume_from=sealed)


# ---------------------------------------------------------------------------
# Negative control: three execute refusals, separately
# ---------------------------------------------------------------------------


def test_execute_raises_without_explicit_flag_even_with_lease_and_quiet():
    with pytest.raises(qp.ExecuteRefused, match="explicit --execute") as excinfo:
        qp.execute(
            explicit_execute=False,
            lease=_lease(present=True),
            contamination_class="QUIESCENT",
        )
    assert excinfo.value.condition == "explicit_execute"
    assert excinfo.value.fault == "explicit_execute"


def test_execute_raises_without_existing_lease_even_with_flag_and_quiet():
    with pytest.raises(qp.ExecuteRefused, match="no existing HCLI lease") as excinfo:
        qp.execute(
            explicit_execute=True,
            lease=_lease(present=False),
            contamination_class="QUIESCENT",
        )
    assert excinfo.value.condition == "existing_lease"


def test_execute_raises_when_machine_not_quiescent_even_with_flag_and_lease():
    with pytest.raises(qp.ExecuteRefused, match="machine is HEAVY") as excinfo:
        qp.execute(
            explicit_execute=True,
            lease=_lease(present=True),
            contamination_class="HEAVY",
        )
    assert excinfo.value.condition == "machine_quiescence"
    with pytest.raises(qp.ExecuteRefused, match="machine is LIGHT"):
        qp.execute(
            explicit_execute=True,
            lease=_lease(present=True),
            contamination_class="LIGHT",
        )
    with pytest.raises(qp.ExecuteRefused, match="machine is UNKNOWN"):
        qp.execute(
            explicit_execute=True,
            lease=_lease(present=True),
            contamination_class="UNKNOWN",
        )


def test_execute_still_raises_when_all_three_pass():
    with pytest.raises(qp.ExecuteRefused, match="no GPU authority") as excinfo:
        qp.execute(
            explicit_execute=True,
            lease=_lease(present=True),
            contamination_class="QUIESCENT",
        )
    assert excinfo.value.condition == "gpu_authority"


def test_refuse_functions_actually_fire():
    with pytest.raises(qp.AuthorityBoundaryError, match="start_benchmark"):
        qp.refuse_start_benchmark()
    with pytest.raises(qp.AuthorityBoundaryError, match="create_lease"):
        qp.refuse_create_lease()
    with pytest.raises(qp.AuthorityBoundaryError, match="signal_process"):
        qp.refuse_signal_process()
    with pytest.raises(qp.AuthorityBoundaryError, match="quiesce_worker"):
        qp.refuse_quiesce_worker()


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


def test_module_does_not_import_lease_runner_or_signal():
    tree = ast.parse(pathlib.Path(qp.__file__).read_text())
    imported = _imported_modules(tree)
    for name in FORBIDDEN_IMPORTS:
        assert name not in imported, f"forbidden import {name}"
    # Sibling composition is required.
    assert "tools.future.candidate_planner" in imported or any(
        m.startswith("tools.future") and "candidate_planner" in m for m in imported
    )
    assert any("contamination" in m for m in imported)
    assert any("static_kernel_verify" in m for m in imported)
    assert any("workunit_species" in m for m in imported)
    assert any("repro_science" in m for m in imported)
    assert any("benchmark_boundary" in m for m in imported)


def test_no_stage_function_can_start_benchmark_create_lease_or_signal():
    source = pathlib.Path(qp.__file__).read_text()
    tree = ast.parse(source)
    # Module-level calls.
    called = _called_names(tree)
    leaked = FORBIDDEN_CALLS & called
    assert not leaked, f"forbidden calls present: {sorted(leaked)}"
    # Per-stage function code objects.
    for name in (
        "read_hcli_lease_state",
        "assess_quiescence",
        "classify_worker",
        "select_ready_candidates",
        "drop_blocking_candidates",
        "emit_ab_plan",
        "parity_spec",
        "failure_classification_spec",
        "promotion_prerequisites_spec",
        "lease_request_spec",
        "protected_measurement_spec",
        "scoreboard_update_spec",
        "derive_next_workunits",
        "_run_one_stage",
        "run_pipeline",
        "execute",
        "main",
        "build",
    ):
        fn = getattr(qp, name)
        names = set(fn.__code__.co_names)
        hit = FORBIDDEN_CALLS & names
        assert not hit, f"{name} names forbidden {sorted(hit)}"
        # Stages must not call the runner even by string-constructed import.
        assert "run_protected_accelerator_benchmark" not in names


def test_every_stage_payload_declares_no_gpu_authority():
    result = qp.run_pipeline(**_world())
    for rec in result["stages"]:
        payload = rec["payload"]
        assert payload["executes_benchmark"] is False, rec["name"]
        assert payload["acquires_lease"] is False, rec["name"]
        assert payload["signals_process"] is False, rec["name"]
        assert payload["quiesces_worker"] is False, rec["name"]
        assert payload["gpu_authority"] is False, rec["name"]


def test_run_pipeline_does_not_call_execute():
    # Planning with a fake present lease + quiet machine still does not execute.
    result = qp.run_pipeline(**_world(lease=_lease(present=True), snap=_quiet_snap()))
    assert result["planning_walk_complete"] is True
    assert result["stages"][9]["payload"]["acquired"] is False
    assert result["stages"][9]["payload"]["kind"] == "REQUEST"
    # First execution blocker after a present lease is either quiescence (if
    # heavy) or the request stage. Quiet + present => request stage.
    stop = result["execution_stop"]
    assert stop["stage_id"] in {"protected_lease_request", "identify_lease_availability", "assess_machine_quiescence"}


def test_receipt_cannot_carry_a_hardware_number():
    result = qp.run_pipeline(**_world())
    _assert_no_hardware_claims(result)
    planted = {"tps": 12.0, "nested": {"bandwidth_gbps": 1}}
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims(planted)
    for field in HARDWARE_FIELDS:
        with pytest.raises(HardwareClaimError):
            _assert_no_hardware_claims({field: 1})


def test_pipeline_does_not_import_sibling_lanes_it_must_not_rewrite():
    tree = ast.parse(pathlib.Path(qp.__file__).read_text())
    imported = _imported_modules(tree)
    assert not any("global_frontier" in m for m in imported)
    assert not any("mutation_surface" in m for m in imported)
    assert not any("odyssey2_law_store" in m for m in imported)
