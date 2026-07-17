#!/usr/bin/env python3.12
"""Leaf-module selftests + a full-stack fixture lifecycle (succ_* successor plane)."""
import pathlib
import sys

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import succ_admission as adm  # noqa: E402
import succ_transition as tr  # noqa: E402
import succ_watchdog as wd  # noqa: E402
import succ_telegram as tg  # noqa: E402
import succ_doctor as doc  # noqa: E402
import succ_gc as gc  # noqa: E402
import succ_eta as eta  # noqa: E402
import succ_state as st  # noqa: E402
import succ_engine as eng  # noqa: E402


# -- leaf selftests (each module's own battery) ---------------------------------------
def test_admission_selftest():
    r = adm.selftest()
    assert r["ok"] is True
    # source-bound probe: qwen method=none ready, lora not ready, gpt-oss not ready/reviewed
    assert r["qwen_none_ready"] is True and r["qwen_lora_ready"] is False
    assert r["gptoss_ready"] is False and r["gptoss_reviewed"] is False


def test_transition_selftest():
    r = tr.selftest()
    assert r["ok"] is True
    # the load-bearing fail-closed properties
    assert r["running_refused"] and r["second_execute_refused"] and r["all_pass_bypass_refused"]
    assert r["expired_refused"] and r["rollback_restores"]


def test_watchdog_selftest():
    assert wd.selftest()["ok"] is True


def test_telegram_selftest():
    r = tg.selftest()
    assert r["ok"] is True
    assert r["idempotent"] and r["redaction"] and r["corrupt_state_fail_closed"]


def test_doctor_selftest():
    r = doc.selftest()
    assert r["ok"] is True
    # honesty gate: unwired treatment hooks cannot be selected for execution
    assert r["blocked_hooks_not_selectable"] is True
    assert r["allocator_matches_brute_force"] is True


def test_gc_selftest():
    assert gc.selftest()["ok"] is True


def test_eta_selftest():
    assert eta.selftest()["ok"] is True


# -- full-stack fixture lifecycle (master goal 18.5, controller + engine form) ---------
def test_full_stack_controller_lifecycle(tmp_path):
    # boot -> wait_old_release (legacy running) -> heartbeat -> (release) -> import -> reconcile
    c = st.Controller(tmp_path / "gen", generation="gen")
    c.boot()
    c.transition("AUDIT")
    c.transition("WAIT_OLD_RELEASE", {"reason": "legacy running"})
    c.transition("WAIT_OLD_RELEASE", {"heartbeat": 1})
    # exact resume from a crash at this point
    assert st.Controller(tmp_path / "gen", generation="gen").resume()["resumed_state"] == "WAIT_OLD_RELEASE"
    # release: proceed through the science states to a lightweight launch
    c.transition("IMPORT_LEGACY")
    c.transition("RECONCILE")
    c.transition("FIT_PRIORS")
    c.transition("CHOOSE_PARENT")
    c.transition("BRACKET_HORIZON")
    c.transition("DIAGNOSE")
    c.transition("PRESCRIBE")
    c.transition("MATERIALIZE_PROGRAM")
    c.transition("VALIDATE_PROGRAM")
    c.transition("RESOURCE_ADMISSION")
    c.transition("LAUNCH")
    c.transition("MONITOR")
    c.transition("CHECKPOINT")
    c.transition("ATTEST")
    c.transition("EVALUATE")
    c.transition("INGEST_RESULT")
    c.transition("UPDATE_FRONTIER")
    c.transition("CHOOSE_NEXT")
    c.transition("SEALED_PARENT")
    assert c.current_state() == "SEALED_PARENT"
    ok, why = c.log.verify_chain()
    assert ok, why
    # a fresh handle resumes to SEALED_PARENT
    assert st.Controller(tmp_path / "gen", generation="gen").resume()["resumed_state"] == "SEALED_PARENT"


def test_engine_lightweight_dispatch_and_idempotent_ingest():
    plan = {"parents": [{
        "binding": {"model_label": "7B"}, "params_b": 7.6,
        "event_horizon_bracket": {"lowest_pass_bpw": 2.0},
        "boundary_probes_needed": [
            {"rate_bpw": 1.0, "current_verdict": "INCONCLUSIVE",
             "next_feasibility_tier": "F3_full_model_quality", "doctor_program": {"promote": []}}]}]}
    pick = eng.next_experiment(plan)
    admission = {"adapter_id": "a", "adapter_source_sha256": "a" * 64,
                 "ready_for_execution": True, "blockers": []}
    prog = eng.materialize_program(pick, admission, source_manifest_sha256="b" * 64,
                                   controls=["zero_treatment", "equal_byte_codec"])
    ok, why = eng.validate_program(prog, admission)
    assert ok, why
    res = eng.dispatch_lightweight(prog, "adapter.py", "capabilities",
                                   runner=lambda a: {"returncode": 0, "parsed": {"ok": True}})
    f = eng.ingest_result({}, res)
    assert eng.ingest_result(f, res)["last_action"] == "idempotent_skip"
