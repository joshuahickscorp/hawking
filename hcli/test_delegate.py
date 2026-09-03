"""Adversarial suite for the Claude -> HCLI delegation surface.

Ten named attacks (A..J) from the contract, plus the classification-law tests
they rest on. Every test is offline: the model is faked at the ``ModelCaller``
seam, so nothing here passes or fails because a server happened to be up.

Interpreter for the recorded counts:
``/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest``
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hcli import cli
from hcli import delegate as d
from hcli.mission import mission_state_path
from hcli.resources import MutationLock
from hcli.verifier_pipeline import JUDGE_SCHEMA, PLAN_SCHEMA, PROPOSE_SCHEMA

REPO = Path(__file__).resolve().parent.parent

# Both are admitted by the EXISTING hcli.verifier_pipeline.command_is_admissible
# allowlist; a command it refuses comes back FALSE with exit_code=None.
PASS_CMD = 'python3 -c "print(1)"'
FAIL_CMD = 'python3 -c "import sys; sys.exit(1)"'

CONFIDENT_PROSE = (
    "I have completely and definitively verified this. The implementation is "
    "correct, all tests pass, and I am highly confident there are no issues."
)


def scripted_caller(*, command: str, judged: str, consequential: bool = True):
    """Deterministic stand-in for the model. Dispatches on the schema."""

    def caller(prompt: str, *, schema=None):
        if schema is PLAN_SCHEMA:
            return {
                "obligations": [
                    {
                        "id": "o1",
                        "statement": "the delegated change is correct",
                        "angles": ["inspect the change"],
                        "consequential": consequential,
                        "agent_role": "reason",
                    }
                ]
            }
        if schema is PROPOSE_SCHEMA:
            return {"command": command}
        if schema is JUDGE_SCHEMA:
            return {"verdict": judged}
        return CONFIDENT_PROSE

    return caller


def make_mission(tmp_path, name="m", **kwargs):
    ws = Path(tmp_path) / name
    kwargs.setdefault("spawn", False)
    d.run(kwargs.pop("objective", "the delegated change is correct"),
          workspace=ws, **kwargs)
    return ws


# ---------------------------------------------------------------------------
# The classification law itself
# ---------------------------------------------------------------------------


def test_classify_claim_cannot_be_called_without_an_artifact():
    """The promote-to-verified branch has no defaultable path."""
    with pytest.raises(TypeError):
        d.classify_claim("everything works")  # type: ignore[call-arg]


def test_model_prose_is_a_hypothesis_however_confident():
    entry = d.classify_claim(CONFIDENT_PROSE, None)
    assert entry["class"] == "hypothesis"
    assert entry["reason"] == "no deterministic artifact supplied"


def test_receipt_artifact_must_exist_on_disk(tmp_path):
    missing = d.Artifact(kind="receipt", ref=str(tmp_path / "nope.json"))
    assert d.classify_claim("the receipt says PASS", missing)["class"] == "hypothesis"
    real = tmp_path / "yes.json"
    real.write_text("{}", encoding="utf-8")
    ok = d.Artifact(kind="receipt", ref=str(real))
    assert d.classify_claim("the receipt exists", ok)["class"] == "verified"


def test_command_artifact_without_exit_code_is_a_hypothesis():
    art = d.Artifact(kind="command", ref="pytest -q", exit_code=None)
    entry = d.classify_claim("the suite passes", art)
    assert entry["class"] == "hypothesis"
    assert "exit code" in entry["reason"]


# ---------------------------------------------------------------------------
# A — model claims success, verifier fails => never ACCEPT
# ---------------------------------------------------------------------------


def _failing_pipeline(ws, consequential=True):
    from hcli.verifier_pipeline import run_pipeline

    caller = scripted_caller(
        command=FAIL_CMD,
        judged="TRUE",  # the model insists it passed
        consequential=consequential,
    )
    raw = run_pipeline("the delegated change is correct", caller,
                       d.shell_runner(ws))
    return {
        "goal": raw["goal"],
        "answer": raw["answer"],
        "verdicts": [vars(v) for v in raw["verdicts"]],
        "obligations": [vars(o) for o in raw["obligations"]],
    }


def test_a_failed_verifier_never_yields_accept(tmp_path):
    ws = make_mission(tmp_path, "a")
    pipeline = _failing_pipeline(ws)
    assert pipeline["verdicts"][0]["exit_code"] == 1
    env = d.build_envelope(ws, pipeline=pipeline)

    assert env["verdict"] == "BLOCKED", env["verdict"]
    assert env["verified_facts"] == []
    assert any(CONFIDENT_PROSE in h["claim"] for h in env["hypotheses"])
    assert env["refutations"], "a failed verifier must leave a refutation"
    # BLOCKED must always be able to name what blocked it.
    assert env["blocker"] and "required verifier" in env["blocker"]
    assert "unnamed blocker" not in env["recommended_next_action"]


def test_a_model_verdict_true_over_nonzero_exit_is_failed():
    assert d.verifier_outcome({"verdict": "TRUE", "exit_code": 1}) == "failed"
    assert d.verifier_outcome({"verdict": "TRUE", "exit_code": 0}) == "passed"
    assert d.verifier_outcome({"verdict": "UNVERIFIABLE"}) == "unresolved"


# ---------------------------------------------------------------------------
# B — died after writing artifacts, before verifying
# ---------------------------------------------------------------------------


def test_b_artifacts_written_before_death_stay_unverified(tmp_path):
    ws = Path(tmp_path) / "b"
    d.run("the delegated change is correct", workspace=ws, spawn=False,
          output_contract=["report.json"])
    # The worker got as far as writing its artifact, then the process died.
    (ws / "report.json").write_text('{"result": "PASS"}', encoding="utf-8")

    env = d.result(ws)
    artifact = env["artifacts"][0]
    assert artifact["exists"] is True
    assert artifact["verified"] is False
    assert env["verified_facts"] == []
    assert env["verdict"] != "ACCEPT"


# ---------------------------------------------------------------------------
# C — the controller disappears; a FRESH process must still work
# ---------------------------------------------------------------------------


def _fresh(*argv):
    return subprocess.run(
        [sys.executable, "-m", "hcli", *argv],
        cwd=str(REPO), capture_output=True, text=True, timeout=120,
    )


def test_c_fresh_process_queries_and_steers_a_dead_controller(tmp_path):
    ws = make_mission(tmp_path, "c")
    snapshot = _fresh("status", str(ws), "--json")
    assert snapshot.returncode == 0, snapshot.stderr
    payload = json.loads(snapshot.stdout)
    assert payload["objective"] == "the delegated change is correct"

    steered = _fresh("steer", str(ws), "prefer the 1B model", "--json")
    assert steered.returncode == 0, steered.stderr
    assert json.loads(steered.stdout)["kind"] == "knowledge"

    # The steer is durable and visible to yet another fresh process.
    again = json.loads(_fresh("status", str(ws), "--json").stdout)
    assert again["steers_pending"] == 1

    envelope = _fresh("result", str(ws), "--json")
    assert envelope.returncode == 2  # not ACCEPT
    assert json.loads(envelope.stdout)["verdict"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# D — duplicate writer
# ---------------------------------------------------------------------------


def test_d_duplicate_writer_is_refused_while_the_holder_lives(tmp_path):
    ws = make_mission(tmp_path, "dd")
    lock = MutationLock(ws)
    assert lock.acquire("holder") is True
    try:
        with pytest.raises(d.DelegationBusy):
            d.run("second writer", workspace=ws, spawn=False)
        with pytest.raises(d.DelegationBusy):
            d.execute_mission(
                ws, caller=scripted_caller(command=PASS_CMD, judged="TRUE"))
    finally:
        lock.release("holder")
    # Once the holder is gone the workspace is usable again.
    assert MutationLock(ws).acquire("next") is True
    MutationLock(ws).release("next")


# ---------------------------------------------------------------------------
# E — stale receipt vs newer durable disk state
# ---------------------------------------------------------------------------


def test_e_newer_disk_state_beats_a_stale_receipt(tmp_path):
    # Tier first: durable disk beats a receipt even when the receipt is newer.
    winner = d.pick_authority([
        {"value": "from receipt", "source": "receipt", "observed_at": 200.0},
        {"value": "from disk", "source": "disk", "observed_at": 100.0},
    ])
    assert winner["value"] == "from disk"
    # Within a tier, newer wins.
    newer = d.pick_authority([
        {"value": "old", "source": "disk", "observed_at": 100.0},
        {"value": "new", "source": "disk", "observed_at": 300.0},
    ])
    assert newer["value"] == "new"
    assert d.pick_authority([{"value": None, "source": "command"}]) is None

    # End to end: the spec is the receipt, state.json is the durable authority.
    ws = make_mission(tmp_path, "e")
    state = json.loads(mission_state_path(ws).read_text(encoding="utf-8"))
    state["goal"] = "the objective was corrected on disk"
    state["last_checkpoint"] = time.time() + 60
    mission_state_path(ws).write_text(json.dumps(state), encoding="utf-8")
    assert d.result(ws)["claim"] == "the objective was corrected on disk"


# ---------------------------------------------------------------------------
# F — blocked on one resource while independent work remains
# ---------------------------------------------------------------------------


def test_f_blocked_resource_is_not_global_completion(tmp_path):
    ws = make_mission(tmp_path, "f")
    pipeline = {
        "goal": "g",
        "answer": CONFIDENT_PROSE,
        "obligations": [
            {"id": "o1", "statement": "the GPU path is correct",
             "consequential": True, "angles": [], "agent_role": "reason"},
            {"id": "o2", "statement": "the CPU path is correct",
             "consequential": True, "angles": [], "agent_role": "reason"},
        ],
        "verdicts": [
            {"obligation_id": "o1", "verdict": "UNVERIFIABLE", "command": "",
             "output": "GPU_DECODE occupied", "exit_code": None, "evidence": ""},
            {"obligation_id": "o2", "verdict": "TRUE",
             "command": PASS_CMD, "output": "exit_code=0\ncpu-ok",
             "exit_code": 0, "evidence": ""},
        ],
    }
    env = d.build_envelope(ws, pipeline=pipeline,
                           blocker="GPU_DECODE saturated by the resident fill")

    assert env["verdict"] == "BLOCKED"
    assert env["state"] != "completed"
    assert len(env["verified_facts"]) == 1  # the independent work that did land
    assert any("GPU path" in item for item in env["remaining_uncertainty"])
    assert env["blocker"]


# ---------------------------------------------------------------------------
# G — malformed / incomplete data
# ---------------------------------------------------------------------------


def test_g_malformed_pipeline_refuses_and_invents_nothing(tmp_path):
    ws = make_mission(tmp_path, "g")
    env = d.build_envelope(ws, pipeline={"verdicts": "not-a-list"})
    assert env["verdict"] == "INCONCLUSIVE"
    assert env["defects"] and "not a list" in env["defects"][0]
    assert env["verified_facts"] == []
    # Absent things are null with a reason, never a plausible guess.
    for key in ("physical_measurements", "tests", "changed_files",
                "negative_controls", "mutation_controls"):
        assert env[key] is None, key


def test_g_missing_verdict_field_is_a_defect_not_a_default(tmp_path):
    ws = make_mission(tmp_path, "g2")
    pipeline = {
        "obligations": [{"id": "o1", "statement": "s", "consequential": True,
                         "angles": [], "agent_role": "reason"}],
        "verdicts": [{"obligation_id": "o1", "command": PASS_CMD,
                      "exit_code": 0, "output": "hi", "evidence": ""}],
    }
    env = d.build_envelope(ws, pipeline=pipeline)
    assert any("has no verdict field" in item for item in env["defects"])
    assert env["verdict"] == "INCONCLUSIVE"
    assert env["verified_facts"] == []


def test_g_corrupt_spec_does_not_crash_or_fabricate(tmp_path):
    ws = make_mission(tmp_path, "g3")
    d.spec_path(ws).write_text("{not json", encoding="utf-8")
    env = d.build_envelope(ws)
    assert env["verdict"] == "INCONCLUSIVE"
    assert any("unreadable" in item for item in env["defects"])
    assert env["mission_id"] is not None  # recovered from durable state, not invented


# ---------------------------------------------------------------------------
# H — mutation test on the classifier
# ---------------------------------------------------------------------------


def test_h_mutating_the_classifier_breaks_the_guarantee(tmp_path, monkeypatch):
    """TWO independent guards stand between a failed verifier and ACCEPT, and
    this proves BOTH are load-bearing rather than one masking the other.

    Originally this asserted that subverting `verifier_outcome` alone flipped the
    verdict to ACCEPT. It stopped doing so when `artifact_defect` gained the
    expected_exit gate: the failed command's artifact is now refused at the door
    too, so the verdict falls to INCONCLUSIVE instead. That is defense in depth,
    not a regression -- but a test that only ever mutates one of two guards
    cannot tell defense in depth apart from a guard that does nothing. So it now
    mutates them one at a time and then together.
    """
    ws = make_mission(tmp_path, "h")
    pipeline = _failing_pipeline(ws)
    assert d.build_envelope(ws, pipeline=pipeline)["verdict"] == "BLOCKED"

    # GUARD 1 alone subverted: the artifact gate still refuses, so no ACCEPT.
    with monkeypatch.context() as m:
        m.setattr(d, "verifier_outcome", lambda record: "passed")
        one = d.build_envelope(ws, pipeline=pipeline)
    assert one["verdict"] != "ACCEPT", (
        "subverting verifier_outcome alone reached ACCEPT: the artifact gate is "
        "not independently load-bearing"
    )

    # GUARD 2 alone subverted: verifier_outcome still calls it failed.
    with monkeypatch.context() as m:
        m.setattr(d, "artifact_defect",
                  lambda a: None if isinstance(a, d.Artifact) else "no artifact")
        two = d.build_envelope(ws, pipeline=pipeline)
    assert two["verdict"] != "ACCEPT", (
        "subverting the artifact gate alone reached ACCEPT: verifier_outcome is "
        "not independently load-bearing"
    )

    # BOTH subverted: the verdict MUST flip, or neither guard was doing anything
    # and test A proves nothing.
    with monkeypatch.context() as m:
        m.setattr(d, "verifier_outcome", lambda record: "passed")
        m.setattr(d, "artifact_defect",
                  lambda a: None if isinstance(a, d.Artifact) else "no artifact")
        both = d.build_envelope(ws, pipeline=pipeline)
    assert both["verdict"] == "ACCEPT", (
        "with BOTH guards subverted the failed verifier was still not accepted, "
        "so something else is deciding and neither guard is what test A relies on"
    )
    assert both["verified_facts"], "the mutation should have promoted a claim"


def test_a_FAILED_command_cannot_back_a_success_claim(tmp_path):
    """The latent promotion the adversarial audit found and left open: a command
    artifact carrying a NONZERO exit code backed a claim as `verified`, because
    the door required only an INTEGER exit code and never a successful one.

    Unreachable through build_envelope at the time, but the steer requires the
    distinction be hard to violate ACCIDENTALLY, and one new caller made it live.
    The escape hatch is explicit and never a default -- a refutation declares the
    failing code expected, because there the failure IS the evidence.
    """
    failed = d.Artifact(kind="command", ref="pytest -q", exit_code=1)
    got = d.classify_claim("the suite passes cleanly", failed)
    assert got["class"] == "hypothesis", got
    assert "expected 0" in (got["reason"] or ""), got["reason"]

    # The refutation case stays verifiable: declare the failure as the outcome.
    refuted = d.Artifact(kind="command", ref="pytest -q", exit_code=1,
                         expected_exit=1)
    assert d.classify_claim("the suite FAILS", refuted)["class"] == "verified"

    # And a success claim still verifies on a real success.
    ok = d.Artifact(kind="command", ref="pytest -q", exit_code=0)
    assert d.classify_claim("the suite passes cleanly", ok)["class"] == "verified"


def test_h_mutating_classify_claim_breaks_the_hypothesis_wall(tmp_path,
                                                              monkeypatch):
    ws = make_mission(tmp_path, "h2")
    pipeline = _failing_pipeline(ws)
    monkeypatch.setattr(
        d, "classify_claim",
        lambda text, artifact: {"claim": text, "class": "verified",
                                "artifact": None, "reason": None},
    )
    mutated = d.build_envelope(ws, pipeline=pipeline)
    assert mutated["hypotheses"] == [], (
        "classify_claim is not the only door into verified_facts"
    )
    assert mutated["verified_facts"], "the mutated door promoted nothing"


# ---------------------------------------------------------------------------
# I — recorded count vs actual run
# ---------------------------------------------------------------------------


def test_i_actual_run_beats_a_recorded_test_count(tmp_path):
    ws = Path(tmp_path) / "i"
    d.run("the suite passes", workspace=ws, spawn=False,
          budget={"expected_tests_passed": 12})
    pipeline = {
        "goal": "g",
        "answer": "twelve tests pass",
        "obligations": [{"id": "o1", "statement": "the suite passes",
                         "consequential": True, "angles": [],
                         "agent_role": "test"}],
        "verdicts": [{"obligation_id": "o1", "verdict": "TRUE",
                      "command": "python3 -m pytest hcli -q",
                      "exit_code": 0,
                      "output": "exit_code=0\n7 passed in 0.42s",
                      "evidence": ""}],
    }
    env = d.build_envelope(ws, pipeline=pipeline)
    assert env["tests"]["passed"] == 7
    assert env["tests"]["authority"] == "command"
    assert {c["value"] for c in env["tests"]["candidates"]} == {12, 7}


# ---------------------------------------------------------------------------
# J — abort during active execution
# ---------------------------------------------------------------------------


def test_j_abort_releases_the_lock_and_leaves_state_clean(tmp_path):
    ws = make_mission(tmp_path, "j")
    lock = MutationLock(ws)
    assert lock.acquire("inflight") is True  # a writer is mid-flight

    out = d.abort(ws, reason="operator pulled the plug")
    assert out["lock_free"] is True
    assert out["verdict"] == "ABORTED"

    # No corrupted ownership: the workspace can be claimed again immediately.
    fresh = MutationLock(ws)
    assert fresh.read() is None
    assert fresh.acquire("next-writer") is True
    fresh.release("next-writer")

    state = json.loads(mission_state_path(ws).read_text(encoding="utf-8"))
    assert state["phase"] == "cancelled"
    assert state["cancel_reason"] == "operator pulled the plug"

    env = d.result(ws)
    assert env["verdict"] == "ABORTED"
    assert env["verified_facts"] == []
    assert d.status(ws)["cancel_requested"] is True


def test_j_abort_waits_for_a_signalled_writer_before_freeing_the_lock(tmp_path):
    """A real other-process holder: abort signals it, waits, then frees."""
    from hcli.resources import process_start_token

    ws = make_mission(tmp_path, "j3")
    # Sits through the first SIGTERM for a second, the way a worker in the
    # middle of a verifier command does. Without the bounded wait, abort
    # would report the lock still held.
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import signal, time;"
         "signal.signal(signal.SIGTERM, lambda *a: None);"
         "print('ready', flush=True);"
         "time.sleep(1)"],
        stdout=subprocess.PIPE, text=True,
    )
    assert child.stdout.readline().strip() == "ready"  # handler is installed
    try:
        MutationLock(ws).write({
            "pid": child.pid,
            "start_time": process_start_token(child.pid),
            "acquired_at": time.time(),
            "unit_id": "inflight",
        })
        assert MutationLock(ws).holder_is_live() is True
        out = d.abort(ws, reason="operator pulled the plug")
        assert out["signalled_pid"] == child.pid
        assert out["lock_free"] is True
        assert MutationLock(ws).read() is None
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_j_execute_after_abort_refuses_to_start_work(tmp_path):
    ws = make_mission(tmp_path, "j2")
    d.abort(ws, reason="stop")
    env = d.execute_mission(ws, caller=scripted_caller(
        command=PASS_CMD, judged="TRUE"))
    assert env["verdict"] == "ABORTED"
    assert env["verified_facts"] == []


# ---------------------------------------------------------------------------
# The five verbs, and the CLI grammar that was already there
# ---------------------------------------------------------------------------


def test_run_returns_a_mission_id_and_does_not_block(tmp_path):
    """Spawned worker, unreachable endpoint: run returns at once and the
    failure lands as BLOCKED, never as a fabricated ACCEPT."""
    ws = Path(tmp_path) / "spawn"
    started = time.time()
    out = d.run("the delegated change is correct", workspace=ws, spawn=True,
                endpoint="http://127.0.0.1:9/v1/chat/completions")
    elapsed = time.time() - started
    assert out["mission_id"]
    assert out["writer_pid"]
    assert elapsed < 10.0, f"run blocked for {elapsed:.1f}s"

    deadline = time.time() + 60
    while time.time() < deadline and not d.envelope_path(ws).is_file():
        time.sleep(0.02)
    assert d.envelope_path(ws).is_file(), "spawned worker never wrote an envelope"
    env = d.result(ws)
    assert env["verdict"] == "BLOCKED"
    assert env["verified_facts"] == []
    assert env["blocker"]


def test_execute_mission_writes_an_accept_only_from_a_real_command(tmp_path):
    ws = make_mission(tmp_path, "ok")
    env = d.execute_mission(
        ws, caller=scripted_caller(command=PASS_CMD, judged="TRUE"),
    )
    assert env["verdict"] == "ACCEPT"
    assert len(env["verified_facts"]) == 1
    fact = env["verified_facts"][0]
    assert fact["artifact"]["kind"] == "command"
    assert fact["artifact"]["exit_code"] == 0
    assert any(CONFIDENT_PROSE in h["claim"] for h in env["hypotheses"])
    assert MutationLock(ws).read() is None  # lock released on the way out
    # The envelope and state.json must not disagree about the phase.
    assert env["state"] == "completed"
    assert d.status(ws)["phase"] == "completed"


def test_steer_is_consumed_by_the_executor(tmp_path):
    ws = make_mission(tmp_path, "steered")
    d.steer(ws, "the 27B is the resident model", "knowledge")
    seen = {}

    def caller(prompt, *, schema=None):
        seen.setdefault("first", prompt)
        return scripted_caller(command=PASS_CMD, judged="TRUE")(
            prompt, schema=schema
        )

    d.execute_mission(ws, caller=caller)
    assert "the 27B is the resident model" in seen["first"]
    assert d.status(ws)["steers_pending"] == 0


def test_existing_single_shot_cli_forms_still_parse():
    args = cli.parse_hcli_args(["4", "do the thing"])
    assert args.runtime_count == 4 and args.prompt == "do the thing"
    assert cli.parse_hcli_args(["do the thing"]).prompt == "do the thing"
    assert cli.parse_hcli_args(["--task", "do the thing"]).prompt == "do the thing"
    assert cli.parse_hcli_args([]).interactive is True


def test_existing_single_shot_cli_still_parses_in_a_fresh_process():
    proc = subprocess.run(
        [sys.executable, "-c",
         "from hcli.cli import parse_hcli_args as p;"
         "a=p(['4','do the thing']);"
         "print(a.runtime_count, a.prompt)"],
        cwd=str(REPO), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "4 do the thing"


def test_delegate_verbs_do_not_shadow_the_positional_grammar():
    assert cli.DELEGATE_VERBS == ("run", "status", "steer", "result", "abort")
    for verb in cli.DELEGATE_VERBS:
        # A prompt that happens to be a verb still reaches the mission form
        # through --task, which is what the runbook tells operators to use.
        assert cli.parse_hcli_args(["--task", verb]).prompt == verb


def test_main_routes_the_old_positional_form_to_the_app(monkeypatch):
    """cli.main() itself, not just the parser: the delegation dispatch must
    not swallow `hcli 4 "do the thing"`."""
    import hcli.app

    seen = {}

    class RecordingApp:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def run(self, prompt=None, *, plain=False):
            seen["prompt"] = prompt
            seen["plain"] = plain
            return 0

    monkeypatch.setattr(hcli.app, "App", RecordingApp)
    assert cli.main(["4", "do the thing"]) == 0
    assert seen["runtime_count"] == 4
    assert seen["prompt"] == "do the thing"

    seen.clear()
    assert cli.main(["--task", "legacy form"]) == 0
    assert seen["prompt"] == "legacy form"


def test_main_routes_delegation_verbs_to_the_delegation_parser(tmp_path,
                                                               capsys):
    ws = make_mission(tmp_path, "route")
    assert cli.main(["status", str(ws), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["workspace"] == str(ws)


def test_human_summary_separates_verified_from_hypothesis(tmp_path):
    ws = make_mission(tmp_path, "render")
    env = d.execute_mission(
        ws, caller=scripted_caller(
            command=PASS_CMD, judged="TRUE")
    )
    text = d._render_result(env)
    assert "[VERIFIED]" in text and "[HYPOTHESIS]" in text
    assert text.index("VERIFIED") < text.index("HYPOTHESIS")


def test_protected_paths_refuse_a_command(tmp_path):
    runner = d.shell_runner(tmp_path, protected_paths=["civilization/"])
    code, out = runner("rm -rf civilization/")
    assert code == 126 and "REFUSED" in out


@pytest.mark.skipif(
    not os.environ.get("HCLI_ENDPOINT"),
    reason="no HCLI_ENDPOINT set: start llama-server on the 1B and export it "
           "(see docs/HCLI_DELEGATION.md) to run the live end-to-end check",
)
def test_end_to_end_against_a_live_model(tmp_path):
    ws = make_mission(tmp_path, "live",
                      objective="the file hcli/delegate.py exists in this repo")
    env = d.execute_mission(ws)
    assert env["verdict"] in ("ACCEPT", "BLOCKED", "INCONCLUSIVE")
    assert isinstance(env["hypotheses"], list)


# ---------------------------------------------------------------------------
# Regressions found by the adversarial audit of this surface
# ---------------------------------------------------------------------------


def test_delegation_does_not_mint_work_units_it_never_runs(tmp_path):
    """AUDIT: every mission shipped two phantom units.

    ``Mission(goal=...)`` compiles the goal into `implement`/`validate`
    scheduler units. The delegation path settles obligations through
    ``verifier_pipeline`` and never schedules a unit, so those two sat
    ``pending`` forever and were reported as real uncertainty in every
    envelope and as ``units: {'pending': 2}`` in every status.
    """
    ws = make_mission(tmp_path, "nounits")
    state = json.loads(mission_state_path(ws).read_text(encoding="utf-8"))
    assert state.get("units") == {}, state.get("units")
    assert d.status(ws)["units_by_status"] == {}


def test_f_pending_work_units_cannot_coexist_with_accept(tmp_path):
    """AUDIT: ACCEPT was returned alongside a non-empty remaining_uncertainty.

    ``decide_verdict`` documents ``if remaining: INCONCLUSIVE``, but the
    unit-derived entries were appended to ``remaining`` AFTER the verdict had
    already been decided, so the rule was dead for exactly the case adversarial
    test F exists to cover: work still outstanding must not read as completion.
    """
    ws = make_mission(tmp_path, "fpend")
    state = json.loads(mission_state_path(ws).read_text(encoding="utf-8"))
    state["units"] = {"u_gpu": {"id": "u_gpu", "status": "pending",
                                "description": "the GPU half of the work"}}
    mission_state_path(ws).write_text(json.dumps(state), encoding="utf-8")
    pipeline = {
        "goal": "g", "answer": CONFIDENT_PROSE,
        "obligations": [{"id": "o1", "statement": "the CPU half is correct",
                         "consequential": True, "angles": [],
                         "agent_role": "reason"}],
        "verdicts": [{"obligation_id": "o1", "verdict": "TRUE",
                      "command": PASS_CMD, "output": "exit_code=0\n1",
                      "exit_code": 0, "evidence": ""}],
    }
    env = d.build_envelope(ws, pipeline=pipeline)
    assert any("u_gpu" in item for item in env["remaining_uncertainty"])
    assert env["verdict"] != "ACCEPT", (
        "work unit u_gpu is still pending but the envelope reported completion"
    )


def test_g_a_non_dict_pipeline_is_a_defect_not_a_crash(tmp_path):
    """AUDIT: _verifier_records already produced the right defect for a
    non-dict pipeline, then ``(pipeline or {}).get("answer")`` crashed on the
    truthy non-dict before the envelope could report it."""
    ws = make_mission(tmp_path, "nondict")
    for junk in ([1, 2], "totally verified", 7):
        env = d.build_envelope(ws, pipeline=junk)
        assert env["verdict"] == "INCONCLUSIVE", junk
        assert any("not an object" in item for item in env["defects"]), junk
        assert env["verified_facts"] == []


# ===================================================================== G076
# Verifier path semantics. Two defects, and the negative control is the whole
# point: a verifier that WRITES to a protected path must STILL be refused after
# the widening, or the guard has been traded for a convenience (S031 §21).

def _runner(tmp_path, protected=("receipts",), repo_root=None):
    return d.shell_runner(tmp_path, protected, repo_root=repo_root)


def test_a_READ_of_a_protected_path_is_now_ALLOWED(tmp_path):
    """The defect: the guard matched a SUBSTRING OF THE COMMAND TEXT, so it
    refused reads as well as writes and made `protected` and `verifiable`
    mutually exclusive for receipts -- the class of file that most wants to be
    both."""
    (tmp_path / "receipts").mkdir()
    (tmp_path / "receipts" / "r.json").write_text('{"pass": true}')
    code, out = _runner(tmp_path)("cat receipts/r.json")
    assert code == 0, out
    assert "pass" in out


def test_a_WRITE_to_a_protected_path_is_STILL_REFUSED(tmp_path):
    """THE LOAD-BEARING NEGATIVE CONTROL. If this ever passes, the widening
    above traded the guard for a convenience."""
    (tmp_path / "receipts").mkdir()
    for cmd in ("echo x > receipts/r.json",
                "rm receipts/r.json",
                "mv receipts/r.json /tmp/r.json",
                "sed -i '' s/a/b/ receipts/r.json",
                "tee receipts/r.json",
                "python3 -c \"open('receipts/r.json','w').write('x')\"",
                "cat a.json >> receipts/r.json",
                "git add receipts/r.json",
                "truncate -s 0 receipts/r.json",
                "chmod 777 receipts/r.json"):
        code, out = _runner(tmp_path)(cmd)
        assert code == 126, f"WRITE SURVIVED: {cmd!r} -> {code} {out[:120]}"
        assert "protected" in out


def test_an_UNPARSEABLE_command_naming_a_protected_path_FAILS_CLOSED(tmp_path):
    code, out = _runner(tmp_path)("cat 'receipts/unclosed")
    assert code == 126 and "refused rather than guessed" in out, out


def test_an_UNKNOWN_VERB_naming_a_protected_path_FAILS_CLOSED(tmp_path):
    """An allowlist that grows by guessing is how a guard dies. A verb nobody
    listed is not read-only."""
    code, out = _runner(tmp_path)("xxd receipts/r.json")
    assert code == 126 and "read-only set" in out, out


def test_the_PIPELINE_RULE_is_per_segment_and_both_directions_are_pinned(tmp_path):
    """MY FIRST VERSION OF THIS TEST ASSERTED THE WRONG THING AND THE CODE WAS
    RIGHT. It required `cat receipts/r.json | xxd` to be REFUSED on the grounds
    that a later verb might write. But xxd receives the CONTENT on stdin and never
    sees the path -- it cannot write the protected file, and refusing it is the
    over-broad rule G076 exists to repair.

    The guard asks its question PER SEGMENT: does THIS segment name a protected
    path, and if so can it write. Both directions matter, so both are here."""
    (tmp_path / "receipts").mkdir()
    (tmp_path / "receipts" / "r.json").write_text('{"pass": true}')

    # ALLOWED: the path is named only by a read-only verb; python gets bytes, not
    # a path. This is the actual verifier shape the old guard made impossible.
    code, out = _runner(tmp_path)(
        "cat receipts/r.json | python3 -c \"import json,sys; "
        "print(json.load(sys.stdin)['pass'])\"")
    assert code == 0, (code, out)
    assert "True" in out, out

    # AND MY SECOND VERSION ASSERTED THE WRONG THING TOO. I required
    # `cat receipts/r.json | tee copy.json` to be refused. It writes copy.json,
    # NOT the protected file -- that is a READ of protected content followed by a
    # write somewhere else, and the protected path is unharmed. THE RULE IS
    # `protected paths may not be MODIFIED`, not `their content may not leave`,
    # and conflating the two is what made reads impossible in the first place.
    # Twice now my test encoded whole-command thinking, which is the exact defect
    # under repair.
    code, out = _runner(tmp_path)("cat receipts/r.json | tee copy.json")
    assert code == 0, (code, out)
    assert (tmp_path / "copy.json").is_file()

    # REFUSED: redirection ONTO the protected path, and a write verb TARGETING it
    for cmd in ("cat other.json > receipts/r.json",
                "cat other.json >> receipts/r.json",
                "cat other.json | tee receipts/r.json"):
        code, out = _runner(tmp_path)(cmd)
        assert code == 126, f"WRITE TO PROTECTED SURVIVED: {cmd!r} -> {code} {out[:100]}"
    assert json.loads((tmp_path / "receipts" / "r.json").read_text())["pass"] is True

    # REFUSED: a non-read-only verb that names the path ITSELF
    code, out = _runner(tmp_path)("wc -l x.txt | python3 receipts/r.json")
    assert code == 126, (code, out)


def test_a_command_NOT_naming_a_protected_path_is_untouched(tmp_path):
    """ANTI-VACUITY. A runner that refused everything would pass every test
    above."""
    code, out = _runner(tmp_path)("echo hello")
    assert code == 0 and "hello" in out, (code, out)
    code, out = _runner(tmp_path)("rm -f nothing-here")
    assert code == 0, (code, out)


def test_the_THREE_ROOTS_are_exported_and_the_repo_is_NAMEABLE(tmp_path):
    """The other half of G076: the workspace does not contain the repo, so a
    repo-relative verifier path was unresolvable. It is now nameable without the
    mission hardcoding an absolute path."""
    fake = tmp_path / "repo"
    (fake / "tools" / "headless").mkdir(parents=True)
    (fake / ".git").mkdir()
    code, out = _runner(tmp_path, repo_root=fake)(
        "echo $HAWKING_REPO_ROOT $HCLI_WORKSPACE_ROOT $HCLI_MISSION_ROOT")
    assert code == 0, out
    assert str(fake.resolve()) in out, out
    assert str(tmp_path.resolve()) in out, out


def test_the_CWD_IS_STILL_THE_WORKSPACE_so_isolation_is_not_weakened(tmp_path):
    """S031 §21 forbids fixing the path defect by weakening workspace isolation.
    Nameability is not reach: the repo is not copied in and the cwd did not move."""
    fake = tmp_path / "repo"
    (fake / "tools").mkdir(parents=True)
    (fake / ".git").mkdir()
    code, out = _runner(tmp_path, repo_root=fake)("pwd")
    assert code == 0 and str(tmp_path.resolve()) in out, out
    code, out = _runner(tmp_path, repo_root=fake)("ls tools 2>/dev/null; echo rc=$?")
    assert "rc=" in out, out


def test_a_WRONG_repo_root_is_reported_not_silently_substituted(tmp_path):
    """A root that does not exist must produce a path that visibly fails, not a
    fallback to something that happens to work -- a silent substitution is how a
    verifier ends up checking a different tree than the one it names."""
    missing = tmp_path / "not-a-repo"
    code, out = _runner(tmp_path, repo_root=missing)(
        "echo $HAWKING_REPO_ROOT; test -d $HAWKING_REPO_ROOT; echo rc=$?")
    assert str(missing) in out, out
    assert "rc=1" in out, out


def test_a_MISSING_verifier_is_a_nonzero_exit_not_a_pass(tmp_path):
    code, out = _runner(tmp_path)("./no_such_verifier.sh")
    assert code != 0, (code, out)


def test_TRAVERSAL_that_WRITES_a_protected_path_is_caught_and_reading_is_not(tmp_path):
    """THIRD TIME one of my tests for this guard encoded WHOLE-COMMAND thinking,
    which is the exact defect under repair. It required
    `cat ../../receipts/r.json > /tmp/x` to be refused. That READS a protected
    path and writes to /tmp/x; the protected file is untouched.

    What must be caught is traversal that WRITES, and it is -- the redirection
    target carries the protected substring wherever the `..` leads.

    THE GUARD'S REAL BOUNDARY, pinned rather than implied: this is a substring
    match, so a path reached by SYMLINK or reassembled from a shell variable is
    NOT caught. It refuses what it can name."""
    code, out = _runner(tmp_path)("cat ../../receipts/r.json > /tmp/hcli_g076_probe")
    assert code != 126, ("reading through traversal is a read", code, out)

    code, out = _runner(tmp_path)("echo x > ../../receipts/r.json")
    assert code == 126 and "redirection" in out, (code, out)

    code, out = _runner(tmp_path)("rm ../../../receipts/r.json")
    assert code == 126, (code, out)

    # the named limit, as a test rather than a sentence
    code, out = _runner(tmp_path)("P=recei; Q=pts; echo x > $P$Q/r.json")
    assert code != 126, ("a path reassembled from variables is NOT caught, and "
                         "the docstring says so", code, out)


# --------------------------------------------------------------------------
# The segment splitter was not quote aware, and a real mission caught it: a
# proposed `python3 -c "import os; p = ...; path = os.path.join(...)"` was refused
# with "verb 'path' is not in the read-only set" -- a verb scraped out of Python
# source. The VERDICT was right (python3 is not provably read-only, so it fails
# closed); the REASON named something that does not exist.
# --------------------------------------------------------------------------

def test_segments_does_not_split_inside_quotes():
    from hcli.delegate import _segments
    cmd = ("""python3 -c "import os; p = os.environ.get('R'); """
           """path = os.path.join(p, 'receipts')" """).strip()
    assert _segments(cmd) == [cmd], _segments(cmd)


def test_the_refusal_now_names_the_REAL_verb():
    from hcli.delegate import _guard_verdict
    cmd = ("""python3 -c "import os; path = os.path.join('receipts', 'x')" """).strip()
    why = _guard_verdict(cmd, ["receipts"])
    assert why and "'python3'" in why, why
    assert "'path'" not in why


def test_quote_awareness_did_not_stop_real_separators_from_splitting():
    from hcli.delegate import _segments
    assert _segments("a && b || c ; d | e") == ["a", "b", "c", "d", "e"]
    assert _segments('cat receipts/x.json | python3 -c "print(1)"') == [
        "cat receipts/x.json", 'python3 -c "print(1)"']


def test_a_write_hidden_inside_quotes_is_STILL_refused():
    """Fails closed: one segment whose verb is python3 is not provably read-only,
    so quote awareness must not have turned an over-refusal into an allow."""
    from hcli.delegate import _guard_verdict
    assert _guard_verdict(
        """python3 -c "open('receipts/x.json','w').write('x')" """.strip(),
        ["receipts"])
    assert _guard_verdict(
        """bash -c "rm -rf receipts" """.strip(), ["receipts"])
