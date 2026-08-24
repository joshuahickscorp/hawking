#!/usr/bin/env python3
"""Protected HCLI verifier-pipeline checks.

Plain python3 + assert. No pytest fixtures. Must also pass under pytest.
Must exit non-zero on failure and print one line per check.

No real model call, no GPU, no network. Every check supplies a deterministic
fake ModelCaller and a fake or trivial run_command.

These eight checks were watched FAILING against a naive first draft that
accepted task-list obligations, let execute return a verdict, skipped the
subprocess, promoted an empty command to TRUE, trusted the model's TRUE
against a nonzero exit, dropped a failed obligation, let synthesize take a
command runner, and ignored the test-role fast path. Observed FAIL text is
in the lane report.

Run:
    python3 tools/headless/hcli_verifier_pipeline_test.py
    pytest tools/headless/hcli_verifier_pipeline_test.py -q
"""
from __future__ import annotations

import inspect
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.verifier_pipeline import (  # noqa: E402
    Obligation,
    execute,
    plan,
    run_pipeline,
    synthesize,
    verify,
)

MODULE_PATH = Path(__import__("hcli.verifier_pipeline", fromlist=["x"]).__file__).resolve()


class FakeCaller:
    """Deterministic ModelCaller: records prompts and dispatches to a handler."""

    def __init__(self, handler):
        self.prompts = []
        self.schemas = []
        self.n = 0
        self.handler = handler

    def __call__(self, prompt: str, *, schema=None):
        self.n += 1
        self.prompts.append(prompt)
        self.schemas.append(schema)
        return self.handler(prompt, schema)


def _trivial_run(cmd: str):
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _obligation(**kwargs) -> Obligation:
    defaults = dict(
        id="o1",
        statement="foo.py exports add",
        angles=["read foo.py"],
        consequential=False,
        agent_role="read",
    )
    defaults.update(kwargs)
    return Obligation(**defaults)


def check_1_plan_rejects_task_list():
    """plan must reject 'Implement the retry logic' (retry once, then raise).

    Naive accept-anything plan() would return that statement as an obligation
    and this assertion would fail with:
        plan accepted imperative obligation: Implement the retry logic
    """

    def handler(prompt, schema):
        return {
            "obligations": [
                {
                    "id": "retry",
                    "statement": "Implement the retry logic",
                    "angles": ["read retry.py"],
                    "consequential": False,
                    "agent_role": "locate",
                }
            ]
        }

    caller = FakeCaller(handler)
    accepted = False
    raised = None
    result = None
    try:
        result = plan("add retries", caller)
        accepted = any(
            getattr(o, "statement", None) == "Implement the retry logic"
            for o in result
        )
    except Exception as exc:
        raised = exc
    assert not accepted, (
        "plan accepted imperative obligation: Implement the retry logic"
    )
    assert raised is not None, (
        "plan dropped the imperative without raising after retry"
    )
    assert caller.n >= 2, (
        f"plan did not retry before giving up (calls={caller.n})"
    )
    joined = "\n".join(caller.prompts)
    assert "Not a task list" in joined, (
        "plan prompt did not forbid a task list: "
        f"{caller.prompts!r}"
    )


def check_2_execute_never_returns_verdict():
    """execute returns a bare string; its prompt forbids TRUE/FALSE."""

    def handler(prompt, schema):
        return "/abs/path/foo.py:12: def add"

    caller = FakeCaller(handler)
    result = execute(_obligation(agent_role="read"), caller)
    assert isinstance(result, str), (
        f"execute returned {type(result).__name__}, not str: {result!r}"
    )
    assert not (isinstance(result, dict) and "verdict" in result), result
    joined = "\n".join(caller.prompts)
    assert caller.n >= 1, "execute never called the model"
    assert "TRUE" in joined and "FALSE" in joined, (
        f"execute prompt did not mention TRUE/FALSE: {joined!r}"
    )
    lowered = joined.lower()
    assert (
        "do not" in lowered
        or "must not" in lowered
        or "not render" in lowered
        or "never" in lowered
    ), f"execute prompt did not forbid rendering a verdict: {joined!r}"
    assert "concrete refs" in lowered, (
        f"execute prompt missing concrete-refs instruction: {joined!r}"
    )


def check_3_verify_actually_runs_command():
    """verify must run the proposed command; marker file is the proof."""
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "marker"
        inner = "open(%r, 'w').write('x')" % str(marker)
        cmd = "python3 -c " + repr(inner)

        def handler(prompt, schema):
            if "MECHANICALLY" in prompt or "cheapest command" in prompt.lower():
                return {"command": cmd, "output": "INVENTED-OUTPUT"}
            return {"verdict": "TRUE", "output": "INVENTED-OUTPUT"}

        caller = FakeCaller(handler)
        v = verify(
            _obligation(statement="the marker file is written", agent_role="settle"),
            "no evidence yet",
            caller,
            _trivial_run,
        )
        assert marker.is_file(), (
            f"verify did not run the proposed command; marker missing: {marker}"
        )
        assert v.command == cmd, (
            f"Verdict.command was not the real command: {v.command!r}"
        )
        assert "INVENTED-OUTPUT" not in (v.output or ""), (
            f"Verdict.output used the model's invented text: {v.output!r}"
        )


def check_4_empty_command_is_unverifiable():
    """Empty proposed command => UNVERIFIABLE; run_command is never invoked."""
    ran = []

    def run_command(cmd):
        ran.append(cmd)
        return 0, "should-not-run"

    def handler(prompt, schema):
        return {"command": "", "verdict": "TRUE", "output": "guessing"}

    caller = FakeCaller(handler)
    v = verify(
        _obligation(statement="the cosmos is fine-tuned", agent_role="reason"),
        "no mechanical evidence",
        caller,
        run_command,
    )
    assert v.verdict == "UNVERIFIABLE", (
        f"empty command was promoted to {v.verdict!r} instead of UNVERIFIABLE"
    )
    assert ran == [], (
        f"run_command was invoked for an empty command: {ran!r}"
    )
    assert v.command == "", f"expected empty command, got {v.command!r}"


def check_5_real_output_overrides_model_true():
    """Model saying TRUE against a nonzero exit is not trusted.

    Decision: OVERRIDE to FALSE. A nonzero exit from the command that was
    supposed to settle the claim cannot confirm it. The real (exit_code,
    output) is attached so the mismatch is auditable even if a later reader
    disagrees with the override.
    """
    ran = []

    def run_command(cmd):
        ran.append(cmd)
        return 1, "boom-real-nonzero"

    def handler(prompt, schema):
        if "MECHANICALLY" in prompt or "cheapest command" in prompt.lower():
            return {"command": "python3 -c 'import sys; sys.exit(1)'"}
        return {"verdict": "TRUE", "output": "the model invented success"}

    caller = FakeCaller(handler)
    v = verify(
        _obligation(statement="the module compiles", agent_role="settle"),
        "evidence-here",
        caller,
        run_command,
    )
    assert ran, "verify never ran the proposed command"
    assert "boom-real-nonzero" in (v.output or ""), (
        f"real output missing from Verdict: {v.output!r}"
    )
    assert v.verdict != "TRUE", (
        f"model said TRUE against nonzero exit and the module trusted it: {v!r}"
    )
    assert v.verdict == "FALSE", (
        f"expected mechanical override to FALSE, got {v.verdict!r}"
    )


def check_6_failed_obligation_is_not_dropped():
    """execute raising on 1 of 3 still yields 3 verdicts; the failed one is UNVERIFIABLE."""

    three = {
        "obligations": [
            {
                "id": "alpha",
                "statement": "Alpha module exports foo",
                "angles": ["read a.py"],
                "consequential": False,
                "agent_role": "read",
            },
            {
                "id": "bravo",
                "statement": "Bravo module exports bar",
                "angles": ["read b.py"],
                "consequential": False,
                "agent_role": "read",
            },
            {
                "id": "charlie",
                "statement": "Charlie module exports baz",
                "angles": ["read c.py"],
                "consequential": False,
                "agent_role": "read",
            },
        ]
    }

    def handler(prompt, schema):
        lowered = prompt.lower()
        execute_like = (
            "ATTACK IT THIS WAY" in prompt
            or "concrete refs" in lowered
            or "look into this:" in lowered
            or prompt.strip().startswith("OBLIGATION:")
        )
        if execute_like:
            if "Bravo" in prompt:
                raise RuntimeError("execute exploded on bravo")
            return "/abs/path/ok.py:1: export"
        if "MECHANICALLY" in prompt or "cheapest command" in lowered:
            return {"command": "python3 -c 'print(0)'"}
        if "launder a guess" in prompt or "THIS material only" in prompt:
            return "synthesis from collected verdicts only"
        if "exit_code" in prompt or "harness" in lowered:
            return {"verdict": "TRUE"}
        return three

    caller = FakeCaller(handler)
    result = run_pipeline(
        "three claims",
        caller,
        lambda cmd: (0, "ok"),
    )
    verdicts = result["verdicts"]
    assert len(verdicts) == 3, (
        f"failed obligation was dropped; got {len(verdicts)} verdicts: {verdicts!r}"
    )
    by_id = {getattr(v, "obligation_id", None): v for v in verdicts}
    assert "bravo" in by_id, f"bravo missing from verdicts: {by_id.keys()!r}"
    failed = by_id["bravo"]
    assert failed.verdict == "UNVERIFIABLE", (
        f"failed obligation marked {failed.verdict!r}, not UNVERIFIABLE"
    )
    blob = (failed.output or "") + (failed.evidence or "")
    assert "explod" in blob.lower() or "failure" in blob.lower(), (
        f"failure reason not recorded: {failed!r}"
    )


def check_7_synthesize_receives_no_run_command():
    """Signature (and source) make it impossible to pass a command-runner."""
    sig = inspect.signature(synthesize)
    assert "run_command" not in sig.parameters, (
        f"synthesize accepts run_command: {sig}"
    )
    src = MODULE_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"def synthesize\((.*?)\)\s*(?:->[^:]*)?:",
        src,
        re.S,
    )
    assert match, "def synthesize(...) not found in verifier_pipeline.py"
    assert "run_command" not in match.group(1), (
        f"synthesize source signature mentions run_command: {match.group(0)!r}"
    )


def check_8_test_role_fast_path_skips_model():
    """agent_role=test with an exact command in angles must not call the model."""

    def handler(prompt, schema):
        return "should-not-be-called"

    caller = FakeCaller(handler)
    ob = Obligation(
        id="t1",
        statement="python3 prints 7",
        angles=["python3 -c 'print(7)'"],
        consequential=False,
        agent_role="test",
    )
    n_before = caller.n
    result = execute(ob, caller)
    assert caller.n == n_before, (
        f"test-role fast path called the model {caller.n - n_before} time(s)"
    )
    assert isinstance(result, str), type(result)


CHECKS = [
    ("plan_rejects_task_list", check_1_plan_rejects_task_list),
    ("execute_never_returns_verdict", check_2_execute_never_returns_verdict),
    ("verify_actually_runs_command", check_3_verify_actually_runs_command),
    ("empty_command_is_unverifiable", check_4_empty_command_is_unverifiable),
    ("real_output_overrides_model_true", check_5_real_output_overrides_model_true),
    ("failed_obligation_is_not_dropped", check_6_failed_obligation_is_not_dropped),
    ("synthesize_receives_no_run_command", check_7_synthesize_receives_no_run_command),
    ("test_role_fast_path_skips_model", check_8_test_role_fast_path_skips_model),
]


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"ok {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    return 1 if failed else 0


def test_hcli_verifier_pipeline():
    """pytest entry: the same checks as running this file directly."""
    rc = main()
    assert rc == 0


if __name__ == "__main__":
    sys.exit(main())
