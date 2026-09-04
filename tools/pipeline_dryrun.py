#!/usr/bin/env python3
"""Exercise the stages AFTER a valid patch, without spending a resident cycle.

Across 204 receipts, validation has only ever been `none` or `read_only`: apply,
run tests, verify, accept have NEVER executed. Every defect found so far came
from the stages upstream of them, one per 22-minute resident attempt. At that
discovery rate the four unexercised stages are hours away, and they are hours
that teach nothing about the model.

They do not need the model. `_apply_operations` and `_validate` take a patch and
a test list as plain arguments, so a known-good patch drives the whole downstream
pipeline in seconds. This is not a substitute for Gate 1 -- Gate 1 requires HCLI
to AUTHOR the patch -- it clears the road so that when HCLI finally emits a valid
one, the failure it meets is its own and not a harness defect nobody had reached.

Everything happens in a throwaway file. The daemon is live in this tree and a
mutation to a real source file underneath it is not worth the risk.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import traceback

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PROBE = REPO / "hcli" / "tests" / "_dryrun_probe.py"
PROBE_TEST = REPO / "hcli" / "tests" / "test__dryrun_probe.py"

BEFORE = '''"""Throwaway target for tools/pipeline_dryrun.py. Safe to delete."""


def total_lines(text: str) -> int:
    return 0
'''

AFTER = '''"""Throwaway target for tools/pipeline_dryrun.py. Safe to delete."""


def total_lines(text: str) -> int:
    return len(text.splitlines())
'''

TEST = '''from hcli.tests._dryrun_probe import total_lines


def test_counts_the_lines():
    assert total_lines("a\\nb\\nc\\n") == 3
'''


def _engine():
    from hcli.engine import Engine
    from hcli.workspace import Workspace

    class _Pool:
        model_path = "sealed-3.14"
        topology = "process"
        requested_n = 1
        admitted_n = 1
        repo_root = str(REPO)

    return Engine(Workspace(str(REPO)), runtime_provider=lambda: _Pool())


def _cleanup():
    for p in (PROBE, PROBE_TEST):
        if p.exists():
            p.unlink()
    for cache in (REPO / "hcli" / "tests" / "__pycache__",):
        for junk in list(cache.glob("*_dryrun_probe*")) + list(cache.glob("*_dryrun_vacuous*")):
            junk.unlink(missing_ok=True)


def stage(name, fn):
    try:
        ok, detail = fn()
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for line in str(detail).splitlines():
        print(f"        {line}")
    return ok


def main() -> int:
    _cleanup()
    PROBE.write_text(BEFORE)
    PROBE_TEST.write_text(TEST)
    eng = _engine()
    results = []

    def apply_stage():
        ops = [{
            "op": "replace",
            "path": str(PROBE.relative_to(REPO)),
            "old_text": "    return 0\n",
            "new_text": "    return len(text.splitlines())\n",
        }]
        out = eng._apply_operations(ops)
        body = PROBE.read_text()
        if "splitlines" not in body:
            return False, f"the operation reported {out} but the file is unchanged"
        return True, f"applied; file now ends: {body.strip().splitlines()[-1]!r}"

    def validate_stage():
        out = eng._validate(
            [PROBE], tests=[str(PROBE_TEST.relative_to(REPO))]
        )
        checks = out.get("checks") or []
        kinds = [c.get("kind") for c in checks]
        test_checks = [c for c in checks if c.get("kind") == "test"]
        if not test_checks:
            return False, f"validation ran no test at all: kinds={kinds}"
        t = test_checks[0]
        if not t.get("executed"):
            return False, f"a test was admitted but never executed: {t}"
        if not out.get("ok"):
            return False, f"a correct patch FAILED validation: {out}"
        return True, (f"kinds={kinds} passed={t.get('passed')}/{t.get('collected')} "
                      f"exit={t.get('exit_code')}")

    def red_before_green():
        """A test that cannot fail is not evidence.

        The validator carries red_before_green as ADVISORY and, on the run
        above, did not compute it at all: reason 'pre_mutation_pass_not_run'.
        So a mutation whose test passes BEFORE the change reads exactly like a
        mutation whose test the change made pass.
        """
        PROBE.write_text(AFTER)  # already correct: the test is green pre-mutation
        vacuous = REPO / "hcli" / "tests" / "test__dryrun_vacuous.py"
        vacuous.write_text("def test_always():\n    assert True\n")
        try:
            # Mirror the LIVE path: it runs a pre-mutation validation first
            # (engine.py:1987) and feeds it to the post one, so testing without
            # that would be testing a path production never takes.
            pre = eng._validate([PROBE], tests=[str(vacuous.relative_to(REPO))])
            out = eng._validate(
                [PROBE], tests=[str(vacuous.relative_to(REPO))], pre_mutation=pre
            )
            rbg = out.get("red_before_green")
            if out.get("ok") and rbg is not True:
                return False, (
                    f"a test that cannot fail was accepted: ok={out.get('ok')} "
                    f"red_before_green={rbg!r} "
                    f"reason={out.get('red_before_green_reason')!r} "
                    f"advisory={out.get('red_before_green_advisory')!r}"
                )
            return True, f"red_before_green={rbg!r}"
        finally:
            vacuous.unlink(missing_ok=True)

    def catches_a_bad_patch():
        """A validator that passes everything is not a validator."""
        PROBE.write_text(BEFORE.replace("return 0", "return 999"))
        out = eng._validate([PROBE], tests=[str(PROBE_TEST.relative_to(REPO))])
        if out.get("ok"):
            return False, "validation PASSED a patch that breaks its own test"
        return True, "a wrong patch is rejected, so the pass above means something"

    results.append(stage("apply the operation", apply_stage))
    results.append(stage("validate: run the real test", validate_stage))
    results.append(stage("validate rejects a wrong patch", catches_a_bad_patch))
    results.append(stage("a test that cannot fail is refused", red_before_green))

    _cleanup()
    print()
    print(f"{sum(results)}/{len(results)} stages clear")
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        _cleanup()
