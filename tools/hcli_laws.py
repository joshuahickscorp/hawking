#!/usr/bin/env python3
"""HCLI laws, checked rather than remembered.

LAW 1 -- REMEDIATION MUST BE ACTIONABLE FROM OBSERVABLE STATE.

  Every remediation instruction must be actionable from the recipient's
  observable state. Never reference hidden post-transform coordinates, omitted
  output, unavailable filesystem state, or a capability the recipient cannot
  invoke.

Three defects in one day were self-inflicted violations of exactly this, and
each was invisible until a receipt was read:

  * "closing parenthesis does not match ... at line 592 of the RESULTING file"
    -- a file the model never sees. Three attempts, three identical errors.
  * "use fs.search to find the line a symbol is on" -- and fs.search then
    raised NotADirectoryError on the file it had just been pointed at.
  * an 800-character rejected-reply excerpt that spent its whole budget on
    prose and elided the operation the rejection was about.

Remembering three bugs is worse than enforcing one law, so this checks the law.
A remediation message may name a capability only if that capability actually
supports the use the message describes, and it must quote the state it refers
to rather than a coordinate into something the recipient cannot open.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


class _Ctx:
    def resolve_read_path(self, p):
        return pathlib.Path(p).resolve()


def check_promised_workflow_actually_works() -> tuple[bool, str]:
    """The truncation notice promises search-then-windowed-read. Do it."""
    from hcli import tool_registry as tr

    target = REPO / "hcli" / "tool_registry.py"
    found = tr._search_files(_Ctx(), {"path": str(target), "pattern": "def _read_file"})
    matches = found.get("matches") or []
    if not matches:
        return False, "fs.search on a file found nothing; the notice promises it will"
    line = matches[0].get("line")
    if not line:
        return False, "fs.search returned a match with no line number to read around"
    window = tr._read_file(
        _Ctx(), {"path": str(target), "start_line": line, "end_line": line + 2}
    )
    if "def _read_file" not in (window.get("content") or ""):
        return False, "fs.read with start_line did not return the line fs.search named"
    return True, f"search found line {line}, windowed read returned it"


def check_syntax_error_quotes_the_line() -> tuple[bool, str]:
    from hcli.engine import _python_syntax_violation

    src = (REPO / "hcli" / "tool_registry.py").read_text()
    anchor = "    clipped = raw[:limit]\n"
    if src.count(anchor) != 1:
        return False, "fixture anchor is no longer unique; this check is vacuous"
    payload = json.dumps({"kind": "mutation", "operations": [{
        "op": "replace", "path": "hcli/tool_registry.py",
        "old_text": anchor, "new_text": anchor + "    x = len(f(\n",
    }]})
    message = _python_syntax_violation(payload)
    if not message:
        return False, "a broken fragment produced no remediation at all"
    if "reads there" not in message:
        return False, "syntax remediation names a line without quoting it"
    return True, "syntax remediation quotes the resulting file"


def check_missing_file_says_what_to_do() -> tuple[bool, str]:
    from hcli import tool_registry as tr

    try:
        tr._read_file(_Ctx(), {"path": str(REPO / "hcli" / "tests" / "nope_xyz.py")})
    except FileNotFoundError as exc:
        text = str(exc)
        if "fs.list" not in text and "Did you mean" not in text:
            return False, "a missing file names no way to find the right one"
        if "CREATE" not in text:
            return False, "a model reading a file it should write is not told so"
        return True, "missing-file remediation names a next action"
    return False, "reading a missing file did not raise"


def check_truncation_names_only_real_capabilities() -> tuple[bool, str]:
    """Every tool a remediation names must exist in the registry."""
    from hcli.engine import Engine
    from hcli import tool_registry as tr

    notice = Engine.__new__(Engine)._clamp_observation("y" * 200_000)
    named = {word.strip(".,]") for word in notice.split() if word.startswith("fs.")}
    if not named:
        return False, "the truncation notice names no remedy at all"
    # Build the REAL registry. Falling through to a pass when the tool list
    # could not be found made this check unable to fail, which is the same
    # class of defect it exists to catch.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        registry = tr.default_tool_registry(tmp, repo_root=str(REPO))
    known = set(getattr(registry, "_tools", {}) or {})
    if not known:
        try:
            known = {getattr(t, "name", t) for t in registry.discover()}
        except Exception:
            known = set()
    if not known:
        return False, "could not enumerate the registry, so this check cannot fail"
    missing = {n for n in named if n not in known}
    if missing:
        return False, f"remediation names tools that do not exist: {sorted(missing)}"
    return True, f"named {sorted(named)}, all registered"


CHECKS = [
    ("promised workflow works end to end", check_promised_workflow_actually_works),
    ("syntax error quotes the line it names", check_syntax_error_quotes_the_line),
    ("missing file says what to do", check_missing_file_says_what_to_do),
    ("remediation names only real capabilities", check_truncation_names_only_real_capabilities),
]


def main() -> int:
    print("LAW 1 -- remediation must be actionable from observable state\n")
    failed = 0
    for name, check in CHECKS:
        try:
            ok, detail = check()
        except Exception as exc:  # a check that cannot run is a failing check
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print(f"        {detail}")
        failed += not ok
    print()
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} upheld")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
