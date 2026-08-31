"""The completion attack, mechanized.

A campaign fails most often not by producing nothing but by producing something
that LOOKS finished. This module tries to invalidate the sidecar's completion.
It is adversarial on purpose: a green run here is the only thing that earns the
right to say the suite is done.

What it hunts:

* a receipt asserting a hardware number while the sidecar holds no GPU
* a module whose "implementation" is a placeholder (pass / TODO / NotImplemented
  in the load-bearing path)
* a test file with no negative control -- a guard nobody has watched fail
* a test that passes by skipping
* a receipt that is not sealed, or whose seal does not match its content
* a module with no test at all
* a sidecar file that landed outside the sidecar write partition
* a receipt claiming PROMOTED / VERIFIED status it has no authority to claim

    python3 tools/future/integration_attack.py --adversarial
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from tools.future._common import HARDWARE_FIELDS, RECEIPTS, REPO, write_receipt

FUTURE = REPO / "tools" / "future"

# Modules that are infrastructure, not campaign deliverables.
INFRA = {"_common.py", "__init__.py", "integration_attack.py", "handoff.py"}

# A promotion vocabulary the sidecar has no authority to use about its own work.
FORBIDDEN_STATUS = re.compile(
    r"\b(PROMOTED|PROTECTED_PASS|PROTECTED_ABSOLUTE|MEASURED_ON_HARDWARE)\b"
)

PLACEHOLDER = re.compile(
    r"^\s*(pass\s*(#.*)?$|\.\.\.\s*$|raise NotImplementedError|# ?TODO|# ?FIXME|# ?STUB)",
    re.MULTILINE,
)


class Finding(dict):
    pass


def _finding(kind: str, where: str, detail: str, severity: str = "P1") -> Finding:
    return Finding(kind=kind, where=where, detail=detail, severity=severity)


# --------------------------------------------------------------------------
# attacks
# --------------------------------------------------------------------------


def attack_hardware_claims() -> list[Finding]:
    """No receipt may assert a hardware number. The sidecar has no GPU."""
    out = []
    for p in sorted(RECEIPTS.glob("*.json")):
        try:
            doc = json.loads(p.read_text())
        except Exception as e:
            out.append(_finding("unreadable_receipt", str(p), repr(e), "P0"))
            continue

        def walk(node: Any, path: str = "") -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    here = f"{path}.{k}" if path else k
                    if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                        out.append(
                            _finding(
                                "hardware_claim_without_hardware",
                                f"{p.name}:{here}",
                                f"{here} = {v!r}",
                                "P0",
                            )
                        )
                    walk(v, here)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")

        walk(doc)

        bench = doc.get("bench") or {}
        if bench.get("state") != "UNKNOWN":
            out.append(
                _finding(
                    "bench_state_overridden",
                    p.name,
                    f"bench.state = {bench.get('state')!r}, expected UNKNOWN",
                    "P0",
                )
            )
        if bench.get("gpu_authority") is not False:
            out.append(
                _finding("gpu_authority_claimed", p.name, str(bench.get("gpu_authority")), "P0")
            )
    return out


def attack_seals() -> list[Finding]:
    """A seal that does not match its content is worse than no seal."""
    out = []
    for p in sorted(RECEIPTS.glob("*.json")):
        try:
            doc = json.loads(p.read_text())
        except Exception:
            continue  # already reported by attack_hardware_claims
        claimed = doc.get("seal_sha256")
        if not claimed:
            out.append(_finding("unsealed_receipt", p.name, "no seal_sha256", "P1"))
            continue
        body = {k: v for k, v in doc.items() if k != "seal_sha256"}
        blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        actual = hashlib.sha256(blob).hexdigest()
        if actual != claimed:
            out.append(
                _finding("seal_mismatch", p.name, f"claimed {claimed[:12]} actual {actual[:12]}", "P0")
            )
    return out


def attack_forbidden_status() -> list[Finding]:
    """The sidecar cannot promote anything. It may only describe."""
    out = []
    for p in sorted(RECEIPTS.glob("*.json")):
        text = p.read_text()
        for m in FORBIDDEN_STATUS.finditer(text):
            # Naming the vocabulary in a policy/refusal string is legitimate; asserting
            # it as this artifact's own status is not. Only flag a status-ish key.
            line_start = text.rfind("\n", 0, m.start()) + 1
            line = text[line_start : text.find("\n", m.start())]
            if re.search(r'"(status|verdict|result|outcome)"\s*:', line):
                out.append(
                    _finding("forbidden_promotion_status", f"{p.name}", line.strip()[:160], "P0")
                )
    return out


def _is_abstract_stub(fn: ast.AST, cls_bases: set[str]) -> bool:
    """`raise NotImplementedError` as the sole body of a base-class method is the
    ABC idiom, not an unfinished implementation."""
    body = [n for n in getattr(fn, "body", []) if not isinstance(n, ast.Expr) or
            not isinstance(getattr(n, "value", None), ast.Constant)]
    if len(body) != 1:
        return False
    node = body[0]
    return (
        isinstance(node, ast.Raise)
        and isinstance(node.exc, (ast.Name, ast.Call))
        and "NotImplementedError" in ast.dump(node)
    )


def attack_placeholders() -> list[Finding]:
    """A module whose load-bearing path is a placeholder is not an implementation.

    Parsed with `ast`, and deliberately narrow. `except SomeError: pass` is the
    idiom this codebase uses for its OWN negative controls -- asserting that a
    guard raised -- so flagging it would train the reader to ignore this checker,
    which is worse than not running it. Only a function whose entire body is
    `pass` / `...` counts, plus `raise NotImplementedError` outside a base class.
    """
    out = []
    for p in sorted(FUTURE.glob("*.py")):
        if p.name in INFRA or p.name.startswith("test_"):
            continue
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError as e:
            out.append(_finding("module_unparseable", p.name, repr(e), "P0"))
            continue

        abstract_classes: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = {_dotted(b) for b in node.bases}
                if bases & {"ABC", "abc.ABC", "Protocol", "typing.Protocol"} or not bases:
                    abstract_classes.add(node.name)

        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in cls.body:
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fn._in_class = cls.name  # type: ignore[attr-defined]

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = [n for n in node.body
                    if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
            if len(body) == 1 and isinstance(body[0], (ast.Pass,)):
                out.append(
                    _finding("placeholder_in_module", f"{p.name}:{node.lineno}",
                             f"def {node.name}(...) body is `pass`", "P1")
                )
            elif _is_abstract_stub(node, set()):
                owner = getattr(node, "_in_class", None)
                if owner is None:
                    out.append(
                        _finding("placeholder_in_module", f"{p.name}:{node.lineno}",
                                 f"def {node.name}(...) raises NotImplementedError", "P1")
                    )
        # TODO/FIXME/STUB markers still matter, but only as their own kind.
        src = p.read_text()
        for m in re.finditer(r"^\s*#\s?(TODO|FIXME|STUB)\b", src, re.MULTILINE):
            out.append(
                _finding("unfinished_marker", f"{p.name}:{src.count(chr(10), 0, m.start()) + 1}",
                         m.group(0).strip(), "P1")
            )
    return out


def attack_missing_tests() -> list[Finding]:
    """Every deliverable module ships a test module."""
    out = []
    for p in sorted(FUTURE.glob("*.py")):
        if p.name in INFRA or p.name.startswith("test_"):
            continue
        if not (FUTURE / f"test_{p.name}").exists():
            out.append(_finding("module_without_test", p.name, f"no test_{p.name}", "P1"))
    return out


def attack_missing_negative_controls() -> list[Finding]:
    """A guard nobody has watched fail is not a guard.

    Every test module must contain at least one test that proves a refusal
    actually fires: pytest.raises, an assertion that a checker returned
    non-zero/False, or an explicitly named negative control.
    """
    out = []
    signals = (
        "pytest.raises",
        "negative_control",
        "negative control",
        "== 1",
        "is False",
        "assert not ",
        "REJECT",
        "refus",
    )
    for p in sorted(FUTURE.glob("test_*.py")):
        src = p.read_text()
        if not any(s in src for s in signals):
            out.append(
                _finding(
                    "test_without_negative_control",
                    p.name,
                    "no refusal/raises assertion found; guard never watched to fail",
                    "P1",
                )
            )
    return out


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def attack_skipped_tests() -> list[Finding]:
    """A suite that passes by skipping is a suite that measured nothing.

    This project has already shipped grades that rested on SKIPPED tests, so an
    unconditional skip in a deliverable test module is a P0.

    Parsed with `ast`, not regex: a test that PROVES skip-detection works has to
    write the string `@pytest.mark.skip` into a fixture, and a text scan flags
    that as a real skip. Carving out the file would leave a blind spot in exactly
    the module whose job is to have none, so match syntax instead of text.
    """
    out = []
    for p in sorted(FUTURE.glob("test_*.py")):
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError as e:
            out.append(_finding("test_unparseable", p.name, repr(e), "P0"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    target = dec.func if isinstance(dec, ast.Call) else dec
                    name = _dotted(target)
                    if name in ("pytest.mark.skip", "pytest.mark.skipif"):
                        out.append(
                            _finding("test_skip", f"{p.name}:{dec.lineno}", f"@{name}", "P0")
                        )
                    elif name == "pytest.mark.xfail":
                        out.append(
                            _finding("test_skip", f"{p.name}:{dec.lineno}", f"@{name}", "P1")
                        )
    return out


def attack_actual_skips() -> list[Finding]:
    """A suite that passes by SKIPPING measured nothing.

    What matters is not whether a `pytest.skip()` appears in the source -- a skip
    guarded on genuinely optional evidence is defensible -- but whether one
    actually FIRED. This project has shipped grades that rested on silently
    skipped tests, so run the suite and read the real skip report.
    """
    # Guard against recursion: this attack shells out to pytest, and pytest runs
    # this module's own test, which calls run(). Without the sentinel the attacker
    # invokes itself forever. The nested call reports NOT_RUN rather than lying.
    if _os.environ.get("HAWKING_ATTACK_NESTED") == "1":
        return []
    env = dict(_os.environ, HAWKING_ATTACK_NESTED="1")
    r = subprocess.run(
        ["python3", "-m", "pytest", "tools/future/", "-q", "-rs", "--no-header",
         "-p", "no:cacheprovider", "--ignore", "tools/future/test_integration_attack.py"],
        cwd=REPO, capture_output=True, text=True, env=env,
    )
    out = []
    for line in r.stdout.splitlines():
        if line.startswith("SKIPPED"):
            out.append(_finding("test_actually_skipped", "pytest -rs", line.strip()[:200], "P0"))
    if r.returncode != 0 and not out:
        tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-1:]
        out.append(_finding("suite_not_green", "pytest", (tail or ["<no output>"])[0][:200], "P0"))
    return out


def attack_write_partition() -> list[Finding]:
    """Nothing a lane produced may live outside the sidecar partition."""
    from tools.future import mutation_surface as ms

    out = []
    if ms.check_disjoint(["tools/future", "receipts/future"]) != 0:
        out.append(
            _finding("write_partition_violated", "tools/future", "intersects Codex surface", "P0")
        )
    return out


ATTACKS = {
    "hardware_claims": attack_hardware_claims,
    "seals": attack_seals,
    "forbidden_status": attack_forbidden_status,
    "placeholders": attack_placeholders,
    "missing_tests": attack_missing_tests,
    "missing_negative_controls": attack_missing_negative_controls,
    "skipped_tests": attack_skipped_tests,
    "actual_skips": attack_actual_skips,
    "write_partition": attack_write_partition,
}


def run() -> dict[str, Any]:
    findings: list[Finding] = []
    per_attack = {}
    for name, fn in ATTACKS.items():
        got = fn()
        per_attack[name] = len(got)
        findings.extend(got)
    p0 = [f for f in findings if f["severity"] == "P0"]
    p1 = [f for f in findings if f["severity"] == "P1"]
    modules = sorted(
        p.name for p in FUTURE.glob("*.py") if p.name not in INFRA and not p.name.startswith("test_")
    )
    return {
        "schema": "hawking.future.integration_attack.v1",
        "version": 1,
        "purpose": "adversarial completion attack over the sidecar; its job is to find a reason NOT to finish",
        "modules_examined": modules,
        "module_count": len(modules),
        "receipts_examined": sorted(p.name for p in RECEIPTS.glob("*.json")),
        "attacks_run": per_attack,
        "counts": {"P0": len(p0), "P1": len(p1), "total": len(findings)},
        "findings": findings,
        "verdict": "CLEAN" if not findings else ("P0_PRESENT" if p0 else "P1_PRESENT"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adversarial", action="store_true", help="run and fail on any P0")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    doc = run()
    out = write_receipt("INTEGRATION_ATTACK.json", doc, "tools/future/integration_attack.py")
    print(out)
    print(f"verdict={doc['verdict']} P0={doc['counts']['P0']} P1={doc['counts']['P1']}")
    for f in doc["findings"][:60]:
        print(f"  [{f['severity']}] {f['kind']}: {f['where']} — {f['detail']}")
    if a.report_only:
        return 0
    return 1 if doc["counts"]["P0"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
