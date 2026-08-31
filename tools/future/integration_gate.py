#!/usr/bin/env python3
"""The canonical writer refuses to land while the required gate is red.

This exists because "don't commit red" as a remembered rule failed. A path-to-71
receipt was committed with a failing assertion in its own suite. The behaviour
under test was correct and the assertion compared 1.0027 against 1.0027008, but a
behaviourally-correct-red suite is still a red integration state, and the commit
should not have happened.

So it stops being a rule and becomes a door.

    python3 tools/future/integration_gate.py --check tools/future/foo.py ...
    python3 tools/future/integration_gate.py --land -F msg.txt -- <paths>

--land runs the required checks for the paths being landed and refuses to invoke
git if any fail. A deliberate exception must be named KNOWN_RED_CHECKPOINT with a
justification, which is recorded in the receipt and in the commit body — it is
possible, and it is not quiet.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402
from tools.future import status_causality as sc  # noqa: E402

RECEIPT = REPO / "receipts" / "future" / "INTEGRATION_GATE.json"
TEST_TIMEOUT_S = 1200

FIVE_RECORDED_FIELDS: tuple[str, ...] = getattr(
    sc,
    "FIVE_RECORDED_FIELDS",
    (
        "probe_performed",
        "direct_observation",
        "interpretation",
        "confidence",
        "alternatives",
    ),
)


def _bind_emit() -> None:
    """Consumer-side emit. Sibling owns the routine; this checkout may predate it."""
    if hasattr(sc, "emit"):
        return

    def emit(
        status: str,
        *,
        probe_performed: str = "",
        direct_observation: Any = "",
        interpretation: str = "",
        probe_kind: str = "",
        claim_kind: str | None = None,
        falsifier: str = "",
        source: str = "",
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "status": status,
            "probe_performed": probe_performed,
            "direct_observation": direct_observation,
            "interpretation": interpretation or status,
            "probe_kind": probe_kind,
            "use_catalog": False,
            "source": source or "<emit>",
        }
        if claim_kind:
            row["claim_kind"] = claim_kind
        if falsifier:
            row["falsifier"] = falsifier
        out = sc.challenge(row)
        out["entry"] = "emit"
        return out

    sc.emit = emit  # type: ignore[attr-defined]


_bind_emit()


def records_five_fields(node: Any) -> bool:
    fn = getattr(sc, "records_five_fields", None)
    if callable(fn):
        return bool(fn(node))
    if not isinstance(node, dict):
        return False
    if not all(k in node for k in FIVE_RECORDED_FIELDS):
        return False
    if not str(node.get("probe_performed") or "").strip():
        return False
    if node.get("direct_observation") in (None, "", [], {}):
        return False
    if not str(node.get("interpretation") or "").strip():
        return False
    conf = node.get("confidence")
    if not isinstance(conf, dict):
        return False
    if not {"would_raise", "would_lower", "level", "about"} <= set(conf):
        return False
    alts = node.get("alternatives")
    return isinstance(alts, list) and bool(alts)


def record_check_causality(
    result: dict[str, Any],
    *,
    probe_performed: str = "",
    direct_observation: Any = "",
    interpretation: str | None = None,
    probe_kind: str = "",
    claim_kind: str | None = None,
) -> dict[str, Any]:
    """Stamp the five causality fields. Does not change green/verdict.

    An unsupplied observation is UNTESTED, never a restatement of GREEN/RED.
    """
    green_before = result.get("green")
    verdict_before = result.get("verdict")
    status = str(result.get("verdict") or ("GREEN" if result.get("green") else "RED"))
    unsupplied = direct_observation in (None, "", [], {})
    rec = sc.emit(
        status,
        probe_performed=str(probe_performed or ""),
        direct_observation="" if unsupplied else direct_observation,
        interpretation=interpretation if interpretation is not None else status,
        probe_kind="" if unsupplied else (probe_kind or sc.PROBE_MEASURED_FLAGS),
        claim_kind=None if unsupplied else claim_kind,
        source="tools/future/integration_gate.py::check",
    )
    for key in FIVE_RECORDED_FIELDS:
        result[key] = rec[key]
    result["causality_verdict"] = rec["verdict"]
    result["falsifier"] = rec.get("falsifier")
    if rec.get("probe_kind"):
        result["probe_kind"] = rec["probe_kind"]
    if rec.get("claim_kind") is not None:
        result["claim_kind"] = rec["claim_kind"]
    if result.get("green") != green_before or result.get("verdict") != verdict_before:
        raise RuntimeError("status_causality.emit mutated the gate verdict")
    return rec


class GateRed(Exception):
    """Raised instead of committing. The whole point of the module."""


def required_tests(paths: list[str]) -> list[str]:
    """For each module being landed, its own test file if one exists.

    Deliberately narrow. A full-suite gate would be ignored for being slow, and
    an ignored gate is worse than a narrow one.
    """
    want: list[str] = []
    for p in paths:
        path = Path(p)
        if path.suffix != ".py" or "tools/future" not in str(path):
            continue
        name = path.name
        cand = path.parent / (name if name.startswith("test_") else f"test_{name}")
        if (REPO / cand).is_file() and str(cand) not in want:
            want.append(str(cand))
    return want


def run_tests(tests: list[str]) -> dict[str, Any]:
    if not tests:
        return {"ran": False, "why": "no test module corresponds to these paths",
                "green": True, "tests": []}
    cmd = [sys.executable, "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider"]
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           timeout=TEST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"ran": True, "green": False, "tests": tests,
                "why": f"pytest exceeded {TEST_TIMEOUT_S}s; a gate that hangs is red"}
    tail = (p.stdout or "").strip().splitlines()[-3:]
    return {"ran": True, "green": p.returncode == 0, "returncode": p.returncode,
            "tests": tests, "tail": tail,
            "why": "pytest exit 0" if p.returncode == 0 else "pytest failed"}


def receipts_parse(paths: list[str]) -> dict[str, Any]:
    bad = []
    for p in paths:
        if not p.endswith(".json"):
            continue
        f = REPO / p
        if not f.is_file():
            continue
        try:
            json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            bad.append({"path": p, "error": f"{type(exc).__name__}: {exc}"})
    return {"green": not bad, "malformed": bad}


def check(paths: list[str]) -> dict[str, Any]:
    tests = run_tests(required_tests(paths))
    rec = receipts_parse(paths)
    green = tests["green"] and rec["green"]
    result = {
        "paths": paths,
        "tests": tests,
        "receipts": rec,
        "green": green,
        "verdict": "GREEN" if green else "RED",
    }
    test_list = list(tests.get("tests") or [])
    if tests.get("ran"):
        probe_performed = (
            f"subprocess.run {[sys.executable, '-m', 'pytest', *test_list, '-q', '-p', 'no:cacheprovider']} "
            f"cwd=REPO timeout={TEST_TIMEOUT_S}s; json.loads each .json path in {paths}"
        )
        direct_observation = (
            f"pytest.ran=True pytest.green={tests.get('green')} "
            f"returncode={tests.get('returncode')} why={tests.get('why')!r} "
            f"tail={tests.get('tail')!r} receipts.green={rec.get('green')} "
            f"malformed={rec.get('malformed')}"
        )
    else:
        probe_performed = (
            f"required_tests({paths}) looked for tools/future test_*.py; "
            f"no corresponding test module. json.loads each .json path in {paths}"
        )
        direct_observation = (
            f"pytest.ran=False why={tests.get('why')!r} "
            f"receipts.green={rec.get('green')} malformed={rec.get('malformed')}"
        )
    record_check_causality(
        result,
        probe_performed=probe_performed,
        direct_observation=direct_observation,
        interpretation=(
            f"{result['verdict']}: tests.why={tests.get('why')!r} "
            f"receipts.green={rec.get('green')}"
        ),
        probe_kind=sc.PROBE_MEASURED_FLAGS,
        claim_kind=sc.CLAIM_FIELD_VALUE if green else sc.CLAIM_MEASURED_UNMET,
    )
    return result


def land(paths: list[str], message_file: str, known_red: str | None = None) -> dict[str, Any]:
    result = check(paths)
    if not result["green"] and not known_red:
        raise GateRed(
            "REFUSED: required gate is RED. "
            f"tests={result['tests'].get('why')} "
            f"receipts={'malformed: ' + str(result['receipts']['malformed']) if result['receipts']['malformed'] else 'ok'}. "
            "Fix it, or pass --known-red with a justification, which is recorded."
        )
    body = Path(message_file).read_text()
    if known_red:
        body += (
            "\n\nKNOWN_RED_CHECKPOINT: " + known_red +
            "\nThe required gate was RED when this landed and it was landed anyway, "
            "deliberately. See receipts/future/INTEGRATION_GATE.json.\n"
        )
        tmp = REPO / ".git" / "GATE_MSG"
        tmp.write_text(body)
        message_file = str(tmp)
    subprocess.run(["git", "add", "--", *paths], cwd=REPO, check=True)
    p = subprocess.run(["git", "commit", "-q", "-F", message_file, "--", *paths],
                       cwd=REPO, capture_output=True, text=True)
    result["committed"] = p.returncode == 0
    result["known_red"] = known_red
    result["git_stderr"] = (p.stderr or "").strip()[:400] or None
    return result


def build(last: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": "hawking.future.integration_gate.v1",
        "version": 1,
        "recorded_by": "tools/future/integration_gate.py",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "law": (
            "NO CANONICAL SOURCE COMMIT WHILE THE REQUIRED ACCEPTANCE SUITE IS "
            "RED. A behaviourally-correct-but-red test is still a red "
            "integration state."
        ),
        "why_it_is_a_door_not_a_rule": (
            "The rule already existed and was broken: commit db4dacede landed the "
            "path-to-71 receipt while its own suite had a failing assertion. The "
            "failure was a rounding comparison and the behaviour was correct, "
            "which is exactly the case a remembered rule loses. Enforcement has "
            "to be mechanical."
        ),
        "what_is_checked": [
            "for every tools/future module being landed, its own test_*.py if one "
            "exists — narrow on purpose, because a slow full-suite gate gets "
            "bypassed and a bypassed gate is worse than a narrow one",
            "every .json path being landed must parse",
        ],
        "escape_hatch": {
            "flag": "--known-red <justification>",
            "behaviour": "lands anyway, appends KNOWN_RED_CHECKPOINT and the "
                         "justification to the commit body, and records it here",
            "not_quiet": True,
        },
        "last_check": last,
        "claim_boundary": (
            "This gate checks the tests that correspond to the paths being "
            "landed and that receipts parse. It does not run the full suite, does "
            "not typecheck, and does not verify provenance. A green gate means "
            "those two things passed, not that the change is correct."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--land", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("-F", dest="message_file")
    ap.add_argument("--known-red", dest="known_red")
    ap.add_argument("paths", nargs="*")
    a = ap.parse_args()
    if a.record and not (a.check or a.land):
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n")
        print(f"wrote {RECEIPT}")
        return 0
    paths = [p for p in a.paths if p != "--"]
    if a.land:
        try:
            r = land(paths, a.message_file, a.known_red)
        except GateRed as exc:
            print(str(exc))
            RECEIPT.parent.mkdir(parents=True, exist_ok=True)
            RECEIPT.write_text(json.dumps(build(check(paths)), indent=1, sort_keys=True) + "\n")
            return 1
    else:
        r = check(paths)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(build(r), indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "paths"}, indent=1)[:900])
    return 0 if r["green"] or r.get("known_red") else 1


if __name__ == "__main__":
    raise SystemExit(main())
