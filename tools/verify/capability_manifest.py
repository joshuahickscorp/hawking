#!/usr/bin/env python3.12
"""Prove that a retired entrypoint is still invocable through whatever replaced it.

`tools/loc/hawking_inventory.py` counts capability as entrypoints: `__main__` blocks,
argparse surfaces, binaries, scripts. Campaign section 7.2 asks for the opposite shape --
replace repeated controllers with typed specs run by one engine. So a spec-driven rewrite
fails the capability gate by construction: 77 modules with `__main__` become 77 rows in a
spec table, and the counter sees 76 capabilities vanish.

The rule (workspace/campaign/governance/control/catalog/manifests/REBUILD_ACCOUNTING_RULES.json,
`capability_equivalence_for_spec_driven_designs`)
is that a spec may replace an entrypoint only if the replacement is **invocable and proven**.
This runs that proof.

    capability_manifest.py --check
    capability_manifest.py --gate \
      --before workspace/campaign/governance/control/rungs/before/pre-s2 \
      --after workspace/campaign/governance/control/rungs/after/post-s2b
    capability_manifest.py --scaffold \
      --before workspace/campaign/governance/control/rungs/before/pre-s2 \
      --after workspace/campaign/governance/control/rungs/after/post-s2b

`--scaffold` writes a manifest skeleton listing every entrypoint that disappeared between
two inventory snapshots, so the lane that retired them has to fill in how each is reached
now. An entry left unfilled is a lost capability, which is exactly the reading we want.

Disposition semantics:
  - replaced / invocable: `invocation` is required and is executed (dry-run / --help style).
  - retired / released: product capability deliberately released; `invocation` must be
    null/absent; nonempty exact `evidence` is required (product decision + rollback).
    Never execute a fake command such as `true` or `lab --help` to paper over a release.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.layout import CONTROL_ROOT, resolve_workspace_path

MANIFEST = CONTROL_ROOT / "catalog/manifests/CAPABILITY_MANIFEST.json"
TIMEOUT = 120

RELEASE_DISPOSITIONS = frozenset({"retired", "released"})


def resolve_snapshot(snapshot: str) -> Path:
    """Resolve a live or historical inventory name to its physical caps file."""
    p = Path(snapshot)
    if not p.suffix:
        p = p.with_suffix(".caps.json")
    if not p.is_absolute() and p.parts and p.parts[0] in {"control", "workspace"}:
        return resolve_workspace_path(p)
    return p


def load_caps(snapshot: str) -> set[str]:
    p = resolve_snapshot(snapshot)
    d = json.loads(p.read_text(encoding="utf-8"))
    return set(d.get("python_entrypoint_list", [])) | {
        b if isinstance(b, str) else b.get("name", str(b)) for b in d.get("rust_binaries", [])
    }


def evidence_is_meaningful(evidence) -> bool:
    """True iff evidence is a nonempty string or nonempty list/object with meaningful values.

    Rejects empty string, empty list, empty object, and arbitrary values whose
    ``str(...)`` happens to be ``"{}"`` / ``"[]"`` (those used to pass the old
    truthiness check). Only str / list / tuple / dict shapes are accepted.
    """
    if evidence is None:
        return False
    if isinstance(evidence, str):
        return bool(evidence.strip())
    if isinstance(evidence, (list, tuple)):
        return bool(evidence) and any(evidence_is_meaningful(x) for x in evidence)
    if isinstance(evidence, dict):
        return bool(evidence) and any(evidence_is_meaningful(v) for v in evidence.values())
    return False


def run_entry(e: dict) -> dict:
    cmd = e.get("invocation")
    if not cmd:
        return {**e, "status": "fail", "detail": "no invocation recorded"}
    try:
        r = subprocess.run(
            cmd, cwd=ROOT, shell=True, capture_output=True, text=True, timeout=TIMEOUT
        )
    except subprocess.TimeoutExpired:
        return {**e, "status": "fail", "detail": f"timed out after {TIMEOUT}s"}
    ok = r.returncode == 0
    tail = ((r.stdout or "") + (r.stderr or "")).strip()[-200:]
    return {**e, "status": "pass" if ok else "fail",
            "detail": f"exit {r.returncode}" + ("" if ok else f": {tail}")}


def classify_entry(e: dict) -> dict:
    """Classify one manifest entry without waiving unaccounted losses.

    released/retired: evidence-backed product release — not executed.
    replaced/other: must have a real invocable command.
    """
    disposition = e.get("disposition")
    evidence = e.get("evidence")
    if disposition in RELEASE_DISPOSITIONS:
        if not evidence_is_meaningful(evidence):
            return {
                **e,
                "status": "fail",
                "detail": f"{disposition} disposition requires nonempty exact evidence "
                          "(nonempty string or nonempty list/object with meaningful values)",
            }
        if e.get("invocation"):
            return {
                **e,
                "status": "fail",
                "detail": f"{disposition} entry must not claim an invocation "
                          "(no true/lab --help facade)",
            }
        return {
            **e,
            "status": "released",
            "detail": "evidence-backed product release (not invocable)",
        }
    return run_entry(e)


def check() -> dict:
    if not MANIFEST.exists():
        return {"schema": "hawking.capability_manifest_report.v1",
                "entries": [], "passed": 0, "released": 0, "failed": 0,
                "note": "workspace/campaign/governance/control/catalog/manifests/"
                        "CAPABILITY_MANIFEST.json does not exist; nothing claimed"}
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = [classify_entry(e) for e in man.get("entries", [])]
    return {
        "schema": "hawking.capability_manifest_report.v1",
        "entries": results,
        "passed": sum(1 for r in results if r["status"] == "pass"),
        "released": sum(1 for r in results if r["status"] == "released"),
        "failed": sum(1 for r in results if r["status"] == "fail"),
    }


def scaffold(before: str, after: str) -> dict:
    lost = sorted(load_caps(before) - load_caps(after))
    return {
        "schema": "hawking.capability_manifest.v1",
        "note": ("Every entry below is an entrypoint that existed before this rung and does "
                 "not exist after it. Fill `invocation` with the exact command that reaches "
                 "the same capability now, or set `disposition` to \"released\"/\"retired\" "
                 "with the receipt that released it. An entry left as-is is a lost capability."),
        "entries": [
            {"retired_entrypoint": p, "invocation": None, "disposition": None,
             "evidence": None}
            for p in lost
        ],
    }


def gate(before: str, after: str) -> tuple[dict, list[str]]:
    lost = load_caps(before) - load_caps(after)
    rep = check()
    claimed = {e.get("retired_entrypoint") for e in
               (json.loads(MANIFEST.read_text(encoding="utf-8")).get("entries", [])
                if MANIFEST.exists() else [])}
    bad: list[str] = []
    unaccounted = sorted(lost - claimed)
    if unaccounted:
        bad.append(f"{len(unaccounted)} retired entrypoints are not in the manifest at all: "
                   f"{unaccounted[:6]}{' …' if len(unaccounted) > 6 else ''}")
    for e in rep["entries"]:
        if e["status"] == "fail":
            bad.append(
                f"not accounted: {e.get('retired_entrypoint')} -- {e['detail'][:120]}"
            )
        # pass (invocable replacement) and released (evidence-backed) are OK
    rep["lost_entrypoints"] = len(lost)
    rep["unaccounted"] = unaccounted
    return rep, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--scaffold", action="store_true")
    ap.add_argument("--before"); ap.add_argument("--after")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.scaffold:
        if not (args.before and args.after):
            print("--scaffold needs --before and --after", file=sys.stderr); return 2
        doc = scaffold(args.before, args.after)
        out = Path(args.out) if args.out else MANIFEST
        out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"{len(doc['entries'])} retired entrypoints -> {out}")
        return 0

    if args.gate:
        if not (args.before and args.after):
            print("--gate needs --before and --after", file=sys.stderr); return 2
        rep, bad = gate(args.before, args.after)
        print(
            f"manifest: {rep['passed']} invocable, {rep.get('released', 0)} released, "
            f"{rep['failed']} invalid; "
            f"{rep['lost_entrypoints']} entrypoints retired this rung; "
            f"{len(rep.get('unaccounted', []))} unaccounted"
        )
        for b in bad:
            print(f"  ! {b}")
        if args.out:
            Path(args.out).write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        return 1 if bad else 0

    rep = check()
    print(
        f"{rep['passed']} invocable, {rep.get('released', 0)} released, "
        f"{rep['failed']} invalid"
    )
    for e in rep["entries"]:
        if e["status"] == "fail":
            print(f"  ! {e.get('retired_entrypoint')}: {e['detail'][:140]}")
        elif e["status"] == "released":
            print(f"  ~ released: {e.get('retired_entrypoint')}")
    return 1 if rep["failed"] else 0


def _selfcheck() -> None:
    """Smallest thing that fails if the accounting logic breaks."""
    e_ok = run_entry({"retired_entrypoint": "x", "invocation": "true"})
    e_no = run_entry({"retired_entrypoint": "y", "invocation": "false"})
    assert e_ok["status"] == "pass" and e_no["status"] == "fail", (e_ok, e_no)
    assert run_entry({"retired_entrypoint": "z"})["status"] == "fail", "missing invocation must fail"

    r_ok = classify_entry({
        "retired_entrypoint": "runtime:eagle5_event_horizon",
        "disposition": "released",
        "invocation": None,
        "evidence": "BC-ACCEL-009 B-RT3 product release; control/BRT3-report.md",
    })
    assert r_ok["status"] == "released", r_ok

    r_list_ok = classify_entry({
        "retired_entrypoint": "runtime:eagle5_event_horizon",
        "disposition": "released",
        "invocation": None,
        "evidence": ["BC-ACCEL-009", "control/BRT3-report.md"],
    })
    assert r_list_ok["status"] == "released", r_list_ok

    r_obj_ok = classify_entry({
        "retired_entrypoint": "runtime:eagle5_event_horizon",
        "disposition": "released",
        "invocation": None,
        "evidence": {"product_decision": "BC-ACCEL-009", "receipt": "control/BRT3-report.md"},
    })
    assert r_obj_ok["status"] == "released", r_obj_ok

    for bad_ev, label in (
        (None, "None"),
        ("", "empty string"),
        ("   ", "whitespace string"),
        ([], "empty list"),
        ({}, "empty object"),
        ([""], "list of empty strings"),
        ({"x": ""}, "object with empty string values"),
        ({"x": []}, "object with empty list values"),
        ({"x": {}}, "object with empty object values"),
        (0, "zero"),
        (False, "false"),
    ):
        r_bad = classify_entry({
            "retired_entrypoint": "runtime:eagle5_event_horizon",
            "disposition": "released",
            "invocation": None,
            "evidence": bad_ev,
        })
        assert r_bad["status"] == "fail", (label, r_bad)

    r_fake = classify_entry({
        "retired_entrypoint": "runtime:eagle5_event_horizon",
        "disposition": "released",
        "invocation": "true",
        "evidence": "some evidence",
    })
    assert r_fake["status"] == "fail", r_fake

    r_fake_help = classify_entry({
        "retired_entrypoint": "runtime:eagle5_event_horizon",
        "disposition": "released",
        "invocation": "lab --help",
        "evidence": "some evidence",
    })
    assert r_fake_help["status"] == "fail", r_fake_help

    # str({}) / str([]) used to pass the old nonempty check — must not now.
    assert not evidence_is_meaningful({})
    assert not evidence_is_meaningful([])
    assert not evidence_is_meaningful("")
    assert evidence_is_meaningful("x")
    assert str({}).strip() == "{}" and str([]).strip() == "[]"

    lost, claimed = {"a", "b"}, {"a"}
    assert sorted(lost - claimed) == ["b"]
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck(); raise SystemExit(0)
    raise SystemExit(main())
