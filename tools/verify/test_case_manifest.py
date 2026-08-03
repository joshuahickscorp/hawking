#!/usr/bin/env python3.12
"""Test-case manifest reconciler skeleton (Core F F1).

Pairs workspace/campaign/governance/control/catalog/manifests/ASSERTION_LEDGER.json
(immutable seal of prior identities) with
workspace/campaign/governance/control/catalog/manifests/TEST_CASE_MANIFEST.json
(live dispositions). F1 ships an empty manifest:
--seal-check may pass; normal --gate must fail and list every unaccounted ledger id.

Stdlib only. Does not reimplement cargo/pytest/vitest or claim execution receipts
that do not exist.

    python3.12 tools/verify/test_case_manifest.py --seal-check LEDGER MANIFEST
    python3.12 tools/verify/test_case_manifest.py --enumerate MANIFEST
    python3.12 tools/verify/test_case_manifest.py --dry-run MANIFEST
    python3.12 tools/verify/test_case_manifest.py --gate --before LEDGER --after MANIFEST
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.layout import resolve_workspace_path

SCHEMA = "hawking.test_case_manifest.v1"
LEDGER_SCHEMA = "hawking.assertion_ledger.v1"

CLOSED_DISPOSITIONS = frozenset(
    {
        "execute",
        "rewrite",
        "retired_subject",
        "product_waiver",
        "blocked_fixture",
        "blocked_model",
        "superseded",
    }
)

ACCOUNTING_DISPOSITIONS = CLOSED_DISPOSITIONS
BLOCKED = frozenset({"blocked_fixture", "blocked_model"})
F1_PHASES = frozenset({"f1_scaffold", "f1", "scaffold"})
F1_STATUSES = frozenset(
    {
        "no_rewrite_accounting",
        "no_rewrite_accounting_begun",
        "empty_scaffold",
    }
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else resolve_workspace_path(p)


def validate_manifest_schema(man: dict[str, Any]) -> list[str]:
    bad: list[str] = []
    if man.get("schema") != SCHEMA:
        bad.append(f"schema must be {SCHEMA}, got {man.get('schema')!r}")
    if "version" not in man:
        bad.append("missing version")
    if "ledger_ref" not in man or not isinstance(man["ledger_ref"], dict):
        bad.append("missing ledger_ref object")
    else:
        if not man["ledger_ref"].get("path"):
            bad.append("ledger_ref.path required")
        if not man["ledger_ref"].get("sha256"):
            bad.append("ledger_ref.sha256 required")
    if "entries" not in man or not isinstance(man["entries"], list):
        bad.append("entries must be a list")
    return bad


def validate_entries(entries: list[dict[str, Any]]) -> list[str]:
    bad: list[str] = []
    seen: dict[str, int] = {}
    by_id: dict[str, dict[str, Any]] = {}

    for i, e in enumerate(entries):
        cid = e.get("case_id")
        if not cid:
            bad.append(f"entries[{i}]: missing case_id")
            continue
        if cid in seen:
            bad.append(f"duplicate case_id in manifest: {cid}")
        seen[cid] = i
        by_id[cid] = e

        disp = e.get("disposition")
        if disp not in CLOSED_DISPOSITIONS:
            bad.append(f"{cid}: disposition {disp!r} not in closed set")
            continue

        replaces = e.get("replaces") or []
        if not isinstance(replaces, list):
            bad.append(f"{cid}: replaces must be a list")
            replaces = []

        if disp == "rewrite" and not replaces:
            bad.append(f"{cid}: rewrite requires non-empty replaces")

        if disp == "retired_subject":
            if not e.get("deleted_subject") and not e.get("product_decision"):
                bad.append(
                    f"{cid}: retired_subject requires deleted_subject or product_decision"
                )

        if disp == "product_waiver":
            if not e.get("product_decision"):
                bad.append(f"{cid}: product_waiver requires product_decision")

        if disp in BLOCKED:
            expected = e.get("expected_status")
            if expected in ("pass", "passed", "ok", "green"):
                bad.append(
                    f"{cid}: blocked disposition cannot claim expected_status={expected!r}"
                )
            if e.get("status") in ("pass", "passed", "ok"):
                bad.append(f"{cid}: blocked row cannot report status=pass")

        if disp == "superseded":
            target = e.get("superseded_by") or e.get("target")
            if not target:
                bad.append(f"{cid}: superseded requires superseded_by")
            elif target == cid:
                bad.append(f"{cid}: superseded cannot point to self")

        if disp == "execute":
            if e.get("claimed_pass") and not e.get("receipt_ref"):
                bad.append(f"{cid}: claimed_pass without receipt_ref is forbidden")

    supersede_edges: dict[str, str] = {}
    for cid, e in by_id.items():
        if e.get("disposition") == "superseded":
            target = e.get("superseded_by") or e.get("target")
            if target:
                supersede_edges[cid] = target

    for src, tgt in supersede_edges.items():
        if tgt not in by_id:
            bad.append(f"{src}: superseded_by {tgt!r} not present in manifest")

    def cycle_from(start: str) -> bool:
        seen_n: set[str] = set()
        cur = start
        while cur in supersede_edges:
            if cur in seen_n:
                return True
            seen_n.add(cur)
            cur = supersede_edges[cur]
            if cur == start:
                return True
        return False

    for src in supersede_edges:
        if cycle_from(src):
            bad.append(f"circular supersession involving {src}")
            break

    for cid, e in by_id.items():
        replaces = e.get("replaces") or []
        if replaces and len(replaces) != len(set(replaces)):
            bad.append(f"{cid}: replaces contains duplicates")

    return bad


def seal_check(ledger_path: Path, manifest_path: Path) -> tuple[int, list[str]]:
    msgs: list[str] = []
    if not ledger_path.exists():
        return 2, [f"missing ledger {ledger_path}"]
    if not manifest_path.exists():
        return 2, [f"missing manifest {manifest_path}"]

    ledger = load_json(ledger_path)
    man = load_json(manifest_path)

    if ledger.get("schema") != LEDGER_SCHEMA:
        msgs.append(f"ledger schema {ledger.get('schema')!r} != {LEDGER_SCHEMA}")

    msgs.extend(validate_manifest_schema(man))
    if msgs:
        return 1, msgs

    want_hash = man["ledger_ref"]["sha256"]
    got_hash = sha256_file(ledger_path)
    if want_hash != got_hash:
        msgs.append(
            f"ledger_ref.sha256 mismatch: manifest={want_hash} file={got_hash}"
        )

    ref_path = man["ledger_ref"]["path"].replace("\\", "/")
    if ref_path not in {
        str(ledger_path.relative_to(ROOT)) if ledger_path.is_relative_to(ROOT) else "",
        "control/ASSERTION_LEDGER.json",
        ledger_path.name,
    } and not str(ledger_path).endswith(ref_path):
        msgs.append(
            f"ledger_ref.path {ref_path!r} does not match provided ledger {ledger_path}"
        )

    phase = str(man.get("phase", "")).lower()
    status = str(man.get("status", "")).lower()
    entries = man.get("entries") or []

    if entries:
        msgs.extend(validate_entries(entries))
    else:
        if phase not in F1_PHASES and status not in F1_STATUSES:
            msgs.append(
                "empty F1 manifest requires phase in "
                f"{sorted(F1_PHASES)} or status in {sorted(F1_STATUSES)}"
            )

    if msgs:
        return 1, msgs
    return 0, [
        f"seal-check PASS  ledger_sha256={got_hash[:16]}…  entries={len(entries)}  "
        f"phase={man.get('phase')!r} status={man.get('status')!r}"
    ]


def enumerate_manifest(manifest_path: Path) -> tuple[int, dict[str, Any]]:
    man = load_json(manifest_path)
    bad = validate_manifest_schema(man)
    if bad:
        return 2, {"errors": bad}
    entries = man.get("entries") or []
    bad = validate_entries(entries)
    ids = sorted(e["case_id"] for e in entries if e.get("case_id"))
    return (1 if bad else 0), {
        "schema": "hawking.test_case_manifest_enumerate.v1",
        "count": len(ids),
        "case_ids": ids,
        "errors": bad,
    }


def dry_run(manifest_path: Path) -> tuple[int, dict[str, Any]]:
    """Resolve schema/fixtures; no product side effects; no false execution claims."""
    man = load_json(manifest_path)
    bad = validate_manifest_schema(man)
    bad.extend(validate_entries(man.get("entries") or []))
    results = []
    for e in man.get("entries") or []:
        disp = e.get("disposition")
        cid = e.get("case_id")
        if disp == "execute":
            exe = e.get("executor") or {}
            cmd = exe.get("command")
            if not cmd:
                results.append(
                    {
                        "case_id": cid,
                        "dry_run": "unresolved",
                        "detail": "execute without executor.command",
                    }
                )
            else:
                results.append(
                    {
                        "case_id": cid,
                        "dry_run": "resolved",
                        "command": cmd,
                        "executed": False,
                    }
                )
        elif disp in BLOCKED:
            results.append(
                {
                    "case_id": cid,
                    "dry_run": "blocked",
                    "disposition": disp,
                    "status": "blocked",
                }
            )
        else:
            results.append(
                {
                    "case_id": cid,
                    "dry_run": "accounted",
                    "disposition": disp,
                    "executed": False,
                }
            )
    return (1 if bad else 0), {
        "schema": "hawking.test_case_manifest_dry_run.v1",
        "errors": bad,
        "results": results,
        "note": "F1 dry-run does not execute tests or claim receipts",
    }


def accounting_owners(entries: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Map identity -> list of owner tags. Exactly one owner required per ledger id."""
    owners: dict[str, list[str]] = {}

    def add(lid: str, tag: str) -> None:
        owners.setdefault(lid, []).append(tag)

    for e in entries:
        cid = e.get("case_id")
        disp = e.get("disposition")
        if not cid or disp not in ACCOUNTING_DISPOSITIONS:
            continue
        add(cid, f"self:{disp}:{cid}")
        for old in e.get("replaces") or []:
            add(old, f"replaced_by:{cid}")
    return owners


def rewrite_receipt_identities(
    entry: dict[str, Any],
    receipt: dict[str, Any],
) -> set[str] | None:
    """Return the set of old identities covered by receipt evidence, or None if absent.

    A complete N-entry identity mapping is required for N replaces. Accepted forms:
    - entry.receipt_map: {old_id: ...} covering exactly the replaced ids
    - receipt.identity_map / replacement_map / replaces_map keyed by new case_id
    - receipt.anti_gaming.identity_map[new_id] as dict or list of old ids
    - result rows with rewrite_of / old_case_id, or rows whose case_id is an old id
    """
    cid = entry.get("case_id")
    replaces = list(entry.get("replaces") or [])
    if not replaces:
        return set()

    mapped = entry.get("receipt_map")
    if isinstance(mapped, dict) and mapped:
        return set(mapped.keys())

    for key in ("identity_map", "replacement_map", "replaces_map"):
        top = receipt.get(key)
        if isinstance(top, dict) and cid in top:
            v = top[cid]
            if isinstance(v, dict):
                return set(v.keys())
            if isinstance(v, list):
                return set(v)

    ag = receipt.get("anti_gaming") or {}
    im = ag.get("identity_map") or ag.get("replaced_ids")
    if isinstance(im, dict) and cid in im:
        v = im[cid]
        if isinstance(v, dict):
            return set(v.keys())
        if isinstance(v, list):
            return set(v)

    covered: set[str] = set()
    for r in receipt.get("results", []) or []:
        old = r.get("rewrite_of") or r.get("old_case_id")
        if old in replaces:
            covered.add(old)
        rid = r.get("case_id")
        if rid in replaces:
            covered.add(rid)
    if covered:
        return covered
    return None


def gate(
    ledger_path: Path,
    manifest_path: Path,
    receipt_path: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Every ledger case_id must have exactly one live disposition path.

    Empty F1 manifest MUST fail and enumerate every unaccounted id.
    There is no 'ledger-only gate pass'.
    """
    bad: list[str] = []
    ledger = load_json(ledger_path)
    man = load_json(manifest_path)

    if ledger.get("schema") != LEDGER_SCHEMA:
        bad.append(f"bad ledger schema {ledger.get('schema')!r}")
    bad.extend(validate_manifest_schema(man))
    entries = list(man.get("entries") or [])
    bad.extend(validate_entries(entries))

    ref = man.get("ledger_ref") or {}
    if ref.get("sha256") and ref["sha256"] != sha256_file(ledger_path):
        bad.append("manifest ledger_ref.sha256 does not match --before ledger file")

    ledger_ids = [e["case_id"] for e in ledger.get("entries", [])]
    ledger_set = set(ledger_ids)
    owners = accounting_owners(entries)

    # Reject unknown / manifest-only case ids (unless rewrite introducing a new id).
    for e in entries:
        cid = e.get("case_id")
        if not cid:
            continue
        replaces = e.get("replaces") or []
        disp = e.get("disposition")
        if replaces:
            for old in replaces:
                if old not in ledger_set:
                    bad.append(f"{cid}: replaces unknown ledger id {old}")
            # New rewrite identity may sit outside the sealed ledger.
        else:
            if cid not in ledger_set:
                bad.append(f"{cid}: manifest case_id not in ledger (unknown)")

        # replaces without rewrite/execute accounting path is odd but allowed only
        # on rewrite; already validated.

    # Exactly one accounting owner per sealed ledger id that is claimed.
    for lid, tags in sorted(owners.items()):
        if lid in ledger_set and len(tags) > 1:
            bad.append(
                f"{lid}: multiple accounting owners ({len(tags)}): {tags}"
            )

    accounted_ledger = {lid for lid in owners if lid in ledger_set}
    unaccounted = sorted(ledger_set - accounted_ledger)

    receipt = None
    if receipt_path is not None:
        if not receipt_path.exists():
            bad.append(f"receipt missing: {receipt_path}")
        else:
            receipt = load_json(receipt_path)
            results = {
                r["case_id"]: r
                for r in receipt.get("results", [])
                if r.get("case_id")
            }
            for e in entries:
                cid = e.get("case_id")
                disp = e.get("disposition")
                if disp in BLOCKED:
                    r = results.get(cid)
                    if r and r.get("status") in ("pass", "passed", "ok"):
                        bad.append(
                            f"{cid}: blocked disposition reported pass in receipt"
                        )
                if disp in ("execute", "rewrite"):
                    replaces = list(e.get("replaces") or [])
                    if replaces:
                        covered = rewrite_receipt_identities(e, receipt)
                        want = set(replaces)
                        if covered is None:
                            bad.append(
                                f"{cid}: rewrite of {len(replaces)} ids requires "
                                f"complete N-entry receipt identity map (got none)"
                            )
                        else:
                            missing = sorted(want - covered)
                            extra = sorted(covered - want)
                            if missing:
                                bad.append(
                                    f"{cid}: receipt identity map missing "
                                    f"{len(missing)} replaced id(s): {missing[:5]}"
                                )
                            if extra:
                                bad.append(
                                    f"{cid}: receipt identity map has "
                                    f"{len(extra)} extra id(s): {extra[:5]}"
                                )
                            if len(covered) != len(want) and not missing and not extra:
                                # duplicate coverage collapsed by set — still wrong count
                                bad.append(
                                    f"{cid}: receipt identity map size mismatch"
                                )
                    else:
                        if cid not in results and e.get("expected_status") != "unavailable_env":
                            bad.append(f"{cid}: execute missing from receipt results")

            ag = receipt.get("anti_gaming") or {}
            if (
                ag
                and "runner_reported_n" in ag
                and "enumerated_n" in ag
                and ag.get("runner_reported_n") != ag.get("enumerated_n")
            ):
                bad.append(
                    f"receipt anti_gaming mismatch: "
                    f"runner_reported_n={ag.get('runner_reported_n')} "
                    f"enumerated_n={ag.get('enumerated_n')}"
                )

    rep = {
        "schema": "hawking.test_case_manifest_gate.v1",
        "ledger_total": len(ledger_ids),
        "manifest_entries": len(entries),
        "accounted": len(accounted_ledger),
        "unaccounted_count": len(unaccounted),
        "unaccounted": unaccounted,
        "errors": bad,
        "pass": not bad and not unaccounted,
        "note": (
            "Normal --gate requires every ledger id to be accounted by exactly one "
            "manifest disposition path. An empty F1 scaffold must fail here; only "
            "--seal-check may pass with empty entries. Manifest-only unknown ids "
            "and dual ownership of a sealed id fail. N replaces require a complete "
            "N-entry receipt identity map when a receipt is supplied."
        ),
    }
    if unaccounted:
        bad.append(f"{len(unaccounted)} ledger case_ids unaccounted in manifest")
        rep["errors"] = bad
        rep["pass"] = False

    # Recompute pass if bad accumulated after initial set.
    if bad:
        rep["pass"] = False
        rep["errors"] = bad

    return (0 if rep["pass"] else 1), rep


def scaffold_manifest(ledger_path: Path) -> dict[str, Any]:
    """Build the F1 empty scaffold pointing at the sealed ledger."""
    return {
        "schema": SCHEMA,
        "version": 1,
        "phase": "f1_scaffold",
        "status": "no_rewrite_accounting",
        "note": (
            "F1 seal only. No rewrite accounting has begun. entries is empty by design. "
            "tools/verify/test_case_manifest.py --seal-check may pass; normal --gate must "
            "fail and list every unaccounted ledger identity."
        ),
        "ledger_ref": {
            "path": "control/ASSERTION_LEDGER.json",
            "sha256": sha256_file(ledger_path),
        },
        "entries": [],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seal-check",
        nargs=2,
        metavar=("LEDGER", "MANIFEST"),
        help="validate ledger hash + F1 scaffold (empty entries allowed)",
    )
    ap.add_argument(
        "--enumerate",
        metavar="MANIFEST",
        help="emit sorted case_id list from manifest (no execute)",
    )
    ap.add_argument(
        "--dry-run",
        metavar="MANIFEST",
        help="resolve commands/schema only; no product side effects",
    )
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--before", help="ledger path for --gate")
    ap.add_argument("--after", help="manifest path for --gate")
    ap.add_argument("--receipt", help="optional execution receipt for --gate")
    ap.add_argument(
        "--write-scaffold",
        metavar="MANIFEST",
        help="write F1 empty manifest for the ledger at --before",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.write_scaffold:
        if not args.before:
            print("--write-scaffold requires --before LEDGER", file=sys.stderr)
            return 2
        doc = scaffold_manifest(resolve(args.before))
        out = resolve(args.write_scaffold)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote scaffold {out}  ledger_sha256={doc['ledger_ref']['sha256'][:16]}…")
        return 0

    if args.seal_check:
        rc, msgs = seal_check(resolve(args.seal_check[0]), resolve(args.seal_check[1]))
        for m in msgs:
            print(m)
        return rc

    if args.enumerate:
        rc, doc = enumerate_manifest(resolve(args.enumerate))
        if args.json:
            print(json.dumps(doc, indent=2, sort_keys=True))
        else:
            print(f"enumerated={doc.get('count', 0)}")
            for e in doc.get("errors") or []:
                print(f"  ! {e}")
        return rc

    if args.dry_run:
        rc, doc = dry_run(resolve(args.dry_run))
        if args.json:
            print(json.dumps(doc, indent=2, sort_keys=True))
        else:
            print(
                f"dry-run results={len(doc.get('results', []))} "
                f"errors={len(doc.get('errors', []))}"
            )
            for e in doc.get("errors") or []:
                print(f"  ! {e}")
        return rc

    if args.gate:
        if not (args.before and args.after):
            print("--gate requires --before LEDGER --after MANIFEST", file=sys.stderr)
            return 2
        rc, rep = gate(
            resolve(args.before),
            resolve(args.after),
            resolve(args.receipt) if args.receipt else None,
        )
        if args.json:
            print(json.dumps(rep, indent=2, sort_keys=True))
        else:
            status = "PASS" if rep["pass"] else "FAIL"
            print(
                f"TEST CASE MANIFEST GATE {status}  "
                f"ledger={rep['ledger_total']} accounted={rep['accounted']} "
                f"unaccounted={rep['unaccounted_count']}"
            )
            for e in rep.get("errors") or []:
                print(f"  ! {e}")
            if rep["unaccounted"]:
                print("unaccounted_ids:")
                for cid in rep["unaccounted"]:
                    print(f"  {cid}")
        return rc

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
