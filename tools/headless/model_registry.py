#!/usr/bin/env python3
"""Model Registry — the parent/child lineage the Gravity loop needs (§10, §13, §15, §23).

    "CHILD NEVER PROMOTES ITSELF. Promotion belongs to protected deterministic
     verification."

So promotion is not a method on a candidate. It is a function of three
independent artifacts that must all already exist on disk:

    a PerformanceLedger row      (comparable measurements, §25)
    a capability receipt          (deterministic behavioural checks, §13)
    an incumbent to be compared against

`promote()` reads those and refuses if any is missing. There is deliberately no
argument that lets a caller assert a candidate is good.

The registry also carries the thing §23 asks for: when a child loses, its BYTES
become reclaimable but its SCIENCE does not. A rejected child leaves behind
parent hash, recipe, representation settings, benchmark, reason rejected and
reproduction instructions — so the next campaign does not pay to rediscover a
dead end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(os.path.expanduser("~/Downloads/hawking-copy"))
REGISTRY = REPO / "receipts/headless/MODEL_REGISTRY.json"
LEDGER = REPO / "receipts/headless/PERFORMANCE_LEDGER.jsonl"
GIB = 1024 ** 3


def sh(c: str) -> str:
    return subprocess.run(["bash", "-lc", c], capture_output=True, text=True).stdout.strip()


def artifact_identity(path: str) -> dict:
    """Size + a head/tail digest. Cheap, detects truncation and corruption, and
    the receipt says plainly that it is not a full content hash — an identity
    quoted as stronger than it is, is how two different artifacts end up sharing
    a name."""
    p = Path(os.path.expanduser(path))
    if p.is_dir():
        files = sorted(x for x in p.rglob("*") if x.is_file() and not x.is_symlink())
        total = sum(f.stat().st_size for f in files)
        h = hashlib.sha256()
        h.update(str(total).encode())
        for f in files[:64]:
            h.update(str(f.relative_to(p)).encode())
            h.update(str(f.stat().st_size).encode())
        return {"kind": "dir", "path": str(p), "bytes": total,
                "gib": round(total / GIB, 3), "file_count": len(files),
                "identity_kind": "sha256_over_sorted_name_size_manifest",
                "identity": h.hexdigest(),
                "caveat": "structural identity, NOT a content hash of the weights"}
    if p.is_file():
        size = p.stat().st_size
        span = 8 << 20
        h = hashlib.sha256()
        h.update(str(size).encode())
        with open(p, "rb") as fh:
            h.update(fh.read(span))
            if size > span:
                fh.seek(max(0, size - span))
                h.update(fh.read(span))
        return {"kind": "file", "path": str(p), "bytes": size,
                "gib": round(size / GIB, 3),
                "identity_kind": "sha256_size_head_tail_8MiB",
                "identity": h.hexdigest(),
                "caveat": "detects truncation and corruption, NOT a full content hash"}
    return {"kind": "missing", "path": str(p), "identity": None}


def load_registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"schema": "hawking.headless.model_registry.v1",
            "parent": None, "rollback_parent": None,
            "candidates": {}, "sealed_negative_science": [], "history": []}


def save_registry(r: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(REGISTRY) + ".tmp"
    Path(tmp).write_text(json.dumps(r, indent=1))
    os.replace(tmp, REGISTRY)


def ledger_rows() -> dict:
    if not LEDGER.exists():
        return {}
    out = {}
    for line in LEDGER.read_text().splitlines():
        if line.strip():
            try:
                row = json.loads(line)
                out[row["id"]] = row
            except Exception:
                pass
    return out


def capability_receipt(label: str) -> Path | None:
    p = REPO / f"receipts/headless/CAPABILITY_{label}.json"
    return p if p.exists() else None


def register(name: str, path: str, role: str, notes: str = "", recipe: dict | None = None) -> dict:
    r = load_registry()
    entry = {
        "name": name, "role": role, "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact": artifact_identity(path),
        "recipe": recipe or {},
        "notes": notes,
        "source_sha": sh(f"git -C {REPO} rev-parse HEAD"),
    }
    r["candidates"][name] = entry
    if role == "parent":
        if r.get("parent") and r["parent"] != name:
            r["rollback_parent"] = r["parent"]
        r["parent"] = name
    r["history"].append({"at": entry["registered_at"], "action": f"register:{role}", "name": name})
    save_registry(r)
    return entry


def promote(candidate: str, incumbent: str, axis: str,
            cand_ledger_id: str, inc_ledger_id: str,
            cand_capability: str, inc_capability: str) -> dict:
    """Refuses unless BOTH gates already exist on disk. There is no override."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "perfledger", Path(__file__).resolve().parent / "performance_ledger.py")
    pl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pl)

    r = load_registry()
    reasons = []
    for who, nm in (("candidate", candidate), ("incumbent", incumbent)):
        if nm not in r["candidates"]:
            reasons.append(f"{who} {nm!r} is not registered")

    rows = ledger_rows()
    perf = None
    if cand_ledger_id not in rows:
        reasons.append(f"no PerformanceLedger row {cand_ledger_id!r} for the candidate")
    if inc_ledger_id not in rows:
        reasons.append(f"no PerformanceLedger row {inc_ledger_id!r} for the incumbent")
    if not reasons:
        perf = pl.can_promote(rows[cand_ledger_id], rows[inc_ledger_id], axis)
        if not perf.get("allowed"):
            reasons.extend(f"performance gate: {x}" for x in perf.get("reasons", []))
        elif str(perf.get("verdict", "")).startswith("REJECT"):
            reasons.append(f"performance gate: {perf['verdict']}")

    cap_c, cap_i = capability_receipt(cand_capability), capability_receipt(inc_capability)
    cap = None
    if cap_c is None:
        reasons.append(f"no capability receipt CAPABILITY_{cand_capability}.json for the candidate "
                       f"— §13 requires a capability verdict, and 'feels about the same' is not one")
    if cap_i is None:
        reasons.append(f"no capability receipt CAPABILITY_{inc_capability}.json for the incumbent")
    if cap_c and cap_i:
        c, i = json.loads(cap_c.read_text()), json.loads(cap_i.read_text())
        regressions = []
        for axis_name, iv in (i.get("per_axis") or {}).items():
            cv = (c.get("per_axis") or {}).get(axis_name)
            if cv is None:
                regressions.append(f"{axis_name}: candidate was not evaluated on this axis")
            elif cv["rate"] < iv["rate"]:
                regressions.append(f"{axis_name}: {iv['rate']} -> {cv['rate']}")
        cap = {"candidate_overall": c.get("overall"), "incumbent_overall": i.get("overall"),
               "regressions": regressions}
        if regressions:
            reasons.append("capability gate: regressions on " + "; ".join(regressions))

    decision = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate": candidate, "incumbent": incumbent, "change_axis": axis,
        "performance": perf, "capability": cap,
        "allowed": not reasons, "reasons": reasons,
        "rule": ("§10 a child never promotes itself; §13 Doctor/Tabula capability is required; "
                 "§25 promotions without comparable measurements are forbidden"),
    }
    if not reasons:
        r["rollback_parent"] = r.get("parent")
        r["parent"] = candidate
        r["candidates"][candidate]["role"] = "parent"
        if r["rollback_parent"] and r["rollback_parent"] in r["candidates"]:
            r["candidates"][r["rollback_parent"]]["role"] = "rollback_parent"
    r["history"].append({**decision, "action": "promote"})
    save_registry(r)
    return decision


def reject(name: str, reason: str, reproduce: str) -> dict:
    """Seal the science, free the bytes. §23: rejected children leave metadata,
    not weights."""
    r = load_registry()
    if name not in r["candidates"]:
        raise SystemExit(f"{name!r} is not registered")
    e = r["candidates"][name]
    sealed = {
        "name": name, "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent_at_the_time": r.get("parent"),
        "artifact_identity": e["artifact"],
        "recipe": e.get("recipe", {}),
        "reason_rejected": reason,
        "reproduce": reproduce,
        "bytes_reclaimable": e["artifact"].get("bytes"),
        "note": ("the weights may be deleted; this record is what must survive so the next "
                 "campaign does not pay to rediscover the same dead end"),
    }
    r["sealed_negative_science"].append(sealed)
    e["role"] = "rejected"
    r["history"].append({"at": sealed["sealed_at"], "action": "reject", "name": name,
                         "reason": reason})
    save_registry(r)
    return sealed


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("register")
    g.add_argument("--name", required=True)
    g.add_argument("--path", required=True)
    g.add_argument("--role", default="candidate",
                   choices=["parent", "rollback_parent", "candidate", "rejected"])
    g.add_argument("--notes", default="")
    g.add_argument("--recipe", default="{}")
    sub.add_parser("show")
    p = sub.add_parser("promote")
    for a in ("candidate", "incumbent", "cand-ledger-id", "inc-ledger-id",
              "cand-capability", "inc-capability"):
        p.add_argument(f"--{a}", required=True)
    p.add_argument("--axis", default="model")
    j = sub.add_parser("reject")
    j.add_argument("--name", required=True)
    j.add_argument("--reason", required=True)
    j.add_argument("--reproduce", required=True)
    args = ap.parse_args()

    if args.cmd == "register":
        e = register(args.name, args.path, args.role, args.notes, json.loads(args.recipe))
        print(json.dumps(e, indent=1))
        return 0
    if args.cmd == "show":
        r = load_registry()
        print(f"parent           {r.get('parent')}")
        print(f"rollback parent  {r.get('rollback_parent')}")
        print(f"candidates       {len(r.get('candidates', {}))}")
        for n, e in sorted(r.get("candidates", {}).items()):
            a = e["artifact"]
            print(f"  {n:<38} {e['role']:<16} {a.get('gib')} GiB  {str(a.get('identity'))[:12]}")
        if r.get("sealed_negative_science"):
            print(f"sealed negative science  {len(r['sealed_negative_science'])}")
            for s in r["sealed_negative_science"]:
                print(f"  {s['name']:<38} {s['reason_rejected'][:60]}")
        return 0
    if args.cmd == "reject":
        print(json.dumps(reject(args.name, args.reason, args.reproduce), indent=1))
        return 0
    d = promote(args.candidate, args.incumbent, args.axis,
                args.cand_ledger_id, args.inc_ledger_id,
                args.cand_capability, args.inc_capability)
    print(json.dumps(d, indent=1))
    return 0 if d["allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
