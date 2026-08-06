#!/usr/bin/env python3.12
"""The single command that decides whether a rebuild checkpoint closes.

Every rung of the 250k campaign has to clear measurement, build, test, behaviour,
performance, migration and rollback at once. Run by hand these drift apart, and this
repository has already closed a checkpoint on half an inventory diff -- a merge deleted a
live 220-line entrypoint and was reported capability-preserving because only the detector
half ran, never the assertion half.

So this orchestrates the existing tools; it reimplements none of them.

    rung_gate.py --capture --label baseline --out workspace/campaign/governance/control/rungs/baseline.json
    rung_gate.py --rung 400k --before workspace/campaign/governance/control/rungs/baseline.json --out workspace/campaign/governance/control/rungs/400k.json
    rung_gate.py --check workspace/campaign/governance/control/rungs/400k.json
    rung_gate.py --quick

The refusal that matters most: a check that STOPS RUNNING is a regression, not a neutral.
If a check passed in --before and is skipped or unavailable now, the gate goes red.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.layout import CONTROL_ROOT, evidence_dir, resolve_workspace_path

RULES = CONTROL_ROOT / "catalog/manifests/REBUILD_ACCOUNTING_RULES.json"
FILES_OVER_1500_START = 26  # campaign start value; a rung may not push it above this
PERF_GATE_PCT = 2.0

PY = "python3.12"


def sh(cmd: list[str] | str, timeout: int = 3600, cwd: Path | None = None) -> tuple[int, str]:
    try:
        r = subprocess.run(
            cmd, cwd=cwd or ROOT, shell=isinstance(cmd, str),
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s"
    except FileNotFoundError as e:
        return 127, str(e)


def _first_json(text: str) -> dict:
    """Pull the first JSON object out of mixed stdout+stderr. Tools print a report and
    then a human-readable refusal line; a plain json.loads chokes on the tail."""
    dec = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        i = text.find("{", i + 1)
    return {}


def check(name: str, status: str, reason: str, **extra) -> dict:
    assert status in ("pass", "fail", "skip", "unavailable")
    return {"name": name, "status": status, "reason": reason, **extra}


# --------------------------------------------------------------------------- measurement

def measure() -> tuple[dict, list[dict]]:
    checks: list[dict] = []
    m: dict = {}

    rc, out = sh([PY, "tools/loc/hawking_loc.py", "--json"], timeout=900)
    if rc != 0:  # the tool's human mode is the default; fall back and parse nothing
        rc2, out2 = sh([PY, "tools/loc/hawking_loc.py"], timeout=900)
        checks.append(check("loc", "fail" if rc2 else "pass", out2.strip()[-400:]))
        m["loc_raw"] = None
    else:
        try:
            m["loc_raw"] = json.loads(out)["combined_active_monorepo_LOC"]
            checks.append(check("loc", "pass", f"{m['loc_raw']:,}"))
        except (json.JSONDecodeError, KeyError) as e:
            m["loc_raw"] = None
            checks.append(check("loc", "fail", f"unparseable loc output: {e}"))

    rc, out = sh([PY, "tools/loc/hawking_topology.py", "--json"], timeout=900)
    m["topology"] = None
    try:
        m["topology"] = json.loads(out)
        checks.append(check("topology", "pass", "measured"))
    except json.JSONDecodeError:
        rc2, out2 = sh([PY, "tools/loc/hawking_topology.py"], timeout=900)
        m["topology"] = _parse_topology_text(out2)
        checks.append(
            check("topology", "pass" if m["topology"] else "fail",
                  "parsed from human output" if m["topology"] else out2.strip()[-300:])
        )

    rc, out = sh([PY, "tools/loc/hawking_generation_audit.py", "--gate", "--json"], timeout=1800)
    g = _first_json(out)
    m["generation_reclassified"] = g.get("reclassified_active_LOC", 0)
    checks.append(
        check("generation_audit", "pass" if rc == 0 else "fail",
              f"{g.get('unearned_files', '?')} unearned, "
              f"{m['generation_reclassified']:,} LOC reclassified active")
    )

    if m["loc_raw"] is not None:
        m["loc_gated"] = m["loc_raw"] + m["generation_reclassified"]
    else:
        m["loc_gated"] = None
    return m, checks


def _parse_topology_text(text: str) -> dict:
    keys = {
        "directories_all": r"directories_all\s+([\d,]+)",
        "source_files": r"source_files\s+([\d,]+)",
        "rust_crates": r"rust_crates\s+([\d,]+)",
        "hide_crates": r"hide_crates\s+([\d,]+)",
        "hide_files": r"hide_files\s+([\d,]+)",
        "hide_directories": r"hide_directories\s+([\d,]+)",
        "public_symbols": r"public_symbols\s+([\d,]+)",
        "functions": r"functions\s+([\d,]+)",
        "files_over_1500_lines": r"files >1500 lines\s+([\d,]+)",
        "tiny_forwarders": r"tiny forwarders\s+([\d,]+)",
    }
    out = {}
    for k, pat in keys.items():
        mm = re.search(pat, text)
        if mm:
            out[k] = int(mm.group(1).replace(",", ""))
    return out


# ------------------------------------------------------------------------------- groups

def group_inventory(before: dict | None) -> list[dict]:
    snap = CONTROL_ROOT / "rungs" / "current" / "_inventory_current"
    rc, out = sh([PY, "tools/loc/hawking_inventory.py", "--snapshot", str(snap)], timeout=1800)
    if rc != 0:
        return [check("inventory_snapshot", "fail", out.strip()[-300:])]
    checks = [check("inventory_snapshot", "pass", str(snap))]
    prev = (before or {}).get("inventory_snapshot")
    if not prev:
        # Never a skip. A missing baseline half is exactly the failure mode this exists for.
        checks.append(check("inventory_gate", "fail",
                            "no previous snapshot to diff against; both halves are required"))
    else:
        rc, out = sh([PY, "tools/loc/hawking_inventory.py", "--gate", prev, str(snap)], timeout=1800)
        checks.append(check("inventory_gate", "pass" if rc == 0 else "fail", out.strip()[-400:]))
    return checks


def group_build() -> list[dict]:
    checks = []
    for name, cmd, t in (
        ("cargo_build_workspace", ["cargo", "build", "--workspace"], 3600),
        ("cargo_build_hawking_release", ["cargo", "build", "-p", "hawking", "--release"], 3600),
    ):
        rc, out = sh(cmd, timeout=t)
        checks.append(check(name, "pass" if rc == 0 else "fail",
                            "ok" if rc == 0 else out.strip()[-600:]))
    return checks


def group_rust_tests() -> list[dict]:
    rc, out = sh(["cargo", "test", "--workspace", "--no-fail-fast"], timeout=5400)
    passed = failed = ignored = 0
    for mm in re.finditer(
        r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored", out
    ):
        passed += int(mm.group(1)); failed += int(mm.group(2)); ignored += int(mm.group(3))
    return [check(
        "cargo_test", "pass" if failed == 0 and rc == 0 else "fail",
        f"{passed} passed, {failed} failed, {ignored} ignored",
        passed=passed, failed=failed, ignored=ignored,
    )]


def group_python_tests() -> list[dict]:
    rc, out = sh([PY, "-m", "pytest", "tools/", "ramanujan/scaffold/", "workspace/campaign/governance/odyssey/",
                  "-q", "--no-header", "--tb=no", "-p", "no:cacheprovider"], timeout=5400)
    mm = re.search(
        r"(?:(\d+) failed)?(?:, )?(\d+) passed(?:, (\d+) skipped)?"
        r"(?:, \d+ warnings?)?(?:, (\d+) errors?)?", out
    )
    f = int(mm.group(1) or 0) if mm else -1
    p = int(mm.group(2) or 0) if mm else -1
    s = int(mm.group(3) or 0) if mm else -1
    e = int(mm.group(4) or 0) if mm else -1
    return [check(
        "pytest", "pass" if (f == 0 and e == 0) else "fail",
        f"{p} passed, {f} failed, {s} skipped, {e} errors",
        passed=p, failed=f, skipped=s, errors=e,
    )]


def group_blackbox() -> list[dict]:
    tool = ROOT / "tools" / "verify" / "blackbox.py"
    if not tool.exists():
        return [check("blackbox", "unavailable", "tools/verify/blackbox.py does not exist yet")]
    rc, out = sh([PY, str(tool), "--json"], timeout=3600)
    d = _first_json(out)
    if not d:
        return [check("blackbox", "fail", f"no JSON report from blackbox.py: {out.strip()[-300:]}")]
    return [check("blackbox", "pass" if rc == 0 else "fail",
                  f"{d.get('passed','?')} passed, {d.get('failed','?')} failed, "
                  f"{d.get('skipped','?')} skipped",
                  runnable=d.get("runnable"), covered=d.get("covered"))]


def group_perf(before_path: str | None) -> list[dict]:
    tool = ROOT / "tools" / "verify" / "perfgate.py"
    if not tool.exists():
        return [check("performance", "unavailable", "tools/verify/perfgate.py does not exist yet")]
    if not before_path:
        rc, out = sh([PY, str(tool), "--capture"], timeout=5400)
        return [check("performance", "pass" if rc == 0 else "fail",
                      "baseline captured" if rc == 0 else out.strip()[-400:])]
    rc, out = sh([PY, str(tool), "--compare", before_path, "-", "--gate", str(PERF_GATE_PCT)],
                 timeout=5400)
    return [check("performance", "pass" if rc == 0 else "fail", out.strip()[-600:])]


def group_clean_clone() -> list[dict]:
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "clone"
        rc, out = sh(["git", "clone", "--no-hardlinks", "--depth", "1", str(ROOT), str(dst)],
                     timeout=1800)
        if rc != 0:
            return [check("clean_clone", "fail", out.strip()[-300:])]
        env = ("CARGO_NET_OFFLINE=true PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 "
               "TRANSFORMERS_OFFLINE=1 no_proxy='*'")
        rc, out = sh(f"{env} cargo build -p hawking", timeout=3600, cwd=dst)
        return [check(
            "clean_clone_offline", "pass" if rc == 0 else "fail",
            ("built offline" if rc == 0 else out.strip()[-600:])
            + " | approximation: offline enforced by env vars, not a network sandbox",
        )]


def group_rollback(tags: list[str]) -> list[dict]:
    if not tags:
        return [check("rollback_tags", "fail", "no rollback tag named for this rung")]
    checks = []
    for tag in tags:
        rc, _ = sh(["git", "rev-parse", "--verify", f"{tag}^{{commit}}"], timeout=60)
        if rc != 0:
            checks.append(check(f"rollback:{tag}", "fail", "tag does not resolve"))
            continue
        with tempfile.TemporaryDirectory() as td:
            wt = Path(td) / "wt"
            rc, out = sh(["git", "worktree", "add", "--detach", str(wt), tag], timeout=600)
            ok = rc == 0
            sh(["git", "worktree", "remove", "--force", str(wt)], timeout=300)
            checks.append(check(f"rollback:{tag}", "pass" if ok else "fail",
                                "checkout ok" if ok else out.strip()[-300:]))
    return checks


def group_migration() -> list[dict]:
    contract = evidence_dir("rebuild") / "REBUILD_DATA_MIGRATION_CONTRACT.json"
    if not contract.exists():
        return [check("migration", "unavailable",
                      "REBUILD_DATA_MIGRATION_CONTRACT.json does not exist yet")]
    try:
        d = json.loads(contract.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [check("migration", "fail", f"contract is not valid JSON: {e}")]
    missing = [
        s for e in d.get("entries", [])
        if (s := e.get("sample_path")) and not resolve_workspace_path(s).exists()
    ]
    return [check("migration", "pass" if not missing else "fail",
                  "all sample files present" if not missing
                  else f"{len(missing)} sample files missing: {missing[:5]}")]


# ------------------------------------------------------------------------------ refusals

def refusals(rec: dict, before: dict | None, ledger: dict | None) -> list[str]:
    bad: list[str] = []
    st = {c["name"]: c for c in rec["checks"]}

    for c in rec["checks"]:
        if c["status"] == "fail":
            bad.append(f"check failed: {c['name']} -- {c['reason'][:160]}")

    # The anti-gaming rule. A check that stops running is a regression.
    if before:
        for c in before.get("checks", []):
            if c["status"] == "pass":
                now = st.get(c["name"])
                if now is None:
                    bad.append(f"check disappeared: {c['name']} passed at the previous rung")
                elif now["status"] in ("skip", "unavailable"):
                    bad.append(
                        f"check stopped running: {c['name']} passed before, is now "
                        f"{now['status']} ({now['reason'][:120]})"
                    )

    topo = rec["measured"].get("topology") or {}
    prev_topo = ((before or {}).get("measured") or {}).get("topology") or {}
    for k, v in topo.items():
        if k in prev_topo and isinstance(v, int) and isinstance(prev_topo[k], int):
            if v > prev_topo[k]:
                bad.append(f"topology regressed: {k} {prev_topo[k]} -> {v}")
    if topo.get("files_over_1500_lines", 0) > FILES_OVER_1500_START:
        bad.append(
            f"files_over_1500_lines {topo['files_over_1500_lines']} exceeds the campaign "
            f"start value {FILES_OVER_1500_START}"
        )

    if ledger:
        if ledger.get("facade", 0) > 0:
            bad.append(f"facade LOC remaining: {ledger['facade']}")
        if before and rec["measured"].get("loc_gated") and before["measured"].get("loc_gated"):
            delta = rec["measured"]["loc_gated"] - before["measured"]["loc_gated"]
            claimed = -(ledger.get("eliminated", 0) + ledger.get("rewritten", 0)) \
                + ledger.get("generated", 0) + ledger.get("relocated", 0) + ledger.get("facade", 0)
            residual = delta - claimed
            rec["ledger_residual"] = residual
            if residual != 0:
                bad.append(
                    f"ledgers do not reconcile: measured delta {delta}, ledgers claim "
                    f"{claimed}, residual {residual}"
                )
    else:
        bad.append("no ledger file supplied; the five-ledger accounting is required to close")

    return bad


# ---------------------------------------------------------------------------------- main

def run(args) -> dict:
    before = json.loads(Path(args.before).read_text(encoding="utf-8")) if args.before else None
    checks: list[dict] = []
    measured, mchecks = measure()
    checks += mchecks

    if args.quick:
        print("=" * 72)
        print("  --quick: MEASUREMENT ONLY. This is NOT a gate result and writes no receipt.")
        print("=" * 72)
    else:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {
                "inventory": ex.submit(group_inventory, before),
                "build": ex.submit(group_build),
                "blackbox": ex.submit(group_blackbox),
                "perf": ex.submit(group_perf, args.perf_before),
                "clone": ex.submit(group_clean_clone),
                "migration": ex.submit(group_migration),
                "rollback": ex.submit(group_rollback, args.rollback_tag or []),
            }
            for f in futs.values():
                checks += f.result()
        # Tests are serialised after the build so they do not race cargo's lock.
        checks += group_rust_tests()
        checks += group_python_tests()

    rec = {
        "schema": "hawking.rung_receipt.v1",
        "rung": args.rung or args.label or "unlabelled",
        "commit": sh(["git", "rev-parse", "HEAD"])[1].strip(),
        "measured": measured,
        "checks": checks,
        "inventory_snapshot": str(CONTROL_ROOT / "rungs" / "current" / "_inventory_current"),
        "rollback_tag": args.rollback_tag or [],
        "before": args.before,
    }
    ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8")) if args.ledger else None
    rec["ledger"] = ledger
    rec["refusals"] = [] if args.quick else refusals(rec, before, ledger)
    rec["closes"] = (not rec["refusals"]) and not args.quick
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung"); ap.add_argument("--label")
    ap.add_argument("--before", help="previous rung receipt to diff against")
    ap.add_argument("--perf-before", help="previous performance baseline json")
    ap.add_argument("--ledger", help="workspace/campaign/governance/control/rungs/<rung>-ledger.json")
    ap.add_argument("--rollback-tag", action="append")
    ap.add_argument("--out"); ap.add_argument("--capture", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--check", help="re-verify an existing receipt and exit")
    args = ap.parse_args()

    if args.check:
        rec = json.loads(Path(args.check).read_text(encoding="utf-8"))
        for r in rec.get("refusals", []):
            print(f"  ! {r}")
        print(f"closes: {rec.get('closes')}")
        return 0 if rec.get("closes") else 1

    rec = run(args)

    m = rec["measured"]
    print(f"rung {rec['rung']} @ {rec['commit'][:12]}")
    if m.get("loc_gated") is not None:
        print(f"  LOC {m['loc_raw']:,} + {m['generation_reclassified']:,} reclassified "
              f"= {m['loc_gated']:,}")
    t = m.get("topology") or {}
    if t:
        print(f"  dirs {t.get('directories_all')}  files {t.get('source_files')}  "
              f"crates {t.get('rust_crates')}  symbols {t.get('public_symbols')}  "
              f"functions {t.get('functions')}")
    for c in rec["checks"]:
        mark = {"pass": "ok  ", "fail": "FAIL", "skip": "skip", "unavailable": "n/a "}[c["status"]]
        print(f"  {mark} {c['name']}: {c['reason'][:110]}")
    for r in rec["refusals"]:
        print(f"  ! {r}")

    if args.out and not args.quick:
        p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(f"  receipt -> {args.out}")

    if args.quick:
        return 0
    print(f"closes: {rec['closes']}")
    return 0 if rec["closes"] else 1


def _selfcheck() -> None:
    """Smallest thing that fails if the refusal logic breaks."""
    before = {"checks": [{"name": "blackbox", "status": "pass", "reason": ""}],
              "measured": {"loc_gated": 100, "topology": {"source_files": 10}}}
    now = {"checks": [{"name": "blackbox", "status": "skip", "reason": "no model"}],
           "measured": {"loc_gated": 90, "topology": {"source_files": 12}}}
    bad = refusals(now, before, {"eliminated": 10, "rewritten": 0, "generated": 0,
                                 "relocated": 0, "facade": 0})
    assert any("stopped running" in b for b in bad), bad
    assert any("topology regressed" in b for b in bad), bad
    bad2 = refusals({"checks": [], "measured": {"loc_gated": 90, "topology": {}}},
                    {"checks": [], "measured": {"loc_gated": 100, "topology": {}}},
                    {"eliminated": 5, "rewritten": 0, "generated": 0,
                     "relocated": 0, "facade": 0})
    assert any("do not reconcile" in b for b in bad2), bad2
    bad3 = refusals({"checks": [], "measured": {"loc_gated": 90, "topology": {}}},
                    {"checks": [], "measured": {"loc_gated": 100, "topology": {}}},
                    {"eliminated": 10, "rewritten": 0, "generated": 0,
                     "relocated": 0, "facade": 0})
    assert not bad3, bad3
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck(); raise SystemExit(0)
    raise SystemExit(main())
