#!/usr/bin/env python3.12
"""Deterministic condensation autopilot for the 7B frontier run.

The ladder executes `{OUTP}_inject.py` between configs. This module can be used
from that inject to append the next justified configs from the current JSONL, and
it can be run by the keepalive supervisor to plant the inject after a restart.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


RUNG_4A3F = {
    "q_proj": 4, "k_proj": 4, "v_proj": 4, "o_proj": 4,
    "gate_proj": 3, "up_proj": 3, "down_proj": 3,
}


def _read_records(outbase: str) -> dict[str, dict]:
    records = {}
    path = Path(outbase + ".jsonl")
    if not path.exists():
        return records
    for line in path.read_text(errors="ignore").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        cfg = rec.get("config")
        if cfg:
            records[cfg] = rec
    return records


def _is_done(rec: dict | None) -> bool:
    return bool(rec and ("ppl" in rec or "error" in rec))


def _degr(records: dict[str, dict], name: str) -> float | None:
    rec = records.get(name)
    if not rec or "degr_pct" not in rec:
        return None
    try:
        return float(rec["degr_pct"])
    except Exception:
        return None


def _gain(records: dict[str, dict], base: str, doctor: str) -> float | None:
    b = _degr(records, base)
    d = _degr(records, doctor)
    if b is None or d is None:
        return None
    return b - d


def _candidate(name: str, args: tuple, reason: str) -> dict:
    return {"name": name, "builder": "build_recover", "args": args, "reason": reason}


def plan(records: dict[str, dict], known: set[str], max_new: int = 8) -> tuple[list[dict], list[str]]:
    """Return deterministic next configs and human-readable skipped reasons."""
    skipped = []
    out = []

    def add(c):
        if c["name"] in known or any(x["name"] == c["name"] for x in out):
            skipped.append(f"{c['name']}: already known")
            return
        out.append(c)

    # Always ensure the hardened seed set exists.
    seeds = [
        _candidate("mp-4a3f+dr-r16-v3", (3, 90, 16, 1e-4, 0.5, RUNG_4A3F),
                   "seed: best sub-4bpw point, low adapter overhead"),
        _candidate("mp-4a3f+dr-r32-v3", (3, 90, 32, 1e-4, 0.5, RUNG_4A3F),
                   "seed: rank ladder for best sub-4bpw point"),
        _candidate("3-AWQ+dr-r16-v3", (3, 90, 16, 1e-4, 0.5),
                   "seed: dense 3-bit point"),
        _candidate("4-AWQ+dr-r16-v3", (4, 90, 16, 1e-4, 0.5),
                   "seed: quality anchor"),
    ]
    for c in seeds:
        add(c)

    min_gain = float(os.environ.get("AUTOPILOT_MIN_GAIN_PCT", "0.25"))
    min_rank_gain = float(os.environ.get("AUTOPILOT_MIN_RANK_GAIN_PCT", "0.15"))

    def if_helped(base: str, doc: str) -> bool:
        g = _gain(records, base, doc)
        if g is None:
            skipped.append(f"{doc}: no completed result yet")
            return False
        if g < min_gain:
            skipped.append(f"{doc}: gain {g:.2f} < {min_gain:.2f}")
            return False
        return True

    # Capacity ladders: only spend more rank/steps if the smaller adapter moved PPL.
    if if_helped("mp-4a3f", "mp-4a3f+dr-r16-v3"):
        add(_candidate("mp-4a3f-a40+dr-r16-v3", (3, 90, 16, 1e-4, 0.4, RUNG_4A3F),
                       "alpha local search around successful mp-4a3f basin"))
        add(_candidate("mp-4a3f-a60+dr-r16-v3", (3, 90, 16, 1e-4, 0.6, RUNG_4A3F),
                       "alpha local search around successful mp-4a3f basin"))
        add(_candidate("mp-4a3f-o2+dr-r16-v3", (3, 90, 16, 1e-4, 0.5, RUNG_4A3F, 2.0),
                       "sparse-outlier sweep after doctor gain"))
        if _is_done(records.get("mp-4a3f+dr-r32-v3")):
            d16 = _degr(records, "mp-4a3f+dr-r16-v3")
            d32 = _degr(records, "mp-4a3f+dr-r32-v3")
            if d16 is not None and d32 is not None and d16 - d32 >= min_rank_gain:
                add(_candidate("mp-4a3f+dr-r64-v3", (3, 90, 64, 1e-4, 0.5, RUNG_4A3F),
                               "rank still pays on mp-4a3f"))
                add(_candidate("mp-4a3f+dr-r32-s180-v3", (3, 180, 32, 7e-5, 0.5, RUNG_4A3F),
                               "longer lower-lr doctor after rank gain"))
            else:
                skipped.append("mp-4a3f rank32: diminishing rank return")

    if if_helped("3-AWQ", "3-AWQ+dr-r16-v3"):
        add(_candidate("3-AWQ+dr-r32-v3", (3, 90, 32, 1e-4, 0.5),
                       "rank ladder for dense 3-bit"))
        add(_candidate("3-AWQ-a40+dr-r16-v3", (3, 90, 16, 1e-4, 0.4),
                       "alpha local search around dense 3-bit"))
        add(_candidate("3-AWQ-a60+dr-r16-v3", (3, 90, 16, 1e-4, 0.6),
                       "alpha local search around dense 3-bit"))
        add(_candidate("3-AWQ-o2+dr-r16-v3", (3, 90, 16, 1e-4, 0.5, None, 2.0),
                       "sparse-outlier sweep for dense 3-bit"))
        if _is_done(records.get("3-AWQ+dr-r32-v3")):
            d16 = _degr(records, "3-AWQ+dr-r16-v3")
            d32 = _degr(records, "3-AWQ+dr-r32-v3")
            if d16 is not None and d32 is not None and d16 - d32 >= min_rank_gain:
                add(_candidate("3-AWQ+dr-r64-v3", (3, 90, 64, 1e-4, 0.5),
                               "rank still pays on dense 3-bit"))

    if if_helped("4-AWQ", "4-AWQ+dr-r16-v3"):
        add(_candidate("4-AWQ+dr-r32-v3", (4, 90, 32, 1e-4, 0.5),
                       "quality-anchor rank check"))

    # If a doctor makes things worse, try the same tiny adapter with a gentler optimizer once.
    for base, doc, args in [
        ("mp-4a3f", "mp-4a3f+dr-r16-v3", (3, 90, 16, 5e-5, 0.5, RUNG_4A3F)),
        ("3-AWQ", "3-AWQ+dr-r16-v3", (3, 90, 16, 5e-5, 0.5)),
    ]:
        g = _gain(records, base, doc)
        if g is not None and g < 0:
            add(_candidate(doc.replace("-v3", "-lr5e5-v3"), args,
                           "doctor hurt quality; retry once with lower lr"))

    return out[:max_new], skipped


def _known_from_ns(ns: dict, records: dict[str, dict]) -> set[str]:
    known = set(records)
    configs = ns.get("CONFIGS", {}).get(ns.get("SETNAME", "frontier"), [])
    known.update(c[0] for c in configs if c)
    return known


def _write_rearm(outbase: str):
    ipath = Path(outbase + "_inject.py")
    ipath.write_text(
        "import runpy\n"
        "_m = runpy.run_path('tools/condense/frontier_autopilot.py')\n"
        "_m['activate'](globals())\n"
    )


def activate(ns: dict, rearm: bool = True) -> list[dict]:
    outbase = str(ns["OUTP"])
    records = _read_records(outbase)
    known = _known_from_ns(ns, records)
    max_new = int(os.environ.get("AUTOPILOT_MAX_NEW", "8"))
    candidates, skipped = plan(records, known, max_new=max_new)

    os.environ.setdefault("DOCTOR_TIMEOUT", "28800")
    os.environ.setdefault("DOCTOR_SWAP_CEIL", "12000")
    os.environ.setdefault("DOCTOR_SWAP_HARD_CEIL", "18000")
    os.environ.setdefault("DOCTOR_TERMINATE_GRACE", "600")
    os.environ.setdefault("DOCTOR_USE_PARTIAL", "1")
    os.environ.setdefault("DOCTOR_SAVE_MODE", "adapter")
    os.environ.setdefault("KD_TOPK", "64")

    configs = ns["CONFIGS"][ns["SETNAME"]]
    builder = ns["build_recover"]
    queued = []
    for c in candidates:
        configs.append((c["name"], builder, tuple(c["args"])))
        queued.append(c)
        ns["log"](f"# AUTOPILOT: queued {c['name']} — {c['reason']}")

    pending = [c[0] for c in configs if c and not _is_done(records.get(c[0]))]
    state = {
        "records": len(records),
        "queued": queued,
        "pending_count": len(pending),
        "pending_head": pending[:12],
        "skipped": skipped[:40],
    }
    Path(outbase + "_autopilot_state.json").write_text(json.dumps(state, indent=2) + "\n")

    if rearm and (queued or pending):
        _write_rearm(outbase)
        ns["log"]("# AUTOPILOT: rearmed for next config boundary")
    elif rearm:
        ns["log"]("# AUTOPILOT: no pending work and no new candidates")
    return queued


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outbase", default="reports/cron/7b_frontier")
    ap.add_argument("--emit-inject", action="store_true")
    ap.add_argument("--max-new", type=int, default=int(os.environ.get("AUTOPILOT_MAX_NEW", "8")))
    args = ap.parse_args()

    records = _read_records(args.outbase)
    known = set(records)
    candidates, skipped = plan(records, known, max_new=args.max_new)
    state = {
        "records": len(records),
        "candidates": candidates,
        "skipped": skipped[:40],
    }
    Path(args.outbase + "_autopilot_state.json").write_text(json.dumps(state, indent=2) + "\n")
    if args.emit_inject and candidates:
        _write_rearm(args.outbase)
        print(f"emitted {args.outbase}_inject.py with {len(candidates)} candidate(s)")
        return 10
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
