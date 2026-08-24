#!/usr/bin/env python3
"""PerformanceLedger (directive §25) — append-only, comparable, promotion-gating.

    "Promotions without comparable measurements are forbidden."

That sentence is the whole point of this module, so the gate is code, not custom.
`can_promote()` refuses unless the candidate and the incumbent were measured on
the same axes under a comparable environment, and it says exactly which axis is
missing when it refuses.

Rows live in receipts/headless/PERFORMANCE_LEDGER.jsonl. Append-only: a row is
never edited or deleted, because a ledger you can rewrite is not evidence.

Usage:
    python3 tools/headless/performance_ledger.py record --from-receipts
    python3 tools/headless/performance_ledger.py list
    python3 tools/headless/performance_ledger.py promote --candidate <id> --incumbent <id> \
        [--axis model|quantization|runtime|topology|source]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.path.expanduser("~/Downloads/hawking-copy"))
LEDGER = REPO / "receipts/headless/PERFORMANCE_LEDGER.jsonl"

# Directive §25. Split into what a promotion decision genuinely needs versus what
# is desirable context — refusing a promotion for a missing "tool_wait" would be
# theatre, refusing it for a missing decode rate would not.
REQUIRED_FOR_PROMOTION = [
    "model_identity",       # exact artifact this generation ran
    "runtime_build",        # exact runtime binary/version
    "source_sha",           # exact source state
    "context",              # context window
    "decode_tps",           # steady decode rate
    "complete_token_wall_ns",  # TOKEN_NS: the honest per-token cost
    "workers_resident",
    "workers_decoding",
    "measurement_reps",     # a single run is page-cache confounded
    "measurement_spread_pct",
]
DESIRABLE = [
    "bpw", "memory_bytes", "prefill_tps", "ttft_ms", "tpot_ms",
    "cache_hit_rate", "kv_restored", "tool_wait_s", "test_time_s",
    "compile_time_s", "validation_time_s", "accepted_workunits_per_hour",
    "mission_wall_s", "fallbacks",
]


def sh(c: str) -> str:
    return subprocess.run(["bash", "-lc", c], capture_output=True, text=True).stdout.strip()


def row_id(row: dict) -> str:
    key = json.dumps({k: row.get(k) for k in
                      ("generation", "model_identity", "runtime_build", "source_sha")},
                     sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load() -> list:
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def append(row: dict) -> str:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("recorded_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    rid = row_id(row)
    row["id"] = rid
    existing = {r.get("id") for r in load()}
    if rid in existing:
        print(f"  (row {rid} already present — ledger is append-only, not re-appending)")
        return rid
    with LEDGER.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return rid


def missing_axes(row: dict) -> list:
    return [k for k in REQUIRED_FOR_PROMOTION if row.get(k) in (None, "", [])]


# Which field is ALLOWED to differ, per kind of promotion. Everything else must
# be held equal or the comparison explains nothing.
#
# An earlier version of this gate hard-coded "runtime_build must match", which
# made a runtime promotion structurally impossible: it refused MLX-vs-llama.cpp
# because the runtime differed, which is the entire point of a runtime A/B. A
# gate that cannot pass the comparison it exists to judge is not strict, it is
# broken. The axis of change is declared, exempted, and recorded.
CHANGE_AXES = {
    "model":        ["model_identity", "model_bytes", "bpw", "quantization", "lineage"],
    "quantization": ["quantization", "model_identity", "model_bytes", "bpw"],
    "runtime":      ["runtime_build", "quantization", "model_identity", "model_bytes", "bpw"],
    "topology":     ["workers_resident", "workers_decoding", "topology"],
    "source":       ["source_sha"],
}
HELD_EQUAL = ["runtime_build", "context", "workers_resident", "workers_decoding",
              "model_identity", "source_sha"]


def comparable(a: dict, b: dict, axis: str = "model") -> tuple[bool, list]:
    """Two rows are comparable when everything except the declared axis of change
    is held equal. `axis` names what the promotion is actually changing."""
    if axis not in CHANGE_AXES:
        return False, [f"unknown change axis {axis!r}; expected one of {sorted(CHANGE_AXES)}"]
    exempt = set(CHANGE_AXES[axis])
    problems = []
    for field in HELD_EQUAL:
        if field in exempt:
            continue
        if a.get(field) != b.get(field):
            detail = ""
            if field in ("workers_resident", "workers_decoding"):
                detail = " — throughput is not comparable across different concurrency"
            problems.append(f"{field} differs: {a.get(field)!r} vs {b.get(field)!r}{detail} "
                            f"(not exempt under axis={axis})")
    return (not problems), problems


def can_promote(cand: dict, inc: dict, axis: str = "model") -> dict:
    """The §25 gate. Returns a decision with reasons, never a bare boolean.

    `axis` declares what this promotion changes; everything else must be held
    equal. Declaring a wide axis to dodge a refusal is possible and is exactly
    the kind of thing a reviewer should look for — the axis is recorded in the
    decision so it is visible."""
    reasons = []
    m_c, m_i = missing_axes(cand), missing_axes(inc)
    if m_c:
        reasons.append(f"candidate is missing required measurements: {', '.join(m_c)}")
    if m_i:
        reasons.append(f"incumbent is missing required measurements: {', '.join(m_i)}")
    ok, problems = comparable(cand, inc, axis)
    if not ok:
        reasons.extend(problems)

    if reasons:
        return {"allowed": False, "change_axis": axis, "reasons": reasons,
                "rule": "directive §25 — promotions without comparable measurements are forbidden"}

    # Only now is a numeric comparison meaningful.
    spread = max(cand.get("measurement_spread_pct") or 0, inc.get("measurement_spread_pct") or 0)
    gain = 100 * (cand["decode_tps"] - inc["decode_tps"]) / inc["decode_tps"]
    token_ns_gain = (100 * (inc["complete_token_wall_ns"] - cand["complete_token_wall_ns"])
                     / inc["complete_token_wall_ns"])
    within_noise = abs(gain) <= spread
    return {
        "allowed": True,
        "change_axis": axis,
        "held_equal": [f for f in HELD_EQUAL if f not in set(CHANGE_AXES[axis])],
        "decode_tps_gain_pct": round(gain, 2),
        "complete_token_ns_gain_pct": round(token_ns_gain, 2),
        "measurement_spread_pct": spread,
        "gain_within_noise": within_noise,
        "verdict": ("REJECT — gain is inside the measurement spread, which is noise, not a win"
                    if within_noise and gain > 0 else
                    "REJECT — candidate is slower" if gain < 0 else
                    "ELIGIBLE — gain exceeds measurement noise; capability gates still apply"),
        "note": ("Eligibility here is PERFORMANCE only. Directive §13 still requires Doctor and "
                 "Tabula capability verdicts before a child becomes the parent, and §10 forbids a "
                 "child promoting itself."),
    }


def harvest() -> list:
    """Seed the ledger from receipts already on disk, so the first rows are real
    measurements rather than placeholders."""
    rows = []
    src_sha = sh(f"git -C {REPO} rev-parse HEAD")
    llama_build = sh("llama-server --version 2>&1 | head -1")

    g = REPO / "receipts/headless/MACHINE_GENOME.json"
    if g.exists():
        d = json.loads(g.read_text())
        ri = d.get("runtime_identity", {})
        rows.append({
            "generation": "bootstrap-llamacpp-q5k",
            "measured_by": "tools/headless/machine_probe.py",
            "model_identity": ri.get("model_path"),
            "model_bytes": ri.get("model_size_bytes"),
            "runtime_build": ri.get("llama_version") or llama_build,
            "source_sha": src_sha,
            "context": ri.get("ctx"),
            "decode_tps": d.get("single_decoder_tps"),
            "complete_token_wall_ns": (int(1e9 / d["single_decoder_tps"])
                                       if d.get("single_decoder_tps") else None),
            "workers_resident": 1,
            "workers_decoding": 1,
            "measurement_reps": (d.get("concurrency_curve") or [{}])[0].get("reps") and
                                len((d.get("concurrency_curve") or [{}])[0].get("reps", [])),
            "measurement_spread_pct": (d.get("concurrency_curve") or [{}])[0].get("spread_pct"),
            "resident_runtime_limit": d.get("RESIDENT_RUNTIME_LIMIT"),
            "active_decode_limit": d.get("ACTIVE_DECODE_LIMIT"),
            "best_aggregate_tps": d.get("best_aggregate_tps"),
        })

    t = REPO / "receipts/headless/DECODE_TOPOLOGY.json"
    if t.exists():
        d = json.loads(t.read_text())
        for arm in ("process", "slot"):
            s = (d.get("summary") or {}).get(arm, {})
            if not s:
                continue
            best_k = max(s, key=lambda k: s[k].get("aggregate_tps_median") or 0)
            rows.append({
                "generation": f"bootstrap-llamacpp-q5k-{arm}-topology-k{best_k}",
                "measured_by": "tools/headless/decode_topology_probe.py",
                "model_identity": d.get("model"),
                "model_bytes": d.get("model_bytes"),
                "runtime_build": d.get("llama_version") or llama_build,
                "source_sha": src_sha,
                "context": (d.get("params") or {}).get("per_slot_ctx"),
                "decode_tps": s[best_k].get("aggregate_tps_median"),
                "complete_token_wall_ns": (int(1e9 / s[best_k]["aggregate_tps_median"])
                                           if s[best_k].get("aggregate_tps_median") else None),
                "workers_resident": int(best_k) if arm == "process" else 1,
                "workers_decoding": int(best_k),
                "measurement_reps": (d.get("params") or {}).get("reps"),
                "measurement_spread_pct": s[best_k].get("spread_pct"),
                "topology": arm,
                "scaling_vs_1": s[best_k].get("scaling_vs_1"),
            })

    ab = REPO / "receipts/headless/RUNTIME_AB.json"
    if ab.exists():
        d = json.loads(ab.read_text())
        for name, arm in (d.get("arms") or {}).items():
            if not arm.get("decode_tps_median"):
                continue
            rows.append({
                "generation": f"runtime-ab-{name}",
                "measured_by": "tools/headless/runtime_ab.py",
                "model_identity": arm.get("model"),
                "model_bytes": arm.get("bytes"),
                "quantization": arm.get("quant"),
                "lineage": arm.get("lineage"),
                "runtime_build": arm.get("version") or name,
                "source_sha": src_sha,
                "context": (d.get("params") or {}).get("ctx"),
                "decode_tps": arm.get("decode_tps_median"),
                "complete_token_wall_ns": int(1e9 / arm["decode_tps_median"]),
                "workers_resident": 1,
                "workers_decoding": 1,
                "measurement_reps": (d.get("params") or {}).get("reps"),
                "measurement_spread_pct": arm.get("spread_pct"),
                "confound_declared": d.get("confound_declared"),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--from-receipts", action="store_true")
    r.add_argument("--json", help="a JSON object to append as one row")
    sub.add_parser("list")
    p = sub.add_parser("promote")
    p.add_argument("--candidate", required=True)
    p.add_argument("--incumbent", required=True)
    p.add_argument("--axis", default="model", choices=sorted(CHANGE_AXES),
                   help="what this promotion changes; everything else must be held equal")
    args = ap.parse_args()

    if args.cmd == "record":
        rows = harvest() if args.from_receipts else []
        if args.json:
            rows.append(json.loads(args.json))
        if not rows:
            print("nothing to record (no source receipts found yet)")
            return 0
        for row in rows:
            rid = append(row)
            miss = missing_axes(row)
            print(f"  {rid}  {row['generation']:<44} decode={row.get('decode_tps')}"
                  + (f"  MISSING: {','.join(miss)}" if miss else "  (promotion-complete)"))
        return 0

    if args.cmd == "list":
        rows = load()
        if not rows:
            print("ledger empty")
            return 0
        print(f"{'id':<18}{'generation':<46}{'decode_tps':>11}{'token_ns':>12}  missing")
        for row in rows:
            miss = missing_axes(row)
            print(f"{row['id']:<18}{row['generation'][:45]:<46}"
                  f"{str(row.get('decode_tps')):>11}{str(row.get('complete_token_wall_ns')):>12}"
                  f"  {','.join(miss) if miss else '-'}")
        return 0

    rows = {r["id"]: r for r in load()}
    for k in (args.candidate, args.incumbent):
        if k not in rows:
            print(f"no ledger row with id {k}", file=sys.stderr)
            return 2
    d = can_promote(rows[args.candidate], rows[args.incumbent], args.axis)
    print(json.dumps(d, indent=1))
    return 0 if d.get("allowed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
