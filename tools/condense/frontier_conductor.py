#!/usr/bin/env python3.12
"""Detached conductor for the 7B condensation frontier.

This process does not kill work. It observes the JSONL, promotes good results
into a ledger, plants autopilot injects when useful, and asks the launcher to
adopt missing supervisors. The goal is a deterministic outer loop that keeps
the frontier moving without human timing.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT = runpy.run_path(str(ROOT / "tools/condense/frontier_autopilot.py"))


BASE_FOR = {
    "mp-4a3f": "mp-4a3f",
    "3-AWQ": "3-AWQ",
    "4-AWQ": "4-AWQ",
}


def read_records(outbase: str) -> dict[str, dict]:
    records = {}
    path = ROOT / f"{outbase}.jsonl"
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


def num(rec: dict, key: str) -> float | None:
    try:
        return float(rec[key])
    except Exception:
        return None


def parent_for(config: str) -> str | None:
    if config in BASE_FOR.values():
        return None
    for prefix, base in BASE_FOR.items():
        if config.startswith(prefix + "+") or config.startswith(prefix + "-"):
            return base
    if config.startswith("mp-4a3f"):
        return "mp-4a3f"
    if config.startswith("3-AWQ"):
        return "3-AWQ"
    if config.startswith("4-AWQ"):
        return "4-AWQ"
    return None


def pareto(records: dict[str, dict]) -> list[dict]:
    rows = [r for r in records.values() if "ppl" in r and "eff_bpw" in r and "degr_pct" in r]
    out = []
    for r in rows:
        rb, rd = num(r, "eff_bpw"), num(r, "degr_pct")
        if rb is None or rd is None:
            continue
        dominated = False
        for o in rows:
            if o is r:
                continue
            ob, od = num(o, "eff_bpw"), num(o, "degr_pct")
            if ob is None or od is None:
                continue
            if ob <= rb and od <= rd and (ob < rb or od < rd):
                dominated = True
                break
        if not dominated:
            out.append(r)
    return sorted(out, key=lambda x: (num(x, "eff_bpw") or 99, num(x, "degr_pct") or 999999))


def classify(records: dict[str, dict], rec: dict, min_gain: float) -> dict:
    cfg = rec.get("config", "")
    base = parent_for(cfg)
    degr = num(rec, "degr_pct")
    bpw = num(rec, "eff_bpw")
    base_degr = num(records.get(base, {}), "degr_pct") if base else None
    gain = base_degr - degr if base_degr is not None and degr is not None else None
    if "error" in rec:
        verdict = "error"
    elif gain is None:
        verdict = "baseline"
    elif gain >= max(1.0, min_gain * 4):
        verdict = "excellent"
    elif gain >= min_gain:
        verdict = "good"
    elif gain > 0:
        verdict = "small"
    else:
        verdict = "bad"
    return {
        "config": cfg,
        "parent": base,
        "eff_bpw": bpw,
        "degr_pct": degr,
        "parent_degr_pct": base_degr,
        "gain_pct_points": gain,
        "verdict": verdict,
    }


def load_seen(path: Path) -> set[str]:
    seen = set()
    if not path.exists():
        return seen
    for line in path.read_text(errors="ignore").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        key = r.get("promotion_key")
        if key:
            seen.add(key)
    return seen


def append_promotion(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def write_markdown(path: Path, promotions: list[dict], frontier: list[dict], state: dict):
    lines = [
        "# 7B Frontier Promotions",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Branch: `{state.get('branch')}`",
        f"Pending candidates: `{state.get('candidate_count')}`",
        "",
        "## Pareto Frontier",
        "",
        "| config | bpw | degr |",
        "|---|--:|--:|",
    ]
    for r in frontier:
        lines.append(f"| {r.get('config')} | {num(r, 'eff_bpw'):.3f} | {num(r, 'degr_pct'):.2f}% |")
    doctor_promotions = [p for p in promotions if str(p.get("promotion_key", "")).startswith("doctor_gain:")]
    lines += ["", "## Promoted Doctor Results", "", "| config | verdict | gain | degr | bpw |", "|---|---|--:|--:|--:|"]
    for r in doctor_promotions[-20:]:
        lines.append(
            f"| {r.get('config')} | {r.get('verdict')} | "
            f"{r.get('gain_pct_points') if r.get('gain_pct_points') is not None else ''} | "
            f"{r.get('degr_pct') if r.get('degr_pct') is not None else ''} | "
            f"{r.get('eff_bpw') if r.get('eff_bpw') is not None else ''} |"
        )
    path.write_text("\n".join(lines) + "\n")


def live_pid(path: Path) -> int | None:
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)
        return pid
    except Exception:
        return None


def adopt_if_needed(outbase: str, log):
    base = ROOT / outbase
    missing = []
    for suffix in (".lock", "_health.pid", "_caffeinate.pid", "_keepalive.pid"):
        if live_pid(Path(str(base) + suffix)) is None:
            missing.append(suffix)
    if missing:
        log(f"adopt: missing {','.join(missing)}; invoking launcher")
        subprocess.run(["./run_7b_frontier.sh", "launch"], cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def plant_inject_if_needed(outbase: str, records: dict[str, dict], log) -> tuple[int, list[str]]:
    inject = ROOT / f"{outbase}_inject.py"
    if inject.exists():
        return 0, ["inject already present"]
    candidates, skipped = AUTOPILOT["plan"](records, set(records), max_new=int(os.environ.get("AUTOPILOT_MAX_NEW", "8")))
    if candidates:
        AUTOPILOT["_write_rearm"](str(ROOT / outbase))
        log(f"inject: planted autopilot with {len(candidates)} candidate(s)")
        return len(candidates), skipped
    return 0, skipped


def branch_from_promotions(promotions: list[dict], candidate_count: int) -> str:
    doctor = [p for p in promotions if str(p.get("promotion_key", "")).startswith("doctor_gain:")]
    verdicts = [p.get("verdict") for p in doctor]
    if any(v in {"excellent", "good"} for v in verdicts[-4:]):
        return "good-results-expand"
    if any(v == "small" for v in verdicts[-4:]) or candidate_count:
        return "mid-results-probe"
    if any(v == "bad" for v in verdicts[-4:]):
        return "bad-results-prune"
    return "waiting-for-v3"


def step(outbase: str, log) -> dict:
    records = read_records(outbase)
    min_gain = float(os.environ.get("AUTOPILOT_MIN_GAIN_PCT", "0.25"))
    frontier = pareto(records)
    ledger = ROOT / f"{outbase}_promotions.jsonl"
    seen = load_seen(ledger)
    promotions = []
    if ledger.exists():
        for line in ledger.read_text(errors="ignore").splitlines():
            try:
                promotions.append(json.loads(line))
            except Exception:
                pass

    for rec in records.values():
        cfg = rec.get("config", "")
        if "+dr" not in cfg or "ppl" not in rec:
            continue
        c = classify(records, rec, min_gain)
        if c["verdict"] not in {"good", "excellent"}:
            continue
        key = f"doctor_gain:{cfg}"
        if key in seen:
            continue
        c["promotion_key"] = key
        c["ts"] = time.time()
        append_promotion(ledger, c)
        promotions.append(c)
        seen.add(key)
        log(f"promote: {cfg} {c['verdict']} gain={c.get('gain_pct_points')}")

    for rec in frontier:
        cfg = rec.get("config", "")
        key = f"pareto:{cfg}"
        if key in seen:
            continue
        c = classify(records, rec, min_gain)
        c["promotion_key"] = key
        c["promotion_type"] = "pareto"
        c["ts"] = time.time()
        append_promotion(ledger, c)
        promotions.append(c)
        seen.add(key)

    candidate_count, skipped = plant_inject_if_needed(outbase, records, log)
    state = {
        "ts": time.time(),
        "records": len(records),
        "pareto": [r.get("config") for r in frontier],
        "candidate_count": candidate_count,
        "branch": branch_from_promotions(promotions, candidate_count),
        "skipped": skipped[:30],
    }
    state_path = ROOT / f"{outbase}_conductor_state.json"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    write_markdown(ROOT / f"{outbase}_promotions.md", promotions, frontier, state)
    return state


def run_loop(outbase: str, interval: int):
    pid_path = ROOT / f"{outbase}_conductor.pid"
    log_path = ROOT / f"{outbase}_conductor.log"
    pid_path.write_text(str(os.getpid()) + "\n")

    def log(msg: str):
        with log_path.open("a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    log(f"conductor start interval={interval}s")
    try:
        while True:
            adopt_if_needed(outbase, log)
            state = step(outbase, log)
            log(f"state branch={state['branch']} records={state['records']} candidates={state['candidate_count']}")
            time.sleep(interval)
    finally:
        try:
            if pid_path.read_text().strip() == str(os.getpid()):
                pid_path.unlink()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outbase", default="reports/cron/7b_frontier")
    ap.add_argument("--interval", type=int, default=int(os.environ.get("CONDUCTOR_INTERVAL", "180")))
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    if args.once:
        state = step(args.outbase, print)
        print(json.dumps(state, indent=2))
        return 0
    run_loop(args.outbase, args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
