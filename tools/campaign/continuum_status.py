#!/usr/bin/env python3.12
"""Continuum meta-controller: derive the campaign's state from live evidence.

The Continuum spans more phases than one agent context survives, so the authority on
"where are we" cannot be an agent's memory or a hand-written note.  It is this: read the
live filesystem, ledgers, launchd, and git, decide the state, and republish the four
files the campaign names.  Every field traces to something on disk.  Nothing is asserted.

    python3.12 tools/campaign/continuum_status.py          # republish
    python3.12 tools/campaign/continuum_status.py --json   # print only

Idempotent: a ledger row is appended only when the derived state actually changes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = Path.home() / "Library/Application Support/Hawking"
PASS3 = SUPPORT / "GLM52MathPrometheus/pass3"
ARTIFACT_DIR = (SUPPORT / "Models/GLM-5.2/b4734de4facf877f85769a911abafc5283eab3d9")
MATH_PRESERVE = ARTIFACT_DIR / "GLM-5.2-H0.98-Math-Preserve.gravity"
GENERAL = ARTIFACT_DIR / "General-R0"

STATUS_MD = ROOT / "HAWKING_CONTINUUM_STATUS.md"
STATUS_JSON = ROOT / "HAWKING_CONTINUUM_STATUS.json"
LEDGER = ROOT / "HAWKING_CONTINUUM_LEDGER.jsonl"
NEXT_COMMAND = ROOT / "HAWKING_CONTINUUM_NEXT_COMMAND.sh"

SHARDS_TOTAL = 282

# The Continuum's ordered programme.  `state` is whichever step is not yet complete;
# every earlier step's evidence is what proves it may be skipped.
STEPS = [
    ("OPTIMIZE_PASS3", "instrument PASS3 and retain only proven deterministic speedups"),
    ("FINISH_MATH_PRESERVE", "pack 282/282 Math-Preserve shards"),
    ("ASSEMBLE_MATH_PRESERVE", "assemble and verify the Math-Preserve artifact"),
    ("SEAL_CLAIM_A", "equal-budget Uniform/General/Math/RandomPolicy Claim A"),
    ("HAWKING_ODYSSEY_READY", "seal the pre-Odyssey checkpoint"),
    ("BASE_RUNTIME", "finish the base .gravity runtime and measured BASE_TRUE_TPS"),
    ("ACCELERATION", "generic speculative/parallel-token acceleration tournament"),
    ("HIDE_WIRED", "every claimed subsystem REAL_WIRED: model actually serving, kernel "
                   "turn default-on behind a closed approval round-trip, Context OS, "
                   "tools/effects, permissions, durable sessions, agents, fleets, "
                   "worktrees, verification"),
    ("ODYSSEY", "run Odyssey T0-T7"),
    ("MATH_FROZEN", "rerun Prometheus and pack Math-Frozen"),
    ("RECALIBRATE", "recalibrate acceleration and qualify HIDE"),
    ("FABRIC_BRIDGE", "Fabric, Bridge, adapters, schemas, canonical events, CLI"),
    ("CONSOLIDATE_HAWKING", "deep refactor: reclaim the hot foundry path to Rust/Metal "
                            "(the codec is half-ported -- Rust reads a .gravity, only "
                            "Python writes one), archive the one-shot campaign machinery, "
                            "one authority per subsystem. Speed and line count are "
                            "SEPARATE objectives with separate targets"),
    ("HAWKING_EVOLUTION_COMPLETE", "seal the evolution endpoint"),
    ("MIGRATE_RAMANUJAN", "create ~/Downloads/ramanujan and migrate owned contracts"),
    ("TRAIN_LOCAL_FORGE", "fully local retriever/formalizer/prover/repair training"),
    ("RAMANUJAN_COGNITION", "the discovery loop the consensus stack lacks: retina, "
                            "definitional library, obstruction objects, significance, "
                            "dual-channel fusion, analogy, competence map -- each with a "
                            "falsifier and a kill condition"),
    ("BUILD_SEARCH_GOVERNANCE", "search, roles, memories, Ledger, Tribunal, plus MOP's "
                                "refusal layer ported"),
    ("QUALIFY_SANDBOX", "Q0-Q6 and the multi-day pre-sandbox rehearsal"),
    ("RAMANUJAN_SANDBOX_READY", "terminal gate"),
]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _sh(*args: str) -> str:
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _launchd() -> dict[str, dict]:
    """Every loaded com.hawking.* job, with its PID and last exit status."""
    jobs: dict[str, dict] = {}
    for line in _sh("launchctl", "list").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) != 3 or not parts[2].startswith("com.hawking."):
            continue
        pid, status, label = parts
        jobs[label] = {"pid": None if pid == "-" else int(pid), "last_exit": int(status),
                       "running": pid != "-"}
    return jobs


def _pass3() -> dict:
    packed = sorted(MATH_PRESERVE.glob("*.gravity")) if MATH_PRESERVE.is_dir() else []
    progress = _read_json(PASS3 / "progress.json") or {}
    resident = ([p.name for p in (PASS3 / "source").glob("*.safetensors")]
                if (PASS3 / "source").is_dir() else [])
    telemetry = []
    path = PASS3 / "PASS3_TELEMETRY.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    telemetry.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    shard_rows = [r for r in telemetry if r.get("event") == "SHARD_TIMING"]
    # Rate the CURRENT scheduler only. Averaging across a scheduler change mixes two
    # different machines and produces an ETA that is true of neither.
    scheduler = shard_rows[-1].get("scheduler") if shard_rows else None
    recent = [r for r in shard_rows if r.get("scheduler") == scheduler][-8:]
    return {
        "shards_packed": len(packed),
        "shards_total": SHARDS_TOTAL,
        "percent": round(len(packed) * 100.0 / SHARDS_TOTAL, 2),
        "artifact_bytes": sum(p.stat().st_size for p in packed),
        "progress_file": progress,
        "resident_source_shards": resident,
        "scheduler": scheduler,
        "rate_sample_shards": len(recent),
        "live_pids": [int(p) for p in _sh("pgrep", "-f", "math_pass3_pack.py").split()],
        "mean_shard_seconds_recent": (
            round(sum(float(r["total_seconds"]) for r in recent) / len(recent), 1)
            if recent else None
        ),
        "free_disk_bytes": _free_bytes(),
    }


def _free_bytes() -> int:
    import shutil

    return shutil.disk_usage(str(SUPPORT if SUPPORT.exists() else ROOT)).free


def _delegated_lanes() -> list[dict]:
    """In-flight and finished delegations, so a successor does not re-run one.

    A delegation that produced no report is either still running or died; either way the
    next agent needs to know it exists before starting the same work again.
    """
    root = Path.home() / ".claude-grok/tasks"
    lanes = []
    for meta_path in sorted(root.glob("*/metadata.json")) if root.is_dir() else []:
        meta = _read_json(meta_path) or {}
        report = meta_path.parent / "grok-report.md"
        lanes.append({
            "task_id": meta.get("task_id"),
            "mode": meta.get("mode"),
            "profile": meta.get("profile"),
            "branch": meta.get("branch"),
            "worktree": meta.get("workdir"),
            "base_commit": meta.get("base_commit"),
            "finished": report.exists(),
            "reviewed_by_claude": bool(
                list(ROOT.glob(f"*{str(meta.get('task_id', '')).split('-2026')[0].upper()}*REVIEW*.json"))
            ),
        })
    return lanes


def _derive_state(pass3: dict, jobs: dict) -> tuple[str, str, list[str]]:
    """Return (state, why, blockers) from evidence alone."""
    blockers: list[str] = []
    receipt = ROOT / "GLM52_H0_98_MATH_PRESERVE_RECEIPT.json"
    audit = _read_json(ROOT / "HAWKING_ODYSSEY_READY_AUDIT.json") or {}

    if pass3["shards_packed"] < SHARDS_TOTAL:
        # launchd is not the only way PASS3 legitimately runs -- a bounded gate run is
        # started by hand.  Reporting "not running" at a live worker is the one error
        # that gets a healthy 30-hour job restarted from zero, so ask the process table.
        running = (any(j["running"] for label, j in jobs.items() if "pass3" in label)
                   or bool(pass3["live_pids"]))
        if not running:
            blockers.append(
                "PASS3 is not running: `launchctl bootstrap` the pass3 job or run "
                "`HAWKING_CONTINUUM_NEXT_COMMAND.sh`")
        return ("FINISH_MATH_PRESERVE",
                f"{pass3['shards_packed']}/{SHARDS_TOTAL} shards packed "
                f"({pass3['percent']}%)", blockers)
    if not receipt.exists():
        return ("ASSEMBLE_MATH_PRESERVE",
                "282/282 packed; finalization has not written the receipt yet", blockers)
    # The audit records its outcome in `verdict`, not a `sealed` flag. Reading the wrong key
    # made a sealed checkpoint report as unsealed, which is the exact failure this file
    # exists to prevent -- a resume authority that understates progress gets work redone.
    if audit.get("verdict") != "HAWKING_ODYSSEY_READY":
        return ("SEAL_CLAIM_A",
                "Math-Preserve receipt exists; Odyssey-ready audit not sealed", blockers)
    if not (ROOT / "GLM52_CLAIM_A.json").exists():
        return ("SEAL_CLAIM_A",
                "HAWKING_ODYSSEY_READY sealed; equal-budget Claim A not yet sealed", blockers)
    return ("BASE_RUNTIME", "Odyssey-ready and Claim A sealed", blockers)


def _next_command(state: str, jobs: dict) -> str:
    if state == "FINISH_MATH_PRESERVE":
        if any(j["running"] for label, j in jobs.items() if "pass3" in label) or _sh(
                "pgrep", "-f", "math_pass3_pack.py"):
            return (
                "# PASS3 is already running detached under launchd. Do not restart it.\n"
                "launchctl list | grep math-prometheus-pass3\n"
                "python3.12 tools/prometheus/math_pass3_pack.py status\n")
        return (
            "launchctl bootstrap gui/$(id -u) "
            "~/Library/LaunchAgents/com.hawking.glm52.math-prometheus-pass3.plist\n"
            "python3.12 tools/prometheus/math_pass3_pack.py status\n")
    if state == "ASSEMBLE_MATH_PRESERVE":
        return "python3.12 tools/prometheus/math_pass3_pack.py run\n"
    return "python3.12 tools/campaign/continuum_status.py\n"


def build() -> dict:
    jobs = _launchd()
    pass3 = _pass3()
    state, why, blockers = _derive_state(pass3, jobs)
    done = [name for name, _ in STEPS[:[n for n, _ in STEPS].index(state)]]
    remaining = [f"{name}: {desc}" for name, desc in STEPS
                 if name not in done and name != state]
    eta = None
    if pass3["mean_shard_seconds_recent"] and pass3["shards_packed"] < SHARDS_TOTAL:
        workers = int(os.environ.get("GLM52_PASS3_PACK_WORKERS", "4"))
        left = SHARDS_TOTAL - pass3["shards_packed"]
        eta = round(left * pass3["mean_shard_seconds_recent"] / workers / 3600.0, 2)
    return {
        "schema": "hawking.continuum.status.v1",
        "at": _now(),
        "state": state,
        "why": why,
        "blockers": blockers,
        "terminal_endpoint": "RAMANUJAN_SANDBOX_READY",
        "research_launch_fence": {
            "ODYSSEY_LAUNCH_AUTHORIZED": (ROOT / "odyssey/launch/ODYSSEY_LAUNCH_AUTHORIZED")
            .read_text().strip() if (ROOT / "odyssey/launch/ODYSSEY_LAUNCH_AUTHORIZED").exists()
            else "absent",
            "RAMANUJAN_RESEARCH_AUTHORIZED": "false",
        },
        "pass3": pass3,
        "estimated_pass3_hours_remaining": eta,
        "general_artifact_bytes": (
            sum(p.stat().st_size for p in GENERAL.rglob("*") if p.is_file())
            if GENERAL.is_dir() else 0),
        "launchd": jobs,
        "git": {
            "branch": _sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "head": _sh("git", "rev-parse", "HEAD"),
            "dirty": bool(_sh("git", "status", "--porcelain")),
        },
        "delegated_lanes": _delegated_lanes(),
        "completed_steps": done,
        "remaining_steps": remaining,
        "next_command_file": str(NEXT_COMMAND.relative_to(ROOT)),
    }


def publish(status: dict) -> None:
    previous = _read_json(STATUS_JSON) or {}
    STATUS_JSON.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    if previous.get("state") != status["state"]:
        with open(LEDGER, "a") as fh:
            fh.write(json.dumps({"at": status["at"], "event": "STATE",
                                 "from": previous.get("state"), "to": status["state"],
                                 "why": status["why"]}, sort_keys=True) + "\n")

    jobs = "\n".join(
        f"- `{label}` pid={j['pid']} last_exit={j['last_exit']}"
        for label, j in sorted(status["launchd"].items()))
    blockers = "\n".join(f"- {b}" for b in status["blockers"]) or "- none"
    remaining = "\n".join(f"- {s}" for s in status["remaining_steps"])
    eta = (f"{status['estimated_pass3_hours_remaining']} h at the current measured rate"
           if status["estimated_pass3_hours_remaining"] else "not estimable yet")
    # Deferred work is only real if a successor can find it. Anything queued rather than
    # done gets a receipt in the repo naming what is missing and why it was not fixed.
    queued = "\n".join(
        f"- `{p.name}` -- {_read_json(p).get('scope') or _read_json(p).get('question') or ''}"[:200]
        for p in sorted(ROOT.glob("*_GAPS.json")) + sorted(ROOT.glob("*_PRECLEARANCE.json"))
    ) or "- none recorded"
    lanes = "\n".join(
        f"- `{lane['task_id']}` {lane['mode']}/{lane['profile']} on `{lane['branch']}` "
        f"-- {'finished' if lane['finished'] else 'STILL RUNNING'}"
        for lane in status["delegated_lanes"]
    ) or "- none"
    STATUS_MD.write_text(f"""# HAWKING CONTINUUM STATUS

Generated from live evidence by `tools/campaign/continuum_status.py`. Do not hand-edit.

    at:       {status['at']}
    state:    {status['state']}
    why:      {status['why']}
    endpoint: {status['terminal_endpoint']} (not reached)

## Math-Preserve PASS3

    shards:     {status['pass3']['shards_packed']}/{status['pass3']['shards_total']} \
({status['pass3']['percent']}%)
    scheduler:  {status['pass3']['scheduler']}
    per shard:  {status['pass3']['mean_shard_seconds_recent']} s (recent mean, per worker)
    remaining:  {eta}
    artifact:   {status['pass3']['artifact_bytes']:,} bytes
    resident:   {len(status['pass3']['resident_source_shards'])} source shards
    free disk:  {status['pass3']['free_disk_bytes']:,} bytes

## Detached work

{jobs}

## Blockers

{blockers}

## Remaining programme

{remaining}

## Delegated lanes (do not restart one that is still running)

{lanes}

## Queued work with a written receipt

{queued}

## Next command

    bash {NEXT_COMMAND.name}
""")
    NEXT_COMMAND.write_text(
        "#!/bin/sh\n"
        "# Regenerated by tools/campaign/continuum_status.py -- the executable resume point.\n"
        f"# state: {status['state']}\n"
        f"cd {ROOT}\n" + _next_command(status["state"], status["launchd"]))
    NEXT_COMMAND.chmod(0o755)


def main() -> int:
    status = build()
    if "--json" in sys.argv:
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    publish(status)
    print(f"{status['state']}: {status['why']}")
    for blocker in status["blockers"]:
        print(f"  BLOCKER: {blocker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
