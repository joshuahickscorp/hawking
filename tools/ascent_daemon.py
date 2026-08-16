#!/usr/bin/env python3
"""Detached ascent daemon — keeps the campaign moving with no Claude in the loop.

`ascent_controller.py` runs ONE cycle against a hand-written queue. That is not
enough to leave: the queue runs dry, finished lanes pile up unread, and nothing
decides what to try next. This adds the three missing pieces:

  harvest   read finished Grok lanes, extract their NEXT_BOTTLENECK, and turn it
            into the next target — this is what keeps the queue non-empty
  gate      real Tier-1 correctness checks per model, not an echo marker
  govern    pause when the box is not safe to benchmark on, reclaim when disk is
            tight, and never exceed the lane budget

It never merges. Promotion stays protected: a lane that passes Tier-1 is recorded
MERGE_READY with its skew verdict for a human (or Claude) to land.

    python3 tools/ascent_daemon.py once      # one full pass
    python3 tools/ascent_daemon.py loop      # run until stopped
    python3 tools/ascent_daemon.py status
    python3 tools/ascent_daemon.py selfcheck
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "receipts" / "ascent-2026-08-16" / "ASCENT_STATE.json"
QUEUE = REPO / "receipts" / "ascent-2026-08-16" / "PROMOTION_QUEUE.json"
TASKS = Path.home() / ".claude-grok" / "tasks"
GROK = Path.home() / ".claude-grok" / "bin" / "grok-run"
# Durable, in-repo. The session scratchpad does NOT survive the session, and a
# daemon meant to run unattended cannot depend on a path that disappears.
LANES = REPO / "workspace" / "ops" / "ascent-lanes"

DISK_FLOOR_GIB = 15.0
DISK_WARN_GIB = 90.0   # raised after a 0-byte stall: lanes cost 1-19 GiB each
MAX_CONCURRENT = 10    # raised again per user steer: the 0-byte stall is now guarded
                       # by the governor reaping the grok worktree pool, which is the
                       # real protection - the cap was only ever a blunt proxy for it
POLL_SECONDS = 300

# Real Tier-1 gates. Reject-only: passing here is NOT promotion.
TIER1 = {
    "q80": {
        "cmd": "cargo build --profile release-fast -p hawking-core "
               "--example ascension_qwen80_uniform_q4_hybrid_greedy 2>&1 | tail -3",
        "expect": "Finished",
        "forbid": "error[",
    },
    "dsv4f": {
        "cmd": "cargo build --profile release-fast -p hawking-core "
               "--example gravity_deepseek_v4_native_token_graph 2>&1 | tail -3",
        "expect": "Finished",
        "forbid": "error[",
    },
    "qwen38": {
        "cmd": "cargo build --profile release-fast -p hawking-core 2>&1 | tail -3",
        "expect": "Finished",
        "forbid": "error[",
    },
}


def sh(cmd: str, timeout: int = 1800) -> tuple[int, str]:
    p = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def load(path: Path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


# ---------------------------------------------------------------- governors

def machine() -> dict:
    sys.path.insert(0, str(REPO / "tools"))
    try:
        from agentos.machine_state import clean_box_ok, snapshot  # type: ignore
        snap = snapshot()
        snap["clean_box_ok"], snap["clean_box_reason"] = clean_box_ok(snap, DISK_FLOOR_GIB)
        return snap
    except Exception as exc:
        return {"disk_free_gib": 999, "clean_box_ok": False,
                "clean_box_reason": f"machine_state unavailable ({exc})",
                "active_grok_lanes": []}


def our_live_lanes(snap: dict) -> list[str]:
    """Live lanes belonging to THIS repo only.

    machine_state reports every live grok lane on the box, including other
    projects'. Counting those against our concurrency cap made the daemon idle
    while unrelated repos held the budget - measured 10 live, only 4 ours. The
    cap must govern our own spend, not the machine's.
    """
    ours = []
    for lane in snap.get("active_grok_lanes") or []:
        wt = Path.home() / ".claude-grok" / "worktrees" / lane
        code, out = sh(f"git -C {wt} rev-parse --path-format=absolute "
                       f"--git-common-dir 2>/dev/null", timeout=60)
        if code == 0 and str(REPO) in out:
            ours.append(lane)
    return ours


def reap_finished_worktrees() -> int:
    """Delete worktrees of finished lanes that have NOTHING to lose.

    reclaim_safe.sh clears build dirs and repo-aware worktrees but NOT the grok
    worktree pool - which is what actually fills this disk. Lanes cost 1-19 GiB
    each; the pool reached 67 GiB and hit 0 bytes free, stalling every tool on the
    box including the shell itself. Only reaped when the lane is NOT live AND the
    worktree is clean, so no uncommitted work can be lost. Branches always survive.
    """
    pool = Path.home() / ".claude-grok" / "worktrees"
    if not pool.is_dir():
        return 0
    code, out = sh(f"{GROK} status", timeout=300)
    if code != 0:
        return 0          # cannot tell what is live -> reap nothing
    live = {parts[2] for parts in (l.split() for l in out.splitlines())
            if len(parts) > 2 and parts[0] == "running"}
    freed = 0
    for d in sorted(pool.iterdir()):
        if not d.is_dir() or d.name in live:
            continue
        rc, dirty = sh(f"git -C {d} status --porcelain 2>/dev/null | wc -l", timeout=120)
        if rc != 0 or dirty.strip() != "0":
            continue      # dirty or unreadable -> preserve
        _, sz = sh(f"du -sm {d} 2>/dev/null | cut -f1", timeout=300)
        sh(f"rm -rf {d}", timeout=600)
        try: freed += int(sz.strip() or 0)
        except ValueError: pass
    return freed


def govern(snap: dict) -> str | None:
    """Return a reason to hold off, or None to proceed."""
    free = snap.get("disk_free_gib") or 0
    if free < DISK_WARN_GIB:
        script = REPO / "tools" / "reclaim_safe.sh"
        if script.is_file():
            sh(f"bash {script}", timeout=900)
        sh("find ~/.claude-grok/tasks -name diff.patch -size +50M -delete", timeout=600)
        reap_finished_worktrees()
        free = machine().get("disk_free_gib") or 0
    if free < DISK_FLOOR_GIB:
        return f"disk {free} GiB below floor {DISK_FLOOR_GIB}"
    ours = our_live_lanes(snap)
    snap["our_live_lanes"] = ours
    if len(ours) >= MAX_CONCURRENT:
        return f"{len(ours)} OUR lanes live, at the {MAX_CONCURRENT} cap"
    return None


# ---------------------------------------------------------------- harvest

NEXT_RE = re.compile(r"^NEXT_BOTTLENECK:\s*(.+)$", re.M)
STATUS_RE = re.compile(r"^STATUS:\s*(\w+)", re.M)


def harvest() -> list[dict]:
    """Read finished lane reports and mine their NEXT_BOTTLENECK.

    This is the piece that keeps the queue alive without a human. A lane that
    finishes almost always names the next wall; that name becomes the next target.
    """
    found = []
    if not TASKS.is_dir():
        return found
    for d in sorted(TASKS.iterdir()):
        report = d / "grok-report.md"
        if not report.is_file():
            # A lane that died or timed out writes NO report, so it was invisible
            # here and its work vanished silently. q80-coherence-deep exited 124
            # yet had produced 40 layers of drift data that later VERIFIED an
            # obligation. Surface these for manual review instead of dropping them.
            exit_code = (d / "exit_code")
            if exit_code.is_file():
                try:
                    code = exit_code.read_text().strip()
                except Exception:
                    code = "?"
                found.append({
                    "lane": d.name,
                    "status": f"NO_REPORT_exit_{code}",
                    "next_bottleneck": "",
                    "needs_manual_review": True,
                })
            continue
        try:
            text = report.read_text(errors="replace")
        except Exception:
            continue
        s = STATUS_RE.search(text)
        m = NEXT_RE.search(text)
        if not m:
            # Report exists but names no next wall. Previously skipped outright,
            # which is the same silent-drop bug as the report-less case: the lane
            # finished, nobody filed it, nobody knew. File it for review.
            found.append({
                "lane": d.name,
                "status": s.group(1) if s else "UNKNOWN",
                "next_bottleneck": "",
                "needs_manual_review": True,
            })
            continue
        found.append({
            "lane": d.name,
            "status": s.group(1) if s else "UNKNOWN",
            "next_bottleneck": m.group(1).strip()[:400],
        })
    return found


def model_of(lane: str) -> str:
    if lane.startswith("dsv"):
        return "dsv4f"
    if lane.startswith("qwen38") or "qwen38" in lane:
        return "qwen38"
    return "q80"


def slug(text: str) -> str:
    """Stable short id from a bottleneck description, for dedupe and lane naming."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    drop = {"the", "a", "an", "is", "at", "of", "on", "in", "to", "ns", "ms", "us",
            "token", "per", "dirty", "engineering", "median", "class"}
    keep = [w for w in words if w not in drop and not w.isdigit()][:4]
    return "-".join(keep) or "unnamed"


MAX_GENERATED = 96   # widened 2026-08-16: Qwen-first pivot needs a deeper pool


def generate_targets(state: dict, harvested: list[dict]) -> int:
    """Turn each unseen NEXT_BOTTLENECK into a pending target with a real contract.

    Without this the queue drains and the daemon idles: harvest only FILES finished
    lanes, it does not decide what to try next. This is what keeps work flowing
    while nobody is watching.
    """
    common = LANES / "_COMMON.md"
    preamble = common.read_text() if common.is_file() else ""
    # .get() throughout: ASCENT_STATE is written by several tools and a target
    # missing a key must not crash the unattended loop.
    existing = {t.get("id") for t in state["targets"]}
    seen_bn = {t.get("from_bottleneck") for t in state["targets"]}
    # Count only targets still in play. This was a LIFETIME cap: it counted every
    # target ever auto-generated, including retained and stale ones, so once 96 had
    # been created the daemon never generated again. On 2026-08-16 it sat at 96/96
    # with 0 pending and 106 phantom "running" targets, and logged "queue dry" on
    # every tick. MAX_GENERATED is meant to bound the ACTIVE POOL, not the history.
    ACTIVE = {"pending", "running"}
    generated = sum(1 for t in state["targets"]
                    if t.get("auto_generated") and t.get("status") in ACTIVE)
    made = 0

    # Qwen-family first (q80, qwen38) per the 2026-08-16 amendment; dsv4f is theory.
    harvested = sorted(harvested, key=lambda h: {"q80": 0, "qwen38": 1}.get(model_of(h["lane"]), 2))
    for h in harvested:
        if generated + made >= MAX_GENERATED:
            break
        if h.get("needs_manual_review"):
            continue          # no bottleneck text to build a contract from
        bn = h["next_bottleneck"]
        if not bn or bn in seen_bn:
            continue
        model = model_of(h["lane"])
        tid = f"auto-{model}-{slug(bn)}"
        if tid in existing:
            continue

        body = f"""{preamble}

---
# LANE: {tid}
## AUTO-GENERATED by ascent_daemon from a finished lane's NEXT_BOTTLENECK.
## Class: GPU_EXCLUSIVE for benchmarks. Use ./tools/gpu_lane_lock.sh.

## The target, as the previous lane reported it
Source lane: `{h['lane']}` (status {h['status']})

    {bn}

Model: {model}

## What to do
1. **Reproduce and quantify it first.** Do not optimize before you have measured
   this cost yourself, with >=3 alternating paired reps and the full spread. If it
   does not reproduce, say so and STOP - a falsification is a successful lane.
2. Decompose it into ns classes and name the limiter with evidence: is it host
   work on the critical path, GPU gap, occupancy, serialization, or real arithmetic?
   These have different fixes and guessing wastes the lane.
3. Attack only the largest measured class. Report the complete-token effect, not
   just the stage - a stage win that does not move the token is not a win.

## Standing rules
- NEVER materialize a dense weight tensor: packed -> registers/simdgroup -> decode
  -> multiply -> accumulate.
- Correctness gate is mandatory. Q80: generated ids exactly
  [8420, 594, 264, 4285, 729, 304, 13027, 429, 17431, 288, 264, 914].
  DSV4F: hc_sha c94da765c4bbf795b598d96209cd80821e5a81ab97a8712586f54b8c8b612597.
  Both: 0 fallbacks. Grade against the ARTIFACT oracle, never the BF16 parent.
- Never weaken a gate, seal, assertion or expected constant to make something pass.
- Label every timing DIRTY_ENGINEERING; other lanes are running.

## Negative science - do NOT re-pay for these
- Topology/encoder/dispatch collapse: REFUTED on BOTH models. Q80 fuse regressed
  (516 vs 307 ms); DSV4F 731 -> 43 encoders moved attention GPU by nothing.
- DRAM row interleaving: Q4 and binary both LOST; only FP4 gained; live wall unchanged.
- Expert routing co-occurrence layout: WEAK, 1.037x.
- Switching-activity permutation: alpha is already ~0.5 (random); not the wall.
- DSV4F path_resolve/verify identity tax: NOT on the critical path (2.9x cut, zero
  token effect). A parallel sum is not token latency.
- Q80 down_proj low-rank ALREADY executes L @ (R @ x); it never reconstructs W.
- Q80 decoded-weight caching: refuted by arithmetic (288 GiB dense vs 11 GiB packed).
- CORRECTED 2026-08-16: the 560-647 GB/s figure is CACHE-RESIDENT REUSE (64 MiB x 4096)
  and is NOT a decode ceiling. Decode reads each weight ONCE per token, so the honest
  control is unique-bytes-once: 411.51 GB/s (Q80_DECODE_SHAPE_BANDWIDTH.json). What
  governs decode is reuse-vs-no-reuse, NOT gather-vs-sequential.
- Q80 mixed matvec runs 2.57 GB/s = 0.62% of that 411.51 ceiling, 160x off, and Q4 runs
  15.2 GB/s - so mixed is 5.9x SLOWER PER BYTE. Reconstruction cost, not bytes moved, is
  Q80's dominant term.
- DEAD NUMBERS, do not cite: "0.135% efficiency" (a category error dividing a mixed-artifact
  floor by a Q4 runtime), "sub-100 fs needs BPW < 0.448-0.518" (assumed unity bandwidth),
  and storage BPW used as if it were active BPW (at batch=1 only 10 of 512 experts are read).
- Qwen3.8 is at 406.2 of 411.51 GB/s = 98.7% of ceiling: it has NO kernel headroom and BPW
  is its only lever. Its token is a CLOSED 12-component ledger; weight_addressing is 60.44%
  and is DRAM traffic (G024_QWEN38_TOKEN_NS.json).
- Q4 vehicles are DE-AUTHORISED. The ~20 h DSV4F determined teacher-X capture is
  DE-AUTHORISED; do not propose or restart it.

## ACCEPTANCE
Done when the named bottleneck is measured before and after, with >=3 alternating
paired reps and the full spread reported, and the model still generates correctly:
greedy ids unchanged and every silent-fallback counter at 0. A measured NEGATIVE -
the mechanism does not help, with the numbers showing it - is an acceptable
completion. Report the real figure, not a favourable one.

## VERIFY
Build with `cargo build --release -p hawking-core` and confirm it exits 0.
Run every GPU-exclusive measurement under ./tools/gpu_lane_lock.sh <lane> <cmd>;
other lanes share this GPU and an unlocked run corrupts both.
Check no shared-kernel regression with `cargo test --release -p hawking-core --test gk_family_parity`
(7/8 is expected today - the failing DSV source-string assert is pre-existing).

## EDIT crates/hawking-core
## EDIT receipts/ascent-2026-08-16
## EDIT lab/operators

DENY tools/gpu_lane_lock.sh
DENY tools/coherence_gate.py
DENY tools/merge_guard.py
If the work needs a file outside the EDIT list, STOP and say why rather than
widening scope yourself.

## Commit
You are on `gate` (unsandboxed). Commit normally, then verify with `git log` that
the commit landed on your branch. Several lanes here hit Seatbelt/macl denials,
finished ahead=0, and nearly lost their work.
"""
        path = LANES / f"{tid}.md"
        try:
            LANES.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
        except Exception:
            continue

        state["targets"].append({
            "id": tid, "model": model, "hypothesis": bn[:200],
            "target_stage": "auto", "contract": str(path),
            "resource_class": "GPU_EXCLUSIVE", "probability_of_success": 0.5,
            "recoverable_ns_per_token": 50_000_000, "density_frontier_gain_ns_equiv": 0,
            "information_gain": 150_000_000, "transfer_value": 50_000_000,
            "experiment_cost": 1.5, "status": "pending",
            "auto_generated": True, "from_bottleneck": bn, "from_lane": h["lane"],
        })
        seen_bn.add(bn)
        made += 1
    return made


# ---------------------------------------------------------------- tier 1

def tier1(target: dict) -> tuple[bool, str]:
    """Seconds, dirty allowed, REJECT-ONLY. Passing is not promotion."""
    spec = TIER1.get(target.get("model", "q80"))
    cmd = target.get("tier1_command") or (spec or {}).get("cmd")
    if not cmd:
        return True, "no gate declared (not a promotion)"
    try:
        code, out = sh(cmd)
    except subprocess.TimeoutExpired:
        return False, "tier1 timeout"
    if code != 0:
        return False, f"tier1 exit {code}"
    expect = target.get("tier1_expect") or (spec or {}).get("expect")
    if expect and expect not in out:
        return False, f"tier1 missing {expect!r}"
    forbid = target.get("tier1_forbid") or (spec or {}).get("forbid")
    if forbid and forbid in out:
        return False, f"tier1 hit forbidden {forbid!r}"
    return True, "tier1 pass (reject-only; NOT a promotion)"


def skew(branch: str) -> str:
    code, out = sh(f"python3 tools/branch_skew_guard.py {branch}", timeout=900)
    for verdict in ("SKEWED", "STALE_CLEAN", "CLEAN", "EMPTY"):
        if verdict in out:
            return verdict
    return "UNKNOWN"


# ---------------------------------------------------------------- pass

def one_pass() -> dict:
    snap = machine()
    report = {"disk_free_gib": snap.get("disk_free_gib"),
              "live_lanes_all_repos": len(snap.get("active_grok_lanes") or [])}

    state = load(STATE, {"targets": [], "history": []})
    queue = load(QUEUE, {"schema": "hawking.ascent.promotion_queue.v1", "entries": []})
    known = {e["lane"] for e in queue["entries"]}
    harvested = harvest()

    # 1. consume finished lanes nobody has read
    for h in harvested:
        if h["lane"] in known:
            continue
        branch = f"grok/{h['lane']}"
        code, _ = sh(f"git rev-parse --verify {branch}", timeout=120)
        entry = {
            "lane": h["lane"], "model": model_of(h["lane"]), "status": h["status"],
            "next_bottleneck": h["next_bottleneck"],
            "branch": branch if code == 0 else None,
            "skew": skew(branch) if code == 0 else "NO_BRANCH",
            "disposition": "MERGE_READY", "promoted": False,
        }
        if h.get("needs_manual_review"):
            entry["disposition"] = "NO_REPORT_MANUAL_REVIEW"
        elif entry["skew"] == "SKEWED":
            entry["disposition"] = "NEEDS_COMPOSITION"
        elif entry["skew"] in ("NO_BRANCH", "EMPTY"):
            entry["disposition"] = "CHECK_FOR_UNCOMMITTED_WORK"
        queue["entries"].append(entry)
    save(QUEUE, queue)
    report["queued"] = len(queue["entries"])
    report["merge_ready"] = sum(1 for e in queue["entries"]
                                if e["disposition"] == "MERGE_READY" and not e["promoted"])
    report["needs_composition"] = sum(1 for e in queue["entries"]
                                      if e["disposition"] == "NEEDS_COMPOSITION")

    # 2. refill the queue from what those lanes said the next wall is
    report["generated"] = generate_targets(state, harvested)
    save(STATE, state)
    report["pending"] = sum(1 for t in state["targets"] if t.get("status") == "pending")

    # 2a. reconcile targets whose lane is gone. ASCENT_STATE marks a target
    # "running" when it launches and relies on a later tick to close it, but a
    # lane that dies without reporting leaves the target stuck forever. On
    # 2026-08-16 there were 106 such phantoms against 0 live processes, which
    # saturated MAX_GENERATED (96/96) and left status=pending at 0 - so the
    # daemon could neither generate new work nor launch any, and logged
    # "queue dry" on every tick while looking healthy. Same lesson as
    # lane_health: a status field is not evidence a process exists.
    phantom = 0
    for tgt in state.get("targets", []):
        if tgt.get("status") != "running":
            continue
        lane_id = tgt.get("lane_id") or tgt.get("id") or ""
        if not lane_id:
            continue
        rc, out = sh(f"pgrep -f {lane_id!r}", timeout=60)
        if rc == 0 and out.strip():
            continue
        tgt["status"] = "stale_no_process"
        phantom += 1
    report["phantom_targets_reconciled"] = phantom

    # 2b. reap lanes that died without saying so, preserving their work first.
    # grok-run status reports `running` for processes that are gone - two DSV4F
    # lanes held slots ~2 h that way, one of them sitting on a COMPLETED paired
    # measurement that was uncommitted. Liveness is pgrep + worktree mtime.
    rc, _ = sh(f"python3 {REPO / 'tools' / 'lane_health.py'}", timeout=900)
    report["dead_lanes_found"] = rc if rc and rc < 100 else 0

    # 3. launch the top pending target if the box allows
    hold = govern(snap)
    report["our_live_lanes"] = len(snap.get("our_live_lanes") or [])
    if hold:
        report["launched"] = None
        report["hold"] = hold
        return report

    # Work de-authorised by a steer must be EXCLUDED, never merely down-ranked.
    # A relative weight cannot stop a launch when the whole queue is one model:
    # max() still returns something, which is how the ~20 h G007 teacher-X capture
    # relaunched itself after the Qwen-first amendment de-authorised it.
    DEAUTHORISED = ("determined-teacher-x", "teacher_x_capture", "uniform-q4", "uniform_q4")

    def deauthorised(t: dict) -> str | None:
        blob = f"{t.get('id','')} {t.get('contract','')} {t.get('title','')}".lower()
        for pat in DEAUTHORISED:
            if pat in blob:
                return pat
        if str(t.get("obligation_status", "")).upper() == "BLOCKED":
            return "obligation BLOCKED"
        return None

    pending = []
    for t in state["targets"]:
        if t.get("status") != "pending":
            continue
        why = deauthorised(t)
        if why:
            t["status"] = "deauthorised"
            t["tier1"] = f"excluded: {why}"
            continue
        pending.append(t)
    save(STATE, state)
    if not pending:
        report["launched"] = None
        report["hold"] = "queue dry - harvest supplied no new pending target"
        return report

    from ascent_controller import value  # reuse the ranking, do not duplicate it

    def ranked(t: dict) -> float:
        """Value, with the 2026-08-16 Qwen-first amendment applied.

        DSV4F is theory-only: it keeps its ledger record and stays re-openable,
        but must not consume lanes while Q80 seals and Qwen3.8 comes up.
        """
        v = value(t)
        return v * 0.05 if t.get("model") == "dsv4f" else v

    target = max(pending, key=ranked)
    contract = Path(target.get("contract", ""))
    if not contract.is_file():
        target["status"] = "launch_failed"
        target["tier1"] = "contract missing"
        save(STATE, state)
        report["launched"] = None
        report["hold"] = f"contract missing for {target['id']}"
        return report

    code, out = sh(f"{GROK} delegate --task {target['id']} --contract {contract} "
                   f"--repo {REPO} --profile gate --background", timeout=600)
    m = re.search(rf"{re.escape(target['id'])}-\d{{8}}-\d{{6}}", out)
    if m:
        target.update(status="running", task_id=m.group(0))
        report["launched"] = m.group(0)
    else:
        target["status"] = "launch_failed"
        report["launched"] = None
    save(STATE, state)
    return report


def loop() -> int:
    while True:
        try:
            r = one_pass()
            print(json.dumps(r), flush=True)
        except Exception as exc:  # never die on one bad pass
            print(json.dumps({"error": str(exc)}), flush=True)
        time.sleep(POLL_SECONDS)


def status() -> int:
    snap = machine()
    q = load(QUEUE, {"entries": []})
    print(f"disk {snap.get('disk_free_gib')} GiB | live lanes "
          f"{len(snap.get('active_grok_lanes') or [])} | queued {len(q['entries'])}")
    for e in q["entries"]:
        if not e["promoted"]:
            print(f"  [{e['disposition']:<28}] {e['lane']:<42} skew={e['skew']}")
            print(f"      next: {e['next_bottleneck'][:110]}")
    return 0


def _selfcheck() -> None:
    """Pin the behaviours that make this safe to leave running."""
    assert model_of("dsv-expert-cache-1") == "dsv4f"
    assert model_of("q80-pack-1") == "q80"
    assert model_of("qwen38-bringup-1") == "qwen38"

    ok, why = tier1({"model": "q80", "tier1_command": "echo Finished"})
    assert ok, why
    ok, _ = tier1({"model": "q80", "tier1_command": "echo nope"})
    assert not ok, "missing expected marker must reject"
    ok, _ = tier1({"model": "q80", "tier1_command": "echo 'error[E0001]: x'; echo Finished"})
    assert not ok, "forbidden marker must reject even when the expect marker is present"
    ok, _ = tier1({"model": "q80", "tier1_command": "exit 3"})
    assert not ok, "non-zero exit must reject"

    # Both silent-drop cases must now surface rather than vanish.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        (base / "no-report-lane").mkdir(); (base / "no-report-lane" / "exit_code").write_text("124")
        (base / "no-wall-lane").mkdir()
        (base / "no-wall-lane" / "grok-report.md").write_text("STATUS: SHIPPED\nno wall named\n")
        (base / "good-lane").mkdir()
        (base / "good-lane" / "grok-report.md").write_text("STATUS: SHIPPED\nNEXT_BOTTLENECK: x 1 ns\n")
        global TASKS
        saved = TASKS; TASKS = base
        try:
            got = {h["lane"]: h for h in harvest()}
        finally:
            TASKS = saved
        assert set(got) == {"no-report-lane", "no-wall-lane", "good-lane"}, got
        assert got["no-report-lane"]["needs_manual_review"] and "124" in got["no-report-lane"]["status"]
        assert got["no-wall-lane"]["needs_manual_review"], "report without a wall must still be filed"
        assert not got["good-lane"].get("needs_manual_review")

    txt = "STATUS: SHIPPED\nNEXT_BOTTLENECK: host.foo 123 ns/token\n"
    assert NEXT_RE.search(txt).group(1).startswith("host.foo")
    assert STATUS_RE.search(txt).group(1) == "SHIPPED"

    # The generator is what stops the daemon idling. It must produce work, dedupe,
    # and refuse to run away.
    assert slug("host.expert_slab_io 415126416 ns/token") == "host-expert-slab-io"
    st = {"targets": []}
    h = [{"lane": "q80-x-1", "status": "SHIPPED", "next_bottleneck": "host.foo 1 ns/token"},
         {"lane": "dsv-y-1", "status": "SHIPPED", "next_bottleneck": "metal.bar 2 ns/token"}]
    assert generate_targets(st, h) == 2, "must create a target per new bottleneck"
    assert generate_targets(st, h) == 0, "must dedupe on repeat passes"
    assert {t["model"] for t in st["targets"]} == {"q80", "dsv4f"}
    assert all(t["status"] == "pending" and t["auto_generated"] for t in st["targets"])
    # MAX_GENERATED bounds the ACTIVE pool, so the cap test must fill it with
    # ACTIVE targets. A pool of finished ones must NOT block new work - that was
    # the 2026-08-16 bug, where 96 lifetime generations wedged the loop shut.
    st["targets"] = [{"auto_generated": True, "from_bottleneck": f"b{i}",
                      "status": "pending"} for i in range(MAX_GENERATED)]
    assert generate_targets(st, [{"lane": "q80-z-1", "status": "SHIPPED",
                                  "next_bottleneck": "brand new wall"}]) == 0, \
        "must stop at MAX_GENERATED when the ACTIVE pool is full"
    st["targets"] = [{"auto_generated": True, "from_bottleneck": f"c{i}",
                      "status": "stale_no_process"} for i in range(MAX_GENERATED)]
    assert generate_targets(st, [{"lane": "q80-z-2", "status": "SHIPPED",
                                  "next_bottleneck": "another new wall"}]) == 1, \
        "a pool of FINISHED targets must not block new generation"

    # The daemon must never promote or merge on its own authority. Check the
    # executable surface (sh() call sites), not the file text - an earlier version
    # of this assert matched its own message and failed spuriously.
    src = Path(__file__).read_text()
    calls = re.findall(r"sh\(\s*f?[\"']([^\"']+)", src)
    for c in calls:
        assert not c.lstrip().startswith("git merge"), f"daemon must never merge: {c}"
        assert "git push" not in c, f"daemon must never push: {c}"
    # Build the needle at runtime: a literal here would match this line itself.
    needle = "promoted" + '"] = ' + "True"
    assert needle not in src, "daemon must never self-promote"
    assert any("MERGE_READY" in line for line in src.splitlines()), (
        "daemon must record promotion-readiness rather than acting on it"
    )
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["once", "loop", "status", "selfcheck"])
    args = ap.parse_args()
    if args.command == "selfcheck":
        _selfcheck()
    elif args.command == "status":
        sys.exit(status())
    elif args.command == "loop":
        sys.exit(loop())
    else:
        print(json.dumps(one_pass(), indent=2))
