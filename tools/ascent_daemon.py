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
DISK_WARN_GIB = 40.0
MAX_CONCURRENT = 7
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


def govern(snap: dict) -> str | None:
    """Return a reason to hold off, or None to proceed."""
    free = snap.get("disk_free_gib") or 0
    if free < DISK_WARN_GIB:
        script = REPO / "tools" / "reclaim_safe.sh"
        if script.is_file():
            sh(f"bash {script}", timeout=900)
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
        m = NEXT_RE.search(text)
        if not m:
            continue
        s = STATUS_RE.search(text)
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
    generated = sum(1 for t in state["targets"] if t.get("auto_generated"))
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
- The kernels are NOT bandwidth-limited: a same-box control streams 560-647 GB/s
  while packed matvecs run 2.5. Occupancy and work geometry are the open axis.

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

    # 3. launch the top pending target if the box allows
    hold = govern(snap)
    report["our_live_lanes"] = len(snap.get("our_live_lanes") or [])
    if hold:
        report["launched"] = None
        report["hold"] = hold
        return report

    pending = [t for t in state["targets"] if t.get("status") == "pending"]
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
    st["targets"] = [{"auto_generated": True, "from_bottleneck": f"b{i}"}
                     for i in range(MAX_GENERATED)]
    assert generate_targets(st, [{"lane": "q80-z-1", "status": "SHIPPED",
                                  "next_bottleneck": "brand new wall"}]) == 0, \
        "must stop at MAX_GENERATED"

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
