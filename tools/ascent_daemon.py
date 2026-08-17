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
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "receipts" / "ascent-2026-08-16" / "ASCENT_STATE.json"
QUEUE = REPO / "receipts" / "ascent-2026-08-16" / "PROMOTION_QUEUE.json"
TASKS = Path.home() / ".claude-grok" / "tasks"
GROK = Path.home() / ".claude-grok" / "bin" / "grok-run"
# Durable, in-repo. The session scratchpad does NOT survive the session, and a
# daemon meant to run unattended cannot depend on a path that disappears.
LANES = REPO / "workspace" / "ops" / "ascent-lanes"
# The resident and every protected benchmark use the same exclusive GPU lane.
# A resident proposal is a real decode, not CPU-only planning, so it must never
# be started merely to wait behind a protected measurement.  Besides wasting
# the serial body, that makes health and the other logical sessions unavailable
# for the full proposal duration.
GPU_LANE_LOCK = Path("/tmp/hawking-gpu-lane.lock")
GPU_RESOURCE_CLASSES = frozenset(
    {"GPU_DIRTY", "GPU-LAB", "GPU_PROTECTED", "GPU_EXCLUSIVE", "MIXED"}
)

DISK_FLOOR_GIB = 15.0
DISK_WARN_GIB = 90.0   # raised after a 0-byte stall: lanes cost 1-19 GiB each
MAX_ATTEMPTS_PER_BOTTLENECK = 12   # a dominant cost deserves many mechanisms,
                       # but not an unbounded grind; 9 have already failed on
                       # weight_addressing and it is still the right target
MAX_CONCURRENT = 10    # raised again per user steer: the 0-byte stall is now guarded
                       # by the governor reaping the grok worktree pool, which is the
                       # real protection - the cap was only ever a blunt proxy for it
# A completion or failed launcher should be visible promptly enough to keep the
# organism advancing, without turning the campaign ledger into a polling storm.
POLL_SECONDS = 60
DELEGATE_LOG = REPO / "workspace" / "ops" / "ascent-delegates.log"
# Candidate requests are produced by durable AgentOS work but never promote from
# inside a model session. The daemon may *dispatch* this separate external
# controller when an inbox item is ready; all protected evidence and lineage
# mutation remain in tools/genesis_lifecycle.py.
GENESIS_CANDIDATE_ROOT = REPO / "workspace" / "ops" / "genesis-candidates"
GENESIS_LIFECYCLE = REPO / "tools" / "genesis_lifecycle.py"
GENESIS_LIFECYCLE_LOG = REPO / "workspace" / "ops" / "genesis-lifecycle.log"
GENESIS_LIFECYCLE_STATE = REPO / "workspace" / "ops" / "genesis-lifecycle-controller.json"
# One bounded HCLI turn gives each logical worker a real implementation surface
# between protected experiments.  It is a separate process because the daemon
# remains only a scheduler: the worker cannot gain lifecycle authority from it.
GENESIS_AGENTOS = REPO / "tools" / "genesis_agentos.py"
GENESIS_AGENTOS_LOG = REPO / "workspace" / "ops" / "genesis-agentos.log"
GENESIS_AGENTOS_STATE = REPO / "workspace" / "ops" / "genesis-agentos-controller.json"

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


def process_alive(pid: object) -> bool:
    """Return real process liveness, never a stale task-state assertion.

    A Grok task's ``status`` file is written before the executor starts.  It is
    therefore useful telemetry but not proof that an optimizer still exists.
    A detached launcher PID is the authoritative liveness signal for new lanes;
    legacy lanes retain the conservative pgrep fallback in ``one_pass`` below.
    """
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except OSError:
        return False
    # ``kill(pid, 0)`` still succeeds for an unreaped zombie.  The original
    # background-launch failure left exactly that shape: a dead runner held a
    # PID forever and suppressed retries.  Treat Z* process states as dead;
    # an uncertain ``ps`` result remains conservatively live.
    try:
        proc = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(value)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    state = proc.stdout.strip()
    return bool(state) and not state.startswith("Z")


def launch_target(target: dict, resource_class: str, profile: str) -> int | None:
    """Start one lane under a durable, independently observable launcher.

    ``grok-run --background`` forks a child that can disappear after only its
    optimistic ``running`` marker is written.  Calling the runner in the
    foreground from a detached process preserves its normal task receipts while
    giving the Genesis supervisor a PID it can actually verify and reap.
    """
    contract = Path(str(target.get("contract") or ""))
    task = str(target.get("id") or "").strip()
    if not task or not contract.is_file():
        return None
    DELEGATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(GROK),
        "delegate",
        "--task",
        task,
        "--contract",
        str(contract),
        "--repo",
        str(REPO),
        "--profile",
        profile,
    ]
    try:
        with DELEGATE_LOG.open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except OSError:
        return None
    target.update(
        status="running",
        launcher_pid=proc.pid,
        admitted_resource_class=resource_class,
        admitted_profile=profile,
        launch_backend="detached_foreground_grok_run",
    )
    return proc.pid


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
    cap = memory_lane_cap()
    snap["memory_lane_cap"] = cap
    if len(ours) >= cap:
        return f"{len(ours)} OUR lanes live, at the memory-derived cap {cap}"
    return None


# A Grok lane measured 6 GiB of worktree and working set on this box. The old cap was
# a flat 10, which either starved a big box or over-subscribed a busy one; drive it
# from what is ACTUALLY free instead, and keep the generation reserve untouchable so
# a promoted successor always has somewhere to launch.
LANE_WORKING_SET_GIB = 6.0
GENERATION_RESERVE_GIB = 14.08
NO_SWAP_FLOOR_GIB = 4.0
TARGET_FILL = 0.90


LINEAGE_STATE = REPO / "receipts" / "ascent-2026-08-16" / "GENESIS_LINEAGE_CURRENT.json"


def _candidate_pending() -> bool:
    """True when a successor is nominated and could be launched this tick.

    Fail SAFE: if the lineage file is missing or unreadable, assume a candidate is
    pending and keep the reserve. Under-reserving risks a promoted successor having
    nowhere to launch, which is the one failure the lineage cannot recover from.
    """
    try:
        if bool(json.loads(LINEAGE_STATE.read_text())["slots"].get("CANDIDATE")):
            return True
    except (OSError, ValueError, KeyError):
        return True
    # A valid-looking request has not yet been nominated, but it can become a
    # real test subject next. Reserve room now rather than launching enough
    # unrelated lanes to make a protected child impossible to start.
    try:
        return any(
            any((GENESIS_CANDIDATE_ROOT / name).glob("*.json"))
            for name in ("inbox", "active")
        )
    except OSError:
        return True


def memory_lane_cap() -> int:
    """How many concurrent lanes fit under TARGET_FILL, reserving a generation slot."""
    try:
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                   capture_output=True, text=True, timeout=30).stdout.strip())
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return MAX_CONCURRENT
    page, vals = 16384, {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip().rstrip(".")
        if v.isdigit():
            vals[k.strip()] = int(v) * page
    gib = 1024 ** 3
    # Inactive and purgeable pages are reclaimable; counting only "free" understates
    # what is available badly enough to starve the loop on an otherwise idle box.
    available = (vals.get("Pages free", 0) + vals.get("Pages inactive", 0)
                 + vals.get("Pages purgeable", 0)) / gib
    used = total / gib - available
    # Reserve a successor body only when a promotion can actually happen this tick,
    # i.e. when CANDIDATE is occupied. Reserving unconditionally cost 14.08 GiB of
    # permanent idle headroom and pinned the cap at 1 lane on a box that was 34%
    # free - starving the loop to protect an event that could not occur.
    reserve = GENERATION_RESERVE_GIB if _candidate_pending() else 0.0
    budget = min(TARGET_FILL * total / gib - used,
                 available - NO_SWAP_FLOOR_GIB) - reserve
    if budget <= 0:
        return 1                       # never zero: the loop must still make progress
    return max(1, min(40, int(budget // LANE_WORKING_SET_GIB)))


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
    """Name the model a lane belongs to.

    The q80 default is load-bearing in the wrong direction: q80 is sealed and not in
    ACTIVE_MODELS, so anything falling through here has its bottleneck dropped
    silently AND is then caught by the `-q80-` deauthorisation pattern. The `q38-*`
    lanes are Qwen3.8 work that was being banned as sealed-model work for exactly
    that reason - the 4.253 BPW attention roof and the 964-dispatch GPU body among
    them. Match the abbreviation too.
    """
    if lane.startswith("dsv"):
        return "dsv4f"
    if (
        lane.startswith("qwen38")
        or "qwen38" in lane
        or lane.startswith("q38")
        or "q38-" in lane
        or lane.startswith("genesis")
    ):
        return "qwen38"
    return "q80"


# Q80 lost the tournament and DSV4F was sealed out; both have had their weights
# DELETED, so a lane targeting either cannot build, cannot measure, and cannot
# promote. The harvest still surfaces their old NEXT_BOTTLENECK lines from finished
# lanes, and unattended that is how a whole night gets spent on a dead model - the
# first retry this fix generated was for Q80.
ACTIVE_MODELS = {"qwen38"}

RESOURCE_PROFILE_ALIASES: dict[str, tuple[str, str]] = {
    "CPU_ONLY": ("CPU_ONLY", "maximum"),
    "LIGHT": ("CPU_ONLY", "maximum"),
    "LIGHT-READONLY": ("CPU_ONLY", "maximum"),
    "GPU_DIRTY": ("GPU_DIRTY", "gate"),
    "GPU-LAB": ("GPU_DIRTY", "gate"),
    "GPU_PROTECTED": ("GPU_PROTECTED", "gate"),
    "GPU_EXCLUSIVE": ("GPU_PROTECTED", "gate"),
    "MIXED": ("MIXED", "gate"),
}

DEAUTHORISED_PATTERNS = (
    "determined-teacher-x",
    "teacher_x_capture",
    "auto-q80-",
    "-q80-",
    "auto-dsv4f-",
    "dsv4f",
)


def resource_profile(target: dict) -> tuple[str, str]:
    """Return canonical resource class and compatible Grok profile.

    Metal work is admitted only to the unsandboxed gate profile. Unknown and
    missing classes fail before delegation instead of discovering the mismatch
    after a model load or benchmark setup.
    """
    raw = str(target.get("resource_class") or "").strip().upper()
    if not raw:
        raise ValueError(f"resource admission refused for {target.get('id')!r}: class missing")
    resolved = RESOURCE_PROFILE_ALIASES.get(raw)
    if resolved is None:
        raise ValueError(
            f"resource admission refused for {target.get('id')!r}: unknown class {raw!r}"
        )
    canonical, profile = resolved
    if canonical in {"GPU_DIRTY", "GPU_PROTECTED", "MIXED"} and profile != "gate":
        raise ValueError(
            f"resource admission refused for {target.get('id')!r}: "
            f"{canonical} requires Metal-capable gate profile"
        )
    return canonical, profile


def target_deauthorised(target: dict) -> str | None:
    """Apply sealed-model exclusion at launch, including stale queue entries."""
    model = str(target.get("model") or "").lower()
    if model and model not in ACTIVE_MODELS:
        return f"sealed model {model}"
    blob = (
        f"{target.get('id', '')} {target.get('contract', '')} "
        f"{target.get('title', '')}"
    ).lower()
    for pattern in DEAUTHORISED_PATTERNS:
        if pattern in blob:
            return pattern
    if str(target.get("obligation_status", "")).upper() == "BLOCKED":
        return "obligation BLOCKED"
    return None


def slug(text: str) -> str:
    """Stable short id from a bottleneck description, for dedupe and lane naming."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    drop = {"the", "a", "an", "is", "at", "of", "on", "in", "to", "ns", "ms", "us",
            "token", "per", "dirty", "engineering", "median", "class"}
    keep = [w for w in words if w not in drop and not w.isdigit()][:4]
    return "-".join(keep) or "unnamed"


def next_target_id(model: str, bottleneck: str, existing: set[object]) -> str:
    """Allocate a fresh retry id without treating historical holes as a stop.

    The old counter was derived from every historical row.  Once an empty proposal
    had created ``try2`` through ``try12``, a later real proposal calculated an
    already-used id and the scheduler silently skipped it.  The attempt budget is
    now based on named mechanisms; the id is only an immutable record name, so it
    must be selected independently and may legitimately be ``try13`` after a run
    of invalid historical placeholders.
    """
    base = f"auto-{model}-{slug(bottleneck)}"
    if base not in existing:
        return base
    retry = 2
    while f"{base}-try{retry}" in existing:
        retry += 1
    return f"{base}-try{retry}"


def has_named_mechanism(target: dict) -> bool:
    """Whether this row represents a real research attempt, not a placeholder."""
    return bool(str(target.get("mechanism") or "").strip())


MAX_GENERATED = 96   # widened 2026-08-16: Qwen-first pivot needs a deeper pool


_DEFAULT_GENESIS_BIN = (
    REPO
    / "workspace"
    / "ops"
    / "build"
    / "rust"
    / "release-fast"
    / "examples"
    / "ascension_qwen38_hybrid_greedy"
)
_DEFAULT_GENESIS_ARTIFACT = (
    REPO / "workspace" / "campaign" / "records" / "runs" / "qwen38-27b" / "uniform-q4-v1"
)
_DEFAULT_GENESIS_TOKENIZER = (
    REPO / "workspace" / "campaign" / "records" / "runs" / "qwen38-27b" / "bf16" / "tokenizer.json"
)
# Resolved at call time so tests can override via the environment.
GENESIS_BIN = _DEFAULT_GENESIS_BIN
GENESIS_ARTIFACT = _DEFAULT_GENESIS_ARTIFACT
GENESIS_TOKENIZER = _DEFAULT_GENESIS_TOKENIZER
GENESIS_RESIDENT_CLIENT = REPO / "tools" / "agentos" / "genesis_resident.py"
GENESIS_RESIDENT_PROPOSE_TIMEOUT_S = 1800
# A reasoning model spends most of its budget inside <think>. At 900 the body ran
# out mid-reasoning and never emitted an answer, so every proposal came back empty
# and the named-mechanism gate refused every target.
#
# The parent proposal asks for four machine-minimal fields, not an essay.  A 2,600
# token cap let a speculative parent decode monopolize the one resident for minutes,
# starving the child_a/child_b HCLI action plane.  512 is ample for a concrete
# mechanism/discriminator while preserving frequent closed-loop scheduling.  The
# 8,192-token resident still leaves generous room for the integrity capsule and task.
GENESIS_PROPOSE_MAX_NEW_TOKENS = 512
# A resident proposal is serial GPU work.  Keep the generic generator capable of
# batch construction for CPU/test callers, but production one_pass admits at most
# this many model decodes before yielding to AgentOS and protected work.
MAX_RESIDENT_PROPOSALS_PER_PASS = 1
GENESIS_SYSTEM_CONTRACT = (
    REPO / "contracts" / "genesis" / "QWEN38_GENESIS_SYSTEM_DIRECTIVE.md"
)


def _genesis_paths() -> tuple[Path, Path, Path]:
    bin_p = Path(os.environ["GENESIS_BIN"]) if os.environ.get("GENESIS_BIN") else _DEFAULT_GENESIS_BIN
    art = Path(os.environ["GENESIS_ARTIFACT"]) if os.environ.get("GENESIS_ARTIFACT") else _DEFAULT_GENESIS_ARTIFACT
    tok = Path(os.environ["GENESIS_TOKENIZER"]) if os.environ.get("GENESIS_TOKENIZER") else _DEFAULT_GENESIS_TOKENIZER
    return bin_p, art, tok


def _genesis_prompt(bottleneck: str) -> str:
    return (
        "You are HAWKING GENESIS, optimizing your own execution. Treat every inherited "
        "performance number as historical until a current-main protected measurement "
        "revalidates it. The old 411.51 GB/s / 97.6% weight-addressing story is REFUTED: "
        "it mixed the wrong bytes, whole-token GPU time, and a sequential control that "
        "was not the Q4 grouped-GEMV roof. Do not infer that density is the only lever.\n\n"
        f"MEASURED BOTTLENECK NOW: {bottleneck}\n\n"
        "MISSION PRIORITY (non-negotiable): make this Genesis materially smaller and "
        "faster. Until >=100 VALID TPS, attack physical BPW, unique-once weight bytes, "
        "or complete-token wall before any convenience feature. Be resourceful in the "
        "actual checkout: inspect current receipts, code, and negative science; choose "
        "one high-leverage mechanism that changes a measured cost; and give its cheapest "
        "falsifier. AgentOS/HCLI work may improve tool-wait utilization only when it is "
        "CPU-safe and cannot delay protected Gravity/kernel work. Do not ask for a plan, "
        "repeat dead theories, or substitute aspiration for a measurable mechanism.\n\n"
        "Already REFUTED by measurement - do not propose these again: fusing tiny kernels "
        "into the following GEMV (+10.68 ms REGRESSION); cross-token cache reuse (hot/cold "
        "gap only 2.5%); amortizing DRAM by issuing N INDEPENDENT dispatches against one "
        "weight body (4 separate GEMVs on one lm_head still cost 4x - they re-read W four "
        "times by construction, so that control could only ever return 4x).\n\n"
        "STILL OPEN, do not treat as refuted: a SINGLE dispatch computing W @ [x1..xN] that "
        "reads W once and reuses it across N columns. Same bottleneck, different mechanism.\n\n"
        "A roof is conditioned on the current genome, never physics. Name the ASSUMPTION "
        "that generates this cost, then ONE mechanism that removes the cost rather than "
        "shrinking it, then the cheapest experiment that would prove it does NOT work.\n\n"
        "Keep reasoning SHORT and reach the answer. Output is read by a scheduler, not a "
        "human, and an answer you never reach is worth nothing. Emit exactly these fields "
        "and no prose around them:\n"
        "ASSUMPTION:\nMECHANISM:\nDISCRIMINATOR:\nREJECT_IF:"
    )


def _resident_mod():
    path = Path(__file__).resolve().parent / "agentos" / "genesis_resident.py"
    spec = importlib.util.spec_from_file_location("genesis_resident", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _genesis_contract_mod():
    """Load the fail-closed canonical contract compiler in this checkout."""
    path = REPO / "tools" / "agentos" / "genesis_contract.py"
    spec = importlib.util.spec_from_file_location("genesis_system_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Genesis system contract helper at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mechanism_mod():
    """Load the semantic mechanism gate without making tools/ a package."""
    path = Path(__file__).resolve().parent / "ascent" / "mechanism_dedup.py"
    ascent_dir = str(path.parent)
    if ascent_dir not in sys.path:
        sys.path.insert(0, ascent_dir)
    spec = importlib.util.spec_from_file_location("genesis_mechanism_dedup", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load mechanism gate at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def mechanism_admission(target: dict, state: dict) -> tuple[bool, dict]:
    """Require a named, non-duplicate mechanism before a target can launch."""
    mechanism = str(target.get("mechanism") or "").strip()
    if not mechanism:
        return False, {
            "verdict": "ERROR",
            "gate": "named_mechanism_required",
            "reason": "target has no named mechanism; bottleneck-only retries are refused",
        }
    try:
        gate = _mechanism_mod()
        attempts = [
            row
            for row in state.get("targets", [])
            if row is not target
            and row.get("mechanism")
            and row.get("status") not in {"pending", "running"}
        ]
        running = [
            row
            for row in state.get("targets", [])
            if row is not target
            and row.get("mechanism")
            and row.get("status") in {"pending", "running"}
        ]
        decision = gate.admit(
            target,
            attempts=attempts,
            running=running,
            include_campaign_exhausted=True,
        )
        payload = decision.to_dict()
        return decision.verdict.value == "ALLOW", payload
    except Exception as exc:
        return False, {
            "verdict": "ERROR",
            "gate": "mechanism_gate_unavailable",
            "reason": str(exc),
        }


def _resident_process_alive() -> bool:
    """Is a resident body process alive, regardless of whether it can answer now?

    The body serves decodes serially, so its health socket goes quiet for the whole
    length of another session's generation. Health alone cannot tell BUSY from DEAD.
    """
    try:
        p = subprocess.run(["pgrep", "-f", "genesis-resident"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(p.stdout.strip())


def gpu_lane_busy() -> bool:
    """Whether a *live* owner currently holds the shared GPU lane.

    An incomplete lock still fails closed.  A numeric PID that is definitely
    dead is different: a fresh resident acquisition will atomically reclaim it
    in ``gpu_lane_lock.sh``/``GpuLaneGuard``.  Treating that stale directory as
    permanently busy stranded the post-restart AgentOS loop before it could
    make the very acquisition that repairs the lock.
    """
    if not GPU_LANE_LOCK.exists():
        return False
    try:
        pid = int((GPU_LANE_LOCK / "pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return True
    return process_alive(pid)


def protected_gpu_target_active() -> bool:
    """Whether the durable queue says a protected GPU experiment is in flight.

    There is a short interval before an executor creates the on-disk GPU lock.
    Treat that interval as occupied too.  Otherwise a parent proposal can win the
    race, block the one resident body, and force the real benchmark to wait.  A
    stale ``running`` target only delays new resident speculation until the next
    reconciliation pass; it is safer than contaminating an active measurement.
    """
    try:
        state = load(STATE, {"targets": []})
        targets = state.get("targets", [])
    except (OSError, ValueError, AttributeError):
        return True
    if not isinstance(targets, list):
        return True
    for target in targets:
        # A durable pending protected target reserves the lane too.  A parent
        # proposal may otherwise start in the interval before the dispatcher
        # launches the worker, seize the serial resident, and make the worker
        # wait behind speculative text generation.
        if not isinstance(target, dict) or target.get("status") not in {"pending", "running"}:
            continue
        raw = str(target.get("resource_class") or "").strip().upper()
        if raw in GPU_RESOURCE_CLASSES:
            return True
    return False


def protected_gpu_target_running(state: dict | None = None) -> bool:
    """Whether an already-launched protected target occupies the GPU now.

    Pending targets still reserve the lane from speculative parent proposals,
    but a short AgentOS tool turn may run between them. This distinction keeps
    both loops alive while never overlapping a resident decode with an actual
    protected capture.
    """
    try:
        source = state if state is not None else load(STATE, {"targets": []})
        targets = source.get("targets", [])
    except (OSError, ValueError, AttributeError):
        return True
    if not isinstance(targets, list):
        return True
    for target in targets:
        if not isinstance(target, dict) or target.get("status") != "running":
            continue
        raw = str(target.get("resource_class") or target.get("admitted_resource_class") or "").strip().upper()
        if raw in GPU_RESOURCE_CLASSES:
            return True
    return False


def _candidate_inbox_entries(name: str) -> list[Path]:
    """Return only durable candidate request files, never inferred success."""
    directory = GENESIS_CANDIDATE_ROOT / name
    try:
        if not directory.is_dir():
            return []
        return sorted(path for path in directory.glob("*.json") if path.is_file() and not path.is_symlink())
    except OSError:
        return []


def candidate_lifecycle_status() -> dict:
    """Observe the external controller without gaining promotion authority."""
    inbox = _candidate_inbox_entries("inbox")
    active = _candidate_inbox_entries("active")
    saved = load(GENESIS_LIFECYCLE_STATE, {})
    if not isinstance(saved, dict):
        saved = {}
    pid = saved.get("pid")
    if process_alive(pid):
        return {
            "status": "running",
            "pid": int(pid),
            "inbox": [path.name for path in inbox],
            "active": [path.name for path in active],
        }
    if active:
        # A controller process that vanished while owning a request is not
        # retried blindly. It may have changed lineage after its last durable
        # write, so this is an explicit recovery state for the next controller.
        return {
            "status": "recovery_required",
            "inbox": [path.name for path in inbox],
            "active": [path.name for path in active],
        }
    if inbox:
        return {"status": "pending", "inbox": [path.name for path in inbox], "active": []}
    return {"status": "idle", "inbox": [], "active": []}


def dispatch_candidate_lifecycle() -> dict:
    """Start one external one-shot lifecycle controller if a request awaits it.

    This function deliberately knows neither candidate contents nor lineage
    mutation APIs. It is a durable process dispatcher in the same sense as
    ``launch_target``; the separately executable controller owns promotion.
    """
    status = candidate_lifecycle_status()
    if status["status"] != "pending":
        return status
    if not GENESIS_LIFECYCLE.is_file():
        return {**status, "status": "unavailable", "reason": "genesis lifecycle CLI missing"}
    GENESIS_LIFECYCLE_LOG.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(GENESIS_LIFECYCLE),
        "process-inbox",
        "--repo",
        str(REPO),
    ]
    try:
        with GENESIS_LIFECYCLE_LOG.open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                command,
                cwd=REPO,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except OSError as exc:
        return {**status, "status": "launch_failed", "reason": str(exc)}
    save(
        GENESIS_LIFECYCLE_STATE,
        {
            "schema": "hawking.genesis.lifecycle_dispatch.v1",
            "status": "running",
            "pid": proc.pid,
            "started_at": time.time(),
            "command": command,
            "inbox_before_launch": status["inbox"],
        },
    )
    return {**status, "status": "started", "pid": proc.pid}


def _agentos_interval_s() -> int:
    """Bound retry pressure if the resident emits malformed/no tool calls."""
    try:
        requested = int(os.environ.get("GENESIS_AGENTOS_MIN_INTERVAL_S", "180"))
    except ValueError:
        requested = 180
    return min(max(requested, 30), 3_600)


def agentos_turn_status() -> dict:
    """Observe one bounded HCLI turn without treating a worker as a child.

    A live AgentOS process owns a serial resident decode even though it does
    not hold the protected GPU benchmark lock.  The scheduler uses this state
    to avoid launching a protected target into that decode.
    """
    try:
        saved = load(GENESIS_AGENTOS_STATE, {})
    except (OSError, ValueError):
        saved = {}
    if not isinstance(saved, dict):
        saved = {}
    pid = saved.get("pid")
    worker_id = saved.get("worker_id")
    if process_alive(pid):
        return {
            "status": "running",
            "pid": int(pid),
            "worker_id": worker_id,
            "started_at": saved.get("started_at"),
        }
    try:
        started_at = float(saved.get("started_at", 0.0))
    except (TypeError, ValueError):
        started_at = 0.0
    remaining = max(0, int(_agentos_interval_s() - (time.time() - started_at))) if started_at else 0
    if remaining:
        return {
            "status": "cooldown",
            "worker_id": worker_id,
            "next_turn_in_s": remaining,
        }
    return {"status": "idle", "worker_id": worker_id}


def _next_agentos_worker() -> str:
    """Round-robin separate durable fronts instead of starving kernel work."""
    try:
        saved = load(GENESIS_AGENTOS_STATE, {})
    except (OSError, ValueError):
        saved = {}
    previous = saved.get("worker_id") if isinstance(saved, dict) else None
    return "kernel" if previous == "gravity" else "gravity"


def dispatch_agentos_turn() -> dict:
    """Launch one bounded non-authoritative AgentOS/HCLI implementation turn."""
    status = agentos_turn_status()
    if status["status"] != "idle":
        return status
    if not GENESIS_AGENTOS.is_file():
        return {**status, "status": "unavailable", "reason": "genesis AgentOS CLI missing"}
    worker_id = _next_agentos_worker()
    GENESIS_AGENTOS_LOG.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(GENESIS_AGENTOS),
        "tick",
        "--repo",
        str(REPO),
        "--worker",
        worker_id,
        "--max-rounds",
        "1",
    ]
    try:
        with GENESIS_AGENTOS_LOG.open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                command,
                cwd=REPO,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except OSError as exc:
        return {**status, "status": "launch_failed", "worker_id": worker_id, "reason": str(exc)}
    save(
        GENESIS_AGENTOS_STATE,
        {
            "schema": "hawking.genesis.agentos_dispatch.v1",
            "status": "running",
            "pid": proc.pid,
            "worker_id": worker_id,
            "started_at": time.time(),
            "command": command,
        },
    )
    return {**status, "status": "started", "pid": proc.pid, "worker_id": worker_id}


def _try_resident_propose(prompt: str) -> str:
    """Run one resident inference against the resident body.

    A body that cannot answer is only a fast no-op when it is genuinely gone. A BUSY
    body fails health too - it is mid-decode for another session - and treating that
    as dead returned an empty proposal, which the caller recorded as a target with no
    mechanism, spending one of the bounded attempts on a timing coincidence.

    The client is passed the prompt as one argv item; no prompt text is ever
    interpreted by a shell.
    """
    try:
        resident = _resident_mod()
    except Exception:
        return ""
    if resident is None:
        return ""
    try:
        sock = resident.default_socket(REPO)
        if not resident.body_is_up(sock) and not _resident_process_alive():
            return ""
    except Exception:
        return ""

    if not GENESIS_RESIDENT_CLIENT.is_file():
        return ""
    # Deliberately NOT under gpu_lane_lock.sh. The body re-acquires that same
    # /tmp/hawking-gpu-lane.lock per generate, so wrapping the client in it made the
    # body spin against our own wrapper's live pid until the 1500 s deadline, on
    # every call, machine-wide. Generation does not take the lane lock; the body
    # already serializes its own decode.
    cmd = [
        sys.executable,
        str(GENESIS_RESIDENT_CLIENT),
        "propose",
        "--repo",
        str(REPO),
        "--socket",
        str(sock),
        "--max-new-tokens",
        str(GENESIS_PROPOSE_MAX_NEW_TOKENS),
        "--session",
        "parent",
        "--prompt",
        prompt,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=GENESIS_RESIDENT_PROPOSE_TIMEOUT_S,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    text, marker, suffix = proc.stdout.rpartition("\nFALLBACKS:")
    fallback_lines = suffix.strip().splitlines()
    if (
        not marker
        or not fallback_lines
        or not fallback_lines[0].strip().isdigit()
        or int(fallback_lines[0].strip()) != 0
    ):
        return ""
    return text.strip()


def _try_shell_propose(prompt: str) -> str:
    prompt = _genesis_contract_mod().inject_runtime_contract(
        prompt,
        role="parent",
        path=GENESIS_SYSTEM_CONTRACT,
    )
    bin_p, artifact, tokenizer = _genesis_paths()
    if not bin_p.is_file() or not artifact.is_dir():
        return ""
    lock = REPO / "tools" / "gpu_lane_lock.sh"
    cmd = [str(bin_p),
           "--artifact-root", str(artifact), "--tokenizer", str(tokenizer),
           "--prompt", prompt, "--max-new-tokens", str(GENESIS_PROPOSE_MAX_NEW_TOKENS), "--max-seq-len", "8192"]
    # Env-overridden binaries are test/operator stubs; do not wait 90 min on
    # the GPU lock for a script that does not load the body.
    if lock.is_file() and bin_p == _DEFAULT_GENESIS_BIN:
        cmd = [str(lock), "genesis-propose", *cmd]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError):
        return ""
    m = re.search(r"GENERATED_TEXT_VERBATIM:\s*(.*?)\nFALLBACKS:", p.stdout, re.S)
    return m.group(1).strip() if m else ""


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)


def emitted_mechanism(raw: str) -> str:
    """Keep what Genesis emitted; discard what it was merely thinking.

    The recorded mechanism becomes a permanent row in the dedup corpus, so a
    truncated thought recorded as a mechanism is worse than no proposal: it
    burns one of the bounded attempts against a bottleneck AND teaches the
    semantic gate to refuse the real mechanism later as a duplicate of it.

    Under the Genesis Output Law internal work may be deep but only the emitted
    fields are output, so a closed <think> block is stripped. An UNCLOSED one
    means generation hit the token budget mid-reasoning and never reached an
    answer - there is nothing to record, so refuse rather than invent.
    """
    text = _THINK_BLOCK.sub(" ", raw or "").strip()
    if "<think>" in text.lower():
        return ""
    return text


def genesis_proposes(bottleneck: str) -> str:
    """Ask GENESIS itself for the mechanism before a lane is written.

    Production is resident-only: if that body is unavailable, the lane still
    goes out without a proposal instead of cold-loading a second 15 GB body.
    The historical shell path is retained only behind the explicit
    GENESIS_ALLOW_COLD_FALLBACK=1 escape hatch for tests/manual recovery.
    """
    # Verify even when the resident is down and cold fallback is disabled. The
    # organism must not silently continue with conversation memory as authority.
    _genesis_contract_mod().contract_provenance(GENESIS_SYSTEM_CONTRACT)
    # A proposal is GPU work.  Do not queue a multi-minute parent decode behind
    # another protected GPU task: the four logical sessions share one serial
    # body, so doing so starves the very worker topology this daemon is meant to
    # keep alive.  Catalog synthesis and CPU-only work can continue meanwhile.
    if gpu_lane_busy() or protected_gpu_target_active():
        return ""
    prompt = _genesis_prompt(bottleneck)
    text = emitted_mechanism(_try_resident_propose(prompt))
    if text:
        return text
    if os.environ.get("GENESIS_ALLOW_COLD_FALLBACK") == "1":
        return emitted_mechanism(_try_shell_propose(prompt))
    return ""


def dry_synthesis_source(state: dict) -> tuple[dict, str] | None:
    """Return the next catalog mechanism when the measured Qwen queue is dry.

    This is deliberately a replay of the semantic gate's checked catalog, not an
    invented fallback.  It is safe to prefer over a multi-minute resident thought
    pass when the state already has Qwen history but no live Qwen experiment.
    """
    try:
        gate = _mechanism_mod()
        decision = gate.synthesize(
            measured=REPO / "receipts" / "ascent-2026-08-16" / "TOKEN_NS_QWEN38.json",
            attempts=state["targets"],
            include_campaign_exhausted=True,
            model="qwen38",
        )
    except Exception:
        return None
    if getattr(getattr(decision, "verdict", None), "value", None) != "SYNTHESIZE":
        return None
    synthesized = getattr(decision, "synthesized", None)
    if synthesized is None:
        return None
    return ({
        "lane": f"qwen38-synthesis-{synthesized.mechanism_id}",
        "status": "SYNTHESIZED",
        "next_bottleneck": synthesized.target["from_bottleneck"],
        "synthesized": True,
        "synthesis_gate": decision.to_dict(),
    }, synthesized.mechanism)


def generate_targets(
    state: dict,
    harvested: list[dict],
    *,
    proposer: Callable[[str], str] = genesis_proposes,
    allow_synthesis: bool = True,
    resident_proposal_budget: int | None = None,
) -> int:
    """Turn each unseen NEXT_BOTTLENECK into a pending target with a real contract.

    Without this the queue drains and the daemon idles: harvest only FILES finished
    lanes, it does not decide what to try next. This is what keeps work flowing
    while nobody is watching.
    """
    contract_mod = _genesis_contract_mod()
    genesis_contract_block = contract_mod.lane_contract_reference(
        GENESIS_SYSTEM_CONTRACT
    )
    genesis_contract_binding = contract_mod.contract_provenance(
        GENESIS_SYSTEM_CONTRACT
    )
    common = LANES / "_COMMON.md"
    preamble = common.read_text() if common.is_file() else ""
    # .get() throughout: ASCENT_STATE is written by several tools and a target
    # missing a key must not crash the unattended loop.
    existing = {t.get("id") for t in state["targets"]}
    # Dedup against ACTIVE work only, not against history. Keying on every target ever
    # created meant a bottleneck could be attacked exactly once, forever: on 2026-08-16
    # all 105 auto-targets were terminal, weight_addressing had been attacked nine times
    # and never solved, and the loop logged "queue dry" on every tick while the dominant
    # 21.293 ms cost sat untouched. A FAILED attempt is a reason to try a different
    # mechanism, not a reason to stop. Bounded so a permanently-stuck bottleneck cannot
    # spin forever.
    ACTIVE_ = {"pending", "running"}
    seen_bn = {t.get("from_bottleneck") for t in state["targets"]
               if t.get("status") in ACTIVE_}
    # Only named mechanisms are attempts.  A body timeout, an unclosed think
    # block, or a rejected bottleneck-only response contains no hypothesis to
    # test and must not consume the finite research budget.
    attempts: dict[str, int] = {}
    for t in state["targets"]:
        b = t.get("from_bottleneck")
        if b and has_named_mechanism(t):
            attempts[b] = attempts.get(b, 0) + 1
    # Count only targets still in play. This was a LIFETIME cap: it counted every
    # target ever auto-generated, including retained and stale ones, so once 96 had
    # been created the daemon never generated again. On 2026-08-16 it sat at 96/96
    # with 0 pending and 106 phantom "running" targets, and logged "queue dry" on
    # every tick. MAX_GENERATED is meant to bound the ACTIVE POOL, not the history.
    ACTIVE = {"pending", "running"}
    generated = sum(1 for t in state["targets"]
                    if t.get("auto_generated") and t.get("status") in ACTIVE)
    made = 0

    # CPU-only Hawking work is deliberately allowed to overlap a protected
    # Gravity/kernel lane.  Treating every Qwen-labelled task as an active
    # experiment made an HCLI backfill suppress the next catalog kernel trial
    # after the current GPU lane completed—the inverse of the continuity
    # directive's "A + B dominate until >=100 TPS" rule.  Unknown admission
    # classes fail closed as GPU-consuming; only classes explicitly mapped to
    # CPU_ONLY may coexist without holding this part of the scheduler back.
    def is_active_qwen_gpu_target(target: dict) -> bool:
        if target.get("model") != "qwen38" or target.get("status") not in ACTIVE_:
            return False
        resolved = RESOURCE_PROFILE_ALIASES.get(
            str(target.get("resource_class") or "").strip().upper()
        )
        return resolved is None or resolved[0] != "CPU_ONLY"

    has_active_qwen_gpu = any(is_active_qwen_gpu_target(t) for t in state["targets"])
    has_qwen_history = any(t.get("model") == "qwen38" for t in state["targets"])
    # A full resident proposal can occupy the serial body for many minutes.  Once
    # a campaign already has history, dispatch the next evidence-gated catalog item
    # before asking the body for a novel one.  That keeps real measurement moving;
    # when the catalog is exhausted, normal resident proposal remains the path.
    if allow_synthesis and has_qwen_history and not has_active_qwen_gpu:
        dry = dry_synthesis_source(state)
        if dry is not None:
            source, mechanism = dry
            return generate_targets(
                state,
                [source],
                proposer=lambda _b: mechanism,
                allow_synthesis=False,
            )

    if resident_proposal_budget is not None and resident_proposal_budget < 0:
        raise ValueError("resident_proposal_budget must be non-negative or None")
    resident_proposals = 0

    # Qwen3.8 is the sole active vehicle. Sorting keeps its reports ahead of
    # retained science from sealed models, which is filtered below.
    harvested = sorted(harvested, key=lambda h: 0 if model_of(h["lane"]) == "qwen38" else 1)
    for h in harvested:
        if generated + made >= MAX_GENERATED:
            break
        # A failed/empty resident response still consumed a real decode, so it
        # consumes budget too.  Otherwise a batch of empty answers could hold
        # the serial body for an entire unattended pass and starve AgentOS.
        if (
            resident_proposal_budget is not None
            and resident_proposals >= resident_proposal_budget
        ):
            break
        if h.get("needs_manual_review"):
            continue          # no bottleneck text to build a contract from
        bn = h["next_bottleneck"]
        if not bn or bn in seen_bn:
            continue                      # already being worked RIGHT NOW
        n_try = attempts.get(bn, 0)
        if n_try >= MAX_ATTEMPTS_PER_BOTTLENECK:
            continue                      # bounded: stop grinding a stuck target
        model = model_of(h["lane"])
        if model not in ACTIVE_MODELS:
            continue                      # sealed model, weights deleted - unbuildable
        resident_proposals += 1
        proposal = str(proposer(bn) or "").strip()
        if not proposal:
            # Do not write a bottleneck-only lane.  It will be refused later and,
            # worse, used to exhaust the retry budget without a real experiment.
            continue

        tid = next_target_id(model, bn, existing)
        target = {
            "id": tid, "model": model, "hypothesis": bn[:200],
            "mechanism": proposal,
            "target_stage": "auto", "resource_class": "GPU_PROTECTED",
            "probability_of_success": 0.5,
            "representation": "uniform-q4-group64",
            "implementation": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
            "launch_geometry": "tpr64_tg128",
            "command_topology": "one_command_buffer_production_catalog",
            "artifact": "uniform-q4-v1",
            "bytes_per_token": 13_611_663_360,
            "recoverable_ns_per_token": 50_000_000, "density_frontier_gain_ns_equiv": 0,
            "information_gain": 150_000_000, "transfer_value": 50_000_000,
            "experiment_cost": 1.5, "status": "pending",
            "auto_generated": True, "from_bottleneck": bn, "from_lane": h["lane"],
            "genesis_system_contract": genesis_contract_binding,
        }
        if h.get("synthesized"):
            target["synthesized"] = True
            target["synthesis_gate"] = h.get("synthesis_gate")

        # Admit at creation time too.  Deferring this check allowed the same
        # rejected wording to fill the target ledger before the launch loop got
        # a chance to reject it, making a dry queue look like real progress.
        admitted, mechanism_decision = mechanism_admission(target, state)
        if not admitted:
            continue
        target["mechanism_admission"] = mechanism_decision
        genesis_block = (
            f"\n## GENESIS PROPOSED THIS MECHANISM\nThe resident model read this bottleneck "
            f"and proposed the following. Treat it as a HYPOTHESIS to test, never as a "
            f"result, and reject it if the evidence does not support it.\n\n{proposal}\n"
        )

        body = f"""{genesis_contract_block}

{preamble}
{genesis_block}
---
# LANE: {tid}
## AUTO-GENERATED by ascent_daemon from a finished lane's NEXT_BOTTLENECK.
## Class: GPU_PROTECTED for benchmarks. Use ./tools/gpu_lane_lock.sh.

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
- Correctness gate is mandatory for Qwen3.8. For `Say hi.` the greedy 16 ids are
  [248068, 198, 760, 1156, 4777, 6587, 728, 310, 1910, 328, 5834, 1149,
  1061, 369, 264, 1546], with 0 fallbacks. Also run the protected prompt set in
  `receipts/ascent-2026-08-16/QWEN38_COHERENCE_SEAL.json`. Grade against the
  Qwen3.8 ARTIFACT oracle, never the BF16 parent.
- Never weaken a gate, seal, assertion or expected constant to make something pass.
- Label every timing DIRTY_ENGINEERING; other lanes are running.

## Negative science - do NOT re-pay for these
- The old `411.51 GB/s / 97.6%` Qwen3.8 roof is REFUTED. It mixed the wrong
  byte count, whole-token GPU time, and a sequential control with the grouped-Q4
  addressing roof. Do not quote it.
- The landed, provisional honest-roof run defended 13,611,663,360 bytes and
  measured 639.25 GB/s sealed addressing against a 699.57 GB/s single-GEMV
  addressing roof (91.4%). The 401-production-shape catalog measured 530.65 GB/s
  addressing and 505.81 GB/s full-kernel. The box was CPU-contended, so rerun
  cleanly before treating the absolute roof as physical authority.
- Therefore Qwen3.8's current geometry is bandwidth-saturated enough that lower
  active bytes remains a lever, while the catalog/single-GEMV gap proves dispatch
  and kernel topology still have headroom. Pursue BOTH representation and execution.
- The historical 12-component TOKEN_NS ledger force-closed its residual. Complete
  token wall is authority; report any unclosed residual rather than assigning it.
- The generator-residual `shared_r64` net-byte headline is REFUTED: its stated
  3,781,882,584 bytes came from `binary_meanabs_g128`, while the receipt's own
  `shared_r64.residual.coder.q4.bytes` sum is 14,287,109,840 bytes. The measured
  4.049% explained fraction is negative science, not a 1.125-BPW win.
- Fusing tiny kernels into the following GEMV regressed complete-token wall by
  10.68 ms. Cross-token cache reuse showed only a 2.5% hot/cold gap. Four logical
  sessions sharing weights still execute four GEMVs; residency saves model loads,
  not per-session token work.
- Q80 and DeepSeek V4 are sealed models with deleted heavyweight weights. Never
  target, reconstruct, launch, or use their model-specific receipts as Qwen3.8
  performance authority. Qwen3.8's current uniform-Q4 artifact is ACTIVE.

## ACCEPTANCE
Done when the named bottleneck is measured before and after, with >=3 alternating
paired reps and the full spread reported, and the model still generates correctly:
greedy ids unchanged and every silent-fallback counter at 0. A measured NEGATIVE -
the mechanism does not help, with the numbers showing it - is an acceptable
completion. Report the real figure, not a favourable one.

## VERIFY
Build with `cargo build --profile release-fast -p hawking-core` and confirm it exits 0.
Run every GPU-protected measurement under ./tools/gpu_lane_lock.sh <lane> <cmd>;
other lanes share this GPU and an unlocked run corrupts both.
Check no shared-kernel regression with
`cargo test --profile release-fast -p hawking-core --test gk_family_parity`.

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

        target["contract"] = str(path)
        state["targets"].append(target)
        existing.add(tid)
        attempts[bn] = n_try + 1
        seen_bn.add(bn)
        made += 1

    # The mechanism gate already has a bounded, evidence-backed synthesis catalog
    # for this exact situation.  It was written for a dry queue but never wired into
    # the resident daemon, so Genesis could finish a valid negative and then idle.
    # Use it only when there is no active Qwen lane and normal resident generation
    # supplied no admissible target.  The recursive call reuses the standard lane
    # contract writer while disabling another synthesis attempt.
    if made or not allow_synthesis or has_active_qwen_gpu:
        return made
    dry = dry_synthesis_source(state)
    if dry is None:
        return made
    source, mechanism = dry
    return made + generate_targets(
        state,
        [source],
        proposer=lambda _b: mechanism,
        allow_synthesis=False,
    )


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

    # 2. reconcile targets whose lane is gone. ASCENT_STATE marks a target
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
        # New lanes carry their own detached foreground runner PID.  This is
        # more precise than an id search and avoids counting a stale task file
        # as a live optimizer.
        if process_alive(tgt.get("launcher_pid")):
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

    # A candidate already built by the Gravity/kernel fronts has priority over
    # another speculative lane: it is the only work that can prove a better
    # parent exists. The daemon dispatches a separate external controller; it
    # does not parse the request, run a gate, or write a lineage slot itself.
    lifecycle = candidate_lifecycle_status()
    agentos = agentos_turn_status()
    report["candidate_lifecycle"] = lifecycle
    report["agentos_turn"] = agentos
    if lifecycle["status"] != "idle":
        save(STATE, state)
        # A worker turn owns a serial resident decode but does not acquire the
        # protected benchmark lock. Never launch the protected lifecycle
        # controller into that decode; the candidate stays durably pending.
        if agentos["status"] == "running":
            report["launched"] = None
            report["hold"] = "candidate lifecycle waiting for active AgentOS turn"
            return report
        if lifecycle["status"] == "pending":
            priority_hold = govern(snap)
            report["our_live_lanes"] = len(snap.get("our_live_lanes") or [])
            if priority_hold:
                report["launched"] = None
                report["hold"] = f"candidate lifecycle pending: {priority_hold}"
                return report
            lifecycle = dispatch_candidate_lifecycle()
            report["candidate_lifecycle"] = lifecycle
        report["launched"] = None
        report["hold"] = f"candidate lifecycle {lifecycle['status']}"
        return report

    # One worker at a time may use child_a/child_b to implement a bounded
    # source/test step. It is intentionally interleaved with protected work:
    # an already-running capture wins, but a merely pending queue item cannot
    # starve the AgentOS/HCLI front forever. While the turn is live, hold this
    # scheduler so no new protected target can contend for the serial body.
    if agentos["status"] == "running":
        save(STATE, state)
        report["launched"] = None
        report["hold"] = f"AgentOS {agentos.get('worker_id', 'worker')} turn running"
        return report
    if (
        agentos["status"] == "idle"
        and not _candidate_pending()
        and not gpu_lane_busy()
        and not protected_gpu_target_running(state)
        and _resident_process_alive()
        and float(snap.get("disk_free_gib") or 0.0) >= DISK_FLOOR_GIB
    ):
        launched_agentos = dispatch_agentos_turn()
        report["agentos_turn"] = launched_agentos
        if launched_agentos["status"] in {"started", "running"}:
            save(STATE, state)
            report["launched"] = None
            report["hold"] = f"AgentOS {launched_agentos.get('worker_id', 'worker')} turn {launched_agentos['status']}"
            return report

    # 2a. Refill only after liveness reconciliation.  Otherwise a dead target
    # can suppress a retry for five minutes while the parent body wastes time
    # proposing behind work that no longer exists.
    report["generated"] = generate_targets(
        state,
        harvested,
        resident_proposal_budget=MAX_RESIDENT_PROPOSALS_PER_PASS,
    )
    save(STATE, state)
    report["pending"] = sum(1 for t in state["targets"] if t.get("status") == "pending")

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
    # q80/dsv4f: both lost the active fleet and their WEIGHTS ARE DELETED, so a lane
    # targeting either cannot build, measure or promote. They stayed in the pending
    # queue from before the tournament, and the first two passes after the starvation
    # fix both launched a q80 retry - unattended, that is a whole night on a dead model.
    pending = []
    for t in state["targets"]:
        if t.get("status") != "pending":
            continue
        why = target_deauthorised(t)
        if why:
            t["status"] = "deauthorised"
            t["tier1"] = f"excluded: {why}"
            continue
        admitted, mechanism_decision = mechanism_admission(t, state)
        t["mechanism_admission"] = mechanism_decision
        if not admitted:
            t["status"] = "mechanism_refused"
            t["tier1"] = (
                f"mechanism gate {mechanism_decision.get('verdict')}: "
                f"{mechanism_decision.get('reason')}"
            )
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

    try:
        resource_class, profile = resource_profile(target)
    except ValueError as exc:
        target["status"] = "admission_refused"
        target["tier1"] = str(exc)
        save(STATE, state)
        report["launched"] = None
        report["hold"] = str(exc)
        return report
    launcher_pid = launch_target(target, resource_class, profile)
    if launcher_pid is not None:
        report["launched"] = f"{target['id']}@{launcher_pid}"
    else:
        target["status"] = "launch_failed"
        report["launched"] = None
    save(STATE, state)
    return report


def loop() -> int:
    stopfile = REPO / "workspace" / "ops" / "GENESIS_STOP"
    while True:
        # Checked every tick, not only at startup: an unattended loop that can only be
        # halted by killing it leaves lanes mid-flight and their work uncommitted.
        if stopfile.exists():
            print(json.dumps({"stopped": True, "reason": "GENESIS_STOP present"}), flush=True)
            return 0
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
    lifecycle = candidate_lifecycle_status()
    print(
        f"candidate lifecycle {lifecycle['status']} | "
        f"inbox {len(lifecycle.get('inbox') or [])} | active {len(lifecycle.get('active') or [])}"
    )
    for e in q["entries"]:
        if not e["promoted"]:
            print(f"  [{e['disposition']:<28}] {e['lane']:<42} skew={e['skew']}")
            print(f"      next: {e['next_bottleneck'][:110]}")
    return 0


def _selfcheck() -> None:
    """Pin the behaviours that make this safe to leave running."""
    # A truncated thought must never be recorded as a mechanism. It would burn one
    # of the bounded attempts against a bottleneck AND teach the semantic dedup gate
    # to later refuse the real mechanism as a duplicate of a half-finished sentence.
    assert emitted_mechanism("<think>reasoning</think>\nMECHANISM: batch") == "MECHANISM: batch"
    assert emitted_mechanism("<think>ran out of budget mid-thought") == ""
    assert emitted_mechanism("<think>a</think> MECHANISM: x <think>b") == ""
    assert emitted_mechanism("MECHANISM: persistent kernel") == "MECHANISM: persistent kernel"
    assert emitted_mechanism("") == ""

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
        global TASKS, LANES
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
    # and refuse to run away. Use a temporary contract directory and a pure proposer:
    # selfcheck must never load a 15 GB model or touch the live queue.
    assert slug("host.expert_slab_io 415126416 ns/token") == "host-expert-slab-io"
    with tempfile.TemporaryDirectory() as td:
        saved_lanes = LANES
        LANES = Path(td) / "lanes"
        try:
            propose = lambda bottleneck: f"test mechanism for {bottleneck}"
            st = {"targets": []}
            h = [
                {"lane": "qwen38-x-1", "status": "SHIPPED", "next_bottleneck": "host.foo 1 ns/token"},
                {"lane": "qwen38-y-1", "status": "SHIPPED", "next_bottleneck": "metal.bar 2 ns/token"},
            ]
            assert generate_targets(st, h, proposer=propose) == 2, \
                "must create a target per new bottleneck"
            assert generate_targets(st, h, proposer=propose) == 0, \
                "must dedupe on repeat passes"
            assert {t["model"] for t in st["targets"]} == {"qwen38"}
            assert all(t["status"] == "pending" and t["auto_generated"] for t in st["targets"])
            # A missing resident answer is not a research attempt.  It must leave
            # the state untouched so a later healthy body can answer the same wall.
            empty = {"targets": []}
            assert generate_targets(
                empty, h[:1], proposer=lambda _b: "", allow_synthesis=False
            ) == 0
            assert not empty["targets"], "empty proposals must not create placeholders"
            # Historical placeholder rows may have occupied every visible retry id,
            # but they do not exhaust the named-mechanism budget.  A real retry gets
            # a fresh immutable id rather than colliding with an old blank row.
            wall = "weight_addressing 1 ns/token"
            historical = {
                "targets": [
                    {
                        "id": "auto-qwen38-weight-addressing"
                        if i == 1 else f"auto-qwen38-weight-addressing-try{i}",
                        "from_bottleneck": wall,
                        "mechanism": "",
                        "status": "mechanism_refused",
                    }
                    for i in range(1, MAX_ATTEMPTS_PER_BOTTLENECK + 1)
                ]
            }
            assert generate_targets(
                historical,
                [{"lane": "qwen38-retry", "status": "SHIPPED", "next_bottleneck": wall}],
                proposer=lambda _b: "test mechanism for a new weight representation",
                allow_synthesis=False,
            ) == 1
            assert historical["targets"][-1]["id"] == "auto-qwen38-weight-addressing-try13"
            # If normal generation is dry, the existing semantic catalog must
            # supply its next unused evidence-backed mechanism instead of idling.
            dry = {"targets": []}
            assert generate_targets(dry, [], proposer=lambda _b: "") == 1
            assert dry["targets"][0].get("synthesized") is True
            assert "per layer and per head" in dry["targets"][0]["mechanism"].lower()
            # MAX_GENERATED bounds the ACTIVE pool, so the cap test must fill it with
            # ACTIVE targets. A pool of finished ones must NOT block new work.
            st["targets"] = [
                {"auto_generated": True, "from_bottleneck": f"b{i}", "status": "pending"}
                for i in range(MAX_GENERATED)
            ]
            assert generate_targets(
                st,
                [{"lane": "qwen38-z-1", "status": "SHIPPED", "next_bottleneck": "brand new wall"}],
                proposer=propose,
            ) == 0, "must stop at MAX_GENERATED when the ACTIVE pool is full"
            st["targets"] = [
                {"auto_generated": True, "from_bottleneck": f"c{i}", "status": "stale_no_process"}
                for i in range(MAX_GENERATED)
            ]
            assert generate_targets(
                st,
                [{"lane": "qwen38-z-2", "status": "SHIPPED", "next_bottleneck": "another new wall"}],
                proposer=propose,
            ) == 1, "a pool of FINISHED targets must not block new generation"
        finally:
            LANES = saved_lanes

    assert resource_profile({"id": "cpu", "resource_class": "CPU_ONLY"}) == (
        "CPU_ONLY", "maximum"
    )
    assert resource_profile({"id": "gpu", "resource_class": "GPU_PROTECTED"}) == (
        "GPU_PROTECTED", "gate"
    )
    assert resource_profile({"id": "legacy", "resource_class": "GPU_EXCLUSIVE"}) == (
        "GPU_PROTECTED", "gate"
    )
    try:
        resource_profile({"id": "missing"})
        raise AssertionError("missing resource class must fail at admission")
    except ValueError:
        pass
    assert target_deauthorised({"id": "innocent", "model": "q80"}) == "sealed model q80"
    assert target_deauthorised({"id": "innocent", "model": "dsv4f"}) == "sealed model dsv4f"
    assert target_deauthorised({"id": "qwen38-work", "model": "qwen38"}) is None
    allowed, decision = mechanism_admission(
        {
            "id": "new-mechanism",
            "model": "qwen38",
            "mechanism": "batch production-shaped GEMV dispatch metadata into an indirect command table",
            "from_bottleneck": "weight_addressing 1 ns",
        },
        {"targets": []},
    )
    assert allowed and decision["verdict"] == "ALLOW", decision
    allowed, decision = mechanism_admission(
        {
            "id": "duplicate-mechanism",
            "model": "qwen38",
            "mechanism": "fuse small Metal kernels into the next GEMV",
            "from_bottleneck": "weight_addressing 1 ns",
        },
        {"targets": []},
    )
    assert not allowed and decision["verdict"] == "REFUSE", decision
    allowed, decision = mechanism_admission(
        {"id": "bottleneck-only", "model": "qwen38", "from_bottleneck": "weight_addressing"},
        {"targets": []},
    )
    assert not allowed and decision["gate"] == "named_mechanism_required", decision

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
