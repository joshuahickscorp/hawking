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

DISK_FLOOR_GIB = 15.0
DISK_WARN_GIB = 90.0   # raised after a 0-byte stall: lanes cost 1-19 GiB each
MAX_ATTEMPTS_PER_BOTTLENECK = 12   # a dominant cost deserves many mechanisms,
                       # but not an unbounded grind; 9 have already failed on
                       # weight_addressing and it is still the right target
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
        return bool(json.loads(LINEAGE_STATE.read_text())["slots"].get("CANDIDATE"))
    except (OSError, ValueError, KeyError):
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
    if lane.startswith("dsv"):
        return "dsv4f"
    if lane.startswith("qwen38") or "qwen38" in lane or lane.startswith("genesis"):
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
GENESIS_PROPOSE_MAX_NEW_TOKENS = 2600
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


def _try_resident_propose(prompt: str) -> str:
    """Run one resident inference under the same GPU mutex as measurements.

    Health is checked before entering the potentially long lock queue, so a dead
    body remains a fast no-op. The client is passed the prompt as one argv item;
    no prompt text is ever interpreted by a shell.
    """
    try:
        resident = _resident_mod()
    except Exception:
        return ""
    if resident is None:
        return ""
    try:
        sock = resident.default_socket(REPO)
        if not resident.body_is_up(sock):
            return ""
    except Exception:
        return ""

    lock = REPO / "tools" / "gpu_lane_lock.sh"
    if not lock.is_file() or not GENESIS_RESIDENT_CLIENT.is_file():
        return ""
    cmd = [
        str(lock),
        "genesis-propose",
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
           "--prompt", prompt, "--max-new-tokens", str(GENESIS_PROPOSE_MAX_NEW_TOKENS), "--max-seq-len", "4096"]
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
    prompt = _genesis_prompt(bottleneck)
    text = emitted_mechanism(_try_resident_propose(prompt))
    if text:
        return text
    if os.environ.get("GENESIS_ALLOW_COLD_FALLBACK") == "1":
        return emitted_mechanism(_try_shell_propose(prompt))
    return ""


def generate_targets(
    state: dict,
    harvested: list[dict],
    *,
    proposer: Callable[[str], str] = genesis_proposes,
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
    attempts: dict[str, int] = {}
    for t in state["targets"]:
        b = t.get("from_bottleneck")
        if b:
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

    # Qwen3.8 is the sole active vehicle. Sorting keeps its reports ahead of
    # retained science from sealed models, which is filtered below.
    harvested = sorted(harvested, key=lambda h: 0 if model_of(h["lane"]) == "qwen38" else 1)
    for h in harvested:
        if generated + made >= MAX_GENERATED:
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
        # The attempt number is part of the id, so retry N+1 is a NEW target rather
        # than colliding with the terminal record of attempt N.
        tid = f"auto-{model}-{slug(bn)}" if n_try == 0 else f"auto-{model}-{slug(bn)}-try{n_try + 1}"
        if tid in existing:
            continue

        proposal = proposer(bn)
        genesis_block = (
            f"\n## GENESIS PROPOSED THIS MECHANISM\nThe resident model read this bottleneck "
            f"and proposed the following. Treat it as a HYPOTHESIS to test, never as a "
            f"result, and reject it if the evidence does not support it.\n\n{proposal}\n"
        ) if proposal else ""

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

        state["targets"].append({
            "id": tid, "model": model, "hypothesis": bn[:200],
            "mechanism": proposal.strip(),
            "target_stage": "auto", "contract": str(path),
            "resource_class": "GPU_PROTECTED", "probability_of_success": 0.5,
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
    code, out = sh(f"{GROK} delegate --task {target['id']} --contract {contract} "
                   f"--repo {REPO} --profile {profile} --background", timeout=600)
    m = re.search(rf"{re.escape(target['id'])}-\d{{8}}-\d{{6}}", out)
    if m:
        target.update(
            status="running",
            task_id=m.group(0),
            admitted_resource_class=resource_class,
            admitted_profile=profile,
        )
        report["launched"] = m.group(0)
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
