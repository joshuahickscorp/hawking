#!/usr/bin/env python3
"""Drive HCLI through a competence ladder, unattended.

Claude teaches by writing the goal and reading the receipt. HCLI does the work.
This driver only keeps the loop turning: it restarts a resident that has
stopped, hands over the next goal, waits for the mission to reach a terminal
phase, records what happened, and decides whether to advance or retry.

Two things are load-bearing and were learned the hard way:

  * ONE IMPERATIVE SENTENCE per goal. `goal_tokenizer` emits one WorkUnit per
    sentence, so explanatory prose becomes fake obligations -- a five-sentence
    goal produced eleven units, six of them sentence fragments.

  * SMALL EDITS. The measured model weakness is long verbatim code inside a
    JSON string: a 2-line source patch was correct while a 15-line test rewrite
    in the same reply dropped three closing parens.

Run:  python3 tools/hcli_school.py [--once] [--start-level N]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / ".hcli" / "mission" / "state.json"
RESIDENT = REPO / ".hcli" / "resident" / "state.json"
LOG = REPO / ".hcli" / "school" / "log.jsonl"
GOALS = REPO / ".hcli" / "school" / "goals"

#: One imperative sentence each. Levels follow the campaign ladder: read,
#: patch, patch+test, author a regression test, repair a failure, optimise,
#: benchmark, then choose its own work.
LADDER_FILE = REPO / ".hcli" / "school" / "ladder.json"

#: Fallback when no ladder file is present. The file is read fresh before every
#: cycle, so goals can be added or reordered while the loop is running.
LADDER = [
    (2, "In hcli/tool_registry.py, make a whole-file fs.read result also report total_lines, the way the windowed read already does."),
    (3, "In hcli/resources.py, add a one-line docstring to pid_is_alive saying it reaps a zombie before testing liveness."),
    (4, "Write hcli/tests/test_pid_is_alive_contract.py asserting pid_is_alive returns False for a pid of 0, -1 and None."),
    (5, "In hcli/context_budget.py, report per_request_ctx and usable_input_tokens in the ContextBudget repr so a budget refusal names its own numbers."),
    (6, "In hcli/engine.py, record prefill_tokens_stepped and prefix_reused_tokens as a single reuse_fraction field on each model_call entry."),
    (7, "Write hcli/tests/test_prefix_reuse_fraction.py asserting reuse_fraction is prefix_reused_tokens divided by prompt_tokens and is None when either is absent."),
    (8, "Read .hcli/school/log.jsonl and name in one sentence the single change to HCLI that would most raise effective_prompt_tps."),
]

#: Appended when a level is retried, so the next attempt carries what failed.
RETRY_HINT = (
    " Change at most five lines of source and add at most one test function of"
    " three lines or fewer."
)

TERMINAL = {"completed", "failed", "cancelled", "evacuated"}
#: Below this, pause rather than write. A long unattended run generates
#: receipts and logs steadily, and filling the disk would take the daemon down
#: with it. Pause and say so -- never delete, because everything here is
#: evidence somebody may want to read.
MIN_FREE_BYTES = 20 * 1024**3
MAX_RETRIES = 2
POLL_S = 20
#: Bound on INACTIVITY, not on total time. A flat wall cut a mission off at
#: 2400 s while it was still advancing through its repair chain -- two units
#: failed, one ready, one pending -- and the driver then evacuated work that
#: was making progress. Same error as a fixed sleep instead of a rendezvous.
#: A repair round costs minutes at ~25 prompt tok/s, so allow a long quiet
#: gap; the hard cap only stops a mission that is genuinely wedged.
#: A model call costs ~200 s, so 15 minutes is four calls of headroom. 30
#: minutes was a wedged mission burning half an hour of a run that only gets so
#: many cycles in a night.
IDLE_TIMEOUT_S = 900
HARD_CAP_S = 10800


def run(*argv: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "hcli.agentos.resident", *argv],
        cwd=REPO, capture_output=True, text=True, timeout=timeout,
    )


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def newest_receipt_mtime() -> float:
    files = sorted((REPO / ".hcli" / "receipts").glob("*.json"),
                   key=lambda p: p.stat().st_mtime)
    return files[-1].stat().st_mtime if files else 0.0


def newest_receipt() -> dict:
    files = sorted((REPO / ".hcli" / "receipts").glob("*.json"),
                   key=lambda p: p.stat().st_mtime)
    return read_json(files[-1]) if files else {}


def verdict() -> dict:
    proc = run("verdict", timeout=120)
    out = {}
    for line in (proc.stdout or "").splitlines():
        if "measured:" in line:
            for pair in line.split("measured:", 1)[1].split():
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = v
        parts = line.split()
        if len(parts) >= 2 and parts[0] in {"PASS", "FAIL", "UNKNOWN"}:
            out.setdefault("criteria", {})[parts[1]] = parts[0]
    return out


def start(goal: str, level: int, attempt: int) -> None:
    GOALS.mkdir(parents=True, exist_ok=True)
    path = GOALS / f"L{level}.{attempt}.txt"
    path.write_text(goal.rstrip() + "\n", encoding="utf-8")
    run("replace", "--goal-file", str(path), timeout=600)


def _last_activity() -> float:
    """When the mission last produced anything: a receipt or a state write."""
    newest = 0.0
    for path in (REPO / ".hcli" / "receipts").glob("*.json"):
        newest = max(newest, path.stat().st_mtime)
    for path in (STATE, REPO / ".hcli" / "mission" / "events.jsonl"):
        if path.exists():
            newest = max(newest, path.stat().st_mtime)
    return newest


def wait_for_mission(started: float, previous_mission: str) -> str:
    """Wait for THIS mission to finish.

    Two things went wrong reading the phase naively, and both made the loop
    spin without doing any work:

      * the phase read right after `replace` is the PREVIOUS mission's, which
        is `evacuated` -- a terminal phase -- so the cycle "finished" in 31 s.
      * `_last_activity()` reads receipt mtimes that are stale at cycle start,
        so the inactivity bound fired immediately and the cycle "finished" in
        0.3 s.

    So: wait for a new mission id first, and treat the cycle's own start as
    activity until the mission produces some.
    """
    hard_stop = started + HARD_CAP_S
    # Phase one: the new mission must exist before its phase means anything.
    while time.time() < hard_stop:
        state = read_json(STATE)
        # The field is `id`. `mission_id` does not exist in mission state and
        # reading it returned None every time, so "a new mission started" was
        # never true and the driver watched the OLD mission's terminal phase.
        if str(state.get("id") or "") not in ("", previous_mission):
            break
        if time.time() - started > IDLE_TIMEOUT_S:
            return "never_started"
        time.sleep(POLL_S)

    while time.time() < hard_stop:
        phase = str(read_json(STATE).get("phase") or "")
        if phase in TERMINAL:
            return phase
        if str(read_json(RESIDENT).get("state") or "") == "FAILED":
            return "failed"
        if time.time() - max(_last_activity(), started) > IDLE_TIMEOUT_S:
            return "idle"
        time.sleep(POLL_S)
    return "timeout"


def landed(receipt: dict) -> bool:
    """Did the goal actually change the tree?

    `kind: answer` carries operations that are never applied, and a unit can
    complete without its deliverable existing. Only a mutation that was not
    rolled back and passed validation is a landed contribution.
    """
    return (
        receipt.get("kind") == "mutation"
        and receipt.get("status") == "completed"
        and not receipt.get("rolled_back")
        and bool((receipt.get("validation") or {}).get("ok"))
    )


def free_bytes() -> int:
    try:
        st = os.statvfs(REPO)
        return st.f_bavail * st.f_frsize
    except OSError:
        return MIN_FREE_BYTES + 1


def wait_for_disk() -> None:
    """Hold while the disk is too full to run safely."""
    warned = False
    while free_bytes() < MIN_FREE_BYTES:
        if not warned:
            gb = free_bytes() / 1024**3
            print(f"PAUSED: {gb:.1f} GB free, below the {MIN_FREE_BYTES/1024**3:.0f} GB floor", flush=True)
            record({"phase": "paused_low_disk", "free_gb": round(gb, 1), "landed": False})
            warned = True
        time.sleep(300)
    if warned:
        print("resumed: disk recovered", flush=True)


def record(row: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def cycle(level: int, goal: str, attempt: int) -> dict:
    wait_for_disk()
    started = time.time()
    previous_mission = str(read_json(STATE).get("id") or "")
    start(goal, level, attempt)
    phase = wait_for_mission(started, previous_mission)
    receipt = newest_receipt()
    # A receipt older than this cycle belongs to the PREVIOUS one. Reporting it
    # as this cycle's result is how three cycles logged identical operations
    # they never performed.
    if receipt and newest_receipt_mtime() < started:
        receipt = {}
    row = {
        "level": level,
        "attempt": attempt,
        "phase": phase,
        "goal": goal,
        "kind": receipt.get("kind"),
        "status": receipt.get("status"),
        "landed": landed(receipt),
        "rolled_back": receipt.get("rolled_back"),
        "tests": receipt.get("tests"),
        "operations": [(o.get("op"), o.get("path")) for o in (receipt.get("operations") or [])],
        "error": str(receipt.get("error") or "")[:400],
        "model_calls": len(receipt.get("model_calls") or []),
        "grammar_enforced": [c.get("grammar_enforced") for c in (receipt.get("model_calls") or [])][:4],
        "verdict": verdict(),
        "wall_s": round(time.time() - started, 1),
    }
    record(row)
    return row


def ladder() -> list:
    """Read the ladder fresh each cycle, so it can be steered while running."""
    try:
        rows = json.loads(LADDER_FILE.read_text())
        out = [(int(r["level"]), str(r["goal"])) for r in rows if r.get("goal")]
        return out or LADDER
    except (OSError, ValueError, KeyError, TypeError):
        return LADDER


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--start-level", type=int, default=2)
    args = ap.parse_args()

    rungs = ladder()
    index = next((i for i, (lvl, _) in enumerate(rungs) if lvl >= args.start_level), 0)
    passes = 0
    while True:
        rungs = ladder()
        if not rungs:
            time.sleep(POLL_S)
            continue
        if index >= len(rungs):
            # The ladder is a loop, not a queue. An unattended run that exits
            # when the last rung is reached stops improving at exactly the
            # point the rungs start being about HCLI rather than the harness.
            # New rungs can be appended to the ladder file while this runs.
            index = 0
            passes += 1
            print(f"--- ladder pass {passes} complete, restarting", flush=True)
            if args.once:
                return 0
        level, goal = rungs[index]
        last_error = None
        for attempt in range(1, MAX_RETRIES + 2):
            text = goal if attempt == 1 else goal + RETRY_HINT
            try:
                row = cycle(level, text, attempt)
            except Exception as exc:  # noqa: BLE001 - unattended: never die
                # One bad cycle must not end the run. Record it and carry on;
                # a driver that exits on an exception is a driver that is not
                # running by morning.
                record({
                    "level": level, "attempt": attempt, "phase": "driver_error",
                    "goal": text, "landed": False,
                    "error": f"{type(exc).__name__}: {exc}"[:400],
                })
                print(f"L{level} attempt {attempt}: DRIVER ERROR {type(exc).__name__}: {exc}", flush=True)
                time.sleep(POLL_S)
                continue
            print(
                f"L{level} attempt {attempt}: phase={row['phase']} "
                f"kind={row['kind']} landed={row['landed']} "
                f"calls={row['model_calls']} {row['wall_s']}s",
                flush=True,
            )
            if row["landed"]:
                break
            # A retry that reproduces the SAME error is a cycle spent to learn
            # nothing. Three attempts per rung is 3x the wall clock for one
            # result when the failure is deterministic, and a night only holds
            # so many cycles. Retry a failure that MOVED; abandon one that did
            # not.
            error = (row.get("error") or "")[:200]
            if attempt > 1 and error and error == last_error:
                print(f"L{level}: identical failure on retry, moving on", flush=True)
                break
            last_error = error
        index += 1
        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
