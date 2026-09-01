#!/usr/bin/env python3
"""S033: the resident owns the frontier and Claude watches.

The loop, once per iteration:

    recover the durable mission kernel from disk
        -> assemble a bounded context pack (mission, evidence delta, scars)
        -> ask the resident for a structured belief update and next work
        -> VALIDATE what it asked for against authority and resources
        -> execute what the deterministic layer knows how to run
        -> ingest the receipt, update the kernel
        -> repeat

No accumulating chat session: every call is rebuilt from the kernel and then
exits. That is what killed the earlier multi-turn attempt, where a truncated
64-token echo poisoned every following turn.

WHAT CLAUDE DOES NOT DO HERE: choose the hypothesis, choose the experiment, or
interpret the result. The kernel carries evidence; the resident draws the
conclusion. S033 §2 and §28.

    python3 tools/future/hcli_sovereign.py --init
    python3 tools/future/hcli_sovereign.py --run --minutes 10
    python3 tools/future/hcli_sovereign.py --build
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/hcli_sovereign.py"
RECEIPT_NAME = "HCLI_SOVEREIGN.json"
KERNEL_REL = "receipts/future/HCLI_MISSION_KERNEL.json"
LOG_REL = "receipts/future/_HCLI_SOVEREIGN_log.jsonl"

RESIDENT_MODE = "ACTIVE_ORCHESTRATOR"
MAX_NEW_TOKENS = 800

# The deterministic layer executes exactly these. Anything else the resident
# asks for is recorded as UNSUPPORTED_REQUEST - which is signal about what the
# harness is missing, not a failure of the resident.
MAX_WORK_PER_TURN = 3
EXECUTABLE = {
    "PERTURB": "tools/future/perturbation_workunit.py",
    "COMPUTE": "deterministic arithmetic over existing receipts",
    "READ_RECEIPT": "read a receipt the kernel already names",
}


class SovereignRefused(RuntimeError):
    """The kernel is missing or malformed; the loop will not invent state."""


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def kernel_path() -> Path:
    return REPO / KERNEL_REL


def load_kernel() -> dict[str, Any]:
    p = kernel_path()
    if not p.is_file():
        raise SovereignRefused(
            f"{KERNEL_REL} is not on disk. The mission kernel IS the resident's "
            "memory; without it a reasoning call would be a fresh chatbot turn "
            "rather than a continuing mission. Run --init."
        )
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        # A corrupt kernel used to raise JSONDecodeError out of --run, so the
        # loop crash-looped instead of refusing once and saying what to do.
        raise SovereignRefused(
            f"{KERNEL_REL} is on disk but is not valid JSON ({exc}). The mission "
            "kernel IS the resident's memory - refusing rather than continuing "
            "with none. Restore it, or --init a fresh one, which will lose "
            "every hypothesis verdict and scar it held."
        ) from exc


def save_kernel(k: dict[str, Any]) -> None:
    """Atomic. The mission kernel is the resident's ONLY memory.

    write_text truncates and then writes, so a crash mid-write left a truncated
    file and the previous kernel was gone. hcli/persist.py has done temp +
    fsync + os.replace since long before this module existed and was simply not
    used here.
    """
    k["updated_unix"] = time.time()
    p = kernel_path()
    tmp = p.with_suffix(p.suffix + ".tmp")
    body = json.dumps(k, indent=1, sort_keys=True) + "\n"
    with tmp.open("w") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def _receipt(rel: str) -> dict[str, Any] | None:
    p = REPO / rel
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def init_kernel() -> dict[str, Any]:
    """Durable mission state, built from receipts rather than typed."""
    gap = _receipt("receipts/future/GAP_LEDGER_60.json") or {}
    floor = _receipt("receipts/future/REPRESENTATION_FLOOR.json") or {}
    probe = _receipt("receipts/future/FUNCTIONAL_ROLE_PROBE.json") or {}
    mix = _receipt("../../noetic/NOETIC_PARENT_A/MIX_REPORT.json") or {}
    live = (gap.get("live") or {})

    k = {
        "schema": "hawking.future.hcli_mission_kernel.v1",
        "resident_mode": RESIDENT_MODE,
        "objective": (
            "Discover the lowest capability-preserving complete EBPW for the "
            "current dense resident. First milestone: 2.0 complete EBPW or less."
        ),
        "frontier": "SUB2_EBPW",
        "measured_state": {
            "complete_bpw": (floor.get("floor") or {}).get("incumbent_bpw"),
            "payload_bytes": (floor.get("floor") or {}).get("incumbent_bytes"),
            # The ledger's basis is GPU now (G132), so calling these "wall"
            # would tell the resident the wrong thing about what it is looking
            # at. The basis travels with the numbers.
            "token_ms": live.get("ms_per_token"),
            "tps": live.get("tps"),
            "basis": live.get("basis"),
            "conventional_floor_bpw_if_every_untested_move_worked":
                (floor.get("floor") or {}).get("if_every_untested_move_worked_bpw"),
            "source": [
                "receipts/future/REPRESENTATION_FLOOR.json",
                "receipts/future/GAP_LEDGER_60.json",
            ],
        },
        "hypotheses": [
            {
                "id": "H1.gate_up_mutual_information",
                "proposer": "resident",
                "claim": "the up code is partly predictable from the gate code "
                         "at the same position, so joint coding beats marginal",
                "verdict": "REFUTED",
                "evidence": "mutual information 0.00059 bits per weight pair "
                            "across layers 0/21/42/63",
            },
            {
                "id": "H2.functional_role_gate_dominant",
                "proposer": "resident",
                "claim": "the SwiGLU gate is control and deserves literal "
                         "storage; up and down are linear bulk and can be "
                         "generated or shared",
                "verdict": "REFUTED",
                "evidence": "receipts/future/FUNCTIONAL_ROLE_PROBE.json - gate "
                            "never exceeds 1.31x up per matched element, and "
                            "down is most sensitive at 9 of 12 points",
            },
        ],
        "observations": [
            {
                "id": "O1.local_robustness",
                "text": (
                    "zeroing 40% of a tensor's output rows - about 35.6 million "
                    "elements - moves the hidden state two layers later by "
                    f"{(probe.get('robustness') or {}).get('worst_damage')} of cosine"
                ),
                "source": "receipts/future/FUNCTIONAL_ROLE_PROBE.json",
                "interpretation": "NONE RECORDED - this is evidence, not a conclusion",
            },
        ],
        "scars": [
            "MLP 2-bit codes carry ~1.87 bits entropy per 2 stored bits",
            "affine groups 256 and 1024: CAPABILITY REFUTED",
            "shared linear low-rank across MLP factors: REFUTED at relative L2 0.9",
            "auxiliary broadcast bytes bill 0.000 ms/GB - smaller, not faster",
            "only weight codes bill time, at 0.547282 ms/GB",
            "TOP_LEVEL_TOKEN_REORDERING_HAS_NO_CURRENT_SLACK",
            "REMOVING_ONE_OP_CLASS_IS_NOT_A_LEVER_WHEN_FOUR_SHARE_THE_COST",
        ],
        "scars_bind_methods_not_goals": (
            "a scar constrains the method it measured, at the scope it "
            "measured. It does not forbid the objective."
        ),
        "authority": {
            "may": [
                "propose hypotheses and experiments",
                "request perturbation experiments on any MLP tensor and layer",
                "request deterministic computation",
                "request a receipt be read",
            ],
            "may_not": [
                "delete or overwrite any artifact",
                "modify the canonical worktree directly",
                "claim a hardware number it did not receive from a tool",
            ],
        },
        "executable_work_types": sorted(EXECUTABLE),
        "iterations": [],
        "created_unix": time.time(),
    }
    if mix:
        k["measured_state"]["mlp_elements"] = mix.get("mlp_elements")
        k["measured_state"]["storage_bpw"] = mix.get("storage_bpw")
    save_kernel(k)
    return k


def refresh_measured_state(k: dict[str, Any]) -> dict[str, Any]:
    """Re-read the body's numbers into a LIVING kernel, changing nothing else.

    The kernel is durable state - hypotheses, scars, iterations - so --init is
    not the way to pick up a new baseline: it would discard everything the
    resident has learned. But leaving it stale is worse than either, because
    measured_state is what the context pack tells the resident about the body it
    is reasoning over, and it said 27.2896 ms / 36.644 TPS for hours after the
    promotion measured 21.9464 / 45.566.
    """
    from tools.future import gap_ledger_60 as gl
    live = gl.live()
    ms = k.setdefault("measured_state", {})
    before = {kk: ms.get(kk) for kk in
              ("token_ms", "tps", "basis", "wall_ms_per_token", "wall_tps")}
    ms["token_ms"] = live.get("ms_per_token")
    ms["tps"] = live.get("tps")
    ms["basis"] = live.get("basis")
    # The old field names are REMOVED, not left beside the new ones. Two names
    # for one quantity is how a reader picks the stale one.
    ms.pop("wall_ms_per_token", None)
    ms.pop("wall_tps", None)
    ms["refreshed_unix"] = time.time()
    ms["refreshed_from"] = live.get("source")
    return {"before": before,
            "after": {kk: ms.get(kk) for kk in ("token_ms", "tps", "basis")}}


def context_pack(k: dict[str, Any], *, terse: bool = False) -> str:
    """Bounded. Mission + evidence + the work vocabulary. No conclusions.

    SHORT ON PURPOSE. A 2667-character pack made this body restate the pack
    instead of answering it - fourteen consecutive turns of echo, byte-identical
    under greedy decoding because the pack never changed. The schema goes FIRST
    and LAST; evidence is compressed to one line each; the scar list is a count
    plus the two that bear on the objective. Measured: the turn that worked was
    a 1380-character reply to a pack under 1500 characters.
    """
    ms = k["measured_state"]
    dead = ", ".join(h["id"].split(".", 1)[-1] for h in k["hypotheses"]
                     if h["verdict"] == "REFUTED")
    obs = k["observations"][0]["text"] if k["observations"] else ""
    last = ""
    if k["iterations"]:
        rs = (k["iterations"][-1].get("results_summary") or [])[:2]
        last = "LAST TURN: " + "; ".join(rs)
    tried = k.get("tried_params") or []
    avoid = ("ALREADY RUN (pick different params): "
             + "; ".join(tried[-6:])) if tried else ""
    # The resident's OWN last hypotheses were recorded on the iteration and
    # never shown back to it. A scientist that cannot see what it proposed last
    # turn cannot carry an investigation across turns - it can only restart one.
    mine = ""
    live = [h for h in (k.get("live_hypotheses") or []) if isinstance(h, dict)]
    if not live and k.get("iterations"):
        prev = (k["iterations"][-1] or {}).get("live_hypotheses")
        live = [h for h in (prev or []) if isinstance(h, dict)]
    if live:
        # BOUNDED. Feeding hypotheses back is what lets an investigation cross
        # turns, but an unbounded feed grows the pack every turn and this body
        # degenerates with length - the pack that worked was under 1600 chars.
        # Two hypotheses, each claim clipped, is the feed; the full text lives
        # on the iteration record where nothing has to re-read it.
        mine = ("YOUR LAST HYPOTHESES (advance or kill them): "
                + "; ".join(f"{str(h.get('id'))[:40]}: "
                            f"{str(h.get('claim'))[:110]}"
                            for h in live[-2:]))
    # IDENTICAL_REPLY_LOOP: under greedy decoding a byte-identical pack returns
    # a byte-identical reply. After one failed-parse turn LAST TURN froze to a
    # constant and tried_params stopped changing, so the pack never moved and
    # the body repeated itself for fourteen turns. The turn number is one token
    # and makes repetition impossible to reach by construction.
    turn = f"TURN {len(k.get('iterations') or []) + 1}."

    schema = (
        'Reply with ONLY this JSON, no prose:\n'
        '{"belief_update":"one sentence",'
        # The placeholder used to be "x" and the body COPIED IT VERBATIM: 16 of
        # 26 recorded hypothesis ids were literally "x", plus 2 of "y". An id
        # that is not distinct makes the whole hypothesis register unjoinable -
        # you cannot tell which turn advanced which claim. A placeholder that
        # reads as an instruction is harder to copy than one that reads as a
        # value.
        '"live_hypotheses":[{"id":"NAME_THIS_CLAIM",'
        '"claim":"one sentence",'
        '"cheapest_falsifier":"one sentence"}],'
        '"selected_work":[{"type":"PERTURB","params":'
        '{"tensor":"gate|up|down","layer":0,"side":"rows|cols","fraction":0.5},'
        '"why":"one sentence"}],"escalation_needed":false}'
    )
    if terse:
        return (
            schema
            + "\n\nYou choose the next experiment. PERTURB damages part of an "
              "MLP tensor and measures the effect two layers later.\n"
            + (avoid + "\n" if avoid else "")
            + (mine + "\n" if mine else "")
            + f"\n{turn} Output the JSON now."
        )
    return f"""{schema}

You are the scientific orchestrator. You choose what to investigate; nobody will tell you.

GOAL: lowest capability-preserving complete EBPW. Milestone 2.0 or less.
NOW: {ms.get('complete_bpw')} BPW. Any conventional encoding bottoms out at {ms.get('conventional_floor_bpw_if_every_untested_move_worked')} BPW, so the goal needs something other than a better code.

DEAD (do not re-propose): {dead}
EVIDENCE: {obs}
SCARS: {len(k['scars'])} recorded; entropy coding and larger groups are spent.
{last}
{avoid}
{mine}
{turn}

PERTURB damages part of an MLP tensor (gate|up|down, layer 0-63, rows|cols, fraction 0.01-0.95) and measures the hidden state two layers later.

{schema}"""


def validate(obj: dict[str, Any] | None) -> dict[str, Any]:
    """Authority and shape check. The resident proposes; this decides."""
    if not isinstance(obj, dict):
        # Same shape on every path. An earlier version omitted the counts here
        # and the loop crashed on the first unparsed reply - the harness failing,
        # not the resident.
        return {"ok": False, "why": "reply did not parse as a JSON object",
                "accepted": [], "rejected": [], "n_accepted": 0, "n_rejected": 0}
    accepted, rejected = [], []
    # The body has returned selected_work as a dict, not a list. Slicing a dict
    # raised KeyError and killed the loop - the third harness crash caused by
    # assuming a shape the model is free not to produce. Coerce, never assume.
    sel = obj.get("selected_work")
    if isinstance(sel, dict):
        sel = [sel]
    elif not isinstance(sel, list):
        # A string here used to become [] and the request VANISHED with
        # ok=True, n_rejected=0. Silence is worse than rejection: the resident
        # cannot correct what it is never told about.
        if sel not in (None, ""):
            rejected.append({"work": sel,
                             "why": f"selected_work is {type(sel).__name__}, "
                                    "not a list of work objects"})
        sel = []
    if len(sel) > MAX_WORK_PER_TURN:
        # sel[:3] used to drop the rest with n_rejected unchanged. Same silence.
        for w in sel[MAX_WORK_PER_TURN:]:
            rejected.append({"work": w,
                             "why": f"more than {MAX_WORK_PER_TURN} work items "
                                    "in one turn; not run"})
        sel = sel[:MAX_WORK_PER_TURN]
    for w in sel:
        if not isinstance(w, dict):
            rejected.append({"work": w, "why": "not an object"})
            continue
        t = str(w.get("type") or "").upper()
        if t not in EXECUTABLE:
            rejected.append({"work": w, "why": f"{t!r} is not an executable work type"})
            continue
        # FOURTH SHAPE CRASH, found by the adversarial lane before the body
        # produced it: params as a list or a string is truthy, so `or {}` does
        # not fire and .get raises AttributeError. Coerce, never assume - the
        # same rule that fixed selected_work-as-a-dict one level up.
        p = w.get("params")
        if not isinstance(p, dict):
            if p not in (None, {}):
                rejected.append({"work": w, "why": f"params is {type(p).__name__}, not an object"})
                continue
            p = {}
        if t == "PERTURB":
            tensor = str(p.get("tensor") or "")
            side = str(p.get("side") or "rows")
            try:
                layer = int(p.get("layer"))
                frac = float(p.get("fraction"))
            except (TypeError, ValueError):
                rejected.append({"work": w, "why": "layer/fraction not numeric"})
                continue
            if tensor not in ("gate", "up", "down"):
                rejected.append({"work": w, "why": f"tensor {tensor!r} unknown"})
                continue
            if not (0 <= layer <= 63) or not (0.01 <= frac <= 0.95):
                rejected.append({"work": w, "why": "layer or fraction out of range"})
                continue
            if side not in ("rows", "cols"):
                rejected.append({"work": w, "why": f"side {side!r} unknown"})
                continue
        accepted.append({"type": t, "params": p, "why": w.get("why")})
    return {"ok": True, "accepted": accepted, "rejected": rejected,
            "n_accepted": len(accepted), "n_rejected": len(rejected)}


def execute(work: dict[str, Any]) -> dict[str, Any]:
    """Run one accepted work item. Only PERTURB actually touches the model."""
    t = work["type"]
    if t == "READ_RECEIPT":
        # Declared executable since the first version and never implemented, so
        # every READ_RECEIPT the resident chose was accepted and silently not
        # run. Reading a receipt is the cheapest thing in this repo.
        rel = str((work.get("params") or {}).get("receipt") or "")
        if not rel.startswith("receipts/") or ".." in rel:
            return {"type": t, "ran": False, "params": work.get("params"),
                    "why": f"receipt path {rel!r} is not under receipts/"}
        doc = _receipt(rel)
        return {"type": t, "ran": doc is not None, "params": work.get("params"),
                "result": {"keys": sorted(doc)[:40]} if doc else {},
                "why": None if doc else f"{rel} is not on disk or is not JSON"}
    if t == "COMPUTE":
        # S030 §22: models do not establish numeric truth. This runs the
        # deterministic calculator, it does not ask the body to do arithmetic.
        expr = str((work.get("params") or {}).get("expression") or "")
        return {"type": t, "ran": False, "params": work.get("params"),
                "why": "COMPUTE has no deterministic evaluator bound yet; the "
                       "request is RECORDED rather than silently accepted",
                "requested_expression": expr}
    if t != "PERTURB":
        return {"type": t, "ran": False, "params": work.get("params"),
                "why": f"{t} is declared executable but has no runner yet"}
    p = work["params"]
    cmd = [sys.executable, str(REPO / "tools/future/perturbation_workunit.py"),
           "--tensor", str(p["tensor"]), "--layer", str(int(p["layer"])),
           "--side", str(p.get("side", "rows")), "--fraction", str(float(p["fraction"]))]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=3600)
    out = {}
    if r.returncode == 0:
        try:
            out = json.loads(r.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            out = {"stdout_tail": r.stdout[-400:]}
        if not isinstance(out, dict):
            # A tool that prints a JSON LIST used to reach results_summary,
            # which does result.get('damage'). Same coerce-never-assume rule as
            # selected_work and params, now applied at the TOOL boundary too.
            out = {"non_object_result": out,
                   "why": f"runner printed {type(out).__name__}, not an object"}
    return {"type": t, "ran": r.returncode == 0, "params": p,
            "seconds": round(time.time() - t0, 1), "result": out,
            "stderr_tail": r.stderr[-300:] if r.returncode else None}


def _log(rec: dict[str, Any]) -> None:
    """Every entry carries an ABSOLUTE clock.

    G114 asks for CLAUDE_INTERVENTIONS per frontier move AND PER HOUR. The 83
    entries written before this line only had t_s - seconds since that run's
    start - so nothing in the log could be placed on a wall clock, and the
    per-hour half of the metric was unmeasurable from the resident's own stream.
    A log that cannot be joined to any other event stream is a log about itself.
    """
    rec.setdefault("unix", time.time())
    with (REPO / LOG_REL).open("a") as f:
        f.write(json.dumps(rec) + "\n")


def run(minutes: float) -> dict[str, Any]:
    sys.path.insert(0, str(REPO))
    from tools.future import resident_provider as rp
    from tools.future.resident_science_loop import parse
    from resident_output_contract import admit, SOVEREIGN_REPLY_SCHEMA

    k = load_kernel()
    prov, _h = rp.start(ready_timeout_s=900)
    t0 = time.time()
    deadline = t0 + minutes * 60
    n_iter = 0
    interventions = 0
    try:
        while time.time() < deadline:
            n_iter += 1
            pack = context_pack(k)
            ta = time.time()
            try:
                raw = prov.ask(f"sov_{n_iter}_{_digest(pack)}", pack,
                               MAX_NEW_TOKENS, timeout_s=900)
                reply = raw.get("text") or ""
            except Exception as exc:
                reply = f"<<ASK FAILED {type(exc).__name__}: {exc}>>"
            clean = rp.degenerate_prefix(reply)
            # G127: admit() is the single shape boundary. It never raises and
            # always returns the same key set, so the three crashes this loop
            # took on reply SHAPE cannot recur. parse() stays as a second
            # opinion on the salvaged prefix - admit is authority on shape,
            # not on whether a prefix was worth salvaging.
            adm = admit(reply, SOVEREIGN_REPLY_SCHEMA)
            # admit's schema is STRICTER than validate: it requires `why` and
            # `side`, which validate does not need to run the work. So a
            # recovered reply is used whenever it carries executable work, even
            # if the schema calls it incomplete - otherwise the narrow re-ask
            # would burn a turn asking for a field nothing consumes.
            has_work = bool(adm["value"].get("selected_work"))
            obj = (adm["value"] if (adm["ok"] or has_work)
                   else (parse(clean) or parse(reply)))
            retried = False
            reask_kind = None
            if obj is None and adm["reask"]["needed"] and adm["parse"]["recovered"]:
                # NARROW RE-ASK. A usable prefix that is missing fields does not
                # need the whole scientific prompt again - it needs the missing
                # fields. Shorter output is the point: this body degenerates
                # with length, so the shortest question that can close the gap
                # is also the one most likely to come back clean.
                retried = True
                reask_kind = "narrow"
                try:
                    raw2 = prov.ask(f"sov_{n_iter}_narrow",
                                    adm["reask"]["prompt_fragment"], 300,
                                    timeout_s=600)
                    reply2 = raw2.get("text") or ""
                except Exception:
                    reply2 = ""
                adm2 = admit(reply2, SOVEREIGN_REPLY_SCHEMA)
                merged = dict(adm["value"])
                for f, val in (adm2["value"] or {}).items():
                    if val not in (None, [], {}, ""):
                        merged[f] = val
                obj = merged
            elif obj is None:
                # The body echoes a long pack instead of answering it. Re-ask
                # with schema only rather than burning the turn.
                retried = True
                reask_kind = "terse"
                terse = context_pack(k, terse=True)
                try:
                    raw2 = prov.ask(f"sov_{n_iter}_terse", terse, 400, timeout_s=600)
                    reply2 = raw2.get("text") or ""
                except Exception:
                    reply2 = ""
                obj = parse(rp.degenerate_prefix(reply2)) or parse(reply2)
            v = validate(obj)
            results = []
            unlaunched = []
            # OVERLAP. execute() is a numpy CPU replay taking minutes, and the
            # provider serialises MODEL calls only - nothing stops the resident
            # reasoning while a perturbation runs. The loop used to sit idle
            # through every experiment it had just commissioned, so wall-clock
            # was ask + run + ask + run when it could be max(ask, run).
            pending = [w for w in v["accepted"]]
            with cf.ThreadPoolExecutor(max_workers=MAX_WORK_PER_TURN) as pool:
                futs = {}
                for i, w in enumerate(pending):
                    if time.time() >= deadline:
                        # The loop used to just break, and validation stored
                        # only the COUNTS. Work the resident chose and the
                        # harness accepted then vanished with no record that it
                        # existed.
                        unlaunched = pending[i:]
                        break
                    futs[pool.submit(execute, w)] = w
                launched_at = time.time()
                for f in cf.as_completed(futs):
                    try:
                        results.append(f.result())
                    except Exception as exc:
                        # A runner that raises is a RESULT about that runner,
                        # not a reason to lose the turn.
                        results.append({
                            "type": futs[f].get("type"), "ran": False,
                            "params": futs[f].get("params"),
                            "why": f"runner raised {type(exc).__name__}: {exc}",
                        })
            overlap_s = round(time.time() - launched_at, 1) if futs else 0.0
            it = {
                "n": n_iter,
                "t_s": round(time.time() - t0, 1),
                "ask_seconds": round(time.time() - ta, 1),
                "degenerated": rp.is_degenerate(reply),
                "salvaged_chars": len(clean),
                "reply_chars": len(reply),
                "parsed": obj is not None,
                "terse_retry_used": retried,
                "reask_kind": reask_kind,
                "n_launched_concurrently": len(futs),
                "concurrent_window_s": overlap_s,
                "admit": {kk: adm[kk] for kk in ("ok", "missing", "extra", "parse")},
                "belief_update": (obj or {}).get("belief_update"),
                "live_hypotheses": (obj or {}).get("live_hypotheses"),
                "validation": {kk: v[kk] for kk in ("n_accepted", "n_rejected", "rejected")},
                "accepted": v["accepted"],
                "unlaunched": unlaunched,
                "n_unlaunched": len(unlaunched),
                "results": results,
                "results_summary": [
                    f"{r['type']} {r.get('params',{})} -> "
                    f"{'damage ' + str((r.get('result') or {}).get('damage')) if r.get('ran') else 'DID NOT RUN'}"
                    for r in results
                ] + ([f"{len(unlaunched)} accepted item(s) NOT LAUNCHED: "
                      "the window closed first"] if unlaunched else [])
                or ["no work was accepted from that turn"],
            }
            # Persist the resident's OWN hypotheses onto the kernel so the next
            # pack can show them back. They were recorded on the iteration and
            # never read, so every turn started the investigation over.
            lh = (obj or {}).get("live_hypotheses")
            if isinstance(lh, list) and lh:
                k["live_hypotheses"] = [h for h in lh if isinstance(h, dict)][-4:]
            for r in results:
                pp = r.get("params") or {}
                if isinstance(pp, dict) and pp:
                    k.setdefault("tried_params", []).append(
                        f"{pp.get('tensor')}/L{pp.get('layer')}/"
                        f"{pp.get('side')}/{pp.get('fraction')}")
            k["iterations"].append(it)
            save_kernel(k)
            _log(it)
            print(json.dumps({kk: it[kk] for kk in
                              ("n", "t_s", "parsed", "degenerated",
                               "belief_update", "results_summary")}, indent=1))
            sys.stdout.flush()
    finally:
        try:
            prov.stop()
        except Exception:
            pass
    return {"iterations": n_iter, "minutes": minutes,
            "claude_interventions": interventions,
            "elapsed_s": round(time.time() - t0, 1)}


def build() -> dict[str, Any]:
    k = load_kernel()
    its = k.get("iterations") or []
    ran = [r for it in its for r in it.get("results", []) if r.get("ran")]
    return {
        "obligation": "G115",
        "authority": "S033",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "resident_mode": k.get("resident_mode"),
        "frontier": k.get("frontier"),
        "n_iterations": len(its),
        "n_experiments_run": len(ran),
        "n_parsed": sum(1 for it in its if it.get("parsed")),
        "n_degenerated": sum(1 for it in its if it.get("degenerated")),
        "hypotheses_in_kernel": [h["id"] for h in k.get("hypotheses", [])],
        "what_claude_did_not_do": (
            "choose the hypothesis, choose the experiment, or interpret the "
            "result. The kernel carries evidence with interpretation explicitly "
            "left NONE RECORDED, and the context pack offers a work vocabulary "
            "rather than a plan."
        ),
        "unsupported_requests_are_signal": (
            "work the resident asks for that the deterministic layer cannot run "
            "is recorded rather than discarded. It is the cheapest available "
            "measurement of which harness pieces are actually missing."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    if a.init:
        k = init_kernel()
        print(f"{kernel_path()}  frontier={k['frontier']} mode={k['resident_mode']}")
        return 0
    if a.pack:
        print(context_pack(load_kernel()))
        return 0
    if a.run:
        print(json.dumps(run(a.minutes), indent=1))
        return 0
    doc = build()
    if a.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
