#!/usr/bin/env python3
"""K1: is the sovereign loop actually autonomous, or does it only look it?

Attacks tools/future/hcli_sovereign.py and the mission kernel without modifying
either. The live module is imported; it is never edited. run() is never called
(that would start or wrestle the resident). execute() is never called with
PERTURB (that would launch a GPU workunit). save_kernel is only ever pointed
at an explicit temp path and refuses the live kernel.

    python3 tools/future/sovereign_attack_report.py --build
    python3 -m pytest tools/future/test_sovereign_attacks.py -q
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from tools.future._common import REPO, git, write_receipt, _assert_no_hardware_claims

RECORDED_BY = "tools/future/sovereign_attack_report.py"
RECEIPT_NAME = "SOVEREIGN_ATTACKS.json"
SCHEMA = "hawking.future.sovereign_attacks.v1"
VERSION = 1
PYTEST = "python3 -m pytest tools/future/test_sovereign_attacks.py"

_SOV = None


class SovereignAttackRefused(RuntimeError):
    """An input the attack needs is missing; it will not invent one."""


def live_sovereign_path() -> Path:
    """The live module is untracked on the canonical checkout.

    This worktree is a sparse checkout and does not materialize the untracked
    file. Refuse rather than copy it: the tests must import the running loop.
    """
    here = Path(__file__).resolve().parent / "hcli_sovereign.py"
    if here.is_file():
        return here
    common = git("rev-parse", "--git-common-dir")
    if not common:
        raise SovereignAttackRefused(
            "git rev-parse --git-common-dir returned empty; cannot locate "
            "tools/future/hcli_sovereign.py"
        )
    cand = Path(common).resolve().parent / "tools" / "future" / "hcli_sovereign.py"
    if not cand.is_file():
        raise SovereignAttackRefused(
            f"tools/future/hcli_sovereign.py is not on disk at {here} or {cand}; "
            "the live loop cannot be imported"
        )
    return cand


def load_sovereign() -> Any:
    """Import the live module by path. Restores sys.path after exec.

    The live file inserts its own directory onto sys.path so it can
    `from _common import REPO`. That REPO is the canonical checkout. Tests
    must not call save_kernel / _log / run without redirecting those paths.
    """
    global _SOV
    if _SOV is not None:
        return _SOV
    path = live_sovereign_path()
    saved = sys.path[:]
    try:
        spec = importlib.util.spec_from_file_location("hcli_sovereign_live", path)
        if spec is None or spec.loader is None:
            raise SovereignAttackRefused(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    _SOV = mod
    return _SOV


def _repro(node: str) -> str:
    return f"{PYTEST}::{node} -q"


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def synthetic_kernel() -> dict[str, Any]:
    """A kernel with the keys context_pack reads. Not written to disk."""
    return {
        "schema": "hawking.future.hcli_mission_kernel.v1",
        "measured_state": {
            "complete_bpw": None,
            "conventional_floor_bpw_if_every_untested_move_worked": None,
        },
        "hypotheses": [
            {"id": "H1.gate_up_mutual_information", "verdict": "REFUTED"},
            {"id": "H2.functional_role_gate_dominant", "verdict": "REFUTED"},
        ],
        "observations": [{"text": "zeroing 40% of a tensor's output rows moves cosine"}],
        "scars": ["scar-a", "scar-b", "scar-c"],
        "iterations": [],
        "tried_params": [],
    }


def _row(
    attack_id: str,
    verdict: str,
    *,
    node: str,
    detail: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if verdict not in ("HELD", "DEFECT", "UNTESTED"):
        raise SovereignAttackRefused(f"unknown verdict {verdict!r}")
    if not node:
        raise SovereignAttackRefused("reproduction node is required")
    rec = {
        "id": attack_id,
        "verdict": verdict,
        "reproduction": _repro(node),
        "detail": detail,
    }
    if extra:
        rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# Attacks. Each returns a receipt row. None of them call run() or PERTURB.
# ---------------------------------------------------------------------------


def attack_fake_sovereign() -> dict[str, Any]:
    """Does the loop continue without a caller, or block on one?"""
    sov = load_sovereign()
    src = inspect.getsource(sov.run)
    main_src = inspect.getsource(sov.main)
    blocks = ("input(" in src) or ("sys.stdin" in src)
    increments = ("interventions +=" in src) or (
        "interventions =" in src.replace("interventions = 0", "")
    )
    closed_while = "while time.time() < deadline" in src
    ask_inside = "prov.ask(" in src
    # --run is a finite batch; a supervisor must reinvoke. That is not a
    # mid-loop block on a caller.
    finite = "run(a.minutes)" in main_src or "run(minutes)" in src
    held = (not blocks) and closed_while and ask_inside and (not increments)
    return _row(
        "FAKE_SOVEREIGN",
        "HELD" if held else "DEFECT",
        node="test_attack_fake_sovereign_run_does_not_read_stdin",
        detail=(
            "run() is a closed while-time-deadline loop that calls prov.ask and "
            "execute with no input()/sys.stdin. claude_interventions is assigned "
            "0 and never incremented. Autonomy is time-bounded by --minutes; a "
            "caller must reinvoke the process, but nothing inside the loop "
            "blocks on one."
        ),
        extra={
            "reads_stdin": blocks,
            "closed_while_deadline": closed_while,
            "asks_resident_inside_loop": ask_inside,
            "claude_interventions_incremented": increments,
            "process_is_finite_minutes": finite,
        },
    )


def attack_malformed_missing_key() -> dict[str, Any]:
    sov = load_sovereign()
    v = sov.validate({})
    held = (
        isinstance(v, dict)
        and v.get("n_accepted") == 0
        and v.get("n_rejected") == 0
        and "accepted" in v
        and "rejected" in v
    )
    return _row(
        "MALFORMED_REPLY_MISSING_KEY",
        "HELD" if held else "DEFECT",
        node="test_attack_malformed_missing_selected_work",
        detail="missing selected_work returns empty accepted/rejected with counts; crash 1 is fixed",
        extra={"validation": {k: v[k] for k in ("ok", "n_accepted", "n_rejected")}},
    )


def attack_malformed_dict_as_list() -> dict[str, Any]:
    sov = load_sovereign()
    v = sov.validate({
        "selected_work": {
            "type": "PERTURB",
            "params": {"tensor": "gate", "layer": 0, "fraction": 0.5, "side": "rows"},
        }
    })
    held = v.get("n_accepted") == 1 and v.get("n_rejected") == 0
    return _row(
        "MALFORMED_REPLY_DICT_AS_LIST",
        "HELD" if held else "DEFECT",
        node="test_attack_malformed_selected_work_dict",
        detail="selected_work as a dict is coerced to a one-item list; crash 2 is fixed",
        extra={"n_accepted": v.get("n_accepted"), "n_rejected": v.get("n_rejected")},
    )


def attack_malformed_parse_none() -> dict[str, Any]:
    sov = load_sovereign()
    v = sov.validate(None)
    held = (
        v.get("ok") is False
        and v.get("n_accepted") == 0
        and v.get("n_rejected") == 0
        and "rejected" in v
    )
    return _row(
        "MALFORMED_REPLY_PARSE_FAILURE_COUNTS",
        "HELD" if held else "DEFECT",
        node="test_attack_malformed_parse_none_counts",
        detail="validate(None) carries n_accepted/n_rejected; crash 3 is fixed",
        extra={"validation": {k: v[k] for k in ("ok", "why", "n_accepted", "n_rejected")}},
    )


def attack_malformed_params_list() -> dict[str, Any]:
    """The fourth crash: params is a list, validate assumes a dict."""
    sov = load_sovereign()
    obj = {
        "belief_update": "x",
        "selected_work": [{
            "type": "PERTURB",
            "params": ["gate", 0, "rows", 0.5],
            "why": "schema-confused list of values",
        }],
    }
    try:
        v = sov.validate(obj)
    except Exception as exc:
        return _row(
            "MALFORMED_REPLY_PARAMS_LIST",
            "DEFECT",
            node="test_attack_malformed_params_list",
            detail=(
                "validate() does `p = w.get('params') or {}` then `p.get('tensor')`. "
                "A truthy non-dict (list, string, number) raises "
                f"{type(exc).__name__}: {exc}. This is the fourth harness crash: "
                "missing key, dict-where-list, parse-failure counts, then params shape."
            ),
            extra={
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "payload": obj["selected_work"][0]["params"],
            },
        )
    return _row(
        "MALFORMED_REPLY_PARAMS_LIST",
        "HELD",
        node="test_attack_malformed_params_list",
        detail="params as a list was rejected without raising",
        extra={"n_accepted": v.get("n_accepted"), "n_rejected": v.get("n_rejected")},
    )


def attack_malformed_params_string() -> dict[str, Any]:
    sov = load_sovereign()
    try:
        v = sov.validate({
            "selected_work": [{"type": "PERTURB", "params": "gate,0,rows,0.5"}]
        })
    except Exception as exc:
        return _row(
            "MALFORMED_REPLY_PARAMS_STRING",
            "DEFECT",
            node="test_attack_malformed_params_string",
            detail=(
                "params as a string is the same fourth-crash family: "
                f"{type(exc).__name__}: {exc}"
            ),
            extra={"exception_type": type(exc).__name__, "exception": str(exc)},
        )
    return _row(
        "MALFORMED_REPLY_PARAMS_STRING",
        "HELD",
        node="test_attack_malformed_params_string",
        detail="params as a string was rejected without raising",
        extra={"n_rejected": v.get("n_rejected")},
    )


def attack_malformed_tool_result_list() -> dict[str, Any]:
    """run()'s results_summary assumes result is a dict when ran is true."""
    src = inspect.getsource(load_sovereign().run)
    uses_result_get = "r['result'].get('damage')" in src.replace(" ", "")
    r = {"type": "PERTURB", "ran": True, "params": {}, "result": [0.1, 0.2]}
    try:
        _ = (
            f"{r['type']} {r.get('params', {})} -> "
            f"{'damage ' + str(r['result'].get('damage')) if r.get('ran') else 'DID NOT RUN'}"
        )
        crashed = False
        exc_s = None
    except Exception as exc:
        crashed = True
        exc_s = f"{type(exc).__name__}: {exc}"
    return _row(
        "MALFORMED_REPLY_TOOL_RESULT_LIST",
        # THE SOURCE IS AUTHORITY, NOT THE COPY. This probe evaluated a
        # transcription of run()'s expression, so fixing the live code could
        # never change its verdict - it was grading its own copy. The copy is
        # kept below as corroboration of what the OLD expression did.
        "DEFECT" if uses_result_get else "HELD",
        node="test_attack_malformed_tool_result_list",
        detail=(
            "execute() json.loads the last stdout line; if that JSON is a list, "
            "run()'s results_summary does result.get('damage') and crashes. "
            f"{exc_s}"
        ),
        extra={
            "run_source_uses_result_get": uses_result_get,
            "crashed": crashed,
            "exception": exc_s,
        },
    )


def attack_silent_drop_unsupported() -> dict[str, Any]:
    sov = load_sovereign()
    work = {"type": "LAUNCH_WORKUNIT", "params": {"id": "WU.1"}, "why": "launch"}
    v = sov.validate({"selected_work": [work]})
    recorded = v.get("n_rejected") == 1 and v.get("rejected")
    return _row(
        "SILENT_DROP_UNSUPPORTED",
        "HELD" if recorded else "DEFECT",
        node="test_attack_silent_drop_unsupported_is_recorded",
        detail=(
            "an unsupported type is appended to rejected with why "
            "'... is not an executable work type'. It is recorded, not lost. "
            "The live comment names this UNSUPPORTED_REQUEST; the stored why "
            "string is the executable-type message, not that token."
        ),
        extra={
            "n_rejected": v.get("n_rejected"),
            "rejected_why": (v.get("rejected") or [{}])[0].get("why"),
        },
    )


def attack_silent_drop_truncation() -> dict[str, Any]:
    sov = load_sovereign()
    items = [
        {"type": "PERTURB", "params": {"tensor": t, "layer": 0, "fraction": 0.5, "side": "rows"}}
        for t in ("gate", "up", "down")
    ]
    fourth = {"type": "COMPUTE", "params": {"op": "sum"}, "why": "the fourth request"}
    v = sov.validate({"selected_work": items + [fourth]})
    lost = v["n_accepted"] == 3 and v["n_rejected"] == 0
    return _row(
        "SILENT_DROP_TRUNCATION",
        "DEFECT" if lost else "HELD",
        node="test_attack_silent_drop_truncation",
        detail=(
            "validate() iterates sel[:3]. A fourth accepted-or-not item is "
            "dropped with n_rejected=0, so the request is not in rejected either."
        ),
        extra={
            "n_submitted": 4,
            "n_accepted": v["n_accepted"],
            "n_rejected": v["n_rejected"],
            "accepted_types": [w["type"] for w in v["accepted"]],
        },
    )


def attack_silent_drop_string_selected_work() -> dict[str, Any]:
    sov = load_sovereign()
    raw = "PERTURB gate layer 0 rows 0.5"
    v = sov.validate({"selected_work": raw})
    lost = v["n_accepted"] == 0 and v["n_rejected"] == 0 and v.get("ok") is True
    return _row(
        "SILENT_DROP_STRING_SELECTED_WORK",
        "DEFECT" if lost else "HELD",
        node="test_attack_silent_drop_string_selected_work",
        detail=(
            "selected_work as a string is neither list nor dict, so sel becomes "
            "[]. ok=True, n_rejected=0: the request vanishes."
        ),
        extra={"ok": v.get("ok"), "n_accepted": v["n_accepted"], "n_rejected": v["n_rejected"]},
    )


def attack_silent_drop_hypotheses() -> dict[str, Any]:
    sov = load_sovereign()
    k = synthetic_kernel()
    token = "UNIQUE_HYP_XYZ_DO_NOT_ECHO"
    k["iterations"] = [{
        "n": 1,
        "parsed": True,
        "live_hypotheses": [{"id": "H9.unique", "claim": token, "cheapest_falsifier": "measure"}],
        "results_summary": ["no work was accepted from that turn"],
    }]
    pack = sov.context_pack(k)
    fed = token in pack
    src = inspect.getsource(sov.context_pack)
    src_run = inspect.getsource(sov.run)
    return _row(
        "SILENT_DROP_HYPOTHESES",
        "HELD" if fed else "DEFECT",
        node="test_attack_silent_drop_hypotheses_not_fed_back",
        detail=(
            "live_hypotheses are stored on the iteration record but context_pack "
            "does not include them, and run() never appends them to k['hypotheses']. "
            "The next ask cannot see the resident's own last hypotheses."
        ),
        extra={
            "unique_claim_in_next_pack": fed,
            "context_pack_mentions_live_hypotheses_key": "live_hypotheses" in src,
            "run_merges_into_kernel_hypotheses": "k[\"hypotheses\"]" in src_run,
        },
    )


def attack_generated_compute_not_run() -> dict[str, Any]:
    sov = load_sovereign()
    v = sov.validate({"selected_work": [{"type": "COMPUTE", "params": {"expr": "1+1"}}]})
    if v["n_accepted"] != 1:
        raise SovereignAttackRefused(f"COMPUTE was not accepted: {v}")
    result = sov.execute(v["accepted"][0])
    launched = bool(result.get("ran"))
    return _row(
        "GENERATED_BUT_NEVER_LAUNCHED_COMPUTE",
        "DEFECT" if not launched else "HELD",
        node="test_attack_generated_compute_not_run",
        detail=(
            "COMPUTE is in EXECUTABLE and validate() accepts it. execute() "
            "returns ran=False, why='COMPUTE is declared executable but has no "
            "runner yet'. Accepted work is not run. Same for READ_RECEIPT."
        ),
        extra={
            "n_accepted": v["n_accepted"],
            "execute_ran": result.get("ran"),
            "execute_why": result.get("why"),
            "in_executable": "COMPUTE" in sov.EXECUTABLE,
        },
    )


def attack_generated_deadline_drops_work() -> dict[str, Any]:
    sov = load_sovereign()
    v = sov.validate({
        "selected_work": [{
            "type": "PERTURB",
            "params": {"tensor": "gate", "layer": 1, "fraction": 0.2, "side": "rows"},
        }]
    })
    if v["n_accepted"] != 1:
        raise SovereignAttackRefused(f"PERTURB was not accepted: {v}")
    deadline = time.time() - 1.0
    results: list[Any] = []
    for w in v["accepted"]:
        if time.time() >= deadline:
            break
        raise SovereignAttackRefused("deadline was in the past; execute must not run")
    src = inspect.getsource(sov.run)
    # The iteration stores n_accepted/n_rejected/rejected, not the accepted list.
    validation_line = [
        ln for ln in src.splitlines()
        if "n_accepted" in ln and "n_rejected" in ln
    ]
    src_run_all = inspect.getsource(sov.run)
    stores_accepted_list = (
        any("accepted" in ln.replace("n_accepted", "") for ln in validation_line)
        # The list belongs on the ITERATION record, not inside the validation
        # sub-dict. Grepping one line could only ever find one of the two
        # reasonable places to put it.
        or '"accepted": v["accepted"]' in src_run_all
        or '"unlaunched"' in src_run_all
    )
    dropped = (v["n_accepted"] > 0) and (results == []) and (not stores_accepted_list)
    return _row(
        "GENERATED_BUT_NEVER_LAUNCHED_DEADLINE",
        "DEFECT" if dropped else "HELD",
        node="test_attack_generated_deadline_drops_work",
        detail=(
            "run() breaks the accepted-work loop when the deadline hits and "
            "records validation as n_accepted/n_rejected/rejected only. The "
            "accepted list itself is not stored, so unlaunched work vanishes."
        ),
        extra={
            "n_accepted": v["n_accepted"],
            "results_after_past_deadline": results,
            "validation_stores_accepted_list": stores_accepted_list,
            "validation_lines": validation_line,
        },
    )


def attack_context_accumulation() -> dict[str, Any]:
    sov = load_sovereign()
    k = synthetic_kernel()
    lengths: list[int] = []
    for i in range(20):
        k["iterations"].append({"results_summary": [f"{i:04d}|" + ("Z" * 4000)]})
        k["tried_params"].append(f"up/L{i}/rows/0.5")
        lengths.append(len(sov.context_pack(k)))
    # First iteration introduces LAST TURN (~4k). After tried_params[-6:] fills,
    # further history must not accumulate.
    steady = lengths[6:]
    spread = max(steady) - min(steady)
    grew_with_history = (lengths[-1] - lengths[6]) > 1000
    return _row(
        "CONTEXT_ACCUMULATION",
        "DEFECT" if grew_with_history else "HELD",
        node="test_attack_context_pack_does_not_accumulate",
        detail=(
            "context_pack uses only the last turn's results_summary[:2], "
            "tried_params[-6:], a scar COUNT, and refuted hypothesis ids. "
            "20 iterations of 4000-char summaries do not grow the pack by the "
            "history size. The fourteen echo failures were identical-pack, "
            "not accumulation."
        ),
        extra={
            "n_iterations": 20,
            "summary_chars_each": 4005,
            "pack_len_first": lengths[0],
            "pack_len_after_6": lengths[6],
            "pack_len_last": lengths[-1],
            "steady_spread_chars": spread,
            "grew_with_history": grew_with_history,
        },
    )


def attack_identical_reply_loop() -> dict[str, Any]:
    sov = load_sovereign()
    k = synthetic_kernel()
    packs = []
    terse = []
    for _ in range(5):
        packs.append(sov.context_pack(k))
        terse.append(sov.context_pack(k, terse=True))
        k["iterations"].append({
            "results_summary": ["no work was accepted from that turn"],
            "parsed": False,
        })
    frozen = packs[1:]
    frozen_terse = terse[1:]
    identical = all(p == frozen[0] for p in frozen)
    identical_terse = all(p == frozen_terse[0] for p in frozen_terse)
    src = inspect.getsource(sov.context_pack)
    has_entropy = ("n_iter" in src) or ("nonce" in src) or ("iteration" in src.lower() and "n =" in src)
    return _row(
        "IDENTICAL_REPLY_LOOP",
        "DEFECT" if identical else "HELD",
        node="test_attack_identical_reply_loop_can_escape",
        detail=(
            "after one failed-parse turn, LAST TURN is the constant "
            "'no work was accepted from that turn' and tried_params is unchanged. "
            "context_pack (and the terse retry pack) is then byte-identical. "
            "There is no iteration index, nonce, or prior-reply hash. Under "
            "greedy decoding the body returns the same bytes and the loop "
            "cannot escape. The fourteen consecutive echoes already happened."
        ),
        extra={
            "full_pack_identical_after_first_failed_turn": identical,
            "terse_pack_identical_after_first_failed_turn": identical_terse,
            "full_pack_digests": [_digest(p) for p in packs],
            "context_pack_has_iteration_entropy": has_entropy,
        },
    )


def attack_kernel_write_safety(*, kernel_file: Path | None = None) -> dict[str, Any]:
    if kernel_file is None:
        raise SovereignAttackRefused(
            "kernel_file is required; refusing to default to the live mission kernel"
        )
    sov = load_sovereign()
    dest = Path(kernel_file)
    if dest.resolve() == sov.kernel_path().resolve():
        raise SovereignAttackRefused(
            f"refusing to write the live mission kernel at {dest}"
        )
    src = inspect.getsource(sov.save_kernel)
    uses_write_text = "write_text" in src
    uses_replace = ("os.replace" in src) or ("os.rename" in src)
    uses_temp = (".tmp" in src) or ("NamedTemporaryFile" in src) or ("mkstemp" in src)

    orig_kp = sov.kernel_path
    orig_write = Path.write_text
    previous_intact = None
    load_exc = None
    try:
        sov.kernel_path = lambda: dest  # type: ignore[method-assign]
        good = {"schema": "v1", "iterations": [], "sentinel": "GOOD"}
        sov.save_kernel(good)
        before = dest.read_text()

        def boom(self, data, *a, **kw):
            if Path(self).resolve() == dest.resolve():
                raw = data.encode() if isinstance(data, str) else data
                fd = os.open(self, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
                try:
                    os.write(fd, raw[:24])
                finally:
                    os.close(fd)
                raise RuntimeError("simulated crash mid-write")
            return orig_write(self, data, *a, **kw)

        Path.write_text = boom  # type: ignore[method-assign]
        try:
            sov.save_kernel({"schema": "v1", "iterations": [1, 2, 3], "sentinel": "NEW"})
            crashed = False
        except RuntimeError:
            crashed = True
        Path.write_text = orig_write  # type: ignore[method-assign]

        after = dest.read_bytes()
        try:
            previous_intact = json.loads(after.decode())["sentinel"] == "GOOD"
        except Exception:
            previous_intact = False
        try:
            sov.load_kernel()
            load_exc = None
        except Exception as exc:
            load_exc = f"{type(exc).__name__}: {exc}"
    finally:
        Path.write_text = orig_write  # type: ignore[method-assign]
        sov.kernel_path = orig_kp  # type: ignore[method-assign]

    defect = uses_write_text and (not uses_replace) and previous_intact is False
    persist = REPO / "hcli" / "persist.py"
    if not persist.is_file():
        # Sparse checkout: the file is in git even when not on disk.
        persist_blob = git("show", "HEAD:hcli/persist.py")
        persist_has_atomic = "def atomic_write_json" in persist_blob
    else:
        persist_has_atomic = "def atomic_write_json" in persist.read_text()
    return _row(
        "KERNEL_WRITE_SAFETY",
        "DEFECT" if defect else "HELD",
        node="test_attack_kernel_write_survives_crash_mid_write",
        detail=(
            "save_kernel uses Path.write_text, which truncates then writes. A "
            "crash mid-write leaves a truncated file; the previous kernel is "
            "gone. hcli/persist.py::atomic_write_json already does temp + fsync "
            "+ os.replace and is unused here. The mission kernel is the "
            "resident's only memory."
        ),
        extra={
            "save_kernel_uses_write_text": uses_write_text,
            "save_kernel_uses_os_replace": uses_replace,
            "save_kernel_uses_tempfile": uses_temp,
            "crash_simulated": crashed,
            "previous_kernel_intact": previous_intact,
            "load_after_crash": load_exc,
            "atomic_writer_already_exists": persist_has_atomic,
        },
    )


def attack_corrupt_kernel_load(*, kernel_file: Path | None = None) -> dict[str, Any]:
    if kernel_file is None:
        raise SovereignAttackRefused(
            "kernel_file is required; refusing to default to the live mission kernel"
        )
    sov = load_sovereign()
    dest = Path(kernel_file)
    if dest.resolve() == sov.kernel_path().resolve():
        raise SovereignAttackRefused(
            f"refusing to write the live mission kernel at {dest}"
        )
    dest.write_text("{")
    orig_kp = sov.kernel_path
    kind = None
    try:
        sov.kernel_path = lambda: dest  # type: ignore[method-assign]
        try:
            sov.load_kernel()
            kind = "none"
        except sov.SovereignRefused:
            kind = "SovereignRefused"
        except json.JSONDecodeError:
            kind = "JSONDecodeError"
        except Exception as exc:
            kind = type(exc).__name__
    finally:
        sov.kernel_path = orig_kp  # type: ignore[method-assign]
    defect = kind != "SovereignRefused"
    return _row(
        "KERNEL_CORRUPT_LOAD",
        "DEFECT" if defect else "HELD",
        node="test_attack_corrupt_kernel_raises_sovereign_refused",
        detail=(
            "load_kernel json.loads with no handler. A corrupt kernel raises "
            f"{kind}, not SovereignRefused, so the next --run crash-loops "
            "instead of refusing cleanly."
        ),
        extra={"raised": kind},
    )


ATTACKS = (
    attack_fake_sovereign,
    attack_malformed_missing_key,
    attack_malformed_dict_as_list,
    attack_malformed_parse_none,
    attack_malformed_params_list,
    attack_malformed_params_string,
    attack_malformed_tool_result_list,
    attack_silent_drop_unsupported,
    attack_silent_drop_truncation,
    attack_silent_drop_string_selected_work,
    attack_silent_drop_hypotheses,
    attack_generated_compute_not_run,
    attack_generated_deadline_drops_work,
    attack_context_accumulation,
    attack_identical_reply_loop,
    attack_kernel_write_safety,
    attack_corrupt_kernel_load,
)


def run_attacks(*, kernel_file: Path | None = None) -> dict[str, Any]:
    if kernel_file is None:
        raise SovereignAttackRefused(
            "kernel_file is required; the write-safety attacks will not "
            "default to the live mission kernel"
        )
    rows = []
    for fn in ATTACKS:
        params = inspect.signature(fn).parameters
        if "kernel_file" in params:
            rows.append(fn(kernel_file=kernel_file))
        else:
            rows.append(fn())
    defects = [r for r in rows if r["verdict"] == "DEFECT"]
    held = [r for r in rows if r["verdict"] == "HELD"]
    untested = [r for r in rows if r["verdict"] == "UNTESTED"]
    path = live_sovereign_path()
    try:
        rel = str(path.relative_to(REPO))
    except ValueError:
        rel = "tools/future/hcli_sovereign.py"
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "obligation": "K1",
        "authority": "S033",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "question": (
            "Is the sovereign loop actually autonomous, or does it only look autonomous?"
        ),
        "target_module_rel": rel,
        "target_module_imported": str(path),
        "attacks": rows,
        "n_attacks": len(rows),
        "n_defect": len(defects),
        "n_held": len(held),
        "n_untested": len(untested),
        "at_least_one_real_defect": bool(defects),
        "defect_ids": [r["id"] for r in defects],
        "held_ids": [r["id"] for r in held],
        "did_not_call_run": True,
        "did_not_execute_perturb": True,
        "did_not_signal_processes": True,
        "did_not_write_live_kernel": True,
        "what_this_lane_must_not_do": (
            "choose the next experiment or representation. The attacks are "
            "harness-only; the resident owns SUB2_EBPW."
        ),
        "fourth_crash": (
            "selected_work[].params as a list/string: validate() calls .get on it. "
            "Crashes 1-3 (missing key, dict-as-list, parse-failure counts) are held."
        ),
    }
    _assert_no_hardware_claims(doc)
    return doc


def build() -> Path:
    with tempfile.TemporaryDirectory(prefix="sovereign_attack_") as td:
        kernel_file = Path(td) / "HCLI_MISSION_KERNEL.json"
        doc = run_attacks(kernel_file=kernel_file)
    return write_receipt(RECEIPT_NAME, doc, RECORDED_BY)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    if not a.build:
        raise SovereignAttackRefused(
            "pass --build; refusing to default a receipt write or to run the live loop"
        )
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
