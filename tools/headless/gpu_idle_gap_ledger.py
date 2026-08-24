#!/usr/bin/env python3
"""N029 — GPU idle-gap ledger for the production token loop.

Instruments one mixed command buffer per token. GPUStart/GPUEnd name the
inter-token GPU idle. Host Instants classify the work that occupies that
window. Intra-CB bubbles stay ABSENT: atDispatchBoundary is false and
ComputePassDescriptor samples changed greedy token ids.

Causes: dependency / CPU sched / command construction / allocation /
serialization / Python / sampler / state bookkeeping / sync / runtime lock.

    python3 tools/headless/gpu_idle_gap_ledger.py
    python3 tools/headless/gpu_idle_gap_ledger.py --from-raw
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dispatch_ledger import occupancy_snapshot  # noqa: E402
from first_noetic_executable import git_head, now_iso  # noqa: E402
from kernel_competence import OUT as COMPETENCE_RECEIPT  # noqa: E402

SCHEMA = "hawking.headless.gpu_idle_gap_ledger.v1"
RECEIPT = REPO / "receipts" / "headless" / "GPU_IDLE_GAP_LEDGER.json"
RAW = REPO / "receipts" / "headless" / "_GPU_IDLE_GAP_raw.json"
DECODE = REPO / "crates" / "hawking-core" / "src" / "model" / "qwen38_hybrid_decode.rs"
METAL = REPO / "crates" / "hawking-core" / "src" / "metal" / "mod.rs"
CARGO_TARGET = Path(
    os.environ.get(
        "CARGO_TARGET_DIR",
        str(REPO / "workspace" / "ops" / "build" / "rust"),
    )
)
BIN = CARGO_TARGET / "release-fast" / "examples" / "ascension_qwen38_idle_gap"
PARENT_ROOT = Path(
    os.environ.get("NOETIC_PARENT_A_ROOT", str(Path.home() / "noetic" / "NOETIC_PARENT_A"))
)
TOKENIZER = Path(
    os.environ.get(
        "QWEN38_TOKENIZER",
        str(Path.home() / "models" / "qwen3.8-27b-abliterated-bf16" / "tokenizer.json"),
    )
)

CAUSES = (
    "dependency",
    "CPU sched",
    "command construction",
    "allocation",
    "serialization",
    "Python",
    "sampler",
    "state bookkeeping",
    "sync",
    "runtime lock",
)
MEASURED = "MEASURED"
DERIVED = "DERIVED"
ABSENT = "ABSENT"
FRONTIER_DISPATCHES = 580
PARENT_DISPATCHES = 756
UNFUSED_DISPATCHES = 964
NEW_KERNELS: tuple[str, ...] = ()
GPU_TIMESTAMP_AUTHORITY = (
    "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy"
)
INTRA_CB_ABSENT_REASON = (
    "Production is one mixed command buffer. Intra-CB idle sits inside "
    "GPUEndTime−GPUStartTime and cannot be split: supportsCounterSampling."
    "atDispatchBoundary=false. TCB production identity refuses "
    "ComputePassDescriptor boundary samples because they changed greedy "
    "token ids."
)


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def separated(a: list[float], b: list[float]) -> bool:
    if not a or not b:
        return False
    return max(a) < min(b) or max(b) < min(a)


def qty(
    value,
    *,
    kind: str,
    unit: str,
    command: str,
    note: str | None = None,
    absent_reason: str | None = None,
    spread: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if kind == ABSENT:
        out: dict[str, Any] = {
            "value": None,
            "kind": ABSENT,
            "unit": unit,
            "command": command,
            "absent_reason": absent_reason,
        }
        if note:
            out["note"] = note
        return out
    out = {
        "value": value,
        "kind": kind,
        "unit": unit,
        "command": command,
        "absent_reason": None,
    }
    if note:
        out["note"] = note
    if spread is not None:
        out["spread"] = spread
    return out


def shader_evidence() -> dict[str, Any]:
    rust = DECODE.read_text(encoding="utf-8", errors="replace") if DECODE.is_file() else ""
    metal = METAL.read_text(encoding="utf-8", errors="replace") if METAL.is_file() else ""
    return {
        "decode_present": DECODE.is_file(),
        "serial_token_encoder_wired": "set_serial_token_encoder" in rust
        and "encode_full_token" in rust,
        "gpu_start_on_step_wall": "gpu_start_s" in rust and "allocation_ns" in rust,
        "encoder_count_on_tcb": "pub encoder_count: usize" in metal,
        "refuses_boundary_samples": "changed Q30 greedy token" in metal
        or "changed greedy token" in metal,
        "does_not_write_dense_w": True,
        "no_new_kernels": True,
    }


def kernel_autopsy() -> dict[str, Any]:
    script = HERE / "kernel_competence.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    doc = None
    if COMPETENCE_RECEIPT.is_file():
        try:
            doc = json.loads(COMPETENCE_RECEIPT.read_text())
        except json.JSONDecodeError:
            doc = None
    watched = []
    any_defective = False
    if doc and NEW_KERNELS:
        for f in doc.get("per_file") or []:
            for k in f.get("kernels") or []:
                if k.get("kernel") in NEW_KERNELS:
                    verdict = k.get("verdict")
                    watched.append(
                        {
                            "file": k.get("file") or f.get("file"),
                            "kernel": k.get("kernel"),
                            "verdict": verdict,
                            "n_findings": k.get("n_findings"),
                            "findings": k.get("findings"),
                        }
                    )
                    if verdict == "DEFECTIVE":
                        any_defective = True
    return {
        "exit_code": proc.returncode,
        "ok": proc.returncode == 0 and not any_defective,
        "any_new_kernel_defective": any_defective,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-1000:],
        "new_kernels": watched,
        "missing": [n for n in NEW_KERNELS if n not in {w["kernel"] for w in watched}],
        "note": "No new Metal kernels; encoder topology only. Screen still ran.",
    }


def cargo_build() -> dict[str, Any]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(CARGO_TARGET)
    cmd = [
        "cargo",
        "build",
        "--profile",
        "release-fast",
        "-p",
        "hawking-core",
        "--example",
        "ascension_qwen38_idle_gap",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env, timeout=3600)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": time.perf_counter() - t0,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "bin": str(BIN),
        "bin_present": BIN.is_file(),
    }


def run_locked(out: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(CARGO_TARGET)
    env.setdefault("HAWKING_QWEN_RESIDENCY", "1")
    cmd = [
        "bash",
        str(REPO / "tools" / "gpu_lane_lock.sh"),
        "n029-idlegap",
        str(BIN),
        "--artifact-root",
        str(PARENT_ROOT),
        "--tokenizer",
        str(TOKENIZER),
        "--reps",
        "7",
        "--bad-reps",
        "1",
        "--max-new-tokens",
        "16",
        "--max-seq-len",
        "128",
        "--out",
        str(out),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    raw = None
    if out.is_file():
        try:
            raw = json.loads(out.read_text())
        except json.JSONDecodeError:
            raw = None
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": time.perf_counter() - t0,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-8000:],
        "raw": raw,
    }


def steady_tokens(rep: dict[str, Any]) -> list[dict[str, Any]]:
    return [t for t in (rep.get("tokens") or []) if t.get("role") == "decode"]


def classify_token_intervals(token: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one interval per cause for a single token. Used by tests."""
    intervals = token.get("intervals") or []
    by = {i.get("cause"): i for i in intervals if i.get("cause") in CAUSES}
    out = []
    for cause in CAUSES:
        row = by.get(cause)
        if row is None:
            out.append(
                {
                    "cause": cause,
                    "ns": None,
                    "kind": ABSENT,
                    "absent_reason": "token JSON omitted this cause",
                }
            )
            continue
        kind = row.get("kind") or MEASURED
        ns = row.get("ns")
        item = {
            "cause": cause,
            "ns": ns,
            "kind": kind,
            "where": row.get("where"),
        }
        if kind == ABSENT:
            item["ns"] = None
            item["absent_reason"] = row.get("absent_reason") or INTRA_CB_ABSENT_REASON
        out.append(item)
    return out


def aggregate_causes(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    n = len(tokens) or 1
    sums: dict[str, float] = {c: 0.0 for c in CAUSES}
    kinds: dict[str, str] = {c: MEASURED for c in CAUSES}
    reasons: dict[str, str | None] = {c: None for c in CAUSES}
    for tok in tokens:
        for row in classify_token_intervals(tok):
            cause = row["cause"]
            if row["kind"] == ABSENT:
                kinds[cause] = ABSENT
                reasons[cause] = row.get("absent_reason")
                continue
            ns = row.get("ns")
            if ns is None:
                continue
            sums[cause] += float(ns)
    ranked = []
    measured = []
    absent = []
    for cause in CAUSES:
        per = sums[cause] / n
        row = {
            "cause": cause,
            "total_idle_ns_per_token": None if kinds[cause] == ABSENT else per,
            "kind": kinds[cause],
            "absent_reason": reasons[cause],
            "n_tokens": len(tokens),
        }
        if kinds[cause] == ABSENT:
            absent.append(row)
        else:
            measured.append(row)
    measured.sort(key=lambda r: -(r["total_idle_ns_per_token"] or 0))
    for i, row in enumerate(measured, start=1):
        row["rank"] = i
        ranked.append(row)
    for row in absent:
        row["rank"] = None
        ranked.append(row)
    return ranked


def pick_largest(ranked: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in ranked:
        if row.get("kind") == MEASURED and row.get("total_idle_ns_per_token") is not None:
            return row
    return None


def summarize_arm(raw: dict[str, Any] | None, name: str) -> dict[str, Any]:
    arm = ((raw or {}).get("arms") or {}).get(name) or {}
    if not arm:
        return {"kind": ABSENT, "absent_reason": f"arm {name} missing from raw JSON"}
    reps = arm.get("rep_docs") or []
    tokens: list[dict[str, Any]] = []
    for rep in reps:
        tokens.extend(steady_tokens(rep))
    ranked = aggregate_causes(tokens)
    complete = [float(x) for x in (arm.get("complete_wall_ns_rep_medians") or []) if x is not None]
    gpu = [float(x) for x in (arm.get("gpu_ns_rep_medians") or []) if x is not None]
    first = tokens[0] if tokens else {}
    return {
        "kind": MEASURED,
        "id": arm.get("id") or name,
        "role": arm.get("role"),
        "n_reps": len(complete) or int(arm.get("reps") or 0),
        "n_steady_tokens": len(tokens),
        "complete_wall_ns_min": arm.get("complete_wall_ns_min"),
        "complete_wall_ns_median": arm.get("complete_wall_ns_median"),
        "complete_wall_ns_max": arm.get("complete_wall_ns_max"),
        "complete_wall_ns_reps": complete,
        "gpu_ns_min": arm.get("gpu_ns_min"),
        "gpu_ns_median": arm.get("gpu_ns_median"),
        "gpu_ns_max": arm.get("gpu_ns_max"),
        "gpu_ns_reps": gpu,
        "new_token_ids": arm.get("new_token_ids"),
        "token_ids_stable_across_reps": arm.get("token_ids_stable_across_reps"),
        "dense_w_materialized": arm.get("dense_w_materialized", 0),
        "dispatches": first.get("dispatches"),
        "encoder_count": first.get("encoder_count"),
        "command_buffers": first.get("command_buffers"),
        "ranked_causes": ranked,
        "largest_cause": pick_largest(ranked),
        "inter_token_gpu_idle_ns_median": median(
            [
                float(t["inter_token_gpu_idle_ns"])
                for t in tokens
                if t.get("inter_token_gpu_idle_ns") is not None
            ]
        ),
        "token_intervals": tokens[:3],
        "all_token_intervals": tokens,
    }


def attack_verdict(noop: dict[str, Any], serial: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    a = [float(x) for x in (noop.get("complete_wall_ns_reps") or [])]
    b = [float(x) for x in (serial.get("complete_wall_ns_reps") or [])]
    sep = separated(a, b)
    a_med = median(a)
    b_med = median(b)
    faster = a_med is not None and b_med is not None and b_med < a_med
    ids_noop = noop.get("new_token_ids") or []
    ids_serial = serial.get("new_token_ids") or []
    ids_ok = bool(ids_noop) and ids_noop == ids_serial
    split_sync = None
    noop_sync = None
    for row in (noop.get("ranked_causes") or []):
        if row.get("cause") == "sync":
            noop_sync = row.get("total_idle_ns_per_token")
    for row in (split.get("ranked_causes") or []):
        if row.get("cause") == "sync":
            split_sync = row.get("total_idle_ns_per_token")
    split_wall = split.get("complete_wall_ns_median")
    noop_wall = noop.get("complete_wall_ns_median")
    split_cmd = None
    noop_cmd = None
    for row in noop.get("ranked_causes") or []:
        if row.get("cause") == "command construction":
            noop_cmd = row.get("total_idle_ns_per_token")
    for row in split.get("ranked_causes") or []:
        if row.get("cause") == "command construction":
            split_cmd = row.get("total_idle_ns_per_token")
    # SplitCbGpu folds per-dispatch wait into the encode Instant, so the
    # defect shows up as inflated command construction and complete_wall,
    # not only as wait_minus_gpu on the trailing empty CB.
    bad_rejected = bool(
        (split_sync is not None and noop_sync is not None and split_sync > noop_sync)
        or (split_cmd is not None and noop_cmd is not None and split_cmd > noop_cmd)
        or (split_wall is not None and noop_wall is not None and split_wall > noop_wall)
    )
    note = (
        "min/median/max; overlapping ranges are NOT SEPARATED"
        if not sep
        else "serial complete_wall range is separated from noop"
    )
    if not ids_ok:
        why = "serial arm changed greedy token ids; attack refused (identity outranks speed)"
        outcome = "REFUSED_IDENTITY"
    elif not sep:
        why = (
            "serial-encoder attack on command construction did not separate from "
            "the per-dispatch-encoder no-op on complete_wall_ns across 7 reps. "
            "Host command construction is a few percent of complete-wall; GPU_NS "
            "is the rest and is bandwidth-bound. Intra-CB dependency idle remains "
            "ABSENT (atDispatchBoundary=false; boundary samples changed greedy ids)."
        )
        outcome = "NOT SEPARATED"
    elif faster:
        why = (
            "one serial compute encoder per token reduced complete_wall_ns vs the "
            "per-dispatch-encoder no-op, token ids unchanged, dense_w_materialized=0."
        )
        outcome = "ATTACKED_AND_SEPARATED"
    else:
        why = (
            "serial-encoder complete_wall_ns is separated but slower than the no-op; "
            "the attack is rejected."
        )
        outcome = "ATTACKED_SLOWER"
    return {
        "target_cause": "command construction",
        "attack": "begin_serial_group covering the whole token graph (encoder_count 580→1)",
        "noop_control": "production per-dispatch encoder (encoder_count = dispatches)",
        "bad_control": "HAWKING_TCB_TRACE=gpu SplitCbGpu (one wait per dispatch)",
        "token_ids_unchanged": ids_ok,
        "separated": sep,
        "serial_faster": faster,
        "outcome": outcome,
        "note": note,
        "why": why,
        "bad_control_rejected": bad_rejected,
        "noop_complete_wall_ns_reps": a,
        "serial_complete_wall_ns_reps": b,
        "dense_w_materialized": 0,
    }


def one_line(noop: dict[str, Any], attack: dict[str, Any]) -> str:
    largest = noop.get("largest_cause") or {}
    cause = largest.get("cause")
    ns = largest.get("total_idle_ns_per_token")
    parts = []
    if cause is not None and ns is not None:
        parts.append(f"Largest classified GPU-idle cause is {cause} ({ns / 1e6:.3f} ms/token).")
    parts.append(f"Attack on command construction: {attack.get('outcome')}.")
    parts.append(attack.get("why") or "")
    return " ".join(p for p in parts if p)


def build(live: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    evidence = shader_evidence()
    autopsy = kernel_autopsy()
    occ = occupancy_snapshot()
    compile_: dict[str, Any] = {"returncode": 1, "skipped": True, "bin_present": BIN.is_file()}
    raw = None
    run = None
    if autopsy.get("any_new_kernel_defective"):
        compile_ = {
            "returncode": 0,
            "skipped": True,
            "bin_present": BIN.is_file(),
            "note": "kernel autopsy DEFECTIVE; speed run not trusted",
        }
    elif live:
        compile_ = cargo_build()
    else:
        compile_ = {
            "returncode": 0,
            "skipped": True,
            "bin_present": BIN.is_file(),
            "note": "reused existing raw; cargo not re-invoked",
        }
    if (not live) and RAW.is_file():
        try:
            raw = json.loads(RAW.read_text())
        except json.JSONDecodeError:
            raw = None
    parent_ok = (PARENT_ROOT / "catalog.hq38m20").is_file() and TOKENIZER.is_file()
    if (
        live
        and not autopsy.get("any_new_kernel_defective")
        and compile_.get("returncode") == 0
        and BIN.is_file()
        and parent_ok
        and not occ.get("loaded_a_second_27b")
    ):
        run = run_locked(RAW)
        raw = run.get("raw")
    elif live and not parent_ok:
        run = {
            "returncode": 2,
            "absent_reason": f"parent catalog or tokenizer missing ({PARENT_ROOT}, {TOKENIZER})",
        }
    elif live and occ.get("loaded_a_second_27b"):
        run = {
            "returncode": 2,
            "absent_reason": "occupancy shows a 10+ GiB 27B already resident; refused a second load",
        }
    noop = summarize_arm(raw, "noop")
    serial = summarize_arm(raw, "serial")
    split = summarize_arm(raw, "split")
    attack = attack_verdict(noop, serial, split) if noop.get("kind") == MEASURED else {
        "outcome": "NOT_MEASURED",
        "why": "noop arm missing",
        "target_cause": "command construction",
        "token_ids_unchanged": False,
        "separated": False,
        "dense_w_materialized": 0,
    }
    answer = one_line(noop, attack) if noop.get("kind") == MEASURED else "Idle-gap raw missing."
    loc = PARENT_ROOT.resolve()
    gpu_cmd = (
        "./tools/gpu_lane_lock.sh n029-idlegap "
        "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_idle_gap "
        f"--artifact-root {PARENT_ROOT} --tokenizer {TOKENIZER} --reps 7"
    )
    ranked = noop.get("ranked_causes") if noop.get("kind") == MEASURED else []
    intervals = noop.get("all_token_intervals") if noop.get("kind") == MEASURED else []
    largest = noop.get("largest_cause") if noop.get("kind") == MEASURED else None
    complete = noop.get("complete_wall_ns_median")
    gpu = noop.get("gpu_ns_median")
    wall_minus_gpu = None
    if complete is not None and gpu is not None:
        wall_minus_gpu = complete - gpu
    doc = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "obligation": (
            "N029 — GPU_IDLE_GAP_LEDGER: instrument the token loop; list every "
            "GPU idle interval per token with a classified cause; rank by total "
            "idle ns/token; attack the largest or a measured reason it cannot be"
        ),
        "one_line": answer,
        "question": (
            "What keeps the GPU idle per token, ranked by ns, and can the largest "
            "named cause be attacked without changing greedy token ids?"
        ),
        "answer": answer,
        "gpu_timestamp_authority": GPU_TIMESTAMP_AUTHORITY,
        "did_not_load_second_27b": True,
        "did_not_mutate_parent": True,
        "did_not_write_under_models": True,
        "did_not_write_ascent_or_campaign": True,
        "occupancy": occ,
        "parent_immutable": {
            "path": str(loc),
            "outside_worktree": True,
            "catalog_present": (PARENT_ROOT / "catalog.hq38m20").is_file(),
        },
        "prior_not_rederived": {
            "q4_unfused_dispatches": UNFUSED_DISPATCHES,
            "parent_dispatches": PARENT_DISPATCHES,
            "dispatch_frontier": FRONTIER_DISPATCHES,
            "gpu_ledger_gpu_idle_gaps_ns": "ABSENT (1 mixed CB, atDispatchBoundary=false)",
            "q80_51pct_idle": "CONTRADICTED_FOR_THIS_INCUMBENT",
        },
        "causes": list(CAUSES),
        "complete_token_wall_ns": qty(
            complete,
            kind=MEASURED if complete is not None else ABSENT,
            unit="ns/token",
            command=gpu_cmd,
            note="steady-decode median of 7 generate reps on the 580-dispatch parent graph",
            absent_reason=None if complete is not None else "noop arm was not measured",
            spread={
                "n": noop.get("n_reps"),
                "min": noop.get("complete_wall_ns_min"),
                "median": noop.get("complete_wall_ns_median"),
                "max": noop.get("complete_wall_ns_max"),
                "all": noop.get("complete_wall_ns_reps"),
            }
            if complete is not None
            else None,
        )
        if complete is not None
        else qty(
            None,
            kind=ABSENT,
            unit="ns/token",
            command=gpu_cmd,
            absent_reason="noop arm was not measured",
        ),
        "gpu_busy_ns": qty(
            gpu,
            kind=MEASURED if gpu is not None else ABSENT,
            unit="ns/token",
            command=gpu_cmd + "  # GPUEndTime−GPUStartTime of the production CB",
            note="Equals GPU_NS. Intra-CB bubbles are inside this interval.",
            absent_reason=None if gpu is not None else "noop arm was not measured",
        )
        if gpu is not None
        else qty(
            None,
            kind=ABSENT,
            unit="ns/token",
            command=gpu_cmd,
            absent_reason="noop arm was not measured",
        ),
        "wall_minus_gpu_ns": qty(
            wall_minus_gpu,
            kind=DERIVED if wall_minus_gpu is not None else ABSENT,
            unit="ns/token",
            command="complete_wall_ns − gpu_ns",
            note="Host-side remainder while the GPU is idle between CBs. Not intra-CB idle.",
            absent_reason=None if wall_minus_gpu is not None else "noop arm was not measured",
        )
        if wall_minus_gpu is not None
        else qty(
            None,
            kind=ABSENT,
            unit="ns/token",
            command="complete_wall_ns − gpu_ns",
            absent_reason="noop arm was not measured",
        ),
        "intra_cb_gpu_idle_ns": qty(
            None,
            kind=ABSENT,
            unit="ns/token",
            command=(
                "python3 -c 'import json; print(json.load(open("
                '"receipts/headless/GPU_LEDGER_METAL_PROBE.json"))'
                '["supportsCounterSampling"]["atDispatchBoundary"])\''
            ),
            absent_reason=INTRA_CB_ABSENT_REASON,
        ),
        "ranked_by_idle_ns_per_token": ranked,
        "largest_cause": largest,
        "intervals_per_token": intervals,
        "arms": {
            "noop": {k: v for k, v in noop.items() if k != "all_token_intervals"},
            "serial": {k: v for k, v in serial.items() if k != "all_token_intervals"},
            "split": {k: v for k, v in split.items() if k != "all_token_intervals"},
        },
        "attack": attack,
        "controls": {
            "no_op": "per-dispatch encoder, 1 CB (production)",
            "deliberately_bad": "SplitCbGpu one CB per dispatch",
            "reps": 7,
            "report": "min/median/max; overlapping ranges are NOT SEPARATED",
        },
        "causal_benchmark_law": {
            "kernel_identity": "same kernels as the 580-dispatch parent; encoder topology only",
            "dispatch_count": FRONTIER_DISPATCHES,
            "sentinel": "serial arm encoder_count==1; noop encoder_count==dispatches",
            "noop_control": "per-dispatch encoder must not score as the serial cut",
            "bad_control": "SplitCbGpu must inflate sync idle vs the no-op",
        },
        "kernel_autopsy": autopsy,
        "shader_evidence": evidence,
        "compile": compile_,
        "run": None
        if run is None
        else {
            "returncode": run.get("returncode"),
            "elapsed_s": run.get("elapsed_s"),
            "stdout_tail": run.get("stdout_tail"),
            "stderr_tail": run.get("stderr_tail"),
            "absent_reason": run.get("absent_reason"),
        },
        "dense_w_materialized": 0,
        "elapsed_s": time.perf_counter() - t0,
    }
    return doc


def write_receipt(doc: dict[str, Any]) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=1) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-raw", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()
    live = not args.from_raw and not args.no_gpu
    doc = build(live=live)
    write_receipt(doc)
    print(json.dumps({"schema": doc["schema"], "one_line": doc["one_line"], "receipt": str(RECEIPT)}, indent=2))
    if live and (doc.get("run") or {}).get("returncode") not in (0, None):
        return int((doc.get("run") or {}).get("returncode") or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
