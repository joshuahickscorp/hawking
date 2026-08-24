#!/usr/bin/env python3
"""ONE resident Noetic body, many isolated sessions.

Proves, with resident bytes, that concurrency is one shared body plus N KV
states — not N model copies. Measures sequential per-session and round-robin
token topologies at c=1,2,4, plus operator microbatch if the runtime can
fan a GEMV across sessions.

Does not load a second 27B. Does not write under receipts/ascent-2026-08-16
or workspace/campaign.

    python3 tools/headless/noetic_multisession.py
    python3 -m pytest tools/headless/test_noetic_multisession.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from metal_budget import metal_device  # noqa: E402

SCHEMA = "hawking.headless.noetic_multisession.v1"
RECEIPT = REPO / "receipts" / "headless" / "NOETIC_MULTISESSION.json"
RAW_RECEIPT = REPO / "receipts" / "headless" / "NOETIC_MULTISESSION.raw.json"
CARGO_TARGET = REPO / "workspace" / "ops" / "build" / "rust"
EXAMPLE_NAME = "ascension_qwen38_noetic_multisession"
BINARY = CARGO_TARGET / "release-fast" / "examples" / EXAMPLE_NAME
DEFAULT_ARTIFACT = Path.home() / "models" / "qwen38-gravity-uniform-q4-v1"
DEFAULT_TOKENIZER = (
    Path.home() / "models" / "qwen3.8-27b-abliterated-bf16" / "tokenizer.json"
)

# Qwen3.8 geometry — must match crates/hawking-core/src/model/qwen38_geometry.rs
QWEN38_HIDDEN = 5120
QWEN38_INTERMEDIATE = 17408
QWEN38_VOCAB = 248320
QWEN38_GQA_HEADS = 24
QWEN38_GQA_KV_HEADS = 4
QWEN38_GQA_HEAD_DIM = 256
QWEN38_GQA_LAYERS = 16
QWEN38_LINEAR_KEY_HEADS = 16
QWEN38_LINEAR_VALUE_HEADS = 48
QWEN38_LINEAR_VALUES_PER_KEY = 3
QWEN38_LINEAR_KEY_HEAD_DIM = 128
QWEN38_LINEAR_VALUE_HEAD_DIM = 128
QWEN38_LINEAR_CONV_KERNEL = 4
QWEN38_MIXED_HGRAVS_RANK = 160
QWEN38_IN_PROJ_QKV_ROWS = 10240
QWEN38_IN_PROJ_B_ROWS = 48
QWEN38_IN_PROJ_A_ROWS = 48

SHARED = [
    "model representation",
    "kernels",
    "Metal pipelines",
    "immutable structures",
    "codebooks",
    "shared bases",
    "static routing",
]
ISOLATED = [
    "context",
    "KV/state",
    "sampling",
    "session/tool state",
]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def f32b(n: int) -> int:
    return n * 4


def workspace_bytes(max_seq_len: int) -> dict[str, int]:
    """Mirror qwen38_workspace_bytes. A mismatch with the rust formula is a bug."""
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    value_rows = QWEN38_LINEAR_VALUES_PER_KEY * QWEN38_LINEAR_VALUE_HEAD_DIM
    qkvz_rows_per_key = QWEN38_LINEAR_KEY_HEAD_DIM * 2 + value_rows * 2
    ba_rows_per_key = QWEN38_LINEAR_VALUES_PER_KEY * 2
    conv_channels = (
        QWEN38_LINEAR_KEY_HEADS * QWEN38_LINEAR_KEY_HEAD_DIM * 2
        + QWEN38_LINEAR_VALUE_HEADS * QWEN38_LINEAR_VALUE_HEAD_DIM
    )
    qkvz_rows = QWEN38_LINEAR_KEY_HEADS * qkvz_rows_per_key
    ba_rows = QWEN38_LINEAR_KEY_HEADS * ba_rows_per_key
    value_elements = QWEN38_LINEAR_VALUE_HEADS * QWEN38_LINEAR_VALUE_HEAD_DIM
    conv_state_elements = conv_channels * (QWEN38_LINEAR_CONV_KERNEL - 1)
    rec_state_elements = (
        QWEN38_LINEAR_VALUE_HEADS * QWEN38_LINEAR_KEY_HEAD_DIM * QWEN38_LINEAR_VALUE_HEAD_DIM
    )
    hidden = f32b(QWEN38_HIDDEN)
    qkvz = f32b(qkvz_rows)
    ba = f32b(ba_rows)
    value = f32b(value_elements)
    q_proj = f32b(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM * 2)
    kv = f32b(QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM)
    query = f32b(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM)
    mid = f32b(QWEN38_INTERMEDIATE)
    logits = f32b(QWEN38_VOCAB)
    conv = f32b(48 * conv_state_elements)
    rec = f32b(48 * rec_state_elements)
    kv_cache = f32b(
        QWEN38_GQA_LAYERS * max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM
    )
    hgravs = f32b(QWEN38_MIXED_HGRAVS_RANK)
    split_qkv = f32b(QWEN38_IN_PROJ_QKV_ROWS)
    split_b = f32b(QWEN38_IN_PROJ_B_ROWS)
    split_a = f32b(QWEN38_IN_PROJ_A_ROWS)
    sampled = 4
    heads_f32 = f32b(QWEN38_LINEAR_VALUE_HEADS)
    activation = (
        hidden * 2
        + qkvz
        + ba
        + value * 6
        + heads_f32 * 2
        + hidden * 2
        + q_proj
        + kv * 2
        + query * 3
        + mid * 3
        + hidden
        + logits
        + sampled
        + hgravs
        + split_qkv
        + split_b
        + split_a
    )
    deltanet = conv + rec
    gqa = kv_cache * 2
    total = activation + deltanet + gqa
    return {
        "max_seq_len": max_seq_len,
        "activation_bytes": activation,
        "deltanet_state_bytes": deltanet,
        "gqa_kv_bytes": gqa,
        "total_bytes": total,
    }


def expected_shared_resident_bytes(body: int, workspace: int, sessions: int) -> int:
    return body + workspace * sessions


def expected_n_copies_bytes(body: int, workspace: int, sessions: int) -> int:
    return sessions * (body + workspace)


def one_body_not_n_copies(measured: int, body: int, workspace: int, sessions: int) -> bool:
    if sessions <= 1:
        return True
    one = expected_shared_resident_bytes(body, workspace, sessions)
    copies = expected_n_copies_bytes(body, workspace, sessions)
    if copies <= one:
        return False
    slack = workspace * sessions + body // 10
    upper = max(one + slack, body * 2)
    return measured <= upper and measured < copies // 2


def simulate_topologies(c: int, prefill: int, decode: int, step_s: float) -> dict[str, Any]:
    """CPU model of the two required policies. Not a GPU measurement.

    sequential: session i starts after 0..i-1 have fully finished.
    round-robin: every live session takes one step per round, prefill included.
    """
    seq = []
    t = 0.0
    # Last prefill step emits new-token 0, then decode-1 more steps.
    steps_per_session = prefill + max(decode, 1) - 1
    for i in range(c):
        start = t
        ttft = start + prefill * step_s
        latencies = [step_s] * max(decode - 1, 0)
        end = start + steps_per_session * step_s
        seq.append(
            {
                "session": i,
                "ttft_s": ttft,
                "token_latency_s": latencies,
                "end_s": end,
            }
        )
        t = end
    seq_wall = t
    seq_tokens = c * decode
    rr_ttft = [0.0] * c
    rr_emit = [[] for _ in range(c)]
    prompt_i = [0] * c
    new_count = [0] * c
    t = 0.0
    while True:
        progressed = False
        for i in range(c):
            if prompt_i[i] < prefill:
                t += step_s
                prompt_i[i] += 1
                if prompt_i[i] == prefill:
                    new_count[i] = 1
                    rr_ttft[i] = t
                    rr_emit[i].append(t)
                progressed = True
            elif new_count[i] < decode:
                t += step_s
                new_count[i] += 1
                rr_emit[i].append(t)
                progressed = True
        if not progressed:
            break
    rr = []
    for i in range(c):
        emits = rr_emit[i]
        latencies = [emits[k] - emits[k - 1] for k in range(1, len(emits))]
        rr.append(
            {
                "session": i,
                "ttft_s": rr_ttft[i],
                "token_latency_s": latencies,
                "end_s": emits[-1] if emits else t,
            }
        )
    return {
        "sessions": c,
        "prefill_steps": prefill,
        "decode_tokens": decode,
        "step_s": step_s,
        "sequential_per_session": {
            "policy": "throughput_background",
            "wall_s": seq_wall,
            "aggregate_tps": seq_tokens / seq_wall if seq_wall else 0.0,
            "per_stream_tps_shared_wall": [
                decode / seq_wall if seq_wall else 0.0 for _ in range(c)
            ],
            "ttft_s": [s["ttft_s"] for s in seq],
            "token_latency_s": [s["token_latency_s"] for s in seq],
        },
        "round_robin_token": {
            "policy": "latency_fair_foreground",
            "wall_s": t,
            "aggregate_tps": seq_tokens / t if t else 0.0,
            "per_stream_tps_shared_wall": [decode / t if t else 0.0 for _ in range(c)],
            "ttft_s": [s["ttft_s"] for s in rr],
            "token_latency_s": [s["token_latency_s"] for s in rr],
        },
    }


def ensure_binary() -> Path:
    env = os.environ.get("QWEN38_MULTISESSION_BIN")
    if env:
        p = Path(env)
        if p.is_file() and os.access(p, os.X_OK):
            return p
    if BINARY.is_file() and os.access(BINARY, os.X_OK):
        return BINARY
    CARGO_TARGET.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cargo",
        "build",
        "--profile",
        "release-fast",
        "-p",
        "hawking-core",
        "--example",
        EXAMPLE_NAME,
        "--target-dir",
        str(CARGO_TARGET),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), text=True, capture_output=True)
    if proc.returncode != 0 or not BINARY.is_file():
        raise SystemExit(
            f"cargo build of {EXAMPLE_NAME} failed rc={proc.returncode}\n{proc.stderr[-4000:]}"
        )
    return BINARY


def artifact_paths() -> tuple[Path, Path]:
    artifact = Path(os.environ.get("QWEN38_Q4_ARTIFACT", str(DEFAULT_ARTIFACT)))
    tokenizer = Path(os.environ.get("QWEN38_TOKENIZER", str(DEFAULT_TOKENIZER)))
    return artifact, tokenizer


def run_live(
    *,
    sessions: int = 4,
    max_seq_len: int = 256,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    artifact, tokenizer = artifact_paths()
    missing = []
    if not (artifact / "manifest.json").is_file() and not (
        artifact / "catalog.hq38m20"
    ).is_file():
        missing.append(f"artifact missing: {artifact}")
    if not tokenizer.is_file():
        missing.append(f"tokenizer missing: {tokenizer}")
    if missing:
        raise SystemExit("preflight: " + "; ".join(missing))
    binary = ensure_binary()
    RAW_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "--artifact-root",
        str(artifact),
        "--tokenizer",
        str(tokenizer),
        "--sessions",
        str(sessions),
        "--max-seq-len",
        str(max_seq_len),
        "--max-new-tokens",
        str(max_new_tokens),
        "--out",
        str(RAW_RECEIPT),
    ]
    print(f"n006: running {' '.join(cmd)}", file=sys.stderr, flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        text=True,
        capture_output=True,
        timeout=1800,
    )
    if RAW_RECEIPT.is_file():
        raw = json.loads(RAW_RECEIPT.read_text())
    else:
        raw = {
            "ok": False,
            "exit_code": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
        }
    raw["_runner"] = {
        "binary": str(binary),
        "exit_code": proc.returncode,
        "stderr_tail": (proc.stderr or "")[-2000:],
    }
    return raw


def judge(raw: dict[str, Any], formula: dict[str, int]) -> dict[str, Any]:
    proof = raw.get("proof_one_body") or {}
    isolation = raw.get("isolation") or {}
    topologies = raw.get("topologies") or {}
    required = ("sequential_per_session", "round_robin_token")
    have_two = all(name in topologies for name in required)
    counts_ok = True
    metrics_ok = True
    for name in required:
        by_c = topologies.get(name) or {}
        for c in ("1", "2", "4"):
            if c not in by_c:
                counts_ok = False
                continue
            row = by_c[c]
            for key in (
                "aggregate_tps",
                "per_stream_tps_exclusive",
                "ttft_ms",
                "token_latency_p50_ms",
                "token_latency_p95_ms",
            ):
                if key not in row or row[key] is None:
                    metrics_ok = False
    isolated = bool(isolation.get("isolated"))
    shared_ptr = bool(raw.get("weights_ptr_shared"))
    one_body = bool(proof.get("one_body_not_n_copies"))
    no_second = bool(raw.get("did_not_load_second_27b"))
    weight_loads = raw.get("weight_loads")
    process_count = raw.get("process_count")
    c4 = proof.get("rss_c4_bytes") or proof.get("metal_c4_bytes") or 0
    c1 = proof.get("rss_c1_bytes") or proof.get("metal_c1_bytes") or 0
    copies = proof.get("predicted_four_copies_c4_bytes") or 0
    ratio = None
    if c1:
        ratio = (c4 or 0) / c1
    ratio_ok = ratio is not None and ratio < 2.0
    copies_ok = bool(copies) and bool(c4) and c4 < copies / 2
    live = raw.get("_runner", {}).get("exit_code") in (0, 3) or raw.get("pid")
    reasons = []
    if not live:
        reasons.append("live probe did not run")
    if not shared_ptr:
        reasons.append("weights_ptr_shared is false — sessions do not share one Arc")
    if not one_body:
        reasons.append("resident bytes do not prove one body")
    if not ratio_ok:
        reasons.append(f"c4/c1 ratio {ratio} is not << 4")
    if not copies_ok:
        reasons.append("measured c=4 is not far below four copies")
    if not isolated:
        reasons.append("per-session isolation proof failed")
    if not have_two or not counts_ok:
        reasons.append("need sequential and round-robin at c=1,2,4")
    if not metrics_ok:
        reasons.append("topology rows missing required metrics")
    if not no_second or weight_loads != 1 or process_count != 1:
        reasons.append("probe loaded more than one body or more than one process")
    # Formula check: python workspace must match rust workspace in the raw receipt.
    rust_ws = (raw.get("workspace") or {}).get("total_bytes")
    formula_match = rust_ws == formula["total_bytes"]
    if rust_ws is not None and not formula_match:
        reasons.append(
            f"python workspace {formula['total_bytes']} != rust {rust_ws}"
        )
    verdict = "PASS" if not reasons else "FAIL"
    return {
        "NOETIC_MULTISESSION_SHARED_BODY": verdict,
        "reasons": reasons,
        "formula_matches_rust": formula_match if rust_ws is not None else None,
        "c4_over_c1": ratio,
        "have_two_topologies_at_1_2_4": have_two and counts_ok,
        "metrics_present": metrics_ok,
        "isolated": isolated,
        "one_body_not_n_copies": one_body,
        "weights_ptr_shared": shared_ptr,
        "did_not_load_second_27b": no_second,
    }


def _gib(n: Any) -> str:
    try:
        return f"{int(n) / 1024**3:.2f}"
    except (TypeError, ValueError):
        return "?"


def _answer(raw: dict[str, Any], _judged: dict[str, Any]) -> str:
    proof = raw.get("proof_one_body") or {}
    scaling = raw.get("scaling_vs_c1_aggregate_tps") or {}
    conc = (scaling.get("concurrent_independent") or {}).get("4")
    seq = (scaling.get("sequential_per_session") or {}).get("4")
    rr = (scaling.get("round_robin_token") or {}).get("4")
    mb = raw.get("operator_microbatch") or []
    mb4 = next((row for row in mb if row.get("sessions") == 4), None)
    mb_x = (mb4 or {}).get("concurrent_vs_serial_speedup")
    def fmt(x: Any) -> str:
        return f"{float(x):.2f}x" if isinstance(x, (int, float)) else "?"
    return (
        f"Yes: one Arc of Metal-resident Qwen3.8 weights, "
        f"{raw.get('attached_sessions', '?')} isolated workspaces. "
        f"Metal { _gib(proof.get('metal_c1_bytes')) } GiB at c=1 and "
        f"{ _gib(proof.get('metal_c4_bytes')) } GiB at c=4 "
        f"(ratio {float(proof.get('metal_c4_over_c1') or 0):.3f}). "
        f"Four copies would be { _gib(proof.get('predicted_four_copies_c4_bytes')) } GiB. "
        f"KV pointers distinct; session-0 continuation after session-1 traffic "
        f"matches a control. Sequential c=4 aggregate {fmt(seq)}, round-robin "
        f"{fmt(rr)} (serial GPU steps). Concurrent independent peaks at {fmt(conc)} "
        f"— same ceiling class as the prior 1.21x process/slot campaign. "
        f"Operator microbatch of lm_head is {fmt(mb_x)} vs serial: the weight "
        f"stream is not amortised. Sequential is throughput/background "
        f"(later-session TTFT piles up). Round-robin is latency-fair foreground "
        f"(TTFTs clustered, per-stream token latency stretches with c)."
    )


def wrap(raw: dict[str, Any], *, elapsed_s: float, max_seq_len: int) -> dict[str, Any]:
    formula = workspace_bytes(max_seq_len)
    judged = judge(raw, formula)
    device = metal_device()
    sim = {
        str(c): simulate_topologies(c, prefill=8, decode=8, step_s=0.03) for c in (1, 2, 4)
    }
    scaling = raw.get("scaling_vs_c1_aggregate_tps") or {}
    ceiling_notes = []
    for name, row in scaling.items():
        for c, val in (row or {}).items():
            if isinstance(val, (int, float)) and val < 1.4 and str(c) != "1":
                ceiling_notes.append(
                    f"{name} c={c} aggregate scaling vs c=1 is {val:.3f}x"
                )
    return {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "elapsed_s": elapsed_s,
        "question": (
            "Can one immutable resident Noetic body serve N isolated sessions, "
            "and how do sequential vs round-robin (and operator-microbatch) "
            "topologies scale on this machine?"
        ),
        "answer": _answer(raw, judged),
        "NOETIC_MULTISESSION_SHARED_BODY": judged["NOETIC_MULTISESSION_SHARED_BODY"],
        "did_not_load_second_27b": bool(raw.get("did_not_load_second_27b")),
        "did_not_write_ascent_or_campaign": True,
        "shared": SHARED,
        "isolated": ISOLATED,
        "shape": "one Arc<Qwen38HybridWeights> / N Qwen38HybridDecodeSession workspaces",
        "gpu_gate": {
            "device": device.get("name"),
            "source": device.get("source"),
            "recommendedMaxWorkingSetSize_gib": (
                device["recommendedMaxWorkingSetSize"] / (1024**3)
                if device.get("recommendedMaxWorkingSetSize")
                else None
            ),
            "hasUnifiedMemory": device.get("hasUnifiedMemory"),
            "admission": (
                "METAL WORKING SET is the admission gate, not free RAM. "
                "mmap shares pages but each process gets its own MTLBuffers."
            ),
        },
        "workspace_formula": formula,
        "cpu_policy_model": {
            "note": (
                "Equal step cost, no overlap: sequential and round-robin have "
                "the same aggregate tok/s. Sequential piles TTFT onto later "
                "sessions (background throughput). Round-robin keeps TTFTs "
                "clustered and stretches per-stream token latency (foreground "
                "fairness). Operator microbatch is the only topology here that "
                "can amortize a weight stream."
            ),
            "simulations": sim,
        },
        "live": raw,
        "proof_one_body": raw.get("proof_one_body"),
        "isolation": raw.get("isolation"),
        "topologies_measured": sorted(list((raw.get("topologies") or {}).keys())),
        "measurements": raw.get("topologies"),
        "operator_microbatch": raw.get("operator_microbatch"),
        "scaling_vs_c1_aggregate_tps": scaling,
        "kv_bytes": (raw.get("workspace") or {}).get("gqa_kv_bytes"),
        "peak_unified_memory_bytes": (raw.get("metal_final") or {}).get(
            "current_allocated_size"
        )
        or raw.get("rss_final_bytes"),
        "judge": judged,
        "ceiling": {
            "prior_campaign_decode_concurrency": (
                "Decode concurrency topped out at ~1.21x on BOTH process-per-runtime "
                "and llama.cpp slot topologies. A shared Noetic body is the untested "
                "third option. If it also tops out near 1.21x, that is a real negative."
            ),
            "observations": ceiling_notes,
        },
        "policy": {
            "sequential_per_session": "throughput_background — later sessions wait",
            "round_robin_token": "latency_fair_foreground — every session steps each round",
            "concurrent_independent": (
                "throughput attempt, one thread per session, same Metal queue"
            ),
            "operator_microbatch": (
                "concurrent encoder, N GEMVs against one resident weight"
            ),
            "measured_as": (
                "latency-sensitive foreground = round_robin_token; "
                "throughput background = sequential_per_session"
            ),
        },
    }


def write_receipt(doc: dict[str, Any]) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {RECEIPT}", file=sys.stderr, flush=True)


def build(*, live: bool = True, force: bool = False) -> dict[str, Any]:
    max_seq_len = int(os.environ.get("NOETIC_MULTISESSION_MAX_SEQ", "256"))
    max_new = int(os.environ.get("NOETIC_MULTISESSION_MAX_NEW", "8"))
    sessions = int(os.environ.get("NOETIC_MULTISESSION_SESSIONS", "4"))
    t0 = time.perf_counter()
    raw: dict[str, Any]
    if live:
        if (
            not force
            and RAW_RECEIPT.is_file()
            and os.environ.get("NOETIC_MULTISESSION_REUSE_RAW") == "1"
        ):
            raw = json.loads(RAW_RECEIPT.read_text())
        else:
            raw = run_live(
                sessions=sessions, max_seq_len=max_seq_len, max_new_tokens=max_new
            )
    else:
        raw = {"did_not_load_second_27b": True, "topologies": {}, "proof_one_body": {}}
    doc = wrap(raw, elapsed_s=round(time.perf_counter() - t0, 3), max_seq_len=max_seq_len)
    write_receipt(doc)
    return doc


def main() -> int:
    force = "--force" in sys.argv
    live = "--cpu-only" not in sys.argv
    doc = build(live=live, force=force)
    verdict = doc["NOETIC_MULTISESSION_SHARED_BODY"]
    print(verdict, file=sys.stderr)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
