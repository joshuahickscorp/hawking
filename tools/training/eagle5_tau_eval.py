#!/usr/bin/env python3
"""path-to-50 lever 3: τ-at-depth-K eval for the eagle5 v2 head.

Mirrors `eagle4/tau_eval.py`. Reports τ (mean accepted prefix length up
to depth K) by rolling the head autoregressively for K steps and
comparing each step's argmax to V2-Lite's argmax (computed from the
captured residual at that depth).

Acceptance gates per `reports/eagle5_v2_wiring_handoff.md` §8:
- τ-at-depth-4 ≥ 3.0  (better than eagle3's 2.15)
- depth-1 accept ≥ 85%
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import mlx.core as mx
except ImportError:
    print("ERROR: mlx not installed.", file=sys.stderr)
    sys.exit(1)

import numpy as np
import pyarrow.parquet as pq

import eagle5_train as e5


def _load_consecutive_windows(
    shard: Path,
    window: int,
    capture_layer: int,
    max_windows: int,
    seen_fp: set | None = None,
) -> list[dict]:
    """Return up to `max_windows` windows of `window` consecutive
    (token, residual, intermediate) records sourced from a single shard
    (rows are independent sequences in the corpus, so a window must
    come from one row).

    Dedups by token-fingerprint to keep eval metric honest — the corpus
    has ~60% duplicate rows from the watchdog restart bug (per
    corpus_complete_analysis_landed.md). Without dedup, τ is inflated
    by the easy-to-predict duplicates. Pass `seen_fp` to maintain dedup
    state across shards.
    """
    t = pq.read_table(shard)
    windows = []
    seen = seen_fp if seen_fp is not None else set()
    for i in range(t.num_rows):
        r = {c: t[c][i].as_py() for c in t.column_names}
        ex = e5._extract_row(r, capture_layer)
        if ex is None:
            continue
        fp = ex["prev_tokens"][:64].tobytes()
        if fp in seen:
            continue
        seen.add(fp)
        n = len(ex["prev_tokens"])
        for off in range(0, n - window + 1, window):
            windows.append({
                "prev": ex["prev_tokens"][off : off + window],
                "next": ex["next_tokens"][off : off + window],
                "residual": ex["residual"][off : off + window],
                "intermediate": ex["intermediate"][off : off + window],
            })
            if len(windows) >= max_windows:
                return windows
    return windows


def evaluate_tau(
    ckpt: Path,
    frozen: Path,
    corpus_dir: Path,
    depth: int,
    max_windows: int,
    capture_layer: int,
    with_sparsity: bool,
) -> dict:
    head = e5.build_head(frozen, with_sparsity)
    e5.load_ckpt(head, ckpt)

    shards = sorted(corpus_dir.glob("shard_*.parquet"))
    all_windows: list[dict] = []
    seen_fp: set = set()
    # Wall-clock optimization #3 (2026-05-22): parallel shard read for
    # τ-eval. Per-shard work is independent up to the shared `seen_fp`
    # dedup set — we apply dedup serially after parallel load so the set
    # state stays single-threaded.
    max_workers = min(8, (os.cpu_count() or 4))

    def _read_one(shard: Path) -> list[dict]:
        # Use a per-shard private seen_fp=set() so the worker just
        # de-dupes within its own shard; cross-shard dedup happens below.
        return _load_consecutive_windows(
            shard, depth + 1, capture_layer, max_windows, seen_fp=set()
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for shard_windows in pool.map(_read_one, shards):
            for w in shard_windows:
                fp = w["prev"][:64].tobytes()
                if fp in seen_fp:
                    continue
                seen_fp.add(fp)
                all_windows.append(w)
                if len(all_windows) >= max_windows:
                    break
            if len(all_windows) >= max_windows:
                break
    if not all_windows:
        return {"error": "no windows loaded"}
    print(f"[τ-eval] {len(all_windows)} windows × depth {depth}", flush=True)

    W = len(all_windows)
    prev_np = np.stack([w["prev"] for w in all_windows]).astype(np.int32)
    next_np = np.stack([w["next"] for w in all_windows]).astype(np.int32)
    res_np  = np.stack([w["residual"] for w in all_windows]).astype(np.float32)
    inter_np = np.stack([w["intermediate"] for w in all_windows]).astype(np.float32)

    accepted_len = np.zeros(W, dtype=np.int32)
    per_pos_accept = np.zeros(depth, dtype=np.int64)
    cur_prev = mx.array(prev_np[:, 0]).reshape(1, W)

    for d in range(depth):
        residual_d = mx.array(res_np[:, d, :]).reshape(1, W, e5.HIDDEN_DIM)
        inter_d = mx.array(inter_np[:, d, :]).reshape(1, W, e5.HIDDEN_DIM)
        tok_logits, _, _, _ = head(cur_prev, residual_d, inter_d)
        baseline = mx.fast.rms_norm(
            residual_d.reshape(W, e5.HIDDEN_DIM), head._output_norm, e5.RMS_EPS
        )
        target_logits = baseline @ head._lm_head.astype(mx.float32)
        mx.eval(tok_logits, target_logits)

        head_arg = np.argmax(np.array(tok_logits).reshape(W, -1), axis=-1)
        target_arg = np.array(mx.argmax(target_logits, axis=-1))
        match = head_arg == target_arg

        still_accepting = accepted_len == d
        accepted_step = still_accepting & match
        accepted_len = accepted_len + accepted_step.astype(np.int32)
        per_pos_accept[d] = int(accepted_step.sum())
        cur_prev = mx.array(head_arg.astype(np.int32)).reshape(1, W)

    return {
        "windows": W,
        "depth": depth,
        "tau": float(accepted_len.mean()),
        "full_accept_rate": float((accepted_len == depth).sum()) / W,
        "per_pos_accept_rate": (per_pos_accept / W).tolist(),
    }


def main() -> int:
    p = argparse.ArgumentParser(prog="eagle5_tau_eval")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--frozen", required=True, type=Path)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--max-windows", type=int, default=2000)
    p.add_argument("--capture-layer", type=int, default=25)
    p.add_argument("--sparsity-head", choices=["proxy", "off"], default="proxy")
    args = p.parse_args()
    r = evaluate_tau(
        args.ckpt, args.frozen, args.corpus, args.depth, args.max_windows,
        args.capture_layer, with_sparsity=(args.sparsity_head != "off"),
    )
    print(json.dumps({k: v for k, v in r.items() if k != "per_pos_accept_rate"}, indent=2))
    if "per_pos_accept_rate" in r:
        print("per-position accept rate (given prior prefix accepted):")
        for d, rate in enumerate(r["per_pos_accept_rate"]):
            print(f"  depth {d+1}: {rate*100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
