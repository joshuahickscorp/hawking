"""τ-at-depth-K — the spec-decode-honest eval.

Rolls the head autoregressively for K steps, feeding its own argmax as the
next `prev_token`, and counts how many tokens are accepted before the head
disagrees with V2-Lite's argmax at any depth. τ is the mean accepted-prefix
length. Captured hiddens stay tied to corpus positions — same simplification
dismantle's runtime makes during verify.

  python tau_eval.py eval --ckpt ckpt/best.npz --frozen v2lite_frozen.npz \\
                          --parquet data/heldout/*.parquet --depth 4
  python tau_eval.py compare --eagle3 ckpt/eagle3.npz --eagle4 ckpt/eagle4.npz \\
                             --frozen v2lite_frozen.npz \\
                             --parquet data/heldout/*.parquet --depth 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow.parquet as pq

import eagle4


def _load_consecutive_windows(shard: Path, window: int, max_windows: int):
    """Return windows of `window` consecutive (sample_id, position) records."""
    t = pq.read_table(shard)
    rows = [
        {k: t[k][i].as_py() for k in t.column_names}
        for i in range(t.num_rows)
    ]
    rows.sort(key=lambda r: (r["sample_id"], r["position"]))
    windows = []
    cur_sid = None
    cur = []
    for r in rows:
        if r["sample_id"] != cur_sid:
            cur_sid = r["sample_id"]
            cur = []
        cur.append(r)
        if len(cur) == window:
            windows.append(cur)
            cur = []
            if len(windows) >= max_windows:
                return windows
    return windows


def _unpack_window(window, H):
    """(W, K+1) → tensors per field."""
    W = len(window)
    if W == 0:
        return None
    K = len(window[0])  # window length

    def stack(field, dtype):
        return np.frombuffer(b"".join(r[field] for w in window for r in w), dtype=dtype).reshape(W, K, -1)

    prev = np.array([[r["prev_token"] for r in w] for w in window], dtype=np.int32)
    nxt = np.array([[r["next_token"] for r in w] for w in window], dtype=np.int32)
    low = stack("hidden_low", np.float16)
    mid = stack("hidden_mid", np.float16)
    hi = stack("hidden_high", np.float16)
    sh = stack("shared_hidden", np.float16)
    return prev, nxt, low, mid, hi, sh


def evaluate_tau(
    ckpt: Path,
    frozen: Path,
    parquet_paths: list[Path],
    depth: int = 4,
    max_windows: int = 2000,
) -> dict:
    head = eagle4.build_head(frozen)
    eagle4.load_ckpt(head, ckpt)
    H = eagle4.HIDDEN_DIM

    all_windows: list[list[dict]] = []
    for shard in parquet_paths:
        all_windows.extend(
            _load_consecutive_windows(shard, depth + 1, max_windows - len(all_windows))
        )
        if len(all_windows) >= max_windows:
            break
    if not all_windows:
        return {"error": "no windows loaded"}
    print(f"[τ-eval] {len(all_windows)} windows × depth {depth}", flush=True)

    unpacked = _unpack_window(all_windows, H)
    prev_np, nxt_np, low_np, mid_np, hi_np, sh_np = unpacked
    W = prev_np.shape[0]

    accepted_len = np.zeros(W, dtype=np.int32)
    per_pos_accept = np.zeros(depth, dtype=np.int64)
    cur_prev = mx.array(prev_np[:, 0]).reshape(1, W)

    for d in range(depth):
        lo_a = mx.array(low_np[:, d, :]).astype(mx.float32).reshape(1, W, H)
        md_a = mx.array(mid_np[:, d, :]).astype(mx.float32).reshape(1, W, H)
        hi_a = mx.array(hi_np[:, d, :]).astype(mx.float32).reshape(1, W, H)
        sh_a = mx.array(sh_np[:, d, :]).astype(mx.float32).reshape(1, W, H)

        tok_logits, _, _, _ = head(cur_prev, lo_a, md_a, hi_a, sh_a)
        baseline = mx.fast.rms_norm(hi_a.reshape(W, H), head._output_norm, eagle4.RMS_EPS)
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


def compare(eagle3_ckpt, eagle4_ckpt, frozen, parquet_paths, depth, max_windows):
    print(f"\n=== EAGLE-3 baseline τ-at-depth-{depth} ===", flush=True)
    r_e3 = _evaluate_eagle3_tau(eagle3_ckpt, frozen, parquet_paths, depth, max_windows)
    for k, v in r_e3.items():
        print(f"  {k:24s} {v}")
    print(f"\n=== EAGLE-4 τ-at-depth-{depth} ===", flush=True)
    r_e4 = evaluate_tau(Path(eagle4_ckpt), Path(frozen), parquet_paths, depth, max_windows)
    for k, v in r_e4.items():
        if k != "per_pos_accept_rate":
            print(f"  {k:24s} {v}")
        else:
            print(f"  per_pos_accept_rate:")
            for d, rate in enumerate(v):
                print(f"    depth {d+1}: {rate*100:.2f}%")
    return {"eagle3": r_e3, "eagle4": r_e4}


def _evaluate_eagle3_tau(ckpt: Path, frozen: Path, parquet_paths, depth: int, max_windows: int) -> dict:
    import bench

    head = bench.build_eagle3(Path(frozen))
    eagle4.load_ckpt(head, Path(ckpt))
    H = eagle4.HIDDEN_DIM

    all_windows: list[list[dict]] = []
    for shard in parquet_paths:
        all_windows.extend(
            _load_consecutive_windows(Path(shard), depth + 1, max_windows - len(all_windows))
        )
        if len(all_windows) >= max_windows:
            break
    if not all_windows:
        return {"error": "no windows"}
    unpacked = _unpack_window(all_windows, H)
    prev_np, _, low_np, mid_np, hi_np, _ = unpacked
    W = prev_np.shape[0]

    accepted_len = np.zeros(W, dtype=np.int32)
    per_pos_accept = np.zeros(depth, dtype=np.int64)
    cur_prev = mx.array(prev_np[:, 0]).reshape(1, W)

    for d in range(depth):
        lo_a = mx.array(low_np[:, d, :]).astype(mx.float32).reshape(1, W, H)
        md_a = mx.array(mid_np[:, d, :]).astype(mx.float32).reshape(1, W, H)
        hi_a = mx.array(hi_np[:, d, :]).astype(mx.float32).reshape(1, W, H)

        tok_logits, _ = head(cur_prev, lo_a, md_a, hi_a)
        baseline = mx.fast.rms_norm(hi_a.reshape(W, H), head._output_norm, eagle4.RMS_EPS)
        target_logits = baseline @ head._lm_head.astype(mx.float32)
        mx.eval(tok_logits, target_logits)

        head_arg = np.argmax(np.array(tok_logits).reshape(W, -1), axis=-1)
        target_arg = np.array(mx.argmax(target_logits, axis=-1))
        match = head_arg == target_arg
        still = accepted_len == d
        accepted_step = still & match
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
    p = argparse.ArgumentParser(prog="tau_eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    ep = sub.add_parser("eval", help="EAGLE-4 head τ-at-depth-K")
    ep.add_argument("--ckpt", type=Path, required=True)
    ep.add_argument("--frozen", type=Path, required=True)
    ep.add_argument("--parquet", nargs="+", type=Path, required=True)
    ep.add_argument("--depth", type=int, default=4)
    ep.add_argument("--max-windows", type=int, default=2000)

    cp = sub.add_parser("compare", help="EAGLE-3 vs EAGLE-4 τ-at-depth-K")
    cp.add_argument("--eagle3", type=Path, required=True)
    cp.add_argument("--eagle4", type=Path, required=True)
    cp.add_argument("--frozen", type=Path, required=True)
    cp.add_argument("--parquet", nargs="+", type=Path, required=True)
    cp.add_argument("--depth", type=int, default=4)
    cp.add_argument("--max-windows", type=int, default=2000)
    cp.add_argument("--out", type=Path, default=Path("tau_results.json"))

    args = p.parse_args()
    if args.cmd == "eval":
        r = evaluate_tau(args.ckpt, args.frozen, args.parquet, args.depth, args.max_windows)
        print(json.dumps({k: v for k, v in r.items() if k != "per_pos_accept_rate"}, indent=2))
        if "per_pos_accept_rate" in r:
            print("per-position accept rate (given prior prefix accepted):")
            for d, rate in enumerate(r["per_pos_accept_rate"]):
                print(f"  depth {d+1}: {rate*100:.2f}%")
    elif args.cmd == "compare":
        r = compare(args.eagle3, args.eagle4, args.frozen, args.parquet, args.depth, args.max_windows)
        args.out.write_text(json.dumps(r, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
