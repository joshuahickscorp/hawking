#!/usr/bin/env python3
"""path-to-50 lever 3: q4 quantize the eagle5 v2 head + parity check.

Mirrors `eagle4/q4_parity.py`. Quantizes a trained head's linear-layer
weights to q4 (group_size=64) via MLX's `mx.quantize`; dequantizes and
runs argmax over a held-out corpus shard to verify ≥ 99% match with the
bf16 source.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    print("ERROR: mlx not installed.", file=sys.stderr)
    sys.exit(1)

import numpy as np
import pyarrow.parquet as pq

import eagle5_train as e5


def _quantize_npz(in_path: Path, out_path: Path, group_size: int = 64, bits: int = 4):
    src = np.load(in_path, allow_pickle=False)
    out = {"__quant_group_size__": np.int32(group_size), "__quant_bits__": np.int32(bits)}
    for k in src.files:
        if k.startswith("__"):
            out[k] = src[k]
            continue
        arr = src[k]
        # Only quantize 2D linear-layer weights; everything else copied as-is.
        if arr.ndim == 2 and arr.shape[1] % group_size == 0:
            qw, scales, biases = mx.quantize(mx.array(arr), group_size, bits)
            mx.eval(qw, scales, biases)
            out[k] = np.array(qw)
            out[f"{k}.scales"] = np.array(scales)
            out[f"{k}.biases"] = np.array(biases)
        else:
            out[k] = arr
    np.savez(out_path, **out)


def _dequant_to_dense(q_path: Path, dense_out: Path):
    z = np.load(q_path, allow_pickle=False)
    gs = int(z["__quant_group_size__"])
    bits = int(z["__quant_bits__"])
    out = {}
    for k in z.files:
        if k.startswith("__") or k.endswith(".scales") or k.endswith(".biases"):
            continue
        if k + ".scales" in z.files:
            w = mx.dequantize(
                mx.array(z[k]),
                mx.array(z[k + ".scales"]),
                mx.array(z[k + ".biases"]),
                gs, bits,
            )
            mx.eval(w)
            out[k] = np.array(w).astype(np.float32)
        else:
            out[k] = z[k]
    np.savez(dense_out, **out)


def _eval_argmax(head, shard: Path, capture_layer: int, max_records: int = 1000):
    t = pq.read_table(shard)
    prev_list = []
    res_list = []
    inter_list = []
    for i in range(t.num_rows):
        r = {c: t[c][i].as_py() for c in t.column_names}
        ex = e5._extract_row(r, capture_layer)
        if ex is None:
            continue
        prev_list.extend(ex["prev_tokens"].tolist())
        res_list.append(ex["residual"])
        inter_list.append(ex["intermediate"])
        if len(prev_list) >= max_records:
            break
    n = min(len(prev_list), max_records)
    prev = np.asarray(prev_list[:n], dtype=np.int32)
    res = np.concatenate(res_list, axis=0)[:n]
    inter = np.concatenate(inter_list, axis=0)[:n]

    ph = mx.array(prev).reshape(1, n)
    rh = mx.array(res).reshape(1, n, e5.HIDDEN_DIM)
    ih = mx.array(inter).reshape(1, n, e5.HIDDEN_DIM)
    tok, _, _, _ = head(ph, rh, ih)
    mx.eval(tok)
    return np.argmax(np.array(tok).reshape(n, -1), axis=-1)


def main() -> int:
    p = argparse.ArgumentParser(prog="eagle5_quantize")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quantize", help="produce q4 ckpt from bf16 ckpt")
    q.add_argument("--in", dest="in_path", required=True, type=Path)
    q.add_argument("--out", dest="out_path", required=True, type=Path)
    q.add_argument("--group-size", type=int, default=64)
    q.add_argument("--bits", type=int, default=4)

    pr = sub.add_parser("parity", help="argmax match between bf16 and q4")
    pr.add_argument("--bf16", required=True, type=Path)
    pr.add_argument("--q4", required=True, type=Path)
    pr.add_argument("--frozen", required=True, type=Path)
    pr.add_argument("--shard", required=True, type=Path)
    pr.add_argument("--max-records", type=int, default=1000)
    pr.add_argument("--capture-layer", type=int, default=25)
    pr.add_argument("--sparsity-head", choices=["proxy", "off"], default="proxy")

    args = p.parse_args()

    if args.cmd == "quantize":
        _quantize_npz(args.in_path, args.out_path, args.group_size, args.bits)
        sz_in = args.in_path.stat().st_size / 1e6
        sz_out = args.out_path.stat().st_size / 1e6
        print(f"q4: {sz_in:.1f} MB → {sz_out:.1f} MB ({sz_in/sz_out:.2f}× smaller)")
        return 0

    # parity
    with_sparsity = args.sparsity_head != "off"
    head_bf = e5.build_head(args.frozen, with_sparsity)
    e5.load_ckpt(head_bf, args.bf16)
    arg_bf = _eval_argmax(head_bf, args.shard, args.capture_layer, args.max_records)
    del head_bf

    tmp = args.q4.with_suffix(".dequant.tmp.npz")
    _dequant_to_dense(args.q4, tmp)
    head_q = e5.build_head(args.frozen, with_sparsity)
    e5.load_ckpt(head_q, tmp)
    arg_q = _eval_argmax(head_q, args.shard, args.capture_layer, args.max_records)
    tmp.unlink()

    match = float((arg_bf == arg_q).mean())
    print(f"Q4 parity: argmax-match={match*100:.2f}% over {len(arg_bf)} tokens")
    if match < 0.99:
        print(f"WARNING: parity {match*100:.2f}% below the 99% ship gate", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
