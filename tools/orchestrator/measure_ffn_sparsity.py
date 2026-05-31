#!/usr/bin/env python3
"""measure_ffn_sparsity — Track B Step 2 GATE.

Reads the FFN-sparsity capture (packed by pack_ffn.py) and measures the
*oracle* skippable-block fraction per layer: keeping the top blocks by true
activation magnitude, how many blocks can we drop while preserving the FFN
output to a target recall? The oracle is an upper bound — a learned predictor
(Step 3) is strictly worse — so if the oracle can't skip >50% of blocks at
high recall, the whole track is dead and we halt before building kernels.

A block's contribution to `ffn_down @ a` scales with its activation L2 norm
`||a_block||_2` (captured as act_blockl2). Treating per-block contributions as
~orthogonal (reasonable in high-dim), the relative L2 error of dropping a set
S of blocks is sqrt(sum_{b in S} l2_b^2) / sqrt(sum_b l2_b^2). For a target
recall r we drop the smallest-l2 blocks while that relative error stays
<= (1 - r). We also report an L1 variant (triangle-inequality bound, more
conservative) and the raw max-based fraction (the trainer's label signal) for
comparison.

Usage:
    python3.12 tools/orchestrator/measure_ffn_sparsity.py \
        --ffn-dir _capture/q3b_ffn_shards
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

SENTINEL = 0xFFFFFFFF


def deq(row, stem):
    q = np.frombuffer(row[f"{stem}_q"], dtype=np.int8).astype(np.float32)
    return q * float(row[f"{stem}_scale"])


def load_layers_parquet(ffn_dir):
    """{layer: (blockmax[N,nb], blockl2[N,nb])} from packed int8 parquet."""
    import pyarrow.parquet as pq

    shards = sorted(Path(ffn_dir).glob("shard_*.parquet"))
    if not shards:
        sys.stderr.write(f"no shards in {ffn_dir}\n")
        sys.exit(1)
    acc: dict[int, tuple[list, list]] = {}
    for sh in shards:
        t = pq.read_table(sh)
        cols = t.column_names
        for i in range(t.num_rows):
            r = {c: t[c][i].as_py() for c in cols}
            layer = int(r["layer"])
            n_tok, _hidden, n_blocks = (int(x) for x in r["shape"])
            mx = deq(r, "act_blockmax").reshape(n_tok, n_blocks)
            l2 = deq(r, "act_blockl2").reshape(n_tok, n_blocks)
            slot = acc.setdefault(layer, ([], []))
            slot[0].append(mx)
            slot[1].append(l2)
    return {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in acc.items()}


def load_layers_bin(path):
    """{layer: (blockmax[N,nb], blockl2[N,nb])} from the raw f32 capture stream
    (no int8 quantization — the trustworthy precision for the gate)."""
    data = Path(path).read_bytes()
    n = len(data)
    off = 0
    hidden = n_blocks = None
    acc: dict[int, tuple[list, list]] = {}
    while off + 8 <= n:
        a, b = struct.unpack_from("<II", data, off)
        if a == SENTINEL and b == SENTINEL:
            if off + 16 > n:
                break
            _, _, hidden, n_blocks = struct.unpack_from("<IIII", data, off)
            hidden, n_blocks = int(hidden), int(n_blocks)
            off += 16
            continue
        if hidden is None:
            raise ValueError(f"stream did not start with a sentinel at off={off}")
        rec = 4 + hidden * 4 + 2 * n_blocks * 4
        if off + rec > n:
            break
        (layer,) = struct.unpack_from("<I", data, off)
        foff = off + 4 + hidden * 4
        mx = np.frombuffer(data, np.float32, n_blocks, foff).copy()
        l2 = np.frombuffer(data, np.float32, n_blocks, foff + n_blocks * 4).copy()
        slot = acc.setdefault(int(layer), ([], []))
        slot[0].append(mx)
        slot[1].append(l2)
        off += rec
    return {k: (np.stack(v[0]), np.stack(v[1])) for k, v in acc.items()}


def kept_fraction_l2(vals: np.ndarray, eps: float) -> np.ndarray:
    """Per-row: fraction of blocks that must be KEPT so the relative L2 error
    of dropping the rest is <= eps. vals: [N, nb] non-negative (l2 per block)."""
    energy = vals ** 2
    total = energy.sum(axis=1, keepdims=True)
    total = np.maximum(total, 1e-30)
    order = np.argsort(energy, axis=1)  # ascending: smallest first
    sorted_e = np.take_along_axis(energy, order, axis=1)
    cum_dropped = np.cumsum(sorted_e, axis=1)  # energy dropped if we skip the first k
    budget = (eps ** 2) * total  # may drop up to this much energy
    # number of smallest blocks we can drop = count where cum_dropped <= budget
    n_drop = (cum_dropped <= budget).sum(axis=1)
    nb = vals.shape[1]
    return (nb - n_drop) / nb


def kept_fraction_l1(vals: np.ndarray, eps: float) -> np.ndarray:
    """Conservative triangle-inequality bound: drop smallest-magnitude blocks
    while cumulative dropped magnitude <= eps * total magnitude."""
    total = vals.sum(axis=1, keepdims=True)
    total = np.maximum(total, 1e-30)
    order = np.argsort(vals, axis=1)
    sorted_v = np.take_along_axis(vals, order, axis=1)
    cum_dropped = np.cumsum(sorted_v, axis=1)
    budget = eps * total
    n_drop = (cum_dropped <= budget).sum(axis=1)
    nb = vals.shape[1]
    return (nb - n_drop) / nb


def main() -> int:
    ap = argparse.ArgumentParser(prog="measure_ffn_sparsity")
    ap.add_argument("--ffn-dir", default=None, help="packed int8 parquet dir")
    ap.add_argument("--bin", default=None, help="raw f32 capture .bin (preferred for the gate)")
    ap.add_argument("--out", default=None, help="optional JSON report path")
    ap.add_argument("--max-thresh", type=float, default=0.05,
                    help="raw blockmax>thresh active fraction (trainer label) to report")
    args = ap.parse_args()

    if args.bin:
        data = load_layers_bin(args.bin)
        src = f"{args.bin} (raw f32)"
    elif args.ffn_dir:
        data = load_layers_parquet(args.ffn_dir)
        src = f"{args.ffn_dir} (int8 parquet)"
    else:
        sys.stderr.write("provide --bin or --ffn-dir\n")
        return 1
    layers = sorted(data.keys())
    print(f"source: {src}")

    print(f"{'lyr':>3} {'N':>6} {'keepL2@99':>9} {'keepL2@99.9':>11} "
          f"{'keepL1@99':>9} {'keepL1@99.9':>11} {'max>thr':>8}")
    rows = {}
    agg = {k: [] for k in ("l2_99", "l2_999", "l1_99", "l1_999", "maxthr")}
    n_tok_total = 0
    for layer in layers:
        bmax, bl2 = data[layer]
        if bl2 is None:
            continue
        N = bl2.shape[0]
        n_tok_total = max(n_tok_total, N)
        k_l2_99 = float(kept_fraction_l2(bl2, 0.01).mean())
        k_l2_999 = float(kept_fraction_l2(bl2, 0.001).mean())
        k_l1_99 = float(kept_fraction_l1(bl2, 0.01).mean())
        k_l1_999 = float(kept_fraction_l1(bl2, 0.001).mean())
        maxthr = float((bmax > args.max_thresh).mean())
        rows[layer] = dict(N=N, keep_l2_99=k_l2_99, keep_l2_999=k_l2_999,
                           keep_l1_99=k_l1_99, keep_l1_999=k_l1_999,
                           max_active=maxthr)
        agg["l2_99"].append(k_l2_99)
        agg["l2_999"].append(k_l2_999)
        agg["l1_99"].append(k_l1_99)
        agg["l1_999"].append(k_l1_999)
        agg["maxthr"].append(maxthr)
        print(f"{layer:>3} {N:>6} {k_l2_99:>9.3f} {k_l2_999:>11.3f} "
              f"{k_l1_99:>9.3f} {k_l1_999:>11.3f} {maxthr:>8.3f}")

    def m(key):
        return float(np.mean(agg[key])) if agg[key] else float("nan")

    keep99 = m("l2_99")
    keep999 = m("l2_999")
    print("\n=== OVERALL (mean over layers) ===")
    print(f"  oracle KEEP fraction @99% recall (L2):   {keep99:.3f}  "
          f"-> skippable {1-keep99:.3f}")
    print(f"  oracle KEEP fraction @99.9% recall (L2): {keep999:.3f}  "
          f"-> skippable {1-keep999:.3f}")
    print(f"  conservative KEEP @99% (L1):   {m('l1_99'):.3f} -> skippable {1-m('l1_99'):.3f}")
    print(f"  conservative KEEP @99.9% (L1): {m('l1_999'):.3f} -> skippable {1-m('l1_999'):.3f}")
    print(f"  raw blockmax>{args.max_thresh} active fraction: {m('maxthr'):.3f}")
    print(f"\n  FFN is ~72% of bytes/token. At 99% recall, FFN-byte cut ~= "
          f"{1-keep99:.3f} -> bytes/token cut ~= {0.72*(1-keep99):.3f}")
    # GATE verdict
    skip99 = 1 - keep99
    print("\n=== GATE (handoff: >50% skippable at high recall = GO) ===")
    if skip99 >= 0.50:
        print(f"  GO: {skip99:.1%} of blocks skippable @99% recall (oracle). "
              f"Worth training a predictor (Step 3) to approach this.")
    elif skip99 >= 0.30:
        print(f"  MARGINAL: only {skip99:.1%} skippable @99% recall (oracle). "
              f"Below the 50% bar; ceiling lift is modest and a real predictor "
              f"will be worse. Reset expectations before building kernels.")
    else:
        print(f"  NO-GO: only {skip99:.1%} skippable @99% recall even with an "
              f"oracle selector. q3b FFN is not block-sparse enough; halt track.")

    if args.out:
        json.dump({"layers": rows, "overall": {
            "keep_l2_99": keep99, "keep_l2_999": keep999,
            "keep_l1_99": m("l1_99"), "keep_l1_999": m("l1_999"),
            "max_active": m("maxthr"), "skippable_99": skip99,
        }}, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
