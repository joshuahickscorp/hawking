"""Q4 parity check — dequantize a quantized head ckpt and verify its argmax
matches the bf16 source on a held-out shard."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow.parquet as pq

import eagle4


def _dequant_to_dense(q_path: Path, dense_out: Path) -> None:
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
                gs,
                bits,
            )
            mx.eval(w)
            out[k] = np.array(w).astype(np.float32)
        else:
            out[k] = z[k]
    np.savez(dense_out, **out)


def _eval_argmax(head, shard: Path, max_records: int = 1000) -> np.ndarray:
    t = pq.read_table(shard)
    n = min(max_records, t.num_rows)
    H = eagle4.HIDDEN_DIM

    def stack(field, dtype):
        return np.frombuffer(
            b"".join(t[field][i].as_py() for i in range(n)), dtype=dtype
        ).reshape(n, -1)

    prev = mx.array(
        np.array([t["prev_token"][i].as_py() for i in range(n)], dtype=np.int32)
    ).reshape(1, n)
    low = mx.array(stack("hidden_low", np.float16)).astype(mx.float32).reshape(1, n, H)
    mid = mx.array(stack("hidden_mid", np.float16)).astype(mx.float32).reshape(1, n, H)
    hi = mx.array(stack("hidden_high", np.float16)).astype(mx.float32).reshape(1, n, H)
    sh = mx.array(stack("shared_hidden", np.float16)).astype(mx.float32).reshape(1, n, H)
    tok, _, _, _ = head(prev, low, mid, hi, sh)
    mx.eval(tok)
    return np.argmax(np.array(tok).reshape(n, -1), axis=-1)


def main() -> int:
    if len(sys.argv) < 5:
        print(
            "usage: q4_parity.py <bf16_ckpt.npz> <q4_ckpt.npz> <frozen.npz> <heldout.parquet> [max_records]",
            file=sys.stderr,
        )
        return 1
    bf16 = Path(sys.argv[1])
    q4 = Path(sys.argv[2])
    frozen = Path(sys.argv[3])
    shard = Path(sys.argv[4])
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 1000

    head_bf = eagle4.build_head(frozen)
    eagle4.load_ckpt(head_bf, bf16)
    arg_bf = _eval_argmax(head_bf, shard, n)
    del head_bf

    tmp = q4.with_suffix(".dequant.tmp.npz")
    _dequant_to_dense(q4, tmp)
    head_q = eagle4.build_head(frozen)
    eagle4.load_ckpt(head_q, tmp)
    arg_q = _eval_argmax(head_q, shard, n)
    tmp.unlink()

    match = float((arg_bf == arg_q).mean())
    size_bf = bf16.stat().st_size
    size_q = q4.stat().st_size
    print(
        f"Q4 parity: argmax-match={match*100:.2f}% over {n} tokens  "
        f"({size_bf/1e6:.1f}MB → {size_q/1e6:.1f}MB, {size_bf/size_q:.2f}× smaller)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
