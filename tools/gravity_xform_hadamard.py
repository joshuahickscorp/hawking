#!/usr/bin/env python3
"""G032 / G-XFORM: does a structured orthogonal transform buy bits at equal function?

W x = (W H) (H^T x) for any orthogonal H. So a transform that leaves the function
EXACTLY unchanged can still change how many bits the weights need, because it
changes the distribution the codec has to fit. Sylvester-Hadamard is the cheap
member of that family: entries are +-1, it needs zero stored bytes because it is
generated, and it applies in O(n log n) per activation vector.

What is measured here, at FIXED bit width and fixed group size, on REAL parent
tensors:

  hold      cosine(W, dequant(quant(W)))          the campaign's own metric
  rel_fro   ||W - What||_F / ||W||_F
  entropy   order-0 entropy of the code symbols, bits/elem

Entropy matters as much as hold: FINDING F1's rANS route codes the symbol stream,
and the compiler law in the A1 directive says a locally better scale can code
worse. A transform that improves hold while raising entropy may be a net loss
once rANS follows, so both are reported and neither is allowed to stand alone.

Reads the bf16 parent directly. No dependency on safetensors -- the header is
JSON and the payload is raw.

  ./tools/gravity_xform_hadamard.py --layers 0,31,63 --bits 3 \
      --out receipts/ascent-2026-08-16/G032_XFORM_HADAMARD.json
"""
from __future__ import annotations
import argparse, json, math, pathlib, struct, subprocess, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BF16 = ROOT / "workspace/campaign/records/runs/qwen38-27b/bf16"


def load_tensor(name):
    """One tensor out of the sharded bf16 parent, as float32."""
    index = json.loads((BF16 / "model.safetensors.index.json").read_text())
    shard = BF16 / index["weight_map"][name]
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header[name]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    assert meta["dtype"] == "BF16", meta["dtype"]
    u16 = np.frombuffer(raw, dtype=np.uint16)
    # bf16 -> f32 is a 16-bit left shift into the mantissa.
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(meta["shape"])


def hadamard(n):
    """Sylvester-Hadamard, normalized so H is orthogonal (H @ H.T == I)."""
    assert n & (n - 1) == 0, f"{n} is not a power of two"
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


def block_hadamard_apply(w, block):
    """Right-multiply by a block-diagonal Hadamard along the contraction axis.

    5120 and 17408 are both 1024 * odd, so a 1024 block tiles them exactly and
    no padding is needed. Applied as a reshape plus one matmul per block, which
    is also how it would lower: an in-place FWHT over each block of the input
    vector, O(n log n), zero stored bytes.
    """
    rows, cols = w.shape
    assert cols % block == 0, (cols, block)
    h = hadamard(block)
    return (w.reshape(rows, cols // block, block) @ h).reshape(rows, cols)


def quantize_group(w, bits, group):
    """Symmetric absmax group quantization. Returns (dequantized, codes)."""
    rows, cols = w.shape
    assert cols % group == 0
    g = w.reshape(rows, cols // group, group)
    qmax = (1 << (bits - 1)) - 1          # q3 -> 3, q4 -> 7
    absmax = np.abs(g).max(axis=2, keepdims=True)
    scale = np.where(absmax > 0, absmax / qmax, 1.0).astype(np.float32)
    # Round-to-nearest, clipped into the signed code range.
    codes = np.clip(np.rint(g / scale), -qmax - 1, qmax).astype(np.int32)
    return (codes * scale).reshape(rows, cols), codes


def cosine(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def rel_fro(a, b):
    return float(np.linalg.norm((a - b).ravel()) / np.linalg.norm(a.ravel()))


def code_entropy(codes):
    """Order-0 entropy in bits/symbol -- the floor any entropy coder can reach."""
    vals, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def score(w, bits, group):
    deq, codes = quantize_group(w, bits, group)
    return {
        "hold_cosine": cosine(w, deq),
        "rel_fro": rel_fro(w, deq),
        "code_entropy_bits": code_entropy(codes),
        "distinct_symbols": int(np.unique(codes).size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,31,63")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    tensors = []
    for l in [int(x) for x in a.layers.split(",")]:
        for organ in ("gate_proj", "up_proj", "down_proj"):
            tensors.append(f"language_model.model.layers.{l}.mlp.{organ}.weight")

    rows = []
    for name in tensors:
        w = load_tensor(name).astype(np.float32)
        base = score(w, a.bits, a.group)

        wt = block_hadamard_apply(w, a.block)
        deq_t, codes_t = quantize_group(wt, a.bits, a.group)
        # Undo the transform so the comparison is against the SAME function.
        # H is orthogonal and block-diagonal, so the inverse is its transpose,
        # which for a symmetric Sylvester block is itself.
        back = block_hadamard_apply(deq_t, a.block)
        xf = {
            "hold_cosine": cosine(w, back),
            "rel_fro": rel_fro(w, back),
            "code_entropy_bits": code_entropy(codes_t),
            "distinct_symbols": int(np.unique(codes_t).size),
        }
        rows.append({
            "tensor": name, "shape": list(w.shape),
            "baseline": base, "hadamard": xf,
            "delta_hold": xf["hold_cosine"] - base["hold_cosine"],
            "delta_rel_fro_pct": (xf["rel_fro"] - base["rel_fro"]) / base["rel_fro"] * 100.0,
            "delta_entropy_bits": xf["code_entropy_bits"] - base["code_entropy_bits"],
        })
        print(f"{name.split('layers.')[1]:<28} "
              f"hold {base['hold_cosine']:.6f} -> {xf['hold_cosine']:.6f} "
              f"({xf['hold_cosine']-base['hold_cosine']:+.6f})  "
              f"relfro {base['rel_fro']:.5f} -> {xf['rel_fro']:.5f}  "
              f"H {base['code_entropy_bits']:.4f} -> {xf['code_entropy_bits']:.4f} "
              f"({xf['code_entropy_bits']-base['code_entropy_bits']:+.4f} b/elem)")
        del w, wt, deq_t, codes_t, back

    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    verdict_hold = mean("delta_hold")
    verdict_entropy = mean("delta_entropy_bits")
    doc = {
        "schema": "hawking.nos.gxform_hadamard.v1",
        "obligation": "G032 -- structured transform that lowers achievable bits at equal function",
        "transform": {
            "family": "block-diagonal Sylvester-Hadamard, right-multiplied on the contraction axis",
            "block": a.block,
            "exactly_function_preserving": "H is orthogonal, so W x == (W H)(H^T x) in exact "
                                           "arithmetic. Both 5120 and 17408 are 1024*odd, so the "
                                           "block tiles with no padding.",
            "stored_bytes": 0,
            "stored_bytes_reason": "Sylvester-Hadamard is generated from its size, not stored.",
            "runtime_cost": "one in-place FWHT per activation vector per bound tensor, "
                            "O(n log n) adds, no multiplies. NOT YET MEASURED ON DEVICE.",
            "fold_at_pack_time": "W H is packed instead of W. The H^T x half is the part that "
                                 "costs runtime and is what a device measurement must price.",
        },
        "fixed": {"bits": a.bits, "group": a.group,
                  "quantizer": "symmetric absmax group, round-to-nearest"},
        "tensors": rows,
        "summary": {
            "mean_delta_hold": verdict_hold,
            "mean_delta_rel_fro_pct": mean("delta_rel_fro_pct"),
            "mean_delta_entropy_bits": verdict_entropy,
        },
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    print(f"\nmean hold delta      {verdict_hold:+.6f}")
    print(f"mean rel_fro delta   {mean('delta_rel_fro_pct'):+.2f}%")
    print(f"mean entropy delta   {verdict_entropy:+.4f} bits/elem")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
