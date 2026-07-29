#!/usr/bin/env python3
"""Extract the Qwen-3B-Instruct "frozen baseline" .npz for eagle5_train.py.

Analog of `eagle4/eagle4.py frozen` (which produces `eagle4/v2lite_frozen.npz`
for DeepSeek-V2-Lite), but for Qwen2.5-3B-Instruct.

The output schema is what `tools/training/eagle5_train.py :: build_head()`
loads via `np.load(frozen_npz)` and accesses by these exact keys (see
eagle5_train.py:138-150):

    token_embd   : float16, shape (hidden_dim, vocab_size) = (2048, 151936)
    lm_head      : float16, shape (hidden_dim, vocab_size) = (2048, 151936)
    output_norm  : float32, shape (hidden_dim,)            = (2048,)

Note the (hidden, vocab) layout — eagle5_train builds `embed_table` via
`mx.transpose(self._token_embd, (1, 0))` (eagle5_train.py:119) and uses
`draft_hidden @ self._lm_head` (eagle5_train.py:126). This matches
eagle4's frozen-export convention (eagle4.py:434-435).

Qwen2.5-3B-Instruct specifics
-----------------------------
- hidden_dim:        2048
- vocab_size:        151936
- n_layers:          36   (GGUF metadata key `qwen2.block_count`)
- intermediate_size: 11008
- rms_norm_eps:      1e-6
- Embedding is TIED to the LM head: the GGUF only contains
  `token_embd.weight` (Q6_K in Q4_K_M quant) and there is NO
  separate `output.weight`. We dequantize once and write it under
  both keys so eagle5_train doesn't need to know about tying.

Dependencies
------------
    pip install gguf numpy

The `gguf` Python package ships with a built-in `dequantize` that handles
Q6_K (used for `token_embd.weight` in Qwen-3B-Q4_K_M). Output is fp32 from
gguf; we downcast to fp16 to match eagle4's convention and halve the
on-disk size (~620 MB → ~310 MB for the embed/lm_head tables alone, then
2× for the duplicated lm_head copy).

Usage
-----
    python3 tools/training/build_qwen3b_frozen.py
    # or with custom paths:
    python3 tools/training/build_qwen3b_frozen.py \\
        --gguf models/qwen2.5-3b-instruct-q4_k_m.gguf \\
        --out  eagle4/qwen3b_frozen.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    print(
        "ERROR: gguf python package not installed. Install via `pip install gguf`.\n"
        "(eagle4/.venv already has it; activate that venv or pip-install into "
        "the venv you're running eagle5_train.py from.)",
        file=sys.stderr,
    )
    sys.exit(1)


# Qwen2.5-3B-Instruct constants. Kept here for sanity-check assertions
# against the GGUF metadata so we fail loudly on the wrong file rather
# than silently exporting a mismatched frozen.npz.
QWEN3B_HIDDEN_DIM = 2048
QWEN3B_VOCAB_SIZE = 151936
QWEN3B_N_LAYERS = 36
QWEN3B_RMS_EPS = 1e-6
QWEN3B_INTERMEDIATE = 11008


def _gguf_meta(reader: gguf.GGUFReader, key: str):
    """Return the first scalar value for a GGUF metadata key, or None."""
    field = reader.fields.get(key)
    if field is None:
        return None
    if not field.data:
        return None
    return field.parts[field.data[0]]


def _find_tensor(reader: gguf.GGUFReader, name: str):
    for t in reader.tensors:
        if t.name == name:
            return t
    return None


def _dequantize_tensor(t) -> np.ndarray:
    """Dequantize a GGUF tensor to fp32 using gguf.quants.dequantize().

    Works for Q6_K, Q4_K, F32, F16, etc. Returns numpy fp32 in the
    tensor's logical (un-byte-packed) shape.
    """
    raw = t.data  # np.ndarray of uint8 (or float32 for F32 tensors)
    qtype = t.tensor_type
    logical_shape = tuple(int(x) for x in t.shape)
    if qtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        # Already non-quantized; just view as the right dtype + reshape.
        if qtype == gguf.GGMLQuantizationType.F32:
            arr = raw.view(np.float32)
        else:
            arr = raw.view(np.float16).astype(np.float32)
        return arr.reshape(logical_shape)
    # Quantized path — gguf.quants.dequantize handles Q4_K, Q6_K, Q5_K, Q8_0, ...
    arr = gguf.quants.dequantize(raw, qtype)
    return arr.reshape(logical_shape).astype(np.float32, copy=False)


def extract_frozen(gguf_path: Path, out_path: Path) -> None:
    if not gguf_path.exists():
        raise SystemExit(f"GGUF not found: {gguf_path}")

    print(f"[frozen] opening {gguf_path}", flush=True)
    reader = gguf.GGUFReader(str(gguf_path), "r")

    # --- Sanity-check metadata against hardcoded Qwen-3B specs ---------
    hidden = _gguf_meta(reader, "qwen2.embedding_length")
    n_layers = _gguf_meta(reader, "qwen2.block_count")
    if hidden is not None and int(hidden) != QWEN3B_HIDDEN_DIM:
        raise SystemExit(
            f"hidden_dim mismatch: GGUF says {int(hidden)}, expected "
            f"{QWEN3B_HIDDEN_DIM} for Qwen-3B-Instruct. Wrong GGUF?"
        )
    if n_layers is not None and int(n_layers) != QWEN3B_N_LAYERS:
        print(
            f"[frozen] WARNING: block_count={int(n_layers)} != "
            f"{QWEN3B_N_LAYERS}; proceeding but you may have the wrong "
            f"GGUF for the Qwen-3B-Instruct preset.",
            flush=True,
        )

    # --- Locate the three tensors we need ------------------------------
    t_embed = _find_tensor(reader, "token_embd.weight")
    t_norm = _find_tensor(reader, "output_norm.weight")
    # Qwen-3B ties lm_head to embed: there is NO output.weight in the GGUF.
    # We will reuse the dequantized embed for both keys.
    t_lm_head = _find_tensor(reader, "output.weight")  # may be None
    if t_embed is None:
        raise SystemExit(
            "token_embd.weight not found in GGUF. Tensor list head: "
            f"{[x.name for x in reader.tensors[:8]]}"
        )
    if t_norm is None:
        raise SystemExit("output_norm.weight not found in GGUF.")

    tied = t_lm_head is None
    if tied:
        print(
            "[frozen] no output.weight in GGUF → lm_head is tied to "
            "token_embd (expected for Qwen2.5-3B-Instruct).",
            flush=True,
        )
    else:
        print(
            "[frozen] found explicit output.weight; lm_head is UNTIED. "
            "Exporting separate token_embd and lm_head from GGUF.",
            flush=True,
        )

    # --- Dequantize ---------------------------------------------------
    t0 = time.time()
    print(
        f"[frozen] dequantizing token_embd.weight "
        f"({t_embed.tensor_type.name} {list(t_embed.shape)})...",
        flush=True,
    )
    embed = _dequantize_tensor(t_embed)  # fp32, shape (hidden, vocab)
    print(f"[frozen]   → {embed.dtype} {embed.shape}  ({time.time()-t0:.1f}s)", flush=True)

    if tied:
        lm = embed  # same buffer; the np.savez call will write it under both keys
    else:
        t1 = time.time()
        print(
            f"[frozen] dequantizing output.weight "
            f"({t_lm_head.tensor_type.name} {list(t_lm_head.shape)})...",
            flush=True,
        )
        lm = _dequantize_tensor(t_lm_head)
        print(f"[frozen]   → {lm.dtype} {lm.shape}  ({time.time()-t1:.1f}s)", flush=True)

    print(f"[frozen] reading output_norm.weight ({t_norm.tensor_type.name})", flush=True)
    norm = _dequantize_tensor(t_norm)  # fp32 already

    # --- Shape sanity-checks against eagle5_train's expectations ------
    # eagle5_train uses `mx.transpose(token_embd, (1, 0))` then indexes
    # `embed_table[prev_tok]`, so token_embd must be (hidden, vocab).
    # That matches the GGUF tensor's logical shape directly — no
    # transpose needed (unlike from a HF/safetensors checkpoint).
    if embed.shape != (QWEN3B_HIDDEN_DIM, QWEN3B_VOCAB_SIZE):
        raise SystemExit(
            f"unexpected token_embd shape {embed.shape}; expected "
            f"({QWEN3B_HIDDEN_DIM}, {QWEN3B_VOCAB_SIZE}). Refusing to "
            f"write a frozen .npz that eagle5_train.py won't load."
        )
    if lm.shape != (QWEN3B_HIDDEN_DIM, QWEN3B_VOCAB_SIZE):
        raise SystemExit(f"unexpected lm_head shape {lm.shape}")
    if norm.shape != (QWEN3B_HIDDEN_DIM,):
        raise SystemExit(f"unexpected output_norm shape {norm.shape}")

    # --- Downcast to eagle4's storage convention ----------------------
    # eagle4/v2lite_frozen.npz uses fp16 for the two big tables and
    # fp32 for the norm vector. Match that so eagle5_train's expectations
    # don't shift and the on-disk footprint stays comparable.
    embed_fp16 = embed.astype(np.float16, copy=False)
    lm_fp16 = lm.astype(np.float16, copy=False) if not tied else embed_fp16
    norm_fp32 = norm.astype(np.float32, copy=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[frozen] writing → {out_path}", flush=True)
    np.savez(
        out_path,
        token_embd=embed_fp16,
        lm_head=lm_fp16,
        output_norm=norm_fp32,
    )

    # --- Summary ------------------------------------------------------
    total_mb = (
        embed_fp16.nbytes
        + (0 if tied else lm_fp16.nbytes)
        + norm_fp32.nbytes
    ) / 1e6
    on_disk_mb = out_path.stat().st_size / 1e6
    print("[frozen] DONE", flush=True)
    print(f"  token_embd  : {embed_fp16.dtype} {embed_fp16.shape}  ({embed_fp16.nbytes/1e6:.1f} MB)")
    print(f"  lm_head     : {lm_fp16.dtype} {lm_fp16.shape}  ({lm_fp16.nbytes/1e6:.1f} MB)"
          f"{'  [tied to token_embd; np.savez writes a copy]' if tied else ''}")
    print(f"  output_norm : {norm_fp32.dtype} {norm_fp32.shape}  ({norm_fp32.nbytes/1e6:.3f} MB)")
    print(f"  logical mem : {total_mb:.1f} MB (in-RAM peak during training)")
    print(f"  on-disk     : {on_disk_mb:.1f} MB (np.savez uncompressed)")
    print(
        f"  schema      : matches eagle4/v2lite_frozen.npz "
        f"(keys=['token_embd','lm_head','output_norm']); "
        f"eagle5_train.py:build_head will load it as-is."
    )


def main() -> int:
    p = argparse.ArgumentParser(prog="build_qwen3b_frozen")
    p.add_argument(
        "--gguf",
        type=Path,
        default=Path("models/qwen2.5-3b-instruct-q4_k_m.gguf"),
        help="path to Qwen-3B-Instruct GGUF (Q4_K_M is the supported preset)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("eagle4/qwen3b_frozen.npz"),
        help="output .npz path (eagle5_train --frozen consumes this)",
    )
    args = p.parse_args()
    extract_frozen(args.gguf, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
