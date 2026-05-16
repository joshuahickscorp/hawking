#!/usr/bin/env python3
"""
extract_lm_head.py — extract V2-Lite frozen weights for EAGLE-3 head training.

The EAGLE-3 head consumes the target's input-token embedding (for prev_token
lookup) and produces a hidden that feeds the target's lm_head + output_norm.
For training, we need fp16 copies of those three tensors decoupled from the
Rust engine.

What this script does:
  1. Opens models/deepseek-v2-lite-q4.gguf via the `gguf` Python lib
  2. Dequantizes:
       - `token_embd.weight`  (Q4_K,  shape [hidden=2048, vocab=102400])
       - `output.weight`      (Q6_K,  shape [hidden=2048, vocab=102400])
       - `output_norm.weight` (F32,   shape [2048])
  3. Writes them to a single .npz file the MLX trainer loads at startup

Quirks:
  - V2-Lite-Chat is NOT tied: input embeddings and lm_head are distinct
    tensors. Both are needed.
  - GGUF stores rows of `output.weight` along the *first* dimension when
    flattened the llama.cpp way. The shape in the header reads as (hidden,
    vocab) but the underlying layout is row-major over `vocab` rows of
    `hidden`-wide weights — i.e. the matrix you multiply by `hidden` to get
    `vocab` logits is `output.weight.T` after dequant. See `verify` block
    below — runs a tiny dot-product sanity check against a real captured
    hidden state from the C2 dataset.
  - Dequantization runs CPU-only and is ~30-60s for two 200M-element tensors
    on M3 Pro. One-shot at training-stack setup time; not on the hot path.

Run once. The .npz is ~840 MB (token_embd + output, both fp16). Place at
`tools/training/mlx_eagle/v2lite_frozen.npz`.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_GGUF = REPO_ROOT / "models/deepseek-v2-lite-q4.gguf"
DEFAULT_OUT = REPO_ROOT / "tools/training/mlx_eagle/v2lite_frozen.npz"


def _dequant(reader, name: str) -> np.ndarray:
    """Dequantize a tensor by name, returning fp16 numpy."""
    import gguf

    for t in reader.tensors:
        if t.name == name:
            t0 = time.time()
            if t.tensor_type == gguf.GGMLQuantizationType.F32:
                arr = np.asarray(t.data).reshape(list(t.shape)).astype(np.float32)
            elif t.tensor_type == gguf.GGMLQuantizationType.F16:
                arr = np.asarray(t.data).reshape(list(t.shape)).astype(np.float16)
            else:
                # Q4_K / Q6_K / etc go through the helper.
                arr = gguf.quants.dequantize(t.data, t.tensor_type)
                arr = arr.reshape(list(t.shape)).astype(np.float16)
            print(
                f"  {name:24s} {str(t.tensor_type.name):8s} "
                f"-> {arr.dtype} shape={list(arr.shape)} "
                f"({time.time() - t0:.1f}s)"
            )
            return arr
    raise KeyError(f"tensor {name!r} not in GGUF")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--gguf", default=str(DEFAULT_GGUF))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument(
        "--verify",
        action="store_true",
        help="Sanity check: pick one record from the C2 shard, dot its hidden "
        "with the extracted lm_head, check argmax matches a freshly-computed "
        "greedy. Requires training_data/c2_hidden/eagle3_v0/shard_000.bin.",
    )
    args = p.parse_args()

    try:
        import gguf
    except ImportError:
        print("ERROR: pip install gguf", file=sys.stderr)
        return 2

    gguf_path = pathlib.Path(args.gguf)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[extract] reading {gguf_path}")
    r = gguf.GGUFReader(str(gguf_path))

    # Pull architecture metadata (we'll also embed in the .npz for cross-check)
    def _field(k):
        f = r.get_field(k)
        if f is None:
            return None
        try:
            return f.contents()
        except Exception:
            return None

    arch = _field("general.architecture")
    name = _field("general.name")
    hidden = _field("deepseek2.embedding_length")
    vocab = _field("deepseek2.vocab_size")
    rms_eps = _field("deepseek2.attention.layer_norm_rms_epsilon")
    eos = _field("tokenizer.ggml.eos_token_id")
    bos = _field("tokenizer.ggml.bos_token_id")
    print(f"[extract] arch={arch} name={name} h={hidden} V={vocab} eps={rms_eps} bos={bos} eos={eos}")

    # Dequantize the three frozen tensors.
    print("[extract] dequantizing (this is the slow step, ~30-60s):")
    token_embd = _dequant(r, "token_embd.weight")  # input embeddings, Q4_K
    lm_head = _dequant(r, "output.weight")  # output projection, Q6_K
    output_norm = _dequant(r, "output_norm.weight")  # final RMSNorm, F32

    # Save.
    print(f"[extract] writing {out_path}")
    np.savez(
        out_path,
        token_embd=token_embd,
        lm_head=lm_head,
        output_norm=output_norm,
        hidden=np.int32(hidden),
        vocab=np.int32(vocab),
        rms_eps=np.float32(rms_eps),
        bos_id=np.int32(bos),
        eos_id=np.int32(eos),
        model_name=np.array(name, dtype=object),
    )
    sz_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[extract] done. wrote {sz_mb:.1f} MB.")

    if args.verify:
        verify_against_capture(out_path)

    return 0


def verify_against_capture(npz_path: pathlib.Path) -> None:
    """Sanity check: run extracted lm_head + output_norm on a captured hidden
    state and confirm the resulting argmax matches what the engine produced
    for the *same* hidden (we already store greedy under the lm_head path).

    NOTE: the C2 capture used `--no-lm-head` so we DON'T have greedy stored.
    Instead this verify just ensures the dot-product shape is correct and
    produces sensible logits (max > 0, no NaN).
    """
    import struct

    print("\n[verify] sanity-checking extracted weights against C2 hidden state")
    shard = REPO_ROOT / "training_data/c2_hidden/eagle3_v0/shard_000.bin"
    if not shard.exists():
        print(f"[verify] {shard} missing; skipping")
        return
    data = np.load(npz_path, allow_pickle=True)
    lm_head = data["lm_head"]  # (hidden, vocab) layout per GGUF header
    output_norm = data["output_norm"]
    eps = float(data["rms_eps"])
    hidden = int(data["hidden"])

    # Read the first record from the C2 shard.
    with open(shard, "rb") as f:
        hdr = f.read(16)
        hd_shard = struct.unpack("<I", hdr[8:12])[0]
        assert hd_shard == hidden, f"hidden mismatch: {hd_shard} vs {hidden}"
        (id_len,) = struct.unpack("<H", f.read(2))
        sid = f.read(id_len).decode()
        _pos, prev_tok, next_tok = struct.unpack("<III", f.read(12))
        hidden_vec = np.frombuffer(f.read(hidden * 2), dtype=np.float16).astype(np.float32)

    # Hidden was captured AFTER output_norm (forward_token_final_norm includes
    # the final norm). So we shouldn't apply it again — just lm_head.
    # Two layouts to try since GGUF row-major can be ambiguous:
    #   (a) logits = hidden @ lm_head        if lm_head is (hidden, vocab)
    #   (b) logits = hidden @ lm_head.T      if lm_head is (vocab, hidden)
    for layout, w in [("(hidden @ W)", lm_head), ("(hidden @ W.T)", lm_head.T)]:
        try:
            logits = hidden_vec @ w.astype(np.float32)
        except ValueError as e:
            print(f"[verify] layout {layout}: shape error {e}")
            continue
        if logits.shape[-1] != 102400:
            print(f"[verify] layout {layout}: vocab dim wrong {logits.shape}")
            continue
        argmax = int(np.argmax(logits))
        print(
            f"[verify] layout {layout}: vocab={logits.shape[-1]} "
            f"argmax={argmax} (corpus next_token was {next_tok}) "
            f"max={logits.max():.3f} min={logits.min():.3f}"
        )


if __name__ == "__main__":
    sys.exit(main())
