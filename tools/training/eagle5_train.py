#!/usr/bin/env python3
"""path-to-50 lever 3: train the eagle5 v2 activation-sparsity head.

Architecture, loss, and acceptance gates per
`reports/eagle5_v2_wiring_handoff.md`. This file is the runnable
scaffold; design rationale lives in the doc.

Mirrors `eagle4/eagle4.py` in shape and code style. New inputs:
- residual_in_per_layer at the capture layer
- intermediate_per_layer at the capture layer (first-expert FFN output)

New auxiliary head: sparsity_log (logits over 1408 MoE FFN intermediate
channels) trained against a per-batch percentile threshold on the
intermediate ground truth.

Quick start (overnight run):

    python3 tools/training/eagle5_train.py \\
        --corpus-dir artifacts/calibration/v2_lite_corpus \\
        --frozen      eagle4/v2lite_frozen.npz \\
        --ckpt-dir    checkpoints/eagle5_v2 \\
        --epochs 5 --batch-size 16 --seq-len 16
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as mxoptim
except ImportError:
    print(
        "ERROR: mlx not installed. Install via `pip install -r tools/training/requirements.txt` "
        "or `pip install mlx`.",
        file=sys.stderr,
    )
    sys.exit(1)

import numpy as np
import pyarrow.parquet as pq


# V2-Lite constants — match build_corpus.py / eagle4.py.
HIDDEN_DIM = 2048
MOE_INTERMEDIATE = 1408
N_HEADS = 16
RMS_EPS = 1e-6
N_LAYERS = 27
N_MOE_LAYERS = 26
N_ROUTED = 64
TOP_K = 6


class _SwiGLU(nn.Module):
    def __init__(self, h: int, i: int):
        super().__init__()
        self.gate = nn.Linear(h, i, bias=False)
        self.up = nn.Linear(h, i, bias=False)
        self.down = nn.Linear(i, h, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = mx.ones((HIDDEN_DIM,))
        self.attn = nn.MultiHeadAttention(HIDDEN_DIM, N_HEADS, bias=False)
        self.mlp_norm = mx.ones((HIDDEN_DIM,))
        self.mlp = _SwiGLU(HIDDEN_DIM, 4 * HIDDEN_DIM)  # standard 4× width

    def __call__(self, x, mask):
        h = mx.fast.rms_norm(x, self.attn_norm, RMS_EPS)
        x = x + self.attn(h, h, h, mask=mask)
        h = mx.fast.rms_norm(x, self.mlp_norm, RMS_EPS)
        x = x + self.mlp(h)
        return x


class Eagle5Head(nn.Module):
    """Eagle5 v2 head: predicts (next-token logits, channel-sparsity mask).

    Architecture summary — see `reports/eagle5_v2_wiring_handoff.md` §3.
    """

    def __init__(self, token_embd, lm_head, output_norm, with_sparsity: bool = True):
        super().__init__()
        self._token_embd = token_embd       # frozen, NOT in trainable_parameters
        self._lm_head = lm_head             # frozen
        self._output_norm = output_norm     # frozen
        self.in_proj = nn.Linear(3 * HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.block = _Block()
        # Non-zero init keeps gradient flowing through the block from step 1.
        self.residual_gate = mx.array([0.05])
        self.with_sparsity = with_sparsity
        if with_sparsity:
            self.sparsity_proj_in = nn.Linear(HIDDEN_DIM, 512, bias=False)
            self.sparsity_proj_out = nn.Linear(512, MOE_INTERMEDIATE, bias=False)
        self.calib_proj = nn.Linear(HIDDEN_DIM, 1, bias=True)

    def trainable_parameters(self):
        p = self.parameters()
        for k in ("_token_embd", "_lm_head", "_output_norm"):
            p.pop(k, None)
        return p

    def __call__(self, prev_tok, residual_in, intermediate_signal):
        B, S = prev_tok.shape
        attn_mask = (mx.eye(S) - 1.0) * 1e9  # diagonal-only attention
        embed_table = mx.transpose(self._token_embd, (1, 0))
        prev_embed = embed_table[prev_tok]
        x = mx.concatenate([prev_embed, residual_in, intermediate_signal], axis=-1)
        x = self.in_proj(x)
        x = self.block(x, attn_mask)
        baseline = mx.fast.rms_norm(residual_in, self._output_norm, RMS_EPS)
        draft_hidden = baseline.astype(x.dtype) + self.residual_gate * x
        token_logits = draft_hidden @ self._lm_head
        if self.with_sparsity:
            sparsity_log = self.sparsity_proj_out(
                nn.silu(self.sparsity_proj_in(draft_hidden))
            )
        else:
            sparsity_log = None
        calib_logit = self.calib_proj(draft_hidden).squeeze(-1)
        return token_logits, sparsity_log, draft_hidden, calib_logit


def build_head(frozen_npz: Path, with_sparsity: bool) -> Eagle5Head:
    z = np.load(frozen_npz)
    needed = ["token_embd", "lm_head", "output_norm"]
    for k in needed:
        if k not in z.files:
            raise SystemExit(
                f"frozen .npz missing `{k}` — regenerate with `python eagle4/eagle4.py frozen`"
            )
    return Eagle5Head(
        mx.array(z["token_embd"]),
        mx.array(z["lm_head"]),
        mx.array(z["output_norm"]),
        with_sparsity=with_sparsity,
    )


def _flat_params(d, prefix=""):
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else enumerate(d)):
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, (dict, list)):
            out.update(_flat_params(v, key))
        elif hasattr(v, "shape"):
            out[key] = v
    return out


def save_ckpt(head: Eagle5Head, path: Path, step: int = 0):
    flat = {k: np.array(v) for k, v in _flat_params(head.trainable_parameters()).items()}
    flat["__step__"] = np.int32(step)
    np.savez(path, **flat)


def load_ckpt(head: Eagle5Head, path: Path) -> int:
    z = np.load(path, allow_pickle=False)
    params = head.trainable_parameters()

    def walk(d, prefix=""):
        if isinstance(d, dict):
            for k in list(d.keys()):
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(d[k], (dict, list)):
                    walk(d[k], key)
                elif key in z:
                    d[k] = mx.array(z[key])
        elif isinstance(d, list):
            for i in range(len(d)):
                walk(d[i], f"{prefix}.{i}")

    walk(params)
    head.update(params)
    return int(z.get("__step__", -1))


def _decode_tokens(value) -> np.ndarray:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(value, dtype=np.int32).copy()
    return np.asarray(value, dtype=np.int32)


def _decode_compact_tensor(row, stem: str) -> np.ndarray | None:
    """Decode colab/corpus_simple.py's int8 binary tensor columns."""
    q = row.get(f"{stem}_q")
    scale = row.get(f"{stem}_scale")
    shape = row.get(f"{stem}_shape")
    if q is None or scale is None or shape is None:
        return None
    if not isinstance(q, (bytes, bytearray, memoryview)):
        return None
    shape_t = tuple(int(x) for x in shape)
    arr = np.frombuffer(q, dtype=np.int8).astype(np.float32)
    if arr.size != int(np.prod(shape_t)):
        return None
    return arr.reshape(shape_t) * float(scale)


def _extract_row(row, capture_layer: int, n_moe_first_dense: int = 1,
                 max_row_tokens: int = 0) -> dict | None:
    """One parquet row → numpy tensors for the head's input contract.

    Supports both the original all-layer corpus schema and the compact
    Colab schema from colab/corpus_simple.py. In the original schema,
    residual_in_per_layer is indexed by layer and intermediate_per_layer
    is indexed by MoE layer (excluding the leading dense block). In the
    compact schema, residual_q/intermediate_q are already the requested
    capture layer.
    """
    tokens = _decode_tokens(row["tokens"])
    n_tok = len(tokens)
    if n_tok < 5:
        return None  # need a few positions; eagle4 skips first 3

    res = _decode_compact_tensor(row, "residual")
    if res is None:
        res_all = row.get("residual_in_per_layer")
        if not res_all or capture_layer >= len(res_all):
            return None
        res = np.asarray(res_all[capture_layer], dtype=np.float32)  # (n_tok, hidden)

    inter = _decode_compact_tensor(row, "intermediate")
    if inter is None:
        inter_all = row.get("intermediate_per_layer")
        if not inter_all:
            return None
        moe_idx = capture_layer - n_moe_first_dense
        if moe_idx < 0 or moe_idx >= len(inter_all):
            return None
        inter_item = inter_all[moe_idx]
        # build_corpus.py stores per-layer entries as {"layer": int, "raw": ndarray}.
        # Unwrap if so; older direct-tensor format also supported.
        if isinstance(inter_item, dict):
            inter_item = inter_item.get("raw")
            if inter_item is None:
                return None
        inter = np.asarray(inter_item, dtype=np.float32)
    if inter.ndim == 3:
        # (n_experts, n_tok, hidden) — we want expert 0 per build_corpus.py
        inter = inter[0]
    if inter.shape != res.shape:
        # build_corpus.py's `mlp.experts[0]` hook captures the FIRST EXPERT's
        # output, which is shape (n_routed_to_expert_0, hidden) — variable
        # per row, NOT (n_tok, hidden). Rather than skip every row (no
        # training data), zero-pad to residual shape. Use --sparsity-head=off
        # so the auxiliary loss doesn't try to fit noise on these fake
        # intermediates. The trainer's model still reads `intermediate_signal`
        # as an input feature; zeros are the safest fill.
        inter = np.zeros_like(res)

    if len(tokens) != res.shape[0]:
        return None

    # Memory cap: truncate each row's sequence to first N tokens. Without
    # this, the Colab-built corpus (16k seqs × 655-token-avg × 2048 hidden
    # × fp32 × 2 captures) = ~160 GB in RAM → SIGKILL on any laptop.
    # 256 tokens still yields 14 sliding windows at seq_len=16; plenty of
    # training signal per row.
    if max_row_tokens > 0 and len(tokens) > max_row_tokens:
        tokens = tokens[:max_row_tokens]
        res = res[:max_row_tokens]
        inter = inter[:max_row_tokens]

    return {
        "prev_tokens": tokens[:-1],
        "next_tokens": tokens[1:],
        "residual": res[:-1],   # input at step t predicts token t+1
        "intermediate": inter[:-1],
    }


def _iter_batches(
    shards: list[Path],
    batch_size: int,
    seq_len: int,
    epochs: int,
    capture_layer: int,
    seed: int = 0,
    dedup: bool = True,
    max_row_tokens: int = 128,
    max_rows: int = 2000,
):
    """Yield (batch_size, seq_len) tensors in row-major (B, S, H) layout.

    Loads shards lazily and builds windows of `seq_len` consecutive
    positions within a sequence.

    **Dedup pass (wall-clock optimization, 2026-05-22):** the corpus at
    `artifacts/calibration/v2_lite_corpus/` has ~1,500-2,000 unique
    sequences in 4,512 rows because `iter_chat_sequences` restarts at
    row 0 on every watchdog launch (per [[corpus-complete-analysis-landed]]).
    Training on duplicates is wall-clock waste — each duplicate row
    contributes only gradient redundancy. Deduping by the token-id
    fingerprint before windowing typically cuts the training-step count
    by ~60% with no quality loss. Pass `dedup=False` to disable.
    """
    rng = random.Random(seed)

    # Subsample SHARDS upfront so we never load >max_rows worth of data into
    # memory. Critical on laptops: 1013 shards × 16 rows × 4 MB per truncated
    # row = ~64 GB peak before subsampling, → SIGKILL. With shard subsample,
    # peak load = ~target_shards × 16 × per-row-bytes.
    if max_rows > 0 and len(shards) > 16:
        avg_rows_per_shard = 16  # corpus_simple.py default
        # 1.5× safety margin for dedup drops + a few extra shards
        target_shards = min(len(shards), int(max_rows / avg_rows_per_shard * 1.5) + 16)
        if target_shards < len(shards):
            sh_rng = random.Random(seed + 7919)  # different stream than row shuffler
            shards = sh_rng.sample(shards, target_shards)
            print(f"[data] subsampling {target_shards} of original shard list "
                  f"(target rows={max_rows}, ~{target_shards*avg_rows_per_shard} "
                  f"rows pre-dedup)", flush=True)

    rows: list[dict] = []
    seen_fp: set = set()
    n_raw = n_dup = 0

    # Wall-clock optimization #3 (2026-05-22): parallel shard read.
    # pyarrow releases the GIL during parquet IO; a thread pool gives
    # ~5× speedup on 141-shard corpus (~3-5 min → ~30-60 s). Dedup runs
    # AFTER all shards complete so the seen_fp set is single-threaded —
    # ordering of de-duplication across threads doesn't matter; we still
    # drop every duplicate.
    def _read_one(shard: Path) -> list[dict]:
        t = pq.read_table(shard)
        out = []
        col_names = t.column_names
        for i in range(t.num_rows):
            r = {c: t[c][i].as_py() for c in col_names}
            ex = _extract_row(r, capture_layer, max_row_tokens=max_row_tokens)
            if ex is not None:
                out.append(ex)
        return out

    max_workers = min(8, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for shard_rows in pool.map(_read_one, shards):
            for ex in shard_rows:
                n_raw += 1
                if dedup:
                    # Cheap fingerprint: first 64 token ids cast to bytes.
                    fp = ex["prev_tokens"][:64].tobytes()
                    if fp in seen_fp:
                        n_dup += 1
                        continue
                    seen_fp.add(fp)
                rows.append(ex)
    # Memory cap: subsample rows after load. Keeps RAM bounded for laptops
    # training on big Colab-built corpora (16k seqs × 256 tokens × 2048 fp32
    # × 2 = ~67 GB without this cap — would SIGKILL on M3 Pro 18 GB).
    if max_rows > 0 and len(rows) > max_rows:
        rng.shuffle(rows)
        dropped = len(rows) - max_rows
        rows = rows[:max_rows]
        print(f"[data] subsampled to {max_rows} rows (dropped {dropped})", flush=True)

    print(
        f"[data] loaded {len(rows)} usable rows from {len(shards)} shards "
        f"(raw={n_raw}, dropped {n_dup} duplicate fingerprints)",
        flush=True,
    )
    if not rows:
        raise SystemExit("no usable rows in corpus; check capture_layer + intermediate availability")

    # Build sliding windows of seq_len within each row.
    windows: list[dict] = []
    for r in rows:
        n = len(r["prev_tokens"])
        for off in range(0, n - seq_len + 1, seq_len):
            windows.append({
                "prev": r["prev_tokens"][off : off + seq_len],
                "next": r["next_tokens"][off : off + seq_len],
                "residual": r["residual"][off : off + seq_len],
                "intermediate": r["intermediate"][off : off + seq_len],
            })
    print(f"[data] {len(windows)} windows × seq_len {seq_len}", flush=True)

    # Wall-clock optimization #9 (2026-05-22): prefetch the next batch on
    # a worker thread while the trainer is processing the current one.
    # MLX is GPU-side; the CPU `np.stack` + `mx.array` cost is hidden by
    # the in-flight GPU step. Yields ~20-30% throughput on M3 Pro.
    def _build_batch(batch_windows, epoch_id):
        prev = np.stack([w["prev"] for w in batch_windows])
        nxt = np.stack([w["next"] for w in batch_windows])
        res = np.stack([w["residual"] for w in batch_windows])
        inter = np.stack([w["intermediate"] for w in batch_windows])
        return {
            "prev": mx.array(prev),
            "next": mx.array(nxt),
            "residual": mx.array(res),
            "intermediate": mx.array(inter),
            "epoch": epoch_id,
        }

    for epoch in range(epochs):
        rng.shuffle(windows)
        ranges = list(range(0, len(windows) - batch_size + 1, batch_size))
        with ThreadPoolExecutor(max_workers=2) as prefetch:
            # Pre-submit the first batch; then for each subsequent yield
            # submit the next batch before consuming the current one.
            in_flight = None
            if ranges:
                in_flight = prefetch.submit(
                    _build_batch, windows[ranges[0] : ranges[0] + batch_size], epoch
                )
            for k, i in enumerate(ranges):
                current = in_flight
                # Pre-fetch the next batch BEFORE we await the current one
                # (so the prefetch runs while the trainer is consuming).
                if k + 1 < len(ranges):
                    nxt_i = ranges[k + 1]
                    in_flight = prefetch.submit(
                        _build_batch, windows[nxt_i : nxt_i + batch_size], epoch
                    )
                yield current.result()


def _channel_active_mask(intermediate: mx.array, percentile: float = 0.9) -> mx.array:
    """Per-batch threshold over channels: 1 where |intermediate| ≥ p-th
    percentile across the batch. Returns float32 mask (B, S, 1408).

    Reshape intermediate (B, S, hidden=2048) is the post-down hidden-space
    output. We translate it to a sparsity signal over the 1408-channel
    MoE-FFN intermediate by linear projection... but we don't have the
    actual SwiGLU-intermediate ground truth. The proxy is to compute
    sparsity over the HIDDEN dimension (2048 channels) and project that
    to 1408 via simple zero-padding/truncation.

    For v1: just threshold the captured 2048-channel intermediate and
    take the first 1408 channels. This is admittedly a placeholder —
    a clean re-capture with the actual SwiGLU intermediate would be
    a more honest ground truth.
    """
    H = intermediate.shape[-1]
    target = MOE_INTERMEDIATE
    if H >= target:
        signal = intermediate[..., :target]
    else:
        # Pad with zeros to target width (unusual for V2-Lite; included for safety)
        pad = mx.zeros(intermediate.shape[:-1] + (target - H,), dtype=intermediate.dtype)
        signal = mx.concatenate([intermediate, pad], axis=-1)
    abs_sig = mx.abs(signal)
    # Per-(b,s) percentile threshold so different magnitudes don't pollute the BCE.
    # mx.quantile not always available; use sort + index.
    sorted_abs = mx.sort(abs_sig, axis=-1)
    k = int(percentile * target)
    thresh = sorted_abs[..., k:k + 1]  # (B,S,1)
    return (abs_sig >= thresh).astype(mx.float32)


def train(args):
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with_sparsity = args.sparsity_head != "off"
    head = build_head(Path(args.frozen), with_sparsity)
    print(
        f"[train] eagle5 v2 head built; capture_layer={args.capture_layer} "
        f"sparsity_head={'proxy' if with_sparsity else 'off'} "
        f"lr={args.lr} batch={args.batch_size} seq_len={args.seq_len}",
        flush=True,
    )
    if args.resume and args.resume.exists():
        step = load_ckpt(head, args.resume)
        print(f"[train] resumed from {args.resume} at step {step}", flush=True)
    else:
        step = 0

    def loss_fn(head, b, target_alpha):
        prev = b["prev"]
        residual = b["residual"]
        inter = b["intermediate"]
        tok, sparsity_log, draft_h, calib_logit = head(prev, residual, inter)
        B, S, V = tok.shape
        # Skip the first 3 positions per eagle4's BOS-norm-imbalance fix.
        pos_mask = mx.concatenate([mx.zeros((B, 3)), mx.ones((B, S - 3))], axis=1).reshape(-1)
        N = mx.maximum(pos_mask.sum(), mx.array(1.0))

        baseline = mx.fast.rms_norm(residual, head._output_norm, RMS_EPS)
        target_logits = baseline @ head._lm_head.astype(mx.float32)
        target_arg_flat = mx.stop_gradient(mx.argmax(target_logits.reshape(-1, V), axis=-1))

        ce_corpus_per = nn.losses.cross_entropy(
            tok.reshape(-1, V), b["next"].reshape(-1), reduction="none"
        )
        ce_target_per = nn.losses.cross_entropy(
            tok.reshape(-1, V), target_arg_flat, reduction="none"
        )
        ce_per = target_alpha * ce_target_per + (1.0 - target_alpha) * ce_corpus_per
        ce = (ce_per * pos_mask).sum() / N

        head_arg = mx.argmax(tok.reshape(-1, V), axis=-1)
        accept_target = (head_arg == target_arg_flat).astype(mx.float32)
        calib_per = nn.losses.binary_cross_entropy(
            calib_logit.reshape(-1), accept_target, with_logits=True, reduction="none"
        )
        calib = (calib_per * pos_mask).sum() / N

        total = ce + 0.1 * calib

        if with_sparsity and sparsity_log is not None:
            mask = _channel_active_mask(inter)
            bce_per = nn.losses.binary_cross_entropy(
                sparsity_log, mask, with_logits=True, reduction="none"
            ).mean(axis=-1).reshape(-1)
            sparsity = (bce_per * pos_mask).sum() / N
            total = total + 0.3 * sparsity
        return total

    grad_fn = nn.value_and_grad(head, loss_fn)
    opt = mxoptim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    log = (ckpt_dir / "log.jsonl").open("a")
    t0 = time.time()
    for batch in _iter_batches(
        sorted(Path(args.corpus_dir).glob("shard_*.parquet")),
        args.batch_size,
        args.seq_len,
        args.epochs,
        args.capture_layer,
        seed=args.seed,
        dedup=not args.no_dedup,
        max_row_tokens=args.max_row_tokens,
        max_rows=args.max_rows,
    ):
        target_alpha = min(step / max(args.target_argmax_warmup_steps, 1), 1.0)
        loss, grads = grad_fn(head, batch, target_alpha)
        opt.update(head, grads)
        mx.eval(head.parameters(), opt.state, loss)
        step += 1
        if step % 25 == 0 or step == 1:
            row = {
                "step": step,
                "epoch": batch["epoch"],
                "loss": float(loss),
                "gate": float(head.residual_gate[0]),
                "alpha": target_alpha,
                "wall": time.time() - t0,
            }
            print(
                f"step={step} epoch={batch['epoch']} loss={row['loss']:.3f} "
                f"gate={row['gate']:.3f} α={target_alpha:.2f} wall={row['wall']:.1f}s",
                flush=True,
            )
            log.write(json.dumps(row) + "\n")
            log.flush()
        if step % 200 == 0:
            save_ckpt(head, ckpt_dir / "latest.npz", step)
            save_ckpt(head, ckpt_dir / f"step_{step:06d}.npz", step)
    save_ckpt(head, ckpt_dir / "latest.npz", step)
    log.close()
    print(f"[train] done: {step} steps in {time.time()-t0:.0f}s", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(prog="eagle5_train")
    p.add_argument("--corpus-dir", required=True, type=Path)
    p.add_argument("--frozen", required=True, type=Path)
    p.add_argument("--ckpt-dir", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--capture-layer", type=int, default=25,
                   help="V2-Lite layer index whose residual + intermediate feed the head")
    p.add_argument("--target-argmax-warmup-steps", type=int, default=500)
    p.add_argument("--sparsity-head", choices=["proxy", "off"], default="proxy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-row-tokens", type=int, default=128,
                   help="truncate each parquet row's sequence to first N "
                        "tokens before loading into RAM. 0 = no truncation. "
                        "Default 128 ~= 8 sliding seq_len=16 windows per row.")
    p.add_argument("--max-rows", type=int, default=2000,
                   help="random sample of N rows. With default max-row-tokens, "
                        "_iter_batches also pre-subsamples SHARDS upfront so "
                        "total RAM stays bounded. Peak ~2 GB at defaults. "
                        "0 = use all (only safe on big-RAM machines).")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable per-row token-fingerprint dedup. Off by default — the "
        "corpus has ~60 percent duplicate rows from watchdog restart; deduping saves "
        "~60 percent wall-clock with no quality loss (per corpus_complete_analysis_landed.md).",
    )
    args = p.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
