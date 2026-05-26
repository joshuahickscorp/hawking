#!/usr/bin/env python3
"""PyTorch port of `tools/training/eagle5_train.py` (MLX) for Colab GPUs.

Trains the Eagle5 v2 head for Qwen-3B (dense) on the compact parquet
corpus produced by `colab/mega_calibrate.py`. Architecture, loss, and
data contract mirror the MLX version 1:1 so the output `latest.npz`
is drop-in compatible with the existing tau-eval and quantize
toolchain.

Key differences vs the MLX version
----------------------------------
* Runs on CUDA (H100 / A100 / L4 / T4). bf16/fp16 autocast for the
  attention + projection forward; fp32 for the baseline/target-logit
  path that drives the cross-entropy target (numerical match with MLX
  which always does that path in fp32).
* Sparsity head is hard-coded OFF for Qwen-3B (dense; no MoE). The
  flag is still parsed for parity with the MLX CLI but `proxy` is a
  no-op here.
* Dispatcher loop uses `torch.optim.AdamW` instead of `mlx.optimizers`.
* Optionally writes `head_final.safetensors` alongside `latest.npz`
  for direct consumption by `dismantle --eagle5-head <path>`.
* `torch.compile` is opt-in (`--compile`) — on cold start its trace
  cost outweighs benefit for short Colab sessions.

Input contract (`--corpus-dir`)
-------------------------------
Parquet shards `shard_*.parquet`, each row:
* `tokens` : bytes (int32 packed)
* `residual_q`         : bytes (int8 packed)
* `residual_scale`     : f32 scalar
* `residual_shape`     : list[int]   (n_tokens, hidden)
* `intermediate_q`     : bytes (int8 packed)
* `intermediate_scale` : f32 scalar
* `intermediate_shape` : list[int]

Frozen weights (`--frozen`)
---------------------------
NPZ with `token_embd` (hidden, vocab) fp16, `lm_head` (hidden, vocab)
fp16, `output_norm` (hidden,) fp32. Produced by
`eagle4/eagle4.py frozen` or the Qwen-3B equivalent at
`eagle4/qwen3b_frozen.npz`.
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
from typing import Optional

import numpy as np
import pyarrow.parquet as pq

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("ERROR: torch not installed. `pip install torch`.", file=sys.stderr)
    sys.exit(1)


# Qwen-3B constants — see CLAUDE.md / handoff doc.
HIDDEN_DIM = 2048
N_HEADS = 16
RMS_EPS = 1e-6
MOE_INTERMEDIATE = 1408  # unused for dense Qwen, kept for API parity


# ────────────────────────────────────────────────────────────────────────
# Architecture
# ────────────────────────────────────────────────────────────────────────

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    """LLaMA-style RMSNorm: x * w / sqrt(mean(x^2) + eps). fp32 inside."""
    in_dtype = x.dtype
    x32 = x.float()
    rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x32 * rms * weight.float()).to(in_dtype)


class _SwiGLU(nn.Module):
    def __init__(self, h: int, i: int):
        super().__init__()
        self.gate = nn.Linear(h, i, bias=False)
        self.up = nn.Linear(h, i, bias=False)
        self.down = nn.Linear(i, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _Block(nn.Module):
    """Pre-norm transformer block with multi-head self-attention + SwiGLU."""

    def __init__(self):
        super().__init__()
        self.attn_norm = nn.Parameter(torch.ones(HIDDEN_DIM))
        # mlx.nn.MultiHeadAttention has bias=False, so set bias=False here.
        # We implement attention manually so we can pass an additive mask.
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.out_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.mlp_norm = nn.Parameter(torch.ones(HIDDEN_DIM))
        self.mlp = _SwiGLU(HIDDEN_DIM, 4 * HIDDEN_DIM)

    def _attn(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, _ = h.shape
        head_dim = HIDDEN_DIM // N_HEADS
        q = self.q_proj(h).view(B, S, N_HEADS, head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, N_HEADS, head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, N_HEADS, head_dim).transpose(1, 2)
        # scores: (B, n_heads, S, S)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
        scores = scores + mask
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, HIDDEN_DIM)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = _rms_norm(x, self.attn_norm, RMS_EPS)
        x = x + self._attn(h, mask)
        h = _rms_norm(x, self.mlp_norm, RMS_EPS)
        x = x + self.mlp(h)
        return x


class Eagle5Head(nn.Module):
    """Mirror of MLX Eagle5Head. Sparsity head only enabled when MoE."""

    def __init__(
        self,
        token_embd: torch.Tensor,   # fp16 (hidden, vocab)
        lm_head: torch.Tensor,      # fp16 (hidden, vocab)
        output_norm: torch.Tensor,  # fp32 (hidden,)
        with_sparsity: bool = False,
    ):
        super().__init__()
        # Frozen buffers, NOT registered as parameters. Persistent so they
        # ride with the module move-to-device but are NOT in the optimizer.
        self.register_buffer("_token_embd", token_embd, persistent=False)
        self.register_buffer("_lm_head", lm_head, persistent=False)
        self.register_buffer("_output_norm", output_norm, persistent=False)

        self.in_proj = nn.Linear(3 * HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.block = _Block()
        # Scalar gate — non-zero init keeps gradient flowing through the block.
        self.residual_gate = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))
        self.with_sparsity = with_sparsity
        if with_sparsity:
            self.sparsity_proj_in = nn.Linear(HIDDEN_DIM, 512, bias=False)
            self.sparsity_proj_out = nn.Linear(512, MOE_INTERMEDIATE, bias=False)
        self.calib_proj = nn.Linear(HIDDEN_DIM, 1, bias=True)

    def forward(
        self,
        prev_tok: torch.Tensor,         # (B, S) int64
        residual_in: torch.Tensor,      # (B, S, H) fp32
        intermediate_signal: torch.Tensor,  # (B, S, H) fp32
    ):
        B, S = prev_tok.shape
        # Diagonal-only attention mask: -1e9 off-diagonal, 0 on-diagonal.
        eye = torch.eye(S, device=prev_tok.device, dtype=torch.float32)
        attn_mask = (eye - 1.0) * 1e9
        # Broadcast to (1, 1, S, S).
        attn_mask = attn_mask.view(1, 1, S, S)

        # token_embd is stored (hidden, vocab); transpose to (vocab, hidden)
        # for the row lookup.
        embed_table = self._token_embd.transpose(0, 1)  # (vocab, hidden)
        prev_embed = embed_table[prev_tok]              # (B, S, hidden)

        x = torch.cat([prev_embed.to(residual_in.dtype), residual_in, intermediate_signal], dim=-1)
        x = self.in_proj(x)
        x = self.block(x, attn_mask.to(x.dtype))

        # Baseline = RMSNorm(residual_in) — done in fp32 like MLX.
        baseline = _rms_norm(residual_in, self._output_norm, RMS_EPS)
        draft_hidden = baseline.to(x.dtype) + self.residual_gate * x

        # lm_head is (hidden, vocab) — matmul gives (B, S, vocab).
        token_logits = torch.matmul(draft_hidden, self._lm_head.to(draft_hidden.dtype))
        sparsity_log = None
        if self.with_sparsity:
            sparsity_log = self.sparsity_proj_out(F.silu(self.sparsity_proj_in(draft_hidden)))
        calib_logit = self.calib_proj(draft_hidden).squeeze(-1)
        return token_logits, sparsity_log, draft_hidden, calib_logit

    def trainable_state(self) -> dict[str, torch.Tensor]:
        """State dict of trainable params only (frozen buffers excluded)."""
        out = {}
        for k, v in self.named_parameters():
            out[k] = v.detach()
        return out


def build_head(frozen_npz: Path, with_sparsity: bool, device: str) -> Eagle5Head:
    z = np.load(frozen_npz)
    needed = ["token_embd", "lm_head", "output_norm"]
    for k in needed:
        if k not in z.files:
            raise SystemExit(
                f"frozen .npz missing `{k}` — regenerate with `python eagle4/eagle4.py frozen`"
            )
    head = Eagle5Head(
        torch.from_numpy(np.asarray(z["token_embd"], dtype=np.float16)),
        torch.from_numpy(np.asarray(z["lm_head"], dtype=np.float16)),
        torch.from_numpy(np.asarray(z["output_norm"], dtype=np.float32)),
        with_sparsity=with_sparsity,
    )
    head = head.to(device)
    return head


# ────────────────────────────────────────────────────────────────────────
# Checkpoint I/O — `latest.npz` matches MLX format for tau-eval compat.
# ────────────────────────────────────────────────────────────────────────

def _state_to_npz_flat(head: Eagle5Head) -> dict[str, np.ndarray]:
    """Flatten trainable params into the MLX-compatible key naming."""
    flat: dict[str, np.ndarray] = {}
    for k, v in head.named_parameters():
        flat[k] = v.detach().cpu().numpy()
    return flat


def save_ckpt(head: Eagle5Head, path: Path, step: int = 0) -> None:
    flat = _state_to_npz_flat(head)
    flat["__step__"] = np.int32(step)
    np.savez(path, **flat)


def save_safetensors(head: Eagle5Head, path: Path) -> None:
    """Write trainable params as a safetensors file for dismantle loading."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("WARN: safetensors not installed; skipping safetensors export.",
              file=sys.stderr)
        return
    state = {k: v.detach().cpu().contiguous() for k, v in head.named_parameters()}
    # Also dump the frozen tensors so the loader can reconstruct without
    # needing a separate frozen.npz on the dismantle side.
    state["_token_embd"] = head._token_embd.detach().cpu().contiguous()
    state["_lm_head"] = head._lm_head.detach().cpu().contiguous()
    state["_output_norm"] = head._output_norm.detach().cpu().contiguous()
    save_file(state, str(path))


# ────────────────────────────────────────────────────────────────────────
# Data loading — port of MLX _iter_batches.
# ────────────────────────────────────────────────────────────────────────

def _decode_tokens(value) -> np.ndarray:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(value, dtype=np.int32).copy()
    return np.asarray(value, dtype=np.int32)


def _decode_compact_tensor(row, stem: str) -> Optional[np.ndarray]:
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
    # Dequantize in fp32 for precision, hold in fp16 for memory.
    return (arr.reshape(shape_t) * float(scale)).astype(np.float16)


def _extract_row(row, max_row_tokens: int = 0) -> Optional[dict]:
    tokens = _decode_tokens(row["tokens"])
    n_tok = len(tokens)
    if n_tok < 5:
        return None

    res = _decode_compact_tensor(row, "residual")
    if res is None:
        return None
    inter = _decode_compact_tensor(row, "intermediate")
    if inter is None:
        return None

    if inter.ndim == 3:
        # (n_experts, n_tok, hidden) — eagle4 corpus oddity. Take expert 0.
        inter = inter[0]
    if inter.shape != res.shape:
        # Zero-pad to residual shape when first-expert capture is sparse.
        inter = np.zeros_like(res)
    if len(tokens) != res.shape[0]:
        return None

    if max_row_tokens > 0 and len(tokens) > max_row_tokens:
        tokens = tokens[:max_row_tokens]
        res = res[:max_row_tokens]
        inter = inter[:max_row_tokens]

    return {
        "prev_tokens": tokens[:-1],
        "next_tokens": tokens[1:],
        "residual": res[:-1],
        "intermediate": inter[:-1],
    }


def _iter_batches(
    shards: list[Path],
    batch_size: int,
    seq_len: int,
    epochs: int,
    seed: int = 0,
    dedup: bool = True,
    max_row_tokens: int = 128,
    max_rows: int = 4000,
):
    rng = random.Random(seed)

    # Subsample shards upfront so we never load >max_rows worth of data.
    if max_rows > 0 and len(shards) > 16:
        avg_rows_per_shard = 16
        target_shards = min(len(shards), int(max_rows / avg_rows_per_shard * 1.5) + 16)
        if target_shards < len(shards):
            sh_rng = random.Random(seed + 7919)
            shards = sh_rng.sample(shards, target_shards)
            print(f"[data] subsampling {target_shards} of original shard list "
                  f"(target rows={max_rows})", flush=True)

    rows: list[dict] = []
    seen_fp: set = set()
    n_raw = n_dup = 0

    def _read_one(shard: Path) -> list[dict]:
        t = pq.read_table(shard)
        out = []
        col_names = t.column_names
        for i in range(t.num_rows):
            r = {c: t[c][i].as_py() for c in col_names}
            ex = _extract_row(r, max_row_tokens=max_row_tokens)
            if ex is not None:
                out.append(ex)
        return out

    max_workers = min(8, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for shard_rows in pool.map(_read_one, shards):
            for ex in shard_rows:
                n_raw += 1
                if dedup:
                    fp = ex["prev_tokens"][:64].tobytes()
                    if fp in seen_fp:
                        n_dup += 1
                        continue
                    seen_fp.add(fp)
                rows.append(ex)

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
        raise SystemExit("no usable rows in corpus")

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
    if not windows:
        raise SystemExit(
            f"no training windows built; lower --seq-len {seq_len} or raise "
            f"--max-row-tokens {max_row_tokens}"
        )

    def _build_batch(batch_windows, epoch_id):
        prev = np.stack([w["prev"] for w in batch_windows])         # (B, S) i32
        nxt = np.stack([w["next"] for w in batch_windows])
        res = np.stack([w["residual"] for w in batch_windows]).astype(np.float32)
        inter = np.stack([w["intermediate"] for w in batch_windows]).astype(np.float32)
        return {
            "prev": torch.from_numpy(prev.astype(np.int64)),
            "next": torch.from_numpy(nxt.astype(np.int64)),
            "residual": torch.from_numpy(res),
            "intermediate": torch.from_numpy(inter),
            "epoch": epoch_id,
        }

    for epoch in range(epochs):
        rng.shuffle(windows)
        ranges = list(range(0, len(windows) - batch_size + 1, batch_size))
        with ThreadPoolExecutor(max_workers=2) as prefetch:
            in_flight = None
            if ranges:
                in_flight = prefetch.submit(
                    _build_batch, windows[ranges[0] : ranges[0] + batch_size], epoch
                )
            for k, i in enumerate(ranges):
                current = in_flight
                if k + 1 < len(ranges):
                    nxt_i = ranges[k + 1]
                    in_flight = prefetch.submit(
                        _build_batch, windows[nxt_i : nxt_i + batch_size], epoch
                    )
                yield current.result()


# ────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────

def train(args) -> None:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARN: cuda requested but unavailable; falling back to cpu", file=sys.stderr)
        device = "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with_sparsity = args.sparsity_head != "off"
    if with_sparsity:
        print("[train] sparsity_head=proxy requested; this is intended for MoE only — "
              "ignoring for Qwen-3B (dense)", flush=True)
        with_sparsity = False

    head = build_head(Path(args.frozen), with_sparsity, device)
    print(
        f"[train] eagle5 v2 head built; capture_layer={args.capture_layer} "
        f"sparsity_head=off lr={args.lr} batch={args.batch_size} "
        f"seq_len={args.seq_len} device={device}",
        flush=True,
    )

    # Mixed precision: bf16 on Ampere+, fp16 fallback. fp32 on CPU.
    if device == "cuda":
        major = torch.cuda.get_device_capability()[0]
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        amp_dtype = torch.float32
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if (use_amp and amp_dtype == torch.float16) else None

    opt = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    fwd_fn = head
    if args.compile:
        try:
            fwd_fn = torch.compile(head, mode="reduce-overhead")
            print("[train] torch.compile enabled (reduce-overhead)", flush=True)
        except Exception as e:
            print(f"[train] torch.compile failed: {e}; running eager", flush=True)
            fwd_fn = head

    log = (ckpt_dir / "log.jsonl").open("a")
    t0 = time.time()
    step = 0
    V = head._lm_head.shape[1]

    for batch in _iter_batches(
        sorted(Path(args.corpus_dir).glob("shard_*.parquet")),
        args.batch_size,
        args.seq_len,
        args.epochs,
        seed=args.seed,
        dedup=not args.no_dedup,
        max_row_tokens=args.max_row_tokens,
        max_rows=args.max_rows,
    ):
        # Move tensors to device once per batch.
        prev = batch["prev"].to(device, non_blocking=True)
        nxt = batch["next"].to(device, non_blocking=True)
        residual = batch["residual"].to(device, non_blocking=True)
        inter = batch["intermediate"].to(device, non_blocking=True)
        B, S = prev.shape

        target_alpha = min(step / max(args.target_argmax_warmup_steps, 1), 1.0)

        opt.zero_grad(set_to_none=True)
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else torch.enable_grad()
        with ctx:
            token_logits, _sparsity, draft_h, calib_logit = fwd_fn(prev, residual, inter)

            # Position mask: skip first 3 positions (BOS-norm-imbalance fix).
            pos_mask = torch.ones(B, S, device=device, dtype=torch.float32)
            pos_mask[:, :3] = 0.0
            pos_mask_flat = pos_mask.reshape(-1)
            N = pos_mask_flat.sum().clamp(min=1.0)

            # Baseline + target-argmax — fp32 for parity with MLX.
            with torch.amp.autocast(device_type="cuda", enabled=False) if use_amp else torch.enable_grad():
                baseline = _rms_norm(residual, head._output_norm, RMS_EPS).float()
                target_logits = torch.matmul(baseline, head._lm_head.float())
                target_arg_flat = target_logits.reshape(-1, V).argmax(dim=-1).detach()

            tok_flat = token_logits.reshape(-1, V).float()
            ce_corpus_per = F.cross_entropy(tok_flat, nxt.reshape(-1), reduction="none")
            ce_target_per = F.cross_entropy(tok_flat, target_arg_flat, reduction="none")
            ce_per = target_alpha * ce_target_per + (1.0 - target_alpha) * ce_corpus_per
            ce = (ce_per * pos_mask_flat).sum() / N

            head_arg = tok_flat.argmax(dim=-1)
            accept_target = (head_arg == target_arg_flat).float()
            calib_per = F.binary_cross_entropy_with_logits(
                calib_logit.reshape(-1).float(), accept_target, reduction="none"
            )
            calib = (calib_per * pos_mask_flat).sum() / N

            residual_delta = torch.zeros((), device=device, dtype=torch.float32)
            if args.residual_delta_loss_weight > 0.0 and S > 1:
                pred_next = _rms_norm(draft_h[:, :-1, :], head._output_norm, RMS_EPS).float()
                target_next = _rms_norm(
                    residual[:, 1:, :], head._output_norm, RMS_EPS
                ).float().detach()
                delta_per = (pred_next - target_next).pow(2).mean(dim=-1)
                delta_mask = pos_mask[:, 1:]
                residual_delta = (
                    (delta_per * delta_mask).sum()
                    / delta_mask.sum().clamp(min=1.0)
                )

            total = (
                ce
                + args.calib_loss_weight * calib
                + args.residual_delta_loss_weight * residual_delta
            )

        if scaler is not None:
            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()
        else:
            total.backward()
            opt.step()

        step += 1
        if step % 25 == 0 or step == 1:
            row = {
                "step": step,
                "epoch": batch["epoch"],
                "loss": float(total.detach()),
                "gate": float(head.residual_gate.detach()[0]),
                "alpha": target_alpha,
                "calib": float(calib.detach()),
                "residual_delta": float(residual_delta.detach()),
                "wall": time.time() - t0,
            }
            print(
                f"step={step} epoch={row['epoch']} loss={row['loss']:.3f} "
                f"gate={row['gate']:.3f} α={target_alpha:.2f} "
                f"calib={row['calib']:.3f} rd={row['residual_delta']:.4f} "
                f"wall={row['wall']:.1f}s",
                flush=True,
            )
            log.write(json.dumps(row) + "\n")
            log.flush()
        if step % 500 == 0:
            save_ckpt(head, ckpt_dir / "latest.npz", step)
            save_ckpt(head, ckpt_dir / f"step_{step:06d}.npz", step)

    save_ckpt(head, ckpt_dir / "latest.npz", step)
    if args.save_safetensors:
        path = ckpt_dir / "head_final.safetensors"
        save_safetensors(head, path)
        print(f"[train] safetensors → {path}", flush=True)
    log.close()
    print(f"[train] done: {step} steps in {time.time()-t0:.0f}s", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(prog="eagle5_train_pytorch")
    p.add_argument("--corpus-dir", required=True, type=Path)
    p.add_argument("--frozen", required=True, type=Path,
                   help="Qwen-3B frozen.npz (eagle4/qwen3b_frozen.npz)")
    p.add_argument("--ckpt-dir", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--capture-layer", type=int, default=32,
                   help="metadata only — actual capture is baked into the corpus")
    p.add_argument("--target-argmax-warmup-steps", type=int, default=500)
    p.add_argument("--calib-loss-weight", type=float, default=0.1,
                   help="Weight for the confidence/calibration BCE head.")
    p.add_argument("--residual-delta-loss-weight", type=float, default=0.0,
                   help="Optional frontier objective: make draft_hidden track "
                        "the next residual state, improving multi-step "
                        "simulation readiness without adding runtime params.")
    p.add_argument("--sparsity-head", choices=["proxy", "off"], default="off",
                   help="off for Qwen-3B (dense); proxy is MoE-only and ignored here")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-row-tokens", type=int, default=128)
    p.add_argument("--max-rows", type=int, default=4000,
                   help="random sample of N rows. On Colab H100 8000-16000 is fine.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--save-safetensors", action="store_true",
                   help="also write head_final.safetensors for dismantle --eagle5-head")
    p.add_argument("--compile", action="store_true",
                   help="enable torch.compile (slow first step, faster steady-state)")
    p.add_argument("--no-dedup", action="store_true")
    args = p.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
