"""EAGLE-4 head for DeepSeek-V2-Lite-Chat — one file, ~500 lines.

  EagleHead       5-input fusion + transformer block + token/mask/calib heads
  train           hybrid CE ramp + aux MSE + BCE on captured data
  evaluate        target-argmax acceptance + per-layer mask recall
  quantize_head   post-training Q4 quantization of dense weights
  extract_frozen  V2-Lite token_embd / lm_head / output_norm → .npz

The head's `draft_hidden = post_norm(h_high) + α·block(...)` with α=0.05
at init: at step 1 the head is near-identity on V2-Lite's own output,
and training learns the small refinement. See ARCHITECTURE.md.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as mxoptim
import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants — V2-Lite specifics
# ---------------------------------------------------------------------------
HIDDEN_DIM = 2048
VOCAB = 102_400
N_MOE_LAYERS = 26
N_ROUTED = 64
TOP_K = 6
N_HEADS = 16
INTERMEDIATE = 5_632
RMS_EPS = 1e-6


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------
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
        self.mlp = _SwiGLU(HIDDEN_DIM, INTERMEDIATE)

    def __call__(self, x, mask):
        h = mx.fast.rms_norm(x, self.attn_norm, RMS_EPS)
        x = x + self.attn(h, h, h, mask=mask)
        h = mx.fast.rms_norm(x, self.mlp_norm, RMS_EPS)
        x = x + self.mlp(h)
        return x


class EagleHead(nn.Module):
    def __init__(self, token_embd, lm_head, output_norm, gate_init: float = 0.05):
        super().__init__()
        self._token_embd = token_embd
        self._lm_head = lm_head
        self._output_norm = output_norm
        self.in_proj = nn.Linear(5 * HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.block = _Block()
        # path-to-125 L8 — `gate_init` is the residual_gate's initial value.
        # The default (0.05) matches v3's init. Path-to-125 L8 closeout
        # found that v3-warm-started chain training plateaus at ~7% accept
        # because the gate stays clamped at ~0.001 (block_output magnitudes
        # are tiny under MSE pressure). Re-training FROM SCRATCH with a
        # larger gate_init (typically 0.1) forces the head to either learn
        # to use the block's output or actively zero the gate; reports
        # /path_to_90/session_closeout_2026-05-19b.md § Branch 3 has the
        # diagnosis. The training script's `--gate-init` flag threads
        # through to this constructor.
        self.residual_gate = mx.array([float(gate_init)])
        self.mask_proj_in = nn.Linear(HIDDEN_DIM, 512, bias=False)
        self.mask_proj_out = nn.Linear(512, N_MOE_LAYERS * N_ROUTED, bias=False)
        self.calib_proj = nn.Linear(HIDDEN_DIM, 1, bias=True)

    def trainable_parameters(self):
        p = self.parameters()
        for k in ("_token_embd", "_lm_head", "_output_norm"):
            p.pop(k, None)
        return p

    def __call__(self, prev_tok, h_low, h_mid, h_high, h_shared):
        B, S = prev_tok.shape
        attn_mask = (mx.eye(S) - 1.0) * 1e9  # diagonal-only attention
        embed_table = mx.transpose(self._token_embd, (1, 0))
        prev_embed = embed_table[prev_tok]
        x = mx.concatenate([prev_embed, h_low, h_mid, h_high, h_shared], axis=-1)
        x = self.in_proj(x)
        x = self.block(x, attn_mask)
        baseline = mx.fast.rms_norm(h_high, self._output_norm, RMS_EPS)
        draft_hidden = baseline.astype(x.dtype) + self.residual_gate * x
        token_logits = draft_hidden @ self._lm_head
        mask_logits = self.mask_proj_out(nn.silu(self.mask_proj_in(draft_hidden)))
        mask_logits = mask_logits.reshape(B, S, N_MOE_LAYERS, N_ROUTED)
        calib_logit = self.calib_proj(draft_hidden).squeeze(-1)  # (B, S), sigmoid at use-time
        return token_logits, mask_logits, draft_hidden, calib_logit


def build_head(frozen_npz: Path, gate_init: float = 0.05) -> EagleHead:
    z = np.load(frozen_npz)
    return EagleHead(
        mx.array(z["token_embd"]),
        mx.array(z["lm_head"]),
        mx.array(z["output_norm"]),
        gate_init=gate_init,
    )


# ---------------------------------------------------------------------------
# Checkpoint I/O — flat npz, dict-walked
# ---------------------------------------------------------------------------
def _flat_params(d, prefix=""):
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else enumerate(d)):
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, (dict, list)):
            out.update(_flat_params(v, key))
        elif hasattr(v, "shape"):
            out[key] = v
    return out


def save_ckpt(head: EagleHead, path: Path, step: int = 0):
    flat = {k: np.array(v) for k, v in _flat_params(head.trainable_parameters()).items()}
    flat["__step__"] = np.int32(step)
    np.savez(path, **flat)


def load_ckpt(head: EagleHead, path: Path) -> int:
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


# ---------------------------------------------------------------------------
# Data loader (parquet → MLX batches)
# ---------------------------------------------------------------------------
def _iter_batches(shards: list[Path], batch_size: int, seq_len: int, epochs: int, seed: int = 0):
    """Yield (batch_size, seq_len)-shaped batches of contiguous-within-conversation rows."""
    rng = random.Random(seed)
    rows = []
    for s in shards:
        t = pq.read_table(s)
        for i in range(t.num_rows):
            rows.append({k: t[k][i].as_py() for k in t.column_names})
    print(f"[data] loaded {len(rows)} records from {len(shards)} shard(s)", flush=True)

    # Group by sample_id, sort by position, build seq_len-length sliding windows.
    rows.sort(key=lambda r: (r["sample_id"], r["position"]))
    windows: list[list[dict]] = []
    cur_sid = None
    cur: list[dict] = []
    for r in rows:
        if r["sample_id"] != cur_sid:
            cur_sid = r["sample_id"]
            cur = []
        cur.append(r)
        if len(cur) == seq_len:
            windows.append(cur)
            cur = []
    print(f"[data] {len(windows)} ordered windows of len {seq_len}", flush=True)

    for epoch in range(epochs):
        rng.shuffle(windows)
        for i in range(0, len(windows) - batch_size + 1, batch_size):
            batch = windows[i : i + batch_size]
            flat = [r for w in batch for r in w]  # row-major (B, S)
            prev = np.array([r["prev_token"] for r in flat], dtype=np.int32).reshape(batch_size, seq_len)
            nxt = np.array([r["next_token"] for r in flat], dtype=np.int32).reshape(batch_size, seq_len)

            def stack(field, dtype):
                return np.frombuffer(b"".join(r[field] for r in flat), dtype=dtype).reshape(
                    batch_size, seq_len, -1
                )

            yield {
                "prev": mx.array(prev),
                "next": mx.array(nxt),
                "low": mx.array(stack("hidden_low", np.float16)).astype(mx.float32),
                "mid": mx.array(stack("hidden_mid", np.float16)).astype(mx.float32),
                "high": mx.array(stack("hidden_high", np.float16)).astype(mx.float32),
                "shared": mx.array(stack("shared_hidden", np.float16)).astype(mx.float32),
                "mask": mx.array(
                    stack("routed_mask_per_layer", np.uint8)
                    .reshape(batch_size, seq_len, N_MOE_LAYERS, N_ROUTED)
                    .astype(np.float32)
                ),
                "epoch": epoch,
            }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(
    parquet_paths: list[Path],
    frozen: Path,
    ckpt_dir: Path,
    epochs: int = 1,
    batch_size: int = 32,
    seq_len: int = 16,
    lr: float = 3e-4,
    aux_weight: float = 0.5,
    mask_weight: float = 0.3,
    calib_weight: float = 0.1,
    multi_step_k: int = 1,
    multi_step_decay: float = 0.7,
    target_argmax_warmup_steps: int = 500,
    chain_h_high: bool = False,
    resume_ckpt: Path | None = None,
    multi_step_aux_decay: float = 1.0,
    gate_init: float = 0.05,
    gate_lr_multiplier: float = 1.0,
    k_curriculum: bool = False,
):
    """Train the head. Token CE linearly ramps from corpus tokens (α=0) to
    V2-Lite argmax (α=1) over `target_argmax_warmup_steps` — aligning the
    loss with the eval metric.

    multi_step_k>1 rolls the head's own argmax as the next prev_token for
    k passes. If `chain_h_high=True`, ALSO substitutes the head's own
    `draft_hidden` as the next pass's `h_high` input — the EAGLE-3-style
    autoregressive self-consumption regime that path-to-125 Eagle4 chain
    decode (K=4 verify) requires. Defaults to False for backward
    compatibility with the v3 K=1 training recipe.

    `resume_ckpt` loads weights from an .npz before training. Useful for
    warm-starting EAGLE-3-style retrains from the existing v3/best.npz
    instead of training from scratch.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    head = build_head(frozen, gate_init=gate_init)
    if resume_ckpt is not None:
        resume_step = load_ckpt(head, resume_ckpt)
        print(
            f"[train] resumed from {resume_ckpt} (step {resume_step})",
            flush=True,
        )
    else:
        print(
            f"[train] starting from scratch; residual_gate init = {gate_init}",
            flush=True,
        )
    print(
        f"[train] head built (aux={aux_weight} mask={mask_weight} calib={calib_weight} "
        f"k={multi_step_k} decay={multi_step_decay} chain_h_high={chain_h_high})",
        flush=True,
    )

    def _step_loss(head, b, prev_tok, h_high, weight: float,
                   target_alpha: float, k_offset: int,
                   step_aux_weight: float):
        """One step of the multi-step training loss.

        `k_offset` is the autoregressive depth: at step 0 we predict the
        token at position P+1 from V2-Lite state at P (the original
        single-step regime); at step k we predict the token at P+k+1
        from rolled state at P (matches chain-decode inference where
        the head has been rolled k times forward of the verifier
        capture). When `k_offset > 0`, targets b["next"], b["mask"],
        and the b["high"] MSE baseline are all shifted left by
        k_offset along the seq dim (with zero-padding at the tail
        that the position_mask excludes from the loss).
        """
        tok, mask_logits, draft_h, calib_logit = head(
            prev_tok, b["low"], b["mid"], h_high, b["shared"]
        )
        B, S, V = tok.shape

        # Position mask: valid positions are [3, S - k_offset). Skipping the
        # first 3 positions matches the original loss (warm-up of attention);
        # excluding the last k_offset positions avoids targets that would
        # shift past the end of the batch sequence.
        valid_end = max(S - k_offset, 3)
        if valid_end <= 3:
            return mx.zeros(()), tok, draft_h
        pos_mask = mx.concatenate([
            mx.zeros((B, 3)),
            mx.ones((B, valid_end - 3)),
            mx.zeros((B, S - valid_end)),
        ], axis=1).reshape(-1)
        N = mx.maximum(pos_mask.sum(), mx.array(1.0))

        # Shift targets / baseline / mask by k_offset along seq dim.
        if k_offset == 0:
            next_shifted = b["next"]
            high_shifted = b["high"]
            mask_shifted = b["mask"]
        else:
            H = b["high"].shape[-1]
            ML, MR = b["mask"].shape[-2], b["mask"].shape[-1]
            pad_next = mx.zeros((B, k_offset), dtype=b["next"].dtype)
            pad_high = mx.zeros((B, k_offset, H), dtype=b["high"].dtype)
            pad_mask = mx.zeros((B, k_offset, ML, MR), dtype=b["mask"].dtype)
            next_shifted = mx.concatenate([b["next"][:, k_offset:], pad_next], axis=1)
            high_shifted = mx.concatenate([b["high"][:, k_offset:, :], pad_high], axis=1)
            mask_shifted = mx.concatenate([b["mask"][:, k_offset:, :, :], pad_mask], axis=1)

        # MSE / target-argmax target tracks the SHIFTED V2-Lite h_high
        # (= the actual h_high at the position the head is predicting for).
        baseline = mx.fast.rms_norm(high_shifted, head._output_norm, RMS_EPS)
        target_logits = baseline @ head._lm_head.astype(mx.float32)
        target_arg_flat = mx.stop_gradient(mx.argmax(target_logits.reshape(-1, V), axis=-1))

        ce_corpus_per = nn.losses.cross_entropy(
            tok.reshape(-1, V), next_shifted.reshape(-1), reduction="none"
        )
        ce_target_per = nn.losses.cross_entropy(
            tok.reshape(-1, V), target_arg_flat, reduction="none"
        )
        ce_per = target_alpha * ce_target_per + (1.0 - target_alpha) * ce_corpus_per
        ce = (ce_per * pos_mask).sum() / N

        mse = (((draft_h - baseline) ** 2).mean(axis=-1).reshape(-1) * pos_mask).sum() / N

        bce_per = nn.losses.binary_cross_entropy(
            mask_logits, mask_shifted, with_logits=True, reduction="none"
        ).mean(axis=(-1, -2)).reshape(-1)
        bce = (bce_per * pos_mask).sum() / N

        head_arg = mx.argmax(tok.reshape(-1, V), axis=-1)
        accept_target = (head_arg == target_arg_flat).astype(mx.float32)
        calib_per = nn.losses.binary_cross_entropy(
            calib_logit.reshape(-1), accept_target, with_logits=True, reduction="none"
        )
        calib = (calib_per * pos_mask).sum() / N

        return weight * (ce + step_aux_weight * mse + mask_weight * bce + calib_weight * calib), tok, draft_h

    def loss_fn(head, b, target_alpha, active_k):
        # path-to-125 EAGLE-3-style chain training:
        #   - roll prev_token (existing multi_step_k behavior)
        #   - roll draft_hidden → h_high              (gated by chain_h_high)
        #   - shift targets by k_offset per step      (gated by chain_h_high)
        # Combined, this matches the inference-time chain-decode loop:
        # at step k the head sees a rolled prev_token + its own previous
        # draft_hidden as h_high, and is asked to predict the corpus
        # token at position P+k+1 (i.e., truly k tokens ahead of the
        # verifier capture). When chain_h_high=False, behaviour
        # collapses to the historical multi_step_k regime (no roll,
        # same target every k).
        #
        # path-to-125 efficiency patch — `active_k` lets the caller pass
        # a step-dependent K (curriculum learning over the chain depth).
        # When k_curriculum is off this is just multi_step_k each step.
        prev = b["prev"]
        cur_h_high = b["high"]
        total = mx.zeros(())
        for k in range(active_k):
            w = multi_step_decay ** k
            k_offset = k if chain_h_high else 0
            # path-to-125 aux-decay: at chain step k>0, scale the MSE
            # auxiliary by multi_step_aux_decay**k. Default 1.0 keeps
            # the historical regime (aux at every step). Values <1
            # taper aux toward later chain steps, giving the
            # residual_gate room to grow so the chain block actually
            # contributes to predictions — see commit message for
            # the gate-plateau diagnosis.
            step_aux_w = aux_weight * (multi_step_aux_decay ** k)
            step_loss, tok, draft_h = _step_loss(
                head, b, prev, cur_h_high, w, target_alpha, k_offset, step_aux_w,
            )
            total = total + step_loss
            if k + 1 < active_k:
                prev = mx.stop_gradient(mx.argmax(tok, axis=-1))
                if chain_h_high:
                    cur_h_high = mx.stop_gradient(draft_h)
        return total

    grad_fn = nn.value_and_grad(head, loss_fn)
    opt = mxoptim.AdamW(learning_rate=lr, weight_decay=0.01)

    # path-to-125 efficiency patch — K-curriculum: ramp active_k from 1 to
    # multi_step_k linearly over `target_argmax_warmup_steps`. Lets the head
    # learn K=1 (matching V2-Lite argmax) before being asked to predict
    # deep chain rollouts. Closer to standard curriculum learning.
    def _active_k_for_step(s: int) -> int:
        if not k_curriculum or multi_step_k <= 1:
            return multi_step_k
        ramp = min(s / max(target_argmax_warmup_steps, 1), 1.0)
        return max(1, min(multi_step_k, int(round(1 + ramp * (multi_step_k - 1)))))

    log = (ckpt_dir / "log.jsonl").open("w")
    step = 0
    t0 = time.time()
    for batch in _iter_batches(parquet_paths, batch_size, seq_len, epochs):
        target_alpha = min(step / max(target_argmax_warmup_steps, 1), 1.0)
        active_k = _active_k_for_step(step)
        loss, grads = grad_fn(head, batch, target_alpha, active_k)

        # path-to-125 efficiency patch — gate-LR multiplier. The
        # residual_gate's per-step gradient is ~(∂loss/∂draft_h) ⋅
        # block_output; block_output magnitudes are tiny so the gate
        # gets a much smaller effective gradient than other params at
        # the shared LR. Scaling its grad here gives it an effective
        # per-step learning rate of `lr × gate_lr_multiplier` and lets
        # it move freely during the α ramp instead of drifting toward
        # zero by default.
        if gate_lr_multiplier != 1.0 and "residual_gate" in grads:
            grads["residual_gate"] = grads["residual_gate"] * gate_lr_multiplier

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
                "active_k": active_k,
                "wall": time.time() - t0,
            }
            print(
                f"step={step} epoch={batch['epoch']} loss={row['loss']:.3f} "
                f"gate={row['gate']:.3f} α={target_alpha:.2f} k={active_k} "
                f"wall={row['wall']:.1f}s",
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


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def evaluate(
    ckpt: Path,
    frozen: Path,
    parquet_paths: list[Path],
    max_records: int = 5000,
    mask_top_k: int = 8,
    dump_logits: Path | None = None,
) -> dict:
    head = build_head(frozen)
    load_ckpt(head, ckpt)

    n = n_top1_target = n_top1_corpus = 0
    per_layer_topk = np.zeros(N_MOE_LAYERS, dtype=np.int64)

    # When dump_logits is set, accumulate per-record arrays across all
    # shards (truncated to max_records) and save as a single NPZ at the
    # end. Shapes (after concat):
    #   token_logits  (N, VOCAB)         float32   head output
    #   mask_logits   (N, N_MOE, N_ROUTED) float32 head output
    #   calib_logit   (N,)               float32   head output
    #   draft_hidden  (N, HIDDEN_DIM)    float32   head output
    #   prev_token    (N,)               int32     head input
    #   next_token    (N,)               int32     ground truth
    #   h_low         (N, HIDDEN_DIM)    float32   head input
    #   h_mid         (N, HIDDEN_DIM)    float32   head input
    #   h_high        (N, HIDDEN_DIM)    float32   head input
    #   h_shared     (N, HIDDEN_DIM)     float32   head input
    # Consumed by `crates/dismantle-core/tests/eagle4_parity.rs` for the
    # cross-language Rust-vs-Python parity diff (path-to-90 steps 5-6).
    # The inputs (h_low/mid/high/shared) are dumped so Rust can feed
    # identical inputs to its Eagle4Head::forward_full without needing
    # a parquet reader.
    dump_token_logits: list[np.ndarray] = []
    dump_mask_logits: list[np.ndarray] = []
    dump_calib_logit: list[np.ndarray] = []
    dump_draft_hidden: list[np.ndarray] = []
    dump_prev_token: list[np.ndarray] = []
    dump_next_token: list[np.ndarray] = []
    dump_h_low: list[np.ndarray] = []
    dump_h_mid: list[np.ndarray] = []
    dump_h_high: list[np.ndarray] = []
    dump_h_shared: list[np.ndarray] = []

    for shard in parquet_paths:
        t = pq.read_table(shard)
        take = min(max_records - n, t.num_rows)
        if take <= 0:
            break

        def stack(field, dtype):
            buf = b"".join(t[field][i].as_py() for i in range(take))
            return np.frombuffer(buf, dtype=dtype).reshape(take, -1)

        prev = np.array([t["prev_token"][i].as_py() for i in range(take)], dtype=np.int32)
        nxt = np.array([t["next_token"][i].as_py() for i in range(take)], dtype=np.int32)
        low = stack("hidden_low", np.float16)
        mid = stack("hidden_mid", np.float16)
        hi = stack("hidden_high", np.float16)
        sh = stack("shared_hidden", np.float16)
        true_mask = stack("routed_mask_per_layer", np.uint8).reshape(take, N_MOE_LAYERS, N_ROUTED)

        ph = mx.array(prev).reshape(1, take)
        lo_a = mx.array(low).astype(mx.float32).reshape(1, take, HIDDEN_DIM)
        md_a = mx.array(mid).astype(mx.float32).reshape(1, take, HIDDEN_DIM)
        hi_a = mx.array(hi).astype(mx.float32).reshape(1, take, HIDDEN_DIM)
        sh_a = mx.array(sh).astype(mx.float32).reshape(1, take, HIDDEN_DIM)

        tok, mask_logits, draft_hidden, calib_logit = head(ph, lo_a, md_a, hi_a, sh_a)
        baseline = mx.fast.rms_norm(hi_a.reshape(take, HIDDEN_DIM), head._output_norm, RMS_EPS)
        target_logits = baseline @ head._lm_head.astype(mx.float32)
        mx.eval(tok, mask_logits, draft_hidden, target_logits, calib_logit)

        head_arg = np.argmax(np.array(tok).reshape(take, -1), axis=-1)
        target_arg = np.array(mx.argmax(target_logits, axis=-1))
        n_top1_target += int((head_arg == target_arg).sum())
        n_top1_corpus += int((head_arg == nxt).sum())

        mask_np = np.array(mask_logits).reshape(take, N_MOE_LAYERS, N_ROUTED)
        for L in range(N_MOE_LAYERS):
            top = np.argpartition(-mask_np[:, L, :], kth=mask_top_k, axis=-1)[:, :mask_top_k]
            for ti in range(take):
                if np.intersect1d(top[ti], np.where(true_mask[ti, L] > 0)[0]).size >= 6:
                    per_layer_topk[L] += 1

        if dump_logits is not None:
            # Reshape: drop the leading batch=1 dim so first axis = record.
            tok_np = np.array(tok).reshape(take, -1).astype(np.float32, copy=False)
            calib_np = np.array(calib_logit).reshape(take).astype(np.float32, copy=False)
            dh_np = np.array(draft_hidden).reshape(take, HIDDEN_DIM).astype(np.float32, copy=False)
            dump_token_logits.append(tok_np)
            dump_mask_logits.append(mask_np.astype(np.float32, copy=False))
            dump_calib_logit.append(calib_np)
            dump_draft_hidden.append(dh_np)
            dump_prev_token.append(prev)
            dump_next_token.append(nxt)
            # Input hiddens — parquet stores fp16; we promote to fp32
            # here so Rust gets a single dtype to read.
            dump_h_low.append(low.astype(np.float32, copy=False))
            dump_h_mid.append(mid.astype(np.float32, copy=False))
            dump_h_high.append(hi.astype(np.float32, copy=False))
            dump_h_shared.append(sh.astype(np.float32, copy=False))

        n += take
        if n >= max_records:
            break

    if dump_logits is not None:
        dump_logits.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dump_logits,
            token_logits=np.concatenate(dump_token_logits, axis=0),
            mask_logits=np.concatenate(dump_mask_logits, axis=0),
            calib_logit=np.concatenate(dump_calib_logit, axis=0),
            draft_hidden=np.concatenate(dump_draft_hidden, axis=0),
            prev_token=np.concatenate(dump_prev_token, axis=0),
            next_token=np.concatenate(dump_next_token, axis=0),
            h_low=np.concatenate(dump_h_low, axis=0),
            h_mid=np.concatenate(dump_h_mid, axis=0),
            h_high=np.concatenate(dump_h_high, axis=0),
            h_shared=np.concatenate(dump_h_shared, axis=0),
        )

    n = max(n, 1)
    return {
        "scored": n,
        "top1_vs_target": n_top1_target / n,
        "top1_vs_corpus": n_top1_corpus / n,
        "mask_top_k": mask_top_k,
        "mask_topk_per_layer_recall": (per_layer_topk / n).tolist(),
        "mask_topk_mean_recall": float((per_layer_topk / n).mean()),
        "dump_logits": str(dump_logits) if dump_logits is not None else None,
    }


# ---------------------------------------------------------------------------
# Quantize: head .npz → 4-bit (group_size=64), matching V2-Lite's Q4_K_M
# ---------------------------------------------------------------------------
def quantize_head(ckpt_in: Path, ckpt_out: Path, bits: int = 4, group_size: int = 64) -> dict:
    z = np.load(ckpt_in, allow_pickle=False)
    out = {}
    quantized = []
    for k in z.files:
        a = z[k]
        if (
            a.ndim == 2
            and a.dtype in (np.float32, np.float16)
            and "weight" in k
            and a.shape[-1] % group_size == 0
        ):
            qw, scales, biases = mx.quantize(mx.array(a), group_size=group_size, bits=bits)
            mx.eval(qw, scales, biases)
            out[k] = np.array(qw)
            out[k + ".scales"] = np.array(scales)
            out[k + ".biases"] = np.array(biases)
            quantized.append(k)
        else:
            out[k] = a
    out["__quant_bits__"] = np.int32(bits)
    out["__quant_group_size__"] = np.int32(group_size)
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt_out, **out)
    return {"n_quantized": len(quantized), "out": str(ckpt_out)}


# ---------------------------------------------------------------------------
# Frozen-weight extraction (V2-Lite → .npz)
# ---------------------------------------------------------------------------
def extract_frozen(model_id: str, out_path: Path, verify: bool = True) -> None:
    from mlx_lm.utils import load
    from mlx_lm.models.base import create_attention_mask

    print(f"[frozen] loading {model_id}", flush=True)
    model, tok = load(model_id)

    def dq(layer):
        if hasattr(layer, "scales"):
            return mx.dequantize(layer.weight, layer.scales, layer.biases, layer.group_size, layer.bits)
        return layer.weight

    lm = dq(model.lm_head)
    emb = dq(model.model.embed_tokens)
    norm = model.model.norm.weight

    if verify:
        ids = mx.array([tok.encode("Hello")])
        h = model.model.embed_tokens(ids)
        m = create_attention_mask(h, None)
        for layer in model.model.pipeline_layers:
            h = layer(h, m, cache=None)
        h = model.model.norm(h)
        native = mx.argmax(model.lm_head(h)[0, -1])
        ours = mx.argmax(h[0, -1] @ mx.transpose(lm.astype(mx.float32), (1, 0)))
        mx.eval(native, ours)
        assert int(native) == int(ours), f"frozen-weight extraction mismatch: native={int(native)} ours={int(ours)}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        token_embd=np.array(mx.transpose(emb.astype(mx.float16), (1, 0))),
        lm_head=np.array(mx.transpose(lm.astype(mx.float16), (1, 0))),
        output_norm=np.array(norm.astype(mx.float32)),
    )
    print(f"[frozen] saved → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(prog="eagle4")
    sub = p.add_subparsers(dest="cmd", required=True)

    fp = sub.add_parser("frozen", help="extract V2-Lite frozen weights → .npz")
    fp.add_argument("--model", default="mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx")
    fp.add_argument("--out", type=Path, default=Path("v2lite_frozen.npz"))

    tp = sub.add_parser("train", help="train head on captured per-layer parquet")
    tp.add_argument("--parquet", nargs="+", type=Path, required=True)
    tp.add_argument("--frozen", type=Path, required=True)
    tp.add_argument("--ckpt-dir", type=Path, required=True)
    tp.add_argument("--epochs", type=int, default=1)
    tp.add_argument("--batch-size", type=int, default=32)
    tp.add_argument("--seq-len", type=int, default=16)
    tp.add_argument("--lr", type=float, default=3e-4)
    tp.add_argument("--aux-weight", type=float, default=0.5)
    tp.add_argument("--mask-weight", type=float, default=0.3)
    tp.add_argument("--calib-weight", type=float, default=0.1)
    tp.add_argument("--multi-step-k", type=int, default=1)
    tp.add_argument("--multi-step-decay", type=float, default=0.7)
    tp.add_argument("--target-warmup-steps", type=int, default=500,
                    help="steps over which token CE ramps from corpus → V2-Lite argmax")
    tp.add_argument("--chain-h-high", action="store_true",
                    help="path-to-125 EAGLE-3-style: roll draft_hidden as next-step "
                         "h_high through multi_step_k passes. Required for chain spec decode.")
    tp.add_argument("--resume", type=Path, default=None,
                    help="warm-start from an existing .npz checkpoint")
    tp.add_argument("--gate-init", type=float, default=0.05,
                    help="path-to-125 L8 — initial value for EagleHead.residual_gate. "
                         "Default 0.05 (v3 init). Path-to-125 closeout § Branch 3 "
                         "showed v3-warm-started training plateaus chain accept at ~7%% "
                         "because the gate stays at ~0.001; from-scratch training "
                         "with a larger gate_init (try 0.1) forces the head to learn "
                         "to use the block output. Ignored on --resume (the resumed "
                         "checkpoint's gate value overrides).")
    tp.add_argument("--gate-lr-multiplier", type=float, default=1.0,
                    help="path-to-125 L8-eff — scale residual_gate's gradient by this "
                         "factor per step. Effective LR for the gate becomes "
                         "`lr × gate_lr_multiplier`. Use ~10.0 to give the gate "
                         "enough signal to survive the α ramp instead of decaying "
                         "to ~0.001. Cheapest targeted attack on gate-collapse.")
    tp.add_argument("--k-curriculum", action="store_true",
                    help="path-to-125 L8-eff — ramp active chain depth K from 1 to "
                         "`--multi-step-k` linearly over `--target-warmup-steps`. "
                         "Lets the head master K=1 prediction first before being "
                         "asked to predict deep rollouts. Matches standard curriculum "
                         "learning and avoids early-training waste on impossibly-hard "
                         "K=4 chain CE losses.")
    tp.add_argument("--multi-step-aux-decay", type=float, default=1.0,
                    help="path-to-125: per-step decay applied to aux MSE weight at "
                         "chain step k (aux *= decay**k). <1.0 tapers MSE toward later "
                         "chain steps so residual_gate can grow.")

    ep = sub.add_parser("eval", help="acceptance + per-layer mask recall")
    ep.add_argument("--ckpt", type=Path, required=True)
    ep.add_argument("--frozen", type=Path, required=True)
    ep.add_argument("--parquet", nargs="+", type=Path, required=True)
    ep.add_argument("--max-records", type=int, default=5000)
    ep.add_argument("--mask-top-k", type=int, default=8)
    ep.add_argument(
        "--dump-logits",
        type=Path,
        default=None,
        help="If set, write per-record (token_logits, mask_logits, "
        "calib_logit, draft_hidden, prev_token, next_token) as an NPZ "
        "at this path. Consumed by dismantle's eagle4 parity test.",
    )

    qp = sub.add_parser("quantize", help="Q4_K_M-like 4-bit head")
    qp.add_argument("--in", dest="ckpt_in", type=Path, required=True)
    qp.add_argument("--out", dest="ckpt_out", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "frozen":
        extract_frozen(args.model, args.out)
    elif args.cmd == "train":
        train(
            args.parquet, args.frozen, args.ckpt_dir,
            epochs=args.epochs, batch_size=args.batch_size, seq_len=args.seq_len,
            lr=args.lr, aux_weight=args.aux_weight, mask_weight=args.mask_weight,
            calib_weight=args.calib_weight,
            multi_step_k=args.multi_step_k, multi_step_decay=args.multi_step_decay,
            target_argmax_warmup_steps=args.target_warmup_steps,
            chain_h_high=args.chain_h_high,
            resume_ckpt=args.resume,
            multi_step_aux_decay=args.multi_step_aux_decay,
            gate_init=args.gate_init,
            gate_lr_multiplier=args.gate_lr_multiplier,
            k_curriculum=args.k_curriculum,
        )
    elif args.cmd == "eval":
        r = evaluate(
            args.ckpt,
            args.frozen,
            args.parquet,
            args.max_records,
            args.mask_top_k,
            dump_logits=args.dump_logits,
        )
        print(json.dumps({k: v for k, v in r.items() if k != "mask_topk_per_layer_recall"}, indent=2))
        for i, v in enumerate(r["mask_topk_per_layer_recall"]):
            print(f"  layer {i+1:>2}: {v*100:5.1f}%")
    elif args.cmd == "quantize":
        print(json.dumps(quantize_head(args.ckpt_in, args.ckpt_out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
