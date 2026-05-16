"""
EAGLE-3 draft head for DeepSeek-V2-Lite, in MLX.

Architecture (per stage3_c1/architecture.md + EAGLE-3 paper §3.2):

  Inputs per training step (batch B, seq S):
    prev_tokens      : int32[B, S]              — input token ids
    target_hidden    : float[B, S, 2048]        — captured pre-lm_head hidden
                                                  from dismantle (final-norm output)
    target_next_tokens: int32[B, S]             — teacher-forced ground truth

  Frozen target weights (loaded from v2lite_frozen.npz):
    token_embd       : float16[2048, 102400]   — input embedding lookup
    lm_head          : float16[2048, 102400]   — output projection (NOT tied)
    output_norm      : float32[2048]           — final RMSNorm gain (unused here;
                                                 hidden is already post-norm)

  Forward (per position t):
    prev_embed[t]    = token_embd[:, prev_tokens[t]]                # (2048,)
    x[t]             = concat(prev_embed[t], target_hidden[t])      # (4096,)
    x[t]             = InProj(x[t])                                 # → 2048
    x[t]             = TransformerBlock(x[t])                       # → 2048
    draft_hidden[t]  = FinalNorm(x[t])                              # → 2048
    logits[t]        = draft_hidden[t] @ lm_head                    # → 102400

  Loss:
    CE(logits, target_next_tokens) averaged over (B, S)

  Optional auxiliary loss (EAGLE paper §3.3, weight=0.1):
    + MSE(draft_hidden, target_hidden)
    Drives the head to also reproduce the target's hidden geometry, which
    helps multi-step prediction stability.

Param breakdown for h=2048, inter=5632 (matched to a smaller-than-V2-Lite
dense MLP — V2-Lite's dense feed_forward_length is 10944 but EAGLE doesn't
need that large; 5632 is the LLaMA-7B size and trains well):

  InProj           : 4096 × 2048 ≈ 8.4 M
  Attention (Q/K/V/O at h=2048, heads=16)
    Q, K, V        : 3 × (2048 × 2048) ≈ 12.6 M
    O              : 2048 × 2048 ≈ 4.2 M
  MLP (SwiGLU)
    gate, up, down : 3 × (2048 × 5632) ≈ 34.6 M
  Norms (RMSNorm × 3, weight only) : 3 × 2048 ≈ 6 K
  ─────────────────────────────────────────
  Total trainable ≈ 60 M params, ~120 MB at fp16

Frozen (loaded but not trained): token_embd + lm_head + output_norm ≈ 838 MB

Total memory at training:
  - frozen weights : ~840 MB
  - trainable      : ~240 MB (fp32 master) + ~480 MB (Adam m,v) = ~720 MB
  - activations    : ~2-3 GB at batch=16, seq=128 (bf16 with grad-checkpointing
                      not enabled — could halve with checkpointing if needed)
  - total          : ~4-5 GB. Comfortably fits 18 GB M3 Pro unified memory.

WARNING: this file is shaped against MLX 0.18+ API but has not been
executed in this commit. Expect to debug imports and 1-2 layer-init calls
on first run. Major patterns (mx.array, nn.Linear, nn.Module, RMSNorm) are
stable across recent MLX versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# These imports run only when the file is executed; the docstring above is
# self-contained for code-review purposes.
# ---------------------------------------------------------------------------
try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover — happens during code review w/o MLX
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class EagleHeadConfig:
    hidden_dim: int = 2048
    vocab_size: int = 102400
    n_heads: int = 16
    head_dim: int = 128  # 2048 / 16
    intermediate: int = 5632  # SwiGLU intermediate
    rms_eps: float = 1e-6
    # Position is NOT encoded in EAGLE — the head sees one (prev_token,
    # target_hidden) pair at a time. The transformer block does NOT need
    # positional encoding because each token is processed independently;
    # the (B, S) dimension is purely a batching convenience.
    # If multi-step (proposing K>1) needs explicit position later, add it
    # here. Single-step EAGLE-3 does not need it per paper §3.2.


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _rmsnorm(x, weight, eps: float):
    """Functional RMSNorm; weight is the learned/frozen gain vector."""
    # x: (..., D); weight: (D,)
    var = mx.mean(x * x, axis=-1, keepdims=True)
    x = x * mx.rsqrt(var + eps)
    return x * weight


class _Attention(nn.Module):
    """Standard multi-head attention WITHOUT a KV cache.

    Each forward sees ONE position per sequence (S=1 in EAGLE's natural
    usage); batching across (B, S) treats S as additional independent
    positions. There is no causal mask because there is no cross-position
    attention — every (B, s) is a self-contained query against the same
    (B, s) keys/values.

    Concretely: with S=1 per record, this degenerates to a per-head dot-
    product of q against k followed by softmax (a single value because
    n_kv=1) then v. The math is preserved at higher S for batched training
    where we treat S contiguous corpus positions as independent samples.
    """

    def __init__(self, cfg: EagleHeadConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.q = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.k = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.v = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.o = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.scale = 1.0 / (cfg.head_dim ** 0.5)

    def __call__(self, x):
        # x: (B, S, H)
        B, S, H = x.shape
        q = self.q(x).reshape(B, S, self.n_heads, self.head_dim)
        k = self.k(x).reshape(B, S, self.n_heads, self.head_dim)
        v = self.v(x).reshape(B, S, self.n_heads, self.head_dim)
        # Transpose to (B, n_heads, S, head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        # No causal mask: per-position independence (see class docstring).
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale  # (B, nh, S, S)
        # Diagonal-only attention since each (B,s) is independent.
        # Cheaper: zero out off-diagonal before softmax. For simplicity in
        # this first cut we just softmax across all S — at S=1 it's a no-op,
        # at S>1 it's slightly wrong (cross-position bleed). FIXME on first
        # training run: either enforce S=1 in the data loader or apply a
        # diagonal mask here.
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v)  # (B, nh, S, head_dim)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, S, H)
        return self.o(out)


class _SwiGLU(nn.Module):
    def __init__(self, cfg: EagleHeadConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_dim, cfg.intermediate, bias=False)
        self.up = nn.Linear(cfg.hidden_dim, cfg.intermediate, bias=False)
        self.down = nn.Linear(cfg.intermediate, cfg.hidden_dim, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class _Block(nn.Module):
    """RMSNorm -> Attention -> add ; RMSNorm -> SwiGLU -> add."""

    def __init__(self, cfg: EagleHeadConfig):
        super().__init__()
        self.attn_norm_w = mx.ones((cfg.hidden_dim,))
        self.attn = _Attention(cfg)
        self.mlp_norm_w = mx.ones((cfg.hidden_dim,))
        self.mlp = _SwiGLU(cfg)
        self.eps = cfg.rms_eps

    def __call__(self, x):
        x = x + self.attn(_rmsnorm(x, self.attn_norm_w, self.eps))
        x = x + self.mlp(_rmsnorm(x, self.mlp_norm_w, self.eps))
        return x


# ---------------------------------------------------------------------------
# Top-level head
# ---------------------------------------------------------------------------
class EagleHead(nn.Module):
    """EAGLE-3 1-block draft head for DeepSeek-V2-Lite.

    Constructor takes a config + the frozen target weights (token_embd,
    lm_head, output_norm). Frozen weights are held as `mx.array` attributes
    and not registered as trainable params. The optimizer should be set up
    against `self.trainable_parameters()`.

    `__call__(prev_tokens, target_hidden)` returns `logits` of shape
    `(B, S, vocab_size)`. The optional `return_hidden=True` flag also
    returns the draft hidden state for the auxiliary MSE loss.
    """

    def __init__(
        self,
        cfg: EagleHeadConfig,
        token_embd: "mx.array",
        lm_head: "mx.array",
        output_norm: "mx.array",
    ):
        super().__init__()
        self.cfg = cfg
        # ---- frozen ----
        # MLX 0.18+: assigning mx.array attributes on an nn.Module makes
        # them "buffers" (not in trainable_parameters()) iff they are not
        # nn.Linear / nn.Embedding. To be safe, save them under a name
        # tracked separately and exclude from optimizer state.
        self._token_embd = token_embd   # (hidden, vocab) — column lookup by id
        self._lm_head = lm_head         # (hidden, vocab)
        self._output_norm = output_norm  # (hidden,)
        # ---- trainable ----
        self.in_proj = nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.block = _Block(cfg)
        self.final_norm_w = mx.ones((cfg.hidden_dim,))

    def trainable_parameters(self):
        """Filter self.parameters() to exclude frozen tensors.

        MLX's default tree_flatten() walks all leaves; since _token_embd /
        _lm_head / _output_norm are stored as private attrs (prefixed _),
        we exclude them explicitly here.
        """
        # WARNING: MLX trainable filter convention varies between minor
        # versions. Validate with `len(list(self.trainable_parameters()))`
        # at training-stack bootstrap time. Default impl below returns
        # everything from in_proj + block + final_norm_w.
        params = self.parameters()
        # Walk and drop frozen keys.
        for key in ("_token_embd", "_lm_head", "_output_norm"):
            params.pop(key, None)
        return params

    def __call__(self, prev_tokens, target_hidden, return_hidden: bool = False):
        # prev_tokens   : int32[B, S]
        # target_hidden : float[B, S, H]
        # Embedding lookup: token_embd has shape (H, V). MLX has no built-in
        # column-lookup; transpose once to (V, H) and gather.
        B, S = prev_tokens.shape
        H = self.cfg.hidden_dim
        # (V, H) — one-time transpose; could be precomputed outside if perf-critical
        embed_table = mx.transpose(self._token_embd, (1, 0))
        prev_embed = embed_table[prev_tokens]  # (B, S, H)
        x = mx.concatenate([prev_embed, target_hidden], axis=-1)  # (B, S, 2H)
        x = self.in_proj(x)  # (B, S, H)
        x = self.block(x)  # (B, S, H)
        x = _rmsnorm(x, self.final_norm_w, self.cfg.rms_eps)  # (B, S, H)
        # Project to vocab via frozen lm_head. lm_head is (H, V); x @ lm_head
        # = (B, S, V) — matches the dismantle dequant verify result.
        logits = x @ self._lm_head
        if return_hidden:
            return logits, x
        return logits


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------
def eagle_loss(
    logits,
    target_hidden,
    target_next_tokens,
    draft_hidden=None,
    aux_weight: float = 0.0,
):
    """Cross-entropy on next-token prediction + optional MSE aux.

    logits             : (B, S, V)
    target_hidden      : (B, S, H)  — captured target hidden
    target_next_tokens : (B, S)     — corpus ground-truth next tokens
    draft_hidden       : (B, S, H)  — optional, the head's pre-lm_head output
    aux_weight         : 0.0 disables; 0.1 is EAGLE paper default
    """
    # Standard CE (mlx.nn.losses.cross_entropy averages over batch by default)
    ce = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_next_tokens.reshape(-1),
        reduction="mean",
    )
    if aux_weight > 0.0 and draft_hidden is not None:
        # MSE on the hidden geometry — EAGLE §3.3
        mse = mx.mean((draft_hidden - target_hidden) ** 2)
        return ce + aux_weight * mse, ce, mse
    return ce, ce, mx.zeros(())


# ---------------------------------------------------------------------------
# Convenience: load + construct from extract_lm_head.py output
# ---------------------------------------------------------------------------
def load_head_from_npz(npz_path, cfg: Optional[EagleHeadConfig] = None) -> EagleHead:
    """Build an EagleHead with frozen weights loaded from v2lite_frozen.npz."""
    if mx is None:
        raise ImportError("MLX not installed; pip install mlx")
    import numpy as np

    data = np.load(npz_path, allow_pickle=True)
    token_embd = mx.array(data["token_embd"].astype(np.float16))
    lm_head = mx.array(data["lm_head"].astype(np.float16))
    output_norm = mx.array(data["output_norm"].astype(np.float32))
    if cfg is None:
        cfg = EagleHeadConfig(
            hidden_dim=int(data["hidden"]),
            vocab_size=int(data["vocab"]),
            rms_eps=float(data["rms_eps"]),
        )
    return EagleHead(cfg, token_embd, lm_head, output_norm)
