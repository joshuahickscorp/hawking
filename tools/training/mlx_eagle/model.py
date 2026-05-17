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
    # Attention mode (see top of this module). "independent" is the safe
    # default for single-token training where each (B, s) position is an
    # independent example. Switch to "causal" for MTP, or
    # "block_diagonal" for packed-MTP.
    attention_mode: str = "independent"
    # Number of next-token prediction heads. 1 = standard EAGLE-3.
    # k > 1 = MTP — the trunk processes one input, then k independent
    # output heads predict tokens at positions p+1 .. p+k respectively.
    # Pairs with attention_mode="causal" + data loader emitting k-token
    # input sequences.
    n_predict_steps: int = 1


# ---------------------------------------------------------------------------
# Attention mode
# ---------------------------------------------------------------------------
ATTENTION_MODE_INDEPENDENT = "independent"
"""Each (B, s) position is self-contained — diagonal-only attention.

Use for single-token training where (prev_token, target_hidden) tuples in
the same batch row are unrelated. The implementation uses a diagonal mask
so attention math is correct (each position attends only to itself), which
is equivalent to S=1, B=B*S in compute terms but lets us keep the (B, S)
shape stable for compile.
"""

ATTENTION_MODE_CAUSAL = "causal"
"""Standard causal mask — position s attends to positions 0..s.

Use for multi-token prediction (MTP) training where each batch row is a
length-S sub-sequence of consecutive corpus positions. The head learns to
predict k tokens ahead from one forward, sharing the backbone.
"""

ATTENTION_MODE_BLOCK_DIAGONAL = "block_diagonal"
"""Block-diagonal causal mask — pack multiple length-K sub-sequences per row.

Use for sequence packing on top of MTP: batch row stacks N sub-sequences
of length K each (S = N*K) with a block-diagonal causal mask so cross-
sub-sequence positions don't attend to each other. Requires the data
loader to emit a `block_lengths` tensor alongside the inputs.
"""


def _build_mask(B: int, S: int, mode: str, block_lengths=None):
    """Return an attention bias `(B, 1, S, S)` of 0 and -inf that mx.fast.sdpa applies.

    None is acceptable for full self-attention (mx.fast.sdpa supports `mask="causal"`
    for the common case; we return None and let the caller pass mask="causal").
    """
    if mode == ATTENTION_MODE_INDEPENDENT:
        # Diagonal-only: -inf everywhere except the diagonal.
        eye = mx.eye(S)  # (S, S) with 1 on diagonal
        bias = mx.where(eye > 0, mx.zeros((S, S)), mx.full((S, S), -1e9))
        return mx.broadcast_to(bias[None, None, :, :], (B, 1, S, S))
    if mode == ATTENTION_MODE_CAUSAL:
        # Standard causal: 0 at and below diagonal, -inf above.
        tri = mx.tri(S, S, k=0)  # (S, S) lower-triangular with 1s
        bias = mx.where(tri > 0, mx.zeros((S, S)), mx.full((S, S), -1e9))
        return mx.broadcast_to(bias[None, None, :, :], (B, 1, S, S))
    if mode == ATTENTION_MODE_BLOCK_DIAGONAL:
        if block_lengths is None:
            raise ValueError("block_diagonal mode requires block_lengths")
        # Build per-batch block-diagonal causal masks. Slow-path Python
        # construction is acceptable because masks are cacheable per-batch-
        # shape (and packing typically uses fixed shapes).
        import numpy as np
        masks = np.full((B, S, S), -1e9, dtype=np.float32)
        for b in range(B):
            offset = 0
            for blen in block_lengths[b]:
                blen_i = int(blen)
                if blen_i <= 0:
                    continue
                for i in range(blen_i):
                    for j in range(i + 1):
                        masks[b, offset + i, offset + j] = 0.0
                offset += blen_i
        return mx.array(masks)[:, None, :, :]
    raise ValueError(f"unknown attention mode {mode!r}")


class _Attention(nn.Module):
    """Multi-head attention using mx.fast.scaled_dot_product_attention.

    The attention mask is passed at forward time via the `mask` kwarg so
    we can switch between independent / causal / block-diagonal without
    rebuilding the module. mx.fast.sdpa hits the fused Metal kernel and
    is 2-3x faster than the hand-rolled (Q@K^T)*scale + softmax + V path.
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

    def __call__(self, x, mask=None):
        # x: (B, S, H); mask: None | (B, 1, S, S) | "causal"
        B, S, H = x.shape
        # Project + reshape to (B, n_heads, S, head_dim).
        q = self.q(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        # Fused SDPA. mask=None → full self-attention. Pass tensor or "causal".
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, H)
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
    """RMSNorm -> Attention -> add ; RMSNorm -> SwiGLU -> add.

    Uses mx.fast.rms_norm (fused Metal kernel) — ~2x faster than the
    hand-rolled `x * rsqrt(mean(x*x) + eps) * weight` path.
    """

    def __init__(self, cfg: EagleHeadConfig):
        super().__init__()
        self.attn_norm_w = mx.ones((cfg.hidden_dim,))
        self.attn = _Attention(cfg)
        self.mlp_norm_w = mx.ones((cfg.hidden_dim,))
        self.mlp = _SwiGLU(cfg)
        self.eps = cfg.rms_eps

    def __call__(self, x, mask=None):
        x = x + self.attn(mx.fast.rms_norm(x, self.attn_norm_w, self.eps), mask=mask)
        x = x + self.mlp(mx.fast.rms_norm(x, self.mlp_norm_w, self.eps))
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
        # MTP per-head offsets (only when n_predict_steps > 1). Tiny but
        # essential to differentiate the k logit slices — without them all
        # heads predict the same distribution. No underscore prefix so MLX
        # treats them as proper trainable parameters (MLX's parameters()
        # walker skips underscore-prefixed attrs).
        if cfg.n_predict_steps > 1:
            for k in range(cfg.n_predict_steps):
                # init to zero so k=0 reproduces standard EAGLE-3 behavior.
                setattr(self, f"mtp_offset_{k}", mx.zeros((cfg.hidden_dim,)))

    def trainable_parameters(self):
        """Filter self.parameters() to exclude frozen tensors.

        Frozen: _token_embd, _lm_head, _output_norm — V2-Lite weights
        loaded from v2lite_frozen.npz and not updated.

        Trainable (returned): in_proj, block (attention + MLP), final_norm_w,
        and `_mtp_offset_*` when n_predict_steps > 1.
        """
        params = self.parameters()
        for key in ("_token_embd", "_lm_head", "_output_norm"):
            params.pop(key, None)
        return params

    def propose_tree(
        self,
        prev_token: int,
        target_hidden,  # mx.array of shape (hidden_dim,) — the committed position's hidden
        topology,  # list[int] e.g. [3, 2, 1, 1] = branching per depth (top-3, then top-2, …)
    ):
        """Tree-decoding tree proposer (Option A: naive — one head forward per node).

        See reports/path_to_90/tree_decode/design.md for the math.

        Returns a dict with:
          node_tokens   : list[int]                    — N predicted tokens
          node_parents  : list[int]                    — parent index per node (-1 for root)
          node_depths   : list[int]                    — depth in the tree (root depth = 0)
          node_paths    : list[list[int]]              — full token path from root
          node_hidden   : list[mx.array]               — head's draft hidden per node (for downstream)

        N = sum_d prod(topology[:d+1])  e.g. [3,2,1,1] → 3 + 6 + 6 + 6 = 21 nodes.

        Limitations called out in the design doc:
          - One head forward per node — at large N (> ~16) becomes
            non-trivial fraction of the target forward cost
          - Uses target_hidden as the seed for the root; subsequent nodes
            use the head's own predicted hidden as the seed (this is the
            standard EAGLE-3 trick, but acceptance compounds with depth)
        """
        if mx is None:
            raise ImportError("MLX not installed — pip install mlx")
        if not topology:
            return {
                "node_tokens": [], "node_parents": [], "node_depths": [],
                "node_paths": [], "node_hidden": [],
            }

        H = self.cfg.hidden_dim
        # Root: run head on (prev_token, target_hidden) → top-B0 candidates.
        prev_tok_arr = mx.array([[prev_token]], dtype=mx.int32)
        hid_arr = target_hidden.reshape(1, 1, H)
        logits, draft_hid = self(prev_tok_arr, hid_arr, return_hidden=True)
        mx.eval(logits, draft_hid)
        import numpy as np

        node_tokens: list = []
        node_parents: list = []
        node_depths: list = []
        node_paths: list = []
        node_hidden: list = []
        # Queue of (parent_idx, parent_hidden, parent_path, depth_of_next).
        # depth_of_next is 0 for the root expansion (we're producing depth-1 children).
        queue = []
        root_topk = int(topology[0])
        # Extract top-K candidate tokens at root level.
        root_logits = np.array(logits).reshape(self.cfg.vocab_size)
        top_idx = np.argpartition(-root_logits, kth=root_topk - 1)[:root_topk]
        top_idx = top_idx[np.argsort(-root_logits[top_idx])]
        for i, tok in enumerate(top_idx):
            node_tokens.append(int(tok))
            node_parents.append(-1)
            node_depths.append(1)
            node_paths.append([int(tok)])
            node_hidden.append(draft_hid.reshape(H))  # children of root reuse root's hidden
            queue.append((len(node_tokens) - 1, draft_hid.reshape(H), [int(tok)], 1))

        # BFS-expand by depth.
        for d in range(1, len(topology)):
            branching = int(topology[d])
            next_queue = []
            for parent_idx, parent_hid, parent_path, parent_depth in queue:
                # Run head on (parent_token, parent_hidden).
                pt_arr = mx.array([[parent_path[-1]]], dtype=mx.int32)
                ph_arr = parent_hid.reshape(1, 1, H)
                logits_d, draft_hid_d = self(pt_arr, ph_arr, return_hidden=True)
                mx.eval(logits_d, draft_hid_d)
                row = np.array(logits_d).reshape(self.cfg.vocab_size)
                if branching == 1:
                    top_idx_d = np.array([int(np.argmax(row))])
                else:
                    top_idx_d = np.argpartition(-row, kth=branching - 1)[:branching]
                    top_idx_d = top_idx_d[np.argsort(-row[top_idx_d])]
                child_hidden = draft_hid_d.reshape(H)
                for tok in top_idx_d:
                    node_tokens.append(int(tok))
                    node_parents.append(parent_idx)
                    node_depths.append(parent_depth + 1)
                    node_paths.append(parent_path + [int(tok)])
                    node_hidden.append(child_hidden)
                    next_queue.append(
                        (len(node_tokens) - 1, child_hidden,
                         parent_path + [int(tok)], parent_depth + 1)
                    )
            queue = next_queue
        return {
            "node_tokens": node_tokens,
            "node_parents": node_parents,
            "node_depths": node_depths,
            "node_paths": node_paths,
            "node_hidden": node_hidden,
        }

    def __call__(
        self,
        prev_tokens,
        target_hidden,
        return_hidden: bool = False,
        mask=None,
        block_lengths=None,
    ):
        """Forward pass.

        prev_tokens   : int32[B, S]
        target_hidden : float[B, S, H]
        return_hidden : if True, also return draft hidden state
        mask          : optional pre-built attention mask. If None, built
                        from self.cfg.attention_mode (cached on first call
                        for that (B, S) shape).
        block_lengths : required for block_diagonal mode; list[list[int]]
                        of per-batch sub-sequence lengths.

        Returns (logits, draft_hidden) if return_hidden else just logits.
        For n_predict_steps > 1, logits shape is (B, S, k, V) where k is
        the MTP head count and the kth slice predicts token at pos+k+1.
        """
        B, S = prev_tokens.shape
        H = self.cfg.hidden_dim
        # Embedding lookup: (V, H) gather. mx.transpose is metadata-only
        # (free), so we don't cache it as an attribute — caching would
        # pollute the parameter tree.
        embed_table = mx.transpose(self._token_embd, (1, 0))
        prev_embed = embed_table[prev_tokens]  # (B, S, H)
        x = mx.concatenate([prev_embed, target_hidden], axis=-1)  # (B, S, 2H)
        x = self.in_proj(x)  # (B, S, H)
        # Attention mask: build per-call if not supplied. mx.compile caches.
        if mask is None and self.cfg.attention_mode != ATTENTION_MODE_INDEPENDENT:
            mask = _build_mask(B, S, self.cfg.attention_mode, block_lengths)
        elif mask is None:
            mask = _build_mask(B, S, ATTENTION_MODE_INDEPENDENT)
        x = self.block(x, mask=mask)  # (B, S, H)
        draft_hidden = mx.fast.rms_norm(x, self.final_norm_w, self.cfg.rms_eps)
        # Project to vocab via frozen lm_head.
        if self.cfg.n_predict_steps == 1:
            logits = draft_hidden @ self._lm_head  # (B, S, V)
        else:
            # MTP: apply k separate head-specific projections of the draft
            # hidden into vocab. We don't add per-head MLPs here (paper-
            # standard is a small per-head residual); v0 = shared lm_head
            # with a per-head learned shift. Implement: stack k learned
            # offset vectors in the hidden dim and project k logit slices.
            # Falls back to single-head if no per-head bias was added.
            logits_list = []
            for k in range(self.cfg.n_predict_steps):
                offset_attr = f"mtp_offset_{k}"
                if hasattr(self, offset_attr):
                    h_k = draft_hidden + getattr(self, offset_attr)
                else:
                    h_k = draft_hidden
                logits_list.append(h_k @ self._lm_head)
            logits = mx.stack(logits_list, axis=-2)  # (B, S, k, V)
        if return_hidden:
            return logits, draft_hidden
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
    aux_target_kind: str = "next",
):
    """Cross-entropy on next-token prediction + optional MSE aux.

    logits             : (B, S, V)  for k=1
                         (B, S, K, V) for MTP with k=K heads
    target_hidden      : (B, S, H)  — captured target hidden (current pos)
    target_next_tokens : (B, S)     for k=1
                         (B, S, K)  for MTP — token at pos+1..pos+K
    draft_hidden       : (B, S, H)  — head's pre-lm_head output
    aux_weight         : 0.0 disables MSE aux
    aux_target_kind    : "current" → MSE(draft_hidden, target_hidden[pos])
                         "next"    → MSE(draft_hidden[s], target_hidden[s+1])
                         The "next" variant directly trains the geometric
                         alignment that drives EAGLE-3 acceptance — predicting
                         the NEXT hidden, not echoing the current one.
                         Tracks the loss the verifier actually scores.
    """
    # ---- CE ----
    if logits.ndim == 4:
        # MTP: (B, S, K, V) → flatten (B*S*K, V); targets (B, S, K) → (B*S*K,)
        B, S, K, V = logits.shape
        flat_logits = logits.reshape(-1, V)
        flat_targets = target_next_tokens.reshape(-1)
    else:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = target_next_tokens.reshape(-1)
    ce = nn.losses.cross_entropy(flat_logits, flat_targets, reduction="mean")

    if aux_weight > 0.0 and draft_hidden is not None:
        if aux_target_kind == "next":
            # Predict the NEXT hidden: shift target_hidden by 1 along S.
            # Loss on positions 0..S-2 only (last position has no next target).
            if target_hidden.shape[1] < 2:
                mse = mx.zeros(())
            else:
                pred = draft_hidden[:, :-1, :]    # (B, S-1, H)
                tgt = target_hidden[:, 1:, :]      # (B, S-1, H)
                mse = mx.mean((pred - tgt) ** 2)
        else:  # "current"
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
