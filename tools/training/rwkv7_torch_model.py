"""Pure-PyTorch RWKV-7 ("Goose") forward, faithful to dismantle's validated Rust
oracle (`crates/hawking-core/src/model_rwkv7.rs`, bit-exact vs llama.cpp).

This is the correctness/training-shape forward: a full-sequence prefill that
runs the WKV-7 recurrence over time with the per-head SxS state, matching the
scalar recurrence in `ggml_compute_forward_rwkv_wkv7_f32`.

Every [V] math detail (lerp sign, decay/iclr/value/gate nonlinearities, k
adjust, l2-norm eps, group-norm eps, r*k*r_k bonus, gate-before-o_proj,
squared-relu channel mix) is transcribed directly from `rwkv7.rs`.

Shapes (g1-0.4B): 24 layers, 16 heads x 64 head_dim, n_embd 1024, n_ff 4096,
vocab 65536, ln_eps 1e-5. HF `nn.Linear` weight is [out, in] so a projection is
`x @ W.T`. tie_word_embeddings=False.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


@dataclass
class RWKV7Config:
    n_embd: int = 1024
    n_layer: int = 24
    n_ff: int = 4096
    head_dim: int = 64
    n_head: int = 16  # 1024 / 64 — matches the Rust oracle (head_count = n_embd/head_size)
    vocab_size: int = 65536
    ln_eps: float = 1e-5
    decay_lora: int = 64
    iclr_lora: int = 64
    value_res_lora: int = 32
    gate_lora: int = 128
    # Opt-in chunked / parallel-scan WKV-7 (7-9x faster fwd+bwd, numerically equal
    # to the sequential loop). Default off so the validated sequential path remains
    # the parity reference; turn on for training speed once parity is re-confirmed.
    use_chunked: bool = False
    chunk_size: int = 32
    # WKV-7 head group-norm eps = head_dim * ln_eps (= 64e-5 for 0.4B). See
    # rwkv7.rs `gn_eps = 64e-5`.
    @property
    def gn_eps(self) -> float:
        return self.head_dim * self.ln_eps


def _layernorm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm over the last dim, population variance, then *w + b — matches
    ggml_norm / the Rust `layernorm`. Use unbiased=False so the divisor is N."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) * torch.rsqrt(var + eps) * w + b


class RWKV7TimeMix(nn.Module):
    """WKV-7 time-mix block. Mirrors `RwkvSeven::time_mix`."""

    def __init__(self, cfg: RWKV7Config, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        n = cfg.n_embd

        # Main projections (HF nn.Linear, [out, in], no bias).
        self.r_proj = nn.Linear(n, n, bias=False)
        self.k_proj = nn.Linear(n, n, bias=False)
        self.v_proj = nn.Linear(n, n, bias=False)
        self.o_proj = nn.Linear(n, n, bias=False)

        # Token-shift lerp coefficients (HF stores [1,1,n]; we keep as [n]).
        self.x_r = nn.Parameter(torch.zeros(n))
        self.x_w = nn.Parameter(torch.zeros(n))
        self.x_k = nn.Parameter(torch.zeros(n))
        self.x_v = nn.Parameter(torch.zeros(n))
        self.x_a = nn.Parameter(torch.zeros(n))
        self.x_g = nn.Parameter(torch.zeros(n))

        # Decay LoRA (w): w1=[decay,n], w2=[n,decay], w0=bias[n].
        self.w1 = nn.Parameter(torch.zeros(cfg.decay_lora, n))
        self.w2 = nn.Parameter(torch.zeros(n, cfg.decay_lora))
        self.w0 = nn.Parameter(torch.zeros(n))

        # In-context-learning-rate LoRA (a): a1=[iclr,n], a2=[n,iclr], a0=bias[n].
        self.a1 = nn.Parameter(torch.zeros(cfg.iclr_lora, n))
        self.a2 = nn.Parameter(torch.zeros(n, cfg.iclr_lora))
        self.a0 = nn.Parameter(torch.zeros(n))

        # Value-residual LoRA (v): layer > 0 only. v1=[vres,n], v2=[n,vres], v0=bias[n].
        if layer_idx > 0:
            self.v1 = nn.Parameter(torch.zeros(cfg.value_res_lora, n))
            self.v2 = nn.Parameter(torch.zeros(n, cfg.value_res_lora))
            self.v0 = nn.Parameter(torch.zeros(n))
        else:
            self.register_parameter("v1", None)
            self.register_parameter("v2", None)
            self.register_parameter("v0", None)

        # Gate LoRA (g): g1=[gate,n], g2=[n,gate], NO bias.
        self.g1 = nn.Parameter(torch.zeros(cfg.gate_lora, n))
        self.g2 = nn.Parameter(torch.zeros(n, cfg.gate_lora))

        # Per-channel vectors.
        self.k_k = nn.Parameter(torch.zeros(n))
        self.k_a = nn.Parameter(torch.zeros(n))
        self.r_k = nn.Parameter(torch.zeros(cfg.n_head, cfg.head_dim))  # [H, D]

        # WKV head group-norm.
        self.g_norm_w = nn.Parameter(torch.ones(n))
        self.g_norm_b = nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor, v_first: torch.Tensor | None):
        """x: [B, T, n_embd] (already attn_norm'd). Returns (out[B,T,n], v_first).

        Full-sequence prefill: token-shift uses the previous timestep (zeros at
        t=0); the WKV-7 recurrence loops over t carrying per-head SxS state.
        """
        cfg = self.cfg
        B, T, n = x.shape
        H, D = cfg.n_head, cfg.head_dim

        # token-shift: x_prev[t] = x[t-1], x_prev[0] = 0. sx = x_prev - x.
        x_prev = F.pad(x, (0, 0, 1, -1))  # shift right by one along T
        sx = x_prev - x

        # Per-slot lerp: xg = x + sx * x_g  (== x + (x_prev - x) * x_g).
        xr = x + sx * self.x_r
        xw = x + sx * self.x_w
        xk = x + sx * self.x_k
        xv = x + sx * self.x_v
        xa = x + sx * self.x_a
        xg = x + sx * self.x_g

        # r = Wr @ xr
        r = self.r_proj(xr)  # [B,T,n]

        # w = exp(-0.606531 * sigmoid(w0 + tanh(xw @ w1.T) @ w2.T))
        w = torch.tanh(xw @ self.w1.T) @ self.w2.T + self.w0
        w = torch.exp(-0.606531 * torch.sigmoid(w))  # [B,T,n]

        # k = Wk @ xk ; v = Wv @ xv
        k = self.k_proj(xk)
        v = self.v_proj(xv)

        # value-residual mix: layer 0 establishes v_first; deeper layers blend.
        if self.layer_idx == 0:
            v_first = v
        else:
            g_v = torch.sigmoid((xv @ self.v1.T) @ self.v2.T + self.v0)
            v = v + (v_first - v) * g_v

        # gate: g = sigmoid(xg @ g1.T) @ g2.T  (sigmoid on the LoRA-hidden, no bias)
        g = torch.sigmoid(xg @ self.g1.T) @ self.g2.T  # [B,T,n]

        # a = sigmoid(a0 + (xa @ a1.T) @ a2.T)  (in-context learning rate)
        a = torch.sigmoid((xa @ self.a1.T) @ self.a2.T + self.a0)

        # kk = l2norm_per_head(k * k_k), eps 1e-12 (ggml_l2_norm).
        kk = k * self.k_k
        kk = kk.view(B, T, H, D)
        kk = kk / kk.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        kk = kk.view(B, T, n)

        # k = k + (a - 1) * (k * k_a)
        k = k + (a - 1.0) * (k * self.k_a)

        # WKV-7 op inputs: a_op = -kk, b_op = kk * a.
        a_op = -kk
        b_op = kk * a

        # ---- WKV-7 recurrence over time ----
        # Reshape to per-head [B, T, H, D]. State S is [B, H, D(out=i), D(in=j)].
        rh = r.view(B, T, H, D)
        kh = k.view(B, T, H, D)
        vh = v.view(B, T, H, D)
        wh = w.view(B, T, H, D)
        ah = a_op.view(B, T, H, D)
        bh = b_op.view(B, T, H, D)

        if self.cfg.use_chunked:
            # Parallel-scan WKV-7: 7-9x faster, numerically equal to the loop below.
            from rwkv7_chunked import wkv7_chunked
            out = wkv7_chunked(rh, wh, kh, vh, ah, bh, chunk_size=self.cfg.chunk_size)
            out = out.reshape(B, T, n)
        else:
            S = torch.zeros(B, H, D, D, dtype=x.dtype, device=x.device)
            out = torch.empty(B, T, H, D, dtype=x.dtype, device=x.device)
            for t in range(T):
                w_t = wh[:, t]   # [B,H,D]  (index j)
                k_t = kh[:, t]   # [B,H,D]  (index j)
                v_t = vh[:, t]   # [B,H,D]  (index i)
                a_t = ah[:, t]   # [B,H,D]  (index j)  (a_op)
                b_t = bh[:, t]   # [B,H,D]  (index j)  (b_op)
                r_t = rh[:, t]   # [B,H,D]  (index j)

                # sa[i] = sum_j a_op[j] * S[i][j]   (contract over j)
                sa = torch.einsum("bhij,bhj->bhi", S, a_t)  # [B,H,D] (index i)
                # S[i][j] = S[i][j]*w[j] + v[i]*k[j] + sa[i]*b_op[j]
                S = (
                    S * w_t.unsqueeze(2)               # *w[j] broadcast over i
                    + v_t.unsqueeze(3) * k_t.unsqueeze(2)   # v[i]*k[j]
                    + sa.unsqueeze(3) * b_t.unsqueeze(2)    # sa[i]*b_op[j]
                )
                # out[i] = sum_j S[i][j] * r[j]
                out[:, t] = torch.einsum("bhij,bhj->bhi", S, r_t)

            out = out.reshape(B, T, n)

        # group-norm per head (eps = head_dim * ln_eps), then *g_norm_w + g_norm_b.
        og = out.view(B, T, H, D)
        mean = og.mean(dim=-1, keepdim=True)
        var = og.var(dim=-1, keepdim=True, unbiased=False)
        og = (og - mean) * torch.rsqrt(var + cfg.gn_eps)
        out = og.reshape(B, T, n) * self.g_norm_w + self.g_norm_b

        # bonus: out += v * (rowsum_per_head(k * r * r_k))
        kr = (kh * rh * self.r_k).sum(dim=-1, keepdim=True)  # [B,T,H,1]
        out = (out.view(B, T, H, D) + vh * kr).reshape(B, T, n)

        # gate applied BEFORE o_proj.
        out = self.o_proj(out * g)
        return out, v_first


class RWKV7ChannelMix(nn.Module):
    """ReLU^2 (sqrelu) MLP with token-shift. Mirrors `RwkvSeven::channel_mix`."""

    def __init__(self, cfg: RWKV7Config):
        super().__init__()
        self.cfg = cfg
        self.x_k = nn.Parameter(torch.zeros(cfg.n_embd))
        self.key = nn.Linear(cfg.n_embd, cfg.n_ff, bias=False)
        self.value = nn.Linear(cfg.n_ff, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # xk = x + (x_prev - x) * x_k
        x_prev = F.pad(x, (0, 0, 1, -1))
        xk = x + (x_prev - x) * self.x_k
        k = torch.relu(self.key(xk))
        k = k * k  # squared-relu
        return self.value(k)


class RWKV7Block(nn.Module):
    def __init__(self, cfg: RWKV7Config, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        # LN0 (embedding norm) lives on layer 0 only.
        if layer_idx == 0:
            self.pre_norm_w = nn.Parameter(torch.ones(cfg.n_embd))
            self.pre_norm_b = nn.Parameter(torch.zeros(cfg.n_embd))
        else:
            self.register_parameter("pre_norm_w", None)
            self.register_parameter("pre_norm_b", None)
        self.attn_norm_w = nn.Parameter(torch.ones(cfg.n_embd))
        self.attn_norm_b = nn.Parameter(torch.zeros(cfg.n_embd))
        self.ffn_norm_w = nn.Parameter(torch.ones(cfg.n_embd))
        self.ffn_norm_b = nn.Parameter(torch.zeros(cfg.n_embd))
        self.attn = RWKV7TimeMix(cfg, layer_idx)
        self.ffn = RWKV7ChannelMix(cfg)

    def forward(self, hidden: torch.Tensor, v_first: torch.Tensor | None):
        eps = self.cfg.ln_eps
        # LN0 only at layer 0.
        if self.layer_idx == 0:
            residual = _layernorm(hidden, self.pre_norm_w, self.pre_norm_b, eps)
        else:
            residual = hidden
        h = _layernorm(residual, self.attn_norm_w, self.attn_norm_b, eps)
        h, v_first = self.attn(h, v_first)
        residual = residual + h
        h = _layernorm(residual, self.ffn_norm_w, self.ffn_norm_b, eps)
        h = self.ffn(h)
        residual = residual + h
        return residual, v_first


class RWKV7Model(nn.Module):
    def __init__(self, cfg: RWKV7Config):
        super().__init__()
        self.cfg = cfg
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.layers = nn.ModuleList([RWKV7Block(cfg, i) for i in range(cfg.n_layer)])
        self.norm_w = nn.Parameter(torch.ones(cfg.n_embd))
        self.norm_b = nn.Parameter(torch.zeros(cfg.n_embd))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Set True to recompute each block in backward (saves activation memory for
        # the long WKV-7 recurrence graph). Only active in training mode.
        self.grad_checkpoint = False

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False,
                return_final_hidden: bool = False):
        """input_ids: [B, T] long. Returns logits [B, T, vocab].

        If return_hidden, also returns the list of per-block hidden states (post
        each block) for the layerwise parity harness.
        If return_final_hidden, returns the post-final-norm hidden [B, T, n_embd]
        WITHOUT the lm_head — lets the SFT loss apply the 65K-vocab projection only
        on supervised positions (same loss, far less compute/memory).
        """
        hidden = self.embeddings(input_ids)  # raw embedding (LN0 applied inside block 0)
        v_first = None
        hidden_states = []
        ckpt = self.grad_checkpoint and self.training and not return_hidden
        for block in self.layers:
            if ckpt:
                # v_first is None at block 0 (a non-tensor arg checkpoint passes
                # through). use_reentrant=False handles that and is the lossless mode.
                hidden, v_first = torch.utils.checkpoint.checkpoint(
                    block, hidden, v_first, use_reentrant=False
                )
            else:
                hidden, v_first = block(hidden, v_first)
            if return_hidden:
                hidden_states.append(hidden)
        hidden = _layernorm(hidden, self.norm_w, self.norm_b, self.cfg.ln_eps)
        if return_final_hidden:
            return hidden
        logits = self.lm_head(hidden)
        if return_hidden:
            return logits, hidden_states
        return logits
