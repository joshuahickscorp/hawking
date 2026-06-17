"""Chunked / parallel-scan WKV-7 (RWKV-7 "Goose") recurrence in pure PyTorch.

Numerically equivalent to the validated *sequential* WKV-7 time-mix loop in
``tools/training/rwkv7_torch_model.py`` (``RWKV7TimeMix.forward``), but processes
a chunk of ``chunk_size`` tokens with matmuls (intra-chunk) while carrying the
per-head SxS chunk-boundary state (inter-chunk). This turns the O(T) sequential
state recurrence into O(T / chunk_size) sequential chunk steps, each of which is
internally parallel — the point being to speed up training and inference prefill.

Algorithm
---------
The WKV-7 recurrence is a *diagonal-plus-low-rank* (DPLR) generalized delta rule.
Per head, with the sequential state ``S[i][j]`` (i = value/out dim, j = key/in dim)::

    sa[i]    = sum_j a_op[j] * S[i][j]
    S[i][j]  = S[i][j] * w[j] + v[i] * k[j] + sa[i] * b_op[j]
    out[i]   = sum_j S[i][j] * r[j]

This is exactly the recurrence implemented by ``flash-linear-attention``'s
``chunk_rwkv7`` -> ``chunk_dplr_delta_rule`` (RWKV-7 reduces to a generalized
delta rule). The chunked closed form is transcribed from fla's reference
``fla/ops/generalized_delta_rule/dplr/naive.py::dplr_chunkwise`` (and verified
term-by-term against fla's sequential ``dplr_recurrence`` and the fla triton
``fused_recurrent_rwkv7_fwd_kernel`` in ``fla/ops/rwkv7/fused_recurrent.py``).

Index/parameter mapping  (dismantle  ->  fla DPLR):

    r      -> q   (query)          [B, T, H, D]
    k      -> k   (key)            [B, T, H, D]
    v      -> v   (value)          [B, T, H, D]
    a_op   -> alpha  (= -kk)       [B, T, H, D]
    b_op   -> beta   (= kk * a)    [B, T, H, D]
    w      -> exp(gk)              [B, T, H, D]   so  gk = log(w)  (log-decay)
    scale  = 1.0                   (fla applies q * d_k**-0.5 internally; we do
                                    NOT pre-scale, and pass scale=1.0)

fla's state ``S_fla[j, i]`` (j = key dim, i = value dim) is the transpose of
dismantle's ``S[i][j]``; the two recurrences are algebraically identical (the
``v[i]*k[j]`` outer product and the ``sa[i]*b_op[j]`` rank-1 update are symmetric
under that transpose). This file works in the fla ``[B, H, T, D]`` layout
internally and returns dismantle's ``[B, T, H, D]`` layout.

The chunked decomposition, per chunk of length ``c`` (cumulative log-decay
``gk_cs`` = cumsum of ``gk`` within the chunk):

  Intra-chunk attention matrices (lower-triangular, causal):
    A_qk[t,s] = <q_t, k_s> * exp(gk_cs[t] - gk_cs[s])          (s <= t)
    A_qb[t,s] = <q_t, beta_s> * exp(gk_cs[t] - gk_cs[s])       (s <= t)
    A_ak[t,s] = <alpha_t, k_s> * exp(gk_cs[t] - gk[t] - gk_cs[s])     (s < t)
    A_ab[t,s] = <alpha_t, beta_s> * exp(gk_cs[t] - gk[t] - gk_cs[s])  (s < t)

  The strictly-lower-triangular ``A_ab`` is the rank-1 "delta" coupling within a
  chunk; ``T_ab = (I - A_ab)^{-1}`` (forward-substitution; A_ab is nilpotent
  strictly-lower-triangular so the Neumann series terminates) "unrolls" the
  in-chunk delta updates. Then:
    u   = T_ab @ (A_ak @ v)               (in-chunk value contribution)
    wmat= T_ab @ (exp(gk_cs - gk) * alpha) (in-chunk -> state projection)
    v2  = u + wmat @ S_chunk_start         (full effective values incl. carry-in)
    out = A_qk @ v + A_qb @ v2 + (q * exp(gk_cs)) @ S_chunk_start
  Inter-chunk state carry (decay-to-chunk-end ``d`` = exp(gk_cs[-1] - gk_cs)):
    S_next = S * exp(gk_cs[-1]) + (k * d)^T @ v + (beta * d)^T @ v2

Autograd flows through everything (no in-place index writes on autograd tensors),
so this can be used directly for training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def wkv7_sequential_ref(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a_op: torch.Tensor,
    b_op: torch.Tensor,
) -> torch.Tensor:
    """Reference: the *exact* sequential WKV-7 recurrence from
    ``RWKV7TimeMix.forward`` (the ``for t in range(T)`` loop), extracted as a
    standalone function over per-head tensors.

    All inputs are ``[B, T, H, D]``. Returns ``out`` of shape ``[B, T, H, D]``
    (the per-head WKV output, *before* group-norm / bonus / gate / o_proj).

    Recurrence (per batch/head, S[i][j] with i = value dim, j = key dim):
        sa[i]   = sum_j a_op[j] * S[i][j]
        S[i][j] = S[i][j] * w[j] + v[i] * k[j] + sa[i] * b_op[j]
        out[i]  = sum_j S[i][j] * r[j]
    """
    B, T, H, D = r.shape
    S = torch.zeros(B, H, D, D, dtype=r.dtype, device=r.device)
    out = torch.empty(B, T, H, D, dtype=r.dtype, device=r.device)
    for t in range(T):
        w_t = w[:, t]      # [B,H,D]  (index j)
        k_t = k[:, t]      # [B,H,D]  (index j)
        v_t = v[:, t]      # [B,H,D]  (index i)
        a_t = a_op[:, t]   # [B,H,D]  (index j)
        b_t = b_op[:, t]   # [B,H,D]  (index j)
        r_t = r[:, t]      # [B,H,D]  (index j)

        sa = torch.einsum("bhij,bhj->bhi", S, a_t)  # [B,H,D] (index i)
        S = (
            S * w_t.unsqueeze(2)
            + v_t.unsqueeze(3) * k_t.unsqueeze(2)
            + sa.unsqueeze(3) * b_t.unsqueeze(2)
        )
        out[:, t] = torch.einsum("bhij,bhj->bhi", S, r_t)
    return out


def wkv7_chunked(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a_op: torch.Tensor,
    b_op: torch.Tensor,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Chunked WKV-7 forward, numerically equal to :func:`wkv7_sequential_ref`.

    All inputs ``[B, T, H, D]``; ``w`` is the *multiplicative* per-step decay
    (in (0, 1)), ``a_op = -kk`` and ``b_op = kk * a`` as built by the time-mix.
    ``T`` need not be divisible by ``chunk_size`` (the sequence is right-padded
    with inert steps and the output is sliced back to ``T``).

    Returns ``out`` of shape ``[B, T, H, D]``.
    """
    raise NotImplementedError("chunked WKV-7 not implemented yet")
