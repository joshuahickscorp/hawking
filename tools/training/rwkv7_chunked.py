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

    Numerical note: the intra-chunk attention uses the factored decay
    ``exp(gk_cs[t]) * exp(-gk_cs[s])`` form. With ``w in (0.5, 1)`` we have
    ``gk in (-0.694, 0)``, so ``exp(-gk_cs[s])`` is bounded by
    ``exp(0.694 * chunk_size)``; this stays within fp32 range for
    ``chunk_size`` up to ~120 and comfortably so for the practical sizes used
    here (<= 64).
    """
    B, T, H, D = r.shape
    c = int(chunk_size)
    assert c >= 1

    # --- right-pad T up to a multiple of c with inert steps ---
    # The recurrence is causal, so padded tail steps never affect earlier
    # outputs; we slice the output back to T at the end. Pad value 0 for
    # r/k/v/alpha/beta is inert; w-pad = 1 (gk-pad = 0) keeps state finite.
    pad = (c - T % c) % c
    if pad:
        zpad = (0, 0, 0, 0, 0, pad)  # pad along the T axis (dim=1) at the end
        r = F.pad(r, zpad)
        k = F.pad(k, zpad)
        v = F.pad(v, zpad)
        a_op = F.pad(a_op, zpad)
        b_op = F.pad(b_op, zpad)
        w = F.pad(w, zpad, value=1.0)  # decay 1.0 -> gk 0.0 (no decay, inert)
    Tp = T + pad
    N = Tp // c

    # gk = log(w) is the additive log-decay fla operates on.
    gk = torch.log(w)

    # --- reshape to per-chunk [B, H, N, c, D] (fla layout: heads before time) ---
    def chunkify(x):
        # [B, T, H, D] -> [B, H, N, c, D]
        return x.view(B, N, c, H, D).permute(0, 3, 1, 2, 4).contiguous()

    q = chunkify(r)
    kc = chunkify(k)
    vc = chunkify(v)
    alpha = chunkify(a_op)
    beta = chunkify(b_op)
    gkc = chunkify(gk)

    # cumulative log-decay within each chunk (inclusive): gk_cs[t] = sum_{s<=t} gk[s]
    gk_cs = gkc.cumsum(dim=-2)  # [B,H,N,c,D]
    exp_cs = gk_cs.exp()              # exp(gk_cs[t])           (<= 1)
    exp_neg_cs = (-gk_cs).exp()       # exp(-gk_cs[s])          (>= 1, bounded)
    exp_cs_excl = (gk_cs - gkc).exp() # exp(gk_cs[t] - gk[t]) = exp(gk_cs[t-1])

    # Decay-weighted factors so that A[t,s] = <x_t * exp(gk_cs[t]), y_s * exp(-gk_cs[s])>
    # == sum_d x[t,d] y[s,d] exp(gk_cs[t,d] - gk_cs[s,d]).
    q_dec = q * exp_cs            # query decayed from chunk start
    a_dec = alpha * exp_cs_excl  # alpha decayed (one step short: uses gk_cs[t-1])
    k_inv = kc * exp_neg_cs      # key   decayed by -gk_cs[s]
    b_inv = beta * exp_neg_cs    # beta  decayed by -gk_cs[s]

    # Causal masks over the (c x c) chunk attention.
    tri_le = torch.tril(torch.ones(c, c, dtype=torch.bool, device=r.device))   # s <= t
    tri_lt = torch.tril(torch.ones(c, c, dtype=torch.bool, device=r.device), -1)  # s < t

    def masked_attn(x_dec, y_inv, mask):
        # [B,H,N,c,D] x [B,H,N,c,D] -> [B,H,N,c,c]  (A[t,s]), then mask.
        A = torch.einsum("bhntd,bhnsd->bhnts", x_dec, y_inv)
        return A * mask  # mask is [c,c] broadcast over leading dims

    A_qk = masked_attn(q_dec, k_inv, tri_le)  # s <= t
    A_qb = masked_attn(q_dec, b_inv, tri_le)  # s <= t
    A_ak = masked_attn(a_dec, k_inv, tri_lt)  # s <  t
    A_ab = masked_attn(a_dec, b_inv, tri_lt)  # s <  t  (strictly lower-tri)

    # --- T_ab = (I - A_ab)^{-1}, A_ab strictly lower-triangular (nilpotent) ---
    # Solve L @ T_ab = I with L = I - A_ab by forward substitution, building the
    # result row-by-row out-of-place so autograd flows (no in-place index writes).
    # Row t:  T[t] = e_t + sum_{s<t} A_ab[t,s] * T[s].
    eye = torch.eye(c, dtype=q.dtype, device=r.device).expand(B, H, N, c, c)
    rows = [eye[..., 0, :]]  # T[0] = e_0 (A_ab row 0 is all-zero)
    for t in range(1, c):
        coeff = A_ab[..., t, :t]                       # [B,H,N,t]
        prev = torch.stack(rows, dim=-2)               # [B,H,N,t,c]
        rows.append(eye[..., t, :] + torch.einsum("bhns,bhnsc->bhnc", coeff, prev))
    T_ab = torch.stack(rows, dim=-2)                   # [B,H,N,c,c]

    # u and wmat (fla: u = T_ab @ (A_ak @ v); wmat = T_ab @ (exp(gk_cs-gk)*alpha))
    u = T_ab @ (A_ak @ vc)               # [B,H,N,c,D]   in-chunk value contrib
    wmat = T_ab @ (exp_cs_excl * alpha)  # [B,H,N,c,D]   in-chunk -> state proj

    # decay from each position to the chunk end (for the inter-chunk state update)
    gk_last = gk_cs[..., -1:, :]               # [B,H,N,1,D]
    decay_to_end = (gk_last - gk_cs).exp()     # exp(gk_cs[-1] - gk_cs[t])  (<= 1)
    k_end = kc * decay_to_end
    b_end = beta * decay_to_end
    chunk_decay = gk_last[..., 0, :].exp()     # exp(gk_cs[-1]) total chunk decay [B,H,N,D]

    # --- sequential scan over the N chunks, carrying the SxS state S[j,i] ---
    # (fla state layout: rows = key dim j, cols = value dim i. dismantle's S[i][j]
    #  is the transpose; the chunked algebra is the fla one throughout.)
    S = torch.zeros(B, H, D, D, dtype=q.dtype, device=r.device)
    outs = []
    for n in range(N):
        v2 = u[:, :, n] + wmat[:, :, n] @ S          # [B,H,c,D] effective values
        o = (
            A_qk[:, :, n] @ vc[:, :, n]              # intra: query x key x value
            + A_qb[:, :, n] @ v2                     # intra: query x beta x v2
            + (q[:, :, n] * exp_cs[:, :, n]) @ S     # inter: carry-in state
        )
        outs.append(o)
        S = (
            S * chunk_decay[:, :, n].unsqueeze(-1)             # decay whole state
            + k_end[:, :, n].transpose(-1, -2) @ vc[:, :, n]   # key  outer value
            + b_end[:, :, n].transpose(-1, -2) @ v2            # beta outer v2
        )

    out = torch.stack(outs, dim=2)                  # [B,H,N,c,D]
    # back to [B, T, H, D] and drop the padding.
    out = out.permute(0, 2, 3, 1, 4).reshape(B, Tp, H, D)
    return out[:, :T]


def _bench(B=2, T=512, H=16, D=64, chunks=(8, 16, 32, 64), iters=5):
    """CPU wall-clock benchmark: sequential vs chunked forward+backward.

    Run with ``python tools/training/rwkv7_chunked.py`` to reproduce the
    measured speedup. CPU-only (the GPU is reserved for training).
    """
    import time

    torch.manual_seed(0)

    def mk(requires_grad=True):
        f = torch.float32

        def rn():
            return torch.randn(B, T, H, D, dtype=f)

        kk = rn()
        kk = kk / kk.norm(dim=-1, keepdim=True)
        a = torch.sigmoid(rn())
        w = torch.exp(-0.606531 * torch.sigmoid(rn()))
        ts = [rn(), w, rn(), rn(), -kk, kk * a]  # r, w, k, v, a_op, b_op
        if requires_grad:
            ts = [t.requires_grad_(True) for t in ts]
        return ts

    def timeit(fn):
        for _ in range(2):  # warmup
            o = fn(mk())
            o.sum().backward()
        samples = []
        for _ in range(iters):
            ins = mk()
            t0 = time.perf_counter()
            fn(ins).sum().backward()
            samples.append(time.perf_counter() - t0)
        return sorted(samples)[len(samples) // 2]

    t_seq = timeit(lambda ins: wkv7_sequential_ref(*ins))
    print(f"sequential fwd+bwd  B={B} T={T} H={H} D={D}: {t_seq*1e3:8.1f} ms (median/{iters})")
    print(f"{'chunk':>6} {'ms':>9} {'speedup':>8} {'fwd max_abs':>12}")
    fixed = [t.detach() for t in mk(False)]
    ref = wkv7_sequential_ref(*fixed)
    for c in chunks:
        if c > T:
            continue
        t = timeit((lambda c: lambda ins: wkv7_chunked(*ins, chunk_size=c))(c))
        md = (wkv7_chunked(*fixed, chunk_size=c) - ref).abs().max().item()
        print(f"{c:>6} {t*1e3:>9.1f} {t_seq/t:>7.2f}x {md:>12.2e}")


if __name__ == "__main__":
    _bench()
