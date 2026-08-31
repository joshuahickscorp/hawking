"""FPGA engine school — bit-exact functional references, no HDL.

These are golden models a later cycle model, HW emulation, and eventually real
hardware are checked against. They are not an FPGA backend and emit no HDL.
FPGA here is part of Accelerator / Physical Compiler / Fusion, not its own
civilization. Everything this module writes is STATIC_ONLY; bench state stays
UNKNOWN. No Era VI, no Odyssey IV.

    python3 tools/future/fpga_engines.py --selftest
    python3 tools/future/fpga_engines.py --build
    python3 -m pytest tools/future/test_fpga_engines.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, REPO, git

import argparse
import hashlib
import subprocess
from typing import Any

import numpy as np

RECEIPT = "FPGA_ENGINE_SCHOOL.json"
SCHEMA = "hawking.future.fpga_engines.v1"

# Recovered from receipts/headless/FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json
# candidates.shared_bf16_basis_nf4_residual.residual_codebook (NF4, 16 entries).
NF4_CODEBOOK = np.array(
    [
        -1.0,
        -0.6961928,
        -0.52507305,
        -0.3949175,
        -0.28444138,
        -0.18477343,
        -0.09105004,
        0.0,
        0.0795803,
        0.1609302,
        0.2461123,
        0.33791524,
        0.44070983,
        0.562617,
        0.72295684,
        1.0,
    ],
    dtype=np.float32,
)

# Symmetric signed bound used by gravity_native.pack_q4_g64 / Flash independent_q4_g64.
# bits=4 -> [-7, +7], not two's-complement [-8, +7].
def signed_bound(weight_bits: int) -> int:
    if weight_bits < 1 or weight_bits > 8:
        raise ValueError(f"weight_bits={weight_bits} not in 1..8")
    return (1 << (weight_bits - 1)) - 1 if weight_bits > 1 else 1


def _f32(x: Any) -> np.float32:
    return np.float32(x)


def _fadd(a: Any, b: Any) -> np.float32:
    return np.add(_f32(a), _f32(b), dtype=np.float32)


def _fmul(a: Any, b: Any) -> np.float32:
    return np.multiply(_f32(a), _f32(b), dtype=np.float32)


def _fsub(a: Any, b: Any) -> np.float32:
    return np.subtract(_f32(a), _f32(b), dtype=np.float32)


def _fdiv(a: Any, b: Any) -> np.float32:
    return np.divide(_f32(a), _f32(b), dtype=np.float32)


def _require_divides(k: int, group_size: int) -> None:
    if group_size < 1 or k % group_size != 0:
        raise ValueError(f"K={k} must be a positive multiple of group_size={group_size}")


def _check_codes(codes: np.ndarray, weight_bits: int) -> None:
    bound = signed_bound(weight_bits)
    if weight_bits == 1:
        if not np.isin(codes, np.array([-1, 1], dtype=np.int8)).all():
            raise ValueError("1-bit codes must be in {-1,+1} (sign code, not deletion)")
        return
    if np.any(codes < -bound) or np.any(codes > bound):
        raise ValueError(f"codes outside symmetric signed range [{-bound},{bound}] for bits={weight_bits}")


# ---------------------------------------------------------------------------
# Engines. Sequential left-to-right float32 is the declared reduction order.
# A spatial tree is a different number; see reductions().
# ---------------------------------------------------------------------------


def qgemv(
    codes: np.ndarray,
    scales: np.ndarray,
    x: np.ndarray,
    *,
    weight_bits: int,
    group_size: int,
) -> np.ndarray:
    """Quantized matrix-vector. y[m] = sum_k (code[m,k] * scale[m,k//G]) * x[k].

    dtype/width: codes int8[M,K] holding a symmetric signed `weight_bits` value;
    scales float32[M,K/G]; x float32[K]; y float32[M].
    Associativity: sequential left-to-right over K, fused dequant-then-multiply.
    Does not materialize a dense weight tensor.
    """
    codes = np.asarray(codes, dtype=np.int8)
    scales = np.asarray(scales, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if codes.ndim != 2:
        raise ValueError("codes must be rank-2 [M,K]")
    m, k = codes.shape
    _require_divides(k, group_size)
    n_g = k // group_size
    if scales.shape != (m, n_g):
        raise ValueError(f"scales shape {scales.shape} != {(m, n_g)}")
    if x.shape != (k,):
        raise ValueError(f"x shape {x.shape} != {(k,)}")
    _check_codes(codes, weight_bits)
    y = np.empty(m, dtype=np.float32)
    for row in range(m):
        acc = np.float32(0)
        for col in range(k):
            s = scales[row, col // group_size]
            w = _fmul(codes[row, col], s)
            acc = _fadd(acc, _fmul(w, x[col]))
        y[row] = acc
    return y


def lowbit_gemm(
    codes: np.ndarray,
    scales: np.ndarray,
    b: np.ndarray,
    *,
    weight_bits: int,
    group_size: int,
) -> np.ndarray:
    """Low-bit matrix-matrix. C = dequant(A_codes) @ B, sequential over K.

    dtype/width: codes int8[M,K]; scales float32[M,K/G]; B float32[K,N]; C float32[M,N].
    Associativity: sequential left-to-right over K for each (m,n).
    Integer add would be associative; the per-group float scale makes it not.
    """
    codes = np.asarray(codes, dtype=np.int8)
    scales = np.asarray(scales, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if codes.ndim != 2 or b.ndim != 2:
        raise ValueError("codes and b must be rank-2")
    m, k = codes.shape
    if b.shape[0] != k:
        raise ValueError(f"b rows {b.shape[0]} != K={k}")
    n = b.shape[1]
    _require_divides(k, group_size)
    n_g = k // group_size
    if scales.shape != (m, n_g):
        raise ValueError(f"scales shape {scales.shape} != {(m, n_g)}")
    _check_codes(codes, weight_bits)
    out = np.empty((m, n), dtype=np.float32)
    for row in range(m):
        for col in range(n):
            acc = np.float32(0)
            for t in range(k):
                s = scales[row, t // group_size]
                w = _fmul(codes[row, t], s)
                acc = _fadd(acc, _fmul(w, b[t, col]))
            out[row, col] = acc
    return out


def basis_projection(basis: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Project activations onto a shared basis. p[r] = sum_d basis[r,d] * x[d].

    dtype/width: basis float32[R,D]; x float32[D]; p float32[R].
    Associativity: sequential left-to-right over D.
    Recovers Flash 'shared_bf16_basis' / representation_library shared_basis:
    K shared basis vectors plus a projection of the activation, not a GEMV
    against a per-row dense weight.
    """
    basis = np.asarray(basis, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if basis.ndim != 2:
        raise ValueError("basis must be rank-2 [R,D]")
    r, d = basis.shape
    if x.shape != (d,):
        raise ValueError(f"x shape {x.shape} != {(d,)}")
    p = np.empty(r, dtype=np.float32)
    for row in range(r):
        acc = np.float32(0)
        for col in range(d):
            acc = _fadd(acc, _fmul(basis[row, col], x[col]))
        p[row] = acc
    return p


def factorized_projection(u: np.ndarray, v: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Two-stage low-rank projection y = U @ (V @ x). Inner-first, never U@V.

    dtype/width: U float32[M,R]; V float32[R,D]; x float32[D]; y float32[M].
    Associativity: sequential over D for the inner matvec, then sequential over
    R for the outer. (U@V)@x is a different float32 number and a different
    spatial schedule (it materializes a dense MD matrix).
    """
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError("U and V must be rank-2")
    m, r = u.shape
    r2, d = v.shape
    if r != r2:
        raise ValueError(f"U cols {r} != V rows {r2}")
    if x.shape != (d,):
        raise ValueError(f"x shape {x.shape} != {(d,)}")
    t = np.empty(r, dtype=np.float32)
    for row in range(r):
        acc = np.float32(0)
        for col in range(d):
            acc = _fadd(acc, _fmul(v[row, col], x[col]))
        t[row] = acc
    y = np.empty(m, dtype=np.float32)
    for row in range(m):
        acc = np.float32(0)
        for col in range(r):
            acc = _fadd(acc, _fmul(u[row, col], t[col]))
        y[row] = acc
    return y


def codebook_arithmetic(
    codebook: np.ndarray,
    indices: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Vector/product codebook lookup then accumulate (fused ADC, not decode-to-dense).

    Vector codebook: codebook float32[card, sub], indices int32[M, nchunk],
    x float32[nchunk*sub], y float32[M].
    Product codebook: codebook float32[S, card, sub], indices int32[M, nchunk, S].
    Associativity: sequential over (chunk, subspace, sub) in that order.
    Recovers C4 fused_adc_pq_matvec / gravity_pq: each chunk contributes one
    codebook entry dotted against its slice of x; the dense row is never written.
    """
    codebook = np.asarray(codebook, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int32)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if codebook.ndim == 2:
        codebook = codebook[np.newaxis, ...]
        if indices.ndim != 2:
            raise ValueError("vector codebook expects indices [M, nchunk]")
        indices = indices[..., np.newaxis]
    elif codebook.ndim == 3:
        if indices.ndim != 3:
            raise ValueError("product codebook expects indices [M, nchunk, S]")
    else:
        raise ValueError("codebook must be [card,sub] or [S,card,sub]")
    n_s, card, sub = codebook.shape
    m, nchunk, s2 = indices.shape
    if s2 != n_s:
        raise ValueError(f"indices S={s2} != codebook S={n_s}")
    if x.shape != (nchunk * sub,):
        raise ValueError(f"x shape {x.shape} != {(nchunk * sub,)}")
    if np.any(indices < 0) or np.any(indices >= card):
        raise ValueError("codebook index out of range")
    y = np.empty(m, dtype=np.float32)
    for row in range(m):
        acc = np.float32(0)
        for chunk in range(nchunk):
            base = chunk * sub
            for sub_i in range(n_s):
                entry = codebook[sub_i, indices[row, chunk, sub_i]]
                for s in range(sub):
                    acc = _fadd(acc, _fmul(entry[s], x[base + s]))
        y[row] = acc
    return y


def dictionary_arithmetic(
    dictionary: np.ndarray,
    atom_ids: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Dictionary atom selection then weighted sum. y = sum_i coeffs[i] * D[:, atom_ids[i]].

    dtype/width: dictionary float32[D, n_atoms]; atom_ids int32[K]; coeffs float32[K];
    y float32[D].
    Associativity: sequential over selected atoms in the given order (not resorted
    by atom id). Colliding atoms add; they are not unique'd.
    """
    dictionary = np.asarray(dictionary, dtype=np.float32)
    atom_ids = np.asarray(atom_ids, dtype=np.int32).reshape(-1)
    coeffs = np.asarray(coeffs, dtype=np.float32).reshape(-1)
    if dictionary.ndim != 2:
        raise ValueError("dictionary must be rank-2 [D, n_atoms]")
    d, n_atoms = dictionary.shape
    if atom_ids.shape != coeffs.shape:
        raise ValueError("atom_ids and coeffs length mismatch")
    if np.any(atom_ids < 0) or np.any(atom_ids >= n_atoms):
        raise ValueError("atom id out of range")
    y = np.zeros(d, dtype=np.float32)
    for i, atom in enumerate(atom_ids):
        c = coeffs[i]
        for row in range(d):
            y[row] = _fadd(y[row], _fmul(c, dictionary[row, atom]))
    return y


def sparse_residual(
    partial: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Index+value residual applied to a dense partial. y = partial; y[i_k] += v_k.

    dtype/width: partial float32[D]; indices int32[nnz]; values float32[nnz]; y float32[D].
    Associativity: sequential over residual terms in the given order. Colliding
    indices accumulate left-to-right. Recovers representation_library
    binary_sparse_residual / low_rank_plus_sparse exception set.
    """
    partial = np.asarray(partial, dtype=np.float32).reshape(-1)
    indices = np.asarray(indices, dtype=np.int32).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if indices.shape != values.shape:
        raise ValueError("indices and values length mismatch")
    d = partial.shape[0]
    if np.any(indices < 0) or np.any(indices >= d):
        raise ValueError("residual index out of range")
    y = np.array(partial, dtype=np.float32, copy=True)
    for i, v in zip(indices.tolist(), values.tolist()):
        y[i] = _fadd(y[i], v)
    return y


def routed_expert_accumulate(
    expert_outputs: np.ndarray,
    topk_ids: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Top-K routed accumulation with per-expert scale, in the given top-K order.

    dtype/width: expert_outputs float32[E,D]; topk_ids int32[K]; scales float32[K];
    y float32[D].
    Associativity: sequential over the given top-K order (router order, not
    resorted by expert id). Recovers Flash routed_plus_shared_expert /
    router_topk_and_gather: y = sum_t scales[t] * expert_outputs[topk_ids[t]].
    """
    expert_outputs = np.asarray(expert_outputs, dtype=np.float32)
    topk_ids = np.asarray(topk_ids, dtype=np.int32).reshape(-1)
    scales = np.asarray(scales, dtype=np.float32).reshape(-1)
    if expert_outputs.ndim != 2:
        raise ValueError("expert_outputs must be rank-2 [E,D]")
    n_e, d = expert_outputs.shape
    if topk_ids.shape != scales.shape:
        raise ValueError("topk_ids and scales length mismatch")
    if np.any(topk_ids < 0) or np.any(topk_ids >= n_e):
        raise ValueError("expert id out of range")
    y = np.zeros(d, dtype=np.float32)
    for t, e in enumerate(topk_ids.tolist()):
        s = scales[t]
        for col in range(d):
            y[col] = _fadd(y[col], _fmul(s, expert_outputs[e, col]))
    return y


def recurrent_state_transition(
    state: np.ndarray,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    decay: float,
    beta: float,
) -> dict[str, np.ndarray]:
    """DeltaNet-shaped gated-delta state update.

    Layout recovered from crates/hawking-core/shaders/qwen38_device_activations.metal
    (`qwen38_gated_delta_decode_vi`): state[ki, vi],
        decayed = S * decay
        kv[vi]  = sum_ki decayed[ki,vi] * key[ki]     # sequential over ki
        delta   = (value - kv) * beta
        S'      = decayed + key[:,None] * delta[None,:]
        out[vi] = sum_ki S'[ki,vi] * query[ki]        # sequential over ki

    dtype/width: state float32[K,V]; q,k float32[K]; v float32[V]; decay,beta float32;
    returns {state: float32[K,V], output: float32[V]}.
    Associativity: sequential left-to-right over the key axis for both reductions.
    A simd_sum / tree over ki is a different float32 number (the Metal sibling
    kernels use simd_sum; this golden is the sequential form they are checked
    against only after an associativity waiver).
    """
    s = np.asarray(state, dtype=np.float32)
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    k = np.asarray(key, dtype=np.float32).reshape(-1)
    v = np.asarray(value, dtype=np.float32).reshape(-1)
    d = _f32(decay)
    b = _f32(beta)
    if s.ndim != 2:
        raise ValueError("state must be rank-2 [K,V]")
    kdim, vdim = s.shape
    if q.shape != (kdim,) or k.shape != (kdim,) or v.shape != (vdim,):
        raise ValueError("q/k must be [K] and v must be [V]")
    decayed = np.empty_like(s)
    for i in range(kdim):
        for j in range(vdim):
            decayed[i, j] = _fmul(s[i, j], d)
    kv = np.empty(vdim, dtype=np.float32)
    for j in range(vdim):
        acc = np.float32(0)
        for i in range(kdim):
            acc = _fadd(acc, _fmul(decayed[i, j], k[i]))
        kv[j] = acc
    delta = np.empty(vdim, dtype=np.float32)
    for j in range(vdim):
        delta[j] = _fmul(_fsub(v[j], kv[j]), b)
    s_new = np.empty_like(s)
    for i in range(kdim):
        for j in range(vdim):
            s_new[i, j] = _fadd(decayed[i, j], _fmul(k[i], delta[j]))
    out = np.empty(vdim, dtype=np.float32)
    for j in range(vdim):
        acc = np.float32(0)
        for i in range(kdim):
            acc = _fadd(acc, _fmul(s_new[i, j], q[i]))
        out[j] = acc
    return {"state": s_new, "output": out}


def attention_score(q: np.ndarray, k: np.ndarray, scale: float) -> np.ndarray:
    """scores[i,j] = (sum_d q[i,d]*k[j,d]) / scale, sequential over d.

    dtype/width: q float32[Tq,D]; k float32[Tk,D]; scores float32[Tq,Tk].
    Associativity: sequential over D, then divide by scale.
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError("q and k must be rank-2")
    tq, d = q.shape
    tk, d2 = k.shape
    if d != d2:
        raise ValueError("q/k width mismatch")
    if scale == 0:
        raise ValueError("attention scale must be nonzero")
    sc = _f32(scale)
    scores = np.empty((tq, tk), dtype=np.float32)
    for i in range(tq):
        for j in range(tk):
            acc = np.float32(0)
            for t in range(d):
                acc = _fadd(acc, _fmul(q[i, t], k[j, t]))
            scores[i, j] = _fdiv(acc, sc)
    return scores


def attention_softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise sequential softmax: max, exp(x-max), sum, divide.

    Associativity: sequential left-to-right max and sequential left-to-right
    sum of exp. A tree-max or tree-sum is a different float32 number.
    exp is libm via numpy float32; both goldens must call the same exp.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("scores must be rank-2 [Tq,Tk]")
    tq, tk = scores.shape
    weights = np.empty_like(scores)
    for i in range(tq):
        m = scores[i, 0]
        for j in range(1, tk):
            m = np.maximum(m, scores[i, j])
        exps = np.empty(tk, dtype=np.float32)
        acc = np.float32(0)
        for j in range(tk):
            exps[j] = np.exp(_fsub(scores[i, j], m))
            acc = _fadd(acc, exps[j])
        for j in range(tk):
            weights[i, j] = _fdiv(exps[j], acc)
    return weights


def attention_weighted_values(weights: np.ndarray, v: np.ndarray) -> np.ndarray:
    """out[i,d] = sum_j weights[i,j] * v[j,d], sequential over j.

    dtype/width: weights float32[Tq,Tk]; v float32[Tk,Dv]; out float32[Tq,Dv].
    Associativity: sequential left-to-right over the key axis.
    """
    weights = np.asarray(weights, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if weights.ndim != 2 or v.ndim != 2:
        raise ValueError("weights and v must be rank-2")
    tq, tk = weights.shape
    tk2, dv = v.shape
    if tk != tk2:
        raise ValueError("weights/V key-length mismatch")
    out = np.empty((tq, dv), dtype=np.float32)
    for i in range(tq):
        for d in range(dv):
            acc = np.float32(0)
            for j in range(tk):
                acc = _fadd(acc, _fmul(weights[i, j], v[j, d]))
            out[i, d] = acc
    return out


def attention_primitives(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    scale: float,
) -> dict[str, np.ndarray]:
    """Score, softmax, weighted-value as separable pieces, then composed."""
    scores = attention_score(q, k, scale)
    weights = attention_softmax(scores)
    output = attention_weighted_values(weights, v)
    return {"scores": scores, "weights": weights, "output": output}


def sequential_reduce(x: np.ndarray) -> np.ndarray:
    """Left fold of +. Integer uses int64 (associative). Float32 uses sequential fadd.

    Associativity: sequential left-to-right. Empty input yields 0.
    """
    a = np.asarray(x).reshape(-1)
    if a.size == 0:
        return np.int64(0) if np.issubdtype(a.dtype, np.integer) else np.float32(0)
    if np.issubdtype(a.dtype, np.integer):
        acc = np.int64(0)
        for v in a:
            acc = acc + np.int64(v)
        return acc
    acc = np.float32(0)
    for v in a:
        acc = _fadd(acc, v)
    return acc


def tree_reduce(x: np.ndarray) -> np.ndarray:
    """Balanced binary tree of +. Split at n//2, reduce(left) + reduce(right).

    Associativity: recursive midpoint split, not pairwise-with-leftover and not
    sequential. For float32 this is a different number than sequential_reduce
    on some inputs (see VECTORS). Integer int64 + is associative so they match.
    Empty input yields 0; a singleton is itself.
    """
    a = np.asarray(x).reshape(-1)
    integer = bool(np.issubdtype(a.dtype, np.integer))

    def rec(lo: int, hi: int):
        n = hi - lo
        if n == 0:
            return np.int64(0) if integer else np.float32(0)
        if n == 1:
            return np.int64(a[lo]) if integer else _f32(a[lo])
        mid = lo + n // 2
        left, right = rec(lo, mid), rec(mid, hi)
        if integer:
            return np.int64(left) + np.int64(right)
        return _fadd(left, right)

    return rec(0, int(a.size))


def reductions(x: np.ndarray, *, mode: str) -> np.ndarray:
    """Tree or sequential reduction with declared associativity."""
    if mode == "sequential":
        return sequential_reduce(x)
    if mode == "tree":
        return tree_reduce(x)
    raise ValueError(f"unknown reduction mode {mode!r}")


def quantize_in_transit(x: np.ndarray, *, bits: int = 8) -> tuple[np.ndarray, np.float32]:
    """Symmetric signed quantize. scale = amax/bound, codes = rint(x/scale) clipped.

    amax is a sequential max of abs over C-order. dtype: x float32; codes int8;
    scale float32. bits in 1..8, same signed_bound as qgemv.
    """
    x = np.asarray(x, dtype=np.float32)
    bound = signed_bound(bits)
    amax = np.float32(0)
    flat = x.reshape(-1)
    for v in flat:
        amax = np.maximum(amax, np.abs(_f32(v)))
    scale = _f32(1) if amax == np.float32(0) else _fdiv(amax, np.float32(bound))
    codes = np.empty(x.shape, dtype=np.int8)
    lo, hi = -bound, bound
    for idx in np.ndindex(x.shape):
        q = np.rint(_fdiv(x[idx], scale))
        q = np.clip(q, lo, hi)
        codes[idx] = np.int8(q)
    return codes, scale


def transpose_in_transit(x: np.ndarray) -> np.ndarray:
    """Layout transpose, C-contiguous copy. Rank-2 only."""
    a = np.asarray(x)
    if a.ndim != 2:
        raise ValueError("transpose_in_transit expects rank-2")
    return np.ascontiguousarray(a.T)


def reduce_in_transit(x: np.ndarray) -> np.ndarray:
    """Sequential reduce of the last axis. Integer -> int64; float -> float32."""
    a = np.asarray(x)
    if a.ndim == 0:
        raise ValueError("reduce_in_transit needs at least a vector")
    if a.ndim == 1:
        return sequential_reduce(a)
    rows = a.shape[0]
    out_dtype = np.int64 if np.issubdtype(a.dtype, np.integer) else np.float32
    out = np.empty(rows, dtype=out_dtype)
    for i in range(rows):
        out[i] = sequential_reduce(a[i])
    return out


def digest_in_transit(x: np.ndarray) -> str:
    """SHA-256 of the tensor's little-endian C-contiguous raw bytes. Not a HMAC."""
    a = np.ascontiguousarray(x)
    if a.dtype == np.float32:
        blob = a.astype("<f4", copy=False).tobytes()
    elif a.dtype == np.int8:
        blob = a.astype("<i1", copy=False).tobytes()
    elif a.dtype == np.int32:
        blob = a.astype("<i4", copy=False).tobytes()
    elif a.dtype == np.int64:
        blob = a.astype("<i8", copy=False).tobytes()
    elif a.dtype == np.uint8:
        blob = a.tobytes()
    else:
        raise ValueError(f"unsupported digest dtype {a.dtype}")
    return hashlib.sha256(blob).hexdigest()


def pack_in_transit(codes: np.ndarray, *, bits: int = 4) -> np.ndarray:
    """Pack symmetric-signed codes. bits<=4: two offset-binary codes per uint8.

    stored = code + bound; packed[i] = lo | (hi << 4). Length must be even.
    Recovers gravity_native.pack_q4_g64 nibble layout (BOUND=7 at 4 bits).
    """
    codes = np.asarray(codes, dtype=np.int8).reshape(-1)
    _check_codes(codes, bits)
    if bits > 4:
        raise ValueError("pack_in_transit nibble pack supports bits<=4")
    if codes.size % 2 != 0:
        raise ValueError("pack_in_transit needs an even number of codes")
    bound = signed_bound(bits)
    stored = (codes.astype(np.int32) + bound).astype(np.uint8)
    return (stored[0::2] | (stored[1::2].astype(np.uint16) << 4).astype(np.uint8)).astype(np.uint8)


def semantic_transport_transform(
    payload: np.ndarray,
    *,
    pipeline: list[str] | tuple[str, ...],
    quant_bits: int = 8,
    pack_bits: int = 4,
) -> dict[str, Any]:
    """Quantize / transpose / reduce / digest / pack in transit, in the given order.

    This is the functional half of Fusion's REDUCE/DIGEST/COPY vocabulary
    (tools/accelerator/fusion_isa.py) plus the preboard transport policy
    "activations and partial reductions only". No byte moved on a wire; no
    bandwidth number.
    """
    state: dict[str, Any] = {
        "tensor": np.asarray(payload),
        "scale": None,
        "digest": None,
        "packed": None,
        "pipeline": list(pipeline),
    }
    for step in pipeline:
        if step == "quantize":
            codes, scale = quantize_in_transit(state["tensor"], bits=quant_bits)
            state["tensor"] = codes
            state["scale"] = scale
        elif step == "transpose":
            state["tensor"] = transpose_in_transit(state["tensor"])
        elif step == "reduce":
            state["tensor"] = reduce_in_transit(state["tensor"])
        elif step == "digest":
            state["digest"] = digest_in_transit(state["tensor"])
        elif step == "pack":
            state["packed"] = pack_in_transit(state["tensor"], bits=pack_bits)
        else:
            raise ValueError(f"unknown transport step {step!r}")
    return state


ENGINE_FNS: dict[str, Any] = {
    "qgemv": qgemv,
    "lowbit_gemm": lowbit_gemm,
    "basis_projection": basis_projection,
    "factorized_projection": factorized_projection,
    "codebook_arithmetic": codebook_arithmetic,
    "dictionary_arithmetic": dictionary_arithmetic,
    "sparse_residual": sparse_residual,
    "routed_expert_accumulate": routed_expert_accumulate,
    "recurrent_state_transition": recurrent_state_transition,
    "attention_primitives": attention_primitives,
    "reductions": reductions,
    "semantic_transport_transform": semantic_transport_transform,
}

# Organs recovered from receipts/headless/{FLASH_NEXT,QWEN27}_FPGA_ORGAN_MAP.json
# via git show (this worktree is a sparse checkout; those files are not materialized).
ORGAN_ENGINE_MAP: dict[str, dict[str, list[str]]] = {
    "flash-next": {
        "expert_bank": ["qgemv", "codebook_arithmetic", "dictionary_arithmetic", "basis_projection"],
        "router_topk_and_gather": ["routed_expert_accumulate"],
        "routed_plus_shared_expert": ["routed_expert_accumulate", "qgemv", "sparse_residual"],
        "deltanet_persistent_state": ["recurrent_state_transition"],
        "ngram_lookup_or_generator": ["dictionary_arithmetic", "codebook_arithmetic"],
        "sparse_attention": ["attention_primitives", "sparse_residual", "reductions"],
        "mtp_draft_verify_rollback": ["reductions", "semantic_transport_transform"],
    },
    "qwen27": {
        "mlp_gate_up_down": ["qgemv", "lowbit_gemm", "factorized_projection"],
        "gqa_qkv_and_output": [
            "qgemv",
            "attention_primitives",
            "basis_projection",
            "factorized_projection",
            "semantic_transport_transform",
        ],
        "deltanet_state_and_input_projection": ["recurrent_state_transition", "qgemv"],
        "norm_add_epilogues": ["reductions"],
        "lm_head_and_sampling": ["qgemv", "reductions"],
        "command_buffer_graph": ["semantic_transport_transform"],
    },
}

ENGINE_SPECS: list[dict[str, Any]] = [
    {
        "name": "qgemv",
        "dtype_contract": {
            "codes": "int8[M,K] symmetric signed weight_bits",
            "scales": "float32[M,K/group_size]",
            "x": "float32[K]",
            "y": "float32[M]",
            "weight_bits": "1..8",
            "group_size": "positive divisor of K",
        },
        "associativity": "sequential_left_to_right over K of (dequant(code)*x)",
        "scheme_recovered": "symmetric_signed_4bit_group64_with_fp16_scales",
    },
    {
        "name": "lowbit_gemm",
        "dtype_contract": {
            "codes": "int8[M,K]",
            "scales": "float32[M,K/group_size]",
            "b": "float32[K,N]",
            "c": "float32[M,N]",
        },
        "associativity": "sequential_left_to_right over K for each (m,n)",
        "scheme_recovered": "packed low-bit GEMV lifted to matrix-matrix; same dequant",
    },
    {
        "name": "basis_projection",
        "dtype_contract": {"basis": "float32[R,D]", "x": "float32[D]", "p": "float32[R]"},
        "associativity": "sequential_left_to_right over D",
        "scheme_recovered": "shared_bf16_basis / representation_library.shared_basis",
    },
    {
        "name": "factorized_projection",
        "dtype_contract": {
            "u": "float32[M,R]",
            "v": "float32[R,D]",
            "x": "float32[D]",
            "y": "float32[M]",
        },
        "associativity": "inner-first: sequential over D then sequential over R; never materialize U@V",
        "scheme_recovered": "representation_library.low_rank",
    },
    {
        "name": "codebook_arithmetic",
        "dtype_contract": {
            "codebook": "float32[card,sub] or float32[S,card,sub]",
            "indices": "int32[M,nchunk] or int32[M,nchunk,S]",
            "x": "float32[nchunk*sub]",
            "y": "float32[M]",
        },
        "associativity": "sequential over (chunk, subspace, sub)",
        "scheme_recovered": "C4 fused_adc_pq_matvec / gravity_pq; NF4 residual codebook",
    },
    {
        "name": "dictionary_arithmetic",
        "dtype_contract": {
            "dictionary": "float32[D,n_atoms]",
            "atom_ids": "int32[K]",
            "coeffs": "float32[K]",
            "y": "float32[D]",
        },
        "associativity": "sequential over selected atoms in given order",
        "scheme_recovered": "ngram lookup/compositional generator; dictionary execution",
    },
    {
        "name": "sparse_residual",
        "dtype_contract": {
            "partial": "float32[D]",
            "indices": "int32[nnz]",
            "values": "float32[nnz]",
            "y": "float32[D]",
        },
        "associativity": "sequential over residual terms; collisions add in given order",
        "scheme_recovered": "binary_sparse_residual / NF4 residual on a shared-basis partial",
    },
    {
        "name": "routed_expert_accumulate",
        "dtype_contract": {
            "expert_outputs": "float32[E,D]",
            "topk_ids": "int32[K]",
            "scales": "float32[K]",
            "y": "float32[D]",
        },
        "associativity": "sequential over given top-K order (not resorted by expert id)",
        "scheme_recovered": "Flash router_topk_and_gather / routed_plus_shared_expert",
    },
    {
        "name": "recurrent_state_transition",
        "dtype_contract": {
            "state": "float32[K,V]",
            "query": "float32[K]",
            "key": "float32[K]",
            "value": "float32[V]",
            "decay": "float32",
            "beta": "float32",
            "out_state": "float32[K,V]",
            "output": "float32[V]",
        },
        "associativity": "sequential over ki for kv and output reductions",
        "scheme_recovered": "qwen38_gated_delta_decode_vi (DeltaNet gated-delta)",
    },
    {
        "name": "attention_primitives",
        "dtype_contract": {
            "q": "float32[Tq,D]",
            "k": "float32[Tk,D]",
            "v": "float32[Tk,Dv]",
            "scale": "float32",
            "scores": "float32[Tq,Tk]",
            "weights": "float32[Tq,Tk]",
            "output": "float32[Tq,Dv]",
        },
        "associativity": "score: sequential over D; softmax: sequential max then sequential exp-sum; values: sequential over Tk",
        "scheme_recovered": "separable score/softmax/value; Flash sparse_attention reductions",
    },
    {
        "name": "reductions",
        "dtype_contract": {"x": "int64-or-float32[N]", "y": "scalar same family"},
        "associativity": "mode=sequential: left fold; mode=tree: midpoint split. Integer + matches; float32 may not.",
        "scheme_recovered": "ACCELERATOR reduce tails (serial vs tree) as a functional fact, not a timed one",
    },
    {
        "name": "semantic_transport_transform",
        "dtype_contract": {
            "payload": "float32 or integer tensor",
            "pipeline": "ordered subset of quantize|transpose|reduce|digest|pack",
            "digest": "sha256 hex of little-endian C-contiguous bytes",
        },
        "associativity": "reduce step is sequential over the last axis; digest is of that reduced tensor",
        "scheme_recovered": "fusion_isa REDUCE/DIGEST/COPY + preboard activations/partial-reductions transport",
    },
]


def _input_dtypes(engine: str) -> dict[str, Any]:
    return {
        "qgemv": {"codes": np.int8, "scales": np.float32, "x": np.float32},
        "lowbit_gemm": {"codes": np.int8, "scales": np.float32, "b": np.float32},
        "basis_projection": {"basis": np.float32, "x": np.float32},
        "factorized_projection": {"u": np.float32, "v": np.float32, "x": np.float32},
        "codebook_arithmetic": {"codebook": np.float32, "indices": np.int32, "x": np.float32},
        "dictionary_arithmetic": {
            "dictionary": np.float32,
            "atom_ids": np.int32,
            "coeffs": np.float32,
        },
        "sparse_residual": {"partial": np.float32, "indices": np.int32, "values": np.float32},
        "routed_expert_accumulate": {
            "expert_outputs": np.float32,
            "topk_ids": np.int32,
            "scales": np.float32,
        },
        "recurrent_state_transition": {
            "state": np.float32,
            "query": np.float32,
            "key": np.float32,
            "value": np.float32,
        },
        "attention_primitives": {"q": np.float32, "k": np.float32, "v": np.float32},
        "reductions": {"x": None},
        "semantic_transport_transform": {"payload": np.float32},
    }[engine]


def _hydrate(engine: str, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    dtypes = _input_dtypes(engine)
    out: dict[str, Any] = dict(params)
    for key, val in inputs.items():
        dt = dtypes.get(key)
        if engine == "reductions" and key == "x":
            out[key] = np.asarray(val, dtype=np.int32 if params.get("int_mode") else np.float32)
        elif dt is None:
            out[key] = val
        else:
            out[key] = np.asarray(val, dtype=dt)
    out.pop("int_mode", None)
    return out


def run_vector(vec: dict[str, Any]) -> Any:
    fn = ENGINE_FNS[vec["engine"]]
    kwargs = _hydrate(vec["engine"], vec["inputs"], vec["params"])
    return fn(**kwargs)


def bit_exact_equal(a: Any, b: Any) -> bool:
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return False
        return all(bit_exact_equal(a[k], b[k]) for k in sorted(a))
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, (list, tuple)):
        b_arr = np.asarray(b)
        a_arr = np.asarray(a)
        if a_arr.dtype == object or b_arr.dtype == object:
            return list(a) == list(b)
        return np.array_equal(a_arr, b_arr)
    return np.array_equal(np.asarray(a), np.asarray(b))


def _jsonable(node: Any) -> Any:
    if isinstance(node, dict):
        return {str(k): _jsonable(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_jsonable(v) for v in node]
    if isinstance(node, np.ndarray):
        return _jsonable(node.tolist())
    if isinstance(node, (np.floating, np.integer)):
        return node.item()
    return node


# VECTORS: constructed literals (seed 0) plus one seeded integer draw (seed 20260829).
# Expected values are hardcoded so an engine change fails this table, not just the
# independent golden. Associativity is stated per row because a spatial rewrite
# that reorders a reduction is a real numerical difference.
VECTORS: list[dict[str, Any]] = [
    {
        "id": "qgemv_hand",
        "engine": "qgemv",
        "seed": 0,
        "associativity": "sequential_left_to_right over K",
        "params": {"weight_bits": 4, "group_size": 4},
        "inputs": {
            "codes": [[1, -1, 2, 0], [3, 1, -2, 1]],
            "scales": [[2.0], [0.5]],
            "x": [1.0, 2.0, 3.0, 4.0],
        },
        "expected": [10.0, 1.5],
        "note": "one group per row; integer-valued so float32 is exact",
    },
    {
        "id": "qgemv_grouped",
        "engine": "qgemv",
        "seed": 0,
        "associativity": "sequential_left_to_right over K",
        "params": {"weight_bits": 4, "group_size": 2},
        "inputs": {
            "codes": [[1, 1, 2, 2]],
            "scales": [[3.0, 4.0]],
            "x": [1.0, 1.0, 1.0, 1.0],
        },
        "expected": [22.0],
        "note": "two groups; scale changes at k=2",
    },
    {
        "id": "qgemv_1bit_sign",
        "engine": "qgemv",
        "seed": 0,
        "associativity": "sequential_left_to_right over K",
        "params": {"weight_bits": 1, "group_size": 4},
        "inputs": {
            "codes": [[1, -1, 1, -1]],
            "scales": [[2.0]],
            "x": [1.0, 1.0, 1.0, 1.0],
        },
        "expected": [0.0],
        "note": "1-bit is a sign code, not deletion (tools/headless/test_lowbit_codec.py)",
    },
    {
        "id": "qgemv_seeded",
        "engine": "qgemv",
        "seed": 20260829,
        "associativity": "sequential_left_to_right over K",
        "params": {"weight_bits": 4, "group_size": 4},
        "inputs": {
            "codes": [
                [-6, 1, 4, -6, 3, 2, -6, -6],
                [6, -3, -1, -6, 2, -1, 4, 2],
                [5, -5, -2, 4, 0, 3, 6, -1],
            ],
            "scales": [[3.0, 2.0], [1.0, 3.0], [2.0, 3.0]],
            "x": [0.0, -1.0, 0.0, -2.0, 0.0, 0.0, 1.0, -3.0],
        },
        "expected": [57.0, 9.0, 21.0],
        "note": "RandomState(20260829) integer draw, still exact in float32",
    },
    {
        "id": "lowbit_gemm_hand",
        "engine": "lowbit_gemm",
        "seed": 0,
        "associativity": "sequential_left_to_right over K",
        "params": {"weight_bits": 4, "group_size": 4},
        "inputs": {
            "codes": [[1, -1, 2, 0], [3, 1, -2, 1]],
            "scales": [[2.0], [0.5]],
            "b": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 2.0]],
        },
        "expected": [[6.0, 2.0], [0.5, 0.5]],
    },
    {
        "id": "lowbit_gemm_seeded",
        "engine": "lowbit_gemm",
        "seed": 20260829,
        "associativity": "sequential_left_to_right over K",
        "params": {"weight_bits": 4, "group_size": 4},
        "inputs": {
            "codes": [
                [-6, 1, 4, -6, 3, 2, -6, -6],
                [6, -3, -1, -6, 2, -1, 4, 2],
                [5, -5, -2, 4, 0, 3, 6, -1],
            ],
            "scales": [[3.0, 2.0], [1.0, 3.0], [2.0, 3.0]],
            "b": [
                [2.0, -1.0, 2.0],
                [0.0, -2.0, 2.0],
                [2.0, 1.0, 1.0],
                [2.0, 0.0, 0.0],
                [-1.0, -1.0, 2.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 2.0, -2.0],
                [-2.0, -1.0, 1.0],
            ],
        },
        "expected": [[-22.0, 6.0, 6.0], [-29.0, 11.0, -1.0], [7.0, 45.0, -43.0]],
    },
    {
        "id": "basis_projection_hand",
        "engine": "basis_projection",
        "seed": 0,
        "associativity": "sequential_left_to_right over D",
        "params": {},
        "inputs": {
            "basis": [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
            "x": [2.0, 3.0, 4.0, 5.0],
        },
        "expected": [5.0, 9.0],
    },
    {
        "id": "factorized_projection_hand",
        "engine": "factorized_projection",
        "seed": 0,
        "associativity": "inner-first sequential (V then U)",
        "params": {},
        "inputs": {
            "u": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "v": [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            "x": [1.0, 2.0, 3.0, 4.0],
        },
        "expected": [4.0, 6.0, 10.0],
    },
    {
        "id": "codebook_arithmetic_vector",
        "engine": "codebook_arithmetic",
        "seed": 0,
        "associativity": "sequential over (chunk, sub)",
        "params": {},
        "inputs": {
            "codebook": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]],
            "indices": [[0, 1], [2, 3]],
            "x": [3.0, 4.0, 5.0, 6.0],
        },
        "expected": [9.0, 17.0],
    },
    {
        "id": "codebook_arithmetic_nf4_product",
        "engine": "codebook_arithmetic",
        "seed": 0,
        "associativity": "sequential over (chunk, subspace, sub)",
        "params": {},
        "inputs": {
            "codebook": [
                [[-1.0], [0.0], [1.0]],
                [[0.5], [-0.5], [0.25]],
            ],
            "indices": [[[0, 2], [2, 1]]],
            "x": [4.0, 8.0],
        },
        "expected": [1.0],
        "note": "product codebook S=2, sub=1: (-1)*4+(0.25)*4 + (1)*8+(-0.5)*8 = 1",
    },
    {
        "id": "dictionary_arithmetic_hand",
        "engine": "dictionary_arithmetic",
        "seed": 0,
        "associativity": "sequential over selected atoms in given order",
        "params": {},
        "inputs": {
            "dictionary": [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            "atom_ids": [0, 2],
            "coeffs": [3.0, 4.0],
        },
        "expected": [11.0, 0.0, 4.0, 3.0],
    },
    {
        "id": "sparse_residual_collision",
        "engine": "sparse_residual",
        "seed": 0,
        "associativity": "sequential over residual terms; index 1 is hit twice",
        "params": {},
        "inputs": {
            "partial": [10.0, 20.0, 30.0, 40.0],
            "indices": [1, 3, 1],
            "values": [5.0, -2.0, 1.0],
        },
        "expected": [10.0, 26.0, 30.0, 38.0],
    },
    {
        "id": "routed_expert_accumulate_hand",
        "engine": "routed_expert_accumulate",
        "seed": 0,
        "associativity": "sequential over given top-K order [2,0], not sorted ids",
        "params": {},
        "inputs": {
            "expert_outputs": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "topk_ids": [2, 0],
            "scales": [0.5, 2.0],
        },
        "expected": [4.5, 7.0],
    },
    {
        "id": "recurrent_state_transition_hand",
        "engine": "recurrent_state_transition",
        "seed": 0,
        "associativity": "sequential over ki",
        "params": {"decay": 1.0, "beta": 1.0},
        "inputs": {
            "state": [[1.0, 0.0], [0.0, 1.0]],
            "query": [0.0, 1.0],
            "key": [1.0, 0.0],
            "value": [2.0, 3.0],
        },
        "expected": {"state": [[2.0, 3.0], [0.0, 1.0]], "output": [0.0, 1.0]},
    },
    {
        "id": "attention_primitives_hand",
        "engine": "attention_primitives",
        "seed": 0,
        "associativity": "sequential over D, sequential softmax, sequential over Tk",
        "params": {"scale": 1.0},
        "inputs": {
            "q": [[1.0, 0.0]],
            "k": [[1.0, 0.0], [0.0, 1.0]],
            "v": [[2.0, 3.0], [4.0, 5.0]],
        },
        "expected": {
            "scores": [[1.0, 0.0]],
            "weights": [[0.7310585975646973, 0.2689414322376251]],
            "output": [[2.5378828048706055, 3.5378828048706055]],
        },
    },
    {
        "id": "reductions_int_both_modes",
        "engine": "reductions",
        "seed": 0,
        "associativity": "int64 + is associative: sequential and tree match",
        "params": {"mode": "sequential", "int_mode": True},
        "inputs": {"x": [3, 1, 4, 2]},
        "expected": 10,
        "paired_tree_expected": 10,
    },
    {
        "id": "reductions_float_sequential",
        "engine": "reductions",
        "seed": 0,
        "associativity": "sequential left-to-right float32",
        "params": {"mode": "sequential"},
        "inputs": {"x": [1.0, 5.960464477539063e-08, 5.960464477539063e-08]},
        "expected": 1.0,
        "note": "1.0 + 2^-24 + 2^-24 loses both ulps sequentially",
    },
    {
        "id": "reductions_float_tree",
        "engine": "reductions",
        "seed": 0,
        "associativity": "tree midpoint-split float32",
        "params": {"mode": "tree"},
        "inputs": {"x": [1.0, 5.960464477539063e-08, 5.960464477539063e-08]},
        "expected": 1.0000001192092896,
        "note": "tree pairs the two ulps first; 1.0 + 2^-23 is the next float32 after 1.0",
    },
    {
        "id": "semantic_transport_qtrd",
        "engine": "semantic_transport_transform",
        "seed": 0,
        "associativity": "quantize sequential amax; reduce sequential last axis",
        "params": {"pipeline": ["quantize", "transpose", "reduce", "digest"], "quant_bits": 8},
        "inputs": {"payload": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 9.0]]},
        "expected": {
            "tensor": [85, 113, 141, 183],
            "scale": 0.07086614519357681,
            "digest": "2dd32742d69eed1efc2bb463da0b46aaff58abbf6252dd62d823e3402793988f",
            "packed": None,
            "pipeline": ["quantize", "transpose", "reduce", "digest"],
        },
    },
    {
        "id": "semantic_transport_pack4",
        "engine": "semantic_transport_transform",
        "seed": 0,
        "associativity": "n/a (bit pack, no reduction)",
        "params": {"pipeline": ["pack"], "pack_bits": 4},
        "inputs": {"payload": [1, -1, 2, 0, 3, 1, -2, 1]},
        "expected": {
            "tensor": [1, -1, 2, 0, 3, 1, -2, 1],
            "scale": None,
            "digest": None,
            "packed": [104, 121, 138, 133],
            "pipeline": ["pack"],
        },
        "note": "payload already int-valued codes; pack nibble-packs with bound=7",
    },
]


def vector_by_id(vid: str) -> dict[str, Any]:
    for v in VECTORS:
        if v["id"] == vid:
            return v
    raise KeyError(vid)


def _head_has(rel: str) -> bool:
    p = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{rel}"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return p.returncode == 0


def _selftest_vectors() -> list[dict[str, Any]]:
    rows = []
    for vec in VECTORS:
        got = run_vector(vec)
        if not bit_exact_equal(got, vec["expected"]):
            raise AssertionError(
                f"{vec['id']}: engine output != hardcoded expected\n"
                f"  got={_jsonable(got)}\n  expected={vec['expected']}"
            )
        rows.append(
            {
                "id": vec["id"],
                "engine": vec["engine"],
                "seed": vec["seed"],
                "associativity": vec["associativity"],
                "match": True,
            }
        )
    seq = run_vector(vector_by_id("reductions_float_sequential"))
    tree = run_vector(vector_by_id("reductions_float_tree"))
    if bit_exact_equal(seq, tree):
        raise AssertionError("float32 sequential and tree reductions must diverge on the ulp vector")
    return rows


RECOVERED_PATHS = (
    "receipts/headless/FLASH_NEXT_FPGA_ORGAN_MAP.json",
    "receipts/headless/QWEN27_FPGA_ORGAN_MAP.json",
    "receipts/headless/HCLI_FPGA_PREBOARD.json",
    "hcli/agentos/fpga_preboard.py",
    "tools/accelerator/gemm.py",
    "tools/accelerator/kernel_forge.py",
    "tools/accelerator/gravity_native.py",
    "tools/accelerator/fusion_isa.py",
    "tools/headless/c4codebook_design.py",
    "tools/headless/representation_library.py",
    "tools/headless/deltanet_organ.py",
    "tools/headless/test_lowbit_codec.py",
    "receipts/headless/FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json",
    "receipts/headless/FLASH_NEXT_NOETIC_EXECUTABLE.json",
    "receipts/headless/C4CODEBOOK_DESIGN.json",
    "receipts/headless/REPRESENTATION_LIBRARY.json",
    "crates/hawking-core/shaders/qwen38_device_activations.metal",
    "crates/hawking-core/shaders/gravity_pq.metal",
    "receipts/headless/FLASH_COMPLETE_V2.nr.json",
    "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json",
)


def build() -> Any:
    vector_rows = _selftest_vectors()
    recovered = []
    for rel in RECOVERED_PATHS:
        recovered.append(
            {
                "path": rel,
                "present_in_head": _head_has(rel),
                "materialized": (REPO / rel).exists(),
            }
        )
    present = {r["path"]: r["present_in_head"] for r in recovered}
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Bit-exact functional references for every arithmetic engine a spatial "
            "accelerator needs to execute a Hawking model. Golden models only. "
            "Not an FPGA backend; no HDL; no cycle count; no throughput."
        ),
        "civilization_boundary": (
            "FPGA is Accelerator / Physical Compiler / Fusion. Not its own "
            "civilization. Five eras, three odysseys. Sidecar has no GPU and "
            "produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE."
        ),
        "evidence_class": "STATIC_ONLY",
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "engines": ENGINE_SPECS,
        "engine_names": [s["name"] for s in ENGINE_SPECS],
        "organ_engine_map": ORGAN_ENGINE_MAP,
        "nf4_codebook": [float(v) for v in NF4_CODEBOOK.tolist()],
        "nf4_codebook_source": (
            "receipts/headless/FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json "
            "candidates.shared_bf16_basis_nf4_residual.residual_codebook"
        ),
        "vectors": [_jsonable(v) for v in VECTORS],
        "selftest_vectors": vector_rows,
        "associativity_policy": (
            "Every reduction names sequential_left_to_right or tree_midpoint_split. "
            "A spatial implementation that reorders a float32 reduction is a "
            "different number and must either match this golden's declared order "
            "or carry an explicit associativity waiver. Integer + is associative."
        ),
        "flash_chosen_representation": {
            "selected_candidate": "independent_q4_g64",
            "quality_alternate": "shared_bf16_basis_nf4_residual",
            "source": "receipts/headless/FLASH_NEXT_NOETIC_EXECUTABLE.json",
            "note": (
                "FLASH_COMPLETE_V2.nr.json is not in HEAD; this is the closest "
                "sealed representation record."
            ),
        },
        "recovered_implementation": {
            "paths": recovered,
            "summary": (
                "Organ maps and the FPGA preboard name organs and a mock link "
                "simulator; they do not implement arithmetic. tools/accelerator/"
                "gemm.py is a Metal tiled GEMM that loses to MLX and is not a "
                "NumPy golden. kernel_forge.py searches kernel variants; it does "
                "not define engine contracts. gravity_native.py has the sequential "
                "q4 group-64 dequant-matvec used as the qgemv width contract. "
                "C4 codebook design names fused ADC. DeltaNet gated-delta lives in "
                "the qwen38 Metal shader. Representation library lists families "
                "(shared_basis, low_rank, binary_sparse_residual, conventional_low_bit) "
                "without executable goldens. No tools/future/fpga_engines.py existed."
            ),
            "adequate_existing_golden": False,
            "flash_complete_v2_nr_present_in_head": present.get(
                "receipts/headless/FLASH_COMPLETE_V2.nr.json", False
            ),
            "architecture_atlas_present_in_head": present.get(
                "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json", False
            ),
        },
        "gaps_closed": [
            "twelve engine functional references with explicit dtype/width contracts",
            "VECTORS table with hardcoded expected outputs and declared associativity",
            "organ-to-engine map recovered from Flash and Qwen27 FPGA organ maps",
            "NF4 codebook constant recovered from the Flash representation experiment",
            "tree vs sequential float32 divergence recorded as a load-bearing fact",
        ],
        "negative_findings": [
            "receipts/headless/FLASH_COMPLETE_V2.nr.json is not a blob at HEAD; "
            "representation recovered instead from FLASH_NEXT_NOETIC_EXECUTABLE.json "
            "and FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json",
            "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json is not a blob at HEAD "
            "(frontier F012 probed it on a machine where it was materialized)",
            "no existing NumPy bit-exact golden for these twelve engines; Metal kernels "
            "and design receipts are not a substitute",
            "cycle counts, throughput, HBM bandwidth, and board timing remain UNKNOWN "
            "because this sidecar has no FPGA and no GPU lease",
            "this worktree is a sparse checkout; recovered files were read with git show, "
            "not from a materialized working tree",
        ],
        "hardware_required_to_measure": [
            "cycle_count",
            "throughput",
            "board_timing",
            "hbm_bandwidth",
            "gpu_token_timing",
        ],
        "not_an_fpga_backend": True,
        "emits_hdl": False,
    }
    return write_receipt(RECEIPT, doc, "tools/future/fpga_engines.py")


def selftest() -> Any:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = selftest()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
