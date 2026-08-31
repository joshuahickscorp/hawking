"""Independent goldens for tools/future/fpga_engines.py.

The engine module is not allowed to be its own oracle. Every bit-exact check
below compares the engine to a second implementation written here, with a
different loop structure, plus the hardcoded VECTORS table. A golden checked
only against itself is worthless.

A negative control perturbs one weight and proves the equality assertion
actually fires.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools.future import fpga_engines as fe
from tools.future._common import RECEIPTS


def gadd(a, b):
    return np.add(np.float32(a), np.float32(b), dtype=np.float32)


def gmul(a, b):
    return np.multiply(np.float32(a), np.float32(b), dtype=np.float32)


def gsub(a, b):
    return np.subtract(np.float32(a), np.float32(b), dtype=np.float32)


def gdiv(a, b):
    return np.divide(np.float32(a), np.float32(b), dtype=np.float32)


# ---------------------------------------------------------------------------
# Independent goldens. Materialize, fold, or recurse — do not copy the engine
# fused loops.
# ---------------------------------------------------------------------------


def golden_qgemv(codes, scales, x, *, weight_bits, group_size):
    """Dequantize into an explicit dense matrix, then left-fold the row dot."""
    del weight_bits
    codes = np.asarray(codes, dtype=np.int8)
    scales = np.asarray(scales, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    m, k = codes.shape
    dense = np.empty((m, k), dtype=np.float32)
    groups = k // group_size
    g = 0
    while g < groups:
        sl = slice(g * group_size, (g + 1) * group_size)
        for row in range(m):
            s = np.float32(scales[row, g])
            dense[row, sl] = np.multiply(codes[row, sl].astype(np.float32), s, dtype=np.float32)
        g += 1
    y = np.empty(m, dtype=np.float32)
    for row in range(m):
        acc = np.float32(0)
        k_i = 0
        while k_i < k:
            acc = gadd(acc, gmul(dense[row, k_i], x[k_i]))
            k_i += 1
        y[row] = acc
    return y


def golden_lowbit_gemm(codes, scales, b, *, weight_bits, group_size):
    """Dequantize A, then for each output column left-fold against that column."""
    codes = np.asarray(codes, dtype=np.int8)
    scales = np.asarray(scales, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    m, k = codes.shape
    n = b.shape[1]
    cols = []
    for col in range(n):
        cols.append(
            golden_qgemv(
                codes,
                scales,
                b[:, col],
                weight_bits=weight_bits,
                group_size=group_size,
            )
        )
    out = np.empty((m, n), dtype=np.float32)
    for col, y in enumerate(cols):
        out[:, col] = y
    return out


def golden_basis_projection(basis, x):
    basis = np.asarray(basis, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    out = []
    for row in basis:
        acc = np.float32(0)
        for w, xi in zip(row.tolist(), x.tolist()):
            acc = gadd(acc, gmul(w, xi))
        out.append(acc)
    return np.asarray(out, dtype=np.float32)


def golden_factorized_projection(u, v, x):
    """Two explicit matvecs via python lists. Inner first."""
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    hidden = golden_basis_projection(v, x)
    return golden_basis_projection(u, hidden)


def golden_codebook_arithmetic(codebook, indices, x):
    """Materialize each looked-up codeword, then fold. Opposite of fused ADC."""
    codebook = np.asarray(codebook, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int32)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if codebook.ndim == 2:
        codebook = codebook[None, ...]
        indices = indices[..., None]
    n_s, _card, sub = codebook.shape
    m, nchunk, _s = indices.shape
    y = np.empty(m, dtype=np.float32)
    for row in range(m):
        pieces = []
        for chunk in range(nchunk):
            chunk_x = x[chunk * sub : (chunk + 1) * sub]
            for sub_i in range(n_s):
                pieces.append((codebook[sub_i, int(indices[row, chunk, sub_i])], chunk_x))
        acc = np.float32(0)
        for entry, chunk_x in pieces:
            for a, b in zip(entry.tolist(), chunk_x.tolist()):
                acc = gadd(acc, gmul(a, b))
        y[row] = acc
    return y


def golden_dictionary_arithmetic(dictionary, atom_ids, coeffs):
    dictionary = np.asarray(dictionary, dtype=np.float32)
    atom_ids = np.asarray(atom_ids, dtype=np.int32).reshape(-1)
    coeffs = np.asarray(coeffs, dtype=np.float32).reshape(-1)
    y = [np.float32(0)] * dictionary.shape[0]
    for atom, c in zip(atom_ids.tolist(), coeffs.tolist()):
        col = dictionary[:, int(atom)].tolist()
        y = [gadd(yi, gmul(c, cj)) for yi, cj in zip(y, col)]
    return np.asarray(y, dtype=np.float32)


def golden_sparse_residual(partial, indices, values):
    y = [np.float32(v) for v in np.asarray(partial, dtype=np.float32).reshape(-1).tolist()]
    for i, v in zip(
        np.asarray(indices, dtype=np.int32).tolist(),
        np.asarray(values, dtype=np.float32).tolist(),
    ):
        y[int(i)] = gadd(y[int(i)], v)
    return np.asarray(y, dtype=np.float32)


def golden_routed_expert_accumulate(expert_outputs, topk_ids, scales):
    expert_outputs = np.asarray(expert_outputs, dtype=np.float32)
    y = [np.float32(0)] * expert_outputs.shape[1]
    for e, s in zip(
        np.asarray(topk_ids, dtype=np.int32).tolist(),
        np.asarray(scales, dtype=np.float32).tolist(),
    ):
        row = expert_outputs[int(e)].tolist()
        y = [gadd(yi, gmul(s, rj)) for yi, rj in zip(y, row)]
    return np.asarray(y, dtype=np.float32)


def golden_recurrent_state_transition(state, query, key, value, decay, beta):
    s = np.asarray(state, dtype=np.float32)
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    k = np.asarray(key, dtype=np.float32).reshape(-1)
    v = np.asarray(value, dtype=np.float32).reshape(-1)
    d = np.float32(decay)
    b = np.float32(beta)
    kdim, vdim = s.shape
    decayed = np.multiply(s, d, dtype=np.float32)
    kv = []
    for j in range(vdim):
        acc = np.float32(0)
        for i in range(kdim):
            acc = gadd(acc, gmul(decayed[i, j], k[i]))
        kv.append(acc)
    delta = [gmul(gsub(v[j], kv[j]), b) for j in range(vdim)]
    s_new = np.empty_like(s)
    for i in range(kdim):
        for j in range(vdim):
            s_new[i, j] = gadd(decayed[i, j], gmul(k[i], delta[j]))
    out = []
    for j in range(vdim):
        acc = np.float32(0)
        for i in range(kdim):
            acc = gadd(acc, gmul(s_new[i, j], q[i]))
        out.append(acc)
    return {"state": s_new, "output": np.asarray(out, dtype=np.float32)}


def golden_attention_primitives(q, k, v, scale):
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    tq, d = q.shape
    tk = k.shape[0]
    dv = v.shape[1]
    scores = np.empty((tq, tk), dtype=np.float32)
    for i in range(tq):
        for j in range(tk):
            acc = np.float32(0)
            for t in range(d):
                acc = gadd(acc, gmul(q[i, t], k[j, t]))
            scores[i, j] = gdiv(acc, scale)
    weights = np.empty_like(scores)
    for i in range(tq):
        m = scores[i, 0]
        for j in range(1, tk):
            m = np.maximum(m, scores[i, j])
        exps = [np.exp(gsub(scores[i, j], m)) for j in range(tk)]
        tot = np.float32(0)
        for e in exps:
            tot = gadd(tot, e)
        for j, e in enumerate(exps):
            weights[i, j] = gdiv(e, tot)
    out = np.empty((tq, dv), dtype=np.float32)
    for i in range(tq):
        for dim in range(dv):
            acc = np.float32(0)
            for j in range(tk):
                acc = gadd(acc, gmul(weights[i, j], v[j, dim]))
            out[i, dim] = acc
    return {"scores": scores, "weights": weights, "output": out}


def golden_sequential_reduce(x):
    a = np.asarray(x).reshape(-1)
    if a.size == 0:
        return np.int64(0) if np.issubdtype(a.dtype, np.integer) else np.float32(0)
    if np.issubdtype(a.dtype, np.integer):
        acc = 0
        for v in a.tolist():
            acc = acc + int(v)
        return np.int64(acc)
    acc = np.float32(0)
    for v in a.tolist():
        acc = gadd(acc, v)
    return acc


def golden_tree_reduce(x):
    a = np.asarray(x).reshape(-1)
    integer = bool(np.issubdtype(a.dtype, np.integer))
    vals = list(a.tolist())
    if not vals:
        return np.int64(0) if integer else np.float32(0)

    def rec(xs):
        if len(xs) == 1:
            return np.int64(xs[0]) if integer else np.float32(xs[0])
        mid = len(xs) // 2
        left, right = rec(xs[:mid]), rec(xs[mid:])
        if integer:
            return np.int64(left) + np.int64(right)
        return gadd(left, right)

    return rec(vals)


def golden_reductions(x, *, mode, int_mode=False):
    del int_mode
    arr = np.asarray(x)
    if mode == "sequential":
        return golden_sequential_reduce(arr)
    if mode == "tree":
        return golden_tree_reduce(arr)
    raise ValueError(mode)


def golden_semantic_transport_transform(payload, *, pipeline, quant_bits=8, pack_bits=4):
    tensor = np.asarray(payload)
    scale = None
    digest = None
    packed = None
    for step in pipeline:
        if step == "quantize":
            bound = fe.signed_bound(quant_bits)
            amax = np.float32(0)
            for v in np.asarray(tensor, dtype=np.float32).reshape(-1).tolist():
                amax = np.maximum(amax, np.abs(np.float32(v)))
            scale = np.float32(1) if amax == np.float32(0) else gdiv(amax, np.float32(bound))
            codes = np.empty(np.asarray(tensor).shape, dtype=np.int8)
            src = np.asarray(tensor, dtype=np.float32)
            for idx in np.ndindex(src.shape):
                q = np.clip(np.rint(gdiv(src[idx], scale)), -bound, bound)
                codes[idx] = np.int8(q)
            tensor = codes
        elif step == "transpose":
            tensor = np.ascontiguousarray(np.asarray(tensor).T)
        elif step == "reduce":
            a = np.asarray(tensor)
            if a.ndim == 1:
                tensor = golden_sequential_reduce(a)
            else:
                dt = np.int64 if np.issubdtype(a.dtype, np.integer) else np.float32
                out = np.empty(a.shape[0], dtype=dt)
                for i, row in enumerate(a):
                    out[i] = golden_sequential_reduce(row)
                tensor = out
        elif step == "digest":
            a = np.ascontiguousarray(tensor)
            if a.dtype == np.float32:
                blob = a.astype("<f4").tobytes()
            elif a.dtype == np.int8:
                blob = a.astype("<i1").tobytes()
            elif a.dtype == np.int32:
                blob = a.astype("<i4").tobytes()
            elif a.dtype == np.int64:
                blob = a.astype("<i8").tobytes()
            elif a.dtype == np.uint8:
                blob = a.tobytes()
            else:
                raise ValueError(a.dtype)
            digest = hashlib.sha256(blob).hexdigest()
        elif step == "pack":
            codes = np.asarray(tensor, dtype=np.int8).reshape(-1)
            bound = fe.signed_bound(pack_bits)
            stored = [int(c) + bound for c in codes.tolist()]
            packed = np.asarray(
                [stored[i] | (stored[i + 1] << 4) for i in range(0, len(stored), 2)],
                dtype=np.uint8,
            )
        else:
            raise ValueError(step)
    return {
        "tensor": tensor,
        "scale": scale,
        "digest": digest,
        "packed": packed,
        "pipeline": list(pipeline),
    }


GOLDENS = {
    "qgemv": golden_qgemv,
    "lowbit_gemm": golden_lowbit_gemm,
    "basis_projection": golden_basis_projection,
    "factorized_projection": golden_factorized_projection,
    "codebook_arithmetic": golden_codebook_arithmetic,
    "dictionary_arithmetic": golden_dictionary_arithmetic,
    "sparse_residual": golden_sparse_residual,
    "routed_expert_accumulate": golden_routed_expert_accumulate,
    "recurrent_state_transition": golden_recurrent_state_transition,
    "attention_primitives": golden_attention_primitives,
    "reductions": golden_reductions,
    "semantic_transport_transform": golden_semantic_transport_transform,
}


def _golden_of(vec):
    fn = GOLDENS[vec["engine"]]
    kwargs = fe._hydrate(vec["engine"], vec["inputs"], vec["params"])
    return fn(**kwargs)


# ---------------------------------------------------------------------------
# Receipt / school surface
# ---------------------------------------------------------------------------


def test_build_emits_sealed_receipt():
    out = fe.build()
    assert out.parent == RECEIPTS
    assert out.name == "FPGA_ENGINE_SCHOOL.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == "hawking.future.fpga_engines.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    import hashlib as _h

    assert doc["seal_sha256"] == _h.sha256(blob).hexdigest()
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["not_an_fpga_backend"] is True
    assert doc["emits_hdl"] is False
    assert set(doc["engine_names"]) == set(fe.ENGINE_FNS)
    assert "tps" not in doc
    assert "gpu_ns" not in doc


def test_twelve_engines_each_have_a_vector():
    covered = {v["engine"] for v in fe.VECTORS}
    assert covered == set(fe.ENGINE_FNS)
    assert len(fe.ENGINE_SPECS) == 12
    for spec in fe.ENGINE_SPECS:
        assert spec["associativity"]
        assert spec["dtype_contract"]


def test_every_vector_matches_engine_and_independent_golden():
    for vec in fe.VECTORS:
        got = fe.run_vector(vec)
        gold = _golden_of(vec)
        assert fe.bit_exact_equal(got, vec["expected"]), vec["id"]
        assert fe.bit_exact_equal(got, gold), vec["id"]
        assert fe.bit_exact_equal(gold, vec["expected"]), vec["id"]


def test_tree_and_sequential_diverge_on_float32_ulps():
    xs = np.array([1.0, 2.0**-24, 2.0**-24], dtype=np.float32)
    seq = fe.sequential_reduce(xs)
    tree = fe.tree_reduce(xs)
    assert not np.array_equal(np.asarray(seq), np.asarray(tree))
    assert np.array_equal(np.asarray(seq), np.asarray(golden_sequential_reduce(xs)))
    assert np.array_equal(np.asarray(tree), np.asarray(golden_tree_reduce(xs)))
    ints = np.array([3, 1, 4, 2], dtype=np.int32)
    assert fe.sequential_reduce(ints) == fe.tree_reduce(ints) == np.int64(10)


def test_attention_pieces_are_separable():
    vec = fe.vector_by_id("attention_primitives_hand")
    kwargs = fe._hydrate(vec["engine"], vec["inputs"], vec["params"])
    scores = fe.attention_score(kwargs["q"], kwargs["k"], kwargs["scale"])
    weights = fe.attention_softmax(scores)
    out = fe.attention_weighted_values(weights, kwargs["v"])
    composed = fe.attention_primitives(kwargs["q"], kwargs["k"], kwargs["v"], kwargs["scale"])
    assert np.array_equal(scores, composed["scores"])
    assert np.array_equal(weights, composed["weights"])
    assert np.array_equal(out, composed["output"])


def test_nf4_codebook_recovered_from_flash_receipt():
    assert fe.NF4_CODEBOOK.shape == (16,)
    assert fe.NF4_CODEBOOK.dtype == np.float32
    assert fe.NF4_CODEBOOK[0] == np.float32(-1.0)
    assert fe.NF4_CODEBOOK[7] == np.float32(0.0)
    assert fe.NF4_CODEBOOK[15] == np.float32(1.0)


def test_one_bit_rejects_zero_codes():
    with pytest.raises(ValueError, match="sign code"):
        fe.qgemv(
            np.array([[0, 1, -1, 1]], dtype=np.int8),
            np.array([[1.0]], dtype=np.float32),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            weight_bits=1,
            group_size=4,
        )


def test_qgemv_rejects_codes_outside_symmetric_bound():
    with pytest.raises(ValueError, match="outside symmetric"):
        fe.qgemv(
            np.array([[8, 0, 0, 0]], dtype=np.int8),
            np.array([[1.0]], dtype=np.float32),
            np.ones(4, dtype=np.float32),
            weight_bits=4,
            group_size=4,
        )


def test_organ_map_covers_both_models():
    assert set(fe.ORGAN_ENGINE_MAP) == {"flash-next", "qwen27"}
    for engines in fe.ORGAN_ENGINE_MAP["flash-next"].values():
        assert set(engines) <= set(fe.ENGINE_FNS)
    for engines in fe.ORGAN_ENGINE_MAP["qwen27"].values():
        assert set(engines) <= set(fe.ENGINE_FNS)
    assert "qgemv" in fe.ORGAN_ENGINE_MAP["qwen27"]["mlp_gate_up_down"]
    assert "recurrent_state_transition" in fe.ORGAN_ENGINE_MAP["flash-next"]["deltanet_persistent_state"]


# ---------------------------------------------------------------------------
# Negative control: a guard nobody has watched fail is not a guard.
# ---------------------------------------------------------------------------


def test_negative_control_perturbed_weight_is_caught():
    vec = fe.vector_by_id("qgemv_hand")
    y = fe.run_vector(vec)
    y_gold = _golden_of(vec)
    assert np.array_equal(y, y_gold)

    codes = np.array(vec["inputs"]["codes"], dtype=np.int8)
    scales = np.array(vec["inputs"]["scales"], dtype=np.float32)
    x = np.array(vec["inputs"]["x"], dtype=np.float32)
    codes_bad = codes.copy()
    codes_bad[0, 0] = np.int8(codes_bad[0, 0] + 1)
    y_bad = fe.qgemv(codes_bad, scales, x, weight_bits=4, group_size=4)
    assert not np.array_equal(y, y_bad)

    # The bit-exact assertion must actually refuse. Watching it raise is the
    # whole point: comparing the engine to itself would stay green here.
    with pytest.raises(AssertionError):
        assert np.array_equal(y_gold, y_bad)

    with pytest.raises(AssertionError):
        assert fe.bit_exact_equal(y_gold, y_bad)


def test_wrong_reference_that_drops_the_scale_is_refused():
    vec = fe.vector_by_id("qgemv_hand")
    y = fe.run_vector(vec)
    codes = np.array(vec["inputs"]["codes"], dtype=np.float32)
    x = np.array(vec["inputs"]["x"], dtype=np.float32)
    # Deliberately wrong: integer codes dotted against x, scale forgotten.
    wrong = np.asarray(codes @ x, dtype=np.float32)
    assert not np.array_equal(y, wrong)
    with pytest.raises(AssertionError):
        assert np.array_equal(y, wrong)
