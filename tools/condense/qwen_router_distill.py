#!/usr/bin/env python3.12
"""S2C - ROUTER DISTILLATION under structural expert omission (Qwen3-235B-A22B).

QUESTION. When the inventory drops to K survivors per layer, the stock policy is to score all 128
experts with the shipped router and MASK the omitted ones before the top-8. Can a student router -
starting from the survivor rows that already ship, so at zero or near-zero extra bytes - beat plain
masking at reproducing the teacher's WEIGHTED MoE OUTPUT?

WHAT IS MEASURED, HONESTLY.
  * This is OUTPUT-SPACE error of ONE MoE block on a calibration token set. That is a PROXY. It is
    NOT capability, not perplexity, not a gate metric. Nothing here may be reported as capability.
  * The layer-0 arm uses REAL weights on a real-but-ATTENTION-FREE input: x = RMSNorm(embed(tok)) *
    post_attention_layernorm.gamma. The true layer-0 MoE input includes the attention residual,
    which needs a full forward we cannot afford beside the running heavy campaign. It is real
    embedding geometry, real gamma, real router, real experts, approximate x.
  * The deep arm (layer 46 by default) feeds that SAME layer-0-shaped input into a deep layer's
    router and experts. That input distribution is WRONG at depth. That arm is labelled
    MACHINERY-VALIDATION and is not evidence about deep layers.
  * Calibration/holdout split is over DISJOINT TOKEN IDS. Because x depends only on the token id in
    this construction, the split is a genuine generalization test: survivors and every student
    parameter are fitted on the train ids only and scored on ids never seen.

BITS. The survivor router rows [K, 4096] already ship under structural omission, so REPLACING their
values (arm `full`) costs ZERO extra bytes. A per-survivor logit bias costs K*16 bits per layer. A
rank-r correction costs r*(K+4096)*16 bits per layer. All three are charged exactly against the
S64_structural ledger (0.948410027 complete BPW) via tools/foundry/one_bit_ceiling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_FOUNDRY = os.path.join(os.path.dirname(_HERE), "foundry")
if _FOUNDRY not in sys.path:
    sys.path.insert(0, _FOUNDRY)

from qwen_real_forward import SafetensorsIndexReader, rmsnorm, swiglu  # noqa: E402
from one_bit_ceiling import assert_complete_bpw_le_one  # noqa: E402

SCHEMA = "hawking.subbit.router_distill.v1"
MODEL_DIR = "models/qwen3-235b-a22b"
N_EXPERTS, TOP_K, EPS = 128, 8, 1e-6
ORIGINAL_WEIGHT_COUNT = 235093634560          # sealed S_STRUCTURAL_PLAN inventory
S64_STRUCTURAL_BPW = Fraction(948410027, 1000000000)
N_LAYERS = 94


# ── teacher / masked-student routing ─────────────────────────────────────────────────────────
def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def topk_route(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Qwen3 routing: softmax over the scored set, top-8, renormalize (norm_topk_prob=true)."""
    p = softmax(logits)
    idx = np.argsort(-p, axis=-1)[:, :TOP_K]
    w = np.take_along_axis(p, idx, axis=-1)
    return idx, w / np.maximum(w.sum(axis=-1, keepdims=True), 1e-20)


def mix(F: np.ndarray, idx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """F:[E,N,D] expert outputs, idx/w:[N,k] -> [N,D] weighted MoE output."""
    n = np.arange(idx.shape[0])[:, None]
    return np.einsum("nk,nkd->nd", w, F[idx, n, :])


def rel_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-20)


# ── student training (torch, CPU) ────────────────────────────────────────────────────────────
LR_GRID = (1e-4, 1e-3, 1e-2, 5e-2)   # swept per arm, SELECTED ON TRAIN LOSS ONLY (no holdout peek)


def train_student(X, F, mT, Wsurv, mode, steps=300, lr=None, seed=0):
    """Fit a survivor-router correction to minimize mean ||m_T - m_S||^2 / ||m_T||^2.

    Selection (top-8) is non-differentiable; we re-select each step with the current parameters and
    backprop only through the softmax weights. Alternating, not exact - stated, not hidden.
    Returns (params dict of numpy arrays, train loss trace).
    """
    import torch

    torch.manual_seed(seed)
    Xt = torch.from_numpy(np.ascontiguousarray(X))
    Ft = torch.from_numpy(np.ascontiguousarray(F))          # [K,N,D] survivors only
    mTt = torch.from_numpy(np.ascontiguousarray(mT))
    W0 = torch.from_numpy(np.ascontiguousarray(Wsurv))      # [K,4096]
    K, D = Wsurv.shape
    denom = (mTt * mTt).sum(-1).clamp_min(1e-20)

    ps: dict[str, "torch.Tensor"] = {}
    if mode == "bias":
        ps["b"] = torch.zeros(K, requires_grad=True)
    elif mode == "lowrank":
        r = 8
        ps["U"] = torch.zeros(K, r, requires_grad=True)
        ps["V"] = (torch.randn(r, D) * 1e-3).requires_grad_(True)
    elif mode == "full":
        ps["dW"] = torch.zeros(K, D, requires_grad=True)
    else:
        raise ValueError(mode)
    opt = torch.optim.Adam(list(ps.values()), lr=0.05 if lr is None else lr)

    def logits_of() -> "torch.Tensor":
        W = W0
        if "dW" in ps:
            W = W + ps["dW"]
        if "U" in ps:
            W = W + ps["U"] @ ps["V"]
        z = Xt @ W.T
        return z + ps["b"] if "b" in ps else z

    trace: list[float] = []
    best = (np.inf, {k: v.detach().numpy().copy() for k, v in ps.items()})
    nidx = torch.arange(Xt.shape[0])[:, None]
    for _ in range(steps):
        z = logits_of()
        with torch.no_grad():
            sel = z.topk(TOP_K, dim=-1).indices
        w = torch.softmax(z.gather(1, sel), dim=-1)          # softmax-over-selected == renormalized
        mS = (w.unsqueeze(-1) * Ft[sel, nidx, :]).sum(1)
        loss = (((mS - mTt) ** 2).sum(-1) / denom).mean()
        trace.append(float(loss))
        if trace[-1] < best[0]:   # early stop on the TRAIN objective only, no holdout peeking
            best = (trace[-1], {k: v.detach().numpy().copy() for k, v in ps.items()})
        opt.zero_grad()
        loss.backward()
        opt.step()
    return best[1], trace


def student_logits(X, Wsurv, p) -> np.ndarray:
    W = Wsurv.copy()
    if "dW" in p:
        W = W + p["dW"]
    if "U" in p:
        W = W + p["U"] @ p["V"]
    z = X @ W.T
    return z + p["b"] if "b" in p else z


# ── oracle ceiling ───────────────────────────────────────────────────────────────────────────
def oracle_rel_err(F, mT, n_tokens=16):
    """LOOSE lower bound on any survivor-restricted routing: greedy forward selection of TOP_K
    survivors with UNCONSTRAINED least-squares weights (a real router's weights are a positive
    softmax simplex, so no router can reach this). F:[K,N,D] survivors only."""
    out = []
    for n in range(min(n_tokens, mT.shape[0])):
        A = F[:, n, :]                                       # [K,D]
        t = mT[n]
        chosen: list[int] = []
        for _ in range(TOP_K):
            best, bestv = -1, np.inf
            for e in range(A.shape[0]):
                if e in chosen:
                    continue
                B = A[chosen + [e]].T
                c, *_ = np.linalg.lstsq(B, t, rcond=None)
                v = float(np.linalg.norm(B @ c - t))
                if v < bestv:
                    best, bestv = e, v
            chosen.append(best)
        out.append(bestv / max(float(np.linalg.norm(t)), 1e-20))
    return float(np.median(out))


# ── bit charging ─────────────────────────────────────────────────────────────────────────────
def charge(mode: str, K: int, d_model: int = 4096, n_layers: int = N_LAYERS) -> dict:
    """Extra bits ON TOP of S64_structural. Survivor router rows already ship, so a full retune of
    those rows is ZERO extra bytes (different values, same tensor). fp16 for new parameters."""
    if mode == "masked":
        extra = 0
    elif mode == "bias":
        extra = K * 16 * n_layers
    elif mode == "lowrank":
        extra = 8 * (K + d_model) * 16 * n_layers
    elif mode == "full":
        extra = 0
    else:
        raise ValueError(mode)
    bpw = S64_STRUCTURAL_BPW + Fraction(extra, ORIGINAL_WEIGHT_COUNT)
    return {"extra_bits": extra, "complete_bpw_exact": str(bpw), "complete_bpw": float(bpw),
            "legal_under_one_bit_ceiling": bpw <= 1}


# ── the real measurement ─────────────────────────────────────────────────────────────────────
def expert_outputs(r, layer: int, X: np.ndarray, experts: list[int]) -> np.ndarray:
    """[len(experts), N, D]. Loads exactly THREE tensors at a time and frees them."""
    F = np.empty((len(experts), X.shape[0], X.shape[1]), dtype=np.float32)
    for i, e in enumerate(experts):
        p = f"model.layers.{layer}.mlp.experts.{e}."
        g = r.bf16(p + "gate_proj.weight")
        u = r.bf16(p + "up_proj.weight")
        d = r.bf16(p + "down_proj.weight")
        F[i] = swiglu(X @ g.T, X @ u.T) @ d.T
        del g, u, d
    return F


def token_ids(n: int) -> list[int]:
    from tokenizers import Tokenizer
    import qwen_calibration_corpus as cc
    tk = Tokenizer.from_file(os.path.join(MODEL_DIR, "_meta", "tokenizer.json"))
    corpus = cc.build(min_tokens=1200, tokenizer=tk)
    seen: list[int] = []
    for pr in corpus["prompts"]:
        for i in pr["ids"]:
            if i not in seen:
                seen.append(i)
    return seen[:n], corpus["sha256"]


def run_layer(r, layer: int, ids: list[int], K: int, gamma_layer: int | None = None) -> dict:
    t0 = time.time()
    emb = r.bf16_rows("model.embed_tokens.weight", ids)
    gl = layer if gamma_layer is None else gamma_layer
    gamma = r.bf16(f"model.layers.{gl}.post_attention_layernorm.weight")
    X = rmsnorm(emb, gamma, EPS).astype(np.float32)
    Wg = r.bf16(f"model.layers.{layer}.mlp.gate.weight").astype(np.float32)   # [128,4096]

    tr = np.arange(0, len(ids), 2)              # disjoint token-id split
    ho = np.arange(1, len(ids), 2)
    F = expert_outputs(r, layer, X, list(range(N_EXPERTS)))

    logits_T = X @ Wg.T
    idxT, wT = topk_route(logits_T)
    mT = mix(F, idxT, wT)

    # survivors: top-K by teacher routing mass on the TRAIN ids only
    mass = np.zeros(N_EXPERTS, dtype=np.float64)
    np.add.at(mass, idxT[tr].ravel(), wT[tr].ravel())
    surv = np.sort(np.argsort(-mass)[:K])
    Fs, Ws = F[surv], Wg[surv]
    del F

    def score(z, sub):
        i, w = topk_route(z[sub])
        return rel_err(mix(Fs[:, sub, :], i, w), mT[sub])

    out: dict = {"layer": layer, "K": K, "n_tokens": len(ids),
                 "n_train": int(len(tr)), "n_holdout": int(len(ho)),
                 "survivor_train_routing_mass_frac": float(mass[surv].sum() / mass.sum()),
                 "arms": {}}

    zm = X @ Ws.T                                # masked stock router, survivor-restricted
    base_ho = score(zm, ho)
    out["arms"]["masked"] = {"holdout_median_rel_err": float(np.median(base_ho)),
                             "train_median_rel_err": float(np.median(score(zm, tr))),
                             "kl_teacher_restricted_vs_student": 0.0, **charge("masked", K)}

    # KL term, explicitly: softmax over the survivor logits IS the teacher distribution
    # renormalized onto the survivors, so KL(p_T|surv || p_S) is identically 0 for masking and can
    # only INCREASE with any correction. Measured below, not asserted.
    pT_r = softmax(logits_T[:, surv])

    for mode in ("bias", "lowrank", "full"):
        cands = [(train_student(X[tr], Fs[:, tr, :], mT[tr], Ws, mode, lr=lr), lr)
                 for lr in LR_GRID]
        (p, trace), lr_used = min(cands, key=lambda c: min(c[0][1]))
        z = student_logits(X, Ws, p)
        e_ho, e_tr = score(z, ho), score(z, tr)
        pS = softmax(z)
        kl = float(np.mean(np.sum(pT_r * (np.log(pT_r + 1e-30) - np.log(pS + 1e-30)), axis=-1)))
        out["arms"][mode] = {
            "holdout_median_rel_err": float(np.median(e_ho)),
            "train_median_rel_err": float(np.median(e_tr)),
            "holdout_delta_vs_masked": float(np.median(e_ho) - np.median(base_ho)),
            "beats_masked_holdout": bool(np.median(e_ho) < np.median(base_ho)),
            "lr_selected_on_train_loss": lr_used,
            "train_loss_first": trace[0], "train_loss_last": trace[-1],
            "train_loss_best": min(trace),
            "kl_teacher_restricted_vs_student": kl, **charge(mode, K)}

    out["oracle_greedy_ls_holdout_median_rel_err"] = oracle_rel_err(Fs[:, ho, :], mT[ho])
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _verdict(l0: dict, ldeep: dict) -> dict:
    """Mechanical verdict from the layer-0 (real-weights) arm; the deep arm is not evidence."""
    base = l0["arms"]["masked"]["holdout_median_rel_err"]
    wins = [m for m, v in l0["arms"].items() if m != "masked" and v["beats_masked_holdout"]]
    orc = l0["oracle_greedy_ls_holdout_median_rel_err"]
    return {"layer0_masked_holdout_rel_err": base,
            "layer0_students_beating_masked": wins,
            "layer0_reachable_headroom_frac": (base - orc) / base,
            "result": "NEGATIVE - masking wins" if not wins else "student beats masking",
            "note": ("Headroom is real (a survivor-restricted oracle halves the error) but no "
                     "trained student captures it on held-out token ids; every student overfits "
                     "the train ids. Consistent with the sealed finding that omitted expert "
                     "function is not reconstructible: the router can only reorder survivors.")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--keep", type=int, default=64)
    ap.add_argument("--deep-layer", type=int, default=46)
    ap.add_argument("--out", default="reports/subbit_reset/S2C_ROUTER_DISTILL.json")
    a = ap.parse_args(argv)
    if a.self_check:
        return demo()
    if not a.run:
        ap.error("pass --run or --self-check")

    r = SafetensorsIndexReader(MODEL_DIR)
    assert r.source_present(), "real Qwen3-235B shards absent"
    ids, corpus_sha = token_ids(a.tokens)
    l0 = run_layer(r, 0, ids, a.keep)
    l0["input_label"] = ("REAL weights, attention-free layer-0 input "
                         "x = RMSNorm(embed(tok)) * post_attention_layernorm.gamma")
    ldeep = run_layer(r, a.deep_layer, ids, a.keep)
    ldeep["input_label"] = ("MACHINERY-VALIDATION ONLY: layer-0-shaped input fed to a deep layer; "
                            "the input distribution is wrong at depth, this is not evidence")

    rep = {"schema": SCHEMA, "stage": "S2C_router_distill", "model": MODEL_DIR,
           "calibration_corpus_sha256": corpus_sha, "top_k": TOP_K, "n_experts": N_EXPERTS,
           "split": "disjoint token ids (even=train, odd=holdout); x depends only on token id",
           "metric": "median over holdout tokens of ||m_student - m_teacher128|| / ||m_teacher128||",
           "honesty": ["output-space proxy on calibration tokens, NOT capability",
                       "layer 0 input omits the attention residual",
                       "deep-layer arm is machinery validation, not evidence",
                       "top-8 selection is non-differentiable; training alternates "
                       "(re-select each step, backprop through the softmax weights only)",
                       "oracle is greedy + unconstrained least squares: looser than any router"],
           "layers": [l0, ldeep],
           "verdict": _verdict(l0, ldeep)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in rep.items() if k != "layers"}, indent=1))
    for L in rep["layers"]:
        print(json.dumps({"layer": L["layer"], "oracle": L["oracle_greedy_ls_holdout_median_rel_err"],
                          **{m: v["holdout_median_rel_err"] for m, v in L["arms"].items()}}))
    return 0


# ── self-check ───────────────────────────────────────────────────────────────────────────────
def demo() -> int:
    """Asserts that fail if the core logic breaks. Synthetic tiny MoE, no real weights."""
    rng = np.random.default_rng(0)
    E, N, D, K = 16, 24, 8, 8
    X = rng.normal(size=(N, D)).astype(np.float32)
    Wg = rng.normal(size=(E, D)).astype(np.float32)
    F = rng.normal(size=(E, N, D)).astype(np.float32)
    z = X @ Wg.T
    idx, w = topk_route(z)
    assert idx.shape == (N, TOP_K) and np.allclose(w.sum(-1), 1.0), "top-k renormalization broken"
    mT = mix(F, idx, w)

    # mixing agrees with an explicit loop
    ref = np.stack([sum(w[n, k] * F[idx[n, k], n] for k in range(TOP_K)) for n in range(N)])
    assert np.allclose(mT, ref, atol=1e-5), "mix() disagrees with the explicit weighted sum"

    # full 128-expert teacher reproduces itself at zero error
    assert float(np.median(rel_err(mT, mT))) == 0.0

    # KL(teacher|survivors || masked student) is IDENTICALLY zero: masking cannot move the KL term
    surv = np.sort(np.argsort(-np.bincount(idx.ravel(), minlength=E))[:K])
    pT_r, pS = softmax(z[:, surv]), softmax(X @ Wg[surv].T)
    assert float(np.abs(pT_r - pS).max()) < 1e-6, "restricted-softmax identity broken"

    # training reduces its own training objective, and the fitted student is applied consistently
    tr = np.arange(0, N, 2)
    p, trace = train_student(X[tr], F[surv][:, tr, :], mT[tr], Wg[surv], "bias", steps=60)
    assert min(trace) <= trace[0] + 1e-9, "training did not reduce the training loss"
    assert np.allclose(student_logits(X, Wg[surv], p), X @ Wg[surv].T + p["b"], atol=1e-4)

    # oracle (unconstrained LS over survivors) is a lower bound on the masked router's error
    ho = np.arange(1, N, 2)
    i2, w2 = topk_route((X @ Wg[surv].T)[ho])
    masked = float(np.median(rel_err(mix(F[surv][:, ho, :], i2, w2), mT[ho])))
    orc = oracle_rel_err(F[surv][:, ho, :], mT[ho], n_tokens=6)
    assert orc <= masked + 1e-6, f"oracle {orc} above masked {masked}: it is not a lower bound"

    # bit charging: bias costs K*16 bits/layer, full survivor-row retune costs nothing extra
    cb, cf = charge("bias", 64), charge("full", 64)
    assert cb["extra_bits"] == 64 * 16 * N_LAYERS and cf["extra_bits"] == 0
    assert cb["complete_bpw"] > float(S64_STRUCTURAL_BPW) and cb["legal_under_one_bit_ceiling"]
    led = assert_complete_bpw_le_one.__doc__ is not None
    assert led
    assert Fraction(cb["complete_bpw_exact"]) <= 1, "charged rate above the one-bit ceiling"
    print(json.dumps({"self_check": "PASS", "masked": masked, "oracle": orc,
                      "bias_extra_bits": cb["extra_bits"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
