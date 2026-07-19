#!/usr/bin/env python3.12
"""Generation-M Fidelity D - short hybrid GPT-OSS end-to-end bridge (Part III sec 9).

Builds a SHORT hybrid GPT-OSS-120B path: a few REAL transformer layers over the verified tokenizer
with Harmony formatting and deterministic seeds. Attention / embedding / unembedding / norms / router
/ biases are SOURCE-NATIVE (dequantized straight from the sealed provenance shards, every byte
billed). The MoE expert projections in the short stack are packed with the proposed Gen-M mapping
(gravity_forge product-quant, ~0.5 base bpw) and are executed through a SWAPPABLE execution grammar:

  * grammar="genf"  -> B1 direct/compact: decode PQ codes into BOUNDED row tiles, tile matmul
                        (mech_measure.b1_reconstruct_matvec_np) - the Generation-F direct path.
  * grammar="genm"  -> M1 lookup-linear: build codeword tables once, accumulate by index gathers
                        (mech_measure.m1_lookup_linear_np) - the Generation-M execution grammar.
  * grammar="source"-> reference source-native fp32 experts (NOT packed) - the faithful path used
                        only to size the honest capability degradation of the sub-bit representation.

KEY closeout (execution-grammar parity): genf and genm execute the SAME packed artifact, so the END-
TO-END logits must be ~identical. We measure logit cosine, logit KL, top-k overlap and next-token
agreement. That parity is what promotes the Gen-M execution grammar (it is representation-neutral).

SEPARATELY (honest absolute capability): the sub-bit representation is Gravity-NEGATIVE. We record it
without rescue - NLL/perplexity slice (short-stack early-exit; relative, NOT full-depth), and the
source-native-vs-subbit agreement over a code/math/reasoning/instruction fixture. No tools, no
retrieval, no spec-decode. We claim NO capability. A catastrophic protected-domain regression blocks
REPRESENTATION promotion but NOT the execution-grammar promotion (which only needs parity).

Honesty / boundary:
  * Short stack = K real blocks then early-exit through the real final norm + unembedding. Absolute
    NLL/perplexity is a SHORT-STACK EARLY-EXIT signal (the model is not trained to decode at layer K),
    valid for RELATIVE comparison (source-native reference vs sub-bit) but NOT the model's true full-
    depth perplexity. Labelled as such; never dressed up as a capability pass.
  * No dense shadow: packed experts are executed via bounded tiles / codeword tables; the decoded
    tensor is never materialized whole. Packed-artifact .recon is dropped after packing.
  * from-config attention (RoPE/SwiGLU-clamp/eps not HF-parity-validated). Approximations are shared
    by every grammar and by the source-native reference, so they cancel in the relative signals.
  * Energy UNAVAILABLE (no sudo powermetrics). No invented estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gravity_forge as gf          # frozen: PQ pack (READ ONLY)
import gptoss_moe_runtime as rt     # frozen: provenance reader + swiglu (READ ONLY)
import mech_measure as mm           # frozen: b1 direct (genf) + m1 lookup-linear (genm) (READ ONLY)

SCHEMA = "hawking.generation_m.fidelity_d.v1"
MODEL_DIR = "models/gpt-oss-120b"
REPORT_DIR = "reports/mechanics_thermodynamics"

HIDDEN, N_Q, N_KV, HEAD_DIM = 2880, 64, 8, 64
ROPE_THETA, RMS_EPS = 150000.0, 1e-5
TOP_K = 4
VOCAB = 201088

# Gen-M base mapping: d8 PQ ~0.5 base bpw (S*log2(k)/D = 8*4/64), matching the sealed Gen-M artifact.
PQ_CFG = {"dim": 64, "subspaces": 8, "k": 16, "seed": 0}
TILE_ROWS = 512

# Harmony system preamble (deterministic, minimal).
HARMONY_SYSTEM = "You are a helpful assistant."

# Deterministic, domain-spread prompt set. role in {calibration, validation, holdout};
# domain in {general, code, math, reasoning, instruction}. Protected domains = code/math/
# reasoning/instruction (a catastrophic regression on any blocks REPRESENTATION promotion).
PROMPTS = [
    {"role": "calibration", "domain": "general",
     "user": "The capital of France is", "target": " Paris."},
    {"role": "validation", "domain": "code",
     "user": "Complete the Python function to add two numbers:\ndef add(a, b):\n    return",
     "target": " a + b"},
    {"role": "validation", "domain": "math",
     "user": "Solve for x: 3x + 7 = 22. The value of x is", "target": " 5."},
    {"role": "holdout", "domain": "reasoning",
     "user": "If all cats are mammals and all mammals are animals, then all cats are",
     "target": " animals."},
    {"role": "holdout", "domain": "instruction",
     "user": "Reply with exactly one word: what color is a clear daytime sky?",
     "target": " Blue"},
]


# --------------------------------------------------------------------------------------------
# Tokenizer + Harmony formatting (verified).
# --------------------------------------------------------------------------------------------
def load_tokenizer():
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
    return tk


def tokenizer_report(tk) -> dict[str, Any]:
    probe = "def add(a, b):\n    return a + b"
    ids = tk.encode(probe, add_special_tokens=False).ids
    roundtrip = probe.strip() in tk.decode(ids)
    specials = {}
    for s in ("<|start|>", "<|message|>", "<|end|>", "<|return|>", "<|channel|>",
              "<|startoftext|>", "<|endoftext|>"):
        specials[s] = tk.encode(s, add_special_tokens=False).ids
    return {"vocab_size": tk.get_vocab_size(), "roundtrip_ok": bool(roundtrip),
            "special_tokens": specials, "harmony_markers_single_token":
            all(len(v) == 1 for v in specials.values()),
            "chat_template_present": os.path.exists(f"{MODEL_DIR}/chat_template.jinja")}


def harmony_ids(tk, system: str, user: str, target: str) -> tuple[list[int], int]:
    """Build a Harmony-formatted token id sequence and return (ids, prompt_len) where prompt_len is
    the number of tokens BEFORE the target continuation (NLL is scored on the target tokens)."""
    def enc(s: str) -> list[int]:
        return [int(i) for i in tk.encode(s, add_special_tokens=False).ids]
    st, msg, end = enc("<|start|>"), enc("<|message|>"), enc("<|end|>")
    chan = enc("<|channel|>")
    prompt = (st + enc("system") + msg + enc(system) + end
              + st + enc("user") + msg + enc(user) + end
              + st + enc("assistant") + chan + enc("final") + msg)
    tgt = enc(target)
    return prompt + tgt, len(prompt)


# --------------------------------------------------------------------------------------------
# Short hybrid stack: source-native attention, Gen-M-packed MoE (swappable grammar).
# --------------------------------------------------------------------------------------------
def rmsnorm(x: np.ndarray, scale: np.ndarray, eps: float = RMS_EPS) -> np.ndarray:
    ms = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)) * scale


def _rope(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    half = HEAD_DIM // 2
    freqs = ROPE_THETA ** (-np.arange(half, dtype=np.float32) / half)
    ang = np.outer(pos.astype(np.float32), freqs)
    cos = np.cos(ang)[:, None, :]; sin = np.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def block_attention(reader: rt.ProvenanceReader, b: int, x: np.ndarray) -> np.ndarray:
    """Source-native block-b attention over x:[seq,HIDDEN]. GQA + RoPE + causal + per-head sinks.
    Same math as the frozen gptoss_block.block0_attention, generalized to any block index."""
    seq = x.shape[0]
    nrm = reader.bf16(f"block.{b}.attn.norm.scale")
    qkvw = reader.bf16(f"block.{b}.attn.qkv.weight")
    qkvb = reader.bf16(f"block.{b}.attn.qkv.bias")
    outw = reader.bf16(f"block.{b}.attn.out.weight")
    outb = reader.bf16(f"block.{b}.attn.out.bias")
    sinks = reader.bf16(f"block.{b}.attn.sinks")
    h = rmsnorm(x, nrm)
    qkv = h @ qkvw.T + qkvb
    q = qkv[:, :N_Q * HEAD_DIM].reshape(seq, N_Q, HEAD_DIM)
    k = qkv[:, N_Q * HEAD_DIM:(N_Q + N_KV) * HEAD_DIM].reshape(seq, N_KV, HEAD_DIM)
    v = qkv[:, (N_Q + N_KV) * HEAD_DIM:].reshape(seq, N_KV, HEAD_DIM)
    pos = np.arange(seq)
    q = _rope(q, pos); k = _rope(k, pos)
    grp = N_Q // N_KV
    scale = 1.0 / np.sqrt(HEAD_DIM)
    causal = np.triu(np.full((seq, seq), -1e30, dtype=np.float32), 1)
    out = np.zeros((seq, N_Q, HEAD_DIM), dtype=np.float32)
    for hh in range(N_Q):
        kv = hh // grp
        scores = (q[:, hh] @ k[:, kv].T) * scale + causal
        aug = np.concatenate([scores, np.full((seq, 1), sinks[hh], np.float32)], axis=1)
        aug -= aug.max(axis=1, keepdims=True)
        w = np.exp(aug); w /= w.sum(axis=1, keepdims=True)
        out[:, hh] = w[:, :seq] @ v[:, kv]
    return out.reshape(seq, N_Q * HEAD_DIM) @ outw.T + outb


class ExpertProvider:
    """Lazy per-(block,expert) cache. Source fp32 experts and Gen-M packed PQ artifacts (recon dropped).
    Records touched experts and billed bytes."""

    def __init__(self, reader: rt.ProvenanceReader):
        self.reader = reader
        self.src: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        self.art: dict[tuple[int, int], dict[str, Any]] = {}
        self.pack_seconds = 0.0
        self.n_packed = 0

    def source(self, b: int, e: int) -> dict[str, np.ndarray]:
        key = (b, e)
        if key not in self.src:
            self.src[key] = self.reader_load(b, e)
        return self.src[key]

    def reader_load(self, b: int, e: int) -> dict[str, np.ndarray]:
        ex = rt.load_expert(self.reader, b, e)
        return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in ex.items()}

    def packed(self, b: int, e: int) -> dict[str, Any]:
        key = (b, e)
        if key not in self.art:
            ex = self.source(b, e)
            t0 = time.time()
            a1 = gf.pack_product_quant(ex["mlp1"], **PQ_CFG)
            a2 = gf.pack_product_quant(ex["mlp2"], **PQ_CFG)
            self.pack_seconds += time.time() - t0
            self.n_packed += 1
            # drop the dense recon shadow; b1/m1 read only pq_codes
            a1.recon = np.empty((0, 0), dtype=np.float32)
            a2.recon = np.empty((0, 0), dtype=np.float32)
            self.art[key] = {"art1": a1, "art2": a2,
                             "mlp1_bias": ex["mlp1_bias"], "mlp2_bias": ex["mlp2_bias"],
                             "bytes": int(a1.physical_bytes + a2.physical_bytes),
                             "weights": int(ex["mlp1"].size + ex["mlp2"].size),
                             "base_bpw": float(a1.base_bpw)}
        return self.art[key]


def _mlp1_matvec(pk: dict[str, Any], x: np.ndarray, grammar: str) -> np.ndarray:
    if grammar == "genf":
        return mm.b1_reconstruct_matvec_np(pk["art1"], x, tile_rows=TILE_ROWS) + pk["mlp1_bias"]
    return mm.m1_lookup_linear_np(pk["art1"], x) + pk["mlp1_bias"]


def _mlp2_matvec(pk: dict[str, Any], a: np.ndarray, grammar: str) -> np.ndarray:
    if grammar == "genf":
        return mm.b1_reconstruct_matvec_np(pk["art2"], a, tile_rows=TILE_ROWS) + pk["mlp2_bias"]
    return mm.m1_lookup_linear_np(pk["art2"], a) + pk["mlp2_bias"]


def moe_block(reader, b, moe_in, provider: ExpertProvider, grammar: str, routed_log: set) -> np.ndarray:
    """MoE over a sequence moe_in:[seq,HIDDEN]. grammar in {source, genf, genm}. Router source-native."""
    router = reader_router(reader, b)
    seq = moe_in.shape[0]
    out = np.zeros((seq, HIDDEN), dtype=np.float32)
    for t in range(seq):
        x = moe_in[t]
        logits = router["weight"] @ x + router["bias"]
        idx = np.argsort(-logits)[:TOP_K]
        w = logits[idx]; w = np.exp(w - w.max()); w = w / w.sum()
        acc = np.zeros(HIDDEN, dtype=np.float32)
        for e, gw in zip(idx, w):
            e = int(e)
            routed_log.add((b, e))
            if grammar == "source":
                ex = provider.source(b, e)
                h = ex["mlp1"] @ x + ex["mlp1_bias"]
                a = rt._swiglu(h)
                y = ex["mlp2"] @ a + ex["mlp2_bias"]
            else:
                pk = provider.packed(b, e)
                h = _mlp1_matvec(pk, x, grammar)
                a = rt._swiglu(h)
                y = _mlp2_matvec(pk, a, grammar)
            acc += gw * y
        out[t] = acc
    return out


_ROUTER_CACHE: dict[tuple[int, int], dict[str, np.ndarray]] = {}


def reader_router(reader, b):
    key = (id(reader), b)
    if key not in _ROUTER_CACHE:
        r = rt.load_router(reader, b)
        _ROUTER_CACHE[key] = {"weight": np.ascontiguousarray(r["weight"], np.float32),
                              "bias": np.ascontiguousarray(r["bias"], np.float32)}
    return _ROUTER_CACHE[key]


def forward_logits(reader, emb, unemb, final_norm, token_ids: list[int], K: int,
                   provider: ExpertProvider, grammar: str) -> tuple[np.ndarray, set]:
    """Short hybrid forward: embed -> K blocks (source attn + swappable MoE) -> final norm ->
    unembedding. Returns logits[seq, VOCAB] and the routed (b,e) set."""
    x = np.ascontiguousarray(emb[token_ids], dtype=np.float32)   # [seq, HIDDEN]
    routed: set = set()
    for b in range(K):
        attn = block_attention(reader, b, x)
        x_mid = x + attn
        moe_in = rmsnorm(x_mid, reader.bf16(f"block.{b}.mlp.norm.scale"))
        moe_out = moe_block(reader, b, moe_in, provider, grammar, routed)
        x = x_mid + moe_out
    final = rmsnorm(x, final_norm)
    logits = final @ unemb.T                                     # [seq, VOCAB]
    return logits, routed


# --------------------------------------------------------------------------------------------
# Metrics.
# --------------------------------------------------------------------------------------------
def _logsoftmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _cos_rows(a: np.ndarray, b: np.ndarray) -> float:
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return float(np.mean(num / den))


def _kl_rows(logp: np.ndarray, logq: np.ndarray) -> float:
    p = np.exp(logp)
    return float(np.mean((p * (logp - logq)).sum(-1)))


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    ov = []
    for i in range(a.shape[0]):
        ta = set(np.argpartition(-a[i], k)[:k].tolist())
        tb = set(np.argpartition(-b[i], k)[:k].tolist())
        ov.append(len(ta & tb) / k)
    return float(np.mean(ov))


def logit_agreement(A: np.ndarray, B: np.ndarray, k: int = 10) -> dict[str, Any]:
    """Agreement between two logit sets over matched positions. A is treated as reference for KL."""
    lpA, lpB = _logsoftmax(A), _logsoftmax(B)
    argA, argB = A.argmax(-1), B.argmax(-1)
    return {
        "logit_cosine_mean": round(_cos_rows(A, B), 8),
        "kl_A_to_B_mean_nats": round(_kl_rows(lpA, lpB), 8),
        "kl_B_to_A_mean_nats": round(_kl_rows(lpB, lpA), 8),
        "topk_overlap_mean": round(_topk_overlap(A, B, k), 6),
        "next_token_agreement": round(float(np.mean(argA == argB)), 6),
        "positions": int(A.shape[0]),
    }


def nll_slice(logits: np.ndarray, token_ids: list[int], prompt_len: int) -> dict[str, Any]:
    """Teacher-forced NLL over the target continuation tokens (positions prompt_len-1 .. seq-2 predict
    tokens prompt_len .. seq-1). Short-stack early-exit signal; relative use only."""
    lp = _logsoftmax(logits)
    ids = np.asarray(token_ids)
    nlls = []
    for i in range(prompt_len - 1, len(ids) - 1):
        nlls.append(-float(lp[i, ids[i + 1]]))
    if not nlls:
        return {"n_target_tokens": 0, "nll_mean": None, "perplexity": None}
    m = float(np.mean(nlls))
    return {"n_target_tokens": len(nlls), "nll_mean": round(m, 5),
            "perplexity": round(float(np.exp(min(m, 50.0))), 3),
            "per_token_nll": [round(v, 4) for v in nlls]}


# --------------------------------------------------------------------------------------------
# Run.
# --------------------------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(*, K: int = 2, report_dir: str = REPORT_DIR) -> dict[str, Any]:
    reader = rt.ProvenanceReader()
    src_shard = reader.by_name["block.0.mlp.gate.weight"]["shard_path"]
    if not Path(src_shard).exists():
        return {"schema": SCHEMA, "green": False, "reason": "120B source shards absent"}

    tk = load_tokenizer()
    tkrep = tokenizer_report(tk)

    t_load = time.time()
    emb = reader.bf16("embedding.weight")          # [VOCAB, HIDDEN] fp32
    unemb = reader.bf16("unembedding.weight")       # [VOCAB, HIDDEN] fp32
    final_norm = reader.bf16("norm.scale")
    load_seconds = round(time.time() - t_load, 1)

    provider = ExpertProvider(reader)

    prompts_out = []
    parity_acc = {"logit_cosine": [], "kl_fb": [], "kl_bf": [], "topk": [], "nexttok": [], "max_abs": []}
    cap_acc = {"cos": [], "kl": [], "topk": [], "nexttok": [], "nll_ref": [], "nll_sub": []}
    domain_reg: dict[str, dict[str, Any]] = {}
    all_ids_len = []

    for spec in PROMPTS:
        ids, plen = harmony_ids(tk, HARMONY_SYSTEM, spec["user"], spec["target"])
        all_ids_len.append(len(ids))
        # three forwards
        log_src, routed_src = forward_logits(reader, emb, unemb, final_norm, ids, K, provider, "source")
        log_gf, routed_pk = forward_logits(reader, emb, unemb, final_norm, ids, K, provider, "genf")
        log_gm, _ = forward_logits(reader, emb, unemb, final_norm, ids, K, provider, "genm")

        # --- execution-grammar parity: genf vs genm (same packed artifact) ---
        parity = logit_agreement(log_gf, log_gm)
        parity["max_abs_logit_diff"] = round(float(np.max(np.abs(log_gf - log_gm))), 8)
        parity["mean_abs_logit_diff"] = round(float(np.mean(np.abs(log_gf - log_gm))), 8)
        parity["rel_frobenius"] = round(float(np.linalg.norm(log_gf - log_gm)
                                               / (np.linalg.norm(log_gm) + 1e-12)), 10)

        # --- absolute capability (honest): sub-bit (genm) vs source-native reference ---
        cap = logit_agreement(log_src, log_gm)      # source is reference for KL
        nll_ref = nll_slice(log_src, ids, plen)
        nll_sub = nll_slice(log_gm, ids, plen)

        parity_acc["logit_cosine"].append(parity["logit_cosine_mean"])
        parity_acc["kl_fb"].append(parity["kl_A_to_B_mean_nats"])
        parity_acc["kl_bf"].append(parity["kl_B_to_A_mean_nats"])
        parity_acc["topk"].append(parity["topk_overlap_mean"])
        parity_acc["nexttok"].append(parity["next_token_agreement"])
        parity_acc["max_abs"].append(parity["max_abs_logit_diff"])

        cap_acc["cos"].append(cap["logit_cosine_mean"])
        cap_acc["kl"].append(cap["kl_A_to_B_mean_nats"])
        cap_acc["topk"].append(cap["topk_overlap_mean"])
        cap_acc["nexttok"].append(cap["next_token_agreement"])
        if nll_ref["nll_mean"] is not None:
            cap_acc["nll_ref"].append(nll_ref["nll_mean"])
            cap_acc["nll_sub"].append(nll_sub["nll_mean"])

        # protected-domain regression (reference vs sub-bit)
        dom = spec["domain"]
        nll_delta = (None if nll_ref["nll_mean"] is None
                     else round(nll_sub["nll_mean"] - nll_ref["nll_mean"], 4))
        catastrophic = bool(cap["next_token_agreement"] < 0.5
                            or (nll_delta is not None and nll_delta > 2.0))
        domain_reg[dom] = {"role": spec["role"],
                           "next_token_agreement_ref_vs_subbit": cap["next_token_agreement"],
                           "topk_overlap_ref_vs_subbit": cap["topk_overlap_mean"],
                           "logit_cosine_ref_vs_subbit": cap["logit_cosine_mean"],
                           "nll_ref": nll_ref["nll_mean"], "nll_subbit": nll_sub["nll_mean"],
                           "nll_delta_subbit_minus_ref": nll_delta,
                           "catastrophic_regression": catastrophic}

        prompts_out.append({
            "role": spec["role"], "domain": spec["domain"],
            "user": spec["user"], "target": spec["target"],
            "n_tokens": len(ids), "prompt_len": plen, "n_target_tokens": nll_sub["n_target_tokens"],
            "routed_experts_source": len(routed_src), "routed_experts_packed": len(routed_pk),
            "routing_diverged_src_vs_packed": routed_src != routed_pk,
            "execution_parity_genf_vs_genm": parity,
            "capability_ref_vs_subbit": cap,
            "nll_reference": nll_ref, "nll_subbit": nll_sub,
        })

    def _agg(v):
        return {"mean": round(float(np.mean(v)), 8), "min": round(float(np.min(v)), 8),
                "max": round(float(np.max(v)), 8)} if v else None

    n_pk = len(provider.art)
    packed_bytes = sum(a["bytes"] for a in provider.art.values())
    packed_weights = sum(a["weights"] for a in provider.art.values())
    moe_base_bpw = (round(float(np.mean([a["base_bpw"] for a in provider.art.values()])), 5)
                    if provider.art else None)

    parity_ok = bool(np.min(parity_acc["logit_cosine"]) > 0.99999
                     and np.max(parity_acc["kl_fb"]) < 1e-4
                     and np.min(parity_acc["nexttok"]) >= 0.999)

    any_catastrophic = any(d["catastrophic_regression"] for d in domain_reg.values())

    doc = {
        "schema": SCHEMA, "generation": "M", "part": "III.9", "fidelity": "D",
        "generated_at": _now(),
        "green": bool(tkrep["roundtrip_ok"] and n_pk > 0 and len(prompts_out) == len(PROMPTS)),
        "hardware_profile": "Apple M3 Ultra 20P+8E 96GB MPS torch2.6 (CPU-authoritative grammars)",
        "device_note": "CPU numpy grammars are authoritative; energy UNAVAILABLE (no sudo powermetrics)",
        "stack": {
            "kind": "short_hybrid_early_exit",
            "K_real_blocks": K, "total_blocks_in_model": 36,
            "source_native_tensors": ["embedding", "attention(qkv/out/sinks/norm)",
                                      "mlp.norm", "router(gate)", "mlp1_bias", "mlp2_bias",
                                      "final_norm", "unembedding"],
            "gen_m_packed_tensors": ["mlp1_weight(up/gate)", "mlp2_weight(down)"],
            "pq_config": PQ_CFG, "moe_base_bpw_measured": moe_base_bpw,
            "early_exit_caveat": ("logits are produced by the real unembedding after K blocks; the model "
                                  "is NOT trained to decode at layer K, so absolute NLL/perplexity is a "
                                  "SHORT-STACK signal for RELATIVE comparison only, not full-depth perplexity"),
        },
        "tokenizer": tkrep,
        "harmony": {"system": HARMONY_SYSTEM,
                    "format": "<|start|>{role}<|message|>...<|end|>...<|start|>assistant<|channel|>final<|message|>{target}",
                    "deterministic_seeds": {"pq_seed": PQ_CFG["seed"], "numpy": "no sampling; greedy/teacher-forced"}},
        "execution_grammar_parity": {
            "headline": "Gen-M lookup-linear (M1/genm) vs Gen-F direct/compact (B1/genf) on END-TO-END logits",
            "same_packed_artifact": True,
            "logit_cosine": _agg(parity_acc["logit_cosine"]),
            "kl_genf_to_genm_nats": _agg(parity_acc["kl_fb"]),
            "kl_genm_to_genf_nats": _agg(parity_acc["kl_bf"]),
            "topk10_overlap": _agg(parity_acc["topk"]),
            "next_token_agreement": _agg(parity_acc["nexttok"]),
            "max_abs_logit_diff": _agg(parity_acc["max_abs"]),
            "PARITY_HOLDS": parity_ok,
            "verdict": ("execution-grammar parity CONFIRMED: Gen-M produces logits ~identical to Gen-F "
                        "(representation-neutral); Gen-M execution grammar is promotable"
                        if parity_ok else "PARITY FAILED - do NOT promote Gen-M execution grammar"),
        },
        "absolute_capability_honest": {
            "claim": "NONE. sub-bit representation is Gravity-NEGATIVE; recorded without rescue "
                     "(no tools/retrieval/spec-decode).",
            "reference": "source-native fp32 experts on the identical short stack (faithful path)",
            "subbit_vs_reference_logit_cosine": _agg(cap_acc["cos"]),
            "subbit_vs_reference_kl_nats": _agg(cap_acc["kl"]),
            "subbit_vs_reference_topk10_overlap": _agg(cap_acc["topk"]),
            "subbit_vs_reference_next_token_agreement": _agg(cap_acc["nexttok"]),
            "nll_reference_short_stack": _agg(cap_acc["nll_ref"]),
            "nll_subbit_short_stack": _agg(cap_acc["nll_sub"]),
            "sealed_per_tensor_quality_ref": "HAWKING_GENERATION_M.json: rel_err 0.65-0.88 mlp1 / 0.20-0.36 mlp2",
            "capability_pass": False, "event_horizon": False,
        },
        "protected_domain_regression": {
            "domains": domain_reg,
            "any_catastrophic": any_catastrophic,
            "blocks_representation_promotion": any_catastrophic,
            "blocks_execution_grammar_promotion": False,
            "note": ("a catastrophic protected-domain regression blocks REPRESENTATION promotion but NOT "
                     "the execution-grammar promotion, which only requires genf==genm parity"),
        },
        "byte_accounting": {
            "packed_experts": n_pk,
            "packed_moe_physical_bytes": int(packed_bytes),
            "packed_moe_weights": int(packed_weights),
            "packed_moe_whole_bpw": (round(packed_bytes * 8 / max(1, packed_weights), 5)
                                     if packed_weights else None),
            "moe_base_bpw": moe_base_bpw,
            "source_native_billed_at": "source dtype (bf16 attention/embed/unembed/norms/router/biases)",
            "no_dense_shadow": True, "recon_dropped_after_pack": True,
        },
        "cost": {"load_seconds": load_seconds, "pack_seconds": round(provider.pack_seconds, 1),
                 "experts_packed": provider.n_packed, "prompt_seq_lens": all_ids_len},
        "honesty": {
            "energy": "UNAVAILABLE (no sudo powermetrics; no invented estimates)",
            "attention": "from-config (RoPE/SwiGLU-clamp/eps not HF-parity-validated); shared by all "
                         "grammars and the reference so approximations cancel in relative signals",
            "timing": "not a timing claim; this is a fidelity/parity + capability bridge",
            "no_capability_claim": True,
        },
        "prompts": prompts_out,
    }
    doc["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in doc.items() if k != "sha256"}, sort_keys=True,
                   separators=(",", ":"), default=str).encode()).hexdigest()
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generation-M Fidelity D short hybrid end-to-end bridge.")
    ap.add_argument("--blocks", type=int, default=2, help="K real transformer layers in the short stack")
    ap.add_argument("--out-json", default=f"{REPORT_DIR}/GENERATION_M_FIDELITY_D.json")
    args = ap.parse_args(argv)
    t0 = time.time()
    doc = run(K=args.blocks)
    doc["cost"]["total_seconds"] = round(time.time() - t0, 1)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
    print(json.dumps({
        "green": doc.get("green"),
        "PARITY_HOLDS": doc.get("execution_grammar_parity", {}).get("PARITY_HOLDS"),
        "parity_logit_cosine": doc.get("execution_grammar_parity", {}).get("logit_cosine"),
        "parity_next_token": doc.get("execution_grammar_parity", {}).get("next_token_agreement"),
        "subbit_vs_ref_next_token": doc.get("absolute_capability_honest", {}).get(
            "subbit_vs_reference_next_token_agreement"),
        "any_catastrophic": doc.get("protected_domain_regression", {}).get("any_catastrophic"),
        "experts_packed": doc.get("cost", {}).get("experts_packed"),
        "total_seconds": doc["cost"].get("total_seconds"),
    }, indent=2, default=str))
    return 0 if doc.get("green") else 1


if __name__ == "__main__":
    raise SystemExit(main())
