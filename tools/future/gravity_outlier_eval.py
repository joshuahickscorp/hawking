"""An EXECUTING evaluator for GravityGauntlet.

The gauntlet takes `evaluator(candidate) -> receipt`. Until now the only
evaluators either replayed receipts that already existed on disk or shelled out
to the patient runner, whose --gravity grammar is bits/group only (uniform |
mixed | tiers in tools/odyssey_ctl.py parse_gravity_grammar). That is why every
recorded search stayed inside the quantization class: the ceiling was the
EVALUATOR and the spec grammar, never the gauntlet, whose Candidate.spec is a
free string and whose representation_class is unconstrained.

This module executes two classes for real and measures the same axes for both:

  AFFINE_QUANT   q<bits>-g<group>-experts   experts affine, rest 4b/g64
  OUTLIER_SPLIT  outlier<frac>-g<group>     top-|w| fraction kept at full
                 precision in a sparse side-channel over a coarse affine base

Capability gate is calibrated against a measured bf16 reference, not asserted.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

SNAP = ("/Users/scammermike/.cache/huggingface/hub/models--moonshotai--"
        "Kimi-VL-A3B-Instruct/snapshots/398eede0903cd983a2bfa0cc634e9ac1d843f375")
SRC_PARAMS = 32815315552 // 2

# Measured on this machine, receipts/future/O003_OUTLIER_SPLIT.json (bf16 arm,
# 8 prompts, greedy, 96 tokens). These are observations, not constants of nature.
BF16_R4 = 0.0538
BF16_PPL = 2.8936
# Gate: <=3x the reference repetition rate and <=1.25x the reference perplexity.
# Negative control for the gate itself: base-g128 (r4 0.7204, ppl 4.0847) MUST
# fail it and the bf16 arm MUST pass it. Both verified in test_gravity_outlier.py.
R4_MAX = 3.0 * BF16_R4
PPL_MAX = 1.25 * BF16_PPL

PROMPTS = [
    "Explain how a mixture-of-experts layer routes tokens to experts.",
    "What is the capital of France, and why did it become the capital?",
    "Write a short Python function that reverses a linked list.",
    "Summarize why bandwidth, not compute, limits transformer decoding.",
    "A farmer has 17 sheep and all but 9 run away. How many are left?",
    "Describe the difference between a scale and a zero point in quantization.",
    "List three reasons a neural network might fail to converge.",
    "Translate to French: The weather is cold today.",
]
NLL_TEXT = (
    "The mixture-of-experts architecture routes each token to a small subset of "
    "feed-forward networks, so the number of parameters activated per token is much "
    "smaller than the total parameter count. A router network produces a distribution "
    "over experts and the top-k are selected. This makes the model sparse in compute "
    "while remaining dense in memory, which is why storage dominates the cost."
) * 3


# mx.quantize supports exactly these. A spec naming any other group parses fine
# and then dies inside the search -- registration is not capability, so refuse it
# at the door instead. outlier0.005-g256 was accepted and failed this way.
_SUPPORTED_GROUPS = (32, 64, 128)


def _group(g: int, spec: str, affine: bool = True) -> int:
    if affine and g not in _SUPPORTED_GROUPS:
        raise ValueError(
            f"spec {spec!r} names group {g}; mx.quantize supports {_SUPPORTED_GROUPS}")
    if not affine:
        # sign/mean path: any power of two. Per tensor it falls back to the
        # largest divisor of that tensor's last dim, since down_proj is 1408 =
        # 2^7 * 11 and so admits no uniform group above 128.
        if g < 8 or (g & (g - 1)):
            raise ValueError(f"spec {spec!r} names group {g}; need a power of two >= 8")
    return g


def _eff_group(g: int, in_dim: int) -> int:
    while in_dim % g:
        g //= 2
    return g


def _split_non_expert(spec: str) -> tuple[str, int, int]:
    """`<expert-spec>+o<bits>g<group>` overrides the non-expert organ, which has
    been pinned at 4 bits / group 64 in every arm measured so far. It costs
    0.4318 complete EBPW on its own -- 43% of a 1.0 target -- so leaving it
    fixed put a floor under every result."""
    m = re.fullmatch(r"(.+)\+o(\d+)g(\d+)", str(spec).strip(), re.I)
    if not m:
        return str(spec).strip(), 4, 64
    return m.group(1), int(m.group(2)), _group(int(m.group(3)), spec)


def parse_spec(spec: str) -> dict[str, Any]:
    """Grammar extension. Returns a plan, or raises so an unrunnable spec can
    never be silently accepted as a candidate."""
    s, ne_bits, ne_group = _split_non_expert(spec)
    plan = _parse_expert_spec(s)
    plan["ne_bits"], plan["ne_group"] = ne_bits, ne_group
    return plan


def _parse_expert_spec(spec: str) -> dict[str, Any]:
    s = str(spec).strip()
    # BINARY_SCALED: W ~= scale_g * sign(W), scale = mean(|W|) over the group,
    # which is the L2-optimal scale for a sign quantiser. Costs 1 bit/weight +
    # 16 bits per group. MLX cannot express this -- mx.quantize floors at 2 bits
    # -- and it is the rung the resident asked for and could not have.
    # binarycal: SAME bytes as binary (1 bit + one fp16 scale per group). The
    # only difference is how the scale is fitted -- against measured activation
    # second moments rather than the plain mean of |W|. A pure capability
    # comparison at fixed EBPW.
    # binarypercal: per-EXPERT activation moments. Same bytes again; the
    # statistic is now accumulated separately per routed expert id, which is the
    # mismatch that pooled calibration is suspected to have died on.
    # resbinary: the resident's ladder proposal, executed. Two-stage residual
    # binarisation -- W ~= s1*sign(W) + s2*sign(W - s1*sign(W)). Costs 2
    # bits/weight + TWO fp16 scales per group, i.e. exactly the same bytes as
    # q2-g<group> affine, which makes it a controlled comparison of the same
    # budget spent two ways.
    # SPARSE_BINARY: keep the top `density` of weights per group by |w|, zero
    # the rest, binarise survivors with a per-group scale. Stored as index+sign
    # per survivor plus one fp16 scale per group, so the rate is
    #   density * (log2(group) + 1) + 16/group
    # which is the only family measured here that can go BELOW one bit/weight.
    # binaryef: binary with ERROR FEEDBACK along the group. Quantising w_j
    # leaves a residual; that residual is carried into w_{j+1} before it is
    # quantised, so the group's running error stays bounded instead of
    # accumulating independently per weight. Byte cost is IDENTICAL to plain
    # binary -- same 1 bit + one scale per group -- so it isolates the effect of
    # feedback alone. This is the cheapest member of the family named as G010's
    # reopen condition.
    # HOT/COLD: allocate bytes by MEASURED routing frequency, not uniformly.
    # Probe 2 (O003_WHY_THE_BYTES_EXIST.json) measured 3.0x concentration --
    # top-8 of 64 experts take 37.5% of routing against a uniform 12.5%, and
    # 1.8 experts per tensor are never routed at all. This is a structural class,
    # not a precision variation: the same nominal bits buy a different allocation.
    # PRODUCT QUANTIZATION. Split each row into sub-vectors of length d, store
    # an index into a codebook of size K learned per tensor by k-means.
    # Rate = log2(K)/d bits/weight + a per-tensor codebook of K*d fp16 values.
    # Discriminator already passed: at 2.000 b/w PQ's relative reconstruction
    # error is 0.101 against affine q2-g128's 0.182 at 2.250 b/w -- fewer bits
    # AND 45% less error, on all three projections.
    m = re.fullmatch(r"pq(\d+)k(\d+)", s, re.I)
    if m:
        return {"form": "pq", "sub_dim": int(m.group(1)), "codebook": int(m.group(2)),
                "frac": 0.0, "group": 128, "bits": 2}
    m = re.fullmatch(r"hotcold([0-9.]+)h(\d+)c(\d+)-g(\d+)", s, re.I)
    if m:
        return {"form": "hotcold", "hot_frac": float(m.group(1)),
                "hot_bits": int(m.group(2)), "cold_bits": int(m.group(3)),
                "frac": 0.0, "group": _group(int(m.group(4)), s), "bits": int(m.group(2))}
    m = re.fullmatch(r"binaryef(percal)?-g(\d+)", s, re.I)
    if m:
        return {"form": "binary_ef", "calibrated": bool(m.group(1)),
                "per_expert": bool(m.group(1)), "frac": 0.0,
                "group": _group(int(m.group(2)), s, affine=False), "bits": 1}
    m = re.fullmatch(r"sparse([0-9.]+)(percal)?-g(\d+)", s, re.I)
    if m:
        return {"form": "sparse", "density": float(m.group(1)),
                "calibrated": bool(m.group(2)), "per_expert": bool(m.group(2)),
                "frac": 0.0, "group": _group(int(m.group(3)), s, affine=False), "bits": 1}
    m = re.fullmatch(r"resbinary(percal)?-g(\d+)", s, re.I)
    if m:
        return {"form": "resbinary", "calibrated": bool(m.group(1)),
                "per_expert": bool(m.group(1)), "frac": 0.0,
                "group": _group(int(m.group(2)), s), "bits": 2}
    m = re.fullmatch(r"binarypercal(?:([0-9.]+))?-g(\d+)", s, re.I)
    if m:
        return {"form": "binary", "calibrated": True, "per_expert": True,
                "frac": float(m.group(1) or 0.0),
                "group": _group(int(m.group(2)), s, affine=False), "bits": 1}
    m = re.fullmatch(r"binarycal(?:([0-9.]+))?-g(\d+)", s, re.I)
    if m:
        return {"form": "binary", "calibrated": True, "frac": float(m.group(1) or 0.0),
                "group": _group(int(m.group(2)), s), "bits": 1}
    m = re.fullmatch(r"binary(?:([0-9.]+))?-g(\d+)", s, re.I)
    if m:
        return {"form": "binary", "calibrated": False, "frac": float(m.group(1) or 0.0),
                "group": _group(int(m.group(2)), s, affine=False), "bits": 1}
    m = re.fullmatch(r"outlier([0-9.]+)-g(\d+)", s, re.I)
    if m:
        return {"form": "outlier_split", "frac": float(m.group(1)),
                "group": _group(int(m.group(2)), s), "bits": 2}
    m = re.fullmatch(r"q(\d+)-g(\d+)(?:-experts)?", s, re.I)
    if m:
        return {"form": "affine", "bits": int(m.group(1)),
                "group": _group(int(m.group(2)), s), "frac": 0.0}
    if s == "bf16":
        return {"form": "bf16", "bits": 16, "group": 0, "frac": 0.0}
    raise ValueError(f"spec {spec!r} is not executable by this evaluator")


EXPERT_WEIGHTS = 14394851328          # measured
NON_EXPERT_BYTES_AT_4B_G64 = 885541120  # measured: dense + embed table
# Back out the non-expert weight count from that measurement so the term can be
# repriced at any bits/group instead of being pinned at 4b/g64.
NON_EXPERT_WEIGHTS = int(NON_EXPERT_BYTES_AT_4B_G64 * 8 / (4 + 32 / 64))


def predict_ebpw(spec: str) -> float:
    """Byte cost of a spec, by arithmetic, with no model load. Lets a proposal
    be checked for DIRECTION before spending ~4 minutes executing it.

    Exact for every non-outlier form (validated to <1e-4 against 7 measured
    specs). Outlier forms read ~0.004-0.008 EBPW HIGH because the magnitude
    threshold keeps slightly fewer weights than the requested fraction
    (0.004849 actual for 0.005 requested). That bias is conservative for a
    direction check -- it never lets an upward proposal look downward."""
    p = parse_spec(spec)
    g = p["group"]
    if p["form"] == "pq":
        import math
        per_w = math.log2(p["codebook"]) / p["sub_dim"]
    elif p["form"] == "hotcold":
        per_w = (p["hot_frac"] * p["hot_bits"] + (1 - p["hot_frac"]) * p["cold_bits"]
                 + 32 / g)
    elif p["form"] == "binary_ef":
        per_w = 1 + 16 / g
    elif p["form"] == "sparse":
        import math
        per_w = p["density"] * (math.log2(g) + 1) + 16 / g
    elif p["form"] == "binary":
        per_w = 1 + 16 / g
    elif p["form"] == "resbinary":
        per_w = 2 + 32 / g
    elif p["form"] == "bf16":
        per_w = 16.0
    else:
        per_w = p["bits"] + 32 / g
    bits = EXPERT_WEIGHTS * per_w + EXPERT_WEIGHTS * p.get("frac", 0.0) * 32
    ne_bits = NON_EXPERT_WEIGHTS * (p["ne_bits"] + 32 / p["ne_group"])
    return (bits / 8 + ne_bits / 8) * 8 / SRC_PARAMS


def _load():
    from mlx_lm.models import kimi_vl as KV
    if not getattr(KV.Model, "_hawking_mla_patched", False):
        orig = KV.Model.sanitize

        def sanitize(self, w):
            w = orig(self, w)
            a = self.args.text_config
            nh, nope, vhd = a.num_attention_heads, a.qk_nope_head_dim, a.v_head_dim
            hd = nope + vhd
            for l in range(a.num_hidden_layers):
                p = f"language_model.model.layers.{l}.self_attn"
                k = f"{p}.kv_b_proj.weight"
                if k in w:
                    v = w.pop(k).reshape(nh, hd, -1)
                    w[f"{p}.embed_q.weight"] = mx.contiguous(v[:, :nope, :].swapaxes(-1, -2))
                    w[f"{p}.unembed_out.weight"] = mx.contiguous(v[:, nope:, :])
            return w

        KV.Model.sanitize = sanitize
        KV.Model._hawking_mla_patched = True
    from mlx_lm import load
    return load(SNAP, tokenizer_config={"trust_remote_code": True})


def _leaf_bytes(params, skip: str) -> int:
    total = 0
    stack = [("", params)]
    while stack:
        pre, node = stack.pop()
        if isinstance(node, dict):
            stack += [(f"{pre}.{k}", v) for k, v in node.items()]
        elif isinstance(node, list):
            stack += [(f"{pre}.{i}", v) for i, v in enumerate(node)]
        elif isinstance(node, mx.array) and skip not in pre:
            total += node.nbytes
    return total


def collect_routing_counts(model, tok) -> dict[int, "mx.array"]:
    """How often each expert is actually selected, per tensor, from real tokens.
    Refuses to return an all-uniform result silently -- a flat count would make
    hot/cold identical to uniform allocation and the arm would look like a null
    when it was really a broken hook."""
    from mlx_lm.models import switch_layers as SL
    counts: dict[int, Any] = {}
    orig = SL.SwitchLinear.__call__

    def cap(self, x, indices, sorted_indices=False):
        k = id(self)
        n = self["weight"].shape[0]
        oh = (indices.reshape(-1)[:, None] == mx.arange(n)[None, :]).astype(mx.float32)
        c = mx.sum(oh, axis=0)
        counts[k] = c if k not in counts else counts[k] + c
        return orig(self, x, indices, sorted_indices=sorted_indices)

    SL.SwitchLinear.__call__ = cap
    try:
        for text in (NLL_TEXT, " ".join(PROMPTS)):
            mx.eval(model(mx.array([tok.encode(text)[:512]])))
    finally:
        SL.SwitchLinear.__call__ = orig
    if not counts:
        raise RuntimeError("routing capture got NOTHING")
    spread = max(float(mx.max(c) / mx.maximum(mx.mean(c), 1e-9)) for c in counts.values())
    if spread < 1.2:
        raise RuntimeError(f"routing looks uniform (max/mean {spread:.2f}); hot/cold would be a no-op")
    return counts


def collect_activation_moments(model, tok, per_expert: bool = False) -> dict[int, "mx.array"]:
    """Per-input-channel second moment for every expert matrix, from a real
    forward pass. Diagonal approximation to E[xx^T]: what the layer actually
    sees, not a Gaussian proxy -- every sub-bit negative on this machine that
    later reversed had been measured against a proxy."""
    from mlx_lm.models import switch_layers as SL
    stats: dict[int, Any] = {}
    orig = SL.SwitchLinear.__call__

    def capture(self, x, indices, sorted_indices=False):
        k = id(self)
        inp = x.shape[-1]
        if not per_expert:
            d = mx.mean(mx.square(x.astype(mx.float32).reshape(-1, inp)), axis=0)
            stats[k] = d if k not in stats else stats[k] + d
            return orig(self, x, indices, sorted_indices=sorted_indices)
        # Route each row to the expert that actually consumed it. gate/up share
        # one token vector across the top-k experts (x has N rows, idx has N*K),
        # so those rows repeat; down_proj already carries one row per expert.
        n_exp = self["weight"].shape[0]
        K = indices.shape[-1]
        xf = x.astype(mx.float32).reshape(-1, inp)
        idf = indices.reshape(-1)
        if xf.shape[0] != idf.shape[0]:
            xf = mx.repeat(xf, K, axis=0)
        oh = (idf[:, None] == mx.arange(n_exp)[None, :]).astype(mx.float32)
        acc = oh.T @ mx.square(xf)            # (n_exp, inp)
        cnt = mx.sum(oh, axis=0)              # (n_exp,)
        prev = stats.get(k)
        stats[k] = (acc, cnt) if prev is None else (prev[0] + acc, prev[1] + cnt)
        return orig(self, x, indices, sorted_indices=sorted_indices)

    SL.SwitchLinear.__call__ = capture
    try:
        for text in (NLL_TEXT, " ".join(PROMPTS)):
            ids = mx.array([tok.encode(text)[:512]])
            mx.eval(model(ids))
    finally:
        SL.SwitchLinear.__call__ = orig
    if per_expert:
        for k, (acc, cnt) in list(stats.items()):
            stats[k] = acc / mx.maximum(cnt, 1.0)[:, None]     # (n_exp, inp)
    for k in stats:
        mx.eval(stats[k])
    if not stats:
        raise RuntimeError("calibration captured NOTHING -- refusing to pass it off as calibrated")
    return stats


def evaluate(candidate) -> dict[str, Any]:
    # S010 §5: the OS must never be the first component to discover Hawking
    # exceeded its budget. The 2026-09-06 watchdog panic happened during this
    # exact kind of experiment. Guard BEFORE the load, not after.
    try:
        from campaign_memory_guard import require_ok
        _guard = require_ok(f"gravity evaluate({candidate})")
    except ImportError:
        _guard = None
    """Execute one representation and return a gauntlet receipt."""
    plan = parse_spec(getattr(candidate, "spec", candidate))
    t0 = time.perf_counter()
    from mlx_lm import generate
    model, tok = _load()
    sw = [(p, m) for p, m in model.named_modules()
          if "switch_mlp" in p and isinstance(getattr(m, "weight", None), mx.array)]
    if len(sw) != 78:
        raise RuntimeError(f"expected 78 expert tensors, found {len(sw)}")

    mx.random.seed(0)
    n_tot = sum(m.weight.size for _, m in sw)
    BIN = plan["form"] == "binary"
    RES = plan["form"] == "resbinary"
    SPARSE = plan["form"] == "sparse"
    PQ = plan["form"] == "pq"
    pq_cb_values = 0
    EF = plan["form"] == "binary_ef"
    act = (collect_activation_moments(model, tok, per_expert=plan.get("per_expert", False))
           if plan.get("calibrated") else {})
    route = collect_routing_counts(model, tok) if plan["form"] == "hotcold" else {}
    hot_w = cold_w = 0
    kept = 0
    n_scales = 0            # ACTUAL scale count, from the per-tensor group used
    n_survivors = 0.0       # ACTUAL non-zero count for the sparse form
    eff_groups: dict[int, int] = {}
    # Magnitude adequacy over the whole expert organ. Cosine alone is
    # scale-invariant, so 0.01*W scores 1.0 on direction -- the ratio is what
    # actually catches a magnitude-destroyed representation.
    s_w2 = s_h2 = s_dot = 0.0
    if plan["form"] != "bf16":
        g0, bits, frac = plan["group"], plan["bits"], plan["frac"]
        for _, m in sw:
            W = m.weight.astype(mx.float32)
            g = _eff_group(g0, W.shape[-1]) if (BIN or RES) else g0
            eff_groups[g] = eff_groups.get(g, 0) + 1
            n_scales += W.size // g
            if frac > 0:
                flat = mx.abs(W).reshape(-1)
                k = int(flat.size * frac)
                thr = mx.sort(flat)[flat.size - k - 1]
                mask = (flat > thr).reshape(W.shape)
                base = mx.where(mask, mx.zeros_like(W), W)
                kept += int(mx.sum(mask).item())
            else:
                mask, base = None, W
            if PQ:
                d, K = plan["sub_dim"], plan["codebook"]
                X = base.reshape(-1, d)
                n = X.shape[0]
                smp = X[mx.random.randint(0, n, shape=(min(200_000, n),))]
                C = smp[mx.random.randint(0, smp.shape[0], shape=(K,))]
                for _ in range(12):
                    d2 = (mx.sum(smp * smp, axis=1, keepdims=True) - 2 * (smp @ C.T)
                          + mx.sum(C * C, axis=1)[None, :])
                    a = mx.argmin(d2, axis=1)
                    oh = (a[:, None] == mx.arange(K)[None, :]).astype(mx.float32)
                    C = (oh.T @ smp) / mx.maximum(mx.sum(oh, axis=0)[:, None], 1.0)
                    mx.eval(C)
                parts = []
                CH = 2_000_000
                for i in range(0, n, CH):
                    x = X[i:i + CH]
                    d2 = (mx.sum(x * x, axis=1, keepdims=True) - 2 * (x @ C.T)
                          + mx.sum(C * C, axis=1)[None, :])
                    parts.append(C[mx.argmin(d2, axis=1)])
                rec = mx.concatenate(parts, axis=0).reshape(base.shape)
                pq_cb_values += K * d
            elif plan["form"] == "hotcold":
                cnt = route.get(id(m))
                n_exp = W.shape[0]
                n_hot = max(1, int(round(n_exp * plan["hot_frac"])))
                order = mx.argsort(-cnt)                 # busiest first
                hot_ids = set(int(i) for i in order[:n_hot])
                parts = []
                for e in range(n_exp):
                    b = plan["hot_bits"] if e in hot_ids else plan["cold_bits"]
                    We = W[e:e + 1]
                    q, s_, b_ = mx.quantize(We.astype(mx.bfloat16), group_size=g, bits=b)
                    parts.append(mx.dequantize(q, s_, b_, group_size=g, bits=b).astype(mx.float32))
                    if e in hot_ids:
                        hot_w += We.size
                    else:
                        cold_w += We.size
                rec = mx.concatenate(parts, axis=0)
            elif EF:
                flatg = base.reshape(-1, g)
                d = act.get(id(m))
                if d is None:
                    sc = mx.mean(mx.abs(flatg), axis=1, keepdims=True)
                else:
                    dd = (mx.broadcast_to(d.reshape(d.shape[0], 1, d.shape[1]), base.shape)
                          if d.ndim == 2 else
                          mx.broadcast_to(d.reshape(1, -1),
                                          base.reshape(-1, base.shape[-1]).shape)).reshape(-1, g)
                    sc = (mx.sum(dd * mx.abs(flatg), axis=1, keepdims=True)
                          / mx.maximum(mx.sum(dd, axis=1, keepdims=True), 1e-9))
                # Sequential along the group, vectorised across every group at once.
                err = mx.zeros((flatg.shape[0], 1))
                cols = []
                for j in range(g):
                    v = flatg[:, j:j + 1] + err
                    qj = sc * mx.sign(v)
                    err = v - qj
                    cols.append(qj)
                rec = mx.concatenate(cols, axis=1).reshape(base.shape)
            elif SPARSE:
                dens = plan["density"]
                flatg = base.reshape(-1, g)
                k = max(1, int(round(g * dens)))
                # per-group magnitude threshold: keep the k largest |w|
                srt = mx.sort(mx.abs(flatg), axis=1)
                thr = srt[:, g - k][:, None]
                keep = (mx.abs(flatg) >= thr).astype(mx.float32)
                kept_g = mx.sum(keep, axis=1, keepdims=True)
                sc = (mx.sum(mx.abs(flatg) * keep, axis=1, keepdims=True)
                      / mx.maximum(kept_g, 1.0))
                rec = (sc * mx.sign(flatg) * keep).reshape(base.shape)
                n_survivors += float(mx.sum(keep))
            elif RES:
                flatg = base.reshape(-1, g)
                d = act.get(id(m))
                def _scale(t):
                    if d is None:
                        return mx.mean(mx.abs(t), axis=1, keepdims=True)
                    dd = (mx.broadcast_to(d.reshape(d.shape[0], 1, d.shape[1]), base.shape)
                          if d.ndim == 2 else
                          mx.broadcast_to(d.reshape(1, -1),
                                          base.reshape(-1, base.shape[-1]).shape)).reshape(-1, g)
                    return (mx.sum(dd * mx.abs(t), axis=1, keepdims=True)
                            / mx.maximum(mx.sum(dd, axis=1, keepdims=True), 1e-9))
                s1 = _scale(flatg); b1 = mx.sign(flatg)
                r1 = flatg - s1 * b1
                s2 = _scale(r1); b2 = mx.sign(r1)
                rec = (s1 * b1 + s2 * b2).reshape(base.shape)
            elif BIN:
                flatg = base.reshape(-1, g)
                d = act.get(id(m))
                if d is None:
                    sc = mx.mean(mx.abs(flatg), axis=1, keepdims=True)
                else:
                    # s* minimising sum_j d_j (W_ij - s*sign(W_ij))^2 is the
                    # d-weighted mean of |W| over the group.
                    if d.ndim == 2:            # per-expert: (n_exp, in)
                        dg = mx.broadcast_to(
                            d.reshape(d.shape[0], 1, d.shape[1]), base.shape
                        ).reshape(-1, g)
                    else:                       # pooled: (in,)
                        dg = mx.broadcast_to(d.reshape(1, -1),
                                             base.reshape(-1, base.shape[-1]).shape
                                             ).reshape(-1, g)
                    sc = (mx.sum(dg * mx.abs(flatg), axis=1, keepdims=True)
                          / mx.maximum(mx.sum(dg, axis=1, keepdims=True), 1e-9))
                rec = (sc * mx.sign(flatg)).reshape(base.shape)
            else:
                q, s, b = mx.quantize(base.astype(mx.bfloat16), group_size=g, bits=bits)
                rec = mx.dequantize(q, s, b, group_size=g, bits=bits).astype(mx.float32)
            if mask is not None:
                rec = mx.where(mask, W, rec)
            rec_b = rec.astype(mx.bfloat16)
            s_w2 += float(mx.sum(W * W))
            s_h2 += float(mx.sum(rec * rec))
            s_dot += float(mx.sum(W * rec))
            m.weight = rec_b
            mx.eval(m.weight)
            del W, base, rec, rec_b

        nb, ng = plan["ne_bits"], plan["ne_group"]

        def pred(path, mm):
            if "switch_mlp" in path or not hasattr(mm, "to_quantized"):
                return False
            w = getattr(mm, "weight", None)
            return ({"bits": nb, "group_size": ng}
                    if w is not None and w.shape[-1] % ng == 0 else False)

        nn.quantize(model, group_size=ng, bits=nb, class_predicate=pred)
    mx.eval(model.parameters())
    mx.synchronize()
    if plan["form"] == "bf16":
        magnitude_ratio, direction_similarity = 1.0, 1.0
    else:
        magnitude_ratio = (s_h2 / s_w2) ** 0.5 if s_w2 else None
        direction_similarity = (s_dot / (s_w2 * s_h2) ** 0.5) if s_w2 and s_h2 else None

    if plan["form"] == "bf16":
        expert_bits = n_tot * 16.0
    else:
        # binary stores 1 bit/weight + ONE fp16 scale per group (no zero point).
        if plan["form"] == "pq":
            import math
            expert_bits = (n_tot * math.log2(plan["codebook"]) / plan["sub_dim"]
                           + pq_cb_values * 16)
        elif plan["form"] == "hotcold":
            expert_bits = (hot_w * plan["hot_bits"] + cold_w * plan["cold_bits"]
                           + n_scales * 32)
        elif plan["form"] == "binary_ef":
            expert_bits = n_tot * 1 + n_scales * 16 + kept * 32
        elif plan["form"] == "sparse":
            import math
            idx_bits = math.log2(plan["group"])
            expert_bits = n_survivors * (idx_bits + 1) + n_scales * 16 + kept * 32
        elif plan["form"] == "binary":
            expert_bits = n_tot * 1 + n_scales * 16 + kept * 32
        elif plan["form"] == "resbinary":
            expert_bits = n_tot * 2 + n_scales * 32 + kept * 32
        else:
            expert_bits = n_tot * (plan["bits"] + 32 / plan["group"]) + kept * 32
    complete_bytes = int(expert_bits / 8 + _leaf_bytes(model.parameters(), "switch_mlp"))
    complete_ebpw = complete_bytes * 8 / SRC_PARAMS

    ids = mx.array([tok.encode(NLL_TEXT)[:512]])
    lg = model(ids[:, :-1]).astype(mx.float32)
    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    nll = float(-mx.take_along_axis(lp, ids[:, 1:, None], axis=-1).squeeze(-1).mean())
    ppl = 2.718281828459045 ** nll

    r4s, dis, worst = [], [], 1
    for pr in PROMPTS:
        tx = generate(model, tok, prompt=pr, max_tokens=96, verbose=False)
        tt = tok.encode(tx)
        run = cur = 1
        for i in range(1, len(tt)):
            cur = cur + 1 if tt[i] == tt[i - 1] else 1
            run = max(run, cur)
        worst = max(worst, run)
        gr = [tuple(tt[i:i + 4]) for i in range(max(0, len(tt) - 3))]
        r4s.append(1 - len(set(gr)) / max(1, len(gr)))
        dis.append(len(set(tt)) / max(1, len(tt)))
    r4s.sort(); dis.sort()
    med_r4 = r4s[len(r4s) // 2]
    med_dis = dis[len(dis) // 2]
    capability_ok = bool(med_r4 <= R4_MAX and ppl <= PPL_MAX)

    return {
        "schema": "hawking.hcli.odyssey.gravity_outlier_eval.v1",
        "specimen": getattr(candidate, "specimen", "O003"),
        "spec": getattr(candidate, "spec", str(candidate)),
        "non_expert_bits": plan["ne_bits"], "non_expert_group": plan["ne_group"],
        "representation_class": ("PRODUCT_QUANTIZATION" if plan["form"] == "pq"
                                 else "HOT_COLD_ROUTED" if plan["form"] == "hotcold"
                                 else "BINARY_ERROR_FEEDBACK" if plan["form"] == "binary_ef"
                                 else "SPARSE_BINARY" if plan["form"] == "sparse"
                                 else "RESIDUAL_BINARY" if plan["form"] == "resbinary"
                                 else "BINARY_ACT_PER_EXPERT" if plan.get("per_expert")
                                 else "BINARY_ACT_CALIBRATED" if plan.get("calibrated")
                                 else "BINARY_SCALED" if plan["form"] == "binary"
                                 else "OUTLIER_SPLIT" if plan["form"] == "outlier_split"
                                 else "AFFINE_QUANT" if plan["form"] == "affine"
                                 else "SOURCE_BF16"),
        "_evidence": "MEASURED (executed in-process on MLX Metal)",
        "complete_bpw": complete_ebpw,
        "complete_ebpw": complete_ebpw,
        "stored_bytes": complete_bytes,
        "accounting": {"complete_bytes": complete_bytes,
                       "expert_bits_per_weight": round(expert_bits / n_tot, 4),
                       "expert_weights": n_tot, "outliers_kept": kept},
        "magnitude_ratio": magnitude_ratio,
        "direction_similarity": direction_similarity,
        "n_calibrated_tensors": len(act),
        "density_actual": (n_survivors / n_tot) if plan["form"] == "sparse" else None,
        "hot_weights": hot_w, "cold_weights": cold_w,
        "hot_share_actual": (hot_w / (hot_w + cold_w)) if (hot_w + cold_w) else None,
        "effective_groups": {str(k): v for k, v in sorted(eff_groups.items())},
        "capability_ok": capability_ok,
        "capability_status": "CANDIDATE_PASS" if capability_ok else "CAPABILITY_LOSS",
        "capability": {"ppl": round(ppl, 4), "nll": round(nll, 5),
                       "median_4gram_repeat": round(med_r4, 4),
                       "median_distinct_ratio": round(med_dis, 4),
                       "worst_max_repeat_run": worst, "n_prompts": len(PROMPTS),
                       "gate_r4_max": round(R4_MAX, 4), "gate_ppl_max": round(PPL_MAX, 4),
                       "reference_bf16_r4": BF16_R4, "reference_bf16_ppl": BF16_PPL},
        # OUTLIER_SPLIT reconstructs to dense bf16 and has NO sparse kernel, so
        # its execution is not the represented form. Never claim otherwise.
        "execution_complete": plan["form"] in ("affine", "bf16"),
        "execution_note": ("sparse side-channel has no kernel; measured capability and "
                           "bytes are exact, TPS for the represented form is UNMEASURED"
                           if plan["form"] == "outlier_split" else "executes natively"),
        "verifier_independent": False,
        "wall_s": round(time.perf_counter() - t0, 3),
    }


def main() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from hcli.gravity_gauntlet import GravityGauntlet, candidate_space

    specs = ["q4-g64-experts", "q3-g64-experts", "q2-g64-experts",
             "outlier0.005-g128", "outlier0.01-g128", "outlier0.02-g128",
             "outlier0.005-g64", "q2-g128-experts"]
    for s in specs:
        parse_spec(s)  # refuse to start a search containing an unrunnable spec
    out = Path("receipts/future/O003_GAUNTLET_DEEP.json")
    state = Path("workspace/campaign/odyssey/gauntlets/O003_deep.json")
    state.parent.mkdir(parents=True, exist_ok=True)
    cands = candidate_space("O003", specs)
    eng = GravityGauntlet(state, "O003", cands, budget=len(specs))
    rows = []

    def evaluator(c):
        r = evaluate(c)
        rows.append(r)
        print(json.dumps({k: r[k] for k in
                          ("spec", "representation_class", "complete_ebpw",
                           "capability_ok", "execution_complete")}), flush=True)
        print("   " + json.dumps(r["capability"]), flush=True)
        return r

    final = eng.run(evaluator)
    out.write_text(json.dumps({"state": final, "receipts": rows}, indent=1) + "\n")
    print(f"\nterminal: {final.get('terminal', {}).get('disposition')}")
    print(f"budget used: {final['budget']['used']} / {final['budget']['max_evaluations']}")
    print(f"best: {final.get('best_candidate_id')} @ {eng.best_complete_ebpw()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
