#!/usr/bin/env python3.12
"""Qwen3-235B-A22B Gravity campaign controller - durable, detachable, restartable.

The deliverable the integrator launches. It asks ONE question per ladder rung at REAL fidelity:
does the packed model still behave like the parent? Nothing here scores weight-space
reconstruction error; the sealed GPT-OSS-120B lesson is that weight-space proxy != capability, so
every non-parent row carries a REAL parent-vs-packed forward through the full 94-block Qwen3-MoE
stack with reconstructed experts substituted in, and the verdict is
    PASS iff mean_sym_kl <= 0.10 AND next_token_argmax_agreement >= 0.95.

CANDIDATE LADDER (decisive-first; a truncated run is still conclusive)
  R0_parent          real parent reference (logits persisted to disk, never recomputed)
  R1_c1_corrected    gate/up product_quant d32 s8 k8 ; down d16 s2 k16      (whole 0.684)
  R2_subhalf_best    G_sg_d16k1024s1 + D_sg_d64k1024s1                      (whole 0.4930)
  R3_routing_aware   R2 with the coldest 50% one step harsher               (whole ~0.3554)
  R4_highdim_vq      gate/up shared grammar dim32 k65536 stages1 (chunked k-means)
  R5_rownorm_strat   row-norm-stratified codebooks on gate/up (94% of rows collapse to one codeword)

THE THREE MEASURED FIXES THIS CONTROLLER EXISTS TO CARRY (from the completed audit)

 1. LOOP RESTRUCTURE - expert OUTER, candidate INNER. The baseline wave ran one candidate's whole
    forward at a time, so the same ~105 distinct experts per layer were re-streamed once per
    candidate (2.7 h of pure re-reading). Here every variant advances through the 94 blocks in
    LOCKSTEP: at layer L each expert in the union of everyone's routing is read and bf16-converted
    exactly ONCE, then packed for every candidate from that one resident copy and applied to every
    routed position. Attention/router/norm tensors are likewise read once per layer instead of once
    per (variant, prompt) via a layer-scoped reader cache.

 2. PARENT LOGITS PERSIST TO DISK (53 MB for the whole holdout) and are loaded BEFORE the
    checkpoint-skip decision, not after it. The baseline skipped on an existing checkpoint before
    filling orig_cache and never persisted parent logits, so every restart re-ran all 6 parent
    forwards (1.33 h burned per restart). Here a restart with sealed parent logits drops the parent
    variant out of the lockstep entirely.

 3. NO READER CHURN. The SafetensorsIndexReader's persistent per-shard mmap is already correct
    (118 headers in 0.485 s); it is built once for the whole campaign and never closed between
    candidates. Only the layer-scoped tensor cache is reset.

 Plus: chunked k-means. gravity_forge._kmeans materializes the full [N,k] distance matrix in one
 shot (OOMs at k=65536) and updates centroids with boolean-mask indexing (gravity_forge.py:167-168),
 which forces a per-iteration MPS device sync. _kmeans_chunked here blocks the distance matrix and
 updates with torch.where over index_add_ counts - no mask indexing, no sync stall. gravity_forge is
 NOT modified; this is a local replacement used only by the shared-grammar packers.

MEMORY. Expert residency is a bounded_cache.PressureAwareCache honouring HAWKING_CACHE_MAX_GB
(default 64 here) and HAWKING_CACHE_FLOOR_GB (default 12) - fill the 103 GB box hard, never spill
into swap.

DURABILITY. Campaign dir QWEN_GRAVITY/, lease com.hawking.qwen_gravity under the one-heavy-lease
law, heartbeat, per-row sha256 checkpoints with resume-skip, a rolling layer-boundary residual-state
checkpoint so a crash at hour 40 resumes at the layer it died on, state file, run/detach/status CLI,
caffeinate + start_new_session detach.

SOURCE GATE. run() checks source_present() FIRST: shards absent seals WAITING_SOURCE and exits 0,
so a premature launch is a clean no-op, never a crash and never a lease.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Memory policy must be set before any PressureAwareCache is constructed.
os.environ.setdefault("HAWKING_CACHE_MAX_GB", "64")
os.environ.setdefault("HAWKING_CACHE_FLOOR_GB", "12")
os.environ.setdefault("HAWKING_CACHE_MIN_ENTRIES", "2")

import bounded_cache as bc          # noqa: E402
import gravity_forge as gf          # noqa: E402
import qwen3_moe_adapter as A
import qwen_function_aware_codec as FAC
import qwen_structural_plan as SP       # noqa: E402
import qwen_real_forward as Q       # noqa: E402
import qwen_subhalfbit_search as SHB  # noqa: E402

ROOT = Path(_HERE).resolve().parents[1]
CAMPAIGN = ROOT / "reports/condense/general_frontier/QWEN_GRAVITY"
LEASES = CAMPAIGN / "leases"
HEARTBEAT = CAMPAIGN / "heartbeat"
CHECKPOINTS = CAMPAIGN / "checkpoints"
PARENT_LOGITS = CAMPAIGN / "parent_logits"
CONTROLLER = CAMPAIGN / "controller"
STATE_PATH = CAMPAIGN / "QWEN_GRAVITY_STATE.json"
WAITING_RECEIPT = CAMPAIGN / "QWEN_GRAVITY_WAITING_SOURCE.json"
PASS_STATE_NPZ = CAMPAIGN / "pass_state.npz"
PASS_STATE_JSON = CAMPAIGN / "pass_state.json"
LEASE_PATH = LEASES / "qwen_gravity.lease"
HB_PATH = HEARTBEAT / "qwen_gravity.heartbeat.json"
LABEL = "com.hawking.qwen_gravity"
PY = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
TOKENIZER_PATH = ROOT / "models/qwen3-235b-a22b/_meta/tokenizer.json"

# One-heavy-lease law: refuse to start while any other heavy campaign lease is held live.
OTHER_LEASE_GLOBS = ["reports/condense/general_frontier/QWEN_TRANSFER/leases/qwen_transfer.lease",
                     "reports/condense/general_frontier/DOCTOR_CAMPAIGN/leases/doctor_campaign.lease",
                     "reports/condense/general_frontier/CORRECTION_WAVE/leases/correction_wave.lease",
                     "reports/condense/general_frontier/G4/leases/frontier_g4.lease",
                     "reports/condense/second_light/leases/second_light.lease"]

PROMOTE_KL = 0.10
PROMOTE_ARGMAX_AGREE = 0.95
DEPLOY_CLUSTER = SHB.DEPLOY_CLUSTER      # 128 experts share one codebook in the real artifact

# 6-prompt holdout (88 tokens), tokenized with the Qwen tokenizer at run time.
HOLDOUT: list[dict[str, str]] = [
    {"id": "gen_paris", "domain": "factual", "text": "The capital of France is"},
    {"id": "gen_science", "domain": "general", "text": "Water is made of two hydrogen atoms and one"},
    {"id": "code_py", "domain": "code", "text": "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n -"},
    {"id": "math_add", "domain": "math", "text": "If a train travels 60 miles in 2 hours, its average speed is 30"},
    {"id": "reason_syllogism", "domain": "reasoning", "text": "All humans are mortal. Socrates is a human. Therefore, Socrates is"},
    {"id": "instr_list", "domain": "instruction", "text": "Here are three primary colors: red, green, and"},
]

# ── the ladder ────────────────────────────────────────────────────────────────────────────────
# Data-driven so rungs can be added: each packed rung carries a gate_up spec and a down spec, and
# optionally a cold_frac + cold_* pair (routing-frequency-aware allocation). Organ inversion is
# respected throughout: gate/up is the SENSITIVE organ and gets the higher index rate, down is
# robust and gets the harshest.
LADDER: dict[str, dict[str, Any]] = {
    "R0_parent": {
        "kind": "parent",
        "note": "source-native bf16 experts; the divergence + self-PPL reference. Logits persisted.",
    },
    "R1_c1_corrected": {
        "kind": "packed",
        "gate_up": {"family": "product_quant", "dim": 32, "subspaces": 8, "k": 8},
        "down": {"family": "product_quant", "dim": 16, "subspaces": 2, "k": 16},
        "note": "the sealed C1 corrected allocation, organ-inverted (whole-model 0.684).",
    },
    "A1_1p0": {
        "kind": "packed",
        "gate_up": {"family": "product_quant", "dim": 32, "subspaces": 8, "k": 32},
        "down": {"family": "product_quant", "dim": 16, "subspaces": 2, "k": 16},
        "note": "BRACKET HIGH (~1.02 whole): gate/up 1.25 bpw, down 0.5 bpw. Organ-inverted. The "
                "explicit 1-BPW target and the best shot at a genuine capability PASS. A pass here "
                "plus a fail lower down brackets the cliff instead of only proving collapse.",
    },
    "A2_0p85": {
        "kind": "packed",
        "gate_up": {"family": "product_quant", "dim": 32, "subspaces": 8, "k": 16},
        "down": {"family": "product_quant", "dim": 16, "subspaces": 2, "k": 16},
        "note": "BRACKET HIGH-SUB (~0.85 whole): gate/up 1.0 bpw, down 0.5 bpw. Bisects between "
                "A1_1p0 and R1_c1_corrected once their verdicts are known.",
    },
    "R2_subhalf_best": {
        "kind": "packed",
        "gate_up": {"family": "shared_grammar", "dim": 16, "k": 1024, "stages": 1},
        "down": {"family": "shared_grammar", "dim": 64, "k": 1024, "stages": 1},
        "note": "sub-half-bit search #1 G_sg_d16k1024s1 + D_sg_d64k1024s1 (whole-model 0.4930).",
    },
    "R3_routing_aware": {
        "kind": "packed",
        "gate_up": {"family": "shared_grammar", "dim": 16, "k": 1024, "stages": 1},
        "down": {"family": "shared_grammar", "dim": 64, "k": 1024, "stages": 1},
        "cold_frac": 0.25,
        "cold_gate_up": {"family": "shared_grammar", "dim": 32, "k": 1024, "stages": 1},
        "cold_down": {"family": "shared_grammar", "dim": 128, "k": 1024, "stages": 1},
        "note": "ALIVE lever 1: coldest QUARTILE one step harsher (double the VQ dim at fixed k). "
                "Deliberately the quartile band, NOT the median split: the 88-token calibration is "
                "only 63.6 percent stable at the median (bootstrap), and the count-split disagrees "
                "with the lower-variance softmax_mass split on 15 percent of cells. The instability "
                "is concentrated at the median, so the coldest quartile is the defensible partition. "
                "Membership from qwen_routing_calibration.load_partition when present.",
    },
    "R4_highdim_vq": {
        "kind": "packed",
        "gate_up": {"family": "shared_grammar", "dim": 32, "k": 65536, "stages": 1},
        "down": {"family": "shared_grammar", "dim": 64, "k": 1024, "stages": 1},
        "note": "ALIVE lever 2: space-filling gain at fixed index rate 0.5. k=65536 needs the "
                "chunked k-means (gravity_forge._kmeans OOMs); codebook amortizes to 0.042 BPW.",
    },
    "S64_structural": {
        "kind": "packed",
        "keep_experts": 64,
        "gate_up": {"family": "function_aware", "dim": 8, "k": 1024, "stages": 2},
        "down": {"family": "function_aware", "dim": 16, "k": 1024, "stages": 1},
        "note": "STRUCTURAL (Lane D M05+M09 x Lane A M01'). The inventory is a free variable under a "
                "fixed complete ceiling: keeping the 64 hottest experts per layer and spending the "
                "freed budget on the survivors is BUDGET-NEUTRAL, so gate/up rises 0.625 -> 2.5 "
                "index bpw and its rate-distortion floor falls 0.6484 -> 0.1768. Complete 0.9484 "
                "BPW by qwen_structural_plan.ledger, legal under the 1/1 ceiling. Survivors are "
                "coded with the scale-invariant function-aware codec (per-row bf16 scale billed). "
                "The router's top-k is taken over survivors ONLY: omitted experts are masked to "
                "-inf before the top-k, then the k weights renormalize as usual. That is a real "
                "change to the model and only the forward may judge it.",
    },
    "D1_route_only": {
        "kind": "packed",
        "diagnostic_only": True,
        "keep_experts": 64,
        "gate_up": {"family": "passthrough"},
        "down": {"family": "passthrough"},
        "note": "CAUSAL CONTROL for the S1 decomposition, isolating ROUTING loss. Every expert is "
                "served at source bf16 (perfect reconstruction) while the router is masked to the "
                "SAME 64-expert survivor set S64_structural uses. Whatever quality this loses is "
                "caused by omitted routing mass and nothing else. DIAGNOSTIC ONLY: its experts are "
                "uncompressed, so its complete BPW is 16 and it is permanently ineligible for "
                "promotion. It exists to answer a causal question, not to be a candidate.",
    },
    "D2_recon_only": {
        "kind": "packed",
        "diagnostic_only": True,
        "gate_up": {"family": "function_aware", "dim": 8, "k": 1024, "stages": 2},
        "down": {"family": "function_aware", "dim": 16, "k": 1024, "stages": 1},
        "note": "CAUSAL CONTROL for the S1 decomposition, isolating RECONSTRUCTION loss. The full "
                "128-expert router runs UNMASKED (no routing mass is lost at all) while every "
                "expert is packed at the exact rates S64_structural gives its survivors. Whatever "
                "quality this loses is caused by reconstruction error and nothing else. "
                "DIAGNOSTIC ONLY: coding 128 experts at the survivor rate costs about 1.9 complete "
                "BPW, which is ILLEGAL under the 1/1 ceiling. It is permanently ineligible for "
                "promotion and is never reported as a candidate. Measuring a causal control above "
                "the ceiling is not the same as proposing an artifact above the ceiling, and this "
                "one is fenced so it can never become one.",
    },
    "S2A_adaptive_k": {
        "kind": "packed",
        "adaptive_program": "QWEN235B_ADAPTIVE_EXPERT_PROGRAM.json",
        "note": "GENERATION S2A. Per-layer expert inventory K_l and per-layer organ rungs chosen by "
                "an EXACT global byte auction (qwen_adaptive_k), not a hard-coded 64. The auction "
                "minimises sum_l [(1 - C_l(K)) * MISS_COST + C_l(K) * recon_err(rate)] subject to "
                "the complete <= 1/1 ceiling, solved as a separable knapsack by Lagrangian "
                "bisection - exact, not greedy. MISS_COST = 1.0 is a MEASUREMENT, not a guess: "
                "Lane E showed an omitted expert is not reconstructible from survivors (best "
                "single survivor median held-out relative error 0.885-0.995, i.e. the trivial zero "
                "predictor). Result: K spans 48..96 with mean 65.4 (7 layers at 48, 77 at 64, 5 at "
                "80, 5 at 96), and 84 of 94 layers buy a RICHER down rung (1.25) than S1 used "
                "(0.625), funded by cheaper rungs on the remaining 10. Complete 0.996853694 BPW "
                "(exact 915445149/918334510), legal. Predicted mean layer error 0.46213 vs 0.521819 "
                "for uniform-64 under the identical predictor and ledger. That predictor orders "
                "candidates; it does not select. Only this forward does.",
    },
    "S64_gamma": {
        "kind": "packed",
        "keep_experts": 64,
        "gate_up": {"family": "function_aware", "dim": 8, "k": 1024, "stages": 2, "gamma_weighted": True},
        "down": {"family": "function_aware", "dim": 16, "k": 1024, "stages": 1,
                 "doctor": {"dim": 16, "k": 1024, "stages": 1, "protect_frac": 0.5}},
        "note": "S64_doctor + DATA-FREE output-aware coding on gate/up. The Lane C adversary showed "
                "that the ~60 percent layer-0 output-error cut credited to routed-token calibration "
                "is 83 percent recoverable from h = post_attention_layernorm.weight^2 alone "
                "(log-corr 0.9918), so the load-bearing quantity is the LayerNorm gamma, not the "
                "activations. gamma is ALREADY shipped native as a pass-through tensor, so this "
                "costs ZERO additional bits and needs ZERO calibration: complete BPW is identical "
                "to S64_doctor at 0.999769787. Applied to gate/up ONLY, because their input IS the "
                "post-attention-layernorm output; down_proj's input is the SwiGLU intermediate and "
                "gamma does not describe it. Measured gamma^2 anisotropy over all 94 layers: mean "
                "0.293 decades, 45.7 percent of layers above 0.3, layer 0 an outlier at 1.843. "
                "Expected gain is large at layer 0 and a few percent at median depth. The trade is "
                "real and must be judged by the forward, not asserted: output-aware coding makes "
                "WEIGHT-space error substantially worse (0.237 -> 0.806 on L0 gate) while cutting "
                "output error, which is the correct direction only if output is what survives.",
    },
    "S64_doctor": {
        "kind": "packed",
        "keep_experts": 64,
        "gate_up": {"family": "function_aware", "dim": 8, "k": 1024, "stages": 2},
        "down": {"family": "function_aware", "dim": 16, "k": 1024, "stages": 1,
                 "doctor": {"dim": 16, "k": 1024, "stages": 1, "protect_frac": 0.5}},
        "note": "SAME-BUDGET DOCTOR (Lane B M10) on top of S64_structural. The structural arm "
                "leaves 0.0516 BPW of headroom under the 1/1 ceiling; this spends ALL of it on a "
                "diagnosis-driven sparse residual over down_proj, the organ whose rate-distortion "
                "floor is now the worst in the artifact. Protected rows are chosen by measured "
                "RELATIVE residual energy after the base pass - where the base representation "
                "actually lost the function - not uniformly and not by row norm. Doctor bytes are "
                "billed INSIDE the ceiling: correction indices, amortized correction codebook, one "
                "bf16 scale per protected row, and a one-bit-per-row protection bitmap. Measured "
                "on real down_proj at layer 46: rel_error 0.7038 -> 0.6074 for +0.319 organ bpw.",
    },
    "R5_rownorm_strat": {
        "kind": "packed",
        "gate_up": {"family": "shared_grammar", "dim": 16, "k": 1024, "stages": 1, "strata": 2},
        "down": {"family": "shared_grammar", "dim": 64, "k": 1024, "stages": 1},
        "note": "ALIVE lever 3: row-norm-stratified codebooks. Row norms span 1e-5..0.91 so 94% of "
                "gate/up rows collapse onto ONE codeword; separate strata get their own codebook. "
                "Billed: extra codebooks amortized over the cluster + 1 stratum bit per row.",
    },
}
# Decisive-first = BISECTION, not a march down. Parent, then the 1-BPW target (best shot at a real
# PASS), then the aggressive end. If A1 passes and R2 fails, the cliff is already bracketed and rows
# 4-5 bisect it. If R2 passes, the anchors are moot and the remaining budget goes to the moonshots.
# A truncated run therefore still yields a cliff location rather than only "everything collapsed".
LADDER_ORDER = ["R0_parent", "S64_structural", "S64_doctor", "D1_route_only",
                "D2_recon_only", "S2A_adaptive_k", "S64_gamma", "A1_1p0", "R2_subhalf_best", "R1_c1_corrected",
                "A2_0p85", "R4_highdim_vq", "R5_rownorm_strat", "R3_routing_aware"]

ORGANS = ("gate", "up", "down")
_GROUP = {"gate": "gate_up", "up": "gate_up", "down": "down"}


# ── small utilities ───────────────────────────────────────────────────────────────────────────
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def _pid_alive(pid: int) -> bool:
    try:
        if pid is None or int(pid) <= 0:      # 0 would broadcast to the process group
            return False
        os.kill(int(pid), 0); return True
    except PermissionError:
        return True
    except Exception:
        return False


def _tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def _ladder_sha() -> str:
    return _sha(LADDER)[:16]


# ── chunked k-means (replaces gravity_forge._kmeans for the shared-grammar lane) ───────────────
def _row_chunk(k: int) -> int:
    """Rows per distance-matrix block. Caps the [chunk, k] intermediate at ~16M floats (64 MB)."""
    return max(64, int(16_000_000 // max(1, k)))


def _assign_chunked(v, cb):
    """argmin over centroids, blocked over rows so [N, k] is never materialized (k=65536 OOMs)."""
    torch = gf._torch()
    n, k = v.shape[0], cb.shape[0]
    step = _row_chunk(k)
    if n <= step:
        return gf._assign(v, cb)
    out = torch.empty(n, dtype=torch.long, device=v.device)
    cbn = (cb * cb).sum(1)
    cbt = cb.t().contiguous()
    for i in range(0, n, step):
        c = v[i:i + step]
        d2 = (c * c).sum(1, keepdim=True) - 2.0 * (c @ cbt) + cbn
        out[i:i + step] = d2.argmin(1)
    return out


def _kmeans_chunked(v, k: int, *, iters: int = 6, seed: int = 0):
    """Lloyd on v:[N,D] -> centroids [k,D]. Two deviations from gravity_forge._kmeans, both measured:
    the distance matrix is blocked (so large k does not OOM), and the centroid update uses
    torch.where over the index_add_ counts instead of boolean-mask indexing - the mask form at
    gravity_forge.py:167-168 forces a device sync every iteration, which is 100% of pack time."""
    torch = gf._torch()
    n = v.shape[0]
    k = int(min(k, n))
    g = torch.Generator(device="cpu").manual_seed(seed)
    cb = v[torch.randperm(n, generator=g)[:k].to(v.device)].clone()
    ones = torch.ones(n, device=v.device, dtype=v.dtype)
    for _ in range(iters):
        idx = _assign_chunked(v, cb)
        new = torch.zeros_like(cb)
        cnt = torch.zeros(k, device=v.device, dtype=v.dtype)
        new.index_add_(0, idx, v)
        cnt.index_add_(0, idx, ones)
        cb = torch.where(cnt.unsqueeze(1) > 0, new / cnt.clamp(min=1.0).unsqueeze(1), cb)
    return cb


# ── shared-grammar fit / apply (cluster codebook, optional row-norm strata) ────────────────────
def _strata_rows(w: np.ndarray, strata: int) -> list[np.ndarray]:
    """Partition matrix rows by row L2 norm into `strata` equal-size groups. The stratum id is
    stored per row (billed), so the split may be per-expert."""
    if strata <= 1:
        return [np.arange(w.shape[0])]
    order = np.argsort(np.linalg.norm(w, axis=1))
    return [np.sort(part) for part in np.array_split(order, strata)]


def _fit_grammar(spec: dict[str, Any], mats: list[np.ndarray], seed: int):
    """Fit the shared codebook stack for one organ group from a sample of the layer's expert
    cluster. Returns [stratum][stage] -> centroid tensor. Deployment semantics: ONE codebook per
    (layer, organ group) shared by all 128 experts, which is exactly how it is billed."""
    torch = gf._torch()
    dev = gf._device()
    d, k, stages = int(spec["dim"]), int(spec["k"]), int(spec.get("stages", 1))
    strata = int(spec.get("strata", 1))
    books = []
    for s in range(strata):
        chunks = []
        for m in mats:
            rows = _strata_rows(m, strata)[s]
            chunks.append(torch.from_numpy(np.ascontiguousarray(m[rows], np.float32))
                          .to(dev).reshape(-1, d))
        pool = torch.cat(chunks, 0)
        del chunks
        res = pool.clone()
        cbs = []
        for st in range(stages):
            cb = _kmeans_chunked(res, k, iters=6, seed=seed + 17 * s + st)
            res = res - cb[_assign_chunked(res, cb)]
            cbs.append(cb)
        del pool, res
        books.append(cbs)
    return books


def _apply_grammar(spec: dict[str, Any], books, w: np.ndarray) -> np.ndarray:
    """Encode one expert tensor against the already-fitted shared codebooks (indices only)."""
    torch = gf._torch()
    dev = gf._device()
    d = int(spec["dim"])
    strata = int(spec.get("strata", 1))
    out = np.empty_like(w, dtype=np.float32)
    parts = _strata_rows(w, strata)
    for s, cbs in enumerate(books):
        rows = parts[s]
        sub = torch.from_numpy(np.ascontiguousarray(w[rows], np.float32)).to(dev).reshape(-1, d)
        rec = torch.zeros_like(sub)
        res = sub
        for cb in cbs:
            idx = _assign_chunked(res, cb)
            q = cb[idx]
            rec = rec + q
            res = res - q
        # scatter back on the numpy side: aten::index_copy is not implemented for MPS
        out[rows] = rec.reshape(len(rows), w.shape[1]).detach().cpu().numpy()
    return out


_GAMMA: dict[int, Any] = {}


def _gamma_importance(rd, layer: int):
    """h = post_attention_layernorm.weight^2, mean-normalized. DATA-FREE and already shipped.

    For gate/up the organ input is exactly the post-attention-LayerNorm output, so gamma^2 is the
    diagonal of that input's second moment up to the (isotropic) normalized residual - which makes
    it the correct per-column weight for OUTPUT error, obtainable with no calibration pass at all.
    """
    hit = _GAMMA.get(layer)
    if hit is None:
        g = rd.bf16(f"model.layers.{layer}.post_attention_layernorm.weight").astype(np.float64)
        h = g * g
        m = float(h.mean())
        hit = (h / m).astype(np.float32) if m > 0 else np.ones_like(h, np.float32)
        _GAMMA[layer] = hit
    return hit


def _mps_gc() -> None:
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ── structural survivor sets (Lane D) ─────────────────────────────────────────────────────────
_ADAPTIVE: dict[str, dict[int, dict]] = {}


def _adaptive_program(cand: str) -> dict[int, dict] | None:
    """Per-layer {K, gate_rung, down_rung} for a candidate driven by the byte auction."""
    path = LADDER[cand].get("adaptive_program")
    if not path:
        return None
    hit = _ADAPTIVE.get(cand)
    if hit is None:
        doc = json.loads((ROOT / path).read_text())
        hit = {int(r["layer"]): r for r in doc["per_layer"]}
        _ADAPTIVE[cand] = hit
    return hit


_SURV_CACHE: dict[tuple[str, int], frozenset] = {}
_ROUTING_DOC: dict[str, Any] = {}


def _routing_doc() -> dict[str, Any] | None:
    """The 1200-token calibration if it is on disk, else the sealed 88-token run, else None."""
    if "doc" in _ROUTING_DOC:
        return _ROUTING_DOC["doc"]
    for cand in (ROOT / "reports/subbit_reset/QWEN3_235B_ROUTING_CALIBRATION_1200.json",
                 ROOT / "reports/condense/general_frontier/QWEN3_235B_ROUTING_FREQUENCY.json"):
        if cand.is_file():
            _ROUTING_DOC["doc"] = json.loads(cand.read_text())
            _ROUTING_DOC["path"] = str(cand)
            return _ROUTING_DOC["doc"]
    _ROUTING_DOC["doc"] = None
    return None


def survivor_set(cand: str, layer: int, n_experts: int) -> frozenset | None:
    """The experts a structural candidate keeps at this layer, or None when it keeps all of them.

    Falls back to the hottest-by-index stand-in ONLY if no calibration is on disk, and the receipt
    records which source was used - an allocation fitted on an absent calibration is not evidence.
    """
    prog = _adaptive_program(cand)
    keep = prog[layer]["K"] if prog else LADDER[cand].get("keep_experts")
    if not keep or int(keep) >= n_experts:
        return None
    key = (cand, layer)
    hit = _SURV_CACHE.get(key)
    if hit is not None:
        return hit
    doc = _routing_doc()
    if doc is not None:
        ids = SP.survivors(doc, layer, int(keep))
    else:
        ids = list(range(int(keep)))
    out = frozenset(int(e) for e in ids)
    _SURV_CACHE[key] = out
    return out


def _route_masked(logits: np.ndarray, keep: frozenset | None, top_k: int, norm: bool):
    """top-k over the SURVIVORS only. Omitted experts are masked to -inf BEFORE the top-k, then the
    k surviving weights renormalize exactly as the unmasked router does."""
    if keep is None:
        return Q.route_topk(logits, top_k, norm)
    masked = np.full_like(logits, -np.inf)
    idx = np.fromiter(keep, dtype=np.int64, count=len(keep))
    masked[idx] = logits[idx]
    return Q.route_topk(masked, top_k, norm)


# ── routing-frequency partition (ALIVE lever 1) ───────────────────────────────────────────────
_PARTITION_SOURCE = {"source": None}
_COLD_CACHE: dict[tuple[int, int, float], frozenset] = {}


def _cold_experts(layer: int, n_experts: int, cold_frac: float) -> frozenset:
    """Coldest `cold_frac` of the layer's experts by routing frequency. Uses the real calibration
    when qwen_routing_calibration.load_partition is available; otherwise falls back to the SAME
    deterministic expert-index stand-in the byte accounting uses, and says so in the receipt."""
    key = (layer, n_experts, cold_frac)
    hit = _COLD_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        import qwen_routing_calibration as RC   # type: ignore
        cold = frozenset(int(e) for e in RC.load_partition(layer, cold_frac=cold_frac))
        _PARTITION_SOURCE["source"] = "qwen_routing_calibration.load_partition"
    except Exception:
        _PARTITION_SOURCE["source"] = "deterministic_expert_index_standin"
        cut = int(round(cold_frac * 100))
        cold = frozenset(e for e in range(n_experts) if (e % 100) < cut)
    _COLD_CACHE[key] = cold
    return cold


# ── per-candidate expert packing ──────────────────────────────────────────────────────────────
def _specs_for(cand: str, layer: int, expert: int,
               n_experts: int) -> tuple[dict[str, dict[str, Any]], bool]:
    """Resolve the (hot|cold) spec for each organ group of one expert under one candidate."""
    c = LADDER[cand]
    prog = _adaptive_program(cand)
    if prog is not None:
        row = prog[layer]
        return {"gate_up": dict(SP.GATE_RUNGS[row["gate_rung"]], family="function_aware"),
                "down": dict(SP.DOWN_RUNGS[row["down_rung"]], family="function_aware")}, False
    cold_frac = float(c.get("cold_frac", 0.0))
    cold = bool(cold_frac > 0 and expert in _cold_experts(layer, n_experts, cold_frac))
    if cold:
        return {"gate_up": c.get("cold_gate_up", c["gate_up"]),
                "down": c.get("cold_down", c["down"])}, True
    return {"gate_up": c["gate_up"], "down": c["down"]}, False


def _pack_organ(spec: dict[str, Any], books, w: np.ndarray, seed: int) -> np.ndarray:
    fam = spec["family"]
    if fam == "passthrough":
        return np.ascontiguousarray(w, np.float32)
    if fam == "function_aware":
        # scale-invariant decode with the closed-form optimal per-row scale substituted back in.
        # Identical artifact layout to shared_grammar plus the billed bf16 scale per row.
        wf = np.ascontiguousarray(w, np.float32)
        imp = spec.get("_importance")
        doc = spec.get("doctor")
        if doc:
            base_bk, doc_bk = books
            return FAC.apply_doctor(base_bk, doc_bk, wf, dim=int(spec["dim"]),
                                    doctor_dim=int(doc["dim"]),
                                    protect_frac=float(doc["protect_frac"]))
        return FAC.apply_refit(books, wf, dim=int(spec["dim"]), importance=imp)
    if fam == "shared_grammar":
        return _apply_grammar(spec, books, w)
    if fam in ("product_quant", "transform_pq"):
        fn = gf.pack_product_quant if fam == "product_quant" else gf.pack_transform_pq
        art = fn(np.ascontiguousarray(w, np.float32), dim=int(spec["dim"]),
                 subspaces=int(spec["subspaces"]), k=int(spec["k"]), seed=seed)
        return np.ascontiguousarray(art.recon, np.float32).reshape(w.shape)
    raise ValueError(f"unknown family {fam}")


# ── exact byte ledger: per-organ + whole-model BPW (never expert-only mislabelled) ─────────────
def _acct_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Map a runtime spec onto the exact-bit vocabulary of qwen_subhalfbit_search.expert_bits.
    product_quant is billed as transform_pq: identical index accounting, and transform_pq bills an
    extra 64-bit rotation seed per tensor (1e-5 BPW), so the charge is conservative."""
    s = dict(spec)
    s.pop("strata", None)
    if s["family"] == "product_quant":
        s["family"] = "transform_pq"
    return s


def _strata_extra_bits(inv, cand: str) -> int:
    """Bits the row-norm stratification adds beyond the flat shared-grammar charge: (strata-1) extra
    codebook copies amortized over the deploy cluster, plus ceil(log2 strata) stratum bits per row."""
    c = LADDER[cand]
    total = 0
    for organ_key, classes in (("gate_up", (A.ORGAN_EXP_GATE, A.ORGAN_EXP_UP)),
                               ("down", (A.ORGAN_EXP_DOWN,))):
        spec = c.get(organ_key) or {}
        strata = int(spec.get("strata", 1))
        if strata <= 1 or spec.get("family") != "shared_grammar":
            continue
        cb_bits = (strata - 1) * int(spec.get("stages", 1)) * int(spec["k"]) * int(spec["dim"]) * 16
        for t in inv.tensors:
            if t.organ_class in classes:
                total += cb_bits // DEPLOY_CLUSTER + t.shape[0] * math.ceil(math.log2(strata))
    return int(total)


def _ladder_bpw(inv, cand: str) -> dict[str, Any]:
    """Exact per-organ + whole-model BPW for one rung, over EVERY tensor class (experts, attention,
    embed/head, router, norms). Expert-only rates are never reported as whole-model."""
    c = LADDER[cand]
    if c["kind"] == "parent":
        return {"whole_model_bpw": 16.0, "scope": "bf16 source parent", "organ_bpw": {}}
    if c.get("diagnostic_only"):
        if c["gate_up"].get("family") == "passthrough":
            return {"whole_model_bpw": 16.0, "scope": "DIAGNOSTIC: bf16 experts, masked router",
                    "legal_under_one_bit_ceiling": False, "promotable": False, "organ_bpw": {}}
        led = SP.ledger(inv, 128, c["gate_up"], c["down"], None)
        return {"whole_model_bpw": led["complete_bpw"],
                "legal_under_one_bit_ceiling": led["legal_under_one_bit_ceiling"],
                "promotable": False,
                "scope": "DIAGNOSTIC: all 128 experts at the survivor rate, ILLEGAL as an artifact",
                "organ_bpw": {}}
    if c.get("adaptive_program"):
        doc = json.loads((ROOT / c["adaptive_program"]).read_text())
        led = doc["ledger"]
        return {"whole_model_bpw": led["complete_bpw"],
                "complete_bpw_exact": led["complete_bpw_exact"],
                "legal_under_one_bit_ceiling": led["legal_under_one_bit_ceiling"],
                "complete_bytes": math.ceil(led["complete_bits"] / 8),
                "K_histogram": doc["K_histogram"], "rung_histogram": doc["rung_histogram"],
                "predicted_mean_layer_error": led["predicted_mean_layer_error"],
                "scope": "complete artifact, per-layer adaptive inventory", "organ_bpw": {}}
    if c.get("keep_experts"):
        # Structural candidates are billed by the omission-aware ledger: omitted expert tensors
        # contribute ZERO payload bits, the codebook amortizes over SURVIVORS only (a real cost of
        # omission, charged not hidden), and the survivor bitmap is a declared runtime table.
        led = SP.ledger(inv, int(c["keep_experts"]), c["gate_up"], c["down"], _routing_doc())
        return {"whole_model_bpw": led["complete_bpw"],
                "complete_bpw_exact": led["complete_bpw_exact"],
                "legal_under_one_bit_ceiling": led["legal_under_one_bit_ceiling"],
                "complete_bytes": led["complete_bytes"], "components": led["components"],
                "coded_expert_tensors": led["coded_expert_tensors"],
                "omitted_expert_tensors": led["omitted_expert_tensors"],
                "scope": "complete artifact, omission-aware", "organ_bpw": {}}
    kw: dict[str, Any] = {}
    if c.get("cold_frac"):
        kw = {"cold_frac": float(c["cold_frac"]),
              "cold_gate": _acct_spec(c["cold_gate_up"]), "cold_down": _acct_spec(c["cold_down"])}
    acct = SHB.whole_model_bpw(inv, _acct_spec(c["gate_up"]), _acct_spec(c["down"]), **kw)
    extra = _strata_extra_bits(inv, cand)
    total_bits = acct["total_artifact_bits"] + extra
    return {
        "whole_model_bpw": round(total_bits / inv.grand_params, 9),
        "strata_extra_bits": extra,
        "total_artifact_bytes": math.ceil(total_bits / 8),
        "organ_bpw": {o: r["realized_bpw"] for o, r in acct["allocation"].items()},
        "expert_organ_bpw": {
            "gate": acct["allocation"][A.ORGAN_EXP_GATE]["realized_bpw"],
            "up": acct["allocation"][A.ORGAN_EXP_UP]["realized_bpw"],
            "down": acct["allocation"][A.ORGAN_EXP_DOWN]["realized_bpw"]},
        "routing_partition_source": _PARTITION_SOURCE["source"],
        "scope": "whole model: every tensor class billed, codebooks amortized over "
                 f"{DEPLOY_CLUSTER} experts",
    }


# ── metrics ───────────────────────────────────────────────────────────────────────────────────
def _logsoftmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def _self_nll(logits, ids):
    ls = _logsoftmax(logits.astype(np.float64))
    nlls = [-ls[i, ids[i + 1]] for i in range(len(ids) - 1)]
    m = float(np.mean(nlls)) if nlls else float("nan")
    return {"nll": round(m, 5), "perplexity": round(float(np.exp(m)), 4), "n_positions": len(nlls)}


def _divergence(orig, packed):
    o, p = orig.astype(np.float64), packed.astype(np.float64)
    lo, lp = _logsoftmax(o), _logsoftmax(p)
    po, pp = np.exp(lo), np.exp(lp)
    sym_kl = 0.5 * ((po * (lo - lp)).sum(-1) + (pp * (lp - lo)).sum(-1))
    cos = (o * p).sum(-1) / (np.linalg.norm(o, axis=-1) * np.linalg.norm(p, axis=-1) + 1e-12)
    k = 5
    to, tp = np.argsort(-o, -1)[:, :k], np.argsort(-p, -1)[:, :k]
    overlap = [len(set(to[i]) & set(tp[i])) / k for i in range(o.shape[0])]
    return {"mean_sym_kl": round(float(sym_kl.mean()), 5),
            "mean_logit_cosine": round(float(cos.mean()), 5),
            "mean_top5_overlap": round(float(np.mean(overlap)), 4),
            "next_token_argmax_agreement": round(float(np.mean(o.argmax(-1) == p.argmax(-1))), 4)}


def _verdict(div: dict[str, float]) -> tuple[bool, str]:
    ok = (div["mean_sym_kl"] <= PROMOTE_KL
          and div["next_token_argmax_agreement"] >= PROMOTE_ARGMAX_AGREE)
    return ok, ("PASS" if ok else ("degraded" if div["mean_sym_kl"] < 2.0 else "collapse"))


# ── durability ────────────────────────────────────────────────────────────────────────────────
def _other_heavy_lease_live() -> str | None:
    for rel in OTHER_LEASE_GLOBS:
        pth = ROOT / rel
        if pth.exists():
            try:
                h = json.loads(pth.read_text())
                if _pid_alive(int(h.get("pid", -1))):
                    return f"{rel} pid {h['pid']}"
            except Exception:
                pass
    return None


def _acquire_lease() -> None:
    LEASES.mkdir(parents=True, exist_ok=True)
    other = _other_heavy_lease_live()
    if other:
        raise SystemExit(f"another heavy lease is live ({other}); one-heavy-lease law - refusing to start")
    if LEASE_PATH.exists():
        try:
            held = json.loads(LEASE_PATH.read_text())
            if _pid_alive(int(held.get("pid", -1))):
                raise SystemExit(f"qwen_gravity lease held by live pid {held['pid']}")
        except (json.JSONDecodeError, ValueError):
            pass
    _atomic(LEASE_PATH, {"acquired_at": _now(), "owner": LABEL, "pid": os.getpid(),
                         "schema": "hawking.successor.watchdog_lease.v1"})


def _beat(phase: str, layer: int, n_layers: int, done: int, total: int, cache=None) -> None:
    import shutil
    payload = {"beat_at": _now(), "label": LABEL, "campaign": "qwen_gravity", "pid": os.getpid(),
               "phase": phase, "layer": layer, "layers_total": n_layers,
               "rows_done": done, "rows_total": total,
               "free_disk_gb": round(shutil.disk_usage(str(ROOT)).free / 1e9, 1),
               "available_ram_gb": round(bc.available_ram_bytes() / 2**30, 1),
               "schema": "hawking.successor.watchdog.v1", "status": "RUNNING"}
    if cache is not None:
        payload["expert_cache"] = cache.stats()
    _atomic(HB_PATH, payload)


# ── rows ──────────────────────────────────────────────────────────────────────────────────────
def _rows(candidates: list[str] | None = None) -> list[dict[str, Any]]:
    """Decisive-first: the parent block, then each rung across the holdout in ladder order."""
    order = [c for c in LADDER_ORDER if candidates is None or c in candidates]
    rows = []
    for cand in order:
        for h in HOLDOUT:
            rows.append({"row_id": f"{h['id']}__{cand}", "prompt": h, "candidate": cand,
                         "kind": LADDER[cand]["kind"]})
    return rows


def _parent_forward():
    """Isolated so tests can monkeypatch a source-absent stub without touching real weights."""
    return Q.from_source()


def _load_parent_logits() -> dict[str, np.ndarray]:
    """FIX 2: parent logits are a durable artifact (53 MB for the whole holdout). This is called
    BEFORE any checkpoint-skip decision, so a restart never recomputes a parent forward."""
    out: dict[str, np.ndarray] = {}
    for h in HOLDOUT:
        p = PARENT_LOGITS / f"{h['id']}.npy"
        if p.is_file():
            try:
                out[h["id"]] = np.load(p)
            except Exception:
                pass
    return out


def _save_parent_logits(pid: str, logits: np.ndarray) -> None:
    PARENT_LOGITS.mkdir(parents=True, exist_ok=True)
    p = PARENT_LOGITS / f"{pid}.npy"
    tmp = p.with_suffix(".npy.tmp")
    with open(tmp, "wb") as fh:                       # file object: np.save does not append .npy
        np.save(fh, np.asarray(logits, np.float32))
    os.replace(tmp, p)


# ── layer-scoped reader cache (FIX 3: never rebuild the 118 mmaps) ────────────────────────────
class _LayerCacheReader:
    """Wraps the persistent SafetensorsIndexReader and memoizes tensors for the CURRENT layer only.
    Attention/router/norm weights are then read + bf16-converted once per layer instead of once per
    (variant, prompt). drop() clears the layer scope; the underlying mmaps are never touched."""

    def __init__(self, inner):
        self.inner = inner
        self._c: dict[str, np.ndarray] = {}

    def bf16(self, name: str) -> np.ndarray:
        v = self._c.get(name)
        if v is None:
            v = self.inner.bf16(name)
            self._c[name] = v
        return v

    def bf16_rows(self, name: str, rows: list[int]) -> np.ndarray:
        return self.inner.bf16_rows(name, rows)

    def has(self, name: str) -> bool:
        return self.inner.has(name)

    def source_present(self) -> bool:
        return self.inner.source_present()

    @property
    def source_dir(self):
        return self.inner.source_dir

    def drop(self) -> None:
        self._c.clear()

    def close(self) -> None:
        self.inner.close()


# ── the lockstep pass (FIX 1: expert OUTER, candidate INNER) ──────────────────────────────────
def _pass_signature(variants: list[str], pids: dict[str, list[str]], max_layers) -> str:
    return _sha({"variants": variants, "pids": pids, "max_layers": max_layers,
                 "ladder": _ladder_sha()})


def _save_pass_state(states: dict[tuple[str, str], np.ndarray], layer: int, sig: str) -> None:
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    tmp = PASS_STATE_NPZ.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:                       # file object: savez does not append .npz
        np.savez(fh, **{f"{v}||{p}": a for (v, p), a in states.items()})
    os.replace(tmp, PASS_STATE_NPZ)
    _atomic(PASS_STATE_JSON, {"next_layer": layer, "signature": sig, "saved_at": _now()})


def _load_pass_state(sig: str):
    if not (PASS_STATE_NPZ.is_file() and PASS_STATE_JSON.is_file()):
        return None, 0
    try:
        meta = json.loads(PASS_STATE_JSON.read_text())
        if meta.get("signature") != sig:
            return None, 0
        z = np.load(PASS_STATE_NPZ)
        states = {tuple(k.split("||")): np.asarray(z[k], np.float32) for k in z.files}
        return states, int(meta["next_layer"])
    except Exception:
        return None, 0


def _read_expert(reader, L: int, e: int) -> dict[str, np.ndarray]:
    return {"gate": reader.bf16(f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight"),
            "up": reader.bf16(f"model.layers.{L}.mlp.experts.{e}.up_proj.weight"),
            "down": reader.bf16(f"model.layers.{L}.mlp.experts.{e}.down_proj.weight")}


# bounded_cache._entry_bytes only measures arrays and array tuples/lists, so experts are cached as
# an organ-ordered tuple - a dict would be billed as 0 bytes and defeat HAWKING_CACHE_MAX_GB.
def _cache_put(cache, key, ex: dict[str, np.ndarray]) -> None:
    cache.put(key, tuple(ex[o] for o in ORGANS))


def _cache_get(cache, key) -> dict[str, np.ndarray] | None:
    v = cache.get(key)
    return None if v is None else dict(zip(ORGANS, v))


def lockstep_logits(fwd, variants: list[str], plan: dict[str, list[str]],
                    ids_by_pid: dict[str, list[int]], *, max_layers: int | None = None,
                    fit_experts: int = 4, rows_total: int = 0,
                    resume: bool = True) -> dict[tuple[str, str], np.ndarray]:
    """Advance every (variant, prompt) state through the blocks in lockstep.

    plan maps variant -> the prompt ids that variant must cover. Per layer: read every weight once,
    take the UNION of everyone's routed experts, and for each expert read+convert it ONCE, pack it
    for every candidate from that one resident copy, and apply it to every routed position. This is
    the whole point of the controller: the ~105 distinct experts per layer are streamed once per
    campaign instead of once per candidate.
    """
    g = fwd.g
    rd = fwd.reader if isinstance(fwd.reader, _LayerCacheReader) else _LayerCacheReader(fwd.reader)
    fwd.reader = rd
    cache = bc.PressureAwareCache("qwen_gravity_experts", disk_path=str(rd.source_dir), verbose=False)
    n_layers = g.n_layers if max_layers is None else min(int(max_layers), g.n_layers)
    sig = _pass_signature(variants, plan, max_layers)

    states, start = (_load_pass_state(sig) if resume else (None, 0))
    if states is None:
        states = {(v, pid): rd.bf16_rows("model.embed_tokens.weight", list(ids_by_pid[pid]))
                  for v in variants for pid in plan[v]}
        start = 0
    packed_vars = [v for v in variants if LADDER[v]["kind"] == "packed"]

    for L in range(start, n_layers):
        t_layer = time.time()
        rd.drop()
        _beat("attention", L, n_layers, 0, rows_total, cache)
        pln = f"model.layers.{L}.post_attention_layernorm.weight"
        hs: dict[tuple[str, str], np.ndarray] = {}
        for key in states:
            x = states[key] + fwd._attention(L, states[key])
            states[key] = x
            hs[key] = Q.rmsnorm(x, rd.bf16(pln), g.eps)

        # routing: expert -> variant -> [(key, position, gate_weight)]
        gate_w = rd.bf16(f"model.layers.{L}.mlp.gate.weight")
        need: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        outs = {key: np.zeros_like(h) for key, h in hs.items()}
        for key, h in hs.items():
            lg = h @ gate_w.T
            keep = survivor_set(key[0], L, g.n_experts)
            for p in range(h.shape[0]):
                idx, wts = _route_masked(lg[p], keep, g.top_k, g.norm_topk_prob)
                for e, gwt in zip(idx, wts):
                    need[int(e)][key[0]].append((key, p, float(gwt)))
        order = sorted(need)

        # fit the per-layer shared codebooks from a small resident sample of the cluster
        _beat("grammar_fit", L, n_layers, 0, rows_total, cache)
        # Fit sample per variant: a structural candidate must never fit its codebook on an expert
        # it omits. Keyed by the variant's own routed set, falling back to the shared order.
        fit_by_var: dict[str, list[int]] = {}
        for v in packed_vars:
            routed = sorted(e for e in order if v in need[e])
            fit_by_var[v] = (routed or order)[:max(1, int(fit_experts))]
        fit_ids = sorted({e for ids in fit_by_var.values() for e in ids} or set(
            order[:max(1, int(fit_experts))]))
        for e in fit_ids:
            if _cache_get(cache, (L, e)) is None:
                _cache_put(cache, (L, e), _read_expert(rd, L, e))
        def _mats(ids: list[int]) -> dict[str, list]:
            m = {"gate_up": [], "down": []}
            for e in ids:
                ex = _cache_get(cache, (L, e))
                m["gate_up"] += [ex["gate"], ex["up"]]
                m["down"].append(ex["down"])
            return m
        books: dict[tuple[str, str, bool], Any] = {}
        for v in packed_vars:
            fit_mats = _mats(fit_by_var[v])
            for grp in ("gate_up", "down"):
                for cold in (False, True):
                    spec = (LADDER[v].get(f"cold_{grp}") if cold else LADDER[v].get(grp))
                    if spec is None or spec.get("family") not in ("shared_grammar",
                                                                  "function_aware"):  # noqa: E501
                        continue
                    if cold and not LADDER[v].get("cold_frac"):
                        continue
                    if spec["family"] == "function_aware":
                        mats = [np.ascontiguousarray(m, np.float32) for m in fit_mats[grp]]
                        imp = _gamma_importance(rd, L) if spec.get("gamma_weighted") else None
                        base = FAC.fit(mats, dim=int(spec["dim"]), k=int(spec["k"]),
                                       stages=int(spec.get("stages", 1)), seed=L * 131 + 7,
                                       iters=4, importance=imp)
                        doc = spec.get("doctor")
                        books[(v, grp, cold)] = base if not doc else (
                            base, FAC.fit_doctor(
                                mats, base, dim=int(spec["dim"]), k=int(spec["k"]),
                                stages=int(spec.get("stages", 1)), doctor_dim=int(doc["dim"]),
                                doctor_k=int(doc["k"]), doctor_stages=int(doc.get("stages", 1)),
                                protect_frac=float(doc["protect_frac"]), seed=L * 131 + 7,
                                iters=4))
                    else:
                        books[(v, grp, cold)] = _fit_grammar(spec, fit_mats[grp], seed=L * 131 + 7)

        # expert OUTER, candidate INNER
        _beat("experts", L, n_layers, 0, rows_total, cache)
        for e in order:
            ex = _cache_get(cache, (L, e))
            if ex is None:
                ex = _read_expert(rd, L, e)
                _cache_put(cache, (L, e), ex)
            for v, hits in need[e].items():
                if LADDER[v]["kind"] == "parent":
                    w = ex
                else:
                    specs, cold = _specs_for(v, L, e, g.n_experts)
                    w = {}
                    for organ in ORGANS:
                        grp = _GROUP[organ]
                        spec = specs[grp]
                        if spec.get("gamma_weighted"):
                            spec = dict(spec, _importance=_gamma_importance(rd, L))
                        bk = books.get((v, grp, cold))
                        w[organ] = _pack_organ(spec, bk, ex[organ], seed=L * 1000 + e)
                for key, p, gwt in hits:
                    a = Q.swiglu(w["gate"] @ hs[key][p], w["up"] @ hs[key][p])
                    outs[key][p] += gwt * (w["down"] @ a)
                if LADDER[v]["kind"] == "packed":
                    del w
            _mps_gc()

        for key in states:
            states[key] = states[key] + outs[key]
        del hs, outs, need, books
        if resume:                       # a bounded PROBE must not clobber a real run's resume point
            _save_pass_state(states, L + 1, sig)
        print(f"  layer {L:2d}/{n_layers - 1}  {len(order)} experts  "
              f"{time.time() - t_layer:6.1f}s  cache={cache.stats()['cache_gb']}GB", flush=True)

    # final norm + untied lm_head, read once for every state
    rd.drop()
    _beat("lm_head", n_layers, n_layers, 0, rows_total, cache)
    lm = rd.bf16("model.embed_tokens.weight" if g.tie else "lm_head.weight")
    nw = rd.bf16("model.norm.weight")
    return {key: (Q.rmsnorm(x, nw, g.eps) @ lm.T).astype(np.float32) for key, x in states.items()}


# ── run ───────────────────────────────────────────────────────────────────────────────────────
def _seal_waiting_source(candidates=None) -> int:
    """Source-absent no-op: seal WAITING_SOURCE and exit 0. Never a crash, never a lease."""
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "hawking.qwen_gravity.state.v1", "status": "WAITING_SOURCE",
               "generated_at": _now(), "final": False, "rows_done": 0,
               "rows_total": len(_rows(candidates)),
               "gate": "qwen3_235b_gravity_real_forward",
               "reason": "Qwen3-235B source shards not fully present; no lease taken, no forward run.",
               "source_dir": str(ROOT / Q.DEFAULT_SOURCE),
               "ladder": {c: LADDER[c].get("note", "") for c in LADDER_ORDER},
               "ladder_order": LADDER_ORDER,
               "promote_thresholds": {"mean_sym_kl_max": PROMOTE_KL,
                                      "argmax_agreement_min": PROMOTE_ARGMAX_AGREE},
               "honesty": "UNTESTED-PENDING-SOURCE; run() is a safe no-op until the weights are staged"}
    _atomic(STATE_PATH, payload)
    _atomic(WAITING_RECEIPT, payload)
    print(json.dumps({"status": "WAITING_SOURCE", "state": str(STATE_PATH)}, indent=2))
    return 0


def run(max_rows: int | None = None, candidates: list[str] | None = None,
        max_layers: int | None = None, fit_experts: int = 4) -> int:
    fwd = _parent_forward()
    if not fwd.source_present():
        return _seal_waiting_source(candidates)
    for d in (LEASES, HEARTBEAT, CHECKPOINTS, CONTROLLER, PARENT_LOGITS):
        d.mkdir(parents=True, exist_ok=True)
    _acquire_lease()
    try:
        return _run_inner(fwd, max_rows, candidates, max_layers, fit_experts)
    finally:
        if LEASE_PATH.exists():
            LEASE_PATH.unlink()


def _run_inner(fwd, max_rows, candidates, max_layers, fit_experts) -> int:
    t0 = time.time()
    tk = _tokenizer()
    ids_by_pid = {h["id"]: tk.encode(h["text"]).ids for h in HOLDOUT}
    prompt_by_pid = {h["id"]: h for h in HOLDOUT}
    inv = A.build_inventory(A.load_config(), A.load_index())
    partial = max_layers is not None and int(max_layers) < fwd.g.n_layers

    # FIX 2: parent logits load FIRST, before any checkpoint-skip decision. (A bounded PROBE must
    # NOT reuse full-stack parent logits: it recomputes the parent through the same truncated stack.)
    parent_logits = {} if partial else _load_parent_logits()

    rows = _rows(candidates)
    pending = [r for r in rows if not (CHECKPOINTS / f"{r['row_id']}.json").exists()]
    if max_rows is not None:
        pending = pending[:int(max_rows)]
    if not pending:
        _write_state(rows, t0, final=True)
        print(json.dumps({"status": "SEALED", "rows_total": len(rows)}, indent=2))
        return 0

    # variant -> prompts it must cover. The parent joins the pass ONLY for prompts whose logits are
    # not already sealed on disk.
    plan: dict[str, list[str]] = {}
    for r in pending:
        if r["kind"] == "parent":
            continue
        plan.setdefault(r["candidate"], [])
        if r["prompt"]["id"] not in plan[r["candidate"]]:
            plan[r["candidate"]].append(r["prompt"]["id"])
    needed_pids = sorted({p for ps in plan.values() for p in ps}
                         | {r["prompt"]["id"] for r in pending if r["kind"] == "parent"})
    missing_parent = [p for p in needed_pids if p not in parent_logits]
    if missing_parent:
        plan["R0_parent"] = missing_parent
    variants = [v for v in LADDER_ORDER if v in plan]
    if not variants:
        # everything pending is a parent row whose logits are already on disk: seal and finish.
        logits = {}
    else:
        print(f"lockstep pass: variants={variants} prompts={ {v: len(p) for v, p in plan.items()} } "
              f"parent_logits_on_disk={sorted(parent_logits)}", flush=True)
        logits = lockstep_logits(fwd, variants, plan, ids_by_pid, max_layers=max_layers,
                                 fit_experts=fit_experts, rows_total=len(rows),
                                 resume=not partial)

    for v in variants:
        if LADDER[v]["kind"] != "parent":
            continue
        for pid in plan[v]:
            parent_logits[pid] = logits[(v, pid)]
            if not partial:
                _save_parent_logits(pid, logits[(v, pid)])

    bpw_cache: dict[str, dict[str, Any]] = {}
    sealed = 0
    for r in pending:
        cand, pid = r["candidate"], r["prompt"]["id"]
        lg = logits.get((cand, pid))
        if lg is None and r["kind"] == "parent":
            lg = parent_logits.get(pid)
        if lg is None:
            continue
        ids = ids_by_pid[pid]
        rec: dict[str, Any] = {"row_id": r["row_id"], "variant": cand, "kind": r["kind"],
                               "prompt_id": pid, "domain": prompt_by_pid[pid]["domain"],
                               "n_tokens": len(ids), "quality": _self_nll(lg, ids),
                               "logits_finite": bool(np.isfinite(lg).all()),
                               "sealed_at": _now(), "ladder_sha": _ladder_sha(),
                               "spec": {k: v for k, v in LADDER[cand].items() if k != "note"},
                               "note": LADDER[cand].get("note", "")}
        if cand not in bpw_cache:
            bpw_cache[cand] = _ladder_bpw(inv, cand)
        rec["bpw"] = bpw_cache[cand]
        if r["kind"] == "packed":
            parent = parent_logits.get(pid)
            if parent is None:
                continue
            div = _divergence(parent, lg)
            ok, verdict = _verdict(div)
            rec["divergence_vs_parent"] = div
            diag = bool(LADDER[cand].get("diagnostic_only"))
            rec["capability_pass"] = bool(ok) and not diag
            rec["verdict"] = ("CAUSAL_CONTROL_not_a_candidate" if diag else verdict)
            if diag:
                rec["diagnostic_only"] = True
                rec["promotion_blocked_reason"] = (
                    "causal control for the S1 failure decomposition; its byte cost is not a legal "
                    "artifact and it may never be promoted regardless of measured quality")
        else:
            rec["verdict"] = "parent_reference"
        rec["honesty"] = ("REAL parent-vs-packed Qwen3-MoE forward with expert_hook substitution; "
                          "BPW is whole-model from the exact byte ledger, never expert-only; "
                          "weight-space error is NOT a pass criterion")
        if partial:
            rec["verdict"] = "PROBE_partial_stack"
            rec["capability_pass"] = False
            rec["partial_layers"] = int(max_layers)
        rec["sha256"] = _sha({k: v for k, v in rec.items() if k != "sha256"})
        if partial:
            _atomic(CHECKPOINTS / f"PROBE__{r['row_id']}.json", rec)   # never a sealed science row
        else:
            _atomic(CHECKPOINTS / f"{r['row_id']}.json", rec)
        sealed += 1
        extra = ""
        if "divergence_vs_parent" in rec:
            d_ = rec["divergence_vs_parent"]
            extra = (f"  symKL={d_['mean_sym_kl']} agree={d_['next_token_argmax_agreement']}"
                     f" bpw={rec['bpw']['whole_model_bpw']} -> {rec['verdict']}")
        print(f"[seal {sealed}/{len(pending)}] {r['row_id']}  ppl={rec['quality']['perplexity']}{extra}",
              flush=True)

    if not partial:
        for p in (PASS_STATE_NPZ, PASS_STATE_JSON):
            if p.exists():
                p.unlink()
    _write_state(rows, t0, final=not partial)
    return 0


def _write_state(rows, t0, final=False) -> None:
    seals = sorted(p for p in CHECKPOINTS.glob("*.json") if not p.name.startswith("PROBE__"))
    res = [json.loads(p.read_text()) for p in seals]
    graded = [r for r in res if r.get("divergence_vs_parent")]
    best = sorted(graded, key=lambda r: r["divergence_vs_parent"]["mean_sym_kl"])[:5]
    done, total = len(res), len(rows)
    elapsed = time.time() - t0
    _atomic(STATE_PATH, {
        "schema": "hawking.qwen_gravity.state.v1",
        "status": "SEALED" if final and done >= total else "RUNNING",
        "generated_at": _now(), "final": bool(final and done >= total),
        "gate": "qwen3_235b_gravity_real_forward",
        "rows_done": done, "rows_total": total,
        "ladder": {c: LADDER[c].get("note", "") for c in LADDER_ORDER},
        "ladder_order": LADDER_ORDER, "ladder_sha": _ladder_sha(),
        "promote_thresholds": {"mean_sym_kl_max": PROMOTE_KL,
                               "argmax_agreement_min": PROMOTE_ARGMAX_AGREE},
        "routing_partition_source": _PARTITION_SOURCE["source"],
        "parent_logits_persisted": sorted(p.stem for p in PARENT_LOGITS.glob("*.npy")),
        "least_divergent": [{"row_id": r["row_id"], "variant": r["variant"],
                             "mean_sym_kl": r["divergence_vs_parent"]["mean_sym_kl"],
                             "argmax_agreement": r["divergence_vs_parent"]["next_token_argmax_agreement"],
                             "whole_model_bpw": (r.get("bpw") or {}).get("whole_model_bpw"),
                             "verdict": r.get("verdict")} for r in best],
        "capability_passes": [r["row_id"] for r in res if r.get("capability_pass")],
        "eta_seconds_remaining": round(elapsed / max(1, done) * (total - done), 0) if not final else 0,
        "honest_note": "every non-parent row is a REAL parent-vs-packed forward. Weight-space "
                       "reconstruction error is never a pass criterion (the sealed GPT-OSS-120B "
                       "lesson). BPW is whole-model from the exact byte ledger.",
    })


# ── CLI ───────────────────────────────────────────────────────────────────────────────────────
def detach(extra: list[str] | None = None) -> int:
    CONTROLLER.mkdir(parents=True, exist_ok=True)
    log = CONTROLLER / "detached.log"
    argv = ["caffeinate", "-i", "-s", PY, os.path.abspath(__file__), "run"] + list(extra or [])
    with open(log, "ab") as lh:
        proc = subprocess.Popen(argv, stdout=lh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(ROOT), start_new_session=True)
    print(json.dumps({"detached": True, "pid": proc.pid, "log": str(log),
                      "argv": argv}, indent=2))
    return 0


def status() -> int:
    st = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {"status": "no state"}
    hb = json.loads(HB_PATH.read_text()) if HB_PATH.exists() else {}
    lease = json.loads(LEASE_PATH.read_text()) if LEASE_PATH.exists() else {}
    ps = json.loads(PASS_STATE_JSON.read_text()) if PASS_STATE_JSON.exists() else {}
    print(json.dumps({"lease": lease,
                      "lease_pid_alive": _pid_alive(int(lease["pid"])) if lease.get("pid") else False,
                      "heartbeat": hb, "resume_point": ps, "state": st}, indent=2, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "detach", "status"])
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap how many unsealed rows this run targets (decisive-first order)")
    ap.add_argument("--candidates", default=None,
                    help="comma-separated ladder subset, e.g. R0_parent,R1_c1_corrected")
    ap.add_argument("--max-layers", type=int, default=None,
                    help="bounded PROBE over the first N blocks; rows are written PROBE__ and never "
                         "sealed as science")
    ap.add_argument("--fit-experts", type=int, default=4,
                    help="experts sampled per layer to fit the shared codebooks")
    a = ap.parse_args(argv)
    cands = [c.strip() for c in a.candidates.split(",")] if a.candidates else None
    if cands:
        unknown = [c for c in cands if c not in LADDER]
        if unknown:
            raise SystemExit(f"unknown candidates: {unknown}; known: {LADDER_ORDER}")
    if a.cmd == "run":
        return run(a.max_rows, cands, a.max_layers, a.fit_experts)
    if a.cmd == "detach":
        extra = []
        if a.max_rows is not None:
            extra += ["--max-rows", str(a.max_rows)]
        if a.candidates:
            extra += ["--candidates", a.candidates]
        if a.max_layers is not None:
            extra += ["--max-layers", str(a.max_layers)]
        if a.fit_experts != 4:
            extra += ["--fit-experts", str(a.fit_experts)]
        return detach(extra)
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
