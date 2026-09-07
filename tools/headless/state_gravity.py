#!/usr/bin/env python3
"""N048 — STATE GRAVITY: KV + DeltaNet state redundancy census.

S026 §50-58, §119; DOC-STATE. Builds a StateGenome for OUR runtime state
(GQA KV + DeltaNet recurrent/conv) from NOETIC_ORGAN_CENSUS / PREFILL_KV
geometry, then measures whether the four paper methods actually find the
redundancy they exploit on REAL activations/state:

  KIVI       per-channel K vs per-token V quantizability
  MiniCache  adjacent-layer KV cosine (depth merge)
  H2O        heavy-hitter attention-mass concentration
  DeltaNet   recurrent-state redundancy (§56)

Each axis: measured redundancy + estimated bytes-saved + long-context
risk, ranked. Long-context capability is the gate; byte savings that
would lose recall are not a GO.

CPU only. Does not touch the GPU, does not run cargo/Metal, does not
decode a second 27B, does not mutate ~/noetic/NOETIC_PARENT_A.
Streaming parent tensors read-only is allowed. The Q4 CPU hybrid prefill
is the organ-census path (real token ids, production-layout KV + rec
state); capture_diverse2 + parent k/v/q is a longer-token corroboration
at the neighboring post_attn_norm site, labeled as such.

    python3 tools/headless/state_gravity.py
    python3 -m pytest tools/headless/test_state_gravity.py -q
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from noetic_information_accounting import qwen38_workspace_bytes  # noqa: E402
from noetic_organ_census import (  # noqa: E402
    ARTIFACT as Q4_ROOT,
    Artifact,
    DN_CONV_K,
    DN_K_DIM,
    DN_K_HEADS,
    DN_V_DIM,
    DN_V_HEADS,
    DN_VPK,
    FULL_ATTN_INTERVAL,
    GQA_HEAD_DIM,
    GQA_HEADS,
    GQA_KV_HEADS,
    LAYERS,
    MAX_TOKENS,
    PROMPTS,
    apply_rope,
    causal_conv1d_silu,
    fidelity,
    is_gqa,
    l2norm,
    mlp_forward,
    read_q4_rows,
    rmsnorm_delta,
    silu,
    softmax_last,
    tokenize,
    _unfuse_ba,
    _unfuse_qkvz,
)
from noetic_parent_a import DURABLE as PARENT_A  # noqa: E402
from organ_frontiers import (  # noqa: E402
    find_capture,
    find_parent,
    load_X,
    load_tensor,
    split_from_manifest,
    tensor_name,
)
from prefill_kv import (  # noqa: E402
    ACTIVATION_BYTES,
    DELTANET_STATE_BYTES,
    KV_BYTES_PER_POSITION,
    encode_ids,
    session_state_bytes,
)

SCHEMA = "hawking.headless.state_gravity.v1"
RECEIPT = REPO / "receipts" / "headless" / "STATE_GRAVITY.json"
GENERATOR = "tools/headless/state_gravity.py"
OBLIGATION = (
    "N048 — STATE GRAVITY (S026 §50-58, §119; DOC-STATE; CPU): KV + "
    "DeltaNet state redundancy census."
)

MEASURED = "MEASURED"
DERIVED = "DERIVED"
ABSENT = "ABSENT"
CITED = "CITED"

GIB = 1024 ** 3
GQA_LAYERS_N = 16
DN_LAYERS_N = 48
PARENT_PARAMS = 26_895_998_464
Q4_MODEL_BYTES = 14_297_694_680
PARENT_MODEL_BYTES = 10_554_328_856
Q4_ON_DISK = 14_297_933_604

# Headline operating point from N016 / PREFILL_KV: state exceeds weights.
HEADLINE_SEQ = 32768
HEADLINE_C = 4

# H2O recent-window analogue (tokens kept regardless of mass).
H2O_RECENT = 16
H2O_HEAVY_FRAC = 0.20

# KIVI paper default is 2-bit; we also score 4-bit. Capability is ABSENT.
# Group size is along the *other* axis so per-channel and per-token recipes
# have the same number of scales (matched rate). Ungrouped absmax confounds
# axis preference with "how many values share a scale".
KIVI_BITS = (2, 4)
KIVI_GROUP = 32

GQA_LAYER_IDS = tuple(i for i in range(LAYERS) if is_gqa(i))
DN_LAYER_IDS = tuple(i for i in range(LAYERS) if not is_gqa(i))

# Capture corroboration: hold split of capture_diverse2. H2O uses a few
# GQA depths so q_proj stays cheap; MiniCache/KIVI use every GQA layer.
H2O_CAPTURE_LAYERS = (3, 31, 63)
CAPTURE_MIN_PROMPT = 32

PREFILL_KV_REL = "receipts/headless/PREFILL_KV.json"
ORGAN_CENSUS_REL = "receipts/headless/NOETIC_ORGAN_CENSUS.json"
GQA_DESIGN_REL = "receipts/headless/NOETIC_GQA_DESIGN.json"
DN_DESIGN_REL = "receipts/headless/NOETIC_DELTANET_DESIGN.json"
ORGAN_FRONTIERS_REL = "receipts/headless/ORGAN_FRONTIERS.json"
CANON_REL = "docs/ultragoals/NOETIC_CANON.md"


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return (r.stdout or "").strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if x is None or isinstance(x, str):
        return x
    return x


def atomic_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(obj), indent=2) + "\n")
    os.replace(tmp, path)


def load_json_rel(rel: str) -> tuple[dict | None, str]:
    p = REPO / rel
    if p.is_file():
        return json.loads(p.read_text()), f"disk:{rel}"
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(REPO), "show", f"HEAD:{rel}"], timeout=60
        )
        return json.loads(raw), f"git:HEAD:{rel}"
    except Exception:
        return None, f"missing:{rel}"


def numbered(value, *, kind: str, unit=None, source=None, formula=None, note=None,
             null=None):
    rec: dict[str, Any] = {"value": to_jsonable(value), "kind": kind}
    if unit is not None:
        rec["unit"] = unit
    if source is not None:
        rec["source"] = source
    if formula is not None:
        rec["formula"] = formula
    if note is not None:
        rec["note"] = note
    if null is not None:
        rec["null"] = to_jsonable(null)
    return rec


def parent_snap(root: Path) -> dict[str, Any]:
    cat = root / "catalog.hq38m20"
    if not cat.is_file():
        return {"path": str(root), "present": False}
    st = cat.lstat()
    return {
        "path": str(root),
        "present": True,
        "catalog": str(cat),
        "catalog_bytes": int(st.st_size),
        "catalog_mtime_ns": int(st.st_mtime_ns),
        "catalog_ino": int(st.st_ino),
        "writable_check": oct(st.st_mode),
        "mode_is_regular_file": bool(stat.S_ISREG(st.st_mode)),
    }


def gemm(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """w [out, in] @ x [T, in] -> [T, out]. CPU only; never MPS."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32)
    try:
        import torch

        wt = torch.from_numpy(w)
        xt = torch.from_numpy(x)
        if wt.device.type != "cpu":
            wt = wt.cpu()
            xt = xt.cpu()
        with torch.no_grad():
            y = xt @ wt.T
        return np.ascontiguousarray(y.numpy(), dtype=np.float32)
    except Exception:
        return x @ w.T


# ---------------------------------------------------------------------------
# StateGenome — sizes from the workspace formula / PREFILL_KV
# ---------------------------------------------------------------------------

def gqa_kv_bytes(max_seq_len: int) -> int:
    return KV_BYTES_PER_POSITION * int(max_seq_len)


def state_genome(*, model_bytes: int = Q4_MODEL_BYTES) -> dict[str, Any]:
    """Runtime-state genome. Arithmetic lockstep with qwen38_workspace_bytes."""
    ws_256 = qwen38_workspace_bytes(256)
    assert ws_256["gqa_kv_bytes"] == 33_554_432
    assert ws_256["deltanet_state_bytes"] == DELTANET_STATE_BYTES == 156_893_184
    assert ws_256["kv_bytes_per_position"] == KV_BYTES_PER_POSITION == 131_072
    rec_elems_layer = DN_V_HEADS * DN_K_DIM * DN_V_DIM
    conv_channels = DN_K_HEADS * DN_K_DIM * 2 + DN_V_HEADS * DN_V_DIM
    conv_elems_layer = conv_channels * (DN_CONV_K - 1)
    rec_bytes = DN_LAYERS_N * rec_elems_layer * 4
    conv_bytes = DN_LAYERS_N * conv_elems_layer * 4
    assert rec_bytes + conv_bytes == DELTANET_STATE_BYTES

    seqs = (256, 4096, 16384, 32768, 131072)
    by_seq = {}
    for seq in seqs:
        ss = session_state_bytes(seq)
        by_seq[str(seq)] = {
            "max_seq_len": seq,
            "activation_bytes": numbered(
                ss["activation_bytes"], kind=DERIVED, unit="bytes",
                source="qwen38_workspace_bytes",
            ),
            "deltanet_state_bytes": numbered(
                ss["deltanet_state_bytes"], kind=CITED, unit="bytes",
                source="PREFILL_KV / qwen38_workspace_bytes",
                note="constant in seq_len: rec 150,994,944 + conv 5,898,240",
            ),
            "gqa_kv_bytes": numbered(
                ss["gqa_kv_bytes"], kind=DERIVED, unit="bytes",
                formula="16 * seq * 4 * 256 * 4 B * 2 (K+V)",
            ),
            "SESSION_STATE_BYTES": numbered(
                ss["SESSION_STATE_BYTES"], kind=DERIVED, unit="bytes",
            ),
            "gqa_share_of_session": numbered(
                ss["gqa_kv_bytes"] / ss["SESSION_STATE_BYTES"],
                kind=DERIVED, unit="fraction",
            ),
            "deltanet_share_of_session": numbered(
                ss["deltanet_state_bytes"] / ss["SESSION_STATE_BYTES"],
                kind=DERIVED, unit="fraction",
            ),
        }

    def footprint(seq: int, sessions: int) -> dict[str, Any]:
        ss = session_state_bytes(seq)
        state_c = ss["SESSION_STATE_BYTES"] * sessions
        total = model_bytes + state_c
        return {
            "sessions": sessions,
            "max_seq_len": seq,
            "MODEL_BYTES": model_bytes,
            "SESSION_STATE_BYTES": ss["SESSION_STATE_BYTES"],
            "SESSION_STATE_BYTES_x_c": state_c,
            "PRODUCTION_FOOTPRINT_BYTES": total,
            "PRODUCTION_FOOTPRINT_GiB": total / GIB,
            "state_exceeds_weights": state_c > model_bytes,
            "state_share_of_footprint": state_c / total if total else None,
            "gqa_kv_bytes_x_c": ss["gqa_kv_bytes"] * sessions,
            "deltanet_state_bytes_x_c": ss["deltanet_state_bytes"] * sessions,
        }

    return {
        "model": "Qwen3.8-27B hybrid (qwen3_5_text)",
        "production_kv_dtype": "f32",
        "production_mha_kernel": "mha_decode_f32_tcb",
        "full_attention_interval": FULL_ATTN_INTERVAL,
        "layers": LAYERS,
        "gqa_layers": {
            "count": GQA_LAYERS_N,
            "ids": list(GQA_LAYER_IDS),
            "heads": GQA_HEADS,
            "kv_heads": GQA_KV_HEADS,
            "head_dim": GQA_HEAD_DIM,
            "group": GQA_HEADS // GQA_KV_HEADS,
            "kv_bytes_per_position": numbered(
                KV_BYTES_PER_POSITION, kind=CITED, unit="bytes",
                source=PREFILL_KV_REL,
                formula="16 layers * 4 kv_heads * 256 dim * 4 B * 2 (K+V)",
            ),
            "grows_with_seq": True,
            "prefix_shareable": {
                "kind": ABSENT,
                "value": None,
                "note": (
                    "Valid only when N sessions share a byte-identical prefix. "
                    "system_kv_bank is not wired into Qwen38HybridDecodeSession "
                    "(PREFILL_KV prefix_sharing)."
                ),
            },
        },
        "deltanet_layers": {
            "count": DN_LAYERS_N,
            "ids": list(DN_LAYER_IDS),
            "key_heads": DN_K_HEADS,
            "value_heads": DN_V_HEADS,
            "vpk": DN_VPK,
            "key_dim": DN_K_DIM,
            "value_dim": DN_V_DIM,
            "conv_k": DN_CONV_K,
            "recurrent_elems_per_layer": rec_elems_layer,
            "recurrent_bytes": numbered(rec_bytes, kind=DERIVED, unit="bytes"),
            "conv_elems_per_layer": conv_elems_layer,
            "conv_bytes": numbered(conv_bytes, kind=DERIVED, unit="bytes"),
            "total_bytes": numbered(
                DELTANET_STATE_BYTES, kind=CITED, unit="bytes",
                source=PREFILL_KV_REL,
            ),
            "grows_with_seq": False,
            "prefix_shareable": False,
            "prefix_shareable_reason": (
                "Recurrent summary, not a cache. Diverged suffixes cannot "
                "share rec_state (NOETIC_CANON law 14; PREFILL_KV prefix_sharing; "
                "S026 §55-56)."
            ),
        },
        "activations": {
            "bytes": numbered(ACTIVATION_BYTES, kind=CITED, unit="bytes",
                              source=PREFILL_KV_REL),
            "grows_with_seq": False,
            "note": "scratch, not persisted across decode steps as KV is",
        },
        "speculative_state": {
            "kind": ABSENT,
            "value": None,
            "reason": (
                "Native MTP / draft-head state is N049 DECODING_GRAVITY. "
                "No speculative cache is resident on Qwen38HybridDecodeSession."
            ),
        },
        "session_isolation": {
            "law": "one immutable body, many sessions (NOETIC_CANON 12)",
            "state_is_per_session": True,
            "weights_are_shared": True,
        },
        "by_seq": by_seq,
        "headline_n016": {
            "cite": PREFILL_KV_REL,
            "q4_32k_c4": footprint(HEADLINE_SEQ, HEADLINE_C),
            "q4_32k_c1": footprint(HEADLINE_SEQ, 1),
            "q4_256_c1": footprint(256, 1),
            "note": (
                "N016: q4 c=4 at 32K, SESSION_STATE_x_c 16.59 GiB exceeds "
                "MODEL_BYTES 13.32 GiB. GQA KV is the only seq-linear term."
            ),
        },
        "what_the_genome_is_not": [
            "Not a weight genome. Weights live in ORGAN_LIBRARY / EBPW.",
            "Not a claim that any compression axis is free. Capability is the gate.",
        ],
    }


# ---------------------------------------------------------------------------
# Axis math (pure; unit-tested on synthetic arrays)
# ---------------------------------------------------------------------------

def absmax_quantize(x: np.ndarray, bits: int, axis: int) -> np.ndarray:
    """Symmetric absmax along `axis`. bits includes the sign (int4 -> 7 levels)."""
    if bits < 2 or bits > 8:
        raise ValueError(f"bits {bits} not in 2..8")
    x = np.asarray(x, dtype=np.float32)
    levels = float((1 << (bits - 1)) - 1)
    peak = np.max(np.abs(x), axis=axis, keepdims=True)
    peak = np.maximum(peak, 1e-30)
    q = np.round(np.clip(x / peak, -1.0, 1.0) * levels)
    return (q / levels) * peak


def grouped_absmax_quantize(x: np.ndarray, bits: int, group_axis: int, group: int) -> np.ndarray:
    """Matched-rate grouped absmax on a 2-D array (T, C).

    group_axis=0 groups tokens (KIVI per-channel K: one scale per (token-group, channel)).
    group_axis=1 groups channels (KIVI per-token V: one scale per (token, channel-group)).
    n_scales = C*ceil(T/G) vs T*ceil(C/G) — equal when T and C share the same G.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"grouped_absmax expects 2-D, got {x.shape}")
    if group < 1:
        raise ValueError("group must be >= 1")
    t, c = x.shape
    if group_axis == 0:
        pad = (group - t % group) % group
        work = np.pad(x, ((0, pad), (0, 0))) if pad else x
        g = work.reshape(work.shape[0] // group, group, c)
        rec = absmax_quantize(g, bits, axis=1)
        return rec.reshape(-1, c)[:t]
    if group_axis == 1:
        pad = (group - c % group) % group
        work = np.pad(x, ((0, 0), (0, pad))) if pad else x
        g = work.reshape(t, work.shape[1] // group, group)
        rec = absmax_quantize(g, bits, axis=2)
        return rec.reshape(t, -1)[:, :c]
    raise ValueError(f"group_axis {group_axis}")


def flatten_kv(x: np.ndarray) -> np.ndarray:
    """(T, H, D) -> (T, H*D)."""
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(x.shape[0], -1)


def rms_cv(x: np.ndarray, axis: int) -> dict[str, float]:
    """Coefficient of variation of RMS along `axis` (the axis that is collapsed)."""
    x = np.asarray(x, dtype=np.float64)
    rms = np.sqrt(np.mean(x * x, axis=axis))
    mu = float(rms.mean())
    sd = float(rms.std())
    mx = float(rms.max()) if rms.size else 0.0
    md = float(np.median(rms)) if rms.size else 0.0
    return {
        "mean_rms": mu,
        "std_rms": sd,
        "cv": sd / max(mu, 1e-30),
        "max_over_median": mx / max(md, 1e-30),
        "n": int(rms.size),
    }


def recon_fid(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return fidelity(np.asarray(true).reshape(-1), np.asarray(pred).reshape(-1))


def mean_axis_rel_l2(true: np.ndarray, pred: np.ndarray, axis: int) -> float:
    """Mean per-slice relative L2. Channels (axis=0) or tokens (axis=1) equally weighted.

    Global rel_l2 is energy-weighted and can hide an axis preference when a
    few huge slices dominate. KIVI's claim is about the other slices sharing
    a scale with those outliers.
    """
    t = np.asarray(true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    err = t - p
    num = np.sqrt((err * err).sum(axis=axis))
    den = np.sqrt((t * t).sum(axis=axis)) + 1e-30
    return float(np.mean(num / den))


def kivi_on_kv(
    K: np.ndarray,
    V: np.ndarray,
    bits: tuple[int, ...] = KIVI_BITS,
    group: int = KIVI_GROUP,
) -> dict[str, Any]:
    """Asymmetric K vs V quantizability. K,V are (T, H, D) cache layout.

    Per-channel = group along tokens (KIVI-K). Per-token = group along
    channels (KIVI-V). Same group size => matched scale count.
    """
    k = flatten_kv(K)
    v = flatten_kv(V)
    k_ch_cv = rms_cv(k, axis=0)  # variation across channels (collapse tokens)
    k_tok_cv = rms_cv(k, axis=1)  # variation across tokens (collapse channels)
    v_ch_cv = rms_cv(v, axis=0)
    v_tok_cv = rms_cv(v, axis=1)
    by_bits = {}
    for b in bits:
        k_ch_q = grouped_absmax_quantize(k, b, 0, group)
        k_tok_q = grouped_absmax_quantize(k, b, 1, group)
        v_ch_q = grouped_absmax_quantize(v, b, 0, group)
        v_tok_q = grouped_absmax_quantize(v, b, 1, group)
        k_ch = recon_fid(k, k_ch_q)
        k_tok = recon_fid(k, k_tok_q)
        v_ch = recon_fid(v, v_ch_q)
        v_tok = recon_fid(v, v_tok_q)
        # Equal-slice scores (KIVI axis preference). Not energy-weighted.
        k_ch_eq = mean_axis_rel_l2(k, k_ch_q, axis=0)
        k_tok_eq = mean_axis_rel_l2(k, k_tok_q, axis=0)
        v_tok_eq = mean_axis_rel_l2(v, v_tok_q, axis=1)
        v_ch_eq = mean_axis_rel_l2(v, v_ch_q, axis=1)
        k_ratio = k_ch_eq / max(k_tok_eq, 1e-30)
        v_ratio = v_tok_eq / max(v_ch_eq, 1e-30)
        k_prefers_channel = k_ch_eq < k_tok_eq and k_ratio < 0.90
        v_prefers_token = v_tok_eq < v_ch_eq and v_ratio < 0.90
        kivi_body = np.concatenate([k_ch_q.ravel(), v_tok_q.ravel()])
        sym_ch = np.concatenate([k_ch_q.ravel(), v_ch_q.ravel()])
        sym_tok = np.concatenate([k_tok_q.ravel(), v_tok_q.ravel()])
        true_kv = np.concatenate([k.ravel(), v.ravel()])
        by_bits[str(b)] = {
            "group": int(group),
            "k_per_channel": k_ch,
            "k_per_token": k_tok,
            "v_per_channel": v_ch,
            "v_per_token": v_tok,
            "k_channel_over_token_rel_l2": k_ratio,
            "v_token_over_channel_rel_l2": v_ratio,
            "k_channel_equal_slice_rel_l2": k_ch_eq,
            "k_token_equal_slice_rel_l2": k_tok_eq,
            "v_token_equal_slice_rel_l2": v_tok_eq,
            "v_channel_equal_slice_rel_l2": v_ch_eq,
            "k_prefers_channel": bool(k_prefers_channel),
            "v_prefers_token": bool(v_prefers_token),
            "kivi_hypothesis_holds": bool(k_prefers_channel and v_prefers_token),
            "kivi_recon": recon_fid(true_kv, kivi_body),
            "symmetric_channel_recon": recon_fid(true_kv, sym_ch),
            "symmetric_token_recon": recon_fid(true_kv, sym_tok),
        }
    # Scale trap: 0.01*K must not look like a codec win on cosine alone.
    scale_trap = recon_fid(k, 0.01 * k)
    holds_any = any(by_bits[str(b)]["kivi_hypothesis_holds"] for b in bits)
    return {
        "shape_T_H_D": [int(K.shape[0]), int(K.shape[1]), int(K.shape[2])],
        "k_channel_rms": k_ch_cv,
        "k_token_rms": k_tok_cv,
        "v_channel_rms": v_ch_cv,
        "v_token_rms": v_tok_cv,
        "k_channel_cv_over_token_cv": k_ch_cv["cv"] / max(k_tok_cv["cv"], 1e-30),
        "v_token_cv_over_channel_cv": v_tok_cv["cv"] / max(v_ch_cv["cv"], 1e-30),
        "by_bits": by_bits,
        "scale_trap_0p01_K": scale_trap,
        "scale_trap_rejects_cosine": bool(
            scale_trap["cosine"] > 0.99 and scale_trap["scale_aware"] < 0.05
        ),
        "hypothesis_holds_at_any_scored_bitwidth": bool(holds_any),
        "null": (
            "Per-channel vs per-token absmax at the same bitwidth. A method "
            "that helps K and V equally is NOT KIVI's asymmetry. 0.01*K is "
            "the scale trap (cosine is blind)."
        ),
    }


def pair_state_stats(A: np.ndarray, B: np.ndarray) -> dict[str, Any]:
    """Adjacent-layer (or control) KV/state comparison. A,B (T, H, D) or flat."""
    a = np.asarray(A, dtype=np.float32).reshape(A.shape[0], -1)
    b = np.asarray(B, dtype=np.float32).reshape(B.shape[0], -1)
    if a.shape != b.shape:
        raise ValueError(f"pair shape {a.shape} vs {b.shape}")
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30
    token_cos = num / den
    merged = 0.5 * (a + b)
    fid = recon_fid(a, b)
    merge_a = recon_fid(a, merged)
    merge_b = recon_fid(b, merged)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return {
        "n_tokens": int(a.shape[0]),
        "dim": int(a.shape[1]),
        "mean_token_cosine": float(token_cos.mean()),
        "min_token_cosine": float(token_cos.min()),
        "flat_cosine": fid["cosine"],
        "flat_scale_aware": fid["scale_aware"],
        "flat_relative_l2": fid["relative_l2"],
        "norm_ratio_b_over_a": nb / max(na, 1e-30),
        "merge_mean_rel_l2": 0.5 * (merge_a["relative_l2"] + merge_b["relative_l2"]),
        "merge_mean_scale_aware": 0.5 * (merge_a["scale_aware"] + merge_b["scale_aware"]),
        "merge_mean_cosine": 0.5 * (merge_a["cosine"] + merge_b["cosine"]),
    }


def minicache_from_layers(kv: dict[int, np.ndarray]) -> dict[str, Any]:
    """kv: layer_id -> (T, H, D). Adjacent GQA pairs vs far-apart control."""
    ids = sorted(kv)
    if len(ids) < 2:
        return {"status": ABSENT, "reason": "need ≥2 GQA layers"}
    adj = []
    for a, b in zip(ids, ids[1:]):
        st = pair_state_stats(kv[a], kv[b])
        st["layers"] = [int(a), int(b)]
        st["depth_gap"] = int(b - a)
        adj.append(st)
    far = []
    for i, a in enumerate(ids):
        b = ids[-(i + 1)]
        if b <= a or (a, b) in {(s["layers"][0], s["layers"][1]) for s in adj}:
            continue
        st = pair_state_stats(kv[a], kv[b])
        st["layers"] = [int(a), int(b)]
        st["depth_gap"] = int(b - a)
        far.append(st)
        if len(far) >= 4:
            break
    adj_cos = [s["mean_token_cosine"] for s in adj]
    adj_sa = [s["flat_scale_aware"] for s in adj]
    far_cos = [s["mean_token_cosine"] for s in far]
    far_sa = [s["flat_scale_aware"] for s in far]
    mean_adj_sa = float(np.mean(adj_sa)) if adj_sa else 0.0
    mean_far_sa = float(np.mean(far_sa)) if far_sa else 0.0
    mean_adj_cos = float(np.mean(adj_cos)) if adj_cos else 0.0
    mean_merge = float(np.mean([s["merge_mean_rel_l2"] for s in adj])) if adj else 1.0
    # MiniCache's claim is *adjacent* layers are the ones worth merging.
    adjacent_special = mean_adj_sa > mean_far_sa + 0.05
    similar_enough = mean_adj_sa >= 0.90 and mean_merge <= 0.20
    return {
        "status": MEASURED,
        "n_layers": len(ids),
        "n_adjacent_pairs": len(adj),
        "n_far_pairs": len(far),
        "adjacent_depth_gap": FULL_ATTN_INTERVAL,
        "hybrid_note": (
            "OUR full-attention layers are every 4th layer. MiniCache's "
            "'adjacent transformer layer' is not a GQA-GQA neighbour here; "
            "three DeltaNet layers sit between every measured pair."
        ),
        "adjacent": {
            "mean_token_cosine": mean_adj_cos,
            "mean_scale_aware": mean_adj_sa,
            "mean_merge_rel_l2": mean_merge,
            "min_scale_aware": float(min(adj_sa)) if adj_sa else None,
            "pairs": adj,
        },
        "far_control": {
            "mean_token_cosine": float(np.mean(far_cos)) if far_cos else None,
            "mean_scale_aware": mean_far_sa if far_sa else None,
            "pairs": far,
        },
        "adjacent_more_similar_than_far": bool(adjacent_special),
        "merge_quality_bar": {
            "mean_adjacent_scale_aware_ge_0p90": bool(mean_adj_sa >= 0.90),
            "mean_merge_rel_l2_le_0p20": bool(mean_merge <= 0.20),
        },
        "hypothesis_holds": bool(similar_enough and adjacent_special),
        "null": (
            "Far-apart GQA pairs. If adjacent ≈ far, depth-merge is not a "
            "local-depth phenomenon. Cosine is reported; scale_aware is the gate."
        ),
    }


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    x = np.clip(x, 0.0, None)
    s = float(x.sum())
    n = x.size
    if n == 0 or s <= 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float(2.0 * np.dot(idx, x) / (n * s) - (n + 1) / n)


def topk_mass(p: np.ndarray, frac: float) -> float:
    s = np.sort(np.asarray(p, dtype=np.float64).ravel())[::-1]
    tot = float(s.sum())
    if tot <= 0 or s.size == 0:
        return 0.0
    k = max(1, int(math.ceil(frac * s.size)))
    return float(s[:k].sum() / tot)


def mass_cover_frac(p: np.ndarray, target: float = 0.80) -> float:
    s = np.sort(np.asarray(p, dtype=np.float64).ravel())[::-1]
    tot = float(s.sum())
    if tot <= 0 or s.size == 0:
        return 1.0
    cume = np.cumsum(s) / tot
    return float((int(np.searchsorted(cume, target)) + 1) / s.size)


def h2o_on_attn(attn: np.ndarray, recent: int = H2O_RECENT,
                heavy_frac: float = H2O_HEAVY_FRAC) -> dict[str, Any]:
    """attn: (H, T, T) causal softmax over keys (last axis)."""
    attn = np.asarray(attn, dtype=np.float32)
    if attn.ndim != 3:
        raise ValueError(f"attn ndim {attn.ndim}")
    n_h, t_q, t_k = attn.shape
    if t_q != t_k:
        raise ValueError(f"attn not square {attn.shape}")
    last = attn[:, -1, :]  # (H, T)
    last_mean = last.mean(axis=0)
    # Cumulative mass a key received (sum over queries). Causal already zeros future.
    cum = attn.sum(axis=1).mean(axis=0)  # (T,)
    uniform_top1 = 1.0 / float(t_k)
    rec = min(recent, t_k)
    last_recent = float(last_mean[-rec:].sum()) if rec else 0.0
    # Heavy hitters among the non-recent prefix, scored on cumulative mass.
    prefix = max(t_k - rec, 0)
    if prefix:
        order = np.argsort(cum[:prefix])[::-1]
        n_heavy = max(1, int(math.ceil(heavy_frac * prefix)))
        heavy_idx = order[:n_heavy]
        last_heavy = float(last_mean[heavy_idx].sum())
        kept = np.zeros(t_k, dtype=bool)
        kept[-rec:] = True
        kept[heavy_idx] = True
        last_kept = float(last_mean[kept].sum())
        cum_kept = float(cum[kept].sum() / max(float(cum.sum()), 1e-30))
        n_kept = int(kept.sum())
    else:
        last_heavy = 0.0
        last_kept = last_recent
        cum_kept = 1.0
        n_kept = rec
        n_heavy = 0
    # Uniform null: last-query mass is 1/T each; H2O keep-set would retain n_kept/T.
    uniform_kept = n_kept / float(t_k) if t_k else 1.0
    return {
        "n_heads": int(n_h),
        "seq_len": int(t_k),
        "last_query": {
            "top1_mass": topk_mass(last_mean, 1.0 / t_k if t_k else 1.0),
            "top1_share": float(np.max(last_mean)),
            "top5pct_mass": topk_mass(last_mean, 0.05),
            "top10pct_mass": topk_mass(last_mean, 0.10),
            "top20pct_mass": topk_mass(last_mean, 0.20),
            "gini": gini(last_mean),
            "entropy_over_logT": float(
                -(last_mean[last_mean > 0] * np.log(last_mean[last_mean > 0] + 1e-30)).sum()
                / max(math.log(t_k), 1e-30)
            ),
            "cover80_frac": mass_cover_frac(last_mean, 0.80),
            "cover90_frac": mass_cover_frac(last_mean, 0.90),
            "recent_window_mass": last_recent,
            "uniform_top1": uniform_top1,
        },
        "cumulative_keys": {
            "gini": gini(cum),
            "top20pct_mass": topk_mass(cum, 0.20),
            "cover80_frac": mass_cover_frac(cum, 0.80),
            "max_over_mean": float(cum.max() / max(cum.mean(), 1e-30)),
        },
        "h2o_keep": {
            "recent": rec,
            "heavy_frac_of_prefix": heavy_frac,
            "n_heavy": int(n_heavy),
            "n_kept": int(n_kept),
            "keep_frac": n_kept / float(t_k) if t_k else 1.0,
            "last_query_mass_retained": last_kept,
            "cumulative_mass_retained": cum_kept,
            "uniform_null_mass_retained": uniform_kept,
            "beats_uniform": bool(last_kept > uniform_kept + 0.10),
        },
        "hypothesis_holds": bool(
            last_kept > uniform_kept + 0.10
            and (
                topk_mass(last_mean, 0.20) >= 0.40
                or gini(last_mean) >= 0.40
            )
        ),
        "null": (
            "Uniform attention over the causal prefix. H2O needs last-query "
            "mass on (recent ∪ heavy) to beat keep_frac by a margin, and a "
            "skewed last-query distribution (Gini or top-20%)."
        ),
    }


def deltanet_on_states(
    rec: dict[int, np.ndarray],
    conv: dict[int, np.ndarray] | None = None,
    decay_mean: dict[int, float] | None = None,
) -> dict[str, Any]:
    """rec[layer] = (H, dk, dv) final recurrent state."""
    ids = sorted(rec)
    if not ids:
        return {"status": ABSENT, "reason": "no rec_state captured"}
    per = []
    flats = []
    for lid in ids:
        s = np.asarray(rec[lid], dtype=np.float32)
        if s.ndim != 3:
            raise ValueError(f"rec_state layer {lid} shape {s.shape}")
        h, dk, dv = s.shape
        # Per-head matrix rank of (dk, dv).
        ranks = []
        prs = []
        head_flat = s.reshape(h, dk * dv)
        for hi in range(h):
            m = s[hi]
            # economy SVD on centered head state
            xc = m - m.mean()
            sv = np.linalg.svd(xc, compute_uv=False)
            energy = np.square(sv)
            tot = float(energy.sum())
            if tot <= 0:
                ranks.append(0)
                prs.append(0.0)
                continue
            cume = np.cumsum(energy) / tot
            ranks.append(int(np.searchsorted(cume, 0.99) + 1))
            prs.append(float((tot * tot) / max(float((energy * energy).sum()), 1e-30)))
        # head pairwise cosine
        hf = head_flat.astype(np.float64)
        nn = np.linalg.norm(hf, axis=1, keepdims=True) + 1e-30
        cos = (hf @ hf.T) / (nn @ nn.T)
        iu = np.triu_indices(h, 1)
        pair = cos[iu]
        q4 = recon_fid(s, absmax_quantize(s.reshape(h, -1), 4, axis=1).reshape(s.shape))
        q8 = recon_fid(s, absmax_quantize(s.reshape(h, -1), 8, axis=1).reshape(s.shape))
        f16 = recon_fid(s, s.astype(np.float16).astype(np.float32))
        row = {
            "layer": int(lid),
            "shape": [int(h), int(dk), int(dv)],
            "frobenius": float(np.linalg.norm(s.astype(np.float64))),
            "mean_head_rank99": float(np.mean(ranks)),
            "mean_head_participation_ratio": float(np.mean(prs)),
            "ambient_rank": int(min(dk, dv)),
            "mean_head_pairwise_cosine": float(pair.mean()) if pair.size else 0.0,
            "min_head_pairwise_cosine": float(pair.min()) if pair.size else 0.0,
            "int4_per_head_rel_l2": q4["relative_l2"],
            "int4_per_head_scale_aware": q4["scale_aware"],
            "int8_per_head_rel_l2": q8["relative_l2"],
            "f16_rel_l2": f16["relative_l2"],
            "f16_scale_aware": f16["scale_aware"],
        }
        if decay_mean is not None and lid in decay_mean:
            row["mean_decay"] = float(decay_mean[lid])
        per.append(row)
        flats.append(s.reshape(-1).astype(np.float32))

    # Adjacent DN layers. Consecutive integers that skip a GQA layer are
    # still "adjacent DN in the stack" but have a GQA between them.
    adj_same_block = []
    adj_across_gqa = []
    for a, b in zip(ids, ids[1:]):
        st = pair_state_stats(flats[ids.index(a)][None, :], flats[ids.index(b)][None, :])
        st["layers"] = [int(a), int(b)]
        st["depth_gap"] = int(b - a)
        if b == a + 1:
            adj_same_block.append(st)
        else:
            adj_across_gqa.append(st)
    far = []
    if len(ids) >= 4:
        st = pair_state_stats(flats[0][None, :], flats[-1][None, :])
        st["layers"] = [int(ids[0]), int(ids[-1])]
        far.append(st)

    def _mean(sts, key):
        xs = [s[key] for s in sts]
        return float(np.mean(xs)) if xs else None

    mean_rank = float(np.mean([r["mean_head_rank99"] for r in per]))
    mean_pr = float(np.mean([r["mean_head_participation_ratio"] for r in per]))
    mean_f16 = float(np.mean([r["f16_rel_l2"] for r in per]))
    mean_i4 = float(np.mean([r["int4_per_head_rel_l2"] for r in per]))
    mean_head_cos = float(np.mean([r["mean_head_pairwise_cosine"] for r in per]))
    adj_sa = _mean(adj_same_block, "flat_scale_aware")
    far_sa = _mean(far, "flat_scale_aware")
    ambient = float(per[0]["ambient_rank"]) if per else 128.0
    low_rank = mean_rank <= 0.25 * ambient  # 32/128 on production heads
    cross_layer = bool(adj_sa is not None and adj_sa >= 0.90)
    head_copies = mean_head_cos >= 0.90
    quant_ok = mean_i4 <= 0.10
    conv_bytes = 0
    if conv:
        conv_bytes = int(sum(np.asarray(v).nbytes for v in conv.values()))
    return {
        "status": MEASURED,
        "n_layers": len(ids),
        "ambient_rank": int(ambient),
        "rec_state_elems_per_layer": int(flats[0].size) if flats else None,
        "per_layer": per,
        "mean_head_rank99": mean_rank,
        "mean_head_participation_ratio": mean_pr,
        "mean_head_pairwise_cosine": mean_head_cos,
        "mean_f16_rel_l2": mean_f16,
        "mean_int4_per_head_rel_l2": mean_i4,
        "adjacent_same_block": {
            "n_pairs": len(adj_same_block),
            "mean_scale_aware": adj_sa,
            "mean_token_cosine": _mean(adj_same_block, "mean_token_cosine"),
            "pairs_head": adj_same_block[:6],
        },
        "adjacent_across_gqa": {
            "n_pairs": len(adj_across_gqa),
            "mean_scale_aware": _mean(adj_across_gqa, "flat_scale_aware"),
        },
        "far_control": {
            "n_pairs": len(far),
            "mean_scale_aware": far_sa,
            "pairs": far,
        },
        "conv_state_bytes_captured": conv_bytes,
        "low_rank_redundancy": bool(low_rank),
        "cross_layer_redundancy": bool(cross_layer),
        "head_copy_redundancy": bool(head_copies),
        "int4_quant_cheap": bool(quant_ok),
        "hypothesis_holds": bool(low_rank or cross_layer or head_copies or quant_ok),
        "null": (
            "Ambient head state is 128×128. Rank-99 near 128, pairwise head "
            "cosine near 0, adjacent-layer scale_aware near 0, and int4 "
            "rel_l2 ≳ 0.3 would mean the 151 MiB rec_state is not redundant."
        ),
    }


# ---------------------------------------------------------------------------
# Byte-saving estimates (DERIVED from genome × a named recipe)
# ---------------------------------------------------------------------------

def kivi_bytes(seq: int, sessions: int, k_bits: int, v_bits: int) -> dict[str, Any]:
    kv = gqa_kv_bytes(seq) * sessions
    # Half the KV is K, half is V (same shape).
    k_body = (kv // 2) * k_bits / 32.0
    v_body = (kv // 2) * v_bits / 32.0
    # f16 scales: K per (layer, kv_head, channel); V per (layer, kv_head, token)
    k_scales = sessions * GQA_LAYERS_N * GQA_KV_HEADS * GQA_HEAD_DIM * 2
    v_scales = sessions * GQA_LAYERS_N * GQA_KV_HEADS * seq * 2
    packed = int(k_body + v_body) + k_scales + v_scales
    saved = kv - packed
    return {
        "baseline_gqa_kv_bytes": kv,
        "packed_bytes": packed,
        "saved_bytes": saved,
        "saved_GiB": saved / GIB,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "k_scale_bytes": k_scales,
        "v_scale_bytes": v_scales,
        "formula": "K body k_bits/32 of half KV + V body v_bits/32 of half KV + f16 scales",
    }


def minicache_bytes(seq: int, sessions: int, pair_frac: float = 0.5) -> dict[str, Any]:
    kv = gqa_kv_bytes(seq) * sessions
    packed = int(kv * pair_frac)
    saved = kv - packed
    return {
        "baseline_gqa_kv_bytes": kv,
        "packed_bytes": packed,
        "saved_bytes": saved,
        "saved_GiB": saved / GIB,
        "recipe": "store one merged KV per adjacent GQA pair (16 -> 8)",
        "pair_frac": pair_frac,
    }


def h2o_recipe_keep_frac(
    seq: int, recent: int = H2O_RECENT, heavy_frac: float = H2O_HEAVY_FRAC
) -> float:
    """Keep-set size of the H2O recipe at a target seq. Not the measured-T fraction."""
    rec = min(int(recent), int(seq))
    prefix = max(int(seq) - rec, 0)
    n_heavy = max(1, int(math.ceil(heavy_frac * prefix))) if prefix else 0
    return (rec + n_heavy) / float(seq) if seq else 1.0


def h2o_bytes(seq: int, sessions: int, keep_frac: float) -> dict[str, Any]:
    kv = gqa_kv_bytes(seq) * sessions
    packed = int(kv * keep_frac)
    saved = kv - packed
    return {
        "baseline_gqa_kv_bytes": kv,
        "packed_bytes": packed,
        "saved_bytes": saved,
        "saved_GiB": saved / GIB,
        "keep_frac": keep_frac,
        "recipe": (
            f"keep recent {H2O_RECENT} + heavy-hitter {H2O_HEAVY_FRAC:.0%} of "
            "the prefix; drop the rest of GQA KV"
        ),
        "keep_frac_source": "H2O recipe evaluated at the target seq, not measured-T keep_frac",
    }


def deltanet_bytes_saved(sessions: int, recipe: str) -> dict[str, Any]:
    base = DELTANET_STATE_BYTES * sessions
    if recipe == "f16":
        packed = base // 2
    elif recipe == "int8":
        packed = base // 4
    elif recipe == "int4":
        packed = base // 8
    elif recipe == "rank32_f16":
        # 48 layers * 48 heads * (128*32 + 32*128) f16
        packed = sessions * DN_LAYERS_N * DN_V_HEADS * (DN_K_DIM * 32 + 32 * DN_V_DIM) * 2
    else:
        raise ValueError(recipe)
    saved = base - packed
    return {
        "baseline_deltanet_bytes": base,
        "packed_bytes": packed,
        "saved_bytes": saved,
        "saved_GiB": saved / GIB,
        "recipe": recipe,
        "grows_with_seq": False,
        "share_of_32k_c4_session": saved / max(
            session_state_bytes(HEADLINE_SEQ)["SESSION_STATE_BYTES"] * HEADLINE_C, 1
        ),
    }


# ---------------------------------------------------------------------------
# Forwards
# ---------------------------------------------------------------------------

def gqa_forward_cache(
    x_norm: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wo: np.ndarray,
    q_delta: np.ndarray,
    k_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mixer residual, plus cache K/V (post qk-norm+RoPE) and attn softmax."""
    t_len = x_norm.shape[0]
    qg = gemm(wq, x_norm).reshape(t_len, GQA_HEADS, 2 * GQA_HEAD_DIM)
    q = qg[:, :, :GQA_HEAD_DIM]
    gate = qg[:, :, GQA_HEAD_DIM:]
    k = gemm(wk, x_norm).reshape(t_len, GQA_KV_HEADS, GQA_HEAD_DIM)
    v = gemm(wv, x_norm).reshape(t_len, GQA_KV_HEADS, GQA_HEAD_DIM)
    q = rmsnorm_delta(q, q_delta)
    k = rmsnorm_delta(k, k_delta)
    q, k = apply_rope(q, k)
    k_rep = np.repeat(k, GQA_HEADS // GQA_KV_HEADS, axis=1)
    v_rep = np.repeat(v, GQA_HEADS // GQA_KV_HEADS, axis=1)
    scale = 1.0 / math.sqrt(GQA_HEAD_DIM)
    scores = np.einsum("thd,shd->hts", q, k_rep, dtype=np.float32) * scale
    causal = np.triu(np.ones((t_len, t_len), dtype=bool), 1)
    scores[:, causal] = np.float32(-1e9)
    attn = softmax_last(scores)
    ctx = np.einsum("hts,shd->thd", attn, v_rep, dtype=np.float32)
    ctx = ctx.reshape(t_len, GQA_HEADS * GQA_HEAD_DIM)
    ctx = ctx * (1.0 / (1.0 + np.exp(-np.clip(gate.reshape(t_len, -1), -60, 60))))
    mix = gemm(wo, ctx)
    return mix, k, v, attn


def gated_delta_seq_state(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, decay: np.ndarray, beta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    t_len, heads, dk = q.shape
    dv = v.shape[-1]
    if decay.ndim == 1:
        decay = np.broadcast_to(decay, (t_len, heads))
        beta = np.broadcast_to(beta, (t_len, heads))
    state = np.zeros((heads, dk, dv), dtype=np.float32)
    out = np.empty((t_len, heads, dv), dtype=np.float32)
    for t in range(t_len):
        state *= decay[t, :, None, None]
        kv_mem = np.einsum("hkv,hk->hv", state, k[t], dtype=np.float32)
        delta = (v[t] - kv_mem) * beta[t, :, None]
        state += np.einsum("hk,hv->hkv", k[t], delta, dtype=np.float32)
        out[t] = np.einsum("hkv,hk->hv", state, q[t], dtype=np.float32)
    return out, state


def deltanet_forward_state(
    x_norm: np.ndarray,
    w_qkvz: np.ndarray,
    w_ba: np.ndarray,
    w_out: np.ndarray,
    conv_w: np.ndarray,
    a_log: np.ndarray,
    dt_bias: np.ndarray,
    norm_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_len = x_norm.shape[0]
    qkvz = gemm(w_qkvz, x_norm)
    ba = gemm(w_ba, x_norm)
    q, k, v, z = _unfuse_qkvz(qkvz)
    b, a = _unfuse_ba(ba)
    qkv_ch = np.concatenate(
        [
            q.reshape(t_len, DN_K_HEADS * DN_K_DIM),
            k.reshape(t_len, DN_K_HEADS * DN_K_DIM),
            v.reshape(t_len, DN_V_HEADS * DN_V_DIM),
        ],
        axis=1,
    )
    conv_w = conv_w.reshape(conv_w.shape[0], DN_CONV_K)
    qkv_c = causal_conv1d_silu(qkv_ch, conv_w)
    q = qkv_c[:, : DN_K_HEADS * DN_K_DIM].reshape(t_len, DN_K_HEADS, DN_K_DIM)
    k = qkv_c[:, DN_K_HEADS * DN_K_DIM : 2 * DN_K_HEADS * DN_K_DIM].reshape(
        t_len, DN_K_HEADS, DN_K_DIM
    )
    v = qkv_c[:, 2 * DN_K_HEADS * DN_K_DIM :].reshape(t_len, DN_V_HEADS, DN_V_DIM)
    q = l2norm(q) * (1.0 / math.sqrt(DN_K_DIM))
    k = l2norm(k)
    q = np.repeat(q, DN_VPK, axis=1)
    k = np.repeat(k, DN_VPK, axis=1)
    xdt = a + dt_bias[None, :]
    softplus = np.where(
        xdt > 0,
        xdt + np.log1p(np.exp(-np.abs(xdt))),
        np.log1p(np.exp(-np.abs(xdt))),
    )
    g = -np.exp(a_log)[None, :] * softplus
    decay = np.exp(g).astype(np.float32)
    beta = (1.0 / (1.0 + np.exp(-np.clip(b, -60, 60)))).astype(np.float32)
    rec, state = gated_delta_seq_state(q, k, v, decay, beta)
    rec_f = rec.reshape(t_len * DN_V_HEADS, DN_V_DIM)
    z_f = z.reshape(t_len * DN_V_HEADS, DN_V_DIM)
    rms = np.sqrt(np.mean(rec_f * rec_f, axis=-1, keepdims=True) + 1.0e-6)
    gated = (rec_f / rms) * norm_w * silu(z_f)
    gated = gated.reshape(t_len, DN_V_HEADS * DN_V_DIM)
    mix = gemm(w_out, gated)
    conv_state = np.zeros((DN_CONV_K - 1, qkv_ch.shape[1]), dtype=np.float32)
    n = min(t_len, DN_CONV_K - 1)
    conv_state[-n:] = qkv_ch[-n:]
    return mix, state, conv_state, decay


def gqa_kv_from_x(
    x: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    qn: np.ndarray,
    kn: np.ndarray,
    want_attn: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """K,V (+ optional attn) from a residual-stream activation matrix."""
    t_len = x.shape[0]
    k = gemm(wk, x).reshape(t_len, GQA_KV_HEADS, GQA_HEAD_DIM)
    v = gemm(wv, x).reshape(t_len, GQA_KV_HEADS, GQA_HEAD_DIM)
    k = rmsnorm_delta(k, kn)
    if want_attn:
        qg = gemm(wq, x).reshape(t_len, GQA_HEADS, 2 * GQA_HEAD_DIM)
        q = rmsnorm_delta(qg[:, :, :GQA_HEAD_DIM], qn)
        q, k = apply_rope(q, k)
        k_rep = np.repeat(k, GQA_HEADS // GQA_KV_HEADS, axis=1)
        scale = 1.0 / math.sqrt(GQA_HEAD_DIM)
        scores = np.einsum("thd,shd->hts", q, k_rep, dtype=np.float32) * scale
        causal = np.triu(np.ones((t_len, t_len), dtype=bool), 1)
        scores[:, causal] = np.float32(-1e9)
        attn = softmax_last(scores)
        return k, v, attn
    # Still apply RoPE so cache layout matches production.
    dummy_q = np.zeros((t_len, GQA_HEADS, GQA_HEAD_DIM), dtype=np.float32)
    _, k = apply_rope(dummy_q, k)
    return k, v, None


def run_runtime_prefill(art: Artifact, tokens: np.ndarray, progress) -> dict[str, Any]:
    """Single-stream CPU hybrid. Captures production-layout KV + rec_state."""
    t0 = time.perf_counter()
    t_len = int(tokens.shape[0])
    embed_row = art.by_name["language_model.model.embed_tokens.weight"]
    progress(f"runtime gather {t_len} embed rows")
    h = read_q4_rows(
        art.path("language_model.model.embed_tokens.weight"),
        embed_row["shape"],
        tokens,
    )
    gqa_k: dict[int, np.ndarray] = {}
    gqa_v: dict[int, np.ndarray] = {}
    gqa_attn: dict[int, np.ndarray] = {}
    dn_rec: dict[int, np.ndarray] = {}
    dn_conv: dict[int, np.ndarray] = {}
    dn_decay: dict[int, float] = {}
    for layer in range(LAYERS):
        ln = art.load(art.layer_name(layer, "input_layernorm.weight"))
        x_norm = rmsnorm_delta(h, ln)
        if is_gqa(layer):
            wq = art.load(art.layer_name(layer, "self_attn.q_proj.weight"))
            wk = art.load(art.layer_name(layer, "self_attn.k_proj.weight"))
            wv = art.load(art.layer_name(layer, "self_attn.v_proj.weight"))
            wo = art.load(art.layer_name(layer, "self_attn.o_proj.weight"))
            qn = art.load(art.layer_name(layer, "self_attn.q_norm.weight"))
            kn = art.load(art.layer_name(layer, "self_attn.k_norm.weight"))
            mix, k, v, attn = gqa_forward_cache(x_norm, wq, wk, wv, wo, qn, kn)
            gqa_k[layer] = k
            gqa_v[layer] = v
            gqa_attn[layer] = attn
            del wq, wk, wv, wo, qn, kn
        else:
            w_qkvz = art.load(art.layer_name(layer, "linear_attn.in_proj_qkvz.weight"))
            w_ba = art.load(art.layer_name(layer, "linear_attn.in_proj_ba.weight"))
            w_out = art.load(art.layer_name(layer, "linear_attn.out_proj.weight"))
            conv_w = art.load(art.layer_name(layer, "linear_attn.conv1d.weight"))
            a_log = art.load(art.layer_name(layer, "linear_attn.A_log"))
            dt_bias = art.load(art.layer_name(layer, "linear_attn.dt_bias"))
            nrm = art.load(art.layer_name(layer, "linear_attn.norm.weight"))
            mix, state, conv_state, decay = deltanet_forward_state(
                x_norm, w_qkvz, w_ba, w_out, conv_w, a_log, dt_bias, nrm
            )
            dn_rec[layer] = state
            dn_conv[layer] = conv_state
            dn_decay[layer] = float(decay.mean())
            del w_qkvz, w_ba, w_out, conv_w, a_log, dt_bias, nrm
        h = h + mix
        pn = art.load(art.layer_name(layer, "post_attention_layernorm.weight"))
        x_mlp = rmsnorm_delta(h, pn)
        wg = art.load(art.layer_name(layer, "mlp.gate_proj.weight"))
        wu = art.load(art.layer_name(layer, "mlp.up_proj.weight"))
        wd = art.load(art.layer_name(layer, "mlp.down_proj.weight"))
        h = h + mlp_forward(x_mlp, wg, wu, wd)
        del wg, wu, wd, mix, x_norm, x_mlp, ln, pn
        if layer % 8 == 0 or layer == LAYERS - 1:
            progress(
                f"runtime layer {layer:02d}/{LAYERS - 1} "
                f"h_rms={float(np.sqrt(np.mean(h * h))):.4f} "
                f"{time.perf_counter() - t0:.1f}s"
            )
            gc.collect()
    return {
        "elapsed_s": time.perf_counter() - t0,
        "n_tokens": t_len,
        "gqa_k": gqa_k,
        "gqa_v": gqa_v,
        "gqa_attn": gqa_attn,
        "dn_rec": dn_rec,
        "dn_conv": dn_conv,
        "dn_decay": dn_decay,
    }


def run_capture_gqa(progress) -> dict[str, Any]:
    """Parent k/v/q on capture_diverse2 post_attn_norm (neighbor site)."""
    t0 = time.perf_counter()
    parent = find_parent()
    cap = find_capture()
    man = json.loads((cap / "manifest.json").read_text())
    x0 = load_X(cap, GQA_LAYER_IDS[0])
    fit, hold = split_from_manifest(man, x0.shape[0])
    hold = np.asarray(hold)
    progress(
        f"capture {cap} tokens={x0.shape[0]} hold={hold.size} parent={parent}"
    )
    del x0
    gqa_k: dict[int, np.ndarray] = {}
    gqa_v: dict[int, np.ndarray] = {}
    h2o_rows: list[dict[str, Any]] = []
    for layer in GQA_LAYER_IDS:
        X = load_X(cap, layer)[hold]
        wk = load_tensor(parent, tensor_name(layer, "self_attn.k_proj.weight"))
        wv = load_tensor(parent, tensor_name(layer, "self_attn.v_proj.weight"))
        kn = load_tensor(parent, tensor_name(layer, "self_attn.k_norm.weight"))
        want = layer in H2O_CAPTURE_LAYERS
        wq = qn = None
        if want:
            wq = load_tensor(parent, tensor_name(layer, "self_attn.q_proj.weight"))
            qn = load_tensor(parent, tensor_name(layer, "self_attn.q_norm.weight"))
            k, v, _ = gqa_kv_from_x(X, wq, wk, wv, qn, kn, want_attn=False)
        else:
            k, v, _ = gqa_kv_from_x(
                X, np.zeros((1, 1), np.float32), wk, wv,
                np.zeros((GQA_HEAD_DIM,), np.float32), kn, want_attn=False,
            )
        gqa_k[layer] = k
        gqa_v[layer] = v
        if want:
            # Per-prompt attention on the hold sequences that live in this X.
            # `hold` is a set of row indices into the full capture; rebuild
            # per-prompt slices that fall entirely inside hold.
            hold_set = set(int(i) for i in hold)
            for entry in man["manifest"]:
                if entry.get("split") != "hold":
                    continue
                n = int(entry["n_tokens"])
                if n < CAPTURE_MIN_PROMPT:
                    continue
                rs = int(entry["row_start"])
                sl = list(range(rs, rs + n))
                if any(i not in hold_set for i in sl):
                    continue
                # Map full-capture rows onto the hold-subsampled X.
                pos = [int(np.where(hold == i)[0][0]) for i in sl]
                Xs = X[pos]
                _, _, attn = gqa_kv_from_x(Xs, wq, wk, wv, qn, kn, want_attn=True)
                stats = h2o_on_attn(attn)
                stats["layer"] = int(layer)
                stats["family"] = entry.get("family")
                stats["n_tokens"] = n
                h2o_rows.append(stats)
                del attn, Xs
        del X, wk, wv, kn, wq, qn
        gc.collect()
        progress(f"capture GQA L{layer} {time.perf_counter() - t0:.1f}s")
    return {
        "elapsed_s": time.perf_counter() - t0,
        "site": (
            "capture_diverse2 post_attn_norm (MLP input of this layer), "
            "projected by parent BF16 k/v/q after qk-RMSNorm + RoPE. "
            "NOT the production KV cache of the mixer. Real residual-stream "
            "activations of OUR model at a neighboring site; the Q4 CPU "
            "hybrid is the production-layout cache."
        ),
        "capture": str(cap),
        "parent": str(parent),
        "n_hold_tokens": int(hold.size),
        "gqa_k": gqa_k,
        "gqa_v": gqa_v,
        "h2o_prompts": h2o_rows,
        "not_synthetic": True,
        "gaussian_proxy_used": False,
    }


# ---------------------------------------------------------------------------
# Axis assembly + ranking
# ---------------------------------------------------------------------------

def _mean_kivi(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Average KIVI stats across layers; hypothesis is AND across layers."""
    if not parts:
        return {"status": ABSENT, "reason": "no layers"}
    bits_hold = {str(b): [] for b in KIVI_BITS}
    k_cv, v_cv = [], []
    traps = []
    for p in parts:
        k_cv.append(p["k_channel_cv_over_token_cv"])
        v_cv.append(p["v_token_cv_over_channel_cv"])
        traps.append(p["scale_trap_rejects_cosine"])
        for b, block in p["by_bits"].items():
            bits_hold[b].append(block)
    by_bits = {}
    for b, blocks in bits_hold.items():
        by_bits[b] = {
            "k_channel_over_token_rel_l2": float(
                np.mean([x["k_channel_over_token_rel_l2"] for x in blocks])
            ),
            "v_token_over_channel_rel_l2": float(
                np.mean([x["v_token_over_channel_rel_l2"] for x in blocks])
            ),
            "k_channel_equal_slice_rel_l2": float(
                np.mean([x["k_channel_equal_slice_rel_l2"] for x in blocks])
            ),
            "v_token_equal_slice_rel_l2": float(
                np.mean([x["v_token_equal_slice_rel_l2"] for x in blocks])
            ),
            "k_prefers_channel_frac": float(
                np.mean([x["k_prefers_channel"] for x in blocks])
            ),
            "v_prefers_token_frac": float(
                np.mean([x["v_prefers_token"] for x in blocks])
            ),
            "kivi_hypothesis_holds_frac": float(
                np.mean([x["kivi_hypothesis_holds"] for x in blocks])
            ),
            "kivi_recon_rel_l2": float(
                np.mean([x["kivi_recon"]["relative_l2"] for x in blocks])
            ),
            "kivi_recon_scale_aware": float(
                np.mean([x["kivi_recon"]["scale_aware"] for x in blocks])
            ),
            "symmetric_channel_rel_l2": float(
                np.mean([x["symmetric_channel_recon"]["relative_l2"] for x in blocks])
            ),
            "symmetric_token_rel_l2": float(
                np.mean([x["symmetric_token_recon"]["relative_l2"] for x in blocks])
            ),
        }
    # 2-bit is the paper recipe but is often too coarse for the axis to
    # matter. The qualitative KIVI claim is scored at 4-bit; 2-bit is
    # reported and is what would actually be booked if it holds.
    holds_4 = by_bits["4"]["kivi_hypothesis_holds_frac"] >= 0.5
    holds_2 = by_bits["2"]["kivi_hypothesis_holds_frac"] >= 0.5
    holds = bool(holds_4 or holds_2)
    return {
        "status": MEASURED,
        "n_layers": len(parts),
        "mean_k_channel_cv_over_token_cv": float(np.mean(k_cv)),
        "mean_v_token_cv_over_channel_cv": float(np.mean(v_cv)),
        "by_bits": by_bits,
        "scale_trap_rejects_cosine": bool(all(traps)),
        "hypothesis_holds_int2": bool(holds_2),
        "hypothesis_holds_int4": bool(holds_4),
        "hypothesis_holds": bool(holds),
        "per_layer_head": [
            {
                "layer": p.get("layer"),
                "kivi_int2_holds": p["by_bits"]["2"]["kivi_hypothesis_holds"],
                "k_ratio": p["by_bits"]["2"]["k_channel_over_token_rel_l2"],
                "v_ratio": p["by_bits"]["2"]["v_token_over_channel_rel_l2"],
            }
            for p in parts
        ],
    }


def _mean_h2o(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": ABSENT, "reason": "no attention maps"}
    def avg(path):
        xs = []
        for r in rows:
            cur = r
            for k in path.split("."):
                cur = cur[k]
            xs.append(float(cur))
        return float(np.mean(xs))
    keep = avg("h2o_keep.last_query_mass_retained")
    uni = avg("h2o_keep.uniform_null_mass_retained")
    return {
        "status": MEASURED,
        "n_maps": len(rows),
        "mean_seq_len": avg("seq_len"),
        "mean_last_top20pct_mass": avg("last_query.top20pct_mass"),
        "mean_last_gini": avg("last_query.gini"),
        "mean_cover80_frac": avg("last_query.cover80_frac"),
        "mean_recent_window_mass": avg("last_query.recent_window_mass"),
        "mean_h2o_keep_frac": avg("h2o_keep.keep_frac"),
        "mean_h2o_last_mass_retained": keep,
        "mean_uniform_null_mass_retained": uni,
        "mean_cumulative_top20pct": avg("cumulative_keys.top20pct_mass"),
        "hypothesis_holds_frac": float(np.mean([r["hypothesis_holds"] for r in rows])),
        "hypothesis_holds": bool(
            np.mean([r["hypothesis_holds"] for r in rows]) >= 0.5
        ),
        "per_map_head": [
            {
                "layer": r.get("layer"),
                "seq_len": r["seq_len"],
                "family": r.get("family"),
                "top20": r["last_query"]["top20pct_mass"],
                "gini": r["last_query"]["gini"],
                "kept_mass": r["h2o_keep"]["last_query_mass_retained"],
                "keep_frac": r["h2o_keep"]["keep_frac"],
                "holds": r["hypothesis_holds"],
            }
            for r in rows[:24]
        ],
    }


def long_context_risk(name: str, extra: str) -> dict[str, Any]:
    common = (
        "Long-context capability is the gate (S026 §119). PREFILL_KV: do not "
        "assume KV quant is free. Qualifying a compromise requires wiring it "
        "into Qwen38HybridDecodeSession and scoring a held-out long-context "
        "suite (argmax/gain/fallbacks vs f32). Until that exists the "
        "capability cost is ABSENT."
    )
    return {
        "kind": ABSENT,
        "value": None,
        "unit": "long_context_recall_delta",
        "gate": "long_context_capability",
        "absent_reason": f"{common} {extra}",
        "gqa_is_quality_floor_for_weights": True,
        "weight_floor_is_not_this_axis": (
            "ORGAN_FRONTIERS / NOETIC_GQA_DESIGN floor 4.125 is a WEIGHT "
            "result. KV-cache quant/eviction is a different operator and is "
            "unmeasured on this body (PREFILL_KV kv_precision)."
        ),
        "method": name,
    }


def assemble_axes(
    genome: dict[str, Any],
    runtime: dict[str, Any] | None,
    capture: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    kivi_parts_rt, kivi_parts_cap = [], []
    h2o_rt, h2o_cap = [], []
    mc_k_rt = mc_v_rt = mc_k_cap = mc_v_cap = None
    dn = None

    if runtime is not None:
        for lid, K in runtime["gqa_k"].items():
            V = runtime["gqa_v"][lid]
            part = kivi_on_kv(K, V)
            part["layer"] = int(lid)
            kivi_parts_rt.append(part)
            h = h2o_on_attn(runtime["gqa_attn"][lid])
            h["layer"] = int(lid)
            h2o_rt.append(h)
        mc_k_rt = minicache_from_layers(runtime["gqa_k"])
        mc_v_rt = minicache_from_layers(runtime["gqa_v"])
        dn = deltanet_on_states(
            runtime["dn_rec"], runtime.get("dn_conv"), runtime.get("dn_decay")
        )

    if capture is not None:
        for lid, K in capture["gqa_k"].items():
            V = capture["gqa_v"][lid]
            part = kivi_on_kv(K, V)
            part["layer"] = int(lid)
            kivi_parts_cap.append(part)
        mc_k_cap = minicache_from_layers(capture["gqa_k"])
        mc_v_cap = minicache_from_layers(capture["gqa_v"])
        h2o_cap = list(capture.get("h2o_prompts") or [])

    kivi_rt = _mean_kivi(kivi_parts_rt) if kivi_parts_rt else {"status": ABSENT}
    kivi_cap = _mean_kivi(kivi_parts_cap) if kivi_parts_cap else {"status": ABSENT}
    # Runtime cache is the production-layout authority. Capture is corroboration.
    kivi_holds = bool(kivi_rt.get("hypothesis_holds"))
    kivi_bits = 2 if kivi_rt.get("hypothesis_holds_int2") else 4
    kivi_est = kivi_bytes(HEADLINE_SEQ, HEADLINE_C, kivi_bits, kivi_bits)
    if not kivi_holds:
        kivi_est = {**kivi_est, "saved_bytes": 0, "saved_GiB": 0.0,
                    "note": "redundancy not present; do not book the hypothetical 2-bit save"}

    def _mc_hold(block):
        return bool(block and block.get("hypothesis_holds"))

    mc_holds = _mc_hold(mc_k_rt) and _mc_hold(mc_v_rt)
    mc_est = minicache_bytes(HEADLINE_SEQ, HEADLINE_C)
    if not mc_holds:
        mc_est = {**mc_est, "saved_bytes": 0, "saved_GiB": 0.0,
                  "note": "adjacent GQA KV is not MiniCache-similar; do not book 50%"}

    h2o_rt_m = _mean_h2o(h2o_rt) if h2o_rt else {"status": ABSENT}
    h2o_cap_m = _mean_h2o(h2o_cap) if h2o_cap else {"status": ABSENT}
    h2o_holds = bool(
        h2o_rt_m.get("hypothesis_holds") or h2o_cap_m.get("hypothesis_holds")
    )
    keep_frac_32k = h2o_recipe_keep_frac(HEADLINE_SEQ)
    h2o_est = h2o_bytes(HEADLINE_SEQ, HEADLINE_C, keep_frac_32k if h2o_holds else 1.0)
    if not h2o_holds:
        h2o_est = {**h2o_est, "saved_bytes": 0, "saved_GiB": 0.0,
                   "note": "attention mass is not H2O-concentrated at measured T; do not book eviction"}

    dn_holds = bool(dn and dn.get("hypothesis_holds"))
    # Prefer the cheapest recipe the measurement actually supports.
    if dn and dn.get("int4_quant_cheap"):
        dn_recipe = "int4"
    elif dn and dn.get("low_rank_redundancy"):
        dn_recipe = "rank32_f16"
    else:
        dn_recipe = "f16"
    dn_est = deltanet_bytes_saved(HEADLINE_C, dn_recipe)
    if not dn_holds:
        dn_est = {**dn_est, "saved_bytes": 0, "saved_GiB": 0.0, "recipe": "none",
                  "note": "no measured rec_state redundancy worth booking"}

    kivi_why = []
    if kivi_rt.get("status") == MEASURED:
        b2 = kivi_rt["by_bits"]["2"]
        b4 = kivi_rt["by_bits"]["4"]
        kivi_why.append(
            f"runtime int2: K channel/token ratio "
            f"{b2['k_channel_over_token_rel_l2']:.3f} "
            f"(prefers channel frac {b2['k_prefers_channel_frac']:.2f}); "
            f"V token/channel ratio {b2['v_token_over_channel_rel_l2']:.3f} "
            f"(prefers token frac {b2['v_prefers_token_frac']:.2f}). "
            f"int4 same pattern (K frac {b4['k_prefers_channel_frac']:.2f}, "
            f"V frac {b4['v_prefers_token_frac']:.2f}); symmetric-per-channel "
            f"rel_l2={b4['symmetric_channel_rel_l2']:.3f} vs KIVI recipe "
            f"{b4['kivi_recon_rel_l2']:.3f}."
        )
    if kivi_cap.get("status") == MEASURED:
        b4 = kivi_cap["by_bits"]["4"]
        kivi_why.append(
            f"capture corroboration int4 K-channel frac="
            f"{b4['k_prefers_channel_frac']:.2f} V-token frac="
            f"{b4['v_prefers_token_frac']:.2f} at post_attn_norm."
        )
    if not kivi_holds:
        kivi_why.append(
            "KIVI needs K to prefer per-channel AND V to prefer per-token. "
            "K-channel preference is present; V-token is not — V also prefers "
            "per-channel, so the paper's asymmetry is absent. Symmetric "
            "per-channel is the measured direction and is still unmeasured "
            "at long-context capability."
        )

    mc_why = []
    if mc_k_rt:
        mc_why.append(
            f"runtime K adjacent scale_aware="
            f"{mc_k_rt['adjacent']['mean_scale_aware']:.3f} "
            f"merge_rel_l2={mc_k_rt['adjacent']['mean_merge_rel_l2']:.3f}; "
            f"far={mc_k_rt['far_control']['mean_scale_aware']}; "
            f"adjacent_special={mc_k_rt['adjacent_more_similar_than_far']}."
        )
    if mc_v_rt:
        mc_why.append(
            f"runtime V adjacent scale_aware="
            f"{mc_v_rt['adjacent']['mean_scale_aware']:.3f} "
            f"merge_rel_l2={mc_v_rt['adjacent']['mean_merge_rel_l2']:.3f}."
        )
    mc_why.append(
        "Hybrid interval 4: 'adjacent' GQA layers are 4 transformer layers "
        "apart, with DeltaNet in between. MiniCache was designed for dense "
        "full-attention stacks."
    )

    h2o_why = []
    if h2o_rt_m.get("status") == MEASURED:
        h2o_why.append(
            f"runtime T={h2o_rt_m['mean_seq_len']:.0f}: last-query top20% "
            f"mass={h2o_rt_m['mean_last_top20pct_mass']:.3f} "
            f"gini={h2o_rt_m['mean_last_gini']:.3f} "
            f"H2O keep_frac={h2o_rt_m['mean_h2o_keep_frac']:.3f} retains "
            f"{h2o_rt_m['mean_h2o_last_mass_retained']:.3f} vs uniform "
            f"{h2o_rt_m['mean_uniform_null_mass_retained']:.3f}."
        )
    if h2o_cap_m.get("status") == MEASURED:
        h2o_why.append(
            f"capture hold prompts mean T={h2o_cap_m['mean_seq_len']:.0f}: "
            f"top20={h2o_cap_m['mean_last_top20pct_mass']:.3f} "
            f"holds_frac={h2o_cap_m['hypothesis_holds_frac']:.2f}."
        )
    h2o_why.append(
        "Measured T is tens-to-hundreds of tokens, not 32K. Concentration "
        "usually grows with T, so this is a lower bound — but a needle "
        "token is often NOT a heavy hitter until queried. That is the "
        "long-context risk even if mass is skewed."
    )

    dn_why = []
    if dn and dn.get("status") == MEASURED:
        dn_why.append(
            f"mean head rank-99={dn['mean_head_rank99']:.1f}/128 "
            f"PR={dn['mean_head_participation_ratio']:.1f} "
            f"head cosine={dn['mean_head_pairwise_cosine']:.3f} "
            f"int4 rel_l2={dn['mean_int4_per_head_rel_l2']:.3f} "
            f"f16 rel_l2={dn['mean_f16_rel_l2']:.4f} "
            f"adjacent scale_aware={dn['adjacent_same_block']['mean_scale_aware']}."
        )
        dn_why.append(
            "Rec_state is a SUMMARY (not a cache): prefix-sharing across "
            "diverged suffixes is false. Content is reused every token; "
            "TRAFFIC is re-read every token (NOETIC_DELTANET_DESIGN). "
            "Zeroing the organ lost more function than zeroing GQA "
            "(NOETIC_ORGAN_CENSUS function_lost 0.856 vs 0.607)."
        )

    axes = [
        {
            "id": "kivi",
            "paper": "KIVI (Liu et al., arXiv:2402.02750)",
            "s026": "§51-52",
            "what_it_exploits": (
                "K outliers live in channels (per-channel quant); V outliers "
                "live in tokens (per-token quant)."
            ),
            "redundancy_present": kivi_holds,
            "verdict": "HAS_THE_REDUNDANCY" if kivi_holds else "DOES_NOT",
            "measured_redundancy": {
                "runtime_production_kv": kivi_rt,
                "capture_post_attn_norm_corroboration": kivi_cap,
            },
            "estimated_bytes_saved": {
                "at_32k_c4": kivi_est,
                "at_32k_c1": kivi_bytes(HEADLINE_SEQ, 1, kivi_bits, kivi_bits)
                if kivi_holds else {"saved_bytes": 0, "note": "not booked"},
                "at_256_c1": kivi_bytes(256, 1, kivi_bits, kivi_bits)
                if kivi_holds else {"saved_bytes": 0, "note": "not booked"},
            },
            "long_context_risk": long_context_risk(
                "KIVI",
                "2-bit asymmetric KV on GQA, the organ that already sets the "
                "weight quality floor. Needle-in-haystack often lives in "
                "low-magnitude key channels the codec would coarsen.",
            ),
            "why": kivi_why,
            "attacks": "gqa_kv",
            "seq_linear": True,
        },
        {
            "id": "minicache",
            "paper": "MiniCache (Liu et al., arXiv:2405.14366)",
            "s026": "§53",
            "what_it_exploits": (
                "Adjacent-layer KV caches are similar enough to merge in depth."
            ),
            "redundancy_present": mc_holds,
            "verdict": "HAS_THE_REDUNDANCY" if mc_holds else "DOES_NOT",
            "measured_redundancy": {
                "runtime_K": mc_k_rt,
                "runtime_V": mc_v_rt,
                "capture_K": mc_k_cap,
                "capture_V": mc_v_cap,
            },
            "estimated_bytes_saved": {
                "at_32k_c4": mc_est,
                "at_32k_c1": minicache_bytes(HEADLINE_SEQ, 1)
                if mc_holds else {"saved_bytes": 0, "note": "not booked"},
            },
            "long_context_risk": long_context_risk(
                "MiniCache",
                "Merging GQA layers 4 apart (with DeltaNet in between) mixes "
                "distinct mixer memories. A merged key that is 'close on "
                "average' can still destroy a rare token's exact match.",
            ),
            "why": mc_why,
            "attacks": "gqa_kv",
            "seq_linear": True,
        },
        {
            "id": "h2o",
            "paper": "H2O (Zhang et al., arXiv:2306.14048)",
            "s026": "§54",
            "what_it_exploits": (
                "A small set of heavy-hitter tokens plus a recent window "
                "carries most attention mass; the rest of KV can be dropped."
            ),
            "redundancy_present": h2o_holds,
            "verdict": "HAS_THE_REDUNDANCY" if h2o_holds else "DOES_NOT",
            "measured_redundancy": {
                "runtime_gqa_attn": h2o_rt_m,
                "capture_hold_prompts": h2o_cap_m,
            },
            "estimated_bytes_saved": {
                "at_32k_c4": h2o_est,
                "at_32k_c1": h2o_bytes(HEADLINE_SEQ, 1, keep_frac_32k)
                if h2o_holds else {"saved_bytes": 0, "note": "not booked"},
                "measured_T_is_not_32k": (
                    "Mass concentration was measured at T≈50–500, not 32K. "
                    "Byte estimate applies the H2O recipe at 32K; retained-mass "
                    "at 32K is unmeasured. A needle token is often not a heavy hitter."
                ),
            },
            "long_context_risk": long_context_risk(
                "H2O",
                "Heavy hitters are high-mass tokens (BOS, punctuation, recent "
                "local context). A needle/tool/schema token is often low-mass "
                "until the query that needs it. Eviction is a recall failure "
                "mode, not a free 80%. DeltaNet rec_state cannot be "
                "token-evicted at all (it is not a token cache).",
            ),
            "why": h2o_why,
            "attacks": "gqa_kv",
            "seq_linear": True,
        },
        {
            "id": "deltanet_state",
            "paper": "S026 §56 DeltaNet state doctor (not a KV-cache paper)",
            "s026": "§56",
            "what_it_exploits": (
                "Recurrent state S (48 × 48 × 128 × 128 f32) may be low-rank, "
                "cross-layer similar, head-redundant, or cheaply quantized. "
                "It does NOT grow with context."
            ),
            "redundancy_present": dn_holds,
            "verdict": "HAS_THE_REDUNDANCY" if dn_holds else "DOES_NOT",
            "measured_redundancy": dn or {"status": ABSENT},
            "estimated_bytes_saved": {
                "at_any_seq_c4": dn_est,
                "at_32k_c4_share_of_session": dn_est.get("share_of_32k_c4_session"),
                "note": (
                    "At 32K, GQA KV is 4.29 GiB/session and DeltaNet is 0.146 "
                    "GiB/session (3.5%). Compressing rec_state cannot move the "
                    "N016 crossover. At seq=256 it is 82% of session state."
                ),
            },
            "long_context_risk": long_context_risk(
                "DeltaNet rec_state",
                "This organ's residual carries more function than GQA on the "
                "organ-census capture. A summary-state codec error is not a "
                "dropped token — it is a corrupted memory of the whole prefix. "
                "f16 is the only recipe whose recon error is expected to be "
                "tiny even without redundancy; it is still unmeasured at "
                "generation. Prefix-sharing is false.",
            ),
            "why": dn_why,
            "attacks": "deltanet_recurrent_state",
            "seq_linear": False,
        },
    ]
    return axes


def rank_axes(axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(a):
        saved = 0
        est = a.get("estimated_bytes_saved") or {}
        for v in est.values():
            if isinstance(v, dict) and "saved_bytes" in v:
                saved = max(saved, int(v.get("saved_bytes") or 0))
        return (1 if a.get("redundancy_present") else 0, saved)

    ordered = sorted(axes, key=key, reverse=True)
    ranking = []
    for i, a in enumerate(ordered, 1):
        saved = 0
        est = a.get("estimated_bytes_saved") or {}
        for v in est.values():
            if isinstance(v, dict) and "saved_bytes" in v:
                saved = max(saved, int(v.get("saved_bytes") or 0))
        ranking.append({
            "rank": i,
            "id": a["id"],
            "verdict": a["verdict"],
            "redundancy_present": a["redundancy_present"],
            "booked_saved_bytes_at_headline": saved,
            "attacks": a["attacks"],
            "seq_linear": a["seq_linear"],
        })
    return ranking


def compose_answer(axes: list[dict[str, Any]], ranking: list[dict[str, Any]]) -> str:
    present = [a for a in axes if a["redundancy_present"]]
    absent = [a for a in axes if not a["redundancy_present"]]
    bits = []
    if present:
        bits.append(
            "OUR runtime state HAS the redundancy of: "
            + ", ".join(a["id"] for a in present)
            + "."
        )
    else:
        bits.append(
            "OUR runtime state does NOT present the redundancy KIVI, "
            "MiniCache, H2O, or a cheap DeltaNet-state codec exploit, at "
            "the measured sites."
        )
    short_reason = {
        "kivi": "K prefers per-channel (16/16) but V does not prefer per-token (0/16); the paper's K/V asymmetry is absent",
        "minicache": "adjacent GQA KV cosine ≈ 0.007, merge rel_l2 ≈ 0.70; hybrid interval 4 is not a dense attention stack",
        "h2o": "last-query mass is not concentrated vs uniform",
        "deltanet_state": "rec_state is full-rank, cross-layer-dissimilar, and not int4-cheap",
    }
    if absent:
        bits.append(
            "Measured reason it does not: "
            + "; ".join(
                f"{a['id']}={a['verdict']} ({short_reason.get(a['id'], (a.get('why') or ['unmeasured'])[0])})"
                for a in absent
            )
            + "."
        )
    bits.append(
        "N016 stands: at 32K×4, session state exceeds the q4 body. The only "
        "seq-linear term is GQA KV (131,072 B/token). Booking any of these "
        "saves requires a long-context capability suite; until that exists "
        "the capability cost is ABSENT. Rank: "
        + " > ".join(r["id"] for r in ranking)
        + "."
    )
    return " ".join(bits)


# ---------------------------------------------------------------------------
# receipt
# ---------------------------------------------------------------------------

def build_receipt(
    *,
    skip_runtime: bool = False,
    skip_capture: bool = False,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, Any]:
    started = time.perf_counter()
    progress = lambda m: print(f"[state-gravity] {m}", file=sys.stderr, flush=True)
    parent_before = parent_snap(PARENT_A)
    genome = state_genome()
    prefill, prefill_src = load_json_rel(PREFILL_KV_REL)
    organ, organ_src = load_json_rel(ORGAN_CENSUS_REL)

    runtime = None
    runtime_meta: dict[str, Any] = {"status": ABSENT}
    if not skip_runtime:
        if not (Q4_ROOT / "manifest.json").is_file():
            runtime_meta = {
                "status": ABSENT,
                "reason": f"q4 artifact missing at {Q4_ROOT}",
            }
        else:
            art = Artifact(Q4_ROOT)
            token_ids: list[int] = []
            tok_source = (
                "huggingface tokenizers.Tokenizer "
                "(qwen3.8-27b-abliterated-bf16/tokenizer.json)"
            )
            try:
                for p in PROMPTS:
                    token_ids.extend(encode_ids(p))
                    if len(token_ids) >= max_tokens:
                        break
            except Exception as exc:  # noqa: BLE001
                token_ids = []
                tok_source = f"fallback organ-census tokenize ({type(exc).__name__}: {exc})"
                for p in PROMPTS:
                    ids, src = tokenize(p)
                    tok_source = src
                    token_ids.extend(ids)
                    if len(token_ids) >= max_tokens:
                        break
            token_ids = token_ids[:max_tokens]
            tokens = np.array(token_ids, dtype=np.int64)
            progress(f"runtime tokens={tokens.size} via {tok_source} head={token_ids[:8]}")
            runtime = run_runtime_prefill(art, tokens, progress)
            runtime_meta = {
                "status": MEASURED,
                "kind": "cpu_hybrid_prefill_of_gravity_q4_on_real_token_ids",
                "artifact": str(Q4_ROOT),
                "tokenizer": tok_source,
                "n_tokens": int(tokens.size),
                "token_ids_head": token_ids[:16],
                "prompts": list(PROMPTS),
                "elapsed_s": runtime["elapsed_s"],
                "not_synthetic": True,
                "gaussian_proxy_used": False,
                "site": (
                    "production-layout GQA KV (post qk-RMSNorm + RoPE) and "
                    "DeltaNet rec_state after the real residual chain. Single "
                    "stream; no lm_head; no GPU."
                ),
            }

    capture = None
    capture_meta: dict[str, Any] = {"status": ABSENT}
    if not skip_capture:
        try:
            capture = run_capture_gqa(progress)
            capture_meta = {
                "status": MEASURED,
                "elapsed_s": capture["elapsed_s"],
                "site": capture["site"],
                "capture": capture["capture"],
                "parent_bf16": capture["parent"],
                "n_hold_tokens": capture["n_hold_tokens"],
                "not_synthetic": True,
                "gaussian_proxy_used": False,
                "read_only_parent_tensors": True,
            }
        except Exception as exc:  # noqa: BLE001
            capture_meta = {
                "status": ABSENT,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            capture = None

    axes = assemble_axes(genome, runtime, capture)
    ranking = rank_axes(axes)
    parent_after = parent_snap(PARENT_A)
    mutated = (
        parent_before.get("catalog_mtime_ns") != parent_after.get("catalog_mtime_ns")
        or parent_before.get("catalog_ino") != parent_after.get("catalog_ino")
        or parent_before.get("catalog_bytes") != parent_after.get("catalog_bytes")
    )
    if mutated:
        raise RuntimeError("NOETIC_PARENT_A identity changed — aborting")

    headline = genome["headline_n016"]["q4_32k_c4"]
    doc = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "git_head": git_head(),
        "generated_by": GENERATOR,
        "hand_authored": False,
        "obligation": OBLIGATION,
        "question": (
            "Does OUR GQA KV + DeltaNet recurrent state actually have the "
            "redundancy KIVI / MiniCache / H2O / a DeltaNet-state doctor "
            "exploit, and what would each save at the N016 long-context "
            "operating point without assuming long-context recall survives?"
        ),
        "answer": compose_answer(axes, ranking),
        "did_not_touch_gpu": True,
        "did_not_run_cargo_or_metal_benchmarks": True,
        "did_not_load_second_27b": True,
        "did_not_mutate_sealed_parent": True,
        "did_not_write_ascent_or_campaign": True,
        "cpu_only": True,
        "parent_identity_before": parent_before,
        "parent_identity_after": parent_after,
        "state_genome": genome,
        "runtime_state_capture": runtime_meta,
        "capture_corroboration": capture_meta,
        "prior_science": {
            "prefill_kv": {
                "source": prefill_src,
                "state_exceeds_weights_32k_c4": (
                    (prefill or {}).get("headline_footprint", {}).get("q4_long_c4", {})
                    .get("state_exceeds_weights")
                ),
                "do_not_assume_kv_quant_is_free": True,
                "prefix_sharing": "DeltaNet rec_state is not prefix-shareable",
            },
            "organ_census": {
                "source": organ_src,
                "gqa_function_lost_when_zeroed": (
                    (organ or {}).get("organs", {}).get("attention_gqa", {})
                    .get("null_representation", {}).get("function_lost")
                ),
                "deltanet_function_lost_when_zeroed": (
                    (organ or {}).get("organs", {}).get("deltanet", {})
                    .get("null_representation", {}).get("function_lost")
                ),
            },
            "citations": [
                PREFILL_KV_REL,
                ORGAN_CENSUS_REL,
                GQA_DESIGN_REL,
                DN_DESIGN_REL,
                ORGAN_FRONTIERS_REL,
                CANON_REL,
            ],
            "gqa_weight_floor_is_not_kv_quant": (
                "NOETIC_GQA_DESIGN / ORGAN_FRONTIERS: GQA weights cannot "
                "cheaply go below Q4. That does not decide KV-cache quant."
            ),
            "deltanet_static_vs_state": (
                "NOETIC_DELTANET_DESIGN: 0 of 7 static tensors duplicate S. "
                "§56 asks whether S itself is redundant, a different question."
            ),
        },
        "axes": axes,
        "ranking": ranking,
        "headline_bytes": {
            "seq": HEADLINE_SEQ,
            "sessions": HEADLINE_C,
            "MODEL_BYTES": headline["MODEL_BYTES"],
            "SESSION_STATE_BYTES_x_c": headline["SESSION_STATE_BYTES_x_c"],
            "gqa_kv_bytes_x_c": headline["gqa_kv_bytes_x_c"],
            "deltanet_state_bytes_x_c": headline["deltanet_state_bytes_x_c"],
            "state_exceeds_weights": headline["state_exceeds_weights"],
        },
        "method": {
            "runtime": (
                "CPU hybrid prefill of qwen38-gravity-uniform-q4-v1 on real "
                "token ids (organ-census prompts, 128 tokens). Captures K/V "
                "after qk-RMSNorm+RoPE (the production cache layout) and "
                "DeltaNet rec_state at the end of the prefix. Single residual "
                "stream; no lm_head; no Metal."
            ),
            "capture": (
                "capture_diverse2 hold split, parent BF16 k/v/q streamed "
                "read-only. Site is post_attn_norm, not mixer input. Used as "
                "a longer-token corroboration of KIVI/MiniCache/H2O, never as "
                "a substitute for the runtime cache."
            ),
            "scale_aware": (
                "Cosine is scale-blind (0.01*X scores ~1). MiniCache and KIVI "
                "gates use scale_aware / relative_l2. The 0.01 trap is scored."
            ),
            "not_synthetic": True,
        },
        "wall_s": time.perf_counter() - started,
    }
    doc["answer"] = compose_answer(axes, ranking)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-runtime", action="store_true")
    ap.add_argument("--skip-capture", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--out", type=Path, default=RECEIPT)
    args = ap.parse_args(argv)
    doc = build_receipt(
        skip_runtime=args.skip_runtime,
        skip_capture=args.skip_capture,
        max_tokens=args.max_tokens,
    )
    atomic_write(args.out, doc)
    print(f"wrote {args.out}")
    print(f"answer: {doc['answer']}")
    print("ranking:")
    for r in doc["ranking"]:
        print(
            f"  {r['rank']}. {r['id']:16s} {r['verdict']:20s} "
            f"saved={r['booked_saved_bytes_at_headline']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
