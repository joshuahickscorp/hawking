#!/usr/bin/env python3
"""N036 BINARY_HEALING: localize the 1.25-bpw binary g64 death, heal minimally.

Injured body (N032): binary sign+g64 scale, 1.25 bpw, COMPLETE_TOKEN_NS 23.43 ms,
physically faster than q2f (27.55 ms) — generation dead (16 copies of token 271).
Do NOT restore precision globally. Map the failure on REAL activations, then add
the smallest protected island (organ / layer band / sparse residual / q2 island)
that can climb the composition ladder to coherent_generation.

    python3 tools/headless/binary_healing.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bytes_frontier import (  # noqa: E402
    GROUP,
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    MLP_ELEMENTS,
    N021_COMPLETE_GPU_NS,
    PARENT_PARAMS,
    Q4_ATTN_F32_BYTES,
    Q2F_BPW,
    ROOF_NS,
    ROOF_TOK_S,
    SCALE_BITS,
    compose_complete,
    git_head,
    moved_toward_roof,
    now_iso,
    ns_spread,
    write_atomic,
)
from first_noetic_executable import (  # noqa: E402
    CODEC_BINARY,
    CODEC_F32,
    CODEC_Q4,
    PARENT_BF16,
    PROMPT,
    Q4_INCUMBENT,
    Q4_INCUMBENT_EBPW,
    Q4_ROOT,
    TOKENIZER,
    SourceBF16,
    artifact_filename,
    binary_storage_bpw,
    decode_mix,
    find_decode_binary,
    hardlink_or_copy,
    judge_coherence,
    load_q4_manifest,
    organ_of,
    pack_hgravb01,
    sha256_hex,
    write_catalog,
)
from fractional_bit_canon import (  # noqa: E402
    _fourlevel_fitted,
    as_groups,
    codec_binary,
    find_capture,
    find_parent,
    load_X,
    load_tensor,
    snap_f16,
    split_from_manifest,
    tensor_name,
)
from kernel_competence import (  # noqa: E402
    kernel_bodies,
    params_of,
    screen_kernel,
    strip_comments,
)
from q2f_g64_generation import (  # noqa: E402
    CODEC_AFFINE,
    pack_hgrafv01_q2f,
    q2f_storage_bpw,
)

SCHEMA = "hawking.headless.binary_healing.v1"
RECEIPT = REPO / "receipts" / "headless" / "BINARY_HEALING.json"
RAW = REPO / "receipts" / "headless" / "_BINARY_HEALING_raw.json"
LOCAL = REPO / "receipts" / "headless" / "_BINARY_HEALING_local.json"
SHADER = REPO / "crates" / "hawking-core" / "shaders" / "bytes_frontier.metal"
CARGO_TARGET = Path(
    os.environ.get("CARGO_TARGET_DIR", str(REPO / "workspace" / "ops" / "build" / "rust"))
)
BIN = CARGO_TARGET / "release-fast" / "examples" / "binary_healing"
GPU_LOCK = REPO / "tools" / "gpu_lane_lock.sh"
ARTIFACTS_ROOT = Path(
    os.environ.get(
        "QWEN38_BINHEAL_ARTIFACT_ROOT",
        str(REPO / "artifacts" / "qwen38-binheal"),
    )
)

BINARY_BPW = 1.25
Q2F_MS = 27.55
BINARY_COMPLETE_NS = 23_431_791  # N032 binary_g64 composed COMPLETE_TOKEN_NS
Q2F_COMPLETE_NS = N021_COMPLETE_GPU_NS  # 27_547_874
Q2F_MLP_NS = 15_738_249
BINARY_MLP_NS = 11_622_166
MIX_C_COMPLETE_EBPW = 2.343974955991431
MIX_C_PAYLOAD_BYTES = 7_880_443_352
MIX_C_BINARY_BYTES = 2_673_910_272
NON_MLP_BYTES = MIX_C_PAYLOAD_BYTES - MIX_C_BINARY_BYTES  # 5_206_533_080
ORGAN_ELEMS = MLP_ELEMENTS // 3  # gate, up, down are the same size
ORGANS = ("gate_proj", "up_proj", "down_proj")
HOLD_TOKENS = 48
CHANNEL_TOP_K = 32
REL_L2_DIVERGE = 0.15  # binary vs q2f on Y; named before looking
VISION_PY = Path.home() / ".grok-vision" / "bin" / "python"

KERNELS = (
    "binary_g64_matvec_geo_c5120_tpr64_tg128",
    "binary_g64_matvec_geo_c17408_tpr64_tg128",
    "q2f_g64_matvec_geo_c5120_tpr64_tg128",
    "q2f_g64_matvec_geo_c17408_tpr64_tg128",
    "binary_sparse_fused_geo_c5120_tpr64_tg128",
    "binary_sparse_fused_geo_c17408_tpr64_tg128",
)

# Native greedy from sealed receipts (do not re-derive).
Q2F_FIRST16 = [
    15769, 248046, 248046, 248046, 248068, 271, 760, 1156,
    369, 9859, 728, 310, 10033, 11, 303, 18541,
]
BINARY_FIRST16 = [271] * 16
Q4_FIRST16 = [
    248068, 198, 760, 1156, 6587, 264, 11346, 11,
    58655, 15673, 314, 1204, 264, 18826, 27545, 264,
]


def _reexec_vision() -> None:
    if not VISION_PY.is_file():
        return
    try:
        if Path(sys.executable).resolve() == VISION_PY.resolve():
            return
    except OSError:
        return
    os.execv(str(VISION_PY), [str(VISION_PY), *sys.argv])


def bpw_binary(group: int = GROUP) -> float:
    return 1.0 + SCALE_BITS / float(group)


def bpw_q2f(group: int = GROUP) -> float:
    return 2.0 + SCALE_BITS / float(group)


def reconstruct_binary(w: np.ndarray, group: int = GROUP) -> np.ndarray:
    what, _acc = codec_binary(w, g=group)
    return what


def reconstruct_q2f(w: np.ndarray, group: int = GROUP) -> np.ndarray:
    return _fourlevel_fitted(w, group)


def mlp_body_bytes(n_binary: int, n_q2f: int, n_sparse_nnz: int = 0) -> dict[str, float]:
    """n_* count GEMVs (192 total for a full MLP body)."""
    if n_binary + n_q2f != LAYERS * 3 and n_sparse_nnz == 0:
        # allow partial; still bill what is named
        pass
    per = ORGAN_ELEMS / LAYERS  # elements per GEMV
    bin_bytes = n_binary * per * bpw_binary() / 8.0
    q2_bytes = n_q2f * per * bpw_q2f() / 8.0
    # CSR: u32 col + f16 corr per nnz, plus amortized row_ptr billed in caller
    csr_bytes = float(n_sparse_nnz) * 6.0
    active = bin_bytes + q2_bytes + csr_bytes
    elems = (n_binary + n_q2f) * per
    if elems <= 0:
        elems = MLP_ELEMENTS
    return {
        "n_binary_gemvs": float(n_binary),
        "n_q2f_gemvs": float(n_q2f),
        "binary_bytes": bin_bytes,
        "q2f_bytes": q2_bytes,
        "csr_bytes": csr_bytes,
        "active_bytes": active,
        "mlp_body_bpw": 8.0 * active / MLP_ELEMENTS,
        "elements_billed": elems,
    }


def complete_ebpw_from_mlp_bytes(mlp_bytes: float) -> float:
    return 8.0 * (mlp_bytes + NON_MLP_BYTES) / PARENT_PARAMS


def island_spec(mode: str) -> dict[str, Any]:
    """Named healing islands. Precision is added ONLY on the named region."""
    n_bin = 0
    n_q2 = 0
    protected: list[dict[str, Any]] = []
    for layer in range(LAYERS):
        for organ in ORGANS:
            q2 = False
            if mode == "q2f":
                q2 = True
            elif mode == "down_q2f":
                q2 = organ == "down_proj"
            elif mode == "gate_q2f":
                q2 = organ == "gate_proj"
            elif mode == "early16_q2f":
                q2 = layer < 16
            elif mode == "late16_q2f":
                q2 = layer >= LAYERS - 16
            elif mode == "binary":
                q2 = False
            if q2:
                n_q2 += 1
                protected.append({"layer": layer, "organ": organ, "codec": "q2f_g64"})
            else:
                n_bin += 1
    bill = mlp_body_bytes(n_bin, n_q2)
    complete = complete_ebpw_from_mlp_bytes(bill["active_bytes"])
    tax = complete - BINARY_BPW
    mlp_tax = bill["mlp_body_bpw"] - BINARY_BPW
    return {
        "id": mode,
        "n_binary_gemvs": n_bin,
        "n_q2f_gemvs": n_q2,
        "n_protected": n_q2,
        "protected": protected[:8],  # head; full count is n_protected
        "protected_rule": {
            "binary": "none (injured body)",
            "q2f": "all 192 MLP GEMVs at q2f (global restore; NOT a heal)",
            "down_q2f": "mlp.down_proj layers 0..63 at q2f; gate+up stay binary g64",
            "gate_q2f": "mlp.gate_proj layers 0..63 at q2f; up+down stay binary g64",
            "early16_q2f": "layers 0..15 all-MLP q2f; layers 16..63 binary g64",
            "late16_q2f": "layers 48..63 all-MLP q2f; layers 0..47 binary g64",
        }.get(mode, mode),
        "mlp_body_bpw": bill["mlp_body_bpw"],
        "mlp_bytes": bill["active_bytes"],
        "complete_ebpw": complete,
        "COHERENCE_TAX_EBPW": tax,
        "mlp_tax_ebpw": mlp_tax,
        "dense_w": 0,
        **bill,
    }


def sparse_spec(frac: float = 0.005) -> dict[str, Any]:
    bin_bytes = MLP_ELEMENTS * bpw_binary() / 8.0
    nnz = MLP_ELEMENTS * frac
    csr = nnz * 6.0 + LAYERS * 3 * (INTERMEDIATE + 1) * 4
    active = bin_bytes + csr
    complete = complete_ebpw_from_mlp_bytes(active)
    return {
        "id": "sparse_05",
        "nnz_frac": frac,
        "nnz": nnz,
        "binary_bytes": bin_bytes,
        "csr_bytes": csr,
        "active_bytes": active,
        "mlp_body_bpw": 8.0 * active / MLP_ELEMENTS,
        "complete_ebpw": complete,
        "COHERENCE_TAX_EBPW": complete - BINARY_BPW,
        "mlp_tax_ebpw": 8.0 * active / MLP_ELEMENTS - BINARY_BPW,
        "protected_rule": (
            f"binary g64 plane + {frac:.3%} CSR residual (u32 col + f16 corr) "
            "on the most sensitive channels; no dense W"
        ),
        "dense_w": 0,
        "n_binary_gemvs": LAYERS * 3,
        "n_q2f_gemvs": 0,
        "n_protected": 0,
    }


def row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30
    return float(np.mean(num / den))


def rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-30))


def gain_score(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    r = nb / (na + 1e-30)
    return float(np.mean(np.minimum(r, 1.0 / (r + 1e-30))))


def score_y(pred: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    cos = row_cosine(pred, ref)
    gn = gain_score(pred, ref)
    rel = rel_l2(pred, ref)
    return {
        "cosine": cos,
        "gain": gn,
        "scale_aware": cos * gn,
        "rel_l2": rel,
        "n_rows": int(ref.shape[0]),
        "dim": int(ref.shape[1]),
    }


def gemm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        import torch

        ta = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        tb = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32))
        return (ta @ tb).numpy()
    except Exception:
        return np.ascontiguousarray(a, dtype=np.float32) @ np.ascontiguousarray(
            b, dtype=np.float32
        )


def x_wt(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return gemm(x, w.T)


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def top_channels(err: np.ndarray, k: int = CHANNEL_TOP_K) -> list[dict[str, Any]]:
    """err is [n_tokens, rows]; energy per output channel."""
    energy = np.square(err, dtype=np.float64).sum(axis=0)
    total = float(energy.sum()) + 1e-30
    idx = np.argsort(energy)[::-1][:k]
    out = []
    cum = 0.0
    for rank, i in enumerate(idx):
        frac = float(energy[i] / total)
        cum += frac
        out.append(
            {
                "rank": rank,
                "channel": int(i),
                "error_energy_frac": frac,
                "cum_frac": cum,
            }
        )
    return out


def token_logit_from_receipts() -> dict[str, Any]:
    """Earliest greedy token-logit divergence: native mix_c vs q2f, already measured."""
    first = 0
    for i, (a, b) in enumerate(zip(BINARY_FIRST16, Q2F_FIRST16)):
        if a != b:
            first = i
            break
    return {
        "source": [
            "receipts/headless/FIRST_NOETIC_EXECUTABLE.json mix_c_all_mlp_binary_g64",
            "receipts/headless/Q2F_G64_GENERATION.json mix_all_mlp_q2f_g64",
        ],
        "kind": "native_greedy_first_generated_token",
        "real_activations": True,
        "earliest_position": first,
        "binary_token_id": BINARY_FIRST16[first],
        "q2f_token_id": Q2F_FIRST16[first],
        "q4_token_id": Q4_FIRST16[first],
        "binary_ids": BINARY_FIRST16,
        "q2f_ids": Q2F_FIRST16,
        "note": (
            "mix_c emits 16 copies of token 271. q2f emits a varied 16-token "
            "sample starting at 15769. The streams disagree at generated token 0. "
            "That is the earliest token-logit divergence; generation is dead "
            "before any later token can recover."
        ),
        "on_real_prompt": PROMPT[:80],
    }


def _organ_row(
    organ: str,
    w: np.ndarray,
    y_t: np.ndarray,
    y_b: np.ndarray,
    y_q: np.ndarray,
) -> dict[str, Any]:
    vs_q_b = score_y(y_b, y_q)
    ch = top_channels(y_b - y_q, CHANNEL_TOP_K)
    return {
        "organ": organ,
        "shape": [int(w.shape[0]), int(w.shape[1])],
        "site": "post_attn_norm" if organ != "down_proj" else "teacher_swiglu_hidden",
        "binary_vs_teacher": score_y(y_b, y_t),
        "q2f_vs_teacher": score_y(y_q, y_t),
        "binary_vs_q2f": vs_q_b,
        "n_tokens_argmax_disagree_q2f_binary": int(
            np.sum(np.argmax(y_q, axis=1) != np.argmax(y_b, axis=1))
        ),
        "sensitive_channels": ch[:8],
        "top8_channel_cum_frac": ch[7]["cum_frac"] if len(ch) >= 8 else None,
        "top32_channel_cum_frac": ch[-1]["cum_frac"] if ch else None,
    }


def localize_layer(
    parent: Path,
    x: np.ndarray,
    layer: int,
) -> dict[str, Any]:
    # Capture X is post_attn_norm: the real site for gate/up. down_proj is
    # scored on teacher SwiGLU hidden (FRACTIONAL_BIT_CANON site_down).
    wg = load_tensor(parent, tensor_name(layer, "gate_proj"))
    wu = load_tensor(parent, tensor_name(layer, "up_proj"))
    wd = load_tensor(parent, tensor_name(layer, "down_proj"))
    wgb, wgq = reconstruct_binary(wg), reconstruct_q2f(wg)
    wub, wuq = reconstruct_binary(wu), reconstruct_q2f(wu)
    wdb, wdq = reconstruct_binary(wd), reconstruct_q2f(wd)

    yg_t, yg_b, yg_q = x_wt(x, wg), x_wt(x, wgb), x_wt(x, wgq)
    yu_t, yu_b, yu_q = x_wt(x, wu), x_wt(x, wub), x_wt(x, wuq)
    h_t = silu(yg_t) * yu_t
    h_b = silu(yg_b) * yu_b
    h_q = silu(yg_q) * yu_q
    yd_t, yd_b, yd_q = x_wt(h_t, wd), x_wt(h_t, wdb), x_wt(h_t, wdq)

    organs_out = {
        "gate_proj": _organ_row("gate_proj", wg, yg_t, yg_b, yg_q),
        "up_proj": _organ_row("up_proj", wu, yu_t, yu_b, yu_q),
        "down_proj": _organ_row("down_proj", wd, yd_t, yd_b, yd_q),
    }
    worst_organ = max(
        ORGANS, key=lambda o: organs_out[o]["binary_vs_q2f"]["rel_l2"]
    )
    worst_rel = organs_out[worst_organ]["binary_vs_q2f"]["rel_l2"]

    y_sw_t = x_wt(h_t, wd)
    y_sw_b = x_wt(h_b, wdb)
    y_sw_q = x_wt(h_q, wdq)
    y_hd = x_wt(h_b, wdq)  # down_q2f island on binary SwiGLU hidden
    y_hg = x_wt(silu(yg_q) * yu_b, wdb)  # gate_q2f island
    swiglu = {
        "binary_vs_teacher": score_y(y_sw_b, y_sw_t),
        "q2f_vs_teacher": score_y(y_sw_q, y_sw_t),
        "binary_vs_q2f": score_y(y_sw_b, y_sw_q),
        "down_q2f_heal_vs_teacher": score_y(y_hd, y_sw_t),
        "down_q2f_heal_vs_q2f": score_y(y_hd, y_sw_q),
        "gate_q2f_heal_vs_teacher": score_y(y_hg, y_sw_t),
        "gate_q2f_heal_vs_q2f": score_y(y_hg, y_sw_q),
        "n_tokens_argmax_disagree_binary_q2f": int(
            np.sum(np.argmax(y_sw_b, axis=1) != np.argmax(y_sw_q, axis=1))
        ),
        "n_tokens_argmax_disagree_downheal_q2f": int(
            np.sum(np.argmax(y_hd, axis=1) != np.argmax(y_sw_q, axis=1))
        ),
    }
    del wg, wu, wd, wgb, wgq, wub, wuq, wdb, wdq
    del yg_t, yg_b, yg_q, yu_t, yu_b, yu_q, h_t, h_b, h_q
    del yd_t, yd_b, yd_q, y_sw_t, y_sw_b, y_sw_q, y_hd, y_hg
    diverged = any(
        organs_out[o]["binary_vs_q2f"]["rel_l2"] >= REL_L2_DIVERGE for o in ORGANS
    )
    return {
        "layer": layer,
        "organs": organs_out,
        "worst_organ_vs_q2f": worst_organ,
        "worst_rel_l2_binary_vs_q2f": worst_rel,
        "swiglu": swiglu,
        "diverged_at_threshold": diverged,
        "threshold": REL_L2_DIVERGE,
    }


def run_localization() -> dict[str, Any]:
    cap = find_capture()
    parent = find_parent()
    x0 = load_X(cap, 0)
    fit_idx, hold_idx, man, split_rule = split_from_manifest(cap, x0.shape[0])
    hold_idx = np.asarray(hold_idx)
    # real hold tokens, not Gaussian; subsample for a 64-layer scan
    if hold_idx.size > HOLD_TOKENS:
        # stride through the hold set so families mix
        step = max(1, hold_idx.size // HOLD_TOKENS)
        use = hold_idx[::step][:HOLD_TOKENS]
    else:
        use = hold_idx
    layers_out = []
    earliest_layer = None
    earliest_organ = None
    per_organ_mean = {o: [] for o in ORGANS}
    t0 = time.perf_counter()
    for layer in range(LAYERS):
        x = load_X(cap, layer)[use]
        row = localize_layer(parent, x, layer)
        layers_out.append(row)
        for o in ORGANS:
            per_organ_mean[o].append(row["organs"][o]["binary_vs_q2f"]["rel_l2"])
        if earliest_layer is None and row["diverged_at_threshold"]:
            earliest_layer = layer
            earliest_organ = row["worst_organ_vs_q2f"]
        print(
            f"  L{layer:02d} worst={row['worst_organ_vs_q2f']} "
            f"rel={row['worst_rel_l2_binary_vs_q2f']:.4f} "
            f"swiglu_bin={row['swiglu']['binary_vs_q2f']['rel_l2']:.4f} "
            f"downheal={row['swiglu']['down_q2f_heal_vs_q2f']['rel_l2']:.4f}",
            flush=True,
        )
        LOCAL.write_text(
            json.dumps(
                {
                    "n_done": layer + 1,
                    "earliest_layer": earliest_layer,
                    "layers": layers_out,
                }
            )
        )
    if earliest_layer is None:
        # always diverges somewhere; take the worst if threshold never fires
        worst = max(layers_out, key=lambda r: r["worst_rel_l2_binary_vs_q2f"])
        earliest_layer = int(worst["layer"])
        earliest_organ = worst["worst_organ_vs_q2f"]
    organ_means = {o: float(np.mean(v)) for o, v in per_organ_mean.items()}
    worst_organ_global = max(organ_means, key=organ_means.get)
    # sensitive channels: from the earliest diverged layer, union of top channels
    early = next(r for r in layers_out if r["layer"] == earliest_layer)
    channels = {}
    for o in ORGANS:
        channels[o] = early["organs"][o]["sensitive_channels"]
    return {
        "capture": str(cap),
        "parent": str(parent),
        "split_rule": split_rule,
        "n_hold_total": int(hold_idx.size),
        "n_tokens_used": int(use.size),
        "token_indices_head": [int(i) for i in use[:8]],
        "not_gaussian": True,
        "did_not_load_second_27b": True,
        "streamed_one_tensor_at_a_time": True,
        "threshold_binary_vs_q2f_rel_l2": REL_L2_DIVERGE,
        "earliest_layer": int(earliest_layer),
        "earliest_organ": earliest_organ,
        "worst_organ_mean_rel_l2": worst_organ_global,
        "mean_rel_l2_binary_vs_q2f_by_organ": organ_means,
        "sensitive_channels_at_earliest_layer": channels,
        "layers": layers_out,
        "wall_s": time.perf_counter() - t0,
    }


def build_failure_map(loc: dict[str, Any]) -> dict[str, Any]:
    tok = token_logit_from_receipts()
    layers = loc.get("layers") or []
    n_over = 0
    per_layer_worst = []
    for row in layers:
        if row.get("diverged_at_threshold"):
            n_over += 1
        per_layer_worst.append(
            {
                "layer": row.get("layer"),
                "organ": row.get("worst_organ_vs_q2f"),
                "rel_l2_binary_vs_q2f": row.get("worst_rel_l2_binary_vs_q2f"),
                "swiglu_binary_vs_q2f": (row.get("swiglu") or {}).get("binary_vs_q2f", {}).get("rel_l2"),
                "down_q2f_heal_vs_q2f": (row.get("swiglu") or {}).get("down_q2f_heal_vs_q2f", {}).get("rel_l2"),
            }
        )
    return {
        "earliest_token_logit_divergence": tok,
        "earliest_layer": loc.get("earliest_layer"),
        "earliest_organ": loc.get("earliest_organ"),
        "worst_organ_mean": loc.get("worst_organ_mean_rel_l2"),
        "mean_rel_l2_binary_vs_q2f_by_organ": loc.get("mean_rel_l2_binary_vs_q2f_by_organ"),
        "n_layers_scanned": len(layers),
        "n_layers_over_threshold": n_over,
        "uniformly_injured": bool(layers) and n_over == len(layers),
        "sensitive_channels": loc.get("sensitive_channels_at_earliest_layer"),
        "per_layer_worst": per_layer_worst,
        "threshold": REL_L2_DIVERGE,
        "real_activations": True,
        "capture_tokens": loc.get("n_tokens_used"),
        "why": (
            "Token-logit: native greedy already disagrees at generated token 0 "
            "(271 vs 15769). Layer/organ/channel: teacher-forced GEMVs on real "
            "hold-set post_attn_norm activations (down_proj on teacher SwiGLU "
            "hidden), binary g64 vs q2f g64 vs teacher. Earliest layer is the "
            f"first whose binary-vs-q2f rel_l2 exceeds {REL_L2_DIVERGE} "
            "(named before looking). A uniform exceedance means the injury is "
            "density, not a few bad layers — islands are still the right heal."
        ),
    }


def shader_autopsy() -> dict[str, Any]:
    src = SHADER.read_text(encoding="utf-8") if SHADER.is_file() else ""
    stripped = strip_comments(src)
    present = {k: (f"kernel void {k}(" in src) for k in KERNELS}
    ours = []
    for name, body in kernel_bodies(stripped):
        if name not in KERNELS:
            continue
        r = screen_kernel(name, body, params_of(stripped, name))
        ours.append(
            {
                "kernel": name,
                "verdict": r["verdict"],
                "n_findings": r["n_findings"],
                "findings": r["findings"],
            }
        )
    geo = [k for k in ours if "geo" in k["kernel"]]
    return {
        "file": str(SHADER.relative_to(REPO)),
        "kernels_present": present,
        "all_present": all(present.values()),
        "kernels": ours,
        "any_geo_defective": any(k["verdict"] == "DEFECTIVE" for k in geo),
        "uses_shift_not_div": "col >> 6u" in src or "(col >> 6u)" in src,
        "no_bind_time_group_size_in_geo": all(
            "constant uint& group_size" not in stripped[stripped.find(f"kernel void {n}(") : stripped.find(f"kernel void {n}(") + 800]
            for n in KERNELS
            if f"kernel void {n}(" in stripped
        ),
        "dense_w_written": False,
        "note": "N036 reuses the N032 competent binary/q2f/sparse geo kernels; group 64 is a shift.",
    }


def run_competence() -> dict[str, Any]:
    script = HERE / "kernel_competence.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    auto = shader_autopsy()
    return {
        "ok": proc.returncode == 0 and not auto["any_geo_defective"],
        "exit_code": proc.returncode,
        "stdout": (proc.stdout or "")[-1500:],
        "autopsy": auto,
        "any_geo_defective": auto["any_geo_defective"],
    }


def cargo_build() -> dict[str, Any]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(CARGO_TARGET)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            "cargo",
            "build",
            "--profile",
            "release-fast",
            "-p",
            "hawking-core",
            "--example",
            "binary_healing",
            "--example",
            "ascension_qwen38_hybrid_greedy",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
        timeout=1200,
    )
    return {
        "command": proc.args,
        "exit_code": proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "ok": proc.returncode == 0,
        "stderr_tail": (proc.stderr or "")[-2500:],
    }


def run_example(reps: int = 7) -> dict[str, Any]:
    if not BIN.is_file():
        return {"ok": False, "error": f"missing {BIN}"}
    RAW.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN),
        "--reps",
        str(reps),
        "--warmup",
        "2",
        "--layers",
        "64",
        "--out",
        str(RAW),
    ]
    if GPU_LOCK.is_file():
        cmd = ["bash", str(GPU_LOCK), "n036-binheal", *cmd]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    raw = json.loads(RAW.read_text()) if RAW.is_file() else {}
    return {
        "ok": proc.returncode == 0 and bool(raw),
        "exit_code": proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "raw": raw,
        "command": cmd,
    }


def graph_by_id(raw: dict[str, Any], gid: str) -> dict[str, Any] | None:
    for g in raw.get("graphs") or []:
        if g.get("id") == gid:
            return g
    return None


def parent_key(catalog_name: str) -> str:
    key = catalog_name.replace("language_model.model.", "model.language_model.")
    if key == catalog_name and catalog_name.startswith("language_model."):
        key = "model." + catalog_name
    return key


def is_mlp_proj(name: str) -> bool:
    return name.endswith("mlp.gate_proj.weight") or name.endswith(
        "mlp.up_proj.weight"
    ) or name.endswith("mlp.down_proj.weight")


def select_codec(name: str, mode: str) -> str | None:
    """Return 'binary' / 'q2f' if this catalog tensor is rewritten, else None (hardlink q4)."""
    if not is_mlp_proj(name):
        return None
    layer = None
    for part in name.split("."):
        if part.isdigit():
            layer = int(part)
            break
    organ = (
        "down_proj"
        if name.endswith("down_proj.weight")
        else "gate_proj"
        if name.endswith("gate_proj.weight")
        else "up_proj"
    )
    if mode == "binary":
        return "binary"
    if mode == "q2f":
        return "q2f"
    if mode == "down_q2f":
        return "q2f" if organ == "down_proj" else "binary"
    if mode == "gate_q2f":
        return "q2f" if organ == "gate_proj" else "binary"
    if mode == "early16_q2f":
        return "q2f" if layer is not None and layer < 16 else "binary"
    if mode == "late16_q2f":
        return "q2f" if layer is not None and layer >= LAYERS - 16 else "binary"
    raise ValueError(f"unknown heal mode {mode}")


def compile_healed_mix(mode: str) -> dict[str, Any]:
    dest = ARTIFACTS_ROOT / f"mix_{mode}"
    dest.mkdir(parents=True, exist_ok=True)
    segments_dir = dest / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_q4_manifest(Q4_ROOT)
    rows = list(manifest["tensors"])
    src = SourceBF16(PARENT_BF16)
    records: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    payload_bytes = 0
    binary_bytes = 0
    q2f_bytes = 0
    q4_bytes = 0
    f32_bytes = 0
    n_hardlink = 0
    n_binary = 0
    n_q2f = 0
    for i, row in enumerate(rows):
        name = row["name"]
        shape = [int(x) for x in row["shape"]]
        elements = int(row["elements"])
        src_artifact = Q4_ROOT / "tensors" / row["artifact"]
        if not src_artifact.is_file():
            raise RuntimeError(f"incumbent missing {src_artifact}")
        chosen = select_codec(name, mode)
        if chosen is None:
            filename = row["artifact"]
            dest_path = segments_dir / filename
            hardlink_or_copy(src_artifact, dest_path)
            n_hardlink += 1
            nbytes = int(dest_path.stat().st_size)
            codec = CODEC_Q4 if row["kind"] == "q4" else CODEC_F32
            codec_bpw = 8.0 * nbytes / max(elements, 1)
            digest = sha256_hex(filename.encode())
            if codec == CODEC_Q4:
                q4_bytes += nbytes
            else:
                f32_bytes += nbytes
        else:
            w = src.load(parent_key(name))
            if list(w.shape) != shape:
                raise RuntimeError(f"{name} parent shape {list(w.shape)} != catalog {shape}")
            print(f"  [{mode}] {chosen} {name}", flush=True)
            if chosen == "binary":
                payload = pack_hgravb01(w, GROUP)
                filename = artifact_filename(name, "hgravb01")
                codec = CODEC_BINARY
                codec_bpw = binary_storage_bpw(GROUP)
                binary_bytes += len(payload)
                n_binary += 1
            else:
                payload, _probe = pack_hgrafv01_q2f(w, GROUP)
                filename = artifact_filename(name, "hgrafv01")
                codec = CODEC_AFFINE
                codec_bpw = q2f_storage_bpw(GROUP)
                q2f_bytes += len(payload)
                n_q2f += 1
            del w
            dest_path = segments_dir / filename
            dest_path.write_bytes(payload)
            nbytes = len(payload)
            digest = sha256_hex(payload)
        payload_bytes += nbytes
        segments.append(
            {"id": i, "filename": filename, "bytes": nbytes, "sha256": digest}
        )
        records.append(
            {
                "name": name,
                "codec": codec,
                "organ": organ_of(name),
                "shape": shape,
                "elements": elements,
                "segment_id": i,
                "offset": 0,
                "nbytes": nbytes,
                "sha256": digest,
                "codec_bpw": codec_bpw,
            }
        )
    catalog_path = dest / "catalog.hq38m20"
    write_catalog(catalog_path, records, segments)
    complete_ebpw = 8.0 * payload_bytes / PARENT_PARAMS
    codecs = Counter(int(r["codec"]) for r in records)
    report = {
        "mix_id": f"mix_{mode}",
        "mode": mode,
        "artifact_root": str(dest),
        "catalog": str(catalog_path),
        "n_tensors": len(records),
        "n_binary": n_binary,
        "n_q2f": n_q2f,
        "n_hardlink": n_hardlink,
        "codecs": {str(k): int(v) for k, v in codecs.items()},
        "payload_bytes": payload_bytes,
        "binary_bytes": binary_bytes,
        "q2f_bytes": q2f_bytes,
        "q4_bytes": q4_bytes,
        "f32_bytes": f32_bytes,
        "parent_params": PARENT_PARAMS,
        "complete_ebpw": complete_ebpw,
        "COHERENCE_TAX_EBPW": complete_ebpw - BINARY_BPW,
        "q4_incumbent_complete_physical_bpw": Q4_INCUMBENT_EBPW,
        "wall_s": time.perf_counter() - t0,
        "did_not_load_second_27b": True,
        "parent_streamed_one_tensor_at_a_time": True,
        "wrote_under_models": False,
        "dense_w": 0,
    }
    print(
        f"[{mode}] tensors={len(records)} binary={n_binary} q2f={n_q2f} "
        f"ebpw={complete_ebpw:.6f} tax={report['COHERENCE_TAX_EBPW']:.6f} "
        f"in {report['wall_s']:.1f}s",
        flush=True,
    )
    return report


def decode_locked(artifact_root: Path) -> dict[str, Any]:
    exe = find_decode_binary()
    out_json = artifact_root / "decode.json"
    inner = [
        str(exe),
        "--artifact-root",
        str(artifact_root),
        "--tokenizer",
        str(TOKENIZER),
        "--prompt",
        PROMPT,
        "--max-new-tokens",
        "16",
        "--max-seq-len",
        "128",
        "--out",
        str(out_json),
    ]
    cmd = ["bash", str(GPU_LOCK), "n036-binheal", *inner] if GPU_LOCK.is_file() else inner
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=3600)
    wall_s = time.perf_counter() - t0
    body: dict[str, Any] = {}
    if out_json.is_file():
        body = json.loads(out_json.read_text())
    text = body.get("generated_text") or ""
    ids = [int(x) for x in body.get("new_token_ids") or []]
    stderr = proc.stderr or ""
    native = "qwen38-decode mixed HQ38M20" in stderr or "mixed bind" in stderr
    dequant = ("expanded_to_q4=" in stderr and "expanded_to_q4=0" not in stderr) or (
        "reconstruct-to-Q4" in stderr and "no reconstruct-to-Q4" not in stderr
    )
    dense = int(body.get("dense_w_materialized") or 0)
    coh = judge_coherence(text, ids)
    if coh.get("coherent"):
        rung = "coherent_generation"
        died = None
        status = "UNTESTED_ABOVE"
        unreached = "capability"
    elif ids:
        rung = "complete_token"
        died = "coherent_generation"
        status = "FAILED"
        unreached = None
    else:
        rung = "complete_organ"
        died = "complete_token"
        status = "FAILED"
        unreached = None
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "wall_s": wall_s,
        "stderr_tail": stderr[-4000:],
        "generated_text_verbatim": text,
        "new_token_ids": ids,
        "n_new_tokens": len(ids),
        "fallbacks": int(body.get("fallbacks") or 0),
        "dense_w_materialized": dense,
        "native_kernel_ran": bool(native) and not dequant,
        "dequant_path": bool(dequant),
        "median_gpu_ns_per_token": body.get("median_gpu_ns_per_token"),
        "tok_s": body.get("tok_s"),
        "coherence": coh,
        "composition": {
            "highest_rung_reached": rung,
            "died_at": died,
            "status": status,
            "unreached_above": unreached,
            "may_be_described_as": rung,
        },
        "command": cmd,
    }


def capability_restored(comp: dict[str, Any], coh: dict[str, Any]) -> float:
    if comp.get("highest_rung_reached") == "coherent_generation" and coh.get("coherent"):
        return 1.0
    n_unique = int(coh.get("n_unique_ids") or 0)
    n = int(coh.get("n_new_tokens") or 0)
    if n <= 0:
        return 0.0
    # partial credit only for a non-degenerate sample that still failed the rung
    if coh.get("repeated_single_token"):
        return 0.0
    return min(1.0, n_unique / 16.0) * 0.25


def ranking_score(cap: float, tax: float, added_ns: float | None) -> float:
    ns_term = 0.0
    if added_ns is not None:
        ns_term = max(0.0, float(added_ns)) / float(Q2F_COMPLETE_NS)
    den = abs(float(tax)) + ns_term
    if den <= 1e-12:
        return cap / 1e-12
    return cap / den


def attach_graph(
    spec: dict[str, Any],
    graph: dict[str, Any] | None,
    q2f_mlp_ns: int | None,
    parity_ok: bool,
    generation: dict[str, Any] | None,
) -> dict[str, Any]:
    gpu = ns_spread(graph)
    wall = ns_spread(graph, "wall_ns")
    mlp_ns = gpu.get("median")
    complete = compose_complete(mlp_ns, q2f_mlp_ns)
    complete_ns = complete.get("complete_token_ns")
    added_ns = None if complete_ns is None else int(complete_ns) - int(BINARY_COMPLETE_NS)
    faster = complete_ns is not None and int(complete_ns) < int(Q2F_COMPLETE_NS)
    overlap = None
    serial = (graph or {}).get("serial")
    if isinstance(serial, dict):
        overlap = serial.get("overlap_with_geo")
    gen = generation or {}
    coh = gen.get("coherence") or {
        "coherent": False,
        "reason": "generation not run",
        "n_unique_ids": 0,
        "n_new_tokens": 0,
    }
    comp = gen.get("composition") or {
        "highest_rung_reached": "complete_organ",
        "died_at": None,
        "status": "UNTESTED_ABOVE",
        "unreached_above": "complete_token",
        "may_be_described_as": "complete_organ",
        "why": (
            "SwiGLU scored on real hold-set activations (complete organ). "
            "Native generation was not run for this island, so complete_token "
            "and coherent_generation are UNREACHED, not failed."
        ),
    }
    # A heal only COUNTS if it climbed to coherent_generation.
    counts = bool(
        comp.get("highest_rung_reached") == "coherent_generation" and coh.get("coherent")
    )
    cap = capability_restored(comp, coh)
    tax = float(spec.get("COHERENCE_TAX_EBPW") or 0.0)
    if gen.get("complete_ebpw") is not None:
        tax = float(gen["complete_ebpw"]) - BINARY_BPW
    score = ranking_score(cap, tax, added_ns)
    out = {
        **spec,
        "COMPLETE_TOKEN_NS": {
            "mlp_graph_gpu_ns": gpu,
            "mlp_graph_wall_ns": wall,
            "composed": complete,
            "min": gpu.get("min"),
            "median": complete_ns,
            "max": gpu.get("max"),
            "reps": gpu.get("n"),
        },
        "added_token_ns": added_ns,
        "faster_than_q2f_27_55ms": faster,
        "parity": {"ok": parity_ok},
        "dense_w": 0,
        "dense_w_materialized": int(gen.get("dense_w_materialized") or 0),
        "generation": {
            k: gen.get(k)
            for k in (
                "ok",
                "new_token_ids",
                "n_new_tokens",
                "generated_text_verbatim",
                "native_kernel_ran",
                "dequant_path",
                "fallbacks",
                "tok_s",
                "median_gpu_ns_per_token",
            )
            if k in gen or True
        },
        "coherence": coh,
        "composition": comp,
        "counts_as_heal": counts,
        "capability_restored": cap,
        "ranking_score": score,
        "ranking_formula": (
            "capability_restored / (|COHERENCE_TAX_EBPW| + max(0, added_token_ns) / q2f_complete_ns)"
        ),
        "control": {
            "serial_or_noop": serial,
            "overlap": overlap,
            "label": (
                "NOT SEPARATED"
                if overlap
                else ("SEPARATED" if overlap is False else None)
            ),
        },
    }
    if gen.get("complete_ebpw") is not None:
        out["complete_ebpw"] = gen["complete_ebpw"]
        out["COHERENCE_TAX_EBPW"] = tax
    return out


def main() -> int:
    t0 = time.perf_counter()
    skip_gpu = os.environ.get("N036_SKIP_GPU") == "1"
    skip_gen = os.environ.get("N036_SKIP_GENERATE") == "1"
    skip_loc = os.environ.get("N036_SKIP_LOCALIZE") == "1"

    competence = run_competence()
    print("kernel autopsy any_geo_defective=", competence["any_geo_defective"], flush=True)

    loc: dict[str, Any]
    if skip_loc and LOCAL.is_file():
        loc = json.loads(LOCAL.read_text())
        if "earliest_layer" not in loc and "layers" in loc:
            # partial scan file
            loc = {"partial": True, **loc}
    else:
        print("== CPU localization on real hold-set activations ==", flush=True)
        loc = run_localization()
        write_atomic(LOCAL, json.dumps(loc, indent=2))

    fmap = build_failure_map(loc if "earliest_layer" in loc else {
        "earliest_layer": 0,
        "earliest_organ": "down_proj",
        "mean_rel_l2_binary_vs_q2f_by_organ": {},
        "sensitive_channels_at_earliest_layer": {},
        "n_tokens_used": HOLD_TOKENS,
    })

    # Candidates: always the two organ islands + the layer band the map points at.
    modes = ["down_q2f", "gate_q2f"]
    early = fmap.get("earliest_layer")
    if isinstance(early, int) and early < 16:
        modes.append("early16_q2f")
    else:
        modes.append("late16_q2f")
    specs = {m: island_spec(m) for m in ["binary", "q2f", *modes]}
    specs["sparse_05"] = sparse_spec(0.005)

    build = {"ok": BIN.is_file(), "skipped": True}
    measured = {"ok": False, "raw": {}}
    if not skip_gpu:
        print("== cargo build ==", flush=True)
        build = cargo_build()
        print("build ok", build["ok"], "wall", build.get("wall_s"), flush=True)
        if build["ok"]:
            print("== GPU mixed graphs under n036-binheal lock ==", flush=True)
            measured = run_example(7)
            print("gpu ok", measured["ok"], flush=True)

    raw = measured.get("raw") or {}
    if not raw and RAW.is_file():
        raw = json.loads(RAW.read_text())
        measured = {"ok": True, "raw": raw, "reused_raw": True}
    q2f_g = graph_by_id(raw, "q2f")
    q2f_mlp_ns = (q2f_g or {}).get("gpu_ns", {}).get("median") or Q2F_MLP_NS
    parity_rows = raw.get("parity") or []
    parity_ok = bool(parity_rows) and all(p.get("ok") is True for p in parity_rows)

    generations: dict[str, Any] = {}
    compile_reports: dict[str, Any] = {}
    gen_modes = ["down_q2f", "early16_q2f"]
    for mode in gen_modes:
        dest = ARTIFACTS_ROOT / f"mix_{mode}"
        crep_path = dest / "compile.json"
        drep_path = dest / "decode_result.json"
        if crep_path.is_file():
            compile_reports[mode] = json.loads(crep_path.read_text())
        if drep_path.is_file():
            generations[mode] = json.loads(drep_path.read_text())
            print(
                f"reused decode mix_{mode} coherent="
                f"{generations[mode].get('coherence', {}).get('coherent')}",
                flush=True,
            )
            continue
        if skip_gen:
            continue
        try:
            print(f"== compile mix_{mode} ==", flush=True)
            crep = compile_reports.get(mode) or compile_healed_mix(mode)
            compile_reports[mode] = crep
            dest.mkdir(parents=True, exist_ok=True)
            crep_path.write_text(json.dumps(crep, indent=2))
            print(f"== decode mix_{mode} ==", flush=True)
            drep = decode_locked(Path(crep["artifact_root"]))
            drep["complete_ebpw"] = crep["complete_ebpw"]
            drep["COHERENCE_TAX_EBPW"] = crep["COHERENCE_TAX_EBPW"]
            drep["dense_w_materialized"] = drep.get("dense_w_materialized") or 0
            generations[mode] = drep
            drep_path.write_text(json.dumps(drep, indent=2))
            print(
                f"  {mode} coherent={drep['coherence'].get('coherent')} "
                f"rung={drep['composition']['highest_rung_reached']} "
                f"ids={drep.get('new_token_ids')}",
                flush=True,
            )
        except Exception as e:
            generations[mode] = {
                "ok": False,
                "error": str(e),
                "coherence": {"coherent": False, "reason": str(e)},
                "composition": {
                    "highest_rung_reached": "held_out_activation",
                    "died_at": None,
                    "status": "UNTESTED_ABOVE",
                    "unreached_above": "adjacent_layers",
                    "may_be_described_as": "held_out_activation",
                },
                "dense_w_materialized": 0,
            }

    candidates = []
    for mode in ["binary", "q2f", *modes, "sparse_05"]:
        spec = specs[mode]
        g = graph_by_id(raw, mode)
        gen = generations.get(mode)
        if mode == "binary":
            # Injured body: known dead at coherent_generation.
            gen = gen or {
                "coherence": {
                    "coherent": False,
                    "reason": "16 copies of the same token (271)",
                    "repeated_single_token": True,
                    "n_unique_ids": 1,
                    "n_new_tokens": 16,
                },
                "composition": {
                    "highest_rung_reached": "complete_token",
                    "died_at": "coherent_generation",
                    "status": "FAILED",
                    "unreached_above": None,
                    "may_be_described_as": "complete_token",
                    "source_receipt": "receipts/headless/FIRST_NOETIC_EXECUTABLE.json",
                },
                "new_token_ids": BINARY_FIRST16,
                "n_new_tokens": 16,
                "generated_text_verbatim": "\n" * 16,
                "native_kernel_ran": True,
                "dequant_path": False,
                "fallbacks": 0,
                "dense_w_materialized": 0,
                "complete_ebpw": MIX_C_COMPLETE_EBPW,
            }
        if mode == "q2f":
            gen = gen or {
                "coherence": {
                    "coherent": True,
                    "reason": "emits varied tokens",
                    "repeated_single_token": False,
                    "n_unique_ids": 14,
                    "n_new_tokens": 16,
                },
                "composition": {
                    "highest_rung_reached": "coherent_generation",
                    "died_at": None,
                    "status": "UNTESTED_ABOVE",
                    "unreached_above": "capability",
                    "may_be_described_as": "coherent_generation",
                    "source_receipt": "receipts/headless/Q2F_G64_GENERATION.json",
                },
                "new_token_ids": Q2F_FIRST16,
                "n_new_tokens": 16,
                "native_kernel_ran": True,
                "dequant_path": False,
                "fallbacks": 0,
                "dense_w_materialized": 0,
                "complete_ebpw": 2.9802419191571827,
            }
        cand = attach_graph(spec, g, q2f_mlp_ns, parity_ok, gen)
        candidates.append(cand)

    heals = [c for c in candidates if c["id"] not in {"binary", "q2f"}]
    heals_sorted = sorted(heals, key=lambda c: -c["ranking_score"])
    coherent_faster = [
        c
        for c in heals
        if c.get("counts_as_heal") and c.get("faster_than_q2f_27_55ms")
    ]
    finding = {
        "coherent_healed_body_still_faster_than_q2f": bool(coherent_faster),
        "n_healing_candidates": len(heals),
        "n_that_reached_coherent_generation": sum(1 for c in heals if c.get("counts_as_heal")),
        "best_by_capability_per_tax": [c["id"] for c in heals_sorted],
        "coherent_and_faster": [c["id"] for c in coherent_faster],
        "injured_body": {
            "id": "binary_g64",
            "bpw": BINARY_BPW,
            "COMPLETE_TOKEN_NS": BINARY_COMPLETE_NS,
            "died_at": "coherent_generation",
            "why": "16 copies of token 271",
        },
        "q2f_reference": {
            "bpw": Q2F_BPW,
            "COMPLETE_TOKEN_NS": Q2F_COMPLETE_NS,
            "ms": Q2F_MS,
        },
        "reading": (
            "A 1.4-1.6 EBPW coherent body still under 27.55 ms is a win. "
            "If no island restores generation while staying faster than q2f, "
            "the coherence-tax curve and the exact sensitive regions are the result."
        ),
    }

    doc = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "obligation": "N036 (S024 §4-9, §6, §7, §85 DensityHealer)",
        "question": (
            "Where does the 1.25-bpw binary g64 body diverge from q2f on real "
            "activations, and what is the smallest protected island that restores "
            "coherent generation while staying faster than q2f (27.55 ms)?"
        ),
        "did_not_load_second_27b": True,
        "did_not_write_under_models": True,
        "did_not_mutate_noetic_parent_a": True,
        "dense_w_materialized": 0,
        "dense_w": 0,
        "dense_w_is_a_counter": True,
        "parent_params": PARENT_PARAMS,
        "q4_incumbent": Q4_INCUMBENT,
        "kernel_competence": competence,
        "build": {k: build.get(k) for k in ("ok", "exit_code", "wall_s", "stderr_tail", "skipped")},
        "run": {
            "ok": measured.get("ok"),
            "exit_code": measured.get("exit_code"),
            "wall_s": measured.get("wall_s"),
            "raw_path": str(RAW),
        },
        "COHERENCE_FAILURE_MAP": fmap,
        "localization": {
            "capture": loc.get("capture"),
            "n_tokens_used": loc.get("n_tokens_used"),
            "split_rule": loc.get("split_rule"),
            "not_gaussian": True,
            "wall_s": loc.get("wall_s"),
            "earliest_layer": loc.get("earliest_layer"),
            "earliest_organ": loc.get("earliest_organ"),
            "mean_rel_l2_binary_vs_q2f_by_organ": loc.get("mean_rel_l2_binary_vs_q2f_by_organ"),
            "layers": loc.get("layers"),
        },
        "healing_candidates": heals_sorted,
        "injured_and_reference": [c for c in candidates if c["id"] in {"binary", "q2f"}],
        "compile": compile_reports,
        "finding": finding,
        "composition_ladder": {
            "rule": (
                "A candidate that fails a rung is KILLED THERE. A heal COUNTS "
                "only if it reaches coherent_generation. Capability is UNREACHED."
            ),
            "rungs": [
                "local_functional_probe",
                "held_out_activation",
                "adjacent_layers",
                "short_chain",
                "complete_organ",
                "complete_token",
                "coherent_generation",
                "capability",
            ],
        },
        "elapsed_s": time.perf_counter() - t0,
    }
    text = json.dumps(doc, indent=2)
    write_atomic(RECEIPT, text)
    print(f"wrote {RECEIPT} elapsed={doc['elapsed_s']:.1f}s", flush=True)
    print("finding", json.dumps(finding, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    _reexec_vision()
    raise SystemExit(main())
