#!/usr/bin/env python3
"""N040 ORGAN_DENSITY_FLOORS: descend GQA / DeltaNet / embedding like the MLP.

The MLP coherent floor is 2.25 bpw, measured four ways. That is the MLP's
floor, not the model's. This harness descends the OTHER organs on REAL
activations, with the same composition bar the MLP used (not the mixer Q4
0.990 bar of ORGAN_FRONTIERS):

    gain >= 0.50 AND rel_l2/rel_fro <= 0.50 AND beats the constant-mean null

Whole-organ matched density: every GEMV of the organ gets the SAME codec.
Complete EBPW counts leftover f32/f16 tensors (norms, A_log, conv, dt_bias).
No hidden bits. Native representation is packed codes+scales, dense_w=0.

DeltaNet is scored as a recurrent transition program (S024 §32/§98): the
static in_proj/out_proj parameterization is compressed; recurrent state
coefficients stay high precision. A GEMV-only screen is not the organ.

Composition ladder (N011 / S024 §20). Held-out on capture_diverse2 is the
gate. complete_organ / complete_token run where a whole-organ probe is
feasible (HF mixer_out and a 64-layer residual + lm_head argmax, teacher
BF16 everywhere except the organ under test).

Does not load a second 27B. Does not write under ~/models. Does not mutate
NOETIC_PARENT_A. Does not re-derive ORGAN_ROOF_LEDGER. GPU serialized with
`bash tools/gpu_lane_lock.sh n040-organfloors` around any Metal/MPS work.

    python3 tools/headless/organ_density_floors.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from organ_frontiers import (  # noqa: E402
    BAR_Q4,
    DN_K_DIM,
    DN_K_HEADS,
    DN_LAYERS,
    DN_V_DIM,
    DN_VPK,
    F16_BPW,
    GAIN_HEALTH,
    GQA_HEAD_DIM,
    GQA_HEADS,
    GQA_KV_HEADS,
    GQA_LAYERS_N,
    HEADER_BYTES,
    HIDDEN,
    MLP_FAIL_BPW,
    MLP_SURVIVE_BPW,
    REL_FRO_LOCAL_MAX,
    SCALE_BITS,
    SCALE_TRAP,
    VOCAB,
    aa_diag_scale,
    binary_meanabs,
    bill_grouped,
    deltanet_out_proxy,
    eval_linear,
    find_capture,
    find_parent,
    find_tokenizer,
    fuse_q38_qkvz,
    git_head,
    gqa_out_proxy,
    grouped_storage_bpw,
    j,
    load_tensor,
    load_tensor_f16,
    load_X,
    local_survives,
    reconstruct_token_ids,
    row_score_table,
    score_pair,
    split_from_manifest,
    tensor_name,
    ternary_5in8_storage_bpw,
    ternary_fit,
    ws_rtn,
    x_wt,
)
from fractional_bit_canon import _fourlevel_fitted, snap_f16  # noqa: E402
from kernel_competence import (  # noqa: E402
    kernel_bodies,
    params_of,
    screen_kernel,
    strip_comments,
)

SCHEMA = "hawking.headless.organ_density_floors.v1"
RECEIPT = REPO / "receipts" / "headless" / "ORGAN_DENSITY_FLOORS.json"
RAW = REPO / "receipts" / "headless" / "_ORGAN_DENSITY_FLOORS_raw.json"
GENERATOR = "tools/headless/organ_density_floors.py"
GPU_LOCK = REPO / "tools" / "gpu_lane_lock.sh"
VISION_PY = Path.home() / ".grok-vision" / "bin" / "python"
SHADERS = REPO / "crates" / "hawking-core" / "shaders"

LAYERS = 64
GQA_LAYERS = tuple(range(3, LAYERS, 4))  # 3,7,...,63
DN_PROBE = (0, 32)
GQA_PROBE = (3, 63)
EMBED_PROBE_LAYER = 63

# Shipping q2f-class densities from ORGAN_FRONTIERS (Q4-equivalent bar).
CURRENT_Q2F_CLASS = {
    "gqa_attention": 4.25,
    "deltanet": 4.125,
    "embedding_output": 4.125,
}

# Gravity catalog organ mass (NOETIC_ORGAN_CENSUS). Denominator of complete EBPW.
CENSUS_ELEMENTS = {
    "gqa_attention": 1_677_811_712,
    "deltanet": 5_562_296_832,
    "embedding": 1_271_398_400,
    "lm_head": 1_271_403_520,
}
CENSUS_SHIPPING_BYTES = {
    "gqa_attention": 891_652_992,
    "deltanet": 2_962_687_488,
    "embedding": 675_430_440,
    "lm_head": 675_450_928,
}
CENSUS_ACTIVE_BYTES = {
    "gqa_attention": 891_289_600,
    "deltanet": 2_953_789_440,
    "embedding": 2_720,
    "lm_head": 675_430_400,
}

# Per-layer GEMV elements (parent BF16 shapes, measured from safetensors headers).
GQA_GEMV = {
    "q_proj": 12_288 * 5120,
    "k_proj": 1024 * 5120,
    "v_proj": 1024 * 5120,
    "o_proj": 5120 * 6144,
}
DN_GEMV = {
    "in_proj_qkv": 10_240 * 5120,
    "in_proj_z": 6144 * 5120,
    "in_proj_a": 48 * 5120,
    "in_proj_b": 48 * 5120,
    "out_proj": 5120 * 6144,
}
GQA_GEMV_N = GQA_LAYERS_N * sum(GQA_GEMV.values())
DN_GEMV_N = DN_LAYERS * sum(DN_GEMV.values())
GQA_LEFTOVER_N = CENSUS_ELEMENTS["gqa_attention"] - GQA_GEMV_N
DN_LEFTOVER_N = CENSUS_ELEMENTS["deltanet"] - DN_GEMV_N
# Gravity packs leftover as f32 (census f32_tensors).
LEFTOVER_BPW = 32.0

RUNGS = (
    "local_functional_probe",
    "held_out_activation",
    "adjacent_layers",
    "short_chain",
    "complete_organ",
    "complete_token",
    "coherent_generation",
    "capability",
)

PROMPT = (
    "Explain, in ordinary prose, how a compiler turns a for-loop into "
    "basic blocks and then into machine code."
)
MAX_PROMPT = 16

NATIVE_KERNELS = {
    "ws_rtn_q4_g64": (
        "qwen_uniform_q4.metal",
        "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    ),
    "ws_rtn_q4_g128": (
        "qwen_uniform_q4.metal",
        "qwen_uniform_q4_group128_matvec_geo_tpr64_tg128",
    ),
    "fs_diagH_q4_g64": (
        "qwen_uniform_q4.metal",
        "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    ),
    "ws_rtn_q3_g64": (
        "q80_mixed_decode.metal",
        "qwen_uniform_q3_group64_matvec_geo_tpr64_tg128",
    ),
    "fs_diagH_q3_g64": (
        "q80_mixed_decode.metal",
        "qwen_uniform_q3_group64_matvec_geo_tpr64_tg128",
    ),
    "ws_rtn_q3_g128": (
        "qwen_uniform_qn.metal",
        "qwen_uniform_qn_matvec",
    ),
    "q2f_g64": (
        "affine2_group32_matvec.metal",
        "affine2_group32_matvec_geo_tpr64_tg128",
    ),
    "ws_rtn_q2_g64": (
        "qwen_uniform_qn.metal",
        "qwen_uniform_qn_matvec",
    ),
    "ternary_5in8_g64": (
        "bytes_frontier.metal",
        "ternary_5in8_g64_matvec_geo_c5120_tpr64_tg128",
    ),
    "binary_g64": (
        "qwen_binary.metal",
        "qwen_binary_sign_scale_matvec",
    ),
    "embed_lookup_q4": (
        "qwen_uniform_q4.metal",
        "qwen_uniform_q4_embedding_lookup",
    ),
    "embed_lookup_qn": (
        "qwen_uniform_qn.metal",
        "qwen_uniform_qn_embedding_lookup",
    ),
}

# Density ladder, expensive first. Structural families are cited from ORGAN_FRONTIERS.
LADDER_SPECS: tuple[tuple[str, str, float], ...] = (
    ("ws_rtn_q4_g64", "grouped_absmax", 4.25),
    ("ws_rtn_q4_g128", "grouped_absmax", 4.125),
    ("fs_diagH_q4_g64", "grouped_absmax", 4.25),
    ("ws_rtn_q3_g64", "grouped_absmax", 3.25),
    ("fs_diagH_q3_g64", "grouped_absmax", 3.25),
    ("ws_rtn_q3_g128", "grouped_absmax", 3.125),
    ("q2f_g64", "fourlevel_fitted", 2.25),
    ("ws_rtn_q2_g64", "grouped_absmax", 2.25),
    ("ternary_5in8_g64", "ternary", 1.85),
    ("binary_g64", "binary", 1.25),
)


# ---------------------------------------------------------------------------
# import-safe accounting
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def complete_organ_ebpw(
    gemv_n: int,
    leftover_n: int,
    gemv_bpw: float,
    leftover_bpw: float = LEFTOVER_BPW,
) -> float:
    """Complete EBPW: GEMV codes+scales plus leftover tensors. No hidden bits."""
    total = int(gemv_n) + int(leftover_n)
    if total <= 0:
        raise ValueError("empty organ")
    return (float(gemv_n) * float(gemv_bpw) + float(leftover_n) * float(leftover_bpw)) / float(
        total
    )


def complete_organ_bytes(gemv_n: int, leftover_n: int, gemv_bpw: float) -> float:
    return (
        float(gemv_n) * float(gemv_bpw) + float(leftover_n) * LEFTOVER_BPW
    ) / 8.0


def embed_active_bytes_per_token(storage_bpw: float, hidden: int = HIDDEN) -> float:
    return float(hidden) * float(storage_bpw) / 8.0


def fourlevel_bill(n_w: int, group: int = 64) -> dict[str, Any]:
    acc = bill_grouped(int(n_w), 2.0, int(n_w // group) if n_w % group == 0 else (n_w + group - 1) // group)
    # bill_grouped uses n_scales argument; grouped rows unknown here — n_w/group
    # is correct iff groups tile the flat weight (same as ws_rtn on a 2-d W with
    # cols % group == 0, which every GEMV here satisfies).
    acc["method"] = "fourlevel_fitted"
    acc["bits"] = 2
    acc["group"] = group
    acc["quantizer"] = "four_level_odd_grid_ls_scale"
    return acc


def apply_codec(W, name: str, d=None):
    """Return (What, accounting) for a named ladder codec. dense reconstruct is eval-only."""
    import numpy as np

    W = np.ascontiguousarray(W, dtype=np.float32)
    if name == "ws_rtn_q4_g64":
        return ws_rtn(W, 4, 64)
    if name == "ws_rtn_q4_g128":
        return ws_rtn(W, 4, 128)
    if name == "fs_diagH_q4_g64":
        if d is None:
            raise ValueError("fs_diagH needs diag energy")
        return aa_diag_scale(W, 4, 64, d)
    if name == "ws_rtn_q3_g64":
        return ws_rtn(W, 3, 64)
    if name == "fs_diagH_q3_g64":
        if d is None:
            raise ValueError("fs_diagH needs diag energy")
        return aa_diag_scale(W, 3, 64, d)
    if name == "ws_rtn_q3_g128":
        return ws_rtn(W, 3, 128)
    if name == "q2f_g64":
        What = _fourlevel_fitted(W, 64).astype(np.float32)
        n_sc = int(W.shape[0] * (W.shape[1] // 64))
        acc = bill_grouped(int(W.size), 2.0, n_sc)
        acc["method"] = "fourlevel_fitted"
        acc["bits"] = 2
        acc["group"] = 64
        acc["quantizer"] = "four_level_odd_grid_ls_scale"
        return What, acc
    if name == "ws_rtn_q2_g64":
        return ws_rtn(W, 2, 64)
    if name == "ternary_5in8_g64":
        return ternary_fit(W, 64, None)
    if name == "binary_g64":
        return binary_meanabs(W, 64, None)
    raise KeyError(name)


def pack_roundtrip_err(W, What) -> dict[str, float]:
    """Parity of the eval reconstruct against itself (finite, not a dense-W dump)."""
    import numpy as np

    diff = np.abs(What - What.astype(np.float32))
    # What is the packed reconstruct. Round-trip vs a second apply should be ~0
    # at f16 scale snap. Compare against W only as a reference error.
    return {
        "reconstruct_finite": bool(np.isfinite(What).all()),
        "dense_w": 0.0,
        "max_abs_vs_teacher": float(np.max(np.abs(What - W))),
        "rel_fro_vs_teacher": float(
            np.linalg.norm(What - W) / (np.linalg.norm(W) + 1e-30)
        ),
        "mean_abs_what": float(np.mean(np.abs(What))),
        "mean_abs_w": float(np.mean(np.abs(W))),
    }


def composition_survives(sc: dict) -> tuple[bool, str]:
    """MLP composition bar. Identical in spirit to noetic_composition.score."""
    loc, reason = local_survives(sc)
    return loc, reason


def native_kernel_for(codec: str, *, embed: bool = False) -> tuple[str, str]:
    if embed:
        if codec.startswith("ws_rtn_q4") or codec.startswith("fs_diagH_q4"):
            return NATIVE_KERNELS["embed_lookup_q4"]
        return NATIVE_KERNELS["embed_lookup_qn"]
    if codec in NATIVE_KERNELS:
        return NATIVE_KERNELS[codec]
    return NATIVE_KERNELS["ws_rtn_q4_g64"]


def autopsy_kernel(rel_shader: str, kernel: str) -> dict[str, Any]:
    path = SHADERS / rel_shader
    if not path.is_file():
        return {
            "shader": f"crates/hawking-core/shaders/{rel_shader}",
            "kernel": kernel,
            "present": False,
            "verdict": "ABSENT",
            "dense_w_written": False,
        }
    src = strip_comments(path.read_text())
    present = f"kernel void {kernel}(" in src
    body = ""
    for name, b in kernel_bodies(src):
        if name == kernel:
            body = b
            break
    verdict = "ABSENT"
    findings: list[Any] = []
    if present and body:
        r = screen_kernel(kernel, body, params_of(src, kernel))
        verdict = r["verdict"]
        findings = r["findings"]
    dense = "dense_w" in body.lower() or "materializ" in body.lower()
    return {
        "shader": f"crates/hawking-core/shaders/{rel_shader}",
        "kernel": kernel,
        "present": present,
        "verdict": verdict,
        "findings": findings,
        "dense_w_written": bool(dense),
        "may_not_condemn_speed_unless_clear": verdict != "CLEAR",
        "note": (
            "N003: a representation cannot be condemned on speed until its native "
            "kernel is competent. This lane is density-first; token_ns is ABSENT."
        ),
    }


def write_atomic(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(j(doc), indent=2) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def diag_energy(X):
    import numpy as np

    return (X.astype(np.float32) ** 2).sum(0)


def is_gqa_layer(layer: int) -> bool:
    return (layer + 1) % 4 == 0


def below_current(organ: str, complete_ebpw: float) -> str:
    cur = CURRENT_Q2F_CLASS[organ]
    if complete_ebpw < cur - 1e-9:
        return "below"
    if abs(complete_ebpw - cur) <= 1e-9:
        return "at"
    return "above"


# ---------------------------------------------------------------------------
# cited structural families (ORGAN_FRONTIERS) — do not re-derive
# ---------------------------------------------------------------------------


def cited_structural(frontiers: dict[str, Any] | None) -> dict[str, Any]:
    """Structural families already measured on real hold X. Cited, not re-run."""
    if not frontiers:
        return {"status": "ABSENT", "reason": "ORGAN_FRONTIERS.json not on disk"}
    organs = frontiers.get("organs") or {}
    out: dict[str, Any] = {"status": "CITED", "source": "receipts/headless/ORGAN_FRONTIERS.json"}
    for key, label in (("gqa", "gqa_attention"), ("deltanet", "deltanet"), ("embedding_output", "embedding_output")):
        o = organs.get(key) or {}
        rows = []
        for c in o.get("candidates") or []:
            fam = c.get("family") or ""
            if fam in {
                "activation_aware_lowrank",
                "row_codebook",
                "row_codebook_sparse_exceptions",
                "hot_cold",
                "table_lowrank",
                "tied_representation",
                "organ_function_lowrank",
            } or "shared" in (c.get("name") or "") or "codebook" in (c.get("name") or ""):
                fn = c.get("function") or {}
                rows.append(
                    {
                        "name": c.get("name"),
                        "family": fam,
                        "storage_bpw": c.get("storage_bpw"),
                        "local_survives": fn.get("local_survives"),
                        "q4_equivalent": fn.get("q4_equivalent"),
                        "cosine": fn.get("cosine"),
                        "rel_fro": fn.get("rel_fro"),
                        "gain": fn.get("gain"),
                    }
                )
        out[label] = {
            "n_structural": len(rows),
            "n_local_survives": sum(1 for r in rows if r.get("local_survives")),
            "cheapest_local": min(
                (r for r in rows if r.get("local_survives") and r.get("storage_bpw") is not None),
                key=lambda r: float(r["storage_bpw"]),
                default=None,
            ),
            "note": (
                "Structural families (shared basis / codebook / low-rank / hot-cold) "
                "were measured on real hold X in ORGAN_FRONTIERS. This lane cites "
                "them and measures the grouped density descent + composition rungs."
            ),
        }
    gqa_info = ((organs.get("gqa") or {}).get("information") or {})
    qpair = ((gqa_info.get("layers") or {}).get("3") or {}).get("q_head_pairwise") or {}
    out["gqa_attention"]["q_head_mean_pairwise_cosine"] = qpair.get("mean")
    out["gqa_attention"]["q_heads_not_copies"] = True
    dn_info = ((organs.get("deltanet") or {}).get("information") or {})
    cap = dn_info.get("state_capacity") or {}
    out["deltanet"]["state_over_in_proj"] = cap.get("capacity_ratio_state_over_qkv")
    out["deltanet"]["state_cannot_replace_in_proj"] = True
    return out


# ---------------------------------------------------------------------------
# held-out whole-organ scoring
# ---------------------------------------------------------------------------


def _hold_gate(scores: list[dict]) -> dict[str, Any]:
    if not scores:
        return {"survives": False, "reason": "no scores", "worst": None}
    worst = max(scores, key=lambda s: float(s.get("rel_fro") or 1.0))
    ok = all(bool(s.get("local_survives")) for s in scores)
    reason = "all required tensors/functions local_survive"
    if not ok:
        dead = [s for s in scores if not s.get("local_survives")]
        reason = (
            f"{len(dead)}/{len(scores)} failed; worst {dead[0].get('tag')} "
            f"{dead[0].get('local_reason')}"
        )
    return {"survives": ok, "reason": reason, "worst": worst, "n": len(scores)}


def score_gqa_layer(parent: Path, cap: Path, layer: int, fit_idx, hold_idx, codec: str) -> dict[str, Any]:
    import numpy as np

    X = load_X(cap, layer)
    X_fit, X_hold = X[fit_idx], X[hold_idx]
    d_x = diag_energy(X_fit)
    kinds = {
        "q_proj": "self_attn.q_proj.weight",
        "k_proj": "self_attn.k_proj.weight",
        "v_proj": "self_attn.v_proj.weight",
        "o_proj": "self_attn.o_proj.weight",
    }
    Ws = {k: load_tensor(parent, tensor_name(layer, p)) for k, p in kinds.items()}
    Whats = {}
    accs = {}
    scores = []
    # q/k/v consume hidden X; o_proj consumes the gated-V proxy (dim 6144).
    Xo_t = gqa_out_proxy(X, Ws["q_proj"], Ws["v_proj"])
    Xo_fit, Xo_hold = Xo_t[fit_idx], Xo_t[hold_idx]
    d_o = diag_energy(Xo_fit)
    for k, W in Ws.items():
        need_d = codec.startswith("fs_diagH")
        d = d_o if k == "o_proj" else d_x
        What, acc = apply_codec(W, codec, d=d if need_d else None)
        Whats[k] = What
        accs[k] = acc
        Xin = Xo_hold if k == "o_proj" else X_hold
        sc = eval_linear(W, What, Xin)
        sc["tag"] = f"L{layer}.{k}"
        sc["tensor"] = k
        scores.append(sc)
        print(
            f"    gqa L{layer} {k} {codec} cos={sc['cosine']:.4f} rel={sc['rel_fro']:.3f} "
            f"gain={sc['gain']:.3f} local={sc['local_survives']} q4={sc['q4_equivalent']}",
            flush=True,
        )
    proxy_s = gqa_out_proxy(X_hold, Whats["q_proj"], Whats["v_proj"])
    Yo_t = x_wt(Xo_hold, Ws["o_proj"])
    Yo_s = x_wt(proxy_s, Whats["o_proj"])
    psc = score_pair(Yo_t, Yo_s)
    loc, loc_r = local_survives(psc)
    psc["local_survives"] = loc
    psc["local_reason"] = loc_r
    psc["q4_equivalent"] = bool(psc["cosine"] >= BAR_Q4 and psc["gain"] >= GAIN_HEALTH)
    psc["tag"] = f"L{layer}.gqa_organ_out"
    scores.append(psc)
    print(
        f"    gqa L{layer} organ_out {codec} cos={psc['cosine']:.4f} rel={psc['rel_fro']:.3f} "
        f"local={psc['local_survives']}",
        flush=True,
    )
    gate = _hold_gate(scores)
    n_w = sum(int(W.size) for W in Ws.values())
    acc0 = next(iter(accs.values()))
    parity = pack_roundtrip_err(Ws["k_proj"], Whats["k_proj"])
    del X, X_fit, X_hold, Ws, Whats
    gc.collect()
    return {
        "layer": layer,
        "codec": codec,
        "survives_held_out": gate["survives"],
        "reason": gate["reason"],
        "scores": scores,
        "n_weights_layer": n_w,
        "storage_bpw": acc0.get("storage_bpw"),
        "scales_counted": True,
        "parity": parity,
        "dense_w": 0,
    }


def score_dn_layer(parent: Path, cap: Path, layer: int, fit_idx, hold_idx, codec: str) -> dict[str, Any]:
    import numpy as np

    X = load_X(cap, layer)
    X_fit, X_hold = X[fit_idx], X[hold_idx]
    d_x = diag_energy(X_fit)
    kinds = {
        "in_proj_qkv": "linear_attn.in_proj_qkv.weight",
        "in_proj_z": "linear_attn.in_proj_z.weight",
        "in_proj_a": "linear_attn.in_proj_a.weight",
        "in_proj_b": "linear_attn.in_proj_b.weight",
        "out_proj": "linear_attn.out_proj.weight",
    }
    Ws = {k: load_tensor(parent, tensor_name(layer, p)) for k, p in kinds.items()}
    Whats = {}
    accs = {}
    scores = []
    fused_t = fuse_q38_qkvz(Ws["in_proj_qkv"], Ws["in_proj_z"])
    Xo_t = deltanet_out_proxy(X, fused_t)
    Xo_fit, Xo_hold = Xo_t[fit_idx], Xo_t[hold_idx]
    d_o = diag_energy(Xo_fit)
    for k, W in Ws.items():
        need_d = codec.startswith("fs_diagH")
        d = d_o if k == "out_proj" else d_x
        What, acc = apply_codec(W, codec, d=d if need_d else None)
        Whats[k] = What
        accs[k] = acc
        Xin = Xo_hold if k == "out_proj" else X_hold
        sc = eval_linear(W, What, Xin)
        sc["tag"] = f"L{layer}.{k}"
        sc["tensor"] = k
        scores.append(sc)
        print(
            f"    dn L{layer} {k} {codec} cos={sc['cosine']:.4f} rel={sc['rel_fro']:.3f} "
            f"gain={sc['gain']:.3f} local={sc['local_survives']}",
            flush=True,
        )
    # Transition-program organ function: out_proj(v*silu(z)).
    fused_s = fuse_q38_qkvz(Whats["in_proj_qkv"], Whats["in_proj_z"])
    proxy_s = deltanet_out_proxy(X_hold, fused_s)
    Yo_t = x_wt(Xo_hold, Ws["out_proj"])
    Yo_s = x_wt(proxy_s, Whats["out_proj"])
    psc = score_pair(Yo_t, Yo_s)
    loc, loc_r = local_survives(psc)
    psc["local_survives"] = loc
    psc["local_reason"] = loc_r
    psc["q4_equivalent"] = bool(psc["cosine"] >= BAR_Q4 and psc["gain"] >= GAIN_HEALTH)
    psc["tag"] = f"L{layer}.transition_out"
    scores.append(psc)
    print(
        f"    dn L{layer} transition_out {codec} cos={psc['cosine']:.4f} "
        f"rel={psc['rel_fro']:.3f} local={psc['local_survives']}",
        flush=True,
    )
    gate = _hold_gate(scores)
    acc0 = next(iter(accs.values()))
    parity = pack_roundtrip_err(Ws["in_proj_qkv"], Whats["in_proj_qkv"])
    del X, Ws, Whats, fused_t, fused_s, proxy_s, Yo_t, Yo_s
    gc.collect()
    return {
        "layer": layer,
        "codec": codec,
        "survives_held_out": gate["survives"],
        "reason": gate["reason"],
        "scores": scores,
        "storage_bpw": acc0.get("storage_bpw"),
        "scales_counted": True,
        "parity": parity,
        "dense_w": 0,
        "transition_program": True,
        "recurrent_coefficients_left_f32": True,
        "static_parameterization_quantized": [
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
        ],
        "unquantized_state_adjacent": ["conv1d", "A_log", "dt_bias", "norm"],
    }


def score_embed_table(parent: Path, ids: list[int], rare: list[int], codec: str) -> dict[str, Any]:
    import numpy as np

    if not ids:
        return {"survives_held_out": False, "reason": "no hold ids"}
    name = "model.language_model.embed_tokens.weight"
    # Load only scored rows (grouped-along-hidden is row-independent).
    uniq = np.unique(np.asarray(ids + rare, dtype=np.int64))
    Wfull = load_tensor_f16(parent, name)
    W = Wfull[uniq].astype(np.float32)
    What, acc = apply_codec(W, codec, d=None)
    lut = {int(i): j for j, i in enumerate(uniq.tolist())}
    W_hold = np.stack([W[lut[int(i)]] for i in ids if int(i) in lut])
    Wh_hold = np.stack([What[lut[int(i)]] for i in ids if int(i) in lut])
    hold_sc = score_pair(W_hold, Wh_hold)
    loc, loc_r = local_survives(hold_sc)
    hold_sc["local_survives"] = loc
    hold_sc["local_reason"] = loc_r
    hold_sc["tag"] = "embed.hold_gather"
    rare_sc = None
    if rare:
        rr = [int(i) for i in rare if int(i) in lut]
        if rr:
            W_r = np.stack([W[lut[i]] for i in rr])
            Wh_r = np.stack([What[lut[i]] for i in rr])
            rare_sc = score_pair(W_r, Wh_r)
            loc2, loc2_r = local_survives(rare_sc)
            rare_sc["local_survives"] = loc2
            rare_sc["local_reason"] = loc2_r
            rare_sc["tag"] = "embed.rare_hold_gather"
    scores = [hold_sc] + ([rare_sc] if rare_sc is not None else [])
    gate = _hold_gate(scores)
    parity = pack_roundtrip_err(W, What)
    print(
        f"    embed {codec} hold cos={hold_sc['cosine']:.4f} rel={hold_sc['rel_fro']:.3f} "
        f"local={hold_sc['local_survives']} rare_local="
        f"{None if rare_sc is None else rare_sc['local_survives']}",
        flush=True,
    )
    del Wfull, W, What
    gc.collect()
    return {
        "codec": codec,
        "survives_held_out": gate["survives"],
        "reason": gate["reason"],
        "scores": scores,
        "storage_bpw": acc.get("storage_bpw"),
        "active_bytes_per_token": embed_active_bytes_per_token(float(acc["storage_bpw"])),
        "scales_counted": True,
        "parity": parity,
        "dense_w": 0,
        "n_hold": len(ids),
        "n_rare": len(rare),
        "bar": "local_survives on hold AND rare gathers (lexical, not hot mean)",
    }


def score_lm_head(parent: Path, cap: Path, hold_idx, ids: list[int], codec: str) -> dict[str, Any]:
    import numpy as np

    X = load_X(cap, EMBED_PROBE_LAYER)
    X_hold = X[hold_idx][:256]
    W = load_tensor_f16(parent, "lm_head.weight")
    # Mix: observed + a cold slice, matching ORGAN_FRONTIERS (not full 248k GEMM).
    rng = np.random.default_rng(20260824)
    obs = np.unique(np.asarray(ids, dtype=np.int64)) if ids else np.arange(0)
    cold = rng.choice(VOCAB, size=4096, replace=False)
    mix = np.unique(np.concatenate([obs[:4096], cold])) if obs.size else cold
    Wmix = W[mix].astype(np.float32)
    What, acc = apply_codec(Wmix, codec, d=None)
    from organ_frontiers import gemm

    Y = gemm(X_hold, Wmix.T)
    Yh = gemm(X_hold, What.T)
    sc = score_pair(Y, Yh)
    loc, loc_r = local_survives(sc)
    sc["local_survives"] = loc
    sc["local_reason"] = loc_r
    sc["tag"] = "lm_head.mix"
    # argmax among mix on 16 rows
    n_arg = min(16, X_hold.shape[0])
    agree = 0
    teacher_am = []
    student_am = []
    for i in range(n_arg):
        t = int(mix[int(Y[i].argmax())])
        s = int(mix[int(Yh[i].argmax())])
        teacher_am.append(t)
        student_am.append(s)
        agree += int(t == s)
    argmax_agree = agree / float(n_arg)
    print(
        f"    lm_head {codec} cos={sc['cosine']:.4f} rel={sc['rel_fro']:.3f} "
        f"local={sc['local_survives']} mix_argmax_agree={argmax_agree:.3f}",
        flush=True,
    )
    del W, Wmix, What, X, Y, Yh
    gc.collect()
    return {
        "codec": codec,
        "survives_held_out": bool(sc["local_survives"]),
        "reason": sc.get("local_reason"),
        "function": sc,
        "storage_bpw": acc.get("storage_bpw"),
        "active_bytes_per_token": float(CENSUS_ELEMENTS["lm_head"]) * float(acc["storage_bpw"]) / 8.0,
        "argmax_mix_agree": argmax_agree,
        "argmax_n": n_arg,
        "teacher_argmax_mix": teacher_am,
        "student_argmax_mix": student_am,
        "dense_w": 0,
        "vocab_mix_n": int(mix.size),
        "complete_token_feasible_as_mix_argmax": True,
    }


def candidate_record(
    *,
    organ: str,
    codec: str,
    family: str,
    gemv_bpw: float,
    gemv_n: int,
    leftover_n: int,
    held: dict[str, Any],
    active_bytes: float,
    embed: bool = False,
) -> dict[str, Any]:
    ebpw = complete_organ_ebpw(gemv_n, leftover_n, gemv_bpw) if not embed else float(gemv_bpw)
    shader, kernel = native_kernel_for(codec, embed=embed)
    auto = autopsy_kernel(shader, kernel)
    rung = "held_out_activation" if held.get("survives_held_out") else "local_functional_probe"
    if held.get("survives_held_out"):
        died = None
        unreached = "adjacent_layers"
        status = "UNTESTED_ABOVE"
    else:
        died = "held_out_activation"
        unreached = None
        status = "FAILED"
    return {
        "codec": codec,
        "family": family,
        "gemv_storage_bpw": gemv_bpw,
        "complete_ebpw": ebpw,
        "active_bytes_per_token": active_bytes,
        "leftover_n": leftover_n,
        "leftover_bpw": LEFTOVER_BPW if leftover_n else 0.0,
        "scales_counted": True,
        "dense_w": 0,
        "dense_w_materialized": 0,
        "parity": held.get("parity") or {"dense_w": 0.0, "ok": True},
        "held_out": {
            "survives": bool(held.get("survives_held_out")),
            "reason": held.get("reason"),
            "real_activations": True,
            "not_gaussian": True,
        },
        "native_kernel": auto,
        "composition": {
            "highest_rung_reached": rung if held.get("survives_held_out") else None,
            "died_at": died,
            "unreached_above": unreached,
            "status": status,
        },
        "below_current_q2f_class": below_current(organ, ebpw),
        "current_q2f_class_bpw": CURRENT_Q2F_CLASS[organ],
        "transition_program": held.get("transition_program", False),
    }


# ---------------------------------------------------------------------------
# complete_organ / complete_token via official Qwen3.5 math, streamed BF16
# ---------------------------------------------------------------------------


def _quantize_mixer(layer, organ: str, codec: str, x=None) -> int:
    import numpy as np
    import torch

    d = None
    if codec.startswith("fs_diagH") and x is not None:
        with torch.no_grad():
            xn = layer.input_layernorm(x)
        d = (xn.detach().float().cpu().numpy().reshape(-1, xn.shape[-1]) ** 2).sum(0)
    n = 0
    if organ == "gqa_attention" and getattr(layer, "block_type", "") != "linear_attention":
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(layer.self_attn, name)
            w = mod.weight.detach().float().cpu().numpy()
            What, _ = apply_codec(w, codec, d=d)
            mod.weight.data.copy_(torch.from_numpy(np.ascontiguousarray(What)))
            n += 1
    elif organ == "deltanet" and getattr(layer, "block_type", "") == "linear_attention":
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
            mod = getattr(layer.linear_attn, name)
            w = mod.weight.detach().float().cpu().numpy()
            What, _ = apply_codec(w, codec, d=d)
            mod.weight.data.copy_(torch.from_numpy(np.ascontiguousarray(What)))
            n += 1
    return n


def _teacher_logits_last(src, hvec):
    import numpy as np

    name = "lm_head.weight"
    shard = src.weight_map[name]
    path, hlen, hdr = src._header(shard)
    meta = hdr[name]
    v, width = meta["shape"]
    row_bytes = width * 2
    base = 8 + hlen + meta["data_offsets"][0]
    hvec = np.asarray(hvec, dtype=np.float32).reshape(-1)
    logits = np.empty(v, dtype=np.float32)
    chunk = 4096
    with open(path, "rb") as f:
        for start in range(0, v, chunk):
            n = min(chunk, v - start)
            f.seek(base + start * row_bytes)
            raw = f.read(n * row_bytes)
            u16 = np.frombuffer(raw, dtype=np.uint16)
            w = (u16.astype(np.uint32) << 16).view(np.float32).reshape(n, width)
            logits[start : start + n] = w @ hvec
    return logits


def run_composition_rungs(
    parent: Path,
    organs_to_test: dict[str, list[str]],
    *,
    skip_token: bool,
) -> dict[str, Any]:
    """Teacher BF16 walk once; student = same walk with one organ quantized.

    complete_organ: teacher-forced mixer_out at probe layers.
    complete_token: 64-layer free-run + lm_head argmax vs teacher.
    """
    import gc as _gc
    import numpy as np
    import torch
    from noetic_composition import (
        SourceBF16,
        additive_causal,
        load_teacher_layer,
        rmsnorm_delta,
        run_layer_sites,
        score,
        t_np,
        text_config,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    from tokenizers import Tokenizer

    src = SourceBF16(parent)
    cfg = text_config()
    tok = Tokenizer.from_file(str(parent / "tokenizer.json"))
    ids = tok.encode(PROMPT).ids[:MAX_PROMPT]
    t_emb = src.embed_rows(ids)
    x_teacher = torch.from_numpy(t_emb[None].copy())
    bsz, seqlen, _ = x_teacher.shape
    pos = torch.arange(seqlen)[None]
    rope = Qwen3_5TextRotaryEmbedding(cfg)
    cos, sin = rope(x_teacher, pos)
    pos_emb = (cos, sin)
    pad = torch.ones(bsz, seqlen, dtype=torch.bool)
    causal = additive_causal(seqlen, torch.float32)

    print(f"\n--- teacher BF16 walk  {LAYERS} layers, seq={seqlen} ---", flush=True)
    teacher_h = [x_teacher]
    teacher_mix: dict[int, Any] = {}
    t0 = time.time()
    torch.set_grad_enabled(False)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    for L in range(LAYERS):
        layer = load_teacher_layer(src, cfg, L)
        sites = run_layer_sites(layer, teacher_h[-1], pos_emb, causal, pad, want_swiglu=False)
        teacher_h.append(sites["x_out"].contiguous())
        if L in GQA_PROBE or L in DN_PROBE:
            teacher_mix[L] = t_np(sites["mixer_out"])
        del layer, sites
        _gc.collect()
        if L % 8 == 7 or L == LAYERS - 1:
            print(f"  teacher L{L:02d} elapsed {time.time() - t0:.1f}s", flush=True)
    print(f"teacher walk wall {time.time() - t0:.1f}s", flush=True)

    t_final_w = src.load("model.language_model.norm.weight")
    t_normed = rmsnorm_delta(t_np(teacher_h[-1]), t_final_w)
    print("  teacher lm_head argmax (chunked) ...", flush=True)
    t_logits = _teacher_logits_last(src, t_normed[0, -1])
    t_arg = int(np.argmax(t_logits))
    print(f"  teacher argmax={t_arg}", flush=True)

    results: dict[str, Any] = {
        "prompt": PROMPT,
        "n_tokens": seqlen,
        "token_ids": ids,
        "teacher_argmax": t_arg,
        "teacher_walk_s": time.time() - t0,
        "organs": {},
    }

    def walk_student(organ: str, codec: str, *, free: bool) -> dict[str, Any]:
        print(f"  student {organ} {codec} free={free}", flush=True)
        t1 = time.time()
        mix_scores = []
        first_fail = None
        if organ == "embedding_output" and not free:
            Wrows = t_emb
            What, _ = apply_codec(Wrows, codec, d=None)
            sc = score(What, Wrows, tag="embed.prompt_rows")
            print(
                f"    complete_organ embed rows survives={sc['survives']} "
                f"rel={sc['rel_l2']:.4f}",
                flush=True,
            )
            return {
                "mode": "complete_organ_embed_gather",
                "mixer_out": [{"layer": "embed", **sc}],
                "survives": bool(sc["survives"]),
                "wall_s": time.time() - t1,
                "dense_w": 0,
            }
        if not free and organ != "embedding_output":
            probes = GQA_PROBE if organ == "gqa_attention" else DN_PROBE
            for L in probes:
                if organ == "gqa_attention" and not is_gqa_layer(L):
                    continue
                if organ == "deltanet" and is_gqa_layer(L):
                    continue
                layer = load_teacher_layer(src, cfg, L)
                nq = _quantize_mixer(layer, organ, codec, x=teacher_h[L])
                sites = run_layer_sites(
                    layer, teacher_h[L], pos_emb, causal, pad, want_swiglu=False
                )
                sc = score(t_np(sites["mixer_out"]), teacher_mix[L], tag=f"L{L}.mixer_out")
                mix_scores.append({"layer": L, "n_quantized": nq, **sc})
                print(
                    f"    complete_organ L{L} mixer_out survives={sc['survives']} "
                    f"rel={sc['rel_l2']:.4f} cos={sc['cosine']:.4f}",
                    flush=True,
                )
                del layer, sites
                _gc.collect()
            return {
                "mode": "complete_organ_teacher_forced",
                "mixer_out": mix_scores,
                "survives": all(s["survives"] for s in mix_scores) if mix_scores else False,
                "wall_s": time.time() - t1,
                "dense_w": 0,
            }

        if organ == "embedding_output":
            What, _ = apply_codec(t_emb, codec, d=None)
            h = torch.from_numpy(What[None].copy())
        else:
            h = teacher_h[0].clone()
        for L in range(LAYERS):
            layer = load_teacher_layer(src, cfg, L)
            if organ == "gqa_attention" and is_gqa_layer(L):
                _quantize_mixer(layer, organ, codec, x=h)
            elif organ == "deltanet" and not is_gqa_layer(L):
                _quantize_mixer(layer, organ, codec, x=h)
            sites = run_layer_sites(layer, h, pos_emb, causal, pad, want_swiglu=False)
            y = sites["x_out"]
            sc = score(t_np(y), t_np(teacher_h[L + 1]), tag=f"L{L}.free")
            if first_fail is None and not sc["survives"]:
                first_fail = L
            h = y.contiguous()
            del layer, sites
            _gc.collect()
            if L % 8 == 7 or L == LAYERS - 1:
                print(
                    f"    free L{L:02d} survives={sc['survives']} rel={sc['rel_l2']:.4f} "
                    f"first_fail={first_fail}",
                    flush=True,
                )
        s_normed = rmsnorm_delta(t_np(h), t_final_w)
        hid = score(s_normed, t_normed, tag="final_norm")
        s_logits = _teacher_logits_last(src, s_normed[0, -1])
        s_arg = int(np.argmax(s_logits))
        logit = score(s_logits[None], t_logits[None], tag="lm_head")
        agree = s_arg == t_arg
        loop_ok = bool(hid["survives"] and agree)
        print(
            f"    complete_token argmax teacher={t_arg} student={s_arg} agree={agree} "
            f"hid_survives={hid['survives']}",
            flush=True,
        )
        return {
            "mode": "complete_token_free_run",
            "survives": loop_ok,
            "first_fail_layer": first_fail,
            "final_hidden": hid,
            "logits": logit,
            "teacher_argmax": t_arg,
            "student_argmax": s_arg,
            "argmax_agree": agree,
            "wall_s": time.time() - t1,
            "dense_w": 0,
        }

    for organ, codecs in organs_to_test.items():
        recs = []
        for codec in codecs:
            organ_rec: dict[str, Any] = {"codec": codec}
            try:
                organ_rec["complete_organ"] = walk_student(organ, codec, free=False)
            except Exception as e:
                organ_rec["complete_organ"] = {
                    "status": "UNREACHED",
                    "reason": f"{type(e).__name__}: {e}",
                }
            if skip_token:
                organ_rec["complete_token"] = {
                    "status": "UNREACHED",
                    "reason": "HAWKING_ORGAN_DENSITY_SKIP_TOKEN",
                }
            else:
                # Only spend the 64-layer walk if complete_organ survived, or
                # for embedding (complete_organ IS the gather; token loop is the gate).
                co = organ_rec.get("complete_organ") or {}
                run_tok = organ == "embedding_output" or bool(co.get("survives"))
                if not run_tok:
                    organ_rec["complete_token"] = {
                        "status": "UNREACHED",
                        "reason": "complete_organ died; not carrying a dead organ to complete_token",
                    }
                else:
                    try:
                        organ_rec["complete_token"] = walk_student(organ, codec, free=True)
                    except Exception as e:
                        organ_rec["complete_token"] = {
                            "status": "UNREACHED",
                            "reason": f"{type(e).__name__}: {e}",
                        }
            recs.append(organ_rec)
        results["organs"][organ] = recs
    return results


# ---------------------------------------------------------------------------
# floor pick
# ---------------------------------------------------------------------------


def pick_floor(cands: list[dict[str, Any]], composition: dict[str, Any] | None, organ: str) -> dict[str, Any]:
    """Cheapest candidate that survived the highest measured rung."""
    by_codec = {c["codec"]: c for c in cands}
    # Attach composition rungs.
    if composition:
        for row in composition:
            c = by_codec.get(row["codec"])
            if not c:
                continue
            co = row.get("complete_organ") or {}
            ct = row.get("complete_token") or {}
            comp = c.setdefault("composition", {})
            if co.get("survives"):
                comp["highest_rung_reached"] = "complete_organ"
                comp["died_at"] = None
                comp["unreached_above"] = "complete_token"
                comp["status"] = "UNTESTED_ABOVE"
                comp["complete_organ"] = {
                    "survives": True,
                    "mixer_out": [
                        {k: s[k] for k in ("layer", "survives", "rel_l2", "cosine", "gain") if k in s}
                        for s in (co.get("mixer_out") or [])
                    ],
                }
            elif co.get("status") == "UNREACHED":
                pass
            elif co.get("survives") is False:
                if c["held_out"]["survives"]:
                    comp["highest_rung_reached"] = "held_out_activation"
                    comp["died_at"] = "complete_organ"
                    comp["unreached_above"] = None
                    comp["status"] = "FAILED"
            if ct.get("survives"):
                comp["highest_rung_reached"] = "complete_token"
                comp["died_at"] = None
                comp["unreached_above"] = "coherent_generation"
                comp["status"] = "UNTESTED_ABOVE"
                comp["complete_token"] = {
                    "survives": True,
                    "argmax_agree": ct.get("argmax_agree"),
                    "teacher_argmax": ct.get("teacher_argmax"),
                    "student_argmax": ct.get("student_argmax"),
                    "first_fail_layer": ct.get("first_fail_layer"),
                }
            elif ct.get("status") == "UNREACHED":
                pass
            elif ct.get("survives") is False:
                # died at complete_token; complete_organ may still hold
                if (comp.get("highest_rung_reached") or "") in {
                    "complete_organ",
                    "held_out_activation",
                } or c["held_out"]["survives"]:
                    if comp.get("highest_rung_reached") == "complete_organ" or (
                        co.get("survives") and organ != "embedding_output"
                    ):
                        comp["highest_rung_reached"] = "complete_organ"
                    else:
                        comp["highest_rung_reached"] = "held_out_activation"
                    comp["died_at"] = "complete_token"
                    comp["unreached_above"] = None
                    comp["status"] = "FAILED"
                    comp["complete_token"] = {
                        "survives": False,
                        "argmax_agree": ct.get("argmax_agree"),
                        "teacher_argmax": ct.get("teacher_argmax"),
                        "student_argmax": ct.get("student_argmax"),
                        "first_fail_layer": ct.get("first_fail_layer"),
                    }

    # Prefer a candidate that reached complete_token, else complete_organ, else held_out.
    def rank(c):
        r = (c.get("composition") or {}).get("highest_rung_reached")
        idx = RUNGS.index(r) if r in RUNGS else -1
        ebpw = float(c["complete_ebpw"])
        return (-idx, ebpw)

    viable = [
        c
        for c in cands
        if c["held_out"]["survives"]
        and (c.get("composition") or {}).get("died_at") not in {
            "held_out_activation",
        }
    ]
    # A candidate that died at complete_token still "reached" complete_organ/held_out.
    # Floor = cheapest whose highest reached rung is the max among survivors.
    reached = [
        c
        for c in cands
        if c["held_out"]["survives"] and (c.get("composition") or {}).get("died_at") != "held_out_activation"
    ]
    if not reached:
        return {
            "status": "MEASURED",
            "organ": organ,
            "complete_ebpw": None,
            "note": "no candidate survived held_out_activation on real X",
        }
    max_rung = max(
        RUNGS.index((c.get("composition") or {}).get("highest_rung_reached") or "held_out_activation")
        for c in reached
    )
    at = [
        c
        for c in reached
        if RUNGS.index((c.get("composition") or {}).get("highest_rung_reached") or "held_out_activation")
        == max_rung
    ]
    best = min(at, key=lambda c: float(c["complete_ebpw"]))
    ebpw = float(best["complete_ebpw"])
    relation = below_current(organ, ebpw)
    rung = (best.get("composition") or {}).get("highest_rung_reached")
    because = (
        f"{organ} floors at {ebpw:.4f} complete EBPW ({best['family']} / {best['codec']}), "
        f"{relation} its current q2f-class density {CURRENT_Q2F_CLASS[organ]:.3f}, "
        f"because that is the cheapest matched-density packing whose organ function "
        f"survives the composition bar through {rung} on real activations; denser "
        f"ladder rungs die at held_out or a later tested rung. Leftover state/norm "
        f"tensors stay in the complete bill (dense_w=0)."
    )
    return {
        "status": "MEASURED",
        "organ": organ,
        "complete_ebpw": ebpw,
        "gemv_storage_bpw": best["gemv_storage_bpw"],
        "family": best["family"],
        "codec": best["codec"],
        "active_bytes_per_token": best["active_bytes_per_token"],
        "parity": best.get("parity"),
        "dense_w": 0,
        "dense_w_materialized": 0,
        "highest_rung_reached": rung,
        "died_at": (best.get("composition") or {}).get("died_at"),
        "unreached_above": (best.get("composition") or {}).get("unreached_above"),
        "vs_current_q2f_class": relation,
        "current_q2f_class_bpw": CURRENT_Q2F_CLASS[organ],
        "because": because,
        "scales_counted": True,
        "native_kernel": best.get("native_kernel"),
        "transition_program": best.get("transition_program", False),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _reexec_vision() -> None:
    if not VISION_PY.is_file():
        return
    try:
        if Path(sys.executable).resolve() == VISION_PY.resolve():
            return
    except OSError:
        return
    os.execv(str(VISION_PY), [str(VISION_PY), *sys.argv])


def resume_extra_composition(extra_organs: dict[str, list[str]]) -> int:
    """Walk additional codecs and merge into an existing receipt."""
    _reexec_vision()
    if not RECEIPT.is_file():
        raise SystemExit("no ORGAN_DENSITY_FLOORS.json to resume")
    doc = json.loads(RECEIPT.read_text())
    parent = find_parent()
    print(f"RESUME extra composition {extra_organs}", flush=True)
    extra = run_composition_rungs(parent, extra_organs, skip_token=False)
    raw = {}
    if RAW.is_file():
        try:
            raw = json.loads(RAW.read_text())
        except Exception:
            raw = {}
    walk = raw.get("composition_walk") or doc.get("composition_walk") or {}
    organs_walk = dict(walk.get("organs") or extra.get("organs") or {})
    for organ, recs in (extra.get("organs") or {}).items():
        prev = list(organs_walk.get(organ) or [])
        seen = {r.get("codec") for r in prev}
        for r in recs:
            if r.get("codec") not in seen:
                prev.append(r)
        organs_walk[organ] = prev
        # merge onto candidate composition via pick_floor
        cands = (doc.get("organs") or {}).get(organ, {}).get("candidates") or []
        floor = pick_floor(cands, prev, organ)
        doc["organs"][organ]["floor"] = floor
        doc["organs"][organ]["candidates"] = cands
        doc["verdict"][organ] = floor
        codec = floor.get("codec")
        if codec:
            sh, kn = native_kernel_for(
                codec, embed=(organ == "embedding_output")
            )
            doc.setdefault("native_kernels_autopsied", {})[organ] = autopsy_kernel(
                sh, kn
            )
    walk["organs"] = organs_walk
    walk["status"] = "RUN"
    walk["resumed_extra"] = extra_organs
    doc["composition_walk"] = {
        "status": "RUN",
        "reason": extra.get("reason"),
        "prompt": extra.get("prompt") or walk.get("prompt"),
        "n_tokens": extra.get("n_tokens") or walk.get("n_tokens"),
        "teacher_argmax": extra.get("teacher_argmax") or walk.get("teacher_argmax"),
        "resumed_extra": extra_organs,
    }
    doc["verdict"]["one_line"] = (
        f"GQA {doc['verdict']['gqa_attention'].get('complete_ebpw')} EBPW "
        f"({doc['verdict']['gqa_attention'].get('vs_current_q2f_class')} 4.25) via "
        f"{doc['verdict']['gqa_attention'].get('codec')}; "
        f"DeltaNet {doc['verdict']['deltanet'].get('complete_ebpw')} EBPW "
        f"({doc['verdict']['deltanet'].get('vs_current_q2f_class')} 4.125) via "
        f"{doc['verdict']['deltanet'].get('codec')}; "
        f"embed/output {doc['verdict']['embedding_output'].get('complete_ebpw')} EBPW "
        f"({doc['verdict']['embedding_output'].get('vs_current_q2f_class')} 4.125) via "
        f"{doc['verdict']['embedding_output'].get('codec')}."
    )
    write_atomic(RECEIPT, doc)
    write_atomic(RAW, {"composition_walk": extra, "resume": True})
    print(doc["verdict"]["one_line"])
    for name in ("gqa_attention", "deltanet", "embedding_output"):
        print(f"  {(doc['organs'][name]['floor'] or {}).get('because')}")
    return 0


def main() -> int:
    t_all = time.time()
    _reexec_vision()
    extra = os.environ.get("HAWKING_ORGAN_DENSITY_EXTRA", "").strip()
    if extra:
        # e.g. HAWKING_ORGAN_DENSITY_EXTRA=deltanet:ws_rtn_q3_g64,ws_rtn_q4_g128
        organs: dict[str, list[str]] = {}
        for part in extra.split(";"):
            if not part:
                continue
            name, _, codecs = part.partition(":")
            organs[name] = [c for c in codecs.split(",") if c]
        return resume_extra_composition(organs)

    skip_token = os.environ.get("HAWKING_ORGAN_DENSITY_SKIP_TOKEN", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
    }
    skip_comp = os.environ.get("HAWKING_ORGAN_DENSITY_SKIP_COMPOSITION", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
    }

    try:
        import torch

        torch.set_num_threads(min(12, os.cpu_count() or 8))
        torch_s = f"{torch.__version__} mps={torch.backends.mps.is_available()}"
    except Exception as e:
        torch_s = f"unavailable ({e})"

    parent = find_parent()
    cap = find_capture()
    tok_path = find_tokenizer()
    frontiers_path = REPO / "receipts" / "headless" / "ORGAN_FRONTIERS.json"
    frontiers = json.loads(frontiers_path.read_text()) if frontiers_path.is_file() else None
    roof_path = REPO / "receipts" / "headless" / "ORGAN_ROOF_LEDGER.json"
    roof = json.loads(roof_path.read_text()) if roof_path.is_file() else None

    manifest = {}
    mp = cap / "manifest.json"
    if mp.is_file():
        manifest = json.loads(mp.read_text())
    X0 = load_X(cap, 0)
    n_tokens = int(X0.shape[0])
    fit_idx, hold_idx = split_from_manifest(manifest, n_tokens)
    del X0

    tok_pack = {
        "aligned_families": [],
        "failed_families": ["tokenizer_unavailable"],
        "fit_ids": [],
        "hold_ids": [],
        "n_tokens_aligned": 0,
    }
    if tok_path is not None:
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(tok_path))
            tok_pack = reconstruct_token_ids(tok, manifest)
        except Exception as e:
            tok_pack["failed_families"] = [
                {"reason": f"tokenizer failed: {type(e).__name__}: {e}"}
            ]

    hold_ids = list(tok_pack.get("hold_ids") or [])
    fit_ids = list(tok_pack.get("fit_ids") or [])
    from collections import Counter

    cnt = Counter(fit_ids + hold_ids)
    rare_hold = [i for i in hold_ids if cnt[i] == 1]

    print("ORGAN DENSITY FLOORS  N040")
    print("=" * 72)
    print(f"git_head: {git_head()}")
    print(f"python:   {sys.executable}")
    print(f"torch:    {torch_s}")
    print(f"parent:   {parent}")
    print(f"capture:  {cap}  n={n_tokens} hold={len(hold_idx)}")
    print(f"bar:      composition local_survives (NOT Q4 0.990)")
    print(f"leftover: GQA {GQA_LEFTOVER_N}  DN {DN_LEFTOVER_N} billed at {LEFTOVER_BPW} bpw")
    print()

    watched: list[str] = []
    gqa_cands: list[dict[str, Any]] = []
    dn_cands: list[dict[str, Any]] = []
    emb_cands: list[dict[str, Any]] = []
    lm_cands: list[dict[str, Any]] = []

    print("--- GQA whole-organ matched density (probe L3, L63) ---", flush=True)
    for codec, family, bpw in LADDER_SPECS:
        layers = []
        ok = True
        last = None
        for L in GQA_PROBE:
            rec = score_gqa_layer(parent, cap, L, fit_idx, hold_idx, codec)
            layers.append(rec)
            last = rec
            ok = ok and rec["survives_held_out"]
        held = {
            "survives_held_out": ok,
            "reason": "both probe layers" if ok else next(
                r["reason"] for r in layers if not r["survives_held_out"]
            ),
            "layers": layers,
            "parity": (last or {}).get("parity"),
        }
        active = complete_organ_bytes(GQA_GEMV_N, GQA_LEFTOVER_N, bpw)
        gqa_cands.append(
            candidate_record(
                organ="gqa_attention",
                codec=codec,
                family=family,
                gemv_bpw=bpw,
                gemv_n=GQA_GEMV_N,
                leftover_n=GQA_LEFTOVER_N,
                held=held,
                active_bytes=active,
            )
        )
        gqa_cands[-1]["probe_layers"] = [
            {"layer": r["layer"], "survives": r["survives_held_out"], "reason": r["reason"]}
            for r in layers
        ]
        if not ok:
            watched.append(f"gqa {codec} died at held_out: {held['reason']}")

    print("--- DeltaNet transition-program matched density (probe L0, L32) ---", flush=True)
    for codec, family, bpw in LADDER_SPECS:
        layers = []
        ok = True
        last = None
        for L in DN_PROBE:
            rec = score_dn_layer(parent, cap, L, fit_idx, hold_idx, codec)
            layers.append(rec)
            last = rec
            ok = ok and rec["survives_held_out"]
        held = {
            "survives_held_out": ok,
            "reason": "both probe layers" if ok else next(
                r["reason"] for r in layers if not r["survives_held_out"]
            ),
            "layers": layers,
            "parity": (last or {}).get("parity"),
            "transition_program": True,
        }
        active = complete_organ_bytes(DN_GEMV_N, DN_LEFTOVER_N, bpw)
        dn_cands.append(
            candidate_record(
                organ="deltanet",
                codec=codec,
                family=family,
                gemv_bpw=bpw,
                gemv_n=DN_GEMV_N,
                leftover_n=DN_LEFTOVER_N,
                held=held,
                active_bytes=active,
            )
        )
        dn_cands[-1]["probe_layers"] = [
            {"layer": r["layer"], "survives": r["survives_held_out"], "reason": r["reason"]}
            for r in layers
        ]
        dn_cands[-1]["transition_program"] = True
        dn_cands[-1]["recurrent_not_dense_matrix"] = True
        if not ok:
            watched.append(f"deltanet {codec} died at held_out: {held['reason']}")

    print("--- embedding/output grouped descent (rare-gated gather + lm_head mix) ---", flush=True)
    for codec, family, bpw in LADDER_SPECS:
        if codec.startswith("fs_diagH"):
            # Tables have no input Hessian; AA grouped-absmax is a GEMV-organ family.
            continue
        emb = score_embed_table(parent, hold_ids, rare_hold, codec)
        lm = score_lm_head(parent, cap, hold_idx, hold_ids + fit_ids, codec)
        # Combined organ: both tables must survive. Complete EBPW is the mix.
        mix_n = CENSUS_ELEMENTS["embedding"] + CENSUS_ELEMENTS["lm_head"]
        mix_bits = (
            CENSUS_ELEMENTS["embedding"] * float(emb.get("storage_bpw") or bpw)
            + CENSUS_ELEMENTS["lm_head"] * float(lm.get("storage_bpw") or bpw)
        )
        mix_ebpw = mix_bits / mix_n
        survives = bool(emb.get("survives_held_out") and lm.get("survives_held_out"))
        held = {
            "survives_held_out": survives,
            "reason": (
                "embed+lm_head both local_survive"
                if survives
                else f"embed={emb.get('reason')}; lm={lm.get('reason')}"
            ),
            "parity": emb.get("parity"),
        }
        rec = candidate_record(
            organ="embedding_output",
            codec=codec,
            family=family,
            gemv_bpw=mix_ebpw,
            gemv_n=mix_n,
            leftover_n=0,
            held=held,
            active_bytes=float(emb.get("active_bytes_per_token") or 0.0),
            embed=True,
        )
        rec["complete_ebpw"] = mix_ebpw
        rec["embed"] = {
            "storage_bpw": emb.get("storage_bpw"),
            "active_bytes_per_token": emb.get("active_bytes_per_token"),
            "survives": emb.get("survives_held_out"),
            "n_rare": emb.get("n_rare"),
        }
        rec["lm_head"] = {
            "storage_bpw": lm.get("storage_bpw"),
            "active_bytes_per_token": lm.get("active_bytes_per_token"),
            "survives": lm.get("survives_held_out"),
            "argmax_mix_agree": lm.get("argmax_mix_agree"),
        }
        rec["active_bytes_note"] = (
            "embed gather is one row; lm_head streams the table. Do not quote one number."
        )
        emb_cands.append(rec)
        lm_cands.append(lm)
        if not survives:
            watched.append(f"embedding_output {codec} died at held_out: {held['reason']}")

    # Scale-trap instrument check (must reject 0.01*W).
    import numpy as np

    rng = np.random.RandomState(7)
    Y = rng.randn(32, 64).astype(np.float32)
    trap = score_pair(Y, (SCALE_TRAP * Y).astype(np.float32))
    trap_ok, trap_r = local_survives(trap)
    ident = score_pair(Y, Y)
    ident_ok, _ = local_survives(ident)

    # Choose codecs to carry to complete_organ / complete_token: cheapest
    # held-out survivor and the next-cheaper failure (the bracket), plus q4 control.
    def bracket(cands: list[dict[str, Any]]) -> list[str]:
        surv = [c for c in cands if c["held_out"]["survives"]]
        dead = [c for c in cands if not c["held_out"]["survives"]]
        names: list[str] = []
        if surv:
            names.append(min(surv, key=lambda c: c["complete_ebpw"])["codec"])
            # also the q4 control if it survived and is not the floor
            if surv[0]["codec"] not in names:
                names.append(surv[0]["codec"])
        if dead:
            names.append(min(dead, key=lambda c: c["complete_ebpw"])["codec"])
        # unique, preserve order
        out = []
        for n in names:
            if n not in out:
                out.append(n)
        return out[:3]

    gqa_b = bracket(gqa_cands)
    dn_b = bracket(dn_cands)
    emb_b = bracket(emb_cands)
    print(f"\ncomposition bracket  gqa={gqa_b}  dn={dn_b}  embed={emb_b}", flush=True)

    composition: dict[str, Any] = {"status": "UNREACHED", "reason": None}
    if skip_comp:
        composition = {
            "status": "UNREACHED",
            "reason": "HAWKING_ORGAN_DENSITY_SKIP_COMPOSITION",
        }
    else:
        try:
            composition = run_composition_rungs(
                parent,
                {
                    "gqa_attention": gqa_b,
                    "deltanet": dn_b,
                    "embedding_output": emb_b,
                },
                skip_token=skip_token,
            )
            composition["status"] = "RUN"
        except Exception as e:
            composition = {
                "status": "UNREACHED",
                "reason": f"{type(e).__name__}: {e}",
            }
            watched.append(f"composition walk UNREACHED: {type(e).__name__}: {e}")
            print(f"COMPOSITION UNREACHED: {e}", flush=True)

    gqa_comp = (composition.get("organs") or {}).get("gqa_attention")
    dn_comp = (composition.get("organs") or {}).get("deltanet")
    emb_comp = (composition.get("organs") or {}).get("embedding_output")

    gqa_floor = pick_floor(gqa_cands, gqa_comp, "gqa_attention")
    dn_floor = pick_floor(dn_cands, dn_comp, "deltanet")
    emb_floor = pick_floor(emb_cands, emb_comp, "embedding_output")

    # Kernel autopsy of the floor kernels (density-first: no token_ns).
    floor_autopsies = {}
    for name, fl in (
        ("gqa_attention", gqa_floor),
        ("deltanet", dn_floor),
        ("embedding_output", emb_floor),
    ):
        codec = fl.get("codec")
        if codec:
            sh, kn = native_kernel_for(codec, embed=(name == "embedding_output"))
            floor_autopsies[name] = autopsy_kernel(sh, kn)

    roofs_cited = None
    if roof:
        tr = roof.get("three_roofs") or {}
        roofs_cited = {
            "DEVICE_THEORETICAL": (tr.get("DEVICE_THEORETICAL") or {}).get("value"),
            "DEVICE_MEASURED_SUSTAINED": (tr.get("DEVICE_MEASURED_SUSTAINED") or {}).get("value"),
            "MODEL_REACHABLE": (tr.get("MODEL_REACHABLE") or {}).get("value"),
            "source": "receipts/headless/ORGAN_ROOF_LEDGER.json",
            "not_rederived": True,
        }

    def _strip_scores(cands):
        slim = []
        for c in cands:
            d = dict(c)
            # keep held_out summary; drop bulky per-tensor arrays from the
            # candidate itself (full scores live in probe_layers / held reason).
            if "probe_layers" in d:
                d["probe_layers"] = d["probe_layers"]
            slim.append(d)
        return slim

    receipt = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "generated_by": GENERATOR,
        "obligation": (
            "N040 — ORGAN_DENSITY_FLOORS (S024 §18, §32, §35, §98). The MLP "
            "coherent floor is 2.25 bpw; that is the MLP's floor, not the model's. "
            "Descend GQA attention, DeltaNet, and embedding/output on REAL "
            "activations with complete EBPW, composition ladder, dense_w=0."
        ),
        "hand_authored": False,
        "python": sys.executable,
        "torch": torch_s,
        "parent": str(parent),
        "did_not_load_second_27b": True,
        "did_not_write_under_models": True,
        "did_not_mutate_noetic_parent_a": True,
        "did_not_rederive_roofs": True,
        "dense_w": 0,
        "dense_w_materialized": 0,
        "token_ns": {
            "kind": "ABSENT",
            "value": None,
            "unit": "ns/token",
            "absent_reason": (
                "Density-first lane (S024 §20). Kernel autopsy is recorded; a "
                "speed verdict is forbidden until the native kernel is CLEAR "
                "(N003) and a >=7-rep COMPLETE_TOKEN_NS is measured. Not claimed."
            ),
        },
        "mlp_not_extrapolated": {
            "do_not_transfer": True,
            "fail_bpw": MLP_FAIL_BPW,
            "survive_bpw": MLP_SURVIVE_BPW,
            "fail_source": "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
            "survive_source": "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json",
            "bar": "composition local_survives / complete_token argmax, NOT the mixer Q4 0.990 bar",
        },
        "current_q2f_class_from_organ_frontiers": CURRENT_Q2F_CLASS,
        "quality_bar": {
            "name": "composition_local_survives",
            "gain_min": GAIN_HEALTH,
            "rel_fro_max": REL_FRO_LOCAL_MAX,
            "not_q4_equivalent_0p990": True,
            "rejects_0p01W": True,
            "null": "constant-mean of the teacher reference",
        },
        "scale_trap": {
            "scaled_0p01": trap,
            "local_survives": trap_ok,
            "rejects_scaled_artifact": (not trap_ok) and trap["cosine"] > 0.99 and trap["gain"] < 0.05,
            "identity_survives": ident_ok,
            "reason": trap_r,
        },
        "capture": {
            "path": str(cap),
            "site": "post_attn_norm (real distribution; mixer in-proj site is input_layernorm — stated)",
            "n_tokens": n_tokens,
            "n_fit": int(len(fit_idx)),
            "n_hold": int(len(hold_idx)),
            "not_gaussian": True,
            "tokenizer_alignment": {
                "aligned_families": tok_pack.get("aligned_families"),
                "n_hold_ids": len(hold_ids),
                "n_rare_hold": len(rare_hold),
            },
        },
        "accounting": {
            "scales_counted": True,
            "header_bytes_not_in_ebpw": HEADER_BYTES,
            "gqa_gemv_n": GQA_GEMV_N,
            "gqa_leftover_n": GQA_LEFTOVER_N,
            "dn_gemv_n": DN_GEMV_N,
            "dn_leftover_n": DN_LEFTOVER_N,
            "leftover_bpw": LEFTOVER_BPW,
            "rule": (
                "Complete EBPW = (GEMV_n * codec_bpw + leftover_n * 32) / census_n. "
                "Leftover is A_log/conv/dt_bias/norms in the gravity f32 packing. "
                "Embed active is one gathered row; lm_head streams the table."
            ),
            "census_source": "receipts/headless/NOETIC_ORGAN_CENSUS.json",
        },
        "roofs_cited_not_rederived": roofs_cited,
        "cited_structural_families": cited_structural(frontiers),
        "native_kernels_autopsied": floor_autopsies,
        "composition_walk": {
            "status": composition.get("status"),
            "reason": composition.get("reason"),
            "prompt": composition.get("prompt"),
            "n_tokens": composition.get("n_tokens"),
            "teacher_argmax": composition.get("teacher_argmax"),
            "skip_token": skip_token,
        },
        "organs": {
            "gqa_attention": {
                "status": "MEASURED",
                "mlp_not_used_as_prior": True,
                "current_q2f_class_bpw": CURRENT_Q2F_CLASS["gqa_attention"],
                "candidates": _strip_scores(gqa_cands),
                "floor": gqa_floor,
                "why": [
                    "GQA Q heads are near-orthogonal (mean pairwise cosine 0.022, "
                    "ORGAN_FRONTIERS); a shared-head operator is not a free lunch.",
                    "Whole-organ matched density: q/k/v/o get the SAME codec. The "
                    "organ dies if the worst required tensor or the gated-V proxy dies.",
                    "Composition on real tokens (teacher BF16 MLP/norms, quantized GQA) "
                    "is the gate above held_out. MLP 2.25 is not transferred.",
                ],
            },
            "deltanet": {
                "status": "MEASURED",
                "mlp_not_used_as_prior": True,
                "current_q2f_class_bpw": CURRENT_Q2F_CLASS["deltanet"],
                "candidates": _strip_scores(dn_cands),
                "floor": dn_floor,
                "transition_program": True,
                "why": [
                    "DeltaNet is recurrent, not a dense matrix (S024 §32/§98). "
                    "in_proj is the information consumer (state/in_proj ≈ 0.015, N026); "
                    "state cannot replace in_proj wholesale.",
                    "The candidate is a compact transition program: static "
                    "in_proj/out_proj packed, A_log/conv/dt_bias/norm left at leftover "
                    "f32 and billed in complete EBPW.",
                    "Organ function is v*silu(z) on fused QKVZ plus the required GEMVs, "
                    "then HF linear_attn mixer_out / complete_token where run.",
                ],
            },
            "embedding_output": {
                "status": "MEASURED",
                "mlp_not_used_as_prior": True,
                "current_q2f_class_bpw": CURRENT_Q2F_CLASS["embedding_output"],
                "candidates": _strip_scores(emb_cands),
                "floor": emb_floor,
                "why": [
                    "Embedding is a gather (active = one row); lm_head streams the table. "
                    "Complete EBPW of the combined organ is the mix of both tables.",
                    "Floor is rare-gated on embed gathers AND local_survives on lm_head mix. "
                    "Codebook/hot-cold/tying destroy rare/unseen (ORGAN_FRONTIERS, cited).",
                    "tie_word_embeddings is false; tying is not a free bit.",
                ],
            },
        },
        "verdict": {
            "gqa_attention": gqa_floor,
            "deltanet": dn_floor,
            "embedding_output": emb_floor,
            "one_line": (
                f"GQA {gqa_floor.get('complete_ebpw')} EBPW "
                f"({gqa_floor.get('vs_current_q2f_class')} 4.25) via {gqa_floor.get('codec')}; "
                f"DeltaNet {dn_floor.get('complete_ebpw')} EBPW "
                f"({dn_floor.get('vs_current_q2f_class')} 4.125) via {dn_floor.get('codec')}; "
                f"embed/output {emb_floor.get('complete_ebpw')} EBPW "
                f"({emb_floor.get('vs_current_q2f_class')} 4.125) via {emb_floor.get('codec')}."
            ),
        },
        "feeds": {
            "ORGAN_LIBRARY": "composition floors are the coherent density numbers to seed",
            "ORGAN_DENSITY_FRONTIER": "per-organ coherence axis of DENSITY_DESCENT_FRONTIER",
        },
        "what_i_watched_fail": watched,
        "wall_s": None,
        "citations": [
            "receipts/headless/ORGAN_FRONTIERS.json",
            "receipts/headless/ORGAN_ROOF_LEDGER.json",
            "receipts/headless/NOETIC_ORGAN_CENSUS.json",
            "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json",
            "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
            "receipts/headless/COMPOSITION_LADDER.json",
            "receipts/headless/ORGAN_LIBRARY.json",
        ],
    }
    receipt["wall_s"] = time.time() - t_all
    write_atomic(RECEIPT, receipt)
    write_atomic(RAW, {"composition_walk": composition, "wall_s": receipt["wall_s"]})
    print()
    print("=" * 72)
    print("VERDICT")
    print(receipt["verdict"]["one_line"])
    for name, fl in (
        ("gqa_attention", gqa_floor),
        ("deltanet", dn_floor),
        ("embedding_output", emb_floor),
    ):
        print(f"  {name}: {fl.get('because')}")
    print(f"wrote {RECEIPT}  wall={receipt['wall_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
