#!/usr/bin/env python3
"""GLOBAL_ALLOCATOR: spend each model-specific byte where function-space gain is highest.

ORGAN_FRONTIERS measured three independent non-MLP floors. The arithmetic they
force is the reason this lane exists:

    organ             elements        share    measured floor
    deltanet          5,562,296,832   20.7%    4.125 bpw
    gqa               1,677,811,712    6.2%    4.250
    mlp              17,112,760,320   63.6%    2.250   (CITED survive; not transferred)
    embedding_output  2,543,129,600    9.5%    4.125

    implied whole-model floor = 2.9398 EBPW
    even if MLP went to ZERO bits the other three still cost 1.5082 EBPW

Uniform-per-organ is therefore the wrong shape if the floors are tradeable.
This harness estimates, on real held-out activations, the marginal capability
gain per model-specific byte and the marginal throughput change per active
byte, then allocates GLOBALLY. A layer may take ~0.3 EBPW while another takes
4 EBPW if that is the better use of the same total bytes.

Protected islands: a cheap dominant code plus a small high-precision set of
hypersensitive channels / tensors / route structures. An island that buys
little is billed and then refused — it is an expensive exception, not a
representation.

The comparison that decides it: uniform allocation at total bytes B against
this global allocation at the SAME B. If global does not win, that is the
result (floors local, not tradeable).

    python3 tools/headless/global_allocator.py
    python3 -m pytest tools/headless/test_global_allocator.py -q

Function-space on real captured activations only; never Gaussian. Scales
counted. Null on every quality number. The GO metric rejects 0.01*W. Storage
AND active bpw. Does not load a second 27B: parent BF16 tensors stream one at
a time; the gravity Q4 artifact is never opened as a model.
"""
from __future__ import annotations

import gc
import json
import math
import os
import struct
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VISION_PY = Path.home() / ".grok-vision" / "bin" / "python"
RECEIPT = ROOT / "receipts" / "headless" / "GLOBAL_ALLOCATOR.json"
SCHEMA = "hawking.headless.global_allocator.v1"

# Reuse the organ-frontiers instrument, not a second copy of the metric.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from organ_frontiers import (  # noqa: E402
    CHUNK,
    F16_BPW,
    HIDDEN,
    PARENT_CANDIDATES,
    SCALE_BITS,
    SCALE_TRAP,
    SEED,
    VOCAB,
    aa_diag_scale,
    _silu,
    aa_rank_hat,
    bill_factors,
    bill_grouped,
    binary_meanabs,
    binary_storage_bpw,
    deltanet_out_proxy,
    diag_energy,
    eval_linear,
    find_capture,
    find_parent,
    find_tokenizer,
    fuse_q38_qkvz,
    git_head,
    git_json,
    gqa_out_proxy,
    grouped_storage_bpw,
    input_pcs,
    j,
    load_X,
    load_tensor,
    load_tensor_f16,
    numbered,
    q4_healthy,
    reconstruct_token_ids,
    row_score_table,
    score_pair,
    snap_f16,
    split_from_manifest,
    tensor_name,
    ternary_5in8_storage_bpw,
    ternary_fit,
    x_wt,
    ws_rtn,
)

SOURCE_PARAM_COUNT = 26_895_998_464
INTERMEDIATE = 17_408
HOLD_CAP = 768
FIT_CAP = 1024
ISLAND_FRAC = 0.01
RANK_KS = (64, 256)
HEADER_BYTES = 64  # billed in extra_bits when a packed tensor is stored

# Cited, not re-derived. ORGAN_FRONTIERS + composition ladder + fused parent A.
CITED_FLOOR_ELEMS = {
    "deltanet": 5_562_296_832,
    "gqa": 1_677_811_712,
    "mlp": 17_112_760_320,
    "embedding_output": 2_543_129_600,
}
CITED_FLOOR_BPW = {
    "deltanet": 4.125,
    "gqa": 4.250,
    "mlp": 2.250,
    "embedding_output": 4.125,
}
MLP_SURVIVE_BPW = 2.250
MLP_FAIL_BPW = 1.850
LEADER_EBPW = 3.1393
LEADER_SOURCE = "receipts/headless/NOETIC_FUSED_SUBBIT.json"
Q4_INCUMBENT_EBPW = 4.252735126866492

# Organ preciousness from the null-representation ranking (NOETIC_ORGAN_CENSUS).
# function_lost = 1 - survival when the organ's residual is zeroed.
ORGAN_FUNCTION_LOST = {
    "embedding": 1.0,
    "embedding_output": 1.0,
    "gqa": 0.6070411601515365,
    "deltanet": 0.8556759274497626,
    "mlp": 0.9153984562611263,
}

# q_inject measured at real captured operating points (tools/gravity_error_chain.py).
# Linear interpolation between sampled depths. Not re-derived here.
Q_INJECT = {
    0: 1.597e-04,
    7: 1.910e-04,
    15: 2.715e-04,
    23: 4.281e-04,
    31: 4.634e-04,
    39: 5.279e-04,
    47: 6.534e-04,
    55: 9.065e-04,
    63: 2.577e-03,
}
_QL = sorted(Q_INJECT)

# Throughput model. GPU_LEDGER: decode is bandwidth-bound enough that active
# bytes rank above stored size. TOKEN_NS and ACTIVE_BYTES are MEASURED there;
# the per-byte derivative here is DERIVED, not a new GPU run.
ROOF_GB_S = 778.8
ACTIVE_INCUMBENT = 13_622_266_960
TOKEN_NS_INCUMBENT = 30_375_208
GPU_LEDGER = "receipts/headless/GPU_LEDGER.json"

# Probe layers. GQA is every 4th starting at 3; DN otherwise. MLP on both.
DN_PROBE = (0, 32)
GQA_PROBE = (3, 63)
MLP_PROBE = (0, 31, 63)

GEMV_CLASSES = {
    "mlp.gate_proj.weight": "mlp",
    "mlp.up_proj.weight": "mlp",
    "mlp.down_proj.weight": "mlp",
    "linear_attn.in_proj_qkv.weight": "deltanet",
    "linear_attn.in_proj_z.weight": "deltanet",
    "linear_attn.out_proj.weight": "deltanet",
    "linear_attn.in_proj_a.weight": "deltanet",
    "linear_attn.in_proj_b.weight": "deltanet",
    "self_attn.q_proj.weight": "gqa",
    "self_attn.k_proj.weight": "gqa",
    "self_attn.v_proj.weight": "gqa",
    "self_attn.o_proj.weight": "gqa",
}

ROUTE_CLASSES = {
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
}

# Small tensors are held, not searched (same cutoff as gravity_allocator).
MIN_SEARCH_ELEMS = 1_000_000


# ---------------------------------------------------------------------------
# import-safe arithmetic (no torch, no 27B)
# ---------------------------------------------------------------------------

def implied_floor_bits() -> float:
    return sum(CITED_FLOOR_ELEMS[o] * CITED_FLOOR_BPW[o] for o in CITED_FLOOR_ELEMS)


def implied_floor_ebpw() -> float:
    return implied_floor_bits() / SOURCE_PARAM_COUNT


def non_mlp_floor_ebpw() -> float:
    bits = sum(
        CITED_FLOOR_ELEMS[o] * CITED_FLOOR_BPW[o]
        for o in CITED_FLOOR_ELEMS
        if o != "mlp"
    )
    return bits / SOURCE_PARAM_COUNT


def q_inject(layer: int | None) -> float:
    if layer is None:
        return Q_INJECT[0]
    if layer <= _QL[0]:
        return Q_INJECT[_QL[0]]
    if layer >= _QL[-1]:
        return Q_INJECT[_QL[-1]]
    lo = max(l for l in _QL if l <= layer)
    hi = min(l for l in _QL if l >= layer)
    if lo == hi:
        return Q_INJECT[lo]
    t = (layer - lo) / (hi - lo)
    return Q_INJECT[lo] * (1.0 - t) + Q_INJECT[hi] * t


def q_mult(layer: int | None) -> float:
    """Residual-sensitivity relative to L0. Endpoints: embed uses L0, lm_head L63."""
    return q_inject(layer) / q_inject(0)


def organ_of_class(cls: str) -> str:
    if cls in GEMV_CLASSES:
        return GEMV_CLASSES[cls]
    if cls in {"embed_tokens.weight", "embed"}:
        return "embedding"
    if cls in {"lm_head.weight", "lm_head"}:
        return "embedding_output"
    return "other"


def item_weight(elements: int, organ: str, layer: int | None) -> float:
    lost = ORGAN_FUNCTION_LOST.get(organ, 1.0)
    return float(elements) * float(lost) * float(q_mult(layer))


def storage_bytes_of(elements: int, bpw: float, extra_bits: float = 0.0) -> float:
    return (float(elements) * float(bpw) + float(extra_bits)) / 8.0


def grouped_bpw(bits: int, group: int) -> float:
    return grouped_storage_bpw(bits, group)


def bill_island(elements: int, n_kept_elems: int, cheap_bpw: float, n_index: int) -> dict:
    """Cheap dominant codes + f16 islands + 32-bit ids. Scales of the cheap code stay."""
    rest = int(elements) - int(n_kept_elems)
    if rest < 0:
        raise ValueError("island larger than tensor")
    index_bits = float(n_index) * 32.0
    bits = float(n_kept_elems) * F16_BPW + float(rest) * float(cheap_bpw) + index_bits
    bpw = bits / float(elements)
    return {
        "n_weights": int(elements),
        "n_island_elems": int(n_kept_elems),
        "n_index": int(n_index),
        "index_bits": index_bits,
        "cheap_bpw": float(cheap_bpw),
        "island_bpw": F16_BPW,
        "storage_bits": bits,
        "storage_bpw": bpw,
        "active_fused_bpw": bpw,
        "active_cached_f16_bpw": F16_BPW,
        "scales_counted": True,
        "note": (
            "dominant cheap code (scales counted) plus f16 island rows/cols "
            "plus 32-bit ids. fused active = storage; densified W_hat = 16."
        ),
    }


def damage_of(sc: dict) -> float:
    """1 - scale_aware. Sees 0.01*W (gain ~ 0.01). Cosine alone does not."""
    sa = float(sc.get("scale_aware") if sc.get("scale_aware") is not None else 0.0)
    return float(max(0.0, 1.0 - sa))


def lerp(a: float, b: float, t: float) -> float:
    return float(a) + float(t) * (float(b) - float(a))


def dense_levels(levels: list[dict]) -> list[dict]:
    return [lv for lv in levels if not lv.get("island") and not lv.get("control")]


def interpolate_level(levels: list[dict], bpw: float) -> dict:
    """Piecewise-linear RD on the dense (non-island) curve. Uniform lives here."""
    pts = sorted(dense_levels(levels), key=lambda x: float(x["storage_bpw"]))
    if not pts:
        raise ValueError("no dense levels to interpolate")
    if bpw <= float(pts[0]["storage_bpw"]):
        out = dict(pts[0])
        out["interpolated"] = False
        out["name"] = f"clamp_{pts[0]['name']}"
        return out
    if bpw >= float(pts[-1]["storage_bpw"]):
        out = dict(pts[-1])
        out["interpolated"] = False
        out["name"] = f"clamp_{pts[-1]['name']}"
        return out
    for lo, hi in zip(pts, pts[1:]):
        a, b = float(lo["storage_bpw"]), float(hi["storage_bpw"])
        if a <= bpw <= b:
            t = 0.0 if b <= a else (bpw - a) / (b - a)
            elems = int(lo["n_weights"])
            extra = lerp(float(lo.get("extra_bits") or 0), float(hi.get("extra_bits") or 0), t)
            storage_bytes = storage_bytes_of(elems, bpw, extra)
            active_bpw = lerp(float(lo["active_fused_bpw"]), float(hi["active_fused_bpw"]), t)
            out = {
                "name": f"lerp_{lo['name']}_{hi['name']}",
                "family": "interpolated_uniform",
                "storage_bpw": float(bpw),
                "active_fused_bpw": active_bpw,
                "active_cached_f16_bpw": F16_BPW,
                "storage_bytes": storage_bytes,
                "active_bytes": storage_bytes_of(elems, active_bpw, extra),
                "scale_aware": lerp(float(lo["scale_aware"]), float(hi["scale_aware"]), t),
                "damage": lerp(float(lo["damage"]), float(hi["damage"]), t),
                "rel_fro": lerp(float(lo["rel_fro"]), float(hi["rel_fro"]), t),
                "cosine": lerp(float(lo["cosine"]), float(hi["cosine"]), t),
                "gain": lerp(float(lo["gain"]), float(hi["gain"]), t),
                "null": float(lo["null"]),
                "n_weights": elems,
                "interpolated": True,
                "t": t,
                "lo": lo["name"],
                "hi": hi["name"],
                "scales_counted": True,
                "island": False,
                "control": False,
            }
            return out
    raise RuntimeError("interpolate_level fell through")


def legal_levels(levels: list[dict]) -> list[dict]:
    """Greedy may pick grouped, rank, and islands. Not identity, not the trap, not zero."""
    out = []
    for lv in levels:
        if lv.get("control"):
            continue
        if lv.get("name") in {"identity_f16", "scale_001W", "zero"}:
            continue
        out.append(lv)
    out.sort(key=lambda x: (float(x["storage_bytes"]), x["name"]))
    return out


def greedy_descend(items: list[dict], budget_bytes: float) -> dict:
    """Drop the cheapest damage-per-byte until total storage <= budget.

    Each item starts at its richest legal level. Ties break by item id so the
    walk is deterministic. Non-monotonic islands are allowed: they are just
    another point on the discrete curve.
    """
    state = []
    legal = []
    for it in items:
        lv = legal_levels(it["levels"])
        if not lv:
            raise ValueError(f"{it.get('id')} has no legal levels")
        legal.append(lv)
        state.append(len(lv) - 1)

    def total_bytes() -> float:
        return sum(legal[i][state[i]]["storage_bytes"] for i in range(len(items)))

    steps = []
    guard = 0
    max_steps = sum(len(lv) for lv in legal) + 2
    while total_bytes() > budget_bytes and guard < max_steps:
        guard += 1
        best = None
        for i, it in enumerate(items):
            k = state[i]
            if k <= 0:
                continue
            cur = legal[i][k]
            nxt = legal[i][k - 1]
            db = float(cur["storage_bytes"]) - float(nxt["storage_bytes"])
            if db <= 1e-9:
                continue
            dd = (float(nxt["damage"]) - float(cur["damage"])) * float(it["weight"])
            cost = dd / db
            cand = (cost, db, it["id"], i, k - 1, dd)
            if best is None or cand[0] < best[0] or (cand[0] == best[0] and cand[2] < best[2]):
                best = cand
        if best is None:
            break
        cost, db, iid, i, new_k, dd = best
        prev = legal[i][state[i]]["name"]
        state[i] = new_k
        steps.append(
            {
                "item": iid,
                "from": prev,
                "to": legal[i][new_k]["name"],
                "damage_per_byte": cost,
                "bytes_saved": db,
                "weighted_damage_delta": dd,
            }
        )

    assignment = [legal[i][state[i]] for i in range(len(items))]
    return {
        "assignment": assignment,
        "state": state,
        "steps": steps,
        "storage_bytes": total_bytes(),
        "hit_budget": total_bytes() <= budget_bytes + 1.0,
        "cheapest_exhausted": any(s == 0 for s in state) and total_bytes() > budget_bytes,
    }


def summarize_alloc(items: list[dict], assignment: list[dict]) -> dict:
    num_sa = den = num_dmg = 0.0
    storage = active = 0.0
    by_organ = defaultdict(lambda: {"elements": 0, "storage_bytes": 0.0, "active_bytes": 0.0, "w_sa": 0.0, "w": 0.0})
    by_name = defaultdict(int)
    bpws = []
    for it, lv in zip(items, assignment):
        w = float(it["weight"])
        sa = float(lv["scale_aware"])
        dmg = float(lv["damage"])
        num_sa += w * sa
        num_dmg += w * dmg
        den += w
        storage += float(lv["storage_bytes"])
        active += float(lv["active_bytes"])
        bpws.append(float(lv["storage_bpw"]))
        by_name[lv["name"]] += 1
        o = it["organ"]
        rec = by_organ[o]
        rec["elements"] += int(it["elements"])
        rec["storage_bytes"] += float(lv["storage_bytes"])
        rec["active_bytes"] += float(lv["active_bytes"])
        rec["w_sa"] += w * sa
        rec["w"] += w
    organs = {}
    for o, rec in by_organ.items():
        elems = max(int(rec["elements"]), 1)
        organs[o] = {
            "elements": int(rec["elements"]),
            "storage_bytes": rec["storage_bytes"],
            "active_bytes": rec["active_bytes"],
            "storage_bpw": 8.0 * rec["storage_bytes"] / elems,
            "active_fused_bpw": 8.0 * rec["active_bytes"] / elems,
            "weighted_scale_aware": rec["w_sa"] / rec["w"] if rec["w"] else None,
        }
    return {
        "weighted_scale_aware": num_sa / den if den else 0.0,
        "weighted_damage": num_dmg / den if den else 1.0,
        "storage_bytes": storage,
        "active_bytes": active,
        "storage_ebpw_alloc_set": 8.0 * storage / max(sum(it["elements"] for it in items), 1),
        "n_items": len(items),
        "min_storage_bpw": min(bpws) if bpws else None,
        "max_storage_bpw": max(bpws) if bpws else None,
        "level_histogram": dict(sorted(by_name.items(), key=lambda kv: -kv[1])),
        "organs": organs,
    }


def uniform_assignment(items: list[dict], bpw: float) -> list[dict]:
    out = []
    for it in items:
        lv = interpolate_level(it["levels"], bpw)
        # Rebill storage at the exact target bpw so bytes equal  bpw * elements / 8.
        elems = int(it["elements"])
        lv = dict(lv)
        lv["storage_bpw"] = float(bpw)
        lv["storage_bytes"] = storage_bytes_of(elems, bpw)
        if it.get("embed_table"):
            lv["active_bytes"] = HIDDEN * float(bpw) / 8.0
            lv["active_fused_bpw"] = float(bpw)
        else:
            lv["active_bytes"] = lv["storage_bytes"]
            lv["active_fused_bpw"] = float(bpw)
        out.append(lv)
    return out


def per_organ_uniform_assignment(items: list[dict], floors: dict[str, float]) -> list[dict]:
    out = []
    for it in items:
        organ = it["organ"]
        key = "embedding_output" if organ in {"embedding", "embedding_output"} else organ
        bpw = float(floors[key])
        out.append(uniform_assignment([it], bpw)[0])
    return out


def predicted_token_ns(active_bytes: float) -> dict:
    """DERIVED from GPU_LEDGER. Not a new Metal run."""
    scale = float(active_bytes) / float(ACTIVE_INCUMBENT)
    ns = float(TOKEN_NS_INCUMBENT) * scale
    achieved_gb_s = (ACTIVE_INCUMBENT / (TOKEN_NS_INCUMBENT / 1e9)) / 1e9
    d_ns_d_byte = float(TOKEN_NS_INCUMBENT) / float(ACTIVE_INCUMBENT)
    # tps = 1e9 / ns; d(tps)/d(byte) = -1e9 / ns^2 * d_ns/d_byte
    d_tps_d_byte = -1e9 / (ns * ns) * d_ns_d_byte if ns else None
    return {
        "token_ns": ns,
        "tok_s": 1e9 / ns if ns else None,
        "status": "DERIVED",
        "null": "a GPU measurement of this mix; this is bytes/incumbent-TOKEN_NS",
        "source": GPU_LEDGER,
        "incumbent_achieved_gb_s": achieved_gb_s,
        "roof_gb_s": ROOF_GB_S,
        "d_token_ns_per_active_byte": d_ns_d_byte,
        "d_tok_s_per_active_byte": d_tps_d_byte,
        "note": (
            "Incumbent measured 448 GB/s against a 595.9 GB/s roof. Scaling "
            "TOKEN_NS with active bytes is the bandwidth-bound model GPU_LEDGER "
            "ranks above stored size. Positive extra active bytes LOWER tok/s."
        ),
    }


def marginals_along_curve(levels: list[dict]) -> list[dict]:
    """Capability gain per model-specific byte and throughput change per active byte."""
    pts = sorted(dense_levels(levels), key=lambda x: float(x["storage_bytes"]))
    out = []
    for lo, hi in zip(pts, pts[1:]):
        db = float(hi["storage_bytes"]) - float(lo["storage_bytes"])
        da = float(hi["active_bytes"]) - float(lo["active_bytes"])
        d_cap = float(hi["scale_aware"]) - float(lo["scale_aware"])
        cap_per_byte = d_cap / db if db > 0 else None
        # throughput: tok/s falls as active bytes rise
        thr = predicted_token_ns(float(hi["active_bytes"]))
        thr_lo = predicted_token_ns(float(lo["active_bytes"]))
        d_tps = (thr["tok_s"] or 0) - (thr_lo["tok_s"] or 0)
        tps_per_active = d_tps / da if da else None
        out.append(
            {
                "from": lo["name"],
                "to": hi["name"],
                "delta_storage_bytes": db,
                "delta_active_bytes": da,
                "delta_scale_aware": d_cap,
                "marginal_capability_per_model_specific_byte": cap_per_byte,
                "marginal_tok_s_per_active_byte": tps_per_active,
                "null_capability": "constant-mean output; a step that does not beat the cheaper level's scale_aware is not a gain",
            }
        )
    return out


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def _ensure_torch() -> None:
    try:
        import torch  # noqa: F401

        return
    except ImportError:
        pass
    if VISION_PY.is_file() and Path(sys.executable).resolve() != VISION_PY.resolve():
        os.execv(str(VISION_PY), [str(VISION_PY), *sys.argv])
    sys.exit("torch required (tried sys python and ~/.grok-vision/bin/python)")


def _write(obj: dict) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    tmp = RECEIPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(j(obj), indent=2) + "\n")
    tmp.replace(RECEIPT)


def cap_idx(idx, cap: int, seed: int):
    import numpy as np

    idx = np.asarray(idx)
    if idx.size <= cap:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=cap, replace=False))


def quality_pack(sc: dict, acc: dict, *, name: str, family: str, n_w: int, extra=None) -> dict:
    st = float(acc["storage_bpw"])
    act = float(acc.get("active_fused_bpw", st))
    rec = {
        "name": name,
        "family": family,
        "storage_bpw": st,
        "active_fused_bpw": act,
        "active_cached_f16_bpw": F16_BPW,
        "storage_bytes": float(acc.get("storage_bits", st * n_w)) / 8.0,
        "active_bytes": float(act) * n_w / 8.0,
        "n_weights": int(n_w),
        "extra_bits": 0.0,  # storage_bpw already includes scales; do not double-count
        "scales_counted": True,
        "scale_aware": float(sc["scale_aware"]),
        "damage": damage_of(sc),
        "rel_fro": float(sc["rel_fro"]),
        "cosine": float(sc["cosine"]),
        "gain": float(sc["gain"]),
        "null": float(sc["null"]),
        "beats_null": bool(sc["beats_null"]),
        "surplus_over_null": float(sc["surplus_over_null"]),
        "q4_equivalent": bool(sc.get("q4_equivalent", False)),
        "island": False,
        "control": False,
    }
    if extra:
        rec.update(extra)
    return rec


def score_hat(W, Wh, X_hold, Y=None) -> dict:
    sc = eval_linear(W, Wh, X_hold, Y=Y)
    return sc


def pick_functional_rows(Y, Yh, frac: float):
    import numpy as np

    err2 = ((Y - Yh).astype(np.float64) ** 2).sum(0)
    k = max(1, int(round(frac * Y.shape[1])))
    k = min(k, int(Y.shape[1]))
    return np.argpartition(err2, -k)[-k:]


def pick_functional_cols(W, Wh, d, frac: float):
    import numpy as np

    r = (W - Wh).astype(np.float64)
    score = (r * r).sum(0) * d.astype(np.float64)
    k = max(1, int(round(frac * W.shape[1])))
    k = min(k, int(W.shape[1]))
    return np.argpartition(score, -k)[-k:]


def apply_row_island(W, Wh_base, rows):
    import numpy as np

    Wh = np.array(Wh_base, copy=True)
    Wh[rows] = snap_f16(W[rows])
    return Wh


def apply_col_island(W, Wh_base, cols):
    import numpy as np

    Wh = np.array(Wh_base, copy=True)
    Wh[:, cols] = snap_f16(W[:, cols])
    return Wh


def measure_curve(W, X_fit, X_hold, *, organ: str, tensor: str, do_rank: bool, route: bool):
    """Function-space RD curve on real X_hold. Never Gaussian."""
    import numpy as np

    n_w = int(W.size)
    d = diag_energy(X_fit)
    Y = x_wt(X_hold, W)
    levels = []
    kept = {}

    specs = [
        ("binary_aa_g64", "binary", lambda: binary_meanabs(W, 64, d)),
        ("ternary_aa_g64", "ternary", lambda: ternary_fit(W, 64, d)),
        ("q2_fs_g64", "grouped_absmax", lambda: aa_diag_scale(W, 2, 64, d)),
        ("q3_fs_g64", "grouped_absmax", lambda: aa_diag_scale(W, 3, 64, d)),
        ("q4_fs_g128", "grouped_absmax", lambda: aa_diag_scale(W, 4, 128, d)),
        ("q4_fs_g64", "grouped_absmax", lambda: aa_diag_scale(W, 4, 64, d)),
    ]
    for name, family, fn in specs:
        if W.shape[1] % (128 if "g128" in name else 64) != 0:
            continue
        Wh, acc = fn()
        sc = score_hat(W, Wh, X_hold, Y=Y)
        pack = quality_pack(sc, acc, name=name, family=family, n_w=n_w)
        print(
            f"    {organ} {tensor} {name} sa={pack['scale_aware']:.4f} "
            f"gain={pack['gain']:.3f} rel={pack['rel_fro']:.3f} "
            f"st={pack['storage_bpw']:.3f} q4={pack['q4_equivalent']}",
            flush=True,
        )
        levels.append(pack)
        if name in {"ternary_aa_g64", "q4_fs_g64"}:
            kept[name] = Wh
        else:
            del Wh

    if do_rank:
        max_k = max(RANK_KS)
        V, s, fro2, method = input_pcs(
            X_fit, max_k, seed=SEED + (sum(ord(c) for c in tensor) % 10_000)
        )
        rows, cols = W.shape
        energy = s.astype("float64") ** 2
        total = float(fro2) + 1e-30
        for k in RANK_KS:
            if k > V.shape[1]:
                continue
            Wh = aa_rank_hat(W, V, k)
            acc = bill_factors(rows, cols, k)
            sc = score_hat(W, Wh, X_hold, Y=Y)
            pack = quality_pack(
                sc,
                acc,
                name=f"aa_rank_{k}",
                family="activation_aware_lowrank",
                n_w=n_w,
                extra={
                    "rank": int(k),
                    "input_energy_captured": float(energy[:k].sum()) / total,
                    "pca_method": method,
                    "null_energy_uniform": k / cols,
                },
            )
            print(
                f"    {organ} {tensor} aa_rank_{k} sa={pack['scale_aware']:.4f} "
                f"st={pack['storage_bpw']:.4f}",
                flush=True,
            )
            levels.append(pack)
            del Wh

    # identity (curve upper bound; not a legal greedy pick)
    Wh_id = snap_f16(W)
    sc_id = score_hat(W, Wh_id, X_hold, Y=Y)
    acc_id = {
        "storage_bpw": F16_BPW,
        "active_fused_bpw": F16_BPW,
        "storage_bits": F16_BPW * n_w,
        "scale_bits": 0.0,
    }
    ident = quality_pack(sc_id, acc_id, name="identity_f16", family="reference", n_w=n_w)
    ident["control"] = False  # on the dense curve for interpolation
    levels.append(ident)
    del Wh_id

    # 0.01*W trap — control, not a candidate
    sc_trap = score_hat(W, (SCALE_TRAP * W).astype("float32"), X_hold, Y=Y)
    trap = quality_pack(
        sc_trap,
        {"storage_bpw": F16_BPW, "active_fused_bpw": F16_BPW, "storage_bits": F16_BPW * n_w, "scale_bits": 0.0},
        name="scale_001W",
        family="control",
        n_w=n_w,
        extra={"control": True, "trap": True},
    )
    levels.append(trap)
    zero_sc = score_hat(W, W * 0, X_hold, Y=Y)
    zero = quality_pack(
        zero_sc,
        {"storage_bpw": 0.0, "active_fused_bpw": 0.0, "storage_bits": 0.0, "scale_bits": 0.0},
        name="zero",
        family="control",
        n_w=n_w,
        extra={"control": True, "deletion": True},
    )
    levels.append(zero)

    # Protected islands on ternary (cheap dominant) and, for route structures, on q4.
    island_bases = ["ternary_aa_g64"]
    if route:
        island_bases.append("q4_fs_g64")
    for base_name in island_bases:
        Wh_b = kept.get(base_name)
        if Wh_b is None:
            continue
        Yh_b = x_wt(X_hold, Wh_b)
        rows = pick_functional_rows(Y, Yh_b, ISLAND_FRAC)
        Wh_r = apply_row_island(W, Wh_b, rows)
        sc_r = score_hat(W, Wh_r, X_hold, Y=Y)
        cheap = next(lv for lv in levels if lv["name"] == base_name)
        n_kept = int(rows.size) * int(W.shape[1])
        acc_r = bill_island(n_w, n_kept, cheap["storage_bpw"], int(rows.size))
        pack_r = quality_pack(
            sc_r,
            acc_r,
            name=f"island_rows_{base_name}_{ISLAND_FRAC:.3f}",
            family="protected_island",
            n_w=n_w,
            extra={
                "island": True,
                "island_kind": "output_channels",
                "island_frac": ISLAND_FRAC,
                "n_island": int(rows.size),
                "n_rows": int(W.shape[0]),
                "base": base_name,
                "selection": "functional residual energy of Y[:,r] on hold",
            },
        )
        d_cap = pack_r["scale_aware"] - cheap["scale_aware"]
        d_b = pack_r["storage_bytes"] - cheap["storage_bytes"]
        pack_r["marginal_capability_per_byte"] = (d_cap / d_b) if d_b > 0 else None
        pack_r["base_scale_aware"] = cheap["scale_aware"]
        print(
            f"    {organ} {tensor} {pack_r['name']} sa={pack_r['scale_aware']:.4f} "
            f"dsa={d_cap:.5f} st={pack_r['storage_bpw']:.3f} "
            f"marg={pack_r['marginal_capability_per_byte']}",
            flush=True,
        )
        levels.append(pack_r)
        del Wh_r

        cols = pick_functional_cols(W, Wh_b, d, ISLAND_FRAC)
        Wh_c = apply_col_island(W, Wh_b, cols)
        sc_c = score_hat(W, Wh_c, X_hold, Y=Y)
        n_kept_c = int(cols.size) * int(W.shape[0])
        acc_c = bill_island(n_w, n_kept_c, cheap["storage_bpw"], int(cols.size))
        pack_c = quality_pack(
            sc_c,
            acc_c,
            name=f"island_cols_{base_name}_{ISLAND_FRAC:.3f}",
            family="protected_island",
            n_w=n_w,
            extra={
                "island": True,
                "island_kind": "input_channels",
                "island_frac": ISLAND_FRAC,
                "n_island": int(cols.size),
                "n_cols": int(W.shape[1]),
                "base": base_name,
                "selection": "d_j * ||W[:,j]-Wh[:,j]||^2  (function-space Hessian diag)",
            },
        )
        d_cap_c = pack_c["scale_aware"] - cheap["scale_aware"]
        d_b_c = pack_c["storage_bytes"] - cheap["storage_bytes"]
        pack_c["marginal_capability_per_byte"] = (d_cap_c / d_b_c) if d_b_c > 0 else None
        pack_c["base_scale_aware"] = cheap["scale_aware"]
        print(
            f"    {organ} {tensor} {pack_c['name']} sa={pack_c['scale_aware']:.4f} "
            f"dsa={d_cap_c:.5f} st={pack_c['storage_bpw']:.3f} "
            f"marg={pack_c['marginal_capability_per_byte']}",
            flush=True,
        )
        levels.append(pack_c)
        del Wh_c, Yh_b

    for Wh in kept.values():
        del Wh
    del Y, d
    return levels


def inventory_language(parent: Path) -> list[dict]:
    """2-D language GEMV + embed + lm_head from headers. No payloads."""
    index = json.loads((parent / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(index.values()))
    items = []
    for shard in shards:
        with open(parent / shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            if not (name.startswith("model.language_model.") or name == "lm_head.weight"):
                continue
            shape = tuple(meta["shape"])
            if len(shape) != 2:
                continue
            elems = int(shape[0]) * int(shape[1])
            if name.endswith("embed_tokens.weight"):
                items.append(
                    {
                        "name": name,
                        "cls": "embed_tokens.weight",
                        "organ": "embedding",
                        "layer": None,
                        "shape": list(shape),
                        "elements": elems,
                        "embed_table": True,
                    }
                )
                continue
            if name == "lm_head.weight":
                items.append(
                    {
                        "name": name,
                        "cls": "lm_head.weight",
                        "organ": "embedding_output",
                        "layer": 63,
                        "shape": list(shape),
                        "elements": elems,
                    }
                )
                continue
            parts = name.split(".")
            if "layers" not in parts:
                continue
            i = parts.index("layers")
            layer = int(parts[i + 1])
            cls = ".".join(parts[i + 2 :])
            organ = organ_of_class(cls)
            if organ == "other":
                continue
            items.append(
                {
                    "name": name,
                    "cls": cls,
                    "organ": organ,
                    "layer": layer,
                    "shape": list(shape),
                    "elements": elems,
                }
            )
    return items


def lerp_named_levels(a: list[dict], b: list[dict], t: float) -> list[dict]:
    by_b = {lv["name"]: lv for lv in b}
    out = []
    for lo in a:
        hi = by_b.get(lo["name"])
        if hi is None:
            out.append(dict(lo))
            continue
        rec = dict(lo)
        for k in (
            "storage_bpw",
            "active_fused_bpw",
            "storage_bytes",
            "active_bytes",
            "scale_aware",
            "damage",
            "rel_fro",
            "cosine",
            "gain",
        ):
            rec[k] = lerp(float(lo[k]), float(hi[k]), t)
        if lo.get("marginal_capability_per_byte") is not None and hi.get("marginal_capability_per_byte") is not None:
            rec["marginal_capability_per_byte"] = lerp(
                float(lo["marginal_capability_per_byte"]),
                float(hi["marginal_capability_per_byte"]),
                t,
            )
        rec["lerped_from_probes"] = True
        rec["t"] = t
        out.append(rec)
    return out


def curve_for_item(item: dict, probes: dict) -> list[dict] | None:
    """Nearest (or lerp-between) probed curve of the same class."""
    cls = item["cls"]
    if cls not in probes:
        return None
    by_layer = probes[cls]
    if item["layer"] is None:
        # embed
        if None in by_layer:
            return by_layer[None]
        if by_layer:
            return next(iter(by_layer.values()))
        return None
    layers = sorted(l for l in by_layer if l is not None)
    if not layers:
        return None
    L = int(item["layer"])
    if L in by_layer:
        return by_layer[L]
    if L <= layers[0]:
        return by_layer[layers[0]]
    if L >= layers[-1]:
        return by_layer[layers[-1]]
    lo = max(l for l in layers if l <= L)
    hi = min(l for l in layers if l >= L)
    if lo == hi:
        return by_layer[lo]
    t = (L - lo) / (hi - lo)
    return lerp_named_levels(by_layer[lo], by_layer[hi], t)


def rebilled(item: dict, levels: list[dict]) -> list[dict]:
    """Probe curve is per-element; rebill storage/active for THIS tensor's element count."""
    out = []
    n = int(item["elements"])
    ratio = n / float(levels[0]["n_weights"]) if levels and levels[0]["n_weights"] else 1.0
    for lv in levels:
        rec = dict(lv)
        bpw = float(lv["storage_bpw"])
        act = float(lv["active_fused_bpw"])
        rec["n_weights"] = n
        rec["storage_bytes"] = storage_bytes_of(n, bpw)
        if item.get("embed_table"):
            rec["active_bytes"] = HIDDEN * act / 8.0
        else:
            rec["active_bytes"] = storage_bytes_of(n, act)
        # island index bits scale with the axis length, not with a naive ratio
        if lv.get("island") and ratio != 1.0:
            n_id = int(round(float(lv.get("n_island") or 0) * ratio))
            kind = lv.get("island_kind")
            rows, cols = item["shape"]
            if kind == "output_channels":
                n_kept = n_id * cols
                n_index = n_id
            else:
                n_kept = n_id * rows
                n_index = n_id
            acc = bill_island(n, n_kept, float(lv.get("base") and 0 or lv["storage_bpw"]), n_index)
            # Prefer the already-interpolated bpw when the curve was lerped;
            # only recompute when we have a real island_frac and shape.
            if lv.get("island_frac") is not None and not lv.get("lerped_from_probes"):
                cheap = float(lv.get("base_scale_aware") and bpw)  # unused; keep interpolated
                rec["n_island"] = n_id
        rec["source_probe_n"] = int(lv["n_weights"])
        out.append(rec)
    return out


def measure_embed(parent, hold_ids, rare_hold) -> list[dict]:
    """Grouped codecs on a capture-aligned row slice. Bill is the full table."""
    import numpy as np

    print("  embed table (f16 gather, bill full vocab)", flush=True)
    W = load_tensor_f16(parent, "model.language_model.embed_tokens.weight")
    rng = np.random.default_rng(SEED)
    sample = rng.choice(VOCAB, size=4096, replace=False)
    ids = []
    if hold_ids:
        ids.append(np.asarray(hold_ids, dtype=np.int64))
    if rare_hold:
        ids.append(np.asarray(rare_hold, dtype=np.int64))
    ids.append(sample)
    slice_idx = np.unique(np.concatenate(ids))
    slice_idx = slice_idx[(slice_idx >= 0) & (slice_idx < VOCAB)]
    Wslice = W[slice_idx].astype(np.float32)
    n_w = VOCAB * HIDDEN
    levels = []
    for bits, g, name in ((2, 64, "q2_fs_g64"), (3, 64, "q3_fs_g64"), (4, 128, "q4_fs_g128"), (4, 64, "q4_fs_g64")):
        What, acc = ws_rtn(Wslice, bits, g)
        acc_full = bill_grouped(n_w, bits, VOCAB * (HIDDEN // g))
        sc = score_pair(Wslice, What)
        ok, reason = q4_healthy(sc)
        sc["q4_equivalent"] = ok
        sc["q4_reason"] = reason
        if rare_hold:
            rare_set = set(int(t) for t in rare_hold)
            pos = {int(t): i for i, t in enumerate(slice_idx.tolist())}
            ri = [pos[t] for t in rare_set if t in pos]
            if ri:
                rare_sc = score_pair(Wslice[ri], What[ri])
                ok_r, _ = q4_healthy(rare_sc)
                rare_sc["q4_equivalent"] = ok_r
                headline = rare_sc
            else:
                headline = sc
        else:
            headline = sc
        pack = quality_pack(headline, acc_full, name=name, family="grouped_absmax_table", n_w=n_w)
        pack["active_bytes"] = HIDDEN * pack["active_fused_bpw"] / 8.0
        pack["function_slice_n"] = int(slice_idx.size)
        pack["note"] = "row reconstruction on capture-aligned + random slice; bill is full table"
        print(
            f"    embedding embed {name} sa={pack['scale_aware']:.4f} st={pack['storage_bpw']:.3f}",
            flush=True,
        )
        levels.append(pack)
        del What
    # identity / trap on the slice
    ident = score_pair(Wslice, Wslice)
    ident["q4_equivalent"] = True
    acc16 = {"storage_bpw": F16_BPW, "active_fused_bpw": F16_BPW, "storage_bits": F16_BPW * n_w, "scale_bits": 0.0}
    p16 = quality_pack(ident, acc16, name="identity_f16", family="reference", n_w=n_w)
    p16["active_bytes"] = HIDDEN * 2.0
    levels.append(p16)
    trap = score_pair(Wslice, SCALE_TRAP * Wslice)
    ptrap = quality_pack(
        trap,
        acc16,
        name="scale_001W",
        family="control",
        n_w=n_w,
        extra={"control": True, "trap": True},
    )
    ptrap["active_bytes"] = HIDDEN * 2.0
    levels.append(ptrap)
    # binary/ternary on the slice so the dense curve covers the implied-floor region
    for tag, fn, fam in (
        ("binary_aa_g64", lambda: binary_meanabs(Wslice, 64, None), "binary"),
        ("ternary_aa_g64", lambda: ternary_fit(Wslice, 64, None), "ternary"),
    ):
        What, acc = fn()
        acc_full = bill_grouped(
            n_w,
            1.0 if "binary" in tag else (8.0 / 5.0),
            VOCAB * (HIDDEN // 64),
        )
        sc = score_pair(Wslice, What)
        ok, _ = q4_healthy(sc)
        sc["q4_equivalent"] = ok
        pack = quality_pack(sc, acc_full, name=tag, family=fam, n_w=n_w)
        pack["active_bytes"] = HIDDEN * pack["active_fused_bpw"] / 8.0
        levels.append(pack)
        print(f"    embedding embed {tag} sa={pack['scale_aware']:.4f} st={pack['storage_bpw']:.3f}", flush=True)
        del What
    del W, Wslice
    return levels


def measure_lm_head(parent, cap, hold_idx, hold_ids, fit_ids) -> list[dict]:
    import numpy as np

    print("  lm_head mix (observed+cold columns, L63 X)", flush=True)
    W = load_tensor_f16(parent, "lm_head.weight")
    X = load_X(cap, 63)
    Xh = X[hold_idx][: min(256, len(hold_idx))]
    rng = np.random.default_rng(SEED + 3)
    vocab_perm = rng.permutation(VOCAB)
    cold = vocab_perm[:4096]
    obs = np.unique(np.asarray((hold_ids or []) + (fit_ids or []), dtype=np.int64))
    if obs.size == 0:
        obs = vocab_perm[4096:8192]
    mix = np.unique(np.concatenate([obs[:4096], cold]))
    Wmix = W[mix].astype(np.float32)
    Y = x_wt(Xh, Wmix)
    n_w = VOCAB * HIDDEN
    d = diag_energy(Xh[: min(FIT_CAP, Xh.shape[0])])
    # d is over hidden, which is the column axis of Wmix. Good for aa_diag_scale.
    levels = []
    for bits, g, name in ((2, 64, "q2_fs_g64"), (3, 64, "q3_fs_g64"), (4, 128, "q4_fs_g128"), (4, 64, "q4_fs_g64")):
        Wh, acc = aa_diag_scale(Wmix, bits, g, d)
        sc = score_hat(Wmix, Wh, Xh, Y=Y)
        acc_full = bill_grouped(n_w, bits, VOCAB * (HIDDEN // g))
        pack = quality_pack(sc, acc_full, name=name, family="grouped_absmax", n_w=n_w)
        pack["vocab_mix_n"] = int(mix.size)
        pack["n_hold_rows"] = int(Xh.shape[0])
        print(
            f"    embedding_output lm_head {name} sa={pack['scale_aware']:.4f} "
            f"st={pack['storage_bpw']:.3f} q4={pack['q4_equivalent']}",
            flush=True,
        )
        levels.append(pack)
        del Wh
    for tag, fn, fam in (
        ("binary_aa_g64", lambda: binary_meanabs(Wmix, 64, d), "binary"),
        ("ternary_aa_g64", lambda: ternary_fit(Wmix, 64, d), "ternary"),
    ):
        Wh, acc = fn()
        acc_full = bill_grouped(
            n_w, 1.0 if "binary" in tag else (8.0 / 5.0), VOCAB * (HIDDEN // 64)
        )
        sc = score_hat(Wmix, Wh, Xh, Y=Y)
        pack = quality_pack(sc, acc_full, name=tag, family=fam, n_w=n_w)
        levels.append(pack)
        print(f"    embedding_output lm_head {tag} sa={pack['scale_aware']:.4f}", flush=True)
        del Wh
    ident = score_hat(Wmix, Wmix, Xh, Y=Y)
    acc16 = {"storage_bpw": F16_BPW, "active_fused_bpw": F16_BPW, "storage_bits": F16_BPW * n_w, "scale_bits": 0.0}
    levels.append(quality_pack(ident, acc16, name="identity_f16", family="reference", n_w=n_w))
    trap = score_hat(Wmix, (SCALE_TRAP * Wmix).astype("float32"), Xh, Y=Y)
    levels.append(
        quality_pack(
            trap, acc16, name="scale_001W", family="control", n_w=n_w, extra={"control": True, "trap": True}
        )
    )
    del W, Wmix, X, Y
    return levels


def island_verdict(items: list[dict], assignment: list[dict], dense_marg_floor: float | None) -> list[dict]:
    """Track island-specific marginal gain. An island that buys little is refused."""
    out = []
    for it, lv in zip(items, assignment):
        if not lv.get("island"):
            continue
        marg = lv.get("marginal_capability_per_byte")
        buys = True
        reason = "kept: marginal capability per byte is defined and positive"
        if marg is None:
            buys = False
            reason = "no measurable byte delta"
        elif marg <= 0:
            buys = False
            reason = "island does not raise scale_aware over its cheap base"
        elif dense_marg_floor is not None and marg < dense_marg_floor:
            buys = False
            reason = (
                f"marginal {marg:.4e} < dense-step floor {dense_marg_floor:.4e}; "
                "expensive exception, not a representation"
            )
        out.append(
            {
                "item": it["id"],
                "organ": it["organ"],
                "layer": it["layer"],
                "cls": it["cls"],
                "island": lv["name"],
                "storage_bpw": lv["storage_bpw"],
                "scale_aware": lv["scale_aware"],
                "base_scale_aware": lv.get("base_scale_aware"),
                "marginal_capability_per_byte": marg,
                "buys_enough": buys,
                "reason": reason,
                "route": it.get("route", False),
            }
        )
    return out


def strip_islands_from_assignment(items, assignment, refused_ids: set[str]) -> list[dict]:
    """If a selected island was judged too expensive, fall back to its base."""
    out = []
    for it, lv in zip(items, assignment):
        if it["id"] in refused_ids and lv.get("island"):
            base = lv.get("base")
            fallback = None
            for cand in legal_levels(it["levels"]):
                if cand["name"] == base:
                    fallback = cand
                    break
            if fallback is None:
                dens = [c for c in legal_levels(it["levels"]) if not c.get("island")]
                fallback = dens[-1] if dens else lv
            out.append(fallback)
        else:
            out.append(lv)
    return out


def main() -> int:
    t_all = time.time()
    _ensure_torch()
    try:
        import torch

        torch.set_num_threads(min(12, os.cpu_count() or 8))
        torch_s = f"{torch.__version__} mps={torch.backends.mps.is_available()}"
    except Exception as e:
        torch_s = f"unavailable ({e})"

    parent = find_parent()
    cap = find_capture()
    tok_path = find_tokenizer()
    inv = inventory_language(parent)
    manifest = {}
    mp = cap / "manifest.json"
    if mp.is_file():
        manifest = json.loads(mp.read_text())
    X0 = load_X(cap, 0)
    n_tokens = int(X0.shape[0])
    fit_idx, hold_idx = split_from_manifest(manifest, n_tokens)
    del X0
    fit_use = cap_idx(fit_idx, FIT_CAP, SEED)
    hold_use = cap_idx(hold_idx, HOLD_CAP, SEED + 1)

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

    hold_ids = tok_pack.get("hold_ids") or []
    fit_ids = tok_pack.get("fit_ids") or []
    from collections import Counter

    freq = Counter(hold_ids)
    rare_hold = [t for t, c in freq.items() if c == 1]

    print("GLOBAL INFORMATION ALLOCATOR")
    print("=" * 72)
    print(f"git_head: {git_head()}")
    print(f"python:   {sys.executable}")
    print(f"torch:    {torch_s}")
    print(f"parent:   {parent}")
    print(f"capture:  {cap} n={n_tokens} fit={len(fit_idx)} hold={len(hold_idx)}")
    print(f"used:     fit={len(fit_use)} hold={len(hold_use)} (real hold split, capped)")
    print("teacher:  qualified parent BF16, one tensor at a time. no second 27B.")
    print(f"implied floor: {implied_floor_ebpw():.6f} EBPW")
    print()

    results = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git_head(),
        "python": sys.executable,
        "torch": torch_s,
        "parent": str(parent),
        "did_not_load_second_27b": True,
        "question": (
            "Does a GLOBAL allocation beat uniform allocation at equal total "
            "model-specific bytes, measured in function space on real held-out "
            "activations, once the non-MLP floors dominate the implied 2.9398 EBPW?"
        ),
        "why_this_is_the_obligation": {
            "cited_floors_storage_bpw": CITED_FLOOR_BPW,
            "cited_elements": CITED_FLOOR_ELEMS,
            "implied_whole_model_floor_ebpw": numbered(
                implied_floor_ebpw(),
                status="CITED_ARITHMETIC",
                null="a uniform-per-organ mix is not a measured whole-model floor",
                unit="ebpw",
                formula="sum(elems_o * floor_o) / N",
                source="ORGAN_FRONTIERS + NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64",
            ),
            "non_mlp_floor_if_mlp_zero_ebpw": numbered(
                non_mlp_floor_ebpw(),
                status="CITED_ARITHMETIC",
                null="not a proposal to delete the MLP",
                unit="ebpw",
                note="even if MLP went to zero bits the other three organs still cost this",
            ),
            "leader_ebpw": numbered(
                LEADER_EBPW,
                status="CITED",
                null="not this lane's measurement",
                source=LEADER_SOURCE,
            ),
            "leader_gap_over_implied_floor": LEADER_EBPW / implied_floor_ebpw() - 1.0,
            "mlp_survive_bpw_not_transferred": True,
            "mlp_fail_bpw": MLP_FAIL_BPW,
        },
        "capture": {
            "path": str(cap),
            "site": "post_attn_norm",
            "n_tokens": n_tokens,
            "n_fit": int(len(fit_idx)),
            "n_hold": int(len(hold_idx)),
            "n_fit_used": int(len(fit_use)),
            "n_hold_used": int(len(hold_use)),
            "hold_cap_rule": (
                f"real hold split from the capture manifest; uniformly subsampled "
                f"to {HOLD_CAP} rows with seed {SEED+1}. Not a Gaussian draw, not a fit leak."
            ),
            "hidden": HIDDEN,
            "not_gaussian": True,
            "not_llama_server_teacher": True,
            "split_rule": manifest.get("split_rule"),
            "tokenizer": str(tok_path) if tok_path else None,
        },
        "accounting_rules": {
            "scales_counted": True,
            "q4_g64_storage_bpw": grouped_bpw(4, 64),
            "q4_g128_storage_bpw": grouped_bpw(4, 128),
            "q3_g64_storage_bpw": grouped_bpw(3, 64),
            "q2_g64_storage_bpw": grouped_bpw(2, 64),
            "ternary_5in8_g64_storage_bpw": ternary_5in8_storage_bpw(64),
            "binary_g64_storage_bpw": binary_storage_bpw(64),
            "rule": (
                "A 16-bit scale per group of 64 is 0.25 bpw, not free. "
                "Report storage AND active, or neither. Island ids are 32-bit."
            ),
        },
        "quality": {
            "capability_proxy": "scale_aware = cosine * gain, on held-out real Y = X W^T",
            "damage": "1 - scale_aware",
            "null": "constant-mean output cosine",
            "rejects_0p01W": (
                "cosine(Y, 0.01Y) ~ 1, gain ~ 0.01, rel_fro ~ 0.99; GO uses "
                "scale_aware/gain/rel_fro, never cosine alone"
            ),
            "weight": "elements * organ_function_lost * q_inject(layer)/q_inject(0)",
            "q_inject_source": "tools/gravity_error_chain.py via gravity_allocator.Q_INJECT (CITED)",
            "function_lost_source": "receipts/headless/NOETIC_ORGAN_CENSUS.json ranking (CITED)",
        },
        "probes": {},
        "scale_trap": {},
        "what_i_watched_fail": [],
        "wall_s": None,
    }
    _write(results)

    # Scale trap on GQA q_proj L3 — required instrument, same site as ORGAN_FRONTIERS.
    print("## SCALE TRAP", flush=True)
    Wq = load_tensor(parent, tensor_name(3, "self_attn.q_proj.weight"))
    X3 = load_X(cap, 3)
    Y = x_wt(X3[hold_use], Wq)
    trap = score_pair(Y, SCALE_TRAP * Y)
    ident = score_pair(Y, Y)
    rejects = bool(trap["cosine"] > 0.99 and trap["gain"] < 0.05 and trap["rel_fro"] > 0.9)
    print(
        f"  identity cosine={ident['cosine']:.6f} gain={ident['gain']:.6f} rel_fro={ident['rel_fro']:.6f}",
        flush=True,
    )
    print(
        f"  0.01*W   cosine={trap['cosine']:.6f} gain={trap['gain']:.6f} "
        f"rel_fro={trap['rel_fro']:.6f} rejects={rejects}",
        flush=True,
    )
    results["scale_trap"] = {
        "site": "L3.self_attn.q_proj on real hold X",
        "identity": ident,
        "scaled_0p01": trap,
        "rejects_scaled_artifact": rejects,
        "pass_if": "cosine~1 and gain~0.01 and rel_fro~0.99. GO uses gain+rel_fro, never cosine alone.",
        "null": "a metric that accepts 0.01*W is not a GO metric",
    }
    results["what_i_watched_fail"].append(
        f"0.01*W on L3 q_proj: cosine={trap['cosine']:.6f} (blind) "
        f"gain={trap['gain']:.6f} rel_fro={trap['rel_fro']:.6f} (rejects={rejects})"
    )
    if not rejects:
        results["verdict"] = {"decision": "NO-GO", "reason": "scale trap failed"}
        results["wall_s"] = time.time() - t_all
        _write(results)
        return 2
    del Wq, X3, Y
    gc.collect()
    _write(results)

    probes: dict[str, dict] = defaultdict(dict)
    X_cache = {}

    def X_parts(layer: int):
        if layer not in X_cache:
            X_cache[layer] = load_X(cap, layer)
        X = X_cache[layer]
        return X[fit_use], X[hold_use]

    # ----- MLP -----
    print("\n## MLP", flush=True)
    for layer in MLP_PROBE:
        Xf, Xh = X_parts(layer)
        Wg = load_tensor(parent, tensor_name(layer, "mlp.gate_proj.weight"))
        Wu = load_tensor(parent, tensor_name(layer, "mlp.up_proj.weight"))
        Wd = load_tensor(parent, tensor_name(layer, "mlp.down_proj.weight"))
        for W, cls, do_rank in (
            (Wg, "mlp.gate_proj.weight", True),
            (Wu, "mlp.up_proj.weight", True),
        ):
            print(f"  L{layer} {cls} {tuple(W.shape)}", flush=True)
            probes[cls][layer] = measure_curve(
                W, Xf, Xh, organ="mlp", tensor=f"L{layer}.{cls}", do_rank=do_rank, route=False
            )
        # down_proj: real SwiGLU intermediate from captured X, not Gaussian.
        print(f"  L{layer} mlp.down_proj.weight via real SwiGLU(X)", flush=True)
        H_fit = _silu(x_wt(Xf, Wg)) * x_wt(Xf, Wu)
        H_hold = _silu(x_wt(Xh, Wg)) * x_wt(Xh, Wu)
        probes["mlp.down_proj.weight"][layer] = measure_curve(
            Wd, H_fit, H_hold, organ="mlp", tensor=f"L{layer}.mlp.down_proj.weight", do_rank=False, route=False
        )
        del Wg, Wu, Wd, H_fit, H_hold
        gc.collect()
    results["probe_progress"] = "mlp"
    _write(results)

    # ----- DeltaNet -----
    print("\n## DELTANET", flush=True)
    for layer in DN_PROBE:
        Xf, Xh = X_parts(layer)
        W_qkv = load_tensor(parent, tensor_name(layer, "linear_attn.in_proj_qkv.weight"))
        W_z = load_tensor(parent, tensor_name(layer, "linear_attn.in_proj_z.weight"))
        W_out = load_tensor(parent, tensor_name(layer, "linear_attn.out_proj.weight"))
        for W, cls, route in (
            (W_qkv, "linear_attn.in_proj_qkv.weight", True),
            (W_z, "linear_attn.in_proj_z.weight", True),
        ):
            print(f"  L{layer} {cls} {tuple(W.shape)}", flush=True)
            probes[cls][layer] = measure_curve(
                W, Xf, Xh, organ="deltanet", tensor=f"L{layer}.{cls}", do_rank=False, route=route
            )
        W_qkvz = fuse_q38_qkvz(W_qkv, W_z)
        Xof = deltanet_out_proxy(Xf, W_qkvz)
        Xoh = deltanet_out_proxy(Xh, W_qkvz)
        print(f"  L{layer} linear_attn.out_proj.weight via v*silu(z)", flush=True)
        probes["linear_attn.out_proj.weight"][layer] = measure_curve(
            W_out, Xof, Xoh, organ="deltanet", tensor=f"L{layer}.out_proj", do_rank=False, route=False
        )
        del W_qkv, W_z, W_out, W_qkvz, Xof, Xoh
        gc.collect()
    results["probe_progress"] = "deltanet"
    _write(results)

    # ----- GQA -----
    print("\n## GQA", flush=True)
    for layer in GQA_PROBE:
        Xf, Xh = X_parts(layer)
        Wq = load_tensor(parent, tensor_name(layer, "self_attn.q_proj.weight"))
        Wk = load_tensor(parent, tensor_name(layer, "self_attn.k_proj.weight"))
        Wv = load_tensor(parent, tensor_name(layer, "self_attn.v_proj.weight"))
        Wo = load_tensor(parent, tensor_name(layer, "self_attn.o_proj.weight"))
        for W, cls, route in (
            (Wq, "self_attn.q_proj.weight", False),
            (Wk, "self_attn.k_proj.weight", True),
            (Wv, "self_attn.v_proj.weight", True),
        ):
            print(f"  L{layer} {cls} {tuple(W.shape)}", flush=True)
            probes[cls][layer] = measure_curve(
                W, Xf, Xh, organ="gqa", tensor=f"L{layer}.{cls}", do_rank=False, route=route
            )
        Xof = gqa_out_proxy(Xf, Wq, Wv)
        Xoh = gqa_out_proxy(Xh, Wq, Wv)
        print(f"  L{layer} self_attn.o_proj.weight via gated-V proxy", flush=True)
        probes["self_attn.o_proj.weight"][layer] = measure_curve(
            Wo, Xof, Xoh, organ="gqa", tensor=f"L{layer}.o_proj", do_rank=False, route=False
        )
        del Wq, Wk, Wv, Wo, Xof, Xoh
        gc.collect()
    results["probe_progress"] = "gqa"
    _write(results)

    # ----- endpoints -----
    print("\n## EMBED / LM_HEAD", flush=True)
    probes["embed_tokens.weight"][None] = measure_embed(parent, hold_ids, rare_hold)
    gc.collect()
    probes["lm_head.weight"][63] = measure_lm_head(parent, cap, hold_use, hold_ids, fit_ids)
    gc.collect()

    # Drop raw X cache before allocation.
    X_cache.clear()
    gc.collect()

    # Compact probe dump: drop nothing essential, but do not store weights.
    probe_out = {}
    for cls, by_l in probes.items():
        probe_out[cls] = {
            str(L): {
                "n_levels": len(lvs),
                "levels": lvs,
                "marginals": marginals_along_curve(lvs),
            }
            for L, lvs in by_l.items()
        }
    results["probes"] = probe_out
    _write(results)

    # Build the global item list: searched GEMVs + endpoints. Tiny tensors held.
    alloc_items = []
    held_small = []
    missing_curve = []
    for raw in inv:
        if raw["elements"] < MIN_SEARCH_ELEMS and raw["cls"] not in {
            "embed_tokens.weight",
            "lm_head.weight",
        }:
            held_small.append(raw)
            continue
        curve = curve_for_item(raw, probes)
        if curve is None:
            missing_curve.append(raw)
            continue
        levels = rebilled(raw, curve)
        organ = raw["organ"]
        layer = raw["layer"]
        if organ == "embedding":
            wlayer = 0
        elif organ == "embedding_output":
            wlayer = 63
        else:
            wlayer = layer
        iid = raw["name"]
        alloc_items.append(
            {
                "id": iid,
                "cls": raw["cls"],
                "organ": organ,
                "layer": layer,
                "shape": raw["shape"],
                "elements": raw["elements"],
                "embed_table": bool(raw.get("embed_table")),
                "route": raw["cls"] in ROUTE_CLASSES,
                "weight": item_weight(raw["elements"], organ, wlayer),
                "q_inject": q_inject(wlayer),
                "function_lost": ORGAN_FUNCTION_LOST.get(organ, 1.0),
                "levels": levels,
            }
        )

    held_elems = sum(t["elements"] for t in held_small)
    held_bytes = storage_bytes_of(held_elems, F16_BPW)  # 1-D / tiny stay f16
    alloc_elems = sum(it["elements"] for it in alloc_items)
    results["inventory"] = {
        "language_2d_plus_tables": len(inv),
        "allocatable_items": len(alloc_items),
        "allocatable_elements": alloc_elems,
        "held_small_tensors": len(held_small),
        "held_small_elements": held_elems,
        "held_small_bytes_f16": held_bytes,
        "missing_curve": [t["name"] for t in missing_curve],
        "source_param_count": SOURCE_PARAM_COUNT,
        "visual_and_mtp_excluded": True,
        "note": "visual encoder and MTP are not in the 26.9B language count and are not allocated here",
    }

    B_floor_bits = implied_floor_bits()
    B_floor_bytes = B_floor_bits / 8.0
    # The cited floor includes small tensors at their organ floor bpw. Subtract
    # the f16-held tiny tensors from the searchable budget so equal-B is over
    # the same allocatable set as greedy.
    tiny_at_floor = sum(
        t["elements"]
        * CITED_FLOOR_BPW[
            "embedding_output"
            if t["organ"] in {"embedding", "embedding_output"}
            else t["organ"]
        ]
        / 8.0
        for t in held_small
    )
    budget = B_floor_bytes - tiny_at_floor

    print(f"\n## ALLOCATE  budget={budget:.0f} B  ({implied_floor_ebpw():.4f} EBPW implied)", flush=True)
    greedy = greedy_descend(alloc_items, budget)
    assignment = greedy["assignment"]

    # Dense-step marginal floor: median positive capability-per-byte of dense
    # steps actually taken. Islands below that are expensive exceptions.
    dense_margs = []
    for it in alloc_items:
        for row in marginals_along_curve(it["levels"]):
            m = row.get("marginal_capability_per_model_specific_byte")
            if m is not None and m > 0:
                dense_margs.append(m)
    dense_margs.sort()
    dense_floor = dense_margs[len(dense_margs) // 2] if dense_margs else None
    islands = island_verdict(alloc_items, assignment, dense_floor)
    refused = {row["item"] for row in islands if not row["buys_enough"]}
    if refused:
        assignment = strip_islands_from_assignment(alloc_items, assignment, refused)
        # After stripping, we may be under budget; that is fine (equal-B compare
        # uses the global's actual bytes as B).
    global_sum = summarize_alloc(alloc_items, assignment)
    global_bytes = global_sum["storage_bytes"] + held_bytes

    # Uniform at EQUAL bytes: every allocatable tensor at the same bpw.
    uniform_bpw = 8.0 * global_sum["storage_bytes"] / max(alloc_elems, 1)
    uni_assign = uniform_assignment(alloc_items, uniform_bpw)
    uni_sum = summarize_alloc(alloc_items, uni_assign)

    # Per-organ uniform (the implied-floor mix) rebilled onto this item set.
    organ_assign = per_organ_uniform_assignment(alloc_items, CITED_FLOOR_BPW)
    organ_sum = summarize_alloc(alloc_items, organ_assign)

    # Second comparison at the leader's 3.1393 EBPW, equal bytes.
    leader_bytes = LEADER_EBPW * SOURCE_PARAM_COUNT / 8.0 - tiny_at_floor
    greedy_leader = greedy_descend(alloc_items, leader_bytes)
    lead_assign = greedy_leader["assignment"]
    lead_islands = island_verdict(alloc_items, lead_assign, dense_floor)
    lead_refused = {row["item"] for row in lead_islands if not row["buys_enough"]}
    if lead_refused:
        lead_assign = strip_islands_from_assignment(alloc_items, lead_assign, lead_refused)
    lead_sum = summarize_alloc(alloc_items, lead_assign)
    lead_uni_bpw = 8.0 * lead_sum["storage_bytes"] / max(alloc_elems, 1)
    lead_uni = summarize_alloc(alloc_items, uniform_assignment(alloc_items, lead_uni_bpw))

    beats_uniform = (
        global_sum["weighted_scale_aware"] > uni_sum["weighted_scale_aware"]
        and global_sum["weighted_damage"] < uni_sum["weighted_damage"]
    )
    beats_organ = (
        global_sum["weighted_scale_aware"] > organ_sum["weighted_scale_aware"]
        and global_sum["weighted_damage"] < organ_sum["weighted_damage"]
    )
    beats_leader_uniform = (
        lead_sum["weighted_scale_aware"] > lead_uni["weighted_scale_aware"]
        and lead_sum["weighted_damage"] < lead_uni["weighted_damage"]
    )

    # Per-organ / per-layer histogram of the winning global mix.
    layer_bpw = defaultdict(list)
    for it, lv in zip(alloc_items, assignment):
        layer_bpw[(it["organ"], it["layer"])].append((it["elements"], lv["storage_bpw"], lv["name"]))
    layer_rows = []
    for (organ, layer), recs in sorted(layer_bpw.items(), key=lambda kv: (kv[0][0], kv[0][1] is None, kv[0][1] or -1)):
        e = sum(r[0] for r in recs)
        bits = sum(r[0] * r[1] for r in recs)
        names = sorted({r[2] for r in recs})
        layer_rows.append(
            {
                "organ": organ,
                "layer": layer,
                "elements": e,
                "storage_bpw": bits / e,
                "levels": names,
            }
        )

    global_active = global_sum["active_bytes"] + held_bytes
    uni_active = uni_sum["active_bytes"] + held_bytes
    thr_g = predicted_token_ns(global_active)
    thr_u = predicted_token_ns(uni_active)

    # Equal-bytes check.
    rel_bytes = abs(global_sum["storage_bytes"] - uni_sum["storage_bytes"]) / max(global_sum["storage_bytes"], 1)
    equal_bytes = rel_bytes < 1e-6

    # Hypersensitive structures: steepest last dense step (q3→q4 or similar).
    steep = []
    for it in alloc_items:
        margs = marginals_along_curve(it["levels"])
        if not margs:
            continue
        last = margs[-1]
        steep.append(
            (
                last.get("marginal_capability_per_model_specific_byte") or 0.0,
                it["id"],
                it["organ"],
                it["layer"],
                it["cls"],
                last,
            )
        )
    steep.sort(reverse=True)
    hypersensitive = [
        {
            "item": s[1],
            "organ": s[2],
            "layer": s[3],
            "cls": s[4],
            "marginal_capability_per_byte_at_rich_end": s[0],
            "step": s[5],
            "route": s[4] in ROUTE_CLASSES,
        }
        for s in steep[:16]
    ]

    alloc_rows = []
    for it, lv in zip(alloc_items, assignment):
        alloc_rows.append(
            {
                "id": it["id"],
                "organ": it["organ"],
                "layer": it["layer"],
                "cls": it["cls"],
                "elements": it["elements"],
                "route": it["route"],
                "level": lv["name"],
                "storage_bpw": lv["storage_bpw"],
                "active_fused_bpw": lv["active_fused_bpw"],
                "storage_bytes": lv["storage_bytes"],
                "active_bytes": lv["active_bytes"],
                "scale_aware": lv["scale_aware"],
                "damage": lv["damage"],
                "null": lv["null"],
                "island": bool(lv.get("island")),
                "q4_equivalent": bool(lv.get("q4_equivalent")),
            }
        )

    results["allocation"] = {
        "budget_rule": (
            "implied whole-model floor bytes, minus tiny tensors held at f16. "
            "Greedy starts at the richest legal grouped/rank/island level and "
            "drops the lowest weighted-damage-per-byte until it meets the budget."
        ),
        "budget_bytes_allocatable": budget,
        "held_small_bytes_f16": held_bytes,
        "greedy": {
            "n_steps": len(greedy["steps"]),
            "hit_budget": greedy["hit_budget"],
            "summary": global_sum,
            "items": alloc_rows,
            "layer_bpw": layer_rows,
        },
        "protected_islands": {
            "frac": ISLAND_FRAC,
            "selection": (
                "output channels: residual energy of Y[:,r] on hold; "
                "input channels: d_j * ||W[:,j]-q(W)[:,j]||^2 with d = diag(X_fit^T X_fit). "
                "Not magnitude. Not Gaussian."
            ),
            "dense_marginal_median": dense_floor,
            "measured": [
                {
                    "item": it["id"],
                    "organ": it["organ"],
                    "layer": it["layer"],
                    "cls": it["cls"],
                    "name": lv["name"],
                    "storage_bpw": lv["storage_bpw"],
                    "scale_aware": lv["scale_aware"],
                    "base_scale_aware": lv.get("base_scale_aware"),
                    "marginal_capability_per_byte": lv.get("marginal_capability_per_byte"),
                    "route": it["route"],
                }
                for it in alloc_items
                for lv in it["levels"]
                if lv.get("island")
            ],
            "selected": islands,
            "refused_as_expensive_exception": sorted(refused),
            "rule": (
                "an island whose marginal capability per byte is <= 0 or below the "
                "median dense-step marginal is refused and replaced by its cheap base"
            ),
        },
        "hypersensitive_structures": hypersensitive,
        "route_structures": sorted(ROUTE_CLASSES),
    }
    results["comparison"] = {
        "equal_total_model_specific_bytes": equal_bytes,
        "relative_byte_gap": rel_bytes,
        "B_is": "global allocation's actual storage_bytes (uniform is interpolated to the same bpw)",
        "uniform": {
            "rule": "every allocatable tensor at the same storage_bpw, interpolated on its dense RD curve",
            "storage_bpw": uniform_bpw,
            "summary": uni_sum,
        },
        "per_organ_uniform": {
            "rule": "each organ at its cited floor bpw (the implied-floor mix)",
            "floors": CITED_FLOOR_BPW,
            "summary": organ_sum,
            "note": "this is the allocation the floors themselves imply; it is a baseline, not a prior on greedy",
        },
        "global_beats_uniform": beats_uniform,
        "global_beats_per_organ_uniform": beats_organ,
        "delta_weighted_scale_aware_vs_uniform": (
            global_sum["weighted_scale_aware"] - uni_sum["weighted_scale_aware"]
        ),
        "delta_weighted_damage_vs_uniform": (
            global_sum["weighted_damage"] - uni_sum["weighted_damage"]
        ),
        "delta_weighted_scale_aware_vs_per_organ": (
            global_sum["weighted_scale_aware"] - organ_sum["weighted_scale_aware"]
        ),
        "at_leader_ebpw": {
            "leader_ebpw": LEADER_EBPW,
            "global": lead_sum,
            "uniform": lead_uni,
            "global_beats_uniform": beats_leader_uniform,
        },
    }
    results["throughput"] = {
        "model": predicted_token_ns(global_active),
        "global_active_bytes_per_token": global_active,
        "uniform_active_bytes_per_token": uni_active,
        "global_storage_bytes": global_bytes,
        "uniform_storage_bytes": uni_sum["storage_bytes"] + held_bytes,
        "global_storage_ebpw": 8.0 * global_bytes / SOURCE_PARAM_COUNT,
        "uniform_storage_ebpw": 8.0 * (uni_sum["storage_bytes"] + held_bytes) / SOURCE_PARAM_COUNT,
        "global_active_ebpw": 8.0 * global_active / SOURCE_PARAM_COUNT,
        "uniform_active_ebpw": 8.0 * uni_active / SOURCE_PARAM_COUNT,
        "d_tok_s_global_minus_uniform": (thr_g["tok_s"] or 0) - (thr_u["tok_s"] or 0),
        "status": "DERIVED",
        "null": "a Metal complete-wall of this mix; GPU_LEDGER measured the q4 incumbent only",
    }
    results["marginal_per_organ"] = {
        o: {
            "storage_bpw": rec["storage_bpw"],
            "active_fused_bpw": rec["active_fused_bpw"],
            "weighted_scale_aware": rec["weighted_scale_aware"],
            "elements": rec["elements"],
            "null": "constant-mean output on the organ's probed sites",
        }
        for o, rec in global_sum["organs"].items()
    }

    reading = []
    if beats_uniform:
        reading.append(
            f"GLOBAL beats whole-model uniform at equal bytes "
            f"(weighted scale_aware {global_sum['weighted_scale_aware']:.6f} vs "
            f"{uni_sum['weighted_scale_aware']:.6f}). The floors are tradeable."
        )
    else:
        reading.append(
            "GLOBAL does not beat whole-model uniform at equal bytes. "
            "That is a real result: the measured RD curves do not justify a mix."
        )
    if beats_organ:
        reading.append(
            "GLOBAL also beats the implied-floor per-organ mix at the same total "
            "bytes — layer- and tensor-heterogeneity (and any island that paid) "
            "is worth spending against a uniform-per-organ budget."
        )
    else:
        reading.append(
            "GLOBAL does not beat the implied-floor per-organ mix. The floors "
            "behave as locally binding; trading bytes across organs did not help "
            "on this capture."
        )
    bpw_span = (global_sum["max_storage_bpw"] or 0) - (global_sum["min_storage_bpw"] or 0)
    reading.append(
        f"Allocated storage_bpw spans {global_sum['min_storage_bpw']:.3f} .. "
        f"{global_sum['max_storage_bpw']:.3f} (span {bpw_span:.3f}). "
        "That is the point of a global allocator, not a rounding of a uniform budget."
    )

    results["verdict"] = {
        "decision": "GLOBAL_BEATS_UNIFORM" if beats_uniform else "UNIFORM_WINS_OR_TIES",
        "global_beats_uniform_at_equal_bytes": beats_uniform,
        "global_beats_per_organ_uniform": beats_organ,
        "equal_bytes": equal_bytes,
        "min_storage_bpw": global_sum["min_storage_bpw"],
        "max_storage_bpw": global_sum["max_storage_bpw"],
        "implied_floor_ebpw": implied_floor_ebpw(),
        "global_storage_ebpw": 8.0 * global_bytes / SOURCE_PARAM_COUNT,
        "reading": " ".join(reading),
        "did_not_load_second_27b": True,
        "not_gaussian": True,
        "scales_counted": True,
        "mlp_floor_not_transferred_as_a_prior": True,
    }
    results["citations"] = {
        "organ_frontiers": "receipts/headless/ORGAN_FRONTIERS.json",
        "organ_census": "receipts/headless/NOETIC_ORGAN_CENSUS.json",
        "mlp_survive": "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json",
        "mlp_fail": "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
        "leader": LEADER_SOURCE,
        "gpu_ledger": GPU_LEDGER,
        "q_inject": "tools/gravity_allocator.py Q_INJECT (gravity_error_chain)",
    }
    results["wall_s"] = time.time() - t_all
    _write(results)
    print()
    print(f"WROTE {RECEIPT}  wall={results['wall_s']:.1f}s")
    print(f"global vs uniform: beats={beats_uniform}  sa {global_sum['weighted_scale_aware']:.6f} vs {uni_sum['weighted_scale_aware']:.6f}")
    print(f"bpw span {global_sum['min_storage_bpw']:.3f} .. {global_sum['max_storage_bpw']:.3f}")
    print(f"storage EBPW {8.0 * global_bytes / SOURCE_PARAM_COUNT:.4f}  active EBPW {8.0 * global_active / SOURCE_PARAM_COUNT:.4f}")
    return 0 if beats_uniform else 1


# ---------------------------------------------------------------------------
# unit tests (also imported by test_global_allocator.py)
# ---------------------------------------------------------------------------

def test_implied_floor_arithmetic():
    ebpw = implied_floor_ebpw()
    assert abs(ebpw - 2.9398) < 5e-5, ebpw
    nm = non_mlp_floor_ebpw()
    assert abs(nm - 1.5082) < 5e-5, nm
    # even if MLP is free the other three still cost > 1.5 EBPW
    assert nm > 1.5
    assert ebpw < LEADER_EBPW
    assert grouped_bpw(4, 128) == CITED_FLOOR_BPW["deltanet"]
    assert grouped_bpw(4, 64) == CITED_FLOOR_BPW["gqa"]
    assert grouped_bpw(2, 64) == CITED_FLOOR_BPW["mlp"]


def test_grouped_bpw_counts_scales():
    assert abs(grouped_bpw(4, 64) - 4.25) < 1e-12
    assert abs(grouped_bpw(4, 128) - 4.125) < 1e-12
    assert abs(grouped_bpw(2, 64) - 2.25) < 1e-12
    assert abs(ternary_5in8_storage_bpw(64) - 1.85) < 1e-12
    assert grouped_bpw(4, 64) != 4.0


def test_island_billing_counts_ids_and_f16():
    n = 64 * 64
    n_kept = 4 * 64  # 4 rows
    acc = bill_island(n, n_kept, cheap_bpw=1.85, n_index=4)
    expect = (n_kept * 16.0 + (n - n_kept) * 1.85 + 4 * 32.0) / n
    assert abs(acc["storage_bpw"] - expect) < 1e-12
    assert acc["scales_counted"] is True
    assert acc["active_fused_bpw"] == acc["storage_bpw"]
    assert acc["active_cached_f16_bpw"] == F16_BPW


def _toy_item(iid, elems, organ, layer, curve):
    """curve: list of (name, bpw, scale_aware)."""
    levels = []
    for name, bpw, sa in curve:
        island = name.startswith("island")
        levels.append(
            {
                "name": name,
                "family": "toy",
                "storage_bpw": bpw,
                "active_fused_bpw": bpw,
                "active_cached_f16_bpw": F16_BPW,
                "storage_bytes": storage_bytes_of(elems, bpw),
                "active_bytes": storage_bytes_of(elems, bpw),
                "scale_aware": sa,
                "damage": 1.0 - sa,
                "rel_fro": 1.0 - sa,
                "cosine": min(1.0, sa + 0.05),
                "gain": sa,
                "null": 0.2,
                "n_weights": elems,
                "island": island,
                "control": False,
                "q4_equivalent": sa >= 0.99,
            }
        )
    return {
        "id": iid,
        "cls": iid,
        "organ": organ,
        "layer": layer,
        "shape": [elems, 1],
        "elements": elems,
        "weight": item_weight(elems, organ, layer),
        "levels": levels,
        "route": False,
    }


def test_greedy_beats_uniform_on_heterogeneous_rd():
    """Steep tensor + flat tensor, equal average bits: greedy must protect the steep one."""
    # A: needs bits (steep). B: almost insensitive.
    A = _toy_item(
        "steep",
        1000,
        "gqa",
        63,
        [("cheap", 2.0, 0.20), ("mid", 3.0, 0.70), ("rich", 4.25, 0.99)],
    )
    B = _toy_item(
        "flat",
        1000,
        "mlp",
        0,
        [("cheap", 2.0, 0.80), ("mid", 3.0, 0.85), ("rich", 4.25, 0.88)],
    )
    items = [A, B]
    # Budget = 3.125 bpw average → 2*1000*3.125/8 = 781.25 bytes.
    budget = storage_bytes_of(2000, 3.125)
    g = greedy_descend(items, budget)
    sm = summarize_alloc(items, g["assignment"])
    uni = summarize_alloc(items, uniform_assignment(items, 3.125))
    assert sm["storage_bytes"] <= budget + 1e-6
    assert sm["weighted_scale_aware"] > uni["weighted_scale_aware"]
    names = {it["id"]: lv["name"] for it, lv in zip(items, g["assignment"])}
    # Steep tensor should not be left at cheap if the flat one can give the bytes.
    assert names["steep"] != "cheap"


def test_metric_rejects_001W_and_cosine_does_not():
    import numpy as np

    rng = np.random.RandomState(1)
    Y = rng.randn(32, 64).astype(np.float32)
    Yh = (SCALE_TRAP * Y).astype(np.float32)
    sc = score_pair(Y, Yh)
    assert sc["cosine"] > 0.99
    assert sc["gain"] < 0.05
    assert sc["rel_fro"] > 0.9
    assert damage_of(sc) > 0.9
    ok, reason = q4_healthy(sc)
    assert ok is False
    ident = score_pair(Y, Y)
    assert ident["gain"] > 0.99
    assert damage_of(ident) < 1e-6


def test_q_inject_spans_depth():
    assert q_inject(63) / q_inject(0) > 10.0
    assert q_mult(0) == 1.0
    assert q_mult(63) > 10.0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--self-test", "--unit"):
        test_implied_floor_arithmetic()
        test_grouped_bpw_counts_scales()
        test_island_billing_counts_ids_and_f16()
        test_greedy_beats_uniform_on_heterogeneous_rd()
        test_metric_rejects_001W_and_cosine_does_not()
        test_q_inject_spans_depth()
        print("unit tests passed")
        sys.exit(0)
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        raise
