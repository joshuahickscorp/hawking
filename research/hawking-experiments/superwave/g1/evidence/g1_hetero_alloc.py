#!/usr/bin/env python3
"""Qwen3.8 heterogeneous bit allocator. CPU only. No pack, no GPU."""
from __future__ import annotations

import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/Users/scammermike/.claude-grok/worktrees/13-heterogeneous-allocation-20260817-105156")
MAIN = Path("/Users/scammermike/Downloads/hawking")
MANIFEST = MAIN / "workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json"
CAPTURE = MAIN / "workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
OUT = Path("/tmp/g1_hetero_alloc_out.json")

N_LAYERS = 64
HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
SOURCE_ELEMENTS = 26895998464  # catalog authority

# Cheap in-register rungs a packing run can consume.
# bits is the nominal integer width; physical includes group scales.
RUNGS = [
    {"bits": 1, "codec": "binary_g128", "group": 128},
    {"bits": 2, "codec": "ternary_t0.7_g128", "group": 128},
    {"bits": 3, "codec": "uniform_q3_g64", "group": 64},
    {"bits": 4, "codec": "uniform_q4_g64", "group": 64},
]
BIT_TO_CODEC = {r["bits"]: r["codec"] for r in RUNGS}

# Gravity evidence-derived class priors (GPT-OSS organ table), applied as
# residual-path importance ONLY. Quantization difficulty lives in e(b).
# Labeled PROXY — not Qwen3.8-measured Jacobians.
CLASS_PRIOR = {
    "embed": 1.5,
    "lm_head": 2.0,
    "mlp.gate_proj": 1.0,
    "mlp.up_proj": 1.1,
    "mlp.down_proj": 1.4,
    "dn.in_proj_qkvz": 1.3,
    "dn.in_proj_ba": 0.4,
    "dn.out_proj": 1.3,
    "gqa.q_proj": 1.3,
    "gqa.k_proj": 1.2,
    "gqa.v_proj": 1.2,
    "gqa.o_proj": 1.3,
}

# Protection floors. PROXY policy: lm_head is output-adjacent and unscored.
# Embed is the residual origin and unscored. ba is 0.088% mass — pin cheap.
MIN_BITS = {
    "lm_head": 3,
    "embed": 2,
    "dn.in_proj_ba": 2,
}
# Unconstrained (all GEMV min=1) is also solved for comparison.


def git_show(path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(REPO), "show", f"HEAD:{path}"])


def uniform_qn_bytes(n: int, bits: int, rank: int = 2, group: int = 64) -> int:
    """HQ30UQ4-faithful: header 32+4*rank, f16 scale/group, packed codes."""
    groups = (n + group - 1) // group
    header = 32 + 4 * rank
    return header + groups * 2 + groups * group * bits // 8


def binary_g128_bytes(n: int) -> int:
    # Calibrated to descent L0 gate: n=89128960 -> 12534021 (header 261).
    groups = (n + 127) // 128
    return 261 + groups * 2 + (n + 7) // 8


def ternary_g128_bytes(n: int) -> int:
    # Calibrated to descent L0 gate: n=89128960 -> 25067853.
    groups = (n + 127) // 128
    return 261 + groups * 2 + groups * 2 + (n * 2 + 7) // 8


def f32v2_bytes(n: int) -> int:
    return 8 + n * 4


def payload_bytes(n: int, bits: int, rank: int) -> int:
    if bits == 32:
        return f32v2_bytes(n)
    if bits == 1:
        return binary_g128_bytes(n)
    if bits == 2:
        return ternary_g128_bytes(n)
    if bits in (3, 4):
        return uniform_qn_bytes(n, bits, rank=rank, group=64)
    raise ValueError(bits)


def classify(name: str) -> str:
    if name.endswith("embed_tokens.weight"):
        return "embed"
    if name.endswith("lm_head.weight"):
        return "lm_head"
    if name.endswith("model.norm.weight"):
        return "final_norm"
    if "input_layernorm" in name:
        return "input_layernorm"
    if "post_attention_layernorm" in name:
        return "post_attention_layernorm"
    if "mlp.gate_proj" in name:
        return "mlp.gate_proj"
    if "mlp.up_proj" in name:
        return "mlp.up_proj"
    if "mlp.down_proj" in name:
        return "mlp.down_proj"
    if "linear_attn.in_proj_qkvz" in name:
        return "dn.in_proj_qkvz"
    if "linear_attn.in_proj_ba" in name:
        return "dn.in_proj_ba"
    if "linear_attn.out_proj" in name:
        return "dn.out_proj"
    if "linear_attn.conv1d" in name:
        return "dn.conv1d"
    if "linear_attn.A_log" in name:
        return "dn.A_log"
    if "linear_attn.dt_bias" in name:
        return "dn.dt_bias"
    if "linear_attn.norm" in name:
        return "dn.norm"
    if "self_attn.q_proj" in name:
        return "gqa.q_proj"
    if "self_attn.k_proj" in name:
        return "gqa.k_proj"
    if "self_attn.v_proj" in name:
        return "gqa.v_proj"
    if "self_attn.o_proj" in name:
        return "gqa.o_proj"
    if "self_attn.q_norm" in name:
        return "gqa.q_norm"
    if "self_attn.k_norm" in name:
        return "gqa.k_norm"
    raise KeyError(name)


def parse_layer(name: str) -> int | None:
    if ".layers." not in name:
        return None
    return int(name.split(".layers.")[1].split(".")[0])


def is_gqa(layer: int) -> bool:
    return (layer + 1) % 4 == 0


def load_descent_curves() -> dict:
    d = json.loads(git_show("receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json"))
    curves = {}  # (layer, role) -> {codec: row}
    for organ in d["organs"]:
        key = (int(organ["layer"]), organ["role"])
        curves[key] = {c["codec"]: c for c in organ["candidates"]}
    return {
        "curves": curves,
        "mass_fractions": d["mass_fractions"],
        "baseline": d["baseline"],
        "claim_boundary": d["claim_boundary"],
        "coherence_floor": d["coherence_floor"],
        "summary": d["summary"],
        "seal": d["seal_sha256"],
    }


def rung_error_from_candidate(cand: dict, role: str) -> tuple[float, str]:
    """Return (error, source_label). Prefer hold_output_rel_l2."""
    if cand.get("hold_output_rel_l2") is not None:
        return float(cand["hold_output_rel_l2"]), "MEASURED_hold_output_rel_l2"
    if cand.get("weight_rel_l2") is not None:
        return float(cand["weight_rel_l2"]), "PROXY_weight_rel_l2"
    raise KeyError(role)


CODEC_FOR_BITS = {
    1: "binary_g128",
    2: "ternary_t0.7_g128",
    3: "uniform_q3_g64",
    4: "uniform_q4_g64",
}


def role_for_class(cls: str) -> str | None:
    return {
        "mlp.gate_proj": "gate_proj",
        "mlp.up_proj": "up_proj",
        "mlp.down_proj": "down_proj",
        "dn.in_proj_qkvz": "attn_in",
        "dn.out_proj": "attn_out",
        "gqa.q_proj": "attn_in",
        "gqa.k_proj": "attn_in",
        "gqa.v_proj": "attn_in",
        "gqa.o_proj": "attn_out",
        "dn.in_proj_ba": "attn_in",
        "embed": None,
        "lm_head": None,
    }.get(cls)


def interpolate_error(curves: dict, layer: int, role: str, codec: str, mixer: str) -> tuple[float, str]:
    """Linear interpolate hold/weight error in layer index among measured layers of this role."""
    if mixer == "delta_net" and role in ("attn_in", "attn_out"):
        # Only L0 scored for DeltaNet in-proj (shape 10240x5120). PROXY all DN from L0.
        row = curves[(0, role)][codec]
        e, src = rung_error_from_candidate(row, role)
        return e, f"{src}+PROXY_all_dn_from_L0"

    measured = sorted(L for (L, r) in curves if r == role)
    if mixer == "gqa" and role in ("attn_in", "attn_out"):
        measured = [L for L in measured if is_gqa(L)]
    if layer in measured:
        e, src = rung_error_from_candidate(curves[(layer, role)][codec], role)
        return e, src
    left = max((L for L in measured if L < layer), default=None)
    right = min((L for L in measured if L > layer), default=None)
    if left is None and right is None:
        raise RuntimeError(f"no measured layers for {role}")
    if left is None:
        e, src = rung_error_from_candidate(curves[(right, role)][codec], role)
        return e, f"{src}+PROXY_extrap_from_L{right}"
    if right is None:
        e, src = rung_error_from_candidate(curves[(left, role)][codec], role)
        return e, f"{src}+PROXY_extrap_from_L{left}"
    e0, s0 = rung_error_from_candidate(curves[(left, role)][codec], role)
    e1, s1 = rung_error_from_candidate(curves[(right, role)][codec], role)
    t = (layer - left) / (right - left)
    return (1 - t) * e0 + t * e1, f"PROXY_interp_L{left}_L{right}:{s0}"


def mean_measured_error(curves: dict, codec: str, field: str) -> float:
    vals = []
    for (_L, role), cmap in curves.items():
        row = cmap[codec]
        if row.get(field) is not None:
            vals.append(float(row[field]))
    return float(np.mean(vals))


def load_activation_rms() -> dict:
    hidden_dir = CAPTURE / "hidden"
    rows = []
    for layer in range(N_LAYERS):
        path = hidden_dir / f"L{layer:02d}.f32"
        x = np.fromfile(path, dtype=np.float32)
        if x.size != 256 * HIDDEN:
            raise RuntimeError(f"{path} size {x.size}")
        x = x.reshape(256, HIDDEN).astype(np.float64)
        rms = float(np.sqrt(np.mean(x * x)))
        mean_abs = float(np.mean(np.abs(x)))
        rms_tok = np.sqrt(np.mean(x * x, axis=1))
        rows.append(
            {
                "layer": layer,
                "rms": rms,
                "mean_abs": mean_abs,
                "rms_p10": float(np.quantile(rms_tok, 0.10)),
                "rms_p90": float(np.quantile(rms_tok, 0.90)),
                "max_abs": float(np.max(np.abs(x))),
            }
        )
    mean_rms = float(np.mean([r["rms"] for r in rows]))
    for r in rows:
        r["rms_norm"] = r["rms"] / mean_rms
    return {"layers": rows, "mean_rms": mean_rms}


def act_for_tensor(cls: str, layer: int | None, act: dict) -> float:
    layers = act["layers"]
    if cls == "embed":
        return layers[0]["rms_norm"]
    if cls == "lm_head":
        return layers[63]["rms_norm"]
    if layer is None:
        return 1.0
    return layers[layer]["rms_norm"]


def build_organs(manifest: dict, descent: dict, act: dict) -> list[dict]:
    curves = descent["curves"]
    # PROXY fallback for embed/lm_head: mean weight_rel_l2 across scored organs.
    proxy_w = {
        bits: mean_measured_error(curves, CODEC_FOR_BITS[bits], "weight_rel_l2")
        for bits in (1, 2, 3, 4)
    }
    organs = []
    for row in manifest["tensors"]:
        name = row["name"]
        cls = classify(name)
        layer = parse_layer(name)
        n = int(row["elements"])
        rank = len(row["shape"])
        kind = row["kind"]
        pinned = kind == "f32"
        mixer = None
        if layer is not None:
            mixer = "gqa" if is_gqa(layer) else "delta_net"
        role = role_for_class(cls)
        errors = {}
        sources = {}
        if pinned:
            errors = {32: 0.0}
            sources = {32: "PINNED_f32_exact"}
            bits_opts = [32]
            min_bits = 32
        else:
            bits_opts = [1, 2, 3, 4]
            min_bits = MIN_BITS.get(cls, 1)
            for b in bits_opts:
                codec = CODEC_FOR_BITS[b]
                if role is None:
                    errors[b] = proxy_w[b]
                    sources[b] = "PROXY_mean_weight_rel_l2_over_scored_organs"
                else:
                    e, src = interpolate_error(curves, layer if layer is not None else 0, role, codec, mixer or "delta_net")
                    # k/v/z/ba use attn_in curve as PROXY (same X, different W)
                    if cls in ("gqa.k_proj", "gqa.v_proj", "dn.in_proj_ba"):
                        src = src + f"+PROXY_from_attn_in_for_{cls}"
                    if cls == "dn.in_proj_qkvz":
                        src = src + "+PROXY_fused_qkvz_scored_as_qkv_only"
                    if cls == "gqa.o_proj":
                        src = src + "+PROXY_gqa_o_from_attn_out_weight"
                    errors[b] = e
                    sources[b] = src
        s_act = act_for_tensor(cls, layer, act)
        s_class = CLASS_PRIOR.get(cls, 1.0)
        organs.append(
            {
                "name": name,
                "cls": cls,
                "layer": layer,
                "mixer": mixer,
                "elements": n,
                "rank": rank,
                "pinned": pinned,
                "bits_opts": bits_opts,
                "min_bits": min_bits,
                "errors": {str(k): v for k, v in errors.items()},
                "error_source": {str(k): v for k, v in sources.items()},
                "s_act": s_act,
                "s_class": s_class,
                "s_primary": s_act * s_class,
                "s_unit": 1.0,
                "bytes_at": {str(b): payload_bytes(n, b, rank) for b in bits_opts},
                "catalog_bytes": int(row["bytes"]),
                "catalog_kind": kind,
            }
        )
    return organs


def objective(organs: list[dict], assign: dict[str, int], weight_key: str) -> float:
    acc = 0.0
    for o in organs:
        b = assign[o["name"]]
        e = o["errors"][str(b)]
        s = o[weight_key] if not o["pinned"] else 0.0
        acc += (s * e) ** 2
    return acc


def total_bytes(organs: list[dict], assign: dict[str, int]) -> int:
    return sum(o["bytes_at"][str(assign[o["name"]])] for o in organs)


def complete_bpw(nbytes: int) -> float:
    return 8.0 * nbytes / SOURCE_ELEMENTS


def floor_assign(organs: list[dict], unconstrained: bool) -> dict[str, int]:
    a = {}
    for o in organs:
        if o["pinned"]:
            a[o["name"]] = 32
        else:
            a[o["name"]] = 1 if unconstrained else o["min_bits"]
    return a


def greedy(organs: list[dict], budget: int, weight_key: str, unconstrained: bool) -> dict:
    assign = floor_assign(organs, unconstrained)
    used = total_bytes(organs, assign)
    if used > budget:
        return {
            "feasible": False,
            "assign": assign,
            "bytes": used,
            "bpw": complete_bpw(used),
            "objective": objective(organs, assign, weight_key),
            "raises": 0,
            "reason": "floor_exceeds_budget",
        }
    name_ix = {o["name"]: o for o in organs}
    raises = 0
    while True:
        best = None
        best_u = 0.0
        for o in organs:
            if o["pinned"]:
                continue
            b = assign[o["name"]]
            opts = o["bits_opts"]
            if b >= opts[-1]:
                continue
            nxt = opts[opts.index(b) + 1]
            step = o["bytes_at"][str(nxt)] - o["bytes_at"][str(b)]
            if used + step > budget:
                continue
            e0 = o["errors"][str(b)]
            e1 = o["errors"][str(nxt)]
            s = o[weight_key]
            dobj = (s * e0) ** 2 - (s * e1) ** 2
            if step <= 0:
                continue
            u = dobj / step
            if u > best_u:
                best_u = u
                best = (o["name"], nxt, step, dobj)
        if best is None:
            break
        name, nxt, step, _ = best
        assign[name] = nxt
        used += step
        raises += 1
    return {
        "feasible": True,
        "assign": assign,
        "bytes": used,
        "bpw": complete_bpw(used),
        "objective": objective(organs, assign, weight_key),
        "raises": raises,
        "slack_bytes": budget - used,
    }


def uniform_mix_at_target(organs: list[dict], target_bpw: float, weight_key: str, unconstrained: bool) -> dict:
    """Every free tensor uses the same (lo,hi,p) mix so complete BPW == target.
    Pinned stay f32. Floors are honored by lifting lo if needed — if floors
    already exceed target, infeasible.
    """
    floor = floor_assign(organs, unconstrained)
    floor_bytes = total_bytes(organs, floor)
    floor_bpw = complete_bpw(floor_bytes)
    pinned_bytes = sum(o["bytes_at"]["32"] for o in organs if o["pinned"])
    free = [o for o in organs if not o["pinned"]]
    # Max (all 4)
    all4 = {o["name"]: (32 if o["pinned"] else 4) for o in organs}
    max_bpw = complete_bpw(total_bytes(organs, all4))
    if target_bpw + 1e-12 < floor_bpw:
        return {
            "feasible": False,
            "reason": "target_below_floor",
            "floor_bpw": floor_bpw,
            "target_bpw": target_bpw,
        }
    if target_bpw > max_bpw + 1e-12:
        return {"feasible": False, "reason": "target_above_max", "max_bpw": max_bpw}

    def bytes_if_all(bits: int) -> int:
        a = {o["name"]: (32 if o["pinned"] else max(bits, floor[o["name"]])) for o in organs}
        return total_bytes(organs, a)

    # Find adjacent rungs that bracket the target.
    rung_bpw = []
    for b in (1, 2, 3, 4):
        rung_bpw.append((b, complete_bpw(bytes_if_all(b))))
    # If floors force some tensors above `b`, bytes_if_all already accounts for it.
    lo = 1
    hi = 1
    for b, bpw in rung_bpw:
        if bpw <= target_bpw + 1e-15:
            lo = b
            hi = b
        else:
            hi = b
            break
    if lo == hi:
        # Exact rung (or target equals a rung after floors).
        assign = {o["name"]: (32 if o["pinned"] else max(lo, floor[o["name"]])) for o in organs}
        return {
            "feasible": True,
            "kind": "uniform_single_rung",
            "lo": lo,
            "hi": lo,
            "p_hi": 0.0,
            "assign_bits": lo,
            "bytes": total_bytes(organs, assign),
            "bpw": complete_bpw(total_bytes(organs, assign)),
            "objective": objective(organs, assign, weight_key),
        }

    b_lo = bytes_if_all(lo)
    b_hi = bytes_if_all(hi)
    # Want (1-p)*b_lo + p*b_hi = target_bytes
    target_bytes = target_bpw * SOURCE_ELEMENTS / 8.0
    p = (target_bytes - b_lo) / (b_hi - b_lo)
    p = min(1.0, max(0.0, p))
    # Expected error mix per tensor
    obj = 0.0
    for o in organs:
        if o["pinned"]:
            continue
        blo = max(lo, floor[o["name"]])
        bhi = max(hi, floor[o["name"]])
        e = (1 - p) * o["errors"][str(blo)] + p * o["errors"][str(bhi)]
        s = o[weight_key]
        obj += (s * e) ** 2
    return {
        "feasible": True,
        "kind": "uniform_two_rung_mix",
        "lo": lo,
        "hi": hi,
        "p_hi": p,
        "bytes": target_bytes,
        "bpw": target_bpw,
        "objective": obj,
        "note": "same mix on every free tensor; error linearly mixed in bit rungs",
    }


def summarize_assign(organs: list[dict], assign: dict[str, int]) -> dict:
    by_cls = defaultdict(lambda: Counter())
    by_layer_cls = {}
    for o in organs:
        b = assign[o["name"]]
        by_cls[o["cls"]][b] += 1
        key = (o["layer"] if o["layer"] is not None else -1, o["cls"])
        by_layer_cls[f"{key[0]}|{key[1]}"] = {
            "layer": o["layer"],
            "cls": o["cls"],
            "bits": b,
            "codec": "f32v2" if b == 32 else BIT_TO_CODEC[b],
            "elements": o["elements"],
            "bytes": o["bytes_at"][str(b)],
            "error": o["errors"][str(b)],
            "error_source": o["error_source"][str(b)],
            "s_primary": o["s_primary"],
            "mixer": o["mixer"],
        }
    hist = {cls: dict(sorted(cnt.items())) for cls, cnt in sorted(by_cls.items())}
    return {"class_bit_histogram": hist, "per_tensor": by_layer_cls}


def compact_layer_table(organs: list[dict], assign: dict[str, int]) -> list[dict]:
    """One row per layer: bit width per GEMV class. f32 omitted."""
    rows = []
    for layer in range(N_LAYERS):
        rec = {"layer": layer, "mixer": "gqa" if is_gqa(layer) else "delta_net"}
        for o in organs:
            if o["layer"] != layer or o["pinned"]:
                continue
            rec[o["cls"]] = assign[o["name"]]
        rows.append(rec)
    return rows


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["source_weight_elements"] == SOURCE_ELEMENTS
    descent = load_descent_curves()
    act = load_activation_rms()
    organs = build_organs(manifest, descent, act)

    # Sanity: catalog q4 bytes vs formula
    q4_formula = 0
    q4_catalog = 0
    f32_catalog = 0
    for o in organs:
        if o["catalog_kind"] == "q4":
            q4_formula += o["bytes_at"]["4"]
            q4_catalog += o["catalog_bytes"]
        else:
            f32_catalog += o["catalog_bytes"]
    catalog_bpw = 8.0 * manifest["tensor_payload_bytes"] / SOURCE_ELEMENTS

    # Measured curve extract for the report
    curve_table = []
    for (L, role), cmap in sorted(descent["curves"].items()):
        rec = {"layer": L, "role": role}
        for b, codec in CODEC_FOR_BITS.items():
            c = cmap[codec]
            rec[f"b{b}_codec"] = codec
            rec[f"b{b}_bpw"] = c["physical_bpw"]
            rec[f"b{b}_hold_l2"] = c.get("hold_output_rel_l2")
            rec[f"b{b}_hold_cos"] = c.get("hold_output_cosine")
            rec[f"b{b}_w_l2"] = c.get("weight_rel_l2")
            rec[f"b{b}_w_cos"] = c.get("weight_cosine")
        curve_table.append(rec)

    # Correlation weight vs hold on output-scored organs
    xs, ys = [], []
    for (L, role), cmap in descent["curves"].items():
        for codec in CODEC_FOR_BITS.values():
            c = cmap[codec]
            if c.get("hold_output_rel_l2") is not None:
                xs.append(c["weight_rel_l2"])
                ys.append(c["hold_output_rel_l2"])
    corr = float(np.corrcoef(xs, ys)[0, 1]) if xs else None

    targets = [2.0, 1.5, 1.2, 1.0]
    policies = [
        ("primary_floors", "s_primary", False),
        ("primary_unconstrained", "s_primary", True),
        ("unit_weight_floors", "s_unit", False),  # s_i = 1, still floors
    ]

    results = {}
    for target in targets:
        budget = int(math.floor(target * SOURCE_ELEMENTS / 8.0))
        results[str(target)] = {"budget_bytes": budget, "policies": {}}
        for pname, wkey, uncon in policies:
            g = greedy(organs, budget, wkey, uncon)
            u = uniform_mix_at_target(organs, target, wkey, uncon)
            pack = None
            if g["feasible"]:
                pack = {
                    "complete_physical_bpw": g["bpw"],
                    "tensor_payload_bytes": g["bytes"],
                    "slack_bytes": g["slack_bytes"],
                    "objective": g["objective"],
                    "raises": g["raises"],
                    "class_bit_histogram": summarize_assign(organs, g["assign"])["class_bit_histogram"],
                    "layer_table": compact_layer_table(organs, g["assign"]),
                    "per_tensor": summarize_assign(organs, g["assign"])["per_tensor"],
                }
            gap = None
            if g["feasible"] and u.get("feasible"):
                ju, jh = u["objective"], g["objective"]
                gap = {
                    "J_uniform": ju,
                    "J_hetero": jh,
                    "abs_reduction": ju - jh,
                    "rel_reduction": (ju - jh) / ju if ju > 0 else None,
                    "uniform": {k: u[k] for k in u if k != "assign"},
                }
            results[str(target)]["policies"][pname] = {
                "weight_key": wkey,
                "unconstrained": uncon,
                "greedy": {
                    "feasible": g["feasible"],
                    "bpw": g.get("bpw"),
                    "bytes": g.get("bytes"),
                    "objective": g.get("objective"),
                    "raises": g.get("raises"),
                    "reason": g.get("reason"),
                    "slack_bytes": g.get("slack_bytes"),
                },
                "uniform": u,
                "gap": gap,
                "recipe": pack,
            }

    # Also evaluate historical mixed-2p0-like class assignment on this objective
    # (gate=1, up=1, down=1, attn/embed=4) — discrete cheap stand-in, NOT rice/HGRAVS01.
    # And the actual mixed-2p0 bytes from the pack report (reference only).

    floor_u = floor_assign(organs, True)
    floor_f = floor_assign(organs, False)
    all4 = {o["name"]: (32 if o["pinned"] else 4) for o in organs}
    all1 = {o["name"]: (32 if o["pinned"] else 1) for o in organs}
    all2 = {o["name"]: (32 if o["pinned"] else 2) for o in organs}
    all3 = {o["name"]: (32 if o["pinned"] else 3) for o in organs}

    baselines = {}
    for label, a in [
        ("all_binary_plus_f32", all1),
        ("all_ternary_plus_f32", all2),
        ("all_q3_plus_f32", all3),
        ("all_q4_plus_f32", all4),
        ("floor_protected", floor_f),
        ("floor_unconstrained", floor_u),
    ]:
        baselines[label] = {
            "bytes": total_bytes(organs, a),
            "bpw": complete_bpw(total_bytes(organs, a)),
            "J_primary": objective(organs, a, "s_primary"),
            "J_act": objective(organs, a, "s_act"),
        }

    out = {
        "schema": "hawking.g1.qwen38_heterogeneous_allocation.v1",
        "model": "Qwen3.8-27B language-only fused-in_proj",
        "source_weight_elements": SOURCE_ELEMENTS,
        "catalog_complete_physical_bpw": catalog_bpw,
        "catalog_payload_bytes": manifest["tensor_payload_bytes"],
        "q4_formula_bytes": q4_formula,
        "q4_catalog_bytes": q4_catalog,
        "q4_formula_minus_catalog": q4_formula - q4_catalog,
        "f32_catalog_bytes": f32_catalog,
        "n_organs": len(organs),
        "n_free": sum(1 for o in organs if not o["pinned"]),
        "n_pinned_f32": sum(1 for o in organs if o["pinned"]),
        "activation": {
            "mean_rms": act["mean_rms"],
            "layers": act["layers"],
            "source": str(CAPTURE / "capture-result.json"),
        },
        "weight_vs_hold_l2_corr": corr,
        "n_hold_pairs": len(xs),
        "proxy_mean_weight_rel_l2": {
            str(b): mean_measured_error(descent["curves"], CODEC_FOR_BITS[b], "weight_rel_l2")
            for b in (1, 2, 3, 4)
        },
        "min_bits_policy": MIN_BITS,
        "class_prior": CLASS_PRIOR,
        "rungs": RUNGS,
        "baselines": baselines,
        "targets": results,
        "curve_table": curve_table,
        "descent_seal": descent["seal"],
        "descent_claim_boundary": descent["claim_boundary"],
        "descent_coherence_floor": descent["coherence_floor"],
        "mass_fractions": descent["mass_fractions"],
        "g0_baseline": descent["baseline"],
    }
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT, "bytes", OUT.stat().st_size)
    print("catalog_bpw", catalog_bpw)
    print("q4 formula-catalog", q4_formula - q4_catalog)
    print("corr(weight_l2, hold_l2)", corr)
    print("baselines:")
    for k, v in baselines.items():
        print(f"  {k:28s} bpw={v['bpw']:.6f} J={v['J_primary']:.6f}")
    for t, block in results.items():
        print(f"\nTARGET {t} budget={block['budget_bytes']}")
        for pname, p in block["policies"].items():
            g = p["greedy"]
            gap = p["gap"]
            print(
                f"  {pname:24s} feas={g['feasible']} bpw={g.get('bpw')} "
                f"J={g.get('objective')} raises={g.get('raises')} reason={g.get('reason')}"
            )
            if gap:
                print(
                    f"    vs uniform: Ju={gap['J_uniform']:.6f} Jh={gap['J_hetero']:.6f} "
                    f"rel={(gap['rel_reduction']*100 if gap['rel_reduction'] is not None else None)}"
                )
                print("    uniform:", {k: p["uniform"].get(k) for k in ("kind", "lo", "hi", "p_hi", "bpw", "feasible", "reason")})


if __name__ == "__main__":
    main()
