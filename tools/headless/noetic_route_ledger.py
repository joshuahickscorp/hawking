#!/usr/bin/env python3
"""RouteLedger — measure routing as a physical resource, on a real execution.

Qwen3.8 is dense. There is no MoE router. This script finds what the decode
path actually decides per token, measures that, and refuses to invent routes
to count.

The property the ledger must have: it can show a route that is NOT worth it.
A route replacing a million explicit values may be excellent; a route
replacing eight cheap values is stupid. ROUTES_PER_MILLION_PARENT_WEIGHTS is
always paired with its inverse PARENT_WEIGHT_EQUIVALENTS_PER_ROUTE.

Run:
    python3 tools/headless/noetic_route_ledger.py

Writes receipts/headless/NOETIC_ROUTE_LEDGER.json.

Does not spawn a second 27B. Selector kernels are timed on Metal at the
production launch geometry (tiny activations, no weight set). Token rate
comes from the llama-server already resident on 52484. Native complete-wall
figures are cited from the gravity receipt, not re-run.
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPT = REPO / "receipts/headless/NOETIC_ROUTE_LEDGER.json"
ARTIFACT = Path.home() / "models/qwen38-gravity-uniform-q4-v1"
LLAMA_PORT = int(os.environ.get("NOETIC_LLAMA_PORT", "52484"))
GEOMETRY_RS = REPO / "crates/hawking-core/src/model/qwen38_geometry.rs"
DECODE_RS = REPO / "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
SCHEDULE_RS = REPO / "crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs"
TOKEN_NS_RS = REPO / "crates/hawking-core/src/model/qwen38_token_ns_ledger.rs"
MHA_METAL = REPO / "crates/hawking-core/shaders/mha.metal"
SIGMOID_METAL = REPO / "crates/hawking-core/shaders/qwen38_device_activations.metal"
BA_METAL = REPO / "crates/hawking-core/shaders/qwen80_device_activations.metal"
KERNELS_RS = REPO / "crates/hawking-core/src/kernels/mod.rs"
SHADER_DIR = REPO / "crates/hawking-core/shaders"
NATIVE_RECEIPT = REPO / "receipts/headless/QWEN38_GRAVITY_NATIVE.json"

# Anchors (measured; do not re-derive).
ANCHOR_TPS = 32.73
ANCHOR_MS_PER_TOKEN = 30.606
ANCHOR_ROOF_GB_S = 778.8
ANCHOR_PARAM_COUNT = 26_895_998_464
ANCHOR_ARTIFACT_FILES = 756
ANCHOR_ARTIFACT_BYTES = 14_297_933_604
ANCHOR_TENSORS = 755
LIVE_27B_WARNING = (
    "GPU occupancy is NOT free: a native run measured 3.986 tok/s with two "
    "model servers resident against 33.47 with one. This process will not "
    "load the gravity artifact."
)

SCHEMA = "hawking.headless.noetic_route_ledger.v1"

# A route replacing >= 1e6 parent values AND skipping their compute is worth
# it. A route replacing < 64 values, or skipping nothing, is not. The numbers
# are the steer: "a million explicit values" vs "eight cheap values".
WORTH_IT_PARENT_WEIGHTS = 1_000_000
NOT_WORTH_IT_PARENT_WEIGHTS = 64


# --------------------------------------------------------------------------- helpers


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sh(cmd: str, timeout: int = 30) -> str:
    p = subprocess.run(
        ["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout
    )
    return (p.stdout or "").strip()


def git_sha() -> str:
    return sh(f"git -C {REPO} rev-parse HEAD") or "unknown"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def first_line(path: Path, needle: str, start_line: int = 1) -> dict:
    """Return {path, line, excerpt} for the first line containing needle."""
    rel = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
    if not path.is_file():
        return {"path": rel, "line": None, "excerpt": None, "missing": True}
    for i, line in enumerate(read_text(path).splitlines(), 1):
        if i < start_line:
            continue
        if needle in line:
            return {"path": rel, "line": i, "excerpt": line.strip()}
    return {"path": rel, "line": None, "excerpt": None, "missing": False}


def rust_usize_consts(path: Path, prefix: str = "QWEN38_") -> dict:
    out = {}
    pat = re.compile(
        rf"pub const ({re.escape(prefix)}\w+):\s*(?:usize|u32|i32)\s*=\s*([0-9_]+);"
    )
    for m in pat.finditer(read_text(path)):
        out[m.group(1)] = int(m.group(2).replace("_", ""))
    return out


def extract_kernel(src: str, name: str) -> str:
    """Slice one `kernel void name(...) { ... }` from a Metal translation unit."""
    key = f"kernel void {name}"
    start = src.find(key)
    if start < 0:
        raise RuntimeError(f"kernel {name} not in shader source")
    # Brace match from the first '{' after the signature.
    brace = src.find("{", start)
    depth = 0
    i = brace
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
        i += 1
    raise RuntimeError(f"kernel {name} brace mismatch")


def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def shannon_bits(counts) -> float:
    n = sum(counts)
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log2(p)
    return h


def dropped(reason: str, **extra) -> dict:
    row = {"measured": False, "dropped": True, "reason": reason, "value": None}
    row.update(extra)
    return row


def measured(value, **extra) -> dict:
    row = {"measured": True, "dropped": False, "value": value}
    row.update(extra)
    return row


# --------------------------------------------------------------------------- ledger core


def paired_rates(n_routes: float, parent_weights: float) -> dict:
    """Always pair ROUTES_PER_MILLION_PARENT_WEIGHTS with its inverse.

    Numerically awkward when one route governs a huge structure — that is
    why the inverse exists. A zero parent-weight denominator is not a
    rate of zero; it is unmeasurable as a rate and is dropped.
    """
    if n_routes < 0:
        raise ValueError("n_routes must be >= 0")
    if parent_weights <= 0:
        return {
            "n_routes": n_routes,
            "parent_weights": parent_weights,
            "ROUTES_PER_MILLION_PARENT_WEIGHTS": dropped(
                "parent_weights <= 0; a rate against nothing is not a measurement"
            ),
            "PARENT_WEIGHT_EQUIVALENTS_PER_ROUTE": dropped(
                "parent_weights <= 0; inverse is undefined, not zero"
            ),
        }
    if n_routes == 0:
        return {
            "n_routes": n_routes,
            "parent_weights": parent_weights,
            "ROUTES_PER_MILLION_PARENT_WEIGHTS": measured(
                0.0,
                unit="routes / 1e6 parent weights",
                note="zero routes against a real parent-weight denominator",
            ),
            "PARENT_WEIGHT_EQUIVALENTS_PER_ROUTE": dropped(
                "n_routes == 0; inverse would be inf and is dropped rather than faked"
            ),
        }
    rpm = n_routes / (parent_weights / 1e6)
    inv = parent_weights / n_routes
    return {
        "n_routes": n_routes,
        "parent_weights": parent_weights,
        "ROUTES_PER_MILLION_PARENT_WEIGHTS": measured(
            rpm, unit="routes / 1e6 parent weights"
        ),
        "PARENT_WEIGHT_EQUIVALENTS_PER_ROUTE": measured(
            inv, unit="parent weights / route"
        ),
    }


def verdict(parent_weight_equivalents, skips_parent_compute: bool) -> dict:
    """The ledger's job is to be able to say NOT_WORTH_IT."""
    if parent_weight_equivalents is None:
        return {
            "label": "UNMEASURABLE",
            "because": "parent-weight equivalents were dropped; no verdict from a missing rate",
        }
    eq = float(parent_weight_equivalents)
    if eq < NOT_WORTH_IT_PARENT_WEIGHTS:
        label = "NOT_WORTH_IT"
        because = (
            f"one route replaces {eq:.3f} parent values "
            f"(< {NOT_WORTH_IT_PARENT_WEIGHTS}, the 'eight cheap values' bar)"
        )
    elif not skips_parent_compute and eq < WORTH_IT_PARENT_WEIGHTS:
        label = "NOT_WORTH_IT"
        because = (
            f"one route replaces {eq:.3f} parent values and skips no parent compute "
            f"(below {WORTH_IT_PARENT_WEIGHTS} and not a skip)"
        )
    elif skips_parent_compute and eq >= WORTH_IT_PARENT_WEIGHTS:
        label = "WORTH_IT"
        because = (
            f"one route replaces {eq:.3f} parent values "
            f"(>= {WORTH_IT_PARENT_WEIGHTS}) and skips their compute"
        )
    elif skips_parent_compute:
        label = "MARGINAL"
        because = (
            f"skips parent compute but one route replaces only {eq:.3f} parent values "
            f"(below {WORTH_IT_PARENT_WEIGHTS})"
        )
    else:
        label = "NOT_WORTH_IT"
        because = (
            f"one route replaces {eq:.3f} parent values but skips no parent compute "
            f"— the structure still streams"
        )
    return {"label": label, "because": because, "skips_parent_compute": skips_parent_compute}


def evaluate_route(
    name: str,
    n_routes: float,
    parent_weights: float,
    skips_parent_compute: bool,
    *,
    constructed: bool,
    mechanism: dict,
    notes: str,
) -> dict:
    rates = paired_rates(n_routes, parent_weights)
    inv = rates["PARENT_WEIGHT_EQUIVALENTS_PER_ROUTE"]
    eq = inv["value"] if inv.get("measured") else None
    if n_routes == 0:
        v = {
            "label": "ABSENT",
            "because": "measured route count is zero; nothing to judge as worth or not",
            "skips_parent_compute": skips_parent_compute,
        }
    elif parent_weights <= 0:
        v = {
            "label": "NOT_WORTH_IT",
            "because": (
                f"{n_routes} routes replace 0 parent values and skip no parent compute"
                if not skips_parent_compute
                else f"{n_routes} routes replace 0 parent values"
            ),
            "skips_parent_compute": skips_parent_compute,
        }
    else:
        v = verdict(eq, skips_parent_compute)
    return {
        "name": name,
        "constructed": constructed,
        "mechanism": mechanism,
        "notes": notes,
        "skips_parent_compute": skips_parent_compute,
        **rates,
        "verdict": v,
    }


def self_check_ledger() -> list:
    """The gate is only real if it can refuse. Construct both poles first."""
    fails = []
    fav = evaluate_route(
        "self-check favourable",
        n_routes=4,
        parent_weights=4 * 2_000_000,
        skips_parent_compute=True,
        constructed=True,
        mechanism={"kind": "self-check"},
        notes="four routes, 2e6 parent values each, skip compute",
    )
    unfav = evaluate_route(
        "self-check unfavourable",
        n_routes=8,
        parent_weights=8,
        skips_parent_compute=False,
        constructed=True,
        mechanism={"kind": "self-check"},
        notes="eight routes, eight cheap values, skip nothing",
    )
    if fav["verdict"]["label"] != "WORTH_IT":
        fails.append(f"favourable self-check verdict {fav['verdict']}")
    if unfav["verdict"]["label"] != "NOT_WORTH_IT":
        fails.append(f"unfavourable self-check verdict {unfav['verdict']}")
    if fav["verdict"]["label"] == unfav["verdict"]["label"]:
        fails.append("ledger cannot distinguish the two poles")
    # Inverse pairing.
    fav_rpm = fav["ROUTES_PER_MILLION_PARENT_WEIGHTS"]["value"]
    fav_inv = fav["PARENT_WEIGHT_EQUIVALENTS_PER_ROUTE"]["value"]
    if abs(fav_rpm * (fav_inv / 1e6) - 1.0) > 1e-9:
        fails.append("favourable rates are not inverses")
    zero_parent = evaluate_route(
        "self-check replaces-nothing",
        n_routes=24,
        parent_weights=0,
        skips_parent_compute=False,
        constructed=True,
        mechanism={"kind": "self-check"},
        notes="gates that skip nothing and replace nothing",
    )
    if zero_parent["verdict"]["label"] != "NOT_WORTH_IT":
        fails.append(f"zero-parent self-check verdict {zero_parent['verdict']}")
    if zero_parent["ROUTES_PER_MILLION_PARENT_WEIGHTS"].get("dropped") is not True:
        fails.append("zero-parent rates must be dropped, not zero-filled")
    return fails


# --------------------------------------------------------------------------- source + artifact census


def parse_geometry() -> dict:
    c = rust_usize_consts(GEOMETRY_RS)
    need = [
        "QWEN38_LAYERS",
        "QWEN38_DELTANET_LAYERS",
        "QWEN38_GQA_LAYERS",
        "QWEN38_FULL_ATTENTION_INTERVAL",
        "QWEN38_HIDDEN",
        "QWEN38_INTERMEDIATE",
        "QWEN38_VOCAB",
        "QWEN38_GQA_HEADS",
        "QWEN38_GQA_KV_HEADS",
        "QWEN38_GQA_HEAD_DIM",
        "QWEN38_LINEAR_KEY_HEADS",
        "QWEN38_LINEAR_VALUE_HEADS",
        "QWEN38_LINEAR_VALUES_PER_KEY",
        "QWEN38_LINEAR_KEY_HEAD_DIM",
        "QWEN38_LINEAR_VALUE_HEAD_DIM",
        "QWEN38_LINEAR_CONV_KERNEL",
        "QWEN38_Q_PROJ_ROWS",
        "QWEN38_KV_PROJ_ROWS",
    ]
    missing = [k for k in need if k not in c]
    if missing:
        raise RuntimeError(f"geometry.rs missing consts: {missing}")
    layers = c["QWEN38_LAYERS"]
    interval = c["QWEN38_FULL_ATTENTION_INTERVAL"]
    mixer = []
    for layer in range(layers):
        kind = "gqa" if (layer + 1) % interval == 0 else "delta_net"
        mixer.append(kind)
    dn = mixer.count("delta_net")
    gqa = mixer.count("gqa")
    if dn != c["QWEN38_DELTANET_LAYERS"] or gqa != c["QWEN38_GQA_LAYERS"]:
        raise RuntimeError(f"mixer census drifted: dn={dn} gqa={gqa} vs consts")
    group = c["QWEN38_GQA_HEADS"] // c["QWEN38_GQA_KV_HEADS"]
    rec_elems = (
        c["QWEN38_LINEAR_VALUE_HEADS"]
        * c["QWEN38_LINEAR_KEY_HEAD_DIM"]
        * c["QWEN38_LINEAR_VALUE_HEAD_DIM"]
    )
    conv_channels = (
        c["QWEN38_LINEAR_KEY_HEADS"] * c["QWEN38_LINEAR_KEY_HEAD_DIM"] * 2
        + c["QWEN38_LINEAR_VALUE_HEADS"] * c["QWEN38_LINEAR_VALUE_HEAD_DIM"]
    )
    conv_elems = conv_channels * (c["QWEN38_LINEAR_CONV_KERNEL"] - 1)
    return {
        "consts": c,
        "mixer_per_layer": mixer,
        "delta_net_layers": dn,
        "gqa_layers": gqa,
        "gqa_group_size": group,
        "recurrent_state_elements_per_layer": rec_elems,
        "conv_state_elements_per_layer": conv_elems,
        "sites": {
            "mixer_kind_rule": first_line(GEOMETRY_RS, "GQA iff"),
            "mixer_kind_fn": first_line(GEOMETRY_RS, "pub fn qwen38_mixer_kind"),
            "dense_refuse_moe": first_line(GEOMETRY_RS, "qwen38 is dense"),
            "encode_mixer": first_line(DECODE_RS, "fn encode_mixer("),
            "encode_layers_match": first_line(DECODE_RS, "fn encode_layers("),
            "encode_deltanet": first_line(DECODE_RS, "fn encode_deltanet("),
            "encode_gqa": first_line(DECODE_RS, "fn encode_gqa("),
            "ba_to_decay_dispatch": first_line(
                DECODE_RS,
                '"qwen80_ba_to_decay_beta_f32"',
                start_line=first_line(DECODE_RS, "fn encode_deltanet(").get("line") or 1,
            ),
            "sigmoid_dispatch": first_line(
                DECODE_RS,
                '"qwen38_attention_apply_sigmoid_gate"',
                start_line=first_line(DECODE_RS, "fn encode_gqa(").get("line") or 1,
            ),
            "gated_delta_dispatch": first_line(DECODE_RS, "fn encode_gated_delta("),
            "mha_tcb": first_line(KERNELS_RS, "pub fn mha_decode_f32_tcb("),
            "mha_group_size": first_line(KERNELS_RS, "let group_size = (n_heads / n_kv_heads)"),
            "mha_kv_h": first_line(MHA_METAL, "const uint kv_h   = h / GROUP;"),
            "schedule_no_moe": first_line(
                SCHEDULE_RS, 'name.contains("expert") || name.contains("router")'
            ),
            "ba_kernel": first_line(BA_METAL, "kernel void qwen80_ba_to_decay_beta_f32"),
            "sigmoid_kernel": first_line(SIGMOID_METAL, "kernel void qwen38_attention_apply_sigmoid_gate"),
        },
    }


def kernel_binding_census() -> dict:
    declared = 0
    per_file = []
    for p in sorted(SHADER_DIR.glob("*.metal")):
        n = len(re.findall(r"\bkernel void ", read_text(p)))
        declared += n
        per_file.append({"file": p.name, "kernel_void": n})
    decode = read_text(DECODE_RS)
    names = set(re.findall(r'"([a-z][a-z0-9_]+)"', decode))
    exclude = {"mha_16", "silu_64"}  # isolated-family labels, not kernels
    bindable = sorted(
        n
        for n in names
        if n not in exclude
        and (
            n.startswith("qwen")
            or n.startswith("q80_")
            or n.startswith("sample_")
            or n.startswith("mha_decode")
        )
    )
    # Default-on production names (the flags in decode.rs default these).
    production_default = sorted(
        {
            "qwen_uniform_q4_embedding_lookup",
            "qwen80_residual_rmsnorm_tg",  # HAWKING_RMSNORM_TG default 1024
            "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
            "qwen38_qkvz_rearrange_conv_l2_f32",
            "qwen80_ba_to_decay_beta_f32",
            "qwen38_gated_delta_decode_vi_simd",  # HAWKING_DN_VI_SIMD default on
            "qwen80_deltanet_gated_rmsnorm_tg",  # HAWKING_DN_RMSNORM_TG default 256
            "qwen_next_add_residual",
            "qwen38_gqa_qk_norm_rope_cache_tg",  # HAWKING_ROPE_TG default 256
            "mha_decode_f32",
            "qwen38_attention_apply_sigmoid_gate",
            "qwen80_silu_mul_f32",  # via decode_family::swiglu_f32
            "sample_argmax_f32",  # HAWKING_ARGMAX_TWO_PASS default off
        }
    )
    sched = rust_usize_consts(SCHEDULE_RS, prefix="QWEN38_")
    layers = 64
    mix = sched.get("QWEN38_MIXER_PREFIX_DISPATCHES", 9)
    mlp = sched.get("QWEN38_DENSE_MLP_SUFFIX_DISPATCHES", 6)
    full = mix + mlp
    terminal = 3
    embed = 1
    dispatches_per_token = embed + layers * full + terminal
    return {
        "shader_declared_kernel_voids": declared,
        "shader_files": per_file,
        "hybrid_decode_bindable_kernel_name_literals": len(bindable),
        "hybrid_decode_bindable_names": bindable,
        "production_default_unique_kernels": production_default,
        "production_default_unique_count": len(production_default),
        "production_dispatches_per_token": dispatches_per_token,
        "dispatch_identity": (
            f"1 embed + {layers} layers × {full} + {terminal} terminal = "
            f"{dispatches_per_token}"
        ),
        "anchor_note": (
            "Anchor '38 dispatched against 554 declared' is the bindable-name "
            f"literals in qwen38_hybrid_decode.rs ({len(bindable)}) against "
            f"kernel-void count in crates/hawking-core/shaders ({declared}). "
            "Default-on unique production kernels are fewer; dispatches/token "
            f"are {dispatches_per_token}."
        ),
    }


def read_f32v2(path: Path) -> list:
    """f32v2 envelope: little-endian u64 numel + numel f32 values."""
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{path} too short")
    n = int.from_bytes(raw[:8], "little")
    need = 8 + n * 4
    if len(raw) != need:
        raise ValueError(f"{path} len {len(raw)} != {need} for numel {n}")
    import struct
    return list(struct.unpack("<" + "f" * n, raw[8:]))


def census_alog_payloads(man: dict | None) -> dict:
    """Read every A_log f32v2 off disk. Real tensors, not a model of them."""
    if not man:
        return dropped("no manifest")
    try:
        tensors_dir = ARTIFACT / "tensors"
        values = []
        files = 0
        for t in man.get("tensors") or []:
            if not str(t.get("name") or "").endswith("linear_attn.A_log"):
                continue
            p = tensors_dir / t["artifact"]
            vec = read_f32v2(p)
            values.extend(vec)
            files += 1
        if not values:
            return dropped("no A_log payloads found")
        return measured(
            {
                "files": files,
                "elements": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.fmean(values),
            },
            note="48 f32 per DeltaNet layer, read from the gravity artifact on disk",
        )
    except Exception as e:
        return dropped(f"A_log payload census failed: {type(e).__name__}: {e}")


def census_artifact(geo: dict) -> dict:
    if not ARTIFACT.is_dir():
        return {"present": False, "path": str(ARTIFACT)}
    nfiles = 0
    nbytes = 0
    for f in ARTIFACT.rglob("*"):
        if f.is_file():
            nfiles += 1
            nbytes += f.stat().st_size
    man_path = ARTIFACT / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.is_file() else None
    by_suffix = defaultdict(lambda: {"n": 0, "bytes": 0, "elements": 0, "kind": None, "shape": None})
    moe_like = []
    for t in (man or {}).get("tensors") or []:
        name = t["name"]
        suf = ".".join(name.split(".")[-2:])
        by_suffix[suf]["n"] += 1
        by_suffix[suf]["bytes"] += t["bytes"]
        by_suffix[suf]["elements"] += t["elements"]
        by_suffix[suf]["kind"] = t.get("kind")
        by_suffix[suf]["shape"] = t.get("shape")
        low = name.lower()
        if any(s in low for s in ("expert", "router", "moe", "gate_score")):
            moe_like.append(name)
    # Route-table candidates that actually exist.
    tables = {}
    for key in (
        "in_proj_ba.weight",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
    ):
        if key in by_suffix:
            tables[key] = dict(by_suffix[key])
    c = geo["consts"]
    rec_bytes = geo["delta_net_layers"] * geo["recurrent_state_elements_per_layer"] * 4
    conv_bytes = geo["delta_net_layers"] * geo["conv_state_elements_per_layer"] * 4
    kv_one = c["QWEN38_GQA_KV_HEADS"] * c["QWEN38_GQA_HEAD_DIM"] * 4
    gqa_kv_write = geo["gqa_layers"] * 2 * kv_one
    return {
        "present": True,
        "path": str(ARTIFACT),
        "files": nfiles,
        "bytes": nbytes,
        "files_match_anchor": nfiles == ANCHOR_ARTIFACT_FILES,
        "bytes_match_anchor": nbytes == ANCHOR_ARTIFACT_BYTES,
        "manifest_tensor_count": None if man is None else man.get("tensor_count"),
        "manifest_source_weight_elements": None if man is None else man.get("source_weight_elements"),
        "complete_physical_bpw": None if man is None else man.get("complete_physical_bpw"),
        "q4_tensors": None if man is None else man.get("q4_tensors"),
        "f32_tensors": None if man is None else man.get("f32_tensors"),
        "moe_like_tensor_names": moe_like,
        "route_table_candidates": tables,
        "a_log_payload": (
            census_alog_payloads(man)
            if man
            else dropped("no manifest")
        ),
        "state_bytes": {
            "deltanet_recurrent_resident_bytes": rec_bytes,
            "deltanet_conv_resident_bytes": conv_bytes,
            "gqa_kv_write_bytes_one_pos": gqa_kv_write,
            "note": (
                "recurrent/conv state is the DeltaNet path's memory; GQA KV is "
                "the full-attention path's memory. Neither is an expert cache."
            ),
        },
    }


# --------------------------------------------------------------------------- Metal selector microbench (real GPU, no 27B)


SWIFT_TIMER = r'''
import Foundation
import Metal

struct Bench: Codable {
    var name: String
    var gpu_ns_reps: [Double]
    var median_gpu_ns: Double
    var dispatches: Int
    var threads_per_dispatch: Int
    var layers: Int
    var kernel: String
}

struct Out: Codable {
    var device: String
    var unified_memory_bytes: UInt64
    var gpu_timestamp_authority: String
    var benches: [Bench]
    var error: String?
}

func median(_ xs: [Double]) -> Double {
    let s = xs.sorted()
    return s[s.count / 2]
}

func die(_ msg: String) -> Never {
    let o = Out(device: "", unified_memory_bytes: 0,
                gpu_timestamp_authority: "", benches: [], error: msg)
    let data = try! JSONEncoder().encode(o)
    FileHandle.standardOutput.write(data)
    exit(2)
}

guard let device = MTLCreateSystemDefaultDevice() else { die("no metal device") }
guard let queue = device.makeCommandQueue() else { die("no command queue") }

let src = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : ""
if src.isEmpty { die("missing metal source path") }
let metalSrc: String
do {
    metalSrc = try String(contentsOfFile: src, encoding: .utf8)
} catch {
    die("read metal source: \(error)")
}

let opts = MTLCompileOptions()
let lib: MTLLibrary
do {
    lib = try device.makeLibrary(source: metalSrc, options: opts)
} catch {
    die("compile: \(error)")
}

func pso(_ name: String) -> MTLComputePipelineState {
    guard let fn = lib.makeFunction(name: name) else { die("missing function \(name)") }
    do { return try device.makeComputePipelineState(function: fn) }
    catch { die("pso \(name): \(error)") }
}

func buf(_ nFloats: Int, fill: Float) -> MTLBuffer {
    let b = device.makeBuffer(length: nFloats * 4, options: .storageModeShared)!
    let p = b.contents().bindMemory(to: Float.self, capacity: nFloats)
    for i in 0..<nFloats { p[i] = fill + Float(i % 17) * 0.01 }
    return b
}

func timeCB(warmup: Int, reps: Int, encode: (MTLComputeCommandEncoder) -> Void) -> [Double] {
    func once() -> Double {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        encode(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return (cb.gpuEndTime - cb.gpuStartTime) * 1e9
    }
    for _ in 0..<warmup { _ = once() }
    var out: [Double] = []
    for _ in 0..<reps { out.append(once()) }
    return out
}

let baPSO = pso("qwen80_ba_to_decay_beta_f32")
let sigPSO = pso("qwen38_attention_apply_sigmoid_gate")
let eightPSO = pso("eight_value_route")

let baIn = buf(96, fill: 0.25)
let aLog = buf(48, fill: -1.2)
let dtBias = buf(48, fill: 0.1)
let decay = buf(48, fill: 0)
let beta = buf(48, fill: 0)
var keyHeads: UInt32 = 16
var vpk: UInt32 = 3

let attn = buf(24 * 256, fill: 0.05)
let qproj = buf(24 * 512, fill: 0.2)
let gated = buf(24 * 256, fill: 0)
var elems: UInt32 = 24 * 256
var headDim: UInt32 = 256

let eightIn = buf(8, fill: 1)
let eightOut = buf(8, fill: 0)
var eightN: UInt32 = 8

func encodeBA(_ enc: MTLComputeCommandEncoder, times: Int) {
    enc.setComputePipelineState(baPSO)
    enc.setBuffer(baIn, offset: 0, index: 0)
    enc.setBuffer(aLog, offset: 0, index: 1)
    enc.setBuffer(dtBias, offset: 0, index: 2)
    enc.setBuffer(decay, offset: 0, index: 3)
    enc.setBuffer(beta, offset: 0, index: 4)
    enc.setBytes(&keyHeads, length: 4, index: 5)
    enc.setBytes(&vpk, length: 4, index: 6)
    let tg = MTLSize(width: 16, height: 1, depth: 1)
    let grid = MTLSize(width: 48, height: 1, depth: 1)
    for _ in 0..<times { enc.dispatchThreads(grid, threadsPerThreadgroup: tg) }
}

func encodeSig(_ enc: MTLComputeCommandEncoder, times: Int) {
    enc.setComputePipelineState(sigPSO)
    enc.setBuffer(attn, offset: 0, index: 0)
    enc.setBuffer(qproj, offset: 0, index: 1)
    enc.setBuffer(gated, offset: 0, index: 2)
    enc.setBytes(&elems, length: 4, index: 3)
    enc.setBytes(&headDim, length: 4, index: 4)
    let tg = MTLSize(width: 256, height: 1, depth: 1)
    let grid = MTLSize(width: Int(elems), height: 1, depth: 1)
    for _ in 0..<times { enc.dispatchThreads(grid, threadsPerThreadgroup: tg) }
}

func encodeEight(_ enc: MTLComputeCommandEncoder, times: Int) {
    enc.setComputePipelineState(eightPSO)
    enc.setBuffer(eightIn, offset: 0, index: 0)
    enc.setBuffer(eightOut, offset: 0, index: 1)
    enc.setBytes(&eightN, length: 4, index: 2)
    let tg = MTLSize(width: 8, height: 1, depth: 1)
    let grid = MTLSize(width: 8, height: 1, depth: 1)
    for _ in 0..<times { enc.dispatchThreads(grid, threadsPerThreadgroup: tg) }
}

let warmup = 8
let reps = 7

var benches: [Bench] = []

func record(_ name: String, kernel: String, dispatches: Int, threads: Int, layers: Int,
            encode: @escaping (MTLComputeCommandEncoder) -> Void) {
    let ns = timeCB(warmup: warmup, reps: reps, encode: encode)
    benches.append(Bench(
        name: name, gpu_ns_reps: ns, median_gpu_ns: median(ns),
        dispatches: dispatches, threads_per_dispatch: threads, layers: layers, kernel: kernel
    ))
}

record("ba_to_decay_48_layers", kernel: "qwen80_ba_to_decay_beta_f32",
       dispatches: 48, threads: 48, layers: 48) { enc in encodeBA(enc, times: 48) }
record("ba_to_decay_1_layer", kernel: "qwen80_ba_to_decay_beta_f32",
       dispatches: 1, threads: 48, layers: 1) { enc in encodeBA(enc, times: 1) }
record("sigmoid_gate_16_layers", kernel: "qwen38_attention_apply_sigmoid_gate",
       dispatches: 16, threads: 24 * 256, layers: 16) { enc in encodeSig(enc, times: 16) }
record("sigmoid_gate_1_layer", kernel: "qwen38_attention_apply_sigmoid_gate",
       dispatches: 1, threads: 24 * 256, layers: 1) { enc in encodeSig(enc, times: 1) }
record("eight_value_48_dispatches", kernel: "eight_value_route",
       dispatches: 48, threads: 8, layers: 48) { enc in encodeEight(enc, times: 48) }
record("eight_value_1_dispatch", kernel: "eight_value_route",
       dispatches: 1, threads: 8, layers: 1) { enc in encodeEight(enc, times: 1) }
record("selector_token_shaped", kernel: "ba_to_decay_48 + sigmoid_16",
       dispatches: 48 + 16, threads: 0, layers: 64) { enc in
    encodeBA(enc, times: 48)
    encodeSig(enc, times: 16)
}

let out = Out(
    device: device.name,
    unified_memory_bytes: device.hasUnifiedMemory ? 103079215104 : 0,
    gpu_timestamp_authority: "MTLCommandBuffer.gpuEndTime - gpuStartTime after waitUntilCompleted; never a CPU-wait proxy",
    benches: benches,
    error: nil
)
let enc = JSONEncoder()
enc.outputFormatting = [.sortedKeys]
let data = try! enc.encode(out)
FileHandle.standardOutput.write(data)
print("")
'''


def metal_selector_bench() -> dict:
    """Compile the REAL ba_to_decay and sigmoid kernels; time them on this GPU.

    Does not load the 27B. Launch geometry matches encode_ba_to_decay /
    encode_sigmoid_gate in qwen38_hybrid_decode.rs.
    """
    try:
        ba_src = extract_kernel(read_text(BA_METAL), "qwen80_ba_to_decay_beta_f32")
        sig_src = extract_kernel(read_text(SIGMOID_METAL), "qwen38_attention_apply_sigmoid_gate")
    except Exception as e:
        return dropped(f"could not extract production kernels: {e}")
    eight = """
kernel void eight_value_route(
    device const float* inn [[buffer(0)]],
    device float* out       [[buffer(1)]],
    constant uint& n        [[buffer(2)]],
    uint i                  [[thread_position_in_grid]])
{
    if (i >= n) return;
    out[i] = inn[i];
}
"""
    metal_src = (
        "#include <metal_stdlib>\nusing namespace metal;\n\n"
        + ba_src
        + "\n\n"
        + sig_src
        + "\n\n"
        + eight
    )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "kernels.metal").write_text(metal_src)
        (td / "timer.swift").write_text(SWIFT_TIMER)
        try:
            p = subprocess.run(
                ["swift", str(td / "timer.swift"), str(td / "kernels.metal")],
                capture_output=True,
                text=True,
                timeout=180,
            )
        except Exception as e:
            return dropped(f"swift timer failed to launch: {type(e).__name__}: {e}")
        if p.returncode != 0:
            err = (p.stderr or p.stdout or "")[-1500:]
            needs_gate = "no metal device" in err.lower() or "NO_DEVICE" in err
            return dropped(
                f"swift timer exit {p.returncode}: {err}",
                needs_gate_profile=needs_gate,
                note=(
                    "MTLCreateSystemDefaultDevice() is nil in this sandbox. "
                    "GPU timestamps for selector kernels need the unsandboxed "
                    "gate profile. Not filled with zeros."
                    if needs_gate
                    else None
                ),
            )
        line = (p.stdout or "").strip().splitlines()
        if not line:
            return dropped("swift timer produced no stdout")
        try:
            doc = json.loads(line[-1])
        except Exception as e:
            return dropped(f"swift timer stdout not JSON: {e}; tail={(p.stdout or '')[-400:]}")
        if doc.get("error"):
            return dropped(f"swift timer error: {doc['error']}")
        benches = {b["name"]: b for b in doc.get("benches") or []}
        shaped = benches.get("selector_token_shaped") or {}
        return measured(
            shaped.get("median_gpu_ns"),
            unit="ns",
            gpu_timestamp_authority=doc.get("gpu_timestamp_authority"),
            device=doc.get("device"),
            benches=benches,
            kernels_compiled_from=[
                "crates/hawking-core/shaders/qwen80_device_activations.metal",
                "crates/hawking-core/shaders/qwen38_device_activations.metal",
            ],
            launch_geometry={
                "ba_to_decay": "grid (48,1,1) tg (16,1,1) × 48 DeltaNet layers",
                "sigmoid_gate": "grid (6144,1,1) tg (256,1,1) × 16 GQA layers",
            },
            note=(
                "Isolated selector command buffer at production launch geometry. "
                "Not inside the production 964-dispatch token CB, so this is the "
                "selector's own GPU time, not its contribution to token wall "
                "(a small kernel's isolated ns overstates its in-token cost)."
            ),
        )


# --------------------------------------------------------------------------- live llama-server execution (the resident 27B)


def llama_get(path: str, timeout: float = 5.0):
    url = f"http://127.0.0.1:{LLAMA_PORT}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def live_llama_decode(n_predict: int = 32) -> dict:
    """One completion against the already-resident llama-server. No second 27B."""
    try:
        health = llama_get("/health")
    except Exception as e:
        return dropped(f"llama-server :{LLAMA_PORT} health failed: {type(e).__name__}: {e}")
    try:
        models = llama_get("/v1/models")
        slots = llama_get("/slots")
    except Exception as e:
        models, slots = {"error": str(e)}, []
    busy = [s.get("id") for s in (slots or []) if s.get("is_processing")]
    if busy:
        # Do not pile a second decode on a busy slot; wait briefly once.
        time.sleep(2.0)
        try:
            slots = llama_get("/slots")
            busy = [s.get("id") for s in (slots or []) if s.get("is_processing")]
        except Exception:
            pass
    prompt = (
        "RouteLedger probe. Reply with the single word READY and nothing else. "
        "This is a timing probe, not a question."
    )
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "ignore_eos": True,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{LLAMA_PORT}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        return dropped(f"completion failed: {type(e).__name__}: {e}")
    wall_s = time.perf_counter() - t0
    timings = body.get("timings") or {}
    pred_n = timings.get("predicted_n")
    pred_ms = timings.get("predicted_ms")
    tps = timings.get("predicted_per_second")
    content = body.get("content") or ""
    model_id = None
    try:
        data = (models or {}).get("data") or (models or {}).get("models") or []
        if data:
            model_id = data[0].get("id") or data[0].get("name")
    except Exception:
        pass
    return measured(
        {
            "predicted_n": pred_n,
            "predicted_ms": pred_ms,
            "predicted_per_second": tps,
            "prompt_n": timings.get("prompt_n"),
            "prompt_ms": timings.get("prompt_ms"),
            "prompt_per_second": timings.get("prompt_per_second"),
            "client_wall_s": round(wall_s, 3),
            "content_excerpt": content[:80],
        },
        port=LLAMA_PORT,
        health=health,
        model=model_id,
        n_predict=n_predict,
        slots_busy_at_start=busy,
        n_slots=len(slots) if isinstance(slots, list) else None,
        decode_flags="temperature=0 ignore_eos=true cache_prompt=false",
        note="Resident llama-server. This is a real Qwen3.8 forward pass on this box.",
    )


def native_receipt_tps() -> dict:
    if not NATIVE_RECEIPT.is_file():
        return dropped("QWEN38_GRAVITY_NATIVE.json not on this worktree")
    d = json.loads(NATIVE_RECEIPT.read_text())
    dec = d.get("decode") or {}
    return measured(
        dec.get("median_complete_wall_tps"),
        median_complete_wall_ns=dec.get("median_complete_wall_ns"),
        n_process_runs=dec.get("n_process_runs"),
        measurement_spread_pct=dec.get("measurement_spread_pct"),
        source="receipts/headless/QWEN38_GRAVITY_NATIVE.json",
        note=(
            "Prior native complete-wall on the gravity artifact. Not re-run: "
            + LIVE_27B_WARNING
        ),
        bar=d.get("bar"),
    )


# --------------------------------------------------------------------------- cases


def construct_cases(geo: dict, artifact: dict) -> dict:
    c = geo["consts"]
    hidden = c["QWEN38_HIDDEN"]
    gqa_layers = geo["gqa_layers"]
    dn_layers = geo["delta_net_layers"]
    q_heads = c["QWEN38_GQA_HEADS"]
    kv_heads = c["QWEN38_GQA_KV_HEADS"]
    head_dim = c["QWEN38_GQA_HEAD_DIM"]
    group = geo["gqa_group_size"]
    missing_kv_heads = q_heads - kv_heads  # 20

    # Favourable: GQA KV sharing. Each of 4 KV heads serves `group` query heads.
    # Parent weights replaced = the K+V projections of the 20 heads MHA would have.
    saved_per_layer = missing_kv_heads * head_dim * hidden * 2  # K and V
    saved_total = saved_per_layer * gqa_layers
    n_kv_routes = kv_heads * gqa_layers  # 64 static KV-head slots
    fav = evaluate_route(
        "GQA KV-sharing (real)",
        n_routes=n_kv_routes,
        parent_weights=saved_total,
        skips_parent_compute=True,
        constructed=False,
        mechanism={
            "kind": "static_head_grouping",
            "per_token_decision": False,
            "group_size": group,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "gqa_layers": gqa_layers,
            "sites": {
                "geometry": geo["sites"]["mixer_kind_rule"],
                "group_size_rust": geo["sites"]["mha_group_size"],
                "kv_head_index_metal": geo["sites"]["mha_kv_h"],
                "encode_gqa": geo["sites"]["encode_gqa"],
            },
        },
        notes=(
            f"{q_heads} query heads share {kv_heads} KV heads (group={group}) on "
            f"{gqa_layers} layers. Each KV-head route skips {missing_kv_heads} "
            f"explicit K/V head projections of {head_dim}×{hidden}. Static: the "
            f"same grouping every token. Kernel: mha_decode_f32 `kv_h = h / GROUP`."
        ),
    )

    # Unfavourable (real): GQA sigmoid attention gate. Runs AFTER full attention.
    # Extra q_proj rows (24 * 256) stream every GQA layer and skip nothing.
    gate_rows_per_layer = q_heads * head_dim  # the second half of q_proj
    gate_parent = gate_rows_per_layer * hidden * gqa_layers
    n_gate_routes = q_heads * gqa_layers  # 384 per-head gates
    unfav_gate = evaluate_route(
        "GQA sigmoid attention gate (real)",
        n_routes=n_gate_routes,
        parent_weights=0,  # it REPLACES nothing; it ADDS a gate after attention
        skips_parent_compute=False,
        constructed=False,
        mechanism={
            "kind": "per_token_continuous_gate",
            "per_token_decision": True,
            "discrete": False,
            "gates_per_gqa_layer": q_heads,
            "gqa_layers": gqa_layers,
            "added_q_proj_parent_weights": gate_parent,
            "sites": {
                "dispatch": geo["sites"]["sigmoid_dispatch"],
                "kernel": geo["sites"]["sigmoid_kernel"],
            },
        },
        notes=(
            "qwen38_attention_apply_sigmoid_gate multiplies already-computed "
            f"attention by a per-head sigmoid from the extra {gate_rows_per_layer} "
            f"q_proj rows. {n_gate_routes} gates/token. Parent weights REPLACED = 0 "
            f"(the extra q_proj rows are a cost of {gate_parent} values, not a skip). "
            "A zero denominator drops the rates rather than reporting 0."
        ),
    )

    # Unfavourable (real tensor, wrong interpretation): A_log as 48 "routes".
    a_log = ((artifact.get("route_table_candidates") or {}).get("linear_attn.A_log") or {})
    a_log_elems = a_log.get("elements") or (dn_layers * c["QWEN38_LINEAR_VALUE_HEADS"])
    unfav_alog = evaluate_route(
        "DeltaNet A_log as if it were a route table (real tensor, dishonest reading)",
        n_routes=a_log_elems,
        parent_weights=a_log_elems,  # 1 f32 per 'route'
        skips_parent_compute=False,
        constructed=True,
        mechanism={
            "kind": "parameter_vector_misread_as_routing",
            "per_token_decision": False,
            "tensor": "linear_attn.A_log",
            "bytes": a_log.get("bytes"),
            "elements": a_log_elems,
            "sites": {"ba_kernel": geo["sites"]["ba_kernel"]},
        },
        notes=(
            f"A_log is {a_log_elems} f32 decay-scale parameters "
            f"({a_log.get('bytes')} bytes). Treating each scalar as a 'route' "
            "replaces one cheap value and skips no head of gated-delta. This is "
            "the constructed unfavourable pole using a real tensor."
        ),
    )

    # Textbook unfavourable the steer named: eight cheap values.
    unfav_eight = evaluate_route(
        "eight cheap values (constructed)",
        n_routes=8,
        parent_weights=8,
        skips_parent_compute=False,
        constructed=True,
        mechanism={"kind": "constructed_eight_value_route"},
        notes="The steer: a route replacing eight cheap values is stupid.",
    )

    # Discrete MoE routes — the thing people expect, which this artifact does not have.
    moe = evaluate_route(
        "MoE expert top-k (absent)",
        n_routes=0,
        parent_weights=ANCHOR_PARAM_COUNT,
        skips_parent_compute=False,
        constructed=False,
        mechanism={
            "kind": "absent_moe",
            "per_token_decision": False,
            "sites": {
                "dense_refuse": geo["sites"]["dense_refuse_moe"],
                "schedule_no_moe": geo["sites"]["schedule_no_moe"],
            },
        },
        notes=(
            "qwen38_accept_config refuses num_experts / moe_intermediate_size. "
            "The 64-layer schedule asserts the MLP suffix contains no expert/"
            "router/moe kernel. Zero discrete expert decisions is the finding."
        ),
    )

    # Continuous DeltaNet beta/decay selectors: 48 heads × 48 layers, skip nothing.
    n_beta = c["QWEN38_LINEAR_VALUE_HEADS"] * dn_layers
    ba_elems = ((artifact.get("route_table_candidates") or {}).get("in_proj_ba.weight") or {}).get(
        "elements"
    ) or (dn_layers * 96 * hidden)
    beta_case = evaluate_route(
        "DeltaNet ba_to_decay_beta per value-head (real, continuous, not a skip)",
        n_routes=n_beta,
        parent_weights=ba_elems,
        skips_parent_compute=False,
        constructed=False,
        mechanism={
            "kind": "per_token_continuous_gate",
            "per_token_decision": True,
            "discrete": False,
            "value_heads": c["QWEN38_LINEAR_VALUE_HEADS"],
            "delta_net_layers": dn_layers,
            "sites": {
                "dispatch": geo["sites"]["ba_to_decay_dispatch"],
                "kernel": geo["sites"]["ba_kernel"],
                "gated_delta": geo["sites"]["gated_delta_dispatch"],
            },
        },
        notes=(
            f"{c['QWEN38_LINEAR_VALUE_HEADS']} beta/decay scalars per DeltaNet layer "
            f"× {dn_layers} layers = {n_beta} selector applications per token. "
            "Every value head still runs gated-delta. in_proj_ba is the selector "
            "projection (96×5120 q4), not a skip table."
        ),
    )

    return {
        "favourable": fav,
        "unfavourable_real_gate": unfav_gate,
        "unfavourable_real_alog_misread": unfav_alog,
        "unfavourable_eight_cheap_values": unfav_eight,
        "moe_absent": moe,
        "deltanet_beta_continuous": beta_case,
        "distinguishes": fav["verdict"]["label"] != unfav_alog["verdict"]["label"]
        and fav["verdict"]["label"] == "WORTH_IT"
        and unfav_alog["verdict"]["label"] == "NOT_WORTH_IT",
    }


def per_token_decision_census(geo: dict) -> dict:
    c = geo["consts"]
    dn = geo["delta_net_layers"]
    gqa = geo["gqa_layers"]
    discrete_moe = 0
    discrete_mixer = 0  # static, not per-token
    discrete_gqa_group = 0  # static
    continuous_beta = c["QWEN38_LINEAR_VALUE_HEADS"] * dn
    continuous_sigmoid = c["QWEN38_GQA_HEADS"] * gqa
    mixer_H = shannon_bits([dn, gqa])
    return {
        "discrete_expert_routes_per_token": measured(
            discrete_moe,
            kind="measured_absent",
            note="dense model; zero is a count, not a missing measurement",
            site=geo["sites"]["dense_refuse_moe"],
        ),
        "discrete_mixer_kind_choices_per_token": measured(
            discrete_mixer,
            kind="static_architecture",
            note=(
                "qwen38_mixer_kind(layer) is a function of layer index only "
                f"((layer+1)%{c['QWEN38_FULL_ATTENTION_INTERVAL']}==0 → GQA). "
                "Not a per-token decision."
            ),
            site=geo["sites"]["mixer_kind_fn"],
        ),
        "discrete_gqa_grouping_choices_per_token": measured(
            discrete_gqa_group,
            kind="static_architecture",
            note=(
                f"kv_h = h / {geo['gqa_group_size']} in mha_decode_f32. "
                "Index arithmetic, identical every token."
            ),
            site=geo["sites"]["mha_kv_h"],
        ),
        "continuous_deltanet_beta_decay_per_token": measured(
            continuous_beta,
            kind="per_token_continuous_gate",
            note="48 value heads × 48 DeltaNet layers; all heads still run",
            site=geo["sites"]["ba_to_decay_dispatch"],
        ),
        "continuous_gqa_sigmoid_gates_per_token": measured(
            continuous_sigmoid,
            kind="per_token_continuous_gate",
            note="24 heads × 16 GQA layers; attention already computed",
            site=geo["sites"]["sigmoid_dispatch"],
        ),
        "genuine_discrete_route_decisions_per_token": measured(
            0,
            kind="measured_absent",
            note=(
                "The decode path makes no learned discrete per-token route "
                "decision. Mixer kind and GQA grouping are static. DeltaNet "
                "beta and the GQA sigmoid gate are continuous and skip no "
                "parent weights. Recording zero is the finding."
            ),
        ),
        "mixer_layer_type_entropy_bits": measured(
            mixer_H,
            unit="bits / layer",
            note="entropy of the 48/16 DeltaNet/GQA mix across layers, not per token",
            distribution={"delta_net": dn, "gqa": gqa},
        ),
        "mixer_entropy_bits_given_layer": measured(
            0.0,
            unit="bits",
            note="given layer index the mixer is determined; entropy is zero",
        ),
        "route_reuse_across_tokens": {
            "mixer_kind": measured(1.0, unit="fraction", note="static, identical every token"),
            "gqa_grouping": measured(1.0, unit="fraction", note="static, identical every token"),
            "deltanet_beta": measured(
                0.0,
                unit="fraction",
                note="beta/decay recomputed every token from in_proj_ba; decision not cached",
            ),
            "gqa_sigmoid_gate": measured(
                0.0,
                unit="fraction",
                note="gate taken from this token's q_proj; decision not cached",
            ),
        },
        "route_reuse_across_layers": {
            "mixer_pattern_period": measured(
                c["QWEN38_FULL_ATTENTION_INTERVAL"],
                note="GQA on layers 3,7,…,63; the 4-layer pattern repeats 16 times",
            ),
            "gqa_grouping_identical_on_all_gqa_layers": measured(True),
            "deltanet_beta_shared_across_layers": measured(
                False, note="each DeltaNet layer has its own in_proj_ba, A_log, dt_bias"
            ),
        },
    }


def route_cache_metric() -> dict:
    # expert_cache.rs exists for other models. Qwen3.8 decode does not use it.
    mentions = []
    if DECODE_RS.is_file():
        t = read_text(DECODE_RS)
        if re.search(r"expert_cache|route_cache|ExpertCache", t):
            mentions.append("qwen38_hybrid_decode.rs names an expert/route cache")
    return dropped(
        "no route/expert cache on the Qwen3.8 decode path; a 0% hit rate would mean "
        "'not measured' and is the defect this campaign keeps finding",
        searched="crates/hawking-core/src/model/qwen38_hybrid_decode.rs for expert_cache/route_cache",
        mentions=mentions,
    )


def route_table_bytes(artifact: dict) -> dict:
    tables = artifact.get("route_table_candidates") or {}
    ba = tables.get("in_proj_ba.weight") or {}
    alog = tables.get("linear_attn.A_log") or {}
    dt = tables.get("linear_attn.dt_bias") or {}
    moe_like = artifact.get("moe_like_tensor_names") or []
    return {
        "moe_router_bytes": measured(
            0,
            kind="measured_absent",
            note="no router/expert tensors in the 755-tensor catalog",
            moe_like_names=moe_like,
        ),
        "deltanet_selector_projection_bytes": measured(
            ba.get("bytes"),
            n=ba.get("n"),
            elements=ba.get("elements"),
            kind=ba.get("kind"),
            shape=ba.get("shape"),
            tensor="linear_attn.in_proj_ba.weight",
            note="96×5120 q4 per DeltaNet layer; computes beta/decay, does not skip heads",
        ),
        "deltanet_A_log_bytes": measured(alog.get("bytes"), elements=alog.get("elements"), kind=alog.get("kind")),
        "deltanet_dt_bias_bytes": measured(dt.get("bytes"), elements=dt.get("elements"), kind=dt.get("kind")),
        "gqa_grouping_table_bytes": measured(
            0,
            kind="measured_absent",
            note="grouping is `kv_h = h / GROUP`; there is no table to store",
        ),
    }


def entropy_dropped_activations() -> dict:
    return dropped(
        "per-token DeltaNet beta / GQA sigmoid-gate entropy needs device buffers "
        "from a native decode. Loading the gravity 27B while llama-server is "
        "resident is forbidden (occupancy collapse 33.47 → 3.986 tok/s). "
        "Not filled with zeros.",
        would_require="ascension_qwen38_hybrid_greedy against qwen38-gravity-uniform-q4-v1",
    )


# --------------------------------------------------------------------------- assemble + print


def watched_fail(cases, self_fails, llama, metal, artifact, geo) -> list:
    rows = [
        {
            "what": "treat mixer_kind as a per-token route",
            "what_happened": (
                "qwen38_mixer_kind is a function of layer index only "
                f"({geo['sites']['mixer_kind_fn'].get('path')}:"
                f"{geo['sites']['mixer_kind_fn'].get('line')}). "
                "Counting 64 mixer choices/token would invent routes. "
                "Recorded as static, 0 per-token decisions."
            ),
        },
        {
            "what": "count MoE expert top-k on Qwen3.8",
            "what_happened": (
                "geometry refuses num_experts; the 64-layer schedule asserts the "
                "MLP suffix has no expert/router/moe kernel; the artifact catalog "
                f"has {len(artifact.get('moe_like_tensor_names') or [])} moe-like names. "
                "Zero discrete expert routes is the finding."
            ),
        },
        {
            "what": "count continuous DeltaNet beta as discrete routes that skip work",
            "what_happened": (
                "ba_to_decay_beta writes 48 decay/beta scalars per DeltaNet layer "
                "and encode_gated_delta still runs all 48 value heads. The ledger "
                "marks skips_parent_compute=False and verdict NOT_WORTH_IT / MARGINAL."
            ),
        },
        {
            "what": "report route_cache_hit_rate = 0 because there is no cache",
            "what_happened": (
                "Dropped. A zero that means 'not measured' is the defect this "
                "campaign keeps finding."
            ),
        },
        {
            "what": "measure beta entropy from a native 27B decode",
            "what_happened": LIVE_27B_WARNING,
        },
        {
            "what": "apply tree-state.patch onto tools/haider in this sparse checkout",
            "what_happened": (
                "git apply failed with 'hcli/*.py: No such file or "
                "directory'. Applied with --exclude tools/haider. Baseline suite "
                "was run from ~/Downloads/hawking-copy."
            ),
        },
        {
            "what": "reproduce 464 passed, 1 skipped",
            "what_happened": (
                "HCLI_SWAP_CEILING_GIB=64 pytest hcli/tests from "
                "hawking-copy: 463 passed, 2 skipped. The extra skip is "
                "test_mlx_backend.py: mlx_lm.server --help did not answer in this "
                "environment. The expected skip is the live grok-run audit."
            ),
        },
        {
            "what": "a ledger that only ever reports routing is fine",
            "what_happened": (
                f"favourable verdict={cases['favourable']['verdict']['label']}; "
                f"A_log-as-route verdict={cases['unfavourable_real_alog_misread']['verdict']['label']}; "
                f"eight-cheap-values verdict={cases['unfavourable_eight_cheap_values']['verdict']['label']}; "
                f"sigmoid-gate rates dropped because parent_weights replaced = 0. "
                f"distinguishes={cases['distinguishes']}"
            ),
        },
    ]
    if self_fails:
        rows.append({"what": "ledger self-check", "what_happened": self_fails})
    if llama.get("dropped"):
        rows.append({"what": "live llama decode", "what_happened": llama.get("reason")})
    if metal.get("dropped"):
        rows.append({"what": "Metal selector microbench", "what_happened": metal.get("reason")})
    return rows


def rates_per_second(decisions_per_token, tps) -> dict:
    if tps is None:
        return dropped("no measured tokens/s to multiply by")
    if decisions_per_token is None:
        return dropped("no decisions_per_token")
    return measured(
        decisions_per_token * tps,
        unit="1/s",
        decisions_per_token=decisions_per_token,
        tokens_per_second=tps,
    )


def print_report(doc: dict) -> None:
    print("NOETIC ROUTE LEDGER")
    print(f"schema {doc['schema']}")
    print(f"recorded_at {doc['recorded_at']}")
    print()
    print("## WHAT THE DECODE PATH ACTUALLY DECIDES")
    d = doc["per_token_decisions"]
    g = d["genuine_discrete_route_decisions_per_token"]
    print(
        f"genuine discrete route decisions per token: {g['value']} "
        f"({g.get('kind')})"
    )
    print(f"  {g.get('note')}")
    for key in (
        "discrete_expert_routes_per_token",
        "discrete_mixer_kind_choices_per_token",
        "discrete_gqa_grouping_choices_per_token",
        "continuous_deltanet_beta_decay_per_token",
        "continuous_gqa_sigmoid_gates_per_token",
    ):
        row = d[key]
        print(f"- {key}: {row['value']}  [{row.get('kind')}]")
        site = (row.get("site") or {})
        if site.get("path"):
            print(f"    at {site['path']}:{site['line']}")
    print()
    print("## RATES (paired with inverse)")
    tps_live = doc["executions"]["llama_server"].get("value") or {}
    live_tps = tps_live.get("predicted_per_second") if isinstance(tps_live, dict) else None
    print(f"live llama predicted_per_second: {live_tps}")
    native = doc["executions"]["native_gravity_receipt"]
    print(f"native complete-wall tps (receipt, not re-run): {native.get('value')}")
    rps = doc["route_decisions_per_second"]
    def fmt_rate(row):
        if not row:
            return "n/a"
        if row.get("dropped"):
            return f"DROPPED ({row.get('reason')})"
        return f"{row.get('value')} {row.get('unit', '')}".strip()

    print(f"discrete routes/s (live tps × 0): {fmt_rate(rps['discrete_live'])}")
    print(
        f"continuous selector applications/s (live tps × 2688): "
        f"{fmt_rate(rps['continuous_live'])}"
    )
    print()
    print("## SELECTOR token_ns")
    sel = doc["selector_token_ns"]
    if sel.get("dropped"):
        print(f"DROPPED: {sel.get('reason')}")
    else:
        print(f"median GPU ns of production-shaped selector CB: {sel.get('value')}")
        print(f"  authority: {sel.get('gpu_timestamp_authority')}")
        benches = sel.get("benches") or {}
        for name in (
            "ba_to_decay_48_layers",
            "sigmoid_gate_16_layers",
            "eight_value_48_dispatches",
            "selector_token_shaped",
        ):
            b = benches.get(name)
            if b:
                print(
                    f"  {name}: median {b['median_gpu_ns']:.0f} ns "
                    f"({b['dispatches']} dispatches, reps={b['gpu_ns_reps']})"
                )
    print()
    print("## CASES")
    cases = doc["cases"]
    for key in (
        "favourable",
        "unfavourable_real_gate",
        "unfavourable_real_alog_misread",
        "unfavourable_eight_cheap_values",
        "moe_absent",
        "deltanet_beta_continuous",
    ):
        c = cases[key]
        rpm = c["ROUTES_PER_MILLION_PARENT_WEIGHTS"]
        inv = c["PARENT_WEIGHT_EQUIVALENTS_PER_ROUTE"]
        rpm_s = "DROPPED" if rpm.get("dropped") else f"{rpm['value']:.6g}"
        inv_s = "DROPPED" if inv.get("dropped") else f"{inv['value']:.6g}"
        print(f"- {c['name']}")
        print(f"    routes={c['n_routes']} parent_weights={c['parent_weights']}")
        print(f"    ROUTES_PER_MILLION_PARENT_WEIGHTS={rpm_s}")
        print(f"    PARENT_WEIGHT_EQUIVALENTS_PER_ROUTE={inv_s}")
        print(f"    skips_parent_compute={c['skips_parent_compute']}")
        print(f"    verdict={c['verdict']['label']}: {c['verdict']['because']}")
    print(f"ledger distinguishes favourable vs unfavourable: {cases['distinguishes']}")
    print()
    print("## ROUTE TABLE / STATE")
    tb = doc["route_table_bytes"]
    for k, v in tb.items():
        extra = (" (dropped: " + v["reason"]) if v.get("dropped") else ""
        print(f"- {k}: {v.get('value')}{extra}")
    st = doc["route_state_bytes"]
    print(f"state: {json.dumps(st, sort_keys=True)}")
    alog = (doc.get("artifact") or {}).get("a_log_payload") or {}
    if alog.get("measured"):
        print(f"A_log on disk: {alog.get('value')}")
    elif alog.get("dropped"):
        print(f"A_log on disk DROPPED: {alog.get('reason')}")
    print()
    print("## DROPPED (not filled with zeros)")
    for k, v in doc["dropped"].items():
        print(f"- {k}: {v.get('reason')}")
    print()
    print("## KERNEL BINDING")
    kb = doc["kernel_binding"]
    print(
        f"declared kernel void in shaders: {kb['shader_declared_kernel_voids']}; "
        f"bindable name literals in hybrid_decode.rs: "
        f"{kb['hybrid_decode_bindable_kernel_name_literals']}; "
        f"production default unique: {kb['production_default_unique_count']}; "
        f"dispatches/token: {kb['production_dispatches_per_token']}"
    )
    print()
    print("## WHAT I WATCHED FAIL")
    for row in doc["what_i_watched_fail"]:
        print(f"- {row['what']}")
        print(f"    {row['what_happened']}")
    print()
    print(f"receipt: {RECEIPT}")


def main() -> int:
    self_fails = self_check_ledger()
    if self_fails:
        print("FAIL ledger self-check:", self_fails)
        return 2

    geo = parse_geometry()
    kernels = kernel_binding_census()
    artifact = census_artifact(geo)
    cases = construct_cases(geo, artifact)
    decisions = per_token_decision_census(geo)
    metal = metal_selector_bench()
    llama = live_llama_decode()
    native = native_receipt_tps()

    live_tps = None
    if llama.get("measured") and isinstance(llama.get("value"), dict):
        live_tps = llama["value"].get("predicted_per_second")
    native_tps = native.get("value") if native.get("measured") else None

    discrete_pt = decisions["genuine_discrete_route_decisions_per_token"]["value"]
    cont_pt = (
        decisions["continuous_deltanet_beta_decay_per_token"]["value"]
        + decisions["continuous_gqa_sigmoid_gates_per_token"]["value"]
    )

    dropped_metrics = {
        "route_cache_hit_rate": route_cache_metric(),
        "deltanet_beta_entropy_from_activations": entropy_dropped_activations(),
        "gqa_sigmoid_entropy_from_activations": entropy_dropped_activations(),
        "selector_token_ns_inside_production_cb": dropped(
            "selector GPU ns below is an isolated CB at production launch "
            "geometry, not a slice of the 964-dispatch production token CB. "
            "Partitioning production GPU time requires the native 27B session."
        ),
    }

    doc = {
        "schema": SCHEMA,
        "recorded_at": utc_now(),
        "git": git_sha(),
        "repo": str(REPO),
        "anchors_cited_not_rederived": {
            "native_tps": ANCHOR_TPS,
            "native_ms_per_token": ANCHOR_MS_PER_TOKEN,
            "roof_GB_s": ANCHOR_ROOF_GB_S,
            "parameter_count": ANCHOR_PARAM_COUNT,
            "artifact_files": ANCHOR_ARTIFACT_FILES,
            "artifact_bytes": ANCHOR_ARTIFACT_BYTES,
            "tensors": ANCHOR_TENSORS,
            "machine": "Apple M3 Ultra, 60 GPU cores, 103079215104 B unified, Metal 4",
        },
        "live_27b_policy": LIVE_27B_WARNING,
        "artifact": artifact,
        "geometry": {
            "layers": geo["consts"]["QWEN38_LAYERS"],
            "delta_net_layers": geo["delta_net_layers"],
            "gqa_layers": geo["gqa_layers"],
            "gqa_group_size": geo["gqa_group_size"],
            "hidden": geo["consts"]["QWEN38_HIDDEN"],
            "mixer_per_layer": geo["mixer_per_layer"],
            "sites": geo["sites"],
        },
        "kernel_binding": kernels,
        "per_token_decisions": decisions,
        "route_table_bytes": route_table_bytes(artifact),
        "route_state_bytes": (artifact.get("state_bytes") if artifact.get("present") else dropped("artifact missing")),
        "selector_token_ns": metal,
        "executions": {
            "llama_server": llama,
            "native_gravity_receipt": native,
            "metal_selector_microbench": {
                "spawned_second_27b": False,
                "loaded_gravity_weights": False,
                "compiled_production_kernels": not metal.get("dropped"),
            },
        },
        "route_decisions_per_token": {
            "discrete": decisions["genuine_discrete_route_decisions_per_token"],
            "continuous_selector_applications": measured(
                cont_pt,
                breakdown={
                    "deltanet_beta_decay": decisions["continuous_deltanet_beta_decay_per_token"]["value"],
                    "gqa_sigmoid": decisions["continuous_gqa_sigmoid_gates_per_token"]["value"],
                },
            ),
        },
        "route_decisions_per_second": {
            "discrete_live": rates_per_second(discrete_pt, live_tps),
            "discrete_native_receipt": rates_per_second(discrete_pt, native_tps),
            "continuous_live": rates_per_second(cont_pt, live_tps),
            "continuous_native_receipt": rates_per_second(cont_pt, native_tps),
        },
        "cases": cases,
        "dropped": dropped_metrics,
        "what_i_watched_fail": [],
        "self_check_fails": self_fails,
    }
    doc["what_i_watched_fail"] = watched_fail(cases, self_fails, llama, metal, artifact, geo)

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print_report(doc)

    # Hard fail if the ledger cannot distinguish the poles.
    if not cases["distinguishes"]:
        print("FAIL: ledger did not distinguish WORTH_IT from NOT_WORTH_IT")
        return 2
    if discrete_pt != 0:
        print("FAIL: invented discrete routes")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
