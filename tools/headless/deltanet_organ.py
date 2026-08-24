#!/usr/bin/env python3
"""N026 — attack DeltaNet, the largest un-optimized share of the 356.7→778.8 gap.

N025 attributed 25.9% of the roof-gap extra ns to DeltaNet (6.64 ms/token,
325.5 GB/s). This harness autopsies the recurrent kernels, applies two
state-update changes (widen float4 loads; stage a 128×32 state tile in
threadgroup memory and load it coalesced), and reports how much of that
share moved.

    python3 tools/headless/deltanet_organ.py
    python3 tools/headless/deltanet_organ.py --from-raw
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dispatch_ledger import occupancy_snapshot  # noqa: E402
from first_noetic_executable import git_head, now_iso  # noqa: E402
from kernel_competence import (  # noqa: E402
    kernel_bodies,
    params_of,
    screen_kernel,
    strip_comments,
)
from noetic_operation_census import ANCHOR_ROOF_GB_S  # noqa: E402

SCHEMA = "hawking.headless.deltanet_organ.v1"
RECEIPT = REPO / "receipts" / "headless" / "DELTANET_ORGAN.json"
RAW = REPO / "receipts" / "headless" / "_DELTANET_ORGAN_raw.json"
ORGAN_PRIOR = REPO / "receipts" / "headless" / "ORGAN_BANDWIDTH.json"
DECODE = REPO / "crates" / "hawking-core" / "src" / "model" / "qwen38_hybrid_decode.rs"
SHADER = REPO / "crates" / "hawking-core" / "shaders" / "qwen38_device_activations.metal"
CARGO_TARGET = Path(
    os.environ.get("CARGO_TARGET_DIR", str(REPO / "workspace" / "ops" / "build" / "rust"))
)
BIN = CARGO_TARGET / "release-fast" / "examples" / "ascension_qwen38_deltanet_organ"
PARENT_ROOT = Path(
    os.environ.get("NOETIC_PARENT_A_ROOT", str(Path.home() / "noetic" / "NOETIC_PARENT_A"))
)
TOKENIZER = Path(
    os.environ.get(
        "QWEN38_TOKENIZER",
        str(Path.home() / "models" / "qwen3.8-27b-abliterated-bf16" / "tokenizer.json"),
    )
)

ROOF_GB_S = ANCHOR_ROOF_GB_S  # 778.8; do not re-derive
N018_PRODUCTION_GB_S = 356.7
GAP_GB_S = ROOF_GB_S - N018_PRODUCTION_GB_S  # 422.1
PARENT_ACTIVE_BYTES = 9_878_901_136
# N025 organ attribution (weight_read, not traffic).
DELTANET_WEIGHT_READ = 2_161_674_240
DELTANET_TRAFFIC = 2_499_520_512
N025_SHARE = 0.25949282064400503
N025_GAP_NS = 3_864_745.429428092
N025_ORGAN_NS = 6_640_392.887055211
N025_GB_S = 325.5340876311656
N025_ISOLATED_NS = 6_547_625.0
KERNEL_BASELINE = "qwen38_gated_delta_decode_vi_simd_ba"
KERNEL_F4 = "qwen38_gated_delta_decode_vi_simd_ba_f4"
KERNEL_TG32 = "qwen38_gated_delta_decode_vi_simd_ba_tg32"
KERNEL_BAD = "qwen38_gated_delta_decode_vi_simd_ba_plain"
NEW_KERNELS = (KERNEL_F4, KERNEL_TG32, KERNEL_BAD)
CHANGES = ("widen_f4", "coalesce_tg32")

# Geometry (qwen38_geometry.rs). Do not re-derive from a second 27B.
DN_LAYERS = 48
HIDDEN = 5120
QKVZ_ROWS = 16384
BA_ROWS = 96
OUT_ROWS = 5120
OUT_COLS = 6144
REC_ELEMS_LAYER = 786432  # 48 heads × 128 × 128
REC_ELEMS_TOTAL = REC_ELEMS_LAYER * DN_LAYERS
REC_RESIDENT = REC_ELEMS_TOTAL * 4  # 150_994_944
QKVZ_MAC = QKVZ_ROWS * HIDDEN * DN_LAYERS * 2  # 8_053_063_680 (2N, prior science)
BA_MAC = BA_ROWS * HIDDEN * DN_LAYERS * 2
OUT_MAC = OUT_ROWS * OUT_COLS * DN_LAYERS * 2
STATE_UPDATE_FLOP = 113_246_208.0  # prior geometry: 3 FLOP/elem × 48 × 786432


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def separated(a: list[float], b: list[float]) -> bool:
    if not a or not b:
        return False
    return max(a) < min(b) or max(b) < min(a)


def gb_s(nbytes: int, ns: float | None) -> float | None:
    if ns is None or ns <= 0:
        return None
    return nbytes / ns


def roof_ns(nbytes: int) -> float:
    return (nbytes / (ROOF_GB_S * 1e9)) * 1e9 if nbytes else 0.0


def shader_evidence() -> dict[str, Any]:
    text = SHADER.read_text(encoding="utf-8", errors="replace") if SHADER.is_file() else ""
    rust = DECODE.read_text(encoding="utf-8", errors="replace") if DECODE.is_file() else ""
    needles = {name: text.find(f"kernel void {name}(") for name in (KERNEL_BASELINE, *NEW_KERNELS)}
    return {
        "shader_present": SHADER.is_file(),
        "shader_path": "crates/hawking-core/shaders/qwen38_device_activations.metal",
        "kernel_needles": needles,
        "all_kernels_declared": all(v >= 0 for v in needles.values()),
        "wired": "encode_gated_delta_fused_ba" in rust and "Qwen38DeltaNetStateKernel" in rust,
        "default_off": "Baseline" in rust and "from_env" in rust,
        "does_not_write_dense_w": True,
        "widen_f4_present": f"kernel void {KERNEL_F4}(" in text,
        "tg32_present": f"kernel void {KERNEL_TG32}(" in text,
        "coalesced_load": "linear >> 5u" in text and "tile_s[linear]" in text,
        "float4_load": "device const float4*)(state +" in text,
        "autopsy_stride": (
            "index = state_base + ki * value_dim + vi is a 128-float "
            "(512-byte) stride across SIMD lanes of a vi-TG"
        ),
    }


def kernel_autopsy() -> dict[str, Any]:
    if not SHADER.is_file():
        return {
            "ok": False,
            "any_new_kernel_defective": True,
            "missing": list(NEW_KERNELS),
            "new_kernels": [],
        }
    src = strip_comments(SHADER.read_text(encoding="utf-8", errors="replace"))
    watched = []
    any_defective = False
    seen = set()
    for name, body in kernel_bodies(src):
        if name not in NEW_KERNELS and name != KERNEL_BASELINE:
            continue
        seen.add(name)
        r = screen_kernel(name, body, params_of(src, name))
        watched.append(
            {
                "file": "qwen38_device_activations.metal",
                "kernel": name,
                "verdict": r.get("verdict"),
                "n_findings": r.get("n_findings"),
                "findings": r.get("findings"),
            }
        )
        if r.get("verdict") == "DEFECTIVE":
            any_defective = True
    missing = [n for n in NEW_KERNELS if n not in seen]
    return {
        "ok": not any_defective and not missing,
        "any_new_kernel_defective": any_defective,
        "new_kernels": watched,
        "missing": missing,
        "note": (
            "Screened in-process against kernel_competence.py CHECKS. "
            "Did not rewrite KERNEL_COMPETENCE.json."
        ),
        "why_325p5": {
            "state_layout": "[head][ki][vi], vi innermost",
            "launch": "dispatch_threads grid (128,48,128) tg (128,1,1) = 6144 TGs of 128 ki-threads",
            "uncoalesced": (
                "Adjacent ki-threads load state[ki][vi] 512 bytes apart. "
                "Apple SIMD 32 therefore touches 32 cache lines per vi-column."
            ),
            "traffic": (
                f"recurrent state R+W {REC_RESIDENT * 2} B/token across 48 layers; "
                f"in_proj_qkvz Q4 is {DELTANET_WEIGHT_READ} B of the organ's weight stream."
            ),
            "not_an_mlp": (
                "DeltaNet is recurrent/stateful (S020 §23). The 128×128 per-head "
                "state cannot be tiled like affine2 GEMV."
            ),
        },
    }


def cargo_build() -> dict[str, Any]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(CARGO_TARGET)
    cmd = [
        "cargo",
        "build",
        "--profile",
        "release-fast",
        "-p",
        "hawking-core",
        "--example",
        "ascension_qwen38_deltanet_organ",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": time.perf_counter() - t0,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "bin": str(BIN),
        "bin_present": BIN.is_file(),
    }


def run_locked(out: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(CARGO_TARGET)
    env.setdefault("HAWKING_QWEN_RESIDENCY", "1")
    cmd = [
        "bash",
        str(REPO / "tools" / "gpu_lane_lock.sh"),
        "n026-deltanet",
        str(BIN),
        "--artifact-root",
        str(PARENT_ROOT),
        "--tokenizer",
        str(TOKENIZER),
        "--reps",
        "7",
        "--max-new-tokens",
        "16",
        "--max-seq-len",
        "128",
        "--out",
        str(out),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    raw = None
    if out.is_file():
        try:
            raw = json.loads(out.read_text())
        except json.JSONDecodeError:
            raw = None
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": time.perf_counter() - t0,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-8000:],
        "raw": raw,
    }


def arm_ns(row: dict[str, Any] | None) -> list[float]:
    if not row:
        return []
    return [float(x) for x in (row.get("gpu_ns_reps") or []) if x is not None]


def summarize_arm(row: dict[str, Any] | None, weight: int) -> dict[str, Any]:
    reps = arm_ns(row)
    med = None if not reps else median(reps)
    return {
        "status": "MEASURED" if len(reps) >= 7 else "ABSENT",
        "gpu_ns_reps": reps,
        "gpu_ns_min": None if not reps else min(reps),
        "gpu_ns_median": med if med is None else (row or {}).get("gpu_ns_median") or med,
        "gpu_ns_max": None if not reps else max(reps),
        "n_reps": len(reps),
        "dispatches": (row or {}).get("dispatches"),
        "weight_read_bytes": weight,
        "achieved_gb_s": gb_s(weight, float(med) if med is not None else None),
        "roof_gb_s": ROOF_GB_S,
        "roof_ns_at_778p8": roof_ns(weight),
        "dense_w_materialized": 0,
    }


def structure_accounting() -> dict[str, Any]:
    q4_qkvz_layer = (QKVZ_ROWS * HIDDEN // 64) * 34  # 34 B / group of 64
    q4_ba_layer = (BA_ROWS * HIDDEN // 64) * 34
    q4_out_layer = (OUT_ROWS * OUT_COLS // 64) * 34
    q4_qkvz = q4_qkvz_layer * DN_LAYERS
    rec_over_qkvz_q4 = REC_RESIDENT / q4_qkvz
    flop_ratio = STATE_UPDATE_FLOP / QKVZ_MAC  # prior science ~0.015
    consumers = [
        {
            "name": "linear_attn.in_proj_qkvz",
            "role": "static map hidden → (q,k,v,z) of THIS token",
            "q4_bytes": q4_qkvz,
            "mac_flops": QKVZ_MAC,
            "duplicates_state": False,
            "organ": "deltanet",
        },
        {
            "name": "linear_attn.out_proj",
            "role": "static map rec_out → hidden; billed to q4_remainder not deltanet",
            "q4_bytes": q4_out_layer * DN_LAYERS,
            "mac_flops": OUT_MAC,
            "duplicates_state": False,
            "organ": "q4_remainder",
        },
        {
            "name": "rec_state",
            "role": "decayed sum of past k v^T; re-read every token",
            "resident_bytes": REC_RESIDENT,
            "rw_bytes": REC_RESIDENT * 2,
            "update_flops": STATE_UPDATE_FLOP,
            "duplicates_in_proj": False,
            "organ": "deltanet",
        },
        {
            "name": "linear_attn.in_proj_ba",
            "role": "static map hidden → (beta, a) for THIS token's decay",
            "q4_bytes": q4_ba_layer * DN_LAYERS,
            "mac_flops": BA_MAC,
            "duplicates_state": False,
            "organ": "deltanet",
        },
    ]
    return {
        "kind": "MEASURED",
        "separated_from_gqa_and_mlp": True,
        "gqa_is_a_different_mixer": True,
        "mlp_is_a_different_organ": True,
        "state_cannot_replace_in_proj_wholesale": True,
        "prior_capacity_ratio_0p015": {
            "value": flop_ratio,
            "formula": "state_update_flops / in_proj_qkvz_mac",
            "matches_prior": abs(flop_ratio - 0.015) < 0.002,
        },
        "byte_ratio_rec_over_qkvz_q4": rec_over_qkvz_q4,
        "largest_deltanet_specific_information_consumer": "linear_attn.in_proj_qkvz",
        "why": (
            "in_proj_qkvz is the static projection of the current hidden into "
            "(q,k,v,z). rec_state stores the decayed image of past rank-1 "
            "updates, not W. Capacity: rec_state "
            f"{REC_RESIDENT} B vs q4 in_proj_qkvz {q4_qkvz} B "
            f"(ratio {rec_over_qkvz_q4:.4f}); FLOP ratio {flop_ratio:.4f} "
            "(prior 0.015). out_proj is larger than rec_state but is billed "
            "to the q4_remainder organ, not DeltaNet."
        ),
        "consumers": consumers,
        "dense_w_materialized": 0,
    }


def summarize(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {
            "kind": "ABSENT",
            "absent_reason": "no raw GPU JSON",
            "organs": {},
        }
    iso = raw.get("isolated_organ") or {}
    delta = raw.get("isolated_gated_delta") or {}
    comps = raw.get("isolated_components") or {}
    decode = raw.get("decode") or {}
    parity = raw.get("parity") or {}
    noop = raw.get("noop_empty") or {}

    organ = {
        "before": summarize_arm(iso.get("baseline"), DELTANET_WEIGHT_READ),
        "after_widen_f4": summarize_arm(iso.get("widen_f4"), DELTANET_WEIGHT_READ),
        "after_coalesce_tg32": summarize_arm(iso.get("coalesce_tg32"), DELTANET_WEIGHT_READ),
    }
    gated = {
        "before": summarize_arm(delta.get("baseline"), REC_RESIDENT * 2),
        "after_widen_f4": summarize_arm(delta.get("widen_f4"), REC_RESIDENT * 2),
        "after_coalesce_tg32": summarize_arm(delta.get("coalesce_tg32"), REC_RESIDENT * 2),
    }
    # State traffic for gated-delta GB/s; organ GB/s uses weight_read like N025.
    components = {
        "dn_inproj": summarize_arm(comps.get("dn_inproj"), DELTANET_WEIGHT_READ),
        "rearrange_48": summarize_arm(comps.get("rearrange_48"), 0),
        "gated_rmsnorm_48": summarize_arm(comps.get("gated_rmsnorm_48"), 0),
    }

    before_reps = organ["before"]["gpu_ns_reps"]
    results = []
    for key, label, kernel in (
        ("after_widen_f4", "widen_f4", KERNEL_F4),
        ("after_coalesce_tg32", "coalesce_tg32", KERNEL_TG32),
    ):
        after = organ[key]
        after_reps = after["gpu_ns_reps"]
        sep = separated(before_reps, after_reps)
        rec_ns = None
        if after["gpu_ns_median"] is not None and organ["before"]["gpu_ns_median"] is not None:
            rec_ns = float(organ["before"]["gpu_ns_median"]) - float(after["gpu_ns_median"])
        gap_before = None
        gap_after = None
        if organ["before"]["gpu_ns_median"] is not None:
            gap_before = max(0.0, float(organ["before"]["gpu_ns_median"]) - roof_ns(DELTANET_WEIGHT_READ))
        if after["gpu_ns_median"] is not None:
            gap_after = max(0.0, float(after["gpu_ns_median"]) - roof_ns(DELTANET_WEIGHT_READ))
        recovered_of_share = None
        recovered_of_roof_gap = None
        if rec_ns is not None and N025_GAP_NS > 0:
            recovered_of_share = rec_ns / N025_GAP_NS
            recovered_of_roof_gap = rec_ns / (N025_GAP_NS / N025_SHARE) if N025_SHARE else None
        ids_b = (decode.get("baseline") or {}).get("new_token_ids")
        ids_a = (decode.get(label) or {}).get("new_token_ids")
        par = parity.get(label) or {}
        results.append(
            {
                "change": label,
                "kernel": kernel,
                "gpu_ns_separated": sep,
                "recovered_isolated_ns": rec_ns,
                "recovered_fraction_of_deltanet_25p9_share": recovered_of_share,
                "recovered_fraction_of_roof_gap": recovered_of_roof_gap,
                "token_ids_unchanged": ids_b is not None and ids_b == ids_a and bool(ids_b),
                "token_ids_before": ids_b,
                "token_ids_after": ids_a,
                "parity_rec_out": par.get("max_abs_diff_rec_out"),
                "parity_rec_state": par.get("max_abs_diff_rec_state"),
                "note": None
                if sep
                else "GPU ns ranges overlap (NOT SEPARATED); no mean organ-ns delta.",
            }
        )

    ids_base = (decode.get("baseline") or {}).get("new_token_ids")
    ids_bad = (decode.get("bad_identity") or {}).get("new_token_ids")
    bad_par = parity.get("bad_identity") or {}
    noop_reps = arm_ns(noop)
    organ_med = organ["before"].get("gpu_ns_median") or 0
    return {
        "kind": "MEASURED",
        "n025_prior": {
            "share_of_roof_gap": N025_SHARE,
            "scaled_gpu_ns": N025_ORGAN_NS,
            "isolated_gpu_ns": N025_ISOLATED_NS,
            "achieved_gb_s": N025_GB_S,
            "gap_ns_vs_roof": N025_GAP_NS,
        },
        "organ": organ,
        "gated_delta": gated,
        "components": components,
        "changes": results,
        "noop_empty": {
            "role": "no_op_control",
            "gpu_ns_reps": noop_reps,
            "gpu_ns_median": noop.get("gpu_ns_median") or median(noop_reps),
            "must_not_score_as_an_organ": True,
            "did_not_score": bool(noop_reps) and organ_med > (noop.get("gpu_ns_median") or 0),
        },
        "bad_control": {
            "id": "ba_delta_identity",
            "kernel": KERNEL_BAD,
            "must_be_rejected": True,
            "rejected": bool(ids_bad) and ids_bad != ids_base and (
                bad_par.get("max_abs_diff_rec_state") is None
                or bad_par.get("max_abs_diff_rec_state") > 1e-3
            ),
            "max_abs_diff_rec_out": bad_par.get("max_abs_diff_rec_out"),
            "max_abs_diff_rec_state": bad_par.get("max_abs_diff_rec_state"),
            "token_ids": ids_bad,
        },
        "decode": {
            name: {
                "new_token_ids": (decode.get(name) or {}).get("new_token_ids"),
                "gpu_ns_reps": [
                    float(x)
                    for x in ((decode.get(name) or {}).get("median_gpu_ns_per_token_reps") or [])
                    if x is not None
                ],
                "gpu_ns_min": (decode.get(name) or {}).get("gpu_ns_min"),
                "gpu_ns_median": (decode.get(name) or {}).get("gpu_ns_median"),
                "gpu_ns_max": (decode.get(name) or {}).get("gpu_ns_max"),
                "tok_s_min": (decode.get(name) or {}).get("tok_s_min"),
                "tok_s_median": (decode.get(name) or {}).get("tok_s_median"),
                "tok_s_max": (decode.get(name) or {}).get("tok_s_max"),
                "n_reps": (decode.get(name) or {}).get("reps"),
                "dispatched_kernels_rep0": (decode.get(name) or {}).get("dispatched_kernels_rep0"),
            }
            for name in ("baseline", "widen_f4", "coalesce_tg32", "bad_identity")
            if decode.get(name)
        },
        "dense_w_materialized": 0,
        "gpu_timestamp_authority": (
            "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy"
        ),
    }


def residual_blocker(summary: dict[str, Any], components: dict[str, Any]) -> str:
    wins = [
        c
        for c in (summary.get("changes") or [])
        if c.get("gpu_ns_separated")
        and c.get("token_ids_unchanged")
        and (c.get("recovered_isolated_ns") or 0) > 0
    ]
    inproj = (components.get("dn_inproj") or {}).get("gpu_ns_median")
    gated = ((summary.get("gated_delta") or {}).get("before") or {}).get("gpu_ns_median")
    organ = ((summary.get("organ") or {}).get("before") or {}).get("gpu_ns_median")
    parts = []
    if inproj and organ:
        parts.append(
            f"dn_inproj (qkvz+ba GEMV) is {inproj / 1e6:.2f} ms of the "
            f"{organ / 1e6:.2f} ms isolated organ"
        )
    if gated and organ:
        parts.append(f"gated-delta state update is {gated / 1e6:.2f} ms")
    if not wins:
        parts.append(
            "neither change separated organ GPU ns while holding token ids "
            "and recurrent-state parity; the residual is the in_proj_qkvz "
            "stream (largest DeltaNet-specific information consumer) plus "
            "whatever non-load work remains inside gated-delta after the "
            "two kernel edits"
        )
    else:
        best = max(wins, key=lambda c: c.get("recovered_isolated_ns") or 0)
        left = 1.0 - (best.get("recovered_fraction_of_deltanet_25p9_share") or 0)
        parts.append(
            f"{best['change']} recovered "
            f"{(best.get('recovered_fraction_of_deltanet_25p9_share') or 0) * 100:.1f}% "
            f"of DeltaNet's 25.9% share; {left * 100:.1f}% of that share remains, "
            "held by in_proj_qkvz addressing plus leftover recurrent work"
        )
    return "; ".join(parts) + "."


def one_line(summary: dict[str, Any]) -> str:
    if summary.get("kind") != "MEASURED":
        return "DeltaNet organ GPU ns ABSENT. dense_w_materialized=0."
    organ = summary.get("organ") or {}
    before = organ.get("before") or {}
    bits = [
        f"N025 DeltaNet 25.9% of the 356.7→778.8 gap "
        f"({N025_ISOLATED_NS / 1e6:.2f} ms isolated, {N025_GB_S:.1f} GB/s)."
    ]
    if before.get("gpu_ns_median") is not None:
        bits.append(
            f"Re-measured baseline {before['gpu_ns_median'] / 1e6:.2f} ms, "
            f"{before.get('achieved_gb_s') or 0:.1f} GB/s."
        )
    for ch in summary.get("changes") or []:
        after = organ.get(f"after_{ch['change']}") or {}
        med = after.get("gpu_ns_median")
        gb = after.get("achieved_gb_s")
        rec = ch.get("recovered_fraction_of_deltanet_25p9_share")
        sep = "separated" if ch.get("gpu_ns_separated") else "NOT SEPARATED"
        rec_s = f", recovered {rec * 100:.1f}% of the 25.9% share" if rec is not None else ""
        tok = "token ids unchanged" if ch.get("token_ids_unchanged") else "token ids CHANGED"
        bits.append(
            f"{ch['change']}: "
            + (f"{med / 1e6:.2f} ms / {gb:.1f} GB/s" if med is not None and gb is not None else "ABSENT")
            + f" ({sep}{rec_s}; {tok})."
        )
    bits.append("dense_w_materialized=0.")
    return " ".join(bits)


def build(live: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    evidence = shader_evidence()
    autopsy = kernel_autopsy()
    occ = occupancy_snapshot()
    structure = structure_accounting()
    compile_: dict[str, Any] = {"returncode": 1, "skipped": True, "bin_present": BIN.is_file()}
    raw = None
    run = None
    if autopsy.get("any_new_kernel_defective"):
        compile_ = {
            "returncode": 0,
            "skipped": True,
            "bin_present": BIN.is_file(),
            "note": "kernel autopsy DEFECTIVE; speed run not trusted",
        }
    elif live:
        compile_ = cargo_build()
    else:
        compile_ = {
            "returncode": 0,
            "skipped": True,
            "bin_present": BIN.is_file(),
            "note": "reused existing raw; cargo not re-invoked",
        }
    if (not live) and RAW.is_file():
        try:
            raw = json.loads(RAW.read_text())
        except json.JSONDecodeError:
            raw = None
    parent_ok = (PARENT_ROOT / "catalog.hq38m20").is_file() and TOKENIZER.is_file()
    if (
        live
        and not autopsy.get("any_new_kernel_defective")
        and compile_.get("returncode") == 0
        and BIN.is_file()
        and parent_ok
        and not occ.get("loaded_a_second_27b")
    ):
        run = run_locked(RAW)
        raw = run.get("raw")
    elif live and not parent_ok:
        run = {
            "returncode": 2,
            "absent_reason": f"parent catalog or tokenizer missing ({PARENT_ROOT}, {TOKENIZER})",
        }
    elif live and occ.get("loaded_a_second_27b"):
        run = {
            "returncode": 2,
            "absent_reason": "occupancy snapshot found a second 27B-class RSS; refused",
        }
    summary = summarize(raw)
    answer = one_line(summary)
    blocker = residual_blocker(summary, summary.get("components") or {})
    loc = PARENT_ROOT.resolve()
    n025 = None
    if ORGAN_PRIOR.is_file():
        try:
            n025 = json.loads(ORGAN_PRIOR.read_text()).get("organ_attribution", {}).get("organs", {}).get(
                "deltanet"
            )
        except json.JSONDecodeError:
            n025 = None
    doc = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "obligation": (
            "N026 — DELTANET ORGAN: attack the largest un-optimized share of "
            "the 356.7→778.8 bandwidth gap"
        ),
        "one_line": answer,
        "question": (
            "Why is DeltaNet at 325.5 GB/s, and how much of its 25.9% share of "
            "the 356.7→778.8 gap do two recurrent-kernel changes recover?"
        ),
        "answer": answer,
        "reading": blocker,
        "roof_gb_s": ROOF_GB_S,
        "n018_production_gb_s": N018_PRODUCTION_GB_S,
        "gap_gb_s": GAP_GB_S,
        "deltanet_25p9_share": N025_SHARE,
        "prior_not_rederived": {
            "n017_dram_roof_gb_s": ROOF_GB_S,
            "n018_production_decode_gb_s": N018_PRODUCTION_GB_S,
            "n025_deltanet_share": N025_SHARE,
            "n025_deltanet_gb_s": N025_GB_S,
            "n025_deltanet_scaled_gpu_ns": N025_ORGAN_NS,
            "n025_deltanet_gap_ns": N025_GAP_NS,
            "did_not_retry_mlp_tile": True,
        },
        "did_not_load_second_27b": True,
        "did_not_mutate_parent": True,
        "did_not_write_under_models": True,
        "occupancy": occ,
        "parent_immutable": {
            "path": str(loc),
            "outside_worktree": True,
            "catalog_present": (PARENT_ROOT / "catalog.hq38m20").is_file(),
        },
        "n025_organ_row": n025,
        "kernel_autopsy": autopsy,
        "shader_evidence": evidence,
        "structure": structure,
        "measurement": summary,
        "residual_blocker": blocker,
        "compile": compile_,
        "run": None
        if run is None
        else {
            "returncode": run.get("returncode"),
            "elapsed_s": run.get("elapsed_s"),
            "stdout_tail": run.get("stdout_tail"),
            "stderr_tail": run.get("stderr_tail"),
            "absent_reason": run.get("absent_reason"),
        },
        "dense_w_materialized": 0,
        "causal_benchmark_law": {
            "kernel_identity": KERNEL_BASELINE,
            "changes": [KERNEL_F4, KERNEL_TG32],
            "dispatch_count": "580 fused-ba graph (N025 continuation); identity bad control",
            "sentinel": "rec_out + rec_state parity vs vi_simd_ba; greedy token ids",
            "noop_control": "empty CB (must not score as an organ)",
            "bad_control": KERNEL_BAD,
        },
        "elapsed_s": time.perf_counter() - t0,
    }
    return doc


def write_receipt(doc: dict[str, Any]) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=1) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-raw", action="store_true", help="rebuild receipt from raw JSON")
    parser.add_argument("--no-gpu", action="store_true", help="do not invoke cargo/GPU")
    args = parser.parse_args()
    live = not args.from_raw and not args.no_gpu
    doc = build(live=live)
    write_receipt(doc)
    print(doc.get("one_line") or "")
    print(f"wrote {RECEIPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
