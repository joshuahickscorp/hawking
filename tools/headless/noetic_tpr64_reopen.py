#!/usr/bin/env python3
"""NNS-011 — tpr64 reconstruction is free; codecs killed for 5.9× are quality-eligible.

The 5.9× per-byte decode penalty was measured on Q80's serial 1-thread-per-row
extract, then transferred onto Qwen3.8 rice without re-measurement. Isolated
tpr64 kernels on real Qwen3.8 activations put reconstruction excess at 0 ns
on 32 of 33 variants. This tool states that result precisely, names every
codec that died for the penalty, and separates genuinely reopened from
still-dead-for-another-reason.

It does not pack, generate, or dispatch Metal. It confirms numbers already
in receipts (git-show if the sparse checkout has not materialized them).

Write: receipts/headless/NOETIC_TPR64_REOPEN.json
Run:   python3 tools/headless/noetic_tpr64_reopen.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "hawking.headless.noetic_tpr64_reopen.v1"
NULL_COSINE = 0.898  # GLM / this-family constant-mean null (NS-013)
ASCENT = "receipts/ascent-2026-08-16"

REQUIRED = [
    f"{ASCENT}/QWEN38_RECONSTRUCTION_IS_FREE.json",
    f"{ASCENT}/QWEN38_RECON_MEASURED.json",
    f"{ASCENT}/Q80_MIXED_RECONSTRUCTION_WALL.json",
    f"{ASCENT}/Q80_RECONSTRUCTION_WON.json",
    f"{ASCENT}/NEGATIVE_SCIENCE_REGISTER.json",
    f"{ASCENT}/QWEN38_BPW_DESCENT.json",
    f"{ASCENT}/QWEN38_BPW_DESCENT_REVIEW.json",
    f"{ASCENT}/QWEN_ATTENTION_DENSITY_VERDICT.json",
    f"{ASCENT}/Q80_MIXED_GENERATE.json",
    f"{ASCENT}/QWEN38_NATIVE_MIXED_READER.json",
    f"{ASCENT}/QWEN38_NATIVE_MIXED_2P0_GENERATE.json",
    f"{ASCENT}/PROMOTION_QUEUE.json",
    f"{ASCENT}/QWEN38_TOKEN_NS_DN_VI_SIMD.json",
    f"{ASCENT}/QWEN38_COMPLETE_TOKEN_WALL.json",
    f"{ASCENT}/STALE_BASELINE_PATTERN.json",
    f"{ASCENT}/G109_ORGAN_GEOMETRY.json",
]


def find_repo() -> Path:
    env = os.environ.get("HAWKING_REPO")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "Cargo.toml").exists() and (p / "tools" / "headless").is_dir():
            return p
        marker = p / ASCENT / "QWEN38_RECONSTRUCTION_IS_FREE.json"
        if marker.exists():
            return p
    fallback = Path("/Users/scammermike/Downloads/hawking-copy")
    return fallback if fallback.exists() else Path.cwd()


REPO = find_repo()
COPY = Path("/Users/scammermike/Downloads/hawking-copy")


def git_head() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return (r.stdout or "").strip() or "UNKNOWN"


def load_json(rel: str) -> tuple[dict[str, Any], str]:
    """Load a tracked JSON receipt. Sparse checkouts are not evidence of absence."""
    candidates = [
        REPO / rel,
        COPY / rel if COPY.exists() else None,
    ]
    for p in candidates:
        if p is not None and p.is_file():
            return json.loads(p.read_text()), f"disk:{p}"
    r = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if r.returncode == 0 and r.stdout:
        return json.loads(r.stdout), f"git-show:HEAD:{rel}"
    raise FileNotFoundError(
        f"{rel} not on disk under {REPO} and git-show HEAD:{rel} failed "
        f"(exit {r.returncode}). Sparse checkout is not evidence the file "
        f"does not exist."
    )


def gget(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("/"):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def spread(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "min": min(xs),
        "max": max(xs),
        "median": median(xs),
        "range_ns": max(xs) - min(xs),
        "protocol": "isolated-kernel 5-rep GPU timestamps, NOT paired alternating token-wall",
    }


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-30)


def gain_axis(a: list[float], b: list[float]) -> float:
    """min(r, 1/r) on vector norms — the doctor-gate magnitude axis."""
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    r = nb / (na + 1e-30)
    return min(r, 1.0 / (r + 1e-30))


def scale_invariance_probe() -> dict[str, Any]:
    """Show cosine accepting 0.01·W and gain rejecting it. Pure Python, no numpy."""
    w = [1.0, -2.0, 3.5, 0.25, 8.0, -0.5]
    wh = [0.01 * x for x in w]
    c = cosine(w, wh)
    g = gain_axis(w, wh)
    return {
        "construction": "Wh = 0.01 * W on a 6-vector (same construction as tools/gravity_doctor_gate.py:_gain)",
        "cosine_Wh_vs_W": c,
        "cosine_is_1": abs(c - 1.0) < 1e-12,
        "gain_min_r_1_over_r": g,
        "gain_is_0_01": abs(g - 0.01) < 1e-12,
        "verdict": (
            "cosine ACCEPTS the scaled artifact (1.000000). "
            "gain REJECTS it (0.01). Any fidelity number in this receipt "
            "that is cosine-only is flagged."
        ),
        "doctor_gate_citation": (
            "tools/gravity_doctor_gate.py:_gain — L0 gate_proj Wh=0.01*W scores "
            "observed/probed/worst_unit = 1.000000 with relative weight error 0.9898"
        ),
    }


def confirm(field: str, observed: Any, expected: Any, tol: float | None = None) -> dict[str, Any]:
    if tol is not None and isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
        ok = abs(float(observed) - float(expected)) <= tol
    else:
        ok = observed == expected
    return {
        "field": field,
        "observed": observed,
        "expected": expected,
        "confirmed": ok,
    }


def family_of(name: str) -> str:
    n = name.split("/")[0]
    if n.startswith("uniform_q"):
        return n
    if n.startswith("prod_q4"):
        return "prod_q4_nibble_g64"
    if n.startswith("binary"):
        return "binary_g128"
    if n.startswith("ternary"):
        return "ternary_t0.7_g128"
    if n.startswith("additive"):
        return "additive_q2q2_g64"
    if n.startswith("hadamard"):
        return "hadamard_q2_g128"
    if n.startswith("rice"):
        return "rice_q1_rms_2pct"
    if n.startswith("hgravs"):
        return "hgravs01_r160_q3"
    if n.startswith("f32"):
        return "f32_tpr64"
    return n


def kernel_class(name: str, kernel: str) -> str:
    if "serial" in name or "serial" in kernel:
        return "SERIAL_ARTIFACT"
    if "tg256" in name or "tg256" in kernel:
        return "TG256"
    if "hadamard" in name or "walsh" in kernel:
        return "TPR64_PLUS_WH"
    if "hgravs" in name or "L@(R@x)" in kernel:
        return "TWO_STAGE_ALGEBRA"
    if "tpr64" in name or "tpr64" in kernel or name.startswith("ternary") or name.startswith("additive"):
        return "TPR64_INREGISTER"
    return "OTHER"


def analyze_measured(measured: dict) -> dict[str, Any]:
    organs = []
    all_v = []
    for org in measured["organs"]:
        rows = []
        for v in org["variants"]:
            corr = v.get("correctness") or {}
            cos = corr.get("cosine") if corr else None
            max_abs = corr.get("max_abs") if corr else None
            gpu = [float(x) for x in v.get("gpu_ns") or []]
            rec = {
                "organ": org["name"],
                "rows": org["rows"],
                "cols": org["cols"],
                "name": v["name"],
                "family": family_of(v["name"]),
                "kernel": v["kernel"],
                "kernel_class": kernel_class(v["name"], v["kernel"]),
                "note": v.get("note"),
                "median_gpu_ns": v["median_gpu_ns"],
                "recon_excess_ns": v["recon_excess_ns"],
                "bandwidth_floor_ns": v.get("bandwidth_floor_ns"),
                "packed_gbps": v.get("packed_gbps"),
                "traffic_bytes": v.get("traffic_bytes"),
                "storage_bytes": v.get("storage_bytes"),
                "storage_bpw": v.get("storage_bpw"),
                "cosine": cos,
                "max_abs": max_abs,
                "cosine_is_one": (cos is not None and abs(cos - 1.0) < 1e-6),
                "max_abs_tiny": (max_abs is not None and abs(max_abs) < 1e-5),
                "spread": spread(gpu),
            }
            if rec["storage_bytes"] and rec["traffic_bytes"]:
                rec["traffic_over_storage"] = rec["traffic_bytes"] / rec["storage_bytes"]
            rows.append(rec)
            all_v.append(rec)
        organs.append(
            {
                "name": org["name"],
                "rows": org["rows"],
                "cols": org["cols"],
                "n_variants": len(rows),
                "variants": rows,
            }
        )

    f32 = [v for v in all_v if v["family"] == "f32_tpr64"]
    f32_ns = {v["organ"]: v["median_gpu_ns"] for v in f32}
    nonzero = [v for v in all_v if v["recon_excess_ns"] != 0]
    cos_one = [v for v in all_v if v["cosine_is_one"]]
    cos_none = [v for v in all_v if v["cosine"] is None]
    cos_not_one = [v for v in all_v if v["cosine"] is not None and not v["cosine_is_one"]]
    mag_fail = [v for v in all_v if v["cosine_is_one"] and not v["max_abs_tiny"]]

    # Strict "same ns as f32 tpr64" on GATE: the 15124–15541 band in the summary.
    gate_f32 = f32_ns.get("gate")
    in_gate_band = [
        v
        for v in all_v
        if v["organ"] == "gate"
        and v["kernel_class"] == "TPR64_INREGISTER"
        and gate_f32
        and abs(v["median_gpu_ns"] - gate_f32) <= 500
    ]
    tpr64_inreg = [v for v in all_v if v["kernel_class"] == "TPR64_INREGISTER"]
    tg256 = [v for v in all_v if v["kernel_class"] == "TG256"]
    serial = [v for v in all_v if v["kernel_class"] == "SERIAL_ARTIFACT"]
    hadamard = [v for v in all_v if v["family"] == "hadamard_q2_g128"]
    rice_csr = [v for v in all_v if v["name"].endswith("csr_inregister")]
    rice_serial = [v for v in all_v if "serial_one_thread" in v["name"]]
    hgravs = [v for v in all_v if v["family"] == "hgravs01_r160_q3"]

    return {
        "n_variants": len(all_v),
        "organs": organs,
        "f32_control_tpr64_ns": f32_ns,
        "recon_excess_nonzero": [
            {
                "organ": v["organ"],
                "name": v["name"],
                "recon_excess_ns": v["recon_excess_ns"],
                "median_gpu_ns": v["median_gpu_ns"],
                "cosine": v["cosine"],
                "max_abs": v["max_abs"],
                "note": v["note"],
            }
            for v in nonzero
        ],
        "counts": {
            "total": len(all_v),
            "recon_excess_zero": len(all_v) - len(nonzero),
            "cosine_approx_1": len(cos_one),
            "cosine_none": len(cos_none),
            "cosine_not_1": len(cos_not_one),
            "cosine_1_but_max_abs_not_tiny": len(mag_fail),
            "tpr64_inregister": len(tpr64_inreg),
            "tg256": len(tg256),
            "serial_artifact": len(serial),
        },
        "gate_tpr64_inregister_within_500ns_of_f32": [
            {"name": v["name"], "median_gpu_ns": v["median_gpu_ns"]} for v in in_gate_band
        ],
        "hadamard": [
            {
                "organ": v["organ"],
                "median_gpu_ns": v["median_gpu_ns"],
                "f32_ns": f32_ns.get(v["organ"]),
                "delta_vs_f32_ns": v["median_gpu_ns"] - f32_ns[v["organ"]] if v["organ"] in f32_ns else None,
                "cosine": v["cosine"],
                "max_abs": v["max_abs"],
                "recon_excess_ns": v["recon_excess_ns"],
            }
            for v in hadamard
        ],
        "rice_csr": rice_csr,
        "rice_serial": [
            {"organ": v["organ"], "median_gpu_ns": v["median_gpu_ns"], "cosine": v["cosine"]}
            for v in rice_serial
        ],
        "hgravs": hgravs,
        "tg256_gate_ns": [v["median_gpu_ns"] for v in tg256 if v["organ"] == "gate"],
        "serial_q4_gate_ns": [
            v["median_gpu_ns"]
            for v in serial
            if v["organ"] == "gate" and "rice" not in v["name"]
        ],
        "what_free_means": {
            "recon_excess_definition": (
                "recon_excess_ns = max(0, median_gpu_ns - bandwidth_floor_ns). "
                "Zero means the kernel is not slower than streaming the packed bytes "
                "at the honest ceiling. Negative values are clamped to 0, so a kernel "
                "slower than f32 but faster than its (tiny) byte-floor still reads as free."
            ),
            "same_ns_as_f32": (
                "The summary claim '15,124–15,541 ns = uncompressed f32' is the GATE "
                "tpr64 in-register family (q4/q3/q2/binary/ternary/additive/rice-CSR). "
                "It is NOT hadamard (gate 17333), NOT tg256 (~26541), NOT serial, "
                "NOT hgravs01."
            ),
            "kernel_fidelity_not_teacher_quality": (
                "cosine ≈ 1.0 here is packed-kernel vs reference unpack, not vs BF16 W@x. "
                "hgravs cosine 0.9317 is the exception: two-stage algebra vs the real "
                "activation product on rank-160, i.e. approximation error, not unpack error."
            ),
        },
    }


def descent_quality(descent: dict) -> dict[str, Any]:
    rows = []
    for r in descent.get("summary", {}).get("by_role_codec") or []:
        rows.append(
            {
                "role": r.get("role"),
                "codec": r.get("codec"),
                "physical_bpw_mean": r.get("physical_bpw_mean"),
                "hold_min": r.get("hold_min"),
                "hold_mean": r.get("hold_mean"),
                "weight_cosine_min": r.get("weight_cosine_min"),
                "frac_clears_0.86": r.get("frac_clears_q80_bar"),
                "frac_clears_0.90": r.get("frac_clears_moderate"),
                "recon_penalty_assumed": r.get("recon_penalty_assumed"),
                "hold_mean_minus_null": (
                    None
                    if r.get("hold_mean") is None
                    else r["hold_mean"] - NULL_COSINE
                ),
                "beats_null_on_mean": (
                    None
                    if r.get("hold_mean") is None
                    else r["hold_mean"] > NULL_COSINE
                ),
            }
        )
    catalog = descent.get("codec_catalog") or []
    rice_pen = [c for c in catalog if c.get("recon_penalty") == 5.9]
    return {
        "catalog": catalog,
        "catalog_with_penalty_5_9": rice_pen,
        "by_role_codec": rows,
        "coherence_floor": descent.get("coherence_floor"),
        "candidate_rice_containing": [
            {
                "codec": c.get("codec"),
                "projected_bpw": c.get("projected_bpw"),
                "recon_penalty": c.get("recon_penalty"),
                "projected_tps": c.get("projected_tps"),
                "projected_tps_cost_adjusted": c.get("projected_tps_cost_adjusted"),
                "verdict": c.get("verdict"),
            }
            for c in descent.get("candidate_table") or []
            if "rice" in (c.get("codec") or "").lower()
            or "binary_rice" in (c.get("codec") or "").lower()
            or c.get("recon_penalty") == 5.9
            or (isinstance(c.get("recon_penalty"), (int, float)) and c.get("recon_penalty", 0) > 2)
        ],
        "bars": descent.get("bars"),
        "claim_boundary": descent.get("claim_boundary"),
        "activation": {
            "not_synthetic": gget(descent, "activation/not_synthetic"),
            "n_tokens": gget(descent, "activation/n_tokens"),
            "fit_n": gget(descent, "activation/fit_n"),
            "hold_n": gget(descent, "activation/hold_n"),
            "path": gget(descent, "activation/path"),
            "sha256_self": gget(descent, "activation/sha256_self"),
        },
    }


def mixed_reader_kernels(reader: dict) -> dict[str, Any]:
    landed = reader.get("what_landed") or {}
    mlp = landed.get("mlp") or {}
    gen = reader.get("mixed_2p0_v1_native_generate") or {}
    return {
        "gate_proj_bind": mlp.get("gate_proj"),
        "up_proj_bind": mlp.get("up_proj"),
        "down_proj_bind": mlp.get("down_proj"),
        "uses_tpr64": "tpr64" in json.dumps(mlp).lower(),
        "uses_tg256": "tg256" in json.dumps(mlp).lower(),
        "coherence_verdict": gen.get("coherence_verdict"),
        "coherence_verdict_plain": gen.get("coherence_verdict_plain"),
        "fallbacks_total": gen.get("fallbacks_total"),
        "dense_w_materialized_total": gen.get("dense_w_materialized_total"),
        "reconstruct_to_q4": gen.get("reconstruct_to_q4"),
    }


def classify_codecs(
    measured: dict,
    descent_q: dict,
    ns: dict,
    wall: dict,
    won: dict,
    attn: dict,
    q80_gen: dict,
    reader_k: dict,
    review: dict,
) -> list[dict[str, Any]]:
    """Every codec whose kill citation includes the 5.9× penalty."""
    ns006 = next((e for e in ns.get("entries") or [] if e.get("id") == "NS-006"), {})
    ns018 = next((e for e in ns.get("entries") or [] if e.get("id") == "NS-018"), {})
    ns019 = next((e for e in ns.get("entries") or [] if e.get("id") == "NS-019"), {})
    ns031 = next((e for e in ns.get("entries") or [] if e.get("id") == "NS-031"), {})

    def role_row(codec: str, role: str) -> dict | None:
        for r in descent_q["by_role_codec"]:
            if r["codec"] == codec and r["role"] == role:
                return r
        return None

    rice_gate = role_row("rice_q1_rms_2pct", "gate_proj")
    rice_up = role_row("rice_q1_rms_2pct", "up_proj")
    rice_down = role_row("rice_q1_rms_2pct", "down_proj")
    rice_attn = role_row("rice_q1_rms_2pct", "attn_in")
    bin_down = role_row("binary_g128", "down_proj")
    bin_up = role_row("binary_g128", "up_proj")

    rice_csr = measured["rice_csr"]
    rice_csr_free = all(v["recon_excess_ns"] == 0 for v in rice_csr) and bool(rice_csr)
    hgravs = measured["hgravs"]

    cost_classes = attn.get("reconstruction_cost_classes") or {}
    codec_appl = attn.get("codec_applicability") or {}

    out: list[dict[str, Any]] = []

    out.append(
        {
            "codec": "rice_q1_rms_2pct / HGRAVR02 (in-register CSR, tpr64)",
            "family": "HGRAVR02",
            "killing_receipts": [
                {
                    "path": f"{ASCENT}/QWEN38_BPW_DESCENT.json",
                    "field": "codec_catalog[rice_q1_rms_2pct].recon_penalty",
                    "number": 5.9,
                    "how": "only catalog entry priced at 5.9; 'Q80 paid 5.9x / byte'",
                },
                {
                    "path": f"{ASCENT}/Q80_MIXED_RECONSTRUCTION_WALL.json",
                    "field": "DENSITY_IS_COSTING_SPEED.slowdown_per_byte_x",
                    "number": 5.9,
                    "how": "named rice_q1 as needing expensive per-weight reconstruction",
                },
                {
                    "path": f"{ASCENT}/NEGATIVE_SCIENCE_REGISTER.json",
                    "field": "entries/NS-006/what_was_measured/slowdown_per_byte_x",
                    "number": 5.9,
                    "how": "density-is-velocity refutation; mixed codecs include rice_q1",
                },
            ],
            "verdict": "REOPENED",
            "why_reopened": (
                "disc_binary_csr_tpr64 on real post-norm X matches f32 gate ns "
                f"({rice_csr[0]['median_gpu_ns'] if rice_csr else 'n/a'} vs "
                f"{measured['f32_control_tpr64_ns'].get('gate')}) with recon_excess_ns=0 "
                "and kernel cosine 1.0 at max_abs 6e-8. The 5.9× was the serial extract, "
                "not the codec. Bind-time CSR expand is NS-018-allowed (~84 KiB/up)."
            ),
            "second_reason": None,
            "quality_at_bpw": {
                "storage_bpw": rice_gate["physical_bpw_mean"] if rice_gate else None,
                "gate_hold_min": rice_gate["hold_min"] if rice_gate else None,
                "gate_hold_mean": rice_gate["hold_mean"] if rice_gate else None,
                "gate_beats_null_mean": rice_gate["beats_null_on_mean"] if rice_gate else None,
                "up_hold_min": rice_up["hold_min"] if rice_up else None,
                "up_hold_mean": rice_up["hold_mean"] if rice_up else None,
                "up_beats_null_mean": rice_up["beats_null_on_mean"] if rice_up else None,
                "down_hold_min": rice_down["hold_min"] if rice_down else None,
                "down_hold_mean": rice_down["hold_mean"] if rice_down else None,
                "down_beats_null_mean": rice_down["beats_null_on_mean"] if rice_down else None,
                "null_cosine": NULL_COSINE,
                "note": (
                    "gate mean 0.9015 is +0.0035 over the 0.898 null — not a GO "
                    "(NS-013). up/down means sit BELOW the null. Generation is the gate. "
                    "mixed-2p0 used rice on up as part of a bundle that collapsed; "
                    "that does not isolate rice-gate-only."
                ),
            },
            "geometry_covered": True,
            "geometry_note": (
                "COVERED on isolated disc_binary_csr_tpr64 for gate 17408×5120 and "
                "down 5120×17408. NOT the shipped mixed reader (tg256 / simd_bytes)."
            ),
            "storage_vs_active": {
                "gate_storage_bytes": rice_csr[0].get("storage_bytes") if rice_csr else None,
                "gate_traffic_bytes": rice_csr[0].get("traffic_bytes") if rice_csr else None,
                "gate_traffic_over_storage": (
                    rice_csr[0].get("traffic_over_storage") if rice_csr else None
                ),
                "note": "CSR bind-expand raises traffic ~1.39× over stored bytes. Report both.",
            },
            "rice_csr_free_confirmed": rice_csr_free,
        }
    )

    out.append(
        {
            "codec": "rice_q1_rms_2pct / HGRAVR02 serial one-thread bitstream",
            "family": "HGRAVR02_SERIAL",
            "killing_receipts": [
                {
                    "path": f"{ASCENT}/NEGATIVE_SCIENCE_REGISTER.json",
                    "field": "entries/NS-031",
                    "number": None,
                    "how": "rice_q1 serial bitstream expand on the per-token path — never",
                },
                {
                    "path": f"{ASCENT}/QWEN38_RECON_MEASURED.json",
                    "field": "variants[rice_q1_rms_2pct/serial_one_thread].note",
                    "number": None,
                    "how": "ARTIFACT PATH: one thread walks every outlier. Not used for ranking. cosine=None",
                },
            ],
            "verdict": "STILL_DEAD",
            "why_reopened": None,
            "second_reason": (
                f"NS-031: {ns031.get('retry_when', 'never as a per-token kernel')} "
                "Free reconstruction at tpr64 is a different kernel. Serial rice is "
                "the original 5.9× vehicle, not the reopen."
            ),
            "quality_at_bpw": None,
            "geometry_covered": False,
            "geometry_note": "Serial 1-thread walk is the DISPROVEN PATH. Do not reopen it.",
        }
    )

    out.append(
        {
            "codec": "HGRAVR02 rice residual on attention GEMVs",
            "family": "HGRAVR02_ATTENTION",
            "killing_receipts": [
                {
                    "path": f"{ASCENT}/QWEN_ATTENTION_DENSITY_VERDICT.json",
                    "field": "codec_applicability.HGRAVR02_binary_residual.reconstruction",
                    "number": 5.9,
                    "how": "EXPENSIVE_SCATTER — a LOSS even if quality passed (mixed 5.9x slower per byte)",
                },
                {
                    "path": f"{ASCENT}/QWEN_ATTENTION_DENSITY_VERDICT.json",
                    "field": "reconstruction_cost_classes.LOSS_EVEN_IF_QUALITY_PASSED",
                    "number": None,
                    "how": "HGRAVR02 listed as LOSS_EVEN_IF_QUALITY_PASSED",
                },
            ],
            "verdict": "STILL_DEAD",
            "why_reopened": (
                "Cost veto is lifted at tpr64 in-register CSR on LARGE sequential GEMVs. "
                "That does not make attention rice quality-legal."
            ),
            "second_reason": (
                "Quality: attention bar is mean row output cosine ≥ 0.990 vs BF16. "
                f"HGRAVR02 max cosine 0.958 (q80 L3 q), typical 0.82–0.91, FAILS 0.99 "
                f"everywhere. Qwen3.8 attn_in rice hold_min="
                f"{rice_attn['hold_min'] if rice_attn else 'n/a'} vs 0.99. "
                "encode_yes_recipe_no: 2% rice residual was fit for up_proj, not Q/K/V/O."
            ),
            "quality_at_bpw": {
                "attn_in_bpw": rice_attn["physical_bpw_mean"] if rice_attn else None,
                "attn_in_hold_min": rice_attn["hold_min"] if rice_attn else None,
                "attn_in_hold_mean": rice_attn["hold_mean"] if rice_attn else None,
                "required_bar": 0.99,
            },
            "geometry_covered": False,
            "geometry_note": (
                "tpr64 free-recon was measured on MLP gate/down, not on k_proj 1024×5120 "
                "(G109: small organs are launch-bound, 2.5× worse ps/element)."
            ),
        }
    )

    out.append(
        {
            "codec": "binary_g128 / HGRAVB01",
            "family": "HGRAVB01",
            "killing_receipts": [
                {
                    "path": f"{ASCENT}/Q80_MIXED_RECONSTRUCTION_WALL.json",
                    "field": "DENSITY_IS_COSTING_SPEED.claim",
                    "number": 5.9,
                    "how": "lumped 'binary / rice_q1 / low-rank' as expensive reconstruction",
                },
                {
                    "path": f"{ASCENT}/NEGATIVE_SCIENCE_REGISTER.json",
                    "field": "entries/NS-006/why_it_failed",
                    "number": 5.9,
                    "how": "Binary / rice_q1 / low-rank need expensive per-weight reconstruction",
                },
            ],
            "verdict": "STILL_DEAD",
            "why_reopened": (
                "The 5.9× attribution was a method artifact. Descent priced binary at "
                "recon_penalty=1.0 CHEAP_INREGISTER. disc_binary_tpr64 recon_excess=0, "
                "gate 15416 ns vs f32 15125. Q80_RECONSTRUCTION_WON: 'The codecs were "
                "never the cause.'"
            ),
            "second_reason": (
                "Quality, independently of cost. Descent verdict REJECT QUALITY: "
                f"down hold_min={bin_down['hold_min'] if bin_down else 'n/a'}, "
                f"up hold_min={bin_up['hold_min'] if bin_up else 'n/a'}. "
                "Attention HGRAVB01 max cosine 0.946, typical 0.75–0.88, FAILS 0.99. "
                "mixed-2p0 native generate used binary gate and collapsed to newline/"
                "punctuation salad (INCOHERENT, 0 fallbacks, reconstruct_to_q4=false)."
            ),
            "quality_at_bpw": {
                "bpw": 1.125,
                "down_hold_min": bin_down["hold_min"] if bin_down else None,
                "up_hold_min": bin_up["hold_min"] if bin_up else None,
            },
            "geometry_covered": True,
            "geometry_note": "tpr64 in-register binary is cost-free on the measured MLP shapes. Quality still dead.",
        }
    )

    out.append(
        {
            "codec": "HGRAVS01 reconstruct-W (materialize dense W then multiply)",
            "family": "HGRAVS01_MATERIALIZE",
            "killing_receipts": [
                {
                    "path": f"{ASCENT}/Q80_MIXED_RECONSTRUCTION_WALL.json",
                    "field": "DENSITY_IS_COSTING_SPEED.claim",
                    "number": 5.9,
                    "how": "lumped low-rank with rice/binary as expensive reconstruction",
                },
                {
                    "path": f"{ASCENT}/QWEN_ATTENTION_DENSITY_VERDICT.json",
                    "field": "reconstruction_cost_classes.LOSS_EVEN_IF_QUALITY_PASSED",
                    "number": None,
                    "how": "HGRAVS01 materialized listed as LOSS",
                },
                {
                    "path": f"{ASCENT}/NEGATIVE_SCIENCE_REGISTER.json",
                    "field": "entries/NS-018 and NS-019",
                    "number": None,
                    "how": "cache decoded W forbidden; reconstruct down W then multiply already refuted",
                },
            ],
            "verdict": "STILL_DEAD",
            "why_reopened": None,
            "second_reason": (
                f"NS-018: {ns018.get('retry_when', '')} "
                f"NS-019: {ns019.get('retry_when', '')} "
                "Free reconstruction of a packed code is not permission to materialize W."
            ),
            "quality_at_bpw": None,
            "geometry_covered": False,
            "geometry_note": "Not a tpr64 in-register codec path.",
        }
    )

    h0 = hgravs[0] if hgravs else None
    out.append(
        {
            "codec": "HGRAVS01 fused y=L@(R@x) two-stage (r160 q3)",
            "family": "HGRAVS01_TWOSTAGE",
            "killing_receipts": [
                {
                    "path": f"{ASCENT}/Q80_MIXED_RECONSTRUCTION_WALL.json",
                    "field": "DENSITY_IS_COSTING_SPEED.claim",
                    "number": 5.9,
                    "how": "lumped low-rank into the 5.9× reconstruct class",
                }
            ],
            "verdict": "STILL_DEAD",
            "why_reopened": (
                "Descent already treated two-stage as CHEAP_ALGEBRA (not 5.9×). "
                "The tpr64 free-recon result does not transfer: this IS the 33rd variant."
            ),
            "second_reason": (
                f"Measured NOT free. down hgravs01_r160_q3 median_gpu_ns="
                f"{h0['median_gpu_ns'] if h0 else 'n/a'} vs f32 down "
                f"{measured['f32_control_tpr64_ns'].get('down')}, recon_excess_ns="
                f"{h0['recon_excess_ns'] if h0 else 'n/a'} "
                f"(~19.8× vs its bandwidth floor). cosine="
                f"{h0['cosine'] if h0 else 'n/a'} max_abs="
                f"{h0['max_abs'] if h0 else 'n/a'} on one real token — no doctor-gate "
                "health verdict. NS-019 remaining cost is occupancy of the two-stage "
                "factor matvec. Attention: 0 clears of 0.99 at ranks whose BPW beats Q4. "
                "Production mixed reader keeps HGRAVS on simd3 two-stage, not tpr64."
            ),
            "quality_at_bpw": {
                "local_bpw_quote": 0.13,
                "health_verdict": "UNSCORED — 0.13 is a COMPONENT of down_proj only; 223 structured rows below 0.5 local BPW had healthy=true: 0",
                "recon_measured_cosine": h0["cosine"] if h0 else None,
            },
            "geometry_covered": False,
            "geometry_note": "The one variant reconstruction-is-free excludes. Do not reopen a family on the exception.",
        }
    )

    out.append(
        {
            "codec": "Q80 mixed-1p5 vehicle (gate binary + up rice_q1 + down hgravs01, nonexpert q8)",
            "family": "Q80_MIXED_1P5",
            "killing_receipts": [
                {
                    "path": f"{ASCENT}/Q80_MIXED_RECONSTRUCTION_WALL.json",
                    "field": "DENSITY_IS_COSTING_SPEED.slowdown_per_byte_x",
                    "number": 5.9,
                    "how": "THE 5.9× measurement: q4 15.2 GB/s vs mixed 2.57 GB/s; token 225 vs 1171 ms",
                },
                {
                    "path": f"{ASCENT}/NEGATIVE_SCIENCE_REGISTER.json",
                    "field": "entries/NS-006",
                    "number": 5.9,
                    "how": "density is velocity, refuted on this vehicle",
                },
            ],
            "verdict": "ALREADY_CONSUMED",
            "why_reopened": (
                "Quality was never the kill: mixed-1p5 generated COHERENT text at "
                f"{gget(q80_gen, 'artifact/complete_physical_bpw')} complete_physical_bpw "
                f"(status={q80_gen.get('status')}, codec="
                f"{gget(q80_gen, 'execution/weight_codec')}). "
                "Speed kill was the serial extract. Q80_RECONSTRUCTION_WON: occupancy "
                "tiles took gpu_matvec 867.04 → 36.60 ms (23.7×) without changing a codec. "
                "NS-006 retry_when (in-register / fused) already holds on this vehicle."
            ),
            "second_reason": (
                "Do not cite tpr64-on-Qwen3.8 as the Q80 win. Q80 experts are 512×2048; "
                "the shipped Q80 fix is occupancy tiles, not geo_tpr64_tg128. G109: small "
                "organs are launch-bound. Transferring 'reconstruction is free' onto "
                "512×2048 without a same-shape remeasure is the trap."
            ),
            "quality_at_bpw": {
                "complete_physical_bpw": gget(q80_gen, "artifact/complete_physical_bpw"),
                "coherence_class": q80_gen.get("coherence_class"),
                "generation_gate": True,
            },
            "geometry_covered": False,
            "geometry_note": "Q80 tile geometry ≠ Qwen3.8 tpr64 on 17408-row GEMVs.",
        }
    )

    out.append(
        {
            "codec": "Qwen3.8 mixed-2p0-v1 (same Q80 recipe on dense MLP: binary gate / rice up / hgravs down / q4 attn)",
            "family": "QWEN38_MIXED_2P0",
            "killing_receipts": [
                {
                    "path": f"{ASCENT}/QWEN38_BPW_DESCENT.json",
                    "field": "candidate_table[REF_sibling_binary_rice_hgravs01_q4rest]",
                    "number": 2.925,
                    "how": "blended recon_penalty from rice 5.9×; adjusted 20.8 TPS called a REGRESSION vs Q4 29.8",
                },
                {
                    "path": f"{ASCENT}/QWEN38_BPW_DESCENT_REVIEW.json",
                    "field": "A_CAVEAT_THAT_MAY_INVALIDATE_THE_SCREEN",
                    "number": 5.9,
                    "how": "review: screen penalised rice on a number that was a kernel artifact",
                },
            ],
            "verdict": "STILL_DEAD",
            "why_reopened": (
                "Cost veto on the isolated tpr64 kernels is lifted. The review's "
                "re-screen-with-penalty-removed is the correct cost move."
            ),
            "second_reason": (
                "Quality, natively, after the mixed reader shipped. "
                f"coherence_verdict={reader_k.get('coherence_verdict')} "
                "on 6/6 prompts (newline/punctuation collapse). fallbacks=0, "
                "dense_w_materialized=0, reconstruct_to_q4=false — attributable to the "
                "packed representation plus Q80 occupancy-tile numerics, not double quant. "
                "Second geometry reason: shipped reader binds "
                f"gate={reader_k.get('gate_proj_bind')} ; "
                f"up={reader_k.get('up_proj_bind')} ; "
                f"down={reader_k.get('down_proj_bind')} — tg256/simd_bytes/simd3, NOT tpr64. "
                f"uses_tpr64={reader_k.get('uses_tpr64')} uses_tg256={reader_k.get('uses_tg256')}."
            ),
            "quality_at_bpw": {
                "sibling_projected_bpw": 2.0853,
                "native_generate": "INCOHERENT",
            },
            "geometry_covered": False,
            "geometry_note": (
                "Isolated tpr64 discriminator kernels cover the codec math. The production "
                "mixed path does not use those kernels."
            ),
            "review_caveat": gget(review, "A_CAVEAT_THAT_MAY_INVALIDATE_THE_SCREEN"),
        }
    )

    # Named so the 5.9× class is not silently widened to q2/q3/ternary/hadamard/additive.
    out.append(
        {
            "codec": "uniform_q2/q3, ternary_t0.7, hadamard_q2, additive_q2q2 (NOT 5.9×-killed)",
            "family": "NOT_IN_CLASS",
            "killing_receipts": [],
            "verdict": "NOT_KILLED_FOR_5_9X",
            "why_reopened": None,
            "second_reason": (
                "Descent priced these at 1.0–1.2, not 5.9. q3 is the cheap coherence floor "
                "(hold_min 0.9679 at 3.25 BPW). ternary is the best cheap 2-bit rung. "
                "hadamard is worse than q2 on hold cosine AND the recon-measured cosine 1.0 "
                "hides max_abs 0.89/0.56 (scale trap). Do not reopen them under NNS-011; "
                "they were never in this graveyard."
            ),
            "quality_at_bpw": None,
            "geometry_covered": True,
            "geometry_note": "tpr64 in-register for q2/q3/ternary/additive is free. hadamard pays WH(x).",
        }
    )
    return out


def geometry_transfer(measured: dict, reader_k: dict, simd: dict, g109: dict, wall: dict) -> dict[str, Any]:
    return {
        "result_scope": {
            "kernel_path": "isolated discriminator kernels disc_*_tpr64 (tools/qwen38_recon_disc)",
            "launch_geometry": measured.get("launch_primary")
            or "64 threads/row, TG 128, 2 rows/TG (production Qwen3.8 q4 winner)",
            "organs_measured": [
                {"name": o["name"], "rows": o["rows"], "cols": o["cols"]}
                for o in measured["organs"]
            ],
            "activation": "REAL captured BF16 post-norm hidden, token 192 of 256-token holdout",
            "gpu_authority": "MTLCommandBuffer.GPUEndTime-GPUStartTime after wait",
            "reps": 5,
            "paired_alternating_token_wall": False,
        },
        "production_q4_path": {
            "kernel": simd.get("kernel_runtime_genome")
            if isinstance(simd.get("kernel_runtime_genome"), str)
            else gget(wall, "identity/kernel"),
            "vehicle_bpw": simd.get("bpw") or gget(wall, "vehicle/complete_physical_bpw"),
            "covers": True,
            "why": "geo_tpr64_tg128 is the production Qwen3.8 q4 winner. Uniform q4/q3 at this geometry is in-scope.",
        },
        "covers": [
            "Qwen3.8 gate_proj 17408×5120 at tpr64 TG128 2 rows/TG, in-register occupancy tile",
            "Qwen3.8 down_proj 5120×17408 at the same launch class",
            "Families: prod_q4_nibble, uniform_q4/q3/q2, binary_g128, ternary_t0.7, additive_q2q2, rice CSR in-register",
            "Kernel-numeric fidelity of those packed paths (cosine≈1, max_abs~1e-7) against the unpack reference",
        ],
        "does_not_cover": [
            "tg256 (Q80-won 256 threads/row): gate ~26541 ns vs tpr64 ~15125 — penalty is launch geometry, still recon_excess=0 vs byte floor",
            "serial 1-thread-per-row extract (DISPROVEN PATH; the original 5.9× vehicle)",
            "rice serial bitstream (NS-031; cosine not even scored)",
            "hgravs01 two-stage L@(R@x) — the 33rd variant, recon_excess 67852 ns",
            "hadamard as 'same ns as f32': gate 17333 vs 15125; cosine 1.0 with max_abs 0.89 is the scale trap",
            "Q80 routed experts 512×2048 (occupancy-starved vs 17408-row GEMV; Q80 win was occupancy tiles, not this tpr64 result)",
            "Qwen3.8 mixed production reader: HGRAVB01/HGRAVR02 bind q80_*_tg256 / simd_bytes, HGRAVS simd3 — uses_tpr64="
            + str(reader_k.get("uses_tpr64")),
            "attention k/v 1024-row organs (G109: 2.5× worse ps/element, launch-bound; increasing R makes them worse)",
            "lm_head 248320×5120 (geo_tpr64 exists; reconstruction-is-free did not measure it; uint32 addressing overflow is a separate HGRAVU01 story)",
            "teacher quality vs BF16 (except hgravs cosine 0.9317, which is approximation error)",
        ],
        "g109_small_organs": g109.get("small_organs_are_launch_bound"),
        "g109_optimum_organ_dependent": g109.get("the_optimum_IS_organ_dependent"),
        "judgement": (
            "The free-reconstruction result is real and narrow. It covers in-register "
            "occupancy-tile GEMV of the production tpr64 launch class on the two large "
            "Qwen3.8 MLP shapes. It does not transfer to the shipped mixed reader, to "
            "Q80 expert shapes, to two-stage algebra, to serial rice, or to small "
            "attention organs. Reopening a codec is legal only on a named launch "
            "geometry that has been measured for THAT codec and shape."
        ),
        "mixed_reader": reader_k,
    }


def rank_reopened(classes: list[dict]) -> dict[str, Any]:
    reopened = [c for c in classes if c["verdict"] == "REOPENED"]
    # Only rice CSR tpr64 is REOPENED. Rank its organs by hold_mean at measured BPW.
    q = reopened[0]["quality_at_bpw"] if reopened else {}
    ranking = [
        {
            "rank": 1,
            "codec": "rice_q1_rms_2pct in-register CSR @ tpr64 on Qwen3.8 gate_proj",
            "storage_bpw": q.get("storage_bpw") or q.get("gate_hold_mean") and 1.2876,
            "hold_min": q.get("gate_hold_min"),
            "hold_mean": q.get("gate_hold_mean"),
            "vs_null": None if q.get("gate_hold_mean") is None else q["gate_hold_mean"] - NULL_COSINE,
            "kernel_fidelity": "cosine 1.0, max_abs 6e-8, recon_excess 0, median_ns = f32 15125",
            "why_best": (
                "Only codec uniquely priced at 5.9× in the descent catalog whose tpr64 "
                "in-register path is measured free on the exact gate shape. Highest rice "
                "MLP hold (min 0.8768, mean 0.9015, 100% ≥ 0.86). Still not a GO: +0.0035 "
                "over the 0.898 null and no generation."
            ),
        },
        {
            "rank": 2,
            "codec": "rice_q1_rms_2pct on Qwen3.8 attn_in (quality second-reason on 0.99 bar)",
            "storage_bpw": 1.2879,
            "hold_min": None,
            "hold_mean": None,
            "why_not_best": "Attention 0.99 bar fails (HGRAVR02 typical 0.82–0.91, max 0.958).",
        },
        {
            "rank": 3,
            "codec": "rice_q1_rms_2pct on Qwen3.8 up_proj / down_proj",
            "why_not_best": "hold means 0.872 / 0.865 sit BELOW the 0.898 null.",
        },
    ]
    best = {
        "name": "rice_q1_rms_2pct / HGRAVR02 in-register CSR at production tpr64 on Qwen3.8 gate_proj",
        "storage_bpw": 1.287565523035386,
        "active_traffic_over_storage": 1.39,
        "expected_quality": "organ-cosine screen near-null on gate; unknown on doctor-gate gain; unknown on generate",
        "cheapest_decisive_experiment": {
            "what": (
                "Pack a single layer (L0) with rice_q1_rms_2pct on gate_proj only; leave "
                "every other tensor HQ30UQ4. Bind the rice organ through a geo_tpr64_tg128 "
                "CSR kernel (disc_binary_csr_tpr64 class) — NOT q80_binary_group_csr_matvec_tg256, "
                "NOT serial expand, NOT reconstruct-to-Q4. Score vs BF16 on activation-capture-v1 "
                "(real X, fit_n=192, hold_n=64) with doctor-gate axes observed/probed/worst_unit/"
                "GAIN. The 0.01·W probe must FAIL gain and PASS cosine (this tool already shows "
                "that split). Report every cosine against the 0.898 null. If doctor-gate is "
                "healthy relative to same-tensor Q4, run one greedy generate (France/Paris + "
                "reverse-string) with 0 fallbacks, GPU timestamps, 3 paired alternating reps "
                "vs uniform-q4-v1 on the same binary. Kill on doctor-gate fail, null-only "
                "cosine, generate collapse, or silent dense-W. Promote only if generate stays "
                "coherent at 1.29 BPW on that organ with the tpr64 bind."
            ),
            "why_this_is_cheapest": (
                "One organ, one layer, existing capture, no full mixed pack, no Q80 expert "
                "remeasure. Arithmetic re-screen of candidate_table with penalty=1.0 is "
                "cheaper but not decisive (cannot promote or kill quality). mixed-2p0 already "
                "killed the BUNDLE; it does not kill rice-gate-only."
            ),
            "must_not": [
                "synthetic / Gaussian X",
                "cosine-only GO",
                "tg256 Q80 mixed reader",
                "rice serial per-token expand",
                "page-cache-confounded single Metal run offered as a wall",
                "storage BPW quoted without traffic bytes",
            ],
        },
    }
    return {"ranking": ranking, "best": best, "n_reopened": len(reopened)}


def watched_fail(measured: dict, classes: list[dict], probe: dict, confirms: list[dict]) -> list[dict]:
    had = measured["hadamard"]
    return [
        {
            "n": 1,
            "what": "5.9× transferred from Q80 serial extract to Qwen3.8 rice without re-measurement",
            "evidence": (
                "Q80_MIXED_RECONSTRUCTION_WALL slowdown_per_byte_x=5.9 (2.57 vs 15.2 GB/s). "
                "Descent catalog rice recon_penalty=5.9 'Q80 paid 5.9x / byte'. "
                "STALE_BASELINE_PATTERN case 'BPW descent screen rice penalty' marks it "
                "invalidated. QWEN38_RECON_MEASURED: rice CSR tpr64 recon_excess=0."
            ),
        },
        {
            "n": 2,
            "what": "reconstruction-is-free cosine 1.000000 is scale-invariant",
            "evidence": (
                f"hadamard gate cosine=1.0 max_abs={had[0]['max_abs'] if had else 'n/a'}; "
                f"down cosine=1.0 max_abs={had[1]['max_abs'] if len(had)>1 else 'n/a'}. "
                f"This tool's 0.01·W probe: cosine={probe['cosine_Wh_vs_W']} (ACCEPTS), "
                f"gain={probe['gain_min_r_1_over_r']} (REJECTS). Standing law in "
                "gravity_doctor_gate._gain."
            ),
        },
        {
            "n": 3,
            "what": "prose 32/33 cosine and 15124–15541-includes-hadamard do not match the table",
            "evidence": (
                f"counts={measured['counts']}. Nonzero recon_excess is 1/33 (hgravs) — that "
                "32/33 is exact. cosine≈1 is 30/33 (2 rice serial None, 1 hgravs 0.9317). "
                "hadamard gate 17333 ns is outside 15124–15541. recon_excess=0 because it "
                "is vs the byte floor, not vs f32."
            ),
        },
        {
            "n": 4,
            "what": "shipped mixed reader is tg256, not tpr64",
            "evidence": (
                "QWEN38_NATIVE_MIXED_READER what_landed.mlp binds HGRAVB01/HGRAVR02 to "
                "q80_*_tg256 / simd_bytes and HGRAVS to simd3. Reconstruction-is-free "
                "does not cover the production mixed path. tg256 gate ~26541 ns."
            ),
        },
        {
            "n": 5,
            "what": "Q80 mixed recipe on dense Qwen3.8 is natively INCOHERENT",
            "evidence": (
                "mixed-2p0-v1 6/6 prompts collapsed (newlines / ')' / '.'). 0 fallbacks, "
                "0 dense_w, reconstruct_to_q4=false. Q80 mixed-1p5 on experts was COHERENT "
                "at 1.444 BPW. Recipe does not transfer."
            ),
        },
        {
            "n": 6,
            "what": "serial kernels look faster and are the original artifact",
            "evidence": (
                f"gate rice serial {measured['rice_serial']}; gate q4 serial "
                f"{measured['serial_q4_gate_ns']} ns vs tpr64 15125. Notes say DISPROVEN / "
                "ARTIFACT PATH. NS-031 forbids per-token rice serial."
            ),
        },
        {
            "n": 7,
            "what": "0.13 BPW hgravs looks like a result until paired with health and active cost",
            "evidence": (
                "hgravs is the 33rd: recon_excess 67852 ns, cosine 0.9317. Archaeology: "
                "223 tensor-operator rows with local_bpw<0.5, healthy=true: 0. "
                "Q80 storage 0.6462 vs active 2.518 is the category-error cousin."
            ),
        },
        {
            "n": 8,
            "what": "0.8604 organ-cosine bar sits below the 0.898 null",
            "evidence": (
                "NS-013 / NS-016. mixed-1p5 generated coherent text with down_proj "
                "holdout cosine 0.7684. rice gate 100% ≥ 0.86 is not a GO."
            ),
        },
        {
            "n": 9,
            "what": "isolated 5-rep GPU timestamps are not a paired token wall",
            "evidence": (
                "QWEN38_RECON_MEASURED: 5 gpu_ns per variant, isolated organs. "
                "QWEN38_COMPLETE_TOKEN_WALL timing_label DIRTY_ENGINEERING. A single "
                "Metal run is page-cache confounded. This reopen does not offer a TPS."
            ),
        },
        {
            "n": 10,
            "what": "sparse checkout missing a file is not evidence it does not exist",
            "evidence": (
                "receipts/ascent-2026-08-16 is tracked and loaded via git-show when "
                "not materialized. Do not 'discover' that reconstruction-is-free is absent."
            ),
        },
        {
            "n": 11,
            "what": "number confirmations that this tool itself would have gotten wrong",
            "evidence": [
                c for c in confirms if not c["confirmed"]
            ] or "all pinned numbers confirmed against the receipts",
        },
    ]


def build() -> dict[str, Any]:
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sources: dict[str, str] = {}
    loaded: dict[str, dict] = {}
    missing: list[str] = []
    for rel in REQUIRED:
        try:
            obj, src = load_json(rel)
            loaded[rel] = obj
            sources[rel] = src
        except Exception as e:
            missing.append(f"{rel}: {e}")

    probe = scale_invariance_probe()
    if missing:
        return {
            "schema": SCHEMA,
            "generated_at": generated_at,
            "git_head": git_head(),
            "repo": str(REPO),
            "status": "BLOCKED_MISSING_EVIDENCE",
            "missing": missing,
            "scale_invariance_probe": probe,
        }

    free = loaded[f"{ASCENT}/QWEN38_RECONSTRUCTION_IS_FREE.json"]
    measured_raw = loaded[f"{ASCENT}/QWEN38_RECON_MEASURED.json"]
    wall = loaded[f"{ASCENT}/Q80_MIXED_RECONSTRUCTION_WALL.json"]
    won = loaded[f"{ASCENT}/Q80_RECONSTRUCTION_WON.json"]
    ns = loaded[f"{ASCENT}/NEGATIVE_SCIENCE_REGISTER.json"]
    descent = loaded[f"{ASCENT}/QWEN38_BPW_DESCENT.json"]
    review = loaded[f"{ASCENT}/QWEN38_BPW_DESCENT_REVIEW.json"]
    attn = loaded[f"{ASCENT}/QWEN_ATTENTION_DENSITY_VERDICT.json"]
    q80_gen = loaded[f"{ASCENT}/Q80_MIXED_GENERATE.json"]
    reader = loaded[f"{ASCENT}/QWEN38_NATIVE_MIXED_READER.json"]
    mixed_gen = loaded[f"{ASCENT}/QWEN38_NATIVE_MIXED_2P0_GENERATE.json"]
    pq = loaded[f"{ASCENT}/PROMOTION_QUEUE.json"]
    simd = loaded[f"{ASCENT}/QWEN38_TOKEN_NS_DN_VI_SIMD.json"]
    ctw = loaded[f"{ASCENT}/QWEN38_COMPLETE_TOKEN_WALL.json"]
    stale = loaded[f"{ASCENT}/STALE_BASELINE_PATTERN.json"]
    g109 = loaded[f"{ASCENT}/G109_ORGAN_GEOMETRY.json"]

    measured = analyze_measured(measured_raw)
    measured["launch_primary"] = measured_raw.get("launch_primary")
    measured["activation"] = measured_raw.get("activation")
    measured["gpu_time_authority"] = measured_raw.get("gpu_time_authority")
    measured["device_name"] = measured_raw.get("device_name")
    measured["honest_ceiling_gbps"] = measured_raw.get("honest_ceiling_gbps")

    descent_q = descent_quality(descent)
    reader_k = mixed_reader_kernels(reader)
    # native generate file is the prompt table; reader embeds the verdict
    reader_k["native_generate_receipt_prompts"] = len(mixed_gen.get("prompts") or [])
    reader_k["native_generate_fallbacks_total"] = mixed_gen.get("fallbacks_total")
    reader_k["native_generate_dense_w"] = mixed_gen.get("dense_w_materialized_total")

    classes = classify_codecs(
        measured, descent_q, ns, wall, won, attn, q80_gen, reader_k, review
    )
    transfer = geometry_transfer(measured, reader_k, simd, g109, ctw)
    transfer["result_scope"]["launch_geometry"] = measured_raw.get("launch_primary")
    ranking = rank_reopened(classes)

    density = wall.get("DENSITY_IS_COSTING_SPEED") or {}
    won_base = gget(won, "measured/base/gpu_matvec_ns")
    won_ours = gget(won, "measured/ours/gpu_matvec_ns")
    ns006 = next((e for e in ns.get("entries") or [] if e.get("id") == "NS-006"), {})
    ns013 = next((e for e in ns.get("entries") or [] if e.get("id") == "NS-013"), {})

    confirms = [
        confirm("recon_measured.n_variants", measured["n_variants"], 33),
        confirm(
            "recon_excess_zero",
            measured["counts"]["recon_excess_zero"],
            32,
        ),
        confirm(
            "f32_gate_ns",
            measured["f32_control_tpr64_ns"].get("gate"),
            15125,
        ),
        confirm(
            "f32_down_ns",
            measured["f32_control_tpr64_ns"].get("down"),
            7083,
        ),
        confirm(
            "free_receipt.f32_gate",
            gget(free, "research/evidence/f32_control_tpr64_ns/gate"),
            15125,
        ),
        confirm(
            "slowdown_per_byte_x",
            density.get("slowdown_per_byte_x"),
            5.9,
        ),
        confirm("q4_gb_s", density.get("q4_gb_s"), 15.2),
        confirm("mixed_gb_s", density.get("mixed_gb_s"), 2.57),
        confirm("ns006.slowdown", gget(ns006, "what_was_measured/slowdown_per_byte_x"), 5.9),
        confirm("won.base_gpu_matvec_ns", won_base, 867040696),
        confirm("won.ours_gpu_matvec_ns", won_ours, 36598269),
        confirm("q80_mixed_generate.status", q80_gen.get("status"), "COHERENT"),
        confirm(
            "q80_mixed_bpw",
            round(float(gget(q80_gen, "artifact/complete_physical_bpw") or 0), 6),
            1.444446,
            tol=1e-6,
        ),
        confirm("mixed2p0.coherence", reader_k.get("coherence_verdict"), "INCOHERENT"),
        confirm("mixed2p0.fallbacks", mixed_gen.get("fallbacks_total"), 0),
        confirm("scale_probe.cosine_is_1", probe["cosine_is_1"], True),
        confirm("scale_probe.gain_is_0_01", probe["gain_is_0_01"], True),
        confirm(
            "null_cosine_ns013",
            gget(ns013, "what_was_measured/constant_mean_null_baseline"),
            0.898,
        ),
        confirm("rice_catalog_5_9_count", len(descent_q["catalog_with_penalty_5_9"]), 1),
        confirm("hgravs_is_the_33rd", len(measured["recon_excess_nonzero"]), 1),
        confirm(
            "hgravs_name",
            measured["recon_excess_nonzero"][0]["name"] if measured["recon_excess_nonzero"] else None,
            "hgravs01_r160_q3",
        ),
        confirm("reader.uses_tpr64", reader_k.get("uses_tpr64"), False),
        confirm("reader.uses_tg256", reader_k.get("uses_tg256"), True),
    ]

    pq_hits = []
    for e in pq.get("entries") or []:
        lane = e.get("lane") or ""
        if any(
            s in lane
            for s in (
                "qwen38-bpw-descent",
                "q80-matvec-reconstruction",
                "qwen38-descent",
                "geo-tpr64",
                "hgravu01-geo-tpr64",
            )
        ):
            pq_hits.append(
                {
                    "lane": lane,
                    "status": e.get("status"),
                    "disposition": e.get("disposition"),
                    "promoted": e.get("promoted"),
                }
            )

    stale_rice = None
    for inst in stale.get("three_instances_today") or []:
        if "rice" in json.dumps(inst).lower():
            stale_rice = inst

    what_fail = watched_fail(measured, classes, probe, confirms)
    unconfirmed = [c for c in confirms if not c["confirmed"]]

    free_precise = {
        "claim": free.get("claim"),
        "date": free.get("date"),
        "schema": free.get("schema"),
        "what_free_means": (
            "At production tpr64 (64 threads/row, TG 128, 2 rows/TG) on Qwen3.8 MLP "
            "GEMVs, in-register packed kernels take the same GPU ns as uncompressed "
            "f32 (~15125 ns gate, ~7083 ns down). Reconstruction adds 0 ns above the "
            "packed-byte bandwidth floor on 32 of 33 variants. Codec choice at this "
            "geometry is not constrained by reconstruction time. 'Free' is NOT 'free "
            "at every geometry', NOT 'quality vs BF16', and NOT 'the shipped mixed reader'."
        ),
        "measured_how": {
            "activation": measured_raw.get("activation"),
            "gpu_authority": measured_raw.get("gpu_time_authority"),
            "device": measured_raw.get("device_name"),
            "launch": measured_raw.get("launch_primary"),
            "organs": 2,
            "variants": 33,
            "reps_per_variant": 5,
            "honest_ceiling_gbps": measured_raw.get("honest_ceiling_gbps"),
            "not_synthetic": gget(measured_raw, "activation/not_synthetic"),
        },
        "the_32": (
            "Every variant except down/hgravs01_r160_q3 has recon_excess_ns=0. "
            "That includes f32 control, q4/q3/q2 (tpr64 and tg256), binary, ternary, "
            "additive, hadamard, rice CSR, and both serial artifact paths."
        ),
        "the_33rd": measured["recon_excess_nonzero"][0] if measured["recon_excess_nonzero"] else None,
        "prose_vs_table": {
            "receipt_says_cosine_1_on_32_of_33": gget(free, "research/evidence/cosine_1.000000_on"),
            "table_cosine_approx_1": measured["counts"]["cosine_approx_1"],
            "table_cosine_none": measured["counts"]["cosine_none"],
            "table_cosine_not_1": measured["counts"]["cosine_not_1"],
            "receipt_says_ns_band_includes_hadamard": True,
            "hadamard_gate_ns": measured["hadamard"][0]["median_gpu_ns"] if measured["hadamard"] else None,
            "band_claimed": gget(free, "research/evidence/codecs_at_tpr64_ns"),
            "correction": (
                "Trust the per-variant table in QWEN38_RECON_MEASURED.json over the "
                "summary string in QWEN38_RECONSTRUCTION_IS_FREE.json."
            ),
        },
        "how_this_survived": free.get("how_this_survived"),
        "what_this_retires": free.get("what_this_retires"),
        "corroboration_q80": free.get("corroboration"),
        "q80_won_gpu_matvec_ms": {
            "base": won_base / 1e6 if won_base else None,
            "ours": won_ours / 1e6 if won_ours else None,
            "speedup": gget(won, "measured/gpu_matvec_speedup"),
        },
    }

    receipt = {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "git_head": git_head(),
        "repo": str(REPO),
        "obligation": "NNS-011 — tpr64 reconstruction is free (32/33). Codecs killed for the 5.9× penalty are quality-eligible again.",
        "census_kind": "ARTIFACT_OF_METHOD",
        "sources": sources,
        "scale_invariance_probe": probe,
        "null_baseline_cosine": {
            "value": NULL_COSINE,
            "source": f"{ASCENT}/NEGATIVE_SCIENCE_REGISTER.json entries/NS-013",
            "law": "Report every fidelity cosine against this null or it means nothing.",
        },
        "free_reconstruction": free_precise,
        "measured": {
            "n_variants": measured["n_variants"],
            "counts": measured["counts"],
            "f32_control_tpr64_ns": measured["f32_control_tpr64_ns"],
            "launch_primary": measured["launch_primary"],
            "activation": measured["activation"],
            "the_33rd": measured["recon_excess_nonzero"],
            "hadamard": measured["hadamard"],
            "rice_csr_storage_vs_traffic": [
                {
                    "organ": v["organ"],
                    "storage_bpw": v.get("storage_bpw"),
                    "storage_bytes": v.get("storage_bytes"),
                    "traffic_bytes": v.get("traffic_bytes"),
                    "traffic_over_storage": v.get("traffic_over_storage"),
                    "median_gpu_ns": v["median_gpu_ns"],
                    "recon_excess_ns": v["recon_excess_ns"],
                    "cosine": v["cosine"],
                    "max_abs": v["max_abs"],
                    "spread": v["spread"],
                }
                for v in measured["rice_csr"]
            ],
            "gate_tpr64_inregister_within_500ns_of_f32": measured[
                "gate_tpr64_inregister_within_500ns_of_f32"
            ],
            "what_free_means": measured["what_free_means"],
            "variants": [
                {
                    "organ": v["organ"],
                    "name": v["name"],
                    "family": v["family"],
                    "kernel": v["kernel"],
                    "kernel_class": v["kernel_class"],
                    "median_gpu_ns": v["median_gpu_ns"],
                    "recon_excess_ns": v["recon_excess_ns"],
                    "cosine": v["cosine"],
                    "max_abs": v["max_abs"],
                    "spread": v["spread"],
                    "note": v["note"],
                }
                for org in measured["organs"]
                for v in org["variants"]
            ],
        },
        "five_point_nine": {
            "slowdown_per_byte_x": density.get("slowdown_per_byte_x"),
            "q4_gb_s": density.get("q4_gb_s"),
            "mixed_gb_s": density.get("mixed_gb_s"),
            "q4_token_ms": gget(ns006, "what_was_measured/q4_token_ms"),
            "mixed_token_ms": gget(ns006, "what_was_measured/mixed_token_ms"),
            "claim": density.get("claim"),
            "ns006_retry_when": ns006.get("retry_when"),
            "stale_baseline_case": stale_rice,
            "q80_won_correction_0": (won.get("CORRECTIONS_TO_MY_EARLIER_RECORD") or [None])[0],
        },
        "codecs": classes,
        "geometry_transfer": transfer,
        "ranking": ranking,
        "promotion_queue_related": pq_hits,
        "descent_quality": {
            "catalog": descent_q["catalog"],
            "rice_roles": [
                r for r in descent_q["by_role_codec"] if r["codec"] == "rice_q1_rms_2pct"
            ],
            "binary_roles": [
                r for r in descent_q["by_role_codec"] if r["codec"] == "binary_g128"
            ],
            "candidate_rice_containing": descent_q["candidate_rice_containing"],
            "coherence_floor": descent_q["coherence_floor"],
            "activation": descent_q["activation"],
        },
        "confirms": confirms,
        "all_pinned_numbers_confirmed": len(unconfirmed) == 0,
        "unconfirmed": unconfirmed,
        "what_i_watched_fail": what_fail,
        "simd_vehicle": {
            "bpw": simd.get("bpw"),
            "kernel_runtime_genome": simd.get("kernel_runtime_genome"),
            "occupancy_gate": (simd.get("occupancy") or [None])[0],
        },
        "complete_token_wall": {
            "timing_label": ctw.get("timing_label"),
            "timing_label_reason": ctw.get("timing_label_reason"),
            "kernel": gget(ctw, "identity/kernel"),
            "bpw": gget(ctw, "vehicle/complete_physical_bpw"),
        },
    }
    return receipt


def print_report(r: dict) -> None:
    print("NOETIC TPR64 REOPEN — NNS-011")
    print("=" * 72)
    print(f"schema     {r.get('schema')}")
    print(f"generated  {r.get('generated_at')}")
    print(f"head       {r.get('git_head')}")
    print(f"repo       {r.get('repo')}")
    if r.get("status") == "BLOCKED_MISSING_EVIDENCE":
        print("STATUS     BLOCKED_MISSING_EVIDENCE")
        for m in r.get("missing") or []:
            print(f"  missing  {m}")
        return

    print(f"pinned     {'ALL CONFIRMED' if r.get('all_pinned_numbers_confirmed') else 'UNCONFIRMED'}")
    print(f"wrote      {r.get('wrote_to')}")
    print()

    fr = r["free_reconstruction"]
    print("## 1. FREE RECONSTRUCTION (precise)")
    print(fr["what_free_means"])
    print()
    mh = fr["measured_how"]
    print(f"  receipt     {ASCENT}/QWEN38_RECONSTRUCTION_IS_FREE.json")
    print(f"  table       {ASCENT}/QWEN38_RECON_MEASURED.json  (authority for the 33)")
    print(f"  date        {fr['date']}")
    print(f"  launch      {mh.get('launch')}")
    print(f"  device      {mh.get('device')}")
    print(f"  gpu time    {mh.get('gpu_authority')}")
    print(f"  activation  not_synthetic={gget(mh, 'activation/not_synthetic')} "
          f"token_index={gget(mh, 'activation/token_index')} "
          f"sha256_self={gget(mh, 'activation/sha256_self')}")
    print(f"  organs      {mh.get('organs')}  variants={mh.get('variants')}  "
          f"reps={mh.get('reps_per_variant')}")
    print(f"  f32 ns      gate={r['measured']['f32_control_tpr64_ns'].get('gate')}  "
          f"down={r['measured']['f32_control_tpr64_ns'].get('down')}")
    print(f"  32 of 33    recon_excess_ns=0  ({r['measured']['counts']['recon_excess_zero']}/"
          f"{r['measured']['counts']['total']})")
    t33 = fr["the_33rd"] or {}
    print(f"  33rd        {t33.get('organ')}/{t33.get('name')}  "
          f"median_gpu_ns={t33.get('median_gpu_ns')}  "
          f"recon_excess_ns={t33.get('recon_excess_ns')}  "
          f"cosine={t33.get('cosine')}")
    print(f"  the 32      {fr['the_32']}")
    print()
    print("  prose vs table:")
    pv = fr["prose_vs_table"]
    print(f"    receipt cosine_1.000000_on = {pv['receipt_says_cosine_1_on_32_of_33']}")
    print(f"    table   cosine≈1 {pv['table_cosine_approx_1']} / none {pv['table_cosine_none']} / "
          f"not-1 {pv['table_cosine_not_1']}")
    print(f"    hadamard gate ns = {pv['hadamard_gate_ns']}  (claimed band includes hadamard)")
    print(f"    {pv['correction']}")
    print()
    print("  Q80 corroboration (different geometry, same moral): "
          f"gpu_matvec {fr['q80_won_gpu_matvec_ms']['base']:.1f} → "
          f"{fr['q80_won_gpu_matvec_ms']['ours']:.1f} ms "
          f"(speedup {fr['q80_won_gpu_matvec_ms']['speedup']}) without changing a codec.")
    print()

    print("## 2. CODECS KILLED FOR THE 5.9× PENALTY")
    fx = r["five_point_nine"]
    print(f"  5.9× = q4 {fx['q4_gb_s']} GB/s vs mixed {fx['mixed_gb_s']} GB/s "
          f"(token {fx['q4_token_ms']} vs {fx['mixed_token_ms']} ms) on the Q80 mixed vehicle.")
    print(f"  {fx['claim']}")
    print()
    for c in r["codecs"]:
        print(f"  [{c['verdict']:<20}] {c['codec']}")
        for kr in c.get("killing_receipts") or []:
            print(f"      killed by {kr['path']}")
            print(f"               {kr['field']}  number={kr['number']}")
            print(f"               {kr['how']}")
        if c.get("why_reopened"):
            print(f"      reopen   {c['why_reopened']}")
        if c.get("second_reason"):
            print(f"      2nd reason {c['second_reason']}")
        print(f"      geometry covered={c.get('geometry_covered')}  {c.get('geometry_note')}")
        print()

    print("## 3. GEOMETRY / SHAPE TRANSFER")
    gt = r["geometry_transfer"]
    print(f"  launch   {gt['result_scope'].get('launch_geometry')}")
    print(f"  organs   {gt['result_scope'].get('organs_measured')}")
    print(f"  judgement")
    print(f"    {gt['judgement']}")
    print("  COVERS:")
    for x in gt["covers"]:
        print(f"    + {x}")
    print("  DOES NOT COVER:")
    for x in gt["does_not_cover"]:
        print(f"    - {x}")
    print()

    print("## 4. RANKING OF REOPENED + BEST EXPERIMENT")
    rk = r["ranking"]
    for row in rk["ranking"]:
        print(f"  #{row['rank']} {row['codec']}")
        for k, v in row.items():
            if k in ("rank", "codec"):
                continue
            print(f"      {k}: {v}")
    best = rk["best"]
    print()
    print(f"  BEST  {best['name']}")
    print(f"        storage_bpw={best['storage_bpw']}  "
          f"traffic/storage≈{best['active_traffic_over_storage']}")
    print(f"        expected quality: {best['expected_quality']}")
    exp = best["cheapest_decisive_experiment"]
    print("  CHEAPEST DECISIVE EXPERIMENT")
    print(f"        {exp['what']}")
    print(f"        why cheapest: {exp['why_this_is_cheapest']}")
    print(f"        must not: {exp['must_not']}")
    print()

    print("## 5. SCALE / NULL / STORAGE-VS-ACTIVE / HEALTH")
    p = r["scale_invariance_probe"]
    print(f"  0.01·W cosine={p['cosine_Wh_vs_W']} accept={p['cosine_is_1']}  "
          f"gain={p['gain_min_r_1_over_r']} reject_ok={p['gain_is_0_01']}")
    print(f"  {p['verdict']}")
    print(f"  null cosine {r['null_baseline_cosine']['value']}  "
          f"({r['null_baseline_cosine']['law']})")
    for row in r["measured"]["rice_csr_storage_vs_traffic"]:
        print(f"  rice CSR {row['organ']}: storage_bpw={row['storage_bpw']}  "
              f"storage_bytes={row['storage_bytes']}  traffic_bytes={row['traffic_bytes']}  "
              f"traffic/storage={row['traffic_over_storage']}")
    print("  health: recon-measured cosine is kernel unpack, not a doctor-gate verdict. "
          "223 structured components below 0.5 local BPW had healthy=true: 0. "
          "rice at 1.29 BPW still needs a health verdict before it is a result.")
    print()

    print("## WHAT I WATCHED FAIL")
    for w in r["what_i_watched_fail"]:
        print(f"  {w['n']}. {w['what']}")
        ev = w["evidence"]
        if isinstance(ev, list):
            print(f"     {ev}")
        else:
            print(f"     {ev}")
    print()
    print("## CONFIRMS")
    for c in r["confirms"]:
        mark = "OK" if c["confirmed"] else "FAIL"
        print(f"  [{mark}] {c['field']}: observed={c['observed']} expected={c['expected']}")
    print("=" * 72)


def write_receipt(r: dict) -> Path:
    dest = REPO / "receipts" / "headless" / "NOETIC_TPR64_REOPEN.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(r, indent=2, default=str) + "\n")
    return dest


def main() -> int:
    receipt = build()
    try:
        path = write_receipt({k: v for k, v in receipt.items() if k != "wrote_to"})
        receipt["wrote_to"] = str(path)
        # rewrite with wrote_to
        path.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    except OSError as e:
        receipt["wrote_to"] = f"WRITE_FAILED: {e}"
        print_report(receipt)
        print(f"WRITE_FAILED {e}", file=sys.stderr)
        return 2
    print_report(receipt)
    if receipt.get("status") == "BLOCKED_MISSING_EVIDENCE":
        return 2
    if not receipt.get("all_pinned_numbers_confirmed"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
