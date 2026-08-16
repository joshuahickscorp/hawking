#!/usr/bin/env python3
"""Q80 mixed-codec teacher-forced layer-drift probe (coherence-refutation lane).

Closes the cheapest refutation of the <=1.5 mixed-representation plan:

1. Capture-sufficiency audit of the 25258-token source-BF16 capture.
2. Reconstruct gate/up/down for a contiguous layer span with the receipt codecs.
3. Invoke the Rust layer-major BF16 streamer to measure mixed vs null drift
   and span-end logits against the true BF16 residual.

Does not pack a full artifact. Does not write a Metal kernel.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab.operators.doctor6.rungs import quant_binary, quant_residual_compact
from lab.operators.hgravs01_adapter import encode_hgravs01
from lab.operators.q80_capture_index import inspect_index
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map

MAIN_HAWKING = Path("/Users/scammermike/Downloads/hawking")
REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = MAIN_HAWKING / "workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next"
CAPTURE = (
    MAIN_HAWKING
    / "workspace/campaign/records/ascension-sandbox/physical/qwen80"
    / "quality-diagnostics/source-bf16-capture-n192-scale64"
)
GPU_LOCK = MAIN_HAWKING / "tools/gpu_lane_lock.sh"
ROW_CAP = 2048
ROW_SEED = 0xD0C70A
HIDDEN = 2048
INTERMEDIATE = 512
N_EXPERTS = 512
N_LAYERS = 48
F_EXPERT = 0.9703169371044981
F_NONEXPERT = 0.029683062895501933
GATE_BPW_RECEIPT = 1.1269
UP_BPW_RECEIPT = 1.2918
DOWN_BPW_RECEIPT = 1.27
NULL_SEED = 20260816

SCHEMA = "hawking.ascension.qwen80_mixed_codec_coherence_probe.v1"


def f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    rounding_bias = ((bits >> 16) & 1) + 0x7FFF
    return ((bits + rounding_bias) >> 16).astype("<u2")


def complete_bpw(expert_bpw: float, nonexpert_bits: float) -> float:
    return F_EXPERT * float(expert_bpw) + F_NONEXPERT * float(nonexpert_bits)


def _summarize_rows(name: str, dim: int, arr: np.ndarray) -> dict[str, Any]:
    n = int(arr.size)
    under = int((arr < dim).sum())
    zero = int((arr == 0).sum())
    ge_dim = int((arr >= dim).sum())
    return {
        "organ": name,
        "fitted_dimension": int(dim),
        "n_layer_expert_pairs": n,
        "zero_rows": zero,
        "rows_lt_dimension_UNDERDETERMINED": under,
        "frac_underdetermined": float(under / max(n, 1)),
        "rows_ge_dimension_wellposed_for_score": ge_dim,
        "frac_wellposed_for_score": float(ge_dim / max(n, 1)),
        "rows_ge_160_hgravs_rank_unclamped": int((arr >= 160).sum()),
        "rows_ge_64": int((arr >= 64).sum()),
        "min": int(arr.min()) if n else 0,
        "p10": float(np.percentile(arr, 10)) if n else 0.0,
        "p50": float(np.percentile(arr, 50)) if n else 0.0,
        "p90": float(np.percentile(arr, 90)) if n else 0.0,
        "max": int(arr.max()) if n else 0,
        "mean": float(arr.mean()) if n else 0.0,
    }


def audit_capture(capture: Path) -> dict[str, Any]:
    status, root, header = inspect_index(capture)
    if status != "ok" or root is None or header is None:
        raise SystemExit(f"capture index not ok: {status} under {capture}")
    key_layer = np.load(root / "key_layer.npy")
    key_expert = np.load(root / "key_expert.npy")
    key_offsets = np.load(root / "key_offsets.npy")
    rows = np.diff(key_offsets).astype(np.int64)
    grid = np.zeros((N_LAYERS, N_EXPERTS), dtype=np.int64)
    for layer, expert, n in zip(key_layer, key_expert, rows):
        grid[int(layer), int(expert)] = int(n)
    capped = np.minimum(grid.reshape(-1), ROW_CAP)
    gate = _summarize_rows("gate_proj", HIDDEN, capped)
    up = _summarize_rows("up_proj", HIDDEN, capped)
    down = _summarize_rows("down_proj", INTERMEDIATE, capped)
    receipt_pairs = {
        "gate_up_screen_organs": [
            {"layer": 10, "expert": 453, "rows": int(grid[10, 453]), "dim": HIDDEN},
            {"layer": 3, "expert": 494, "rows": int(grid[3, 494]), "dim": HIDDEN},
        ],
        "down_screen_organs": [
            {"layer": 1, "expert": 265, "rows": int(grid[1, 265]), "dim": INTERMEDIATE},
            {"layer": 32, "expert": 179, "rows": int(grid[32, 179]), "dim": INTERMEDIATE},
            {"layer": 46, "expert": 428, "rows": int(grid[46, 428]), "dim": INTERMEDIATE},
            {"layer": 35, "expert": 330, "rows": int(grid[35, 330]), "dim": INTERMEDIATE},
        ],
    }
    for row in receipt_pairs["gate_up_screen_organs"]:
        row["underdetermined"] = int(row["rows"]) < int(row["dim"])
    for row in receipt_pairs["down_screen_organs"]:
        row["underdetermined"] = int(row["rows"]) < int(row["dim"])
    return {
        "index_status": status,
        "index_dir": str(root),
        "n_tokens": header.get("n_tokens"),
        "max_hidden_tokens_per_expert_policy": (header.get("bounded_storage") or {}).get(
            "max_hidden_tokens_per_expert"
        ),
        "row_cap_applied": ROW_CAP,
        "row_sample_seed": ROW_SEED,
        "gate_proj": gate,
        "up_proj": up,
        "down_proj": down,
        "receipt_screen_organs": receipt_pairs,
        "verdict": {
            "gate_up_typical_organ_UNDERDETERMINED": gate["frac_underdetermined"] > 0.5,
            "down_typical_organ_UNDERDETERMINED": down["frac_underdetermined"] > 0.5,
            "receipt_down_proj_cosine_cannot_be_trusted": all(
                r["underdetermined"] for r in receipt_pairs["down_screen_organs"]
            ),
            "receipt_gate_up_cosine_is_busy_organ_only": not any(
                r["underdetermined"] for r in receipt_pairs["gate_up_screen_organs"]
            ),
        },
        "note": (
            "gate/up X is router-input 2048-dim. down_proj X is post-SwiGLU 512-dim "
            "recomputed from the same retained router-input rows "
            "(silu(X @ W_gate.T) * (X @ W_up.T); verified in "
            "receipts/QWEN80_SWIGLU_INTERMEDIATE_VERIFY.json). Same row counts apply."
        ),
    }


def _silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


_WORKER_MODEL: Path | None = None
_WORKER_WMAP: dict[str, str] | None = None


def _init_recon_worker(model_dir: str) -> None:
    global _WORKER_MODEL, _WORKER_WMAP
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    _WORKER_MODEL = Path(model_dir)
    _WORKER_WMAP = load_weight_map(_WORKER_MODEL)


def _recon_one_expert(payload: dict[str, Any]) -> dict[str, Any]:
    layer = int(payload["layer"])
    expert = int(payload["expert"])
    model_dir = _WORKER_MODEL or Path(payload["model_dir"])
    wmap = _WORKER_WMAP or payload["wmap"]
    x_path = payload.get("x_path")
    mixed_dir = Path(payload["mixed_dir"])
    null_dir = Path(payload["null_dir"])
    capture_identity = payload["capture_identity"]

    w_gate = np.asarray(
        load_tensor(
            model_dir, wmap, f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
        ),
        dtype=np.float32,
    )
    w_up = np.asarray(
        load_tensor(
            model_dir, wmap, f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"
        ),
        dtype=np.float32,
    )
    w_down = np.asarray(
        load_tensor(
            model_dir, wmap, f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
        ),
        dtype=np.float32,
    )
    gate_hat, gate_bytes = quant_binary(w_gate)
    up_hat, up_bytes = quant_residual_compact(
        w_up,
        outlier_ratio=0.02,
        index_mode="rice",
        value_bits=1,
        value_scale="rms",
    )
    n_fit = 0
    rank_achieved = 0
    rank_clamped = False
    down_cold = True
    down_bytes = 0
    if x_path and Path(x_path).is_file():
        x_hidden = np.load(x_path)
        if x_hidden.ndim == 2 and x_hidden.shape[0] > 0 and x_hidden.shape[1] == HIDDEN:
            x_sw = _silu(x_hidden @ w_gate.T) * (x_hidden @ w_up.T)
            encoded = encode_hgravs01(
                w_down,
                x_sw.astype(np.float32, copy=False),
                rank=160,
                bits=3,
                capture_identity=capture_identity,
            )
            down_hat = np.asarray(encoded["W_hat"], dtype=np.float32)
            n_fit = int(encoded["n_fit_rows"])
            rank_achieved = int(encoded["achieved_rank"])
            rank_clamped = bool(encoded["rank_clamped_to_n_fit"])
            down_bytes = int(encoded["payload_bytes"])
            down_cold = False
        else:
            down_hat = w_down
    else:
        down_hat = w_down

    rng = np.random.default_rng(NULL_SEED + layer * 10_000 + expert)
    def _null(src: np.ndarray, hat: np.ndarray) -> np.ndarray:
        err = (hat - src).reshape(-1).copy()
        rng.shuffle(err)
        return (src.reshape(-1) + err).reshape(src.shape).astype(np.float32)

    prefix_m = mixed_dir / f"L{layer:02d}"
    prefix_n = null_dir / f"L{layer:02d}"
    for role, hat, src in (
        ("gate", gate_hat, w_gate),
        ("up", up_hat, w_up),
        ("down", down_hat, w_down),
    ):
        mm = np.memmap(
            prefix_m / f"{role}.bf16",
            dtype="<u2",
            mode="r+",
            shape=(N_EXPERTS, hat.size),
        )
        mm[expert] = f32_to_bf16_u16(np.ascontiguousarray(hat).reshape(-1))
        mm.flush()
        nn = np.memmap(
            prefix_n / f"{role}.bf16",
            dtype="<u2",
            mode="r+",
            shape=(N_EXPERTS, src.size),
        )
        nn[expert] = f32_to_bf16_u16(np.ascontiguousarray(_null(src, hat)).reshape(-1))
        nn.flush()

    return {
        "layer": layer,
        "expert": expert,
        "n_fit_rows": n_fit,
        "down_cold_left_bf16": down_cold,
        "hgravs_rank_achieved": rank_achieved,
        "hgravs_rank_clamped": rank_clamped,
        "gate_payload_bytes": int(gate_bytes),
        "up_payload_bytes": int(up_bytes),
        "down_payload_bytes": int(down_bytes),
        "gate_bpw": 8.0 * gate_bytes / max(w_gate.size, 1),
        "up_bpw": 8.0 * up_bytes / max(w_up.size, 1),
        "down_bpw": 8.0 * down_bytes / max(w_down.size, 1) if down_bytes else None,
    }


def _prepare_layer_files(dest: Path, layer: int) -> None:
    prefix = dest / f"L{layer:02d}"
    prefix.mkdir(parents=True, exist_ok=True)
    role_elems = INTERMEDIATE * HIDDEN
    nbytes = N_EXPERTS * role_elems * 2
    for role in ("gate", "up", "down"):
        path = prefix / f"{role}.bf16"
        if path.exists() and path.stat().st_size == nbytes:
            continue
        with path.open("wb") as handle:
            handle.truncate(nbytes)


def reconstruct_span(
    *,
    model_dir: Path,
    capture: Path,
    mixed_dir: Path,
    null_dir: Path,
    span_start: int,
    span_end: int,
    workers: int,
) -> dict[str, Any]:
    from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
        collect_expert_activations,
    )

    wmap = load_weight_map(model_dir)
    src = capture / "capture-result.json"
    st = src.stat()
    capture_identity = {
        "path": str(src),
        "capture_result_path": str(src),
        "sha256": "17a1e9b60a53cc491601a549880c2d215ff16395ee36abaa05fb95eb7fe2aabe",
        "schema": "hawking.doctor6.hgravs01.capture_binding.v1",
        "status": "Q80_COHERENCE_PROBE_FIT",
        "fit_kind": "real_routed_activation_capture",
        "not_synthetic_unit_direction": True,
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    mixed_dir.mkdir(parents=True, exist_ok=True)
    null_dir.mkdir(parents=True, exist_ok=True)
    x_scratch = mixed_dir / "_x_hidden"
    x_scratch.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    organ_rows: list[dict[str, Any]] = []
    for layer in range(span_start, span_end):
        _prepare_layer_files(mixed_dir, layer)
        _prepare_layer_files(null_dir, layer)
        wanted = {(layer, e) for e in range(N_EXPERTS)}
        print(f"[recon] collecting router-input X for L{layer}", flush=True)
        by_le, _prov = collect_expert_activations(
            capture,
            wanted_keys=wanted,
            max_rows_per_expert=ROW_CAP,
            row_sample_seed=ROW_SEED,
            use_index=True,
            x_kind="router_input",
        )
        jobs = []
        for expert in range(N_EXPERTS):
            x_path = None
            x = by_le.get((layer, expert))
            if x is not None and np.asarray(x).size:
                xp = x_scratch / f"L{layer:02d}_E{expert:03d}.npy"
                np.save(xp, np.asarray(x, dtype=np.float32))
                x_path = str(xp)
            jobs.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "model_dir": str(model_dir),
                    "wmap": None,
                    "x_path": x_path,
                    "mixed_dir": str(mixed_dir),
                    "null_dir": str(null_dir),
                    "capture_identity": capture_identity,
                }
            )
        print(f"[recon] encoding L{layer} ({len(jobs)} experts, workers={workers})", flush=True)
        if workers <= 1:
            _init_recon_worker(str(model_dir))
            for job in jobs:
                organ_rows.append(_recon_one_expert(job))
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_recon_worker,
                initargs=(str(model_dir),),
            ) as pool:
                futs = [pool.submit(_recon_one_expert, job) for job in jobs]
                for i, fut in enumerate(as_completed(futs), 1):
                    organ_rows.append(fut.result())
                    if i % 64 == 0 or i == len(futs):
                        print(f"[recon] L{layer} {i}/{len(futs)}", flush=True)
        for expert in range(N_EXPERTS):
            xp = x_scratch / f"L{layer:02d}_E{expert:03d}.npy"
            if xp.exists():
                xp.unlink()

    cold = [r for r in organ_rows if r["down_cold_left_bf16"]]
    clamped = [r for r in organ_rows if r["hgravs_rank_clamped"]]
    billed_down = [r for r in organ_rows if r["down_bpw"] is not None]
    mixed_expert_bpw = (
        GATE_BPW_RECEIPT + UP_BPW_RECEIPT + DOWN_BPW_RECEIPT
    ) / 3.0
    if billed_down:
        measured = (
            float(np.mean([r["gate_bpw"] for r in organ_rows]))
            + float(np.mean([r["up_bpw"] for r in organ_rows]))
            + float(np.mean([r["down_bpw"] for r in billed_down]))
        ) / 3.0
        mixed_expert_bpw = measured
    meta = {
        "span_start": span_start,
        "span_end": span_end,
        "n_organs": len(organ_rows),
        "down_cold_left_bf16": len(cold),
        "hgravs_rank_clamped": len(clamped),
        "mean_gate_bpw": float(np.mean([r["gate_bpw"] for r in organ_rows])),
        "mean_up_bpw": float(np.mean([r["up_bpw"] for r in organ_rows])),
        "mean_down_bpw_fitted": (
            float(np.mean([r["down_bpw"] for r in billed_down])) if billed_down else None
        ),
        "mixed_expert_bpw_measured": float(mixed_expert_bpw),
        "complete_bpw_8bit_nonexpert": complete_bpw(mixed_expert_bpw, 8.0),
        "complete_bpw_6bit_nonexpert": complete_bpw(mixed_expert_bpw, 6.0),
        "complete_bpw_4bit_nonexpert": complete_bpw(mixed_expert_bpw, 4.0),
        "wall_secs": time.perf_counter() - started,
        "organs": organ_rows,
    }
    (mixed_dir / "recon-meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _growth_ratios(rel_l2: list[float]) -> list[float]:
    out = []
    for i in range(1, len(rel_l2)):
        prev = max(rel_l2[i - 1], 1e-12)
        out.append(rel_l2[i] / prev)
    return out


def analyze_probe(probe: dict[str, Any], audit: dict[str, Any], recon: dict[str, Any]) -> dict[str, Any]:
    span_start = int(probe["span_start"])
    span_end = int(probe["span_end"])
    span_layers = [row for row in probe["layers"] if row.get("in_span")]
    mixed_rel = [float(row["mixed"]["last_token_rel_l2"]) for row in span_layers]
    null_rel = [float(row["null"]["last_token_rel_l2"]) for row in span_layers]
    mixed_cos = [float(row["mixed"]["last_token_cosine"]) for row in span_layers]
    null_cos = [float(row["null"]["last_token_cosine"]) for row in span_layers]
    mixed_g = _growth_ratios(mixed_rel)
    null_g = _growth_ratios(null_rel)
    geo_mixed = float(math.exp(sum(math.log(max(g, 1e-12)) for g in mixed_g) / max(len(mixed_g), 1))) if mixed_g else float("nan")
    geo_null = float(math.exp(sum(math.log(max(g, 1e-12)) for g in null_g) / max(len(null_g), 1))) if null_g else float("nan")
    residual_true = [float(row["true_residual_growth"]) for row in probe["layers"]]

    n_after = N_LAYERS - span_end
    last_rel = mixed_rel[-1] if mixed_rel else float("nan")
    if mixed_g and math.isfinite(geo_mixed):
        extra_rel_48 = last_rel * (geo_mixed ** n_after)
        # cosine proxy under a small-angle model is not claimed; report rel-L2 only
    else:
        extra_rel_48 = float("nan")

    logits = probe.get("logits") or {}
    mixed_sep = None
    if mixed_rel and null_rel:
        mixed_sep = float(mixed_rel[-1] / max(null_rel[-1], 1e-12))

    multiplicative = bool(mixed_g and geo_mixed > 1.0 + 1e-6)
    separated_from_null = bool(
        mixed_rel
        and null_rel
        and mixed_rel[-1] + 1e-6 < null_rel[-1] * 0.85
    )
    logit_alive = bool(logits.get("mixed_top1_agree")) or float(logits.get("mixed_top5_overlap") or 0.0) >= 0.4

    if multiplicative and extra_rel_48 > 2.0 and not logit_alive:
        verdict = "NO_GO"
        reason = (
            f"error multiplies at geo-mean {geo_mixed:.4f}x/layer; "
            f"extrapolated rel-L2 at 48 is {extra_rel_48:.3f}"
        )
        fix = None
    elif multiplicative and extra_rel_48 > 1.0:
        verdict = "GO_WITH_FIX"
        reason = (
            f"sub-span already multiplicative (geo {geo_mixed:.4f}x/layer); "
            f"48-layer rel-L2 extrapolates to {extra_rel_48:.3f}"
        )
        fix = {
            "change": (
                "protect first 4 and last 4 layers at 8-bit routed experts, "
                "keep mixed codecs on the middle 40; spend the 8->6-bit non-expert "
                "headroom if the protected mass still exceeds 1.5"
            ),
            "protected_layer_fraction": 8.0 / 48.0,
            "identity_check": {
                "complete_bpw_if_middle_stays_mixed_and_nonexpert_6bit": complete_bpw(
                    recon.get("mixed_expert_bpw_measured") or 1.22957, 6.0
                ),
            },
        }
    elif not separated_from_null:
        verdict = "NO_GO"
        reason = (
            "mixed drift is not cleanly separated from the matched-magnitude "
            f"shuffled null (span-end rel-L2 mixed={mixed_rel[-1] if mixed_rel else None} "
            f"null={null_rel[-1] if null_rel else None}); metric is not certifying the plan"
        )
        fix = None
    else:
        verdict = "GO"
        reason = (
            f"span-end mixed rel-L2 {last_rel:.4f} with geo growth {geo_mixed:.4f} "
            f"(<=1); 48-layer extra {extra_rel_48:.4f}; separated from null"
        )
        fix = None

    return {
        "span": [span_start, span_end],
        "mixed_last_token_cosine": mixed_cos,
        "mixed_last_token_rel_l2": mixed_rel,
        "null_last_token_cosine": null_cos,
        "null_last_token_rel_l2": null_rel,
        "mixed_growth_ratios": mixed_g,
        "null_growth_ratios": null_g,
        "mixed_geo_growth_per_layer": geo_mixed,
        "null_geo_growth_per_layer": geo_null,
        "true_residual_stream_growth": residual_true,
        "true_residual_geo_growth": (
            float(math.exp(sum(math.log(max(g, 1e-12)) for g in residual_true[1:]) / max(len(residual_true) - 1, 1)))
            if len(residual_true) > 1
            else None
        ),
        "extrapolated_rel_l2_at_48_from_span_end": extra_rel_48,
        "layers_after_span": n_after,
        "arithmetic": (
            f"relL2_48 = relL2_span_end * geo^{n_after} = "
            f"{last_rel} * {geo_mixed}^{n_after} = {extra_rel_48}"
        ),
        "mixed_vs_null_span_end_rel_l2_ratio": mixed_sep,
        "separated_from_null": separated_from_null,
        "logits": logits,
        "verdict": verdict,
        "reason": reason,
        "fix": fix,
        "capture_sufficiency": audit,
        "reconstruction": {
            k: recon.get(k)
            for k in (
                "n_organs",
                "down_cold_left_bf16",
                "hgravs_rank_clamped",
                "mean_gate_bpw",
                "mean_up_bpw",
                "mean_down_bpw_fitted",
                "mixed_expert_bpw_measured",
                "complete_bpw_8bit_nonexpert",
                "complete_bpw_6bit_nonexpert",
                "complete_bpw_4bit_nonexpert",
                "wall_secs",
            )
        },
    }


def run_rust_probe(args: argparse.Namespace) -> dict[str, Any]:
    example = (
        REPO
        / "workspace/ops/build/rust/release-fast/examples"
        / "ascension_qwen80_mixed_codec_layer_drift_probe"
    )
    if not example.is_file():
        raise SystemExit(f"missing built example at {example}; compile first")
    cmd = [
        str(GPU_LOCK),
        "q80-coherence-probe",
        str(example),
        "--source-model-dir",
        str(args.model_dir),
        "--mixed-override-dir",
        str(args.mixed_dir),
        "--null-override-dir",
        str(args.null_dir),
        "--output-dir",
        str(args.probe_dir),
        "--prompt",
        args.prompt,
        "--span-start",
        str(args.span_start),
        "--span-end",
        str(args.span_end),
    ]
    if args.tokenizer:
        cmd.extend(["--tokenizer-path", str(args.tokenizer)])
    if args.no_continue_remaining:
        cmd.append("--no-continue-remaining")
    print("[probe] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return json.loads((Path(args.probe_dir) / "probe-result.json").read_text())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--capture", type=Path, default=CAPTURE)
    p.add_argument("--mixed-dir", type=Path, default=Path("/tmp/q80-coherence-probe-overrides/mixed"))
    p.add_argument("--null-dir", type=Path, default=Path("/tmp/q80-coherence-probe-overrides/null"))
    p.add_argument("--probe-dir", type=Path, default=Path("/tmp/q80-coherence-probe-overrides/run"))
    p.add_argument("--receipt", type=Path, default=None)
    p.add_argument("--prompt", default="Write a function that reverses a string.")
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--span-start", type=int, default=0)
    p.add_argument("--span-end", type=int, default=4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--reconstruct-only", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--skip-reconstruct", action="store_true")
    p.add_argument("--no-continue-remaining", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_capture(args.capture)
    print(json.dumps({"audit": {k: audit[k] for k in audit if k != "receipt_screen_organs"}}, indent=2), flush=True)
    if args.audit_only:
        return 0

    recon: dict[str, Any]
    if args.skip_reconstruct or args.analyze_only:
        recon = json.loads((args.mixed_dir / "recon-meta.json").read_text())
    else:
        recon = reconstruct_span(
            model_dir=args.model_dir,
            capture=args.capture,
            mixed_dir=args.mixed_dir,
            null_dir=args.null_dir,
            span_start=args.span_start,
            span_end=args.span_end,
            workers=args.workers,
        )
        print(
            json.dumps({k: recon[k] for k in recon if k != "organs"}, indent=2),
            flush=True,
        )
    if args.reconstruct_only:
        return 0

    if args.analyze_only:
        probe = json.loads((Path(args.probe_dir) / "probe-result.json").read_text())
    else:
        probe = run_rust_probe(args)
    analysis = analyze_probe(probe, audit, recon)
    receipt = {
        "schema": SCHEMA,
        "status": analysis["verdict"],
        "lane": "q80-coherence-probe",
        "timing_label": "DIRTY_ENGINEERING",
        "prompt": args.prompt,
        "analysis": analysis,
        "probe_wall_secs": probe.get("wall_secs"),
        "probe_peak_rss_gib": probe.get("peak_rss_gib"),
        "claim_boundary": {
            "artifact_packed": False,
            "decode_kernel_exists": False,
            "coherence_generation_tested": False,
            "teacher_forced_span_probe": True,
            "full_autoregressive_generation_not_run": True,
        },
    }
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"[receipt] {args.receipt}", flush=True)
    print(json.dumps({"verdict": analysis["verdict"], "reason": analysis["reason"], "geo": analysis["mixed_geo_growth_per_layer"], "extra_48": analysis["extrapolated_rel_l2_at_48_from_span_end"], "logits": analysis["logits"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
