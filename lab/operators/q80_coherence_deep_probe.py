#!/usr/bin/env python3
"""Q80 mixed-codec full-depth coherence probe (coherence-deep lane).

Closes the three defects of the 4-layer GO_WITH_FIX probe:

1. Teacher-forced drift across all 48 layers (no geo^44 extrapolation).
2. Honest rank policy: never present a row-clamped HGRAVS01 fit as r160.
   Binding constraint is capture rows (q80-capture-coverage census).
3. Matched-magnitude null *distribution* (several seeds) so separation is
   a margin, not one sample.

Then runs mixed-hat greedy generation on the reverse-string prompt. Coherent
text is the only thing that settles the <=1.5 path.

Does not pack a full artifact. Does not write a Metal kernel.
Does not raise STREAMED_PEAK_RSS_HARD_CAP_BYTES.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab.operators.doctor6.rungs import quant_binary, quant_residual_compact
from lab.operators.hgravs01_adapter import encode_hgravs01
from lab.operators.q80_coherence_layer_drift_probe import (
    F_EXPERT,
    F_NONEXPERT,
    GATE_BPW_RECEIPT,
    HIDDEN,
    INTERMEDIATE,
    MODEL_DIR,
    N_EXPERTS,
    N_LAYERS,
    UP_BPW_RECEIPT,
    audit_capture,
    complete_bpw,
    f32_to_bf16_u16,
)

MAIN_HAWKING = Path("/Users/scammermike/Downloads/hawking")
REPO = Path(__file__).resolve().parents[2]
CAPTURE = (
    MAIN_HAWKING
    / "workspace/campaign/records/ascension-sandbox/physical/qwen80"
    / "quality-diagnostics/source-bf16-capture-n192-scale64"
)
GPU_LOCK = MAIN_HAWKING / "tools/gpu_lane_lock.sh"
# Use every retained row. The 4-layer probe capped at 2048; that did not cause
# the r160 clamps (those organs had <160 rows) but it did throw away samples
# on the well-posed tail.
ROW_CAP: int | None = None
ROW_SEED = 0xD0C70A
HGRAVS_RANK = 160
HGRAVS_BITS = 3
MIN_ROWS_FOR_HGRAVS = 1
LANE = "q80-coherence-deep"
SCHEMA = "hawking.ascension.qwen80_mixed_codec_coherence_deep.v1"
REQUIRED_REVERSE_STRING_IDS = [
    8420, 594, 264, 4285, 729, 304, 13027, 429, 17431, 288, 264, 914,
]
DEFAULT_NULL_SEEDS = [20260816, 20260817, 20260818, 20260819, 20260820, 20260821, 20260822]


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
    from lab.operators.qwen30b_gravity_pack import load_weight_map

    _WORKER_MODEL = Path(model_dir)
    _WORKER_WMAP = load_weight_map(_WORKER_MODEL)


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


def _recon_one_expert(payload: dict[str, Any]) -> dict[str, Any]:
    from lab.operators.qwen30b_gravity_pack import load_tensor

    layer = int(payload["layer"])
    expert = int(payload["expert"])
    model_dir = _WORKER_MODEL or Path(payload["model_dir"])
    wmap = _WORKER_WMAP or payload["wmap"]
    x_path = payload.get("x_path")
    mixed_dir = Path(payload["mixed_dir"])
    capture_identity = payload["capture_identity"]
    requested_rank = int(payload["requested_rank"])
    requested_bits = int(payload["requested_bits"])

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
    n_fit_full = int(payload.get("n_fit_full") or 0)
    rank_requested = requested_rank
    rank_achieved = 0
    rank_clamped = False
    down_cold = True
    down_policy = "source_bf16_cold"
    down_bytes = 0
    down_hat = w_down
    if x_path and Path(x_path).is_file():
        x_hidden = np.load(x_path)
        if x_hidden.ndim == 2 and x_hidden.shape[0] > 0 and x_hidden.shape[1] == HIDDEN:
            n_fit = int(x_hidden.shape[0])
            if n_fit < MIN_ROWS_FOR_HGRAVS:
                down_policy = "source_bf16_lt_min_rows"
            else:
                x_sw = _silu(x_hidden @ w_gate.T) * (x_hidden @ w_up.T)
                encoded = encode_hgravs01(
                    w_down,
                    x_sw.astype(np.float32, copy=False),
                    rank=requested_rank,
                    bits=requested_bits,
                    capture_identity=capture_identity,
                )
                down_hat = np.asarray(encoded["W_hat"], dtype=np.float32)
                n_fit = int(encoded["n_fit_rows"])
                rank_achieved = int(encoded["achieved_rank"])
                rank_clamped = bool(encoded["rank_clamped_to_n_fit"])
                down_bytes = int(encoded["payload_bytes"])
                down_cold = False
                if rank_clamped:
                    down_policy = "hgravs01_rank_clamped_to_n_fit"
                else:
                    down_policy = f"hgravs01_r{rank_achieved}_b{requested_bits}"

    prefix = mixed_dir / f"L{layer:02d}"
    for role, hat in (("gate", gate_hat), ("up", up_hat), ("down", down_hat)):
        mm = np.memmap(
            prefix / f"{role}.bf16",
            dtype="<u2",
            mode="r+",
            shape=(N_EXPERTS, hat.size),
        )
        mm[expert] = f32_to_bf16_u16(np.ascontiguousarray(hat).reshape(-1))
        mm.flush()

    return {
        "layer": layer,
        "expert": expert,
        "n_fit_rows": n_fit,
        "n_fit_full_before_any_cap": n_fit_full,
        "gate_dim": HIDDEN,
        "up_dim": HIDDEN,
        "down_dim": INTERMEDIATE,
        "rows_vs_gate_dim": n_fit / HIDDEN,
        "rows_vs_down_dim": n_fit / INTERMEDIATE,
        "rows_vs_requested_rank": n_fit / max(rank_requested, 1),
        "down_cold_left_bf16": down_cold,
        "down_policy": down_policy,
        "hgravs_rank_requested": rank_requested,
        "hgravs_rank_achieved": rank_achieved,
        "hgravs_rank_clamped": rank_clamped,
        "gate_payload_bytes": int(gate_bytes),
        "up_payload_bytes": int(up_bytes),
        "down_payload_bytes": int(down_bytes),
        "gate_bpw": 8.0 * gate_bytes / max(w_gate.size, 1),
        "up_bpw": 8.0 * up_bytes / max(w_up.size, 1),
        "down_bpw": (
            8.0 * down_bytes / max(w_down.size, 1)
            if down_bytes
            else (16.0 if down_cold else None)
        ),
    }


def reconstruct_layer(
    *,
    model_dir: Path,
    capture: Path,
    mixed_dir: Path,
    layer: int,
    workers: int,
    capture_identity: dict[str, Any],
    organs_jsonl: Path,
) -> list[dict[str, Any]]:
    from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
        collect_expert_activations,
    )

    _prepare_layer_files(mixed_dir, layer)
    ready = mixed_dir / f"L{layer:02d}" / "READY"
    done = mixed_dir / f"L{layer:02d}" / "DONE"
    if ready.exists():
        ready.unlink()
    if done.exists():
        done.unlink()

    x_scratch = mixed_dir / "_x_hidden"
    x_scratch.mkdir(parents=True, exist_ok=True)
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
        n_fit_full = 0
        x = by_le.get((layer, expert))
        if x is not None and np.asarray(x).size:
            arr = np.asarray(x, dtype=np.float32)
            n_fit_full = int(arr.shape[0])
            xp = x_scratch / f"L{layer:02d}_E{expert:03d}.npy"
            np.save(xp, arr)
            x_path = str(xp)
        jobs.append(
            {
                "layer": layer,
                "expert": expert,
                "model_dir": str(model_dir),
                "wmap": None,
                "x_path": x_path,
                "n_fit_full": n_fit_full,
                "mixed_dir": str(mixed_dir),
                "capture_identity": capture_identity,
                "requested_rank": HGRAVS_RANK,
                "requested_bits": HGRAVS_BITS,
            }
        )
    print(f"[recon] encoding L{layer} ({len(jobs)} experts, workers={workers})", flush=True)
    organ_rows: list[dict[str, Any]] = []
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
    with organs_jsonl.open("a") as handle:
        for row in organ_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    # READY last, after data is flushed.
    (mixed_dir / f"L{layer:02d}" / "READY").write_text("ok\n")
    return organ_rows


def summarize_organs(organ_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(organ_rows)
    clamped = [r for r in organ_rows if r["hgravs_rank_clamped"]]
    cold = [r for r in organ_rows if r["down_cold_left_bf16"]]
    billed_down = [r for r in organ_rows if r["down_bpw"] is not None]
    rows = np.array([r["n_fit_rows"] for r in organ_rows], dtype=np.float64)
    mixed_expert_bpw = (GATE_BPW_RECEIPT + UP_BPW_RECEIPT + 1.27) / 3.0
    if billed_down:
        mixed_expert_bpw = (
            float(np.mean([r["gate_bpw"] for r in organ_rows]))
            + float(np.mean([r["up_bpw"] for r in organ_rows]))
            + float(np.mean([r["down_bpw"] for r in billed_down]))
        ) / 3.0
    return {
        "n_organs": n,
        "down_cold_left_bf16": len(cold),
        "hgravs_rank_clamped": len(clamped),
        "frac_rank_clamped": float(len(clamped) / max(n, 1)),
        "frac_cold": float(len(cold) / max(n, 1)),
        "rows_min": int(rows.min()) if n else 0,
        "rows_p10": float(np.percentile(rows, 10)) if n else 0.0,
        "rows_p50": float(np.percentile(rows, 50)) if n else 0.0,
        "rows_p90": float(np.percentile(rows, 90)) if n else 0.0,
        "rows_max": int(rows.max()) if n else 0,
        "rows_mean": float(rows.mean()) if n else 0.0,
        "organs_rows_lt_160": int((rows < 160).sum()) if n else 0,
        "organs_rows_lt_512": int((rows < 512).sum()) if n else 0,
        "organs_rows_lt_2048": int((rows < 2048).sum()) if n else 0,
        "mean_gate_bpw": float(np.mean([r["gate_bpw"] for r in organ_rows])) if n else None,
        "mean_up_bpw": float(np.mean([r["up_bpw"] for r in organ_rows])) if n else None,
        "mean_down_bpw": (
            float(np.mean([r["down_bpw"] for r in billed_down])) if billed_down else None
        ),
        "mixed_expert_bpw_measured": float(mixed_expert_bpw),
        "complete_bpw_8bit_nonexpert": complete_bpw(mixed_expert_bpw, 8.0),
        "complete_bpw_6bit_nonexpert": complete_bpw(mixed_expert_bpw, 6.0),
        "complete_bpw_4bit_nonexpert": complete_bpw(mixed_expert_bpw, 4.0),
        "identity": "complete_bpw = 0.97032*expert_bpw + 0.02968*nonexpert_bpw",
        "identity_check_8bit": abs(
            complete_bpw(mixed_expert_bpw, 8.0)
            - (F_EXPERT * mixed_expert_bpw + F_NONEXPERT * 8.0)
        ),
    }


def _growth_ratios(rel_l2: list[float]) -> list[float]:
    out = []
    for i in range(1, len(rel_l2)):
        prev = max(rel_l2[i - 1], 1e-12)
        out.append(rel_l2[i] / prev)
    return out


def _geo(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(math.exp(sum(math.log(max(g, 1e-12)) for g in values) / len(values)))


def analyze_drift(probe: dict[str, Any], recon: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    body = probe.get("result") or probe
    layers = body.get("layers") or []
    span_layers = [row for row in layers if row.get("in_span") and isinstance(row.get("mixed"), dict)]
    mixed_rel = [float(row["mixed"]["last_token_rel_l2"]) for row in span_layers]
    mixed_cos = [float(row["mixed"]["last_token_cosine"]) for row in span_layers]
    mixed_g = _growth_ratios(mixed_rel)
    residual_true = [float(row.get("true_residual_growth") or 1.0) for row in layers]

    null_by_layer: list[list[float]] = []
    null_seeds: list[int] = []
    for row in span_layers:
        vals = []
        for item in row.get("nulls") or []:
            metrics = item.get("metrics") or {}
            if "last_token_rel_l2" in metrics:
                vals.append(float(metrics["last_token_rel_l2"]))
            if not null_seeds and "seed" in item:
                pass
            if "seed" in item and int(item["seed"]) not in null_seeds:
                null_seeds.append(int(item["seed"]))
        null_by_layer.append(vals)

    null_mean = [float(np.mean(v)) if v else float("nan") for v in null_by_layer]
    null_std = [float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for v in null_by_layer]
    null_min = [float(np.min(v)) if v else float("nan") for v in null_by_layer]
    null_max = [float(np.max(v)) if v else float("nan") for v in null_by_layer]
    z_scores = []
    for m, mu, sd in zip(mixed_rel, null_mean, null_std):
        if sd and math.isfinite(sd) and sd > 1e-12:
            z_scores.append(float((m - mu) / sd))
        else:
            z_scores.append(float("nan"))

    def _span_geo(end: int) -> float:
        g = _growth_ratios(mixed_rel[:end])
        return _geo(g)

    windows = {}
    for end in (4, 8, 12, 24, 36, 48):
        if len(mixed_rel) >= end:
            windows[f"geo_growth_layers_0_{end}"] = _span_geo(end)
            windows[f"rel_l2_at_{end - 1}"] = mixed_rel[end - 1]

    # Separation: mixed must sit outside the null envelope with a margin.
    # Require last-layer mixed > null_max * 1.25 OR z > 3, else the metric
    # cannot certify the representation (same class of failure as cosine 0.898).
    last_mixed = mixed_rel[-1] if mixed_rel else float("nan")
    last_null_max = null_max[-1] if null_max else float("nan")
    last_null_mean = null_mean[-1] if null_mean else float("nan")
    last_z = z_scores[-1] if z_scores else float("nan")
    separated = bool(
        math.isfinite(last_mixed)
        and math.isfinite(last_null_max)
        and (last_mixed > last_null_max * 1.25 or (math.isfinite(last_z) and abs(last_z) >= 3.0))
    )
    # Also refuse "mixed quieter than null" as a GO from this metric alone.
    mixed_below_null = bool(
        math.isfinite(last_mixed)
        and math.isfinite(last_null_mean)
        and last_mixed + 1e-6 < last_null_mean
    )

    # Do NOT use extrapolation as the verdict. Report it only as a diagnostic
    # of what the 4-layer probe would have claimed from the same prefix.
    prefix4_geo = _span_geo(4) if len(mixed_rel) >= 4 else float("nan")
    diagnostic_extra = (
        mixed_rel[3] * (prefix4_geo ** 44) if len(mixed_rel) >= 4 and math.isfinite(prefix4_geo) else float("nan")
    )

    logits = (body.get("logits") or {})
    measured_48 = last_mixed
    growth_not_constant = bool(
        len(mixed_g) >= 8
        and abs(_span_geo(4) - _geo(mixed_g)) > 0.05
    )

    return {
        "n_measured_layers": len(mixed_rel),
        "mixed_last_token_rel_l2": mixed_rel,
        "mixed_last_token_cosine": mixed_cos,
        "mixed_growth_ratios": mixed_g,
        "mixed_geo_growth_full_depth": _geo(mixed_g),
        "windows": windows,
        "true_residual_stream_growth": residual_true,
        "true_residual_geo_growth": _geo(residual_true[1:]) if len(residual_true) > 1 else None,
        "null_seeds": null_seeds or probe.get("null_seeds"),
        "null_last_token_rel_l2_mean": null_mean,
        "null_last_token_rel_l2_std": null_std,
        "null_last_token_rel_l2_min": null_min,
        "null_last_token_rel_l2_max": null_max,
        "mixed_z_vs_null": z_scores,
        "span_end_mixed_rel_l2": last_mixed,
        "span_end_null_mean": last_null_mean,
        "span_end_null_max": last_null_max,
        "span_end_z": last_z,
        "separated_from_null": separated,
        "mixed_below_null_mean": mixed_below_null,
        "growth_not_constant_vs_4layer_prefix": growth_not_constant,
        "four_layer_prefix_geo": prefix4_geo,
        "diagnostic_only_4layer_extrapolation_at_48": diagnostic_extra,
        "diagnostic_note": (
            "The 4-layer probe's 16211x number is a prefix extrapolation. "
            "This receipt's verdict uses the measured 48-layer curve, not that product."
        ),
        "logits": logits,
        "reconstruction": recon,
        "capture_sufficiency": {
            k: audit[k]
            for k in audit
            if k != "receipt_screen_organs"
        },
    }


def decide_verdict(
    analysis: dict[str, Any],
    generation: dict[str, Any] | None,
) -> dict[str, Any]:
    gen_ids = list((generation or {}).get("generated_token_ids") or [])
    required = REQUIRED_REVERSE_STRING_IDS
    prefix = min(len(gen_ids), len(required))
    token_match = prefix > 0 and gen_ids[:prefix] == required[:prefix]
    full_match = gen_ids[: len(required)] == required if len(gen_ids) >= len(required) else False
    text = str((generation or {}).get("generated_text") or "")
    looks_code = any(
        tok in text for tok in ("def ", "return ", "[::-1]", "reversed(", "for ")
    )
    looks_gibberish = bool(text) and not looks_code and len(set(text)) < 8

    measured = analysis.get("span_end_mixed_rel_l2")
    geo = analysis.get("mixed_geo_growth_full_depth")
    separated = bool(analysis.get("separated_from_null"))
    n_layers = int(analysis.get("n_measured_layers") or 0)

    if generation is not None and full_match:
        return {
            "verdict": "GO",
            "reason": (
                "greedy generation on the reverse-string prompt reproduced the "
                f"required {len(required)}-token BF16 sequence; representation is live"
            ),
            "generation_gate": "required_token_ids_match",
        }
    if generation is not None and token_match and prefix >= 8 and looks_code:
        return {
            "verdict": "GO",
            "reason": (
                f"greedy generation matched the first {prefix} required tokens "
                "and continued as code; representation is live"
            ),
            "generation_gate": "prefix_match_and_code",
        }
    if generation is not None and looks_gibberish:
        return {
            "verdict": "NO_GO",
            "reason": (
                "autoregressive generation is not coherent text "
                f"(ids={gen_ids!r} text={text!r})"
            ),
            "generation_gate": "incoherent",
        }
    if generation is not None and looks_code and not full_match:
        return {
            "verdict": "GO_WITH_FIX",
            "reason": (
                "generation produced code-shaped text but not the required token "
                f"sequence (ids={gen_ids!r}). Drift curve still required for the fix."
            ),
            "generation_gate": "code_but_not_required_ids",
        }

    # Instrumentation-only path (generation not run or inconclusive).
    if n_layers < 48:
        return {
            "verdict": "NO_GO" if (measured or 0) > 2.0 and (geo or 1.0) > 1.05 else "PARTIAL",
            "reason": f"only {n_layers}/48 layers measured; generation not conclusive",
            "generation_gate": "not_run_or_short",
        }
    if not separated:
        return {
            "verdict": "NO_GO" if (measured or 0) > 1.5 and (geo or 1.0) > 1.08 else "INCONCLUSIVE_METRIC",
            "reason": (
                "full-depth mixed drift is not separated from the matched-magnitude "
                f"null distribution (mixed={measured} null_mean={analysis.get('span_end_null_mean')} "
                f"null_max={analysis.get('span_end_null_max')} z={analysis.get('span_end_z')}). "
                "This metric cannot certify the representation. Generation is the remaining gate."
            ),
            "generation_gate": "not_run_or_short",
        }
    if (geo or 1.0) > 1.08 and (measured or 0) > 1.0:
        return {
            "verdict": "NO_GO",
            "reason": (
                f"measured 48-layer rel-L2={measured:.4f} with geo growth {geo:.4f}/layer "
                "and null-separated; error compounds through depth"
            ),
            "generation_gate": "not_run_or_short",
        }
    if (measured or 0) < 0.5 and (geo or 1.0) <= 1.02:
        return {
            "verdict": "GO",
            "reason": (
                f"full-depth mixed rel-L2={measured:.4f}, geo={geo:.4f}, null-separated; "
                "generation still preferred as the public gate"
            ),
            "generation_gate": "not_run_or_short",
        }
    return {
        "verdict": "GO_WITH_FIX",
        "reason": (
            f"full-depth mixed rel-L2={measured} geo={geo}; separated={separated}. "
            "A named protection (early/late layers or clamped organs) is required "
            "before treating the <=1.5 path as live."
        ),
        "generation_gate": "not_run_or_short",
    }


def rust_bin() -> Path:
    return (
        REPO
        / "workspace/ops/build/rust/release-fast/examples"
        / "ascension_qwen80_mixed_codec_coherence_deep"
    )


def run_locked(cmd: list[str]) -> None:
    wrapped = [str(GPU_LOCK), LANE, *cmd]
    print("[probe] " + " ".join(wrapped), flush=True)
    subprocess.run(wrapped, check=True)


def wait_done(path: Path, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while not path.is_file():
        if time.time() > deadline:
            raise TimeoutError(f"timeout waiting for {path}")
        time.sleep(0.1)


def capture_identity_for(capture: Path) -> dict[str, Any]:
    src = capture / "capture-result.json"
    st = src.stat()
    return {
        "path": str(src),
        "capture_result_path": str(src),
        "sha256": "17a1e9b60a53cc491601a549880c2d215ff16395ee36abaa05fb95eb7fe2aabe",
        "schema": "hawking.doctor6.hgravs01.capture_binding.v1",
        "status": "Q80_COHERENCE_DEEP_FIT",
        "fit_kind": "real_routed_activation_capture",
        "not_synthetic_unit_direction": True,
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "row_cap": ROW_CAP,
        "note": "sha256 is the capture-result binding recorded by the 4-layer probe; not rehashed (1.38 GiB wall)",
    }


def free_gib(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--capture", type=Path, default=CAPTURE)
    p.add_argument(
        "--work-dir",
        type=Path,
        default=REPO / "workspace/ops/q80-coherence-deep",
    )
    p.add_argument(
        "--receipt",
        type=Path,
        default=REPO / "receipts/ascent-2026-08-16/Q80_COHERENCE_DEEP.json",
    )
    p.add_argument("--prompt", default="Write a function that reverses a string.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--span-start", type=int, default=0)
    p.add_argument("--span-end", type=int, default=48)
    p.add_argument("--generate-tokens", type=int, default=16)
    p.add_argument("--skip-drift", action="store_true")
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-reconstruct", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--keep-mixed", action="store_true")
    p.add_argument("--null-seeds", default=",".join(str(s) for s in DEFAULT_NULL_SEEDS))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    work = args.work_dir
    mixed_dir = work / "mixed"
    drift_dir = work / "drift"
    gen_dir = work / "generate"
    organs_jsonl = work / "organs.jsonl"
    work.mkdir(parents=True, exist_ok=True)
    mixed_dir.mkdir(parents=True, exist_ok=True)
    drift_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    audit = audit_capture(args.capture)
    print(json.dumps({"audit_frac_under_gate": audit["gate_proj"]["frac_underdetermined"]}, indent=2), flush=True)
    if args.audit_only:
        return 0

    identity = capture_identity_for(args.capture)
    all_organs: list[dict[str, Any]] = []
    recon_meta: dict[str, Any] = {}
    probe: dict[str, Any] = {}
    generation: dict[str, Any] | None = None

    example = rust_bin()
    if not example.is_file() and not args.skip_drift:
        raise SystemExit(f"missing built example at {example}; compile first")

    if not args.skip_reconstruct and organs_jsonl.exists():
        organs_jsonl.unlink()

    if not args.skip_drift:
        cmd = [
            str(example),
            "--source-model-dir",
            str(args.model_dir),
            "--mixed-override-dir",
            str(mixed_dir),
            "--output-dir",
            str(drift_dir),
            "--mode",
            "drift",
            "--prompt",
            args.prompt,
            "--span-start",
            str(args.span_start),
            "--span-end",
            str(args.span_end),
            "--null-seeds",
            args.null_seeds,
            "--wait-for-ready",
            "--write-done",
            "--ready-timeout-secs",
            "2400",
        ]
        print("[drift] launching rust under gpu_lane_lock", flush=True)
        proc = subprocess.Popen(
            [str(GPU_LOCK), LANE, *cmd],
        )
        recon_t0 = time.perf_counter()
        try:
            if args.skip_reconstruct:
                for layer in range(args.span_start, args.span_end):
                    ready = mixed_dir / f"L{layer:02d}" / "READY"
                    if not ready.is_file():
                        raise SystemExit(f"--skip-reconstruct but {ready} missing")
            else:
                def _recon(layer: int) -> list[dict[str, Any]]:
                    if proc.poll() is not None:
                        raise SystemExit(f"rust drift exited early rc={proc.returncode}")
                    return reconstruct_layer(
                        model_dir=args.model_dir,
                        capture=args.capture,
                        mixed_dir=mixed_dir,
                        layer=layer,
                        workers=args.workers,
                        capture_identity=identity,
                        organs_jsonl=organs_jsonl,
                    )

                def _delete_hats(layer: int) -> None:
                    if args.keep_mixed:
                        return
                    for role in ("gate", "up", "down"):
                        p = mixed_dir / f"L{layer:02d}" / f"{role}.bf16"
                        if p.exists():
                            p.unlink()
                    print(
                        f"[recon] L{layer} consumed+deleted hats; free={free_gib(work):.1f} GiB",
                        flush=True,
                    )

                # Pipeline: reconstruct L+1 while rust consumes L.
                with ThreadPoolExecutor(max_workers=1) as pipeline:
                    fut = pipeline.submit(_recon, args.span_start)
                    for layer in range(args.span_start, args.span_end):
                        rows = fut.result()
                        all_organs.extend(rows)
                        nxt = layer + 1
                        if nxt < args.span_end:
                            fut = pipeline.submit(_recon, nxt)
                        wait_done(mixed_dir / f"L{layer:02d}" / "DONE", timeout_s=5400)
                        _delete_hats(layer)
            rc = proc.wait()
            if rc != 0:
                raise SystemExit(f"rust drift failed rc={rc}")
        except Exception:
            if proc.poll() is None:
                proc.terminate()
            raise
        recon_meta = summarize_organs(all_organs)
        recon_meta["wall_secs"] = time.perf_counter() - recon_t0
        recon_meta["row_cap"] = ROW_CAP
        recon_meta["hgravs_requested_rank"] = HGRAVS_RANK
        recon_meta["binding_constraint"] = (
            "capture rows: q80-capture-coverage reports 7536/24576 organs have "
            "<160 retained rows on the existing 25258-token capture. No extended "
            "capture was available in that lane's receipt. Rank cannot be unclamped "
            "without more rows; clamped organs are flagged and billed at achieved rank."
        )
        (work / "recon-meta.json").write_text(json.dumps(recon_meta, indent=2) + "\n")
        probe = json.loads((drift_dir / "probe-result.json").read_text())
    else:
        probe = json.loads((drift_dir / "probe-result.json").read_text())
        if organs_jsonl.exists():
            all_organs = [json.loads(line) for line in organs_jsonl.read_text().splitlines() if line]
        recon_meta = json.loads((work / "recon-meta.json").read_text()) if (work / "recon-meta.json").exists() else summarize_organs(all_organs)

    analysis = analyze_drift(probe, recon_meta, audit)
    (work / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(
        json.dumps(
            {
                "phase": "drift_complete",
                "n_layers": analysis.get("n_measured_layers"),
                "span_end_rel_l2": analysis.get("span_end_mixed_rel_l2"),
                "geo_full": analysis.get("mixed_geo_growth_full_depth"),
                "windows": analysis.get("windows"),
                "separated": analysis.get("separated_from_null"),
                "z": analysis.get("span_end_z"),
            },
            indent=2,
        ),
        flush=True,
    )

    if not args.skip_generate:
        need_gib = 150.0
        have = free_gib(work)
        print(f"[generate] free={have:.1f} GiB need~{need_gib:.0f} to keep 48 mixed layers", flush=True)
        if have < 40:
            print("[generate] BLOCKED: disk too tight to rebuild mixed hats", flush=True)
            generation = {
                "status": "SKIPPED_DISK",
                "free_gib": have,
            }
        else:
            try:
                # Rebuild mixed hats for all 48 layers and keep them for generate.
                if organs_jsonl.exists() and not args.skip_drift:
                    gen_organs = work / "organs-generate.jsonl"
                else:
                    gen_organs = organs_jsonl
                if not args.skip_reconstruct:
                    if gen_organs.exists() and gen_organs != organs_jsonl:
                        gen_organs.unlink()
                    for layer in range(N_LAYERS):
                        reconstruct_layer(
                            model_dir=args.model_dir,
                            capture=args.capture,
                            mixed_dir=mixed_dir,
                            layer=layer,
                            workers=args.workers,
                            capture_identity=identity,
                            organs_jsonl=gen_organs,
                        )
                cmd = [
                    str(example),
                    "--source-model-dir",
                    str(args.model_dir),
                    "--mixed-override-dir",
                    str(mixed_dir),
                    "--output-dir",
                    str(gen_dir),
                    "--mode",
                    "generate",
                    "--prompt",
                    args.prompt,
                    "--generate-tokens",
                    str(args.generate_tokens),
                    "--no-nulls",
                ]
                run_locked(cmd)
                generation = json.loads((gen_dir / "probe-result.json").read_text())
                generation = generation.get("result") or generation
                if not args.keep_mixed:
                    for layer in range(N_LAYERS):
                        prefix = mixed_dir / f"L{layer:02d}"
                        for name in ("gate.bf16", "up.bf16", "down.bf16", "READY", "DONE"):
                            p = prefix / name
                            if p.exists():
                                p.unlink()
            except Exception as exc:  # noqa: BLE001 — generate must not eat the drift receipt
                print(f"[generate] FAILED: {exc!r}", flush=True)
                generation = {"status": "FAILED", "error": repr(exc)}

    decision = decide_verdict(analysis, generation)
    receipt = {
        "schema": SCHEMA,
        "lane": LANE,
        "status": decision["verdict"],
        "timing_label": "DIRTY_ENGINEERING",
        "prompt": args.prompt,
        "required_reverse_string_ids": REQUIRED_REVERSE_STRING_IDS,
        "decision": decision,
        "analysis": analysis,
        "generation": None
        if generation is None
        else {
            "generated_token_ids": generation.get("generated_token_ids"),
            "generated_text": generation.get("generated_text"),
            "status": generation.get("status"),
            "free_gib": generation.get("free_gib"),
        },
        "probe_wall_secs": probe.get("wall_secs"),
        "probe_peak_rss_gib": probe.get("peak_rss_gib"),
        "claim_boundary": {
            "artifact_packed": False,
            "decode_kernel_exists": False,
            "coherence_generation_tested": generation is not None
            and generation.get("generated_token_ids") is not None,
            "teacher_forced_full_depth": int(analysis.get("n_measured_layers") or 0) >= 48,
            "used_bf16_hats_of_mixed_codecs_on_source_streamer": True,
            "not_packed_runtime": True,
            "rss_cap_raised": False,
            "existing_gates_weakened": False,
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[receipt] {args.receipt}", flush=True)
    print(
        json.dumps(
            {
                "verdict": decision["verdict"],
                "reason": decision["reason"],
                "n_layers": analysis.get("n_measured_layers"),
                "span_end_rel_l2": analysis.get("span_end_mixed_rel_l2"),
                "geo_full": analysis.get("mixed_geo_growth_full_depth"),
                "geo_0_4": (analysis.get("windows") or {}).get("geo_growth_layers_0_4"),
                "separated": analysis.get("separated_from_null"),
                "generated": None if generation is None else generation.get("generated_token_ids"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
