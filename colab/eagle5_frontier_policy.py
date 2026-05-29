#!/usr/bin/env python3
"""Frontier policy search for Qwen-3B Eagle5 heads.

This script turns a trained head into runtime policy candidates:

* fixed-K Eagle baseline
* variable-K confidence gating from the trained calibration head
* entropy/margin routed Eagle, skipping speculation on chaotic positions
* tiny draft-lattice oracle coverage for branch widths 2/3/4
* residual-delta simulator readiness probe

It does not claim the laptop runtime already implements every policy. It writes
the compact JSON needed to choose what to wire next, and to rank Colab-trained
heads by projected throughput under bounded Colab Pro compute.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eagle5_tau_eval_pytorch import _load_eval_windows, _load_head
from eagle5_train_pytorch import N_HEADS, RMS_EPS, _rms_norm


def _prefix_len(mask: np.ndarray, depth: int) -> np.ndarray:
    """Consecutive true count per row, capped at depth."""
    out = np.zeros(mask.shape[0], dtype=np.int32)
    alive = np.ones(mask.shape[0], dtype=bool)
    for d in range(depth):
        alive &= mask[:, d]
        out += alive.astype(np.int32)
    return out


def _project(
    accepted: np.ndarray,
    *,
    base_tps: float,
    w4a8_multiplier: float,
    efficiency: float,
) -> dict:
    accepted_mean = float(np.mean(accepted))
    tokens_per_verify = 1.0 + accepted_mean
    projected = base_tps * tokens_per_verify * w4a8_multiplier * efficiency
    return {
        "accepted_draft_tokens_per_verify": accepted_mean,
        "tokens_per_verify": tokens_per_verify,
        "projected_dec_tps": projected,
        "base_tps": base_tps,
        "w4a8_multiplier": w4a8_multiplier,
        "efficiency": efficiency,
        "formula": "base_tps * (1 + accepted_draft_tokens) * w4a8_multiplier * efficiency",
    }


@torch.inference_mode()
def collect_signals(args) -> dict[str, np.ndarray | dict[int, np.ndarray] | int]:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[frontier] WARN: CUDA unavailable; falling back to CPU", flush=True)
        device = "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    head = _load_head(
        args.ckpt,
        args.frozen,
        device,
        num_blocks=args.num_blocks,
        n_heads=args.head_heads,
        ff_mult=args.head_ff_mult,
    )
    head.eval()
    lm_head_f = head._lm_head.float()
    windows = _load_eval_windows(
        args.corpus,
        args.max_depth,
        args.max_windows,
        args.max_row_tokens,
        args.seed,
    )
    if not windows:
        raise RuntimeError("no usable policy windows loaded")

    widths = sorted(set(args.lattice_widths))
    max_width = max(widths) if widths else 1
    max_width = max(1, max_width)

    match_parts = []
    conf_parts = []
    draft_margin_parts = []
    base_margin_parts = []
    resid_cos_parts = []
    lattice_parts = {w: [] for w in widths}

    W = len(windows)
    for start in range(0, W, args.eval_batch_size):
        batch = windows[start : start + args.eval_batch_size]
        prev = torch.from_numpy(np.stack([w["prev"] for w in batch]).astype(np.int64)).to(device)
        residual = torch.from_numpy(
            np.stack([w["residual"] for w in batch]).astype(np.float32)
        ).to(device)
        inter = torch.from_numpy(
            np.stack([w["intermediate"] for w in batch]).astype(np.float32)
        ).to(device)
        B = prev.shape[0]
        cur_prev = prev[:, :1]

        match_d = []
        conf_d = []
        draft_margin_d = []
        base_margin_d = []
        resid_cos_d = []
        lattice_d = {w: [] for w in widths}

        for d in range(args.max_depth):
            residual_d = residual[:, d : d + 1, :]
            inter_d = inter[:, d : d + 1, :]
            token_logits, _sparsity, draft_h, calib_logit = head(cur_prev, residual_d, inter_d)
            draft_logits = token_logits[:, 0, :].float()
            draft_vals, draft_idx = torch.topk(draft_logits, k=max(max_width, 2), dim=-1)
            draft_margin = draft_vals[:, 0] - draft_vals[:, 1]

            baseline = _rms_norm(residual_d, head._output_norm, RMS_EPS).reshape(B, head.hidden_dim)
            target_logits = torch.matmul(baseline.float(), lm_head_f)
            base_vals, base_idx = torch.topk(target_logits, k=2, dim=-1)
            target_arg = base_idx[:, 0]
            base_margin = base_vals[:, 0] - base_vals[:, 1]

            match = draft_idx[:, 0] == target_arg
            conf = torch.sigmoid(calib_logit[:, 0].float())
            match_d.append(match.detach().cpu().numpy())
            conf_d.append(conf.detach().cpu().numpy())
            draft_margin_d.append(draft_margin.detach().cpu().numpy())
            base_margin_d.append(base_margin.detach().cpu().numpy())
            for w in widths:
                contains = (draft_idx[:, :w] == target_arg[:, None]).any(dim=-1)
                lattice_d[w].append(contains.detach().cpu().numpy())

            if d + 1 < args.max_depth:
                pred = _rms_norm(draft_h[:, 0, :], head._output_norm, RMS_EPS).float()
                nxt = _rms_norm(residual[:, d + 1, :], head._output_norm, RMS_EPS).float()
                cos = F.cosine_similarity(pred, nxt, dim=-1)
                resid_cos_d.append(cos.detach().cpu().numpy())

            cur_prev = draft_idx[:, :1]
            del token_logits, draft_logits, target_logits

        match_parts.append(np.stack(match_d, axis=1))
        conf_parts.append(np.stack(conf_d, axis=1))
        draft_margin_parts.append(np.stack(draft_margin_d, axis=1))
        base_margin_parts.append(np.stack(base_margin_d, axis=1))
        if resid_cos_d:
            resid_cos_parts.append(np.stack(resid_cos_d, axis=1))
        for w in widths:
            lattice_parts[w].append(np.stack(lattice_d[w], axis=1))

        if device == "cuda":
            torch.cuda.empty_cache()

    return {
        "windows": W,
        "match": np.concatenate(match_parts, axis=0),
        "confidence": np.concatenate(conf_parts, axis=0),
        "draft_margin": np.concatenate(draft_margin_parts, axis=0),
        "base_margin": np.concatenate(base_margin_parts, axis=0),
        "residual_cosine": (
            np.concatenate(resid_cos_parts, axis=0)
            if resid_cos_parts
            else np.zeros((W, 0), dtype=np.float32)
        ),
        "lattice": {
            w: np.concatenate(parts, axis=0)
            for w, parts in lattice_parts.items()
        },
    }


def search_policies(args, signals: dict) -> dict:
    match = signals["match"]
    confidence = signals["confidence"]
    base_margin = signals["base_margin"]
    lattice = signals["lattice"]
    depth_cap = match.shape[1]
    depths = [d for d in args.depths if 1 <= d <= depth_cap]
    if not depths:
        depths = [depth_cap]

    fixed = []
    for depth in depths:
        accepted = _prefix_len(match[:, :depth], depth)
        fixed.append({
            "kind": "fixed_k",
            "max_depth": depth,
            **_project(
                accepted,
                base_tps=args.base_tps,
                w4a8_multiplier=args.w4a8_multiplier,
                efficiency=args.spec_efficiency,
            ),
            "full_accept_rate": float(np.mean(accepted == depth)),
        })

    variable = []
    for depth in depths:
        for conf_th in args.conf_thresholds:
            active = confidence[:, :depth] >= conf_th
            planned = _prefix_len(active, depth)
            accepted = _prefix_len(active & match[:, :depth], depth)
            variable.append({
                "kind": "variable_k_confidence",
                "max_depth": depth,
                "confidence_threshold": float(conf_th),
                "attempt_rate": float(np.mean(planned > 0)),
                "planned_draft_tokens_mean": float(np.mean(planned)),
                **_project(
                    accepted,
                    base_tps=args.base_tps,
                    w4a8_multiplier=args.w4a8_multiplier,
                    efficiency=args.spec_efficiency * 0.98,
                ),
            })

    routed = []
    for depth in depths:
        for conf_th in args.conf_thresholds:
            active = confidence[:, :depth] >= conf_th
            for margin_th in args.margin_thresholds:
                route = base_margin[:, 0] >= margin_th
                accepted = _prefix_len(active & match[:, :depth], depth)
                accepted = np.where(route, accepted, 0)
                routed.append({
                    "kind": "entropy_margin_routed_variable_k",
                    "max_depth": depth,
                    "confidence_threshold": float(conf_th),
                    "base_margin_threshold": float(margin_th),
                    "route_rate": float(np.mean(route)),
                    **_project(
                        accepted,
                        base_tps=args.base_tps,
                        w4a8_multiplier=args.w4a8_multiplier,
                        efficiency=args.spec_efficiency * 0.99,
                    ),
                })

    lattice_results = []
    for width, contains in lattice.items():
        branch_eff = args.spec_efficiency * max(
            args.min_lattice_efficiency,
            1.0 - args.lattice_branch_penalty * float(width - 1),
        )
        for depth in depths:
            accepted = _prefix_len(contains[:, :depth], depth)
            lattice_results.append({
                "kind": "draft_lattice_oracle",
                "branch_width": int(width),
                "max_depth": depth,
                "note": "Upper bound: target token is somewhere in the tiny draft lattice, not necessarily selected by current runtime.",
                **_project(
                    accepted,
                    base_tps=args.base_tps,
                    w4a8_multiplier=args.w4a8_multiplier,
                    efficiency=branch_eff,
                ),
                "full_path_coverage": float(np.mean(accepted == depth)),
            })

    key = lambda r: r["projected_dec_tps"]
    resid = signals["residual_cosine"]
    residual_probe = {
        "kind": "residual_delta_simulator_probe",
        "cosine_mean": float(np.mean(resid)) if resid.size else None,
        "cosine_p10": float(np.percentile(resid, 10)) if resid.size else None,
        "cosine_p50": float(np.percentile(resid, 50)) if resid.size else None,
        "note": "Higher means draft_hidden is already close to the next residual state. Residual-delta-loss variants should improve this.",
    }

    deployable = fixed + variable + routed
    all_policies = deployable + lattice_results
    return {
        "fixed_k": sorted(fixed, key=key, reverse=True),
        "variable_k": sorted(variable, key=key, reverse=True),
        "entropy_routed": sorted(routed, key=key, reverse=True),
        "draft_lattice": sorted(lattice_results, key=key, reverse=True),
        "residual_delta_probe": residual_probe,
        "best_deployable": max(deployable, key=key),
        "best_overall": max(all_policies, key=key),
    }


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--frozen", required=True, type=Path)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--depths", type=_parse_csv_ints, default=[4, 6, 8, 12])
    p.add_argument("--lattice-widths", type=_parse_csv_ints, default=[2, 3, 4])
    p.add_argument("--conf-thresholds", type=_parse_csv_floats,
                   default=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                            0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    p.add_argument("--margin-thresholds", type=_parse_csv_floats,
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    p.add_argument("--max-windows", type=int, default=6000)
    p.add_argument("--max-row-tokens", type=int, default=192)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=444)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--num-blocks", type=int, default=1)
    p.add_argument("--head-heads", type=int, default=N_HEADS)
    p.add_argument("--head-ff-mult", type=float, default=4.0)
    p.add_argument("--base-tps", type=float, default=26.6)
    p.add_argument("--w4a8-multiplier", type=float, default=1.25)
    p.add_argument("--spec-efficiency", type=float, default=0.85)
    p.add_argument("--lattice-branch-penalty", type=float, default=0.08)
    p.add_argument("--min-lattice-efficiency", type=float, default=0.62)
    args = p.parse_args()

    if args.max_depth <= 0:
        raise SystemExit("--max-depth must be positive")
    args.depths = [d for d in args.depths if d <= args.max_depth]
    signals = collect_signals(args)
    policies = search_policies(args, signals)
    payload = {
        "schema": "eagle5-frontier-policy-v1",
        "ckpt": str(args.ckpt),
        "frozen": str(args.frozen),
        "corpus": str(args.corpus),
        "windows": int(signals["windows"]),
        "max_depth": args.max_depth,
        "search": {
            "depths": args.depths,
            "lattice_widths": args.lattice_widths,
            "conf_thresholds": args.conf_thresholds,
            "margin_thresholds": args.margin_thresholds,
        },
        "policies": {
            "fixed_k_top5": policies["fixed_k"][:5],
            "variable_k_top5": policies["variable_k"][:5],
            "entropy_routed_top5": policies["entropy_routed"][:5],
            "draft_lattice_top5": policies["draft_lattice"][:5],
            "residual_delta_probe": policies["residual_delta_probe"],
            "best_deployable": policies["best_deployable"],
            "best_overall": policies["best_overall"],
        },
        "runtime_hints": {
            "variable_k": {
                "env": {
                    "DISMANTLE_EAGLE5_VARIABLE_K": "1",
                    "DISMANTLE_EAGLE5_MAX_K": str(policies["variable_k"][0]["max_depth"]),
                    "DISMANTLE_EAGLE5_CONF_THRESH": f"{policies['variable_k'][0]['confidence_threshold']:.3f}",
                }
            },
            "entropy_routing": {
                "env": {
                    "DISMANTLE_EAGLE5_ENTROPY_ROUTE": "1",
                    "DISMANTLE_EAGLE5_MARGIN_THRESH": f"{policies['entropy_routed'][0]['base_margin_threshold']:.3f}",
                }
            },
            "draft_lattice": {
                "env": {
                    "DISMANTLE_EAGLE5_LATTICE_WIDTH": str(policies["draft_lattice"][0]["branch_width"]),
                    "DISMANTLE_EAGLE5_LATTICE_K": str(policies["draft_lattice"][0]["max_depth"]),
                }
            },
        },
    }
    _write_json_atomic(args.out, payload)
    best = payload["policies"]["best_deployable"]
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(
        f"[frontier] deployable_best={best['kind']} projected={best['projected_dec_tps']:.1f} "
        f"accepted={best['accepted_draft_tokens_per_verify']:.2f} → {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
