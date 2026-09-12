#!/usr/bin/env python3
"""Probe one real KIMI model graph with a routed Nova organ replacement.

Only layer 10 ``up_proj`` is replaced.  Every other KIMI_BASE weight remains
unchanged, so this is intentionally an integration discriminator rather than a
reduced-model claim.  Baseline and patched NLL/generation are paired inside
one guarded process; promotion and Odyssey gates are never touched.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools" / "future") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "future"))

from tools.future import gravity_outlier_eval as G  # noqa: E402
from tools.future.kimi_nova_routed_kernel import ExpertLowRankRoutedLinear  # noqa: E402

SCHEMA = "hawking.future.kimi_nova_full_model_organ_integration_probe.v1"
TARGET_LAYER = 10
TARGET_SUFFIX = "up_proj"


def _repeat_metrics(text: str, tok) -> dict[str, Any]:
    ids = tok.encode(text)
    runs = 1
    cur = 1
    fourgrams: list[tuple[int, ...]] = []
    for i in range(1, len(ids)):
        cur = cur + 1 if ids[i] == ids[i - 1] else 1
        runs = max(runs, cur)
    for i in range(max(0, len(ids) - 3)):
        fourgrams.append(tuple(ids[i:i + 4]))
    return {
        "tokens": len(ids),
        "unique_token_fraction": len(set(ids)) / max(1, len(ids)),
        "fourgram_repeat_fraction": 1 - len(set(fourgrams)) / max(1, len(fourgrams)),
        "max_run": runs,
    }


def _nll(model, tok, text: str, mx) -> tuple[float, float, float]:
    ids_np = tok.encode(text)[:512]
    ids = mx.array([ids_np])
    started = time.perf_counter()
    logits = model(ids[:, :-1]).astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    value = -mx.take_along_axis(
        log_probs, ids[:, 1:, None], axis=-1
    ).squeeze(-1).mean()
    mx.eval(value)
    elapsed = time.perf_counter() - started
    nll = float(value.item())
    return nll, 2.718281828459045 ** nll, elapsed


def _generate(model, tok, prompts: list[str], max_tokens: int, generate) -> dict[str, Any]:
    rows = []
    started = time.perf_counter()
    for prompt in prompts:
        one_started = time.perf_counter()
        text = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
        rows.append({"prompt": prompt, "text": text,
                     "timing_s": time.perf_counter() - one_started,
                     "metrics": _repeat_metrics(text, tok)})
    timings = [row["timing_s"] for row in rows]
    metrics = [row["metrics"] for row in rows]
    return {
        "rows": rows,
        "wall_s": time.perf_counter() - started,
        "median_s": sorted(timings)[len(timings) // 2],
        "median_fourgram_repeat_fraction": sorted(
            row["fourgram_repeat_fraction"] for row in metrics
        )[len(metrics) // 2],
        "median_unique_token_fraction": sorted(
            row["unique_token_fraction"] for row in metrics
        )[len(metrics) // 2],
        "max_run": max(row["max_run"] for row in metrics),
    }


def probe(receipt_path: Path, output: Path, max_tokens: int) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import generate
    from tools.future.campaign_memory_guard import require_ok, sample, watch

    entry = require_ok("KIMI Nova one-organ integration probe", expected_gb=32.0).as_dict()
    load_started = time.perf_counter()
    model, tok = G._load()
    load_s = time.perf_counter() - load_started
    loaded = sample(expected_gb=32.0).as_dict()

    receipt = json.loads(receipt_path.read_text())
    best = min(receipt["results"], key=lambda row: row["heldout"]["mean_relative_l2"])
    artifact = Path(best["factor_artifact"]["path"])
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    consumer = ExpertLowRankRoutedLinear.from_bf16_artifact(
        artifact, int(receipt["expert_count"])
    )
    try:
        target = model.layers[TARGET_LAYER].mlp.switch_mlp.up_proj
    except (AttributeError, IndexError) as exc:
        raise RuntimeError("KIMI model graph lacks layer10 switch_mlp.up_proj") from exc
    target_name = next(
        (name for name, module in model.named_modules() if module is target),
        f"language_model.model.layers.{TARGET_LAYER}.mlp.switch_mlp.{TARGET_SUFFIX}",
    )

    texts = [G.NLL_TEXT, " ".join(G.PROMPTS)]
    baseline_nll = []
    patched_nll = []
    with watch(interval_s=2.0) as live_watch:
        for text in texts:
            baseline_nll.append(_nll(model, tok, text, mx))
        baseline_generation = _generate(model, tok, G.PROMPTS, max_tokens, generate)

        model.layers[TARGET_LAYER].mlp.switch_mlp.up_proj = consumer
        for text in texts:
            patched_nll.append(_nll(model, tok, text, mx))
        patched_generation = _generate(model, tok, G.PROMPTS, max_tokens, generate)

    def summarize(values: list[tuple[float, float, float]]) -> dict[str, Any]:
        return {
            "rows": [{"nll": nll, "ppl": ppl, "elapsed_s": elapsed}
                     for nll, ppl, elapsed in values],
            "mean_nll": sum(row[0] for row in values) / len(values),
            "mean_ppl": sum(row[1] for row in values) / len(values),
            "mean_elapsed_s": sum(row[2] for row in values) / len(values),
        }

    baseline_summary = summarize(baseline_nll)
    patched_summary = summarize(patched_nll)
    out = {
        "schema": SCHEMA,
        "status": "NOVA_ONE_ORGAN_FULL_GRAPH_PROBE_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "training_receipt": str(receipt_path),
        "selected_variant": best["variant"],
        "factor_artifact": best["factor_artifact"],
        "target": {"layer": TARGET_LAYER, "module": target_name,
                   "replacement": "independent_expert_lowrank_AeBeT"},
        "model_scope": {
            "replaced_organs": 1,
            "replaced_tensor_family": "layer10 routed up_proj",
            "other_weights": "unchanged KIMI_BASE dense/MLX weights",
            "full_model_direct_execution": False,
        },
        "guard": {"entry": entry, "loaded": loaded,
                  "watch_worst_state": live_watch.worst,
                  "watch_samples": live_watch.samples,
                  "end": sample().as_dict()},
        "load": {"elapsed_s": load_s, "runtime": "MLX"},
        "workload": {"nll_text_count": len(texts), "generation_prompts": len(G.PROMPTS),
                      "max_tokens": max_tokens, "sampling": "mlx_lm.generate default"},
        "baseline": {"nll": baseline_summary, "generation": baseline_generation},
        "patched": {"nll": patched_summary, "generation": patched_generation},
        "deltas": {
            "mean_nll_patched_minus_baseline": patched_summary["mean_nll"] - baseline_summary["mean_nll"],
            "mean_ppl_ratio": patched_summary["mean_ppl"] / max(baseline_summary["mean_ppl"], 1e-12),
            "generation_median_s_delta": patched_generation["median_s"] - baseline_generation["median_s"],
            "generation_repeat_delta": patched_generation["median_fourgram_repeat_fraction"] - baseline_generation["median_fourgram_repeat_fraction"],
            "generation_unique_fraction_delta": patched_generation["median_unique_token_fraction"] - baseline_generation["median_unique_token_fraction"],
        },
        "claim_boundary": (
            "Paired one-organ replacement in the real KIMI_BASE MLX graph. The dense model "
            "remains present for all unreplaced organs, so this is an integration discriminator, "
            "not a complete reduced representation, direct full-model execution, capability "
            "qualification, TPS result, NX artifact, or promotion evidence."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "script": str(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": out["status"],
                      "target": out["target"], "deltas": out["deltas"],
                      "baseline_nll": baseline_summary, "patched_nll": patched_summary,
                      "watch_worst_state": live_watch.worst}, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()
    if args.max_tokens <= 0:
        raise SystemExit("max-tokens must be positive")
    probe(args.receipt, args.output, args.max_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
