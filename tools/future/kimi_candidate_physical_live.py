"""Measure OPE physical parity against KIMI_BASE on a fixed local workload."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = ROOT / "tools" / "future"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from kimi_epistemic_live import TASKS
from moe_refusal_locality import (
    HARMFUL,
    HARMLESS,
    capture_observations,
    heldout_separation,
    refusal_labels,
    stratified_split,
    writer_weight_intervention,
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timed_workload(model, tok, device, *, max_new_tokens: int) -> dict[str, Any]:
    import torch

    rows = []
    started = time.perf_counter()
    for task in TASKS:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": task["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok(prompt, return_tensors="pt").to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **ids, max_new_tokens=int(max_new_tokens), do_sample=False
            )
        elapsed = time.perf_counter() - t0
        completion_tokens = int(generated.shape[-1] - ids["input_ids"].shape[-1])
        rows.append({
            "task_id": task["task_id"],
            "elapsed_s": round(elapsed, 4),
            "completion_tokens": completion_tokens,
            "tokens_per_second": round(completion_tokens / max(elapsed, 1e-9), 3),
        })
        del generated, ids
        if str(device).startswith("mps"):
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass
    elapsed_total = time.perf_counter() - started
    tokens = sum(row["completion_tokens"] for row in rows)
    return {
        "tasks": len(rows),
        "elapsed_s": round(elapsed_total, 3),
        "completion_tokens": tokens,
        "tokens_per_second": round(tokens / max(elapsed_total, 1e-9), 3),
        "rows": rows,
    }


def _parity(baseline: dict[str, Any], treated: dict[str, Any], null: dict[str, Any]) -> dict[str, Any]:
    b = float(baseline["elapsed_s"])
    t = float(treated["elapsed_s"])
    n = float(null["elapsed_s"])
    ratio = t / max(b, 1e-9)
    null_ratio = n / max(b, 1e-9)
    return {
        "baseline_elapsed_s": b,
        "treated_elapsed_s": t,
        "null_elapsed_s": n,
        "treated_to_baseline_ratio": ratio,
        "null_to_baseline_ratio": null_ratio,
        "same_task_count": (
            baseline.get("tasks") == treated.get("tasks") == null.get("tasks")
        ),
        "no_physical_regression": bool(
            baseline.get("tasks") == treated.get("tasks") == null.get("tasks")
            and ratio <= 1.25
        ),
        "threshold": "treated wall time <= 1.25x KIMI_BASE on fixed seven-task workload",
    }


def run(spec: Path, *, device: str, dtype: str, max_new_tokens: int) -> dict[str, Any]:
    import torch
    import dsv3_native_loader as loader
    from transformers import AutoTokenizer

    started = time.time()
    tok = AutoTokenizer.from_pretrained(
        str(spec), trust_remote_code=True, use_fast=False
    )
    model, _cfg = loader.load(spec, device=device, dtype=dtype, min_free_gib=12.0)
    calibration_prompts = list(HARMFUL) + list(HARMLESS)
    labels, transcripts = refusal_labels(model, tok, calibration_prompts, device)
    observations = list(capture_observations(
        model, tok, calibration_prompts, device, {}
    ))
    refused = [row for row, label in zip(observations, labels) if label]
    complied = [row for row, label in zip(observations, labels) if not label]
    if len(refused) < 4 or len(complied) < 4:
        return {
            "schema": "hawking.kimi.candidate_physical_live.v1",
            "status": "INSUFFICIENT_CALIBRATION_BEHAVIOR",
            "source_model": "KIMI_BASE",
            "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPE",
            "specimen": str(spec),
            "calibration": {"refused": len(refused), "complied": len(complied)},
        }
    p_train, p_test, n_train, n_test = stratified_split(
        refused, complied, train_fraction=0.75, seed=8201
    )
    metrics: dict[str, Any] = {}
    directions: dict[int, Any] = {}
    for hidden_index in sorted(observations[0]["hidden"]):
        positive = [row["hidden"][hidden_index] for row in refused]
        negative = [row["hidden"][hidden_index] for row in complied]
        fit = heldout_separation(
            positive,
            negative,
            train_positive=[positive[i] for i in p_train],
            train_negative=[negative[i] for i in n_train],
            heldout_positive=[positive[i] for i in p_test],
            heldout_negative=[negative[i] for i in n_test],
        )
        direction = fit.pop("direction")
        fit.pop("_fit_anchor", None)
        fit.pop("fit_threshold", None)
        fit.pop("fit_anchor_norm", None)
        if fit.get("direction_valid"):
            directions[hidden_index] = direction
        metrics[str(hidden_index)] = fit
    peak_hidden = max(
        directions,
        key=lambda index: (
            metrics[str(index)].get("heldout_auroc") or 0.5,
            metrics[str(index)].get("heldout_margin") or 0.0,
        ),
    )
    layer = int(peak_hidden) - 1
    direction = directions[peak_hidden]
    baseline = _timed_workload(model, tok, device, max_new_tokens=max_new_tokens)
    with writer_weight_intervention(
        model, layer, direction, 1.0, "shared_experts_down"
    ):
        treated = _timed_workload(model, tok, device, max_new_tokens=max_new_tokens)
    generator = torch.Generator(device="cpu").manual_seed(8202)
    random_direction = torch.randn(len(direction), generator=generator)
    with writer_weight_intervention(
        model, layer, random_direction, 1.0, "shared_experts_down"
    ):
        null = _timed_workload(model, tok, device, max_new_tokens=max_new_tokens)
    parity = _parity(baseline, treated, null)
    return {
        "schema": "hawking.kimi.candidate_physical_live.v1",
        "status": (
            "CANDIDATE_PHYSICAL_PARITY_PASSED"
            if parity["no_physical_regression"]
            else "CANDIDATE_PHYSICAL_PARITY_FAILED"
        ),
        "source_model": "KIMI_BASE",
        "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPE",
        "candidate_method": "OPE",
        "candidate_scope": "peak_layer_shared_writer_on_fixed_epistemic_workload",
        "specimen": str(spec),
        "device": device,
        "dtype": dtype,
        "elapsed_s": round(time.time() - started, 1),
        "direction_fit": {
            "fit_scope": "generic_behavior_train_only; physical_tasks_disjoint",
            "train_refused": len(p_train),
            "train_complied": len(n_train),
            "heldout_refused": len(p_test),
            "heldout_complied": len(n_test),
            "peak_hidden_index": int(peak_hidden),
            "peak_model_layer": layer,
            "peak_heldout_auroc": metrics[str(peak_hidden)].get("heldout_auroc"),
        },
        "parity": parity,
        "baseline": baseline,
        "treated": treated,
        "matched_random_null": null,
        "authority": {
            "weights_written": False,
            "external_writes": False,
            "tools_executed": False,
            "daemon_contacted": False,
        },
        "claim_boundary": (
            "Candidate-specific OPE physical parity only. Measurements use a fixed "
            "local workload and reversible in-memory writer changes; this does not "
            "create an artifact or authorize promotion."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "receipts" / "future" / "KIMI_CANDIDATE_OPE_PHYSICAL_LIVE_20260910.json",
    )
    args = parser.parse_args()
    result = run(
        args.spec,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["source_script_sha256"] = _sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "parity": result.get("parity"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
