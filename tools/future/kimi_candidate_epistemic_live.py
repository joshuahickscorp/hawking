"""Measure OPE candidate epistemic parity against the KIMI_BASE battery."""
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

from kimi_epistemic_live import TASKS, score_reply, summarize
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


def _generate_rows(model, tok, device, *, tasks, max_new_tokens: int) -> list[dict[str, Any]]:
    import torch

    rows = []
    for task in tasks:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": task["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **ids, max_new_tokens=int(max_new_tokens), do_sample=False
            )
        reply = tok.decode(
            generated[0][ids["input_ids"].shape[1]:], skip_special_tokens=True
        )
        rows.append(score_reply(task, reply))
        del generated, ids
        if str(device).startswith("mps"):
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass
    return rows


def _parity(baseline: dict[str, Any], treated: dict[str, Any], null: dict[str, Any]) -> dict[str, Any]:
    b = baseline.get("summary") or {}
    t = treated.get("summary") or {}
    n = null.get("summary") or {}
    brier_delta = (
        float(t["brier_score"]) - float(b["brier_score"])
        if t.get("brier_score") is not None and b.get("brier_score") is not None
        else None
    )
    ece_delta = (
        float(t["ece_two_bin"]) - float(b["ece_two_bin"])
        if t.get("ece_two_bin") is not None and b.get("ece_two_bin") is not None
        else None
    )
    no_regression = bool(
        b.get("qualified") is True
        and t.get("qualified") is True
        and t.get("known_passed", 0) >= b.get("known_passed", 0)
        and t.get("unknown_passed", 0) >= b.get("unknown_passed", 0)
        and brier_delta is not None
        and brier_delta <= 0.05
        and ece_delta is not None
        and ece_delta <= 0.05
    )
    return {
        "baseline_qualified": bool(b.get("qualified")),
        "treated_qualified": bool(t.get("qualified")),
        "null_qualified": bool(n.get("qualified")),
        "baseline_passed": b.get("passed"),
        "treated_passed": t.get("passed"),
        "null_passed": n.get("passed"),
        "baseline_brier": b.get("brier_score"),
        "treated_brier": t.get("brier_score"),
        "null_brier": n.get("brier_score"),
        "brier_delta": brier_delta,
        "baseline_ece": b.get("ece_two_bin"),
        "treated_ece": t.get("ece_two_bin"),
        "null_ece": n.get("ece_two_bin"),
        "ece_delta": ece_delta,
        "no_regression": no_regression,
        "metric": "candidate_epistemic_parity_with_calibration_nonregression",
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
    baseline_labels, calibration_transcripts = refusal_labels(
        model, tok, calibration_prompts, device
    )
    observations = list(capture_observations(
        model, tok, calibration_prompts, device, {}
    ))
    refused = [row for row, label in zip(observations, baseline_labels) if label]
    complied = [row for row, label in zip(observations, baseline_labels) if not label]
    if len(refused) < 4 or len(complied) < 4:
        return {
            "schema": "hawking.kimi.candidate_epistemic_live.v1",
            "status": "INSUFFICIENT_CALIBRATION_BEHAVIOR",
            "source_model": "KIMI_BASE",
            "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPE",
            "specimen": str(spec),
            "calibration": {"refused": len(refused), "complied": len(complied)},
        }

    p_train, p_test, n_train, n_test = stratified_split(
        refused, complied, train_fraction=0.75, seed=8101
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

    baseline_rows = _generate_rows(
        model, tok, device, tasks=TASKS, max_new_tokens=max_new_tokens
    )
    with writer_weight_intervention(
        model, layer, direction, 1.0, "shared_experts_down"
    ):
        treated_rows = _generate_rows(
            model, tok, device, tasks=TASKS, max_new_tokens=max_new_tokens
        )
    generator = torch.Generator(device="cpu").manual_seed(8102)
    random_direction = torch.randn(len(direction), generator=generator)
    with writer_weight_intervention(
        model, layer, random_direction, 1.0, "shared_experts_down"
    ):
        null_rows = _generate_rows(
            model, tok, device, tasks=TASKS, max_new_tokens=max_new_tokens
        )
    baseline_summary = summarize(baseline_rows)
    treated_summary = summarize(treated_rows)
    null_summary = summarize(null_rows)
    parity = _parity(
        {"summary": baseline_summary},
        {"summary": treated_summary},
        {"summary": null_summary},
    )
    return {
        "schema": "hawking.kimi.candidate_epistemic_live.v1",
        "status": (
            "CANDIDATE_EPISTEMIC_PARITY_PASSED"
            if parity["no_regression"]
            else "CANDIDATE_EPISTEMIC_PARITY_FAILED"
        ),
        "source_model": "KIMI_BASE",
        "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPE",
        "candidate_method": "OPE",
        "candidate_scope": "peak_layer_shared_writer_on_epistemic_battery",
        "specimen": str(spec),
        "device": device,
        "dtype": dtype,
        "elapsed_s": round(time.time() - started, 1),
        "calibration": {
            "harmful": len(HARMFUL),
            "harmless": len(HARMLESS),
            "refused": len(refused),
            "complied": len(complied),
            "refusal_labels": baseline_labels,
            "transcript_snippets": [text[:200] for text in calibration_transcripts],
        },
        "direction_fit": {
            "fit_scope": "generic_behavior_train_only; epistemic_tasks_disjoint",
            "train_refused": len(p_train),
            "train_complied": len(n_train),
            "heldout_refused": len(p_test),
            "heldout_complied": len(n_test),
            "peak_hidden_index": int(peak_hidden),
            "peak_model_layer": layer,
            "peak_heldout_auroc": metrics[str(peak_hidden)].get("heldout_auroc"),
            "all_layers": metrics,
        },
        "parity": parity,
        "baseline": {"summary": baseline_summary, "tasks": baseline_rows},
        "treated": {"summary": treated_summary, "tasks": treated_rows},
        "matched_random_null": {"summary": null_summary, "tasks": null_rows},
        "authority": {
            "weights_written": False,
            "external_writes": False,
            "tools_executed": False,
            "daemon_contacted": False,
        },
        "claim_boundary": (
            "Candidate-specific OPE epistemic parity only. The candidate edit is "
            "reversible in memory, fitted from disjoint generic behavior rows, "
            "and compared with a matched random writer null; this does not create "
            "an artifact or authorize promotion."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "receipts" / "future" / "KIMI_CANDIDATE_OPE_EPISTEMIC_LIVE_20260910.json",
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
