"""Evaluate an OPE candidate on the independent HCLI contract battery.

The candidate direction is fit from a separate generic refusal/compliance
calibration set, then applied only in memory to the 16 HCLI cases. Baseline,
treated, and matched-random writer-null transcripts are retained. No weights
are written and no external action is possible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = ROOT / "tools" / "future"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from moe_refusal_locality import (
    HARMFUL,
    HARMLESS,
    capture_observations,
    hcli_contract_intervention_report,
    hcli_contract_labels,
    refusal_labels,
    run_hcli_writer_contract_intervention,
    heldout_separation,
    stratified_split,
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(spec: Path, *, device: str, dtype: str, max_new_tokens: int, limit: int = 0) -> dict[str, Any]:
    import torch
    import dsv3_native_loader as loader
    from transformers import AutoTokenizer

    started = time.time()
    tok = AutoTokenizer.from_pretrained(str(spec), trust_remote_code=True)
    model, cfg = loader.load(spec, device=device, dtype=dtype, min_free_gib=12.0)
    harmful = HARMFUL[:limit] if limit else HARMFUL
    harmless = HARMLESS[:limit] if limit else HARMLESS
    calibration_prompts = harmful + harmless
    baseline_classes, calibration_transcripts = refusal_labels(
        model, tok, calibration_prompts, device)
    sink: dict[Any, Any] = {}
    observations = list(capture_observations(
        model, tok, calibration_prompts, device, sink))
    refused_observations = [
        row for row, refused in zip(observations, baseline_classes) if refused
    ]
    complied_observations = [
        row for row, refused in zip(observations, baseline_classes) if not refused
    ]
    if len(refused_observations) < 4 or len(complied_observations) < 4:
        return {
            "schema": "hawking.kimi.candidate_hcli_live.v1",
            "status": "INSUFFICIENT_CALIBRATION_BEHAVIOR",
            "source_model": "KIMI_BASE",
            "specimen": str(spec),
            "calibration": {
                "harmful": len(harmful),
                "harmless": len(harmless),
                "refused": len(refused_observations),
                "complied": len(complied_observations),
            },
            "claim_boundary": "No candidate intervention was run because the calibration set did not contain four rows per measured behavior class.",
        }

    pos_train, pos_test, neg_train, neg_test = stratified_split(
        refused_observations, complied_observations, train_fraction=0.75, seed=6101)
    metrics: dict[str, Any] = {}
    directions: dict[int, Any] = {}
    for hidden_index in sorted(observations[0]["hidden"]):
        positive = [row["hidden"][hidden_index] for row in refused_observations]
        negative = [row["hidden"][hidden_index] for row in complied_observations]
        fit = heldout_separation(
            positive,
            negative,
            train_positive=[positive[i] for i in pos_train],
            train_negative=[negative[i] for i in neg_train],
            heldout_positive=[positive[i] for i in pos_test],
            heldout_negative=[negative[i] for i in neg_test],
        )
        direction = fit.pop("direction")
        # Fit anchors/threshold internals are useful to the intervention
        # context but are not receipt data and may be NumPy arrays. Keep the
        # scalar fit metrics only; the actual candidate edit is still purely
        # in-memory below.
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
    peak_model_layer = int(peak_hidden) - 1
    hcli = run_hcli_writer_contract_intervention(
        model,
        tok,
        device,
        layer=peak_model_layer,
        direction=directions[peak_hidden],
        writer="shared_experts_down",
        strength=1.0,
        norm_preserve=False,
        seed=6102,
    )
    return {
        "schema": "hawking.kimi.candidate_hcli_live.v1",
        "status": (
            "CANDIDATE_HCLI_EFFECT_ESTABLISHED"
            if hcli.get("causal_effect_established") and hcli.get("causal_sample_sufficient")
            else "CANDIDATE_HCLI_NULL_OR_INCONSISTENT"
        ),
        "source_model": "KIMI_BASE",
        "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPE",
        "candidate_method": "OPE",
        "candidate_scope": "peak_layer_shared_writer_on_independent_hcli_contract",
        "specimen": str(spec),
        "device": device,
        "dtype": dtype,
        "elapsed_s": round(time.time() - started, 1),
        "calibration": {
            "harmful": len(harmful),
            "harmless": len(harmless),
            "refused": len(refused_observations),
            "complied": len(complied_observations),
            "refusal_labels": baseline_classes,
            "transcript_snippets": [text[:200] for text in calibration_transcripts],
        },
        "direction_fit": {
            "fit_scope": "generic_behavior_train_only; HCLI_cases_disjoint",
            "train_refused": len(pos_train),
            "train_complied": len(neg_train),
            "heldout_refused": len(pos_test),
            "heldout_complied": len(neg_test),
            "peak_hidden_index": int(peak_hidden),
            "peak_model_layer": peak_model_layer,
            "peak_heldout_auroc": metrics[str(peak_hidden)].get("heldout_auroc"),
            "all_layers": metrics,
        },
        "hcli_contract_intervention": hcli,
        "authority": {
            "weights_written": False,
            "external_writes": False,
            "tools_executed": False,
            "daemon_contacted": False,
        },
        "claim_boundary": (
            "Candidate-specific HCLI outcome evidence only. The direction is fit "
            "on a disjoint generic calibration set and all candidate changes are "
            "reversible in memory. This does not create an artifact, prove broad "
            "capability, or authorize resident deployment."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "receipts" / "future" / "KIMI_CANDIDATE_OPE_HCLI_LIVE_20260910.json",
    )
    args = parser.parse_args()
    result = run(
        args.spec,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["source_script_sha256"] = _sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "hcli": {
            key: result.get("hcli_contract_intervention", {}).get(key)
            for key in ("causal_effect_established", "causal_sample_sufficient", "pass_rate_delta", "null_pass_rate_delta")
            if key in result.get("hcli_contract_intervention", {})
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
