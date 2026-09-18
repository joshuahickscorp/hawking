"""Evaluate a fit-magnitude residual-translation candidate directly on HCLI.

This is a separate experimental arm from the unit-vector residual addition.
The translation magnitude is derived from the training-fold centroid distance,
then applied out-of-fold with the same magnitude to a matched random-direction
null. Optional prefill-only mode applies the offset to the prompt prefill and
skips cached one-token decode passes. Conditional mode gates the offset to a
train-derived refusal-like state. It never writes weights or creates a
candidate artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = ROOT / "tools" / "future"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from moe_refusal_locality import run_crossfit_hcli_contract_residual_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument(
        "--translation-scale", type=float, default=0.25,
        help="fraction of each fold's train centroid distance used as the offset",
    )
    parser.add_argument(
        "--prefill-only", action="store_true",
        help="apply the activation edit to prompt prefill, not cached decode steps",
    )
    parser.add_argument(
        "--conditional-pass", action="store_true",
        help="gate translation to train-calibrated refusal-like residual states",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "receipts" / "future" / "KIMI_EXPERIMENTAL_HCLI_FIT_TRANSLATION_LIVE_20260910.json",
    )
    args = parser.parse_args()

    import dsv3_native_loader as loader
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(args.spec), trust_remote_code=True, use_fast=False
    )
    model, _cfg = loader.load(
        args.spec, device=args.device, dtype=args.dtype, min_free_gib=12.0
    )
    report = run_crossfit_hcli_contract_residual_candidate(
        model,
        tok,
        args.device,
        strength=args.strength,
        translation_scale=args.translation_scale,
        prefill_only=args.prefill_only,
        conditional_pass=args.conditional_pass,
    )
    effect = bool(report.get("causal_effect_established"))
    result = {
        "schema": "hawking.kimi.experimental_hcli_fit_translation_live.v1",
        "status": (
            "EXPERIMENTAL_HCLI_FIT_TRANSLATION_POSITIVE"
            if effect else "EXPERIMENTAL_HCLI_FIT_TRANSLATION_NEGATIVE"
        ),
        "source_model": "KIMI_BASE",
        "candidate_id": (
            "KIMI_EXPERIMENTAL_HCLI_FIT_CONDITIONAL_TRANSLATION"
            if args.conditional_pass else
            "KIMI_EXPERIMENTAL_HCLI_FIT_PREFILL_TRANSLATION"
            if args.prefill_only else
            "KIMI_EXPERIMENTAL_HCLI_FIT_TRANSLATION"
        ),
        "candidate_method": (
            "experimental_hcli_fit_conditional_residual_translation"
            if args.conditional_pass else
            "experimental_hcli_fit_magnitude_prefill_residual_translation"
            if args.prefill_only else
            "experimental_hcli_fit_magnitude_residual_translation"
        ),
        "specimen": str(args.spec),
        "device": args.device,
        "dtype": args.dtype,
        "strength": float(args.strength),
        "translation_scale": float(args.translation_scale),
        "prefill_only": bool(args.prefill_only),
        "conditional_pass": bool(args.conditional_pass),
        "report": report,
        "authority": {
            "weights_written": False,
            "external_writes": False,
            "tools_executed": False,
            "daemon_contacted": False,
        },
        "claim_boundary": (
            "Experimental candidate-specific HCLI contract evidence only. The "
            "fit-magnitude residual translation is applied out-of-fold in memory; "
            "this does not create a candidate artifact or authorize promotion."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "report": {
            key: report.get(key)
            for key in (
                "baseline_passed", "treated_passed", "null_passed",
                "pass_rate_delta", "null_pass_rate_delta",
                "causal_effect_established", "causal_sample_sufficient",
                "layer", "hidden_index", "translation_magnitudes",
            )
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
