"""Cross-fit an experimental additive residual candidate directly on HCLI."""
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
        "--output", type=Path,
        default=ROOT / "receipts" / "future" / "KIMI_EXPERIMENTAL_HCLI_FIT_RESIDUAL_LIVE_20260910.json",
    )
    args = parser.parse_args()

    import dsv3_native_loader as loader
    from transformers import AutoTokenizer

    # This specimen supplies a TikTokenTokenizer without tokenizer.json; the
    # isolated runtime must select its slow/custom implementation explicitly.
    tok = AutoTokenizer.from_pretrained(
        str(args.spec), trust_remote_code=True, use_fast=False
    )
    model, _cfg = loader.load(
        args.spec, device=args.device, dtype=args.dtype, min_free_gib=12.0
    )
    report = run_crossfit_hcli_contract_residual_candidate(
        model, tok, args.device, strength=args.strength
    )
    effect = bool(report.get("causal_effect_established"))
    result = {
        "schema": "hawking.kimi.experimental_hcli_fit_residual_live.v1",
        "status": (
            "EXPERIMENTAL_HCLI_FIT_RESIDUAL_POSITIVE"
            if effect else "EXPERIMENTAL_HCLI_FIT_RESIDUAL_NEGATIVE"
        ),
        "source_model": "KIMI_BASE",
        "candidate_id": "KIMI_EXPERIMENTAL_HCLI_FIT_RESIDUAL",
        "candidate_method": "experimental_hcli_fit_residual_addition",
        "specimen": str(args.spec),
        "device": args.device,
        "dtype": args.dtype,
        "report": report,
        "authority": {
            "weights_written": False,
            "external_writes": False,
            "tools_executed": False,
            "daemon_contacted": False,
        },
        "claim_boundary": (
            "Experimental candidate-specific HCLI contract evidence only. The "
            "pass-direction is fit and evaluated out-of-fold in memory as a "
            "reversible activation addition; this does not create a candidate "
            "artifact or authorize promotion."
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
                "layer", "hidden_index", "peak_heldout_auroc",
            )
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
