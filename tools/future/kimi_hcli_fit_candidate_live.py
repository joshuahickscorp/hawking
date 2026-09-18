"""Cross-fit an experimental writer candidate directly on HCLI outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = ROOT / "tools" / "future"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from moe_refusal_locality import run_crossfit_hcli_contract_candidate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument(
        "--output-translation-scale", type=float, default=0.0,
        help="use a train-scaled writer-output translation instead of projection",
    )
    parser.add_argument(
        "--prefill-only", action="store_true",
        help="in translation mode, skip cached one-token decode passes",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "receipts" / "future" / "KIMI_EXPERIMENTAL_HCLI_FIT_WRITER_LIVE_20260910.json",
    )
    args = parser.parse_args()
    import dsv3_native_loader as loader
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.spec), trust_remote_code=True)
    model, _cfg = loader.load(
        args.spec, device=args.device, dtype=args.dtype, min_free_gib=12.0
    )
    report = run_crossfit_hcli_contract_candidate(
        model,
        tok,
        args.device,
        strength=args.strength,
        output_translation_scale=args.output_translation_scale,
        prefill_only=args.prefill_only,
    )
    effect = bool(report.get("causal_effect_established"))
    translated = args.output_translation_scale > 0.0
    result = {
        "schema": "hawking.kimi.experimental_hcli_fit_writer_live.v1",
        "status": "EXPERIMENTAL_HCLI_FIT_WRITER_POSITIVE" if effect else "EXPERIMENTAL_HCLI_FIT_WRITER_NEGATIVE",
        "source_model": "KIMI_BASE",
        "candidate_id": (
            "KIMI_EXPERIMENTAL_HCLI_FIT_WRITER_TRANSLATION"
            if translated else "KIMI_EXPERIMENTAL_HCLI_FIT_WRITER"
        ),
        "candidate_method": (
            "experimental_hcli_fit_shared_writer_output_translation"
            if translated else "experimental_hcli_fit_shared_writer"
        ),
        "specimen": str(args.spec),
        "device": args.device,
        "dtype": args.dtype,
        "output_translation_scale": float(args.output_translation_scale),
        "prefill_only": bool(args.prefill_only),
        "report": report,
        "authority": {
            "weights_written": False,
            "external_writes": False,
            "tools_executed": False,
            "daemon_contacted": False,
        },
        "claim_boundary": (
            "Experimental candidate-specific HCLI contract evidence only. The "
            "writer directions are fit and evaluated out-of-fold in memory; this "
            "does not create a candidate artifact or authorize promotion."
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
            )
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
