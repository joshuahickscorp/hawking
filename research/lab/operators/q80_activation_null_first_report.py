#!/usr/bin/env python3
"""Report constant-mean null on a Q80 route capture BEFORE any family result.

Mirrors q30_activation_null_first_report. Accepts the Q30-compatible all-layer
capture shape (per-step layers[] with router_input_hidden_f32le + selected
expert ids) once a Q80 capture exists.

Refuses to invent nulls when the capture is missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (  # noqa: E402
    capture_is_all_layer,
    collect_expert_activations,
)
from lab.operators.q30_activation_null_first_report import (  # noqa: E402
    _headline,
    _score_layer,
)
from lab.operators.q30_activation_aware_family_probe import (  # noqa: E402
    load_capture,
    sha256_file,
    utc_now,
)
from lab.operators.qwen30b_gravity_pack import load_weight_map  # noqa: E402

DEFAULT_MODEL_DIR = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/"
    "qwen-80b/Qwen3-Coder-Next"
)

ALL_LAYER_RESULT_SCHEMA = (
    "hawking.ascension.qwen80_broad_activation_all_layer_route_capture_result.v1"
)
# Accept the Q30 all-layer schema too if a shared writer is used.
Q30_ALL_LAYER_RESULT_SCHEMA = (
    "hawking.ascension.qwen30_broad_activation_all_layer_route_capture_result.v1"
)


def report_null(
    *,
    capture_run: Path,
    model_dir: Path,
    label: str,
    min_tokens: int = 32,
    components: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
    top_experts: int = 6,
    max_layers: int | None = None,
) -> dict:
    capture = load_capture(capture_run)
    schema = str(capture.get("schema") or "")
    all_layer = capture_is_all_layer(capture) or schema in {
        ALL_LAYER_RESULT_SCHEMA,
        Q30_ALL_LAYER_RESULT_SCHEMA,
    }
    by_layer_expert, prov = collect_expert_activations(capture_run, capture)
    layers = sorted({layer for layer, _ in by_layer_expert})
    if max_layers is not None:
        layers = [L for L in layers if L < int(max_layers)]
    if not layers:
        raise RuntimeError("capture has no layers with retained hidden hits")

    weight_map = load_weight_map(model_dir)
    rows: list[dict] = []
    per_layer: list[dict] = []
    for layer in layers:
        layer_rows = _score_layer(
            layer=layer,
            by_layer_expert=by_layer_expert,
            model_dir=model_dir,
            weight_map=weight_map,
            min_tokens=min_tokens,
            components=components,
            top_experts=top_experts,
        )
        rows.extend(layer_rows)
        per_layer.append(
            {
                "layer": int(layer),
                "headline": _headline(layer_rows),
                "n_experts_scored": len({r["expert"] for r in layer_rows}),
                "n_rows": len(layer_rows),
            }
        )

    overall = _headline(rows)
    return {
        "schema": "hawking.ascension.qwen80_activation_null_first.v1",
        "label": label,
        "reported_at": utc_now(),
        "capture_run": str(capture_run),
        "capture_result_sha256": sha256_file(capture_run / "capture-result.json"),
        "capture_schema": schema,
        "all_layer_capture": all_layer,
        "layers_scored": layers,
        "activation_provenance": prov,
        "headline": overall,
        "per_layer": per_layer,
        "rows": rows,
        "claim_boundary": {
            "null_only_no_family_result": True,
            "diagnostic_not_capability": True,
            "per_layer_null_precedes_any_family_result": True,
            "model": "qwen80",
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-run", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--label", type=str, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--min-tokens", type=int, default=32)
    p.add_argument("--top-experts", type=int, default=6)
    p.add_argument("--max-layers", type=int, default=None)
    args = p.parse_args()

    capture_run = args.capture_run.expanduser().resolve()
    if not (capture_run / "capture-result.json").is_file():
        missing = {
            "schema": "hawking.ascension.qwen80_activation_null_first.v1",
            "status": "REFUSED_CAPTURE_MISSING",
            "label": args.label,
            "reported_at": utc_now(),
            "capture_run": str(capture_run),
            "headline": {
                "verdict": "CAPTURE_MISSING_NULL_NOT_MEASURABLE",
                "mean_null_all_scored": None,
                "n_rows": 0,
            },
            "per_layer": [],
            "claim_boundary": {
                "null_only_no_family_result": True,
                "refused_to_invent_nulls": True,
            },
            "note": (
                "No capture-result.json. Run q80_activation_capture_readiness for "
                "the exact missing pieces (GQA encode + capture binary)."
            ),
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(missing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(missing["headline"], indent=2))
        return 2

    doc = report_null(
        capture_run=capture_run,
        model_dir=args.model_dir.expanduser().resolve(),
        label=args.label,
        min_tokens=args.min_tokens,
        top_experts=args.top_experts,
        max_layers=args.max_layers,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "headline": doc["headline"],
                "per_layer_summary": [
                    {"layer": r["layer"], **r["headline"]} for r in doc["per_layer"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
