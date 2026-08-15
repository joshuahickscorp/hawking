#!/usr/bin/env python3
"""Build a sealed Q80 complete candidate using activation_weighted_svd_low_rank_q.

Mirrors ascension_qwen30_activation_weighted_svd_repack with Q80 constants.
**require_all_layer_capture defaults to True** so an L0-only or 3-layer pack
cannot pretend to be coherent (the Q30 coverage failure class).

Does not start a server or claim coherence. Surplus-over-null is the selection
metric; weight cosine is secondary/distribution-local only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen30_activation_weighted_svd_repack as q30  # noqa: E402
from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (  # noqa: E402
    ActivationWeightedRepackError,
    ActivationWeightedSvdRepack,
    capture_is_all_layer,
    coverage_report,
)

MAIN_HAWKING = Path("/Users/scammermike/Downloads/hawking")
MODEL_DIR = MAIN_HAWKING / (
    "workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next"
)
QWEN80_ROOT = MAIN_HAWKING / (
    "workspace/campaign/records/ascension-sandbox/physical/qwen80"
)
BASELINE_ROOT = QWEN80_ROOT / "complete-gravity"
SOURCE_AUDIT = MAIN_HAWKING / (
    "workspace/campaign/records/ascension-sandbox/physical/qwen80-acquisition"
    "/QWEN80_SOURCE_BODY_AUDIT_CANDIDATE.json"
)
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen80"
    / "quality-candidates/activation-weighted-svd-v1"
)

# Q80-specific constants (override Q30 class attributes via subclass).
SCHEMA = "hawking.ascension.qwen80_activation_weighted_svd_repack_candidate.v1"
SELECTION_SCHEMA = "hawking.ascension.qwen80_activation_weighted_svd_selection.v1"
SOURCE_SNAPSHOT_SCHEMA = "hawking.ascension.qwen80_activation_weighted_svd_source_snapshot.v1"
COVERAGE_SCHEMA = "hawking.ascension.qwen80_activation_weighted_svd_coverage.v1"
BRANCH_ID = "qwen80-activation-weighted-svd-v1"
ARTIFACT_PREFIX = "QWEN80_ACTIVATION_WEIGHTED_SVD_V1"
MODEL_ID = "Qwen3-Coder-Next-80B-activation-weighted-svd-v1"
EXPECTED_TENSOR_COUNT = 74_391
MODEL_LAYERS = 48
BASELINE_MANIFEST_NAME = "QWEN80_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
BASELINE_ADMISSION_NAME = "QWEN80_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json"
BASELINE_REVALIDATION_NAME = "QWEN80_CURRENT_SOURCE_SHARD_REVALIDATION.json"
ALL_LAYER_RESULT_SCHEMA = (
    "hawking.ascension.qwen80_broad_activation_all_layer_route_capture_result.v1"
)


class Qwen80ActivationWeightedSvdRepack(ActivationWeightedSvdRepack):
    """Q80 surplus-first complete candidate; all-layer capture required by default."""

    def __init__(
        self,
        *,
        model_dir: Path = MODEL_DIR,
        baseline_root: Path = BASELINE_ROOT,
        source_audit: Path = SOURCE_AUDIT,
        capture_run: Path | None = None,
        root: Path = DEFAULT_ROOT,
        max_experts: int | None = None,
        max_layers: int | None = None,
        workers: int = 4,
        require_all_layer_capture: bool = True,
    ) -> None:
        if capture_run is None:
            raise ActivationWeightedRepackError(
                "Q80 repack requires an explicit --capture-run binding a real "
                "all-layer activation capture; no default L0 capture exists"
            )
        super().__init__(
            model_dir=model_dir,
            baseline_root=baseline_root,
            source_audit=source_audit,
            capture_run=capture_run,
            root=root,
            max_experts=max_experts,
            max_layers=max_layers,
            workers=workers,
            require_all_layer_capture=require_all_layer_capture,
        )
        # Override Q30 hard-coded paths/names after super init.
        self.manifest_path = self.root / f"{ARTIFACT_PREFIX}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
        self.selection_path = self.root / f"{ARTIFACT_PREFIX}_SELECTION_RECEIPT.json"
        self.snapshot_path = self.root / f"{ARTIFACT_PREFIX}_SOURCE_BINDING_SNAPSHOT.json"
        self.terminal_path = self.root / f"{ARTIFACT_PREFIX}_COMPLETE_GRAVITY_TERMINAL_RECEIPT.json"
        self.status_path = self.root / f"{ARTIFACT_PREFIX}_COMPLETE_GRAVITY_STATUS.json"
        self.coverage_path = self.root / f"{ARTIFACT_PREFIX}_COVERAGE_RECEIPT.json"
        self.baseline_manifest_path = self.baseline_root / BASELINE_MANIFEST_NAME
        self.baseline_admission_path = self.baseline_root / BASELINE_ADMISSION_NAME
        self.baseline_revalidation_path = self.baseline_root / BASELINE_REVALIDATION_NAME
        self.baseline_tensor_dir = self.baseline_root / "tensors"

    def run(self) -> dict[str, Any]:
        # Preflight: refuse incomplete capture coverage before any fit work.
        capture_result = self.capture_run / "capture-result.json"
        if not capture_result.is_file():
            raise ActivationWeightedRepackError(
                f"capture-result.json missing at {self.capture_run}; "
                "all-layer activation capture is the open question for Q80 "
                "(see q80_activation_capture_readiness)"
            )
        capture = json.loads(capture_result.read_text(encoding="utf-8"))
        schema = str(capture.get("schema") or "")
        all_layer = (
            capture_is_all_layer(capture)
            or schema == ALL_LAYER_RESULT_SCHEMA
            or bool(capture.get("capture_summary", {}).get("all_layer_activation_capture"))
        )
        if self.require_all_layer_capture and not all_layer:
            raise ActivationWeightedRepackError(
                "Q80 activation-weighted repack requires an all-layer capture "
                f"(schema={schema!r}). Fitting a partial-layer capture would "
                "repeat the Q30 coverage failure (L0-only / 1.7% tensors)."
            )
        # Temporarily patch module-level constants the parent uses for coverage.
        old = {
            "SCHEMA": q30.SCHEMA,
            "SELECTION_SCHEMA": q30.SELECTION_SCHEMA,
            "SOURCE_SNAPSHOT_SCHEMA": q30.SOURCE_SNAPSHOT_SCHEMA,
            "BRANCH_ID": q30.BRANCH_ID,
            "ARTIFACT_PREFIX": q30.ARTIFACT_PREFIX,
            "MODEL_ID": q30.MODEL_ID,
            "EXPECTED_TENSOR_COUNT": q30.EXPECTED_TENSOR_COUNT,
            "MODEL_LAYERS": q30.MODEL_LAYERS,
            "BASELINE_MANIFEST_NAME": q30.BASELINE_MANIFEST_NAME,
            "BASELINE_ADMISSION_NAME": q30.BASELINE_ADMISSION_NAME,
            "BASELINE_REVALIDATION_NAME": q30.BASELINE_REVALIDATION_NAME,
        }
        q30.SCHEMA = SCHEMA
        q30.SELECTION_SCHEMA = SELECTION_SCHEMA
        q30.SOURCE_SNAPSHOT_SCHEMA = SOURCE_SNAPSHOT_SCHEMA
        q30.BRANCH_ID = BRANCH_ID
        q30.ARTIFACT_PREFIX = ARTIFACT_PREFIX
        q30.MODEL_ID = MODEL_ID
        q30.EXPECTED_TENSOR_COUNT = EXPECTED_TENSOR_COUNT
        q30.MODEL_LAYERS = MODEL_LAYERS
        q30.BASELINE_MANIFEST_NAME = BASELINE_MANIFEST_NAME
        q30.BASELINE_ADMISSION_NAME = BASELINE_ADMISSION_NAME
        q30.BASELINE_REVALIDATION_NAME = BASELINE_REVALIDATION_NAME
        try:
            result = super().run()
        finally:
            for key, value in old.items():
                setattr(q30, key, value)
        # Ensure coverage receipt uses Q80 schema if parent wrote Q30 schema.
        if self.coverage_path.is_file():
            cov = json.loads(self.coverage_path.read_text(encoding="utf-8"))
            if cov.get("schema") != COVERAGE_SCHEMA:
                cov["schema"] = COVERAGE_SCHEMA
                cov["model"] = "qwen80"
                self.coverage_path.write_text(
                    json.dumps(cov, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
        return result


def preflight_coverage_from_capture(capture_run: Path) -> dict[str, Any]:
    """Coverage estimate without packing — for readiness gates."""
    capture_result = capture_run / "capture-result.json"
    if not capture_result.is_file():
        return {
            "status": "CAPTURE_MISSING",
            "all_layer_capture": False,
            "layers_covered": [],
            "n_layers_covered": 0,
            "cannot_be_coherent": True,
            "reason": "no capture-result.json",
        }
    capture = json.loads(capture_result.read_text(encoding="utf-8"))
    all_layer = capture_is_all_layer(capture) or str(capture.get("schema") or "") == ALL_LAYER_RESULT_SCHEMA
    _by, prov = q30.collect_expert_activations(capture_run, capture)
    layers = list(prov.get("layers_with_hidden_hits") or [])
    # Fake empty selected organs to reuse coverage_report shape for capture-side.
    cov = coverage_report(
        selected_organs=[],
        act_prov=prov,
        model_layers=MODEL_LAYERS,
        total_tensors=EXPECTED_TENSOR_COUNT,
    )
    cov["status"] = "CAPTURE_ONLY_COVERAGE_PREFLIGHT"
    cov["all_layer_capture"] = all_layer
    cov["capture_layers_with_hidden_hits"] = layers
    cov["cannot_be_coherent"] = (not all_layer) or len(layers) < MODEL_LAYERS
    cov["require_all_layer_capture"] = True
    return cov


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--source-audit", type=Path, default=SOURCE_AUDIT)
    parser.add_argument("--capture-run", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-experts", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--allow-partial-layer-capture",
        action="store_true",
        help="DANGEROUS: disable all-layer requirement (for instrument tests only)",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only emit coverage preflight from capture; do not pack",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    capture_run = args.capture_run.expanduser().resolve()
    if args.preflight_only:
        cov = preflight_coverage_from_capture(capture_run)
        print(json.dumps(cov, indent=2, sort_keys=True))
        return 0 if not cov.get("cannot_be_coherent") else 2

    worker = Qwen80ActivationWeightedSvdRepack(
        model_dir=args.model_dir,
        baseline_root=args.baseline_root,
        source_audit=args.source_audit,
        capture_run=capture_run,
        root=args.root,
        max_experts=args.max_experts,
        max_layers=args.max_layers,
        workers=args.workers,
        require_all_layer_capture=not args.allow_partial_layer_capture,
    )
    try:
        result = worker.run()
    except ActivationWeightedRepackError as exc:
        print(json.dumps({"status": "REFUSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
