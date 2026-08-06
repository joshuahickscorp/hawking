#!/usr/bin/env python3.12
"""Functional-transfer program seal: 7-layer stack, A–G ablation, built vs gated.

Does not declare PROTO_FRANKENSTEIN from projected weights.  Seals the program
descriptor and can emit the companion JSON under evidence/.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators.frankenstein_ablation import (
    FUNCTIONAL_TRANSFER_ARMS,
    functional_transfer_arm_catalog,
    run_ag_ablation,
)
from lab.operators.frankenstein_baseline_freeze import freeze_base_dsv4f_baseline
from lab.operators.frankenstein_bridges import (
    LOSS_DEFINITIONS,
    build_adapter_bank,
    trainer_interface_spec,
)
from lab.operators.frankenstein_gates import (
    LINEAR_SUBSPACE_INITIALIZATION,
    REQUIRES_BENCHMARK_CORPUS,
    REQUIRES_GLM_RUNTIME,
    REQUIRES_TRAINING_LOOP,
    REQUIRES_VERIFIER,
    inventory_built_vs_gated,
    linear_init_claim_boundary,
)
from lab.operators.frankenstein_promotion_gate import (
    evaluate_promotion,
    frozen_targets,
    secondary_non_regression_suite_framework,
)
from lab.operators.frankenstein_verifier_loop import loop_interface_spec
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = (
    REPO_ROOT / "workspace" / "campaign" / "evidence" / "models" / "frankenstein"
)
DEFAULT_SEAL_PATH = EVIDENCE_ROOT / "FUNCTIONAL_TRANSFER_PROGRAM.json"
PROGRAM_SCHEMA = "hawking.frankenstein.functional_transfer_program.v1"


class FunctionalTransferError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    raw = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    encoded = raw.encode("utf-8")
    _ensure_dir(path.parent)
    if path.exists():
        node = os.lstat(path)
        if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
            raise FunctionalTransferError(f"not a regular file: {path}")
        if path.read_bytes() != encoded:
            raise FunctionalTransferError(
                f"refusing to overwrite different immutable evidence: {path}"
            )
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def seven_layer_transfer() -> list[dict[str, Any]]:
    """The functional transfer stack (program layers, not model layers)."""

    return [
        {
            "layer": 1,
            "name": "BASE_DSV4F_FREEZE",
            "description": "Freeze capability/runtime/routing/HCLI baseline",
            "status": "BUILT",
            "gates": [],
        },
        {
            "layer": 2,
            "name": "PAIRED_EVIDENCE",
            "description": (
                "Paired GLM/DSV4F traces on disjoint train/calib/public/hidden"
            ),
            "status": "FORMAT_BUILT_CAPTURE_GATED",
            "gates": [REQUIRES_GLM_RUNTIME, REQUIRES_BENCHMARK_CORPUS],
        },
        {
            "layer": 3,
            "name": "TOKENIZER_INDEPENDENT_ALIGNMENT",
            "description": "Align spans/bytes/actions/tools — never token IDs",
            "status": "BUILT",
            "gates": [],
        },
        {
            "layer": 4,
            "name": "LAYER_CARTOGRAPHY",
            "description": "CKA/CCA/Procrustes/causal-trace functional phase map",
            "status": "FRAMEWORK_BUILT_LIVE_GATED",
            "gates": [REQUIRES_GLM_RUNTIME],
        },
        {
            "layer": 5,
            "name": "REVERSIBLE_NONLINEAR_BRIDGES",
            "description": "norm→proj→gatedMLP→low-rank residual; train gated",
            "status": "ARCHITECTURE_BUILT_TRAIN_GATED",
            "gates": [REQUIRES_TRAINING_LOOP],
        },
        {
            "layer": 6,
            "name": "DISTILLED_ADAPTERS",
            "description": (
                "METHOD/DECOMPOSITION/FORMALIZATION/REPAIR + VALUE_HEAD + route bias"
            ),
            "status": "ARCHITECTURE_BUILT_TRAIN_GATED",
            "gates": [REQUIRES_TRAINING_LOOP],
        },
        {
            "layer": 7,
            "name": "VERIFIED_EXPERT_ITERATION_AND_PROMOTION",
            "description": (
                "Verifier loop + A–G ablation + frozen promotion gate"
            ),
            "status": "HARNESS_BUILT_RUNTIME_GATED",
            "gates": [REQUIRES_VERIFIER, REQUIRES_TRAINING_LOOP, REQUIRES_BENCHMARK_CORPUS],
        },
    ]


def build_program_document() -> dict[str, Any]:
    """Assemble the sealed functional-transfer program descriptor."""

    baseline = freeze_base_dsv4f_baseline(write=False)
    ag = run_ag_ablation(arm_scores=None)
    promotion = evaluate_promotion()
    secondary = secondary_non_regression_suite_framework()
    adapters = build_adapter_bank(d_model=4096, rank=4, seed=0)
    # Drop live modules from sealed JSON.
    adapters_public = {k: v for k, v in adapters.items() if not k.startswith("_")}
    trainer = trainer_interface_spec()
    verifier = loop_interface_spec()
    inventory = inventory_built_vs_gated()

    document = {
        "schema": PROGRAM_SCHEMA,
        "name": "FUNCTIONAL_TRANSFER_PROGRAM",
        "recorded_at": _utc_now(),
        "status": "SCAFFOLD_SEALED",
        "directive": (
            "The hardened linear GLM→DSV4F mapping is infrastructure and "
            "initialization, not sufficient inheritance. Do not declare "
            "PROTO_FRANKENSTEIN from projected weights."
        ),
        "linear_mapping": {
            "role": LINEAR_SUBSPACE_INITIALIZATION,
            "claim_boundary": linear_init_claim_boundary(),
            "use_for": [
                "layer_cartography_init",
                "bridge_weight_initialization",
            ],
            "not_for": [
                "PROTO_FRANKENSTEIN_COMPLETE",
                "math_capability_claim",
                "promotion",
            ],
        },
        "seven_layer_transfer": seven_layer_transfer(),
        "ablation_ag": {
            "arms": [
                {"id": a, "description": d} for a, d in FUNCTIONAL_TRANSFER_ARMS
            ],
            "catalog": functional_transfer_arm_catalog(),
            "harness_status": ag.get("status"),
            "harness_verdict": ag.get("verdict"),
            "harness_seal_sha256": ag.get("seal_sha256"),
        },
        "promotion_gate": {
            "targets": frozen_targets(),
            "current_verdict": promotion.get("verdict"),
            "reason": promotion.get("reason"),
            "seal_sha256": promotion.get("seal_sha256"),
        },
        "secondary_non_regression": {
            "status": secondary.get("status"),
            "seal_sha256": secondary.get("seal_sha256"),
        },
        "baseline_freeze": {
            "schema": baseline.get("schema"),
            "seal_sha256": baseline.get("seal_sha256"),
            "receipts_bound": baseline.get("runtime", {}).get("receipts_bound"),
        },
        "adapter_bank": {
            "schema": adapters_public.get("schema"),
            "seal_sha256": adapters_public.get("seal_sha256"),
            "total_parameter_bytes": adapters_public.get("total_parameter_bytes"),
            "trained": False,
            "glm_router_weights_copied": False,
        },
        "losses": dict(LOSS_DEFINITIONS),
        "trainer_interface": {
            "status": trainer.get("status"),
            "seal_sha256": trainer.get("seal_sha256"),
        },
        "verifier_loop": {
            "status": verifier.get("status"),
            "seal_sha256": verifier.get("seal_sha256"),
        },
        "built_vs_gated": inventory,
        "missing_infra": inventory["missing_infra"],
        "proto_frankenstein_complete": False,
        "fabricated_training_or_eval": False,
        "claim_boundary": {
            **linear_init_claim_boundary(),
            "scaffold_only": True,
            "promotion_pending": True,
        },
    }
    return seal(document)


def seal_program(*, out_path: Path | None = None, write: bool = True) -> dict[str, Any]:
    doc = build_program_document()
    verify(doc, label="functional transfer program")
    if write:
        path = Path(out_path) if out_path is not None else DEFAULT_SEAL_PATH
        _atomic_write_json(path, doc)
        doc = dict(doc)
        doc["_written_path"] = str(path)
    return doc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Seal FUNCTIONAL_TRANSFER_PROGRAM")
    p.add_argument("--out", type=Path, default=DEFAULT_SEAL_PATH)
    p.add_argument("--no-write", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    doc = seal_program(out_path=args.out, write=not args.no_write)
    print(
        json.dumps(
            {
                "status": doc["status"],
                "seal_sha256": doc["seal_sha256"],
                "proto_frankenstein_complete": doc["proto_frankenstein_complete"],
                "linear_role": doc["linear_mapping"]["role"],
                "promotion_verdict": doc["promotion_gate"]["current_verdict"],
                "written": doc.get("_written_path"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
