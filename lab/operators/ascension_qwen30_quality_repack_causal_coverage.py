"""Fail-closed CPU-only causal-coverage check for the Qwen30 HQ30GR2 branch.

The isolated sparse-residual candidate may improve a selected packed organ,
but that is not evidence that it can repair prompt-level coherence globally.
This small lane binds the candidate to the admitted control, proves the exact
mutation locus, and checks representative early/middle/late depth bands.  It
does not execute a model, use Metal, promote a candidate, or infer that an
early residual cannot indirectly influence later hidden states.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/complete-gravity"
    / "QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
)
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates"
    / "gate-up-residual-v1"
)
DEFAULT_CANDIDATE = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
DEFAULT_SELECTION = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_SELECTION_RECEIPT.json"
DEFAULT_SCALAR_CURRENT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CPU_SCALAR_PARITY_CURRENT.json"
DEFAULT_PACKED_CURRENT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CPU_PACKED_MATVEC_PARITY_CURRENT.json"
DEFAULT_HCLI_NEGATIVE = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/tps-gate/negative-science"
    / "QWEN30_HCLI_COHERENCE_6959979797825d3cedf17b073e7c7a6071b23292c6b490c439daa41a0afda79e.json"
)

SCHEMA = "hawking.ascension.qwen30_quality_repack_causal_coverage.v1"
STATUS = "INSUFFICIENT_EARLY_MIDDLE_LATE_DIRECT_COVERAGE_FOR_GLOBAL_COHERENCE_PROMOTION"
SELECTED_ORGANS = (
    "model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.layers.0.mlp.experts.0.up_proj.weight",
)
DEPTH_BANDS = {"early": 0, "middle": 24, "late": 47}


class CoverageError(RuntimeError):
    """A malformed or changed candidate is a fail-closed condition."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sealed(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        value = verify(raw, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise CoverageError(f"{label} is absent or invalid: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CoverageError(f"{label} is not an object")
    return dict(value)


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CoverageError(f"{label} must be a non-empty string")
    return value


def _tensor_map(document: Mapping[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    tensors = document.get("tensors")
    if not isinstance(tensors, list):
        raise CoverageError(f"{label} has no tensor list")
    result: dict[str, dict[str, Any]] = {}
    for row in tensors:
        if not isinstance(row, Mapping):
            raise CoverageError(f"{label} contains a non-object tensor")
        name = _require_text(row.get("tensor_name"), f"{label} tensor name")
        if name in result:
            raise CoverageError(f"{label} duplicates tensor {name}")
        result[name] = dict(row)
    if not result:
        raise CoverageError(f"{label} has an empty tensor list")
    return result


def _layer_prefix(layer: int) -> str:
    return f"model.layers.{layer}."


def analyze_mutation_coverage(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    selected_organs: Sequence[str],
) -> dict[str, Any]:
    """Prove the direct mutation locus without overclaiming indirect effects."""

    baseline_tensors = _tensor_map(baseline, label="baseline manifest")
    candidate_tensors = _tensor_map(candidate, label="candidate manifest")
    if set(baseline_tensors) != set(candidate_tensors):
        raise CoverageError("candidate and baseline tensor catalogues differ")
    expected = tuple(SELECTED_ORGANS)
    if tuple(selected_organs) != expected:
        raise CoverageError("candidate selected-organ authority is not the exact HQ30GR2 pair")

    changed: list[str] = []
    for name, candidate_row in candidate_tensors.items():
        mutation = candidate_row.get("candidate_mutation")
        if not isinstance(mutation, Mapping):
            raise CoverageError(f"candidate tensor {name} has no mutation record")
        changed_from_control = mutation.get("changed_from_admitted_control")
        if not isinstance(changed_from_control, bool):
            raise CoverageError(f"candidate tensor {name} has invalid mutation state")
        baseline_sha = _require_text(baseline_tensors[name].get("artifact_sha256"), f"baseline {name} hash")
        candidate_sha = _require_text(candidate_row.get("artifact_sha256"), f"candidate {name} hash")
        if changed_from_control:
            changed.append(name)
            if candidate_sha == baseline_sha:
                raise CoverageError(f"candidate marks {name} changed but retains its control payload hash")
        elif candidate_sha != baseline_sha:
            raise CoverageError(f"candidate changes unselected control payload {name}")
    if tuple(sorted(changed)) != tuple(sorted(expected)):
        raise CoverageError("candidate changed organs do not equal the sealed HQ30GR2 pair")

    bands: dict[str, dict[str, Any]] = {}
    for band, layer in DEPTH_BANDS.items():
        prefix = _layer_prefix(layer)
        names = sorted(name for name in candidate_tensors if name.startswith(prefix))
        if not names:
            raise CoverageError(f"candidate has no tensors for representative {band} layer {layer}")
        direct_changes = [name for name in names if name in changed]
        payloads_match_control = all(
            candidate_tensors[name]["artifact_sha256"] == baseline_tensors[name]["artifact_sha256"]
            for name in names
        )
        bands[band] = {
            "layer": layer,
            "tensor_count": len(names),
            "directly_changed_organs": direct_changes,
            "all_layer_payload_hashes_match_admitted_control": payloads_match_control,
        }

    return {
        "changed_organs": sorted(changed),
        "changed_layer_indices": sorted(
            {
                int(name.split(".")[2])
                for name in changed
                if len(name.split(".")) > 2 and name.split(".")[2].isdigit()
            }
        ),
        "depth_bands": bands,
        "direct_mutation_covers_all_representative_depths": all(
            bool(row["directly_changed_organs"]) for row in bands.values()
        ),
        "indirect_later_hidden_state_propagation": "UNMEASURED_NOT_INFERRED",
    }


def _validate_current_parity(pointer_path: Path, *, expected_status: str, label: str) -> dict[str, Any]:
    pointer = _sealed(pointer_path, label=label)
    if pointer.get("status") != expected_status:
        raise CoverageError(f"{label} has an unexpected status")
    receipt_key = "scalar_parity_receipt" if "SCALAR" in expected_status else "packed_matvec_parity_receipt"
    receipt = pointer.get(receipt_key)
    if not isinstance(receipt, Mapping):
        raise CoverageError(f"{label} has no selected parity receipt")
    receipt_path = Path(_require_text(receipt.get("path"), f"{label} receipt path"))
    checked = _sealed(receipt_path, label=f"{label} selected receipt")
    if checked.get("seal_sha256") != receipt.get("seal_sha256"):
        raise CoverageError(f"{label} selected receipt seal no longer matches its pointer")
    return {"pointer_path": str(pointer_path.resolve()), "pointer_seal_sha256": pointer.get("seal_sha256"), "receipt_path": str(receipt_path.resolve()), "receipt_seal_sha256": checked.get("seal_sha256")}


def run_once(
    *,
    root: Path,
    baseline_path: Path,
    candidate_path: Path,
    selection_path: Path,
    scalar_current_path: Path,
    packed_current_path: Path,
    hcli_negative_path: Path,
) -> dict[str, Any]:
    """Seal one candidate-local structural result, or reuse its exact twin."""

    baseline = _sealed(baseline_path, label="admitted baseline manifest")
    candidate = _sealed(candidate_path, label="isolated HQ30GR2 candidate manifest")
    if baseline.get("seal_sha256") != "3321a99d719e70499663b7bfebe14dd6c732bfc533bb05b9277eb398e44d6357":
        raise CoverageError("baseline manifest is not the preserved admitted Qwen30 control")
    if candidate.get("schema") != "hawking.ascension.qwen30_quality_repack_candidate.v1":
        raise CoverageError("candidate does not use the isolated quality-repack manifest grammar")
    selection = _sealed(selection_path, label="HQ30GR2 selection receipt")
    selected = selection.get("binding", {}).get("selected_organs") if isinstance(selection.get("binding"), Mapping) else None
    if not isinstance(selected, list):
        raise CoverageError("selection receipt has no selected organs")
    coverage = analyze_mutation_coverage(baseline, candidate, selected_organs=selected)
    scalar = _validate_current_parity(
        scalar_current_path,
        expected_status="CURRENT_QWEN30_QUALITY_REPACK_CPU_SCALAR_PARITY_RECEIPT_SELECTED",
        label="selected CPU scalar parity",
    )
    packed = _validate_current_parity(
        packed_current_path,
        expected_status="CURRENT_QWEN30_QUALITY_REPACK_CPU_PACKED_MATVEC_PARITY_RECEIPT_SELECTED",
        label="selected CPU packed matvec parity",
    )
    negative = _sealed(hcli_negative_path, label="current Qwen30 HCLI coherence negative")
    if negative.get("status") != "BLOCKED_HCLI_PROMPT_DEPENDENT_COHERENCE_NOT_EARNED":
        raise CoverageError("target HCLI receipt is not the sealed coherence negative")

    binding = {
        "baseline_manifest_path": str(baseline_path.resolve()),
        "baseline_manifest_seal_sha256": baseline.get("seal_sha256"),
        "candidate_manifest_path": str(candidate_path.resolve()),
        "candidate_manifest_seal_sha256": candidate.get("seal_sha256"),
        "selection_receipt_path": str(selection_path.resolve()),
        "selection_receipt_seal_sha256": selection.get("seal_sha256"),
        "hcli_coherence_negative_path": str(hcli_negative_path.resolve()),
        "hcli_coherence_negative_seal_sha256": negative.get("seal_sha256"),
        "scalar_parity": scalar,
        "packed_matvec_parity": packed,
    }
    receipt_root = root / "causal-coverage" / "receipts"
    receipt_path = receipt_root / f"QWEN30_HQ30GR2_CAUSAL_COVERAGE_{candidate['seal_sha256']}.json"
    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "recorded_at": _utc_now(),
        "binding": binding,
        "cpu_only_discriminators": {
            "selected_organ_scalar_parity": "EARNED_AND_CURRENT",
            "selected_organ_direct_packed_matvec_parity": "EARNED_AND_CURRENT",
            "early_middle_late_direct_mutation_coverage": coverage,
        },
        "assessment": {
            "candidate_can_be_proven_to_change_only": coverage["changed_organs"],
            "direct_middle_and_late_coverage": "ABSENT",
            "plausible_global_coherence_repair": "NOT_EARNED",
            "reason": "the candidate changes only the layer-0/expert-0 gate/up pair; source-bound full-prompt route membership and all-layer activation propagation were not measured",
            "next_cheapest_discriminator": "capture source-bound route membership for the three failed HCLI prompts at layer 0, then reject immediately if expert 0 is absent from any prompt; only a route-covered candidate may proceed to a bounded all-layer candidate runtime comparison",
        },
        "claim_boundary": {
            "cpu_only_manifest_and_packed_operator_evidence": True,
            "does_not_execute_candidate_metal_or_full_model_runtime": True,
            "does_not_claim_indirect_hidden_state_nonimpact": True,
            "does_not_claim_candidate_generation_hcli_tps_tg_capability_or_tournament_qualification": True,
            "baseline_runtime_server_and_current_pointers_untouched": True,
        },
    }
    sealed = seal(body)
    if receipt_path.exists():
        existing = _sealed(receipt_path, label="existing causal coverage receipt")
        if existing.get("binding") != sealed.get("binding") or existing.get("status") != STATUS:
            raise CoverageError("refusing to overwrite a historical causal coverage receipt")
        result = existing
        reused = True
    else:
        _atomic_json(receipt_path, sealed)
        result = sealed
        reused = False
    pointer = seal(
        {
            "schema": "hawking.ascension.qwen30_quality_repack_causal_coverage_current.v1",
            "status": "CURRENT_QWEN30_HQ30GR2_CAUSAL_COVERAGE_RECEIPT_SELECTED",
            "recorded_at": _utc_now(),
            "candidate_root": str(root.resolve()),
            "coverage_receipt": {
                "path": str(receipt_path.resolve()),
                "seal_sha256": result.get("seal_sha256"),
            },
            "isolation": body["claim_boundary"],
        }
    )
    _atomic_json(root / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CAUSAL_COVERAGE_CURRENT.json", pointer)
    return {
        "status": result.get("status"),
        "receipt_path": str(receipt_path),
        "receipt_seal_sha256": result.get("seal_sha256"),
        "current_path": str(root / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CAUSAL_COVERAGE_CURRENT.json"),
        "current_seal_sha256": pointer.get("seal_sha256"),
        "reused": reused,
        "coverage": coverage,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--scalar-current", type=Path, default=DEFAULT_SCALAR_CURRENT)
    parser.add_argument("--packed-current", type=Path, default=DEFAULT_PACKED_CURRENT)
    parser.add_argument("--hcli-negative", type=Path, default=DEFAULT_HCLI_NEGATIVE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_once(
            root=args.root.expanduser().resolve(),
            baseline_path=args.baseline.expanduser().resolve(),
            candidate_path=args.candidate.expanduser().resolve(),
            selection_path=args.selection.expanduser().resolve(),
            scalar_current_path=args.scalar_current.expanduser().resolve(),
            packed_current_path=args.packed_current.expanduser().resolve(),
            hcli_negative_path=args.hcli_negative.expanduser().resolve(),
        )
    except CoverageError as exc:
        print(json.dumps({"status": "BLOCKED_QWEN30_HQ30GR2_CAUSAL_COVERAGE_FAIL_CLOSED", "detail": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
