"""Kimi operator candidates, built on Tabula's existing write owner.

This module is deliberately an in-memory candidate generator. It does not load
Kimi, write ModelLake, or promote a child. ``tools.future.tabula`` remains the
owner of the projection law and authority boundary; this file only supplies the
Kimi-specific candidate matrix and provenance envelope needed by a later live
WorkUnit.

The seven named methods are explicit because a later result must say exactly
what changed:

  OPA conventional rank-1, OPB norm-preserving rank-1, OPC two-sided,
  OPD low-rank subspace, OPE layer-local rank-1, OPF selected-expert, and
  OPG shared-plus-selected-expert.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.future import negative_index, tabula

SCHEMA = "hawking.future.kimi_operator_candidate.v1"
METHODS = ("OPA", "OPB", "OPC", "OPD", "OPE", "OPF", "OPG")
REFUSAL_INTERVENTION_FAMILY = "kimi refusal intervention family"
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
EXPERT_RE = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")


class CandidateContractError(ValueError):
    """A candidate cannot be applied without an explicit declared input."""


def refusal_intervention_preflight(
    method: str,
    *,
    hypothesis_family: str = REFUSAL_INTERVENTION_FAMILY,
    reproduce_scarred_family: bool = False,
) -> dict[str, Any]:
    """Screen a KIMI behavior intervention before model or accelerator work.

    The OPA--OPH family is closed negative science.  Keeping this boundary in
    the candidate owner makes the live residual/writer path and the separate
    router path consult the same Scar.  An explicit reproduction remains
    possible for audit, but is labelled reproduction-only and cannot be
    interpreted as a new promotion candidate.
    """
    proposal = {
        "model": "KIMI_BASE",
        "organ": "whole_model",
        "hypothesis_family": hypothesis_family,
    }
    scar = negative_index.refuse_if_dead(proposal)
    if scar is None:
        return {
            "allowed": True,
            "status": "HYPOTHESIS_OPEN",
            "method": str(method),
            "proposal": proposal,
            "scar": None,
            "reproduction_only": False,
        }
    if reproduce_scarred_family:
        return {
            "allowed": True,
            "status": "SCARRED_FAMILY_REPRODUCTION_ONLY",
            "method": str(method),
            "proposal": proposal,
            "scar": scar,
            "reproduction_only": True,
            "claim_boundary": (
                "Explicit reproduction of closed negative science only; this "
                "run cannot reopen, rename, promote, or extend OPA--OPH."
            ),
        }
    return {
        "allowed": False,
        "status": "NEGATIVE_SCIENCE_REFUSED",
        "method": str(method),
        "proposal": proposal,
        "scar": scar,
        "reproduction_only": False,
        "reopen_condition": scar.get("reopen_condition"),
    }


@dataclass(frozen=True)
class OperatorCandidate:
    candidate_id: str
    method: str
    source_model: str = "KIMI_BASE"
    target_layers: tuple[int, ...] = ()
    selected_experts: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    strength: float = 1.0
    norm_preserve: bool = False
    reversible: bool = True
    calibration_data_class: str = "heldout_refused_vs_complied"
    subspace_rank: int = 1
    requires_input_direction: bool = False

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise CandidateContractError(f"unknown operator method {self.method!r}")
        if not self.candidate_id.startswith("KIMI_OPERATOR_CANDIDATE_"):
            raise CandidateContractError("candidate id must be KIMI_OPERATOR_CANDIDATE_*")
        if self.source_model != "KIMI_BASE":
            raise CandidateContractError("candidate source must remain immutable KIMI_BASE")
        if not 0.0 <= float(self.strength) <= 1.0:
            raise CandidateContractError("candidate strength must be between 0 and 1")
        if int(self.subspace_rank) < 1:
            raise CandidateContractError("subspace_rank must be positive")
        if self.method == "OPC" and not self.requires_input_direction:
            raise CandidateContractError("OPC must declare an input direction")
        if self.method == "OPD" and int(self.subspace_rank) < 2:
            raise CandidateContractError("OPD is the low-rank candidate and needs rank >= 2")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "candidate_id": self.candidate_id,
            "method": self.method,
            "source_model": self.source_model,
            "target_layers": list(self.target_layers),
            "selected_experts": {
                str(k): list(v) for k, v in sorted(self.selected_experts.items())
            },
            "strength": float(self.strength),
            "norm_preserve": bool(self.norm_preserve),
            "reversible": bool(self.reversible),
            "calibration_data_class": self.calibration_data_class,
            "subspace_rank": int(self.subspace_rank),
            "requires_input_direction": bool(self.requires_input_direction),
            "artifact_hash": None,
            "nova_lineage_receipt": None,
            "claim_boundary": (
                "In-memory candidate transformation only; no live capability, "
                "refusal, hardware, or promotion claim."
            ),
        }


def catalog(
    *,
    n_layers: int,
    target_layers: Sequence[int] | None = None,
    selected_experts: Mapping[int, Sequence[int]] | None = None,
    subspace_rank: int = 4,
    calibration_data_class: str = "heldout_refused_vs_complied",
) -> tuple[OperatorCandidate, ...]:
    """Return the pre-registered candidate matrix with no measured ranking."""
    if int(n_layers) <= 0:
        raise CandidateContractError("n_layers must be positive")
    layers = tuple(range(int(n_layers))) if target_layers is None else tuple(
        sorted({int(x) for x in target_layers}))
    if not layers or any(layer < 0 or layer >= int(n_layers) for layer in layers):
        raise CandidateContractError("target_layers must be within the model")
    experts = {
        int(layer): tuple(sorted({int(x) for x in values}))
        for layer, values in (selected_experts or {}).items()
    }
    return (
        OperatorCandidate(
            "KIMI_OPERATOR_CANDIDATE_OPA", "OPA", target_layers=layers,
            calibration_data_class=calibration_data_class),
        OperatorCandidate(
            "KIMI_OPERATOR_CANDIDATE_OPB", "OPB", target_layers=layers,
            norm_preserve=True, calibration_data_class=calibration_data_class),
        OperatorCandidate(
            "KIMI_OPERATOR_CANDIDATE_OPC", "OPC", target_layers=layers,
            norm_preserve=True, requires_input_direction=True,
            reversible=False,
            calibration_data_class=calibration_data_class),
        OperatorCandidate(
            "KIMI_OPERATOR_CANDIDATE_OPD", "OPD", target_layers=layers,
            norm_preserve=True, subspace_rank=int(subspace_rank),
            calibration_data_class=calibration_data_class),
        OperatorCandidate(
            "KIMI_OPERATOR_CANDIDATE_OPE", "OPE", target_layers=(layers[-1],),
            calibration_data_class=calibration_data_class),
        OperatorCandidate(
            "KIMI_OPERATOR_CANDIDATE_OPF", "OPF", target_layers=layers,
            selected_experts=experts, calibration_data_class=calibration_data_class),
        OperatorCandidate(
            "KIMI_OPERATOR_CANDIDATE_OPG", "OPG", target_layers=layers,
            selected_experts=experts, norm_preserve=True,
            calibration_data_class=calibration_data_class),
    )


def _layer_for_key(key: str) -> int | None:
    match = LAYER_RE.search(key)
    return int(match.group(1)) if match else None


def _expert_for_key(key: str) -> int | None:
    match = EXPERT_RE.search(key)
    return int(match.group(1)) if match else None


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _state_hash(weights: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(weights):
        value = np.asarray(weights[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm == 0.0:
        raise CandidateContractError("candidate direction has zero norm")
    return value / norm


def _biproject(
    W: np.ndarray,
    output_direction: np.ndarray,
    input_direction: np.ndarray,
    *,
    strength: float,
    norm_preserve: bool,
) -> np.ndarray:
    """Two-sided projection for OPC: P_out W P_in."""
    W = np.asarray(W, dtype=np.float64)
    out = _unit(output_direction)
    inp = _unit(input_direction)
    if W.ndim != 2 or W.shape[0] != out.size or W.shape[1] != inp.size:
        raise CandidateContractError(
            f"OPC directions do not match weight shape {W.shape}: "
            f"out={out.size}, in={inp.size}")
    parent_norm = float(np.linalg.norm(W, ord="fro"))
    result = W - float(strength) * np.outer(out, out @ W)
    result = result - float(strength) * np.outer(result @ inp, inp)
    if norm_preserve:
        result_norm = float(np.linalg.norm(result, ord="fro"))
        if result_norm > 0:
            result = result * (parent_norm / result_norm)
    return result


def _transform(
    W: np.ndarray,
    candidate: OperatorCandidate,
    direction: np.ndarray,
    input_direction: np.ndarray | None,
    subspace: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    original_dtype = np.asarray(W).dtype
    if candidate.method == "OPC":
        if input_direction is None:
            raise CandidateContractError("OPC requires input_directions for every touched layer")
        result = _biproject(
            W, direction, input_direction, strength=candidate.strength,
            norm_preserve=candidate.norm_preserve)
        metrics = {"method": "two_sided_projection", "reversible": False}
    elif candidate.method == "OPD":
        if subspace is None:
            raise CandidateContractError("OPD requires a subspace for every touched layer")
        if tabula.orthonormal_basis(subspace).shape[1] < candidate.subspace_rank:
            raise CandidateContractError(
                f"OPD subspace rank is below declared rank {candidate.subspace_rank}")
        result, recipe, metrics = tabula.project_subspace(
            W, subspace, norm_preserve=candidate.norm_preserve,
            store_component=candidate.reversible, scale=candidate.strength)
        metrics = {**metrics, "recipe": recipe.to_dict() if recipe else None}
    else:
        result, recipe, metrics = tabula.project(
            W, direction, norm_preserve=candidate.norm_preserve,
            store_component=candidate.reversible, scale=candidate.strength)
        metrics = {**metrics, "recipe": recipe.to_dict() if recipe else None}
    if np.issubdtype(original_dtype, np.floating):
        result = result.astype(original_dtype, copy=False)
    return result, metrics


def apply_candidate(
    weights: Mapping[str, np.ndarray],
    candidate: OperatorCandidate,
    directions: Mapping[int, np.ndarray],
    *,
    input_directions: Mapping[int, np.ndarray] | None = None,
    subspaces: Mapping[int, np.ndarray] | None = None,
    code_commit: str = "UNRECORDED",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply a candidate to a copied state mapping and emit provenance.

    The input mapping is never mutated. Non-target tensors are copied byte for
    byte. Missing causal directions are a contract error rather than a silent
    skip, and expert candidates require an explicit selected-expert map.
    """
    if not weights:
        raise CandidateContractError("KIMI_BASE weight mapping is empty")
    if not directions:
        raise CandidateContractError("candidate directions are required")
    if candidate.method in {"OPF", "OPG"} and not any(candidate.selected_experts.values()):
        raise CandidateContractError(
            f"{candidate.method} requires an explicit selected-expert scope from causal evidence")
    before = {key: np.array(value, copy=True) for key, value in weights.items()}
    after = {key: np.array(value, copy=True) for key, value in weights.items()}
    selected_layers = set(candidate.target_layers)
    touched: list[dict[str, Any]] = []
    for key in sorted(before):
        layer = _layer_for_key(key)
        if layer is None or layer not in selected_layers:
            continue
        expert = _expert_for_key(key)
        if candidate.method == "OPF":
            allowed = set(candidate.selected_experts.get(layer, ()))
            if expert is None or expert not in allowed:
                continue
        elif candidate.method == "OPG":
            allowed = set(candidate.selected_experts.get(layer, ()))
            is_shared = ".shared_experts." in key
            if not is_shared and (expert is None or expert not in allowed):
                continue
        elif candidate.method in {"OPA", "OPB", "OPC", "OPD", "OPE"} and expert is not None:
            # The generic layer candidates operate on shared/residual writers;
            # OPF/OPG are the only candidates allowed to touch routed experts.
            continue
        direction = directions.get(layer)
        if direction is None:
            raise CandidateContractError(f"missing output direction for layer {layer}")
        transformed, metrics = _transform(
            before[key], candidate, direction,
            (input_directions or {}).get(layer),
            (subspaces or {}).get(layer),
        )
        after[key] = transformed
        touched.append({
            "key": key,
            "layer": layer,
            "expert": expert,
            "before_sha256": _array_hash(before[key]),
            "after_sha256": _array_hash(after[key]),
            "metrics": metrics,
        })
    if not touched:
        raise CandidateContractError(
            f"{candidate.candidate_id} touched no tensors; check layer/organ/expert scope")
    receipt = {
        "schema": SCHEMA,
        "candidate": candidate.to_dict(),
        "source_model": "KIMI_BASE",
        "source_state_sha256": _state_hash(before),
        "candidate_state_sha256": _state_hash(after),
        "base_artifact_hash": _state_hash(before),
        "artifact_hash": _state_hash(after),
        "code_commit": code_commit,
        "touched_tensors": touched,
        "touched_layers": sorted({row["layer"] for row in touched}),
        "touched_experts": sorted({row["expert"] for row in touched if row["expert"] is not None}),
        "weights_written": False,
        "promotion": "NOT_PERFORMED",
        "evidence_class": "IN_MEMORY_TRANSFORM_ONLY",
        "learned_organism_change": True,
        "nova_lineage_required_for_promotion": True,
        "nova_lineage_receipt": None,
        "claim_boundary": (
            "This receipt proves only that a declared transformation was applied "
            "to a copied KIMI_BASE mapping. It does not measure behavior or capability."
        ),
    }
    return after, receipt


def _selftest() -> int:
    rng = np.random.default_rng(0)
    weights = {
        "model.layers.0.mlp.down_proj.weight": rng.normal(size=(8, 6)).astype(np.float32),
        "model.layers.0.mlp.shared_experts.down_proj.weight": rng.normal(size=(8, 6)).astype(np.float32),
        "model.layers.0.mlp.experts.2.down_proj.weight": rng.normal(size=(8, 6)).astype(np.float32),
        "model.layers.1.mlp.down_proj.weight": rng.normal(size=(8, 6)).astype(np.float32),
    }
    original = {key: value.copy() for key, value in weights.items()}
    direction = {0: rng.normal(size=8), 1: rng.normal(size=8)}
    input_direction = {0: rng.normal(size=6), 1: rng.normal(size=6)}
    subspace = {0: np.column_stack((direction[0], rng.normal(size=8), rng.normal(size=8), rng.normal(size=8))),
                1: np.column_stack((direction[1], rng.normal(size=8), rng.normal(size=8), rng.normal(size=8)))}
    candidates = catalog(n_layers=2, target_layers=(0, 1), selected_experts={0: (2,)})
    for candidate in candidates:
        _, receipt = apply_candidate(
            weights, candidate, direction,
            input_directions=input_direction,
            subspaces=subspace,
            code_commit="selftest",
        )
        assert receipt["source_model"] == "KIMI_BASE"
        assert receipt["weights_written"] is False
    for key in weights:
        assert np.array_equal(weights[key], original[key]), "KIMI_BASE mapping was mutated"
    try:
        apply_candidate(weights, candidates[2], direction, code_commit="selftest")
    except CandidateContractError as exc:
        assert "OPC requires" in str(exc)
    else:
        raise AssertionError("OPC accepted missing input direction")
    print(f"selftest OK: {len(candidates)} candidates; base unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
