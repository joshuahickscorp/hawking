"""Forward-compatible Hawking nomenclature and semantic aliases.

This module is deliberately small: it gives active serializers one canonical
version and a machine-readable vocabulary without renaming sealed historical
paths, schemas, or receipts.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping


NOMENCLATURE_VERSION = "HAWKING_NOMENCLATURE_V2"

# Active artifact extensions. Gravity is a process, not a file format. The
# ``.gravity`` suffix remains readable for sealed historical work; new
# representation shards use NR and a machine-bound executable uses NX.
NR_EXTENSION = ".nr"
NX_EXTENSION = ".nx"

CANONICAL_PIPELINE = (
    "SourceSpecimen",
    "Gravity",
    "NR",
    "PhysicalPlan",
    "BackendProgram",
    "NXCandidate",
    "ParetoFrontier",
    "SelectedNX",
    "ResidentInstance",
)

CANONICAL_DEFINITIONS = {
    "SourceSpecimen": "Pinned cold source checkpoint used to derive candidates.",
    "Hawking": "The complete local physical AI environment and public surface.",
    "Doctor": "Measurement, prescription, verification, and rejection mechanism.",
    "Gravity": "Hawking's science and machinery for discovering capability-preserving cognitive, representational, executable, and physical forms suited to an objective and machine; it is not a file format.",
    "Collapse": "A transformation performed by Gravity; not an artifact status or subsystem.",
    "Nova": "A Gravity transformation that changes the learned organism and records lineage.",
    "NR": "Portable Noetic Representation: a versioned, lineage-aware representation whose closed, direct, source-independent revision is a runnable Hawking model; no machine binding is required.",
    "NoeticRepresentation": "The semantic form of an NR: a mutable, inspectable, portable representation produced by Gravity collapse and eligible for direct Hawking execution when its declared closure is qualified.",
    "NX": "Optional immutable machine/deployment specialization of one exact NR revision; NX creation is not required for a qualified NR to run.",
    "NoeticIR": "Portable representation/executable-intelligence ontology; not limited to tensors.",
    "NoeticExecutable": "A declared execution realization of a closed NR, independent of the cold Source Specimen; it may be portable or machine-specialized.",
    "PhysicalPlan": "Planning projection over stable semantic identities, machine facts, placement, state, effects and synchronization.",
    "BackendProgram": "One backend realization of a selected legal execution region.",
    "NXCandidate": "Runnable, qualified-to-its-claim deployment candidate derived from an exact NR revision.",
    "ParetoFrontier": "Set of non-dominated qualified Noetic Executables.",
    "ParetoArchive": "Durable record of frontier candidates and qualification/rejection reasons.",
    "SelectedNX": "NX selected from the Pareto Frontier for an explicit execution profile.",
    "ResidentInstance": "Currently instantiated/running Noetic Executable.",
}

# These are semantic compatibility views, not mechanical rename instructions.
COMPATIBILITY_ALIASES = {
    "source model": "SourceSpecimen",
    "checkpoint": "SourceSpecimen",
    "model lake": "SourceSpecimenStore",
    "Doctor": "GravityDiagnosis",
    "Tabula": "GravityDiscovery",
    "Deep Gravity": "GravityRun",
    "Model Gravity": "GravityTargetModel",
    "Context Gravity": "GravityTargetContext",
    "State Gravity": "GravityTargetState",
    "Noetic IR": "NRInternalFormat",
    "Noetic Program": "NRInternalFormat",
    "Noetic Compiler": "GravityCompilation",
    "Singularity": "SelectedNX",
    "Singularity Profile": "ExecutionProfile",
    "Hawking Accelerator": "HawkingExecutionBackend",
    "quantization": "GravityOperator",
    "quantizer": "GravityOperator",
    "compressed model": "NoeticRepresentation",
    "compact model": "NoeticRepresentation",
    "artifact": "SemanticInspectionRequired",
    "winner": "ParetoCandidateOrSelectedNX",
    "best model": "ParetoCandidateOrSelectedNX",
    "final model": "SelectedNXOrUnqualifiedCandidate",
    "production model": "SelectedNXOrUnqualifiedCandidate",
    "resident model": "ResidentInstance",
}


def nomenclature_metadata() -> Dict[str, Any]:
    """Return immutable-by-convention metadata for new active receipts."""
    return {
        "nomenclature_version": NOMENCLATURE_VERSION,
        "artifact_extensions": {"representation_shard": NR_EXTENSION, "final_executable": NX_EXTENSION},
        "canonical_pipeline": list(CANONICAL_PIPELINE),
    }


def annotate_receipt(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy a receipt payload and add the forward nomenclature marker."""
    result = dict(payload)
    result.setdefault("nomenclature_version", NOMENCLATURE_VERSION)
    return result


__all__ = [
    "CANONICAL_DEFINITIONS",
    "CANONICAL_PIPELINE",
    "COMPATIBILITY_ALIASES",
    "NOMENCLATURE_VERSION",
    "NR_EXTENSION",
    "NX_EXTENSION",
    "annotate_receipt",
    "nomenclature_metadata",
]
