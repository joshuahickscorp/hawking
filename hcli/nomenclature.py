"""Forward-compatible Hawking nomenclature and semantic aliases.

This module is deliberately small: it gives active serializers one canonical
version and a machine-readable vocabulary without renaming sealed historical
paths, schemas, or receipts.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping


NOMENCLATURE_VERSION = "HAWKING_NOMENCLATURE_V1"

CANONICAL_PIPELINE = (
    "SourceSpecimen",
    "Doctor",
    "Gravity",
    "NoeticIR",
    "NoeticCompiler",
    "PhysicalGraphCompiler",
    "HawkingAccelerator",
    "NoeticExecutableCandidate",
    "ParetoFrontier",
    "Singularity",
    "ResidentInstance",
)

CANONICAL_DEFINITIONS = {
    "SourceSpecimen": "Pinned cold source checkpoint used to derive candidates.",
    "Doctor": "Measurement, prescription, verification, and rejection mechanism.",
    "Gravity": "Capability-preserving search/reduction of physical cost; quantization is one operator.",
    "NoeticIR": "Portable representation/executable-intelligence ontology; not limited to tensors.",
    "NoeticExecutable": "Complete runnable bundle independent of the cold Source Specimen.",
    "ParetoFrontier": "Set of non-dominated qualified Noetic Executables.",
    "ParetoArchive": "Durable record of frontier candidates and qualification/rejection reasons.",
    "Singularity": "Profile-specific promoted Noetic Executable selected from the Pareto Frontier.",
    "ResidentInstance": "Currently instantiated/running Noetic Executable.",
}

# These are semantic compatibility views, not mechanical rename instructions.
COMPATIBILITY_ALIASES = {
    "source model": "SourceSpecimen",
    "checkpoint": "SourceSpecimen",
    "model lake": "SourceSpecimenStore",
    "quantization": "GravityOperator",
    "quantizer": "GravityOperator",
    "compressed model": "NoeticRepresentation",
    "compact model": "NoeticRepresentation",
    "artifact": "SemanticInspectionRequired",
    "winner": "ParetoCandidateOrSingularity",
    "best model": "ParetoCandidateOrSingularity",
    "final model": "SingularityOrUnqualifiedCandidate",
    "production model": "SingularityOrUnqualifiedCandidate",
    "resident model": "ResidentInstance",
}


def nomenclature_metadata() -> Dict[str, Any]:
    """Return immutable-by-convention metadata for new active receipts."""
    return {
        "nomenclature_version": NOMENCLATURE_VERSION,
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
    "annotate_receipt",
    "nomenclature_metadata",
]
