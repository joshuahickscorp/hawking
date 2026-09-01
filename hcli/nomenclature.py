"""Forward-compatible Hawking nomenclature and semantic aliases.

This module is deliberately small: it gives active serializers one canonical
version and a machine-readable vocabulary without renaming sealed historical
paths, schemas, or receipts.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping


NOMENCLATURE_VERSION = "HAWKING_NOMENCLATURE_V1"

# Active artifact extensions. Gravity is a process, not a file format. The
# ``.gravity`` suffix remains readable for sealed historical work; new
# representation shards use NR and a machine-bound executable uses NX.
NR_EXTENSION = ".nr"
NX_EXTENSION = ".nx"

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
    "Gravity": "The search/research process that discovers lower-information representations of useful model function; it is not a file format.",
    "NR": "Portable Noetic Representation: transient shard/container between source and a Noetic executable; no machine binding.",
    "NX": "Machine-bound Noetic Executable: final compiled executable derived from an NR representation.",
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
