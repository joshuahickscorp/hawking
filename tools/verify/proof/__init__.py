"""Verifier backends and the expert-iteration (attempt → verify → repair) loop.

This package is the functional-transfer verification surface:

* a common ``VerifierBackend`` interface with honest REAL vs GATED status;
* at least one REAL host backend (exact numeric / sympy when importable);
* a Lean backend that fails closed when the formal toolchain is incomplete —
  it never invents proofs or container hashes;
* an expert-iteration harness that only emits *verified* trajectories into
  the scaffold paired-trace / TraceRecord training format.

Nothing here sets ``RAMANUJAN_RESEARCH_AUTHORIZED``. Teacher critique is gated
on ``REQUIRES_GLM_ACCESS`` and is never faked.
"""
from __future__ import annotations

from tools.verify.proof.base import (
    BackendAvailability,
    BackendStatus,
    VerificationRequest,
    VerificationResult,
    Verdict,
)
from tools.verify.proof.expert_iteration import (
    ExpertIterationHarness,
    ExpertIterationResult,
    StudentAttempt,
    TeacherCritiqueGate,
)
from tools.verify.proof.registry import (
    VerifierRegistry,
    default_registry,
    probe_backends,
)
from tools.verify.proof.trajectory import (
    PAIRED_TRACE_SCHEMA,
    VERIFIED_TRAJECTORY_SCHEMA,
    emit_paired_trace_record,
    emit_verified_trajectory,
)

__all__ = [
    "BackendAvailability",
    "BackendStatus",
    "ExpertIterationHarness",
    "ExpertIterationResult",
    "PAIRED_TRACE_SCHEMA",
    "StudentAttempt",
    "TeacherCritiqueGate",
    "VERIFIED_TRAJECTORY_SCHEMA",
    "VerificationRequest",
    "VerificationResult",
    "Verdict",
    "VerifierRegistry",
    "default_registry",
    "emit_paired_trace_record",
    "emit_verified_trajectory",
    "probe_backends",
]

