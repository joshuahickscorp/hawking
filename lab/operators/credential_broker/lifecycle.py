"""Source lifecycle state machine: stream → verify → Gravity transform → seal → evict.

Generalises:
- ``GravityController`` (``lab.operators.condense_controller``):
  N-1 sealed+evicted before N may process; one heavy lease; seal_and_evict
- GLM ``glm52_source_fetch``: VERIFIED receipt survives body eviction; six
  independent eviction conditions; deferred refusals
- GLM ``LayerStream.evict``: source-only reclaim of stream_root shards
- DeepSeek header executor: floor before/during/after; assert_source_evicted
- Kimi admission: never materialise weight bodies; claim boundary explicit

This scaffold tracks phases and enforces transition laws. It does not download,
transform, or unlink files — executors perform those acts and report outcomes
here, matching the controller/executor split already used by Condense/Gravity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LifecycleError(RuntimeError):
    """Illegal lifecycle transition."""


class LifecyclePhase(str, Enum):
    PENDING = "PENDING"
    PREFLIGHT_SEALED = "PREFLIGHT_SEALED"
    STREAMING = "STREAMING"
    VERIFIED = "VERIFIED"
    TRANSFORMING = "TRANSFORMING"
    TRANSFORMED = "TRANSFORMED"
    SEALED = "SEALED"
    EVICTED = "EVICTED"
    FAILED = "FAILED"


# Legal forward transitions. FAILED is reachable from any non-terminal phase.
_FORWARD: dict[LifecyclePhase, frozenset[LifecyclePhase]] = {
    LifecyclePhase.PENDING: frozenset({LifecyclePhase.PREFLIGHT_SEALED}),
    LifecyclePhase.PREFLIGHT_SEALED: frozenset({LifecyclePhase.STREAMING}),
    LifecyclePhase.STREAMING: frozenset({LifecyclePhase.VERIFIED}),
    LifecyclePhase.VERIFIED: frozenset({LifecyclePhase.TRANSFORMING}),
    LifecyclePhase.TRANSFORMING: frozenset({LifecyclePhase.TRANSFORMED}),
    LifecyclePhase.TRANSFORMED: frozenset({LifecyclePhase.SEALED}),
    LifecyclePhase.SEALED: frozenset({LifecyclePhase.EVICTED}),
    LifecyclePhase.EVICTED: frozenset(),
    LifecyclePhase.FAILED: frozenset(),
}


@dataclass
class SourceLifecycle:
    """One source window/range family's lifecycle under Bible §7."""

    task_id: str
    repository: str
    revision_commit: str
    phase: LifecyclePhase = LifecyclePhase.PENDING
    events: list[dict[str, Any]] = field(default_factory=list)
    verify_sha256: str | None = None
    seal_sha256: str | None = None
    bytes_streamed: int = 0
    bytes_reclaimed: int = 0
    source_resident: bool = False
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise LifecycleError("task_id must be a non-empty string")
        if not isinstance(self.repository, str) or not self.repository.strip():
            raise LifecycleError("repository must be a non-empty string")
        if not isinstance(self.revision_commit, str) or len(self.revision_commit) != 40:
            raise LifecycleError("revision_commit must be a 40-character pin")

    def _record(self, event: str, **payload: Any) -> None:
        self.events.append({"event": event, "phase": self.phase.value, **payload})

    def _transition(self, target: LifecyclePhase) -> None:
        if self.phase is LifecyclePhase.FAILED:
            raise LifecycleError("failed lifecycle cannot transition further")
        if self.phase is LifecyclePhase.EVICTED:
            raise LifecycleError("evicted lifecycle is terminal")
        allowed = _FORWARD[self.phase]
        if target not in allowed:
            raise LifecycleError(
                f"illegal transition {self.phase.value} -> {target.value}; "
                f"allowed={sorted(p.value for p in allowed)}"
            )
        self.phase = target

    def seal_preflight(self, preflight_seal_sha256: str) -> None:
        """Bind a validated AcquisitionPreflight before any stream."""
        if not isinstance(preflight_seal_sha256, str) or len(preflight_seal_sha256) != 64:
            raise LifecycleError("preflight seal must be a 64-character sha256")
        self._transition(LifecyclePhase.PREFLIGHT_SEALED)
        self._record("preflight_sealed", seal_sha256=preflight_seal_sha256.lower())

    def begin_stream(self, *, expected_bytes: int) -> None:
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise LifecycleError("expected_bytes must be a positive integer")
        self._transition(LifecyclePhase.STREAMING)
        self.source_resident = True
        self._record("stream_started", expected_bytes=expected_bytes)

    def complete_verify(self, *, observed_bytes: int, content_sha256: str) -> None:
        if self.phase is not LifecyclePhase.STREAMING:
            raise LifecycleError("verify requires STREAMING phase")
        if isinstance(observed_bytes, bool) or not isinstance(observed_bytes, int) or observed_bytes < 0:
            raise LifecycleError("observed_bytes must be a non-negative integer")
        if not isinstance(content_sha256, str) or len(content_sha256) != 64:
            raise LifecycleError("content_sha256 must be a 64-character hex digest")
        self.bytes_streamed = observed_bytes
        self.verify_sha256 = content_sha256.lower()
        self._transition(LifecyclePhase.VERIFIED)
        self._record(
            "verified",
            observed_bytes=observed_bytes,
            content_sha256=self.verify_sha256,
        )

    def begin_transform(self) -> None:
        """Gravity transform may start only after verify (never on raw unverified source)."""
        self._transition(LifecyclePhase.TRANSFORMING)
        self._record("gravity_transform_started")

    def complete_transform(self, *, artifact_ids: list[str]) -> None:
        if self.phase is not LifecyclePhase.TRANSFORMING:
            raise LifecycleError("complete_transform requires TRANSFORMING phase")
        if not artifact_ids:
            raise LifecycleError("transform must emit at least one artifact id")
        self._transition(LifecyclePhase.TRANSFORMED)
        self._record("gravity_transform_completed", artifact_ids=list(artifact_ids))

    def seal(self, seal_sha256: str) -> None:
        if not isinstance(seal_sha256, str) or len(seal_sha256) != 64:
            raise LifecycleError("seal_sha256 must be a 64-character hex digest")
        self._transition(LifecyclePhase.SEALED)
        self.seal_sha256 = seal_sha256.lower()
        self._record("sealed", seal_sha256=self.seal_sha256)

    def evict_source(self, *, bytes_reclaimed: int) -> None:
        """Source eviction only after seal — never discard the only copy early.

        Matches GLM six-condition eviction spirit: verified + transformed +
        sealed before unlink. This controller records the claim; the executor
        performs the unlink and supplies ``bytes_reclaimed``.
        """
        if isinstance(bytes_reclaimed, bool) or not isinstance(bytes_reclaimed, int) or bytes_reclaimed < 0:
            raise LifecycleError("bytes_reclaimed must be a non-negative integer")
        self._transition(LifecyclePhase.EVICTED)
        self.source_resident = False
        self.bytes_reclaimed = bytes_reclaimed
        self._record(
            "source_evicted",
            bytes_reclaimed=bytes_reclaimed,
            policy="source_only_after_seal",
        )

    def fail(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise LifecycleError("failure reason must be non-empty")
        if self.phase in {LifecyclePhase.EVICTED, LifecyclePhase.FAILED}:
            raise LifecycleError(f"cannot fail from terminal phase {self.phase.value}")
        previous = self.phase
        self.phase = LifecyclePhase.FAILED
        self.failure_reason = reason.strip()
        # On failure, source should be treated as reclaimable by the executor;
        # we mark non-resident only if it never sealed (executors decide).
        self._record("failed", from_phase=previous.value, reason=self.failure_reason)

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "hawking.ascension.source_lifecycle.v1",
            "task_id": self.task_id,
            "repository": self.repository,
            "revision_commit": self.revision_commit,
            "phase": self.phase.value,
            "verify_sha256": self.verify_sha256,
            "seal_sha256": self.seal_sha256,
            "bytes_streamed": self.bytes_streamed,
            "bytes_reclaimed": self.bytes_reclaimed,
            "source_resident": self.source_resident,
            "failure_reason": self.failure_reason,
            "events": list(self.events),
            "lifecycle_law": [
                "stream",
                "verify",
                "gravity_transform",
                "seal",
                "evict_source",
            ],
        }
