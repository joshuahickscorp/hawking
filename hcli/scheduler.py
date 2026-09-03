from __future__ import annotations

import itertools

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .dag_store import DagStore
from .resources import (
    MutationLock,
    ResourceLimits,
    normalize_resource_class,
    occupancy_of,
)
from .workunit import (
    MAX_REPAIR_DEPTH,
    MAX_REPAIRS_PER_ROOT,
    AdmitOutcome,
    IdentityConflict,
    WorkUnit,
    admit_unit,
    assign_ready,
    emit_repair,
    identify_ready,
    rebuild_repair_budget,
    serialize_repair_budget,
    transition_status,
)

LOG = logging.getLogger("hcli.scheduler")

# Re-exported from workunit.py (imported above). Do not reassign these
# names: a second `MAX_REPAIR_* = N` here used to shadow the import and
# split the retry-policy authority. Dispatch is FIFO by ready_at; the
# retired `_remaining_depth` helper lived next to that shadow and is gone.
# Measurement copy: tools/headless/hcli_scheduler_quality.py::remaining_depth.


DEFAULT_NO_PROGRESS_THRESHOLD = 3


def verification_passed(outcome: Any) -> bool:
    """True iff ``outcome`` is a deterministic verifier dict that passed."""
    return isinstance(outcome, dict) and outcome.get("ok") is True


class UnverifiedCompletion(ValueError):
    """Raised when complete() is asked to accept a unit without a passing verifier."""


class NO_PROGRESS(Exception):
    """Raised when a mission fingerprint repeats without progress.

    The caller must handle this: do not silently continue, and do not
    silently stop. Surface it.
    """

    def __init__(self, fingerprint: str, count: int, threshold: int) -> None:
        self.fingerprint = fingerprint
        self.count = count
        self.threshold = threshold
        super().__init__(
            f"NO_PROGRESS fingerprint={fingerprint} count={count} "
            f"threshold={threshold}"
        )


class Scheduler:
    """Dispatch existing WorkUnits. Does not invent work.

    Admission is ``submit`` / ``replan``: same content is idempotent, same
    id with different content is ``IdentityConflict``. ``dispatch`` only
    admits units already in ``self.units``. The sole synthesis path is
    ``fail`` -> ``_emit_repair`` -> ``workunit.emit_repair``, which creates
    a repair of an existing failed unit under a bounded budget. Idle
    resource classes stay idle.

    Observable at dispatch time today: per-class requested/admitted,
    occupancy, and ``mutation_blocked`` (MUTATION-ready minus admitted).
    Verifier backlog is not a scheduler input — it lives on the ledger
    and is formatted by ``/status``.
    """

    def __init__(
        self,
        units: Dict[str, WorkUnit],
        runtime_count: int,
        workspace: Optional[Union[str, Path]] = None,
        *,
        no_progress_threshold: int = DEFAULT_NO_PROGRESS_THRESHOLD,
        repo_root: Optional[Union[str, Path]] = None,
        limits: Optional[ResourceLimits] = None,
    ) -> None:
        self.units = units
        self.runtime_count = runtime_count
        self.workspace = Path(workspace) if workspace is not None else None
        self.no_progress_threshold = max(1, int(no_progress_threshold))
        self.limits = limits if limits is not None else ResourceLimits.resolve(
            repo_root=repo_root
        )
        self.active_decode_limit = self.limits.gpu_decode
        self.active_decode_limit_source = self.limits.gpu_decode_source
        self.store = DagStore(self.workspace) if self.workspace is not None else None
        self.mutation_lock = MutationLock(self.workspace)
        self._ready_seq: Dict[str, int] = {}
        self._ready_counter = itertools.count()
        boot = rebuild_repair_budget(self.units)
        self._repair_signatures: Dict[str, set] = {
            str(k): set(v) for k, v in (boot.get("signatures") or {}).items()
        }
        self._repair_counts: Dict[str, int] = dict(boot.get("counts") or {})
        self._fingerprints: List[str] = []
        self.last_dispatch: Optional[Dict[str, Any]] = None
        self._persist()

    @classmethod
    def from_workspace(
        cls,
        workspace: Union[str, Path],
        runtime_count: int = 1,
        grok_liveness: Optional[Any] = None,
        **kwargs: Any,
    ) -> "Scheduler":
        """Rebuild from disk, adopting Grok work that is still genuinely alive.

        `grok_liveness` answers "is this backend task still running?" for a unit
        that was mid-flight when the process died. Without it every such unit is
        failed and re-readied, and the executor launches a SECOND task for work
        that is still running -- a duplicate plus an orphan, on every restart.

        Defaulting to GrokBridge.status keeps that reconciliation on by default,
        and any failure to construct the bridge falls back to None, which means
        "cannot observe" and therefore "do not adopt" -- the safe direction.
        """
        store = DagStore(workspace)
        if grok_liveness is None:
            try:
                from .grok_bridge import GrokBridge

                grok_liveness = GrokBridge(workspace).status
            except Exception:
                grok_liveness = None
        units = store.load(recover_running=True, grok_liveness=grok_liveness)
        sched = cls(units, runtime_count, workspace=workspace, **kwargs)
        meta = store.last_meta or {}
        fps = meta.get("fingerprints")
        if isinstance(fps, list):
            sched._fingerprints = [str(item) for item in fps]
        if "no_progress_threshold" in meta and "no_progress_threshold" not in kwargs:
            try:
                sched.no_progress_threshold = max(1, int(meta["no_progress_threshold"]))
            except (TypeError, ValueError):
                pass
        ld = meta.get("last_dispatch")
        if isinstance(ld, dict):
            sched.last_dispatch = ld
        # Disk floor wins over the in-process maps, which reset on death.
        # Repair units plus persisted counts are the lineage anchor.
        budget = rebuild_repair_budget(
            sched.units, meta.get("repair_budget") or meta
        )
        sched._repair_counts = dict(budget.get("counts") or {})
        sched._repair_signatures = {
            str(k): set(v) for k, v in (budget.get("signatures") or {}).items()
        }
        sched._persist()
        return sched

    def submit(self, wu: WorkUnit) -> AdmitOutcome:
        """Admit one unit.

        Same content (any id) is idempotent: the existing unit is returned
        and is not dispatched a second time. Same id with different content
        raises ``IdentityConflict`` and does not write.
        """
        if not isinstance(wu, WorkUnit):
            raise TypeError(f"submit requires a WorkUnit, got {type(wu)!r}")
        outcome = admit_unit(self.units, wu)
        self._persist()
        return outcome

    def replan(
        self,
        incoming: Union[Dict[str, WorkUnit], Iterable[WorkUnit]],
    ) -> List[AdmitOutcome]:
        """Admit a recomputed DAG without changing identity of existing work.

        The existing unit wins, so status, ``repair_root``, and repair
        lineage stay put. Units not in ``incoming`` are kept — a replan
        does not drop in-flight or repair work. Same id with different
        content raises ``IdentityConflict`` and leaves the live graph
        unchanged.
        """
        if isinstance(incoming, dict):
            batch = list(incoming.values())
        else:
            batch = list(incoming)
        probe: Dict[str, WorkUnit] = dict(self.units)
        outcomes: List[AdmitOutcome] = []
        for wu in batch:
            if not isinstance(wu, WorkUnit):
                raise TypeError(f"replan requires WorkUnits, got {type(wu)!r}")
            outcomes.append(admit_unit(probe, wu))
        self.units = probe
        self._persist()
        return outcomes

    def dispatch(self) -> List[tuple]:
        """Return assignments without waiting. A finished lane is released
        by complete()/fail() and the next dependency-ready unit is claimed
        on the subsequent dispatch — there is no barrier across unrelated
        chains.
        """
        t0 = time.monotonic()
        ready = identify_ready(self.units)
        # Deterministic FIFO: longest-ready first, ties by id.
        #
        # This USED to sort by remaining graph depth, on the theory that a
        # flood of cheap units buries the chain that unblocks everything else.
        # An independent lane measured that theory across four DAG shapes and
        # it does not survive: critical-path ordering helps only when a
        # hop-deep chain is ALSO the duration bottleneck, is a no-op at
        # concurrency 1 and whenever the resource class is not at its cap, and
        # actively HURTS when a hop-deep cheap chain delays a
        # duration-dominant unit. `remaining_depth` counts hops, and hops are
        # not time -- it cannot see the one long unit that actually sets this
        # campaign's wall clock. The "54 dispatch ticks versus 4" figure that
        # justified it did not reproduce; the chain tail was depth 1.
        # Depth ordering also starves: a unit that always sorts last can be
        # passed over indefinitely.
        #
        # Unstamped units sort LAST, not as 0.0 -- `ready_at or 0.0` made a
        # unit that had never been stamped jump ahead of every unit that had.
        # `ready_at` is wall-clock and its resolution is coarse enough that two
        # units stamped in the same pass frequently share a value. The id then
        # broke the tie, which made dispatch order alphabetical-by-accident and
        # nondeterministic run to run: a GPU_EXCLUSIVE unit and a GPU_DECODE unit
        # created together went 31/29 either way across 60 runs. Exclusion always
        # held -- they never ran together -- but which one went first was a coin
        # flip, and a scheduler nobody can predict is one nobody can test.
        #
        # `_ready_seq` records the order units were FIRST seen ready, in this
        # process, and breaks the tie strictly. It lives on the scheduler rather
        # than the WorkUnit because readiness order is not durable state: after a
        # restart the surviving order is `ready_at`, which is what should carry
        # across processes.
        for unit in ready:
            if unit.id not in self._ready_seq:
                self._ready_seq[unit.id] = next(self._ready_counter)
        ready.sort(key=lambda u: (
            u.ready_at if u.ready_at is not None else float("inf"),
            self._ready_seq.get(u.id, 0),
            u.id,
        ))
        requested: Dict[str, int] = {}
        for unit in ready:
            rc = normalize_resource_class(unit.resource_class)
            requested[rc] = requested.get(rc, 0) + 1
        assignments = assign_ready(
            ready,
            self.runtime_count,
            all_units=self.units,
            limits=self.limits,
            mutation_lock=self.mutation_lock,
        )
        admitted: Dict[str, int] = {}
        for wu, _slot in assignments:
            rc = normalize_resource_class(wu.resource_class)
            admitted[rc] = admitted.get(rc, 0) + 1
        mutation_ready = requested.get("MUTATION", 0)
        mutation_admitted = admitted.get("MUTATION", 0)
        self.last_dispatch = {
            "requested": requested,
            "admitted": admitted,
            "overhead_s": time.monotonic() - t0,
            "occupancy": dict(occupancy_of(self.units.values())),
            "mutation_blocked": max(0, mutation_ready - mutation_admitted),
        }
        self._persist()
        return assignments

    def complete(
        self,
        wu_id: str,
        fingerprint: Optional[str] = None,
        verification: Optional[Dict[str, Any]] = None,
    ) -> None:
        wu = self.units.get(wu_id)
        if not wu:
            return
        outcome = verification if verification is not None else wu.verification
        if not verification_passed(outcome):
            raise UnverifiedCompletion(
                f"WorkUnit {wu_id} cannot complete without a passing verifier outcome"
            )
        wu.verification = dict(outcome)
        was_running = wu.status == "running"
        transition_status(wu, "completed")
        wu.finished_at = time.time()
        if was_running:
            self._release_unit(wu)
        self._persist()
        self._record_fingerprint(fingerprint)

    def fail(
        self,
        wu_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[WorkUnit]:
        wu = self.units.get(wu_id)
        if not wu:
            return None
        was_running = wu.status == "running"
        transition_status(wu, "failed")
        wu.finished_at = time.time()
        if was_running:
            self._release_unit(wu)
        # The reason belongs on the unit that FAILED, not only on the descendant
        # that repairs it. `_integrate` computes the validation and the error and
        # hands them here, and they were being attached exclusively to the repair
        # unit -- so anyone reading a failed unit saw `failure_context: {}` and
        # could not tell whether the verifier bit, the model refused, or the
        # engine errored. Three units failed in the live run with no reason
        # recorded anywhere.
        if isinstance(context, dict) and context:
            merged = dict(wu.failure_context or {})
            merged.update({k: v for k, v in context.items() if v is not None})
            wu.failure_context = merged
        repair = self._emit_repair(wu, context)
        self._persist()
        return repair

    def is_done(self) -> bool:
        return all(u.status in ("completed", "failed") for u in self.units.values())

    def _release_unit(self, wu: WorkUnit) -> None:
        wu.assigned_runtime = None
        if normalize_resource_class(wu.resource_class) == "MUTATION":
            self.mutation_lock.release(wu.id)

    def _repair_budget(self) -> Dict[str, Any]:
        """Per-root counts and signatures: units, disk, then in-process maps.

        The in-process maps reset on death. Units and the DAG document do
        not. Take the max so a dropped repair unit cannot reopen the cap.
        """
        counts: Dict[str, int] = dict(self._repair_counts)
        signatures: Dict[str, set] = {
            str(k): set(v) for k, v in self._repair_signatures.items()
        }
        persisted = None
        if self.store is not None:
            meta = self.store.last_meta or {}
            persisted = meta.get("repair_budget") or meta
        disk = rebuild_repair_budget(self.units, persisted)
        for root, n in (disk.get("counts") or {}).items():
            counts[root] = max(int(counts.get(root, 0) or 0), int(n))
        for root, seen in (disk.get("signatures") or {}).items():
            signatures.setdefault(str(root), set()).update(seen)
        return rebuild_repair_budget(
            self.units, {"counts": counts, "signatures": signatures}
        )

    def _emit_repair(
        self,
        wu: WorkUnit,
        context: Optional[Dict[str, Any]],
    ) -> Optional[WorkUnit]:
        """Create one repair unit, or refuse and mark the lineage exhausted.

        This is the only path that synthesises a WorkUnit. It is a repair
        of an existing failed unit, not filler: the new id is derived from
        the failed id, ``repairs`` and ``repair_root`` point at the original,
        and the per-root / depth caps refuse to grow the tree forever.

        Without a bound this recurses: a permanently unavailable backend fails
        the repair, which emits a repair of the repair, and so on. Observed on
        this box as grokfail.repair.1.repair.1.repair.1... The retry budget
        bounds attempts of ONE unit; it never bounded the DEPTH of
        repair-of-a-repair, so a dead backend manufactured work forever.

        Depth is carried on the unit, not parsed out of its id. The emitter
        is ``workunit.emit_repair`` so a restart reconstructs the budget
        from units plus the DAG document instead of from these maps.
        """
        budget = self._repair_budget()
        repair = emit_repair(self.units, wu, context, budget=budget)
        self._repair_counts = dict(budget.get("counts") or {})
        self._repair_signatures = {
            str(k): set(v) for k, v in (budget.get("signatures") or {}).items()
        }
        if repair is None and getattr(wu, "repair_exhausted", False):
            LOG.warning(
                "repair refused: root=%s reason=%s unit=%s",
                wu.repair_root,
                wu.repair_reason,
                wu.id,
            )
        return repair

    def _record_fingerprint(self, fingerprint: Optional[str]) -> None:
        if fingerprint is None:
            return
        self._fingerprints.append(fingerprint)
        count = self._fingerprints.count(fingerprint)
        if count >= self.no_progress_threshold:
            raise NO_PROGRESS(
                fingerprint=fingerprint,
                count=count,
                threshold=self.no_progress_threshold,
            )

    def _persist(self, extra: Optional[Dict[str, Any]] = None) -> None:
        if self.store is None:
            return
        budget = self._repair_budget()
        self._repair_counts = dict(budget.get("counts") or {})
        self._repair_signatures = {
            str(k): set(v) for k, v in (budget.get("signatures") or {}).items()
        }
        metadata = {
            "fingerprints": list(self._fingerprints),
            "no_progress_threshold": self.no_progress_threshold,
            "active_decode_limit": self.active_decode_limit,
            "active_decode_limit_source": self.active_decode_limit_source,
            "last_dispatch": self.last_dispatch,
            "repair_budget": serialize_repair_budget(budget),
        }
        if extra:
            metadata.update(dict(extra))
        self.store.save(
            self.units,
            extra=metadata,
        )
