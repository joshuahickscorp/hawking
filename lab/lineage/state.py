"""Three named lineage slots. There is never zero valid Genesis.

Slots are always CURRENT, CANDIDATE, LAST_KNOWN_GOOD. A failed successor
launch restores CURRENT from LAST_KNOWN_GOOD. Promotion authority is not
here — this module only stores occupants and exercises rollback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from lab.lineage.canon import utc_now
from lab.lineage.identity import GenesisInstance, Invoker, as_instance, as_invoker
from lab.lineage.transfer import TransferError, accept_transfer, pack_state
from lab.receipts import seal, verify

SCHEMA = "hawking.lineage.state.v1"
LAUNCH_SCHEMA = "hawking.lineage.successor_launch.v1"
ROLLBACK_SCHEMA = "hawking.lineage.rollback.v1"
HANDOVER_SCHEMA = "hawking.lineage.handover.v1"

CURRENT = "CURRENT"
CANDIDATE = "CANDIDATE"
LAST_KNOWN_GOOD = "LAST_KNOWN_GOOD"
SLOT_NAMES: tuple[str, ...] = (CURRENT, CANDIDATE, LAST_KNOWN_GOOD)


class LineageError(ValueError):
    """Unlawful lineage mutation."""


class LineageInvariantError(LineageError):
    """Would leave the organism with zero valid Genesis."""


@dataclass(frozen=True)
class LaunchResult:
    ok: bool
    rolled_back: bool
    reason: str
    receipt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "rolled_back": self.rolled_back,
            "reason": self.reason,
            "receipt": dict(self.receipt),
        }


class LineageState:
    """Exactly three named slots. Armed after the first install."""

    def __init__(self) -> None:
        self._slots: dict[str, GenesisInstance | None] = {
            CURRENT: None,
            CANDIDATE: None,
            LAST_KNOWN_GOOD: None,
        }
        self._armed = False
        self._events: list[dict[str, Any]] = []

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def current(self) -> GenesisInstance | None:
        return self._slots[CURRENT]

    @property
    def candidate(self) -> GenesisInstance | None:
        return self._slots[CANDIDATE]

    @property
    def last_known_good(self) -> GenesisInstance | None:
        return self._slots[LAST_KNOWN_GOOD]

    def slot(self, name: str) -> GenesisInstance | None:
        if name not in SLOT_NAMES:
            raise LineageError(f"unknown lineage slot {name!r}; exactly {list(SLOT_NAMES)}")
        return self._slots[name]

    def valid_instances(self) -> list[GenesisInstance]:
        found: list[GenesisInstance] = []
        for name in SLOT_NAMES:
            occupant = self._slots[name]
            if occupant is not None and occupant.valid:
                found.append(occupant)
        return found

    def valid_count(self) -> int:
        return len(self.valid_instances())

    def _put(self, name: str, occupant: GenesisInstance | None) -> None:
        if name not in SLOT_NAMES:
            raise LineageError(f"refusing unnamed slot {name!r}")
        if occupant is None:
            self._slots[name] = None
            return
        copy = occupant.copy()
        copy.role = name.lower()
        if name == CURRENT:
            copy.live = True
        elif name == LAST_KNOWN_GOOD:
            copy.live = False
        self._slots[name] = copy

    def _assert_invariant(self) -> None:
        if set(self._slots) != set(SLOT_NAMES):
            raise LineageInvariantError(
                f"lineage slots must be exactly {list(SLOT_NAMES)}; got {sorted(self._slots)}"
            )
        if self._armed and self.valid_count() == 0:
            raise LineageInvariantError(
                "zero valid Genesis forbidden; rollback to LAST_KNOWN_GOOD or refuse the mutation"
            )

    def _record(self, kind: str, payload: Mapping[str, Any]) -> None:
        self._events.append({"ts": utc_now(), "kind": kind, "payload": dict(payload)})

    def install(self, genesis: GenesisInstance | Mapping[str, Any]) -> dict[str, Any]:
        """Seat the first Genesis. CURRENT and LAST_KNOWN_GOOD both become valid copies."""
        if self._armed and self.valid_count() > 0:
            raise LineageError("lineage already armed; successors go through nominate/handover")
        instance = as_instance(genesis, "genesis")
        if not instance.valid:
            raise LineageError("refusing to install an invalid Genesis")
        live = instance.copy()
        live.live = True
        live.terminated = False
        live.role = "current"
        reserve = instance.copy()
        reserve.live = False
        reserve.terminated = False
        reserve.role = "last_known_good"
        self._put(CURRENT, live)
        self._put(LAST_KNOWN_GOOD, reserve)
        self._put(CANDIDATE, None)
        self._armed = True
        self._assert_invariant()
        self._record("install", {"instance_id": live.instance_id, "generation": live.generation})
        return self.snapshot()

    def nominate(self, child: GenesisInstance | Mapping[str, Any]) -> GenesisInstance:
        if not self._armed:
            raise LineageError("install Genesis before nominating a candidate")
        if self.current is None or not self.current.valid:
            raise LineageError("CURRENT is not a valid Genesis; rollback or reinstall")
        child_i = as_instance(child, "candidate")
        if child_i.instance_id == self.current.instance_id:
            raise LineageError("candidate instance_id must differ from CURRENT")
        if child_i.generation <= self.current.generation:
            raise LineageError(
                f"candidate generation {child_i.generation} must exceed "
                f"CURRENT generation {self.current.generation}"
            )
        placed = child_i.copy()
        placed.live = False
        placed.launched = False
        placed.terminated = False
        placed.valid = True
        placed.role = "candidate"
        self._put(CANDIDATE, placed)
        self._assert_invariant()
        self._record("nominate", {"instance_id": placed.instance_id, "generation": placed.generation})
        return placed

    def snapshot_current_as_lkg(self) -> GenesisInstance:
        """Refresh LAST_KNOWN_GOOD from the seated CURRENT before a successor attempt."""
        current = self.current
        if current is None or not current.valid:
            raise LineageError("cannot snapshot an invalid CURRENT")
        reserve = current.copy()
        reserve.live = False
        reserve.terminated = False
        reserve.role = "last_known_good"
        self._put(LAST_KNOWN_GOOD, reserve)
        self._assert_invariant()
        self._record("snapshot_lkg", {"instance_id": reserve.instance_id})
        return reserve

    def launch_successor(
        self,
        launcher: Callable[[GenesisInstance], bool],
    ) -> LaunchResult:
        """Attempt to bring CANDIDATE up. Failure rolls back to LAST_KNOWN_GOOD."""
        if not self._armed:
            raise LineageError("lineage is not armed")
        candidate = self.candidate
        if candidate is None:
            raise LineageError("no CANDIDATE to launch")
        lkg = self.last_known_good
        if lkg is None or not lkg.valid:
            raise LineageError(
                "LAST_KNOWN_GOOD missing or invalid; refusing launch that cannot roll back"
            )
        if self.current is None or not self.current.valid:
            self.rollback(reason="CURRENT invalid before launch; restoring LAST_KNOWN_GOOD")
        try:
            ok = bool(launcher(candidate.copy()))
        except Exception as exc:  # noqa: BLE001 — launchers are foreign; any failure rolls back
            return self.rollback(reason=f"successor launch exception: {type(exc).__name__}: {exc}")
        if not ok:
            return self.rollback(reason="successor launch returned failure")
        launched = candidate.copy()
        launched.launched = True
        launched.valid = True
        launched.live = False
        self._put(CANDIDATE, launched)
        self._assert_invariant()
        receipt = seal(
            {
                "schema": LAUNCH_SCHEMA,
                "ok": True,
                "rolled_back": False,
                "reason": "successor launched; authority has not moved",
                "candidate_id": launched.instance_id,
                "current_id": self.current.instance_id if self.current else None,
                "lkg_id": lkg.instance_id,
                "valid_count": self.valid_count(),
                "recorded_at": utc_now(),
            }
        )
        self._record("launch_ok", {"candidate_id": launched.instance_id})
        return LaunchResult(ok=True, rolled_back=False, reason=receipt["reason"], receipt=receipt)

    def rollback(self, *, reason: str) -> LaunchResult:
        """Restore CURRENT from LAST_KNOWN_GOOD. CANDIDATE is marked failed."""
        lkg = self.last_known_good
        if lkg is None or not lkg.valid:
            raise LineageInvariantError(
                "rollback requested but LAST_KNOWN_GOOD is not a valid Genesis"
            )
        restored = lkg.copy()
        restored.live = True
        restored.terminated = False
        restored.valid = True
        restored.role = "current"
        failed_id = None
        if self.candidate is not None:
            failed = self.candidate.copy()
            failed.live = False
            failed.launched = False
            failed.valid = False
            failed.role = "failed_candidate"
            failed_id = failed.instance_id
            self._put(CANDIDATE, failed)
        else:
            self._put(CANDIDATE, None)
        self._put(CURRENT, restored)
        self._assert_invariant()
        receipt = seal(
            {
                "schema": ROLLBACK_SCHEMA,
                "ok": False,
                "rolled_back": True,
                "reason": reason,
                "restored_id": restored.instance_id,
                "failed_candidate_id": failed_id,
                "valid_count": self.valid_count(),
                "zero_valid_genesis": False,
                "recorded_at": utc_now(),
            }
        )
        self._record("rollback", {"reason": reason, "restored_id": restored.instance_id})
        return LaunchResult(ok=False, rolled_back=True, reason=reason, receipt=receipt)

    def handover(
        self,
        *,
        package: Mapping[str, Any],
        invoker: Invoker | Mapping[str, Any],
        verdict: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Move authority only after an external ACCEPT and a verified checksum."""
        from lab.lineage.promotion import (
            PROMOTION_SCHEMA,
            SelfCertificationRefused,
            refuse_self_certification,
        )

        if not self._armed:
            raise LineageError("lineage is not armed")
        parent = self.current
        child = self.candidate
        if parent is None or not parent.valid:
            raise LineageError("CURRENT must be a valid Genesis before handover")
        if child is None:
            raise LineageError("CANDIDATE missing; nothing to hand to")
        if self.last_known_good is None or not self.last_known_good.valid:
            raise LineageError("rollback artifact missing; refusing handover")
        inv = as_invoker(invoker)
        refuse_self_certification(inv, parent, child)
        verify(verdict, label="promotion_verdict")
        if verdict.get("schema") != PROMOTION_SCHEMA:
            raise LineageError("handover requires a sealed lineage promotion verdict")
        if verdict.get("verdict") != "ACCEPT":
            raise LineageError(
                f"handover requires ACCEPT; got {verdict.get('verdict')!r}"
            )
        if verdict.get("authority_level") != "authoritative":
            raise LineageError("handover requires an authoritative ACCEPT")
        accepted = accept_transfer(package)
        if accepted["from_instance"] != parent.instance_id:
            raise TransferError("transfer from_instance is not CURRENT")
        if accepted["to_instance"] != child.instance_id:
            raise TransferError("transfer to_instance is not CANDIDATE")

        retired = parent.copy()
        retired.live = False
        retired.terminated = True
        retired.valid = True
        retired.role = "last_known_good"
        successor = child.copy()
        successor.live = True
        successor.valid = True
        successor.terminated = False
        successor.launched = True
        successor.role = "current"
        successor.research_state = dict(accepted["payload"])
        self._put(LAST_KNOWN_GOOD, retired)
        self._put(CURRENT, successor)
        self._put(CANDIDATE, None)
        self._assert_invariant()
        document = {
            "schema": HANDOVER_SCHEMA,
            "from_instance": retired.instance_id,
            "to_instance": successor.instance_id,
            "checksum_sha256": accepted["checksum_sha256"],
            "checksum_verified": True,
            "parent_terminated": True,
            "valid_count": self.valid_count(),
            "invoker": inv.to_dict(),
            "promotion_seal": verdict.get("seal_sha256"),
            "recorded_at": utc_now(),
        }
        sealed = seal(document)
        self._record("handover", {"to": successor.instance_id, "from": retired.instance_id})
        return sealed

    def snapshot(self) -> dict[str, Any]:
        self._assert_invariant()
        return {
            "schema": SCHEMA,
            "armed": self._armed,
            "slots": {
                name: None if occ is None else occ.to_dict() for name, occ in self._slots.items()
            },
            "valid_count": self.valid_count(),
            "zero_valid_genesis": False if not self._armed else self.valid_count() == 0,
            "events": list(self._events),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.snapshot()
