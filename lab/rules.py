"""Governance ledger and promotion/burial enforcement (single entrypoint)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from lab.receipts import _sha256_hex as sha256_hex, _utc_now as utc_now
from lab.spec import BurialRule, ExperimentSpec, PromotionRule

LEDGER_SCHEMA = "hawking.lab.governance_ledger.v1"
GENESIS = "0" * 64


class GovernanceError(RuntimeError):
    pass


@dataclass
class GovernanceLedger:
    path: Path
    _head: str = GENESIS
    _count: int = 0
    _loaded: bool = False

    def load(self) -> None:
        self._head = GENESIS
        self._count = 0
        if not self.path.is_file():
            self._loaded = True
            return
        prev = GENESIS
        count = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if event.get("prev_sha256") != prev:
                    raise GovernanceError(f"ledger chain break at {self.path}:{line_no}")
                body = {k: v for k, v in event.items() if k != "event_sha256"}
                if event.get("event_sha256") != sha256_hex(body):
                    raise GovernanceError(f"ledger seal mismatch at {self.path}:{line_no}")
                prev = event["event_sha256"]
                count += 1
        self._head = prev
        self._count = count
        self._loaded = True

    def append(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self._loaded:
            self.load()
        body = {
            "schema": LEDGER_SCHEMA,
            "prev_sha256": self._head,
            "at": utc_now(),
            "kind": kind,
            "payload": dict(payload),
        }
        sealed = {**body, "event_sha256": sha256_hex(body)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sealed, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._head = sealed["event_sha256"]
        self._count += 1
        return sealed


def apply_governance(
    spec: ExperimentSpec,
    *,
    ledger: GovernanceLedger,
    verdict: str,
    gate_results: Mapping[str, bool],
    author: str = "",
    admitter: str = "",
    measurement_kind: str = "real",
    action: str = "promote",
    artifacts: list[Path] | None = None,
    receipts: list[Path] | None = None,
) -> dict[str, Any]:
    """Enforce promotion or burial and append a ledger event."""
    if action == "promote":
        rule: PromotionRule = spec.promotion
        if (
            not rule.author_may_admit
            and author
            and admitter
            and author == admitter
        ):
            raise GovernanceError(
                f"author-is-not-admitter: author={author!r} cannot admit own work"
            )
        if not rule.allow_synthetic_as_real and measurement_kind == "synthetic":
            raise GovernanceError(
                "synthetic measurement is not a measurement of the real thing; promotion refused"
            )
        if verdict != rule.require_verdict:
            raise GovernanceError(
                f"tier promotion refused: verdict {verdict!r} != required {rule.require_verdict!r}"
            )
        for gate in rule.require_gates:
            if not gate_results.get(gate):
                raise GovernanceError(f"tier promotion refused: gate {gate!r} failed")
        event = ledger.append(
            "promote",
            {
                "campaign_id": spec.campaign_id,
                "verdict": verdict,
                "gates": dict(gate_results),
                "admitter": admitter,
            },
        )
        return {"action": "promote", "event": event}
    if action == "bury":
        rule_b: BurialRule = spec.burial
        missing: list[str] = []
        arts = list(artifacts or [])
        recs = list(receipts or [])
        if rule_b.retain_artifacts:
            for p in arts:
                if not Path(p).exists():
                    missing.append(str(p))
        if rule_b.retain_receipts:
            for p in recs:
                if not Path(p).exists():
                    missing.append(str(p))
        if missing:
            raise GovernanceError(
                "burial-is-not-deletion: retain paths missing after bury: "
                + ", ".join(missing)
            )
        result = {
            "status": rule_b.status_value,
            "retained_artifacts": [str(p) for p in arts],
            "retained_receipts": [str(p) for p in recs],
            "deleted": False,
        }
        event = ledger.append("bury", {"campaign_id": spec.campaign_id, **result})
        return {"action": "bury", "event": event, **result}
    raise GovernanceError(f"unknown governance action {action!r}")
