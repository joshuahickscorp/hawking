#!/usr/bin/env python3.12
"""Receipt interface — one seal/verify path for campaign outcomes.

Receipts are scientific records. Encoding is canonical JSON; writes are atomic.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


RECEIPT_SCHEMA = "hawking.condense.campaign_receipt.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {k: v for k, v in value.items() if k != "seal_sha256"}
    return {
        **unsigned,
        "seal_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }


def verify_receipt(value: Mapping[str, Any], *, label: str = "receipt") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a JSON object")
    recorded = value.get("seal_sha256")
    expected = seal_receipt(dict(value))["seal_sha256"]
    if recorded != expected:
        raise ValueError(
            f"{label} seal mismatch: recorded={recorded!r} expected={expected}"
        )
    return dict(value)


@dataclass
class Receipt:
    campaign_id: str
    status: str
    phase: str
    summary: Mapping[str, Any]
    at: str = ""
    schema: str = RECEIPT_SCHEMA
    reproduction: str = ""
    artifacts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": self.schema,
            "campaign_id": self.campaign_id,
            "status": self.status,
            "phase": self.phase,
            "summary": dict(self.summary),
            "at": self.at or _utc_now(),
            "reproduction": self.reproduction,
            "artifacts": list(self.artifacts),
        }
        return seal_receipt(body)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Receipt":
        verify_receipt(raw)
        return cls(
            campaign_id=str(raw["campaign_id"]),
            status=str(raw.get("status") or ""),
            phase=str(raw.get("phase") or ""),
            summary=dict(raw.get("summary") or {}),
            at=str(raw.get("at") or ""),
            schema=str(raw.get("schema") or RECEIPT_SCHEMA),
            reproduction=str(raw.get("reproduction") or ""),
            artifacts=tuple(str(x) for x in (raw.get("artifacts") or ())),
        )


class ReceiptStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, campaign_id: str) -> Path:
        safe = campaign_id.replace("/", "_")
        return self.root / f"{safe}.receipt.json"

    def write(self, receipt: Receipt | Mapping[str, Any]) -> Path:
        if isinstance(receipt, Receipt):
            payload = receipt.to_dict()
            campaign_id = receipt.campaign_id
        else:
            payload = seal_receipt(dict(receipt))
            campaign_id = str(payload.get("campaign_id") or "unknown")
        path = self.path_for(campaign_id)
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return path

    def read(self, campaign_id: str) -> dict[str, Any]:
        path = self.path_for(campaign_id)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return verify_receipt(raw, label=str(path))
