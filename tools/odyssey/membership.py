"""Content-addressed membership records for training items and corpora."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.odyssey import SCHEMA_MEMBERSHIP
from tools.odyssey.dedup import content_sha256
from tools.odyssey.normalize import extract_comparison_text, normalize_text


def item_content_address(item: dict[str, Any]) -> str:
    """Canonical content address: sha256 over sorted-keys JSON of the item body.

    Drops ephemeral keys (status, membership_*) so re-wrapping does not change
    the address of the underlying payload.
    """
    skip = {
        "content_sha256",
        "membership_status",
        "contamination_hits",
        "normalized_text",
        "exact_text_sha256",
    }
    body = {k: v for k, v in item.items() if k not in skip}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class ItemMembership:
    content_sha256: str
    exact_text_sha256: str
    source_id: str | None
    status: str  # admitted | rejected_exact_dup | rejected_near_dup | rejected_contamination
    reasons: list[str] = field(default_factory=list)
    contamination: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CorpusMembership:
    schema: str = SCHEMA_MEMBERSHIP
    corpus_id: str = ""
    role: str = "train"  # train | fixture | eval | unknown
    created_at: str = ""
    source_path: str = ""
    n_input: int = 0
    n_admitted: int = 0
    n_rejected_exact_dup: int = 0
    n_rejected_near_dup: int = 0
    n_rejected_contamination: int = 0
    membership_sha256: str = ""
    items: list[dict[str, Any]] = field(default_factory=list)
    invariants: list[str] = field(
        default_factory=lambda: [
            "every admitted item is content-addressed before a step may read it",
            "no admitted item overlaps any evaluation set (exact or near-dup)",
            "licence is recorded per corpus, not per collection run",
        ]
    )
    licence: str | None = None
    note: str | None = None

    def seal(self) -> None:
        payload = {
            "corpus_id": self.corpus_id,
            "items": [
                {
                    "content_sha256": it["content_sha256"],
                    "exact_text_sha256": it["exact_text_sha256"],
                    "status": it["status"],
                }
                for it in self.items
            ],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.membership_sha256 = hashlib.sha256(blob).hexdigest()
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def to_dict(self) -> dict[str, Any]:
        self.seal()
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_item_record(
    item: dict[str, Any],
    *,
    status: str,
    reasons: list[str] | None = None,
    contamination: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    text = extract_comparison_text(item)
    return {
        "content_sha256": item_content_address(item),
        "exact_text_sha256": content_sha256(text),
        "normalized_preview": normalize_text(text)[:120],
        "source_id": item.get("id"),
        "status": status,
        "reasons": reasons or [],
        "contamination": contamination or [],
    }
