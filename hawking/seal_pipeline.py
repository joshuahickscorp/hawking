"""Deterministic seal receipt writer — Kimi should not hand-write boilerplate."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def write_seal(
    workspace: Path | str,
    *,
    workunit_id: str,
    seal_id: str,
    title: str,
    phase: str,
    change: Mapping[str, Any],
    validation: Mapping[str, Any],
    claim_boundary: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    ws = Path(workspace)
    path = ws / "receipts" / "future" / "workunits" / f"{seal_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc: Dict[str, Any] = {
        "receipt_id": seal_id,
        "schema": "hawking.workunit.seal.v1",
        "workunit_id": workunit_id,
        "phase": phase,
        "title": title,
        "at": datetime.now(timezone.utc).isoformat(),
        "written_at_unix": int(time.time()),
        "change": dict(change),
        "validation": dict(validation),
        "claim_boundary": claim_boundary,
        "generator": "hawking.seal_pipeline",
    }
    if extra:
        doc.update(dict(extra))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(path)
    return path
