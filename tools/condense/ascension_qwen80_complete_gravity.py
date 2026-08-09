#!/usr/bin/env python3
"""Wait for the verified Qwen80 source then build its complete binary baseline."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.operators.ascension_qwen30_complete_gravity import CompleteBinaryGravity


MODEL_DIR = ROOT / "workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next"
SOURCE_AUDIT = ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen80-acquisition/QWEN80_SOURCE_BODY_AUDIT_CANDIDATE.json"
OUTPUT_ROOT = ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-gravity"
SCHEMA = "hawking.ascension.qwen80_complete_binary_gravity.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _waiting_status() -> None:
    path = OUTPUT_ROOT / "QWEN80_COMPLETE_GRAVITY_STATUS.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = 0
    try:
        import json
        prior = int(json.loads(path.read_text()).get("heartbeat", 0))
    except Exception:
        pass
    payload = {
        "schema": SCHEMA, "recorded_at": _now(), "pid": os.getpid(), "heartbeat": prior + 1,
        "phase": "BLOCKED_BY_EXACT_DEPENDENCY",
        "dependency": "QWEN80_SOURCE_BODY_AUDIT_CANDIDATE.json must exist and verify before complete Gravity packing",
        "source_audit_path": str(SOURCE_AUDIT),
        "claim_boundary": {"does_not_treat_downloaded_bytes_as_verified_source": True, "no_qwen80_gravity_artifact_before_source_audit": True, "not_manager_or_tournament_qualified": True},
    }
    fd, temporary = tempfile.mkstemp(prefix=".status.", dir=path.parent)
    try:
        import json
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def main() -> int:
    while not SOURCE_AUDIT.is_file():
        _waiting_status()
        time.sleep(60)
    worker = CompleteBinaryGravity(
        model_dir=MODEL_DIR, source_audit=SOURCE_AUDIT, root=OUTPUT_ROOT,
        repository="Qwen/Qwen3-Coder-Next", model_id="Qwen3-Coder-Next-80B",
        artifact_prefix="QWEN80", schema=SCHEMA,
    )
    return worker.run(max_tensors=16)


if __name__ == "__main__":
    raise SystemExit(main())
