"""Measurement recorder — one JSONL stream per experiment run."""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

class MeasurementRecorder:

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, 'a', encoding='utf-8')

    def close(self) -> None:
        if self._fh and (not self._fh.closed):
            self._fh.close()

    def __enter__(self) -> 'MeasurementRecorder':
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def record(self, kind: str, **fields: Any) -> dict[str, Any]:
        row = {'ts': _stamp(), 'kind': kind, **fields}
        self._fh.write(json.dumps(row, sort_keys=True, default=str) + '\n')
        self._fh.flush()
        os.fsync(self._fh.fileno())
        return row

    def stage_start(self, stage_id: str, **extra: Any) -> dict[str, Any]:
        return self.record('stage_start', stage=stage_id, t0=time.time(), **extra)

    def stage_end(self, stage_id: str, *, rc: int, seconds: float, state: str, **extra: Any) -> dict[str, Any]:
        return self.record('stage_end', stage=stage_id, rc=rc, seconds=round(seconds, 3), state=state, **extra)

    def metric(self, name: str, value: Any, **extra: Any) -> dict[str, Any]:
        return self.record('metric', name=name, value=value, **extra)

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
