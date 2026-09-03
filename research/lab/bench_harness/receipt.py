"""Receipt writer — sealed JSON artifacts with schema + content hash."""
from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def content_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(',', ':')).encode()
    return hashlib.sha256(blob).hexdigest()

class ReceiptWriter:

    def __init__(self, schema: str='hawking.lab.receipt.v1'):
        self.schema = schema

    def build(self, *, experiment_id: str, stages: list[dict[str, Any]], measures: list[dict[str, Any]] | None=None, status: str='complete', meta: dict[str, Any] | None=None) -> dict[str, Any]:
        body = {'schema': self.schema, 'ts': _stamp(), 'experiment_id': experiment_id, 'status': status, 'stages': stages, 'measures': measures or [], 'meta': meta or {}}
        body['content_sha256'] = content_hash({k: v for k, v in body.items() if k != 'content_sha256'})
        return body

    def write(self, path: str | Path, receipt: dict[str, Any]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + '.tmp')
        tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
        os.replace(tmp, p)
        return p

    def emit(self, path: str | Path, *, experiment_id: str, stages: list[dict[str, Any]], measures: list[dict[str, Any]] | None=None, status: str='complete', meta: dict[str, Any] | None=None) -> dict[str, Any]:
        receipt = self.build(experiment_id=experiment_id, stages=stages, measures=measures, status=status, meta=meta)
        self.write(path, receipt)
        return receipt
