"""Declarative experiment spec — the only way to describe a lab run."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
SPEC_SCHEMA = 'hawking.lab.experiment.v1'
REQUIRED = ('schema', 'id', 'stages')

@dataclass
class Stage:
    id: str
    argv: list[str] = field(default_factory=list)
    shell: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    on_fail: str = 'abort'
    measure_keys: list[str] = field(default_factory=list)
    timeout_s: int | None = None
    skip_if_done: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'Stage':
        if 'id' not in d:
            raise ValueError('stage missing id')
        return cls(id=str(d['id']), argv=[str(x) for x in d.get('argv', [])], shell=d.get('shell'), env={str(k): str(v) for k, v in (d.get('env') or {}).items()}, cwd=d.get('cwd'), on_fail=str(d.get('on_fail', 'abort')), measure_keys=[str(x) for x in d.get('measure_keys', [])], timeout_s=d.get('timeout_s'), skip_if_done=bool(d.get('skip_if_done', True)))

@dataclass
class ExperimentSpec:
    schema: str
    id: str
    stages: list[Stage]
    title: str = ''
    artifact_dir: str = 'artifacts/runs/lab'
    status_name: str = 'status.json'
    log_name: str = 'chain.log'
    pid_name: str = 'pid'
    measures_name: str = 'measures.jsonl'
    receipt_name: str = 'receipt.json'
    report_name: str = 'report.md'
    pause_flag: str = 'artifacts/runs/PAUSE'
    resume_flag: str = 'artifacts/runs/RESUME'
    pause_poll_s: float = 10.0
    matrix: list[dict[str, Any]] = field(default_factory=list)
    fixtures: list[dict[str, Any]] = field(default_factory=list)
    receipt_schema: str = 'hawking.lab.receipt.v1'
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'ExperimentSpec':
        validate_spec(d)
        stages = [Stage.from_dict(s) for s in d['stages']]
        return cls(schema=d['schema'], id=str(d['id']), stages=stages, title=str(d.get('title') or d['id']), artifact_dir=str(d.get('artifact_dir', 'artifacts/runs/lab')), status_name=str(d.get('status_name', 'status.json')), log_name=str(d.get('log_name', 'chain.log')), pid_name=str(d.get('pid_name', 'pid')), measures_name=str(d.get('measures_name', 'measures.jsonl')), receipt_name=str(d.get('receipt_name', 'receipt.json')), report_name=str(d.get('report_name', 'report.md')), pause_flag=str(d.get('pause_flag', 'artifacts/runs/PAUSE')), resume_flag=str(d.get('resume_flag', 'artifacts/runs/RESUME')), pause_poll_s=float(d.get('pause_poll_s', 10.0)), matrix=list(d.get('matrix') or []), fixtures=list(d.get('fixtures') or []), receipt_schema=str(d.get('receipt_schema', 'hawking.lab.receipt.v1')), meta=dict(d.get('meta') or {}))

def validate_spec(d: dict[str, Any]) -> None:
    if not isinstance(d, dict):
        raise ValueError('spec must be a JSON object')
    for k in REQUIRED:
        if k not in d:
            raise ValueError(f'spec missing required field {k!r}')
    if d['schema'] != SPEC_SCHEMA:
        raise ValueError(f'unsupported spec schema {d['schema']!r}; want {SPEC_SCHEMA}')
    if not isinstance(d['stages'], list) or not d['stages']:
        raise ValueError('spec.stages must be a non-empty list')
    for i, s in enumerate(d['stages']):
        if not isinstance(s, dict) or 'id' not in s:
            raise ValueError(f'stages[{i}] must be an object with id')
        if not s.get('argv') and (not s.get('shell')):
            raise ValueError(f'stages[{i}] ({s.get('id')}) needs argv or shell')

def load_spec(path: str | Path) -> ExperimentSpec:
    p = Path(path)
    raw = json.loads(p.read_text(encoding='utf-8'))
    return ExperimentSpec.from_dict(raw)
