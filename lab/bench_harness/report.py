"""Report renderer — markdown (+ optional JSON) from run status + measures."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

class ReportRenderer:

    def render_md(self, *, title: str, experiment_id: str, status: dict[str, Any], stages: list[dict[str, Any]], measures: list[dict[str, Any]] | None=None, receipt_path: str | None=None) -> str:
        lines = [f'# {title}', '', f'- experiment_id: `{experiment_id}`', f'- state: **{status.get('state', 'unknown')}**', f'- current_stage: `{status.get('current_stage', '')}`', f'- uptime_seconds: {status.get('uptime_seconds', '')}', '', '## Stages', '', '| stage | state | rc | seconds | note |', '|---|---|---:|---:|---|']
        for s in stages:
            lines.append(f'| {s.get('id', '')} | {s.get('state', '')} | {s.get('rc', '')} | {s.get('seconds', '')} | {s.get('note', '')} |')
        lines.append('')
        if measures:
            lines += ['## Measures', '']
            metrics = [m for m in measures if m.get('kind') == 'metric']
            if metrics:
                lines += ['| name | value |', '|---|---|']
                for m in metrics:
                    lines.append(f'| {m.get('name')} | {m.get('value')} |')
                lines.append('')
            else:
                lines.append(f'_{len(measures)} measure rows in JSONL._')
                lines.append('')
        if receipt_path:
            lines += [f'Receipt: `{receipt_path}`', '']
        return '\n'.join(lines)

    def write_md(self, path: str | Path, text: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding='utf-8')
        return p

    def write_json(self, path: str | Path, obj: Any) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
        return p
