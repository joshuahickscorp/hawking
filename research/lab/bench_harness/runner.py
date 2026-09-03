"""Experiment runner — fixtures, matrix, stages, pause/resume, receipts."""
from __future__ import annotations
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .measure import MeasurementRecorder
from .receipt import ReceiptWriter
from .report import ReportRenderer
from .spec import ExperimentSpec, Stage
REPO = Path(__file__).resolve().parents[2]

def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def free_disk_gb(path: Path=REPO) -> int | None:
    try:
        st = os.statvfs(path)
        return int(st.f_bavail * st.f_frsize / 1024 ** 3)
    except OSError:
        return None

class Runner:

    def __init__(self, spec: ExperimentSpec, *, root: Path | None=None, dry_run: bool=False):
        self.spec = spec
        self.root = (root or REPO).resolve()
        self.dry_run = dry_run
        self.artifact_dir = (self.root / spec.artifact_dir).resolve()
        self.status_path = self.artifact_dir / spec.status_name
        self.log_path = self.artifact_dir / spec.log_name
        self.pid_path = self.artifact_dir / spec.pid_name
        self.measures_path = self.artifact_dir / spec.measures_name
        self.receipt_path = self.artifact_dir / spec.receipt_name
        self.report_path = self.artifact_dir / spec.report_name
        self.pause_flag = self.root / spec.pause_flag
        self.resume_flag = self.root / spec.resume_flag
        self._start = time.time()
        self._stage_results: list[dict[str, Any]] = []
        self.receipts = ReceiptWriter(spec.receipt_schema)
        self.reports = ReportRenderer()

    def log(self, msg: str) -> None:
        line = f'[{_stamp()}] {msg}'
        print(line, flush=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')

    def write_status(self, stage: str, state: str, note: str='') -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {'ts': _stamp(), 'pipeline_pid': os.getpid(), 'experiment_id': self.spec.id, 'current_stage': stage, 'state': state, 'note': note, 'uptime_seconds': int(time.time() - self._start), 'free_disk_gb': free_disk_gb(self.root)}
        tmp = self.status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        os.replace(tmp, self.status_path)

    def _honor_pause(self) -> None:
        if not self.pause_flag.exists():
            return
        self.log(f'PAUSE flag present at {self.pause_flag}; waiting for resume')
        self.write_status('paused', 'paused', 'PAUSE flag')
        while self.pause_flag.exists() and (not self.resume_flag.exists()):
            time.sleep(self.spec.pause_poll_s)
        if self.resume_flag.exists():
            try:
                self.resume_flag.unlink()
            except OSError:
                pass
            try:
                self.pause_flag.unlink()
            except OSError:
                pass
            self.log('resume signal consumed; continuing')

    def _done_marker(self, stage_id: str) -> Path:
        return self.artifact_dir / f'.stage_{stage_id}.done'

    def _expand(self, text: str, slot: dict[str, Any]) -> str:
        out = text
        for k, v in slot.items():
            out = out.replace('{' + str(k) + '}', str(v))
        return out

    def _run_stage(self, stage: Stage, slot: dict[str, Any], measures: MeasurementRecorder) -> dict[str, Any]:
        self._honor_pause()
        sid = stage.id
        if slot:
            sid = f'{stage.id}__' + '_'.join((f'{k}={v}' for k, v in sorted(slot.items())))
        if stage.skip_if_done and self._done_marker(sid).exists():
            self.log(f'SKIP stage {sid} (done marker)')
            result = {'id': sid, 'state': 'skipped_done', 'rc': 0, 'seconds': 0, 'note': 'done marker'}
            self._stage_results.append(result)
            return result
        self.log(f'=== STAGE {sid} ===')
        self.write_status(sid, 'running')
        measures.stage_start(sid, slot=slot or None)
        env = dict(os.environ)
        env.update({k: self._expand(v, slot) for k, v in stage.env.items()})
        cwd = self.root / self._expand(stage.cwd, slot) if stage.cwd else self.root
        if stage.shell:
            cmd = self._expand(stage.shell, slot)
            argv = ['/bin/bash', '-lc', cmd]
        else:
            argv = [self._expand(a, slot) for a in stage.argv]
        t0 = time.time()
        if self.dry_run:
            self.log(f'DRY-RUN would exec: {argv!r} cwd={cwd}')
            rc, state, note = (0, 'dry_run', 'dry-run')
        else:
            try:
                p = subprocess.run(argv, cwd=str(cwd), env=env, timeout=stage.timeout_s)
                rc = int(p.returncode)
            except subprocess.TimeoutExpired:
                rc, note = (124, 'timeout')
                state = 'failed'
                seconds = time.time() - t0
                measures.stage_end(sid, rc=rc, seconds=seconds, state=state, note=note)
                result = {'id': sid, 'state': state, 'rc': rc, 'seconds': round(seconds, 3), 'note': note}
                self._stage_results.append(result)
                self.write_status(sid, state, note)
                return result
            except OSError as e:
                rc, note = (127, f'spawn failed: {e}')
                state = 'failed'
                seconds = time.time() - t0
                measures.stage_end(sid, rc=rc, seconds=seconds, state=state, note=note)
                result = {'id': sid, 'state': state, 'rc': rc, 'seconds': round(seconds, 3), 'note': note}
                self._stage_results.append(result)
                self.write_status(sid, state, note)
                return result
            state = 'done' if rc == 0 else 'failed'
            note = '' if rc == 0 else f'rc={rc}'
        seconds = time.time() - t0
        measures.stage_end(sid, rc=rc, seconds=seconds, state=state, note=note)
        if state == 'done':
            self._done_marker(sid).write_text(_stamp() + '\n', encoding='utf-8')
        result = {'id': sid, 'state': state, 'rc': rc, 'seconds': round(seconds, 3), 'note': note}
        self._stage_results.append(result)
        self.write_status(sid, state, note)
        self.log(f'stage {sid} -> {state} rc={rc} ({seconds:.1f}s)')
        return result

    def _iter_slots(self) -> list[dict[str, Any]]:
        if not self.spec.matrix:
            return [{}]
        return [dict(x) for x in self.spec.matrix]

    def run(self) -> int:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(os.getpid()) + '\n', encoding='utf-8')
        self.write_status('init', 'running')
        self.log(f'lab harness start experiment={self.spec.id} dry_run={self.dry_run}')
        for fix in self.spec.fixtures:
            kind = fix.get('kind')
            if kind == 'mkdir':
                Path(self.root / fix['path']).mkdir(parents=True, exist_ok=True)
            elif kind == 'env':
                os.environ[str(fix['key'])] = str(fix['value'])
            elif kind == 'shell' and (not self.dry_run):
                subprocess.run(['/bin/bash', '-lc', fix['shell']], cwd=str(self.root), check=False)
        overall = 0
        with MeasurementRecorder(self.measures_path) as measures:
            for slot in self._iter_slots():
                for stage in self.spec.stages:
                    result = self._run_stage(stage, slot, measures)
                    if result['state'] == 'failed':
                        overall = result['rc'] or 1
                        if stage.on_fail == 'abort':
                            self.log(f'aborting on stage failure {result['id']}')
                            self._finalize('failed', measures)
                            return overall
                        if stage.on_fail == 'skip_rest':
                            self.log(f'skip_rest after {result['id']}')
                            break
            self._finalize('complete' if overall == 0 else 'failed', measures)
        return overall

    def _finalize(self, status: str, measures: MeasurementRecorder) -> None:
        rows = measures.read_all()
        receipt = self.receipts.emit(self.receipt_path, experiment_id=self.spec.id, stages=self._stage_results, measures=[r for r in rows if r.get('kind') == 'metric'], status=status, meta=self.spec.meta)
        st = {'state': status, 'current_stage': self._stage_results[-1]['id'] if self._stage_results else '', 'uptime_seconds': int(time.time() - self._start)}
        md = self.reports.render_md(title=self.spec.title or self.spec.id, experiment_id=self.spec.id, status=st, stages=self._stage_results, measures=rows, receipt_path=str(self.receipt_path.relative_to(self.root)) if self.receipt_path.is_relative_to(self.root) else str(self.receipt_path))
        self.reports.write_md(self.report_path, md)
        self.write_status('final', status, receipt.get('content_sha256', '')[:12])
        self.log(f'lab harness finished status={status} receipt={self.receipt_path}')
