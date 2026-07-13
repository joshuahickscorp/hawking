#!/usr/bin/env python3.12
"""One-shot, read-only status for live Hawking Doctor workers.

This intentionally does not daemonize or poll. It reads the append-only progress ledgers,
checkpoint metadata, and process table once, then exits so remote status checks do not compete
with training. JSON is the default; pass ``--pretty`` for a compact human view.
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import pathlib
import re
import subprocess
import sys
import time


PROGRESS_GLOBS = (
    "/tmp/aud_*_doctor_progress.jsonl",
    "reports/cron/**/*_doctor_progress.jsonl",
)


def _events(path):
    rows = []
    try:
        with open(path, errors="ignore") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        pass
    return rows


def _last(rows, event):
    return next((row for row in reversed(rows) if row.get("event") == event), None)


def _label(path):
    name = pathlib.Path(path).name
    match = re.search(r"aud_(.+)_doctor_progress\.jsonl$", name)
    if match:
        return match.group(1)
    return name.removesuffix("_doctor_progress.jsonl")


def _processes():
    found = {}
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,state=,etime=,%cpu=,rss=,command="],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except Exception:
        return found
    for line in output.splitlines():
        if "tools/condense/doctor.py lora" not in line:
            continue
        fields = line.strip().split(None, 6)
        if len(fields) != 7:
            continue
        pid, ppid, state, elapsed, cpu, rss, command = fields
        match = re.search(r"/tmp/aud_(.+?)_rbase\.safetensors", command)
        label = match.group(1) if match else f"pid-{pid}"
        found[label] = {
            "pid": int(pid), "ppid": int(ppid), "state": state, "elapsed": elapsed,
            "cpu_percent": float(cpu), "rss_gib": round(int(rss) / (1024 * 1024), 3),
            "command": command,
        }
    return found


def _checkpoint(label):
    candidates = [
        f"/tmp/aud_{label}_adapter.safetensors",
        f"/tmp/aud_{label}_adapter.latest.safetensors",
    ]
    path = next((candidate for candidate in candidates if os.path.isfile(candidate)), None)
    if not path:
        return None
    record = {
        "path": path,
        "bytes": os.path.getsize(path),
        "updated_unix": os.path.getmtime(path),
    }
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as handle:
            record["tensor_count"] = len(handle.keys())
            record["metadata"] = handle.metadata()
    except Exception as exc:
        record["metadata_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _estimated_eta(rows, now):
    train = _last(rows, "train")
    if train and isinstance(train.get("eta_s"), (int, float)):
        return max(0.0, float(train["eta_s"])), "train-heartbeat"
    evals = [row for row in rows if row.get("event") == "eval" and row.get("step") is not None]
    if len(evals) < 2:
        return None, None
    first, last = evals[-2], evals[-1]
    delta_steps = int(last["step"]) - int(first["step"])
    delta_s = float(last.get("ts", 0)) - float(first.get("ts", 0))
    total = int(last.get("steps") or 0)
    if delta_steps <= 0 or delta_s <= 0 or total <= 0:
        return None, None
    seconds_per_step = delta_s / delta_steps
    # Include time since the last visible eval: the worker can be partway through the next interval.
    projected_end = float(last["ts"]) + max(0, total - 1 - int(last["step"])) * seconds_per_step
    return max(0.0, projected_end - now), "eval-cadence"


def snapshot():
    now = time.time()
    paths = []
    for pattern in PROGRESS_GLOBS:
        paths.extend(glob.glob(pattern, recursive=True))
    processes = _processes()
    labels = sorted(set(processes) | {_label(path) for path in paths})
    workers = []
    for label in labels:
        path = next((p for p in paths if _label(p) == label), None)
        rows = _events(path) if path else []
        latest = rows[-1] if rows else None
        evaluation = _last(rows, "eval")
        final = _last(rows, "final")
        phase = _last(rows, "phase")
        eta_s, eta_source = _estimated_eta(rows, now)
        base_ppl = None
        heldout_ppl = evaluation.get("heldout_ppl") if evaluation else None
        checkpoint = _checkpoint(label)
        if checkpoint:
            metadata = checkpoint.get("metadata") or {}
            try:
                base_ppl = float(metadata.get("base_ppl"))
            except (TypeError, ValueError):
                pass
        quality = None
        if base_ppl is not None and heldout_ppl is not None:
            quality = "improved" if float(heldout_ppl) < base_ppl else (
                "flat" if float(heldout_ppl) == base_ppl else "regressed"
            )
        workers.append({
            "label": label,
            "process": processes.get(label),
            "progress_path": path,
            "progress_events": len(rows),
            "latest_event": latest,
            "latest_eval": evaluation,
            "latest_phase": phase,
            "final": final,
            "checkpoint": checkpoint,
            "internal_validation_status": quality,
            "eta_s": eta_s,
            "eta_source": eta_source,
            "progress_age_s": (now - float(latest["ts"])) if latest and latest.get("ts") else None,
        })
    return {
        "schema": "hawking.doctor_status.v1",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "worker_count": len(workers),
        "live_worker_count": sum(1 for worker in workers if worker["process"]),
        "workers": workers,
    }


def pretty(report):
    if not report["workers"]:
        print("Doctor: no workers or progress ledgers found")
        return
    for worker in report["workers"]:
        proc = worker.get("process") or {}
        event = worker.get("latest_event") or {}
        step = event.get("step")
        total = event.get("steps")
        progress = f"step {int(step)+1}/{total}" if step is not None and total else event.get("phase", "no progress")
        eta = worker.get("eta_s")
        eta_text = f", ETA {eta/3600:.1f}h ({worker['eta_source']})" if eta is not None else ", ETA pending"
        checkpoint = worker.get("checkpoint")
        ckpt = f", checkpoint {checkpoint['bytes']/1e6:.1f}MB" if checkpoint else ", no checkpoint"
        validation = worker.get("internal_validation_status") or "pending"
        print(f"{worker['label']}: pid={proc.get('pid','-')} cpu={proc.get('cpu_percent','-')}% "
              f"rss={proc.get('rss_gib','-')}GiB, {progress}{eta_text}{ckpt}, heldout={validation}")


if __name__ == "__main__":
    report = snapshot()
    if "--pretty" in sys.argv:
        pretty(report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
