#!/usr/bin/env python3
"""Clean one-line progress tail for rwkv7_qat.py events.jsonl."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tail RWKV QAT progress as clean one-liners.")
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="QAT run directory")
    ap.add_argument("--follow", "-f", action="store_true", help="follow events.jsonl")
    ap.add_argument("--target-step", type=int, default=0, help="override final optimizer step")
    ap.add_argument(
        "--next-checkpoint",
        action="store_true",
        help="show progress to the next checkpoint gate instead of the full run",
    )
    ap.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="checkpoint interval used by --next-checkpoint",
    )
    ap.add_argument(
        "--refresh-seconds",
        type=int,
        default=60,
        help="with --follow, refresh wall-clock ETA even when no new event landed",
    )
    ap.add_argument(
        "--steps-only",
        action="store_true",
        help="with --follow, print only when a new training step lands",
    )
    return ap.parse_args()


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone()


def resume_step(config: dict[str, Any]) -> int:
    if int(config.get("resume_step") or 0) > 0:
        return int(config["resume_step"])
    resume_from = str(config.get("resume_from") or "")
    match = re.search(r"step_0*(\d+)", resume_from)
    return int(match.group(1)) if match else 0


def infer_target_step(run_dir: Path, override: int) -> int | None:
    if override > 0:
        return override

    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return None

    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = Path(str(config["data"]))
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    if not data_path.exists():
        return None

    rows = sum(1 for _ in data_path.open("r", encoding="utf-8"))
    if int(config.get("max_rows") or 0) > 0:
        rows = min(rows, int(config["max_rows"]))

    grad_accum = int(config.get("grad_accum") or 1)
    epochs = int(config.get("epochs") or 1)
    start_step = resume_step(config)
    consumed_rows = min(start_step * grad_accum, rows)
    remaining_epoch0 = max(rows - consumed_rows, 0)
    later_rows = max(epochs - 1, 0) * rows
    return start_step + (remaining_epoch0 + later_rows) // grad_accum


def next_checkpoint_step(step: int, interval: int) -> int:
    interval = max(interval, 1)
    return ((step + 1 + interval - 1) // interval) * interval


def load_segments(events_path: Path) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_step: int | None = None

    if not events_path.exists():
        return segments

    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        step = int(event.get("step") or 0)
        if current and previous_step is not None and step <= previous_step:
            segments.append(current)
            current = []
        current.append(event)
        previous_step = step

    if current:
        segments.append(current)
    return segments


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "?"
    minutes = int(round(seconds / 60))
    days, rem_minutes = divmod(minutes, 24 * 60)
    hours, mins = divmod(rem_minutes, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def rate_seconds(segment: list[dict[str, Any]]) -> float | None:
    intervals: list[float] = []
    for prev, cur in zip(segment, segment[1:]):
        prev_step = int(prev.get("step") or 0)
        cur_step = int(cur.get("step") or 0)
        if cur_step <= prev_step:
            continue
        dt = (parse_timestamp(cur["timestamp"]) - parse_timestamp(prev["timestamp"])).total_seconds()
        if dt > 0:
            intervals.append(dt / (cur_step - prev_step))
    if not intervals:
        return None
    return sum(intervals[-5:]) / min(len(intervals), 5)


def ema_trend(segment: list[dict[str, Any]], window: int = 5) -> tuple[float | None, str]:
    if len(segment) < 2:
        return None, "new"

    current = float(segment[-1].get("loss_ema") or segment[-1].get("loss") or 0.0)
    start_index = max(0, len(segment) - window - 1)
    previous = float(segment[start_index].get("loss_ema") or segment[start_index].get("loss") or current)
    delta = current - previous

    if delta <= -0.03:
        label = "improving"
    elif delta >= 0.03:
        label = "rising"
    else:
        label = "flat"
    return delta, label


def render(segment: list[dict[str, Any]], target_step: int | None) -> str:
    event = segment[-1]
    step = int(event.get("step") or 0)
    loss = float(event.get("loss") or 0.0)
    loss_ema = float(event.get("loss_ema") or loss)
    ppl = float(event.get("ppl_ema") or math.exp(min(loss_ema, 20.0)))
    tok_s = float(event.get("tok_s") or 0.0)
    stamp = parse_timestamp(event["timestamp"])
    now = datetime.now(stamp.tzinfo)
    elapsed_since_event = max(0.0, (now - stamp).total_seconds())
    sec_per_step = rate_seconds(segment)
    trend_delta, trend_label = ema_trend(segment)

    if target_step:
        remaining_steps = max(target_step - step, 0)
        progress = f"{step}/{target_step}"
        pct_completed = step / target_step * 100
    else:
        remaining_steps = None
        progress = str(step)
        pct_completed = None

    step_phase = "in-step ?"
    if sec_per_step is not None and step > 0:
        if target_step is None or step < target_step:
            next_step = step + 1
            in_step_fraction = min(elapsed_since_event / sec_per_step, 0.999)
            elapsed = fmt_duration(elapsed_since_event)
            if elapsed_since_event > sec_per_step:
                overdue = fmt_duration(elapsed_since_event - sec_per_step)
                step_phase = f"next {next_step} overdue +{overdue} (elapsed {elapsed})"
            else:
                step_phase = f"next {next_step} ~{in_step_fraction * 100:.0f}% (elapsed {elapsed})"
        else:
            in_step_fraction = 0.0
            step_phase = "target reached"
    else:
        in_step_fraction = 0.0

    if pct_completed is None:
        pct = "?"
    else:
        pct = f"{pct_completed:.1f}%"

    if sec_per_step is not None and remaining_steps is not None:
        if remaining_steps <= 0:
            remaining_seconds = 0.0
        else:
            current_step_left = max(sec_per_step - elapsed_since_event, 0.0)
            later_steps_left = max(remaining_steps - 1, 0) * sec_per_step
            remaining_seconds = current_step_left + later_steps_left
        eta = (now + timedelta(seconds=remaining_seconds)).strftime("%a %b %-d %I:%M%p")
        left = fmt_duration(remaining_seconds)
        step_time = fmt_duration(sec_per_step)
    else:
        eta = "?"
        left = "?"
        step_time = "?"

    if trend_delta is None:
        trend = "new"
    else:
        trend = f"{trend_label} {trend_delta:+.3f}/5"

    return (
        f"step {progress} ({pct}) | batch_loss {loss:.3f} | ema {loss_ema:.3f} | "
        f"ppl {ppl:.0f} | tok/s {tok_s:.1f} | step_time {step_time} | "
        f"ETA {eta} ({left} left)"
    )


def render_latest(segment: list[dict[str, Any]], args: argparse.Namespace, static_target: int | None) -> str:
    if args.next_checkpoint:
        step = int(segment[-1].get("step") or 0)
        target_step = next_checkpoint_step(step, args.checkpoint_interval)
    else:
        target_step = static_target
    return render(segment, target_step)


def print_latest(run_dir: Path, args: argparse.Namespace, static_target: int | None) -> int:
    events_path = run_dir / "events.jsonl"
    segments = load_segments(events_path)
    if not segments:
        print(f"waiting for {events_path}", flush=True)
        return 0
    latest = segments[-1]
    print(render_latest(latest, args, static_target), flush=True)
    return sum(len(segment) for segment in segments)


def main() -> int:
    args = parse_args()
    if args.steps_only:
        args.refresh_seconds = 0
    run_dir = Path(args.run_dir).expanduser().resolve()
    static_target = None if args.next_checkpoint else infer_target_step(run_dir, args.target_step)
    printed = print_latest(run_dir, args, static_target)

    if not args.follow:
        return 0

    events_path = run_dir / "events.jsonl"
    last_refresh = time.monotonic()
    while True:
        time.sleep(5)
        segments = load_segments(events_path)
        count = sum(len(segment) for segment in segments)
        refresh_due = args.refresh_seconds > 0 and (time.monotonic() - last_refresh) >= args.refresh_seconds
        if segments and (count > printed or refresh_due):
            print(render_latest(segments[-1], args, static_target), flush=True)
            printed = count
            last_refresh = time.monotonic()


if __name__ == "__main__":
    raise SystemExit(main())
