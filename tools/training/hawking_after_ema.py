#!/usr/bin/env python3
"""One-command post-EMA handoff for the Hawking/Dismantle RWKV7 arc.

The script waits for the training EMA to cross a target, freezes the next stable
checkpoint, optionally stops the live training rail to free MPS, runs
deterministic verification, exports a release-shaped HF directory, and writes the
next-frontier queue.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts/lowbit_rwkv7/hawking_arc"


def python_cmd() -> str:
    venv = ROOT / ".venv-rwkv/bin/python"
    return str(venv) if venv.exists() else sys.executable


def now_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(obj)
    return rows


def latest_segment(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segment: list[dict[str, Any]] = []
    previous: int | None = None
    for event in events:
        try:
            step = int(event.get("step") or 0)
        except (TypeError, ValueError):
            continue
        if segment and previous is not None and step <= previous:
            segment = []
        event = dict(event)
        event["step"] = step
        segment.append(event)
        previous = step
    return segment


def fmt_event(event: dict[str, Any] | None) -> str:
    if not event:
        return "no events yet"
    return (
        f"step={event.get('step')} loss={float(event.get('loss') or 0.0):.4f} "
        f"ema={float(event.get('loss_ema') or 0.0):.4f} "
        f"ppl={float(event.get('ppl_ema') or 0.0):.2f} "
        f"tok_s={float(event.get('tok_s') or 0.0):.2f}"
    )


def first_threshold_event(segment: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    for event in segment:
        try:
            ema = float(event.get("loss_ema"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(ema) and ema <= threshold:
            return event
    return None


def ceil_to_interval(step: int, interval: int) -> int:
    if interval <= 1:
        return step
    return ((step + interval - 1) // interval) * interval


def checkpoint_dir(run_dir: Path, step: int) -> Path:
    return run_dir / f"step_{step:06d}"


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def wait_for_stable_checkpoint(path: Path, min_bytes: int, poll_seconds: int) -> None:
    state = path / "state_dict.pt"
    last_size = -1
    stable = 0
    while True:
        size = file_size(state)
        if size >= min_bytes:
            if size == last_size:
                stable += 1
            else:
                stable = 0
                last_size = size
            if stable >= 2:
                return
        print(f"[wait] checkpoint not stable yet: {state} size={size}", flush=True)
        time.sleep(poll_seconds)


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def run_command(
    cmd: list[str],
    log_path: Path,
    env: dict[str, str],
    cwd: Path = ROOT,
    check: bool = True,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {' '.join(cmd)}")
    print(f"[run] log={log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)} (log={log_path})")
    return proc.returncode


def process_rows() -> list[tuple[int, int, str]]:
    try:
        out = subprocess.check_output(
            ["ps", "ax", "-o", "pid=", "-o", "ppid=", "-o", "command="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows: list[tuple[int, int, str]] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    return rows


def stop_processes(run_dir: Path, stop_autocycle: bool) -> list[dict[str, Any]]:
    mine = os.getpid()
    stopped: list[dict[str, Any]] = []
    run_dir_s = str(run_dir)
    try:
        run_dir_rel = str(run_dir.relative_to(ROOT))
    except ValueError:
        run_dir_rel = run_dir_s
    for pid, _ppid, cmd in process_rows():
        if pid == mine:
            continue
        is_run_dir = run_dir_s in cmd or run_dir_rel in cmd
        is_qat = "rwkv7_qat.py" in cmd and is_run_dir
        is_autocycle = stop_autocycle and "autocycle_step50_ozempic.sh" in cmd
        is_caffeinate = "caffeinate" in cmd and is_run_dir
        if not (is_qat or is_autocycle or is_caffeinate):
            continue
        print(f"[stop] SIGTERM pid={pid} cmd={cmd[:180]}")
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append({"pid": pid, "cmd": cmd, "signal": "TERM"})
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            stopped.append({"pid": pid, "cmd": cmd, "error": str(exc)})
    time.sleep(5)
    return stopped


def write_eval_text(eval_prompts: Path, out_path: Path, max_rows: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with eval_prompts.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if count >= max_rows:
                break
            obj = json.loads(line)
            prompt = obj.get("prompt_rwkv") or ""
            gold = obj.get("gold") or ""
            if not prompt and obj.get("user"):
                prompt = f"<|rwkv_tokenizer_end_of_text|>User: {obj['user']}\n\nAssistant:"
            if not gold:
                continue
            dst.write(prompt)
            dst.write(" ")
            dst.write(gold)
            dst.write("\n\n")
            count += 1
    if count == 0:
        raise RuntimeError(f"no eval rows could be written from {eval_prompts}")
    return count


def load_latest_ppl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def write_frontier_queue(out_dir: Path, frozen_state: Path, step: int, args: argparse.Namespace) -> Path:
    queue = out_dir / "frontier_queue.sh"
    run_dir = args.run_dir
    targets_512 = " ".join(str(x) for x in range(step + 5, min(step + 65, args.max_auto_step) + 1, 5))
    targets_anchor = " ".join(str(x) for x in range(step + 5, min(step + 35, args.max_auto_step) + 1, 5))
    queue.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "{ROOT}"

# Generated by hawking_after_ema.py at {iso_now()}.
# Run these one at a time. Do not run multiple MPS training branches together on 18 GB.

BASE_CKPT="{frozen_state}"
RUN_DIR="{run_dir}"

echo "[frontier] 1/3 Fast learner: 512 tokens / grad_accum 8"
echo "screen -S hawking_branch_512_g8 -dm bash -lc 'cd \"$PWD\" && PYTHONHASHSEED={args.seed} PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 exec caffeinate -dimsu .venv-rwkv/bin/python tools/training/rwkv7_qat.py --model models/rwkv7-g1-04-hf/model.safetensors --hf-dir models/rwkv7-g1-04-hf --data artifacts/rwkv7_posttrain/sft.jsonl --out artifacts/lowbit_rwkv7/runs/hawking_branch_512_g8 --stage ffn --quant ternary --last-n-layers 8 --max-length 512 --grad-accum 8 --lr 5e-6 --epochs 1 --save-every 5 --eval-every 0 --eval-tokens 4096 --device mps --run-id hawking512 --seed {args.seed} --pretokenize-workers 4 --mps-empty-cache-every 5 --use-chunked --chunk-size 32 --resume-from \"$BASE_CKPT\" > artifacts/lowbit_rwkv7/runs/hawking_branch_512_g8/train.log 2>&1'"

echo "[frontier] 2/3 Balanced continuation: 768 tokens / grad_accum 8"
echo "OZEMPIC_TARGET_STEPS='{targets_512}' OZEMPIC_TARGET_INTERVAL=5 OZEMPIC_SAVE_EVERY=5 OZEMPIC_COOLDOWN_SECONDS=10 OZEMPIC_KEEP_LAST_N_CHECKPOINTS=8 OZEMPIC_GRAD_ACCUM=8 OZEMPIC_MAX_LENGTH=768 OZEMPIC_MPS_EMPTY_CACHE_EVERY=5 OZEMPIC_DETERMINISTIC=0 OZEMPIC_PRETOKENIZE_WORKERS=4 OZEMPIC_USE_CHUNKED=1 OZEMPIC_CHUNK_SIZE=32 OZEMPIC_MAX_AUTO_STEP={args.max_auto_step} bash tools/training/autocycle_step50_ozempic.sh"

echo "[frontier] 3/3 Quality anchor: 1024 tokens / grad_accum 16"
echo "screen -S hawking_anchor_1024_g16 -dm bash -lc 'cd \"$PWD\" && PYTHONHASHSEED={args.seed} PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 exec caffeinate -dimsu .venv-rwkv/bin/python tools/training/rwkv7_qat.py --model models/rwkv7-g1-04-hf/model.safetensors --hf-dir models/rwkv7-g1-04-hf --data artifacts/rwkv7_posttrain/sft.jsonl --out artifacts/lowbit_rwkv7/runs/hawking_anchor_1024_g16 --stage ffn --quant ternary --last-n-layers 8 --max-length 1024 --grad-accum 16 --lr 5e-6 --epochs 1 --save-every 5 --eval-every 0 --eval-tokens 4096 --device mps --run-id hawking1024 --seed {args.seed} --pretokenize-workers 4 --mps-empty-cache-every 5 --use-chunked --chunk-size 32 --resume-from \"$BASE_CKPT\" > artifacts/lowbit_rwkv7/runs/hawking_anchor_1024_g16/train.log 2>&1'"

echo "[frontier] Anchor target steps would be: {targets_anchor}"
""",
        encoding="utf-8",
    )
    queue.chmod(0o755)
    return queue


def write_report(
    out_dir: Path,
    manifest: dict[str, Any],
    ppl_path: Path,
    samples_path: Path,
    frontier_queue: Path,
) -> Path:
    ppl_rows = load_latest_ppl(ppl_path)
    lines = [
        "# Hawking post-EMA handoff report",
        "",
        f"- created: {iso_now()}",
        f"- threshold: EMA <= {manifest['threshold']}",
        f"- trigger step: {manifest['trigger_event']['step']}",
        f"- frozen checkpoint step: {manifest['checkpoint_step']}",
        f"- frozen state: `{manifest['frozen_state']}`",
        f"- trigger EMA: {float(manifest['trigger_event']['loss_ema']):.6f}",
        f"- trigger PPL EMA: {float(manifest['trigger_event'].get('ppl_ema') or 0.0):.2f}",
        "",
        "## Evaluation",
        "",
    ]
    if ppl_rows:
        for row in ppl_rows:
            lines.append(
                f"- `{row.get('run_id')}`: corpus={row.get('corpus')} "
                f"tokens={row.get('n_tokens')} nll={row.get('nll')} ppl={row.get('ppl')}"
            )
    else:
        lines.append("- PPL rows pending or skipped.")
    lines.extend([
        "",
        "## Samples",
        "",
        f"- samples JSONL: `{samples_path}`",
        "",
        "## Next Frontier",
        "",
        f"- generated queue: `{frontier_queue}`",
        "- Run one frontier branch at a time on 18 GB.",
        "- Promote only checkpoints that beat this frozen candidate on PPL and sample sanity.",
        "",
    ])
    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Wait for EMA target, freeze checkpoint, evaluate, export, and queue next recipes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    ap.add_argument("--threshold", type=float, default=6.0)
    ap.add_argument("--checkpoint-interval", type=int, default=5)
    ap.add_argument("--poll-seconds", type=int, default=120)
    ap.add_argument("--min-checkpoint-bytes", type=int, default=1_000_000_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--hf-dir", default=str(ROOT / "models/rwkv7-g1-04-hf"))
    ap.add_argument("--base-model", default=str(ROOT / "models/rwkv7-g1-04-hf/model.safetensors"))
    ap.add_argument("--eval-prompts", default=str(ROOT / "artifacts/rwkv7_posttrain/eval_prompts.jsonl"))
    ap.add_argument("--eval-text-rows", type=int, default=128)
    ap.add_argument("--eval-short-tokens", type=int, default=8192)
    ap.add_argument("--eval-long-tokens", type=int, default=32768)
    ap.add_argument("--eval-stride", type=int, default=512)
    ap.add_argument("--sample-prompts", type=int, default=8)
    ap.add_argument("--sample-new-tokens", type=int, default=160)
    ap.add_argument("--max-auto-step", type=int, default=250)
    ap.add_argument("--once", action="store_true", help="check once and exit if threshold is not met")
    ap.add_argument("--no-stop-training", action="store_true")
    ap.add_argument("--no-stop-autocycle", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-long-eval", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-samples", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    events_path = run_dir / "events.jsonl"

    print(f"[hawking] waiting for EMA <= {args.threshold} in {events_path}")
    trigger: dict[str, Any] | None = None
    latest: dict[str, Any] | None = None
    while trigger is None:
        segment = latest_segment(read_jsonl(events_path))
        latest = segment[-1] if segment else None
        trigger = first_threshold_event(segment, args.threshold)
        print(f"[hawking] {fmt_event(latest)}", flush=True)
        if trigger is not None:
            break
        if args.once:
            print("[hawking] threshold not reached; exiting due to --once")
            return 2
        time.sleep(args.poll_seconds)

    assert trigger is not None
    trigger_step = int(trigger["step"])
    checkpoint_step = ceil_to_interval(trigger_step, args.checkpoint_interval)
    ckpt_dir = checkpoint_dir(run_dir, checkpoint_step)
    print(f"[hawking] target crossed at step {trigger_step}; freezing checkpoint step {checkpoint_step}")
    wait_for_stable_checkpoint(ckpt_dir, args.min_checkpoint_bytes, args.poll_seconds)

    out_dir = artifact_root / f"ema{str(args.threshold).replace('.', 'p')}_step_{checkpoint_step:06d}_{now_slug()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_state = out_dir / "state_dict.pt"
    link_mode = link_or_copy(ckpt_dir / "state_dict.pt", frozen_state)
    print(f"[hawking] frozen state via {link_mode}: {frozen_state}")

    stopped: list[dict[str, Any]] = []
    if not args.no_stop_training:
        stopped = stop_processes(run_dir, stop_autocycle=not args.no_stop_autocycle)

    eval_text = out_dir / "eval_prompts_text.txt"
    eval_rows = write_eval_text(Path(args.eval_prompts), eval_text, args.eval_text_rows)
    print(f"[hawking] wrote eval text rows={eval_rows}: {eval_text}")

    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": str(args.seed),
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    })
    py = python_cmd()
    ppl_path = out_dir / "ppl.jsonl"

    eval_cmd_base = [
        py,
        str(ROOT / "tools/training/rwkv7_eval_ppl.py"),
        "--hf-dir", args.hf_dir,
        "--corpus", "text-file",
        "--text-file", str(eval_text),
        "--stride", str(args.eval_stride),
        "--device", args.device,
        "--out", str(ppl_path),
    ]

    if not args.skip_baseline:
        run_command(
            eval_cmd_base + [
                "--model", args.base_model,
                "--tokens", str(args.eval_short_tokens),
                "--run-id", f"base_short_{args.eval_short_tokens}",
            ],
            out_dir / "logs/eval_base_short.log",
            env,
        )

    run_command(
        eval_cmd_base + [
            "--model", str(frozen_state),
            "--tokens", str(args.eval_short_tokens),
            "--run-id", f"candidate_step_{checkpoint_step}_short_{args.eval_short_tokens}",
        ],
        out_dir / "logs/eval_candidate_short.log",
        env,
    )

    if not args.skip_long_eval and args.eval_long_tokens > args.eval_short_tokens:
        run_command(
            eval_cmd_base + [
                "--model", str(frozen_state),
                "--tokens", str(args.eval_long_tokens),
                "--run-id", f"candidate_step_{checkpoint_step}_long_{args.eval_long_tokens}",
            ],
            out_dir / "logs/eval_candidate_long.log",
            env,
        )

    samples_path = out_dir / "samples.jsonl"
    if not args.skip_samples:
        run_command(
            [
                py,
                str(ROOT / "tools/training/rwkv7_sample_prompts.py"),
                "--model", str(frozen_state),
                "--hf-dir", args.hf_dir,
                "--prompts-jsonl", args.eval_prompts,
                "--out", str(samples_path),
                "--device", args.device,
                "--seed", str(args.seed),
                "--max-prompts", str(args.sample_prompts),
                "--max-new-tokens", str(args.sample_new_tokens),
                "--max-context", "768",
                "--temperature", "0",
            ],
            out_dir / "logs/samples.log",
            env,
        )

    export_dir = out_dir / "hf"
    if not args.skip_export:
        run_command(
            [
                py,
                str(ROOT / "tools/training/rwkv7_export_hf.py"),
                "--state-dict", str(frozen_state),
                "--hf-dir", args.hf_dir,
                "--out-dir", str(export_dir),
                "--no-gguf",
            ],
            out_dir / "logs/export_hf.log",
            env,
        )

    manifest = {
        "timestamp": iso_now(),
        "threshold": args.threshold,
        "trigger_event": trigger,
        "latest_event_at_start": latest,
        "checkpoint_step": checkpoint_step,
        "source_checkpoint": str(ckpt_dir),
        "frozen_state": str(frozen_state),
        "link_mode": link_mode,
        "stopped_processes": stopped,
        "eval_text": str(eval_text),
        "eval_rows": eval_rows,
        "ppl_jsonl": str(ppl_path),
        "samples_jsonl": str(samples_path),
        "export_dir": str(export_dir) if not args.skip_export else None,
        "args": vars(args),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    frontier_queue = write_frontier_queue(out_dir, frozen_state, checkpoint_step, args)
    report = write_report(out_dir, manifest, ppl_path, samples_path, frontier_queue)
    print(f"[hawking] complete: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
