#!/usr/bin/env python3
"""Fresh producer: start the resident body, send one generate, record TPS."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
BIN = "/Users/scammermike/Downloads/hawking/workspace/ops/build/rust/release-fast/examples/ascension_qwen38_resident"
LOCK = "/Users/scammermike/Downloads/hawking/tools/gpu_lane_lock.sh"
ARTIFACT = os.path.expanduser("~/noetic/NOETIC_PARENT_A")
TOKENIZER = os.path.expanduser("~/noetic/NOETIC_PARENT_A/tokenizer.json")
PROMPT = "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n"
MAX_NEW = 64
MAX_SEQ = 8192


def snapshot_contention() -> dict:
    loadavg = subprocess.check_output(["sysctl", "-n", "vm.loadavg"], text=True).strip()
    uptime = subprocess.check_output(["uptime"], text=True).strip()
    ps = subprocess.check_output(["ps", "-axo", "pid=,pcpu=,rss=,command="], text=True)
    hf = []
    hot = []
    resident = []
    for line in ps.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split(None, 3)
        if len(parts) < 4:
            continue
        pid, cpu, rss, cmd = parts
        rec = {"pid": int(pid), "cpu_pct": float(cpu), "rss_kib": int(rss), "cmd": cmd}
        if "hf download" in cmd:
            hf.append(rec)
        if "ascension_qwen38_resident" in cmd:
            resident.append(rec)
        if float(cpu) >= 5.0:
            hot.append(rec)
    hot.sort(key=lambda r: -r["cpu_pct"])
    return {
        "loadavg": loadavg,
        "uptime": uptime,
        "hf_downloads": hf,
        "other_residents": resident,
        "hot_cpu": hot[:15],
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def run_one(label: str, extra_env: dict, out_name: str) -> dict:
    contention_before = snapshot_contention()
    stderr_path = OUTDIR / f"{out_name}.stderr.log"
    stdout_path = OUTDIR / f"{out_name}.stdout.jsonl"
    meta_path = OUTDIR / f"{out_name}.meta.json"

    cmd = [
        LOCK,
        f"resident-tps-recon-{label}",
        BIN,
        "--artifact-root",
        ARTIFACT,
        "--tokenizer",
        TOKENIZER,
        "--max-seq-len",
        str(MAX_SEQ),
        "--resident-identity",
        "sealed-3.14",
    ]
    env = os.environ.copy()
    env.update(extra_env)
    env.setdefault("HAWKING_GPU_LANE_LOCK", "/tmp/hawking-gpu-lane.lock")

    t0 = time.perf_counter()
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(stderr_path, "w") as errf, open(stdout_path, "w") as outf:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errf,
            env=env,
            text=True,
            bufsize=1,
        )
        assert proc.stdin is not None and proc.stdout is not None
        ready_line = None
        deadline = time.time() + 600
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            outf.write(line)
            outf.flush()
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("status") == "ready":
                ready_line = obj
                break
        ready_at = time.perf_counter()
        if ready_line is None:
            proc.kill()
            rc = proc.wait(timeout=10)
            raise SystemExit(f"{label}: no ready line; rc={rc}")

        req = {
            "id": f"{label}-1",
            "prompt": PROMPT,
            "max_new_tokens": MAX_NEW,
        }
        gen_t0 = time.perf_counter()
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        reply_line = proc.stdout.readline()
        gen_t1 = time.perf_counter()
        if reply_line:
            outf.write(reply_line)
            outf.flush()
        proc.stdin.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        t1 = time.perf_counter()

    reply = None
    if reply_line:
        try:
            reply = json.loads(reply_line)
        except json.JSONDecodeError as e:
            reply = {"parse_error": str(e), "raw": reply_line[:4000]}

    contention_after = snapshot_contention()
    record = {
        "label": label,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": cmd,
        "extra_env": extra_env,
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW,
        "max_seq_len": MAX_SEQ,
        "ready": ready_line,
        "reply": reply,
        "wall_s_until_ready": ready_at - t0,
        "wall_s_generate": gen_t1 - gen_t0,
        "wall_s_process": t1 - t0,
        "host_generate_s": gen_t1 - gen_t0,
        "returncode": proc.returncode,
        "contention_before": contention_before,
        "contention_after": contention_after,
        "stderr_path": str(stderr_path),
        "stdout_path": str(stdout_path),
        "binary": BIN,
        "binary_mtime": datetime.fromtimestamp(os.path.getmtime(BIN), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "binary_size": os.path.getsize(BIN),
    }
    meta_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({
        "label": label,
        "decode_tps": (reply or {}).get("decode_tps"),
        "complete_tps": (reply or {}).get("complete_tps"),
        "generated_tokens": (reply or {}).get("generated_tokens"),
        "decode_wall_ns": (reply or {}).get("decode_wall_ns"),
        "fallbacks": (reply or {}).get("fallbacks"),
        "hf_before": len(contention_before["hf_downloads"]),
        "other_residents_before": len(contention_before["other_residents"]),
        "wall_s_until_ready": record["wall_s_until_ready"],
        "wall_s_generate": record["wall_s_generate"],
        "returncode": proc.returncode,
    }, indent=2))
    return record


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "control"
    extra = {}
    if len(sys.argv) > 2:
        # argv[2] is HAWKING_QWEN38_Q4_GEO value, empty means unset
        geo = sys.argv[2]
        if geo:
            extra["HAWKING_QWEN38_Q4_GEO"] = geo
    run_one(label, extra, label)


if __name__ == "__main__":
    main()
