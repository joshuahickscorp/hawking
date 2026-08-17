#!/usr/bin/env python3
"""Measure resident Genesis: one load, three serves, stopfile, fallback.

Does not touch workspace/ops/GENESIS_STOP (that would halt launchd).
Uses a private stopfile and socket under --out.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import genesis_resident as gr  # noqa: E402

PARENT = Path("/Users/scammermike/Downloads/hawking")
BODY = Path("/tmp/genesis-resident-target/release/genesis-resident")
GREEDY = PARENT / "workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy"
ARTIFACT = PARENT / "workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1"
TOKENIZER = PARENT / "workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json"


def _rss(pid: int) -> int | None:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
        return int(out.strip()) * 1024
    except (subprocess.CalledProcessError, ValueError):
        return None


def _footprint(pid: int, dest: Path) -> dict | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["footprint", "-p", str(pid), "-j", str(dest), "-f", "bytes"],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not dest.is_file():
        return None
    try:
        return json.loads(dest.read_text())
    except json.JSONDecodeError:
        return None


def _wait_health(sock: Path, proc: subprocess.Popen, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            err = (proc.stderr.read() if proc.stderr else "") or ""
            raise SystemExit(f"body exited {proc.returncode} before health: {err[-2000:]}")
        last = gr.health(sock, timeout=2.0)
        if last and last.get("body_resident"):
            return last
        time.sleep(0.5)
    raise SystemExit(f"body never reported resident: last={last}")


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/gr-live")
    out_dir.mkdir(parents=True, exist_ok=True)
    sock = Path("/tmp/gr-live.sock")
    stop = out_dir / "GENESIS_STOP"
    try:
        sock.unlink()
    except FileNotFoundError:
        pass
    try:
        stop.unlink()
    except FileNotFoundError:
        pass

    receipt: dict = {
        "schema": "hawking.ascent.genesis_resident.v1",
        "date": "2026-08-16",
        "timing_label": "DIRTY_ENGINEERING",
        "timing_label_reason": (
            "Measured on the campaign box while other CPU lanes may be live. "
            "Not offered as CLEAN_CANDIDATE or BASE_TRUE_TPS. Win claimed is "
            "load time / residency, not tokens/s."
        ),
        "not_a_throughput_claim": True,
        "concurrent_decode_ceiling": 1,
    }

    # --- cold one-shot (today's loop) ---
    if not GREEDY.is_file():
        raise SystemExit(f"missing greedy {GREEDY}")
    cold_cmd = [
        str(GREEDY),
        "--artifact-root",
        str(ARTIFACT),
        "--tokenizer",
        str(TOKENIZER),
        "--prompt",
        "Say hi.",
        "--max-new-tokens",
        "4",
        "--max-seq-len",
        "128",
    ]
    t0 = time.monotonic()
    cold = subprocess.run(cold_cmd, capture_output=True, text=True, timeout=1800)
    cold_wall_ns = int((time.monotonic() - t0) * 1e9)
    receipt["cold_oneshot"] = {
        "argv": cold_cmd,
        "exit": cold.returncode,
        "wall_ns": cold_wall_ns,
        "stdout_tail": cold.stdout[-1500:],
        "stderr_tail": cold.stderr[-1500:],
    }
    if cold.returncode != 0:
        (out_dir / "GENESIS_RESIDENT.json").write_text(json.dumps(receipt, indent=2) + "\n")
        raise SystemExit(f"cold greedy failed: {cold.stderr[-800:]}")

    # --- start resident body ---
    if not BODY.is_file():
        raise SystemExit(f"missing body {BODY}")
    body_cmd = [
        str(BODY),
        "--artifact-root",
        str(ARTIFACT),
        "--tokenizer",
        str(TOKENIZER),
        "--socket",
        str(sock),
        "--stopfile",
        str(stop),
        "--repo",
        str(REPO),
        "--max-seq-len",
        "128",
        "--max-new-tokens",
        "4",
    ]
    t_start = time.monotonic()
    proc = subprocess.Popen(body_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    health = _wait_health(sock, proc, timeout_s=300)
    ready_ns = int((time.monotonic() - t_start) * 1e9)
    fp1 = _footprint(proc.pid, out_dir / "footprint-after-load.json")
    receipt["start"] = {
        "pid": proc.pid,
        "ready_wall_ns": ready_ns,
        "health": health,
        "rss_bytes": _rss(proc.pid),
        "process_alive": gr.process_alive(proc.pid),
        "footprint": fp1,
    }

    # --- three serves, one body ---
    serves = []
    for i in range(3):
        t1 = time.monotonic()
        resp = gr.propose("Say hi.", sock_path=sock, max_new_tokens=4, raw=False, timeout=300)
        wall_ns = int((time.monotonic() - t1) * 1e9)
        if resp is None:
            raise SystemExit(f"propose {i} failed")
        serves.append(
            {
                "i": i,
                "client_wall_ns": wall_ns,
                "response": {
                    k: resp.get(k)
                    for k in (
                        "ok",
                        "text",
                        "fallbacks",
                        "wall_ns",
                        "load_count",
                        "serve_index",
                        "body_resident",
                        "pid",
                        "rpc_ns",
                        "prompt_len",
                        "new_tokens",
                    )
                },
                "rss_bytes": _rss(proc.pid),
                "process_alive": gr.process_alive(proc.pid),
            }
        )
    health_after = gr.health(sock, timeout=2.0)
    fp2 = _footprint(proc.pid, out_dir / "footprint-after-three.json")
    receipt["three_serves"] = {
        "serves": serves,
        "health": health_after,
        "rss_bytes": _rss(proc.pid),
        "footprint": fp2,
        "one_load": bool(health_after and health_after.get("load_count") == 1),
        "three_serves": bool(health_after and health_after.get("serve_count") == 3),
        "same_pid": all(s["response"]["pid"] == proc.pid for s in serves),
    }

    # --- stopfile ---
    stop.write_text("")
    t_stop = time.monotonic()
    try:
        code = proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        code = proc.wait(timeout=5)
    stop_ns = int((time.monotonic() - t_stop) * 1e9)
    stderr = proc.stderr.read() if proc.stderr else ""
    stdout = proc.stdout.read() if proc.stdout else ""
    receipt["stopfile"] = {
        "path": str(stop),
        "exit_code": code,
        "wait_ns": stop_ns,
        "alive_after": gr.process_alive(proc.pid),
        "health_after": gr.health(sock, timeout=0.4),
        "stderr_tail": stderr[-1500:],
        "stdout_tail": stdout[-800:],
    }

    # --- fallback: service is dead, daemon still proposes via stub shell ---
    import importlib.util

    spec = importlib.util.spec_from_file_location("ascent_daemon", REPO / "tools" / "ascent_daemon.py")
    assert spec and spec.loader
    daemon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(daemon)
    stub = out_dir / "greedy-stub"
    stub.write_text("#!/usr/bin/env python3\nprint('GENERATED_TEXT_VERBATIM: fallback-after-kill')\nprint('FALLBACKS: 0')\n")
    stub.chmod(0o755)
    artifact = out_dir / "artifact"
    artifact.mkdir(exist_ok=True)
    os.environ["GENESIS_RESIDENT_SOCK"] = str(sock)
    os.environ["GENESIS_BIN"] = str(stub)
    os.environ["GENESIS_ARTIFACT"] = str(artifact)
    os.environ["GENESIS_TOKENIZER"] = str(out_dir / "tok.json")
    t_fb = time.monotonic()
    text = daemon.genesis_proposes("weight_addressing after kill")
    fb_ns = int((time.monotonic() - t_fb) * 1e9)
    receipt["fallback"] = {
        "service_live": gr.body_is_up(sock),
        "proposal": text,
        "wall_ns": fb_ns,
        "ok": text == "fallback-after-kill",
    }

    dest = out_dir / "GENESIS_RESIDENT.json"
    dest.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"wrote": str(dest), "cold_wall_ns": cold_wall_ns, "load_ns": health.get("load_ns"), "serves": [s["client_wall_ns"] for s in serves], "stop_exit": code, "fallback": text}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
