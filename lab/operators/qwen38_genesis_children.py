"""Spawn / poll / kill for Qwen3.8 genesis children.

Liveness is the OS process, never a status file. Per-child stdout/stderr/json
go to disk. Timing workloads require the GPU lock; text workloads record the
flag and proceed.

Primary hosting: one process, N shared-weight sessions.
Fallback: staggered process-pool (spawn serialized, run concurrent).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from lab.operators.qwen38_host_admission import (
    DEFAULT_RESERVE_BYTES,
    VERDICT_ADMIT,
    VERDICT_REFUSE,
    decide_admission,
    gpu_lock_held,
    host_memory_snapshot,
    pid_alive,
    process_pool_child_cost_bytes,
    process_rss_bytes,
    prove_refuse,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = REPO / "target" / "release" / "examples" / "ascension_qwen38_shared_sessions"
DEFAULT_ARTIFACT = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1"
)
DEFAULT_TOKENIZER = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json"
)

TEXT_WORKLOADS = frozenset({"gravity-recipe"})
TIMING_WORKLOADS = frozenset({"kernel-floor", "probe"})


class Child:
    def __init__(
        self,
        child_id: str,
        popen: subprocess.Popen[bytes],
        stdout_path: Path,
        stderr_path: Path,
        json_path: Path,
        workload: str,
        lock_held: bool,
    ) -> None:
        self.child_id = child_id
        self.popen = popen
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.json_path = json_path
        self.workload = workload
        self.lock_held = lock_held
        self.pid = popen.pid

    def poll(self) -> dict[str, Any]:
        code = self.popen.poll()
        alive = pid_alive(self.pid) if code is None else False
        # Liveness from the process, not the json receipt.
        return {
            "child_id": self.child_id,
            "pid": self.pid,
            "alive": alive,
            "exit_code": code,
            "workload": self.workload,
            "lock_held": self.lock_held,
            "stdout": str(self.stdout_path),
            "stderr": str(self.stderr_path),
            "json": str(self.json_path),
        }

    def kill(self, sig: int = signal.SIGTERM) -> dict[str, Any]:
        if self.popen.poll() is None:
            try:
                os.kill(self.pid, sig)
            except OSError:
                pass
            try:
                self.popen.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.kill(self.pid, signal.SIGKILL)
                self.popen.wait(timeout=5)
        return self.poll()


def _binary(path: Path | None) -> Path:
    binary = path or DEFAULT_BINARY
    if not binary.is_file():
        raise FileNotFoundError(f"missing worker binary {binary}")
    return binary


def spawn_child(
    *,
    out_dir: Path,
    child_id: str,
    workload: str,
    sessions: int,
    max_seq_len: int,
    max_new_tokens: int,
    artifact: Path,
    tokenizer: Path,
    binary: Path | None = None,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
    extra: list[str] | None = None,
) -> Child:
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / f"{child_id}.stdout"
    stderr_path = out_dir / f"{child_id}.stderr"
    json_path = out_dir / f"{child_id}.json"
    lock = gpu_lock_held()
    lock_held = bool(lock["held"])
    if workload in TIMING_WORKLOADS and not lock_held:
        raise RuntimeError(
            f"{child_id}: {workload} is lock-bound; gpu lock is not held"
        )
    cmd = [
        str(_binary(binary)),
        "--mode",
        workload,
        "--artifact-root",
        str(artifact),
        "--tokenizer",
        str(tokenizer),
        "--sessions",
        str(sessions),
        "--max-seq-len",
        str(max_seq_len),
        "--max-new-tokens",
        str(max_new_tokens),
        "--child-id",
        child_id,
        "--reserve-bytes",
        str(reserve_bytes),
        "--out",
        str(json_path),
    ]
    if lock_held:
        cmd.append("--lock-held")
    if extra:
        cmd.extend(extra)
    stdout = open(stdout_path, "wb")
    stderr = open(stderr_path, "wb")
    popen = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
    stdout.close()
    stderr.close()
    return Child(child_id, popen, stdout_path, stderr_path, json_path, workload, lock_held)


def wait_children(children: list[Child], timeout_s: float) -> list[dict[str, Any]]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        states = [c.poll() for c in children]
        if all(s["exit_code"] is not None for s in states):
            return states
        time.sleep(0.5)
    for child in children:
        if child.popen.poll() is None:
            child.kill()
    return [c.poll() for c in children]


def run_shared(
    *,
    out_dir: Path,
    sessions: int,
    max_seq_len: int,
    max_new_tokens: int,
    artifact: Path,
    tokenizer: Path,
    workload: str = "probe",
    binary: Path | None = None,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
) -> dict[str, Any]:
    memory = host_memory_snapshot()
    cost = process_pool_child_cost_bytes(max_seq_len)
    admission = decide_admission(
        memory,
        label="shared-worker",
        cost_bytes=cost,
        kind="shared_process_load",
        reserve_bytes=reserve_bytes,
    )
    if admission["verdict"] != VERDICT_ADMIT:
        return {
            "hosting": "shared-sessions",
            "status": "REFUSED",
            "admission": admission,
        }
    child = spawn_child(
        out_dir=out_dir,
        child_id="shared-0",
        workload=workload,
        sessions=sessions,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        artifact=artifact,
        tokenizer=tokenizer,
        binary=binary,
        reserve_bytes=reserve_bytes,
    )
    states = wait_children([child], timeout_s=7200)
    return {
        "hosting": "shared-sessions",
        "status": "RAN",
        "admission": admission,
        "children": states,
        "parallelizes": workload in TEXT_WORKLOADS or workload == "probe",
        "lock_bound": workload in TIMING_WORKLOADS,
    }


def run_process_pool(
    *,
    out_dir: Path,
    n_children: int,
    max_seq_len: int,
    max_new_tokens: int,
    artifact: Path,
    tokenizer: Path,
    workload: str,
    binary: Path | None = None,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
    stagger_s: float = 8.0,
) -> dict[str, Any]:
    """Fallback: spawn serialized, run concurrent. Simultaneous load is the fail."""
    children: list[Child] = []
    admissions = []
    for index in range(n_children):
        memory = host_memory_snapshot()
        admission = decide_admission(
            memory,
            label=f"proc-{index}",
            cost_bytes=process_pool_child_cost_bytes(max_seq_len),
            kind="process_pool_child",
            reserve_bytes=reserve_bytes,
        )
        admissions.append(admission)
        if admission["verdict"] != VERDICT_ADMIT:
            for child in children:
                child.kill()
            return {
                "hosting": "process-pool-staggered",
                "status": "REFUSED",
                "admissions": admissions,
                "spawned": [c.poll() for c in children],
            }
        child = spawn_child(
            out_dir=out_dir,
            child_id=f"proc-{index}",
            workload=workload,
            sessions=1,
            max_seq_len=max_seq_len,
            max_new_tokens=max_new_tokens,
            artifact=artifact,
            tokenizer=tokenizer,
            binary=binary,
            reserve_bytes=reserve_bytes,
        )
        children.append(child)
        # Stagger: wait until this child's RSS is visible, then admit the next.
        time.sleep(stagger_s)
    states = wait_children(children, timeout_s=7200)
    return {
        "hosting": "process-pool-staggered",
        "status": "RAN",
        "admissions": admissions,
        "children": states,
        "stagger_s": stagger_s,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prove-refuse", "spawn-poll-kill", "run"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hosting", choices=["shared-sessions", "process-pool"], default="shared-sessions")
    parser.add_argument("--workload", default="probe")
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--children", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "prove-refuse":
        decision = prove_refuse()
        if decision["verdict"] != VERDICT_REFUSE:
            print("gate failed to refuse", file=sys.stderr)
            return 2
        path = args.out_dir / "ADMISSION_REFUSE.json"
        path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(decision, indent=2))
        return 0

    if args.command == "spawn-poll-kill":
        # Tiny no-model refuse worker is `--mode refuse`; still a real process.
        binary = args.binary or DEFAULT_BINARY
        if not binary.is_file():
            # Fall back to /usr/bin/true-shaped python so the control plane
            # can be demonstrated without the Metal binary.
            cmd_child = [
                sys.executable,
                "-c",
                "import time,sys; time.sleep(30); sys.exit(0)",
            ]
            stdout = open(args.out_dir / "ctl-0.stdout", "wb")
            stderr = open(args.out_dir / "ctl-0.stderr", "wb")
            popen = subprocess.Popen(cmd_child, stdout=stdout, stderr=stderr)
            stdout.close()
            stderr.close()
            child = Child(
                "ctl-0",
                popen,
                args.out_dir / "ctl-0.stdout",
                args.out_dir / "ctl-0.stderr",
                args.out_dir / "ctl-0.json",
                "control",
                False,
            )
        else:
            child = spawn_child(
                out_dir=args.out_dir,
                child_id="ctl-0",
                workload="refuse",
                sessions=1,
                max_seq_len=args.max_seq_len,
                max_new_tokens=1,
                artifact=args.artifact,
                tokenizer=args.tokenizer,
                binary=binary,
            )
        live = child.poll()
        if not live["alive"] and live["exit_code"] is None:
            print("child not alive after spawn", file=sys.stderr)
            return 2
        killed = child.kill()
        if killed["alive"]:
            print("child still alive after kill", file=sys.stderr)
            return 2
        receipt = {
            "spawn": live,
            "after_kill": killed,
            "liveness_source": "os.kill(pid,0)+waitpid",
            "status_file_consulted": False,
        }
        (args.out_dir / "SPAWN_POLL_KILL.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(receipt, indent=2))
        return 0

    if args.hosting == "shared-sessions":
        result = run_shared(
            out_dir=args.out_dir,
            sessions=args.sessions,
            max_seq_len=args.max_seq_len,
            max_new_tokens=args.max_new_tokens,
            artifact=args.artifact,
            tokenizer=args.tokenizer,
            workload=args.workload,
            binary=args.binary,
            reserve_bytes=args.reserve_bytes,
        )
    else:
        result = run_process_pool(
            out_dir=args.out_dir,
            n_children=args.children,
            max_seq_len=args.max_seq_len,
            max_new_tokens=args.max_new_tokens,
            artifact=args.artifact,
            tokenizer=args.tokenizer,
            workload=args.workload,
            binary=args.binary,
            reserve_bytes=args.reserve_bytes,
        )
    (args.out_dir / "RUN.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") != "REFUSED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
