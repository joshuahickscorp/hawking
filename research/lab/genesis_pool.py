"""Spawnable / killable Qwen3.8 worker pool for Genesis.

PRIMARY decode hosting is one process / N sessions / one resident weight
set (`Qwen38HybridDecodeSession::attach`). This module is the process-pool
FALLBACK: spawn serialized, run concurrent. Artifact pages are not shared
across processes (measured 2026-08-16).

Genesis drives two search workloads through the same API:

  spawn(prompt, budget) -> child_id
  poll(child_id)        -> running | done(text, wall_ns) | failed(reason)
  kill(child_id)        -> terminate the process group, no orphans

Liveness is always read from process state (kill(0) + ps + waitpid). A status
file is never consulted. Per-child stdout/stderr land on disk at spawn time so
a crashed pool loses no completed work.

Admission refuses to spawn past the measured safe N. The N=4 sample on this
box (2026-08-16) showed each child holding a *private* ~14.73 GB Metal
IOAccelerator buffer; mmap of the 8.5 GB artifact is not shared in a way that
reduces residency. N=4 already engaged swap. Default safe_n is therefore 3.

hold_gpu_lock is an explicit per-task flag:
  TEXT   (recipe eval, diagnosis, ranking) -> False, children overlap
  TIMING (kernel floor, complete-token wall) -> True, wraps gpu_lane_lock.sh
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_HAWKING = Path("/Users/scammermike/Downloads/hawking")
LOCK_SCRIPT = REPO_ROOT / "tools" / "gpu_lane_lock.sh"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "workspace" / "ops" / "local" / "genesis-pool"
BINARY_NAME = "ascension_qwen38_hybrid_greedy"

# Measured 2026-08-16 on this 96 GB M-series box, N=4 at --max-seq-len 2048.
# phys_footprint mean 15.970 GB; IOAccelerator dirty 14.728 GB *private* each;
# mapped-file clean ~4.5 MB (the mmap is not the residency). N=4 used 751 MB
# swap with 0.81 GB free. N=8 was not launched: it would swap and lie.
MEASURED_PHYS_FOOTPRINT_PER_CHILD = 15_970_043_138
MEASURED_IOACCELERATOR_PER_CHILD = 14_727_856_128
MEASURED_SAFE_N = 3
PAGE_SIZE = 16384

# Source-derived (not measured): GQA KV is 16 layers * 2 (K,V) * 4 heads * 256 * 4 B.
# ESTIMATE: 131_072 bytes per sequence position. 8192 vs 128 is ~1.05 GB, not a
# doubling of child count — the 14.73 GB Metal weight copy dominates.
KV_BYTES_PER_POSITION_ESTIMATE = 131_072

PollState = Literal["running", "done", "failed"]
LiveState = Literal["running", "zombie", "dead"]

TEXT_MARKER = "GENERATED_TEXT_VERBATIM:"
FALLBACKS_MARKER = "FALLBACKS:"
WALL_RE = re.compile(r"^WALL_NS:\s*(\d+)\s*$", re.M)
STEADY_RE = re.compile(
    r"^STEADY_DECODE_WALL_NS_PER_TOKEN:\s*(?:Some\()?(\d+)", re.M
)
NEW_TOKENS_RE = re.compile(r"^NEW_TOKENS:\s*(\[.*\])\s*$", re.M)


class AdmissionRefused(RuntimeError):
    """Spawn blocked: another child would oversubscribe the measured safe N."""

    def __init__(self, message: str, *, safe_n: int, alive: int) -> None:
        super().__init__(message)
        self.safe_n = safe_n
        self.alive = alive


class SpawnError(RuntimeError):
    pass


class UnknownChild(KeyError):
    pass


@dataclass(frozen=True)
class ChildBudget:
    max_new_tokens: int
    max_seq_len: int | None = None
    wall_timeout_s: float | None = None


def _as_budget(budget: int | ChildBudget) -> ChildBudget:
    if isinstance(budget, ChildBudget):
        if budget.max_new_tokens <= 0:
            raise ValueError("budget.max_new_tokens must be positive")
        return budget
    if int(budget) <= 0:
        raise ValueError("budget must be a positive token count")
    return ChildBudget(max_new_tokens=int(budget))


@dataclass
class ChildPoll:
    state: PollState
    text: str | None = None
    wall_ns: int | None = None
    reason: str | None = None
    exit_code: int | None = None
    child_wall_ns: int | None = None
    complete_token_ns: int | None = None
    output_dir: str | None = None
    pid: int | None = None
    hold_gpu_lock: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChildRecord:
    child_id: str
    pid: int
    pgid: int
    output_dir: str
    prompt: str
    max_new_tokens: int
    max_seq_len: int
    hold_gpu_lock: bool
    started_monotonic_ns: int
    started_wall_ns: int
    argv: list[str] = field(default_factory=list)
    binary: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "ChildRecord":
        return cls(
            child_id=str(raw["child_id"]),
            pid=int(raw["pid"]),
            pgid=int(raw["pgid"]),
            output_dir=str(raw["output_dir"]),
            prompt=str(raw.get("prompt") or ""),
            max_new_tokens=int(raw.get("max_new_tokens") or 0),
            max_seq_len=int(raw.get("max_seq_len") or 0),
            hold_gpu_lock=bool(raw.get("hold_gpu_lock")),
            started_monotonic_ns=int(raw.get("started_monotonic_ns") or 0),
            started_wall_ns=int(raw.get("started_wall_ns") or 0),
            argv=list(raw.get("argv") or []),
            binary=str(raw.get("binary") or ""),
        )


@dataclass
class PoolConfig:
    binary: Path
    output_root: Path
    artifact_root: Path | None = None
    tokenizer: Path | None = None
    safe_n: int = MEASURED_SAFE_N
    lock_script: Path = LOCK_SCRIPT
    max_seq_len: int = 128
    min_free_bytes: int = 0
    extra_args: list[str] = field(default_factory=list)
    lane_prefix: str = "genesis-child"


def discover_artifact() -> Path | None:
    for cand in (
        REPO_ROOT / "workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1",
        PARENT_HAWKING / "workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1",
    ):
        if (cand / "manifest.json").is_file():
            return cand
    return None


def discover_tokenizer() -> Path | None:
    for cand in (
        REPO_ROOT / "workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json",
        PARENT_HAWKING / "workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json",
    ):
        if cand.is_file():
            return cand
    return None


def discover_binary() -> Path | None:
    env = os.environ.get("CARGO_TARGET_DIR")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env) / "release/examples" / BINARY_NAME)
    candidates.extend(
        [
            REPO_ROOT / "workspace/ops/build/rust/release/examples" / BINARY_NAME,
            PARENT_HAWKING / "workspace/ops/build/rust/release/examples" / BINARY_NAME,
            Path(
                "/private/tmp/claude-503/-Users-scammermike-Downloads-hawking/"
                "d51d4904-9fa1-4f81-8170-5e7eb27a291d/scratchpad/main_target/"
                "release/examples"
            )
            / BINARY_NAME,
        ]
    )
    for cand in candidates:
        if os.access(cand, os.X_OK):
            return cand
    return None


def live_ready() -> tuple[Path, Path, Path] | None:
    binary = discover_binary()
    artifact = discover_artifact()
    tokenizer = discover_tokenizer()
    if binary and artifact and tokenizer:
        return binary, artifact, tokenizer
    return None


# ---------------------------------------------------------------------------
# Process state. Never a status file.
# ---------------------------------------------------------------------------


def process_liveness(pid: int) -> LiveState:
    """Kernel-facing liveness. A zombie is not 'running'."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    except PermissionError:
        return "running"
    state = _ps_state(pid)
    if state is None:
        return "dead"
    if state.startswith("Z"):
        return "zombie"
    return "running"


def _ps_state(pid: int) -> str | None:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "state=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None
    line = out.strip()
    return line or None


def process_rss_bytes(pid: int) -> int | None:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None
    text = out.strip()
    if not text:
        return None
    return int(text) * 1024


def _try_waitpid(pid: int) -> int | None:
    try:
        wpid, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return None
    except OSError:
        return None
    if wpid == 0:
        return None
    if hasattr(os, "waitstatus_to_exitcode"):
        try:
            return int(os.waitstatus_to_exitcode(status))
        except ValueError:
            return -1
    if os.WIFEXITED(status):
        return int(os.WEXITSTATUS(status))
    if os.WIFSIGNALED(status):
        return -int(os.WTERMSIG(status))
    return -1


def pgrep_binary_pids() -> list[int]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-x", BINARY_NAME],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    return [int(line) for line in out.splitlines() if line.strip().isdigit()]


# ---------------------------------------------------------------------------
# Memory samples. RSS and footprint answer different questions.
# ---------------------------------------------------------------------------


def _parse_vm_stat(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        digits = re.sub(r"[^0-9]", "", rest)
        if digits:
            out[key.strip()] = int(digits)
    return out


def sample_machine() -> dict[str, Any]:
    raw = subprocess.check_output(["vm_stat"], text=True)
    pages = _parse_vm_stat(raw)
    page = PAGE_SIZE
    def gb(pages_n: int) -> float:
        return pages_n * page / 1_000_000_000.0

    free = pages.get("Pages free", 0)
    active = pages.get("Pages active", 0)
    inactive = pages.get("Pages inactive", 0)
    wired = pages.get("Pages wired down", 0)
    spec = pages.get("Pages speculative", 0)
    purge = pages.get("Pages purgeable", 0)
    fileb = pages.get("File-backed pages", 0)
    anon = pages.get("Anonymous pages", 0)
    comp = pages.get("Pages occupied by compressor", 0)
    swap_used = _swap_used_bytes()
    return {
        "page_size": page,
        "free_bytes": free * page,
        "active_bytes": active * page,
        "inactive_bytes": inactive * page,
        "wired_bytes": wired * page,
        "speculative_bytes": spec * page,
        "purgeable_bytes": purge * page,
        "file_backed_bytes": fileb * page,
        "anonymous_bytes": anon * page,
        "compressor_occupied_bytes": comp * page,
        "swap_used_bytes": swap_used,
        # Activity-monitor-like "used" counts purgeable and file cache. It
        # overstates pressure. We report it and do not trust it for admission.
        "used_including_purgeable_bytes": (active + inactive + wired + spec + comp) * page,
        # What we trust for "is the box full": wired Metal + anonymous + compressor.
        "trusted_pressure_bytes": (wired + anon + comp) * page,
        "free_gb": gb(free),
        "wired_gb": gb(wired),
        "anonymous_gb": gb(anon),
        "file_backed_gb": gb(fileb),
        "trusted_pressure_gb": gb(wired + anon + comp),
        "swap_used_gb": swap_used / 1_000_000_000.0,
        "trusted_figure": "wired + anonymous + compressor; plus per-pid phys_footprint",
    }


def _swap_used_bytes() -> int:
    try:
        out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    except subprocess.CalledProcessError:
        return 0
    m = re.search(r"used\s*=\s*([\d.]+)M", out)
    if not m:
        return 0
    return int(float(m.group(1)) * 1024 * 1024)


def sample_footprint(pid: int) -> dict[str, Any] | None:
    """Parse `footprint --format bytes`. Returns None if the pid is gone."""
    tmp = f"/tmp/genesis-footprint-{pid}-{os.getpid()}.json"
    try:
        proc = subprocess.run(
            [
                "/usr/bin/footprint",
                "-p",
                str(pid),
                "--swapped",
                "--wired",
                "--format",
                "bytes",
                "-j",
                tmp,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    payload: dict[str, Any] | None = None
    path = Path(tmp)
    if path.is_file():
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            payload = None
        try:
            path.unlink()
        except OSError:
            pass
    if payload is None:
        return _parse_footprint_text(proc.stdout) if proc.stdout else None
    return _normalize_footprint_json(payload, pid)


def _normalize_footprint_json(payload: Any, pid: int) -> dict[str, Any] | None:
    processes = payload
    if isinstance(payload, dict):
        processes = payload.get("processes") or payload.get("process") or [payload]
    if isinstance(processes, dict):
        processes = [processes]
    if not isinstance(processes, list) or not processes:
        return None
    rec = processes[0]
    if not isinstance(rec, dict):
        return None
    raw_cats = rec.get("categories") or rec.get("Categories") or {}
    ioacc = 0
    mapped = 0
    malloc_large = 0
    items: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(raw_cats, dict):
        items = [(str(k), v) for k, v in raw_cats.items() if isinstance(v, Mapping)]
    elif isinstance(raw_cats, list):
        for cat in raw_cats:
            if isinstance(cat, Mapping):
                items.append((str(cat.get("category") or cat.get("name") or ""), cat))
    for name, cat in items:
        dirty = int(cat.get("dirty") or cat.get("Dirty") or 0)
        clean = int(cat.get("clean") or cat.get("Clean") or 0)
        if "IOAccelerator" in name and "graphics" in name:
            ioacc += dirty
        elif name == "mapped file":
            mapped += clean + dirty
        elif name == "Malloc Large":
            malloc_large += dirty
    phys = rec.get("footprint") or rec.get("phys_footprint") or rec.get("physFootprint")
    if phys is None:
        phys = rec.get("total") or rec.get("dirty")
    return {
        "pid": pid,
        "phys_footprint_bytes": int(phys or 0),
        "ioaccelerator_dirty_bytes": ioacc,
        "mapped_file_bytes": mapped,
        "malloc_large_bytes": malloc_large,
        "rss_bytes": process_rss_bytes(pid),
    }


def _parse_footprint_text(text: str) -> dict[str, Any] | None:
    phys = None
    ioacc = 0
    mapped = 0
    malloc_large = 0
    for line in text.splitlines():
        if "phys_footprint:" in line and "peak" not in line:
            m = re.search(r"phys_footprint:\s*(\d+)", line)
            if m:
                phys = int(m.group(1))
        if "IOAccelerator (graphics)" in line:
            m = re.match(r"\s*(\d+)\s+B", line)
            if m:
                ioacc = int(m.group(1))
        if re.search(r"\bmapped file\b", line):
            nums = re.findall(r"(\d+)\s+B", line)
            if len(nums) >= 3:
                mapped = int(nums[0]) + int(nums[2])
        if "Malloc Large" in line:
            m = re.match(r"\s*(\d+)\s+B", line)
            if m:
                malloc_large = int(m.group(1))
    if phys is None:
        return None
    return {
        "phys_footprint_bytes": phys,
        "ioaccelerator_dirty_bytes": ioacc,
        "mapped_file_bytes": mapped,
        "malloc_large_bytes": malloc_large,
    }


def sample_children(pids: Iterable[int]) -> dict[str, Any]:
    per = []
    for pid in pids:
        fp = sample_footprint(int(pid)) or {}
        rss = process_rss_bytes(int(pid))
        per.append(
            {
                "pid": int(pid),
                "rss_bytes": rss,
                "liveness": process_liveness(int(pid)),
                **fp,
            }
        )
    phys = [p.get("phys_footprint_bytes") or 0 for p in per]
    rss = [p.get("rss_bytes") or 0 for p in per]
    ioacc = [p.get("ioaccelerator_dirty_bytes") or 0 for p in per]
    mapped = [p.get("mapped_file_bytes") or 0 for p in per]
    return {
        "n": len(per),
        "per_process": per,
        "sum_phys_footprint_bytes": sum(phys),
        "sum_rss_bytes": sum(rss),
        "sum_ioaccelerator_bytes": sum(ioacc),
        "sum_mapped_file_bytes": sum(mapped),
        "machine": sample_machine(),
        "trust": (
            "Trust per-pid phys_footprint and machine wired+anonymous+compressor. "
            "RSS both undercounts Metal wired (seen 5.9-12.9 GB vs 15.97 GB phys) "
            "and would overcount shared file pages if the artifact were mapped. "
            "On this genome the artifact is copied into private IOAccelerator "
            "buffers (~14.73 GB each); mapped-file clean is ~4.5 MB."
        ),
    }


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def extract_generated_text(stdout: str) -> str | None:
    if TEXT_MARKER not in stdout:
        return None
    m = re.search(
        rf"{re.escape(TEXT_MARKER)}\s*(.*?)\n{re.escape(FALLBACKS_MARKER)}",
        stdout,
        re.S,
    )
    if m:
        return m.group(1)
    # Fallback: everything after the marker.
    return stdout.split(TEXT_MARKER, 1)[1]


def parse_child_metrics(stdout: str) -> dict[str, Any]:
    wall = WALL_RE.search(stdout)
    steady = STEADY_RE.search(stdout)
    tokens = NEW_TOKENS_RE.search(stdout)
    new_tokens: list[int] | None = None
    if tokens:
        try:
            parsed = json.loads(tokens.group(1))
            if isinstance(parsed, list):
                new_tokens = [int(x) for x in parsed]
        except (TypeError, ValueError, json.JSONDecodeError):
            new_tokens = None
    return {
        "text": extract_generated_text(stdout),
        "child_wall_ns": int(wall.group(1)) if wall else None,
        "complete_token_ns": int(steady.group(1)) if steady else None,
        "new_tokens": new_tokens,
    }


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class GenesisPool:
    def __init__(self, config: PoolConfig) -> None:
        if config.safe_n < 1:
            raise ValueError("safe_n must be >= 1")
        self.config = config
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        self._mu = threading.Lock()
        self._children: dict[str, ChildRecord] = {}
        self._final: dict[str, ChildPoll] = {}
        # Keep Popen objects alive. If they are collected, CPython's
        # Popen.__del__ reaps the child and later waitpid returns ECHILD,
        # which used to make a finished generate look like a mystery death.
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for meta in sorted(self.config.output_root.glob("*/meta.json")):
            try:
                raw = json.loads(meta.read_text())
                rec = ChildRecord.from_json(raw)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            self._children[rec.child_id] = rec

    def _meta_path(self, child_id: str) -> Path:
        return self.config.output_root / child_id / "meta.json"

    def _write_meta(self, rec: ChildRecord) -> None:
        path = self._meta_path(rec.child_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec.to_json(), indent=2, sort_keys=True) + "\n")
        tmp.replace(path)

    def _write_result(self, rec: ChildRecord, poll: ChildPoll) -> None:
        path = Path(rec.output_dir) / "result.json"
        body = {
            "child_id": rec.child_id,
            "pid": rec.pid,
            "observed_from": "process_state",
            **poll.to_dict(),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)

    def alive_count(self) -> int:
        with self._mu:
            return self._alive_locked()

    def _alive_locked(self) -> int:
        n = 0
        for rec in self._children.values():
            if rec.child_id in self._final:
                continue
            live = process_liveness(rec.pid)
            if live == "running":
                n += 1
        return n

    def running_ids(self) -> list[str]:
        with self._mu:
            out = []
            for rec in self._children.values():
                if rec.child_id in self._final:
                    continue
                if process_liveness(rec.pid) == "running":
                    out.append(rec.child_id)
            return out

    def _refuse_if_full(self) -> None:
        alive = self._alive_locked()
        if alive >= self.config.safe_n:
            raise AdmissionRefused(
                f"admission refused: {alive} live children at safe_n={self.config.safe_n}",
                safe_n=self.config.safe_n,
                alive=alive,
            )
        if self.config.min_free_bytes > 0:
            machine = sample_machine()
            if machine["free_bytes"] < self.config.min_free_bytes:
                raise AdmissionRefused(
                    f"admission refused: free_bytes {machine['free_bytes']} "
                    f"< min_free_bytes {self.config.min_free_bytes}",
                    safe_n=self.config.safe_n,
                    alive=alive,
                )
            if machine["swap_used_bytes"] > 256 * 1024 * 1024 and machine["free_bytes"] < (
                4 * 1024 * 1024 * 1024
            ):
                raise AdmissionRefused(
                    f"admission refused: swap already in use "
                    f"({machine['swap_used_bytes']} B) with only "
                    f"{machine['free_bytes']} B free",
                    safe_n=self.config.safe_n,
                    alive=alive,
                )

    def spawn(
        self,
        prompt: str,
        budget: int | ChildBudget,
        *,
        hold_gpu_lock: bool = False,
        extra_args: list[str] | None = None,
    ) -> str:
        """Start a child. Non-blocking. hold_gpu_lock is the TIMING/TEXT switch."""
        b = _as_budget(budget)
        seq = b.max_seq_len if b.max_seq_len is not None else self.config.max_seq_len
        with self._mu:
            self._refuse_if_full()
            child_id = uuid.uuid4().hex[:12]
            out_dir = self.config.output_root / child_id
            out_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = out_dir / "stdout.txt"
            stderr_path = out_dir / "stderr.txt"
            argv = self._argv(
                prompt,
                b.max_new_tokens,
                seq,
                hold_gpu_lock,
                child_id,
                extra_args or [],
            )
            stdout_fh = stdout_path.open("w", encoding="utf-8")
            stderr_fh = stderr_path.open("w", encoding="utf-8")
            try:
                proc = subprocess.Popen(
                    argv,
                    stdout=stdout_fh,
                    stderr=stderr_fh,
                    start_new_session=True,
                    cwd=str(REPO_ROOT),
                )
            except OSError as exc:
                stdout_fh.close()
                stderr_fh.close()
                raise SpawnError(f"failed to spawn {argv[0]}: {exc}") from exc
            finally:
                stdout_fh.close()
                stderr_fh.close()
            try:
                pgid = os.getpgid(proc.pid)
            except OSError:
                pgid = proc.pid
            rec = ChildRecord(
                child_id=child_id,
                pid=int(proc.pid),
                pgid=int(pgid),
                output_dir=str(out_dir),
                prompt=prompt,
                max_new_tokens=b.max_new_tokens,
                max_seq_len=seq,
                hold_gpu_lock=hold_gpu_lock,
                started_monotonic_ns=time.monotonic_ns(),
                started_wall_ns=time.time_ns(),
                argv=[str(x) for x in argv],
                binary=str(self.config.binary),
            )
            self._children[child_id] = rec
            self._procs[child_id] = proc
            self._write_meta(rec)
            return child_id

    def _argv(
        self,
        prompt: str,
        max_new_tokens: int,
        max_seq_len: int,
        hold_gpu_lock: bool,
        child_id: str,
        extra_args: list[str],
    ) -> list[str]:
        binary = Path(self.config.binary)
        if not os.access(binary, os.X_OK) and not str(binary).endswith(".py"):
            if not binary.is_file():
                raise SpawnError(f"binary is not executable: {binary}")
        inner = [str(binary)]
        if str(binary).endswith(".py") or Path(binary).name == "genesis_pool.py":
            inner = [sys.executable, str(binary), "stub-child"]
        if self.config.artifact_root is not None:
            inner.extend(["--artifact-root", str(self.config.artifact_root)])
        if self.config.tokenizer is not None:
            inner.extend(["--tokenizer", str(self.config.tokenizer)])
        inner.extend(
            [
                "--prompt",
                prompt,
                "--max-new-tokens",
                str(max_new_tokens),
                "--max-seq-len",
                str(max_seq_len),
            ]
        )
        inner.extend(self.config.extra_args)
        inner.extend(extra_args)
        if hold_gpu_lock:
            lock = Path(self.config.lock_script)
            if not os.access(lock, os.X_OK):
                raise SpawnError(f"gpu lock script missing or not executable: {lock}")
            return [str(lock), f"{self.config.lane_prefix}-{child_id}", *inner]
        return inner

    def poll(self, child_id: str) -> ChildPoll:
        with self._mu:
            return self._poll_locked(child_id)

    def _poll_locked(self, child_id: str) -> ChildPoll:
        if child_id in self._final:
            return self._final[child_id]
        rec = self._children.get(child_id)
        if rec is None:
            raise UnknownChild(child_id)
        exit_code = self._exit_code(rec)
        live = process_liveness(rec.pid)
        if live == "running" and exit_code is None:
            if self._timed_out(rec):
                self._signal_group(rec, signal.SIGTERM)
                return self._finalize(
                    rec,
                    "failed",
                    reason="wall_timeout",
                    exit_code=None,
                )
            return ChildPoll(
                state="running",
                output_dir=rec.output_dir,
                pid=rec.pid,
                hold_gpu_lock=rec.hold_gpu_lock,
            )
        if live == "zombie" and exit_code is None:
            exit_code = self._exit_code(rec)
        return self._finalize(rec, None, reason=None, exit_code=exit_code)

    def _exit_code(self, rec: ChildRecord) -> int | None:
        proc = self._procs.get(rec.child_id)
        if proc is not None:
            code = proc.poll()
            if code is not None:
                return int(code)
        return _try_waitpid(rec.pid)

    def _timed_out(self, rec: ChildRecord) -> bool:
        # Timeout lives on the budget; we stored only tokens. Pool-level
        # timeout is expressed by killing from the caller. No implicit timeout.
        return False

    def _finalize(
        self,
        rec: ChildRecord,
        state: PollState | None,
        *,
        reason: str | None,
        exit_code: int | None,
    ) -> ChildPoll:
        stdout = _read_text(Path(rec.output_dir) / "stdout.txt")
        stderr = _read_text(Path(rec.output_dir) / "stderr.txt")
        metrics = parse_child_metrics(stdout)
        wall_ns = time.monotonic_ns() - rec.started_monotonic_ns
        text = metrics["text"]
        if state is None:
            if text is not None and (exit_code is None or exit_code == 0):
                # A vanished waitpid (ECHILD after Popen.__del__, or an
                # adopted child) must not hide a completed generate.
                state = "done"
            elif exit_code == 0 and text is None:
                state = "failed"
                reason = "no_generated_text"
            else:
                state = "failed"
                if reason is None:
                    if exit_code is None:
                        reason = f"process_dead:{_tail(stderr, 400)}"
                    else:
                        reason = f"exit_{exit_code}:{_tail(stderr, 400)}"
        poll = ChildPoll(
            state=state,
            text=text if state == "done" else text,
            wall_ns=wall_ns,
            reason=reason if state == "failed" else None,
            exit_code=exit_code,
            child_wall_ns=metrics["child_wall_ns"],
            complete_token_ns=metrics["complete_token_ns"],
            output_dir=rec.output_dir,
            pid=rec.pid,
            hold_gpu_lock=rec.hold_gpu_lock,
        )
        self._final[rec.child_id] = poll
        self._write_result(rec, poll)
        return poll

    def kill(self, child_id: str, *, grace_s: float = 5.0) -> ChildPoll:
        with self._mu:
            rec = self._children.get(child_id)
            if rec is None:
                raise UnknownChild(child_id)
            if child_id in self._final:
                return self._final[child_id]
            live = process_liveness(rec.pid)
            if live != "dead":
                self._signal_group(rec, signal.SIGTERM)
                deadline = time.monotonic() + grace_s
                while time.monotonic() < deadline and process_liveness(rec.pid) != "dead":
                    _try_waitpid(rec.pid)
                    time.sleep(0.05)
                if process_liveness(rec.pid) != "dead":
                    self._signal_group(rec, signal.SIGKILL)
                    time.sleep(0.05)
            exit_code = self._exit_code(rec)
            if process_liveness(rec.pid) == "zombie":
                exit_code = self._exit_code(rec)
            return self._finalize(
                rec,
                "failed",
                reason="killed",
                exit_code=exit_code,
            )

    def _signal_group(self, rec: ChildRecord, sig: int) -> None:
        try:
            os.killpg(rec.pgid, sig)
        except ProcessLookupError:
            try:
                os.kill(rec.pid, sig)
            except ProcessLookupError:
                return
        except PermissionError:
            try:
                os.kill(rec.pid, sig)
            except (ProcessLookupError, PermissionError):
                return

    def shutdown(self, *, kill: bool = True) -> None:
        with self._mu:
            ids = list(self._children)
        for child_id in ids:
            if kill:
                try:
                    self.kill(child_id, grace_s=2.0)
                except UnknownChild:
                    continue
            else:
                try:
                    self.poll(child_id)
                except UnknownChild:
                    continue

    def wait(self, child_id: str, *, timeout_s: float | None = None) -> ChildPoll:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            poll = self.poll(child_id)
            if poll.state != "running":
                return poll
            if deadline is not None and time.monotonic() >= deadline:
                return poll
            time.sleep(0.05)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _tail(text: str, n: int) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= n:
        return text
    return text[-n:]


# ---------------------------------------------------------------------------
# Stub child — same flags as the real binary, for tests and lock pipelining.
# ---------------------------------------------------------------------------


def stub_child_main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    prompt = "stub"
    sleep_s = 0.0
    die = False
    alloc_mb = 0
    exit_code = 0
    max_new = 8
    out_path: Path | None = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--prompt" and i + 1 < len(args):
            prompt = args[i + 1]
            i += 2
            continue
        if a == "--sleep" and i + 1 < len(args):
            sleep_s = float(args[i + 1])
            i += 2
            continue
        if a == "--alloc-mb" and i + 1 < len(args):
            alloc_mb = int(args[i + 1])
            i += 2
            continue
        if a == "--exit-code" and i + 1 < len(args):
            exit_code = int(args[i + 1])
            i += 2
            continue
        if a == "--max-new-tokens" and i + 1 < len(args):
            max_new = int(args[i + 1])
            i += 2
            continue
        if a == "--out" and i + 1 < len(args):
            out_path = Path(args[i + 1])
            i += 2
            continue
        if a == "--die":
            die = True
            i += 1
            continue
        if a in {
            "--artifact-root",
            "--tokenizer",
            "--max-seq-len",
        } and i + 1 < len(args):
            i += 2
            continue
        if a == "--raw-prompt":
            i += 1
            continue
        i += 1
    blob = None
    if alloc_mb > 0:
        blob = bytearray(alloc_mb * 1024 * 1024)
        blob[0] = 1
        blob[-1] = 2
    if die:
        os.kill(os.getpid(), signal.SIGKILL)
    if sleep_s > 0:
        time.sleep(sleep_s)
    text = f"stub:{prompt}"
    wall_ns = int(sleep_s * 1e9)
    print(f"{TEXT_MARKER} {text}")
    print(f"{FALLBACKS_MARKER} 0")
    print("DENSE_W_MATERIALIZED: 0")
    print(f"NEW_TOKENS: {list(range(max_new))}")
    print(f"WALL_NS: {wall_ns}")
    print("PREFILL_WALL_NS: 0")
    print(f"DECODE_WALL_NS: {wall_ns}")
    print("STEADY_DECODE_WALL_NS_PER_TOKEN: Some(0)")
    sys.stdout.flush()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"generated_text": text, "wall_ns": wall_ns}, indent=2)
        )
    del blob
    return exit_code


# ---------------------------------------------------------------------------
# Measurement helpers (used by tools/genesis_pool.py)
# ---------------------------------------------------------------------------


def recommended_safe_n(seq_len: int) -> int:
    """Measured default. N=4 at seq-len 2048 already swapped; KV is not the lever."""
    del seq_len
    return MEASURED_SAFE_N


def kv_bytes_estimate(seq_len: int) -> int:
    return KV_BYTES_PER_POSITION_ESTIMATE * int(seq_len)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stub-child":
        raise SystemExit(stub_child_main(sys.argv[2:]))
    print(
        "use tools/genesis_pool.py for spawn/poll/kill/measure; "
        "this module is the library plus stub-child",
        file=sys.stderr,
    )
    raise SystemExit(2)
