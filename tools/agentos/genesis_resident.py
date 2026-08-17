#!/usr/bin/env python3
"""Resident Genesis client, stub server, and launcher.

Production body is tools/agentos/genesis_body (one process, one weight set,
four isolated logical sessions). This module is the socket protocol + the
client the ascent loop calls. Liveness is always process state (kill 0 / the
live socket answering), never a status file.

    python3 tools/agentos/genesis_resident.py serve
    python3 tools/agentos/genesis_resident.py serve --stub
    python3 tools/agentos/genesis_resident.py health
    python3 tools/agentos/genesis_resident.py propose --prompt "..."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

_AGENTOS_DIR = Path(__file__).resolve().parent
if str(_AGENTOS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTOS_DIR))

from genesis_contract import (  # noqa: E402
    RESIDENT_SESSION_ROLES,
    contract_provenance,
    inject_runtime_contract,
)

# macOS sockaddr_un.sun_path is 104 bytes including NUL.
_UNIX_PATH_MAX = 103

PROTOCOL = "hawking.genesis.resident.v1"
SESSION_ROLES = RESIDENT_SESSION_ROLES
PARENT_HAWKING = Path("/Users/scammermike/Downloads/hawking")
CONNECT_TIMEOUT_S = 0.25
PROPOSE_TIMEOUT_S = 1800.0


def default_repo() -> Path:
    env = os.environ.get("HAWKING_REPO")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


def default_socket(repo: Path | None = None) -> Path:
    env = os.environ.get("GENESIS_RESIDENT_SOCK")
    if env:
        return Path(env)
    root = repo or default_repo()
    cand = root / "workspace" / "ops" / "genesis-resident.sock"
    if len(os.fsencode(str(cand))) < _UNIX_PATH_MAX:
        return cand
    digest = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]
    return Path(f"/tmp/hawking-genesis-{digest}.sock")


def default_stopfile(repo: Path | None = None) -> Path:
    env = os.environ.get("GENESIS_STOP")
    if env:
        return Path(env)
    return (repo or default_repo()) / "workspace" / "ops" / "GENESIS_STOP"


def default_lineage(repo: Path | None = None) -> Path:
    return (repo or default_repo()) / "receipts" / "ascent-2026-08-16" / "GENESIS_LINEAGE_CURRENT.json"


def _lineage_current(repo: Path) -> dict[str, Any] | None:
    """Return the seated CURRENT identity, failing closed on malformed state.

    A successor may carry a new artifact or resident executable.  Starting the
    static G0 paths after such a state has become authoritative is worse than a
    failed start: it would make the process look live while executing the wrong
    generation.  ``None`` means no lineage file exists yet; an empty mapping
    means a lineage file exists but cannot safely select a body.
    """
    path = default_lineage(repo)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
        slots = raw.get("slots")
        current = slots.get("CURRENT") if isinstance(slots, dict) else None
        if not isinstance(current, dict) or not isinstance(current.get("identity"), dict):
            return {}
        return current
    except (OSError, ValueError, TypeError):
        return {}


def _resolve_repo_path(value: object, repo: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = repo / candidate
    try:
        return candidate.resolve(strict=True)
    except OSError:
        return None


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def discover_artifact(repo: Path | None = None) -> Path | None:
    root = repo or default_repo()
    current = _lineage_current(root)
    if current is not None:
        identity = current.get("identity") if isinstance(current, dict) else None
        path = _resolve_repo_path(identity.get("artifact") if isinstance(identity, dict) else None, root)
        claimed = current.get("artifact_sha") if isinstance(current, dict) else None
        manifest = path / "manifest.json" if path is not None else None
        if (
            path is None
            or not path.is_dir()
            or manifest is None
            or not manifest.is_file()
            or not isinstance(claimed, str)
            or _sha256_file(manifest) != claimed
        ):
            return None
        return path
    for cand in (
        root / "workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1",
        PARENT_HAWKING / "workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1",
    ):
        if (cand / "manifest.json").is_file():
            return cand
    return None


def discover_tokenizer(repo: Path | None = None) -> Path | None:
    root = repo or default_repo()
    current = _lineage_current(root)
    if isinstance(current, dict):
        identity = current.get("identity")
        selected = _resolve_repo_path(
            identity.get("tokenizer") if isinstance(identity, dict) else None,
            root,
        )
        if selected is not None:
            if selected.is_file():
                return selected
            return None
    for cand in (
        root / "workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json",
        PARENT_HAWKING / "workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json",
    ):
        if cand.is_file():
            return cand
    return None


def discover_body_bin(repo: Path | None = None) -> Path | None:
    env = os.environ.get("GENESIS_BODY_BIN")
    if env:
        p = Path(env)
        if os.access(p, os.X_OK):
            return p
        return None
    root = repo or default_repo()
    current = _lineage_current(root)
    if current is not None:
        identity = current.get("identity") if isinstance(current, dict) else None
        selected = _resolve_repo_path(
            identity.get("resident_executable") if isinstance(identity, dict) else None,
            root,
        )
        claimed = current.get("runtime_sha") if isinstance(current, dict) else None
        if (
            selected is None
            or not selected.is_file()
            or not os.access(selected, os.X_OK)
            or not isinstance(claimed, str)
            or _sha256_file(selected) != claimed
        ):
            return None
        return selected
    here = Path(__file__).resolve().parent
    candidates = [
        root / "workspace/ops/build/rust/release/genesis-resident",
        PARENT_HAWKING / "workspace/ops/build/rust/release/genesis-resident",
        here / "genesis_body" / "target" / "release" / "genesis-resident",
        Path("/tmp/genesis-resident-target/release/genesis-resident"),
    ]
    target = os.environ.get("CARGO_TARGET_DIR")
    if target:
        candidates.insert(0, Path(target) / "release" / "genesis-resident")
    for cand in candidates:
        if os.access(cand, os.X_OK):
            return cand
    return None


def process_alive(pid: int) -> bool:
    """Kernel liveness. A pid that cannot be signaled is not live."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    try:
        out = os.popen(f"ps -o state= -p {int(pid)}").read().strip()
    except OSError:
        return False
    if not out:
        return False
    return not out.startswith("Z")


def _recv_line(sock: socket.socket) -> dict[str, Any]:
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("resident socket closed")
        buf += chunk
        if len(buf) > 16 * 1024 * 1024:
            raise ConnectionError("resident response exceeded 16 MiB")
    line, _sep, _rest = buf.partition(b"\n")
    return json.loads(line.decode())


def rpc(sock_path: Path, request: dict[str, Any], timeout: float) -> dict[str, Any]:
    payload = json.dumps(request, separators=(",", ":")) + "\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(sock_path))
        sock.sendall(payload.encode())
        return _recv_line(sock)


def health(sock_path: Path | None = None, timeout: float = CONNECT_TIMEOUT_S) -> dict[str, Any] | None:
    binding = contract_provenance()
    path = sock_path or default_socket()
    try:
        resp = rpc(path, {"op": "health"}, timeout=timeout)
    except (OSError, ConnectionError, TimeoutError, json.JSONDecodeError, socket.timeout):
        return None
    pid = resp.get("pid")
    try:
        pid_i = int(pid)
    except (TypeError, ValueError):
        return None
    if not process_alive(pid_i):
        return None
    if not resp.get("ok"):
        return None
    if resp.get("genesis_system_contract") != binding:
        return None
    return resp


def body_is_up(sock_path: Path | None = None) -> bool:
    info = health(sock_path)
    return bool(info and info.get("body_resident") and process_alive(int(info["pid"])))


def propose(
    prompt: str,
    *,
    sock_path: Path | None = None,
    max_new_tokens: int = 900,
    raw: bool = False,
    session: str = "parent",
    protected_capability: bool = False,
    timeout: float = PROPOSE_TIMEOUT_S,
) -> dict[str, Any] | None:
    path = sock_path or default_socket()
    if session not in SESSION_ROLES:
        raise ValueError(f"unknown Genesis session {session!r}; expected one of {SESSION_ROLES!r}")
    binding = contract_provenance()
    if protected_capability:
        if session != "protected_test":
            raise ValueError(
                "protected capability prompt preservation requires session='protected_test'"
            )
        wire_prompt = prompt
        wire_raw = bool(raw)
        contract_mode = "protected_capability_prompt_preserved"
    else:
        wire_prompt = inject_runtime_contract(
            prompt,
            role=session,
            raw_prompt=bool(raw),
        )
        wire_raw = True
        contract_mode = "runtime_capsule_injected"
    if not body_is_up(path):
        return None
    try:
        resp = rpc(
            path,
            {
                "op": "propose",
                "prompt": wire_prompt,
                "max_new_tokens": int(max_new_tokens),
                "raw": wire_raw,
                "session": session,
                "genesis_system_contract": binding,
                "genesis_contract_mode": contract_mode,
            },
            timeout=timeout,
        )
    except (OSError, ConnectionError, TimeoutError, json.JSONDecodeError, socket.timeout):
        return None
    if not resp.get("ok"):
        return None
    if resp.get("fallbacks") != 0:
        return None
    if resp.get("genesis_system_contract") != binding:
        return None
    if resp.get("genesis_contract_mode") != contract_mode:
        return None
    text = resp.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    pid = resp.get("pid")
    if pid is not None and not process_alive(int(pid)):
        return None
    return resp


def request_stop(sock_path: Path | None = None) -> bool:
    path = sock_path or default_socket()
    try:
        resp = rpc(path, {"op": "stop"}, timeout=2.0)
    except (OSError, ConnectionError, TimeoutError, json.JSONDecodeError, socket.timeout):
        return False
    return bool(resp.get("ok"))


def request_reload(
    sock_path: Path | None = None,
    *,
    artifact: Path | None = None,
    generation: int | None = None,
    artifact_sha: str | None = None,
) -> dict[str, Any] | None:
    path = sock_path or default_socket()
    req: dict[str, Any] = {"op": "reload"}
    if artifact is not None:
        req["artifact"] = str(artifact)
    if generation is not None:
        req["generation"] = int(generation)
    if artifact_sha is not None:
        req["artifact_sha"] = artifact_sha
    try:
        return rpc(path, req, timeout=PROPOSE_TIMEOUT_S)
    except (OSError, ConnectionError, TimeoutError, json.JSONDecodeError, socket.timeout):
        return None


class StubServer:
    """In-process / child stub that speaks the resident protocol without weights."""

    def __init__(
        self,
        sock_path: Path,
        stopfile: Path,
        *,
        load_s: float = 0.01,
        reply: str = "stub-genesis-proposal",
    ) -> None:
        self.sock_path = sock_path
        self.stopfile = stopfile
        self.load_s = load_s
        self.reply = reply
        self.started = time.monotonic()
        self.started_unix_ns = time.time_ns()
        self.load_count = 0
        self.serve_count = 0
        self.session_serve_counts = {role: 0 for role in SESSION_ROLES}
        self.load_ns = 0
        self.body_resident = False
        self.stop = False
        self.generation = 0
        self.artifact = "stub"
        self.artifact_sha = ""

    def _load(self) -> None:
        t0 = time.monotonic()
        time.sleep(self.load_s)
        self.load_ns = int((time.monotonic() - t0) * 1e9)
        self.load_count += 1
        self.body_resident = True

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "protocol": PROTOCOL,
            "pid": os.getpid(),
            "body_resident": self.body_resident,
            "uptime_ns": int((time.monotonic() - self.started) * 1e9),
            "started_unix_ns": self.started_unix_ns,
            "load_ns": self.load_ns,
            "load_count": self.load_count,
            "serve_count": self.serve_count,
            "resident_weight_bytes": 0,
            "workspace_bytes": 0,
            "session_count": len(SESSION_ROLES),
            "session_roles": list(SESSION_ROLES),
            "session_semantics": {
                "parent": {"kind": "lineage_parent", "front": "integration_profiling_synthesis"},
                "child_a": {"kind": "worker_session", "front": "gravity_doctor_representation"},
                "child_b": {"kind": "worker_session", "front": "execution_metal_kernel_runtime"},
                "protected_test": {"kind": "protected_test_slot", "front": "candidate_qualification_ab"},
            },
            "lineage_children": 0,
            "session_workspace_bytes": {role: 0 for role in SESSION_ROLES},
            "session_serve_counts": dict(self.session_serve_counts),
            "decode_concurrency": 1,
            "artifact": self.artifact,
            "artifact_sha": self.artifact_sha,
            "generation": self.generation,
            "max_seq_len": 4096,
            "genesis_system_contract": contract_provenance(),
            "stub": True,
        }

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        op = req.get("op")
        if op in ("health", "ping"):
            return self.health()
        if op == "propose":
            prompt = req.get("prompt") or ""
            if not prompt:
                return {"ok": False, "error": "prompt required", "protocol": PROTOCOL}
            session = str(req.get("session") or "parent")
            if session not in SESSION_ROLES:
                return {
                    "ok": False,
                    "error": f"unknown session role {session!r}",
                    "allowed_sessions": list(SESSION_ROLES),
                    "protocol": PROTOCOL,
                }
            self.serve_count += 1
            self.session_serve_counts[session] += 1
            return {
                "ok": True,
                "protocol": PROTOCOL,
                "text": self.reply,
                "fallbacks": 0,
                "wall_ns": 1,
                "new_tokens": [1],
                "prompt_len": 1,
                "serve_index": self.serve_count,
                "session": session,
                "session_serve_index": self.session_serve_counts[session],
                "load_count": self.load_count,
                "load_ns": self.load_ns,
                "body_resident": True,
                "pid": os.getpid(),
                "genesis_system_contract": req.get("genesis_system_contract"),
                "genesis_contract_mode": req.get("genesis_contract_mode"),
                "stub": True,
            }
        if op == "reload":
            self._load()
            self.generation = int(req.get("generation") or (self.generation + 1))
            if req.get("artifact"):
                self.artifact = str(req["artifact"])
            if req.get("artifact_sha"):
                self.artifact_sha = str(req["artifact_sha"])
            return self.health()
        if op == "stop":
            self.stop = True
            return {"ok": True, "stopped": True, "protocol": PROTOCOL, "pid": os.getpid()}
        return {"ok": False, "error": f"unknown op {op!r}", "protocol": PROTOCOL}

    def serve_forever(self) -> int:
        if self.stopfile.exists():
            print(json.dumps({"stopped": True, "reason": "GENESIS_STOP present"}), flush=True)
            return 0
        self._load()
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.sock_path.unlink()
        except FileNotFoundError:
            pass
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.sock_path))
        listener.listen(8)
        listener.settimeout(0.2)
        print(
            json.dumps(
                {
                    "listening": str(self.sock_path),
                    "pid": os.getpid(),
                    "stub": True,
                    "body_resident": True,
                    "load_ns": self.load_ns,
                }
            ),
            flush=True,
        )
        try:
            while not self.stop:
                if self.stopfile.exists():
                    print(json.dumps({"stopped": True, "reason": "GENESIS_STOP present"}), flush=True)
                    return 0
                try:
                    conn, _addr = listener.accept()
                except socket.timeout:
                    continue
                with conn:
                    buf = b""
                    conn.settimeout(30)
                    try:
                        while b"\n" not in buf:
                            chunk = conn.recv(65536)
                            if not chunk:
                                break
                            buf += chunk
                    except OSError:
                        continue
                    if not buf.strip():
                        continue
                    try:
                        req = json.loads(buf.split(b"\n", 1)[0].decode())
                    except json.JSONDecodeError as exc:
                        resp = {"ok": False, "error": f"bad json: {exc}", "protocol": PROTOCOL}
                    else:
                        resp = self.handle(req)
                    try:
                        conn.sendall((json.dumps(resp) + "\n").encode())
                    except OSError:
                        pass
        finally:
            listener.close()
            try:
                self.sock_path.unlink()
            except FileNotFoundError:
                pass
        return 0


def _exec_body(ns: argparse.Namespace) -> int:
    # Production resident startup is not allowed without the canonical Genesis
    # authority. Per-request injection verifies it again to catch later drift.
    binding = contract_provenance()
    repo = Path(ns.repo) if ns.repo else default_repo()
    binary = discover_body_bin(repo)
    if binary is None:
        print("genesis-resident: body binary not built", file=sys.stderr)
        print(
            "build: CARGO_TARGET_DIR=/tmp/genesis-resident-target "
            "cargo build --release --manifest-path tools/agentos/genesis_body/Cargo.toml",
            file=sys.stderr,
        )
        return 2
    artifact = Path(ns.artifact_root) if ns.artifact_root else discover_artifact(repo)
    tokenizer = Path(ns.tokenizer) if ns.tokenizer else discover_tokenizer(repo)
    if artifact is None or tokenizer is None:
        print("genesis-resident: artifact or tokenizer missing", file=sys.stderr)
        return 2
    sock = Path(ns.socket) if ns.socket else default_socket(repo)
    stopfile = Path(ns.stopfile) if ns.stopfile else default_stopfile(repo)
    lineage = Path(ns.lineage) if ns.lineage else default_lineage(repo)
    argv = [
        str(binary),
        "--artifact-root",
        str(artifact),
        "--tokenizer",
        str(tokenizer),
        "--socket",
        str(sock),
        "--stopfile",
        str(stopfile),
        "--repo",
        str(repo),
        "--genesis-contract-provenance",
        json.dumps(binding),
        "--max-seq-len",
        str(int(ns.max_seq_len)),
        "--max-new-tokens",
        str(int(ns.max_new_tokens)),
    ]
    if lineage.is_file():
        argv.extend(["--lineage", str(lineage)])
    if ns.follow_lineage:
        argv.append("--follow-lineage")
    os.execv(str(binary), argv)
    return 2


def _cmd_serve(ns: argparse.Namespace) -> int:
    if ns.stub:
        repo = Path(ns.repo) if ns.repo else default_repo()
        sock = Path(ns.socket) if ns.socket else default_socket(repo)
        stopfile = Path(ns.stopfile) if ns.stopfile else default_stopfile(repo)
        server = StubServer(sock, stopfile, reply=ns.stub_reply)
        return server.serve_forever()
    return _exec_body(ns)


def _cmd_health(ns: argparse.Namespace) -> int:
    repo = Path(ns.repo) if ns.repo else default_repo()
    sock = Path(ns.socket) if ns.socket else default_socket(repo)
    info = health(sock, timeout=2.0)
    if info is None:
        print(json.dumps({"ok": False, "body_resident": False, "reason": "not live"}))
        return 1
    print(json.dumps(info))
    return 0


def _cmd_propose(ns: argparse.Namespace) -> int:
    repo = Path(ns.repo) if ns.repo else default_repo()
    sock = Path(ns.socket) if ns.socket else default_socket(repo)
    resp = propose(
        ns.prompt,
        sock_path=sock,
        max_new_tokens=int(ns.max_new_tokens),
        raw=bool(ns.raw),
        session=ns.session,
        protected_capability=bool(ns.protected_capability),
    )
    if resp is None:
        print(json.dumps({"ok": False, "error": "resident not live or propose failed"}))
        return 1
    print(resp["text"])
    print(f"FALLBACKS: {resp.get('fallbacks', 0)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    repo = default_repo()
    ap = argparse.ArgumentParser(prog="genesis_resident")
    sub = ap.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve")
    serve.add_argument("--stub", action="store_true")
    serve.add_argument("--stub-reply", default="stub-genesis-proposal")
    serve.add_argument("--socket")
    serve.add_argument("--stopfile")
    serve.add_argument("--artifact-root")
    serve.add_argument("--tokenizer")
    serve.add_argument("--lineage")
    serve.add_argument(
        "--follow-lineage",
        action="store_true",
        help="legacy/manual polling mode; normal promotion is controller-driven",
    )
    serve.add_argument("--repo", default=str(repo))
    serve.add_argument("--max-seq-len", type=int, default=4096)
    serve.add_argument("--max-new-tokens", type=int, default=900)
    serve.set_defaults(func=_cmd_serve)

    h = sub.add_parser("health")
    h.add_argument("--socket")
    h.add_argument("--repo", default=str(repo))
    h.set_defaults(func=_cmd_health)

    p = sub.add_parser("propose")
    p.add_argument("--prompt", required=True)
    p.add_argument("--socket")
    p.add_argument("--repo", default=str(repo))
    p.add_argument("--max-new-tokens", type=int, default=900)
    p.add_argument("--raw", action="store_true")
    p.add_argument("--session", choices=SESSION_ROLES, default="parent")
    p.add_argument(
        "--protected-capability",
        action="store_true",
        help=(
            "preserve prompt bytes for a protected capability measurement; "
            "valid only with --session protected_test"
        ),
    )
    p.set_defaults(func=_cmd_propose)

    st = sub.add_parser("stop")
    st.add_argument("--socket")
    st.add_argument("--repo", default=str(repo))

    def _cmd_stop(ns: argparse.Namespace) -> int:
        sock = Path(ns.socket) if ns.socket else default_socket(Path(ns.repo) if ns.repo else repo)
        ok = request_stop(sock)
        print(json.dumps({"ok": ok}))
        return 0 if ok else 1

    st.set_defaults(func=_cmd_stop)

    ns = ap.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    sys.exit(main())
