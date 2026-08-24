#!/usr/bin/env python3
"""Observed executable closure of the sealed uniform-q4-v1 program.

The claim: the artifact is a compiled program whose whole closure is hashed,
not a model file sitting next to loose helpers.

The method: observe the files a process actually opens (DYLD __interpose on
the native decode binary; instrumented open on the I/O half of load), hash
that set, then remove each hashed member on a COPY and show execution breaks.

Do not build the hashed set by grepping the loader source. If execution reads
a file that is not hashed, the closure is incomplete and the gate FAILS.

    python3 tools/headless/noetic_closure.py
    python3 -m pytest tools/headless -q

Identity is sha256(file bytes). A prior attempt keyed identity on st_dev, a
mount artifact, and paid ~28s of startup for nothing.
"""
from __future__ import annotations

import builtins
import hashlib
import json
import os
import shutil
import stat as statmod
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "hawking.headless.noetic_closure.v1"
MIXED_CATALOG = "catalog.hq38m20"
RECEIPT_NAME = "NOETIC_CLOSURE.json"
DEFAULT_ARTIFACT = Path.home() / "models" / "qwen38-gravity-uniform-q4-v1"
DEFAULT_TOKENIZER = (
    Path.home() / "models" / "qwen3.8-27b-abliterated-bf16" / "tokenizer.json"
)
MODELS_ROOT = Path.home() / "models"
READ_OPS = frozenset({"open", "openat", "fopen"})


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "tools" / "headless").is_dir() and (p / "Cargo.toml").is_file():
            return p
    return Path.cwd()


def git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNKNOWN"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 8 << 20) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def merkle(entries: list[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for ident, digest in sorted(entries, key=lambda x: x[0]):
        raw = ident.encode()
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
        h.update(bytes.fromhex(digest))
    return h.hexdigest()


def resolve_path(path: object) -> str:
    """Resolve without going through a live OpenWatcher (os.path, not Path)."""
    s = os.fspath(path)
    try:
        return os.path.realpath(s)
    except OSError:
        return os.path.abspath(s)


def under(path: Path, root: Path, *, follow_symlinks: bool = True) -> bool:
    try:
        p = path.resolve() if follow_symlinks else Path(os.path.abspath(path))
        r = root.resolve() if follow_symlinks else Path(os.path.abspath(root))
        p.relative_to(r)
        return True
    except (ValueError, OSError):
        return False


def assert_not_under_models(path: Path) -> None:
    """Refuse to unlink a path whose location (not its symlink target) is ~/models.

    A symlink in /tmp pointing at ~/models is safe to unlink: unlink removes
    the symlink, not the target. Path.resolve() would follow it and false-trip.
    """
    if MODELS_ROOT.is_dir() and under(path, MODELS_ROOT, follow_symlinks=False):
        raise RuntimeError(f"refusing to mutate {path} (under ~/models)")


# ---------------------------------------------------------------------------
# observed opens
# ---------------------------------------------------------------------------


class OpenWatcher:
    """Record builtins.open and os.open issued by this process.

    Python 3.14 FileIO talks to the kernel directly: wrapping only os.open
    misses `open()` and Path.read_bytes. Wrapping stat/lstat is forbidden —
    Path.resolve uses them and would recurse. Callers that want a path
    observed must go through open()/os.open, not Path.read_bytes.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, str]] = []
        self._real_os_open = os.open
        self._real_builtin_open = builtins.open
        self._depth = 0

    def __enter__(self) -> "OpenWatcher":
        os.open = self._os_open  # type: ignore[assignment]
        builtins.open = self._builtin_open  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        os.open = self._real_os_open  # type: ignore[assignment]
        builtins.open = self._real_builtin_open  # type: ignore[assignment]

    def _os_open(self, path, flags, *a, **k):  # noqa: ANN001
        self._depth += 1
        try:
            if self._depth == 1:
                self.events.append({"op": "open", "path": os.fspath(path)})
            return self._real_os_open(path, flags, *a, **k)
        finally:
            self._depth -= 1

    def _builtin_open(self, file, *a, **k):  # noqa: ANN001
        self._depth += 1
        try:
            if self._depth == 1:
                self.events.append({"op": "open", "path": os.fspath(file)})
            return self._real_builtin_open(file, *a, **k)
        finally:
            self._depth -= 1


def unique_read_paths(events: Iterable[dict[str, str]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for ev in events:
        if ev.get("op") not in READ_OPS:
            continue
        raw = ev.get("path") or ""
        if not raw:
            continue
        p = resolve_path(raw)
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def is_model_specific(path: str, artifact: Path, tokenizer: Path) -> bool:
    try:
        rp = Path(path).resolve()
    except OSError:
        return False
    if rp == tokenizer.resolve():
        return True
    return under(rp, artifact)


def ident_for(path: Path, artifact: Path, tokenizer: Path) -> str:
    rp = path.resolve()
    if rp == tokenizer.resolve():
        return "tokenizer.json"
    try:
        rel = rp.relative_to(artifact.resolve())
        return f"artifact/{rel.as_posix()}"
    except ValueError:
        return rp.as_posix()


# ---------------------------------------------------------------------------
# I/O executor — path discovery happens at runtime from the manifest bytes
# ---------------------------------------------------------------------------


def execute_load_io(
    artifact: Path,
    tokenizer: Path,
    *,
    consume: str = "open",
) -> dict[str, Any]:
    """I/O half of load + generate.

    Discovers tensor paths by reading manifest.json, not from a hardcoded list.
    consume:
      "open"  — open each member read-only (removal tests)
      "hash"  — read every byte and sha256 it (closure construction)
    """
    if consume not in {"open", "hash"}:
        raise ValueError(consume)
    mixed = artifact / MIXED_CATALOG
    mixed_present = mixed.is_file()
    members: list[dict[str, Any]] = []
    if mixed_present:
        if consume == "hash":
            digest, n = sha256_file(mixed)
        else:
            fd = os.open(mixed, os.O_RDONLY)
            os.close(fd)
            digest, n = "", mixed.stat().st_size
        members.append(
            {
                "ident": ident_for(mixed, artifact, tokenizer),
                "path": str(mixed.resolve()),
                "sha256": digest,
                "bytes": n,
                "role": "mixed_catalog",
            }
        )
        tok = Path(tokenizer)
        if consume == "hash":
            digest, n = sha256_file(tok)
        else:
            fd = os.open(tok, os.O_RDONLY)
            os.close(fd)
            digest, n = "", tok.stat().st_size
        members.append(
            {
                "ident": ident_for(tok, artifact, tokenizer),
                "path": str(tok.resolve()),
                "sha256": digest,
                "bytes": n,
                "role": "tokenizer",
            }
        )
        return {
            "ok": True,
            "diverted_to_mixed": True,
            "members": members,
            "tensor_count": 0,
        }

    man_path = artifact / "manifest.json"
    with open(man_path, "rb") as fh:
        raw = fh.read()
    if consume == "hash":
        digest, n = sha256_bytes(raw), len(raw)
    else:
        digest, n = "", len(raw)
    members.append(
        {
            "ident": ident_for(man_path, artifact, tokenizer),
            "path": str(man_path.resolve()),
            "sha256": digest,
            "bytes": n,
            "role": "manifest",
        }
    )
    doc = json.loads(raw)
    rows = doc["tensors"]
    tensors_dir = artifact / "tensors"
    for i, row in enumerate(rows):
        path = tensors_dir / row["artifact"]
        if consume == "hash":
            digest, n = sha256_file(path)
            if (i + 1) % 150 == 0 or i + 1 == len(rows):
                print(f"  hashed tensors {i + 1}/{len(rows)}", flush=True)
        else:
            fd = os.open(path, os.O_RDONLY)
            os.close(fd)
            digest, n = "", path.stat().st_size
        rec = {
            "ident": ident_for(path, artifact, tokenizer),
            "path": str(path.resolve()),
            "sha256": digest,
            "bytes": n,
            "role": "tensor",
            "tensor_name": row.get("name"),
            "kind": row.get("kind"),
        }
        members.append(rec)
    tok = Path(tokenizer)
    if consume == "hash":
        digest, n = sha256_file(tok)
    else:
        fd = os.open(tok, os.O_RDONLY)
        os.close(fd)
        digest, n = "", tok.stat().st_size
    members.append(
        {
            "ident": ident_for(tok, artifact, tokenizer),
            "path": str(tok.resolve()),
            "sha256": digest,
            "bytes": n,
            "role": "tokenizer",
        }
    )
    return {
        "ok": True,
        "diverted_to_mixed": False,
        "members": members,
        "tensor_count": len(rows),
        "manifest_schema": doc.get("schema"),
    }


def execute_load_io_watched(
    artifact: Path,
    tokenizer: Path,
    *,
    consume: str = "open",
) -> dict[str, Any]:
    watcher = OpenWatcher()
    err: str | None = None
    result: dict[str, Any] | None = None
    with watcher:
        try:
            result = execute_load_io(artifact, tokenizer, consume=consume)
        except (OSError, KeyError, json.JSONDecodeError, TypeError) as exc:
            err = f"{type(exc).__name__}: {exc}"
    out: dict[str, Any] = result or {
        "ok": False,
        "diverted_to_mixed": False,
        "members": [],
        "tensor_count": 0,
    }
    if err is not None:
        out["ok"] = False
        out["error"] = err
    out["watcher_events"] = watcher.events
    out["watcher_read_paths"] = unique_read_paths(watcher.events)
    return out


# ---------------------------------------------------------------------------
# native process observation (DYLD __interpose)
# ---------------------------------------------------------------------------


def interpose_source() -> Path:
    return Path(__file__).resolve().with_name("noetic_closure_interpose.c")


def compile_interpose(dest_dir: Path) -> Path:
    src = interpose_source()
    dylib = dest_dir / "noetic_closure_openlog.dylib"
    cmd = ["clang", "-dynamiclib", "-o", str(dylib), str(src)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"clang interpose failed: {proc.stderr or proc.stdout}")
    return dylib


def parse_openlog(text: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line or line.startswith("CTOR "):
            continue
        op, _, rest = line.partition(" ")
        path = rest.strip()
        if not op or not path:
            continue
        events.append({"op": op, "path": resolve_path(path)})
    return events


def locate_decode_binary(repo: Path) -> Path | None:
    names = [
        repo / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy",
        Path(
            "/Users/scammermike/Downloads/hawking-copy/workspace/ops/build/rust/"
            "release-fast/examples/ascension_qwen38_hybrid_greedy"
        ),
        Path(
            "/Users/scammermike/Downloads/hawking/workspace/ops/build/rust/"
            "release-fast/examples/ascension_qwen38_hybrid_greedy"
        ),
        repo / "workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy",
    ]
    for p in names:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def trace_native(
    binary: Path,
    argv: list[str],
    *,
    timeout: int = 120,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Spawn an unsigned binary with DYLD __interpose and collect opens."""
    tmp = Path(tempfile.mkdtemp(prefix="noetic_closure_trace_")) if work_dir is None else work_dir
    log_path = tmp / "openlog.txt"
    try:
        dylib = compile_interpose(tmp)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return {
            "ok": False,
            "error": f"clang interpose failed: {exc}",
            "events": [],
            "read_paths": [],
        }
    env = os.environ.copy()
    env["OPENLOG_PATH"] = str(log_path)
    env["DYLD_INSERT_LIBRARIES"] = str(dylib)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(binary), *argv],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        elapsed = round(time.time() - t0, 3)
        text = log_path.read_text() if log_path.is_file() else ""
        events = parse_openlog(text)
        return {
            "ok": True,
            "binary": str(binary),
            "argv": argv,
            "exit_code": proc.returncode,
            "elapsed_s": elapsed,
            "stdout": (proc.stdout or "")[:4000],
            "stderr": (proc.stderr or "")[:4000],
            "events": events,
            "read_paths": unique_read_paths(events),
            "log_bytes": len(text),
        }
    except subprocess.TimeoutExpired as exc:
        text = log_path.read_text() if log_path.is_file() else ""
        events = parse_openlog(text)
        return {
            "ok": False,
            "error": f"timeout: {exc}",
            "binary": str(binary),
            "events": events,
            "read_paths": unique_read_paths(events),
        }
    finally:
        if work_dir is None:
            shutil.rmtree(tmp, ignore_errors=True)


def trace_native_helper_open(path: Path, work_dir: Path) -> dict[str, Any]:
    """Compile a tiny unsigned helper that opens argv[1]; prove interpose sees it."""
    helper_c = work_dir / "helper.c"
    helper = work_dir / "helper"
    helper_c.write_text(
        "#include <fcntl.h>\n#include <unistd.h>\n"
        "int main(int argc, char **argv) {\n"
        "  if (argc < 2) return 2;\n"
        "  int fd = open(argv[1], O_RDONLY);\n"
        "  if (fd >= 0) close(fd);\n"
        "  return 0;\n"
        "}\n"
    )
    proc = subprocess.run(
        ["clang", "-o", str(helper), str(helper_c)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"clang helper failed: {proc.stderr or proc.stdout}")
    return trace_native(helper, [str(path)], work_dir=work_dir, timeout=15)


# ---------------------------------------------------------------------------
# compare + removal
# ---------------------------------------------------------------------------


def compare_sets(
    observed: Iterable[str],
    hashed: Iterable[str],
) -> dict[str, Any]:
    obs = {resolve_path(p) for p in observed}
    hsh = {resolve_path(p) for p in hashed}
    read_not_hashed = sorted(obs - hsh)
    hashed_not_read = sorted(hsh - obs)
    return {
        "read_but_not_hashed": read_not_hashed,
        "hashed_but_not_read": hashed_not_read,
        "n_observed": len(obs),
        "n_hashed": len(hsh),
        "n_read_but_not_hashed": len(read_not_hashed),
        "n_hashed_but_not_read": len(hashed_not_read),
    }


def make_shadow(artifact: Path, dest: Path) -> None:
    """Symlink every regular file under artifact into dest. Never copies bytes."""
    dest.mkdir(parents=True)
    for dirpath, dirnames, filenames in os.walk(artifact, followlinks=False):
        dirnames.sort()
        rel_dir = Path(dirpath).relative_to(artifact)
        here = dest / rel_dir
        here.mkdir(parents=True, exist_ok=True)
        for name in sorted(filenames):
            src = Path(dirpath) / name
            try:
                st = src.lstat()
            except OSError:
                continue
            if not statmod.S_ISREG(st.st_mode):
                continue
            os.symlink(src.resolve(), here / name)


def removal_test_each(
    artifact: Path,
    tokenizer: Path,
    members: list[dict[str, Any]],
) -> dict[str, Any]:
    """On a COPY, drop each hashed member and show the I/O executor breaks.

    Never unlinks anything under ~/models.
    """
    tmp = Path(tempfile.mkdtemp(prefix="noetic_closure_rm_"))
    trials: list[dict[str, Any]] = []
    ceremony: list[str] = []
    try:
        shadow = tmp / "artifact"
        make_shadow(artifact, shadow)
        tok_src = Path(tokenizer).resolve()
        tok_shadow = tmp / "tokenizer.json"
        os.symlink(tok_src, tok_shadow)
        originals_ok = True
        for m in members:
            ident = m["ident"]
            orig = Path(m["path"])
            if not orig.is_file():
                originals_ok = False
                trials.append(
                    {
                        "ident": ident,
                        "broke": False,
                        "error": "original missing before trial",
                    }
                )
                ceremony.append(ident)
                continue
            if ident == "tokenizer.json":
                target = tok_shadow
            else:
                rel = ident[len("artifact/") :] if ident.startswith("artifact/") else Path(m["path"]).name
                target = shadow / rel
            assert_not_under_models(target)
            if not target.exists() and not target.is_symlink():
                trials.append(
                    {
                        "ident": ident,
                        "broke": False,
                        "error": f"shadow missing {target}",
                    }
                )
                ceremony.append(ident)
                continue
            os.unlink(target)
            ran = execute_load_io_watched(shadow, tok_shadow, consume="open")
            broke = ran.get("ok") is not True
            trials.append(
                {
                    "ident": ident,
                    "broke": broke,
                    "error": None if broke else "execution still succeeded",
                    "executor_error": ran.get("error"),
                }
            )
            if not broke:
                ceremony.append(ident)
            os.symlink(orig.resolve(), target)
            if not orig.is_file():
                originals_ok = False
                raise RuntimeError(f"original disappeared during trial: {orig}")
            if len(trials) % 100 == 0 or len(trials) == len(members):
                print(f"  removal {len(trials)}/{len(members)}", flush=True)
        return {
            "copy_only": True,
            "original_untouched": originals_ok,
            "n_members": len(members),
            "n_broke": sum(1 for t in trials if t["broke"]),
            "n_ceremony": len(ceremony),
            "all_load_bearing": not ceremony and len(trials) == len(members),
            "ceremony_members": ceremony,
            "trials": trials,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def identity_finding(repo: Path) -> dict[str, Any]:
    """Content identity vs the mount-artifact key that cost 28s of startup."""
    warm = (
        repo
        / "crates/hawking-core/src/model/qwen_complete_binary/admission_warm_receipt.rs"
    )
    note = (
        "A prior closure attempt keyed identity on st_dev, a mount artifact, "
        "and cost ~28 seconds of startup for nothing. This harness hashes "
        "sha256(file bytes) only."
    )
    finding: dict[str, Any] = {
        "closure_identity": "sha256_of_file_bytes",
        "not_used": ["st_dev", "st_ino", "mtime_ns", "ctime_ns"],
        "why": note,
        "admission_warm_receipt_path": str(warm),
        "admission_warm_receipt_present": warm.is_file(),
    }
    if warm.is_file():
        text = warm.read_text(errors="replace")
        finding["admission_warm_receipt"] = {
            "records_st_dev": "device: metadata.dev()" in text or "device:" in text,
            "matches_st_dev": "observed.size == expected.size" in text
            and "observed.inode == expected.inode" in text
            and "device (st_dev) is deliberately NOT compared" in text,
            "match_key": ["size", "mtime_ns", "inode"],
            "device_excluded_because": "mount-time artifact; remount reassigns it",
            "match_key_is_content": False,
            "note": (
                "Warm admission still keys skip-rehash on size+mtime+inode, "
                "not on content. That is cheap, but it is not a property of "
                "the bytes. The closure hash is the content."
            ),
        }
    return finding


def hashed_members_from_observation(
    io_run: dict[str, Any],
    artifact: Path,
    tokenizer: Path,
) -> list[dict[str, Any]]:
    """Hashed set = model-specific files the I/O executor actually opened.

    Members already carry content hashes when consume='hash'. We still require
    that the watcher saw an open of each path — a member the replica claimed
    to hash without an observed open is dropped (and surfaces as hashed-but-
    not-read if something else put it in).
    """
    observed = set(io_run.get("watcher_read_paths") or [])
    out: list[dict[str, Any]] = []
    for m in io_run.get("members") or []:
        path = resolve_path(m["path"])
        if not is_model_specific(path, artifact, tokenizer):
            continue
        if path not in observed:
            continue
        rec = {
            "ident": m["ident"],
            "path": path,
            "sha256": m["sha256"],
            "bytes": m["bytes"],
            "role": m["role"],
            "model_specific": True,
        }
        if m.get("tensor_name"):
            rec["tensor_name"] = m["tensor_name"]
        if m.get("kind"):
            rec["kind"] = m["kind"]
        if rec["ident"] == "tokenizer.json" and not under(Path(path), artifact):
            rec["outside_artifact"] = True
            rec["why"] = (
                "generate opens tokenizer.json from the bf16 parent, not from "
                "the artifact root. A loose helper next to the model. Binding "
                "it into the closure hash is what makes the program a hashed "
                "unit instead of a weight file plus friends."
            )
        out.append(rec)
    return out


def live_model_specific_reads(
    live: dict[str, Any],
    artifact: Path,
    tokenizer: Path,
) -> list[str]:
    paths: list[str] = []
    for p in live.get("read_paths") or []:
        if is_model_specific(p, artifact, tokenizer) and Path(p).is_file():
            paths.append(resolve_path(p))
    return paths


def build_receipt(
    *,
    repo: Path,
    artifact: Path,
    tokenizer: Path,
    io_run: dict[str, Any],
    members: list[dict[str, Any]],
    live: dict[str, Any],
    removal: dict[str, Any],
    elapsed_s: float,
    receipt_path: Path,
    decode_bin: Path | None,
) -> dict[str, Any]:
    hashed_paths = [m["path"] for m in members]
    io_obs = [
        p
        for p in (io_run.get("watcher_read_paths") or [])
        if is_model_specific(p, artifact, tokenizer) and Path(p).is_file()
    ]
    live_obs = live_model_specific_reads(live, artifact, tokenizer)
    io_cmp = compare_sets(io_obs, hashed_paths)
    live_cmp = compare_sets(live_obs, hashed_paths)
    gate_fail = io_cmp["n_read_but_not_hashed"] > 0 or live_cmp["n_read_but_not_hashed"] > 0
    program: dict[str, Any] | None = None
    if decode_bin is not None and decode_bin.is_file():
        dgst, n = sha256_file(decode_bin)
        program = {
            "ident": "runtime/ascension_qwen38_hybrid_greedy",
            "path": str(decode_bin.resolve()),
            "sha256": dgst,
            "bytes": n,
            "why": "the compiled program; shaders and geometry are include_str'd into it",
        }
    closure_entries = [(m["ident"], m["sha256"]) for m in members]
    if program is not None:
        closure_entries.append((program["ident"], program["sha256"]))
    mixed = artifact / MIXED_CATALOG
    tokenizer_outside = not under(Path(tokenizer), artifact)
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(repo),
        "elapsed_s": elapsed_s,
        "receipt_path": str(receipt_path),
        "claim": (
            "The artifact is a compiled program whose whole closure is hashed, "
            "not a model file sitting next to loose helpers."
        ),
        "identity": identity_finding(repo),
        "artifact": {
            "path": str(artifact.resolve()),
            "tokenizer": str(Path(tokenizer).resolve()),
            "tokenizer_outside_artifact": tokenizer_outside,
            "mixed_catalog_present": mixed.is_file(),
            "decode_binary": None if decode_bin is None else str(decode_bin),
        },
        "observation": {
            "method": (
                "Two observed processes, neither built by grepping the loader. "
                "(1) DYLD __interpose of open/openat/stat/lstat/getattrlist/"
                "fopen into the unsigned ascension_qwen38_hybrid_greedy. "
                "(2) Instrumented os.open/builtins.open around an I/O executor "
                "that discovers tensor paths by reading the manifest it just "
                "opened. Hashed members are the model-specific files process "
                "(2) actually opened. Process (1) dies at MetalContext::new "
                "before the tensor loop; that is reported as HASHED BUT NOT "
                "READ on the live binary, not hidden."
            ),
            "live_decode": {
                "ok": live.get("ok"),
                "binary": live.get("binary"),
                "exit_code": live.get("exit_code"),
                "elapsed_s": live.get("elapsed_s"),
                "stderr": live.get("stderr", "")[:2000],
                "error": live.get("error"),
                "n_events": len(live.get("events") or []),
                "n_read_paths": len(live.get("read_paths") or []),
                "model_specific_reads": live_obs,
                "metal_refused": "no Metal-capable GPU" in (live.get("stderr") or ""),
            },
            "io_executor": {
                "ok": io_run.get("ok"),
                "diverted_to_mixed": io_run.get("diverted_to_mixed"),
                "tensor_count": io_run.get("tensor_count"),
                "manifest_schema": io_run.get("manifest_schema"),
                "watcher_event_count": len(io_run.get("watcher_events") or []),
                "n_read_paths": len(io_run.get("watcher_read_paths") or []),
                "error": io_run.get("error"),
            },
        },
        "compiled_program": program,
        "hashed_members": members,
        "n_hashed_members": len(members),
        "closure_sha256": merkle(closure_entries),
        "compare": {
            "io_executor": io_cmp,
            "live_decode": live_cmp,
            "gate_fails_on": "READ BUT NOT HASHED (closure incomplete)",
            "hashed_but_not_read_severity": "less severe; ceremony / masked by Metal death",
            "gate": "FAIL" if gate_fail else "PASS",
        },
        "removal": removal,
        "escapes": [
            "Live GPU decode is not observed here (MetalContext::new refuses a GPU). "
            "Tensor opens come from the I/O executor, not from a Metal session. A "
            "loader that opened extra files only after MetalContext::new would not "
            "have been watched in the live process.",
            "In-memory generated state that is not a file. None on this vehicle.",
            "Load does not consult the closure hash: corrupting bytes at a stable "
            "name-address still loads. Removal breaks; silent bitflips do not.",
        ],
        "did_not_modify": [
            str(MODELS_ROOT),
            "receipts/ascent-2026-08-16",
            "receipts/ascent-2026-08-18",
            "workspace/campaign",
        ],
    }
    return doc


def run(
    *,
    repo: Path | None = None,
    artifact: Path | None = None,
    tokenizer: Path | None = None,
    write_receipt: bool = True,
    do_removal: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    repo = repo or repo_root()
    artifact = Path(
        os.environ.get("NOETIC_ARTIFACT", str(artifact or DEFAULT_ARTIFACT))
    )
    tokenizer = Path(
        os.environ.get("NOETIC_TOKENIZER", str(tokenizer or DEFAULT_TOKENIZER))
    )
    receipt_path = repo / "receipts" / "headless" / RECEIPT_NAME
    if not artifact.is_dir():
        raise SystemExit(f"artifact root missing: {artifact}")
    if not tokenizer.is_file():
        raise SystemExit(f"tokenizer missing: {tokenizer}")

    decode_bin = locate_decode_binary(repo)
    live: dict[str, Any]
    if decode_bin is None:
        live = {
            "ok": False,
            "error": "decode binary not found",
            "events": [],
            "read_paths": [],
        }
    else:
        print("== live decode (DYLD __interpose) ==", flush=True)
        live = trace_native(
            decode_bin,
            [
                "--artifact-root",
                str(artifact),
                "--tokenizer",
                str(tokenizer),
                "--prompt",
                "Hi",
                "--max-new-tokens",
                "1",
                "--max-seq-len",
                "32",
            ],
        )
        print(
            f"  exit={live.get('exit_code')} events={len(live.get('events') or [])} "
            f"reads={len(live.get('read_paths') or [])}",
            flush=True,
        )

    print("== I/O executor (instrumented open, discover from manifest) ==", flush=True)
    io_run = execute_load_io_watched(artifact, tokenizer, consume="hash")
    if not io_run.get("ok"):
        raise SystemExit(f"I/O executor failed: {io_run.get('error')}")
    members = hashed_members_from_observation(io_run, artifact, tokenizer)
    print(f"  hashed members: {len(members)}", flush=True)

    if do_removal:
        print("== removal test (copy only, every hashed member) ==", flush=True)
        removal = removal_test_each(artifact, tokenizer, members)
        print(
            f"  broke={removal['n_broke']}/{removal['n_members']} "
            f"ceremony={removal['n_ceremony']} "
            f"original_untouched={removal['original_untouched']}",
            flush=True,
        )
    else:
        removal = {
            "copy_only": True,
            "skipped": True,
            "n_members": len(members),
            "n_broke": 0,
            "n_ceremony": 0,
            "all_load_bearing": False,
            "ceremony_members": [],
            "trials": [],
        }

    elapsed = round(time.time() - t0, 3)
    doc = build_receipt(
        repo=repo,
        artifact=artifact,
        tokenizer=tokenizer,
        io_run=io_run,
        members=members,
        live=live,
        removal=removal,
        elapsed_s=elapsed,
        receipt_path=receipt_path,
        decode_bin=decode_bin,
    )
    if write_receipt:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = receipt_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n")
        tmp.replace(receipt_path)
        print(f"receipt {receipt_path}", flush=True)
    print_report(doc)
    return doc


def print_report(doc: dict[str, Any]) -> None:
    w = sys.stdout.write
    w("NOETIC CLOSURE\n")
    w("=" * 72 + "\n")
    w(f"schema     {doc['schema']}\n")
    w(f"generated  {doc['generated_at']}\n")
    w(f"head       {doc['git_head']}\n")
    w(f"elapsed_s  {doc['elapsed_s']}\n")
    w(f"receipt    {doc['receipt_path']}\n")
    w(f"gate       {doc['compare']['gate']}\n")
    w(f"closure    {doc['closure_sha256']}\n")
    w(f"members    {doc['n_hashed_members']}\n")
    ident = doc["identity"]
    w(f"identity   {ident['closure_identity']} (not {ident['not_used']})\n")
    w("\n")
    w("## READ BUT NOT HASHED  (incomplete closure — gate FAIL)\n")
    for label, cmp_ in doc["compare"].items():
        if not isinstance(cmp_, dict):
            continue
        names = cmp_.get("read_but_not_hashed") or []
        w(f"  {label}: {len(names)}\n")
        for p in names[:20]:
            w(f"    {p}\n")
    w("\n")
    w("## HASHED BUT NOT READ  (ceremony / Metal-masked)\n")
    for label, cmp_ in doc["compare"].items():
        if not isinstance(cmp_, dict):
            continue
        names = cmp_.get("hashed_but_not_read") or []
        w(f"  {label}: {len(names)}\n")
        for p in names[:8]:
            w(f"    {p}\n")
        if len(names) > 8:
            w(f"    … {len(names) - 8} more\n")
    w("\n")
    rem = doc["removal"]
    w("## REMOVAL (copy only)\n")
    w(
        f"  broke {rem.get('n_broke')}/{rem.get('n_members')} "
        f"ceremony={rem.get('n_ceremony')} "
        f"original_untouched={rem.get('original_untouched')}\n"
    )
    if rem.get("ceremony_members"):
        for ident in rem["ceremony_members"][:12]:
            w(f"    CEREMONY {ident}\n")
    live = doc["observation"]["live_decode"]
    w("\n")
    w("## LIVE DECODE\n")
    w(
        f"  ok={live.get('ok')} exit={live.get('exit_code')} "
        f"metal_refused={live.get('metal_refused')} "
        f"model_specific_reads={len(live.get('model_specific_reads') or [])}\n"
    )
    for p in live.get("model_specific_reads") or []:
        w(f"    {p}\n")
    w("=" * 72 + "\n")


def main() -> int:
    doc = run()
    rem = doc["removal"]
    gate = doc["compare"]["gate"]
    if gate == "FAIL":
        return 1
    if not rem.get("all_load_bearing"):
        return 1
    if not rem.get("original_untouched"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
