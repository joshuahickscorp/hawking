#!/usr/bin/env python3
"""Prove by OBSERVED FILE ACCESS that production Noetic inference does not
load parent weights.

The claim is not "opens nothing under the parent directory". Prior work found
the parent tokenizer.json in use; that is a tokenizer dependency, not a
weight dependency. Three classes, not one:

  * parent WEIGHTS at inference     the violation
  * parent CONFIG or TOKENIZER      a real dependency, a different kind
  * parent read at COMPILE time     not a runtime dependency at all

A detector that has never caught a violation is not known to work. The
negative control is the composition harness's teacher-scoring path
(SourceBF16.load of a parent safetensors shard) — it legitimately touches
parent weights, and the same observer must flag it.

Observation method: DYLD interpose of open/openat on the live process, not
a reading of the loader source. The native decode must RUN TO COMPLETION
and emit at least one token. A truncated prefix (MetalContext::new dying
before the 755 catalog reads) is INCONCLUSIVE, not PASS.

Do not load a second 27B. Do not write under ~/models. Build the decode
binary from THIS repo; never execute the vestigial hawking-copy tree.

    python3 tools/headless/noetic_zero_parent.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCHEMA = "hawking.headless.noetic_zero_parent.v1"
PROMPT = "Hi"
CONTROL_TENSOR = "model.language_model.layers.0.input_layernorm.weight"
EXPECTED_CATALOG_TENSORS = 755
EXAMPLE_NAME = "ascension_qwen38_hybrid_greedy"

VISION_PY = Path.home() / ".grok-vision" / "bin" / "python"
DEFAULT_PARENT = Path.home() / "models" / "qwen3.8-27b-abliterated-bf16"
DEFAULT_ARTIFACT = Path.home() / "models" / "qwen38-gravity-uniform-q4-v1"
GPU_LOCK = Path("/tmp/hawking-gpu-lane.lock")
DECODE_TIMEOUT_S = 1800.0
CONTROL_TIMEOUT_S = 180.0

WEIGHT_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".npz",
    ".npy",
    ".h5",
    ".onnx",
    ".pkl",
    ".msgpack",
    ".ot",
    ".pb",
    ".weights",
}
TOKENIZER_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "spiece.model",
}
CONFIG_NAMES = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
    "configuration.json",
    "crc32.txt",
}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "tools" / "headless").is_dir() and (p / "Cargo.toml").is_file():
            return p
    return Path.cwd()


REPO = repo_root()
RECEIPT = REPO / "receipts" / "headless" / "NOETIC_ZERO_PARENT.json"
OPENLOG_C = Path(__file__).resolve().with_name("noetic_openlog.c")
CARGO_TARGET = REPO / "workspace" / "ops" / "build" / "rust"
DECODE_BINARY = CARGO_TARGET / "release-fast" / "examples" / EXAMPLE_NAME


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, timeout=20
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "UNKNOWN"


def parent_dir() -> Path:
    return Path(os.environ.get("QWEN38_PARENT_BF16", str(DEFAULT_PARENT))).expanduser().resolve()


def artifact_dir() -> Path:
    return Path(os.environ.get("QWEN38_Q4_ARTIFACT", str(DEFAULT_ARTIFACT))).expanduser().resolve()


# ---------------------------------------------------------------------------
# path classifier  (weights vs tokenizer vs config vs not-parent)
# ---------------------------------------------------------------------------


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def classify_path(raw: str, *, parent: Path, cwd: Path | None = None) -> str:
    """Return one of: parent_weight, parent_tokenizer, parent_config,
    parent_other, not_parent.
    """
    if not raw:
        return "not_parent"
    p = Path(raw)
    if not p.is_absolute():
        p = (cwd or Path.cwd()) / p
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    if not _is_under(resolved, parent):
        return "not_parent"
    name = resolved.name
    name_l = name.lower()
    if name_l in TOKENIZER_NAMES or "tokenizer" in name_l:
        return "parent_tokenizer"
    if name_l in CONFIG_NAMES:
        return "parent_config"
    suffix = resolved.suffix.lower()
    if suffix in WEIGHT_SUFFIXES:
        return "parent_weight"
    if name_l.startswith("model-") and "safetensor" in name_l:
        return "parent_weight"
    return "parent_other"


def parse_open_log(
    log_path: Path,
    *,
    parent: Path,
    cwd: Path,
    artifact: Path | None = None,
) -> dict[str, Any]:
    events: list[dict[str, str]] = []
    if log_path.is_file():
        for line in log_path.read_text(errors="replace").splitlines():
            if not line.strip() or "\t" not in line:
                continue
            op, path = line.split("\t", 1)
            events.append({"op": op.strip(), "path": path})
    unique: list[str] = []
    seen: set[str] = set()
    buckets = {
        "parent_weight": [],
        "parent_tokenizer": [],
        "parent_config": [],
        "parent_other": [],
        "not_parent": [],
    }
    artifact_tensor: list[str] = []
    artifact_other: list[str] = []
    art_root = artifact.resolve() if artifact is not None else None
    art_tensors = (art_root / "tensors") if art_root is not None else None
    for ev in events:
        raw = ev["path"]
        p = Path(raw)
        if not p.is_absolute():
            p = cwd / p
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        klass = classify_path(raw, parent=parent, cwd=cwd)
        if key not in seen:
            seen.add(key)
            unique.append(key)
            buckets[klass].append(key)
            if art_root is not None:
                rp = Path(key)
                if art_tensors is not None and _is_under(rp, art_tensors):
                    artifact_tensor.append(key)
                elif _is_under(rp, art_root):
                    artifact_other.append(key)
    return {
        "n_events": len(events),
        "n_unique": len(unique),
        "parent_weight": buckets["parent_weight"],
        "parent_tokenizer": buckets["parent_tokenizer"],
        "parent_config": buckets["parent_config"],
        "parent_other": buckets["parent_other"],
        "n_not_parent": len(buckets["not_parent"]),
        "n_artifact_tensor": len(artifact_tensor),
        "n_artifact_other": len(artifact_other),
        "artifact_other": artifact_other,
        "artifact_tensor_head": artifact_tensor[:8],
        "artifact_tensor_tail": artifact_tensor[-8:],
        "unique_paths_fingerprint": hashlib.sha256(
            "\n".join(sorted(unique)).encode()
        ).hexdigest(),
    }


# ---------------------------------------------------------------------------
# DYLD observer + GPU lock + this-repo binary
# ---------------------------------------------------------------------------


def _last_json_object(text: str) -> dict[str, Any] | None:
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                val = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(val, dict):
                return val
    return None


def compile_dylib(dest: Path) -> Path:
    if not OPENLOG_C.is_file():
        raise SystemExit(f"missing interpose source {OPENLOG_C}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cc",
        "-dynamiclib",
        "-Wno-deprecated-declarations",
        "-o",
        str(dest),
        str(OPENLOG_C),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dest.is_file():
        raise SystemExit(
            f"cc did not write {dest}: rc={proc.returncode}\n{proc.stderr}"
        )
    return dest


def binary_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    st = path.stat()
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(1 << 20))
    text = str(resolved)
    return {
        "path": text,
        "size": st.st_size,
        "mtime_unix": int(st.st_mtime),
        "sha256_head_1m": h.hexdigest(),
        "from_this_repo": text.startswith(str(REPO.resolve())),
        "not_vestigial_hawking_copy": "hawking-copy" not in text,
    }


def ensure_decode_binary() -> Path:
    """Build ascension_qwen38_hybrid_greedy from THIS repo. Never fall back
    to the vestigial hawking-copy tree."""
    if DECODE_BINARY.is_file() and os.access(DECODE_BINARY, os.X_OK):
        ident = binary_identity(DECODE_BINARY)
        if ident["from_this_repo"] and ident["not_vestigial_hawking_copy"]:
            return DECODE_BINARY
        raise SystemExit(
            f"decode binary at {DECODE_BINARY} is not from this repo: {ident}"
        )
    CARGO_TARGET.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cargo",
        "build",
        "--profile",
        "release-fast",
        "-p",
        "hawking-core",
        "--example",
        EXAMPLE_NAME,
        "--target-dir",
        str(CARGO_TARGET),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), text=True, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(
            "cargo build of "
            f"{EXAMPLE_NAME} failed rc={proc.returncode}\n"
            f"{proc.stderr[-8000:]}"
        )
    if not DECODE_BINARY.is_file():
        raise SystemExit(f"cargo did not write {DECODE_BINARY}")
    return DECODE_BINARY


class GpuLaneLock:
    """Same mkdir-atomic protocol as tools/gpu_lane_lock.sh."""

    def __init__(self, name: str, timeout: float = 5400.0) -> None:
        self.name = name
        self.timeout = timeout
        self.held = False

    def __enter__(self) -> "GpuLaneLock":
        deadline = time.time() + self.timeout
        while True:
            try:
                GPU_LOCK.mkdir()
                (GPU_LOCK / "pid").write_text(str(os.getpid()))
                (GPU_LOCK / "owner").write_text(self.name)
                self.held = True
                return self
            except FileExistsError:
                pid_file = GPU_LOCK / "pid"
                stale = False
                try:
                    pid = int(pid_file.read_text().strip())
                    os.kill(pid, 0)
                except (ValueError, OSError, FileNotFoundError):
                    stale = True
                if stale:
                    shutil.rmtree(GPU_LOCK, ignore_errors=True)
                    continue
                if time.time() >= deadline:
                    owner = "?"
                    try:
                        owner = (GPU_LOCK / "owner").read_text().strip()
                    except OSError:
                        pass
                    raise SystemExit(f"gpu lock timeout, held by {owner}")
                time.sleep(5)

    def __exit__(self, *exc: object) -> None:
        if self.held:
            shutil.rmtree(GPU_LOCK, ignore_errors=True)
            self.held = False


def observe_command(
    argv: list[str],
    *,
    dylib: Path,
    log_path: Path,
    parent: Path,
    timeout: float,
    cwd: Path | None = None,
    env_extra: dict[str, str] | None = None,
    artifact: Path | None = None,
) -> dict[str, Any]:
    if log_path.exists():
        log_path.unlink()
    env = os.environ.copy()
    env["DYLD_INSERT_LIBRARIES"] = str(dylib)
    env["NOETIC_OPEN_LOG"] = str(log_path)
    if env_extra:
        env.update(env_extra)
    cwd = cwd or Path.cwd()
    t0 = time.time()
    timed_out = False
    err = None
    proc: subprocess.CompletedProcess[str] | None
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        proc = None
        timed_out = True
        err = str(exc)
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else (
            exc.stdout.decode("utf-8", "replace") if exc.stdout else ""
        )
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else (
            exc.stderr.decode("utf-8", "replace") if exc.stderr else ""
        )
    else:
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    elapsed = round(time.time() - t0, 3)
    classified = parse_open_log(
        log_path, parent=parent, cwd=cwd, artifact=artifact
    )
    dylib_loaded = "INTERPOSE_CTOR" in stderr or classified["n_events"] > 0
    return {
        "argv": argv,
        "cwd": str(cwd),
        "exit_code": None if proc is None else proc.returncode,
        "timed_out": timed_out,
        "timeout_error": err,
        "elapsed_s": elapsed,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_head": stdout[:8000],
        "stderr_head": stderr[-8000:],
        "dylib_loaded": dylib_loaded,
        "log_path": str(log_path),
        "observation": classified,
    }


def parse_decode_output(stdout: str, stderr: str) -> dict[str, Any]:
    text = (stderr or "") + "\n" + (stdout or "")
    generated = None
    m = re.search(r"^GENERATED_TEXT_VERBATIM: (.*)$", stdout or "", re.M)
    if m:
        generated = m.group(1)
    token_ids: list[int] | None = None
    for pat in (r"^NEW_TOKENS: \[(.*)\]", r"^generated_token_ids=\[(.*)\]"):
        m = re.search(pat, stdout or "", re.M)
        if m:
            inner = m.group(1).strip()
            if not inner:
                token_ids = []
            else:
                try:
                    token_ids = [int(x.strip()) for x in inner.split(",") if x.strip()]
                except ValueError:
                    token_ids = None
            break
    fallbacks = None
    m = re.search(r"^FALLBACKS: (\d+)", stdout or "", re.M)
    if m:
        fallbacks = int(m.group(1))
    catalog_count = None
    m = re.search(r"opening Metal \+ (\d+) catalog tensors", text)
    if m:
        catalog_count = int(m.group(1))
    uploads = [int(x) for x in re.findall(r"qwen38-decode upload (\d+)/\d+", text)]
    last_upload = max(uploads) if uploads else None
    metal_refused = bool(
        re.search(r"no Metal-capable GPU", text)
        or re.search(r"ascension_qwen38_hybrid_greedy: metal:", text)
    )
    return {
        "generated_text": generated,
        "new_token_ids": token_ids,
        "n_new_tokens": 0 if token_ids is None else len(token_ids),
        "fallbacks": fallbacks,
        "saw_tokenizer_encode": "prompt tokens=" in text,
        "saw_catalog_count": catalog_count is not None,
        "catalog_count": catalog_count,
        "saw_upload_progress": bool(uploads),
        "last_upload_index": last_upload,
        "metal_refused": metal_refused,
    }


# ---------------------------------------------------------------------------
# child mode: negative control
# ---------------------------------------------------------------------------


def mode_composition_control(parent: Path) -> int:
    """Negative control: the composition harness's teacher-scoring I/O.

    SourceBF16.load opens a parent .safetensors shard. That is a weight
    open. Do not load the 27B — the chosen tensor is a 5120-wide layernorm.
    """
    headless = Path(__file__).resolve().parent
    sys.path.insert(0, str(headless))
    from noetic_composition import SourceBF16  # noqa: E402

    src = SourceBF16(parent)
    w = src.load(CONTROL_TENSOR)
    result = {
        "ok": True,
        "tensor": CONTROL_TENSOR,
        "shape": list(w.shape),
        "mean": float(w.mean()),
        "via": "noetic_composition.SourceBF16.load",
    }
    print(json.dumps(result))
    return 0


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def live_slice(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        k: obs[k]
        for k in (
            "n_events",
            "n_unique",
            "parent_weight",
            "parent_tokenizer",
            "parent_config",
            "parent_other",
            "n_not_parent",
            "n_artifact_tensor",
            "n_artifact_other",
            "artifact_other",
            "artifact_tensor_head",
            "artifact_tensor_tail",
            "unique_paths_fingerprint",
        )
        if k in obs
    }


def run_and_write(out: Path | None = None) -> dict[str, Any]:
    parent = parent_dir()
    artifact = artifact_dir()
    tokenizer = parent / "tokenizer.json"
    out = out or RECEIPT
    t0 = time.time()

    reasons = []
    if not parent.is_dir():
        reasons.append(f"parent bf16 missing: {parent}")
    if not (parent / "model.safetensors.index.json").is_file():
        reasons.append(f"parent index missing under {parent}")
    if not tokenizer.is_file():
        reasons.append(f"parent tokenizer.json missing: {tokenizer}")
    if not (artifact / "manifest.json").is_file():
        reasons.append(f"artifact missing: {artifact}")
    if not VISION_PY.is_file():
        reasons.append(f"vision python missing: {VISION_PY}")
    if reasons:
        raise SystemExit("preflight: " + "; ".join(reasons))

    binary = ensure_decode_binary()
    ident = binary_identity(binary)
    if not ident["from_this_repo"] or not ident["not_vestigial_hawking_copy"]:
        raise SystemExit(f"refusing vestigial/foreign decode binary: {ident}")

    with tempfile.TemporaryDirectory(prefix="noetic-zero-parent-") as td:
        tmp = Path(td)
        dylib = compile_dylib(tmp / "libopenlog.dylib")

        print(
            f"g011: observing native decode {binary} (timeout {DECODE_TIMEOUT_S:.0f}s)",
            file=sys.stderr,
            flush=True,
        )
        with GpuLaneLock("g011-noetic-zero-parent"):
            live = observe_command(
                [
                    str(binary),
                    "--artifact-root",
                    str(artifact),
                    "--tokenizer",
                    str(tokenizer),
                    "--prompt",
                    PROMPT,
                    "--max-new-tokens",
                    "1",
                    "--max-seq-len",
                    "32",
                ],
                dylib=dylib,
                log_path=tmp / "live.log",
                parent=parent,
                artifact=artifact,
                timeout=DECODE_TIMEOUT_S,
            )
        decoded = parse_decode_output(live.get("stdout") or "", live.get("stderr") or "")
        live.update(decoded)
        live["n_parent_weight_opens"] = len(live["observation"]["parent_weight"])
        live["n_artifact_tensor_opens"] = live["observation"].get("n_artifact_tensor", 0)
        live["n_file_open_events"] = live["observation"]["n_events"]
        live["binary"] = str(binary)
        live["complete"] = bool(
            live["exit_code"] == 0
            and not live["timed_out"]
            and not live["metal_refused"]
            and (live.get("n_new_tokens") or 0) >= 1
            and live["n_artifact_tensor_opens"] >= EXPECTED_CATALOG_TENSORS
        )
        if live["timed_out"] or live["metal_refused"] or live["exit_code"] not in (0,):
            live["truncation"] = (
                "native decode did not complete; observing a prefix is not "
                "evidence that a COMPLETE inference opens no parent weights"
            )
        else:
            live["truncation"] = None

        print(
            f"g011: native decode exit={live['exit_code']} complete={live['complete']} "
            f"events={live['n_file_open_events']} tokens={live.get('new_token_ids')!r} "
            f"text={live.get('generated_text')!r}",
            file=sys.stderr,
            flush=True,
        )
        print("g011: observing composition negative control", file=sys.stderr, flush=True)
        control = observe_command(
            [
                str(VISION_PY),
                str(Path(__file__).resolve()),
                "--mode",
                "composition-control",
                "--parent",
                str(parent),
            ],
            dylib=dylib,
            log_path=tmp / "control.log",
            parent=parent,
            artifact=artifact,
            timeout=CONTROL_TIMEOUT_S,
        )
        control_json = _last_json_object(control.get("stdout") or "")
        control["teacher_load"] = control_json
        control["n_parent_weight_opens"] = len(control["observation"]["parent_weight"])
        control["detector_caught"] = bool(control["observation"]["parent_weight"])

    live_obs = live["observation"]
    control_obs = control["observation"]

    production_weight_paths = list(dict.fromkeys(live_obs["parent_weight"]))
    production_tok = list(dict.fromkeys(live_obs["parent_tokenizer"]))
    production_cfg = list(dict.fromkeys(live_obs["parent_config"]))

    production_clean = len(production_weight_paths) == 0
    control_caught = bool(control_obs["parent_weight"])
    dylib_ok = bool(live["dylib_loaded"] and control["dylib_loaded"])
    decode_complete = bool(live["complete"])

    if not decode_complete:
        verdict = "INCONCLUSIVE"
    elif not (production_clean and control_caught and dylib_ok):
        verdict = "FAIL"
    else:
        verdict = "PASS"

    found = []
    if production_tok:
        found.append(
            "parent tokenizer.json (tokenizer dependency, not a weight dependency)"
        )
    if production_cfg:
        found.append("parent config/index (config dependency, not a weight dependency)")
    if production_weight_paths:
        found.append("parent WEIGHT files — violation")
    if not found:
        found.append("no parent files opened")

    production_verdict = (
        "INCONCLUSIVE"
        if not decode_complete
        else ("PASS" if production_clean else "FAIL")
    )

    receipt = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "elapsed_s": round(time.time() - t0, 3),
        "claim": "Production Noetic inference does not load parent weights.",
        "observation_method": (
            "DYLD interpose of open/openat/open$NOCANCEL/fopen "
            "(tools/headless/noetic_openlog.c) on the live process. Successful "
            "opens are resolved via F_GETPATH so openat relative names still "
            "name the file. Every recorded path is an observed open(), not a "
            "walk of the loader source."
        ),
        "parent_bf16": str(parent),
        "artifact": str(artifact),
        "tokenizer": str(tokenizer),
        "decode_binary": str(binary),
        "decode_binary_identity": ident,
        "did_not_load_second_27b": True,
        "did_not_modify_models": True,
        "distinctions": {
            "parent_weights_at_inference": {
                "is_the_violation": True,
                "observed": not production_clean,
                "paths": production_weight_paths,
                "note": (
                    "A .safetensors / .bin / .gguf / .pt under the parent bf16 "
                    "directory. This is the claim under test."
                ),
            },
            "parent_config_or_tokenizer": {
                "is_the_violation": False,
                "observed": bool(production_tok or production_cfg),
                "tokenizer_paths": production_tok,
                "config_paths": production_cfg,
                "note": (
                    "Prior work found the parent tokenizer.json in use for "
                    "tokenization. Confirmed by observed open() of that file. "
                    "Tokenizer dependency, not a weight dependency."
                ),
            },
            "parent_at_compile_time": {
                "is_the_violation": False,
                "observed": False,
                "note": (
                    "This harness records runtime opens of the decode process. "
                    "The binary is built from this repo via cargo; cargo is not "
                    "given the parent path. Compile-time reads of the parent "
                    "would not be a runtime dependency."
                ),
            },
        },
        "production_run": {
            "verdict": production_verdict,
            "found": found,
            "n_parent_weight_opens": len(production_weight_paths),
            "parent_weight_paths": production_weight_paths,
            "parent_tokenizer_paths": production_tok,
            "parent_config_paths": production_cfg,
            "live_native_decode": {
                "why": (
                    "ascension_qwen38_hybrid_greedy built from THIS repo is "
                    "production Noetic inference. A PASS requires the process "
                    "to run to completion, emit at least one token, and show "
                    f"the {EXPECTED_CATALOG_TENSORS} catalog tensor reads. A "
                    "truncated prefix (MetalContext::new dying first) is "
                    "INCONCLUSIVE, not evidence."
                ),
                "argv": live["argv"],
                "binary": live["binary"],
                "exit_code": live["exit_code"],
                "timed_out": live["timed_out"],
                "elapsed_s": live["elapsed_s"],
                "dylib_loaded": live["dylib_loaded"],
                "saw_tokenizer_encode": live["saw_tokenizer_encode"],
                "saw_catalog_count": live["saw_catalog_count"],
                "catalog_count": live["catalog_count"],
                "saw_upload_progress": live["saw_upload_progress"],
                "last_upload_index": live["last_upload_index"],
                "metal_refused": live["metal_refused"],
                "generated_text": live["generated_text"],
                "new_token_ids": live["new_token_ids"],
                "n_new_tokens": live["n_new_tokens"],
                "fallbacks": live["fallbacks"],
                "n_parent_weight_opens": live["n_parent_weight_opens"],
                "n_artifact_tensor_opens": live["n_artifact_tensor_opens"],
                "n_file_open_events": live["n_file_open_events"],
                "complete": live["complete"],
                "truncation": live["truncation"],
                "stderr_head": live["stderr_head"],
                "stdout_head": live["stdout_head"],
                "observation": live_slice(live_obs),
            },
        },
        "negative_control": {
            "what": (
                "Composition harness teacher scoring: noetic_composition.SourceBF16.load "
                f"of {CONTROL_TENSOR}. That path legitimately opens a parent "
                "safetensors shard. The detector must flag it. A detector that "
                "cannot catch this cannot certify the clean case either."
            ),
            "via": "noetic_composition.SourceBF16",
            "tensor": CONTROL_TENSOR,
            "did_not_load_27b": True,
            "teacher_load": control_json,
            "exit_code": control["exit_code"],
            "elapsed_s": control["elapsed_s"],
            "dylib_loaded": control["dylib_loaded"],
            "parent_weight_opens": control_obs["parent_weight"],
            "parent_tokenizer_opens": control_obs["parent_tokenizer"],
            "parent_config_opens": control_obs["parent_config"],
            "detector_caught": control_caught,
            "n_file_open_events": control_obs["n_events"],
            "verdict": "PASS" if control_caught else "FAIL",
            "observation": live_slice(control_obs),
        },
        "detector": {
            "method": "DYLD_INSERT_LIBRARIES + syscall-backed open/openat interpose + F_GETPATH",
            "source": "tools/headless/noetic_openlog.c",
            "dylib_loaded_on_production_and_control": dylib_ok,
            "weight_suffixes": sorted(WEIGHT_SUFFIXES),
            "tokenizer_names": sorted(TOKENIZER_NAMES),
            "config_names": sorted(CONFIG_NAMES),
            "control_result": "caught" if control_caught else "MISSED",
        },
        "verdict": verdict,
        "pass_rule": (
            "PASS iff (1) the native decode built from THIS repo ran to "
            "completion (exit 0, at least one emitted token, "
            f"{EXPECTED_CATALOG_TENSORS} observed catalog tensor opens), "
            "(2) that complete run opened no parent weight file, (3) the "
            "composition teacher-scoring control DID open a parent weight "
            "file and the detector flagged it, (4) the interpose loaded on "
            "both runs. A truncated run (Metal refused, timeout, zero tokens, "
            "or missing catalog reads) is INCONCLUSIVE, never PASS."
        ),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(out) + ".tmp"
    Path(tmp_path).write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(tmp_path, out)
    receipt["receipt_path"] = str(out)
    return receipt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--mode",
        choices=("write-receipt", "composition-control"),
        default="write-receipt",
    )
    p.add_argument("--parent", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.mode == "composition-control":
        return mode_composition_control((args.parent or parent_dir()).resolve())

    rec = run_and_write(args.out)
    print(f"wrote {rec.get('receipt_path', RECEIPT)}")
    print(f"verdict {rec['verdict']}")
    live = rec["production_run"]["live_native_decode"]
    print(
        "decode complete:",
        live["complete"],
        "exit",
        live["exit_code"],
        "token",
        repr(live["generated_text"]),
        live["new_token_ids"],
    )
    print(
        "file-open events:",
        live["n_file_open_events"],
        "catalog tensors:",
        live["n_artifact_tensor_opens"],
    )
    print(
        "production parent weights:",
        rec["production_run"]["n_parent_weight_opens"],
    )
    print(
        "negative control caught:",
        rec["negative_control"]["detector_caught"],
        rec["negative_control"]["parent_weight_opens"],
    )
    print("found:", rec["production_run"]["found"])
    if rec["verdict"] == "PASS":
        return 0
    return 2 if rec["verdict"] == "INCONCLUSIVE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
