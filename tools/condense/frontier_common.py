#!/usr/bin/env python3.12
"""Shared primitives for frontier receipt tooling."""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import pathlib
import re
import subprocess
from typing import Any, Callable

SIGN_ALG = "sha256-json-v1"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def git_commit(root: pathlib.Path) -> str:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label)


def read_json(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        return json.load(open(path))
    except Exception:
        return None


def write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def canonical_digest(data: dict[str, Any]) -> str:
    unsigned = copy.deepcopy(data)
    unsigned.pop("signature", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def placeholder(value: Any, *, case_insensitive_todo: bool = False) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    todo_text = text.upper() if case_insensitive_todo else text
    return not text or "<" in text or "TODO" in todo_text or "..." in text


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.match(value))


def commands(record: dict[str, Any], *extra_keys: str) -> list[str]:
    out = []
    cmds = record.get("commands")
    if isinstance(cmds, list):
        out.extend(str(cmd) for cmd in cmds if cmd)
    for key in ("command", *extra_keys):
        if record.get(key):
            out.append(str(record[key]))
    return out


def signature_status(record: dict[str, Any], *, sign_alg: str = SIGN_ALG) -> dict[str, Any]:
    sig = record.get("signature") if isinstance(record.get("signature"), dict) else {}
    expected = canonical_digest(record)
    ok = sig.get("algorithm") == sign_alg and sig.get("digest") == expected
    problems = []
    if sig.get("algorithm") != sign_alg:
        problems.append(f"signature algorithm must be {sign_alg}")
    if sig.get("digest") != expected:
        problems.append("signature digest mismatch")
    return {
        "ok": ok,
        "algorithm": sig.get("algorithm"),
        "digest": sig.get("digest"),
        "expected_digest": expected,
        "problems": problems,
    }


def sign_record(
    record: dict[str, Any],
    status_fn: Callable[..., dict[str, Any]],
    *,
    root: pathlib.Path,
    status_kwargs: dict[str, Any] | None = None,
    allow_blocked_draft: bool = False,
    sign_alg: str = SIGN_ALG,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signed = copy.deepcopy(record)
    signed.pop("signature", None)
    signed.setdefault("generated_at", now_utc())
    signed.setdefault("git_commit", git_commit(root))
    signed["signed_at"] = now_utc()
    kwargs = status_kwargs or {}
    status = status_fn(signed, require_signature=False, **kwargs)
    if not status["ok"] and not allow_blocked_draft:
        return signed, status
    signed["signature"] = {"algorithm": sign_alg, "digest": canonical_digest(signed)}
    return signed, status_fn(signed, require_signature=True, **kwargs)
