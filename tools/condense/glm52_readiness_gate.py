#!/usr/bin/env python3.12
"""GLM-5.2 readiness gate: a small, restart-safe tick that waits for Telegram to be
configured, then seals the expected campaign contract and bootstraps the controller
and worker configs.

Scope is deliberately narrow. Everything past this point (the live Xet autotune
execution-authority protocol, the oracle pilot, freezing the candidate program, and
wiring real dispatch) requires an actively-driven, reviewed session -- not a blind
tick -- so this gate stops cleanly at DONE_FOR_NOW rather than fabricating further
automated progress.

Each tick is idempotent: re-running a completed phase is a no-op check, never a
redo. A failed step just retries next tick; nothing here deletes or overwrites
sealed campaign evidence.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv/glm52/bin/python")
STATE_DIR = Path(
    "/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity/readiness_gate"
)
STATE_FILE = STATE_DIR / "state.json"
LOG = STATE_DIR / "gate.log"
CREATED_AT_FILE = STATE_DIR / "created_at.txt"

CONTRACT_PATH = ROOT / "GLM52_EXPECTED_CAMPAIGN_CONTRACT.json"
CONTROLLER_CONFIG_PATH = ROOT / "GLM52_CONTROLLER_CONFIG.json"
WORKER_CONFIG_PATH = ROOT / "GLM52_WORKER_CONFIG.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _log(msg: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(f"{_now()} {msg}\n")


def _run(args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)


def _pinned_created_at() -> str:
    # campaign_contract.py build requires a deterministic created_at: pick it once
    # and reuse forever so retries stay idempotent instead of resealing each tick.
    if CREATED_AT_FILE.exists():
        return CREATED_AT_FILE.read_text().strip()
    value = _now()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CREATED_AT_FILE.write_text(value)
    return value


def h_wait_telegram(st: dict) -> None:
    r = _run([PY, "tools/condense/glm52_telegram.py", "status"])
    if r.returncode != 0:
        _log(f"telegram status check failed: {r.stderr.strip()[:300]}")
        return
    try:
        reported = json.loads(r.stdout)
    except Exception:
        _log("telegram status returned non-JSON output")
        return
    if not reported.get("ready"):
        return  # quiet no-op, nothing has changed yet
    digest = reported.get("chat_identity_digest")
    if not digest:
        _log("telegram reports ready but no chat_identity_digest present yet; waiting")
        return
    _log("telegram configured; advancing to BUILD_CONTRACT")
    _write(STATE_FILE, {**st, "phase": "BUILD_CONTRACT", "chat_identity_digest": digest,
                        "entered_at": _now()})


def h_build_contract(st: dict) -> None:
    if CONTRACT_PATH.exists():
        _log("campaign contract already present; advancing to BOOTSTRAP_CONFIGS")
        _write(STATE_FILE, {**st, "phase": "BOOTSTRAP_CONFIGS", "entered_at": _now()})
        return
    digest = st.get("chat_identity_digest")
    if not digest:
        _log("no chat_identity_digest recorded; reverting to WAIT_TELEGRAM")
        _write(STATE_FILE, {**st, "phase": "WAIT_TELEGRAM", "entered_at": _now()})
        return
    created_at = _pinned_created_at()
    r = _run([PY, "tools/condense/glm52_campaign_contract.py", "build",
              "--root", str(ROOT),
              "--chat-identity-digest", digest,
              "--created-at", created_at])
    if r.returncode != 0 or not CONTRACT_PATH.exists():
        _log(f"contract build failed (exit {r.returncode}): {r.stderr.strip()[:500]}")
        return  # retry next tick
    _log("campaign contract sealed")
    _write(STATE_FILE, {**st, "phase": "BOOTSTRAP_CONFIGS", "entered_at": _now()})


def h_bootstrap_configs(st: dict) -> None:
    if CONTROLLER_CONFIG_PATH.exists() and WORKER_CONFIG_PATH.exists():
        _log("controller + worker configs already present; DONE_FOR_NOW")
        _write(STATE_FILE, {**st, "phase": "DONE_FOR_NOW", "entered_at": _now()})
        return
    r = _run([PY, "tools/condense/glm52_worker.py", "bootstrap-configs"])
    if r.returncode != 0 or not (CONTROLLER_CONFIG_PATH.exists() and WORKER_CONFIG_PATH.exists()):
        _log(f"bootstrap-configs failed (exit {r.returncode}): {r.stderr.strip()[:500]}")
        return  # retry next tick
    _log("controller + worker configs bootstrapped")
    _write(STATE_FILE, {**st, "phase": "DONE_FOR_NOW", "entered_at": _now()})


def h_done(st: dict) -> None:
    pass  # terminal: mechanical prerequisites satisfied; live autotune onward needs a reviewed session


HANDLERS = {
    "WAIT_TELEGRAM": h_wait_telegram,
    "BUILD_CONTRACT": h_build_contract,
    "BOOTSTRAP_CONFIGS": h_bootstrap_configs,
    "DONE_FOR_NOW": h_done,
}


def tick() -> int:
    st = _read(STATE_FILE, {"phase": "WAIT_TELEGRAM", "entered_at": _now()})
    try:
        HANDLERS.get(st.get("phase"), h_wait_telegram)(st)
    except Exception as exc:  # noqa: BLE001
        import traceback
        sys.stderr.write("[glm52-readiness-gate] tick error:\n" + traceback.format_exc())
        _log(f"tick error in {st.get('phase')}: {type(exc).__name__}: {exc}")
    return 0


def status() -> int:
    print(json.dumps(_read(STATE_FILE, {"phase": "WAIT_TELEGRAM"}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        raise SystemExit(status())
    raise SystemExit(tick())
