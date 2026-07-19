#!/usr/bin/env python3.12
"""Durable, supervised, one-use, restart-safe watcher for the 120B Doctor campaign.

Runs as a launchd StartInterval job (every 5 min): each tick is a fresh short-lived process, so it
is NOT a shell child of any interactive session - it survives terminal close, logout, and reboot,
and launchd restarts it if a tick crashes. Each tick:
  1. writes its OWN heartbeat (observable supervision),
  2. reads the Doctor campaign state + controller pid,
  3. RUNNING  -> sends a rate-limited Telegram progress update (on rows_done change or >=30 min),
  4. FAULT    -> controller pid dead AND state not final -> one Telegram fault alert (deduped),
  5. COMPLETE -> state final=true -> ONE-USE: under a lock/marker it runs seal_120b_conclusion.py,
                 sends the sealed-conclusion Telegram (Outcome A pass / Outcome B boundary), writes
                 the CONCLUSION_SEALED marker, and boots itself out of launchd. Idempotent: a later
                 tick that sees the marker does nothing.

Restart-safe: state is re-read every tick; the seal is marker-guarded so a restart never double-seals
and a replay never floods the chat. Telegram is fail-closed: a delivery error is logged, never raised
into the tick (the supervisor must not crash on a network blip). Secrets stay in the Keychain.

It does NOT touch the Doctor controller, packing, thresholds, prompts, source, or byte accounting, and
it does NOT merge/tag (that deliberate git step is left to an operator/session); it only seals the
science and notifies so nothing is lost if the interactive session dies.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ROOT = Path(_HERE).resolve().parents[1]
CAMP = ROOT / "reports/condense/general_frontier/DOCTOR_CAMPAIGN"
STATE = CAMP / "DOCTOR_CAMPAIGN_STATE.json"
HB = CAMP / "heartbeat/doctor_campaign.heartbeat.json"
LEASE = CAMP / "leases/doctor_campaign.lease"
SUP_DIR = CAMP / "supervisor"
SUP_HB = SUP_DIR / "supervisor_heartbeat.json"
SUP_STATE = SUP_DIR / "supervisor_state.json"
SEALED_MARKER = SUP_DIR / "CONCLUSION_SEALED.marker"
LABEL = "com.hawking.doctor.campaign.supervisor"
PY = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
SEALER = str(ROOT / "tools/condense/seal_120b_conclusion.py")
PROGRESS_MIN_SECONDS = 1800  # send at most one heartbeat progress per 30 min unless rows change


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, p)


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0); return True
    except PermissionError:
        return True
    except Exception:
        return False


def _telegram(text: str) -> bool:
    """Fail-closed send via the campaign notifier's Keychain creds. Never raises."""
    try:
        import doctor_v5_telegram_rung_notifier as N
        tok = N._keychain_get(N.TOKEN_SERVICE); chat = N._keychain_get(N.CHAT_SERVICE)
        if not tok or not chat:
            return False
        N._telegram(tok, "sendMessage", {"chat_id": chat, "text": text[:4000]})
        return True
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[supervisor] telegram send failed: {type(exc).__name__}\n")
        return False


def _eta_line(rows_dir: Path, done, total) -> str:
    """ETA from the ACTUAL sealed rows' real forward times (accounts for mop contention). Approximate:
    treated rows (packing) run slower than the early parent rows, so this tightens as they seal."""
    try:
        done = int(done); total = int(total)
        secs = []
        for f in rows_dir.glob("*.json"):
            try:
                secs.append(float(json.loads(f.read_text()).get("forward_seconds") or 0))
            except Exception:
                pass
        secs = [s for s in secs if s > 0]
        remaining = total - done
        if not secs or remaining <= 0:
            return "ETA: computing" if remaining > 0 else "ETA: done"
        mean = sum(secs) / len(secs)
        eta = mean * remaining
        finish = time.strftime("%H:%M", time.gmtime(time.time() + eta)) + "Z"
        h, m = int(eta // 3600), int((eta % 3600) // 60)
        span = (f"{h}h {m}m" if h else f"{m}m")
        return f"ETA ~{span} remaining (~{finish}, approx; avg {int(mean)}s/row)"
    except Exception:
        return "ETA: n/a"


def _candidate_summary(rows_dir: Path) -> str:
    try:
        rows = [json.loads(f.read_text()) for f in rows_dir.glob("*.json")]
    except Exception:
        return ""
    cand = [r for r in rows if r.get("divergence_vs_parent")]
    cand.sort(key=lambda r: r["divergence_vs_parent"]["mean_sym_kl"])
    if not cand:
        return ""
    b = cand[0]
    d = b["divergence_vs_parent"]
    return (f"best so far {b['row_id']}: symKL {d['mean_sym_kl']} agree "
            f"{d['next_token_argmax_agreement']} ({b.get('verdict')})")


def _bootout_self() -> None:
    try:
        uid = os.getuid()
        subprocess.run(["/bin/launchctl", "bootout", f"gui/{uid}/{LABEL}"],
                       capture_output=True, text=True, timeout=15)
    except Exception:
        pass


def tick() -> int:
    SUP_DIR.mkdir(parents=True, exist_ok=True)
    sup_state = _read(SUP_STATE)
    runs = int(sup_state.get("runs", 0)) + 1
    state = _read(STATE)
    hb = _read(HB)
    lease = _read(LEASE)
    pid = lease.get("pid") or hb.get("pid")
    alive = _pid_alive(pid) if pid else False
    final = bool(state.get("final"))
    done = state.get("rows_done"); total = state.get("rows_total")

    # supervisor heartbeat (observable)
    _write(SUP_HB, {"schema": "hawking.doctor_campaign.supervisor_heartbeat.v1", "beat_at": _now(),
                    "label": LABEL, "runs": runs, "campaign_pid": pid, "campaign_alive": alive,
                    "final": final, "rows_done": done, "rows_total": total,
                    "sealed": SEALED_MARKER.exists()})

    # already sealed -> one-use no-op (idempotent), stand down
    if SEALED_MARKER.exists():
        _write(SUP_STATE, {**sup_state, "runs": runs, "status": "done_marker_present"})
        _bootout_self()
        return 0

    # COMPLETE -> seal once under the marker lock
    if final:
        try:
            SEALED_MARKER.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(SEALED_MARKER), os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # atomic one-use lock
        except FileExistsError:
            return 0
        seal = subprocess.run([PY, SEALER, "seal"], capture_output=True, text=True, cwd=str(ROOT), timeout=1800)
        outcome = ""
        try:
            res = _read(ROOT / "reports/condense/general_frontier/GPT_OSS_120B_FINAL_FRONTIER_RESULT.json")
            outcome = res.get("outcome") or res.get("status") or ""
        except Exception:
            pass
        os.write(fd, json.dumps({"sealed_at": _now(), "seal_exit": seal.returncode, "outcome": outcome}).encode())
        os.close(fd)
        _telegram("Hawking 120B Doctor campaign COMPLETE.\n"
                  f"rows {done}/{total}. Conclusion sealed (seal exit {seal.returncode}).\n"
                  f"outcome: {outcome or 'see GPT_OSS_120B_FINAL_FRONTIER_RESULT.json'}\n"
                  f"{_candidate_summary(CAMP / 'checkpoints')}\n"
                  "Next: operator commits/merges/tags + evaluates the 15 source-release gates before any Qwen transfer.")
        _write(SUP_STATE, {**sup_state, "runs": runs, "status": "sealed", "seal_exit": seal.returncode})
        _bootout_self()
        return 0

    # FAULT -> controller dead but not final
    if not alive:
        if sup_state.get("fault_notified") != pid:
            _telegram("Hawking 120B Doctor campaign FAULT.\n"
                      f"controller pid {pid} is not alive but state is not final (rows {done}/{total}, "
                      f"row {hb.get('row_id')}).\nNeeds a resume: python3.12 "
                      "tools/condense/gravity_frontier_correction_wave.py detach")
            _write(SUP_STATE, {**sup_state, "runs": runs, "status": "fault", "fault_notified": pid})
        else:
            _write(SUP_STATE, {**sup_state, "runs": runs, "status": "fault_known"})
        return 0

    # RUNNING -> rate-limited progress
    last_done = sup_state.get("last_progress_rows")
    last_ts = float(sup_state.get("last_progress_epoch", 0))
    now = time.time()
    if last_done != done or (now - last_ts) >= PROGRESS_MIN_SECONDS:
        _telegram("Hawking 120B Doctor campaign running.\n"
                  f"rows {done}/{total}  row {hb.get('row_id')} {hb.get('phase')}  pid {pid} alive.\n"
                  f"{_eta_line(CAMP / 'checkpoints', done, total)}\n"
                  f"{_candidate_summary(CAMP / 'checkpoints')}")
        sup_state = {**sup_state, "last_progress_rows": done, "last_progress_epoch": now}
    _write(SUP_STATE, {**sup_state, "runs": runs, "status": "running"})
    return 0


if __name__ == "__main__":
    raise SystemExit(tick())
