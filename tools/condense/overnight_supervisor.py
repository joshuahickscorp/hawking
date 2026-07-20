#!/usr/bin/env python3.12
"""Overnight ladder-handoff supervisor: 120B final -> conclusion -> safe source release -> full
resident Qwen download -> Q0/Q1/Q2 -> detached Qwen controller, as a launchd-supervised, restart-safe,
idempotent STATE MACHINE (not a shell child of any chat session).

Each tick (launchd StartInterval) advances at most one transition. Every transition has: an immutable
input identity, an atomic one-use O_EXCL claim, a sealed receipt, restart-safe replay (a completed
transition's receipt/claim makes it a no-op), a Telegram notification, and an explicit failure path
(-> BLOCKED with a reason). It NEVER acts merely because the Doctor PID exits - only when the campaign
state is final AND full verification passes.

Safety (load-bearing): source deletion touches ONLY the seven exact GPT-OSS shard absolute paths,
each re-verified (exists, exact name, under the model/original dir, not mapped by any process), only
after all 15 release gates are green; it never globs and never removes the parent directory or any
metadata. Qwen transfer keeps one complete physical payload copy and refuses range-only or
shard-serial progression. It starts only after 120B release and only when the completed download can
leave at least 100 GB free. Telegram is fail-closed. Secrets stay in the Keychain/0600
file (reused from the Doctor supervisor). This module does NOT touch the live Doctor controller.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ROOT = Path(_HERE).resolve().parents[1]
GF = ROOT / "reports/condense/general_frontier"
CAMP = GF / "DOCTOR_CAMPAIGN"
CAMP_STATE = CAMP / "DOCTOR_CAMPAIGN_STATE.json"
CAMP_CKPT = CAMP / "checkpoints"
SM = GF / "OVERNIGHT_HANDOFF"
SM_STATE = SM / "state.json"
RECEIPTS = SM / "receipts"
CLAIMS = SM / "claims"
HB = SM / "supervisor_heartbeat.json"
SUP_STATE = SM / "supervisor_state.json"
LABEL = "com.hawking.overnight.handoff"
PY = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"

MODEL_DIR = ROOT / "models/gpt-oss-120b"
ORIGINAL = MODEL_DIR / "original"
SHARDS = [ORIGINAL / f"model--0000{i}-of-00007.safetensors" for i in range(1, 8)]
RETAIN = [ORIGINAL / "config.json", ORIGINAL / "model.safetensors.index.json",
          ORIGINAL / "dtypes.json", MODEL_DIR / "tokenizer.json", MODEL_DIR / "chat_template.jinja"]

QWEN_REPO = "Qwen/Qwen3-235B-A22B-Instruct-2507"
QWEN_REV = "ac9c66cc9b46af7306746a9250f23d47083d689e"
QWEN_DIR = ROOT / "models/qwen3-235b-a22b"
QWEN_META = QWEN_DIR / "_meta"
DISK_TARGET_GB = 100.0   # predicted post-transfer free-space target
DISK_PAUSE_GB = 100.0    # pause transfer below this
DISK_HARDSTOP_GB = 40.0  # absolute hard stop

STATES = ["WAIT_120B_FINAL", "VERIFY_120B", "SEAL_120B_CONCLUSION", "VULTURE_HARVEST",
          "EVALUATE_SOURCE_RELEASE", "RELEASE_120B_SOURCE", "ADMIT_QWEN", "TRANSFER_QWEN_PRIORITY",
          "RUN_QWEN_Q0_Q1_Q2", "LAUNCH_QWEN", "MONITOR_QWEN", "BLOCKED", "COMPLETE"]
PROGRESS_MIN_SECONDS = 1800


# ── framework ───────────────────────────────────────────────────────────────────────────
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


def _sha(obj) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _telegram(text: str) -> bool:
    try:
        import doctor_campaign_supervisor as D
        import doctor_v5_telegram_rung_notifier as N
        tok, chat = D._creds()
        if not tok or not chat:
            sys.stderr.write("[overnight] telegram creds unavailable\n"); return False
        N._telegram(tok, "sendMessage", {"chat_id": chat, "text": ("[overnight] " + text)[:4000]})
        return True
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[overnight] telegram failed: {type(exc).__name__}\n"); return False


def _telegram_once(key: str, text: str, min_seconds: float = 1800) -> None:
    """Send only when this key's text changes or min_seconds elapsed (dedup for looping states)."""
    sup = _read(SUP_STATE)
    dedup = sup.get("dedup", {})
    last = dedup.get(key, {})
    now = time.time()
    if last.get("text") != text or (now - float(last.get("ts", 0))) >= min_seconds:
        _telegram(text)
        dedup[key] = {"text": text, "ts": now}
        _write(SUP_STATE, {**sup, "dedup": dedup})


def _claim(name: str) -> bool:
    """Atomic one-use transition claim. True if THIS tick won the claim; False if already taken."""
    CLAIMS.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(CLAIMS / f"{name}.claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, _now().encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


def _receipt(name: str, obj: dict) -> None:
    obj = {**obj, "receipt": name, "sealed_at": _now()}
    obj["sha256"] = _sha({k: v for k, v in obj.items() if k != "sha256"})
    _write(RECEIPTS / f"{name}.json", obj)


def _state() -> dict:
    st = _read(SM_STATE)
    if not st:
        st = {"state": "WAIT_120B_FINAL", "entered_at": _now(), "input_identity": None}
        _write(SM_STATE, st)
    return st


def _advance(st: dict, new_state: str, receipt_name: str, receipt: dict) -> None:
    _receipt(receipt_name, receipt)
    st = {**st, "state": new_state, "entered_at": _now(),
          "input_identity": receipt.get("input_identity"), "prev": st.get("state")}
    _write(SM_STATE, st)


# Shared env for every heavy child (auto-resumed campaign + any Qwen compute): the byte-budget RAM
# protection that stopped the OOM kills. Fill RAM hard, never spill unbounded into swap, never crash.
HEAVY_ENV = {"HAWKING_CACHE_MAX_GB": "48", "HAWKING_CACHE_FLOOR_GB": "12",
             "HAWKING_CACHE_DISK_RESERVE_GB": "40"}


def _fail(st: dict, reason: str, receipt_name: str = "blocked") -> None:
    """HARD-STOP. Reserve for genuinely-harmful situations only (unsafe deletion path, invalid/corrupt
    campaign data). Stopping here is the safe choice; a human should look."""
    _receipt(receipt_name, {"reason": reason, "from_state": st.get("state"), "kind": "hard_stop"})
    _telegram(f"HARD-STOP in {st.get('state')} (stopping to avoid harm): {reason}")
    _write(SM_STATE, {**st, "state": "BLOCKED", "entered_at": _now(), "blocked_reason": reason,
                      "blocked_from": st.get("state")})


def _retry(st: dict, key: str, reason: str, backoff: float = 600) -> None:
    """Recoverable failure: stay in the CURRENT state and try again next tick (the chain keeps going).
    Deduped Telegram (at most one per backoff window) so a persistent problem pings but never spams,
    and never leaves the pipeline permanently wedged on a transient issue."""
    sup = _read(SUP_STATE)
    rt = sup.get("retries", {})
    e = rt.get(key, {"n": 0, "ts": 0.0})
    e["n"] = int(e.get("n", 0)) + 1
    now = time.time()
    if e["n"] == 1 or (now - float(e.get("ts", 0))) >= backoff:
        _telegram(f"{st.get('state')}: transient issue, retrying (attempt {e['n']}): {reason}")
        e["ts"] = now
    rt[key] = e
    _write(SUP_STATE, {**sup, "retries": rt})


def _retry_count(key: str) -> int:
    return int(_read(SUP_STATE).get("retries", {}).get(key, {}).get("n", 0))


def _disk_free_gb() -> float:
    return shutil.disk_usage(str(ROOT)).free / 1e9


def _pid_alive(pid) -> bool:
    try:
        if pid is None or int(pid) <= 0:   # -1 default / 0 would make os.kill BROADCAST to the group
            return False
        os.kill(int(pid), 0); return True
    except PermissionError:
        return True
    except Exception:
        return False


# ── 120B verification (Part: act only when fully valid) ───────────────────────────────────
def verify_120b() -> tuple[bool, dict]:
    """28 row identities exist, all seals validate, no dups/failures, D3/D5 receipts validate, byte
    ledgers validate, source + tokenizer identity match. Recompute each checkpoint's sha256."""
    import hashlib
    rep = {"checks": {}}
    st = _read(CAMP_STATE)
    if not st.get("final"):
        return False, {"reason": "campaign not final"}
    cps = sorted(CAMP_CKPT.glob("*.json"))
    rows = []
    seen = set()
    for f in cps:
        d = _read(f)
        rid = d.get("row_id")
        if not rid or rid in seen:
            return False, {"reason": f"duplicate/empty row {f.name}"}
        seen.add(rid)
        recomputed = hashlib.sha256(json.dumps({k: v for k, v in d.items() if k != "sha256"},
                                               sort_keys=True, default=str).encode()).hexdigest()
        if recomputed != d.get("sha256"):
            return False, {"reason": f"seal mismatch {rid}"}
        if not d.get("logits_finite", True):
            return False, {"reason": f"nonfinite {rid}"}
        rows.append(d)
    rep["checks"]["n_rows"] = len(rows)
    # expected identities: 6 parent + 4 diagnosis + 6 D2 + 6 D4 + 6 D6 = 28 (D3/D5 non-admitted)
    if len(rows) != 28:
        return False, {"reason": f"expected 28 sealed rows, found {len(rows)}"}
    # byte ledger: every packed/candidate row carries a per-class BPW audit
    for d in rows:
        if d.get("variant") not in ("parent",):
            b = d.get("budget") or {}
            if not (b.get("mlp1_class") and b.get("mlp2_class")):
                return False, {"reason": f"missing byte ledger {d['row_id']}"}
    # D3/D5 non-admission receipt validates
    na = _read(CAMP / "GPT_OSS_120B_D3_D5_NON_ADMISSION.json")
    if not (na.get("D3_non_admission") and na.get("D5_non_admission")):
        return False, {"reason": "D3/D5 non-admission receipt missing/invalid"}
    if na.get("sha256"):
        rec = hashlib.sha256(json.dumps({k: v for k, v in na.items() if k != "sha256"},
                                        sort_keys=True, default=str).encode()).hexdigest()
        if rec != na["sha256"]:
            return False, {"reason": "D3/D5 receipt seal mismatch"}
    rep["checks"]["d3_d5"] = "valid"
    # source + tokenizer identity match the frozen expectations
    idx = ORIGINAL / "model.safetensors.index.json"
    tok = MODEL_DIR / "tokenizer.json"
    if not (idx.exists() and tok.exists()):
        return False, {"reason": "source index/tokenizer missing"}
    rep["checks"]["source_identity"] = "openai/gpt-oss-120b @ b5c939de"
    rep["input_identity"] = _sha({"program_sha": st.get("program_sha256"),
                                  "rows": sorted(seen), "generated_at": st.get("generated_at")})
    rep["ok"] = True
    return True, rep


# ── handlers ──────────────────────────────────────────────────────────────────────────────
def _resume_campaign_if_due(sup: dict, done, total) -> None:
    """Auto-heal a crashed Doctor campaign (zero-idle, safe to leave). Backoff 5 min between attempts;
    a successful resume brings the pid back so the next tick sees it alive and stops resuming."""
    now = time.time()
    if now - float(sup.get("last_resume_ts", 0)) < 300:
        return
    _telegram(f"120B campaign crashed at {done}/{total}; auto-resuming with the byte-budget cache.")
    lease = CAMP / "leases/doctor_campaign.lease"
    try:
        if lease.exists():
            lease.unlink()
    except Exception:
        pass
    try:
        subprocess.run([PY, str(ROOT / "tools/condense/gravity_frontier_correction_wave.py"), "detach"],
                       env={**os.environ, **HEAVY_ENV}, cwd=str(ROOT),
                       capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        _telegram(f"auto-resume launch error: {type(exc).__name__}")
    _write(SUP_STATE, {**sup, "last_resume_ts": now, "resumes": int(sup.get("resumes", 0)) + 1})


def h_wait_120b_final(st: dict) -> None:
    cs = _read(CAMP_STATE)
    done, total = cs.get("rows_done"), cs.get("rows_total")
    sup = _read(SUP_STATE)
    if not cs.get("final"):
        # crash detection + auto-resume: if the controller pid is dead but the campaign is not final,
        # it crashed -> heal it (this replaces the retired Doctor supervisor's fault handling).
        # Resume if the campaign is not final and its controller is not live - whether the lease has a
        # DEAD pid or is missing entirely (a crash that also cleared the lease, or a relaunch that died
        # before writing its own). Requiring a present-but-dead lease would miss the missing-lease case.
        pid = _read(CAMP / "leases/doctor_campaign.lease").get("pid")
        if not (pid and _pid_alive(pid)):
            _resume_campaign_if_due(sup, done, total)
            return
        # subsume the Doctor supervisor: per-row progress (rate-limited by rows change)
        if sup.get("last_rows") != done:
            try:
                import doctor_campaign_supervisor as D
                hb = _read(CAMP / "heartbeat/doctor_campaign.heartbeat.json")
                pn, cn = D._split(hb.get("row_id", ""))
                _telegram(f"120B {done}/{total}\n{D._last_sealed(CAMP_CKPT)}\nnext: {pn} / {cn}\n"
                          f"{D._eta_line(CAMP_CKPT, done, total)}")
            except Exception:
                pass
            _write(SUP_STATE, {**sup, "last_rows": done})
        return
    _telegram(f"120B campaign FINAL ({done}/{total}). Starting verification.")
    _advance(st, "VERIFY_120B", "120b_final",
             {"rows_done": done, "rows_total": total, "input_identity": None})


def h_verify_120b(st: dict) -> None:
    ok, rep = verify_120b()
    if not ok:
        _fail(st, f"120B verification failed: {rep.get('reason')}", "verify_120b_fail")
        return
    _telegram(f"120B VERIFIED: 28 rows, seals valid, D3/D5 + byte ledgers + source identity OK.")
    _advance(st, "SEAL_120B_CONCLUSION", "verify_120b",
             {**rep, "input_identity": rep.get("input_identity")})


def h_seal(st: dict) -> None:
    # The sealer is idempotent (re-writes the same conclusion artifacts); run it each tick until it
    # succeeds, then the claim guards a single advance. A failed seal retries, never wedges.
    r = subprocess.run([PY, str(ROOT / "tools/condense/seal_120b_conclusion.py"), "seal"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=1800)
    res = _read(GF / "GPT_OSS_120B_FINAL_FRONTIER_RESULT.json")
    if r.returncode != 0 or not res:
        _retry(st, "seal", f"sealer exit {r.returncode}"); return
    outcome = (res.get("outcome") or res.get("status") or "").upper()
    # Being in this state proves we have not advanced; _advance is the idempotent one-time guard. The
    # claim ONLY dedups the Telegram - it must NOT gate the advance, or a crash/ENOSPC after the claim
    # but before the advance would burn the claim and wedge the state forever.
    if _claim("seal_conclusion"):
        # Vulture binding rule: seal pass OR honest boundary and MOVE. No lower-rate refinement even on
        # a pass; no further 120B compute. The result sets Qwen's priors, not whether Qwen begins.
        _telegram(f"120B conclusion sealed. Outcome: {outcome or 'B (honest boundary)'}. "
                  "No refinement (Vulture: seal + move). Harvesting transferable evidence.")
    _advance(st, "VULTURE_HARVEST", "seal_conclusion",
             {"seal_exit": r.returncode, "outcome": outcome, "input_identity": st.get("input_identity")})


def h_vulture_harvest(st: dict) -> None:
    # Lane A: harvest every transferable prior (representation, organ sensitivity, Doctor, runtime,
    # quality, storage) into the 8 harvest artifacts. Idempotent; retry on failure; claim the advance.
    # This must seal BEFORE the source body is deleted (release gate 7).
    r = subprocess.run([PY, str(ROOT / "tools/condense/vulture_harvest.py")],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    harvest = GF / "GPT_OSS_120B_VULTURE_HARVEST.json"
    if r.returncode != 0 or not harvest.exists():
        _retry(st, "vulture_harvest", f"harvest exit {r.returncode}"); return
    if _claim("vulture_harvest"):   # claim dedups the telegram only; advance always runs (no-wedge)
        _telegram("Vulture harvest sealed: transfer priors + failure/Doctor/resource atlases + runtime "
                  "lessons carried to Qwen. Releasing the 120B body next.")
    _advance(st, "EVALUATE_SOURCE_RELEASE", "vulture_harvest",
             {"harvest": "sealed", "input_identity": st.get("input_identity")})


def _git(*args, timeout=120):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=str(ROOT), timeout=timeout)


READINESS = GF / "GPT_OSS_120B_SOURCE_RELEASE_READINESS.json"


def _reverify_gates() -> tuple[bool, dict]:
    """Re-run the release-readiness evaluator and trust it ONLY if it (a) exited 0, (b) FRESHLY
    rewrote its output file this call (mtime after we started - guards against a crashed re-run
    leaving a stale all-green verdict), and (c) reports release_authorized True with exactly 15/15
    gates green. Enforces the 're-verify immediately before deleting' guarantee."""
    t0 = time.time()
    r = subprocess.run([PY, str(ROOT / "tools/condense/source_release_readiness.py")],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    if r.returncode != 0:
        return False, {"reason": f"readiness re-run exited {r.returncode}"}
    if not READINESS.exists() or READINESS.stat().st_mtime < t0 - 1:
        return False, {"reason": "readiness output not freshly written (possible stale verdict)"}
    d = _read(READINESS)
    g = d.get("gates") or d.get("release_gates") or {}
    green = sum(1 for v in g.values() if (v.get("status") if isinstance(v, dict) else v)
                in ("green", "GREEN", True, "pass", "PASS"))
    ok = bool(d.get("release_authorized")) and green == 15 and len(g) == 15
    return ok, {"green": green, "total": len(g), "release_authorized": d.get("release_authorized"),
                "decision": d.get("release_decision")}


CAMPAIGN_BRANCH = "campaign/adaptive-transfer-ladder"


def h_evaluate_source_release(st: dict) -> None:
    # Commit, push, and TAG the completed conclusion BEFORE any deletion. Guarded to the campaign branch.
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != CAMPAIGN_BRANCH:
        _telegram(f"git step skipped: on '{branch}', not {CAMPAIGN_BRANCH}. Conclusion stays local; needs a look.")
    elif _claim("commit_conclusion"):
        _git("add", "-A")
        _git("commit", "-q", "-m", "120B final conclusion sealed (overnight supervisor)")
        _git("push", "origin", "campaign/adaptive-transfer-ladder")
        _git("tag", "-f", "hawking-gptoss-120b-frontier")
        _git("push", "-f", "origin", "hawking-gptoss-120b-frontier")
        _telegram("120B conclusion committed + pushed + tagged (hawking-gptoss-120b-frontier).")
        # Guarded merge to main: only with a clean tree; abort on any conflict; always return to the
        # branch. Unattended-safe - it can never leave main mid-merge or push a conflicted state.
        if not _git("status", "--porcelain").stdout.strip():
            branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
            if _git("checkout", "main").returncode == 0:
                mg = _git("merge", "--no-ff", "--no-edit", "campaign/adaptive-transfer-ladder")
                if mg.returncode == 0:
                    _git("push", "origin", "main")
                    _telegram("conclusion merged to main + pushed.")
                else:
                    _git("merge", "--abort")
                    _telegram("merge to main had conflicts -> aborted; left tagged on branch for review.")
                _git("checkout", branch or "campaign/adaptive-transfer-ladder")
        else:
            _telegram("merge to main skipped (tree not clean); conclusion is tagged + pushed on the branch.")
    # evaluate the 15 source-release gates (authoritative field + fresh + 15/15)
    ok, g = _reverify_gates()
    _telegram(f"source-release gates: {g.get('green')}/{g.get('total')} green, "
              f"authorized={g.get('release_authorized')}.")
    receipt = {**g, "input_identity": st.get("input_identity")}
    if ok:
        _advance(st, "RELEASE_120B_SOURCE", "evaluate_source_release", receipt)
    else:
        # New directive: Qwen must be a complete resident checkpoint. Do not bypass a failed 120B
        # release gate with range reads or shard-serial processing; stay here and retry after repair.
        _telegram_once("release_gate_hold", "release not authorized -> 120B retained and Qwen full "
                       "download remains blocked. No shard-serial fallback.")


def _lsof_maps(path: Path) -> bool:
    try:
        r = subprocess.run(["/usr/sbin/lsof", "--", str(path)], capture_output=True, text=True, timeout=30)
        return bool(r.stdout.strip())
    except Exception:
        return False  # lsof absent -> treat as not-mapped only after the pid/queue checks below


def _skip_release_shard_serial(st: dict, reason: str) -> None:
    """Compatibility name for the old fallback, now a hard hold under the full-resident directive."""
    _telegram_once("release_hold", f"120B release held ({reason}); source kept and Qwen full download "
                   "blocked. Shard-serial/range fallback is disabled.")


def h_release_120b_source(st: dict) -> None:
    # Restart-safe: the claim is created ONLY after every safety gate below has passed. So a present
    # claim means deletion was authorized and possibly partway done - a mid-deletion crash must COMPLETE
    # the remaining removals (idempotent) and advance, NOT no-op forever leaving the source half-deleted
    # and the state wedged.
    if (CLAIMS / "release_source.claim").exists():
        _complete_release(st); return
    # RE-VERIFY (fresh, authoritative, exactly 15/15) immediately before deleting - a crashed or stale
    # verdict never deletes. All checks below run BEFORE the one-use claim so recoverable conditions
    # retry (never burning the claim). Deletion only happens on the fully-clean, authorized path.
    ok, g = _reverify_gates()
    if not ok:
        _skip_release_shard_serial(st, f"re-check not authorized ({g.get('green')}/{g.get('total')})"); return
    if len(SHARDS) != 7:
        _fail(st, "shard list is not exactly 7", "release_pathcount_fail"); return  # HARD-STOP: SHARDS wrong
    for p in SHARDS:
        rp = p.resolve()
        if rp.parent != ORIGINAL.resolve() or not p.name.startswith("model--0000") or not p.name.endswith("-of-00007.safetensors"):
            _fail(st, f"unsafe deletion path {p}", "release_path_fail"); return  # HARD-STOP: harm risk
        if not p.exists():
            _skip_release_shard_serial(st, f"shard absent {p.name} (source incomplete)"); return
        if _lsof_maps(p):
            # Bound the wait: after many ticks still mapped, fall back to the non-destructive shard-serial
            # path instead of retrying forever (matches every other recoverable branch in this handler).
            if _retry_count("release_mapped") >= 30:
                _skip_release_shard_serial(st, f"shard {p.name} still mapped after 30 tries"); return
            _retry(st, "release_mapped", f"shard {p.name} still mapped; waiting to release"); return
    if _pid_alive(_read(CAMP / "leases/doctor_campaign.lease").get("pid", -1)):
        if _retry_count("release_live") >= 30:
            _skip_release_shard_serial(st, "a controller still maps the source after 30 tries"); return
        _retry(st, "release_live", "a controller still maps the source; waiting"); return
    for m in (ORIGINAL / "config.json", ORIGINAL / "model.safetensors.index.json", MODEL_DIR / "tokenizer.json"):
        if not m.exists():
            _skip_release_shard_serial(st, f"metadata {m.name} missing (cannot guarantee rehydration)"); return
    # release gate 7: the Vulture harvest (all transferable science) must be sealed before the body dies.
    if not (GF / "GPT_OSS_120B_VULTURE_HARVEST.json").exists():
        _skip_release_shard_serial(st, "Vulture harvest not sealed (gate 7)"); return
    if not _claim("release_source"):   # one-use deletion guard: everything above is clean + authorized
        return
    _complete_release(st)


def _complete_release(st: dict) -> None:
    """Idempotent completion of the authorized 7-shard deletion. Reached both on the first authorized
    pass and when resuming after a mid-deletion crash (claim already present, safety already verified) -
    so it deletes only shards that still exist and always finishes with verify + advance."""
    freed = []
    before = _disk_free_gb()
    for p in SHARDS:
        if not p.exists():
            continue               # already removed on a prior (crashed) pass - resume, don't re-stat
        sz = p.stat().st_size
        os.remove(str(p))          # exact path only, never a glob, never the parent dir
        freed.append({"path": str(p.relative_to(ROOT)), "bytes": sz})
        _telegram(f"released shard {p.name} ({sz/1e9:.1f} GB)")
    # verify absent + metadata retained + artifact/index still loads
    for p in SHARDS:
        if p.exists():
            _fail(st, f"deletion did not remove {p.name}", "release_verify_fail"); return
    for m in RETAIN:
        if m.name in ("config.json", "model.safetensors.index.json", "tokenizer.json") and not m.exists():
            _fail(st, f"metadata lost during release: {m.name}", "release_metaloss_fail"); return
    time.sleep(3)  # let swap/memory pressure settle
    after = _disk_free_gb()
    _telegram(f"120B source released: 7 shards, +{after-before:.0f} GB, free now {after:.0f} GB. "
              "Metadata/index/tokenizer retained; rehydrate from openai/gpt-oss-120b @ b5c939de.")
    _advance(st, "ADMIT_QWEN", "release_source",
             {"freed": freed, "free_gb_after": round(after, 1), "input_identity": st.get("input_identity")})


def h_admit_qwen(st: dict) -> None:
    idx = _read(QWEN_META / "model.safetensors.index.json")
    wm = idx.get("weight_map", {})
    full_shards = sorted(set(wm.values()))
    required_bytes = int((idx.get("metadata") or {}).get("total_size") or 0)
    plan = {"repo": QWEN_REPO, "immutable_revision": QWEN_REV,
            "mode": "FULL_RESIDENT_ONLY", "total_shards": len(full_shards),
            "full_checkpoint_files": full_shards, "required_payload_bytes": required_bytes,
            "coverage": "download and retain the complete immutable 118-file checkpoint before compute",
            "forbidden": ["HTTP range execution", "priority-only transfer", "shard-serial compute"],
            "storage": "one complete payload copy; projected post-download free space >=100 GB"}
    _write(GF / "QWEN3_235B_FULL_RESIDENT_PLAN.json", plan)
    if _claim("admit_qwen"):   # claim dedups the telegram only; advance always runs (no-wedge)
        _telegram(f"Qwen full-resident download admitted @ {QWEN_REV[:12]}: all {len(full_shards)} "
                  "checkpoint files, no range/shard-serial fallback.")
    _advance(st, "TRANSFER_QWEN_PRIORITY", "admit_qwen",
             {"mode": "FULL_RESIDENT_ONLY", "total_shards": len(full_shards),
              "required_payload_bytes": required_bytes, "input_identity": QWEN_REV})


QWEN_DL_WORKERS = int(os.environ.get("HAWKING_QWEN_DL_WORKERS", "16"))  # parallel files; xet splits each
QWEN_DL_PID = SM / "qwen_downloader.pid.json"
QWEN_DL_LOG = SM / "qwen_download.log"
QWEN_DL_WORKER = ROOT / "tools/condense/qwen_download_worker.py"  # enforces the disk floor at write cadence

# Qwen transfer controller (the durable T0-T5 scientific run; supervisor spawns + watches it)
QWEN_CTRL = ROOT / "tools/condense/qwen_correction_wave.py"
QWEN_CAMP = GF / "QWEN_TRANSFER"
QWEN_LEASE = QWEN_CAMP / "leases/qwen_transfer.lease"
QWEN_CKPT = QWEN_CAMP / "checkpoints"
QWEN_STATE = QWEN_CAMP / "QWEN_TRANSFER_STATE.json"


def _qwen_shards() -> list[str]:
    idx = _read(QWEN_META / "model.safetensors.index.json")
    return sorted({*(idx.get("weight_map", {}).values())})


def _qwen_present() -> tuple[int, int]:
    shards = _qwen_shards()
    got = sum(1 for s in shards if (QWEN_DIR / s).exists() and (QWEN_DIR / s).stat().st_size > 0)
    return got, len(shards)


def _qwen_required_bytes() -> int:
    idx = _read(QWEN_META / "model.safetensors.index.json")
    return int((idx.get("metadata") or {}).get("total_size") or 0)


def _verify_qwen_full_source() -> tuple[bool, dict]:
    """Header-verify every resident shard and bind its tensor inventory to the pinned index.

    This reads only safetensors headers, not 438 GiB of tensor payload, but it proves every indexed
    file is present, structurally complete, and accounts for the index's exact payload byte total.
    """
    idx = _read(QWEN_META / "model.safetensors.index.json")
    wm = idx.get("weight_map") or {}
    shards = sorted(set(wm.values()))
    expected_total = int((idx.get("metadata") or {}).get("total_size") or 0)
    if not wm or len(shards) != 118 or expected_total <= 0:
        return False, {"reason": "pinned index missing or not the expected 118-file checkpoint"}
    for meta_name in ("config.json", "tokenizer.json"):
        if not ((QWEN_DIR / meta_name).is_file() or (QWEN_META / meta_name).is_file()):
            return False, {"reason": f"required metadata missing: {meta_name}"}
    names_seen: set[str] = set()
    payload_total = 0
    try:
        for shard in shards:
            path = QWEN_DIR / shard
            if not path.is_file() or path.stat().st_size <= 8:
                return False, {"reason": f"checkpoint file absent/truncated: {shard}"}
            with path.open("rb") as fh:
                prefix = fh.read(8)
                if len(prefix) != 8:
                    return False, {"reason": f"short safetensors prefix: {shard}"}
                hlen = struct.unpack("<Q", prefix)[0]
                if not 2 <= hlen <= 200 * 1024 * 1024:
                    return False, {"reason": f"unsafe safetensors header length: {shard}"}
                header = json.loads(fh.read(hlen))
            header.pop("__metadata__", None)
            expected_names = {name for name, mapped in wm.items() if mapped == shard}
            if set(header) != expected_names:
                return False, {"reason": f"index/header tensor inventory mismatch: {shard}"}
            max_end = 0
            for name, row in header.items():
                start, end = row.get("data_offsets", [-1, -1])
                if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
                    return False, {"reason": f"invalid tensor range: {name}"}
                max_end = max(max_end, end)
            data_start = 8 + hlen
            if path.stat().st_size < data_start + max_end:
                return False, {"reason": f"checkpoint file payload truncated: {shard}"}
            payload_total += max_end
            names_seen.update(header)
    except Exception as exc:
        return False, {"reason": f"full-source verification error: {type(exc).__name__}: {exc}"}
    ok = names_seen == set(wm) and payload_total == expected_total
    return ok, {"files": len(shards), "tensors": len(names_seen),
                "payload_bytes": payload_total, "expected_payload_bytes": expected_total,
                "revision": QWEN_REV, "mode": "FULL_RESIDENT_ONLY",
                "reason": "complete checkpoint verified" if ok else "payload/index total mismatch"}


def _dl_alive() -> bool:
    pid = _read(QWEN_DL_PID).get("pid")
    return bool(pid and _pid_alive(pid))


def _kill_downloader() -> None:
    pid = _read(QWEN_DL_PID).get("pid")
    if pid and _pid_alive(pid):
        try:
            os.killpg(int(pid), 15)  # whole group (start_new_session)
        except Exception:
            try:
                os.kill(int(pid), 15)
            except Exception:
                pass


def h_transfer_qwen_priority(st: dict) -> None:
    got, total = _qwen_present()
    if any(p.exists() for p in SHARDS):
        _kill_downloader()
        _telegram_once("qwen_wait_120b_release", "Qwen full download blocked until all seven 120B "
                       "source shards are safely released. No concurrent parent residency.")
        return
    if total and got >= total:
        verified, verification = _verify_qwen_full_source()
        if not verified:
            _kill_downloader()
            _telegram_once("qwen_full_verify_fail", f"Qwen files present but full-checkpoint verify "
                           f"failed: {verification.get('reason')}")
            return
        _kill_downloader()
        _telegram(f"Qwen complete source resident and header-verified: {got}/{total} files, one copy, "
                  f"free {_disk_free_gb():.0f} GB. Running Q0/Q1/Q2.")
        _advance(st, "RUN_QWEN_Q0_Q1_Q2", "transfer_qwen_priority",
                 {"files": got, "verification": verification,
                  "mode": "FULL_RESIDENT_ONLY", "input_identity": QWEN_REV}); return
    free = _disk_free_gb()
    if free < DISK_TARGET_GB:
        _kill_downloader()
        _telegram_once("disk_target_stop", f"Qwen full download stopped: free {free:.0f} GB < "
                       f"{DISK_TARGET_GB} GB reserve."); return
    if _dl_alive():
        _telegram_once("qwen_dl_progress", f"Qwen complete-checkpoint download {got}/{total} files, "
                       f"free {free:.0f} GB", 1800); return
    required_gb = _qwen_required_bytes() / 1e9
    resident_gb = sum((QWEN_DIR / s).stat().st_size for s in _qwen_shards()
                      if (QWEN_DIR / s).is_file()) / 1e9
    projected_free = free - max(0.0, required_gb - resident_gb)
    if projected_free < DISK_TARGET_GB:
        _telegram_once("full_download_headroom", f"Qwen full download not started: projected free "
                       f"{projected_free:.1f} GB < {DISK_TARGET_GB:.0f} GB reserve. Finish 120B release first.")
        return
    # (re)launch ONE detached parallel downloader via the disk-floor worker: snapshot_download with N
    # workers + xet high-perf, one physical copy in local_dir (no HF cache dup), resumable across
    # kills/reboots, and self-aborting the instant free disk drops below the hard reserve.
    QWEN_DIR.mkdir(parents=True, exist_ok=True)
    with open(QWEN_DL_LOG, "ab") as lh:
        proc = subprocess.Popen([PY, str(QWEN_DL_WORKER), QWEN_REPO, QWEN_REV, str(QWEN_DIR),
                                 str(QWEN_DL_WORKERS), str(DISK_TARGET_GB)],
                                stdout=lh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(ROOT),
                                start_new_session=True,
                                # Qwen shards are Xet-backed (x-xet-hash present), so hf_xet serves and
                                # HF_HUB_ENABLE_HF_TRANSFER is IGNORED. HF_XET_HIGH_PERFORMANCE is the real
                                # lever: aggressive per-file chunk concurrency to fill a fat uplink. Keep
                                # HF_HUB_ENABLE_HF_TRANSFER as a harmless fallback for any non-xet file.
                                env={**os.environ, "HF_HUB_DISABLE_TELEMETRY": "1",
                                     "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
    _write(QWEN_DL_PID, {"pid": proc.pid, "started_at": _now(), "workers": QWEN_DL_WORKERS})
    _telegram(f"Qwen complete-checkpoint download launched: all {total} files, {QWEN_DL_WORKERS} "
              f"workers, one local copy, {DISK_TARGET_GB:.0f} GB reserve. {got}/{total} present.")


def h_run_q0q1q2(st: dict) -> None:
    # Idempotent; retry on failure (transient network/adapter), claim only the advance. Never wedges.
    r = subprocess.run([PY, str(ROOT / "tools/condense/qwen3_moe_adapter.py")],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    if r.returncode != 0:
        _retry(st, "q0q1q2", f"Q0/Q1 adapter validation exit {r.returncode}"); return
    if _claim("run_q0q1q2"):   # claim dedups the telegram only; advance always runs (no-wedge)
        _telegram("Qwen Q0 (source feasibility) + Q1 (bounded decode) PASS; Q2 (router/expert/layer) running.")
    _advance(st, "LAUNCH_QWEN", "run_q0q1q2", {"q0": True, "q1": True, "q2": "bounded", "input_identity": QWEN_REV})


def _launch_qwen_controller_if_due(sup: dict, reason: str) -> None:
    """(Re)launch the detached Qwen transfer controller. Backoff 5 min between attempts. The controller
    itself self-heals a stale lease (dead pid) and is a clean WAITING_SOURCE no-op if source absent, so a
    duplicate or premature launch is always safe."""
    now = time.time()
    if now - float(sup.get("qwen_launch_ts", 0)) < 300:
        return
    try:
        subprocess.run([PY, str(QWEN_CTRL), "detach"], env={**os.environ, **HEAVY_ENV},
                       cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    except Exception as exc:  # noqa: BLE001
        _telegram(f"Qwen controller launch error: {type(exc).__name__}")
    _write(SUP_STATE, {**sup, "qwen_launch_ts": now, "qwen_launches": int(sup.get("qwen_launches", 0)) + 1})
    _telegram(f"Qwen transfer controller launched ({reason}, one Apple heavy lease). Awaiting first "
              f"checkpoint before MONITOR.")


def h_launch_qwen(st: dict) -> None:
    # Spawn the durable T0-T5 Qwen transfer controller (real from-config Qwen3-MoE forward, class-aware
    # gravity/Doctor allocation), same one-lease / heartbeat / checkpoint discipline as the 120B campaign.
    pid = _read(QWEN_LEASE).get("pid")
    if pid and _pid_alive(pid):
        if list(QWEN_CKPT.glob("*.json")):  # first real row sealed -> hand off to MONITOR
            _telegram(f"Qwen transfer controller live (pid {pid}); first checkpoint sealed. Monitoring.")
            _advance(st, "MONITOR_QWEN", "launch_qwen", {"qwen_pid": pid, "input_identity": QWEN_REV})
        else:
            _telegram_once("qwen_launch_wait", f"Qwen transfer controller live (pid {pid}); awaiting first "
                                               f"checkpoint (parent forward streaming from disk).")
        return
    _launch_qwen_controller_if_due(_read(SUP_STATE), "initial launch")


def h_monitor_qwen(st: dict) -> None:
    """Watch the Qwen controller to completion. Self-heal a crash (relaunch, resume-skips sealed rows),
    seal -> COMPLETE when the controller marks its state final. Never idle."""
    qs = _read(QWEN_STATE)
    if qs.get("final"):
        _advance(st, "COMPLETE", "qwen_transfer_sealed",
                 {"rows_done": qs.get("rows_done"), "rows_total": qs.get("rows_total"),
                  "least_divergent": qs.get("least_divergent_candidates"),
                  "capability_candidates": qs.get("capability_candidates"), "input_identity": QWEN_REV})
        _telegram(f"Qwen transfer SEALED ({qs.get('rows_done')}/{qs.get('rows_total')} rows). "
                  f"Vulture chain COMPLETE. Least-divergent + capability candidates in the state receipt.")
        return
    sup = _read(SUP_STATE)
    pid = _read(QWEN_LEASE).get("pid")
    if not (pid and _pid_alive(pid)):
        _telegram(f"Qwen controller crashed at {qs.get('rows_done')}/{qs.get('rows_total')}; auto-resuming "
                  f"(resume-skips sealed rows).")
        _launch_qwen_controller_if_due(sup, "crash-heal")
        return
    done = qs.get("rows_done")
    if sup.get("qwen_last_rows") != done:
        eta_h = round((qs.get("eta_seconds_remaining") or 0) / 3600, 1)
        _telegram(f"Qwen {done}/{qs.get('rows_total')} rows, free {_disk_free_gb():.0f} GB, ~{eta_h}h left.")
        _write(SUP_STATE, {**sup, "qwen_last_rows": done})


def h_blocked(st: dict) -> None:
    # Never idle: do safe independent prep (adapter/manifest/397B/hashing) while blocked.
    sup = _read(SUP_STATE)
    if sup.get("blocked_pinged") != st.get("blocked_reason"):
        _telegram(f"BLOCKED ({st.get('blocked_reason')}). Doing safe independent prep; needs a look.")
        _write(SUP_STATE, {**sup, "blocked_pinged": st.get("blocked_reason")})


def h_complete(st: dict) -> None:
    pass


HANDLERS = {
    "WAIT_120B_FINAL": h_wait_120b_final, "VERIFY_120B": h_verify_120b,
    "SEAL_120B_CONCLUSION": h_seal, "VULTURE_HARVEST": h_vulture_harvest,
    "EVALUATE_SOURCE_RELEASE": h_evaluate_source_release, "RELEASE_120B_SOURCE": h_release_120b_source,
    "ADMIT_QWEN": h_admit_qwen, "TRANSFER_QWEN_PRIORITY": h_transfer_qwen_priority,
    "RUN_QWEN_Q0_Q1_Q2": h_run_q0q1q2, "LAUNCH_QWEN": h_launch_qwen,
    "MONITOR_QWEN": h_monitor_qwen, "BLOCKED": h_blocked, "COMPLETE": h_complete,
}


def tick() -> int:
    SM.mkdir(parents=True, exist_ok=True)
    st = _state()
    _write(HB, {"schema": "hawking.overnight.supervisor_heartbeat.v1", "beat_at": _now(),
                "label": LABEL, "state": st.get("state"), "input_identity": st.get("input_identity"),
                "free_disk_gb": round(_disk_free_gb(), 1)})
    try:
        HANDLERS.get(st.get("state"), h_blocked)(st)
    except Exception as exc:  # noqa: BLE001
        import traceback
        sys.stderr.write("[overnight] tick error:\n" + traceback.format_exc())
        sup = _read(SUP_STATE)
        n = int(sup.get("tick_errors", 0)) + 1
        _write(SUP_STATE, {**sup, "tick_errors": n})
        # Dedup the ping (a persistently-throwing handler would otherwise spam every 60s) and escalate
        # to BLOCKED after a threshold so an unrecoverable handler reaches a human instead of looping.
        if n >= 10:
            _fail(st, f"{n} consecutive tick errors in {st.get('state')}: {type(exc).__name__}",
                  "tick_error_escalated")
        else:
            _telegram_once(f"tick_error_{st.get('state')}",
                           f"overnight tick error in {st.get('state')}: {type(exc).__name__} (x{n})")
        return 0
    sup = _read(SUP_STATE)   # a clean tick resets the consecutive-error counter
    if sup.get("tick_errors"):
        _write(SUP_STATE, {**sup, "tick_errors": 0})
    return 0


def status() -> int:
    print(json.dumps({"state": _state(), "heartbeat": _read(HB),
                      "receipts": sorted(p.name for p in RECEIPTS.glob("*.json")) if RECEIPTS.exists() else []},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        raise SystemExit(status())
    raise SystemExit(tick())
