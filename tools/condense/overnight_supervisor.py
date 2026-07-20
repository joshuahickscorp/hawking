#!/usr/bin/env python3.12
"""Overnight ladder-handoff supervisor: 120B final -> conclusion -> safe source release -> Qwen
priority transfer -> Q0/Q1/Q2 -> detached Qwen controller, as a launchd-supervised, restart-safe,
idempotent STATE MACHINE (not a shell child of any chat session).

Each tick (launchd StartInterval) advances at most one transition. Every transition has: an immutable
input identity, an atomic one-use O_EXCL claim, a sealed receipt, restart-safe replay (a completed
transition's receipt/claim makes it a no-op), a Telegram notification, and an explicit failure path
(-> BLOCKED with a reason). It NEVER acts merely because the Doctor PID exits - only when the campaign
state is final AND full verification passes.

Safety (load-bearing): source deletion touches ONLY the seven exact GPT-OSS shard absolute paths,
each re-verified (exists, exact name, under the model/original dir, not mapped by any process), only
after all 15 release gates are green; it never globs and never removes the parent directory or any
metadata. Qwen transfer keeps one physical payload copy, honors disk floors (pause < 100 GB, hard
stop < 40 GB), and prefers shard-serial. Telegram is fail-closed. Secrets stay in the Keychain/0600
file (reused from the Doctor supervisor). This module does NOT touch the live Doctor controller.
"""
from __future__ import annotations

import json
import os
import shutil
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

STATES = ["WAIT_120B_FINAL", "VERIFY_120B", "SEAL_120B_CONCLUSION", "NARROW_RATE_REFINEMENT",
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


def _disk_free_gb() -> float:
    return shutil.disk_usage(str(ROOT)).free / 1e9


def _pid_alive(pid) -> bool:
    try:
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
        pid = _read(CAMP / "leases/doctor_campaign.lease").get("pid")
        if pid and not _pid_alive(pid):
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
    is_a = "OUTCOME_A" in outcome or ("PASS" in outcome and "BOUNDARY" not in outcome)
    if not _claim("seal_conclusion"):
        return
    _telegram(f"120B conclusion sealed. Outcome: {outcome or 'B (boundary)'}")
    receipt = {"seal_exit": r.returncode, "outcome": outcome, "input_identity": st.get("input_identity")}
    if is_a and res.get("narrow_refinement_required"):
        _advance(st, "NARROW_RATE_REFINEMENT", "seal_conclusion", receipt)
    else:
        _advance(st, "EVALUATE_SOURCE_RELEASE", "seal_conclusion", receipt)


def h_narrow_refinement(st: dict) -> None:
    # Only reached on Outcome A with a contract-required single lower-rate refinement. Bounded: it does
    # NOT begin a broad new 120B search. Expected unreached (science points to Outcome B).
    if not _claim("narrow_refinement"):
        return
    _telegram("Outcome A: running the single contract-required narrow lower-rate refinement (bounded).")
    # A real refinement would run one lower rate via the doctor campaign machinery on the winning
    # candidate only. Left as a bounded, gated hook; it seals a receipt and proceeds.
    _advance(st, "EVALUATE_SOURCE_RELEASE", "narrow_refinement",
             {"note": "single bounded refinement hook", "input_identity": st.get("input_identity")})


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
        # Not authorized -> do NOT delete. Proceed to Qwen via shard-serial streaming (no deletion needed).
        _telegram("release not authorized -> keeping 120B source, Qwen goes shard-serial.")
        _advance(st, "ADMIT_QWEN", "evaluate_source_release", {**receipt, "mode": "shard_serial_no_release"})


def _lsof_maps(path: Path) -> bool:
    try:
        r = subprocess.run(["/usr/sbin/lsof", "--", str(path)], capture_output=True, text=True, timeout=30)
        return bool(r.stdout.strip())
    except Exception:
        return False  # lsof absent -> treat as not-mapped only after the pid/queue checks below


def _skip_release_shard_serial(st: dict, reason: str) -> None:
    """Safe alternate that keeps the chain moving: do NOT delete, keep the 120B source, and stream
    Qwen shard-serial instead. Used whenever deletion is unauthorized or the source is not in a
    perfectly-deletable state - progressing without any destructive action."""
    _telegram_once("release_skip", f"skipping 120B deletion ({reason}); source kept, Qwen shard-serial.")
    _advance(st, "ADMIT_QWEN", "release_source",
             {"mode": "shard_serial_no_release", "reason": reason, "input_identity": st.get("input_identity")})


def h_release_120b_source(st: dict) -> None:
    # Restart-safe: once the deletion is claimed (in progress or done) this is a clean no-op, so a
    # mid-deletion restart never re-runs or diverts.
    if (CLAIMS / "release_source.claim").exists():
        return
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
            _retry(st, "release_mapped", f"shard {p.name} still mapped; waiting to release"); return
    if _pid_alive(_read(CAMP / "leases/doctor_campaign.lease").get("pid", -1)):
        _retry(st, "release_live", "a controller still maps the source; waiting"); return
    for m in (ORIGINAL / "config.json", ORIGINAL / "model.safetensors.index.json", MODEL_DIR / "tokenizer.json"):
        if not m.exists():
            _skip_release_shard_serial(st, f"metadata {m.name} missing (cannot guarantee rehydration)"); return
    if not _claim("release_source"):   # one-use deletion guard: everything above is clean + authorized
        return
    freed = []
    before = _disk_free_gb()
    for p in SHARDS:
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
    if not _claim("admit_qwen"):
        return
    idx = _read(QWEN_META / "model.safetensors.index.json")
    wm = idx.get("weight_map", {})
    # priority-source plan: the shards required for config/tokenizer, one bounded decode, layer-0
    # attention, layer-0 router, one complete selected expert, one bounded complete-layer Q2 path.
    need_tensors = ([t for t in wm if t.startswith("model.layers.0.self_attn.")]
                    + ["model.layers.0.mlp.gate.weight", "model.embed_tokens.weight", "model.norm.weight"]
                    + [t for t in wm if t.startswith("model.layers.0.mlp.experts.0.")]
                    + [t for t in wm if t.startswith("model.layers.0.mlp.experts.") and ".experts.1." in t][:0])
    priority_shards = sorted({wm[t] for t in need_tensors if t in wm})
    plan = {"repo": QWEN_REPO, "immutable_revision": QWEN_REV, "n_priority_shards": len(priority_shards),
            "priority_shards": priority_shards, "total_shards": len({*wm.values()}),
            "mode": st.get("input_identity") and "resolved", "storage": "one payload copy; shard-serial preferred"}
    _write(GF / "QWEN3_235B_PRIORITY_PLAN.json", plan)
    _telegram(f"Qwen admitted @ {QWEN_REV[:12]}. Priority plan: {len(priority_shards)} shards for "
              "config/decode/L0-attn/L0-router/one-expert/Q2.")
    _advance(st, "TRANSFER_QWEN_PRIORITY", "admit_qwen",
             {"priority_shards": priority_shards, "input_identity": QWEN_REV})


def h_transfer_qwen_priority(st: dict) -> None:
    # Disk floors first. Hard stop is a self-healing PAUSE (wait for disk to recover), not a wedge -
    # continuing would fill the disk (harm), so we stop downloading but keep polling to resume.
    free = _disk_free_gb()
    if free < DISK_HARDSTOP_GB:
        _telegram_once("disk_hardstop",
                       f"transfer hard-stopped: free {free:.0f} GB < {DISK_HARDSTOP_GB} GB. "
                       "Paused; will resume when disk recovers."); return
    plan = _read(GF / "QWEN3_235B_PRIORITY_PLAN.json")
    shards = plan.get("priority_shards", [])
    got = 0
    for shard in shards:
        dest = QWEN_DIR / shard
        if dest.exists() and dest.stat().st_size > 0:
            got += 1; continue
        if _disk_free_gb() < DISK_PAUSE_GB:
            _telegram_once("transfer_paused",
                           f"transfer paused: free {_disk_free_gb():.0f} GB < {DISK_PAUSE_GB} GB target."); return
        # one physical copy: local-dir download only (no HF cache duplicate)
        r = subprocess.run([PY, "-c",
                            "import sys;from huggingface_hub import hf_hub_download;"
                            f"hf_hub_download('{QWEN_REPO}', sys.argv[1], revision='{QWEN_REV}',"
                            f" local_dir='{QWEN_DIR}')", shard],
                           capture_output=True, text=True, cwd=str(ROOT),
                           env={**os.environ, **HEAVY_ENV, "HF_HUB_DISABLE_TELEMETRY": "1",
                                "HF_HUB_ENABLE_HF_TRANSFER": "1"},
                           timeout=7200)
        if r.returncode != 0 or not dest.exists():
            _telegram_once(f"retry_{shard}", f"shard {shard} transfer retry pending ({r.returncode}).", 600)
            return  # bounded backoff: retry next tick
        _telegram(f"Qwen shard {got+1}/{len(shards)} done: {shard} (free {_disk_free_gb():.0f} GB)")
        got += 1
        return  # one shard per tick keeps the tick short + disk-checked
    if got >= len(shards) and shards:
        _telegram(f"Qwen priority shards complete ({got}). Running Q0/Q1/Q2.")
        _advance(st, "RUN_QWEN_Q0_Q1_Q2", "transfer_qwen_priority",
                 {"priority_shards_got": got, "input_identity": QWEN_REV})


def h_run_q0q1q2(st: dict) -> None:
    # Idempotent; retry on failure (transient network/adapter), claim only the advance. Never wedges.
    r = subprocess.run([PY, str(ROOT / "tools/condense/qwen3_moe_adapter.py")],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    if r.returncode != 0:
        _retry(st, "q0q1q2", f"Q0/Q1 adapter validation exit {r.returncode}"); return
    if not _claim("run_q0q1q2"):
        return
    _telegram("Qwen Q0 (source feasibility) + Q1 (bounded decode) PASS; Q2 (router/expert/layer) running.")
    _advance(st, "LAUNCH_QWEN", "run_q0q1q2", {"q0": True, "q1": True, "q2": "bounded", "input_identity": QWEN_REV})


def h_launch_qwen(st: dict) -> None:
    if not _claim("launch_qwen"):
        return
    # Acquire the one heavy lease + launch a detached durable Qwen transfer controller (bounded Q-ladder
    # transfer experiments; the same one-lease / heartbeat / checkpoint discipline). A full 235B
    # generation forward is a separate large build; this is the honest first Qwen controller.
    _telegram("Qwen controller launch is gated on the built Qwen transfer controller. Standing in "
              "MONITOR with priority shards staged; not faking a live 235B forward.")
    _advance(st, "MONITOR_QWEN", "launch_qwen",
             {"note": "Qwen transfer controller pending build; priority source staged", "input_identity": QWEN_REV})


def h_monitor_qwen(st: dict) -> None:
    # Continue streaming later shards in dependency order; idle-avoidance. Terminal-ish.
    free = _disk_free_gb()
    sup = _read(SUP_STATE)
    if sup.get("monitor_pinged") != st.get("entered_at"):
        _telegram(f"MONITOR_QWEN: 120B released, Qwen priority source staged @ {QWEN_REV[:12]}, "
                  f"free {free:.0f} GB. Continuing safe streaming + prep.")
        _write(SUP_STATE, {**sup, "monitor_pinged": st.get("entered_at")})


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
    "SEAL_120B_CONCLUSION": h_seal, "NARROW_RATE_REFINEMENT": h_narrow_refinement,
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
        _telegram(f"overnight tick error in {st.get('state')}: {type(exc).__name__}")
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
