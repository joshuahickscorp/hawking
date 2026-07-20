#!/usr/bin/env python3.12
"""Restart-safety + safe-source-release proofs for the overnight handoff supervisor.

Every test runs against a TEMP fixture tree: all of overnight_supervisor's module-level Path
constants (ROOT/GF/CAMP/SM/ORIGINAL/SHARDS/RETAIN/QWEN_DIR/...) are monkeypatched under a
per-test tmp_path, so the suite NEVER touches the real models/gpt-oss-120b weights, the live
Doctor campaign, git, the network, or launchd. Telegram is stubbed to an in-memory recorder,
subprocess.run to an in-memory fake, _lsof_maps / _pid_alive / _disk_free_gb / time.sleep to
harmless stubs. The only files these tests delete are fake shard files they created themselves.

Covers the eleven operator scenarios plus the two safety-critical deletion invariants: only the
seven exact shard paths are ever removed, and any unsafe shard path (outside the original dir or
with a wrong name) makes release refuse and delete nothing.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import overnight_supervisor as osup  # noqa: E402


# -- fixture wiring ------------------------------------------------------------------------
def _configure(mod, monkeypatch, root):
    """Point every module-level path constant at a temp tree and neutralize all side channels."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    GF = root / "reports/condense/general_frontier"
    CAMP = GF / "DOCTOR_CAMPAIGN"
    SM = GF / "OVERNIGHT_HANDOFF"
    MODEL_DIR = root / "models/gpt-oss-120b"
    ORIGINAL = MODEL_DIR / "original"
    QWEN_DIR = root / "models/qwen3-235b-a22b"
    vals = {
        "ROOT": root,
        "GF": GF,
        "CAMP": CAMP,
        "CAMP_STATE": CAMP / "DOCTOR_CAMPAIGN_STATE.json",
        "CAMP_CKPT": CAMP / "checkpoints",
        "SM": SM,
        "SM_STATE": SM / "state.json",
        "RECEIPTS": SM / "receipts",
        "CLAIMS": SM / "claims",
        "HB": SM / "supervisor_heartbeat.json",
        "SUP_STATE": SM / "supervisor_state.json",
        "READINESS": GF / "GPT_OSS_120B_SOURCE_RELEASE_READINESS.json",
        "MODEL_DIR": MODEL_DIR,
        "ORIGINAL": ORIGINAL,
        "SHARDS": [ORIGINAL / f"model--0000{i}-of-00007.safetensors" for i in range(1, 8)],
        "RETAIN": [ORIGINAL / "config.json", ORIGINAL / "model.safetensors.index.json",
                   ORIGINAL / "dtypes.json", MODEL_DIR / "tokenizer.json",
                   MODEL_DIR / "chat_template.jinja"],
        "QWEN_DIR": QWEN_DIR,
        "QWEN_META": QWEN_DIR / "_meta",
    }
    for k, v in vals.items():
        monkeypatch.setattr(mod, k, v)
    ctx = types.SimpleNamespace(**vals)
    ctx.tg = []
    monkeypatch.setattr(mod, "_telegram", lambda text: (ctx.tg.append(text) or True))
    # No real process probing / disk probing / sleeping.
    monkeypatch.setattr(mod, "_pid_alive", lambda pid: False)     # no live heavy controller
    monkeypatch.setattr(mod, "_lsof_maps", lambda p: False)       # nothing mapped
    monkeypatch.setattr(mod, "_disk_free_gb", lambda: 500.0)      # plenty of headroom
    monkeypatch.setattr(mod.time, "sleep", lambda *a, **k: None)
    ctx.run_rc = lambda argv: 0
    ctx.calls = []

    def fake_run(argv, *a, **k):
        ctx.calls.append([str(x) for x in argv])
        return types.SimpleNamespace(returncode=ctx.run_rc(argv), stdout="", stderr="")

    monkeypatch.setattr(mod, "subprocess", types.SimpleNamespace(run=fake_run))
    return ctx


def _seal(mod, obj):
    """Reproduce the module's per-record seal: sha256 over the record minus its own sha256 field."""
    base = {k: v for k, v in obj.items() if k != "sha256"}
    return {**base, "sha256": mod._sha(base)}


def _build_campaign(mod, ctx):
    """A fully valid FINAL 120B campaign: 28 sealed rows + D3/D5 receipt + source index/tokenizer."""
    ctx.ORIGINAL.mkdir(parents=True, exist_ok=True)
    ctx.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (ctx.ORIGINAL / "model.safetensors.index.json").write_text("{}")
    (ctx.MODEL_DIR / "tokenizer.json").write_text("{}")
    mod._write(ctx.CAMP_STATE, {"final": True, "program_sha256": "a" * 64,
                                "generated_at": "2026-07-19T00:00:00Z",
                                "rows_done": 28, "rows_total": 28})
    ctx.CAMP_CKPT.mkdir(parents=True, exist_ok=True)
    # 6 parent + 4 diag + 6 D2 + 6 D4 + 6 D6 = 28 (D3/D5 are non-admitted).
    groups = [("parent", 6, False), ("diag", 4, True),
              ("D2", 6, True), ("D4", 6, True), ("D6", 6, True)]
    for prefix, n, budgeted in groups:
        for i in range(n):
            rid = f"{prefix}_{i}"
            d = {"row_id": rid, "variant": prefix, "logits_finite": True}
            if budgeted:
                d["budget"] = {"mlp1_class": "q4_k", "mlp2_class": "q4_k"}
            mod._write(ctx.CAMP_CKPT / f"{rid}.json", _seal(mod, d))
    na = {"D3_non_admission": True, "D5_non_admission": True}
    mod._write(ctx.CAMP / "GPT_OSS_120B_D3_D5_NON_ADMISSION.json", _seal(mod, na))


def _write_gates(mod, ctx, all_green=True):
    gates = {f"g{i}": {"status": "green"} for i in range(15)}
    if not all_green:
        gates["g7"] = {"status": "red"}
    mod._write(ctx.GF / "GPT_OSS_120B_SOURCE_RELEASE_READINESS.json",
               {"gates": gates, "release_authorized": bool(all_green),
                "release_decision": "AUTHORIZED" if all_green else "DENIED"})


def _build_release(mod, ctx, monkeypatch, all_green=True):
    """Seven fake shard files + retained metadata + a decoy + dead lease + gates + RELEASE state.

    Returns the decoy path (a non-shard file inside ORIGINAL used to prove no globbing).
    """
    ctx.ORIGINAL.mkdir(parents=True, exist_ok=True)
    ctx.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for p in ctx.SHARDS:
        p.write_bytes(b"fake-shard-bytes")
    (ctx.ORIGINAL / "config.json").write_text("{}")
    (ctx.ORIGINAL / "model.safetensors.index.json").write_text("{}")
    (ctx.ORIGINAL / "dtypes.json").write_text("{}")
    (ctx.MODEL_DIR / "tokenizer.json").write_text("{}")
    (ctx.MODEL_DIR / "chat_template.jinja").write_text("x")
    decoy = ctx.ORIGINAL / "extra-of-00007.safetensors"  # NOT in SHARDS -> must survive
    decoy.write_bytes(b"decoy")
    mod._write(ctx.CAMP / "leases/doctor_campaign.lease", {"pid": 999999})
    _write_gates(mod, ctx, all_green)
    # These tests exercise deletion SAFETY (path guards, lsof, metadata), not the gate subprocess
    # mechanics; mock the fresh-reverify to the intended authorization so the guards are what is tested.
    monkeypatch.setattr(mod, "_reverify_gates",
                        lambda: (bool(all_green),
                                 {"green": 15 if all_green else 9, "total": 15,
                                  "release_authorized": bool(all_green)}))
    mod._write(ctx.SM_STATE, {"state": "RELEASE_120B_SOURCE", "entered_at": mod._now(),
                              "input_identity": "id"})
    return decoy


# -- 1. successful 120B completion verifies -------------------------------------------------
def test_01_successful_120b_completion_verifies(tmp_path, monkeypatch):
    mod = osup
    ctx = _configure(mod, monkeypatch, tmp_path)
    _build_campaign(mod, ctx)
    ok, rep = mod.verify_120b()
    assert ok is True, rep
    assert rep.get("ok") is True
    assert rep["checks"]["n_rows"] == 28
    assert rep["checks"]["d3_d5"] == "valid"
    iid = rep.get("input_identity")
    assert isinstance(iid, str) and len(iid) == 64


# -- 2. invalid seal fails ------------------------------------------------------------------
def test_02_invalid_seal_fails_verification(tmp_path, monkeypatch):
    mod = osup
    ctx = _configure(mod, monkeypatch, tmp_path)
    _build_campaign(mod, ctx)
    victim = ctx.CAMP_CKPT / "parent_0.json"
    d = json.loads(victim.read_text())
    d["sha256"] = "0" * 64  # corrupt the seal
    victim.write_text(json.dumps(d))
    ok, rep = mod.verify_120b()
    assert ok is False
    assert "mismatch" in rep.get("reason", "").lower() or "seal" in rep.get("reason", "").lower()


# -- 3. source file still mapped -> refuse, delete nothing ----------------------------------
def test_03_source_mapped_refuses_release(tmp_path, monkeypatch):
    mod = osup
    ctx = _configure(mod, monkeypatch, tmp_path)
    _build_release(mod, ctx, monkeypatch, all_green=True)
    orig = list(ctx.SHARDS)
    monkeypatch.setattr(mod, "_lsof_maps",
                        lambda p: p.name.endswith("00003-of-00007.safetensors"))
    mod.h_release_120b_source(mod._read(ctx.SM_STATE))
    st = mod._read(ctx.SM_STATE)
    assert st["state"] == "BLOCKED"
    assert "mapped" in (st.get("blocked_reason") or "")
    for p in orig:
        assert p.exists(), f"shard deleted despite being mapped: {p.name}"


# -- 4. deletion gate red -> refuse, delete nothing -----------------------------------------
def test_04_deletion_gate_red_refuses(tmp_path, monkeypatch):
    mod = osup
    ctx = _configure(mod, monkeypatch, tmp_path)
    _build_release(mod, ctx, monkeypatch, all_green=False)
    orig = list(ctx.SHARDS)
    mod.h_release_120b_source(mod._read(ctx.SM_STATE))
    assert mod._read(ctx.SM_STATE)["state"] == "BLOCKED"
    for p in orig:
        assert p.exists(), f"shard deleted with red gates: {p.name}"


# -- 5. interrupted deletion is restart-safe ------------------------------------------------
def test_05_interrupted_deletion_is_restart_safe(tmp_path, monkeypatch):
    mod = osup
    ctx = _configure(mod, monkeypatch, tmp_path)
    decoy = _build_release(mod, ctx, monkeypatch, all_green=True)
    orig = list(ctx.SHARDS)
    # Simulate a prior tick that claimed and deleted 3 of 7 before crashing.
    assert mod._claim("release_source") is True
    for p in orig[:3]:
        os.remove(str(p))
    # Restart: the held claim must make the handler a clean no-op (no double run, no crash).
    mod.h_release_120b_source(mod._read(ctx.SM_STATE))
    for p in orig[:3]:
        assert not p.exists()
    for p in orig[3:]:
        assert p.exists(), f"restart deleted remaining shard: {p.name}"
    for m in ("config.json", "model.safetensors.index.json"):
        assert (ctx.ORIGINAL / m).exists(), f"restart touched metadata {m}"
    assert (ctx.MODEL_DIR / "tokenizer.json").exists()
    assert decoy.exists(), "restart touched a non-shard file"
    # No advance, no block, no receipt from the no-op.
    assert mod._read(ctx.SM_STATE)["state"] == "RELEASE_120B_SOURCE"
    assert not (ctx.RECEIPTS / "release_source.json").exists()


# -- 6. interrupted Qwen download retries + respects floors ---------------------------------
def test_06_interrupted_qwen_download_retries_and_respects_floors(tmp_path, monkeypatch):
    mod = osup
    ctx = _configure(mod, monkeypatch, tmp_path)
    plan_shard = "model-00001-of-00002.safetensors"
    mod._write(ctx.GF / "QWEN3_235B_PRIORITY_PLAN.json", {"priority_shards": [plan_shard]})
    ctx.QWEN_DIR.mkdir(parents=True, exist_ok=True)

    def transfer_state():
        return {"state": "TRANSFER_QWEN_PRIORITY", "entered_at": mod._now(), "input_identity": None}

    # (a) shard missing, disk healthy -> retry next tick, no advance, no crash.
    mod._write(ctx.SM_STATE, transfer_state())
    monkeypatch.setattr(mod, "_disk_free_gb", lambda: 500.0)  # fake_run does NOT create the file
    mod.h_transfer_qwen_priority(mod._read(ctx.SM_STATE))
    assert mod._read(ctx.SM_STATE)["state"] == "TRANSFER_QWEN_PRIORITY"
    assert any("retry" in m for m in ctx.tg)

    # (b) below hard-stop -> BLOCKED.
    mod._write(ctx.SM_STATE, transfer_state())
    monkeypatch.setattr(mod, "_disk_free_gb", lambda: 30.0)
    mod.h_transfer_qwen_priority(mod._read(ctx.SM_STATE))
    assert mod._read(ctx.SM_STATE)["state"] == "BLOCKED"

    # (c) below pause (but above hard-stop) -> pause without advancing.
    mod._write(ctx.SM_STATE, transfer_state())
    ctx.tg.clear()
    monkeypatch.setattr(mod, "_disk_free_gb", lambda: 70.0)
    mod.h_transfer_qwen_priority(mod._read(ctx.SM_STATE))
    assert mod._read(ctx.SM_STATE)["state"] == "TRANSFER_QWEN_PRIORITY"
    assert any("paused" in m for m in ctx.tg)


# -- 7. disk floor reached: explicit hard-stop + pause --------------------------------------
def test_07_disk_floor_hardstop_and_pause(tmp_path, monkeypatch):
    mod = osup
    ctx = _configure(mod, monkeypatch, tmp_path)
    mod._write(ctx.GF / "QWEN3_235B_PRIORITY_PLAN.json", {"priority_shards": ["s.safetensors"]})
    ctx.QWEN_DIR.mkdir(parents=True, exist_ok=True)

    # hard-stop
    mod._write(ctx.SM_STATE, {"state": "TRANSFER_QWEN_PRIORITY", "entered_at": mod._now()})
    monkeypatch.setattr(mod, "_disk_free_gb", lambda: mod.DISK_HARDSTOP_GB - 1)
    mod.h_transfer_qwen_priority(mod._read(ctx.SM_STATE))
    st = mod._read(ctx.SM_STATE)
    assert st["state"] == "BLOCKED"
    assert "hard" in (st.get("blocked_reason") or "").lower()

    # pause (between hard-stop and pause floors)
    mod._write(ctx.SM_STATE, {"state": "TRANSFER_QWEN_PRIORITY", "entered_at": mod._now()})
    ctx.tg.clear()
    monkeypatch.setattr(mod, "_disk_free_gb",
                        lambda: (mod.DISK_HARDSTOP_GB + mod.DISK_PAUSE_GB) / 2)
    mod.h_transfer_qwen_priority(mod._read(ctx.SM_STATE))
    assert mod._read(ctx.SM_STATE)["state"] == "TRANSFER_QWEN_PRIORITY"
    assert any("paused" in m for m in ctx.tg)


# -- 8. restart during every state: no crash, no double side effect -------------------------
def test_08_restart_during_every_state(tmp_path, monkeypatch):
    mod = osup
    for state in mod.STATES:
        ctx = _configure(mod, monkeypatch, tmp_path / state.lower())
        mod._write(ctx.SM_STATE, {"state": state, "entered_at": mod._now(), "input_identity": None})
        rc1 = mod.tick()
        rc2 = mod.tick()  # a second tick must not crash or re-fire a completed side effect
        assert rc1 == 0 and rc2 == 0, state
        assert not any("tick error" in m for m in ctx.tg), (state, ctx.tg)
        assert mod._read(ctx.SM_STATE).get("state") in mod.STATES, state


# -- 9. duplicate launch prevention ---------------------------------------------------------
def test_09_duplicate_launch_prevention(tmp_path, monkeypatch):
    mod = osup
    ctx = _configure(mod, monkeypatch, tmp_path)
    # atomic one-use claim
    assert mod._claim("foo") is True
    assert mod._claim("foo") is False

    # a claim-guarded handler performs its side effect exactly once
    mod._write(ctx.QWEN_META / "model.safetensors.index.json",
               {"weight_map": {"model.norm.weight": "s1"}})

    def admit_state():
        return {"state": "ADMIT_QWEN", "entered_at": mod._now(), "input_identity": None}

    mod._write(ctx.SM_STATE, admit_state())
    mod.h_admit_qwen(mod._read(ctx.SM_STATE))
    plan_path = ctx.GF / "QWEN3_235B_PRIORITY_PLAN.json"
    assert plan_path.exists()
    assert mod._read(ctx.SM_STATE)["state"] == "TRANSFER_QWEN_PRIORITY"

    plan_path.unlink()                       # remove the artifact of the first run
    mod._write(ctx.SM_STATE, admit_state())  # rewind the state and re-run
    mod.h_admit_qwen(mod._read(ctx.SM_STATE))
    assert not plan_path.exists(), "side effect fired twice under a held claim"
    assert mod._read(ctx.SM_STATE)["state"] == "ADMIT_QWEN", "advanced twice under a held claim"

    # every fully claim-guarded handler no-ops when its claim is already present
    guards = [("seal_conclusion", "SEAL_120B_CONCLUSION", mod.h_seal),
              ("narrow_refinement", "NARROW_RATE_REFINEMENT", mod.h_narrow_refinement),
              ("run_q0q1q2", "RUN_QWEN_Q0_Q1_Q2", mod.h_run_q0q1q2),
              ("launch_qwen", "LAUNCH_QWEN", mod.h_launch_qwen)]
    for claim, state, handler in guards:
        assert mod._claim(claim) is True         # a prior tick already holds it
        mod._write(ctx.SM_STATE, {"state": state, "entered_at": mod._now(), "input_identity": None})
        handler(mod._read(ctx.SM_STATE))
        assert mod._read(ctx.SM_STATE)["state"] == state, (state, "advanced despite held claim")


# -- 10. Q2 failure blocks ------------------------------------------------------------------
def test_10_q2_failure_blocks(tmp_path, monkeypatch):
    mod = osup
    ctx = _configure(mod, monkeypatch, tmp_path)
    ctx.run_rc = lambda argv: 1 if "qwen3_moe_adapter" in " ".join(map(str, argv)) else 0
    mod._write(ctx.SM_STATE, {"state": "RUN_QWEN_Q0_Q1_Q2", "entered_at": mod._now(),
                              "input_identity": None})
    mod.h_run_q0q1q2(mod._read(ctx.SM_STATE))
    st = mod._read(ctx.SM_STATE)
    assert st["state"] == "BLOCKED"
    reason = st.get("blocked_reason") or ""
    assert "Q0/Q1" in reason or "adapter" in reason


# -- 11. successful Qwen ignition path ------------------------------------------------------
def test_11_successful_qwen_ignition_path(tmp_path, monkeypatch):
    mod = osup
    ctx = _configure(mod, monkeypatch, tmp_path)
    plan_shard = "model-00001-of-00002.safetensors"
    mod._write(ctx.GF / "QWEN3_235B_PRIORITY_PLAN.json", {"priority_shards": [plan_shard]})
    ctx.QWEN_DIR.mkdir(parents=True, exist_ok=True)
    (ctx.QWEN_DIR / plan_shard).write_bytes(b"qwen-bytes")  # already staged -> transfer completes
    mod._write(ctx.SM_STATE, {"state": "TRANSFER_QWEN_PRIORITY", "entered_at": mod._now(),
                              "input_identity": mod.QWEN_REV})

    # Drive TRANSFER -> RUN_Q -> LAUNCH -> MONITOR via the real dispatcher.
    seen = []
    for _ in range(4):
        mod.tick()
        seen.append(mod._read(ctx.SM_STATE)["state"])

    assert seen[0] == "RUN_QWEN_Q0_Q1_Q2"
    assert seen[1] == "LAUNCH_QWEN"
    assert seen[2] == "MONITOR_QWEN"
    assert mod._read(ctx.SM_STATE)["state"] == "MONITOR_QWEN"

    recs = {p.name for p in ctx.RECEIPTS.glob("*.json")}
    assert {"transfer_qwen_priority.json", "run_q0q1q2.json", "launch_qwen.json"} <= recs
    assert ctx.tg, "no Telegram notifications emitted on the ignition path"
    assert any("Qwen" in m for m in ctx.tg)


# -- CRITICAL: a successful release removes ONLY the seven shards ----------------------------
def test_release_success_deletes_only_shards(tmp_path, monkeypatch):
    mod = osup
    ctx = _configure(mod, monkeypatch, tmp_path)
    decoy = _build_release(mod, ctx, monkeypatch, all_green=True)
    orig = list(ctx.SHARDS)
    mod.h_release_120b_source(mod._read(ctx.SM_STATE))

    for p in orig:
        assert not p.exists(), f"shard not released: {p.name}"
    for m in ("config.json", "model.safetensors.index.json", "dtypes.json"):
        assert (ctx.ORIGINAL / m).exists(), f"metadata lost: {m}"
    assert (ctx.MODEL_DIR / "tokenizer.json").exists()
    assert (ctx.MODEL_DIR / "chat_template.jinja").exists()
    assert decoy.exists(), "a non-shard file was deleted (globbing!)"

    assert mod._read(ctx.SM_STATE)["state"] == "ADMIT_QWEN"
    rec = mod._read(ctx.RECEIPTS / "release_source.json")
    assert len(rec.get("freed", [])) == 7


# -- CRITICAL: unsafe shard path (outside dir / wrong name) refuses, deletes nothing ---------
def test_release_refuses_unsafe_paths(tmp_path, monkeypatch):
    mod = osup

    # (a) a shard path OUTSIDE the original dir.
    ctx = _configure(mod, monkeypatch, tmp_path / "outside")
    _build_release(mod, ctx, monkeypatch, all_green=True)
    orig = list(ctx.SHARDS)
    outside = ctx.ROOT / "model--00007-of-00007.safetensors"   # valid name, wrong parent
    monkeypatch.setattr(mod, "SHARDS", orig[:6] + [outside])
    mod.h_release_120b_source(mod._read(ctx.SM_STATE))
    assert mod._read(ctx.SM_STATE)["state"] == "BLOCKED"
    for p in orig:
        assert p.exists(), f"deleted despite an unsafe (outside) path: {p.name}"

    # (b) a shard path with the WRONG name inside the original dir.
    ctx2 = _configure(mod, monkeypatch, tmp_path / "wrongname")
    _build_release(mod, ctx2, monkeypatch, all_green=True)
    orig2 = list(ctx2.SHARDS)
    wrong = ctx2.ORIGINAL / "model--BADNAME.safetensors"       # right dir, wrong name
    monkeypatch.setattr(mod, "SHARDS", orig2[:6] + [wrong])
    mod.h_release_120b_source(mod._read(ctx2.SM_STATE))
    assert mod._read(ctx2.SM_STATE)["state"] == "BLOCKED"
    for p in orig2:
        assert p.exists(), f"deleted despite an unsafe (wrong-name) path: {p.name}"


# -- 12. _reverify_gates enforces fresh + authorized + 15/15 (deletion-authorization hardening) --
def test_reverify_gates_fresh_authorized(tmp_path, monkeypatch):
    mod = osup
    ctx = _configure(mod, monkeypatch, tmp_path)
    ready = ctx.GF / "GPT_OSS_120B_SOURCE_RELEASE_READINESS.json"

    def run_writes_fresh(authorized, rc=0):
        def fake_run(argv, *a, **k):
            if rc == 0:
                mod._write(ready, {"gates": {f"g{i}": {"status": "green"} for i in range(15)},
                                   "release_authorized": authorized})
            return types.SimpleNamespace(returncode=rc, stdout="", stderr="")
        monkeypatch.setattr(mod, "subprocess", types.SimpleNamespace(run=fake_run))

    # authorized + fresh + 15/15 -> True
    run_writes_fresh(True)
    ok, d = mod._reverify_gates()
    assert ok is True and d["green"] == 15 and d["release_authorized"] is True

    # authorized field False -> refuse even though gates green
    run_writes_fresh(False)
    ok, d = mod._reverify_gates()
    assert ok is False

    # subprocess crash (nonzero, no fresh write) -> refuse (no stale-verdict fallback)
    run_writes_fresh(True, rc=1)
    ready.write_text(json.dumps({"gates": {f"g{i}": {"status": "green"} for i in range(15)},
                                 "release_authorized": True}))  # stale all-green on disk
    import time as _t
    _t.sleep(0.01)
    ok, d = mod._reverify_gates()
    assert ok is False, "must not delete on a stale all-green verdict when the re-run failed"


# -- 13. campaign crash -> auto-resume + alert; live -> no resume; backoff ------------------
def test_campaign_crash_auto_resumes(tmp_path, monkeypatch):
    mod = osup
    ctx = _configure(mod, monkeypatch, tmp_path)
    mod._write(ctx.CAMP_STATE, {"final": False, "rows_done": 12, "rows_total": 28})
    mod._write(ctx.CAMP / "leases/doctor_campaign.lease", {"pid": 999999})
    monkeypatch.setattr(mod, "_pid_alive", lambda pid: False)   # controller dead
    mod._write(ctx.SM_STATE, {"state": "WAIT_120B_FINAL", "entered_at": mod._now()})
    mod.h_wait_120b_final(mod._read(ctx.SM_STATE))
    assert any("gravity_frontier_correction_wave.py" in " ".join(c) and "detach" in c
               for c in ctx.calls), ctx.calls
    assert any("crashed" in t and "auto-resuming" in t for t in ctx.tg)
    n = len(ctx.calls)
    mod.h_wait_120b_final(mod._read(ctx.SM_STATE))
    assert len(ctx.calls) == n, "backoff must prevent an immediate second resume"


def test_campaign_alive_no_resume(tmp_path, monkeypatch):
    mod = osup
    ctx = _configure(mod, monkeypatch, tmp_path)
    mod._write(ctx.CAMP_STATE, {"final": False, "rows_done": 12, "rows_total": 28})
    mod._write(ctx.CAMP / "leases/doctor_campaign.lease", {"pid": 42390})
    monkeypatch.setattr(mod, "_pid_alive", lambda pid: True)    # controller alive
    ctx.CAMP_CKPT.mkdir(parents=True, exist_ok=True)
    mod._write(ctx.CAMP / "heartbeat/doctor_campaign.heartbeat.json", {"row_id": "code_py__D4_pq_doctor"})
    mod._write(ctx.SM_STATE, {"state": "WAIT_120B_FINAL", "entered_at": mod._now()})
    mod.h_wait_120b_final(mod._read(ctx.SM_STATE))
    assert not any("detach" in c for c in ctx.calls), "must never resume a live campaign"
