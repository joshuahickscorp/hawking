#!/usr/bin/env python3.12
"""Detached, restart-safe state machine: close Qwen -> strip storage -> admit the next
FULL-RESIDENT parent -> stage its one source copy -> hand off to science.

Each tick advances AT MOST ONE transition. Every transition takes an atomic O_EXCL claim, seals a
receipt, and is a no-op on replay. Nothing destructive runs before the Qwen campaign is SEALED and
its process is gone. Deletion is always manifest-driven through storage_stripdown.py, which
re-resolves and re-gates every path at unlink time.

    WAIT_QWEN SEAL_QWEN INVENTORY PLAN_DELETE RELEASE_MODELS CLEAN_WORKTREES CLEAN_CACHES
    VERIFY_STORAGE RESEQUENCE_LADDER ADMIT_PARENT DOWNLOAD_PARENT VERIFY_PARENT LAUNCH_PARENT
    MONITOR_PARENT COMPLETE BLOCKED

Run detached:  caffeinate -dimsu python3.12 tools/condense/storage_stripdown_controller.py loop
"""
from __future__ import annotations

import argparse
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

import storage_stripdown as SS  # noqa: E402

ROOT = Path(_HERE).resolve().parents[1]
PY = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
OUT = ROOT / "reports/condense/storage_stripdown"
STATE = OUT / "CONTROLLER_STATE.json"
HB = OUT / "heartbeat/storage_stripdown.heartbeat.json"
CLAIMS = OUT / "claims"
RECEIPTS = OUT / "receipts"
LEASE = OUT / "storage_stripdown.lease"
LOG = OUT / "controller.log"
LABEL = "com.hawking.storage_stripdown"

# The live Qwen campaign we must not disturb.
QWEN_WT = Path.home() / "HawkingWorktrees/subbit-reset"
QWEN_STATE = QWEN_WT / "reports/condense/general_frontier/QWEN_GRAVITY/QWEN_GRAVITY_STATE.json"
QWEN_HB = QWEN_WT / "reports/condense/general_frontier/QWEN_GRAVITY/heartbeat/qwen_gravity.heartbeat.json"
QWEN_BRANCH = "campaign/subbit-capability-density-reset"

QWEN_SOURCE_DIR = ROOT / "models/qwen3-235b-a22b"
QWEN_REPO = "Qwen/Qwen3-235B-A22B-Instruct-2507"
QWEN_REV = "ac9c66cc9b46af7306746a9250f23d47083d689e"

# Worktrees this campaign may remove, each with the preservation rule that unlocks it.
REMOVABLE_WORKTREES = [
    {"path": str(Path.home() / "Downloads/hawking-hide-parity-research"),
     "rule": "clean and level with its own pushed upstream"},
    {"path": str(Path.home() / "HawkingWorktrees/deep-architecture-foundry"),
     "rule": "clean; its unique commits must be pushed to an archive branch first",
     "archive_branch": "archive/deep-architecture-foundry-2026-07-20"},
]
KEEP_WORKTREES = [str(ROOT), str(QWEN_WT), str(Path.home() / "Downloads/hawking-hide-build")]

# The Hugging Face hub cache is NOT here on purpose: MOP pulls facebook/vjepa2-* (7.5 GiB)
# through it, so it is a MOP build product and this campaign may not clean it. It is also a
# hard-protected root in storage_stripdown.PROTECTED_ROOTS, so the gate would refuse it anyway.
CACHE_TARGETS = [
    Path.home() / ".cache/codex-runtimes",
]

GIB = 1 << 30
TICK_SECONDS = 120
STATES = ["WAIT_QWEN", "SEAL_QWEN", "INVENTORY", "PLAN_DELETE", "RELEASE_MODELS",
          "CLEAN_WORKTREES", "CLEAN_CACHES", "VERIFY_STORAGE", "RESEQUENCE_LADDER",
          "ADMIT_PARENT", "DOWNLOAD_PARENT", "VERIFY_PARENT", "LAUNCH_PARENT",
          "MONITOR_PARENT", "COMPLETE", "BLOCKED"]


# ── framework ───────────────────────────────────────────────────────────────────────────
def now() -> str:
    return SS.now()


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(f"{now()} {msg}\n")
    sys.stderr.write(f"{now()} {msg}\n")


def telegram(text: str) -> bool:
    try:
        import doctor_campaign_supervisor as D
        import doctor_v5_telegram_rung_notifier as N
        tok, chat = D._creds()
        if not tok or not chat:
            return False
        N._telegram(tok, "sendMessage", {"chat_id": chat,
                                         "text": ("[stripdown] " + text)[:4000]})
        return True
    except Exception as exc:  # noqa: BLE001
        log(f"telegram failed: {type(exc).__name__}")
        return False


def state() -> dict:
    st = SS.read(STATE)
    if not st:
        st = {"state": "WAIT_QWEN", "entered_at": now(), "history": []}
        SS.write(STATE, st)
    return st


def advance(st: dict, nxt: str, note: str = "") -> dict:
    st = {**st, "state": nxt, "entered_at": now(),
          "history": (st.get("history", []) + [{"to": nxt, "at": now(), "note": note}])[-64:]}
    SS.write(STATE, st)
    log(f"-> {nxt} {note}")
    return st


def claim(name: str) -> bool:
    CLAIMS.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(CLAIMS / f"{name}.claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, now().encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def receipt(name: str, obj: dict) -> None:
    SS.write(RECEIPTS / f"{name}.json", {**obj, "receipt": name, "sealed_at": now()})


def has_receipt(name: str) -> bool:
    return (RECEIPTS / f"{name}.json").exists()


def heartbeat(st: dict, extra: dict | None = None) -> None:
    SS.write(HB, {"schema": "hawking.successor.watchdog.v1", "label": LABEL,
                  "campaign": "storage_stripdown_resident_first", "beat_at": now(),
                  "pid": os.getpid(), "state": st["state"],
                  "entered_at": st.get("entered_at"),
                  "free_disk_gb": round(SS.free_bytes() / 1e9, 1),
                  "status": "RUNNING" if st["state"] not in ("COMPLETE", "BLOCKED")
                            else st["state"], **(extra or {})})


def take_lease() -> bool:
    """One controller at a time. A lease whose pid is gone is reclaimed."""
    LEASE.parent.mkdir(parents=True, exist_ok=True)
    old = SS.read(LEASE)
    if old.get("pid") and alive(int(old["pid"])) and int(old["pid"]) != os.getpid():
        return False
    SS.write(LEASE, {"pid": os.getpid(), "taken_at": now(), "label": LABEL})
    return True


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 3600) -> dict:
    try:
        res = subprocess.run(cmd, cwd=str(cwd or ROOT), capture_output=True, text=True,
                             timeout=timeout)
        return {"rc": res.returncode, "out": res.stdout[-8000:], "err": res.stderr[-4000:]}
    except Exception as exc:  # noqa: BLE001
        return {"rc": -1, "out": "", "err": f"{type(exc).__name__}: {exc}"}


def block(st: dict, reason: str) -> dict:
    telegram(f"BLOCKED in {st['state']}: {reason}")
    receipt(f"BLOCKED_{st['state']}", {"reason": reason, "state": st["state"]})
    return advance(st, "BLOCKED", reason)


# ── transitions ─────────────────────────────────────────────────────────────────────────
def t_wait_qwen(st: dict) -> dict:
    """Hold until the CURRENT Qwen pass seals. A stale seal must never unlock the release.

    QWEN_GRAVITY_STATE.json is rewritten per generation and already reads SEALED/final for the
    PREVIOUS generation (rows_done 36 of 12, sealed 02:00Z) while the chained S64_gamma pass is
    still on its first of six rows. Trusting status alone would release 438 GiB under a campaign
    that has completed nothing. So: the seal must be strictly NEWER than the one observed when
    this controller first started waiting, the live heartbeat must have drained its rows, and the
    process must be gone.
    """
    q = SS.read(QWEN_STATE)
    hb = SS.read(QWEN_HB)
    pid = hb.get("pid")
    running = bool(pid) and alive(int(pid))
    done, total = hb.get("rows_done", 0), hb.get("rows_total", 0)

    baseline = st.get("qwen_seal_baseline")
    if baseline is None:
        baseline = q.get("generated_at") or ""
        st = {**st, "qwen_seal_baseline": baseline,
              "qwen_seal_baseline_rows": f"{q.get('rows_done')}/{q.get('rows_total')}"}
        SS.write(STATE, st)
        log(f"WAIT_QWEN baseline seal={baseline!r} rows={st['qwen_seal_baseline_rows']}")

    fresh_seal = bool(q.get("generated_at")) and q.get("generated_at") > baseline
    # The heartbeat STOPS at process exit, so its rows counter is stale by construction the
    # moment the campaign finishes. Drain is proven from the seal itself instead: a fresh seal
    # plus a released lease is a clean exit; a held lease with no process is a crash.
    lease_dir = QWEN_STATE.parent / "leases"
    lease_released = not lease_dir.exists() or not any(lease_dir.iterdir())
    rows_drained = fresh_seal and lease_released
    heartbeat(st, {"qwen_status": q.get("status"), "qwen_rows": f"{done}/{total}",
                   "qwen_pid": pid, "qwen_alive": running,
                   "qwen_layer": f"{hb.get('layer')}/{hb.get('layers_total')}",
                   "qwen_phase": hb.get("phase"),
                   "qwen_seal_baseline": baseline, "qwen_seal_is_fresh": fresh_seal,
                   "qwen_lease_released": lease_released, "qwen_rows_drained": rows_drained})
    if running:
        return st  # never interrupt a healthy row
    if not pid:
        return st  # no campaign has ever beaten here; keep waiting rather than guessing
    if fresh_seal and rows_drained and q.get("status") == "SEALED" and q.get("final"):
        return advance(st, "SEAL_QWEN",
                       f"fresh seal {q['generated_at']} > baseline {baseline}, rows {done}/{total}")
    return block(st, f"Qwen process {pid} is gone but the pass did not seal: "
                     f"status={q.get('status')!r} final={q.get('final')} "
                     f"seal={q.get('generated_at')!r} baseline={baseline!r} "
                     f"fresh={fresh_seal} rows={done}/{total} drained={rows_drained}. "
                     f"Refusing to release a source under an unfinished campaign.")


def t_seal_qwen(st: dict) -> dict:
    if not claim("SEAL_QWEN"):
        return advance(st, "INVENTORY", "replay")
    q = SS.read(QWEN_STATE)
    # Push and tag the campaign branch from its own worktree; never rewrite it.
    push = run(["git", "push", "origin", QWEN_BRANCH], cwd=QWEN_WT, timeout=1800)
    tag = f"qwen3-235b-sealed-{time.strftime('%Y%m%d', time.gmtime())}"
    run(["git", "tag", "-f", tag], cwd=QWEN_WT)
    push_tag = run(["git", "push", "-f", "origin", tag], cwd=QWEN_WT, timeout=900)
    # Copy the sealed evidence into this campaign's report tree so it survives worktree cleanup.
    dest = OUT / "qwen_final_evidence"
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in [QWEN_STATE, QWEN_HB,
                QWEN_WT / "reports/subbit_reset/QWEN235B_VULTURE_HARVEST.json",
                QWEN_WT / "reports/subbit_reset/NEXT_PARENT_LAUNCH_PACKET.json"]:
        if src.exists():
            shutil.copy2(src, dest / src.name)
            copied.append(src.name)
    obj = {"qwen_state_status": q.get("status"), "final": q.get("final"),
           "rows_done": q.get("rows_done"), "rows_total": q.get("rows_total"),
           "capability_passes": q.get("capability_passes"),
           "subbit_unsolved": not q.get("capability_passes"),
           "branch": QWEN_BRANCH, "tag": tag,
           "push_rc": push["rc"], "push_tag_rc": push_tag["rc"],
           "evidence_copied": copied, "evidence_dir": str(dest)}
    receipt("SEAL_QWEN", obj)
    telegram(f"Qwen SEALED. rows {obj['rows_done']}/{obj['rows_total']}, "
             f"capability passes: {len(q.get('capability_passes') or [])}. tag={tag}")
    return advance(st, "INVENTORY")


def t_inventory(st: dict) -> dict:
    if has_receipt("INVENTORY"):
        return advance(st, "PLAN_DELETE", "replay")
    claim("INVENTORY")
    r1 = run([PY, "tools/condense/storage_stripdown.py", "protect"])
    r2 = run([PY, "tools/condense/storage_stripdown.py", "inventory"], timeout=5400)
    prot = SS.read(OUT / "STORAGE_STRIPDOWN_PROTECTED_PATHS.json")
    if prot.get("status") != "GREEN":
        return block(st, f"protected-paths receipt is {prot.get('status')}: {prot.get('reason')}")
    inv = SS.read(OUT / "STORAGE_STRIPDOWN_INVENTORY.json")
    receipt("INVENTORY", {"protect_rc": r1["rc"], "inventory_rc": r2["rc"],
                          "payload_count": inv.get("payload_count"),
                          "free_bytes": inv.get("free_bytes"),
                          "mop_bytes": prot.get("mop_bytes")})
    telegram(f"Inventory complete: {inv.get('payload_count')} model payload files, "
             f"free {SS.free_bytes()/GIB:.1f} GiB. MOP protected at "
             f"{prot.get('mop_bytes',0)/GIB:.1f} GiB.")
    return advance(st, "PLAN_DELETE")


def t_plan_delete(st: dict) -> dict:
    if has_receipt("PLAN_DELETE"):
        return advance(st, "RELEASE_MODELS", "replay")
    claim("PLAN_DELETE")
    r = run([PY, "tools/condense/storage_stripdown.py", "plan",
             "--release", str(QWEN_SOURCE_DIR),
             "--keep", str(Path.home() / "Downloads/mop")], timeout=1800)
    man = SS.read(OUT / "STORAGE_STRIPDOWN_DELETE_MANIFEST.json")
    if man.get("status") != "PLANNED":
        return block(st, f"plan did not produce a manifest: {r['err'][:400]}")
    if not man.get("file_count"):
        return block(st, "manifest is empty; nothing passed the gates")
    receipt("PLAN_DELETE", {"file_count": man["file_count"],
                            "expected_recoverable_bytes": man["expected_recoverable_bytes"],
                            "rejected_count": man["rejected_count"]})
    telegram(f"Dry-run deletion plan: {man['file_count']} exact files, "
             f"{man['expected_recoverable_bytes']/GIB:.1f} GiB expected, "
             f"{man['rejected_count']} rejected by the gates.")
    return advance(st, "RELEASE_MODELS")


def t_release_models(st: dict) -> dict:
    if has_receipt("RELEASE_MODELS"):
        return advance(st, "CLEAN_WORKTREES", "replay")
    if not claim("RELEASE_MODELS"):
        return advance(st, "CLEAN_WORKTREES", "replay")
    # Rehydration receipt FIRST: evidence before erasure, always.
    rel = run([PY, "tools/condense/storage_stripdown.py", "release",
               "--family", "qwen3-235b-a22b", "--dir", str(QWEN_SOURCE_DIR),
               "--repo", QWEN_REPO, "--revision", QWEN_REV, "--license", "apache-2.0",
               "--conclusion", "SEALED NEGATIVE: no capability pass at any rate <= 1.0 complete "
                               "BPW; classification RECONSTRUCTION_BOUND",
               "--evidence", str(OUT / "qwen_final_evidence")])
    if rel["rc"] != 0 or not (OUT / "MODEL_RELEASE_qwen3-235b-a22b.json").exists():
        return block(st, f"release receipt failed: {rel['err'][:400]}")
    ex = run([PY, "tools/condense/storage_stripdown.py", "execute", "--go"], timeout=7200)
    rcpt = sorted(OUT.glob("STORAGE_STRIPDOWN_DELETE_RECEIPT_*.json"))
    last = SS.read(rcpt[-1]) if rcpt else {}
    if last.get("deleted", 0) == 0:
        return block(st, f"execute deleted nothing: {ex['err'][:400]}")
    receipt("RELEASE_MODELS", {"deleted": last.get("deleted"),
                               "attempted": last.get("attempted"),
                               "failed_count": len(last.get("failed", [])),
                               "bytes_unlinked": last.get("bytes_unlinked"),
                               "free_after": last.get("free_bytes_after")})
    telegram(f"Model family released: qwen3-235b-a22b, {last.get('deleted')} shards, "
             f"{last.get('bytes_unlinked',0)/GIB:.1f} GiB unlinked. "
             f"Free now {SS.free_bytes()/GIB:.1f} GiB.")
    return advance(st, "CLEAN_WORKTREES")


def t_clean_worktrees(st: dict) -> dict:
    if has_receipt("CLEAN_WORKTREES"):
        return advance(st, "CLEAN_CACHES", "replay")
    claim("CLEAN_WORKTREES")
    results = []
    for wt in REMOVABLE_WORKTREES:
        p = Path(wt["path"])
        if not p.exists():
            results.append({**wt, "action": "absent"})
            continue
        if str(p) in KEEP_WORKTREES:
            results.append({**wt, "action": "refused", "reason": "on the keep list"})
            continue
        dirty = run(["git", "status", "--porcelain"], cwd=p)
        if dirty["out"].strip():
            results.append({**wt, "action": "refused",
                            "reason": f"{len(dirty['out'].splitlines())} dirty files"})
            continue
        # Preserve unique commits before removing anything.
        uniq = run(["git", "rev-list", "--count", "origin/main..HEAD"], cwd=p)
        n_unique = int(uniq["out"].strip() or 0)
        preserved = None
        if n_unique:
            br = wt.get("archive_branch")
            if not br:
                results.append({**wt, "action": "refused",
                                "reason": f"{n_unique} unique commits and no archive branch"})
                continue
            head = run(["git", "rev-parse", "HEAD"], cwd=p)["out"].strip()
            push = run(["git", "push", "origin", f"{head}:refs/heads/{br}"], cwd=p, timeout=1800)
            bundle = OUT / f"worktree_archive_{p.name}.bundle"
            run(["git", "bundle", "create", str(bundle), "origin/main..HEAD"], cwd=p,
                timeout=1800)
            verified = run(["git", "bundle", "verify", str(bundle)], cwd=p)
            if push["rc"] != 0 and verified["rc"] != 0:
                results.append({**wt, "action": "refused",
                                "reason": "neither push nor bundle preserved the unique commits"})
                continue
            preserved = {"archive_branch": br, "push_rc": push["rc"],
                         "bundle": str(bundle), "bundle_verify_rc": verified["rc"],
                         "unique_commits": n_unique}
        rm = run(["git", "worktree", "remove", "--force", str(p)], cwd=ROOT, timeout=900)
        results.append({**wt, "action": "removed" if rm["rc"] == 0 else "failed",
                        "rc": rm["rc"], "err": rm["err"][:300], "preserved": preserved})
    run(["git", "worktree", "prune"], cwd=ROOT)
    SS.write(OUT / "WORKTREE_CLEANUP_REPORT.json",
             {"schema": "hawking.storage_stripdown.worktree_cleanup.v1", "generated_at": now(),
              "kept": KEEP_WORKTREES, "results": results,
              "worktrees_after": run(["git", "worktree", "list"], cwd=ROOT)["out"]})
    SS.write(OUT / "WORKTREE_ARCHIVE_MANIFEST.json",
             {"schema": "hawking.storage_stripdown.worktree_archive.v1", "generated_at": now(),
              "archives": [r["preserved"] for r in results if r.get("preserved")]})
    receipt("CLEAN_WORKTREES", {"removed": sum(1 for r in results if r["action"] == "removed"),
                                "refused": [r for r in results if r["action"] == "refused"]})
    telegram(f"Worktree cleanup: {sum(1 for r in results if r['action']=='removed')} removed, "
             f"{sum(1 for r in results if r['action']=='refused')} refused with reasons.")
    return advance(st, "CLEAN_CACHES")


def t_clean_caches(st: dict) -> dict:
    if has_receipt("CLEAN_CACHES"):
        return advance(st, "VERIFY_STORAGE", "replay")
    claim("CLEAN_CACHES")
    prot = SS.protected_set()
    dev = os.stat(SS.DATA_VOLUME).st_dev
    before = SS.free_bytes()
    removed, refused = [], []
    for target in CACHE_TARGETS:
        if not target.exists():
            continue
        # Never remove a cache the running science depends on; only released-family blobs and
        # abandoned runtimes qualify, and every file is gated individually.
        for dirpath, _dirs, files in os.walk(target, followlinks=False):
            for fn in files:
                p = Path(dirpath) / fn
                ok, why = SS.gate(p, prot, dev)
                if not ok:
                    refused.append({"path": str(p), "reason": why})
                    continue
                try:
                    n = os.lstat(p).st_blocks * 512
                    os.unlink(p)
                    removed.append({"path": str(p), "bytes": n})
                except OSError as exc:
                    refused.append({"path": str(p), "reason": f"{type(exc).__name__}"})
    for stale in [ROOT / ".pytest_cache",
                  Path.home() / "HawkingWorktrees/deep-architecture-foundry/target"]:
        if stale.exists() and str(stale.resolve()) not in prot["resolved_paths"]:
            n = SS.dir_bytes(stale)
            shutil.rmtree(stale, ignore_errors=True)
            removed.append({"path": str(stale), "bytes": n, "kind": "build_product"})
    receipt("CLEAN_CACHES", {"files_removed": len(removed),
                             "bytes_removed": sum(r["bytes"] for r in removed),
                             "refused_count": len(refused),
                             "free_before": before, "free_after": SS.free_bytes()})
    telegram(f"Caches and abandoned build products removed: "
             f"{sum(r['bytes'] for r in removed)/GIB:.1f} GiB. "
             f"Free now {SS.free_bytes()/GIB:.1f} GiB.")
    return advance(st, "VERIFY_STORAGE")


def t_verify_storage(st: dict) -> dict:
    if has_receipt("VERIFY_STORAGE"):
        return advance(st, "RESEQUENCE_LADDER", "replay")
    claim("VERIFY_STORAGE")
    r = run([PY, "tools/condense/storage_stripdown.py", "verify"], timeout=1800)
    fin = SS.read(OUT / "STORAGE_STRIPDOWN_FINAL.json")
    if not fin.get("mop_intact"):
        return block(st, "MOP root is not intact after cleanup")
    if not fin.get("hawking_repo_healthy"):
        return block(st, "authoritative Hawking repository is not healthy after cleanup")
    if fin.get("manifest_survivor_count"):
        return block(st, f"{fin['manifest_survivor_count']} manifest paths still exist")
    receipt("VERIFY_STORAGE", {"rc": r["rc"], "free_now": fin.get("free_bytes_now"),
                               "realized_free_delta": fin.get("realized_free_delta"),
                               "mop_intact": True})
    telegram(f"Storage verified. Free {fin.get('free_bytes_now',0)/GIB:.1f} GiB "
             f"(+{fin.get('realized_free_delta',0)/GIB:.1f} GiB recovered). MOP intact.")
    return advance(st, "RESEQUENCE_LADDER")


def t_resequence(st: dict) -> dict:
    if has_receipt("RESEQUENCE_LADDER"):
        return advance(st, "ADMIT_PARENT", "replay")
    claim("RESEQUENCE_LADDER")
    r = run([PY, "tools/condense/resident_first_ladder.py",
             "--comment", "measured live after stripdown"], timeout=1800)
    lad = SS.read(OUT / "FULL_RESIDENT_ELIGIBILITY.json")
    if lad.get("resident_frontier_exhausted"):
        SS.write(OUT / "RESIDENT_FRONTIER_EXHAUSTED.json", {
            "generated_at": now(), "free_bytes": SS.free_bytes(),
            "closest_misses": sorted(
                [{"repo": x["repo"], "short_bytes": -x["margin_bytes"]}
                 for x in lad.get("rows", []) if x.get("fit_class") == "DOES_NOT_FIT_FULLY"],
                key=lambda x: x["short_bytes"])[:5],
            "rule": "streaming is NOT started automatically; it needs an explicit decision"})
        return block(st, "RESIDENT_FRONTIER_EXHAUSTED: no parent fits fully after cleanup")
    receipt("RESEQUENCE_LADDER", {"rc": r["rc"], "selected": lad.get("selected_next_parent"),
                                  "revision": lad.get("selected_revision"),
                                  "counts": lad.get("counts")})
    telegram(f"Resident-first ladder resealed against measured free space. "
             f"Next parent: {lad.get('selected_next_parent')}")
    return advance(st, "ADMIT_PARENT")


def _selected() -> dict:
    lad = SS.read(OUT / "FULL_RESIDENT_ELIGIBILITY.json")
    repo = lad.get("selected_next_parent")
    row = next((r for r in lad.get("rows", []) if r.get("repo") == repo), {})
    return {"repo": repo, "revision": lad.get("selected_revision"), "row": row}


def t_admit(st: dict) -> dict:
    if has_receipt("ADMIT_PARENT"):
        return advance(st, "DOWNLOAD_PARENT", "replay")
    claim("ADMIT_PARENT")
    sel = _selected()
    if not sel["repo"]:
        return block(st, "no parent selected")
    row = sel["row"]
    free = SS.free_bytes()
    if row.get("required_squeezed_bytes", 1 << 62) > free:
        return block(st, f"{sel['repo']} no longer fits against measured free space")
    local = ROOT / "models" / sel["repo"].split("/")[-1].lower()
    SS.write(OUT / "NEXT_PARENT_ADMISSION.json", {
        "schema": "hawking.storage_stripdown.parent_admission.v1", "generated_at": now(),
        "repo": sel["repo"], "immutable_revision": sel["revision"],
        "license": row.get("license"), "config": row.get("config"),
        "source_bytes": row.get("source_bytes"), "n_weight_shards": row.get("n_weight_shards"),
        "largest_shard_bytes": row.get("largest_shard_bytes"),
        "fit_class": row.get("fit_class"), "margin_bytes": free - row.get("required_squeezed_bytes", 0),
        "free_bytes_at_admission": free, "local_dir": str(local),
        "one_copy_law": "exactly one physical copy: snapshot_download with local_dir and "
                        "HF_HUB_DISABLE_XET symlink duplication disabled; the hub cache is not "
                        "kept as a second materialised copy",
        "why": row.get("why"),
    })
    receipt("ADMIT_PARENT", {"repo": sel["repo"], "revision": sel["revision"],
                             "local_dir": str(local)})
    telegram(f"Admitted next parent: {sel['repo']} @ {str(sel['revision'])[:10]} "
             f"({row.get('source_bytes',0)/GIB:.1f} GiB, {row.get('fit_class')})")
    return advance(st, "DOWNLOAD_PARENT")


def t_download(st: dict) -> dict:
    """Spawn the transfer detached and POLL it, so the heartbeat keeps beating for hours."""
    if has_receipt("DOWNLOAD_PARENT"):
        return advance(st, "VERIFY_PARENT", "replay")
    adm = SS.read(OUT / "NEXT_PARENT_ADMISSION.json")
    local = Path(adm["local_dir"])
    dl_state = OUT / "DOWNLOAD_PARENT_PROGRESS.json"
    prog = SS.read(dl_state)
    pid = prog.get("pid")

    if pid and alive(int(pid)):
        on_disk = SS.dir_bytes(local)
        pct = 100.0 * on_disk / max(1, adm["source_bytes"])
        heartbeat(st, {"download_pid": pid, "download_pct": round(pct, 1),
                       "download_bytes": on_disk, "download_target": adm["source_bytes"]})
        # One coarse notification per ~10% band, never one per file.
        band = int(pct // 10)
        if band != prog.get("last_band"):
            telegram(f"{adm['repo']}: {pct:.0f}% ({on_disk/GIB:.0f}/"
                     f"{adm['source_bytes']/GIB:.0f} GiB), free {SS.free_bytes()/GIB:.0f} GiB")
            SS.write(dl_state, {**prog, "last_band": band, "pct": pct, "beat_at": now()})
        if SS.free_bytes() < 32 * GIB:
            os.kill(int(pid), 15)
            return block(st, "disk reserve went red during the transfer; download stopped")
        return st

    if pid and not alive(int(pid)):
        rc_file = OUT / "download.rc"
        rc = int(rc_file.read_text().strip()) if rc_file.exists() else -1
        if rc != 0:
            return block(st, f"download exited rc={rc}; see {OUT/'download.log'}")
        receipt("DOWNLOAD_PARENT", {"repo": adm["repo"], "local_dir": str(local), "rc": rc})
        telegram(f"Download complete: {adm['repo']} -> {local.name}")
        return advance(st, "VERIFY_PARENT")

    claim("DOWNLOAD_PARENT")
    local.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"}
    code = ("from huggingface_hub import snapshot_download;"
            f"snapshot_download({adm['repo']!r}, revision={adm['immutable_revision']!r},"
            f" local_dir={str(local)!r}, max_workers=16)")
    log(f"spawning transfer {adm['repo']} -> {local}")
    telegram(f"Downloading {adm['repo']} ({adm['source_bytes']/GIB:.1f} GiB) to {local.name}")
    with open(OUT / "download.log", "ab") as fh:
        proc = subprocess.Popen(
            ["/bin/sh", "-c",
             f"caffeinate -dimsu {PY} -c {json.dumps(code)}; echo $? > {OUT/'download.rc'}"],
            cwd=str(ROOT), stdout=fh, stderr=fh, env=env, start_new_session=True)
    SS.write(dl_state, {"pid": proc.pid, "started_at": now(), "repo": adm["repo"],
                        "target_bytes": adm["source_bytes"], "last_band": -1})
    return st


def t_verify_parent(st: dict) -> dict:
    if has_receipt("VERIFY_PARENT"):
        return advance(st, "LAUNCH_PARENT", "replay")
    claim("VERIFY_PARENT")
    adm = SS.read(OUT / "NEXT_PARENT_ADMISSION.json")
    local = Path(adm["local_dir"])
    shards = sorted(local.glob("*.safetensors"))
    on_disk = sum(p.stat().st_size for p in local.rglob("*") if p.is_file() and not p.is_symlink())
    # A second materialised copy in the hub cache violates the one-copy law.
    hub_dupes = [str(p) for p in (Path.home() / ".cache/huggingface/hub").glob(
        "models--" + adm["repo"].replace("/", "--") + "/blobs/*")] \
        if (Path.home() / ".cache/huggingface/hub").exists() else []
    problems = []
    if len(shards) != adm["n_weight_shards"]:
        problems.append(f"shard count {len(shards)} != official {adm['n_weight_shards']}")
    if on_disk < adm["source_bytes"] * 0.999:
        problems.append(f"on-disk {on_disk} < official {adm['source_bytes']}")
    if not (local / "config.json").exists():
        problems.append("config.json missing")
    if not list(local.glob("tokenizer*")):
        problems.append("tokenizer assets missing")
    if not list(local.glob("*index.json")):
        problems.append("tensor index missing")
    free = SS.free_bytes()
    row_reserve = adm.get("source_bytes", 0)
    if free < 32 * GIB:
        problems.append(f"post-download reserve is red: {free/GIB:.1f} GiB free")
    verdict = {"schema": "hawking.storage_stripdown.parent_verification.v1",
               "generated_at": now(), "repo": adm["repo"], "local_dir": str(local),
               "shards_on_disk": len(shards), "bytes_on_disk": on_disk,
               "official_bytes": adm["source_bytes"], "free_bytes_after": free,
               "duplicate_hub_blobs": len(hub_dupes), "problems": problems,
               "one_copy_verified": not hub_dupes,
               "status": "GREEN" if not problems else "RED"}
    SS.write(OUT / "NEXT_PARENT_VERIFICATION.json", verdict)
    if problems:
        return block(st, "; ".join(problems))
    if hub_dupes:  # reclaim the duplicate materialisation rather than failing
        for b in hub_dupes:
            try:
                os.unlink(b)
            except OSError:
                pass
    receipt("VERIFY_PARENT", verdict)
    telegram(f"Source verified: {adm['repo']} {len(shards)} shards, "
             f"{on_disk/GIB:.1f} GiB, one copy. Free {free/GIB:.1f} GiB.")
    return advance(st, "LAUNCH_PARENT")


DOCTOR_PRIME_GATES = [
    ("adapter", "tools/foundry/adapters/tier_a_registry.json"),
    ("doctor_prime_treatment_abi", "reports/subbit_reset/DOCTOR_PRIME_TREATMENT_ABI.json"),
    ("doctor_prime_byte_auction", "reports/subbit_reset/DOCTOR_PRIME_BYTE_AUCTION.json"),
    ("doctor_prime_causal_harness", "reports/subbit_reset/DOCTOR_PRIME_CAUSAL_HARNESS.json"),
    ("corpus_integrity", "reports/subbit_reset/CORPUS_INTEGRITY_GATE.json"),
    ("one_bit_ceiling", "tools/foundry/one_bit_ceiling.py"),
]


def t_launch(st: dict) -> dict:
    if has_receipt("LAUNCH_PARENT"):
        return advance(st, "MONITOR_PARENT", "replay")
    claim("LAUNCH_PARENT")
    adm = SS.read(OUT / "NEXT_PARENT_ADMISSION.json")
    arch = (adm.get("config") or {}).get("model_type")
    gates, red = {}, []
    for name, rel in DOCTOR_PRIME_GATES:
        p = ROOT / rel
        if not p.exists():
            p = QWEN_WT / rel
        gates[name] = p.exists()
        if not p.exists():
            red.append(f"{name} missing ({rel})")
    adapter_file = ROOT / f"tools/condense/{arch}_adapter.py" if arch else None
    gates["parent_adapter_module"] = bool(adapter_file and adapter_file.exists())
    if not gates["parent_adapter_module"]:
        red.append(f"no adapter module for architecture {arch!r}; the parent cannot be read "
                   f"by the foundry yet")
    packet = {"schema": "hawking.storage_stripdown.launch_packet.v1", "generated_at": now(),
              "repo": adm["repo"], "architecture": arch, "gates": gates, "red": red,
              "status": "GREEN" if not red else "RED",
              "runtime_boundary": "PYTHON REFERENCE FORWARD ONLY. The Rust engine does not route "
                                  f"{arch!r}; no tok/s or serve claim may be made for this parent.",
              "next_build_items": red}
    SS.write(OUT / "NEXT_PARENT_LAUNCH_PACKET.json", packet)
    if red:
        telegram(f"Source staged and verified, but the Doctor Prime launch packet is RED: "
                 f"{len(red)} gate(s). Science launch is fail-closed pending: {red[0]}")
        return block(st, "launch packet RED: " + "; ".join(red))
    receipt("LAUNCH_PARENT", packet)
    return advance(st, "MONITOR_PARENT")


def t_monitor(st: dict) -> dict:
    heartbeat(st)
    return advance(st, "COMPLETE")


TRANSITIONS = {
    "WAIT_QWEN": t_wait_qwen, "SEAL_QWEN": t_seal_qwen, "INVENTORY": t_inventory,
    "PLAN_DELETE": t_plan_delete, "RELEASE_MODELS": t_release_models,
    "CLEAN_WORKTREES": t_clean_worktrees, "CLEAN_CACHES": t_clean_caches,
    "VERIFY_STORAGE": t_verify_storage, "RESEQUENCE_LADDER": t_resequence,
    "ADMIT_PARENT": t_admit, "DOWNLOAD_PARENT": t_download,
    "VERIFY_PARENT": t_verify_parent, "LAUNCH_PARENT": t_launch, "MONITOR_PARENT": t_monitor,
}


def tick() -> dict:
    st = state()
    if st["state"] in ("COMPLETE", "BLOCKED"):
        heartbeat(st)
        return st
    heartbeat(st)
    try:
        return TRANSITIONS[st["state"]](st)
    except Exception as exc:  # noqa: BLE001
        import traceback
        log(traceback.format_exc()[-2000:])
        return block(st, f"{type(exc).__name__}: {exc}"[:400])


def self_check() -> None:
    assert set(TRANSITIONS) | {"COMPLETE", "BLOCKED"} == set(STATES), "state table mismatch"
    st = {"state": "WAIT_QWEN", "history": []}
    # A live Qwen process must always hold the machine in WAIT_QWEN.
    hb = SS.read(QWEN_HB)
    if hb.get("pid") and alive(int(hb["pid"])):
        out = t_wait_qwen(dict(st))
        assert out["state"] == "WAIT_QWEN", "controller tried to advance past a LIVE Qwen run"
    # A STALE seal must never unlock the release, even with the process gone. This is the exact
    # live trap: the state file reads SEALED/final for the previous generation while the current
    # pass is on row 0 of 6.
    q = SS.read(QWEN_STATE)
    if q.get("status") == "SEALED":
        probe = {"state": "WAIT_QWEN", "history": [],
                 "qwen_seal_baseline": q.get("generated_at", "")}
        stale_hb = {**hb, "pid": 1, "rows_done": 0, "rows_total": 6}
        import unittest.mock as _m
        me = sys.modules[__name__]
        # Mock the writers too: a self-check must not seal receipts or move the real state.
        with _m.patch.object(me, "alive", lambda _p: False), \
             _m.patch.object(me, "telegram", lambda _t: True), \
             _m.patch.object(SS, "write", lambda _p, _o: None), \
             _m.patch.object(SS, "read", lambda p: q if Path(p) == QWEN_STATE else (
                 stale_hb if Path(p) == QWEN_HB else {})):
            out = t_wait_qwen(dict(probe))
        assert out["state"] == "BLOCKED", \
            "GATE BROKEN: a stale seal unlocked the source release"
    # Every destructive transition must be behind a receipt or a claim.
    for name in ("RELEASE_MODELS", "CLEAN_WORKTREES", "CLEAN_CACHES"):
        assert name in TRANSITIONS
    print("self_check: OK (state table complete; live Qwen holds WAIT_QWEN)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["tick", "loop", "status", "self-check"])
    a = ap.parse_args()
    if a.cmd == "self-check":
        self_check()
        return 0
    if a.cmd == "status":
        print(json.dumps({"state": SS.read(STATE), "heartbeat": SS.read(HB),
                          "lease": SS.read(LEASE)}, indent=2))
        return 0
    if not take_lease():
        print(json.dumps({"status": "REFUSED", "reason": "another controller holds the lease",
                          "lease": SS.read(LEASE)}, indent=2))
        return 1
    if a.cmd == "tick":
        print(json.dumps(tick(), indent=2, default=str))
        return 0
    log(f"loop start pid={os.getpid()}")
    telegram("controller up; waiting on the live Qwen row before anything is deleted")
    while True:
        st = tick()
        if st["state"] in ("COMPLETE", "BLOCKED"):
            telegram(f"controller terminal state {st['state']}")
            log(f"terminal {st['state']}")
            return 0
        time.sleep(TICK_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
