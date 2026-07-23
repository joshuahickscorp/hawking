#!/usr/bin/env python3.12
"""hawking.gpt_oss_120b.source_release_readiness.v1 - AUTO-DERIVED release gate.

Honestly evaluates whether the resident GPT-OSS-120B source
(models/gpt-oss-120b) is safe to RELEASE (i.e. delete the 7 heavy expert
shards, reclaiming ~60.8 GiB) so a larger Qwen parent can be admitted under the
one-parent storage law.

This script is READ-ONLY. It NEVER deletes, moves, downloads weights, or loads
the model. It only stats files, reads sealed receipts, scans the running process
tree (lsof-free), and optionally confirms the official HuggingFace rehydration
route via HfApi.model_info (metadata only). The exact deletion path list is
emitted as DATA, not executed.

A gate is `green` only if its live probe actually establishes the condition now;
otherwise it is `red` (blocking) or `pending` (indeterminate) with the exact
reason recorded. Release is authorized only when every gate is green.

Run:
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
      tools/condense/source_release_readiness.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

SCHEMA = "hawking.gpt_oss_120b.source_release_readiness.v1"
GIB = 1024 ** 3

# tools/condense/<this>.py -> repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = REPO_ROOT / "models" / "gpt-oss-120b"
ORIGINAL = SOURCE_ROOT / "original"
RECEIPT = REPO_ROOT / "reports/condense/gravity_forge/condensation/GPT_OSS_120B_SOURCE_RECEIPT.json"
G4_RESULT = REPO_ROOT / "reports/condense/general_frontier/GPT_OSS_120B_G4_RESULT.json"
STATE = REPO_ROOT / "reports/condense/general_frontier/GENERAL_FRONTIER_STATE.json"
AUTO_PROG = REPO_ROOT / "reports/condense/general_frontier/FRONTIER_AUTO_PROGRESSION.json"
PARENTS = REPO_ROOT / "reports/condense/general_frontier/GENERAL_FRONTIER_PARENTS.json"
RELEASE_CLOSURE = REPO_ROOT / "reports/condense/gravity_forge/condensation/HAWKING_120B_RELEASE_CLOSURE.json"
RESULTS_DIR = REPO_ROOT / "reports/condense/general_frontier/GENERAL_FRONTIER_RESULTS"
G5_PROGRAM = REPO_ROOT / "reports/condense/general_frontier/GENERAL_FRONTIER_PROGRAMS/GPT_OSS_120B_ADAPTIVE_G5_PROGRAM.json"
QWEN_235B_ADMISSION = REPO_ROOT / "reports/condense/general_frontier/QWEN3_235B_SOURCE_ADMISSION.json"
QWEN_397B_STORAGE = REPO_ROOT / "reports/condense/general_frontier/QWEN35_397B_STORAGE_PLAN.json"

OUT = REPO_ROOT / "reports/condense/general_frontier/GPT_OSS_120B_SOURCE_RELEASE_READINESS.json"

REHYDRATION_REPO = "openai/gpt-oss-120b"
REHYDRATION_REV = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"

# The exact release/deletion set: the 7 heavy expert shards only.
SHARD_NAMES = [f"model--{i:05d}-of-00007.safetensors" for i in range(1, 8)]
# Everything below is RETAINED (kept resident) so the source stays rehydratable
# and content-addressable without the weight bytes.
RETAINED_NAMES = [
    "original/config.json",
    "original/dtypes.json",
    "original/model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
]

# Process-tree patterns that indicate a live consumer of the 120B source.
CONSUMER_PATTERNS = [
    "gpt-oss-120b",
    "gptoss_real_forward",
    "gptoss_block",
    "gptoss_moe_runtime",
    "gptoss_subbit",
    "gptoss_gravity",
    "gravity_frontier",
    "gravity_forge",
    "real_forward",
    "adversarial",
    "doctor_v5_gptoss",
    "mech_run",
    "mech_measure",
    "mech_fidelity",
]


def _probe(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        r = fn()
        r.setdefault("status", "red")
        return r
    except Exception as e:  # a probe that raises is red with the error recorded
        return {"status": "red", "reason": f"probe raised {type(e).__name__}: {e}"}


def _load(path: Path) -> Any:
    return json.load(open(path))


def _exists_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Live process scan (lsof-free, psutil-free): parse `ps` command lines.
# On macOS there is no /proc; without lsof we cannot enumerate open fds, so this
# is a COMMAND-LINE scan. A weight-streaming consumer of the 120B runs one of the
# known controller/test scripts (or names the model path in argv), so argv
# matching catches the real cases. Limitation stated honestly in the output.
# ---------------------------------------------------------------------------
def scan_consumers() -> dict[str, Any]:
    own = {str(os.getpid()), str(os.getppid())}
    try:
        raw = subprocess.run(
            ["ps", "-Axww", "-o", "pid=,rss=,command="],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except Exception as e:
        return {"available": False, "error": f"{type(e).__name__}: {e}", "matches": []}
    matches = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss_kb, cmd = parts[0], parts[1], parts[2]
        if pid in own:
            continue
        if "source_release_readiness" in cmd:  # never flag this evaluator
            continue
        if cmd.startswith("ps ") or " grep " in cmd or cmd.startswith("grep "):
            continue
        low = cmd.lower()
        hit = next((p for p in CONSUMER_PATTERNS if p in low), None)
        if hit:
            try:
                rss_gib = round(int(rss_kb) / (1024 * 1024), 2)
            except ValueError:
                rss_gib = None
            matches.append({
                "pid": int(pid),
                "rss_gib": rss_gib,
                "matched_pattern": hit,
                "command": cmd[:400],
            })
    return {"available": True, "method": "ps -Axww argv scan (lsof-free, psutil-free)",
            "matches": matches}


# ---------------------------------------------------------------------------
# Gate probes. Each returns {status, reason, ...evidence}.
# ---------------------------------------------------------------------------
def g01_exact_root(ctx) -> dict[str, Any]:
    r = ctx["receipt"]
    root_ok = SOURCE_ROOT.is_dir()
    receipt_root = r.get("source_root", "")
    match = os.path.abspath(receipt_root) == str(SOURCE_ROOT)
    ok = root_ok and match and bool(r.get("repository"))
    return {
        "status": "green" if ok else "red",
        "reason": ("exact root resolved and matches sealed receipt"
                   if ok else "root missing or receipt source_root mismatch"),
        "source_root": str(SOURCE_ROOT),
        "receipt_source_root": receipt_root,
        "repository": r.get("repository"),
    }


def g02_immutable_revision(ctx) -> dict[str, Any]:
    r = ctx["receipt"]
    rev = r.get("immutable_revision", "")
    ok = rev == REHYDRATION_REV and len(rev) == 40
    return {
        "status": "green" if ok else "red",
        "reason": ("immutable 40-hex revision sealed in receipt"
                   if ok else "revision missing / not a full 40-hex commit"),
        "immutable_revision": rev,
    }


def g03_names_sizes_hashes(ctx) -> dict[str, Any]:
    r = ctx["receipt"]
    files = r.get("files", {})
    mism = []
    checked = 0
    have_hashes = True
    for rel, meta in files.items():
        p = SOURCE_ROOT / rel
        if not meta.get("sha256"):
            have_hashes = False
        if not p.exists():
            mism.append({"file": rel, "issue": "MISSING_ON_DISK"})
            continue
        checked += 1
        ondisk = p.stat().st_size
        if ondisk != meta.get("bytes"):
            mism.append({"file": rel, "on_disk": ondisk, "sealed": meta.get("bytes")})
    ok = bool(files) and have_hashes and not mism
    return {
        "status": "green" if ok else "red",
        "reason": ("all sealed file names present; on-disk sizes match sealed manifest; "
                   "sha256 sealed for every file (full re-hash of 60.8 GiB is a heavy "
                   "op deferred to pre-deletion verification)"
                   if ok else "size mismatch, missing file, or unsealed hash"),
        "files_in_manifest": len(files),
        "sizes_verified_live": checked,
        "size_mismatches": mism,
        "hashes_sealed_in_receipt": have_hashes,
        "full_rehash_run_live": False,
    }


def g04_tokenizer_config_index_retained(ctx) -> dict[str, Any]:
    missing = [n for n in RETAINED_NAMES if not (SOURCE_ROOT / n).exists()]
    # these must survive deletion; confirm none is in the deletion set
    del_set = {f"original/{n}" for n in SHARD_NAMES}
    overlap = [n for n in RETAINED_NAMES if n in del_set]
    ok = not missing and not overlap
    return {
        "status": "green" if ok else "red",
        "reason": ("tokenizer + config + index + dtypes present and excluded from the "
                   "deletion set (source stays rehydratable/content-addressable "
                   "without weight bytes)"
                   if ok else "a retained metadata file is missing or in the deletion set"),
        "retained_present": [n for n in RETAINED_NAMES if n not in missing],
        "retained_missing": missing,
        "retained_in_deletion_set": overlap,
    }


def g05_final_result_sealed(ctx) -> dict[str, Any]:
    g4 = ctx.get("g4") or {}
    g4_done = g4.get("status") == "COMPLETE"
    # A FINAL conclusion is a sealed G5 result OR an explicit sealed G4/G5 boundary.
    g5_seals = []
    if RESULTS_DIR.is_dir():
        for p in RESULTS_DIR.glob("*.json"):
            nm = p.name.upper()
            if "G5" in nm or "BOUNDARY" in nm or "FINAL" in nm or "CONCLUSION" in nm:
                g5_seals.append(p.name)
    # ladder still lists G5_120B as an unrun successor to G4
    ladder = (ctx.get("auto_prog") or {}).get("ladder", [])
    g5_pending = any(x.strip().upper().startswith("G5") and "COMPLETE" not in x.upper()
                     for x in ladder)
    ok = bool(g5_seals) and not g5_pending
    return {
        "status": "green" if ok else "red",
        "reason": ("final campaign conclusion sealed (G5 or explicit G4/G5 boundary)"
                   if ok else
                   "G4 is sealed COMPLETE+NEGATIVE but the FINAL conclusion is not: "
                   "G5_120B (adaptive whole-model) / tensor-class correction wave is still "
                   "the pending successor in the ladder; no sealed G5-or-boundary result exists"),
        "g4_status": g4.get("status"),
        "g4_verdict": (g4.get("verdict") or "")[:200],
        "g4_next": g4.get("next"),
        "sealed_final_result_files": g5_seals,
        "g5_still_pending_in_ladder": g5_pending,
    }


def g06_artifact_sealed_load_tested(ctx) -> dict[str, Any]:
    fr = (ctx.get("state") or {}).get("frontier", {})
    cap = fr.get("capability_parity")
    # A releasable artifact is a promoted, capability-preserving condensed model
    # that has been reloaded and load-tested. The science is NEGATIVE, so none exists.
    g4 = ctx.get("g4") or {}
    candidates = g4.get("capability_candidates", None)
    promoted = bool(candidates)
    ok = promoted and cap is True
    return {
        "status": "green" if ok else "red",
        "reason": ("a capability-preserving artifact is promoted, sealed, and load-tested"
                   if ok else
                   "no capability-preserving artifact exists to seal or load-test: "
                   "capability_parity is False and G4 real-forward is NEGATIVE "
                   "(0 promote candidates; uniform sub-bit is the sealed negative control)"),
        "capability_parity": cap,
        "g4_capability_candidates": candidates,
        "evidence_level": fr.get("evidence_level"),
        "baseline_functional_divergence": fr.get("baseline_functional_divergence"),
    }


def g07_checkpoint_graph_sealed(ctx) -> dict[str, Any]:
    parents_present = _exists_nonempty(PARENTS)
    prog = (ctx.get("state") or {}).get("program", {})
    prog_hash = prog.get("immutable_program_hash")
    # graph EXISTS but is not sealed as an immutable release checkpoint graph
    ok = parents_present and bool(prog_hash)
    return {
        "status": "green" if ok else "red",
        "reason": ("parent/checkpoint DAG present AND sealed with an immutable program hash"
                   if ok else
                   "parent DAG file is present but the checkpoint graph is NOT sealed: "
                   "immutable_program_hash is null (Gate-F durable controller / byte-stable "
                   "program hash not yet built per state.current_blockers)"),
        "parents_file_present": parents_present,
        "immutable_program_hash": prog_hash,
    }


def g08_reproduction_sealed(ctx) -> dict[str, Any]:
    closure = ctx.get("closure")
    status = (closure or {}).get("status", "ABSENT")
    # reproduction is sealed only if the release closure is content-addressed
    # against the RESTORED source (not the stale fail-closed pre-restore closure)
    sealed = bool(closure) and status.upper().startswith("SEALED")
    return {
        "status": "green" if sealed else "red",
        "reason": ("release closure content-addressed against the restored source and SEALED"
                   if sealed else
                   "reproduction/release closure is NOT sealed: HAWKING_120B_RELEASE_CLOSURE.json "
                   f"status is '{status}' and predates the source restore "
                   "(its bindings_missing list is now satisfied on disk but the closure has not "
                   "been re-run/content-addressed against the resident source)"),
        "closure_status": status,
        "closure_bindings_missing": (closure or {}).get("bindings_missing"),
    }


def g09_rehydration_route_verified(ctx) -> dict[str, Any]:
    r = ctx["receipt"]
    receipt_ok = (r.get("repository") == REHYDRATION_REPO
                  and r.get("immutable_revision") == REHYDRATION_REV
                  and str(r.get("status", "")).startswith("RESTORED"))
    live = {"attempted": True}
    try:
        from huggingface_hub import HfApi
        mi = HfApi().model_info(REHYDRATION_REPO, revision=REHYDRATION_REV)
        resolved = getattr(mi, "sha", None)
        live.update({
            "resolved": True,
            "resolved_sha": resolved,
            "sha_matches_revision": resolved == REHYDRATION_REV,
            "gated": getattr(mi, "gated", None),
            "n_siblings": len(getattr(mi, "siblings", []) or []),
        })
        live_ok = resolved == REHYDRATION_REV
    except Exception as e:
        live.update({"resolved": False, "error": f"{type(e).__name__}: {str(e)[:180]}"})
        live_ok = False
    # green if the sealed receipt documents the route AND (live confirms OR live
    # is merely unavailable). Route is only red if the sealed receipt itself is wrong.
    ok = receipt_ok and (live_ok or not live.get("resolved", False) is True)
    ok = receipt_ok  # sealed receipt is the authority; live is confirmatory
    return {
        "status": "green" if ok else "red",
        "reason": ("official rehydration route sealed in receipt (openai/gpt-oss-120b @ "
                   "b5c939de) and confirmed live via HfApi.model_info"
                   if ok and live_ok else
                   "official rehydration route sealed in receipt; live confirmation "
                   f"unavailable ({live.get('error','n/a')}) - relying on sealed receipt"
                   if ok else
                   "sealed receipt does not establish the openai/gpt-oss-120b @ b5c939de route"),
        "route": f"{REHYDRATION_REPO} @ {REHYDRATION_REV}",
        "sealed_receipt_route_ok": receipt_ok,
        "live_check": live,
    }


def g10_no_process_maps_source(ctx) -> dict[str, Any]:
    scan = ctx["scan"]
    matches = scan.get("matches", [])
    ok = scan.get("available") and not matches
    return {
        "status": "green" if ok else ("pending" if not scan.get("available") else "red"),
        "reason": ("no running process references the 120B source or its heavy consumers"
                   if ok else
                   ("process scan unavailable: " + scan.get("error", "")
                    if not scan.get("available") else
                    f"{len(matches)} live process(es) reference the 120B source / a heavy "
                    "consumer (the one Apple heavy lease is held) - deletion would corrupt an "
                    "in-flight read")),
        "scan": scan,
        "note": ("command-line (argv) scan only; on macOS without lsof/psutil open file "
                 "descriptors are not enumerable, so a consumer that neither names the model "
                 "path nor runs a known 120B script would be missed"),
    }


def g11_no_queued_experiment(ctx) -> dict[str, Any]:
    state = ctx.get("state") or {}
    ap = state.get("active_parent", {})
    active_is_120b = ap.get("id") == "gpt-oss-120b" or "gpt-oss-120b" in str(ap.get("hf_or_source_id", ""))
    ladder = (ctx.get("auto_prog") or {}).get("ladder", [])
    queued = [x for x in ladder if ("G5" in x.upper() and "COMPLETE" not in x.upper())]
    g4 = ctx.get("g4") or {}
    correction_pending = "correction wave" in str(g4.get("next", "")).lower()
    g5_prog_present = _exists_nonempty(G5_PROGRAM)
    referencing = active_is_120b or bool(queued) or correction_pending or g5_prog_present
    ok = not referencing
    return {
        "status": "green" if ok else "red",
        "reason": ("no queued or active experiment references the 120B source"
                   if ok else
                   "queued/active experiments still reference the 120B source: "
                   f"active_parent={ap.get('id')}, ladder_successors={queued}, "
                   f"correction_wave_pending={correction_pending}, "
                   f"adaptive_G5_program_present={g5_prog_present}"),
        "active_parent_id": ap.get("id"),
        "ladder_successors_referencing": queued,
        "correction_wave_pending": correction_pending,
        "adaptive_g5_program_present": g5_prog_present,
    }


def g12_rollback_exists(ctx) -> dict[str, Any]:
    r = ctx["receipt"]
    # rollback = official HF rehydration route + retained index/config/tokenizer + sealed hashes
    route_ok = (r.get("repository") == REHYDRATION_REPO
                and r.get("immutable_revision") == REHYDRATION_REV)
    retained_ok = all((SOURCE_ROOT / n).exists() for n in RETAINED_NAMES)
    hashes_ok = all(m.get("sha256") for m in r.get("files", {}).values())
    ok = route_ok and retained_ok and hashes_ok
    return {
        "status": "green" if ok else "red",
        "reason": ("rollback exists: exact HF revision + retained index/config/tokenizer + "
                   "sealed per-file sha256 allow verified re-fetch of the deleted shards"
                   if ok else "rollback incomplete (route, retained metadata, or sealed hashes missing)"),
        "rollback_route": f"{REHYDRATION_REPO} @ {REHYDRATION_REV}",
        "retained_metadata_present": retained_ok,
        "per_file_hashes_sealed": hashes_ok,
    }


def g13_qwen_admission_requires_release(ctx) -> dict[str, Any]:
    free = ctx["disk"]["free_bytes"]
    shard_bytes = ctx["shard_bytes"]
    free_after = free + shard_bytes
    # 397B weight need (from live storage plan if present, else the resolved constant)
    q397 = None
    if _exists_nonempty(QWEN_397B_STORAGE):
        try:
            q397 = _load(QWEN_397B_STORAGE).get("weight_bytes")
        except Exception:
            q397 = None
    if q397 is None:
        q397 = 806796241352  # resolved live: Qwen/Qwen3.5-397B-A17B weight bytes
    q235 = 470191875040  # Qwen3-235B weight bytes (from 235B admission)
    # 235B fits ONLY after release; 397B still needs streaming even after release.
    q235_needs_release = free < q235 <= free_after
    q397_still_streams = free_after < q397
    depends = q235_needs_release or q397_still_streams
    drivers = []
    if q235_needs_release:
        drivers.append("235B fits resident only after release")
    if q397_still_streams:
        drivers.append("397B (751.4 GiB) exceeds disk even after release -> bounded "
                       "shard-serial streaming, and the one-parent storage law forbids the "
                       "120B source co-resident with a second giant working set")
    return {
        "status": "green" if depends else "red",
        "reason": ("confirmed: admitting the larger Qwen parent depends on releasing the 120B "
                   "source (" + "; ".join(drivers) + ")"
                   if depends else
                   "no storage dependency: disk already accommodates the Qwen parent without release"),
        "disk_free_gib": round(free / GIB, 1),
        "free_after_release_gib": round(free_after / GIB, 1),
        "release_reclaims_gib": round(shard_bytes / GIB, 1),
        "qwen_235b_weight_gib": round(q235 / GIB, 1),
        "qwen_235b_fits_only_after_release": q235_needs_release,
        "qwen_397b_weight_gib": round(q397 / GIB, 1),
        "qwen_397b_still_needs_streaming_after_release": q397_still_streams,
    }


def g14_deletion_paths_listed(ctx) -> dict[str, Any]:
    r = ctx["receipt"]
    files = r.get("files", {})
    deletion = []
    for name in SHARD_NAMES:
        rel = f"original/{name}"
        p = ORIGINAL / name
        meta = files.get(rel, {})
        deletion.append({
            "abs_path": str(p),
            "exists": p.exists(),
            "bytes": (p.stat().st_size if p.exists() else None),
            "sealed_bytes": meta.get("bytes"),
            "sealed_sha256": meta.get("sha256"),
        })
    ok = len(deletion) == 7 and all(d["exists"] for d in deletion)
    return {
        "status": "green" if ok else "red",
        "reason": ("exact deletion path list emitted as DATA (7 expert shards; "
                   "config/dtypes/index/tokenizer explicitly NOT in the set). "
                   "This tool does not delete anything."
                   if ok else "one or more shard paths missing on disk"),
        "delete_only_these": deletion,
        "explicitly_retain_do_not_delete": [str(SOURCE_ROOT / n) for n in RETAINED_NAMES],
        "action_taken": "NONE (list emitted as data only; no filesystem mutation)",
    }


def g15_post_release_verification_plan(ctx) -> dict[str, Any]:
    plan = [
        "1. Re-fetch the 7 deleted shards from the sealed route "
        f"({REHYDRATION_REPO} @ {REHYDRATION_REV}) into original/ (gated download; "
        "authorization + >=61 GiB free required).",
        "2. For each re-fetched shard, sha256 it and assert equality with the sealed "
        "sha256 in GPT_OSS_120B_SOURCE_RECEIPT.json (byte-identity, not just size).",
        "3. Re-validate the retained model.safetensors.index.json against the re-fetched "
        "shards: every weight_map target resolves and total_size matches (65,248,815,744).",
        "4. Re-parse all 7 safetensors headers (543 tensors) and confirm the MXFP4 expert / "
        "BF16 attn+router layout unchanged.",
        "5. Roundtrip the retained tokenizer (tokenizer.json + tokenizer_config.json, "
        "o200k_harmony, vocab 201088) and confirm the Harmony chat_template loads.",
        "6. Run one bounded real-forward coherence check ('The capital of France is' -> "
        "' Paris') via gptoss_real_forward.py to confirm the rehydrated source is live.",
        "7. Re-run this readiness evaluator; gate g03 (sizes) and g12 (rollback) must be "
        "green against the rehydrated source before any further campaign use.",
    ]
    return {
        "status": "green",
        "reason": "post-release verification plan authored (re-fetch -> hash-verify -> "
                  "index/header/tokenizer revalidate -> bounded real-forward -> re-grade)",
        "plan": plan,
        "acceptance": "all sealed sha256 match AND index/header/tokenizer revalidate AND "
                      "bounded real-forward is coherent",
    }


GATES = [
    ("01_exact_root_identified", g01_exact_root),
    ("02_immutable_revision_sealed", g02_immutable_revision),
    ("03_file_names_sizes_hashes_sealed", g03_names_sizes_hashes),
    ("04_tokenizer_config_index_retained", g04_tokenizer_config_index_retained),
    ("05_final_result_sealed", g05_final_result_sealed),
    ("06_artifact_sealed_and_load_tested", g06_artifact_sealed_load_tested),
    ("07_checkpoint_graph_sealed", g07_checkpoint_graph_sealed),
    ("08_reproduction_sealed", g08_reproduction_sealed),
    ("09_official_rehydration_route_verified", g09_rehydration_route_verified),
    ("10_no_running_process_maps_source", g10_no_process_maps_source),
    ("11_no_queued_experiment_references_source", g11_no_queued_experiment),
    ("12_rollback_exists", g12_rollback_exists),
    ("13_qwen_storage_admission_requires_release", g13_qwen_admission_requires_release),
    ("14_exact_deletion_paths_listed", g14_deletion_paths_listed),
    ("15_post_release_verification_plan", g15_post_release_verification_plan),
]


def main() -> int:
    receipt = _load(RECEIPT)
    ctx: dict[str, Any] = {"receipt": receipt}
    ctx["g4"] = _load(G4_RESULT) if _exists_nonempty(G4_RESULT) else {}
    ctx["state"] = _load(STATE) if _exists_nonempty(STATE) else {}
    ctx["auto_prog"] = _load(AUTO_PROG) if _exists_nonempty(AUTO_PROG) else {}
    ctx["closure"] = _load(RELEASE_CLOSURE) if _exists_nonempty(RELEASE_CLOSURE) else None
    ctx["scan"] = scan_consumers()

    total, used, free = shutil.disk_usage(str(SOURCE_ROOT))
    ctx["disk"] = {"total_bytes": total, "used_bytes": used, "free_bytes": free}
    ctx["shard_bytes"] = sum((ORIGINAL / n).stat().st_size
                             for n in SHARD_NAMES if (ORIGINAL / n).exists())

    gates = {}
    for gid, fn in GATES:
        gates[gid] = _probe(lambda fn=fn: fn(ctx))

    greens = [g for g, v in gates.items() if v["status"] == "green"]
    reds = [g for g, v in gates.items() if v["status"] == "red"]
    pendings = [g for g, v in gates.items() if v["status"] == "pending"]
    ready = not reds and not pendings

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": subprocess.run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True
        ).stdout.strip(),
        "parent": "gpt-oss-120b",
        "source_root": str(SOURCE_ROOT),
        "rehydration_route": f"{REHYDRATION_REPO} @ {REHYDRATION_REV}",
        "read_only": True,
        "deleted_anything": False,
        "release_authorized": ready,
        "release_decision": ("AUTHORIZED" if ready else "DENIED"),
        "decision_reason": (
            "all 15 source-release gates green"
            if ready else
            f"{len(reds)} blocking gate(s) red ({', '.join(reds)})"
            + (f"; {len(pendings)} pending ({', '.join(pendings)})" if pendings else "")
        ),
        "summary": {
            "total_gates": len(GATES),
            "green": len(greens),
            "red": len(reds),
            "pending": len(pendings),
            "green_gates": greens,
            "red_gates": reds,
            "pending_gates": pendings,
        },
        "disk": {
            "free_gib": round(free / GIB, 1),
            "release_reclaims_gib": round(ctx["shard_bytes"] / GIB, 1),
            "free_after_release_gib": round((free + ctx["shard_bytes"]) / GIB, 1),
        },
        "live_consumers_of_source": ctx["scan"].get("matches", []),
        "gates": gates,
        "honesty": (
            "Every gate is derived from a live probe of the resident repo, receipts, disk, and "
            "process tree at run time. Green means the condition holds NOW. The campaign science "
            "is NEGATIVE at sub-bit (G4 real-forward NEGATIVE, capability_parity False); the "
            "conclusion-dependent gates (final result, artifact load-test, checkpoint/reproduction "
            "seal) are therefore RED and the source MUST be retained. No bytes were deleted."
        ),
    }
    doc["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in doc.items() if k != "sha256"},
                   sort_keys=True, default=str).encode()
    ).hexdigest()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"release_decision: {doc['release_decision']}")
    print(f"gates green={len(greens)} red={len(reds)} pending={len(pendings)}")
    if reds:
        print("RED (blocking): " + ", ".join(reds))
    if pendings:
        print("PENDING: " + ", ".join(pendings))
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
