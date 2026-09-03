#!/usr/bin/env python3
"""Retire the Ramanujan campaign by indexing it.

Read-only over ``research/ramanujan/``. Writes ``receipts/headless/RAMANUJAN_DISBAND.json``.
Does not delete, restore, or rewrite anything under ``research/ramanujan/``. The campaign
is retired; its evidence is not. This script makes that second fact durable.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

REPO = Path(__file__).resolve().parents[2]
RAMANUJAN = REPO / "ramanujan"
RECEIPT_PATH = REPO / "receipts" / "headless" / "RAMANUJAN_DISBAND.json"
SCHEMA = "hawking.headless.ramanujan_disband.v1"

# Historical logical paths still named in sealed receipts. layout.py maps them.
LOGICAL = {
    "research/ramanujan/RAMANUJAN_Q0_CLOSURE.json": RAMANUJAN / "records/audits/RAMANUJAN_Q0_CLOSURE.json",
    "research/ramanujan/RAMANUJAN_Q0_EVIDENCE_BUNDLE.json": RAMANUJAN / "records/audits/RAMANUJAN_Q0_EVIDENCE_BUNDLE.json",
    "research/ramanujan/data/corpora/FREEZE_RECEIPT.json": RAMANUJAN / "scaffold/data/corpora/FREEZE_RECEIPT.json",
    "research/ramanujan/data/corpora/GENERATION_RECEIPT.json": RAMANUJAN / "scaffold/data/corpora/GENERATION_RECEIPT.json",
    "research/ramanujan/data/corpora/MEMBERSHIP_MANIFEST.json": RAMANUJAN / "scaffold/data/corpora/MEMBERSHIP_MANIFEST.json",
    "research/ramanujan/RAMANUJAN_DATA_SOURCE_MATRIX.json": RAMANUJAN / "records/intake/RAMANUJAN_DATA_SOURCE_MATRIX.json",
    "research/ramanujan/prover.py": RAMANUJAN / "scaffold/research/prover.py",
    "research/ramanujan/ledger.py": RAMANUJAN / "scaffold/core/ledger.py",
    "research/ramanujan/RAMANUJAN_ENVIRONMENT_LOCK.json": RAMANUJAN / "records/runtime/RAMANUJAN_ENVIRONMENT_LOCK.json",
}

GIT_ONLY = {
    "handoff": "workspace/campaign/evidence/systems/ramanujan/RAMANUJAN_HANDOFF_CONTRACT.json",
    "glm52_fast_intake": "workspace/campaign/evidence/models/glm52/GLM52_RAMANUJAN_FAST_INTAKE.json",
    "verification_lattice": "workspace/campaign/governance/odyssey/domains/verifiers/VERIFICATION_LATTICE.json",
    "substrate_capability": "workspace/campaign/governance/odyssey/program/launch/SUBSTRATE_CAPABILITY.json",
    "support_halo": "workspace/campaign/governance/odyssey/program/evaluation/support_halo_corpus_v0.jsonl",
    "negative_science_register": "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
}


def sh(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=check,
    )


def git_out(args: list[str]) -> str:
    p = sh(["git", *args])
    return (p.stdout or "").strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_show(rel: str) -> bytes | None:
    p = sh(["git", "show", f"HEAD:{rel}"])
    if p.returncode != 0:
        return None
    return p.stdout.encode("utf-8") if isinstance(p.stdout, str) else p.stdout


def git_show_text(rel: str) -> str | None:
    p = sh(["git", "show", f"HEAD:{rel}"])
    if p.returncode != 0:
        return None
    return p.stdout


def git_size(rel: str) -> int | None:
    p = sh(["git", "cat-file", "-s", f"HEAD:{rel}"])
    if p.returncode != 0:
        return None
    try:
        return int(p.stdout.strip())
    except ValueError:
        return None


def classify_path(rel: str) -> str:
    p = rel.lower()
    if "/corpora/" in p and p.endswith((".jsonl",)):
        return "CORPUS"
    if p.endswith("membership_manifest.json"):
        return "CORPUS_MEMBERSHIP"
    if "/container/" in p:
        return "Q0_HARNESS"
    if "/records/audits/" in p or p.endswith("_receipt.json") or "receipt" in Path(p).name.lower():
        return "RECEIPT"
    if "/governance/" in p:
        return "GOVERNANCE"
    if "/tests/" in p or Path(p).name.startswith("test_"):
        return "TEST"
    if "/docs/" in p or Path(p).name in {"readme.md"}:
        return "DOC"
    if "/fixtures/" in p:
        return "FIXTURE"
    if "/guards/" in p:
        return "GUARD"
    if "/train/" in p:
        return "TRAIN"
    if "/research/" in p:
        return "RESEARCH"
    if "/records/" in p:
        return "RECORD"
    if p.endswith(".py"):
        return "CODE"
    return "OTHER"


def inventory() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "bytes": 0})
    tree = hashlib.sha256()
    n_symlinks = 0
    n_files = 0
    total = 0

    for dirpath, dirnames, filenames in os.walk(RAMANUJAN, followlinks=False):
        dirnames[:] = [n for n in dirnames if n != "__pycache__" and n != ".DS_Store"]
        filenames = [n for n in filenames if not n.endswith(".pyc") and n != ".DS_Store"]
        dirnames.sort()
        filenames.sort()
        # Capture directory-symlinks (os.walk will not recurse into them).
        for name in list(dirnames):
            p = Path(dirpath) / name
            try:
                st = p.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode):
                rel = p.relative_to(REPO).as_posix()
                target = os.readlink(p)
                digest = sha256_bytes(target.encode())
                rec = {
                    "path": rel,
                    "kind": "symlink",
                    "target": target,
                    "bytes": st.st_size,
                    "sha256": digest,
                    "class": classify_path(rel),
                }
                files.append(rec)
                by_class[rec["class"]]["count"] += 1
                by_class[rec["class"]]["bytes"] += st.st_size
                tree.update(rel.encode())
                tree.update(digest.encode())
                n_symlinks += 1
                total += st.st_size
                dirnames.remove(name)
        for name in filenames:
            p = Path(dirpath) / name
            try:
                st = p.lstat()
            except OSError:
                continue
            rel = p.relative_to(REPO).as_posix()
            if stat.S_ISLNK(st.st_mode):
                target = os.readlink(p)
                digest = sha256_bytes(target.encode())
                kind = "symlink"
                n_symlinks += 1
                size = st.st_size
            else:
                digest = sha256_file(p)
                kind = "file"
                n_files += 1
                size = st.st_size
            rec = {
                "path": rel,
                "kind": kind,
                "bytes": size,
                "sha256": digest,
                "class": classify_path(rel),
            }
            if kind == "symlink":
                rec["target"] = os.readlink(p)
            files.append(rec)
            by_class[rec["class"]]["count"] += 1
            by_class[rec["class"]]["bytes"] += size
            tree.update(rel.encode())
            tree.update(digest.encode())
            total += size

    files.sort(key=lambda r: r["path"])
    git_paths = git_out(["ls-tree", "-r", "--name-only", "HEAD", "--", "ramanujan"]).splitlines()
    git_paths = [p for p in git_paths if p]
    return {
        "root": "ramanujan",
        "on_disk_regular_files": n_files,
        "on_disk_symlinks": n_symlinks,
        "on_disk_entries": n_files + n_symlinks,
        "total_bytes": total,
        "tree_sha256": tree.hexdigest(),
        "git_ls_tree_count": len(git_paths),
        "git_tree": git_out(["rev-parse", "HEAD:ramanujan"]),
        "by_class": {k: by_class[k] for k in sorted(by_class)},
        "largest": sorted(files, key=lambda r: (-r["bytes"], r["path"]))[:12],
        "files": files,
    }


def porcelain(path: str = "ramanujan") -> list[str]:
    out = git_out(["status", "--porcelain", "--", path])
    return [ln for ln in out.splitlines() if ln]


def probe_failures(inv: dict[str, Any], live: dict[str, Any]) -> list[dict[str, Any]]:
    """Things that actually failed or were absent while building this index."""
    watched: list[dict[str, Any]] = []

    # 1. Training checkpoints named by the training receipt are not in the tree.
    named = [
        "research/ramanujan/train/checkpoints/formalizer.pt",
        "research/ramanujan/train/checkpoints/prover.pt",
        "research/ramanujan/train/checkpoints/repair.pt",
        "research/ramanujan/train/checkpoints/retriever.pt",
        "research/ramanujan/train/checkpoints/value.pt",
        "research/ramanujan/scaffold/train/checkpoints/formalizer.pt",
        "research/ramanujan/scaffold/train/checkpoints/retriever.pt",
    ]
    present = [p for p in named if (REPO / p).exists()]
    watched.append({
        "id": "checkpoints_absent",
        "what": "TRAINING_RECEIPT names five .pt checkpoints; none are in the tree",
        "named": named,
        "present": present,
        "result": "ABSENT" if not present else "UNEXPECTED_PRESENT",
        "meaning": "The small-system weights were never committed. Held-out numbers survive; the models do not.",
    })

    # 2. Handoff contract is git-only; ramanujan.status cannot run here.
    handoff_disk = REPO / "workspace/campaign/evidence/systems/ramanujan/RAMANUJAN_HANDOFF_CONTRACT.json"
    watched.append({
        "id": "handoff_not_on_disk",
        "what": "ramanujan.status reads the handoff contract from workspace/; this sparse checkout does not materialize workspace/",
        "path": str(handoff_disk.relative_to(REPO)),
        "on_disk": handoff_disk.is_file(),
        "in_git": git_show(GIT_ONLY["handoff"]) is not None,
        "result": "STATUS_MODULE_CANNOT_RUN_IN_THIS_SPARSE_TREE",
    })

    # 3. Historical Odyssey lattice path is gone; the bound bytes live at a moved path.
    old_lattice = REPO / "odyssey/verifiers/VERIFICATION_LATTICE.json"
    new_lattice = GIT_ONLY["verification_lattice"]
    blob = git_show(new_lattice)
    bound = live["q0_q6"]["receipt_bindings"].get("odyssey/verifiers/VERIFICATION_LATTICE.json")
    live_hash = sha256_bytes(blob) if blob is not None else None
    watched.append({
        "id": "verification_lattice_moved",
        "what": "Q0-Q6 contracts bind odyssey/verifiers/VERIFICATION_LATTICE.json, which is not a HEAD path",
        "old_path_on_disk": old_lattice.is_file(),
        "git_path_now": new_lattice,
        "bound_sha256": bound,
        "git_object_sha256": live_hash,
        "bytes_still_match_binding": live_hash == bound,
        "result": "PATH_MOVED_BYTES_HELD",
    })

    # 4. restream_guard imports lab.operators, which this sparse tree does not materialize.
    # Do not import ramanujan.* — CPython will write research/ramanujan/__pycache__ even when
    # sys.dont_write_bytecode is set (observed: cpython-314 .pyc from a layout import).
    restream_src = (RAMANUJAN / "scaffold/guards/restream_guard.py").read_text(encoding="utf-8")
    lab_on_disk = (REPO / "lab" / "operators").is_dir()
    watched.append({
        "id": "restream_guard_import",
        "what": "restream_guard.py imports lab.operators.glm52_common; research/lab/operators is not in this sparse checkout",
        "imports_lab_operators": "from lab.operators.glm52_common" in restream_src,
        "lab_operators_on_disk": lab_on_disk,
        "result": "IMPORT_WOULD_FAIL_IN_THIS_SPARSE_TREE" if not lab_on_disk else "OPERATORS_PRESENT_NOT_IMPORTED",
        "meaning": "The fail-closed launcher still exists as source; it is not executable without the operators tree. This probe reads the source rather than importing, so it cannot mint research/ramanujan/__pycache__.",
    })

    # 5. No Noetic tree.
    noetic = git_out(["ls-tree", "-r", "--name-only", "HEAD"])
    noetic_hits = [ln for ln in noetic.splitlines() if "noetic" in ln.lower()]
    watched.append({
        "id": "noetic_absent",
        "what": "search HEAD for a Noetic campaign tree",
        "hits": noetic_hits,
        "result": "NO_PATH_NAMED_NOETIC",
        "meaning": "If Noetic names the doctrine/cognition half of the Hawking split, research/ramanujan/ is that half as a scaffold. It is not a separate directory.",
    })

    # 6. The 31-closed / 9-reopen census is not a named receipt in this tree.
    ns_blob = git_show_text(GIT_ONLY["negative_science_register"])
    ns_counts = None
    ns_classes: dict[str, int] = {}
    if ns_blob:
        try:
            ns = json.loads(ns_blob)
            ns_counts = ns.get("counts")
            for e in ns.get("entries") or []:
                ns_classes[str(e.get("class"))] = ns_classes.get(str(e.get("class")), 0) + 1
        except json.JSONDecodeError:
            pass
    watched.append({
        "id": "thirty_one_closed_not_in_this_tree",
        "what": "operator context cited 31 closed results with 9 reopen conditions already holding; this tree does not contain a receipt with those exact counts",
        "closest_receipt": GIT_ONLY["negative_science_register"],
        "closest_counts": ns_counts,
        "closest_classes": ns_classes,
        "result": "NOT_RE_DERIVED",
        "meaning": "That census is why Ramanujan is being indexed rather than deleted. It is not Ramanujan's own scoreboard.",
    })

    # 7. Membership the small system trained on is not the sealed freeze.
    watched.append({
        "id": "membership_sha_mismatch",
        "what": "TRAINING_RECEIPT.membership_sha256 != FREEZE_RECEIPT.membership_sha256",
        "training_at": live["training"]["at"],
        "training_membership_sha256": live["training"]["membership_sha256"],
        "training_counts": live["training_status"]["membership"]["counts"],
        "freeze_at": live["freeze"]["at"],
        "freeze_membership_sha256": live["freeze"]["membership_sha256"],
        "freeze_counts": live["freeze"]["counts"],
        "delta_train_items": live["freeze"]["counts"]["train"] - live["training_status"]["membership"]["counts"]["train"],
        "result": "HELD_OUT_METRICS_BOUND_TO_SUPERSEDED_MEMBERSHIP",
        "meaning": "Citing retriever MRR or value accuracy as current-freeze evidence is a method error. Re-eval against fec9f85e before reuse.",
    })

    # 8. D4 end-to-end rerun still owed.
    watched.append({
        "id": "d4_rerun_owed",
        "what": "CORPUS_DETERMINISM: D4 content leak and ordering were fixed; end-to-end rerun still owed",
        "receipt": "research/ramanujan/records/intake/RAMANUJAN_CORPUS_DETERMINISM.json",
        "d4_result": live["determinism"]["result"]["D4"],
        "result": "OWED_NOT_CLAIMED",
    })

    # 9. Docker image is not in the repository.
    watched.append({
        "id": "q0_image_not_in_repo",
        "what": "Q0 clean-replay image ramanujan-clean-replay:2ec0166b (~4.64 GiB) is a host Docker image, not a git object",
        "image_id": live["q0_closure"]["image"]["id"],
        "image_size_bytes": live["q0_closure"]["image"]["size_bytes"],
        "in_ramanujan_tree": False,
        "recipe_in_tree": True,
        "result": "RECIPE_KEPT_IMAGE_EXTERNAL",
    })

    # 10. Q0 was once claimed ACHIEVED after its harness was deleted.
    watched.append({
        "id": "q0_false_achieved",
        "what": "RAMANUJAN_Q0_CLOSURE.history: previous receipt claimed ACHIEVED while the named harness had been deleted by two LOC-reduction passes",
        "later": "8b0c54053 deleted more of research/ramanujan/; 5ae13da07 restored it; 8230904e2 closed Q0 for real",
        "current_status": live["q0_closure"]["status"],
        "leaf_hashes_match": live["q0_leaves_match"],
        "result": "RE_VERIFIED_PROVEN_AFTER_METHOD_FAILURE",
    })

    # 11. CPython bytecode: do not import ramanujan.*
    pycache = RAMANUJAN / "__pycache__"
    watched.append({
        "id": "import_writes_pycache",
        "what": "CPython 3.14 wrote research/ramanujan/__pycache__/{__init__,layout}.cpython-314.pyc from `from ramanujan.layout import ramanujan_path` even with sys.dont_write_bytecode=True",
        "result": "BYTECODE_IS_A_TREE_MUTATION",
        "pycache_present_now": pycache.is_dir(),
        "mitigation": "this script reads research/ramanujan/ as bytes and JSON; it does not import the package",
    })

    # 12. .gravity artifacts deleted with no receipt — restream cannot name a substrate.
    watched.append({
        "id": "gravity_deleted_no_receipt",
        "what": "RAMANUJAN_RESTREAM_PLAN: all .gravity artifacts were deleted on 2026-07-29 with no receipt",
        "result": "NO_ARTIFACT_TO_NAME",
        "reopen": "rebuild a .gravity and pass the capability gate with live generation evidence, or run the preregistered representation tournament",
    })

    return watched


def load_live() -> dict[str, Any]:
    q0 = load_json(RAMANUJAN / "records/audits/RAMANUJAN_Q0_CLOSURE.json")
    bundle = load_json(RAMANUJAN / "records/audits/RAMANUJAN_Q0_EVIDENCE_BUNDLE.json")
    q0q6 = load_json(RAMANUJAN / "governance/contracts/RAMANUJAN_Q0_Q6_CONTRACTS.json")
    freeze = load_json(RAMANUJAN / "scaffold/data/corpora/FREEZE_RECEIPT.json")
    gen = load_json(RAMANUJAN / "scaffold/data/corpora/GENERATION_RECEIPT.json")
    scale = load_json(RAMANUJAN / "scaffold/data/corpora/SCALEUP_RECEIPT.json")
    contam = load_json(RAMANUJAN / "scaffold/data/corpora/CONTAMINATION_RECEIPT.json")
    metrics = load_json(RAMANUJAN / "scaffold/train/HELD_OUT_METRICS.json")
    train = load_json(RAMANUJAN / "scaffold/train/TRAINING_RECEIPT.json")
    status = load_json(RAMANUJAN / "records/runtime/SMALL_SYSTEM_TRAINING_STATUS.json")
    gate = load_json(RAMANUJAN / "governance/boundary/HAWKING_COMPLETION_GATE.json")
    gov = load_json(RAMANUJAN / "governance/boundary/RAMANUJAN_GOVERNANCE_STATUS.json")
    green = load_json(RAMANUJAN / "governance/boundary/RAMANUJAN_GREEN_LIGHT_TRANSITION.json")
    offline = load_json(RAMANUJAN / "governance/boundary/RAMANUJAN_OFFLINE_MANIFEST.json")
    owner = load_json(RAMANUJAN / "governance/boundary/RAMANUJAN_OWNER_DECISIONS_REQUIRED.json")
    restream = load_json(RAMANUJAN / "governance/contracts/RAMANUJAN_RESTREAM_PLAN.json")
    env = load_json(RAMANUJAN / "records/runtime/RAMANUJAN_ENVIRONMENT_LOCK.json")
    selftest = load_json(RAMANUJAN / "records/runtime/RAMANUJAN_TOOLCHAIN_SELFTEST.json")
    cog = load_json(RAMANUJAN / "records/runtime/RAMANUJAN_COGNITION_REGISTER.json")
    det = load_json(RAMANUJAN / "records/intake/RAMANUJAN_CORPUS_DETERMINISM.json")
    audit = load_json(RAMANUJAN / "records/audits/RAMANUJAN_PRE_RESTREAM_AUDIT.json")
    replay = load_json(RAMANUJAN / "container/REPLAY_RECEIPT.json")
    build = load_json(RAMANUJAN / "container/BUILD_RECEIPT.json")
    roles = load_json(RAMANUJAN / "governance/contracts/RAMANUJAN_ROLES_ECONOMICS.json")
    manifest = load_json(RAMANUJAN / "scaffold/data/corpora/MEMBERSHIP_MANIFEST.json")

    # Q0 leaf hashes vs live bytes via layout mapping.
    leaf_rows = []
    all_match = True
    for logical, expected in bundle["leaf_sha256"].items():
        physical = LOGICAL.get(logical, RAMANUJAN / logical.removeprefix("research/ramanujan/"))
        if not physical.is_file():
            leaf_rows.append({"logical": logical, "physical": str(physical), "status": "MISSING"})
            all_match = False
            continue
        live_h = sha256_file(physical)
        ok = live_h == expected
        all_match = all_match and ok
        leaf_rows.append({
            "logical": logical,
            "physical": physical.relative_to(REPO).as_posix(),
            "bound": expected,
            "live": live_h,
            "match": ok,
        })

    binding_rows = []
    for logical, expected in q0q6["receipt_bindings"].items():
        if logical in LOGICAL:
            physical = LOGICAL[logical]
            live_h = sha256_file(physical) if physical.is_file() else None
            binding_rows.append({
                "logical": logical,
                "physical": physical.relative_to(REPO).as_posix() if physical.is_file() else None,
                "bound": expected,
                "live": live_h,
                "match": live_h == expected,
            })
        elif logical == "odyssey/verifiers/VERIFICATION_LATTICE.json":
            blob = git_show(GIT_ONLY["verification_lattice"])
            live_h = sha256_bytes(blob) if blob is not None else None
            binding_rows.append({
                "logical": logical,
                "physical": None,
                "git_path_now": GIT_ONLY["verification_lattice"],
                "bound": expected,
                "live": live_h,
                "match": live_h == expected,
            })
        else:
            binding_rows.append({"logical": logical, "bound": expected, "match": None})

    handoff_text = git_show_text(GIT_ONLY["handoff"])
    handoff = json.loads(handoff_text) if handoff_text else None
    intake_text = git_show_text(GIT_ONLY["glm52_fast_intake"])
    intake = json.loads(intake_text) if intake_text else None

    jsonl_counts = {}
    for src, fname in (
        ("D1", "d1_proof_traces.jsonl"),
        ("D2", "d2_state_transitions.jsonl"),
        ("D3", "d3_premise_pairs.jsonl"),
        ("D4", "d4_repair_pairs.jsonl"),
        ("D6", "d6_counterexamples.jsonl"),
        ("D7", "d7_tool_use_traces.jsonl"),
    ):
        p = RAMANUJAN / "scaffold/data/corpora" / fname
        with p.open("rb") as f:
            n = sum(1 for _ in f)
        jsonl_counts[src] = {"path": p.relative_to(REPO).as_posix(), "lines": n, "bytes": p.stat().st_size}

    return {
        "q0_closure": q0,
        "q0_bundle": bundle,
        "q0_q6": q0q6,
        "q0_leaves": leaf_rows,
        "q0_leaves_match": all_match,
        "q0_q6_bindings": binding_rows,
        "freeze": freeze,
        "generation": gen,
        "scaleup": scale,
        "contamination": contam,
        "metrics": metrics,
        "training": train,
        "training_status": status,
        "gate": gate,
        "governance": gov,
        "green": green,
        "offline": offline,
        "owner": owner,
        "restream": restream,
        "environment": env,
        "selftest": selftest,
        "cognition": cog,
        "determinism": det,
        "audit": audit,
        "replay": replay,
        "build": build,
        "roles": roles,
        "manifest": manifest,
        "handoff": handoff,
        "glm52_fast_intake": intake,
        "jsonl_counts": jsonl_counts,
    }


def measured(live: dict[str, Any]) -> list[dict[str, Any]]:
    m = live["metrics"]["components"]
    t = live["training"]
    gen = live["generation"]
    scale = live["scaleup"]
    freeze = live["freeze"]
    q0 = live["q0_closure"]
    env = live["environment"]
    audit = live["audit"]
    green = live["green"]
    restream = live["restream"]
    replay = live["replay"]
    rows: list[dict[str, Any]] = []

    def add(claim: str, value: Any, receipt: str, extra: dict[str, Any] | None = None) -> None:
        rec = {"claim": claim, "value": value, "receipt": receipt}
        if extra:
            rec.update(extra)
        rows.append(rec)

    add(
        "Q0 capsule replay in network-disabled pinned container",
        {
            "status": q0["status"],
            "capsule": q0["capsule"],
            "image_tag": q0["image"]["tag"],
            "image_id": q0["image"]["id"],
            "image_size_bytes": q0["image"]["size_bytes"],
            "tier3": q0["tier3_requirements"],
            "replay_exit_code": replay["exit_code"],
            "replay_network": replay["network"],
            "hawking_commit": q0["hawking_commit"],
        },
        "research/ramanujan/records/audits/RAMANUJAN_Q0_CLOSURE.json",
        {"supporting": [
            "research/ramanujan/container/REPLAY_RECEIPT.json",
            "research/ramanujan/records/audits/RAMANUJAN_Q0_EVIDENCE_BUNDLE.json",
        ]},
    )
    add(
        "Q0 evidence-bundle leaf hashes still match live bytes",
        live["q0_leaves_match"],
        "research/ramanujan/records/audits/RAMANUJAN_Q0_EVIDENCE_BUNDLE.json",
        {"leaves": live["q0_leaves"]},
    )
    add(
        "Lean-derived corpora generated from pinned Mathlib",
        {
            "mathlib_commit": gen["mathlib_commit"],
            "n_modules": gen["n_modules"],
            "n_decls_parsed": gen["n_decls_parsed"],
            "counts": gen["counts"],
            "jsonl_lines_on_disk": {k: v["lines"] for k, v in live["jsonl_counts"].items()},
        },
        "research/ramanujan/scaffold/data/corpora/GENERATION_RECEIPT.json",
    )
    add(
        "Scale-up wall clock and contamination negative control",
        {
            "wall_clock_seconds": scale["wall_clock"]["seconds"],
            "counts_before": scale["counts_before"],
            "counts_after": scale["counts_after"],
            "d4_all_real_lean_errors": scale["d4"]["all_real_lean_errors"],
            "contamination_total_rejected": scale["contamination"]["total_rejected"],
            "negative_control": scale["contamination"]["negative_control"],
        },
        "research/ramanujan/scaffold/data/corpora/SCALEUP_RECEIPT.json",
    )
    add(
        "Frozen membership (current seal)",
        {
            "status": freeze["status"],
            "at": freeze["at"],
            "counts": freeze["counts"],
            "membership_sha256": freeze["membership_sha256"],
            "manifest_file_sha256": freeze["manifest_file_sha256"],
            "total_rejected_vs_eval": freeze["contamination"]["total_rejected_vs_eval"],
            "n_eval_items_indexed": freeze["contamination"]["n_eval_items_indexed"],
            "negative_control_pass": freeze["negative_control_pass"],
        },
        "research/ramanujan/scaffold/data/corpora/FREEZE_RECEIPT.json",
    )
    add(
        "Small-system training (CPU, 103.52s) — bound to membership c0c13806, not the later freeze",
        {
            "at": t["at"],
            "device": t["device"],
            "wall_clock_seconds": t["wall_clock_seconds"],
            "membership_sha256": t["membership_sha256"],
            "converged_components": t["converged_components"],
            "trained_but_did_not_converge": t["trained_but_did_not_converge"],
            "teacher_from_math_preserve": t["teacher_from_math_preserve"],
            "RAMANUJAN_RESEARCH_AUTHORIZED": t["RAMANUJAN_RESEARCH_AUTHORIZED"],
        },
        "research/ramanujan/scaffold/train/TRAINING_RECEIPT.json",
        {"metrics_receipt": "research/ramanujan/scaffold/train/HELD_OUT_METRICS.json"},
    )
    add(
        "Retriever held-out test (D3 dual-encoder vs token-overlap baseline)",
        {
            "status": m["retriever"]["status"],
            "improved_vs_baseline": m["retriever"]["improved_vs_baseline"],
            "held_out_test": m["retriever"]["held_out_test"],
            "baseline_test": m["retriever"]["baseline_test"],
        },
        "research/ramanujan/scaffold/train/HELD_OUT_METRICS.json",
    )
    add(
        "Value held-out test (D2 closed-next + remaining-steps)",
        {
            "status": m["value"]["status"],
            "improved_vs_baseline": m["value"]["improved_vs_baseline"],
            "held_out_test": m["value"]["held_out_test"],
        },
        "research/ramanujan/scaffold/train/HELD_OUT_METRICS.json",
    )
    add(
        "Formalizer held-out test (D1 first-tactic closed-vocab) — did not beat majority",
        {
            "status": m["formalizer"]["status"],
            "improved_vs_baseline": m["formalizer"]["improved_vs_baseline"],
            "held_out_test": m["formalizer"]["held_out_test"],
        },
        "research/ramanujan/scaffold/train/HELD_OUT_METRICS.json",
    )
    add(
        "Prover held-out test (D2 next-tactic closed-vocab) — did not beat majority",
        {
            "status": m["prover"]["status"],
            "improved_vs_baseline": m["prover"]["improved_vs_baseline"],
            "held_out_test": m["prover"]["held_out_test"],
        },
        "research/ramanujan/scaffold/train/HELD_OUT_METRICS.json",
    )
    add(
        "Repair held-out test (D4) including Lean compile under isolated wrapper",
        {
            "status": m["repair"]["status"],
            "improved_vs_baseline": m["repair"]["improved_vs_baseline"],
            "held_out_test": m["repair"]["held_out_test"],
            "held_out_test_lean": m["repair"]["held_out_test_lean"],
        },
        "research/ramanujan/scaffold/train/HELD_OUT_METRICS.json",
    )
    add(
        "Solver machine-checks (tools execute; not research capability)",
        env.get("proof_solvers_work"),
        "research/ramanujan/records/runtime/RAMANUJAN_ENVIRONMENT_LOCK.json",
    )
    add(
        "Toolchain selftest verdict",
        {
            "verdict": live["selftest"]["verdict"],
            "missing": live["selftest"]["missing"],
            "q0_reproducibility": live["selftest"]["q0_reproducibility"],
            "lean_version_when_probed": live["selftest"]["binaries"]["lean"]["version"],
        },
        "research/ramanujan/records/runtime/RAMANUJAN_TOOLCHAIN_SELFTEST.json",
    )
    add(
        "Green-light storage sample (not a future admission receipt)",
        green["details"]["storage"],
        "research/ramanujan/governance/boundary/RAMANUJAN_GREEN_LIGHT_TRANSITION.json",
        {"exact_next_action": green["exact_next_action"], "status": green["status"]},
    )
    add(
        "Pre-restream audit: Llama-3.3-70B Metal decode floor",
        next(
            (fam for fam in audit["large_family_gauntlet"]["families"] if fam["family"] == "Llama-3.3-70B"),
            None,
        ),
        "research/ramanujan/records/audits/RAMANUJAN_PRE_RESTREAM_AUDIT.json",
    )
    add(
        "Pre-restream audit: Qwen2.5-72B first-forward swap floor",
        audit["qwen_first_dispatch_closure"]["same_source_reprobe"],
        "research/ramanujan/records/audits/RAMANUJAN_PRE_RESTREAM_AUDIT.json",
    )
    add(
        "Pre-restream audit: Q4_K fixture representation matrix (p50 microseconds)",
        audit["qwen_first_dispatch_closure"]["same_fixture_representation_matrix"],
        "research/ramanujan/records/audits/RAMANUJAN_PRE_RESTREAM_AUDIT.json",
    )
    add(
        "GLM-5.2 restream schedule (technical, not authorized)",
        {
            "parent_revision": restream["parent"]["revision"],
            "weight_shards": restream["parent"]["weight_shards"],
            "pack_seconds_per_shard": restream["shard_rotation"]["measured_costs"]["pack_seconds_per_shard"],
            "fetch_seconds_per_shard": restream["shard_rotation"]["measured_costs"]["fetch_seconds_per_shard"],
            "windows": audit["restream_machinery"]["schedule"]["windows"],
            "ranges": audit["restream_machinery"]["schedule"]["ranges"],
            "logical_source_payload_bytes": audit["restream_machinery"]["schedule"]["logical_source_payload_bytes"],
            "peak_incremental_bytes": audit["restream_machinery"]["schedule"]["peak_incremental_bytes"],
            "activation_aware_cosine_at_0.167_bpw": restream["representation"]["new_input_since_preregistration"],
        },
        "research/ramanujan/governance/contracts/RAMANUJAN_RESTREAM_PLAN.json",
        {"supporting": ["research/ramanujan/records/audits/RAMANUJAN_PRE_RESTREAM_AUDIT.json"]},
    )
    add(
        "Code-topology condensation recorded at pre-restream audit",
        audit["code_topology"],
        "research/ramanujan/records/audits/RAMANUJAN_PRE_RESTREAM_AUDIT.json",
    )
    add(
        "Research ledger has a single Q0 machine-check event at seq 0",
        {
            "path": "research/ramanujan/scaffold/data/records/research_ledger.jsonl",
            "lines": 1,
            "payload": json.loads(
                (RAMANUJAN / "scaffold/data/records/research_ledger.jsonl").read_text(encoding="utf-8").splitlines()[0]
            ),
        },
        "research/ramanujan/scaffold/data/records/research_ledger.jsonl",
    )
    if live["glm52_fast_intake"]:
        add(
            "GLM52 Ramanujan fast-intake (Hawking-side; blocked)",
            {
                "status": live["glm52_fast_intake"].get("status"),
                "decode_checked_rows": live["glm52_fast_intake"].get("gates", {}).get("DECODE_PERFORMANCE", {}).get("checked_rows"),
            },
            GIT_ONLY["glm52_fast_intake"],
            {"note": "git-only sibling; not inside research/ramanujan/"},
        )
    if live["handoff"]:
        mp = None
        for b in live["handoff"]["bound_read_only_by_hash"]["bindings"]:
            if b["name"].startswith("Math-Preserve"):
                mp = b
        add(
            "Handoff-bound Math-Preserve artifact (never copied into Ramanujan)",
            mp,
            GIT_ONLY["handoff"],
        )
    return rows


def refutations(live: dict[str, Any]) -> list[dict[str, Any]]:
    m = live["metrics"]["components"]
    return [
        {
            "id": "R1_math_preserve_teacher",
            "claim": "Generate teacher traces from the Math-Preserve .gravity artifact",
            "result": "REFUSED",
            "classification": "PROPERTY_OF_THE_IDEA",
            "why": (
                "The artifact is semantically collapsed: it predicts ' combust' for the capital of "
                "France and 'rus' for 2+2, confirmed by numpy oracle against the runtime. Traces "
                "would teach a student to reproduce the collapse and would look like data. "
                "L-TEACHER-01 is permanent for this lane. Cheapest Falsifier's own note: a 2+2 "
                "forward pass refutes the entire substrate."
            ),
            "receipts": [
                "research/ramanujan/governance/boundary/RAMANUJAN_OFFLINE_MANIFEST.json",
                "research/ramanujan/scaffold/train/TRAINING_RECEIPT.json#limit_consults",
                "research/ramanujan/records/runtime/RAMANUJAN_COGNITION_REGISTER.json#CheapestFalsifier",
                GIT_ONLY["substrate_capability"],
            ],
            "reopen_condition": (
                "Never from Math-Preserve. Reopen teacher-trace generation only against an artifact "
                "whose capability_verdict is APPROVED, with live generation evidence bound to that "
                "exact content hash. Absence is UNVERIFIED, which is also refused."
            ),
        },
        {
            "id": "R2_live_research_and_restream",
            "claim": "Ramanujan can launch research or parent-restream",
            "result": "BLOCKED_ON_HAWKING_COMPLETION / OWNER_GATE_READY / restream_started=false",
            "classification": "ARTIFACT_OF_METHOD",
            "why": (
                "The split was gated on HAWKING_EVOLUTION_COMPLETE and on Math-Frozen existing. "
                "A checkpoint is not automatically Ramanujan. The restream command is written down "
                "to the argv and still exits closed: the capability gate has no APPROVED artifact "
                "to name. This is sequencing, not a refutation of the freeze-the-giant / train-the-small doctrine."
            ),
            "receipts": [
                "research/ramanujan/governance/boundary/HAWKING_COMPLETION_GATE.json",
                "research/ramanujan/governance/boundary/RAMANUJAN_GREEN_LIGHT_TRANSITION.json",
                "research/ramanujan/governance/contracts/RAMANUJAN_RESTREAM_PLAN.json",
                GIT_ONLY["handoff"],
            ],
            "reopen_condition": (
                "Validated Hawking completion evidence PLUS owner D5/D8/D9 public freeze receipt "
                "PLUS parent-restream authorization PLUS production clean GPU lease PLUS owner-"
                "approved physical GLM-5.2 window operator PLUS an APPROVED (not REFUSED) .gravity "
                "substrate. Ramanujan code must never flip any of these itself. Storage must be "
                "recomputed immediately before admission: free_bytes - 58885799936 >= 200005889556."
            ),
        },
        {
            "id": "R3_closed_vocab_formalizer",
            "claim": "A closed-vocab first-tactic classifier is a formalizer",
            "result": "TRAINED_BUT_NO_BETTER_THAN_MAJORITY",
            "classification": "ARTIFACT_OF_METHOD",
            "why": (
                f"Held-out test exact_match={m['formalizer']['held_out_test']['exact_match']} vs "
                f"majority {m['formalizer']['held_out_test']['majority_baseline_exact_match']} "
                f"(n={m['formalizer']['held_out_test']['n']}, 365 classes, 10 CPU epochs). "
                "The majority label is 'exact ( rfl'. This does not refute autoformalization (Q4 "
                "is still DEFINED_CONTROLLER_SCAFFOLD_READY); D5 informal/formal pairs were never acquired."
            ),
            "receipts": [
                "research/ramanujan/scaffold/train/HELD_OUT_METRICS.json",
                "research/ramanujan/governance/contracts/RAMANUJAN_Q0_Q6_CONTRACTS.json#Q4",
            ],
            "reopen_condition": (
                "Seq2seq proof generation beyond closed vocab, plus owner-licensed D5 pairs, plus "
                "evaluation bound to current membership fec9f85e (not the 2026-07-27 membership)."
            ),
        },
        {
            "id": "R4_closed_vocab_prover",
            "claim": "A closed-vocab next-tactic classifier is a prover",
            "result": "TRAINED_BUT_NO_BETTER_THAN_MAJORITY",
            "classification": "ARTIFACT_OF_METHOD",
            "why": (
                f"Held-out test exact_match={m['prover']['held_out_test']['exact_match']} vs "
                f"majority {m['prover']['held_out_test']['majority_baseline_exact_match']} "
                f"(n={m['prover']['held_out_test']['n']}, 381 classes). "
                "end_to_end_search_policy remains still_scaffold. Tree search was specified, not tried at production scale."
            ),
            "receipts": ["research/ramanujan/scaffold/train/HELD_OUT_METRICS.json"],
            "reopen_condition": (
                "Interactive Lean state value from real goals plus best-first search with the "
                "retriever/value pair that DID converge — not another closed-vocab classifier."
            ),
        },
        {
            "id": "R5_repair_wrapper_ceiling",
            "claim": "The D4 repair controller compiles fixes under pinned Mathlib",
            "result": "predicted_fix_compiles=0.05; gold_fix_compiles=0.10 under the isolated wrapper",
            "classification": "MIXED_METHOD_AND_WRAPPER",
            "why": (
                "Closed-vocab repair lost to the majority label 'rfl' on exact match. Separately, "
                "the isolated example wrapper lacks typeclass context: gold compile rate is 0.10 "
                "on the 40-item Lean subset (honest ceiling). Nat.Basic subset is the fairer metric "
                "(gold 0.667, predicted 0.333). Compiler-feedback repair as an idea is not refuted; "
                "D4 is named as the cheapest real training signal."
            ),
            "receipts": [
                "research/ramanujan/scaffold/train/HELD_OUT_METRICS.json",
                "research/ramanujan/records/intake/RAMANUJAN_CORPUS_DETERMINISM.json",
            ],
            "reopen_condition": (
                "A wrapper that preserves typeclass context; D4 content_digest end-to-end rerun "
                "(still owed, about an hour of Lean); do not treat majority-class 'rfl' as a repair."
            ),
        },
        {
            "id": "R6_q0_false_achieved",
            "claim": "Q0 was ACHIEVED (first receipt)",
            "result": "the named harness had been deleted; later re-verified PROVEN",
            "classification": "ARTIFACT_OF_METHOD",
            "why": (
                "Two LOC-reduction passes deleted the harness a receipt still named. 8b0c54053 "
                "then deleted more of research/ramanujan/; 5ae13da07 restored it; 8230904e2 closed Q0 for "
                "real with five MET requirements and a composing chain. The capsule idea was fine; "
                "the measurement was a ghost."
            ),
            "receipts": [
                "research/ramanujan/records/audits/RAMANUJAN_Q0_CLOSURE.json",
                "git:8b0c54053",
                "git:5ae13da07",
                "git:8230904e2",
            ],
            "reopen_condition": (
                "Not applicable as a refutation — current Q0 is PROVEN and leaf hashes match. "
                "Any leaf-byte drift refuses Q0 until an independent offline rebuild/replay emits a new bundle."
            ),
        },
        {
            "id": "R7_file_hash_freeze",
            "claim": "Sealing the jsonl file sha256 freezes the corpus",
            "result": "unverifiable by construction (provenance.at is a wall clock)",
            "classification": "ARTIFACT_OF_METHOD",
            "why": (
                "Every record carries provenance.at, so the file sha256 changes on every run even "
                "when content is identical. Fixing this with content_digest immediately exposed D4 "
                "temp-path leakage into hashed error text, D4 worker-completion ordering, and D7 "
                "ids built from salted str.__hash__ (30 of 86 ids moved)."
            ),
            "receipts": ["research/ramanujan/records/intake/RAMANUJAN_CORPUS_DETERMINISM.json"],
            "reopen_condition": (
                "D1/D2/D3/D6 already REPRODUCE. D7 ids verified by recomputation. D4 end-to-end "
                "rerun under the sorted, path-stripped code is still owed."
            ),
        },
        {
            "id": "R8_cvc5_gap",
            "claim": "Second SMT solver (cvc5) and GAP are part of the locked environment",
            "result": "UNAVAILABLE_VIA_PACKAGE_MANAGER / NOT_INSTALLED",
            "classification": "ARTIFACT_OF_METHOD",
            "why": (
                "Neither has a Homebrew formula. Acquiring them means downloading and executing a "
                "release binary, which needs explicit owner authorisation that was never given. "
                "Recorded as absent rather than worked around. Does not block Q0. It DOES block "
                "the second-solver disagreement signal: z3 answers alone."
            ),
            "receipts": [
                "research/ramanujan/records/runtime/RAMANUJAN_ENVIRONMENT_LOCK.json",
                "research/ramanujan/records/runtime/RAMANUJAN_TOOLCHAIN_SELFTEST.json",
            ],
            "reopen_condition": "Explicit owner authorisation to fetch the cvc5 and GAP release binaries, then pin sha256.",
        },
        {
            "id": "R9_gravity_artifacts_deleted",
            "claim": "An executable restream command exists",
            "result": "NOT RUNNABLE — no .gravity to name",
            "classification": "ARTIFACT_OF_METHOD",
            "why": (
                "All .gravity artifacts were deleted on 2026-07-29 with no receipt. The capability "
                "gate refuses anything not listed; absence is UNVERIFIED, also refused. Flipping "
                "ODYSSEY_LAUNCH_AUTHORIZED still exits 5 — the fence and the capability gate are independent."
            ),
            "receipts": ["research/ramanujan/governance/contracts/RAMANUJAN_RESTREAM_PLAN.json"],
            "reopen_condition": (
                "(a) rebuild a .gravity and submit it with live generation evidence, or (b) run the "
                "preregistered representation tournament (A1 first, then A4). (b) subsumes (a)."
            ),
        },
        {
            "id": "R10_odyssey_apparatus_missing",
            "claim": "The Odyssey apparatus does not exist",
            "result": "CORRECTED: the apparatus exists and refuses",
            "classification": "ARTIFACT_OF_METHOD",
            "why": (
                "f1c55a302 corrected a finding that looked for files and missed fail-closed behaviour. "
                "Fixture self-test covers T0-T2; full T0-T12/F0-F12/Q0-Q12 runners require injected "
                "fixtures and cannot enable research authority."
            ),
            "receipts": [
                "git:f1c55a302",
                "research/ramanujan/scaffold/research/odyssey.py",
                "research/ramanujan/README.md",
            ],
            "reopen_condition": "Not a refutation of Odyssey. Reopen live Odyssey only under R2's conditions.",
        },
        {
            "id": "R11_independent_layerwise_flash",
            "claim": "Train a DeepSeek-V4 Flash student independently per layer / cascade",
            "result": "REFUSED by the proto program; Flash cascade evidence already ruled it out",
            "classification": "PROPERTY_OF_THE_IDEA",
            "why": (
                "ramanujan.odyssey proto refuses the independently trained layerwise route. The "
                "cascade negative lives outside research/ramanujan/; this tree records the refusal, not a "
                "re-derivation of the cascade measurement."
            ),
            "receipts": [
                "research/ramanujan/README.md",
                "research/ramanujan/scaffold/research/odyssey.py#ProtoStudent",
                "research/ramanujan/scaffold/tests/test_odyssey_harness.py",
            ],
            "reopen_condition": (
                "Only if new evidence invalidates the Flash cascade negative. Default student remains "
                "route-aware rollout on pinned DeepSeek-V4-Flash revision 60d8d70770c6776ff598c94bb586a859a38244f1."
            ),
        },
        {
            "id": "R12_metrics_vs_current_freeze",
            "claim": "Held-out metrics describe the currently sealed membership",
            "result": "FALSE — training 2026-07-27 membership c0c13806; freeze 2026-08-01 membership fec9f85e; 7-item train/test shift",
            "classification": "ARTIFACT_OF_METHOD",
            "why": (
                "Generation/freeze were re-run on 2026-08-01. Training numbers were not. "
                "Retraining was never done against the current freeze. The metrics remain true of "
                "the older membership and are not current-freeze evidence."
            ),
            "receipts": [
                "research/ramanujan/scaffold/train/TRAINING_RECEIPT.json",
                "research/ramanujan/scaffold/data/corpora/FREEZE_RECEIPT.json",
                "research/ramanujan/records/runtime/SMALL_SYSTEM_TRAINING_STATUS.json",
            ],
            "reopen_condition": "Re-train or re-evaluate the five components against membership fec9f85e. Do not cite 0.879 MRR as current-freeze evidence until then.",
        },
        {
            "id": "R13_q2_q3_pending_not_zero",
            "claim": "Q2 rediscovery / Q3 frontier-variant have been measured at zero",
            "result": "DEFINED_PENDING_OWNER_D8/D9 — an absent D8 set is PENDING, not zero credit",
            "classification": "GATED_NOT_REFUTED",
            "why": "Owner never selected licensed D5/D8/D9 sources. Q0-Q6 contracts explicitly forbid treating absence as a score.",
            "receipts": [
                "research/ramanujan/governance/contracts/RAMANUJAN_Q0_Q6_CONTRACTS.json",
                "research/ramanujan/governance/boundary/RAMANUJAN_OWNER_DECISIONS_REQUIRED.json",
            ],
            "reopen_condition": live["owner"]["decisions"][1]["reopen_condition"] if live["owner"].get("decisions") else "owner-signed D8/D9 freeze",
        },
        {
            "id": "R14_python_pin_disagreement",
            "claim": "The environment lock's Python pin is the interpreter the tooling uses",
            "result": "lock records python 3.14.6; selftest ran 3.12.6; README commands use python3.12",
            "classification": "ARTIFACT_OF_METHOD",
            "why": "The lock itself warns: 'the repository's own tooling targets python3.12; reconcile before locking'. Two authorities on the interpreter.",
            "receipts": [
                "research/ramanujan/records/runtime/RAMANUJAN_ENVIRONMENT_LOCK.json",
                "research/ramanujan/records/runtime/RAMANUJAN_TOOLCHAIN_SELFTEST.json",
            ],
            "reopen_condition": "Pick one interpreter, write a hashed lockfile, re-run toolchain_selftest.",
        },
    ]


def reusable_now(live: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "item": "Fail-closed authority fences",
            "paths": [
                "research/ramanujan/governance/boundary/HAWKING_COMPLETION_GATE.json",
                "research/ramanujan/scaffold/core/limits.py",
                "research/ramanujan/scaffold/core/roles.py",
                "research/ramanujan/scaffold/guards/status.py",
                "research/ramanujan/scaffold/guards/restream_guard.py",
            ],
            "what": (
                "RAMANUJAN_RESEARCH_AUTHORIZED has no flip path. Generators cannot promote. "
                "Burial is not deletion. L-TEACHER-01 / L-LAUNCH-01 / L-NET-01. These laws are "
                "still the right ones for Odyssey-i / HCLI."
            ),
        },
        {
            "item": "Q0 clean-container recipe and hash-bound capsule",
            "paths": [
                "research/ramanujan/container/",
                "research/ramanujan/records/audits/RAMANUJAN_Q0_EVIDENCE_BUNDLE.json",
                "research/ramanujan/records/audits/RAMANUJAN_Q0_CLOSURE.json",
            ],
            "what": (
                "Dockerfile, pins, replay scripts, two_plus_two capsule, leaf sha256s that still "
                "match live bytes. The 4.64 GiB image is NOT in git; the recipe is. Re-build, "
                "don't re-invent."
            ),
        },
        {
            "item": "Historical-path map",
            "paths": ["research/ramanujan/layout.py"],
            "what": (
                "Sealed receipts name research/ramanujan/RAMANUJAN_Q0_CLOSURE.json and research/ramanujan/ledger.py. "
                "Those files physically live under records/ and scaffold/. Deleting layout.py "
                "makes the seals unreadable without archaeology."
            ),
        },
        {
            "item": "Lean-derived corpora D1 D2 D3 D4 D6 D7 plus freeze",
            "paths": ["research/ramanujan/scaffold/data/corpora/"],
            "what": (
                "16188 items, content_digest sealed, contamination negative control (tl02_bpw) "
                "caught. Regenerable from Mathlib 2ec0166b except D4's owed rerun. This is most "
                "of the 27M."
            ),
        },
        {
            "item": "Q0-Q6 qualification lattice",
            "paths": ["research/ramanujan/governance/contracts/RAMANUJAN_Q0_Q6_CONTRACTS.json"],
            "what": (
                "Q0 PROVEN, Q1 PROVEN_OFFLINE, Q2-Q3 pending owner, Q4-Q6 scaffold-ready. "
                "File hashes of the bound receipts still match live bytes (lattice path moved, bytes held)."
            ),
        },
        {
            "item": "Cognition register with kill conditions",
            "paths": ["research/ramanujan/records/runtime/RAMANUJAN_COGNITION_REGISTER.json"],
            "what": (
                "13 mechanisms, each with kill/ablation/self-deception written before implementation. "
                "Four exist on fixtures: Research Object Graph, Cheapest Falsifier, Calibration, Capsule. "
                "A mechanism without a kill condition is a mechanism that will be defended rather than tested."
            ),
        },
        {
            "item": "Odyssey fixture control plane (T0-T12 / F0-F12 / Q0-Q12) plus proto contracts",
            "paths": [
                "research/ramanujan/scaffold/research/odyssey.py",
                "research/ramanujan/scaffold/tests/test_odyssey_harness.py",
            ],
            "what": (
                "Fail-closed Director environment; refuses research authorization; ProtoGravityRenderer "
                "emits a contract not a .gravity file; Condense spec is capability-first (4→3→2→1.5→1.25→1 BPW) "
                "and cannot invent a TPS number. Required Gravity gates are the reopen checklist."
            ),
        },
        {
            "item": "Determinism defects already found",
            "paths": ["research/ramanujan/records/intake/RAMANUJAN_CORPUS_DETERMINISM.json"],
            "what": (
                "file-hash freeze, D4 mkdtemp path leak, D4 worker order, D7 PYTHONHASHSEED. "
                "Do not rediscover these by regenerating."
            ),
        },
        {
            "item": "Hawking/Ramanujan split contract",
            "paths": [GIT_ONLY["handoff"]],
            "what": (
                "Ramanujan owns doctrine, cognition, governance, the small trainable system. "
                "Hawking owns the .gravity codec, runtime, kernels, HIDE, Fabric. Giant artifacts "
                "are bound by hash and never copied. git-only in this sparse tree."
            ),
        },
        {
            "item": "Retriever and value architectures (not their 2026-07-27 weights)",
            "paths": [
                "research/ramanujan/scaffold/train/models.py",
                "research/ramanujan/scaffold/train/train_components.py",
                "research/ramanujan/scaffold/research/search.py",
            ],
            "what": (
                "Dual-encoder retriever beat token-overlap; value beat majority closed. "
                "Checkpoints are gone. Code and metrics remain. Re-train against fec9f85e before citing."
            ),
        },
    ]


def deletion_cost(inv: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    corpora_bytes = sum(v["bytes"] for v in live["jsonl_counts"].values())
    manifest_bytes = (RAMANUJAN / "scaffold/data/corpora/MEMBERSHIP_MANIFEST.json").stat().st_size
    return {
        "if_ramanujan_deleted": [
            {
                "lost": "the entire 27 MiB campaign tree",
                "bytes": inv["total_bytes"],
                "git_entries": inv["git_ls_tree_count"],
                "on_disk_entries": inv["on_disk_entries"],
            },
            {
                "lost": "six Lean-derived corpora (16188 items) plus the membership manifest",
                "corpus_jsonl_bytes": corpora_bytes,
                "manifest_bytes": manifest_bytes,
                "regenerable": (
                    "D1 D2 D3 D6 from Mathlib 2ec0166b in ~10 minutes; D4 needs ~1 hour Lean and "
                    "the owed determinism rerun; D7 is small and verified without rerun"
                ),
                "not_regenerable_if_mathlib_pin_moves": True,
            },
            {
                "lost": "Q0 hash-bound capsule, Dockerfile, pins, replay scripts, and the leaf-seal that still matches",
                "image_not_in_git_bytes": live["q0_closure"]["image"]["size_bytes"],
                "image_note": "the 4.64 GiB Docker image is already not in git; deleting research/ramanujan/ deletes the only recipe that can rebuild it",
            },
            {
                "lost": "layout.py — the map from sealed logical paths to compact physical paths",
                "consequence": "Q0-Q6 receipt_bindings and the Q0 evidence bundle become archaeology",
            },
            {
                "lost": "the Q0 false-ACHIEVED then re-PROVEN history",
                "consequence": "the next LOC-reduction pass will delete a harness and keep a receipt again",
            },
            {
                "lost": "cognition kill conditions, role/economics law, Limit Registry, YOU-research controller",
                "consequence": "the doctrine half of the Hawking split has no home",
            },
            {
                "lost": "held-out numbers (retriever MRR 0.879 vs 0.714; value closed 0.969 vs 0.701; three majority losses)",
                "consequence": "a later lane will retrain a closed-vocab formalizer and think the idea is new",
            },
            {
                "lost": "L-TEACHER-01 / Math-Preserve collapse record inside this tree",
                "consequence": "someone will distill traces from a collapsed substrate because it looks like data",
            },
            {
                "lost": "D5/D8/D9 owner packet and restream argv that correctly will not run",
                "consequence": "the next restream attempt will be designed under pressure instead of executed from a written command",
            },
            {
                "lost": "Odyssey fixture control plane and proto-Gravity/Condense contracts",
                "consequence": "T/F/Q sequencing and the 1-BPW Flash student contract have to be re-specified",
            },
            {
                "already_absent_even_without_deletion": [
                    "five .pt checkpoints named by TRAINING_RECEIPT",
                    "Math-Frozen Director (never created)",
                    ".gravity artifacts (deleted 2026-07-29 with no receipt)",
                    "D5/D8/D9 source bytes (never selected)",
                    "cvc5 and GAP binaries",
                    "the 7.1 GiB host Mathlib clone (outside the repo)",
                    "the 4.64 GiB Docker image (outside the repo)",
                ],
            },
        ],
        "what_this_receipt_saves_if_the_tree_goes": (
            "Goal, period, mechanism, lineage to Gravity/Doctor/Odyssey, full inventory with "
            "sizes and sha256, every measured number with its receipt, every refutation classified "
            "with a reopen condition, the reusable-now list, and the deletion-cost list. It does "
            "not save the corpora bytes, the Q0 recipe, or the runnable scaffold. Indexing is not "
            "a backup."
        ),
        "prior_deletion_already_happened": {
            "deleted": "8b0c54053 2026-07-29T19:17:12-04:00 Install counted verification and retire historical recovery executables",
            "restored": "5ae13da07 2026-07-30T16:24:23-04:00 Restore the Ramanujan system that 8b0c5405 deleted",
            "lesson": "This repo has already deleted Ramanujan once and had to put it back. Index first.",
        },
    }


def what_it_was(live: dict[str, Any]) -> dict[str, Any]:
    return {
        "one_sentence": (
            "Ramanujan is a fail-closed, non-authorizing local scaffold for a future mathematical "
            "research system: freeze a giant Hawking/Gravity Director, train a small formal system "
            "against it, verify everything, and search — blocked on Hawking completion by design."
        ),
        "goal": live["handoff"]["purpose"] if live["handoff"] else (
            "Define the Hawking/Ramanujan split so doctrine/cognition/governance live apart from "
            "the compression foundry, bound to a frozen giant by hash."
        ),
        "doctrine": live["offline"]["doctrine"],
        "period": {
            "first_commit": {
                "hash": "19e3e7f2f93c0af94b836acad75a342fe893ea38",
                "at": "2026-07-26T23:04:52-04:00",
                "subject": "Ramanujan governance made runnable, and the cognition register with kill conditions",
            },
            "last_commit_touching_tree": {
                "hash": "a883c0d9c350605860640d47f92f41dcf0aeae2d",
                "at": "2026-08-05T15:26:55-04:00",
                "subject": "verifier-expert-iteration-20260805-151323",
            },
            "n_commits_touching_ramanujan": int(git_out(["rev-list", "--count", "HEAD", "--", "ramanujan"]) or "0"),
            "disband_date": "2026-08-23",
            "idle_after": "The tree sat while Hawking / Odyssey-i / HCLI continued on odyssey-i.",
        },
        "mechanism": {
            "authority": "NON_PRODUCTION_AUTHORITY; RAMANUJAN_RESEARCH_AUTHORIZED stays false; no flip path",
            "stores": live["governance"]["implemented"]["seven_stores"]["names"],
            "roles": live["roles"]["roles"],
            "qualification": [c["id"] + " " + c["name"] + " = " + c["status"] for c in live["q0_q6"]["contracts"]],
            "data": "D1-D4, D6, D7 generated locally from pinned Mathlib; D5/D8/D9 pending owner license",
            "small_system": "retriever, formalizer, prover, repair, value — CPU, fixture-scale, two of five converged",
            "odyssey_control_plane": "research/ramanujan/scaffold/research/odyssey.py — fixture-only T0-T12/F0-F12/Q0-Q12",
            "restream": "fail-closed launcher + green-light state machine that cannot self-promote",
            "entrypoints": [
                "research/ramanujan/scaffold/guards/status.py",
                "research/ramanujan/scaffold/guards/restream_guard.py",
                "research/ramanujan/scaffold/guards/toolchain_selftest.py",
                "research/ramanujan/scaffold/research/odyssey.py",
                "research/ramanujan/scaffold/guards/RAMANUJAN_FINAL_PARENT_NEXT_COMMAND.sh",
            ],
        },
        "paths": {
            "readme": "research/ramanujan/README.md",
            "gate": "research/ramanujan/governance/boundary/HAWKING_COMPLETION_GATE.json",
            "layout": "research/ramanujan/layout.py",
            "handoff": GIT_ONLY["handoff"],
            "dependency_doc": "research/ramanujan/docs/HAWKING_DEPENDENCY.md",
        },
        "relation_to_current_line": {
            "Hawking": (
                "Owns substrate, runtime, Gravity codec, kernels, HIDE, Fabric. Ramanujan consumes "
                "Hawking contracts and is blocked until HAWKING_EVOLUTION_COMPLETE. The giant never moves."
            ),
            "Gravity": (
                "The .gravity codec stays in Hawking (crates/hawking-core gravity_*). Ramanujan "
                "binds artifacts by content hash and refuses to copy them. ramanujan.odyssey "
                "--proto-gravity-plan emits a future render *contract* with eight required gates "
                "(source admission, DeepSeek-V4 Gravity adapter + dtype parity, condense receipt, "
                "teacher arbitration, route-aware rollout retention, Lean/exact math retention, "
                "checkpoint tournament, owner render window). It will not write a .gravity file. "
                "All .gravity artifacts were deleted 2026-07-29 with no receipt."
            ),
            "Doctor": (
                "Zero Doctor code lives under research/ramanujan/. Doctor is Hawking's adversarial screen "
                "for Gravity representations (tools/gravity_doctor_gate.py: observed vs probed "
                "cosine; later doctor6 / odyssey-i O00*_DOCTOR_SEAL.json). Ramanujan cannot "
                "consume a pack that fails Doctor; Doctor is a dependency, not a child."
            ),
            "Odyssey": (
                "Three layers. (1) tools/odyssey contamination/economics/lattice — the machinery "
                "Ramanujan actually imports for freeze and D5 no-leak. (2) ramanujan.odyssey — a "
                "compact fixture control plane that refuses research authority. (3) branch odyssey-i "
                "— the current Hawking resident campaign, not a Ramanujan launch. f1c55a302 "
                "corrected 'Odyssey is missing' to 'Odyssey exists and refuses'."
            ),
            "Noetic": (
                "No path in HEAD contains the string 'noetic'. If Noetic names the cognition/"
                "doctrine half of the Hawking split, that half is research/ramanujan/ as a scaffold — "
                "cognition register, YOU-research controller, verification lattice, Tribunal. "
                "It is not a separate campaign tree in this repository."
            ),
        },
        "proto_future_not_run": {
            "student": "deepseek-ai/DeepSeek-V4-Flash @ 60d8d70770c6776ff598c94bb586a859a38244f1 (284B total / 13B active)",
            "teachers_sequential": [
                "DeepSeek-V4-Pro structured planning",
                "GLM Math critique/repair",
                "Kimi K3 independent alternatives",
            ],
            "rule": "admit data only if all three teachers bind the same formal statement; mix verifier-dispositioned traces, never weights",
        },
        "current_status": {
            "gate": live["gate"]["status"],
            "governance": live["governance"]["status"],
            "green_light": live["green"]["status"],
            "research_authorized": False,
            "production_authority": False,
            "q0": live["q0_closure"]["status"],
            "toolchain_selftest": live["selftest"]["verdict"],
        },
    }


def lock_or_fail(live: dict[str, Any], failures: list[str]) -> None:
    def eq(name: str, got: Any, wanted: Any) -> None:
        if got != wanted:
            failures.append(f"{name}: got {got!r} wanted {wanted!r}")

    eq("q0.status", live["q0_closure"]["status"], "PROVEN")
    eq("q0.image_id", live["q0_closure"]["image"]["id"],
       "sha256:21114fb4b7066b5a7c535d36685211147a920233fc7544a922846056c8ec03ad")
    eq("q0.image_size", live["q0_closure"]["image"]["size_bytes"], 4644372611)
    eq("replay.exit", live["replay"]["exit_code"], 0)
    eq("gen.mathlib", live["generation"]["mathlib_commit"], "2ec0166b31100827cd34bacca4d3b9ea3da9d618")
    eq("gen.D1", live["generation"]["counts"]["D1"], 5000)
    eq("gen.D2", live["generation"]["counts"]["D2"], 5000)
    eq("gen.D3", live["generation"]["counts"]["D3"], 5000)
    eq("gen.D4", live["generation"]["counts"]["D4"], 800)
    eq("gen.D6", live["generation"]["counts"]["D6"], 302)
    eq("gen.D7", live["generation"]["counts"]["D7"], 86)
    for src, n in live["generation"]["counts"].items():
        eq(f"jsonl.{src}", live["jsonl_counts"][src]["lines"], n)
    eq("freeze.status", live["freeze"]["status"], "PASS")
    eq("freeze.membership", live["freeze"]["membership_sha256"],
       "fec9f85e677921433df4761205c391a4733a358660e1effcfe9f1b8e7651fbc6")
    eq("freeze.manifest_file", live["freeze"]["manifest_file_sha256"],
       "36059fb95459c80d50c25a967ced2f6f52a411dc975f2c66181561dcbcadcb29")
    eq("train.membership", live["training"]["membership_sha256"],
       "c0c13806a1f14553dfa3cdafa747071a088c084e7c462230ed83f1f9011675ce")
    eq("train.converged", live["training"]["converged_components"], ["retriever", "value"])
    eq("retriever.n", live["metrics"]["components"]["retriever"]["held_out_test"]["n"], 496)
    eq("value.n", live["metrics"]["components"]["value"]["held_out_test"]["n"], 545)
    eq("formalizer.n_correct", live["metrics"]["components"]["formalizer"]["held_out_test"]["n_correct"], 53)
    eq("prover.n_correct", live["metrics"]["components"]["prover"]["held_out_test"]["n_correct"], 22)
    eq("repair.n_correct", live["metrics"]["components"]["repair"]["held_out_test"]["n_correct"], 12)
    eq("repair.pred_compiles", live["metrics"]["components"]["repair"]["held_out_test_lean"]["n_predicted_compiles"], 2)
    eq("gate.status", live["gate"]["status"], "BLOCKED_ON_HAWKING_COMPLETION")
    eq("gov.research", live["governance"]["RAMANUJAN_RESEARCH_AUTHORIZED"], False)
    eq("green.production", live["green"]["production_authority"], False)
    eq("q0_leaves", live["q0_leaves_match"], True)
    eq("scale.seconds", live["scaleup"]["wall_clock"]["seconds"], 629)
    eq("selftest.verdict", live["selftest"]["verdict"], "TOOLCHAIN_INCOMPLETE")
    eq("cognition.n", len(live["cognition"]["mechanisms"]), 13)
    git_n = len([ln for ln in git_out(["ls-tree", "-r", "--name-only", "HEAD", "--", "ramanujan"]).splitlines() if ln])
    eq("git_ls_tree", git_n, 110)
    # Q0-Q6 live bindings for files that are on disk
    for row in live["q0_q6_bindings"]:
        if row.get("match") is False:
            failures.append(f"q0q6 binding drifted: {row['logical']}")


def git_only_index() -> list[dict[str, Any]]:
    rows = []
    for key, rel in GIT_ONLY.items():
        blob = git_show(rel)
        size = git_size(rel)
        rows.append({
            "id": key,
            "git_path": rel,
            "on_disk": (REPO / rel).is_file(),
            "bytes": size,
            "sha256": sha256_bytes(blob) if blob is not None else None,
            "present_in_git": blob is not None,
        })
    return rows


def assemble(inv: dict[str, Any], live: dict[str, Any], watched: list[dict[str, Any]],
             start_porcelain: list[str], start_tree: str, lock_failures: list[str]) -> dict[str, Any]:
    end_porcelain = porcelain("ramanujan")
    end_tree = inventory()["tree_sha256"]
    return {
        "schema": SCHEMA,
        "status": "CAMPAIGN_RETIRED_EVIDENCE_RETAINED",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "operator_instruction": "disband ramanujan that is an old idea keep the information we may need",
        "git": {
            "head": git_out(["rev-parse", "HEAD"]),
            "branch": git_out(["rev-parse", "--abbrev-ref", "HEAD"]),
            "ramanujan_tree": git_out(["rev-parse", "HEAD:ramanujan"]),
            "HEAD_subject": git_out(["log", "-1", "--format=%s"]),
        },
        "what_it_was": what_it_was(live),
        "inventory": {
            "summary": {k: inv[k] for k in (
                "root", "on_disk_regular_files", "on_disk_symlinks", "on_disk_entries",
                "total_bytes", "tree_sha256", "git_ls_tree_count", "git_tree", "by_class",
            )},
            "largest": inv["largest"],
            "files": inv["files"],
            "git_only_siblings": git_only_index(),
            "jsonl_counts": live["jsonl_counts"],
        },
        "measured": measured(live),
        "refutations": refutations(live),
        "reusable_now": reusable_now(live),
        "reopen_the_campaign": {
            "required_all_of": [
                "HAWKING_EVOLUTION_COMPLETE with validated evidence",
                "owner-signed D5/D8/D9 public freeze receipt",
                "HAWKING_PARENT_RESTREAM_AUTHORIZED=YES",
                "production CLEAN GPU lease identity + receipt",
                "owner-approved physical GLM-5.2 window operator",
                "an APPROVED .gravity substrate (Math-Preserve stays REFUSED)",
                "Math-Frozen Director created and bound by hash",
                "storage recomputed: free_bytes - 58885799936 >= 200005889556",
                "re-bind or re-train small-system metrics to membership fec9f85e",
                "D4 content_digest end-to-end rerun",
            ],
            "will_not_reopen_it": [
                "editing a Ramanujan JSON file",
                "passing fixture tests",
                "flipping ODYSSEY_LAUNCH_AUTHORIZED",
                "this disband receipt",
            ],
        },
        "deletion_cost": deletion_cost(inv, live),
        "ramanujan_untouched": {
            "start_porcelain": start_porcelain,
            "end_porcelain": end_porcelain,
            "start_tree_sha256": start_tree,
            "end_tree_sha256": end_tree,
            "unchanged": start_tree == end_tree and start_porcelain == end_porcelain,
            "how_verified": [
                "git status --porcelain -- ramanujan at start and end",
                "sha256 roll of every regular file and symlink under research/ramanujan/ before and after writing the receipt",
                "this script never opens research/ramanujan/ paths for write",
                "this script never imports ramanujan.* (CPython 3.14 wrote research/ramanujan/__pycache__ from a layout import even with sys.dont_write_bytecode)",
                "inventory walk skips __pycache__ and .pyc so bytecode cannot inflate the census",
            ],
        },
        "lock_failures": lock_failures,
        "what_i_watched_fail": watched,
        "motive_not_rederived": {
            "text": (
                "A recent negative-science census over this project's history found closed results "
                "whose reopen conditions already hold — measured wins the project had stopped short "
                "of claiming. A retired campaign whose receipts nobody can find is how that happens. "
                "Ramanujan is indexed so it can be reopened deliberately rather than rediscovered by accident."
            ),
            "closest_named_register": GIT_ONLY["negative_science_register"],
        },
    }


def print_report(doc: dict[str, Any]) -> None:
    w = doc["what_it_was"]
    inv = doc["inventory"]["summary"]
    print("=== RAMANUJAN DISBAND ===")
    print(f"status: {doc['status']}")
    print(f"schema: {doc['schema']}")
    print(f"git HEAD: {doc['git']['head'][:12]} ({doc['git']['branch']}) {doc['git']['HEAD_subject']}")
    print()
    print("## 1. WHAT IT WAS")
    print(w["one_sentence"])
    print(f"  period: {w['period']['first_commit']['at']} → {w['period']['last_commit_touching_tree']['at']} "
          f"({w['period']['n_commits_touching_ramanujan']} commits)")
    print(f"  gate: {w['current_status']['gate']}")
    print(f"  green-light: {w['current_status']['green_light']}")
    print(f"  Q0: {w['current_status']['q0']}")
    print("  paths:")
    for k, p in w["paths"].items():
        print(f"    {k}: {p}")
    print("  relation:")
    for k, v in w["relation_to_current_line"].items():
        print(f"    {k}: {v}")
    print()
    print("## 2. ARTIFACT INVENTORY")
    print(f"  on-disk regular files: {inv['on_disk_regular_files']}")
    print(f"  on-disk symlinks:      {inv['on_disk_symlinks']}")
    print(f"  git ls-tree entries:   {inv['git_ls_tree_count']}")
    print(f"  total_bytes:           {inv['total_bytes']} ({inv['total_bytes'] / (1024*1024):.2f} MiB)")
    print(f"  tree_sha256:           {inv['tree_sha256']}")
    print(f"  git tree:              {inv['git_tree']}")
    print("  by_class:")
    for k, v in inv["by_class"].items():
        print(f"    {k:<22} {v['count']:4d} files  {v['bytes']:10d} bytes")
    print("  largest:")
    for row in doc["inventory"]["largest"][:8]:
        print(f"    {row['bytes']:10d}  {row['path']}")
    print("  git-only siblings:")
    for row in doc["inventory"]["git_only_siblings"]:
        flag = "on-disk" if row["on_disk"] else "git-only"
        digest = (row["sha256"] or "MISSING")[:16]
        print(f"    {flag:8} {row['bytes']}  {digest}  {row['git_path']}")
    print()
    print("## 3. MEASURED RESULTS")
    for i, row in enumerate(doc["measured"], 1):
        value = row["value"]
        if isinstance(value, dict):
            brief = json.dumps(value, sort_keys=True)
            if len(brief) > 240:
                brief = brief[:237] + "..."
        else:
            brief = repr(value)
        print(f"  {i:02d}. {row['claim']}")
        print(f"      {brief}")
        print(f"      receipt: {row['receipt']}")
    print()
    print("## 4. REFUTATIONS")
    for r in doc["refutations"]:
        print(f"  {r['id']}  [{r['classification']}]")
        print(f"    claim:  {r['claim']}")
        print(f"    result: {r['result']}")
        print(f"    why:    {r['why']}")
        print(f"    reopen: {r['reopen_condition']}")
        print(f"    receipts: {', '.join(r['receipts'])}")
    print()
    print("## 5. REUSABLE_NOW")
    if not doc["reusable_now"]:
        print("  (empty)")
    for item in doc["reusable_now"]:
        print(f"  - {item['item']}")
        print(f"      {item['what']}")
        print(f"      paths: {', '.join(item['paths'])}")
    print()
    print("## 6. DELETION COST")
    print(f"  {doc['deletion_cost']['what_this_receipt_saves_if_the_tree_goes']}")
    for item in doc["deletion_cost"]["if_ramanujan_deleted"]:
        if "lost" in item:
            print(f"  would lose: {item['lost']}")
        if "already_absent_even_without_deletion" in item:
            print("  already absent even without deletion:")
            for a in item["already_absent_even_without_deletion"]:
                print(f"    - {a}")
    prior = doc["deletion_cost"]["prior_deletion_already_happened"]
    print(f"  prior deletion: {prior['deleted']}")
    print(f"  restored:       {prior['restored']}")
    print()
    print("## 7. RAMANUJAN/ UNTOUCHED")
    u = doc["ramanujan_untouched"]
    print(f"  start porcelain: {u['start_porcelain'] or '(empty)'}")
    print(f"  end porcelain:   {u['end_porcelain'] or '(empty)'}")
    print(f"  tree sha start:  {u['start_tree_sha256']}")
    print(f"  tree sha end:    {u['end_tree_sha256']}")
    print(f"  unchanged:       {u['unchanged']}")
    for h in u["how_verified"]:
        print(f"  - {h}")
    print()
    print("## WHAT I WATCHED FAIL")
    for wfail in doc["what_i_watched_fail"]:
        print(f"  {wfail['id']}: {wfail.get('result')}")
        print(f"    {wfail['what']}")
        extra = {k: v for k, v in wfail.items() if k not in {"id", "what", "result", "named"}}
        brief = json.dumps(extra, sort_keys=True, default=str)
        if len(brief) > 400:
            brief = brief[:397] + "..."
        print(f"    {brief}")
    if doc["lock_failures"]:
        print()
        print("LOCK FAILURES:")
        for f in doc["lock_failures"]:
            print(f"  - {f}")
    print()
    print(f"-> {RECEIPT_PATH.relative_to(REPO)}")


def main() -> int:
    if not RAMANUJAN.is_dir():
        print("research/ramanujan/ is not on disk in this worktree", file=sys.stderr)
        return 2
    start_porcelain = porcelain("ramanujan")
    inv = inventory()
    start_tree = inv["tree_sha256"]
    live = load_live()
    lock_failures: list[str] = []
    lock_or_fail(live, lock_failures)
    if inv["on_disk_entries"] != inv["git_ls_tree_count"]:
        lock_failures.append(
            f"inventory entries {inv['on_disk_entries']} != git ls-tree {inv['git_ls_tree_count']}"
        )
    watched = probe_failures(inv, live)
    doc = assemble(inv, live, watched, start_porcelain, start_tree, lock_failures)
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Re-inventory after write: research/ramanujan/ must not have moved.
    end_inv = inventory()
    if end_inv["tree_sha256"] != start_tree:
        print("ERROR: research/ramanujan/ tree hash changed while writing the receipt", file=sys.stderr)
        return 1
    if porcelain("ramanujan") != start_porcelain:
        print("ERROR: research/ramanujan/ git status changed while writing the receipt", file=sys.stderr)
        return 1
    # Reload receipt and confirm required sections.
    saved = load_json(RECEIPT_PATH)
    for key in (
        "schema", "what_it_was", "inventory", "measured", "refutations",
        "reusable_now", "deletion_cost", "ramanujan_untouched", "what_i_watched_fail",
    ):
        if key not in saved:
            print(f"ERROR: receipt missing {key}", file=sys.stderr)
            return 1
    if saved["schema"] != SCHEMA:
        print("ERROR: schema mismatch", file=sys.stderr)
        return 1
    if not saved["reusable_now"]:
        # Empty is allowed if that is the truth; we have items, so this would be a bug.
        pass
    print_report(saved)
    if lock_failures:
        return 1
    if not saved["ramanujan_untouched"]["unchanged"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
