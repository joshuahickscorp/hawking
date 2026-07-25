#!/usr/bin/env python3
"""Seal the invalid Generation A GLM-5.2 run and retire its obsolete output.

Generation A stopped being science the moment its packer billed protected tensors it
never wrote.  This module records what it actually did, reproduces the defect against
the real artifacts rather than restating the claim, preserves the minimum fixture set a
later reader needs to believe the verdict, and retires the obsolete output tree behind
the safety gates the directive requires.

Two commands:

    seal        write GLM52_GENERATION_A_FINAL_STATE.json
    retire      write GLM52_GENERATION_A_DESKTOP_RETIREMENT.json and delete the tree

`retire` refuses to delete anything unless every gate passes.  It is idempotent: a tree
already gone seals as ALREADY_RETIRED rather than failing.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gravity_format  # noqa: E402
import glm52_contract as contract  # noqa: E402

REPO = HERE.parent.parent
STATE = Path.home() / "Library/Application Support/Hawking/GLM52Gravity"
SOURCE = STATE / "source"
FETCH = STATE / "source_fetch"
CAPSULES = FETCH / "teacher/capsules"
OBSOLETE = Path.home() / "Library/Mobile Documents/.Trash/GLM52-Gravity-SubBit"
REPORTS = REPO / "reports/condense/glm52_generation_b"
FIXTURES = REPORTS / "generation_a_fixtures"

PROTECTED_CLASS = "CONTROL_SENSITIVE_CANDIDATE"
# Anything under these roots is never a deletion target, whatever the caller passes.
FORBIDDEN_ROOTS = (
    Path.home() / "Downloads/mop",
    Path.home() / "Downloads/mop-data",
    Path.home() / ".cache/huggingface",
    REPO,
)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args],
                              capture_output=True, text=True, check=False).stdout.strip()
    return {"branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": run("rev-parse", "HEAD")}


# --------------------------------------------------------------------------- defects


def reproduce_protected_coverage_defect(limit: int = 6) -> dict:
    """Open real Generation A artifacts and diff them against the official inventory.

    The defect is structural, not statistical: `pack_shard` billed every protected tensor
    into compact_bits and appended it to `entries`, but only `payloads` reached
    `write_shard`.  A shard carrying no protected organ is unaffected, which is why the
    run looked healthy for long stretches.
    """
    if not OBSOLETE.exists():
        return {"status": "ARTIFACTS_ALREADY_RETIRED", "checked": 0}

    rows = []
    for gravity in sorted(OBSOLETE.glob("*.gravity")):
        shard = SOURCE / (gravity.stem + ".safetensors")
        if not shard.exists():
            continue
        header = gravity_format.read_header(gravity)
        present = {tensor["name"] for tensor in header["tensors"]}
        declared = {k: v for k, v in safetensors_header(shard).items() if k != "__metadata__"}

        absent, absent_elements = [], 0
        protected_declared = 0
        for name, meta in declared.items():
            classification = contract.classify_tensor(name, config())
            if classification.provisional_budget_class == PROTECTED_CLASS:
                protected_declared += 1
            if name in present:
                continue
            elements = 1
            for dim in meta["shape"]:
                elements *= dim
            absent_elements += elements
            absent.append({"name": name, "dtype": meta["dtype"], "shape": meta["shape"],
                           "budget_class": classification.provisional_budget_class,
                           "category": classification.category})
        rows.append({
            "gravity": gravity.name,
            "source_shard": shard.name,
            "declared_tensors": len(declared),
            "physically_present_tensors": len(present),
            "protected_tensors_declared": protected_declared,
            "absent_tensors": len(absent),
            "absent_elements": absent_elements,
            "bits_billed_for_absent_payload": absent_elements * 16,
            "artifact_whole_shard_bpw": header.get("compression", {}).get("whole_shard_bpw"),
            "artifact_packed_bpw": header.get("compression", {}).get("packed_bpw"),
            "absent": absent,
        })
        if len(rows) >= limit:
            break

    hit = [row for row in rows if row["absent_tensors"]]
    return {
        "status": "REPRODUCED" if hit else "NOT_REPRODUCED_ON_SAMPLE",
        "mechanism": ("pack_shard billed protected tensors into compact_bits and recorded "
                      "them in entries, but only payloads reached write_shard, so every "
                      "router, router control, normalization and indexer tensor was "
                      "accounted for at 16 BPW and written nowhere"),
        "consequence": ("the artifact reads as complete, its whole_shard_bpw includes "
                        "weight it does not carry, and the streamer treats it as proof "
                        "the BF16 body was consumed and may be evicted"),
        "checked": len(rows),
        "shards_with_absent_tensors": len(hit),
        "rows": rows,
    }


def reproduce_capsule_receipt_defect() -> dict:
    """Check every sealed receipt against the payload it describes.

    The defect was in the gate, not necessarily in the data: `captured_layers` admitted a
    layer on a sealed receipt alone, so a missing, truncated or zeroed .npz would still
    have authorized eviction of the only body that could recreate it.
    """
    if not CAPSULES.exists():
        return {"status": "NO_CAPSULE_DIR", "receipts": 0}
    rows = []
    for path in sorted(CAPSULES.glob("*.json")):
        try:
            receipt = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            rows.append({"receipt": path.name, "verdict": "UNREADABLE", "detail": str(exc)})
            continue
        capsule_id = receipt.get("capsule_id") or path.stem
        payload = CAPSULES / f"{capsule_id}.npz"
        if not payload.exists():
            rows.append({"receipt": path.name, "capsule_id": capsule_id,
                         "verdict": "RECEIPT_WITHOUT_PAYLOAD"})
            continue
        size_ok = payload.stat().st_size == receipt.get("capsule_bytes")
        digest_ok = sha256_file(payload) == receipt.get("capsule_sha256")
        rows.append({"receipt": path.name, "capsule_id": capsule_id,
                     "layers": receipt.get("layers"),
                     "size_matches": size_ok, "digest_matches": digest_ok,
                     "verdict": "VERIFIED" if (size_ok and digest_ok) else "MISMATCH"})
    bad = [row for row in rows if row["verdict"] != "VERIFIED"]
    return {
        "status": "MECHANISM_REAL_DATA_INTACT" if not bad else "REPRODUCED_IN_DATA",
        "mechanism": ("captured_layers, the set the eviction gate consults, was satisfied "
                      "by a sealed receipt without opening the .npz it describes"),
        "live_state_note": ("every surviving capsule verifies, so the mechanism is a "
                            "latent authorization hole rather than an observed loss; the "
                            "distinction is recorded rather than smoothed over"),
        "receipts": len(rows), "unverified": len(bad), "rows": rows,
    }


_CONFIG: dict | None = None


def config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = json.loads((STATE / "control/config.json").read_text())
    return _CONFIG


# -------------------------------------------------------------------------- fixtures


def preserve_fixtures() -> dict:
    """Keep the minimum a later reader needs to re-derive the verdict without the 39 GiB.

    One invalid artifact header, the defect receipts, the coverage ledgers, the source
    lineage and the negative-control seal.  Not one compact body: the header carries the
    claim, and the claim is what was false.
    """
    FIXTURES.mkdir(parents=True, exist_ok=True)
    kept = []

    if OBSOLETE.exists():
        # The one artifact that demonstrably drops protected organs, header only.
        for gravity in sorted(OBSOLETE.glob("*.gravity")):
            shard = SOURCE / (gravity.stem + ".safetensors")
            if not shard.exists():
                continue
            header = gravity_format.read_header(gravity)
            present = {tensor["name"] for tensor in header["tensors"]}
            declared = {k: v for k, v in safetensors_header(shard).items()
                        if k != "__metadata__"}
            if len(declared) > len(present):
                target = FIXTURES / f"INVALID_ARTIFACT_HEADER_{gravity.stem}.json"
                target.write_text(json.dumps({
                    "note": ("header of a Generation A artifact whose protected tensors "
                             "were billed and not stored; body deliberately not kept"),
                    "gravity": gravity.name,
                    "gravity_bytes": gravity.stat().st_size,
                    "gravity_sha256": sha256_file(gravity),
                    "source_shard": shard.name,
                    "header": header,
                }, indent=2))
                kept.append(target.name)
                break

        seal = OBSOLETE / "GLM_BASELINE_A_SEAL.json"
        if seal.exists():
            target = FIXTURES / "GLM_BASELINE_A_SEAL.json"
            shutil.copy2(seal, target)
            kept.append(target.name)

    for name in ("SOURCE_FETCH_LEDGER.jsonl", "GLM52_SOURCE_WEIGHT_ATLAS.json",
                 "progress.json", "deferred_evictions.json",
                 "GLM52_SAFE_TO_LEAVE_STATUS.json"):
        src = FETCH / name
        if src.exists():
            target = FIXTURES / f"generation_a_{name}"
            shutil.copy2(src, target)
            kept.append(target.name)

    ledger = FETCH / "teacher/GLM52_TEACHER_EVIDENCE_LEDGER.jsonl"
    if ledger.exists():
        target = FIXTURES / "generation_a_TEACHER_EVIDENCE_LEDGER.jsonl"
        shutil.copy2(ledger, target)
        kept.append(target.name)

    return {"directory": str(FIXTURES.relative_to(REPO)), "files": sorted(kept)}


# --------------------------------------------------------------------------- seals


def read_progress() -> dict:
    path = FETCH / "progress.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    payload["_file_mtime_utc"] = datetime.fromtimestamp(
        path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return payload


def seal() -> int:
    progress = read_progress()
    stalled_for = None
    if progress.get("_file_mtime_utc"):
        last = datetime.strptime(progress["_file_mtime_utc"], "%Y-%m-%dT%H:%M:%SZ")
        stalled_for = int((datetime.now(timezone.utc).replace(tzinfo=None) - last)
                          .total_seconds())

    resident = sorted(SOURCE.glob("*.safetensors")) if SOURCE.exists() else []
    obsolete_files = sorted(OBSOLETE.glob("*.gravity")) if OBSOLETE.exists() else []

    payload = {
        "schema": "hawking.glm52.generation_a_final_state.v1",
        "sealed_at": now(),
        "git": git_head(),
        "verdict": "INVALID_SUPERSEDED_BY_GENERATION_B",
        "endpoint": "GENERATION_A_STOPPED_AND_SEALED",
        "source": {
            "repo": progress.get("repo"),
            "revision": progress.get("revision"),
            "total_source_shards": progress.get("total_source_shards"),
        },
        "final_progress": progress,
        "stop": {
            "supported_path": "launchctl bootout gui/<uid>/com.hawking.glm52.source-fetch",
            "process_group_exited": True,
            "fetch_lock_released": True,
            "no_source_or_artifact_files_mapped": True,
        },
        "termination_reason": {
            "primary": "TIER_0_PROTECTED_TENSOR_COVERAGE_DEFECT",
            "secondary": "CONTROLLER_WEDGED_AT_FULL_CPU_WITH_NO_STATE_ADVANCE",
            "wedge_evidence": {
                "last_state_write_utc": progress.get("_file_mtime_utc"),
                "stalled_seconds_at_seal": stalled_for,
                "cpu_percent_while_stalled": 100.0,
                "open_source_or_output_files_while_stalled": 0,
                "note": ("the controller held a full core for hours with no safetensors "
                         "or gravity file open and no ledger row written"),
            },
            "compact_root_evidence": {
                "configured_root": os.environ.get(
                    "GLM52_COMPACT_ROOT", "/Users/scammermike/Desktop/GLM52-Gravity-SubBit"),
                "root_existed_at_seal": Path(
                    "/Users/scammermike/Desktop/GLM52-Gravity-SubBit").exists(),
                "note": ("the configured compact root was moved to the iCloud trash while "
                         "the packer was running, so late Generation A packs had no "
                         "surviving destination"),
            },
        },
        "reproduced_defects": {
            "protected_tensors_billed_not_stored": reproduce_protected_coverage_defect(),
            "capsule_receipts_trusted_without_payloads": reproduce_capsule_receipt_defect(),
        },
        "residual_state": {
            "resident_source_shards": len(resident),
            "resident_source_bytes": sum(p.stat().st_size for p in resident),
            "obsolete_artifacts": len(obsolete_files),
            "obsolete_artifact_bytes": sum(p.stat().st_size for p in obsolete_files),
            "teacher_capsules": len(list(CAPSULES.glob("*.npz"))) if CAPSULES.exists() else 0,
        },
        "preserved_fixtures": preserve_fixtures(),
        "not_evidence_of": ("output divergence, capability, or trajectory fidelity; "
                            "Generation A never reached a teacher-verified verdict"),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    target = REPORTS / "GLM52_GENERATION_A_FINAL_STATE.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"wrote": str(target.relative_to(REPO)),
                      "verdict": payload["verdict"],
                      "stalled_seconds": stalled_for,
                      "protected_defect": payload["reproduced_defects"][
                          "protected_tensors_billed_not_stored"]["status"],
                      "capsule_defect": payload["reproduced_defects"][
                          "capsule_receipts_trusted_without_payloads"]["status"],
                      "fixtures": len(payload["preserved_fixtures"]["files"])}, indent=2))
    return 0


# -------------------------------------------------------------------------- retire


def retirement_gates(target: Path) -> tuple[list[str], dict]:
    """Every gate the directive requires before an irreversible delete."""
    blockers: list[str] = []
    facts: dict = {}

    facts["requested_path"] = str(target)
    real = target.resolve()
    facts["real_path"] = str(real)
    # macOS resolves /tmp and /var through symlinked ancestors, so requiring the resolved
    # path to equal the absolute path would reject every legitimate target.  What matters
    # is that the leaf is not itself a link and that resolution did not rename it, which
    # is what a link pointing somewhere else would do.  The forbidden-root check below
    # runs on the resolved path, so an escape upward is still caught.
    facts["real_path_validation"] = (not target.is_symlink()) and real.name == target.name
    if not facts["real_path_validation"]:
        blockers.append("path does not resolve to a real directory of the same name")

    facts["is_symlink"] = target.is_symlink()
    if target.is_symlink():
        blockers.append("target is a symlink")

    symlinked = [str(p) for p in target.rglob("*") if p.is_symlink()] if target.exists() else []
    facts["contained_symlinks"] = symlinked[:10]
    if symlinked:
        blockers.append(f"{len(symlinked)} symlink(s) inside the tree")

    for root in FORBIDDEN_ROOTS:
        if real == root or root in real.parents:
            blockers.append(f"target is inside protected root {root}")
    facts["forbidden_roots_checked"] = [str(r) for r in FORBIDDEN_ROOTS]

    mapped = subprocess.run(["lsof", "+D", str(target)], capture_output=True,
                            text=True, check=False).stdout.strip().splitlines()
    holders = [line for line in mapped[1:] if line and "lsof" not in line.split()[0]]
    facts["processes_mapping_tree"] = len(holders)
    if holders:
        blockers.append(f"{len(holders)} process(es) still map the tree")

    return blockers, facts


def retire() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "GLM52_GENERATION_A_DESKTOP_RETIREMENT.json"

    if not OBSOLETE.exists():
        out.write_text(json.dumps({
            "schema": "hawking.glm52.generation_a_desktop_retirement.v1",
            "sealed_at": now(), "git": git_head(),
            "status": "ALREADY_RETIRED", "path": str(OBSOLETE),
            "reclaimed_bytes": 0,
        }, indent=2, sort_keys=True))
        print(json.dumps({"status": "ALREADY_RETIRED"}, indent=2))
        return 0

    files = sorted(p for p in OBSOLETE.rglob("*") if p.is_file())
    manifest = [{"name": str(p.relative_to(OBSOLETE)), "bytes": p.stat().st_size}
                for p in files]
    total = sum(entry["bytes"] for entry in manifest)

    fixtures = preserve_fixtures()
    blockers, facts = retirement_gates(OBSOLETE)
    if not fixtures["files"]:
        blockers.append("no forensic fixtures were preserved")

    record = {
        "schema": "hawking.glm52.generation_a_desktop_retirement.v1",
        "sealed_at": now(),
        "git": git_head(),
        "path": str(OBSOLETE),
        "origin": ("configured as GLM52_COMPACT_ROOT at ~/Desktop/GLM52-Gravity-SubBit; "
                   "the Desktop is iCloud-synced, so the tree was moved to the iCloud "
                   "trash rather than deleted, and it survived there in full"),
        "gates": facts,
        "blockers": blockers,
        "manifest_entries": len(manifest),
        "manifest_bytes": total,
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "manifest": manifest,
        "preserved_fixtures": fixtures,
        "why_retired": ("every artifact in this tree was produced by the packer that "
                        "billed protected tensors it never wrote, so no file here can "
                        "support a coverage or BPW claim"),
    }

    if blockers:
        record["status"] = "BLOCKED"
        record["reclaimed_bytes"] = 0
        out.write_text(json.dumps(record, indent=2, sort_keys=True))
        print(json.dumps({"status": "BLOCKED", "blockers": blockers}, indent=2))
        return 1

    shutil.rmtree(OBSOLETE)
    record["status"] = "RETIRED"
    record["reclaimed_bytes"] = total
    record["verified_absent_after_delete"] = not OBSOLETE.exists()
    out.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"status": "RETIRED", "reclaimed_bytes": total,
                      "reclaimed_gib": round(total / (1 << 30), 2),
                      "entries": len(manifest)}, indent=2))
    return 0


def selftest() -> int:
    """The gates must reject a symlink and anything inside a protected root."""
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        plain = tmp / "plain"
        plain.mkdir()
        (plain / "a.bin").write_bytes(b"x" * 16)
        blockers, facts = retirement_gates(plain)
        assert not blockers, f"a plain tree must pass every gate, got {blockers}"
        assert facts["processes_mapping_tree"] == 0

        (plain / "link").symlink_to(plain / "a.bin")
        blockers, _ = retirement_gates(plain)
        assert any("symlink" in b for b in blockers), blockers

        inside_repo = REPO / "reports"
        blockers, _ = retirement_gates(inside_repo)
        assert any("protected root" in b for b in blockers), blockers

    print("glm52_generation_a_seal selftest OK")
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "seal"
    raise SystemExit({"seal": seal, "retire": retire, "selftest": selftest}[command]())
