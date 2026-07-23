#!/usr/bin/env python3.12
"""hawking.glm52.source_release.v1 - release the closed GLM-5.2 BF16 body.

GLM-5.2 is sealed FUNCTIONAL_PARTIAL_ONLY. The 405.4 GB BF16 source at
``~/Library/Application Support/Hawking/GLM52Gravity/source`` is a re-fetchable,
content-addressable checkpoint pinned to an immutable Hugging Face revision, and
every scientific claim that depended on it has been distilled into sealed local
evidence. This module releases that body so the next parent can be admitted under
the one-parent storage law.

Two commands, deliberately separated so nothing is deleted by the same call that
decides deletion is safe:

    rehydrate   fetch the immutable HF blobs manifest live (metadata only) and
                seal a rehydration receipt: repo, revision, and a git-lfs sha256
                per resident shard, cross-checked against the local fetch ledger.
    gate        READ-ONLY. Run every preservation and safety gate, fresh-process
                verify the evidence that must outlive the source, scan the live
                process tree for any map/dep on the source, and emit the exact
                deletion path as DATA. Never deletes.
    release     Re-run the gate. Refuse unless every gate is green AND --confirm
                is passed. Delete ONLY the exact source root. Seal a
                reclaimed-byte receipt.

The gate is green only if its probe establishes the condition now. A missing
capsule, a codec selftest that will not pass in a fresh process, a live reader of
the source: any of these is red and blocks release.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

SCHEMA = "hawking.glm52.source_release.v1"
GB = 10 ** 9
GIB = 1024 ** 3

CONDENSE = Path(__file__).resolve().parent
REPO_ROOT = CONDENSE.parents[1]
GEN_B = REPO_ROOT / "reports" / "condense" / "glm52_generation_b"

SUPPORT = Path(os.environ.get(
    "GLM52_SUPPORT_ROOT",
    "/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity"))
SOURCE_ROOT = SUPPORT / "source"
FETCH = SUPPORT / "source_fetch"
CAPSULES = FETCH / "teacher" / "capsules_generation_b"
FUNCTIONAL_ARTIFACTS = SUPPORT / "compact" / "generation_b_functional"

REPO = "zai-org/GLM-5.2"
REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
BLOBS_API = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"

MANIFEST = REPO_ROOT / "GLM52_OFFICIAL_MANIFEST.json"
CORRECTION_LEDGER = REPO_ROOT / "GLM52_METRIC_CORRECTION_LEDGER.jsonl"
METRIC_CONTRACT = REPO_ROOT / "HAWKING_NULL_CORRECTED_METRIC_CONTRACT.json"
DECISION = REPO_ROOT / "GLM52_FUNCTIONAL_DECISION.json"
CPU_PARITY = REPO_ROOT / "GLM52_FUNCTIONAL_CPU_PARITY.json"
TRANSFER_PACKET = GEN_B / "GLM52_NEXT_PARENT_TRANSFER.md"

REHYDRATION_RECEIPT = REPO_ROOT / "GLM52_REHYDRATION_RECEIPT.json"
READINESS = REPO_ROOT / "GLM52_SOURCE_RELEASE_READINESS.json"
RELEASE_RECEIPT = REPO_ROOT / "GLM52_SOURCE_RELEASE_RECEIPT.json"

AMPLIFICATION = [GEN_B / f"GLM52_FUNCTIONAL_DEPTH_THRESHOLD_{s}.json"
                 for s in ("L03", "L38", "L74")]
# The minimal runtime fixtures a fresh reader needs to reconstruct the functional
# result without the source: the payload contract, the codec, and the CPU authority.
RUNTIME_FIXTURES = [
    CONDENSE / "gravity_functional_codec.py",
    CONDENSE / "gravity_functional_metal.py",
    REPO_ROOT / "GLM52_FUNCTIONAL_STUDENT_CONTRACT.json",
    REPO_ROOT / "GRAVITY_FUNCTIONAL_CODEC_SPEC.md",
]

VENV_PY = REPO_ROOT / ".venv" / "glm52" / "bin" / "python"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------- rehydrate


def rehydrate() -> dict:
    """Seal the rehydration route from the immutable HF blobs manifest, live.

    The HF manifest is the authoritative content-addressable index: it lists a
    git-lfs sha256 for every file at the pinned revision, so anyone can re-fetch
    the body byte-exact and verify it. Resident shards are cross-checked against
    the local fetch ledger where it recorded a sha256.
    """
    request = urllib.request.Request(BLOBS_API, headers={"User-Agent": "hawking-release"})
    with urllib.request.urlopen(request, timeout=60) as response:
        manifest = json.loads(response.read())

    lfs = {}
    for entry in manifest.get("siblings", []):
        name = entry.get("rfilename")
        blob = entry.get("lfs") or {}
        sha = blob.get("sha256") or blob.get("oid")
        if name and sha:
            lfs[name] = {"sha256": sha, "size": blob.get("size")}

    ledger_sha = {}
    ledger = FETCH / "SOURCE_FETCH_LEDGER.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("shard") and row.get("sha256"):
                ledger_sha[row["shard"]] = row["sha256"]

    resident = sorted(p.name for p in SOURCE_ROOT.glob("*.safetensors")) \
        if SOURCE_ROOT.exists() else []
    covered = [name for name in resident if name in lfs]
    # git-lfs sha256 is over the file content; the local ledger sha256 is over the
    # same bytes, so where both exist they must agree.
    ledger_conflicts = [name for name in resident
                        if name in ledger_sha and name in lfs
                        and ledger_sha[name] != lfs[name]["sha256"]]

    receipt = {
        "schema": "hawking.glm52.rehydration_receipt.v1",
        "sealed_at": _now(),
        "repo": REPO,
        "revision": REVISION,
        "immutable_tree_url": f"https://huggingface.co/{REPO}/tree/{REVISION}",
        "blobs_api_url": BLOBS_API,
        "route": "huggingface content-addressable git-lfs; re-fetch at the pinned "
                 "revision and verify each file against its sha256",
        "manifest_files_total": len(lfs),
        "resident_shards": len(resident),
        "resident_shards_with_authoritative_sha256": len(covered),
        "all_resident_shards_rehydratable": len(covered) == len(resident) and bool(resident),
        "ledger_cross_check_conflicts": ledger_conflicts,
        "per_file_sha256": {name: lfs[name] for name in sorted(lfs)},
    }
    receipt["seal_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in receipt.items() if k != "seal_sha256"},
                   sort_keys=True).encode()).hexdigest()
    REHYDRATION_RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return {"resident": len(resident), "covered": len(covered),
            "conflicts": ledger_conflicts, "receipt": str(REHYDRATION_RECEIPT)}


# --------------------------------------------------------------------------- process scan


def _process_map_scan() -> dict:
    """Two independent probes for anyone reading the source root.

    lsof enumerates open file descriptors and memory maps; the argv scan catches
    a process that names the source on its command line even if it has not opened
    a file yet. A reference from either is a red gate.
    """
    target = str(SOURCE_ROOT)
    findings = {"lsof": {"available": False}, "argv": {"available": False}, "matches": []}

    lsof = shutil.which("lsof")
    if lsof:
        try:
            out = subprocess.run([lsof, "--", target], capture_output=True, text=True,
                                 timeout=60)
            # lsof exits 1 when nothing has the path open, which is the green case.
            lines = [ln for ln in out.stdout.splitlines()[1:] if ln.strip()]
            findings["lsof"] = {"available": True, "open_references": len(lines)}
            findings["matches"] += [{"probe": "lsof", "line": ln[:200]} for ln in lines]
        except Exception as error:  # noqa: BLE001
            findings["lsof"] = {"available": False, "error": repr(error)[:200]}

    try:
        out = subprocess.run(["ps", "-Axww", "-o", "pid=,command="],
                             capture_output=True, text=True, timeout=60)
        hits = [ln for ln in out.stdout.splitlines()
                if target in ln and "glm52_source_release" not in ln]
        findings["argv"] = {"available": True, "referencing_processes": len(hits)}
        findings["matches"] += [{"probe": "argv", "line": ln.strip()[:200]} for ln in hits]
    except Exception as error:  # noqa: BLE001
        findings["argv"] = {"available": False, "error": repr(error)[:200]}

    findings["clean"] = (findings.get("lsof", {}).get("open_references", 0) == 0
                         and findings.get("argv", {}).get("referencing_processes", 0) == 0
                         and not findings["matches"])
    findings["any_probe_ran"] = (findings["lsof"].get("available")
                                 or findings["argv"].get("available"))
    return findings


def _fresh(argv: list[str]) -> tuple[bool, str]:
    """Run a verification in a fresh interpreter, so nothing this process cached counts."""
    python = str(VENV_PY) if VENV_PY.exists() else sys.executable
    try:
        out = subprocess.run([python, *argv], capture_output=True, text=True, timeout=600,
                             cwd=str(REPO_ROOT))
        return out.returncode == 0, (out.stdout + out.stderr)[-600:]
    except Exception as error:  # noqa: BLE001
        return False, repr(error)[:300]


# --------------------------------------------------------------------------- gates


def _gate(status: bool, reason: str, **extra) -> dict:
    return {"status": "green" if status else "red", "reason": reason, **extra}


def g01_source_root_exact() -> dict:
    if not SOURCE_ROOT.exists():
        return _gate(False, "source root does not exist; already released?",
                     path=str(SOURCE_ROOT))
    shards = list(SOURCE_ROOT.glob("*.safetensors"))
    total = sum(p.stat().st_size for p in SOURCE_ROOT.rglob("*") if p.is_file())
    return _gate(
        bool(shards) and total > 300 * GB,
        f"source root is a directory of {len(shards)} shards, {total/GB:.1f} GB",
        path=str(SOURCE_ROOT), bytes=total, gib=round(total / GIB, 1),
        gb=round(total / GB, 1), shards=len(shards))


def g02_immutable_revision_manifest() -> dict:
    if not MANIFEST.exists():
        return _gate(False, "official manifest missing")
    manifest = json.loads(MANIFEST.read_text())
    ok = manifest.get("revision") == REVISION and manifest.get("repo") == REPO
    return _gate(ok, f"manifest pins {manifest.get('repo')} @ {manifest.get('revision')}",
                 sealed=bool(manifest.get("seal_sha256")))


def g03_rehydration_receipt() -> dict:
    if not REHYDRATION_RECEIPT.exists():
        return _gate(False, "rehydration receipt not built; run `rehydrate` first")
    receipt = json.loads(REHYDRATION_RECEIPT.read_text())
    ok = (receipt.get("revision") == REVISION
          and receipt.get("all_resident_shards_rehydratable")
          and not receipt.get("ledger_cross_check_conflicts"))
    return _gate(
        bool(ok),
        f"{receipt.get('resident_shards_with_authoritative_sha256')} of "
        f"{receipt.get('resident_shards')} resident shards carry an authoritative "
        f"HF sha256; {len(receipt.get('ledger_cross_check_conflicts', []))} ledger conflicts",
        conflicts=receipt.get("ledger_cross_check_conflicts", []))


def g04_teacher_capsules_preserved() -> dict:
    if not CAPSULES.exists():
        return _gate(False, "capsule directory missing")
    capsules = list(CAPSULES.rglob("*.npz"))
    if not capsules:
        return _gate(False, "no capsules present")
    # A capsule must actually load in a fresh process, not merely exist on disk.
    sample = sorted(capsules)[0]
    ok, detail = _fresh(["-c",
                         f"import numpy as np; d=np.load(r'{sample}'); "
                         f"assert len(d.files)>0; print('capsule_ok', len(d.files))"])
    outside = SOURCE_ROOT not in sample.parents
    return _gate(ok and outside,
                 f"{len(capsules)} capsules preserved outside the source root; "
                 f"fresh-load {'ok' if ok else 'FAILED'}",
                 capsules=len(capsules), sample_loads=ok, detail=detail if not ok else "")


def g05_metric_contract_and_ledger() -> dict:
    if not (METRIC_CONTRACT.exists() and CORRECTION_LEDGER.exists()):
        return _gate(False, "metric contract or correction ledger missing")
    rows = [ln for ln in CORRECTION_LEDGER.read_text().splitlines() if ln.strip()]
    return _gate(len(rows) >= 22,
                 f"metric contract present; correction ledger has {len(rows)} sealed rows",
                 correction_rows=len(rows))


def g06_functional_fixtures() -> dict:
    payloads = list(FUNCTIONAL_ARTIFACTS.glob("*.gravity")) \
        if FUNCTIONAL_ARTIFACTS.exists() else []
    ok, detail = _fresh(["tools/condense/gravity_functional_codec.py", "selftest"])
    contract_ok = (REPO_ROOT / "GLM52_FUNCTIONAL_STUDENT_CONTRACT.json").exists()
    return _gate(ok and contract_ok,
                 f"codec selftest {'passes' if ok else 'FAILS'} fresh; "
                 f"{len(payloads)} sealed .gravity payloads; contract present",
                 gravity_payloads=len(payloads), detail=detail if not ok else "")


def g07_parity_fixtures() -> dict:
    if not CPU_PARITY.exists():
        return _gate(False, "CPU parity fixture missing")
    ok, detail = _fresh(["tools/condense/gravity_functional_metal.py", "selftest"])
    return _gate(ok, f"CPU/Metal parity re-verified fresh: {'pass' if ok else 'FAIL'}",
                 detail=detail if not ok else "")


def g08_amplification_evidence() -> dict:
    present = [p for p in AMPLIFICATION if p.exists()]
    if len(present) != len(AMPLIFICATION):
        return _gate(False, f"only {len(present)}/{len(AMPLIFICATION)} strata present")
    expansive = all(json.loads(p.read_text())
                    .get("stack_is_expansive_at_every_tested_magnitude")
                    for p in present)
    return _gate(expansive,
                 "early/middle/late amplification evidence present and expansive",
                 strata=[p.name for p in present])


def g09_runtime_gravity_fixture() -> dict:
    """Build and verify a minimal functional .gravity shard in a fresh process.

    This proves the runtime path still reconstructs the functional organ from a
    seed and a readout with the source gone, which is the whole point of keeping
    the fixtures instead of the weights.
    """
    ok, detail = _fresh(["-c",
                         "import sys; sys.path.insert(0,'tools/condense'); "
                         "import gravity_functional_codec as c, numpy as np, tempfile; "
                         "from pathlib import Path; "
                         "b=c.serialize(np.random.default_rng(0).standard_normal((64,32))"
                         ".astype(np.float32), seed=17, width=48, layer=38); "
                         "d=tempfile.mkdtemp(); p=Path(d)/'f.gravity'; "
                         "c.write_shard(p,[(38,b)],model={'repo':'probe'}); "
                         "r=c.verify(p); assert r['verified'] and r['all_deterministic']; "
                         "print('gravity_ok')"])
    return _gate(ok, f"minimal .gravity functional fixture builds and verifies fresh: "
                 f"{'ok' if ok else 'FAIL'}", detail=detail if not ok else "")


def g10_transfer_packet_rollback() -> dict:
    packet = TRANSFER_PACKET.exists()
    decision = DECISION.exists() and json.loads(DECISION.read_text()).get(
        "closure", {}).get("state") == "FINAL_SEALED"
    return _gate(packet and decision,
                 f"transfer packet {'present' if packet else 'MISSING'}; "
                 f"decision closure {'FINAL_SEALED' if decision else 'not final'}",
                 rollback="re-fetch at the immutable revision per the rehydration receipt")


def g11_no_process_maps_source() -> dict:
    scan = _process_map_scan()
    if not scan["any_probe_ran"]:
        return {"status": "pending", "reason": "no process probe could run", "scan": scan}
    return _gate(scan["clean"],
                 "no live process opens or names the source root" if scan["clean"]
                 else f"{len(scan['matches'])} live reference(s) to the source",
                 scan=scan)


def g12_isolation() -> dict:
    """The delete target must be the source and nothing else we must keep."""
    protected = [REPO_ROOT, CAPSULES, SUPPORT / "compact", FUNCTIONAL_ARTIFACTS,
                 Path.home() / "Downloads" / "mop",
                 Path.home() / ".cache" / "huggingface"]
    inside = [str(p) for p in protected if SOURCE_ROOT == p or SOURCE_ROOT in p.parents
              or p == SOURCE_ROOT]
    contains = [str(p) for p in protected
                if p != SOURCE_ROOT and (p == SOURCE_ROOT or SOURCE_ROOT in p.parents)]
    ok = (SOURCE_ROOT.name == "source"
          and SUPPORT in SOURCE_ROOT.parents
          and REPO_ROOT not in SOURCE_ROOT.parents
          and not inside and not contains)
    return _gate(ok,
                 "source root is isolated from repo, MOP, capsules, compact and evidence",
                 target=str(SOURCE_ROOT), protected_not_under_target=ok)


def g13_deletion_paths_listed() -> dict:
    return _gate(True, "the deletion set is exactly the source root and nothing else",
                 deletion_paths=[str(SOURCE_ROOT)],
                 reclaims_gb=round(sum(p.stat().st_size for p in SOURCE_ROOT.rglob("*")
                                       if p.is_file()) / GB, 1) if SOURCE_ROOT.exists() else 0)


GATES: list[tuple[str, Callable[[], dict]]] = [
    ("g01_source_root_exact", g01_source_root_exact),
    ("g02_immutable_revision_manifest", g02_immutable_revision_manifest),
    ("g03_rehydration_receipt", g03_rehydration_receipt),
    ("g04_teacher_capsules_preserved", g04_teacher_capsules_preserved),
    ("g05_metric_contract_and_ledger", g05_metric_contract_and_ledger),
    ("g06_functional_fixtures", g06_functional_fixtures),
    ("g07_parity_fixtures", g07_parity_fixtures),
    ("g08_amplification_evidence", g08_amplification_evidence),
    ("g09_runtime_gravity_fixture", g09_runtime_gravity_fixture),
    ("g10_transfer_packet_rollback", g10_transfer_packet_rollback),
    ("g11_no_process_maps_source", g11_no_process_maps_source),
    ("g12_isolation", g12_isolation),
    ("g13_deletion_paths_listed", g13_deletion_paths_listed),
]


def gate() -> dict:
    results = {name: probe() for name, probe in GATES}
    greens = sum(1 for r in results.values() if r["status"] == "green")
    all_green = all(r["status"] == "green" for r in results.values())
    report = {
        "schema": SCHEMA,
        "evaluated_at": _now(),
        "source_root": str(SOURCE_ROOT),
        "repo": REPO,
        "revision": REVISION,
        "gates": results,
        "green": greens,
        "total": len(results),
        "all_green": all_green,
        "release_authorized": all_green,
    }
    READINESS.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


# --------------------------------------------------------------------------- release


def release(confirm: bool) -> dict:
    report = gate()
    if not report["all_green"]:
        red = [n for n, r in report["gates"].items() if r["status"] != "green"]
        raise SystemExit(f"release refused: gates not green: {red}")
    if not confirm:
        raise SystemExit("release refused: pass --confirm to perform the deletion")
    if not SOURCE_ROOT.exists():
        raise SystemExit("release refused: source root already gone")

    before = sum(p.stat().st_size for p in SOURCE_ROOT.rglob("*") if p.is_file())
    shard_count = len(list(SOURCE_ROOT.glob("*.safetensors")))
    disk_before = shutil.disk_usage(str(SUPPORT)).free

    # Delete only the exact source root.
    shutil.rmtree(SOURCE_ROOT)

    disk_after = shutil.disk_usage(str(SUPPORT)).free
    receipt = {
        "schema": "hawking.glm52.source_release_receipt.v1",
        "released_at": _now(),
        "deleted_path": str(SOURCE_ROOT),
        "deleted_exists_after": SOURCE_ROOT.exists(),
        "shards_deleted": shard_count,
        "reclaimed_bytes_measured": before,
        "reclaimed_gb": round(before / GB, 1),
        "reclaimed_gib": round(before / GIB, 1),
        "free_before_bytes": disk_before,
        "free_after_bytes": disk_after,
        "free_delta_gb": round((disk_after - disk_before) / GB, 1),
        "free_after_gib": round(disk_after / GIB, 1),
        "rehydration_receipt": str(REHYDRATION_RECEIPT),
        "rollback": f"re-fetch {REPO} @ {REVISION} and verify against the rehydration receipt",
        "readiness_seal": hashlib.sha256(READINESS.read_bytes()).hexdigest(),
        "capsules_retained": len(list(CAPSULES.rglob("*.npz"))) if CAPSULES.exists() else 0,
        "preserved_and_untouched": [
            "teacher capsules", "compact fixtures", "metric contract",
            "correction ledger", "functional payloads", "parity fixtures",
            "amplification evidence", "transfer packet", "repository", "MOP"],
    }
    receipt["seal_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in receipt.items() if k != "seal_sha256"},
                   sort_keys=True).encode()).hexdigest()
    RELEASE_RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def _print_summary(report: dict) -> None:
    for name, r in report["gates"].items():
        mark = {"green": "OK ", "red": "RED", "pending": "?? "}[r["status"]]
        print(f"  {mark} {name}: {r['reason']}")
    print(f"\n{report['green']}/{report['total']} green; "
          f"release_authorized={report['release_authorized']}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if command == "rehydrate":
        print(json.dumps(rehydrate(), indent=2))
    elif command == "gate":
        report = gate()
        _print_summary(report)
        raise SystemExit(0 if report["all_green"] else 1)
    elif command == "release":
        print(json.dumps(release("--confirm" in sys.argv), indent=2))
    else:
        raise SystemExit(f"unknown command: {command}")
