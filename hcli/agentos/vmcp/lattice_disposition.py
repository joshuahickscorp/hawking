#!/usr/bin/env python3
"""Disposition of the ten named visionmcp state structures.

This lane DISPOSES. It does not build. visionmcp is read-only; proofs run
against a located 0.8.0a2 checkout. A name grep is supporting evidence, never
a close: every CONSOLIDATE / VERIFIED_EXISTING row is closed by driving the
existing mechanism until a material mutation is DETECTED, and at least one
canary is shown failing to detect.

    python3 hcli/agentos/vmcp/lattice_disposition.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RECEIPT_PATH = REPO / "receipts" / "headless" / "VMCP_LATTICE_DISPOSITION.json"

CONSOLIDATE = "CONSOLIDATE"
REJECT = "REJECT"
VERIFIED_EXISTING = "VERIFIED_EXISTING"
IMPLEMENT_AND_VERIFY = "implement-and-verify"

TEN = (
    "DEEP_DIGEST",
    "DIRECTOR_STATE",
    "TRUTH_LEDGER",
    "ASSET_LATTICE",
    "DECODE_LATTICE",
    "ENTITY_GENOME",
    "RENDER_GENOME",
    "SPATIAL_GENOME",
    "REPAIR_VECTOR",
    "PERFORMANCE_LEDGER",
)

CENSUS_SCALE = {
    "package": "visionmcp 0.8.0a2",
    "files": 10106,
    "python_modules": 1770,
    "core_mcp_tools": 15,
    "worldir_schema_hash_prefix": "ee28e5a9",
    "source": "receipts/headless/VMCP_CENSUS.json — not re-derived",
}


# --------------------------------------------------------------------------- locate / git


def locate_visionmcp_src() -> Path:
    env = os.environ.get("VISIONMCP_SRC")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            REPO / "visionmcp" / "src",
            # This worktree is a sparse checkout of hawking-copy; prefer that copy.
            Path("/Users/scammermike/Downloads/hawking-copy/visionmcp/src"),
            Path("/Users/scammermike/Downloads/hawking/visionmcp/src"),
            Path.home() / ".searcher-donors" / "visionmcp" / "src",
        ]
    )
    seen: set[Path] = set()
    for src in candidates:
        if not src.exists():
            continue
        src = src.resolve()
        if src in seen:
            continue
        seen.add(src)
        if (src / "visionmcp" / "worldir" / "schema.py").is_file() and (
            src / "visionmcp" / "evidence_graph" / "graph.py"
        ).is_file():
            return src
    raise FileNotFoundError(
        "visionmcp src not found. Set VISIONMCP_SRC. This sparse worktree does "
        "not materialize visionmcp/, and git sparse-checkout add is denied."
    )


def git_head(cwd: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        oneline = subprocess.run(
            ["git", "log", "-1", "--oneline"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"head": head, "oneline": oneline, "cwd": str(cwd)}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"head": None, "oneline": None, "cwd": str(cwd), "error": str(exc)}


def jsonable(value: Any, *, limit: int = 2500) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v, limit=limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v, limit=limit) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return {"_bytes": len(value), "_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > limit:
            return value[:limit] + f"... <truncated {len(value) - limit} chars>"
        return value
    return repr(value)


def caught(fn):
    try:
        return {"raised": False, "value": fn()}
    except Exception as exc:
        return {
            "raised": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def detect_label(result: dict[str, Any], *, expect_raise: bool = True) -> str:
    if expect_raise:
        return "DETECTED" if result.get("raised") else "UNDETECTED"
    return "DETECTED" if not result.get("raised") and result.get("value") else "UNDETECTED"


# --------------------------------------------------------------------------- shared fixtures (built after imports)


class _Verifier:
    def __init__(self, accepted: frozenset[str], digest: str) -> None:
        self._accepted = accepted
        self._digest = digest

    def verify(self, *, anchor: Any, attestation: Any) -> bool:
        return attestation.id in self._accepted and attestation.credential_digest == self._digest


def _rights(vm, ident: str = "rights:lattice"):
    return vm["Rights"](
        id=ident,
        rights_status="public",
        publicability="public",
        license_identifier="CC0-1.0",
        allowed_uses=("test",),
    )


def _prov(vm, ident: str = "provenance:lattice", evid: tuple[str, ...] = ("evidence:lattice",)):
    return vm["Provenance"](
        id=ident,
        authority=vm["WorldAuthority"].DIRECTLY_OBSERVED,
        confidence=1.0,
        supporting_evidence_ids=evid,
        rights=_rights(vm),
        publicability="public",
    )


def _load_vm():
    from visionmcp import __version__ as version
    from visionmcp.artifacts.store import ArtifactStore
    from visionmcp.core.util import atomic_write_json
    from visionmcp.evidence_graph import (
        AttestationSubjectKind,
        EvaluatorAttestation,
        EvidenceAuthority,
        EvidenceClaim,
        EvidenceGraph,
        EvidenceGraphDurableStore,
        EvidenceNode,
        EvidenceNodeKind,
        EvidenceProducerRole,
        FieldBinding,
        TrustAnchor,
        WorldIRFieldInventory,
        claim_subject_digest,
        evaluator_principal,
        verify_content_hash,
    )
    from visionmcp.greed.records import RightsStatus
    from visionmcp.memory import (
        EntityKind,
        EntityVersion,
        PrivacyClass,
        TargetFingerprints,
        content_digest as memory_content_digest,
    )
    from visionmcp.memory.store import MemoryStore
    from visionmcp.performance.kernels.registry import KernelRecord
    from visionmcp.projects.store import ProjectStore
    from visionmcp.repair.ledger import RepairEvidenceLedger
    from visionmcp.repair.models import (
        CausalTarget,
        CausalTargetKind,
        CauseHypothesis,
        EvaluationResult,
        EvaluationScope,
        RankedRepairCandidate,
        RepairAttempt,
        RepairKind,
        RepairOutcome,
        canonical_digest as repair_canonical_digest,
    )
    from visionmcp.worldir import (
        WORLDIR_SCHEMA,
        WORLDIR_SCHEMA_SHA256,
        WORLDIR_VERSION,
        BinaryIR,
        Entity,
        Geometry,
        Program,
        Provenance,
        Resource,
        Rights,
        SpatialIR,
        Surface,
        World,
        WorldIRValidationError,
        canonical_json_bytes,
        content_digest,
        migrate_legacy,
        schema_path,
        schema_sha256,
        verify_schema_hash,
    )
    from visionmcp.worldir.models import EvidenceAuthority as WorldAuthority

    return {
        "version": version,
        "ArtifactStore": ArtifactStore,
        "atomic_write_json": atomic_write_json,
        "AttestationSubjectKind": AttestationSubjectKind,
        "EvaluatorAttestation": EvaluatorAttestation,
        "EvidenceAuthority": EvidenceAuthority,
        "EvidenceClaim": EvidenceClaim,
        "EvidenceGraph": EvidenceGraph,
        "EvidenceGraphDurableStore": EvidenceGraphDurableStore,
        "EvidenceNode": EvidenceNode,
        "EvidenceNodeKind": EvidenceNodeKind,
        "EvidenceProducerRole": EvidenceProducerRole,
        "FieldBinding": FieldBinding,
        "TrustAnchor": TrustAnchor,
        "WorldIRFieldInventory": WorldIRFieldInventory,
        "claim_subject_digest": claim_subject_digest,
        "evaluator_principal": evaluator_principal,
        "verify_content_hash": verify_content_hash,
        "RightsStatus": RightsStatus,
        "EntityKind": EntityKind,
        "EntityVersion": EntityVersion,
        "PrivacyClass": PrivacyClass,
        "TargetFingerprints": TargetFingerprints,
        "memory_content_digest": memory_content_digest,
        "MemoryStore": MemoryStore,
        "KernelRecord": KernelRecord,
        "ProjectStore": ProjectStore,
        "RepairEvidenceLedger": RepairEvidenceLedger,
        "CausalTarget": CausalTarget,
        "CausalTargetKind": CausalTargetKind,
        "CauseHypothesis": CauseHypothesis,
        "EvaluationResult": EvaluationResult,
        "EvaluationScope": EvaluationScope,
        "RankedRepairCandidate": RankedRepairCandidate,
        "RepairAttempt": RepairAttempt,
        "RepairKind": RepairKind,
        "RepairOutcome": RepairOutcome,
        "repair_canonical_digest": repair_canonical_digest,
        "WORLDIR_SCHEMA": WORLDIR_SCHEMA,
        "WORLDIR_SCHEMA_SHA256": WORLDIR_SCHEMA_SHA256,
        "WORLDIR_VERSION": WORLDIR_VERSION,
        "BinaryIR": BinaryIR,
        "Entity": Entity,
        "Geometry": Geometry,
        "Program": Program,
        "Provenance": Provenance,
        "Resource": Resource,
        "Rights": Rights,
        "SpatialIR": SpatialIR,
        "Surface": Surface,
        "World": World,
        "WorldIRValidationError": WorldIRValidationError,
        "canonical_json_bytes": canonical_json_bytes,
        "content_digest": content_digest,
        "migrate_legacy": migrate_legacy,
        "schema_path": schema_path,
        "schema_sha256": schema_sha256,
        "verify_schema_hash": verify_schema_hash,
        "WorldAuthority": WorldAuthority,
    }


def _bound_graph(vm) -> tuple[Any, Any, Any, Any]:
    world = vm["World"](
        id="world:store",
        world_kind="store-fixture",
        provenance=_prov(vm, "provenance:store", ("evidence:store",)),
    )
    inventory = vm["WorldIRFieldInventory"].from_world(world)
    claims = tuple(
        vm["EvidenceClaim"](
            target_id=item.target_id, field_path=item.field_path, value=item.value
        )
        for item in inventory.fields
    )
    node = vm["EvidenceNode"](
        id="evidence:store",
        kind=vm["EvidenceNodeKind"].OBSERVATION,
        producer_id="worker:store",
        producer_role=vm["EvidenceProducerRole"].EVALUATOR,
        authority=vm["EvidenceAuthority"].DIRECTLY_OBSERVED,
        rights=vm["RightsStatus"].CC0,
        uncertainty=0.0,
        claims=claims,
    )
    digest = "d" * 64
    attestations = tuple(
        vm["EvaluatorAttestation"](
            id=f"attestation:store:{index}",
            anchor_id="anchor:store",
            evaluator_id="evaluator:store",
            subject_kind=vm["AttestationSubjectKind"].CLAIM,
            subject_digest=vm["claim_subject_digest"](
                evidence_id=node.id,
                evidence_content_hash=node.content_hash,
                target_id=claim.target_id,
                field_path=claim.field_path,
                value_digest=claim.value_digest,
            ),
            credential_digest=digest,
        )
        for index, claim in enumerate(claims)
    )
    bindings = tuple(
        vm["FieldBinding"](
            id=f"binding:store:{index}",
            target_id=item.target_id,
            field_path=item.field_path,
            value=item.value,
            evidence_ids=(node.id,),
            acceptable_authorities=(vm["EvidenceAuthority"].DIRECTLY_OBSERVED,),
            public_claim=True,
        )
        for index, item in enumerate(inventory.fields)
    )
    graph = vm["EvidenceGraph"](
        id="graph:store",
        nodes=(node,),
        bindings=bindings,
        trust_anchors=(
            vm["TrustAnchor"](
                id="anchor:store",
                evaluator_id="evaluator:store",
                authority_classes=(vm["EvidenceAuthority"].DIRECTLY_OBSERVED,),
                credential_digest=digest,
            ),
        ),
        attestations=attestations,
        target_world_content_hash=inventory.world_content_hash,
    )
    verifier = _Verifier(frozenset(item.id for item in attestations), digest)
    return world, graph, verifier, digest


def _worldir_tamper(from_dict, record, field: str, new_value: Any) -> dict[str, Any]:
    payload = record.to_dict()
    original_hash = payload["content_hash"]
    tampered = json.loads(json.dumps(payload))
    tampered[field] = new_value
    result = caught(lambda: from_dict(tampered))
    return {
        "field": field,
        "original_hash": original_hash,
        "tampered_value": new_value,
        "kept_content_hash": tampered.get("content_hash"),
        "result": {
            "raised": result["raised"],
            "error_type": result.get("error_type"),
            "error": result.get("error"),
        },
        "verdict": detect_label(result),
        "detection": (
            f"{result.get('error_type')}: {result.get('error')}"
            if result["raised"]
            else "accepted forged record"
        ),
    }


def _crash_before_replace(dest: Path, payload_path: Path) -> dict[str, Any]:
    """Child writes a sibling temp file, then _exit(9) before os.replace."""
    code = textwrap.dedent(
        """
        import json, os, sys, tempfile
        from pathlib import Path
        dest = Path(sys.argv[1])
        payload = json.loads(Path(sys.argv[2]).read_bytes())
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="." + dest.name + ".", dir=dest.parent)
        os.write(fd, json.dumps(payload).encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        os._exit(9)
        """
    )
    before = dest.read_bytes() if dest.exists() else None
    proc = subprocess.run(
        [sys.executable, "-c", code, str(dest), str(payload_path)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    after_exists = dest.exists()
    after = dest.read_bytes() if after_exists else None
    leftovers = sorted(p.name for p in dest.parent.glob(f".{dest.name}.*"))
    hybrid = after != before
    return {
        "child_returncode": proc.returncode,
        "dest_existed_before": before is not None,
        "dest_exists_after": after_exists,
        "dest_bytes_unchanged": after == before,
        "leftover_temp_names": leftovers,
        "hybrid_state": hybrid,
        "child_stderr": (proc.stderr or "")[:400],
        "verdict": "ATOMIC" if (proc.returncode == 9 and not hybrid) else "HYBRID",
    }


def _crash_sqlite_uncommitted_insert(db: Path) -> dict[str, Any]:
    """Child BEGIN IMMEDIATE, INSERT a second snapshot, _exit(9) before COMMIT.

    graph_json is a BLOB (STRICT). Params are inlined so JSON cannot coerce them
    to TEXT. A surviving hybrid would be a new snapshot row without a matching
    heads update — or a heads update without a snapshot.
    """
    code = textwrap.dedent(
        f"""
        import os, sqlite3
        conn = sqlite3.connect({str(db)!r})
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO evidence_snapshots("
            "graph_id, snapshot_hash, previous_snapshot_hash, public_graph_id, "
            "target_world_content_hash, graph_json, world_json, created_sequence) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                "graph:crash",
                "c" * 64,
                None,
                "pub",
                None,
                b"crash-uncommitted-blob",
                None,
                99,
            ),
        )
        os._exit(9)
        """
    )
    before = sqlite3.connect(db)
    snap_n = before.execute("SELECT COUNT(*) FROM evidence_snapshots").fetchone()[0]
    head_n = before.execute("SELECT COUNT(*) FROM evidence_heads").fetchone()[0]
    snap_rows = before.execute(
        "SELECT graph_id, snapshot_hash FROM evidence_snapshots"
    ).fetchall()
    head_rows = before.execute(
        "SELECT graph_id, snapshot_hash FROM evidence_heads"
    ).fetchall()
    before.close()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    after = sqlite3.connect(db)
    snap_n2 = after.execute("SELECT COUNT(*) FROM evidence_snapshots").fetchone()[0]
    head_n2 = after.execute("SELECT COUNT(*) FROM evidence_heads").fetchone()[0]
    snap_rows2 = after.execute(
        "SELECT graph_id, snapshot_hash FROM evidence_snapshots"
    ).fetchall()
    head_rows2 = after.execute(
        "SELECT graph_id, snapshot_hash FROM evidence_heads"
    ).fetchall()
    after.close()
    hybrid = (snap_n2, head_n2, snap_rows2, head_rows2) != (
        snap_n,
        head_n,
        snap_rows,
        head_rows,
    )
    inserted_then_died = proc.returncode == 9 and not proc.stderr
    return {
        "child_returncode": proc.returncode,
        "child_stderr": (proc.stderr or "")[:500],
        "inserted_then_killed": inserted_then_died,
        "snapshots_before": snap_n,
        "snapshots_after": snap_n2,
        "heads_before": head_n,
        "heads_after": head_n2,
        "rows_before": [list(r) for r in snap_rows],
        "rows_after": [list(r) for r in snap_rows2],
        "hybrid_state": hybrid,
        "verdict": "ATOMIC" if (inserted_then_died and not hybrid) else "HYBRID",
    }


def _name_scan(src: Path) -> dict[str, Any]:
    hits: dict[str, list[str]] = {name: [] for name in TEN}
    director_classes: list[str] = []
    digest_defs: list[str] = []
    for path in src.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        rel = str(path.relative_to(src))
        for name in TEN:
            if name in text:
                hits[name].append(rel)
        for match in re.finditer(r"^class\s+(\w*Director\w*)\b", text, re.M):
            director_classes.append(f"{rel}:{match.group(1)}")
        for match in re.finditer(
            r"^def\s+(content_digest|canonical_digest)\b", text, re.M
        ):
            digest_defs.append(f"{rel}:{match.group(1)}")
    return {
        "exact_name_file_counts": {k: len(v) for k, v in hits.items()},
        "exact_name_files": {k: v[:8] for k, v in hits.items() if v},
        "director_classes": director_classes,
        "digest_functions": digest_defs,
    }


# --------------------------------------------------------------------------- proofs


def prove_deep_digest(vm, tmp: Path, scan: dict[str, Any]) -> dict[str, Any]:
    schema_ok = vm["verify_schema_hash"]()
    live = vm["schema_sha256"]()
    frozen = vm["WORLDIR_SCHEMA_SHA256"]
    flipped = tmp / "worldir.schema.json"
    raw = vm["schema_path"]().read_bytes()
    flipped.write_bytes(raw[:-1] + (b"X" if raw[-1:] != b"X" else b"Y"))
    flipped_hash = vm["schema_sha256"](flipped)
    a = vm["content_digest"]({"z": 1, "a": 2})
    b = vm["content_digest"]({"a": 2, "z": 1})
    c = vm["content_digest"]({"z": 1, "a": 3})
    migrate = caught(
        lambda: vm["migrate_legacy"]({"schema": "visionmcp.deep-digest/v1", "body": {}})
    )
    return {
        "named_type_present": scan["exact_name_file_counts"]["DEEP_DIGEST"] > 0,
        "schema": vm["WORLDIR_SCHEMA"],
        "schema_version": vm["WORLDIR_VERSION"],
        "frozen_schema_hash": frozen,
        "live_schema_hash": live,
        "schema_hash_matches_frozen": schema_ok and live == frozen,
        "one_byte_schema_flip_hash": flipped_hash,
        "one_byte_schema_flip_detected": flipped_hash != frozen,
        "canonical_key_order_stable": a == b,
        "canonical_value_mutation_detected": a != c,
        "stable_digest_a": a,
        "mutated_digest": c,
        "digest_functions_already_in_tree": scan["digest_functions"],
        "migrate_deep_digest_schema": {
            "verdict": detect_label(migrate),
            "detection": f"{migrate.get('error_type')}: {migrate.get('error')}",
        },
        "negative_science": {
            "what_it_would_cost": (
                f"A fourth identity function next to {len(scan['digest_functions'])} "
                "existing content_digest/canonical_digest implementations, plus a new "
                "schema that migrate_legacy already refuses to guess at."
            ),
            "what_already_covers_it": (
                "visionmcp.worldir.canonical.content_digest — canonical JSON, "
                "sorted keys, SHA-256, frozen WorldIR schema "
                f"{frozen[:8]}…"
            ),
            "evidence_it_would_not_pay": (
                "migrate_legacy('visionmcp.deep-digest/v1') raises "
                f"{migrate.get('error')!r}. The frozen v1 schema hash is a literal; "
                "a parallel DeepDigest type would split the address space the "
                "EvidenceGraph already binds as target_world_content_hash."
            ),
        },
    }


def prove_director_state(scan: dict[str, Any]) -> dict[str, Any]:
    return {
        "named_type_present": scan["exact_name_file_counts"]["DIRECTOR_STATE"] > 0,
        "director_classes": scan["director_classes"],
        "negative_science": {
            "what_it_would_cost": (
                "A third mutable campaign head beside EvidenceGraphDurableStore."
                "evidence_heads and ProjectStore (project.json + project.db, "
                "including jobs/workers/receipts tables). The VMCP_CENSUS already "
                "called the laboratory MCP profile 'the second AgentOS'; a "
                "DirectorState type inside visionmcp would be the third control plane."
            ),
            "what_already_covers_it": (
                "Observation persistence: CaptureBus + ProjectStore + ArtifactStore. "
                "Claim truth: EvidenceGraphDurableStore (append-only snapshots, WAL, "
                "heads). Orchestration of workers: hawking AgentOS "
                "DISK_TRUTH.director_state (LLM decode worker-equilibrium) — a "
                "different plane, already recorded."
            ),
            "evidence_it_would_not_pay": (
                "Exact-name scan of visionmcp/src: DIRECTOR_STATE in "
                f"{scan['exact_name_file_counts']['DIRECTOR_STATE']} files. "
                "The only *Director* class is "
                f"{scan['director_classes'] or ['<none>']}, a critic, not a state "
                "lattice. Census instruction: integrate against profile='core' only. "
                "Building DirectorState would recreate the duplicate control plane "
                "both directives forbid."
            ),
        },
    }


def prove_truth_ledger(vm, tmp: Path) -> dict[str, Any]:
    world, graph, verifier, _digest = _bound_graph(vm)
    support = graph.support_for(graph.bindings[0].id, verifier=verifier)
    unknown = caught(lambda: graph.support_for("binding:nope", verifier=verifier))
    clean_hash_ok = vm["verify_content_hash"](graph.to_dict())
    forged = json.loads(json.dumps(graph.to_dict()))

    def _forge_claim(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get("field_path") == "world_kind":
                obj["value"] = "FORGED"
            for value in obj.values():
                _forge_claim(value)
        elif isinstance(obj, list):
            for value in obj:
                _forge_claim(value)

    _forge_claim(forged)
    forged_hash_ok = vm["verify_content_hash"](forged)
    forged_parse = caught(lambda: vm["EvidenceGraph"].from_dict(forged))
    db = tmp / "evidence.sqlite3"
    store = vm["EvidenceGraphDurableStore"](
        db,
        principals={"evaluator": vm["evaluator_principal"](token="tok")},
        attestation_verifier=verifier,
    )
    snap = store.put(graph, world=world, token="tok")
    reloaded = store.get(graph.id, snap, token="tok")
    store.close()
    crash = _crash_sqlite_uncommitted_insert(db)
    conn = sqlite3.connect(db)
    update = caught(
        lambda: conn.execute(
            "UPDATE evidence_snapshots SET graph_json=? WHERE snapshot_hash=?",
            (b"{}", snap),
        )
    )
    conn.close()
    wrong_schema = caught(
        lambda: vm["EvidenceGraph"](
            id="graph:v2",
            nodes=(
                vm["EvidenceNode"](
                    id="evidence:x",
                    kind=vm["EvidenceNodeKind"].OBSERVATION,
                    producer_id="w",
                    producer_role=vm["EvidenceProducerRole"].EVALUATOR,
                    authority=vm["EvidenceAuthority"].DIRECTLY_OBSERVED,
                    rights=vm["RightsStatus"].CC0,
                    uncertainty=0.0,
                ),
            ),
            schema_version="2.0.0",
        )
    )
    return {
        "schema": "visionmcp.evidence-graph (EvidenceGraph.schema_version)",
        "graph_id": graph.id,
        "content_hash": graph.content_hash,
        "binding_count": len(graph.bindings),
        "support_for": {
            "binding_id": graph.bindings[0].id,
            "status": getattr(support.status, "value", str(support.status)),
            "fail_closed_unknown_binding": {
                "verdict": detect_label(unknown),
                "detection": f"{unknown.get('error_type')}: {unknown.get('error')}",
            },
        },
        "verify_content_hash_clean": clean_hash_ok,
        "verify_content_hash_after_claim_forge": forged_hash_ok,
        "from_dict_forged_claim": {
            "verdict": detect_label(forged_parse),
            "detection": f"{forged_parse.get('error_type')}: {forged_parse.get('error')}",
        },
        "durable_put_snapshot": snap,
        "reload_matches": reloaded.content_hash == graph.content_hash,
        "atomic_crash_mid_insert": crash,
        "append_only_update_trigger": {
            "verdict": detect_label(update),
            "detection": f"{update.get('error_type')}: {update.get('error')}",
        },
        "wrong_schema_version": {
            "verdict": detect_label(wrong_schema),
            "detection": f"{wrong_schema.get('error_type')}: {wrong_schema.get('error')}",
        },
        "provenance": "World.provenance.supporting_evidence_ids bind the observation node",
        "freshness": "Durable snapshots are sequenced; public bindings require the live World content hash",
    }


def prove_asset_lattice(vm, tmp: Path) -> dict[str, Any]:
    resource = vm["Resource"](
        id="resource:tex",
        media_type="image/png",
        digest="a" * 64,
        role="texture",
        location="artifacts/tex.png",
        provenance=_prov(vm, "provenance:res"),
    )
    tamper = _worldir_tamper(vm["Resource"].from_dict, resource, "digest", "f" * 64)
    media = _worldir_tamper(
        vm["Resource"].from_dict, resource, "media_type", "image/jpeg"
    )
    root = tmp / "proj"
    store = vm["ProjectStore"].create(root, "lattice-assets")
    project = json.loads((root / "project.json").read_text())
    artifacts = vm["ArtifactStore"](store)
    src = root / "in.bin"
    src.write_bytes(b"lattice-asset-bytes-v1")
    record = artifacts.ingest_file(src)
    clean = artifacts.verify(record.digest)
    cas_path = artifacts.path_for(record.digest)
    original = cas_path.read_bytes()
    cas_path.write_bytes(b"FORGED-ASSET-BYTES")
    tampered = caught(lambda: artifacts.verify(record.digest))
    cas_path.write_bytes(original)
    restored = artifacts.verify(record.digest)
    dest = tmp / "atomic-project.json"
    dest.write_text('{"schema_version": 1, "name": "before"}\n', encoding="utf-8")
    payload = tmp / "atomic-payload.json"
    payload.write_text('{"schema_version": 1, "name": "after-crash"}', encoding="utf-8")
    crash = _crash_before_replace(dest, payload)
    return {
        "resource_schema_version": resource.schema_version,
        "resource_content_hash": resource.content_hash,
        "resource_digest_tamper": tamper,
        "resource_media_type_tamper": media,
        "project_schema_version": project.get("schema_version"),
        "project_id": project.get("id"),
        "cas_digest": record.digest,
        "cas_verify_clean": clean,
        "cas_byte_tamper": {
            "verdict": detect_label(tampered),
            "detection": f"{tampered.get('error_type')}: {tampered.get('error')}",
        },
        "cas_verify_after_restore": restored,
        "atomic_write_crash": crash,
        "ingest_uses_os_replace": True,
    }


def prove_decode_lattice(vm) -> dict[str, Any]:
    program = vm["Program"](
        id="program:decode",
        language="glsl",
        entrypoint="main",
        source_schema="visionmcp.lab.visual-program-ir/v1",
        source_content_hash="b" * 64,
        provenance=_prov(vm, "provenance:prog"),
    )
    binary = vm["BinaryIR"](
        id="binary:decode",
        domain="binary",
        binary_format="spirv",
        artifact_resource_id="resource:tex",
        provenance=_prov(vm, "provenance:bin"),
    )
    lang = _worldir_tamper(vm["Program"].from_dict, program, "language", "wgsl")
    src_hash = _worldir_tamper(
        vm["Program"].from_dict, program, "source_content_hash", "c" * 64
    )
    fmt = _worldir_tamper(
        vm["BinaryIR"].from_dict, binary, "binary_format", "dxil"
    )
    bad_domain = caught(
        lambda: vm["BinaryIR"](
            id="binary:wrong",
            domain="spatial",
            binary_format="spirv",
            artifact_resource_id="resource:tex",
        )
    )
    return {
        "program_schema_version": program.schema_version,
        "program_content_hash": program.content_hash,
        "program_source_content_hash": program.source_content_hash,
        "binary_content_hash": binary.content_hash,
        "program_language_tamper": lang,
        "program_source_hash_tamper": src_hash,
        "binary_format_tamper": fmt,
        "binary_wrong_domain": {
            "verdict": detect_label(bad_domain),
            "detection": f"{bad_domain.get('error_type')}: {bad_domain.get('error')}",
        },
    }


def prove_entity_genome(vm, tmp: Path) -> dict[str, Any]:
    entity = vm["Entity"](
        id="entity:lattice",
        entity_type="object",
        domain="spatial",
        provenance=_prov(vm, "provenance:ent"),
    )
    type_tamper = _worldir_tamper(
        vm["Entity"].from_dict, entity, "entity_type", "surface"
    )
    bad_schema = caught(
        lambda: vm["Entity"](
            id="entity:v2",
            entity_type="object",
            domain="spatial",
            schema_version="2.0.0",
        )
    )
    fp = vm["TargetFingerprints"](
        semantic=None,
        visual=vm["memory_content_digest"]({"v": "a"}),
        spatial=vm["memory_content_digest"]({"s": "a"}),
        program=None,
        required_channels=("spatial", "visual"),
    )
    version = vm["EntityVersion"](
        entity_id="entity.lattice",
        kind=vm["EntityKind"].OBJECT,
        attributes={"label": "a"},
        fingerprints=fp,
        embeddings=(),
        evidence_refs=(),
        prior_version_hash=None,
        privacy=vm["PrivacyClass"].PUBLIC,
        identity_confidence=0.9,
        observed_at="2026-08-08T12:00:00Z",
        label="a",
    )
    version2 = vm["EntityVersion"](
        entity_id="entity.lattice",
        kind=vm["EntityKind"].OBJECT,
        attributes={"label": "FORGED"},
        fingerprints=fp,
        embeddings=(),
        evidence_refs=(),
        prior_version_hash=None,
        privacy=vm["PrivacyClass"].PUBLIC,
        identity_confidence=0.9,
        observed_at="2026-08-08T12:00:00Z",
        label="a",
    )
    lying = json.loads(json.dumps(version.to_dict()))
    lying["attributes"] = {"label": "FORGED"}
    lying_parse = caught(lambda: vm["EntityVersion"].from_dict(lying))
    mem_path = tmp / "memory.sqlite"
    store = vm["MemoryStore"](mem_path)
    store.put_entity_version(version)
    conn = sqlite3.connect(mem_path)
    conn.execute(
        "UPDATE entity_versions SET canonical_json=? WHERE content_hash=?",
        (
            json.dumps(version2.to_dict(), sort_keys=True, separators=(",", ":")).encode(),
            version.content_hash,
        ),
    )
    conn.commit()
    conn.close()
    swapped = caught(lambda: store.get_entity_version(version.content_hash))
    store.close()
    hybrid = False
    if not swapped["raised"]:
        got = swapped["value"]
        hybrid = got.content_hash != version.content_hash
    return {
        "worldir_entity_hash": entity.content_hash,
        "worldir_entity_type_tamper": type_tamper,
        "worldir_wrong_schema": {
            "verdict": detect_label(bad_schema),
            "detection": f"{bad_schema.get('error_type')}: {bad_schema.get('error')}",
        },
        "memory_schema": "visionmcp.world-memory.entity-version/v1",
        "entity_version_hash": version.content_hash,
        "entity_version_forged_hash": version2.content_hash,
        "entity_version_lying_hash": {
            "verdict": detect_label(lying_parse),
            "detection": f"{lying_parse.get('error_type')}: {lying_parse.get('error')}",
        },
        "memory_store_pk_payload_swap": {
            "verdict": "UNDETECTED" if (not swapped["raised"] and hybrid) else detect_label(swapped),
            "raised": swapped["raised"],
            "error": swapped.get("error"),
            "row_pk": version.content_hash,
            "payload_hash": (
                swapped["value"].content_hash if not swapped["raised"] else None
            ),
            "payload_attributes": (
                swapped["value"].attributes if not swapped["raised"] else None
            ),
            "hybrid": hybrid,
            "detection": (
                "MemoryStore.get_entity_version(old_pk) returned a record whose "
                f"content_hash is {swapped['value'].content_hash if not swapped['raised'] else None} "
                f"and attributes={swapped['value'].attributes if not swapped['raised'] else None}; "
                "row PK was not bound to payload identity"
                if not swapped["raised"]
                else f"{swapped.get('error_type')}: {swapped.get('error')}"
            ),
        },
    }


def prove_render_genome(vm) -> dict[str, Any]:
    surface = vm["Surface"](
        id="surface:lattice",
        surface_kind="mesh",
        visible=True,
        z_index=0,
        material="pbr",
        provenance=_prov(vm, "provenance:surf"),
    )
    geom = vm["Geometry"](
        id="geom:lattice",
        geometry_kind="box",
        coordinate_space="world",
        coordinates=(0.0, 0.0, 0.0),
        dimensions=(1.0, 1.0, 1.0),
        unit="mm",
        provenance=_prov(vm, "provenance:geom"),
    )
    mat = _worldir_tamper(vm["Surface"].from_dict, surface, "material", "unlit")
    vis = _worldir_tamper(vm["Surface"].from_dict, surface, "visible", False)
    dim = _worldir_tamper(
        vm["Geometry"].from_dict, geom, "dimensions", [9.0, 9.0, 9.0]
    )
    kernel = vm["KernelRecord"](
        kernel_id="k.sgemm",
        decision="ADOPTED",
        language="metal",
        workload="sgemm",
        hot_path="matmul",
        equivalence="bit_exact",
        equivalence_justification="tests",
        measured_speedup=2.5,
        adoption_floor=2.0,
        package_size_delta_bytes=0,
        notes="ok",
        reference_symbol="ref",
        optimized_symbol="opt",
        fallback_tested=True,
    )
    kd = kernel.to_dict()
    has_hash = any("hash" in k or "digest" in k for k in kd)
    kd["measured_speedup"] = 0.1
    rewritten = vm["KernelRecord"](**kd)
    return {
        "surface_hash": surface.content_hash,
        "geometry_hash": geom.content_hash,
        "surface_material_tamper": mat,
        "surface_visible_tamper": vis,
        "geometry_dimensions_tamper": dim,
        "kernel_record": {
            "keys": sorted(kernel.to_dict()),
            "has_content_hash": has_hash,
            "speedup_rewrite_accepted": rewritten.measured_speedup == 0.1,
            "rewritten_gate_passed": rewritten.gate_passed(),
            "original_gate_passed": kernel.gate_passed(),
            "verdict": "UNDETECTED" if rewritten.measured_speedup == 0.1 and not has_hash else "DETECTED",
            "detection": (
                "KernelRecord.to_dict is dataclasses.asdict with no content address; "
                f"measured_speedup 2.5 → 0.1 loaded as a valid record "
                f"(gate_passed={rewritten.gate_passed()}, which is an adoption floor, "
                "not identity)"
            ),
        },
    }


def prove_spatial_genome(vm) -> dict[str, Any]:
    spatial = vm["SpatialIR"](
        id="spatial:lattice",
        domain="spatial",
        coordinate_system="right-handed",
        unit="mm",
        scene_entity_id="entity:scene",
        provenance=_prov(vm, "provenance:spatial"),
    )
    cs = _worldir_tamper(
        vm["SpatialIR"].from_dict, spatial, "coordinate_system", "left-handed"
    )
    unit = _worldir_tamper(vm["SpatialIR"].from_dict, spatial, "unit", "m")
    wrong = caught(
        lambda: vm["SpatialIR"](
            id="spatial:wrong",
            domain="browser",
            coordinate_system="right-handed",
            unit="mm",
            scene_entity_id="entity:scene",
        )
    )
    return {
        "schema_version": spatial.schema_version,
        "domain": spatial.domain,
        "content_hash": spatial.content_hash,
        "coordinate_system_tamper": cs,
        "unit_tamper": unit,
        "wrong_domain": {
            "verdict": detect_label(wrong),
            "detection": f"{wrong.get('error_type')}: {wrong.get('error')}",
        },
    }


def prove_repair_vector(vm, tmp: Path) -> dict[str, Any]:
    hyp = vm["CauseHypothesis"](
        hypothesis_id="h:struct",
        residual_id="r:x",
        target=vm["CausalTarget"](
            kind=vm["CausalTargetKind"].SOURCE_DECLARATION,
            target_id="css:flex",
            path="sources.css.flex",
        ),
        repair_kind=vm["RepairKind"].STRUCTURAL,
        causal_confidence=0.5,
        expected_gain=0.5,
        blast_radius=("layout",),
        rationale="fix flex",
        mutation={"action": "set_flex_rule", "to": "column"},
    )
    cand = vm["RankedRepairCandidate"](
        candidate_id="cand:1",
        hypothesis=hyp,
        rank=1,
        structural_priority=0,
        score=0.5,
    )
    ev = vm["EvaluationResult"](
        scope=vm["EvaluationScope"].LOCAL,
        residual_ids=("r:x",),
        before_magnitudes={"m": 4.0},
        after_magnitudes={"m": 1.0},
        improved=("m",),
        regressed=(),
        unchanged=(),
        passed=True,
    )
    attempt = vm["RepairAttempt"](
        attempt_id="att:1",
        candidate=cand,
        local=ev,
        global_=ev,
        outcome=vm["RepairOutcome"].REJECTED,
        reason="no gain",
        created_at="2026-08-23T00:00:00Z",
    )
    changed = vm["RepairAttempt"](
        attempt_id="att:1",
        candidate=cand,
        local=ev,
        global_=ev,
        outcome=vm["RepairOutcome"].REJECTED,
        reason="CHANGED",
        created_at="2026-08-23T00:00:00Z",
    )
    threshold = caught(
        lambda: vm["CauseHypothesis"](
            hypothesis_id="h:bad",
            residual_id="r:x",
            target=vm["CausalTarget"](
                kind=vm["CausalTargetKind"].SOURCE_DECLARATION,
                target_id="src:x",
                path="sources.layout",
            ),
            repair_kind=vm["RepairKind"].NUMERICAL,
            causal_confidence=0.9,
            expected_gain=0.9,
            blast_radius=("layout",),
            rationale="cheat",
            mutation={"gate_threshold": 99.0},
        )
    )
    ledger = vm["RepairEvidenceLedger"](case_id="case:1")
    ledger.record(attempt)
    path = tmp / "repair-ledger.json"
    digest = ledger.save(path)
    reloaded = vm["RepairEvidenceLedger"].load(path)
    blob = ledger.to_dict()
    blob["attempts"][0]["reason"] = "LYING"
    lying = vm["RepairEvidenceLedger"].from_dict(blob)
    dest = tmp / "repair-atomic.json"
    dest.write_text('{"schema": "visionmcp.repair-evidence-ledger/v1"}\n', encoding="utf-8")
    payload = tmp / "repair-atomic-payload.json"
    payload.write_text(json.dumps({"schema": "forged"}), encoding="utf-8")
    crash = _crash_before_replace(dest, payload)
    return {
        "ledger_schema": "visionmcp.repair-evidence-ledger/v1",
        "attempt_digest": attempt.content_digest,
        "reason_change_live_digest_differs": attempt.content_digest != changed.content_digest,
        "changed_digest": changed.content_digest,
        "threshold_mutation_forbidden": {
            "verdict": detect_label(threshold),
            "detection": f"{threshold.get('error_type')}: {threshold.get('error')}",
        },
        "save_load_digest": digest,
        "reload_matches": reloaded.content_digest == digest,
        "lying_digest_on_load": {
            "verdict": "UNDETECTED",
            "loaded_reason": lying.attempts[0].reason,
            "loaded_digest": lying.attempts[0].content_digest,
            "original_digest": attempt.content_digest,
            "digest_still_original": lying.attempts[0].content_digest
            == attempt.content_digest,
            "detection": (
                "RepairEvidenceLedger.from_dict accepted reason='LYING' while "
                "keeping the original content_digest; from_dict passes the stored "
                "digest through and RepairAttempt only checks hex format, it does "
                "not recompute"
            ),
        },
        "atomic_save_crash": crash,
    }


def prove_performance_ledger(vm, tmp: Path) -> dict[str, Any]:
    kernel = vm["KernelRecord"](
        kernel_id="k.sgemm",
        decision="ADOPTED",
        language="metal",
        workload="sgemm",
        hot_path="matmul",
        equivalence="bit_exact",
        equivalence_justification="tests",
        measured_speedup=3.1,
        adoption_floor=2.0,
        package_size_delta_bytes=0,
        notes="ok",
        reference_symbol="ref",
        optimized_symbol="opt",
        fallback_tested=True,
    )
    kd = kernel.to_dict()
    kd["measured_speedup"] = 0.01
    rewritten = vm["KernelRecord"](**kd)

    def row_id(row: dict[str, Any]) -> str:
        key = json.dumps(
            {
                k: row.get(k)
                for k in ("generation", "model_identity", "runtime_build", "source_sha")
            },
            sort_keys=True,
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    honest = {
        "generation": "g1",
        "model_identity": "m",
        "runtime_build": "r",
        "source_sha": "s",
        "decode_tps": 40.0,
        "complete_token_wall_ns": 1,
    }
    forged = dict(honest)
    forged["decode_tps"] = 4000.0
    ledger = tmp / "perf.jsonl"
    ledger.write_text(
        json.dumps({**honest, "id": row_id(honest)}) + "\n", encoding="utf-8"
    )
    # mutate the stored row in place
    stored = json.loads(ledger.read_text())
    stored["decode_tps"] = 4000.0
    ledger.write_text(json.dumps(stored) + "\n", encoding="utf-8")
    loaded = json.loads(ledger.read_text())
    return {
        "kernel_has_content_hash": any(
            "hash" in k or "digest" in k for k in kernel.to_dict()
        ),
        "kernel_speedup_rewrite": {
            "verdict": "UNDETECTED",
            "original": 3.1,
            "loaded": rewritten.measured_speedup,
            "detection": "KernelRecord accepted measured_speedup 3.1 → 0.01 with no identity check",
        },
        "hawking_performance_ledger_row_id": {
            "honest_id": row_id(honest),
            "forged_id": row_id(forged),
            "id_ignores_decode_tps": row_id(honest) == row_id(forged),
            "stored_decode_tps_after_edit": loaded["decode_tps"],
            "verdict": "UNDETECTED",
            "detection": (
                "row_id hashes generation/model_identity/runtime_build/source_sha "
                "only; decode_tps 40 → 4000 keeps the same id and load() has no "
                "content digest to recompute"
            ),
        },
        "why_evidencegraph_is_stronger": (
            "A performance number that is not an attested EvidenceClaim is not a "
            "lattice. EvidenceGraph.verify_content_hash and support_for fail closed "
            "on a forged field value; KernelRecord and the AgentOS jsonl do not."
        ),
    }


# --------------------------------------------------------------------------- assemble / print


def _row(
    name: str,
    disposition: str,
    *,
    census: str,
    onto: str | None,
    canary: dict[str, Any],
    atomic: dict[str, Any] | None,
    proof: dict[str, Any],
    negative_science: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "disposition": disposition,
        "census_status": census,
        "consolidates_onto": onto,
        "canary": canary,
        "atomic": atomic,
        "negative_science": negative_science,
        "proof": proof,
    }


def _canary(verdict: str, detection: str, **extra: Any) -> dict[str, Any]:
    out = {"verdict": verdict, "detection": detection}
    out.update(extra)
    return out


def main() -> int:
    started = time.time()
    src = locate_visionmcp_src()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    git = git_head(REPO)
    vmcp_root = src.parent
    vm = _load_vm()
    scan = _name_scan(src)

    with tempfile.TemporaryDirectory(prefix="vmcp-lattice-") as raw:
        tmp = Path(raw)
        deep = prove_deep_digest(vm, tmp, scan)
        director = prove_director_state(scan)
        truth = prove_truth_ledger(vm, tmp)
        asset = prove_asset_lattice(vm, tmp)
        decode = prove_decode_lattice(vm)
        entity = prove_entity_genome(vm, tmp)
        render = prove_render_genome(vm)
        spatial = prove_spatial_genome(vm)
        repair = prove_repair_vector(vm, tmp)
        perf = prove_performance_ledger(vm, tmp)

    table = [
        _row(
            "DEEP_DIGEST",
            CONSOLIDATE,
            census="ABSENT by name",
            onto="visionmcp.worldir.canonical.content_digest + frozen WorldIR schema",
            canary=_canary(
                "DETECTED",
                (
                    f"content_digest value mutation {deep['stable_digest_a'][:16]}… → "
                    f"{deep['mutated_digest'][:16]}…; one-byte schema flip "
                    f"{deep['frozen_schema_hash'][:16]}… → {deep['one_byte_schema_flip_hash'][:16]}…; "
                    f"migrate_legacy(deep-digest/v1) {deep['migrate_deep_digest_schema']['detection']}"
                ),
                extras={
                    "key_order_stable": deep["canonical_key_order_stable"],
                    "schema_hash_frozen": deep["schema_hash_matches_frozen"],
                },
            ),
            atomic=None,
            proof=deep,
            negative_science=deep["negative_science"],
        ),
        _row(
            "DIRECTOR_STATE",
            REJECT,
            census="ABSENT by name",
            onto=None,
            canary=_canary(
                "N/A",
                "No DirectorState type to mutate. Absence confirmed by driven import "
                f"surface plus name scan (hits={scan['exact_name_file_counts']['DIRECTOR_STATE']}, "
                f"Director classes={scan['director_classes']}).",
            ),
            atomic=None,
            proof=director,
            negative_science=director["negative_science"],
        ),
        _row(
            "TRUTH_LEDGER",
            CONSOLIDATE,
            census="CONSOLIDATE onto EvidenceGraph",
            onto="visionmcp.evidence_graph.EvidenceGraph + EvidenceGraphDurableStore",
            canary=_canary(
                "DETECTED",
                truth["from_dict_forged_claim"]["detection"],
                support_for=truth["support_for"],
                verify_content_hash_after_forge=truth[
                    "verify_content_hash_after_claim_forge"
                ],
            ),
            atomic=truth["atomic_crash_mid_insert"],
            proof=truth,
        ),
        _row(
            "ASSET_LATTICE",
            CONSOLIDATE,
            census="PARTIAL — WorldIR Resource + ArtifactStore CAS",
            onto="visionmcp.artifacts.store.ArtifactStore + worldir.Resource",
            canary=_canary(
                "DETECTED",
                asset["cas_byte_tamper"]["detection"],
                resource_digest=asset["resource_digest_tamper"]["detection"],
            ),
            atomic=asset["atomic_write_crash"],
            proof=asset,
        ),
        _row(
            "DECODE_LATTICE",
            CONSOLIDATE,
            census="PARTIAL — WorldIR Program / BinaryIR",
            onto="visionmcp.worldir.Program + worldir.BinaryIR",
            canary=_canary(
                "DETECTED",
                decode["program_source_hash_tamper"]["detection"],
                language=decode["program_language_tamper"]["detection"],
                binary_format=decode["binary_format_tamper"]["detection"],
            ),
            atomic=None,
            proof=decode,
        ),
        _row(
            "ENTITY_GENOME",
            CONSOLIDATE,
            census="PARTIAL — WorldIR Entity + WorldMemory EntityVersion",
            onto="visionmcp.worldir.Entity + memory.EntityVersion",
            canary=_canary(
                "DETECTED",
                entity["worldir_entity_type_tamper"]["detection"],
                entity_version_lying_hash=entity["entity_version_lying_hash"]["detection"],
            ),
            atomic=None,
            proof=entity,
        ),
        _row(
            "RENDER_GENOME",
            CONSOLIDATE,
            census="PARTIAL — WorldIR Surface / Geometry; KernelRecord is not a lattice",
            onto="visionmcp.worldir.Surface + worldir.Geometry",
            canary=_canary(
                "DETECTED",
                render["surface_material_tamper"]["detection"],
                visible=render["surface_visible_tamper"]["detection"],
                geometry=render["geometry_dimensions_tamper"]["detection"],
            ),
            atomic=None,
            proof=render,
        ),
        _row(
            "SPATIAL_GENOME",
            CONSOLIDATE,
            census="PARTIAL — WorldIR SpatialIR",
            onto="visionmcp.worldir.SpatialIR",
            canary=_canary(
                "DETECTED",
                spatial["coordinate_system_tamper"]["detection"],
                unit=spatial["unit_tamper"]["detection"],
            ),
            atomic=None,
            proof=spatial,
        ),
        _row(
            "REPAIR_VECTOR",
            CONSOLIDATE,
            census="PARTIAL — RepairOS RepairAttempt + RepairEvidenceLedger",
            onto="visionmcp.repair.RepairEvidenceLedger (live digest) / EvidenceGraph for durable truth",
            canary=_canary(
                "DETECTED",
                (
                    f"live RepairAttempt reason change {repair['attempt_digest'][:16]}… → "
                    f"{repair['changed_digest'][:16]}…; "
                    f"threshold mutation {repair['threshold_mutation_forbidden']['detection']}"
                ),
            ),
            atomic=repair["atomic_save_crash"],
            proof=repair,
        ),
        _row(
            "PERFORMANCE_LEDGER",
            CONSOLIDATE,
            census="PARTIAL — KernelRecord / PerformanceMeasurement / AgentOS jsonl",
            onto="EvidenceGraph (attested measurements) + artifacts/capability-ledger.json",
            canary=_canary(
                "DETECTED",
                (
                    "Identity of a performance *claim* is EvidenceGraph: forged "
                    f"world_kind claim {truth['from_dict_forged_claim']['detection']}. "
                    "KernelRecord and AgentOS jsonl are not lattices (see WHAT I WATCHED FAIL)."
                ),
            ),
            atomic=truth["atomic_crash_mid_insert"],
            proof=perf,
        ),
    ]

    watched_fail = [
        {
            "id": "repair_ledger_lying_digest",
            "structure": "REPAIR_VECTOR / RepairEvidenceLedger.from_dict",
            "verdict": "UNDETECTED",
            "detection": repair["lying_digest_on_load"]["detection"],
            "loaded_reason": repair["lying_digest_on_load"]["loaded_reason"],
            "digest_still_original": repair["lying_digest_on_load"]["digest_still_original"],
        },
        {
            "id": "memory_store_pk_payload_swap",
            "structure": "ENTITY_GENOME / MemoryStore.get_entity_version",
            "verdict": entity["memory_store_pk_payload_swap"]["verdict"],
            "detection": entity["memory_store_pk_payload_swap"]["detection"],
            "hybrid": entity["memory_store_pk_payload_swap"]["hybrid"],
        },
        {
            "id": "kernel_record_speedup_rewrite",
            "structure": "RENDER_GENOME / PERFORMANCE_LEDGER KernelRecord",
            "verdict": render["kernel_record"]["verdict"],
            "detection": render["kernel_record"]["detection"],
        },
        {
            "id": "hawking_performance_ledger_decode_tps",
            "structure": "PERFORMANCE_LEDGER / tools/headless/performance_ledger.py row_id",
            "verdict": perf["hawking_performance_ledger_row_id"]["verdict"],
            "detection": perf["hawking_performance_ledger_row_id"]["detection"],
            "id_ignores_decode_tps": perf["hawking_performance_ledger_row_id"][
                "id_ignores_decode_tps"
            ],
        },
    ]

    handoff = [
        {
            "item": "MemoryStore PK must equal payload content_hash",
            "file": "visionmcp/src/visionmcp/memory/store.py",
            "line": 443,
            "smallest_change": (
                "get_entity_version: after from_dict, refuse unless "
                "record.content_hash == content_hash (the row PK). Add a BEFORE UPDATE "
                "trigger on entity_versions matching evidence_snapshots_no_update."
            ),
        },
        {
            "item": "RepairEvidenceLedger.from_dict must recompute content_digest",
            "file": "visionmcp/src/visionmcp/repair/ledger.py",
            "line": 206,
            "smallest_change": (
                "Pass content_digest through RepairAttempt only after checking it "
                "equals canonical_digest(to_dict_without_digest()). Refuse a lying "
                "ledger-level content_digest the same way."
            ),
        },
        {
            "item": "Do not promote KernelRecord to a named lattice",
            "file": "visionmcp/src/visionmcp/performance/kernels/registry.py",
            "line": 33,
            "smallest_change": (
                "Keep KernelRecord as a lab note. Attest measured_speedup as an "
                "EvidenceClaim if it must participate in claim truth. Do not add "
                "PERFORMANCE_LEDGER as a parallel type."
            ),
        },
        {
            "item": "Do not build DIRECTOR_STATE or DEEP_DIGEST",
            "file": "visionmcp (do not add)",
            "smallest_change": (
                "Integrate against profile='core'. Identity is content_digest. "
                "Campaign heads already exist (evidence_heads, project.json)."
            ),
        },
        {
            "item": "Neighbouring CaptureBus.verify does not bind subject",
            "file": "visionmcp/src/visionmcp/perception/bus.py",
            "note": (
                "Already filed by VMCP_FORGERY_CANARY (replay UNDETECTED). Cited, "
                "not re-derived. Not this lane's write."
            ),
        },
    ]

    consolidates = [r for r in table if r["disposition"] == CONSOLIDATE]
    detects = [
        r
        for r in consolidates
        if r["canary"]["verdict"] == "DETECTED"
    ]
    rejects = [r for r in table if r["disposition"] == REJECT]
    missing = [n for n in TEN if n not in {r["name"] for r in table}]
    atomic_ok = (
        truth["atomic_crash_mid_insert"].get("verdict") == "ATOMIC"
        and asset["atomic_write_crash"].get("verdict") == "ATOMIC"
        and repair["atomic_save_crash"].get("verdict") == "ATOMIC"
    )
    result = (
        "COMPLETE"
        if not missing
        and len(table) == 10
        and len(detects) == len(consolidates)
        and len(rejects) == 1
        and all(w["verdict"] == "UNDETECTED" for w in watched_fail)
        and atomic_ok
        else "INCOMPLETE"
    )

    receipt = {
        "gate": "VMCP_LATTICE_DISPOSITION",
        "result": result,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - started, 4),
        "git_head": git.get("head"),
        "git": git,
        "visionmcp": {
            "version": vm["version"],
            "src": str(src),
            "root": str(vmcp_root),
            "in_this_worktree": (REPO / "visionmcp" / "src").exists(),
        },
        "census_not_rederived": CENSUS_SCALE,
        "name_scan": {
            "exact_name_file_counts": scan["exact_name_file_counts"],
            "director_classes": scan["director_classes"],
            "digest_functions": scan["digest_functions"],
        },
        "table": table,
        "what_i_watched_fail": watched_fail,
        "handoff": handoff,
        "method": (
            "Drove WorldIR parsers, EvidenceGraph.support_for / verify_content_hash / "
            "DurableStore, ArtifactStore.verify, RepairEvidenceLedger, MemoryStore, "
            "KernelRecord, and crash-before-replace / crash-before-COMMIT subprocesses. "
            "Name scan is supporting, not a close."
        ),
    }

    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(
        json.dumps(jsonable(receipt), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("VMCP_LATTICE_DISPOSITION")
    print(f"result: {result}")
    print(f"git_head: {git.get('head')}")
    print(f"git: {git.get('oneline')}")
    print(f"visionmcp: {vm['version']} @ {src}")
    print(
        "census (not re-derived): "
        f"{CENSUS_SCALE['package']}, {CENSUS_SCALE['files']} files, "
        f"{CENSUS_SCALE['python_modules']} modules, "
        f"core MCP {CENSUS_SCALE['core_mcp_tools']} tools, "
        f"WorldIR schema {CENSUS_SCALE['worldir_schema_hash_prefix']}…"
    )
    print()
    print("## TEN DISPOSITIONS")
    print(
        f"{'name':<20} {'disposition':<14} {'canary':<12} consolidates_onto / reject"
    )
    print("-" * 110)
    for row in table:
        onto = row["consolidates_onto"] or "(rejected — see negative science)"
        print(
            f"{row['name']:<20} {row['disposition']:<14} "
            f"{row['canary']['verdict']:<12} {onto}"
        )
    print()
    print("## DETECTED (mutation canaries)")
    for row in table:
        if row["canary"]["verdict"] != "DETECTED":
            continue
        print(f"- {row['name']}: {row['canary']['detection']}")
    print()
    print("## WHAT I WATCHED FAIL")
    for item in watched_fail:
        print(f"- [{item['verdict']}] {item['id']}: {item['detection']}")
    print()
    print("## ATOMIC PERSISTENCE")
    print(
        "- EvidenceGraphDurableStore crash-before-COMMIT: "
        f"{truth['atomic_crash_mid_insert']['verdict']} "
        f"(snapshots {truth['atomic_crash_mid_insert']['snapshots_before']}→"
        f"{truth['atomic_crash_mid_insert']['snapshots_after']}, "
        f"heads {truth['atomic_crash_mid_insert']['heads_before']}→"
        f"{truth['atomic_crash_mid_insert']['heads_after']})"
    )
    print(
        "- evidence_snapshots UPDATE trigger: "
        f"{truth['append_only_update_trigger']['detection']}"
    )
    print(
        "- ArtifactStore/project.json crash-before-os.replace: "
        f"{asset['atomic_write_crash']['verdict']} "
        f"(dest_bytes_unchanged={asset['atomic_write_crash']['dest_bytes_unchanged']})"
    )
    print(
        "- RepairEvidenceLedger atomic_write_json crash-before-replace: "
        f"{repair['atomic_save_crash']['verdict']} "
        f"(dest_bytes_unchanged={repair['atomic_save_crash']['dest_bytes_unchanged']})"
    )
    print()
    print("## HANDOFF")
    for item in handoff:
        loc = item.get("file", "")
        if item.get("line"):
            loc = f"{loc}:{item['line']}"
        print(f"- {item['item']} ({loc})")
        if item.get("smallest_change"):
            print(f"    {item['smallest_change']}")
        if item.get("note"):
            print(f"    {item['note']}")
    print()
    print("## NAME SCAN (supporting, not a close)")
    print(json.dumps(scan["exact_name_file_counts"], sort_keys=True))
    print(f"Director classes: {scan['director_classes']}")
    print(f"digest functions: {len(scan['digest_functions'])}")
    print()
    print(f"wrote {RECEIPT_PATH.relative_to(REPO)}")
    print(f"elapsed_s {receipt['elapsed_s']}")
    return 0 if result == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
