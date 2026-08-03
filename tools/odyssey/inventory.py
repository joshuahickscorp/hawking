"""Honest inventory of Odyssey-related data that actually exists on disk.

Walks repository-owned paths only. Never reads ~/.cache/huggingface.
Distinguishes evaluation material (must never train) from anything else.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab.layout import REPORTS_ROOT
from tools.odyssey import SCHEMA_INVENTORY
from tools.odyssey._paths import (
    DATA_DIR,
    DATA_MANIFEST,
    EVAL_DIR,
    EXPECTED_SUPPORT_HALO_CORPUS_SHA256,
    FIXTURE_DIR,
    HIDDEN_COMMITMENT,
    HIDDEN_ITEMS,
    PUBLIC_SELECTION,
    ROOT,
    SUPPORT_HALO_CORPUS,
    SUPPORT_HALO_SEAL,
    TEACHER_DIR,
    TEACHER_MANIFEST,
)
from tools.odyssey.contamination import verify_hidden_commitment, verify_support_halo_seal


def _count_jsonl(path: Path) -> int | None:
    if not path.is_file():
        return None
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            n += 1
    return n


def _bytes(path: Path) -> int | None:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total
    return None


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _entry(
    *,
    path: Path | str,
    status: str,
    role: str,
    format: str | None = None,
    items: int | None = None,
    licence: str | None = None,
    evidence: str,
    bytes_: int | None = None,
    purpose: str | None = None,
    id: str | None = None,
) -> dict[str, Any]:
    p = Path(path) if not isinstance(path, Path) else path
    return {
        "id": id or (p.name if str(p) != "." else "unknown"),
        "path": _rel(p),
        "status": status,
        "bytes": bytes_ if bytes_ is not None else _bytes(p),
        "items": items,
        "format": format,
        "licence": licence,
        "role": role,
        "purpose": purpose,
        "evidence": evidence,
    }


def check_declared_corpora() -> list[dict[str, Any]]:
    """Membership contract for each corpus in ODYSSEY_DATA_MANIFEST.json."""
    if not DATA_MANIFEST.is_file():
        return [
            _entry(
                path=DATA_MANIFEST,
                status="DECLARED_NOT_PRESENT",
                role="unknown",
                evidence="data manifest itself missing",
            )
        ]
    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for corpus in manifest.get("corpora") or []:
        cid = str(corpus.get("id"))
        candidates = [
            DATA_DIR / cid,
            DATA_DIR / f"{cid}.jsonl",
            DATA_DIR / f"{cid}.json",
            DATA_DIR / cid / "admitted.jsonl",
            DATA_DIR / "membership" / cid / "admitted.jsonl",
        ]
        if corpus.get("path"):
            candidates.insert(0, Path(corpus["path"]))
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            status = "DECLARED_NOT_PRESENT"
            items = None
            b = None
            fmt = None
            evidence = (
                f"manifest declares corpus id={cid!r} (present={corpus.get('present')}); "
                f"checked {[str(_rel(c)) for c in candidates[:4]]}; none exist on disk"
            )
        else:
            if found.is_dir():
                jsonls = list(found.glob("**/*.jsonl"))
                items = sum(_count_jsonl(j) or 0 for j in jsonls) if jsonls else 0
                fmt = "directory"
                # PARTIAL if directory exists but empty of training items
                status = "PARTIAL" if items == 0 else "PRESENT"
                evidence = f"directory present at {_rel(found)}; jsonl_items={items}"
            else:
                items = _count_jsonl(found) if found.suffix == ".jsonl" else None
                fmt = found.suffix.lstrip(".") or "file"
                status = "PRESENT" if (items is None or items > 0) else "PARTIAL"
                evidence = f"file present at {_rel(found)}; items={items}"
            b = _bytes(found)
        missing_path = DATA_DIR / cid
        out.append(
            _entry(
                id=cid,
                path=found if found else missing_path,
                status=status,
                role="train",
                format=fmt,
                items=items,
                bytes_=b,
                licence=corpus.get("license_required"),
                purpose=corpus.get("purpose"),
                evidence=evidence,
            )
        )
    return out


def inventory_evaluation() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seal = verify_support_halo_seal()
    n_halo = _count_jsonl(SUPPORT_HALO_CORPUS)
    entries.append(
        _entry(
            id="support_halo_corpus_v0",
            path=SUPPORT_HALO_CORPUS,
            status="PRESENT" if seal.get("ok") else ("PARTIAL" if SUPPORT_HALO_CORPUS.is_file() else "DECLARED_NOT_PRESENT"),
            role="eval",
            format="jsonl",
            items=n_halo,
            licence="repo-internal; sealed evaluation; must never train",
            purpose="G5/T7 support-halo tournament judge",
            evidence=(
                f"sha256_ok={seal.get('ok')} computed={seal.get('computed_sha256')} "
                f"expected={EXPECTED_SUPPORT_HALO_CORPUS_SHA256}"
            ),
        )
    )
    for name in (
        "SUPPORT_HALO_SEAL.json",
        "SUPPORT_HALO_SCORING_RULES.json",
        "SUPPORT_HALO_BASELINE.json",
        "ODYSSEY_EVALUATION_CONTRACT.json",
    ):
        p = EVAL_DIR / name
        entries.append(
            _entry(
                id=name,
                path=p,
                status="PRESENT" if p.is_file() else "DECLARED_NOT_PRESENT",
                role="eval",
                format="json",
                items=1 if p.is_file() else 0,
                licence="repo-internal",
                purpose="evaluation contract / seal / rules (not training data)",
                evidence=f"exists={p.is_file()} bytes={_bytes(p)}",
            )
        )

    hc = verify_hidden_commitment()
    n_hid = _count_jsonl(HIDDEN_ITEMS)
    entries.append(
        _entry(
            id="hidden_memberships",
            path=HIDDEN_ITEMS if HIDDEN_ITEMS.is_file() else HIDDEN_DIR,
            status="PRESENT" if hc.get("ok") else ("PARTIAL" if HIDDEN_ITEMS.is_file() else "DECLARED_NOT_PRESENT"),
            role="eval",
            format="jsonl",
            items=n_hid,
            licence="repo-internal T0 seed; evaluation only",
            purpose="held-out evaluation memberships (hash-committed, not naming-convention)",
            evidence=(
                f"commitment_ok={hc.get('ok')} n_hidden={hc.get('n_hidden')} "
                f"committed={hc.get('committed')}"
            ),
        )
    )
    if HIDDEN_COMMITMENT.is_file():
        entries.append(
            _entry(
                id="HIDDEN_MEMBERSHIP_COMMITMENT",
                path=HIDDEN_COMMITMENT,
                status="PRESENT",
                role="eval",
                format="json",
                items=1,
                licence="repo-internal",
                purpose="training-visible commitment surface for hidden set",
                evidence=f"exists bytes={_bytes(HIDDEN_COMMITMENT)}",
            )
        )
    if PUBLIC_SELECTION.is_file():
        entries.append(
            _entry(
                id="public_selection",
                path=PUBLIC_SELECTION,
                status="PRESENT",
                role="eval",
                format="jsonl",
                items=_count_jsonl(PUBLIC_SELECTION),
                licence="repo-internal T0 seed",
                purpose="selection-set eval visible to training path as items but must not train",
                evidence="T0 public selection set",
            )
        )
    return entries


def inventory_teacher() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not TEACHER_MANIFEST.is_file():
        return [
            _entry(
                path=TEACHER_MANIFEST,
                status="DECLARED_NOT_PRESENT",
                role="unknown",
                evidence="teacher manifest missing",
            )
        ]
    manifest = json.loads(TEACHER_MANIFEST.read_text(encoding="utf-8"))
    entries.append(
        _entry(
            id="ODYSSEY_TEACHER_TRACE_MANIFEST",
            path=TEACHER_MANIFEST,
            status="PRESENT",
            role="unknown",
            format="json",
            items=1,
            licence="repo-internal",
            purpose="declares teacher-trace requirements and existing pointers",
            evidence=f"manifest_status={manifest.get('status')}",
        )
    )
    existing = manifest.get("existing") or {}
    for key, meta in existing.items():
        ledger = meta.get("ledger")
        p = Path(ledger) if ledger else None
        on_disk = p is not None and p.is_file()
        n = None
        b = None
        if on_disk:
            # Count lines without loading whole file into inventory role logic.
            n = 0
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        n += 1
            b = p.stat().st_size
        entries.append(
            _entry(
                id=key,
                path=p if p else TEACHER_DIR / key,
                status="PRESENT" if on_disk else "DECLARED_NOT_PRESENT",
                role="unknown",  # teacher evidence is not train text corpus
                format="jsonl",
                items=n,
                bytes_=b,
                licence="derived from zai-org/GLM-5.2 weights under their terms; not a text train set",
                purpose=meta.get("note"),
                evidence=(
                    f"ledger_path={ledger} exists={on_disk} lines={n}; "
                    "layer-scoped capsules, not trajectory traces"
                ),
            )
        )
    return entries


def inventory_repo_fixtures() -> list[dict[str, Any]]:
    """Small in-repo fixtures that exist but are NOT Odyssey training corpora."""
    entries: list[dict[str, Any]] = []
    candidates = [
        (
            ROOT / "tools/eval/thesis_smoke_corpus_v0.jsonl",
            "eval",
            "thesis gate smoke prompts — evaluation only, never train",
        ),
        (
            ROOT / "tools/eval/thesis_rust_corpus_v0.jsonl",
            "eval",
            "thesis rust corpus — evaluation only, never train",
        ),
        (
            ROOT / "tools/training/data/rwkv7_sft_sample.jsonl",
            "fixture",
            "RWKV7 draft-model SFT sample fixture; not an Odyssey declared corpus",
        ),
        (
            REPORTS_ROOT / "condense/glm52_generation_b/generation_a_fixtures/generation_a_TEACHER_EVIDENCE_LEDGER.jsonl",
            "fixture",
            "generation-A fixture copy of teacher evidence ledger (16 lines); not T3 traces",
        ),
    ]
    for path, role, purpose in candidates:
        if path.is_file():
            entries.append(
                _entry(
                    id=path.name,
                    path=path,
                    status="PRESENT",
                    role=role,
                    format="jsonl",
                    items=_count_jsonl(path),
                    licence="repo-internal fixture",
                    purpose=purpose,
                    evidence=f"present bytes={_bytes(path)}",
                )
            )
        else:
            entries.append(
                _entry(
                    id=path.name,
                    path=path,
                    status="DECLARED_NOT_PRESENT",
                    role=role,
                    purpose=purpose,
                    evidence="path not found in this worktree",
                )
            )

    # Ingestion fixture (written by this lane).
    fix = FIXTURE_DIR / "raw_fixture.jsonl"
    if fix.is_file():
        entries.append(
            _entry(
                id="ingestion_fixture_v0",
                path=FIXTURE_DIR,
                status="PRESENT",
                role="fixture",
                format="jsonl",
                items=_count_jsonl(fix),
                licence="synthetic fixture created by odyssey-data lane; NOT real training data",
                purpose="prove ingestion + contamination barrier end-to-end",
                evidence="labelled fixture under odyssey/data/fixtures/ingestion_fixture_v0/",
            )
        )
    return entries


def build_inventory() -> dict[str, Any]:
    declared = check_declared_corpora()
    evaluation = inventory_evaluation()
    teacher = inventory_teacher()
    fixtures = inventory_repo_fixtures()
    corpora = declared + evaluation + teacher + fixtures

    n_present = sum(1 for c in corpora if c["status"] == "PRESENT")
    n_missing = sum(1 for c in corpora if c["status"] == "DECLARED_NOT_PRESENT")
    n_partial = sum(1 for c in corpora if c["status"] == "PARTIAL")

    train_ready = [
        c for c in declared if c["status"] == "PRESENT" and c.get("role") == "train"
    ]

    return {
        "schema": SCHEMA_INVENTORY,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "TRAINING_CORPORA_ABSENT" if not train_ready else "PARTIAL",
        "summary": {
            "n_entries": len(corpora),
            "n_present": n_present,
            "n_declared_not_present": n_missing,
            "n_partial": n_partial,
            "n_declared_train_corpora": len(declared),
            "n_declared_train_corpora_present": sum(
                1 for c in declared if c["status"] == "PRESENT"
            ),
            "n_eval_entries": len(evaluation),
            "binding_constraint": "missing authorized training corpora (not memory)",
        },
        "manifest_claim": {
            "path": _rel(DATA_MANIFEST),
            "status_field": json.loads(DATA_MANIFEST.read_text(encoding="utf-8")).get("status")
            if DATA_MANIFEST.is_file()
            else None,
            "note": "manifest present flag is a claim; membership_check is the mechanical truth",
        },
        "corpora": corpora,
        "invariants": [
            "evaluation material must never enter training",
            "DECLARED_NOT_PRESENT is a successful classification, not a quiet skip",
            "fixtures are labelled fixture and are not production corpora",
            "no network acquisition performed by this inventory",
        ],
    }


def write_inventory(path: Path | None = None) -> dict[str, Any]:
    inv = build_inventory()
    path = path or (DATA_DIR / "ODYSSEY_DATA_INVENTORY.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inv, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return inv
