#!/usr/bin/env python3.12
"""Freeze train/dev/test memberships for Ramanujan corpora.

The split is sealed by hash so it cannot drift later. Assignment is a pure
function of each item's existing `content_hash` (bucketed into 80/10/10).

The generation-time `split` field on every item is currently the provisional
value "train". That field is NOT mutated here: mutating it would change
`content_hash` (which includes `split` in the stamped body). The sealed
MEMBERSHIP_MANIFEST is the sole authority for train/dev/test membership.

Also re-runs the Odyssey contamination barrier and records the negative
control: a sealed support-halo item presented as training text must come
back `exact_match`.

    nice -n 15 python3.12 -m ramanujan.data.freeze_memberships

Does not flip RAMANUJAN_RESEARCH_AUTHORIZED. Does not touch Math-Preserve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from ramanujan.data.common import file_sha256, read_jsonl
from ramanujan.data.paths import CORPORA, ROOT, SOURCE_FILES

SCHEMA = "hawking.ramanujan.membership_manifest.v1"
SEED_LABEL = "ramanujan-freeze-v1-content_hash_bucket"
# 80 / 10 / 10 by content_hash bucket (stable forever for a given hash).
TRAIN_PCT = 80
DEV_PCT = 10  # buckets [80, 90)
# TEST_PCT = 10  # buckets [90, 100)

MANIFEST_PATH = CORPORA / "MEMBERSHIP_MANIFEST.json"
FREEZE_RECEIPT = CORPORA / "FREEZE_RECEIPT.json"
EXTERNAL_INTAKE_SCHEMA = "hawking.ramanujan.external_source_owner_intake.v1"
EXTERNAL_RECEIPT_SCHEMA = "hawking.ramanujan.external_source_freeze_receipt.v1"
EXTERNAL_SOURCE_IDS = ("D5", "D8", "D9")
_LOWER_HEX = frozenset("0123456789abcdef")


def assign_split(content_hash: str) -> str:
    """Deterministic split from content_hash. Pure; no RNG, no wall clock."""
    if not content_hash or len(content_hash) < 8:
        raise ValueError(f"content_hash too short for freeze assignment: {content_hash!r}")
    bucket = int(content_hash[:8], 16) % 100
    if bucket < TRAIN_PCT:
        return "train"
    if bucket < TRAIN_PCT + DEV_PCT:
        return "dev"
    return "test"


def _canonical_assignment_blob(per_source: dict[str, Any]) -> bytes:
    """Canonical bytes over which membership_sha256 is taken.

    Only (source_id, content_hash, split) triples, sorted. Counts and paths
    are derived and must not affect the seal.
    """
    triples: list[list[str]] = []
    for sid in sorted(per_source):
        block = per_source[sid]
        for split_name in ("train", "dev", "test"):
            for h in block["by_split"][split_name]:
                triples.append([sid, h, split_name])
    triples.sort()
    return json.dumps(
        {
            "seed_label": SEED_LABEL,
            "train_pct": TRAIN_PCT,
            "dev_pct": DEV_PCT,
            "triples": triples,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sealed(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
    return {**unsigned, "seal_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest()}


def _verify_sealed(value: dict[str, Any], label: str) -> None:
    recorded = value.get("seal_sha256")
    expected = _sealed(value)["seal_sha256"]
    if not isinstance(recorded, str) or recorded != expected:
        raise ValueError(f"{label} seal mismatch")


def _verified_file(binding: Any, label: str, *, executable: bool = False) -> Path:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError(f"{label} must bind exactly path and sha256")
    raw_path, expected = binding.get("path"), binding.get("sha256")
    if (
        not isinstance(raw_path, str) or not raw_path
        or not isinstance(expected, str) or len(expected) != 64
        or any(char not in _LOWER_HEX for char in expected)
    ):
        raise ValueError(f"{label} has an invalid path or sha256")
    path = Path(raw_path).expanduser()
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} is missing or differs from its exact sha256")
    if executable and not os.access(path, os.X_OK):
        raise ValueError(f"{label} must be executable")
    return path


def external_intake_template() -> dict[str, Any]:
    """Return the exact owner-fillable schema without granting approval."""
    rows: list[dict[str, Any]] = []
    for source_id in EXTERNAL_SOURCE_IDS:
        generator = None
        if source_id == "D9":
            generator = {
                "actor": "<distinct-variant-generator-role>",
                "path": "<owner-approved-executable-path>",
                "sha256": "<64-lowercase-hex>",
                "seed_commitment_sha256": "<64-lowercase-hex>",
            }
        rows.append({
            "id": source_id,
            "owner_approved": False,
            "owner_actor": "<owner-role>",
            "version": "<immutable-source-version>",
            "source": {"path": "<licensed-jsonl-path>", "sha256": "<64-lowercase-hex>"},
            "license": {"spdx": "<SPDX-or-LicenseRef>", "path": "<license-text-path>", "sha256": "<64-lowercase-hex>"},
            "membership_sealer_actor": "<distinct-membership-sealer-role>",
            "adjudicator": {
                "actor": "<distinct-independent-adjudicator-role>",
                "path": "<owner-approved-executable-path>",
                "sha256": "<64-lowercase-hex>",
            },
            "variant_generator": generator,
        })
    return {
        "schema": EXTERNAL_INTAKE_SCHEMA,
        "status": "PENDING_OWNER_APPROVAL",
        "owner_authority_receipt": {"path": "<owner-authority-receipt-path>", "sha256": "<64-lowercase-hex>"},
        "candidate_launch_started": False,
        "sources": rows,
        "seal_sha256": "<sha256-of-canonical-object-without-seal_sha256>",
    }


def _canonical_jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], str]:
    objects: list[dict[str, Any]] = []
    canonical_lines: list[bytes] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{number} is not JSON") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{label}:{number} must be a JSON object")
        objects.append(item)
        canonical_lines.append(_canonical_json(item))
    if not objects:
        raise ValueError(f"{label} must contain at least one item")
    if len(canonical_lines) != len(set(canonical_lines)):
        raise ValueError(f"{label} contains duplicate canonical items")
    digest = hashlib.sha256()
    for line in canonical_lines:
        digest.update(line)
        digest.update(b"\n")
    return objects, digest.hexdigest()


def _external_no_leak_audit(source_id: str, objects: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply Odyssey's exact/near-duplicate law without exposing item ids."""
    from tools.odyssey._paths import JACCARD_TRAIN_VS_EVAL, SHINGLE_SIZE
    from tools.odyssey.contamination import build_barrier
    from tools.odyssey.dedup import char_shingles, content_sha256, jaccard
    from tools.odyssey.normalize import extract_comparison_text

    texts = [extract_comparison_text(item) for item in objects]
    if any(not text.strip() for text in texts):
        raise ValueError(f"{source_id} contains an item with no comparison text")
    exact = [content_sha256(text) for text in texts]
    if len(exact) != len(set(exact)):
        raise ValueError(f"{source_id} contains duplicate normalized evaluation text")

    comparisons = 0
    if source_id == "D5":
        barrier = build_barrier()
        hits = 0
        for text in texts:
            comparisons += 1
            if barrier.check(text):
                hits += 1
        if hits:
            raise ValueError("D5 overlaps sealed evaluation material")
        return {
            "direction": "D5_training_candidate_against_sealed_odyssey_evaluation",
            "items_checked": len(texts),
            "comparisons": comparisons,
            "exact_or_near_matches": 0,
            "jaccard_threshold": JACCARD_TRAIN_VS_EVAL,
            "shingle_size": SHINGLE_SIZE,
        }
    if source_id != "D8":
        return {
            "direction": "NOT_TRAINING_VISIBLE_SOURCE",
            "items_checked": len(texts),
            "comparisons": 0,
            "exact_or_near_matches": 0,
            "jaccard_threshold": JACCARD_TRAIN_VS_EVAL,
            "shingle_size": SHINGLE_SIZE,
        }

    hidden_exact = set(exact)
    hidden_shingles = [char_shingles(text, SHINGLE_SIZE) for text in texts]
    hits = 0
    training_items = 0
    for path in SOURCE_FILES.values():
        for item in read_jsonl(path):
            training_text = extract_comparison_text(item)
            if not training_text.strip():
                continue
            training_items += 1
            training_hash = content_sha256(training_text)
            comparisons += len(texts)
            if training_hash in hidden_exact:
                hits += 1
                continue
            shingles = char_shingles(training_text, SHINGLE_SIZE)
            if any(jaccard(shingles, target) >= JACCARD_TRAIN_VS_EVAL for target in hidden_shingles):
                hits += 1
    if hits:
        raise ValueError("D8 hidden membership overlaps current frozen training material")
    return {
        "direction": "all_current_D1_D7_training_against_D8_hidden_membership",
        "items_checked": len(texts),
        "training_items_checked": training_items,
        "comparisons": comparisons,
        "exact_or_near_matches": 0,
        "jaccard_threshold": JACCARD_TRAIN_VS_EVAL,
        "shingle_size": SHINGLE_SIZE,
    }


def freeze_external_sources(
    intake_path: Path,
    *,
    receipt_path: Path,
) -> dict[str, Any]:
    """Verify owner-bound D5/D8/D9 inputs and emit a non-authorizing freeze.

    The public receipt deliberately contains no source path and no hidden D8
    item id.  D8's secret JSONL must already have owner-only filesystem mode;
    this authority never copies it into the repository.  D9's generator is
    hash-sealed but not executed here, keeping owner approval distinct from a
    future candidate or counterexample run.
    """
    try:
        intake = json.loads(intake_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"external source intake is unreadable: {exc}") from exc
    if not isinstance(intake, dict):
        raise ValueError("external source intake root must be an object")
    _verify_sealed(intake, "external source intake")
    required_root = {
        "schema", "status", "owner_authority_receipt", "candidate_launch_started",
        "sources", "seal_sha256",
    }
    if set(intake) != required_root or intake.get("schema") != EXTERNAL_INTAKE_SCHEMA:
        raise ValueError("external source intake schema is incomplete or has unknown fields")
    if intake.get("status") != "OWNER_APPROVED" or intake.get("candidate_launch_started") is not False:
        raise ValueError("owner approval must precede every candidate launch")
    owner_authority = _verified_file(intake.get("owner_authority_receipt"), "owner authority receipt")
    rows = intake.get("sources")
    if not isinstance(rows, list) or [row.get("id") for row in rows if isinstance(row, dict)] != list(EXTERNAL_SOURCE_IDS):
        raise ValueError(f"external sources must be exactly {EXTERNAL_SOURCE_IDS} in order")

    public_rows: list[dict[str, Any]] = []
    for row in rows:
        required = {
            "id", "owner_approved", "owner_actor", "version", "source", "license",
            "membership_sealer_actor", "adjudicator", "variant_generator",
        }
        if not isinstance(row, dict) or set(row) != required or row.get("owner_approved") is not True:
            raise ValueError("every external source row needs the complete owner-approved schema")
        source_id = str(row["id"])
        source_path = _verified_file(row.get("source"), f"{source_id} source")
        license_binding = row.get("license")
        if not isinstance(license_binding, dict) or set(license_binding) != {"spdx", "path", "sha256"}:
            raise ValueError(f"{source_id} license must bind spdx, path, and sha256")
        spdx = license_binding.get("spdx")
        if not isinstance(spdx, str) or not spdx or spdx == "PENDING":
            raise ValueError(f"{source_id} license SPDX remains pending")
        license_path = _verified_file(
            {"path": license_binding["path"], "sha256": license_binding["sha256"]},
            f"{source_id} license text",
        )
        adjudicator = row.get("adjudicator")
        if not isinstance(adjudicator, dict) or set(adjudicator) != {"actor", "path", "sha256"}:
            raise ValueError(f"{source_id} adjudicator binding is incomplete")
        adjudicator_path = _verified_file(
            {"path": adjudicator["path"], "sha256": adjudicator["sha256"]},
            f"{source_id} adjudicator",
            executable=True,
        )
        owner_actor = row.get("owner_actor")
        sealer_actor = row.get("membership_sealer_actor")
        adjudicator_actor = adjudicator.get("actor")
        actors = [owner_actor, sealer_actor, adjudicator_actor]
        if any(not isinstance(actor, str) or not actor.strip() for actor in actors) or len(set(actors)) != 3:
            raise ValueError(f"{source_id} owner, sealer, and adjudicator roles must be distinct")
        version = row.get("version")
        if not isinstance(version, str) or not version or version == "PENDING":
            raise ValueError(f"{source_id} immutable version is absent")

        objects, canonical_commitment = _canonical_jsonl(source_path, f"{source_id} source")
        no_leak_audit = _external_no_leak_audit(source_id, objects)
        generator_public: dict[str, Any] | None = None
        if source_id == "D8":
            if source_path.stat().st_mode & 0o077:
                raise ValueError("D8 hidden source must have owner-only filesystem mode")
            if any(item.get("set", item.get("split")) not in {"hidden", "held_out", "eval"} for item in objects):
                raise ValueError("every D8 item must be mechanically marked hidden/held_out/eval")
            if row.get("variant_generator") is not None:
                raise ValueError("D8 must not substitute a variant generator for hidden membership")
        elif source_id == "D9":
            generator = row.get("variant_generator")
            if not isinstance(generator, dict) or set(generator) != {"actor", "path", "sha256", "seed_commitment_sha256"}:
                raise ValueError("D9 variant generator binding is incomplete")
            generator_path = _verified_file(
                {"path": generator["path"], "sha256": generator["sha256"]},
                "D9 variant generator",
                executable=True,
            )
            generator_actor = generator.get("actor")
            seed = generator.get("seed_commitment_sha256")
            if (
                not isinstance(generator_actor, str) or not generator_actor.strip()
                or generator_actor in set(actors)
                or not isinstance(seed, str) or len(seed) != 64
                or any(char not in _LOWER_HEX for char in seed)
            ):
                raise ValueError("D9 generator role/seed commitment is invalid or not independent")
            generator_public = {
                "actor": generator_actor,
                "executable_sha256": file_sha256(generator_path),
                "locator_sha256": hashlib.sha256(str(generator_path.resolve()).encode()).hexdigest(),
                "seed_commitment_sha256": seed,
                "executed_by_freeze": False,
            }
        elif row.get("variant_generator") is not None:
            raise ValueError(f"{source_id} does not admit a variant generator")

        public_rows.append(
            {
                "id": source_id,
                "status": "FROZEN_PENDING_INDEPENDENT_EVALUATION",
                "version": version,
                "n_items": len(objects),
                "source_sha256": file_sha256(source_path),
                "canonical_membership_commitment_sha256": canonical_commitment,
                "source_locator_sha256": hashlib.sha256(str(source_path.resolve()).encode()).hexdigest(),
                "source_path_or_item_ids_serialized": False,
                "license": {"spdx": spdx, "text_sha256": file_sha256(license_path)},
                "roles": {
                    "owner": owner_actor,
                    "membership_sealer": sealer_actor,
                    "independent_adjudicator": adjudicator_actor,
                },
                "adjudicator": {
                    "executable_sha256": file_sha256(adjudicator_path),
                    "locator_sha256": hashlib.sha256(str(adjudicator_path.resolve()).encode()).hexdigest(),
                    "executed_by_freeze": False,
                },
                "variant_generator": generator_public,
                "no_leak_audit": no_leak_audit,
            }
        )

    receipt = _sealed(
        {
            "schema": EXTERNAL_RECEIPT_SCHEMA,
            "status": "PASS_INPUTS_FROZEN_RESEARCH_AND_CANDIDATE_AUTHORITY_FALSE",
            "intake_seal_sha256": intake["seal_sha256"],
            "owner_authority_receipt_sha256": file_sha256(owner_authority),
            "sources": public_rows,
            "training_visible": {
                "D8_hidden_item_ids": None,
                "D8_commitment_only": True,
                "D9_generator_executed": False,
            },
            "RAMANUJAN_RESEARCH_AUTHORIZED": False,
            "candidate_launch_authorized": False,
            "independent_adjudication_complete": False,
            "counterexample_search_complete": False,
        }
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def freeze_memberships(
    *,
    source_files: dict[str, Path] | None = None,
    manifest_path: Path | None = None,
    receipt_path: Path | None = None,
    run_contamination: bool = True,
) -> dict[str, Any]:
    files = source_files or SOURCE_FILES
    per_source: dict[str, Any] = {}
    global_counts = {"train": 0, "dev": 0, "test": 0, "total": 0}
    all_hashes: list[str] = []

    for sid, path in files.items():
        items = read_jsonl(path) if path.is_file() else []
        by_split: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
        id_to_split: dict[str, str] = {}
        missing_hash = 0
        for it in items:
            h = it.get("content_hash")
            if not h:
                missing_hash += 1
                continue
            split = assign_split(h)
            by_split[split].append(h)
            id_to_split[str(it.get("id"))] = split
            all_hashes.append(h)
            global_counts[split] += 1
            global_counts["total"] += 1
        for k in by_split:
            by_split[k] = sorted(by_split[k])
        try:
            rel = str(path.relative_to(ROOT))
        except ValueError:
            rel = str(path)
        per_source[sid] = {
            "path": rel,
            "file_sha256": file_sha256(path) if path.is_file() else None,
            "n_items": len(items),
            "n_missing_content_hash": missing_hash,
            "n": {k: len(v) for k, v in by_split.items()},
            "by_split": by_split,
            # id map is audit-only; seal does not cover it (ids are not the address).
            "id_to_split_sample": dict(list(id_to_split.items())[:5]),
            "n_ids_mapped": len(id_to_split),
        }

    assignment_blob = _canonical_assignment_blob(per_source)
    membership_sha256 = hashlib.sha256(assignment_blob).hexdigest()

    # Corpus-level hash of all content hashes (order-independent via sort).
    corpus_items_sha256 = hashlib.sha256(
        ("\n".join(sorted(all_hashes)) + "\n").encode("utf-8")
    ).hexdigest()

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_label": SEED_LABEL,
        "assignment_rule": (
            f"bucket = int(content_hash[:8], 16) % 100; "
            f"[0,{TRAIN_PCT})->train, [{TRAIN_PCT},{TRAIN_PCT + DEV_PCT})->dev, "
            f"else->test"
        ),
        "ratios": {"train": TRAIN_PCT / 100, "dev": DEV_PCT / 100, "test": (100 - TRAIN_PCT - DEV_PCT) / 100},
        "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        "teacher_from_math_preserve": False,
        "note": (
            "MEMBERSHIP_MANIFEST is the sole authority for train/dev/test. "
            "Item bodies keep generation-time split='train' so content_hash "
            "identity stays stable; loaders must use this manifest."
        ),
        "counts": global_counts,
        "corpus_items_sha256": corpus_items_sha256,
        "membership_sha256": membership_sha256,
        "per_source": per_source,
        "invariants": [
            "assignment is a pure function of content_hash",
            "membership_sha256 seals (source, content_hash, split) triples only",
            "no training loop may read dev/test hashes as train",
            "support-halo / hidden memberships remain outside this freeze",
        ],
    }

    out = manifest_path or MANIFEST_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_file_sha256 = file_sha256(out)

    # Contamination barrier + negative control.
    contamination_summary: dict[str, Any] = {"ran": False}
    negative_control: dict[str, Any] = {"ran": False}
    if run_contamination:
        from tools.odyssey.contamination import (
            barrier_rules_document,
            build_barrier,
            verify_support_halo_seal,
        )
        from tools.odyssey.normalize import extract_comparison_text
        from tools.odyssey._paths import SUPPORT_HALO_CORPUS

        seal = verify_support_halo_seal()
        barrier = build_barrier()
        rules = barrier_rules_document(barrier)

        total_in = 0
        total_rejected = 0
        for sid, path in files.items():
            for it in read_jsonl(path) if path.is_file() else []:
                total_in += 1
                text = it.get("text") or it.get("statement") or it.get("goal") or ""
                if barrier.check(str(text)):
                    total_rejected += 1

        # Negative control: sealed support-halo item must exact-match.
        halo_items = [
            json.loads(ln)
            for ln in SUPPORT_HALO_CORPUS.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        # Prefer the known tl02_bpw probe; fall back to first item.
        probe = next((x for x in halo_items if x.get("id") == "tl02_bpw"), halo_items[0])
        probe_text = extract_comparison_text(probe)
        hits = barrier.check(probe_text)
        negative_control = {
            "ran": True,
            "description": (
                f"sealed support-halo item ({probe.get('id')}) presented as training text"
            ),
            "probe_id": probe.get("id"),
            "rejected": bool(hits),
            "reason": hits[0].reason if hits else None,
            "eval_source": hits[0].eval_source if hits else None,
            "eval_id": hits[0].eval_id if hits else None,
            "pass": bool(hits) and any(h.reason == "exact_match" for h in hits),
        }
        contamination_summary = {
            "ran": True,
            "support_halo_seal_ok": bool(seal.get("ok")),
            "support_halo_sha256": seal.get("computed_sha256"),
            "n_eval_items_indexed": len(barrier.eval_items),
            "total_corpus_items_checked": total_in,
            "total_rejected_vs_eval": total_rejected,
            "barrier": rules,
            "negative_control": negative_control,
        }

    # Verify seal recompute.
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    recompute = hashlib.sha256(
        _canonical_assignment_blob(reloaded["per_source"])
    ).hexdigest()
    seal_ok = recompute == membership_sha256

    receipt = {
        "schema": "hawking.ramanujan.freeze_receipt.v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_path": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
        "manifest_file_sha256": manifest_file_sha256,
        "membership_sha256": membership_sha256,
        "membership_seal_recomputed_ok": seal_ok,
        "counts": global_counts,
        "contamination": contamination_summary,
        "negative_control_pass": bool(negative_control.get("pass")),
        "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        "status": (
            "PASS"
            if seal_ok and (not run_contamination or negative_control.get("pass"))
            else "FAIL"
        ),
    }
    rpath = receipt_path or FREEZE_RECEIPT
    rpath.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": manifest, "receipt": receipt}


def load_membership(path: Path | None = None) -> dict[str, Any]:
    p = path or MANIFEST_PATH
    if not p.is_file():
        raise FileNotFoundError(f"membership manifest missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def verify_membership_sources(
    path: Path | None = None,
    *,
    source_files: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Compare current corpus membership with the committed manifest.

    The manifest seal alone only proves that the manifest has not changed.  A
    corpus can change after freezing while retaining a syntactically valid
    manifest, which would otherwise let training silently skip or admit
    records.  This check deliberately compares content-hash memberships and
    not JSONL file hashes because provenance timestamps make byte-identical
    regeneration impossible.
    """
    manifest = load_membership(path)
    files = source_files or SOURCE_FILES
    expected_sources = set(manifest.get("per_source", {}))
    actual_sources = set(files)
    details: dict[str, Any] = {}

    for source_id in sorted(expected_sources | actual_sources):
        block = manifest.get("per_source", {}).get(source_id)
        corpus_path = files.get(source_id)
        if not isinstance(block, dict) or corpus_path is None:
            details[source_id] = {
                "ok": False,
                "reason": "MISSING_MANIFEST_SOURCE" if not isinstance(block, dict) else "MISSING_CURRENT_SOURCE",
            }
            continue

        expected_by_split = block.get("by_split")
        if not isinstance(expected_by_split, dict) or set(expected_by_split) != {"train", "dev", "test"}:
            details[source_id] = {"ok": False, "reason": "INVALID_MANIFEST_SPLITS"}
            continue

        current_by_split: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
        missing_hashes = 0
        for item in read_jsonl(corpus_path):
            content = item.get("content_hash")
            if not isinstance(content, str) or not content:
                missing_hashes += 1
                continue
            current_by_split[assign_split(content)].append(content)
        for split in current_by_split:
            current_by_split[split].sort()

        expected = {split: list(expected_by_split[split]) for split in current_by_split}
        current_total = sum(len(values) for values in current_by_split.values())
        expected_total = sum(len(values) for values in expected.values())
        missing = {
            split: sorted(set(expected[split]) - set(current_by_split[split]))
            for split in current_by_split
        }
        unexpected = {
            split: sorted(set(current_by_split[split]) - set(expected[split]))
            for split in current_by_split
        }
        membership_match = all(current_by_split[split] == expected[split] for split in current_by_split)
        count_match = current_total == expected_total == block.get("n_items")
        details[source_id] = {
            "ok": membership_match and count_match and missing_hashes == 0,
            "current_count": current_total,
            "committed_count": expected_total,
            "manifest_n_items": block.get("n_items"),
            "missing_content_hashes": missing,
            "unexpected_content_hashes": unexpected,
            "missing_content_hash_count": missing_hashes,
        }

    return {
        "ok": expected_sources == actual_sources and all(item.get("ok") is True for item in details.values()),
        "expected_sources": sorted(expected_sources),
        "current_sources": sorted(actual_sources),
        "sources": details,
    }


def verify_membership_seal(
    path: Path | None = None,
    *,
    source_files: dict[str, Path] | None = None,
) -> dict[str, Any]:
    m = load_membership(path)
    recomputed = hashlib.sha256(_canonical_assignment_blob(m["per_source"])).hexdigest()
    manifest_ok = recomputed == m.get("membership_sha256")
    sources = verify_membership_sources(path, source_files=source_files)
    return {
        "ok": manifest_ok and sources["ok"],
        "manifest_ok": manifest_ok,
        "sources_ok": sources["ok"],
        "committed": m.get("membership_sha256"),
        "recomputed": recomputed,
        "counts": m.get("counts"),
        "sources": sources,
    }


def split_of(content_hash: str, membership: dict[str, Any] | None = None) -> str:
    """Look up split for a content_hash. Falls back to pure assign_split."""
    if membership is None:
        return assign_split(content_hash)
    for sid, block in membership.get("per_source", {}).items():
        for split_name, hashes in block.get("by_split", {}).items():
            # linear ok for 16k; callers should use index_for_source for bulk
            if content_hash in hashes:
                return split_name
    return assign_split(content_hash)


def index_for_source(membership: dict[str, Any], source_id: str) -> dict[str, str]:
    """content_hash -> split for one source."""
    block = membership["per_source"][source_id]
    out: dict[str, str] = {}
    for split_name, hashes in block["by_split"].items():
        for h in hashes:
            out[h] = split_name
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-contamination", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--external-intake", type=Path)
    p.add_argument("--external-receipt", type=Path)
    p.add_argument("--print-external-template", action="store_true")
    args = p.parse_args(argv)

    if args.print_external_template:
        if args.external_intake is not None or args.external_receipt is not None or args.verify_only:
            p.error("--print-external-template cannot be combined with another mode")
        print(json.dumps(external_intake_template(), indent=2, sort_keys=True))
        return 0

    if args.external_intake is not None:
        if args.external_receipt is None:
            p.error("--external-intake requires --external-receipt")
        receipt = freeze_external_sources(args.external_intake, receipt_path=args.external_receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.external_receipt is not None:
        p.error("--external-receipt requires --external-intake")

    if args.verify_only:
        result = verify_membership_seal()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1

    out = freeze_memberships(run_contamination=not args.skip_contamination)
    receipt = out["receipt"]
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
