#!/usr/bin/env python3
"""Build Flash-Next's exact, resumable source-shard identity ledger.

The acquisition path verified Hub LFS object IDs but retained only aggregate
counts in the ModelLake manifest.  The external source-teacher gate needs the
per-shard evidence.  This producer performs one streaming SHA-256 pass over
every indexed shard, checks each digest against its retained Hugging Face LFS
metadata, and atomically checkpoints progress after every shard.

It never loads the model, starts a GPU/provider process, or authorizes a source
teacher.  Its final receipt is only one input to the independently produced,
owner-authorized V2 source-reference contract.
"""
from __future__ import annotations

import argparse
import json
import stat
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hawking.persist import (  # noqa: E402
    atomic_write_json,
    read_json_object_or_none,
    sha256_bytes,
    sha256_file_or_none,
)
from tools.odyssey import flash_repeated_accepted_decode as gate  # noqa: E402


PROGRESS_SCHEMA = "hawking.flash.source_shard_ledger_progress.v1"
PROGRESS_STATUS = "HASHING_EXACT_SOURCE_SHARDS"
DEFAULT_OUT = ROOT / "receipts/headless/FLASH_SOURCE_SHARD_LEDGER_20260912.json"


def _canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _seal(document: dict[str, Any]) -> dict[str, Any]:
    body = dict(document)
    body.pop("seal_sha256", None)
    body["seal_sha256"] = gate._compact_utf8_sorted_seal(body)
    return body


def _regular_single_link(path: Path, label: str) -> tuple[Path, dict[str, int]]:
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular file: {resolved}")
    if info.st_nlink != 1:
        raise ValueError(f"{label} must not be hard-linked: {resolved}")
    return resolved, {
        "device": info.st_dev,
        "inode": info.st_ino,
        "nlink": info.st_nlink,
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _metadata_oid(model_root: Path, shard_name: str) -> tuple[str, str]:
    metadata_path = model_root / ".cache" / "huggingface" / "download" / f"{shard_name}.metadata"
    metadata, _ = _regular_single_link(metadata_path, f"Hub metadata for {shard_name}")
    try:
        lines = metadata.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read Hub metadata for {shard_name}") from exc
    if len(lines) < 2 or lines[0] != gate.PINNED_REVISION:
        raise ValueError(f"Hub metadata for {shard_name} does not bind the pinned revision")
    expected = gate._require_sha256(lines[1], f"Hub LFS SHA-256 for {shard_name}")
    return expected, sha256_file_or_none(metadata) or ""


def _source_inputs(model_root: Path) -> dict[str, Any]:
    model_root = model_root.expanduser().resolve()
    manifest_path = gate._canonical_model_lake_manifest(model_root)
    manifest_sha256 = sha256_file_or_none(manifest_path)
    if manifest_sha256 is None:
        raise ValueError("canonical ModelLake manifest is unreadable")
    gate._verify_model_lake_manifest(
        {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "repo": gate.REPO_ID,
            "revision": gate.PINNED_REVISION,
        },
        contract_dir=manifest_path.parent,
        model_root=model_root,
    )
    index_path = gate._source_file(
        model_root,
        "model.safetensors.index.json",
        "source safetensors index",
    )
    index_sha256 = sha256_file_or_none(index_path)
    if index_sha256 is None:
        raise ValueError("source safetensors index is unreadable")
    index = gate._load_safetensors_index(index_path)
    names = sorted(set(index["weight_map"].values()))
    inventory: list[dict[str, Any]] = []
    for name in names:
        shard = gate._source_file(model_root, name, f"indexed source shard {name}")
        shard, fingerprint = _regular_single_link(shard, f"indexed source shard {name}")
        expected_sha256, metadata_sha256 = _metadata_oid(model_root, name)
        inventory.append({
            "path": name,
            "absolute_path": str(shard),
            "bytes": fingerprint["bytes"],
            "expected_lfs_sha256": expected_sha256,
            "hub_metadata_sha256": metadata_sha256,
            "source_stat": fingerprint,
        })
    identity = {
        "model": gate.REPO_ID,
        "pinned_revision": gate.PINNED_REVISION,
        "model_root": str(model_root),
        "model_lake_manifest_sha256": manifest_sha256,
        "safetensors_index_sha256": index_sha256,
        "indexed_shard_count": len(inventory),
        "indexed_shard_bytes": sum(row["bytes"] for row in inventory),
        "inventory_sha256": _canonical_sha256(inventory),
    }
    return {
        "identity": identity,
        "manifest_path": manifest_path,
        "index_path": index_path,
        "inventory": inventory,
    }


def _progress_path(out: Path) -> Path:
    return out.with_name(f"{out.name}.progress.json")


def _load_progress(path: Path, identity: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    progress = read_json_object_or_none(path)
    if progress is None:
        raise ValueError(f"source ledger progress is not a JSON object: {path}")
    gate._verify_compact_seal(progress, "source shard ledger progress")
    if (
        progress.get("schema") != PROGRESS_SCHEMA
        or progress.get("status") != PROGRESS_STATUS
        or progress.get("source") != identity
    ):
        raise ValueError("source shard ledger progress belongs to a different source inventory")
    records = progress.get("completed_shards")
    if not isinstance(records, list):
        raise ValueError("source shard ledger progress omitted completed_shards")
    by_name: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("source shard ledger progress contains a malformed record")
        if record["path"] in by_name:
            raise ValueError("source shard ledger progress contains a duplicate shard")
        by_name[record["path"]] = record
    return by_name


def _write_progress(
    path: Path,
    identity: dict[str, Any],
    records: dict[str, dict[str, Any]],
    *,
    started_unix_ns: int,
) -> None:
    atomic_write_json(path, _seal({
        "schema": PROGRESS_SCHEMA,
        "status": PROGRESS_STATUS,
        "source": identity,
        "started_unix_ns": started_unix_ns,
        "updated_unix_ns": time.time_ns(),
        "completed_shards": [records[name] for name in sorted(records)],
        "completed_shard_count": len(records),
        "completed_shard_bytes": sum(record["bytes"] for record in records.values()),
        "claim_boundary": "resumable hashing progress only; not a complete source ledger or teacher admission",
    }))


def _record_reusable(record: dict[str, Any], inventory: dict[str, Any]) -> bool:
    return (
        record.get("path") == inventory["path"]
        and record.get("bytes") == inventory["bytes"]
        and record.get("sha256") == inventory["expected_lfs_sha256"]
        and record.get("expected_lfs_sha256") == inventory["expected_lfs_sha256"]
        and record.get("hub_metadata_sha256") == inventory["hub_metadata_sha256"]
        and record.get("source_stat") == inventory["source_stat"]
        and record.get("verification") == "SHA256_EXACT_FILE_BYTES"
    )


def build_source_shard_ledger(
    model_root: Path,
    out: Path,
    *,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    source = _source_inputs(model_root)
    identity = source["identity"]
    out = out.expanduser().resolve()
    progress_path = _progress_path(out) if progress_path is None else progress_path.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing source shard ledger: {out}")
    if out == progress_path:
        raise ValueError("source shard ledger and progress paths must differ")
    started = time.time_ns()
    records = _load_progress(progress_path, identity)
    inventory_names = {row["path"] for row in source["inventory"]}
    if not set(records).issubset(inventory_names):
        raise ValueError("source shard ledger progress contains a shard outside the current index")

    total = len(source["inventory"])
    for position, row in enumerate(source["inventory"], start=1):
        prior = records.get(row["path"])
        if prior is not None and _record_reusable(prior, row):
            print(f"[{position}/{total}] reuse {row['path']}", file=sys.stderr, flush=True)
            continue
        shard = Path(row["absolute_path"])
        _, before = _regular_single_link(shard, f"indexed source shard {row['path']}")
        print(
            f"[{position}/{total}] sha256 {row['path']} ({row['bytes']} bytes)",
            file=sys.stderr,
            flush=True,
        )
        observed = sha256_file_or_none(shard)
        if observed is None:
            raise ValueError(f"cannot hash indexed source shard {row['path']}")
        _, after = _regular_single_link(shard, f"indexed source shard {row['path']}")
        if before != after or after != row["source_stat"]:
            raise ValueError(f"indexed source shard changed while hashing: {row['path']}")
        if observed != row["expected_lfs_sha256"]:
            raise ValueError(f"indexed source shard differs from its retained Hub LFS identity: {row['path']}")
        records[row["path"]] = {
            "path": row["path"],
            "bytes": row["bytes"],
            "sha256": observed,
            "expected_lfs_sha256": row["expected_lfs_sha256"],
            "hub_metadata_sha256": row["hub_metadata_sha256"],
            "source_stat": row["source_stat"],
            "verification": "SHA256_EXACT_FILE_BYTES",
        }
        _write_progress(progress_path, identity, records, started_unix_ns=started)

    ordered = [records[row["path"]] for row in source["inventory"]]
    if len(ordered) != identity["indexed_shard_count"] or any(
        not _record_reusable(record, inventory)
        for record, inventory in zip(ordered, source["inventory"], strict=True)
    ):
        raise ValueError("source shard ledger did not close over the exact current inventory")
    receipt = _seal({
        "schema": gate.SOURCE_SHARD_LEDGER_SCHEMA,
        "status": gate.SOURCE_SHARD_LEDGER_STATUS,
        "source": {
            "model": gate.REPO_ID,
            "pinned_revision": gate.PINNED_REVISION,
            "model_root": identity["model_root"],
            "model_lake_manifest_sha256": identity["model_lake_manifest_sha256"],
            "safetensors_index_sha256": identity["safetensors_index_sha256"],
        },
        "verification": {
            "method": "sha256_exact_file_bytes",
            "expected_identity": "retained_huggingface_lfs_sha256",
            "all_indexed_shards_verified": True,
            "inventory_sha256": identity["inventory_sha256"],
        },
        "shards": ordered,
        "summary": {
            "indexed_shard_count": identity["indexed_shard_count"],
            "indexed_shard_bytes": identity["indexed_shard_bytes"],
            "started_unix_ns": started,
            "finished_unix_ns": time.time_ns(),
            "model_loaded": False,
            "gpu_or_provider_started": False,
        },
        "producer": {
            "module": "tools.odyssey.flash_source_shard_ledger",
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file_or_none(Path(__file__).resolve()),
        },
        "claim_boundary": "exact indexed source-shard identity only; not provider semantics, source logits, teacher admission, native parity, capability, EBPW, or TPS",
    })
    atomic_write_json(out, receipt)
    _write_progress(progress_path, identity, records, started_unix_ns=started)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=gate.DEFAULT_MODEL_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    source = _source_inputs(args.root)
    lane = gate._protected_native_lane()
    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_SOURCE_SHARD_LEDGER",
            "source": source["identity"],
            "out": str(args.out.expanduser().resolve()),
            "progress": str((args.progress or _progress_path(args.out)).expanduser().resolve()),
            "protected_native_lane": lane,
            "claim_boundary": "inventory only; zero shard payload bytes hashed",
        }, indent=2))
        return 0
    if not lane["clean"]:
        print(json.dumps({
            "status": "BLOCKED_PROTECTED_NATIVE_LANE",
            "protected_native_lane": lane,
            "reopen_condition": "wait for the protected Hawking native/provider lane to become clean",
        }, indent=2))
        return 2
    receipt = build_source_shard_ledger(
        args.root,
        args.out,
        progress_path=args.progress,
    )
    print(json.dumps({
        "status": receipt["status"],
        "out": str(args.out.expanduser().resolve()),
        "seal_sha256": receipt["seal_sha256"],
        "summary": receipt["summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
