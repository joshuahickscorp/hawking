#!/usr/bin/env python3.12
"""Validate and assemble activation-aware ``.aap`` shards without copying them.

The packer's output is already the physical artifact.  Assembly adds the
runtime model index, complete official-tensor coverage proof, pinned source
dtypes for pass-through decoding, shard hashes, and tokenizer hardlinks.

    python3.12 tools/condense/glm52_activation_aware_assemble.py check --artifact <dir>
    python3.12 tools/condense/glm52_activation_aware_assemble.py assemble --artifact <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from activation_aware_format import (
    ActivationAwareFormatError,
    read_index,
    sha256_file,
    validate_payload_magics,
)
from glm52_adapter import OFFICIAL_TOKENIZER_ASSETS
from glm52_assemble import (
    MANIFEST,
    REVISION,
    official_weight_map,
    synthesize_architecture,
)


INDEX_NAME = "model.activation_aware.index.json"
MEASUREMENT_NAME = "MEASUREMENT.json"
ALLOCATION_NAME = "ALLOCATION.json"
RECEIPT_NAME = "ACTIVATION_AWARE_ASSEMBLY_RECEIPT.json"
DEFAULT_ARTIFACT = (
    Path.home()
    / "Library/Application Support/Hawking/GLM52Gravity/activation_aware_pack"
)
DEFAULT_TOKENIZER = (
    Path.home()
    / "Library/Application Support/Hawking/Models/GLM-5.2"
    / REVISION
    / "General-R0/tokenizer"
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measurement_dtypes(path: Path) -> dict[str, str]:
    if not Path(path).is_file():
        raise ActivationAwareFormatError(
            f"{path}: missing; pass-through dtype was intentionally recovered "
            "from the packer's measurement record"
        )
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != "hawking.glm52.activation_aware_measurement.v1":
        raise ActivationAwareFormatError(
            f"{path.name}: unexpected schema {doc.get('schema')!r}"
        )
    out: dict[str, str] = {}
    for row in doc.get("measurements", []):
        name, dtype = row.get("name"), row.get("dtype")
        if not name or not dtype or name in out:
            raise ActivationAwareFormatError(
                f"{path.name}: missing/duplicate tensor dtype record for {name!r}"
            )
        out[str(name)] = str(dtype)
    return out


def scan_artifact(artifact: Path) -> dict[str, Any]:
    artifact = Path(artifact)
    shards = sorted(artifact.glob("model-*.aap"))
    if not shards:
        raise ActivationAwareFormatError(f"{artifact}: no model-*.aap shards")
    tensors: dict[str, dict[str, Any]] = {}
    shard_indices: dict[str, dict[str, Any]] = {}
    for path in shards:
        index, body_offset = read_index(path)
        validate_payload_magics(path, index, body_offset)
        shard_indices[path.name] = index
        for row in index["tensors"]:
            name = row["name"]
            if name in tensors:
                raise ActivationAwareFormatError(
                    f"tensor {name!r} appears in both "
                    f"{tensors[name]['shard']} and {path.name}"
                )
            tensors[name] = {
                "shard": path.name,
                "disposition": row.get("disposition"),
                "dtype": row.get("dtype"),
                "bytes": int(row["bytes"]),
                "shape": list(row["shape"]),
            }
    return {
        "paths": shards,
        "shard_indices": shard_indices,
        "tensors": tensors,
    }


def check(
    artifact: Path,
    *,
    expected_weight_map: dict[str, str] | None = None,
    measurement_path: Path | None = None,
) -> dict[str, Any]:
    artifact = Path(artifact)
    scan = scan_artifact(artifact)
    expected_weight_map = expected_weight_map or official_weight_map()
    dtypes = measurement_dtypes(
        measurement_path or artifact / MEASUREMENT_NAME
    )
    packed = scan["tensors"]

    missing: list[str] = []
    missing_count = 0
    misplaced: list[dict[str, str]] = []
    bad_disposition: list[dict[str, str]] = []
    dtype_missing: list[str] = []
    counts = {"ACTIVATION_AWARE_PAYLOAD": 0, "PASS_THROUGH_PAYLOAD": 0}
    for name, source_shard in expected_weight_map.items():
        row = packed.get(name)
        # A name in the packer's index with zero billed bytes is a descriptor, not a
        # payload — same rule as gravity assemble. Count it as missing for completeness
        # even when the key is present; the sample list and missing_count must agree.
        if row is None or int(row["bytes"]) <= 0:
            missing_count += 1
            if len(missing) < 20:
                missing.append(name)
            continue
        expected_shard = source_shard.replace(".safetensors", ".aap")
        if row["shard"] != expected_shard and len(misplaced) < 20:
            misplaced.append(
                {"tensor": name, "expected": expected_shard, "found": row["shard"]}
            )
        disposition = row["disposition"]
        if disposition == "activation_aware":
            counts["ACTIVATION_AWARE_PAYLOAD"] += 1
        elif disposition == "pass_through":
            counts["PASS_THROUGH_PAYLOAD"] += 1
        elif len(bad_disposition) < 20:
            bad_disposition.append(
                {"tensor": name, "disposition": str(disposition)}
            )
        if name not in dtypes and len(dtype_missing) < 20:
            dtype_missing.append(name)
        elif row.get("dtype") is not None and row["dtype"] != dtypes[name]:
            if len(dtype_missing) < 20:
                dtype_missing.append(
                    f"{name}: shard={row['dtype']} measurement={dtypes[name]}"
                )

    undeclared = sorted(set(packed) - set(expected_weight_map))
    expected_shards = {
        filename.replace(".safetensors", ".aap")
        for filename in expected_weight_map.values()
    }
    present_shards = {path.name for path in scan["paths"]}
    complete = (
        missing_count == 0
        and not misplaced
        and not bad_disposition
        and not undeclared
        and not dtype_missing
        and present_shards >= expected_shards
    )
    return {
        "schema": "hawking.glm52.activation_aware_assembly_coverage.v1",
        "at": _now(),
        "artifact": str(artifact),
        "revision": REVISION,
        "official_manifest": str(MANIFEST),
        "official_tensors": len(expected_weight_map),
        "packed_tensors": len(packed),
        "official_shards": len(expected_shards),
        "packed_shards": len(present_shards),
        "dispositions": counts,
        "missing_count": missing_count,
        "missing_sample": missing,
        "misplaced_sample": misplaced,
        "undeclared_count": len(undeclared),
        "undeclared_sample": undeclared[:20],
        "bad_disposition_sample": bad_disposition,
        "dtype_missing_sample": dtype_missing,
        "shards_outstanding_count": len(expected_shards - present_shards),
        "shards_outstanding_sample": sorted(expected_shards - present_shards)[:20],
        "complete": complete,
        "verdict": "COMPLETE" if complete else "INCOMPLETE",
    }


def _install_tokenizer(source: Path, artifact: Path) -> dict[str, Any]:
    source = Path(source)
    destination = Path(artifact) / "tokenizer"
    destination.mkdir(parents=True, exist_ok=True)
    installed: dict[str, dict[str, Any]] = {}
    for name, (expected_bytes, expected_hash) in OFFICIAL_TOKENIZER_ASSETS.items():
        src = source / name
        if not src.is_file():
            if name == "generation_config.json":
                # The assembled runtime needs the tokenizer/template triplet;
                # generation stop policy lives in the architecture contract.
                continue
            raise ActivationAwareFormatError(
                f"pinned tokenizer asset missing: {src}"
            )
        observed_hash = sha256_file(src)
        if src.stat().st_size != expected_bytes or observed_hash != expected_hash:
            raise ActivationAwareFormatError(
                f"pinned tokenizer identity mismatch for {src}"
            )
        dst = destination / name
        if dst.exists():
            if sha256_file(dst) != expected_hash:
                raise ActivationAwareFormatError(
                    f"existing tokenizer asset has wrong identity: {dst}"
                )
            method = "already_present"
        else:
            try:
                os.link(src, dst)
                method = "hardlink"
            except OSError:
                shutil.copy2(src, dst)
                method = "copy"
        installed[name] = {
            "sha256": expected_hash,
            "bytes": expected_bytes,
            "method": method,
        }
    return installed


def assemble(
    artifact: Path,
    *,
    tokenizer_source: Path = DEFAULT_TOKENIZER,
    expected_weight_map: dict[str, str] | None = None,
    measurement_path: Path | None = None,
) -> dict[str, Any]:
    artifact = Path(artifact)
    measurement_path = measurement_path or artifact / MEASUREMENT_NAME
    coverage = check(
        artifact,
        expected_weight_map=expected_weight_map,
        measurement_path=measurement_path,
    )
    if not coverage["complete"]:
        return {
            "action": "REFUSED",
            "reason": "activation-aware coverage is incomplete",
            "coverage": coverage,
        }

    scan = scan_artifact(artifact)
    dtypes = measurement_dtypes(measurement_path)
    shard_hashes = {
        path.name: sha256_file(path) for path in scan["paths"]
    }
    tokenizer = _install_tokenizer(tokenizer_source, artifact)
    weight_map = {
        name: row["shard"] for name, row in sorted(scan["tensors"].items())
    }
    index = {
        "schema": "hawking.activation_aware.model_index.v1",
        "assembled_at": _now(),
        "model": {
            "repo": "zai-org/GLM-5.2",
            "revision": REVISION,
            "representation": "ACTIVATION_AWARE_LOW_RANK",
        },
        "architecture": synthesize_architecture(),
        "shards": sorted(shard_hashes),
        "shard_count": len(shard_hashes),
        "shard_sha256": shard_hashes,
        "tensor_count": len(weight_map),
        "weight_map": weight_map,
        "tensor_dtypes": {name: dtypes[name] for name in sorted(weight_map)},
        "coverage": coverage,
        "source_receipts": {
            MEASUREMENT_NAME: _json_sha256(measurement_path),
            ALLOCATION_NAME: (
                _json_sha256(artifact / ALLOCATION_NAME)
                if (artifact / ALLOCATION_NAME).is_file()
                else None
            ),
        },
        "tokenizer": tokenizer,
        "byte_provenance": (
            "model-*.aap are the packer's original bytes; assembly created no "
            "second model copy"
        ),
    }
    encoded = json.dumps(index, indent=1, sort_keys=True) + "\n"
    index_path = artifact / INDEX_NAME
    tmp = index_path.with_suffix(index_path.suffix + ".partial")
    tmp.write_text(encoded)
    os.replace(tmp, index_path)
    index_hash = _json_sha256(index_path)
    receipt = {
        "schema": "hawking.glm52.activation_aware_assembly_receipt.v1",
        "at": _now(),
        "action": "ASSEMBLED",
        "artifact": str(artifact),
        "index": str(index_path),
        "index_sha256": index_hash,
        "shards_hashed": len(shard_hashes),
        "tensors": len(weight_map),
        "coverage": "COMPLETE",
        "model_bytes_copied": 0,
        "tokenizer": tokenizer,
    }
    receipt_path = artifact / RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "assemble"))
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--measurement", type=Path)
    parser.add_argument("--tokenizer-source", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()
    try:
        if args.command == "check":
            result = check(args.artifact, measurement_path=args.measurement)
        else:
            result = assemble(
                args.artifact,
                tokenizer_source=args.tokenizer_source,
                measurement_path=args.measurement,
            )
    except (ActivationAwareFormatError, OSError, ValueError) as exc:
        print(json.dumps({"action": "ERROR", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    if args.command == "check":
        return 0 if result["complete"] else 1
    return 0 if result.get("action") == "ASSEMBLED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
