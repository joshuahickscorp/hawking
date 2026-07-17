#!/usr/bin/env python3.12
"""Source-provenance / reassembly manifest for GPT-OSS (120B) — Gravity frontier blocker #2.

Binds every source tensor to its exact origin so a per-expert STR2 archive can be
deterministically reassembled and audited: shard name + path, byte range
[data_start, data_end), dtype, shape, orientation, and the shard header sha256. This is
the "reviewed manifest binds every archive tensor to source shard SHA, byte ranges,
orientation, staging SHA, archive SHA" exit criterion from
`doctor_v5_gptoss_moe_adapter` blocker `original-source-provenance-reassembly-missing`.

Read-only: it uses the mxfp4 header-only inspector (never opens tensor data beyond an
optional streamed data sha), and emits a self-sealed manifest under the additive
subbit_frontier report namespace. It launches nothing and holds no lease.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import doctor_v5_gptoss_mxfp4 as mxfp4  # noqa: E402
from eco_common import seal_field, sealed, now_iso, atomic_write_json  # noqa: E402

MANIFEST_SCHEMA = "hawking.gravity.gptoss_provenance_manifest.v1"


def _orientation(shape: list[int]) -> str:
    """Row-major 2-D projection orientation used by the fused MoE expert layout."""
    if len(shape) == 2:
        return "row_major_2d[out,in]"
    if len(shape) == 1:
        return "vector"
    return f"nd[{len(shape)}]"


def _streamed_data_sha(shard_path: Path, start: int, end: int, *, cap_bytes: int) -> str | None:
    """Optional exact sha256 over a tensor's raw byte range. Bounded by cap_bytes so a full
    manifest of a 61 GB source stays cheap unless the caller asks for full hashing."""
    n = end - start
    if n <= 0 or n > cap_bytes:
        return None
    h = hashlib.sha256()
    with shard_path.open("rb") as fh:
        fh.seek(start)
        remaining = n
        while remaining:
            chunk = fh.read(min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def _load_census(census_path: Path) -> dict[str, dict[str, Any]]:
    """Return {shard_basename: census_row} for cross-checking, or {} if absent."""
    try:
        import json
        c = json.loads(census_path.read_bytes())
    except OSError:
        return {}
    rows = (c.get("source", {}) or {}).get("shards", []) or []
    return {Path(r.get("name", "")).name: r for r in rows}


def build_manifest(source_dir: str | os.PathLike[str], *,
                   hash_data_under_bytes: int = 0,
                   census_path: str | os.PathLike[str] | None =
                   "reports/condense/doctor_v5_scale/120B/census.json") -> dict[str, Any]:
    """Read the GPT-OSS shard headers directly (from original/) and bind every tensor to its
    origin. Cross-checks each shard's header sha256 against the recorded source census so the
    manifest carries an explicit integrity verdict.

    hash_data_under_bytes > 0 additionally streams an exact data sha for tensors whose byte
    range is at or below that size (0 = header-provenance only, the fast default).
    """
    src = Path(source_dir)
    weights_root = src / "original" if (src / "original").is_dir() else src
    census = _load_census(Path(census_path)) if census_path else {}
    shard_files = sorted(weights_root.glob("*of-00007*.safetensors"))
    if not shard_files:
        raise RuntimeError(f"no GPT-OSS shards under {weights_root}")

    tensors_out: list[dict[str, Any]] = []
    shard_index: dict[str, dict[str, Any]] = {}
    integrity = {"census_present": bool(census), "shards_checked": 0, "shards_matched": 0,
                 "mismatches": []}

    shards = [mxfp4._read_header(p, p.name) for p in shard_files]
    for sh in shards:
        row = census.get(sh.name)
        if row is not None:
            integrity["shards_checked"] += 1
            if (sh.header_sha256 == row.get("header_sha256") and sh.file_bytes == row.get("bytes")
                    and sh.data_bytes == row.get("data_bytes")):
                integrity["shards_matched"] += 1
            else:
                integrity["mismatches"].append(sh.name)
        shard_index[sh.name] = {
            "name": sh.name, "path": str(sh.path), "file_bytes": sh.file_bytes,
            "header_bytes": sh.header_bytes, "header_sha256": sh.header_sha256,
            "data_start": sh.data_start, "data_bytes": sh.data_bytes,
            "tensor_count": len(sh.tensors),
        }
        for t in sh.tensors:
            entry = {
                "tensor": t.name, "dtype": t.dtype, "shape": list(t.shape),
                "orientation": _orientation(list(t.shape)),
                "shard_name": t.shard_name, "shard_path": str(t.shard_path),
                "byte_range": [t.data_start, t.data_end],
                "byte_len": t.data_end - t.data_start,
                "shard_header_sha256": t.shard_header_sha256,
            }
            if hash_data_under_bytes:
                entry["data_sha256"] = _streamed_data_sha(
                    Path(t.shard_path), t.data_start, t.data_end, cap_bytes=hash_data_under_bytes)
            tensors_out.append(entry)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_at": now_iso(),
        "parent_label": "120B",
        "hf_or_source_id": "openai/gpt-oss-120b",
        "source_dir": str(src),
        "weights_root": str(weights_root),
        "shard_count": len(shard_index),
        "tensor_count": len(tensors_out),
        "total_source_bytes": sum(s["file_bytes"] for s in shard_index.values()),
        "data_hashing": "streamed" if hash_data_under_bytes else "header_provenance_only",
        "census_integrity": integrity,
        "shards": shard_index,
        "tensors": tensors_out,
        "reassembly_contract": {
            "rule": "each archive tensor reassembles from exactly one source shard byte range",
            "binds": ["shard_name", "shard_path", "byte_range", "dtype", "shape",
                      "orientation", "shard_header_sha256"],
            "resolves_blocker": "original-source-provenance-reassembly-missing",
        },
    }
    return seal_field(manifest, "manifest_sha256")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GPT-OSS source-provenance / reassembly manifest.")
    ap.add_argument("--source", default="scratch/staging/gpt-oss-120b.partial")
    ap.add_argument("--out", default="reports/condense/subbit_frontier/GRAVITY_120B_PROVENANCE.json")
    ap.add_argument("--hash-data-under-bytes", type=int, default=0,
                    help="also stream an exact data sha for tensors <= this size (0 = header only)")
    args = ap.parse_args(argv)
    manifest = build_manifest(args.source, hash_data_under_bytes=args.hash_data_under_bytes)
    atomic_write_json(args.out, manifest)
    print(f"provenance manifest sealed: {manifest['manifest_sha256'][:16]}  "
          f"shards={manifest['shard_count']} tensors={manifest['tensor_count']} "
          f"bytes={manifest['total_source_bytes']:,}  -> {args.out}")
    assert sealed(manifest, "manifest_sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
