"""Binary expert-activation pack for Q30 capture runs.

The historical capture stores a multi-GB JSON route table plus one small
f32le file per retained (layer, probe, position).  Loading that into the
repack's per-(layer, expert) matrices is dominated by JSON parse + hundreds
of thousands of tiny file opens — tens of seconds to minutes.

This pack is the repack's native load format:

  expert-pack.v1/
    header.json     small index (schema, keys, offsets, content hashes)
    rows.f32le      unique retained router-input rows, row-major f32le
    index.u32le     concatenated uint32 row indices into rows.f32le

`collect` rebuilds the exact stacked matrices the JSON path would produce
(same row order, same content).  Existing captures stay readable via the
JSON fallback; convert once with `convert_capture_to_expert_pack`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA = "hawking.capture.expert_pack.v1"
PACK_DIRNAME = "expert-pack.v1"
HEADER_NAME = "header.json"
ROWS_NAME = "rows.f32le"
INDEX_NAME = "index.u32le"
DEFAULT_WIDTH = 2048


class ExpertPackError(RuntimeError):
    """Expert-pack is missing, corrupt, or not byte-identical to the JSON path."""


def pack_dir(run_dir: Path) -> Path:
    return Path(run_dir) / PACK_DIRNAME


def pack_is_present(run_dir: Path) -> bool:
    root = pack_dir(run_dir)
    return (
        (root / HEADER_NAME).is_file()
        and (root / ROWS_NAME).is_file()
        and (root / INDEX_NAME).is_file()
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _content_hash_stacked(stacked: Mapping[tuple[int, int], np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in sorted(stacked):
        h.update(np.asarray(key, dtype=np.int32).tobytes())
        arr = np.ascontiguousarray(stacked[key], dtype=np.float32)
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        h.update(arr.tobytes())
    return h.hexdigest()


def build_from_capture_walk(
    run_dir: Path,
    capture: Mapping[str, Any],
    *,
    all_layer: bool,
    default_layer: int = 0,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any], dict[str, Any]]:
    """Walk the JSON capture the same way as the historical collector.

    Returns (stacked, provenance_bits, pack_build_meta).
    """

    # Unique rows keyed by relative_path (or synthetic for L0) so multi-expert
    # hits share one physical row without re-reading the file.
    row_ids: dict[str, int] = {}
    rows: list[np.ndarray] = []
    by_key_ids: dict[tuple[int, int], list[int]] = {}
    total_steps = 0
    hidden_steps = 0
    route_only_steps = 0
    layers_seen: set[int] = set()
    unique_files = 0

    def load_row(rel: str, elements: int) -> int:
        nonlocal unique_files
        existing = row_ids.get(rel)
        if existing is not None:
            return existing
        path = run_dir / rel
        x = np.fromfile(path, dtype="<f4")
        if x.size != int(elements):
            raise ExpertPackError(f"hidden size mismatch at {path}: {x.size} != {elements}")
        rid = len(rows)
        rows.append(np.ascontiguousarray(x, dtype=np.float32))
        row_ids[rel] = rid
        unique_files += 1
        return rid

    for probe in capture["probes"]:
        for step in probe["steps"]:
            total_steps += 1
            if all_layer:
                layer_rows = step.get("layers") or []
                if not layer_rows:
                    raise ExpertPackError(
                        f"all-layer capture step missing layers at position {step.get('position')}"
                    )
                any_hidden = False
                for layer_row in layer_rows:
                    layer = int(layer_row["layer"])
                    layers_seen.add(layer)
                    hidden_meta = layer_row.get("router_input_hidden_f32le")
                    if not hidden_meta:
                        continue
                    any_hidden = True
                    rid = load_row(hidden_meta["relative_path"], int(hidden_meta["elements"]))
                    for expert_id in layer_row["selected_expert_ids"]:
                        by_key_ids.setdefault((layer, int(expert_id)), []).append(rid)
                if any_hidden:
                    hidden_steps += 1
                else:
                    route_only_steps += 1
            else:
                layers_seen.add(default_layer)
                hidden_meta = step.get("router_input_hidden_f32le")
                if not hidden_meta:
                    raise ExpertPackError("L0 capture step missing router_input_hidden_f32le")
                rid = load_row(hidden_meta["relative_path"], int(hidden_meta["elements"]))
                hidden_steps += 1
                for expert_id in step["selected_expert_ids"]:
                    by_key_ids.setdefault((default_layer, int(expert_id)), []).append(rid)

    if not rows:
        raise ExpertPackError("capture produced no retained hidden rows")

    width = int(rows[0].size)
    rows_mat = np.stack(rows, axis=0)
    if rows_mat.dtype != np.float32:
        rows_mat = rows_mat.astype(np.float32, copy=False)
    if not rows_mat.flags["C_CONTIGUOUS"]:
        rows_mat = np.ascontiguousarray(rows_mat)

    stacked: dict[tuple[int, int], np.ndarray] = {}
    for key, ids in by_key_ids.items():
        stacked[key] = rows_mat[np.asarray(ids, dtype=np.int64)]

    provenance = {
        "total_steps": total_steps,
        "hidden_retained_steps": hidden_steps,
        "route_only_steps": route_only_steps,
        "token_expert_pairs": int(sum(v.shape[0] for v in stacked.values())),
        "layer_expert_pairs_with_hits": len(stacked),
        "experts_with_hits": len({e for (_, e) in stacked}),
        "layers_with_hidden_hits": sorted(layers_seen),
        "n_layers_with_hidden_hits": len(layers_seen),
        "all_layer_capture": all_layer,
        "capture_schema": capture.get("schema"),
        "bounded_storage": capture.get("bounded_storage"),
    }
    meta = {
        "n_unique_rows": int(rows_mat.shape[0]),
        "hidden_width": width,
        "unique_files_read": unique_files,
        "content_hash": _content_hash_stacked(stacked),
        "by_key_ids": by_key_ids,
        "rows_mat": rows_mat,
    }
    return stacked, provenance, meta


def write_expert_pack(
    run_dir: Path,
    *,
    rows_mat: np.ndarray,
    by_key_ids: Mapping[tuple[int, int], list[int]],
    content_hash: str,
    capture_schema: str | None = None,
    extra_header: Mapping[str, Any] | None = None,
) -> Path:
    """Write expert-pack.v1 under run_dir. Returns the pack directory."""

    run_dir = Path(run_dir)
    root = pack_dir(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows_path = root / ROWS_NAME
    index_path = root / INDEX_NAME
    header_path = root / HEADER_NAME

    rows = np.ascontiguousarray(rows_mat, dtype=np.float32)
    if rows.ndim != 2:
        raise ExpertPackError(f"rows_mat must be 2-D, got shape {rows.shape}")
    n_rows, width = int(rows.shape[0]), int(rows.shape[1])

    # Deterministic key order: layer, expert.
    ordered_keys = sorted(by_key_ids.keys())
    index_chunks: list[np.ndarray] = []
    key_entries: list[dict[str, Any]] = []
    cursor = 0
    for layer, expert in ordered_keys:
        ids = np.asarray(by_key_ids[(layer, expert)], dtype=np.uint32)
        if ids.size and int(ids.max()) >= n_rows:
            raise ExpertPackError(
                f"index out of range for L{layer}.E{expert}: max={ids.max()} n_rows={n_rows}"
            )
        # Per-key content hash of the stacked view (order-sensitive).
        stacked = rows[ids.astype(np.int64)]
        key_hash = hashlib.sha256(np.ascontiguousarray(stacked, dtype=np.float32).tobytes()).hexdigest()
        key_entries.append(
            {
                "layer": int(layer),
                "expert": int(expert),
                "n_rows": int(ids.size),
                "index_offset": int(cursor),
                "stacked_sha256": key_hash,
            }
        )
        index_chunks.append(ids)
        cursor += int(ids.size)

    if index_chunks:
        index = np.concatenate(index_chunks).astype(np.uint32, copy=False)
    else:
        index = np.zeros((0,), dtype=np.uint32)

    # Atomic-ish write: temp then replace.
    tmp_rows = root / f".{ROWS_NAME}.tmp"
    tmp_index = root / f".{INDEX_NAME}.tmp"
    tmp_header = root / f".{HEADER_NAME}.tmp"
    try:
        rows.astype("<f4", copy=False).tofile(tmp_rows)
        index.astype("<u4", copy=False).tofile(tmp_index)
        rows_sha = _sha256_file(tmp_rows)
        index_sha = _sha256_file(tmp_index)
        header: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "EARNED_EXPERT_PACK_V1",
            "hidden_width": width,
            "n_unique_rows": n_rows,
            "n_index_entries": int(index.size),
            "n_layer_expert_keys": len(key_entries),
            "rows_relative_path": ROWS_NAME,
            "index_relative_path": INDEX_NAME,
            "rows_sha256": rows_sha,
            "index_sha256": index_sha,
            "stacked_content_sha256": content_hash,
            "capture_schema": capture_schema,
            "dtype": "float32_le",
            "index_dtype": "uint32_le",
            "keys": key_entries,
            "claim_boundary": {
                "pack_is_a_lossless_reindex_of_retained_router_input_hiddens": True,
                "row_order_matches_json_collect_expert_activations": True,
                "json_capture_result_remains_the_binding_identity_for_sha256": True,
            },
        }
        if extra_header:
            header.update(dict(extra_header))
        tmp_header.write_text(
            json.dumps(header, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp_rows.replace(rows_path)
        tmp_index.replace(index_path)
        tmp_header.replace(header_path)
    finally:
        for p in (tmp_rows, tmp_index, tmp_header):
            if p.exists():
                p.unlink()
    return root


def convert_capture_to_expert_pack(
    run_dir: Path,
    capture: Mapping[str, Any] | None = None,
    *,
    all_layer: bool | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build expert-pack.v1 from an existing JSON capture run."""

    run_dir = Path(run_dir).expanduser().resolve()
    if pack_is_present(run_dir) and not force:
        header = json.loads((pack_dir(run_dir) / HEADER_NAME).read_text(encoding="utf-8"))
        return {
            "status": "ALREADY_PRESENT",
            "pack_dir": str(pack_dir(run_dir)),
            "stacked_content_sha256": header.get("stacked_content_sha256"),
            "n_unique_rows": header.get("n_unique_rows"),
            "n_layer_expert_keys": header.get("n_layer_expert_keys"),
        }

    started = time.perf_counter()
    if capture is None:
        result_path = run_dir / "capture-result.json"
        if not result_path.is_file():
            raise ExpertPackError(f"missing capture-result.json under {run_dir}")
        capture = json.loads(result_path.read_bytes())
    if all_layer is None:
        schema = str(capture.get("schema") or "")
        all_layer = (
            schema == "hawking.ascension.qwen30_broad_activation_all_layer_route_capture_result.v1"
            or bool(capture.get("capture_summary", {}).get("all_layer_activation_capture"))
            or bool(
                capture.get("probes")
                and isinstance(capture["probes"][0], Mapping)
                and (capture["probes"][0].get("steps") or [{}])[0].get("layers") is not None
            )
        )

    stacked, prov, meta = build_from_capture_walk(
        run_dir, capture, all_layer=bool(all_layer)
    )
    root = write_expert_pack(
        run_dir,
        rows_mat=meta["rows_mat"],
        by_key_ids=meta["by_key_ids"],
        content_hash=meta["content_hash"],
        capture_schema=str(capture.get("schema") or ""),
        extra_header={
            "source": "converted_from_json_capture_result",
            "conversion_wall_secs": time.perf_counter() - started,
            "total_steps": prov.get("total_steps"),
            "hidden_retained_steps": prov.get("hidden_retained_steps"),
            "route_only_steps": prov.get("route_only_steps"),
            "all_layer_capture": prov.get("all_layer_capture"),
            "layers_with_hidden_hits": prov.get("layers_with_hidden_hits"),
            "bounded_storage": prov.get("bounded_storage"),
        },
    )
    return {
        "status": "WRITTEN",
        "pack_dir": str(root),
        "stacked_content_sha256": meta["content_hash"],
        "n_unique_rows": meta["n_unique_rows"],
        "n_layer_expert_keys": len(stacked),
        "token_expert_pairs": int(sum(v.shape[0] for v in stacked.values())),
        "wall_secs": time.perf_counter() - started,
        "unique_files_read": meta["unique_files_read"],
    }


def load_expert_pack(
    run_dir: Path,
    *,
    verify_key_hashes: bool = False,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    """Load stacked activations from expert-pack.v1.

    `verify_key_hashes` re-hashes every stacked matrix against the header.  That
    is correct but scans ~10 GB of float data and is not the warm-path default;
    the convert step already checked identity against the JSON path.
    """

    run_dir = Path(run_dir)
    root = pack_dir(run_dir)
    header_path = root / HEADER_NAME
    if not header_path.is_file():
        raise ExpertPackError(f"expert pack header missing: {header_path}")
    header = json.loads(header_path.read_text(encoding="utf-8"))
    if header.get("schema") != SCHEMA:
        raise ExpertPackError(f"unexpected expert pack schema: {header.get('schema')}")

    width = int(header["hidden_width"])
    n_rows = int(header["n_unique_rows"])
    rows_path = root / str(header.get("rows_relative_path") or ROWS_NAME)
    index_path = root / str(header.get("index_relative_path") or INDEX_NAME)

    # mmap unique rows — gather still copies, but avoids a full second resident buffer.
    rows = np.memmap(rows_path, dtype="<f4", mode="r", shape=(n_rows, width))
    index = np.fromfile(index_path, dtype="<u4")
    if index.size != int(header["n_index_entries"]):
        raise ExpertPackError(
            f"index size mismatch: file has {index.size}, header {header['n_index_entries']}"
        )

    stacked: dict[tuple[int, int], np.ndarray] = {}
    for entry in header["keys"]:
        layer = int(entry["layer"])
        expert = int(entry["expert"])
        n = int(entry["n_rows"])
        off = int(entry["index_offset"])
        ids = index[off : off + n].astype(np.int64, copy=False)
        # Contiguous copy so later BLAS fits own their buffers (mmap views would pin).
        mat = np.ascontiguousarray(rows[ids], dtype=np.float32)
        if verify_key_hashes:
            expected = entry.get("stacked_sha256")
            if expected:
                got = hashlib.sha256(mat.tobytes()).hexdigest()
                if got != expected:
                    raise ExpertPackError(
                        f"stacked sha256 mismatch for L{layer}.E{expert}: {got} != {expected}"
                    )
        stacked[(layer, expert)] = mat

    hit_counts = {
        f"L{layer}.E{expert}": int(arr.shape[0])
        for (layer, expert), arr in sorted(
            stacked.items(), key=lambda kv: (-kv[1].shape[0], kv[0][0], kv[0][1])
        )
    }
    provenance = {
        "total_steps": header.get("total_steps"),
        "hidden_retained_steps": header.get("hidden_retained_steps"),
        "route_only_steps": header.get("route_only_steps"),
        "token_expert_pairs": int(sum(v.shape[0] for v in stacked.values())),
        "layer_expert_pairs_with_hits": len(stacked),
        "experts_with_hits": len({e for (_, e) in stacked}),
        "layers_with_hidden_hits": sorted({k[0] for k in stacked}),
        "n_layers_with_hidden_hits": len({k[0] for k in stacked}),
        "all_layer_capture": bool(header.get("all_layer_capture", True)),
        "hit_counts": hit_counts,
        "capture_schema": header.get("capture_schema"),
        "bounded_storage": header.get("bounded_storage"),
        "expert_pack": {
            "schema": SCHEMA,
            "pack_dir": str(root),
            "n_unique_rows": n_rows,
            "hidden_width": width,
            "stacked_content_sha256": header.get("stacked_content_sha256"),
            "load_path": "binary_expert_pack_v1",
        },
    }
    # Prefer provenance fields recorded at write time when present.
    for k in (
        "total_steps",
        "hidden_retained_steps",
        "route_only_steps",
        "all_layer_capture",
        "bounded_storage",
    ):
        if k in header and header[k] is not None:
            provenance[k] = header[k]
    if "layers_with_hidden_hits" in header and header["layers_with_hidden_hits"] is not None:
        provenance["layers_with_hidden_hits"] = list(header["layers_with_hidden_hits"])
        provenance["n_layers_with_hidden_hits"] = len(provenance["layers_with_hidden_hits"])
    return stacked, provenance


def write_pack_from_stacked(
    run_dir: Path,
    stacked: Mapping[tuple[int, int], np.ndarray],
    *,
    provenance: Mapping[str, Any] | None = None,
    capture_schema: str | None = None,
) -> Path:
    """Write a pack by de-duplicating stacked rows via content (slower; for tests)."""

    # Identity map: keep each stacked row as its own unique row (no de-dup).
    # Order of keys and rows is preserved for identity with the producer.
    rows_list: list[np.ndarray] = []
    by_key_ids: dict[tuple[int, int], list[int]] = {}
    for key in sorted(stacked.keys()):
        arr = np.ascontiguousarray(stacked[key], dtype=np.float32)
        ids: list[int] = []
        for i in range(arr.shape[0]):
            ids.append(len(rows_list))
            rows_list.append(arr[i])
        by_key_ids[key] = ids
    rows_mat = np.stack(rows_list, axis=0) if rows_list else np.zeros((0, DEFAULT_WIDTH), np.float32)
    content_hash = _content_hash_stacked(stacked)
    extra = dict(provenance or {})
    return write_expert_pack(
        run_dir,
        rows_mat=rows_mat,
        by_key_ids=by_key_ids,
        content_hash=content_hash,
        capture_schema=capture_schema,
        extra_header=extra,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-run", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-json", action="store_true",
                        help="After convert, re-walk JSON and assert content hash match.")
    args = parser.parse_args(argv)
    result = convert_capture_to_expert_pack(args.capture_run, force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.verify_json:
        from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (  # noqa: WPS433
            capture_is_all_layer,
            collect_expert_activations_from_json,
        )

        run = args.capture_run.expanduser().resolve()
        capture = json.loads((run / "capture-result.json").read_bytes())
        t0 = time.perf_counter()
        stacked_json, _ = collect_expert_activations_from_json(run, capture)
        t_json = time.perf_counter() - t0
        t0 = time.perf_counter()
        stacked_bin, _ = load_expert_pack(run)
        t_bin = time.perf_counter() - t0
        h_json = _content_hash_stacked(stacked_json)
        h_bin = _content_hash_stacked(stacked_bin)
        print(
            json.dumps(
                {
                    "json_load_secs": t_json,
                    "binary_load_secs": t_bin,
                    "json_content_hash": h_json,
                    "binary_content_hash": h_bin,
                    "identical": h_json == h_bin,
                },
                indent=2,
            )
        )
        if h_json != h_bin:
            raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
