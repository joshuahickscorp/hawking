#!/usr/bin/env python3
"""Merge disjoint GLM teacher-forced sequence shards into one coherent capture.

Each worker must have captured a disjoint [seq_start, seq_end) into its own
--output-dir (never shared files). This script:

  * concatenates layers/*.npz arrays on the batch axis in global seq order
  * merges carry/*.npz (last-layer carry) similarly
  * copies paired_traces in order
  * seals GLM_TEACHER_FORCED_CAPTURE_MERGED_RECEIPT.json with per-worker provenance

Correctness bar: bit-exact array_sha256 / npz bytes vs a single serial run on
the same sequence set. Safe only when workers did NOT independently re-stream
the same donor weights (see --allow-weight-stream-amplification refuse path).

Usage:
  python3 tools/condense/merge_glm_teacher_forced_shards.py \\
    --out /path/to/merged \\
    --worker /path/to/w0 \\
    --worker /path/to/w1
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

RECEIPT_CANDIDATES = (
    "GLM_TEACHER_FORCED_CAPTURE_RECEIPT.json",
    "FRANKENSTEIN_TEACHER_FORCED_CAPTURE_RECEIPT.json",
)
MERGED_RECEIPT_NAME = "GLM_TEACHER_FORCED_CAPTURE_MERGED_RECEIPT.json"
MERGED_SCHEMA = "hawking.frankenstein.glm_teacher_forced_capture_merged.v1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_worker(path: Path) -> dict[str, Any]:
    path = Path(path)
    receipt = None
    receipt_path = None
    for name in RECEIPT_CANDIDATES:
        candidate = path / name
        if candidate.is_file():
            receipt = json.loads(candidate.read_text(encoding="utf-8"))
            receipt_path = candidate
            break
    if receipt is None:
        raise FileNotFoundError(
            f"missing capture receipt under {path} "
            f"(tried {', '.join(RECEIPT_CANDIDATES)})"
        )
    shard = receipt.get("shard") or (receipt.get("corpus") or {}).get("shard") or {}
    seq_start = int(shard.get("seq_start", 0))
    seq_end = shard.get("seq_end")
    if seq_end is None:
        seq_end = int((receipt.get("corpus") or {}).get("n_sequences", 0))
    seq_end = int(seq_end)
    worker_id = shard.get("worker_id") or path.name
    example_ids = list(shard.get("example_ids") or [])
    if not example_ids:
        frozen = path / f"FROZEN_CORPUS_{(receipt.get('corpus') or {}).get('level', 'L0')}.json"
        if frozen.is_file():
            doc = json.loads(frozen.read_text(encoding="utf-8"))
            example_ids = [
                s["example_id"] for s in doc.get("sequences", []) if "example_id" in s
            ]
    return {
        "path": path,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "seq_start": seq_start,
        "seq_end": seq_end,
        "worker_id": worker_id,
        "example_ids": example_ids,
        "n_sequences": int((receipt.get("corpus") or {}).get("n_sequences", seq_end - seq_start)),
        "stream_enabled": bool(receipt.get("stream_enabled")),
        "weight_stream_amplification_risk": bool(
            shard.get("weight_stream_amplification_risk")
        ),
    }


def _assert_disjoint(workers: list[dict[str, Any]]) -> None:
    ranges = sorted((w["seq_start"], w["seq_end"], w["worker_id"]) for w in workers)
    for i in range(len(ranges) - 1):
        a0, a1, aid = ranges[i]
        b0, b1, bid = ranges[i + 1]
        if a1 > b0:
            raise ValueError(
                f"overlapping shards: worker {aid} [{a0},{a1}) vs {bid} [{b0},{b1})"
            )
    seen: set[str] = set()
    for w in workers:
        for eid in w["example_ids"]:
            if eid in seen:
                raise ValueError(f"duplicate example_id across workers: {eid}")
            seen.add(eid)


def _concat_npz_axis0(paths: list[Path], out_path: Path) -> dict[str, Any]:
    """Concatenate npz archives along axis 0 for every shared array key."""
    if not paths:
        raise ValueError("no npz paths to concat")
    loaded: list[dict[str, np.ndarray]] = []
    for p in paths:
        with np.load(p) as z:
            loaded.append({k: np.asarray(z[k]) for k in z.files})
    keys = list(loaded[0].keys())
    for part in loaded[1:]:
        if set(part.keys()) != set(keys):
            raise ValueError(
                f"npz key mismatch merging {paths}: "
                f"{sorted(loaded[0].keys())} vs {sorted(part.keys())}"
            )
    merged: dict[str, np.ndarray] = {}
    for k in keys:
        parts = [part[k] for part in loaded]
        # Scalars / 0-d: take first (metrics-like).
        if all(np.asarray(p).ndim == 0 for p in parts):
            merged[k] = np.asarray(parts[0])
            continue
        # Default: concat on axis 0. Matches the serial executor's own
        # microbatch aggregation (including expert_hit_count /
        # expert_contribution_l2 which are per-microbatch histograms stacked
        # along axis 0, not reduced). Sequence shards must be merged in
        # global seq_start order so the stacked microbatch layout matches.
        try:
            merged[k] = np.concatenate([np.asarray(p) for p in parts], axis=0)
        except ValueError as exc:
            raise ValueError(f"cannot concat key {k!r} across shards: {exc}") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, **{k: np.ascontiguousarray(v) for k, v in merged.items()})
    payload = buf.getvalue()
    out_path.write_bytes(payload)
    return {
        "path": str(out_path),
        "npz_sha256": _sha256_bytes(payload),
        "npz_bytes": len(payload),
        "array_sha256": {k: _array_sha256(v) for k, v in sorted(merged.items())},
        "array_names": sorted(merged),
        "n_rows_axis0": {
            k: int(np.asarray(v).shape[0]) if np.asarray(v).ndim >= 1 else 0
            for k, v in merged.items()
        },
    }


def _merge_layer_json_meta(
    workers: list[dict[str, Any]], layer_stem: str
) -> dict[str, Any] | None:
    metas = []
    for w in workers:
        jp = w["path"] / "layers" / f"{layer_stem}.json"
        if jp.is_file():
            metas.append(json.loads(jp.read_text(encoding="utf-8")))
    if not metas:
        return None
    base = dict(metas[0])
    base.pop("seal_sha256", None)
    base["merged_from_workers"] = [w["worker_id"] for w in workers]
    base["note"] = "merged layer meta; trust array_sha256 from merged npz"
    return base


def merge_workers(out: Path, worker_paths: list[Path]) -> dict[str, Any]:
    workers = [_load_worker(p) for p in worker_paths]
    workers.sort(key=lambda w: (w["seq_start"], w["seq_end"], w["worker_id"]))
    _assert_disjoint(workers)

    amplification = any(w["weight_stream_amplification_risk"] for w in workers)
    stream_enabled = any(w["stream_enabled"] for w in workers)

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    layers_out = out / "layers"
    carry_out = out / "carry"
    traces_out = out / "paired_traces"
    layers_out.mkdir()
    carry_out.mkdir()
    traces_out.mkdir()

    # Discover layer stems from first worker (embedding, L00.., final).
    first_layers = workers[0]["path"] / "layers"
    stems = sorted(
        {p.stem for p in first_layers.glob("*.json")}
        | {p.stem for p in first_layers.glob("*.npz")}
    )

    layer_merge: dict[str, Any] = {}
    for stem in stems:
        npz_paths = []
        for w in workers:
            npz = w["path"] / "layers" / f"{stem}.npz"
            if npz.is_file():
                npz_paths.append(npz)
        if npz_paths:
            if len(npz_paths) != len(workers):
                raise FileNotFoundError(
                    f"layer {stem}: expected npz from every worker, got {len(npz_paths)}"
                )
            info = _concat_npz_axis0(npz_paths, layers_out / f"{stem}.npz")
            meta = _merge_layer_json_meta(workers, stem) or {}
            meta.update(
                {
                    "layer_id": stem,
                    "array_sha256": info["array_sha256"],
                    "npz_sha256": info["npz_sha256"],
                    "npz_bytes": info["npz_bytes"],
                    "array_names": info["array_names"],
                }
            )
            (layers_out / f"{stem}.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            layer_merge[stem] = {
                "npz_sha256": info["npz_sha256"],
                "array_sha256": info["array_sha256"],
            }
        else:
            # JSON-only (npz gitignored / stripped): concat nothing; record per-worker hashes.
            metas = []
            for w in workers:
                jp = w["path"] / "layers" / f"{stem}.json"
                if jp.is_file():
                    metas.append(json.loads(jp.read_text(encoding="utf-8")))
            if metas:
                layer_merge[stem] = {
                    "json_only": True,
                    "per_worker_array_sha256": [
                        m.get("array_sha256") or (m.get("meta") or {}).get("array_sha256")
                        for m in metas
                    ],
                }
                (layers_out / f"{stem}.json").write_text(
                    json.dumps(
                        {
                            "layer_id": stem,
                            "merged_from_workers": [w["worker_id"] for w in workers],
                            "json_only": True,
                            "per_worker": metas,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

    # Carry: merge the deepest common after_L* if present.
    carry_stems = sorted(
        {
            p.stem
            for w in workers
            for p in (w["path"] / "carry").glob("after_L*.npz")
        }
    )
    carry_merge: dict[str, Any] = {}
    for stem in carry_stems:
        paths = [w["path"] / "carry" / f"{stem}.npz" for w in workers]
        if all(p.is_file() for p in paths):
            info = _concat_npz_axis0(paths, carry_out / f"{stem}.npz")
            carry_merge[stem] = info

    # Paired traces: copy in global order.
    n_traces = 0
    for w in workers:
        src = w["path"] / "paired_traces"
        if not src.is_dir():
            continue
        for tf in sorted(src.glob("*.json")):
            shutil.copy2(tf, traces_out / tf.name)
            n_traces += 1

    # Frozen corpus: concatenate sequence lists in order.
    level = (workers[0]["receipt"].get("corpus") or {}).get("level", "L0")
    seq_docs = []
    for w in workers:
        fp = w["path"] / f"FROZEN_CORPUS_{level}.json"
        if fp.is_file():
            seq_docs.append(json.loads(fp.read_text(encoding="utf-8")))
    if seq_docs:
        merged_corpus = dict(seq_docs[0])
        merged_corpus.pop("seal_sha256", None)
        sequences = []
        for d in seq_docs:
            sequences.extend(d.get("sequences") or [])
        merged_corpus["sequences"] = sequences
        merged_corpus["n_sequences"] = len(sequences)
        merged_corpus["source"] = "merged_sequence_shards"
        merged_corpus["merged_from_workers"] = [w["worker_id"] for w in workers]
        (out / f"FROZEN_CORPUS_{level}.json").write_text(
            json.dumps(merged_corpus, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    example_ids = []
    for w in workers:
        example_ids.extend(w["example_ids"])

    receipt = {
        "schema": MERGED_SCHEMA,
        "status": "MERGED",
        "n_workers": len(workers),
        "workers": [
            {
                "worker_id": w["worker_id"],
                "path": str(w["path"]),
                "seq_start": w["seq_start"],
                "seq_end": w["seq_end"],
                "n_sequences": w["n_sequences"],
                "receipt_status": w["receipt"].get("status"),
                "receipt_seal_sha256": w["receipt"].get("seal_sha256"),
                "stream_enabled": w["stream_enabled"],
                "weight_stream_amplification_risk": w[
                    "weight_stream_amplification_risk"
                ],
            }
            for w in workers
        ],
        "seq_coverage": {
            "seq_start": workers[0]["seq_start"],
            "seq_end": workers[-1]["seq_end"],
            "n_sequences": sum(w["n_sequences"] for w in workers),
            "example_ids": example_ids,
        },
        "layers": layer_merge,
        "carry": {
            k: {"npz_sha256": v["npz_sha256"], "array_sha256": v["array_sha256"]}
            for k, v in carry_merge.items()
        },
        "paired_traces_n": n_traces,
        "stream_enabled_any_worker": stream_enabled,
        "weight_stream_amplification_risk_any_worker": amplification,
        "note": (
            "Sequence-shard merge for GLM teacher-forced capture. "
            "Does not re-run the model. Bit-exact vs serial when workers used "
            "resident weights on the same frozen corpus indices."
        ),
        "fabricated": False,
    }
    if amplification:
        receipt["warning"] = (
            "One or more workers flagged weight_stream_amplification_risk: "
            "they independently re-streamed donor weights. Merge is still "
            "numerically valid if each worker completed, but wall-clock likely "
            "suffered N× network cost."
        )
    # Seal-like fingerprint without depending on lab.seal for standalone use.
    body = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    receipt["merge_sha256"] = _sha256_bytes(body)
    (out / MERGED_RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--worker",
        type=Path,
        action="append",
        required=True,
        help="Worker output directory (repeatable)",
    )
    args = parser.parse_args(argv)
    try:
        receipt = merge_workers(args.out, list(args.worker))
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
