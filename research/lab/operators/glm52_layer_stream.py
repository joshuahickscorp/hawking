#!/usr/bin/env python3.12
"""Direct layer-major HF streaming for official GLM-5.2 teacher-forced capture.

Ceremony-free: downloads via ``hf_hub_download`` (same transport pattern as
``workspace/campaign/records/runs/frankenstein/full_glm_math_mapping.py``).
Does NOT route through GLM52_STREAMING_SCHEDULE restream or the mapping's
glm-donor working set.

Contract:
  - control_root holds config.json + model.safetensors.index.json (+ optional
    tokenizer assets). Never rewritten by this module.
  - stream_root holds at most the current + prefetched layer shards.
  - Per-shard LFS sha256 verified against the sealed official manifest before
    admission. Fail closed on mismatch.
  - 25 GiB free-space floor + source-only reclaim (unlink admitted shards only).
  - Double-buffer: prefetch N+1 while N executes.
  - Multi-shard downloads run concurrently (default 4 workers). Measured GLM
    official capture was download-bound at ~95–110 MiB/s/shard with only 2
    workers and sequential ensure(); Kimi (96 shards, 1.56 TB) needs the
    concurrency. The frozen public-path winner (194 MiB/s direct_presigned
    range) is a different transport; this module stays on hf_hub_download but
    applies the compatible HF_XET knobs and multi-file fan-out.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lab.operators.glm52_adapter import (
    IMMUTABLE_REVISION,
    PROFILE_OFFICIAL,
    REPO_ID,
    Inventory,
    load_index,
    load_json_strict,
    validate_config,
    verify_streaming_window,
)
from lab.operators.glm52_common import Glm52Error, sha256_file, utc_now
from lab.layout import EVIDENCE_ROOT


MIN_FREE_FLOOR_BYTES = 25 * 1024**3
# GLM layers average ~4.5 shards; 4 concurrent downloads saturates the ~200
# MiB/s aggregate observed when two 100 MiB/s streams ran in parallel without
# blowing the 2-layer double-buffer working set.
DEFAULT_PREFETCH_WORKERS = int(os.environ.get("FRANK_PREFETCH_WORKERS", "4"))
DEFAULT_STREAM_ROOT = Path(
    "/Users/scammermike/Library/Application Support/hawking/"
    "GLM52Gravity/stream_scratch"
)
DEFAULT_CONTROL_ROOT = Path(
    "/Users/scammermike/Library/Application Support/hawking/"
    "GLM52Gravity/stream_control/"
    + IMMUTABLE_REVISION
)
OFFICIAL_MANIFEST = (
    EVIDENCE_ROOT / "models" / "glm52" / "GLM52_OFFICIAL_MANIFEST.json"
)
# Compatible subset of the frozen public-path profile that applies to
# official_hf_xet / hf_hub_download (not the custom direct_presigned transport).
# Must be set before hf_xet is imported for full effect; setdefault so callers
# can override.
_PUBLIC_PATH_COMPAT_ENV: dict[str, str] = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
    "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY": "false",
    "HF_XET_RECONSTRUCTION_USE_VECTORED_WRITE": "true",
    # High-performance single-file reconstruction + multi-range fan-out.
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_XET_NUM_CONCURRENT_RANGE_GETS": os.environ.get(
        "HF_XET_NUM_CONCURRENT_RANGE_GETS", "8"
    ),
    "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": os.environ.get(
        "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS", "4"
    ),
}


def apply_public_path_compat_env() -> dict[str, str]:
    """Apply HF/Xet knobs compatible with the frozen public-path winner profile.

    The sustained winner (194 MiB/s) uses custom ``direct_presigned_range`` +
    ``python_http11_reuse``. Full-shard capture stays on ``hf_hub_download``;
    these knobs are the subset that still applies (vectored reconstruct, range
    concurrency, no chunk-cache growth, high-performance). Returns the keys
    that were newly set (previously unset).
    """
    newly: dict[str, str] = {}
    for key, value in _PUBLIC_PATH_COMPAT_ENV.items():
        if key not in os.environ:
            os.environ[key] = value
            newly[key] = value
    return newly


# Apply as early as import for processes that load this module before hub/xet.
apply_public_path_compat_env()


class LayerStreamError(Glm52Error):
    """Streaming residency / hash / floor failure (fail closed)."""


def free_bytes(path: Path | None = None) -> int:
    target = path if path is not None else Path.cwd()
    try:
        return int(shutil.disk_usage(target).free)
    except OSError:
        return int(shutil.disk_usage("/").free)


def assert_floor(path: Path | None = None, *, label: str = "stream") -> dict[str, Any]:
    free = free_bytes(path)
    ok = free >= MIN_FREE_FLOOR_BYTES
    row = {
        "label": label,
        "free_bytes": free,
        "floor_bytes": MIN_FREE_FLOOR_BYTES,
        "floor_preserved": ok,
        "at": utc_now(),
    }
    if not ok:
        raise LayerStreamError(
            f"25 GiB floor breached under {label}: free={free} floor={MIN_FREE_FLOOR_BYTES}"
        )
    return row


def load_official_lfs_hashes(
    manifest_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Map weight shard filename → {lfs_sha256, logical_bytes} from sealed manifest."""
    path = Path(manifest_path or OFFICIAL_MANIFEST)
    if not path.is_file():
        raise LayerStreamError(f"official manifest absent: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "files" not in doc:
        raise LayerStreamError("official manifest missing files[]")
    out: dict[str, dict[str, Any]] = {}
    for row in doc["files"]:
        if not isinstance(row, dict) or not row.get("is_weight"):
            continue
        name = row.get("path")
        digest = row.get("lfs_sha256")
        size = row.get("logical_bytes")
        if not isinstance(name, str) or not isinstance(digest, str) or not isinstance(size, int):
            raise LayerStreamError(f"malformed weight row in manifest: {name!r}")
        out[name] = {"lfs_sha256": digest, "logical_bytes": size}
    if len(out) < 200:
        raise LayerStreamError(f"manifest weight inventory too small: {len(out)}")
    return out


def layer_to_shards_from_index(weight_map: Mapping[str, str]) -> tuple[
    dict[int, set[str]], set[str], dict[str, set[int]]
]:
    """Derive layer↔shard membership purely from the official weight map."""
    import re

    layer_re = re.compile(r"^model\.layers\.(\d+)\.")
    layer_to_shards: dict[int, set[str]] = {}
    global_shards: set[str] = set()
    shard_to_layers: dict[str, set[int]] = {}
    for name, shard in weight_map.items():
        shard_to_layers.setdefault(shard, set())
        match = layer_re.match(name)
        if match is None:
            global_shards.add(shard)
            continue
        layer = int(match.group(1))
        layer_to_shards.setdefault(layer, set()).add(shard)
        shard_to_layers[shard].add(layer)
    return layer_to_shards, global_shards, shard_to_layers


@dataclass
class StreamEvent:
    kind: str
    detail: dict[str, Any] = field(default_factory=dict)


class LayerMajorStreamer:
    """Fetch / verify / admit / evict official BF16 shards layer-major."""

    def __init__(
        self,
        *,
        control_root: Path,
        stream_root: Path,
        repo: str = REPO_ID,
        revision: str = IMMUTABLE_REVISION,
        manifest_path: Path | None = None,
        require_floor: bool = True,
        prefetch_workers: int | None = None,
    ) -> None:
        # Env before any hub/xet import path this instance may trigger.
        self._env_applied = apply_public_path_compat_env()
        self.control_root = Path(control_root).resolve()
        self.stream_root = Path(stream_root).resolve()
        self.repo = repo
        self.revision = revision
        self.require_floor = require_floor
        workers = (
            DEFAULT_PREFETCH_WORKERS
            if prefetch_workers is None
            else int(prefetch_workers)
        )
        self.prefetch_workers = max(1, workers)
        self.lfs = load_official_lfs_hashes(manifest_path)
        self.events: list[dict[str, Any]] = []
        self.verified_hashes: dict[str, str] = {}
        self.bytes_fetched = 0
        self.bytes_reclaimed = 0
        self.fetch_seconds = 0.0
        self._prefetch_pool = ThreadPoolExecutor(
            max_workers=self.prefetch_workers, thread_name_prefix="glm52-tf-prefetch"
        )
        self._prefetch_futures: dict[str, Future] = {}
        self._lock = threading.Lock()

        if not (self.control_root / "config.json").is_file():
            raise LayerStreamError(f"config.json absent at {self.control_root}")
        if not (self.control_root / "model.safetensors.index.json").is_file():
            raise LayerStreamError(
                f"model.safetensors.index.json absent at {self.control_root}"
            )
        self.stream_root.mkdir(parents=True, exist_ok=True)

        config = load_json_strict(self.control_root / "config.json")
        if not isinstance(config, dict):
            raise LayerStreamError("config.json must be an object")
        self.config = config
        self.geometry = validate_config(config, profile=PROFILE_OFFICIAL)
        self.index = load_index(self.control_root / "model.safetensors.index.json")
        weight_map = self.index["weight_map"]
        if not isinstance(weight_map, dict):
            raise LayerStreamError("weight_map missing from official index")
        (
            self.layer_to_shards,
            self.global_shards,
            self.shard_to_layers,
        ) = layer_to_shards_from_index(weight_map)
        self.weight_map: dict[str, str] = dict(weight_map)
        self._log(
            "STREAMER_INIT",
            prefetch_workers=self.prefetch_workers,
            public_path_compat_env_newly_set=sorted(self._env_applied),
            public_path_compat_env_active={
                k: os.environ.get(k) for k in _PUBLIC_PATH_COMPAT_ENV
            },
        )

    def _log(self, kind: str, **detail: Any) -> None:
        row = {"event": kind, "at": utc_now(), **detail}
        self.events.append(row)

    def shards_for_layer(self, layer: int, *, include_global: bool = False) -> set[str]:
        names = set(self.layer_to_shards.get(int(layer), set()))
        if include_global:
            names |= set(self.global_shards)
        return names

    def resident(self) -> set[str]:
        return {
            p.name
            for p in self.stream_root.glob("model-*.safetensors")
            if p.is_file()
        }

    def missing(self, shards: Iterable[str]) -> list[str]:
        present = self.resident()
        return sorted(name for name in shards if name not in present)

    def _download_one(self, name: str) -> dict[str, Any]:
        if name not in self.lfs:
            raise LayerStreamError(f"shard not in official manifest: {name}")
        expected = self.lfs[name]
        dest = self.stream_root / name
        if dest.is_file() and dest.stat().st_size == expected["logical_bytes"]:
            # Already on disk — still hash-verify once if unseen this process.
            if name not in self.verified_hashes:
                self.verify_shard(name)
            return {
                "shard": name,
                "status": "ALREADY_RESIDENT",
                "bytes": dest.stat().st_size,
            }

        if self.require_floor:
            need = int(expected["logical_bytes"])
            free = free_bytes(self.stream_root)
            if free - need < MIN_FREE_FLOOR_BYTES:
                raise LayerStreamError(
                    f"download of {name} would breach 25 GiB floor "
                    f"(free={free} need={need} floor={MIN_FREE_FLOOR_BYTES})"
                )

        from huggingface_hub import hf_hub_download

        t0 = time.time()
        got = Path(
            hf_hub_download(
                repo_id=self.repo,
                filename=name,
                revision=self.revision,
                local_dir=str(self.stream_root),
                token=False,
            )
        )
        # hf may place under nested dirs; normalize to stream_root/name.
        if got.resolve() != dest.resolve():
            if dest.exists():
                dest.unlink()
            # Prefer hardlink when same volume; else rename/copy.
            try:
                os.link(got, dest)
            except OSError:
                shutil.copy2(got, dest)
            # Do not delete HF cache blob — leave hub cache alone; only our
            # stream_root copy is subject to source-only reclaim.
        elapsed = max(time.time() - t0, 1e-6)
        size = dest.stat().st_size
        if size != expected["logical_bytes"]:
            quarantine = dest.with_suffix(dest.suffix + ".badsize")
            os.replace(dest, quarantine)
            raise LayerStreamError(
                f"size mismatch for {name}: expected={expected['logical_bytes']} got={size}"
            )
        digest = self.verify_shard(name)
        with self._lock:
            self.bytes_fetched += size
            self.fetch_seconds += elapsed
        self._log(
            "SHARD_FETCHED",
            shard=name,
            bytes=size,
            seconds=round(elapsed, 2),
            mib_s=round(size / 1e6 / elapsed, 1),
            sha256=digest,
        )
        return {
            "shard": name,
            "status": "FETCHED_VERIFIED",
            "bytes": size,
            "sha256": digest,
            "seconds": round(elapsed, 2),
        }

    def verify_shard(self, name: str) -> str:
        path = self.stream_root / name
        if not path.is_file():
            raise LayerStreamError(f"cannot verify absent shard {name}")
        expected = self.lfs[name]["lfs_sha256"]
        size = path.stat().st_size
        if size != self.lfs[name]["logical_bytes"]:
            raise LayerStreamError(
                f"size mismatch before hash for {name}: {size} != {self.lfs[name]['logical_bytes']}"
            )
        prior = self.verified_hashes.get(name)
        if prior is not None:
            # Re-hash only if mtime/size changed; for simplicity always trust prior
            # within a single process after first full-file hash.
            return prior
        digest = sha256_file(path)
        if digest != expected:
            quarantine = path.with_suffix(path.suffix + ".badhash")
            os.replace(path, quarantine)
            raise LayerStreamError(
                f"hash mismatch for {name}: expected={expected} got={digest}"
            )
        self.verified_hashes[name] = digest
        self._log("SHARD_HASH_OK", shard=name, sha256=digest)
        return digest

    def _submit_download(self, name: str, *, event: str) -> Future | None:
        """Submit one shard download if not resident and not already in flight.

        Returns the Future to join, or None when the shard is already on disk.
        Thread-safe w.r.t. concurrent prefetch/ensure for the same name.
        """
        if name not in self.lfs:
            raise LayerStreamError(f"unknown shard {name}")
        with self._lock:
            if name in self._prefetch_futures:
                return self._prefetch_futures[name]
            dest = self.stream_root / name
            expected = self.lfs[name]["logical_bytes"]
            if dest.is_file() and dest.stat().st_size == expected:
                return None
            fut = self._prefetch_pool.submit(self._download_one, name)
            self._prefetch_futures[name] = fut
        self._log(event, shard=name)
        return fut

    def ensure(self, shards: Sequence[str] | Iterable[str]) -> list[dict[str, Any]]:
        """Block until every named shard is resident and hash-verified.

        In-flight prefetches and any still-missing shards are collected in
        parallel on the prefetch pool (not serial). Correctness is unchanged:
        every shard is LFS-sha256 verified before return.
        """
        wanted = sorted(set(shards))
        results: list[dict[str, Any]] = []
        errors: list[BaseException] = []
        pending: list[tuple[str, Future]] = []

        for name in wanted:
            fut = self._submit_download(name, event="ENSURE_SUBMIT")
            if fut is not None:
                pending.append((name, fut))

        for name, fut in pending:
            with self._lock:
                self._prefetch_futures.pop(name, None)
            try:
                results.append(fut.result())
            except BaseException as exc:  # noqa: BLE001 — aggregate then re-raise
                errors.append(exc)
                self._log(
                    "ENSURE_FAIL",
                    shard=name,
                    error=f"{type(exc).__name__}: {exc}",
                )

        if errors:
            raise LayerStreamError(
                f"ensure failed for {len(errors)} shard(s): {errors[0]}"
            ) from errors[0]

        # Final verify pass for anything already resident without a digest.
        for name in wanted:
            if name not in self.verified_hashes:
                self.verify_shard(name)
        return results

    def prefetch(self, shards: Sequence[str] | Iterable[str]) -> None:
        """Kick off background downloads for shards not yet resident / in flight."""
        for name in sorted(set(shards)):
            self._submit_download(name, event="PREFETCH_SUBMIT")

    def evict(self, shards: Sequence[str] | Iterable[str]) -> dict[str, Any]:
        """Source-only reclaim: unlink stream_root shards (never control assets)."""
        removed: list[dict[str, Any]] = []
        for name in sorted(set(shards)):
            path = self.stream_root / name
            if not path.is_file():
                continue
            # Refuse to touch anything outside stream_root.
            if path.resolve().parent != self.stream_root.resolve():
                raise LayerStreamError(f"refusing to evict outside stream_root: {path}")
            digest = self.verified_hashes.get(name) or (
                sha256_file(path) if path.is_file() else None
            )
            size = path.stat().st_size
            path.unlink()
            self.bytes_reclaimed += size
            removed.append({"shard": name, "bytes": size, "sha256": digest})
            self.verified_hashes.pop(name, None)
            fut = self._prefetch_futures.pop(name, None)
            if fut is not None:
                fut.cancel()
        self._log(
            "EVICT",
            removed=[r["shard"] for r in removed],
            bytes_reclaimed=sum(r["bytes"] for r in removed),
        )
        return {
            "removed": removed,
            "bytes_reclaimed": int(sum(r["bytes"] for r in removed)),
            "at": utc_now(),
            "policy": "source_only_reclaim_stream_scratch",
        }

    def admit_inventory(self) -> Inventory:
        """Parse headers of every currently resident stream shard into Inventory."""
        resident = sorted(self.resident())
        if not resident:
            raise LayerStreamError("no resident shards to admit")
        # verify_streaming_window algebra: source = new_fetch, carry_out = source, evict = []
        window = {
            "window_id": f"TF_RESIDENT_{len(resident)}",
            "source_shards": list(resident),
            "carry_in_shards": [],
            "new_fetch_shards": list(resident),
            "refetch_shards": [],
            "carry_out_shards": list(resident),
            "evict_after_seal_shards": [],
        }
        inv = verify_streaming_window(
            self.control_root,
            self.stream_root,
            window,
            profile=PROFILE_OFFICIAL,
            view="full",
        )
        self._log("INVENTORY_ADMITTED", shards=resident, tensor_count=inv.tensor_count)
        return inv

    def close(self) -> None:
        self._prefetch_pool.shutdown(wait=False, cancel_futures=True)

    def receipt_block(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "revision": self.revision,
            "control_root": str(self.control_root),
            "stream_root": str(self.stream_root),
            "bytes_fetched": self.bytes_fetched,
            "bytes_reclaimed": self.bytes_reclaimed,
            "fetch_seconds": round(self.fetch_seconds, 3),
            "verified_shard_count": len(self.verified_hashes),
            "verified_hashes_head": dict(list(sorted(self.verified_hashes.items()))[:8]),
            "events_tail": self.events[-40:],
            "transport": "huggingface_hub.hf_hub_download",
            "prefetch_workers": self.prefetch_workers,
            "public_path_compat_env": {
                k: os.environ.get(k) for k in _PUBLIC_PATH_COMPAT_ENV
            },
            "public_path_note": (
                "Uses hf_hub_download with multi-shard concurrency + HF_XET knobs "
                "compatible with the frozen public-path profile. Does NOT use the "
                "custom direct_presigned_range python_http11_reuse transport "
                "(194 MiB/s sustained winner) — that remains a separate path."
            ),
            "not_used": [
                "GLM52_STREAMING_SCHEDULE restream",
                "glm-donor mapping cache as authority",
                "direct_presigned_range python_http11_reuse (winner transport)",
            ],
        }


def ensure_control_assets(
    control_root: Path,
    *,
    repo: str = REPO_ID,
    revision: str = IMMUTABLE_REVISION,
) -> Path:
    """Download config + index + tokenizer into a revision-bound control root."""
    apply_public_path_compat_env()
    control_root = Path(control_root)
    control_root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    needed = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ]
    for name in needed:
        target = control_root / name
        if target.is_file() and target.stat().st_size > 0:
            continue
        got = Path(
            hf_hub_download(
                repo_id=repo,
                filename=name,
                revision=revision,
                local_dir=str(control_root),
                token=False,
            )
        )
        if got.resolve() != target.resolve() and got.is_file():
            shutil.copy2(got, target)
    return control_root
