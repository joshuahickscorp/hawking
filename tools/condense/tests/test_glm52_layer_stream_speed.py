#!/usr/bin/env python3.12
"""Speed-audit unit tests for layer-major streaming concurrency + env knobs.

Does not hit the network. Proves:
  * ensure() fans out missing shards to the pool (not serial call order only)
  * apply_public_path_compat_env is idempotent and sets expected keys
  * default prefetch worker count is >= 4 (Kimi-prep concurrency)
"""
from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lab.operators import glm52_layer_stream as ls  # noqa: E402


def test_apply_public_path_compat_env_sets_and_is_idempotent(monkeypatch):
    for key in list(ls._PUBLIC_PATH_COMPAT_ENV):
        monkeypatch.delenv(key, raising=False)
    first = ls.apply_public_path_compat_env()
    assert "HF_XET_HIGH_PERFORMANCE" in first
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert os.environ["HF_XET_RECONSTRUCTION_USE_VECTORED_WRITE"] == "true"
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
    second = ls.apply_public_path_compat_env()
    assert second == {}  # already set → no overwrite


def test_default_prefetch_workers_at_least_four():
    assert ls.DEFAULT_PREFETCH_WORKERS >= 4


def test_ensure_downloads_missing_shards_in_parallel(tmp_path, monkeypatch):
    """Synthetic streamer: four missing shards must be submitted concurrently."""
    control = tmp_path / "control"
    stream = tmp_path / "stream"
    control.mkdir()
    stream.mkdir()
    # Minimal control plane files so __init__ can proceed after we stub heavy bits.
    (control / "config.json").write_text("{}")
    (control / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":0},"weight_map":{}}'
    )

    # Build a streamer without real official manifest / geometry.
    with patch.object(ls, "load_official_lfs_hashes", return_value={
        f"model-{i:05d}-of-00004.safetensors": {
            "lfs_sha256": "a" * 64,
            "logical_bytes": 16,
        }
        for i in range(4)
    }), patch.object(ls, "load_json_strict", return_value={"num_hidden_layers": 1}), patch.object(
        ls, "validate_config", return_value=MagicMock()
    ), patch.object(
        ls,
        "load_index",
        return_value={"weight_map": {}},
    ), patch.object(
        ls,
        "layer_to_shards_from_index",
        return_value=({0: set()}, set(), {}),
    ):
        streamer = ls.LayerMajorStreamer(
            control_root=control,
            stream_root=stream,
            require_floor=False,
            prefetch_workers=4,
            manifest_path=None,
        )

    active = 0
    peak = 0
    lock = threading.Lock()
    call_order: list[str] = []

    def fake_download(name: str):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            call_order.append(f"start:{name}")
        time.sleep(0.08)
        dest = stream / name
        dest.write_bytes(b"\x00" * 16)
        with lock:
            active -= 1
            call_order.append(f"end:{name}")
        with streamer._lock:
            streamer.verified_hashes[name] = "a" * 64
            streamer.bytes_fetched += 16
        return {
            "shard": name,
            "status": "FETCHED_VERIFIED",
            "bytes": 16,
            "sha256": "a" * 64,
            "seconds": 0.08,
        }

    streamer._download_one = fake_download  # type: ignore[method-assign]
    # Bypass size/hash verify on already-written files.
    streamer.verify_shard = lambda name: streamer.verified_hashes.setdefault(name, "a" * 64)  # type: ignore[method-assign]
    streamer.lfs = {
        f"model-{i:05d}-of-00004.safetensors": {
            "lfs_sha256": "a" * 64,
            "logical_bytes": 16,
        }
        for i in range(4)
    }

    names = [f"model-{i:05d}-of-00004.safetensors" for i in range(4)]
    t0 = time.perf_counter()
    results = streamer.ensure(names)
    elapsed = time.perf_counter() - t0
    streamer.close()

    assert len(results) == 4
    assert peak >= 3, f"expected concurrent downloads, peak={peak} order={call_order}"
    # Serial would be ~0.32s; parallel ~0.08–0.15s with overhead.
    assert elapsed < 0.25, f"ensure too slow (likely serial): {elapsed:.3f}s peak={peak}"
    for name in names:
        assert (stream / name).is_file()


def test_ensure_idempotent_when_already_resident(tmp_path):
    control = tmp_path / "control"
    stream = tmp_path / "stream"
    control.mkdir()
    stream.mkdir()
    (control / "config.json").write_text("{}")
    (control / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":0},"weight_map":{}}'
    )
    name = "model-00000-of-00001.safetensors"
    (stream / name).write_bytes(b"\x00" * 16)

    with patch.object(ls, "load_official_lfs_hashes", return_value={
        name: {"lfs_sha256": "b" * 64, "logical_bytes": 16}
    }), patch.object(ls, "load_json_strict", return_value={}), patch.object(
        ls, "validate_config", return_value=MagicMock()
    ), patch.object(ls, "load_index", return_value={"weight_map": {}}), patch.object(
        ls, "layer_to_shards_from_index", return_value=({0: {name}}, set(), {name: {0}})
    ):
        streamer = ls.LayerMajorStreamer(
            control_root=control,
            stream_root=stream,
            require_floor=False,
            prefetch_workers=2,
        )
    streamer.verified_hashes[name] = "b" * 64
    calls = {"n": 0}

    def boom(n):
        calls["n"] += 1
        raise AssertionError("should not download resident shard")

    streamer._download_one = boom  # type: ignore[method-assign]
    streamer.verify_shard = lambda n: streamer.verified_hashes[n]  # type: ignore[method-assign]
    out = streamer.ensure([name])
    streamer.close()
    assert out == []
    assert calls["n"] == 0
