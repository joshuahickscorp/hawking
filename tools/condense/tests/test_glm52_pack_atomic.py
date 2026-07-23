#!/usr/bin/env python3.12
"""A .gravity file must never exist in a half-written state.

The streamer treats any file with the right name as proof the source shard was
consumed, so a truncated artifact would read as complete and authorize eviction
of the BF16 body it came from.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_pack as pack  # noqa: E402
import gravity_format  # noqa: E402


def _tiny_shard(tmp_path: pathlib.Path):
    """One BF16 tensor written raw: pack_shard reads by offset, not by header."""
    rng = np.random.default_rng(0)
    weights = rng.standard_normal((32, 6144)).astype(np.float32)
    raw = (weights.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    shard = tmp_path / "model-00001-of-00282.safetensors"
    shard.write_bytes(raw.tobytes())
    rows = [{
        "name": "model.layers.0.self_attn.indexer.weights_proj.weight",
        "category": "indexer", "layer": 0, "expert": None,
        "dtype": "BF16", "shape": [32, 6144],
        "absolute_start": 0, "payload_bytes": raw.nbytes,
        "provisional_budget_class": "COMPRESSIBLE_CANDIDATE",
    }]
    return shard, rows


def test_pack_leaves_no_partial_and_verifies(tmp_path):
    shard, rows = _tiny_shard(tmp_path)
    out = tmp_path / "compact"
    receipt = pack.pack_shard(shard, rows, out)

    gravity = out / "model-00001-of-00282.gravity"
    assert gravity.exists(), "pack produced no compact artifact"
    assert list(out.glob("*.partial")) == [], "a partial file survived the pack"
    assert receipt["shard"] == shard.name
    assert gravity_format.verify(gravity)["ok"], "packed artifact does not verify"


def test_a_partial_write_never_takes_the_final_name(tmp_path, monkeypatch):
    """If the write dies, the .gravity name must still be absent."""
    shard, rows = _tiny_shard(tmp_path)
    out = tmp_path / "compact"

    real = gravity_format.write_shard

    def die(path, payloads, **kwargs):
        real(path, payloads, **kwargs)  # write the partial, then fail before rename
        raise OSError("simulated crash after the body was written")

    monkeypatch.setattr(pack.gravity_format, "write_shard", die)
    with pytest.raises(OSError):
        pack.pack_shard(shard, rows, out)

    assert not (out / "model-00001-of-00282.gravity").exists(), \
        "a killed pack claimed the final name and would authorize eviction"
    assert list(out.glob("*.partial")), "the partial write should be left visible"
