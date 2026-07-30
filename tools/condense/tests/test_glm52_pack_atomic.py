#!/usr/bin/env python3.12
"""A .gravity file must never exist in a half-written state.

The streamer treats any file with the right name as proof the source shard was
consumed, so a truncated artifact would read as complete and authorize eviction
of the BF16 body it came from.
"""
from __future__ import annotations
import sys
from pathlib import Path as _Path_repo
_REPO = _Path_repo(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pathlib
import sys

import numpy as np
import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]

from lab.operators import glm52_pack as pack  # noqa: E402
from tools.condense import artifact_client as gravity_format  # noqa: E402

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

def test_ladder_survey_samples_experts_but_never_the_target_rung(tmp_path, monkeypatch):
    """Sampling may thin the rate survey; it may never thin the artifact."""
    monkeypatch.setattr(pack, "LADDER_SAMPLE_EVERY", 2)
    shard, rows = _tiny_shard(tmp_path)
    base = rows[0]
    rows = []
    for index in range(4):  # four identically shaped routed-expert tensors
        row = dict(base)
        row["name"] = f"model.layers.0.mlp.experts.{index}.gate_proj.weight"
        row["category"], row["expert"] = "routed_expert", index
        rows.append(row)
    out = tmp_path / "compact"
    receipt = pack.pack_shard(shard, rows, out)

    assert receipt["ladder_sample_every_nth_routed_expert"] == 2
    assert receipt["ladder_tensors_fully_surveyed"] == 2, "expected every 2nd expert surveyed"

    header = gravity_format.read_header(out / "model-00001-of-00282.gravity")
    survey = header["compression"]["ladder_survey"]
    assert survey["target_rung_coverage"] == "ALL_TENSORS"
    assert survey["routed_expert_tensors_seen"] == 4

    full = thin = 0
    for entry in header["tensors"]:
        by_rung = {r["rung"]: r for r in entry["ladder"]}
        assert by_rung[pack.PRODUCTION_RUNG].get("admitted"), \
            "the production rung must be fitted on every tensor"
        if by_rung["R2"].get("sampled_out"):
            thin += 1
            # a skipped measurement must be legible as skipped, not as a failure
            assert by_rung["R2"]["reason"] == "NOT_IN_THIS_TENSOR_LADDER_SAMPLE"
        else:
            full += 1
    assert (full, thin) == (2, 2)

def test_the_fit_kernel_is_recorded_and_never_moves_the_ledger(tmp_path, monkeypatch):
    """Either k-means arithmetic may be selected; the shard must say which, and the billed size must not de"""
    shard, rows = _tiny_shard(tmp_path)
    sizes = {}
    for kernel in pack.forge.FIT_KERNELS:
        monkeypatch.setattr(pack.forge, "FIT_KERNEL", kernel)
        out = tmp_path / kernel
        pack.pack_shard(shard, rows, out)
        header = gravity_format.read_header(out / "model-00001-of-00282.gravity")
        assert header["compression"]["fit_kernel"] == kernel, \
            "a shard that does not name its fit kernel is not reproducible from (seed, iters)"
        sizes[kernel] = [entry["bytes"] for entry in header["tensors"]]

    assert sizes["v1_full_distance"] == sizes["v2_lean_argmin"], \
        "the fit kernel changed a billed payload size; the frozen byte auction would be void"

def test_an_overridden_tensor_is_sampled_like_any_other_routed_expert(tmp_path, monkeypatch):
    """A Prometheus override must move the target rung, not exempt the schedule. This is the case Math-Pres"""
    monkeypatch.setattr(pack, "LADDER_SAMPLE_EVERY", 2)
    shard, rows = _tiny_shard(tmp_path)
    base = rows[0]
    rows = []
    for index in range(4):
        row = dict(base)
        row["name"] = f"model.layers.0.mlp.experts.{index}.gate_proj.weight"
        row["category"], row["expert"] = "routed_expert", index
        rows.append(row)
    override = {row["name"]: "R2" for row in rows}

    out = tmp_path / "compact"
    receipt = pack.pack_shard(shard, rows, out, rate_override=override)
    assert receipt["ladder_tensors_fully_surveyed"] == 2, \
        "an override must not exempt a routed expert from the survey schedule"

    header = gravity_format.read_header(out / "model-00001-of-00282.gravity")
    sampled_out_production = 0
    for entry in header["tensors"]:
        assert entry["rung"] == "R2", "the artifact must carry the overridden rung"
        by_rung = {r["rung"]: r for r in entry["ladder"]}
        assert by_rung["R2"].get("admitted"), "the target rung must be fitted"
        if by_rung[pack.PRODUCTION_RUNG].get("sampled_out"):
            # the whole point: a rung nothing consumes is no longer paid for
            sampled_out_production += 1
    assert sampled_out_production == 2, \
        "the unused production rung should be skipped on every non-survey tensor"

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
