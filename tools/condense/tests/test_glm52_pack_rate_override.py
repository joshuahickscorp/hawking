#!/usr/bin/env python3.12
"""`rate_override` is Prometheus's only entry point into the packer: it must be a
no-op when unused, and when used it must change exactly the tensors named and
nothing else -- a coalition decision that leaked onto an uninvolved expert would
silently rewrite Claim A's byte-matching without anyone asking it to.
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
import artifact_client as gravity_format  # noqa: E402

def _shard_with_experts(tmp_path: pathlib.Path, n: int = 4):
    """`n` identically-shaped routed-expert tensors, one per (layer 0, expert i)."""
    rng = np.random.default_rng(0)
    tensors = [rng.standard_normal((32, 6144)).astype(np.float32) for _ in range(n)]
    raws = [(w.view(np.uint32) >> np.uint32(16)).astype(np.uint16) for w in tensors]
    shard = tmp_path / "model-00001-of-00282.safetensors"
    shard.write_bytes(b"".join(r.tobytes() for r in raws))

    rows = []
    offset = 0
    for index, raw in enumerate(raws):
        rows.append({
            "name": f"model.layers.0.mlp.experts.{index}.gate_proj.weight",
            "category": "routed_expert", "layer": 0, "expert": index,
            "dtype": "BF16", "shape": [32, 6144],
            "absolute_start": offset, "payload_bytes": raw.nbytes,
            "provisional_budget_class": "COMPRESSIBLE_CANDIDATE",
        })
        offset += raw.nbytes
    return shard, rows

def _descriptors_by_expert(gravity_path: pathlib.Path) -> dict[int, dict]:
    header = gravity_format.read_header(gravity_path)
    return {entry["expert"]: entry for entry in header["tensors"]}

def test_no_override_is_byte_identical_to_the_existing_packer(tmp_path):
    shard, rows = _shard_with_experts(tmp_path)
    baseline_dir, overridden_dir = tmp_path / "baseline", tmp_path / "overridden"

    # Warmup: pack_shard's first call in a process produces different bytes than
    # every later call with identical arguments (pre-existing, unrelated to
    # rate_override -- see task_bfe9935a). Discard one call before the real
    # comparison so this test isn't sensitive to that separate bug.
    pack.pack_shard(shard, rows, tmp_path / "warmup", seed=0)

    pack.pack_shard(shard, rows, baseline_dir, seed=0)
    pack.pack_shard(shard, rows, overridden_dir, seed=0, rate_override=None)
    pack.pack_shard(shard, rows, tmp_path / "empty_dict", seed=0, rate_override={})

    baseline = (baseline_dir / "model-00001-of-00282.gravity").read_bytes()
    for other in ("overridden", "empty_dict"):
        name = "model-00001-of-00282.gravity"
        got = (tmp_path / other / name).read_bytes()
        assert got == baseline, f"rate_override={other!r} changed packed bytes with no override entries"

def test_telemetry_is_a_side_channel_and_does_not_change_artifact_bytes(tmp_path):
    shard, rows = _shard_with_experts(tmp_path, n=1)
    pack.pack_shard(shard, rows, tmp_path / "warmup", seed=0)
    pack.pack_shard(shard, rows, tmp_path / "baseline", seed=0)

    telemetry = {}
    pack.pack_shard(
        shard, rows, tmp_path / "instrumented", seed=0, telemetry=telemetry
    )
    name = "model-00001-of-00282.gravity"
    assert (tmp_path / "instrumented" / name).read_bytes() == (
        tmp_path / "baseline" / name
    ).read_bytes()
    assert telemetry["timing_side_channel_only"] is True
    assert telemetry["excluded_from_artifact_and_canonical_receipts"] is True
    assert telemetry["stage_seconds"]["fit"] > 0
    assert telemetry["categories"]["routed_expert"]["tensors"] == 1
    assert telemetry["tensors_per_second"] > 0
    assert telemetry["weights_per_second"] > 0

def test_ladder_sampling_does_not_change_any_tensor_payload(tmp_path):
    """Surveying fewer rungs must move only the survey record, never the artifact. The schedule change that"""
    shard, rows = _shard_with_experts(tmp_path, n=8)
    override = {f"model.layers.0.mlp.experts.{i}.gate_proj.weight": "R4" for i in range(8)}
    pack.pack_shard(shard, rows, tmp_path / "warmup", seed=0, rate_override=override)

    name = "model-00001-of-00282.gravity"
    payloads = {}
    for label, every in (("exhaustive", 1), ("sampled", 10_000)):
        original, pack.LADDER_SAMPLE_EVERY = pack.LADDER_SAMPLE_EVERY, every
        try:
            pack.pack_shard(shard, rows, tmp_path / label, seed=0, rate_override=override)
        finally:
            pack.LADDER_SAMPLE_EVERY = original
        header, body = gravity_format.open_shard(tmp_path / label / name)
        blob = (tmp_path / label / name).read_bytes()
        payloads[label] = {
            entry["name"]: blob[body + entry["offset"]: body + entry["offset"] + entry["bytes"]]
            for entry in header["tensors"]
        }
        assert all(e["rung"] == "R4" for e in header["tensors"]), label

    assert payloads["exhaustive"] == payloads["sampled"], (
        "ladder survey density changed a packed tensor payload")

    # ...and the survey record itself must move, or the schedule did nothing.
    surveyed = {}
    for label in ("exhaustive", "sampled"):
        header = gravity_format.read_header(tmp_path / label / name)
        surveyed[label] = header["compression"]["ladder_survey"]["tensors_fully_surveyed"]
    assert surveyed["exhaustive"] == 8 and surveyed["sampled"] == 1, surveyed

def test_native_override_protects_exactly_the_named_expert(tmp_path):
    shard, rows = _shard_with_experts(tmp_path, n=4)
    out = tmp_path / "compact"
    pack.pack_shard(shard, rows, out, seed=0, rate_override={(0, 2): "native"})

    by_expert = _descriptors_by_expert(out / "model-00001-of-00282.gravity")
    assert by_expert[2]["codec"].startswith("native."), "overridden expert must be native"
    assert by_expert[2]["terminal_state"] == "PROTECTED_SOURCE_NATIVE"
    assert by_expert[2]["reason"] == "PROMETHEUS_COALITION_PROTECTED"
    assert by_expert[2]["bpw"] == pytest.approx(16.0, abs=0.1)

    for expert in (0, 1, 3):
        assert by_expert[expert]["codec"] == "gravity-pq", \
            f"expert {expert} was not overridden and must pack exactly as before"
        assert by_expert[expert]["rung"] == pack.PRODUCTION_RUNG

def test_rung_override_picks_a_different_rung_for_exactly_that_expert(tmp_path):
    shard, rows = _shard_with_experts(tmp_path, n=4)
    out = tmp_path / "compact"
    pack.pack_shard(shard, rows, out, seed=0, rate_override={(0, 1): "R2"})

    by_expert = _descriptors_by_expert(out / "model-00001-of-00282.gravity")
    assert by_expert[1]["codec"] == "gravity-pq"
    assert by_expert[1]["rung"] == "R2"
    for expert in (0, 2, 3):
        assert by_expert[expert]["rung"] == pack.PRODUCTION_RUNG

    # An overridden tensor must survey every rung (it is not the sampled-out common
    # case LADDER_SAMPLE_EVERY exists to bound), so its own ladder record proves the
    # override rung was actually admitted, not defaulted to on a missing lookup.
    ladder_by_rung = {r["rung"]: r for r in by_expert[1]["ladder"]}
    assert ladder_by_rung["R2"]["admitted"]

def test_override_key_absent_from_this_shard_is_silently_irrelevant(tmp_path):
    """A coalition manifest is whole-model; a shard only ever sees its own tensors.
    An override naming a (layer, expert) this shard does not carry must not error --
    that is the ordinary case for every shard, not an edge case."""
    shard, rows = _shard_with_experts(tmp_path, n=2)
    out = tmp_path / "compact"
    receipt = pack.pack_shard(
        shard, rows, out, seed=0,
        rate_override={(47, 200): "native", (0, 0): "native"},
    )
    assert receipt["tensors"] == 2
    by_expert = _descriptors_by_expert(out / "model-00001-of-00282.gravity")
    assert by_expert[0]["codec"].startswith("native.")
    assert by_expert[1]["codec"] == "gravity-pq"

def test_exact_tensor_name_takes_precedence_over_expert_fallback(tmp_path):
    """PASS2 freezes per-tensor decisions.  A name-level decision must therefore
    be able to protect one matrix without leaking onto the expert's other matrices,
    and must win if a coarser expert fallback is also present."""
    shard, rows = _shard_with_experts(tmp_path, n=2)
    out = tmp_path / "compact"
    selected = rows[1]["name"]
    pack.pack_shard(
        shard, rows, out, seed=0,
        rate_override={(0, 1): "R4", selected: "native"},
    )
    by_expert = _descriptors_by_expert(out / "model-00001-of-00282.gravity")
    assert by_expert[1]["codec"].startswith("native.")
    assert by_expert[1]["reason"] == "PROMETHEUS_COALITION_PROTECTED"
    assert by_expert[0]["codec"] == "gravity-pq"
    assert by_expert[0]["rung"] == pack.PRODUCTION_RUNG
