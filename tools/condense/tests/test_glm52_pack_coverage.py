#!/usr/bin/env python3.12
"""Complete tensor coverage: every declared source tensor lands in the artifact.

The Tier-0 defect that invalidated Generation B's predecessor was an accounting split:
`pack_shard` added protected tensors to the billed total and to the descriptor list, but
only handed compressed payloads to `write_shard`.  Routers, router controls, norms and
indexer tensors were billed at 16 BPW and written nowhere, and the resulting artifact
still read as proof the BF16 body could be evicted.

These tests pin the repaired contract: a tensor is billed only where it is stored, the
complete rate reconciles against physical bytes, and an artifact missing any declared
tensor is malformed rather than merely thin.
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


def _bf16(values: np.ndarray) -> bytes:
    return (values.view(np.uint32) >> np.uint32(16)).astype(np.uint16).tobytes()


def _organ_shard(tmp_path: pathlib.Path):
    """A shard shaped like a real GLM layer boundary: experts plus protected organs.

    Every category the reproduction found absent from Generation A output is present
    here: a routed expert (compressible), a router weight and a router score-correction
    bias (protected, and the bias is F32 rather than BF16), a post-attention norm, and
    two indexer tensors.
    """
    rng = np.random.default_rng(7)
    blobs: list[bytes] = []
    rows: list[dict] = []
    offset = 0

    def add(name: str, category: str, shape: tuple[int, ...], dtype: str,
            budget_class: str, expert: int | None = None) -> None:
        nonlocal offset
        values = rng.standard_normal(shape).astype(np.float32)
        raw = _bf16(values) if dtype == "BF16" else values.tobytes()
        blobs.append(raw)
        rows.append({
            "name": name, "category": category, "layer": 34, "expert": expert,
            "dtype": dtype, "shape": list(shape),
            "absolute_start": offset, "payload_bytes": len(raw),
            "provisional_budget_class": budget_class,
        })
        offset += len(raw)

    add("model.layers.34.mlp.experts.0.gate_proj.weight", "routed_expert",
        (256, 1536), "BF16", "COMPRESSIBLE_CANDIDATE", expert=0)
    add("model.layers.34.mlp.experts.1.gate_proj.weight", "routed_expert",
        (256, 1536), "BF16", "COMPRESSIBLE_CANDIDATE", expert=1)
    add("model.layers.34.mlp.gate.weight", "router",
        (256, 6144), "BF16", pack.PROTECTED_BUDGET_CLASS)
    add("model.layers.34.mlp.gate.e_score_correction_bias", "router_control",
        (256,), "F32", pack.PROTECTED_BUDGET_CLASS)
    add("model.layers.34.post_attention_layernorm.weight", "normalization",
        (6144,), "BF16", pack.PROTECTED_BUDGET_CLASS)
    add("model.layers.34.self_attn.indexer.wk.weight", "indexer",
        (128, 6144), "BF16", pack.PROTECTED_BUDGET_CLASS)
    add("model.layers.34.self_attn.indexer.k_norm.weight", "indexer",
        (128,), "BF16", pack.PROTECTED_BUDGET_CLASS)

    shard = tmp_path / "model-00097-of-00282.safetensors"
    shard.write_bytes(b"".join(blobs))
    return shard, rows


def test_every_declared_tensor_is_physically_present(tmp_path):
    shard, rows = _organ_shard(tmp_path)
    out = tmp_path / "compact"
    receipt = pack.pack_shard(shard, rows, out)

    header = gravity_format.read_header(out / "model-00097-of-00282.gravity")
    present = {tensor["name"] for tensor in header["tensors"]}
    declared = {row["name"] for row in rows}
    assert present == declared, f"absent from the artifact: {sorted(declared - present)}"

    for tensor in header["tensors"]:
        assert int(tensor["bytes"]) > 0, f"{tensor['name']} was billed with no payload"

    assert receipt["tensor_coverage"] == "COMPLETE_EVERY_DECLARED_TENSOR_PHYSICALLY_PRESENT"
    assert receipt["native_tensors"] == 5, receipt
    assert receipt["compressed_tensors"] == 2, receipt


def test_protected_organs_come_back_bit_identical(tmp_path):
    """A native organ is not an approximation.  The bytes must survive the round trip."""
    shard, rows = _organ_shard(tmp_path)
    out = tmp_path / "compact"
    pack.pack_shard(shard, rows, out)
    gravity = out / "model-00097-of-00282.gravity"

    source = shard.read_bytes()
    for row in rows:
        if row["provisional_budget_class"] != pack.PROTECTED_BUDGET_CLASS:
            continue
        expected = source[row["absolute_start"]:row["absolute_start"] + row["payload_bytes"]]
        assert gravity_format.read_tensor(gravity, row["name"]) == expected, row["name"]


def test_non_bf16_control_tensors_are_carried_not_dropped(tmp_path):
    """F32 controls used to fail the dtype check and never reach the descriptor list."""
    shard, rows = _organ_shard(tmp_path)
    out = tmp_path / "compact"
    pack.pack_shard(shard, rows, out)

    header = gravity_format.read_header(out / "model-00097-of-00282.gravity")
    bias = next(t for t in header["tensors"]
                if t["name"].endswith("e_score_correction_bias"))
    assert bias["codec"] == "native.f32"
    assert bias["elements"] == 256
    assert int(bias["bytes"]) == 256 * 4
    assert bias["bpw"] == pytest.approx(32.0)


def test_complete_rate_reconciles_against_physical_bytes(tmp_path):
    """complete_bpw counts every weight and every byte, so organs cannot hide in it."""
    shard, rows = _organ_shard(tmp_path)
    out = tmp_path / "compact"
    receipt = pack.pack_shard(shard, rows, out)
    gravity = out / "model-00097-of-00282.gravity"

    report = gravity_format.verify(gravity)
    assert report["ok"], report
    assert report["tensors_without_payload"] == []
    assert report["complete_rate_self_consistent"], report
    assert report["packed_rate_self_consistent"], report

    body = gravity.stat().st_size - gravity_format._body_offset(
        gravity_format.read_header(gravity))
    declared_elements = sum(int(np.prod(row["shape"])) for row in rows)
    assert report["observed_complete_bpw"] == pytest.approx(
        body * 8 / declared_elements, rel=1e-9)

    # The organs cost 16 or 32 bits each, so the complete rate must sit above the
    # compressed rate.  A complete rate at or below the packed rate would mean the
    # native bytes were not counted.
    assert receipt["complete_bpw"] > receipt["packed_bpw"]


def test_an_artifact_missing_its_organs_is_rejected(tmp_path, monkeypatch):
    """Reproduce Generation A exactly, then show both defences reject the result.

    Dropping the native payloads on the way to the writer is precisely what the old code
    did.  The artifact that comes out is byte-perfect, hashes clean, and describes weight
    it does not carry, which is the whole reason a hash check was never enough.
    """
    shard, rows = _organ_shard(tmp_path)
    out = tmp_path / "compact"
    original = gravity_format.write_shard

    def strip_organs(path, payloads, **kwargs):
        kept = [(d, b) for d, b in payloads if not str(d["codec"]).startswith("native.")]
        return original(path, kept, **kwargs)

    monkeypatch.setattr(pack.gravity_format, "write_shard", strip_organs)
    pack.pack_shard(shard, rows, out)
    gravity = out / "model-00097-of-00282.gravity"

    header = gravity_format.read_header(gravity)
    assert len(header["tensors"]) < len(rows), "the fixture failed to reproduce the defect"

    report = gravity_format.verify(gravity)
    assert report["body_sha256_ok"], "the defect is invisible to an integrity check"
    assert report["bad_tensors"] == [], "the defect is invisible to per-tensor hashes"
    assert not report["ok"], "an artifact missing its organs was accepted"
    assert not report["complete_rate_self_consistent"], report


# The in-packer PackCoverageError guard has no test because it is now unreachable through
# the public API: every branch in the read loop hands its bytes to the writer.  It stays as
# a backstop against a future branch that forgets to, and gravity_format.verify above is
# the defence that actually runs against every artifact the campaign produces.
