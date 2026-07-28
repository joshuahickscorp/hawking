#!/usr/bin/env python3.12
"""Synthetic ABI tests for activation-aware assembly and numpy execution."""
from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import sys

import numpy as np
import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import activation_aware_format as fmt  # noqa: E402
import glm52_activation_aware_assemble as assemble  # noqa: E402
import glm52_activation_aware_pack as pack  # noqa: E402
from glm52_activation_aware_source import ActivationAwareGlmSource  # noqa: E402


def _fixture(root: pathlib.Path) -> dict:
    shard_name = "model-00001-of-00001.aap"
    source_name = "model-00001-of-00001.safetensors"
    shard = root / shard_name

    basis = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.5, -0.25]], dtype=np.float32
    )
    input_left = np.array([[2.0, -1.0], [0.5, 3.0]], dtype=np.float32)
    output_left = np.array([[1.0, 2.0], [-0.5, 4.0]], dtype=np.float32)
    input_blob = pack.serialize_tensor_payload(
        input_left,
        basis,
        side="input",
        rows=2,
        cols=3,
        rank=2,
        basis_layer=7,
        bill_basis=False,
    )
    output_blob = pack.serialize_tensor_payload(
        output_left,
        basis,
        side="output",
        rows=3,
        cols=2,
        rank=2,
        basis_layer=7,
        bill_basis=False,
    )
    native = np.array([1.5, -2.0, 0.25], dtype=np.float32)
    native_u16 = (native.view(np.uint32) >> np.uint32(16)).astype("<u2")
    native_header = struct.pack("<8sIII", b"GLM52PT0", 1, 3, 0)
    native_blob = native_header + b"\0" * (64 - len(native_header)) + native_u16.tobytes()

    basis_header = struct.pack("<8sII", b"GLM52BAS", 3, 2)
    basis_blob = (
        basis_header
        + b"\0" * (64 - len(basis_header))
        + np.asarray(basis, dtype="<f2").tobytes()
    )
    payloads = [basis_blob, input_blob, output_blob, native_blob]
    offsets = []
    offset = 0
    for blob in payloads:
        offsets.append(offset)
        offset += len(blob)

    names = {
        "input": "model.layers.0.input.weight",
        "output": "model.layers.0.output.weight",
        "native": "model.layers.0.norm.weight",
    }
    index = {
        "schema": fmt.SCHEMA,
        "shard": source_name,
        "shared_bases": True,
        "bases": [
            {
                "basis_layer": 7,
                "rank": 2,
                "bytes": len(basis_blob),
                "offset": offsets[0],
            }
        ],
        "tensors": [
            {
                "name": names["input"],
                "disposition": "activation_aware",
                "shape": [2, 3],
                "side": "input",
                "rank": 2,
                "bytes": len(input_blob),
                "offset": offsets[1],
            },
            {
                "name": names["output"],
                "disposition": "activation_aware",
                "shape": [3, 2],
                "side": "output",
                "rank": 2,
                "bytes": len(output_blob),
                "offset": offsets[2],
            },
            {
                "name": names["native"],
                "disposition": "pass_through",
                "shape": [3],
                "bytes": len(native_blob),
                "offset": offsets[3],
            },
        ],
    }
    encoded = json.dumps(index, sort_keys=True, separators=(",", ":")).encode()
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads))
    shard_hash = hashlib.sha256(shard.read_bytes()).hexdigest()

    measurement = {
        "schema": "hawking.glm52.activation_aware_measurement.v1",
        "measurements": [
            {"name": name, "dtype": "BF16"} for name in names.values()
        ],
    }
    (root / "MEASUREMENT.json").write_text(json.dumps(measurement))
    manifest = {
        "schema": "hawking.activation_aware.model_index.v1",
        "architecture": {"hidden_size": 3},
        "shards": [shard_name],
        "shard_sha256": {shard_name: shard_hash},
        "weight_map": {name: shard_name for name in names.values()},
        "tensor_dtypes": {name: "BF16" for name in names.values()},
    }
    (root / "model.activation_aware.index.json").write_text(json.dumps(manifest))
    return {
        "shard": shard,
        "source_name": source_name,
        "shard_name": shard_name,
        "names": names,
        "basis": basis,
        "input_left": input_left,
        "output_left": output_left,
        "native": native,
    }


def test_format_source_executes_factorized_matvec_and_rows(tmp_path):
    fx = _fixture(tmp_path)
    index, body_offset = fmt.read_index(fx["shard"])
    fmt.validate_payload_magics(fx["shard"], index, body_offset)
    source = ActivationAwareGlmSource(tmp_path, verify_hash=True)

    input_weight = fx["input_left"] @ fx["basis"].T
    output_weight = fx["basis"] @ fx["output_left"]
    np.testing.assert_allclose(
        source.tensor(fx["names"]["input"]), input_weight, rtol=0, atol=2e-3
    )
    np.testing.assert_allclose(
        source.tensor(fx["names"]["output"]), output_weight, rtol=0, atol=2e-3
    )

    x_input = np.array([[0.25], [2.0], [-1.0]], dtype=np.float32)
    x_output = np.array([[1.5], [-0.75]], dtype=np.float32)
    np.testing.assert_allclose(
        source.matvec(fx["names"]["input"], x_input),
        input_weight @ x_input,
        rtol=0,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        source.matvec(fx["names"]["output"], x_output),
        output_weight @ x_output,
        rtol=0,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        source.rows(fx["names"]["output"], np.array([[2, 0]])),
        output_weight[[[2, 0]]],
        rtol=0,
        atol=2e-3,
    )
    np.testing.assert_array_equal(
        source.tensor(fx["names"]["native"]), fx["native"]
    )


def test_shard_hash_binds_runtime_bytes(tmp_path):
    fx = _fixture(tmp_path)
    raw = bytearray(fx["shard"].read_bytes())
    raw[-1] ^= 0x80
    fx["shard"].write_bytes(raw)
    source = ActivationAwareGlmSource(tmp_path, verify_hash=True)
    with pytest.raises(fmt.ActivationAwareFormatError, match="SHA-256 mismatch"):
        source.tensor(fx["names"]["native"])


def test_assembly_check_grades_official_map_not_packer_against_itself(tmp_path):
    fx = _fixture(tmp_path)
    expected = {
        name: fx["source_name"] for name in fx["names"].values()
    }
    coverage = assemble.check(tmp_path, expected_weight_map=expected)
    assert coverage["verdict"] == "COMPLETE"
    assert coverage["official_tensors"] == 3
    assert coverage["dispositions"] == {
        "ACTIVATION_AWARE_PAYLOAD": 2,
        "PASS_THROUGH_PAYLOAD": 1,
    }

    expected["model.layers.0.missing.weight"] = fx["source_name"]
    incomplete = assemble.check(tmp_path, expected_weight_map=expected)
    assert incomplete["verdict"] == "INCOMPLETE"
    assert incomplete["missing_count"] == 1


def test_zero_byte_descriptor_counts_as_missing_not_complete(tmp_path, monkeypatch):
    """A named tensor billing zero bytes is a hole — sample and missing_count agree.

    The format reader rejects spans under HEADER_BYTES for real shards; this still
    pins the completeness arithmetic so a descriptor cannot promote the coverage
    proof the way a self-consistent packer index once could.
    """
    fx = _fixture(tmp_path)
    scan = assemble.scan_artifact(tmp_path)
    ghost = "model.layers.0.ghost.weight"
    scan["tensors"][ghost] = {
        "shard": fx["shard_name"],
        "disposition": "pass_through",
        "dtype": "BF16",
        "bytes": 0,
        "shape": [1],
    }
    expected = {name: fx["source_name"] for name in fx["names"].values()}
    expected[ghost] = fx["source_name"]
    (tmp_path / "MEASUREMENT.json").write_text(
        json.dumps(
            {
                "schema": "hawking.glm52.activation_aware_measurement.v1",
                "measurements": [
                    {"name": name, "dtype": "BF16"} for name in expected
                ],
            }
        )
    )
    monkeypatch.setattr(assemble, "scan_artifact", lambda _artifact: scan)
    coverage = assemble.check(
        tmp_path,
        expected_weight_map=expected,
        measurement_path=tmp_path / "MEASUREMENT.json",
    )
    assert coverage["verdict"] == "INCOMPLETE"
    assert coverage["missing_count"] == 1, coverage
    assert ghost in coverage["missing_sample"]
    # Before the fix, missing_count used set-difference only and would have been 0
    # while the sample still listed the ghost — complete could have flipped true.
    assert coverage["complete"] is False


def test_assemble_writes_index_openable_for_tensor_fetch(tmp_path, monkeypatch):
    """assemble() success path: index + hashes + dtypes → source.tensor().

    The packer-only fixture hand-wrote the model index.  That left the real
    assembler — the thing the post-pack handoff runs — unexercised.  Partial
    coverage must still refuse; complete coverage must open through the same
    ActivationAwareGlmSource the flagship oracle uses.
    """
    fx = _fixture(tmp_path)
    # Drop the hand-written index so only assemble can create it.
    hand = tmp_path / assemble.INDEX_NAME
    if hand.is_file():
        hand.unlink()

    expected = {name: fx["source_name"] for name in fx["names"].values()}

    # Incomplete official map → refuse; do not write an index over a hole.
    hole = dict(expected)
    hole["model.layers.0.missing.weight"] = fx["source_name"]
    refused = assemble.assemble(
        tmp_path,
        expected_weight_map=hole,
        measurement_path=tmp_path / "MEASUREMENT.json",
    )
    assert refused["action"] == "REFUSED"
    assert refused["reason"] == "activation-aware coverage is incomplete"
    assert not (tmp_path / assemble.INDEX_NAME).exists()

    # Tokenizer install is a pinned-identity hardlink; stub it so the test does
    # not depend on the host's General-R0 tree (still exercised on the live path).
    monkeypatch.setattr(
        assemble,
        "_install_tokenizer",
        lambda _src, _artifact: {
            "tokenizer.json": {
                "sha256": "0" * 64,
                "bytes": 1,
                "method": "test_stub",
            }
        },
    )
    # Architecture synthesis needs sealed contract+config; keep real if present,
    # otherwise pin a minimal header the index will carry.
    monkeypatch.setattr(
        assemble,
        "synthesize_architecture",
        lambda: {
            "model_type": "glm_moe_dsa",
            "hidden_size": 3,
            "num_hidden_layers": 1,
            "n_routed_experts": 1,
            "vocab_size": 8,
        },
    )

    receipt = assemble.assemble(
        tmp_path,
        expected_weight_map=expected,
        measurement_path=tmp_path / "MEASUREMENT.json",
    )
    assert receipt["action"] == "ASSEMBLED"
    assert receipt["model_bytes_copied"] == 0
    assert receipt["shards_hashed"] == 1
    assert receipt["tensors"] == 3
    assert receipt["coverage"] == "COMPLETE"

    index_path = tmp_path / assemble.INDEX_NAME
    assert index_path.is_file()
    doc = json.loads(index_path.read_text())
    assert doc["schema"] == "hawking.activation_aware.model_index.v1"
    assert doc["tensor_count"] == 3
    assert doc["shard_count"] == 1
    assert fx["shard_name"] in doc["shard_sha256"]
    assert len(doc["shard_sha256"][fx["shard_name"]]) == 64
    for name in fx["names"].values():
        assert doc["weight_map"][name] == fx["shard_name"]
        assert doc["tensor_dtypes"][name] == "BF16"

    # Exactly the flagship-oracle open path: index schema + hash-bound tensor fetch.
    source = ActivationAwareGlmSource(tmp_path, index_json=index_path, verify_hash=True)
    native = source.tensor(fx["names"]["native"])
    np.testing.assert_array_equal(native, fx["native"])
    input_weight = fx["input_left"] @ fx["basis"].T
    np.testing.assert_allclose(
        source.tensor(fx["names"]["input"]), input_weight, rtol=0, atol=2e-3
    )
    assert (tmp_path / assemble.RECEIPT_NAME).is_file()


def test_format_rejects_overlapping_payload_spans(tmp_path):
    fx = _fixture(tmp_path)
    raw = fx["shard"].read_bytes()
    index_len = struct.unpack("<Q", raw[:8])[0]
    index = json.loads(raw[8 : 8 + index_len])
    index["tensors"][1]["offset"] = index["tensors"][0]["offset"]
    encoded = json.dumps(index, sort_keys=True, separators=(",", ":")).encode()
    fx["shard"].write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + raw[8 + index_len :]
    )
    with pytest.raises(fmt.ActivationAwareFormatError, match="non-contiguous"):
        fmt.read_index(fx["shard"])
