#!/usr/bin/env python3.12
"""Round-trip proof: activation-aware packer writer → format/source reader.

The live pack writes ``.aap`` shards; the oracle reads them through
``activation_aware_format`` and ``glm52_activation_aware_source``.  Those paths
had never met on a real payload.  This test synthesises every disposition the
allocator can emit, writes with the packer's real functions, and asserts the
reader recovers shape / values (and rejects corruption).

Codec correctness only — reconstruction quality is not on trial here.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import sys

import numpy as np
import pytest

CONDENSE = pathlib.Path(__file__).resolve().parent
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import activation_aware_format as fmt  # noqa: E402
import glm52_activation_aware_pack as pack  # noqa: E402
from glm52_activation_aware_source import (  # noqa: E402
    ActivationAwareGlmSource,
    _decode_basis,
    _decode_pass_through,
)


# Allocator dispositions (see allocate() in glm52_activation_aware_pack.py):
#   * activation_aware — rank-k factored when some rank beats the constant-mean null
#   * pass_through    — 1-D / non-projectable, OR forced native when no rank beats null
ALLOCATOR_DISPOSITIONS = frozenset({"activation_aware", "pass_through"})

# Tiny activation width so the test stays pure-CPU and well under a few hundred MB.
TEST_HIDDEN = 32
TEST_LAYER = 3
TEST_RANKS = (4, 8)


def _float32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.asarray(arr, dtype=np.float32)
    return (f32.view(np.uint32) >> np.uint32(16)).astype("<u2")


def _bf16_u16_to_float32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _write_bf16_safetensors(path: pathlib.Path, tensors: dict[str, np.ndarray]) -> None:
    """Write a minimal safetensors file the packer can stream (BF16 bodies)."""
    header: dict[str, object] = {}
    bodies: list[bytes] = []
    offset = 0
    for name, arr in sorted(tensors.items()):
        f32 = np.ascontiguousarray(arr, dtype=np.float32)
        raw = _float32_to_bf16_u16(f32).tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(f32.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        bodies.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for blob in bodies:
            fh.write(blob)


def _write_capsule(path: pathlib.Path, layer: int, hidden: int, n_rows: int = 256) -> str:
    """Teacher-capsule NPZ with pre_router_hidden activations of width `hidden`."""
    rng = np.random.default_rng(0xA17A7E ^ (layer * 0x9E3779B1))
    # Structured activations so a low-rank projection beats the constant-mean null.
    dirs = rng.standard_normal((hidden, 8)).astype(np.float32)
    dirs, _ = np.linalg.qr(dirs)
    coeffs = rng.standard_normal((n_rows, 8)).astype(np.float32)
    X = coeffs @ dirs.T + 0.05 * rng.standard_normal((n_rows, hidden)).astype(np.float32)
    key = f"layer_{layer}/pre_router_hidden"
    np.savez(path, **{key: X})
    return key


def _synthetic_basis(
    capsule_path: pathlib.Path,
    capsule_key: str,
    layer: int = TEST_LAYER,
    hidden: int = TEST_HIDDEN,
    max_rank: int = 8,
) -> pack.ActivationBasis:
    prov = pack.BasisProvenance(
        tensor_layer=layer,
        basis_layer=layer,
        capsule_file=capsule_path.name,
        capsule_key=capsule_key,
        hidden=hidden,
        n_activation_rows=0,
    )
    capsule_map = {layer: (capsule_path, capsule_key)}
    return pack.build_basis(prov, capsule_map, max_rank)


def _low_rank_input_weight(basis: pack.ActivationBasis, out_rows: int, rank: int) -> np.ndarray:
    rng = np.random.default_rng(11)
    B = basis.columns(rank)
    L = rng.standard_normal((out_rows, rank)).astype(np.float32)
    return (L @ B.T).astype(np.float32)


def _low_rank_output_weight(basis: pack.ActivationBasis, in_cols: int, rank: int) -> np.ndarray:
    rng = np.random.default_rng(22)
    B = basis.columns(rank)
    L = rng.standard_normal((rank, in_cols)).astype(np.float32)
    return (B @ L).astype(np.float32)


def _enumerate_allocator_dispositions(basis: pack.ActivationBasis) -> dict[str, set[str]]:
    """Run measure+allocate on synthetic tensors and return name→disposition map.

    Enumerates dispositions from the allocator rather than guessing them.
    """
    rng = np.random.default_rng(33)
    cases = {
        # Projectable input-side (cols == hidden).
        "model.layers.3.mlp.experts.0.up_proj.weight": _low_rank_input_weight(
            basis, out_rows=12, rank=4
        ),
        # Projectable output-side (rows == hidden).
        "model.layers.3.mlp.experts.0.down_proj.weight": _low_rank_output_weight(
            basis, in_cols=10, rank=4
        ),
        # 1-D → measure_tensor pass_through.
        "model.layers.3.input_layernorm.weight": rng.standard_normal(
            (TEST_HIDDEN,)
        ).astype(np.float32),
        # Neither side matches activation width → measure_tensor pass_through.
        "model.layers.3.nonprojectable.weight": rng.standard_normal(
            (7, 11)
        ).astype(np.float32),
        # 2-D projectable but structured to fail the null (forced native in allocate).
        # Constant rows: projection cannot beat constant-mean null at any rank.
        "model.layers.3.forced_native.weight": np.broadcast_to(
            rng.standard_normal((1, TEST_HIDDEN)).astype(np.float32),
            (9, TEST_HIDDEN),
        ).copy(),
    }
    measurements = []
    for name, W in cases.items():
        measurements.append(
            pack.measure_tensor(
                name,
                W,
                "BF16",
                basis,
                TEST_RANKS,
                bill_basis_per_tensor=False,
            )
        )
    # Force the constant-row tensor's curve to never beat the null so allocate
    # must emit pass_through via the forced-native branch (not measure's 1-D path).
    forced = next(m for m in measurements if m.name.endswith("forced_native.weight"))
    assert forced.disposition == "activation_aware", (
        "forced_native fixture must reach allocate as activation_aware so the "
        "no-rank-beats-null branch is exercised"
    )
    for point in forced.curve:
        point["beats_null"] = False
        point["surplus_over_null"] = -0.5
        point["mean_row_cosine"] = 0.1
        point["constant_mean_cosine_null"] = 0.9

    alloc = pack.allocate(
        measurements,
        pack.Fraction(99, 100),
        shared_bases=True,
        hidden=TEST_HIDDEN,
    )
    by_name = {row["name"]: row["disposition"] for row in alloc["allocations"]}
    seen = set(by_name.values())
    assert seen <= ALLOCATOR_DISPOSITIONS, f"unknown disposition(s): {seen - ALLOCATOR_DISPOSITIONS}"
    assert "activation_aware" in seen, "allocator produced no activation_aware rows"
    assert "pass_through" in seen, "allocator produced no pass_through rows"
    # Explicitly require the forced-native path (allocate demotes AA → pass_through).
    assert by_name["model.layers.3.forced_native.weight"] == "pass_through"
    return cases, measurements, alloc


def _pack_fixture(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Build synthetic source shard + capsule, allocate, pack with real pack_shard."""
    # Never trip the production 141 GiB floor during a synthetic codec test.
    monkeypatch.setattr(pack, "DISK_FLOOR_BYTES", 0)
    monkeypatch.setattr(pack, "assert_disk_floor", lambda *a, **k: 0)

    capsule_path = tmp_path / f"L{TEST_LAYER:02d}_L{TEST_LAYER:02d}.npz"
    capsule_key = _write_capsule(capsule_path, TEST_LAYER, TEST_HIDDEN)
    basis = _synthetic_basis(capsule_path, capsule_key)

    tensors, measurements, alloc = _enumerate_allocator_dispositions(basis)
    # Store BF16-quantised float32 so pass-through value checks are exact.
    source_f32 = {
        name: _bf16_u16_to_float32(_float32_to_bf16_u16(W)) for name, W in tensors.items()
    }

    shard_name = "model-00001-of-00001.safetensors"
    shard_path = tmp_path / "source" / shard_name
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    _write_bf16_safetensors(shard_path, source_f32)

    out_dir = tmp_path / "artifact"
    out_dir.mkdir()
    allocation_by_name = {row["name"]: row for row in alloc["allocations"]}
    # measure records side/shape/basis_provenance; allocate keeps them on AA rows
    # but pass_through rows from measure need shape for pack.  Re-attach fields
    # pack_shard reads from the allocation dict.
    for m in measurements:
        row = allocation_by_name[m.name]
        row.setdefault("side", m.side)
        row.setdefault("shape", list(m.shape))
        row.setdefault("basis_provenance", m.basis_provenance)
        row.setdefault("dtype", m.dtype)

    capsule_map = {TEST_LAYER: (capsule_path, capsule_key)}
    basis_cache = {TEST_LAYER: basis}
    layer_ranks = {
        int(k): int(v) for k, v in alloc.get("basis_rank_by_layer", {}).items()
    }

    receipt = pack.pack_shard(
        shard_path,
        allocation_by_name,
        capsule_map,
        basis_cache,
        out_dir,
        max_basis_rank=max(TEST_RANKS),
        shared_bases=True,
        layer_basis_ranks=layer_ranks,
    )
    aap_path = pathlib.Path(receipt["artifact"])
    assert aap_path.is_file()

    # Measurement record (dtypes for pass-through decode) + runtime model index.
    meas_doc = {
        "schema": "hawking.glm52.activation_aware_measurement.v1",
        "measurements": [
            {"name": m.name, "dtype": "BF16", "shape": list(m.shape)}
            for m in measurements
        ],
    }
    (out_dir / "MEASUREMENT.json").write_text(json.dumps(meas_doc))

    shard_hash = hashlib.sha256(aap_path.read_bytes()).hexdigest()
    weight_map = {m.name: aap_path.name for m in measurements}
    model_index = {
        "schema": "hawking.activation_aware.model_index.v1",
        "architecture": {"hidden_size": TEST_HIDDEN},
        "shards": [aap_path.name],
        "shard_sha256": {aap_path.name: shard_hash},
        "weight_map": weight_map,
        "tensor_dtypes": {name: "BF16" for name in weight_map},
    }
    (out_dir / "model.activation_aware.index.json").write_text(
        json.dumps(model_index, indent=2, sort_keys=True)
    )

    return {
        "out_dir": out_dir,
        "aap_path": aap_path,
        "source_f32": source_f32,
        "measurements": measurements,
        "alloc": alloc,
        "allocation_by_name": allocation_by_name,
        "basis": basis,
        "model_index": model_index,
        "receipt": receipt,
    }


def test_allocator_dispositions_are_only_activation_aware_and_pass_through(tmp_path, monkeypatch):
    """Enumerate dispositions from allocate() — do not invent new codec branches."""
    monkeypatch.setattr(pack, "DISK_FLOOR_BYTES", 0)
    capsule_path = tmp_path / "L03_L03.npz"
    key = _write_capsule(capsule_path, TEST_LAYER, TEST_HIDDEN)
    basis = _synthetic_basis(capsule_path, key)
    _tensors, _meas, alloc = _enumerate_allocator_dispositions(basis)
    seen = {row["disposition"] for row in alloc["allocations"]}
    assert seen == ALLOCATOR_DISPOSITIONS or seen <= ALLOCATOR_DISPOSITIONS
    assert seen == {"activation_aware", "pass_through"}
    # Both sides of the factored path appear among AA tensors.
    aa = [row for row in alloc["allocations"] if row["disposition"] == "activation_aware"]
    sides = {row["side"] for row in aa}
    assert "input" in sides
    assert "output" in sides


def test_roundtrip_writer_reader_preserves_shape_dtype_and_values(tmp_path, monkeypatch):
    fx = _pack_fixture(tmp_path, monkeypatch)
    aap_path = fx["aap_path"]
    source_f32 = fx["source_f32"]
    allocation_by_name = fx["allocation_by_name"]

    index, body_offset = fmt.read_index(aap_path)
    # Index schema + contiguity contract from the writer.
    assert index["schema"] == pack.SCHEMA == fmt.SCHEMA
    assert index["shared_bases"] is True
    assert isinstance(index["bases"], list) and isinstance(index["tensors"], list)
    fmt.validate_payload_magics(aap_path, index, body_offset)

    # Index round-trip: every tensor the allocator decided is present with same disposition.
    index_by_name = {t["name"]: t for t in index["tensors"]}
    for name, alloc_row in allocation_by_name.items():
        assert name in index_by_name, f"missing packed tensor {name}"
        entry = index_by_name[name]
        assert entry["disposition"] == alloc_row["disposition"]
        assert list(entry["shape"]) == list(source_f32[name].shape)

    # Re-read index from disk bytes (not the in-memory dict) and compare.
    raw = aap_path.read_bytes()
    index_len = struct.unpack_from("<Q", raw, 0)[0]
    index_again = json.loads(raw[8 : 8 + index_len])
    assert index_again["schema"] == index["schema"]
    assert [t["name"] for t in index_again["tensors"]] == [t["name"] for t in index["tensors"]]
    assert [b["basis_layer"] for b in index_again["bases"]] == [
        b["basis_layer"] for b in index["bases"]
    ]

    source = ActivationAwareGlmSource(fx["out_dir"], verify_hash=True)

    for name, entry in index_by_name.items():
        recovered = source.tensor(name)
        expected_shape = tuple(source_f32[name].shape)
        assert recovered.shape == expected_shape, (name, recovered.shape, expected_shape)
        assert recovered.dtype == np.float32

        if entry["disposition"] == "pass_through":
            # Lossless BF16 round-trip at the codec boundary.
            np.testing.assert_array_equal(
                recovered,
                source_f32[name].astype(np.float32),
                err_msg=f"pass_through value mismatch for {name}",
            )
        else:
            # Factored path: reader must rebuild the same W_hat the writer stored
            # (float16 factors), not the original pre-projection matrix.
            shard_view = fmt.ActivationAwareShard(aap_path)
            blob = shard_view.read_tensor(name)
            decoded = pack.deserialize_tensor_payload(blob)
            assert decoded["side"] == entry["side"]
            assert decoded["rows"] == expected_shape[0]
            assert decoded["cols"] == expected_shape[1]
            if decoded["has_basis"]:
                B = decoded["B"]
            else:
                # Shared-basis production path: basis lives once per layer.
                full_B = _decode_basis(shard_view.read_basis(int(decoded["basis_layer"])))
                B = full_B[:, : int(decoded["rank"])]
            w_hat = pack.reconstruct(decoded["L"], B, decoded["side"])
            np.testing.assert_allclose(
                recovered,
                w_hat,
                rtol=0,
                atol=0,
                err_msg=f"activation_aware reader/writer mismatch for {name}",
            )
            # Also: factors themselves round-trip through serialize/deserialize.
            if decoded["has_basis"]:
                reblob = pack.serialize_tensor_payload(
                    decoded["L"],
                    decoded["B"],
                    side=decoded["side"],
                    rows=decoded["rows"],
                    cols=decoded["cols"],
                    rank=decoded["rank"],
                    basis_layer=decoded["basis_layer"],
                    bill_basis=True,
                )
                again = pack.deserialize_tensor_payload(reblob)
                np.testing.assert_array_equal(again["L"], decoded["L"])
                np.testing.assert_array_equal(again["B"], decoded["B"])


def test_bf16_pass_through_widening_is_little_endian_shift(tmp_path, monkeypatch):
    """BF16 codec: u16.astype(uint32) << 16.  Endian/truncation bugs look plausible."""
    monkeypatch.setattr(pack, "DISK_FLOOR_BYTES", 0)
    monkeypatch.setattr(pack, "assert_disk_floor", lambda *a, **k: 0)

    # Distinctive values that explode under wrong endianness or a 16-bit truncate.
    values = np.array(
        [
            1.0,
            -2.0,
            0.5,
            3.14159,
            1e-3,
            1e3,
            np.float32(2.0 ** 10 + 1),  # needs high mantissa bits
        ],
        dtype=np.float32,
    )
    # Exact BF16 quantisation the packer will store.
    u16 = _float32_to_bf16_u16(values)
    expected = _bf16_u16_to_float32(u16)

    # Writer path (identical to pack_shard pass_through branch).
    pthead = struct.pack("<8sIII", b"GLM52PT0", 1, int(values.size), 0)
    pthead = pthead + b"\x00" * (pack.HEADER_BYTES - len(pthead))
    blob = pthead + u16.tobytes()

    recovered = _decode_pass_through(blob, "BF16", [int(values.size)])
    np.testing.assert_array_equal(recovered, expected)

    # Wrong endian load must not silently match.
    be_u16 = u16.astype(">u2")
    wrong = pthead + be_u16.tobytes()
    if be_u16.tobytes() != u16.tobytes():
        wrong_vals = _decode_pass_through(wrong, "BF16", [int(values.size)])
        assert not np.array_equal(wrong_vals, expected), (
            "big-endian u16 payload matched little-endian BF16 decode — "
            "widening path is not discriminating endianness"
        )

    # Full shard path also preserves BF16.
    fx = _pack_fixture(tmp_path, monkeypatch)
    pt_names = [
        n
        for n, row in fx["allocation_by_name"].items()
        if row["disposition"] == "pass_through"
    ]
    assert pt_names, "expected at least one pass_through tensor in the fixture"
    source = ActivationAwareGlmSource(fx["out_dir"], verify_hash=True)
    for name in pt_names:
        np.testing.assert_array_equal(
            source.tensor(name),
            fx["source_f32"][name].astype(np.float32),
            err_msg=f"BF16 pass_through full-path mismatch for {name}",
        )
        # Dtype recorded for the reader is BF16.
        assert fx["model_index"]["tensor_dtypes"][name] == "BF16"


def test_validate_payload_magics_accepts_clean_and_rejects_corrupt(tmp_path, monkeypatch):
    fx = _pack_fixture(tmp_path, monkeypatch)
    aap_path = fx["aap_path"]

    index, body_offset = fmt.read_index(aap_path)
    fmt.validate_payload_magics(aap_path, index, body_offset)  # must not raise

    raw = bytearray(aap_path.read_bytes())
    # Corrupt the first tensor magic (first byte of first tensor span).
    first_tensor = min(index["tensors"], key=lambda t: int(t["offset"]))
    magic_at = body_offset + int(first_tensor["offset"])
    raw[magic_at] ^= 0xFF
    corrupt_path = tmp_path / "corrupt.aap"
    corrupt_path.write_bytes(raw)

    c_index, c_body = fmt.read_index(corrupt_path)
    with pytest.raises(fmt.ActivationAwareFormatError, match="magic"):
        fmt.validate_payload_magics(corrupt_path, c_index, c_body)

    # Corrupted body must not be silently decoded by the runtime source either:
    # hash binding fires first when verify_hash=True.
    bad_dir = tmp_path / "bad_art"
    bad_dir.mkdir()
    bad_shard = bad_dir / aap_path.name
    bad_shard.write_bytes(raw)
    (bad_dir / "model.activation_aware.index.json").write_text(
        (fx["out_dir"] / "model.activation_aware.index.json").read_text()
    )
    bad_source = ActivationAwareGlmSource(bad_dir, verify_hash=True)
    with pytest.raises(fmt.ActivationAwareFormatError, match="SHA-256 mismatch"):
        bad_source.tensor(first_tensor["name"])

    # With hash verification off, a bad magic still fails at deserialize / PT decode.
    bad_source_nohash = ActivationAwareGlmSource(bad_dir, verify_hash=False)
    with pytest.raises((fmt.ActivationAwareFormatError, pack.PackError)):
        bad_source_nohash.tensor(first_tensor["name"])


def test_payload_level_serialize_deserialize_roundtrip_both_sides():
    """Direct factor payload round-trip without a full shard (input + output)."""
    rng = np.random.default_rng(7)
    for side, rows, cols, rank in (
        ("input", 16, TEST_HIDDEN, 4),
        ("output", TEST_HIDDEN, 12, 4),
    ):
        if side == "input":
            L = rng.standard_normal((rows, rank)).astype(np.float32)
            B = rng.standard_normal((cols, rank)).astype(np.float32)
        else:
            L = rng.standard_normal((rank, cols)).astype(np.float32)
            B = rng.standard_normal((rows, rank)).astype(np.float32)
        B, _ = np.linalg.qr(B)

        for bill_basis in (True, False):
            blob = pack.serialize_tensor_payload(
                L,
                B,
                side=side,
                rows=rows,
                cols=cols,
                rank=rank,
                basis_layer=TEST_LAYER,
                bill_basis=bill_basis,
            )
            assert blob[:8] == pack.MAGIC == fmt.TENSOR_MAGIC
            decoded = pack.deserialize_tensor_payload(blob)
            assert decoded["side"] == side
            assert decoded["rows"] == rows and decoded["cols"] == cols
            assert decoded["rank"] == rank
            assert decoded["basis_layer"] == TEST_LAYER
            assert decoded["has_basis"] is bill_basis
            # Float16 store/load — compare at the codec's own quantisation.
            L16 = np.asarray(L, dtype=np.float16).astype(np.float32)
            np.testing.assert_array_equal(decoded["L"], L16)
            if bill_basis:
                B16 = np.asarray(B, dtype=np.float16).astype(np.float32)
                np.testing.assert_array_equal(decoded["B"], B16)
