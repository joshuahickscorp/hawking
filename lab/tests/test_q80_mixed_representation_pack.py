"""Codec + catalog + identity tests for the Q80 mixed packer. No source shards."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from lab.operators.ascension_dual_gravity_worker import (
    ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
    MAGIC_ACT_SVD,
    _parse_container,
)
from lab.operators.q80_mixed_representation_pack import (
    CATALOG_MAGIC,
    CODEC_BINARY,
    CODEC_HGRAVS01,
    CODEC_RESIDUAL,
    CODEC_UNIFORM8,
    F_EXPERT,
    F_NONEXPERT,
    ORGAN_DOWN,
    ORGAN_GATE,
    ORGAN_NONEXPERT,
    ORGAN_UP,
    RANK,
    classify,
    cosine_flat,
    encode_down_activation,
    encode_down_weight_space,
    encode_gate,
    encode_uniform8,
    encode_up,
    execution_order,
    is_gqa_layer,
    post_swiglu,
    read_catalog,
    write_catalog,
)


FAKE_IDENTITY = {
    "path": "test-capture",
    "capture_result_path": "test-capture/capture-result.json",
    "sha256": "17a1e9b60a53cc491601a549880c2d215ff16395ee36abaa05fb95eb7fe2aabe",
    "schema": "test",
    "status": "test",
    "fit_kind": "real_routed_activation_capture",
    "not_synthetic_unit_direction": True,
}


def test_classify_recipe_families() -> None:
    assert classify("model.layers.0.mlp.experts.3.gate_proj.weight") == (
        ORGAN_GATE,
        CODEC_BINARY,
    )
    assert classify("model.layers.10.mlp.experts.453.up_proj.weight") == (
        ORGAN_UP,
        CODEC_RESIDUAL,
    )
    assert classify("model.layers.1.mlp.experts.265.down_proj.weight") == (
        ORGAN_DOWN,
        CODEC_HGRAVS01,
    )
    assert classify("model.layers.0.mlp.shared_expert.down_proj.weight") == (
        ORGAN_NONEXPERT,
        CODEC_UNIFORM8,
    )
    assert classify("model.embed_tokens.weight") == (ORGAN_NONEXPERT, CODEC_UNIFORM8)
    assert classify("lm_head.weight") == (ORGAN_NONEXPERT, CODEC_UNIFORM8)
    assert classify("model.layers.3.self_attn.q_proj.weight") == (
        ORGAN_NONEXPERT,
        CODEC_UNIFORM8,
    )


def test_gqa_is_every_fourth_layer_starting_at_3() -> None:
    gqa = [i for i in range(48) if is_gqa_layer(i)]
    assert gqa == list(range(3, 48, 4))
    assert len(gqa) == 12


def test_execution_order_is_a_permutation_and_embed_is_first() -> None:
    names = [
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.linear_attn.out_proj.weight",
        "model.layers.0.linear_attn.in_proj_qkvz.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.layers.0.mlp.shared_expert.up_proj.weight",
        "model.layers.0.mlp.shared_expert.down_proj.weight",
        "model.layers.0.mlp.shared_expert_gate.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.input_layernorm.weight",
        "model.layers.3.post_attention_layernorm.weight",
        "model.layers.3.mlp.gate.weight",
        "model.layers.3.mlp.shared_expert.gate_proj.weight",
        "model.layers.3.mlp.shared_expert.up_proj.weight",
        "model.layers.3.mlp.shared_expert.down_proj.weight",
        "model.layers.3.mlp.shared_expert_gate.weight",
        "model.layers.3.mlp.experts.1.gate_proj.weight",
        "model.layers.3.mlp.experts.1.up_proj.weight",
        "model.layers.3.mlp.experts.1.down_proj.weight",
    ]
    # execution_order walks every expert 0..511; feed the full set it will take.
    extra = []
    for layer in range(48):
        for expert in range(512):
            for comp in ("gate", "up", "down"):
                extra.append(
                    f"model.layers.{layer}.mlp.experts.{expert}.{comp}_proj.weight"
                )
    full = list(dict.fromkeys(names + extra))
    ordered = execution_order(full)
    assert ordered[0] == "model.embed_tokens.weight"
    assert ordered[-1] == "lm_head.weight"
    assert ordered[-2] == "model.norm.weight"
    assert set(ordered) == set(full)
    assert len(ordered) == len(full)
    # gate then up then down inside one expert
    i = ordered.index("model.layers.0.mlp.experts.0.gate_proj.weight")
    assert ordered[i + 1].endswith("up_proj.weight")
    assert ordered[i + 2].endswith("down_proj.weight")


def test_binary_roundtrip_from_physical_bytes() -> None:
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 64), dtype=np.float32)
    payload, decoded = encode_gate(w)
    assert payload[:8] == b"HGRAVB01"
    assert decoded.shape == w.shape
    assert cosine_flat(w, decoded) > 0.7


def test_rice_q1_residual_roundtrip_from_physical_bytes() -> None:
    rng = np.random.default_rng(1)
    w = rng.standard_normal((32, 64), dtype=np.float32)
    payload, decoded = encode_up(w)
    assert payload[:8] == b"HGRAVR02"
    header, _ = _parse_container(payload, expected_magic=b"HGRAVR02")
    assert header["index_mode"] == "rice"
    assert header["value_bits"] == 1
    assert header["value_scale"] == "rms"
    assert header["outlier_ratio_requested"] == 0.02
    assert decoded.shape == w.shape


def test_uniform8_roundtrip_from_physical_bytes() -> None:
    rng = np.random.default_rng(2)
    w = rng.standard_normal((16, 64), dtype=np.float32)
    payload, decoded = encode_uniform8(w)
    assert payload[:8] == b"HGRAVU01"
    header, _ = _parse_container(payload, expected_magic=b"HGRAVU01")
    assert header["bits"] == 8
    assert header["group_size"] == 64
    assert cosine_flat(w, decoded) > 0.99


def test_hgravs01_activation_weighted_keeps_requested_rank() -> None:
    rng = np.random.default_rng(3)
    w = rng.standard_normal((64, 32), dtype=np.float32)
    x = rng.standard_normal((40, 32), dtype=np.float32)
    payload, decoded, meta = encode_down_activation(w, x, FAKE_IDENTITY)
    assert payload[:8] == MAGIC_ACT_SVD
    header, _ = _parse_container(payload, expected_magic=MAGIC_ACT_SVD)
    assert header["representation"] == ACTIVATION_WEIGHTED_SVD_REPRESENTATION
    assert header["activation_capture"]["fit_kind"] == "real_routed_activation_capture"
    # 40 rows < 160, but rank must stay min(160, 64, 32) = 32, not 40.
    assert int(header["rank"]) == 32
    assert decoded.shape == w.shape
    assert "rank_clamped_to_n_fit" not in header.get("fit", {}) or header["fit"].get(
        "rank"
    ) == 32


def test_hgravs01_weight_space_never_routed() -> None:
    rng = np.random.default_rng(4)
    w = rng.standard_normal((64, 32), dtype=np.float32)
    payload, decoded, meta = encode_down_weight_space(
        w, FAKE_IDENTITY, reason="never_routed"
    )
    header, _ = _parse_container(payload, expected_magic=MAGIC_ACT_SVD)
    assert header["fit"]["fit"] == "weight_space_truncated_svd"
    assert header["rank"] == 32
    assert header["activation_capture"]["fit_kind"] == "real_routed_activation_capture"
    assert decoded.shape == w.shape


def test_post_swiglu_geometry() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((7, 2048), dtype=np.float32)
    g = rng.standard_normal((512, 2048), dtype=np.float32)
    u = rng.standard_normal((512, 2048), dtype=np.float32)
    y = post_swiglu(x, g, u)
    assert y.shape == (7, 512)
    assert np.isfinite(y).all()


def test_identity_arithmetic_matches_receipt() -> None:
    mixed_expert = 1.22957
    complete_8 = F_EXPERT * mixed_expert + F_NONEXPERT * 8.0
    assert abs(complete_8 - 1.43051) < 5e-5


def test_catalog_roundtrip(tmp_path: Path) -> None:
    records = [
        {
            "name": "model.embed_tokens.weight",
            "codec": CODEC_UNIFORM8,
            "organ": ORGAN_NONEXPERT,
            "shape": [4, 8],
            "elements": 32,
            "segment_id": 0,
            "achieved_rank": 0,
            "offset": 0,
            "nbytes": 16,
            "sha256": "ab" * 32,
            "flags": 1,
            "n_fit_rows": 0,
            "codec_bpw": 8.25,
        },
        {
            "name": "model.layers.0.mlp.experts.0.down_proj.weight",
            "codec": CODEC_HGRAVS01,
            "organ": ORGAN_DOWN,
            "shape": [2048, 512],
            "elements": 1048576,
            "segment_id": 1,
            "achieved_rank": RANK,
            "offset": 0,
            "nbytes": 200,
            "sha256": "cd" * 32,
            "flags": 8,
            "n_fit_rows": 200,
            "codec_bpw": 1.27,
        },
    ]
    segments = [
        {
            "id": 0,
            "filename": "00_embed.hq80seg",
            "bytes": 16,
            "sha256": "11" * 32,
        },
        {
            "id": 1,
            "filename": "L00.hq80seg",
            "bytes": 200,
            "sha256": "22" * 32,
        },
    ]
    path = tmp_path / "catalog.hq80m15"
    blob = write_catalog(path, records, segments)
    assert blob[:8] == CATALOG_MAGIC
    got = read_catalog(path)
    assert got["records"][0]["name"] == "model.embed_tokens.weight"
    assert got["records"][1]["achieved_rank"] == RANK
    assert got["records"][1]["n_fit_rows"] == 200
    assert got["segments"][1]["filename"] == "L00.hq80seg"
    version, n_tensors, n_segments, _, name_blob_bytes, _ = struct.unpack_from(
        "<IIIIII", blob, 8
    )
    assert version == 1
    assert n_tensors == 2
    assert n_segments == 2
    assert name_blob_bytes > 0
