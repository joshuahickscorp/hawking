"""What must stay true of the recognizer: it does not force-fit, and it stays calibrated."""
import json, re, struct, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import arch_recognizer as ar

REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts/headless/ARCHITECTURE_RECOGNIZER.json"
FIX = REPO / "receipts/headless/ARCHITECTURE_RECOGNIZER_FIXTURES.json"


def _names(repo):
    return json.load(open(FIX))["fixtures"][repo]["tensor_names"]


def test_scrambled_names_yield_no_known_organ():
    """A force-fitting recognizer would still find organs in nonsense."""
    src = json.load(open(FIX))["fixtures"]["Qwen/Qwen3-30B-A3B"]
    scrambled = [re.sub(r"[A-Za-z_]+", lambda m: "zzz" + str(len(m.group(0))), n)
                 for n in src["tensor_names"]]
    r = ar.recognize("CONTROL", "n/a", src["config"], scrambled)
    assert [o for o in r["organs"] if o["status"] == "KNOWN"] == []


def test_alias_robustness():
    """`feed_forward.gate_proj` and `mlp.gate_proj` are the same organ."""
    falcon = ar.recognize("t", "r", {"model_type": "falcon_h1"},
                          _names("tiiuae/Falcon-H1-7B-Instruct"))
    assert "mlp_gate_up" in {o["organ"] for o in falcon["organs"]}


def test_latent_attention_is_not_reported_as_plain_gqa():
    kimi = ar.recognize("t", "r", {"model_type": "kimi_vl"},
                        _names("moonshotai/Kimi-VL-A3B-Instruct"))
    got = {o["organ"] for o in kimi["organs"]}
    assert "latent_attention" in got and "gqa_attention" not in got
    assert kimi["folded_organ"]["from"] == "gqa_attention"


def test_heldout_calibration_holds():
    d = json.load(open(RECEIPT))
    h = d["calibration_heldout"]
    assert h and h["calibrated"], h
    assert h["precision"] is not None and h["recall"] is not None
    assert d["pass"] is True
    assert len(d["heldout_specimens"]) >= 2


def test_no_weights_were_loaded():
    d = json.load(open(RECEIPT))
    assert d["did_not_load_weights"] is True
    assert all(s["result"]["loaded_weights"] is False
               for s in d["specimens"] + d["heldout_specimens"])


def test_local_single_shard_snapshot_reads_header_only(tmp_path):
    """Single-shard bodies have no Hub index but still must be recognizable."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "lfm2_moe"}))
    header = {
        "model.layers.2.feed_forward.experts.0.w1.weight": {
            "dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]
        },
        "model.layers.2.feed_forward.experts.0.w2.weight": {
            "dtype": "BF16", "shape": [2, 2], "data_offsets": [8, 16]
        },
        "model.layers.2.feed_forward.experts.0.w3.weight": {
            "dtype": "BF16", "shape": [2, 2], "data_offsets": [16, 24]
        },
        "model.layers.2.feed_forward.gate.weight": {
            "dtype": "BF16", "shape": [2, 2], "data_offsets": [24, 32]
        },
        "__metadata__": {"format": "pt"},
    }
    raw = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)

    cfg, names = ar.local_snapshot(tmp_path)
    result = ar.recognize("local", "local", cfg, names)
    got = {row["organ"] for row in result["organs"]}
    assert result["loaded_weights"] is False
    assert result["n_tensors"] == 4
    assert {"moe_expert", "moe_router"} <= got


def test_static_trait_projection_binds_config_and_header_identity_without_claiming_runtime():
    cfg = {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "ple_layer_ids": [2],
            "ngram_size": 3,
            "hc_count": 4,
            "use_cache": True,
            "vocab_size": 248320,
            "bos_token_id": 248044,
        },
    }
    names = [
        "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_0.weight",
        "model.language_model.layers.2.ple.proj.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
    ]
    result = ar.recognize("fixture/flash", "fixture-revision", cfg, names)
    traits = result["static_traits"]
    assert traits["schema"] == "hawking.gravity.static_organ_traits.v1"
    assert traits["identity"]["artifact_id"] == "fixture/flash"
    assert traits["identity"]["revision"] == "fixture-revision"
    assert {"flash_qwen4_exp", "hashed_ngram", "sharded_lookup", "stateful_decoder"} <= set(
        traits["architecture_tags"]
    )
    assert traits["static_only"] is True
    assert traits["loaded_weights"] is False
    assert "native execution, capability, latency, or TPS" in traits["does_not_establish"]


def test_static_trait_projection_binds_an_opt_in_payload_free_shape_witness():
    cfg = {"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}}
    name = "model.language_model.embed_tokens.weight"
    descriptor = {
        "name": name,
        "shard": "model-00001-of-00001.safetensors",
        "dtype": "BF16",
        "shape": [248320, 2560],
        "data_offsets": [0, 1271398400],
    }
    headers = {
        "schema": ar.SELECTED_TENSOR_HEADERS_SCHEMA,
        "index_name": "model.safetensors.index.json",
        "index_sha256": "a" * 64,
        "headers": [{
            **descriptor,
            "header_descriptor_sha256": ar._canonical_sha256(descriptor),
        }],
        "header_bytes_read": 512,
        "tensor_payload_bytes_read": 0,
        "loaded_weights": False,
        "source_classification": "INDEX_BOUND_SELECTED_SAFETENSORS_HEADERS_ONLY",
    }
    traits = ar.static_trait_projection(
        "fixture/flash", "fixture-revision", cfg, [name], [], tensor_headers=headers
    )
    assert traits["selected_tensor_headers"] == headers
    assert len(traits["identity"]["tensor_header_manifest_sha256"]) == 64
    assert traits["identity"]["tensor_header_index_sha256"] == "a" * 64
    assert traits["loaded_weights"] is False
    assert traits["selected_tensor_headers"]["tensor_payload_bytes_read"] == 0


def test_static_trait_projection_refuses_an_unbound_shape_witness():
    name = "model.language_model.embed_tokens.weight"
    descriptor = {
        "name": name,
        "shard": "model-00001-of-00001.safetensors",
        "dtype": "BF16",
        "shape": [2, 2],
        "data_offsets": [0, 8],
    }
    headers = {
        "schema": ar.SELECTED_TENSOR_HEADERS_SCHEMA,
        "index_name": "model.safetensors.index.json",
        "index_sha256": "a" * 64,
        "headers": [{**descriptor, "header_descriptor_sha256": "0" * 64}],
        "header_bytes_read": 512,
        "tensor_payload_bytes_read": 0,
        "loaded_weights": False,
        "source_classification": "INDEX_BOUND_SELECTED_SAFETENSORS_HEADERS_ONLY",
    }
    with pytest.raises(ValueError, match="header descriptor hash does not bind"):
        ar.static_trait_projection(
            "fixture/flash", "fixture-revision", {}, [name], [], tensor_headers=headers
        )
