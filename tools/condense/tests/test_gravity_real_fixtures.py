#!/usr/bin/env python3.12
"""Real-artifact fixtures: the safety gate, the labelling, and the completeness logic.

The load-bearing risk is not a wrong number, it is a write into a live campaign's output
directory or a read of a shard the packer has not finished.  Those are what these tests
watch.  Everything runs against synthetic shards in a temp directory; the live artifact
directory is only touched by the one test that is skipped when it is absent.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np
import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_pack  # noqa: E402
import gravity_forge as forge  # noqa: E402
import gravity_format  # noqa: E402
import gravity_real_fixtures as fx  # noqa: E402

OLD = 4 * 3600


def _payload(name: str, rng, rows: int = 64, cols: int = 128):
    weights = rng.standard_normal((rows, cols)).astype(np.float32)
    artifact = forge.pack_product_quant(weights, dim=8, subspaces=1, k=128, seed=0, iters=2)
    return ({"name": name, "category": "routed_expert", "shape": [rows, cols],
             "elements": weights.size, "rung": "R0",
             "bpw": artifact.whole_artifact_bpw}, glm52_pack.serialize(artifact))


def _write(root: pathlib.Path, filename: str, names: list[str], *, age: float = OLD):
    rng = np.random.default_rng(len(names))
    payloads = [_payload(name, rng) for name in names]
    path = root / filename
    gravity_format.write_shard(
        path, payloads, model={"repo": "zai-org/GLM-5.2", "revision": "x"},
        compression={"codec": "gravity-pq", "production_rung": "R0",
                     "packed_bpw": (sum(len(b) for _, b in payloads) * 8
                                    / sum(d["elements"] for d, _ in payloads))})
    os.utime(path, (0, time.time() - age))
    return path


def _expert_names(layer: int, experts: range) -> list[str]:
    return [f"model.layers.{layer}.mlp.experts.{e}.{p}_proj.weight"
            for e in experts for p in fx.PROJECTIONS]


@pytest.fixture()
def shard(tmp_path):
    return _write(tmp_path, "model-00001-of-00002.gravity", _expert_names(7, range(2)))


# ------------------------------------------------------------------ safety age

def test_safety_age_filters_young_shards(tmp_path):
    old = _write(tmp_path, "model-00001-of-00002.gravity", _expert_names(7, range(1)))
    young = _write(tmp_path, "model-00002-of-00002.gravity", _expert_names(7, range(1)),
                   age=60)

    assert fx.is_safe(old) and not fx.is_safe(young)
    assert fx.safe_shards(tmp_path) == [old]

    rows = fx.survey(tmp_path)
    assert rows["shards_total"] == 2 and rows["shards_safe"] == 1
    assert rows["shards_in_flight"] == 1
    # the young shard is enumerated but never opened, so it carries no header fields
    young_row = next(r for r in rows["shards"] if r["shard"] == young.name)
    assert "tensor_count" not in young_row and young_row["safe"] is False


def test_safety_age_is_configurable(tmp_path):
    path = _write(tmp_path, "model-00001-of-00001.gravity", _expert_names(7, range(1)),
                  age=1800)
    assert not fx.is_safe(path)
    assert fx.is_safe(path, min_age=600)
    assert fx.safe_shards(tmp_path, min_age=600) == [path]


def test_in_flight_shard_is_refused_before_its_body_is_read(tmp_path):
    path = _write(tmp_path, "model-00001-of-00001.gravity", _expert_names(7, range(1)),
                  age=5)
    with pytest.raises(fx.FixtureError, match="safety age"):
        fx.verified(path)
    with pytest.raises(fx.FixtureError, match="safety age"):
        fx.load_codes(path, _expert_names(7, range(1))[0])


def test_tmp_files_are_never_enumerated(tmp_path):
    path = _write(tmp_path, "model-00001-of-00001.gravity", _expert_names(7, range(1)))
    partial = tmp_path / "model-00009-of-00282.gravity.tmp"
    partial.write_bytes(path.read_bytes()[:1024])
    os.utime(partial, (0, time.time() - OLD))

    assert fx.shard_paths(tmp_path) == [path]
    assert partial not in fx.safe_shards(tmp_path)
    assert not fx.is_safe(partial)


def test_corrupt_shard_is_refused(tmp_path):
    path = _write(tmp_path, "model-00001-of-00001.gravity", _expert_names(7, range(1)))
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    os.utime(path, (0, time.time() - OLD))
    with pytest.raises(fx.FixtureError, match="integrity"):
        fx.verified(path)


def test_verification_is_memoized_per_shard(tmp_path, monkeypatch):
    path = _write(tmp_path, "model-00001-of-00001.gravity", _expert_names(7, range(2)))
    fx.verified(path)  # prime

    calls = []
    monkeypatch.setattr(gravity_format, "verify",
                        lambda p: calls.append(p) or {"ok": True})
    for name in _expert_names(7, range(2)):
        fx.load_codes(path, name)
    assert calls == [], "the body was re-hashed per tensor"


def test_module_never_writes_to_the_artifact_directory():
    """Structural, not behavioural: the library half may not contain a mutating call.

    Strings and comments are stripped first, so the docstring that describes the rule
    cannot be mistaken for a violation of it.  The selftest (temp dirs) and the CLI report
    writer (the repo's own reports/ tree) are excluded by construction: neither can name a
    path under the artifact root.
    """
    import io
    import tokenize

    source = (CONDENSE / "gravity_real_fixtures.py").read_text().split("def selftest", 1)[0]
    code = " ".join(
        token.string for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type not in (tokenize.STRING, tokenize.COMMENT))
    for forbidden in ("shutil", "unlink", "rename", "replace", "write_bytes",
                      "write_text", "remove", "utime", "mkdir", "open"):
        assert forbidden not in code, f"{forbidden} appears in the read-only library half"


# ------------------------------------------------------------------ cache key

def test_cache_key_is_content_addressed_and_stable(shard):
    name = "model.layers.7.mlp.experts.0.gate_proj.weight"
    codes, descriptor = fx.load_codes(shard, name)
    key = fx.cache_key(shard, descriptor)

    assert descriptor["sha256"] in key and name in key and shard.name in key
    assert fx.cache_key(shard, fx.descriptor_of(shard, name)) == key
    # a second load hits the cache under the same key, so it is the same object
    assert fx.load_codes(shard, name)[0] is codes


def test_cache_key_separates_distinct_tensors(shard):
    keys = {fx.cache_key(shard, fx.descriptor_of(shard, n))
            for n in _expert_names(7, range(2))}
    assert len(keys) == 6, "two tensors collided on one cache key"


def test_cache_key_survives_a_reload(shard):
    name = "model.layers.7.mlp.experts.1.down_proj.weight"
    first = fx.cache_key(shard, fx.descriptor_of(shard, name))
    fx._VERDICTS.clear()
    assert fx.cache_key(shard, fx.descriptor_of(shard, name)) == first


# ------------------------------------------------------------------ labelling

def test_fixture_carries_a_synthetic_activation_label(shard):
    fixture = fx._fixture(shard, "model.layers.7.mlp.experts.0.up_proj.weight")
    assert fixture.activation_source == fx.SYNTHETIC
    assert fixture.as_json()["activation_source"] == fx.SYNTHETIC
    assert fixture.activation().shape == (fixture.shape[1],)


def test_an_unlabelled_activation_source_is_rejected(shard):
    descriptor = fx.descriptor_of(shard, "model.layers.7.mlp.experts.0.up_proj.weight")
    with pytest.raises(fx.FixtureError, match="activation_source"):
        fx.Fixture(shard=shard.name, shard_path=str(shard), tensor=descriptor["name"],
                   sha256=descriptor["sha256"], descriptor=descriptor,
                   activation_source="probably real", activation_provenance="?",
                   cache_key="k")


def test_real_activations_are_reported_unavailable_not_invented():
    status = fx.teacher_activation_status()
    assert status["status"] == "UNAVAILABLE"
    assert "capsules" in status["path"]
    assert status["consequence"].endswith("SYNTHETIC")


def test_synthetic_activation_is_deterministic():
    assert np.array_equal(fx.synthetic_activation(64, seed=3),
                          fx.synthetic_activation(64, seed=3))
    assert not np.array_equal(fx.synthetic_activation(64, seed=3),
                              fx.synthetic_activation(64, seed=4))


# ------------------------------------------------------------------ layer completeness

def test_layer_completeness_counts_only_full_triples(tmp_path):
    names = _expert_names(9, range(8))
    names.append("model.layers.9.mlp.experts.8.gate_proj.weight")  # deliberately partial
    names += [f"model.layers.9.mlp.shared_experts.{p}_proj.weight" for p in fx.PROJECTIONS]
    _write(tmp_path, "model-00001-of-00001.gravity", names)

    index = fx.layer_index(tmp_path)
    entry = index[9]
    assert entry["complete_experts"] == list(range(8))
    assert entry["partial_experts"] == [8]
    assert entry["shared_expert_complete"] is True
    assert entry["moe_layer_executable"] is True
    assert entry["all_256_experts"] is False
    # the router is protected at source precision, so it is not in any .gravity payload
    assert entry["router_present"] is False
    assert entry["routing_executable"] is False
    assert fx.executable_layers(index) == [9]


def test_a_layer_without_its_shared_expert_is_not_executable(tmp_path):
    _write(tmp_path, "model-00001-of-00001.gravity", _expert_names(11, range(8)))
    index = fx.layer_index(tmp_path)
    assert index[11]["complete_expert_count"] == 8
    assert index[11]["shared_expert_complete"] is False
    assert index[11]["moe_layer_executable"] is False
    assert fx.executable_layers(index) == []


def test_experts_split_across_shards_are_still_complete(tmp_path):
    _write(tmp_path, "model-00001-of-00002.gravity",
           [f"model.layers.5.mlp.experts.0.{p}_proj.weight" for p in ("gate", "up")])
    _write(tmp_path, "model-00002-of-00002.gravity",
           ["model.layers.5.mlp.experts.0.down_proj.weight"])
    entry = fx.layer_index(tmp_path)[5]
    assert entry["complete_experts"] == [0]
    assert len(entry["shards"]) == 2


def test_an_in_flight_shard_cannot_complete_a_layer(tmp_path):
    _write(tmp_path, "model-00001-of-00002.gravity",
           [f"model.layers.5.mlp.experts.0.{p}_proj.weight" for p in ("gate", "up")])
    _write(tmp_path, "model-00002-of-00002.gravity",
           ["model.layers.5.mlp.experts.0.down_proj.weight"], age=30)
    entry = fx.layer_index(tmp_path)[5]
    assert entry["complete_experts"] == []
    assert entry["partial_experts"] == [0]


def test_fixture_set_needs_eight_experts_and_a_shared_expert(tmp_path):
    names = _expert_names(13, range(8))
    names += [f"model.layers.13.mlp.shared_experts.{p}_proj.weight" for p in fx.PROJECTIONS]
    names.append("model.layers.13.self_attn.o_proj.weight")
    _write(tmp_path, "model-00001-of-00001.gravity", names)

    fixtures = fx.fixture_set(tmp_path)
    assert fixtures["layer"] == 13
    assert set(fixtures["one_expert"]) == set(fx.PROJECTIONS)
    assert len(fixtures["expert_set"]) == 8
    assert fixtures["attention"].tensor.endswith("o_proj.weight")
    assert fixtures["router_present"] is False

    rendered = fx.manifest(fixtures)
    assert all(f["activation_source"] == fx.SYNTHETIC
               for f in rendered["one_expert"].values())
    assert rendered["one_expert"]["gate"]["sha256"]


def test_fixture_set_refuses_when_no_layer_qualifies(tmp_path):
    _write(tmp_path, "model-00001-of-00001.gravity", _expert_names(4, range(2)))
    with pytest.raises(fx.FixtureError, match="8 complete experts"):
        fx.fixture_set(tmp_path)


# ------------------------------------------------------------------ index distribution

def test_index_distribution_measures_the_real_histogram(shard):
    codes, _ = fx.load_codes(shard, "model.layers.7.mlp.experts.0.gate_proj.weight")
    stats = fx.index_distribution(codes)
    assert stats["codewords"] == 128
    assert stats["indices"] == codes["indices"].size
    assert sum(stats["histogram"]) == stats["indices"]
    assert 0.0 <= stats["entropy_ratio"] <= 1.0
    assert stats["verdict"] in ("NEAR_UNIFORM", "SKEWED")


def test_index_distribution_calls_a_degenerate_stream_skewed():
    codes = {"indices": np.zeros((64, 1), dtype=np.int64),
             "codebooks": [np.zeros((128, 8), dtype=np.float32)]}
    stats = fx.index_distribution(codes)
    assert stats["verdict"] == "SKEWED"
    assert stats["entropy_bits"] == 0.0
    assert stats["unused_codewords"] == 127
    assert stats["codewords_covering_0.5_mass"] == 1


# ------------------------------------------------------------------ live artifacts

@pytest.mark.skipif(not fx.ARTIFACT_DIR.is_dir(),
                    reason=f"no artifact directory at {fx.ARTIFACT_DIR}")
def test_live_artifact_directory_survey_is_read_only_and_sane():
    before = {p.name: p.stat().st_mtime for p in fx.ARTIFACT_DIR.iterdir()}
    rows = fx.survey(fx.ARTIFACT_DIR)
    assert rows["shards_total"] >= rows["shards_safe"]
    for row in rows["shards"]:
        if row["safe"]:
            assert row["production_rung"] == "R0"
            assert row["packed_bpw"] < 1.0
    after = {p.name: p.stat().st_mtime for p in fx.ARTIFACT_DIR.iterdir()}
    assert before == after, "the survey changed the live artifact directory"


def test_survey_of_a_missing_directory_is_empty_not_an_error(tmp_path):
    rows = fx.survey(tmp_path / "absent")
    assert rows["shards_total"] == 0 and rows["shards_safe"] == 0
    assert fx.layer_index(tmp_path / "absent") == {}
