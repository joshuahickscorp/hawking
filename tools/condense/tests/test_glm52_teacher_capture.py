#!/usr/bin/env python3.12
"""Teacher-capture invariants: the gate that stands between a BF16 body and rm."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_synthetic as synthetic  # noqa: E402
import glm52_teacher_capture as tc  # noqa: E402
from glm52_adapter import CORE  # noqa: E402


def _organ_id(spec) -> str:
    if spec.section != CORE:
        return "mtp_layer"
    if spec.layer is None:
        return "global_input" if "embed" in spec.name else "global_output"
    return f"text_layer_{spec.layer:02d}"


def _graph_from(inventory, config) -> dict:
    tensors = []
    organs: dict[str, set[str]] = {}
    for name, record in inventory.tensors.items():
        organ = _organ_id(record.spec)
        organs.setdefault(organ, set()).add(record.shard)
        tensors.append({
            "name": name,
            "shard": record.shard,
            "absolute_start": record.absolute_start,
            "payload_bytes": record.byte_count,
            "dtype": record.spec.dtype,
            "shape": list(record.spec.shape),
            "alias_of": None,
        })
    return {
        "repo": "synthetic/GLM-5.2-twin",
        "revision": "synthetic-revision",
        "seal_sha256": "synthetic-graph-seal",
        "tensors": tensors,
        "organs": [
            {"organ_id": organ, "source_shards": sorted(shards)}
            for organ, shards in sorted(organs.items())
        ],
        "layers": config["num_hidden_layers"],
    }


def _schedule_for(layers: list[int]) -> dict:
    return {
        "seal_sha256": "synthetic-schedule-seal",
        "windows": [{
            "window_id": "W000",
            "organ_ids": [f"text_layer_{layer:02d}" for layer in layers],
            "evict_after_seal_shards": [],
        }],
    }


@pytest.fixture()
def environment(tmp_path, monkeypatch):
    """A real on-disk synthetic checkpoint wired into the capture module."""
    fixture = synthetic.build_synthetic_fixture(tmp_path / "fixture")
    config = dict(fixture.full_inventory.config)
    config["indexer_types"] = list(config["indexer_types"])
    config["mlp_layer_types"] = list(config["mlp_layer_types"])
    graph = _graph_from(fixture.full_inventory, config)
    schedule = _schedule_for([0, 1, 2, 3])

    capsules = tmp_path / "capsules"
    monkeypatch.setattr(tc, "SOURCE_ROOT", fixture.full_dir)
    monkeypatch.setattr(tc, "CAPSULES", capsules)
    monkeypatch.setattr(tc, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(tc, "MANIFEST_PATH", tmp_path / "absent-manifest.json")
    monkeypatch.setattr(tc, "CALIBRATION_TOKENS", 2)
    monkeypatch.setattr(tc, "LOGIT_LENS_ROWS", 4)
    monkeypatch.setattr(tc, "_graph", lambda: graph)
    monkeypatch.setattr(tc, "_schedule", lambda: schedule)
    monkeypatch.setattr(tc, "official_config", lambda: config)
    return {
        "root": fixture.full_dir,
        "graph": graph,
        "schedule": schedule,
        "config": config,
        "capsules": capsules,
        "tmp": tmp_path,
    }


def _capture(environment, layers):
    return tc.capture_layers(layers, capsule_dir=environment["capsules"])


def test_capture_is_deterministic(environment):
    first = _capture(environment, [0])
    (environment["capsules"] / "L00_L00.json").unlink()
    (environment["capsules"] / "L00_L00.npz").unlink()
    second = _capture(environment, [0])
    assert first["array_sha256"] == second["array_sha256"]
    assert first["capsule_sha256"] == second["capsule_sha256"]
    assert first["metrics"] == second["metrics"]


def test_capsule_reloads_and_reproduces_metrics(environment):
    receipt = _capture(environment, [3])
    result = tc.verify_capsule("L03_L03", capsule_dir=environment["capsules"])
    assert result["status"] == "REPRODUCED"
    assert result["metrics"] == receipt["metrics"]
    # A sparse layer must carry the full routing record, not just an output.
    for key in ("router_logits", "topk_indices", "topk_weights",
                "topk_margin_8th_vs_9th", "shared_expert_output",
                "routed_expert_output"):
        assert f"layer_03/{key}" in receipt["array_sha256"]


def test_chained_run_carries_the_previous_output(environment):
    _capture(environment, [0])
    second = _capture(environment, [1])
    assert second["input_provenance"] == "CHAINED_FROM_PREVIOUS_CAPSULE"
    assert second["chain_gap_layers"] == []
    with np.load(environment["capsules"] / "L00_L00.npz") as first:
        carry = first["carry_out_hidden"]
    with np.load(environment["capsules"] / "L01_L01.npz") as loaded:
        assert np.array_equal(loaded["layer_01/input_hidden"], carry)


def test_deep_run_without_a_chain_declares_its_gap(environment):
    receipt = _capture(environment, [3])
    assert receipt["input_provenance"] == "EMBEDDING_SEEDED_NOT_CHAINED"
    assert receipt["chain_gap_layers"] == [0, 1, 2]


def test_partial_write_is_refused_and_recapture_repairs(environment):
    _capture(environment, [0])
    capsule = environment["capsules"] / "L00_L00.npz"
    capsule.write_bytes(capsule.read_bytes()[: 128])
    with pytest.raises(tc.TeacherCaptureError, match="capsule bytes"):
        tc.verify_capsule("L00_L00", capsule_dir=environment["capsules"])
    _capture(environment, [0])
    assert tc.verify_capsule(
        "L00_L00", capsule_dir=environment["capsules"]
    )["status"] == "REPRODUCED"


def test_broken_seal_is_refused(environment):
    _capture(environment, [0])
    path = environment["capsules"] / "L00_L00.json"
    receipt = json.loads(path.read_text())
    receipt["metrics"]["layer_00"]["block_output_l2"] += 1.0
    path.write_text(json.dumps(receipt))
    with pytest.raises(tc.Glm52Error, match="seal mismatch"):
        tc.verify_capsule("L00_L00", capsule_dir=environment["capsules"])


def test_metric_drift_under_a_resealed_receipt_is_refused(environment):
    _capture(environment, [0])
    path = environment["capsules"] / "L00_L00.json"
    receipt = json.loads(path.read_text())
    receipt["metrics"]["layer_00"]["block_output_l2"] += 1.0
    path.write_text(json.dumps(tc.seal(receipt)))
    with pytest.raises(tc.TeacherCaptureError, match="metric reproduction mismatch"):
        tc.verify_capsule("L00_L00", capsule_dir=environment["capsules"])


def test_lineage_mismatch_is_refused(environment):
    _capture(environment, [0])
    path = environment["capsules"] / "L00_L00.json"
    receipt = json.loads(path.read_text())
    receipt["revision"] = "some-other-revision"
    path.write_text(json.dumps(tc.seal(receipt)))
    with pytest.raises(tc.TeacherCaptureError, match="lineage mismatch"):
        tc.verify_capsule("L00_L00", capsule_dir=environment["capsules"])


def test_membership_mismatch_is_refused(environment):
    _capture(environment, [0])
    path = environment["capsules"] / "L00_L00.json"
    receipt = json.loads(path.read_text())
    receipt["calibration_membership_sha256"] = "0" * 64
    path.write_text(json.dumps(tc.seal(receipt)))
    with pytest.raises(tc.TeacherCaptureError, match="membership mismatch"):
        tc.verify_capsule("L00_L00", capsule_dir=environment["capsules"])


def test_capture_refuses_a_layer_that_is_not_resident(environment, monkeypatch):
    graph = environment["graph"]
    organ = f"text_layer_{environment['config']['num_hidden_layers'] - 1:02d}"
    for entry in graph["organs"]:
        if entry["organ_id"] == organ:
            entry["source_shards"] = ["model-99999-of-00003.safetensors"]
    with pytest.raises(tc.TeacherCaptureError, match="not fully resident"):
        _capture(environment, [environment["config"]["num_hidden_layers"] - 1])


def test_eviction_is_refused_before_capture_and_authorized_after(environment):
    shard = environment["graph"]["organs"][0]["source_shards"][0]
    layers = sorted({
        layer
        for entry in environment["graph"]["organs"]
        if shard in entry["source_shards"]
        and (layer := tc.layer_of_organ(entry["organ_id"])) is not None
    })
    assert layers, "fixture shard must carry at least one text layer"

    before = tc.eviction_authority(
        [shard], source_root=environment["root"],
        graph=environment["graph"], capsule_dir=environment["capsules"],
    )
    assert before["authorized"] == []
    assert shard in before["refused_uncaptured_but_capturable"]

    for run in tc.contiguous_runs(layers):
        _capture(environment, run)

    after = tc.eviction_authority(
        [shard], source_root=environment["root"],
        graph=environment["graph"], capsule_dir=environment["capsules"],
    )
    assert after["authorized"] == [shard]
    assert after["refused_uncaptured_but_capturable"] == {}


def _widen_layer_organs(graph, phantom="model-99999-of-00003.safetensors"):
    """Give every text layer one shard that is not on disk."""
    for entry in graph["organs"]:
        if tc.layer_of_organ(entry["organ_id"]) is not None:
            entry["source_shards"] = sorted(set(entry["source_shards"]) | {phantom})
    return phantom


def test_already_destroyed_layers_do_not_deadlock_eviction(environment):
    """A layer whose siblings were fetched and destroyed is lost, not a block."""
    graph = environment["graph"]
    shard = graph["organs"][0]["source_shards"][0]
    phantom = _widen_layer_organs(graph)
    every_shard = {
        name for entry in graph["organs"] for name in entry["source_shards"]
    } | {phantom}
    authority = tc.eviction_authority(
        [shard], source_root=environment["root"], graph=graph,
        capsule_dir=environment["capsules"], ever_verified=every_shard,
    )
    assert authority["authorized"] == [shard]
    assert authority["authorized_with_unrecoverable_organs"][shard]


def test_not_yet_fetched_layers_are_refused_not_written_off(environment):
    """A layer that is merely incomplete must never be called unrecoverable."""
    graph = environment["graph"]
    shard = graph["organs"][0]["source_shards"][0]
    _widen_layer_organs(graph)
    authority = tc.eviction_authority(
        [shard], source_root=environment["root"], graph=graph,
        capsule_dir=environment["capsules"], ever_verified=set(),
    )
    assert authority["authorized"] == []
    assert shard in authority["refused_incomplete_organs"]
    assert authority["authorized_with_unrecoverable_organs"] == {}


def test_capture_for_eviction_captures_then_authorizes(environment):
    shard = environment["graph"]["organs"][0]["source_shards"][0]
    authority = tc.capture_for_eviction(
        [shard], source_root=environment["root"],
        graph=environment["graph"], capsule_dir=environment["capsules"],
    )
    assert authority["authorized"] == [shard]
    assert set(authority["capture_outcome"].values()) == {"CAPTURED"}
    assert authority["refused_uncaptured_but_capturable"] == {}


def test_ledger_row_carries_the_eviction_authorization(environment):
    _capture(environment, [0])
    rows = [
        json.loads(line)
        for line in tc.LEDGER.read_text().splitlines() if line.strip()
    ]
    row = rows[-1]
    assert row["event"] == "TEACHER_CAPTURED"
    assert row["capsule_id"] == "L00_L00"
    assert row["layers"] == [0]
    assert row["eviction_authorized_shards"]
    assert row["capsule_sha256"] and row["seal_sha256"]
    assert row["calibration_membership_sha256"]


def test_calibration_splits_are_disjoint_and_deterministic():
    assert tc.selftest()["status"] == "PASS"


def test_bounded_reader_refuses_an_oversized_tensor(environment):
    source = tc.ShardTensorSource(
        environment["root"], tc._tensor_table(environment["graph"]), max_tensor_bytes=1
    )
    with pytest.raises(tc.TeacherCaptureError, match="bounded read refused"):
        source.tensor("model.embed_tokens.weight")


def test_row_read_matches_a_full_tensor_read(environment):
    table = tc._tensor_table(environment["graph"])
    source = tc.ShardTensorSource(environment["root"], table)
    whole = source.tensor("model.embed_tokens.weight")
    ids = np.asarray([[1, 3]], dtype=np.int64)
    rows = source.rows("model.embed_tokens.weight", ids)
    assert rows.shape == (1, 2, whole.shape[1])
    assert np.array_equal(rows[0, 0], whole[1])
    assert np.array_equal(rows[0, 1], whole[3])
