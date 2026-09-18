"""Focused no-GPU fixture tests for the Flash E06 bounded prefilter."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools.odyssey import flash_e06_conditional_information_prefilter as e06


def _bf16_bytes(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype("<u2")
    return rounded.tobytes()


def _write_bf16_safetensor(path: Path, tensor: str, values: np.ndarray) -> None:
    payload = _bf16_bytes(values)
    header = {
        tensor: {
            "dtype": "BF16",
            "shape": list(values.shape),
            "data_offsets": [0, len(payload)],
        }
    }
    raw_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(raw_header).to_bytes(8, "little") + raw_header + payload)


@pytest.fixture
def e06_fixture(tmp_path: Path) -> tuple[Path, str, Path]:
    root = tmp_path / "specimen"
    root.mkdir()
    tensor = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    # Shape [expert, row, width] exercises a real leading-axis predictor.
    # Each row gets a distinct finite BF16 norm, so held-out coding/null tests
    # cannot accidentally collapse into a one-symbol control.
    values = np.empty((4, 8, 4), dtype=np.float32)
    for expert in range(values.shape[0]):
        for row in range(values.shape[1]):
            values[expert, row, :] = np.float32(1 + expert * 32 + row)
    shard = root / "toy.safetensors"
    _write_bf16_safetensor(shard, tensor, values)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor: shard.name}}, sort_keys=True), encoding="utf-8"
    )
    return root, tensor, shard


def _plan(root: Path, tensor: str, **kwargs: object) -> dict:
    options: dict[str, object] = {
        "root": root,
        "tensor": tensor,
        "rows_per_partition": 8,
        "seed": 71,
        "require_canonical": False,
    }
    options.update(kwargs)
    return e06.build_prefilter_plan(**options)


def _fixture_plan(root: Path, tensor: str, **kwargs: object) -> dict:
    options: dict[str, object] = {
        "root": root,
        "tensor": tensor,
        "rows_per_partition": 8,
        "seed": 71,
        "require_canonical": False,
    }
    options.update(kwargs)
    return e06._synthetic_fixture_plan(**options)


def test_header_only_plan_uses_existing_gate_and_never_preads_payload(e06_fixture, monkeypatch):
    root, tensor, _ = e06_fixture

    def body_read_forbidden(*_args, **_kwargs):
        raise AssertionError("header-only E06 plan must not read any tensor payload")

    monkeypatch.setattr(e06, "_pread_exact", body_read_forbidden)
    plan = _plan(root, tensor)

    assert plan["status"] == "DRY_RUN_HEADER_ONLY"
    assert plan["source_read_accounting"]["header_only"] is True
    assert plan["source_read_accounting"]["payload_bytes_read"] == 0
    assert plan["tensor"]["header_bytes_read"] == plan["tensor"]["header_bytes"]
    assert plan["sampling"]["planned_payload_bytes"] <= e06.MAX_PAYLOAD_BYTES
    assert set(plan["sampling"]["fit_row_ids"]).isdisjoint(plan["sampling"]["heldout_row_ids"])


def test_partitions_are_seeded_deterministic_disjoint_and_budgeted(e06_fixture):
    root, tensor, _ = e06_fixture
    first = _plan(root, tensor)
    second = _plan(root, tensor)
    changed = _plan(root, tensor, seed=72)

    assert first["sampling"] == second["sampling"]
    assert (
        first["sampling"]["fit_row_ids"],
        first["sampling"]["heldout_row_ids"],
    ) != (
        changed["sampling"]["fit_row_ids"],
        changed["sampling"]["heldout_row_ids"],
    )
    assert set(first["sampling"]["fit_row_ids"]).isdisjoint(first["sampling"]["heldout_row_ids"])
    assert first["sampling"]["planned_payload_bytes"] == (
        2 * first["sampling"]["rows_per_partition"] * first["tensor"]["row_bytes"]
    )


def test_plan_refuses_impossible_payload_budget_before_any_body_read(e06_fixture, monkeypatch):
    root, tensor, _ = e06_fixture

    def body_read_forbidden(*_args, **_kwargs):
        raise AssertionError("budget refusal must precede a body read")

    monkeypatch.setattr(e06, "_pread_exact", body_read_forbidden)
    with pytest.raises(e06.E06Refused, match="PAYLOAD_BUDGET_TOO_SMALL"):
        _plan(root, tensor, max_payload_bytes=3 * 8)
    with pytest.raises(e06.E06Refused, match="INVALID_MAX_PAYLOAD_BYTES"):
        _plan(root, tensor, max_payload_bytes=e06.MAX_PAYLOAD_BYTES + 1)


def test_fixture_evaluation_range_reads_only_selected_rows_and_bills_exact_bytes(e06_fixture, monkeypatch):
    root, tensor, _ = e06_fixture
    plan = _fixture_plan(root, tensor)
    calls: list[tuple[int, int]] = []
    original = e06._pread_exact

    def tracked(fd: int, offset: int, length: int) -> bytes:
        calls.append((offset, length))
        return original(fd, offset, length)

    monkeypatch.setattr(e06, "_pread_exact", tracked)
    report = e06._fixture_only_evaluate_sample(plan)

    expected_count = 2 * plan["sampling"]["rows_per_partition"]
    assert len(calls) == expected_count
    assert all(length == plan["tensor"]["row_bytes"] for _, length in calls)
    assert sum(length for _, length in calls) == report["source_read_accounting"]["payload_bytes_read"]
    assert report["source_read_accounting"]["payload_bytes_read"] <= e06.MAX_PAYLOAD_BYTES
    starts = {entry["file_offset"] for entry in report["source_read_accounting"]["range_reads"]}
    assert starts == {offset for offset, _ in calls}
    assert all(value is False for key, value in report["claim_boundary"].items() if key.startswith("is_"))
    assert report["shuffled_null"]["position_derangement"] is True
    assert report["shuffled_null"]["histogram_preserved"] is True
    accounting = report["actual_diagnostic_byte_accounting"]
    assert accounting["total_diagnostic_bytes"] == sum(
        value for key, value in accounting.items() if key != "total_diagnostic_bytes"
    )
    assert accounting["predictor_probability_table_bytes"] > 0
    assert accounting["source_safetensors_header_bytes"] == plan["tensor"]["header_bytes_read"]
    assert accounting["decoder_descriptor_bytes"] > 0
    projected = report["projected_complete_code_accounting"]
    assert projected["symbol_count"] == plan["tensor"]["flattened_rows"]
    assert projected["conditional_auxiliary_bytes"] > projected["global_auxiliary_bytes"]
    assert projected["is_ideal_code_length_not_serialized_codec"] is True
    signal = report["escalation_signal"]
    assert signal["candidate_for_separate_review"] == (
        signal["marginal_occupancy_lane_signal"]
        or signal["conditional_predictor_lane_signal"]
    )


def test_changed_shard_is_refused_before_any_payload_range_read(e06_fixture, monkeypatch):
    root, tensor, shard = e06_fixture
    plan = _fixture_plan(root, tensor)
    replacement = root / "replacement.safetensors"
    _write_bf16_safetensor(
        replacement,
        tensor,
        np.full((4, 8, 4), 99.0, dtype=np.float32),
    )
    os.replace(replacement, shard)

    def body_read_forbidden(*_args, **_kwargs):
        raise AssertionError("identity drift must be refused before body ranges")

    monkeypatch.setattr(e06, "_pread_exact", body_read_forbidden)
    with pytest.raises(e06.E06Refused, match="SOURCE_CHANGED_BEFORE_PAYLOAD_READ"):
        e06._fixture_only_evaluate_sample(plan)


def test_symlink_and_hardlink_shards_are_refused_before_body_reads(e06_fixture, tmp_path, monkeypatch):
    root, tensor, shard = e06_fixture
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(shard.read_bytes())
    symlink = root / "linked.safetensors"
    symlink.symlink_to(outside)

    def body_read_forbidden(*_args, **_kwargs):
        raise AssertionError("unsafe source path must not reach payload reads")

    monkeypatch.setattr(e06, "_pread_exact", body_read_forbidden)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor: symlink.name}}), encoding="utf-8"
    )
    with pytest.raises(e06.E06Refused, match="SOURCE_FILE_INVALID|SYMLINK_OPEN_REFUSED"):
        _plan(root, tensor)

    symlink.unlink()
    hardlink = root / "hardlinked.safetensors"
    os.link(shard, hardlink)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor: hardlink.name}}), encoding="utf-8"
    )
    with pytest.raises(e06.E06Refused, match="HARDLINKED_INPUT_REFUSED"):
        _plan(root, tensor)


def test_normal_fixture_plan_is_refused_by_source_evaluator_and_cli_owns_one_fixed_run(
    e06_fixture, monkeypatch, capsys, tmp_path,
):
    root, tensor, _ = e06_fixture
    plan = _plan(root, tensor)

    def body_read_forbidden(*_args, **_kwargs):
        raise AssertionError("normal plans must not reach fixture-only range reads")

    monkeypatch.setattr(e06, "_pread_exact", body_read_forbidden)
    with pytest.raises(e06.E06Refused, match="CANONICAL_SOURCE_REQUIRED"):
        e06._evaluate_sample(plan, fixture_only=False)

    observed = {}

    def fixed_run(*, emit):
        observed["emit"] = emit
        return {"schema": e06.SCHEMA, "status": "E06_PREFILTER_STOP_AUXILIARY_COST_OR_NULL_CONTROL_NOT_CLEARED"}

    monkeypatch.setattr(e06, "run_canonical_prefilter", fixed_run)
    destination = tmp_path / "e06.json"
    assert e06.main(["--emit", str(destination)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"].startswith("E06_PREFILTER_")
    assert observed["emit"] == destination


def test_fixture_helpers_refuse_canonical_root_and_a_forged_marker_before_pread(
    e06_fixture, monkeypatch
):
    """Fixture-only access is bound to its real root, not a plan label."""
    root, tensor, _ = e06_fixture

    def body_read_forbidden(*_args, **_kwargs):
        raise AssertionError("canonical-root refusal must precede every payload pread")

    monkeypatch.setattr(e06, "_pread_exact", body_read_forbidden)
    with pytest.raises(e06.E06Refused, match="SYNTHETIC_FIXTURE_REQUIRED"):
        e06._synthetic_fixture_plan()
    with pytest.raises(e06.E06Refused, match="SYNTHETIC_FIXTURE_REQUIRED"):
        e06._synthetic_fixture_plan(root=e06.DEFAULT_MODEL_ROOT / "subtree")
    with pytest.raises(e06.E06Refused, match="SYNTHETIC_FIXTURE_REQUIRED"):
        e06._synthetic_fixture_plan(root=e06.DEFAULT_MODEL_ROOT.parent)

    forged = _fixture_plan(root, tensor)
    forged["source_identity"]["source_root"] = str(e06.DEFAULT_MODEL_ROOT)
    forged["source_identity"]["source_mode"] = "noncanonical_header_only"
    forged["plan_sha256"] = e06._sha256_value(
        {key: value for key, value in forged.items() if key != "plan_sha256"}
    )
    with pytest.raises(e06.E06Refused, match="SYNTHETIC_FIXTURE_REQUIRED"):
        e06._fixture_only_evaluate_sample(forged)
    forged["source_identity"]["source_root"] = str(e06.DEFAULT_MODEL_ROOT.parent)
    forged["plan_sha256"] = e06._sha256_value(
        {key: value for key, value in forged.items() if key != "plan_sha256"}
    )
    with pytest.raises(e06.E06Refused, match="SYNTHETIC_FIXTURE_REQUIRED"):
        e06._fixture_only_evaluate_sample(forged)


def test_changed_plan_digest_is_refused_before_payload_read(e06_fixture, monkeypatch):
    root, tensor, _ = e06_fixture
    plan = _fixture_plan(root, tensor)
    plan["sampling"]["fit_row_ids"][0] += 1

    def body_read_forbidden(*_args, **_kwargs):
        raise AssertionError("plan drift must be refused before payload reads")

    monkeypatch.setattr(e06, "_pread_exact", body_read_forbidden)
    with pytest.raises(e06.E06Refused, match="PLAN_DIGEST_MISMATCH"):
        e06._fixture_only_evaluate_sample(plan)
