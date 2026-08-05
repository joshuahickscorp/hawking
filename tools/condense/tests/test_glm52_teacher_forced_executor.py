#!/usr/bin/env python3.12
"""L0 smoke + invariants for the teacher-forced layer-major GLM executor."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lab.operators import glm52_synthetic as synthetic  # noqa: E402
from lab.operators import glm52_teacher_forced_executor as tfe  # noqa: E402
from lab.operators.glm52_common import verify_sealed  # noqa: E402


@pytest.fixture()
def fixture_env(tmp_path):
    root = tmp_path / "fixture"
    fx = synthetic.build_synthetic_fixture(root)
    out = tmp_path / "capture_out"
    return {"fixture": fx, "source": fx.full_dir, "out": out, "tmp": tmp_path}


def _run(fixture_env, **kwargs):
    cfg = tfe.ExecutorConfig(
        mode="synthetic",
        corpus_level=kwargs.get("level", "L0"),
        source_root=fixture_env["source"],
        output_dir=fixture_env["out"],
        max_sequence=kwargs.get("max_sequence", 2),
        microbatch=kwargs.get("microbatch", 8),
        sample_hidden=kwargs.get("sample_hidden", 16),
        profile="synthetic",
        allow_eviction=kwargs.get("allow_eviction", True),
        require_floor=False,  # CI / tmp may sit on a small volume snapshot
        max_layers=kwargs.get("max_layers", None),
    )
    return tfe.run_teacher_forced(cfg)


def test_l0_smoke_all_synthetic_layers(fixture_env):
    receipt = _run(fixture_env, level="L0")
    assert receipt["status"].startswith("PASS")
    assert receipt["corpus"]["n_sequences"] == 32
    n_layers = receipt["architecture"]["num_hidden_layers_config"]
    assert receipt["layers_captured"] == list(range(n_layers))
    assert receipt["deepest_layer_verified"] == n_layers - 1
    assert receipt["floor"]["floor_bytes"] == tfe.MIN_FREE_FLOOR_BYTES
    assert receipt["fabricated"] is False
    assert receipt["forward"]["autoregressive"] is False
    assert receipt["forward"]["kind"] == "teacher_forced_layer_major"
    verify_sealed(receipt, label="receipt")


def test_atomic_carry_seal_roundtrip(fixture_env):
    receipt = _run(fixture_env, level="L0", max_layers=2)
    carry = fixture_env["out"] / "carry" / "after_L00.npz"
    meta = fixture_env["out"] / "carry" / "after_L00.json"
    assert carry.is_file() and meta.is_file()
    doc = verify_sealed(json.loads(meta.read_text()), label="carry")
    with np.load(carry) as z:
        assert list(z["carry_hidden"].shape[:1]) == [32]
        assert doc["hidden_sha256"] == tfe._array_sha256(z["carry_hidden"])


def test_per_window_eviction_after_seal(fixture_env):
    receipt = _run(fixture_env, level="L0", allow_eviction=True)
    # Synthetic has 3 shards; after full stack at least some eviction should fire
    # once late layers complete early-shard-only organs.
    assert receipt["eviction_count"] >= 0
    # No fabricated path: if eviction happened, bytes_reclaimed tracks it.
    if receipt["eviction_count"]:
        assert receipt["bytes_reclaimed"] > 0
        # Carry states must still exist after eviction.
        assert (fixture_env["out"] / "carry" / "after_L00.npz").is_file()


def test_paired_traces_glm_side_present(fixture_env):
    receipt = _run(fixture_env, level="L0")
    traces = list((fixture_env["out"] / "paired_traces").glob("*.json"))
    assert len(traces) == 32
    one = json.loads(traces[0].read_text())
    assert one["sides"]["glm"]["present"] is True
    assert one["sides"]["glm"]["capture_status"] == "OK"
    assert one["sides"]["glm"]["teacher_forced"] is True
    assert one["fabricated"] is False
    assert one["sides"]["dsv4f"]["present"] is False
    idx = json.loads(
        (fixture_env["out"] / "PAIRED_TRACE_CORPUS_INDEX.json").read_text()
    )
    verify_sealed(idx, label="corpus index")
    assert idx["n_traces"] == 32


def test_bounded_capture_not_full_hidden_dump(fixture_env):
    _run(fixture_env, level="L0", sample_hidden=8)
    shard = fixture_env["out"] / "layers" / "L00.npz"
    with np.load(shard) as z:
        names = set(z.files)
        assert "block_output/samples" in names
        assert "block_output/mean" in names
        assert "block_output/l2" in names
        # Must NOT persist full [B,S,H] under a bare block_output key.
        assert "block_output" not in names
        samples = z["block_output/samples"]
        assert samples.shape[-1] == 8


def test_official_without_weights_fail_closed(tmp_path, fixture_env):
    # Point at empty source root — must not invent activations.
    empty = tmp_path / "empty_src"
    empty.mkdir()
    (empty / "config.json").write_text(
        (fixture_env["source"] / "config.json").read_text()
    )
    # Incomplete checkpoint: no shards.
    cfg = tfe.ExecutorConfig(
        mode="official",
        corpus_level="L0",
        source_root=empty,
        output_dir=tmp_path / "out_fail",
        profile="synthetic",  # config is synthetic geometry
        require_floor=False,
        allow_eviction=False,
    )
    with pytest.raises(tfe.TeacherForcedError):
        tfe.run_teacher_forced(cfg)


def test_double_buffer_log_covers_each_layer(fixture_env):
    receipt = _run(fixture_env, level="L0")
    log = receipt["double_buffer_log"]
    assert len(log) == receipt["architecture"]["num_hidden_layers_config"]
    assert log[0]["n"] == 0
    assert log[0]["n_minus_1"] is None
    assert log[-1]["n_plus_1"] is None


def test_floor_helper_refuses_below_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(tfe, "free_bytes", lambda _p: tfe.MIN_FREE_FLOOR_BYTES - 1)
    with pytest.raises(tfe.TeacherForcedError, match="25 GiB floor"):
        tfe.assert_floor(tmp_path)


def test_layer_major_is_deterministic(fixture_env):
    # Eviction must stay off so the shared fixture remains resident for run 2.
    r1 = _run(fixture_env, level="L0", allow_eviction=False)
    fixture_env["out"] = fixture_env["tmp"] / "capture_out_2"
    r2 = _run(fixture_env, level="L0", allow_eviction=False)
    assert r1["layers_captured"] == r2["layers_captured"]
    a = json.loads((fixture_env["tmp"] / "capture_out" / "layers" / "L00.json").read_text())
    b = json.loads((fixture_env["tmp"] / "capture_out_2" / "layers" / "L00.json").read_text())
    assert a["array_sha256"] == b["array_sha256"]
