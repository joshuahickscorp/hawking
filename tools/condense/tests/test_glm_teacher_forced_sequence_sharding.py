#!/usr/bin/env python3.12
"""Sequence-shard multi-worker GLM teacher-forced: bit-exact vs serial + refuse stream amp."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lab.operators import frankenstein_teacher_forced_executor as tfe  # noqa: E402
from lab.operators import glm52_synthetic as synthetic  # noqa: E402
from lab.operators.glm52_common import verify_sealed  # noqa: E402
from tools.condense.merge_glm_teacher_forced_shards import merge_workers  # noqa: E402


@pytest.fixture()
def fixture_env(tmp_path):
    root = tmp_path / "fixture"
    fx = synthetic.build_synthetic_fixture(root)
    return {"fixture": fx, "source": fx.full_dir, "tmp": tmp_path}


def _cfg(source, out, **kwargs):
    # Shared resident source across workers: MUST disable eviction so workers
    # do not unlink each other's (or the serial baseline's) weight shards.
    return tfe.ExecutorConfig(
        mode="synthetic",
        corpus_level=kwargs.get("level", "L0"),
        source_root=source,
        output_dir=out,
        max_sequence=kwargs.get("max_sequence", 2),
        microbatch=kwargs.get("microbatch", 8),
        sample_hidden=16,
        profile="synthetic",
        allow_eviction=kwargs.get("allow_eviction", False),
        require_floor=False,
        max_layers=kwargs.get("max_layers", None),
        seq_start=kwargs.get("seq_start", 0),
        seq_end=kwargs.get("seq_end", None),
        worker_id=kwargs.get("worker_id"),
        stream=False,
        allow_weight_stream_amplification=kwargs.get(
            "allow_weight_stream_amplification", False
        ),
    )


def test_stream_seq_shard_refused_by_default(fixture_env, tmp_path):
    with pytest.raises(tfe.TeacherForcedError, match="amplification|re-fetch|link-ceiling"):
        tfe.run_teacher_forced(
            tfe.ExecutorConfig(
                mode="official",
                corpus_level="L0",
                source_root=fixture_env["source"],
                output_dir=tmp_path / "refuse",
                profile="official",
                stream=True,
                require_floor=False,
                seq_start=0,
                seq_end=4,
                allow_weight_stream_amplification=False,
            )
        )


def test_sequence_shard_bit_exact_vs_serial(fixture_env, tmp_path):
    source = fixture_env["source"]
    serial_out = tmp_path / "serial"
    serial = tfe.run_teacher_forced(_cfg(source, serial_out, level="L0"))
    verify_sealed(serial, label="serial")
    n = int(serial["corpus"]["n_sequences"])
    assert n == 32

    # Two workers: [0,16) + [16,32)
    w0_out = tmp_path / "w0"
    w1_out = tmp_path / "w1"
    r0 = tfe.run_teacher_forced(
        _cfg(source, w0_out, level="L0", seq_start=0, seq_end=16, worker_id="w0")
    )
    r1 = tfe.run_teacher_forced(
        _cfg(source, w1_out, level="L0", seq_start=16, seq_end=32, worker_id="w1")
    )
    assert r0["shard"]["n_sequences_in_shard"] == 16
    assert r1["shard"]["n_sequences_in_shard"] == 16
    assert r0["shard"]["sharded"] is True
    verify_sealed(r0, label="w0")
    verify_sealed(r1, label="w1")

    merged_out = tmp_path / "merged"
    merged = merge_workers(merged_out, [w0_out, w1_out])
    assert merged["status"] == "MERGED"
    assert merged["seq_coverage"]["n_sequences"] == 32

    # Bit-exact layer + carry npz vs serial.
    for sub in ("layers", "carry"):
        for npz in sorted((serial_out / sub).glob("*.npz")):
            other = merged_out / sub / npz.name
            assert other.is_file(), f"missing merged {sub}/{npz.name}"
            with np.load(npz) as a, np.load(other) as b:
                assert set(a.files) == set(b.files)
                for k in a.files:
                    assert np.array_equal(a[k], b[k]), f"{npz.name}:{k}"


def test_four_workers_bit_exact(fixture_env, tmp_path):
    source = fixture_env["source"]
    serial_out = tmp_path / "serial"
    tfe.run_teacher_forced(_cfg(source, serial_out, level="L0", max_layers=2))
    worker_dirs = []
    for i in range(4):
        out = tmp_path / f"w{i}"
        tfe.run_teacher_forced(
            _cfg(
                source,
                out,
                level="L0",
                max_layers=2,
                seq_start=i * 8,
                seq_end=(i + 1) * 8,
                worker_id=f"w{i}",
            )
        )
        worker_dirs.append(out)
    merged_out = tmp_path / "merged"
    merge_workers(merged_out, worker_dirs)
    for npz in sorted((serial_out / "layers").glob("*.npz")):
        other = merged_out / "layers" / npz.name
        with np.load(npz) as a, np.load(other) as b:
            for k in a.files:
                assert np.array_equal(a[k], b[k]), f"{npz.name}:{k}"


def test_overlapping_shards_refused_at_merge(fixture_env, tmp_path):
    source = fixture_env["source"]
    a = tmp_path / "a"
    b = tmp_path / "b"
    tfe.run_teacher_forced(
        _cfg(source, a, level="L0", max_layers=1, seq_start=0, seq_end=20, worker_id="a")
    )
    tfe.run_teacher_forced(
        _cfg(source, b, level="L0", max_layers=1, seq_start=10, seq_end=32, worker_id="b")
    )
    with pytest.raises(ValueError, match="overlap"):
        merge_workers(tmp_path / "merged", [a, b])
