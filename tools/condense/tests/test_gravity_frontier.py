#!/usr/bin/env python3.12
"""Durability proofs for the Gravity Frontier geometry-search controller.

Proves the invariants that keep the geometry search honest and resumable:
  * SINGLETON        - a second controller cannot acquire the lease while one holds it, and the
                       lease label is DISTINCT from second_light (the two campaigns never collide).
  * PROGRAM          - a program hash is verified when declared and adopted for a fixture otherwise.
  * RESUME           - sealed trials are never redone; the queue continues from the next pending one.
  * CRASH/RESUME x5  - every HAWKING_GF_KILL_AT point (fit, pack, eval, after_write, after_receipt)
                       resumes with no duplicate work, no partial output, budget preserved.
  * OVER_BUDGET      - a trial whose physical bits exceed the exact budget is FAILED_OVER_BUDGET.
  * STATUS           - NOT_STARTED with no controller; a stale PID / historical JSON is never RUNNING.
  * WINNER           - select_winner picks the LOWEST functional divergence WITHIN budget per class;
                       an even-lower-divergence trial that is OVER budget is excluded.

Every trial here uses a tiny injected synthetic weight (128x128) and an injected functional
divergence, so the whole suite runs in seconds and never touches the real 120B source or packs 128
experts. The gravity_forge geometry packers still run for real on the tiny weight, so the pack path
is genuinely exercised.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from gravity_frontier_controller import (  # noqa: E402
    FrontierConfig, GravityFrontierController, FrontierError, KILL_POINTS, LEASE_LABEL,
    STATUS_SEALED, STATUS_FAILED_OVER_BUDGET,
)
import gravity_frontier_status as gf_status  # noqa: E402
from second_light_controller import LEASE_LABEL as SECOND_LIGHT_LABEL  # noqa: E402


# -- synthetic fixtures -----------------------------------------------------------------
def _weight_for(tensor_class: str) -> np.ndarray:
    """A small deterministic weight per tensor class (same weight for every geometry, so trials
    compete only on their representation)."""
    seed = 11 if tensor_class == "expert_mlp1" else 23
    rng = np.random.default_rng(seed)
    base = (rng.standard_normal((128, 8)).astype(np.float32)
            @ rng.standard_normal((8, 128)).astype(np.float32))
    return np.ascontiguousarray(base * 0.1, dtype=np.float32)


def _weight_provider(row):
    return _weight_for(row["tensor_class"])


def _divergence_provider(row, pack_fn, weight):
    # Exercise the real geometry pack, then return the trial's controlled functional divergence so
    # winner selection is deterministic. pack_fn must reconstruct the injected weight.
    recon = pack_fn(np.ascontiguousarray(weight, dtype=np.float32))
    assert recon.shape == weight.shape and np.isfinite(recon).all()
    return {"mean_output_rel_div": float(row["_test_func_div"]),
            "signal": "test_injected", "n_experts_exercised": 1}


def _trial(row_id, tensor_class, family, params, *, func_div, target_bits=1_000_000,
           priority_rank=1, rate="3/4"):
    return {
        "row_id": row_id,
        "trial": True,
        "tensor_class": tensor_class,
        "tensor_group_fmt": ("block.{b}.mlp.mlp1_weight" if tensor_class == "expert_mlp1"
                             else "block.{b}.mlp.mlp2_weight"),
        "sample_layers": [0],
        "representation_family": family,
        "family_params": params,
        "protected_island_strategy": params.get("strategy"),
        "exact_budget": {"n_weights_per_matrix": 16384, "rate": rate,
                         "target_total_bits": target_bits},
        "exact_rate": rate,
        "priority_rank": priority_rank,
        "sharing_group": f"{tensor_class}_{family}",
        "functional_metric": "output_divergence (test-injected)",
        "_test_func_div": func_div,
    }


def _write_program(path: Path, rows) -> None:
    doc = {"schema": "hawking.gpt_oss_120b.gravity_frontier_program.v1",
           "generated_at": "2026-07-18T00:00:00+00:00", "rows": rows,
           "totals": {"total_trial_rows": len(rows)}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_rows():
    return [
        _trial("t0000", "expert_mlp1", "transform_pq", {"dim": 16, "k": 16, "subspaces": 2},
               func_div=0.50),
        _trial("t0001", "expert_mlp1", "product_quant", {"dim": 16, "k": 16, "subspaces": 2},
               func_div=0.30),
        _trial("t0002", "expert_mlp1", "naive_rvq", {"dim": 16, "k": 16, "stages": 2},
               func_div=0.40),
        _trial("t0003", "expert_mlp2", "transform_pq", {"dim": 16, "k": 16, "subspaces": 2},
               func_div=0.45),
    ]


def _cfg(tmp_path: Path, rows=None, **kw) -> FrontierConfig:
    program = tmp_path / "PROGRAM.json"
    if not program.exists():
        _write_program(program, rows if rows is not None else _default_rows())
    kw.setdefault("weight_provider", _weight_provider)
    kw.setdefault("divergence_provider", _divergence_provider)
    return FrontierConfig(campaign_root=tmp_path, program_path=program, **kw)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- singleton + distinct lease ---------------------------------------------------------
def test_singleton_lease_refuses_second_controller(tmp_path):
    a = GravityFrontierController(_cfg(tmp_path))
    a.acquire_lease()
    try:
        b = GravityFrontierController(_cfg(tmp_path))
        with pytest.raises(FrontierError):
            b.acquire_lease()
    finally:
        a.release_lease()
    c = GravityFrontierController(_cfg(tmp_path))
    c.acquire_lease()
    c.release_lease()


def test_lease_label_distinct_from_second_light():
    assert LEASE_LABEL == "com.hawking.gravity_frontier"
    assert LEASE_LABEL != SECOND_LIGHT_LABEL


# -- program integrity ------------------------------------------------------------------
def test_program_hash_adopted_for_fixture(tmp_path):
    ctl = GravityFrontierController(_cfg(tmp_path))
    ctl.load_program()
    assert ctl.program_sha256 and len(ctl.program_sha256) == 64
    assert len(ctl.rows) == 4


def test_declared_program_hash_mismatch_is_rejected(tmp_path):
    program = tmp_path / "PROGRAM.json"
    _write_program(program, _default_rows())
    doc = json.loads(program.read_text())
    doc["program_sha256"] = "0" * 64  # wrong on purpose
    program.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    ctl = GravityFrontierController(FrontierConfig(campaign_root=tmp_path, program_path=program,
                                                   weight_provider=_weight_provider,
                                                   divergence_provider=_divergence_provider))
    with pytest.raises(FrontierError):
        ctl.load_program()


# -- resume: seal 2, continue without redo ----------------------------------------------
def test_resume_does_not_redo_sealed_trials(tmp_path):
    ctl = GravityFrontierController(_cfg(tmp_path))
    summary = ctl.run(max_rows=2)
    assert summary["processed_this_invocation"] == 2

    ckpt_dir = tmp_path / "checkpoints"
    sealed_ids = ["t0000", "t0001"]
    for rid in sealed_ids:
        assert (ckpt_dir / f"{rid}.json").exists()
    shas_before = {rid: _file_sha(ckpt_dir / f"{rid}.json") for rid in sealed_ids}

    ctl2 = GravityFrontierController(_cfg(tmp_path))
    resume = ctl2.resume(max_rows=2)
    assert resume["processed_this_invocation"] == 2
    for rid in sealed_ids:
        assert _file_sha(ckpt_dir / f"{rid}.json") == shas_before[rid], f"{rid} was redone"
    assert (ckpt_dir / "t0002.json").exists() and (ckpt_dir / "t0003.json").exists()
    # all four sealed -> a fresh run is a no-op.
    ctl3 = GravityFrontierController(_cfg(tmp_path))
    again = ctl3.run()
    assert again["processed_this_invocation"] == 0
    assert again["pending_rows"] == 0


# -- crash / resume for all five kill points --------------------------------------------
@pytest.mark.parametrize("point", KILL_POINTS)
def test_crash_and_resume_every_kill_point(tmp_path, point):
    row_ckpt = tmp_path / "checkpoints" / "t0000.json"

    crashed = GravityFrontierController(_cfg(tmp_path, only_rows=("t0000",),
                                             kill_at=point, kill_row="t0000"))
    with pytest.raises(SystemExit):
        crashed.run(max_rows=1)

    post_write = point in ("after_write", "after_receipt")
    if post_write:
        assert row_ckpt.exists(), f"{point}: durable trial checkpoint must survive the crash"
        sha_after_crash = _file_sha(row_ckpt)
    else:
        assert not row_ckpt.exists(), f"{point}: no partial trial output before the durable write"

    # lease released by the in-process finally -> singleton preserved, status not RUNNING.
    snap = gf_status.snapshot(_cfg(tmp_path, only_rows=("t0000",)))
    assert snap["lease"]["live"] is False
    assert snap["state"] != "RUNNING"

    resumed = GravityFrontierController(_cfg(tmp_path, only_rows=("t0000",)))
    out = resumed.resume(max_rows=1)

    if post_write:
        assert _file_sha(row_ckpt) == sha_after_crash, f"{point}: sealed trial was recomputed"
        assert out["processed_this_invocation"] == 0
    else:
        assert out["processed_this_invocation"] == 1
        assert row_ckpt.exists()

    assert out["completed_rows"] == 1
    cp = json.loads(row_ckpt.read_text())
    assert cp["status"] == STATUS_SEALED
    m = cp["metrics"]
    assert m["within_budget"] is True
    assert m["physical_bits"] <= m["budget_bits"]
    assert isinstance(m["functional_divergence"], float)

    # a fresh controller can still take the lease (no leak).
    fresh = GravityFrontierController(_cfg(tmp_path))
    fresh.acquire_lease()
    fresh.release_lease()


# -- over budget is a hard trial failure ------------------------------------------------
def test_over_budget_marks_failed(tmp_path):
    rows = [_trial("t0000", "expert_mlp1", "transform_pq", {"dim": 16, "k": 16, "subspaces": 2},
                   func_div=0.5, target_bits=1)]
    ctl = GravityFrontierController(_cfg(tmp_path, rows=rows, only_rows=("t0000",)))
    summary = ctl.run(max_rows=1)
    assert summary["failed_rows"] == 1 and summary["completed_rows"] == 0

    cp = json.loads((tmp_path / "checkpoints" / "t0000.json").read_text())
    assert cp["status"] == STATUS_FAILED_OVER_BUDGET
    m = cp["metrics"]
    assert m["within_budget"] is False
    assert m["physical_bits"] > m["budget_bits"]

    snap = gf_status.snapshot(_cfg(tmp_path, rows=rows, only_rows=("t0000",)))
    assert snap["state"] == "FAILED"


# -- status truth -----------------------------------------------------------------------
def test_status_not_started(tmp_path):
    snap = gf_status.snapshot(_cfg(tmp_path))
    assert snap["state"] == "NOT_STARTED"
    assert snap["lease"]["live"] is False
    assert snap["lease"]["heavy_controller_count"] == 0
    assert snap["lease"]["label"] == LEASE_LABEL


def test_stale_pid_and_historical_json_never_running(tmp_path):
    ctl = GravityFrontierController(_cfg(tmp_path))
    ctl.controller_dir.mkdir(parents=True, exist_ok=True)
    ctl.lease_path.parent.mkdir(parents=True, exist_ok=True)
    ctl.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    from eco_common import seal_field, atomic_write_json
    bogus = seal_field({
        "schema": "hawking.gravity_frontier.controller.v1",
        "controller_pid": 999999, "process_start_time": "2020-01-01T00:00:00+00:00",
        "program_sha256": "0" * 64, "current_row": "t0000", "state_hint": "running",
        "completed_rows": 2, "failed_rows": 0, "pending_rows": 2, "total_working_rows": 4,
        "completed_row_ids": [], "failed_row_ids": [],
    }, "checkpoint_sha256")
    atomic_write_json(ctl.checkpoint_path, bogus)
    ctl.lease_path.write_text(json.dumps({"pid": 999999,
                                          "acquired_at": "2020-01-01T00:00:00+00:00"}) + "\n")

    snap = gf_status.snapshot(_cfg(tmp_path))
    assert snap["state"] != "RUNNING", "a stale PID / historical JSON must never report RUNNING"
    assert snap["lease"]["live"] is False
    assert snap["lease"]["holder_alive"] is False


# -- winner selection: lowest functional divergence WITHIN budget -----------------------
def test_winner_selection_picks_lowest_functional_within_budget(tmp_path):
    rows = [
        _trial("t0000", "expert_mlp1", "transform_pq", {"dim": 16, "k": 16, "subspaces": 2},
               func_div=0.50),                                  # within budget
        _trial("t0001", "expert_mlp1", "product_quant", {"dim": 16, "k": 16, "subspaces": 2},
               func_div=0.30),                                  # within budget, LOWEST valid
        _trial("t0002", "expert_mlp1", "naive_rvq", {"dim": 16, "k": 16, "stages": 2},
               func_div=0.40),                                  # within budget
        _trial("t0003", "expert_mlp1", "transform_pq", {"dim": 16, "k": 16, "subspaces": 2},
               func_div=0.05, target_bits=1),                   # LOWER divergence but OVER budget
        _trial("t0004", "expert_mlp2", "product_quant", {"dim": 16, "k": 16, "subspaces": 2},
               func_div=0.60),                                  # other class, within budget
    ]
    ctl = GravityFrontierController(_cfg(tmp_path, rows=rows))
    ctl.run()
    selection = ctl.select_winner()
    winners = selection["winners_by_tensor_class"]

    # expert_mlp1: the over-budget t0003 (0.05) is excluded; t0001 (0.30) wins over 0.40 and 0.50.
    assert winners["expert_mlp1"]["winner_row_id"] == "t0001"
    assert winners["expert_mlp1"]["functional_divergence"] == pytest.approx(0.30)
    assert "t0003" not in winners["expert_mlp1"]["candidates"]
    assert winners["expert_mlp1"]["n_candidates"] == 3
    # expert_mlp2: the only within-budget trial wins.
    assert winners["expert_mlp2"]["winner_row_id"] == "t0004"

    # the sealed selection file is present and self-seals.
    from eco_common import sealed
    doc = json.loads((tmp_path / "FRONTIER_SELECTION.json").read_text())
    assert sealed(doc, "selection_sha256")

    # status surfaces the frontier: lowest overall within-budget divergence is t0001 (0.30).
    snap = gf_status.snapshot(_cfg(tmp_path, rows=rows))
    best = snap["geometry_frontier"]["best_by_functional_divergence"]
    assert best["row_id"] == "t0001" and best["functional_divergence"] == pytest.approx(0.30)


# -- reset ------------------------------------------------------------------------------
def test_reset_clears_state(tmp_path):
    ctl = GravityFrontierController(_cfg(tmp_path))
    ctl.run(max_rows=1)
    assert (tmp_path / "checkpoints" / "t0000.json").exists()
    ctl.reset()
    assert not ctl.checkpoint_path.exists()
    assert not (tmp_path / "checkpoints" / "t0000.json").exists()
    assert not ctl.selection_path.exists()
    snap = gf_status.snapshot(_cfg(tmp_path))
    assert snap["state"] == "NOT_STARTED"
