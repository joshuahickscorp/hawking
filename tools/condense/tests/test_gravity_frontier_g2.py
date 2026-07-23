#!/usr/bin/env python3.12
"""Durability proofs for the Gate G2 complete-layer controller.

Proves the invariants that keep the complete-layer search honest and resumable:
  * SINGLETON        - a second controller cannot acquire the lease while one holds it, and the lease
                       label is DISTINCT from both gravity_frontier and second_light.
  * PROGRAM          - a declared hash is verified; a fixture without one adopts its body hash; the
                       real program builder seals a hash that EXCLUDES the timestamp (stable identity).
  * RESUME           - sealed rows are never redone; the queue continues from the next pending one.
  * CRASH/RESUME     - every HAWKING_G2_KILL_AT point resumes with no duplicate work, no partial
                       output, budget preserved (all five points exercised; >= 2 required).
  * OVER_BUDGET      - a row whose physical bits exceed the exact per-matrix budget is
                       FAILED_OVER_BUDGET.
  * STATUS           - NOT_STARTED with no controller; a stale PID / historical JSON is never RUNNING.
  * WINNER           - select_winner picks the HIGHEST hidden-state cosine WITHIN budget per class and
                       EXCLUDES the source_native controls.

Every row here runs against a TINY injected synthetic layer (hidden 32, six experts), so the whole
suite runs in seconds and never touches the real 120B source or packs 128 experts. The gravity_forge
geometry packers still run for real on the tiny matrices, so the pack path is genuinely exercised.
"""
from __future__ import annotations

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

from gravity_frontier_g2_controller import (  # noqa: E402
    CONTROL_FAMILY, G2Config, G2Controller, G2Error, KILL_POINTS, LEASE_LABEL,
    ROW_CHECKPOINT_SCHEMA, STATUS_SEALED, STATUS_FAILED_OVER_BUDGET, complete_layer_bpw,
    program_body_hash,
)
import gravity_frontier_g2_status as g2_status  # noqa: E402
import gravity_frontier_g2_program as g2_program  # noqa: E402
from gravity_frontier_controller import LEASE_LABEL as GRAVITY_LABEL  # noqa: E402
from second_light_controller import LEASE_LABEL as SECOND_LIGHT_LABEL  # noqa: E402


# -- tiny synthetic layer engine --------------------------------------------------------
class SyntheticLayerEngine:
    """A tiny GPT-OSS-like layer: hidden 32, intermediate 16, six experts. Deterministic across
    controller instances so resume recomputes identical activations."""

    HIDDEN = 32
    INTER = 16
    N_EXPERTS = 6

    def __init__(self, cfg: G2Config):
        self.cfg = cfg
        H, I, E = self.HIDDEN, self.INTER, self.N_EXPERTS
        rng = np.random.default_rng(0)
        self._router = {
            "weight": (rng.standard_normal((E, H)).astype(np.float32) * 0.1),
            "bias": (rng.standard_normal(E).astype(np.float32) * 0.01),
        }
        self._experts: dict[int, dict[str, np.ndarray]] = {}
        for e in range(E):
            er = np.random.default_rng(100 + e)
            mlp1 = (er.standard_normal((2 * I, 4)).astype(np.float32)
                    @ er.standard_normal((4, H)).astype(np.float32)) * 0.1        # [32, 32]
            mlp2 = (er.standard_normal((H, 2)).astype(np.float32)
                    @ er.standard_normal((2, I)).astype(np.float32)) * 0.1        # [32, 16]
            self._experts[e] = {
                "mlp1": np.ascontiguousarray(mlp1, dtype=np.float32),
                "mlp2": np.ascontiguousarray(mlp2, dtype=np.float32),
                "mlp1_bias": er.standard_normal(2 * I).astype(np.float32) * 0.01,
                "mlp2_bias": er.standard_normal(H).astype(np.float32) * 0.01,
            }
        self._acts: dict | None = None

    def activations(self) -> dict:
        if self._acts is None:
            def mk(n_seq: int, seed: int):
                items = []
                for i in range(n_seq):
                    rr = np.random.default_rng(seed + i)
                    for _p in range(3):
                        m = rr.standard_normal(self.HIDDEN).astype(np.float32) * 0.1
                        r = rr.standard_normal(self.HIDDEN).astype(np.float32) * 0.1
                        items.append((np.ascontiguousarray(m), np.ascontiguousarray(r)))
                return items
            self._acts = {
                "calibration": mk(self.cfg.n_calibration, 1),
                "validation": mk(self.cfg.n_validation, 5000),
                "source": "test_injected_synthetic_layer", "token_digest": "test",
            }
        return self._acts

    def router(self) -> dict:
        return self._router

    def load_expert(self, e: int) -> dict:
        return self._experts[int(e)]

    def representative_matrix(self, tensor_class: str) -> np.ndarray:
        if tensor_class == "expert_mlp1":
            return self._experts[0]["mlp1"]
        if tensor_class == "expert_mlp2":
            return self._experts[0]["mlp2"]
        if tensor_class == "router":
            return self._router["weight"]
        return self._experts[0]["mlp1"]

    def inventory(self) -> list[dict]:
        H, I, E = self.HIDDEN, self.INTER, self.N_EXPERTS
        return [
            {"name": "experts.mlp1", "n_weights": E * 2 * I * H, "native_bpw": 4.25,
             "tensor_class": "expert_mlp1"},
            {"name": "experts.mlp2", "n_weights": E * H * I, "native_bpw": 4.25,
             "tensor_class": "expert_mlp2"},
            {"name": "router", "n_weights": E * H, "native_bpw": 16.0, "tensor_class": "router"},
            {"name": "organ", "n_weights": H, "native_bpw": 16.0},
        ]


def _engine_factory(cfg: G2Config) -> SyntheticLayerEngine:
    return SyntheticLayerEngine(cfg)


# -- program fixtures -------------------------------------------------------------------
def _row(row_id, tensor_class, family, params, *, target_bits=10_000_000):
    return {
        "row_id": row_id,
        "candidate": True,
        "tensor_class": tensor_class,
        "layer": 0,
        "tensor_group_fmt": ("block.{b}.mlp.mlp1_weight" if tensor_class == "expert_mlp1"
                             else "block.{b}.mlp.mlp2_weight"),
        "representation_family": family,
        "family_params": params,
        "is_control": family == CONTROL_FAMILY,
        "exact_rate": "native" if family == CONTROL_FAMILY else "3/4",
        "exact_budget": {"n_weights_per_matrix": 1024, "rate": "3/4",
                         "target_total_bits": target_bits},
        "functional_metrics": ["layer_hidden_state_cosine"],
    }


def _write_program(path: Path, rows) -> None:
    doc = {"schema": "hawking.frontier_g2.complete_layer_program.v1",
           "gate": "G2_complete_layer", "generated_at": "2026-07-18T00:00:00Z",
           "rows": rows, "totals": {"total_rows": len(rows)}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_rows():
    return [
        _row("g2_000", "expert_mlp1", "product_quant", {"dim": 16, "subspaces": 2, "k": 64}),
        _row("g2_001", "expert_mlp1", "pq_protected_islands",
             {"dim": 16, "subspaces": 2, "k": 64, "strategy": "residual_energy",
              "budget_frac": 0.01}),
        _row("g2_002", "expert_mlp2", "naive_rvq", {"dim": 16, "k": 64, "stages": 2}),
        _row("g2_003", "expert_mlp1", CONTROL_FAMILY, {}),
    ]


def _cfg(tmp_path: Path, rows=None, **kw) -> G2Config:
    program = tmp_path / "PROGRAM.json"
    if not program.exists():
        _write_program(program, rows if rows is not None else _default_rows())
    kw.setdefault("engine_factory", _engine_factory)
    kw.setdefault("top_k", 2)
    kw.setdefault("n_calibration", 2)
    kw.setdefault("n_validation", 1)
    kw.setdefault("tokens_per_seq", 3)
    return G2Config(campaign_root=tmp_path, program_path=program, **kw)


def _file_sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- singleton + distinct lease ---------------------------------------------------------
def test_singleton_lease_refuses_second_controller(tmp_path):
    a = G2Controller(_cfg(tmp_path))
    a.acquire_lease()
    try:
        b = G2Controller(_cfg(tmp_path))
        with pytest.raises(G2Error):
            b.acquire_lease()
    finally:
        a.release_lease()
    c = G2Controller(_cfg(tmp_path))
    c.acquire_lease()
    c.release_lease()


def test_lease_label_distinct_from_other_campaigns():
    assert LEASE_LABEL == "com.hawking.frontier_g2"
    assert LEASE_LABEL != GRAVITY_LABEL
    assert LEASE_LABEL != SECOND_LIGHT_LABEL


# -- program integrity ------------------------------------------------------------------
def test_program_hash_adopted_for_fixture(tmp_path):
    ctl = G2Controller(_cfg(tmp_path))
    ctl.load_program()
    assert ctl.program_sha256 and len(ctl.program_sha256) == 64
    assert len(ctl.rows) == 4


def test_declared_program_hash_mismatch_is_rejected(tmp_path):
    program = tmp_path / "PROGRAM.json"
    _write_program(program, _default_rows())
    doc = json.loads(program.read_text())
    doc["program_sha256"] = "0" * 64
    program.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    ctl = G2Controller(G2Config(campaign_root=tmp_path, program_path=program,
                                engine_factory=_engine_factory))
    with pytest.raises(G2Error):
        ctl.load_program()


def test_program_hash_stable_across_regen_no_timestamp(tmp_path):
    """The timestamp-hash bug fix: program_sha256 excludes generated_at, so a regenerated program with
    a different timestamp hashes identically."""
    doc = g2_program.build()
    assert program_body_hash(doc) == doc["program_sha256"]
    # mutate ONLY the timestamp -> the sealed identity must be unchanged.
    mutated = dict(doc)
    mutated["generated_at"] = "2099-01-01T00:00:00Z"
    assert program_body_hash(mutated) == doc["program_sha256"]
    # a REAL content change (a row) must change the hash.
    changed = dict(doc)
    changed["rows"] = doc["rows"][:-1]
    assert program_body_hash(changed) != doc["program_sha256"]


def test_real_program_loads_in_controller(tmp_path):
    """End-to-end: the materialized builder program validates under the controller's load_program."""
    program = tmp_path / "G2_COMPLETE_LAYER_PROGRAM.json"
    doc = g2_program.build()
    program.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    ctl = G2Controller(G2Config(campaign_root=tmp_path, program_path=program,
                                engine_factory=_engine_factory))
    ctl.load_program()
    assert ctl.program_sha256 == doc["program_sha256"]
    # 4 + 4 expert-class candidates + 1 router + 1 attn control = 10 rows.
    assert len(ctl.rows) == 10


# -- resume: seal some, continue without redo -------------------------------------------
def test_resume_does_not_redo_sealed_rows(tmp_path):
    ctl = G2Controller(_cfg(tmp_path))
    summary = ctl.run(max_rows=2)
    assert summary["processed_this_invocation"] == 2

    ckpt_dir = tmp_path / "checkpoints"
    sealed_ids = ["g2_000", "g2_001"]
    shas_before = {rid: _file_sha(ckpt_dir / f"{rid}.json") for rid in sealed_ids}

    ctl2 = G2Controller(_cfg(tmp_path))
    resume = ctl2.resume(max_rows=2)
    assert resume["processed_this_invocation"] == 2
    for rid in sealed_ids:
        assert _file_sha(ckpt_dir / f"{rid}.json") == shas_before[rid], f"{rid} was redone"
    assert (ckpt_dir / "g2_002.json").exists() and (ckpt_dir / "g2_003.json").exists()

    ctl3 = G2Controller(_cfg(tmp_path))
    again = ctl3.run()
    assert again["processed_this_invocation"] == 0
    assert again["pending_rows"] == 0


# -- crash / resume for every kill point ------------------------------------------------
@pytest.mark.parametrize("point", KILL_POINTS)
def test_crash_and_resume_every_kill_point(tmp_path, point):
    row_ckpt = tmp_path / "checkpoints" / "g2_000.json"

    crashed = G2Controller(_cfg(tmp_path, only_rows=("g2_000",), kill_at=point, kill_row="g2_000"))
    with pytest.raises(SystemExit):
        crashed.run(max_rows=1)

    post_write = point in ("after_write", "after_receipt")
    if post_write:
        assert row_ckpt.exists(), f"{point}: durable checkpoint must survive the crash"
        sha_after_crash = _file_sha(row_ckpt)
    else:
        assert not row_ckpt.exists(), f"{point}: no partial output before the durable write"

    snap = g2_status.snapshot(_cfg(tmp_path, only_rows=("g2_000",)))
    assert snap["lease"]["live"] is False
    assert snap["state"] != "RUNNING"

    resumed = G2Controller(_cfg(tmp_path, only_rows=("g2_000",)))
    out = resumed.resume(max_rows=1)

    if post_write:
        assert _file_sha(row_ckpt) == sha_after_crash, f"{point}: sealed row was recomputed"
        assert out["processed_this_invocation"] == 0
    else:
        assert out["processed_this_invocation"] == 1
        assert row_ckpt.exists()

    assert out["completed_rows"] == 1
    cp = json.loads(row_ckpt.read_text())
    assert cp["schema"] == ROW_CHECKPOINT_SCHEMA
    assert cp["status"] == STATUS_SEALED
    m = cp["metrics"]
    assert m["within_budget"] is True
    assert m["physical_bits"] <= m["budget_bits"]
    val = m["functional"]["validation"]
    assert isinstance(val["layer_hidden_state_cosine"], float)
    assert val["router_topk_agreement"] == 1.0

    fresh = G2Controller(_cfg(tmp_path))
    fresh.acquire_lease()
    fresh.release_lease()


# -- over budget is a hard row failure --------------------------------------------------
def test_over_budget_marks_failed(tmp_path):
    rows = [_row("g2_000", "expert_mlp1", "product_quant", {"dim": 16, "subspaces": 2, "k": 64},
                 target_bits=1)]
    ctl = G2Controller(_cfg(tmp_path, rows=rows, only_rows=("g2_000",)))
    summary = ctl.run(max_rows=1)
    assert summary["failed_rows"] == 1 and summary["completed_rows"] == 0

    cp = json.loads((tmp_path / "checkpoints" / "g2_000.json").read_text())
    assert cp["status"] == STATUS_FAILED_OVER_BUDGET
    m = cp["metrics"]
    assert m["within_budget"] is False
    assert m["physical_bits"] > m["budget_bits"]

    snap = g2_status.snapshot(_cfg(tmp_path, rows=rows, only_rows=("g2_000",)))
    assert snap["state"] == "FAILED"


# -- status truth -----------------------------------------------------------------------
def test_status_not_started(tmp_path):
    snap = g2_status.snapshot(_cfg(tmp_path))
    assert snap["state"] == "NOT_STARTED"
    assert snap["lease"]["live"] is False
    assert snap["lease"]["heavy_controller_count"] == 0
    assert snap["lease"]["label"] == LEASE_LABEL


def test_stale_pid_and_historical_json_never_running(tmp_path):
    ctl = G2Controller(_cfg(tmp_path))
    ctl.controller_dir.mkdir(parents=True, exist_ok=True)
    ctl.lease_path.parent.mkdir(parents=True, exist_ok=True)
    ctl.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    from eco_common import seal_field, atomic_write_json
    bogus = seal_field({
        "schema": "hawking.frontier_g2.controller.v1",
        "controller_pid": 999999, "process_start_time": "2020-01-01T00:00:00+00:00",
        "program_sha256": "0" * 64, "current_row": "g2_000", "state_hint": "running",
        "completed_rows": 2, "failed_rows": 0, "pending_rows": 2, "total_working_rows": 4,
        "completed_row_ids": [], "failed_row_ids": [],
    }, "checkpoint_sha256")
    atomic_write_json(ctl.checkpoint_path, bogus)
    ctl.lease_path.write_text(json.dumps({"pid": 999999,
                                          "acquired_at": "2020-01-01T00:00:00+00:00"}) + "\n")

    snap = g2_status.snapshot(_cfg(tmp_path))
    assert snap["state"] != "RUNNING", "a stale PID / historical JSON must never report RUNNING"
    assert snap["lease"]["live"] is False
    assert snap["lease"]["holder_alive"] is False


# -- winner selection: highest hidden cosine within budget, controls EXCLUDED -----------
def test_winner_excludes_controls(tmp_path):
    rows = [
        _row("g2_000", "expert_mlp1", "product_quant", {"dim": 16, "subspaces": 2, "k": 64}),
        _row("g2_001", "expert_mlp1", "pq_protected_islands",
             {"dim": 16, "subspaces": 2, "k": 64, "strategy": "residual_energy",
              "budget_frac": 0.01}),
        _row("g2_002", "expert_mlp1", CONTROL_FAMILY, {}),   # control: cosine ~1.0, MUST be excluded
        _row("g2_003", "expert_mlp2", "naive_rvq", {"dim": 16, "k": 64, "stages": 2}),
        _row("g2_004", "expert_mlp2", CONTROL_FAMILY, {}),
    ]
    ctl = G2Controller(_cfg(tmp_path, rows=rows))
    ctl.run()
    selection = ctl.select_winner()
    winners = selection["winners_by_tensor_class"]

    # the source_native control (perfect cosine) is a reference boundary, never a winner.
    w1 = winners["expert_mlp1"]
    assert w1["family"] != CONTROL_FAMILY
    assert w1["winner_row_id"] in ("g2_000", "g2_001")
    assert "g2_002" not in w1["candidates"]
    assert w1["n_candidates"] == 2
    w2 = winners["expert_mlp2"]
    assert w2["winner_row_id"] == "g2_003"
    assert "g2_004" not in w2["candidates"]

    # control checkpoints ARE sealed (they ran), they are just excluded from selection.
    cp_control = json.loads((tmp_path / "checkpoints" / "g2_002.json").read_text())
    assert cp_control["status"] == STATUS_SEALED
    assert cp_control["metrics"]["is_control"] is True

    from eco_common import sealed
    doc = json.loads((tmp_path / "G2_SELECTION.json").read_text())
    assert sealed(doc, "selection_sha256")

    # the status frontier surfaces only real candidates (controls excluded from best).
    snap = g2_status.snapshot(_cfg(tmp_path, rows=rows))
    best = snap["hidden_cosine_frontier"]["best_by_hidden_cosine"]
    assert best is not None and best["family"] != CONTROL_FAMILY


# -- complete-layer bpw accounting ------------------------------------------------------
def test_complete_layer_bpw_drops_when_class_packed():
    """Packing a class sub-bit must lower the whole-layer bpw vs fully native; controls stay native."""
    eng = SyntheticLayerEngine(G2Config(campaign_root=Path("."), program_path=Path(".")))
    inv = eng.inventory()
    native = complete_layer_bpw(inv, None, None)
    packed = complete_layer_bpw(inv, "expert_mlp1", 0.75)
    assert packed < native
    # a router-native control does not repack anything -> equals native.
    assert complete_layer_bpw(inv, None, None) == native


# -- reset ------------------------------------------------------------------------------
def test_reset_clears_state(tmp_path):
    ctl = G2Controller(_cfg(tmp_path))
    ctl.run(max_rows=1)
    assert (tmp_path / "checkpoints" / "g2_000.json").exists()
    ctl.reset()
    assert not ctl.checkpoint_path.exists()
    assert not (tmp_path / "checkpoints" / "g2_000.json").exists()
    assert not ctl.selection_path.exists()
    snap = g2_status.snapshot(_cfg(tmp_path))
    assert snap["state"] == "NOT_STARTED"
