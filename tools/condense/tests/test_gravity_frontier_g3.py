#!/usr/bin/env python3.12
"""Durability + cross-layer proofs for the Gate G3 cross-layer-transfer controller.

Proves the invariants that keep the transfer search honest and resumable:
  * SINGLETON        - a second controller cannot acquire the lease while one holds it, and the lease
                       label is DISTINCT from frontier_g2, gravity_frontier, second_light and the
                       mechanics namespace.
  * PROGRAM          - a declared hash is verified; a fixture without one adopts its body hash; the
                       real 24-row builder program (0/18/35 x 8) loads under the controller.
  * GENERALIZED FWD  - block_n_moe_inputs runs the layer-parameterized reference forward for layers
                       0, 18 AND 35 and yields FINITE MoE inputs, requesting the correct block-N
                       tensors (a synthetic reader; the real 128-expert source is never touched).
  * RESUME           - sealed rows are never redone; the queue continues from the next pending one.
  * CRASH/RESUME     - every HAWKING_G3_KILL_AT point resumes with no duplicate work, no partial
                       output, budget preserved (all five points exercised; >= 2 required).
  * OVER_BUDGET      - a row whose physical bits exceed the exact per-matrix budget is
                       FAILED_OVER_BUDGET.
  * STATUS           - NOT_STARTED with no controller; a stale PID / historical JSON is never RUNNING;
                       active_generation is M.
  * TRANSFER + GEN-M - select_transfer records the per-(class, layer) winner and the layer-0->mid/late
                       transfer verdict, controls excluded, and every checkpoint binds Generation M.

Every engine row runs against a TINY injected synthetic layer (hidden 32, six experts), so the whole
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

from gravity_frontier_g3_controller import (  # noqa: E402
    BASE_EXECUTION_PROVIDER, CONTROL_FAMILY, EXECUTION_GENERATION, KILL_POINTS, LAYERS, LAYER_ROLES,
    LEASE_LABEL, ROW_CHECKPOINT_SCHEMA, STATUS_FAILED_OVER_BUDGET, STATUS_SEALED, G3Config,
    G3Controller, G3Error, block_n_moe_inputs, program_body_hash,
)
import gravity_frontier_g3_status as g3_status  # noqa: E402
import gravity_frontier_g3_program as g3_program  # noqa: E402
from gravity_frontier_g2_controller import LEASE_LABEL as G2_LABEL  # noqa: E402
from gravity_frontier_controller import LEASE_LABEL as GRAVITY_LABEL  # noqa: E402
from second_light_controller import LEASE_LABEL as SECOND_LIGHT_LABEL  # noqa: E402


# -- tiny synthetic per-layer engine ----------------------------------------------------
class SyntheticLayerEngine:
    """A tiny GPT-OSS-like layer, layer-aware: hidden 32, intermediate 16, six experts. Folding the
    layer into the seeds makes different depths genuinely different (so the transfer verdict has real
    content), while staying deterministic across controller instances so resume recomputes identically."""

    HIDDEN = 32
    INTER = 16
    N_EXPERTS = 6

    def __init__(self, cfg: G3Config, layer: int):
        self.cfg = cfg
        self.layer = int(layer)
        H, I, E = self.HIDDEN, self.INTER, self.N_EXPERTS
        rng = np.random.default_rng(1000 + self.layer)
        self._router = {
            "weight": (rng.standard_normal((E, H)).astype(np.float32) * 0.1),
            "bias": (rng.standard_normal(E).astype(np.float32) * 0.01),
        }
        self._experts: dict[int, dict[str, np.ndarray]] = {}
        for e in range(E):
            er = np.random.default_rng(100 + e + 37 * self.layer)
            mlp1 = (er.standard_normal((2 * I, 4)).astype(np.float32)
                    @ er.standard_normal((4, H)).astype(np.float32)) * 0.1
            mlp2 = (er.standard_normal((H, 2)).astype(np.float32)
                    @ er.standard_normal((2, I)).astype(np.float32)) * 0.1
            self._experts[e] = {
                "mlp1": np.ascontiguousarray(mlp1, dtype=np.float32),
                "mlp2": np.ascontiguousarray(mlp2, dtype=np.float32),
                "mlp1_bias": er.standard_normal(2 * I).astype(np.float32) * 0.01,
                "mlp2_bias": er.standard_normal(H).astype(np.float32) * 0.01,
            }
        self._acts: dict | None = None

    def source_present(self) -> bool:
        return True

    def activations(self) -> dict:
        if self._acts is None:
            def mk(n_seq: int, seed: int):
                items = []
                for i in range(n_seq):
                    rr = np.random.default_rng(seed + i + 11 * self.layer)
                    for _p in range(3):
                        m = rr.standard_normal(self.HIDDEN).astype(np.float32) * 0.1
                        r = rr.standard_normal(self.HIDDEN).astype(np.float32) * 0.1
                        items.append((np.ascontiguousarray(m), np.ascontiguousarray(r)))
                return items
            self._acts = {
                "calibration": mk(self.cfg.n_calibration, 1),
                "validation": mk(self.cfg.n_validation, 5000),
                "source": f"test_injected_synthetic_layer_{self.layer}",
                "token_digest": f"test_L{self.layer}", "layer": self.layer,
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


def _engine_factory(cfg: G3Config, layer: int) -> SyntheticLayerEngine:
    return SyntheticLayerEngine(cfg, layer)


# -- synthetic provenance reader for the generalized block-N forward test ---------------
class FakeReader:
    """Returns correctly-shaped small synthetic tensors for ANY block index, recording which tensor
    names were requested. Proves block_n_moe_inputs is layer-parameterized and numerically sound
    without touching the disk or the 128-expert source."""

    VOCAB = 64

    def __init__(self):
        self.requested: list[str] = []

    def bf16(self, name: str) -> np.ndarray:
        self.requested.append(name)
        rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
        if name == "embedding.weight":
            return (rng.standard_normal((self.VOCAB, 2880)).astype(np.float32) * 0.02)
        if name.endswith("attn.norm.scale") or name.endswith("mlp.norm.scale"):
            return np.ones(2880, dtype=np.float32)
        if name.endswith("attn.qkv.weight"):
            return (rng.standard_normal((5120, 2880)).astype(np.float32) * 0.02)
        if name.endswith("attn.qkv.bias"):
            return (rng.standard_normal(5120).astype(np.float32) * 0.02)
        if name.endswith("attn.out.weight"):
            return (rng.standard_normal((2880, 4096)).astype(np.float32) * 0.02)
        if name.endswith("attn.out.bias"):
            return (rng.standard_normal(2880).astype(np.float32) * 0.02)
        if name.endswith("attn.sinks"):
            return (rng.standard_normal(64).astype(np.float32) * 0.02)
        raise KeyError(name)


# -- program fixtures -------------------------------------------------------------------
def _row(row_id, tensor_class, family, params, layer, *, target_bits=10_000_000):
    role = dict(zip(LAYERS, LAYER_ROLES)).get(layer, "probe")
    return {
        "row_id": row_id,
        "candidate": True,
        "tensor_class": tensor_class,
        "layer": layer,
        "layer_role": role,
        "tensor_group_fmt": ("block.{b}.mlp.mlp1_weight" if tensor_class == "expert_mlp1"
                             else "block.{b}.mlp.mlp2_weight"),
        "representation_family": family,
        "family_params": params,
        "is_control": family == CONTROL_FAMILY,
        "exact_rate": "native" if family == CONTROL_FAMILY else "3/4",
        "exact_budget": {"n_weights_per_matrix": 1024, "rate": "3/4",
                         "target_total_bits": target_bits},
        "functional_metrics": ["layer_hidden_state_cosine", "cross_layer_transfer"],
        "execution_generation": EXECUTION_GENERATION,
        "base_execution_provider": BASE_EXECUTION_PROVIDER,
    }


def _write_program(path: Path, rows) -> None:
    doc = {"schema": "hawking.frontier_g3.cross_layer_program.v1",
           "gate": "G3_cross_layer_transfer", "generated_at": "2026-07-19T00:00:00Z",
           "layers": list(LAYERS), "rows": rows, "totals": {"total_rows": len(rows)}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _transfer_rows():
    """expert_mlp1 {pq, islands, control} crossed with all three probe layers -> a well-posed
    layer-0 -> mid/late transfer question."""
    rows = []
    i = 0
    for layer in LAYERS:
        for family, params in (("product_quant", {"dim": 16, "subspaces": 2, "k": 64}),
                               ("pq_protected_islands",
                                {"dim": 16, "subspaces": 2, "k": 64, "strategy": "residual_energy",
                                 "budget_frac": 0.01}),
                               (CONTROL_FAMILY, {})):
            rows.append(_row(f"g3_{i:03d}", "expert_mlp1", family, params, layer))
            i += 1
    return rows


def _default_rows():
    return _transfer_rows()


def _cfg(tmp_path: Path, rows=None, **kw) -> G3Config:
    program = tmp_path / "PROGRAM.json"
    if not program.exists():
        _write_program(program, rows if rows is not None else _default_rows())
    kw.setdefault("engine_factory", _engine_factory)
    kw.setdefault("top_k", 2)
    kw.setdefault("n_calibration", 2)
    kw.setdefault("n_validation", 1)
    kw.setdefault("tokens_per_seq", 3)
    return G3Config(campaign_root=tmp_path, program_path=program, **kw)


def _file_sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- singleton + distinct lease ---------------------------------------------------------
def test_singleton_lease_refuses_second_controller(tmp_path):
    a = G3Controller(_cfg(tmp_path))
    a.acquire_lease()
    try:
        b = G3Controller(_cfg(tmp_path))
        with pytest.raises(G3Error):
            b.acquire_lease()
    finally:
        a.release_lease()
    c = G3Controller(_cfg(tmp_path))
    c.acquire_lease()
    c.release_lease()


def test_lease_label_distinct_from_other_campaigns():
    assert LEASE_LABEL == "com.hawking.frontier_g3"
    assert LEASE_LABEL != G2_LABEL
    assert LEASE_LABEL != GRAVITY_LABEL
    assert LEASE_LABEL != SECOND_LIGHT_LABEL
    # distinct from the mechanics/thermodynamics namespace too (no shared lock).
    assert LEASE_LABEL != "com.hawking.mechanics_thermodynamics"
    assert len({LEASE_LABEL, G2_LABEL, GRAVITY_LABEL, SECOND_LIGHT_LABEL}) == 4


# -- program integrity ------------------------------------------------------------------
def test_program_hash_adopted_for_fixture(tmp_path):
    ctl = G3Controller(_cfg(tmp_path))
    ctl.load_program()
    assert ctl.program_sha256 and len(ctl.program_sha256) == 64
    assert len(ctl.rows) == 9


def test_declared_program_hash_mismatch_is_rejected(tmp_path):
    program = tmp_path / "PROGRAM.json"
    _write_program(program, _default_rows())
    doc = json.loads(program.read_text())
    doc["program_sha256"] = "0" * 64
    program.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    ctl = G3Controller(G3Config(campaign_root=tmp_path, program_path=program,
                                engine_factory=_engine_factory))
    with pytest.raises(G3Error):
        ctl.load_program()


def test_program_hash_stable_across_regen_no_timestamp():
    doc = g3_program.build()
    assert program_body_hash(doc) == doc["program_sha256"]
    mutated = dict(doc)
    mutated["generated_at"] = "2099-01-01T00:00:00Z"
    assert program_body_hash(mutated) == doc["program_sha256"]
    changed = dict(doc)
    changed["rows"] = doc["rows"][:-1]
    assert program_body_hash(changed) != doc["program_sha256"]


def test_real_program_loads_in_controller(tmp_path):
    """End-to-end: the materialized 24-row builder program (0/18/35 x 8) validates under the
    controller's load_program."""
    program = tmp_path / "G3_CROSS_LAYER_PROGRAM.json"
    doc = g3_program.build()
    program.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    ctl = G3Controller(G3Config(campaign_root=tmp_path, program_path=program,
                                engine_factory=_engine_factory))
    ctl.load_program()
    assert ctl.program_sha256 == doc["program_sha256"]
    assert len(ctl.rows) == 24
    layers = sorted({r["layer"] for r in ctl.rows})
    assert layers == list(LAYERS)


# -- generalized block-N forward: finite MoE inputs at layers 0/18/35 -------------------
@pytest.mark.parametrize("layer", list(LAYERS))
def test_generalized_block_forward_finite(layer):
    reader = FakeReader()
    ids = [1, 7, 3, 42]  # within FakeReader.VOCAB
    mi = block_n_moe_inputs(reader, layer, ids)
    assert mi.shape == (len(ids), 2880)
    assert np.isfinite(mi).all(), f"layer {layer} produced non-finite MoE inputs"
    # the forward requested the CORRECT block-N tensors (proves layer parameterization).
    assert f"block.{layer}.attn.qkv.weight" in reader.requested
    assert f"block.{layer}.attn.sinks" in reader.requested
    assert f"block.{layer}.mlp.norm.scale" in reader.requested


def test_generalized_forward_layers_differ():
    """Different depths give different MoE inputs (block-N tensors genuinely differ)."""
    reader = FakeReader()
    ids = [2, 5, 9]
    a = block_n_moe_inputs(reader, 0, ids)
    b = block_n_moe_inputs(reader, 35, ids)
    assert not np.allclose(a, b)


# -- resume: seal some, continue without redo -------------------------------------------
def test_resume_does_not_redo_sealed_rows(tmp_path):
    ctl = G3Controller(_cfg(tmp_path))
    summary = ctl.run(max_rows=2)
    assert summary["processed_this_invocation"] == 2

    ckpt_dir = tmp_path / "checkpoints"
    sealed_ids = ["g3_000", "g3_001"]
    shas_before = {rid: _file_sha(ckpt_dir / f"{rid}.json") for rid in sealed_ids}

    ctl2 = G3Controller(_cfg(tmp_path))
    resume = ctl2.resume(max_rows=3)
    assert resume["processed_this_invocation"] == 3
    for rid in sealed_ids:
        assert _file_sha(ckpt_dir / f"{rid}.json") == shas_before[rid], f"{rid} was redone"

    ctl3 = G3Controller(_cfg(tmp_path))
    again = ctl3.run()
    assert again["pending_rows"] == 0
    # all three probe layers were exercised.
    assert again["layers_touched"] == list(LAYERS)


# -- crash / resume for every kill point ------------------------------------------------
@pytest.mark.parametrize("point", KILL_POINTS)
def test_crash_and_resume_every_kill_point(tmp_path, point):
    row_ckpt = tmp_path / "checkpoints" / "g3_000.json"

    crashed = G3Controller(_cfg(tmp_path, only_rows=("g3_000",), kill_at=point, kill_row="g3_000"))
    with pytest.raises(SystemExit):
        crashed.run(max_rows=1)

    post_write = point in ("after_write", "after_receipt")
    if post_write:
        assert row_ckpt.exists(), f"{point}: durable checkpoint must survive the crash"
        sha_after_crash = _file_sha(row_ckpt)
    else:
        assert not row_ckpt.exists(), f"{point}: no partial output before the durable write"

    snap = g3_status.snapshot(_cfg(tmp_path, only_rows=("g3_000",)))
    assert snap["lease"]["live"] is False
    assert snap["state"] != "RUNNING"

    resumed = G3Controller(_cfg(tmp_path, only_rows=("g3_000",)))
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
    assert cp["execution_generation"] == "M"
    assert cp["generation_binding"]["generation"] == "M"
    assert isinstance(cp["generation_binding"].get("closure_sha256"), str)
    m = cp["metrics"]
    assert m["within_budget"] is True
    assert m["physical_bits"] <= m["budget_bits"]
    val = m["functional"]["validation"]
    assert isinstance(val["layer_hidden_state_cosine"], float)
    assert val["router_topk_agreement"] == 1.0

    fresh = G3Controller(_cfg(tmp_path))
    fresh.acquire_lease()
    fresh.release_lease()


# -- over budget is a hard row failure --------------------------------------------------
def test_over_budget_marks_failed(tmp_path):
    rows = [_row("g3_000", "expert_mlp1", "product_quant", {"dim": 16, "subspaces": 2, "k": 64},
                 0, target_bits=1)]
    ctl = G3Controller(_cfg(tmp_path, rows=rows, only_rows=("g3_000",)))
    summary = ctl.run(max_rows=1)
    assert summary["failed_rows"] == 1 and summary["completed_rows"] == 0

    cp = json.loads((tmp_path / "checkpoints" / "g3_000.json").read_text())
    assert cp["status"] == STATUS_FAILED_OVER_BUDGET
    m = cp["metrics"]
    assert m["within_budget"] is False
    assert m["physical_bits"] > m["budget_bits"]

    snap = g3_status.snapshot(_cfg(tmp_path, rows=rows, only_rows=("g3_000",)))
    assert snap["state"] == "FAILED"


# -- status truth -----------------------------------------------------------------------
def test_status_not_started(tmp_path):
    snap = g3_status.snapshot(_cfg(tmp_path))
    assert snap["state"] == "NOT_STARTED"
    assert snap["lease"]["live"] is False
    assert snap["lease"]["heavy_controller_count"] == 0
    assert snap["lease"]["label"] == LEASE_LABEL
    assert snap["active_generation"] == "M"


def test_stale_pid_and_historical_json_never_running(tmp_path):
    ctl = G3Controller(_cfg(tmp_path))
    ctl.controller_dir.mkdir(parents=True, exist_ok=True)
    ctl.lease_path.parent.mkdir(parents=True, exist_ok=True)
    ctl.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    from eco_common import seal_field, atomic_write_json
    bogus = seal_field({
        "schema": "hawking.frontier_g3.controller.v1",
        "controller_pid": 999999, "process_start_time": "2020-01-01T00:00:00+00:00",
        "program_sha256": "0" * 64, "current_row": "g3_000", "state_hint": "running",
        "completed_rows": 2, "failed_rows": 0, "pending_rows": 7, "total_working_rows": 9,
        "completed_row_ids": [], "failed_row_ids": [],
    }, "checkpoint_sha256")
    atomic_write_json(ctl.checkpoint_path, bogus)
    ctl.lease_path.write_text(json.dumps({"pid": 999999,
                                          "acquired_at": "2020-01-01T00:00:00+00:00"}) + "\n")

    snap = g3_status.snapshot(_cfg(tmp_path))
    assert snap["state"] != "RUNNING", "a stale PID / historical JSON must never report RUNNING"
    assert snap["lease"]["live"] is False
    assert snap["lease"]["holder_alive"] is False


# -- cross-layer transfer verdict + generation-M binding --------------------------------
def test_transfer_records_per_layer_winner_and_verdict(tmp_path):
    ctl = G3Controller(_cfg(tmp_path))
    ctl.run()
    transfer = ctl.select_transfer()
    tbc = transfer["transfer_by_tensor_class"]
    assert "expert_mlp1" in tbc
    entry = tbc["expert_mlp1"]

    # a winner geometry (never a control) at every probe layer.
    per_layer = entry["winner_per_layer"]
    assert set(per_layer) == {str(x) for x in LAYERS}
    for lk, w in per_layer.items():
        assert w["family"] != CONTROL_FAMILY
        assert w["winner_row_id"] not in ("",)
        assert w["n_candidates"] == 2  # control excluded from the 3 rows per layer

    # the transfer verdict is computed against the layer-0 winner.
    assert entry["layer0_winner_family"] == per_layer[str(LAYERS[0])]["family"]
    assert isinstance(entry["transfers_to_mid"], bool)
    assert isinstance(entry["transfers_to_late"], bool)
    assert entry["fully_transfers"] == bool(entry["transfers_to_mid"] and entry["transfers_to_late"])

    # sealed + capability_parity honesty flag present.
    from eco_common import sealed
    doc = json.loads((tmp_path / "G3_TRANSFER.json").read_text())
    assert sealed(doc, "transfer_sha256")
    assert doc["capability_parity"] is False

    # the status transfer frontier surfaces only real candidates (controls excluded from best).
    snap = g3_status.snapshot(_cfg(tmp_path))
    best = snap["transfer_frontier"]["best_by_hidden_cosine"]
    assert best is not None and best["family"] != CONTROL_FAMILY
    assert snap["transfer_frontier"]["transfer_present"] is True


# -- reset ------------------------------------------------------------------------------
def test_reset_clears_state(tmp_path):
    ctl = G3Controller(_cfg(tmp_path))
    ctl.run(max_rows=1)
    assert (tmp_path / "checkpoints" / "g3_000.json").exists()
    ctl.reset()
    assert not ctl.checkpoint_path.exists()
    assert not (tmp_path / "checkpoints" / "g3_000.json").exists()
    assert not ctl.transfer_path.exists()
    snap = g3_status.snapshot(_cfg(tmp_path))
    assert snap["state"] == "NOT_STARTED"
