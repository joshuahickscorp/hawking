"""Tests for the Hawking Mechanics Generation-M run-all orchestrator (M2..M7).

Enforce the sealed laws on tiny SYNTHETIC fixtures (no 128-expert loads): each stage grammar runs
and is measured (fully-populated, fully-labelled 10-dim mech vector); no execution grammar for M2..M6
materializes a dense shadow (peak temporary bounded well under the full dense tensor); the quality
gate rejects a lower-quality-faster candidate (fake-win ban); the conditional-doctor false-negative
gate fires when corrections are skipped; the Pareto excludes dominated AND inadmissible-dense
candidates; and everything is deterministic under a fixed seed. CPU (numpy) is authoritative.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gravity_forge as gf  # noqa: E402
import mech_measure as mm  # noqa: E402
import mech_run_all as mr  # noqa: E402
import gptoss_moe_runtime as rt  # noqa: E402

_VALID_LABELS = {"ANALYTICAL", "MEASURED", "ESTIMATED", "UNAVAILABLE"}
_H = 32          # tiny synthetic hidden size
_D = 16          # PQ reshape granularity (divides H)


def _expert(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    mlp1 = (rng.standard_normal((2 * _H, 8)).astype(np.float32)
            @ rng.standard_normal((8, _H)).astype(np.float32) * 0.1).astype(np.float32)   # [2H, H]
    mlp2 = (rng.standard_normal((_H, 8)).astype(np.float32)
            @ rng.standard_normal((8, _H)).astype(np.float32) * 0.1).astype(np.float32)     # [H, H]
    return {"mlp1": mlp1, "mlp2": mlp2,
            "mlp1_bias": (rng.standard_normal(2 * _H).astype(np.float32) * 0.01),
            "mlp2_bias": (rng.standard_normal(_H).astype(np.float32) * 0.01)}


def _cluster(n_experts: int = 3, n_acts: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    routed = list(range(n_experts))
    experts = {e: _expert(100 + e) for e in routed}
    router = {"weight": rng.standard_normal((n_experts, _H)).astype(np.float32),
              "bias": rng.standard_normal(n_experts).astype(np.float32) * 0.01}
    xs = [rng.standard_normal(_H).astype(np.float32) * 0.5 for _ in range(n_acts)]
    mlp1 = [experts[e]["mlp1"] for e in routed]
    mlp2 = [experts[e]["mlp2"] for e in routed]
    acts_mlp2 = [rt._swiglu(experts[routed[0]]["mlp1"] @ x + experts[routed[0]]["mlp1_bias"])
                 for x in xs]
    margins = []
    for x in xs:
        logits = router["weight"] @ x + router["bias"]
        order = np.argsort(-logits)
        sm = np.exp(logits[order] - logits[order].max()); sm = sm / sm.sum()
        margins.append(float(sm[0] - sm[1]) if len(sm) > 1 else 1.0)
    return {"router": router, "routed": routed, "experts": experts,
            "mats": {"mlp1": mlp1, "mlp2": mlp2}, "acts": {"mlp1": xs, "mlp2": acts_mlp2},
            "router_margins": margins, "xs": xs}


_CFG = {"D": _D, "k": 8, "stages": 2, "iters": 6, "island_budget_frac": 0.08,
        "doctor_k": 8, "doctor_stages": 1, "cond_threshold": 0.02, "need_delta": 1e-3, "fn_gate": 0.05}


def _quiet(_msg: str) -> None:
    pass


def _mech_fully_labelled(mv_dict: dict) -> None:
    assert set(mv_dict["vector"].keys()) == set(mm.DIMS)
    assert set(mv_dict["labels"].keys()) == set(mm.DIMS)
    for dim in mm.DIMS:
        assert mv_dict["labels"][dim] in _VALID_LABELS, (dim, mv_dict["labels"][dim])
        assert mv_dict["vector"][dim] >= 0.0


# --------------------------------------------------------------------------------------------
# Module selftest + frozen execution parity.
# --------------------------------------------------------------------------------------------
def test_module_selftest_green():
    out = mr.selftest()
    assert out["ok"]
    assert out["exec_parity_ok"]
    assert out["shared_flops_lt_independent"]
    assert out["no_dense_shadow_independent"] and out["no_dense_shadow_shared"]
    assert out["fuse_matches_unfused"]
    assert out["mech_dims"] == 10


def test_staged_execution_matches_frozen_shared_grammar():
    """The new staged lookup must reproduce the FROZEN pack_shared_grammar recon @ x (causal anchor)."""
    experts = [_expert(i)["mlp1"] for i in range(3)]
    x = np.random.default_rng(1).standard_normal(_H).astype(np.float32)
    codes = mr.build_staged_codes(experts, D=_D, k=8, stages=2, shared=True, seed=0, iters=6)
    gram = gf.pack_shared_grammar(experts, dim=_D, k=8, stages=2, corr_rank=0, iters=6)
    for i in range(3):
        assert mm._rel(mr.staged_execute_np(codes[i], x), gram.recon[i] @ x) < 1e-4


# --------------------------------------------------------------------------------------------
# Each stage grammar runs + is measured (fully-labelled mech vector, timing present).
# --------------------------------------------------------------------------------------------
def test_M2_runs_and_measured():
    cl = _cluster()
    r = mr.run_M2(cl, tensor="mlp1", cfg=_CFG, reps=3, baseline_div=0.0, dev=None,
                  metal_ok=False, progress=_quiet)
    assert len(r["rows"]) == 3                                   # independent / shared / layer_group_share
    names = {row["candidate"] for row in r["rows"]}
    assert names == {"independent", "shared", "layer_group_share"}
    for row in r["rows"]:
        _mech_fully_labelled(row["mech_vector"])
        assert row["timing_ms"]                                  # measured
    # the shared-table lever: shared does strictly fewer floating ops than independent
    assert r["causal"]["flops_shared_over_independent"] < 1.0
    assert r["causal"]["reuse_ratio_E"] == 3


def test_M3_M4_M5_M6_run_and_measured():
    cl = _cluster()
    m2 = mr.run_M2(cl, tensor="mlp1", cfg=_CFG, reps=3, baseline_div=0.0, dev=None,
                   metal_ok=False, progress=_quiet)
    m3 = mr.run_M3(cl, tensor="mlp1", cfg=_CFG, reps=3, baseline_div=0.0,
                   m2_built=m2["built"], progress=_quiet)
    assert len(m3["rows"]) == len(gf._ISLAND_STRATEGIES)         # 4 selectors
    pick = max(m3["rows"], key=lambda row: row["causal_delta"]["relerr_improvement"])["extra"]["strategy"]
    m4 = mr.run_M4(cl, tensor="mlp1", cfg=_CFG, reps=3, m3_pick=pick, progress=_quiet)
    m5 = mr.run_M5(cl, tensor="mlp1", cfg=_CFG, reps=3, m4=m4, progress=_quiet)
    m6 = mr.run_M6(cl, tensor="mlp1", cfg=_CFG, reps=3, m4=m4, progress=_quiet)
    for res in (m3, m4, m5, m6):
        for row in res["rows"]:
            _mech_fully_labelled(row["mech_vector"])
            assert row["timing_ms"]
    # M4 fusion is arithmetically identical to the separate-kernel control
    assert m4["rows"][0]["causal_delta"]["quality_matches_unfused"]
    # M6 causal control targets EQUAL bits to M4 (linear-in-stages analytic targeting); on tiny
    # fixtures the stage quantum is coarse, so allow a bounded window around 1.0.
    assert 0.5 <= m6["rows"][0]["causal_delta"]["bits_ratio_m6_over_m4"] <= 1.5


# --------------------------------------------------------------------------------------------
# No dense shadow: every M2..M6 execution grammar keeps peak temporary bounded under half dense.
# --------------------------------------------------------------------------------------------
def test_no_dense_shadow_M2_through_M6():
    cl = _cluster()
    rows = []
    m2 = mr.run_M2(cl, tensor="mlp1", cfg=_CFG, reps=2, baseline_div=0.0, dev=None,
                   metal_ok=False, progress=_quiet)
    rows += m2["rows"]
    m3 = mr.run_M3(cl, tensor="mlp1", cfg=_CFG, reps=2, baseline_div=0.0, m2_built=m2["built"],
                   progress=_quiet)
    rows += m3["rows"]
    pick = m3["rows"][0]["extra"]["strategy"]
    m4 = mr.run_M4(cl, tensor="mlp1", cfg=_CFG, reps=2, m3_pick=pick, progress=_quiet)
    rows += m4["rows"]
    rows += mr.run_M5(cl, tensor="mlp1", cfg=_CFG, reps=2, m4=m4, progress=_quiet)["rows"]
    rows += mr.run_M6(cl, tensor="mlp1", cfg=_CFG, reps=2, m4=m4, progress=_quiet)["rows"]
    full_dense = (2 * _H) * _H * 4
    for row in rows:
        nds = row["no_dense_shadow"]
        assert nds["bounded_under_half_dense"], (row["stage"], row["candidate"], nds)
        assert 0 < nds["peak_temporary_bytes"] < 0.5 * full_dense


# --------------------------------------------------------------------------------------------
# Quality gate rejects a lower-quality-faster candidate (fake-win ban).
# --------------------------------------------------------------------------------------------
def test_quality_gate_rejects_lower_quality():
    ok = mr.quality_gate(0.10, 0.10, tol=5e-3)          # matched
    assert ok["quality_admissible"]
    better = mr.quality_gate(0.05, 0.10, tol=5e-3)      # better
    assert better["quality_admissible"]
    worse = mr.quality_gate(0.20, 0.10, tol=5e-3)       # worse beyond tol -> rejected even if faster
    assert not worse["quality_admissible"]
    assert worse["verdict"] == "worse_quality_rejected"


def test_fake_win_ban_candidate_marked_inadmissible():
    """A candidate whose combine-divergence is worse than the baseline is inadmissible regardless of
    any speed - admissible must be False."""
    cl = _cluster()
    m2 = mr.run_M2(cl, tensor="mlp1", cfg=_CFG, reps=2, baseline_div=0.0, dev=None,
                   metal_ok=False, progress=_quiet)
    row = m2["rows"][0]
    row["quality_gate"] = mr.quality_gate(0.9, 0.1, tol=5e-3)   # force a much-worse quality
    row["admissible"] = bool(row["quality_gate"]["quality_admissible"]
                             and row["no_dense_shadow"]["bounded_under_half_dense"])
    assert row["admissible"] is False


# --------------------------------------------------------------------------------------------
# Conditional-doctor false-negative gate fires when needed corrections are skipped.
# --------------------------------------------------------------------------------------------
def test_conditional_false_negative_gate_fires():
    cl = _cluster()
    m2 = mr.run_M2(cl, tensor="mlp2", cfg=_CFG, reps=2, baseline_div=0.0, dev=None,
                   metal_ok=False, progress=_quiet)
    m3 = mr.run_M3(cl, tensor="mlp2", cfg=_CFG, reps=2, baseline_div=0.0, m2_built=m2["built"],
                   progress=_quiet)
    pick = m3["rows"][0]["extra"]["strategy"]
    m4 = mr.run_M4(cl, tensor="mlp2", cfg=_CFG, reps=2, m3_pick=pick, progress=_quiet)
    # threshold so high the doctor is ALWAYS skipped => every needed correction is a false negative
    cfg_skip = {**_CFG, "cond_threshold": 10.0, "need_delta": 1e-9}
    m5 = mr.run_M5(cl, tensor="mlp2", cfg=cfg_skip, reps=2, m4=m4, progress=_quiet)
    causal = m5["rows"][0]["causal_delta"]
    assert causal["skip_frac"] == pytest.approx(1.0)
    if causal["needed_corrections"] > 0:
        assert causal["false_negative_rate"] > _CFG["fn_gate"]
        assert causal["hard_gate_fires_reject"]
        assert m5["rows"][0]["quality_gate"]["verdict"] == "false_negative_gate_rejected"
        assert m5["rows"][0]["admissible"] is False


# --------------------------------------------------------------------------------------------
# Pareto excludes dominated + inadmissible-dense candidates.
# --------------------------------------------------------------------------------------------
def _row(cand, *, div, bpw, wall, movement, temp, floating, launches, dense=False):
    full = 1_000_000.0
    return {"stage": "MX", "candidate": cand, "tensor_class": "mlp1", "control_vs": "self",
            "mech_vector": mm.MechVector(F32=floating, Llookup=0.0, Mread=movement, Mwrite=0.0,
                                         Klaunch=launches, Ttemporary=temp).to_dict(),
            "rate": {"whole_artifact_bpw": bpw},
            "quality": {"rel_error_vs_dense_mean": div},
            "quality_gate": {"quality_admissible": True},
            "no_dense_shadow": {"bounded_under_half_dense": not dense,
                                "peak_temporary_bytes": temp, "full_dense_bytes": full},
            "timing_ms": {"MX": {"median_ms": wall}}, "causal_delta": None, "extra": {},
            "admissible": (not dense)}


def test_pareto_excludes_dominated_and_dense():
    good = _row("good", div=0.1, bpw=1.0, wall=1.0, movement=100, temp=10, floating=100, launches=1)
    dominated = _row("dominated", div=0.2, bpw=2.0, wall=2.0, movement=200, temp=20,
                     floating=200, launches=2)   # strictly worse than good on every axis
    dense = _row("dense_shadow", div=0.05, bpw=0.5, wall=0.5, movement=50, temp=999999,
                 floating=50, launches=1, dense=True)
    par = mr.build_pareto([good, dominated, dense])
    front = {f["candidate"] for f in par["frontier"]}
    assert "good" in front
    assert "dominated" not in front
    assert "dominated" in par["dominated"]
    assert "dense_shadow" in par["excluded_inadmissible_dense"]
    assert "dense_shadow" not in front
    assert par["n_admissible"] == 2                       # dense excluded from admissible pool


def test_pareto_champions_present():
    rows = [_row(f"c{i}", div=0.1 + 0.01 * i, bpw=1.0 + i, wall=1.0 + i, movement=100 + i,
                 temp=10 + i, floating=100 + i, launches=1 + i) for i in range(4)]
    par = mr.build_pareto(rows)
    champs = par["champions"]
    for key in ("quality_preserving_speed", "lowest_movement", "lowest_floating",
                "lowest_launches", "best_balanced_apple"):
        assert key in champs
    assert champs["lowest_movement"]["candidate"] == "c0"


# --------------------------------------------------------------------------------------------
# Determinism.
# --------------------------------------------------------------------------------------------
def test_deterministic_same_seed():
    experts = [_expert(i)["mlp1"] for i in range(3)]
    a = mr.build_staged_codes(experts, D=_D, k=8, stages=2, shared=True, seed=0, iters=6)
    b = mr.build_staged_codes(experts, D=_D, k=8, stages=2, shared=True, seed=0, iters=6)
    for ca, cb in zip(a, b):
        assert np.array_equal(ca.indices, cb.indices)           # indices bit-identical
    _, bda = mr.cluster_ledger(a)
    _, bdb = mr.cluster_ledger(b)
    assert bda["total_bits"] == bdb["total_bits"]               # accounting bit-identical
    x = np.random.default_rng(2).standard_normal(_H).astype(np.float32)
    assert np.array_equal(mr.staged_execute_np(a[0], x), mr.staged_execute_np(a[0], x))


def test_cluster_ledger_shared_cheaper_than_independent():
    """Shared codebook is billed ONCE; independent bills a codebook per expert -> strictly more bytes."""
    experts = [_expert(i)["mlp1"] for i in range(4)]
    sh = mr.build_staged_codes(experts, D=_D, k=8, stages=2, shared=True, seed=0, iters=6)
    ind = mr.build_staged_codes(experts, D=_D, k=8, stages=2, shared=False, seed=0, iters=6)
    _, bd_sh = mr.cluster_ledger(sh)
    _, bd_ind = mr.cluster_ledger(ind)
    assert bd_sh["whole_artifact_bpw"] < bd_ind["whole_artifact_bpw"]


def test_islands_are_billed_and_increase_bytes():
    experts = [_expert(i)["mlp1"] for i in range(2)]
    base = mr.build_staged_codes(experts, D=_D, k=8, stages=2, shared=True, seed=0, iters=6)
    _, bd0 = mr.cluster_ledger(base)
    isl = mr.build_staged_codes(experts, D=_D, k=8, stages=2, shared=True, seed=0, iters=6)
    for c, w in zip(isl, experts):
        mr.attach_islands(c, w, strategy="residual_energy", budget_frac=0.1)
    _, bd1 = mr.cluster_ledger(isl)
    assert bd1["island_bits"] > 0
    assert bd1["total_bits"] > bd0["total_bits"]                # islands are real, counted bytes
