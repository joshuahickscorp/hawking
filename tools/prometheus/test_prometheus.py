#!/usr/bin/env python3
"""Self-check for the Prometheus L1 spine. No framework: asserts + __main__.

Run:  python3 tools/prometheus/test_prometheus.py
It fails loudly if any load-bearing invariant breaks.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROMETHEUS_DIR = Path(__file__).resolve().parent
REPO = PROMETHEUS_DIR.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(PROMETHEUS_DIR) not in sys.path:
    sys.path.insert(0, str(PROMETHEUS_DIR))

from lab.layout import evidence_dir  # noqa: E402

import prometheus as P  # noqa: E402

LEDGER = evidence_dir("glm52") / "GLM52_LOGICAL_WEIGHT_LEDGER.json"
GLM_DENOMINATOR = 753329940480  # sealed logical_weight_denominator


def run(profile, frac=0.05, protect=4.0, seed=20260723):
    return P.run_pipeline(LEDGER, profile, frac, protect, seed)


def test_graph_matches_sealed_denominator():
    groups = P.load_ledger_graph(LEDGER)
    total = sum(g.logical_weights for g in groups)
    # primary_categories is a partial view; it must not EXCEED the sealed total and
    # must cover the overwhelming majority of weights.
    assert total <= GLM_DENOMINATOR, (total, GLM_DENOMINATOR)
    assert total / GLM_DENOMINATOR > 0.99, total / GLM_DENOMINATOR
    re = next(g for g in groups if g.category == "routed_expert")
    assert re.n_experts and re.n_experts > 1000, re.n_experts  # 58368/3 experts
    print(f"  graph: {total/1e9:.1f}B weights, {re.n_experts} expert-triples, ok")


def test_reconciliation_every_profile():
    for prof in ("uniform-v1", "general-v1", "math-v1", "mathnarrow-v1", "random-v1"):
        d = run(prof)["byte_decomposition"]
        assert d["reconciliation_ok"], prof
        assert d["effective_total_physical_bytes"] == (
            d["protected_tensor_bytes"] + d["main_representation_bytes"]), prof
    print("  reconciliation: 5/5 profiles reconcile physical == protected + main")


def test_math_and_random_are_byte_matched():
    """The control is only valid if Random costs exactly what Math costs. Anything
    else and Math-beats-Random could just be Math-spent-more-bytes."""
    m = run("math-v1")["byte_decomposition"]["effective_total_physical_bytes"]
    r = run("random-v1")["byte_decomposition"]["effective_total_physical_bytes"]
    assert m == r, ("matched-bytes control BROKEN", m, r)
    print(f"  control: Math == Random == {m/1e9:.3f} GB (byte-matched, valid control)")


def test_uniform_differs_from_conditioned():
    """If conditioning did nothing, Uniform and Math would allocate the same. They
    must not: Math excises non-target structure and protects the head."""
    u = run("uniform-v1")["byte_decomposition"]
    m = run("math-v1")["byte_decomposition"]
    assert u["excised_logical_weights"] == 0, "uniform must excise nothing"
    assert m["excised_logical_weights"] > 0, "math must excise mtp+vision"
    assert u["complete_bpw"] != m["complete_bpw"], (u["complete_bpw"], m["complete_bpw"])
    print(f"  conditioning bites: uniform bpw={u['complete_bpw']:.4f} excised=0  vs  "
          f"math bpw={m['complete_bpw']:.4f} excised={m['excised_logical_weights']/1e9:.2f}B")


def test_mathnarrow_excises_more_than_math():
    m = run("math-v1")["byte_decomposition"]["excised_logical_weights"]
    mn = run("mathnarrow-v1")["byte_decomposition"]["excised_logical_weights"]
    assert mn > m, (m, mn)
    print(f"  elimination: mathnarrow excises {mn/1e9:.2f}B > math {m/1e9:.2f}B (H3 surface)")


def test_excised_carry_zero_bytes():
    for b in run("math-v1")["blocks"]:
        if b["excised"]:
            assert b["physical_bytes"] == 0, b
    print("  excision honesty: every excised block carries 0 bytes, excluded from bitrate")


def test_generation_a_guard_fires():
    """The single rule that killed Generation A: a non-excised block with no payload.
    byte_decomposition MUST reject it."""
    bad = [P.Block("phantom", "router", "T0", logical_weights=1000,
                   physical_bytes=0, rate_bpw=0.0, excised=False)]
    try:
        P.byte_decomposition(bad)
    except AssertionError:
        print("  Gen-A guard: non-excised zero-payload block correctly rejected")
        return
    raise AssertionError("Gen-A guard did NOT fire on a billed-but-empty block")


def test_vocab_prune_reclaims_bytes_only_where_declared():
    """L4 down payment: pruning the vocab shrinks the protected head under math-v1
    (which declares vocab_prune) but does nothing under uniform-v1 (which does not)."""
    full = run("math-v1")["byte_decomposition"]["protected_tensor_bytes"]
    pruned = P.run_pipeline(LEDGER, "math-v1", 0.05, 4.0, 20260723,
                            vocab_retain=100000, hidden_size=6144)
    v = pruned["vocab_prune"]
    assert v["applied"] and v["reclaimed_bytes"] > 0, v
    assert pruned["byte_decomposition"]["protected_tensor_bytes"] < full
    # uniform-v1 declares no vocab_prune -> the prune is a no-op there
    u = P.run_pipeline(LEDGER, "uniform-v1", 0.05, 4.0, 20260723, vocab_retain=100000)
    assert u["vocab_prune"]["applied"] is False, u["vocab_prune"]
    print(f"  vocab prune: math reclaims {v['reclaimed_bytes']/1e6:.1f}MB "
          f"(vocab {v['full_vocab']}->{v['retained_vocab']}); uniform unaffected")


def test_synthetic_graph_model_independent():
    """The spine must not depend on GLM specifics: a tiny hand-built graph runs."""
    groups = [
        P.Group("routed_expert", 8_000_000, 16_000_000, 24, n_experts=8),
        P.Group("embedding", 100_000, 200_000, 1),
        P.Group("vision", 500_000, 1_000_000, 10),
        P.Group("router", 1_000, 2_000, 1),
    ]
    prof = P.load_profile("math-v1")
    blocks = P.allocate(groups, prof, coalition_fraction=0.25, coalition_protect_bpw=4.0,
                        seed=1)
    d = P.byte_decomposition(blocks)
    assert d["reconciliation_ok"]
    assert d["excised_logical_weights"] == 500_000  # vision excised
    print("  synthetic: model-independent graph runs, vision excised, reconciles")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"prometheus self-check: {len(tests)} invariants")
    for t in tests:
        t()
    print("ALL PASS")


if __name__ == "__main__":
    main()
