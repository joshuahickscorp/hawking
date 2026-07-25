"""Tests for the kernel selection matrix.

Everything here runs without a Metal device: the point of splitting the tables from the
driver is that the SELECTION is a pure function of measured rows, so it can be checked
without spending GPU. The one device-touching test skips on any device failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gravity_kernel_select as ks  # noqa: E402


R0 = {"S": 1, "rotate": False, "k": 128, "D": 8, "rows": 2048, "cols": 6144,
      "nchunk": 768, "index_bits_on_disk": 7, "rung": "R0"}


def row(kernel, **kw):
    base = {"kernel": kernel, "artifact_compatible": True, "parity_relative_l2": 1e-7,
            "incompatibility_reasons": []}
    base.update(kw)
    return base


# ------------------------------------------------------------------ compatibility (mandate 2)

def test_every_kernel_accepts_the_real_R0_geometry():
    for name, kernel in ks.KERNELS.items():
        ok, reasons = kernel.accepts(R0)
        assert ok, (name, reasons)


@pytest.mark.parametrize("kernel", sorted(ks.KERNELS))
def test_a_rotated_artifact_is_refused_by_every_kernel(kernel):
    ok, reasons = ks.KERNELS[kernel].accepts({**R0, "rotate": True})
    assert not ok
    assert any("rotat" in r for r in reasons)


@pytest.mark.parametrize("kernel", sorted(ks.KERNELS))
def test_a_larger_codebook_than_the_index_width_is_refused(kernel):
    ok, reasons = ks.KERNELS[kernel].accepts({**R0, "k": 512})
    assert not ok
    assert any("k=512" in r for r in reasons)


@pytest.mark.parametrize("kernel", sorted(ks.KERNELS))
def test_multiple_subspaces_are_refused(kernel):
    ok, reasons = ks.KERNELS[kernel].accepts({**R0, "S": 2})
    assert not ok
    assert any("subspaces" in r for r in reasons)


def test_lookup_linear_alone_needs_rows_divisible_by_four():
    odd = {**R0, "rows": 2050}
    assert not ks.KERNELS["lookup_linear"].accepts(odd)[0]
    assert ks.KERNELS["decode_fma_2d"].accepts(odd)[0]
    assert ks.KERNELS["production_v2"].accepts(odd)[0]


def test_decode_fma_alone_needs_D_divisible_by_four():
    odd = {**R0, "D": 6}
    assert not ks.KERNELS["decode_fma_2d"].accepts(odd)[0]
    assert ks.KERNELS["production_v2"].accepts(odd)[0]


def test_an_artifact_at_an_unknown_rung_is_refused():
    ok, reasons = ks.KERNELS["production_v2"].accepts({**R0, "rung": "R2"})
    assert not ok and any("R2" in r for r in reasons)


def test_a_codebook_that_cannot_be_staged_is_refused():
    ok, reasons = ks.KERNELS["production_v2"].accepts({**R0, "k": 256, "D": 64})
    assert not ok
    assert any("threadgroup memory" in r for r in reasons)


def test_every_refusal_names_a_source_line():
    for kernel in ks.KERNELS.values():
        assert kernel.refusal_sites
        for condition, source in kernel.refusal_sites:
            assert condition and source
            assert source.split(".")[0] in ("gravity_metal", "gravity_metal_lab_b")


# ------------------------------------------------------------------ the ranking (mandate 1)

def test_incompatibility_beats_speed():
    """A kernel that cannot consume the artifact loses even when it is 100x faster."""
    decision = ks.select_kernel([
        row("fast_but_incompatible", artifact_compatible=False,
            incompatibility_reasons=["artifact is rotated"],
            latency_wall_median_ms=0.01),
        row("slow_but_compatible", latency_wall_median_ms=1.0),
    ])
    assert decision["selected"] == "slow_but_compatible"
    assert decision["decision_trail"][0]["criterion"] == "artifact_compatible"


def test_a_parity_failure_is_not_a_candidate():
    decision = ks.select_kernel([
        row("wrong", parity_relative_l2=1e-2, latency_wall_median_ms=0.01),
        row("right", latency_wall_median_ms=1.0),
    ])
    assert decision["selected"] == "right"


def test_everything_rejected_raises_rather_than_picking_something():
    with pytest.raises(ks.KernelSelectError):
        ks.select_kernel([row("a", artifact_compatible=False,
                              incompatibility_reasons=["nope"])])


def test_wall_eliminates_a_grossly_slower_candidate():
    decision = ks.select_kernel([row("a", latency_wall_median_ms=1.0),
                                 row("b", latency_wall_median_ms=0.5)])
    assert decision["selected"] == "b"
    wall = next(t for t in decision["decision_trail"]
                if t["criterion"] == "latency_wall_median_ms")
    assert wall["inside_band"] == ["b"]


def test_wall_may_eliminate_but_never_decide():
    """A few percent of wall is host noise on the 215.8 us command-buffer constant.

    Track B's regression: two grammars 7% apart on wall but 1.4x apart on the GPU clock,
    in opposite directions. Wall must not be allowed to announce that winner.
    """
    decision = ks.select_kernel([
        row("fast_wall_slow_gpu", latency_wall_median_ms=0.2779,
            latency_gpu_median_ms=0.0587, latency_gpu_min_ms=0.0570),
        row("slow_wall_fast_gpu", latency_wall_median_ms=0.2985,
            latency_gpu_median_ms=0.0413, latency_gpu_min_ms=0.0400),
    ])
    assert decision["selected"] == "slow_wall_fast_gpu"
    assert decision["decided_by"] == "latency_gpu_median_ms"
    wall = next(t for t in decision["decision_trail"]
                if t["criterion"] == "latency_wall_median_ms")
    assert wall["decided"] is False


def test_a_candidate_with_no_gpu_clock_is_dropped_with_the_reason_recorded():
    decision = ks.select_kernel([
        row("production_v2", latency_wall_median_ms=0.30),
        row("decode_fma_2d", latency_wall_median_ms=0.29,
            latency_gpu_median_ms=0.03, latency_gpu_min_ms=0.028),
    ])
    assert decision["selected"] == "decode_fma_2d"
    gpu = next(t for t in decision["decision_trail"]
               if t["criterion"] == "latency_gpu_median_ms")
    assert gpu["dropped_for_no_measurement"] == ["production_v2"]


def test_median_and_min_must_agree_before_the_gpu_clock_decides():
    """Median inside the band, min disagreeing -> noise -> fall through to the ledgers."""
    decision = ks.select_kernel([
        row("a", latency_wall_median_ms=1.0, latency_gpu_median_ms=0.100,
            latency_gpu_min_ms=0.090, executed_total_bytes=5, executed_fp_ops=1,
            scratch_bytes=1),
        row("b", latency_wall_median_ms=1.0, latency_gpu_median_ms=0.102,
            latency_gpu_min_ms=0.089, executed_total_bytes=4, executed_fp_ops=1,
            scratch_bytes=1),
    ])
    assert decision["decided_by"] == "executed_total_bytes"
    assert decision["selected"] == "b"


def test_inside_the_band_the_gpu_clock_breaks_the_tie():
    """Two walls within 5% cannot separate grammars: wall is mostly command buffer."""
    decision = ks.select_kernel([
        row("a", latency_wall_median_ms=1.00, latency_gpu_median_ms=0.80),
        row("b", latency_wall_median_ms=1.02, latency_gpu_median_ms=0.20),
    ])
    assert decision["selected"] == "b"
    assert decision["decided_by"] == "latency_gpu_median_ms"


def test_a_full_tie_falls_through_to_the_resource_criteria():
    decision = ks.select_kernel([
        row("a", latency_wall_median_ms=1.0, latency_gpu_median_ms=0.5,
            executed_total_bytes=2_000_000, executed_fp_ops=10, scratch_bytes=8192),
        row("b", latency_wall_median_ms=1.0, latency_gpu_median_ms=0.5,
            executed_total_bytes=1_000_000, executed_fp_ops=10, scratch_bytes=8192),
    ])
    assert decision["selected"] == "b"
    assert decision["decided_by"] == "executed_total_bytes"


def test_an_unmeasured_criterion_cannot_decide():
    decision = ks.select_kernel([
        row("a", latency_wall_median_ms=1.0), row("b", latency_wall_median_ms=1.01)])
    trail = {t["criterion"]: t for t in decision["decision_trail"]}
    assert trail["latency_gpu_median_ms"]["outcome"] == "NOT_MEASURED_ON_ANY_SURVIVOR"


def test_the_decision_trail_cites_a_source_for_every_criterion():
    decision = ks.select_kernel([row("a", latency_wall_median_ms=1.0)])
    for step in decision["decision_trail"]:
        assert step["source"], step


def test_selection_carries_no_geometry_knowledge():
    """The ranking function must not name a shape; only the tables may."""
    source = Path(ks.__file__).read_text().split("def select_kernel")[1].split("\ndef ")[0]
    for forbidden in ("2048", "6144", "16384", "gate", "down", "attention"):
        assert forbidden not in source, forbidden


# ------------------------------------------------------------------ the tables cite measurements

def test_every_shape_prior_names_a_measurement():
    for prior in ks.SHAPE_PRIORS:
        assert prior["measurement"].endswith(".") and ".json" in prior["measurement"]
        assert prior["values"]


def test_every_criterion_names_a_source():
    for criterion in ks.CRITERIA:
        assert len(criterion.source) > 40
        assert criterion.role in ("hard_filter", "eliminate_only", "primary", "tiebreak")


def test_every_closed_lever_names_the_measurement_that_closed_it():
    for lever in ks.CLOSED_LEVERS:
        assert ".json" in lever["measurement"]
        assert lever["verdict"].isupper()


def test_the_within_tensor_hybrid_is_declared_unimplemented_not_invented():
    assert ks.HYBRID_WITHIN_TENSOR["status"] == "UNIMPLEMENTED_NOT_MEASURED"
    assert "run_batch" in ks.HYBRID_WITHIN_TENSOR["what_does_exist"]


def test_the_parity_gate_is_the_modules_own_tolerance_not_a_tighter_one():
    assert ks.PARITY_GATE == 2e-3


# ------------------------------------------------------------------ candidate expansion

def test_candidate_configs_come_only_from_the_shape_prior_values():
    facts = dict(R0)
    allowed = {(r["grammar"], r["field"]): set(r["values"]) for r in ks.SHAPE_PRIORS}
    for entry in ks.candidate_configs(facts):
        if entry["kernel"] == "production_v2":
            continue
        for field, values in ((f, v) for (g, f), v in allowed.items()
                              if g == entry["kernel"]):
            if field in entry["config"]:
                assert entry["config"][field] in values, entry


def test_a_config_the_device_would_refuse_never_becomes_a_candidate():
    """cbs=52 needs 32384 B; a wider codebook pushes it over and it must drop out."""
    facts = {**R0, "k": 256}
    kinds = {e["kernel"] for e in ks.candidate_configs(facts)}
    assert "lookup_linear" in kinds  # small cbs still fits
    for entry in ks.candidate_configs(facts):
        if entry["shape"] is not None:
            assert entry["shape"]["scratch_bytes"] <= ks.THREADGROUP_MEMORY_LIMIT


def test_blocks_are_clipped_to_nchunk_so_a_tiny_geometry_still_gets_candidates():
    facts = {**R0, "cols": 512, "nchunk": 64, "rows": 28672}
    entries = ks.candidate_configs(facts)
    assert {"production_v2", "decode_fma_2d", "lookup_linear"} == {e["kernel"] for e in entries}
    for entry in entries:
        if entry["shape"] is not None:
            assert entry["shape"]["blocks"] <= 64


# ------------------------------------------------------------------ byte accounting (mandate 6)

def test_the_census_reproduces_both_published_token_totals():
    census = ks.geometry_census()
    assert census["production_token_bytes"] == ks.PRODUCTION_TOKEN_BYTES
    assert census["artifact_floor_bytes"] == ks.ARTIFACT_FLOOR_BYTES


def test_the_census_carries_every_geometry_class_the_shards_hold():
    classes = ks.geometry_census()["classes"]
    for expected in ("routed_expert::2048x6144", "routed_expert::6144x2048",
                     "shared_expert::2048x6144", "shared_expert::6144x2048",
                     "attention::6144x16384", "attention::28672x512",
                     "attention::576x6144", "attention::16384x2048",
                     "attention::2048x6144", "dense_mlp::6144x12288"):
        assert expected in classes, expected


def test_the_unique_lower_bound_never_exceeds_the_executed_upper_bound():
    for blocks in (1, 8, 64):
        cbs = (768 + blocks - 1) // blocks
        shape = ks.lb.dfma_plan(rows=2048, nchunk=768, D=8, k=128, cbs=cbs, tpg=256)
        cost = ks.lb.dfma_cost(rows=2048, cols=6144, nchunk=768, D=8, k=128, shape=shape)
        lower = ks.unique_read_bytes(rows=2048, nchunk=768, D=8, k=128, blocks=blocks)
        assert lower <= cost["executed_total_bytes"]


def test_the_token_ledger_beats_the_production_total_when_the_kernel_moves_fewer_bytes():
    census = ks.geometry_census()
    selections = {
        key: {"kernel": "decode_fma_2d", "config": {},
              "executed_total_bytes":
                  entry["kernel_device_bytes"] // entry["per_token_tensors"] // 4,
              "unique_read_bytes":
                  entry["kernel_device_bytes"] // entry["per_token_tensors"] // 8}
        for key, entry in census["classes"].items() if entry["kernel_selectable"]
    }
    ledger = ks.token_ledger(census, selections)
    assert ledger["selected_executed_bytes_per_token"] < ks.PRODUCTION_TOKEN_BYTES
    assert ledger["reduction_vs_production_executed"] > 1.0
    assert (ledger["token_ceiling_tok_s"]["selected_executed_upper_bound"]
            > ledger["token_ceiling_tok_s"]["production_kernel"])


def test_an_unselected_class_carries_the_production_bytes_rather_than_vanishing():
    census = ks.geometry_census()
    ledger = ks.token_ledger(census, {})
    assert ledger["selected_executed_bytes_per_token"] == ks.PRODUCTION_TOKEN_BYTES
    bases = {r["selection_basis"] for r in ledger["rows"] if r["kernel"].startswith("UNSELECTED")}
    assert bases == {"PRODUCTION_KERNEL_BYTES_CARRIED_UNCHANGED"}


# ------------------------------------------------------------------ adversarial (mandate 5)

def test_all_same_index_really_is_one_codeword():
    import numpy as np
    codes = {"rows": 8, "cols": 32, "nchunk": 4, "D": 8, "S": 1, "sub": 8, "rotate": False,
             "indices": np.zeros((32, 1), dtype=np.uint8),
             "codebooks": [np.zeros((128, 8), dtype=np.float16)]}
    out = ks.adversarial_indices(codes, "all_same")
    assert set(np.unique(out["indices"])) == {0}
    assert out["codebooks"] is codes["codebooks"]      # the book is untouched


def test_spread_index_touches_every_codeword():
    import numpy as np
    codes = {"rows": 128, "cols": 32, "nchunk": 4, "D": 8, "S": 1, "sub": 8, "rotate": False,
             "indices": np.zeros((512, 1), dtype=np.uint8),
             "codebooks": [np.zeros((128, 8), dtype=np.float16)]}
    out = ks.adversarial_indices(codes, "spread")
    assert len(np.unique(out["indices"])) == 128


def test_an_unknown_adversarial_mode_is_refused():
    with pytest.raises(ks.KernelSelectError):
        ks.adversarial_indices({"rows": 1, "nchunk": 1,
                                "indices": None, "codebooks": [None]}, "whatever")


# ------------------------------------------------------------------ one device test

def test_the_selected_kernel_decodes_a_real_tensor_within_the_parity_gate():
    grf = pytest.importorskip("gravity_real_fixtures")
    try:
        fixtures = ks.collect_fixtures(None)
        bench = ks.Bench(dec=ks.lb.TrackBDecoder(), prod=ks.gravity_metal.decoder(),
                         reps=2, warmup=1)
    except Exception as exc:  # noqa: BLE001 - no device, no safe shard, no test
        pytest.skip(f"device or fixtures unavailable: {exc}")
    name, fixture = fixtures[0]
    result = ks.run_geometry(name, fixture, bench, with_dense=False)
    assert result["selected"]["parity"]["relative_l2"] < ks.PARITY_GATE
    assert result["activation_source"] == grf.SYNTHETIC
    assert result["expert_selection"].startswith("FIXED_LIST")
