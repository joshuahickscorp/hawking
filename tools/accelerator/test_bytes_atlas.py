"""The bytes column must fail when the attribution is wrong, or it measures nothing."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
import bytes_atlas as B  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (B.RESIDENT / "catalog.hq38m20").exists(),
    reason="the sealed resident artifact is not on this machine")


def test_all_three_reconciliations_hold():
    r = B.assert_reconciles()
    rec = r["reconciliation"]
    assert rec["R1_attribution_equals_catalog"]
    assert rec["R2_catalog_equals_disk"] and rec["R2_delta_bytes"] == 0
    assert rec["R3_ebpw_from_bytes"] == pytest.approx(B.SEALED_COMPLETE_EBPW, abs=1e-12)


def test_the_map_reproduces_the_MEASURED_dispatch_count():
    """964 comes from a runtime counter; this map comes from the artifact's own
    tensor inventory. Two independent routes to the same number is the check --
    and it FAILED at 948 until a weight-free kernel missing from the map was
    found, so it demonstrably can."""
    assert B.build()["dispatches_per_token"] == 964


@pytest.mark.parametrize("victim", [
    "language_model.model.layers.N.mlp.down_proj.weight",
    "language_model.model.layers.N.linear_attn.conv1d.weight",
    "language_model.model.norm.weight",
])
def test_DROPPING_ANY_ROLE_BREAKS_THE_RECONCILIATION(monkeypatch, victim):
    """The negative control. An attribution that reconciles no matter what it
    omits is not evidence that the artifact is accounted for."""
    patched = tuple(
        (k, n, tuple(r for r in roles if r != victim), w)
        for k, n, roles, w in B.DISPATCH_MAP)
    assert patched != B.DISPATCH_MAP, f"{victim} is not in the map; the control is vacuous"
    monkeypatch.setattr(B, "DISPATCH_MAP", patched)
    with pytest.raises(B.Unreconciled, match="read by NO dispatch"):
        B.build()


def test_claiming_one_role_twice_is_refused(monkeypatch):
    """The other direction: double-counting a tensor inflates per-token traffic
    and would make the bandwidth ceiling look further away than it is."""
    dup = B.DISPATCH_MAP + (
        ("fictional_second_reader", 1,
         ("language_model.model.layers.N.mlp.up_proj.weight",), True),)
    monkeypatch.setattr(B, "DISPATCH_MAP", dup)
    with pytest.raises(B.Unreconciled, match="claimed by two kernels"):
        B.build()


def test_the_embedding_TABLE_is_not_charged_to_a_token():
    """Decode reads ONE row. Charging the table would add 675 MB per token --
    6.8% of the traffic -- and is the obvious way to get this wrong."""
    r = B.build()
    embed = next(x for x in r["pareto_by_bytes"]
                 if x["kernel"] == "qwen_uniform_q4_embedding_lookup")
    assert embed["weight_bytes"] == B.EMBED_HIDDEN // 2 + (B.EMBED_HIDDEN // 64) * 2
    assert r["active_weight_bytes_per_token"] < r["catalog_total_bytes"]


def test_the_ceiling_does_not_reach_the_accepted_tps_target():
    """The finding, pinned so it cannot rot silently. Every byte this static
    accounting omits makes the ceiling LOWER, so the direction is safe."""
    bw = B.build()["bandwidth"]
    assert bw["raw_tps_ceiling_at_roof"] < bw["raw_tps_needed_for_50_accepted"]
    assert bw["ceiling_reaches_the_target"] is False
