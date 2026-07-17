#!/usr/bin/env python3.12
"""Tests for the High-Parameter Frontier: source authority, physical fit, queue rows, atlas."""
import pathlib
import sys

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import succ_frontier as sf  # noqa: E402
import succ_atlas as atlas  # noqa: E402
from eco_common import sealed  # noqa: E402


def test_frontier_selftest():
    r = sf.selftest()
    assert r["ok"] is True and r["parents"] == 3


def test_atlas_selftest():
    r = atlas.selftest()
    assert r["ok"] is True and r["active_benchmark_deferred"] is True


def test_exact_bound_geometry_not_from_memory():
    # the three parents carry exact HF revisions + real geometry (corrections over hypotheses)
    by = {p.row_id: p for p in sf.PARENTS}
    assert by["deepseek-v3.2-685b"].revision == "a7e62ac04ecb2c0a54d736dc46601c5606cf10a6"
    assert by["deepseek-v3.2-685b"].n_routed_experts == 256
    assert by["deepseek-v3.2-685b"].experts_per_tok == 8
    assert by["deepseek-v3.2-685b"].mtp_layers == 1
    # V4-Pro real config: 6 selected experts (NOT the directive's hypothesized 8)
    assert by["deepseek-v4-pro-1.6t"].experts_per_tok == 6
    assert by["deepseek-v4-pro-1.6t"].n_routed_experts == 384
    # Kimi is multimodal -> claim split required
    assert by["kimi-k2.6-1t"].multimodal is True
    assert by["kimi-k2.6-1t"].mtp_layers == 0  # text-core has no MTP


def test_physical_fit_uses_official_total_denominator():
    for p in sf.PARENTS:
        fit = sf.physical_fit(p)
        assert "official_total" in fit["denominator"] and "never active" in fit["denominator"]
        # resident ceiling = 8 * safe_bytes / official_total (never active)
        assert fit["resident_ceiling_bpw"] > 0
        # anchor artifact uses total params, not active
        anchor = fit["anchor_fit"]["resident_anchor_bpw"]
        assert anchor["artifact_gb"] > 0


def test_regime_selection():
    fits = {p.row_id: sf.physical_fit(p)["selected_regime"] for p in sf.PARENTS}
    # 685B @0.80=68.5GB and 1T @0.55=68.75GB fit the 72GB safe envelope -> RESIDENT
    assert fits["deepseek-v3.2-685b"] == "RESIDENT_EXTREME"
    assert fits["kimi-k2.6-1t"] == "RESIDENT_EXTREME"
    # 1.6T @0.38=76GB > 72GB safe -> not resident at anchor
    assert fits["deepseek-v4-pro-1.6t"] in ("HYBRID_EXPERT_EXTREME", "STREAMED_ARCHIVE_EXTREME")


def test_queue_rows_are_durable_and_honestly_blocked():
    rows = sf.queue_rows()
    assert {r["parent_label"] for r in rows} == {"deepseek-v3.2-685b", "kimi-k2.6-1t",
                                                 "deepseek-v4-pro-1.6t"}
    for r in rows:
        assert sealed(r, "row_sha256")
        assert r["current_status"] == "waiting_adapter"
        assert r["adapter_id"] is None  # never claim an adapter exists
        assert r["blockers"] and r["exit_criteria"]
        assert "disk-walled" in r["disk_envelope"]


def test_manifest_sealed_and_ordered():
    man = sf.frontier_manifest()
    assert sealed(man, "manifest_sha256")
    assert man["heavy_execution_order"][0] == "72B_calibration"
    assert man["heavy_execution_order"][-1] == "deepseek-v4-pro-1.6t"
