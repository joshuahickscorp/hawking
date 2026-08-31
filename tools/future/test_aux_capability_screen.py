"""Tests for the auxiliary-stream capability screen.

Load-bearing negatives:
  * argmax agreement alone is not parity
  * a lever missing a damage stage without an early-stop reason is refused
  * bytes_removed without bytes_added is not a candidate
  * overlap relations stay inside the stated vocabulary
  * variable group-size metadata is billed, not forgotten
  * a tps hardware number cannot land in the receipt
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tools.future import aux_capability_screen as acs
from tools.future import executable_economics as ee
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    HardwareClaimError,
    _assert_no_hardware_claims,
    write_receipt,
)
from tools.future.mlp_auxiliary_information import AUXILIARY_BYTES_TARGET
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def test_argmax_agreement_alone_is_refused_as_parity():
    """NEGATIVE CONTROL: keeping argmax is not a capability screen."""
    with pytest.raises(acs.ArgmaxAloneParityRefuse) as caught:
        acs.report_logit_parity(
            kl_nats=None,
            top_k_agreement=None,
            argmax_agreement=1.0,
            k=5,
        )
    assert "REFUSED" in str(caught.value)
    assert "argmax" in str(caught.value).lower()
    assert "not parity" in str(caught.value).lower()

    with pytest.raises(acs.ArgmaxAloneParityRefuse):
        acs.report_logit_parity(
            kl_nats=None,
            top_k_agreement=0.99,
            argmax_agreement=1.0,
        )
    with pytest.raises(acs.ArgmaxAloneParityRefuse):
        acs.report_logit_parity(
            kl_nats=0.001,
            top_k_agreement=None,
            argmax_agreement=1.0,
        )


def test_logit_parity_requires_kl_and_topk_and_flags_argmax():
    ok = acs.report_logit_parity(
        kl_nats=0.02,
        top_k_agreement=0.8,
        argmax_agreement=1.0,
        k=5,
        n_rows=8,
    )
    assert ok["kl_nats"] == pytest.approx(0.02)
    assert ok["top_k"] == 5
    assert ok["top_k_agreement"] == pytest.approx(0.8)
    assert ok["argmax_agreement"] == pytest.approx(1.0)
    assert ok["argmax_is_not_parity"] is True
    assert "kl_nats" in ok["parity_quantities"]
    assert "top_k_agreement" in ok["parity_quantities"]
    # A drift that preserves argmax is still a measured KL.
    drifted = acs.report_logit_parity(
        kl_nats=0.4,
        top_k_agreement=0.2,
        argmax_agreement=1.0,
        k=5,
    )
    assert drifted["argmax_agreement"] == pytest.approx(1.0)
    assert drifted["kl_nats"] > acs.LOGIT_KL_BAR
    assert acs.logit_fails(drifted) is True


def test_mean_logit_parity_distinguishes_kl_from_argmax():
    """A candidate can keep argmax and still fail KL / top-k."""
    rng = np.random.default_rng(0)
    inc = rng.normal(size=(4, 32)).astype(np.float32)
    # Boost the incumbent argmax, then add a long tail of drift.
    cand = inc.copy()
    for i in range(inc.shape[0]):
        am = int(np.argmax(inc[i]))
        cand[i] = inc[i] + rng.normal(scale=1.5, size=inc.shape[1]).astype(np.float32)
        cand[i, am] = float(inc[i, am]) + 8.0  # keep argmax
    parity = acs.mean_logit_parity(inc, cand, k=5)
    assert parity["argmax_agreement"] == pytest.approx(1.0)
    assert parity["argmax_is_not_parity"] is True
    assert parity["kl_nats"] > 0.0
    assert 0.0 <= parity["top_k_agreement"] <= 1.0
    # Identical logits are actual parity on KL and top-k, not just argmax.
    same = acs.mean_logit_parity(inc, inc, k=5)
    assert same["kl_nats"] == pytest.approx(0.0, abs=1e-12)
    assert same["top_k_agreement"] == pytest.approx(1.0)
    assert same["argmax_agreement"] == pytest.approx(1.0)


def test_incomplete_lever_without_early_stop_is_refused():
    row = {
        "id": "group_size_1024",
        "weight_space": {"relfro_mean": 0.01, "failed": False},
        "organ_space": None,
        "logit_space": None,
        "early_stop_reason": None,
    }
    with pytest.raises(acs.IncompleteScreen) as caught:
        acs.assert_complete_lever(row)
    assert "organ_space" in str(caught.value)

    row["early_stop_reason"] = "weight-space failed the early-stop bar"
    acs.assert_complete_lever(row)

    row["early_stop_reason"] = None
    row["organ_space"] = {"skipped": True, "reason": "corpus absent"}
    row["logit_space"] = {"skipped": True, "reason": "corpus absent"}
    acs.assert_complete_lever(row)

    # Measured logit-space that only has argmax is a refuse, not a skip.
    row["organ_space"] = {"relfro_mean": 0.01, "failed": False}
    row["logit_space"] = {"argmax_agreement": 1.0}
    with pytest.raises(acs.ArgmaxAloneParityRefuse):
        acs.assert_complete_lever(row)


def test_ls_refit_is_a_real_fit_not_an_analytic_error_model():
    """NEGATIVE CONTROL: reconstructing W and refitting G is not a no-op."""
    rng = np.random.default_rng(1)
    # Incumbent affine-Q2 at G=4 on 8x16. Each group hits q=0 and q=3 so
    # the LS grid is identified; random q-only groups are a different test.
    rows, cols, g0 = 8, 16, 4
    q = rng.integers(0, 4, size=(rows, cols // g0, g0)).astype(np.float32)
    q[:, :, 0] = 0
    q[:, :, -1] = 3
    scale = rng.random((rows, cols // g0)).astype(np.float32) * 0.02 + 0.005
    bias = -1.5 * scale
    W = (q * scale[:, :, None] + bias[:, :, None]).reshape(rows, cols)
    # Same-G LS refit of exact affine-Q2 data must nearly recover W.
    # A min/max-only init misses groups that never hit q=0 and q=3;
    # the moment-matched init is what makes this a real LS refit.
    W_same, s_s, b_s = acs.refit_affine_q2(W, g0, n_iters=6)
    assert W_same.shape == W.shape
    assert s_s.shape == (rows, cols // g0)
    assert acs.relfro(W_same, W) < 0.02
    # Coarser G must move W. If this were an analytic approximation of
    # "error that G would cause" it could return W unchanged.
    W_coarse, s_c, b_c = acs.refit_affine_q2(W, 16, n_iters=6)
    assert W_coarse.shape == W.shape
    assert s_c.shape == (rows, cols // 16)
    assert b_c.shape == (rows, cols // 16)
    assert acs.relfro(W_coarse, W) > acs.relfro(W_same, W)
    assert not np.allclose(W_coarse, W)


def test_u8_aux_is_a_real_encode_not_a_cast():
    rng = np.random.default_rng(2)
    rows, gpr, g = 4, 8, 64
    q = rng.integers(0, 4, size=(rows, gpr, g)).astype(np.float32)
    scale = rng.random((rows, gpr)).astype(np.float32) * 0.02 + 0.004
    bias = rng.normal(size=(rows, gpr)).astype(np.float32) * 0.01
    W = (q * scale[:, :, None] + bias[:, :, None]).reshape(rows, gpr * g)
    what, meta = acs.requant_aux_u8(q, scale, bias)
    assert what.shape == W.shape
    assert meta["n_u8_scale"] == scale.size
    assert meta["n_u8_bias"] == bias.size
    assert meta["endpoint_bytes"] == 8
    # Real codes exist: re-decoding the encode is the reconstruction.
    s_q, lo, hi = acs.u8_log_encode(scale)
    assert s_q.dtype == np.uint8
    assert s_q.size == scale.size
    rec = acs.u8_log_decode(s_q, lo, hi).reshape(scale.shape)
    assert acs.relfro(rec, scale) < 0.05
    # u8 of a non-constant array is not bit-identical to f16.
    assert not np.array_equal(what, W)


def test_aux_byte_model_is_exact_and_matches_the_ladder():
    """aux(G) = 4*n_params/G + 58176. Not a ratio."""
    assert acs.aux_bytes_at_group(64) == AUXILIARY_BYTES_TARGET
    assert acs.bytes_eliminated_at_group(64) == 0
    assert acs.bytes_eliminated_at_group(256) == 802_160_640
    assert acs.bytes_eliminated_at_group(1024) == 1_002_700_800
    assert acs.aux_bytes_at_group(1024) == 4 * acs.N_PARAMS // 1024 + 58_176
    u8 = acs.u8_aux_bytes()
    assert u8["bytes_removed"] == 534_773_760
    assert u8["bytes_added_metadata"] == 192 * 8
    # Scoring goes through executable_economics, which refuses a bare ratio.
    with pytest.raises(ee.IncompleteEconomics):
        ee.score(bytes_removed=acs.bytes_eliminated_at_group(1024))
    scored = acs.score_lever(
        lever_id="group_size_1024",
        bytes_removed=acs.bytes_eliminated_at_group(1024),
        bytes_added=0,
    )
    assert scored["bytes_removed"] == 1_002_700_800
    assert scored["bytes_added_total"] == 0
    assert sum(scored["bytes_added"][k] for k in ee.BYTES_ADDED_FIELDS) == 0
    for key in ee.BYTES_ADDED_FIELDS:
        assert key in scored["bytes_added"]
    # NOT > 1.0. That was the organ-average price, and ECONOMICS_CALIBRATION
    # measured the streams separately: aux_keep_50 sits inside noise while
    # codes_keep_50 does not, so broadcast_aux bills at 0.000 ms/GB. A lever that
    # removes a gigabyte of auxiliary and saves no time is the FINDING; asserting
    # it must save more than a millisecond would pin the overcredit.
    assert scored["predicted_ms_saved"] == pytest.approx(0.0, abs=1e-6)
    # The verdict can still be MATERIAL and that is not a contradiction: the
    # materiality bar is "1 ms OR 5% of model bytes OR a reusable family OR a
    # decisive falsifier", so a lever worth zero milliseconds can be worth
    # RUNNING. Time and worth-running are different axes and the receipt keeps
    # them apart; asserting the time is what this test is for.
    assert scored["verdict"] in {"MATERIAL", "IMMATERIAL"}
    # No hardware-field key named tps.
    assert "tps" not in scored
    assert "predicted_tps" in scored


def test_heterogeneous_bills_variable_group_metadata():
    """Coarse G on the one quiet channel, incumbent G elsewhere, metadata billed."""
    # L63 gate: 17408 x 5120, quiet channel rows 13056-17408 at G=1024.
    runs = [
        {
            "layer": 63,
            "organ": "mlp.gate",
            "n_rows": 13056,
            "n_cols": 5120,
            "group_size": 64,
        },
        {
            "layer": 63,
            "organ": "mlp.gate",
            "n_rows": 4352,
            "n_cols": 5120,
            "group_size": 1024,
        },
    ]
    bill = acs.aux_bytes_heterogeneous(runs)
    assert bill["mixed_tensors"] == 1
    assert bill["n_runs"] == 2
    assert bill["metadata_bytes"] == acs.variable_group_metadata_bytes(
        n_mixed_tensors=1, n_runs=2
    )
    assert bill["metadata_bytes"] > 0
    assert bill["bytes_added_metadata"] == bill["metadata_bytes"]
    # Save is strictly less than uniform G=1024 (the rest of the 1.07 GB stays).
    assert 0 < bill["bytes_removed"] < acs.bytes_eliminated_at_group(1024)
    assert bill["net_bytes_saved"] == bill["bytes_removed"] - bill["metadata_bytes"]
    scored = acs.score_lever(
        lever_id=acs.HETERO_ID,
        bytes_removed=bill["bytes_removed"],
        bytes_added={"metadata": bill["metadata_bytes"]},
    )
    assert scored["bytes_added"]["metadata"] == bill["metadata_bytes"]
    assert scored["bytes_added_total"] == bill["metadata_bytes"]
    assert scored["net_bytes"] == -(bill["bytes_removed"] - bill["metadata_bytes"])
    # Forgetting metadata would overstate the save on the byte ledger.
    # 32 bytes do not move predicted_ms_saved at 0.1 us rounding; the
    # billing is the five-field added-byte record, not the rounded ms.
    forgotten = acs.score_lever(
        lever_id=acs.HETERO_ID,
        bytes_removed=bill["bytes_removed"],
        bytes_added=0,
    )
    assert forgotten["bytes_added_total"] == 0
    assert scored["net_bytes"] == forgotten["net_bytes"] + bill["metadata_bytes"]
    assert forgotten["net_bytes"] < scored["net_bytes"]


def test_allocator_coarsens_only_map_supported_slices():
    alloc = {
        "could_take_fewer_bits": ["L63.mlp.gate.channel.rows_13056_17408"],
        "must_keep_or_gain": [
            "L63.mlp.gate.all",
            "L63.mlp.gate.channel.rows_0_4352",
            "L63.mlp.gate.channel.rows_4352_8704",
            "L63.mlp.gate.channel.rows_8704_13056",
            "L63.mlp.up.all",
        ],
    }
    slices = acs.mlp_sensitivity_slices(alloc)
    gate = acs.allocate_group_runs(
        layer=63,
        organ="mlp.gate",
        n_rows=17408,
        n_cols=5120,
        coarse_group=1024,
        slices=slices,
    )
    by = {(r["row0"], r["row1"]): r for r in gate}
    assert by[(13056, 17408)]["group_size"] == 1024
    assert by[(13056, 17408)]["sensitivity"] == "COULD_TAKE"
    assert by[(0, 4352)]["group_size"] == 64
    assert by[(0, 4352)]["sensitivity"] == "MUST_KEEP"
    # Unmeasured layer: refuse to coarsen.
    quiet = acs.allocate_group_runs(
        layer=7,
        organ="mlp.down",
        n_rows=5120,
        n_cols=17408,
        coarse_group=1024,
        slices=slices,
    )
    assert len(quiet) == 1
    assert quiet[0]["group_size"] == 64
    assert quiet[0]["sensitivity"] == "UNMEASURED"


def test_overlap_relations_use_the_stated_vocabulary():
    rows = acs.overlap_relations()
    acs.assert_overlap_vocab(rows)
    rels = {(r["a"], r["b"], r["relation"]) for r in rows}
    assert (
        "group_size_256",
        "group_size_1024",
        "MUTUALLY_EXCLUSIVE",
    ) in rels
    assert ("quantize_aux_u8", "group_size_1024", "OVERLAPPING") in rels
    assert ("quantize_aux_u8", "group_size_1024", "INTERACTING") in rels
    assert (acs.HETERO_ID, "group_size_1024", "SUBSUMED") in rels
    for row in rows:
        assert row["relation"] in acs.OVERLAP_VOCAB
        assert row["why"]
    with pytest.raises(acs.ScreenRefuse):
        acs.assert_overlap_vocab(
            [{"a": "x", "b": "y", "relation": "ADDITIVE", "why": "nope"}]
        )


def test_error_concentration_flag_fires_when_sensitive_slices_hurt_more():
    quiet = acs.error_concentrates_in_sensitive(
        must_keep_relfro=[0.002, 0.003],
        could_take_relfro=[0.002, 0.004],
    )
    assert quiet["concentrates_in_sensitive"] is False
    spiked = acs.error_concentrates_in_sensitive(
        must_keep_relfro=[0.08, 0.09],
        could_take_relfro=[0.002, 0.003],
    )
    assert spiked["concentrates_in_sensitive"] is True
    assert "sensitivity is high" in spiked["why"]


def test_evidence_tier_is_fitted_or_refuted_never_prospective():
    weight = {"relfro_mean": 0.01, "failed": False}
    organ = {"cosine_mean": 0.999, "relfro_mean": 0.01, "failed": False}
    logit = {
        "kl_nats": 0.01,
        "top_k_agreement": 0.9,
        "failed": False,
        "argmax_is_not_parity": True,
    }
    assert (
        acs.evidence_tier_for(weight=weight, organ=organ, logit=logit, early=None)
        == acs.FITTED_HELDOUT
    )
    assert (
        acs.evidence_tier_for(
            weight={**weight, "failed": True},
            organ=organ,
            logit=logit,
            early=None,
        )
        == acs.REFUTED
    )
    assert (
        acs.evidence_tier_for(
            weight=weight,
            organ=organ,
            logit=logit,
            early="weight-space early-stop",
        )
        == acs.REFUTED
    )
    skipped = {"skipped": True, "reason": "corpus absent"}
    assert (
        acs.evidence_tier_for(
            weight=weight, organ=skipped, logit=skipped, early="corpus absent"
        )
        == acs.REFUTED
    )


def test_fused_decode_is_an_atlas_primitive():
    assert "FusedDecodeCompute" in ATLAS_PRIMITIVES


def test_receipt_writer_refuses_a_tps_number():
    """Sidecar has no GPU authority. A tps field is a hardware claim."""
    with pytest.raises(HardwareClaimError) as caught:
        write_receipt(
            "_AUX_CAPABILITY_SCREEN_PROBE.json",
            {"tps": 3.93, "evidence_class": "STATIC_ONLY"},
            acs.RECORDED_BY,
        )
    assert "tps" in str(caught.value)
    probe = RECEIPTS / "_AUX_CAPABILITY_SCREEN_PROBE.json"
    if probe.is_file():
        probe.unlink()


def test_receipt_if_present_is_a_real_screen_not_a_prospective_copy():
    path = RECEIPTS / acs.RECEIPT
    if not path.is_file():
        pytest.skip("AUX_CAPABILITY_SCREEN.json not written yet")
    doc = json.loads(path.read_text())
    _assert_no_hardware_claims(doc)
    for field in HARDWARE_FIELDS:
        assert field not in doc
    assert doc["schema"] == acs.SCHEMA
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["argmax_alone_is_not_parity"] is True
    assert doc["bench"]["gpu_authority"] is False
    ids = [r["id"] for r in doc["levers"]]
    for cid in acs.LEVER_IDS:
        assert cid in ids
    assert acs.HETERO_ID in ids
    for row in doc["levers"]:
        acs.assert_complete_lever(row)
        assert row["evidence_tier"] in {acs.FITTED_HELDOUT, acs.REFUTED}
        assert row["evidence_tier"] != acs.PROSPECTIVE_ECONOMIC
        assert "economics" in row
        assert row["economics"]["bytes_removed"] == row["bytes_removed"]
        for key in ee.BYTES_ADDED_FIELDS:
            assert key in row["economics"]["bytes_added"]
        logit = row["logit_space"]
        if not (isinstance(logit, dict) and logit.get("skipped")):
            assert "kl_nats" in logit
            assert "top_k_agreement" in logit
            assert logit.get("argmax_is_not_parity") is True
        assert row["consuming_primitive"] in ATLAS_PRIMITIVES
    acs.assert_overlap_vocab(doc["overlap_relations"])
    # Byte figures are the economics model's, not a bare GB ratio.
    g1024 = next(r for r in doc["levers"] if r["id"] == "group_size_1024")
    assert g1024["bytes_removed"] == 1_002_700_800
    raw = ee.score(
        bytes_removed=1_002_700_800,
        bytes_added=0,
        organ="mlp",
        stream_class="broadcast_aux",
        consuming_primitive="FusedDecodeCompute",
    )
    assert g1024["economics"]["predicted_ms_saved"] == pytest.approx(
        ee._r(raw["predicted_ms_saved"], 4)
    )
    assert g1024["economics"]["bytes_removed"] == raw["bytes_removed"]
    hetero = next(r for r in doc["levers"] if r["id"] == acs.HETERO_ID)
    assert hetero["bytes_added"]["metadata"] > 0
    assert hetero["heterogeneous"]["unmeasured_keeps_incumbent"] is True
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    import hashlib

    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
