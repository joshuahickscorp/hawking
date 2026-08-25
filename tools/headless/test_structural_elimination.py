"""N050 STRUCTURAL_ELIMINATION: heads / channels / layers / DeltaNet state."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from structural_elimination import (  # noqa: E402
    COPY_COSINE,
    DEAD_REL,
    GQA_GROUP,
    GQA_HEADS,
    GROUP,
    HIDDEN,
    IDENTITY_REL_L2,
    INTERMEDIATE,
    LAYERS,
    PARENT_PARAMS,
    RECEIPT,
    SCHEMA,
    aligned,
    capability_risk,
    candidate,
    col_rms,
    dn_key_head_params,
    ebpw_if_eliminated,
    frobenius_cosine,
    gqa_mixer_params,
    head_pairwise,
    kv_group_params,
    mlp_channel_params,
    mlp_layer_params,
    pairwise_cosine,
    q_head_params,
    rms_report,
    row_rms,
    shape_execution,
    silu,
    token_pair_stats,
)


def test_accounting_q_head_and_kv_group():
    # q+gate+o: 3 * 256 * 5120
    assert q_head_params(1) == 256 * HIDDEN * 3
    assert q_head_params(1) == 3_932_160
    # one GQA group keeps 6:1: 6 Q-heads + 1 K + 1 V
    assert GQA_GROUP == 6
    assert kv_group_params(1) == q_head_params(6) + 256 * HIDDEN * 2
    assert kv_group_params(1) == 26_214_400
    # 64-channel MLP quantum across 64 layers
    assert mlp_channel_params(64, n_layers=LAYERS) == 64 * HIDDEN * 3 * LAYERS
    acc = ebpw_if_eliminated(q_head_params(1) * 16)
    assert acc["ELIMINATED_PARENT_EQUIVALENT_PARAMETERS"] == 3_932_160 * 16
    assert acc["parent_params"] == PARENT_PARAMS
    assert acc["fraction_of_parent"] == pytest.approx(3_932_160 * 16 / PARENT_PARAMS)
    assert acc["bytes_at_bf16"] == acc["ELIMINATED_PARENT_EQUIVALENT_PARAMETERS"] * 2


def test_mlp_layer_and_mixer_mass():
    assert mlp_layer_params() == INTERMEDIATE * HIDDEN * 3
    gqa = gqa_mixer_params()
    # q 12288*5120 + k 1024*5120 + v 1024*5120 + o 5120*6144
    assert gqa == 12288 * HIDDEN + 1024 * HIDDEN * 2 + HIDDEN * 6144
    dn = dn_key_head_params(1)
    assert dn > 0
    # dropping 1 DN key head is cheaper than dropping 1 GQA KV group
    assert dn * 48 < kv_group_params(1) * 16 * 20  # sanity, not a law


def test_head_pairwise_copies_vs_orthogonal():
    rng = np.random.RandomState(0)
    copies = np.repeat(rng.randn(1, 64).astype(np.float32), 8, axis=0)
    pc = head_pairwise(copies.reshape(8, 8, 8))
    assert pc["mean_cosine"] == pytest.approx(1.0, abs=1e-5)
    assert pc["n_pairs_ge_0p99"] == pc["n_pairs"]

    orth = np.eye(8, dtype=np.float32)
    po = pairwise_cosine(orth)
    assert po["mean_cosine"] == pytest.approx(0.0, abs=1e-6)
    assert po["n_pairs_ge_0p50"] == 0
    assert po["null_sigma"] == pytest.approx(1.0 / np.sqrt(8))


def test_scale_trap_is_not_a_copy_test_on_heads():
    """0.01*W is a copy in cosine and is NOT dead in RMS. Cosine is not the GO metric
    for magnitude, but for the copies-vs-not head question it is the right axis."""
    rng = np.random.RandomState(1)
    W = rng.randn(4, 32).astype(np.float32)
    scaled = (0.01 * W).astype(np.float32)
    # pairwise among scaled copies of *different* random heads stays ~0
    stacked = np.stack([W[i] * 0.01 for i in range(4)])
    pc = pairwise_cosine(stacked)
    raw = pairwise_cosine(W)
    assert abs(pc["mean_cosine"] - raw["mean_cosine"]) < 1e-5
    rr = rms_report(row_rms(scaled))
    assert rr["n_dead"] == 0
    # identity copies across heads
    clone = np.stack([W[0], W[0] * 0.01])
    both = pairwise_cosine(clone)
    assert both["mean_cosine"] == pytest.approx(1.0, abs=1e-5)


def test_dead_channel_detector_sees_zeros_not_small_live():
    live = np.ones(16, dtype=np.float32)
    live[3] = 0.0
    live[5] = 1e-12  # below DEAD_REL * median (median=1)
    rr = rms_report(live)
    assert rr["n_dead"] >= 2
    assert 3 in range(16)
    # a 1e-4 tail is LOW not DEAD (LOW_REL=1e-3 of median=1)
    tail = np.ones(32, dtype=np.float32)
    tail[0] = 1e-4
    low = rms_report(tail)
    assert low["n_dead"] == 0
    assert low["n_low"] == 1
    assert DEAD_REL < 1e-3


def test_col_rms_matches_down_proj_channel_axis():
    rng = np.random.RandomState(2)
    W = rng.randn(8, 16).astype(np.float32)
    W[:, 4] = 0
    c = col_rms(W)
    assert c.shape == (16,)
    assert c[4] == pytest.approx(0.0)
    assert c[0] > 0


def test_shape_ragged_is_worse_aligned_constant_change_is_mixed():
    ragged = shape_execution(
        name="one_channel",
        old={"intermediate": INTERMEDIATE, "rows": INTERMEDIATE},
        new={"intermediate": INTERMEDIATE - 1, "rows": INTERMEDIATE - 1},
        remaining_tensor_shapes="RAGGED",
    )
    assert (INTERMEDIATE - 1) % GROUP == 63
    assert ragged["executes"] == "WORSE"
    assert ragged["divisible_by_group64"] is False

    quantum = shape_execution(
        name="64_channel",
        old={"intermediate": INTERMEDIATE, "rows": INTERMEDIATE},
        new={"intermediate": INTERMEDIATE - 64, "rows": INTERMEDIATE - 64, "intermediate": INTERMEDIATE - 64},
        remaining_tensor_shapes="SPECIALIZED_CONST_CHANGED",
    )
    assert aligned(INTERMEDIATE - 64)
    assert quantum["executes"] == "MIXED"
    assert quantum["specialized_constant_changed"] is True

    heads23 = shape_execution(
        name="q23",
        old={"gqa_heads": 24, "gqa_kv_heads": 4, "q_rows": 12288},
        new={"gqa_heads": 23, "gqa_kv_heads": 4, "q_rows": 23 * 2 * 256},
        remaining_tensor_shapes="RAGGED",
    )
    assert heads23["executes"] == "WORSE"
    assert any("grouping" in s.lower() or "GQA" in s for s in heads23["issues"])

    layer = shape_execution(
        name="drop_layer",
        old={"layers": 64, "hidden": HIDDEN, "intermediate": INTERMEDIATE},
        new={"layers": 63, "hidden": HIDDEN, "intermediate": INTERMEDIATE},
        remaining_tensor_shapes="SAME",
        work_removed=True,
    )
    assert layer["executes"] == "BETTER"

    noop = shape_execution(
        name="noop",
        old={"layers": 64},
        new={"layers": 64},
        remaining_tensor_shapes="SAME",
        work_removed=False,
    )
    assert noop["executes"] == "SAME"


def test_aligned_group64_and_17408():
    assert aligned(INTERMEDIATE, GROUP)
    assert aligned(HIDDEN, GROUP)
    assert aligned(12288, GROUP)
    assert aligned(6144, GROUP)
    assert not aligned(INTERMEDIATE - 1, GROUP)
    assert aligned(INTERMEDIATE - 64, GROUP)


def test_token_pair_identity_and_orthogonal():
    rng = np.random.RandomState(3)
    X = rng.randn(32, 64).astype(np.float32)
    ident = token_pair_stats(X, X)
    assert ident["mean_cosine"] == pytest.approx(1.0, abs=1e-5)
    assert ident["mean_rel_l2"] == pytest.approx(0.0, abs=1e-5)
    assert ident["near_identity"] is True
    Y = rng.randn(32, 64).astype(np.float32)
    far = token_pair_stats(X, Y)
    assert far["near_identity"] is False
    assert far["mean_cosine"] < 0.5


def test_frobenius_cosine_copies_and_orthogonal():
    rng = np.random.RandomState(4)
    A = rng.randn(16, 32).astype(np.float32)
    assert frobenius_cosine(A, A) == pytest.approx(1.0, abs=1e-5)
    assert frobenius_cosine(A, 0.01 * A) == pytest.approx(1.0, abs=1e-5)
    B = rng.randn(16, 32).astype(np.float32)
    # two independent Gaussian matrices are not copies
    assert abs(frobenius_cosine(A, B)) < 0.3


def test_silu_zero_and_positive():
    z = silu(np.array([0.0, 10.0, -10.0], dtype=np.float32))
    assert z[0] == pytest.approx(0.0, abs=1e-6)
    assert z[1] > 9.0
    assert z[2] < 0.0
    assert abs(z[2]) < 0.1


def test_candidate_requires_capability_for_allowed():
    risk = capability_risk(
        level="HIGH",
        reason="no capability",
        section_109=True,
        what_would_confirm="generation",
    )
    shape = shape_execution(
        name="x",
        old={"rows": 64},
        new={"rows": 64},
        remaining_tensor_shapes="SAME",
        work_removed=False,
    )
    c = candidate(
        id="toy",
        family="mlp_channel",
        what="toy",
        n_params=100,
        data_verdict="EVIDENCE_SUPPORTED",
        why="zeros",
        risk=risk,
        shape=shape,
        evidence={},
        citations=["receipts/headless/BYTES_FRONTIER.json"],
    )
    assert c["elimination_allowed"] is False
    assert c["accounting"]["ELIMINATED_PARENT_EQUIVALENT_PARAMETERS"] == 100
    assert c["capability_risk"]["removal_requires_capability_evidence"] is True
    assert c["shape_execution"]["executes"] in {"BETTER", "WORSE", "MIXED", "SAME"}


def _receipt() -> dict:
    assert RECEIPT.is_file(), (
        f"missing {RECEIPT} — run python3 tools/headless/structural_elimination.py"
    )
    return json.loads(RECEIPT.read_text())


def test_receipt_schema_cpu_and_no_second_27b():
    doc = _receipt()
    assert doc["schema"] == SCHEMA
    assert doc["obligation"] == "N050"
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_touch_gpu"] is True
    assert doc["did_not_run_cargo_or_metal_bench"] is True
    assert doc["did_not_mutate_noetic_parent_a"] is True
    assert doc["parent"]["opened_noetic_parent_a"] is False
    assert doc["parent"]["params"] == PARENT_PARAMS
    assert doc["capture"]["not_a_second_27b_decode"] is True
    assert "mlx" not in doc["gpu_touch"]["this_module_imports"]
    assert doc["gpu_touch"]["imported_mlx"] is False


def test_receipt_remeasures_q_heads_and_refutes_shared_head():
    doc = _receipt()
    gqa = doc["attention_heads"]
    assert gqa["status"] == "MEASURED"
    assert gqa["n_layers"] == 16
    h = gqa["headline"]
    # L3 prior 0.022. All-layer mean must stay far from copies.
    l3 = h["q_mean_cosine_L3"]
    assert l3 is not None
    assert abs(l3 - 0.021602977067232132) < 0.005
    assert h["prior_match_L3"] is True
    assert abs(h["q_mean_cosine_all_layers"]) < 0.15
    assert h["q_n_pairs_ge_0p99"] == 0
    for key in ("k_mean_cosine_all_layers", "v_mean_cosine_all_layers", "o_mean_cosine_all_layers"):
        assert h[key] is not None
        assert abs(h[key]) < 0.15, key
    # L3 and L63 both present
    assert "3" in gqa["layers"] and "63" in gqa["layers"]
    for proj in ("q", "gate", "k", "v", "o"):
        assert gqa["layers"]["3"][proj]["mean_cosine"] is not None


def test_receipt_candidates_have_params_risk_shape_and_verdict():
    doc = _receipt()
    cands = doc["candidates"]
    assert len(cands) >= 8
    ids = {c["id"] for c in cands}
    for need in (
        "gqa_shared_q_heads",
        "gqa_drop_one_q_head",
        "gqa_drop_one_kv_group",
        "mlp_drop_dead_channels",
        "mlp_drop_low_magnitude_64_quantum",
        "mlp_drop_one_unaligned_channel",
        "drop_near_identity_layer",
        "deltanet_collapse_key_heads",
        "deltanet_drop_one_key_head_state",
        "experts_moe",
    ):
        assert need in ids, need
    for c in cands:
        acc = c["accounting"]
        assert "ELIMINATED_PARENT_EQUIVALENT_PARAMETERS" in acc
        assert acc["parent_params"] == PARENT_PARAMS
        assert "level" in c["capability_risk"]
        assert c["capability_risk"]["removal_requires_capability_evidence"] is True
        assert c["shape_execution"]["executes"] in {"BETTER", "WORSE", "MIXED", "SAME"}
        assert c["data_verdict"] in {"REFUTED", "EVIDENCE_SUPPORTED", "INCONCLUSIVE"}
        assert c["elimination_allowed"] is False
        assert c["citations"], c["id"]

    by_id = {c["id"]: c for c in cands}
    assert by_id["gqa_shared_q_heads"]["data_verdict"] == "REFUTED"
    assert by_id["experts_moe"]["data_verdict"] == "REFUTED"
    assert by_id["mlp_drop_one_unaligned_channel"]["data_verdict"] == "REFUTED"
    assert by_id["mlp_drop_one_unaligned_channel"]["shape_execution"]["executes"] == "WORSE"
    assert by_id["gqa_drop_one_q_head"]["shape_execution"]["executes"] == "WORSE"
    # 23 Q heads break 6:1
    assert by_id["gqa_drop_one_q_head"]["accounting"]["ELIMINATED_PARENT_EQUIVALENT_PARAMETERS"] == q_head_params(1) * 16


def test_receipt_mlp_and_layers_and_deltanet_measured():
    doc = _receipt()
    mlp = doc["mlp_channels"]
    assert mlp["status"] == "MEASURED"
    assert mlp["n_layers"] == 64
    assert mlp["n_channels_per_layer"] == INTERMEDIATE
    assert "frac_dead" in mlp
    assert mlp["adjacent_gate_frobenius_cosine"]["n_pairs"] == 63
    # energy-profile cosine is NOT this number; flattened W copies would be ~1
    assert mlp["adjacent_gate_frobenius_cosine"]["n_pairs_ge_0p99"] == 0
    assert abs(mlp["adjacent_gate_frobenius_cosine"]["mean"]) < COPY_COSINE

    layers = doc["layer_redundancy"]
    assert layers["status"] in {"MEASURED", "ABSENT"}
    if layers["status"] == "MEASURED":
        assert layers["n_pairs"] == 63
        # residual streams can be similar; identity requires tiny rel L2
        assert layers["n_near_identity"] == 0
        assert layers["mean_rel_l2"] > IDENTITY_REL_L2 or layers["max_cosine"] < 0.995

    dn = doc["deltanet_state"]
    assert dn["status"] == "MEASURED"
    assert dn["n_layers"] == 48
    assert dn["cited"]["n_static_tensors_that_duplicate_state"] == 0
    assert abs(dn["headline"]["q_mean_cosine"]) < 0.15
    assert dn["headline"]["n_dead_key_heads"] == 0


def test_receipt_summary_and_section39_and_citations():
    doc = _receipt()
    s = doc["summary"]
    assert s["n_elimination_allowed"] == 0
    assert s["n_REFUTED"] >= 4
    assert "gqa_shared_q_heads" in s["REFUTED"]
    assert doc["verdict"]["shared_q_heads"] == "REFUTED"
    assert doc["verdict"]["any_elimination_allowed"] is False
    assert "smaller badly-shaped" in doc["section_39"]["law"]
    assert "BYTES_FRONTIER.json" in doc["section_39"]["citations"][0] or any(
        "BYTES_FRONTIER" in c for c in doc["section_39"]["citations"]
    )
    for rel in (
        "receipts/headless/ORGAN_FRONTIERS.json",
        "receipts/headless/BYTES_FRONTIER.json",
        "receipts/headless/NOETIC_DELTANET_DESIGN.json",
        "receipts/headless/C1SHAREDBASIS_DESIGN.json",
    ):
        assert rel in doc["citations"]
        assert (REPO / rel).is_file()


def test_receipt_watches_the_known_traps():
    doc = _receipt()
    watched = " ".join(w["what"] + w["why"] for w in doc["what_i_watched_fail"])
    assert "0.997" in watched or "energy-profile" in watched
    assert "0.022" in watched
    assert "109" in watched or "capture_diverse2" in watched
    assert "fewer parameters" in watched or "§39" in watched
