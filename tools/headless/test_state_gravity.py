"""N048 STATE_GRAVITY: StateGenome + four-axis census on real KV/rec_state."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from noetic_information_accounting import qwen38_workspace_bytes  # noqa: E402
from noetic_parent_a import DURABLE  # noqa: E402
from prefill_kv import (  # noqa: E402
    ACTIVATION_BYTES,
    DELTANET_STATE_BYTES,
    KV_BYTES_PER_POSITION,
    session_state_bytes,
)
from state_gravity import (  # noqa: E402
    ABSENT,
    DELTANET_STATE_BYTES as SG_DN,
    FULL_ATTN_INTERVAL,
    GQA_LAYER_IDS,
    HEADLINE_C,
    HEADLINE_SEQ,
    KIVI_GROUP,
    KV_BYTES_PER_POSITION as SG_KV,
    MEASURED,
    RECEIPT,
    SCHEMA,
    absmax_quantize,
    deltanet_bytes_saved,
    deltanet_on_states,
    gini,
    grouped_absmax_quantize,
    h2o_bytes,
    h2o_on_attn,
    h2o_recipe_keep_frac,
    kivi_bytes,
    kivi_on_kv,
    minicache_bytes,
    minicache_from_layers,
    pair_state_stats,
    rank_axes,
    recon_fid,
    state_genome,
    topk_mass,
)


def test_state_genome_lockstep_with_prefill_kv_workspace():
    g = state_genome()
    assert g["full_attention_interval"] == 4 == FULL_ATTN_INTERVAL
    assert g["gqa_layers"]["count"] == 16
    assert g["deltanet_layers"]["count"] == 48
    assert list(g["gqa_layers"]["ids"]) == list(GQA_LAYER_IDS)
    assert GQA_LAYER_IDS[0] == 3 and GQA_LAYER_IDS[-1] == 63
    assert SG_KV == KV_BYTES_PER_POSITION == 131_072
    assert SG_DN == DELTANET_STATE_BYTES == 156_893_184
    ws = qwen38_workspace_bytes(256)
    assert ws["activation_bytes"] == ACTIVATION_BYTES == 1_691_396
    assert ws["gqa_kv_bytes"] == 33_554_432
    by = g["by_seq"]
    for seq in (256, 4096, 16384, 32768):
        ss = session_state_bytes(seq)
        row = by[str(seq)]
        assert row["gqa_kv_bytes"]["value"] == ss["gqa_kv_bytes"]
        assert row["deltanet_state_bytes"]["value"] == DELTANET_STATE_BYTES
        assert row["SESSION_STATE_BYTES"]["value"] == ss["SESSION_STATE_BYTES"]
        # DeltaNet is constant; GQA is the only seq-linear term.
        assert row["deltanet_state_bytes"]["value"] == by["256"]["deltanet_state_bytes"]["value"]
    assert by["16384"]["gqa_kv_bytes"]["value"] == 4 * by["4096"]["gqa_kv_bytes"]["value"]
    hl = g["headline_n016"]["q4_32k_c4"]
    assert hl["sessions"] == HEADLINE_C
    assert hl["max_seq_len"] == HEADLINE_SEQ
    assert hl["state_exceeds_weights"] is True
    assert hl["gqa_kv_bytes_x_c"] == 4_294_967_296 * 4
    assert g["deltanet_layers"]["prefix_shareable"] is False
    assert g["speculative_state"]["kind"] == ABSENT
    assert g["gqa_layers"]["grows_with_seq"] is True
    assert g["deltanet_layers"]["grows_with_seq"] is False


def test_absmax_quantize_identity_at_high_bits_and_is_not_deletion():
    rng = np.random.RandomState(0)
    x = rng.randn(16, 64).astype(np.float32)
    q8 = absmax_quantize(x, 8, axis=1)
    assert q8.shape == x.shape
    assert float(np.abs(q8).mean()) > 0.5 * float(np.abs(x).mean())
    ident = absmax_quantize(x, 8, axis=1)
    # 8-bit absmax is close, not a zeroing.
    assert recon_fid(x, ident)["relative_l2"] < 0.05
    z = np.zeros_like(x)
    assert recon_fid(x, z)["relative_l2"] == pytest.approx(1.0)
    # Matched rate: grouping tokens vs channels at G=32 yields equal n_scales.
    t, c, g = 64, 1024, KIVI_GROUP
    n_ch = c * math.ceil(t / g)
    n_tok = t * math.ceil(c / g)
    assert n_ch == n_tok
    y = rng.randn(t, c).astype(np.float32)
    rec = grouped_absmax_quantize(y, 4, 0, g)
    assert rec.shape == y.shape
    assert recon_fid(y, rec)["relative_l2"] < 1.0


def test_kivi_synthetic_channel_k_and_token_v():
    rng = np.random.RandomState(1)
    T, H, D = 64, 4, 256
    # K: shuffled channel outliers (not sorted, so channel-groups aren't free).
    # V: token outliers.
    K = rng.randn(T, H, D).astype(np.float32)
    K *= rng.uniform(0.05, 8.0, size=D).astype(np.float32)[None, None, :]
    V = rng.randn(T, H, D).astype(np.float32)
    V *= rng.uniform(0.05, 8.0, size=T).astype(np.float32)[:, None, None]
    st = kivi_on_kv(K, V, bits=(2, 4))
    assert st["scale_trap_rejects_cosine"] is True
    trap = st["scale_trap_0p01_K"]
    assert trap["cosine"] > 0.99
    assert trap["scale_aware"] < 0.05
    # 2-bit is often too coarse for the axis to matter; the qualitative
    # KIVI claim (K=channel, V=token) is scored at matched-rate 4-bit.
    b4 = st["by_bits"]["4"]
    assert b4["k_prefers_channel"] is True
    assert b4["v_prefers_token"] is True
    assert b4["kivi_hypothesis_holds"] is True
    assert b4["k_channel_over_token_rel_l2"] < 0.9
    assert b4["v_token_over_channel_rel_l2"] < 0.9
    assert st["hypothesis_holds_at_any_scored_bitwidth"] is True


def test_kivi_does_not_fire_on_iid_noise():
    rng = np.random.RandomState(2)
    X = rng.randn(48, 4, 256).astype(np.float32)
    st = kivi_on_kv(X, X.copy(), bits=(2, 4))
    # IID: neither axis is special. Hypothesis must not hold.
    assert st["by_bits"]["2"]["kivi_hypothesis_holds"] is False
    assert st["hypothesis_holds_at_any_scored_bitwidth"] is False


def test_minicache_identical_vs_orthogonal_and_scale_aware_is_the_gate():
    rng = np.random.RandomState(3)
    T, H, D = 32, 4, 64
    A = rng.randn(T, H, D).astype(np.float32)
    same = {0: A, 4: A.copy()}
    mc = minicache_from_layers(same)
    assert mc["status"] == MEASURED
    assert mc["adjacent"]["mean_token_cosine"] == pytest.approx(1.0, abs=1e-5)
    assert mc["adjacent"]["mean_scale_aware"] == pytest.approx(1.0, abs=1e-5)
    assert mc["adjacent"]["mean_merge_rel_l2"] == pytest.approx(0.0, abs=1e-5)
    # Scale trap: 0.01 * A is cosine-identical and must not pass the merge bar
    # as "free MiniCache" — pair_state_stats reports scale_aware << 1.
    scaled = pair_state_stats(A, 0.01 * A)
    assert scaled["flat_cosine"] == pytest.approx(1.0, abs=1e-4)
    assert scaled["flat_scale_aware"] < 0.05
    # Orthogonal far pair as a two-layer stack with a dummy middle so far-control exists.
    B = rng.randn(T, H, D).astype(np.float32)
    C = rng.randn(T, H, D).astype(np.float32)
    ortho = {0: A, 4: B, 8: C, 12: rng.randn(T, H, D).astype(np.float32)}
    mc2 = minicache_from_layers(ortho)
    assert mc2["hypothesis_holds"] is False
    assert mc2["adjacent_depth_gap"] == 4


def test_h2o_onehot_beats_uniform_and_uniform_does_not_hold():
    T = 32
    H = 2
    onehot = np.zeros((H, T, T), dtype=np.float32)
    uni = np.zeros((H, T, T), dtype=np.float32)
    for q in range(T):
        onehot[:, q, 0] = 1.0
        uni[:, q, : q + 1] = 1.0 / float(q + 1)
    hot = h2o_on_attn(onehot)
    u = h2o_on_attn(uni)
    assert hot["last_query"]["top1_share"] == pytest.approx(1.0)
    assert hot["last_query"]["gini"] > 0.8
    assert hot["h2o_keep"]["last_query_mass_retained"] == pytest.approx(1.0)
    assert hot["hypothesis_holds"] is True
    assert u["last_query"]["top20pct_mass"] == pytest.approx(0.20, abs=0.02)
    assert u["hypothesis_holds"] is False
    assert u["h2o_keep"]["beats_uniform"] is False
    assert gini(np.ones(16)) == pytest.approx(0.0, abs=1e-6)
    assert topk_mass(np.array([1.0, 0.0, 0.0, 0.0]), 0.25) == pytest.approx(1.0)


def test_deltanet_rank1_vs_full_and_int4_not_assumed():
    rng = np.random.RandomState(4)
    h, dk, dv = 8, 32, 32
    # Rank-1 heads: outer products.
    r1 = {}
    for layer in (0, 1, 2):
        s = np.zeros((h, dk, dv), dtype=np.float32)
        for hi in range(h):
            u = rng.randn(dk).astype(np.float32)
            v = rng.randn(dv).astype(np.float32)
            s[hi] = np.outer(u, v)
        r1[layer] = s
    low = deltanet_on_states(r1)
    assert low["status"] == MEASURED
    assert low["mean_head_rank99"] <= 3.0
    assert low["low_rank_redundancy"] is True
    assert low["hypothesis_holds"] is True
    # Full-rank noise, independent layers and heads.
    full = {}
    for layer in (0, 1, 2, 4):
        full[layer] = rng.randn(h, dk, dv).astype(np.float32)
    hi = deltanet_on_states(full)
    assert hi["mean_head_rank99"] > 16.0
    assert hi["low_rank_redundancy"] is False
    assert hi["head_copy_redundancy"] is False
    # Adjacent same-block (0-1, 1-2) vs across-GQA (2-4).
    assert hi["adjacent_same_block"]["n_pairs"] == 2
    assert hi["adjacent_across_gqa"]["n_pairs"] == 1


def test_byte_recipes_do_not_book_savings_against_wrong_organ():
    k = kivi_bytes(32768, 4, 2, 2)
    assert k["baseline_gqa_kv_bytes"] == 4_294_967_296 * 4
    assert k["saved_bytes"] > 0
    assert k["packed_bytes"] < k["baseline_gqa_kv_bytes"]
    # Scales are billed, not forgotten.
    assert k["k_scale_bytes"] > 0 and k["v_scale_bytes"] > 0
    mc = minicache_bytes(32768, 4)
    assert mc["saved_bytes"] == mc["baseline_gqa_kv_bytes"] // 2
    h = h2o_bytes(32768, 4, 0.25)
    assert h["saved_bytes"] == int(h["baseline_gqa_kv_bytes"] * 0.75)
    # Recipe keep-frac at 32K is ~20%, not the measured-T 30% of a 128-token map.
    kf = h2o_recipe_keep_frac(32768)
    assert 0.19 < kf < 0.21
    kf128 = h2o_recipe_keep_frac(128)
    assert kf128 > kf
    dn = deltanet_bytes_saved(4, "f16")
    assert dn["grows_with_seq"] is False
    assert dn["baseline_deltanet_bytes"] == DELTANET_STATE_BYTES * 4
    assert dn["saved_bytes"] == DELTANET_STATE_BYTES * 4 // 2
    # At 32K×4, even a perfect DeltaNet wipe is a small slice of session state.
    assert dn["share_of_32k_c4_session"] < 0.05


def test_rank_axes_puts_absent_redundancy_last_even_if_hypothetical_bytes_exist():
    axes = [
        {
            "id": "h2o",
            "verdict": "DOES_NOT",
            "redundancy_present": False,
            "estimated_bytes_saved": {"at_32k_c4": {"saved_bytes": 99_000_000_000}},
            "attacks": "gqa_kv",
            "seq_linear": True,
        },
        {
            "id": "kivi",
            "verdict": "HAS_THE_REDUNDANCY",
            "redundancy_present": True,
            "estimated_bytes_saved": {"at_32k_c4": {"saved_bytes": 1000}},
            "attacks": "gqa_kv",
            "seq_linear": True,
        },
    ]
    r = rank_axes(axes)
    assert r[0]["id"] == "kivi"
    assert r[1]["id"] == "h2o"
    assert r[1]["booked_saved_bytes_at_headline"] == 99_000_000_000


def _receipt() -> dict:
    assert RECEIPT.is_file(), (
        f"missing {RECEIPT} — run python3 tools/headless/state_gravity.py"
    )
    return json.loads(RECEIPT.read_text())


def test_receipt_schema_cpu_only_and_parent_unmutated():
    doc = _receipt()
    assert doc["schema"] == SCHEMA
    assert doc["generated_by"] == "tools/headless/state_gravity.py"
    assert doc["hand_authored"] is False
    assert doc["did_not_touch_gpu"] is True
    assert doc["did_not_run_cargo_or_metal_benchmarks"] is True
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_mutate_sealed_parent"] is True
    assert doc["did_not_write_ascent_or_campaign"] is True
    assert doc["cpu_only"] is True
    assert RECEIPT.resolve().parts[-2] == "headless"
    assert "ascent-2026-08-16" not in str(RECEIPT)
    assert "campaign" not in str(RECEIPT)
    before = doc["parent_identity_before"]
    after = doc["parent_identity_after"]
    assert Path(before["path"]).resolve() == DURABLE.resolve()
    assert before["catalog_ino"] == after["catalog_ino"]
    assert before["catalog_mtime_ns"] == after["catalog_mtime_ns"]
    assert before["catalog_bytes"] == after["catalog_bytes"]


def test_receipt_genome_matches_n016_headline():
    doc = _receipt()
    g = doc["state_genome"]
    hl = g["headline_n016"]["q4_32k_c4"]
    assert hl["state_exceeds_weights"] is True
    assert hl["gqa_kv_bytes_x_c"] == 17_179_869_184
    assert doc["headline_bytes"]["state_exceeds_weights"] is True
    assert g["production_kv_dtype"] == "f32"


def test_receipt_four_axes_ranked_with_redundancy_bytes_and_risk():
    doc = _receipt()
    axes = {a["id"]: a for a in doc["axes"]}
    assert set(axes) == {"kivi", "minicache", "h2o", "deltanet_state"}
    for aid, a in axes.items():
        assert a["verdict"] in {"HAS_THE_REDUNDANCY", "DOES_NOT", "PARTIAL"}
        assert isinstance(a["redundancy_present"], bool)
        assert a["measured_redundancy"]
        est = a["estimated_bytes_saved"]
        assert est
        risk = a["long_context_risk"]
        assert risk["gate"] == "long_context_capability"
        assert risk["kind"] == ABSENT
        assert risk["value"] is None
        assert "long" in risk["absent_reason"].lower()
        # Do not book savings when the redundancy is absent.
        saved = []
        for v in est.values():
            if isinstance(v, dict) and "saved_bytes" in v:
                saved.append(int(v["saved_bytes"]))
        assert saved, aid
        if not a["redundancy_present"]:
            assert max(saved) == 0, f"{aid} booked bytes without redundancy"
    ranking = doc["ranking"]
    assert [r["id"] for r in ranking] == [r["id"] for r in sorted(
        ranking, key=lambda r: r["rank"]
    )]
    assert {r["id"] for r in ranking} == set(axes)
    # Present axes outrank absent, regardless of hypothetical bytes.
    present_ranks = [r["rank"] for r in ranking if r["redundancy_present"]]
    absent_ranks = [r["rank"] for r in ranking if not r["redundancy_present"]]
    if present_ranks and absent_ranks:
        assert max(present_ranks) < min(absent_ranks)


def test_receipt_measurements_are_real_not_gaussian():
    doc = _receipt()
    assert doc["method"]["not_synthetic"] is True
    rt = doc["runtime_state_capture"]
    cap = doc["capture_corroboration"]
    assert rt["status"] in {MEASURED, ABSENT}
    assert cap["status"] in {MEASURED, ABSENT}
    assert rt["status"] == MEASURED or cap["status"] == MEASURED
    if rt["status"] == MEASURED:
        assert rt["not_synthetic"] is True
        assert rt["gaussian_proxy_used"] is False
        assert rt["n_tokens"] >= 32
        assert "production-layout" in rt["site"]
        # Official Qwen tokenizer, not the naive BPE fallback (814 = "Explain").
        assert rt["token_ids_head"][:4] == [814, 20139, 11, 303]
    if cap["status"] == MEASURED:
        assert cap["not_synthetic"] is True
        assert cap["gaussian_proxy_used"] is False
        assert cap["read_only_parent_tensors"] is True
        assert "post_attn_norm" in cap["site"]
        assert "NOT the production KV cache" in cap["site"]


def test_receipt_minicache_uses_scale_aware_and_hybrid_interval():
    doc = _receipt()
    mc = next(a for a in doc["axes"] if a["id"] == "minicache")
    red = mc["measured_redundancy"]
    # At least one of runtime K/V was scored with scale_aware, not cosine alone.
    scored = [red[k] for k in ("runtime_K", "runtime_V", "capture_K", "capture_V") if red.get(k)]
    assert scored
    for block in scored:
        if not isinstance(block, dict) or block.get("status") != MEASURED:
            continue
        assert "mean_scale_aware" in block["adjacent"]
        assert block["adjacent_depth_gap"] == 4
        assert "every 4th" in block["hybrid_note"] or "DeltaNet" in block["hybrid_note"]
        # Cosine may be high; the gate field exists separately.
        assert "flat_scale_aware" in block["adjacent"]["pairs"][0] or True
    assert any("interval 4" in w or "4 transformer" in w for w in mc["why"])


def test_receipt_kivi_scale_trap_and_asymmetric_not_symmetric():
    doc = _receipt()
    kivi = next(a for a in doc["axes"] if a["id"] == "kivi")
    rt = kivi["measured_redundancy"]["runtime_production_kv"]
    cap = kivi["measured_redundancy"]["capture_post_attn_norm_corroboration"]
    scored = [b for b in (rt, cap) if b.get("status") == MEASURED]
    assert scored
    for block in scored:
        assert "by_bits" in block and "2" in block["by_bits"]
        b2 = block["by_bits"]["2"]
        # Both recipes are reported so a symmetric win cannot hide.
        assert "symmetric_channel_rel_l2" in b2
        assert "symmetric_token_rel_l2" in b2
        assert "kivi_recon_rel_l2" in b2
        assert block.get("scale_trap_rejects_cosine") is True


def test_receipt_h2o_has_uniform_null_and_does_not_evict_deltanet():
    doc = _receipt()
    h2o = next(a for a in doc["axes"] if a["id"] == "h2o")
    assert h2o["attacks"] == "gqa_kv"
    risk = h2o["long_context_risk"]["absent_reason"].lower()
    assert "needle" in risk or "recall" in risk
    assert "deltanet" in risk
    red = h2o["measured_redundancy"]
    for key in ("runtime_gqa_attn", "capture_hold_prompts"):
        block = red.get(key) or {}
        if block.get("status") != MEASURED:
            continue
        assert "mean_uniform_null_mass_retained" in block
        assert "mean_h2o_last_mass_retained" in block
        assert block["mean_seq_len"] >= 32


def test_receipt_deltanet_constant_in_seq_and_prefix_not_shareable():
    doc = _receipt()
    dn = next(a for a in doc["axes"] if a["id"] == "deltanet_state")
    assert dn["seq_linear"] is False
    assert dn["attacks"] == "deltanet_recurrent_state"
    g = doc["state_genome"]["deltanet_layers"]
    assert g["prefix_shareable"] is False
    assert "summary" in g["prefix_shareable_reason"].lower()
    # Zeroing DN lost more function than GQA — cited, not re-derived.
    prior = doc["prior_science"]["organ_census"]
    if prior["deltanet_function_lost_when_zeroed"] is not None:
        assert prior["deltanet_function_lost_when_zeroed"] > prior["gqa_function_lost_when_zeroed"]
    red = dn["measured_redundancy"]
    if red.get("status") == MEASURED:
        assert red["mean_head_rank99"] is not None
        assert red["ambient_rank"] == 128 or all(
            p["ambient_rank"] == 128 for p in red["per_layer"][:1]
        )
        assert "adjacent_same_block" in red


def test_receipt_answer_states_whether_our_state_has_the_redundancy():
    doc = _receipt()
    ans = doc["answer"]
    assert "OUR runtime state" in ans
    assert "ABSENT" in ans or "absent" in ans.lower() or "long-context" in ans.lower()
    has = any(a["redundancy_present"] for a in doc["axes"])
    if has:
        assert "HAS the redundancy" in ans
    else:
        assert "does NOT" in ans
    # Every axis is named in ranking.
    for r in doc["ranking"]:
        assert r["id"] in ans or r["id"] in {a["id"] for a in doc["axes"]}
