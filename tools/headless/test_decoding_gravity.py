"""N049 decoding gravity: parent MTP is real, runtimes drop it, binary fails as draft."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from decoding_gravity import (  # noqa: E402
    BINARY_COMPLETE_NS,
    EXPECTED_MTP_TENSORS,
    G057_BREAK_EVEN_MS,
    GENERATOR,
    MTP_CONFIG_KEYS,
    PARENT_BF16,
    Q2F_COMPLETE_NS,
    RECEIPT,
    SCHEMA,
    build,
    expected_accepted_per_pass,
    spec_cycle,
    write,
)

DOCS = None


def docs() -> dict:
    global DOCS
    if DOCS is None:
        built = build()
        write(built)
        DOCS = built
    return DOCS


def _disk() -> dict:
    assert RECEIPT.is_file(), (
        f"missing {RECEIPT} — run python3 tools/headless/decoding_gravity.py"
    )
    return json.loads(RECEIPT.read_text())


def test_generator_writes_schema_and_cpu_discipline():
    d = docs()
    on_disk = _disk()
    assert on_disk["schema"] == SCHEMA
    assert d["schema"] == SCHEMA
    assert on_disk["generated_by"] == GENERATOR
    assert on_disk["hand_authored"] is False
    assert on_disk["did_not_touch_gpu"] is True
    assert on_disk["did_not_run_cargo_or_metal_benchmarks"] is True
    assert on_disk["did_not_load_second_27b"] is True
    assert on_disk["did_not_mutate_noetic_parent_a"] is True
    assert on_disk["did_not_write_under_models"] is True
    assert on_disk["streamed_parent_headers_only"] is True
    assert on_disk["unmeasured_is_absent"] is True


def test_parent_config_cites_real_mtp_keys_not_assumptions():
    assert PARENT_BF16.is_dir(), "qualified parent bf16 must be on this machine"
    cfg = docs()["native_mtp"]["parent_config"]
    assert cfg["present"] is True
    assert cfg["mtp_num_hidden_layers"] == 1
    assert cfg["mtp_use_dedicated_embeddings"] is False
    found = cfg["keys_found"]
    assert any(k.endswith("mtp_num_hidden_layers") for k in found)
    assert any(k.endswith("mtp_use_dedicated_embeddings") for k in found)
    for k in MTP_CONFIG_KEYS:
        assert k not in cfg["keys_absent"]
    # early-exit / Medusa / EAGLE keys are absent — do not invent them
    for k in (
        "early_exit",
        "medusa_num_heads",
        "eagle_num_layers",
        "num_draft_layers",
    ):
        assert k in cfg["keys_absent"]


def test_parent_has_exactly_the_named_mtp_tensors():
    t = docs()["native_mtp"]["parent_tensors"]
    assert t["n_mtp_tensors"] == 15
    assert t["expected_present"] is True
    names = {row["name"] for row in t["tensors"]}
    assert names == set(EXPECTED_MTP_TENSORS)
    assert t["mtp_bytes"] == 849_398_784
    fc = next(r for r in t["tensors"] if r["name"] == "mtp.fc.weight")
    assert fc["shape"] == [5120, 10240]
    assert fc["dtype"] == "BF16"
    # one extra transformer layer, GQA not DeltaNet
    assert any(r["name"].endswith("self_attn.q_proj.weight") for r in t["tensors"])
    assert not any("linear_attn" in r["name"] for r in t["tensors"])


def test_every_production_runtime_drops_mtp_heads():
    n = docs()["native_mtp"]
    assert n["parent_has_heads"] is True
    assert n["runtimes_drop_heads"] is True
    assert n["gravity_q4"]["n_mtp"] == 0
    assert n["mlx_4bit_artifact"]["n_mtp"] == 0
    # config key may survive convert; weights must not
    assert n["mlx_4bit_artifact"]["mtp_num_hidden_layers"] == 1
    assert n["mlx_sanitize"]["strips_mtp"] is True
    assert n["hawking_runtime"]["packer_cannot_ingest_mtp_dot_tensors"] is True
    assert n["hawking_runtime"]["hf_layer_path_skips_mtp_keys"] is True
    assert n["verdict"] == "PARENT_HAS_NATIVE_MTP_HEADS_RUNTIMES_DROP_THEM"


def test_mtp_token_agreement_is_absent_not_a_number():
    alpha = docs()["native_mtp"]["alpha_mtp"]
    assert alpha["value"] is None
    assert alpha["kind"] == "ABSENT"
    assert alpha["absent_reason"]


def test_llama_exposes_draft_mtp_but_our_gguf_is_not_censused_as_kept():
    llama = docs()["native_mtp"]["llama_cpp"]
    assert llama["exposes_draft_mtp"] is True
    assert llama["gguf_q5k"]["present"] is False
    prior = llama["metal_prior"]
    assert prior["kind"] == "CITED"
    assert prior["not_our_27b"] is True
    assert "23752" in prior["url"]


def test_expected_accepted_per_pass_is_honest_at_the_extremes():
    assert expected_accepted_per_pass(0.0, 4) == 1.0
    assert expected_accepted_per_pass(1.0, 4) == 5.0
    # α=0.5, γ=1 → 1 + 0.5 = 1.5
    assert expected_accepted_per_pass(0.5, 1) == pytest.approx(1.5)


def test_token_level_top1_vs_q2f_is_zero_at_position_zero():
    tok = docs()["self_speculative"]["binary_as_draft"]["token_agreement"]
    assert tok["ok"] is True
    assert tok["real_activations"] is True
    assert tok["binary_token_0"] == 271
    assert tok["q2f_token_0"] == 15769
    assert tok["first_token_agree"] is False
    assert tok["earliest_mismatch"] == 0
    assert tok["speculative_accept_on_this_prompt"] == 0
    assert tok["binary_ids"] == [271] * 16
    assert tok["binary_coherent"] is False
    assert "271" in str(tok["binary_coherence_reason"])


def test_local_top1_is_harvested_from_real_x_and_is_not_high():
    loc = docs()["self_speculative"]["binary_as_draft"]["local_top1_on_real_X"]
    assert loc["ok"] is True
    assert loc["real_activations"] is True
    assert loc["not_gaussian"] is True
    assert loc["n_layers"] == 64
    assert loc["n_tokens_used"] == 48
    mean = loc["draft_quality_proxy"]
    # Honest band: N036 measured ~0.37. A fake 0.9 "good draft" must fail.
    assert 0.20 < mean < 0.50, mean
    sw = loc["swiglu_mlp_output"]
    assert sw["n_layers_below_0_25"] >= 8
    # late-middle layers are the injured band
    assert sw["by_band"]["L16-31"]["top1_agreement_mean"] < 0.40


def test_measured_token_alpha_speculation_loses_section_95():
    """Speculative decoding that rejects most proposals LOSES (§95)."""
    cycles = docs()["self_speculative"]["binary_as_draft"]["arithmetic"]
    token_cycles = [c for c in cycles if c["alpha_kind"] == "token_greedy"]
    assert token_cycles
    for c in token_cycles:
        assert c["alpha"] == 0.0
        assert c["wins_vs_baseline"] is False
        assert c["accepted_tokens_per_pass"] == 1.0
        assert c["ratio_vs_baseline"] < 1.0
    # even treating local channel-argmax as if it were token α, γ=4 Leviathan loses
    proxy = [
        c for c in cycles
        if c["alpha_kind"] == "local_swiglu_channel_argmax_proxy"
        and c["gamma"] == 4
        and c["yield_includes_bonus"] is True
    ]
    assert proxy
    assert proxy[0]["wins_vs_baseline"] is False


def test_perfect_accept_control_does_not_clear_g057_break_even():
    assert BINARY_COMPLETE_NS < Q2F_COMPLETE_NS
    assert (BINARY_COMPLETE_NS / 1e6) > G057_BREAK_EVEN_MS
    g = docs()["self_speculative"]["binary_as_draft"]["g057"]
    assert g["binary_clears_that_break_even"] is False
    # G057-style yield=K, α=1, γ=4: still slower than greedy q2f
    c = spec_cycle(
        alpha=1.0,
        gamma=4,
        draft_ns=BINARY_COMPLETE_NS,
        verify_ns=Q2F_COMPLETE_NS,
        baseline_ns=Q2F_COMPLETE_NS,
        yield_includes_bonus=False,
    )
    assert c["wins_vs_baseline"] is False


def test_binary_is_purpose_classified_rejected_generator_failed_draft():
    rows = {r["id"]: r for r in docs()["purpose_reclassification"]}
    b = rows["binary_g64"]
    assert b["rejected_as"] == "final_generator"
    assert b["evaluated_as"] == "draft_against_q2f"
    assert b["draft_verdict"] == "REJECTED_AS_DRAFT"
    assert b["faster_than_q2f_composed_ns"] is True
    # slower bodies cannot be drafts
    assert rows["ternary_5in8_g64"]["draft_verdict"] == "NOT_A_CHEAPER_DRAFT"
    assert rows["shared_binary_k2"]["draft_verdict"] == "NOT_A_CHEAPER_DRAFT"
    assert rows["native_mtp_head"]["draft_verdict"] == "HEADS_PRESENT_RUNTIME_DROPS_THEM"
    assert rows["reduced_depth_early_exit"]["draft_verdict"] == (
        "NO_NATIVE_HEAD_UNMEASURED_ALPHA"
    )


def test_metric_framing_has_both_axes_and_marks_bench_deferred():
    m = docs()["metric_framing"]
    assert m["bench_deferred"] is True
    per_byte = m["accepted_tokens_per_byte"]
    assert per_byte["baseline_q2f"]["accepted_tokens_per_pass"] == 1.0
    bin_row = per_byte["binary_draft_measured_token_alpha"]
    assert bin_row["wins"] is False
    assert bin_row["vs_baseline"] < 1.0
    mtp_row = per_byte["native_mtp_perfect_alpha_ceiling"]
    assert mtp_row["accepted_tokens_per_pass"] == 2.0
    assert mtp_row["alpha_mtp"]["value"] is None
    gpu = m["accepted_tokens_per_gpu_second"]
    assert gpu["baseline_q2f_composed"]["value"] == pytest.approx(
        1e9 / Q2F_COMPLETE_NS
    )
    assert m["g057_break_even"]["clears_break_even"] is False


def test_self_speculative_verdict_does_not_promote_the_binary():
    d = docs()
    assert d["self_speculative"]["verdict"] == (
        "BINARY_REJECTED_AS_GENERATOR_AND_AS_DRAFT"
    )
    body = json.dumps(d).lower()
    assert "promoted" not in body


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
            print(f"ok  {n}")
    print("passed")
