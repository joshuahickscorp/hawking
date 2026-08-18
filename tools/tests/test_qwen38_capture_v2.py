"""Capture v2: adequacy gate must FAIL on the 256-token cube."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.qwen38_capture_v2 import (  # noqa: E402
    HIDDEN,
    INTERMEDIATE,
    MIXER_IN,
    N_TOKENS,
    NS014_DIM,
    NS014_RPD,
    NS014_ROWS,
    SITES,
    V1_N_TOKENS,
    AdequacyRefused,
    adequacy_from_capture,
    adequacy_gate,
    build_capture_plan,
    build_prompt_plan,
    format_plan,
    gated_score,
    prompt_holdout,
    resolve_v1_capture,
    rows_per_dim,
)


def _load_v1_meta() -> dict:
    root = resolve_v1_capture()
    assert root is not None, (
        "existing 256-token capture-result.json is required as the negative control"
    )
    meta = json.loads((root / "capture-result.json").read_text())
    assert meta["n_tokens"] == V1_N_TOKENS
    return meta


def _synthetic_adequate_meta() -> dict:
    plan = build_prompt_plan()
    return {
        "schema": "hawking.ascension.qwen38_activation_capture.v2",
        "status": "CAPTURED_REAL_BF16_MULTI_SITE",
        "source": {
            "not_synthetic": True,
            "model_dir": "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16",
        },
        "n_tokens": plan.n_tokens,
        "n_layers": 64,
        "hidden": HIDDEN,
        "fit_kind": "real_routed_activation_capture",
        "prompts": [
            {
                "prompt": p.text[:40],
                "n_tokens": p.target_n_tokens,
                "ids": [1] * p.target_n_tokens,
                "split": p.split,
            }
            for p in plan.prompts
        ],
    }


def test_existing_256_token_capture_refused_for_6144_fit() -> None:
    """The gate must be observed FAILING. 256 rows vs 6144 is underdetermined."""
    meta = _load_v1_meta()
    assert meta["n_tokens"] == 256
    assert meta["hidden"] == HIDDEN
    verdict = adequacy_from_capture(
        meta,
        fit_dim=MIXER_IN,
        procedure="fit_from_X",
        path=str(resolve_v1_capture()),
    )
    assert verdict.status == "REFUSED"
    assert verdict.determination == "UNDERDETERMINED"
    assert verdict.emit_score is False
    assert verdict.score is None
    assert verdict.n_fit < MIXER_IN
    assert verdict.n_rows == 256
    assert verdict.rows_per_dim == pytest.approx(verdict.n_fit / MIXER_IN)
    assert verdict.rows_per_dim < NS014_RPD
    assert verdict.worse_than_ns014 is True
    assert "REFUSED" in verdict.reason
    dumped = verdict.as_dict()
    assert dumped["verdict"] == "REFUSED"
    assert dumped["score"] is None
    assert dumped["emit_score"] is False
    with pytest.raises(AdequacyRefused):
        verdict.require()


def test_synthetic_adequate_case_is_accepted() -> None:
    meta = _synthetic_adequate_meta()
    verdict = adequacy_from_capture(meta, fit_dim=INTERMEDIATE, procedure="fit_from_X")
    assert verdict.status == "ACCEPTED"
    assert verdict.determination == "DETERMINED"
    assert verdict.emit_score is True
    assert verdict.n_fit >= INTERMEDIATE
    assert verdict.n_fit == 17412
    assert verdict.n_hold == 5804
    assert verdict.n_rows == N_TOKENS
    assert verdict.n_prompts >= 64
    assert verdict.holdout_by_prompt is True


def test_ns014_92_against_2048_is_refused() -> None:
    report = adequacy_gate(
        n_rows=NS014_ROWS,
        fit_dim=NS014_DIM,
        n_prompts=3,
        procedure="fit_from_X",
        n_fit=NS014_ROWS,
        n_hold=0,
        holdout_by_prompt=True,
        x_source="PARENT_BF16_REAL",
        not_synthetic=True,
    )
    # 92 rows cannot satisfy 25% hold + n_fit >= 2048. Either way REFUSED.
    assert report.status == "REFUSED"
    assert report.emit_score is False
    assert rows_per_dim(NS014_ROWS, NS014_DIM) == pytest.approx(0.044921875)


def test_undersampled_fit_counts_as_refused_even_with_holdout_ok() -> None:
    # 256 rows, 5 prompts, prompt holdout will leave n_fit ~185 << 6144.
    report = adequacy_gate(
        n_rows=256,
        fit_dim=6144,
        n_prompts=5,
        procedure="fit_from_X",
        n_fit=192,
        n_hold=64,
        holdout_by_prompt=True,
        x_source="PARENT_BF16_REAL",
        not_synthetic=True,
    )
    assert report.status == "REFUSED"
    assert report.n_fit == 192
    assert report.rows_per_dim == pytest.approx(192 / 6144)
    assert report.worse_than_ns014 is True


def test_eval_weight_only_thin_emits_eval_not_a_fit() -> None:
    report = adequacy_gate(
        n_rows=256,
        fit_dim=6144,
        n_prompts=5,
        procedure="eval_weight_only",
        n_fit=192,
        n_hold=64,
        holdout_by_prompt=True,
        x_source="PARENT_BF16_REAL",
        not_synthetic=True,
    )
    assert report.status == "ACCEPTED"
    assert report.eval_thin is True
    assert report.procedure == "eval_weight_only"
    assert "not a fit" in (report.reason or "")


def test_gated_score_does_not_call_fn_or_emit_default_on_refuse() -> None:
    called = {"n": 0}

    def boom() -> float:
        called["n"] += 1
        return 1.0

    meta = _load_v1_meta()
    out = gated_score(meta, fit_dim=6144, procedure="fit_from_X", score_fn=boom)
    assert called["n"] == 0
    assert out["verdict"] == "REFUSED"
    assert out["score"] is None
    assert out["adequacy"]["score"] is None
    assert out["adequacy"]["emit_score"] is False
    assert 1.0 not in (out["score"], out["adequacy"]["score"])


def test_prompt_holdout_does_not_leak_same_prompt() -> None:
    sizes = [57, 60, 68, 61, 10]
    fit_ids, hold_ids, n_fit, n_hold = prompt_holdout(sizes)
    assert set(fit_ids).isdisjoint(set(hold_ids))
    assert n_fit + n_hold == 256
    assert n_hold >= max(4, int(__import__("math").ceil(0.25 * 256)))
    # Every token of a held prompt is held.
    assert n_hold == sum(sizes[i] for i in hold_ids)


def test_designated_hold_matches_plan() -> None:
    plan = build_prompt_plan()
    fit_ids, hold_ids, n_fit, n_hold = prompt_holdout(
        plan.sizes(), designated_hold=plan.designated_hold()
    )
    assert n_fit == 17412
    assert n_hold == 5804
    assert set(fit_ids).isdisjoint(set(hold_ids))
    assert len(fit_ids) + len(hold_ids) == plan.n_sequences


def test_prompt_corpus_constraints() -> None:
    plan = build_prompt_plan()
    assert plan.n_tokens == 23216
    assert plan.n_sequences >= 64
    assert plan.n_sequences >= 32
    assert plan.n_fit == 17412
    assert plan.n_hold == 5804
    assert plan.n_fit >= INTERMEDIATE
    assert min(p.target_n_tokens for p in plan.prompts) >= 32
    assert max(p.target_n_tokens for p in plan.prompts) <= 2048
    long_mass = sum(p.target_n_tokens for p in plan.prompts if p.cls == "long")
    assert long_mass == 2322
    assert all(p.target_n_tokens >= 512 for p in plan.prompts if p.cls == "long")
    assert plan.class_mass["prose"] == 5804
    assert plan.class_mass["code"] == 4643
    assert plan.class_mass["adversarial"] == 1161


def test_sites_cover_required_widths() -> None:
    by_id = {s.site_id: s for s in SITES}
    assert by_id["post_input_norm"].width == 5120
    assert by_id["post_attn_norm"].width == 5120
    assert by_id["post_swiglu"].width == 17408
    assert by_id["mixer_x"].width == 6144
    assert by_id["final_norm"].width == 5120
    assert by_id["mixer_x"].n_layers == 64
    assert by_id["final_norm"].n_layers == 1
    assert {s.site_id for s in SITES} == {
        "post_input_norm",
        "post_attn_norm",
        "post_swiglu",
        "mixer_x",
        "final_norm",
    }


def test_candidate_source_is_refused() -> None:
    meta = {
        "n_tokens": 23216,
        "source": {"not_synthetic": True, "model_dir": ".../mixed-2p0-v1"},
        "vehicle_bpw": 2.0856,
        "prompts": [{"prompt": "a", "n_tokens": 363, "ids": [1] * 363} for _ in range(64)],
    }
    verdict = adequacy_from_capture(meta, fit_dim=6144, procedure="fit_from_X")
    assert verdict.status == "REFUSED"
    assert "candidate" in verdict.reason.lower() or "mixed" in verdict.reason.lower()


def test_rank_starve_is_refused() -> None:
    report = adequacy_gate(
        n_rows=23216,
        fit_dim=160,
        n_prompts=64,
        procedure="fit_from_X",
        n_fit=17412,
        n_hold=5804,
        holdout_by_prompt=True,
        x_source="PARENT_BF16_REAL",
        not_synthetic=True,
        rank_claimed=256,
        rank_clamped_to_n_fit=True,
    )
    assert report.status == "REFUSED"
    assert "starve" in report.reason.lower()


def test_interpolated_layer_is_refused() -> None:
    report = adequacy_gate(
        n_rows=23216,
        fit_dim=5120,
        n_prompts=64,
        procedure="fit_from_X",
        n_fit=17412,
        n_hold=5804,
        holdout_by_prompt=True,
        x_source="PARENT_BF16_REAL",
        not_synthetic=True,
        interpolated=True,
    )
    assert report.status == "REFUSED"
    assert "UNMEASURED" in report.reason


def test_row_shuffle_holdout_is_refused() -> None:
    report = adequacy_gate(
        n_rows=23216,
        fit_dim=17408,
        n_prompts=64,
        procedure="fit_from_X",
        n_fit=17412,
        n_hold=5804,
        holdout_by_prompt=False,
        x_source="PARENT_BF16_REAL",
        not_synthetic=True,
    )
    assert report.status == "REFUSED"
    assert "prompt" in report.reason.lower()


def test_plan_contains_required_fields() -> None:
    plan = build_capture_plan()
    text = format_plan(plan)
    for needle in (
        "n_tokens: 23216",
        "post_input_norm",
        "post_attn_norm",
        "post_swiglu",
        "mixer_x",
        "final_norm",
        "parent_bytes_f16:",
        "parent_gb_f16:",
        "peak_process_gb:",
        "wall_s_linear:",
        "gpu: NOT TOUCHED",
        "model_load: NO",
    ):
        assert needle in text
    assert plan.n_tokens == 23216
    assert plan.n_fit == 17412
    assert plan.parent_bytes_f16 > 60 * 10**9
    assert plan.parent_bytes_f16 < 80 * 10**9
    assert all(s.adequacy == "ACCEPTED" for s in plan.sites)
    mixer = next(s for s in plan.sites if s.site_id == "mixer_x")
    assert mixer.store_n == 8192
    assert mixer.width == 6144
    swiglu = next(s for s in plan.sites if s.site_id == "post_swiglu")
    assert swiglu.store_n == 23216
    assert swiglu.width == 17408


def test_plan_cli_runs_without_gpu() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "qwen38_capture_v2.py"), "--plan"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr
    assert "n_tokens: 23216" in proc.stdout
    assert "gpu: NOT TOUCHED" in proc.stdout
    assert "mixer_x" in proc.stdout
    assert "REFUSED" in proc.stdout or "adequacy law" in proc.stdout.lower() or "NS-014" in proc.stdout
