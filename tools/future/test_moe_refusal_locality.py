"""Unit tests for the refusal-locality statistics and causal guardrails."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tools.future.moe_refusal_locality import (
    _auroc,
    causal_rate_report,
    causal_sweep,
    hcli_contract_intervention_report,
    heldout_separation,
    heldout_causal_rows,
    mean_agreement,
    stratified_split,
    _crossfit_folds,
    _select_train_only_hidden_index,
    load_prompt_battery,
)


def test_heldout_probe_fits_train_and_scores_unseen_rows():
    rng = np.random.default_rng(7)
    axis = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    positive = [axis * 2.0 + rng.normal(0.0, 0.05, 5) for _ in range(8)]
    negative = [-axis * 2.0 + rng.normal(0.0, 0.05, 5) for _ in range(8)]
    p_train, p_test, n_train, n_test = stratified_split(positive, negative, seed=11)

    report = heldout_separation(
        positive,
        negative,
        train_positive=[positive[i] for i in p_train],
        train_negative=[negative[i] for i in n_train],
        heldout_positive=[positive[i] for i in p_test],
        heldout_negative=[negative[i] for i in n_test],
    )

    assert report["fit_scope"] == "train_only"
    assert report["n_train_positive"] == 6
    assert report["n_heldout_negative"] == 2
    assert report["heldout_auroc"] > 0.99
    assert report["heldout_balanced_accuracy"] > 0.99
    assert report["direction"].shape == (5,)


def test_heldout_metric_does_not_silently_claim_a_split_when_omitted():
    rows = [np.array([float(i), 0.0]) for i in range(4)]
    report = heldout_separation(rows, [row * -1 for row in rows])
    assert report["fit_scope"] == "train_plus_all_when_no_explicit_split"
    assert report["heldout_auroc"] is None
    assert report["heldout_accuracy"] is None


def test_expanded_prompt_file_is_separate_from_historical_default_battery():
    default_harmful, default_harmless, default_meta = load_prompt_battery()
    expanded_harmful, expanded_harmless, expanded_meta = load_prompt_battery(
        "tools/future/kimi_refusal_battery_expanded_v1.json"
    )
    assert (len(default_harmful), len(default_harmless)) == (16, 16)
    assert (len(expanded_harmful), len(expanded_harmless)) == (32, 32)
    assert default_meta["kind"] == "builtin"
    assert expanded_meta["kind"] == "file"
    assert expanded_meta["sha256"]


def test_zero_contrast_layer_is_recorded_as_invalid_not_fatal():
    rows = [np.zeros(3) for _ in range(4)]
    report = heldout_separation(
        rows,
        rows,
        train_positive=rows[:2],
        train_negative=rows[:2],
        heldout_positive=rows[2:],
        heldout_negative=rows[2:],
    )
    assert report["direction_valid"] is False
    assert report["heldout_auroc"] == pytest.approx(0.5)


def test_split_is_deterministic_and_keeps_two_rows_for_test():
    first = stratified_split(range(8), range(8), seed=42)
    second = stratified_split(range(8), range(8), seed=42)
    assert first == second
    p_train, p_test, n_train, n_test = first
    assert len(p_train) == len(n_train) == 6
    assert len(p_test) == len(n_test) == 2
    assert not set(p_train) & set(p_test)
    assert not set(n_train) & set(n_test)


def test_causal_rows_use_only_heldout_behavior_indices():
    prompts, classes = heldout_causal_rows(
        ["r0", "r1", "r2"], ["c0", "c1", "c2", "c3"], [1], [2, 0])
    assert prompts == ["r1", "c2", "c0"]
    assert classes == [True, False, False]


def test_auroc_gives_half_credit_to_ties():
    assert _auroc([1.0, 1.0], [True, False]) == pytest.approx(0.5)
    assert _auroc([2.0, 0.0], [True, False]) == pytest.approx(1.0)


def test_causal_effect_requires_matched_random_control_and_preserves_compliance():
    report = causal_rate_report(
        [True, True, False, False],
        [False, False, False, False],
        baseline_control=[True, True, False, False],
        treated_control=[True, True, False, False],
        baseline_classes=[True, True, False, False],
    )
    assert report["by_baseline_class"]["refused"]["refusal_rate_delta"] == -1.0
    assert report["by_baseline_class"]["complied"]["refusal_rate_delta"] == 0.0
    assert report["causal_effect_established"] is True
    assert report["causal_sample_sufficient"] is False


def test_causal_effect_does_not_credit_a_shift_also_seen_in_refusal_class_null():
    report = causal_rate_report(
        [True, True, False, False],
        [False, True, False, False],
        baseline_control=[True, True, False, False],
        treated_control=[False, True, False, False],
        baseline_classes=[True, True, False, False],
    )
    assert report["by_baseline_class"]["refused"]["refusal_rate_delta"] == pytest.approx(-0.5)
    assert report["control_by_baseline_class"]["refused"]["refusal_rate_delta"] == pytest.approx(-0.5)
    assert report["causal_effect_margin"] == pytest.approx(0.0)
    assert report["causal_effect_established"] is False


def test_crossfit_folds_cover_each_row_once():
    folds = _crossfit_folds(9, 4, 17)
    flattened = [item for fold in folds for item in fold]
    assert sorted(flattened) == list(range(9))
    assert len({item for item in flattened}) == 9


def test_hidden_selection_reports_outer_train_only_scope():
    hidden_rows = {
        1: [
            np.array([2.0]), np.array([2.1]), np.array([-2.0]), np.array([-2.1]),
        ],
        2: [
            np.array([0.1]), np.array([0.2]), np.array([-0.1]), np.array([-0.2]),
        ],
    }
    selected, metrics = _select_train_only_hidden_index(
        hidden_rows,
        positive_indices=[0, 1],
        negative_indices=[2, 3],
    )
    assert selected == 1
    assert metrics["1"]["selection_fit_scope"] == "outer_train_only"
    assert metrics["1"]["n_positive"] == 2
    assert metrics["1"]["n_negative"] == 2


def test_causal_effect_without_control_is_not_established():
    report = causal_rate_report([True, False], [False, False])
    assert report["direction"] == "reduced"
    assert report["causal_effect_established"] is False
    assert report["control"] is None


def test_causal_effect_rejects_new_refusals_on_compliant_baseline_rows():
    report = causal_rate_report(
        [True, False],
        [False, True],
        baseline_control=[True, False],
        treated_control=[True, False],
        baseline_classes=[True, False],
    )
    assert report["causal_effect_established"] is False


def test_causal_sweep_rejects_an_undeclared_layer_before_running_model():
    with pytest.raises(ValueError, match="no held-out direction"):
        causal_sweep(
            None, None, [], [], None,
            directions={}, layers=[3], strengths=[0.5])


def test_agreement_accepts_numpy_negative_control():
    rng = np.random.default_rng(2)
    shared = {i: np.array([1.0, 0.0]) + rng.normal(0.0, 0.01, 2) for i in range(4)}
    assert mean_agreement(shared) > 0.99


def test_residual_intervention_validation_is_live_torch_only():
    torch = pytest.importorskip("torch")
    from tools.future.moe_refusal_locality import apply_residual_intervention

    hidden = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    out = apply_residual_intervention(hidden, torch.tensor([1.0, 0.0]), 1.0)
    assert torch.equal(hidden[..., :-1, :], out[..., :-1, :])
    assert torch.allclose(out[..., -1, 0], torch.tensor(0.0))
    with pytest.raises(ValueError, match="between 0 and 1"):
        apply_residual_intervention(hidden, torch.tensor([1.0, 0.0]), 1.1)


def test_cli_refuses_closed_causal_family_before_loading_prompts(monkeypatch, capsys):
    from tools.future import moe_refusal_locality as locality

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("prompt/model work must not start for a scarred family")

    monkeypatch.setattr(locality, "load_prompt_battery", fail_if_called)
    monkeypatch.setattr(
        "sys.argv",
        [
            "moe_refusal_locality.py",
            "--spec", "/does/not/need/to/exist",
            "--causal",
            "--causal-candidate", "OPE",
        ],
    )
    assert locality.main() == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "NEGATIVE_SCIENCE_REFUSED"
    assert output["method"] == "OPE"


def test_residual_addition_is_last_token_only_and_normalized():
    torch = pytest.importorskip("torch")
    from tools.future.moe_refusal_locality import apply_residual_addition

    hidden = torch.zeros((1, 2, 2))
    out = apply_residual_addition(hidden, torch.tensor([3.0, 0.0]), 0.5)
    assert torch.equal(hidden[..., :-1, :], out[..., :-1, :])
    assert torch.allclose(out[..., -1, :], torch.tensor([[0.5, 0.0]]))
    with pytest.raises(ValueError, match="zero norm"):
        apply_residual_addition(hidden, torch.zeros(2), 0.5)


def test_residual_translation_uses_explicit_fit_magnitude():
    torch = pytest.importorskip("torch")
    from tools.future.moe_refusal_locality import apply_residual_translation

    hidden = torch.zeros((1, 2, 2))
    out = apply_residual_translation(
        hidden, torch.tensor([3.0, 0.0]), 0.5, magnitude=4.0)
    assert torch.equal(hidden[..., :-1, :], out[..., :-1, :])
    assert torch.allclose(out[..., -1, :], torch.tensor([[2.0, 0.0]]))
    with pytest.raises(ValueError, match="non-negative"):
        apply_residual_translation(hidden, torch.ones(2), 0.5, magnitude=-1.0)


def test_prefill_only_residual_hook_skips_cached_decode_shape():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from tools.future.moe_refusal_locality import install_residual_addition

    layer = torch.nn.Identity()
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    handle = install_residual_addition(
        model, 0, torch.tensor([1.0, 0.0]), 1.0, prefill_only=True)
    try:
        prefill = layer(torch.zeros((1, 2, 2)))
        decode = layer(torch.zeros((1, 1, 2)))
    finally:
        handle.remove()
    assert torch.allclose(prefill[0, -1], torch.tensor([1.0, 0.0]))
    assert torch.equal(decode, torch.zeros((1, 1, 2)))


def test_writer_translation_adds_output_direction_without_mutating_input():
    torch = pytest.importorskip("torch")
    from tools.future.moe_refusal_locality import apply_writer_translation

    output = torch.zeros((2, 3, 2))
    translated = apply_writer_translation(
        output, torch.tensor([2.0, 0.0]), 0.5, magnitude=4.0)
    assert torch.equal(output, torch.zeros((2, 3, 2)))
    assert torch.allclose(translated[:, :-1], torch.zeros((2, 2, 2)))
    assert torch.allclose(translated[:, -1], torch.tensor([[2.0, 0.0], [2.0, 0.0]]))


def test_conditional_residual_translation_only_moves_below_threshold_rows():
    torch = pytest.importorskip("torch")
    from tools.future.moe_refusal_locality import apply_conditional_residual_translation

    hidden = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]])
    out = apply_conditional_residual_translation(
        hidden,
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        threshold=1.0,
        strength=1.0,
        magnitude=2.0,
    )
    assert torch.equal(out[..., :-1, :], hidden[..., :-1, :])
    assert torch.allclose(out[..., -1, :], torch.tensor([[2.0, 0.0]]))
    low = torch.tensor([[[0.0, 0.0], [0.5, 0.0]]])
    low_out = apply_conditional_residual_translation(
        low, torch.tensor([1.0, 0.0]), torch.zeros(2), 1.0, 1.0,
        magnitude=2.0)
    assert torch.allclose(low_out[..., -1, :], torch.tensor([[2.5, 0.0]]))


def test_prefill_only_writer_translation_skips_decode_output():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from tools.future.moe_refusal_locality import install_writer_translation

    target = torch.nn.Identity()
    block = SimpleNamespace(
        mlp=SimpleNamespace(shared_experts=SimpleNamespace(down_proj=target)),
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
    handle = install_writer_translation(
        model, 0, torch.tensor([1.0, 0.0]), 1.0,
        "shared_experts_down", magnitude=1.0, prefill_only=True)
    try:
        prefill = target(torch.zeros((2, 2)))
        decode = target(torch.zeros((1, 2)))
    finally:
        handle.remove()
    assert torch.allclose(prefill[-1], torch.tensor([1.0, 0.0]))
    assert torch.equal(decode, torch.zeros((1, 2)))


def test_conditional_projection_does_not_touch_below_threshold_rows():
    torch = pytest.importorskip("torch")
    from tools.future.moe_refusal_locality import (
        _fit_gate_anchor,
        apply_conditional_residual_intervention,
    )

    anchor, threshold = _fit_gate_anchor(
        [np.array([2.0, 0.0]), np.array([2.2, 0.0])],
        [np.array([-2.0, 0.0]), np.array([-2.2, 0.0])],
        np.array([1.0, 0.0]),
    )
    hidden = torch.tensor([
        [[0.0, 1.0], [3.0, 1.0], [3.0, 1.0]],
        [[0.0, 1.0], [3.0, 1.0], [-3.0, 1.0]],
    ])
    out = apply_conditional_residual_intervention(
        hidden, torch.tensor([1.0, 0.0]), anchor, threshold, 1.0)
    assert torch.equal(hidden[..., :-1, :], out[..., :-1, :])
    assert torch.allclose(out[0, -1, :], torch.tensor([0.0, 1.0]))
    assert torch.allclose(out[1, -1, :], torch.tensor([-3.0, 1.0]))


def test_conditional_projection_rejects_zero_direction():
    torch = pytest.importorskip("torch")
    from tools.future.moe_refusal_locality import apply_conditional_residual_intervention

    hidden = torch.zeros((1, 1, 2))
    with pytest.raises(ValueError, match="zero norm"):
        apply_conditional_residual_intervention(
            hidden, torch.zeros(2), torch.zeros(2), 0.0, 0.5)


def test_writer_weight_intervention_restores_on_exception():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from tools.future.moe_refusal_locality import writer_weight_intervention

    block = SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=torch.nn.Linear(3, 3, bias=False)),
        mlp=SimpleNamespace(
            shared_experts=SimpleNamespace(
                down_proj=torch.nn.Linear(4, 3, bias=False)),
        ),
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
    original = block.mlp.shared_experts.down_proj.weight.detach().clone()
    with pytest.raises(RuntimeError, match="sentinel"):
        with writer_weight_intervention(
            model, 0, torch.ones(3), 0.5, "shared_experts_down"):
            assert not torch.equal(block.mlp.shared_experts.down_proj.weight, original)
            raise RuntimeError("sentinel")
    assert torch.equal(block.mlp.shared_experts.down_proj.weight, original)


def test_writer_weight_intervention_accepts_rank_k_subspace_and_preserves_norm():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from tools.future.moe_refusal_locality import writer_weight_intervention

    block = SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=torch.nn.Linear(4, 3, bias=False)),
        mlp=SimpleNamespace(
            shared_experts=SimpleNamespace(
                down_proj=torch.nn.Linear(3, 4, bias=False)),
        ),
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
    weight = block.mlp.shared_experts.down_proj.weight
    original = weight.detach().clone()
    original_norm = original.norm()
    basis = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    with writer_weight_intervention(
        model, 0, basis, 1.0, "shared_experts_down", norm_preserve=True):
        assert not torch.equal(weight, original)
        assert torch.allclose(weight.norm(), original_norm, atol=1e-5)
    assert torch.equal(weight, original)


def test_writer_weight_intervention_accepts_two_sided_candidate_and_restores():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from tools.future.moe_refusal_locality import writer_weight_intervention

    block = SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=torch.nn.Linear(3, 4, bias=False)),
        mlp=SimpleNamespace(
            shared_experts=SimpleNamespace(
                down_proj=torch.nn.Linear(3, 4, bias=False)),
        ),
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
    weight = block.mlp.shared_experts.down_proj.weight
    original = weight.detach().clone()
    with writer_weight_intervention(
        model, 0, torch.tensor([1.0, 0.0, 0.0, 0.0]), 0.5,
        "shared_experts_down", input_direction=torch.tensor([1.0, 0.0, 0.0])):
        assert not torch.equal(weight, original)
    assert torch.equal(weight, original)


def test_behavior_subspace_keeps_declared_rank_and_starts_with_class_delta():
    from tools.future.moe_refusal_locality import _fit_behavior_subspace

    positive = [np.array([2.0, 0.1, 0.0, 0.0]), np.array([2.1, -0.1, 0.0, 0.0])]
    negative = [np.array([-2.0, 0.0, 0.2, 0.0]), np.array([-2.1, 0.0, -0.2, 0.0])]
    basis = _fit_behavior_subspace(positive, negative, 3)
    assert basis.shape == (4, 3)
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-8)
    assert abs(float(basis[:, 0] @ np.array([4.1, 0.1, -0.1, 0.0]))) > 0.9


def test_hcli_contract_report_requires_improvement_over_matched_null():
    baseline = [{"score": {"passed": i == 0}} for i in range(16)]
    treated = [{"score": {"passed": i < 5}} for i in range(16)]
    null = [{"score": {"passed": i < 2}} for i in range(16)]
    report = hcli_contract_intervention_report(baseline, treated, null)
    assert report["pass_rate_delta"] == pytest.approx(0.25)
    assert report["null_pass_rate_delta"] == pytest.approx(1 / 16)
    assert report["causal_effect_established"] is True
    assert report["causal_sample_sufficient"] is True
