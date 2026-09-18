"""Unit tests for the in-memory OPH router intervention."""
from __future__ import annotations

import pytest


class _Gate:
    num_experts = 4
    num_group = 2
    topk_group = 1
    top_k = 2
    norm_topk_prob = True
    routed_scaling_factor = 1.0

    def __init__(self, torch):
        self.e_score_correction_bias = torch.zeros(self.num_experts)


def _fixture():
    torch = pytest.importorskip("torch")
    from tools.future.kimi_router_operator_live import _recompute_topk

    gate = _Gate(torch)
    logits = torch.tensor([
        [4.0, 3.0, 0.0, -1.0],
        [-1.0, 0.0, 4.0, 3.0],
        [2.0, 1.0, 0.0, -2.0],
    ])
    weights, indices = _recompute_topk(gate, logits)
    return torch, gate, logits, weights, indices


def test_router_projection_preserves_inactive_rows_and_recomputes_exact_topk():
    torch, gate, logits, weights, indices = _fixture()
    from tools.future.kimi_router_operator_live import _recompute_topk, project_router_output

    output = (logits.clone(), weights.clone(), indices.clone(), "extra")
    result = project_router_output(
        gate,
        output,
        direction=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        anchor=torch.zeros(4),
        strength=1.0,
        active_rows=torch.tensor([True, False, True]),
    )

    assert result[0].shape == logits.shape
    assert result[1].shape == weights.shape
    assert result[2].shape == indices.shape
    assert result[3] == "extra"
    assert torch.equal(result[0][1], logits[1])
    assert torch.equal(result[1][1], weights[1])
    assert torch.equal(result[2][1], indices[1])
    assert not torch.equal(result[0][0], logits[0])
    expected_weights, expected_indices = _recompute_topk(gate, result[0])
    assert torch.equal(result[1][0], expected_weights[0])
    assert torch.equal(result[2][0], expected_indices[0])


def test_zero_strength_is_a_noop_for_logits_and_routing():
    torch, gate, logits, weights, indices = _fixture()
    from tools.future.kimi_router_operator_live import project_router_output

    result = project_router_output(
        gate,
        (logits, weights, indices),
        direction=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        anchor=torch.ones(4),
        strength=0.0,
    )
    assert torch.equal(result[0], logits)
    assert torch.equal(result[1], weights)
    assert torch.equal(result[2], indices)


def test_router_projection_rejects_invalid_width_mask_and_strength():
    torch, gate, logits, weights, indices = _fixture()
    from tools.future.kimi_router_operator_live import project_router_output

    with pytest.raises(ValueError, match="between 0 and 1"):
        project_router_output(gate, (logits, weights, indices), torch.ones(4), torch.zeros(4), 1.1)
    with pytest.raises(ValueError, match="width"):
        project_router_output(gate, (logits, weights, indices), torch.ones(3), torch.zeros(4), 1.0)
    with pytest.raises(ValueError, match="mask"):
        project_router_output(
            gate, (logits, weights, indices), torch.ones(4), torch.zeros(4), 1.0,
            active_rows=torch.ones(2, dtype=torch.bool),
        )


def test_prefill_mask_targets_final_token_per_batch_and_skips_decode():
    torch = pytest.importorskip("torch")
    from tools.future.kimi_router_operator_live import _active_prefill_rows

    mask = _active_prefill_rows((torch.zeros(2, 3, 5),), 6, prefill_only=True)
    assert mask.tolist() == [False, False, True, False, False, True]
    decode = _active_prefill_rows((torch.zeros(2, 1, 5),), 2, prefill_only=True)
    assert not bool(decode.any())


def _router_row(layers):
    return {
        "prompt": "probe",
        "observation": {
            "routers": {
                f"model.layers.{layer}.mlp.gate": () for layer in layers
            }
        },
    }


def test_router_layer_coverage_checks_every_behavior_row():
    from tools.future.kimi_router_operator_live import router_layer_coverage

    report = router_layer_coverage(
        [_router_row((0, 1)), _router_row((0,))],
        [_router_row((0, 1)), _router_row((0, 1))],
    )

    assert report["status"] == "OK"
    assert report["common_layers"] == [0]
    assert report["groups"]["refused"]["intersection"] == [0]
    assert report["groups"]["refused"]["missing_rows"] == {"1": [1]}


def test_router_layer_coverage_reports_no_common_layer_without_model_work():
    from tools.future.kimi_router_operator_live import router_layer_coverage

    report = router_layer_coverage(
        [_router_row((0,))] * 4,
        [_router_row((1,))] * 4,
    )

    assert report["status"] == "NO_COMMON_ROUTER_LAYERS"
    assert report["common_layers"] == []
    assert report["groups"]["refused"]["intersection"] == [0]
    assert report["groups"]["complied"]["intersection"] == [1]


def test_crossfit_returns_gradeable_preflight_receipt_before_model_calls():
    pytest.importorskip("torch")
    from tools.future.kimi_router_operator_live import _run_crossfit

    refused = [_router_row((0,)) | {"prompt": f"r{i}"} for i in range(4)]
    complied = [_router_row((1,)) | {"prompt": f"c{i}"} for i in range(4)]
    report = _run_crossfit(
        object(), None, "cpu", refused, complied,
        strength=1.0, n_splits=2, seed=3,
    )

    assert report["status"] == "NO_COMMON_ROUTER_LAYERS"
    assert report["causal_effect_established"] is False
    assert report["router_layer_coverage"]["common_layers"] == []


def test_mps_owner_block_protects_the_single_live_daemon_surface():
    from tools.future.kimi_router_operator_live import mps_owner_block

    blocked = mps_owner_block({
        "owner": {"daemon": "hawkingd", "pid": 21611},
        "resident": "KIMI_P0_OPERATIONAL",
    })
    assert blocked["status"] == "OPH_DEVICE_BUSY"
    assert blocked["owner_pid"] == 21611
    assert mps_owner_block({"owner": {"daemon": "other"}}) is None
    assert mps_owner_block(None) is None


def test_run_refuses_scarred_oph_before_prompt_or_model_work(monkeypatch):
    from pathlib import Path
    from tools.future import kimi_router_operator_live as live

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("prompt/model preflight must not run for scarred OPH")

    monkeypatch.setattr(live, "load_prompt_battery", fail_if_called)
    report = live.run(Path("/does/not/need/to/exist"), device="mps")

    assert report["status"] == "NEGATIVE_SCIENCE_REFUSED"
    assert report["negative_science_preflight"]["allowed"] is False
    assert report["weights_written"] is False
    assert report["artifact_created"] is False


def test_explicit_oph_reproduction_reaches_runtime_preflight(monkeypatch):
    from pathlib import Path
    from tools.future import kimi_router_operator_live as live

    monkeypatch.setattr(
        live,
        "load_prompt_battery",
        lambda _path: (["refused"] * 4, ["complied"] * 4, {"kind": "test"}),
    )
    monkeypatch.setattr(
        live,
        "daemon_owner_snapshot",
        lambda: {"owner": {"daemon": "hawkingd", "pid": 42}},
    )
    report = live.run(
        Path("/does/not/need/to/exist"),
        device="mps",
        reproduce_scarred_family=True,
    )

    assert report["status"] == "OPH_DEVICE_BUSY"
    assert report["negative_science_preflight"]["status"] == (
        "SCARRED_FAMILY_REPRODUCTION_ONLY"
    )


def test_cli_writes_guard_receipt_without_assuming_causal_row(monkeypatch, tmp_path):
    import json
    import sys
    from tools.future import kimi_router_operator_live as live

    output = tmp_path / "guard.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kimi_router_operator_live.py",
            "--spec", "/does/not/need/to/exist",
            "--output", str(output),
        ],
    )
    assert live.main() == 2
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "NEGATIVE_SCIENCE_REFUSED"
