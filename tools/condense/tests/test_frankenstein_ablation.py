"""Stage-1 PROTO_FRANKENSTEIN A-vs-B ablation: reject rule + shard bench."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import frankenstein_ablation as ablation  # noqa: E402
from lab.operators import frankenstein_receipts as receipts  # noqa: E402
from lab.operators.frankenstein_fusion_op import TRANSPLANT_POINT_NAMES  # noqa: E402
from lab.receipts import verify  # noqa: E402


def _scores(fill: float = 0.70) -> dict[str, dict[str, float]]:
    return ablation.default_score_template(fill)


def _bump(scores: dict[str, dict[str, float]], group: str, domain: str, value: float):
    out = {
        "math": dict(scores["math"]),
        "secondary": dict(scores["secondary"]),
    }
    out[group][domain] = value
    return out


def test_reject_rule_fires_on_math_up_coding_down(tmp_path: Path) -> None:
    """Math gain must never override a secondary coding regression past tolerance."""

    base = _scores(0.70)
    proto = _scores(0.70)
    # Math up across the board.
    for domain in ablation.MATH_DOMAINS:
        proto["math"][domain] = 0.90
    # Coding down beyond sealed 0.02 tolerance (0.70 → 0.60 = −0.10).
    proto["secondary"]["coding_and_repository_work"] = 0.60

    fixture = ablation.make_capability_fixture(
        fixture_id="math_up_coding_down",
        bench_scope="SHARD",
        base_math=base["math"],
        base_secondary=base["secondary"],
        proto_math=proto["math"],
        proto_secondary=proto["secondary"],
        transfer_module_id="proto-candidate-regress-coding",
        meta={"intent": "prove reject rule fires"},
    )
    path = tmp_path / "math_up_coding_down.json"
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")

    report = ablation.run_avb_from_fixture(path)
    verify(report, label="ablation report")

    assert report["verdict"] == "REJECT"
    assert report["reject_rule_fired"] is True
    assert report["additive_not_subtractive"] is False
    assert report["secondary"]["verdict"] == "REJECT"
    regs = report["secondary"]["regressions"]
    assert len(regs) == 1
    assert regs[0]["domain"] == "coding_and_repository_work"
    assert regs[0]["delta"] == pytest.approx(-0.10)
    assert regs[0]["gate"] == "FAIL"
    # Math gain was real — and still does not save the candidate.
    assert report["math"]["mean_gain"] == pytest.approx(0.20)
    assert report["math"]["any_gain"] is True
    assert report["gate_policy"]["reject_rule"]["math_gain_cannot_override"] is True


def test_accept_when_secondaries_hold_and_math_gains(tmp_path: Path) -> None:
    base = _scores(0.70)
    proto = _scores(0.70)
    for domain in ablation.MATH_DOMAINS:
        proto["math"][domain] = 0.85
    # Tiny secondary drift inside tolerance.
    proto["secondary"]["tool_use"] = 0.69  # −0.01 ≥ −0.02

    fixture = ablation.make_capability_fixture(
        fixture_id="math_up_secondaries_ok",
        bench_scope="FIXTURE",
        base_math=base["math"],
        base_secondary=base["secondary"],
        proto_math=proto["math"],
        proto_secondary=proto["secondary"],
    )
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")

    report = ablation.run_avb_from_fixture(path)
    assert report["verdict"] == "ACCEPT"
    assert report["reject_rule_fired"] is False
    assert report["additive_not_subtractive"] is True
    assert report["math"]["mean_gain"] == pytest.approx(0.15)


def test_reject_on_routing_stability_regression() -> None:
    base = _scores(0.80)
    proto = _bump(_scores(0.80), "secondary", "routing_stability", 0.70)
    for domain in ablation.MATH_DOMAINS:
        proto["math"][domain] = 0.99

    report = ablation.run_avb_ablation(
        base_math=base["math"],
        base_secondary=base["secondary"],
        proto_math=proto["math"],
        proto_secondary=proto["secondary"],
        bench_scope="BOUNDED_FIXTURE",
        fixture_id="routing-regress",
    )
    assert report["verdict"] == "REJECT"
    domains = {r["domain"] for r in report["secondary"]["regressions"]}
    assert "routing_stability" in domains


def test_shard_bench_refuses_full_model() -> None:
    """Full-model bench requests must fail closed in stage 1."""

    with pytest.raises(ablation.FullModelBenchRefused) as excinfo:
        ablation.run_shard_bench(
            ablation.ShardBenchRequest(
                scope="FULL_MODEL",
                shard_ids=("shard-0",),
            )
        )
    assert "refuses full-model" in str(excinfo.value).lower() or "FULL_MODEL" in str(
        excinfo.value
    )

    for bad in ("FULL", "END_TO_END_MODEL", "COMPLETE_MODEL", "LIVE_RUNTIME"):
        with pytest.raises(ablation.FullModelBenchRefused):
            ablation.assert_shard_bench_scope(bad)


def test_shard_bench_accepts_shard_scope(tmp_path: Path) -> None:
    base = _scores(0.70)
    proto = _scores(0.72)
    for domain in ablation.MATH_DOMAINS:
        proto["math"][domain] = 0.80
    fixture = ablation.make_capability_fixture(
        fixture_id="shard-ok",
        bench_scope="SHARD",
        base_math=base["math"],
        base_secondary=base["secondary"],
        proto_math=proto["math"],
        proto_secondary=proto["secondary"],
    )
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")

    document = ablation.run_shard_bench(
        ablation.ShardBenchRequest(
            scope="SHARD",
            shard_ids=("glm-math-L00", "glm-math-L01"),
            scores_fixture_path=path,
        )
    )
    verify(document, label="shard bench")
    assert document["bench_scope"] == "SHARD"
    assert document["full_model_refused"] is True
    assert document["verdict"] == "ACCEPT"
    assert document["ablation"]["verdict"] == "ACCEPT"
    assert document["claim_boundary"]["full_model_bench_permitted"] is False
    assert document["claim_boundary"]["kimi_consumed"] is False


def test_cli_evaluate_reject_exit_code(tmp_path: Path) -> None:
    base = _scores(0.70)
    proto = _bump(_scores(0.90), "secondary", "agentic_planning", 0.50)
    # math already high from fill 0.90
    fixture = ablation.make_capability_fixture(
        fixture_id="cli-reject",
        bench_scope="SHARD",
        base_math=base["math"],
        base_secondary=base["secondary"],
        proto_math=proto["math"],
        proto_secondary=proto["secondary"],
    )
    fixture_path = tmp_path / "cli.json"
    out_path = tmp_path / "report.json"
    fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")

    code = ablation.main(
        [
            "evaluate",
            "--fixture",
            str(fixture_path),
            "--out",
            str(out_path),
        ]
    )
    assert code == 2
    report = json.loads(out_path.read_text())
    assert report["verdict"] == "REJECT"


def test_cli_shard_bench_refuses_full_model(tmp_path: Path) -> None:
    out_path = tmp_path / "should_not_exist.json"
    code = ablation.main(
        [
            "shard-bench",
            "--scope",
            "FULL_MODEL",
            "--shard-id",
            "x",
            "--out",
            str(out_path),
        ]
    )
    assert code == 3
    assert not out_path.exists()


def test_sealed_gate_policy_is_stable() -> None:
    policy = ablation.sealed_gate_policy()
    verify(policy, label="gate policy")
    assert policy["schema"] == ablation.GATE_POLICY_SCHEMA
    assert policy["secondary_tolerance_absolute"] == 0.02
    assert set(policy["math_domains"]) == set(ablation.MATH_DOMAINS)
    assert set(policy["secondary_capabilities"]) == set(ablation.SECONDARY_CAPABILITIES)
    assert "coding_and_repository_work" in policy["secondary_capabilities"]
    assert policy["reject_rule"]["math_gain_cannot_override"] is True


def test_fixture_schema_and_seal_required(tmp_path: Path) -> None:
    bad = {
        "schema": "wrong",
        "bench_scope": "SHARD",
        "base": _scores(0.5),
        "proto": _scores(0.5),
        "seal_sha256": "0" * 64,
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ablation.AblationError):
        ablation.load_capability_fixture(path)


def test_score_out_of_range_rejected() -> None:
    base = _scores(0.5)
    proto = _scores(0.5)
    proto["math"]["method_selection"] = 1.5
    with pytest.raises(ablation.AblationError, match="\\[0, 1\\]"):
        ablation.make_capability_fixture(
            fixture_id="oor",
            bench_scope="SHARD",
            base_math=base["math"],
            base_secondary=base["secondary"],
            proto_math=proto["math"],
            proto_secondary=proto["secondary"],
        )


# --- receipts + handoff contract ---


def test_handoff_contract_preserves_kimi_bridge_points() -> None:
    contract = receipts.build_proto_to_kimi_handoff_contract(
        bridge_path=None,
        transplant_path=None,
    )
    verify(contract, label="handoff contract")
    assert contract["schema"] == receipts.HANDOFF_CONTRACT_SCHEMA
    assert contract["policy"]["stage1_must_not_mutate_kimi_bridge"] is True
    assert contract["policy"]["no_kimi_consumption_in_stage1"] is True
    assert contract["preserved_bridge"]["name"] == "KIMI_STRATEGIC_BRIDGE"
    assert "planning" in contract["preserved_bridge"]["target_functions"]
    assert "coding_breadth" in contract["preserved_bridge"]["target_functions"]
    assert "tool_policy" in contract["preserved_bridge"]["target_functions"]
    assert "context_management" in contract["preserved_bridge"]["target_functions"]
    assert "critique_and_synthesis" in contract["preserved_bridge"]["target_functions"]
    assert "long_horizon_decomposition" in contract["preserved_bridge"]["target_functions"]

    points = {row["transplant_point"] for row in contract["bridge_points"]}
    assert points == set(TRANSPLANT_POINT_NAMES)
    for row in contract["bridge_points"]:
        assert row["bridge"] == "KIMI_STRATEGIC_BRIDGE"
        assert row["stage1_status"] == "PRESERVED_UNTOUCHED"
        assert row["tensor_state"]["input_shape"] == ["batch", "sequence", 4096]
        assert row["direct_weight_transplant"] is False

    assert contract["stage1_math_bridge"]["name"] == "GLM_MATH_BRIDGE"
    assert contract["claim_boundary"]["kimi_consumed"] is False


def test_adapter_manifest_rejects_kimi_blocks() -> None:
    with pytest.raises(receipts.ReceiptError, match="KIMI_STRATEGIC_BRIDGE"):
        receipts.build_adapter_manifest(
            blocks=[
                {
                    "bridge": "KIMI_STRATEGIC_BRIDGE",
                    "transplant_point": "post_moe_hidden_state",
                    "student_layer": 0,
                }
            ]
        )


def test_inheritance_receipt_binds_ablation(tmp_path: Path) -> None:
    base = _scores(0.70)
    proto = _scores(0.70)
    for domain in ablation.MATH_DOMAINS:
        proto["math"][domain] = 0.80
    fixture = ablation.make_capability_fixture(
        fixture_id="inherit-ok",
        bench_scope="SHARD",
        base_math=base["math"],
        base_secondary=base["secondary"],
        proto_math=proto["math"],
        proto_secondary=proto["secondary"],
    )
    path = tmp_path / "fix.json"
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    report = ablation.run_avb_from_fixture(path)

    receipt = receipts.build_inheritance_receipt(
        transfer_module_id="proto-mod-1",
        ablation_report=report,
        adapter_manifest=receipts.build_adapter_manifest(blocks=[]),
        glm_shards=[
            receipts.build_glm_trace_shard_receipt(
                shard_id="glm-L00", verified=False
            )
        ],
        runtime_storage=receipts.build_runtime_storage_accounting(
            adapter_archive_bytes=1024,
            working_set_bytes=2048,
        ),
        bridge_path=None,
        transplant_path=None,
        glm_subspace_path=None,
    )
    verify(receipt, label="inheritance receipt")
    assert receipt["capability_ablation_vs_base"]["verdict"] == "ACCEPT"
    assert receipt["math_targets"][0]["bridge"] == "GLM_MATH_BRIDGE"
    assert receipt["kimi_handoff"]["bridge_preserved"] == "KIMI_STRATEGIC_BRIDGE"
    assert receipt["coding_agent_tool_non_regression"]["verdict"] == "PASS"


def test_write_handoff_contract_atomic(tmp_path: Path) -> None:
    path, document = receipts.write_handoff_contract(
        tmp_path,
        bridge_path=None,
        transplant_path=None,
    )
    assert path.name == "PROTO_TO_KIMI_HANDOFF_CONTRACT.json"
    on_disk = json.loads(path.read_text())
    assert on_disk["seal_sha256"] == document["seal_sha256"]
    verify(on_disk, label="on-disk handoff")
