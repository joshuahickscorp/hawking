"""Tests for PROTO_FRANKENSTEIN end-to-end full-run orchestrator.

Proves:
  - pipeline wiring (load → compose → forward-gate → ablation → receipt)
  - forward gate refuses cleanly when forward is unavailable (no fake validation)
  - reject rule fires on math-up / coding-down fixture
  - no gradient / optimizer / training call anywhere in the proto_run module
  - PROTO artifact is raw (not gravity-compressed)
  - KIMI_STRATEGIC_BRIDGE remains preserved
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import frankenstein_proto_run as proto  # noqa: E402
from lab.operators import frankenstein_transfer as xfer  # noqa: E402
from lab.operators.frankenstein_fusion_op import FORWARD_GATE  # noqa: E402
from lab.receipts import verify  # noqa: E402


PROTO_SOURCE = Path(proto.__file__)


def _synthetic_module():
    extraction = xfer.extract_from_synthetic_weights(rank=8, seed=11)
    return xfer.build_transfer_module(extraction=extraction, steering_scale=0.05)


def test_no_training_imports_or_calls() -> None:
    source = PROTO_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_modules = {
        "torch.optim",
        "torch.nn.functional",
        "transformers.trainer",
        "tensorflow",
        "jax",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_modules, alias.name
                assert not alias.name.startswith("torch.optim"), alias.name
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert mod not in forbidden_modules, mod
            assert not mod.startswith("torch.optim"), mod
            assert mod != "torch", "torch import is forbidden in training-free proto_run"
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in {"backward", "zero_grad"}:
                pytest.fail(f"forbidden training call: {name}")

    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import torch") or stripped.startswith("from torch"):
            pytest.fail(f"torch import present: {stripped}")

    guard = proto.assert_no_training_path()
    assert guard["training_path_present"] is False
    assert guard["imports_torch"] is False
    assert guard["imports_optimizer"] is False


def test_forward_gate_refuses_when_unavailable() -> None:
    """Missing forward → DEEPSEEK_FORWARD_PENDING, no fabricated activations."""

    result = proto.invoke_student_forward(dry_run=False)
    verify(result, label="forward attempt")
    assert result["status"] == FORWARD_GATE
    assert result["activations_captured"] is False
    assert result["capability_measurement"] is False
    assert result["gate"]["name"] == FORWARD_GATE
    reason = result["gate"]["reason"].lower()
    assert (
        "no callable" in reason
        or "not available" in reason
        or "not exposed" in reason
        or "forward_pending" in reason
        or FORWARD_GATE.lower() in reason
    ), reason


def test_dry_run_forward_explicitly_skips() -> None:
    result = proto.invoke_student_forward(dry_run=True)
    verify(result, label="dry-run forward")
    assert result["status"] == FORWARD_GATE
    assert result["dry_run"] is True
    assert result["activations_captured"] is False
    assert result["capability_measurement"] is False


def test_forward_callable_hook_executes_when_provided() -> None:
    def fake_forward(**kwargs):
        return {
            "activations": {"post_moe_hidden_state": [[0.0]]},
            "scores": {"math": 0.9},
            "note": "test-only callable",
        }

    result = proto.invoke_student_forward(forward=fake_forward, dry_run=False)
    assert result["status"] == "FORWARD_EXECUTED"
    assert result["activations_captured"] is True
    assert result["capability_measurement"] is True


def test_pipeline_wiring_dry_run_with_reject_fixture(tmp_path: Path) -> None:
    """Full dry-run: compose PROTO, gate forward, fire reject rule, seal receipt."""

    module = _synthetic_module()
    # Seal module to disk so provenance bindings have real paths.
    sealed = xfer.seal_transfer_module_files(module, out_dir=tmp_path / "mod")
    loaded = xfer.load_transfer_module(
        Path(sealed["meta_path"]), Path(sealed["module_path"])
    )

    fixture = proto.make_reject_fixture_document(
        transfer_module_id="test-proto-reject"
    )
    fixture_path = tmp_path / "reject.json"
    fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")

    result = proto.run_proto_frankenstein(
        module=loaded,
        module_meta=Path(sealed["meta_path"]),
        body_path=tmp_path / "missing-body.gravity",
        out_dir=tmp_path / "proto-run",
        scores_fixture=fixture_path,
        dry_run=True,
        ensure_handoff=True,
        transfer_module_id="test-proto-reject",
    )

    assert result["dry_run"] is True
    assert result["forward_status"] == FORWARD_GATE
    assert result["capability_validated"] is False
    assert result["capability_claim"] is False
    assert result["gravity_compressed"] is False
    # Reject rule on fixture fires even in dry-run (governance evidence).
    assert result["ablation_verdict"] == "REJECT"
    assert result["reject_rule_fired"] is True
    assert result["verdict"] == "REJECT"

    # Receipt on disk, sealed, honest.
    receipt_path = Path(result["receipt_path"])
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verify(receipt, label="proto run receipt")
    assert receipt["schema"] == proto.PROTO_RUN_RECEIPT_SCHEMA
    assert receipt["capability_validated"] is False
    assert receipt["gravity_compressed"] is False
    assert receipt["trained"] is False
    assert receipt["training_free"] is True
    assert receipt["forward_status"] == FORWARD_GATE
    assert receipt["stages"]["forward_activations"]["activations_captured"] is False
    assert receipt["stages"]["ablation_avb"]["verdict"] == "REJECT"
    assert receipt["kimi_handoff"]["bridge_preserved"] == "KIMI_STRATEGIC_BRIDGE"
    assert receipt["math_domain_gain"]["live_forward_measured"] is False

    # PROTO artifact is raw composition, not gravity.
    artifact_path = Path(result["artifact_path"])
    assert artifact_path.is_file()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    verify(artifact, label="proto artifact")
    assert artifact["schema"] == proto.PROTO_ARTIFACT_SCHEMA
    assert artifact["gravity_compressed"] is False
    assert artifact["byte_merge"] is False
    assert artifact["direct_weight_transplant"] is False
    assert artifact["composition"]["body_rewritten"] is False
    assert artifact["bridges"]["kimi_stage1_status"] == "PRESERVED_UNTOUCHED"
    assert artifact["validation"]["status"] == FORWARD_GATE
    assert artifact["validation"]["capability_claim"] is False
    assert artifact["student_body"]["read_only"] is True

    # Forward attempt written and fail-closed.
    forward_path = Path(result["forward_attempt_path"])
    assert forward_path.is_file()
    forward_doc = json.loads(forward_path.read_text(encoding="utf-8"))
    assert forward_doc["status"] == FORWARD_GATE
    assert forward_doc["dry_run"] is True
    assert forward_doc["activations_captured"] is False


def test_pipeline_wiring_dry_run_accept_fixture_still_forward_gated(
    tmp_path: Path,
) -> None:
    """Accept fixture ablation does NOT promote capability without a real forward."""

    module = _synthetic_module()
    sealed = xfer.seal_transfer_module_files(module, out_dir=tmp_path / "mod")
    loaded = xfer.load_transfer_module(
        Path(sealed["meta_path"]), Path(sealed["module_path"])
    )
    fixture = proto.make_accept_fixture_document()
    fixture_path = tmp_path / "accept.json"
    fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")

    result = proto.run_proto_frankenstein(
        module=loaded,
        body_path=tmp_path / "body.gravity",
        out_dir=tmp_path / "out",
        scores_fixture=fixture_path,
        dry_run=True,
        ensure_handoff=True,
    )

    assert result["ablation_verdict"] == "ACCEPT"
    assert result["reject_rule_fired"] is False
    # Critical: still FORWARD_GATED, not a fabricated capability ACCEPT.
    assert result["verdict"] == "FORWARD_GATED"
    assert result["capability_validated"] is False
    assert result["capability_claim"] is False
    assert result["forward_status"] == FORWARD_GATE

    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["verdict"] == "FORWARD_GATED"
    assert "fail closed" in receipt["reason"].lower() or "not validated" in receipt[
        "honest_status"
    ].lower() or "FORWARD" in receipt["reason"]


def test_full_run_without_forward_also_fail_closed(tmp_path: Path) -> None:
    """Non-dry-run with no forward still gates; never fabricates validation."""

    module = _synthetic_module()
    fixture = proto.make_accept_fixture_document()
    fixture_path = tmp_path / "ok.json"
    fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")

    result = proto.run_proto_frankenstein(
        module=module,
        body_path=tmp_path / "body.gravity",
        out_dir=tmp_path / "full",
        scores_fixture=fixture_path,
        dry_run=False,  # attempt forward; it must gate
        ensure_handoff=True,
    )
    assert result["dry_run"] is False
    assert result["forward_status"] == FORWARD_GATE
    assert result["capability_validated"] is False
    assert result["verdict"] == "FORWARD_GATED"


def test_full_run_with_forward_and_accept_scores_can_accept(tmp_path: Path) -> None:
    """When a real forward callable + accept scores exist, ACCEPT is allowed."""

    module = _synthetic_module()
    fixture = proto.make_accept_fixture_document()
    fixture_path = tmp_path / "ok.json"
    fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")

    def real_forward(**kwargs):
        return {
            "activations": {"post_moe_hidden_state": "captured"},
            "scores": {"from_forward": True},
        }

    result = proto.run_proto_frankenstein(
        module=module,
        body_path=tmp_path / "body.gravity",
        out_dir=tmp_path / "full-accept",
        scores_fixture=fixture_path,
        dry_run=False,
        forward=real_forward,
        ensure_handoff=True,
    )
    assert result["forward_status"] == "FORWARD_EXECUTED"
    assert result["ablation_verdict"] == "ACCEPT"
    assert result["verdict"] == "ACCEPT"
    assert result["capability_validated"] is True


def test_compose_proto_artifact_is_raw_not_gravity(tmp_path: Path) -> None:
    module = _synthetic_module()
    composed = proto.compose_proto_artifact(
        module=module,
        body_path=tmp_path / "body.gravity",
        out_dir=tmp_path / "compose",
    )
    assert composed["gravity_compressed"] is False
    assert composed["capability_claim"] is False
    assert composed["validation_status"] == FORWARD_GATE
    doc = composed["document"]
    assert doc["kind"] == "raw_proto_composition_descriptor"
    assert "gravity" not in doc["composition"]["mode"]
    # Every adapter block is GLM_MATH_BRIDGE only.
    for block in doc["adapter_manifest"]["blocks"]:
        assert block["bridge"] == "GLM_MATH_BRIDGE"
        assert block.get("kimi_strategic_bridge") is False


def test_resolve_forward_default_is_gated() -> None:
    resolution = proto.resolve_deepseek_forward()
    assert resolution["callable"] is False
    assert resolution["status"] == FORWARD_GATE
    assert resolution["gate"] == FORWARD_GATE


def test_cli_guard_and_forward_status() -> None:
    assert proto.main(["guard"]) == 0
    # No forward registered → exit 2
    assert proto.main(["forward-status"]) == 2


def test_cli_dry_run_with_synthetic_module_and_reject_fixture(tmp_path: Path) -> None:
    module = _synthetic_module()
    sealed = xfer.seal_transfer_module_files(module, out_dir=tmp_path / "mod")
    out = tmp_path / "cli-dry"
    # write-reject-fixture generates the coding-down fixture.
    code = proto.main(
        [
            "dry-run",
            "--module-meta",
            sealed["meta_path"],
            "--module-bin",
            sealed["module_path"],
            "--body-path",
            str(tmp_path / "body.gravity"),
            "--out-dir",
            str(out),
            "--write-reject-fixture",
        ]
    )
    assert code == 2  # REJECT exit
    receipt = json.loads((out / proto.RUN_RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["verdict"] == "REJECT"
    assert receipt["forward_status"] == FORWARD_GATE
    assert receipt["capability_validated"] is False
    assert receipt["stages"]["ablation_avb"]["reject_rule_fired"] is True
    # Coding regression domain present.
    domains = receipt["secondary_non_regression"]["domains"] or []
    failed = [d for d in domains if d.get("gate") == "FAIL"]
    assert any(d["domain"] == "coding_and_repository_work" for d in failed)


def test_guard_cli_matches_assert() -> None:
    result = proto.assert_no_training_path()
    assert result["transfer_guard"]["training_path_present"] is False
