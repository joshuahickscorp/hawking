from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from tools.odyssey import flash_source_boundary_transaction as transaction


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        root=tmp_path,
        reference_contract=tmp_path / "reference.json",
        owner_authorization=tmp_path / "authorization.json",
        preflight=tmp_path / "preflight.json",
        executable_closure=tmp_path / "closure.json",
        native_binary=tmp_path / "native",
        hawking_binary=tmp_path / "hawking",
        source_out_dir=tmp_path / "source",
        native_out=tmp_path / "native.json",
        comparison_out=tmp_path / "comparison.json",
        transaction_out=tmp_path / "transaction.json",
        diagnostic_attention_hc_source_bf16_norm=False,
        diagnostic_attention_hc_norm_threadgroup=None,
        diagnostic_attention_hc_norm_full_pairwise=False,
    )


def test_public_execution_refuses_owner_before_outputs_or_executables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.reference_contract.write_text("{}", encoding="utf-8")
    args.owner_authorization.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        transaction.gate,
        "load_admitted_reference_for_boundary",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("owner refused")),
    )
    monkeypatch.setattr(
        transaction,
        "_require_fresh_disjoint_outputs",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("outputs inspected before owner admission")),
    )
    monkeypatch.setattr(
        transaction,
        "_verify_executable_closure",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("binary inspected before owner admission")),
    )
    monkeypatch.setattr(
        transaction.source_capture,
        "_preflight",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("preflight consumed after owner refusal")),
    )
    with pytest.raises(ValueError, match="owner refused"):
        transaction._execute(args)
    assert not args.transaction_out.exists()


def test_leased_execution_requires_exact_direct_canonical_lane_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = tmp_path / "gpu-lane"
    monkeypatch.setenv("HAWKING_GPU_LANE_LOCK", str(lane))
    with pytest.raises(ValueError, match="requires the canonical Hawking GPU lane"):
        transaction._lease_evidence()

    lane.mkdir()
    (lane / "owner").write_text(transaction.LANE_NAME + "\n", encoding="utf-8")
    (lane / "pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="direct child"):
        transaction._lease_evidence()


def test_lane_path_refuses_broad_or_symlink_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWKING_GPU_LANE_LOCK", "/")
    with pytest.raises(ValueError, match="unsafe"):
        transaction._lane_path()
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("HAWKING_GPU_LANE_LOCK", str(link))
    with pytest.raises(ValueError, match="symlink"):
        transaction._lane_path()


def test_output_plan_rejects_existing_or_nested_namespaces(tmp_path: Path) -> None:
    source = tmp_path / "source"
    native = tmp_path / "native.json"
    comparison = tmp_path / "comparison.json"
    receipt = tmp_path / "transaction.json"
    targets = transaction._require_fresh_disjoint_outputs(
        source_out_dir=source,
        native_out=native,
        comparison_out=comparison,
        transaction_out=receipt,
    )
    assert len(targets) == 6
    comparison.write_text("occupied", encoding="utf-8")
    with pytest.raises(ValueError, match="already exist"):
        transaction._require_fresh_disjoint_outputs(
            source_out_dir=source,
            native_out=native,
            comparison_out=comparison,
            transaction_out=receipt,
        )
    with pytest.raises(ValueError, match="must not nest"):
        transaction._require_fresh_disjoint_outputs(
            source_out_dir=source,
            native_out=source / "native.json",
            comparison_out=tmp_path / "other.json",
            transaction_out=receipt,
        )


def test_leased_transaction_forwards_explicit_hc_precision_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.owner_authorization = None
    args.diagnostic_attention_hc_source_bf16_norm = True
    args.diagnostic_attention_hc_norm_full_pairwise = True
    args.reference_contract.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(transaction, "_lease_evidence", lambda: {"lease": "held"})
    monkeypatch.setattr(
        transaction.gate,
        "load_admitted_reference_for_boundary",
        lambda *args, **kwargs: {
            "seal_sha256": "reference-seal",
            "authorization_binding_sha256": None,
            "admission_class": "LOCAL_CANONICAL",
            "owner_authorization": None,
        },
    )
    monkeypatch.setattr(
        transaction.source_capture,
        "_preflight",
        lambda *args, **kwargs: ({}, {"sha256": "preflight-sha256"}),
    )
    monkeypatch.setattr(
        transaction,
        "_verify_executable_closure",
        lambda *args, **kwargs: {"closure": "exact"},
    )
    monkeypatch.setattr(
        transaction.gate,
        "_protected_native_lane",
        lambda: {"clean": True, "matches": []},
    )

    def capture(**kwargs):
        kwargs["out_dir"].mkdir()
        (kwargs["out_dir"] / "capture.json").write_text("{}", encoding="utf-8")
        return {"status": "CAPTURED"}

    def native_run(**kwargs):
        assert kwargs["diagnostic_attention_hc_source_bf16_norm"] is True
        assert kwargs["diagnostic_attention_hc_norm_threadgroup"] is None
        assert kwargs["diagnostic_attention_hc_norm_full_pairwise"] is True
        kwargs["out"].write_text("{}", encoding="utf-8")
        kwargs["out"].with_name(f"{kwargs['out'].stem}.runner.json").write_text(
            "{}", encoding="utf-8"
        )
        return {"status": "CAPTURED_NATIVE"}

    def ordered_compare(**kwargs):
        assert kwargs["diagnostic_attention_hc_source_bf16_norm"] is True
        assert kwargs["diagnostic_attention_hc_norm_threadgroup"] is None
        assert kwargs["diagnostic_attention_hc_norm_full_pairwise"] is True
        kwargs["out"].write_text("{}", encoding="utf-8")
        return {"status": "DIFFERENCE", "first_observed_difference": None}

    monkeypatch.setattr(transaction.source_capture, "capture", capture)
    monkeypatch.setattr(transaction.native, "run", native_run)
    monkeypatch.setattr(transaction.comparison, "compare", ordered_compare)

    result = transaction._leased_execute(args)

    assert result["diagnostic_attention_hc_source_bf16_norm"] is True
    assert result["diagnostic_attention_hc_norm_threadgroup"] == 256
    assert result["diagnostic_attention_hc_norm_full_pairwise"] is True


def test_transaction_command_forwards_explicit_hc_precision_mode(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.diagnostic_attention_hc_source_bf16_norm = True
    args.diagnostic_attention_hc_norm_threadgroup = 128
    command = transaction._command(args)
    assert command.count("--diagnostic-attention-hc-source-bf16-norm") == 1
    threadgroup_flag = "--diagnostic-attention-hc-norm-threadgroup"
    assert command.count(threadgroup_flag) == 1
    assert command[command.index(threadgroup_flag) + 1] == "128"


def test_transaction_command_forwards_full_pairwise_hc_precision_mode(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.diagnostic_attention_hc_source_bf16_norm = True
    args.diagnostic_attention_hc_norm_full_pairwise = True
    command = transaction._command(args)
    assert command.count("--diagnostic-attention-hc-source-bf16-norm") == 1
    assert command.count("--diagnostic-attention-hc-norm-full-pairwise") == 1
    assert "--diagnostic-attention-hc-norm-threadgroup" not in command


def test_historical_release_closure_is_rejected_after_source_guard_change() -> None:
    closure = transaction.ROOT / (
        "receipts/future/HAWKING_FLASH_SOURCE_BOUNDARY_EXECUTABLE_CLOSURE_"
        "PLE_I64_GUARD_20260912.json"
    )
    native = transaction.ROOT / (
        "target/source-boundary-ple-i64-guard-20260912/release/examples/"
        "flash_stateful_complete_token_session"
    )
    hawking = transaction.ROOT / "target/source-boundary-ple-i64-guard-20260912/release/hawking"
    with pytest.raises(ValueError, match="bytes differ from the executable-closure receipt"):
        transaction._verify_executable_closure(
            closure,
            native_binary=native,
            hawking_binary=hawking,
        )


def test_task029_lane_ancestry_closure_is_rejected_after_current_source_drift() -> None:
    closure = transaction.ROOT / (
        "receipts/future/HAWKING_FLASH_SOURCE_BOUNDARY_EXECUTABLE_CLOSURE_"
        "TASK029_LANE_ANCESTRY_20260914.json"
    )
    native = transaction.ROOT / (
        "target/flash-task029-local-canonical-20260914/release/examples/"
        "flash_stateful_complete_token_session"
    )
    hawking = transaction.ROOT / "target/flash-task029-local-canonical-20260914/release/hawking"
    with pytest.raises(ValueError, match="bytes differ from the executable-closure receipt"):
        transaction._verify_executable_closure(
            closure,
            native_binary=native,
            hawking_binary=hawking,
        )


def test_current_source_rebind_closure_is_rejected_after_historical_teacher_reader_change() -> None:
    closure = transaction.ROOT / (
        "receipts/future/HAWKING_FLASH_SOURCE_BOUNDARY_EXECUTABLE_CLOSURE_"
        "TASK029_CURRENT_SOURCE_REBIND_20260914.json"
    )
    with pytest.raises(ValueError, match="bytes differ from the executable-closure receipt"):
        transaction._verify_executable_closure(
            closure,
            native_binary=transaction.DEFAULT_NATIVE_BINARY,
            hawking_binary=transaction.DEFAULT_HAWKING_BINARY,
        )


def test_historical_teacher_rebind_closure_is_rejected_after_cache_contract_change() -> None:
    closure = transaction.ROOT / (
        "receipts/future/HAWKING_FLASH_SOURCE_BOUNDARY_EXECUTABLE_CLOSURE_"
        "TASK029_HISTORICAL_TEACHER_REBIND_20260914.json"
    )
    with pytest.raises(ValueError, match="bytes differ from the executable-closure receipt"):
        transaction._verify_executable_closure(
            closure,
            native_binary=transaction.DEFAULT_NATIVE_BINARY,
            hawking_binary=transaction.DEFAULT_HAWKING_BINARY,
        )


def test_current_default_closure_is_exact() -> None:
    verified = transaction._verify_executable_closure(
        transaction.DEFAULT_CLOSURE,
        native_binary=transaction.DEFAULT_NATIVE_BINARY,
        hawking_binary=transaction.DEFAULT_HAWKING_BINARY,
    )
    assert verified["native_executable"]["sha256"] == (
        "feaf3e2a261e079390f07ee901e8333220a217e19a927220541469c689852b7a"
    )
    assert verified["hawking_executable"]["sha256"] == (
        "dc4e65e468c5dd2273c94039efc7084295f6cbde6f6f3fbd7bad7c1b9239436b"
    )
