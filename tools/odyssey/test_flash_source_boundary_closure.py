from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.odyssey import flash_source_boundary_closure as closure
from tools.odyssey import flash_source_boundary_transaction as transaction


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def test_fresh_closure_is_consumable_by_the_transaction_verifier(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(closure, "ROOT", tmp_path)
    monkeypatch.setattr(closure, "RECEIPT_ROOT", tmp_path / "receipts" / "future")
    monkeypatch.setattr(transaction, "ROOT", tmp_path)
    for relative in (
        *closure.NATIVE_SOURCE_PATHS,
        *closure.HAWKING_SOURCE_PATHS,
        *closure.TRANSACTION_WRAPPER_PATHS,
    ):
        _write(tmp_path / relative, f"source:{relative}\n")
    _write(tmp_path / closure.NATIVE_WRAPPER, "native wrapper\n")
    _write(tmp_path / closure.COMPARISON_WRAPPER, "comparison wrapper\n")
    target_dir = tmp_path / "target" / "task029"
    native = target_dir / "release" / "examples" / "flash_stateful_complete_token_session"
    hawking = target_dir / "release" / "hawking"
    _write(native, "#!/bin/sh\necho native\n", executable=True)
    _write(hawking, "#!/bin/sh\necho hawking 0.2.2\n", executable=True)
    out = tmp_path / "receipts" / "future" / "closure.json"

    document = closure.build_closure(
        native_binary=native,
        hawking_binary=hawking,
        out=out,
        target_dir=target_dir,
        build_commands=["cargo build --release --example flash_stateful_complete_token_session"],
        predecessor=None,
    )

    assert document["native_diagnostic"]["executed"] is False
    assert document["runtime_actions"]["gpu_or_metal_started"] is False
    verified = transaction._verify_executable_closure(
        out,
        native_binary=native,
        hawking_binary=hawking,
    )
    assert verified["receipt"]["seal_sha256"] == document["seal_sha256"]
    assert len(verified["transaction_wrappers"]) == len(closure.TRANSACTION_WRAPPER_PATHS)


def test_transaction_closure_rejects_an_incomplete_python_surface(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(closure, "ROOT", tmp_path)
    monkeypatch.setattr(closure, "RECEIPT_ROOT", tmp_path / "receipts" / "future")
    monkeypatch.setattr(transaction, "ROOT", tmp_path)
    for relative in (
        *closure.NATIVE_SOURCE_PATHS,
        *closure.HAWKING_SOURCE_PATHS,
        *closure.TRANSACTION_WRAPPER_PATHS,
    ):
        _write(tmp_path / relative, f"source:{relative}\n")
    target_dir = tmp_path / "target" / "task029"
    native = target_dir / "release" / "examples" / "flash_stateful_complete_token_session"
    hawking = target_dir / "release" / "hawking"
    _write(native, "#!/bin/sh\necho native\n", executable=True)
    _write(hawking, "#!/bin/sh\necho hawking 0.2.2\n", executable=True)
    out = tmp_path / "receipts" / "future" / "closure.json"
    closure.build_closure(
        native_binary=native,
        hawking_binary=hawking,
        out=out,
        target_dir=target_dir,
        build_commands=["cargo build --release --example flash_stateful_complete_token_session"],
        predecessor=None,
    )
    document = json.loads(out.read_text(encoding="utf-8"))
    document["transaction_wrappers"] = document["transaction_wrappers"][:-1]
    document.pop("seal_sha256")
    document["seal_sha256"] = closure.gate._compact_utf8_sorted_seal(document)
    out.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="transaction wrapper closure is absent or incomplete"):
        transaction._verify_executable_closure(
            out,
            native_binary=native,
            hawking_binary=hawking,
        )
