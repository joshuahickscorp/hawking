"""The stale DSV4F launchd entrypoint must obey the model seal."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ENTRYPOINT = REPO / "tools" / "condense" / "deepseek_v4_resource_supervisor.py"


def _load():
    spec = importlib.util.spec_from_file_location("dsv4f_supervisor_entrypoint", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sealed_receipt_refuses_reconstruction(tmp_path: Path, capsys) -> None:
    module = _load()
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({"status": module.SEALED_STATUS}), encoding="utf-8")
    assert module.sealed_by_policy(seal)

    original = module.SEAL_RECEIPT
    module.SEAL_RECEIPT = seal
    try:
        assert module.main([]) == 0
    finally:
        module.SEAL_RECEIPT = original
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["status"] == "SEALED_MODEL_REFUSED"
    assert refusal["model"] == "dsv4f"
    assert refusal["weights_reconstructed"] is False


def test_missing_or_nonsealed_receipt_does_not_forge_a_seal(tmp_path: Path) -> None:
    module = _load()
    assert not module.sealed_by_policy(tmp_path / "missing.json")
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"status": "ACTIVE"}), encoding="utf-8")
    assert not module.sealed_by_policy(other)
