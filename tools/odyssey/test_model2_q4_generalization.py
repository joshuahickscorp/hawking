"""Pins the model-2 Qwen30 Q4 generalization receipt and its claim boundary."""
import json
from pathlib import Path

from lab.receipts import verify

REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts/headless/NOETIC_MODEL2_Q4_GENERALIZATION.json"


def receipt():
    return json.loads(RECEIPT.read_text())


def test_all_generalization_checks_pass():
    document = receipt()
    verify(document, label=str(RECEIPT))
    assert document["pass"] is True
    assert all(document["checks"].values())
    assert document["specimen"]["repository"] == "Qwen/Qwen3-30B-A3B"


def test_receipt_does_not_promote_wiring_to_physical_capability():
    boundary = receipt()["claim_boundary"]
    assert boundary["current_source_shard_revalidation_completed"] is True
    assert boundary["full_qwen30base_q4_artifact_packed"] is False
    assert boundary["uniform_q4_artifact_admitted"] is False
    assert boundary["native_token_executed"] is False
    assert boundary["cuda_hardware_execution"] is False
