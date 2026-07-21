from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import kimi_k26_campaign as campaign  # noqa: E402


def test_telegram_outbox_advances_only_after_delivery(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(campaign, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(campaign, "TELEGRAM_OUTBOX", tmp_path / "outbox.json")
    monkeypatch.setattr(campaign, "NOTIFY_STATE", tmp_path / "delivery.json")
    monkeypatch.setattr(campaign, "write_status", lambda state: None)
    monkeypatch.setattr(campaign, "status_snapshot", lambda state: {
        "progress": {}, "progress_text": "1/2", "eta_text": "1m",
        "resources": {"free_disk_bytes": 100 * 1024**3,
                      "available_bytes_estimate": 50 * 1024**3},
        "complete_bpw": None, "primary_metrics": None, "best_candidate": None,
        "next_action": "continue",
    })
    delivered = []
    monkeypatch.setattr(campaign, "telegram", lambda message: bool(delivered))
    state = campaign.initial_state()
    campaign.checkpoint(state, "unit:test", "outbox")
    pending = json.loads((tmp_path / "outbox.json").read_text())
    assert len(pending) == 1
    assert not (tmp_path / "delivery.json").exists()
    delivered.append(True)
    campaign.flush_telegram_outbox()
    assert json.loads((tmp_path / "outbox.json").read_text()) == []
    receipt = json.loads((tmp_path / "delivery.json").read_text())
    assert receipt["checkpoint_id"] == "unit:test"
    assert receipt["seal_sha256"] == pending[0]["seal_sha256"]


def test_seal_validator_rejects_mutation() -> None:
    value = campaign.seal({"status": "PASS", "value": 1})
    assert campaign.valid_seal(value)
    value["value"] = 2
    assert not campaign.valid_seal(value)
