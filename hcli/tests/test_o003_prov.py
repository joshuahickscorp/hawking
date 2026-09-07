import json
import os

def test_o003_prov():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "receipts", "future", "O003_PROV.json")
    assert os.path.exists(path), "receipt missing"
    with open(path) as f:
        data = json.load(f)
    assert data["specimen"] == "O003"
    assert data["arch"] == "MoE"
    assert len(data["killed"]) == 6
    assert data["expert_bytes"] == 28789702656
