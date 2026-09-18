import json
import os


def test_o003_attn_share():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "receipts", "future", "O003_ATTN_SHARE.json")
    assert os.path.exists(path), "Receipt missing"
    with open(path) as f:
        data = json.load(f)
    assert round(data["attention_bytes"] / data["total_bytes"] * 1e6) == data["share_ppm"]
