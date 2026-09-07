import pytest
import json
from pathlib import Path

def test_router_share_receipt():
    path = Path("receipts/future/O003_ROUTER_SHARE.json")
    assert path.exists()
    with open(path, 'r') as f:
        data = json.load(f)
    assert data["router_bytes"] == 6819072
    assert data["total_bytes"] == 32815315552
    assert data["share_ppm"] == 208
    assert round(data["router_bytes"] / data["total_bytes"] * 1e6) == 208
