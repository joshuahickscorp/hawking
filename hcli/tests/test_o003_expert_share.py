import json
import pytest

def test_expert_share():
    path = 'receipts/future/O003_EXPERT_SHARE.json'
    with open(path, 'r') as f:
        data = json.load(f)
    assert round(data['expert_bytes'] / data['total_bytes'], 4) == 0.8773
    assert data['total_bytes'] == 32815315552
    assert data['expert_bytes'] == 28789702656
