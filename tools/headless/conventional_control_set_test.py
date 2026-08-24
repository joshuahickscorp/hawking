"""Receipt discipline for CONVENTIONAL_CONTROL_SET.json. No GPU, no 27B load.

The harness writes the receipt. These checks refuse to remesure: a pytest
collection of tools/headless must not load a 27B onto the GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from conventional_control_set import (  # noqa: E402
    RECEIPT,
    SCHEMA,
    _walk_metrics,
    validate,
)


def _doc() -> dict:
    assert RECEIPT.is_file(), (
        f"missing {RECEIPT}; run: python3 tools/headless/conventional_control_set.py"
    )
    return json.loads(RECEIPT.read_text())


def test_harness_writes_conventional_control_set_receipt():
    doc = _doc()
    assert doc.get("schema") == SCHEMA
    problems = validate(doc)
    assert not problems, problems


def test_live_numbers_have_command_and_spread_or_absent_reason():
    doc = _doc()
    live = doc.get("live") or {}
    assert live.get("status") == "LIVE"
    for name, node in _walk_metrics(live):
        st = node.get("status")
        assert st in ("MEASURED", "ABSENT"), (name, st)
        if st == "MEASURED":
            assert node.get("command"), name
            assert node.get("value") is not None, name
            if name not in ("context_limit", "concurrency"):
                assert node.get("repetitions"), name
                assert len(node["repetitions"]) >= 2, name
            if name in ("startup", "prefill", "decode_tps", "tool_shaped_tps"):
                assert node.get("cold_and_warm_stated_separately") is True, name
        if st == "ABSENT":
            assert node.get("reason"), name
            assert node.get("value") not in (0, 0.0), name


def test_archived_numbers_labelled_archived_with_source():
    doc = _doc()
    arch = doc.get("archived") or {}
    assert arch.get("status") == "ARCHIVED"
    assert (arch.get("artifact") or {}).get("present") is False
    for name, node in _walk_metrics(arch):
        assert node.get("status") in ("ARCHIVED", "ABSENT"), (name, node.get("status"))
        assert node.get("status") != "MEASURED", name
        if node.get("status") == "ARCHIVED":
            assert node.get("source_receipt"), name


def test_historical_headline_is_archived():
    doc = _doc()
    h = (doc.get("comparison") or {}).get("historical_headline") or {}
    assert h.get("status") == "ARCHIVED"
    assert "GPU_ATTACK.json" in (h.get("source_receipt") or "")
