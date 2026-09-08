"""The round reached the measurement and there was no tool that could make it.

Round 12 is the furthest any round has got. It chose Qwen3-Embedding-0.6B "over the 45
other owed specimens (1.4-1453.8 GiB) because it is the cheapest to measure" -- using the
hidden-tail range published one commit earlier -- guarded with real numbers, called
lake.census, and read the block correctly: "no key contains 'expert' so it owes a DENSE
anatomy, not an expert-organ one". It then called odyssey.anatomy and got exactly that
refusal back:

    310 tensors across 1 shards, but no expert organ is measurable at layer 0.
    no key contains 'expert'; this looks dense, not MoE (e.g. embed_tokens.weight)

A correct refusal naming a correct mechanism. And then nothing, because the registry has
only lake.census and odyssey.anatomy, and DENSE anatomy has no door at all.

The capability is not missing. `dense_anatomy.anatomy_from_safetensors` measured 34 bodies
for G002 -- it is simply unreachable from HCLI, which is the fourth instance of this exact
disease in one session: the ledger's write helpers, the closing turn's report request, the
observations channel, and now this.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcli.tool_registry import default_tool_registry

REPO = Path(__file__).resolve().parents[2]
DENSE = "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3-Embedding-0.6B@97b0c614be4d"


def _reg():
    return default_tool_registry(REPO, repo_root=REPO)


def test_a_dense_anatomy_tool_is_reachable_at_all():
    names = {t["name"] for t in _reg().discover()}
    assert "odyssey.dense_anatomy" in names, (
        "a round that correctly diagnoses 'this owes a DENSE anatomy' has no tool to make one")


def test_it_refuses_a_MoE_body_by_pointing_at_the_other_tool():
    """The two tools must name each other, or a round bounces between them."""
    spec = next(t for t in _reg().discover() if t["name"] == "odyssey.dense_anatomy")
    assert "odyssey.anatomy" in spec["description"], spec["description"]


def test_the_guard_runs_BEFORE_the_body_is_touched():
    """Dense anatomy ran 796 s and 6.59 GiB peak on the largest sweep body."""
    src = (REPO / "hcli" / "tool_registry.py").read_text()
    body = src.split("def _odyssey_dense_anatomy")[1].split("\ndef ")[0]
    # Skip the docstring. It NAMES anatomy_from_safetensors while explaining the defect,
    # and matching that made this test pass on prose rather than on order of execution.
    code = body.split('"""')[2]
    g = code.index("cmg.sample(")
    a = code.index("da.anatomy_from_safetensors(")
    assert g < a, "the body is opened before the guard is asked"
    assert 'snap.state == "STOP"' in code, "a STOP verdict is not handled, so the guard is decorative"


@pytest.mark.skipif(not Path(DENSE).is_dir(), reason="specimen not mounted")
def test_it_actually_measures_the_body_round_12_could_not():
    # The anatomy path needs mlx, which lives in /usr/local/bin/python3.12 and not in the
    # interpreter carrying pytest. Verified live there: 16.7 s, k_proj 41.09% leading.
    pytest.importorskip("mlx.core", reason="mlx lives in /usr/local/bin/python3.12")
    res = _reg().invoke("odyssey.dense_anatomy", {"snapshot": DENSE})
    assert res.ok, res.error
    v = res.value
    assert v.get("refused") is None, v["refused"]
    head = json.dumps(v)[:500]
    assert "organ_ordering" in head or "organs" in head, ("actionable-first: the ordering "
                                                          f"must survive 500 chars: {head}")
    ordering = v.get("organ_ordering") or []
    assert ordering, v
    assert all("organ" in o and "deficit_pct" in o for o in ordering), ordering[:2]
