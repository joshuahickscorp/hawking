"""The choose() ask must never hide its own schema, and never advertise a scar.

CHOICE_JSON_PROBE measured the mechanism: a control clipped at MAX_PROMPT_CHARS
parsed 0 of 2 on sealed-3.14 and 0 of 2 on Qwen3-0.6B; with the schema in view
both went 2 of 2. A 0.6B specimen failing the same clipped ask is what rules out
"the 27B cannot do structured output" - it is the ask, not the body, not scale.

These are invariants over the ask, not a re-measurement of a model.
"""
from __future__ import annotations

import json

from tools.future import model_bearing as mb

SCHEMA_TAIL = '\nReturn JSON only: {"choice_id":"id"}'


def test_clip_keeps_the_schema_tail():
    prompt = "H" * (mb.MAX_PROMPT_CHARS * 3) + SCHEMA_TAIL
    clipped = mb._clip_keeping_tail(prompt)
    assert len(clipped) <= mb.MAX_PROMPT_CHARS
    assert clipped.endswith(SCHEMA_TAIL), "the clip ate the schema again"


def test_short_prompt_is_untouched():
    assert mb._clip_keeping_tail("short" + SCHEMA_TAIL) == "short" + SCHEMA_TAIL


def test_fit_drops_candidates_and_never_truncates_json():
    rows = [{"id": f"WU.{i:03d}", "title": "t" * 72, "gain": i} for i in range(400)]
    head = "Candidates:\n"
    compact, dropped = mb._fit_entries(head, rows, SCHEMA_TAIL, cap=len(rows))
    body = json.dumps(compact, sort_keys=True)
    assert len(head) + len(body) + len(SCHEMA_TAIL) <= mb.MAX_PROMPT_CHARS
    assert dropped > 0 and compact, "400 entries must not fit, and must not empty"
    json.loads(body)  # a clipped array would raise here


def test_fit_keeps_everything_when_it_already_fits():
    rows = [{"id": "WU.A", "title": "t", "gain": 1}]
    compact, dropped = mb._fit_entries("h\n", rows, SCHEMA_TAIL)
    assert dropped == 0 and len(compact) == 1


def test_fit_keeps_the_highest_ranked_candidate():
    rows = [{"id": f"WU.{i:03d}", "title": "t" * 72, "gain": 1000 - i} for i in range(400)]
    compact, _ = mb._fit_entries("h\n", rows, SCHEMA_TAIL, cap=len(rows))
    assert compact[0]["id"] == "WU.000", "dropping must trim the tail, not the head"
