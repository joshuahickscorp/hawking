"""Every field the engine reads must be one the relay actually relays.

A resident fact reaches a receipt only by surviving three hops: the resident
emits it, hawking_native relays it into the `hawking` block, and the engine
allowlists it out again. Each hop has its own explicit list, and skipping the
middle one fails SILENTLY -- the receipt reads None, which is indistinguishable
from the resident not having sent it.

That has now happened twice. grammar_enforced was reported by the resident and
absent from every receipt. Then stop_reason and layers were added at the first
hop and the third, and still arrived as None, because the relay in the middle
was not touched.

This test is the hop-2 check: whatever the engine expects, the relay must
produce. It is deliberately derived from the engine's own list rather than a
copy of it, so a field added there cannot pass by being forgotten here.
"""
from __future__ import annotations

import inspect
import re

from hcli import engine as engine_mod
from hcli import hawking_native


def _engine_allowlist() -> set:
    """The keys the engine copies out of the `hawking` block."""
    src = inspect.getsource(engine_mod.Engine._record_model_call)
    body = src.split("for key in (", 1)[1].split("):", 1)[0]
    return set(re.findall(r'"([a-z_]+)"', body))


def _relay_keys() -> set:
    """The keys hawking_native puts INTO the `hawking` block."""
    src = inspect.getsource(hawking_native)
    block = src.split('"native_metrics": native_metrics,', 1)[1]
    block = block.split('return {', 1)[0]
    return set(re.findall(r'"([a-z_]+)":', block))


def test_every_field_the_engine_reads_is_one_the_relay_produces():
    expected = _engine_allowlist()
    produced = _relay_keys()
    missing = expected - produced
    assert not missing, (
        f"the engine reads {sorted(missing)} out of the hawking block, but "
        f"hawking_native never puts them there. The receipt will read None and "
        f"look exactly like a resident that did not send them."
    )


def test_the_two_fields_that_were_silently_dropped_are_present():
    produced = _relay_keys()
    assert "stop_reason" in produced
    assert "layers" in produced


def test_the_allowlists_were_actually_found():
    """A regex that matched nothing would make this file vacuously green."""
    assert len(_engine_allowlist()) >= 5
    assert len(_relay_keys()) >= 5
