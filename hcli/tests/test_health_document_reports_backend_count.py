"""The health document must say how many backends it normalised.

_normalize_document builds a record for every name in KNOWN_BACKENDS and then
returns version, updated_at and backends. A reader that wants to know how many
backends the document covers has to count the dict itself, and a truncated or
partially written document is indistinguishable from a complete one.

This test is the SPEC. It fails before the change and passes after it. The fix
is one line: the value is already computed and in scope at the return.
"""
from __future__ import annotations

from hcli.resources import KNOWN_BACKENDS, BackendHealth


def _normalized(data):
    return BackendHealth.__new__(BackendHealth)._normalize_document(data)


def test_the_document_reports_how_many_backends_it_covers():
    out = _normalized({})
    assert out["backend_count"] == len(KNOWN_BACKENDS)


def test_the_count_matches_the_backends_actually_present():
    out = _normalized({"backends": {}})
    assert out["backend_count"] == len(out["backends"])


def test_the_existing_keys_are_untouched():
    out = _normalized({"updated_at": 12.5})
    for key in ("version", "updated_at", "backends"):
        assert key in out
    assert out["updated_at"] == 12.5
