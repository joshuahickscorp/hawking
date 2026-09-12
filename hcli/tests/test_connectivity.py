from __future__ import annotations

from unittest.mock import patch

from hcli.connectivity import _public_probe


class _Response:
    status = 204

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return b""


def test_public_probe_writes_integer_nanoseconds(monkeypatch):
    monkeypatch.setattr(
        "hcli.connectivity.urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(),
    )

    with patch("hcli.connectivity.now_ns", side_effect=[10_000, 10_417]), \
         patch("hcli.connectivity.since_ns", return_value=417):
        result = _public_probe("https://example.invalid/")

    assert result["timing_unit"] == "ns"
    assert result["elapsed_ns"] == 417
    assert isinstance(result["elapsed_ns"], int)
    assert "elapsed_ms" not in result
