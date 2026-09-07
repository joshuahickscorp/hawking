"""Image geometry, cross-checked against an INDEPENDENT tool.

The VMCP priority list in the overnight directive names "documents/images" and
"exact tool-grounded verification". file_eye already read PNG width/height; JPEG
returned kind only, so a JPEG's geometry was simply unknown.

Parsing a header and believing yourself is not verification. These tests require
agreement with `sips`, which read the same bytes through a different
implementation, and require DISAGREEMENT to be reported rather than smoothed over.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from hcli.agentos.vmcp.file_eye import _jpeg, _jpeg_geometry, verify_image_geometry

REAL_JPEG = Path("/System/Library/CoreServices/DefaultBackground.jpg")
FAKE_JPEG = Path("/usr/share/cups/ipptool/gray.jpg")  # actually gzip


def _need(p: Path):
    if not p.is_file():
        pytest.skip(f"fixture absent: {p}")


def test_jpeg_geometry_agrees_with_an_independent_tool():
    _need(REAL_JPEG)
    if not shutil.which("sips"):
        pytest.skip("sips absent; this check REQUIRES a second implementation")
    r = verify_image_geometry(REAL_JPEG)
    assert r["verdict"] == "AGREE", r
    assert r["eye"]["width"] == r["ground_truth"]["width"]
    assert r["eye"]["height"] == r["ground_truth"]["height"]
    assert r["eye"]["width"] > 0 and r["eye"]["height"] > 0


def test_a_file_named_jpg_that_is_not_one_is_refused_on_MAGIC():
    """Extension is not evidence. This fixture is gzip wearing a .jpg name."""
    _need(FAKE_JPEG)
    assert FAKE_JPEG.read_bytes()[:4].hex() == "1f8b0800", "fixture is no longer gzip"
    assert _jpeg(FAKE_JPEG.read_bytes()) is None
    assert verify_image_geometry(FAKE_JPEG)["verdict"] == "NOT_AN_IMAGE_BY_MAGIC"


def test_an_absent_target_is_not_an_empty_success(tmp_path):
    r = verify_image_geometry(tmp_path / "nothing.jpg")
    assert r["verdict"] == "NO_TARGET"
    assert r["eye"] is None


def test_a_corrupted_header_is_read_FAITHFULLY_by_both(tmp_path):
    """Corrupting the SOF does not create a disagreement -- and should not.

    My first version of this control asserted DISAGREE here. That was wrong:
    `sips` reads the SAME frame header, so both implementations faithfully report
    the corrupted value and AGREE. The eye is supposed to report what the file
    says, not what the pixels would have been.
    """
    _need(REAL_JPEG)
    if not shutil.which("sips"):
        pytest.skip("sips absent")
    data = bytearray(REAL_JPEG.read_bytes())
    i, n, patched = 2, len(data), False
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m == 0xFF or m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m in (0xDA, 0xD9):
            break
        seg = int.from_bytes(data[i + 2 : i + 4], "big")
        if m in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            data[i + 5 : i + 7] = (7).to_bytes(2, "big")
            patched = True
            break
        i += 2 + seg
    assert patched, "could not locate a SOF to corrupt; the control would be vacuous"
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(bytes(data))
    assert _jpeg_geometry(bytes(data))[1] == 7, "the corruption did not take"
    r = verify_image_geometry(bad)
    assert r["verdict"] == "AGREE", r
    assert r["eye"]["height"] == 7 and r["ground_truth"]["height"] == 7


def test_the_cross_check_actually_binds(tmp_path, monkeypatch):
    """THE negative control: make the eye lie, and the verdict must flip.

    Without this, every AGREE above could come from a comparison that never
    fails -- the vacuous-test failure mode.
    """
    _need(REAL_JPEG)
    if not shutil.which("sips"):
        pytest.skip("sips absent")
    import hcli.agentos.vmcp.file_eye as fe

    real = fe._jpeg

    def lying(data):
        got = real(data)
        if got and "height" in got:
            got = dict(got)
            got["height"] = got["height"] + 1   # off by exactly one
        return got

    monkeypatch.setattr(fe, "_jpeg", lying)
    r = fe.verify_image_geometry(REAL_JPEG)
    assert r["verdict"] == "DISAGREE", (
        "the eye reported a height one pixel off and the cross-check still said "
        "AGREE; the comparison does not bind"
    )
    assert r["agree"] is False
