"""A compiled GPU blob may not be trusted by its file extension.

The parent campaign recorded attempt 4 as a known open hole: model information
moved into a `.metallib` booked as MACHINE_SPECIFIC because the accountant read
the suffix and never opened the file. Its own prescription was "path is not
provenance" -- a fatbin is exactly where learned constant tables have always
lived, and this campaign's whole pressure is toward lower EBPW, which is the
pressure that makes hiding bytes attractive.

Watched failing twice while closing it:
  v1 scanned floats only from offset 0. The planted blob has an 18-byte header,
     so the payload was misaligned and the detector scored 0 on the blob while
     scoring the full 512 bytes on the payload alone. An accountant defeated by
     an odd-sized header is not an accountant.
  v2 scans all four byte alignments and catches it.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "headless"))
P = REPO / "receipts" / "headless" / "NOETIC_INFORMATION_ACCOUNTING.json"


def _d():
    return json.loads(P.read_text())


def test_all_five_plants_are_counted():
    t = _d()["canary"]["content_tally"]
    assert t["MISSED"] == 0, f"a hiding attempt is uncaught: {t}"
    assert t["COUNTED"] == 5


def test_the_metallib_plant_specifically_is_counted():
    a = [x for x in _d()["canary"]["isolated"] if x["name"] == "machine_specific_metallib"][0]
    assert a["content"] == "COUNTED", (
        "the compiled-blob hide is the one that was open; it must stay closed"
    )


def test_path_only_classifier_still_fails_everything():
    """The negative control. If naive path classification started passing, the
    content accountant would no longer be doing any work."""
    t = _d()["canary"]["path_tally"]
    assert t["COUNTED"] == 0 and t["MISSED"] == 5


def test_detector_is_alignment_independent():
    import noetic_information_accounting as A

    payload = A.canary_f32_payload(128, b"hide-4-metallib")
    for header_len in (0, 1, 2, 3, 18, 33):
        blob = b"\x00" * header_len + payload
        got = A.embedded_weightlike_bytes(blob)
        assert got >= len(payload) - 4, (
            f"header of {header_len} bytes defeated the detector: {got} of {len(payload)}"
        )


def test_no_false_positive_inflation_of_the_artifact():
    aa = _d()["artifact_accounting"]["identity"]
    assert aa["payload_matches_manifest"] is True
    assert aa["payload_bytes_sum"] == 14297694680


def test_completeness_is_not_overclaimed():
    c = _d()["canary"]["completeness"].lower()
    assert "would still miss" in c or "not complete" in c, (
        "catching five plants is not completeness; the remaining gap must be named"
    )


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print(f"ok  {n}")
    print("6/6 passed")
