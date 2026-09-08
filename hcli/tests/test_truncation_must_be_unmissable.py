"""HCLI reasoned correctly over 1.3% of the ledger and never knew.

Round 7 finally reported, and its report was a real scientific comparison: it chose
GLM-4.5-Air over GLM-5.3-Flash and DeepSeek-V4-Flash with named reasons (both
MOE-PREQUANTIZED, anatomy refused because a spectrum over quantization codes measures
the codebook not the organism, remaining axes refused because mlx_lm has no module for
glm5_next / deepseek_v4), and it explicitly REJECTED Qwen2.5-72B as "the next
candidate, not the one that moves the most right now".

It reached that over `fs.read` of the ledger, which returned `truncated: True` with
4,001 characters of a 299,504-byte file. Every specimen it discussed is in the first
few KB. It never saw the two bodies that still owe anatomy -- Inkling-Small and
Qwen3.8-Flash-Next -- because they were not in the 1.3% it was given.

The conclusion was sound over the evidence delivered. S006 28: prove the Resident had
the necessary evidence in its usable context before blaming its cognition.

`truncated: True` is technically honest and operationally useless: it says a cut
happened, not how much survived, so a caller cannot tell 99% from 1.3%. The disclosure
has to carry the proportion.
"""
from __future__ import annotations

from pathlib import Path

from hcli.tool_registry import default_tool_registry

REPO = Path(__file__).resolve().parents[2]
LEDGER = "receipts/future/G034_ODYSSEY_LEDGER.json"


def _reg():
    return default_tool_registry(REPO, repo_root=REPO)


def test_a_truncated_read_says_HOW_MUCH_survived():
    v = _reg().invoke("fs.read", {"path": LEDGER}).value
    assert v["truncated"] is True, "this fixture assumes the ledger is bigger than the cap"
    assert "shown_bytes" in v, "the caller cannot tell 99% from 1.3%"
    assert v["shown_bytes"] < v["bytes"]
    assert "truncation_note" in v, "no sentence a model will actually read"
    note = v["truncation_note"].lower()
    assert "%" in note or "percent" in note, note
    assert str(v["shown_bytes"]) in v["truncation_note"], v["truncation_note"]


def test_shown_bytes_is_WHAT_THE_CALLER_GETS_after_every_layer():
    """The invariant. Two truncations happened and only the first was disclosed.

    fs.read clips to 65,536 bytes and reported that; ToolResult.to_dict then redacts
    every string to 4,000 and reported nothing. So the number the caller was handed
    overstated what it actually received by 16x, on the exact file whose missing tail
    changed the conclusion.
    """
    res = _reg().invoke("fs.read", {"path": LEDGER})
    delivered = res.to_dict()["value"]["content"]
    shown = res.to_dict()["value"]["shown_bytes"]
    assert shown == len(delivered.rstrip("\u2026")), (
        f"shown_bytes={shown} but the caller received {len(delivered)} characters")


def test_an_untruncated_read_carries_no_scare_note():
    """The note must not become noise on every small file."""
    v = _reg().invoke("fs.read", {"path": "hcli/__init__.py"}).value
    assert v["truncated"] is False
    assert not v.get("truncation_note"), v.get("truncation_note")


def test_the_ledger_tool_reports_the_path_it_read():
    """So a round can chain read -> record without a human pasting the path in.

    Round 7 used fs.read instead of odyssey.ledger, and the reason is in the prompt I
    wrote: the record tool's example carried the literal ledger path, which put a file
    in front of the model and it read the file. Returning the path from the read tool
    removes the need to name it anywhere.
    """
    r = _reg()
    # EVERY branch, not just the one this test happened to call first: dropping `path`
    # from the slug branch left this green, which is how a half-wired fix ships.
    owed = r.invoke("odyssey.ledger", {"owed_only": True}).value
    assert owed.get("path", "").endswith("G034_ODYSSEY_LEDGER.json"), owed.keys()
    slug = owed["owed"][0]["slug"]
    one = r.invoke("odyssey.ledger", {"slug": slug}).value
    assert one.get("path", "").endswith("G034_ODYSSEY_LEDGER.json"), one.keys()
    summary = r.invoke("odyssey.ledger", {}).value
    assert summary.get("path", "").endswith("G034_ODYSSEY_LEDGER.json"), summary.keys()
