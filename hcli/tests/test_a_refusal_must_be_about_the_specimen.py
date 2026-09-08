"""The loop closed, and the science it wrote was wrong.

Round 13 is the first autonomous ledger transition of the campaign. It called
odyssey.record_measurement itself and the ledger moved: REFUSED 124 -> 125, OWED 174 ->
173. The mechanism works.

What it recorded does not. It passed the LEDGER'S OWN PATH to odyssey.dense_anatomy as
the snapshot, got back a correct complaint about that argument, and stored the complaint
as the specimen's nr_candidate disposition:

    "odyssey.dense_anatomy returned a named refusal: the ledger's path is a file, not a
     directory, so it cannot be a a snapshot for a DENSE anatomy"

That is a fact about a bad tool call written into the ledger as a fact about
Qwen3-Embedding-0.6B. The existing guard could not see it: `refused()` demands a reason
of at least 20 characters naming a mechanism, and this one is long and does name a
mechanism -- just not one about the specimen.

The invariant that was missing: a refusal is EVIDENCE ABOUT A SPECIMEN, and the ledger
being written to is never evidence about anything in it. Self-reference is the tell, it
is cheap to check, and it is exactly what happened.

Deliberately NOT a general "is this reason relevant" judgement. That cannot be checked and
would reject the two legitimate refusals already on disk, neither of which names its slug:
"expert organ is pre-quantized on disk: expert payload is ['U8'] with ['weight_scale']"
and "cannot be executed on this host: mlx_lm has no module for model_type 'kimi_k3'".
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hcli.tool_registry import default_tool_registry

REPO = Path(__file__).resolve().parents[2]
LIVE = REPO / "receipts" / "future" / "G034_ODYSSEY_LEDGER.json"


@pytest.fixture
def ledger(tmp_path):
    d = REPO / "workspace"
    d.mkdir(exist_ok=True)
    p = d / f"_test_selfref_{tmp_path.name}.json"
    shutil.copy(LIVE, p)
    yield p
    p.unlink(missing_ok=True)
    (p.parent / (p.name + ".tmp")).unlink(missing_ok=True)


def _reg():
    return default_tool_registry(REPO, repo_root=REPO)


def _slug(path: Path) -> str:
    return json.loads(path.read_text())["specimens"][0]["slug"]


def test_a_refusal_about_the_ledger_itself_is_rejected(ledger):
    """The round-13 entry, verbatim in shape."""
    res = _reg().invoke("odyssey.record_measurement", {
        "path": str(ledger), "slug": _slug(ledger), "axis": "nr_candidate",
        "reason": (f"odyssey.dense_anatomy returned a named refusal: {ledger.name} is a file, "
                   f"not a directory, so it cannot be a snapshot for a DENSE anatomy")})
    assert not res.ok, "a refusal about the ledger was written into the ledger"
    assert "about the specimen" in str(res.error).lower(), res.error


def test_the_full_path_form_is_rejected_too(ledger):
    res = _reg().invoke("odyssey.record_measurement", {
        "path": str(ledger), "slug": _slug(ledger), "axis": "gpu",
        "reason": f"cannot proceed: {ledger} is not a snapshot directory for this body"})
    assert not res.ok, res.value


def test_the_two_legitimate_refusals_already_on_disk_still_pass(ledger):
    """The guard must not reject real mechanisms. Neither of these names its slug."""
    r = _reg()
    for axis, reason in [
        ("nr_candidate", "expert organ is pre-quantized on disk: expert payload is ['U8'] "
                         "with ['weight_scale'] scale tensors -- a spectrum over quantization "
                         "codes measures the codebook, not the organism"),
        ("gpu", "cannot be executed on this host: mlx_lm has no module for model_type "
                "\"kimi_k3\""),
    ]:
        res = r.invoke("odyssey.record_measurement", {
            "path": str(ledger), "slug": _slug(ledger), "axis": axis, "reason": reason})
        assert res.ok, f"{axis}: a legitimate mechanism was rejected: {res.error}"


def test_a_receipt_path_is_still_citable(ledger):
    """Citing evidence under receipts/ is normal and must keep working."""
    res = _reg().invoke("odyssey.record_measurement", {
        "path": str(ledger), "slug": _slug(ledger), "axis": "cpu",
        "reason": "no CPU path exists for this architecture; see "
                  "receipts/future/G009_GPU_OWED_MLX_MODULES.json for the module survey"})
    assert res.ok, res.error


def test_the_LEDGER_cannot_be_its_own_receipt(ledger):
    """Round 16 closed the loop and cited the ledger as the evidence for its own cell.

    It wrote nr_candidate = "16.0" with receipt receipts/future/G034_ODYSSEY_LEDGER.json --
    circular, and both existing guards let it through. The refusal guard only inspects
    `reason`, and the existence check passes because the ledger obviously exists.

    Same invariant as the refusal case, other field: a cell's evidence cannot be the file
    the cell lives in.
    """
    res = _reg().invoke("odyssey.record_measurement", {
        "path": str(ledger), "slug": _slug(ledger), "axis": "nr_candidate",
        "value": "16.0", "receipt": str(ledger)})
    assert not res.ok, "the ledger was accepted as evidence for a cell inside itself"
    assert "own receipt" in str(res.error).lower() or "circular" in str(res.error).lower(), res.error


def test_the_error_names_HOW_to_make_a_receipt(ledger):
    """A round that cannot make a receipt will cite whatever file it already knows.

    HCLI has filesystem.write (reversible_repo) and could have written one. Nothing told
    it that, and nothing in the failure said so either, so it reached for the only path in
    its context. The error is the place to say it -- not a second write tool. [S008 3]
    ONE OWNER, ONE GUARD, ONE WRITE PATH.
    """
    res = _reg().invoke("odyssey.record_measurement", {
        "path": str(ledger), "slug": _slug(ledger), "axis": "gpu",
        "value": {"tps": 1.0}, "receipt": "receipts/future/NOT_WRITTEN_YET.json"})
    assert not res.ok
    assert "filesystem.write" in str(res.error), res.error
