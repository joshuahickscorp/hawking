"""Four BUILT gates had no test citing them. This is that test.

FLASH_DENSE_VS_NF_AB, QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED,
QWEN27_RUNTIME_IDENTITY_FROZEN and VMCP_RECEIPT_LAW are wired and
acceptance-receipted, and nothing verified them -- the shape the
defining-property law warns about, where the only evidence a capability works is
a document nobody compared against the obligation.

SCHEMA DRIFT IS REAL AND THIS TEST ACCOMMODATES IT DELIBERATELY. Across 68
acceptance receipts there are three `checks` shapes (absent 36, dict 18,
list-of-dict 14) and three `criterion` shapes (quoted 29, absent 28, quote 11).
Assuming one shape is how a verifier silently mis-reads two thirds of the corpus,
which happened twice while writing this: FLASH_DENSE_VS_NF_AB looked
"ACCEPTED with zero checks" because it records `measured`/`comparison` instead,
and QWEN27_RUNTIME_IDENTITY_FROZEN looked criterion-less because it uses `quote`
rather than `quoted`. Both had verified fine.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# STATIC imports, not importlib.import_module(name). The auditor matches
# verifiers by AST, so a dynamic import is invisible to it -- these gates read
# tests=0 with this very file committed until the imports were written out.
# Same class of blindness as the sibling-import miss: the analyzer sees names,
# not intentions.
from hcli.agentos import flash_router_representation_ab, qwen27_mlp_diagnostic
from hcli.agentos import qwen27_runtime_identity, vmcp_gate

REPO = Path(__file__).resolve().parents[2]
ACCEPTANCE = REPO / "receipts" / "acceptance"

# Importing each gate's module is what ties this test to the gate: an auditor
# matches verifiers by the module a test cites, so a test that checks a
# capability without naming it is invisible to the graph.
GATES = {
    "VMCP_RECEIPT_LAW": "hcli.agentos.vmcp_gate",
    "QWEN27_RUNTIME_IDENTITY_FROZEN": "hcli.agentos.qwen27_runtime_identity",
    "QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED": "hcli.agentos.qwen27_mlp_diagnostic",
    "FLASH_DENSE_VS_NF_AB": "hcli.agentos.flash_router_representation_ab",
}


def _receipt(gate: str) -> dict:
    path = ACCEPTANCE / f"{gate}.json"
    if not path.is_file():
        pytest.skip(f"{gate} has no acceptance receipt on this host")
    return json.loads(path.read_text())


def _criterion_text(doc: dict) -> str:
    """Every known criterion shape, because there are FOUR.

        criterion.quoted        29 receipts
        criterion.quote         11
        criterion_quoted        top-level, written by tools/acceptance/agentos
        criterion as a string   rare

    Missing the third variant made Claude report "8 of 48 gate verdicts have no
    criterion" and, before that, "28 of 68". Both were false: EVERY gate verdict
    records its criterion. The drift is real; the missing criteria were not.
    """
    c = doc.get("criterion")
    if isinstance(c, dict):
        text = c.get("quoted") or c.get("quote") or ""
    else:
        text = c or ""
    return str(text or doc.get("criterion_quoted") or "")


def _evidence_count(doc: dict) -> int:
    """Every known evidence shape: checks-as-list, checks-as-dict, measured."""
    ch = doc.get("checks")
    if isinstance(ch, list):
        return len(ch)
    if isinstance(ch, dict):
        return len(ch)
    if doc.get("measured") or doc.get("comparison"):
        return 1
    return 0


_MODULES = {
    "VMCP_RECEIPT_LAW": vmcp_gate,
    "QWEN27_RUNTIME_IDENTITY_FROZEN": qwen27_runtime_identity,
    "QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED": qwen27_mlp_diagnostic,
    "FLASH_DENSE_VS_NF_AB": flash_router_representation_ab,
}

_SYMBOLS = {
    "VMCP_RECEIPT_LAW": "run_vmcp_gate",
    "QWEN27_RUNTIME_IDENTITY_FROZEN": "run_runtime_archaeology",
    "QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED": "run_qwen27_mlp_diagnostic_ab",
    "FLASH_DENSE_VS_NF_AB": "run_flash_router_representation_ab",
}


@pytest.mark.parametrize("gate", sorted(GATES))
def test_the_symbol_the_gate_is_catalogued_against_exists(gate):
    """A gate catalogued against a symbol that does not exist is not built.

    The catalogue names one symbol per gate as the thing whose call counts. If
    that symbol is absent, every caller the auditor found was matching something
    else, and the gate's wiring evidence is meaningless.
    """
    mod = _MODULES[gate]
    name = _SYMBOLS[gate]
    assert callable(getattr(mod, name, None)), (
        f"{gate} is catalogued against {mod.__name__}.{name}, which is not callable"
    )


@pytest.mark.parametrize("gate", sorted(GATES))
def test_an_accepted_verdict_states_the_criterion_it_was_judged_against(gate):
    """A verdict with no criterion cannot be re-checked, only re-trusted."""
    doc = _receipt(gate)
    if doc.get("verdict") != "ACCEPTED":
        pytest.skip(f"{gate} is not ACCEPTED")
    text = _criterion_text(doc)
    assert text.strip(), (
        f"{gate} was ACCEPTED without recording the criterion it was judged "
        "against; nobody can ask later whether it still holds"
    )


@pytest.mark.parametrize("gate", sorted(GATES))
def test_an_accepted_verdict_shows_evidence_and_did_not_weaken_itself(gate):
    doc = _receipt(gate)
    if doc.get("verdict") != "ACCEPTED":
        pytest.skip(f"{gate} is not ACCEPTED")
    assert _evidence_count(doc) > 0, f"{gate} is ACCEPTED with no evidence of any shape"
    weakened = doc.get("criterion_weakened")
    altered = doc.get("criterion_altered")
    assert weakened in (False, None) and altered in (False, None), (
        f"{gate}: weakened={weakened} altered={altered}"
    )


def test_the_canonical_readers_know_every_shape_in_the_corpus():
    """Sweep every gate verdict with the shared readers, not with a guess.

    This assertion is only trustworthy because the readers were corrected FOUR
    times, each after a miscount:

        criterion.quoted / criterion.quote / criterion_quoted / bare string
        checks list / checks dict / measured / comparison / run

    and because BLOCKED verdicts are excluded: a receipt honestly reporting that
    a gate did not run correctly carries no checks, and demanding evidence from
    it manufactures findings.

    If a future producer invents a sixth shape, this fails and the reader gets
    taught -- which is the point. The alternative is another false alarm.
    """
    import glob

    from tools.acceptance import criterion_text, evidence_count, verdict_needs_evidence

    missing_criterion, missing_evidence, claiming = [], [], 0
    for path in sorted(glob.glob(str(ACCEPTANCE / "*.json"))):
        name = Path(path).stem
        if "." in name:          # .gate/.run/.cycle sidecars are not verdicts
            continue
        try:
            doc = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict) or "verdict" not in doc:
            continue
        if not verdict_needs_evidence(doc):
            continue
        claiming += 1
        if not criterion_text(doc).strip():
            missing_criterion.append(name)
        if evidence_count(doc) == 0:
            missing_evidence.append(name)

    assert claiming > 0, "no claiming verdicts found; the sweep is not looking at anything"
    assert not missing_criterion, f"verdicts with no criterion in ANY shape: {missing_criterion}"
    assert not missing_evidence, f"verdicts with no evidence in ANY shape: {missing_evidence}"
