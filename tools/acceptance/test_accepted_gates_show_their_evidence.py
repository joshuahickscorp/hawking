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
    """Every known criterion shape, because there are three."""
    c = doc.get("criterion")
    if isinstance(c, dict):
        return str(c.get("quoted") or c.get("quote") or "")
    return str(c or "")


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


@pytest.mark.parametrize("gate,module", sorted(GATES.items()))
def test_the_module_the_gate_names_actually_imports(gate, module):
    """A gate catalogued against a module that cannot import is not built."""
    import importlib

    mod = importlib.import_module(module)
    assert mod is not None, f"{gate} names {module}, which does not import"


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
