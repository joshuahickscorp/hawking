"""Physical locations for the compact Ramanujan layout.

Historical receipts intentionally retain their original ``ramanujan/...``
logical paths.  Live readers use this module to find their byte-identical
physical homes without rewriting those sealed records.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RAMANUJAN_ROOT = Path(__file__).resolve().parent

SCAFFOLD_ROOT = RAMANUJAN_ROOT / "scaffold"
CORE_ROOT = SCAFFOLD_ROOT / "core"
RESEARCH_ROOT = SCAFFOLD_ROOT / "research"
GUARDS_ROOT = SCAFFOLD_ROOT / "guards"
TOOLING_ROOT = SCAFFOLD_ROOT / "tooling"
DATA_ROOT = SCAFFOLD_ROOT / "data"
DATA_PIPELINE_ROOT = DATA_ROOT / "pipeline"
DATA_RECORDS_ROOT = DATA_ROOT / "records"
DATA_TESTS_ROOT = DATA_ROOT / "tests"
TRAIN_ROOT = SCAFFOLD_ROOT / "train"
FIXTURES_ROOT = SCAFFOLD_ROOT / "fixtures"
TESTS_ROOT = SCAFFOLD_ROOT / "tests"
CONTAINER_ROOT = RAMANUJAN_ROOT / "container"

GOVERNANCE_ROOT = RAMANUJAN_ROOT / "governance"
BOUNDARY_ROOT = GOVERNANCE_ROOT / "boundary"
CONTRACTS_ROOT = GOVERNANCE_ROOT / "contracts"

RECORDS_ROOT = RAMANUJAN_ROOT / "records"
AUDITS_ROOT = RECORDS_ROOT / "audits"
INTAKE_ROOT = RECORDS_ROOT / "intake"
RUNTIME_RECORDS_ROOT = RECORDS_ROOT / "runtime"
DOCS_ROOT = RAMANUJAN_ROOT / "docs"


_SOURCE_ITEMS = {
    "controllers.py": CORE_ROOT / "controllers.py",
    "economics.py": CORE_ROOT / "economics.py",
    "evidence.py": CORE_ROOT / "evidence.py",
    "forge.py": CORE_ROOT / "forge.py",
    "ledger.py": CORE_ROOT / "ledger.py",
    "limits.py": CORE_ROOT / "limits.py",
    "roles.py": CORE_ROOT / "roles.py",
    "sovereignty.py": CORE_ROOT / "sovereignty.py",
    "stores.py": CORE_ROOT / "stores.py",
    "cognition.py": RESEARCH_ROOT / "cognition.py",
    "prover.py": RESEARCH_ROOT / "prover.py",
    "search.py": RESEARCH_ROOT / "search.py",
    "you_research.py": RESEARCH_ROOT / "you_research.py",
    "restream_guard.py": GUARDS_ROOT / "restream_guard.py",
    "toolchain_selftest.py": GUARDS_ROOT / "toolchain_selftest.py",
    "status.py": GUARDS_ROOT / "status.py",
    "RAMANUJAN_FINAL_PARENT_NEXT_COMMAND.sh": GUARDS_ROOT / "RAMANUJAN_FINAL_PARENT_NEXT_COMMAND.sh",
    "gen_data_matrix.py": TOOLING_ROOT / "gen_data_matrix.py",
    "data": DATA_ROOT,
    "train": TRAIN_ROOT,
    "fixtures": FIXTURES_ROOT,
    "container": CONTAINER_ROOT,
    "test_cognition.py": TESTS_ROOT / "test_cognition.py",
    "test_governance_invariants.py": TESTS_ROOT / "test_governance_invariants.py",
    "test_prover.py": TESTS_ROOT / "test_prover.py",
    "test_ramanujan.py": TESTS_ROOT / "test_ramanujan.py",
    "test_roles_economics.py": TESTS_ROOT / "test_roles_economics.py",
    "test_search.py": TESTS_ROOT / "test_search.py",
    "test_you_research.py": TESTS_ROOT / "test_you_research.py",
}
_DATA_PIPELINE_ITEMS = frozenset({
    "common.py",
    "contamination_pass.py",
    "extractors.py",
    "freeze_memberships.py",
    "generate.py",
    "parse_mathlib.py",
    "paths.py",
})
_DATA_TEST_ITEMS = frozenset({"test_data.py", "test_freeze.py"})

_BOUNDARY_ITEMS = {
    "RAMANUJAN_GOVERNANCE_STATUS.json",
    "RAMANUJAN_GREEN_LIGHT_TRANSITION.json",
    "RAMANUJAN_OFFLINE_MANIFEST.json",
    "RAMANUJAN_OWNER_DECISIONS_REQUIRED.json",
    "HAWKING_COMPLETION_GATE.json",
}
_CONTRACT_ITEMS = {
    "RAMANUJAN_FINAL_PARENT_RESTREAM_PLAN.json",
    "RAMANUJAN_GLM52_WINDOW_OPERATOR_OWNER_PROPOSAL.json",
    "RAMANUJAN_Q0_Q6_CONTRACTS.json",
    "RAMANUJAN_RESTREAM_PLAN.json",
    "RAMANUJAN_ROLES_ECONOMICS.json",
}
_AUDIT_ITEMS = {
    "RAMANUJAN_PRE_RESTREAM_AUDIT.json",
    "RAMANUJAN_PRE_RESTREAM_AUDIT_REFRESH_2026_07_31.json",
    "RAMANUJAN_PRE_RESTREAM_AUDIT_REFRESH_2026_08_01.json",
    "RAMANUJAN_Q0_CLOSURE.json",
    "RAMANUJAN_Q0_EVIDENCE_BUNDLE.json",
}
_INTAKE_ITEMS = {
    "RAMANUJAN_ACQUISITION_QUEUE.json",
    "RAMANUJAN_DATA_SOURCE_MATRIX.json",
    "RAMANUJAN_CORPUS_DETERMINISM.json",
}
_RUNTIME_ITEMS = {
    "RAMANUJAN_COGNITION_REGISTER.json",
    "RAMANUJAN_ENVIRONMENT_LOCK.json",
    "RAMANUJAN_TOOLCHAIN_SELFTEST.json",
    "SMALL_SYSTEM_TRAINING_STATUS.json",
}


def ramanujan_path(*parts: str | Path) -> Path:
    """Return the physical path for a current or legacy Ramanujan location."""
    tokens: list[str] = []
    for part in parts:
        tokens.extend(piece for piece in Path(part).parts if piece not in ("", "."))
    if tokens and tokens[0] == "ramanujan":
        tokens.pop(0)
    if not tokens:
        return RAMANUJAN_ROOT

    first, *rest = tokens
    if first == "data" and rest:
        second, *data_rest = rest
        if second in _DATA_PIPELINE_ITEMS:
            return (DATA_PIPELINE_ROOT / second).joinpath(*data_rest)
        if second in _DATA_TEST_ITEMS:
            return (DATA_TESTS_ROOT / second).joinpath(*data_rest)
        if second == "research_ledger.jsonl":
            return (DATA_RECORDS_ROOT / second).joinpath(*data_rest)
    if first in _SOURCE_ITEMS:
        return _SOURCE_ITEMS[first].joinpath(*rest)
    if first in _BOUNDARY_ITEMS:
        return (BOUNDARY_ROOT / first).joinpath(*rest)
    if first in _CONTRACT_ITEMS:
        return (CONTRACTS_ROOT / first).joinpath(*rest)
    if first in _AUDIT_ITEMS:
        return (AUDITS_ROOT / first).joinpath(*rest)
    if first in _INTAKE_ITEMS:
        return (INTAKE_ROOT / first).joinpath(*rest)
    if first in _RUNTIME_ITEMS:
        return (RUNTIME_RECORDS_ROOT / first).joinpath(*rest)
    return RAMANUJAN_ROOT.joinpath(*tokens)


def resolve_ramanujan_path(value: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a repo-relative historical path without altering sealed text."""
    raw = Path(value)
    if raw.is_absolute():
        try:
            relative = raw.relative_to(repo_root)
        except ValueError:
            return raw
    else:
        relative = raw
    tokens = relative.parts
    if tokens and tokens[0] == "ramanujan":
        return ramanujan_path(*tokens)
    if tokens and tokens[0] in _SOURCE_ITEMS:
        return ramanujan_path(*tokens)
    return repo_root / relative
