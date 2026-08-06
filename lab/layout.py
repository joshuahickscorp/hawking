"""Canonical locations for Hawking's non-runtime workspace material.

The campaign records used to live at the repository root.  Keep their logical
names usable for historical receipts, but make live code resolve the compact
``workspace/`` layout from one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final


REPO_ROOT: Final = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT: Final = REPO_ROOT / "workspace"

CAMPAIGN_ROOT: Final = WORKSPACE_ROOT / "campaign"
CONFIG_ROOT: Final = CAMPAIGN_ROOT / "config"
EVIDENCE_ROOT: Final = CAMPAIGN_ROOT / "evidence"
GOVERNANCE_ROOT: Final = CAMPAIGN_ROOT / "governance"
RECORDS_ROOT: Final = CAMPAIGN_ROOT / "records"
DOCS_ROOT: Final = WORKSPACE_ROOT / "docs"
OPS_ROOT: Final = WORKSPACE_ROOT / "ops"
LOCAL_ROOT: Final = OPS_ROOT / "local"
QUALITY_ROOT: Final = WORKSPACE_ROOT / "quality"
VENDOR_ROOT: Final = WORKSPACE_ROOT / "vendor"

ADAPTERS_ROOT: Final = CONFIG_ROOT / "adapters"
PACKS_ROOT: Final = CONFIG_ROOT / "packs"
PREREGISTRATIONS_ROOT: Final = CONFIG_ROOT / "preregistrations"
PROFILES_ROOT: Final = CONFIG_ROOT / "profiles"
PROMPTS_ROOT: Final = CONFIG_ROOT / "prompts"

CONTROL_ROOT: Final = GOVERNANCE_ROOT / "control"
ODYSSEY_ROOT: Final = GOVERNANCE_ROOT / "odyssey"
RECEIPTS_ROOT: Final = GOVERNANCE_ROOT / "receipts"
REPORTS_ROOT: Final = RECORDS_ROOT / "reports"
RESEARCH_ROOT: Final = RECORDS_ROOT / "research"
TESTS_ROOT: Final = QUALITY_ROOT / "tests"

# Odyssey has a lot of tiny contractual surfaces.  Keep one semantic grouping
# layer so its root stays navigable without merging or rewriting those records.
ODYSSEY_DOMAINS_ROOT: Final = ODYSSEY_ROOT / "domains"
ODYSSEY_PROGRAM_ROOT: Final = ODYSSEY_ROOT / "program"
ODYSSEY_RESOURCES_ROOT: Final = ODYSSEY_ROOT / "resources"
ODYSSEY_STATE_ROOT: Final = ODYSSEY_ROOT / "state"
ODYSSEY_RECORDS_ROOT: Final = ODYSSEY_ROOT / "records"

ODYSSEY_AREAS: Final = {
    "doctrine": "domains",
    "economics": "domains",
    "forge": "domains",
    "instruments": "domains",
    "ledger": "domains",
    "memory": "domains",
    "roles": "domains",
    "sovereignty": "domains",
    "tribunal": "domains",
    "verifiers": "domains",
    "evaluation": "program",
    "experiments": "program",
    "launch": "program",
    "sandbox": "program",
    "substrate": "program",
    "t0": "program",
    "training": "program",
    "profiles": "resources",
    "retrieval": "resources",
    "teacher_traces": "resources",
    "checkpoints": "state",
    "data": "state",
    "graveyard": "state",
    "rollback": "state",
}
ODYSSEY_RECORDS: Final = frozenset({
    "ODYSSEY_CONTRACT_CLOSURE.json",
    "ODYSSEY_DRY_RUN.json",
    "ODYSSEY_FEASIBILITY.json",
    "ODYSSEY_HEAVY_PREREQUISITES.json",
    "ODYSSEY_PACKAGE.json",
    "ODYSSEY_T0_EXECUTABLE_RECEIPT.json",
    "ODYSSEY_T0_RECEIPT.json",
    "ODYSSEY_TRAINER_READINESS.json",
})

# The campaign name is intentionally retained in historical artifacts.  Only
# the physical area changed, so this table lets live readers bridge the two.
EVIDENCE_AREAS: Final = {
    "deepseek-v4": "models",
    "glm52": "models",
    "kimi-k26": "models",
    "qwen235b": "models",
    "acceleration": "runtime",
    "gravity": "runtime",
    "rebuild": "runtime",
    "tg": "runtime",
    "fabric": "systems",
    "hawking": "systems",
    "hide": "systems",
    "ramanujan": "systems",
    "doctor": "research",
    "one-mountain": "research",
    "overread": "research",
    "prometheus": "research",
}


def evidence_dir(campaign: str) -> Path:
    """Return the physical directory for a named evidence campaign."""
    try:
        area = EVIDENCE_AREAS[campaign]
    except KeyError as exc:
        raise ValueError(f"unknown evidence campaign: {campaign!r}") from exc
    return EVIDENCE_ROOT / area / campaign


def odyssey_path(*parts: str) -> Path:
    """Return the physical path for an Odyssey component or record."""
    if not parts:
        return ODYSSEY_ROOT
    first = parts[0]
    relative = Path(*parts)
    area = ODYSSEY_AREAS.get(first)
    if area is not None:
        return ODYSSEY_ROOT / area / relative
    if first in ODYSSEY_RECORDS:
        return ODYSSEY_RECORDS_ROOT / relative
    return ODYSSEY_ROOT / relative


def find_evidence(name: str) -> Path | None:
    """Find an evidence file by basename without changing historical receipts."""
    if not name or name != Path(name).name:
        raise ValueError(f"find_evidence expects a basename, got {name!r}")
    matches = sorted(EVIDENCE_ROOT.glob(f"*/*/{name}"))
    return next((path for path in matches if path.is_file()), None)


def find_control(name: str) -> Path | None:
    """Find a control record by basename across its compact role buckets."""
    if not name or name != Path(name).name:
        raise ValueError(f"find_control expects a basename, got {name!r}")
    matches = sorted(CONTROL_ROOT.rglob(name))
    return next((path for path in matches if path.is_file()), None)


def find_profile(name: str) -> Path | None:
    """Find a profile by basename across its model-family folders."""
    if not name or name != Path(name).name:
        raise ValueError(f"find_profile expects a basename, got {name!r}")
    matches = sorted(PROFILES_ROOT.rglob(name))
    return next((path for path in matches if path.is_file()), None)


def resolve_workspace_path(value: str | Path) -> Path:
    """Resolve an old root-relative campaign path to its current location.

    This is for live readers of historic metadata.  It deliberately does not
    rewrite the metadata itself: sealed receipts continue to state the path
    that was true when they were recorded.
    """
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(REPO_ROOT)
        except ValueError:
            return path
    if not path.parts:
        return REPO_ROOT
    if path.parts[0] == "workspace":
        return REPO_ROOT / path

    head, *tail = path.parts
    rest = Path(*tail)
    direct = {
        "adapters": ADAPTERS_ROOT,
        "build": OPS_ROOT / "build",
        "deploy": OPS_ROOT / "deploy",
        "packs": PACKS_ROOT,
        "preregistrations": PREREGISTRATIONS_ROOT,
        "prompts": PROMPTS_ROOT,
        "receipts": RECEIPTS_ROOT,
        "reports": REPORTS_ROOT,
        "tests": TESTS_ROOT,
        "vendor": VENDOR_ROOT,
    }
    if head in direct:
        return direct[head] / rest
    if head == "ramanujan":
        # Ramanujan retains sealed logical paths while its scaffold, records,
        # and governance material live in a compact local hierarchy.
        from ramanujan.layout import resolve_ramanujan_path

        return resolve_ramanujan_path(path)
    if head == "odyssey":
        return odyssey_path(*tail)
    if head == "evidence" and tail:
        campaign, *campaign_rest = tail
        try:
            return evidence_dir(campaign) / Path(*campaign_rest)
        except ValueError:
            return EVIDENCE_ROOT / rest
    if head == "profiles":
        candidate = PROFILES_ROOT / rest
        if candidate.exists():
            return candidate
        found = find_profile(path.name)
        return found if found is not None else candidate
    if head == "control":
        if tail and tail[0] in {"catalog", "ledgers", "receipts", "rungs", "verdicts"}:
            candidate = CONTROL_ROOT / rest
            if candidate.exists():
                return candidate
            found = find_control(path.name)
            return found if found is not None else candidate
        found = find_control(path.name)
        return found if found is not None else CONTROL_ROOT / rest
    if head == "research":
        return RESEARCH_ROOT / rest
    if head == "logo":
        return DOCS_ROOT / "assets" / "logo" / rest
    if head == "docs":
        candidate = DOCS_ROOT / rest
        if candidate.exists():
            return candidate
        matches = sorted(DOCS_ROOT.rglob(path.name))
        return next((match for match in matches if match.is_file()), candidate)
    return REPO_ROOT / path
