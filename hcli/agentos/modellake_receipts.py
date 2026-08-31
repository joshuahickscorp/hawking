"""Canonical ModelLake receipt names with compatibility fallbacks.

The nomenclature migration changes how new observations are named, not the
identity or bytes of sealed historical receipts.  Readers should therefore
prefer the descriptive current names and continue to understand the original
HCLI names when a checkout only contains the legacy receipt.
"""
from __future__ import annotations

from os import PathLike
from pathlib import Path


CENSUS_RECEIPT_NAMES = (
    "MODELLAKE_FLASH_NEXT_CENSUS.json",
    "HCLI_MODELLAKE_FLASH_CENSUS.json",
)
SUPERVISION_RECEIPT_NAMES = (
    "MODELLAKE_FLASH_NEXT_SUPERVISION.json",
    "HCLI_MODELLAKE_FLASH_ACQUISITION_SUPERVISION.json",
)


def preferred_receipt(
    repo: str | PathLike[str] | Path,
    names: tuple[str, ...],
) -> Path:
    """Return the first present receipt, or the canonical name if absent."""
    headless = Path(repo).expanduser().resolve() / "receipts" / "headless"
    for name in names:
        candidate = headless / name
        if candidate.is_file():
            return candidate
    return headless / names[0]


def preferred_census_receipt(repo: str | PathLike[str] | Path) -> Path:
    return preferred_receipt(repo, CENSUS_RECEIPT_NAMES)


def preferred_supervision_receipt(repo: str | PathLike[str] | Path) -> Path:
    return preferred_receipt(repo, SUPERVISION_RECEIPT_NAMES)


__all__ = [
    "CENSUS_RECEIPT_NAMES",
    "SUPERVISION_RECEIPT_NAMES",
    "preferred_census_receipt",
    "preferred_receipt",
    "preferred_supervision_receipt",
]
