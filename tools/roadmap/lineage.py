"""Resolve the canonical H-ROADMAP, which lives outside the repository.

The roadmap is external by deliberate user placement at ``~/Downloads/H-ROADMAP.md``.
External means it can vanish without a commit, and it did: twelve modules hardcode
that path, ``tools/roadmap/parse.py`` raises on it, and ``tools/roadmap/recompile.py``
silently substitutes an empty file. Worse, four acceptance harnesses degrade a
missing roadmap into a *placeholder string*, so a receipt could swear
``criterion_altered: false`` while quoting ``<H-ROADMAP.md not readable at ...>``.

A byte-identical copy has been preserved in-repo since the 2026-09-02 supersession
and nothing pointed at it. This module points at it.

The preserved copy is admitted ONLY when its sha256 matches the digest recorded in
``docs/roadmap-lineage/PRESERVATION.md``. That check is load-bearing, not ceremony:
PRESERVATION.md also records an EARLIER 9028-line roadmap at
``/Volumes/corpdrive/H-ROADMAP.md``, and a resolver that accepted any file of the
right name would parse the wrong authority in silence -- every acceptance span in
the catalog is a LINE RANGE, so a roadmap of the wrong length quotes the wrong text
while looking perfectly well-formed.

Resolution order:

    $H_ROADMAP                          explicit override, taken as given
    ~/Downloads/H-ROADMAP.md            canonical external authority, taken as given
    docs/roadmap-lineage/H-ROADMAP...   preserved copy, admitted only on digest match

The two authoritative locations are taken as given because they are the authority:
if the operator edits the canonical roadmap, the new text wins. The lineage copy is
a *record* of one specific document, so it must prove it is still that document.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Where the operator keeps the canonical roadmap. Outside the repo on purpose.
EXTERNAL = Path.home() / "Downloads" / "H-ROADMAP.md"

#: The preserved lineage copy, taken 2026-09-02 before the recompilation.
PRESERVED = REPO / "docs" / "roadmap-lineage" / "H-ROADMAP.superseded-2026-09-02.md"

#: sha256 of the 9645-line superseded canonical roadmap, per PRESERVATION.md.
PRESERVED_SHA256 = "d43a6b07ab9590bc11c265bfe8a1466131cce291b0622c076370a01d811328e4"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preserved_is_intact() -> bool:
    """True when the in-repo lineage copy is still the document it claims to be."""
    try:
        return _digest(PRESERVED) == PRESERVED_SHA256
    except OSError:
        return False


def roadmap_path() -> Path:
    """The canonical roadmap, or raise naming every location that was tried.

    Never returns a path that does not exist, and never returns the lineage copy
    when its digest disagrees with PRESERVATION.md.
    """
    override = os.environ.get("H_ROADMAP")
    if override:
        return Path(override)
    if EXTERNAL.is_file():
        return EXTERNAL
    if PRESERVED.is_file():
        if preserved_is_intact():
            return PRESERVED
        raise FileNotFoundError(
            f"canonical roadmap not readable: {EXTERNAL} is absent and the preserved "
            f"copy {PRESERVED} does not match the recorded digest "
            f"{PRESERVED_SHA256} (found {_digest(PRESERVED)}). Refusing to parse a "
            "roadmap whose line numbers may not be the ones the catalog cites."
        )
    raise FileNotFoundError(
        f"canonical roadmap not readable: tried $H_ROADMAP, {EXTERNAL}, {PRESERVED}"
    )


def roadmap_lines() -> list[str]:
    """Lines of the canonical roadmap. Raises rather than yielding an empty file.

    An empty list here is indistinguishable from a roadmap with no content, which
    is how a generated report ends up quoting nothing while claiming a source.
    """
    return roadmap_path().read_text(encoding="utf-8", errors="replace").splitlines()


def quote_span(start: int, end: int) -> str:
    """Quote a 1-indexed inclusive line span, or raise. Never a placeholder.

    Callers used to return ``"<H-ROADMAP.md not readable>"`` here, which is a
    string an acceptance receipt will happily store as the criterion it swears it
    did not alter.
    """
    lines = roadmap_lines()
    return "\n".join(lines[start - 1:end])
