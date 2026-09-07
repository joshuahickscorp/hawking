"""Resolve the canonical H-ROADMAP, which lives outside the repository.

The roadmap is external by deliberate user placement at ``~/Downloads/H-ROADMAP.md``.
External means it can vanish without a commit, and it did: twelve modules hardcode
that path, ``tools/roadmap/parse.py`` raises on it, and ``tools/roadmap/recompile.py``
silently substitutes an empty file. Worse, four acceptance harnesses degrade a
missing roadmap into a *placeholder string*, so a receipt could swear
``criterion_altered: false`` while quoting ``<H-ROADMAP.md not readable at ...>``.

The superseded 2026-09-02 copy is retained by Git history and identified by the
digest in ``docs/roadmap-lineage/PRESERVATION.md``. It is not an active fallback:
the canonical roadmap is deliberately operator-owned and must be present (or
explicitly supplied) when line-numbered obligations are compiled.

Resolution order:

    $H_ROADMAP                          explicit override, taken as given
    ~/Downloads/H-ROADMAP.md            canonical external authority, taken as given
    docs/roadmap-lineage/PRESERVATION.md   historical digest and provenance only

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

def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def roadmap_path() -> Path:
    """The canonical roadmap, or raise naming every location that was tried.

    Never returns a path that does not exist or silently substitutes history.
    """
    override = os.environ.get("H_ROADMAP")
    if override:
        return Path(override)
    if EXTERNAL.is_file():
        return EXTERNAL
    raise FileNotFoundError(
        f"canonical roadmap not readable: tried $H_ROADMAP and {EXTERNAL}"
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
