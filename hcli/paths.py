"""Locate the repository root without encoding directory depth.

``Path(__file__).parents[N]`` broke the moment the package moved. Walk
toward a ``.git`` entry or the repo-root ``Cargo.toml`` instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def find_repo_root(start: Optional[Union[str, Path]] = None) -> Path:
    """Return the hawking repo root containing this package, or *start*.

    Works from an editable install (package lives at ``<repo>/hcli``) and
    from a copied/stamped tree (falls back to the nearest ancestor that
    still looks like this repo). A missing ``.git`` is not fatal: git
    worktrees use a file, and a stamped ``~/.local/share/hcli/current``
    copy has no vcs metadata.
    """
    if start is None:
        here = Path(__file__).resolve()
    else:
        here = Path(start).resolve()
    if here.is_file():
        here = here.parent
    for cand in (here, *here.parents):
        if (cand / ".git").exists() or (cand / "Cargo.toml").is_file():
            if (cand / "tools" / "headless").is_dir() or (cand / "crates").is_dir():
                return cand
            if (cand / "Cargo.toml").is_file():
                return cand
    # Stamped install: hcli/paths.py -> package dir -> container.
    # Container is not the repo; callers that need tools/headless must
    # handle FileNotFoundError themselves.
    return Path(__file__).resolve().parent.parent
