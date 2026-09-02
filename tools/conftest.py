"""Sparse-checkout collection guard for ``pytest tools/``.

``-k 'tabula or vmcp'`` still *collects* every test module under tools/.
Several trees import ``lab.operators``, which is in git HEAD but not in this
sparse cone. Ignore those modules only while that path is absent so the
acceptance filter can finish; they collect again when the tree is widened.
"""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_OPERATORS = _REPO / "lab" / "operators"
_HEADLESS_NAME_COLLISION = {
    "test_organ_bandwidth.py",
    "test_organ_roof_ledger.py",
    "test_whole_model_native.py",
}


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001
    try:
        rel = str(Path(collection_path).resolve().relative_to(_REPO)).replace("\\", "/")
    except ValueError:
        return False
    if not _OPERATORS.is_dir():
        if rel.startswith("tools/condense/tests/") or rel.startswith("tools/foundry/tests/"):
            return True
    # pytest --import-mode=prepend puts tools/future on sys.path when the
    # session also collects that tree. tools/headless/test_organ_*.py then
    # `import organ_bandwidth` and hit tools/future/organ_bandwidth.py.
    # Skip only for a wide `pytest tools/` session; `pytest tools/headless`
    # still collects them.
    name = getattr(collection_path, "name", None) or Path(str(collection_path)).name
    if name in _HEADLESS_NAME_COLLISION and Path(str(collection_path)).parent.name == "headless":
        if _session_collects_future(config):
            return True
    return None


def _session_collects_future(config) -> bool:
    for raw in getattr(config, "args", ()) or ():
        p = Path(str(raw))
        if not p.is_absolute():
            p = (_REPO / p).resolve()
        else:
            p = p.resolve()
        try:
            rel = p.relative_to(_REPO).as_posix()
        except ValueError:
            rel = p.as_posix()
        if rel in {"", ".", "tools", "tools/future"}:
            return True
        if rel.startswith("tools/future/") or rel.startswith("tools/future"):
            return True
    return False
