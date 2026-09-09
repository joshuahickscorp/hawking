"""A digest arrives from tool arguments, so it is attacker-influenced.

ProjectStore.artifact_path only checked len(digest) >= 3, so
'../../etc/passwd' returned '/../etc/passwd' -- a path outside the project
entirely. Content-addressed storage that will hand back any path you name is
not content-addressed.

This file was first drafted by the HCLI resident during an autonomy run. Its
shape was right -- correct target, correct exception -- but it called
artifact_path as if it were a static method, so it went red on TypeError rather
than on the traversal. Corrected here and kept, because the instinct was sound.
"""
import tempfile
from pathlib import Path

import pytest

from hcli.perception.store import ProjectStore


@pytest.fixture
def store():
    return ProjectStore(Path(tempfile.mkdtemp()) / "proj")


@pytest.mark.parametrize("bad", [
    "../../etc/passwd", "../escape", "a/b", "..", "/absolute/path",
    "AB" + "c" * 62, "short", "", "x" * 63, "x" * 65,
])
def test_a_non_digest_is_refused(store, bad):
    with pytest.raises(ValueError):
        store.artifact_path(bad)


def test_a_real_digest_still_resolves_inside_the_tree(store):
    digest = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    got = store.artifact_path(digest)
    artifacts = (store.root / "artifacts").resolve()
    assert artifacts in got.parents, got
    assert got.name == digest[2:]


def test_the_guard_is_not_vacuous(store):
    """If artifact_path stopped raising, the parametrised test must notice."""
    import inspect
    src = inspect.getsource(ProjectStore.artifact_path)
    assert "raise ValueError" in src and "escapes the project" in src
