"""Gravity: experiment search, compilation, candidates.

Gravity lives as ``tools/gravity_*.py`` plus the hawking crates. It is
not relocated into this tree: those scripts are the product, and moving
them would invent a second import identity for the same files.

Active storage law: Gravity's transient representation/shards are ``.nr``
(Noetic Representation).  The machine-bound final executable is ``.nx``
(Noetic Executable).  Historical ``.gravity`` artifacts remain readable.
"""

OWNED_PREFIXES = ("tools/gravity_", "tools/nos_pipeline.py")

# P6_GRAVITY_EXECUTION_BOUNDARY_V3: the execution boundary is the set of
# artifact suffixes Gravity may hand to the machine-bound executor.  Only the
# machine-bound final executable (``.nx``) is executable; transient
# representation/shards (``.nr``) and historical ``.gravity`` artifacts are
# readable inputs, never execution targets.
EXECUTABLE_SUFFIXES = (".nx",)
READABLE_SUFFIXES = (".nr", ".gravity")


def is_executable_artifact(path):
    """Return True iff ``path`` names a machine-bound executable artifact.

    The boundary is suffix-based and case-insensitive so that a transient
    ``.nr`` shard or a historical ``.gravity`` artifact can never be promoted
    to an execution target by casing or by a longer compound suffix.
    """
    if not isinstance(path, str):
        return False
    lowered = path.strip().lower()
    if not lowered:
        return False
    return lowered.endswith(EXECUTABLE_SUFFIXES)


def is_readable_artifact(path):
    """Return True iff ``path`` names a readable (non-executable) artifact."""
    if not isinstance(path, str):
        return False
    lowered = path.strip().lower()
    if not lowered:
        return False
    return lowered.endswith(READABLE_SUFFIXES)
