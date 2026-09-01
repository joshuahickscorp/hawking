"""Gravity: experiment search, compilation, candidates.

Gravity lives as ``tools/gravity_*.py`` plus the hawking crates. It is
not relocated into this tree: those scripts are the product, and moving
them would invent a second import identity for the same files.

Active storage law: Gravity's transient representation/shards are ``.nr``
(Noetic Representation).  The machine-bound final executable is ``.nx``
(Noetic Executable).  Historical ``.gravity`` artifacts remain readable.
"""

OWNED_PREFIXES = ("tools/gravity_", "tools/nos_pipeline.py")
