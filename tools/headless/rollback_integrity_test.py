#!/usr/bin/env python3
"""Rollback byte-integrity (directive §30: "deterministic rollback").

`_snapshot` / `_restore` exist in engine.py and `rolled_back` reaches the receipt,
so rollback LOOKS covered. This asks the only question that matters: after a
rejected mutation, are the bytes on disk **byte-identical** to what they were?

Not "the test passes again". Not "the file looks right". Identical sha256, and
identical file mode, for every file the mutation touched — plus no file left
behind that the mutation created, and no file missing that it deleted.

That distinction is the whole point. A restore that rewrites a file with
equivalent-but-different bytes (different line endings, a dropped trailing
newline, a normalised encoding) passes every behavioural check and still corrupts
a repository over a long autonomous mission, one silent byte at a time.

No GPU, no model, no network.

A note on how NOT to check this. The obvious way to prove these assertions can
fail is to break `_restore` in place and watch them go red. Do not: while a
delegated lane holds `engine.py`, editing it creates exactly the two-writer
hazard this campaign exists to prevent. It was done once here and the live file
was restored from a backup seven seconds later — nothing was lost, but that was
luck, not method. `mutate_a_copy()` below does the same proof against a private
copy of the module, which is safe at any time.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path


FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}: {detail}")
        FAILS.append(f"{name}: {detail}")


def tree_state(root: Path) -> dict:
    """sha256 + mode for every file, so a restore is compared on bytes and not on
    appearance."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".hcli" not in p.parts and ".git" not in p.parts:
            out[str(p.relative_to(root))] = {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "mode": stat.S_IMODE(p.stat().st_mode),
                "bytes": p.stat().st_size,
            }
    return out


def diff_state(a: dict, b: dict) -> dict:
    return {
        "changed": sorted(k for k in set(a) & set(b) if a[k]["sha256"] != b[k]["sha256"]),
        "mode_changed": sorted(k for k in set(a) & set(b) if a[k]["mode"] != b[k]["mode"]),
        "vanished": sorted(set(a) - set(b)),
        "appeared": sorted(set(b) - set(a)),
    }


def make_workspace() -> Path:
    # macOS mktemp hands back /var/... which is a symlink to /private/var/...;
    # the engine resolves through _safe_path, so an unresolved root makes every
    # operation look out-of-workspace and the test would pass for a wrong reason.
    ws = Path(tempfile.mkdtemp()).resolve()
    (ws / "pkg").mkdir()
    # deliberately awkward content: no trailing newline, CRLF, a tab, non-ascii,
    # and an executable bit — all things a careless restore normalises away
    (ws / "pkg" / "calc.py").write_bytes(b"def add(a, b):\r\n\treturn a - b")
    (ws / "pkg" / "notes.txt").write_bytes("café — no trailing newline".encode())
    script = ws / "run.sh"
    script.write_bytes(b"#!/bin/sh\necho hi\n")
    script.chmod(0o755)
    (ws / "keep.md").write_bytes(b"untouched\n")
    return ws


def engine_for(ws: Path):
    from hcli.engine import Engine
    e = Engine.__new__(Engine)
    e.root = ws
    return e


def test_restore_is_byte_identical_after_a_rejected_replace():
    ws = make_workspace()
    try:
        before = tree_state(ws)
        e = engine_for(ws)
        snap = e._snapshot([ws / "pkg" / "calc.py", ws / "pkg" / "notes.txt"])
        e._apply_operations([{"op": "replace", "path": "pkg/calc.py",
                              "old_text": "return a - b", "new_text": "return a + b"}])
        mid = tree_state(ws)
        check("the mutation actually changed the file (guard against a vacuous test)",
              mid["pkg/calc.py"]["sha256"] != before["pkg/calc.py"]["sha256"])
        e._restore(snap)
        after = tree_state(ws)
        d = diff_state(before, after)
        check("restore is byte-identical", not d["changed"], str(d["changed"]))
        check("restore preserves file modes", not d["mode_changed"], str(d["mode_changed"]))
        check("restore leaves nothing behind", not d["appeared"], str(d["appeared"]))
        check("restore loses nothing", not d["vanished"], str(d["vanished"]))
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_restore_preserves_crlf_tabs_and_unicode():
    """The specific bytes a normalising restore would quietly rewrite."""
    ws = make_workspace()
    try:
        raw_before = (ws / "pkg" / "calc.py").read_bytes()
        notes_before = (ws / "pkg" / "notes.txt").read_bytes()
        e = engine_for(ws)
        snap = e._snapshot([ws / "pkg" / "calc.py", ws / "pkg" / "notes.txt"])
        e._apply_operations([{"op": "append", "path": "pkg/notes.txt", "new_text": "\nextra"}])
        e._restore(snap)
        check("CRLF and tab survive a restore",
              (ws / "pkg" / "calc.py").read_bytes() == raw_before,
              repr((ws / "pkg" / "calc.py").read_bytes()[:40]))
        check("unicode and a missing trailing newline survive a restore",
              (ws / "pkg" / "notes.txt").read_bytes() == notes_before,
              repr((ws / "pkg" / "notes.txt").read_bytes()[:60]))
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_restore_removes_a_file_the_mutation_created():
    ws = make_workspace()
    try:
        before = tree_state(ws)
        e = engine_for(ws)
        target = ws / "pkg" / "brand_new.py"
        snap = e._snapshot([target])
        e._apply_operations([{"op": "create", "path": "pkg/brand_new.py",
                              "new_text": "x = 1\n"}])
        check("create actually created the file", target.is_file())
        e._restore(snap)
        after = tree_state(ws)
        d = diff_state(before, after)
        check("a rolled-back create leaves no orphan file",
              "pkg/brand_new.py" not in after and not d["appeared"],
              f"appeared={d['appeared']} exists={target.exists()}")
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_executable_bit_survives_a_rolled_back_replace_file():
    ws = make_workspace()
    try:
        mode_before = stat.S_IMODE((ws / "run.sh").stat().st_mode)
        body_before = (ws / "run.sh").read_bytes()
        e = engine_for(ws)
        snap = e._snapshot([ws / "run.sh"])
        e._apply_operations([{"op": "replace_file", "path": "run.sh",
                              "new_text": "#!/bin/sh\necho changed\n"}])
        e._restore(snap)
        check("executable bit survives rollback",
              stat.S_IMODE((ws / "run.sh").stat().st_mode) == mode_before,
              f"{oct(stat.S_IMODE((ws/'run.sh').stat().st_mode))} vs {oct(mode_before)}")
        check("script body restored byte-identically",
              (ws / "run.sh").read_bytes() == body_before)
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_untouched_files_are_never_rewritten():
    """A restore must not touch what the mutation never touched — otherwise every
    rollback silently widens its own blast radius."""
    ws = make_workspace()
    try:
        keep = ws / "keep.md"
        mtime_before = keep.stat().st_mtime_ns
        sha_before = hashlib.sha256(keep.read_bytes()).hexdigest()
        e = engine_for(ws)
        snap = e._snapshot([ws / "pkg" / "calc.py"])
        e._apply_operations([{"op": "replace", "path": "pkg/calc.py",
                              "old_text": "return a - b", "new_text": "return a + b"}])
        e._restore(snap)
        check("an untouched file keeps its bytes",
              hashlib.sha256(keep.read_bytes()).hexdigest() == sha_before)
        check("an untouched file is not even rewritten (mtime unchanged)",
              keep.stat().st_mtime_ns == mtime_before,
              f"{keep.stat().st_mtime_ns} vs {mtime_before}")
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def mutate_a_copy() -> int:
    """Anti-vacuity, safely: copy the WHOLE hcli package to a temp dir, break
    `_restore` in the copy, and assert the byte-identity checks go red against it.

    Copying the whole package matters — engine.py uses relative imports, so a
    single-file copy will not import and the probe would silently SKIP, which is
    a probe that proves nothing dressed as one that passed.

    The injected bug is the classic one: write_text with newline translation
    instead of write_bytes. It passes every behavioural check and quietly eats
    CRLF, tabs and encodings over a long mission.
    """
    import importlib, shutil as _sh
    pkg_src = Path(__file__).resolve().parents[2] / "hcli"
    marker = "            path.write_bytes(\n                original\n            )"
    engine_src = (pkg_src / "engine.py").read_text()
    if marker not in engine_src:
        print("SKIP mutate_a_copy: _restore no longer matches the expected write; "
              "this probe is STALE and is NOT evidence")
        return 0
    broken_engine = engine_src.replace(
        marker,
        '            path.write_text(\n                original.decode("utf-8", "replace")'
        '.replace("\\r\\n", "\\n")\n            )', 1)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        _sh.copytree(pkg_src, root / "hcli",
                     ignore=_sh.ignore_patterns("__pycache__", "tests"))
        (root / "hcli" / "engine.py").write_text(broken_engine)
        sys.path.insert(0, str(root))
        for mod in [m for m in list(sys.modules) if m == "hcli" or m.startswith("hcli.")]:
            del sys.modules[mod]
        try:
            broken = importlib.import_module("hcli.engine")
        except Exception as e:
            print(f"SKIP mutate_a_copy: broken copy would not import "
                  f"({type(e).__name__}: {e}) — NOT evidence")
            sys.path.remove(str(root))
            return 0

        ws = make_workspace()
        try:
            raw_before = (ws / "pkg" / "calc.py").read_bytes()
            e = broken.Engine.__new__(broken.Engine)
            e.root = ws
            snap = e._snapshot([ws / "pkg" / "calc.py"])
            e._apply_operations([{"op": "replace", "path": "pkg/calc.py",
                                  "old_text": "return a - b", "new_text": "return a + b"}])
            e._restore(snap)
            got = (ws / "pkg" / "calc.py").read_bytes()
            check("a normalising restore IS caught by these assertions",
                  got != raw_before,
                  "the broken copy round-tripped byte-identically, so these checks "
                  "would not catch a real normalising restore")
            if got != raw_before:
                print(f"     broken copy produced {got[:34]!r} "
                      f"(original {raw_before[:34]!r})")
        finally:
            _sh.rmtree(ws, ignore_errors=True)
            sys.path.remove(str(root))
            for mod in [m for m in list(sys.modules) if m == "hcli" or m.startswith("hcli.")]:
                del sys.modules[mod]
    return 0


def main() -> int:
    try:
        import hcli.engine  # noqa: F401
    except Exception as e:
        print(f"FAIL: cannot import hcli.engine: {type(e).__name__}: {e}")
        return 1
    for fn in sorted([f for n, f in globals().items() if n.startswith("test_")],
                     key=lambda f: f.__code__.co_firstlineno):
        try:
            fn()
        except Exception as e:
            check(fn.__name__, False, f"raised {type(e).__name__}: {e}")
    mutate_a_copy()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        for f in FAILS:
            print("  " + f)
        return 1
    print("\nrollback is byte-identical on every checked axis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
