"""Visible process identity for daemon-owned Python children.

The OS process tree is the ownership authority: production providers and UI
clients remain separate subprocesses for fault isolation, but all descend from
one ``hawkingd``. macOS Activity Monitor normally labels those children only
as ``python3.x``, which hides that relationship in its flat memory view. A
Python virtual environment derives its package prefix from the interpreter's
directory. A same-directory native hard link therefore keeps the process in
that environment while giving macOS a Hawking executable identity. A symlink
is insufficient: ps can show its path, but Activity Monitor still obtains
python3.x from the underlying Mach-O image.
"""
from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import List


def branded_python_entrypoint(entrypoint: str, role: str) -> List[str]:
    """Return argv prefix that keeps a Python tool visibly under Hawking.

    Non-Python entrypoints and unwritable environments fall back unchanged.
    Existing aliases are used only when they are the exact interpreter named
    by the entrypoint's shebang; an unrelated file is never overwritten.
    """
    script = Path(str(entrypoint)).expanduser()
    try:
        with script.open("r", encoding="utf-8", errors="replace") as handle:
            first = handle.readline()
        words = shlex.split(first[2:].strip()) if first.startswith("#!") else []
        interpreter = Path(words[0]).expanduser() if words else None
    except (OSError, ValueError):
        interpreter = None
    if (interpreter is None or not interpreter.is_file()
            or not interpreter.name.lower().startswith("python")):
        return [str(entrypoint)]

    resolved = interpreter.resolve()
    # Every production process has the same visible umbrella identity. The
    # role remains explicit in the process tree/argv and in HCLI's native
    # classifier, but Activity Monitor should present one Hawking family
    # rather than generic Python plus several branded variants.
    alias = interpreter.parent / "hawkingd"
    try:
        if alias.exists() or alias.is_symlink():
            if alias.is_symlink() and alias.resolve() == resolved:
                # This was the first implementation. Replace only our exact
                # safe alias; never unlink an unrelated executable.
                alias.unlink()
            elif not os.path.samefile(alias, resolved):
                return [str(entrypoint)]

        if not alias.exists():
            # uv's standalone Python binary expects libpython one directory
            # above the executable. A venv normally contains no dylib because
            # its python is a symlink to the standalone installation. The
            # native alias must live in the venv to preserve pyvenv.cfg/site
            # packages, so provide the same library there without copying it.
            match = re.fullmatch(r"python(?P<version>\d+\.\d+)", resolved.name)
            if match:
                library = (
                    resolved.parent.parent
                    / "lib"
                    / f"libpython{match.group('version')}.dylib"
                )
                local_library = interpreter.parent.parent / "lib" / library.name
                if library.is_file() and not local_library.exists():
                    local_library.symlink_to(library)
            # A hard link preserves the Mach-O bytes but changes the executable
            # basename macOS reports. A racing launcher validates the winner.
            os.link(resolved, alias)
    except FileExistsError:
        try:
            if not os.path.samefile(alias, resolved):
                return [str(entrypoint)]
        except OSError:
            return [str(entrypoint)]
    except OSError:
        # Filesystems that cannot hard-link still receive the useful command
        # path identity, although Activity Monitor may retain python3.x.
        try:
            if not alias.exists() and not alias.is_symlink():
                alias.symlink_to(interpreter.name)
        except OSError:
            pass
        if not alias.exists() and not alias.is_symlink():
            return [str(entrypoint)]
    try:
        if not os.path.samefile(alias, resolved):
            return [str(entrypoint)]
    except OSError:
        return [str(entrypoint)]
    return [str(alias), str(script)]


__all__ = ["branded_python_entrypoint"]
