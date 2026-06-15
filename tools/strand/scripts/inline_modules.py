#!/usr/bin/env python3
"""Inline specific named sub-module files into a Rust source file.

For each named module, finds the `pub mod name;` or `mod name;` declaration in
the target file, reads `name.rs` from the same directory, and replaces the
declaration with an inline `mod name { ... }` block. The `.rs` file is removed.

Modules not listed on the command line are left as external declarations.

Usage:
    python3 scripts/inline_modules.py <target.rs> <mod1> [<mod2> ...]

Examples:
    python3 scripts/inline_modules.py crates/strand-core/src/lib.rs types cdf
    python3 scripts/inline_modules.py crates/strand-container/src/transform/mod.rs med paeth ma_pred frame_delta
    python3 scripts/inline_modules.py crates/strand-container/src/models/mod.rs raw_store
    python3 scripts/inline_modules.py crates/strand-cli/src/main.rs info
"""

import re
import sys
from pathlib import Path


def strip_header(lines):
    """Strip leading #![...] inner-attribute and //! module doc lines."""
    trimmed = []
    in_header = True
    for sl in lines:
        if in_header and (sl.strip().startswith("#![") or sl.strip().startswith("//!")):
            continue
        in_header = False
        trimmed.append(sl)
    return trimmed


def inline_modules(target: Path, modules: list) -> None:
    modules_set = set(modules)
    content = target.read_text()
    lines = content.splitlines(keepends=True)

    result_lines = []
    for line in lines:
        m = re.match(r'^(\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;)', line)
        if m and m.group(2) in modules_set:
            submod_name = m.group(2)
            sub_file = target.parent / f"{submod_name}.rs"
            if sub_file.exists():
                vis = "pub " if "pub" in m.group(1) else ""
                sub_lines = sub_file.read_text().splitlines(keepends=True)
                trimmed = strip_header(sub_lines)
                trimmed_content = "".join(trimmed)
                indented = "\n".join(
                    "    " + l if l.strip() else ""
                    for l in trimmed_content.splitlines()
                )
                result_lines.append(f"{vis}mod {submod_name} {{\n{indented}\n}}\n")
                sub_file.unlink()
                print(f"  inlined + removed: {sub_file.name}  ({len(sub_lines)} lines)")
            else:
                print(f"  WARN: {sub_file} not found — kept declaration unchanged")
                result_lines.append(line)
        else:
            result_lines.append(line)

    target.write_text("".join(result_lines))
    print(f"  updated: {target}  ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: inline_modules.py <target.rs> <mod1> [<mod2> ...]", file=sys.stderr)
        sys.exit(1)
    target = Path(sys.argv[1])
    mods = sys.argv[2:]
    print(f"\nInlining {mods} into {target} ...")
    inline_modules(target, mods)
    print("Done.")
