#!/usr/bin/env python3
"""Convert a Rust `src/name/mod.rs` directory into a single `src/name.rs` file.

For each `pub mod submod;` (or `mod submod;`) declaration in mod.rs, the
corresponding `submod.rs` file is read and the declaration is replaced with an
inline `mod submod { ... }` block. The resulting file is written to the parent
directory as `name.rs` and the `name/` directory is removed.

Usage:
    python3 scripts/collapse_module_dir.py  crates/strand-container/src/engine
    python3 scripts/collapse_module_dir.py  crates/strand-container/src/coder
"""

import re
import sys
import shutil
from pathlib import Path


def collapse(module_dir: Path) -> None:
    mod_file = module_dir / "mod.rs"
    if not mod_file.exists():
        print(f"ERROR: {mod_file} not found", file=sys.stderr)
        sys.exit(1)

    content = mod_file.read_text()
    lines = content.splitlines(keepends=True)

    result_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Match bare or pub-vis mod declarations (no inline content)
        m = re.match(r'^(\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;)', line)
        if m:
            submod_name = m.group(2)
            sub_file = module_dir / f"{submod_name}.rs"
            if sub_file.exists():
                vis = "pub " if "pub" in m.group(1) else ""
                sub_content = sub_file.read_text()
                # Strip file-level inner attributes (#![...]) – they'd be invalid
                # inside a mod block when this is a non-root module file.
                # Also strip the crate-level module doc (//! lines at the top) since
                # they become confusingly nested.
                sub_lines = sub_content.splitlines(keepends=True)
                # Drop leading inner-attribute lines
                trimmed = []
                in_header = True
                for sl in sub_lines:
                    if in_header and (sl.strip().startswith("#![") or sl.strip().startswith("//!")):
                        continue
                    in_header = False
                    trimmed.append(sl)
                trimmed_content = "".join(trimmed)
                # Indent by 4 spaces
                indented = "\n".join("    " + l if l.strip() else "" for l in trimmed_content.splitlines())
                result_lines.append(f"{vis}mod {submod_name} {{\n{indented}\n}}\n")
                print(f"  inlined: {sub_file.name} ({len(sub_lines)} lines)")
            else:
                # No file found — keep the declaration unchanged (external dep or cfg-gated)
                result_lines.append(line)
        else:
            result_lines.append(line)
        i += 1

    out_file = module_dir.parent / f"{module_dir.name}.rs"
    out_file.write_text("".join(result_lines))
    print(f"  wrote:   {out_file}  ({out_file.stat().st_size} bytes)")

    shutil.rmtree(module_dir)
    print(f"  removed: {module_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: collapse_module_dir.py <path/to/module/dir>", file=sys.stderr)
        sys.exit(1)
    for arg in sys.argv[1:]:
        p = Path(arg)
        print(f"\nCollapsing {p} ...")
        collapse(p)
    print("\nDone.")
