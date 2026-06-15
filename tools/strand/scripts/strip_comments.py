#!/usr/bin/env python3
"""Strip all comments from Rust (.rs) and Metal (.metal) source files.

Handles:
- Line comments: // ... (including ///, //!)
- Block comments: /* ... */ (nested, as Rust allows)
- Preserves string literals (", b", r#"..."#, etc.) and char literals/lifetimes
- Collapses runs of blank lines to a single blank line
"""

import sys
from pathlib import Path


def strip_rust_comments(src: str) -> str:
    out = []
    i = 0
    n = len(src)

    while i < n:
        # Block comment /* */ — Rust supports nesting
        if src[i:i+2] == '/*':
            depth = 1
            i += 2
            while i < n and depth > 0:
                if src[i:i+2] == '/*':
                    depth += 1
                    i += 2
                elif src[i:i+2] == '*/':
                    depth -= 1
                    i += 2
                else:
                    if src[i] == '\n':
                        out.append('\n')
                    i += 1
            continue

        # Line comment // (includes ///, //!)
        if src[i:i+2] == '//':
            while i < n and src[i] != '\n':
                i += 1
            continue

        c = src[i]

        # Byte raw string br#"..."# or br"..."
        if c == 'b' and i + 1 < n and src[i + 1] == 'r':
            j = i + 2
            hashes = 0
            while j < n and src[j] == '#':
                hashes += 1
                j += 1
            if j < n and src[j] == '"':
                closing = '"' + '#' * hashes
                end = src.find(closing, j + 1)
                if end != -1:
                    out.append(src[i:end + len(closing)])
                    i = end + len(closing)
                    continue

        # Raw string r#"..."# or r"..."
        if c == 'r':
            j = i + 1
            hashes = 0
            while j < n and src[j] == '#':
                hashes += 1
                j += 1
            if j < n and src[j] == '"':
                closing = '"' + '#' * hashes
                end = src.find(closing, j + 1)
                if end != -1:
                    out.append(src[i:end + len(closing)])
                    i = end + len(closing)
                    continue

        # Byte string b"..."
        if c == 'b' and i + 1 < n and src[i + 1] == '"':
            out.append('b"')
            i += 2
            while i < n:
                if src[i] == '\\' and i + 1 < n:
                    out.append(src[i:i + 2])
                    i += 2
                elif src[i] == '"':
                    out.append('"')
                    i += 1
                    break
                else:
                    out.append(src[i])
                    i += 1
            continue

        # String literal "..."
        if c == '"':
            out.append('"')
            i += 1
            while i < n:
                if src[i] == '\\' and i + 1 < n:
                    out.append(src[i:i + 2])
                    i += 2
                elif src[i] == '"':
                    out.append('"')
                    i += 1
                    break
                else:
                    out.append(src[i])
                    i += 1
            continue

        # Char literal vs lifetime: '
        if c == "'":
            if i + 1 < n:
                nc = src[i + 1]
                if nc == '\\':
                    # Escaped char like '\n', '\\', '\''
                    out.append("'")
                    i += 1
                    out.append('\\')
                    i += 1
                    while i < n and src[i] != "'":
                        out.append(src[i])
                        i += 1
                    if i < n:
                        out.append("'")
                        i += 1
                    continue
                elif i + 2 < n and src[i + 2] == "'":
                    # Simple char: 'x'
                    out.append(src[i:i + 3])
                    i += 3
                    continue
            # Lifetime or other — fall through

        out.append(c)
        i += 1

    return ''.join(out)


def collapse_blank_lines(text: str) -> str:
    lines = text.split('\n')
    result = []
    blank_run = 0
    for line in lines:
        if line.strip() == '':
            blank_run += 1
            if blank_run <= 1:
                result.append(line)
        else:
            blank_run = 0
            result.append(line)
    return '\n'.join(result)


def process_file(path: Path) -> bool:
    original = path.read_text(encoding='utf-8', errors='replace')
    stripped = strip_rust_comments(original)
    cleaned = collapse_blank_lines(stripped)
    if cleaned != original:
        path.write_text(cleaned, encoding='utf-8')
        return True
    return False


if __name__ == '__main__':
    roots = sys.argv[1:] if len(sys.argv) > 1 else ['.']
    changed = 0
    for root in roots:
        for path in Path(root).rglob('*.rs'):
            if 'target' in path.parts:
                continue
            if process_file(path):
                changed += 1
                print(f'  stripped: {path}')
        for path in Path(root).rglob('*.metal'):
            if 'target' in path.parts:
                continue
            if process_file(path):
                changed += 1
                print(f'  stripped: {path}')
    print(f'\n{changed} files updated.')
