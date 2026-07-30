"""Brace-matching scanner that tracks string/char/raw-string literals and comments.

Required for Rust and TypeScript extraction on this codebase (files up to ~13k lines).
A naive regex will mis-span on nested braces inside strings and comments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Span:
    start: int  # byte/char offset
    end: int    # exclusive
    start_line: int
    end_line: int


def line_starts(text: str) -> list[int]:
    """Offsets of the first character of each 1-indexed line; index 0 unused."""
    starts = [0, 0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def offset_to_line(starts: list[int], offset: int) -> int:
    # binary search
    lo, hi = 1, len(starts) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if starts[mid] <= offset:
            if mid == len(starts) - 1 or starts[mid + 1] > offset:
                return mid
            lo = mid + 1
        else:
            hi = mid - 1
    return max(1, hi)


class CodeScanner:
    """Scan source while skipping comments and string literals."""

    def __init__(self, text: str, *, lang: str = "rust"):
        self.text = text
        self.lang = lang
        self.n = len(text)
        self.starts = line_starts(text)
        # Mask: True = code (not comment/string)
        self.code_mask = self._build_mask()

    def _build_mask(self) -> bytearray:
        t = self.text
        n = self.n
        mask = bytearray(b"\x01") * n  # 1 = code
        i = 0
        while i < n:
            c = t[i]
            # line comment
            if c == "/" and i + 1 < n and t[i + 1] == "/":
                j = i
                while j < n and t[j] != "\n":
                    mask[j] = 0
                    j += 1
                i = j
                continue
            # block comment
            if c == "/" and i + 1 < n and t[i + 1] == "*":
                j = i
                mask[j] = 0
                if j + 1 < n:
                    mask[j + 1] = 0
                j += 2
                while j < n - 1:
                    mask[j] = 0
                    if t[j] == "*" and t[j + 1] == "/":
                        mask[j + 1] = 0
                        j += 2
                        break
                    j += 1
                else:
                    if j < n:
                        mask[j] = 0
                    j = n
                i = j
                continue
            # Python / shell hash comments (not rust/ts/metal where # is token)
            if self.lang in ("python", "shell") and c == "#":
                j = i
                while j < n and t[j] != "\n":
                    mask[j] = 0
                    j += 1
                i = j
                continue
            # raw string rust: r#"..."# / r##"..."## / br#"..."#
            if self.lang == "rust" and c in "br":
                k = i
                if t[k] == "b":
                    k += 1
                if k < n and t[k] == "r":
                    k += 1
                    hashes = 0
                    while k < n and t[k] == "#":
                        hashes += 1
                        k += 1
                    if k < n and t[k] == '"':
                        # mark from i through closing
                        j = i
                        while j <= k:
                            mask[j] = 0
                            j += 1
                        # j is past opening quote start... actually k is at "
                        j = k + 1
                        close = '"' + ("#" * hashes)
                        while j < n:
                            mask[j] = 0
                            if t.startswith(close, j):
                                for _ in range(len(close)):
                                    if j < n:
                                        mask[j] = 0
                                        j += 1
                                break
                            j += 1
                        i = j
                        continue
            # ordinary string
            if c == '"':
                mask[i] = 0
                j = i + 1
                while j < n:
                    mask[j] = 0
                    if t[j] == "\\":
                        if j + 1 < n:
                            mask[j + 1] = 0
                        j += 2
                        continue
                    if t[j] == '"':
                        j += 1
                        break
                    # template-ish: no
                    j += 1
                i = j
                continue
            # char literal rust/ts: 'x' or '\n' — careful not to eat lifetimes ('a)
            if c == "'" and self.lang in ("rust", "typescript", "metal"):
                # lifetime or byte-char: if next is letter/_ and then not '
                if i + 1 < n and (t[i + 1].isalnum() or t[i + 1] == "_"):
                    # could be lifetime 'a or 'static or char 'x'
                    if i + 2 < n and t[i + 2] == "'":
                        mask[i] = mask[i + 1] = mask[i + 2] = 0
                        i += 3
                        continue
                    # lifetime — leave as code (ident-like)
                    i += 1
                    continue
                if i + 1 < n and t[i + 1] == "\\":
                    # escaped char
                    j = i
                    while j < n and j < i + 8:
                        mask[j] = 0
                        if j > i and t[j] == "'":
                            j += 1
                            break
                        j += 1
                    i = j
                    continue
                if i + 2 < n and t[i + 2] == "'":
                    mask[i] = mask[i + 1] = mask[i + 2] = 0
                    i += 3
                    continue
            i += 1
        return mask

    def is_code(self, i: int) -> bool:
        return 0 <= i < self.n and self.code_mask[i] == 1

    def line_of(self, offset: int) -> int:
        return offset_to_line(self.starts, offset)

    def match_braces(self, open_off: int) -> int | None:
        """Given offset of '{', return exclusive end offset after matching '}'."""
        t = self.text
        n = self.n
        if open_off >= n or t[open_off] != "{":
            return None
        depth = 0
        i = open_off
        while i < n:
            if not self.code_mask[i]:
                i += 1
                continue
            c = t[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return None

    def find_next_brace(self, start: int, limit: int | None = None) -> int | None:
        end = self.n if limit is None else min(limit, self.n)
        i = start
        while i < end:
            if self.code_mask[i] and self.text[i] == "{":
                return i
            i += 1
        return None

    def slice_code(self, start: int, end: int) -> str:
        """Return text[start:end] with non-code regions replaced by spaces (preserve length)."""
        t = self.text
        parts: list[str] = []
        i = start
        while i < end:
            if self.code_mask[i]:
                parts.append(t[i])
            else:
                parts.append("\n" if t[i] == "\n" else " ")
            i += 1
        return "".join(parts)

    def iter_code_regions(self) -> list[tuple[int, int]]:
        """Contiguous code regions as (start, end) exclusive."""
        regions: list[tuple[int, int]] = []
        i = 0
        n = self.n
        while i < n:
            if not self.code_mask[i]:
                i += 1
                continue
            j = i
            while j < n and self.code_mask[j]:
                j += 1
            regions.append((i, j))
            i = j
        return regions
