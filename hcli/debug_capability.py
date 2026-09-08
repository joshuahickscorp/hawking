"""DEBUG as a capability: reproduce, localize, hypothesize, verify.

S036 s21: when something fails the loop is REPRODUCE -> LOCALIZE -> HYPOTHESIZE
-> DISCRIMINATE -> MEASURE -> PATCH -> VERIFY, never "see error, guess fix".
This owns the first three rungs mechanically -- the parts a harness can do
better than a model guessing -- and hands the model a localized, reproduced
failure instead of a raw traceback to react to.

WHY MECHANICAL. A model shown a 200-line pytest dump reacts to whatever is most
salient, which is rarely the cause. A model shown "this assertion, at this
file:line, reproduced on this exact command, and here is the function it is in"
spends inference on the fix, not on parsing. The reproduction command is the
authority the campaign keeps asking for: a failure you cannot reproduce is a
story, and a failure you can is a fact.

IT ADDS NO RUNTIME. Reproduction runs through the engine's own admitted-test
path -- the same one red-before-green uses -- so there is one test authority,
not two. Localization reads source with the filesystem tools that already
exist. The value is the SHAPE: a structured, reproduced, localized failure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

#: The last frame of a Python traceback that lives in the project, not in the
#: stdlib or site-packages -- the place a fix almost always goes.
_FRAME = re.compile(r'^\s*(?:File "|)(?P<path>[^"\n]+?\.py)"?[,:]\s*'
                    r'(?:line\s+)?(?P<line>\d+)', re.M)
_ASSERT = re.compile(r'^E\s+(?P<kind>\w*(?:Error|Exception|assert\w*))\b.*$', re.M | re.I)


@dataclass
class Failure:
    """A reproduced, localized failure, ready to reason about."""

    reproduced: bool
    command: str
    exit_code: Optional[int]
    kind: str = ""
    message: str = ""
    file: str = ""
    line: int = 0
    function: str = ""
    source_window: str = ""
    handle: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def summary(self) -> str:
        """What the model should see: actionable, not the whole dump."""
        if not self.reproduced:
            return (f"could not reproduce with `{self.command}` "
                    f"(exit {self.exit_code}): {self.note or 'no failure captured'}")
        where = f"{self.file}:{self.line}" if self.file else "location unknown"
        head = f"REPRODUCED `{self.command}` -> {self.kind or 'failure'} at {where}"
        if self.function:
            head += f" in {self.function}()"
        parts = [head]
        if self.message:
            parts.append(f"  {self.message}")
        if self.source_window:
            parts.append(self.source_window)
        if self.handle:
            parts.append(f"  full output: {self.handle}")
        return "\n".join(parts)


def _project_frame(text: str, root: Path) -> Optional[Dict[str, Any]]:
    """The deepest traceback frame inside the project. Stdlib frames are noise."""
    best = None
    for match in _FRAME.finditer(text or ""):
        raw = match.group("path")
        try:
            path = Path(raw)
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
        except Exception:
            continue
        parts = str(path)
        if any(seg in parts for seg in ("/site-packages/", "/lib/python",
                                        "/_pytest/", "/pluggy/")):
            continue
        try:
            path.relative_to(root.resolve())
        except ValueError:
            continue
        best = {"file": str(path), "line": int(match.group("line"))}
    return best


def _enclosing_function(path: Path, line: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for index in range(min(line, len(lines)) - 1, -1, -1):
        stripped = lines[index].lstrip()
        if stripped.startswith(("def ", "async def ")):
            name = stripped.split("(")[0].replace("async def ", "").replace("def ", "")
            return name.strip()
    return ""


def _source_window(path: Path, line: int, radius: int = 4) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    lo = max(0, line - radius - 1)
    hi = min(len(lines), line + radius)
    out = []
    for index in range(lo, hi):
        mark = ">>" if index == line - 1 else "  "
        out.append(f"  {mark} {index + 1:4} {lines[index]}")
    return "\n".join(out)


def diagnose(output: str, *, command: str, exit_code: Optional[int],
             root: Path, cache: Any = None) -> Failure:
    """Turn a raw failed-run output into a localized Failure.

    Pure over its inputs, so it is testable without running anything: hand it
    the captured output and it localizes. The caller owns reproduction (through
    the engine), which is what keeps ONE test authority.
    """
    reproduced = exit_code not in (0, None)
    kind = ""
    message = ""
    assertion = _ASSERT.search(output or "")
    if assertion:
        kind = assertion.group("kind")
        message = assertion.group(0).lstrip("E").strip()
    frame = _project_frame(output or "", root)
    failure = Failure(reproduced=reproduced, command=command, exit_code=exit_code,
                      kind=kind, message=message)
    if frame:
        path = Path(frame["file"])
        failure.file = str(path)
        failure.line = frame["line"]
        failure.function = _enclosing_function(path, frame["line"])
        failure.source_window = _source_window(path, frame["line"])
    if cache is not None and output and len(output) > 1500:
        try:
            ref = cache.store(output)
            failure.handle = ref.id
        except Exception:
            pass
    if not reproduced:
        failure.note = "the command exited cleanly; nothing to debug"
    elif not frame:
        failure.note = ("reproduced, but no project frame in the traceback -- "
                        "the failure is in a dependency or the harness")
    return failure
