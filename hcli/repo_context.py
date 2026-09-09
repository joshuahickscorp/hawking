"""What the model can know about the folder you opened it in.

MEASURED CONSTRAINT, NOT A STYLE CHOICE. Injecting context is not free here: a
cold prompt token costs about what a generated token costs (~30 ms on
sealed-3.14), so a careless 4000-token dump is two minutes before the first
word. But prefix reuse on the native resident is 30x for a prefix that does not
change between turns. Both facts together dictate the layout:

    [ STABLE ..................... ] [ VOLATILE ...... ] [ conversation ]
      repo identity, tree, README     files this question    the messages
      -- byte-identical every turn     happens to touch

The stable block is paid ONCE per session and reused; only the small volatile
block is re-prefilled per question. Putting the question-specific files first --
the obvious order -- would break the shared prefix on every turn and re-pay the
whole thing, which is the difference between a 0.5 s follow-up and a 20 s one.

CWD IS CONTEXT, NOT IDENTITY. Opening HCLI inside a repository must not turn
"explain DeltaNet" into a question about this repository. So the injected block
says what the repo IS and explicitly licenses ignoring it, rather than
instructing the model that every question is about these files.

RETRIEVAL, NOT A TOOL LOOP. This gives the model what is already on disk before
it answers. It cannot go and fetch more, and it does not pretend to: the block
names the files it included so a reader can tell the difference between "the
model reasoned about this file" and "the model was never shown it".
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: Roughly 4 chars per token. These are budgets in CHARACTERS.
STABLE_BUDGET = 6000
VOLATILE_BUDGET = 6000
FILE_HEAD = 2400

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "target",
    "build", "dist", ".mypy_cache", ".pytest_cache", ".worktrees",
    "receipts", "workspace", "evidence", ".hcli",
}
CODE_SUFFIXES = {".py", ".rs", ".ts", ".tsx", ".js", ".go", ".c", ".h", ".cpp",
                 ".metal", ".swift", ".sh", ".toml", ".md", ".json", ".yaml", ".yml"}
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-/]{2,}")


@dataclass
class RepoContext:
    root: Path
    name: str
    is_git: bool
    branch: str = ""
    head: str = ""

    @classmethod
    def detect(cls, start: Optional[str] = None) -> Optional["RepoContext"]:
        """The repository the user opened HCLI in, or None if this is not one."""
        here = Path(os.path.expanduser(start or os.getcwd())).resolve()
        if not here.is_dir():
            return None
        top = here
        for candidate in [here, *here.parents]:
            if (candidate / ".git").exists():
                top = candidate
                break
        else:
            # Not a git repo. A plain directory of source is still context.
            if not any(p.suffix in CODE_SUFFIXES for p in _shallow(here)):
                return None
            return cls(root=here, name=here.name, is_git=False)
        branch = _git(top, "rev-parse", "--abbrev-ref", "HEAD")
        head = _git(top, "rev-parse", "--short", "HEAD")
        return cls(root=top, name=top.name, is_git=True, branch=branch, head=head)

    # -- the two blocks -----------------------------------------------------
    def stable_block(self) -> str:
        """Byte-identical between turns, so the resident's prefix cache keeps it."""
        lines = [
            f"You are running inside the {self.name!r} repository at {self.root}.",
        ]
        if self.is_git:
            lines.append(f"git: branch {self.branch or '?'} at {self.head or '?'}")
        lines.append("")
        lines.append("Top level:")
        for entry in self._top_level():
            lines.append(f"  {entry}")
        readme = self._readme()
        if readme:
            lines.append("")
            lines.append("README (head):")
            lines.append(readme)
        lines.append("")
        lines.append(
            "Use this when the question is about this repository. When it is a "
            "general question, answer it generally -- being launched in a folder "
            "does not make every question about that folder.")
        text = "\n".join(lines)
        return text[:STABLE_BUDGET]

    def volatile_block(self, question: str) -> str:
        """The files this particular question appears to be about."""
        picked = self.files_for(question)
        if not picked:
            return ""
        lines = ["Files from this repository that may be relevant:"]
        budget = VOLATILE_BUDGET
        for path, body in picked:
            if budget <= 0:
                break
            chunk = body[:min(FILE_HEAD, budget)]
            lines.append(f"\n--- {path} ---\n{chunk}")
            budget -= len(chunk)
        lines.append(
            "\nThose are the only files you were shown. If the answer needs one "
            "that is not here, say which file you would need rather than guessing "
            "its contents.")
        return "\n".join(lines)

    # -- retrieval ----------------------------------------------------------
    def files_for(self, question: str, limit: int = 4) -> List[Tuple[str, str]]:
        """Files whose PATH matches words in the question.

        Deliberately path-matching rather than content search: it is fast, it is
        explainable, and a wrong pick is visibly a wrong filename rather than a
        mysterious excerpt. Content search over a large tree is the next step,
        not this one.
        """
        raw = {w.lower() for w in _WORD.findall(question or "")}
        # "hcli/serve.py" is one token AND its parts: the question names a path,
        # a module and a stem all at once, and each is a legitimate way to find
        # the file.
        words = set(raw)
        for token in raw:
            for part in re.split(r"[./]", token):
                if len(part) > 2:
                    words.add(part)
        words -= {"the", "this", "that", "what", "does", "file", "code", "repo",
                  "repository", "function", "where", "which", "how", "why",
                  "explain", "show", "and", "for", "with", "from"}
        if not words:
            return []
        # A path the question NAMED must outrank a file that merely shares a
        # word with it. Measured: asking about "hcli/serve.py and the stream
        # flag" returned visionmcp/ocular/stream.py, because a stem hit on
        # "stream" (10) beat a substring hit on the full path (3). Exactness is
        # the ranking, not word count.
        scored: List[Tuple[int, str]] = []
        for path in self._candidate_files():
            rel = str(path.relative_to(self.root))
            stem = path.stem.lower()
            lowered = rel.lower()
            score = 0
            for word in words:
                if word == lowered:
                    score += 25
                elif lowered.endswith("/" + word) or word == path.name.lower():
                    score += 20
                elif word == stem:
                    score += 10
                elif word in lowered:
                    score += 3
            if score:
                scored.append((score, rel))
        scored.sort(key=lambda row: (-row[0], len(row[1])))
        # PRECISION OVER RECALL when the question named something exactly.
        # Measured: asking about hcli/catalog.py returned it AND three other
        # files called catalog, and the model produced a blended description of
        # all four -- fluent, confident, and about no file that exists. When the
        # best hit is far ahead, the also-rans are noise, not context.
        if scored:
            best = scored[0][0]
            floor = best * 0.5 if best >= 20 else 0
            scored = [row for row in scored if row[0] >= floor]
        out: List[Tuple[str, str]] = []
        for _, rel in scored[:limit]:
            try:
                out.append((rel, (self.root / rel).read_text(
                    encoding="utf-8", errors="replace")))
            except OSError:
                continue
        return out

    def _candidate_files(self) -> List[Path]:
        found: List[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames
                           if d not in SKIP_DIRS and not d.startswith(".")]
            for name in filenames:
                if Path(name).suffix in CODE_SUFFIXES:
                    found.append(Path(dirpath) / name)
            if len(found) > 20000:
                return found
        return found

    def _top_level(self) -> List[str]:
        rows = []
        for entry in sorted(_shallow(self.root), key=lambda p: p.name):
            if entry.name.startswith(".") or entry.name in SKIP_DIRS:
                continue
            if entry.is_dir():
                try:
                    count = sum(1 for _ in entry.iterdir())
                except OSError:
                    count = 0
                rows.append(f"{entry.name}/ ({count} entries)")
            else:
                rows.append(entry.name)
            if len(rows) >= 40:
                break
        return rows

    def _readme(self) -> str:
        for name in ("README.md", "README.rst", "README.txt", "README"):
            path = self.root / name
            if path.is_file():
                try:
                    return path.read_text(encoding="utf-8", errors="replace")[:1500]
                except OSError:
                    return ""
        return ""


def _shallow(root: Path) -> List[Path]:
    try:
        return list(root.iterdir())
    except OSError:
        return []


def _git(root: Path, *args: str) -> str:
    try:
        done = subprocess.run(["git", "-C", str(root), *args],
                              capture_output=True, text=True, timeout=5)
        return done.stdout.strip() if done.returncode == 0 else ""
    except Exception:
        return ""


def inject(messages: Sequence[Dict[str, str]],
           context: Optional[RepoContext]) -> List[Dict[str, str]]:
    """Put the repo in front of the conversation, stable part first.

    Returns the messages unchanged when there is no context, so the caller does
    not have to branch and a general session pays nothing.
    """
    rows = [dict(m) for m in messages]
    if context is None:
        return rows
    question = ""
    for message in reversed(rows):
        if message.get("role") == "user":
            question = str(message.get("content") or "")
            break
    blocks = [context.stable_block()]
    volatile = context.volatile_block(question)
    if volatile:
        blocks.append(volatile)
    return [{"role": "system", "content": "\n\n".join(blocks)}, *rows]
