"""Disk cache for pasted blobs, so a 200 KB log costs one line of context.

A paste is stored EXACTLY as handed over, under ``<root>/.hcli/pastes/<id>.txt``
with its metadata beside it in ``<id>.json``. Working context carries only
``PasteRef.context_ref()`` --
``[PASTE paste_20260901_010412_a81fb3c2 184KB 4210 lines log | ...]`` -- and the
model pulls back what it needs with ``slice`` / ``search``.

Three rules make this safe to point ``drop``/``prune`` at:

* every id goes through ``_ID_RE`` AND a resolved-parent check, so neither a
  crafted id (``../../receipts``) nor a symlink planted in the pastes dir can
  reach a byte outside ``self.dir``.
* writes go through ``hcli.persist.atomic_write_text``, so a crash leaves the
  previous bytes, never a truncated paste that still reads as valid. The text
  lands before the metadata, so the worst crash residue is an orphan ``.txt``
  that ``list()`` (which globs ``*.json``) never reports.
* kind detection only ever feeds the preview line. It cannot alter stored bytes.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from hcli.paths import find_repo_root
from hcli.persist import atomic_write_json, atomic_write_text

PASTES_DIRNAME = "pastes"
_ID_RE = re.compile(r"paste_\d{8}_\d{6}_[0-9a-f]{8}")
# 8 hex of the content sha, not 4: a per-second id would otherwise be one
# 16-bit collision away from two different blobs claiming the same file.
_SHORT_HASH = 8
_TEXT_SUFFIX = ".txt"
_META_SUFFIX = ".json"
_PREVIEW_CHARS = 60
_CONTEXT_REF_MAX = 118

_LOG_RE = re.compile(
    r"^\s*[\[(]?\d{4}-\d{2}-\d{2}[T ]|^\s*[\[(]?\d{2}:\d{2}:\d{2}"
    r"|\b(?:DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE)\b"
)
_CODE_RE = re.compile(
    r"^\s*(?:def |class |import |from \S+ import |function |fn |func |package "
    r"|#include|const |let |var |public |private |return |impl |struct |use )"
)
_DIFF_PREFIXES = ("diff --git", "@@ ", "--- a/", "+++ b/", "Index: ", "*** ")


class PasteNotFound(KeyError):
    """No such paste in this cache."""


@dataclass(frozen=True)
class PasteRef:
    """What goes back to the caller. ``context_ref()`` is what goes to the model."""

    id: str
    size: int
    sha256: str
    lines: int
    kind: str
    created_at: str
    preview: str
    session: Optional[str] = None
    mission: Optional[str] = None

    def context_ref(self) -> str:
        """One compact line, always shorter than ``_CONTEXT_REF_MAX``."""
        head = (
            f"[PASTE {self.id} {_human_size(self.size)} "
            f"{self.lines} lines {self.kind}"
        )
        body = f"{head} | {self.preview}" if self.preview else head
        if len(body) > _CONTEXT_REF_MAX - 1:
            body = body[: _CONTEXT_REF_MAX - 2].rstrip() + "…"
        return body + "]"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "PasteRef":
        return cls(**{k: obj.get(k) for k in cls.__dataclass_fields__})


def _human_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size // 1024}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def _count_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def detect_kind(text: str) -> str:
    """Best-effort label for the preview only. Never consulted when storing."""
    head = text.lstrip()
    if head[:1] in ("{", "["):
        try:
            json.loads(text)
            return "json"
        except ValueError:
            pass
    sample = [ln for ln in head[:8192].splitlines() if ln.strip()][:40]
    if not sample:
        return "text"
    if any(ln.startswith(_DIFF_PREFIXES) for ln in sample):
        return "diff"
    if sum(bool(_LOG_RE.search(ln)) for ln in sample) >= max(2, len(sample) // 3):
        return "log"
    if sum(bool(_CODE_RE.match(ln)) for ln in sample) >= max(1, len(sample) // 8):
        return "code"
    return "text"


def _preview(text: str) -> str:
    for line in text.splitlines():
        stripped = " ".join(line.split())
        if stripped:
            return stripped[:_PREVIEW_CHARS]
    return ""


class PasteCache:
    """``<root>/.hcli/pastes/``. ``root`` defaults to the repo root."""

    def __init__(self, root: Optional[Union[str, Path]] = None) -> None:
        self.root = Path(root) if root is not None else find_repo_root()
        self.dir = self.root / ".hcli" / PASTES_DIRNAME

    # -- paths -------------------------------------------------------------
    def _path(self, paste_id: str, suffix: str) -> Path:
        """The one door to a file in this cache. Refuses anything that escapes."""
        if not _ID_RE.fullmatch(paste_id or ""):
            raise ValueError(f"not a paste id: {paste_id!r}")
        path = self.dir / f"{paste_id}{suffix}"
        # resolve() follows symlinks, so this catches both a crafted id and a
        # symlink planted under the pastes dir that points somewhere else.
        if path.resolve().parent != self.dir.resolve():
            raise ValueError(f"paste id escapes {self.dir}: {paste_id!r}")
        return path

    # -- write -------------------------------------------------------------
    def store(
        self,
        text: str,
        *,
        session: Optional[str] = None,
        mission: Optional[str] = None,
    ) -> PasteRef:
        """Write *text* exactly. Identical content returns the existing ref."""
        if not isinstance(text, str):
            raise TypeError(f"paste must be str, got {type(text).__name__}")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = self._find_by_sha(sha)
        if existing is not None:
            return existing

        now = datetime.now()
        paste_id = f"paste_{now:%Y%m%d_%H%M%S}_{sha[:_SHORT_HASH]}"
        ref = PasteRef(
            id=paste_id,
            size=len(text.encode("utf-8")),
            sha256=sha,
            lines=_count_lines(text),
            kind=detect_kind(text),
            # Microseconds, not seconds: the id is only second-granular and is
            # tie-broken by CONTENT HASH, so it does not order a paste burst.
            # `list()` and `prune(keep_last=)` need a real clock to sort on.
            created_at=now.isoformat(timespec="microseconds"),
            preview=_preview(text),
            session=session,
            mission=mission,
        )
        # Text first: the metadata is the index, so a crash between the two
        # leaves an unreferenced .txt rather than a ref pointing at nothing.
        atomic_write_text(self._path(paste_id, _TEXT_SUFFIX), text)
        atomic_write_json(self._path(paste_id, _META_SUFFIX), ref.to_dict())
        return ref

    def _find_by_sha(self, sha: str) -> Optional[PasteRef]:
        """Dedupe lookup. The id embeds sha[:8], so one glob covers every hit."""
        for meta in sorted(self.dir.glob(f"paste_*_{sha[:_SHORT_HASH]}{_META_SUFFIX}")):
            ref = self._load_meta(meta)
            if ref is not None and ref.sha256 == sha:
                if self._path(ref.id, _TEXT_SUFFIX).is_file():
                    return ref
        return None

    # -- read --------------------------------------------------------------
    def get(self, paste_id: str) -> str:
        """The exact original text back, byte for byte."""
        path = self._path(paste_id, _TEXT_SUFFIX)
        if not path.is_file():
            raise PasteNotFound(paste_id)
        # newline="" disables universal-newline translation; without it a CRLF
        # paste would silently come back with LF endings.
        with open(path, "r", encoding="utf-8", newline="") as handle:
            text = handle.read()
        ref = self._load_meta(self._path(paste_id, _META_SUFFIX))
        if ref is not None and hashlib.sha256(text.encode("utf-8")).hexdigest() != ref.sha256:
            raise ValueError(f"{paste_id} does not match its recorded sha256")
        return text

    def slice(self, paste_id: str, start: int, end: int) -> str:
        """Lines *start*..*end*, 1-based inclusive, matching ``search`` numbers."""
        if start < 1:
            raise ValueError(f"line numbers are 1-based, got start={start}")
        # ponytail: reads the whole paste to cut a window. Fine at paste scale;
        # stream it if these ever get to gigabytes.
        lines = self.get(paste_id).splitlines()
        return "\n".join(lines[start - 1 : end])

    def search(self, paste_id: str, query: str, *, limit: int = 100) -> List[Tuple[int, str]]:
        """Plain substring hits as (1-based line no, line). Capped: this cache
        exists to keep context small, so an unbounded match list defeats it."""
        hits: List[Tuple[int, str]] = []
        for number, line in enumerate(self.get(paste_id).splitlines(), start=1):
            if query in line:
                hits.append((number, line))
                if len(hits) >= limit:
                    break
        return hits

    def list(self) -> List[PasteRef]:
        """Newest first, by recorded clock then by write order.

        NOT by id: the id is ``paste_<date>_<hhmmss>_<sha8>``, so within one
        second it sorts by CONTENT HASH. Sorting on it made ``prune`` delete
        the newest paste and keep an arbitrary middle one, which is exactly
        the sub-second regime a paste burst lands in. ``st_mtime_ns`` breaks
        any remaining tie with the order the sidecars were actually written.
        """
        pairs = []
        for meta in self.dir.glob(f"paste_*{_META_SUFFIX}"):
            ref = self._load_meta(meta)
            if ref is None:
                continue
            try:
                written = meta.stat().st_mtime_ns
            except OSError:
                written = 0
            pairs.append((ref.created_at or "", written, ref))
        pairs.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [ref for _, _, ref in pairs]

    def _load_meta(self, path: Path) -> Optional[PasteRef]:
        """A corrupt sidecar drops out of the listing; it never raises.

        AttributeError is in the net because valid JSON that is not an object
        (``[]``) reaches ``from_dict`` and calls ``.get`` on a list.
        """
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            ref = PasteRef.from_dict(obj)
        except (OSError, ValueError, TypeError, AttributeError):
            return None
        return ref if _ID_RE.fullmatch(ref.id or "") else None

    # -- delete ------------------------------------------------------------
    def drop(self, paste_id: str) -> bool:
        """Delete one paste. True if anything was there."""
        removed = False
        for suffix in (_TEXT_SUFFIX, _META_SUFFIX):
            path = self._path(paste_id, suffix)
            try:
                path.unlink()
                removed = True
            except FileNotFoundError:
                pass
        return removed

    def prune(
        self,
        *,
        older_than_days: Optional[float] = None,
        keep_last: Optional[int] = None,
    ) -> List[str]:
        """Bulk delete. The policy is explicit or there is no delete."""
        if older_than_days is None and keep_last is None:
            raise ValueError("prune needs a policy: older_than_days and/or keep_last")
        refs = self.list()
        doomed = []
        if keep_last is not None:
            doomed.extend(refs[keep_last:])
        if older_than_days is not None:
            cutoff = datetime.now().timestamp() - older_than_days * 86400
            for ref in refs:
                try:
                    born = datetime.fromisoformat(ref.created_at).timestamp()
                except (TypeError, ValueError):
                    continue
                if born < cutoff:
                    doomed.append(ref)
        dropped = []
        for ref in doomed:
            if ref.id not in dropped and self.drop(ref.id):
                dropped.append(ref.id)
        return dropped
