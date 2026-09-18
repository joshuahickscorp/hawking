"""Per-project state for the perception tools. SQLite plus content-addressed blobs.

The foreign package's ProjectStore carries 40+ tables. The nine tools hawking
actually allowlists touch three concepts: project metadata, content-addressed
artifacts, and observation records. This is those three, and nothing else --
sublation, not a clone.

State lives entirely under a root the CALLER supplies. Nothing here reads the
foreign checkout, and none of it needs the 624 MB artifacts/ tree that repo
carries: that is release output, structurally disconnected from these paths.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from hawking.persist import atomic_write_bytes

_HEX64 = re.compile(r"[0-9a-f]{64}")

SCHEMA = """
CREATE TABLE IF NOT EXISTS project_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    digest TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    source_name TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    adapter TEXT NOT NULL,
    adapter_version TEXT NOT NULL,
    status TEXT NOT NULL,
    authority TEXT NOT NULL,
    manifest_digest TEXT,
    subject_path TEXT,
    request_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ProjectStore:
    """One project directory: project.db plus a content-addressed blob tree.

    ``artifact_dirname`` is deliberately small and relative.  The default keeps
    all existing perception projects unchanged; H-Web uses ``pasted`` so its
    canonical local artifact owner has the human-readable H-NOTES layout.
    """

    def __init__(self, root: Path, artifact_dirname: str = "artifacts") -> None:
        self.root = Path(root).expanduser().resolve()
        self.db_path = self.root / "project.db"
        self.artifact_dirname = self._validate_artifact_dirname(artifact_dirname)

    @staticmethod
    def _validate_artifact_dirname(value: str) -> str:
        candidate = Path(str(value or "artifacts"))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("artifact directory must be a relative child")
        normalized = candidate.as_posix().strip("/")
        if not normalized or normalized == ".":
            raise ValueError("artifact directory must not be empty")
        return normalized

    @property
    def artifact_dir(self) -> Path:
        return self.root / self.artifact_dirname

    @classmethod
    def create(
        cls, root: Path, name: str, *, artifact_dirname: str = "artifacts"
    ) -> "ProjectStore":
        store = cls(root, artifact_dirname)
        store.root.mkdir(parents=True, exist_ok=True)
        store.artifact_dir.mkdir(parents=True, exist_ok=True)
        with store.connection() as cx:
            cx.executescript(SCHEMA)
            for k, v in (
                ("name", name),
                ("created_at", utc_now()),
                ("artifact_dir", store.artifact_dirname),
            ):
                cx.execute("INSERT OR IGNORE INTO project_meta(key,value) VALUES(?,?)", (k, v))
        return store

    @classmethod
    def open(cls, root: Path) -> "ProjectStore":
        store = cls(root)
        if not store.db_path.is_file():
            raise FileNotFoundError(f"no project at {store.root}")
        # The directory choice is persisted with the existing project metadata,
        # so opening a H-NOTES project through the compatibility symlink still
        # resolves the same canonical blob tree.
        try:
            with sqlite3.connect(store.db_path) as cx:
                row = cx.execute(
                    "SELECT value FROM project_meta WHERE key='artifact_dir'"
                ).fetchone()
        except sqlite3.Error:
            row = None
        if row and row[0]:
            store.artifact_dirname = cls._validate_artifact_dirname(str(row[0]))
        return store

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.root.mkdir(parents=True, exist_ok=True)
        cx = sqlite3.connect(self.db_path)
        cx.row_factory = sqlite3.Row
        try:
            cx.executescript(SCHEMA)
            yield cx
            cx.commit()
        finally:
            cx.close()

    # -- project ---------------------------------------------------------
    def project(self) -> dict[str, Any]:
        with self.connection() as cx:
            meta = {r["key"]: r["value"] for r in cx.execute("SELECT key,value FROM project_meta")}
        return {"root": str(self.root), "name": meta.get("name"),
                "created_at": meta.get("created_at")}

    def status(self) -> dict[str, Any]:
        with self.connection() as cx:
            counts = {t: cx.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
                      for t in ("artifacts", "observations")}
        return {"project": self.project(), "counts": counts,
                "directory_health": {"root_exists": self.root.is_dir(),
                                     "artifacts_dir": self.artifact_dir.is_dir(),
                                     "db": self.db_path.is_file()}}

    # -- artifacts -------------------------------------------------------
    def artifact_path(self, digest: str) -> Path:
        """Content-addressed: artifacts/<first2>/<rest>.

        The digest is attacker-influenced -- it arrives from tool arguments --
        so it is validated as a digest, not merely as a non-empty string. The
        length check alone let `../../etc/passwd` through and returned a path
        outside the project entirely.
        """
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise ValueError(
                f"not a sha256 digest: {digest!r}. A digest is 64 hex characters; "
                f"anything else could name a path outside the artifact tree.")
        out = (self.artifact_dir / digest[:2] / digest[2:]).resolve()
        # Belt and braces: even a digest that passed the pattern must land
        # inside the tree. A guard that only checks the input shape is one
        # regex edit away from being a guard that checks nothing.
        artifacts = self.artifact_dir.resolve()
        if artifacts != out and artifacts not in out.parents:
            raise ValueError(f"artifact path escapes the project: {out}")
        return out

    def ingest_file(self, src: Path, *, media_type: str,
                    source_name: str | None = None) -> dict[str, Any]:
        src = Path(src)
        digest = sha256_file(src)
        dest = self.artifact_path(digest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(src.read_bytes())
        rel = str(dest.relative_to(self.root))
        row = {"digest": digest, "size": dest.stat().st_size, "media_type": media_type,
               "relative_path": rel, "source_name": source_name or src.name,
               "created_at": utc_now()}
        with self.connection() as cx:
            cx.execute("INSERT OR REPLACE INTO artifacts"
                       "(digest,size,media_type,relative_path,source_name,created_at)"
                       " VALUES(?,?,?,?,?,?)",
                       tuple(row[k] for k in ("digest", "size", "media_type",
                                              "relative_path", "source_name", "created_at")))
        return row

    def ingest_bytes(self, payload: bytes, *, media_type: str,
                     source_name: str | None = None) -> dict[str, Any]:
        """Admit caller-owned bytes into the same content-addressed store.

        Browser uploads have no safe source path to hand to ``ingest_file``.
        This keeps the existing artifact owner while preserving the same
        digest, path validation, and crash-safe write contract.
        """
        if not isinstance(payload, bytes):
            raise TypeError(f"payload must be bytes, got {type(payload).__name__}")
        digest = hashlib.sha256(payload).hexdigest()
        dest = self.artifact_path(digest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            atomic_write_bytes(dest, payload)
        row = {
            "digest": digest,
            "size": dest.stat().st_size,
            "media_type": str(media_type or "application/octet-stream"),
            "relative_path": str(dest.relative_to(self.root)),
            "source_name": str(source_name or "")[:240],
            "created_at": utc_now(),
        }
        with self.connection() as cx:
            cx.execute(
                "INSERT OR IGNORE INTO artifacts"
                "(digest,size,media_type,relative_path,source_name,created_at)"
                " VALUES(?,?,?,?,?,?)",
                tuple(row[key] for key in (
                    "digest", "size", "media_type", "relative_path",
                    "source_name", "created_at",
                )),
            )
        return row

    def list_artifacts(self, limit: int = 100) -> list[dict[str, Any]]:
        bound = max(1, min(int(limit), 10_000))
        with self.connection() as cx:
            return [dict(r) for r in cx.execute(
                "SELECT digest,size,media_type,relative_path,source_name,created_at FROM artifacts"
                " ORDER BY created_at DESC, digest LIMIT ?", (bound,))]

    def artifact(self, digest: str) -> dict[str, Any] | None:
        with self.connection() as cx:
            r = cx.execute("SELECT digest,size,media_type,relative_path,source_name,created_at"
                           " FROM artifacts WHERE digest=?", (digest,)).fetchone()
        return dict(r) if r else None

    # -- observations ----------------------------------------------------
    def record_observation(self, *, adapter: str, adapter_version: str, status: str,
                           authority: str, manifest_digest: str | None,
                           subject_path: str | None, request: dict[str, Any]) -> str:
        cid = str(uuid.uuid4())
        with self.connection() as cx:
            cx.execute("INSERT INTO observations"
                       "(id,adapter,adapter_version,status,authority,manifest_digest,"
                       "subject_path,request_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                       (cid, adapter, adapter_version, status, authority, manifest_digest,
                        subject_path, json.dumps(request, sort_keys=True), utc_now()))
        return cid

    def observation(self, capture_id: str) -> dict[str, Any] | None:
        with self.connection() as cx:
            r = cx.execute("SELECT * FROM observations WHERE id=?", (capture_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["request"] = json.loads(d.pop("request_json"))
        return d

    def observations(self, limit: int = 100) -> list[dict[str, Any]]:
        bound = max(1, min(int(limit), 10_000))
        with self.connection() as cx:
            rows = [dict(r) for r in cx.execute(
                "SELECT id,adapter,adapter_version,status,authority,manifest_digest,"
                "subject_path,created_at FROM observations"
                " ORDER BY created_at DESC, id LIMIT ?", (bound,))]
        return rows
