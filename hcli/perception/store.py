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
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

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
    """One project directory: project.db plus an artifacts/ blob tree."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.db_path = self.root / "project.db"

    @classmethod
    def create(cls, root: Path, name: str) -> "ProjectStore":
        store = cls(root)
        store.root.mkdir(parents=True, exist_ok=True)
        (store.root / "artifacts").mkdir(exist_ok=True)
        with store.connection() as cx:
            cx.executescript(SCHEMA)
            for k, v in (("name", name), ("created_at", utc_now())):
                cx.execute("INSERT OR IGNORE INTO project_meta(key,value) VALUES(?,?)", (k, v))
        return store

    @classmethod
    def open(cls, root: Path) -> "ProjectStore":
        store = cls(root)
        if not store.db_path.is_file():
            raise FileNotFoundError(f"no project at {store.root}")
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
                                     "artifacts_dir": (self.root / "artifacts").is_dir(),
                                     "db": self.db_path.is_file()}}

    # -- artifacts -------------------------------------------------------
    def artifact_path(self, digest: str) -> Path:
        """Content-addressed: artifacts/<first2>/<rest>."""
        if len(digest) < 3:
            raise ValueError(f"implausible digest {digest!r}")
        return self.root / "artifacts" / digest[:2] / digest[2:]

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

    def list_artifacts(self, limit: int = 100) -> list[dict[str, Any]]:
        bound = max(1, min(int(limit), 10_000))
        with self.connection() as cx:
            return [dict(r) for r in cx.execute(
                "SELECT digest,size,media_type,relative_path,created_at FROM artifacts"
                " ORDER BY created_at DESC, digest LIMIT ?", (bound,))]

    def artifact(self, digest: str) -> dict[str, Any] | None:
        with self.connection() as cx:
            r = cx.execute("SELECT digest,size,media_type,relative_path,created_at"
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
