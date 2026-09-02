"""Queryable sidecar over large JSON receipts (object-of-objects maps).

The durable record is the human-readable JSON (REACHABILITY_TRIAGE.json,
CAPABILITY_GRAPH.json). This module never deletes or lossily rewrites those
files. It builds a SQLite sidecar of byte offsets + classification columns so
a targeted question — one module, every UNREACHABLE row, every BUILT gate —
is answered by pread of that slice (or an index seek), not json.loads of 2.7MB.

hawking-index already owns source-code parse/graph/merkle/daemon/query. The
Rust crate `artifact` module (and `hawking-artifact` bin) is the same sidecar
format; this file is the Python query path plus a fallback builder so HCLI
still answers when the bin is not on PATH. Both builders write schema v1.
A fast path that disagrees with a full parse is a bug: callers keep
json.loads as the oracle and as the fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / "_artifact_cache"
SCHEMA = "hawking.index.artifact_map.v1"
SCHEMA_VERSION = 1

TRIAGE_REL = "receipts/future/REACHABILITY_TRIAGE.json"
GRAPH_REL = "civilization/CAPABILITY_GRAPH.json"

GIT_TIMEOUT_S = 120


class ArtifactParityError(AssertionError):
    """Fast path and full parse disagreed. The full parse is the verdict."""


@dataclass
class QueryHit:
    value: Any
    bytes_read: int
    elapsed_s: float
    path: str  # "index" | "full"
    key: str | None = None
    map_name: str | None = None


def capability_id(module_rel: str) -> str:
    """Match tools.audit.reachability_triage.capability_id."""
    parts = Path(module_rel).with_suffix("").parts
    if parts and parts[0] == "tools":
        parts = parts[1:]
    return ".".join(parts)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_show_bytes(rel: str) -> bytes:
    """HEAD blob, unstripped. git()'s .strip() would shift byte offsets."""
    proc = subprocess.run(
        ["git", "--no-optional-locks", "show", f"HEAD:{rel}"],
        cwd=str(REPO),
        capture_output=True,
        timeout=GIT_TIMEOUT_S,
        check=False,
    )
    return proc.stdout or b""


def materialize(rel: str, dest_dir: Path | None = None) -> Path:
    """On-disk JSON for pread. Does not write into receipts/ or civilization/."""
    live = REPO / rel
    if live.is_file():
        return live
    dest_dir = dest_dir or CACHE
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / rel.replace("/", "__")
    blob = git_show_bytes(rel)
    if not blob:
        raise FileNotFoundError(f"no file and no HEAD blob for {rel}")
    if dest.is_file() and dest.stat().st_size == len(blob):
        existing = dest.read_bytes()
        if existing == blob:
            return dest
    dest.write_bytes(blob)
    return dest


def sidecar_path(json_path: Path, dest_dir: Path | None = None) -> Path:
    dest_dir = dest_dir or CACHE
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / (json_path.name + ".sqlite")


def _artifact_bin() -> Path | None:
    env = os.environ.get("CARGO_TARGET_DIR")
    names = ("release/hawking-artifact", "debug/hawking-artifact")
    candidates: list[Path] = []
    if env:
        for n in names:
            candidates.append(Path(env) / n)
    for root in (
        REPO / "target",
        REPO / "workspace" / "ops" / "build" / "rust",
        Path("/tmp") / "hawking-artifact-target",
    ):
        for n in names:
            candidates.append(root / n)
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


# --------------------------------------------------------------------------
# JSON object walker (byte offsets). Not a source-code AST scanner.
# --------------------------------------------------------------------------


class _Cur:
    def __init__(self, b: bytes, i: int = 0) -> None:
        self.b = b
        self.i = i

    def peek(self) -> int | None:
        return self.b[self.i] if self.i < len(self.b) else None

    def skip_ws(self) -> None:
        b, i, n = self.b, self.i, len(self.b)
        while i < n and b[i] in (0x20, 0x09, 0x0A, 0x0D):
            i += 1
        self.i = i

    def eat(self, c: int) -> None:
        if self.peek() != c:
            got = self.peek()
            raise ValueError(f"expected {c!r} at {self.i}, got {got!r}")
        self.i += 1

    def eat_lit(self, lit: bytes) -> None:
        if self.b[self.i : self.i + len(lit)] != lit:
            raise ValueError(f"expected {lit!r} at {self.i}")
        self.i += len(lit)

    def skip_string(self) -> None:
        self.eat(0x22)
        b, i, n = self.b, self.i, len(self.b)
        while i < n:
            c = b[i]
            i += 1
            if c == 0x5C:  # backslash
                if i >= n:
                    raise ValueError("unterminated escape")
                e = b[i]
                i += 1
                if e == 0x75:  # uXXXX
                    if i + 4 > n:
                        raise ValueError("truncated \\uXXXX")
                    i += 4
            elif c == 0x22:
                self.i = i
                return
        raise ValueError("unterminated string")

    def parse_string(self) -> str:
        start = self.i
        self.skip_string()
        return json.loads(self.b[start : self.i].decode("utf-8"))

    def skip_number(self) -> None:
        start = self.i
        if self.peek() == 0x2D:
            self.i += 1
        saw = False
        while self.peek() is not None and 0x30 <= self.peek() <= 0x39:
            saw = True
            self.i += 1
        if self.peek() == 0x2E:
            self.i += 1
            while self.peek() is not None and 0x30 <= self.peek() <= 0x39:
                saw = True
                self.i += 1
        if self.peek() in (0x65, 0x45):
            self.i += 1
            if self.peek() in (0x2B, 0x2D):
                self.i += 1
            while self.peek() is not None and 0x30 <= self.peek() <= 0x39:
                saw = True
                self.i += 1
        if not saw or self.i == start:
            raise ValueError(f"invalid number at {start}")

    def skip_value(self) -> tuple[int, int]:
        self.skip_ws()
        start = self.i
        p = self.peek()
        if p == 0x7B:
            self.skip_object()
        elif p == 0x5B:
            self.skip_array()
        elif p == 0x22:
            self.skip_string()
        elif p == 0x74:
            self.eat_lit(b"true")
        elif p == 0x66:
            self.eat_lit(b"false")
        elif p == 0x6E:
            self.eat_lit(b"null")
        elif p == 0x2D or (p is not None and 0x30 <= p <= 0x39):
            self.skip_number()
        else:
            raise ValueError(f"unexpected value start at {start}: {p!r}")
        return start, self.i

    def skip_object(self) -> None:
        self.eat(0x7B)
        while True:
            self.skip_ws()
            if self.peek() == 0x7D:
                self.i += 1
                return
            self.skip_string()
            self.skip_ws()
            self.eat(0x3A)
            self.skip_value()
            self.skip_ws()
            p = self.peek()
            if p == 0x2C:
                self.i += 1
            elif p == 0x7D:
                self.i += 1
                return
            else:
                raise ValueError(f"expected ',' or '}}' at {self.i}")

    def skip_array(self) -> None:
        self.eat(0x5B)
        while True:
            self.skip_ws()
            if self.peek() == 0x5D:
                self.i += 1
                return
            self.skip_value()
            self.skip_ws()
            p = self.peek()
            if p == 0x2C:
                self.i += 1
            elif p == 0x5D:
                self.i += 1
                return
            else:
                raise ValueError(f"expected ',' or ']' at {self.i}")


def walk_object_members(data: bytes, obj_start: int = 0) -> list[tuple[str, int, int]]:
    c = _Cur(data, obj_start)
    c.skip_ws()
    c.eat(0x7B)
    out: list[tuple[str, int, int]] = []
    while True:
        c.skip_ws()
        if c.peek() == 0x7D:
            c.i += 1
            break
        key = c.parse_string()
        c.skip_ws()
        c.eat(0x3A)
        vs, ve = c.skip_value()
        out.append((key, vs, ve))
        c.skip_ws()
        p = c.peek()
        if p == 0x2C:
            c.i += 1
        elif p == 0x7D:
            c.i += 1
            break
        else:
            raise ValueError(f"expected ',' or '}}' after member at {c.i}")
    return out


def _looks_like_object_map(data: bytes, start: int, end: int) -> bool:
    slice_ = data[start:end]
    try:
        members = walk_object_members(slice_, 0)
    except ValueError:
        return False
    if not members:
        return False
    for _k, s, e in members:
        v = slice_[s:e].lstrip()
        if not v.startswith(b"{"):
            return False
    return True


def _extract_fields(slice_: bytes) -> tuple[str | None, str | None, str | None]:
    try:
        obj = json.loads(slice_)
    except ValueError:
        return None, None, None
    if not isinstance(obj, dict):
        return None, None, None

    def s(key: str) -> str | None:
        v = obj.get(key)
        return v if isinstance(v, str) else None

    return s("classification"), s("disposition"), s("status")


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS entity (
            map_name TEXT NOT NULL,
            key TEXT NOT NULL,
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            json TEXT NOT NULL,
            classification TEXT,
            disposition TEXT,
            status TEXT,
            cap_id TEXT,
            PRIMARY KEY (map_name, key)
        );
        CREATE INDEX IF NOT EXISTS entity_class ON entity(map_name, classification);
        CREATE INDEX IF NOT EXISTS entity_disp ON entity(map_name, disposition);
        CREATE INDEX IF NOT EXISTS entity_status ON entity(map_name, status);
        CREATE INDEX IF NOT EXISTS entity_cap ON entity(map_name, cap_id);
        """
    )


def build_python(
    json_path: Path,
    index_path: Path,
    maps: Iterable[str] | None = None,
) -> Path:
    """Fallback builder. Same schema as hawking-artifact index."""
    data = json_path.read_bytes()
    digest = sha256_hex(data)
    wanted = [m for m in (maps or []) if m]
    top = walk_object_members(data, 0)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        index_path.unlink()
    conn = sqlite3.connect(str(index_path))
    try:
        _init_schema(conn)
        conn.execute("DELETE FROM entity")
        conn.execute("DELETE FROM meta")
        n = 0
        for key, start, end in top:
            slice_ = data[start:end]
            is_map = _looks_like_object_map(data, start, end)
            take = (key in wanted) if wanted else is_map
            if take and is_map:
                members = walk_object_members(data, start)
                for ek, es, ee in members:
                    ent = data[es:ee]
                    classification, disposition, status = _extract_fields(ent)
                    cap = capability_id(ek) if key == "modules" else None
                    conn.execute(
                        "INSERT INTO entity(map_name, key, start, end, json, "
                        "classification, disposition, status, cap_id) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            key,
                            ek,
                            es,
                            ee,
                            ent.decode("utf-8"),
                            classification,
                            disposition,
                            status,
                            cap,
                        ),
                    )
                    n += 1
            else:
                conn.execute(
                    "INSERT INTO entity(map_name, key, start, end, json, "
                    "classification, disposition, status, cap_id) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    ("_root", key, start, end, slice_.decode("utf-8"), None, None, None, None),
                )
                n += 1
        st = json_path.stat()
        meta = {
            "schema": SCHEMA,
            "schema_version": str(SCHEMA_VERSION),
            "source_path": str(json_path),
            "source_sha256": digest,
            "source_size": str(len(data)),
            "source_mtime_ns": str(st.st_mtime_ns),
            "n_entities": str(n),
            "builder": "python",
            "hash_alg": "sha256",
        }
        conn.executemany(
            "INSERT OR REPLACE INTO meta(k, v) VALUES (?, ?)", list(meta.items())
        )
        conn.commit()
    finally:
        conn.close()
    return index_path


def build_rust(
    json_path: Path,
    index_path: Path,
    maps: Iterable[str] | None = None,
) -> Path:
    binary = _artifact_bin()
    if binary is None:
        raise FileNotFoundError("hawking-artifact binary not found")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "index",
        "--input",
        str(json_path),
        "--output",
        str(index_path),
    ]
    maps_s = ",".join(m for m in (maps or []) if m)
    if maps_s:
        cmd.extend(["--maps", maps_s])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"hawking-artifact index failed ({proc.returncode}): {proc.stderr or proc.stdout}"
        )
    return index_path


def _meta(conn: sqlite3.Connection) -> dict[str, str]:
    return {k: v for k, v in conn.execute("SELECT k, v FROM meta")}


def is_fresh(index_path: Path, json_path: Path) -> bool:
    if not index_path.is_file() or not json_path.is_file():
        return False
    conn = sqlite3.connect(f"file:{index_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        meta = _meta(conn)
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    if meta.get("schema") != SCHEMA:
        return False
    if meta.get("schema_version") != str(SCHEMA_VERSION):
        return False
    st = json_path.stat()
    if meta.get("source_size") != str(st.st_size):
        return False
    if meta.get("source_mtime_ns") == str(st.st_mtime_ns):
        return True
    data = json_path.read_bytes()
    alg = meta.get("hash_alg") or "sha256"
    if alg == "sha256":
        return meta.get("source_sha256") == sha256_hex(data)
    # blake3 sidecar (rust builder): Python cannot re-hash; size matched but
    # mtime did not, so treat as stale rather than guess.
    return False


def ensure_index(
    json_path: Path,
    index_path: Path | None = None,
    maps: Iterable[str] | None = None,
    *,
    prefer_rust: bool = True,
) -> Path:
    index_path = index_path or sidecar_path(json_path)
    if is_fresh(index_path, json_path):
        return index_path
    if prefer_rust and _artifact_bin() is not None:
        try:
            return build_rust(json_path, index_path, maps)
        except (RuntimeError, FileNotFoundError):
            pass
    return build_python(json_path, index_path, maps)


def _pread_or_stored(
    json_path: Path | None,
    start: int,
    end: int,
    stored: str,
) -> tuple[bytes, int, str]:
    """Prefer a counted pread of the original JSON; fall back to the stored slice."""
    length = end - start
    if json_path is not None and json_path.is_file() and length >= 0:
        fd = os.open(json_path, os.O_RDONLY)
        try:
            blob = os.pread(fd, length, start)
        finally:
            os.close(fd)
        if blob.decode("utf-8") != stored:
            # Offset drift: refuse the fast path rather than emit a wrong verdict.
            raise ArtifactParityError(
                f"pread[{start}:{end}] != stored json ({len(blob)} vs {len(stored)} bytes)"
            )
        return blob, len(blob), "index"
    encoded = stored.encode("utf-8")
    return encoded, len(encoded), "index"


def get(
    map_name: str,
    key: str,
    *,
    json_path: Path,
    index_path: Path | None = None,
) -> QueryHit:
    t0 = time.perf_counter()
    index_path = ensure_index(json_path, index_path)
    conn = sqlite3.connect(f"file:{index_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT start, end, json FROM entity WHERE map_name = ? AND key = ?",
            (map_name, key),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise KeyError(f"{map_name}:{key}")
    start, end, stored = int(row[0]), int(row[1]), str(row[2])
    blob, n, path = _pread_or_stored(json_path, start, end, stored)
    value = json.loads(blob)
    return QueryHit(
        value=value,
        bytes_read=n,
        elapsed_s=time.perf_counter() - t0,
        path=path,
        key=key,
        map_name=map_name,
    )


def get_by_cap_id(
    cap_id: str,
    *,
    json_path: Path,
    index_path: Path | None = None,
    map_name: str = "modules",
) -> QueryHit | None:
    t0 = time.perf_counter()
    index_path = ensure_index(json_path, index_path)
    conn = sqlite3.connect(f"file:{index_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT key, start, end, json FROM entity WHERE cap_id = ? AND map_name = ? LIMIT 1",
            (cap_id, map_name),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    key, start, end, stored = str(row[0]), int(row[1]), int(row[2]), str(row[3])
    blob, n, path = _pread_or_stored(json_path, start, end, stored)
    return QueryHit(
        value=json.loads(blob),
        bytes_read=n,
        elapsed_s=time.perf_counter() - t0,
        path=path,
        key=key,
        map_name=map_name,
    )


def list_keys(
    map_name: str,
    *,
    json_path: Path,
    index_path: Path | None = None,
    classification: str | None = None,
    disposition: str | None = None,
    status: str | None = None,
) -> tuple[list[str], int, float]:
    t0 = time.perf_counter()
    index_path = ensure_index(json_path, index_path)
    conn = sqlite3.connect(f"file:{index_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        where = ["map_name = ?"]
        args: list[Any] = [map_name]
        if classification is not None:
            where.append("classification = ?")
            args.append(classification)
        if disposition is not None:
            where.append("disposition = ?")
            args.append(disposition)
        if status is not None:
            where.append("status = ?")
            args.append(status)
        sql = f"SELECT key FROM entity WHERE {' AND '.join(where)} ORDER BY key"
        keys = [r[0] for r in conn.execute(sql, args)]
    finally:
        conn.close()
    payload = sum(len(k.encode("utf-8")) for k in keys)
    return keys, payload, time.perf_counter() - t0


def full_parse(json_path: Path) -> tuple[Any, int, float]:
    t0 = time.perf_counter()
    data = json_path.read_bytes()
    doc = json.loads(data)
    return doc, len(data), time.perf_counter() - t0


def parity_map(
    json_path: Path,
    map_name: str,
    index_path: Path | None = None,
) -> dict[str, Any]:
    """Compare every key in map_name: index get == full parse. All keys, not a sample."""
    doc, full_bytes, full_s = full_parse(json_path)
    obj = doc.get(map_name)
    if not isinstance(obj, dict):
        raise ArtifactParityError(f"{json_path} has no object map {map_name!r}")
    index_path = ensure_index(json_path, index_path)
    mismatches: list[str] = []
    bytes_index = 0
    t0 = time.perf_counter()
    for key in obj:
        hit = get(map_name, key, json_path=json_path, index_path=index_path)
        bytes_index += hit.bytes_read
        if hit.value != obj[key]:
            mismatches.append(key)
    keys_idx, key_bytes, _ = list_keys(map_name, json_path=json_path, index_path=index_path)
    index_s = time.perf_counter() - t0
    full_keys = sorted(obj)
    if keys_idx != full_keys:
        extra = sorted(set(keys_idx) - set(full_keys))
        missing = sorted(set(full_keys) - set(keys_idx))
        mismatches.extend(f"keyset extra {k}" for k in extra)
        mismatches.extend(f"keyset missing {k}" for k in missing)
    return {
        "ok": not mismatches,
        "map": map_name,
        "n": len(obj),
        "n_equal": len(obj) - len([m for m in mismatches if not str(m).startswith("keyset ")]),
        "n_mismatch": len(mismatches),
        "mismatches": mismatches[:20],
        "full_bytes": full_bytes,
        "full_s": full_s,
        "index_bytes_sum": bytes_index,
        "index_key_bytes": key_bytes,
        "index_s_all_gets": index_s,
        "source": str(json_path),
        "index": str(index_path),
    }


def measure_one(
    json_path: Path,
    map_name: str,
    key: str,
    index_path: Path | None = None,
) -> dict[str, Any]:
    """Time + bytes-read for one key, both paths. Index is built first so the
    measured fast path is the query, not the (one-time) index build."""
    index_path = ensure_index(json_path, index_path)
    doc, full_bytes, full_s = full_parse(json_path)
    expected = doc[map_name][key]
    hit = get(map_name, key, json_path=json_path, index_path=index_path)
    if hit.value != expected:
        raise ArtifactParityError(f"{map_name}:{key} index != full parse")
    return {
        "key": key,
        "map": map_name,
        "equal": True,
        "full_s": full_s,
        "full_bytes": full_bytes,
        "index_s": hit.elapsed_s,
        "index_bytes": hit.bytes_read,
        "index_path": hit.path,
    }


def measure_filter(
    json_path: Path,
    map_name: str,
    *,
    classification: str | None = None,
    disposition: str | None = None,
    status: str | None = None,
    index_path: Path | None = None,
) -> dict[str, Any]:
    index_path = ensure_index(json_path, index_path)
    doc, full_bytes, full_s = full_parse(json_path)
    obj = doc[map_name]
    def want(row: Mapping[str, Any]) -> bool:
        if classification is not None and row.get("classification") != classification:
            return False
        if disposition is not None and row.get("disposition") != disposition:
            return False
        if status is not None and row.get("status") != status:
            return False
        return True
    full_keys = sorted(k for k, v in obj.items() if want(v))
    keys, idx_bytes, idx_s = list_keys(
        map_name,
        json_path=json_path,
        index_path=index_path,
        classification=classification,
        disposition=disposition,
        status=status,
    )
    if keys != full_keys:
        raise ArtifactParityError(
            f"filter mismatch: extra={sorted(set(keys)-set(full_keys))[:5]} "
            f"missing={sorted(set(full_keys)-set(keys))[:5]}"
        )
    return {
        "map": map_name,
        "classification": classification,
        "disposition": disposition,
        "status": status,
        "n": len(keys),
        "equal": True,
        "full_s": full_s,
        "full_bytes": full_bytes,
        "index_s": idx_s,
        "index_bytes": idx_bytes,
        "keys_head": keys[:8],
    }
