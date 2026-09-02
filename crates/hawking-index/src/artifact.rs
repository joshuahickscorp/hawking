//! Queryable sidecar over large JSON receipts (object-of-objects maps).
//!
//! hawking-index already covers *source code*: [`crate::parse::parse_source`],
//! SCIP ids, the symbol graph, BLAKE3 merkle-diff, the indexing daemon, and
//! SQLite/FTS5 query. This module does **not** scan ASTs and does **not**
//! replace those layers. It answers a different question: given a 2.7MB
//! pretty-printed receipt (`modules` / `gates` maps), return one entity — or
//! the set of keys matching a classification/disposition/status — without
//! `json.loads` of the whole document.
//!
//! Access strategy
//! ---------------
//! A SQLite sidecar next to a materialized copy of the human-readable JSON.
//! The JSON remains the durable record (never deleted, never lossily
//! rewritten). The sidecar stores per-entity **byte offsets into that JSON**
//! plus extracted `classification` / `disposition` / `status` columns and the
//! original JSON substring. A targeted get is `pread(end-start)` + parse of
//! that slice. A filter is an index seek. Freshness is `sha256(source)` in
//! `meta`; a mismatch rebuilds. A fast path that disagrees with a full parse
//! is a bug — callers keep the full parse as the fallback and as the parity
//! oracle.

use hide_core::{HideError, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

pub const SCHEMA_VERSION: u32 = 1;
pub const META_SCHEMA: &str = "hawking.index.artifact_map.v1";

/// One entity in a named object-map (`modules`, `gates`, …).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityRef {
    pub map_name: String,
    pub key: String,
    pub start: u64,
    pub end: u64,
    pub classification: Option<String>,
    pub disposition: Option<String>,
    pub status: Option<String>,
    pub cap_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactMeta {
    pub schema: String,
    pub schema_version: u32,
    pub source_path: String,
    pub source_sha256: String,
    pub source_size: u64,
    pub n_entities: u64,
    pub hash_alg: String,
}

/// Durable sidecar. One SQLite file per source JSON.
pub struct ArtifactIndex {
    conn: Connection,
    path: PathBuf,
}

fn map_err(e: rusqlite::Error) -> HideError {
    HideError::Storage(e.to_string())
}

fn parse_err(msg: impl Into<String>) -> HideError {
    HideError::msg(format!("artifact json: {}", msg.into()))
}

/// Content hash of the source JSON. blake3 is already a hawking-index
/// dependency (merkle); Python stores sha256 under the same column and
/// records `hash_alg` so freshness checks pick the matching digest.
pub fn content_hash_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// Capability id matching `tools.audit.reachability_triage.capability_id`.
pub fn capability_id(module_rel: &str) -> String {
    let trimmed = module_rel.strip_suffix(".py").unwrap_or(module_rel);
    let mut parts: Vec<&str> = trimmed.split('/').filter(|p| !p.is_empty()).collect();
    if parts.first() == Some(&"tools") {
        parts.remove(0);
    }
    parts.join(".")
}

impl ArtifactIndex {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let conn = Connection::open(&path).map_err(map_err)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA busy_timeout=5000;
             PRAGMA temp_store=MEMORY;",
        )
        .map_err(map_err)?;
        Ok(Self { conn, path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            r#"
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
            "#,
        )
        .map_err(map_err)?;
        Ok(())
    }

    /// Build (or rebuild) an index of `json_path` into `index_path`.
    ///
    /// `maps`: named top-level object-of-objects to explode (e.g. `modules`,
    /// `gates`). Empty → auto-detect every top-level object whose values are
    /// all objects.
    pub fn build(
        json_path: impl AsRef<Path>,
        index_path: impl AsRef<Path>,
        maps: &[&str],
    ) -> Result<Self> {
        let json_path = json_path.as_ref();
        let index_path = index_path.as_ref();
        let bytes = fs::read(json_path)?;
        let source_sha = content_hash_hex(&bytes);
        let source_size = bytes.len() as u64;
        let source_mtime_ns = fs::metadata(json_path)
            .ok()
            .and_then(|md| md.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_nanos().to_string())
            .unwrap_or_else(|| "0".into());

        if let Some(parent) = index_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if index_path.exists() {
            let _ = fs::remove_file(index_path);
        }

        let idx = Self::open(index_path)?;
        Self::init_schema(&idx.conn)?;

        let wanted: Option<Vec<String>> = if maps.is_empty() {
            None
        } else {
            Some(maps.iter().map(|s| (*s).to_string()).collect())
        };

        let top = walk_object_members(&bytes, 0)?;
        let mut n_entities: u64 = 0;

        let tx = idx.conn.unchecked_transaction().map_err(map_err)?;
        tx.execute("DELETE FROM entity", []).map_err(map_err)?;
        tx.execute("DELETE FROM meta", []).map_err(map_err)?;

        let mut insert = tx
            .prepare(
                "INSERT INTO entity(map_name, key, start, end, json, classification, disposition, status, cap_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )
            .map_err(map_err)?;

        for (key, start, end) in &top {
            let slice = &bytes[*start..*end];
            let is_map = looks_like_object_map(slice);
            let take = match &wanted {
                Some(names) => names.iter().any(|n| n == key),
                None => is_map,
            };
            if take && is_map {
                let members = walk_object_members(&bytes, *start)?;
                for (ek, es, ee) in members {
                    let ent_slice = &bytes[es..ee];
                    let fields = extract_fields(ent_slice);
                    let cap = if key == "modules" {
                        Some(capability_id(&ek))
                    } else {
                        None
                    };
                    let json_text = std::str::from_utf8(ent_slice)
                        .map_err(|e| parse_err(format!("utf8 in {key}.{ek}: {e}")))?;
                    insert
                        .execute(params![
                            key,
                            ek,
                            es as i64,
                            ee as i64,
                            json_text,
                            fields.classification,
                            fields.disposition,
                            fields.status,
                            cap,
                        ])
                        .map_err(map_err)?;
                    n_entities += 1;
                }
            } else {
                let json_text = std::str::from_utf8(slice)
                    .map_err(|e| parse_err(format!("utf8 in _root.{key}: {e}")))?;
                insert
                    .execute(params![
                        "_root",
                        key,
                        *start as i64,
                        *end as i64,
                        json_text,
                        None::<String>,
                        None::<String>,
                        None::<String>,
                        None::<String>,
                    ])
                    .map_err(map_err)?;
                n_entities += 1;
            }
        }
        drop(insert);

        let put_meta = |tx: &rusqlite::Transaction, k: &str, v: &str| -> Result<()> {
            tx.execute(
                "INSERT OR REPLACE INTO meta(k, v) VALUES (?1, ?2)",
                params![k, v],
            )
            .map_err(map_err)?;
            Ok(())
        };
        put_meta(&tx, "schema", META_SCHEMA)?;
        put_meta(&tx, "schema_version", &SCHEMA_VERSION.to_string())?;
        put_meta(&tx, "source_path", &json_path.display().to_string())?;
        put_meta(&tx, "source_sha256", &source_sha)?;
        put_meta(&tx, "source_size", &source_size.to_string())?;
        put_meta(&tx, "source_mtime_ns", &source_mtime_ns)?;
        put_meta(&tx, "n_entities", &n_entities.to_string())?;
        put_meta(&tx, "builder", "rust")?;
        put_meta(&tx, "hash_alg", "blake3")?;
        tx.commit().map_err(map_err)?;
        Ok(idx)
    }

    pub fn meta(&self) -> Result<ArtifactMeta> {
        let get = |k: &str| -> Result<String> {
            self.conn
                .query_row("SELECT v FROM meta WHERE k = ?1", params![k], |r| r.get(0))
                .map_err(map_err)
        };
        Ok(ArtifactMeta {
            schema: get("schema")?,
            schema_version: get("schema_version")?
                .parse()
                .unwrap_or(0),
            source_path: get("source_path")?,
            source_sha256: get("source_sha256")?,
            source_size: get("source_size")?.parse().unwrap_or(0),
            n_entities: get("n_entities")?.parse().unwrap_or(0),
            hash_alg: get("hash_alg").unwrap_or_else(|_| "blake3".into()),
        })
    }

    fn meta_get(&self, k: &str) -> Result<Option<String>> {
        self.conn
            .query_row("SELECT v FROM meta WHERE k = ?1", params![k], |r| r.get(0))
            .optional()
            .map_err(map_err)
    }

    pub fn is_fresh(&self, json_path: impl AsRef<Path>) -> Result<bool> {
        let meta = self.meta()?;
        if meta.schema != META_SCHEMA || meta.schema_version != SCHEMA_VERSION {
            return Ok(false);
        }
        let md = fs::metadata(json_path.as_ref())?;
        if meta.source_size != md.len() {
            return Ok(false);
        }
        let mtime_ns = md
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_nanos().to_string());
        if let (Some(stored), Some(now)) = (self.meta_get("source_mtime_ns")?, mtime_ns) {
            if stored == now {
                return Ok(true);
            }
        }
        let bytes = fs::read(json_path.as_ref())?;
        Ok(meta.source_sha256 == content_hash_hex(&bytes))
    }

    /// Original JSON substring for one entity. None if missing.
    pub fn get_json(&self, map: &str, key: &str) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT json FROM entity WHERE map_name = ?1 AND key = ?2",
                params![map, key],
                |r| r.get(0),
            )
            .optional()
            .map_err(map_err)
    }

    pub fn get_ref(&self, map: &str, key: &str) -> Result<Option<EntityRef>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT map_name, key, start, end, classification, disposition, status, cap_id
                 FROM entity WHERE map_name = ?1 AND key = ?2",
            )
            .map_err(map_err)?;
        let row = stmt
            .query_row(params![map, key], row_to_ref)
            .optional()
            .map_err(map_err)?;
        Ok(row)
    }

    pub fn get_by_cap_id(&self, cap_id: &str) -> Result<Option<EntityRef>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT map_name, key, start, end, classification, disposition, status, cap_id
                 FROM entity WHERE cap_id = ?1 LIMIT 1",
            )
            .map_err(map_err)?;
        stmt.query_row(params![cap_id], row_to_ref)
            .optional()
            .map_err(map_err)
    }

    pub fn keys(&self, map: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT key FROM entity WHERE map_name = ?1 ORDER BY key")
            .map_err(map_err)?;
        let rows = stmt
            .query_map(params![map], |r| r.get::<_, String>(0))
            .map_err(map_err)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(map_err)?);
        }
        Ok(out)
    }

    /// Keys in `map` whose column equals `value`. `column` is one of
    /// classification / disposition / status.
    pub fn keys_where(&self, map: &str, column: &str, value: &str) -> Result<Vec<String>> {
        let col = match column {
            "classification" | "disposition" | "status" => column,
            _ => {
                return Err(HideError::msg(format!(
                    "artifact filter column must be classification|disposition|status, got {column}"
                )))
            }
        };
        let sql = format!(
            "SELECT key FROM entity WHERE map_name = ?1 AND {col} = ?2 ORDER BY key"
        );
        let mut stmt = self.conn.prepare(&sql).map_err(map_err)?;
        let rows = stmt
            .query_map(params![map, value], |r| r.get::<_, String>(0))
            .map_err(map_err)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(map_err)?);
        }
        Ok(out)
    }

    pub fn maps(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT map_name FROM entity ORDER BY map_name")
            .map_err(map_err)?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(map_err)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(map_err)?);
        }
        Ok(out)
    }
}

fn row_to_ref(r: &rusqlite::Row<'_>) -> rusqlite::Result<EntityRef> {
    Ok(EntityRef {
        map_name: r.get(0)?,
        key: r.get(1)?,
        start: r.get::<_, i64>(2)? as u64,
        end: r.get::<_, i64>(3)? as u64,
        classification: r.get(4)?,
        disposition: r.get(5)?,
        status: r.get(6)?,
        cap_id: r.get(7)?,
    })
}

#[derive(Default)]
struct Fields {
    classification: Option<String>,
    disposition: Option<String>,
    status: Option<String>,
}

fn extract_fields(slice: &[u8]) -> Fields {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(slice) else {
        return Fields::default();
    };
    let s = |key: &str| -> Option<String> {
        v.get(key)
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
    };
    Fields {
        classification: s("classification"),
        disposition: s("disposition"),
        status: s("status"),
    }
}

fn looks_like_object_map(slice: &[u8]) -> bool {
    let mut c = Cursor { b: slice, i: 0 };
    c.skip_ws();
    if c.peek() != Some(b'{') {
        return false;
    }
    let Ok(members) = walk_object_members(slice, 0) else {
        return false;
    };
    if members.is_empty() {
        return false;
    }
    members.iter().all(|(_, s, e)| {
        let v = &slice[*s..*e];
        let mut cc = Cursor { b: v, i: 0 };
        cc.skip_ws();
        cc.peek() == Some(b'{')
    })
}

/// Members of the JSON object whose `{` begins at `obj_start` in `bytes`
/// (or at the first `{` after `obj_start` following whitespace). Returns
/// (decoded_key, value_start, value_end) in byte offsets into `bytes`.
fn walk_object_members(bytes: &[u8], obj_start: usize) -> Result<Vec<(String, usize, usize)>> {
    let mut c = Cursor {
        b: bytes,
        i: obj_start,
    };
    c.skip_ws();
    c.eat(b'{')?;
    let mut out = Vec::new();
    loop {
        c.skip_ws();
        if c.peek() == Some(b'}') {
            break;
        }
        let key = c.parse_string()?;
        c.skip_ws();
        c.eat(b':')?;
        let (vs, ve) = c.skip_value()?;
        out.push((key, vs, ve));
        c.skip_ws();
        match c.peek() {
            Some(b',') => c.i += 1,
            Some(b'}') => break,
            other => {
                return Err(parse_err(format!(
                    "expected ',' or '}}' after object member at byte {}, got {:?}",
                    c.i, other
                )))
            }
        }
    }
    Ok(out)
}

struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<()> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(parse_err(format!(
                "expected {:?} at byte {}, got {:?}",
                c as char,
                self.i,
                self.peek().map(|x| x as char)
            )))
        }
    }

    fn eat_lit(&mut self, lit: &[u8]) -> Result<()> {
        if self.b.get(self.i..self.i + lit.len()) == Some(lit) {
            self.i += lit.len();
            Ok(())
        } else {
            Err(parse_err(format!(
                "expected {} at byte {}",
                String::from_utf8_lossy(lit),
                self.i
            )))
        }
    }

    fn skip_string(&mut self) -> Result<()> {
        self.eat(b'"')?;
        while self.i < self.b.len() {
            let c = self.b[self.i];
            self.i += 1;
            if c == b'\\' {
                if self.i >= self.b.len() {
                    return Err(parse_err("unterminated escape"));
                }
                let e = self.b[self.i];
                self.i += 1;
                if e == b'u' {
                    if self.i + 4 > self.b.len() {
                        return Err(parse_err("truncated \\uXXXX"));
                    }
                    self.i += 4;
                }
            } else if c == b'"' {
                return Ok(());
            }
        }
        Err(parse_err("unterminated string"))
    }

    fn parse_string(&mut self) -> Result<String> {
        let start = self.i;
        self.skip_string()?;
        serde_json::from_slice::<String>(&self.b[start..self.i])
            .map_err(|e| parse_err(format!("string decode at {start}: {e}")))
    }

    fn skip_number(&mut self) -> Result<()> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        let mut saw_digit = false;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            saw_digit = true;
            self.i += 1;
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                saw_digit = true;
                self.i += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                saw_digit = true;
                self.i += 1;
            }
        }
        if !saw_digit || self.i == start {
            return Err(parse_err(format!("invalid number at byte {start}")));
        }
        Ok(())
    }

    fn skip_value(&mut self) -> Result<(usize, usize)> {
        self.skip_ws();
        let start = self.i;
        match self.peek() {
            Some(b'{') => self.skip_object()?,
            Some(b'[') => self.skip_array()?,
            Some(b'"') => self.skip_string()?,
            Some(b't') => self.eat_lit(b"true")?,
            Some(b'f') => self.eat_lit(b"false")?,
            Some(b'n') => self.eat_lit(b"null")?,
            Some(b'-') | Some(b'0'..=b'9') => self.skip_number()?,
            other => {
                return Err(parse_err(format!(
                    "unexpected value start at byte {}: {:?}",
                    start, other
                )))
            }
        }
        Ok((start, self.i))
    }

    fn skip_object(&mut self) -> Result<()> {
        self.eat(b'{')?;
        loop {
            self.skip_ws();
            if self.peek() == Some(b'}') {
                self.i += 1;
                return Ok(());
            }
            self.skip_string()?;
            self.skip_ws();
            self.eat(b':')?;
            self.skip_value()?;
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(());
                }
                other => {
                    return Err(parse_err(format!(
                        "expected ',' or '}}' in object at byte {}, got {:?}",
                        self.i, other
                    )))
                }
            }
        }
    }

    fn skip_array(&mut self) -> Result<()> {
        self.eat(b'[')?;
        loop {
            self.skip_ws();
            if self.peek() == Some(b']') {
                self.i += 1;
                return Ok(());
            }
            self.skip_value()?;
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(());
                }
                other => {
                    return Err(parse_err(format!(
                        "expected ',' or ']' in array at byte {}, got {:?}",
                        self.i, other
                    )))
                }
            }
        }
    }
}

/// Compare sidecar `get_json` against a full `serde_json` parse for every key
/// in `map`. Returns `(n_equal, mismatches)` where mismatches are keys whose
/// parsed values differ. Used as the in-crate parity gate.
pub fn parity_map(
    index: &ArtifactIndex,
    source_bytes: &[u8],
    map: &str,
) -> Result<(usize, Vec<String>)> {
    let full: serde_json::Value = serde_json::from_slice(source_bytes)?;
    let Some(obj) = full.get(map).and_then(|v| v.as_object()) else {
        return Err(HideError::NotFound(format!("top-level map {map}")));
    };
    let mut mismatches = Vec::new();
    let mut n_equal = 0usize;
    for (k, v) in obj {
        match index.get_json(map, k)? {
            None => mismatches.push(k.clone()),
            Some(raw) => {
                let got: serde_json::Value = serde_json::from_str(&raw)?;
                if &got == v {
                    n_equal += 1;
                } else {
                    mismatches.push(k.clone());
                }
            }
        }
    }
    let idx_keys = index.keys(map)?;
    if idx_keys.len() != obj.len() {
        for k in &idx_keys {
            if !obj.contains_key(k) && !mismatches.contains(k) {
                mismatches.push(k.clone());
            }
        }
        for k in obj.keys() {
            if !idx_keys.contains(k) && !mismatches.contains(k) {
                mismatches.push(k.clone());
            }
        }
    }
    Ok((n_equal, mismatches))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_json(dir: &tempfile::TempDir, name: &str, body: &str) -> PathBuf {
        let p = dir.path().join(name);
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    fn sample() -> String {
        // indent=1, sort_keys — same shape as tools.future._common.write_receipt.
        let doc = serde_json::json!({
            "counts": {"modules": 3, "undispositioned": 0},
            "modules": {
                "tools/future/a.py": {
                    "classification": "UNREACHABLE",
                    "disposition": "PARKED",
                    "module": "tools/future/a.py",
                    "summary": "say \"hi\" and a brace {",
                    "wake": {"predicate": "first production importer", "required_kind": "call"}
                },
                "tools/future/b.py": {
                    "classification": "BUILT",
                    "disposition": "CONNECTED",
                    "module": "tools/future/b.py",
                    "status": null
                },
                "tools/accelerator/c.py": {
                    "classification": "DORMANT",
                    "disposition": "PARKED",
                    "module": "tools/accelerator/c.py"
                }
            },
            "schema": "hawking.audit.reachability_triage.v1",
            "version": 1
        });
        let mut s = serde_json::to_string_pretty(&doc).unwrap();
        s.push('\n');
        s
    }

    #[test]
    fn capability_id_strips_tools_prefix() {
        assert_eq!(
            capability_id("tools/future/capacity_inference_rule.py"),
            "future.capacity_inference_rule"
        );
        assert_eq!(
            capability_id("tools/accelerator/c.py"),
            "accelerator.c"
        );
        assert_eq!(capability_id("tools/future/__init__.py"), "future.__init__");
    }

    #[test]
    fn index_get_matches_full_parse() {
        let dir = tempfile::tempdir().unwrap();
        let json = write_json(&dir, "t.json", &sample());
        let db = dir.path().join("t.sqlite");
        let idx = ArtifactIndex::build(&json, &db, &["modules"]).unwrap();
        let bytes = fs::read(&json).unwrap();
        let (n, mismatches) = parity_map(&idx, &bytes, "modules").unwrap();
        assert!(mismatches.is_empty(), "mismatches: {mismatches:?}");
        assert_eq!(n, 3);
        assert_eq!(
            idx.keys_where("modules", "classification", "UNREACHABLE")
                .unwrap(),
            vec!["tools/future/a.py".to_string()]
        );
        assert_eq!(
            idx.keys_where("modules", "disposition", "CONNECTED")
                .unwrap(),
            vec!["tools/future/b.py".to_string()]
        );
        let cap = idx.get_by_cap_id("future.a").unwrap().unwrap();
        assert_eq!(cap.key, "tools/future/a.py");
        // _root still holds scalars that are not maps.
        let schema = idx.get_json("_root", "schema").unwrap().unwrap();
        assert!(schema.contains("reachability_triage"));
        assert!(idx.is_fresh(&json).unwrap());
    }

    #[test]
    fn stale_hash_is_not_fresh() {
        let dir = tempfile::tempdir().unwrap();
        let json = write_json(&dir, "t.json", &sample());
        let db = dir.path().join("t.sqlite");
        let idx = ArtifactIndex::build(&json, &db, &[]).unwrap();
        fs::write(&json, sample().replace("UNREACHABLE", "BUILT")).unwrap();
        assert!(!idx.is_fresh(&json).unwrap());
    }

    #[test]
    fn escaped_quote_inside_value_does_not_end_object() {
        let dir = tempfile::tempdir().unwrap();
        let json = write_json(&dir, "t.json", &sample());
        let db = dir.path().join("t.sqlite");
        let idx = ArtifactIndex::build(&json, &db, &["modules"]).unwrap();
        let raw = idx
            .get_json("modules", "tools/future/a.py")
            .unwrap()
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            v["summary"].as_str().unwrap(),
            "say \"hi\" and a brace {"
        );
    }

    #[test]
    fn auto_detects_object_maps() {
        let dir = tempfile::tempdir().unwrap();
        let body = r#"{
 "gates": {
  "G1": {"id": "G1", "status": "BUILT"},
  "G2": {"id": "G2", "status": "SCAFFOLDED"}
 },
 "schema": "x"
}
"#;
        let json = write_json(&dir, "g.json", body);
        let db = dir.path().join("g.sqlite");
        let idx = ArtifactIndex::build(&json, &db, &[]).unwrap();
        assert_eq!(
            idx.keys_where("gates", "status", "BUILT").unwrap(),
            vec!["G1".to_string()]
        );
        assert_eq!(idx.keys("gates").unwrap().len(), 2);
    }
}
