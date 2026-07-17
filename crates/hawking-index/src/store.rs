//! Durable index store: rusqlite (bundled) + FTS5 (bible §4.10).
//!
//! The relational source of truth. Holds the SCIP-shaped symbol/occurrence model,
//! the unified graph edges WITH materialized reverse edges (so "who calls X?" is
//! an index seek, not a scan), an FTS5 lexical index (BM25 over body+ident), and
//! a generation/MVCC anchor. One writer (the daemon), many readers.

use crate::graph::{EdgeKind, Occurrence, Symbol};
use crate::parse::CodeChunk;
use hide_core::types::TextRange;
use hide_core::{HideError, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use std::sync::Arc;

/// A durable, SQLite-backed index store.
pub struct SqliteStore {
    conn: Arc<parking_lot::Mutex<Connection>>,
}

fn map_err(e: rusqlite::Error) -> HideError {
    HideError::Storage(e.to_string())
}

impl SqliteStore {
    /// Open (or create) a store at `path`. Use `:memory:` for tests.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path).map_err(map_err)?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(map_err)?;
        Self::init(conn)
    }

    fn init(conn: Connection) -> Result<Self> {
        // PRAGMAs per §4.10. `synchronous=NORMAL` + WAL is the IDE sweet spot.
        // (WAL on :memory: is a no-op but harmless.)
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA busy_timeout=5000;
             PRAGMA foreign_keys=ON;
             PRAGMA temp_store=MEMORY;",
        )
        .map_err(map_err)?;
        Self::create_schema(&conn)?;
        Ok(Self {
            conn: Arc::new(parking_lot::Mutex::new(conn)),
        })
    }

    fn create_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS file (
                file_id      INTEGER PRIMARY KEY,
                rel_path     TEXT NOT NULL UNIQUE,
                lang         TEXT,
                content_hash TEXT NOT NULL,
                parse_state  TEXT NOT NULL DEFAULT 'ok',
                generation   INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS symbol (
                symbol_id    TEXT PRIMARY KEY,
                kind         TEXT NOT NULL,
                display_name TEXT,
                file         TEXT,
                generation   INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS symbol_name ON symbol(display_name);

            CREATE TABLE IF NOT EXISTS occurrence (
                occ_id     INTEGER PRIMARY KEY,
                symbol_id  TEXT NOT NULL,
                file       TEXT NOT NULL,
                start_line INTEGER, start_col INTEGER,
                end_line   INTEGER, end_col INTEGER,
                role       TEXT NOT NULL,
                generation INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS occ_symbol ON occurrence(symbol_id);
            CREATE INDEX IF NOT EXISTS occ_role ON occurrence(symbol_id, role);
            CREATE INDEX IF NOT EXISTS occ_file ON occurrence(file);

            -- Unified graph with MATERIALIZED reverse edges.
            CREATE TABLE IF NOT EXISTS edge (
                src        TEXT NOT NULL,
                dst        TEXT NOT NULL,
                kind       TEXT NOT NULL,
                weight     REAL NOT NULL DEFAULT 1.0,
                generation INTEGER NOT NULL,
                PRIMARY KEY (src, kind, dst)
            );
            CREATE INDEX IF NOT EXISTS edge_fwd ON edge(src, kind);
            CREATE INDEX IF NOT EXISTS edge_rev ON edge(dst, kind);

            CREATE TABLE IF NOT EXISTS chunk (
                chunk_id   TEXT PRIMARY KEY,
                file       TEXT NOT NULL,
                symbol_id  TEXT,
                start_byte INTEGER, end_byte INTEGER,
                start_line INTEGER, end_line INTEGER,
                embed_model_id TEXT,           -- NULL = pending embedding
                generation INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunk_file ON chunk(file);

            -- Vectors stored alongside chunks (cosine over stored f32, no ANN dep).
            CREATE TABLE IF NOT EXISTS vector (
                chunk_id   TEXT PRIMARY KEY REFERENCES chunk(chunk_id),
                model_id   TEXT NOT NULL,
                dim        INTEGER NOT NULL,
                data       BLOB NOT NULL       -- little-endian f32 array
            );

            CREATE TABLE IF NOT EXISTS generation (
                generation INTEGER PRIMARY KEY,
                root_hash  TEXT NOT NULL,
                created_ms INTEGER NOT NULL,
                status     TEXT NOT NULL        -- 'committed' | 'in_progress'
            );

            -- FTS5: BM25 over body + identifier columns; ident weighted >> body.
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_body USING fts5(
                body, ident, path UNINDEXED
            );
            "#,
        )
        .map_err(map_err)
    }

    // ---- writes (one writer; batch in a transaction) ----

    /// Upsert a file's parsed facts at a generation, replacing prior rows for that
    /// file. Lexical, symbol, occurrence and chunk layers committed in one txn.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_file(
        &self,
        rel_path: &str,
        lang: &str,
        content_hash: &str,
        parse_state: &str,
        body: &str,
        symbols: &[Symbol],
        occurrences: &[Occurrence],
        chunks: &[CodeChunk],
        generation: u64,
    ) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(map_err)?;

        // Clear prior rows for this file (re-index of a changed file).
        tx.execute("DELETE FROM occurrence WHERE file = ?1", params![rel_path])
            .map_err(map_err)?;
        tx.execute("DELETE FROM symbol WHERE file = ?1", params![rel_path])
            .map_err(map_err)?;
        tx.execute("DELETE FROM chunk WHERE file = ?1", params![rel_path])
            .map_err(map_err)?;
        tx.execute("DELETE FROM fts_body WHERE path = ?1", params![rel_path])
            .map_err(map_err)?;

        tx.execute(
            "INSERT INTO file(rel_path, lang, content_hash, parse_state, generation)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(rel_path) DO UPDATE SET
               lang=excluded.lang, content_hash=excluded.content_hash,
               parse_state=excluded.parse_state, generation=excluded.generation",
            params![rel_path, lang, content_hash, parse_state, generation as i64],
        )
        .map_err(map_err)?;

        for s in symbols {
            tx.execute(
                "INSERT OR REPLACE INTO symbol(symbol_id, kind, display_name, file, generation)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![s.qualified_name, s.kind, s.name, s.file, generation as i64],
            )
            .map_err(map_err)?;
        }

        for o in occurrences {
            let (sl, sc, el, ec) = match &o.range {
                Some(r) => (
                    r.start_line as i64,
                    r.start_col as i64,
                    r.end_line as i64,
                    r.end_col as i64,
                ),
                None => (0, 0, 0, 0),
            };
            tx.execute(
                "INSERT INTO occurrence(symbol_id, file, start_line, start_col, end_line, end_col, role, generation)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![o.symbol, o.file, sl, sc, el, ec, o.role, generation as i64],
            )
            .map_err(map_err)?;
        }

        for c in chunks {
            tx.execute(
                "INSERT OR REPLACE INTO chunk(chunk_id, file, symbol_id, start_byte, end_byte, start_line, end_line, embed_model_id, generation)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, ?8)",
                params![
                    c.chunk_id,
                    c.file,
                    c.symbol,
                    c.start_byte as i64,
                    c.end_byte as i64,
                    c.start_line as i64,
                    c.end_line as i64,
                    generation as i64
                ],
            )
            .map_err(map_err)?;
        }

        // Lexical: index the body and an identifier column (symbol names).
        let ident_blob = symbols
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        tx.execute(
            "INSERT INTO fts_body(body, ident, path) VALUES (?1, ?2, ?3)",
            params![body, ident_blob, rel_path],
        )
        .map_err(map_err)?;

        tx.commit().map_err(map_err)?;
        Ok(())
    }

    /// Remove all rows for a file (deletion path).
    pub fn remove_file(&self, rel_path: &str) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(map_err)?;
        for stmt in [
            "DELETE FROM occurrence WHERE file = ?1",
            "DELETE FROM symbol WHERE file = ?1",
            "DELETE FROM chunk WHERE file = ?1",
            "DELETE FROM fts_body WHERE path = ?1",
            "DELETE FROM file WHERE rel_path = ?1",
        ] {
            tx.execute(stmt, params![rel_path]).map_err(map_err)?;
        }
        tx.commit().map_err(map_err)?;
        Ok(())
    }

    /// Insert/replace a graph edge AND nothing else — the reverse direction is
    /// served by the `edge_rev` index, so a single row is materialized-reverse.
    pub fn add_edge(
        &self,
        src: &str,
        dst: &str,
        kind: EdgeKind,
        weight: f32,
        generation: u64,
    ) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR REPLACE INTO edge(src, dst, kind, weight, generation) VALUES (?1,?2,?3,?4,?5)",
            params![src, dst, edge_kind_str(kind), weight as f64, generation as i64],
        )
        .map_err(map_err)?;
        Ok(())
    }

    pub fn store_vector(&self, chunk_id: &str, model_id: &str, vector: &[f32]) -> Result<()> {
        let conn = self.conn.lock();
        let mut bytes = Vec::with_capacity(vector.len() * 4);
        for v in vector {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        conn.execute(
            "INSERT OR REPLACE INTO vector(chunk_id, model_id, dim, data) VALUES (?1,?2,?3,?4)",
            params![chunk_id, model_id, vector.len() as i64, bytes],
        )
        .map_err(map_err)?;
        conn.execute(
            "UPDATE chunk SET embed_model_id=?1 WHERE chunk_id=?2",
            params![model_id, chunk_id],
        )
        .map_err(map_err)?;
        Ok(())
    }

    // ---- reads ----

    /// Definition occurrences for a symbol — accepts either a SCIP id or a bare
    /// name (resolves name → its definition occurrences by display_name).
    pub fn definitions(&self, symbol: &str) -> Result<Vec<Occurrence>> {
        let conn = self.conn.lock();
        // Direct id match first.
        let mut occs = query_occurrences(&conn, "symbol_id = ?1 AND role = 'definition'", symbol)?;
        if occs.is_empty() {
            // Resolve by bare name: find symbol_ids whose display_name = symbol.
            let ids = symbol_ids_by_name(&conn, symbol)?;
            for id in ids {
                occs.extend(query_occurrences(
                    &conn,
                    "symbol_id = ?1 AND role = 'definition'",
                    &id,
                )?);
            }
        }
        Ok(occs)
    }

    /// Precise definition lookup: ONLY the exact `symbol_id` (no bare-name
    /// fallback). Used by the `Q.precise` query path so a same-named symbol in
    /// another file can't leak in.
    pub fn definitions_exact(&self, symbol: &str) -> Result<Vec<Occurrence>> {
        let conn = self.conn.lock();
        query_occurrences(&conn, "symbol_id = ?1 AND role = 'definition'", symbol)
    }

    /// Precise reference lookup: ONLY occurrences keyed by exactly `symbol`
    /// (references are stored keyed by bare name; no SCIP-suffix stripping).
    pub fn references_exact(&self, symbol: &str) -> Result<Vec<Occurrence>> {
        let conn = self.conn.lock();
        query_occurrences(&conn, "symbol_id = ?1 AND role = 'reference'", symbol)
    }

    /// Reference occurrences for a symbol — by id or bare name.
    pub fn references(&self, symbol: &str) -> Result<Vec<Occurrence>> {
        let conn = self.conn.lock();
        // References are stored keyed by bare name (tier-0); also try id.
        let name = bare_name_of(symbol);
        let mut occs = query_occurrences(&conn, "symbol_id = ?1 AND role = 'reference'", &name)?;
        occs.extend(query_occurrences(
            &conn,
            "symbol_id = ?1 AND role = 'reference'",
            symbol,
        )?);
        Ok(occs)
    }

    /// Lexical search via FTS5 BM25 (ident column weighted >> body).
    pub fn lexical_search(&self, query: &str, limit: usize) -> Result<Vec<LexicalHit>> {
        let conn = self.conn.lock();
        let match_expr = fts_match_expr(query);
        if match_expr.is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = conn
            .prepare(
                "SELECT path, body, bm25(fts_body, 1.0, 5.0) AS rank
                 FROM fts_body WHERE fts_body MATCH ?1
                 ORDER BY rank LIMIT ?2",
            )
            .map_err(map_err)?;
        let rows = stmt
            .query_map(params![match_expr, limit as i64], |row| {
                let path: String = row.get(0)?;
                let body: String = row.get(1)?;
                let rank: f64 = row.get(2)?;
                Ok(LexicalHit { path, body, rank })
            })
            .map_err(map_err)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(map_err)?);
        }
        Ok(out)
    }

    /// Symbols whose display_name matches the query substring (symbol leg).
    pub fn symbol_search(&self, query: &str, limit: usize) -> Result<Vec<Symbol>> {
        let conn = self.conn.lock();
        // Strip LIKE wildcards from the user input (collapsed into one pass).
        let like = format!("%{}%", query.replace(['%', '_'], ""));
        let mut stmt = conn
            .prepare(
                "SELECT symbol_id, kind, display_name, file FROM symbol
                 WHERE display_name LIKE ?1 OR symbol_id LIKE ?1 LIMIT ?2",
            )
            .map_err(map_err)?;
        let rows = stmt
            .query_map(params![like, limit as i64], |row| {
                Ok(Symbol {
                    qualified_name: row.get(0)?,
                    kind: row.get(1)?,
                    name: row.get(2)?,
                    file: row.get(3)?,
                })
            })
            .map_err(map_err)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(map_err)?);
        }
        Ok(out)
    }

    /// Forward edges from `src` of a given kind.
    pub fn out_edges(&self, src: &str, kind: EdgeKind) -> Result<Vec<(String, f32)>> {
        self.edges_dir("src", "dst", src, kind)
    }

    /// Reverse edges into `dst` of a kind ("who calls/depends-on X?") — seek.
    pub fn in_edges(&self, dst: &str, kind: EdgeKind) -> Result<Vec<(String, f32)>> {
        self.edges_dir("dst", "src", dst, kind)
    }

    fn edges_dir(
        &self,
        key_col: &str,
        val_col: &str,
        key: &str,
        kind: EdgeKind,
    ) -> Result<Vec<(String, f32)>> {
        let conn = self.conn.lock();
        let sql = format!("SELECT {val_col}, weight FROM edge WHERE {key_col} = ?1 AND kind = ?2");
        let mut stmt = conn.prepare(&sql).map_err(map_err)?;
        let rows = stmt
            .query_map(params![key, edge_kind_str(kind)], |row| {
                let v: String = row.get(0)?;
                let w: f64 = row.get(1)?;
                Ok((v, w as f32))
            })
            .map_err(map_err)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(map_err)?);
        }
        Ok(out)
    }

    /// All edges (for loading into petgraph).
    pub fn all_edges(&self) -> Result<Vec<(String, String, EdgeKind, f32)>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare("SELECT src, dst, kind, weight FROM edge")
            .map_err(map_err)?;
        let rows = stmt
            .query_map([], |row| {
                let src: String = row.get(0)?;
                let dst: String = row.get(1)?;
                let kind: String = row.get(2)?;
                let w: f64 = row.get(3)?;
                Ok((src, dst, edge_kind_from_str(&kind), w as f32))
            })
            .map_err(map_err)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(map_err)?);
        }
        Ok(out)
    }

    /// Chunks whose vectors are present, for cosine search.
    pub fn all_vectors(&self) -> Result<Vec<StoredVector>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT v.chunk_id, v.model_id, v.dim, v.data, c.file, c.start_line, c.end_line
                 FROM vector v JOIN chunk c ON c.chunk_id = v.chunk_id",
            )
            .map_err(map_err)?;
        let rows = stmt
            .query_map([], |row| {
                let chunk_id: String = row.get(0)?;
                let model_id: String = row.get(1)?;
                let dim: i64 = row.get(2)?;
                let data: Vec<u8> = row.get(3)?;
                let file: String = row.get(4)?;
                let start_line: i64 = row.get(5)?;
                let end_line: i64 = row.get(6)?;
                let vector = bytes_to_f32(&data, dim as usize);
                Ok(StoredVector {
                    chunk_id,
                    model_id,
                    file,
                    start_line: start_line as u32,
                    end_line: end_line as u32,
                    vector,
                })
            })
            .map_err(map_err)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(map_err)?);
        }
        Ok(out)
    }

    /// Chunks lacking a vector (for the embedding daemon to fill).
    pub fn pending_chunks(&self, limit: usize) -> Result<Vec<PendingChunk>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT chunk_id, file, start_byte, end_byte FROM chunk
                 WHERE embed_model_id IS NULL LIMIT ?1",
            )
            .map_err(map_err)?;
        let rows = stmt
            .query_map(params![limit as i64], |row| {
                Ok(PendingChunk {
                    chunk_id: row.get(0)?,
                    file: row.get(1)?,
                    start_byte: row.get::<_, i64>(2)? as usize,
                    end_byte: row.get::<_, i64>(3)? as usize,
                })
            })
            .map_err(map_err)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(map_err)?);
        }
        Ok(out)
    }

    pub fn file_count(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM file", [], |r| r.get(0))
            .map_err(map_err)?;
        Ok(n as usize)
    }

    pub fn file_hash(&self, rel_path: &str) -> Result<Option<String>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT content_hash FROM file WHERE rel_path = ?1",
            params![rel_path],
            |r| r.get(0),
        )
        .optional()
        .map_err(map_err)
    }

    pub fn unparseable_count(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM file WHERE parse_state = 'unparseable'",
                [],
                |r| r.get(0),
            )
            .map_err(map_err)?;
        Ok(n as usize)
    }

    // ---- generation / MVCC ----

    pub fn begin_generation(&self, generation: u64, root_hash: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR REPLACE INTO generation(generation, root_hash, created_ms, status)
             VALUES (?1, ?2, ?3, 'in_progress')",
            params![generation as i64, root_hash, now_ms()],
        )
        .map_err(map_err)?;
        Ok(())
    }

    pub fn commit_generation(&self, generation: u64) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE generation SET status='committed' WHERE generation=?1",
            params![generation as i64],
        )
        .map_err(map_err)?;
        Ok(())
    }

    /// The last committed generation (crash-recovery anchor).
    pub fn last_committed_generation(&self) -> Result<u64> {
        let conn = self.conn.lock();
        let g: Option<i64> = conn
            .query_row(
                "SELECT MAX(generation) FROM generation WHERE status='committed'",
                [],
                |r| r.get(0),
            )
            .optional()
            .map_err(map_err)?
            .flatten();
        Ok(g.unwrap_or(0) as u64)
    }

    /// Truncate any torn (in_progress) generation rows on recovery.
    pub fn recover(&self) -> Result<u64> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM generation WHERE status='in_progress'", [])
            .map_err(map_err)?;
        drop(conn);
        self.last_committed_generation()
    }
}

// ---- helpers ----

#[derive(Debug, Clone)]
pub struct LexicalHit {
    pub path: String,
    pub body: String,
    pub rank: f64,
}

#[derive(Debug, Clone)]
pub struct StoredVector {
    pub chunk_id: String,
    pub model_id: String,
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct PendingChunk {
    pub chunk_id: String,
    pub file: String,
    pub start_byte: usize,
    pub end_byte: usize,
}

fn query_occurrences(conn: &Connection, where_clause: &str, key: &str) -> Result<Vec<Occurrence>> {
    let sql = format!(
        "SELECT symbol_id, file, start_line, start_col, end_line, end_col, role
         FROM occurrence WHERE {where_clause}"
    );
    let mut stmt = conn.prepare(&sql).map_err(map_err)?;
    let rows = stmt
        .query_map(params![key], |row| {
            let symbol: String = row.get(0)?;
            let file: String = row.get(1)?;
            let sl: i64 = row.get(2)?;
            let sc: i64 = row.get(3)?;
            let el: i64 = row.get(4)?;
            let ec: i64 = row.get(5)?;
            let role: String = row.get(6)?;
            let range = if sl == 0 && el == 0 {
                None
            } else {
                Some(TextRange {
                    start_line: sl as u32,
                    start_col: sc as u32,
                    end_line: el as u32,
                    end_col: ec as u32,
                })
            };
            Ok(Occurrence {
                symbol,
                file,
                range,
                role,
            })
        })
        .map_err(map_err)?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(map_err)?);
    }
    Ok(out)
}

fn symbol_ids_by_name(conn: &Connection, name: &str) -> Result<Vec<String>> {
    let mut stmt = conn
        .prepare("SELECT symbol_id FROM symbol WHERE display_name = ?1")
        .map_err(map_err)?;
    let rows = stmt
        .query_map(params![name], |row| row.get::<_, String>(0))
        .map_err(map_err)?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(map_err)?);
    }
    Ok(out)
}

/// Best-effort bare name out of a SCIP id (or pass through if it's already bare).
fn bare_name_of(symbol: &str) -> String {
    // SCIP id: "hawking <lang> <path> <name><suffix>" — take the last token,
    // strip the descriptor suffix.
    if let Some(last) = symbol.rsplit(' ').next() {
        let trimmed = last
            .trim_end_matches("().")
            .trim_end_matches(['#', '.', '!', '/']);
        if trimmed != symbol {
            return trimmed.to_string();
        }
    }
    symbol.to_string()
}

/// Build an FTS5 MATCH expression: tokenize on non-alphanumerics, OR the terms,
/// escape with double-quotes to neutralize FTS operators.
fn fts_match_expr(query: &str) -> String {
    let terms: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{}\"", t.replace('"', "")))
        .collect();
    terms.join(" OR ")
}

fn bytes_to_f32(bytes: &[u8], dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(dim);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

pub(crate) fn edge_kind_str(kind: EdgeKind) -> &'static str {
    match kind {
        EdgeKind::Defines => "defines",
        EdgeKind::References => "references",
        EdgeKind::Calls => "calls",
        EdgeKind::Imports => "imports",
        EdgeKind::Implements => "implements",
        EdgeKind::Tests => "tests",
        EdgeKind::Dataflow => "dataflow",
        EdgeKind::Performance => "performance",
    }
}

pub(crate) fn edge_kind_from_str(s: &str) -> EdgeKind {
    match s {
        "defines" => EdgeKind::Defines,
        "references" => EdgeKind::References,
        "calls" => EdgeKind::Calls,
        "imports" => EdgeKind::Imports,
        "implements" => EdgeKind::Implements,
        "tests" => EdgeKind::Tests,
        "dataflow" => EdgeKind::Dataflow,
        _ => EdgeKind::Performance,
    }
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Legacy config DTOs (kept for serde back-compat with siblings).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct IndexStoreConfig {
    pub sqlite_path: String,
    pub vector_path: String,
    pub cas_path: String,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StoreGeneration {
    pub generation: u64,
    pub manifest_hash: String,
    pub sealed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_source;

    fn seed(store: &SqliteStore, path: &str, src: &str, gen: u64) {
        let out = parse_source(path, src);
        let chunks = crate::parse::chunk_file(path, src);
        store
            .upsert_file(
                path,
                out.lang.map(|l| l.as_str()).unwrap_or("unknown"),
                "hash",
                if out.unparseable { "unparseable" } else { "ok" },
                src,
                &out.symbols,
                &out.occurrences,
                &chunks,
                gen,
            )
            .unwrap();
    }

    #[test]
    fn definitions_and_references_roundtrip() {
        let store = SqliteStore::open_in_memory().unwrap();
        let src = "pub fn helper() {}\npub fn caller() { helper(); }\n";
        seed(&store, "src/m.rs", src, 1);

        let defs = store.definitions("helper").unwrap();
        assert_eq!(defs.len(), 1, "one definition of helper");
        assert_eq!(defs[0].role, "definition");

        let refs = store.references("helper").unwrap();
        assert!(!refs.is_empty(), "references(helper) must be non-empty");
    }

    #[test]
    fn fts_lexical_search_finds_body() {
        let store = SqliteStore::open_in_memory().unwrap();
        seed(&store, "a.rs", "pub fn alpha_widget() { compute(); }", 1);
        seed(&store, "b.rs", "pub fn beta() { unrelated(); }", 1);
        let hits = store.lexical_search("alpha_widget", 10).unwrap();
        assert!(hits.iter().any(|h| h.path == "a.rs"));
        assert!(!hits.iter().any(|h| h.path == "b.rs"));
    }

    #[test]
    fn reverse_edges_are_a_seek() {
        let store = SqliteStore::open_in_memory().unwrap();
        store
            .add_edge("caller", "callee", EdgeKind::Calls, 1.0, 1)
            .unwrap();
        let callers = store.in_edges("callee", EdgeKind::Calls).unwrap();
        assert_eq!(callers.len(), 1);
        assert_eq!(callers[0].0, "caller");
        let callees = store.out_edges("caller", EdgeKind::Calls).unwrap();
        assert_eq!(callees[0].0, "callee");
    }

    #[test]
    fn vectors_store_and_load() {
        let store = SqliteStore::open_in_memory().unwrap();
        seed(&store, "v.rs", "pub fn embed_me() { work(); }", 1);
        let pending = store.pending_chunks(10).unwrap();
        assert!(!pending.is_empty());
        store
            .store_vector(&pending[0].chunk_id, "logits-proxy:test", &[0.1, 0.2, 0.3])
            .unwrap();
        let vecs = store.all_vectors().unwrap();
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].vector, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn generation_commit_and_recovery() {
        let store = SqliteStore::open_in_memory().unwrap();
        store.begin_generation(1, "root1").unwrap();
        store.commit_generation(1).unwrap();
        store.begin_generation(2, "root2").unwrap(); // torn (never committed)
        assert_eq!(store.last_committed_generation().unwrap(), 1);
        assert_eq!(store.recover().unwrap(), 1);
    }

    #[test]
    fn remove_file_clears_rows() {
        let store = SqliteStore::open_in_memory().unwrap();
        seed(&store, "gone.rs", "pub fn x() {}", 1);
        assert_eq!(store.file_count().unwrap(), 1);
        store.remove_file("gone.rs").unwrap();
        assert_eq!(store.file_count().unwrap(), 0);
        assert!(store.definitions("x").unwrap().is_empty());
    }
}
