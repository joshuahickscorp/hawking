//! Hierarchical memory store (bible §4.6, Appendix A.2).
//!
//! Two implementations behind one trait:
//!  - [`InMemoryMemoryStore`] — a RAM `BTreeMap` (kept for tests / siblings).
//!  - [`SqliteMemoryStore`] — the real store at `.hide/memory/memory.db` using
//!    SQLite **FTS5** (keyword) + a stored-vector **cosine** index (relevance),
//!    with the Generative-Agents retrieval score
//!    `α_rec·recency + α_imp·importance + α_rel·relevance`, version chains
//!    (`supersedes`), pins, decay, and provenance/confidence.
//!
//! `MemoryRecord` keeps a stable 8-field public shape (siblings construct it
//! directly); the bible's extended attributes (`pinned`, `version`,
//! `supersedes`, `links`, `decay_half_life_days`, `embedding_ref`) are carried
//! by the store and exposed via [`StoredMemory`] / the typed API.

use crate::embed::{cosine, EmbeddingClient};
use futures::future::BoxFuture;
use hide_core::error::{HideError, Result};
use hide_core::ids::{now_ms, TimestampMs};
use hide_core::types::Provenance;
use parking_lot::{Mutex, RwLock};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// The canonical memory DTO. **Field-stable**: `hawking-research::bridge`
/// constructs this literal, so these eight fields must not change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub kind: MemoryKind,
    pub text: String,
    pub importance: f32,
    pub created_at_ms: TimestampMs,
    pub last_used_at_ms: Option<TimestampMs>,
    pub provenance: Provenance,
    pub tags: Vec<String>,
}

/// The store-managed extended attributes of a memory (bible A.2). Tracked by
/// the store alongside the [`MemoryRecord`] so the DTO stays sibling-stable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryMeta {
    pub embedding_ref: Option<String>,
    pub decay_half_life_days: u32,
    pub links: Vec<String>,
    pub supersedes: Option<String>,
    pub pinned: bool,
    pub version: u32,
    pub access_count: u64,
}

impl MemoryMeta {
    /// Type-dependent defaults: semantic/procedural barely decay; episodic does.
    pub fn defaults_for(kind: MemoryKind) -> Self {
        let half_life = match kind {
            MemoryKind::Working => 1,
            MemoryKind::Episodic => 30,
            MemoryKind::Semantic | MemoryKind::Project => 3650,
            MemoryKind::Procedural => 3650,
        };
        Self {
            embedding_ref: None,
            decay_half_life_days: half_life,
            links: Vec::new(),
            supersedes: None,
            pinned: false,
            version: 1,
            access_count: 0,
        }
    }
}

/// A record plus its store-managed metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredMemory {
    pub record: MemoryRecord,
    pub meta: MemoryMeta,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Working,
    Episodic,
    Semantic,
    Procedural,
    Project,
}

impl MemoryKind {
    /// Stable lowercase wire name for this kind.
    pub fn as_str_public(&self) -> &'static str {
        self.as_str()
    }

    fn as_str(&self) -> &'static str {
        match self {
            MemoryKind::Working => "working",
            MemoryKind::Episodic => "episodic",
            MemoryKind::Semantic => "semantic",
            MemoryKind::Procedural => "procedural",
            MemoryKind::Project => "project",
        }
    }
    fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "working" => MemoryKind::Working,
            "episodic" => MemoryKind::Episodic,
            "semantic" => MemoryKind::Semantic,
            "procedural" => MemoryKind::Procedural,
            "project" => MemoryKind::Project,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryQuery {
    pub text: String,
    pub kinds: Vec<MemoryKind>,
    pub top_k: usize,
}

/// A retrieval result with its blended score and per-signal breakdown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredMemory {
    pub record: MemoryRecord,
    pub meta: MemoryMeta,
    pub score: f32,
    pub recency: f32,
    pub importance: f32,
    pub relevance: f32,
}

/// Back-compat alias: the previous public result type.
pub type RankedMemory = ScoredMemory;

/// The memory store contract (bible A.2). Kept object-safe (`BoxFuture`) so the
/// existing `put`/`query` shape continues to work for siblings; the bible's
/// `retrieve`/`upsert`/`supersede`/`pin` are added.
pub trait MemoryStore: Send + Sync {
    fn put<'a>(&'a self, record: MemoryRecord) -> BoxFuture<'a, Result<()>>;
    fn query<'a>(&'a self, query: MemoryQuery) -> BoxFuture<'a, Result<Vec<RankedMemory>>>;

    /// Generative-Agents retrieval (bible §4.6.3): top-k by
    /// `α_rec·recency + α_imp·importance + α_rel·relevance`, then bump access.
    fn retrieve<'a>(
        &'a self,
        query: &'a str,
        k: usize,
        kinds: &'a [MemoryKind],
    ) -> BoxFuture<'a, Result<Vec<ScoredMemory>>> {
        let q = MemoryQuery {
            text: query.to_string(),
            kinds: kinds.to_vec(),
            top_k: k,
        };
        Box::pin(self.query(q))
    }

    /// Insert/update; creates a new version on id conflict.
    fn upsert<'a>(&'a self, record: MemoryRecord) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            let id = record.id.clone();
            self.put(record).await?;
            Ok(id)
        })
    }

    /// Retire `old` and mint `new` with a `supersedes` edge (default no-op edge).
    fn supersede<'a>(&'a self, _old: &'a str, new: MemoryRecord) -> BoxFuture<'a, Result<String>> {
        self.upsert(new)
    }

    /// Pin/unpin a record (pinned => never decays, always retrievable).
    fn pin<'a>(&'a self, _id: &'a str, _pinned: bool) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

// ---------------------------------------------------------------------------
// In-memory store (kept for tests and siblings)
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct InMemoryMemoryStore {
    records: RwLock<BTreeMap<String, (MemoryRecord, MemoryMeta)>>,
}

impl InMemoryMemoryStore {
    /// Convenience: mint a record with sane defaults (kept for callers).
    pub fn record(
        kind: MemoryKind,
        text: impl Into<String>,
        provenance: Provenance,
    ) -> MemoryRecord {
        MemoryRecord {
            id: format!("mem_{}", now_ms()),
            kind,
            text: text.into(),
            importance: 0.5,
            created_at_ms: now_ms(),
            last_used_at_ms: None,
            provenance,
            tags: Vec::new(),
        }
    }
}

impl MemoryStore for InMemoryMemoryStore {
    fn put<'a>(&'a self, record: MemoryRecord) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let meta = MemoryMeta::defaults_for(record.kind);
            self.records
                .write()
                .insert(record.id.clone(), (record, meta));
            Ok(())
        })
    }

    fn query<'a>(&'a self, query: MemoryQuery) -> BoxFuture<'a, Result<Vec<RankedMemory>>> {
        Box::pin(async move {
            let now = now_ms();
            let mut ranked: Vec<ScoredMemory> = self
                .records
                .read()
                .values()
                .filter(|(r, _)| query.kinds.is_empty() || query.kinds.contains(&r.kind))
                .map(|(r, m)| {
                    let relevance = lexical_overlap(&query.text, &r.text);
                    let recency = recency_score(
                        r.last_used_at_ms.unwrap_or(r.created_at_ms),
                        now,
                        m.decay_half_life_days,
                        m.pinned,
                    );
                    let importance = r.importance.clamp(0.0, 1.0);
                    let score = recency + importance + relevance;
                    ScoredMemory {
                        record: r.clone(),
                        meta: m.clone(),
                        score,
                        recency,
                        importance,
                        relevance,
                    }
                })
                .collect();
            ranked.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked.truncate(query.top_k);
            Ok(ranked)
        })
    }

    fn pin<'a>(&'a self, id: &'a str, pinned: bool) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if let Some((_, m)) = self.records.write().get_mut(id) {
                m.pinned = pinned;
            }
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// SQLite store (FTS5 + stored-vector cosine) — the real store
// ---------------------------------------------------------------------------

/// SQLite-backed memory at `.hide/memory/memory.db` (bible §4.6.1).
///
/// Keyword recall via FTS5; semantic recall via stored embedding vectors with
/// cosine similarity computed in-process (no native ANN dependency — a real
/// lighter alternative per the quality mandate). The connection is mutex-guarded
/// (SQLite is single-writer).
pub struct SqliteMemoryStore {
    conn: Mutex<Connection>,
    embedder: Option<Arc<dyn EmbeddingClient>>,
}

impl SqliteMemoryStore {
    /// Open (creating if needed) the DB at `path`, running the schema migration.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path).map_err(sql_err)?;
        Self::init(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            embedder: None,
        })
    }

    /// Open an in-memory DB (tests).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(sql_err)?;
        Self::init(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            embedder: None,
        })
    }

    /// Attach an embedding client so `relevance` uses cosine over real vectors
    /// (else relevance falls back to FTS5 keyword presence).
    pub fn with_embedder(mut self, embedder: Arc<dyn EmbeddingClient>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    fn init(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS memory (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                text TEXT NOT NULL,
                importance REAL NOT NULL,
                created_at_ms INTEGER NOT NULL,
                last_used_at_ms INTEGER,
                provenance_json TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                embedding_json TEXT,
                decay_half_life_days INTEGER NOT NULL,
                links_json TEXT NOT NULL,
                supersedes TEXT,
                pinned INTEGER NOT NULL,
                version INTEGER NOT NULL,
                access_count INTEGER NOT NULL,
                retired INTEGER NOT NULL DEFAULT 0
            );
            -- FTS5 inverted index over `text`, keyed by `rowid = memory.rowid`
            -- so a `MATCH` joins straight back to the memory row by rowid. A
            -- regular (content-storing) FTS5 table is used — not `content=''` —
            -- because a contentless table cannot return stored columns and makes
            -- row updates awkward; the extra text copy is negligible here.
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING fts5(text);
            "#,
        )
        .map_err(sql_err)?;
        Ok(())
    }

    /// Number of live (non-retired) records.
    pub fn len(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM memory WHERE retired = 0", [], |r| {
                r.get(0)
            })
            .map_err(sql_err)?;
        Ok(n as usize)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    fn insert_record(&self, record: &MemoryRecord, meta: &MemoryMeta) -> Result<()> {
        let conn = self.conn.lock();
        let provenance_json = serde_json::to_string(&record.provenance)?;
        let tags_json = serde_json::to_string(&record.tags)?;
        let links_json = serde_json::to_string(&meta.links)?;
        let embedding_json = meta.embedding_ref.clone();
        conn.execute(
            r#"INSERT INTO memory
               (id, kind, text, importance, created_at_ms, last_used_at_ms,
                provenance_json, tags_json, embedding_json, decay_half_life_days,
                links_json, supersedes, pinned, version, access_count, retired)
               VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,0)
               ON CONFLICT(id) DO UPDATE SET
                 kind=excluded.kind, text=excluded.text, importance=excluded.importance,
                 last_used_at_ms=excluded.last_used_at_ms,
                 provenance_json=excluded.provenance_json, tags_json=excluded.tags_json,
                 embedding_json=excluded.embedding_json,
                 decay_half_life_days=excluded.decay_half_life_days,
                 links_json=excluded.links_json, supersedes=excluded.supersedes,
                 pinned=excluded.pinned, version=memory.version+1, retired=0"#,
            rusqlite::params![
                record.id,
                record.kind.as_str(),
                record.text,
                record.importance as f64,
                record.created_at_ms as i64,
                record.last_used_at_ms.map(|v| v as i64),
                provenance_json,
                tags_json,
                embedding_json,
                meta.decay_half_life_days as i64,
                links_json,
                meta.supersedes,
                meta.pinned as i64,
                meta.version as i64,
                meta.access_count as i64,
            ],
        )
        .map_err(sql_err)?;
        // Mirror into the FTS5 index, keyed by the memory row's rowid so a
        // `MATCH` joins back to `memory` by rowid. On an upsert the row already
        // exists; clear the stale FTS row first, then index the current text.
        let rowid: i64 = conn
            .query_row(
                "SELECT rowid FROM memory WHERE id = ?1",
                [&record.id],
                |r| r.get(0),
            )
            .map_err(sql_err)?;
        conn.execute("DELETE FROM memory_fts WHERE rowid = ?1", [rowid])
            .map_err(sql_err)?;
        conn.execute(
            "INSERT INTO memory_fts(rowid, text) VALUES (?1, ?2)",
            rusqlite::params![rowid, record.text],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn embed_text(&self, text: &str) -> Option<Vec<f32>> {
        match &self.embedder {
            Some(e) => e.embed_one(text).await.ok(),
            None => None,
        }
    }

    fn all_live(&self) -> Result<Vec<StoredMemory>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, kind, text, importance, created_at_ms, last_used_at_ms,
                        provenance_json, tags_json, embedding_json, decay_half_life_days,
                        links_json, supersedes, pinned, version, access_count
                 FROM memory WHERE retired = 0",
            )
            .map_err(sql_err)?;
        let rows = stmt
            .query_map([], row_to_stored)
            .map_err(sql_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(sql_err)?;
        Ok(rows)
    }

    /// Keyword recall via the FTS5 `memory_fts` index. Returns `id -> keyword
    /// relevance in [0,1]` for the rows the FTS5 `MATCH` selects, derived from
    /// the bm25 rank (best row → 1.0, decaying with rank). This is real
    /// inverted-index recall: it ranks by term frequency / document length and
    /// finds rows a naive `text.contains(term)` substring scan would mis-rank.
    ///
    /// `terms` are lowercased, non-empty user tokens. Each is wrapped in an FTS5
    /// string literal (doubling embedded quotes) and OR-combined, so arbitrary
    /// user text can never inject FTS5 query operators.
    fn fts_match(&self, terms: &[String]) -> Result<std::collections::HashMap<String, f32>> {
        use std::collections::HashMap;
        if terms.is_empty() {
            return Ok(HashMap::new());
        }
        // Build `"t1" OR "t2" OR ...` with each term as a quoted FTS5 string.
        let match_query = terms
            .iter()
            .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" OR ");

        let conn = self.conn.lock();
        // Join the FTS rowid back to the live memory row's id.
        let mut stmt = conn
            .prepare(
                "SELECT m.id, bm25(memory_fts) AS rank
                 FROM memory_fts
                 JOIN memory m ON m.rowid = memory_fts.rowid
                 WHERE memory_fts MATCH ?1 AND m.retired = 0
                 ORDER BY rank",
            )
            .map_err(sql_err)?;
        // bm25() returns a score where *more negative* = better match. Map the
        // ordered results to a [0,1] keyword-relevance with the top hit at 1.0.
        let rows: Vec<(String, f64)> = stmt
            .query_map([&match_query], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))
            })
            .map_err(sql_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(sql_err)?;

        let mut out = HashMap::new();
        let n = rows.len();
        for (rank_idx, (id, _bm25)) in rows.into_iter().enumerate() {
            // Rank-decayed relevance: 1.0 for the best, linearly down the list,
            // floored at a small positive so any MATCH still beats a non-match.
            let rel = if n <= 1 {
                1.0
            } else {
                1.0 - 0.5 * (rank_idx as f32) / ((n - 1) as f32)
            };
            out.insert(id, rel);
        }
        Ok(out)
    }

    fn bump_access(&self, id: &str, now: u64) {
        let conn = self.conn.lock();
        let _ = conn.execute(
            "UPDATE memory SET access_count = access_count + 1, last_used_at_ms = ?2 WHERE id = ?1",
            rusqlite::params![id, now as i64],
        );
    }
}

fn row_to_stored(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredMemory> {
    let kind_str: String = row.get(1)?;
    let provenance_json: String = row.get(6)?;
    let tags_json: String = row.get(7)?;
    let embedding_json: Option<String> = row.get(8)?;
    let links_json: String = row.get(10)?;
    let provenance: Provenance =
        serde_json::from_str(&provenance_json).unwrap_or_else(|_| Provenance::trusted("memory"));
    let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
    let links: Vec<String> = serde_json::from_str(&links_json).unwrap_or_default();
    let record = MemoryRecord {
        id: row.get(0)?,
        kind: MemoryKind::from_str(&kind_str).unwrap_or(MemoryKind::Semantic),
        text: row.get(2)?,
        importance: row.get::<_, f64>(3)? as f32,
        created_at_ms: row.get::<_, i64>(4)? as u64,
        last_used_at_ms: row.get::<_, Option<i64>>(5)?.map(|v| v as u64),
        provenance,
        tags,
    };
    let meta = MemoryMeta {
        embedding_ref: embedding_json,
        decay_half_life_days: row.get::<_, i64>(9)? as u32,
        links,
        supersedes: row.get(11)?,
        pinned: row.get::<_, i64>(12)? != 0,
        version: row.get::<_, i64>(13)? as u32,
        access_count: row.get::<_, i64>(14)? as u64,
    };
    Ok(StoredMemory { record, meta })
}

impl MemoryStore for SqliteMemoryStore {
    fn put<'a>(&'a self, record: MemoryRecord) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut meta = MemoryMeta::defaults_for(record.kind);
            if let Some(v) = self.embed_text(&record.text).await {
                meta.embedding_ref = Some(serde_json::to_string(&v)?);
            }
            self.insert_record(&record, &meta)
        })
    }

    fn query<'a>(&'a self, query: MemoryQuery) -> BoxFuture<'a, Result<Vec<RankedMemory>>> {
        Box::pin(async move {
            let now = now_ms();
            let query_vec = self.embed_text(&query.text).await;
            let query_terms: Vec<String> = query
                .text
                .split_whitespace()
                .map(|s| s.to_lowercase())
                .collect();
            // Keyword recall through the FTS5 inverted index (real `MATCH`, not a
            // substring scan): id -> rank-decayed keyword relevance.
            let fts_hits = self.fts_match(&query_terms)?;
            let all = self.all_live()?;

            let mut scored: Vec<ScoredMemory> = all
                .into_iter()
                .filter(|s| query.kinds.is_empty() || query.kinds.contains(&s.record.kind))
                .map(|s| {
                    // Fuse the two recall legs (bible §4.6.1 "FTS5 keyword +
                    // stored-vector cosine"): vector cosine where embeddings
                    // exist, the FTS5 bm25-ranked keyword hit always, and take
                    // the stronger signal so either leg can surface a memory.
                    let keyword = fts_hits.get(&s.record.id).copied().unwrap_or(0.0);
                    let vector = match (&query_vec, &s.meta.embedding_ref) {
                        (Some(qv), Some(ej)) => {
                            let mv: Vec<f32> = serde_json::from_str(ej).unwrap_or_default();
                            ((cosine(qv, &mv) + 1.0) / 2.0).clamp(0.0, 1.0)
                        }
                        _ => 0.0,
                    };
                    let relevance = keyword.max(vector);
                    let recency = recency_score(
                        s.record.last_used_at_ms.unwrap_or(s.record.created_at_ms),
                        now,
                        s.meta.decay_half_life_days,
                        s.meta.pinned,
                    );
                    let importance = s.record.importance.clamp(0.0, 1.0);
                    // Generative-Agents α=1,1,1 blend.
                    let score = recency + importance + relevance;
                    ScoredMemory {
                        record: s.record,
                        meta: s.meta,
                        score,
                        recency,
                        importance,
                        relevance,
                    }
                })
                .collect();

            scored.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.record.id.cmp(&b.record.id))
            });
            scored.truncate(query.top_k);
            // On access, bump access_count + last_used (feeds recency next time).
            for s in &scored {
                self.bump_access(&s.record.id, now);
            }
            Ok(scored)
        })
    }

    fn supersede<'a>(&'a self, old: &'a str, new: MemoryRecord) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            // Retire the old version (hidden from retrieval, kept on disk).
            {
                let conn = self.conn.lock();
                conn.execute("UPDATE memory SET retired = 1 WHERE id = ?1", [old])
                    .map_err(sql_err)?;
            }
            let mut meta = MemoryMeta::defaults_for(new.kind);
            meta.supersedes = Some(old.to_string());
            if let Some(v) = self.embed_text(&new.text).await {
                meta.embedding_ref = Some(serde_json::to_string(&v)?);
            }
            let id = new.id.clone();
            self.insert_record(&new, &meta)?;
            Ok(id)
        })
    }

    fn pin<'a>(&'a self, id: &'a str, pinned: bool) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let conn = self.conn.lock();
            conn.execute(
                "UPDATE memory SET pinned = ?2 WHERE id = ?1",
                rusqlite::params![id, pinned as i64],
            )
            .map_err(sql_err)?;
            Ok(())
        })
    }
}

fn sql_err(e: rusqlite::Error) -> HideError {
    HideError::Storage(format!("memory db: {e}"))
}

/// Exponential recency decay over *days*; pinned records never decay (bible
/// §4.6.4 / §4.7.3).
fn recency_score(ts_ms: u64, now_ms: u64, half_life_days: u32, pinned: bool) -> f32 {
    if pinned {
        return 1.0;
    }
    if half_life_days == 0 {
        return 1.0;
    }
    let age_days = (now_ms.saturating_sub(ts_ms) as f32) / (1000.0 * 60.0 * 60.0 * 24.0);
    0.5f32.powf(age_days / half_life_days as f32)
}

fn lexical_overlap(a: &str, b: &str) -> f32 {
    let a_words: Vec<_> = a.split_whitespace().collect();
    if a_words.is_empty() {
        return 0.0;
    }
    let lb = b.to_lowercase();
    let hits = a_words
        .iter()
        .filter(|word| lb.contains(&word.to_lowercase()))
        .count();
    hits as f32 / a_words.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashingEmbeddingClient;

    fn rec(id: &str, kind: MemoryKind, text: &str, importance: f32) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            kind,
            text: text.to_string(),
            importance,
            created_at_ms: now_ms(),
            last_used_at_ms: None,
            provenance: Provenance::trusted("test"),
            tags: Vec::new(),
        }
    }

    #[tokio::test]
    async fn sqlite_store_retrieves_by_relevance() {
        let store = SqliteMemoryStore::open_in_memory()
            .unwrap()
            .with_embedder(Arc::new(HashingEmbeddingClient::default()));
        store
            .upsert(rec(
                "a",
                MemoryKind::Semantic,
                "the database uses sqlx and postgres",
                0.5,
            ))
            .await
            .unwrap();
        store
            .upsert(rec(
                "b",
                MemoryKind::Semantic,
                "rocket telemetry orbital insertion",
                0.5,
            ))
            .await
            .unwrap();
        let hits = store
            .retrieve("database sqlx", 2, &[MemoryKind::Semantic])
            .await
            .unwrap();
        assert_eq!(
            hits[0].record.id, "a",
            "relevance should rank the db note first"
        );
        assert_eq!(store.len().unwrap(), 2);
    }

    #[tokio::test]
    async fn fts5_match_is_token_aware_not_substring() {
        // No embedder => relevance comes solely from the FTS5 keyword leg.
        let store = SqliteMemoryStore::open_in_memory().unwrap();
        // "cat" is a whole token here.
        store
            .upsert(rec(
                "hit",
                MemoryKind::Semantic,
                "the cat sat on the mat",
                0.1,
            ))
            .await
            .unwrap();
        // "cat" appears only as a *substring* of "concatenate" — a naive
        // `text.contains("cat")` scan would (wrongly) match this row, but the
        // FTS5 inverted index tokenizes and does NOT.
        store
            .upsert(rec(
                "substring_only",
                MemoryKind::Semantic,
                "concatenate adjacent buffers efficiently",
                0.1,
            ))
            .await
            .unwrap();

        let hits = store
            .retrieve("cat", 10, &[MemoryKind::Semantic])
            .await
            .unwrap();

        // The token match gets non-zero relevance.
        let hit = hits
            .iter()
            .find(|h| h.record.id == "hit")
            .expect("token row present");
        assert!(
            hit.relevance > 0.0,
            "FTS5 MATCH must give the token row keyword relevance"
        );
        // The substring-only row gets ZERO relevance from the MATCH path — the
        // proof that retrieval went through FTS5, not a substring scan.
        let sub = hits
            .iter()
            .find(|h| h.record.id == "substring_only")
            .expect("substring row still listed (all_live), but unmatched");
        assert_eq!(
            sub.relevance, 0.0,
            "substring-only row must NOT be matched by FTS5 (a substring scan would)"
        );
        // And the token row therefore outranks the substring-only row.
        assert!(
            hits[0].record.id == "hit",
            "token match should rank first; got {:?}",
            hits.iter().map(|h| &h.record.id).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn supersede_retires_old_and_chains() {
        let store = SqliteMemoryStore::open_in_memory().unwrap();
        store
            .upsert(rec("v1", MemoryKind::Semantic, "old fact", 0.5))
            .await
            .unwrap();
        store
            .supersede("v1", rec("v2", MemoryKind::Semantic, "new fact", 0.5))
            .await
            .unwrap();
        let hits = store.retrieve("fact", 10, &[]).await.unwrap();
        let ids: Vec<_> = hits.iter().map(|h| h.record.id.as_str()).collect();
        assert!(ids.contains(&"v2"));
        assert!(
            !ids.contains(&"v1"),
            "retired version hidden from retrieval"
        );
        assert_eq!(
            hits.iter()
                .find(|h| h.record.id == "v2")
                .unwrap()
                .meta
                .supersedes,
            Some("v1".to_string())
        );
    }

    #[tokio::test]
    async fn pin_keeps_recency_high() {
        let store = SqliteMemoryStore::open_in_memory().unwrap();
        let mut r = rec("p", MemoryKind::Episodic, "pinned thing", 0.1);
        r.created_at_ms = 0; // ancient
        store.upsert(r).await.unwrap();
        store.pin("p", true).await.unwrap();
        let hits = store.retrieve("thing", 1, &[]).await.unwrap();
        assert!((hits[0].recency - 1.0).abs() < 1e-6, "pinned => no decay");
    }

    #[tokio::test]
    async fn in_memory_store_still_works() {
        let store = InMemoryMemoryStore::default();
        store
            .put(rec("x", MemoryKind::Semantic, "hello world", 0.9))
            .await
            .unwrap();
        let hits = store.retrieve("hello", 5, &[]).await.unwrap();
        assert_eq!(hits.len(), 1);
    }
}
