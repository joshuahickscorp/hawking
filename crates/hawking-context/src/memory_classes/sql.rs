//! SQLite schema and row helpers for classed memory durable stores.

use super::types::*;
use crate::budget::estimate_tokens;
use hide_core::error::{HideError, Result};
use parking_lot::Mutex;
use rusqlite::Connection;

pub(super) fn table_for_class(class: MemoryClass) -> Result<&'static str> {
    match class {
        MemoryClass::Episodic => Ok("mem_episodic"),
        MemoryClass::SemanticProject => Ok("mem_semantic_project"),
        MemoryClass::Procedural => Ok("mem_procedural"),
        MemoryClass::Verification => Ok("mem_verification"),
        MemoryClass::Working | MemoryClass::User => Err(HideError::Storage(
            "working/user are not workspace tables".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Ranking / packing helpers
// ---------------------------------------------------------------------------

pub(super) fn rank_for_query(
    task: &str,
    mut records: Vec<ClassMemoryRecord>,
) -> Vec<ClassMemoryRecord> {
    let task_l = task.to_lowercase();
    let terms: Vec<&str> = task_l.split_whitespace().filter(|t| t.len() > 2).collect();
    records.sort_by(|a, b| {
        let sa = score_record(&terms, a);
        let sb = score_record(&terms, b);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    records
}

pub(super) fn score_record(terms: &[&str], r: &ClassMemoryRecord) -> f32 {
    let text = r.text.to_lowercase();
    let overlap = if terms.is_empty() {
        0.5
    } else {
        let hits = terms.iter().filter(|t| text.contains(*t)).count();
        hits as f32 / terms.len() as f32
    };
    overlap + r.importance.clamp(0.0, 1.0)
}

pub(super) fn pack_to_budget(
    ranked: Vec<ClassMemoryRecord>,
    budget: usize,
) -> (Vec<ClassMemoryRecord>, usize) {
    if budget == 0 {
        return (Vec::new(), 0);
    }
    let mut hits = Vec::new();
    let mut used = 0usize;
    for r in ranked {
        let cost = estimate_tokens(&r.text).max(1);
        if used + cost > budget {
            continue;
        }
        used += cost;
        hits.push(r);
    }
    (hits, used)
}

pub(super) fn mint_id(prefix: &str) -> String {
    format!("{prefix}_{}", ulid::Ulid::new())
}

pub(super) fn sql_err(e: rusqlite::Error) -> HideError {
    HideError::Storage(format!("classed memory: {e}"))
}

pub(super) fn init_workspace_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS mem_episodic (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            importance REAL NOT NULL,
            workspace_id TEXT,
            session_id TEXT,
            provenance_json TEXT NOT NULL,
            evidence_tier TEXT,
            scope TEXT NOT NULL DEFAULT 'conversation',
            pinned INTEGER NOT NULL DEFAULT 0,
            expired INTEGER NOT NULL DEFAULT 0,
            expire_at_ms INTEGER,
            supersedes TEXT
        );
        CREATE TABLE IF NOT EXISTS mem_semantic_project (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            importance REAL NOT NULL,
            workspace_id TEXT,
            session_id TEXT,
            provenance_json TEXT NOT NULL,
            evidence_tier TEXT,
            scope TEXT NOT NULL DEFAULT 'workspace',
            pinned INTEGER NOT NULL DEFAULT 0,
            expired INTEGER NOT NULL DEFAULT 0,
            expire_at_ms INTEGER,
            supersedes TEXT
        );
        CREATE TABLE IF NOT EXISTS mem_procedural (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            importance REAL NOT NULL,
            workspace_id TEXT,
            session_id TEXT,
            provenance_json TEXT NOT NULL,
            evidence_tier TEXT,
            scope TEXT NOT NULL DEFAULT 'workspace',
            pinned INTEGER NOT NULL DEFAULT 0,
            expired INTEGER NOT NULL DEFAULT 0,
            expire_at_ms INTEGER,
            supersedes TEXT
        );
        CREATE TABLE IF NOT EXISTS mem_verification (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            importance REAL NOT NULL,
            workspace_id TEXT,
            session_id TEXT,
            provenance_json TEXT NOT NULL,
            evidence_tier TEXT,
            scope TEXT NOT NULL DEFAULT 'workspace',
            pinned INTEGER NOT NULL DEFAULT 0,
            expired INTEGER NOT NULL DEFAULT 0,
            expire_at_ms INTEGER,
            supersedes TEXT
        );
        CREATE TABLE IF NOT EXISTS mem_scope_promotions (
            record_id TEXT NOT NULL,
            from_scope TEXT NOT NULL,
            to_scope TEXT NOT NULL,
            at_ms INTEGER NOT NULL,
            approved_by TEXT NOT NULL
        );
        "#,
    )
    .map_err(sql_err)?;
    // Same class of bug as user.db: CREATE TABLE IF NOT EXISTS never alters an
    // existing workspace file that pre-dates the control columns.
    migrate_add_missing_columns(
        conn,
        "mem_episodic",
        &[
            ("scope", "TEXT NOT NULL DEFAULT 'conversation'"),
            ("pinned", "INTEGER NOT NULL DEFAULT 0"),
            ("expired", "INTEGER NOT NULL DEFAULT 0"),
            ("expire_at_ms", "INTEGER"),
            ("supersedes", "TEXT"),
        ],
    )?;
    for table in ["mem_semantic_project", "mem_procedural", "mem_verification"] {
        migrate_add_missing_columns(
            conn,
            table,
            &[
                ("scope", "TEXT NOT NULL DEFAULT 'workspace'"),
                ("pinned", "INTEGER NOT NULL DEFAULT 0"),
                ("expired", "INTEGER NOT NULL DEFAULT 0"),
                ("expire_at_ms", "INTEGER"),
                ("supersedes", "TEXT"),
            ],
        )?;
    }
    Ok(())
}

pub(super) fn init_user_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS mem_user (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            importance REAL NOT NULL,
            workspace_id TEXT,
            session_id TEXT,
            provenance_json TEXT NOT NULL,
            evidence_tier TEXT,
            scope TEXT NOT NULL DEFAULT 'global',
            pinned INTEGER NOT NULL DEFAULT 0,
            expired INTEGER NOT NULL DEFAULT 0,
            expire_at_ms INTEGER,
            supersedes TEXT
        );
        "#,
    )
    .map_err(sql_err)?;
    migrate_add_missing_columns(
        conn,
        "mem_user",
        &[
            ("scope", "TEXT NOT NULL DEFAULT 'global'"),
            ("pinned", "INTEGER NOT NULL DEFAULT 0"),
            ("expired", "INTEGER NOT NULL DEFAULT 0"),
            ("expire_at_ms", "INTEGER"),
            ("supersedes", "TEXT"),
        ],
    )?;
    Ok(())
}

/// Add columns an older database is missing.
///
/// `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, and the user
/// class is deliberately cross-workspace -- its database lives under the user root and
/// survives every workspace, every test run and every schema change. So a `user.db` written
/// before the scope/retention columns existed stays exactly as it was while the SELECT that
/// reads it grows new columns, and every read fails with "no such column: scope".
///
/// That is what happened here: the file was created by the first six-class lane, the second
/// lane added the columns to CREATE TABLE and to the query, and nothing migrated the file in
/// between. Idempotent, and cheap enough to run on every open.
pub(super) fn migrate_add_missing_columns(
    conn: &Connection,
    table: &str,
    columns: &[(&str, &str)],
) -> Result<()> {
    let mut have: Vec<String> = Vec::new();
    {
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info({table})"))
            .map_err(sql_err)?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(1))
            .map_err(sql_err)?;
        for r in rows {
            have.push(r.map_err(sql_err)?);
        }
    }
    for (name, decl) in columns {
        if !have.iter().any(|h| h == name) {
            conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {name} {decl};"))
                .map_err(sql_err)?;
        }
    }
    Ok(())
}

pub(super) fn insert_workspace(
    conn: &Mutex<Connection>,
    table: &str,
    rec: &ClassMemoryRecord,
) -> Result<()> {
    // Table names are compile-time constants only.
    debug_assert!(matches!(
        table,
        "mem_episodic" | "mem_semantic_project" | "mem_procedural" | "mem_verification"
    ));
    let prov = serde_json::to_string(&rec.provenance)?;
    let sql = format!(
        "INSERT OR REPLACE INTO {table}
         (id, text, importance, workspace_id, session_id, provenance_json, evidence_tier,
          scope, pinned, expired, expire_at_ms, supersedes)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)"
    );
    let conn = conn.lock();
    conn.execute(
        &sql,
        rusqlite::params![
            rec.id,
            rec.text,
            rec.importance as f64,
            rec.workspace_id,
            rec.session_id,
            prov,
            rec.evidence_tier,
            rec.scope.as_str(),
            rec.pinned as i64,
            rec.expired as i64,
            rec.expire_at_ms.map(|v| v as i64),
            rec.supersedes,
        ],
    )
    .map_err(sql_err)?;
    Ok(())
}

pub(super) fn insert_user(conn: &Mutex<Connection>, rec: &ClassMemoryRecord) -> Result<()> {
    let prov = serde_json::to_string(&rec.provenance)?;
    let conn = conn.lock();
    conn.execute(
        "INSERT OR REPLACE INTO mem_user
         (id, text, importance, workspace_id, session_id, provenance_json, evidence_tier,
          scope, pinned, expired, expire_at_ms, supersedes)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
        rusqlite::params![
            rec.id,
            rec.text,
            rec.importance as f64,
            rec.workspace_id, // always None for user
            rec.session_id,
            prov,
            rec.evidence_tier,
            rec.scope.as_str(),
            rec.pinned as i64,
            rec.expired as i64,
            rec.expire_at_ms.map(|v| v as i64),
            rec.supersedes,
        ],
    )
    .map_err(sql_err)?;
    Ok(())
}

pub(super) fn list_workspace(
    conn: &Mutex<Connection>,
    table: &str,
) -> Result<Vec<ClassMemoryRecord>> {
    debug_assert!(matches!(
        table,
        "mem_episodic" | "mem_semantic_project" | "mem_procedural" | "mem_verification"
    ));
    let class = match table {
        "mem_episodic" => MemoryClass::Episodic,
        "mem_semantic_project" => MemoryClass::SemanticProject,
        "mem_procedural" => MemoryClass::Procedural,
        "mem_verification" => MemoryClass::Verification,
        _ => return Err(HideError::Storage(format!("unknown table {table}"))),
    };
    let sql = format!(
        "SELECT id, text, importance, workspace_id, session_id, provenance_json, evidence_tier,
                scope, pinned, expired, expire_at_ms, supersedes
         FROM {table}"
    );
    let conn = conn.lock();
    let mut stmt = conn.prepare(&sql).map_err(sql_err)?;
    let rows = stmt
        .query_map([], |row| row_to_record(row, class))
        .map_err(sql_err)?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(sql_err)?;
    Ok(rows)
}

pub(super) fn list_user(conn: &Mutex<Connection>) -> Result<Vec<ClassMemoryRecord>> {
    let conn = conn.lock();
    let mut stmt = conn
        .prepare(
            "SELECT id, text, importance, workspace_id, session_id, provenance_json, evidence_tier,
                    scope, pinned, expired, expire_at_ms, supersedes
             FROM mem_user",
        )
        .map_err(sql_err)?;
    let rows = stmt
        .query_map([], |row| row_to_record(row, MemoryClass::User))
        .map_err(sql_err)?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(sql_err)?;
    Ok(rows)
}

pub(super) fn row_to_record(
    row: &rusqlite::Row<'_>,
    class: MemoryClass,
) -> rusqlite::Result<ClassMemoryRecord> {
    let provenance_json: String = row.get(5)?;
    let provenance: ClassProvenance =
        serde_json::from_str(&provenance_json).unwrap_or_else(|_| ClassProvenance {
            writer: "unknown".into(),
            written_at_ms: 0,
            turn_id: None,
            run_id: None,
            evidence: Vec::new(),
            authority: WriteAuthority::Turn,
        });
    let scope_s: String = row.get(7)?;
    let scope =
        PersonalScope::parse(&scope_s).unwrap_or_else(|| PersonalScope::default_for_class(class));
    Ok(ClassMemoryRecord {
        id: row.get(0)?,
        class,
        scope,
        text: row.get(1)?,
        importance: row.get::<_, f64>(2)? as f32,
        workspace_id: row.get(3)?,
        session_id: row.get(4)?,
        provenance,
        evidence_tier: row.get(6)?,
        pinned: row.get::<_, i64>(8)? != 0,
        expired: row.get::<_, i64>(9)? != 0,
        expire_at_ms: row.get::<_, Option<i64>>(10)?.map(|v| v as u64),
        supersedes: row.get(11)?,
    })
}
