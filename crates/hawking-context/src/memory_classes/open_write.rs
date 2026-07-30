//! Open paths and capability-gated writes / compile retrieval for classed memory.

use super::sql::*;
use super::types::*;
use super::ClassedMemorySystem;
use hide_core::error::{HideError, Result};
use parking_lot::Mutex;
use rusqlite::Connection;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

impl ClassedMemorySystem {
    /// Open (or create) durable stores. `user_db_path` must NOT live under the
    /// workspace so user memory survives workspace teardown and is shared.
    pub fn open(
        workspace_id: impl Into<String>,
        workspace_db_path: impl AsRef<Path>,
        user_db_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let workspace_db_path = workspace_db_path.as_ref().to_path_buf();
        let user_db_path = user_db_path.as_ref().to_path_buf();
        if let Some(p) = workspace_db_path.parent() {
            std::fs::create_dir_all(p)?;
        }
        if let Some(p) = user_db_path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let wconn = Connection::open(&workspace_db_path).map_err(sql_err)?;
        let uconn = Connection::open(&user_db_path).map_err(sql_err)?;
        init_workspace_schema(&wconn)?;
        init_user_schema(&uconn)?;
        Ok(Self {
            workspace_id: workspace_id.into(),
            working: Mutex::new(BTreeMap::new()),
            workspace_db: Mutex::new(wconn),
            user_db: Mutex::new(uconn),
            last_retrieval: Mutex::new(None),
            workspace_db_path: Some(workspace_db_path),
            user_db_path: Some(user_db_path),
            disabled_classes: Mutex::new(BTreeSet::new()),
        })
    }

    /// In-memory durable stores (tests).
    pub fn open_in_memory(workspace_id: impl Into<String>) -> Result<Self> {
        let wconn = Connection::open_in_memory().map_err(sql_err)?;
        let uconn = Connection::open_in_memory().map_err(sql_err)?;
        init_workspace_schema(&wconn)?;
        init_user_schema(&uconn)?;
        Ok(Self {
            workspace_id: workspace_id.into(),
            working: Mutex::new(BTreeMap::new()),
            workspace_db: Mutex::new(wconn),
            user_db: Mutex::new(uconn),
            last_retrieval: Mutex::new(None),
            workspace_db_path: None,
            user_db_path: None,
            disabled_classes: Mutex::new(BTreeSet::new()),
        })
    }

    /// Re-open durable paths after a simulated session restart (tests / recovery).
    pub fn reopen(&self) -> Result<Self> {
        let wpath = self
            .workspace_db_path
            .as_ref()
            .ok_or_else(|| HideError::Storage("no workspace db path to reopen".into()))?;
        let upath = self
            .user_db_path
            .as_ref()
            .ok_or_else(|| HideError::Storage("no user db path to reopen".into()))?;
        Self::open(self.workspace_id.clone(), wpath, upath)
    }

    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }

    // ----- writes (capability-gated) ---------------------------------------

    fn ensure_class_enabled(&self, class: MemoryClass) -> Result<()> {
        if self.disabled_classes.lock().contains(&class) {
            return Err(HideError::PolicyDenied(format!(
                "memory class {} is disabled by user control",
                class.as_str()
            )));
        }
        Ok(())
    }

    fn resolve_scope(class: MemoryClass, draft: &ClassMemoryDraft) -> PersonalScope {
        draft
            .scope
            .unwrap_or_else(|| PersonalScope::default_for_class(class))
    }

    /// Write working (turn-local) memory. Requires [`TurnWriteCap`].
    pub fn write_working(
        &self,
        cap: &TurnWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.ensure_class_enabled(MemoryClass::Working)?;
        let scope = Self::resolve_scope(MemoryClass::Working, &draft);
        let rec = ClassMemoryRecord {
            id: mint_id("working"),
            class: MemoryClass::Working,
            scope,
            text: draft.text,
            importance: draft.importance.clamp(0.0, 1.0),
            workspace_id: Some(self.workspace_id.clone()),
            session_id: draft.session_id,
            provenance: ClassProvenance::stamped(
                writer,
                WriteAuthority::Turn,
                Some(cap.turn_id.clone()),
                draft.run_id,
                draft.evidence,
            ),
            evidence_tier: None,
            pinned: false,
            expired: false,
            expire_at_ms: draft.expire_at_ms,
            supersedes: draft.supersedes,
        };
        self.working
            .lock()
            .entry(cap.turn_id.clone())
            .or_default()
            .push(rec.clone());
        Ok(rec)
    }

    /// Clear working memory for a turn. Working must not outlive its turn.
    pub fn end_turn(&self, turn_id: &str) {
        self.working.lock().remove(turn_id);
    }

    /// Append an episodic record. Requires [`EpisodicWriteCap`].
    pub fn write_episodic(
        &self,
        _cap: &EpisodicWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.ensure_class_enabled(MemoryClass::Episodic)?;
        let scope = Self::resolve_scope(MemoryClass::Episodic, &draft);
        let rec = ClassMemoryRecord {
            id: mint_id("episodic"),
            class: MemoryClass::Episodic,
            scope,
            text: draft.text,
            importance: draft.importance.clamp(0.0, 1.0),
            workspace_id: Some(self.workspace_id.clone()),
            session_id: draft.session_id,
            provenance: ClassProvenance::stamped(
                writer,
                WriteAuthority::EventStream,
                draft.turn_id,
                draft.run_id,
                draft.evidence,
            ),
            evidence_tier: None,
            pinned: false,
            expired: false,
            expire_at_ms: draft.expire_at_ms,
            supersedes: draft.supersedes,
        };
        insert_workspace(&self.workspace_db, "mem_episodic", &rec)?;
        Ok(rec)
    }

    /// Evict all episodic records for a session (session end / GC).
    /// Real deletion — not a tombstone.
    pub fn evict_session(&self, session_id: &str) -> Result<usize> {
        let conn = self.workspace_db.lock();
        let n = conn
            .execute(
                "DELETE FROM mem_episodic WHERE session_id = ?1",
                [session_id],
            )
            .map_err(sql_err)?;
        // Also drop any durable records that still reference this session id
        // in non-episodic tables (should be rare; keeps ephemeral cleanup honest).
        for table in ["mem_semantic_project", "mem_procedural", "mem_verification"] {
            let _ = conn
                .execute(
                    &format!("DELETE FROM {table} WHERE session_id = ?1"),
                    [session_id],
                )
                .map_err(sql_err)?;
        }
        Ok(n)
    }

    /// Cap unbounded session growth: drop the oldest episodic rows for
    /// `session_id` until at most `keep` remain. Ids are ULIDs (time-ordered),
    /// so ascending order is oldest-first. Returns the number of rows deleted.
    pub fn prune_episodic_session(&self, session_id: &str, keep: usize) -> Result<usize> {
        let conn = self.workspace_db.lock();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM mem_episodic WHERE session_id = ?1",
                [session_id],
                |r| r.get(0),
            )
            .map_err(sql_err)?;
        let excess = (count as usize).saturating_sub(keep);
        if excess == 0 {
            return Ok(0);
        }
        // Same-table DELETE needs a nested subquery in SQLite.
        let n = conn
            .execute(
                "DELETE FROM mem_episodic WHERE id IN (
                    SELECT id FROM (
                        SELECT id FROM mem_episodic
                        WHERE session_id = ?1
                        ORDER BY id ASC
                        LIMIT ?2
                    )
                )",
                rusqlite::params![session_id, excess as i64],
            )
            .map_err(sql_err)?;
        Ok(n)
    }

    /// Write semantic_project. Requires [`ProjectWriteCap`].
    pub fn write_semantic_project(
        &self,
        _cap: &ProjectWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.ensure_class_enabled(MemoryClass::SemanticProject)?;
        let scope = Self::resolve_scope(MemoryClass::SemanticProject, &draft);
        let rec = ClassMemoryRecord {
            id: mint_id("sem_proj"),
            class: MemoryClass::SemanticProject,
            scope,
            text: draft.text,
            importance: draft.importance.clamp(0.0, 1.0),
            workspace_id: Some(self.workspace_id.clone()),
            session_id: draft.session_id,
            provenance: ClassProvenance::stamped(
                writer,
                WriteAuthority::ProjectDistill,
                draft.turn_id,
                draft.run_id,
                draft.evidence,
            ),
            evidence_tier: None,
            pinned: false,
            expired: false,
            expire_at_ms: draft.expire_at_ms,
            supersedes: draft.supersedes,
        };
        insert_workspace(&self.workspace_db, "mem_semantic_project", &rec)?;
        Ok(rec)
    }

    /// Write procedural (successful tool recipe). Requires [`ProceduralWriteCap`].
    pub fn write_procedural(
        &self,
        _cap: &ProceduralWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.ensure_class_enabled(MemoryClass::Procedural)?;
        let scope = Self::resolve_scope(MemoryClass::Procedural, &draft);
        let rec = ClassMemoryRecord {
            id: mint_id("procedural"),
            class: MemoryClass::Procedural,
            scope,
            text: draft.text,
            importance: draft.importance.clamp(0.0, 1.0),
            workspace_id: Some(self.workspace_id.clone()),
            session_id: draft.session_id,
            provenance: ClassProvenance::stamped(
                writer,
                WriteAuthority::ToolReceipt,
                draft.turn_id,
                draft.run_id,
                draft.evidence,
            ),
            evidence_tier: None,
            pinned: false,
            expired: false,
            expire_at_ms: draft.expire_at_ms,
            supersedes: draft.supersedes,
        };
        insert_workspace(&self.workspace_db, "mem_procedural", &rec)?;
        Ok(rec)
    }

    /// Write user preference. Requires [`UserWriteCap`] — never distillation.
    /// Records have `workspace_id = None` (cross-workspace).
    pub fn write_user(
        &self,
        _cap: &UserWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.ensure_class_enabled(MemoryClass::User)?;
        let scope = Self::resolve_scope(MemoryClass::User, &draft);
        let rec = ClassMemoryRecord {
            id: mint_id("user"),
            class: MemoryClass::User,
            scope,
            text: draft.text,
            importance: draft.importance.clamp(0.0, 1.0),
            workspace_id: None, // not workspace-scoped
            session_id: None,
            provenance: ClassProvenance::stamped(
                writer,
                WriteAuthority::UserExplicit,
                draft.turn_id,
                draft.run_id,
                draft.evidence,
            ),
            evidence_tier: None,
            pinned: false,
            expired: false,
            expire_at_ms: draft.expire_at_ms,
            supersedes: draft.supersedes,
        };
        insert_user(&self.user_db, &rec)?;
        Ok(rec)
    }

    /// Write verification claim/evidence. Requires [`VerifierWriteCap`].
    /// Authority is always `Verifier` — cannot be forged from the turn path.
    pub fn write_verification(
        &self,
        _cap: &VerifierWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.ensure_class_enabled(MemoryClass::Verification)?;
        let scope = Self::resolve_scope(MemoryClass::Verification, &draft);
        let rec = ClassMemoryRecord {
            id: mint_id("verify"),
            class: MemoryClass::Verification,
            scope,
            text: draft.text,
            importance: draft.importance.clamp(0.0, 1.0),
            workspace_id: Some(self.workspace_id.clone()),
            session_id: draft.session_id,
            provenance: ClassProvenance::stamped(
                writer,
                WriteAuthority::Verifier,
                draft.turn_id,
                draft.run_id,
                draft.evidence,
            ),
            evidence_tier: draft.evidence_tier.or_else(|| Some("asserted".into())),
            pinned: false,
            expired: false,
            expire_at_ms: draft.expire_at_ms,
            supersedes: draft.supersedes,
        };
        insert_workspace(&self.workspace_db, "mem_verification", &rec)?;
        Ok(rec)
    }

    // ----- reads -----------------------------------------------------------

    pub fn list_working(&self, turn_id: &str) -> Vec<ClassMemoryRecord> {
        self.working
            .lock()
            .get(turn_id)
            .cloned()
            .unwrap_or_default()
    }

    pub fn list_class(&self, class: MemoryClass) -> Result<Vec<ClassMemoryRecord>> {
        match class {
            MemoryClass::Working => {
                // All turns' working (rare; prefer list_working).
                let map = self.working.lock();
                Ok(map.values().flatten().cloned().collect())
            }
            MemoryClass::User => list_user(&self.user_db),
            MemoryClass::Episodic => list_workspace(&self.workspace_db, "mem_episodic"),
            MemoryClass::SemanticProject => {
                list_workspace(&self.workspace_db, "mem_semantic_project")
            }
            MemoryClass::Procedural => list_workspace(&self.workspace_db, "mem_procedural"),
            MemoryClass::Verification => list_workspace(&self.workspace_db, "mem_verification"),
        }
    }

    pub fn count(&self, class: MemoryClass) -> Result<usize> {
        Ok(self.list_class(class)?.len())
    }

    /// Retrieve for context compile: each class is asked its own question and
    /// filled under its own token budget. Results are independent — filling
    /// one class does not borrow from another.
    ///
    /// Disabled classes and expired records are excluded. Pinned records still
    /// participate if not expired.
    pub fn retrieve_for_compile(
        &self,
        task: &str,
        turn_id: Option<&str>,
        session_id: Option<&str>,
        budgets: &ClassBudgets,
    ) -> Result<ClassCompileRetrieval> {
        let disabled = self.disabled_classes.lock().clone();
        let mut slices = Vec::with_capacity(6);
        for class in MemoryClass::all() {
            let budget = budgets.for_class(class);
            if disabled.contains(&class) {
                slices.push(ClassRetrievalSlice {
                    class,
                    question: class.retrieval_question().to_string(),
                    budget_tokens: budget,
                    used_tokens: 0,
                    hits: Vec::new(),
                });
                continue;
            }
            let mut candidates = match class {
                MemoryClass::Working => match turn_id {
                    Some(t) => self.list_working(t),
                    None => Vec::new(),
                },
                MemoryClass::Episodic => {
                    let mut all = self.list_class(class)?;
                    if let Some(sid) = session_id {
                        all.retain(|r| r.session_id.as_deref() == Some(sid));
                    }
                    all
                }
                other => self.list_class(other)?,
            };
            candidates.retain(|r| !r.expired);
            let ranked = rank_for_query(task, candidates);
            let (hits, used) = pack_to_budget(ranked, budget);
            slices.push(ClassRetrievalSlice {
                class,
                question: class.retrieval_question().to_string(),
                budget_tokens: budget,
                used_tokens: used,
                hits,
            });
        }
        let retrieval = ClassCompileRetrieval { slices };
        *self.last_retrieval.lock() = Some(retrieval.clone());
        Ok(retrieval)
    }

    pub fn last_retrieval(&self) -> Option<ClassCompileRetrieval> {
        self.last_retrieval.lock().clone()
    }
}
