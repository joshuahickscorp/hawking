//! User controls: inspect, correct, pin, expire, forget, export, disable.
//!
//! BC-SECURITY-011 (forget is real deletion including dangling edges),
//! BC-SECURITY-020 (pin survives expire; disable blocks write),
//! BC-CONTEXT_OS-010 (export portable, no resurrect).

use super::sql::*;
use super::types::*;
use super::ClassedMemorySystem;
use hide_core::error::{HideError, Result};
use hide_core::ids::now_ms;

impl ClassedMemorySystem {
    // ----- eight user controls ---------------------------------------------

    /// **inspect** — every durable record is reachable. No hidden permanent memory.
    pub fn inspect(&self, filter: &InspectFilter) -> Result<Vec<ClassMemoryRecord>> {
        let mut out = Vec::new();
        let classes: Vec<MemoryClass> = match filter.class {
            Some(c) => vec![c],
            None => MemoryClass::all().to_vec(),
        };
        for class in classes {
            if class == MemoryClass::Working && !filter.include_working {
                continue;
            }
            let mut rows = self.list_class(class)?;
            if let Some(scope) = filter.scope {
                rows.retain(|r| r.scope == scope);
            }
            if !filter.include_expired {
                rows.retain(|r| !r.expired);
            }
            out.extend(rows);
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    /// Look up a single record by id across all stores (including working).
    pub fn get(&self, id: &str) -> Result<Option<ClassMemoryRecord>> {
        for class in MemoryClass::all() {
            for r in self.list_class(class)? {
                if r.id == id {
                    return Ok(Some(r));
                }
            }
        }
        Ok(None)
    }

    /// **correct** (user class). Supersession: new record names the old; both
    /// remain until forgotten. Requires [`UserWriteCap`].
    pub fn correct_user(
        &self,
        cap: &UserWriteCap,
        id: &str,
        writer: impl Into<String>,
        new_text: impl Into<String>,
    ) -> Result<ClassMemoryRecord> {
        let old = self
            .get(id)?
            .ok_or_else(|| HideError::NotFound(format!("memory record {id}")))?;
        if old.class != MemoryClass::User {
            return Err(HideError::InvalidState(
                "correct_user requires a user-class record".into(),
            ));
        }
        self.write_user(
            cap,
            writer,
            ClassMemoryDraft::new(new_text.into())
                .with_scope(old.scope)
                .with_importance(old.importance)
                .with_supersedes(id),
        )
    }

    /// **correct** (verification class). Still requires [`VerifierWriteCap`].
    pub fn correct_verification(
        &self,
        cap: &VerifierWriteCap,
        id: &str,
        writer: impl Into<String>,
        new_text: impl Into<String>,
    ) -> Result<ClassMemoryRecord> {
        let old = self
            .get(id)?
            .ok_or_else(|| HideError::NotFound(format!("memory record {id}")))?;
        if old.class != MemoryClass::Verification {
            return Err(HideError::InvalidState(
                "correct_verification requires a verification-class record".into(),
            ));
        }
        self.write_verification(
            cap,
            writer,
            ClassMemoryDraft::new(new_text.into())
                .with_scope(old.scope)
                .with_importance(old.importance)
                .with_evidence(old.provenance.evidence.clone())
                .with_evidence_tier(
                    old.evidence_tier
                        .clone()
                        .unwrap_or_else(|| "asserted".into()),
                )
                .with_supersedes(id),
        )
    }

    /// **correct** (semantic_project). Requires [`ProjectWriteCap`].
    pub fn correct_semantic_project(
        &self,
        cap: &ProjectWriteCap,
        id: &str,
        writer: impl Into<String>,
        new_text: impl Into<String>,
    ) -> Result<ClassMemoryRecord> {
        let old = self
            .get(id)?
            .ok_or_else(|| HideError::NotFound(format!("memory record {id}")))?;
        if old.class != MemoryClass::SemanticProject {
            return Err(HideError::InvalidState(
                "correct_semantic_project requires a semantic_project record".into(),
            ));
        }
        self.write_semantic_project(
            cap,
            writer,
            ClassMemoryDraft::new(new_text.into())
                .with_scope(old.scope)
                .with_importance(old.importance)
                .with_evidence(old.provenance.evidence.clone())
                .with_supersedes(id),
        )
    }

    /// **correct** (procedural). Requires [`ProceduralWriteCap`].
    pub fn correct_procedural(
        &self,
        cap: &ProceduralWriteCap,
        id: &str,
        writer: impl Into<String>,
        new_text: impl Into<String>,
    ) -> Result<ClassMemoryRecord> {
        let old = self
            .get(id)?
            .ok_or_else(|| HideError::NotFound(format!("memory record {id}")))?;
        if old.class != MemoryClass::Procedural {
            return Err(HideError::InvalidState(
                "correct_procedural requires a procedural record".into(),
            ));
        }
        self.write_procedural(
            cap,
            writer,
            ClassMemoryDraft::new(new_text.into())
                .with_scope(old.scope)
                .with_importance(old.importance)
                .with_evidence(old.provenance.evidence.clone())
                .with_supersedes(id),
        )
    }

    /// **correct** (episodic). Requires [`EpisodicWriteCap`].
    pub fn correct_episodic(
        &self,
        cap: &EpisodicWriteCap,
        id: &str,
        writer: impl Into<String>,
        new_text: impl Into<String>,
    ) -> Result<ClassMemoryRecord> {
        let old = self
            .get(id)?
            .ok_or_else(|| HideError::NotFound(format!("memory record {id}")))?;
        if old.class != MemoryClass::Episodic {
            return Err(HideError::InvalidState(
                "correct_episodic requires an episodic record".into(),
            ));
        }
        let mut draft = ClassMemoryDraft::new(new_text.into())
            .with_scope(old.scope)
            .with_importance(old.importance)
            .with_supersedes(id);
        if let Some(sid) = old.session_id {
            draft = draft.with_session(sid);
        }
        self.write_episodic(cap, writer, draft)
    }

    /// **pin** — pinned records are exempt from expiry, never from forget.
    pub fn pin(&self, id: &str, pinned: bool) -> Result<()> {
        let rec = self
            .get(id)?
            .ok_or_else(|| HideError::NotFound(format!("memory record {id}")))?;
        self.update_record_flags(&rec, Some(pinned), None)
    }

    /// **scope** — explicit, recorded scope transition. Connector content cannot
    /// reach global any other way.
    pub fn set_scope(
        &self,
        id: &str,
        to_scope: PersonalScope,
        approved_by: impl Into<String>,
    ) -> Result<ScopePromotion> {
        let approved_by = approved_by.into();
        if approved_by.is_empty() {
            return Err(HideError::InvalidState(
                "scope promotion requires an explicit approver".into(),
            ));
        }
        let rec = self
            .get(id)?
            .ok_or_else(|| HideError::NotFound(format!("memory record {id}")))?;
        let from = rec.scope;
        if from == to_scope {
            return Ok(ScopePromotion {
                record_id: id.into(),
                from_scope: from,
                to_scope,
                at_ms: now_ms(),
                approved_by,
            });
        }
        self.update_scope_column(&rec, to_scope)?;
        let promo = ScopePromotion {
            record_id: id.into(),
            from_scope: from,
            to_scope,
            at_ms: now_ms(),
            approved_by,
        };
        self.record_promotion(&promo)?;
        Ok(promo)
    }

    /// **expire** — mark due records expired (leave working set). Pinned survive.
    pub fn expire_due(&self, now_ms: u64) -> Result<usize> {
        let mut n = 0usize;
        for class in MemoryClass::all() {
            if class == MemoryClass::Working {
                // Working is turn-local; expire flag is not load-bearing there.
                continue;
            }
            for rec in self.list_class(class)? {
                if rec.expired || rec.pinned {
                    continue;
                }
                if let Some(at) = rec.expire_at_ms {
                    if now_ms >= at {
                        self.update_record_flags(&rec, None, Some(true))?;
                        n += 1;
                    }
                }
            }
        }
        Ok(n)
    }

    /// **forget** — real deletion for user-owned data. Not a tombstone.
    ///
    /// Returns true if a record was removed. Works for every class, including
    /// working (RAM) and durable tables. Also clears dangling edges: scope
    /// promotions for the id, and any `supersedes` pointers that named it.
    /// Export after forget must not reintroduce the forgotten row via residual
    /// audit edges.
    pub fn forget(&self, id: &str) -> Result<bool> {
        let mut removed = false;
        // Working (RAM)
        {
            let mut map = self.working.lock();
            for rows in map.values_mut() {
                let before = rows.len();
                rows.retain(|r| r.id != id);
                if rows.len() < before {
                    removed = true;
                }
                // Clear supersedes edges in RAM that pointed at the forgotten id.
                for r in rows.iter_mut() {
                    if r.supersedes.as_deref() == Some(id) {
                        r.supersedes = None;
                    }
                }
            }
        }
        // Durable workspace tables
        {
            let conn = self.workspace_db.lock();
            for table in [
                "mem_episodic",
                "mem_semantic_project",
                "mem_procedural",
                "mem_verification",
            ] {
                let n = conn
                    .execute(&format!("DELETE FROM {table} WHERE id = ?1"), [id])
                    .map_err(sql_err)?;
                if n > 0 {
                    removed = true;
                }
                // Null supersedes edges that named the forgotten record.
                conn.execute(
                    &format!("UPDATE {table} SET supersedes = NULL WHERE supersedes = ?1"),
                    [id],
                )
                .map_err(sql_err)?;
            }
            // Scope-promotion audit rows for a forgotten record are dangling edges.
            conn.execute(
                "DELETE FROM mem_scope_promotions WHERE record_id = ?1",
                [id],
            )
            .map_err(sql_err)?;
        }
        // User db
        {
            let conn = self.user_db.lock();
            let n = conn
                .execute("DELETE FROM mem_user WHERE id = ?1", [id])
                .map_err(sql_err)?;
            if n > 0 {
                removed = true;
            }
            conn.execute(
                "UPDATE mem_user SET supersedes = NULL WHERE supersedes = ?1",
                [id],
            )
            .map_err(sql_err)?;
        }
        Ok(removed)
    }

    /// Forget every record in a personal scope (real deletion).
    pub fn forget_scope(&self, scope: PersonalScope) -> Result<usize> {
        let ids: Vec<String> = self
            .inspect(&InspectFilter {
                scope: Some(scope),
                include_expired: true,
                include_working: true,
                ..Default::default()
            })?
            .into_iter()
            .map(|r| r.id)
            .collect();
        let mut n = 0usize;
        for id in ids {
            if self.forget(&id)? {
                n += 1;
            }
        }
        Ok(n)
    }

    /// **export** — portable, complete, readable without this tool (JSON).
    pub fn export(&self) -> Result<MemoryExport> {
        let records = self.inspect(&InspectFilter {
            include_expired: true,
            include_working: false,
            ..Default::default()
        })?;
        let promotions = self.list_promotions()?;
        let disabled: Vec<String> = self
            .disabled_classes
            .lock()
            .iter()
            .map(|c| c.as_str().to_string())
            .collect();
        Ok(MemoryExport {
            schema: "hide.you.memory_export.v1".into(),
            exported_at_ms: now_ms(),
            workspace_id: self.workspace_id.clone(),
            records,
            promotions,
            disabled_classes: disabled,
        })
    }

    /// **disable** — user control that blocks new writes and compile retrieval
    /// for a class. Existing records remain inspectable and forgettable.
    pub fn disable_class(&self, class: MemoryClass, disabled: bool) {
        let mut set = self.disabled_classes.lock();
        if disabled {
            set.insert(class);
        } else {
            set.remove(&class);
        }
    }

    pub fn is_class_disabled(&self, class: MemoryClass) -> bool {
        self.disabled_classes.lock().contains(&class)
    }

    pub fn list_promotions(&self) -> Result<Vec<ScopePromotion>> {
        let conn = self.workspace_db.lock();
        let mut stmt = conn
            .prepare(
                "SELECT record_id, from_scope, to_scope, at_ms, approved_by
                 FROM mem_scope_promotions ORDER BY at_ms ASC",
            )
            .map_err(sql_err)?;
        let rows = stmt
            .query_map([], |row| {
                let from_s: String = row.get(1)?;
                let to_s: String = row.get(2)?;
                Ok(ScopePromotion {
                    record_id: row.get(0)?,
                    from_scope: PersonalScope::parse(&from_s).unwrap_or(PersonalScope::Workspace),
                    to_scope: PersonalScope::parse(&to_s).unwrap_or(PersonalScope::Workspace),
                    at_ms: row.get::<_, i64>(3)? as u64,
                    approved_by: row.get(4)?,
                })
            })
            .map_err(sql_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(sql_err)?;
        Ok(rows)
    }

    /// Count durable records that still reference a session id (any class table).
    pub fn durable_refs_to_session(&self, session_id: &str) -> Result<usize> {
        let mut n = 0usize;
        for class in [
            MemoryClass::Episodic,
            MemoryClass::SemanticProject,
            MemoryClass::Procedural,
            MemoryClass::Verification,
        ] {
            n += self
                .list_class(class)?
                .into_iter()
                .filter(|r| r.session_id.as_deref() == Some(session_id))
                .count();
        }
        Ok(n)
    }

    fn record_promotion(&self, promo: &ScopePromotion) -> Result<()> {
        let conn = self.workspace_db.lock();
        conn.execute(
            "INSERT INTO mem_scope_promotions
             (record_id, from_scope, to_scope, at_ms, approved_by)
             VALUES (?1,?2,?3,?4,?5)",
            rusqlite::params![
                promo.record_id,
                promo.from_scope.as_str(),
                promo.to_scope.as_str(),
                promo.at_ms as i64,
                promo.approved_by,
            ],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    fn update_scope_column(&self, rec: &ClassMemoryRecord, to: PersonalScope) -> Result<()> {
        match rec.class {
            MemoryClass::Working => {
                let mut map = self.working.lock();
                for rows in map.values_mut() {
                    for r in rows.iter_mut() {
                        if r.id == rec.id {
                            r.scope = to;
                            return Ok(());
                        }
                    }
                }
                Err(HideError::NotFound(format!("memory record {}", rec.id)))
            }
            MemoryClass::User => {
                let conn = self.user_db.lock();
                let n = conn
                    .execute(
                        "UPDATE mem_user SET scope = ?1 WHERE id = ?2",
                        rusqlite::params![to.as_str(), rec.id],
                    )
                    .map_err(sql_err)?;
                if n == 0 {
                    return Err(HideError::NotFound(format!("memory record {}", rec.id)));
                }
                Ok(())
            }
            other => {
                let table = table_for_class(other)?;
                let conn = self.workspace_db.lock();
                let n = conn
                    .execute(
                        &format!("UPDATE {table} SET scope = ?1 WHERE id = ?2"),
                        rusqlite::params![to.as_str(), rec.id],
                    )
                    .map_err(sql_err)?;
                if n == 0 {
                    return Err(HideError::NotFound(format!("memory record {}", rec.id)));
                }
                Ok(())
            }
        }
    }

    fn update_record_flags(
        &self,
        rec: &ClassMemoryRecord,
        pinned: Option<bool>,
        expired: Option<bool>,
    ) -> Result<()> {
        match rec.class {
            MemoryClass::Working => {
                let mut map = self.working.lock();
                for rows in map.values_mut() {
                    for r in rows.iter_mut() {
                        if r.id == rec.id {
                            if let Some(p) = pinned {
                                r.pinned = p;
                            }
                            if let Some(e) = expired {
                                r.expired = e;
                            }
                            return Ok(());
                        }
                    }
                }
                Err(HideError::NotFound(format!("memory record {}", rec.id)))
            }
            MemoryClass::User => {
                let conn = self.user_db.lock();
                let n = conn
                    .execute(
                        "UPDATE mem_user SET
                           pinned = COALESCE(?1, pinned),
                           expired = COALESCE(?2, expired)
                         WHERE id = ?3",
                        rusqlite::params![
                            pinned.map(|b| b as i64),
                            expired.map(|b| b as i64),
                            rec.id,
                        ],
                    )
                    .map_err(sql_err)?;
                if n == 0 {
                    return Err(HideError::NotFound(format!("memory record {}", rec.id)));
                }
                Ok(())
            }
            other => {
                let table = table_for_class(other)?;
                let conn = self.workspace_db.lock();
                let n = conn
                    .execute(
                        &format!(
                            "UPDATE {table} SET
                               pinned = COALESCE(?1, pinned),
                               expired = COALESCE(?2, expired)
                             WHERE id = ?3"
                        ),
                        rusqlite::params![
                            pinned.map(|b| b as i64),
                            expired.map(|b| b as i64),
                            rec.id,
                        ],
                    )
                    .map_err(sql_err)?;
                if n == 0 {
                    return Err(HideError::NotFound(format!("memory record {}", rec.id)));
                }
                Ok(())
            }
        }
    }
}
