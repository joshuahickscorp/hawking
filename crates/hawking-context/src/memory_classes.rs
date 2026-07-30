//! Six distinct memory systems for the HIDE Context OS.
//!
//! Split along semantic seams (types, open/write, controls, sql) under
//! `memory_classes/` — not numbered chunk files.
//!
//! Write authority is a **type boundary**: verification writes require
//! [`VerifierWriteCap`] that the turn path never holds.

#[path = "memory_classes/controls.rs"]
mod controls;
#[path = "memory_classes/open_write.rs"]
mod open_write;
#[path = "memory_classes/sql.rs"]
mod sql;
#[path = "memory_classes/types.rs"]
mod types;

pub use types::*;

use parking_lot::Mutex;
use rusqlite::Connection;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

/// The six real memory systems.
///
/// - `working`: RAM only, cleared by [`Self::end_turn`].
/// - `episodic` / `semantic_project` / `procedural` / `verification`: workspace SQLite.
/// - `user`: separate user-scoped SQLite (not under the workspace root).
pub struct ClassedMemorySystem {
    workspace_id: String,
    /// Live turn scratch: turn_id → records.
    working: Mutex<BTreeMap<String, Vec<ClassMemoryRecord>>>,
    /// Workspace-durable classes (four tables).
    workspace_db: Mutex<Connection>,
    /// User preferences (cross-workspace).
    user_db: Mutex<Connection>,
    /// Last compile retrieval (for meter explanations).
    last_retrieval: Mutex<Option<ClassCompileRetrieval>>,
    /// Paths kept for restart tests / diagnostics.
    workspace_db_path: Option<PathBuf>,
    user_db_path: Option<PathBuf>,
    /// Classes the user has disabled: no new writes, excluded from compile retrieval.
    disabled_classes: Mutex<BTreeSet<MemoryClass>>,
}

/// Shared handle type used by backend services and context sources.
pub type DynClassedMemory = Arc<ClassedMemorySystem>;

#[cfg(test)]
mod tests {
    use super::*;

    fn system() -> ClassedMemorySystem {
        ClassedMemorySystem::open_in_memory("ws-test").unwrap()
    }
    #[test]
    fn retention_working_dies_at_turn_end() {
        let sys = system();
        let cap = TurnWriteCap::new("turn-1");
        sys.write_working(
            &cap,
            "kernel",
            ClassMemoryDraft::new("open tool: edit foo.rs"),
        )
        .unwrap();
        assert_eq!(sys.list_working("turn-1").len(), 1);
        sys.end_turn("turn-1");
        assert!(
            sys.list_working("turn-1").is_empty(),
            "working must die at turn end"
        );
    }
    #[test]
    fn retention_episodic_evicted_with_session() {
        let sys = system();
        let cap = EpisodicWriteCap::mint();
        sys.write_episodic(
            &cap,
            "event_stream",
            ClassMemoryDraft::new("tried cargo test, failed")
                .with_session("sess-a")
                .with_turn("t1"),
        )
        .unwrap();
        sys.write_episodic(
            &cap,
            "event_stream",
            ClassMemoryDraft::new("other session note").with_session("sess-b"),
        )
        .unwrap();
        assert_eq!(sys.evict_session("sess-a").unwrap(), 1);
        let left = sys.list_class(MemoryClass::Episodic).unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].session_id.as_deref(), Some("sess-b"));
    }
    #[test]
    fn retention_semantic_project_survives_session_restart() {
        let dir = tempfile::tempdir().unwrap();
        let wdb = dir.path().join("ws.db");
        let udb = dir.path().join("user.db");
        let sys = ClassedMemorySystem::open("ws-restart", &wdb, &udb).unwrap();
        let cap = ProjectWriteCap::mint();
        sys.write_semantic_project(
            &cap,
            "distill",
            ClassMemoryDraft::new("layout: crates/hawking-context owns the compiler")
                .with_evidence(vec!["scan:crates/hawking-context".into()]),
        )
        .unwrap();
        drop(sys);
        let sys2 = ClassedMemorySystem::open("ws-restart", &wdb, &udb).unwrap();
        let hits = sys2.list_class(MemoryClass::SemanticProject).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("hawking-context"));
        assert_eq!(hits[0].workspace_id.as_deref(), Some("ws-restart"));
    }
    #[test]
    fn retention_procedural_survives_session_restart() {
        let dir = tempfile::tempdir().unwrap();
        let wdb = dir.path().join("ws.db");
        let udb = dir.path().join("user.db");
        let sys = ClassedMemorySystem::open("ws-proc", &wdb, &udb).unwrap();
        let cap = ProceduralWriteCap::mint();
        sys.write_procedural(
            &cap,
            "tool_receipt",
            ClassMemoryDraft::new("cargo test -p hawking-context works")
                .with_evidence(vec!["exit_code:0".into()]),
        )
        .unwrap();
        let sys2 = ClassedMemorySystem::open("ws-proc", &wdb, &udb).unwrap();
        assert_eq!(sys2.count(MemoryClass::Procedural).unwrap(), 1);
    }
    #[test]
    fn retention_user_is_not_workspace_scoped() {
        let dir = tempfile::tempdir().unwrap();
        let udb = dir.path().join("user_shared.db");
        let w1 = dir.path().join("ws1.db");
        let w2 = dir.path().join("ws2.db");
        let a = ClassedMemorySystem::open("workspace-a", &w1, &udb).unwrap();
        let cap = UserWriteCap::mint();
        a.write_user(
            &cap,
            "user_intent",
            ClassMemoryDraft::new("prefer concise answers"),
        )
        .unwrap();
        let rec = a.list_class(MemoryClass::User).unwrap();
        assert!(
            rec[0].workspace_id.is_none(),
            "user must not be workspace-scoped"
        );
        let b = ClassedMemorySystem::open("workspace-b", &w2, &udb).unwrap();
        let prefs = b.list_class(MemoryClass::User).unwrap();
        assert_eq!(prefs.len(), 1);
        assert!(prefs[0].text.contains("concise"));
        assert!(prefs[0].workspace_id.is_none());
    }
    #[test]
    fn retention_verification_survives_session_restart() {
        let dir = tempfile::tempdir().unwrap();
        let wdb = dir.path().join("ws.db");
        let udb = dir.path().join("user.db");
        let sys = ClassedMemorySystem::open("ws-v", &wdb, &udb).unwrap();
        let cap = VerifierWriteCap::mint();
        sys.write_verification(
            &cap,
            "verifier",
            ClassMemoryDraft::new("claim: memory classes are separate stores")
                .with_evidence(vec!["test:retention_verification".into()])
                .with_evidence_tier("proven")
                .with_run("run-99"),
        )
        .unwrap();
        let sys2 = ClassedMemorySystem::open("ws-v", &wdb, &udb).unwrap();
        let hits = sys2.list_class(MemoryClass::Verification).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].evidence_tier.as_deref(), Some("proven"));
        assert_eq!(hits[0].provenance.authority, WriteAuthority::Verifier);
    }
    #[test]
    fn write_authority_verification_and_user_require_caps() {
        let sys = system();
        let vcap = VerifierWriteCap::mint();
        let v = sys
            .write_verification(
                &vcap,
                "verifier",
                ClassMemoryDraft::new("proven fact").with_evidence_tier("proven"),
            )
            .unwrap();
        assert_eq!(v.provenance.authority, WriteAuthority::Verifier);
        assert_eq!(v.class, MemoryClass::Verification);
        let ucap = UserWriteCap::mint();
        let u = sys
            .write_user(&ucap, "user_intent", ClassMemoryDraft::new("be terse"))
            .unwrap();
        assert_eq!(u.provenance.authority, WriteAuthority::UserExplicit);
        assert!(u.workspace_id.is_none());
        let tcap = TurnWriteCap::new("t");
        let w = sys
            .write_working(&tcap, "kernel", ClassMemoryDraft::new("scratch"))
            .unwrap();
        assert_eq!(w.provenance.authority, WriteAuthority::Turn);
        assert_ne!(w.provenance.authority, WriteAuthority::Verifier);
        assert_ne!(w.provenance.authority, WriteAuthority::UserExplicit);
        assert_eq!(sys.count(MemoryClass::Verification).unwrap(), 1);
        assert_eq!(sys.count(MemoryClass::User).unwrap(), 1);
        assert_eq!(sys.count(MemoryClass::Working).unwrap(), 1);
    }
    #[test]
    fn retrieve_for_compile_uses_independent_class_budgets() {
        let sys = system();
        let pcap = ProjectWriteCap::mint();
        for i in 0..20 {
            sys.write_semantic_project(
                &pcap,
                "distill",
                ClassMemoryDraft::new(format!(
                    "semantic project fact number {i} about the repository layout and conventions that are durable"
                ))
                .with_importance(0.9),
            )
            .unwrap();
        }
        let vcap = VerifierWriteCap::mint();
        sys.write_verification(
            &vcap,
            "verifier",
            ClassMemoryDraft::new("verification: the six classes are real")
                .with_evidence_tier("proven")
                .with_importance(1.0),
        )
        .unwrap();
        let ucap = UserWriteCap::mint();
        sys.write_user(
            &ucap,
            "user_intent",
            ClassMemoryDraft::new("user prefers short diffs").with_importance(1.0),
        )
        .unwrap();
        let budgets = ClassBudgets {
            working: 0,
            episodic: 0,
            semantic_project: 40, // tight — cannot take all 20
            procedural: 0,
            user: 200,
            verification: 200,
        };
        let ret = sys
            .retrieve_for_compile("repository layout conventions", None, None, &budgets)
            .unwrap();
        let sem = ret.slice(MemoryClass::SemanticProject).unwrap();
        let ver = ret.slice(MemoryClass::Verification).unwrap();
        let user = ret.slice(MemoryClass::User).unwrap();
        assert!(sem.used_tokens <= sem.budget_tokens);
        assert!(
            sem.hits.len() < 20,
            "semantic budget must cap hits, got {}",
            sem.hits.len()
        );
        assert_eq!(ver.hits.len(), 1, "verification must not be starved");
        assert_eq!(user.hits.len(), 1, "user must not be starved");
        assert!(ver.used_tokens <= ver.budget_tokens);
        assert!(user.used_tokens <= user.budget_tokens);
        assert_eq!(ret.slices.len(), 6);
        assert!(ret
            .slices
            .iter()
            .any(|s| s.question.contains("durable facts")));
        assert!(ret
            .slices
            .iter()
            .any(|s| s.question.contains("asserted vs proven")));
    }
    #[test]
    fn provenance_authority_not_forgeable_from_turn_path() {
        let sys = system();
        let tcap = TurnWriteCap::new("turn-x");
        let rec = sys
            .write_working(
                &tcap,
                "kernel.turn",
                ClassMemoryDraft::new("I am totally a verifier claim")
                    .with_evidence(vec!["forged".into()])
                    .with_evidence_tier("proven")
                    .with_run("fake-run"),
            )
            .unwrap();
        assert_eq!(rec.provenance.authority, WriteAuthority::Turn);
        assert_eq!(rec.provenance.writer, "kernel.turn");
        assert_eq!(rec.provenance.turn_id.as_deref(), Some("turn-x"));
        assert!(rec.provenance.written_at_ms > 0);
        assert_eq!(sys.count(MemoryClass::Verification).unwrap(), 0);
        let v = sys
            .write_verification(
                &VerifierWriteCap::mint(),
                "hide-verify",
                ClassMemoryDraft::new("real claim")
                    .with_evidence(vec!["test:x".into()])
                    .with_run("run-1"),
            )
            .unwrap();
        assert_eq!(v.provenance.authority, WriteAuthority::Verifier);
        assert_eq!(v.provenance.run_id.as_deref(), Some("run-1"));
        assert!(!v.provenance.evidence.is_empty());
    }
    #[test]
    fn property_no_hidden_permanent_memory_inspect_and_forget() {
        // Every durable record is reachable by inspect and removable by forget;
        let sys = system();
        let u = sys
            .write_user(
                &UserWriteCap::mint(),
                "user_intent",
                ClassMemoryDraft::new("prefer dark mode"),
            )
            .unwrap();
        let v = sys
            .write_verification(
                &VerifierWriteCap::mint(),
                "verifier",
                ClassMemoryDraft::new("claim proven").with_evidence_tier("proven"),
            )
            .unwrap();
        let p = sys
            .write_semantic_project(
                &ProjectWriteCap::mint(),
                "distill",
                ClassMemoryDraft::new("crate layout fact"),
            )
            .unwrap();
        let all = sys
            .inspect(&InspectFilter {
                include_expired: true,
                ..Default::default()
            })
            .unwrap();
        let ids: Vec<_> = all.iter().map(|r| r.id.as_str()).collect();
        assert!(
            ids.contains(&u.id.as_str()),
            "user record must be inspectable"
        );
        assert!(
            ids.contains(&v.id.as_str()),
            "verification must be inspectable"
        );
        assert!(ids.contains(&p.id.as_str()), "semantic must be inspectable");
        assert!(sys.forget(&u.id).unwrap());
        assert!(
            sys.get(&u.id).unwrap().is_none(),
            "forget must really delete"
        );
        assert_eq!(sys.count(MemoryClass::User).unwrap(), 0);
        let exp = sys.export().unwrap();
        assert_eq!(exp.schema, "hide.you.memory_export.v1");
        let json = serde_json::to_string_pretty(&exp).unwrap();
        assert!(json.contains("claim proven") || json.contains(&v.id));
        let round: MemoryExport = serde_json::from_str(&json).unwrap();
        assert_eq!(round.records.len(), exp.records.len());
    }
    #[test]
    fn property_correct_verification_requires_verifier_cap() {
        let sys = system();
        let rec = sys
            .write_verification(
                &VerifierWriteCap::mint(),
                "verifier",
                ClassMemoryDraft::new("old claim").with_evidence_tier("asserted"),
            )
            .unwrap();
        let corrected = sys
            .correct_verification(
                &VerifierWriteCap::mint(),
                &rec.id,
                "verifier",
                "corrected claim",
            )
            .unwrap();
        assert_eq!(corrected.provenance.authority, WriteAuthority::Verifier);
        assert_eq!(corrected.supersedes.as_deref(), Some(rec.id.as_str()));
        assert_eq!(sys.count(MemoryClass::Verification).unwrap(), 2);
        assert!(sys.get(&rec.id).unwrap().is_some());
    }
    #[test]
    fn property_scope_orthogonal_to_class_and_promotion_recorded() {
        let sys = system();
        let rec = sys
            .write_episodic(
                &EpisodicWriteCap::mint(),
                "event_stream",
                ClassMemoryDraft::new("email snippet")
                    .with_scope(PersonalScope::Connector)
                    .with_session("s1"),
            )
            .unwrap();
        assert_eq!(rec.class, MemoryClass::Episodic);
        assert_eq!(rec.scope, PersonalScope::Connector);
        let global = sys
            .inspect(&InspectFilter {
                scope: Some(PersonalScope::Global),
                include_expired: true,
                ..Default::default()
            })
            .unwrap();
        assert!(global.is_empty());
        let promo = sys
            .set_scope(&rec.id, PersonalScope::Global, "user")
            .unwrap();
        assert_eq!(promo.from_scope, PersonalScope::Connector);
        assert_eq!(promo.to_scope, PersonalScope::Global);
        assert_eq!(promo.approved_by, "user");
        let after = sys.get(&rec.id).unwrap().unwrap();
        assert_eq!(after.scope, PersonalScope::Global);
        assert_eq!(after.class, MemoryClass::Episodic); // class unchanged
        assert_eq!(sys.list_promotions().unwrap().len(), 1);
        assert!(sys.set_scope(&rec.id, PersonalScope::Person, "").is_err());
    }
    #[test]
    fn property_pin_expire_disable_controls() {
        let sys = system();
        let a = sys
            .write_user(
                &UserWriteCap::mint(),
                "user",
                ClassMemoryDraft::new("will expire").with_expire_at_ms(100),
            )
            .unwrap();
        let b = sys
            .write_user(
                &UserWriteCap::mint(),
                "user",
                ClassMemoryDraft::new("pinned forever").with_expire_at_ms(100),
            )
            .unwrap();
        sys.pin(&b.id, true).unwrap();
        assert_eq!(sys.expire_due(200).unwrap(), 1);
        let a2 = sys.get(&a.id).unwrap().unwrap();
        let b2 = sys.get(&b.id).unwrap().unwrap();
        assert!(a2.expired);
        assert!(!b2.expired);
        assert!(b2.pinned);
        sys.disable_class(MemoryClass::User, true);
        assert!(sys
            .write_user(
                &UserWriteCap::mint(),
                "user",
                ClassMemoryDraft::new("blocked"),
            )
            .is_err());
        let still = sys
            .inspect(&InspectFilter {
                class: Some(MemoryClass::User),
                include_expired: true,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(still.len(), 2);
        let ret = sys
            .retrieve_for_compile("prefer", None, None, &ClassBudgets::default_small())
            .unwrap();
        assert!(ret.slice(MemoryClass::User).unwrap().hits.is_empty());
    }
    #[test]
    fn property_export_and_real_deletion_path() {
        let sys = system();
        let id = sys
            .write_procedural(
                &ProceduralWriteCap::mint(),
                "tool",
                ClassMemoryDraft::new("cargo test works"),
            )
            .unwrap()
            .id;
        let exp = sys.export().unwrap();
        assert!(exp.records.iter().any(|r| r.id == id));
        assert!(sys.forget(&id).unwrap());
        assert!(!sys.export().unwrap().records.iter().any(|r| r.id == id));
        assert!(sys.list_class(MemoryClass::Procedural).unwrap().is_empty());
    }
    #[test]
    fn property_eight_scopes_closed_vocabulary() {
        assert_eq!(PersonalScope::all().len(), 8);
        for s in PersonalScope::all() {
            assert_eq!(PersonalScope::parse(s.as_str()), Some(s));
        }
        assert!(PersonalScope::parse("ambient_global").is_none());
    }
    #[test]
    fn property_user_db_legacy_schema_migrates_idempotently() {
        let dir = tempfile::tempdir().unwrap();
        let wdb = dir.path().join("ws.db");
        let udb = dir.path().join("user_legacy.db");
        {
            let conn = rusqlite::Connection::open(&udb).unwrap();
            conn.execute_batch(
                r#"
                CREATE TABLE mem_user (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    importance REAL NOT NULL,
                    workspace_id TEXT,
                    session_id TEXT,
                    provenance_json TEXT NOT NULL,
                    evidence_tier TEXT
                );
                INSERT INTO mem_user
                  (id, text, importance, workspace_id, session_id, provenance_json, evidence_tier)
                VALUES
                  ('user_legacy_1', 'prefer terse', 0.9, NULL, NULL,
                   '{"writer":"user","written_at_ms":1,"turn_id":null,"run_id":null,"evidence":[],"authority":"user_explicit"}',
                   NULL);
                "#,
            )
            .unwrap();
            let mut stmt = conn.prepare("PRAGMA table_info(mem_user)").unwrap();
            let cols: Vec<String> = stmt
                .query_map([], |r| r.get::<_, String>(1))
                .unwrap()
                .map(|r| r.unwrap())
                .collect();
            assert!(!cols.iter().any(|c| c == "scope"));
            assert!(!cols.iter().any(|c| c == "pinned"));
        }
        let sys = ClassedMemorySystem::open("ws-mig", &wdb, &udb).unwrap();
        let listed = sys.list_class(MemoryClass::User).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "user_legacy_1");
        assert_eq!(listed[0].text, "prefer terse");
        assert_eq!(listed[0].scope, PersonalScope::Global); // default after migrate
        assert!(!listed[0].pinned);
        sys.pin("user_legacy_1", true).unwrap();
        assert!(sys.get("user_legacy_1").unwrap().unwrap().pinned);
        let sys2 = ClassedMemorySystem::open("ws-mig", &wdb, &udb).unwrap();
        assert_eq!(sys2.count(MemoryClass::User).unwrap(), 1);
        assert!(sys2.get("user_legacy_1").unwrap().unwrap().pinned);
    }
    #[test]
    fn property_forget_clears_dangling_edges_and_export() {
        let sys = system();
        let old = sys
            .write_user(
                &UserWriteCap::mint(),
                "user",
                ClassMemoryDraft::new("old preference"),
            )
            .unwrap();
        let corrected = sys
            .correct_user(&UserWriteCap::mint(), &old.id, "user", "new preference")
            .unwrap();
        assert_eq!(corrected.supersedes.as_deref(), Some(old.id.as_str()));
        let epi = sys
            .write_episodic(
                &EpisodicWriteCap::mint(),
                "event",
                ClassMemoryDraft::new("connector blob")
                    .with_scope(PersonalScope::Connector)
                    .with_session("s-edge"),
            )
            .unwrap();
        sys.set_scope(&epi.id, PersonalScope::Global, "user")
            .unwrap();
        assert_eq!(sys.list_promotions().unwrap().len(), 1);
        assert!(sys.forget(&old.id).unwrap());
        let still = sys.get(&corrected.id).unwrap().unwrap();
        assert!(
            still.supersedes.is_none(),
            "supersedes edge must not dangle after forget"
        );
        assert!(sys.forget(&epi.id).unwrap());
        assert!(sys.list_promotions().unwrap().is_empty());
        let exp = sys.export().unwrap();
        assert!(!exp.records.iter().any(|r| r.id == old.id));
        assert!(!exp.records.iter().any(|r| r.id == epi.id));
        assert!(!exp.promotions.iter().any(|p| p.record_id == epi.id));
        assert!(exp.records.iter().any(|r| r.id == corrected.id));
        let json = serde_json::to_string(&exp).unwrap();
        assert!(!json.contains(&old.id));
        assert!(!json.contains("connector blob"));
    }
    #[test]
    fn property_export_carries_no_capability_and_no_reimport_path() {
        let sys = system();
        let id = sys
            .write_user(
                &UserWriteCap::mint(),
                "user",
                ClassMemoryDraft::new("secret preference xyz"),
            )
            .unwrap()
            .id;
        let exp = sys.export().unwrap();
        let json = serde_json::to_string(&exp).unwrap();
        assert!(!json.contains("\"tools\""));
        assert!(!json.contains("\"connectors\""));
        assert!(!json.contains("JobCapability"));
        assert!(!json.contains("SurfaceCapability"));
        assert_eq!(exp.schema, "hide.you.memory_export.v1");
        assert!(sys.forget(&id).unwrap());
        let after = sys.export().unwrap();
        let after_json = serde_json::to_string(&after).unwrap();
        assert!(!after_json.contains("secret preference xyz"));
        assert!(!after.records.iter().any(|r| r.id == id));
        let stale: MemoryExport = serde_json::from_str(&json).unwrap();
        assert!(stale.records.iter().any(|r| r.id == id));
        assert!(sys.get(&id).unwrap().is_none());
        assert_eq!(sys.count(MemoryClass::User).unwrap(), 0);
    }
}
