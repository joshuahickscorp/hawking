//! Rewind / fork / replay algebra over the event-sourced session log (bible sec
//! 15.4; consolidation Trace E). The pure, model-free core that
//! [`crate::host::BackendHost`]'s checkpoint methods drive.
//!
//! Provenance (see docs/hide-impl/consolidation/HIDE_DONOR_PORT_LEDGER.md):
//!
//! * The rewind fold (collapse a contiguous range of post-boundary edits back to
//!   a boundary; before/after snapshot model) is an ADAPTED PORT of grok-build's
//!   `merge_rewind_points_from` (crates/codegen/xai-grok-workspace/src/session/
//!   file_state.rs, commit ba76b0a683fa, Apache-2.0, Copyright 2023-2026
//!   SpaceXAI). The donor reverts by writing back file snapshots; HERE the revert
//!   is expressed as an event-log fold (drop the reverted domain's post-boundary
//!   events), so the same replay path materializes it. Changed from the SpaceXAI
//!   original per Apache-2.0 section 4(b): reshaped onto HIDE's event log, the
//!   git/jj/rootfs domains dropped, and the domain-scoped (conversation / code /
//!   both) targeting added.
//! * The partial-history fork boundary ([`ForkPoint`], `start_ordinal`) is a
//!   CLEAN-ROOM REIMPLEMENTATION of Codex's `subagent_history_start_ordinal`
//!   (codex-rs/thread-store/src/live_thread.rs + types.rs, commit 678157acaa81,
//!   Apache-2.0). Only the boundary rule (count the inherited prefix, mark the
//!   next ordinal as the child's first own record) is reproduced from the depth
//!   map; no donor code is copied.
//! * The invalidated-receipt scoping reuses `hide_verify::paths_intersect`, the
//!   same containment-aware primitive `run_static_analysis` uses.

use hide_core::event::Event;
use hide_core::ids::{EventId, SessionId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The durable kind of the fork-boundary marker event (the child's first record).
pub const FORK_POINT_KIND: &str = "fork.point";

/// Which state domain(s) a rewind reverts back to the checkpoint boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewindTarget {
    /// Revert conversation (user/assistant/token) records after the boundary;
    /// preserve code edits.
    Conversation,
    /// Revert code edits after the boundary; preserve the conversation.
    Code,
    /// Revert everything after the boundary (a fold back to the checkpoint).
    Both,
}

impl RewindTarget {
    /// Parse a wire label (`conversation` / `code` / `both`, plus friendly
    /// aliases). `None` on an unknown label so the caller can reject it.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "conversation" | "conversation_only" | "chat" | "thread" => Some(Self::Conversation),
            "code" | "code_only" | "repo" | "files" => Some(Self::Code),
            "both" | "all" | "everything" => Some(Self::Both),
            _ => None,
        }
    }

    /// Does a rewind to this target revert (drop after the boundary) the given
    /// event domain? `Both` reverts every domain; a single-domain target reverts
    /// only its own domain (so the other domain's post-boundary records survive).
    fn reverts(self, domain: EventDomain) -> bool {
        match self {
            RewindTarget::Both => true,
            RewindTarget::Code => domain == EventDomain::Code,
            RewindTarget::Conversation => domain == EventDomain::Conversation,
        }
    }
}

/// The state domain an event kind belongs to. Drives which post-boundary events a
/// domain-scoped rewind reverts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventDomain {
    Conversation,
    Code,
    Other,
}

/// Classify an event kind into its rewind domain. Conversation = the transcript
/// items (user/assistant/token/steer/side-chat summary); Code = the file-mutating
/// surface (`diff.*`, `tool.*`, `edit.*`); everything else (verification, plan,
/// goal, job, the fork marker) is `Other` and is never dropped by a
/// single-domain rewind.
pub fn classify(kind: &str) -> EventDomain {
    match kind {
        "user.intent.submit_turn"
        | "agent.message"
        | "token"
        | "token_batch"
        | "session.merge_summary"
        | "turn.steer" => EventDomain::Conversation,
        k if k.starts_with("diff.") || k.starts_with("tool.") || k.starts_with("edit.") => {
            EventDomain::Code
        }
        _ => EventDomain::Other,
    }
}

/// The ordered event list a rewind child re-materializes: the full prefix up to
/// and including `at_seq` (the checkpoint state, inherited context), plus every
/// POST-boundary event whose domain the `target` does NOT revert. A `Both` target
/// reverts every domain, so nothing after the boundary survives (a fold back to
/// the checkpoint). A `Code` target drops post-boundary code edits but keeps the
/// conversation; `Conversation` is the mirror.
pub fn rewind_child_events(events: &[Event], at_seq: u64, target: RewindTarget) -> Vec<&Event> {
    events
        .iter()
        .filter(|e| e.seq <= at_seq || !target.reverts(classify(&e.kind)))
        .collect()
}

/// The number of prefix events at or before `at_seq` (the inherited-context
/// length a fork/rewind/replay child copies; feeds [`ForkPoint::new`]).
pub fn inherited_len(events: &[Event], at_seq: u64) -> usize {
    events.iter().filter(|e| e.seq <= at_seq).count()
}

// --- code state (repo reference + compare) ----------------------------------

/// A minimal read view of a `diff.proposed` payload: the per-hunk file +
/// post-image needed to fold the code state, plus the diff/hunk identity needed to
/// address a hunk for an on-disk revert. Decoupled from `host::DiffProposal` on
/// purpose (no module cycle); extra fields are ignored.
#[derive(Deserialize)]
struct DiffView {
    #[serde(default)]
    diff_id: String,
    #[serde(default)]
    hunks: Vec<HunkView>,
}

#[derive(Deserialize)]
struct HunkView {
    #[serde(default)]
    hunk_id: String,
    file: String,
    #[serde(default)]
    after: String,
}

/// The `(diff_id, hunk_id)` pairs recorded AFTER `after_seq`, in log order: the
/// hunks a code rewind has to revert on disk. Each `diff.proposed` event carries
/// the whole cumulative proposal for its run, so the hunk that event ADDED is the
/// last one in its list. An event without diff/hunk identity contributes nothing.
pub fn post_boundary_hunks(events: &[Event], after_seq: u64) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for e in events {
        if e.seq <= after_seq || e.kind != "diff.proposed" {
            continue;
        }
        let Some(diff) = e.payload_as::<DiffView>() else {
            continue;
        };
        let Some(hunk) = diff.hunks.last() else {
            continue;
        };
        if diff.diff_id.is_empty() || hunk.hunk_id.is_empty() {
            continue;
        }
        out.push((diff.diff_id.clone(), hunk.hunk_id.clone()));
    }
    out
}

/// The code (repo) state reachable by folding every `diff.proposed` event up to
/// `up_to_seq` (or the whole slice when `None`): a map of file -> blake3 of its
/// latest post-image. Last write per file wins in log order. Model-free and
/// deterministic, so two folds of the same prefix are byte-identical.
pub fn code_state(events: &[Event], up_to_seq: Option<u64>) -> BTreeMap<String, String> {
    let mut state = BTreeMap::new();
    for e in events {
        if up_to_seq.is_some_and(|max| e.seq > max) {
            continue;
        }
        if e.kind != "diff.proposed" {
            continue;
        }
        if let Some(diff) = e.payload_as::<DiffView>() {
            for h in diff.hunks {
                state.insert(h.file, blake3::hash(h.after.as_bytes()).to_hex().to_string());
            }
        }
    }
    state
}

/// How a file differs between two code states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeStatus {
    Added,
    Removed,
    Modified,
}

/// One file's difference between a base and a head code state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileChange {
    pub file: String,
    pub status: ChangeStatus,
    pub base_hash: Option<String>,
    pub head_hash: Option<String>,
}

/// Diff two code states (from [`code_state`]) into a deterministic, sorted list
/// of file changes (added / removed / modified). Unchanged files are omitted.
pub fn diff_code_states(
    base: &BTreeMap<String, String>,
    head: &BTreeMap<String, String>,
) -> Vec<FileChange> {
    let mut files: Vec<&String> = base.keys().chain(head.keys()).collect();
    files.sort();
    files.dedup();
    let mut out = Vec::new();
    for f in files {
        match (base.get(f), head.get(f)) {
            (Some(b), Some(h)) if b != h => out.push(FileChange {
                file: f.clone(),
                status: ChangeStatus::Modified,
                base_hash: Some(b.clone()),
                head_hash: Some(h.clone()),
            }),
            (Some(_), Some(_)) => {}
            (None, Some(h)) => out.push(FileChange {
                file: f.clone(),
                status: ChangeStatus::Added,
                base_hash: None,
                head_hash: Some(h.clone()),
            }),
            (Some(b), None) => out.push(FileChange {
                file: f.clone(),
                status: ChangeStatus::Removed,
                base_hash: Some(b.clone()),
                head_hash: None,
            }),
            (None, None) => {}
        }
    }
    out
}

/// The files whose code changed between two states (the paths a code rewind
/// reverts). A thin projection of [`diff_code_states`].
pub fn changed_files(
    base: &BTreeMap<String, String>,
    head: &BTreeMap<String, String>,
) -> Vec<String> {
    diff_code_states(base, head)
        .into_iter()
        .map(|c| c.file)
        .collect()
}

// --- invalidated verification receipts --------------------------------------

/// A verification receipt reference: the event id it was recorded under and the
/// file scope it verified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptScope {
    pub event_id: EventId,
    pub scope: Vec<String>,
}

/// The `verify.result` receipts recorded AFTER `after_seq` (a hide-verify receipt
/// carries a `scope` array; a `hide_kernel` `Verdict`, which shares the kind but
/// has no `scope`, is skipped). These are the candidates a code rewind can
/// invalidate.
pub fn receipt_scopes(events: &[Event], after_seq: u64) -> Vec<ReceiptScope> {
    events
        .iter()
        .filter(|e| e.seq > after_seq && e.kind == "verify.result")
        .filter_map(|e| {
            let scope = e
                .payload
                .get("scope")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect::<Vec<_>>();
            Some(ReceiptScope {
                event_id: e.id.clone(),
                scope,
            })
        })
        .collect()
}

/// The receipts a code rewind INVALIDATES: those whose file scope intersects a
/// reverted file, using `hide_verify::paths_intersect` (the same containment-aware
/// primitive `run_static_analysis` reconciles authority with, so a directory scope
/// intersects a file it contains).
pub fn invalidated_receipts(reverted_files: &[String], receipts: &[ReceiptScope]) -> Vec<EventId> {
    receipts
        .iter()
        .filter(|r| {
            r.scope
                .iter()
                .any(|s| reverted_files.iter().any(|f| hide_verify::paths_intersect(s, f)))
        })
        .map(|r| r.event_id.clone())
        .collect()
}

// --- partial-history fork boundary (Codex clean-room) -----------------------

/// The partial-history fork boundary marker (Codex `subagent_history_start_ordinal`,
/// clean-room). Written as the child's first event so a projection can tell
/// inherited context apart from the child's own records without replaying the
/// parent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkPoint {
    /// The thread this fork copied its inherited prefix from.
    pub parent_thread: SessionId,
    /// The first ordinal (1-based over the child's own non-marker events) that
    /// belongs to the child's OWN history. Every child event at a LOWER ordinal
    /// is inherited context; at or above it is the child's own record.
    pub start_ordinal: u64,
    /// The parent boundary `seq` the inherited prefix was copied up to (inclusive).
    pub at_seq: u64,
}

impl ForkPoint {
    /// The boundary for a child that inherits `inherited` prefix events copied up
    /// to parent seq `at_seq`: its own history starts at ordinal `inherited + 1`.
    pub fn new(parent_thread: SessionId, inherited: usize, at_seq: u64) -> Self {
        Self {
            parent_thread,
            start_ordinal: inherited as u64 + 1,
            at_seq,
        }
    }
}

/// Split a child session's events into its fork boundary, its inherited prefix,
/// and its own records, by reading the leading [`FORK_POINT_KIND`] marker and
/// counting ordinals over the non-marker events. Without a marker there is no
/// inherited context: every event is the session's own.
pub fn split_inherited_own(child_events: &[Event]) -> (Option<ForkPoint>, Vec<&Event>, Vec<&Event>) {
    let fork_point = child_events
        .first()
        .filter(|e| e.kind == FORK_POINT_KIND)
        .and_then(|e| e.payload_as::<ForkPoint>());
    let Some(fp) = fork_point else {
        return (None, Vec::new(), child_events.iter().collect());
    };
    let mut inherited = Vec::new();
    let mut own = Vec::new();
    let mut ordinal = 0u64;
    for e in child_events {
        if e.kind == FORK_POINT_KIND {
            continue; // the marker is metadata, not part of the ordinal space
        }
        ordinal += 1;
        if ordinal < fp.start_ordinal {
            inherited.push(e);
        } else {
            own.push(e);
        }
    }
    (Some(fp), inherited, own)
}

// --- checkpoint coverage references -----------------------------------------

/// A lightweight, model-free reference over a piece of session state: a count and
/// a blake3 digest. The checkpoint carries references (not full snapshots) of
/// each covered domain; the full state is always re-derivable from the sealed
/// event boundary.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRef {
    pub count: u64,
    pub digest: String,
}

impl StateRef {
    /// A reference over a deterministic list of items: their count plus a blake3
    /// digest over the newline-joined items (order-sensitive).
    pub fn of(items: &[String]) -> Self {
        Self {
            count: items.len() as u64,
            digest: blake3::hash(items.join("\n").as_bytes())
                .to_hex()
                .to_string(),
        }
    }

    /// A reference with an explicit count and a digest over arbitrary material
    /// (used when the count is a domain quantity, e.g. plan steps, distinct from
    /// the digested material).
    pub fn counted(count: usize, material: &str) -> Self {
        Self {
            count: count as u64,
            digest: blake3::hash(material.as_bytes()).to_hex().to_string(),
        }
    }
}

/// The goal in force at a checkpoint (a reference, not a historical fold).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoalRef {
    pub goal_id: String,
    pub status: String,
    pub condition: String,
}

/// What a checkpoint covers beyond the event boundary (bible sec 15.4;
/// consolidation Trace E): a repo-state reference, thread + plan + goal state, and
/// artifact references. Each is a lightweight [`StateRef`]/[`GoalRef`], sealed into
/// the checkpoint's integrity digest so a tampered reference is detectable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointCoverage {
    /// Reference over the code (repo) state at the boundary: file count + a digest
    /// over the sorted `file:content-hash` pairs.
    pub repo_state: StateRef,
    /// Reference over the thread (transcript) at the boundary.
    pub thread: StateRef,
    /// Reference over the plan (step count + a digest of the plan) at the boundary.
    pub plan: StateRef,
    /// The goal in force when the checkpoint was created, or `None`.
    #[serde(default)]
    pub goal: Option<GoalRef>,
    /// Reference over the durable artifact ids reachable at the boundary.
    pub artifacts: StateRef,
    /// A live model-state capsule reference. ALWAYS `None`: capturing live model
    /// state is DEFERRED_MODEL_REQUIRED (this plane never loads a model).
    #[serde(default)]
    pub live_state_capsule: Option<String>,
}

impl CheckpointCoverage {
    /// A blake3 digest over the whole coverage (canonical JSON). Folded into the
    /// checkpoint's sealed integrity so tampering any covered reference is caught.
    pub fn digest(&self) -> String {
        let canonical = serde_json::to_string(self).unwrap_or_default();
        blake3::hash(canonical.as_bytes()).to_hex().to_string()
    }
}

/// The durable artifact references reachable by folding events up to `up_to_seq`:
/// the ids of events that carry a durable artifact / process capture (kind
/// `artifact.*`, `process.captured`, or a payload with a `blob` reference). A
/// reference list, empty when the session captured none.
pub fn artifact_refs(events: &[Event], up_to_seq: Option<u64>) -> Vec<String> {
    events
        .iter()
        .filter(|e| !up_to_seq.is_some_and(|m| e.seq > m))
        .filter(|e| {
            e.kind.starts_with("artifact.")
                || e.kind == "process.captured"
                || e.payload.get("blob").is_some()
        })
        .map(|e| e.id.as_str().to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::event::{Event, NewEvent};
    use serde_json::json;

    /// Build a sequenced `Event` from a `NewEvent`-shaped kind + payload for the
    /// pure-function tests (the log assigns seq/id in production; here we set them
    /// directly so classification/fold logic can be exercised without a store).
    fn ev(seq: u64, kind: &str, payload: serde_json::Value) -> Event {
        let session = SessionId::from("s-test");
        Event::new(seq, NewEvent::system(session, kind, payload))
    }

    fn diff_ev(seq: u64, file: &str, after: &str) -> Event {
        ev(
            seq,
            "diff.proposed",
            json!({ "hunks": [ { "file": file, "after": after } ] }),
        )
    }

    #[test]
    fn post_boundary_hunks_addresses_only_the_hunks_after_the_boundary() {
        // A real `diff.proposed` carries the whole cumulative proposal, so the hunk
        // the event ADDED is its last one.
        let cumulative = |seq: u64, n: usize| {
            let hunks: Vec<_> = (0..=n)
                .map(|i| json!({ "hunk_id": format!("d1-h{i}"), "file": "a.rs", "after": "x" }))
                .collect();
            ev(seq, "diff.proposed", json!({ "diff_id": "d1", "hunks": hunks }))
        };
        let events = vec![cumulative(1, 0), cumulative(2, 1), cumulative(3, 2)];
        assert_eq!(
            post_boundary_hunks(&events, 1),
            vec![
                ("d1".to_string(), "d1-h1".to_string()),
                ("d1".to_string(), "d1-h2".to_string())
            ],
            "only the hunks recorded after the boundary are addressed"
        );
        // A minimal fixture with no hunk identity addresses nothing (nothing to
        // revert on disk rather than a wrong guess).
        assert!(post_boundary_hunks(&[diff_ev(2, "a.rs", "v")], 1).is_empty());
    }

    #[test]
    fn classify_maps_kinds_to_domains() {
        assert_eq!(classify("agent.message"), EventDomain::Conversation);
        assert_eq!(classify("user.intent.submit_turn"), EventDomain::Conversation);
        assert_eq!(classify("diff.proposed"), EventDomain::Code);
        assert_eq!(classify("tool.result"), EventDomain::Code);
        assert_eq!(classify("edit.write_file"), EventDomain::Code);
        assert_eq!(classify("verify.result"), EventDomain::Other);
        assert_eq!(classify("plan.updated"), EventDomain::Other);
    }

    #[test]
    fn rewind_targets_drop_the_right_post_boundary_domain() {
        // seq 1 (conv) + 2 (code) are the prefix; 3 (conv) + 4 (code) are after.
        let events = vec![
            ev(1, "agent.message", json!({ "text": "one" })),
            diff_ev(2, "a.rs", "base"),
            ev(3, "agent.message", json!({ "text": "three" })),
            diff_ev(4, "a.rs", "changed"),
        ];
        let boundary = 2;

        // Code rewind: keep prefix + post-boundary conversation, drop post code.
        let code = rewind_child_events(&events, boundary, RewindTarget::Code);
        let code_seqs: Vec<u64> = code.iter().map(|e| e.seq).collect();
        assert_eq!(code_seqs, vec![1, 2, 3], "post-boundary code (4) is reverted");

        // Conversation rewind: keep prefix + post-boundary code, drop post conv.
        let conv = rewind_child_events(&events, boundary, RewindTarget::Conversation);
        let conv_seqs: Vec<u64> = conv.iter().map(|e| e.seq).collect();
        assert_eq!(conv_seqs, vec![1, 2, 4], "post-boundary conversation (3) is reverted");

        // Both: fold back to the boundary, nothing after survives.
        let both = rewind_child_events(&events, boundary, RewindTarget::Both);
        let both_seqs: Vec<u64> = both.iter().map(|e| e.seq).collect();
        assert_eq!(both_seqs, vec![1, 2]);
    }

    #[test]
    fn code_state_folds_last_write_and_diff_reports_changes() {
        let events = vec![
            diff_ev(1, "a.rs", "v1"),
            diff_ev(2, "b.rs", "b1"),
            diff_ev(3, "a.rs", "v2"),
        ];
        let at_boundary = code_state(&events, Some(1));
        let at_tail = code_state(&events, None);
        assert_eq!(at_boundary.len(), 1, "only a.rs exists at seq 1");
        assert_eq!(at_tail.len(), 2, "a.rs + b.rs at the tail");
        assert_ne!(at_boundary["a.rs"], at_tail["a.rs"], "a.rs changed (v1 -> v2)");

        let changes = diff_code_states(&at_boundary, &at_tail);
        // b.rs added, a.rs modified.
        assert_eq!(changes.len(), 2);
        assert!(changes
            .iter()
            .any(|c| c.file == "b.rs" && c.status == ChangeStatus::Added));
        assert!(changes
            .iter()
            .any(|c| c.file == "a.rs" && c.status == ChangeStatus::Modified));
        assert_eq!(changed_files(&at_boundary, &at_tail), vec!["a.rs", "b.rs"]);
    }

    #[test]
    fn invalidated_receipts_scope_by_path_intersection() {
        let reverted = vec!["src/a.rs".to_string()];
        let receipts = vec![
            ReceiptScope {
                event_id: EventId::from("r-a"),
                scope: vec!["src/a.rs".to_string()],
            },
            ReceiptScope {
                event_id: EventId::from("r-dir"),
                scope: vec!["src".to_string()], // directory contains src/a.rs
            },
            ReceiptScope {
                event_id: EventId::from("r-b"),
                scope: vec!["src/b.rs".to_string()], // unrelated file
            },
        ];
        let out = invalidated_receipts(&reverted, &receipts);
        assert!(out.contains(&EventId::from("r-a")));
        assert!(out.contains(&EventId::from("r-dir")), "dir scope contains the file");
        assert!(!out.contains(&EventId::from("r-b")), "unrelated file is untouched");
    }

    #[test]
    fn fork_point_ordinal_boundary_splits_inherited_from_own() {
        // A child with a marker, 2 inherited events, then 1 own event.
        let parent = SessionId::from("parent");
        let fp = ForkPoint::new(parent.clone(), 2, 5);
        assert_eq!(fp.start_ordinal, 3, "own history starts after the 2 inherited");

        let child = vec![
            ev(1, FORK_POINT_KIND, serde_json::to_value(&fp).unwrap()),
            ev(2, "agent.message", json!({ "text": "inherited-1" })),
            ev(3, "agent.message", json!({ "text": "inherited-2" })),
            ev(4, "agent.message", json!({ "text": "own-1" })),
        ];
        let (got, inherited, own) = split_inherited_own(&child);
        assert_eq!(got, Some(fp));
        assert_eq!(inherited.len(), 2);
        assert_eq!(own.len(), 1);
        assert_eq!(
            own[0].payload.get("text").and_then(|t| t.as_str()),
            Some("own-1")
        );

        // Without a marker, everything is the session's own history.
        let no_marker = vec![ev(1, "agent.message", json!({ "text": "x" }))];
        let (none, inh, all_own) = split_inherited_own(&no_marker);
        assert!(none.is_none());
        assert!(inh.is_empty());
        assert_eq!(all_own.len(), 1);
    }

    #[test]
    fn coverage_digest_is_deterministic_and_tamper_evident() {
        let cov = CheckpointCoverage {
            repo_state: StateRef::of(&["a.rs:hash".to_string()]),
            thread: StateRef::of(&["hi".to_string()]),
            plan: StateRef::counted(1, "plan-json"),
            goal: None,
            artifacts: StateRef::default(),
            live_state_capsule: None,
        };
        assert_eq!(cov.digest(), cov.clone().digest(), "same coverage -> same digest");
        let mut tampered = cov.clone();
        tampered.repo_state = StateRef::of(&["a.rs:OTHER".to_string()]);
        assert_ne!(cov.digest(), tampered.digest(), "a changed reference changes the digest");
    }
}
