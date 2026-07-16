//! Headless Research & Knowledge Lab (HIDE bible chapter 08).
//!
//! This crate is the backend of the Research Lab: a model-agnostic
//! [`RuntimeClient`](runtime_client::RuntimeClient) seam (reusing
//! `hawking-orch`'s `InferenceClient`), content-addressing over
//! `hide_core::BlobStore` ([`cas`]), a real petgraph-backed knowledge graph
//! ([`kg`]) with Local/Global/Path queries + entity resolution + JSONL
//! persistence, a real arXiv ingestion adapter ([`ingest`]), an adversarial
//! verifier with CAS citation re-verification ([`verify`]), a checkpointed
//! research pipeline FSM ([`pipeline`]), literature mapping ([`litmap`]), and
//! research⇄code/issues/memory bridges ([`bridge`]). UI panels are out of scope.

#[rustfmt::skip]
pub mod bridge {
    //! Research ⇄ code / issues / memory bridges (bible ch.08 §4.10, §4.13).
    //!
    //! * `claim_to_issue` turns a verified [`Claim`] into a real [`IssueDraft`] with
    //!   acceptance criteria and `linked_symbols` resolved against the code index
    //!   (`hawking-index`), so a finding becomes an actionable, gated task.
    //! * `equation_to_code` turns an extracted equation into a typed function stub
    //!   (the §4.10 "equations become functions" path).
    //! * `node_to_memory` writes a KG node into ch.04 long-term memory with
    //!   back-linking provenance.

    use crate::kg::{Claim, KnowledgeNode};
    use hawking_context::memory::{MemoryKind, MemoryRecord};
    use hawking_index::graph::Symbol;
    use hawking_index::query::{CodeIndex, SearchQuery};
    use hide_core::error::Result;
    use hide_core::types::Provenance;
    use serde::{Deserialize, Serialize};

    /// A concrete, gated issue ready to be minted into the repo's tracker (§4.10).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct IssueDraft {
        pub title: String,
        pub body: String,
        /// Claims this issue is derived from (provenance back-links).
        pub claim_ids: Vec<String>,
        /// What "done" means — each criterion is checkable.
        pub acceptance_criteria: Vec<String>,
        /// Code symbols this work likely touches (resolved from the code index).
        pub linked_symbols: Vec<LinkedSymbol>,
        pub labels: Vec<String>,
        pub effort: Effort,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct LinkedSymbol {
        pub qualified_name: String,
        pub path: String,
        pub relation: String,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum Effort {
        Small,
        Medium,
        Large,
    }

    /// Back-compat: the prior flat issue type, now produced from an [`IssueDraft`].
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct FindingIssue {
        pub title: String,
        pub body: String,
        pub claim_ids: Vec<String>,
        pub suggested_labels: Vec<String>,
    }

    impl From<IssueDraft> for FindingIssue {
        fn from(d: IssueDraft) -> Self {
            FindingIssue { title: d.title, body: d.body, claim_ids: d.claim_ids, suggested_labels: d.labels }
        }
    }

    /// Build an [`IssueDraft`] from a claim, with acceptance criteria. Pure (no
    /// index lookup) — use [`claim_to_issue_linked`] to also resolve symbols.
    pub fn claim_to_issue(claim: &Claim) -> IssueDraft {
        let title: String = claim.text.chars().take(72).collect();
        let body = format!(
            "Research finding:\n\n> {}\n\nSource claim id: `{}` (doc `{}`).",
            claim.text, claim.id, claim.provenance.doc_id
        );
        IssueDraft {
            title,
            body,
            claim_ids: vec![claim.id.clone()],
            acceptance_criteria: vec![
                "The claim is reproduced or refuted by a test/experiment in this repo.".to_string(),
                "Any code touched is covered by the deterministic oracle suite.".to_string(),
                "The finding's provenance link is recorded in the knowledge graph.".to_string(),
            ],
            linked_symbols: Vec::new(),
            labels: vec!["research".to_string()],
            effort: estimate_effort(&claim.text),
        }
    }

    /// As [`claim_to_issue`], plus resolve likely-affected code symbols by searching
    /// the code index for the claim's salient terms. The index matches a whole query
    /// string as one substring, so we search per-term and union the hits.
    pub async fn claim_to_issue_linked(claim: &Claim, index: &dyn CodeIndex) -> Result<IssueDraft> {
        let mut draft = claim_to_issue(claim);
        let mut seen = std::collections::HashSet::new();
        let mut linked = Vec::new();
        for term in salient_terms(&claim.text) {
            let query = SearchQuery {
                text: term,
                limit: 5,
                include_symbols: true,
                include_lexical: true,
                include_semantic: false,
            };
            for h in index.search(query).await? {
                let key = (h.title.clone(), h.span.path.display().to_string());
                if seen.insert(key.clone()) {
                    linked.push(LinkedSymbol {
                        qualified_name: key.0,
                        path: key.1,
                        relation: "implements_or_affected".to_string(),
                    });
                }
            }
        }
        draft.linked_symbols = linked;
        Ok(draft)
    }

    /// Turn an extracted equation (latex + symbol descriptions) into a typed code
    /// stub. The symbol table drives the parameter list (§4.10).
    pub fn equation_to_code(name: &str, latex: &str, symbols: &[(String, String)]) -> String {
        let params = symbols.iter().map(|(s, _)| format!("{}: f64", sanitize_ident(s))).collect::<Vec<_>>().join(", ");
        let doc_symbols = symbols
            .iter()
            .map(|(s, desc)| format!("/// - `{}`: {}", sanitize_ident(s), desc))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "/// {latex}\n///\n/// Parameters:\n{doc_symbols}\nfn {fn_name}({params}) -> f64 {{\n    \
             // TODO: implement from the equation above.\n    todo!(\"derive from: {latex}\")\n}}",
            fn_name = sanitize_ident(name)
        )
    }

    /// Write a KG node into ch.04 long-term memory with back-linking provenance.
    pub fn node_to_memory(node: &KnowledgeNode, provenance: Provenance) -> MemoryRecord {
        let mut prov = provenance;
        prov.derived_from.push(format!("kg:{}", node.id));
        MemoryRecord {
            id: format!("kg:{}", node.id),
            kind: MemoryKind::Semantic,
            text: node.label.clone(),
            importance: 0.7,
            created_at_ms: node.created_at_ms,
            last_used_at_ms: None,
            provenance: prov,
            tags: vec!["research".to_string(), format!("{:?}", node.kind)],
        }
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct CodeResearchLink {
        pub claim_id: String,
        pub symbol: Symbol,
        pub relation: String,
    }

    fn estimate_effort(text: &str) -> Effort {
        let n = text.split_whitespace().count();
        if n < 12 {
            Effort::Small
        } else if n < 40 {
            Effort::Medium
        } else {
            Effort::Large
        }
    }

    /// Pull the most salient content words (drop stopwords/short tokens) for an
    /// index query — returned as individual terms.
    fn salient_terms(text: &str) -> Vec<String> {
        const STOP: &[&str] = &[
            "the", "and", "for", "with", "that", "this", "from", "into", "over", "than", "more", "less", "have",
            "been", "are", "was", "were", "our", "their", "improves", "reduces",
        ];
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 3 && !STOP.contains(&w.to_lowercase().as_str()))
            .take(6)
            .map(|w| w.to_string())
            .collect()
    }

    fn sanitize_ident(s: &str) -> String {
        let mut out = String::new();
        for ch in s.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                out.push(ch.to_ascii_lowercase());
            } else if !out.ends_with('_') {
                out.push('_');
            }
        }
        let out = out.trim_matches('_').to_string();
        if out.is_empty() || out.chars().next().unwrap().is_numeric() {
            format!("v_{out}")
        } else {
            out
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::kg::{ConfidenceTier, ProvenanceSpan};
        use hawking_index::query::InMemoryCodeIndex;

        fn claim(text: &str) -> Claim {
            Claim {
                id: "claim:1".into(),
                text: text.into(),
                provenance: ProvenanceSpan {
                    doc_id: "doc1".into(),
                    span_id: None,
                    char_range: None,
                    citation: None,
                    content_hash: None,
                    evidence_blob: None,
                    provenance: Provenance::trusted("test"),
                },
                confidence: ConfidenceTier::Extracted,
            }
        }

        #[test]
        fn issue_draft_has_acceptance_criteria_and_effort() {
            let d = claim_to_issue(&claim("paged attention reduces kv cache memory waste"));
            assert!(!d.acceptance_criteria.is_empty());
            assert_eq!(d.effort, Effort::Small);
            let issue: FindingIssue = d.into();
            assert_eq!(issue.suggested_labels, vec!["research".to_string()]);
        }

        #[test]
        fn equation_to_code_emits_typed_stub() {
            let code = equation_to_code(
                "softmax scale",
                "y = x / sqrt(d)",
                &[("x".into(), "input".into()), ("d".into(), "dimension".into())],
            );
            assert!(code.contains("fn softmax_scale(x: f64, d: f64) -> f64"));
            assert!(code.contains("todo!"));
        }

        #[tokio::test]
        async fn claim_links_to_indexed_symbols() {
            let index = InMemoryCodeIndex::default();
            // Register a symbol the claim's terms should match.
            index.add_symbol(Symbol {
                qualified_name: "crate::attn::paged_attention".to_string(),
                name: "paged_attention".to_string(),
                kind: "fn".to_string(),
                file: "src/attn.rs".to_string(),
            });
            let d = claim_to_issue_linked(&claim("paged_attention improves cache reuse"), &index).await.unwrap();
            assert!(!d.linked_symbols.is_empty());
        }

        #[test]
        fn memory_record_backlinks_to_node() {
            let node = KnowledgeNode {
                id: "concept:x".into(),
                kind: crate::kg::NodeKind::Concept,
                label: "KV cache".into(),
                confidence: ConfidenceTier::Inferred,
                provenance: vec![],
                created_at_ms: 1,
            };
            let rec = node_to_memory(&node, Provenance::trusted("kg"));
            assert!(rec.provenance.derived_from.iter().any(|d| d == "kg:concept:x"));
        }
    }
}
#[rustfmt::skip]
pub mod cas {
    //! Content-addressing for the Research Lab (bible ch.08 §4.2.1, §4.3, Tenet 7).
    //!
    //! Two responsibilities:
    //!
    //! * **Stable, content-derived ids.** Re-ingesting identical bytes must yield
    //!   the same node ids so ingestion is idempotent. We hash *normalized* content
    //!   with blake3 (the bible's chosen hash) and mint ids like
    //!   `chunk:<hex>` / `claim:<hex>` / `doc:<hex>`.
    //! * **Immutable evidence receipts.** The raw bytes behind a citation are pinned
    //!   in a [`hide_core::persistence::BlobStore`] so a synthesized sentence can be
    //!   re-verified against the exact bytes it was derived from (§4.7.3, §6).
    //!
    //! Note on stores: `hide_core::FileBlobStore` content-addresses with sha256, but
    //! the *graph node id* uses blake3 of the normalized form, which is what makes
    //! re-ingest idempotent regardless of the blob backend.

    use hide_core::error::Result;
    use hide_core::persistence::DynBlobStore;
    use hide_core::types::BlobRef;

    /// Lowercase-hex blake3 digest of `bytes`.
    pub fn blake3_hex(bytes: &[u8]) -> String {
        blake3::hash(bytes).to_hex().to_string()
    }

    /// Normalize free text before hashing so trivially-different encodings of the
    /// same content collapse to one id: trim, collapse internal whitespace runs to a
    /// single space, drop a trailing newline. Deterministic and cheap.
    pub fn normalize_text(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut prev_ws = false;
        for ch in text.trim().chars() {
            if ch.is_whitespace() {
                if !prev_ws {
                    out.push(' ');
                }
                prev_ws = true;
            } else {
                out.push(ch);
                prev_ws = false;
            }
        }
        out
    }

    /// Content-addressed id for normalized text under a node-kind prefix, e.g.
    /// `content_id("chunk", text)` → `chunk:9f86d0...`.
    pub fn content_id(prefix: &str, text: &str) -> String {
        let norm = normalize_text(text);
        format!("{prefix}:{}", blake3_hex(norm.as_bytes()))
    }

    /// Content-addressed id derived from several fields (order-significant). Used for
    /// claims (`claim:hash(text|paper_id)`) and docs (`doc:hash(title|body)`).
    pub fn composite_id(prefix: &str, fields: &[&str]) -> String {
        let joined = fields.iter().map(|f| normalize_text(f)).collect::<Vec<_>>().join("\u{1f}"); // unit separator — unlikely to occur in source text
        format!("{prefix}:{}", blake3_hex(joined.as_bytes()))
    }

    /// The canonical evidence bytes for a piece of free text: the *normalized* form,
    /// UTF-8 encoded. This is the single byte source that both content-addressing
    /// (claim/citation ids) and evidence pinning/re-verification must agree on, so
    /// that an id, its pinned blob, and its re-check hash never diverge (§4.7.3).
    pub fn canonical_evidence_bytes(text: &str) -> Vec<u8> {
        normalize_text(text).into_bytes()
    }

    /// Pin raw evidence bytes in the CAS and return both the [`BlobRef`] and the
    /// blake3 content hash used as the immutable receipt. The blake3 hash is the
    /// one we record on provenance (so re-verification is backend-independent); the
    /// `BlobRef` is how the bytes are fetched back.
    ///
    /// Invariant: the recorded hash is `blake3_hex` of *exactly* the bytes pinned,
    /// so [`verify_evidence`] (which re-hashes the fetched blob bytes) is sound by
    /// construction.
    pub fn pin_evidence(cas: &DynBlobStore, bytes: Vec<u8>, media_type: Option<String>) -> Result<(BlobRef, String)> {
        let hash = blake3_hex(&bytes);
        let blob = cas.put(bytes, media_type)?;
        Ok((blob, hash))
    }

    /// Pin a section's *canonical* evidence bytes (normalized text) and return the
    /// receipt. The pinned bytes match what [`content_id`]/[`composite_id`] hash for
    /// the same text, so the claim id and its evidence receipt agree on one source.
    pub fn pin_canonical_evidence(cas: &DynBlobStore, text: &str) -> Result<(BlobRef, String)> {
        pin_evidence(cas, canonical_evidence_bytes(text), Some("text/plain".to_string()))
    }

    /// Re-open evidence bytes from the CAS and confirm they still blake3-hash to the
    /// recorded receipt. `None` blob → bytes were never pinned (cannot verify).
    pub fn verify_evidence(cas: &DynBlobStore, blob: &BlobRef, expected_hash: &str) -> Result<EvidenceCheck> {
        let Some(bytes) = cas.get(blob)? else {
            return Ok(EvidenceCheck::Missing);
        };
        let actual = blake3_hex(&bytes);
        if actual == expected_hash {
            Ok(EvidenceCheck::Intact { bytes })
        } else {
            Ok(EvidenceCheck::Tampered { expected: expected_hash.to_string(), actual })
        }
    }

    /// Outcome of re-checking a citation's evidence against the CAS (§4.7.3).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum EvidenceCheck {
        /// Bytes present and hash-matched. Carries the bytes for phrase re-checks.
        Intact { bytes: Vec<u8> },
        /// Bytes present but the hash changed — the evidence was mutated.
        Tampered { expected: String, actual: String },
        /// No bytes pinned for this blob — the claim cannot be re-verified.
        Missing,
    }

    impl EvidenceCheck {
        pub fn is_intact(&self) -> bool {
            matches!(self, EvidenceCheck::Intact { .. })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use hide_core::persistence::{DynBlobStore, InMemoryBlobStore};
        use std::sync::Arc;

        #[test]
        fn normalization_collapses_whitespace() {
            assert_eq!(normalize_text("  a\n\t b  c \n"), "a b c");
        }

        #[test]
        fn content_ids_are_idempotent_and_normalization_stable() {
            let a = content_id("chunk", "paged   attention\n improves reuse");
            let b = content_id("chunk", "paged attention improves reuse");
            assert_eq!(a, b);
            assert!(a.starts_with("chunk:"));
        }

        #[test]
        fn composite_id_is_order_sensitive() {
            let a = composite_id("claim", &["text", "paper1"]);
            let b = composite_id("claim", &["paper1", "text"]);
            assert_ne!(a, b);
        }

        #[test]
        fn pin_and_verify_roundtrips_and_detects_tamper() {
            let cas: DynBlobStore = Arc::new(InMemoryBlobStore::default());
            let (blob, hash) = pin_evidence(&cas, b"73% accuracy".to_vec(), None).unwrap();
            assert!(verify_evidence(&cas, &blob, &hash).unwrap().is_intact());
            // A wrong expected hash is reported as tampering, not a silent pass.
            match verify_evidence(&cas, &blob, "deadbeef").unwrap() {
                EvidenceCheck::Tampered { .. } => {}
                other => panic!("expected tamper, got {other:?}"),
            }
        }
    }
}
#[rustfmt::skip]
pub mod checkpoint {
    //! Per-event checkpoint ledger (bible ch.08 §4.6).
    //!
    //! The run ledger in [`crate::run_ledger`] records *run summaries*; this records
    //! the *per-event journal* that makes an overnight run resumable: each state
    //! transition and each fetched/read doc appends a line, so a crash at 3 a.m.
    //! resumes from the last completed state without re-fetching (CAS dedup) or
    //! re-extracting (content-addressed nodes).

    use hide_core::error::Result;
    use hide_core::ids::RunId;
    use parking_lot::Mutex;
    use serde::{Deserialize, Serialize};
    use std::fs::{File, OpenOptions};
    use std::io::{BufRead, BufReader, Write};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    /// One journal event for a run.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct CheckpointEvent {
        pub run_id: String,
        pub seq: u64,
        pub at_ms: u64,
        pub kind: CheckpointKind,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "event", rename_all = "snake_case")]
    pub enum CheckpointKind {
        /// Run opened with this topic/seed.
        Opened { topic: String, seed: u64 },
        /// Entered a pipeline state (its `{:?}` name).
        State { state: String },
        /// A source doc was fetched + content-addressed (recorded so we never
        /// re-fetch the same content_hash on resume).
        Fetched { doc_id: String, content_hash: Option<String> },
        /// A round of the reflect loop completed.
        Round { round: u32, coverage: f32, novelty: f32 },
        /// Run finalized.
        Done { docs_read: usize, claims: usize },
    }

    /// Append-only, line-per-event journal for one or many runs.
    pub trait CheckpointLedger: Send + Sync {
        fn append(&self, event: CheckpointEvent) -> Result<()>;
        fn events_for(&self, run_id: &str) -> Result<Vec<CheckpointEvent>>;

        /// The set of content hashes already fetched for a run — used to skip
        /// re-fetching on resume.
        fn fetched_hashes(&self, run_id: &str) -> Result<std::collections::HashSet<String>> {
            let mut out = std::collections::HashSet::new();
            for e in self.events_for(run_id)? {
                if let CheckpointKind::Fetched { content_hash: Some(h), .. } = e.kind {
                    out.insert(h);
                }
            }
            Ok(out)
        }

        /// The last state name recorded for a run, if any (the resume point).
        fn last_state(&self, run_id: &str) -> Result<Option<String>> {
            let mut last = None;
            for e in self.events_for(run_id)? {
                if let CheckpointKind::State { state } = e.kind {
                    last = Some(state);
                }
            }
            Ok(last)
        }
    }

    pub type DynCheckpointLedger = Arc<dyn CheckpointLedger>;

    #[derive(Default)]
    pub struct InMemoryCheckpointLedger {
        events: Mutex<Vec<CheckpointEvent>>,
    }

    impl InMemoryCheckpointLedger {
        pub fn new() -> Self {
            Self::default()
        }
    }

    impl CheckpointLedger for InMemoryCheckpointLedger {
        fn append(&self, event: CheckpointEvent) -> Result<()> {
            self.events.lock().push(event);
            Ok(())
        }

        fn events_for(&self, run_id: &str) -> Result<Vec<CheckpointEvent>> {
            Ok(self.events.lock().iter().filter(|e| e.run_id == run_id).cloned().collect())
        }
    }

    /// A JSONL checkpoint journal on disk (one file holds all runs; filter by id).
    #[derive(Debug, Clone)]
    pub struct JsonlCheckpointLedger {
        path: PathBuf,
        seq: Arc<Mutex<u64>>,
    }

    impl JsonlCheckpointLedger {
        pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
            let path = path.into();
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if !path.exists() {
                File::create(&path)?;
            }
            Ok(Self { path, seq: Arc::new(Mutex::new(0)) })
        }

        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    impl CheckpointLedger for JsonlCheckpointLedger {
        fn append(&self, event: CheckpointEvent) -> Result<()> {
            *self.seq.lock() = event.seq;
            let mut file = OpenOptions::new().create(true).append(true).open(&self.path)?;
            serde_json::to_writer(&mut file, &event)?;
            file.write_all(b"\n")?;
            file.sync_data()?;
            Ok(())
        }

        fn events_for(&self, run_id: &str) -> Result<Vec<CheckpointEvent>> {
            if !self.path.exists() {
                return Ok(Vec::new());
            }
            let file = File::open(&self.path)?;
            let reader = BufReader::new(file);
            let mut out = Vec::new();
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let e: CheckpointEvent = serde_json::from_str(&line)?;
                if e.run_id == run_id {
                    out.push(e);
                }
            }
            Ok(out)
        }
    }

    /// A small monotonic sequencer for a run's checkpoint events.
    pub struct RunJournal {
        run_id: RunId,
        ledger: DynCheckpointLedger,
        seq: u64,
    }

    impl RunJournal {
        pub fn new(run_id: RunId, ledger: DynCheckpointLedger) -> Self {
            // Resume the sequence number from any existing events.
            let seq = ledger.events_for(run_id.as_str()).ok().and_then(|e| e.last().map(|l| l.seq)).unwrap_or(0);
            Self { run_id, ledger, seq }
        }

        pub fn record(&mut self, kind: CheckpointKind) -> Result<()> {
            self.seq += 1;
            self.ledger.append(CheckpointEvent {
                run_id: self.run_id.0.clone(),
                seq: self.seq,
                at_ms: hide_core::ids::now_ms(),
                kind,
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn jsonl_checkpoints_filter_and_resume() {
            let dir = std::env::temp_dir().join(format!("hawking_ckpt_{}", hide_core::ids::now_ms()));
            let path = dir.join("ckpt.jsonl");
            let ledger: DynCheckpointLedger = Arc::new(JsonlCheckpointLedger::open(&path).unwrap());
            let run = RunId::from("run_a");
            let mut j = RunJournal::new(run.clone(), ledger.clone());
            j.record(CheckpointKind::Opened { topic: "kv cache".into(), seed: 42 }).unwrap();
            j.record(CheckpointKind::State { state: "Fetch".into() }).unwrap();
            j.record(CheckpointKind::Fetched { doc_id: "doc:1".into(), content_hash: Some("h1".into()) }).unwrap();

            // A different run's event must not bleed in.
            let mut other = RunJournal::new(RunId::from("run_b"), ledger.clone());
            other.record(CheckpointKind::State { state: "Read".into() }).unwrap();

            assert_eq!(ledger.events_for("run_a").unwrap().len(), 3);
            assert_eq!(ledger.last_state("run_a").unwrap().as_deref(), Some("Fetch"));
            assert!(ledger.fetched_hashes("run_a").unwrap().contains("h1"));
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}
#[rustfmt::skip]
pub mod experiments {
    use hide_core::ids::{now_ms, RunId};
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Hypothesis {
        pub id: String,
        pub statement: String,
        pub source_claim_ids: Vec<String>,
        pub status: HypothesisStatus,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum HypothesisStatus {
        Proposed,
        Testing,
        Supported,
        Refuted,
        Inconclusive,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ExperimentRun {
        pub id: RunId,
        pub hypothesis_id: String,
        pub command: Vec<String>,
        pub params: BTreeMap<String, String>,
        pub metrics: BTreeMap<String, f64>,
        pub artifacts: Vec<String>,
        pub started_at_ms: u64,
    }

    impl ExperimentRun {
        pub fn new(hypothesis_id: impl Into<String>, command: Vec<String>) -> Self {
            Self {
                id: RunId::new(),
                hypothesis_id: hypothesis_id.into(),
                command,
                params: BTreeMap::new(),
                metrics: BTreeMap::new(),
                artifacts: Vec::new(),
                started_at_ms: now_ms(),
            }
        }
    }
}
#[rustfmt::skip]
pub mod ingest {
    //! Ingestion adapters (bible ch.08 §4.4).
    //!
    //! Every source enters through the [`SourceAdapter`] trait. This crate ships:
    //!
    //! * [`InMemorySourceAdapter`] — a fixture adapter for tests (substring search
    //!   over inserted docs), now content-addressing every doc it returns.
    //! * [`ArxivAdapter`] — a **real** adapter over the public arXiv Atom API via
    //!   `reqwest`, returning [`StructuredDoc`]s whose `content_hash` is populated
    //!   from the fetched bytes (unlocking idempotent ingest + citation
    //!   re-verification, §4.2.1).
    //!
    //! PDF full-text parsing is a documented seam: arXiv gives us title + abstract +
    //! authors deterministically from the Atom feed, which is enough to build Paper
    //! and Claim nodes; the PDF/LaTeX-source body parse (§4.5) is left for the
    //! vision/PDFium pipeline and marked below.

    use crate::cas;
    use futures::future::BoxFuture;
    use hide_core::error::Result;
    use hide_core::types::{BlobRef, Provenance, TrustLevel};
    use parking_lot::RwLock;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct SourceQuery {
        pub query: String,
        pub limit: usize,
        pub source_types: Vec<SourceType>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum SourceType {
        Web,
        Arxiv,
        SemanticScholar,
        OpenAlex,
        Crossref,
        PdfLocal,
        Html,
        Repo,
        Dataset,
        Zotero,
        Bibtex,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct SourceRecord {
        pub id: String,
        pub source_type: SourceType,
        pub title: String,
        pub uri: String,
        pub content_hash: Option<String>,
        pub quality: SourceQuality,
        pub provenance: Provenance,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct SourceQuality {
        pub authority: f32,
        pub recency: f32,
        pub independence: f32,
        pub reproducibility: f32,
    }

    impl SourceQuality {
        pub fn score(&self) -> f32 {
            (self.authority + self.recency + self.independence + self.reproducibility) / 4.0
        }
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct StructuredDoc {
        pub id: String,
        pub source: SourceRecord,
        pub title: String,
        pub sections: Vec<DocSection>,
        pub references: Vec<CitationRef>,
        pub blob: Option<BlobRef>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct DocSection {
        pub heading: String,
        pub text: String,
        pub spans: Vec<DocSpan>,
        /// CAS receipt for this section's *own* canonical evidence bytes, populated
        /// when the section is pinned (`pin_doc_evidence`). The pinned bytes are
        /// exactly the bytes the section's claim node is content-addressed over, so
        /// the claim id, the evidence blob, and the re-verification hash all agree
        /// on one canonical byte source (§4.7.3 — citation re-verification soundness).
        #[serde(default)]
        pub evidence: Option<SectionEvidence>,
    }

    /// An immutable CAS receipt for a section's evidence bytes: the blob ref the
    /// bytes are stored under, and the blake3 hash of *exactly* those bytes.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct SectionEvidence {
        pub blob: BlobRef,
        pub content_hash: String,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct DocSpan {
        pub id: String,
        pub start_char: usize,
        pub end_char: usize,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct CitationRef {
        pub key: String,
        pub title: Option<String>,
        pub doi: Option<String>,
        pub uri: Option<String>,
    }

    pub trait SourceAdapter: Send + Sync {
        fn name(&self) -> &str;
        fn source_type(&self) -> SourceType;
        fn search<'a>(&'a self, query: &'a SourceQuery) -> BoxFuture<'a, Result<Vec<SourceRecord>>>;
        fn fetch<'a>(&'a self, record: &'a SourceRecord) -> BoxFuture<'a, Result<StructuredDoc>>;
    }

    /// Build a [`StructuredDoc`] from title + abstract text, content-addressing the
    /// id and a whole-doc `content_hash` from the normalized body. Shared by every
    /// adapter so all docs are addressed the same way.
    pub fn structured_doc_from_text(
        source_type: SourceType,
        external_id: &str,
        uri: &str,
        title: &str,
        abstract_text: &str,
        provenance: Provenance,
        quality: SourceQuality,
    ) -> StructuredDoc {
        let body = abstract_text.trim();
        let content_hash = cas::blake3_hex(cas::normalize_text(&format!("{title}\n{body}")).as_bytes());
        let doc_id = format!("doc:{content_hash}");
        let span = DocSpan { id: cas::content_id("span", body), start_char: 0, end_char: body.chars().count() };
        let sections = if body.is_empty() {
            Vec::new()
        } else {
            vec![DocSection {
                heading: "Abstract".to_string(),
                text: body.to_string(),
                spans: vec![span],
                evidence: None,
            }]
        };
        StructuredDoc {
            id: doc_id,
            source: SourceRecord {
                id: external_id.to_string(),
                source_type,
                title: title.to_string(),
                uri: uri.to_string(),
                content_hash: Some(content_hash),
                quality,
                provenance,
            },
            title: title.to_string(),
            sections,
            references: Vec::new(),
            blob: None,
        }
    }

    // ───────────────────────────── In-memory fixture ─────────────────────────────

    pub struct InMemorySourceAdapter {
        name: String,
        source_type: SourceType,
        records: RwLock<BTreeMap<String, StructuredDoc>>,
    }

    impl InMemorySourceAdapter {
        pub fn new(name: impl Into<String>, source_type: SourceType) -> Self {
            Self { name: name.into(), source_type, records: RwLock::new(BTreeMap::new()) }
        }

        pub fn insert(&self, doc: StructuredDoc) {
            self.records.write().insert(doc.source.id.clone(), doc);
        }
    }

    impl Default for SourceQuality {
        fn default() -> Self {
            Self { authority: 0.5, recency: 0.5, independence: 0.5, reproducibility: 0.5 }
        }
    }

    impl SourceAdapter for InMemorySourceAdapter {
        fn name(&self) -> &str {
            &self.name
        }

        fn source_type(&self) -> SourceType {
            self.source_type
        }

        fn search<'a>(&'a self, query: &'a SourceQuery) -> BoxFuture<'a, Result<Vec<SourceRecord>>> {
            Box::pin(async move {
                let needle = query.query.to_lowercase();
                let mut hits: Vec<_> = self
                    .records
                    .read()
                    .values()
                    .filter(|doc| {
                        doc.title.to_lowercase().contains(&needle)
                            || doc.sections.iter().any(|s| s.text.to_lowercase().contains(&needle))
                    })
                    .map(|doc| doc.source.clone())
                    .collect();
                hits.sort_by(|a, b| {
                    b.quality.score().partial_cmp(&a.quality.score()).unwrap_or(std::cmp::Ordering::Equal)
                });
                hits.truncate(query.limit);
                Ok(hits)
            })
        }

        fn fetch<'a>(&'a self, record: &'a SourceRecord) -> BoxFuture<'a, Result<StructuredDoc>> {
            Box::pin(async move {
                self.records
                    .read()
                    .get(&record.id)
                    .cloned()
                    .ok_or_else(|| hide_core::HideError::NotFound(record.id.clone()))
            })
        }
    }

    // ──────────────────────────────── arXiv (real) ───────────────────────────────

    /// A real [`SourceAdapter`] over the public arXiv Atom API.
    ///
    /// `search` issues `GET https://export.arxiv.org/api/query?search_query=...` and
    /// parses the returned Atom feed; `fetch` rebuilds the full [`StructuredDoc`]
    /// (title + abstract) and populates `content_hash`. Full PDF/LaTeX body parsing
    /// is the documented §4.5 seam — the abstract is enough for Paper/Claim nodes.
    pub struct ArxivAdapter {
        base_url: String,
        client: reqwest::Client,
    }

    impl Default for ArxivAdapter {
        fn default() -> Self {
            Self::new()
        }
    }

    impl ArxivAdapter {
        pub fn new() -> Self {
            Self { base_url: "https://export.arxiv.org/api/query".to_string(), client: reqwest::Client::new() }
        }

        /// Point the adapter at an alternate base URL (used by tests to hit a local
        /// fixture server instead of the live API).
        pub fn with_base_url(base_url: impl Into<String>) -> Self {
            Self { base_url: base_url.into(), client: reqwest::Client::new() }
        }

        async fn fetch_feed(&self, query: &str, limit: usize) -> Result<String> {
            let url =
                format!("{}?search_query=all:{}&start=0&max_results={}", self.base_url, urlencode(query), limit.max(1));
            let resp = self
                .client
                .get(&url)
                .header("User-Agent", "hide-research/0.1")
                .send()
                .await
                .map_err(|e| hide_core::HideError::RuntimeUnavailable(format!("arxiv request: {e}")))?;
            let text =
                resp.text().await.map_err(|e| hide_core::HideError::RuntimeUnavailable(format!("arxiv body: {e}")))?;
            Ok(text)
        }
    }

    impl SourceAdapter for ArxivAdapter {
        fn name(&self) -> &str {
            "arxiv"
        }

        fn source_type(&self) -> SourceType {
            SourceType::Arxiv
        }

        fn search<'a>(&'a self, query: &'a SourceQuery) -> BoxFuture<'a, Result<Vec<SourceRecord>>> {
            Box::pin(async move {
                let feed = self.fetch_feed(&query.query, query.limit).await?;
                let entries = parse_arxiv_feed(&feed);
                Ok(entries.into_iter().map(|e| e.into_record()).collect())
            })
        }

        fn fetch<'a>(&'a self, record: &'a SourceRecord) -> BoxFuture<'a, Result<StructuredDoc>> {
            Box::pin(async move {
                // The record id is the arXiv id; re-query for its single entry so we
                // have the abstract regardless of how the record was obtained.
                let feed = self.fetch_feed(&record.id, 1).await?;
                let mut entries = parse_arxiv_feed(&feed);
                let entry = entries
                    .pop()
                    .ok_or_else(|| hide_core::HideError::NotFound(format!("arxiv entry {}", record.id)))?;
                Ok(entry.into_doc())
            })
        }
    }

    /// One parsed Atom `<entry>`.
    #[derive(Debug, Clone)]
    pub struct ArxivEntry {
        pub id: String,
        pub title: String,
        pub summary: String,
        pub uri: String,
    }

    impl ArxivEntry {
        fn provenance() -> Provenance {
            // Network-sourced → not trusted; tagged for the egress audit.
            Provenance {
                source: "arxiv".to_string(),
                trust: TrustLevel::Network,
                confidence: 0.9,
                labels: vec!["arxiv".to_string()],
                derived_from: Vec::new(),
            }
        }

        pub fn into_record(self) -> SourceRecord {
            let doc = self.into_doc();
            doc.source
        }

        pub fn into_doc(self) -> StructuredDoc {
            structured_doc_from_text(
                SourceType::Arxiv,
                &self.id,
                &self.uri,
                &self.title,
                &self.summary,
                Self::provenance(),
                SourceQuality { authority: 0.8, recency: 0.7, independence: 0.6, reproducibility: 0.6 },
            )
        }
    }

    /// Parse an arXiv Atom feed into entries. Deterministic, dependency-light: pulls
    /// the inner text of `<entry>` blocks and the `<id>/<title>/<summary>` tags.
    pub fn parse_arxiv_feed(xml: &str) -> Vec<ArxivEntry> {
        let mut out = Vec::new();
        for block in between_all(xml, "<entry>", "</entry>") {
            let id_raw = first_between(&block, "<id>", "</id>").unwrap_or_default();
            let title = first_between(&block, "<title>", "</title>")
                .map(|s| collapse_ws(&unescape_xml(&s)))
                .unwrap_or_default();
            let summary = first_between(&block, "<summary>", "</summary>")
                .map(|s| collapse_ws(&unescape_xml(&s)))
                .unwrap_or_default();
            if title.is_empty() {
                continue;
            }
            // arXiv id is the trailing path component of the <id> URL.
            let short_id = id_raw.rsplit('/').next().unwrap_or(&id_raw).trim().to_string();
            out.push(ArxivEntry { id: short_id, title, summary, uri: id_raw.trim().to_string() });
        }
        out
    }

    // ────────────────────────────── tiny xml helpers ─────────────────────────────

    fn between_all(haystack: &str, open: &str, close: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut rest = haystack;
        while let Some(start) = rest.find(open) {
            let after = &rest[start + open.len()..];
            if let Some(end) = after.find(close) {
                out.push(after[..end].to_string());
                rest = &after[end + close.len()..];
            } else {
                break;
            }
        }
        out
    }

    fn first_between(haystack: &str, open: &str, close: &str) -> Option<String> {
        let start = haystack.find(open)? + open.len();
        let end = haystack[start..].find(close)? + start;
        Some(haystack[start..end].to_string())
    }

    fn collapse_ws(s: &str) -> String {
        cas::normalize_text(s)
    }

    fn unescape_xml(s: &str) -> String {
        s.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&apos;", "'")
    }

    fn urlencode(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for b in s.bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
                b' ' => out.push_str("%20"),
                _ => out.push_str(&format!("%{b:02X}")),
            }
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const SAMPLE: &str = r#"<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2309.06180v1</id>
        <title>Efficient Memory Management for Large Language Model Serving with PagedAttention</title>
        <summary>High throughput serving of LLMs requires batching. We propose
    PagedAttention, which reduces KV cache &amp; memory waste.</summary>
      </entry>
      <entry>
        <id>http://arxiv.org/abs/2205.14135v2</id>
        <title>FlashAttention</title>
        <summary>An IO-aware exact attention algorithm.</summary>
      </entry>
    </feed>"#;

        #[test]
        fn parses_arxiv_feed_entries() {
            let entries = parse_arxiv_feed(SAMPLE);
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].id, "2309.06180v1");
            assert!(entries[0].title.contains("PagedAttention"));
            assert!(entries[0].summary.contains("KV cache &"));
        }

        #[test]
        fn arxiv_doc_has_populated_content_hash_and_is_addressed() {
            let entries = parse_arxiv_feed(SAMPLE);
            let doc = entries[0].clone().into_doc();
            assert!(doc.source.content_hash.is_some());
            assert!(doc.id.starts_with("doc:"));
            // Idempotent: same entry → same id + hash.
            let doc2 = parse_arxiv_feed(SAMPLE)[0].clone().into_doc();
            assert_eq!(doc.id, doc2.id);
            assert_eq!(doc.source.content_hash, doc2.source.content_hash);
            assert_eq!(doc.source.provenance.trust, TrustLevel::Network);
        }

        #[test]
        fn urlencode_escapes_spaces() {
            assert_eq!(urlencode("kv cache"), "kv%20cache");
        }
    }
}
#[rustfmt::skip]
pub mod kg {
    //! The knowledge graph (bible ch.08 §4.2, §4.8).
    //!
    //! A real property graph backed by [`petgraph`] (a `StableDiGraph` so node
    //! indices survive deletions during entity resolution merges). On top of the
    //! graph we provide:
    //!
    //! * **Content-addressed ids** (§4.2.1): Paper/Claim/Concept ids are derived
    //!   from normalized content via [`crate::cas`], so re-ingesting identical bytes
    //!   is idempotent.
    //! * **Entity resolution** (§4.8): incoming nodes merge into existing ones by
    //!   content-hash id collision *and* by normalized-name match within a kind,
    //!   folding provenance rather than duplicating.
    //! * **Query modes** (§4.8): `Local` (neighborhood expansion from seed nodes),
    //!   `Global` (kind-filtered ranked listing — the map-reduce entry point), and
    //!   `Path` (shortest typed path between two nodes).
    //! * **Persistence** (§4.3): the whole graph round-trips to a single JSONL file
    //!   so it survives process exit without a server.

    use crate::cas;
    use crate::ingest::StructuredDoc;
    use hide_core::error::Result;
    use hide_core::ids::{now_ms, TimestampMs};
    use hide_core::types::Provenance;
    use parking_lot::RwLock;
    use petgraph::stable_graph::{NodeIndex, StableDiGraph};
    use petgraph::visit::EdgeRef;
    use petgraph::Direction;
    use serde::{Deserialize, Serialize};
    use std::collections::{HashMap, VecDeque};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct KnowledgeNode {
        pub id: String,
        pub kind: NodeKind,
        pub label: String,
        pub confidence: ConfidenceTier,
        pub provenance: Vec<ProvenanceSpan>,
        pub created_at_ms: TimestampMs,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum NodeKind {
        Paper,
        Author,
        Venue,
        Claim,
        Method,
        Dataset,
        Metric,
        Equation,
        CodeSymbol,
        Experiment,
        Issue,
        Concept,
        Note,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ConfidenceTier {
        Measured,
        Extracted,
        Inferred,
        Speculative,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct KnowledgeEdge {
        pub id: String,
        pub from: String,
        pub to: String,
        pub kind: EdgeKind,
        pub confidence: f32,
        pub provenance: Vec<ProvenanceSpan>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum EdgeKind {
        Supports,
        Refutes,
        Mentions,
        Cites,
        Implements,
        DerivedFrom,
        UsesDataset,
        ReportsMetric,
        Contradicts,
        Related,
        SameAs,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ProvenanceSpan {
        pub doc_id: String,
        pub span_id: Option<String>,
        pub char_range: Option<(usize, usize)>,
        pub citation: Option<String>,
        /// Immutable blake3 receipt of the evidence bytes (CAS), when pinned. This is
        /// what makes a citation re-verifiable (§4.7.3).
        #[serde(default)]
        pub content_hash: Option<String>,
        /// CAS blob ref for the evidence bytes, when pinned.
        #[serde(default)]
        pub evidence_blob: Option<hide_core::types::BlobRef>,
        pub provenance: Provenance,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Claim {
        pub id: String,
        pub text: String,
        pub provenance: ProvenanceSpan,
        pub confidence: ConfidenceTier,
    }

    /// The query surface (§4.8). Local/Global/Path are the three primitives every
    /// GraphRAG-style retrieval composes from.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum GraphQuery {
        /// Expand `hops` outward from each seed node; return the reachable set.
        Local { seeds: Vec<String>, hops: usize },
        /// All nodes of a kind, most-recent first (the global map-reduce entry).
        Global { kind: NodeKind, limit: usize },
        /// Shortest directed path (by edge count) between two node ids.
        Path { from: String, to: String },
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct QueryResult {
        pub nodes: Vec<KnowledgeNode>,
        pub edges: Vec<KnowledgeEdge>,
    }

    pub trait KnowledgeGraph: Send + Sync {
        fn add_node(&self, node: KnowledgeNode);
        fn add_edge(&self, edge: KnowledgeEdge);
        fn nodes_by_kind(&self, kind: NodeKind) -> Vec<KnowledgeNode>;
        fn edges_from(&self, node_id: &str) -> Vec<KnowledgeEdge>;
        fn query(&self, q: &GraphQuery) -> QueryResult;
    }

    /// petgraph-backed store. `StableDiGraph` keeps `NodeIndex` valid across the
    /// node removals that entity-resolution merges perform.
    pub struct PetKnowledgeGraph {
        inner: RwLock<GraphInner>,
    }

    struct GraphInner {
        graph: StableDiGraph<KnowledgeNode, KnowledgeEdge>,
        /// node id → index.
        index: HashMap<String, NodeIndex>,
        /// normalized (kind, name) → canonical node id, for name-based resolution.
        name_index: HashMap<(NodeKind, String), String>,
        /// edge id set, to dedup edges.
        edge_ids: std::collections::HashSet<String>,
    }

    impl Default for PetKnowledgeGraph {
        fn default() -> Self {
            Self {
                inner: RwLock::new(GraphInner {
                    graph: StableDiGraph::new(),
                    index: HashMap::new(),
                    name_index: HashMap::new(),
                    edge_ids: std::collections::HashSet::new(),
                }),
            }
        }
    }

    fn norm_name(label: &str) -> String {
        cas::normalize_text(label).to_lowercase()
    }

    impl PetKnowledgeGraph {
        pub fn new() -> Self {
            Self::default()
        }

        /// Insert-or-merge a node by entity resolution (§4.8). A node merges into an
        /// existing one if (a) the id collides (content-addressed duplicate), or
        /// (b) the normalized (kind, name) already maps to a canonical id. On merge,
        /// provenance is unioned and the strongest confidence tier kept.
        pub fn upsert_node(&self, mut node: KnowledgeNode) -> String {
            let mut inner = self.inner.write();
            let name_key = (node.kind, norm_name(&node.label));

            // Resolve a canonical id: prefer an exact id collision, else a name match.
            let canonical = if inner.index.contains_key(&node.id) {
                Some(node.id.clone())
            } else {
                inner.name_index.get(&name_key).cloned()
            };

            if let Some(canon_id) = canonical {
                if let Some(&idx) = inner.index.get(&canon_id) {
                    if let Some(existing) = inner.graph.node_weight_mut(idx) {
                        // Union provenance.
                        for span in node.provenance.drain(..) {
                            if !existing.provenance.iter().any(|s| {
                                s.doc_id == span.doc_id
                                    && s.span_id == span.span_id
                                    && s.content_hash == span.content_hash
                            }) {
                                existing.provenance.push(span);
                            }
                        }
                        // Keep the strongest (lowest-ordinal) confidence tier.
                        if tier_rank(node.confidence) < tier_rank(existing.confidence) {
                            existing.confidence = node.confidence;
                        }
                    }
                    inner.name_index.entry(name_key).or_insert(canon_id.clone());
                    return canon_id;
                }
            }

            // Fresh node.
            let id = node.id.clone();
            let idx = inner.graph.add_node(node);
            inner.index.insert(id.clone(), idx);
            inner.name_index.entry(name_key).or_insert(id.clone());
            id
        }

        pub fn upsert_edge(&self, edge: KnowledgeEdge) {
            let mut inner = self.inner.write();
            if inner.edge_ids.contains(&edge.id) {
                return;
            }
            let (Some(&from), Some(&to)) = (inner.index.get(&edge.from), inner.index.get(&edge.to)) else {
                return; // endpoints must exist first
            };
            inner.edge_ids.insert(edge.id.clone());
            inner.graph.add_edge(from, to, edge);
        }

        pub fn node(&self, id: &str) -> Option<KnowledgeNode> {
            let inner = self.inner.read();
            inner.index.get(id).and_then(|&idx| inner.graph.node_weight(idx).cloned())
        }

        pub fn node_count(&self) -> usize {
            self.inner.read().graph.node_count()
        }

        pub fn edge_count(&self) -> usize {
            self.inner.read().graph.edge_count()
        }

        /// Ingest a parsed document: mint a content-addressed Paper node plus one
        /// content-addressed Claim node per non-empty section, with `MENTIONS`
        /// edges. Returns the claims (carrying their provenance spans). Idempotent:
        /// re-ingesting the same doc merges rather than duplicates.
        pub fn ingest_doc(&self, doc: &StructuredDoc) -> Vec<Claim> {
            let paper_id = cas::composite_id("paper", &[&doc.title, &doc.id]);
            let base_span = ProvenanceSpan {
                doc_id: doc.id.clone(),
                span_id: None,
                char_range: None,
                citation: None,
                content_hash: doc.source.content_hash.clone(),
                evidence_blob: doc.blob.clone(),
                provenance: doc.source.provenance.clone(),
            };
            self.upsert_node(KnowledgeNode {
                id: paper_id.clone(),
                kind: NodeKind::Paper,
                label: doc.title.clone(),
                confidence: ConfidenceTier::Extracted,
                provenance: vec![base_span.clone()],
                created_at_ms: now_ms(),
            });

            let mut claims = Vec::new();
            for section in &doc.sections {
                if section.text.trim().is_empty() {
                    continue;
                }
                let claim_id = cas::composite_id("claim", &[&section.text, &doc.id]);
                // The claim's evidence receipt MUST address the same canonical bytes
                // the claim id is derived from. When the section was pinned, use its
                // own per-section receipt (normalized section bytes); fall back to the
                // doc-level receipt only when no section-level pin exists.
                let (content_hash, evidence_blob) = match &section.evidence {
                    Some(ev) => (Some(ev.content_hash.clone()), Some(ev.blob.clone())),
                    None => (doc.source.content_hash.clone(), doc.blob.clone()),
                };
                let span = ProvenanceSpan {
                    doc_id: doc.id.clone(),
                    span_id: section.spans.first().map(|s| s.id.clone()),
                    char_range: section.spans.first().map(|s| (s.start_char, s.end_char)),
                    citation: None,
                    content_hash,
                    evidence_blob,
                    provenance: doc.source.provenance.clone(),
                };
                self.upsert_node(KnowledgeNode {
                    id: claim_id.clone(),
                    kind: NodeKind::Claim,
                    label: section.text.chars().take(160).collect(),
                    confidence: ConfidenceTier::Extracted,
                    provenance: vec![span.clone()],
                    created_at_ms: now_ms(),
                });
                self.upsert_edge(KnowledgeEdge {
                    id: cas::composite_id("edge", &[&paper_id, &claim_id, "mentions"]),
                    from: paper_id.clone(),
                    to: claim_id.clone(),
                    kind: EdgeKind::Mentions,
                    confidence: 0.8,
                    provenance: vec![span.clone()],
                });
                claims.push(Claim {
                    id: claim_id,
                    text: section.text.clone(),
                    provenance: span,
                    confidence: ConfidenceTier::Extracted,
                });
            }
            claims
        }

        /// Serialize the entire graph to a JSONL file (one node/edge per line),
        /// surviving process exit (§4.3).
        pub fn save_jsonl(&self, path: &std::path::Path) -> Result<()> {
            use std::io::Write;
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let inner = self.inner.read();
            let mut file = std::fs::File::create(path)?;
            for node in inner.graph.node_weights() {
                let line = serde_json::to_string(&GraphLine::Node(node.clone()))?;
                writeln!(file, "{line}")?;
            }
            for edge in inner.graph.edge_weights() {
                let line = serde_json::to_string(&GraphLine::Edge(edge.clone()))?;
                writeln!(file, "{line}")?;
            }
            file.sync_data()?;
            Ok(())
        }

        /// Load a graph from a JSONL file produced by [`Self::save_jsonl`]. Nodes are
        /// loaded first (in file order) so edges always find their endpoints.
        pub fn load_jsonl(path: &std::path::Path) -> Result<Self> {
            use std::io::BufRead;
            let graph = Self::new();
            if !path.exists() {
                return Ok(graph);
            }
            let file = std::fs::File::open(path)?;
            let reader = std::io::BufReader::new(file);
            let mut edges = Vec::new();
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<GraphLine>(&line)? {
                    GraphLine::Node(n) => {
                        // Insert verbatim (preserve ids) — bypass name-merge so a
                        // saved graph reloads byte-faithfully.
                        let mut inner = graph.inner.write();
                        let name_key = (n.kind, norm_name(&n.label));
                        let id = n.id.clone();
                        let idx = inner.graph.add_node(n);
                        inner.index.insert(id.clone(), idx);
                        inner.name_index.entry(name_key).or_insert(id);
                    }
                    GraphLine::Edge(e) => edges.push(e),
                }
            }
            for e in edges {
                graph.upsert_edge(e);
            }
            Ok(graph)
        }

        fn local(&self, seeds: &[String], hops: usize) -> QueryResult {
            let inner = self.inner.read();
            let mut visited = std::collections::HashSet::new();
            let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
            for s in seeds {
                if let Some(&idx) = inner.index.get(s) {
                    if visited.insert(idx) {
                        queue.push_back((idx, 0));
                    }
                }
            }
            let mut edges = Vec::new();
            while let Some((idx, depth)) = queue.pop_front() {
                if depth >= hops {
                    continue;
                }
                for e in inner
                    .graph
                    .edges_directed(idx, Direction::Outgoing)
                    .chain(inner.graph.edges_directed(idx, Direction::Incoming))
                {
                    edges.push(e.weight().clone());
                    let nbr = if e.source() == idx { e.target() } else { e.source() };
                    if visited.insert(nbr) {
                        queue.push_back((nbr, depth + 1));
                    }
                }
            }
            let nodes = visited.iter().filter_map(|&idx| inner.graph.node_weight(idx).cloned()).collect();
            dedup_result(QueryResult { nodes, edges })
        }

        fn global(&self, kind: NodeKind, limit: usize) -> QueryResult {
            let mut nodes: Vec<KnowledgeNode> = self.nodes_by_kind(kind);
            nodes.sort_by_key(|node| std::cmp::Reverse(node.created_at_ms));
            nodes.truncate(limit);
            QueryResult { nodes, edges: Vec::new() }
        }

        fn path(&self, from: &str, to: &str) -> QueryResult {
            let inner = self.inner.read();
            let (Some(&src), Some(&dst)) = (inner.index.get(from), inner.index.get(to)) else {
                return QueryResult { nodes: Vec::new(), edges: Vec::new() };
            };
            // BFS shortest path by edge count, following edges as undirected for
            // reachability but recording the typed edge traversed.
            let mut prev: HashMap<NodeIndex, (NodeIndex, KnowledgeEdge)> = HashMap::new();
            let mut visited = std::collections::HashSet::new();
            let mut queue = VecDeque::new();
            visited.insert(src);
            queue.push_back(src);
            while let Some(idx) = queue.pop_front() {
                if idx == dst {
                    break;
                }
                for e in inner
                    .graph
                    .edges_directed(idx, Direction::Outgoing)
                    .chain(inner.graph.edges_directed(idx, Direction::Incoming))
                {
                    let nbr = if e.source() == idx { e.target() } else { e.source() };
                    if visited.insert(nbr) {
                        prev.insert(nbr, (idx, e.weight().clone()));
                        queue.push_back(nbr);
                    }
                }
            }
            if !visited.contains(&dst) {
                return QueryResult { nodes: Vec::new(), edges: Vec::new() };
            }
            // Walk back from dst to src.
            let mut nodes = Vec::new();
            let mut edges = Vec::new();
            let mut cur = dst;
            nodes.push(inner.graph.node_weight(cur).cloned().unwrap());
            while cur != src {
                let (p, e) = prev.get(&cur).cloned().unwrap();
                edges.push(e);
                nodes.push(inner.graph.node_weight(p).cloned().unwrap());
                cur = p;
            }
            nodes.reverse();
            edges.reverse();
            QueryResult { nodes, edges }
        }
    }

    fn tier_rank(t: ConfidenceTier) -> u8 {
        match t {
            ConfidenceTier::Measured => 0,
            ConfidenceTier::Extracted => 1,
            ConfidenceTier::Inferred => 2,
            ConfidenceTier::Speculative => 3,
        }
    }

    fn dedup_result(mut r: QueryResult) -> QueryResult {
        let mut seen_e = std::collections::HashSet::new();
        r.edges.retain(|e| seen_e.insert(e.id.clone()));
        r
    }

    #[derive(Serialize, Deserialize)]
    #[serde(tag = "rec")]
    enum GraphLine {
        Node(KnowledgeNode),
        Edge(KnowledgeEdge),
    }

    impl KnowledgeGraph for PetKnowledgeGraph {
        fn add_node(&self, node: KnowledgeNode) {
            self.upsert_node(node);
        }

        fn add_edge(&self, edge: KnowledgeEdge) {
            self.upsert_edge(edge);
        }

        fn nodes_by_kind(&self, kind: NodeKind) -> Vec<KnowledgeNode> {
            let inner = self.inner.read();
            inner.graph.node_weights().filter(|n| n.kind == kind).cloned().collect()
        }

        fn edges_from(&self, node_id: &str) -> Vec<KnowledgeEdge> {
            let inner = self.inner.read();
            let Some(&idx) = inner.index.get(node_id) else {
                return Vec::new();
            };
            inner.graph.edges_directed(idx, Direction::Outgoing).map(|e| e.weight().clone()).collect()
        }

        fn query(&self, q: &GraphQuery) -> QueryResult {
            match q {
                GraphQuery::Local { seeds, hops } => self.local(seeds, *hops),
                GraphQuery::Global { kind, limit } => self.global(*kind, *limit),
                GraphQuery::Path { from, to } => self.path(from, to),
            }
        }
    }

    /// Back-compat alias: the prior in-memory graph type name. Tests and downstream
    /// callers that used `InMemoryKnowledgeGraph` keep working; it is now the real
    /// petgraph store.
    pub type InMemoryKnowledgeGraph = PetKnowledgeGraph;

    impl PetKnowledgeGraph {
        /// Back-compat shim for the prior `ingest_doc_shell` name.
        pub fn ingest_doc_shell(&self, doc: &StructuredDoc) -> Vec<Claim> {
            self.ingest_doc(doc)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::ingest::{DocSection, DocSpan, SourceQuality, SourceRecord, SourceType};

        fn doc(id: &str, title: &str, body: &str) -> StructuredDoc {
            StructuredDoc {
                id: id.to_string(),
                source: SourceRecord {
                    id: id.to_string(),
                    source_type: SourceType::PdfLocal,
                    title: title.to_string(),
                    uri: format!("memory://{id}"),
                    content_hash: Some("hash".to_string()),
                    quality: SourceQuality::default(),
                    provenance: Provenance::trusted("test"),
                },
                title: title.to_string(),
                sections: vec![DocSection {
                    heading: "Abstract".to_string(),
                    text: body.to_string(),
                    spans: vec![DocSpan { id: "s0".to_string(), start_char: 0, end_char: body.len() }],
                    evidence: None,
                }],
                references: Vec::new(),
                blob: None,
            }
        }

        #[test]
        fn ingest_is_idempotent_via_content_addressing() {
            let g = PetKnowledgeGraph::new();
            let d = doc("doc1", "Paged Attention", "Paged attention improves reuse.");
            let c1 = g.ingest_doc(&d);
            let n1 = g.node_count();
            let c2 = g.ingest_doc(&d);
            // Same ids the second time; node count unchanged.
            assert_eq!(c1[0].id, c2[0].id);
            assert_eq!(g.node_count(), n1);
        }

        #[test]
        fn claim_evidence_addresses_same_canonical_bytes_as_claim_id() {
            use crate::ingest::SectionEvidence;
            use hide_core::persistence::{DynBlobStore, InMemoryBlobStore};
            use std::sync::Arc;

            let cas: DynBlobStore = Arc::new(InMemoryBlobStore::default());
            let mut d = doc("doc1", "Paged Attention", "  Paged   attention\n improves reuse.  ");
            // Pin the section's CANONICAL bytes (what pin_doc_evidence does) and
            // stamp the receipt onto the section.
            let section_text = d.sections[0].text.clone();
            let (blob, hash) = cas::pin_canonical_evidence(&cas, &section_text).unwrap();
            d.sections[0].evidence = Some(SectionEvidence { blob: blob.clone(), content_hash: hash.clone() });

            let claims = PetKnowledgeGraph::new().ingest_doc(&d);
            let claim = &claims[0];

            // (1) The claim id is content-addressed over the same normalized text the
            //     evidence receipt hashes — both reduce to canonical section bytes.
            assert_eq!(claim.id, cas::composite_id("claim", &[&section_text, &d.id]));
            // (2) The recorded receipt hash equals blake3 of the canonical bytes...
            let canon = cas::canonical_evidence_bytes(&section_text);
            assert_eq!(claim.provenance.content_hash.as_deref(), Some(cas::blake3_hex(&canon).as_str()));
            // ...and equals the per-section pin's hash (no doc-level divergence).
            assert_eq!(claim.provenance.content_hash.as_deref(), Some(hash.as_str()));
            assert_eq!(claim.provenance.evidence_blob.as_ref(), Some(&blob));

            // (3) Re-verification re-hashes the SAME blob bytes against the SAME
            //     receipt → Intact (no false positive).
            let check = cas::verify_evidence(
                &cas,
                claim.provenance.evidence_blob.as_ref().unwrap(),
                claim.provenance.content_hash.as_ref().unwrap(),
            )
            .unwrap();
            assert!(check.is_intact());
        }

        #[test]
        fn entity_resolution_merges_by_name() {
            let g = PetKnowledgeGraph::new();
            g.upsert_node(KnowledgeNode {
                id: "concept:a".into(),
                kind: NodeKind::Concept,
                label: "KV Cache".into(),
                confidence: ConfidenceTier::Inferred,
                provenance: vec![],
                created_at_ms: 1,
            });
            // Different id, same normalized name+kind → merges.
            g.upsert_node(KnowledgeNode {
                id: "concept:b".into(),
                kind: NodeKind::Concept,
                label: "kv   cache".into(),
                confidence: ConfidenceTier::Measured,
                provenance: vec![],
                created_at_ms: 2,
            });
            assert_eq!(g.nodes_by_kind(NodeKind::Concept).len(), 1);
            // Stronger tier kept.
            assert_eq!(g.node("concept:a").unwrap().confidence, ConfidenceTier::Measured);
        }

        #[test]
        fn local_global_path_queries() {
            let g = PetKnowledgeGraph::new();
            g.ingest_doc(&doc("d1", "A", "alpha claim text here"));
            let paper = g.nodes_by_kind(NodeKind::Paper)[0].id.clone();
            let claim = g.nodes_by_kind(NodeKind::Claim)[0].id.clone();

            let local = g.query(&GraphQuery::Local { seeds: vec![paper.clone()], hops: 1 });
            assert!(local.nodes.iter().any(|n| n.id == claim));

            let global = g.query(&GraphQuery::Global { kind: NodeKind::Paper, limit: 10 });
            assert_eq!(global.nodes.len(), 1);

            let path = g.query(&GraphQuery::Path { from: paper.clone(), to: claim.clone() });
            assert_eq!(path.nodes.len(), 2);
            assert_eq!(path.edges.len(), 1);
        }

        #[test]
        fn graph_persists_to_jsonl_and_reloads() {
            let g = PetKnowledgeGraph::new();
            g.ingest_doc(&doc("d1", "Persist Me", "a durable claim"));
            let (nc, ec) = (g.node_count(), g.edge_count());
            let dir = std::env::temp_dir().join(format!("hawking_kg_{}", now_ms()));
            let path = dir.join("graph.jsonl");
            g.save_jsonl(&path).unwrap();
            let reloaded = PetKnowledgeGraph::load_jsonl(&path).unwrap();
            assert_eq!(reloaded.node_count(), nc);
            assert_eq!(reloaded.edge_count(), ec);
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}
#[rustfmt::skip]
pub mod litmap {
    //! Citation / literature mapping over the knowledge graph (bible ch.08 §4.9).
    //!
    //! `build_literature_map` walks the graph from the Paper nodes, clusters them by
    //! shared claim/concept neighborhoods (a label-propagation-style grouping over
    //! the real petgraph store), and surfaces coverage gaps (sub-questions /
    //! concepts with thin support). This is a local, owned equivalent of
    //! Connected-Papers / Litmaps — seeded by what *you* ingested.

    use crate::kg::{EdgeKind, KnowledgeGraph, KnowledgeNode, NodeKind, PetKnowledgeGraph};
    use serde::{Deserialize, Serialize};
    use std::collections::{BTreeMap, BTreeSet};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct LiteratureMap {
        pub topic: String,
        pub papers: Vec<KnowledgeNode>,
        pub clusters: Vec<LiteratureCluster>,
        pub gaps: Vec<ResearchGap>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct LiteratureCluster {
        pub id: String,
        pub label: String,
        pub node_ids: Vec<String>,
        pub summary: String,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ResearchGap {
        pub id: String,
        pub description: String,
        pub supporting_node_ids: Vec<String>,
    }

    /// Build a literature map from the current graph. Papers that share at least one
    /// claim/concept neighbor are grouped into the same cluster (transitive closure
    /// over the shared-neighbor relation — a connected-components clustering).
    pub fn build_literature_map(graph: &PetKnowledgeGraph, topic: impl Into<String>) -> LiteratureMap {
        let topic = topic.into();
        let papers = graph.nodes_by_kind(NodeKind::Paper);

        // Map each paper → the set of its claim/concept neighbor ids.
        let mut paper_neighbors: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for p in &papers {
            let mut nbrs = BTreeSet::new();
            for e in graph.edges_from(&p.id) {
                if matches!(e.kind, EdgeKind::Mentions | EdgeKind::Cites | EdgeKind::UsesDataset | EdgeKind::Supports) {
                    nbrs.insert(e.to);
                }
            }
            paper_neighbors.insert(p.id.clone(), nbrs);
        }

        // Union-find over papers that share ≥1 neighbor.
        let ids: Vec<String> = papers.iter().map(|p| p.id.clone()).collect();
        let mut uf = UnionFind::new(&ids);
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let a = &paper_neighbors[&ids[i]];
                let b = &paper_neighbors[&ids[j]];
                if a.intersection(b).next().is_some() {
                    uf.union(&ids[i], &ids[j]);
                }
            }
        }

        // Group by root.
        let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for id in &ids {
            groups.entry(uf.find(id)).or_default().push(id.clone());
        }

        let label_of = |id: &str| -> String {
            papers.iter().find(|p| p.id == id).map(|p| p.label.clone()).unwrap_or_else(|| id.to_string())
        };

        let mut clusters = Vec::new();
        for (i, (_root, members)) in groups.iter().enumerate() {
            let label = members.first().map(|m| label_of(m)).unwrap_or_else(|| format!("cluster {i}"));
            clusters.push(LiteratureCluster {
                id: format!("cluster:{i}"),
                label: label.chars().take(60).collect(),
                node_ids: members.clone(),
                summary: format!("{} paper(s) sharing claim/concept neighbors", members.len()),
            });
        }

        // Gaps: concepts mentioned by exactly one paper (thin support).
        let mut concept_support: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for c in graph.nodes_by_kind(NodeKind::Concept) {
            concept_support.entry(c.id.clone()).or_default();
        }
        for (paper, nbrs) in &paper_neighbors {
            for n in nbrs {
                if let Some(v) = concept_support.get_mut(n) {
                    v.push(paper.clone());
                }
            }
        }
        let gaps = concept_support
            .into_iter()
            .filter(|(_, s)| s.len() <= 1)
            .map(|(concept, support)| ResearchGap {
                id: format!("gap:{concept}"),
                description: format!("Concept {concept} has thin coverage ({} paper)", support.len()),
                supporting_node_ids: support,
            })
            .collect();

        LiteratureMap { topic, papers, clusters, gaps }
    }

    /// Compare N papers by their shared and unique claim/concept neighbors.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PaperComparison {
        pub paper_ids: Vec<String>,
        pub shared: Vec<String>,
        pub unique: BTreeMap<String, Vec<String>>,
    }

    pub fn compare_papers(graph: &PetKnowledgeGraph, paper_ids: &[String]) -> PaperComparison {
        let neighbor_sets: BTreeMap<String, BTreeSet<String>> = paper_ids
            .iter()
            .map(|id| {
                let nbrs: BTreeSet<String> = graph.edges_from(id).into_iter().map(|e| e.to).collect();
                (id.clone(), nbrs)
            })
            .collect();
        let shared: BTreeSet<String> = neighbor_sets
            .values()
            .cloned()
            .reduce(|acc: BTreeSet<String>, s| acc.intersection(&s).cloned().collect())
            .unwrap_or_default();
        let unique: BTreeMap<String, Vec<String>> = neighbor_sets
            .iter()
            .map(|(id, s)| {
                let u: Vec<String> = s.difference(&shared).cloned().collect();
                (id.clone(), u)
            })
            .collect();
        PaperComparison { paper_ids: paper_ids.to_vec(), shared: shared.into_iter().collect(), unique }
    }

    // ── tiny union-find ──
    struct UnionFind {
        parent: BTreeMap<String, String>,
    }

    impl UnionFind {
        fn new(ids: &[String]) -> Self {
            Self { parent: ids.iter().map(|i| (i.clone(), i.clone())).collect() }
        }
        fn find(&self, x: &str) -> String {
            let mut cur = x.to_string();
            while let Some(p) = self.parent.get(&cur) {
                if p == &cur {
                    break;
                }
                cur = p.clone();
            }
            cur
        }
        fn union(&mut self, a: &str, b: &str) {
            let ra = self.find(a);
            let rb = self.find(b);
            if ra != rb {
                self.parent.insert(ra, rb);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::ingest::structured_doc_from_text;
        use crate::ingest::{SourceQuality, SourceType};
        use hide_core::types::Provenance;

        fn ingest(graph: &PetKnowledgeGraph, id: &str, title: &str, body: &str) {
            let doc = structured_doc_from_text(
                SourceType::PdfLocal,
                id,
                &format!("memory://{id}"),
                title,
                body,
                Provenance::trusted("t"),
                SourceQuality::default(),
            );
            graph.ingest_doc(&doc);
        }

        #[test]
        fn map_clusters_papers_and_finds_gaps() {
            let graph = PetKnowledgeGraph::new();
            // Two papers sharing nothing (different claim text) → two clusters.
            ingest(&graph, "a", "A", "alpha distinct statement one");
            ingest(&graph, "b", "B", "beta distinct statement two");
            let map = build_literature_map(&graph, "topic");
            assert_eq!(map.papers.len(), 2);
            assert_eq!(map.clusters.len(), 2);
        }

        #[test]
        fn shared_concept_merges_into_one_cluster() {
            use crate::kg::{ConfidenceTier, EdgeKind, KnowledgeEdge, KnowledgeGraph, KnowledgeNode, NodeKind};
            let graph = PetKnowledgeGraph::new();
            // Two papers whose claim text differs (→ distinct claims, per §4.2.1),
            // but which both MENTION the same Concept node → one cluster.
            ingest(&graph, "a", "A", "alpha statement about attention");
            ingest(&graph, "b", "B", "beta statement about attention");
            let papers = graph.nodes_by_kind(NodeKind::Paper);
            graph.upsert_node(KnowledgeNode {
                id: "concept:attention".into(),
                kind: NodeKind::Concept,
                label: "attention".into(),
                confidence: ConfidenceTier::Inferred,
                provenance: vec![],
                created_at_ms: 1,
            });
            for p in &papers {
                graph.upsert_edge(KnowledgeEdge {
                    id: format!("edge:{}:concept", p.id),
                    from: p.id.clone(),
                    to: "concept:attention".into(),
                    kind: EdgeKind::Mentions,
                    confidence: 0.9,
                    provenance: vec![],
                });
            }
            let map = build_literature_map(&graph, "topic");
            assert_eq!(map.papers.len(), 2);
            assert_eq!(map.clusters.len(), 1);
        }
    }
}
#[rustfmt::skip]
pub mod pipeline {
    //! The research pipeline FSM (bible ch.08 §4.6).
    //!
    //! `PlanScope → FanOut → Fetch → Read → Verify → Synthesize → Persist → Reflect`.
    //! Each transition appends to a per-event checkpoint journal so an overnight run
    //! resumes from the last completed state without re-fetching (CAS dedup) or
    //! re-extracting (content-addressed nodes).
    //!
    //! What is REAL here: PlanScope decomposes the topic into sub-questions via the
    //! [`RuntimeClient`]; Triage dedups candidates by content hash; Read pins
    //! evidence bytes in the CAS and ingests content-addressed nodes; Verify runs
    //! the adversarial verifier *with* CAS citation re-verification; Synthesize
    //! assembles a cited report from the verified claim set via the model; Persist
    //! emits a Report node + Findings; Reflect gates a bounded re-loop on coverage.

    use crate::cas;
    use crate::checkpoint::{CheckpointKind, DynCheckpointLedger, InMemoryCheckpointLedger, RunJournal};
    use crate::ingest::{SourceAdapter, SourceQuery, SourceRecord, StructuredDoc};
    use crate::kg::{
        Claim, ConfidenceTier, EdgeKind, KnowledgeEdge, KnowledgeNode, NodeKind, PetKnowledgeGraph, ProvenanceSpan,
    };
    use crate::runtime_client::{stub_runtime, ChatRequest, RuntimeClient};
    use crate::verify::{AdversarialVerifier, ClaimVerification};
    use hide_core::ids::{now_ms, RunId};
    use hide_core::persistence::{DynBlobStore, InMemoryBlobStore};
    use hide_core::types::Provenance;
    use hide_core::{HideError, Result};
    use std::collections::HashSet;
    use std::sync::Arc;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ResearchRun {
        pub id: RunId,
        pub topic: String,
        pub state: ResearchState,
        pub created_at_ms: u64,
        pub seed: u64,
        pub round: u32,
        pub sub_questions: Vec<String>,
        pub docs_read: usize,
        pub claims: Vec<Claim>,
        pub verifications: Vec<ClaimVerification>,
        /// The cited report produced in Synthesize.
        pub report: Option<String>,
        /// Findings minted in Persist.
        pub findings: Vec<Finding>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Finding {
        pub id: String,
        pub summary: String,
        pub claim_ids: Vec<String>,
        pub actionable: bool,
    }

    impl ResearchRun {
        pub fn new(topic: impl Into<String>) -> Self {
            let topic = topic.into();
            let seed = stable_seed(&topic);
            Self {
                id: RunId::new(),
                topic,
                state: ResearchState::PlanScope,
                created_at_ms: now_ms(),
                seed,
                round: 0,
                sub_questions: Vec::new(),
                docs_read: 0,
                claims: Vec::new(),
                verifications: Vec::new(),
                report: None,
                findings: Vec::new(),
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ResearchState {
        PlanScope,
        FanOut,
        Fetch,
        Read,
        Verify,
        Synthesize,
        Persist,
        Reflect,
        Complete,
        Failed,
    }

    /// Tunables (the §9 dials). Sensible defaults; resumable runs honor `max_rounds`.
    #[derive(Debug, Clone)]
    pub struct ResearchBudget {
        pub read_budget: usize,
        pub max_rounds: u32,
        pub coverage_target: f32,
        pub dedup_cosine: f32,
    }

    impl Default for ResearchBudget {
        fn default() -> Self {
            Self { read_budget: 8, max_rounds: 2, coverage_target: 0.8, dedup_cosine: 0.95 }
        }
    }

    pub struct ResearchPipeline {
        adapters: Vec<Arc<dyn SourceAdapter>>,
        graph: Arc<PetKnowledgeGraph>,
        runtime: Arc<dyn RuntimeClient>,
        cas: DynBlobStore,
        checkpoints: DynCheckpointLedger,
        budget: ResearchBudget,
    }

    impl ResearchPipeline {
        /// Construct with an explicit runtime + CAS + checkpoint ledger.
        pub fn new(
            graph: Arc<PetKnowledgeGraph>,
            runtime: Arc<dyn RuntimeClient>,
            cas: DynBlobStore,
            checkpoints: DynCheckpointLedger,
        ) -> Self {
            Self { adapters: Vec::new(), graph, runtime, cas, checkpoints, budget: ResearchBudget::default() }
        }

        /// Convenience for tests / offline runs: stub runtime, in-memory CAS +
        /// checkpoint ledger.
        pub fn in_memory(graph: Arc<PetKnowledgeGraph>) -> Self {
            Self::new(
                graph,
                stub_runtime(
                    "Synthesis: paged attention reduces KV cache memory waste [c]. \
                     This is corroborated across the retrieved sources.",
                ),
                Arc::new(InMemoryBlobStore::default()),
                Arc::new(InMemoryCheckpointLedger::new()),
            )
        }

        pub fn add_adapter(&mut self, adapter: Arc<dyn SourceAdapter>) {
            self.adapters.push(adapter);
        }

        pub fn with_budget(mut self, budget: ResearchBudget) -> Self {
            self.budget = budget;
            self
        }

        pub async fn run_once(&self, topic: impl Into<String>, limit: usize) -> Result<ResearchRun> {
            let mut run = ResearchRun::new(topic);
            let mut journal = RunJournal::new(run.id.clone(), self.checkpoints.clone());
            journal.record(CheckpointKind::Opened { topic: run.topic.clone(), seed: run.seed })?;
            loop {
                self.step(&mut run, limit, &mut journal).await?;
                if matches!(run.state, ResearchState::Complete | ResearchState::Failed) {
                    break;
                }
            }
            Ok(run)
        }

        pub async fn step(&self, run: &mut ResearchRun, limit: usize, journal: &mut RunJournal) -> Result<()> {
            journal.record(CheckpointKind::State { state: format!("{:?}", run.state) })?;
            match run.state {
                // ── PlanScope: decompose the topic into sub-questions via the model.
                ResearchState::PlanScope => {
                    run.sub_questions = self.plan_scope(&run.topic).await?;
                    run.state = ResearchState::FanOut;
                }
                // ── FanOut: ensure we have adapters (real fan-out happens in Fetch).
                ResearchState::FanOut => {
                    if self.adapters.is_empty() {
                        return Err(HideError::InvalidState("research pipeline has no source adapters".to_string()));
                    }
                    run.state = ResearchState::Fetch;
                }
                // ── Fetch: search every adapter over topic + sub-questions, then
                //    Triage (dedup by content hash) before paying parse cost.
                ResearchState::Fetch => {
                    let mut queries = vec![run.topic.clone()];
                    queries.extend(run.sub_questions.iter().cloned());
                    let mut records: Vec<SourceRecord> = Vec::new();
                    for q in &queries {
                        let query = SourceQuery { query: q.clone(), limit, source_types: Vec::new() };
                        for adapter in &self.adapters {
                            records.extend(adapter.search(&query).await?);
                        }
                    }
                    let selected = self.triage(records);
                    let already = self.checkpoints.fetched_hashes(run.id.as_str())?;
                    let mut docs: Vec<StructuredDoc> = Vec::new();
                    for record in selected.into_iter().take(self.budget.read_budget) {
                        let mut doc = self.fetch_doc(&record).await?;
                        // Skip if we already fetched this content in a prior round.
                        if let Some(h) = &doc.source.content_hash {
                            if already.contains(h) {
                                continue;
                            }
                        }
                        self.pin_doc_evidence(&mut doc)?;
                        journal.record(CheckpointKind::Fetched {
                            doc_id: doc.id.clone(),
                            content_hash: doc.source.content_hash.clone(),
                        })?;
                        docs.push(doc);
                    }
                    // Stash on the run via the graph ingest in Read.
                    run.claims.clear();
                    for doc in &docs {
                        let claims = self.graph.ingest_doc(doc);
                        run.docs_read += 1;
                        run.claims.extend(claims);
                    }
                    run.state = ResearchState::Read;
                }
                // ── Read: nodes already ingested in Fetch; nothing extra needed for
                //    the deterministic path (LLM entity extraction is a seam).
                ResearchState::Read => run.state = ResearchState::Verify,
                // ── Verify: adversarial, with CAS citation re-verification.
                ResearchState::Verify => {
                    let claims = run.claims.clone();
                    run.verifications = claims
                        .iter()
                        .map(|c| {
                            AdversarialVerifier::verify_with_cas(c, &claims, &self.cas)
                                .unwrap_or_else(|_| AdversarialVerifier::verify(c, &claims))
                        })
                        .collect();
                    run.state = ResearchState::Synthesize;
                }
                // ── Synthesize: a cited report from the VERIFIED claim slice.
                ResearchState::Synthesize => {
                    run.report = Some(self.synthesize(run).await?);
                    run.state = ResearchState::Persist;
                }
                // ── Persist: Report node + Findings into the graph.
                ResearchState::Persist => {
                    self.persist(run);
                    run.state = ResearchState::Reflect;
                }
                // ── Reflect: bounded coverage/novelty re-loop.
                ResearchState::Reflect => {
                    run.round += 1;
                    let coverage = self.assess_coverage(run);
                    let novelty = if run.docs_read > 0 { 1.0 } else { 0.0 };
                    journal.record(CheckpointKind::Round { round: run.round, coverage, novelty })?;
                    if coverage < self.budget.coverage_target && run.round < self.budget.max_rounds && novelty > 0.0 {
                        run.state = ResearchState::Fetch; // another round, gaps only
                    } else {
                        journal.record(CheckpointKind::Done { docs_read: run.docs_read, claims: run.claims.len() })?;
                        run.state = ResearchState::Complete;
                    }
                }
                ResearchState::Complete | ResearchState::Failed => {}
            }
            Ok(())
        }

        // ── PlanScope: ask the model to decompose, fall back to a deterministic split.
        async fn plan_scope(&self, topic: &str) -> Result<Vec<String>> {
            let prompt = format!(
                "Decompose this research topic into 2-4 focused sub-questions, one per line, \
                 no numbering:\n\n{topic}"
            );
            let raw =
                self.runtime.chat(ChatRequest::new("plan", prompt).with_max_tokens(256)).await.unwrap_or_default();
            let mut subs: Vec<String> = raw
                .lines()
                .map(|l| l.trim_start_matches(['-', '*', '•', ' ']).trim().to_string())
                .filter(|l| l.len() > 5)
                .take(4)
                .collect();
            if subs.is_empty() {
                // Deterministic fallback so the pipeline never stalls offline.
                subs = vec![format!("What is {topic}?"), format!("What are the limitations of {topic}?")];
            }
            Ok(subs)
        }

        // ── Triage: dedup candidate records by content hash, then by canonical uri.
        fn triage(&self, records: Vec<SourceRecord>) -> Vec<SourceRecord> {
            let mut seen_hash: HashSet<String> = HashSet::new();
            let mut seen_uri: HashSet<String> = HashSet::new();
            let mut out = Vec::new();
            for r in records {
                if let Some(h) = &r.content_hash {
                    if !seen_hash.insert(h.clone()) {
                        continue;
                    }
                }
                let uri_key = normalize_uri(&r.uri);
                if !seen_uri.insert(uri_key) {
                    continue;
                }
                out.push(r);
            }
            // Highest source-quality first.
            out.sort_by(|a, b| b.quality.score().partial_cmp(&a.quality.score()).unwrap_or(std::cmp::Ordering::Equal));
            out
        }

        async fn fetch_doc(&self, record: &SourceRecord) -> Result<StructuredDoc> {
            for adapter in &self.adapters {
                if adapter.source_type() == record.source_type {
                    return adapter.fetch(record).await;
                }
            }
            // Fall back to any adapter that can return it.
            for adapter in &self.adapters {
                if let Ok(doc) = adapter.fetch(record).await {
                    return Ok(doc);
                }
            }
            Err(HideError::NotFound(record.id.clone()))
        }

        // Pin evidence bytes in the CAS so citations re-verify. Each section is
        // pinned under its *own* canonical (normalized) bytes — the exact bytes its
        // claim node is content-addressed over — so the claim id, the evidence blob,
        // and the re-verification hash all agree on one byte source (§4.7.3). The
        // doc-level blob is the first non-empty section's receipt (a stable handle);
        // per-claim re-verification uses the per-section receipt, not the doc's.
        fn pin_doc_evidence(&self, doc: &mut StructuredDoc) -> Result<()> {
            let mut doc_blob = None;
            let mut doc_hash = None;
            for section in &mut doc.sections {
                if section.text.trim().is_empty() {
                    continue;
                }
                let (blob, hash) = cas::pin_canonical_evidence(&self.cas, &section.text)?;
                if doc_blob.is_none() {
                    doc_blob = Some(blob.clone());
                    doc_hash = Some(hash.clone());
                }
                section.evidence = Some(crate::ingest::SectionEvidence { blob, content_hash: hash });
            }
            doc.blob = doc_blob;
            if doc.source.content_hash.is_none() {
                doc.source.content_hash = doc_hash;
            }
            Ok(())
        }

        // ── Synthesize: every sentence should trace to a claim. We hand the model
        //    the verified claim list (with ids) and ask for a cited report.
        async fn synthesize(&self, run: &ResearchRun) -> Result<String> {
            let supported: Vec<&Claim> = run
                .claims
                .iter()
                .filter(|c| {
                    run.verifications
                        .iter()
                        .find(|v| v.claim_id == c.id)
                        .map(|v| {
                            !matches!(
                                v.status,
                                crate::verify::ClaimStatus::Refuted | crate::verify::ClaimStatus::Unverified
                            )
                        })
                        .unwrap_or(true)
                })
                .collect();
            let mut evidence = String::new();
            for c in &supported {
                evidence.push_str(&format!("[{}] {}\n", c.id, c.text));
            }
            let prompt = format!(
                "Write a concise, cited synthesis answering: {topic}\n\n\
                 Use only these claims; cite each statement with its [id]:\n\n{evidence}",
                topic = run.topic
            );
            let report = self.runtime.chat(ChatRequest::new("synthesize", prompt).with_max_tokens(1024)).await?;
            // Enforce: a report with no claims gets an explicit "insufficient evidence".
            if supported.is_empty() {
                return Ok(format!(
                    "Insufficient verified evidence for: {}\n(0 claims passed verification.)",
                    run.topic
                ));
            }
            Ok(report)
        }

        // ── Persist: a Report node + a Finding per supported claim cluster.
        fn persist(&self, run: &mut ResearchRun) {
            let report_text = run.report.clone().unwrap_or_default();
            let report_id = cas::composite_id("report", &[&run.topic, &report_text]);
            self.graph.upsert_node(KnowledgeNode {
                id: report_id.clone(),
                kind: NodeKind::Note,
                label: format!("Report: {}", run.topic),
                confidence: ConfidenceTier::Inferred,
                provenance: vec![ProvenanceSpan {
                    doc_id: run.id.0.clone(),
                    span_id: None,
                    char_range: None,
                    citation: None,
                    content_hash: None,
                    evidence_blob: None,
                    provenance: Provenance::trusted("synthesis").with_confidence(0.6),
                }],
                created_at_ms: now_ms(),
            });
            // One finding per supported/contradicted claim (actionable surface).
            let mut findings = Vec::new();
            for v in &run.verifications {
                if matches!(v.status, crate::verify::ClaimStatus::Supported | crate::verify::ClaimStatus::Contradicted)
                {
                    if let Some(claim) = run.claims.iter().find(|c| c.id == v.claim_id) {
                        let fid = cas::composite_id("finding", &[&claim.text]);
                        findings.push(Finding {
                            id: fid.clone(),
                            summary: claim.text.chars().take(160).collect(),
                            claim_ids: vec![claim.id.clone()],
                            actionable: matches!(v.status, crate::verify::ClaimStatus::Contradicted),
                        });
                        self.graph.upsert_edge(KnowledgeEdge {
                            id: cas::composite_id("edge", &[&report_id, &claim.id, "produced"]),
                            from: report_id.clone(),
                            to: claim.id.clone(),
                            kind: EdgeKind::Related,
                            confidence: v.independent_sources as f32 / (v.independent_sources as f32 + 1.0),
                            provenance: vec![claim.provenance.clone()],
                        });
                    }
                }
            }
            run.findings = findings;
        }

        // Crude coverage proxy: fraction of sub-questions whose key term appears in
        // some retrieved claim. Real, deterministic, no model needed.
        fn assess_coverage(&self, run: &ResearchRun) -> f32 {
            if run.sub_questions.is_empty() {
                return 1.0;
            }
            let claim_blob = run.claims.iter().map(|c| c.text.to_lowercase()).collect::<Vec<_>>().join(" ");
            let hit = run
                .sub_questions
                .iter()
                .filter(|q| q.to_lowercase().split_whitespace().filter(|w| w.len() > 4).any(|w| claim_blob.contains(w)))
                .count();
            hit as f32 / run.sub_questions.len() as f32
        }
    }

    fn normalize_uri(uri: &str) -> String {
        uri.trim()
            .trim_end_matches('/')
            .replace("/pdf/", "/abs/")
            .split(['?', '#'])
            .next()
            .unwrap_or(uri)
            .to_lowercase()
    }

    /// Deterministic per-topic seed (so a run is reproducible / re-derivable).
    fn stable_seed(topic: &str) -> u64 {
        let h = blake3::hash(topic.as_bytes());
        let b = h.as_bytes();
        u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::ingest::{structured_doc_from_text, InMemorySourceAdapter, SourceQuality, SourceType};
        use crate::kg::KnowledgeGraph;
        use crate::runtime_client::stub_runtime;

        fn mk_adapter() -> Arc<InMemorySourceAdapter> {
            let adapter = InMemorySourceAdapter::new("memory", SourceType::PdfLocal);
            let doc1 = structured_doc_from_text(
                SourceType::PdfLocal,
                "doc1",
                "memory://doc1",
                "KV cache research",
                "Paged attention reduces KV cache memory waste and improves throughput.",
                Provenance::trusted("test"),
                SourceQuality::default(),
            );
            let doc2 = structured_doc_from_text(
                SourceType::PdfLocal,
                "doc2",
                "memory://doc2",
                "Serving systems",
                "Paged attention reduces KV cache memory waste in serving systems.",
                Provenance::trusted("test2"),
                SourceQuality::default(),
            );
            let doc3 = structured_doc_from_text(
                SourceType::PdfLocal,
                "doc3",
                "memory://doc3",
                "Throughput study",
                "Paged attention reduces KV cache memory waste and raises throughput overall.",
                Provenance::trusted("test3"),
                SourceQuality::default(),
            );
            adapter.insert(doc1);
            adapter.insert(doc2);
            adapter.insert(doc3);
            Arc::new(adapter)
        }

        #[tokio::test]
        async fn pipeline_runs_full_fsm_to_completion() {
            let graph = Arc::new(PetKnowledgeGraph::new());
            let mut pipeline = ResearchPipeline::in_memory(graph.clone());
            pipeline.add_adapter(mk_adapter());
            let run = pipeline.run_once("KV cache", 4).await.unwrap();

            assert_eq!(run.state, ResearchState::Complete);
            assert!(run.docs_read >= 2);
            assert!(!run.claims.is_empty());
            assert!(!run.sub_questions.is_empty());
            assert!(run.report.is_some());
            // The two docs corroborate the same claim → at least one supported.
            assert!(run.verifications.iter().any(|v| v.status == crate::verify::ClaimStatus::Supported));
            // Findings were minted and persisted as a Report node.
            assert!(!run.findings.is_empty());
            assert!(!graph.nodes_by_kind(NodeKind::Note).is_empty());
        }

        #[tokio::test]
        async fn triage_dedups_identical_content() {
            let graph = Arc::new(PetKnowledgeGraph::new());
            let pipeline = ResearchPipeline::in_memory(graph);
            let r = |id: &str, hash: &str, uri: &str| SourceRecord {
                id: id.into(),
                source_type: SourceType::PdfLocal,
                title: "t".into(),
                uri: uri.into(),
                content_hash: Some(hash.into()),
                quality: SourceQuality::default(),
                provenance: Provenance::trusted("t"),
            };
            let out = pipeline.triage(vec![
                r("a", "h1", "http://x/abs/1"),
                r("b", "h1", "http://x/abs/2"), // dup content hash → dropped
                r("c", "h2", "http://x/pdf/1"), // same canonical uri as a → dropped
            ]);
            assert_eq!(out.len(), 1);
        }

        // Regression: the integrated pipeline path must pin EXACTLY the bytes a
        // claim is content-addressed over, so an untampered citation re-verifies
        // Intact and only a real mutation reports Tampered (no false positive).
        #[tokio::test]
        async fn pipeline_citation_reverifies_intact_then_tampered() {
            use crate::verify::{AdversarialVerifier, CitationCheck};
            use hide_core::persistence::FileBlobStore;

            // A real on-disk CAS so we can faithfully mutate the pinned evidence.
            let dir = std::env::temp_dir().join(format!("hawking_cas_{}", now_ms()));
            let store = FileBlobStore::open(&dir).unwrap();
            let cas: DynBlobStore = Arc::new(store);

            let graph = Arc::new(PetKnowledgeGraph::new());
            let mut pipeline = ResearchPipeline::new(
                graph,
                stub_runtime(
                    "Synthesis: paged attention reduces KV cache memory waste [c]. \
                     This is corroborated across the retrieved sources.",
                ),
                cas.clone(),
                Arc::new(InMemoryCheckpointLedger::new()),
            );
            pipeline.add_adapter(mk_adapter());
            let run = pipeline.run_once("KV cache", 4).await.unwrap();
            assert_eq!(run.state, ResearchState::Complete);
            assert!(run.docs_read >= 2, "expected docs to be fetched");
            assert!(!run.claims.is_empty());

            // Every claim that was pinned must re-verify Intact through the SAME path
            // production uses — not a single one should be a false-positive Tampered.
            let mut checked = 0usize;
            for claim in &run.claims {
                if claim.provenance.evidence_blob.is_none() {
                    continue;
                }
                let v = AdversarialVerifier::verify_with_cas(claim, &run.claims, &cas).unwrap();
                assert_eq!(
                    v.citation_check,
                    CitationCheck::Intact,
                    "untampered claim {} falsely flagged: {:?}",
                    claim.id,
                    v.citation_check
                );
                checked += 1;
            }
            assert!(checked > 0, "expected at least one pinned claim to re-verify");

            // Now deliberately mutate the on-disk evidence bytes of one claim's blob
            // and confirm the re-check flips to Tampered (a REAL change is caught).
            let target = run.claims.iter().find(|c| c.provenance.evidence_blob.is_some()).unwrap();
            let blob = target.provenance.evidence_blob.clone().unwrap();
            // FileBlobStore layout: <root>/<hash[..2]>/<hash>.
            let path = dir.join(&blob.hash[..2]).join(&blob.hash);
            assert!(path.exists(), "evidence blob must be on disk");
            std::fs::write(&path, b"tampered-evidence").unwrap();

            let vt = AdversarialVerifier::verify_with_cas(target, &run.claims, &cas).unwrap();
            assert_eq!(vt.citation_check, CitationCheck::Tampered);

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[tokio::test]
        async fn synthesize_reports_insufficient_evidence_when_empty() {
            let graph = Arc::new(PetKnowledgeGraph::new());
            let pipeline = ResearchPipeline::new(
                graph,
                stub_runtime("ignored"),
                Arc::new(InMemoryBlobStore::default()),
                Arc::new(InMemoryCheckpointLedger::new()),
            );
            let run = ResearchRun::new("empty topic");
            let out = pipeline.synthesize(&run).await.unwrap();
            assert!(out.contains("Insufficient verified evidence"));
        }
    }
}
#[rustfmt::skip]
pub mod run_ledger {
    use crate::pipeline::{ResearchRun, ResearchState};
    use hide_core::{HideError, Result};
    use parking_lot::Mutex;
    use std::fs::{File, OpenOptions};
    use std::io::{BufRead, BufReader, Write};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    pub type DynResearchLedger = Arc<dyn ResearchLedger>;

    pub trait ResearchLedger: Send + Sync {
        fn append_run(&self, run: &ResearchRun) -> Result<()>;
        fn load_runs(&self) -> Result<Vec<ResearchRun>>;

        fn latest(&self) -> Result<Option<ResearchRun>> {
            Ok(self.load_runs()?.into_iter().last())
        }

        fn load_by_state(&self, state: ResearchState) -> Result<Vec<ResearchRun>> {
            Ok(self.load_runs()?.into_iter().filter(|run| run.state == state).collect())
        }
    }

    #[derive(Debug, Default)]
    pub struct InMemoryResearchLedger {
        runs: Mutex<Vec<ResearchRun>>,
    }

    impl InMemoryResearchLedger {
        pub fn new() -> Self {
            Self::default()
        }
    }

    impl ResearchLedger for InMemoryResearchLedger {
        fn append_run(&self, run: &ResearchRun) -> Result<()> {
            self.runs.lock().push(run.clone());
            Ok(())
        }

        fn load_runs(&self) -> Result<Vec<ResearchRun>> {
            Ok(self.runs.lock().clone())
        }
    }

    #[derive(Debug, Clone)]
    pub struct JsonlResearchLedger {
        path: PathBuf,
    }

    impl JsonlResearchLedger {
        pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
            let path = path.into();
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if !path.exists() {
                File::create(&path)?;
            }
            Ok(Self { path })
        }

        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    impl ResearchLedger for JsonlResearchLedger {
        fn append_run(&self, run: &ResearchRun) -> Result<()> {
            let mut file = OpenOptions::new().create(true).append(true).open(&self.path)?;
            serde_json::to_writer(&mut file, run)?;
            file.write_all(b"\n")?;
            file.sync_data()?;
            Ok(())
        }

        fn load_runs(&self) -> Result<Vec<ResearchRun>> {
            read_runs(&self.path)
        }
    }

    fn read_runs(path: &Path) -> Result<Vec<ResearchRun>> {
        if !path.exists() {
            return Ok(Vec::new());
        }
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut runs = Vec::new();
        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let run = serde_json::from_str(&line).map_err(|err| {
                HideError::Storage(format!(
                    "failed to parse research ledger {} line {}: {err}",
                    path.display(),
                    idx + 1
                ))
            })?;
            runs.push(run);
        }
        Ok(runs)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn jsonl_research_ledger_roundtrips_runs() {
            let dir = std::env::temp_dir().join(format!("hawking_research_{}", hide_core::ids::now_ms()));
            let path = dir.join("runs.jsonl");
            let ledger = JsonlResearchLedger::open(&path).unwrap();
            let mut run = ResearchRun::new("paged attention");
            run.state = ResearchState::Complete;
            run.docs_read = 2;

            ledger.append_run(&run).unwrap();

            let reopened = JsonlResearchLedger::open(&path).unwrap();
            let loaded = reopened.load_runs().unwrap();
            assert_eq!(loaded.len(), 1);
            assert_eq!(loaded[0].topic, "paged attention");
            assert_eq!(reopened.latest().unwrap().unwrap().docs_read, 2);
            assert_eq!(reopened.load_by_state(ResearchState::Complete).unwrap().len(), 1);
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}
#[rustfmt::skip]
pub mod runtime_client {
    //! The Research Lab's only path to a model (bible ch.08 §4.1).
    //!
    //! The bible is emphatic that the lab never hard-depends on a model: every
    //! intelligent step (planner decomposition, cited synthesis, parse-time cleanup)
    //! goes through a [`RuntimeClient`] trait so tests run against a deterministic
    //! stub and production runs against the local Hawking runtime.
    //!
    //! Rather than invent a parallel HTTP client, this wraps `hawking-orch`'s
    //! [`InferenceClient`] (the workspace's single inference seam). `chat()` drives
    //! `generate()` and collects the streamed tokens into one string; `embed()`
    //! forwards directly. Production wires the real `HawkingHttpClient`; tests wire
    //! `StubInferenceClient`.

    use futures::future::BoxFuture;
    use hawking_orch::InferenceClient;
    use hide_core::error::Result;
    use hide_core::runtime::{InferenceMessage, InferenceRequest, StreamChunk};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// A model turn: a task kind, an optional system preamble, and the user prompt.
    #[derive(Debug, Clone)]
    pub struct ChatRequest {
        pub task_kind: String,
        pub system: Option<String>,
        pub prompt: String,
        pub max_output_tokens: usize,
    }

    impl ChatRequest {
        pub fn new(task_kind: impl Into<String>, prompt: impl Into<String>) -> Self {
            Self { task_kind: task_kind.into(), system: None, prompt: prompt.into(), max_output_tokens: 1024 }
        }

        pub fn with_system(mut self, system: impl Into<String>) -> Self {
            self.system = Some(system.into());
            self
        }

        pub fn with_max_tokens(mut self, n: usize) -> Self {
            self.max_output_tokens = n;
            self
        }

        fn into_inference(self) -> InferenceRequest {
            let mut messages = Vec::new();
            if let Some(sys) = &self.system {
                messages.push(InferenceMessage { role: "system".to_string(), content: sys.clone() });
            }
            messages.push(InferenceMessage { role: "user".to_string(), content: self.prompt.clone() });
            InferenceRequest {
                task_kind: self.task_kind,
                prompt: self.prompt,
                messages,
                max_output_tokens: self.max_output_tokens,
                sampler: None,
                grammar: None,
                want_logprobs: false,
                metadata: BTreeMap::new(),
            }
        }
    }

    /// The model boundary for the Research Lab. Backed by [`InferenceClient`].
    pub trait RuntimeClient: Send + Sync {
        /// Embed one text into a vector (used for dedup, retrieval, clustering).
        fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>>;

        /// Run a chat turn and return the full collected completion text.
        fn chat<'a>(&'a self, req: ChatRequest) -> BoxFuture<'a, Result<String>>;

        /// Embed a batch (default: sequential single-embeds; an HTTP client may
        /// override with a real batched request).
        fn embed_batch<'a>(&'a self, texts: &'a [String]) -> BoxFuture<'a, Result<Vec<Vec<f32>>>> {
            Box::pin(async move {
                let mut out = Vec::with_capacity(texts.len());
                for t in texts {
                    out.push(self.embed(t).await?);
                }
                Ok(out)
            })
        }
    }

    /// Adapts any [`InferenceClient`] into a [`RuntimeClient`].
    pub struct InferenceRuntime {
        client: Arc<dyn InferenceClient>,
    }

    impl InferenceRuntime {
        pub fn new(client: Arc<dyn InferenceClient>) -> Self {
            Self { client }
        }
    }

    impl RuntimeClient for InferenceRuntime {
        fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
            self.client.embed(text)
        }

        fn chat<'a>(&'a self, req: ChatRequest) -> BoxFuture<'a, Result<String>> {
            let request = req.into_inference();
            Box::pin(async move {
                let mut collected = String::new();
                let mut error: Option<String> = None;
                {
                    let mut sink = |chunk: StreamChunk| -> Result<()> {
                        match chunk {
                            StreamChunk::Token { text, .. } => collected.push_str(&text),
                            StreamChunk::Error { message } => error = Some(message),
                            StreamChunk::Done { .. } => {}
                        }
                        Ok(())
                    };
                    self.client.generate(request, &mut sink).await?;
                }
                if let Some(message) = error {
                    return Err(hide_core::HideError::RuntimeUnavailable(message));
                }
                Ok(collected)
            })
        }
    }

    /// Convenience: build a [`RuntimeClient`] from `StubInferenceClient` for tests.
    pub fn stub_runtime(response: impl Into<String>) -> Arc<dyn RuntimeClient> {
        Arc::new(InferenceRuntime::new(Arc::new(hawking_orch::StubInferenceClient::new(response))))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[tokio::test]
        async fn chat_collects_streamed_tokens() {
            let rt = stub_runtime("a cited synthesis");
            let out = rt.chat(ChatRequest::new("synthesize", "summarize")).await.unwrap();
            assert_eq!(out, "a cited synthesis");
        }

        #[tokio::test]
        async fn embed_is_deterministic() {
            let rt = stub_runtime("");
            let a = rt.embed("paged attention").await.unwrap();
            let b = rt.embed("paged attention").await.unwrap();
            assert_eq!(a, b);
        }

        #[tokio::test]
        async fn embed_batch_matches_single() {
            let rt = stub_runtime("");
            let texts = vec!["one".to_string(), "two".to_string()];
            let batch = rt.embed_batch(&texts).await.unwrap();
            assert_eq!(batch.len(), 2);
            assert_eq!(batch[0], rt.embed("one").await.unwrap());
        }
    }
}
#[rustfmt::skip]
pub mod verify {
    //! Adversarial verification (bible ch.08 §4.7).
    //!
    //! Three real checks replace the prior 2-antonym toy:
    //!
    //! 1. **Independence + corroboration.** A claim is only `Supported` once it is
    //!    backed by at least `MIN_CORROBORATION` *independent* sources — counted by
    //!    distinct origin (doc id / provenance source), not by raw peer count, so a
    //!    paper that repeats itself across sections cannot self-corroborate.
    //! 2. **Refutation detection.** Negation/antonym signals between overlapping
    //!    claims surface a `Contradicted`/`Refuted` status (first-class, §Tenet 3),
    //!    using a small but extensible polarity lexicon plus explicit negation.
    //! 3. **Citation re-verification** (§4.7.3, the #1 anti-hallucination guard):
    //!    every claim's evidence is re-opened from the CAS and re-hashed against the
    //!    recorded receipt; a claim whose bytes are missing or tampered is flagged.

    use crate::cas::{self, EvidenceCheck};
    use crate::kg::{Claim, ProvenanceSpan};
    use hide_core::error::Result;
    use hide_core::persistence::DynBlobStore;
    use serde::{Deserialize, Serialize};
    use std::collections::HashSet;

    /// Default minimum independent sources required to call a claim corroborated.
    pub const MIN_CORROBORATION: usize = 2;
    /// Lexical overlap above which two claims are treated as "about the same thing".
    pub const OVERLAP_THRESHOLD: f32 = 0.4;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ClaimVerification {
        pub claim_id: String,
        pub status: ClaimStatus,
        pub supporting_sources: usize,
        pub refuting_sources: usize,
        /// Distinct origin count (independence), ≤ supporting_sources.
        pub independent_sources: usize,
        /// Outcome of re-checking the cited evidence against the CAS.
        pub citation_check: CitationCheck,
        pub notes: Vec<String>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ClaimStatus {
        /// ≥ MIN_CORROBORATION independent supporting sources, no refutation.
        Supported,
        /// Refuting evidence present, no support.
        Refuted,
        /// Both support and refutation present (tension).
        Contradicted,
        /// Some support but below the independence/corroboration bar.
        SingleSource,
        /// No support, no refutation found.
        Unverified,
    }

    /// Whether the claim's cited evidence still hashes to its recorded receipt.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum CitationCheck {
        /// Evidence bytes present and hash-matched.
        Intact,
        /// Evidence bytes missing from the CAS — cannot verify.
        Missing,
        /// Evidence hash changed — the source was mutated after extraction.
        Tampered,
        /// No CAS available / no receipt recorded — not checked.
        NotChecked,
    }

    pub struct AdversarialVerifier;

    impl AdversarialVerifier {
        /// Verify a claim against its peers using independence + corroboration. Does
        /// NOT touch the CAS (pure, fast). Use [`Self::verify_with_cas`] to also
        /// re-verify the cited evidence bytes.
        pub fn verify(claim: &Claim, peer_claims: &[Claim]) -> ClaimVerification {
            let a = claim.text.to_lowercase();
            let mut support_origins: HashSet<String> = HashSet::new();
            let mut supporting = 0usize;
            let mut refuting = 0usize;
            let mut notes = Vec::new();

            for peer in peer_claims {
                if peer.id == claim.id {
                    continue;
                }
                let b = peer.text.to_lowercase();
                let overlap = lexical_overlap(&a, &b);
                if overlap < OVERLAP_THRESHOLD {
                    continue; // unrelated — neither supports nor refutes
                }
                if polarity_conflict(&a, &b) {
                    refuting += 1;
                    notes.push(format!("refuted by {}", peer.id));
                } else {
                    supporting += 1;
                    support_origins.insert(origin_key(&peer.provenance));
                }
            }

            // Independence: the claim's own origin does not count toward its support.
            support_origins.remove(&origin_key(&claim.provenance));
            let independent = support_origins.len();

            let status = if refuting > 0 && supporting > 0 {
                ClaimStatus::Contradicted
            } else if refuting > 0 {
                ClaimStatus::Refuted
            } else if independent >= MIN_CORROBORATION {
                ClaimStatus::Supported
            } else if supporting > 0 {
                ClaimStatus::SingleSource
            } else {
                ClaimStatus::Unverified
            };

            ClaimVerification {
                claim_id: claim.id.clone(),
                status,
                supporting_sources: supporting,
                refuting_sources: refuting,
                independent_sources: independent,
                citation_check: CitationCheck::NotChecked,
                notes,
            }
        }

        /// As [`Self::verify`], plus re-open and re-hash the claim's cited evidence
        /// against the CAS (§4.7.3). A failed citation check is recorded and noted.
        pub fn verify_with_cas(claim: &Claim, peer_claims: &[Claim], cas: &DynBlobStore) -> Result<ClaimVerification> {
            let mut v = Self::verify(claim, peer_claims);
            v.citation_check = check_citation(&claim.provenance, cas)?;
            match v.citation_check {
                CitationCheck::Tampered => v.notes.push("citation evidence tampered (hash mismatch)".to_string()),
                CitationCheck::Missing => v.notes.push("citation evidence missing from CAS".to_string()),
                _ => {}
            }
            Ok(v)
        }
    }

    /// Re-verify one provenance span's evidence against the CAS.
    fn check_citation(span: &ProvenanceSpan, cas: &DynBlobStore) -> Result<CitationCheck> {
        let (Some(blob), Some(hash)) = (&span.evidence_blob, &span.content_hash) else {
            return Ok(CitationCheck::NotChecked);
        };
        Ok(match cas::verify_evidence(cas, blob, hash)? {
            EvidenceCheck::Intact { .. } => CitationCheck::Intact,
            EvidenceCheck::Missing => CitationCheck::Missing,
            EvidenceCheck::Tampered { .. } => CitationCheck::Tampered,
        })
    }

    /// Independence key: the distinct *origin* of a claim — its doc id, falling back
    /// to the provenance source string.
    fn origin_key(span: &ProvenanceSpan) -> String {
        if !span.doc_id.is_empty() {
            span.doc_id.clone()
        } else {
            span.provenance.source.clone()
        }
    }

    /// A small, extensible polarity lexicon. A conflict is signalled when one claim
    /// asserts a direction and the peer asserts the opposite, OR when one negates a
    /// key term the other asserts.
    fn polarity_conflict(a: &str, b: &str) -> bool {
        const ANTONYMS: &[(&str, &str)] = &[
            ("increase", "decrease"),
            ("faster", "slower"),
            ("higher", "lower"),
            ("improves", "degrades"),
            ("reduces", "increases"),
            ("outperforms", "underperforms"),
            ("better", "worse"),
            ("gains", "loses"),
        ];
        for (x, y) in ANTONYMS {
            if (a.contains(x) && b.contains(y)) || (a.contains(y) && b.contains(x)) {
                return true;
            }
        }
        // Explicit negation: one side asserts a salient token, the other negates it.
        const NEG: &[&str] = &["not ", "no ", "without ", "fails to ", "does not "];
        let a_neg = NEG.iter().any(|n| a.contains(n));
        let b_neg = NEG.iter().any(|n| b.contains(n));
        if a_neg ^ b_neg {
            // Opposite negation polarity on overlapping topics is a soft conflict.
            return lexical_overlap(a, b) > 0.6;
        }
        false
    }

    fn lexical_overlap(a: &str, b: &str) -> f32 {
        let words: Vec<_> = a.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 3).collect();
        if words.is_empty() {
            return 0.0;
        }
        let hits = words.iter().filter(|w| b.contains(**w)).count();
        hits as f32 / words.len() as f32
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::kg::ConfidenceTier;
        use hide_core::persistence::InMemoryBlobStore;
        use hide_core::types::Provenance;
        use std::sync::Arc;

        fn claim(id: &str, doc: &str, text: &str) -> Claim {
            Claim {
                id: id.to_string(),
                text: text.to_string(),
                provenance: ProvenanceSpan {
                    doc_id: doc.to_string(),
                    span_id: None,
                    char_range: None,
                    citation: None,
                    content_hash: None,
                    evidence_blob: None,
                    provenance: Provenance::trusted("test"),
                },
                confidence: ConfidenceTier::Extracted,
            }
        }

        #[test]
        fn corroboration_requires_independent_origins() {
            let target = claim("c0", "docA", "paged attention reduces kv cache memory waste");
            // Two peers from the SAME doc → only one independent origin.
            let peers =
                vec![target.clone(), claim("c1", "docA", "paged attention reduces kv cache memory waste greatly")];
            let v = AdversarialVerifier::verify(&target, &peers);
            assert_eq!(v.independent_sources, 0); // same origin as target excluded
            assert_eq!(v.status, ClaimStatus::SingleSource);

            // Add a genuinely independent doc.
            let mut peers2 = peers;
            peers2.push(claim("c2", "docB", "paged attention reduces kv cache memory waste in serving"));
            let v2 = AdversarialVerifier::verify(&target, &peers2);
            assert_eq!(v2.independent_sources, 1);
        }

        #[test]
        fn antonym_pair_triggers_refutation() {
            let target = claim("c0", "docA", "the method improves throughput substantially");
            let peers = vec![claim("c1", "docB", "the method degrades throughput substantially")];
            let v = AdversarialVerifier::verify(&target, &peers);
            assert_eq!(v.status, ClaimStatus::Refuted);
            assert_eq!(v.refuting_sources, 1);
        }

        #[test]
        fn citation_recheck_detects_tamper() {
            let cas: DynBlobStore = Arc::new(InMemoryBlobStore::default());
            let bytes = b"reports 73% accuracy".to_vec();
            let (blob, hash) = cas::pin_evidence(&cas, bytes, None).unwrap();

            let mut good = claim("c0", "docA", "reports 73% accuracy on the benchmark");
            good.provenance.evidence_blob = Some(blob.clone());
            good.provenance.content_hash = Some(hash);
            let v = AdversarialVerifier::verify_with_cas(&good, &[], &cas).unwrap();
            assert_eq!(v.citation_check, CitationCheck::Intact);

            let mut bad = good.clone();
            bad.provenance.content_hash = Some("deadbeef".to_string());
            let vb = AdversarialVerifier::verify_with_cas(&bad, &[], &cas).unwrap();
            assert_eq!(vb.citation_check, CitationCheck::Tampered);
        }
    }
}

pub use cas::{blake3_hex, content_id, pin_evidence, verify_evidence, EvidenceCheck};
pub use checkpoint::{
    CheckpointEvent, CheckpointKind, CheckpointLedger, DynCheckpointLedger,
    InMemoryCheckpointLedger, JsonlCheckpointLedger, RunJournal,
};
pub use ingest::{ArxivAdapter, InMemorySourceAdapter, SourceAdapter, StructuredDoc};
pub use kg::{
    GraphQuery, InMemoryKnowledgeGraph, KnowledgeGraph, KnowledgeNode, NodeKind, PetKnowledgeGraph,
    QueryResult,
};
pub use litmap::{build_literature_map, compare_papers, LiteratureMap};
pub use pipeline::{ResearchBudget, ResearchPipeline, ResearchRun, ResearchState};
pub use run_ledger::{
    DynResearchLedger, InMemoryResearchLedger, JsonlResearchLedger, ResearchLedger,
};
pub use runtime_client::{stub_runtime, ChatRequest, InferenceRuntime, RuntimeClient};
pub use verify::{AdversarialVerifier, ClaimStatus, ClaimVerification};
