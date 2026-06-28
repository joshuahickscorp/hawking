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
    Claim, ConfidenceTier, EdgeKind, KnowledgeEdge, KnowledgeNode, NodeKind, PetKnowledgeGraph,
    ProvenanceSpan,
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
        Self {
            read_budget: 8,
            max_rounds: 2,
            coverage_target: 0.8,
            dedup_cosine: 0.95,
        }
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
        Self {
            adapters: Vec::new(),
            graph,
            runtime,
            cas,
            checkpoints,
            budget: ResearchBudget::default(),
        }
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
        journal.record(CheckpointKind::Opened {
            topic: run.topic.clone(),
            seed: run.seed,
        })?;
        loop {
            self.step(&mut run, limit, &mut journal).await?;
            if matches!(run.state, ResearchState::Complete | ResearchState::Failed) {
                break;
            }
        }
        Ok(run)
    }

    pub async fn step(
        &self,
        run: &mut ResearchRun,
        limit: usize,
        journal: &mut RunJournal,
    ) -> Result<()> {
        journal.record(CheckpointKind::State {
            state: format!("{:?}", run.state),
        })?;
        match run.state {
            // ── PlanScope: decompose the topic into sub-questions via the model.
            ResearchState::PlanScope => {
                run.sub_questions = self.plan_scope(&run.topic).await?;
                run.state = ResearchState::FanOut;
            }
            // ── FanOut: ensure we have adapters (real fan-out happens in Fetch).
            ResearchState::FanOut => {
                if self.adapters.is_empty() {
                    return Err(HideError::InvalidState(
                        "research pipeline has no source adapters".to_string(),
                    ));
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
                    let query = SourceQuery {
                        query: q.clone(),
                        limit,
                        source_types: Vec::new(),
                    };
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
                journal.record(CheckpointKind::Round {
                    round: run.round,
                    coverage,
                    novelty,
                })?;
                if coverage < self.budget.coverage_target
                    && run.round < self.budget.max_rounds
                    && novelty > 0.0
                {
                    run.state = ResearchState::Fetch; // another round, gaps only
                } else {
                    journal.record(CheckpointKind::Done {
                        docs_read: run.docs_read,
                        claims: run.claims.len(),
                    })?;
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
        let raw = self
            .runtime
            .chat(ChatRequest::new("plan", prompt).with_max_tokens(256))
            .await
            .unwrap_or_default();
        let mut subs: Vec<String> = raw
            .lines()
            .map(|l| l.trim_start_matches(['-', '*', '•', ' ']).trim().to_string())
            .filter(|l| l.len() > 5)
            .take(4)
            .collect();
        if subs.is_empty() {
            // Deterministic fallback so the pipeline never stalls offline.
            subs = vec![
                format!("What is {topic}?"),
                format!("What are the limitations of {topic}?"),
            ];
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
        out.sort_by(|a, b| {
            b.quality
                .score()
                .partial_cmp(&a.quality.score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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
            section.evidence = Some(crate::ingest::SectionEvidence {
                blob,
                content_hash: hash,
            });
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
                            crate::verify::ClaimStatus::Refuted
                                | crate::verify::ClaimStatus::Unverified
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
        let report = self
            .runtime
            .chat(ChatRequest::new("synthesize", prompt).with_max_tokens(1024))
            .await?;
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
            if matches!(
                v.status,
                crate::verify::ClaimStatus::Supported | crate::verify::ClaimStatus::Contradicted
            ) {
                if let Some(claim) = run.claims.iter().find(|c| c.id == v.claim_id) {
                    let fid = cas::composite_id("finding", &[&claim.text]);
                    findings.push(Finding {
                        id: fid.clone(),
                        summary: claim.text.chars().take(160).collect(),
                        claim_ids: vec![claim.id.clone()],
                        actionable: matches!(
                            v.status,
                            crate::verify::ClaimStatus::Contradicted
                        ),
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
        let claim_blob = run
            .claims
            .iter()
            .map(|c| c.text.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");
        let hit = run
            .sub_questions
            .iter()
            .filter(|q| {
                q.to_lowercase()
                    .split_whitespace()
                    .filter(|w| w.len() > 4)
                    .any(|w| claim_blob.contains(w))
            })
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
    use crate::ingest::{
        structured_doc_from_text, InMemorySourceAdapter, SourceQuality, SourceType,
    };
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
        assert!(run
            .verifications
            .iter()
            .any(|v| v.status == crate::verify::ClaimStatus::Supported));
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
        let target = run
            .claims
            .iter()
            .find(|c| c.provenance.evidence_blob.is_some())
            .unwrap();
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

