use crate::ingest::{SourceAdapter, SourceQuery, StructuredDoc};
use crate::kg::{Claim, InMemoryKnowledgeGraph};
use crate::verify::{AdversarialVerifier, ClaimVerification};
use hide_core::ids::{now_ms, RunId};
use hide_core::{HideError, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResearchRun {
    pub id: RunId,
    pub topic: String,
    pub state: ResearchState,
    pub created_at_ms: u64,
    pub docs_read: usize,
    pub claims: Vec<Claim>,
    pub verifications: Vec<ClaimVerification>,
}

impl ResearchRun {
    pub fn new(topic: impl Into<String>) -> Self {
        Self {
            id: RunId::new(),
            topic: topic.into(),
            state: ResearchState::PlanScope,
            created_at_ms: now_ms(),
            docs_read: 0,
            claims: Vec::new(),
            verifications: Vec::new(),
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
    Complete,
    Failed,
}

pub struct ResearchPipeline {
    adapters: Vec<Arc<dyn SourceAdapter>>,
    graph: Arc<InMemoryKnowledgeGraph>,
}

impl ResearchPipeline {
    pub fn new(graph: Arc<InMemoryKnowledgeGraph>) -> Self {
        Self {
            adapters: Vec::new(),
            graph,
        }
    }

    pub fn add_adapter(&mut self, adapter: Arc<dyn SourceAdapter>) {
        self.adapters.push(adapter);
    }

    pub async fn run_once(&self, topic: impl Into<String>, limit: usize) -> Result<ResearchRun> {
        let mut run = ResearchRun::new(topic);
        self.step(&mut run, limit).await?;
        while !matches!(run.state, ResearchState::Complete | ResearchState::Failed) {
            self.step(&mut run, limit).await?;
        }
        Ok(run)
    }

    pub async fn step(&self, run: &mut ResearchRun, limit: usize) -> Result<()> {
        match run.state {
            ResearchState::PlanScope => run.state = ResearchState::FanOut,
            ResearchState::FanOut => {
                if self.adapters.is_empty() {
                    return Err(HideError::InvalidState(
                        "research pipeline has no source adapters".to_string(),
                    ));
                }
                run.state = ResearchState::Fetch;
            }
            ResearchState::Fetch => {
                let query = SourceQuery {
                    query: run.topic.clone(),
                    limit,
                    source_types: Vec::new(),
                };
                let mut docs = Vec::<StructuredDoc>::new();
                for adapter in &self.adapters {
                    for record in adapter.search(&query).await? {
                        docs.push(adapter.fetch(&record).await?);
                    }
                }
                for doc in docs {
                    let claims = self.graph.ingest_doc_shell(&doc);
                    run.docs_read += 1;
                    run.claims.extend(claims);
                }
                run.state = ResearchState::Read;
            }
            ResearchState::Read => run.state = ResearchState::Verify,
            ResearchState::Verify => {
                run.verifications = run
                    .claims
                    .iter()
                    .map(|claim| AdversarialVerifier::verify(claim, &run.claims))
                    .collect();
                run.state = ResearchState::Synthesize;
            }
            ResearchState::Synthesize => run.state = ResearchState::Complete,
            ResearchState::Complete | ResearchState::Failed => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::{
        CitationRef, DocSection, InMemorySourceAdapter, SourceQuality, SourceRecord, SourceType,
    };
    use hide_core::types::Provenance;

    #[tokio::test]
    async fn pipeline_ingests_docs_into_claims() {
        let graph = Arc::new(InMemoryKnowledgeGraph::default());
        let adapter = Arc::new(InMemorySourceAdapter::new("memory", SourceType::PdfLocal));
        let source = SourceRecord {
            id: "doc1".to_string(),
            source_type: SourceType::PdfLocal,
            title: "KV cache research".to_string(),
            uri: "memory://doc1".to_string(),
            content_hash: None,
            quality: SourceQuality::default(),
            provenance: Provenance::trusted("test"),
        };
        adapter.insert(StructuredDoc {
            id: "doc1".to_string(),
            source,
            title: "KV cache research".to_string(),
            sections: vec![DocSection {
                heading: "Abstract".to_string(),
                text: "Paged attention improves KV cache reuse.".to_string(),
                spans: Vec::new(),
            }],
            references: Vec::<CitationRef>::new(),
            blob: None,
        });
        let mut pipeline = ResearchPipeline::new(graph);
        pipeline.add_adapter(adapter);
        let run = pipeline.run_once("KV cache", 4).await.unwrap();
        assert_eq!(run.state, ResearchState::Complete);
        assert_eq!(run.docs_read, 1);
        assert_eq!(run.claims.len(), 1);
    }
}
