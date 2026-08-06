//! Evidence-first data procurement for HCLI.
//!
//! HCLI research is a concrete path over the existing Research Lab FSM:
//! `PlanScope → FanOut → Fetch → Read → Verify → Synthesize → Persist →
//! Reflect`.  The first shipped network adapter is deliberately narrow and
//! auditable—arXiv Atom title/abstract retrieval.  It is not presented as
//! generic web search, PDF full-text ingestion, or an unlimited upload path.

use crate::model_provider::{HttpModelProvider, ModelProviderInferenceClient};
use crate::BackendHost;
use hawking_orch::inference::InferenceClient;
use hawking_research::{
    ArxivAdapter, InferenceRuntime, JsonlCheckpointLedger, PetKnowledgeGraph, ResearchBudget,
    ResearchPipeline,
};
use hide_core::ids::now_ms;
use hide_core::Result;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

pub const HCLI_RESEARCH_RECEIPT_SCHEMA: &str = "hcli.research.v1";

/// Bounded parameters for an evidence-backed research run.
#[derive(Debug, Clone)]
pub struct HcliResearchConfig {
    pub topic: String,
    /// A local Hawking-compatible endpoint. Research will not silently use a
    /// cloud model or a deterministic stub in its place.
    pub model_url: Option<String>,
    /// Maximum source records returned for one topic/sub-question query.
    pub per_query_limit: usize,
    /// Maximum selected documents per reflect round.
    pub read_budget: usize,
    /// Total procurement/synthesis rounds. Finite on purpose.
    pub max_rounds: u32,
}

impl Default for HcliResearchConfig {
    fn default() -> Self {
        Self {
            topic: String::new(),
            model_url: None,
            per_query_limit: 12,
            read_budget: 24,
            max_rounds: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HcliResearchResult {
    pub complete: bool,
    pub receipt: Value,
}

/// Execute the real evidence-backed research pipeline and append its durable
/// run summary to the host's research ledger. The raw source evidence is pinned
/// into the workspace blob store by the Research Lab before verification.
pub async fn run_hcli_research(
    host: &BackendHost,
    config: HcliResearchConfig,
) -> Result<HcliResearchResult> {
    let started_ms = now_ms();
    let started = Instant::now();
    let topic = config.topic.trim().to_string();
    let per_query_limit = config.per_query_limit.clamp(1, 100);
    let read_budget = config.read_budget.clamp(1, 100);
    let max_rounds = config.max_rounds.clamp(1, 12);
    let model_url = config
        .model_url
        .as_deref()
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map(str::to_string);

    let mut receipt = json!({
        "schema": HCLI_RESEARCH_RECEIPT_SCHEMA,
        "started_ms": started_ms,
        "topic": {
            "text": config.topic,
            "blake3": blake3::hash(config.topic.as_bytes()).to_hex().to_string(),
        },
        "procurement": {
            "source_adapters": [{
                "name": "arxiv",
                "mode": "public Atom API; title and abstract metadata",
                "full_text_pdf_or_latex": false,
                "network": true,
            }],
            "per_query_limit_requested": config.per_query_limit,
            "per_query_limit_effective": per_query_limit,
            "read_budget_requested": config.read_budget,
            "read_budget_effective": read_budget,
            "max_rounds_requested": config.max_rounds,
            "max_rounds_effective": max_rounds,
            "dedup": "content hash then normalized URI",
            "evidence": "each non-empty source section is canonicalized and pinned in the workspace CAS before claim verification",
        },
        "runtime": {
            "requested_url": model_url,
            "model_kind": "one explicit local Hawking-compatible endpoint",
        },
        "limitations": [
            "This command currently procures from arXiv Atom title/abstract metadata. It does not claim generic web search, PDF full text, local upload-to-context, or arbitrary source adapters.",
            "The Research Lab validates citation evidence against the CAS. A cited synthesis is not itself a proof that every external factual claim is true.",
            "Research model calls use the Research Lab runtime seam, not HIDE's compiled repository-context prompt path.",
        ],
    });

    let Some(model_url) = model_url else {
        receipt["status"] = json!("blocked_no_model_url");
        receipt["finished_ms"] = json!(now_ms());
        receipt["wall_elapsed_ms"] = json!(started.elapsed().as_millis() as u64);
        seal(&mut receipt)?;
        return Ok(HcliResearchResult {
            complete: false,
            receipt,
        });
    };

    let inference: Arc<dyn InferenceClient> = Arc::new(ModelProviderInferenceClient::new(
        HttpModelProvider::new(model_url),
    ));
    let runtime = Arc::new(InferenceRuntime::new(inference));
    let graph = Arc::new(PetKnowledgeGraph::new());
    let journal_path = host
        .services
        .config
        .workspace_root
        .join(".hide")
        .join("research")
        .join("checkpoints.jsonl");
    let checkpoints = Arc::new(JsonlCheckpointLedger::open(&journal_path)?);
    let mut pipeline = ResearchPipeline::new(
        graph,
        runtime,
        host.services.blob_store.clone(),
        checkpoints,
    )
    .with_budget(ResearchBudget {
        read_budget,
        max_rounds,
        ..ResearchBudget::default()
    });
    pipeline.add_adapter(Arc::new(ArxivAdapter::new()));

    match pipeline.run_once(topic, per_query_limit).await {
        Ok(run) => {
            // The research connector's durable ledger is the summary index;
            // the checkpoint journal remains the resume/audit source.
            host.services.research_ledger.append_run(&run)?;
            let supported_claims = run
                .verifications
                .iter()
                .filter(|verification| {
                    matches!(
                        verification.status,
                        hawking_research::ClaimStatus::Supported
                            | hawking_research::ClaimStatus::Contradicted
                    )
                })
                .count();
            let citation_checked = run
                .verifications
                .iter()
                .filter(|verification| {
                    !matches!(
                        verification.citation_check,
                        hawking_research::verify::CitationCheck::NotChecked
                    )
                })
                .count();
            receipt["status"] = json!(match run.state {
                hawking_research::ResearchState::Complete => "completed",
                _ => "incomplete",
            });
            receipt["run"] = json!({
                "id": run.id,
                "state": run.state,
                "round": run.round,
                "sub_questions": run.sub_questions,
                "docs_read": run.docs_read,
                "claims": run.claims.len(),
                "verifications": run.verifications.len(),
                "supported_or_contradicted_claims": supported_claims,
                "citation_checks_performed": citation_checked,
                "findings": run.findings,
                "report": run.report,
                "checkpoint_journal": journal_path,
                "run_summary_ledger": "workspace .hide/research-runs.jsonl",
            });
            receipt["finished_ms"] = json!(now_ms());
            receipt["wall_elapsed_ms"] = json!(started.elapsed().as_millis() as u64);
            seal(&mut receipt)?;
            Ok(HcliResearchResult {
                complete: matches!(run.state, hawking_research::ResearchState::Complete),
                receipt,
            })
        }
        Err(error) => {
            receipt["status"] = json!("failed");
            receipt["failure"] = json!(error.to_string());
            receipt["finished_ms"] = json!(now_ms());
            receipt["wall_elapsed_ms"] = json!(started.elapsed().as_millis() as u64);
            seal(&mut receipt)?;
            Ok(HcliResearchResult {
                complete: false,
                receipt,
            })
        }
    }
}

fn seal(receipt: &mut Value) -> Result<()> {
    let bytes = serde_json::to_vec(receipt)?;
    receipt["content_blake3"] = json!(blake3::hash(&bytes).to_hex().to_string());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::BackendServices;
    use hide_core::config::HideConfig;
    use hide_core::event::InMemoryEventLog;
    use std::sync::Arc;

    #[tokio::test]
    async fn missing_runtime_emits_a_sealed_blocked_receipt_without_network_work() {
        let temp = tempfile::tempdir().unwrap();
        let services = BackendServices::new(
            HideConfig::for_workspace(temp.path()),
            Arc::new(InMemoryEventLog::new()),
        );
        let host = BackendHost::from_services(services).unwrap();
        let result = run_hcli_research(
            &host,
            HcliResearchConfig {
                topic: "kernel optimization".to_string(),
                model_url: None,
                ..HcliResearchConfig::default()
            },
        )
        .await
        .unwrap();

        assert!(!result.complete);
        assert_eq!(
            result.receipt.get("status").and_then(Value::as_str),
            Some("blocked_no_model_url")
        );
        assert_eq!(
            result
                .receipt
                .get("content_blake3")
                .and_then(Value::as_str)
                .map(str::len),
            Some(64)
        );
    }
}
