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
        FindingIssue {
            title: d.title,
            body: d.body,
            claim_ids: d.claim_ids,
            suggested_labels: d.labels,
        }
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
    let params = symbols
        .iter()
        .map(|(s, _)| format!("{}: f64", sanitize_ident(s)))
        .collect::<Vec<_>>()
        .join(", ");
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
        "the", "and", "for", "with", "that", "this", "from", "into", "over", "than", "more",
        "less", "have", "been", "are", "was", "were", "our", "their", "improves", "reduces",
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
            &[
                ("x".into(), "input".into()),
                ("d".into(), "dimension".into()),
            ],
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
        let d = claim_to_issue_linked(&claim("paged_attention improves cache reuse"), &index)
            .await
            .unwrap();
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
        assert!(rec
            .provenance
            .derived_from
            .iter()
            .any(|d| d == "kg:concept:x"));
    }
}
