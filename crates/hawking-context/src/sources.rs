//! Built-in context sources (bible §4.2.1, §4.7, §7).
//!
//! Each source produces ranked candidates with **real trust/confidence**
//! provenance (not blanket `trusted`). The compiler ranks and packs them
//! uniformly; new sources plug in by implementing [`ContextSource`].

use crate::compiler::{CompileInput, ContextCandidate, ContextSource};
use crate::manifest::{ContextSourceKind, PinState};
use crate::memory::{MemoryKind, MemoryStore};
use futures::future::BoxFuture;
use hawking_index::{CodeIndex, SearchQuery, SearchResultSource};
use hide_core::error::Result;
use hide_core::types::{Provenance, TrustLevel};
use std::sync::Arc;

/// A static, in-prompt source (system prompt, fixed instructions). Spans are
/// `never_evict` so they pin to the head.
pub struct StaticContextSource {
    pub name: String,
    pub source: ContextSourceKind,
    pub spans: Vec<(String, String, f32)>,
}

impl ContextSource for StaticContextSource {
    fn name(&self) -> &str {
        &self.name
    }

    fn gather<'a>(
        &'a self,
        _input: &'a CompileInput,
    ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
        Box::pin(async move {
            Ok(self
                .spans
                .iter()
                .enumerate()
                .map(|(idx, (title, text, score))| {
                    ContextCandidate::new(
                        format!("{}:{idx}", self.name),
                        self.source.clone(),
                        title.clone(),
                        text.clone(),
                        *score,
                        Provenance::trusted(self.name.clone()),
                    )
                })
                .collect())
        })
    }
}

/// The system source: a never-evict head band (bible §4.2.3 reservation).
pub struct SystemContextSource {
    pub text: String,
}

impl SystemContextSource {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

impl ContextSource for SystemContextSource {
    fn name(&self) -> &str {
        "system"
    }

    fn gather<'a>(
        &'a self,
        _input: &'a CompileInput,
    ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
        Box::pin(async move {
            let mut c = ContextCandidate::new(
                "system:0",
                ContextSourceKind::System,
                "System",
                self.text.clone(),
                1.0,
                Provenance {
                    source: "system".to_string(),
                    trust: TrustLevel::Trusted,
                    confidence: 1.0,
                    labels: vec!["system".to_string()],
                    derived_from: Vec::new(),
                },
            );
            c.pin = PinState::NeverEvict;
            c.importance = Some(1.0);
            Ok(vec![c])
        })
    }
}

/// Code/symbol source backed by `hawking-index`. Carries `path:line`
/// provenance and propagates the index's result-source as trust signal.
pub struct CodeIndexContextSource {
    pub name: String,
    pub index: Arc<dyn CodeIndex>,
    pub limit: usize,
    pub include_semantic: bool,
}

impl CodeIndexContextSource {
    pub fn new(index: Arc<dyn CodeIndex>, limit: usize) -> Self {
        Self {
            name: "code_index".to_string(),
            index,
            limit,
            include_semantic: true,
        }
    }

    /// Toggle the semantic (embedding) retrieval leg.
    pub fn with_semantic(mut self, on: bool) -> Self {
        self.include_semantic = on;
        self
    }
}

impl ContextSource for CodeIndexContextSource {
    fn name(&self) -> &str {
        &self.name
    }

    fn gather<'a>(
        &'a self,
        input: &'a CompileInput,
    ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
        Box::pin(async move {
            // W-F2-6: route the query by shape (an exact-symbol query skips the
            // fuzzy legs), capped by this source's semantic config, then prefer
            // precise hits over similar-code semantic ones on score ties.
            let mut query = SearchQuery::routed(input.task.clone(), self.limit);
            query.include_semantic = query.include_semantic && self.include_semantic;
            let mut results = self.index.search(query).await?;
            hawking_index::query::rerank_prefer_precise(&mut results);
            Ok(results
                .into_iter()
                .enumerate()
                .map(|(idx, result)| {
                    let range = result
                        .span
                        .range
                        .as_ref()
                        .map(|range| format!(":{}:{}", range.start_line, range.start_col))
                        .unwrap_or_default();
                    let path = result.span.path.display().to_string();
                    // Code from the workspace is `Workspace` trust; confidence
                    // tracks the index leg that found it (symbol > lexical).
                    let confidence = match result.source {
                        SearchResultSource::Symbol => 0.95,
                        SearchResultSource::Graph => 0.9,
                        SearchResultSource::Lexical => 0.8,
                        SearchResultSource::Semantic => 0.75,
                    };
                    let provenance = Provenance {
                        source: format!("code-index:{path}{range}"),
                        trust: TrustLevel::Workspace,
                        confidence,
                        labels: vec![format!("{:?}", result.source).to_lowercase()],
                        derived_from: vec![path.clone()],
                    };
                    ContextCandidate::new(
                        format!("{}:{idx}", self.name),
                        ContextSourceKind::Code,
                        format!("{}{}", result.title, range),
                        result.snippet,
                        result.score,
                        provenance,
                    )
                })
                .collect())
        })
    }
}

/// Memory source: retrieves relevant memories and offers them as candidates
/// (bible §4.7.2 "progressive disclosure" — memory competes, not always-on).
/// Memory-sourced facts carry their stored provenance/confidence (F12).
pub struct MemoryContextSource {
    pub store: Arc<dyn MemoryStore>,
    pub kinds: Vec<MemoryKind>,
    pub k: usize,
}

impl MemoryContextSource {
    pub fn new(store: Arc<dyn MemoryStore>, k: usize) -> Self {
        Self {
            store,
            kinds: Vec::new(),
            k,
        }
    }
}

impl ContextSource for MemoryContextSource {
    fn name(&self) -> &str {
        "memory"
    }

    fn gather<'a>(
        &'a self,
        input: &'a CompileInput,
    ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
        Box::pin(async move {
            let hits = self
                .store
                .retrieve(&input.task, self.k, &self.kinds)
                .await?;
            Ok(hits
                .into_iter()
                .enumerate()
                .map(|(idx, h)| {
                    let pinned = h.meta.pinned;
                    let mut c = ContextCandidate::new(
                        format!("memory:{}", h.record.id),
                        ContextSourceKind::Memory,
                        format!("memory:{}", h.record.kind.as_str_public()),
                        h.record.text.clone(),
                        h.score.clamp(0.0, 1.0),
                        h.record.provenance.clone(),
                    );
                    c.importance = Some(h.importance);
                    c.recency_ms = h.record.last_used_at_ms.or(Some(h.record.created_at_ms));
                    if pinned {
                        c.pin = PinState::UserPinned;
                    }
                    let _ = idx;
                    c
                })
                .collect())
        })
    }
}

/// Plan source: the current plan steps as context (untrusted-derived = the
/// agent's own working state, `Workspace` trust).
pub struct PlanContextSource {
    pub steps: Vec<String>,
}

impl ContextSource for PlanContextSource {
    fn name(&self) -> &str {
        "plan"
    }

    fn gather<'a>(
        &'a self,
        _input: &'a CompileInput,
    ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
        Box::pin(async move {
            Ok(self
                .steps
                .iter()
                .enumerate()
                .map(|(idx, step)| {
                    ContextCandidate::new(
                        format!("plan:{idx}"),
                        ContextSourceKind::Plan,
                        format!("plan step {idx}"),
                        step.clone(),
                        0.9,
                        Provenance {
                            source: "plan".to_string(),
                            trust: TrustLevel::Workspace,
                            confidence: 0.9,
                            labels: vec!["plan".to_string()],
                            derived_from: Vec::new(),
                        },
                    )
                })
                .collect())
        })
    }
}

/// A tool output (untrusted — bible F12: tool-sourced confidence < 1).
pub struct ToolOutputContextSource {
    pub outputs: Vec<(String, String)>, // (tool_call_id, text)
}

impl ContextSource for ToolOutputContextSource {
    fn name(&self) -> &str {
        "tool_output"
    }

    fn gather<'a>(
        &'a self,
        _input: &'a CompileInput,
    ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
        Box::pin(async move {
            Ok(self
                .outputs
                .iter()
                .map(|(call_id, text)| {
                    let mut c = ContextCandidate::new(
                        format!("tool:{call_id}"),
                        ContextSourceKind::ToolOutput,
                        format!("tool output {call_id}"),
                        text.clone(),
                        0.7,
                        Provenance {
                            source: format!("tool_call:{call_id}"),
                            // Tool output is untrusted and low-confidence (F12).
                            trust: TrustLevel::ToolOutput,
                            confidence: 0.6,
                            labels: vec!["tool-output".to_string()],
                            derived_from: vec![call_id.clone()],
                        },
                    );
                    // Tool outputs decay fast (recency now → high, ages quickly).
                    c.recency_ms = Some(hide_core::ids::now_ms());
                    c
                })
                .collect())
        })
    }
}

/// Diagnostics (compiler/linter messages) — high value for debugging profiles.
pub struct DiagnosticsContextSource {
    pub diagnostics: Vec<String>,
}

impl ContextSource for DiagnosticsContextSource {
    fn name(&self) -> &str {
        "diagnostics"
    }

    fn gather<'a>(
        &'a self,
        _input: &'a CompileInput,
    ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
        Box::pin(async move {
            Ok(self
                .diagnostics
                .iter()
                .enumerate()
                .map(|(idx, d)| {
                    let mut c = ContextCandidate::new(
                        format!("diag:{idx}"),
                        ContextSourceKind::Diagnostics,
                        format!("diagnostic {idx}"),
                        d.clone(),
                        0.85,
                        Provenance {
                            source: "diagnostics".to_string(),
                            trust: TrustLevel::Workspace,
                            confidence: 0.9,
                            labels: vec!["diagnostic".to_string()],
                            derived_from: Vec::new(),
                        },
                    );
                    c.importance = Some(0.9);
                    c
                })
                .collect())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{CompileInput, ContextCompiler};
    use crate::memory::{InMemoryMemoryStore, MemoryKind, MemoryStore};
    use crate::profiles::ContextProfile;
    use hawking_index::InMemoryCodeIndex;
    use hide_core::ids::ModelId;
    use hide_core::runtime::{ModelArchitecture, ModelDescriptor};

    fn model() -> ModelDescriptor {
        ModelDescriptor {
            id: ModelId::new(),
            name: "test".to_string(),
            architecture: ModelArchitecture::Transformer,
            context_tokens: 1024,
            tokenizer_signature: "test".to_string(),
            footprint_mb: 1,
        }
    }

    #[tokio::test]
    async fn code_index_source_feeds_compiler() {
        let index = Arc::new(InMemoryCodeIndex::default());
        index.add_text_file(
            "src/lib.rs",
            "pub fn compile_context() {}\n// context compiler bridge\n",
            None,
        );
        let mut compiler = ContextCompiler::new();
        compiler.add_source(CodeIndexContextSource::new(index, 4));
        let compiled = compiler
            .compile(CompileInput {
                profile: ContextProfile::coding_default(1024),
                model: model(),
                task: "context compiler".to_string(),
            })
            .await
            .unwrap();
        assert!(compiled.prompt.contains("context compiler"));
        assert!(!compiled.manifest.retained.is_empty());
        // Provenance is workspace trust with a real path, not blanket-trusted.
        let span = &compiled.manifest.retained[0];
        assert_eq!(span.provenance.trust, TrustLevel::Workspace);
    }

    #[tokio::test]
    async fn memory_source_propagates_provenance_confidence() {
        let store = Arc::new(InMemoryMemoryStore::default());
        let mut rec = InMemoryMemoryStore::record(
            MemoryKind::Semantic,
            "the database layer lives in db and uses sqlx",
            Provenance {
                source: "file_scan".to_string(),
                trust: TrustLevel::ToolOutput,
                confidence: 0.7,
                labels: vec![],
                derived_from: vec![],
            },
        );
        rec.importance = 0.8;
        store.put(rec).await.unwrap();

        let mut compiler = ContextCompiler::new();
        compiler.add_source(MemoryContextSource::new(store, 5));
        let compiled = compiler
            .compile(CompileInput {
                profile: ContextProfile::coding_default(1024),
                model: model(),
                task: "database sqlx".to_string(),
            })
            .await
            .unwrap();
        let mem_span = compiled
            .manifest
            .retained
            .iter()
            .find(|s| matches!(s.source, ContextSourceKind::Memory))
            .expect("memory span retained");
        // Confidence flowed through (not overwritten to 1.0).
        assert!((mem_span.provenance.confidence - 0.7).abs() < 1e-6);
        assert_eq!(mem_span.provenance.trust, TrustLevel::ToolOutput);
    }
}
