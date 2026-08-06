//! Built-in context sources (bible §4.2.1, §4.7, §7).
//!
//! Each source produces ranked candidates with **real trust/confidence**
//! provenance (not blanket `trusted`). The compiler ranks and packs them
//! uniformly; new sources plug in by implementing [`ContextSource`].

use crate::compiler::{CompileInput, ContextCandidate, ContextSource};
use crate::manifest::{ContextSourceKind, PinState};
use crate::memory::{MemoryKind, MemoryStore};
use crate::memory_classes::{ClassBudgets, ClassedMemorySystem, MemoryClass, WriteAuthority};
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
///
/// This is the **legacy** single-store source (one table, kind labels). Prefer
/// [`ClassedMemoryContextSource`] for the six real memory systems.
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

/// Context source over the six real memory classes.
///
/// Each class is asked its own retrieval question and filled under an independent
/// token budget — not a single `SELECT * WHERE kind = ?` over one table.
pub struct ClassedMemoryContextSource {
    pub system: Arc<ClassedMemorySystem>,
    pub budgets: ClassBudgets,
    pub turn_id: Option<String>,
    pub session_id: Option<String>,
}

impl ClassedMemoryContextSource {
    pub fn new(system: Arc<ClassedMemorySystem>, budgets: ClassBudgets) -> Self {
        Self {
            system,
            budgets,
            turn_id: None,
            session_id: None,
        }
    }

    pub fn with_turn(mut self, turn_id: impl Into<String>) -> Self {
        self.turn_id = Some(turn_id.into());
        self
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
}

impl ContextSource for ClassedMemoryContextSource {
    fn name(&self) -> &str {
        "classed_memory"
    }

    fn gather<'a>(
        &'a self,
        input: &'a CompileInput,
    ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
        Box::pin(async move {
            let retrieval = self.system.retrieve_for_compile(
                &input.task,
                self.turn_id.as_deref(),
                self.session_id.as_deref(),
                &self.budgets,
            )?;
            let mut out = Vec::new();
            for slice in &retrieval.slices {
                for hit in &slice.hits {
                    let trust = match hit.provenance.authority {
                        WriteAuthority::Verifier => TrustLevel::Trusted,
                        WriteAuthority::UserExplicit => TrustLevel::Trusted,
                        WriteAuthority::ProjectDistill | WriteAuthority::ToolReceipt => {
                            TrustLevel::Workspace
                        }
                        WriteAuthority::EventStream | WriteAuthority::Turn => TrustLevel::Workspace,
                    };
                    let confidence = match hit.class {
                        MemoryClass::Verification => 0.95,
                        MemoryClass::User => 0.9,
                        MemoryClass::SemanticProject | MemoryClass::Procedural => 0.85,
                        MemoryClass::Episodic => 0.75,
                        MemoryClass::Working => 0.7,
                    };
                    let mut labels = vec![
                        format!("memory_class:{}", hit.class.as_str()),
                        format!("authority:{:?}", hit.provenance.authority),
                    ];
                    if let Some(tier) = &hit.evidence_tier {
                        labels.push(format!("evidence_tier:{tier}"));
                    }
                    let mut derived = hit.provenance.evidence.clone();
                    if let Some(t) = &hit.provenance.turn_id {
                        derived.push(format!("turn:{t}"));
                    }
                    if let Some(r) = &hit.provenance.run_id {
                        derived.push(format!("run:{r}"));
                    }
                    let provenance = Provenance {
                        source: format!(
                            "memory_class:{}:{}",
                            hit.class.as_str(),
                            hit.provenance.writer
                        ),
                        trust,
                        confidence,
                        labels,
                        derived_from: derived,
                    };
                    let mut c = ContextCandidate::new(
                        format!("memclass:{}:{}", hit.class.as_str(), hit.id),
                        ContextSourceKind::Memory,
                        format!("memory:{}", hit.class.as_str()),
                        hit.text.clone(),
                        hit.importance.clamp(0.0, 1.0),
                        provenance,
                    );
                    c.importance = Some(hit.importance);
                    c.recency_ms = Some(hit.provenance.written_at_ms);
                    // User prefs + verification claims float above ambient code.
                    if matches!(hit.class, MemoryClass::User | MemoryClass::Verification) {
                        c.pin = PinState::UserPinned;
                    }
                    out.push(c);
                }
            }
            Ok(out)
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
        assert!((mem_span.provenance.confidence - 0.7).abs() < 1e-6);
        assert_eq!(mem_span.provenance.trust, TrustLevel::ToolOutput);
    }
    #[tokio::test]
    async fn classed_memory_compiler_retrieves_multiple_classes_independent_budgets() {
        use crate::memory_classes::{
            ClassMemoryDraft, MemoryClass, ProjectWriteCap, UserWriteCap, VerifierWriteCap,
        };
        let sys = Arc::new(ClassedMemorySystem::open_in_memory("ws-compile").unwrap());
        sys.write_semantic_project(
            &ProjectWriteCap::mint(),
            "distill",
            ClassMemoryDraft::new(
                "semantic_project: the context compiler lives in crates/hawking-context",
            )
            .with_importance(0.95)
            .with_evidence(vec!["path:crates/hawking-context".into()]),
        )
        .unwrap();
        sys.write_user(
            &UserWriteCap::mint(),
            "user_intent",
            ClassMemoryDraft::new("user: prefer small focused diffs").with_importance(0.9),
        )
        .unwrap();
        sys.write_verification(
            &VerifierWriteCap::mint(),
            "verifier",
            ClassMemoryDraft::new("verification: six memory classes are REAL_WIRED")
                .with_evidence_tier("proven")
                .with_importance(1.0)
                .with_run("run-compile-test"),
        )
        .unwrap();
        let budgets = ClassBudgets {
            working: 32,
            episodic: 32,
            semantic_project: 256,
            procedural: 32,
            user: 128,
            verification: 128,
        };
        let mut compiler = ContextCompiler::new();
        compiler.add_source(ClassedMemoryContextSource::new(sys.clone(), budgets));
        let compiled = compiler
            .compile(CompileInput {
                profile: ContextProfile::coding_default(2048),
                model: model(),
                task: "context compiler memory classes".to_string(),
            })
            .await
            .unwrap();
        let mem_spans: Vec<_> = compiled
            .manifest
            .retained
            .iter()
            .filter(|s| matches!(s.source, ContextSourceKind::Memory))
            .collect();
        assert!(mem_spans.len() >= 2);
        let titles: Vec<&str> = mem_spans.iter().map(|s| s.title.as_str()).collect();
        assert!(titles.iter().any(|t| t.contains("semantic_project")));
        assert!(titles
            .iter()
            .any(|t| t.contains("user") || t.contains("verification")));
        let ret = sys.last_retrieval().expect("retrieval recorded");
        assert_eq!(ret.slices.len(), 6);
        let sem = ret.slice(MemoryClass::SemanticProject).unwrap();
        let ver = ret.slice(MemoryClass::Verification).unwrap();
        assert_eq!(sem.budget_tokens, 256);
        assert_eq!(ver.budget_tokens, 128);
        assert!(sem.budget_tokens != ver.budget_tokens);
        assert!(sem.used_tokens <= sem.budget_tokens);
        assert!(ver.used_tokens <= ver.budget_tokens);
        let lines = ret.budget_explanations();
        assert!(lines
            .iter()
            .any(|l| l.contains("memory_class.semantic_project")));
        assert!(lines
            .iter()
            .any(|l| l.contains("memory_class.verification")));
    }
}
