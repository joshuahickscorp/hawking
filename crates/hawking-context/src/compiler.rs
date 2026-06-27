use crate::budget::estimate_tokens;
use crate::manifest::{
    ContextManifest, ContextSourceKind, ContextSpan, DropReason, DroppedContextSpan,
};
use crate::profiles::ContextProfile;
use futures::future::BoxFuture;
use hide_core::error::Result;
use hide_core::runtime::ModelDescriptor;
use hide_core::types::Provenance;

#[derive(Debug, Clone)]
pub struct CompileInput {
    pub profile: ContextProfile,
    pub model: ModelDescriptor,
    pub task: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledContext {
    pub prompt: String,
    pub manifest: ContextManifest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContextCandidate {
    pub id: String,
    pub source: ContextSourceKind,
    pub title: String,
    pub text: String,
    pub score: f32,
    pub provenance: Provenance,
}

impl ContextCandidate {
    pub fn token_count(&self) -> usize {
        estimate_tokens(&self.text)
    }
}

pub trait ContextSource: Send + Sync {
    fn name(&self) -> &str;
    fn gather<'a>(
        &'a self,
        input: &'a CompileInput,
    ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>>;
}

#[derive(Default)]
pub struct ContextCompiler {
    sources: Vec<Box<dyn ContextSource>>,
}

impl ContextCompiler {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    pub fn add_source<S: ContextSource + 'static>(&mut self, source: S) {
        self.sources.push(Box::new(source));
    }

    pub async fn compile(&self, input: CompileInput) -> Result<CompiledContext> {
        let mut candidates = Vec::new();
        for source in &self.sources {
            candidates.extend(source.gather(&input).await?);
        }
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });

        let capacity = input.profile.budget.available_input();
        let mut used = 0usize;
        let mut manifest = ContextManifest::new(input.model.context_tokens);
        let mut prompt_parts = Vec::new();
        for candidate in candidates {
            let token_count = candidate.token_count();
            if used + token_count <= capacity {
                used += token_count;
                prompt_parts.push(candidate.text.clone());
                manifest.retained.push(ContextSpan {
                    id: candidate.id,
                    source: candidate.source,
                    title: candidate.title,
                    text: candidate.text,
                    token_count,
                    score: candidate.score,
                    provenance: candidate.provenance,
                    blob_ref: None,
                });
            } else {
                manifest.dropped.push(DroppedContextSpan {
                    id: candidate.id,
                    source: candidate.source,
                    token_count,
                    score: candidate.score,
                    reason: DropReason::Budget,
                });
            }
        }
        manifest.used_tokens = used;
        Ok(CompiledContext {
            prompt: prompt_parts.join("\n\n"),
            manifest,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::ContextProfile;
    use hide_core::ids::ModelId;
    use hide_core::runtime::{ModelArchitecture, ModelDescriptor};

    struct StaticSource(Vec<ContextCandidate>);

    impl ContextSource for StaticSource {
        fn name(&self) -> &str {
            "static"
        }

        fn gather<'a>(
            &'a self,
            _input: &'a CompileInput,
        ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
            Box::pin(async { Ok(self.0.clone()) })
        }
    }

    #[tokio::test]
    async fn compiler_keeps_highest_scoring_spans_under_budget() {
        let mut compiler = ContextCompiler::new();
        compiler.add_source(StaticSource(vec![
            ContextCandidate {
                id: "low".to_string(),
                source: ContextSourceKind::Code,
                title: "low".to_string(),
                text: "x ".repeat(200),
                score: 0.1,
                provenance: Provenance::trusted("test"),
            },
            ContextCandidate {
                id: "high".to_string(),
                source: ContextSourceKind::Code,
                title: "high".to_string(),
                text: "important".to_string(),
                score: 1.0,
                provenance: Provenance::trusted("test"),
            },
        ]));
        let input = CompileInput {
            profile: ContextProfile::coding_default(64),
            model: ModelDescriptor {
                id: ModelId::new(),
                name: "test".to_string(),
                architecture: ModelArchitecture::Transformer,
                context_tokens: 64,
                tokenizer_signature: "test".to_string(),
                footprint_mb: 1,
            },
            task: "test".to_string(),
        };
        let compiled = compiler.compile(input).await.unwrap();
        assert_eq!(compiled.manifest.retained[0].id, "high");
        assert_eq!(compiled.manifest.dropped.len(), 1);
    }
}
