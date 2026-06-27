//! Built-in context-source scaffolds.

use crate::compiler::{CompileInput, ContextCandidate, ContextSource};
use crate::manifest::ContextSourceKind;
use futures::future::BoxFuture;
use hawking_index::{CodeIndex, SearchQuery};
use hide_core::error::Result;
use hide_core::types::Provenance;
use std::sync::Arc;

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
                .map(|(idx, (title, text, score))| ContextCandidate {
                    id: format!("{}:{idx}", self.name),
                    source: self.source.clone(),
                    title: title.clone(),
                    text: text.clone(),
                    score: *score,
                    provenance: Provenance::trusted(self.name.clone()),
                })
                .collect())
        })
    }
}

pub struct CodeIndexContextSource {
    pub name: String,
    pub index: Arc<dyn CodeIndex>,
    pub limit: usize,
}

impl CodeIndexContextSource {
    pub fn new(index: Arc<dyn CodeIndex>, limit: usize) -> Self {
        Self {
            name: "code_index".to_string(),
            index,
            limit,
        }
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
            let results = self
                .index
                .search(SearchQuery {
                    text: input.task.clone(),
                    limit: self.limit,
                    include_symbols: true,
                    include_lexical: true,
                    include_semantic: false,
                })
                .await?;
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
                    ContextCandidate {
                        id: format!("{}:{idx}", self.name),
                        source: ContextSourceKind::Code,
                        title: format!("{}{}", result.title, range),
                        text: result.snippet,
                        score: result.score,
                        provenance: Provenance::trusted("code-index"),
                    }
                })
                .collect())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{CompileInput, ContextCompiler};
    use crate::profiles::ContextProfile;
    use hawking_index::InMemoryCodeIndex;
    use hide_core::ids::ModelId;
    use hide_core::runtime::{ModelArchitecture, ModelDescriptor};

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
                model: ModelDescriptor {
                    id: ModelId::new(),
                    name: "test".to_string(),
                    architecture: ModelArchitecture::Transformer,
                    context_tokens: 1024,
                    tokenizer_signature: "test".to_string(),
                    footprint_mb: 1,
                },
                task: "context compiler".to_string(),
            })
            .await
            .unwrap();
        assert!(compiled.prompt.contains("context compiler"));
        assert_eq!(compiled.manifest.retained.len(), 1);
    }
}
