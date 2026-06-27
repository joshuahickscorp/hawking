use crate::graph::{Occurrence, Symbol};
use futures::future::BoxFuture;
use hide_core::error::Result;
use hide_core::types::{FileSpan, TextRange};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchQuery {
    pub text: String,
    pub limit: usize,
    pub include_symbols: bool,
    pub include_lexical: bool,
    pub include_semantic: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    pub span: FileSpan,
    pub title: String,
    pub snippet: String,
    pub score: f32,
    pub source: SearchResultSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchResultSource {
    Symbol,
    Lexical,
    Semantic,
    Graph,
}

pub trait CodeIndex: Send + Sync {
    fn search<'a>(&'a self, query: SearchQuery) -> BoxFuture<'a, Result<Vec<SearchResult>>>;
    fn definition<'a>(&'a self, symbol: &'a str) -> BoxFuture<'a, Result<Vec<Occurrence>>>;
    fn references<'a>(&'a self, symbol: &'a str) -> BoxFuture<'a, Result<Vec<Occurrence>>>;
    fn health<'a>(&'a self) -> BoxFuture<'a, Result<IndexHealth>>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexHealth {
    pub generation: u64,
    pub indexed_files: usize,
    pub stale_files: usize,
    pub degraded: Vec<String>,
}

#[derive(Default)]
pub struct InMemoryCodeIndex {
    generation: RwLock<u64>,
    symbols: RwLock<BTreeMap<String, Symbol>>,
    occurrences: RwLock<BTreeMap<String, Vec<Occurrence>>>,
    files: RwLock<BTreeMap<PathBuf, IndexedFile>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexedFile {
    pub path: PathBuf,
    pub content: String,
    pub content_hash: Option<String>,
}

impl InMemoryCodeIndex {
    pub fn add_symbol(&self, symbol: Symbol) {
        *self.generation.write() += 1;
        self.symbols
            .write()
            .insert(symbol.qualified_name.clone(), symbol);
    }

    pub fn add_occurrence(&self, occurrence: Occurrence) {
        *self.generation.write() += 1;
        self.occurrences
            .write()
            .entry(occurrence.symbol.clone())
            .or_default()
            .push(occurrence);
    }

    pub fn add_text_file(
        &self,
        path: impl Into<PathBuf>,
        content: impl Into<String>,
        content_hash: Option<String>,
    ) {
        let path = path.into();
        let content = content.into();
        self.index_simple_symbols(&path, &content);
        self.files.write().insert(
            path.clone(),
            IndexedFile {
                path,
                content,
                content_hash,
            },
        );
        *self.generation.write() += 1;
    }

    pub fn index_path(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;
        self.add_text_file(path.to_path_buf(), content, None);
        Ok(())
    }

    fn index_simple_symbols(&self, path: &Path, content: &str) {
        for (idx, line) in content.lines().enumerate() {
            if let Some((kind, name)) = simple_definition(line) {
                let qualified_name = format!("{}::{}", path.display(), name);
                self.add_symbol(Symbol {
                    qualified_name: qualified_name.clone(),
                    name: name.clone(),
                    kind,
                    file: path.to_string_lossy().to_string(),
                });
                self.add_occurrence(Occurrence {
                    symbol: qualified_name,
                    file: path.to_string_lossy().to_string(),
                    range: Some(TextRange {
                        start_line: idx as u32 + 1,
                        start_col: 1,
                        end_line: idx as u32 + 1,
                        end_col: line.len() as u32 + 1,
                    }),
                    role: "definition".to_string(),
                });
            }
        }
    }
}

impl CodeIndex for InMemoryCodeIndex {
    fn search<'a>(&'a self, query: SearchQuery) -> BoxFuture<'a, Result<Vec<SearchResult>>> {
        Box::pin(async move {
            let mut results = Vec::new();
            if query.include_symbols {
                let needle = query.text.to_lowercase();
                for symbol in self.symbols.read().values() {
                    if symbol.qualified_name.to_lowercase().contains(&needle)
                        || symbol.name.to_lowercase().contains(&needle)
                    {
                        results.push(SearchResult {
                            span: FileSpan {
                                path: PathBuf::from(&symbol.file),
                                range: None,
                                content_hash: None,
                            },
                            title: symbol.qualified_name.clone(),
                            snippet: symbol.kind.clone(),
                            score: 1.0,
                            source: SearchResultSource::Symbol,
                        });
                    }
                }
            }
            if query.include_lexical && !query.text.trim().is_empty() {
                let needle = query.text.to_lowercase();
                for file in self.files.read().values() {
                    for (idx, line) in file.content.lines().enumerate() {
                        let lower = line.to_lowercase();
                        if let Some(col) = lower.find(&needle) {
                            let score = lexical_score(&lower, &needle);
                            results.push(SearchResult {
                                span: FileSpan {
                                    path: file.path.clone(),
                                    range: Some(TextRange {
                                        start_line: idx as u32 + 1,
                                        start_col: col as u32 + 1,
                                        end_line: idx as u32 + 1,
                                        end_col: (col + query.text.len()) as u32 + 1,
                                    }),
                                    content_hash: file.content_hash.clone(),
                                },
                                title: file.path.display().to_string(),
                                snippet: line.trim().to_string(),
                                score,
                                source: SearchResultSource::Lexical,
                            });
                        }
                    }
                }
            }
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(query.limit);
            Ok(results)
        })
    }

    fn definition<'a>(&'a self, symbol: &'a str) -> BoxFuture<'a, Result<Vec<Occurrence>>> {
        Box::pin(async move {
            Ok(self
                .occurrences
                .read()
                .get(symbol)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|occ| occ.role == "definition")
                .collect())
        })
    }

    fn references<'a>(&'a self, symbol: &'a str) -> BoxFuture<'a, Result<Vec<Occurrence>>> {
        Box::pin(async move {
            Ok(self
                .occurrences
                .read()
                .get(symbol)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|occ| occ.role == "reference")
                .collect())
        })
    }

    fn health<'a>(&'a self) -> BoxFuture<'a, Result<IndexHealth>> {
        Box::pin(async move {
            Ok(IndexHealth {
                generation: *self.generation.read(),
                indexed_files: self.files.read().len(),
                stale_files: 0,
                degraded: Vec::new(),
            })
        })
    }
}

fn simple_definition(line: &str) -> Option<(String, String)> {
    let trimmed = line.trim_start();
    for (prefix, kind) in [
        ("pub fn ", "function"),
        ("fn ", "function"),
        ("pub struct ", "struct"),
        ("struct ", "struct"),
        ("pub enum ", "enum"),
        ("enum ", "enum"),
        ("pub trait ", "trait"),
        ("trait ", "trait"),
    ] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return Some((kind.to_string(), name));
            }
        }
    }
    None
}

fn lexical_score(line: &str, needle: &str) -> f32 {
    let occurrences = line.matches(needle).count().max(1) as f32;
    let density = needle.len() as f32 / line.len().max(needle.len()) as f32;
    0.5 + occurrences.min(5.0) * 0.1 + density.min(0.4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Symbol;

    #[tokio::test]
    async fn search_finds_registered_symbol() {
        let index = InMemoryCodeIndex::default();
        index.add_symbol(Symbol {
            qualified_name: "crate::engine::Engine".to_string(),
            name: "Engine".to_string(),
            kind: "trait".to_string(),
            file: "src/engine.rs".to_string(),
        });
        let results = index
            .search(SearchQuery {
                text: "Engine".to_string(),
                limit: 5,
                include_symbols: true,
                include_lexical: false,
                include_semantic: false,
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn index_text_file_supports_lexical_search_and_simple_symbols() {
        let index = InMemoryCodeIndex::default();
        index.add_text_file(
            "src/lib.rs",
            "pub struct SearchEngine {}\nimpl SearchEngine { fn query(&self) {} }\n",
            Some("hash".to_string()),
        );
        let lexical = index
            .search(SearchQuery {
                text: "SearchEngine".to_string(),
                limit: 5,
                include_symbols: false,
                include_lexical: true,
                include_semantic: false,
            })
            .await
            .unwrap();
        assert_eq!(lexical.len(), 2);
        assert_eq!(lexical[0].source, SearchResultSource::Lexical);

        let symbols = index
            .search(SearchQuery {
                text: "SearchEngine".to_string(),
                limit: 5,
                include_symbols: true,
                include_lexical: false,
                include_semantic: false,
            })
            .await
            .unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(index.health().await.unwrap().indexed_files, 1);
    }
}
