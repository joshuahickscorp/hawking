use futures::future::BoxFuture;
use hide_core::error::Result;
use hide_core::types::{BlobRef, Provenance};
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

pub struct InMemorySourceAdapter {
    name: String,
    source_type: SourceType,
    records: RwLock<BTreeMap<String, StructuredDoc>>,
}

impl InMemorySourceAdapter {
    pub fn new(name: impl Into<String>, source_type: SourceType) -> Self {
        Self {
            name: name.into(),
            source_type,
            records: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn insert(&self, doc: StructuredDoc) {
        self.records.write().insert(doc.source.id.clone(), doc);
    }
}

impl Default for SourceQuality {
    fn default() -> Self {
        Self {
            authority: 0.5,
            recency: 0.5,
            independence: 0.5,
            reproducibility: 0.5,
        }
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
            let mut hits: Vec<_> =
                self.records
                    .read()
                    .values()
                    .filter(|doc| {
                        doc.title
                            .to_lowercase()
                            .contains(&query.query.to_lowercase())
                            || doc.sections.iter().any(|s| {
                                s.text.to_lowercase().contains(&query.query.to_lowercase())
                            })
                    })
                    .map(|doc| doc.source.clone())
                    .collect();
            hits.sort_by(|a, b| {
                b.quality
                    .score()
                    .partial_cmp(&a.quality.score())
                    .unwrap_or(std::cmp::Ordering::Equal)
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
