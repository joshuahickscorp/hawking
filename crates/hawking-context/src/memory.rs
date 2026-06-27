use futures::future::BoxFuture;
use hide_core::error::Result;
use hide_core::ids::{now_ms, TimestampMs};
use hide_core::types::Provenance;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub kind: MemoryKind,
    pub text: String,
    pub importance: f32,
    pub created_at_ms: TimestampMs,
    pub last_used_at_ms: Option<TimestampMs>,
    pub provenance: Provenance,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Working,
    Episodic,
    Semantic,
    Procedural,
    Project,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryQuery {
    pub text: String,
    pub kinds: Vec<MemoryKind>,
    pub top_k: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedMemory {
    pub record: MemoryRecord,
    pub score: f32,
}

pub trait MemoryStore: Send + Sync {
    fn put<'a>(&'a self, record: MemoryRecord) -> BoxFuture<'a, Result<()>>;
    fn query<'a>(&'a self, query: MemoryQuery) -> BoxFuture<'a, Result<Vec<RankedMemory>>>;
}

#[derive(Default)]
pub struct InMemoryMemoryStore {
    records: RwLock<BTreeMap<String, MemoryRecord>>,
}

impl InMemoryMemoryStore {
    pub fn record(
        kind: MemoryKind,
        text: impl Into<String>,
        provenance: Provenance,
    ) -> MemoryRecord {
        MemoryRecord {
            id: format!("mem_{}", now_ms()),
            kind,
            text: text.into(),
            importance: 0.5,
            created_at_ms: now_ms(),
            last_used_at_ms: None,
            provenance,
            tags: Vec::new(),
        }
    }
}

impl MemoryStore for InMemoryMemoryStore {
    fn put<'a>(&'a self, record: MemoryRecord) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.records.write().insert(record.id.clone(), record);
            Ok(())
        })
    }

    fn query<'a>(&'a self, query: MemoryQuery) -> BoxFuture<'a, Result<Vec<RankedMemory>>> {
        Box::pin(async move {
            let mut ranked: Vec<_> = self
                .records
                .read()
                .values()
                .filter(|record| query.kinds.is_empty() || query.kinds.contains(&record.kind))
                .map(|record| {
                    let relevance = lexical_overlap(&query.text, &record.text);
                    let score = relevance + record.importance;
                    RankedMemory {
                        record: record.clone(),
                        score,
                    }
                })
                .collect();
            ranked.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked.truncate(query.top_k);
            Ok(ranked)
        })
    }
}

fn lexical_overlap(a: &str, b: &str) -> f32 {
    let a_words: Vec<_> = a.split_whitespace().collect();
    if a_words.is_empty() {
        return 0.0;
    }
    let hits = a_words.iter().filter(|word| b.contains(**word)).count();
    hits as f32 / a_words.len() as f32
}
