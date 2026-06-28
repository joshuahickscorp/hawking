//! Ingestion adapters (bible ch.08 §4.4).
//!
//! Every source enters through the [`SourceAdapter`] trait. This crate ships:
//!
//! * [`InMemorySourceAdapter`] — a fixture adapter for tests (substring search
//!   over inserted docs), now content-addressing every doc it returns.
//! * [`ArxivAdapter`] — a **real** adapter over the public arXiv Atom API via
//!   `reqwest`, returning [`StructuredDoc`]s whose `content_hash` is populated
//!   from the fetched bytes (unlocking idempotent ingest + citation
//!   re-verification, §4.2.1).
//!
//! PDF full-text parsing is a documented seam: arXiv gives us title + abstract +
//! authors deterministically from the Atom feed, which is enough to build Paper
//! and Claim nodes; the PDF/LaTeX-source body parse (§4.5) is left for the
//! vision/PDFium pipeline and marked below.

use crate::cas;
use futures::future::BoxFuture;
use hide_core::error::Result;
use hide_core::types::{BlobRef, Provenance, TrustLevel};
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
    /// CAS receipt for this section's *own* canonical evidence bytes, populated
    /// when the section is pinned (`pin_doc_evidence`). The pinned bytes are
    /// exactly the bytes the section's claim node is content-addressed over, so
    /// the claim id, the evidence blob, and the re-verification hash all agree
    /// on one canonical byte source (§4.7.3 — citation re-verification soundness).
    #[serde(default)]
    pub evidence: Option<SectionEvidence>,
}

/// An immutable CAS receipt for a section's evidence bytes: the blob ref the
/// bytes are stored under, and the blake3 hash of *exactly* those bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionEvidence {
    pub blob: BlobRef,
    pub content_hash: String,
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

/// Build a [`StructuredDoc`] from title + abstract text, content-addressing the
/// id and a whole-doc `content_hash` from the normalized body. Shared by every
/// adapter so all docs are addressed the same way.
pub fn structured_doc_from_text(
    source_type: SourceType,
    external_id: &str,
    uri: &str,
    title: &str,
    abstract_text: &str,
    provenance: Provenance,
    quality: SourceQuality,
) -> StructuredDoc {
    let body = abstract_text.trim();
    let content_hash = cas::blake3_hex(cas::normalize_text(&format!("{title}\n{body}")).as_bytes());
    let doc_id = format!("doc:{content_hash}");
    let span = DocSpan {
        id: cas::content_id("span", body),
        start_char: 0,
        end_char: body.chars().count(),
    };
    let sections = if body.is_empty() {
        Vec::new()
    } else {
        vec![DocSection {
            heading: "Abstract".to_string(),
            text: body.to_string(),
            spans: vec![span],
            evidence: None,
        }]
    };
    StructuredDoc {
        id: doc_id,
        source: SourceRecord {
            id: external_id.to_string(),
            source_type,
            title: title.to_string(),
            uri: uri.to_string(),
            content_hash: Some(content_hash),
            quality,
            provenance,
        },
        title: title.to_string(),
        sections,
        references: Vec::new(),
        blob: None,
    }
}

// ───────────────────────────── In-memory fixture ─────────────────────────────

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
            let needle = query.query.to_lowercase();
            let mut hits: Vec<_> = self
                .records
                .read()
                .values()
                .filter(|doc| {
                    doc.title.to_lowercase().contains(&needle)
                        || doc
                            .sections
                            .iter()
                            .any(|s| s.text.to_lowercase().contains(&needle))
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

// ──────────────────────────────── arXiv (real) ───────────────────────────────

/// A real [`SourceAdapter`] over the public arXiv Atom API.
///
/// `search` issues `GET https://export.arxiv.org/api/query?search_query=...` and
/// parses the returned Atom feed; `fetch` rebuilds the full [`StructuredDoc`]
/// (title + abstract) and populates `content_hash`. Full PDF/LaTeX body parsing
/// is the documented §4.5 seam — the abstract is enough for Paper/Claim nodes.
pub struct ArxivAdapter {
    base_url: String,
    client: reqwest::Client,
}

impl Default for ArxivAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl ArxivAdapter {
    pub fn new() -> Self {
        Self {
            base_url: "https://export.arxiv.org/api/query".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Point the adapter at an alternate base URL (used by tests to hit a local
    /// fixture server instead of the live API).
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: reqwest::Client::new(),
        }
    }

    async fn fetch_feed(&self, query: &str, limit: usize) -> Result<String> {
        let url = format!(
            "{}?search_query=all:{}&start=0&max_results={}",
            self.base_url,
            urlencode(query),
            limit.max(1)
        );
        let resp = self
            .client
            .get(&url)
            .header("User-Agent", "hide-research/0.1")
            .send()
            .await
            .map_err(|e| hide_core::HideError::RuntimeUnavailable(format!("arxiv request: {e}")))?;
        let text = resp
            .text()
            .await
            .map_err(|e| hide_core::HideError::RuntimeUnavailable(format!("arxiv body: {e}")))?;
        Ok(text)
    }
}

impl SourceAdapter for ArxivAdapter {
    fn name(&self) -> &str {
        "arxiv"
    }

    fn source_type(&self) -> SourceType {
        SourceType::Arxiv
    }

    fn search<'a>(&'a self, query: &'a SourceQuery) -> BoxFuture<'a, Result<Vec<SourceRecord>>> {
        Box::pin(async move {
            let feed = self.fetch_feed(&query.query, query.limit).await?;
            let entries = parse_arxiv_feed(&feed);
            Ok(entries.into_iter().map(|e| e.into_record()).collect())
        })
    }

    fn fetch<'a>(&'a self, record: &'a SourceRecord) -> BoxFuture<'a, Result<StructuredDoc>> {
        Box::pin(async move {
            // The record id is the arXiv id; re-query for its single entry so we
            // have the abstract regardless of how the record was obtained.
            let feed = self.fetch_feed(&record.id, 1).await?;
            let mut entries = parse_arxiv_feed(&feed);
            let entry = entries.pop().ok_or_else(|| {
                hide_core::HideError::NotFound(format!("arxiv entry {}", record.id))
            })?;
            Ok(entry.into_doc())
        })
    }
}

/// One parsed Atom `<entry>`.
#[derive(Debug, Clone)]
pub struct ArxivEntry {
    pub id: String,
    pub title: String,
    pub summary: String,
    pub uri: String,
}

impl ArxivEntry {
    fn provenance() -> Provenance {
        // Network-sourced → not trusted; tagged for the egress audit.
        Provenance {
            source: "arxiv".to_string(),
            trust: TrustLevel::Network,
            confidence: 0.9,
            labels: vec!["arxiv".to_string()],
            derived_from: Vec::new(),
        }
    }

    pub fn into_record(self) -> SourceRecord {
        let doc = self.into_doc();
        doc.source
    }

    pub fn into_doc(self) -> StructuredDoc {
        structured_doc_from_text(
            SourceType::Arxiv,
            &self.id,
            &self.uri,
            &self.title,
            &self.summary,
            Self::provenance(),
            SourceQuality {
                authority: 0.8,
                recency: 0.7,
                independence: 0.6,
                reproducibility: 0.6,
            },
        )
    }
}

/// Parse an arXiv Atom feed into entries. Deterministic, dependency-light: pulls
/// the inner text of `<entry>` blocks and the `<id>/<title>/<summary>` tags.
pub fn parse_arxiv_feed(xml: &str) -> Vec<ArxivEntry> {
    let mut out = Vec::new();
    for block in between_all(xml, "<entry>", "</entry>") {
        let id_raw = first_between(&block, "<id>", "</id>").unwrap_or_default();
        let title = first_between(&block, "<title>", "</title>")
            .map(|s| collapse_ws(&unescape_xml(&s)))
            .unwrap_or_default();
        let summary = first_between(&block, "<summary>", "</summary>")
            .map(|s| collapse_ws(&unescape_xml(&s)))
            .unwrap_or_default();
        if title.is_empty() {
            continue;
        }
        // arXiv id is the trailing path component of the <id> URL.
        let short_id = id_raw.rsplit('/').next().unwrap_or(&id_raw).trim().to_string();
        out.push(ArxivEntry {
            id: short_id,
            title,
            summary,
            uri: id_raw.trim().to_string(),
        });
    }
    out
}

// ────────────────────────────── tiny xml helpers ─────────────────────────────

fn between_all(haystack: &str, open: &str, close: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = haystack;
    while let Some(start) = rest.find(open) {
        let after = &rest[start + open.len()..];
        if let Some(end) = after.find(close) {
            out.push(after[..end].to_string());
            rest = &after[end + close.len()..];
        } else {
            break;
        }
    }
    out
}

fn first_between(haystack: &str, open: &str, close: &str) -> Option<String> {
    let start = haystack.find(open)? + open.len();
    let end = haystack[start..].find(close)? + start;
    Some(haystack[start..end].to_string())
}

fn collapse_ws(s: &str) -> String {
    cas::normalize_text(s)
}

fn unescape_xml(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
}

fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2309.06180v1</id>
    <title>Efficient Memory Management for Large Language Model Serving with PagedAttention</title>
    <summary>High throughput serving of LLMs requires batching. We propose
PagedAttention, which reduces KV cache &amp; memory waste.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2205.14135v2</id>
    <title>FlashAttention</title>
    <summary>An IO-aware exact attention algorithm.</summary>
  </entry>
</feed>"#;

    #[test]
    fn parses_arxiv_feed_entries() {
        let entries = parse_arxiv_feed(SAMPLE);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id, "2309.06180v1");
        assert!(entries[0].title.contains("PagedAttention"));
        assert!(entries[0].summary.contains("KV cache &"));
    }

    #[test]
    fn arxiv_doc_has_populated_content_hash_and_is_addressed() {
        let entries = parse_arxiv_feed(SAMPLE);
        let doc = entries[0].clone().into_doc();
        assert!(doc.source.content_hash.is_some());
        assert!(doc.id.starts_with("doc:"));
        // Idempotent: same entry → same id + hash.
        let doc2 = parse_arxiv_feed(SAMPLE)[0].clone().into_doc();
        assert_eq!(doc.id, doc2.id);
        assert_eq!(doc.source.content_hash, doc2.source.content_hash);
        assert_eq!(doc.source.provenance.trust, TrustLevel::Network);
    }

    #[test]
    fn urlencode_escapes_spaces() {
        assert_eq!(urlencode("kv cache"), "kv%20cache");
    }
}
