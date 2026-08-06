//! Cheap-first orchestration of the perception path.
//!
//! Enforces: never OCR every page at max cost. Stages escalate only when the
//! query needs structure that free text cannot provide, and only within budget.

use crate::coords::RegionKind;
use crate::error::{PerceptionError, Result};
use crate::service::DocumentService;
use crate::types::{
    CostTier, DocumentHandle, StructuredEvidence, TableParse,
};
use serde::{Deserialize, Serialize};

/// Ordered stages of the default path (bible §19).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStage {
    Inspect,
    RetrievePages,
    DetectRegions,
    ParseRegion,
    ParseTable,
    ParseChart,
    VerifyStructure,
    CiteCoordinates,
}

impl PipelineStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Inspect => "inspect",
            Self::RetrievePages => "retrieve_pages",
            Self::DetectRegions => "detect_regions",
            Self::ParseRegion => "parse_region",
            Self::ParseTable => "parse_table",
            Self::ParseChart => "parse_chart",
            Self::VerifyStructure => "verify_structure",
            Self::CiteCoordinates => "cite_coordinates",
        }
    }
}

/// Soft budget so callers can refuse max-cost full-document OCR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineBudget {
    /// Remaining relative cost units (see [`CostTier::cost_units`]).
    pub cost_units: u64,
    /// Hard ceiling on OCR/vision regions.
    pub max_selective_regions: u32,
    /// If false, never escalate past page text (no OCR/vision).
    pub allow_ocr: bool,
    pub allow_vision: bool,
}

impl Default for PipelineBudget {
    fn default() -> Self {
        Self {
            cost_units: 500,
            max_selective_regions: 16,
            allow_ocr: false,
            allow_vision: false,
        }
    }
}

impl PipelineBudget {
    pub fn charge(&mut self, stage: &'static str, tier: CostTier) -> Result<()> {
        let units = tier.cost_units();
        if self.cost_units < units {
            return Err(PerceptionError::BudgetExhausted {
                stage,
                remaining: self.cost_units,
            });
        }
        self.cost_units -= units;
        Ok(())
    }
}

/// Query-shaped request for the orchestrator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptionQuery {
    /// Free-text need (used to rank pages when free text exists).
    pub need: String,
    /// Optional page allow-list; empty means "choose relevant pages".
    #[serde(default)]
    pub pages: Vec<u32>,
    /// Region kinds the caller cares about (empty = all).
    #[serde(default)]
    pub kinds: Vec<RegionKind>,
    pub budget: PipelineBudget,
}

impl PerceptionQuery {
    pub fn new(need: impl Into<String>) -> Self {
        Self {
            need: need.into(),
            pages: Vec::new(),
            kinds: Vec::new(),
            budget: PipelineBudget::default(),
        }
    }
}

/// Orchestrates a [`DocumentService`] along the cheap-first path.
pub struct PerceptionPipeline<S: DocumentService> {
    pub service: S,
}

impl<S: DocumentService> PerceptionPipeline<S> {
    pub fn new(service: S) -> Self {
        Self { service }
    }

    /// Run the default path. Never full-document max-cost OCR.
    pub fn run(
        &self,
        handle: &DocumentHandle,
        mut query: PerceptionQuery,
    ) -> Result<StructuredEvidence> {
        // 1. Cheap inspect
        query.budget.charge(PipelineStage::Inspect.as_str(), CostTier::Metadata)?;
        let meta = self.service.inspect(handle)?;
        let mut evidence = StructuredEvidence::empty(handle.clone());
        evidence.max_cost_tier = CostTier::Metadata;
        if let Some(t) = &meta.title {
            evidence.notes.push(format!("title={t}"));
        }

        // 2. Select pages: explicit list, else rank by free-text overlap / presence.
        let pages = if query.pages.is_empty() {
            select_pages(&meta, &query.need)
        } else {
            query.pages.clone()
        };
        if pages.is_empty() {
            evidence
                .notes
                .push("no pages selected; returning metadata-only evidence".into());
            return Ok(evidence);
        }

        query
            .budget
            .charge(PipelineStage::RetrievePages.as_str(), CostTier::PageText)?;
        let page_contents = self.service.retrieve_pages(handle, &pages)?;
        evidence.pages_used = page_contents.iter().map(|p| p.page).collect();
        evidence.max_cost_tier = CostTier::PageText;

        // 3. Detect regions on retrieved pages only.
        let kind_filter = if query.kinds.is_empty() {
            None
        } else {
            Some(query.kinds.as_slice())
        };
        query
            .budget
            .charge(PipelineStage::DetectRegions.as_str(), CostTier::PageText)?;
        let regions = self
            .service
            .detect_regions(handle, &page_contents, kind_filter)?;

        // 4–6. Selective parse: only relevant regions, capped by budget.
        let mut selective = 0u32;
        let mut tables: Vec<TableParse> = Vec::new();
        for region in regions {
            if selective >= query.budget.max_selective_regions {
                evidence.notes.push(format!(
                    "selective region cap {} reached",
                    query.budget.max_selective_regions
                ));
                break;
            }

            let needs_structure = region.kind.prefers_structured_parse();
            let tier = if needs_structure && query.budget.allow_ocr {
                CostTier::RegionOcr
            } else {
                CostTier::PageText
            };
            if tier >= CostTier::RegionOcr && !query.budget.allow_ocr {
                // Stay on free text for this region.
                if let Ok(parsed) = self.service.parse_region(handle, &region) {
                    if !parsed.text.is_empty() {
                        let cite = self.service.cite_coordinates(
                            handle,
                            region.box_,
                            &parsed.text,
                            Some(&region.id),
                        )?;
                        evidence.citations.push(cite);
                        evidence.regions.push(parsed);
                        selective += 1;
                    }
                }
                continue;
            }

            query
                .budget
                .charge(PipelineStage::ParseRegion.as_str(), tier)?;
            evidence.max_cost_tier = evidence.max_cost_tier.max(tier);

            match region.kind {
                RegionKind::Table => {
                    if let Ok(table) = self.service.parse_table(handle, &region) {
                        let quote = format_table_quote(&table);
                        let cite = self.service.cite_coordinates(
                            handle,
                            table.box_,
                            &quote,
                            Some(&table.region_id),
                        )?;
                        evidence.citations.push(cite);
                        tables.push(table);
                        selective += 1;
                        continue;
                    }
                }
                RegionKind::Chart => {
                    if let Ok(chart) = self.service.parse_chart(handle, &region) {
                        let quote = chart
                            .title
                            .clone()
                            .unwrap_or_else(|| "chart".into());
                        let cite = self.service.cite_coordinates(
                            handle,
                            chart.box_,
                            &quote,
                            Some(&chart.region_id),
                        )?;
                        evidence.citations.push(cite);
                        evidence.charts.push(chart);
                        selective += 1;
                        continue;
                    }
                }
                _ => {}
            }

            let parsed = self.service.parse_region(handle, &region)?;
            if !parsed.text.is_empty() {
                let cite = self.service.cite_coordinates(
                    handle,
                    region.box_,
                    &parsed.text,
                    Some(&region.id),
                )?;
                evidence.citations.push(cite);
            }
            evidence.regions.push(parsed);
            selective += 1;
        }

        evidence.tables = tables;

        // 7. Verify structure
        let report = self.service.verify_structure(
            handle,
            &evidence.regions,
            &evidence.tables,
        )?;
        if !report.ok {
            for issue in report.issues {
                evidence.notes.push(format!("structure: {issue}"));
            }
        } else {
            evidence.notes.push("structure verified".into());
        }

        Ok(evidence)
    }
}

/// Rank pages by free-text presence + naive term overlap with `need`.
fn select_pages(meta: &crate::types::DocumentMeta, need: &str) -> Vec<u32> {
    if meta.page_summaries.is_empty() {
        return (0..meta.page_count.min(4)).collect();
    }
    let terms: Vec<&str> = need
        .split_whitespace()
        .filter(|t| t.len() > 2)
        .collect();
    let cheap = meta.cheap_text.as_deref().unwrap_or("");
    let mut scored: Vec<(u32, i64)> = meta
        .page_summaries
        .iter()
        .map(|s| {
            let mut score = s.free_text_chars as i64;
            if s.likely_has_figures {
                score += 10;
            }
            // Cheap whole-doc text is not page-aligned in the stub; boost first pages
            // that have free text when need terms appear in cheap_text.
            if !terms.is_empty() && !cheap.is_empty() {
                let lower = cheap.to_lowercase();
                for t in &terms {
                    if lower.contains(&t.to_lowercase()) {
                        score += 5;
                    }
                }
            }
            (s.page, score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1));
    scored
        .into_iter()
        .filter(|(_, s)| *s > 0)
        .take(4)
        .map(|(p, _)| p)
        .collect()
}

fn format_table_quote(table: &TableParse) -> String {
    let mut lines = Vec::new();
    if !table.headers.is_empty() {
        lines.push(table.headers.join(" | "));
    }
    for row in table.rows.iter().take(8) {
        lines.push(row.join(" | "));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stubs::StubDocumentService;

    #[test]
    fn pipeline_stays_on_page_text_when_ocr_disabled() {
        let svc = StubDocumentService::fixture_paper();
        let pipe = PerceptionPipeline::new(svc);
        let handle = DocumentHandle::new("doc:fixture-paper");
        let mut q = PerceptionQuery::new("throughput benchmark chart");
        q.budget.allow_ocr = false;
        let ev = pipe.run(&handle, q).unwrap();
        assert!(ev.max_cost_tier <= CostTier::PageText);
        assert_eq!(ev.schema, crate::PERCEPTION_SCHEMA);
        assert!(!ev.pages_used.is_empty());
        assert!(!ev.citations.is_empty());
        for c in &ev.citations {
            assert!(c.box_.is_normalized());
            assert!(!c.content_hash.is_empty());
        }
    }

    #[test]
    fn budget_zero_fails_at_inspect() {
        let svc = StubDocumentService::fixture_paper();
        let pipe = PerceptionPipeline::new(svc);
        let handle = DocumentHandle::new("doc:fixture-paper");
        let mut q = PerceptionQuery::new("anything");
        q.budget.cost_units = 0;
        let err = pipe.run(&handle, q).unwrap_err();
        assert!(matches!(
            err,
            PerceptionError::BudgetExhausted { stage: "inspect", .. }
        ));
    }
}
