//! Stable tool-name surface for agent / HCLI registration.
//!
//! These names match Ascension Bible §19. Wire adapters (hide-kernel tool
//! registry, hcli_bridge capability area) can project from this enum later
//! without inventing alternate spellings.

use crate::coords::{CoordBox, RegionKind};
use crate::error::{PerceptionError, Result};
use crate::service::DocumentService;
use crate::types::{
    ChartParse, Citation, DocumentHandle, DocumentMeta, PageContent, ParsedRegion, Region,
    RegionId, StructureReport, TableParse,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentToolName {
    Inspect,
    RetrievePages,
    DetectRegions,
    ParseRegion,
    ParseTable,
    ParseChart,
    VerifyStructure,
    CiteCoordinates,
}

impl DocumentToolName {
    pub const ALL: [Self; 8] = [
        Self::Inspect,
        Self::RetrievePages,
        Self::DetectRegions,
        Self::ParseRegion,
        Self::ParseTable,
        Self::ParseChart,
        Self::VerifyStructure,
        Self::CiteCoordinates,
    ];

    /// Fully-qualified tool id, e.g. `document.inspect`.
    pub fn qualified(self) -> String {
        format!("{}.{}", crate::PERCEPTION_TOOL_NAMESPACE, self.as_str())
    }

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

    pub fn parse_qualified(name: &str) -> Option<Self> {
        let rest = name.strip_prefix("document.")?;
        match rest {
            "inspect" => Some(Self::Inspect),
            "retrieve_pages" => Some(Self::RetrievePages),
            "detect_regions" => Some(Self::DetectRegions),
            "parse_region" => Some(Self::ParseRegion),
            "parse_table" => Some(Self::ParseTable),
            "parse_chart" => Some(Self::ParseChart),
            "verify_structure" => Some(Self::VerifyStructure),
            "cite_coordinates" => Some(Self::CiteCoordinates),
            _ => None,
        }
    }
}

/// Typed tool request envelope (JSON-friendly for future bridge methods).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tool", rename_all = "snake_case")]
pub enum ToolRequest {
    Inspect {
        handle: DocumentHandle,
    },
    RetrievePages {
        handle: DocumentHandle,
        pages: Vec<u32>,
    },
    DetectRegions {
        handle: DocumentHandle,
        pages: Vec<PageContent>,
        #[serde(default)]
        kinds: Vec<RegionKind>,
    },
    ParseRegion {
        handle: DocumentHandle,
        region: Region,
    },
    ParseTable {
        handle: DocumentHandle,
        region: Region,
    },
    ParseChart {
        handle: DocumentHandle,
        region: Region,
    },
    VerifyStructure {
        handle: DocumentHandle,
        regions: Vec<ParsedRegion>,
        tables: Vec<TableParse>,
    },
    CiteCoordinates {
        handle: DocumentHandle,
        box_: CoordBox,
        quote: String,
        #[serde(default)]
        region_id: Option<RegionId>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tool", rename_all = "snake_case")]
pub enum ToolResponse {
    Inspect { meta: DocumentMeta },
    RetrievePages { pages: Vec<PageContent> },
    DetectRegions { regions: Vec<Region> },
    ParseRegion { parsed: ParsedRegion },
    ParseTable { table: TableParse },
    ParseChart { chart: ChartParse },
    VerifyStructure { report: StructureReport },
    CiteCoordinates { citation: Citation },
}

/// Dispatch a typed tool request against any [`DocumentService`].
pub struct DocumentTool;

impl DocumentTool {
    pub fn dispatch(service: &dyn DocumentService, req: ToolRequest) -> Result<ToolResponse> {
        match req {
            ToolRequest::Inspect { handle } => Ok(ToolResponse::Inspect {
                meta: service.inspect(&handle)?,
            }),
            ToolRequest::RetrievePages { handle, pages } => Ok(ToolResponse::RetrievePages {
                pages: service.retrieve_pages(&handle, &pages)?,
            }),
            ToolRequest::DetectRegions {
                handle,
                pages,
                kinds,
            } => {
                let filter = if kinds.is_empty() {
                    None
                } else {
                    Some(kinds.as_slice())
                };
                Ok(ToolResponse::DetectRegions {
                    regions: service.detect_regions(&handle, &pages, filter)?,
                })
            }
            ToolRequest::ParseRegion { handle, region } => Ok(ToolResponse::ParseRegion {
                parsed: service.parse_region(&handle, &region)?,
            }),
            ToolRequest::ParseTable { handle, region } => Ok(ToolResponse::ParseTable {
                table: service.parse_table(&handle, &region)?,
            }),
            ToolRequest::ParseChart { handle, region } => Ok(ToolResponse::ParseChart {
                chart: service.parse_chart(&handle, &region)?,
            }),
            ToolRequest::VerifyStructure {
                handle,
                regions,
                tables,
            } => Ok(ToolResponse::VerifyStructure {
                report: service.verify_structure(&handle, &regions, &tables)?,
            }),
            ToolRequest::CiteCoordinates {
                handle,
                box_,
                quote,
                region_id,
            } => Ok(ToolResponse::CiteCoordinates {
                citation: service.cite_coordinates(&handle, box_, &quote, region_id.as_ref())?,
            }),
        }
    }

    pub fn require_known_tool(name: &str) -> Result<DocumentToolName> {
        DocumentToolName::parse_qualified(name).ok_or_else(|| {
            PerceptionError::InvalidRequest(format!("unknown document tool: {name}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualified_names_match_bible() {
        let names: Vec<String> = DocumentToolName::ALL
            .iter()
            .map(|t| t.qualified())
            .collect();
        assert_eq!(
            names,
            vec![
                "document.inspect",
                "document.retrieve_pages",
                "document.detect_regions",
                "document.parse_region",
                "document.parse_table",
                "document.parse_chart",
                "document.verify_structure",
                "document.cite_coordinates",
            ]
        );
    }
}
