//! Coordinate-bound citation geometry.
//!
//! Citations must pin *where* evidence came from on a page so a later verifier
//! can re-open the same region without re-OCR of the whole document.

use serde::{Deserialize, Serialize};

/// One page-relative box in normalized page coordinates (0.0 ..= 1.0).
///
/// Origin is top-left. Callers may convert to PDF user space or image pixels
/// outside this crate; the normalized form is the portable citation pin.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CoordBox {
    pub page: u32,
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl CoordBox {
    pub fn new(page: u32, x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self {
            page,
            x0,
            y0,
            x1,
            y1,
        }
    }

    pub fn full_page(page: u32) -> Self {
        Self::new(page, 0.0, 0.0, 1.0, 1.0)
    }

    pub fn is_normalized(&self) -> bool {
        self.x0 >= 0.0
            && self.y0 >= 0.0
            && self.x1 <= 1.0
            && self.y1 <= 1.0
            && self.x0 <= self.x1
            && self.y0 <= self.y1
    }

    pub fn area(&self) -> f32 {
        (self.x1 - self.x0).max(0.0) * (self.y1 - self.y0).max(0.0)
    }
}

/// Discrete page coordinate used when only page identity is known (cheap pass).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageCoord {
    pub page: u32,
}

/// What kind of visual/text region a detector reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionKind {
    Text,
    Table,
    Chart,
    Diagram,
    Equation,
    Image,
    Ui,
    Header,
    Footer,
    Other,
}

impl RegionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Table => "table",
            Self::Chart => "chart",
            Self::Diagram => "diagram",
            Self::Equation => "equation",
            Self::Image => "image",
            Self::Ui => "ui",
            Self::Header => "header",
            Self::Footer => "footer",
            Self::Other => "other",
        }
    }

    /// Tables/charts/diagrams escalate to selective parse; plain text stays cheap.
    pub fn prefers_structured_parse(self) -> bool {
        matches!(self, Self::Table | Self::Chart | Self::Diagram | Self::Equation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_page_is_normalized_unit_area() {
        let b = CoordBox::full_page(0);
        assert!(b.is_normalized());
        assert!((b.area() - 1.0).abs() < 1e-6);
    }
}
