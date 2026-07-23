//! Captured DOM snapshot: a small, queryable tree of the page at one step.
//!
//! The snapshot is intentionally a structured tree (not raw HTML text) so it is
//! deterministically queryable: a selection resolves to a [`DomNode`], and a
//! Design Mode annotation reads that node's box, source symbol, and CSS. The
//! full raw HTML, when kept, is referenced by [`DomSnapshot::html_ref`] as an
//! artifact rather than inlined.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ids::{AccessibilityNodeId, ArtifactRef, DomNodeId};

/// The layout box of an element, in CSS pixels, relative to the viewport. The
/// content box plus the surrounding edges the browser exposes in Design Mode.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BoxModel {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    #[serde(default)]
    pub padding: EdgeSizes,
    #[serde(default)]
    pub border: EdgeSizes,
    #[serde(default)]
    pub margin: EdgeSizes,
}

/// Top/right/bottom/left edge sizes for padding, border, or margin.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EdgeSizes {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

/// A CSS rule that applies to a node, carried for Design Mode. `source` names
/// the stylesheet origin (a file path or `<style>` marker) when known.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CssRule {
    pub selector: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Declared properties, ordered for determinism.
    #[serde(default)]
    pub declarations: BTreeMap<String, String>,
}

/// A dev-time source mapping from a rendered node back to the code that emitted
/// it (for example a JSX component and line). Populated only by tooling that has
/// a source map; absent in production captures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SourceSymbol {
    /// The source file that declared the element (repository-relative).
    pub file: String,
    /// The symbol name (component, function, or selector) that emitted it.
    pub symbol: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column: Option<u32>,
}

/// One node in the captured DOM tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DomNode {
    pub id: DomNodeId,
    /// Lower-case tag name, for example `button`.
    pub tag: String,
    /// Attributes present on the element, ordered for determinism. `id`,
    /// `class`, and `data-testid` participate in selector resolution.
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
    /// The element's own text (not its descendants').
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// The layout box, when captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub box_model: Option<BoxModel>,
    /// The accessibility node this element maps to, when captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a11y_node: Option<AccessibilityNodeId>,
    /// The source symbol that emitted this node, for Design Mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_symbol: Option<SourceSymbol>,
    /// CSS rules that apply, most-specific first, for Design Mode.
    #[serde(default)]
    pub css_rules: Vec<CssRule>,
    #[serde(default)]
    pub children: Vec<DomNode>,
}

impl DomNode {
    /// A leaf node with just an id and tag; other fields default. A convenience
    /// for building fixtures.
    pub fn leaf(id: impl Into<String>, tag: impl Into<String>) -> Self {
        Self {
            id: DomNodeId::new(id),
            tag: tag.into(),
            attributes: BTreeMap::new(),
            text: None,
            box_model: None,
            a11y_node: None,
            source_symbol: None,
            css_rules: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Depth-first search for a descendant (or self) with the given id.
    pub fn find(&self, id: &DomNodeId) -> Option<&DomNode> {
        if &self.id == id {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find(id) {
                return Some(found);
            }
        }
        None
    }

    fn matches_selector(&self, strategy: SelectorMatch<'_>) -> bool {
        match strategy {
            SelectorMatch::Id(v) => self.attributes.get("id").map(|s| s.as_str()) == Some(v),
            SelectorMatch::TestId(v) => {
                self.attributes.get("data-testid").map(|s| s.as_str()) == Some(v)
            }
            SelectorMatch::Tag(v) => self.tag == v,
            SelectorMatch::Class(v) => self
                .attributes
                .get("class")
                .map(|s| s.split_whitespace().any(|c| c == v))
                .unwrap_or(false),
            SelectorMatch::Text(v) => self.text.as_deref().map(|t| t.contains(v)).unwrap_or(false),
        }
    }

    fn find_matching(&self, strategy: SelectorMatch<'_>) -> Option<&DomNode> {
        if self.matches_selector(strategy) {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find_matching(strategy) {
                return Some(found);
            }
        }
        None
    }
}

/// A parsed selector kind used for structural resolution against the snapshot.
#[derive(Debug, Clone, Copy)]
enum SelectorMatch<'a> {
    Id(&'a str),
    Class(&'a str),
    TestId(&'a str),
    Tag(&'a str),
    Text(&'a str),
}

/// The captured DOM at one step: a structured tree plus an optional reference to
/// the raw HTML blob.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DomSnapshot {
    pub root: DomNode,
    /// The full serialized HTML, referenced (never inlined).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub html_ref: Option<ArtifactRef>,
    /// Total node count, for a cheap size signal without walking the tree.
    #[serde(default)]
    pub node_count: u32,
}

impl DomSnapshot {
    pub fn new(root: DomNode) -> Self {
        let node_count = count_nodes(&root);
        Self {
            root,
            html_ref: None,
            node_count,
        }
    }

    /// Find a node by id anywhere in the tree.
    pub fn find(&self, id: &DomNodeId) -> Option<&DomNode> {
        self.root.find(id)
    }

    /// Resolve a CSS-ish or role/text selector to a node. Supports `#id`,
    /// `.class`, `[data-testid=...]` (also the bare test-id string), a bare tag,
    /// and a `text=` prefix. This is a deliberately small resolver: real
    /// full-CSS matching against a live renderer is DEFERRED_MODEL_REQUIRED.
    pub fn resolve(&self, strategy: SelectStrategy, query: &str) -> Option<&DomNode> {
        let m = match strategy {
            SelectStrategy::TestId => SelectorMatch::TestId(query),
            SelectStrategy::Text => SelectorMatch::Text(query),
            SelectStrategy::Role => SelectorMatch::Tag(query),
            SelectStrategy::XPath => return None, // out of scope for the small resolver
            SelectStrategy::Css => {
                if let Some(rest) = query.strip_prefix('#') {
                    SelectorMatch::Id(rest)
                } else if let Some(rest) = query.strip_prefix('.') {
                    SelectorMatch::Class(rest)
                } else if let Some(rest) = query.strip_prefix("[data-testid=") {
                    let v = rest.trim_end_matches(']').trim_matches(|c| c == '"' || c == '\'');
                    SelectorMatch::TestId(v)
                } else {
                    SelectorMatch::Tag(query)
                }
            }
        };
        self.root.find_matching(m)
    }
}

fn count_nodes(node: &DomNode) -> u32 {
    1 + node.children.iter().map(count_nodes).sum::<u32>()
}

/// How a selector string is interpreted. Kept here (rather than in `evidence`)
/// because both the evidence records and the DOM resolver need it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SelectStrategy {
    Css,
    XPath,
    Role,
    Text,
    TestId,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> DomSnapshot {
        let mut button = DomNode::leaf("n-btn", "button");
        button
            .attributes
            .insert("id".into(), "add-to-cart".into());
        button
            .attributes
            .insert("data-testid".into(), "add-cart".into());
        button.text = Some("Add to cart".into());
        let mut root = DomNode::leaf("n-root", "div");
        root.children.push(button);
        DomSnapshot::new(root)
    }

    #[test]
    fn find_by_id_walks_the_tree() {
        let snap = tree();
        let n = snap.find(&DomNodeId::from("n-btn")).unwrap();
        assert_eq!(n.tag, "button");
        assert!(snap.find(&DomNodeId::from("n-missing")).is_none());
    }

    #[test]
    fn resolve_supports_id_testid_and_text() {
        let snap = tree();
        assert_eq!(
            snap.resolve(SelectStrategy::Css, "#add-to-cart").unwrap().id,
            DomNodeId::from("n-btn")
        );
        assert_eq!(
            snap.resolve(SelectStrategy::TestId, "add-cart").unwrap().id,
            DomNodeId::from("n-btn")
        );
        assert_eq!(
            snap.resolve(SelectStrategy::Text, "Add to").unwrap().id,
            DomNodeId::from("n-btn")
        );
        assert!(snap.resolve(SelectStrategy::Css, "#nope").is_none());
    }

    #[test]
    fn node_count_is_captured() {
        assert_eq!(tree().node_count, 2);
    }
}
