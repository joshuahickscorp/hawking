//! Design Mode annotations (Bible Book VIII sec 27).
//!
//! Design Mode is the "click an element, see everything about it" surface: a
//! selection on the page maps to its DOM node, the source symbol that emitted
//! it, the CSS rule that styled it, its layout box, and its accessibility node.
//! [`annotate`] builds that mapping deterministically from a captured
//! [`DomSnapshot`]: everything it needs was recorded, so no live browser is
//! involved.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dom::{BoxModel, CssRule, DomSnapshot, SourceSymbol};
use crate::error::{BrowserError, Result};
use crate::evidence::ElementSelector;
use crate::ids::{AccessibilityNodeId, DomNodeId};

/// The full mapping from a selected element to what Design Mode shows about it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DesignAnnotation {
    /// The selection that produced this annotation.
    pub selected: ElementSelector,
    /// The DOM node the selection resolved to.
    pub dom_node: DomNodeId,
    /// The source symbol that emitted the node, when a source map was captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_symbol: Option<SourceSymbol>,
    /// The most-specific CSS rule that applied, when captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub css_rule: Option<CssRule>,
    /// The element's layout box.
    pub box_model: BoxModel,
    /// The accessibility node the element maps to, when captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a11y_node: Option<AccessibilityNodeId>,
}

/// Build a [`DesignAnnotation`] for a selection against a captured DOM snapshot.
///
/// Resolution order: if the selector already carries a resolved `dom_node`, that
/// node is used; otherwise the selector is resolved structurally against the
/// snapshot (`#id`, `.class`, test id, tag, or text). Fails with
/// [`BrowserError::UnresolvedSelection`] if no node matches.
pub fn annotate(selector: &ElementSelector, dom: &DomSnapshot) -> Result<DesignAnnotation> {
    let node = match &selector.dom_node {
        Some(id) => dom
            .find(id)
            .ok_or_else(|| BrowserError::UnresolvedSelection {
                selector: id.to_string(),
            })?,
        None => dom.resolve(selector.strategy, &selector.query).ok_or_else(|| {
            BrowserError::UnresolvedSelection {
                selector: selector.query.clone(),
            }
        })?,
    };

    Ok(DesignAnnotation {
        selected: selector.clone(),
        dom_node: node.id.clone(),
        source_symbol: node.source_symbol.clone(),
        css_rule: node.css_rules.first().cloned(),
        box_model: node.box_model.clone().unwrap_or_default(),
        a11y_node: node.a11y_node.clone(),
    })
}
