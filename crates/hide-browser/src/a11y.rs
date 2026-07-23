//! Captured accessibility tree: the semantic view the platform exposes.
//!
//! This is the tree an assistive technology would see. It is captured alongside
//! the DOM so a11y requirements can be graded deterministically (does a control
//! have an accessible name? is the expected role present?) without a renderer.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ids::AccessibilityNodeId;

/// One node in the accessibility tree. `role` is an ARIA role string
/// (`button`, `link`, `textbox`, ...); `name` is the computed accessible name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AccessibilityNode {
    pub id: AccessibilityNodeId,
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// ARIA states, ordered for determinism (for example `disabled`,
    /// `checked`, `expanded`).
    #[serde(default)]
    pub states: Vec<String>,
    #[serde(default)]
    pub children: Vec<AccessibilityNode>,
}

impl AccessibilityNode {
    pub fn new(id: impl Into<String>, role: impl Into<String>) -> Self {
        Self {
            id: AccessibilityNodeId::new(id),
            role: role.into(),
            name: None,
            value: None,
            states: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_child(mut self, child: AccessibilityNode) -> Self {
        self.children.push(child);
        self
    }

    /// Visit this node and every descendant, in depth-first order.
    pub fn walk<'a>(&'a self, visit: &mut dyn FnMut(&'a AccessibilityNode)) {
        visit(self);
        for child in &self.children {
            child.walk(visit);
        }
    }
}

/// The captured accessibility tree at one step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AccessibilityTree {
    pub root: AccessibilityNode,
}

impl AccessibilityTree {
    pub fn new(root: AccessibilityNode) -> Self {
        Self { root }
    }

    /// Every node with the given role.
    pub fn nodes_with_role<'a>(&'a self, role: &str) -> Vec<&'a AccessibilityNode> {
        let mut out = Vec::new();
        self.root.walk(&mut |n| {
            if n.role == role {
                out.push(n);
            }
        });
        out
    }

    /// True if any node carries the given role.
    pub fn has_role(&self, role: &str) -> bool {
        !self.nodes_with_role(role).is_empty()
    }

    /// Find a node by id.
    pub fn find(&self, id: &AccessibilityNodeId) -> Option<&AccessibilityNode> {
        let mut found = None;
        self.root.walk(&mut |n| {
            if &n.id == id {
                found = Some(n);
            }
        });
        found
    }
}

/// The set of roles considered interactive for the "no unnamed interactive
/// control" a11y check. Spec-derived from the ARIA roles model (an open W3C
/// spec); no proprietary source is copied.
pub const INTERACTIVE_ROLES: &[&str] = &[
    "button",
    "link",
    "textbox",
    "checkbox",
    "radio",
    "combobox",
    "listbox",
    "menuitem",
    "switch",
    "slider",
    "tab",
    "searchbox",
];

pub fn is_interactive_role(role: &str) -> bool {
    INTERACTIVE_ROLES.contains(&role)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walk_and_role_queries() {
        let tree = AccessibilityTree::new(
            AccessibilityNode::new("ax-root", "document")
                .with_child(AccessibilityNode::new("ax-btn", "button").with_name("Add to cart"))
                .with_child(AccessibilityNode::new("ax-link", "link").with_name("Home")),
        );
        assert!(tree.has_role("button"));
        assert!(!tree.has_role("slider"));
        assert_eq!(tree.nodes_with_role("button").len(), 1);
        assert_eq!(
            tree.find(&AccessibilityNodeId::from("ax-btn"))
                .and_then(|n| n.name.clone()),
            Some("Add to cart".into())
        );
    }

    #[test]
    fn interactive_role_classification() {
        assert!(is_interactive_role("button"));
        assert!(!is_interactive_role("document"));
    }
}
