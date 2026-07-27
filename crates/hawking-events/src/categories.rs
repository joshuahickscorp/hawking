//! Canonical event categories and their open-kind strings.
//!
//! Kinds are dotted strings on [`hide_core::event::Event::kind`]. Categories
//! group them for coverage tests and documentation. Adding a category without
//! a kind here fails the round-trip coverage test.

use serde::{Deserialize, Serialize};

/// Required product-event categories from the bridge-events contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    ModelLifecycle,
    Text,
    Reasoning,
    Plans,
    Tools,
    Permissions,
    Edits,
    Tests,
    Verification,
    Agents,
    Acceleration,
    FabricPlacement,
    FabricNodeState,
    Warnings,
    Errors,
    Usage,
}

impl Category {
    pub fn as_str(self) -> &'static str {
        match self {
            Category::ModelLifecycle => "model_lifecycle",
            Category::Text => "text",
            Category::Reasoning => "reasoning",
            Category::Plans => "plans",
            Category::Tools => "tools",
            Category::Permissions => "permissions",
            Category::Edits => "edits",
            Category::Tests => "tests",
            Category::Verification => "verification",
            Category::Agents => "agents",
            Category::Acceleration => "acceleration",
            Category::FabricPlacement => "fabric_placement",
            Category::FabricNodeState => "fabric_node_state",
            Category::Warnings => "warnings",
            Category::Errors => "errors",
            Category::Usage => "usage",
        }
    }

    pub fn all() -> &'static [Category] {
        &ALL_CATEGORIES
    }
}

const ALL_CATEGORIES: [Category; 16] = [
    Category::ModelLifecycle,
    Category::Text,
    Category::Reasoning,
    Category::Plans,
    Category::Tools,
    Category::Permissions,
    Category::Edits,
    Category::Tests,
    Category::Verification,
    Category::Agents,
    Category::Acceleration,
    Category::FabricPlacement,
    Category::FabricNodeState,
    Category::Warnings,
    Category::Errors,
    Category::Usage,
];

/// Primary open-kind string for each category (one representative kind).
/// Additional legacy kinds may map into the same category via [`category_for_kind`].
pub const CATEGORY_KINDS: &[(Category, &str)] = &[
    (Category::ModelLifecycle, "model.lifecycle"),
    (Category::Text, "model.token"),
    (Category::Reasoning, "model.reasoning"),
    (Category::Plans, "plan.created"),
    (Category::Tools, "tool.call"),
    (Category::Permissions, "security.gate"),
    (Category::Edits, "edit.diff"),
    (Category::Tests, "test.result"),
    (Category::Verification, "verify.receipt"),
    (Category::Agents, "agent.phase"),
    (Category::Acceleration, "accel.status"),
    (Category::FabricPlacement, "fabric.placement"),
    (Category::FabricNodeState, "fabric.node_state"),
    (Category::Warnings, "system.warning"),
    (Category::Errors, "error"),
    (Category::Usage, "model.usage"),
];

pub fn all_categories() -> &'static [Category] {
    Category::all()
}

pub fn kind_for_category(cat: Category) -> &'static str {
    CATEGORY_KINDS
        .iter()
        .find(|(c, _)| *c == cat)
        .map(|(_, k)| *k)
        .expect("every Category has a kind in CATEGORY_KINDS")
}

/// Map a kind string (canonical or legacy) onto a category when known.
pub fn category_for_kind(kind: &str) -> Option<Category> {
    // Primary kinds first.
    for (cat, k) in CATEGORY_KINDS {
        if *k == kind {
            return Some(*cat);
        }
    }
    // Legacy / alternate kinds that adapters emit.
    Some(match kind {
        "tool.result" => Category::Tools,
        "user.intent" => Category::ModelLifecycle,
        "runtime.status" | "model.loaded" | "model.unloaded" => Category::ModelLifecycle,
        "token.batch" | "agent.message" => Category::Text,
        "plan.mutation" => Category::Plans,
        "security.decision" => Category::Permissions,
        "edit.patch" | "diff.applied" | "diff.rejected" => Category::Edits,
        "test.started" => Category::Tests,
        "verify.request" => Category::Verification,
        "agent.spawn" | "agent.result" => Category::Agents,
        "accel.draft" | "accel.accept" | "accel.reject" => Category::Acceleration,
        "fabric.job" => Category::FabricPlacement,
        "fabric.node" => Category::FabricNodeState,
        "system.warning" => Category::Warnings,
        "error" | "system.error" => Category::Errors,
        "model.usage" | "usage.stats" => Category::Usage,
        "seed.transition" => Category::ModelLifecycle,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_category_has_unique_kind() {
        let mut seen = std::collections::BTreeSet::new();
        for (cat, kind) in CATEGORY_KINDS {
            assert!(seen.insert(*kind), "duplicate kind {kind}");
            assert_eq!(kind_for_category(*cat), *kind);
            assert_eq!(category_for_kind(kind), Some(*cat));
        }
        assert_eq!(seen.len(), ALL_CATEGORIES.len());
    }
}
