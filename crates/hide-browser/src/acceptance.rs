//! The visual acceptance oracle (Bible Book VIII sec 27).
//!
//! A [`VisualAcceptance`] is the spec a change is graded against: a target
//! screenshot/annotation, responsive states, semantic requirements, functional
//! interactions, a11y requirements, tolerances, and before/after artifacts. It
//! is a superset of what any single evaluator reads.
//!
//! This crate ships the DETERMINISTIC halves of the oracle:
//!
//! - [`VisualAcceptance::evaluate_functional`] grades the `functional_interactions`
//!   against a recorded [`ResultingState`] and returns a typed [`Verdict`].
//! - [`VisualAcceptance::evaluate_a11y`] grades the `a11y_requirements` against a
//!   recorded [`AccessibilityTree`].
//!
//! The pixel and layout halves -- comparing a captured screenshot to the target
//! within `tolerances`, and judging the `semantic_requirements` (does it *look*
//! right, does the responsive layout match) -- need a real renderer and/or a
//! vision model. Those are marked DEFERRED_MODEL_REQUIRED at
//! [`VisualAcceptance::evaluate_visual`] and are NOT implemented here.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::a11y::{is_interactive_role, AccessibilityTree};
use crate::evidence::{ResultingState, Viewport};
use crate::ids::ArtifactRef;

/// A typed predicate over a [`ResultingState`]. Internally tagged on `type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StateCheck {
    UrlEquals { url: String },
    UrlContains { fragment: String },
    HttpStatus { expected: u16 },
    SignalEquals { key: String, value: String },
    SignalPresent { key: String },
    ElementPresent { selector: String },
    ElementAbsent { selector: String },
    TextPresent { text: String },
    NoConsoleErrors,
    ConsoleErrorsAtMost { max: u32 },
}

impl StateCheck {
    /// Grade this check against a state. `Ok(())` on pass; the typed failure
    /// kind on fail.
    fn grade(&self, state: &ResultingState) -> std::result::Result<(), FailureKind> {
        match self {
            StateCheck::UrlEquals { url } => {
                if &state.url == url {
                    Ok(())
                } else {
                    Err(FailureKind::UrlMismatch {
                        expected: url.clone(),
                        actual: state.url.clone(),
                    })
                }
            }
            StateCheck::UrlContains { fragment } => {
                if state.url.contains(fragment) {
                    Ok(())
                } else {
                    Err(FailureKind::UrlFragmentMissing {
                        fragment: fragment.clone(),
                        actual: state.url.clone(),
                    })
                }
            }
            StateCheck::HttpStatus { expected } => {
                if state.http_status == Some(*expected) {
                    Ok(())
                } else {
                    Err(FailureKind::StatusMismatch {
                        expected: *expected,
                        actual: state.http_status,
                    })
                }
            }
            StateCheck::SignalEquals { key, value } => match state.signals.get(key) {
                Some(v) if v == value => Ok(()),
                Some(v) => Err(FailureKind::SignalMismatch {
                    key: key.clone(),
                    expected: value.clone(),
                    actual: v.clone(),
                }),
                None => Err(FailureKind::SignalMissing { key: key.clone() }),
            },
            StateCheck::SignalPresent { key } => {
                if state.signals.contains_key(key) {
                    Ok(())
                } else {
                    Err(FailureKind::SignalMissing { key: key.clone() })
                }
            }
            StateCheck::ElementPresent { selector } => {
                if state.present_selectors.contains(selector) {
                    Ok(())
                } else {
                    Err(FailureKind::ElementMissing {
                        selector: selector.clone(),
                    })
                }
            }
            StateCheck::ElementAbsent { selector } => {
                if state.present_selectors.contains(selector) {
                    Err(FailureKind::ElementUnexpected {
                        selector: selector.clone(),
                    })
                } else {
                    Ok(())
                }
            }
            StateCheck::TextPresent { text } => {
                if state.visible_text.iter().any(|t| t.contains(text)) {
                    Ok(())
                } else {
                    Err(FailureKind::TextMissing { text: text.clone() })
                }
            }
            StateCheck::NoConsoleErrors => {
                if state.console_error_count == 0 {
                    Ok(())
                } else {
                    Err(FailureKind::ConsoleErrors {
                        count: state.console_error_count,
                        allowed: 0,
                    })
                }
            }
            StateCheck::ConsoleErrorsAtMost { max } => {
                if state.console_error_count <= *max {
                    Ok(())
                } else {
                    Err(FailureKind::ConsoleErrors {
                        count: state.console_error_count,
                        allowed: *max,
                    })
                }
            }
        }
    }
}

/// A typed predicate over an [`AccessibilityTree`]. Internally tagged on `type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum A11yCheck {
    /// Some node carries this role.
    RolePresent { role: String },
    /// Some node with this role has a non-empty accessible name.
    RoleNamed { role: String },
    /// Every interactive node has a non-empty accessible name.
    NoUnnamedInteractive,
    /// A node with this role has exactly this accessible name.
    NameEquals { role: String, name: String },
}

impl A11yCheck {
    fn grade(&self, tree: &AccessibilityTree) -> std::result::Result<(), FailureKind> {
        match self {
            A11yCheck::RolePresent { role } => {
                if tree.has_role(role) {
                    Ok(())
                } else {
                    Err(FailureKind::A11yRoleMissing { role: role.clone() })
                }
            }
            A11yCheck::RoleNamed { role } => {
                let named = tree
                    .nodes_with_role(role)
                    .iter()
                    .any(|n| n.name.as_deref().map(|s| !s.is_empty()).unwrap_or(false));
                if named {
                    Ok(())
                } else if tree.has_role(role) {
                    Err(FailureKind::A11yUnnamed { role: role.clone() })
                } else {
                    Err(FailureKind::A11yRoleMissing { role: role.clone() })
                }
            }
            A11yCheck::NoUnnamedInteractive => {
                let mut offender = None;
                tree.root.walk(&mut |n| {
                    if offender.is_none()
                        && is_interactive_role(&n.role)
                        && n.name.as_deref().map(|s| s.is_empty()).unwrap_or(true)
                    {
                        offender = Some(n.role.clone());
                    }
                });
                match offender {
                    Some(role) => Err(FailureKind::A11yUnnamed { role }),
                    None => Ok(()),
                }
            }
            A11yCheck::NameEquals { role, name } => {
                let matches = tree
                    .nodes_with_role(role)
                    .iter()
                    .any(|n| n.name.as_deref() == Some(name.as_str()));
                if matches {
                    Ok(())
                } else {
                    let actual = tree
                        .nodes_with_role(role)
                        .first()
                        .and_then(|n| n.name.clone());
                    Err(FailureKind::A11yNameMismatch {
                        role: role.clone(),
                        expected: name.clone(),
                        actual,
                    })
                }
            }
        }
    }
}

/// A functional interaction requirement: an id + prose + the typed check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FunctionalRequirement {
    pub id: String,
    pub description: String,
    pub check: StateCheck,
}

/// An accessibility requirement: an id + prose + the typed check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct A11yRequirement {
    pub id: String,
    pub description: String,
    pub check: A11yCheck,
}

/// A semantic (appearance) requirement. Graded by a renderer/vision comparator,
/// which is DEFERRED_MODEL_REQUIRED; carried here as the acceptance spec so a
/// later evaluator has it. The functional structure it implies should be
/// expressed as [`FunctionalRequirement`]s, which ARE graded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SemanticRequirement {
    pub id: String,
    pub description: String,
}

/// A named responsive target the change must satisfy, with the reference
/// screenshot to compare against (comparison is DEFERRED_MODEL_REQUIRED).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResponsiveTarget {
    pub name: String,
    pub viewport: Viewport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screenshot_ref: Option<ArtifactRef>,
}

/// Comparison tolerances for the DEFERRED visual comparator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Tolerances {
    /// Maximum fraction of differing pixels allowed (0.0 = exact).
    pub pixel_diff_ratio: f64,
    /// Maximum cumulative layout shift allowed.
    pub layout_shift: f64,
    /// Per-channel color delta allowed (0-255).
    pub color_delta: u8,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            pixel_diff_ratio: 0.0,
            layout_shift: 0.0,
            color_delta: 0,
        }
    }
}

/// The before/after artifact pair for a change (both referenced, not inlined).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BeforeAfter {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<ArtifactRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<ArtifactRef>,
}

/// The full acceptance spec a change is graded against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct VisualAcceptance {
    pub id: String,
    /// The target screenshot or annotated design the change must match,
    /// referenced (not inlined). Graded by the DEFERRED visual comparator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<ArtifactRef>,
    #[serde(default)]
    pub responsive_states: Vec<ResponsiveTarget>,
    #[serde(default)]
    pub semantic_requirements: Vec<SemanticRequirement>,
    #[serde(default)]
    pub functional_interactions: Vec<FunctionalRequirement>,
    #[serde(default)]
    pub a11y_requirements: Vec<A11yRequirement>,
    #[serde(default)]
    pub tolerances: Tolerances,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_after: Option<BeforeAfter>,
}

impl VisualAcceptance {
    /// Grade the functional interactions against a recorded resulting state.
    /// Deterministic: no renderer, no model.
    pub fn evaluate_functional(&self, state: &ResultingState) -> Verdict {
        let mut reasons = Vec::new();
        for req in &self.functional_interactions {
            if let Err(kind) = req.check.grade(state) {
                reasons.push(FailureReason {
                    requirement_id: req.id.clone(),
                    detail: req.description.clone(),
                    kind,
                });
            }
        }
        Verdict::from_reasons(reasons)
    }

    /// Grade the a11y requirements against a recorded accessibility tree.
    /// Deterministic: no renderer, no model.
    pub fn evaluate_a11y(&self, tree: &AccessibilityTree) -> Verdict {
        let mut reasons = Vec::new();
        for req in &self.a11y_requirements {
            if let Err(kind) = req.check.grade(tree) {
                reasons.push(FailureReason {
                    requirement_id: req.id.clone(),
                    detail: req.description.clone(),
                    kind,
                });
            }
        }
        Verdict::from_reasons(reasons)
    }

    /// DEFERRED_MODEL_REQUIRED: compare a captured screenshot to `target` (and
    /// each `responsive_states` reference) within `tolerances`, and judge the
    /// `semantic_requirements`. This needs a real renderer and/or a vision
    /// model; it is intentionally NOT implemented here and must not be claimed.
    /// Callers that need pixel/appearance grading route to a model-bearing
    /// service outside this crate.
    pub fn evaluate_visual(&self) -> ! {
        unimplemented!(
            "DEFERRED_MODEL_REQUIRED: pixel/appearance grading needs a renderer or vision model"
        )
    }
}

/// A typed reason a requirement failed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FailureKind {
    UrlMismatch { expected: String, actual: String },
    UrlFragmentMissing { fragment: String, actual: String },
    StatusMismatch { expected: u16, actual: Option<u16> },
    SignalMismatch { key: String, expected: String, actual: String },
    SignalMissing { key: String },
    ElementMissing { selector: String },
    ElementUnexpected { selector: String },
    TextMissing { text: String },
    ConsoleErrors { count: u32, allowed: u32 },
    A11yRoleMissing { role: String },
    A11yUnnamed { role: String },
    A11yNameMismatch { role: String, expected: String, actual: Option<String> },
}

/// A single failed requirement: which one, why (typed), and its prose.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FailureReason {
    pub requirement_id: String,
    pub kind: FailureKind,
    pub detail: String,
}

/// The outcome of a deterministic acceptance evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Fail { reasons: Vec<FailureReason> },
}

impl Verdict {
    fn from_reasons(reasons: Vec<FailureReason>) -> Self {
        if reasons.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail { reasons }
        }
    }

    pub fn is_pass(&self) -> bool {
        matches!(self, Verdict::Pass)
    }

    pub fn reasons(&self) -> &[FailureReason] {
        match self {
            Verdict::Pass => &[],
            Verdict::Fail { reasons } => reasons,
        }
    }
}
