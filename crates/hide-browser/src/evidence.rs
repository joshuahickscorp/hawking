//! The browser evidence model (Bible Book VIII sec 27).
//!
//! A [`BrowserStep`] is the full evidence record for one action the agent (or a
//! recorder) took against a page: what happened, why, and everything observable
//! afterward. A [`BrowserSession`] is an ordered list of steps -- the replayable
//! trace. Heavy payloads (screenshots, raw HTML, network bodies) are referenced
//! by [`ArtifactRef`], never inlined, so a step stays small and a trace stays
//! cheap to store and diff.
//!
//! This is a schema layer: it captures and structures evidence. It runs
//! nothing. A real driver that produces these records from a live browser is
//! out of scope here (see [`crate::driver`]).

use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::a11y::AccessibilityTree;
use crate::dom::{DomSnapshot, SelectStrategy};
use crate::ids::{ArtifactRef, BrowserSessionId};

/// Why a navigation happened. Captured on every step; steps that did not
/// navigate carry [`NavigationCause::None`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NavigationCause {
    /// The user (or agent) navigated directly to a URL.
    UserNavigate,
    /// A link click caused the navigation.
    LinkClick,
    /// A form submission caused the navigation.
    FormSubmit,
    /// A server or meta redirect.
    Redirect,
    /// Script (`location.assign`, history API) caused it.
    ScriptNavigation,
    /// Back in history.
    HistoryBack,
    /// Forward in history.
    HistoryForward,
    /// A reload.
    Reload,
    /// No navigation occurred on this step.
    None,
}

/// How an element was located.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ElementSelector {
    pub strategy: SelectStrategy,
    /// The selector query text (a CSS selector, role name, visible text, or
    /// test id) interpreted per `strategy`.
    pub query: String,
    /// The DOM node the selector resolved to, when the recorder captured it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dom_node: Option<crate::ids::DomNodeId>,
}

impl ElementSelector {
    pub fn css(query: impl Into<String>) -> Self {
        Self {
            strategy: SelectStrategy::Css,
            query: query.into(),
            dom_node: None,
        }
    }

    pub fn test_id(query: impl Into<String>) -> Self {
        Self {
            strategy: SelectStrategy::TestId,
            query: query.into(),
            dom_node: None,
        }
    }

    pub fn with_node(mut self, node: impl Into<String>) -> Self {
        self.dom_node = Some(crate::ids::DomNodeId::new(node));
        self
    }

    /// Whether this selector targets the same element as `other`, matched on
    /// strategy and query. The resolved `dom_node` is not required to match (a
    /// request need not know it up front).
    pub fn same_target(&self, other: &ElementSelector) -> bool {
        self.strategy == other.strategy && self.query == other.query
    }
}

/// The operation a step performed. Internally tagged on `kind` so it reads as
/// `{ "kind": "click" }` on the wire. The target of a navigate is the step's
/// `url`; the target of a click/fill is the step's `selected_element`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BrowserAction {
    Navigate,
    Click,
    Fill { value: String },
    Screenshot,
    ReadDom,
    ReadAccessibility,
    ReadConsole,
    ReadNetwork,
    Wait { ms: u64 },
    Custom { name: String },
}

impl BrowserAction {
    /// A short human label used in replay-mismatch diagnostics.
    pub fn label(&self) -> &'static str {
        match self {
            BrowserAction::Navigate => "navigate",
            BrowserAction::Click => "click",
            BrowserAction::Fill { .. } => "fill",
            BrowserAction::Screenshot => "screenshot",
            BrowserAction::ReadDom => "read_dom",
            BrowserAction::ReadAccessibility => "read_accessibility",
            BrowserAction::ReadConsole => "read_console",
            BrowserAction::ReadNetwork => "read_network",
            BrowserAction::Wait { .. } => "wait",
            BrowserAction::Custom { .. } => "custom",
        }
    }
}

/// Severity of a console message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleLevel {
    Log,
    Info,
    Warn,
    Error,
    Debug,
}

/// One console message captured during a step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ConsoleEvent {
    pub level: ConsoleLevel,
    pub text: String,
    pub timestamp_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// One network exchange captured during a step. Request and response bodies are
/// referenced by artifact id, never inlined.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NetworkEvent {
    pub request_id: String,
    pub method: String,
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// The request headers/body blob, referenced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_ref: Option<ArtifactRef>,
    /// The response body blob, referenced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_ref: Option<ArtifactRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing_ms: Option<u64>,
}

/// A named viewport size the page was observed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

/// The responsive state a step was captured in: which named breakpoint and
/// viewport were active.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResponsiveState {
    /// The breakpoint name (for example `mobile`, `tablet`, `desktop`).
    pub name: String,
    pub viewport: Viewport,
}

/// The observable post-action state a functional oracle grades against. This is
/// the deterministic, structured summary of "what is true now" -- distinct from
/// the raw DOM: it names the signals a check cares about.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResultingState {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
    /// Named application signals a functional check reads (for example
    /// `cart.count -> "1"`). Ordered for determinism.
    #[serde(default)]
    pub signals: BTreeMap<String, String>,
    /// Selectors observed present in the page after the action. A set for
    /// deterministic membership tests.
    #[serde(default)]
    pub present_selectors: BTreeSet<String>,
    /// Visible text fragments observed after the action.
    #[serde(default)]
    pub visible_text: Vec<String>,
    /// How many console errors were seen up to and including this step.
    #[serde(default)]
    pub console_error_count: u32,
    /// The responsive state this was captured in, when relevant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub responsive: Option<ResponsiveState>,
}

impl ResultingState {
    /// A minimal state at a URL.
    pub fn at(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            title: None,
            http_status: None,
            signals: BTreeMap::new(),
            present_selectors: BTreeSet::new(),
            visible_text: Vec::new(),
            console_error_count: 0,
            responsive: None,
        }
    }

    pub fn signal(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.signals.insert(key.into(), value.into());
        self
    }

    pub fn present(mut self, selector: impl Into<String>) -> Self {
        self.present_selectors.insert(selector.into());
        self
    }

    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.visible_text.push(text.into());
        self
    }
}

/// The full evidence record for one browser step (Bible Book VIII sec 27).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BrowserStep {
    /// Zero-based position within the session, for stable ordering.
    pub index: u32,
    /// The page URL after the step.
    pub url: String,
    pub navigation_cause: NavigationCause,
    pub action: BrowserAction,
    /// The element a click/fill targeted, when any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_element: Option<ElementSelector>,
    pub dom_snapshot: DomSnapshot,
    pub accessibility_tree: AccessibilityTree,
    /// The screenshot for this step, referenced (never inlined).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screenshot_ref: Option<ArtifactRef>,
    #[serde(default)]
    pub console_events: Vec<ConsoleEvent>,
    #[serde(default)]
    pub network_events: Vec<NetworkEvent>,
    pub resulting_state: ResultingState,
    /// Wall time this step took, in milliseconds.
    pub timing_ms: u64,
}

/// A recorded, replayable browser session: an ordered list of steps plus a
/// header. This is the fixture a [`crate::driver::ReplayDriver`] plays back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BrowserSession {
    pub id: BrowserSessionId,
    pub steps: Vec<BrowserStep>,
    pub created_ms: u64,
}

impl BrowserSession {
    pub fn new(id: impl Into<String>, steps: Vec<BrowserStep>) -> Self {
        Self {
            id: BrowserSessionId::new(id),
            steps,
            created_ms: 0,
        }
    }

    /// Whether the step indices are 0..len in order.
    pub fn indices_are_sequential(&self) -> bool {
        self.steps
            .iter()
            .enumerate()
            .all(|(i, s)| s.index as usize == i)
    }
}
