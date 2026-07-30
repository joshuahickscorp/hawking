//! HIDE YOU automations: durable, permission-bounded background jobs.
//!
//! An [`Automation`] is a declared standing goal (reminder, recurring brief,
//! connector summary, calendar prep, email triage, project status check, watch
//! condition, research monitor, file ingestion pipeline, or agent job). Every
//! automation carries a closed [`PermissionSet`]. The job it spawns receives a
//! [`JobCapability`] *derived from* that set and structurally cannot widen it.
//!
//! # The property that matters most
//!
//! **A background agent cannot inherit broader authority than the automation
//! grants.** Tool use is gated by the job capability; an attempt to call a tool
//! the automation did not grant fails closed and is recorded on the result.
//!
//! # What this module is (and is not)
//!
//! * **Is:** declaration model, capability derivation, durable store, injected
//!   clock, stop-condition enforcement, schedule-slot idempotency, fixture tool
//!   registry, inspectable result history.
//! * **Is not:** a wall-clock daemon, launchd/cron installer, real connector or
//!   model execution, fleets, Fabric, or Metal. Real tool bodies are fixture
//!   stubs; wall-clock wiring is a later step.
//!
//! Model-free throughout. Deterministic under an injected [`Clock`].

use crate::error::{HideError, Result};
use crate::ids::now_ms;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;


// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// Stable automation id (`atm_…`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AutomationId(pub String);

impl AutomationId {
    pub fn new() -> Self {
        Self(format!("atm_{}", mint_ulid_body()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for AutomationId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&str> for AutomationId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for AutomationId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl std::fmt::Display for AutomationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable job id (`ajb_…`) for a single automation run.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AutomationJobId(pub String);

impl AutomationJobId {
    pub fn new() -> Self {
        Self(format!("ajb_{}", mint_ulid_body()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for AutomationJobId {
    fn default() -> Self {
        Self::new()
    }
}

fn mint_ulid_body() -> String {
    // Mirror hide_core::ids: honor deterministic seed when present.
    // We cannot call the private next_ulid_body, so mint via a known id type
    // and strip its prefix.
    let sid = crate::ids::SessionId::new();
    sid.as_str()
        .strip_prefix("ses_")
        .unwrap_or(sid.as_str())
        .to_string()
}

// ---------------------------------------------------------------------------
// Clock (injected; wall-clock wiring is out of scope)
// ---------------------------------------------------------------------------

/// Source of "now" for schedule evaluation. Tests inject a deterministic clock;
/// production may later wrap wall time. No daemon lives here.
pub trait Clock: Send + Sync {
    fn now_ms(&self) -> u64;
}

/// Mutable in-process clock for tests and controlled ticks.
#[derive(Debug, Default)]
pub struct InjectedClock {
    now: Mutex<u64>,
}

impl InjectedClock {
    pub fn new(start_ms: u64) -> Self {
        Self {
            now: Mutex::new(start_ms),
        }
    }

    pub fn set(&self, ms: u64) {
        *self.now.lock() = ms;
    }

    pub fn advance(&self, delta_ms: u64) {
        *self.now.lock() += delta_ms;
    }
}

impl Clock for InjectedClock {
    fn now_ms(&self) -> u64 {
        *self.now.lock()
    }
}

/// Wall-clock adapter (available but not used by the in-process scheduler).
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        now_ms()
    }
}

// ---------------------------------------------------------------------------
// Kind, trigger, budget, stop, notifications
// ---------------------------------------------------------------------------

/// What the automation is for (product surface labels; pure metadata).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutomationKind {
    Reminder,
    RecurringBrief,
    ConnectorSummary,
    CalendarPreparation,
    EmailTriage,
    ProjectStatusCheck,
    WatchCondition,
    ResearchMonitor,
    FileIngestionPipeline,
    AgentJob,
}

/// How / when the automation fires. Parsing real cron is out of scope; slots are
/// opaque keys the caller (or an interval derivation) supplies so idempotency is
/// testable without a daemon.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum TriggerSpec {
    /// Fire once at `at_ms` (inclusive).
    Once { at_ms: u64 },
    /// Fire every `every_ms` from `anchor_ms`. Slot key is the interval index.
    Interval { every_ms: u64, anchor_ms: u64 },
    /// An opaque pre-keyed schedule slot (e.g. a cron tick identity).
    CronSlot { slot_key: String, at_ms: u64 },
    /// Only fires on an explicit manual wake.
    Manual,
    /// Fire when a named watch condition is presented as met.
    Watch { condition: String },
}

impl TriggerSpec {
    /// Deterministic schedule slot identity for `now_ms`, if due.
    /// Returns `None` when the trigger is not due (or is Manual/Watch without
    /// an external event).
    pub fn slot_if_due(&self, now_ms: u64) -> Option<String> {
        match self {
            TriggerSpec::Once { at_ms } => {
                if now_ms >= *at_ms {
                    Some(format!("once:{at_ms}"))
                } else {
                    None
                }
            }
            TriggerSpec::Interval {
                every_ms,
                anchor_ms,
            } => {
                if *every_ms == 0 || now_ms < *anchor_ms {
                    return None;
                }
                let index = (now_ms - anchor_ms) / every_ms;
                // Due on the boundary of the current slot (and any past unfired).
                let slot_start = anchor_ms + index * every_ms;
                if now_ms >= slot_start {
                    Some(format!("interval:{every_ms}:{index}"))
                } else {
                    None
                }
            }
            TriggerSpec::CronSlot { slot_key, at_ms } => {
                if now_ms >= *at_ms {
                    Some(format!("cron:{slot_key}"))
                } else {
                    None
                }
            }
            TriggerSpec::Manual | TriggerSpec::Watch { .. } => None,
        }
    }

    /// Next run time strictly after `from_ms`, if the trigger has one.
    pub fn next_run_after(&self, from_ms: u64) -> Option<u64> {
        match self {
            TriggerSpec::Once { at_ms } => {
                if *at_ms > from_ms {
                    Some(*at_ms)
                } else {
                    None
                }
            }
            TriggerSpec::Interval {
                every_ms,
                anchor_ms,
            } => {
                if *every_ms == 0 {
                    return None;
                }
                if from_ms < *anchor_ms {
                    return Some(*anchor_ms);
                }
                let index = (from_ms - anchor_ms) / every_ms;
                let next = anchor_ms + (index + 1) * every_ms;
                Some(next)
            }
            TriggerSpec::CronSlot { at_ms, .. } => {
                if *at_ms > from_ms {
                    Some(*at_ms)
                } else {
                    None
                }
            }
            TriggerSpec::Manual | TriggerSpec::Watch { .. } => None,
        }
    }
}

/// Resource bounds. Exhaustion is a hard stop (see [`StopReason::BudgetExhausted`]).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub max_runs: Option<u32>,
    pub max_tool_calls: Option<u32>,
    pub max_wall_ms: Option<u64>,
    pub max_tokens: Option<u64>,
}

/// Cumulative spend against a budget.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BudgetUsage {
    pub runs: u32,
    pub tool_calls: u32,
    pub wall_ms: u64,
    pub tokens: u64,
}

impl BudgetUsage {
    /// Which budget axis, if any, is exhausted.
    pub fn exhausted_axis(&self, budget: &ResourceBudget) -> Option<&'static str> {
        if budget.max_runs.is_some_and(|m| self.runs >= m) {
            return Some("max_runs");
        }
        if budget.max_tool_calls.is_some_and(|m| self.tool_calls >= m) {
            return Some("max_tool_calls");
        }
        if budget.max_wall_ms.is_some_and(|m| self.wall_ms >= m) {
            return Some("max_wall_ms");
        }
        if budget.max_tokens.is_some_and(|m| self.tokens >= m) {
            return Some("max_tokens");
        }
        None
    }

    pub fn is_exhausted(&self, budget: &ResourceBudget) -> bool {
        self.exhausted_axis(budget).is_some()
    }
}

/// When the automation must halt. Enforced, not advisory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum StopCondition {
    /// Keep running until budget or cancel.
    Never,
    /// Halt after this many successful or failed runs (total completed).
    AfterRuns { count: u32 },
    /// Halt when a named condition is presented as met.
    ConditionMet { name: String },
    /// Halt after this many recorded failures.
    MaxFailures { count: u32 },
}

/// How the owner is notified about run outcomes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotificationPolicy {
    Silent,
    OnFailure,
    OnSuccess,
    Always,
}

impl Default for NotificationPolicy {
    fn default() -> Self {
        Self::OnFailure
    }
}

/// Lifecycle of an automation declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutomationStatus {
    #[default]
    Active,
    Paused,
    /// Stop condition or budget halted further runs.
    Stopped,
    Cancelled,
}

impl AutomationStatus {
    pub fn may_run(self) -> bool {
        matches!(self, Self::Active)
    }
}

/// Why a job or automation halted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum StopReason {
    BudgetExhausted { axis: String },
    ConditionMet { name: String },
    AfterRuns { count: u32 },
    MaxFailures { count: u32 },
    Cancelled,
    AuthorityDenied { tool: String },
}

// ---------------------------------------------------------------------------
// Permission set → job capability (structural non-widening)
// ---------------------------------------------------------------------------

/// Closed set of tools and connectors an automation is allowed to use.
///
/// This is the sole source of authority for background jobs. A
/// [`JobCapability`] can only be obtained by [`PermissionSet::derive_capability`]
/// (full set) or [`PermissionSet::derive_capability_subset`] (strict subset).
/// There is no public constructor that invents tools outside a permission set,
/// and [`JobCapability`] exposes no method that adds tools or connectors.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PermissionSet {
    tools: BTreeSet<String>,
    connectors: BTreeSet<String>,
}

impl PermissionSet {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn new(
        tools: impl IntoIterator<Item = impl Into<String>>,
        connectors: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            tools: tools.into_iter().map(Into::into).collect(),
            connectors: connectors.into_iter().map(Into::into).collect(),
        }
    }

    pub fn tools(&self) -> &BTreeSet<String> {
        &self.tools
    }

    pub fn connectors(&self) -> &BTreeSet<String> {
        &self.connectors
    }

    pub fn grants_tool(&self, name: &str) -> bool {
        self.tools.contains(name)
    }

    pub fn grants_connector(&self, name: &str) -> bool {
        self.connectors.contains(name)
    }

    /// Derive the full capability the automation grants. The job receives this
    /// and cannot widen it.
    pub fn derive_capability(&self) -> JobCapability {
        JobCapability {
            tools: self.tools.clone(),
            connectors: self.connectors.clone(),
            live: true,
        }
    }

    /// Derive a capability that is a subset of this set. Requesting a tool or
    /// connector not in the set is an error (fail closed at derivation time).
    pub fn derive_capability_subset(
        &self,
        tools: impl IntoIterator<Item = impl AsRef<str>>,
        connectors: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<JobCapability> {
        let mut out_tools = BTreeSet::new();
        for t in tools {
            let name = t.as_ref();
            if !self.tools.contains(name) {
                return Err(HideError::CapabilityMissing(format!(
                    "cannot derive job capability for tool '{name}': not in automation permission set"
                )));
            }
            out_tools.insert(name.to_string());
        }
        let mut out_connectors = BTreeSet::new();
        for c in connectors {
            let name = c.as_ref();
            if !self.connectors.contains(name) {
                return Err(HideError::CapabilityMissing(format!(
                    "cannot derive job capability for connector '{name}': not in automation permission set"
                )));
            }
            out_connectors.insert(name.to_string());
        }
        Ok(JobCapability {
            tools: out_tools,
            connectors: out_connectors,
            live: true,
        })
    }
}

/// Capability handed to a spawned job. **Structurally non-widening**: fields are
/// private; the only construction paths are [`PermissionSet::derive_capability`]
/// and [`PermissionSet::derive_capability_subset`]. No `grant_tool` / `add` API.
///
/// `live` is never serialized. A capability-shaped JSON object deserialized into
/// this type has `live = false` and fails every gate — closing export / handoff
/// smuggling paths.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct JobCapability {
    tools: BTreeSet<String>,
    connectors: BTreeSet<String>,
    #[serde(skip)]
    live: bool,
}

impl JobCapability {
    pub fn tools(&self) -> &BTreeSet<String> {
        &self.tools
    }

    pub fn connectors(&self) -> &BTreeSet<String> {
        &self.connectors
    }

    pub fn is_live(&self) -> bool {
        self.live
    }

    pub fn allows_tool(&self, name: &str) -> bool {
        self.live && self.tools.contains(name)
    }

    pub fn allows_connector(&self, name: &str) -> bool {
        self.live && self.connectors.contains(name)
    }

    /// Fail-closed tool gate. Returns `Ok(())` only when the tool is granted.
    pub fn require_tool(&self, name: &str) -> Result<()> {
        if !self.live {
            return Err(HideError::PolicyDenied(
                "job capability is not live (forged or deserialized; derive only)".into(),
            ));
        }
        if self.tools.contains(name) {
            Ok(())
        } else {
            Err(HideError::PolicyDenied(format!(
                "job capability does not grant tool '{name}'"
            )))
        }
    }

    /// True iff every tool/connector in `self` is also in `parent` (subset or equal).
    pub fn is_within(&self, parent: &PermissionSet) -> bool {
        self.live
            && self.tools.is_subset(parent.tools())
            && self.connectors.is_subset(parent.connectors())
    }
}
