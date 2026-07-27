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
use crate::persistence::DynKeyValueStore;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

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
        })
    }
}

/// Capability handed to a spawned job. **Structurally non-widening**: fields are
/// private; the only construction paths are [`PermissionSet::derive_capability`]
/// and [`PermissionSet::derive_capability_subset`]. No `grant_tool` / `add` API.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct JobCapability {
    tools: BTreeSet<String>,
    connectors: BTreeSet<String>,
}

impl JobCapability {
    pub fn tools(&self) -> &BTreeSet<String> {
        &self.tools
    }

    pub fn connectors(&self) -> &BTreeSet<String> {
        &self.connectors
    }

    pub fn allows_tool(&self, name: &str) -> bool {
        self.tools.contains(name)
    }

    pub fn allows_connector(&self, name: &str) -> bool {
        self.connectors.contains(name)
    }

    /// Fail-closed tool gate. Returns `Ok(())` only when the tool is granted.
    pub fn require_tool(&self, name: &str) -> Result<()> {
        if self.allows_tool(name) {
            Ok(())
        } else {
            Err(HideError::PolicyDenied(format!(
                "job capability does not grant tool '{name}'"
            )))
        }
    }

    /// True iff every tool/connector in `self` is also in `parent` (subset or equal).
    pub fn is_within(&self, parent: &PermissionSet) -> bool {
        self.tools.is_subset(parent.tools()) && self.connectors.is_subset(parent.connectors())
    }
}

// ---------------------------------------------------------------------------
// Fixture tool registry (no real execution)
// ---------------------------------------------------------------------------

/// Outcome of a fixture tool invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureToolResult {
    pub ok: bool,
    pub output: Value,
    pub tokens_used: u64,
}

/// Named fixture tool: deterministic canned response, no side effects.
#[derive(Debug, Clone)]
pub struct FixtureTool {
    pub name: String,
    pub response: Value,
    pub tokens_used: u64,
}

impl FixtureTool {
    pub fn new(name: impl Into<String>, response: Value) -> Self {
        Self {
            name: name.into(),
            response,
            tokens_used: 1,
        }
    }

    pub fn with_tokens(mut self, tokens: u64) -> Self {
        self.tokens_used = tokens;
        self
    }

    pub fn invoke(&self, _args: &Value) -> FixtureToolResult {
        FixtureToolResult {
            ok: true,
            output: self.response.clone(),
            tokens_used: self.tokens_used,
        }
    }
}

/// In-memory fixture registry used by automation jobs instead of real tools.
#[derive(Debug, Default, Clone)]
pub struct FixtureToolRegistry {
    tools: BTreeMap<String, FixtureTool>,
}

impl FixtureToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, tool: FixtureTool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    pub fn with(mut self, tool: FixtureTool) -> Self {
        self.register(tool);
        self
    }

    pub fn get(&self, name: &str) -> Option<&FixtureTool> {
        self.tools.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(String::as_str)
    }
}

/// Standard fixture catalog for tests (read-ish stubs only).
pub fn standard_fixture_registry() -> FixtureToolRegistry {
    FixtureToolRegistry::new()
        .with(FixtureTool::new(
            "email.list",
            json!({"messages": [{"id": "m1", "subject": "hello"}]}),
        ))
        .with(FixtureTool::new(
            "email.summarize",
            json!({"summary": "one unread"}),
        ))
        .with(FixtureTool::new(
            "calendar.list",
            json!({"events": [{"title": "standup", "at": "09:00"}]}),
        ))
        .with(FixtureTool::new(
            "calendar.prepare",
            json!({"brief": "standup at 09:00"}),
        ))
        .with(FixtureTool::new(
            "fs.read",
            json!({"path": "README.md", "bytes": 12}),
        ))
        .with(FixtureTool::new(
            "web.search",
            json!({"hits": [{"title": "fixture", "url": "https://example.test"}]}),
        ))
        .with(FixtureTool::new(
            "shell.run",
            json!({"exit": 0, "stdout": "ok"}),
        ))
        .with(FixtureTool::new(
            "notify.send",
            json!({"delivered": true}),
        ))
}

// ---------------------------------------------------------------------------
// Results and the automation declaration
// ---------------------------------------------------------------------------

/// One recorded tool attempt inside a job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolAttempt {
    pub tool: String,
    pub ok: bool,
    /// When false, the attempt was blocked by capability (authority containment).
    pub authorized: bool,
    pub detail: String,
    pub output: Option<Value>,
}

/// Outcome of one automation job run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunResult {
    pub job_id: AutomationJobId,
    pub automation_id: AutomationId,
    pub started_ms: u64,
    pub finished_ms: u64,
    pub ok: bool,
    pub schedule_slot: Option<String>,
    pub tool_attempts: Vec<ToolAttempt>,
    pub stop_reason: Option<StopReason>,
    pub summary: String,
    pub tokens_used: u64,
    pub notifications: Vec<String>,
}

/// Full automation declaration + live bookkeeping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Automation {
    pub id: AutomationId,
    pub kind: AutomationKind,
    pub goal: String,
    pub trigger: TriggerSpec,
    /// Connector names the automation may use (also mirrored into permissions).
    pub connectors: Vec<String>,
    /// Tool names the automation may use (also mirrored into permissions).
    pub tools: Vec<String>,
    /// Closed permission set. Jobs derive capability from this and cannot widen.
    pub permissions: PermissionSet,
    pub budget: ResourceBudget,
    pub usage: BudgetUsage,
    pub notification_policy: NotificationPolicy,
    pub stop_condition: StopCondition,
    pub status: AutomationStatus,
    pub stop_reason: Option<StopReason>,
    pub last_result: Option<RunResult>,
    /// Last N results (newest last). Capped by the engine's `result_history_limit`.
    pub results: Vec<RunResult>,
    pub next_run_ms: Option<u64>,
    /// Schedule slots already fired (idempotent triggers).
    pub fired_slots: BTreeSet<String>,
    pub failure_count: u32,
    pub created_ms: u64,
    pub updated_ms: u64,
}

impl Automation {
    /// Build a new active automation. Permissions are exactly `tools` ∪ `connectors`.
    pub fn declare(
        kind: AutomationKind,
        goal: impl Into<String>,
        trigger: TriggerSpec,
        tools: impl IntoIterator<Item = impl Into<String>>,
        connectors: impl IntoIterator<Item = impl Into<String>>,
        budget: ResourceBudget,
        notification_policy: NotificationPolicy,
        stop_condition: StopCondition,
        now_ms: u64,
    ) -> Self {
        let tools: Vec<String> = tools.into_iter().map(Into::into).collect();
        let connectors: Vec<String> = connectors.into_iter().map(Into::into).collect();
        let permissions = PermissionSet::new(tools.clone(), connectors.clone());
        let next_run_ms = match &trigger {
            TriggerSpec::Once { at_ms } => Some(*at_ms),
            TriggerSpec::Interval { anchor_ms, .. } => Some(*anchor_ms),
            TriggerSpec::CronSlot { at_ms, .. } => Some(*at_ms),
            TriggerSpec::Manual | TriggerSpec::Watch { .. } => None,
        };
        Self {
            id: AutomationId::new(),
            kind,
            goal: goal.into(),
            trigger,
            connectors,
            tools,
            permissions,
            budget,
            usage: BudgetUsage::default(),
            notification_policy,
            stop_condition,
            status: AutomationStatus::Active,
            stop_reason: None,
            last_result: None,
            results: Vec::new(),
            next_run_ms,
            fired_slots: BTreeSet::new(),
            failure_count: 0,
            created_ms: now_ms,
            updated_ms: now_ms,
        }
    }

    /// Readable full declaration (inspectability).
    pub fn declaration(&self) -> Value {
        json!({
            "id": self.id.as_str(),
            "kind": self.kind,
            "goal": self.goal,
            "trigger": self.trigger,
            "connectors": self.connectors,
            "tools": self.tools,
            "permissions": {
                "tools": self.permissions.tools().iter().cloned().collect::<Vec<_>>(),
                "connectors": self.permissions.connectors().iter().cloned().collect::<Vec<_>>(),
            },
            "budget": self.budget,
            "usage": self.usage,
            "notification_policy": self.notification_policy,
            "stop_condition": self.stop_condition,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "next_run_ms": self.next_run_ms,
            "last_result": self.last_result,
            "failure_count": self.failure_count,
            "created_ms": self.created_ms,
            "updated_ms": self.updated_ms,
        })
    }

    /// Last `n` results (newest last). Inspectability surface.
    pub fn last_n_results(&self, n: usize) -> &[RunResult] {
        let len = self.results.len();
        let start = len.saturating_sub(n);
        &self.results[start..]
    }

    fn should_stop_after_run(&self) -> Option<StopReason> {
        if let Some(axis) = self.usage.exhausted_axis(&self.budget) {
            return Some(StopReason::BudgetExhausted {
                axis: axis.to_string(),
            });
        }
        match &self.stop_condition {
            StopCondition::Never => None,
            StopCondition::AfterRuns { count } => {
                if self.usage.runs >= *count {
                    Some(StopReason::AfterRuns { count: *count })
                } else {
                    None
                }
            }
            StopCondition::ConditionMet { .. } => None, // evaluated on event
            StopCondition::MaxFailures { count } => {
                if self.failure_count >= *count {
                    Some(StopReason::MaxFailures { count: *count })
                } else {
                    None
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Durable store
// ---------------------------------------------------------------------------

/// Persist automations in a [`KeyValueStore`] so they survive process restart.
pub struct AutomationStore;

impl AutomationStore {
    pub const NAMESPACE: &'static str = "you_automations";

    pub fn put(kv: &DynKeyValueStore, automation: &Automation) -> Result<()> {
        let value = serde_json::to_value(automation)?;
        kv.put(Self::NAMESPACE, automation.id.as_str(), value)
    }

    pub fn get(kv: &DynKeyValueStore, id: &str) -> Result<Option<Automation>> {
        match kv.get(Self::NAMESPACE, id)? {
            Some(value) => Ok(Some(serde_json::from_value(value)?)),
            None => Ok(None),
        }
    }

    pub fn list_all(kv: &DynKeyValueStore) -> Result<Vec<Automation>> {
        let mut out: Vec<Automation> = kv
            .list(Self::NAMESPACE)?
            .into_iter()
            .filter_map(|(_, value)| serde_json::from_value(value).ok())
            .collect();
        out.sort_by(|a, b| {
            a.created_ms
                .cmp(&b.created_ms)
                .then_with(|| a.id.as_str().cmp(b.id.as_str()))
        });
        Ok(out)
    }

    pub fn delete(kv: &DynKeyValueStore, id: &str) -> Result<()> {
        kv.delete(Self::NAMESPACE, id)
    }
}

// ---------------------------------------------------------------------------
// Job + engine
// ---------------------------------------------------------------------------

/// A single in-flight (or completed) job spawned from an automation.
///
/// Holds a [`JobCapability`] derived from the parent automation's
/// [`PermissionSet`]. Tool dispatch goes through [`AutomationJob::use_tool`],
/// which fails closed on unauthorized tools.
#[derive(Debug, Clone)]
pub struct AutomationJob {
    pub id: AutomationJobId,
    pub automation_id: AutomationId,
    pub capability: JobCapability,
    pub schedule_slot: Option<String>,
    pub started_ms: u64,
    pub tool_attempts: Vec<ToolAttempt>,
    pub tokens_used: u64,
    pub halted: Option<StopReason>,
}

impl AutomationJob {
    /// Invoke a fixture tool under this job's capability. Unauthorized tools
    /// fail closed and are recorded; the job is marked halted with
    /// [`StopReason::AuthorityDenied`].
    pub fn use_tool(
        &mut self,
        registry: &FixtureToolRegistry,
        tool: &str,
        args: Value,
    ) -> Result<FixtureToolResult> {
        if let Err(err) = self.capability.require_tool(tool) {
            self.tool_attempts.push(ToolAttempt {
                tool: tool.to_string(),
                ok: false,
                authorized: false,
                detail: err.to_string(),
                output: None,
            });
            self.halted = Some(StopReason::AuthorityDenied {
                tool: tool.to_string(),
            });
            return Err(err);
        }
        let fixture = registry.get(tool).ok_or_else(|| {
            HideError::NotFound(format!("fixture tool '{tool}' not in registry"))
        })?;
        let result = fixture.invoke(&args);
        self.tokens_used += result.tokens_used;
        self.tool_attempts.push(ToolAttempt {
            tool: tool.to_string(),
            ok: result.ok,
            authorized: true,
            detail: "ok".to_string(),
            output: Some(result.output.clone()),
        });
        Ok(result)
    }
}

/// Plan of tool calls a job should attempt (fixture-level "agent body").
#[derive(Debug, Clone, Default)]
pub struct JobPlan {
    pub tool_calls: Vec<(String, Value)>,
}

/// Engine: create, persist, tick, spawn, enforce stop, inspect.
pub struct AutomationEngine {
    kv: DynKeyValueStore,
    clock: Arc<dyn Clock>,
    registry: FixtureToolRegistry,
    result_history_limit: usize,
    /// In-memory index of loaded automations (mirrors durable store).
    live: RwLock<BTreeMap<String, Automation>>,
}

impl AutomationEngine {
    pub fn new(
        kv: DynKeyValueStore,
        clock: Arc<dyn Clock>,
        registry: FixtureToolRegistry,
    ) -> Self {
        Self {
            kv,
            clock,
            registry,
            result_history_limit: 32,
            live: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn with_history_limit(mut self, n: usize) -> Self {
        self.result_history_limit = n.max(1);
        self
    }

    /// Load every durable automation into the live map (restart recovery).
    pub fn recover(&self) -> Result<Vec<AutomationId>> {
        let all = AutomationStore::list_all(&self.kv)?;
        let mut live = self.live.write();
        live.clear();
        let mut ids = Vec::new();
        for a in all {
            ids.push(a.id.clone());
            live.insert(a.id.as_str().to_string(), a);
        }
        Ok(ids)
    }

    /// Register a new automation and durably persist it.
    pub fn create(&self, mut automation: Automation) -> Result<Automation> {
        let now = self.clock.now_ms();
        automation.updated_ms = now;
        if automation.created_ms == 0 {
            automation.created_ms = now;
        }
        AutomationStore::put(&self.kv, &automation)?;
        let id = automation.id.as_str().to_string();
        self.live.write().insert(id, automation.clone());
        Ok(automation)
    }

    pub fn get(&self, id: &str) -> Option<Automation> {
        self.live.read().get(id).cloned()
    }

    pub fn list(&self) -> Vec<Automation> {
        let mut out: Vec<_> = self.live.read().values().cloned().collect();
        out.sort_by(|a, b| {
            a.created_ms
                .cmp(&b.created_ms)
                .then_with(|| a.id.as_str().cmp(b.id.as_str()))
        });
        out
    }

    /// Inspectable declaration + last N results.
    pub fn inspect(&self, id: &str, last_n: usize) -> Result<Value> {
        let a = self
            .get(id)
            .ok_or_else(|| HideError::NotFound(format!("automation '{id}'")))?;
        Ok(json!({
            "declaration": a.declaration(),
            "results": a.last_n_results(last_n),
        }))
    }

    fn persist(&self, automation: &Automation) -> Result<()> {
        AutomationStore::put(&self.kv, automation)?;
        self.live
            .write()
            .insert(automation.id.as_str().to_string(), automation.clone());
        Ok(())
    }

    /// Cancel an automation (terminal).
    pub fn cancel(&self, id: &str) -> Result<Automation> {
        let mut a = self
            .get(id)
            .ok_or_else(|| HideError::NotFound(format!("automation '{id}'")))?;
        a.status = AutomationStatus::Cancelled;
        a.stop_reason = Some(StopReason::Cancelled);
        a.next_run_ms = None;
        a.updated_ms = self.clock.now_ms();
        self.persist(&a)?;
        Ok(a)
    }

    /// Present a named watch/stop condition as met. If it matches the stop
    /// condition, the automation halts. If it matches a watch trigger, fires once.
    pub fn signal_condition(&self, id: &str, name: &str) -> Result<Option<RunResult>> {
        let mut a = self
            .get(id)
            .ok_or_else(|| HideError::NotFound(format!("automation '{id}'")))?;

        if matches!(&a.stop_condition, StopCondition::ConditionMet { name: n } if n == name) {
            a.status = AutomationStatus::Stopped;
            a.stop_reason = Some(StopReason::ConditionMet {
                name: name.to_string(),
            });
            a.next_run_ms = None;
            a.updated_ms = self.clock.now_ms();
            self.persist(&a)?;
            return Ok(None);
        }

        if matches!(&a.trigger, TriggerSpec::Watch { condition } if condition == name)
            && a.status.may_run()
        {
            let slot = format!("watch:{name}");
            return self.run_slot(&mut a, Some(slot), JobPlan::default());
        }

        Ok(None)
    }

    /// Manual wake. Each call mints a unique slot (run count + job mint) so
    /// two manual wakes at the same clock time are not collapsed by schedule
    /// idempotency — that guard applies to schedule slots, not owner-initiated runs.
    pub fn run_manual(&self, id: &str, plan: JobPlan) -> Result<RunResult> {
        let mut a = self
            .get(id)
            .ok_or_else(|| HideError::NotFound(format!("automation '{id}'")))?;
        if !a.status.may_run() {
            return Err(HideError::InvalidState(format!(
                "automation '{id}' is {:?}, cannot run",
                a.status
            )));
        }
        let slot = format!(
            "manual:{}:{}",
            self.clock.now_ms(),
            a.usage.runs.saturating_add(1)
        );
        self.run_slot(&mut a, Some(slot), plan)?
            .ok_or_else(|| HideError::InvalidState("manual run produced no result".into()))
    }

    /// Advance the engine: fire every due, not-yet-fired schedule slot once.
    /// Returns results produced on this tick.
    pub fn tick(&self, default_plan: &JobPlan) -> Result<Vec<RunResult>> {
        let now = self.clock.now_ms();
        let ids: Vec<String> = self.live.read().keys().cloned().collect();
        let mut results = Vec::new();
        for id in ids {
            let Some(mut a) = self.get(&id) else {
                continue;
            };
            if !a.status.may_run() {
                continue;
            }
            // Budget check before spawn.
            if let Some(axis) = a.usage.exhausted_axis(&a.budget) {
                a.status = AutomationStatus::Stopped;
                a.stop_reason = Some(StopReason::BudgetExhausted {
                    axis: axis.to_string(),
                });
                a.next_run_ms = None;
                a.updated_ms = now;
                self.persist(&a)?;
                continue;
            }
            let Some(slot) = a.trigger.slot_if_due(now) else {
                continue;
            };
            if a.fired_slots.contains(&slot) {
                // Idempotent: already ran this slot.
                // Still refresh next_run for intervals.
                a.next_run_ms = a.trigger.next_run_after(now);
                a.updated_ms = now;
                self.persist(&a)?;
                continue;
            }
            if let Some(result) = self.run_slot(&mut a, Some(slot), default_plan.clone())? {
                results.push(result);
            }
        }
        Ok(results)
    }

    /// Fire a specific schedule slot (used by tests and tick). Idempotent: a
    /// second fire of the same slot is a no-op that returns `None`.
    pub fn fire_slot(
        &self,
        id: &str,
        slot: impl Into<String>,
        plan: JobPlan,
    ) -> Result<Option<RunResult>> {
        let mut a = self
            .get(id)
            .ok_or_else(|| HideError::NotFound(format!("automation '{id}'")))?;
        if !a.status.may_run() {
            return Ok(None);
        }
        self.run_slot(&mut a, Some(slot.into()), plan)
    }

    fn run_slot(
        &self,
        a: &mut Automation,
        slot: Option<String>,
        plan: JobPlan,
    ) -> Result<Option<RunResult>> {
        let now = self.clock.now_ms();
        if let Some(ref s) = slot {
            if a.fired_slots.contains(s) {
                return Ok(None);
            }
        }
        if let Some(axis) = a.usage.exhausted_axis(&a.budget) {
            a.status = AutomationStatus::Stopped;
            a.stop_reason = Some(StopReason::BudgetExhausted {
                axis: axis.to_string(),
            });
            a.next_run_ms = None;
            a.updated_ms = now;
            self.persist(a)?;
            return Ok(None);
        }

        // STRUCTURAL: capability is derived from the automation permission set.
        let capability = a.permissions.derive_capability();
        debug_assert!(
            capability.is_within(&a.permissions),
            "derived capability must be within the automation permission set"
        );

        let mut job = AutomationJob {
            id: AutomationJobId::new(),
            automation_id: a.id.clone(),
            capability,
            schedule_slot: slot.clone(),
            started_ms: now,
            tool_attempts: Vec::new(),
            tokens_used: 0,
            halted: None,
        };

        // Execute the plan under the capability gate.
        let mut plan_ok = true;
        for (tool, args) in &plan.tool_calls {
            // Per-call budget: max_tool_calls.
            if let Some(max) = a.budget.max_tool_calls {
                if a.usage.tool_calls >= max {
                    job.halted = Some(StopReason::BudgetExhausted {
                        axis: "max_tool_calls".into(),
                    });
                    plan_ok = false;
                    break;
                }
            }
            match job.use_tool(&self.registry, tool, args.clone()) {
                Ok(_) => {
                    a.usage.tool_calls = a.usage.tool_calls.saturating_add(1);
                }
                Err(_) => {
                    plan_ok = false;
                    break;
                }
            }
            if let Some(max_tokens) = a.budget.max_tokens {
                if a.usage.tokens.saturating_add(job.tokens_used) >= max_tokens {
                    // Will be reflected after usage update.
                }
            }
        }

        let finished = self.clock.now_ms();
        let wall = finished.saturating_sub(now);
        a.usage.wall_ms = a.usage.wall_ms.saturating_add(wall);
        a.usage.tokens = a.usage.tokens.saturating_add(job.tokens_used);
        a.usage.runs = a.usage.runs.saturating_add(1);

        if !plan_ok {
            a.failure_count = a.failure_count.saturating_add(1);
        }

        // Job-level halt (e.g. authority denial) is always recorded on the result.
        // Automation-level halt is only for stop conditions / budget — a single
        // unauthorized tool attempt must not permanently privilege-revoke the
        // standing automation; it fails the job closed and is auditable.
        let job_stop = job.halted.clone();
        let automation_stop = a.should_stop_after_run();
        let result_stop = job_stop.clone().or_else(|| automation_stop.clone());

        let notifications = notifications_for(&a.notification_policy, plan_ok);

        let result = RunResult {
            job_id: job.id.clone(),
            automation_id: a.id.clone(),
            started_ms: now,
            finished_ms: finished,
            ok: plan_ok && job.halted.is_none(),
            schedule_slot: slot.clone(),
            tool_attempts: job.tool_attempts.clone(),
            stop_reason: result_stop,
            summary: if plan_ok {
                format!("completed {} tool call(s)", job.tool_attempts.len())
            } else if matches!(
                job.halted,
                Some(StopReason::AuthorityDenied { .. })
            ) {
                "halted: authority denied".into()
            } else {
                "failed".into()
            },
            tokens_used: job.tokens_used,
            notifications,
        };

        if let Some(ref s) = slot {
            a.fired_slots.insert(s.clone());
        }
        a.last_result = Some(result.clone());
        a.results.push(result.clone());
        if a.results.len() > self.result_history_limit {
            let excess = a.results.len() - self.result_history_limit;
            a.results.drain(0..excess);
        }

        if let Some(reason) = automation_stop {
            a.status = AutomationStatus::Stopped;
            a.stop_reason = Some(reason);
            a.next_run_ms = None;
        } else {
            a.next_run_ms = a.trigger.next_run_after(now);
        }
        a.updated_ms = finished;
        self.persist(a)?;
        Ok(Some(result))
    }

    pub fn registry(&self) -> &FixtureToolRegistry {
        &self.registry
    }

    pub fn clock(&self) -> &dyn Clock {
        self.clock.as_ref()
    }
}

fn notifications_for(policy: &NotificationPolicy, ok: bool) -> Vec<String> {
    match policy {
        NotificationPolicy::Silent => Vec::new(),
        NotificationPolicy::OnFailure if !ok => vec!["failure".into()],
        NotificationPolicy::OnSuccess if ok => vec!["success".into()],
        NotificationPolicy::Always => {
            vec![if ok {
                "success".into()
            } else {
                "failure".into()
            }]
        }
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::InMemoryKeyValueStore;

    fn engine_at(ms: u64) -> (Arc<InjectedClock>, AutomationEngine) {
        let clock = Arc::new(InjectedClock::new(ms));
        let kv: DynKeyValueStore = Arc::new(InMemoryKeyValueStore::default());
        let engine = AutomationEngine::new(kv, clock.clone(), standard_fixture_registry());
        (clock, engine)
    }

    fn sample_automation(now: u64) -> Automation {
        Automation::declare(
            AutomationKind::EmailTriage,
            "triage unread mail",
            TriggerSpec::Interval {
                every_ms: 60_000,
                anchor_ms: now,
            },
            ["email.list", "email.summarize"],
            ["gmail"],
            ResourceBudget {
                max_runs: Some(10),
                max_tool_calls: Some(50),
                max_wall_ms: None,
                max_tokens: Some(1_000),
            },
            NotificationPolicy::OnFailure,
            StopCondition::Never,
            now,
        )
    }

    #[test]
    fn capability_is_derived_and_cannot_be_widened() {
        let perms = PermissionSet::new(["email.list", "email.summarize"], ["gmail"]);
        let cap = perms.derive_capability();
        assert!(cap.allows_tool("email.list"));
        assert!(!cap.allows_tool("shell.run"));
        assert!(cap.is_within(&perms));

        // Subset derivation works.
        let sub = perms
            .derive_capability_subset(["email.list"], None::<&str>)
            .unwrap();
        assert!(sub.allows_tool("email.list"));
        assert!(!sub.allows_tool("email.summarize"));

        // Cannot derive a tool the automation never granted.
        let err = perms
            .derive_capability_subset(["shell.run"], None::<&str>)
            .unwrap_err();
        assert!(
            matches!(err, HideError::CapabilityMissing(_)),
            "expected CapabilityMissing, got {err:?}"
        );
    }

    #[test]
    fn authority_containment_fail_closed_and_recorded() {
        let (clock, engine) = engine_at(1_000);
        let a = sample_automation(clock.now_ms());
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();

        // Plan includes a tool the automation did NOT grant.
        let plan = JobPlan {
            tool_calls: vec![
                ("email.list".into(), json!({})),
                ("shell.run".into(), json!({"cmd": "rm -rf /"})),
            ],
        };
        let result = engine.run_manual(&id, plan).unwrap();

        assert!(!result.ok, "job must fail closed on ungranted tool");
        assert!(
            result.tool_attempts.iter().any(|t| {
                t.tool == "shell.run" && !t.authorized && !t.ok
            }),
            "ungranted tool attempt must be recorded as unauthorized: {:?}",
            result.tool_attempts
        );
        assert!(
            matches!(
                result.stop_reason,
                Some(StopReason::AuthorityDenied { ref tool }) if tool == "shell.run"
            ),
            "stop reason must record authority denial: {:?}",
            result.stop_reason
        );

        // Job failed closed; the standing automation remains (failure is audited
        // on the result, not a permanent kill of the declaration).
        let a = engine.get(&id).unwrap();
        assert_eq!(
            a.status,
            AutomationStatus::Active,
            "authority denial fails the job, not the automation declaration"
        );
        assert!(a.last_result.as_ref().is_some_and(|r| !r.ok));
        // Inspectable history contains the denial.
        let inspected = engine.inspect(&id, 5).unwrap();
        let results = inspected["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["ok"], false);
    }

    #[test]
    fn authorized_tools_succeed_under_capability() {
        let (clock, engine) = engine_at(1_000);
        let a = sample_automation(clock.now_ms());
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();

        let plan = JobPlan {
            tool_calls: vec![
                ("email.list".into(), json!({})),
                ("email.summarize".into(), json!({})),
            ],
        };
        let result = engine.run_manual(&id, plan).unwrap();
        assert!(result.ok);
        assert_eq!(result.tool_attempts.len(), 2);
        assert!(result.tool_attempts.iter().all(|t| t.authorized && t.ok));
    }

    #[test]
    fn durable_across_restart_next_run_survives() {
        let clock = Arc::new(InjectedClock::new(10_000));
        let kv: DynKeyValueStore = Arc::new(InMemoryKeyValueStore::default());
        let registry = standard_fixture_registry();

        let next_run = {
            let engine = AutomationEngine::new(kv.clone(), clock.clone(), registry.clone());
            let mut a = sample_automation(clock.now_ms());
            // Pin a known next run.
            a.next_run_ms = Some(99_000);
            let created = engine.create(a).unwrap();
            let id = created.id.as_str().to_string();
            // Drop the engine (process "restart"): only the KV remains.
            drop(engine);
            (id, 99_000u64)
        };

        // Fresh engine, same durable store.
        let engine2 = AutomationEngine::new(kv, clock, registry);
        let recovered = engine2.recover().unwrap();
        assert_eq!(recovered.len(), 1);
        let a = engine2.get(next_run.0.as_str()).unwrap();
        assert_eq!(a.next_run_ms, Some(next_run.1));
        assert_eq!(a.goal, "triage unread mail");
        assert!(a.permissions.grants_tool("email.list"));
    }

    #[test]
    fn stop_condition_after_runs_is_enforced() {
        let (clock, engine) = engine_at(5_000);
        let a = Automation::declare(
            AutomationKind::Reminder,
            "nudge me twice",
            TriggerSpec::Manual,
            ["notify.send"],
            None::<&str>,
            ResourceBudget::default(),
            NotificationPolicy::Silent,
            StopCondition::AfterRuns { count: 2 },
            clock.now_ms(),
        );
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();

        let plan = JobPlan {
            tool_calls: vec![("notify.send".into(), json!({}))],
        };
        let r1 = engine.run_manual(&id, plan.clone()).unwrap();
        assert!(r1.ok);
        assert!(engine.get(&id).unwrap().status.may_run());

        let r2 = engine.run_manual(&id, plan).unwrap();
        assert!(r2.ok);
        assert!(
            matches!(
                r2.stop_reason,
                Some(StopReason::AfterRuns { count: 2 })
            ),
            "second run must record AfterRuns: {:?}",
            r2.stop_reason
        );
        let a = engine.get(&id).unwrap();
        assert_eq!(a.status, AutomationStatus::Stopped);
        assert!(matches!(
            a.stop_reason,
            Some(StopReason::AfterRuns { count: 2 })
        ));

        // Further manual runs refuse.
        let err = engine
            .run_manual(
                &id,
                JobPlan {
                    tool_calls: vec![("notify.send".into(), json!({}))],
                },
            )
            .unwrap_err();
        assert!(matches!(err, HideError::InvalidState(_)));
    }

    #[test]
    fn budget_exhaustion_halts_and_records_why() {
        let (clock, engine) = engine_at(0);
        let a = Automation::declare(
            AutomationKind::ResearchMonitor,
            "watch once",
            TriggerSpec::Manual,
            ["web.search"],
            None::<&str>,
            ResourceBudget {
                max_runs: Some(1),
                max_tool_calls: None,
                max_wall_ms: None,
                max_tokens: None,
            },
            NotificationPolicy::Always,
            StopCondition::Never,
            clock.now_ms(),
        );
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();

        let r = engine
            .run_manual(
                &id,
                JobPlan {
                    tool_calls: vec![("web.search".into(), json!({}))],
                },
            )
            .unwrap();
        assert!(r.ok);
        assert!(
            matches!(
                r.stop_reason,
                Some(StopReason::BudgetExhausted { ref axis }) if axis == "max_runs"
            ),
            "budget stop reason: {:?}",
            r.stop_reason
        );
        let a = engine.get(&id).unwrap();
        assert_eq!(a.status, AutomationStatus::Stopped);
        assert!(!a.results[0].notifications.is_empty());
    }

    #[test]
    fn condition_met_stop_is_enforced() {
        let (clock, engine) = engine_at(0);
        let a = Automation::declare(
            AutomationKind::WatchCondition,
            "watch deploy",
            TriggerSpec::Watch {
                condition: "deploy_green".into(),
            },
            ["notify.send"],
            None::<&str>,
            ResourceBudget::default(),
            NotificationPolicy::Silent,
            StopCondition::ConditionMet {
                name: "deploy_green".into(),
            },
            clock.now_ms(),
        );
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();

        // Signaling the stop condition halts without a run.
        let result = engine.signal_condition(&id, "deploy_green").unwrap();
        assert!(result.is_none());
        let a = engine.get(&id).unwrap();
        assert_eq!(a.status, AutomationStatus::Stopped);
        assert!(matches!(
            a.stop_reason,
            Some(StopReason::ConditionMet { ref name }) if name == "deploy_green"
        ));
    }

    #[test]
    fn schedule_slot_is_idempotent() {
        let (clock, engine) = engine_at(100_000);
        let a = Automation::declare(
            AutomationKind::RecurringBrief,
            "morning brief",
            TriggerSpec::CronSlot {
                slot_key: "2026-07-27T09:00".into(),
                at_ms: 100_000,
            },
            ["calendar.list", "calendar.prepare"],
            ["calendar"],
            ResourceBudget::default(),
            NotificationPolicy::Silent,
            StopCondition::Never,
            clock.now_ms(),
        );
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();

        let plan = JobPlan {
            tool_calls: vec![
                ("calendar.list".into(), json!({})),
                ("calendar.prepare".into(), json!({})),
            ],
        };

        // Tick fires once.
        let r1 = engine.tick(&plan).unwrap();
        assert_eq!(r1.len(), 1, "first tick should run once");
        assert!(r1[0].ok);

        // Same clock, same slot: second tick must not re-run.
        let r2 = engine.tick(&plan).unwrap();
        assert!(
            r2.is_empty(),
            "idempotent: same slot must not run twice, got {:?}",
            r2
        );

        // Explicit fire of the same slot is also a no-op.
        let slot = "cron:2026-07-27T09:00";
        let r3 = engine.fire_slot(&id, slot, plan).unwrap();
        assert!(r3.is_none());

        let a = engine.get(&id).unwrap();
        assert_eq!(a.usage.runs, 1);
        assert_eq!(a.results.len(), 1);
    }

    #[test]
    fn interval_slots_advance_and_do_not_double_fire() {
        let (clock, engine) = engine_at(0);
        let a = Automation::declare(
            AutomationKind::ProjectStatusCheck,
            "hourly status",
            TriggerSpec::Interval {
                every_ms: 3_600_000,
                anchor_ms: 0,
            },
            ["fs.read"],
            None::<&str>,
            ResourceBudget::default(),
            NotificationPolicy::Silent,
            StopCondition::Never,
            0,
        );
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();

        let plan = JobPlan {
            tool_calls: vec![("fs.read".into(), json!({}))],
        };

        // At t=0, slot interval:3600000:0 is due.
        let r0 = engine.tick(&plan).unwrap();
        assert_eq!(r0.len(), 1);
        assert_eq!(r0[0].schedule_slot.as_deref(), Some("interval:3600000:0"));

        // Still in the same hour: no re-fire.
        clock.advance(1_000);
        let r_same = engine.tick(&plan).unwrap();
        assert!(r_same.is_empty());

        // Next hour: new slot.
        clock.set(3_600_000);
        let r1 = engine.tick(&plan).unwrap();
        assert_eq!(r1.len(), 1);
        assert_eq!(r1[0].schedule_slot.as_deref(), Some("interval:3600000:1"));

        let a = engine.get(&id).unwrap();
        assert_eq!(a.usage.runs, 2);
        assert_eq!(a.next_run_ms, Some(7_200_000));
    }

    #[test]
    fn inspect_exposes_full_declaration_and_history() {
        let (clock, engine) = engine_at(0);
        let a = sample_automation(clock.now_ms());
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();
        engine
            .run_manual(
                &id,
                JobPlan {
                    tool_calls: vec![("email.list".into(), json!({}))],
                },
            )
            .unwrap();

        let view = engine.inspect(&id, 10).unwrap();
        assert_eq!(view["declaration"]["goal"], "triage unread mail");
        assert!(view["declaration"]["permissions"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t == "email.list"));
        assert_eq!(view["results"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn job_capability_has_no_public_widen_path() {
        // Compile-time documentation test: JobCapability fields are private,
        // so the only construction is via PermissionSet. Runtime check: a
        // capability equal to a parent set cannot claim a tool outside it.
        let parent = PermissionSet::new(["fs.read"], None::<&str>);
        let cap = parent.derive_capability();
        assert!(!cap.allows_tool("fs.write"));
        assert!(cap.require_tool("fs.write").is_err());
    }
}
