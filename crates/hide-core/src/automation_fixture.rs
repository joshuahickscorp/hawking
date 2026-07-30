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

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;


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
