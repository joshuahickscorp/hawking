use crate::error::{HideError, Result};
use crate::ids::{GrantId, ToolCallId};
use crate::permission::{PermissionEngine, PermissionRequest};
use crate::types::{BlobRef, Decision, EffectSet, RiskLevel};
use futures::future::BoxFuture;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub title: String,
    pub version: String,
    pub wire_version: u16,
    pub description: String,
    pub input_schema: Value,
    pub output_schema: Option<Value>,
    pub annotations: ToolAnnotations,
    pub capabilities_required: Vec<String>,
    pub output_cap_bytes: u64,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolAnnotations {
    pub read_only: bool,
    pub destructive: bool,
    pub idempotent: bool,
    pub open_world: bool,
}

impl Default for ToolAnnotations {
    fn default() -> Self {
        Self {
            read_only: false,
            destructive: true,
            idempotent: false,
            open_world: true,
        }
    }
}

/// The canonical effect request (ch.03 §4.2.2). Maps 1:1 to the `tool.call`
/// event payload. Field names match the bible wire shape exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// ULID; unique within the run; correlates result + events.
    pub call_id: ToolCallId,
    /// `ToolSpec.name` (registry-resolved).
    pub tool: String,
    pub args: Value,
    /// References Ch.01's grant ledger (TT3).
    pub capability_grant_id: Option<GrantId>,
    pub wire_version: u16,
    /// Optional execution directives (dry-run, idempotency, timeout override).
    #[serde(default)]
    pub x: ToolCallExt,
}

/// Optional per-call execution directives (`ToolCall.x`, ch.03 §4.2.2).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ToolCallExt {
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    /// `≤` the spec cap; cannot exceed it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms_override: Option<u64>,
}

impl ToolCall {
    /// Convenience constructor with a fresh `call_id`, current wire version, and
    /// default execution directives.
    pub fn new(tool: impl Into<String>, args: Value) -> Self {
        Self {
            call_id: ToolCallId::new(),
            tool: tool.into(),
            args,
            capability_grant_id: None,
            wire_version: TOOL_WIRE_VERSION,
            x: ToolCallExt::default(),
        }
    }
}

/// The current tool-wire-format version (ch.03 §4.2, TT11).
pub const TOOL_WIRE_VERSION: u16 = 1;

/// The canonical recorded outcome (ch.03 §4.2.3). Maps 1:1 to the `tool.result`
/// event payload (TT4). `output` is the typed body; large bodies spill to
/// `bytes_ref`. `provenance` marks the body as UNTRUSTED data, not instructions
/// (TT8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: ToolCallId,
    /// `false` ⇒ see `error`; maps to MCP `isError` (§4.10). NOTE: a non-zero
    /// process exit is `ok:true` with `exit_code` set — `EXEC_NONZERO` is data,
    /// not a tool failure (§4.2.3).
    pub ok: bool,
    pub status: ToolStatus,
    /// Optional MCP-style multimodal blocks.
    #[serde(default)]
    pub content: Vec<ToolContent>,
    /// Typed body validated against `ToolSpec.output_schema` (`output`).
    #[serde(default)]
    pub structured_content: Option<Value>,
    /// Large bodies spill here as a blake3 CAS ref (TT5).
    pub bytes_ref: Option<BlobRef>,
    /// For process-shaped tools (shell/test/build); `None` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    pub effects: EffectSet,
    /// TT8: the result body is untrusted data. Defaults to `"tool-output"`.
    #[serde(default = "default_tool_provenance")]
    pub provenance: String,
    /// Execution stats (duration, cache hit, dry-run origin).
    #[serde(default)]
    pub stats: ToolStats,
    pub error: Option<ToolError>,
}

fn default_tool_provenance() -> String {
    "tool-output".to_string()
}

/// Execution stats carried on every `ToolResult` (ch.03 §4.2.3 `stats`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ToolStats {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub cached: bool,
    #[serde(default)]
    pub from_dry_run: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    Ok,
    ToolError,
    ProtocolError,
    Cancelled,
    TimedOut,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolContent {
    Text { text: String },
    Json { value: Value },
    Blob { blob: BlobRef },
}

/// The structured failure (when `ok=false`). Designed to be self-correcting:
/// the agent loop feeds it back so the model can fix the call (ch.03 §4.2.3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolError {
    /// Stable taxonomy code (`ARG_INVALID`/`NOT_FOUND`/`EXEC_NONZERO`/… §4.2.3).
    pub code: String,
    pub message: String,
    /// Can the same model fix-and-retry? (Ch.02 uses this.)
    pub retriable: bool,
    /// Actionable hint for the model to repair the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fix_hint: Option<String>,
    /// JSON-pointer into `args` for precise UI/model targeting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_path: Option<String>,
}

impl ToolError {
    /// Minimal constructor; `fix_hint`/`schema_path` default to `None`.
    pub fn new(code: impl Into<String>, message: impl Into<String>, retriable: bool) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            retriable,
            fix_hint: None,
            schema_path: None,
        }
    }
}

impl ToolResult {
    pub fn ok(call_id: ToolCallId, structured_content: Option<Value>, effects: EffectSet) -> Self {
        Self {
            call_id,
            ok: true,
            status: ToolStatus::Ok,
            content: Vec::new(),
            structured_content,
            bytes_ref: None,
            exit_code: None,
            effects,
            provenance: default_tool_provenance(),
            stats: ToolStats::default(),
            error: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolCtx {
    pub grant_id: Option<GrantId>,
    pub deadline_ms: Option<u64>,
    pub output_cap_bytes: u64,
}

pub trait Tool: Send + Sync {
    fn spec(&self) -> &ToolSpec;

    fn call<'a>(&'a self, args: Value, ctx: ToolCtx) -> BoxFuture<'a, ToolResult>;

    fn simulate<'a>(&'a self, _args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
        Box::pin(async { None })
    }

    fn purity(&self) -> Purity {
        Purity::Impure
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Purity {
    Pure,
    PureFs,
    Impure,
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: RwLock<BTreeMap<String, Arc<dyn Tool>>>,
}

impl ToolRegistry {
    pub fn register<T: Tool + 'static>(&self, tool: T) {
        self.tools
            .write()
            .insert(tool.spec().name.clone(), Arc::new(tool));
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.read().get(name).cloned()
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .read()
            .values()
            .map(|tool| tool.spec().clone())
            .collect()
    }
}

/// Watches EVERY call that applies through a [`ToolDispatcher`].
///
/// The dispatcher is the one place every client of the tool surface meets: the kernel agent holds
/// it directly, the host's own wrappers hold the same type. Recording (durable tool events, the
/// pre/post image an edit can be reviewed and reverted from) hangs off THIS, not off a caller's
/// wrapper, because a wrapper only ever records the callers somebody remembered to route through
/// it. `before` runs after the permission verdict and before the tool applies; whatever it returns
/// is handed back to `after` with the result. Neither runs for a dry run (nothing applied).
pub trait DispatchObserver: Send + Sync {
    /// Snapshot taken before the tool applies (the pre-image of a write). `None` to record nothing
    /// beyond the call itself.
    fn before(&self, _call: &ToolCall) -> Option<Value> {
        None
    }

    /// Record the applied call. Runs after the tool returned, whatever its status.
    fn after<'a>(
        &'a self,
        call: &'a ToolCall,
        before: Option<Value>,
        result: &'a ToolResult,
    ) -> BoxFuture<'a, ()>;
}

pub struct ToolDispatcher {
    registry: Arc<ToolRegistry>,
    policy: Arc<dyn PermissionEngine>,
    observer: Option<Arc<dyn DispatchObserver>>,
}

impl ToolDispatcher {
    pub fn new(registry: Arc<ToolRegistry>, policy: Arc<dyn PermissionEngine>) -> Self {
        Self {
            registry,
            policy,
            observer: None,
        }
    }

    /// Attach the recorder every dispatch reports to. Set once, at construction, so no caller can
    /// hold a dispatcher that silently records nothing.
    pub fn with_observer(mut self, observer: Arc<dyn DispatchObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Whether the named tool is registered and declares itself read-only. Used to
    /// decide whether a model-emitted call may be auto-dispatched from a model step
    /// (read-only only) versus requiring an authorized plan step (any mutation).
    /// Unknown tools return `false` (conservative: not auto-dispatchable).
    pub fn is_read_only(&self, name: &str) -> bool {
        self.registry
            .get(name)
            .map(|tool| tool.spec().annotations.read_only)
            .unwrap_or(false)
    }

    pub async fn dispatch(&self, call: ToolCall) -> Result<ToolResult> {
        let call_id = call.call_id.clone();
        let tool = self
            .registry
            .get(&call.tool)
            .ok_or_else(|| HideError::NotFound(format!("tool {}", call.tool)))?;
        let spec = tool.spec().clone();
        let predicted = tool
            .simulate(
                &call.args,
                ToolCtx {
                    grant_id: call.capability_grant_id.clone(),
                    deadline_ms: Some(spec.timeout_ms),
                    output_cap_bytes: spec.output_cap_bytes,
                },
            )
            .await
            .unwrap_or_default();
        let target = predicted
            .effects
            .first()
            .map(|effect| effect.target.clone())
            .unwrap_or_else(|| spec.name.clone());
        let request = PermissionRequest {
            capability_kind: spec
                .capabilities_required
                .first()
                .cloned()
                .unwrap_or_else(|| "tool.call".to_string()),
            target,
            risk: if spec.annotations.destructive {
                RiskLevel::High
            } else {
                RiskLevel::Low
            },
            effects: predicted.effects.clone(),
            grant: None,
        };
        let verdict = self.policy.evaluate(&request);
        if verdict.decision != Decision::Allow {
            return Err(HideError::PolicyDenied(verdict.reason));
        }
        if call.x.dry_run {
            return Ok(ToolResult {
                call_id,
                ok: true,
                status: ToolStatus::Ok,
                content: vec![ToolContent::Json {
                    value: serde_json::to_value(&predicted)?,
                }],
                structured_content: Some(serde_json::json!({
                    "dry_run": true,
                    "tool": spec.name,
                    "predicted_effects": predicted,
                })),
                bytes_ref: None,
                exit_code: None,
                effects: predicted,
                provenance: default_tool_provenance(),
                stats: ToolStats {
                    from_dry_run: true,
                    ..ToolStats::default()
                },
                error: None,
            });
        }
        let before = self.observer.as_ref().and_then(|o| o.before(&call));
        let mut result = tool
            .call(
                call.args.clone(),
                ToolCtx {
                    grant_id: call.capability_grant_id.clone(),
                    deadline_ms: Some(spec.timeout_ms),
                    output_cap_bytes: spec.output_cap_bytes,
                },
            )
            .await;
        result.call_id = call_id;
        if let Some(observer) = self.observer.as_ref() {
            observer.after(&call, before, &result).await;
        }
        Ok(result)
    }
}
