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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: ToolCallId,
    pub tool_name: String,
    pub args: Value,
    pub capability_grant_id: Option<GrantId>,
    pub idempotency_key: Option<String>,
    pub dry_run: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: ToolCallId,
    pub status: ToolStatus,
    pub content: Vec<ToolContent>,
    pub structured_content: Option<Value>,
    pub bytes_ref: Option<BlobRef>,
    pub effects: EffectSet,
    pub error: Option<ToolError>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolError {
    pub code: String,
    pub message: String,
    pub recoverable: bool,
}

impl ToolResult {
    pub fn ok(call_id: ToolCallId, structured_content: Option<Value>, effects: EffectSet) -> Self {
        Self {
            call_id,
            status: ToolStatus::Ok,
            content: Vec::new(),
            structured_content,
            bytes_ref: None,
            effects,
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

pub struct ToolDispatcher {
    registry: Arc<ToolRegistry>,
    policy: Arc<dyn PermissionEngine>,
}

impl ToolDispatcher {
    pub fn new(registry: Arc<ToolRegistry>, policy: Arc<dyn PermissionEngine>) -> Self {
        Self { registry, policy }
    }

    pub async fn dispatch(&self, call: ToolCall) -> Result<ToolResult> {
        let call_id = call.id.clone();
        let tool = self
            .registry
            .get(&call.tool_name)
            .ok_or_else(|| HideError::NotFound(format!("tool {}", call.tool_name)))?;
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
        if call.dry_run {
            return Ok(ToolResult {
                call_id,
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
                effects: predicted,
                error: None,
            });
        }
        let mut result = tool
            .call(
                call.args,
                ToolCtx {
                    grant_id: call.capability_grant_id,
                    deadline_ms: Some(spec.timeout_ms),
                    output_cap_bytes: spec.output_cap_bytes,
                },
            )
            .await;
        result.call_id = call_id;
        Ok(result)
    }
}
