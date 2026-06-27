use futures::future::BoxFuture;
use hide_core::tool::{
    Purity, Tool, ToolAnnotations, ToolContent, ToolCtx, ToolError, ToolResult, ToolSpec,
    ToolStatus,
};
use hide_core::types::{Effect, EffectKind, EffectSet, RiskLevel};
use serde_json::{json, Value};
use std::process::{Command, Stdio};

#[derive(Debug, Clone)]
pub struct ShellPlanTool {
    spec: ToolSpec,
}

impl Default for ShellPlanTool {
    fn default() -> Self {
        Self {
            spec: ToolSpec {
                name: "shell.plan".to_string(),
                title: "Plan shell command".to_string(),
                version: "0.1.0".to_string(),
                wire_version: 1,
                description: "Validate and describe a shell command without executing it."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "argv": { "type": "array", "items": {"type":"string"} } },
                    "required": ["argv"],
                    "additionalProperties": false
                }),
                output_schema: None,
                annotations: ToolAnnotations {
                    read_only: false,
                    destructive: true,
                    idempotent: false,
                    open_world: false,
                },
                capabilities_required: vec!["process.exec".to_string()],
                output_cap_bytes: 64 * 1024,
                timeout_ms: 5_000,
            },
        }
    }
}

impl Tool for ShellPlanTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
        Box::pin(async move {
            let argv: Vec<String> = args
                .get("argv")
                .and_then(|v| v.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|v| v.as_str().map(ToOwned::to_owned))
                        .collect()
                })
                .unwrap_or_default();
            ToolResult {
                call_id: hide_core::ids::ToolCallId::new(),
                status: ToolStatus::Ok,
                content: vec![ToolContent::Text {
                    text: format!("planned command: {:?}", argv),
                }],
                structured_content: Some(json!({
                    "argv": argv,
                    "executed": false,
                    "note": "shell.plan is non-effecting; shell.run must be sandbox-wired separately"
                })),
                bytes_ref: None,
                effects: EffectSet::default(),
                error: None,
            }
        })
    }

    fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
        Box::pin(async move {
            let target = args
                .get("argv")
                .map(|v| v.to_string())
                .unwrap_or_else(|| "[]".to_string());
            Some(EffectSet {
                effects: vec![Effect {
                    kind: EffectKind::Execute,
                    target,
                    bytes_hash: None,
                    risk: RiskLevel::High,
                    metadata: Default::default(),
                }],
            })
        })
    }

    fn purity(&self) -> Purity {
        Purity::Pure
    }
}

#[derive(Debug, Clone)]
pub struct ShellRunTool {
    spec: ToolSpec,
}

impl Default for ShellRunTool {
    fn default() -> Self {
        Self {
            spec: ToolSpec {
                name: "shell.run".to_string(),
                title: "Run shell command".to_string(),
                version: "0.1.0".to_string(),
                wire_version: 1,
                description:
                    "Run an already-authorized non-interactive command with bounded output."
                        .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "argv": { "type": "array", "items": {"type":"string"} },
                        "cwd": { "type": "string" },
                        "env": { "type": "object" }
                    },
                    "required": ["argv"],
                    "additionalProperties": false
                }),
                output_schema: Some(json!({
                    "type": "object",
                    "properties": {
                        "status": {"type":"integer"},
                        "stdout": {"type":"string"},
                        "stderr": {"type":"string"},
                        "truncated": {"type":"boolean"}
                    },
                    "required": ["status", "stdout", "stderr", "truncated"]
                })),
                annotations: ToolAnnotations {
                    read_only: false,
                    destructive: true,
                    idempotent: false,
                    open_world: false,
                },
                capabilities_required: vec!["process.exec".to_string()],
                output_cap_bytes: 256 * 1024,
                timeout_ms: 30_000,
            },
        }
    }
}

impl Tool for ShellRunTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, args: Value, ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
        Box::pin(async move {
            let argv = parse_argv(&args);
            if argv.is_empty() {
                return shell_protocol_error("argv must contain at least one command");
            }
            let mut command = Command::new(&argv[0]);
            command.args(&argv[1..]);
            if let Some(cwd) = args.get("cwd").and_then(|v| v.as_str()) {
                command.current_dir(cwd);
            }
            if let Some(env) = args.get("env").and_then(|v| v.as_object()) {
                for (key, value) in env {
                    if let Some(value) = value.as_str() {
                        command.env(key, value);
                    }
                }
            }
            command.stdin(Stdio::null());
            match command.output() {
                Ok(output) => {
                    let cap = ctx.output_cap_bytes as usize;
                    let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    let mut stderr = String::from_utf8_lossy(&output.stderr).to_string();
                    let truncated = stdout.len() + stderr.len() > cap;
                    if truncated {
                        let half = cap / 2;
                        stdout.truncate(stdout.len().min(half));
                        stderr.truncate(stderr.len().min(half));
                    }
                    let status = output.status.code().unwrap_or(-1);
                    ToolResult {
                        call_id: hide_core::ids::ToolCallId::new(),
                        status: if output.status.success() {
                            ToolStatus::Ok
                        } else {
                            ToolStatus::ToolError
                        },
                        content: vec![ToolContent::Text {
                            text: format!("status={status}\n{stdout}{stderr}"),
                        }],
                        structured_content: Some(json!({
                            "status": status,
                            "stdout": stdout,
                            "stderr": stderr,
                            "truncated": truncated
                        })),
                        bytes_ref: None,
                        effects: EffectSet::default(),
                        error: if output.status.success() {
                            None
                        } else {
                            Some(ToolError {
                                code: "nonzero_exit".to_string(),
                                message: format!("command exited with status {status}"),
                                recoverable: true,
                            })
                        },
                    }
                }
                Err(err) => ToolResult {
                    call_id: hide_core::ids::ToolCallId::new(),
                    status: ToolStatus::ToolError,
                    content: Vec::new(),
                    structured_content: None,
                    bytes_ref: None,
                    effects: EffectSet::default(),
                    error: Some(ToolError {
                        code: "spawn_failed".to_string(),
                        message: err.to_string(),
                        recoverable: true,
                    }),
                },
            }
        })
    }

    fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
        Box::pin(async move {
            Some(EffectSet {
                effects: vec![Effect {
                    kind: EffectKind::Execute,
                    target: parse_argv(args).join(" "),
                    bytes_hash: None,
                    risk: RiskLevel::High,
                    metadata: Default::default(),
                }],
            })
        })
    }

    fn purity(&self) -> Purity {
        Purity::Impure
    }
}

fn parse_argv(args: &Value) -> Vec<String> {
    args.get("argv")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|v| v.as_str().map(ToOwned::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

fn shell_protocol_error(message: impl Into<String>) -> ToolResult {
    ToolResult {
        call_id: hide_core::ids::ToolCallId::new(),
        status: ToolStatus::ProtocolError,
        content: Vec::new(),
        structured_content: None,
        bytes_ref: None,
        effects: EffectSet::default(),
        error: Some(ToolError {
            code: "bad_args".to_string(),
            message: message.into(),
            recoverable: true,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::permission::{PermissionPolicy, StaticPermissionEngine};
    use hide_core::tool::{ToolCall, ToolDispatcher, ToolRegistry};
    use std::sync::Arc;

    #[tokio::test]
    async fn shell_run_executes_bounded_command() {
        let registry = Arc::new(ToolRegistry::default());
        registry.register(ShellRunTool::default());
        let dispatcher = ToolDispatcher::new(
            registry,
            Arc::new(StaticPermissionEngine::new(PermissionPolicy {
                default_decision: hide_core::types::Decision::Allow,
                rules: Vec::new(),
                risk_gates: Vec::new(),
            })),
        );
        let result = dispatcher
            .dispatch(ToolCall {
                id: hide_core::ids::ToolCallId::new(),
                tool_name: "shell.run".to_string(),
                args: json!({ "argv": ["printf", "hello"] }),
                capability_grant_id: None,
                idempotency_key: None,
                dry_run: false,
            })
            .await
            .unwrap();
        assert_eq!(result.status, ToolStatus::Ok);
        assert_eq!(result.structured_content.unwrap()["stdout"], "hello");
    }
}
