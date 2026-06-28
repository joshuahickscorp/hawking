use futures::future::BoxFuture;
use hide_core::tool::{
    Purity, Tool, ToolAnnotations, ToolContent, ToolCtx, ToolError, ToolResult, ToolSpec,
    ToolStatus,
};
use serde_json::{json, Value};
use std::process::{Command, Stdio};

#[derive(Debug, Clone)]
pub struct GitStatusTool {
    spec: ToolSpec,
}

impl Default for GitStatusTool {
    fn default() -> Self {
        Self {
            spec: ToolSpec {
                name: "git.status".to_string(),
                title: "Git status snapshot".to_string(),
                version: "0.1.0".to_string(),
                wire_version: 1,
                description: "Run `git status --short --branch` for an authorized workspace."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "cwd": { "type": "string" } },
                    "required": [],
                    "additionalProperties": false
                }),
                output_schema: None,
                annotations: ToolAnnotations {
                    read_only: true,
                    destructive: false,
                    idempotent: true,
                    open_world: false,
                },
                capabilities_required: vec!["fs.read".to_string()],
                output_cap_bytes: 64 * 1024,
                timeout_ms: 5_000,
            },
        }
    }
}

impl Tool for GitStatusTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, args: Value, ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
        Box::pin(async move {
            let cwd = args.get("cwd").and_then(|v| v.as_str()).unwrap_or(".");
            match Command::new("git")
                .args(["status", "--short", "--branch"])
                .current_dir(cwd)
                .stdin(Stdio::null())
                .output()
            {
                Ok(output) => {
                    let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    let mut stderr = String::from_utf8_lossy(&output.stderr).to_string();
                    let cap = ctx.output_cap_bytes as usize;
                    let truncated = stdout.len() + stderr.len() > cap;
                    if truncated {
                        let half = cap / 2;
                        stdout.truncate(stdout.len().min(half));
                        stderr.truncate(stderr.len().min(half));
                    }
                    let status = output.status.code().unwrap_or(-1);
                    ToolResult {
                        call_id: hide_core::ids::ToolCallId::new(),
                        ok: output.status.success(),
                        status: if output.status.success() {
                            ToolStatus::Ok
                        } else {
                            ToolStatus::ToolError
                        },
                        content: vec![ToolContent::Text {
                            text: stdout.clone(),
                        }],
                        structured_content: Some(json!({
                            "cwd": cwd,
                            "status": status,
                            "stdout": stdout,
                            "stderr": stderr,
                            "truncated": truncated
                        })),
                        bytes_ref: None,
                        exit_code: Some(status),
                        effects: Default::default(),
                        provenance: "tool-output".to_string(),
                        stats: Default::default(),
                        error: if output.status.success() {
                            None
                        } else {
                            // NOTE: non-zero exit is EXEC_NONZERO data per ch.03
                            // §4.2.3; the ok:true rewiring is a hide-tools item, not
                            // this contract migration. Behavior preserved.
                            Some(ToolError::new(
                                "EXEC_NONZERO",
                                format!("git exited with status {status}"),
                                true,
                            ))
                        },
                    }
                }
                Err(err) => ToolResult {
                    call_id: hide_core::ids::ToolCallId::new(),
                    ok: false,
                    status: ToolStatus::ToolError,
                    content: Vec::new(),
                    structured_content: None,
                    bytes_ref: None,
                    exit_code: None,
                    effects: Default::default(),
                    provenance: "tool-output".to_string(),
                    stats: Default::default(),
                    error: Some(ToolError::new("TOOL_FAULT", err.to_string(), true)),
                },
            }
        })
    }

    fn purity(&self) -> Purity {
        Purity::PureFs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::permission::{PermissionPolicy, StaticPermissionEngine};
    use hide_core::tool::{ToolCall, ToolDispatcher, ToolRegistry};
    use std::sync::Arc;

    #[tokio::test]
    async fn git_status_runs_in_current_repo() {
        let registry = Arc::new(ToolRegistry::default());
        registry.register(GitStatusTool::default());
        let dispatcher = ToolDispatcher::new(
            registry,
            Arc::new(StaticPermissionEngine::new(PermissionPolicy {
                default_decision: hide_core::types::Decision::Allow,
                rules: Vec::new(),
                risk_gates: Vec::new(),
            })),
        );
        let result = dispatcher
            .dispatch(ToolCall::new("git.status", json!({ "cwd": "." })))
            .await
            .unwrap();
        assert_eq!(result.status, ToolStatus::Ok);
        assert!(result.structured_content.unwrap()["stdout"]
            .as_str()
            .unwrap()
            .contains("##"));
    }
}
