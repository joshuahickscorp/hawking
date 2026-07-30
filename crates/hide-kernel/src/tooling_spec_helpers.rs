//! Centralized [`ToolSpec`] builders so the catalog declares schemas concisely
//! and consistently (ch.03 §4.2.1). Annotations follow the catalog table (§4.6):
//! read-only tools are `read_only:true` non-destructive; process/edit tools are
//! destructive.

use hide_core::tool::{ToolAnnotations, ToolSpec};
use serde_json::{json, Value};

/// A read-only filesystem/query tool (auto policy, idempotent).
pub fn read_spec(
    name: &str,
    title: &str,
    description: &str,
    input_schema: Value,
    output_schema: Option<Value>,
    cap_bytes: u64,
) -> ToolSpec {
    ToolSpec {
        name: name.to_string(),
        title: title.to_string(),
        version: "0.1.0".to_string(),
        wire_version: 1,
        description: description.to_string(),
        input_schema,
        output_schema,
        annotations: ToolAnnotations {
            read_only: true,
            destructive: false,
            idempotent: true,
            open_world: false,
        },
        capabilities_required: vec!["fs.read".to_string()],
        output_cap_bytes: cap_bytes,
        timeout_ms: 15_000,
    }
}

/// A filesystem write/edit tool (ask policy, destructive when overwriting).
pub fn write_spec(
    name: &str,
    title: &str,
    description: &str,
    input_schema: Value,
    output_schema: Option<Value>,
) -> ToolSpec {
    ToolSpec {
        name: name.to_string(),
        title: title.to_string(),
        version: "0.1.0".to_string(),
        wire_version: 1,
        description: description.to_string(),
        input_schema,
        output_schema,
        annotations: ToolAnnotations {
            read_only: false,
            destructive: true,
            idempotent: false,
            open_world: false,
        },
        capabilities_required: vec!["fs.write".to_string()],
        output_cap_bytes: 256 * 1024,
        timeout_ms: 15_000,
    }
}

/// A process-execution tool (shell/test/build) — destructive, open-world.
pub fn exec_spec(
    name: &str,
    title: &str,
    description: &str,
    cap_bytes: u64,
    timeout_ms: u64,
) -> ToolSpec {
    ToolSpec {
        name: name.to_string(),
        title: title.to_string(),
        version: "0.1.0".to_string(),
        wire_version: 1,
        description: description.to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "argv": { "type": "array", "items": {"type":"string"},
                          "description": "command + args; argv form avoids shell injection" },
                "cwd": { "type": "string" },
                "env": { "type": "object" }
            },
            "required": ["argv"],
            "additionalProperties": false
        }),
        output_schema: Some(json!({
            "type": "object",
            "properties": {
                "exit_code": {"type":"integer"},
                "stdout": {"type":"string"},
                "stderr": {"type":"string"},
                "stdout_truncated": {"type":"boolean"}
            },
            "required": ["exit_code", "stdout", "stderr"]
        })),
        annotations: ToolAnnotations {
            read_only: false,
            destructive: true,
            idempotent: false,
            open_world: true,
        },
        capabilities_required: vec!["shell.exec".to_string()],
        output_cap_bytes: cap_bytes,
        timeout_ms,
    }
}

/// The `shell.plan` spec — pure, non-effecting describe-only tool.
pub fn plan_spec() -> ToolSpec {
    ToolSpec {
        name: "shell.plan".to_string(),
        title: "Plan shell command".to_string(),
        version: "0.1.0".to_string(),
        wire_version: 1,
        description: "Validate a command and render its sandbox profile without executing."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": { "argv": { "type": "array", "items": {"type":"string"} } },
            "required": ["argv"],
            "additionalProperties": false
        }),
        output_schema: None,
        annotations: ToolAnnotations {
            read_only: true,
            destructive: false,
            idempotent: true,
            open_world: false,
        },
        capabilities_required: vec!["shell.exec".to_string()],
        output_cap_bytes: 64 * 1024,
        timeout_ms: 5_000,
    }
}

/// A git read tool.
pub fn git_read_spec(name: &str, title: &str, description: &str, input_schema: Value) -> ToolSpec {
    let mut s = read_spec(name, title, description, input_schema, None, 256 * 1024);
    s.capabilities_required = vec!["git.read".to_string()];
    s
}

/// A git write tool (ask policy).
pub fn git_write_spec(name: &str, title: &str, description: &str, input_schema: Value) -> ToolSpec {
    ToolSpec {
        name: name.to_string(),
        title: title.to_string(),
        version: "0.1.0".to_string(),
        wire_version: 1,
        description: description.to_string(),
        input_schema,
        output_schema: None,
        annotations: ToolAnnotations {
            read_only: false,
            destructive: false,
            idempotent: false,
            open_world: false,
        },
        capabilities_required: vec!["git.write".to_string()],
        output_cap_bytes: 256 * 1024,
        timeout_ms: 30_000,
    }
}
