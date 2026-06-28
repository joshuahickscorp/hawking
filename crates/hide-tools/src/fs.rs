use futures::future::BoxFuture;
use hide_core::tool::{
    Purity, Tool, ToolAnnotations, ToolContent, ToolCtx, ToolError, ToolResult, ToolSpec,
    ToolStatus,
};
use hide_core::types::{Effect, EffectKind, EffectSet, RiskLevel};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct FsReadTool {
    spec: ToolSpec,
}

impl Default for FsReadTool {
    fn default() -> Self {
        Self {
            spec: ToolSpec {
                name: "fs.read".to_string(),
                title: "Read file".to_string(),
                version: "0.1.0".to_string(),
                wire_version: 1,
                description: "Read a UTF-8 file from an already-authorized filesystem scope."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"],
                    "additionalProperties": false
                }),
                output_schema: Some(json!({
                    "type": "object",
                    "properties": { "path": {"type":"string"}, "content": {"type":"string"} },
                    "required": ["path", "content"]
                })),
                annotations: ToolAnnotations {
                    read_only: true,
                    destructive: false,
                    idempotent: true,
                    open_world: false,
                },
                capabilities_required: vec!["fs.read".to_string()],
                output_cap_bytes: 1024 * 1024,
                timeout_ms: 15_000,
            },
        }
    }
}

impl Tool for FsReadTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, args: Value, ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
        Box::pin(async move {
            let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
                return protocol_error("missing string arg: path");
            };
            match std::fs::read(path) {
                Ok(bytes) if bytes.len() as u64 <= ctx.output_cap_bytes => {
                    match String::from_utf8(bytes) {
                        Ok(content) => ToolResult {
                            call_id: hide_core::ids::ToolCallId::new(),
                            ok: true,
                            status: ToolStatus::Ok,
                            content: vec![ToolContent::Text {
                                text: content.clone(),
                            }],
                            structured_content: Some(json!({ "path": path, "content": content })),
                            bytes_ref: None,
                            exit_code: None,
                            effects: EffectSet::default(),
                            provenance: "tool-output".to_string(),
                            stats: Default::default(),
                            error: None,
                        },
                        Err(err) => tool_error("non_utf8", err.to_string()),
                    }
                }
                Ok(bytes) => tool_error(
                    "output_cap_exceeded",
                    format!("{} bytes exceeds cap {}", bytes.len(), ctx.output_cap_bytes),
                ),
                Err(err) => tool_error("io", err.to_string()),
            }
        })
    }

    fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
        Box::pin(async move {
            args.get("path")
                .and_then(|v| v.as_str())
                .map(|path| EffectSet {
                    effects: vec![Effect {
                        kind: EffectKind::Read,
                        target: path.to_string(),
                        bytes_hash: None,
                        risk: RiskLevel::Low,
                        metadata: Default::default(),
                    }],
                })
        })
    }

    fn purity(&self) -> Purity {
        Purity::PureFs
    }
}

#[derive(Debug, Clone)]
pub struct FsListTool {
    spec: ToolSpec,
}

impl Default for FsListTool {
    fn default() -> Self {
        Self {
            spec: ToolSpec {
                name: "fs.list".to_string(),
                title: "List directory".to_string(),
                version: "0.1.0".to_string(),
                wire_version: 1,
                description: "List a directory from an already-authorized filesystem scope."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"],
                    "additionalProperties": false
                }),
                output_schema: Some(json!({
                    "type": "object",
                    "properties": { "path": {"type":"string"}, "entries": {"type":"array"} },
                    "required": ["path", "entries"]
                })),
                annotations: ToolAnnotations {
                    read_only: true,
                    destructive: false,
                    idempotent: true,
                    open_world: false,
                },
                capabilities_required: vec!["fs.read".to_string()],
                output_cap_bytes: 1024 * 1024,
                timeout_ms: 15_000,
            },
        }
    }
}

impl Tool for FsListTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
        Box::pin(async move {
            let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
                return protocol_error("missing string arg: path");
            };
            let mut entries = Vec::new();
            match std::fs::read_dir(PathBuf::from(path)) {
                Ok(read_dir) => {
                    for entry in read_dir.flatten() {
                        entries.push(entry.file_name().to_string_lossy().to_string());
                    }
                    entries.sort();
                    ToolResult {
                        call_id: hide_core::ids::ToolCallId::new(),
                        ok: true,
                        status: ToolStatus::Ok,
                        content: vec![ToolContent::Json {
                            value: json!(entries),
                        }],
                        structured_content: Some(json!({ "path": path, "entries": entries })),
                        bytes_ref: None,
                        exit_code: None,
                        effects: EffectSet::default(),
                        provenance: "tool-output".to_string(),
                        stats: Default::default(),
                        error: None,
                    }
                }
                Err(err) => tool_error("io", err.to_string()),
            }
        })
    }

    fn purity(&self) -> Purity {
        Purity::PureFs
    }
}

#[derive(Debug, Clone)]
pub struct FsWriteTool {
    spec: ToolSpec,
}

impl Default for FsWriteTool {
    fn default() -> Self {
        Self {
            spec: ToolSpec {
                name: "fs.write".to_string(),
                title: "Write file".to_string(),
                version: "0.1.0".to_string(),
                wire_version: 1,
                description: "Write UTF-8 content to an already-authorized filesystem scope."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" },
                        "create_dirs": { "type": "boolean", "default": false }
                    },
                    "required": ["path", "content"],
                    "additionalProperties": false
                }),
                output_schema: Some(json!({
                    "type": "object",
                    "properties": {
                        "path": {"type":"string"},
                        "bytes": {"type":"integer"}
                    },
                    "required": ["path", "bytes"]
                })),
                annotations: ToolAnnotations {
                    read_only: false,
                    destructive: true,
                    idempotent: true,
                    open_world: false,
                },
                capabilities_required: vec!["fs.write".to_string()],
                output_cap_bytes: 64 * 1024,
                timeout_ms: 15_000,
            },
        }
    }
}

impl Tool for FsWriteTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
        Box::pin(async move {
            let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
                return protocol_error("missing string arg: path");
            };
            let Some(content) = args.get("content").and_then(|v| v.as_str()) else {
                return protocol_error("missing string arg: content");
            };
            let create_dirs = args
                .get("create_dirs")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let path_buf = PathBuf::from(path);
            if create_dirs {
                if let Some(parent) = path_buf.parent() {
                    if !parent.as_os_str().is_empty() {
                        if let Err(err) = std::fs::create_dir_all(parent) {
                            return tool_error("io", err.to_string());
                        }
                    }
                }
            }
            match std::fs::write(&path_buf, content.as_bytes()) {
                Ok(()) => {
                    let effects = EffectSet {
                        effects: vec![write_effect(path, content.len(), create_dirs)],
                    };
                    ToolResult {
                        call_id: hide_core::ids::ToolCallId::new(),
                        ok: true,
                        status: ToolStatus::Ok,
                        content: vec![ToolContent::Json {
                            value: json!({ "path": path, "bytes": content.len() }),
                        }],
                        structured_content: Some(json!({
                            "path": path,
                            "bytes": content.len()
                        })),
                        bytes_ref: None,
                        exit_code: None,
                        effects,
                        provenance: "tool-output".to_string(),
                        stats: Default::default(),
                        error: None,
                    }
                }
                Err(err) => tool_error("io", err.to_string()),
            }
        })
    }

    fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
        Box::pin(async move {
            let path = args.get("path").and_then(|v| v.as_str())?;
            let content = args.get("content").and_then(|v| v.as_str())?;
            let create_dirs = args
                .get("create_dirs")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            Some(EffectSet {
                effects: vec![write_effect(path, content.len(), create_dirs)],
            })
        })
    }

    fn purity(&self) -> Purity {
        Purity::PureFs
    }
}

fn write_effect(path: &str, bytes: usize, create_dirs: bool) -> Effect {
    let mut metadata = BTreeMap::new();
    metadata.insert("bytes".to_string(), bytes.to_string());
    metadata.insert("create_dirs".to_string(), create_dirs.to_string());
    Effect {
        kind: EffectKind::Write,
        target: path.to_string(),
        bytes_hash: None,
        risk: RiskLevel::High,
        metadata,
    }
}

fn protocol_error(message: impl Into<String>) -> ToolResult {
    ToolResult {
        call_id: hide_core::ids::ToolCallId::new(),
        ok: false,
        status: ToolStatus::ProtocolError,
        content: Vec::new(),
        structured_content: None,
        bytes_ref: None,
        exit_code: None,
        effects: EffectSet::default(),
        provenance: "tool-output".to_string(),
        stats: Default::default(),
        error: Some(ToolError::new("ARG_INVALID", message, true)),
    }
}

fn tool_error(code: impl Into<String>, message: impl Into<String>) -> ToolResult {
    ToolResult {
        call_id: hide_core::ids::ToolCallId::new(),
        ok: false,
        status: ToolStatus::ToolError,
        content: Vec::new(),
        structured_content: None,
        bytes_ref: None,
        exit_code: None,
        effects: EffectSet::default(),
        provenance: "tool-output".to_string(),
        stats: Default::default(),
        error: Some(ToolError::new(code, message, true)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::permission::{PermissionPolicy, StaticPermissionEngine};
    use hide_core::tool::{ToolCall, ToolDispatcher, ToolRegistry};
    use serde_json::json;
    use std::sync::Arc;

    #[tokio::test]
    async fn fs_read_tool_reads_utf8_file_through_dispatcher() {
        let dir = std::env::temp_dir().join(format!("hide_tools_{}", hide_core::ids::now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("hello.txt");
        std::fs::write(&file, "hello").unwrap();

        let registry = Arc::new(ToolRegistry::default());
        registry.register(FsReadTool::default());
        let dispatcher = ToolDispatcher::new(
            registry,
            Arc::new(StaticPermissionEngine::new(PermissionPolicy {
                default_decision: hide_core::types::Decision::Allow,
                rules: Vec::new(),
                risk_gates: Vec::new(),
            })),
        );
        let result = dispatcher
            .dispatch(ToolCall::new(
                "fs.read",
                json!({ "path": file.to_string_lossy() }),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolStatus::Ok);
        assert!(result.structured_content.unwrap()["content"]
            .as_str()
            .unwrap()
            .contains("hello"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn fs_write_tool_writes_after_policy_prediction() {
        let dir =
            std::env::temp_dir().join(format!("hide_tools_write_{}", hide_core::ids::now_ms()));
        let file = dir.join("nested").join("hello.txt");

        let registry = Arc::new(ToolRegistry::default());
        registry.register(FsWriteTool::default());
        let dispatcher = ToolDispatcher::new(
            registry,
            Arc::new(StaticPermissionEngine::new(PermissionPolicy {
                default_decision: hide_core::types::Decision::Allow,
                rules: Vec::new(),
                risk_gates: Vec::new(),
            })),
        );
        let result = dispatcher
            .dispatch(ToolCall::new(
                "fs.write",
                json!({
                    "path": file.to_string_lossy(),
                    "content": "hello write",
                    "create_dirs": true
                }),
            ))
            .await
            .unwrap();

        assert_eq!(result.status, ToolStatus::Ok);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hello write");
        assert_eq!(result.effects.effects[0].kind, EffectKind::Write);
        let _ = std::fs::remove_dir_all(dir);
    }
}
