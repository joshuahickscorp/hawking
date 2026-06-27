use hide_core::tool::{ToolResult, ToolSpec, ToolStatus};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerDescriptor {
    pub id: String,
    pub transport: McpTransport,
    pub protocol_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    Stdio { command: String, args: Vec<String> },
    StreamableHttp { endpoint: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub input_schema: Value,
    pub output_schema: Option<Value>,
}

pub fn mcp_tool_to_hide_spec(server_id: &str, tool: McpTool) -> ToolSpec {
    ToolSpec {
        name: format!("mcp.{server_id}.{}", tool.name),
        title: tool.title.unwrap_or_else(|| tool.name.clone()),
        version: "0.1.0".to_string(),
        wire_version: 1,
        description: tool.description.unwrap_or_default(),
        input_schema: tool.input_schema,
        output_schema: tool.output_schema,
        annotations: Default::default(),
        capabilities_required: vec!["mcp.call".to_string()],
        output_cap_bytes: 1024 * 1024,
        timeout_ms: 30_000,
    }
}

pub fn hide_result_to_mcp(result: &ToolResult) -> Value {
    json!({
        "isError": result.status != ToolStatus::Ok,
        "structuredContent": result.structured_content,
        "content": result.content,
    })
}
