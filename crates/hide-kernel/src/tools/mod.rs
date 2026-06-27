use hide_core::tool::{ToolCall, ToolResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdempotencyRecord {
    pub key: String,
    pub call_hash: String,
    pub result_event_seq: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDispatchRecord {
    pub call: ToolCall,
    pub result: Option<ToolResult>,
    pub replayed: bool,
}

pub fn lint_tool_call(call: &ToolCall) -> Result<(), String> {
    if call.tool_name.trim().is_empty() {
        return Err("tool name is empty".to_string());
    }
    if !call.args.is_object() {
        return Err("tool args must be a JSON object".to_string());
    }
    Ok(())
}
