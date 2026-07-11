//! Native tool calling on the OpenAI-compatible serve API (Phase 1a of
//! `docs/plans/agentic_tool_system_2026_07_11.md`).
//!
//! Two pure, self-contained pieces the chat endpoint uses:
//! * [`render_tools_preamble`] turns the request's `tools` array into a system
//!   preamble in the Hermes / Qwen2.5 convention (a `<tools>` block plus the
//!   instruction to emit `<tool_call>{...}</tool_call>`), so a local chat model
//!   trained on that format produces callable output.
//! * [`extract_tool_calls`] turns the model's completion text back into the
//!   OpenAI `tool_calls` response shape (arguments as a JSON-encoded string).
//!
//! The serve engine lives in its own dependency universe (`hawking-core`), so this
//! deliberately does not reuse the agent-side parser in `hide-kernel`; it mirrors
//! its format handling (tagged blocks first, then any balanced JSON span, so a
//! call embedded in prose or after a bracket is still recovered) but stays a thin
//! API-shaping utility.

use serde_json::Value;

/// One extracted call in OpenAI response shape: `arguments` is a JSON-encoded
/// string, matching `chat.completion` `tool_calls[].function.arguments`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl ExtractedToolCall {
    /// The `tool_calls[]` entry for a chat-completion message.
    pub fn to_openai(&self) -> Value {
        serde_json::json!({
            "id": self.id,
            "type": "function",
            "function": { "name": self.name, "arguments": self.arguments }
        })
    }
}

/// Render the request `tools` (OpenAI function specs) into a system preamble. An
/// empty list yields an empty string (no-op). The format is the one Qwen2.5 /
/// Hermes models are trained on and is a reasonable default for others.
pub fn render_tools_preamble(tools: &[Value]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut s = String::new();
    s.push_str(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
    );
    for tool in tools {
        // Accept either a bare function object or the OpenAI {"type":"function",
        // "function":{...}} envelope.
        let func = tool.get("function").unwrap_or(tool);
        s.push_str(&func.to_string());
        s.push('\n');
    }
    s.push_str(
        "</tools>\n\nFor each function call, return a json object with the function name and \
         arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n",
    );
    s
}

/// Extract every tool call from a completion, in document order. Empty when the
/// model produced a plain text answer.
pub fn extract_tool_calls(completion: &str) -> Vec<ExtractedToolCall> {
    let mut raw: Vec<(String, Value)> = Vec::new();

    // Hermes / Qwen `<tool_call>...</tool_call>` blocks take precedence.
    let tagged = tagged_blocks(completion);
    if !tagged.is_empty() {
        raw = tagged;
    } else {
        // Fall back to any balanced JSON span (object, array, or an OpenAI
        // {"tool_calls":[...]} envelope).
        for span in all_json_spans(completion) {
            if let Ok(value) = serde_json::from_str::<Value>(&span) {
                let calls = calls_from_value(&value);
                if !calls.is_empty() {
                    raw = calls;
                    break;
                }
            }
        }
    }

    raw.into_iter()
        .enumerate()
        .map(|(i, (name, args))| ExtractedToolCall {
            id: format!("call_{i}"),
            name,
            // OpenAI carries arguments as a JSON-encoded string.
            arguments: args.to_string(),
        })
        .collect()
}

/// Whether the completion contains a recognizable tool call.
pub fn has_tool_call(completion: &str) -> bool {
    completion.contains("<tool_call>") || !extract_tool_calls(completion).is_empty()
}

// ---------------------------------------------------------------------------
// internals (mirror the agent-side parser's robustness)
// ---------------------------------------------------------------------------

fn tagged_blocks(text: &str) -> Vec<(String, Value)> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        let after = &rest[start + OPEN.len()..];
        let Some(end) = after.find(CLOSE) else { break };
        if let Ok(value) = serde_json::from_str::<Value>(after[..end].trim()) {
            out.extend(calls_from_value(&value));
        }
        rest = &after[end + CLOSE.len()..];
    }
    out
}

fn calls_from_value(value: &Value) -> Vec<(String, Value)> {
    match value {
        Value::Array(items) => items.iter().flat_map(calls_from_value).collect(),
        Value::Object(obj) => {
            if let Some(Value::Array(list)) = obj.get("tool_calls") {
                return list.iter().flat_map(calls_from_value).collect();
            }
            single(value).into_iter().collect()
        }
        _ => Vec::new(),
    }
}

fn single(value: &Value) -> Option<(String, Value)> {
    let obj = value.as_object()?;
    let (name_src, args_src) = if let Some(func) = obj.get("function").and_then(|f| f.as_object()) {
        (func.get("name"), func.get("arguments").or_else(|| func.get("parameters")))
    } else {
        (
            obj.get("name").or_else(|| obj.get("tool")),
            obj.get("arguments").or_else(|| obj.get("args")).or_else(|| obj.get("parameters")),
        )
    };
    let name = name_src?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let args = match args_src {
        None | Some(Value::Null) => serde_json::json!({}),
        Some(Value::Object(o)) => Value::Object(o.clone()),
        Some(Value::String(s)) => serde_json::from_str::<Value>(s).unwrap_or_else(|_| serde_json::json!({ "input": s })),
        Some(other) => serde_json::json!({ "value": other.clone() }),
    };
    Some((name, args))
}

fn all_json_spans(s: &str) -> Vec<String> {
    let bytes = s.as_bytes();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' || bytes[i] == b'[' {
            match balanced_end(bytes, i) {
                Some(end) => {
                    spans.push(s[i..=end].to_string());
                    i = end + 1;
                    continue;
                }
                None => break,
            }
        }
        i += 1;
    }
    spans
}

fn balanced_end(bytes: &[u8], start: usize) -> Option<usize> {
    let open = bytes[start];
    let close = if open == b'{' { b'}' } else { b']' };
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            x if x == open => depth += 1,
            x if x == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn preamble_lists_tools_and_is_empty_when_none() {
        assert_eq!(render_tools_preamble(&[]), "");
        let tools = vec![json!({
            "type": "function",
            "function": { "name": "get_weather", "parameters": { "type": "object" } }
        })];
        let p = render_tools_preamble(&tools);
        assert!(p.contains("<tools>"));
        assert!(p.contains("get_weather"));
        assert!(p.contains("<tool_call>"));
        // The OpenAI envelope is unwrapped to the bare function spec.
        assert!(!p.contains("\"type\":\"function\""));
    }

    #[test]
    fn extracts_hermes_block_to_openai_shape() {
        let calls = extract_tool_calls(
            "I'll check.\n<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"NYC\"}}</tool_call>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].id, "call_0");
        // arguments is a JSON-encoded string.
        assert_eq!(calls[0].arguments, "{\"city\":\"NYC\"}");
        let oa = calls[0].to_openai();
        assert_eq!(oa["function"]["name"], "get_weather");
    }

    #[test]
    fn extracts_parallel_calls() {
        let calls = extract_tool_calls(
            "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\
             <tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>",
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[1].id, "call_1");
    }

    #[test]
    fn extracts_bare_call_after_bracket() {
        // Same robustness the agent-side parser has: a leading [...] must not shadow.
        let calls = extract_tool_calls("see [1] {\"name\":\"a\",\"arguments\":{\"x\":1}}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "a");
    }

    #[test]
    fn plain_text_yields_no_calls() {
        assert!(extract_tool_calls("just a normal answer").is_empty());
        assert!(!has_tool_call("just a normal answer"));
    }
}
