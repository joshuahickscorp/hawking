//! Tool-call parser: turn model output text into structured tool calls.
//!
//! This is the keystone the agentic loop was missing (see
//! `docs/plans/agentic_tool_system_2026_07_11.md`, Phase 0). Local models emit
//! tool calls as *text*, not as a typed API field, so the harness must extract
//! them. The parser is deliberately tolerant: real models wrap calls in prose,
//! pick one of several community formats, and occasionally encode the arguments
//! as a JSON string rather than an object. We accept all of the common shapes
//! and skip anything unparseable rather than erroring the whole turn.
//!
//! Formats accepted, in priority order:
//! 1. Hermes / Qwen style: `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
//!    (one block per call; multiple blocks = parallel calls).
//! 2. OpenAI style: a JSON object with a top-level `tool_calls` array, each entry
//!    `{"id"?, "type":"function", "function":{"name","arguments"}}` where
//!    `arguments` is a JSON-encoded string.
//! 3. Fenced JSON: a ```json ... ``` block whose object is tool-call-shaped.
//! 4. Bare JSON: the trimmed text parses as a tool-call object, or an array of them.
//!
//! For each object we accept the name under `name` / `tool` / `function.name`,
//! and the arguments under `arguments` / `args` / `parameters` /
//! `function.arguments` (a string value is re-parsed as JSON; if that fails it is
//! kept as `{"input": "<string>"}` so nothing is silently dropped).

use hide_core::tool::ToolCall;
use serde_json::Value;

/// One tool call extracted from model output, before it becomes a `ToolCall`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    /// The registry tool name the model asked for (e.g. `fs.read`).
    pub name: String,
    /// The arguments object. Always a JSON object (possibly empty).
    pub arguments: Value,
    /// The model-supplied call id, when the format carried one (OpenAI).
    pub id: Option<String>,
}

impl ParsedToolCall {
    /// Convert into a dispatchable `ToolCall` (fresh call id, default directives).
    pub fn into_tool_call(self) -> ToolCall {
        ToolCall::new(self.name, self.arguments)
    }
}

/// Parse every tool call found in `text`. Returns them in document order.
/// Never errors: unparseable candidates are skipped. An empty result means the
/// model produced no recognizable tool call (a plain text turn).
pub fn parse_tool_calls(text: &str) -> Vec<ParsedToolCall> {
    // 1. Hermes / Qwen `<tool_call>...</tool_call>` blocks take precedence: they
    //    are unambiguous and the format most local chat models are trained on.
    let tagged = parse_tagged_blocks(text);
    if !tagged.is_empty() {
        return tagged;
    }

    // 2/3/4: fall back to JSON parsing over fenced or bare content.
    for candidate in json_candidates(text) {
        if let Ok(value) = serde_json::from_str::<Value>(&candidate) {
            let calls = calls_from_value(&value);
            if !calls.is_empty() {
                return calls;
            }
        }
    }
    Vec::new()
}

/// Whether the text contains at least one recognizable tool call. Cheap enough
/// for the decode loop to poll as tokens stream in.
pub fn has_tool_call(text: &str) -> bool {
    text.contains("<tool_call>") || !parse_tool_calls(text).is_empty()
}

// ---------------------------------------------------------------------------
// tagged-block extraction
// ---------------------------------------------------------------------------

fn parse_tagged_blocks(text: &str) -> Vec<ParsedToolCall> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        let after = &rest[start + OPEN.len()..];
        let Some(end) = after.find(CLOSE) else {
            break;
        };
        let inner = after[..end].trim();
        if let Ok(value) = serde_json::from_str::<Value>(inner) {
            out.extend(calls_from_value(&value));
        }
        rest = &after[end + CLOSE.len()..];
    }
    out
}

// ---------------------------------------------------------------------------
// JSON candidate extraction (fenced blocks, then the whole trimmed text)
// ---------------------------------------------------------------------------

fn json_candidates(text: &str) -> Vec<String> {
    let mut candidates = Vec::new();

    // Fenced code blocks ```lang\n...\n``` (lang optional). We keep only the
    // inner body, which is where a JSON tool call would live.
    let mut rest = text;
    while let Some(open) = rest.find("```") {
        let after = &rest[open + 3..];
        let Some(close) = after.find("```") else {
            break;
        };
        let block = &after[..close];
        // Drop an optional language tag on the first line (```json).
        let body = match block.split_once('\n') {
            Some((first, tail)) if !first.trim().is_empty() && !first.contains('{') => tail,
            _ => block,
        };
        candidates.push(body.trim().to_string());
        rest = &after[close + 3..];
    }

    // The whole trimmed text, then EVERY balanced {...} / [...] span within it in
    // document order, so a call embedded in prose ("I'll read it: {...}") is still
    // recoverable even when a bracket span (a markdown link, a citation like [1],
    // a list) precedes the real object. The parser tries each candidate until one
    // yields a tool call, and a non-call span (e.g. "[1]") simply yields none and
    // is skipped, so the following object span is still reached.
    let trimmed = text.trim();
    candidates.push(trimmed.to_string());
    candidates.extend(all_json_spans(trimmed));
    candidates
}

/// Every top-level balanced `{...}` / `[...]` span in `s`, in document order,
/// respecting string literals and escapes so a brace inside a string does not
/// close the span early. Non-overlapping: after a span closes, scanning resumes
/// past its end. A span that never balances stops the scan (nothing after it can
/// close it).
fn all_json_spans(s: &str) -> Vec<String> {
    let bytes = s.as_bytes();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' || bytes[i] == b'[' {
            match balanced_span_end(bytes, i) {
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

/// The byte index of the closing delimiter that balances the opener at `start`,
/// or `None` if it never closes. Tracks only the opener's own delimiter type and
/// skips string literals.
fn balanced_span_end(bytes: &[u8], start: usize) -> Option<usize> {
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

// ---------------------------------------------------------------------------
// value -> ParsedToolCall(s)
// ---------------------------------------------------------------------------

/// Extract every tool call reachable from a parsed JSON value. Handles: a single
/// call object, an array of call objects, and an OpenAI `{"tool_calls":[...]}`
/// envelope.
fn calls_from_value(value: &Value) -> Vec<ParsedToolCall> {
    match value {
        Value::Array(items) => items.iter().flat_map(calls_from_value).collect(),
        Value::Object(obj) => {
            if let Some(Value::Array(list)) = obj.get("tool_calls") {
                return list.iter().flat_map(calls_from_value).collect();
            }
            single_call(value).into_iter().collect()
        }
        _ => Vec::new(),
    }
}

/// Parse one object into a `ParsedToolCall`, if it is tool-call-shaped.
fn single_call(value: &Value) -> Option<ParsedToolCall> {
    let obj = value.as_object()?;

    // OpenAI nests name/arguments under `function`.
    let (name_src, args_src, id) =
        if let Some(func) = obj.get("function").and_then(|f| f.as_object()) {
            let id = obj.get("id").and_then(|v| v.as_str()).map(str::to_string);
            (
                func.get("name"),
                func.get("arguments").or_else(|| func.get("parameters")),
                id,
            )
        } else {
            let id = obj.get("id").and_then(|v| v.as_str()).map(str::to_string);
            (
                obj.get("name").or_else(|| obj.get("tool")),
                obj.get("arguments")
                    .or_else(|| obj.get("args"))
                    .or_else(|| obj.get("parameters")),
                id,
            )
        };

    let name = name_src?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let arguments = normalize_args(args_src);
    Some(ParsedToolCall {
        name,
        arguments,
        id,
    })
}

/// Coerce whatever sat in the arguments slot into a JSON object. A missing slot
/// becomes `{}`; a JSON-encoded string is re-parsed; a string that is not JSON is
/// wrapped as `{"input": ...}` so it is never silently lost; a non-object JSON
/// value is wrapped under `{"value": ...}`.
fn normalize_args(src: Option<&Value>) -> Value {
    match src {
        None | Some(Value::Null) => Value::Object(Default::default()),
        Some(Value::Object(_)) => src.cloned().unwrap(),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(o)) => Value::Object(o),
            Ok(other) => serde_json::json!({ "value": other }),
            Err(_) => serde_json::json!({ "input": s }),
        },
        Some(other) => serde_json::json!({ "value": other.clone() }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_single_hermes_block_amid_prose() {
        let text = "I'll read the config first.\n\
            <tool_call>{\"name\": \"fs.read\", \"arguments\": {\"path\": \"a.txt\"}}</tool_call>\n\
            Then I'll edit it.";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "fs.read");
        assert_eq!(calls[0].arguments, json!({ "path": "a.txt" }));
    }

    #[test]
    fn parses_multiple_parallel_hermes_blocks() {
        let text = "<tool_call>{\"name\":\"fs.read\",\"arguments\":{\"path\":\"a\"}}</tool_call>\
            <tool_call>{\"name\":\"fs.read\",\"arguments\":{\"path\":\"b\"}}</tool_call>";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, json!({ "path": "a" }));
        assert_eq!(calls[1].arguments, json!({ "path": "b" }));
    }

    #[test]
    fn parses_openai_tool_calls_array_with_string_arguments() {
        let text = r#"{"tool_calls":[{"id":"call_1","type":"function",
            "function":{"name":"shell.run","arguments":"{\"argv\":[\"ls\"]}"}}]}"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell.run");
        assert_eq!(calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(calls[0].arguments, json!({ "argv": ["ls"] }));
    }

    #[test]
    fn parses_fenced_json_block() {
        let text = "Here is the call:\n```json\n{\"name\":\"git.status\",\"args\":{}}\n```\n";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "git.status");
        assert_eq!(calls[0].arguments, json!({}));
    }

    #[test]
    fn parses_bare_json_object_with_tool_key() {
        let calls = parse_tool_calls("{\"tool\":\"fs.list\",\"args\":{\"path\":\".\"}}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "fs.list");
        assert_eq!(calls[0].arguments, json!({ "path": "." }));
    }

    #[test]
    fn missing_arguments_become_empty_object() {
        let calls = parse_tool_calls("<tool_call>{\"name\":\"git.status\"}</tool_call>");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, json!({}));
    }

    #[test]
    fn non_json_string_arguments_are_wrapped_not_dropped() {
        let calls = parse_tool_calls(
            "<tool_call>{\"name\":\"shell.run\",\"arguments\":\"ls -la\"}</tool_call>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, json!({ "input": "ls -la" }));
    }

    #[test]
    fn plain_text_turn_yields_no_calls() {
        assert!(parse_tool_calls("Just thinking out loud, no tools yet.").is_empty());
        assert!(!has_tool_call("Just thinking out loud, no tools yet."));
    }

    #[test]
    fn malformed_block_is_skipped_not_fatal() {
        // First block is broken JSON, second is valid: we recover the valid one.
        let text = "<tool_call>{name: broken}</tool_call>\
            <tool_call>{\"name\":\"fs.read\",\"arguments\":{\"path\":\"ok\"}}</tool_call>";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, json!({ "path": "ok" }));
    }

    #[test]
    fn brace_inside_string_does_not_truncate_span() {
        let text = "call: {\"name\":\"fs.write\",\"arguments\":{\"content\":\"a } b\"}}";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, json!({ "content": "a } b" }));
    }

    #[test]
    fn bare_call_after_bracket_citation_is_recovered() {
        // A leading "[1]" must not shadow the real object (was dropped before the
        // all-spans fix; confirmed by adversarial review).
        let calls =
            parse_tool_calls("See [1] for details. {\"name\":\"fs.read\",\"arguments\":{}}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "fs.read");
    }

    #[test]
    fn bare_call_after_markdown_link_is_recovered() {
        let calls = parse_tool_calls(
            "I'll use the [fs.read](docs) tool: {\"name\":\"fs.read\",\"arguments\":{\"path\":\"a\"}}",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, json!({ "path": "a" }));
    }

    #[test]
    fn top_level_array_of_calls_still_parses() {
        // Regression guard: an array whose items are calls must still work.
        let calls = parse_tool_calls("[{\"name\":\"fs.read\",\"arguments\":{\"path\":\"a\"}}]");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "fs.read");
    }

    #[test]
    fn into_tool_call_carries_name_and_args() {
        let parsed = ParsedToolCall {
            name: "fs.read".into(),
            arguments: json!({ "path": "x" }),
            id: None,
        };
        let call = parsed.into_tool_call();
        assert_eq!(call.tool, "fs.read");
        assert_eq!(call.args, json!({ "path": "x" }));
    }
}
