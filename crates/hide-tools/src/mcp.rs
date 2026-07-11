//! MCP host/client (ch.03 §4.10), pinned to protocol revision `2025-11-25`.
//!
//! A real JSON-RPC 2.0 client that speaks both standard transports:
//!
//! * **stdio** — spawn the server subprocess (`tokio::process`), newline-delimited
//!   JSON-RPC over its stdin/stdout (stderr is logging).
//! * **Streamable HTTP** — single endpoint, `reqwest` POST per message, carrying
//!   the `MCP-Session-Id` + `MCP-Protocol-Version` headers (2025-11-25).
//!
//! It performs the `initialize` handshake, `tools/list`, and `tools/call`, and
//! **maps each MCP `Tool` → a HIDE `ToolSpec` carrying its annotations as
//! UNTRUSTED provenance** (a server claiming `read_only` does NOT relax HIDE
//! policy — §4.9.4). Discovered tools become `McpProxyTool`s registered into the
//! standard registry; calling one runs through the full HIDE dispatcher.

use crate::common;
use anyhow::{anyhow, Context, Result};
use futures::future::BoxFuture;
use hide_core::tool::{
    Purity, Tool, ToolAnnotations, ToolContent, ToolCtx, ToolResult, ToolSpec, ToolStatus,
};
use hide_core::types::EffectSet;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::process::Stdio;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;

/// MCP protocol revision this client negotiates.
pub const MCP_PROTOCOL_VERSION: &str = "2025-11-25";

// ---------------------------------------------------------------------------
// configuration / descriptors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerDescriptor {
    pub id: String,
    pub transport: McpTransport,
    #[serde(default = "default_trust")]
    pub trust: String,
}

fn default_trust() -> String {
    "third-party".to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
    },
    StreamableHttp {
        endpoint: String,
    },
}

/// An MCP `Tool` as returned by `tools/list`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
    #[serde(default, rename = "outputSchema")]
    pub output_schema: Option<Value>,
    #[serde(default)]
    pub annotations: Option<McpAnnotations>,
}

/// MCP tool annotations — hints with telling defaults (§3.2). HIDE treats these
/// as UNTRUSTED: they inform the model but never auto-relax policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct McpAnnotations {
    #[serde(rename = "readOnlyHint", default)]
    pub read_only_hint: Option<bool>,
    #[serde(rename = "destructiveHint", default)]
    pub destructive_hint: Option<bool>,
    #[serde(rename = "idempotentHint", default)]
    pub idempotent_hint: Option<bool>,
    #[serde(rename = "openWorldHint", default)]
    pub open_world_hint: Option<bool>,
}

// ---------------------------------------------------------------------------
// projections (kept stable for siblings)
// ---------------------------------------------------------------------------

/// Map an MCP `Tool` → a HIDE `ToolSpec`. Annotations are carried but UNTRUSTED:
/// the spec is namespaced `mcp:<server>/<name>` and `capabilities_required`
/// reflects the host's own classification, not the server's claim.
pub fn mcp_tool_to_hide_spec(server_id: &str, tool: McpTool) -> ToolSpec {
    // The MCP spec's annotation defaults are deliberately pessimistic; we keep
    // them, but we do NOT let a server claim read_only to widen its policy.
    let ann = tool.annotations.unwrap_or_default();
    let annotations = ToolAnnotations {
        read_only: ann.read_only_hint.unwrap_or(false),
        destructive: ann.destructive_hint.unwrap_or(true),
        idempotent: ann.idempotent_hint.unwrap_or(false),
        open_world: ann.open_world_hint.unwrap_or(true),
    };
    ToolSpec {
        name: format!("mcp:{server_id}/{}", tool.name),
        title: tool.title.unwrap_or_else(|| tool.name.clone()),
        version: "0.1.0".to_string(),
        wire_version: 1,
        description: tool.description.unwrap_or_default(),
        input_schema: tool.input_schema,
        output_schema: tool.output_schema,
        annotations,
        // Untrusted external tool: the only capability the host grants by default
        // is the bridged-call capability; real scopes come from the grant ledger.
        capabilities_required: vec!["mcp.call".to_string()],
        output_cap_bytes: 1024 * 1024,
        timeout_ms: 30_000,
    }
}

/// Project a HIDE `ToolResult` to an MCP `CallToolResult` (for HIDE-as-server).
/// `ok` ↔ `!isError`; `structured_content` ↔ `structuredContent`.
pub fn hide_result_to_mcp(result: &ToolResult) -> Value {
    json!({
        "isError": result.status != ToolStatus::Ok,
        "structuredContent": result.structured_content,
        "content": result.content,
    })
}

/// Project an MCP `CallToolResult` → a HIDE `ToolResult`. `isError:true` is a
/// *tool execution* error surfaced as data (the model can self-correct, §4.10).
pub fn mcp_result_to_hide(value: &Value) -> ToolResult {
    let is_error = value
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let structured = value.get("structuredContent").cloned();
    let content = value
        .get("content")
        .and_then(|v| v.as_array())
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|b| match b.get("type").and_then(|t| t.as_str()) {
                    Some("text") => b
                        .get("text")
                        .and_then(|t| t.as_str())
                        .map(|t| ToolContent::Text { text: t.to_string() }),
                    _ => Some(ToolContent::Json { value: b.clone() }),
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    ToolResult {
        call_id: hide_core::ids::ToolCallId::new(),
        ok: !is_error,
        status: if is_error {
            ToolStatus::ToolError
        } else {
            ToolStatus::Ok
        },
        content,
        structured_content: structured,
        bytes_ref: None,
        exit_code: None,
        effects: EffectSet::default(),
        provenance: "tool-output".to_string(), // TT8: MCP output is untrusted data
        stats: Default::default(),
        error: if is_error {
            Some(hide_core::tool::ToolError::new(
                "TOOL_FAULT",
                "mcp tool reported isError",
                true,
            ))
        } else {
            None
        },
    }
}

// ---------------------------------------------------------------------------
// JSON-RPC transport abstraction
// ---------------------------------------------------------------------------

/// A live MCP client connection. Owns the transport and the request-id counter.
pub struct McpClient {
    server_id: String,
    transport: ClientTransport,
    next_id: AtomicI64,
    protocol_version: String,
}

enum ClientTransport {
    /// Boxed because the stdio transport carries a `Child` + two large buffered
    /// handles, dwarfing the `Http` variant (`clippy::large_enum_variant`).
    Stdio(Box<StdioTransport>),
    Http {
        client: reqwest::Client,
        endpoint: String,
        session_id: Mutex<Option<String>>,
    },
}

/// State for an stdio MCP transport: the live child plus its locked I/O handles.
struct StdioTransport {
    _child: Child,
    stdin: Mutex<ChildStdin>,
    stdout: Mutex<BufReader<ChildStdout>>,
}

impl McpClient {
    /// Connect to a server over the descriptor's transport and run `initialize`.
    pub async fn connect(desc: &McpServerDescriptor) -> Result<Self> {
        let transport = match &desc.transport {
            McpTransport::Stdio { command, args } => {
                // kill_on_drop so dropping the client tears down the subprocess
                // instead of leaking it (the registry owns the client via the proxy
                // tools; without this a discarded server keeps running).
                let mut child = tokio::process::Command::new(command)
                    .args(args)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::inherit())
                    .kill_on_drop(true)
                    .spawn()
                    .with_context(|| format!("spawning MCP server {command}"))?;
                let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
                let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
                ClientTransport::Stdio(Box::new(StdioTransport {
                    _child: child,
                    stdin: Mutex::new(stdin),
                    stdout: Mutex::new(BufReader::new(stdout)),
                }))
            }
            McpTransport::StreamableHttp { endpoint } => ClientTransport::Http {
                // A request timeout so a server that accepts the POST but never
                // responds cannot hang a call forever.
                client: reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(30))
                    .build()
                    .unwrap_or_else(|_| reqwest::Client::new()),
                endpoint: endpoint.clone(),
                session_id: Mutex::new(None),
            },
        };
        let client = Self {
            server_id: desc.id.clone(),
            transport,
            next_id: AtomicI64::new(1),
            protocol_version: MCP_PROTOCOL_VERSION.to_string(),
        };
        client.initialize().await?;
        Ok(client)
    }

    pub fn server_id(&self) -> &str {
        &self.server_id
    }

    async fn initialize(&self) -> Result<Value> {
        let params = json!({
            "protocolVersion": self.protocol_version,
            "capabilities": { "roots": { "listChanged": true }, "sampling": {}, "elicitation": {} },
            "clientInfo": { "name": "hide", "version": env!("CARGO_PKG_VERSION") }
        });
        let result = self.request("initialize", params).await?;
        // After initialize, the client SHOULD send the `initialized` notification.
        self.notify("notifications/initialized", json!({})).await?;
        Ok(result)
    }

    /// `tools/list` → bridged `ToolSpec`s.
    pub async fn list_tools(&self) -> Result<Vec<ToolSpec>> {
        let result = self.request("tools/list", json!({})).await?;
        let tools = result
            .get("tools")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("tools/list: missing tools[]"))?;
        let mut specs = Vec::new();
        for t in tools {
            let tool: McpTool =
                serde_json::from_value(t.clone()).context("decoding MCP tool")?;
            specs.push(mcp_tool_to_hide_spec(&self.server_id, tool));
        }
        Ok(specs)
    }

    /// `tools/call` → bridged `ToolResult`. The `tool` arg is the *bare* MCP tool
    /// name (without the `mcp:<server>/` prefix HIDE adds to the spec).
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<ToolResult> {
        let bare = name
            .strip_prefix(&format!("mcp:{}/", self.server_id))
            .unwrap_or(name);
        let result = self
            .request("tools/call", json!({ "name": bare, "arguments": arguments }))
            .await?;
        Ok(mcp_result_to_hide(&result))
    }

    /// Send a JSON-RPC request and await its response.
    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let req = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        let response = match &self.transport {
            ClientTransport::Stdio(t) => {
                let StdioTransport { stdin, stdout, .. } = t.as_ref();
                let mut line = serde_json::to_string(&req)?;
                line.push('\n');
                {
                    let mut w = stdin.lock().await;
                    w.write_all(line.as_bytes()).await?;
                    w.flush().await?;
                }
                // Read lines until we get one with our id (skip notifications).
                let mut reader = stdout.lock().await;
                loop {
                    let mut buf = String::new();
                    let n = reader.read_line(&mut buf).await?;
                    if n == 0 {
                        return Err(anyhow!("MCP stdio closed before response"));
                    }
                    let trimmed = buf.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let v: Value = match serde_json::from_str(trimmed) {
                        Ok(v) => v,
                        Err(_) => continue, // log line on stdout; ignore
                    };
                    if v.get("id").and_then(|x| x.as_i64()) == Some(id) {
                        break v;
                    }
                }
            }
            ClientTransport::Http {
                client,
                endpoint,
                session_id,
            } => {
                let mut builder = client
                    .post(endpoint)
                    .header("Content-Type", "application/json")
                    .header("Accept", "application/json, text/event-stream")
                    .header("MCP-Protocol-Version", &self.protocol_version);
                if let Some(sid) = session_id.lock().await.as_ref() {
                    builder = builder.header("MCP-Session-Id", sid);
                }
                let resp = builder.json(&req).send().await?;
                // Capture a server-assigned session id on the initialize response.
                if let Some(sid) = resp.headers().get("MCP-Session-Id") {
                    if let Ok(s) = sid.to_str() {
                        *session_id.lock().await = Some(s.to_string());
                    }
                }
                let text = resp.text().await?;
                parse_http_jsonrpc(&text, id)?
            }
        };
        if let Some(err) = response.get("error") {
            return Err(anyhow!("JSON-RPC error: {err}"));
        }
        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    }

    /// Send a JSON-RPC notification (no id, no response expected).
    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let note = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        match &self.transport {
            ClientTransport::Stdio(t) => {
                let stdin = &t.stdin;
                let mut line = serde_json::to_string(&note)?;
                line.push('\n');
                let mut w = stdin.lock().await;
                w.write_all(line.as_bytes()).await?;
                w.flush().await?;
            }
            ClientTransport::Http {
                client,
                endpoint,
                session_id,
            } => {
                let mut builder = client
                    .post(endpoint)
                    .header("Content-Type", "application/json")
                    .header("MCP-Protocol-Version", &self.protocol_version);
                if let Some(sid) = session_id.lock().await.as_ref() {
                    builder = builder.header("MCP-Session-Id", sid);
                }
                let _ = builder.json(&note).send().await?;
            }
        }
        Ok(())
    }
}

/// Parse an HTTP JSON-RPC response body, which may be either a plain JSON object
/// or an SSE stream (`data: {…}` lines). Returns the message matching `id`.
fn parse_http_jsonrpc(text: &str, id: i64) -> Result<Value> {
    let trimmed = text.trim_start();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        let v: Value = serde_json::from_str(trimmed).context("decoding JSON-RPC body")?;
        return Ok(v);
    }
    // SSE: scan `data:` lines for the matching id.
    let mut fallback = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(payload) = line.strip_prefix("data:") {
            let payload = payload.trim();
            if payload == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(payload) {
                if v.get("id").and_then(|x| x.as_i64()) == Some(id) {
                    return Ok(v);
                }
                fallback.get_or_insert(v);
            }
        }
    }
    fallback.ok_or_else(|| anyhow!("no JSON-RPC message found in HTTP response"))
}

// ---------------------------------------------------------------------------
// proxy tool — a discovered MCP tool registered into the HIDE registry
// ---------------------------------------------------------------------------

/// A registered proxy over a bridged MCP tool. Calling it runs `tools/call` on
/// the live client. Subject to the full HIDE permission model via the dispatcher.
pub struct McpProxyTool {
    spec: ToolSpec,
    client: Arc<McpClient>,
}

impl McpProxyTool {
    pub fn new(spec: ToolSpec, client: Arc<McpClient>) -> Self {
        Self { spec, client }
    }
}

impl Tool for McpProxyTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
        let name = self.spec.name.clone();
        let client = self.client.clone();
        Box::pin(async move {
            match client.call_tool(&name, args).await {
                Ok(result) => result,
                Err(err) => common::coded("TOOL_FAULT", err.to_string(), true, None),
            }
        })
    }

    fn purity(&self) -> Purity {
        Purity::Impure
    }
}

/// Connect to a server, discover its tools, and register each as an `McpProxyTool`
/// in `registry`. Returns the bridged specs (also the live client for shutdown).
pub async fn discover_and_register(
    desc: &McpServerDescriptor,
    registry: &hide_core::tool::ToolRegistry,
) -> Result<(Arc<McpClient>, Vec<ToolSpec>)> {
    let client = Arc::new(McpClient::connect(desc).await?);
    let specs = client.list_tools().await?;
    for spec in &specs {
        registry.register(McpProxyTool::new(spec.clone(), client.clone()));
    }
    Ok((client, specs))
}

/// Per-server budget for connect + tools/list, so one hung server cannot stall
/// the whole catalog.
pub const MCP_REGISTER_TIMEOUT_SECS: u64 = 30;

/// The outcome of trying to register one MCP server's tools.
pub struct McpRegistration {
    pub server_id: String,
    /// `Some` on success. NOTE: the registry itself owns a clone of the client via
    /// each registered proxy tool, so the tools stay callable even if this handle
    /// is dropped. Keep it if you want an explicit handle to the connection (e.g.
    /// to hold the subprocess); dropping the whole registry is what tears the
    /// server down (the client sets `kill_on_drop`).
    pub client: Option<Arc<McpClient>>,
    /// Names of the tools registered from this server (`mcp:<id>/<tool>`).
    pub tools: Vec<String>,
    /// `Some` if this server failed to connect/list, timed out, or was a duplicate
    /// id (the others still ran).
    pub error: Option<String>,
}

/// Connect to and register every descriptor's tools into `registry`, resiliently:
/// each server gets a [`MCP_REGISTER_TIMEOUT_SECS`] budget, and a server that
/// fails, times out, or has a duplicate id is recorded as an error and does NOT
/// abort the rest (a single bad or hung MCP server must not disable the whole tool
/// catalog). Returns one [`McpRegistration`] per descriptor, in order.
pub async fn register_mcp_servers(
    descriptors: &[McpServerDescriptor],
    registry: &hide_core::tool::ToolRegistry,
) -> Vec<McpRegistration> {
    let dur = std::time::Duration::from_secs(MCP_REGISTER_TIMEOUT_SECS);
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(descriptors.len());
    for desc in descriptors {
        // A duplicate id would silently shadow the first server's tools in the
        // registry (same `mcp:<id>/<tool>` keys), so refuse it explicitly.
        if !seen.insert(desc.id.clone()) {
            out.push(McpRegistration {
                server_id: desc.id.clone(),
                client: None,
                tools: Vec::new(),
                error: Some(format!(
                    "duplicate server id \"{}\" skipped (would shadow the first)",
                    desc.id
                )),
            });
            continue;
        }
        let reg = match tokio::time::timeout(dur, discover_and_register(desc, registry)).await {
            Ok(Ok((client, specs))) => McpRegistration {
                server_id: desc.id.clone(),
                client: Some(client),
                tools: specs.iter().map(|s| s.name.clone()).collect(),
                error: None,
            },
            Ok(Err(e)) => McpRegistration {
                server_id: desc.id.clone(),
                client: None,
                tools: Vec::new(),
                error: Some(e.to_string()),
            },
            Err(_) => McpRegistration {
                server_id: desc.id.clone(),
                client: None,
                tools: Vec::new(),
                error: Some(format!(
                    "timed out after {MCP_REGISTER_TIMEOUT_SECS}s connecting/listing"
                )),
            },
        };
        out.push(reg);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_maps_annotations_as_untrusted() {
        let tool = McpTool {
            name: "deploy".into(),
            title: Some("Deploy".into()),
            description: Some("desc".into()),
            input_schema: json!({"type":"object","additionalProperties":false}),
            output_schema: None,
            annotations: Some(McpAnnotations {
                read_only_hint: Some(true),
                destructive_hint: Some(false),
                idempotent_hint: Some(true),
                open_world_hint: Some(false),
            }),
        };
        let spec = mcp_tool_to_hide_spec("acme", tool);
        assert_eq!(spec.name, "mcp:acme/deploy");
        // server's annotations are carried...
        assert!(spec.annotations.read_only);
        // ...but the capability is only the bridged-call cap (not relaxed).
        assert_eq!(spec.capabilities_required, vec!["mcp.call".to_string()]);
    }

    #[test]
    fn mcp_result_iserror_maps_to_not_ok() {
        let v = json!({ "isError": true, "content": [{"type":"text","text":"boom"}] });
        let r = mcp_result_to_hide(&v);
        assert!(!r.ok);
        assert_eq!(r.status, ToolStatus::ToolError);
        assert!(matches!(&r.content[0], ToolContent::Text { text } if text == "boom"));
    }

    #[test]
    fn mcp_result_success_maps_to_ok() {
        let v = json!({ "structuredContent": {"x":1}, "content": [{"type":"text","text":"ok"}] });
        let r = mcp_result_to_hide(&v);
        assert!(r.ok);
        assert_eq!(r.structured_content.unwrap()["x"], 1);
    }

    #[test]
    fn parse_http_plain_json() {
        let body = r#"{"jsonrpc":"2.0","id":7,"result":{"ok":true}}"#;
        let v = parse_http_jsonrpc(body, 7).unwrap();
        assert_eq!(v["result"]["ok"], true);
    }

    #[test]
    fn parse_http_sse_picks_matching_id() {
        let body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"v\":1}}\n\ndata: [DONE]\n";
        let v = parse_http_jsonrpc(body, 3).unwrap();
        assert_eq!(v["result"]["v"], 1);
    }

    #[tokio::test]
    async fn stdio_client_lists_and_calls_tools_against_a_fake_server() {
        // A tiny Python JSON-RPC server that implements initialize/tools/list/
        // tools/call over stdio. Skips if python3 is unavailable.
        if which_python().is_none() {
            eprintln!("python3 not found; skipping stdio MCP integration test");
            return;
        }
        let py = which_python().unwrap();
        let server_src = FAKE_SERVER;
        let desc = McpServerDescriptor {
            id: "fake".into(),
            transport: McpTransport::Stdio {
                command: py,
                args: vec!["-c".into(), server_src.into()],
            },
            trust: "third-party".into(),
        };
        let client = McpClient::connect(&desc).await.expect("connect");
        let specs = client.list_tools().await.expect("list");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "mcp:fake/echo");
        let result = client
            .call_tool("echo", json!({ "msg": "hi" }))
            .await
            .expect("call");
        assert!(result.ok);
        assert_eq!(result.structured_content.unwrap()["echoed"], "hi");
    }

    #[tokio::test]
    async fn register_mcp_servers_registers_tools_and_survives_a_bad_server() {
        if which_python().is_none() {
            eprintln!("python3 not found; skipping MCP registration test");
            return;
        }
        let py = which_python().unwrap();
        let good = McpServerDescriptor {
            id: "good".into(),
            transport: McpTransport::Stdio {
                command: py,
                args: vec!["-c".into(), FAKE_SERVER.into()],
            },
            trust: "third-party".into(),
        };
        // A server that cannot even launch: it must be recorded as an error, not
        // panic or abort the good one.
        let bad = McpServerDescriptor {
            id: "bad".into(),
            transport: McpTransport::Stdio {
                command: "definitely-not-a-real-binary-xyzzy".into(),
                args: vec![],
            },
            trust: "third-party".into(),
        };
        let registry = hide_core::tool::ToolRegistry::default();
        let results = register_mcp_servers(&[good, bad], &registry).await;

        assert_eq!(results.len(), 2);
        let good_r = results.iter().find(|r| r.server_id == "good").unwrap();
        assert!(good_r.error.is_none(), "good server errored: {:?}", good_r.error);
        assert!(good_r.tools.contains(&"mcp:good/echo".to_string()));
        let bad_r = results.iter().find(|r| r.server_id == "bad").unwrap();
        assert!(bad_r.error.is_some(), "bad server should have recorded an error");
        // The registry actually holds the good server's proxy tool, dispatchable.
        assert!(registry.get("mcp:good/echo").is_some());
    }

    #[tokio::test]
    async fn register_mcp_servers_rejects_duplicate_ids() {
        if which_python().is_none() {
            eprintln!("python3 not found; skipping MCP dup-id test");
            return;
        }
        let py = which_python().unwrap();
        let mk = |id: &str| McpServerDescriptor {
            id: id.to_string(),
            transport: McpTransport::Stdio {
                command: py.clone(),
                args: vec!["-c".into(), FAKE_SERVER.into()],
            },
            trust: "third-party".into(),
        };
        let registry = hide_core::tool::ToolRegistry::default();
        let results = register_mcp_servers(&[mk("dup"), mk("dup")], &registry).await;
        assert_eq!(results.len(), 2);
        // The first registers; the second is refused as a duplicate, not silently
        // clobbering the first's tools.
        assert!(results[0].error.is_none());
        assert!(results[1]
            .error
            .as_deref()
            .unwrap_or("")
            .contains("duplicate"));
    }

    #[tokio::test]
    async fn http_client_lists_and_calls_tools_against_an_inprocess_server() {
        use axum::{
            extract::Json as AxumJson,
            http::HeaderMap,
            response::{IntoResponse, Response},
            routing::post,
            Router,
        };
        use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
        use std::sync::Arc as StdArc;

        // Tracks that the client echoed back the server-assigned session id on the
        // second request — the Streamable-HTTP session leg, end to end.
        let saw_session_id = StdArc::new(AtomicBool::new(false));
        let saw = saw_session_id.clone();

        async fn rpc(
            saw: StdArc<AtomicBool>,
            headers: HeaderMap,
            AxumJson(req): AxumJson<Value>,
        ) -> Response {
            let id = req.get("id").cloned().unwrap_or(Value::Null);
            let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
            // Any request carrying a session id proves the header round-tripped.
            if headers.contains_key("mcp-session-id") {
                saw.store(true, AtomicOrdering::SeqCst);
            }
            let body = match method {
                "initialize" => json!({
                    "jsonrpc": "2.0", "id": id,
                    "result": {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {},
                        "serverInfo": { "name": "fake-http", "version": "0" }
                    }
                }),
                "notifications/initialized" => {
                    // Notification: no body expected. Return 202-ish empty 200.
                    return AxumJson(json!({})).into_response();
                }
                "tools/list" => json!({
                    "jsonrpc": "2.0", "id": id,
                    "result": { "tools": [{
                        "name": "echo",
                        "description": "echo back",
                        "inputSchema": {
                            "type": "object",
                            "properties": { "msg": { "type": "string" } },
                            "required": ["msg"],
                            "additionalProperties": false
                        }
                    }]}
                }),
                "tools/call" => {
                    let msg = req["params"]["arguments"]["msg"].clone();
                    json!({
                        "jsonrpc": "2.0", "id": id,
                        "result": {
                            "isError": false,
                            "structuredContent": { "echoed": msg },
                            "content": [{ "type": "text", "text": msg }]
                        }
                    })
                }
                _ => json!({
                    "jsonrpc": "2.0", "id": id,
                    "error": { "code": -32601, "message": "method not found" }
                }),
            };
            // Always assign a session id so the client must echo it back next time.
            (
                [("MCP-Session-Id", "sess-abc123")],
                AxumJson(body),
            )
                .into_response()
        }

        let app = Router::new().route(
            "/mcp",
            post(move |headers, body| rpc(saw.clone(), headers, body)),
        );

        // Bind an ephemeral port and serve in the background.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let desc = McpServerDescriptor {
            id: "fakehttp".into(),
            transport: McpTransport::StreamableHttp {
                endpoint: format!("http://{addr}/mcp"),
            },
            trust: "third-party".into(),
        };

        // connect() runs initialize + the initialized notification.
        let client = McpClient::connect(&desc).await.expect("connect");
        let specs = client.list_tools().await.expect("list");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "mcp:fakehttp/echo");

        let result = client
            .call_tool("echo", json!({ "msg": "hi-http" }))
            .await
            .expect("call");
        assert!(result.ok);
        assert_eq!(result.structured_content.unwrap()["echoed"], "hi-http");

        // The session id assigned on the initialize response must have been carried
        // on a subsequent request (the Streamable-HTTP session leg).
        assert!(
            saw_session_id.load(AtomicOrdering::SeqCst),
            "client must echo MCP-Session-Id on later requests"
        );

        server.abort();
    }

    fn which_python() -> Option<String> {
        for cand in ["python3", "python"] {
            if std::process::Command::new(cand)
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
            {
                return Some(cand.to_string());
            }
        }
        None
    }

    const FAKE_SERVER: &str = r#"
import sys, json
def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    m = req.get("method"); i = req.get("id")
    if m == "initialize":
        send({"jsonrpc":"2.0","id":i,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"fake","version":"0"}}})
    elif m == "notifications/initialized":
        pass
    elif m == "tools/list":
        send({"jsonrpc":"2.0","id":i,"result":{"tools":[{"name":"echo","description":"echo","inputSchema":{"type":"object","properties":{"msg":{"type":"string"}},"required":["msg"],"additionalProperties":False}}]}})
    elif m == "tools/call":
        msg = req["params"]["arguments"]["msg"]
        send({"jsonrpc":"2.0","id":i,"result":{"isError":False,"structuredContent":{"echoed":msg},"content":[{"type":"text","text":msg}]}})
    else:
        send({"jsonrpc":"2.0","id":i,"error":{"code":-32601,"message":"method not found"}})
"#;
}
