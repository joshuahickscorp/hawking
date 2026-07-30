//! Sole physical Bridge-surface authority (package-contained in hawking-adapters).
//!
//! A route that returns a plausible-looking canned response is strictly worse
//! than a 501. Unimplemented OpenAI Responses / Anthropic Messages endpoints
//! return 501 with a machine-readable body — they do not fake success.

use serde::Serialize;
use serde_json::{json, Value};

/// Endpoint status as required by the bridge-events contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointStatus {
    Live,
    Partial,
    NotImplemented,
}

impl EndpointStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            EndpointStatus::Live => "live",
            EndpointStatus::Partial => "partial",
            EndpointStatus::NotImplemented => "not_implemented",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SurfaceEndpoint {
    pub endpoint: &'static str,
    pub status: EndpointStatus,
    pub entry_path: &'static str,
    pub tests: &'static [&'static str],
}

/// The Bridge surface table (source of truth for HAWKING_BRIDGE_SURFACE.json).
pub fn bridge_surface_endpoints() -> &'static [SurfaceEndpoint] {
    &SURFACE
}

const SURFACE: [SurfaceEndpoint; 15] = [
    SurfaceEndpoint {
        endpoint: "POST /v1/chat/completions",
        status: EndpointStatus::Live,
        entry_path: "crates/hawking-serve/src/http.rs:router -> chat_completions",
        tests: &["crates/hawking-serve/tests/http_integration.rs"],
    },
    SurfaceEndpoint {
        endpoint: "POST /v1/completions",
        status: EndpointStatus::Live,
        entry_path: "crates/hawking-serve/src/http.rs:router -> completions",
        tests: &["crates/hawking-serve/tests/http_integration.rs"],
    },
    SurfaceEndpoint {
        endpoint: "GET /v1/models",
        status: EndpointStatus::Live,
        entry_path: "crates/hawking-serve/src/http.rs:router -> list_models",
        tests: &["crates/hawking-serve/tests/http_integration.rs"],
    },
    SurfaceEndpoint {
        endpoint: "GET /healthz",
        status: EndpointStatus::Live,
        entry_path: "crates/hawking-serve/src/http.rs:router -> healthz",
        tests: &["crates/hawking-serve/tests/http_integration.rs"],
    },
    SurfaceEndpoint {
        endpoint: "GET /metrics",
        status: EndpointStatus::Live,
        entry_path: "crates/hawking-serve/src/http.rs:router -> metrics",
        tests: &[],
    },
    SurfaceEndpoint {
        endpoint: "POST /v1/embeddings",
        status: EndpointStatus::Partial,
        entry_path: "crates/hawking-serve/src/http.rs:router -> embeddings",
        tests: &[],
    },
    SurfaceEndpoint {
        endpoint: "POST /v1/hawking/tokens",
        status: EndpointStatus::Live,
        entry_path: "crates/hawking-serve/src/http.rs:router -> hawking_tokens",
        tests: &["crates/hawking-serve/tests/hawking_native_endpoint.rs"],
    },
    SurfaceEndpoint {
        endpoint: "POST /v1/hawking/generate",
        status: EndpointStatus::Live,
        entry_path: "crates/hawking-serve/src/http.rs:router -> hawking_generate",
        tests: &["crates/hawking-serve/tests/hawking_native_endpoint.rs"],
    },
    SurfaceEndpoint {
        endpoint: "GET /v1/hawking/context",
        status: EndpointStatus::Live,
        entry_path: "crates/hawking-serve/src/http.rs:router -> hawking_context",
        tests: &[],
    },
    SurfaceEndpoint {
        endpoint: "GET /v1/hawking/surface",
        status: EndpointStatus::Live,
        entry_path: "crates/hawking-serve/src/http.rs:router -> hawking_surface",
        tests: &["crates/hawking-serve/tests/http_integration.rs"],
    },
    SurfaceEndpoint {
        endpoint: "POST /v1/responses",
        status: EndpointStatus::NotImplemented,
        entry_path: "crates/hawking-serve/src/http.rs:router -> not_implemented_responses",
        tests: &["crates/hawking-serve/tests/http_integration.rs"],
    },
    SurfaceEndpoint {
        endpoint: "POST /v1/messages",
        status: EndpointStatus::NotImplemented,
        entry_path: "crates/hawking-serve/src/http.rs:router -> not_implemented_anthropic_messages",
        tests: &["crates/hawking-serve/tests/http_integration.rs"],
    },
    SurfaceEndpoint {
        endpoint: "MCP",
        status: EndpointStatus::Partial,
        entry_path: "crates/hide-backend (register_mcp_servers_at_boot on hide tree)",
        tests: &[],
    },
    SurfaceEndpoint {
        endpoint: "ACP",
        status: EndpointStatus::Partial,
        entry_path: "crates/hide-acp (DeferredTurnHandler / capability negotiate)",
        tests: &[],
    },
    SurfaceEndpoint {
        endpoint: "SDK Transport -> hide-serve",
        status: EndpointStatus::NotImplemented,
        entry_path: "crates/hide-sdk/src/client.rs (MockTransport only; real transport deferred)",
        tests: &["crates/hide-sdk/tests/client.rs"],
    },
];

/// JSON document for HAWKING_BRIDGE_SURFACE.json.
pub fn bridge_surface_document() -> Value {
    let endpoints: Vec<Value> = bridge_surface_endpoints()
        .iter()
        .map(|e| {
            json!({
                "endpoint": e.endpoint,
                "status": e.status.as_str(),
                "entry_path": e.entry_path,
                "tests": e.tests,
            })
        })
        .collect();
    json!({
        "schema": "hawking.bridge.surface.v1",
        "endpoints": endpoints,
    })
}

pub fn bridge_surface_json() -> String {
    let mut s = serde_json::to_string_pretty(&bridge_surface_document()).unwrap();
    s.push('\n');
    s
}

/// Stable 501 body for unimplemented Bridge routes.
pub fn not_implemented_body(endpoint: &str) -> Value {
    json!({
        "error": {
            "message": format!("{endpoint} is not implemented"),
            "type": "not_implemented_error",
            "code": "not_implemented",
            "endpoint": endpoint,
            "capability": "not_implemented",
        }
    })
}
