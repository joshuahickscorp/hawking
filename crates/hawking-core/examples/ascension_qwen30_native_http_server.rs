//! Minimal local HTTP/SSE adapter for the direct packed Qwen30 Metal runtime.
//!
//! This is deliberately a narrow bridge, not a replacement generic server:
//! it admits one protected Qwen30 artifact, accepts only a single user prompt
//! through the validated source-template branch, and serializes one request at
//! a time through the same model body.  It never opens raw BF16 weights, uses
//! MPS, or supplies a CPU model-math fallback.  Its endpoint existing is not a
//! capability, HCLI, TPS, TG, or tournament receipt.
//!
//! The small OpenAI-chat alias below deliberately has the *same* bounded
//! source-user path as the native endpoint.  It is not a pretend durable
//! session implementation: the response echoes a validated session header and
//! reports that context/KV reuse and manager-operation controls are still
//! unavailable.  That lets an external Agent-OS preflight fail closed instead
//! of mistaking an HTTP listener for a multi-session manager.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("ascension_qwen30_native_http_server requires macOS Metal");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::model::qwen30_complete_runtime::{
        Qwen30CompleteNativeRuntime, Qwen30CompleteRuntimeOptions, Qwen30GateUpSwiGluKernel,
        Qwen30PackedMatvecKernel,
    };
    use hawking_core::model::qwen_complete_binary::{
        CompleteBinaryAdmission, Qwen30ActivationWeightedSvdAdmission, QwenCompleteBinaryModel,
        QWEN30_ACTIVATION_WEIGHTED_SVD_SCHEMA, QWEN30_COMPLETE_BINARY_SCHEMA,
    };
    use serde_json::{json, Value};
    use std::collections::BTreeMap;
    use std::env;
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    const MAX_HEADER_BYTES: usize = 64 * 1024;
    const MAX_BODY_BYTES: usize = 64 * 1024;
    const QWEN30_GRAVITY_ARTIFACT_ID: &str = "Qwen30-Gravity-Manager-Artifact";
    const QWEN30_SCALAR_KERNEL_ID: &str = "qwen30_packed_binary_scalar_control_v1";
    const QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL_ID: &str =
        "qwen30_paired_scalar_order_no_parity_v1";

    #[derive(Debug)]
    struct Args {
        manifest: PathBuf,
        expected_manifest_seal_sha256: String,
        expected_source_audit_seal_sha256: String,
        expected_source_revision: String,
        activation_weighted: Option<ActivationWeightedBindings>,
        bind: SocketAddr,
        max_seq_len: usize,
        max_output_tokens: usize,
        gate_up_swiglu_kernel: Qwen30GateUpSwiGluKernel,
    }

    #[derive(Clone, Debug)]
    struct ActivationWeightedBindings {
        expected_revalidation_path: PathBuf,
        expected_revalidation_seal_sha256: String,
        expected_selection_path: PathBuf,
        expected_selection_seal_sha256: String,
        expected_source_snapshot_path: PathBuf,
        expected_source_snapshot_seal_sha256: String,
        expected_terminal_path: PathBuf,
        expected_terminal_seal_sha256: String,
    }

    #[derive(Debug)]
    struct Request {
        method: String,
        path: String,
        headers: BTreeMap<String, String>,
        body: Vec<u8>,
    }

    /// The intentionally narrow request shape accepted by either generation
    /// transport.  A generic chat template or a durable history is not
    /// claimed by this adapter.
    #[derive(Debug, Eq, PartialEq)]
    struct GenerationRequest {
        prompt: String,
        max_tokens: usize,
    }

    #[derive(Debug)]
    struct AdapterMetadata {
        server_instance_id: String,
        kernel_id: &'static str,
        custom_kernel_used: bool,
    }

    fn usage() -> &'static str {
        "usage: ascension_qwen30_native_http_server \\
            --manifest ABSOLUTE_PATH \\
            --expected-manifest-seal-sha256 SHA256 \\
            --expected-source-audit-seal-sha256 SHA256 \\
            --expected-source-revision REVISION \\
            --bind 127.0.0.1:PORT \\
            [--max-seq-len N] [--max-output-tokens N] \
            [--gate-up-swiglu-kernel control|paired-scalar-order-production-no-parity] \
            [--expected-revalidation-path PATH --expected-revalidation-seal-sha256 SHA256 \
             --expected-selection-path PATH --expected-selection-seal-sha256 SHA256 \
             --expected-source-snapshot-path PATH --expected-source-snapshot-seal-sha256 SHA256 \
             --expected-terminal-path PATH --expected-terminal-seal-sha256 SHA256]"
    }

    fn required<T>(value: Option<T>, flag: &str) -> Result<T, String> {
        value.ok_or_else(|| format!("missing {flag}; {}", usage()))
    }

    fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
        value
            .parse::<usize>()
            .map_err(|_| format!("{flag} must be an unsigned decimal integer; {}", usage()))
    }

    /// The server deliberately exposes only the scalar control and the
    /// separately named no-parity production candidate. Diagnostic parity
    /// modes are bounded offline experiments and cannot be served through the
    /// HTTP transport.
    fn parse_server_gate_up_swiglu_kernel(value: &str) -> Result<Qwen30GateUpSwiGluKernel, String> {
        match value {
            "control" => Ok(Qwen30GateUpSwiGluKernel::ThreeDispatchControl),
            "paired-scalar-order-production-no-parity" => {
                Ok(Qwen30GateUpSwiGluKernel::PairedScalarOrderProductionNoParity)
            }
            _ => Err(format!(
                "unsupported server --gate-up-swiglu-kernel {value:?}; expected control or paired-scalar-order-production-no-parity; {}",
                usage()
            )),
        }
    }

    fn adapter_kernel_metadata(kernel: Qwen30GateUpSwiGluKernel) -> (&'static str, bool) {
        match kernel {
            Qwen30GateUpSwiGluKernel::ThreeDispatchControl => (QWEN30_SCALAR_KERNEL_ID, false),
            Qwen30GateUpSwiGluKernel::PairedScalarOrderProductionNoParity => {
                (QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL_ID, true)
            }
            _ => unreachable!(
                "server parser admits only control or separately named production no-parity kernel"
            ),
        }
    }

    fn parse_args() -> Result<Args, String> {
        let mut manifest = None;
        let mut expected_manifest_seal_sha256 = None;
        let mut expected_source_audit_seal_sha256 = None;
        let mut expected_source_revision = None;
        let mut expected_revalidation_path = None;
        let mut expected_revalidation_seal_sha256 = None;
        let mut expected_selection_path = None;
        let mut expected_selection_seal_sha256 = None;
        let mut expected_source_snapshot_path = None;
        let mut expected_source_snapshot_seal_sha256 = None;
        let mut expected_terminal_path = None;
        let mut expected_terminal_seal_sha256 = None;
        let mut bind = None;
        let mut max_seq_len = 256usize;
        let mut max_output_tokens = 16usize;
        let mut gate_up_swiglu_kernel = Qwen30GateUpSwiGluKernel::ThreeDispatchControl;
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag:?}; {}", usage()))?;
            match flag.as_str() {
                "--manifest" => {
                    if manifest.replace(PathBuf::from(value)).is_some() {
                        return Err(format!("--manifest supplied more than once; {}", usage()));
                    }
                }
                "--expected-manifest-seal-sha256" => {
                    if expected_manifest_seal_sha256.replace(value).is_some() {
                        return Err(format!(
                            "--expected-manifest-seal-sha256 supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-source-audit-seal-sha256" => {
                    if expected_source_audit_seal_sha256.replace(value).is_some() {
                        return Err(format!(
                            "--expected-source-audit-seal-sha256 supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-source-revision" => {
                    if expected_source_revision.replace(value).is_some() {
                        return Err(format!(
                            "--expected-source-revision supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-revalidation-path" => {
                    if expected_revalidation_path
                        .replace(PathBuf::from(value))
                        .is_some()
                    {
                        return Err(format!(
                            "--expected-revalidation-path supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-revalidation-seal-sha256" => {
                    if expected_revalidation_seal_sha256.replace(value).is_some() {
                        return Err(format!(
                            "--expected-revalidation-seal-sha256 supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-selection-path" => {
                    if expected_selection_path
                        .replace(PathBuf::from(value))
                        .is_some()
                    {
                        return Err(format!(
                            "--expected-selection-path supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-selection-seal-sha256" => {
                    if expected_selection_seal_sha256.replace(value).is_some() {
                        return Err(format!(
                            "--expected-selection-seal-sha256 supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-source-snapshot-path" => {
                    if expected_source_snapshot_path
                        .replace(PathBuf::from(value))
                        .is_some()
                    {
                        return Err(format!(
                            "--expected-source-snapshot-path supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-source-snapshot-seal-sha256" => {
                    if expected_source_snapshot_seal_sha256.replace(value).is_some() {
                        return Err(format!(
                            "--expected-source-snapshot-seal-sha256 supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-terminal-path" => {
                    if expected_terminal_path
                        .replace(PathBuf::from(value))
                        .is_some()
                    {
                        return Err(format!(
                            "--expected-terminal-path supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-terminal-seal-sha256" => {
                    if expected_terminal_seal_sha256.replace(value).is_some() {
                        return Err(format!(
                            "--expected-terminal-seal-sha256 supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--bind" => {
                    if bind
                        .replace(value.parse::<SocketAddr>().map_err(|_| {
                            format!("--bind must be an IP:PORT socket address; {}", usage())
                        })?)
                        .is_some()
                    {
                        return Err(format!("--bind supplied more than once; {}", usage()));
                    }
                }
                "--max-seq-len" => max_seq_len = parse_usize(&value, "--max-seq-len")?,
                "--max-output-tokens" => {
                    max_output_tokens = parse_usize(&value, "--max-output-tokens")?
                }
                "--gate-up-swiglu-kernel" => {
                    gate_up_swiglu_kernel = parse_server_gate_up_swiglu_kernel(&value)?
                }
                _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
            }
        }
        let manifest = required(manifest, "--manifest")?;
        if !manifest.is_absolute() {
            return Err("--manifest must be an absolute path".into());
        }
        if !bind
            .as_ref()
            .is_some_and(|address| address.ip().is_loopback())
        {
            return Err(
                "--bind must be a loopback address; remote serving is deliberately refused".into(),
            );
        }
        if max_seq_len == 0 || max_output_tokens == 0 {
            return Err("--max-seq-len and --max-output-tokens must be positive".into());
        }
        let aw_any = expected_revalidation_path.is_some()
            || expected_revalidation_seal_sha256.is_some()
            || expected_selection_path.is_some()
            || expected_selection_seal_sha256.is_some()
            || expected_source_snapshot_path.is_some()
            || expected_source_snapshot_seal_sha256.is_some()
            || expected_terminal_path.is_some()
            || expected_terminal_seal_sha256.is_some();
        let activation_weighted = if aw_any {
            let revalidation_path =
                required(expected_revalidation_path, "--expected-revalidation-path")?;
            let selection_path = required(expected_selection_path, "--expected-selection-path")?;
            let snapshot_path =
                required(expected_source_snapshot_path, "--expected-source-snapshot-path")?;
            let terminal_path = required(expected_terminal_path, "--expected-terminal-path")?;
            for (flag, path) in [
                ("--expected-revalidation-path", &revalidation_path),
                ("--expected-selection-path", &selection_path),
                ("--expected-source-snapshot-path", &snapshot_path),
                ("--expected-terminal-path", &terminal_path),
            ] {
                if !path.is_absolute() {
                    return Err(format!("{flag} must be an absolute path"));
                }
            }
            Some(ActivationWeightedBindings {
                expected_revalidation_path: revalidation_path,
                expected_revalidation_seal_sha256: required(
                    expected_revalidation_seal_sha256,
                    "--expected-revalidation-seal-sha256",
                )?,
                expected_selection_path: selection_path,
                expected_selection_seal_sha256: required(
                    expected_selection_seal_sha256,
                    "--expected-selection-seal-sha256",
                )?,
                expected_source_snapshot_path: snapshot_path,
                expected_source_snapshot_seal_sha256: required(
                    expected_source_snapshot_seal_sha256,
                    "--expected-source-snapshot-seal-sha256",
                )?,
                expected_terminal_path: terminal_path,
                expected_terminal_seal_sha256: required(
                    expected_terminal_seal_sha256,
                    "--expected-terminal-seal-sha256",
                )?,
            })
        } else {
            None
        };
        Ok(Args {
            manifest,
            expected_manifest_seal_sha256: required(
                expected_manifest_seal_sha256,
                "--expected-manifest-seal-sha256",
            )?,
            expected_source_audit_seal_sha256: required(
                expected_source_audit_seal_sha256,
                "--expected-source-audit-seal-sha256",
            )?,
            expected_source_revision: required(
                expected_source_revision,
                "--expected-source-revision",
            )?,
            activation_weighted,
            bind: required(bind, "--bind")?,
            max_seq_len,
            max_output_tokens,
            gate_up_swiglu_kernel,
        })
    }

    fn find_header_end(bytes: &[u8]) -> Option<usize> {
        bytes
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|index| index + 4)
    }

    fn read_request(stream: &mut TcpStream) -> Result<Request, String> {
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(30)))
            .map_err(|error| format!("set read timeout: {error}"))?;
        let mut raw = Vec::with_capacity(4096);
        let mut temporary = [0u8; 4096];
        let header_end = loop {
            if raw.len() > MAX_HEADER_BYTES {
                return Err("HTTP header exceeds bounded adapter limit".into());
            }
            if let Some(end) = find_header_end(&raw) {
                break end;
            }
            let read = stream
                .read(&mut temporary)
                .map_err(|error| format!("read HTTP request: {error}"))?;
            if read == 0 {
                return Err("connection closed before complete HTTP header".into());
            }
            raw.extend_from_slice(&temporary[..read]);
        };
        let header = std::str::from_utf8(&raw[..header_end - 4])
            .map_err(|_| "HTTP header is not UTF-8/ASCII".to_string())?;
        let mut lines = header.split("\r\n");
        let request_line = lines
            .next()
            .ok_or_else(|| "missing HTTP request line".to_string())?;
        let mut request_parts = request_line.split_ascii_whitespace();
        let method = request_parts
            .next()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "missing HTTP method".to_string())?
            .to_string();
        let path = request_parts
            .next()
            .filter(|value| value.starts_with('/'))
            .ok_or_else(|| "missing or invalid HTTP path".to_string())?
            .to_string();
        if request_parts.next().is_none() {
            return Err("missing HTTP version".into());
        }
        let mut headers = BTreeMap::new();
        for line in lines {
            let (name, value) = line
                .split_once(':')
                .ok_or_else(|| "malformed HTTP header".to_string())?;
            let key = name.trim().to_ascii_lowercase();
            if key.is_empty() || headers.insert(key, value.trim().to_string()).is_some() {
                return Err("duplicate or empty HTTP header".into());
            }
        }
        let content_length = headers
            .get("content-length")
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|_| "invalid Content-Length".to_string())
            })
            .transpose()?
            .unwrap_or(0);
        if content_length > MAX_BODY_BYTES {
            return Err("HTTP body exceeds bounded adapter limit".into());
        }
        let mut body = raw[header_end..].to_vec();
        while body.len() < content_length {
            let remaining = content_length
                .saturating_sub(body.len())
                .min(temporary.len());
            let read = stream
                .read(&mut temporary[..remaining])
                .map_err(|error| format!("read HTTP body: {error}"))?;
            if read == 0 {
                return Err("connection closed before Content-Length bytes arrived".into());
            }
            body.extend_from_slice(&temporary[..read]);
        }
        if body.len() != content_length {
            return Err("request carried bytes after declared Content-Length".into());
        }
        Ok(Request {
            method,
            path,
            headers,
            body,
        })
    }

    fn write_response_with_headers(
        stream: &mut TcpStream,
        code: u16,
        reason: &str,
        content_type: &str,
        body: &[u8],
        extra_headers: &[(String, String)],
    ) -> Result<(), String> {
        let mut head = format!(
            "HTTP/1.1 {code} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n",
            body.len()
        );
        for (name, value) in extra_headers {
            // All dynamic headers originate in `session_id_from_request`,
            // which rejects controls and line breaks before this point.
            head.push_str(name);
            head.push_str(": ");
            head.push_str(value);
            head.push_str("\r\n");
        }
        head.push_str("\r\n");
        stream
            .write_all(head.as_bytes())
            .and_then(|_| stream.write_all(body))
            .and_then(|_| stream.flush())
            .map_err(|error| format!("write HTTP response: {error}"))
    }

    fn write_response(
        stream: &mut TcpStream,
        code: u16,
        reason: &str,
        content_type: &str,
        body: &[u8],
    ) -> Result<(), String> {
        write_response_with_headers(stream, code, reason, content_type, body, &[])
    }

    fn write_json(
        stream: &mut TcpStream,
        code: u16,
        reason: &str,
        value: Value,
    ) -> Result<(), String> {
        let body =
            serde_json::to_vec(&value).map_err(|error| format!("serialize JSON: {error}"))?;
        write_response(stream, code, reason, "application/json", &body)
    }

    fn error_json(
        stream: &mut TcpStream,
        code: u16,
        message: impl Into<String>,
    ) -> Result<(), String> {
        write_json(
            stream,
            code,
            "Bad Request",
            json!({ "error": { "message": message.into() } }),
        )
    }

    fn json_body(request: &Request, route: &str) -> Result<Value, String> {
        if request
            .headers
            .get("content-type")
            .is_some_and(|value| !value.starts_with("application/json"))
        {
            return Err(format!("{route} requires application/json content type"));
        }
        serde_json::from_slice(&request.body)
            .map_err(|error| format!("{route} JSON is invalid: {error}"))
    }

    fn parse_generation_options(
        document: &Value,
        cap: usize,
        route: &str,
    ) -> Result<usize, String> {
        let max_tokens = document
            .get("max_tokens")
            .and_then(Value::as_u64)
            .map(|value| usize::try_from(value).map_err(|_| "max_tokens exceeds usize".to_string()))
            .transpose()?
            .unwrap_or(2);
        if max_tokens == 0 || max_tokens > cap {
            return Err(format!("max_tokens must be within 1..={cap}"));
        }
        let temperature = document
            .get("temperature")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        if !temperature.is_finite() || temperature > 0.0 {
            return Err(format!(
                "{route} only implements native greedy temperature=0; refusing sampling fallback"
            ));
        }
        Ok(max_tokens)
    }

    fn parse_generation_request(
        request: &Request,
        cap: usize,
    ) -> Result<GenerationRequest, String> {
        let document = json_body(request, "native generate")?;
        let prompt = document
            .get("prompt")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "native generate requires non-empty string prompt".to_string())?
            .to_string();
        Ok(GenerationRequest {
            prompt,
            max_tokens: parse_generation_options(&document, cap, "native generate")?,
        })
    }

    fn session_id_from_request(request: &Request) -> Result<String, String> {
        let session_id = request
            .headers
            .get("x-hawking-session-id")
            .ok_or_else(|| "OpenAI chat requires X-Hawking-Session-Id".to_string())?;
        if session_id.is_empty()
            || session_id.len() > 128
            || !session_id.bytes().all(|byte| byte.is_ascii_graphic())
        {
            return Err("X-Hawking-Session-Id must be 1..=128 visible ASCII bytes".into());
        }
        Ok(session_id.clone())
    }

    fn parse_chat_generation_request(
        request: &Request,
        cap: usize,
    ) -> Result<(GenerationRequest, String), String> {
        let document = json_body(request, "OpenAI chat")?;
        if document.get("stream").and_then(Value::as_bool) != Some(true) {
            return Err(
                "OpenAI chat currently requires stream=true; non-stream session semantics are not implemented"
                    .into(),
            );
        }
        let messages = document
            .get("messages")
            .and_then(Value::as_array)
            .ok_or_else(|| "OpenAI chat requires a messages array".to_string())?;
        let [message] = messages.as_slice() else {
            return Err(
                "OpenAI chat currently accepts exactly one user message through the bound source template"
                    .into(),
            );
        };
        if message.get("role").and_then(Value::as_str) != Some("user") {
            return Err("OpenAI chat currently accepts only role=user".into());
        }
        let prompt = message
            .get("content")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "OpenAI chat user content must be a non-empty string".to_string())?
            .to_string();
        Ok((
            GenerationRequest {
                prompt,
                max_tokens: parse_generation_options(&document, cap, "OpenAI chat")?,
            },
            session_id_from_request(request)?,
        ))
    }

    fn sse_body(value: Value) -> Result<Vec<u8>, String> {
        let encoded = serde_json::to_string(&value)
            .map_err(|error| format!("serialize SSE frame: {error}"))?;
        Ok(format!("data: {encoded}\n\n").into_bytes())
    }

    fn diagnostic_decode_stats(
        generation: &hawking_core::model::qwen30_complete_runtime::Qwen30NativeGeneration,
        metadata: &AdapterMetadata,
    ) -> Value {
        // This sum intentionally contains only completed autoregressive full
        // token forwards (`generation.steps`), never source-template prefill,
        // adapter loading, request parsing, or SSE write time.  It is useful
        // endpoint telemetry but *not* a clean sustained TPS authority.
        let decode_us = generation.steps.iter().fold(0u128, |sum, step| {
            sum.saturating_add(step.elapsed.as_micros())
        });
        let completed_decode_forwards = generation.steps.len();
        let decode_ms = decode_us as f64 / 1_000.0;
        let dec_tps = if decode_us > 0 && completed_decode_forwards > 0 {
            Some(completed_decode_forwards as f64 / (decode_us as f64 / 1_000_000.0))
        } else {
            None
        };
        json!({
            "prompt_tokens": generation.prompt_token_ids.len(),
            "completion_tokens": generation.completion_token_ids.len(),
            "completed_decode_forwards": completed_decode_forwards,
            "decode_ms": decode_ms,
            "dec_tps": dec_tps,
            "timing_scope": "generated_complete_model_token_forwards_only_diagnostic_not_clean_tps",
            "diagnostic_not_clean_tps": true,
            "full_token_execution": true,
            "all_layers_executed": true,
            "native_direct_packed_metal": true,
            "model_alone": true,
            "fallback_count": 0,
            "kernel_id": metadata.kernel_id,
            "custom_kernel_used": metadata.custom_kernel_used,
        })
    }

    fn manager_operations_telemetry(
        session_id: &str,
        generation: &hawking_core::model::qwen30_complete_runtime::Qwen30NativeGeneration,
        metadata: &AdapterMetadata,
    ) -> Value {
        let stats = diagnostic_decode_stats(generation, metadata);
        json!({
            "session_id": session_id,
            "session_header": "X-Hawking-Session-Id",
            "gravity_artifact_id": QWEN30_GRAVITY_ARTIFACT_ID,
            "weight_body_id": QWEN30_GRAVITY_ARTIFACT_ID,
            "resident_model_body_count": 1,
            "single_model_body_shared_across_sessions": true,
            "weight_reuse_observed": false,
            "context_reused": false,
            "session_state_supported": false,
            "kv_state_bytes": Value::Null,
            "context_compile_latency_ms": Value::Null,
            "tool_wait_ms": Value::Null,
            "queue_wait_ms": Value::Null,
            "no_fallback": true,
            "native_direct_packed_metal": true,
            "kernel_id": metadata.kernel_id,
            "custom_kernel_used": metadata.custom_kernel_used,
            "server_instance_id": metadata.server_instance_id,
            "generated_complete_forward_stats": stats,
            "manager_operations_controls": {
                "available": false,
                "reason": "quiet lease, residency, tool recovery, rollback, storage rollback, and restart probes are not implemented by this bounded native transport adapter",
            },
        })
    }

    fn write_generation_sse(
        stream: &mut TcpStream,
        runtime: &mut Qwen30CompleteNativeRuntime,
        request: &GenerationRequest,
        metadata: &AdapterMetadata,
    ) -> Result<(), String> {
        // `generate_source_user_chat_greedy` is direct packed native Metal. It
        // emits the real decoded completion or an SSE error; no alternate
        // model/provider has an opportunity to handle this request.
        let generation =
            runtime.generate_source_user_chat_greedy(&request.prompt, request.max_tokens);
        let mut body = Vec::new();
        match generation {
            Ok(generation) => {
                if !generation.completion_text.is_empty() {
                    body.extend_from_slice(&sse_body(json!({
                        "tok_index": 0,
                        "text": generation.completion_text,
                        "native_direct_packed_metal": true,
                    }))?);
                }
                body.extend_from_slice(&sse_body(json!({
                    "stats": diagnostic_decode_stats(&generation, metadata)
                }))?);
                body.extend_from_slice(b"data: [DONE]\n\n");
            }
            Err(error) => {
                body.extend_from_slice(&sse_body(json!({
                    "error": {
                        "message": format!("direct packed native generation refused: {error}"),
                        "type": "native_runtime_error",
                    }
                }))?);
                body.extend_from_slice(b"data: [DONE]\n\n");
            }
        }
        write_response(stream, 200, "OK", "text/event-stream", &body)
    }

    fn write_chat_generation_sse(
        stream: &mut TcpStream,
        runtime: &mut Qwen30CompleteNativeRuntime,
        request: &GenerationRequest,
        session_id: &str,
        metadata: &AdapterMetadata,
    ) -> Result<(), String> {
        // This invokes exactly the same direct packed source-template greedy
        // method as `/v1/hawking/generate`; the OpenAI framing has no alternate
        // inference provider or host-side sampling path.
        let generation =
            runtime.generate_source_user_chat_greedy(&request.prompt, request.max_tokens);
        let mut body = Vec::new();
        match generation {
            Ok(generation) => {
                let telemetry = manager_operations_telemetry(session_id, &generation, metadata);
                let stats = diagnostic_decode_stats(&generation, metadata);
                if !generation.completion_text.is_empty() {
                    body.extend_from_slice(&sse_body(json!({
                        "id": format!("qwen30-{}", metadata.server_instance_id),
                        "object": "chat.completion.chunk",
                        "model": "Qwen3-Coder-30B-A3B-Instruct",
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": generation.completion_text},
                            "finish_reason": Value::Null,
                        }],
                        "hawking_manager_operations": telemetry,
                    }))?);
                }
                body.extend_from_slice(&sse_body(json!({
                    "id": format!("qwen30-{}", metadata.server_instance_id),
                    "object": "chat.completion.chunk",
                    "model": "Qwen3-Coder-30B-A3B-Instruct",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": generation.prompt_token_ids.len(),
                        "completion_tokens": generation.completion_token_ids.len(),
                        "total_tokens": generation.prompt_token_ids.len().saturating_add(generation.completion_token_ids.len()),
                    },
                    "hawking_manager_operations": telemetry,
                    "hawking_direct_packed_stats": stats,
                }))?);
                body.extend_from_slice(b"data: [DONE]\n\n");
            }
            Err(error) => {
                body.extend_from_slice(&sse_body(json!({
                    "error": {
                        "message": format!("direct packed native chat generation refused: {error}"),
                        "type": "native_runtime_error",
                    }
                }))?);
                body.extend_from_slice(b"data: [DONE]\n\n");
            }
        }
        write_response_with_headers(
            stream,
            200,
            "OK",
            "text/event-stream",
            &body,
            &[("X-Hawking-Session-Id".to_string(), session_id.to_string())],
        )
    }

    fn runtime_context(
        runtime: &Qwen30CompleteNativeRuntime,
        max_output_tokens: usize,
        metadata: &AdapterMetadata,
    ) -> Value {
        let template = runtime.source_user_chat_template();
        json!({
            "model_id": "Qwen3-Coder-30B-A3B-Instruct",
            "gravity_artifact_id": QWEN30_GRAVITY_ARTIFACT_ID,
            "arch": "Qwen3MoeForCausalLM",
            "ctx_len_native": runtime.max_seq_len(),
            "ctx_len_effective": runtime.max_seq_len(),
            "tq_multiplier": 1.0,
            "tq_estimated": false,
            "recurrent_state_bytes": Value::Null,
            "active_slots": 0,
            "free_slots": 1,
            "max_batch": 1,
            "max_output_tokens": max_output_tokens,
            "artifact_seal_sha256": runtime.artifact_manifest_seal(),
            "capability_status": "UNQUALIFIED_DIRECT_PACKED_NATIVE_RUNTIME_ONLY",
            "metal_dispatches": Value::Null,
            "chat_template": "source_user_chat_template: one_user_message_no_system_no_tools",
            "source_chat_template_path": template.source_template_path,
            "source_chat_template_sha256": template.source_template_sha256,
            "tokenizer_config_sha256": template.tokenizer_config_sha256,
            "model_alone": true,
            "fallback_count": 0,
            "server_instance_id": metadata.server_instance_id,
            "kernel_id": metadata.kernel_id,
            "custom_kernel_used": metadata.custom_kernel_used,
            "hcli_complete_token_telemetry_available": true,
            "openai_chat_boundary": {
                "path": "/v1/chat/completions",
                "stream_required": true,
                "session_header": "X-Hawking-Session-Id",
                "supported_message_shape": "one_user_message_no_system_no_tools",
                "durable_session_or_kv_reuse": false,
            },
            "manager_operations_controls": {
                "available": false,
                "state": "NOT_YET_IMPLEMENTED_UNEARNED",
            },
            "claim_boundary": {
                "actual_local_direct_packed_native_metal_adapter": true,
                "not_hcli_capability_tps_tg_or_tournament_qualified": true,
            }
        })
    }

    fn handle_connection(
        stream: &mut TcpStream,
        runtime: &mut Qwen30CompleteNativeRuntime,
        max_output_tokens: usize,
        metadata: &AdapterMetadata,
    ) -> Result<(), String> {
        let request = match read_request(stream) {
            Ok(request) => request,
            Err(error) => return error_json(stream, 400, error),
        };
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/healthz") => write_json(
                stream,
                200,
                "OK",
                json!({
                    "ready": true,
                    "provider": "qwen30-direct-packed-native-metal",
                    "model_id": "Qwen3-Coder-30B-A3B-Instruct",
                    "gravity_artifact_id": QWEN30_GRAVITY_ARTIFACT_ID,
                    "model_alone": true,
                    "fallback_count": 0,
                    "server_instance_id": metadata.server_instance_id,
                    "kernel_id": metadata.kernel_id,
                    "custom_kernel_used": metadata.custom_kernel_used,
                    "claim_boundary": "transport readiness only; not an HCLI/capability/TPS pass",
                }),
            ),
            ("GET", "/v1/hawking/context") => write_json(
                stream,
                200,
                "OK",
                runtime_context(runtime, max_output_tokens, metadata),
            ),
            ("POST", "/v1/hawking/generate") => {
                match parse_generation_request(&request, max_output_tokens) {
                    Ok(generation) => write_generation_sse(stream, runtime, &generation, metadata),
                    Err(error) => error_json(stream, 400, error),
                }
            }
            ("POST", "/v1/chat/completions") => {
                match parse_chat_generation_request(&request, max_output_tokens) {
                    Ok((generation, session_id)) => write_chat_generation_sse(
                        stream,
                        runtime,
                        &generation,
                        &session_id,
                        metadata,
                    ),
                    Err(error) => error_json(stream, 400, error),
                }
            }
            _ => write_json(
                stream,
                404,
                "Not Found",
                json!({ "error": { "message": "only GET /healthz, GET /v1/hawking/context, POST /v1/hawking/generate, and source-bound streaming POST /v1/chat/completions are available" } }),
            ),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn native_request(body: Value) -> Request {
            Request {
                method: "POST".into(),
                path: "/v1/hawking/generate".into(),
                headers: BTreeMap::from([("content-type".into(), "application/json".into())]),
                body: serde_json::to_vec(&body).unwrap(),
            }
        }

        #[test]
        fn bounded_http_header_boundary_is_exact() {
            assert_eq!(find_header_end(b"GET / HTTP/1.1\r\n\r\nbody"), Some(18));
            assert_eq!(find_header_end(b"GET / HTTP/1.1\r\n"), None);
        }

        #[test]
        fn native_generation_refuses_non_greedy_or_over_cap_requests() {
            let accepted = parse_generation_request(
                &native_request(json!({"prompt": "hello", "max_tokens": 2, "temperature": 0.0})),
                2,
            )
            .unwrap();
            assert_eq!(
                accepted,
                GenerationRequest {
                    prompt: "hello".into(),
                    max_tokens: 2,
                }
            );
            assert!(parse_generation_request(
                &native_request(json!({"prompt": "hello", "temperature": 0.1})),
                2,
            )
            .is_err());
            assert!(parse_generation_request(
                &native_request(json!({"prompt": "hello", "max_tokens": 3})),
                2,
            )
            .is_err());
        }

        #[test]
        fn openai_chat_is_streaming_single_user_and_session_bound() {
            let request = Request {
                method: "POST".into(),
                path: "/v1/chat/completions".into(),
                headers: BTreeMap::from([
                    ("content-type".into(), "application/json".into()),
                    ("x-hawking-session-id".into(), "session-1".into()),
                ]),
                body: serde_json::to_vec(&json!({
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": true,
                    "temperature": 0,
                    "max_tokens": 2,
                }))
                .unwrap(),
            };
            let (accepted, session_id) = parse_chat_generation_request(&request, 2).unwrap();
            assert_eq!(
                accepted,
                GenerationRequest {
                    prompt: "hello".into(),
                    max_tokens: 2,
                }
            );
            assert_eq!(session_id, "session-1");

            let non_stream = Request {
                body: serde_json::to_vec(&json!({
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": false,
                }))
                .unwrap(),
                ..request
            };
            assert!(parse_chat_generation_request(&non_stream, 2).is_err());
        }

        #[test]
        fn server_production_kernel_is_explicit_and_diagnostic_modes_are_refused() {
            let production =
                parse_server_gate_up_swiglu_kernel("paired-scalar-order-production-no-parity")
                    .unwrap();
            assert_eq!(
                production,
                Qwen30GateUpSwiGluKernel::PairedScalarOrderProductionNoParity
            );
            assert_eq!(
                adapter_kernel_metadata(production),
                (QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL_ID, true)
            );
            assert!(parse_server_gate_up_swiglu_kernel(
                "paired-scalar-order-candidate-device-parity"
            )
            .is_err());
        }
    }

    pub fn run() -> Result<(), String> {
        let args = parse_args()?;
        let schema = {
            let raw = std::fs::read(&args.manifest)
                .map_err(|error| format!("cannot read manifest: {error}"))?;
            let value: Value = serde_json::from_slice(&raw)
                .map_err(|error| format!("manifest is not JSON: {error}"))?;
            value
                .get("schema")
                .and_then(Value::as_str)
                .ok_or_else(|| "manifest lacks schema".to_string())?
                .to_owned()
        };
        let options = Qwen30CompleteRuntimeOptions {
            max_seq_len: args.max_seq_len,
            trace_dispatch: false,
            packed_matvec_kernel: Qwen30PackedMatvecKernel::ScalarControl,
            gate_up_swiglu_kernel: args.gate_up_swiglu_kernel,
        };
        let mut runtime = if schema == QWEN30_ACTIVATION_WEIGHTED_SVD_SCHEMA {
            let aw = args.activation_weighted.as_ref().ok_or_else(|| {
                "activation-weighted SVD candidate requires protected revalidation/selection/snapshot/terminal path+seal bindings".to_string()
            })?;
            // Mixed candidates always use three-dispatch control so HGRAVS01
            // organs are not forced through HQ30G1B1 fused gate/up kernels.
            if options.gate_up_swiglu_kernel != Qwen30GateUpSwiGluKernel::ThreeDispatchControl {
                return Err(
                    "activation-weighted SVD serve path requires --gate-up-swiglu-kernel control"
                        .into(),
                );
            }
            let admission = Qwen30ActivationWeightedSvdAdmission {
                expected_manifest_seal_sha256: args.expected_manifest_seal_sha256.clone(),
                expected_source_audit_seal_sha256: args.expected_source_audit_seal_sha256.clone(),
                expected_source_revision: args.expected_source_revision.clone(),
                expected_revalidation_path: aw.expected_revalidation_path.clone(),
                expected_revalidation_seal_sha256: aw.expected_revalidation_seal_sha256.clone(),
                expected_selection_path: aw.expected_selection_path.clone(),
                expected_selection_seal_sha256: aw.expected_selection_seal_sha256.clone(),
                expected_source_snapshot_path: aw.expected_source_snapshot_path.clone(),
                expected_source_snapshot_seal_sha256: aw
                    .expected_source_snapshot_seal_sha256
                    .clone(),
                expected_terminal_path: aw.expected_terminal_path.clone(),
                expected_terminal_seal_sha256: aw.expected_terminal_seal_sha256.clone(),
            };
            Qwen30CompleteNativeRuntime::load_activation_weighted_svd(
                &args.manifest,
                &admission,
                options,
            )
            .map_err(|error| {
                format!("activation-weighted mixed native runtime load refused: {error}")
            })?
        } else if schema == QWEN30_COMPLETE_BINARY_SCHEMA {
            if args.activation_weighted.is_some() {
                return Err(
                    "activation-weighted handoff bindings were supplied for a direct HQ30G1B1 manifest"
                        .into(),
                );
            }
            let admission = CompleteBinaryAdmission {
                model: QwenCompleteBinaryModel::Qwen30Coder,
                expected_manifest_seal_sha256: args.expected_manifest_seal_sha256,
                expected_source_audit_seal_sha256: args.expected_source_audit_seal_sha256,
                expected_source_revision: args.expected_source_revision,
            };
            Qwen30CompleteNativeRuntime::load(&args.manifest, &admission, options)
                .map_err(|error| format!("direct packed native runtime load refused: {error}"))?
        } else {
            return Err(format!(
                "unsupported Qwen30 native HTTP server manifest schema {schema:?}"
            ));
        };
        let started_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock before UNIX epoch: {error}"))?
            .as_nanos();
        let (kernel_id, custom_kernel_used) = adapter_kernel_metadata(args.gate_up_swiglu_kernel);
        let metadata = AdapterMetadata {
            server_instance_id: format!("qwen30-native-{}-{started_nanos}", std::process::id()),
            kernel_id,
            custom_kernel_used,
        };
        let listener = TcpListener::bind(args.bind)
            .map_err(|error| format!("bind loopback native adapter {}: {error}", args.bind))?;
        eprintln!(
            "qwen30 native HTTP adapter listening on http://{} (activation_weighted_svd={})",
            args.bind,
            runtime.has_activation_weighted_svd_organs()
        );
        for incoming in listener.incoming() {
            match incoming {
                Ok(mut stream) => {
                    if let Err(error) = handle_connection(
                        &mut stream,
                        &mut runtime,
                        args.max_output_tokens,
                        &metadata,
                    ) {
                        eprintln!("qwen30 direct packed native HTTP connection error: {error}");
                    }
                }
                Err(error) => eprintln!("qwen30 direct packed native HTTP accept error: {error}"),
            }
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() {
    if let Err(error) = macos::run() {
        eprintln!("qwen30 direct packed native HTTP adapter refused: {error}");
        std::process::exit(2);
    }
}
