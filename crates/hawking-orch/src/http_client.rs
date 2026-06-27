use crate::inference::InferenceClient;
use futures::future::BoxFuture;
use hide_core::error::{HideError, Result};
use hide_core::runtime::{GenerationStats, InferenceRequest, StreamChunk, TokenSink};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct HawkingHttpClient {
    pub base_url: String,
    pub timeout: Duration,
}

impl HawkingHttpClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            timeout: Duration::from_secs(300),
        }
    }

    pub fn build_native_generate_body(request: &InferenceRequest) -> Value {
        let sampler =
            request
                .sampler
                .clone()
                .unwrap_or_else(|| hide_core::runtime::SamplerProfile {
                    temperature: 0.0,
                    top_k: None,
                    top_p: Some(1.0),
                    repetition_penalty: None,
                    seed: Some(0),
                    deterministic: true,
                });
        json!({
            "prompt": if request.prompt.is_empty() {
                request
                    .messages
                    .iter()
                    .map(|m| format!("{}: {}", m.role, m.content))
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                request.prompt.clone()
            },
            "max_tokens": request.max_output_tokens,
            "temperature": sampler.temperature,
            "top_p": sampler.top_p,
            "seed": sampler.seed,
            "stop": []
        })
    }

    fn endpoint(&self) -> Result<HttpEndpoint> {
        parse_http_url(&self.base_url, "/v1/hawking/generate")
    }
}

impl InferenceClient for HawkingHttpClient {
    fn generate<'a>(
        &'a self,
        request: InferenceRequest,
        sink: TokenSink<'a>,
    ) -> BoxFuture<'a, Result<GenerationStats>> {
        Box::pin(async move {
            let endpoint = self.endpoint()?;
            let body = serde_json::to_string(&Self::build_native_generate_body(&request))?;
            let mut stream = TcpStream::connect((endpoint.host.as_str(), endpoint.port))?;
            stream.set_read_timeout(Some(self.timeout))?;
            stream.set_write_timeout(Some(Duration::from_secs(30)))?;
            let http = format!(
                "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                endpoint.path,
                endpoint.host_header(),
                body.len(),
                body
            );
            stream.write_all(http.as_bytes())?;
            let mut response = String::new();
            stream.read_to_string(&mut response)?;
            parse_native_sse_response(&response, sink)
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HttpEndpoint {
    host: String,
    port: u16,
    path: String,
}

impl HttpEndpoint {
    fn host_header(&self) -> String {
        if self.port == 80 {
            self.host.clone()
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }
}

fn parse_http_url(base_url: &str, path: &str) -> Result<HttpEndpoint> {
    let without_scheme = base_url
        .strip_prefix("http://")
        .ok_or_else(|| HideError::Config("only http:// runtime URLs are supported".to_string()))?;
    let authority = without_scheme.split('/').next().unwrap_or(without_scheme);
    let (host, port) = if let Some((host, port)) = authority.rsplit_once(':') {
        (
            host.to_string(),
            port.parse::<u16>()
                .map_err(|err| HideError::Config(format!("invalid runtime port: {err}")))?,
        )
    } else {
        (authority.to_string(), 80)
    };
    Ok(HttpEndpoint {
        host,
        port,
        path: path.to_string(),
    })
}

fn parse_native_sse_response(response: &str, sink: TokenSink<'_>) -> Result<GenerationStats> {
    let body = response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .unwrap_or(response);
    let mut stats = GenerationStats {
        input_tokens: 0,
        output_tokens: 0,
        decode_tokens_per_second: None,
    };
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            sink(StreamChunk::Done {
                reason: "stop".to_string(),
                stats: Some(stats.clone()),
            })?;
            break;
        }
        let value: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
            stats.output_tokens += 1;
            sink(StreamChunk::Token {
                token_id: value
                    .get("tok_index")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32),
                text: text.to_string(),
            })?;
        }
        if let Some(raw_stats) = value.get("stats") {
            stats.input_tokens = raw_stats
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            stats.output_tokens = raw_stats
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(stats.output_tokens as u64) as usize;
            stats.decode_tokens_per_second = raw_stats
                .get("dec_tps")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32);
        }
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::runtime::InferenceRequest;
    use std::collections::BTreeMap;

    #[test]
    fn native_sse_parser_emits_tokens_and_stats() {
        let response = concat!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n",
            "data: {\"tok_index\":0,\"text\":\"hel\"}\n\n",
            "data: {\"tok_index\":1,\"text\":\"lo\"}\n\n",
            "data: {\"stats\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"dec_tps\":42.0}}\n\n",
            "data: [DONE]\n\n"
        );
        let mut chunks = Vec::new();
        let stats = parse_native_sse_response(response, &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .unwrap();
        assert_eq!(stats.input_tokens, 3);
        assert_eq!(stats.output_tokens, 2);
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn native_body_uses_prompt_and_sampler() {
        let body = HawkingHttpClient::build_native_generate_body(&InferenceRequest {
            task_kind: "test".to_string(),
            prompt: "hello".to_string(),
            messages: Vec::new(),
            max_output_tokens: 7,
            sampler: None,
            grammar: None,
            want_logprobs: false,
            metadata: BTreeMap::new(),
        });
        assert_eq!(body["prompt"], "hello");
        assert_eq!(body["max_tokens"], 7);
    }
}
