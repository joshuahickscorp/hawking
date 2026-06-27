use futures::future::BoxFuture;
use hide_core::error::Result;
use hide_core::runtime::{GenerationStats, InferenceRequest, StreamChunk, TokenSink};

pub trait InferenceClient: Send + Sync {
    fn generate<'a>(
        &'a self,
        request: InferenceRequest,
        sink: TokenSink<'a>,
    ) -> BoxFuture<'a, Result<GenerationStats>>;
}

#[derive(Debug, Clone)]
pub struct StubInferenceClient {
    pub response: String,
}

impl InferenceClient for StubInferenceClient {
    fn generate<'a>(
        &'a self,
        _request: InferenceRequest,
        sink: TokenSink<'a>,
    ) -> BoxFuture<'a, Result<GenerationStats>> {
        Box::pin(async move {
            sink(StreamChunk::Token {
                token_id: None,
                text: self.response.clone(),
            })?;
            sink(StreamChunk::Done {
                reason: "stop".to_string(),
                stats: None,
            })?;
            Ok(GenerationStats {
                input_tokens: 0,
                output_tokens: 1,
                decode_tokens_per_second: None,
            })
        })
    }
}
