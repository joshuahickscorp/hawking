//! `Engine` for `.gravity` artifacts, so the runtime is reachable through the
//! same adapter registry every other architecture goes through.
//!
//! This closes §4.4's production path: a local `.gravity` file, the reviewed
//! registry, hawking-core, a complete token, streamed output — with no source
//! weights consulted and no separate benchmark-only runtime beside it.
//!
//! The tokenizer is named by the artifact's own header. An artifact that has
//! to be told where its tokenizer lives is not self-describing, and emitting
//! raw token ids when it is missing would hide that rather than report it.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::engine::{
    Engine, EngineConfig, GenStats, GenerateRequest, StopReason, StreamEvent,
};
use crate::gravity::GravityShard;
use crate::gravity_glm::gpu::GravityGlmGpu;
use crate::gravity_llama::gpu::GravityLlamaGpu;
use crate::metal::MetalContext;
use crate::sample::Sampler;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};

/// Either resident-GPU backend this registry serves. A caller only ever
/// wants the logits, so this drops each backend's own trace/stats type
/// (`ForwardStats`, `GlmTrace`) rather than inventing a third type wide
/// enough to describe both.
enum GravityModel {
    Llama(GravityLlamaGpu),
    Glm(GravityGlmGpu),
}

impl GravityModel {
    fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        match self {
            GravityModel::Llama(m) => m.forward(tokens).map(|(logits, _)| logits),
            GravityModel::Glm(m) => m.forward(tokens).map(|(logits, _)| logits),
        }
    }

    fn forward_at(&self, tokens: &[u32], pos: usize) -> Result<Vec<f32>> {
        match self {
            GravityModel::Llama(m) => m.forward_at(tokens, pos).map(|(logits, _)| logits),
            GravityModel::Glm(m) => m.forward_at(tokens, pos).map(|(logits, _)| logits),
        }
    }
}

pub struct GravityEngine {
    model: GravityModel,
    tokenizer: Tokenizer,
    model_id: String,
    arch: String,
}

impl GravityEngine {
    /// True when the file carries the `.gravity` container magic, whatever it
    /// is named. Format detection reads bytes rather than trusting a suffix,
    /// because the registry's job is to dispatch on what a file *is*.
    pub fn is_gravity(path: &Path) -> bool {
        use std::io::Read;
        let mut buf = [0u8; 8];
        std::fs::File::open(path)
            .and_then(|mut f| f.read_exact(&mut buf))
            .is_ok()
            && &buf == b"GRAVITY\0"
    }
}

impl Engine for GravityEngine {
    fn load(weights: &Path, config: EngineConfig) -> Result<Self> {
        let shard = GravityShard::open(weights)?;

        // A multi-shard model's own per-shard header carries only a
        // minimal, differently-keyed architecture summary (`type`, not
        // `model_type`) -- it describes that one shard's tensors, not the
        // model. The full, canonically-keyed block the adapters need lives
        // in the assembler's `model.gravity.index.json` beside the shards.
        // Prefer it when present; a single-shard artifact (Llama) has none
        // and falls back to its own header, which already carries both
        // fields directly.
        let index_path = weights.parent().map(|d| d.join("model.gravity.index.json"));
        let index: Option<serde_json::Value> = index_path
            .filter(|p| p.is_file())
            .map(|p| -> Result<serde_json::Value> {
                let bytes = std::fs::read(&p).map_err(|e| Error::Gravity(format!("{p:?}: {e}")))?;
                serde_json::from_slice(&bytes).map_err(|e| Error::Gravity(format!("{p:?}: {e}")))
            })
            .transpose()?;
        let arch_source = index.as_ref().unwrap_or(&shard.extra);

        let arch = arch_source
            .get("architecture")
            .and_then(|a| a.get("model_type"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let model_id = arch_source
            .get("model")
            .and_then(|m| m.get("repo"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("gravity")
            .to_string();
        let tok = shard
            .extra
            .get("tokenizer")
            .ok_or_else(|| Error::Gravity("artifact declares no tokenizer".into()))?;
        // Some packers (GLM's) record only a remote reference (`{"kind":
        // "reference", "source": "org/Model"}`) rather than a local path --
        // the tokenizer was never the thing being packed, so nothing
        // resolved it to a directory at pack time. Fall back to the
        // convention every packer that DOES stage one locally uses:
        // `tokenizer/` beside the shard(s) themselves.
        let tok_path = match tok.get("dir").and_then(serde_json::Value::as_str) {
            Some(dir) => {
                let file = tok
                    .get("source")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("tokenizer.json");
                Path::new(dir).join(file)
            }
            None => weights
                .parent()
                .ok_or_else(|| {
                    Error::Gravity(format!(
                        "{weights:?}: artifact tokenizer names no directory and the shard has \
                         no parent to fall back to"
                    ))
                })?
                .join("tokenizer")
                .join("tokenizer.json"),
        };
        if !tok_path.is_file() {
            return Err(Error::Gravity(format!(
                "artifact names tokenizer {tok_path:?}, which is not present; refusing to \
                 serve token ids in its place"
            )));
        }
        drop(shard);

        let ctx = MetalContext::new_with_trace(config.trace_dispatch)?;
        let model = match arch.as_str() {
            "llama" => GravityModel::Llama(GravityLlamaGpu::open_with(ctx, weights, true)?),
            "glm_moe_dsa" => {
                // GLM is multi-shard: `weights` names one shard file (enough
                // to read the header above), and the model lives in its
                // parent directory alongside every other `model-*.gravity`.
                let dir = weights.parent().ok_or_else(|| {
                    Error::Gravity(format!(
                        "{weights:?}: a glm_moe_dsa shard must live inside its model directory"
                    ))
                })?;
                GravityModel::Glm(GravityGlmGpu::open_dir_with(ctx, dir, true)?)
            }
            other => {
                return Err(Error::Gravity(format!(
                    "no .gravity engine for architecture {other:?} yet; llama and glm_moe_dsa \
                     are wired"
                )))
            }
        };
        Ok(GravityEngine {
            model,
            tokenizer: Tokenizer::from_file(&tok_path)?,
            model_id,
            arch,
        })
    }

    fn generate(
        &mut self,
        req: GenerateRequest,
        sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<GenStats> {
        let mut ids = self.tokenizer.encode(&req.prompt, true)?;
        let prompt_tokens = ids.len();

        let t_prefill = Instant::now();
        let mut logits = self.model.forward(&ids)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;

        let mut sampler = Sampler::new(req.sampling.seed.unwrap_or(0));
        let mut pos = ids.len();
        let mut text = String::new();
        let mut completion = 0usize;
        let t_decode = Instant::now();

        let reason = loop {
            if completion >= req.max_new_tokens {
                break StopReason::MaxTokens;
            }
            if req
                .abort
                .as_ref()
                .is_some_and(|a| a.load(Ordering::Relaxed))
            {
                break StopReason::Aborted;
            }
            let next = sampler.sample(&mut logits, &req.sampling);
            if self.tokenizer.is_eog(next) {
                break StopReason::Eos;
            }
            let piece = self.tokenizer.decode_one(next)?;
            text.push_str(&piece);
            sink(StreamEvent::Token {
                id: next,
                text: piece,
            });
            ids.push(next);
            completion += 1;

            // Stop strings are checked against the accumulated text, not the
            // last piece: a stop string can straddle a token boundary, and
            // checking pieces would miss exactly those.
            if req.stop.iter().any(|s| !s.is_empty() && text.contains(s)) {
                break StopReason::StopString;
            }

            let step = Instant::now();
            logits = self.model.forward_at(&[next], pos)?;
            pos += 1;
            if req.max_stall_ms > 0
                && step.elapsed().as_millis() as u64 > req.max_stall_ms
            {
                break StopReason::Aborted;
            }
        };

        let stats = GenStats {
            prompt_tokens,
            completion_tokens: completion,
            prefill_ms,
            decode_ms: t_decode.elapsed().as_secs_f64() * 1e3,
            ..Default::default()
        };
        sink(StreamEvent::Done {
            reason,
            stats: stats.clone(),
        });
        Ok(stats)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_arch(&self) -> &str {
        &self.arch
    }

    // Deliberately NOT implementing encode_prompt_for_batch /
    // decode_token_for_batch. The trait's contract is that an engine without
    // server-side batching keeps those defaults and the server falls back to
    // `generate`. Implementing them advertises a batching path this engine
    // does not have -- the server then admits the request into a slot,
    // prefill_slot is unimplemented, and the caller gets an empty completion
    // with a 200. Claiming half a capability is worse than claiming none.

    /// Positions must be the contiguous run the cache actually holds. The
    /// runtime writes each position into its own slot, so an arbitrary
    /// position list would silently read slots nobody wrote -- and produce
    /// plausible logits from stale memory rather than an error.
    fn forward_tokens_for_test(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        if tokens.len() != positions.len() {
            return Err(Error::Gravity(format!(
                "forward_tokens_for_test: {} tokens against {} positions",
                tokens.len(),
                positions.len()
            )));
        }
        if positions
            .windows(2)
            .any(|w| w[1] != w[0] + 1)
        {
            return Err(Error::Gravity(
                "forward_tokens_for_test: positions must be contiguous and ascending".into(),
            ));
        }
        let start = positions.first().copied().unwrap_or(0);
        let mut out = Vec::with_capacity(tokens.len());
        for (i, &t) in tokens.iter().enumerate() {
            out.push(self.model.forward_at(&[t], start + i)?);
        }
        Ok(out)
    }
}
