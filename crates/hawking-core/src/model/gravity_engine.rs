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

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::engine::{Engine, EngineConfig, GenStats, GenerateRequest, StopReason, StreamEvent};
use crate::gravity::GravityShard;
use crate::gravity_deepseek::GravityDeepSeek;
use crate::gravity_glm::gpu::GravityGlmGpu;
use crate::gravity_llama::gpu::ForwardStats as LlamaForwardStats;
use crate::gravity_llama::gpu::GravityLlamaGpu;
use crate::metal::MetalContext;
use crate::model::mixtral::MixtralEngine;
use crate::sample::Sampler;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};

/// Either backend this registry serves. Llama's physical forward accounting
/// is retained and projected into the shared `GenStats`; dropping it made the
/// served executable-Gravity path unable to prove D/S or complete forwards.
enum GravityModel {
    Llama(GravityLlamaGpu),
    Glm(GravityGlmGpu),
    DeepSeek(GravityDeepSeek),
    Mixtral(MixtralEngine),
}

impl GravityModel {
    fn forward(&self, tokens: &[u32]) -> Result<(Vec<f32>, Option<LlamaForwardStats>)> {
        match self {
            GravityModel::Llama(m) => m
                .forward(tokens)
                .map(|(logits, stats)| (logits, Some(stats))),
            GravityModel::Glm(m) => m.forward(tokens).map(|(logits, _)| (logits, None)),
            GravityModel::DeepSeek(m) => m.forward(tokens).map(|logits| (logits, None)),
            GravityModel::Mixtral(_) => Err(Error::Unimplemented(
                "Mixtral Gravity generation delegates to its stateful Engine path",
            )),
        }
    }

    fn forward_at(
        &self,
        tokens: &[u32],
        pos: usize,
    ) -> Result<(Vec<f32>, Option<LlamaForwardStats>)> {
        match self {
            GravityModel::Llama(m) => m
                .forward_at(tokens, pos)
                .map(|(logits, stats)| (logits, Some(stats))),
            GravityModel::Glm(m) => m.forward_at(tokens, pos).map(|(logits, _)| (logits, None)),
            GravityModel::DeepSeek(m) => m.forward_at(tokens, pos).map(|logits| (logits, None)),
            GravityModel::Mixtral(_) => Err(Error::Unimplemented(
                "Mixtral Gravity generation delegates to its stateful Engine path",
            )),
        }
    }

    fn device_name(&self) -> Option<String> {
        match self {
            GravityModel::Llama(m) => Some(m.device_name()),
            GravityModel::Glm(m) => Some(m.device_name()),
            GravityModel::DeepSeek(_) => None,
            GravityModel::Mixtral(m) => m.metal_ctx.as_ref().map(|ctx| ctx.device_name()),
        }
    }
}

pub struct GravityEngine {
    model: GravityModel,
    tokenizer: Tokenizer,
    model_id: String,
    arch: String,
    /// SHA-256 of `model.gravity.index.json` beside the shards, when present.
    /// Multi-shard GLM artifacts always carry one; single-shard Llama may not.
    index_sha256: Option<String>,
    /// Absolute path of the artifact's chat template file when present.
    chat_template_path: Option<String>,
    /// Raw chat-template text loaded from the artifact (not a guessed template).
    chat_template: Option<String>,
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

    /// True for a shard selected from an assembled activation-aware model.
    /// The full index/shard ABI and hash are validated by the loader.
    pub fn is_activation_aware(path: &Path) -> bool {
        path.is_file()
            && path.extension().and_then(|value| value.to_str()) == Some("aap")
            && path
                .parent()
                .map(|directory| {
                    directory
                        .join("model.activation_aware.index.json")
                        .is_file()
                })
                .unwrap_or(false)
    }

    /// Resolve a CLI `--gravity` / `HAWKING_GRAVITY` path to a loadable shard
    /// file. Accepts a single `.gravity` shard or an assembled Gravity /
    /// activation-aware model directory.
    pub fn resolve_entry(path: &Path) -> Result<std::path::PathBuf> {
        if path.is_file() {
            let is_aap_entry = Self::is_activation_aware(path);
            if !Self::is_gravity(path) && !is_aap_entry {
                return Err(Error::Gravity(format!(
                    "{path:?}: is neither a GRAVITY\\0 shard nor an indexed .aap shard"
                )));
            }
            return Ok(path.to_path_buf());
        }
        if !path.is_dir() {
            return Err(Error::Gravity(format!(
                "{path:?}: --gravity path is neither a .gravity shard file nor a model directory"
            )));
        }
        let gravity_index = path.join("model.gravity.index.json").is_file();
        let activation_index = path.join("model.activation_aware.index.json").is_file();
        if gravity_index && activation_index {
            return Err(Error::Gravity(format!(
                "{path:?}: both Gravity and activation-aware model indexes exist"
            )));
        }
        // Prefer the first ordered shard so load is deterministic.
        let suffix = if activation_index { ".aap" } else { ".gravity" };
        let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(path)
            .map_err(|e| Error::Gravity(format!("read {path:?}: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("model-") && n.ends_with(suffix))
                        .unwrap_or(false)
            })
            .collect();
        shards.sort();
        let Some(first) = shards.into_iter().next() else {
            return Err(Error::Gravity(format!(
                "{path:?}: model directory contains no model-*{suffix} shards"
            )));
        };
        if !activation_index && !Self::is_gravity(&first) {
            return Err(Error::Gravity(format!(
                "{first:?}: first shard does not start with GRAVITY\\0 magic"
            )));
        }
        Ok(first)
    }

    pub fn index_sha256(&self) -> Option<&str> {
        self.index_sha256.as_deref()
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    pub fn chat_template_path(&self) -> Option<&str> {
        self.chat_template_path.as_deref()
    }
}

#[cfg(test)]
mod resolve_tests {
    use super::GravityEngine;
    use std::path::PathBuf;
    #[test]
    fn resolve_entry_finds_ordered_shard_in_fixture_dir() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gravity_glm");
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gravity") && p.is_file())
            .collect();
        assert!(
            !entries.is_empty(),
            "fixture dir must contain at least one .gravity file"
        );
        let model_shards: Vec<_> = entries
            .iter()
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("model-"))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        if !model_shards.is_empty() {
            let resolved = GravityEngine::resolve_entry(&dir).expect("resolve dir");
            assert!(GravityEngine::is_gravity(&resolved));
        } else {
            let shard = &entries[0];
            let resolved = GravityEngine::resolve_entry(shard).expect("resolve shard file");
            assert_eq!(resolved, *shard);
            assert!(GravityEngine::is_gravity(&resolved));
        }
    }
    #[test]
    fn resolve_entry_rejects_missing_path() {
        let err = GravityEngine::resolve_entry(std::path::Path::new(
            "/tmp/definitely-not-a-gravity-artifact-xyz",
        ))
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("neither") || msg.contains("no model"),
            "unexpected error: {msg}"
        );
    }
    #[test]
    fn resolve_entry_accepts_assembled_activation_aware_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let shard = dir.path().join("model-00001-of-00001.aap");
        std::fs::write(&shard, b"synthetic-aap").unwrap();
        std::fs::write(dir.path().join("model.activation_aware.index.json"), b"{}").unwrap();
        assert!(GravityEngine::is_activation_aware(&shard));
        assert_eq!(
            GravityEngine::resolve_entry(dir.path()).expect("resolve activation-aware dir"),
            shard
        );
    }
}

impl Engine for GravityEngine {
    fn load(weights: &Path, config: EngineConfig) -> Result<Self> {
        let gravity_container = Self::is_gravity(weights);
        let shard = if gravity_container {
            Some(GravityShard::open(weights)?)
        } else {
            None
        };

        // A multi-shard model's own per-shard header carries only a
        // minimal, differently-keyed architecture summary (`type`, not
        // `model_type`) -- it describes that one shard's tensors, not the
        // model. The full, canonically-keyed block the adapters need lives
        // in the assembler's `model.gravity.index.json` beside the shards.
        // Prefer it when present; a single-shard artifact (Llama) has none
        // and falls back to its own header, which already carries both
        // fields directly.
        let index_path = weights.parent().and_then(|directory| {
            let gravity = directory.join("model.gravity.index.json");
            let activation = directory.join("model.activation_aware.index.json");
            if gravity.is_file() {
                Some(gravity)
            } else if activation.is_file() {
                Some(activation)
            } else {
                None
            }
        });
        let index_bytes: Option<(std::path::PathBuf, Vec<u8>)> = index_path
            .filter(|p| p.is_file())
            .map(|p| -> Result<(std::path::PathBuf, Vec<u8>)> {
                let bytes = std::fs::read(&p).map_err(|e| Error::Gravity(format!("{p:?}: {e}")))?;
                Ok((p, bytes))
            })
            .transpose()?;
        let index_sha256 = index_bytes.as_ref().map(|(_, bytes)| {
            use sha2::{Digest, Sha256};
            let dig = Sha256::digest(bytes);
            format!("{dig:x}")
        });
        let index: Option<serde_json::Value> = index_bytes
            .as_ref()
            .map(|(p, bytes)| {
                serde_json::from_slice(bytes).map_err(|e| Error::Gravity(format!("{p:?}: {e}")))
            })
            .transpose()?;
        let arch_source = index
            .as_ref()
            .or_else(|| shard.as_ref().map(|value| &value.extra))
            .ok_or_else(|| {
                Error::Gravity(format!(
                    "{weights:?}: no assembled model index and no Gravity shard header"
                ))
            })?;

        let arch = arch_source
            .get("architecture")
            .and_then(|a| a.get("model_type"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        // DeepSeek is intentionally parsed before the shard mmap is dropped.
        // Its MLA/MoE fields are not a dense-Llama superset; validating the
        // contract here makes a malformed artifact fail with the missing
        // function field instead of reaching a plausible wrong forward.
        let deepseek_arch = if matches!(arch.as_str(), "deepseek2" | "deepseek_v2") {
            Some(crate::gravity_deepseek::DeepSeekGravityArch::from_header(
                arch_source,
            )?)
        } else {
            None
        };
        let model_id = arch_source
            .get("model")
            .and_then(|m| m.get("repo"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("gravity")
            .to_string();
        let tok = shard
            .as_ref()
            .and_then(|value| value.extra.get("tokenizer"));
        // Some packers (GLM's) record only a remote reference (`{"kind":
        // "reference", "source": "org/Model"}`) rather than a local path --
        // the tokenizer was never the thing being packed, so nothing
        // resolved it to a directory at pack time. Fall back to the
        // convention every packer that DOES stage one locally uses:
        // `tokenizer/` beside the shard(s) themselves.
        let tok_path: Option<PathBuf> = match tok
            .and_then(|value| value.get("dir"))
            .and_then(serde_json::Value::as_str)
        {
            Some(dir) => {
                let file = tok
                    .expect("tok exists when its dir exists")
                    .get("source")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("tokenizer.json");
                Some(Path::new(dir).join(file))
            }
            None => weights
                .parent()
                .map(|parent| parent.join("tokenizer").join("tokenizer.json"))
                .filter(|path| path.is_file()),
        };
        // Mixtral's first Gravity pack carries the exact GGUF tokenizer arrays
        // in `gguf_metadata`; no mutable sidecar is needed for this embedded
        // path. Other architectures remain fail-closed when a named sidecar
        // is absent.
        let embedded_mixtral_tokenizer =
            tok_path.is_none() && arch == "mixtral" && gravity_container;
        if tok_path.is_none() && !embedded_mixtral_tokenizer {
            return Err(Error::Gravity(format!(
                "artifact tokenizer is absent ({tok_path:?}); refusing to \
                 serve token ids in its place"
            )));
        }
        // A tokenizer changes prompt IDs, RoPE positions, KV contents, and
        // therefore every continuation.  If the artifact declares an immutable
        // tokenizer digest, reject a substituted sidecar before it can enter a
        // forward.  Older artifacts without this field remain loadable but are
        // intentionally ineligible for hash-bound capability promotion.
        if let Some(expected) = tok
            .and_then(|value| value.get("sha256"))
            .and_then(serde_json::Value::as_str)
        {
            use sha2::{Digest, Sha256};
            let path = tok_path.as_ref().ok_or_else(|| {
                Error::Gravity("tokenizer hash is declared but no sidecar exists".into())
            })?;
            let bytes = std::fs::read(path)
                .map_err(|e| Error::Gravity(format!("read tokenizer {path:?}: {e}")))?;
            let actual = format!("{:x}", Sha256::digest(bytes));
            if actual != expected {
                return Err(Error::Gravity(format!(
                    "tokenizer SHA-256 mismatch for {path:?}: expected {expected}, got {actual}"
                )));
            }
        }
        // Chat template lives next to tokenizer.json. GLM chat without the
        // artifact's real template produces fluent garbage that looks like
        // success — refuse to load rather than guess.
        let template_path = tok_path
            .as_ref()
            .and_then(|path| path.parent().map(|d| d.join("chat_template.jinja")));
        let (chat_template_path, chat_template) = match template_path {
            Some(p) if p.is_file() => {
                let text = std::fs::read_to_string(&p)
                    .map_err(|e| Error::Gravity(format!("read chat template {p:?}: {e}")))?;
                (Some(p.display().to_string()), Some(text))
            }
            Some(p) if arch == "glm_moe_dsa" => {
                return Err(Error::Gravity(format!(
                    "glm_moe_dsa artifact is missing chat template at {p:?}; refusing to \
                     serve with a guessed template"
                )));
            }
            _ => (None, None),
        };
        drop(shard);

        let ctx = MetalContext::new_with_trace(config.trace_dispatch)?;
        let model = match arch.as_str() {
            "llama" | "mistral" | "qwen2" => {
                GravityModel::Llama(GravityLlamaGpu::open_with(ctx, weights, true)?)
            }
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
            "deepseek2" | "deepseek_v2" => {
                let _contract = deepseek_arch
                    .as_ref()
                    .expect("deepseek header was parsed above");
                let parent = weights.parent().ok_or_else(|| {
                    Error::Gravity(format!(
                        "{weights:?}: DeepSeek Gravity shard has no parent directory"
                    ))
                })?;
                let assembled = parent.join("model.gravity.index.json").is_file()
                    || parent.join("model.activation_aware.index.json").is_file();
                let model = if assembled {
                    GravityDeepSeek::open_dir(parent, true)?
                } else {
                    GravityDeepSeek::open(weights, true)?
                };
                GravityModel::DeepSeek(model)
            }
            "mixtral" => GravityModel::Mixtral(MixtralEngine::load(weights, config)?),
            other => {
                return Err(Error::Gravity(format!(
                "no .gravity engine for architecture {other:?} yet; llama/mistral/qwen2/mixtral \
                     and glm_moe_dsa are wired; DeepSeek MLA/MoE now has a correctness-first \
                     CPU path, while resident Metal promotion remains gated on parity"
            )))
            }
        };
        Ok(GravityEngine {
            model,
            tokenizer: if let Some(path) = tok_path.as_ref() {
                Tokenizer::from_file(path)?
            } else {
                Tokenizer::from_gguf(&crate::gguf::GgufFile::open(weights)?)?
            },
            model_id,
            arch,
            index_sha256,
            chat_template_path,
            chat_template,
        })
    }

    fn generate(
        &mut self,
        req: GenerateRequest,
        sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<GenStats> {
        // Mixtral owns a stateful KV/router engine whose complete-token
        // accounting is already implemented in its adapter.  Delegating the
        // whole request keeps Gravity generation on that same resident path;
        // routing it through the dense forward shim would reset state and
        // silently invalidate parity.
        if let GravityModel::Mixtral(model) = &mut self.model {
            return model.generate(req, sink);
        }
        let mut ids = self.tokenizer.encode(&req.prompt, true)?;
        let prompt_tokens = ids.len();

        let t_prefill = Instant::now();
        let (mut logits, prefill_physical) = self.model.forward(&ids)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;

        let mut sampler = Sampler::new(req.sampling.seed.unwrap_or(0));
        let mut pos = ids.len();
        let mut text = String::new();
        let mut completion = 0usize;
        let mut decode_token_ms = Vec::new();
        let mut decode_dispatches = 0usize;
        let mut decode_command_buffers = 0usize;
        let mut completed_decode_forwards = 0usize;
        let mut last_dispatches = 0usize;
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
            // The logits produced by the final requested token are never
            // consumed.  Do not pay for a trailing forward when the request
            // is complete; this keeps complete-token accounting identical to
            // the repaired native Qwen path and removes one needless command
            // graph replay per generation.
            if completion >= req.max_new_tokens {
                break StopReason::MaxTokens;
            }

            let step = Instant::now();
            let (next_logits, physical) = self.model.forward_at(&[next], pos)?;
            logits = next_logits;
            let complete_forward_ms = step.elapsed().as_secs_f64() * 1e3;
            decode_token_ms.push(complete_forward_ms);
            completed_decode_forwards += 1;
            if let Some(physical) = physical {
                decode_dispatches += physical.dispatches;
                decode_command_buffers += physical.command_buffers;
                last_dispatches = physical.dispatches;
            }
            pos += 1;
            if req.max_stall_ms > 0 && step.elapsed().as_millis() as u64 > req.max_stall_ms {
                break StopReason::Aborted;
            }
        };

        let stats = GenStats {
            prompt_tokens,
            completion_tokens: completion,
            prefill_ms,
            decode_ms: t_decode.elapsed().as_secs_f64() * 1e3,
            decode_token_ms,
            device_id: self.model.device_name(),
            metal_dispatches: last_dispatches,
            dispatches_per_forward: last_dispatches,
            decode_metal_dispatches_total: decode_dispatches,
            completed_decode_forwards,
            decode_command_buffers_total: decode_command_buffers,
            prefill_metal_dispatches_total: prefill_physical
                .as_ref()
                .map_or(0, |value| value.dispatches),
            prefill_command_buffers_total: prefill_physical
                .as_ref()
                .map_or(0, |value| value.command_buffers),
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

    fn artifact_index_sha256(&self) -> Option<&str> {
        self.index_sha256.as_deref()
    }

    fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    fn chat_template_path(&self) -> Option<&str> {
        self.chat_template_path.as_deref()
    }

    fn is_base_runtime(&self) -> bool {
        true
    }

    // Tokenize/decode for the serve admit path. We still do NOT implement
    // prefill_slot / multiseq: the continuous-batch loop's prefill fails and
    // hawking-serve falls back to single-stream `generate` (the same path
    // gravity_glm_tps / gravity_generate exercise). Encoding alone is enough
    // for admit; claiming a full batch path we do not have would be a facade.

    fn encode_prompt_for_batch(&self, prompt: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(prompt, true)
    }

    fn decode_token_for_batch(&self, token: u32) -> Result<String> {
        self.tokenizer.decode_one(token)
    }

    fn eos_id_for_batch(&self) -> Option<u32> {
        self.tokenizer.eos_id()
    }

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
        if positions.windows(2).any(|w| w[1] != w[0] + 1) {
            return Err(Error::Gravity(
                "forward_tokens_for_test: positions must be contiguous and ascending".into(),
            ));
        }
        let start = positions.first().copied().unwrap_or(0);
        let mut out = Vec::with_capacity(tokens.len());
        for (i, &t) in tokens.iter().enumerate() {
            let (logits, _) = self.model.forward_at(&[t], start + i)?;
            out.push(logits);
        }
        Ok(out)
    }
}
