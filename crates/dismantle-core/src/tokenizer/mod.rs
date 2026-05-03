//! Tokenizer: wraps a `tokenizers` crate `Tokenizer` instance, with a
//! GGUF-metadata fallback for models that embed their vocab + merges
//! directly in the GGUF file rather than shipping `tokenizer.json`.
//!
//! The fallback is not a full reimplementation of HuggingFace tokenizers;
//! it covers the BPE/SPM cases used by DeepSeek-V2-Lite and Qwen3-MoE.
//! Full custom tokenizer behavior (legacy added-tokens, normalization
//! quirks) is deferred to v0.2.

use crate::gguf::GgufFile;
use crate::{Error, Result};
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    inner: HfTokenizer,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    pad_id: Option<u32>,
}

impl Tokenizer {
    /// Load `tokenizer.json` from a path (preferred — exact behavior
    /// match with the upstream model).
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| Error::Model(format!("tokenizer load: {e}")))?;
        Ok(Self {
            inner,
            bos_id: None,
            eos_id: None,
            pad_id: None,
        })
    }

    /// Load from the GGUF metadata. Uses `tokenizer.ggml.tokens` and
    /// `tokenizer.ggml.merges`. The fallback supports both BPE
    /// ("gpt2"-style merges) and llama.cpp's "llama"-style SPM scoring;
    /// we surface only the merge form here in v0.1.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let model = gguf
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Model("gguf missing tokenizer.ggml.model".into()))?;

        let tokens = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_str_array())
            .ok_or_else(|| Error::Model("gguf missing tokenizer.ggml.tokens".into()))?
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();

        let merges = gguf
            .metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_str_array())
            .map(|a| a.into_iter().map(String::from).collect::<Vec<_>>())
            .unwrap_or_default();

        let bos_id = gguf
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32());
        let eos_id = gguf
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32());
        let pad_id = gguf
            .metadata
            .get("tokenizer.ggml.padding_token_id")
            .and_then(|v| v.as_u32());

        let inner = build_tokenizer(model, &tokens, &merges)?;

        Ok(Self {
            inner,
            bos_id,
            eos_id,
            pad_id,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| Error::Model(format!("encode: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| Error::Model(format!("decode: {e}")))
    }

    /// Decode a single token, preserving leading whitespace markers
    /// (Ġ for BPE, ▁ for SPM). Used by streaming generation where we
    /// emit tokens one at a time.
    pub fn decode_one(&self, id: u32) -> Result<String> {
        self.inner
            .decode(&[id], false)
            .map_err(|e| Error::Model(format!("decode: {e}")))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }
    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }
    pub fn pad_id(&self) -> Option<u32> {
        self.pad_id
    }
}

fn build_tokenizer(model: &str, tokens: &[String], merges: &[String]) -> Result<HfTokenizer> {
    use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;

    match model {
        // llama.cpp's GPT-2-style BPE, used by Qwen and DeepSeek's English vocab.
        "gpt2" | "llama" => {
            let vocab: std::collections::HashMap<String, u32> = tokens
                .iter()
                .enumerate()
                .map(|(i, t)| (t.clone(), i as u32))
                .collect();
            let merges_pairs: Vec<(String, String)> = merges
                .iter()
                .filter_map(|m| {
                    let mut it = m.splitn(2, ' ');
                    Some((it.next()?.to_string(), it.next()?.to_string()))
                })
                .collect();
            let bpe = BPE::builder()
                .vocab_and_merges(vocab, merges_pairs)
                .build()
                .map_err(|e| Error::Model(format!("bpe build: {e}")))?;
            // Configure the byte-level pipeline both ways: pre-tokenizer
            // for encode (so input bytes get mapped through GPT-2's
            // visible-byte alphabet, e.g. ' ' → 'Ġ') and decoder for
            // detokenize (so 'Ġ' → ' ', 'Ċ' → '\n', and high bytes are
            // re-assembled into UTF-8). Without these, encode produces
            // tokens the model wasn't trained on and decode prints
            // marker glyphs verbatim.
            let mut t = HfTokenizer::new(bpe);
            t.with_pre_tokenizer(Some(ByteLevelPre::new(
                /* add_prefix_space */ false, /* trim_offsets */ true,
                /* use_regex */ true,
            )));
            t.with_decoder(Some(ByteLevelDecoder::default()));
            Ok(t)
        }
        other => Err(Error::Model(format!(
            "tokenizer.ggml.model = {other:?} not supported by gguf-fallback path; \
             ship tokenizer.json alongside the gguf"
        ))),
    }
}
