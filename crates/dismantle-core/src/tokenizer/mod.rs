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
    decode_one_mode: DecodeOneMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeOneMode {
    Hf,
    SentencePiece,
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
            decode_one_mode: DecodeOneMode::Hf,
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
        let scores = gguf
            .metadata
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.as_f32_array())
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
        let unk_id = gguf
            .metadata
            .get("tokenizer.ggml.unknown_token_id")
            .and_then(|v| v.as_u32());

        let (inner, decode_one_mode) = build_tokenizer(model, &tokens, &merges, &scores, unk_id)?;

        Ok(Self {
            inner,
            bos_id,
            eos_id,
            pad_id,
            decode_one_mode,
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
        match self.decode_one_mode {
            DecodeOneMode::Hf => self
                .inner
                .decode(&[id], false)
                .map_err(|e| Error::Model(format!("decode: {e}"))),
            DecodeOneMode::SentencePiece => self.decode_sentencepiece_one(id),
        }
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

    fn decode_sentencepiece_one(&self, id: u32) -> Result<String> {
        let raw = self
            .inner
            .id_to_token(id)
            .ok_or_else(|| Error::Model(format!("decode: token id {id} outside vocabulary")))?;
        if raw.len() == 6 && raw.starts_with("<0x") && raw.ends_with('>') {
            if let Ok(byte) = u8::from_str_radix(&raw[3..5], 16) {
                return Ok(String::from_utf8(vec![byte]).unwrap_or_else(|_| "�".into()));
            }
        }
        Ok(raw.replace('▁', " "))
    }
}

fn build_tokenizer(
    model: &str,
    tokens: &[String],
    merges: &[String],
    scores: &[f32],
    unk_id: Option<u32>,
) -> Result<(HfTokenizer, DecodeOneMode)> {
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
    use tokenizers::decoders::sequence::Sequence;
    use tokenizers::decoders::DecoderWrapper;
    use tokenizers::models::bpe::BPE;
    use tokenizers::models::unigram::Unigram;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;
    use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

    match model {
        // Classic LLaMA/Mistral SentencePiece vocabularies carry scores
        // instead of BPE merges. Treating them as byte-level BPE makes
        // common prompts fall back to character/byte tokens.
        "llama" if merges.is_empty() && tokens.iter().any(|t| t.contains('▁')) => {
            if scores.len() != tokens.len() {
                return Err(Error::Model(format!(
                    "llama SentencePiece GGUF vocab has {} tokens but {} scores",
                    tokens.len(),
                    scores.len()
                )));
            }
            let vocab = tokens
                .iter()
                .cloned()
                .zip(scores.iter().map(|s| *s as f64))
                .collect::<Vec<_>>();
            let unigram = Unigram::from(vocab, unk_id.map(|id| id as usize), true)
                .map_err(|e| Error::Model(format!("unigram build: {e}")))?;
            let metaspace = Metaspace::new('▁', PrependScheme::Always, true);
            let decoder = Sequence::new(vec![
                DecoderWrapper::ByteFallback(ByteFallback::new()),
                DecoderWrapper::Metaspace(metaspace.clone()),
            ]);
            let mut t = HfTokenizer::new(unigram);
            t.with_pre_tokenizer(Some(metaspace));
            t.with_decoder(Some(decoder));
            Ok((t, DecodeOneMode::SentencePiece))
        }
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
            Ok((t, DecodeOneMode::Hf))
        }
        other => Err(Error::Model(format!(
            "tokenizer.ggml.model = {other:?} not supported by gguf-fallback path; \
             ship tokenizer.json alongside the gguf"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_scores_build_sentencepiece_unigram() {
        let tokens = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "▁".to_string(),
            "Once".to_string(),
            "▁Once".to_string(),
            "upon".to_string(),
            "▁upon".to_string(),
            "<0x21>".to_string(),
        ];
        let scores = vec![-100.0, 0.0, 0.0, -10.0, -10.0, 0.0, -10.0, 0.0, -1.0];
        let (tokenizer, mode) = build_tokenizer("llama", &tokens, &[], &scores, Some(0)).unwrap();

        assert_eq!(mode, DecodeOneMode::SentencePiece);
        let enc = tokenizer.encode("Once upon", false).unwrap();
        assert_eq!(enc.get_ids(), &[5, 7]);
        assert_eq!(tokenizer.decode(&[5, 7], false).unwrap(), "Once upon");
    }

    #[test]
    fn sentencepiece_decode_one_preserves_leading_space() {
        let tokens = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "▁Once".to_string(),
            "▁upon".to_string(),
        ];
        let scores = vec![-100.0, 0.0, 0.0, 0.0, 0.0];
        let (inner, decode_one_mode) =
            build_tokenizer("llama", &tokens, &[], &scores, Some(0)).unwrap();
        let tokenizer = Tokenizer {
            inner,
            bos_id: Some(1),
            eos_id: Some(2),
            pad_id: None,
            decode_one_mode,
        };

        assert_eq!(tokenizer.decode_one(4).unwrap(), " upon");
    }
}
