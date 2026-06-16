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
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    inner: HfTokenizer,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    pad_id: Option<u32>,
    decode_one_mode: DecodeOneMode,
    llama_spm: Option<LlamaSpmTokenizer>,
    /// RWKV "World" greedy-trie tokenizer (raw-byte longest-match). Set when the
    /// GGUF declares `tokenizer.ggml.model == "rwkv"`; this vocab is neither
    /// BPE nor SPM, so the `tokenizers` crate cannot represent it. When present,
    /// `encode`/`decode`/`decode_one`/`vocab_size`/`eos_id` short-circuit to it.
    rwkv_world: Option<RwkvWorldTokenizer>,
    /// Control/special token ids (e.g. `<|im_end|>`, `<|im_start|>`). These are
    /// chat-template scaffolding and must never appear in streamed output.
    special_ids: HashSet<u32>,
    /// End-of-generation token ids: the GGUF eos plus chat turn terminators
    /// (`<|im_end|>`, `<|eot_id|>`, …). Generation stops on ANY of these.
    eog_ids: HashSet<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeOneMode {
    Hf,
    SentencePiece,
}

/// Token strings that terminate generation (in addition to the declared eos).
/// Different GGUF conversions set `eos_token_id` to either `<|endoftext|>` or
/// `<|im_end|>`; treating the whole family as end-of-generation makes the chat
/// path stop correctly regardless of which the file picked.
const EOG_TOKEN_STRINGS: &[&str] = &[
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|end|>",
    "<|end_of_text|>",
    "<end_of_turn>",
    "</s>",
    "<eos>",
];

/// True for control tokens identified by string shape: the `<|...|>` family
/// (Qwen, Llama-3, Phi) plus the classic SentencePiece sentinels. Real vocab
/// entries never take these forms, so there are no false positives on text.
fn is_special_token_str(s: &str) -> bool {
    (s.len() >= 4 && s.starts_with("<|") && s.ends_with("|>"))
        || matches!(
            s,
            "<s>"
                | "</s>"
                | "<unk>"
                | "<pad>"
                | "<mask>"
                | "<bos>"
                | "<eos>"
                | "<start_of_turn>"
                | "<end_of_turn>"
        )
}

/// Build the (special, end-of-generation) id sets from the vocab + the known
/// sentinel ids. `tokens` is indexed by token id.
fn build_special_sets(
    tokens: &[String],
    bos: Option<u32>,
    eos: Option<u32>,
    pad: Option<u32>,
    unk: Option<u32>,
) -> (HashSet<u32>, HashSet<u32>) {
    let mut special = HashSet::new();
    let mut eog = HashSet::new();
    for id in [bos, eos, pad, unk].into_iter().flatten() {
        special.insert(id);
    }
    if let Some(e) = eos {
        eog.insert(e);
    }
    for (i, s) in tokens.iter().enumerate() {
        let id = i as u32;
        let s = s.as_str();
        if is_special_token_str(s) {
            special.insert(id);
        }
        if EOG_TOKEN_STRINGS.contains(&s) {
            special.insert(id);
            eog.insert(id);
        }
    }
    (special, eog)
}

/// Materialize the vocab as a Vec indexed by token id (for `tokenizer.json`
/// loads, where the GGUF token list isn't available).
fn id_ordered_vocab(inner: &HfTokenizer) -> Vec<String> {
    let size = inner.get_vocab_size(true);
    let mut v = vec![String::new(); size];
    for (s, id) in inner.get_vocab(true) {
        if (id as usize) < v.len() {
            v[id as usize] = s;
        }
    }
    v
}

impl Tokenizer {
    /// Load `tokenizer.json` from a path (preferred — exact behavior
    /// match with the upstream model).
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| Error::Model(format!("tokenizer load: {e}")))?;
        let (special_ids, eog_ids) =
            build_special_sets(&id_ordered_vocab(&inner), None, None, None, None);
        Ok(Self {
            inner,
            bos_id: None,
            eos_id: None,
            pad_id: None,
            decode_one_mode: DecodeOneMode::Hf,
            llama_spm: None,
            rwkv_world: None,
            special_ids,
            eog_ids,
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
        let add_bos = gguf
            .metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let add_eos = gguf
            .metadata
            .get("tokenizer.ggml.add_eos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let (inner, decode_one_mode, llama_spm, rwkv_world) = build_tokenizer(
            model, &tokens, &merges, &scores, bos_id, eos_id, unk_id, add_bos, add_eos,
        )?;

        let (special_ids, eog_ids) = build_special_sets(&tokens, bos_id, eos_id, pad_id, unk_id);

        Ok(Self {
            inner,
            bos_id,
            eos_id,
            pad_id,
            decode_one_mode,
            llama_spm,
            rwkv_world,
            special_ids,
            eog_ids,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        if let Some(rwkv) = &self.rwkv_world {
            return Ok(rwkv.encode(text, add_special_tokens));
        }
        if let Some(spm) = &self.llama_spm {
            return spm.encode(text, add_special_tokens);
        }
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| Error::Model(format!("encode: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        if let Some(rwkv) = &self.rwkv_world {
            return Ok(rwkv.decode(ids, skip_special));
        }
        if let Some(spm) = &self.llama_spm {
            return spm.decode(ids, skip_special);
        }
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| Error::Model(format!("decode: {e}")))
    }

    /// Decode a single token, preserving leading whitespace markers
    /// (Ġ for BPE, ▁ for SPM). Used by streaming generation where we
    /// emit tokens one at a time.
    pub fn decode_one(&self, id: u32) -> Result<String> {
        // Streaming decoders emit one token at a time, so `decode`'s skip-special
        // pass never runs. Guard here so control tokens (`<|im_end|>`,
        // `<|im_start|>`, …) never leak into chat/completion output.
        if self.special_ids.contains(&id) {
            return Ok(String::new());
        }
        if let Some(rwkv) = &self.rwkv_world {
            return Ok(rwkv.decode_one(id));
        }
        match self.decode_one_mode {
            DecodeOneMode::Hf => self
                .inner
                .decode(&[id], false)
                .map_err(|e| Error::Model(format!("decode: {e}"))),
            DecodeOneMode::SentencePiece => self.decode_sentencepiece_one(id),
        }
    }

    /// True for control/special tokens (chat scaffolding) — never user-visible.
    pub fn is_special(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }

    /// True for end-of-generation tokens: the declared eos plus chat turn
    /// terminators like `<|im_end|>`. Stop generation on ANY of these — keying
    /// only on the single GGUF `eos_token_id` misses the chat terminator on
    /// conversions that set eos to `<|endoftext|>`.
    pub fn is_eog(&self, id: u32) -> bool {
        self.eog_ids.contains(&id)
    }

    /// Control/special token ids (sorted) that must never be pruned from the LM
    /// head — the model must be able to emit them (e.g. `<|im_end|>` to end a
    /// chat turn). A vocab-prune that drops these breaks chat generation.
    pub fn control_token_ids(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.special_ids.iter().copied().collect();
        v.sort_unstable();
        v
    }

    pub fn vocab_size(&self) -> usize {
        if let Some(rwkv) = &self.rwkv_world {
            return rwkv.vocab_size();
        }
        if let Some(spm) = &self.llama_spm {
            return spm.vocab_size();
        }
        self.inner.get_vocab_size(true)
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }
    pub fn eos_id(&self) -> Option<u32> {
        if let Some(rwkv) = &self.rwkv_world {
            return rwkv.eos_id();
        }
        self.eos_id
    }
    pub fn pad_id(&self) -> Option<u32> {
        self.pad_id
    }

    fn decode_sentencepiece_one(&self, id: u32) -> Result<String> {
        if let Some(spm) = &self.llama_spm {
            return spm.decode_one(id);
        }
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

#[derive(Clone)]
struct LlamaSpmTokenizer {
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    scores: Vec<f32>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    unk_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
}

impl LlamaSpmTokenizer {
    fn new(
        tokens: &[String],
        scores: &[f32],
        bos_id: Option<u32>,
        eos_id: Option<u32>,
        unk_id: Option<u32>,
        add_bos: bool,
        add_eos: bool,
    ) -> Result<Self> {
        if scores.len() != tokens.len() {
            return Err(Error::Model(format!(
                "llama SentencePiece GGUF vocab has {} tokens but {} scores",
                tokens.len(),
                scores.len()
            )));
        }
        Ok(Self {
            tokens: tokens.to_vec(),
            token_to_id: tokens
                .iter()
                .enumerate()
                .map(|(i, token)| (token.clone(), i as u32))
                .collect(),
            scores: scores.to_vec(),
            bos_id,
            eos_id,
            unk_id,
            add_bos,
            add_eos,
            add_space_prefix: true,
        })
    }

    fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        if add_special_tokens && self.add_bos {
            if let Some(id) = self.bos_id {
                out.push(id);
            }
        }

        let mut escaped = String::new();
        if self.add_space_prefix {
            escaped.push(' ');
        }
        escaped.push_str(text);
        let escaped = escaped.replace(' ', "▁");
        self.tokenize_escaped(&escaped, &mut out)?;

        if add_special_tokens && self.add_eos {
            if let Some(id) = self.eos_id {
                out.push(id);
            }
        }
        Ok(out)
    }

    fn tokenize_escaped(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        let mut symbols = Vec::<SpmSymbol>::new();
        for (start, ch) in text.char_indices() {
            let idx = symbols.len();
            if let Some(prev) = symbols.last_mut() {
                prev.next = Some(idx);
            }
            symbols.push(SpmSymbol {
                prev: idx.checked_sub(1),
                next: None,
                start,
                len: ch.len_utf8(),
            });
        }

        let mut work_queue = BinaryHeap::new();
        for i in 1..symbols.len() {
            self.try_add_bigram(text, &symbols, i - 1, i, &mut work_queue);
        }

        while let Some(bigram) = work_queue.pop() {
            let Some(right) = symbols[bigram.left].next else {
                continue;
            };
            if right != bigram.right
                || symbols[bigram.left].len == 0
                || symbols[bigram.right].len == 0
                || symbols[bigram.left].len + symbols[bigram.right].len != bigram.size
            {
                continue;
            }

            symbols[bigram.left].len = bigram.size;
            symbols[bigram.right].len = 0;
            symbols[bigram.left].next = symbols[bigram.right].next;
            if let Some(next) = symbols[bigram.right].next {
                symbols[next].prev = Some(bigram.left);
            }

            if let Some(prev) = symbols[bigram.left].prev {
                self.try_add_bigram(text, &symbols, prev, bigram.left, &mut work_queue);
            }
            if let Some(next) = symbols[bigram.left].next {
                self.try_add_bigram(text, &symbols, bigram.left, next, &mut work_queue);
            }
        }

        let mut i = symbols.first().map(|_| 0);
        while let Some(idx) = i {
            let sym = &symbols[idx];
            if sym.len > 0 {
                let piece = &text[sym.start..sym.start + sym.len];
                self.emit_piece(piece, out)?;
            }
            i = sym.next;
        }
        Ok(())
    }

    fn try_add_bigram(
        &self,
        text: &str,
        symbols: &[SpmSymbol],
        left: usize,
        right: usize,
        work_queue: &mut BinaryHeap<SpmBigram>,
    ) {
        if symbols[left].len == 0 || symbols[right].len == 0 {
            return;
        }
        let start = symbols[left].start;
        let size = symbols[left].len + symbols[right].len;
        let piece = &text[start..start + size];
        if let Some(&id) = self.token_to_id.get(piece) {
            work_queue.push(SpmBigram {
                left,
                right,
                score: self.scores[id as usize],
                size,
            });
        }
    }

    fn emit_piece(&self, piece: &str, out: &mut Vec<u32>) -> Result<()> {
        if let Some(&id) = self.token_to_id.get(piece) {
            out.push(id);
            return Ok(());
        }

        for byte in piece.as_bytes() {
            let byte_piece = format!("<0x{byte:02X}>");
            if let Some(&id) = self.token_to_id.get(&byte_piece) {
                out.push(id);
            } else if let Some(id) = self.unk_id {
                out.push(id);
            } else {
                return Err(Error::Model(format!(
                    "llama SentencePiece token {piece:?} has no byte fallback"
                )));
            }
        }
        Ok(())
    }

    fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        let mut out = String::new();
        let mut skipped_leading_bos = false;
        for &id in ids {
            if skip_special && self.is_special(id) {
                if Some(id) == self.bos_id {
                    skipped_leading_bos = true;
                }
                continue;
            }
            out.push_str(&self.decode_one(id)?);
        }
        if skip_special && (skipped_leading_bos || self.add_space_prefix) && out.starts_with(' ') {
            out.remove(0);
        }
        Ok(out)
    }

    fn decode_one(&self, id: u32) -> Result<String> {
        let raw = self
            .tokens
            .get(id as usize)
            .ok_or_else(|| Error::Model(format!("decode: token id {id} outside vocabulary")))?;
        if raw.len() == 6 && raw.starts_with("<0x") && raw.ends_with('>') {
            if let Ok(byte) = u8::from_str_radix(&raw[3..5], 16) {
                return Ok(String::from_utf8(vec![byte]).unwrap_or_else(|_| "�".into()));
            }
        }
        Ok(raw.replace('▁', " "))
    }

    fn is_special(&self, id: u32) -> bool {
        Some(id) == self.bos_id || Some(id) == self.eos_id || Some(id) == self.unk_id
    }
}

/// RWKV "World" tokenizer: a greedy longest-match trie over raw bytes.
///
/// Unlike BPE/SPM, the World vocab maps each token id to an arbitrary byte
/// sequence and tokenizes by walking the input bytes, picking the longest vocab
/// entry that matches at each position. The token strings in
/// `tokenizer.ggml.tokens` are stored *escaped* (`\t \n \r \x##`, escaped
/// backslash, etc.) and must be un-escaped to raw bytes before they enter the
/// trie. This mirrors llama.cpp's `llm_tokenizer_rwkv` /
/// `llama_unescape_rwkv_token` / `naive_trie` exactly, so the produced ids are
/// bit-identical to `llama-tokenize` on the same GGUF.
///
/// No BOS/EOS is added by default (llama.cpp's RWKV path adds none, regardless
/// of the `bos_token_id` declared in metadata).
#[derive(Clone)]
struct RwkvWorldTokenizer {
    /// Decode table: token id -> raw bytes (already un-escaped).
    id_to_bytes: Vec<Vec<u8>>,
    /// Encode trie root over raw bytes.
    trie: RwkvTrie,
    /// `tokenizer.ggml.eos_token_id`, surfaced for generation stop.
    eos_id: Option<u32>,
    /// `tokenizer.ggml.unknown_token_id`. llama.cpp emits this when a byte has
    /// no matching trie edge; World vocabs cover all 256 single bytes so it is
    /// effectively never hit, but we mirror the fallback faithfully.
    unk_id: Option<u32>,
}

impl RwkvWorldTokenizer {
    fn new(tokens: &[String], eos_id: Option<u32>, unk_id: Option<u32>) -> Self {
        let id_to_bytes: Vec<Vec<u8>> = tokens
            .iter()
            .map(|t| unescape_rwkv_token(t.as_bytes()))
            .collect();
        let mut trie = RwkvTrie::default();
        for (id, bytes) in id_to_bytes.iter().enumerate() {
            trie.insert(bytes, id as u32);
        }
        Self {
            id_to_bytes,
            trie,
            eos_id,
            unk_id,
        }
    }

    fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    /// Greedy longest-match tokenize. `_add_special` is accepted for signature
    /// parity with the other tokenizers; RWKV adds no BOS/EOS (matching
    /// llama.cpp), so it is intentionally ignored.
    fn encode(&self, text: &str, _add_special: bool) -> Vec<u32> {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut out = Vec::new();
        let mut position = 0usize;
        while position < len {
            // Walk the trie from `position`, remembering the last id whose path
            // terminates a vocab entry (longest match wins). Mirrors
            // `llm_tokenizer_rwkv_session::tokenize`.
            let mut node = self.trie.traverse_root(bytes[position]);
            if node.is_none() {
                // No edge for this byte: emit unk (if any) and advance one byte.
                if let Some(unk) = self.unk_id {
                    out.push(unk);
                }
                position += 1;
                continue;
            }
            let mut token_id = 0u32;
            let mut token_length = 0usize;
            let mut p = position;
            while let Some(n) = node {
                if n.has_value {
                    token_id = n.value;
                    token_length = p + 1;
                }
                p += 1;
                node = if p < len {
                    n.traverse(bytes[p])
                } else {
                    None
                };
            }
            out.push(token_id);
            // `token_length` is always > position here: the first byte matched a
            // trie edge, and World vocabs include every single byte, so at least
            // a length-1 match is recorded before the walk ends.
            position = token_length;
        }
        out
    }

    /// Concatenate each id's raw bytes, then lossily interpret as UTF-8. With
    /// `skip_special`, the eos token is dropped (it decodes to a control byte).
    fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if skip_special && Some(id) == self.eos_id {
                continue;
            }
            if let Some(b) = self.id_to_bytes.get(id as usize) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode a single token to its raw bytes, lossily as UTF-8. A multi-byte
    /// UTF-8 codepoint can straddle several World tokens, so a lone token may be
    /// an incomplete sequence and lossy-decode to the replacement char; the
    /// streaming caller reconstructs full text by concatenating the pieces.
    fn decode_one(&self, id: u32) -> String {
        match self.id_to_bytes.get(id as usize) {
            Some(b) => String::from_utf8_lossy(b).into_owned(),
            None => String::new(),
        }
    }
}

/// Un-escape a stored RWKV token string into its raw bytes. Faithful port of
/// llama.cpp's `llama_unescape_rwkv_token`: `\t`/`\n`/`\r` map to the control
/// bytes, `\x##` consumes two lowercase-or-digit hex chars, any other escaped
/// character is taken literally (so `\\` -> `\`, `\'` -> `'`), and unescaped
/// bytes pass through unchanged.
fn unescape_rwkv_token(escaped: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(escaped.len());
    let mut escaping = false;
    let mut hex_remaining: u8 = 0;
    let mut hex_acc: u8 = 0;
    for &c in escaped {
        if hex_remaining != 0 {
            let value = if c >= b'a' { c - b'a' + 10 } else { c - b'0' };
            hex_acc = (hex_acc << 4).wrapping_add(value);
            hex_remaining -= 1;
            if hex_remaining == 0 {
                output.push(hex_acc);
                hex_acc = 0;
            }
            continue;
        }
        if escaping {
            match c {
                b't' => output.push(b'\t'),
                b'n' => output.push(b'\n'),
                b'r' => output.push(b'\r'),
                b'x' => hex_remaining = 2,
                other => output.push(other),
            }
            escaping = false;
            continue;
        }
        if c == b'\\' {
            escaping = true;
            continue;
        }
        output.push(c);
    }
    output
}

/// Minimal byte-keyed trie matching llama.cpp's `naive_trie` for the RWKV path.
/// Children are kept sorted by byte for a binary-search `traverse`.
#[derive(Clone, Default)]
struct RwkvTrie {
    /// (byte, child) pairs, sorted by byte.
    children: Vec<(u8, RwkvTrie)>,
    has_value: bool,
    value: u32,
}

impl RwkvTrie {
    fn insert(&mut self, key: &[u8], value: u32) {
        match key.split_first() {
            None => {
                self.has_value = true;
                self.value = value;
            }
            Some((&first, rest)) => match self.children.binary_search_by_key(&first, |(b, _)| *b) {
                Ok(idx) => self.children[idx].1.insert(rest, value),
                Err(idx) => {
                    self.children.insert(idx, (first, RwkvTrie::default()));
                    self.children[idx].1.insert(rest, value);
                }
            },
        }
    }

    fn traverse(&self, byte: u8) -> Option<&RwkvTrie> {
        self.children
            .binary_search_by_key(&byte, |(b, _)| *b)
            .ok()
            .map(|idx| &self.children[idx].1)
    }

    fn traverse_root(&self, byte: u8) -> Option<&RwkvTrie> {
        self.traverse(byte)
    }
}

#[derive(Clone, Debug)]
struct SpmSymbol {
    prev: Option<usize>,
    next: Option<usize>,
    start: usize,
    len: usize,
}

#[derive(Clone, Debug)]
struct SpmBigram {
    left: usize,
    right: usize,
    score: f32,
    size: usize,
}

impl PartialEq for SpmBigram {
    fn eq(&self, other: &Self) -> bool {
        self.left == other.left
            && self.right == other.right
            && self.size == other.size
            && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for SpmBigram {}

impl PartialOrd for SpmBigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SpmBigram {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
    }
}

type BuiltTokenizer = (
    HfTokenizer,
    DecodeOneMode,
    Option<LlamaSpmTokenizer>,
    Option<RwkvWorldTokenizer>,
);

fn build_tokenizer(
    model: &str,
    tokens: &[String],
    merges: &[String],
    scores: &[f32],
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    unk_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
) -> Result<BuiltTokenizer> {
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
            let spm =
                LlamaSpmTokenizer::new(tokens, scores, bos_id, eos_id, unk_id, add_bos, add_eos)?;
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
            Ok((t, DecodeOneMode::SentencePiece, Some(spm), None))
        }
        // RWKV "World" vocab: raw-byte greedy-longest-match trie. Not BPE/SPM,
        // so the `tokenizers` crate cannot represent it — we carry a custom
        // `RwkvWorldTokenizer` and route all dispatch to it. `inner` is a stub
        // (an empty byte-level BPE) that is never exercised.
        "rwkv" => {
            let rwkv = RwkvWorldTokenizer::new(tokens, eos_id, unk_id);
            let stub = HfTokenizer::new(BPE::default());
            Ok((stub, DecodeOneMode::Hf, None, Some(rwkv)))
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
            // Register the GGUF's control tokens (`<|im_start|>`, `<|im_end|>`,
            // `<|endoftext|>`, …) as special so `encode` matches them as atomic
            // ids instead of shattering them into byte-level pieces. Without
            // this the chat template is mis-encoded and the model emits `<|>`
            // garbage. Tokens already in the BPE vocab keep their existing id.
            let mut specials: Vec<tokenizers::AddedToken> = Vec::new();
            for s in tokens {
                if is_special_token_str(s) {
                    specials.push(tokenizers::AddedToken::from(s.clone(), true));
                }
            }
            if !specials.is_empty() {
                t.add_special_tokens(&specials);
            }
            Ok((t, DecodeOneMode::Hf, None, None))
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
        let (tokenizer, mode, _spm, _rwkv) = build_tokenizer(
            "llama",
            &tokens,
            &[],
            &scores,
            Some(1),
            Some(2),
            Some(0),
            false,
            false,
        )
        .unwrap();

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
        let (inner, decode_one_mode, llama_spm, rwkv_world) = build_tokenizer(
            "llama",
            &tokens,
            &[],
            &scores,
            Some(1),
            Some(2),
            Some(0),
            false,
            false,
        )
        .unwrap();
        let tokenizer = Tokenizer {
            inner,
            bos_id: Some(1),
            eos_id: Some(2),
            pad_id: None,
            decode_one_mode,
            llama_spm,
            rwkv_world,
            special_ids: HashSet::new(),
            eog_ids: HashSet::new(),
        };

        assert_eq!(tokenizer.decode_one(4).unwrap(), " upon");
    }

    #[test]
    fn rwkv_unescape_handles_all_escape_forms() {
        // `\t \n \r`, `\x##` (lowercase hex), escaped backslash, and a literal
        // byte all round-trip exactly like llama.cpp's llama_unescape_rwkv_token.
        assert_eq!(unescape_rwkv_token(b"\\t"), vec![b'\t']);
        assert_eq!(unescape_rwkv_token(b"\\n\\n"), vec![b'\n', b'\n']);
        assert_eq!(unescape_rwkv_token(b"\\r"), vec![b'\r']);
        assert_eq!(unescape_rwkv_token(b"\\x00"), vec![0u8]);
        assert_eq!(unescape_rwkv_token(b"\\x1b"), vec![0x1bu8]);
        assert_eq!(unescape_rwkv_token(b"\\xff"), vec![0xffu8]);
        assert_eq!(unescape_rwkv_token(b"\\\\"), vec![b'\\']);
        assert_eq!(unescape_rwkv_token(b" "), vec![b' ']);
        assert_eq!(unescape_rwkv_token(b"abc"), b"abc".to_vec());
    }

    #[test]
    fn rwkv_world_greedy_longest_match() {
        // Minimal vocab: single bytes plus a multi-byte entry. Greedy
        // longest-match must prefer the 2-byte "ab" (id 3) over "a"+"b".
        let tokens = vec![
            "<s>".to_string(), // 0
            "a".to_string(),   // 1
            "b".to_string(),   // 2
            "ab".to_string(),  // 3
            "c".to_string(),   // 4
        ];
        let tok = RwkvWorldTokenizer::new(&tokens, Some(0), None);
        assert_eq!(tok.encode("abc", false), vec![3, 4]);
        assert_eq!(tok.encode("ba", false), vec![2, 1]);
        assert_eq!(tok.decode(&[3, 4], false), "abc");
        assert_eq!(tok.decode_one(3), "ab");
        assert_eq!(tok.vocab_size(), 5);
        // skip_special drops the eos id (0 here).
        assert_eq!(tok.decode(&[0, 1], true), "a");
    }
}
