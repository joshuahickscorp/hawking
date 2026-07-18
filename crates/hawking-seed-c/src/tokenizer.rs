//! GPT-2 byte-level BPE tokenizer, reimplemented from the GGUF-embedded vocab + merges. Encode drives
//! the prompt into ids; decode maps ids back to bytes -> UTF-8. Byte<->visible-glyph mapping is the
//! canonical GPT-2 `bytes_to_unicode`. Pre-tokenization follows the GPT-2 regex semantics (contractions,
//! optional single leading space + letter/number/other runs, whitespace runs with the trailing-space
//! lookahead). No `regex`/`tokenizers` crate — this is self-contained.

use crate::gguf::GgufFile;
use crate::{Error, Result};
use std::collections::{HashMap, HashSet};

pub struct Tokenizer {
    id_to_tok: Vec<String>,
    vocab: HashMap<String, u32>,
    merge_rank: HashMap<(String, String), u32>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    special_ids: HashSet<u32>,
    eos: u32,
}

/// Canonical GPT-2 bytes_to_unicode: printable ranges map to themselves; the rest to U+0100+k.
fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = Vec::new();
    for b in 0x21..=0x7E {
        bs.push(b);
    }
    for b in 0xA1..=0xAC {
        bs.push(b);
    }
    for b in 0xAE..=0xFF {
        bs.push(b);
    }
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut map = ['\0'; 256];
    for (k, &b) in bs.iter().enumerate() {
        map[b as usize] = char::from_u32(cs[k]).unwrap();
    }
    map
}

impl Tokenizer {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let model = g.meta_str("tokenizer.ggml.model").unwrap_or("gpt2");
        if model != "gpt2" && model != "llama" {
            return Err(Error::Tokenizer(format!("unsupported tokenizer model {model}")));
        }
        let tokens = g.meta_str_array("tokenizer.ggml.tokens")?;
        let merges = g.meta_str_array("tokenizer.ggml.merges")?;

        let id_to_tok: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
        let mut vocab = HashMap::with_capacity(id_to_tok.len());
        for (i, t) in id_to_tok.iter().enumerate() {
            vocab.insert(t.clone(), i as u32);
        }
        let mut merge_rank = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let mut it = m.splitn(2, ' ');
            if let (Some(a), Some(b)) = (it.next(), it.next()) {
                merge_rank.insert((a.to_string(), b.to_string()), rank as u32);
            }
        }

        let byte_to_char = bytes_to_unicode();
        let mut char_to_byte = HashMap::with_capacity(256);
        for (b, &c) in byte_to_char.iter().enumerate() {
            char_to_byte.insert(c, b as u8);
        }

        let mut special_ids = HashSet::new();
        for k in [
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.unknown_token_id",
        ] {
            if let Ok(v) = g.meta_u32(k) {
                special_ids.insert(v);
            }
        }
        let eos = g.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);

        Ok(Tokenizer {
            id_to_tok,
            vocab,
            merge_rank,
            byte_to_char,
            char_to_byte,
            special_ids,
            eos,
        })
    }

    pub fn eos_id(&self) -> u32 {
        self.eos
    }

    /// GPT-2 pre-tokenization (regex semantics, no regex engine). Returns substrings of `text`.
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let ch: Vec<char> = text.chars().collect();
        let n = ch.len();
        let mut out = Vec::new();
        let mut i = 0usize;
        let is_letter = |c: char| c.is_alphabetic();
        let is_num = |c: char| c.is_numeric();
        let is_ws = |c: char| c.is_whitespace();
        let is_other = |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric();
        while i < n {
            // contractions: 's 't 're 've 'm 'll 'd
            if ch[i] == '\'' && i + 1 < n {
                let rest: String = ch[i + 1..].iter().take(2).collect();
                let mut matched = None;
                for pat in ["re", "ve", "ll", "s", "t", "m", "d"] {
                    if rest.starts_with(pat) {
                        matched = Some(pat);
                        break;
                    }
                }
                if let Some(pat) = matched {
                    let mut s = String::from("'");
                    s.push_str(pat);
                    i += 1 + pat.len();
                    out.push(s);
                    continue;
                }
            }
            let has_space = ch[i] == ' ';
            let nxt = if has_space && i + 1 < n { Some(ch[i + 1]) } else { None };
            // ` ?\p{L}+`
            if (is_letter(ch[i])) || (has_space && nxt.map(is_letter).unwrap_or(false)) {
                let start = i;
                if has_space {
                    i += 1;
                }
                while i < n && is_letter(ch[i]) {
                    i += 1;
                }
                out.push(ch[start..i].iter().collect());
                continue;
            }
            // ` ?\p{N}+`
            if (is_num(ch[i])) || (has_space && nxt.map(is_num).unwrap_or(false)) {
                let start = i;
                if has_space {
                    i += 1;
                }
                while i < n && is_num(ch[i]) {
                    i += 1;
                }
                out.push(ch[start..i].iter().collect());
                continue;
            }
            // ` ?[^\s\p{L}\p{N}]+`
            if (is_other(ch[i])) || (has_space && nxt.map(is_other).unwrap_or(false)) {
                let start = i;
                if has_space {
                    i += 1;
                }
                while i < n && is_other(ch[i]) {
                    i += 1;
                }
                out.push(ch[start..i].iter().collect());
                continue;
            }
            // whitespace run with trailing-space lookahead (`\s+(?!\S)` then `\s+`)
            if is_ws(ch[i]) {
                let start = i;
                while i < n && is_ws(ch[i]) {
                    i += 1;
                }
                // if followed by a non-ws char and the last ws is a space, leave it for that token
                if i < n && i - start >= 1 && ch[i - 1] == ' ' {
                    i -= 1;
                }
                if i > start {
                    out.push(ch[start..i].iter().collect());
                }
                continue;
            }
            // fallthrough (should not happen) — consume one char
            out.push(ch[i..i + 1].iter().collect());
            i += 1;
        }
        out
    }

    /// BPE merge a sequence of single-char symbols into vocab pieces.
    fn bpe(&self, mut symbols: Vec<String>) -> Vec<String> {
        if symbols.len() < 2 {
            return symbols;
        }
        loop {
            let mut best: Option<(usize, u32)> = None;
            for k in 0..symbols.len() - 1 {
                if let Some(&r) = self.merge_rank.get(&(symbols[k].clone(), symbols[k + 1].clone())) {
                    if best.map(|(_, br)| r < br).unwrap_or(true) {
                        best = Some((k, r));
                    }
                }
            }
            let Some((k, _)) = best else { break };
            let merged = format!("{}{}", symbols[k], symbols[k + 1]);
            symbols.splice(k..k + 2, [merged]);
        }
        symbols
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for piece in self.pre_tokenize(text) {
            // map the piece's UTF-8 bytes to visible glyphs, one symbol per byte
            let symbols: Vec<String> =
                piece.bytes().map(|b| self.byte_to_char[b as usize].to_string()).collect();
            for sym in self.bpe(symbols) {
                match self.vocab.get(&sym) {
                    Some(&id) => ids.push(id),
                    None => {
                        return Err(Error::Tokenizer(format!("no vocab entry for symbol {sym:?}")))
                    }
                }
            }
        }
        Ok(ids)
    }

    /// Decode ids to text: collect the byte-level glyphs' underlying bytes, then UTF-8. Special ids
    /// (bos/eos/pad/unk) render as empty, like the predecessor.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if self.special_ids.contains(&id) {
                continue;
            }
            if let Some(tok) = self.id_to_tok.get(id as usize) {
                for c in tok.chars() {
                    if let Some(&b) = self.char_to_byte.get(&c) {
                        bytes.push(b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
