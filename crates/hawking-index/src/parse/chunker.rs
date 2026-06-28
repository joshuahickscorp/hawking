//! cAST / by-symbol chunking (bible §4.7 "Chunking").
//!
//! We chunk by AST symbol, not fixed line windows: each top-level definition is a
//! chunk. A chunk that exceeds the embedding budget is split; small adjacent
//! siblings are greedily merged to fill the budget. Each chunk carries its byte
//! range, enclosing symbol, and a BLAKE3 content hash so unchanged chunks are
//! never re-embedded (the dominant incremental-embedding win).

use super::grammars::{GrammarRegistry, LangId};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tree_sitter::{Node, Parser};

/// A semantic chunk: a unit of code mapped to one embedding vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeChunk {
    /// Content-addressed id: BLAKE3 of the chunk text (hex).
    pub chunk_id: String,
    pub file: String,
    /// Optional symbol id this chunk corresponds to (if it's a single def).
    pub symbol: Option<String>,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: u32,
    pub end_line: u32,
    pub text: String,
}

/// Soft target for chunk size in characters (a proxy for the embedding model's
/// token budget; ~4 chars/token → ~512 tokens). Oversized chunks are split.
const MAX_CHUNK_CHARS: usize = 2000;
/// Chunks smaller than this are merge candidates with their neighbors.
const MIN_CHUNK_CHARS: usize = 200;

/// Chunk a file into cAST-style code chunks.
///
/// For known languages we chunk by top-level definition nodes; for unknown
/// languages (or parse failure) we fall back to a fixed-window split so the
/// semantic leg still has *something* to embed.
pub fn chunk_file(rel_path: &str, source: &str) -> Vec<CodeChunk> {
    let lang = LangId::from_path(Path::new(rel_path));
    if !lang.is_known() {
        return window_chunks(rel_path, source);
    }
    let Some(bundle) = GrammarRegistry::bundle(lang) else {
        return window_chunks(rel_path, source);
    };
    let mut parser = Parser::new();
    if parser.set_language(&bundle.language).is_err() {
        return window_chunks(rel_path, source);
    }
    let Some(tree) = parser.parse(source, None) else {
        return window_chunks(rel_path, source);
    };

    let src = source.as_bytes();
    let mut raw: Vec<(usize, usize)> = Vec::new();
    collect_def_spans(tree.root_node(), src, &mut raw);

    if raw.is_empty() {
        return window_chunks(rel_path, source);
    }
    raw.sort_by_key(|(s, _)| *s);

    // Split oversized, then merge small adjacent siblings (cAST).
    let mut split: Vec<(usize, usize)> = Vec::new();
    for (s, e) in raw {
        if e.saturating_sub(s) > MAX_CHUNK_CHARS {
            split_span(source, s, e, &mut split);
        } else {
            split.push((s, e));
        }
    }
    let merged = merge_small(&split);

    merged
        .into_iter()
        .filter_map(|(s, e)| make_chunk(rel_path, source, s, e))
        .collect()
}

/// Collect byte spans of the definition nodes we want as chunks (functions,
/// methods, classes, structs, enums, traits, impls). We take *top-level* defs
/// and methods, but not nested locals.
fn collect_def_spans(node: Node, _src: &[u8], out: &mut Vec<(usize, usize)>) {
    const DEF_KINDS: &[&str] = &[
        // rust
        "function_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "impl_item",
        "mod_item",
        "macro_definition",
        // python
        "function_definition",
        "class_definition",
        "decorated_definition",
        // typescript / js
        "class_declaration",
        "function_declaration",
        "method_definition",
        "interface_declaration",
        "enum_declaration",
        "lexical_declaration",
        "export_statement",
    ];

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if DEF_KINDS.contains(&child.kind()) {
            out.push((child.start_byte(), child.end_byte()));
            // For impl/class bodies, also surface inner methods as their own
            // chunks (so a big impl block isn't one giant chunk).
            if matches!(
                child.kind(),
                "impl_item" | "class_definition" | "class_declaration"
            ) {
                let mut inner = child.walk();
                for grand in child.children(&mut inner) {
                    collect_def_spans(grand, _src, out);
                }
            }
        } else {
            // Recurse one level into modules/exports to catch nested top-levels.
            if matches!(child.kind(), "mod_item" | "export_statement" | "block") {
                collect_def_spans(child, _src, out);
            }
        }
    }
}

/// Split an oversized span on line boundaries into <= MAX_CHUNK_CHARS pieces.
fn split_span(source: &str, start: usize, end: usize, out: &mut Vec<(usize, usize)>) {
    let slice = &source[start..end.min(source.len())];
    let mut cur = start;
    let mut acc = 0usize;
    let mut last_break = start;
    for (i, ch) in slice.char_indices() {
        acc += ch.len_utf8();
        if ch == '\n' && acc >= MAX_CHUNK_CHARS {
            let break_at = start + i + 1;
            out.push((last_break, break_at));
            last_break = break_at;
            acc = 0;
            cur = break_at;
        }
    }
    if cur < end {
        out.push((last_break, end));
    }
}

/// Greedily merge adjacent small spans up to the budget (cAST merge step).
fn merge_small(spans: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for &(s, e) in spans {
        if let Some(last) = out.last_mut() {
            let last_len = last.1 - last.0;
            let this_len = e - s;
            if last_len < MIN_CHUNK_CHARS && (last_len + this_len) <= MAX_CHUNK_CHARS {
                last.1 = e;
                continue;
            }
        }
        out.push((s, e));
    }
    out
}

fn make_chunk(rel_path: &str, source: &str, start: usize, end: usize) -> Option<CodeChunk> {
    let end = end.min(source.len());
    if start >= end {
        return None;
    }
    let text = source.get(start..end)?.to_string();
    if text.trim().is_empty() {
        return None;
    }
    let chunk_id = blake3::hash(text.as_bytes()).to_hex().to_string();
    let start_line = source[..start].bytes().filter(|b| *b == b'\n').count() as u32 + 1;
    let end_line = source[..end].bytes().filter(|b| *b == b'\n').count() as u32 + 1;
    Some(CodeChunk {
        chunk_id,
        file: rel_path.to_string(),
        symbol: None,
        start_byte: start,
        end_byte: end,
        start_line,
        end_line,
        text,
    })
}

/// Fixed-window fallback for unparseable files (so the semantic leg still works).
fn window_chunks(rel_path: &str, source: &str) -> Vec<CodeChunk> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let bytes = source.as_bytes();
    while start < bytes.len() {
        let mut end = (start + MAX_CHUNK_CHARS).min(bytes.len());
        // snap to a char boundary
        while end < bytes.len() && !source.is_char_boundary(end) {
            end += 1;
        }
        if let Some(c) = make_chunk(rel_path, source, start, end) {
            out.push(c);
        }
        start = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_rust_by_definition() {
        let src = "pub fn alpha() {\n    let x = 1;\n}\n\npub fn beta() {\n    let y = 2;\n}\n";
        let chunks = chunk_file("m.rs", src);
        assert!(chunks.len() >= 1);
        // each chunk is content-addressed and non-empty
        for c in &chunks {
            assert_eq!(c.chunk_id.len(), 64);
            assert!(!c.text.trim().is_empty());
        }
        let joined: String = chunks.iter().map(|c| c.text.clone()).collect();
        assert!(joined.contains("alpha"));
        assert!(joined.contains("beta"));
    }

    #[test]
    fn identical_text_yields_identical_chunk_id() {
        let a = chunk_file("a.rs", "pub fn f() { body(); }");
        let b = chunk_file("b.rs", "pub fn f() { body(); }");
        assert_eq!(a[0].chunk_id, b[0].chunk_id);
    }

    #[test]
    fn unknown_language_uses_window_fallback() {
        let big = "x".repeat(5000);
        let chunks = chunk_file("blob.dat", &big);
        assert!(chunks.len() >= 2, "oversized unknown file should split");
    }
}
