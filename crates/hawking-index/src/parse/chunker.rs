//! cAST / by-symbol chunking (bible §4.7 "Chunking").
//!
//! We chunk by AST symbol, not fixed line windows: each top-level definition is a
//! chunk. A chunk that exceeds the embedding budget is split; small adjacent
//! siblings are greedily merged to fill the budget. Each chunk carries its byte
//! range, enclosing symbol, and a BLAKE3 content hash so unchanged chunks are
//! never re-embedded (the dominant incremental-embedding win).

use super::grammars::{GrammarRegistry, LangId};
use super::{scip_symbol_id, SymKind};
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

    // Build a table of every definition's byte span → its SCIP id, so each chunk
    // can be tagged with the symbol whose span encloses it (bible §4.7). We walk
    // the *whole* tree (not just chunk-able defs) so that, e.g., a method chunk
    // inside a class resolves to the method symbol rather than the class.
    let mut def_symbols: Vec<DefSpan> = Vec::new();
    collect_def_symbols(tree.root_node(), src, lang, rel_path, &mut def_symbols);

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
        .filter_map(|(s, e)| {
            let mut chunk = make_chunk(rel_path, source, s, e)?;
            chunk.symbol = enclosing_symbol(&def_symbols, s, e);
            Some(chunk)
        })
        .collect()
}

/// A definition's byte span paired with its SCIP id.
struct DefSpan {
    start: usize,
    end: usize,
    symbol_id: String,
}

/// The SCIP id of the definition that owns the chunk (bible §4.7: "the symbol
/// whose span contains the chunk").
///
/// A chunk usually IS one definition, but cAST may split an oversized def or
/// merge small siblings; the chunk then maps to its *leading* definition — the
/// smallest def whose span contains the chunk's start byte. For a nested form
/// (a method inside a class) the inner, smaller def wins, so a method chunk maps
/// to the method rather than the enclosing class. `None` only when nothing
/// covers the start (e.g. a window-fallback fragment).
fn enclosing_symbol(defs: &[DefSpan], start: usize, end: usize) -> Option<String> {
    defs.iter()
        // Prefer the smallest def fully containing the chunk (the clean 1:1 case),
        // then fall back to the smallest def containing just the chunk's start
        // (merged/split case).
        .filter(|d| d.start <= start && d.end >= end)
        .min_by_key(|d| d.end - d.start)
        .or_else(|| {
            defs.iter()
                .filter(|d| d.start <= start && d.end > start)
                .min_by_key(|d| d.end - d.start)
        })
        .map(|d| d.symbol_id.clone())
}

/// Walk the tree collecting (byte-span, SCIP id) for every named definition we
/// can attach a symbol to. Mirrors `parse::extract_with_bundle`'s kind mapping so
/// the ids are byte-for-byte the same as the symbols stored in the index — that's
/// what lets a retrieval hit map back to a symbol.
fn collect_def_symbols(
    node: Node,
    src: &[u8],
    lang: LangId,
    rel_path: &str,
    out: &mut Vec<DefSpan>,
) {
    if let Some((name, kind)) = def_name_and_kind(node, src) {
        out.push(DefSpan {
            start: node.start_byte(),
            end: node.end_byte(),
            symbol_id: scip_symbol_id(lang, rel_path, &name, kind),
        });
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_def_symbols(child, src, lang, rel_path, out);
    }
}

/// Map a definition node to its (name, kind), or `None` if it isn't a named
/// definition. Covers the Rust / Python / TS-JS forms the index emits symbols for.
fn def_name_and_kind(node: Node, src: &[u8]) -> Option<(String, SymKind)> {
    let kind = match node.kind() {
        // rust
        "function_item" => SymKind::Function,
        "struct_item" => SymKind::Struct,
        "enum_item" => SymKind::Enum,
        "trait_item" => SymKind::Trait,
        "mod_item" => SymKind::Module,
        "macro_definition" => SymKind::Macro,
        "type_item" => SymKind::TypeAlias,
        "const_item" | "static_item" => SymKind::Constant,
        // python
        "function_definition" => SymKind::Function,
        "class_definition" => SymKind::Class,
        // typescript / javascript
        "function_declaration" | "generator_function_declaration" => SymKind::Function,
        "method_definition" => SymKind::Method,
        "class_declaration" => SymKind::Class,
        "interface_declaration" => SymKind::Interface,
        "enum_declaration" => SymKind::Enum,
        "type_alias_declaration" => SymKind::TypeAlias,
        _ => return None,
    };
    let name = node
        .child_by_field_name("name")
        .and_then(|n| n.utf8_text(src).ok())
        .map(|s| s.to_string())?;
    if name.is_empty() {
        return None;
    }
    Some((name, kind))
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
        assert!(!chunks.is_empty());
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
    fn chunk_symbol_is_enclosing_def_scip_id() {
        use crate::parse::parse_source;
        // Two top-level fns: each chunk must carry the SCIP id of the def it covers,
        // and that id must equal the symbol the parser emits (so hits map back).
        let src = "pub fn alpha() {\n    work();\n}\n\npub fn beta() {\n    other();\n}\n";
        let chunks = chunk_file("m.rs", src);
        assert!(!chunks.is_empty());
        // every chunk that wraps a single def must have a symbol (not None)
        let with_sym: Vec<_> = chunks.iter().filter(|c| c.symbol.is_some()).collect();
        assert!(
            !with_sym.is_empty(),
            "expected at least one chunk tagged with its enclosing symbol"
        );

        // The symbol id must be byte-identical to a parsed symbol's qualified_name.
        let parsed = parse_source("m.rs", src);
        let parsed_ids: std::collections::HashSet<&str> =
            parsed.symbols.iter().map(|s| s.qualified_name.as_str()).collect();
        for c in &with_sym {
            let sym = c.symbol.as_deref().unwrap();
            assert!(
                parsed_ids.contains(sym),
                "chunk symbol {sym:?} must match a parsed symbol id; have {parsed_ids:?}"
            );
        }

        // Specifically, the chunk covering `alpha` resolves to alpha's id.
        let alpha_id = scip_symbol_id(LangId::Rust, "m.rs", "alpha", SymKind::Function);
        assert!(
            chunks.iter().any(|c| c.symbol.as_deref() == Some(alpha_id.as_str())),
            "a chunk should map to alpha's SCIP id {alpha_id:?}"
        );
    }

    #[test]
    fn chunk_symbol_resolves_to_inner_method_not_class() {
        // A method chunk inside a class must resolve to the *method* (smallest
        // enclosing def), not the class. Make the method body large enough that it
        // survives as its own chunk (cAST won't merge a >MIN-size sibling).
        let body: String = (0..40)
            .map(|i| format!("        line_{i}();\n"))
            .collect();
        let src = format!("class Greeter {{\n    render() {{\n{body}    }}\n}}\n");
        let chunks = chunk_file("ui.ts", &src);
        let method_id = scip_symbol_id(LangId::TypeScript, "ui.ts", "render", SymKind::Method);
        // there must be a chunk that maps to the inner method's id (smallest
        // enclosing def), proving nested resolution prefers the method over the class.
        assert!(
            chunks.iter().any(|c| c.symbol.as_deref() == Some(method_id.as_str())),
            "expected a chunk mapped to inner method {method_id:?}; got {:?}",
            chunks.iter().map(|c| c.symbol.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn unknown_language_uses_window_fallback() {
        let big = "x".repeat(5000);
        let chunks = chunk_file("blob.dat", &big);
        assert!(chunks.len() >= 2, "oversized unknown file should split");
    }
}
