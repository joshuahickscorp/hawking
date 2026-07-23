//! The parsing layer (bible §4.2, §4.3).
//!
//! Real tree-sitter parsing replacing the old `simple_definition` prefix scanner.
//! For every file we run the grammar's `tags.scm` query and emit BOTH definitions
//! and references with SCIP-style path-scoped symbol IDs, so `references()` stops
//! returning empty and the reverse-reference / blast-radius moat is reachable.

pub mod chunker;
pub mod grammars;

pub use chunker::{chunk_file, CodeChunk};
pub use grammars::{GrammarBundle, GrammarRegistry, LangId};

use crate::graph::{Occurrence, Symbol};
use hide_core::types::TextRange;
use std::path::Path;
use tree_sitter::{Node, Parser, QueryCursor, StreamingIterator};

/// The roles an occurrence can play (mirrors SCIP role bits; stored as a string
/// on `Occurrence::role` for backward-compat with the existing query API).
pub const ROLE_DEFINITION: &str = "definition";
pub const ROLE_REFERENCE: &str = "reference";

/// One parsed file's extracted facts.
#[derive(Debug, Clone, Default)]
pub struct ParseOutput {
    pub lang: Option<LangId>,
    pub symbols: Vec<Symbol>,
    pub occurrences: Vec<Occurrence>,
    /// Byte ranges of ERROR / MISSING nodes (for the health surface).
    pub error_spans: Vec<(usize, usize)>,
    /// True if the whole file failed to parse into anything structural.
    pub unparseable: bool,
}

/// SCIP-style structured symbol ID.
///
/// Format (Hawking dialect): `hawking <lang> <repo_rel_path> <descriptor>`, e.g.
/// `hawking rust src/model/qwen.rs forward_token().`. The descriptor suffix
/// encodes the kind: `().` method/function, `#` type, `.` term, `/` module.
/// IDs are stable across edits as long as the qualified name is stable, so a
/// reference resolves to its definition by string equality.
pub fn scip_symbol_id(lang: LangId, rel_path: &str, name: &str, kind: SymKind) -> String {
    let suffix = match kind {
        SymKind::Function | SymKind::Method => "().",
        SymKind::Class | SymKind::Struct | SymKind::Enum | SymKind::Trait | SymKind::Interface => {
            "#"
        }
        SymKind::Module => "/",
        SymKind::Macro => "!",
        SymKind::Constant | SymKind::Field | SymKind::TypeAlias | SymKind::Unknown => ".",
    };
    format!("hawking {} {} {}{}", lang.as_str(), rel_path, name, suffix)
}

/// Symbol kinds (a superset of what `tags.scm` distinguishes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Trait,
    Interface,
    Module,
    Macro,
    Constant,
    Field,
    TypeAlias,
    Unknown,
}

impl SymKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SymKind::Function => "function",
            SymKind::Method => "method",
            SymKind::Class => "class",
            SymKind::Struct => "struct",
            SymKind::Enum => "enum",
            SymKind::Trait => "trait",
            SymKind::Interface => "interface",
            SymKind::Module => "module",
            SymKind::Macro => "macro",
            SymKind::Constant => "constant",
            SymKind::Field => "field",
            SymKind::TypeAlias => "type_alias",
            SymKind::Unknown => "symbol",
        }
    }

    /// Map a `tags.scm` capture name (e.g. `definition.function`) to a kind.
    fn from_capture(capture: &str) -> Option<SymKind> {
        let suffix = capture.strip_prefix("definition.")?;
        Some(match suffix {
            "function" => SymKind::Function,
            "method" => SymKind::Method,
            "class" => SymKind::Class,
            "struct" => SymKind::Struct,
            "enum" => SymKind::Enum,
            "trait" => SymKind::Trait,
            "interface" => SymKind::Interface,
            "module" => SymKind::Module,
            "macro" => SymKind::Macro,
            "constant" => SymKind::Constant,
            "field" => SymKind::Field,
            "type" => SymKind::TypeAlias,
            _ => SymKind::Unknown,
        })
    }
}

/// Parse a file's source and extract symbols + occurrences via tree-sitter.
///
/// `rel_path` is the workspace-relative path (used for SCIP IDs and provenance).
/// Returns an empty/unparseable `ParseOutput` for unknown languages — the caller
/// falls back to lexical-only indexing for those.
pub fn parse_source(rel_path: &str, source: &str) -> ParseOutput {
    let lang = LangId::from_path(Path::new(rel_path));
    if !lang.is_known() {
        return ParseOutput {
            lang: Some(lang),
            unparseable: true,
            ..Default::default()
        };
    }
    let bundle = match GrammarRegistry::bundle(lang) {
        Some(b) => b,
        None => {
            return ParseOutput {
                lang: Some(lang),
                unparseable: true,
                ..Default::default()
            }
        }
    };
    extract_with_bundle(&bundle, rel_path, source)
}

fn extract_with_bundle(bundle: &GrammarBundle, rel_path: &str, source: &str) -> ParseOutput {
    let mut parser = Parser::new();
    if parser.set_language(&bundle.language).is_err() {
        return ParseOutput {
            lang: Some(bundle.lang),
            unparseable: true,
            ..Default::default()
        };
    }
    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => {
            return ParseOutput {
                lang: Some(bundle.lang),
                unparseable: true,
                ..Default::default()
            }
        }
    };

    let mut out = ParseOutput {
        lang: Some(bundle.lang),
        ..Default::default()
    };

    // Collect ERROR/MISSING spans (damage localization).
    collect_error_spans(tree.root_node(), &mut out.error_spans);

    let src_bytes = source.as_bytes();
    let query = &bundle.tags_query;
    let capture_names = query.capture_names();

    // Locate the `@name` capture index once.
    let name_idx = capture_names.iter().position(|n| *n == "name");

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, tree.root_node(), src_bytes);

    while let Some(m) = matches.next() {
        // Find the @name node and the role/kind capture in this match.
        let mut name_text: Option<String> = None;
        let mut name_node: Option<Node> = None;
        let mut role_capture: Option<&str> = None;

        for cap in m.captures {
            let cap_name = capture_names[cap.index as usize];
            if Some(cap.index as usize) == name_idx {
                name_node = Some(cap.node);
                name_text = cap.node.utf8_text(src_bytes).ok().map(|s| s.to_string());
            } else if cap_name.starts_with("definition.") || cap_name.starts_with("reference.") {
                role_capture = Some(cap_name);
            }
        }

        let (Some(name), Some(node), Some(role)) = (name_text, name_node, role_capture) else {
            continue;
        };
        if name.is_empty() {
            continue;
        }

        let range = node_to_range(&node);
        if role.starts_with("definition.") {
            let kind = SymKind::from_capture(role).unwrap_or(SymKind::Unknown);
            let symbol_id = scip_symbol_id(bundle.lang, rel_path, &name, kind);
            out.symbols.push(Symbol {
                qualified_name: symbol_id.clone(),
                name: name.clone(),
                kind: kind.as_str().to_string(),
                file: rel_path.to_string(),
            });
            out.occurrences.push(Occurrence {
                symbol: symbol_id,
                file: rel_path.to_string(),
                range: Some(range),
                role: ROLE_DEFINITION.to_string(),
            });
        } else {
            // A reference. We don't yet know which definition it binds to (that's
            // tier-1/2 resolution); store it keyed by the *bare name* so callers
            // can resolve by name equality against defs. We record a name-scoped
            // reference occurrence so `references()` returns real data.
            out.occurrences.push(Occurrence {
                symbol: name.clone(),
                file: rel_path.to_string(),
                range: Some(range),
                role: ROLE_REFERENCE.to_string(),
            });
        }
    }

    out
}

fn node_to_range(node: &Node) -> TextRange {
    let s = node.start_position();
    let e = node.end_position();
    TextRange {
        start_line: s.row as u32 + 1,
        start_col: s.column as u32 + 1,
        end_line: e.row as u32 + 1,
        end_col: e.column as u32 + 1,
    }
}

fn collect_error_spans(node: Node, out: &mut Vec<(usize, usize)>) {
    if node.is_error() || node.is_missing() {
        out.push((node.start_byte(), node.end_byte()));
        // Don't recurse into an error subtree; the span already covers it.
        return;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_error_spans(child, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_rust_definitions_and_references() {
        let src = r#"
pub struct Engine {
    name: String,
}

pub fn run_engine() {
    helper();
}

fn helper() {}
"#;
        let out = parse_source("src/engine.rs", src);
        assert_eq!(out.lang, Some(LangId::Rust));
        assert!(!out.unparseable);

        // definitions present: Engine (struct), run_engine + helper (fn)
        let def_names: Vec<_> = out.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(def_names.contains(&"Engine"), "got {def_names:?}");
        assert!(def_names.contains(&"run_engine"));
        assert!(def_names.contains(&"helper"));

        // a reference to `helper` must exist (so references() is non-empty)
        let refs: Vec<_> = out
            .occurrences
            .iter()
            .filter(|o| o.role == ROLE_REFERENCE)
            .map(|o| o.symbol.as_str())
            .collect();
        assert!(
            refs.contains(&"helper"),
            "expected ref to helper, got {refs:?}"
        );
    }

    #[test]
    fn scip_ids_are_stable_and_kind_scoped() {
        let id_fn = scip_symbol_id(LangId::Rust, "a.rs", "foo", SymKind::Function);
        let id_struct = scip_symbol_id(LangId::Rust, "a.rs", "Foo", SymKind::Struct);
        assert!(id_fn.ends_with("foo()."));
        assert!(id_struct.ends_with("Foo#"));
        // stable across calls
        assert_eq!(
            id_fn,
            scip_symbol_id(LangId::Rust, "a.rs", "foo", SymKind::Function)
        );
    }

    #[test]
    fn typescript_extracts_defs_and_refs() {
        // Ordinary TS source: a function, a class with a method, and call sites.
        // The bundled grammar tags.scm would yield ZERO here (signature-only).
        let src = r#"
function greet(name: string): string {
    return format(name);
}

class Greeter {
    render(): void {
        greet("world");
    }
}

const formatted = format("x");
"#;
        let out = parse_source("ui.ts", src);
        assert_eq!(out.lang, Some(LangId::TypeScript));
        assert!(!out.unparseable);

        let def_names: Vec<_> = out.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(def_names.contains(&"greet"), "got defs {def_names:?}");
        assert!(def_names.contains(&"Greeter"), "got defs {def_names:?}");
        assert!(def_names.contains(&"render"), "got defs {def_names:?}");
        assert!(
            !out.symbols.is_empty(),
            "TS source must yield non-empty definitions"
        );

        let refs: Vec<_> = out
            .occurrences
            .iter()
            .filter(|o| o.role == ROLE_REFERENCE)
            .map(|o| o.symbol.as_str())
            .collect();
        assert!(
            !refs.is_empty(),
            "TS source must yield non-empty references, got {refs:?}"
        );
        assert!(
            refs.contains(&"greet"),
            "expected call ref to greet: {refs:?}"
        );
        assert!(
            refs.contains(&"format"),
            "expected call ref to format: {refs:?}"
        );
    }

    #[test]
    fn javascript_extracts_defs_and_refs() {
        // Plain JS (no types) routed through the TS superset grammar: a function,
        // a class + method, an arrow-fn const, and call sites.
        let src = r#"
function add(a, b) {
    return compute(a, b);
}

class Calculator {
    run() {
        add(1, 2);
    }
}

const square = (n) => add(n, n);
"#;
        let out = parse_source("calc.js", src);
        assert_eq!(out.lang, Some(LangId::TypeScript));
        assert!(!out.unparseable);

        let def_names: Vec<_> = out.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(def_names.contains(&"add"), "got defs {def_names:?}");
        assert!(def_names.contains(&"Calculator"), "got defs {def_names:?}");
        assert!(def_names.contains(&"run"), "got defs {def_names:?}");
        assert!(
            def_names.contains(&"square"),
            "arrow-fn const def: {def_names:?}"
        );

        let refs: Vec<_> = out
            .occurrences
            .iter()
            .filter(|o| o.role == ROLE_REFERENCE)
            .map(|o| o.symbol.as_str())
            .collect();
        assert!(
            !refs.is_empty(),
            "JS source must yield non-empty references, got {refs:?}"
        );
        assert!(
            refs.contains(&"compute"),
            "expected call ref to compute: {refs:?}"
        );
        assert!(refs.contains(&"add"), "expected call ref to add: {refs:?}");
    }

    #[test]
    fn python_extracts_class_and_function() {
        let src = "class Widget:\n    def render(self):\n        draw()\n\ndef draw():\n    pass\n";
        let out = parse_source("ui.py", src);
        let names: Vec<_> = out.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Widget"));
        assert!(names.contains(&"draw"));
    }

    #[test]
    fn records_error_spans_for_broken_code() {
        let src = "pub fn good() {}\npub fn broken( {\n";
        let out = parse_source("x.rs", src);
        // good() still extracted despite the broken neighbor
        assert!(out.symbols.iter().any(|s| s.name == "good"));
        assert!(
            !out.error_spans.is_empty(),
            "expected an ERROR/MISSING span"
        );
    }

    #[test]
    fn unknown_language_is_unparseable() {
        let out = parse_source("data.bin", "\u{0}\u{1}garbage");
        assert!(out.unparseable);
        assert!(out.symbols.is_empty());
    }
}
