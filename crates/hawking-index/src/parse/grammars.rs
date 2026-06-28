//! Grammar registry: maps a language to its tree-sitter `Language` + `tags.scm`.
//!
//! The core set is compiled in (statically linked parsers). Tier-0 tag extraction
//! works immediately for every registered language (bible §4.2, §7.1).

use serde::{Deserialize, Serialize};
use std::path::Path;
use tree_sitter::{Language, Query};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LangId {
    Rust,
    Python,
    TypeScript,
    /// File with no registered grammar — lexical-only (trigram over raw bytes).
    Unknown,
}

impl LangId {
    pub fn as_str(self) -> &'static str {
        match self {
            LangId::Rust => "rust",
            LangId::Python => "python",
            LangId::TypeScript => "typescript",
            LangId::Unknown => "unknown",
        }
    }

    /// Detect from file extension (the cheap, reliable path).
    pub fn from_path(path: &Path) -> LangId {
        match path.extension().and_then(|e| e.to_str()) {
            Some("rs") => LangId::Rust,
            Some("py" | "pyi") => LangId::Python,
            Some("ts" | "tsx" | "mts" | "cts" | "js" | "jsx" | "mjs" | "cjs") => {
                LangId::TypeScript
            }
            _ => LangId::Unknown,
        }
    }

    pub fn is_known(self) -> bool {
        !matches!(self, LangId::Unknown)
    }
}

/// Custom TypeScript/JavaScript `tags.scm`.
///
/// The grammar's *bundled* `queries/tags.scm` only matches ambient/signature
/// declarations (`function_signature`, `method_signature`, `abstract_*`,
/// `interface_declaration`, `module`) — so ordinary `.ts`/`.js` source (a plain
/// `function foo() {}`, `class Bar {}`, a method body, a call site) yields ZERO
/// defs/refs. This query covers the concrete-syntax forms that real source code
/// actually uses, plus reference forms (call sites, `new`, ident uses), so the
/// reverse-reference / blast-radius moat works on TS and JS. Capture names match
/// the `definition.<kind>` / `reference.<kind>` / `@name` contract that
/// `parse::extract_with_bundle` consumes. The TypeScript grammar is a superset of
/// JavaScript, so the same query drives both (`.js`/`.jsx`/`.mjs` route here too).
pub const TS_TAGS_QUERY: &str = r#"
; ---- definitions ----

(function_declaration
  name: (identifier) @name) @definition.function

(generator_function_declaration
  name: (identifier) @name) @definition.function

(class_declaration
  name: (type_identifier) @name) @definition.class

(method_definition
  name: (property_identifier) @name) @definition.method

; arrow-fn / function-expr bound to a const/let/var: `const f = (..) => ..`
(variable_declarator
  name: (identifier) @name
  value: [(arrow_function) (function_expression)]) @definition.function

; interfaces / type aliases / enums (TS)
(interface_declaration
  name: (type_identifier) @name) @definition.interface

(type_alias_declaration
  name: (type_identifier) @name) @definition.type

(enum_declaration
  name: (identifier) @name) @definition.enum

; ---- references ----

; direct call: `foo(..)`
(call_expression
  function: (identifier) @name) @reference.call

; method/qualified call: `obj.foo(..)` — credit the property name
(call_expression
  function: (member_expression
    property: (property_identifier) @name)) @reference.call

; `new Thing(..)`
(new_expression
  constructor: (identifier) @name) @reference.class
"#;

/// A compiled grammar bundle. Holds the tree-sitter `Language` and a compiled
/// `tags.scm` `Query` (defs + refs + name captures).
pub struct GrammarBundle {
    pub lang: LangId,
    pub language: Language,
    pub tags_query: Query,
}

impl GrammarBundle {
    fn build(lang: LangId, language: Language, tags_src: &str) -> Option<Self> {
        let tags_query = Query::new(&language, tags_src).ok()?;
        Some(Self {
            lang,
            language,
            tags_query,
        })
    }
}

/// Static registry. Bundles are built lazily on first request and cached in the
/// caller (see `parse::SymbolExtractor`). Grammars themselves are cheap to clone
/// (an `Arc` internally in tree-sitter), but `Query` compilation is not — so we
/// compile once per registry instance.
pub struct GrammarRegistry;

impl GrammarRegistry {
    pub fn bundle(lang: LangId) -> Option<GrammarBundle> {
        match lang {
            LangId::Rust => GrammarBundle::build(
                lang,
                tree_sitter_rust::LANGUAGE.into(),
                tree_sitter_rust::TAGS_QUERY,
            ),
            LangId::Python => GrammarBundle::build(
                lang,
                tree_sitter_python::LANGUAGE.into(),
                tree_sitter_python::TAGS_QUERY,
            ),
            LangId::TypeScript => GrammarBundle::build(
                lang,
                tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
                // Custom query (the bundled `TAGS_QUERY` only matches ambient
                // signatures, yielding ZERO defs/refs on ordinary `.ts`/`.js`).
                TS_TAGS_QUERY,
            ),
            LangId::Unknown => None,
        }
    }
}
