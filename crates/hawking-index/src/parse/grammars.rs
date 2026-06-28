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
                tree_sitter_typescript::TAGS_QUERY,
            ),
            LangId::Unknown => None,
        }
    }
}
