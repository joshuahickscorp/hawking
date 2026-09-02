//! Python import/call/subprocess facts for the auditor and reachability scan.
//!
//! This is **not** a second parser. It walks the tree produced by the same
//! `GrammarRegistry` Python grammar that `hawking_index::parse_source` uses.
//! The bundled `tags.scm` only emits bare-name defs/refs, which cannot
//! reconstruct bound-name call sites
//! (`from hcli.scheduler import Scheduler; Scheduler()`) or exact-path
//! subprocess launches. Those facts live here so both the roadmap auditor
//! and the capability-reachability scan share one JSON surface.
//!
//! Schema: `hawking.index.python_facts.v1`
//! CLI: `hawking-index-query python-facts [--git-head] [--commit REV] [--repo DIR]`
//! Git-backed dumps read commit blobs, never the working tree.
//!
//! r1 may later expose the same command on the `hawking-index` binary; the
//! schema string and object shape are the merge contract.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tree_sitter::{Node, Parser, Tree};

pub const PYTHON_FACTS_SCHEMA: &str = "hawking.index.python_facts.v1";

const SUBPROCESS_NAMES: &[&str] = &["run", "Popen", "check_call", "check_output", "call"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImportedName {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asname: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImportFact {
    pub line: u32,
    /// `"import"` (`import x`) or `"from"` (`from x import y`).
    pub form: String,
    /// Module path for `from X import ...` (no leading dots). `None` for `import`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
    /// Relative import level (`from .x` => 1). Zero for absolute / plain import.
    pub level: u32,
    pub names: Vec<ImportedName>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CallFact {
    pub line: u32,
    /// Callee name: `Foo()` => `"Foo"`; `mod.Foo()` => `"Foo"`.
    pub name: String,
    /// `Some(obj)` only when the callee is a one-level `Name.attr` (`obj.name()`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NameUseFact {
    pub line: u32,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubprocessLitFact {
    pub line: u32,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DefFact {
    pub name: String,
    /// `"function"` | `"class"` | `"assignment"` — same vocabulary as
    /// `tools.roadmap.gitfs.classify_symbol`.
    pub kind: String,
    pub line: u32,
    /// `"module"` for `tree.body` items; `"nested"` otherwise.
    pub scope: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct PythonFileFacts {
    pub path: String,
    /// Git commit the blob was read from. `None` for overlay-only files
    /// (mutation checks) and for dumps that never touched git.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    pub unparseable: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub definitions: Vec<DefFact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub imports: Vec<ImportFact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<CallFact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub name_uses: Vec<NameUseFact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subprocess_literals: Vec<SubprocessLitFact>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PythonFactsDump {
    pub schema: String,
    /// Resolved commit SHA the git-backed files were parsed from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    pub files: Vec<PythonFileFacts>,
}

impl Default for PythonFactsDump {
    fn default() -> Self {
        Self {
            schema: PYTHON_FACTS_SCHEMA.to_string(),
            commit: None,
            files: Vec::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct OverlayFile {
    path: String,
    content: String,
}

fn line_of(node: Node<'_>) -> u32 {
    node.start_position().row as u32 + 1
}

fn text_of(node: Node<'_>, src: &[u8]) -> String {
    node.utf8_text(src).unwrap_or("").to_string()
}

fn named_kids<'a>(node: Node<'a>) -> Vec<Node<'a>> {
    let mut c = node.walk();
    node.named_children(&mut c).collect()
}

fn first_kind<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    named_kids(node).into_iter().find(|n| n.kind() == kind)
}

fn extract_aliased(node: Node<'_>, src: &[u8]) -> ImportedName {
    let mut name = String::new();
    let mut asname = None;
    for child in named_kids(node) {
        match child.kind() {
            "dotted_name" | "identifier" if name.is_empty() => {
                name = text_of(child, src);
            }
            "identifier" => {
                asname = Some(text_of(child, src));
            }
            _ => {}
        }
    }
    ImportedName { name, asname }
}

fn relative_level(node: Node<'_>, src: &[u8]) -> u32 {
    if let Some(prefix) = first_kind(node, "import_prefix") {
        return text_of(prefix, src).chars().filter(|&ch| ch == '.').count() as u32;
    }
    text_of(node, src)
        .chars()
        .take_while(|&ch| ch == '.')
        .count() as u32
}

fn relative_module(node: Node<'_>, src: &[u8]) -> Option<String> {
    first_kind(node, "dotted_name").map(|n| text_of(n, src))
}

fn extract_import(node: Node<'_>, src: &[u8]) -> ImportFact {
    let mut names = Vec::new();
    for child in named_kids(node) {
        match child.kind() {
            "aliased_import" => names.push(extract_aliased(child, src)),
            "dotted_name" => names.push(ImportedName {
                name: text_of(child, src),
                asname: None,
            }),
            _ => {}
        }
    }
    ImportFact {
        line: line_of(node),
        form: "import".to_string(),
        module: None,
        level: 0,
        names,
    }
}

fn extract_import_from(node: Node<'_>, src: &[u8]) -> ImportFact {
    let mut module = None;
    let mut level = 0u32;
    let mut names = Vec::new();
    let mut seen_module = false;
    for child in named_kids(node) {
        match child.kind() {
            "relative_import" if !seen_module => {
                seen_module = true;
                level = relative_level(child, src);
                module = relative_module(child, src);
            }
            "dotted_name" if !seen_module => {
                seen_module = true;
                module = Some(text_of(child, src));
            }
            "wildcard_import" => names.push(ImportedName {
                name: "*".to_string(),
                asname: None,
            }),
            "aliased_import" => names.push(extract_aliased(child, src)),
            "dotted_name" | "identifier" if seen_module => names.push(ImportedName {
                name: text_of(child, src),
                asname: None,
            }),
            _ => {}
        }
    }
    ImportFact {
        line: line_of(node),
        form: "from".to_string(),
        module,
        level,
        names,
    }
}

fn extract_call(node: Node<'_>, src: &[u8]) -> Option<CallFact> {
    let func = node.child_by_field_name("function")?;
    match func.kind() {
        "identifier" => Some(CallFact {
            line: line_of(node),
            name: text_of(func, src),
            object: None,
        }),
        "attribute" => {
            let attr = func
                .child_by_field_name("attribute")
                .or_else(|| first_kind(func, "identifier"))?;
            let obj = func.child_by_field_name("object")?;
            // AST analyzer only matches `Name()` or `Name.attr()`. Chained
            // `Path(__file__).resolve()` is not a bound-name call of `resolve`.
            if obj.kind() != "identifier" {
                return None;
            }
            Some(CallFact {
                line: line_of(node),
                name: text_of(attr, src),
                object: Some(text_of(obj, src)),
            })
        }
        _ => None,
    }
}

fn is_subprocess_call(node: Node<'_>, src: &[u8]) -> bool {
    let Some(func) = node.child_by_field_name("function") else {
        return false;
    };
    match func.kind() {
        "identifier" => SUBPROCESS_NAMES.iter().any(|n| text_of(func, src) == *n),
        "attribute" => {
            let attr = func
                .child_by_field_name("attribute")
                .or_else(|| first_kind(func, "identifier"));
            match attr {
                Some(a) => SUBPROCESS_NAMES.iter().any(|n| text_of(a, src) == *n),
                None => false,
            }
        }
        _ => false,
    }
}

fn plain_python_string(node: Node<'_>, src: &[u8]) -> Option<String> {
    if node.kind() != "string" {
        return None;
    }
    for child in named_kids(node) {
        if child.kind() == "interpolation" {
            return None;
        }
    }
    parse_python_str_literal(&text_of(node, src))
}

fn parse_python_str_literal(raw: &str) -> Option<String> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    let bytes = s.as_bytes();
    let mut i = 0usize;
    let mut is_raw = false;
    while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
        match bytes[i].to_ascii_lowercase() {
            b'f' => return None,
            b'b' => return None,
            b'r' => is_raw = true,
            b'u' => {}
            _ => return None,
        }
        i += 1;
    }
    let rest = &s[i..];
    let quote: &str = if rest.starts_with("'''") || rest.starts_with("\"\"\"") {
        &rest[..3]
    } else if rest.starts_with('\'') || rest.starts_with('"') {
        &rest[..1]
    } else {
        return None;
    };
    if rest.len() < quote.len() * 2 || !rest.ends_with(quote) {
        return None;
    }
    let inner = &rest[quote.len()..rest.len() - quote.len()];
    if is_raw {
        return Some(inner.to_string());
    }
    Some(unescape_python(inner))
}

fn unescape_python(inner: &str) -> String {
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some('\'') => out.push('\''),
            Some('"') => out.push('"'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn collect_string_lits(node: Node<'_>, src: &[u8], out: &mut Vec<String>) {
    if node.kind() == "string" {
        if let Some(s) = plain_python_string(node, src) {
            out.push(s);
        }
        return;
    }
    if node.kind() == "concatenated_string" {
        let mut parts = Vec::new();
        for child in named_kids(node) {
            if let Some(s) = plain_python_string(child, src) {
                parts.push(s);
            } else {
                return;
            }
        }
        if !parts.is_empty() {
            out.push(parts.concat());
        }
        return;
    }
    let mut c = node.walk();
    for child in node.children(&mut c) {
        collect_string_lits(child, src, out);
    }
}

fn identifier_left_of_assignment(node: Node<'_>, src: &[u8]) -> Option<(String, u32)> {
    let left = node.child_by_field_name("left")?;
    if left.kind() == "identifier" {
        return Some((text_of(left, src), line_of(left)));
    }
    None
}

fn def_name(node: Node<'_>, src: &[u8]) -> Option<(String, u32)> {
    let name = node.child_by_field_name("name")?;
    Some((text_of(name, src), line_of(name)))
}

fn under_kind(node: Node<'_>, kinds: &[&str]) -> bool {
    let mut cur = Some(node);
    while let Some(n) = cur {
        if kinds.iter().any(|k| n.kind() == *k) {
            return true;
        }
        cur = n.parent();
    }
    false
}

struct Extractor<'a> {
    src: &'a [u8],
    facts: PythonFileFacts,
    import_locals: HashSet<String>,
    /// Start bytes of *identifier* nodes that are the func of `name(...)`.
    call_ident_starts: HashSet<usize>,
    /// Start bytes of *attribute* nodes that are the func of `obj.attr(...)`.
    call_attr_starts: HashSet<usize>,
}

impl<'a> Extractor<'a> {
    fn walk_all(&mut self, node: Node<'_>) {
        match node.kind() {
            "import_statement" => {
                let imp = extract_import(node, self.src);
                self.note_import_locals(&imp);
                self.facts.imports.push(imp);
            }
            "import_from_statement" => {
                let imp = extract_import_from(node, self.src);
                self.note_import_locals(&imp);
                self.facts.imports.push(imp);
            }
            "call" => {
                if let Some(func) = node.child_by_field_name("function") {
                    match func.kind() {
                        "identifier" => {
                            self.call_ident_starts.insert(func.start_byte());
                        }
                        "attribute" => {
                            self.call_attr_starts.insert(func.start_byte());
                        }
                        _ => {}
                    }
                }
                if let Some(call) = extract_call(node, self.src) {
                    self.facts.calls.push(call);
                }
                if is_subprocess_call(node, self.src) {
                    let mut lits = Vec::new();
                    collect_string_lits(node, self.src, &mut lits);
                    let line = line_of(node);
                    for value in lits {
                        self.facts
                            .subprocess_literals
                            .push(SubprocessLitFact { line, value });
                    }
                }
            }
            "function_definition" => {
                if let Some((name, line)) = def_name(node, self.src) {
                    self.facts.definitions.push(DefFact {
                        name,
                        kind: "function".to_string(),
                        line,
                        scope: "nested".to_string(),
                    });
                }
            }
            "class_definition" => {
                if let Some((name, line)) = def_name(node, self.src) {
                    self.facts.definitions.push(DefFact {
                        name,
                        kind: "class".to_string(),
                        line,
                        scope: "nested".to_string(),
                    });
                }
            }
            _ => {}
        }
        let mut c = node.walk();
        for child in node.children(&mut c) {
            self.walk_all(child);
        }
    }

    fn note_import_locals(&mut self, imp: &ImportFact) {
        for n in &imp.names {
            if n.name == "*" {
                continue;
            }
            if let Some(a) = &n.asname {
                self.import_locals.insert(a.clone());
            } else if imp.form == "import" {
                let local = n.name.split('.').next().unwrap_or(&n.name);
                self.import_locals.insert(local.to_string());
            } else {
                self.import_locals.insert(n.name.clone());
            }
        }
    }

    fn mark_module_defs(&mut self, root: Node<'_>) {
        for child in named_kids(root) {
            self.mark_module_stmt(child);
        }
    }

    fn mark_module_stmt(&mut self, node: Node<'_>) {
        match node.kind() {
            "decorated_definition" => {
                for child in named_kids(node) {
                    self.mark_module_stmt(child);
                }
            }
            "function_definition" => {
                if let Some((name, line)) = def_name(node, self.src) {
                    if let Some(existing) = self
                        .facts
                        .definitions
                        .iter_mut()
                        .find(|d| d.name == name && d.line == line && d.kind == "function")
                    {
                        existing.scope = "module".to_string();
                    }
                }
            }
            "class_definition" => {
                if let Some((name, line)) = def_name(node, self.src) {
                    if let Some(existing) = self
                        .facts
                        .definitions
                        .iter_mut()
                        .find(|d| d.name == name && d.line == line && d.kind == "class")
                    {
                        existing.scope = "module".to_string();
                    }
                }
            }
            "expression_statement" => {
                for child in named_kids(node) {
                    if child.kind() == "assignment" {
                        if let Some((name, line)) = identifier_left_of_assignment(child, self.src) {
                            self.facts.definitions.push(DefFact {
                                name,
                                kind: "assignment".to_string(),
                                line,
                                scope: "module".to_string(),
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn collect_name_uses(&mut self, node: Node<'_>) {
        if self.import_locals.is_empty() {
            return;
        }
        match node.kind() {
            "identifier" => {
                if self.call_ident_starts.contains(&node.start_byte()) {
                    // bare `Foo()` — that's a call, not a weak name use
                } else if under_kind(node, &["import_statement", "import_from_statement"]) {
                    // `from x import Foo` is not a Load Name in the AST analyzer
                } else if under_kind(node, &["function_definition", "class_definition"])
                    && node
                        .parent()
                        .and_then(|p| p.child_by_field_name("name"))
                        .map(|n| n.start_byte())
                        == Some(node.start_byte())
                {
                    // def/class name
                } else {
                    let name = text_of(node, self.src);
                    if self.import_locals.contains(&name) && !is_store_ident(node) {
                        self.facts.name_uses.push(NameUseFact {
                            line: line_of(node),
                            name,
                            object: None,
                        });
                    }
                }
            }
            "attribute" => {
                if !self.call_attr_starts.contains(&node.start_byte()) {
                    if let Some(obj) = node.child_by_field_name("object") {
                        if obj.kind() == "identifier" {
                            let obj_name = text_of(obj, self.src);
                            if self.import_locals.contains(&obj_name) {
                                if let Some(attr) = node
                                    .child_by_field_name("attribute")
                                    .or_else(|| first_kind(node, "identifier"))
                                {
                                    self.facts.name_uses.push(NameUseFact {
                                        line: line_of(node),
                                        name: text_of(attr, self.src),
                                        object: Some(obj_name),
                                    });
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        let mut c = node.walk();
        for child in node.children(&mut c) {
            self.collect_name_uses(child);
        }
    }
}

fn is_store_ident(node: Node<'_>) -> bool {
    let Some(parent) = node.parent() else {
        return false;
    };
    match parent.kind() {
        "assignment" => {
            if let Some(left) = parent.child_by_field_name("left") {
                return node_contains_start(left, node.start_byte());
            }
            false
        }
        "pattern_list" | "tuple_pattern" | "list_pattern" | "pattern" => is_store_ident(parent),
        "for_statement" => {
            if let Some(left) = parent.child_by_field_name("left") {
                return node_contains_start(left, node.start_byte());
            }
            false
        }
        "as_pattern" => {
            if let Some(alias) = parent.child_by_field_name("alias") {
                return alias.start_byte() == node.start_byte();
            }
            let kids = named_kids(parent);
            kids.last()
                .map(|n| n.start_byte() == node.start_byte())
                .unwrap_or(false)
        }
        "parameters" | "typed_parameter" | "default_parameter" | "typed_default_parameter" => true,
        _ => false,
    }
}

fn node_contains_start(node: Node<'_>, start: usize) -> bool {
    if node.start_byte() == start {
        return true;
    }
    let mut c = node.walk();
    for child in node.children(&mut c) {
        if node_contains_start(child, start) {
            return true;
        }
    }
    false
}

fn extract_from_tree(
    path: &str,
    src: &str,
    tree: &Tree,
    watch: &HashSet<String>,
) -> PythonFileFacts {
    let root = tree.root_node();
    let mut ex = Extractor {
        src: src.as_bytes(),
        facts: PythonFileFacts {
            path: path.to_string(),
            unparseable: false,
            ..Default::default()
        },
        import_locals: HashSet::new(),
        call_ident_starts: HashSet::new(),
        call_attr_starts: HashSet::new(),
    };
    if root.has_error() && root.named_child_count() == 0 {
        ex.facts.unparseable = true;
        return ex.facts;
    }
    ex.walk_all(root);
    ex.mark_module_defs(root);
    ex.collect_name_uses(root);
    if !watch.is_empty() {
        ex.retain_watched(watch);
    }
    ex.facts
}

impl Extractor<'_> {
    /// Keep only facts that can participate in bound-name analysis of `watch`.
    fn retain_watched(&mut self, watch: &HashSet<String>) {
        let mut locals: HashSet<String> = watch.clone();
        for imp in &self.facts.imports {
            for n in &imp.names {
                let orig = n.name.as_str();
                let last = orig.rsplit('.').next().unwrap_or(orig);
                let local = n.asname.as_deref().unwrap_or(last);
                if watch.contains(orig) || watch.contains(last) || watch.contains(local) {
                    locals.insert(local.to_string());
                    locals.insert(last.to_string());
                }
            }
        }
        for d in &self.facts.definitions {
            if watch.contains(&d.name) {
                locals.insert(d.name.clone());
            }
        }
        self.facts.calls.retain(|c| {
            if let Some(obj) = &c.object {
                locals.contains(obj) || locals.contains(&c.name)
            } else {
                locals.contains(&c.name)
            }
        });
        self.facts.name_uses.retain(|u| {
            if let Some(obj) = &u.object {
                locals.contains(obj) || locals.contains(&u.name)
            } else {
                locals.contains(&u.name)
            }
        });
    }
}

fn python_language() -> tree_sitter::Language {
    // Identical to hawking_index::parse::GrammarRegistry::bundle(LangId::Python).
    tree_sitter_python::LANGUAGE.into()
}

/// Extract Python facts from one file using the registry Python grammar.
pub fn extract_python_facts(rel_path: &str, source: &str) -> PythonFileFacts {
    let mut parser = Parser::new();
    if parser.set_language(&python_language()).is_err() {
        return PythonFileFacts {
            path: rel_path.to_string(),
            unparseable: true,
            ..Default::default()
        };
    }
    match parser.parse(source, None) {
        Some(tree) => extract_from_tree(rel_path, source, &tree, &HashSet::new()),
        None => PythonFileFacts {
            path: rel_path.to_string(),
            unparseable: true,
            ..Default::default()
        },
    }
}

/// Extract facts for many files, reusing one parser.
pub fn extract_python_facts_many<'a, I>(files: I) -> PythonFactsDump
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    extract_python_facts_many_watched(files, &HashSet::new())
}

/// Like [`extract_python_facts_many`] but drop call/name-use rows that cannot
/// bind to `watch` names (catalog symbols). Empty `watch` keeps everything.
pub fn extract_python_facts_many_watched<'a, I>(
    files: I,
    watch: &HashSet<String>,
) -> PythonFactsDump
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let mut parser = Parser::new();
    match parser.set_language(&python_language()) {
        Ok(()) => {
            let mut out = PythonFactsDump::default();
            for (path, src) in files {
                let facts = match parser.parse(src, None) {
                    Some(tree) => extract_from_tree(path, src, &tree, watch),
                    None => PythonFileFacts {
                        path: path.to_string(),
                        unparseable: true,
                        ..Default::default()
                    },
                };
                if facts.unparseable
                    || !facts.definitions.is_empty()
                    || !facts.imports.is_empty()
                    || !facts.calls.is_empty()
                    || !facts.name_uses.is_empty()
                    || !facts.subprocess_literals.is_empty()
                {
                    out.files.push(facts);
                }
            }
            out
        }
        Err(_) => PythonFactsDump::default(),
    }
}

/// Read overlay NDJSON (`{"path":"...","content":"..."}` per line) from a reader.
pub fn read_overlay_ndjson<R: Read>(reader: R) -> Result<HashMap<String, String>, String> {
    let mut overlay = HashMap::new();
    let buf = BufReader::new(reader);
    for (i, line) in buf.lines().enumerate() {
        let line = line.map_err(|e| format!("stdin: {e}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: OverlayFile =
            serde_json::from_str(trimmed).map_err(|e| format!("stdin line {}: {e}", i + 1))?;
        overlay.insert(row.path, row.content);
    }
    Ok(overlay)
}

fn git_stdout(repo: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let out = Command::new("git")
        .arg("--no-optional-locks")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(out.stdout)
}

fn git_rev_parse(repo: &Path, rev: &str) -> Result<String, String> {
    let spec = format!("{rev}^{{commit}}");
    let raw = git_stdout(repo, &["rev-parse", "--verify", &spec])?;
    let sha = String::from_utf8_lossy(&raw).trim().to_string();
    if sha.is_empty() {
        return Err(format!("git rev-parse {rev} produced an empty sha"));
    }
    Ok(sha)
}

/// Enumerate `*.py` blobs in `commit`. `git ls-tree -- '*.py'` does not match
/// nested paths, so this lists the whole tree and filters in-process.
/// Untracked and index-only files are invisible: what is in the commit exists.
fn git_ls_py_at(repo: &Path, commit: &str) -> Result<Vec<String>, String> {
    let raw = git_stdout(repo, &["ls-tree", "-r", "--name-only", "-z", commit])?;
    Ok(raw
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .filter_map(|s| String::from_utf8(s.to_vec()).ok())
        .filter(|s| s.ends_with(".py") && !s.contains("__pycache__"))
        .collect())
}

/// Batch-load blobs at `commit:path`. Missing objects become empty (same as
/// `git show` miss). Never reads the working tree.
///
/// Stdin is written on a helper thread so a blob larger than the OS pipe
/// buffer cannot deadlock against unread stdout.
fn git_cat_file_batch(
    repo: &Path,
    commit: &str,
    paths: &[String],
) -> Result<HashMap<String, String>, String> {
    if paths.is_empty() {
        return Ok(HashMap::new());
    }
    let mut child = Command::new("git")
        .arg("--no-optional-locks")
        .arg("-C")
        .arg(repo)
        .args(["cat-file", "--batch"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("git cat-file: {e}"))?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "git cat-file: no stdin".to_string())?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "git cat-file: no stdout".to_string())?;
    let specs: Vec<String> = paths.iter().map(|p| format!("{commit}:{p}\n")).collect();
    let writer = std::thread::spawn(move || -> Result<(), String> {
        for spec in &specs {
            stdin
                .write_all(spec.as_bytes())
                .map_err(|e| format!("git cat-file stdin: {e}"))?;
        }
        drop(stdin);
        Ok(())
    });
    let mut bytes = Vec::new();
    stdout
        .read_to_end(&mut bytes)
        .map_err(|e| format!("git cat-file stdout: {e}"))?;
    drop(stdout);
    writer
        .join()
        .map_err(|_| "git cat-file stdin thread panicked".to_string())??;
    let output = child
        .wait_with_output()
        .map_err(|e| format!("git cat-file wait: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "git cat-file --batch failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_cat_file_batch(&bytes, paths)
}

fn parse_cat_file_batch(bytes: &[u8], paths: &[String]) -> Result<HashMap<String, String>, String> {
    let mut out = HashMap::new();
    let mut i = 0usize;
    let mut pidx = 0usize;
    while i < bytes.len() && pidx < paths.len() {
        let nl = bytes[i..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|n| i + n)
            .ok_or_else(|| "truncated cat-file header".to_string())?;
        let header = std::str::from_utf8(&bytes[i..nl]).unwrap_or("");
        i = nl + 1;
        let path = &paths[pidx];
        pidx += 1;
        if header.ends_with(" missing") {
            continue;
        }
        let size: usize = header
            .rsplit(' ')
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("bad cat-file header: {header}"))?;
        if i + size > bytes.len() {
            return Err("truncated cat-file blob".to_string());
        }
        let blob = &bytes[i..i + size];
        i += size;
        if i < bytes.len() && bytes[i] == b'\n' {
            i += 1;
        }
        out.insert(path.clone(), String::from_utf8_lossy(blob).into_owned());
    }
    Ok(out)
}

/// Load Python sources from `commit` blobs, overlay winning.
///
/// Never reads the working tree. Untracked files are invisible. A dirty
/// worktree cannot contribute a line number.
pub fn load_python_sources(
    repo: &Path,
    overlay: &HashMap<String, String>,
    commit: &str,
) -> Result<Vec<(String, String)>, String> {
    let mut paths = git_ls_py_at(repo, commit)?;
    for extra in overlay.keys() {
        if extra.ends_with(".py") && !paths.iter().any(|p| p == extra) {
            paths.push(extra.clone());
        }
    }
    let need_git: Vec<String> = paths
        .iter()
        .filter(|p| !overlay.contains_key(*p))
        .cloned()
        .collect();
    let from_git = git_cat_file_batch(repo, commit, &need_git)?;
    let mut files = Vec::with_capacity(paths.len());
    for p in paths {
        if let Some(text) = overlay.get(&p) {
            files.push((p, text.clone()));
        } else if let Some(text) = from_git.get(&p) {
            files.push((p, text.clone()));
        } else {
            files.push((p, String::new()));
        }
    }
    Ok(files)
}

/// Index every `*.py` blob at HEAD, overlay winning. See
/// [`dump_python_facts_at_commit`].
pub fn dump_python_facts_git_head(
    repo: &Path,
    overlay: &HashMap<String, String>,
    watch: &HashSet<String>,
) -> Result<PythonFactsDump, String> {
    dump_python_facts_at_commit(repo, "HEAD", overlay, watch)
}

/// Index every `*.py` blob at `commit` (any rev `git rev-parse` accepts).
/// Overlay content replaces the blob for that path and does not inherit
/// the commit stamp.
pub fn dump_python_facts_at_commit(
    repo: &Path,
    commit: &str,
    overlay: &HashMap<String, String>,
    watch: &HashSet<String>,
) -> Result<PythonFactsDump, String> {
    let sha = git_rev_parse(repo, commit)?;
    let files = load_python_sources(repo, overlay, &sha)?;
    let pairs: Vec<(&str, &str)> = files
        .iter()
        .map(|(p, c)| (p.as_str(), c.as_str()))
        .collect();
    let mut dump = extract_python_facts_many_watched(pairs, watch);
    dump.commit = Some(sha.clone());
    for f in &mut dump.files {
        if overlay.contains_key(&f.path) {
            f.commit = None;
        } else {
            f.commit = Some(sha.clone());
        }
    }
    Ok(dump)
}

/// Index only the files supplied as NDJSON (no git). Used by tests / overlays.
pub fn dump_python_facts_from_overlay(
    overlay: &HashMap<String, String>,
    watch: &HashSet<String>,
) -> PythonFactsDump {
    let mut items: Vec<(&str, &str)> = overlay
        .iter()
        .map(|(p, c)| (p.as_str(), c.as_str()))
        .collect();
    items.sort_by(|a, b| a.0.cmp(b.0));
    extract_python_facts_many_watched(items, watch)
}

pub fn default_repo() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn facts(src: &str) -> PythonFileFacts {
        extract_python_facts("hcli/sample.py", src)
    }

    #[test]
    fn import_from_absolute_and_alias() {
        let f = facts("from hcli.scheduler import Scheduler as S\n");
        assert_eq!(f.imports.len(), 1, "imports={:?}", f.imports);
        let imp = &f.imports[0];
        assert_eq!(imp.form, "from");
        assert_eq!(imp.module.as_deref(), Some("hcli.scheduler"));
        assert_eq!(imp.level, 0);
        assert_eq!(imp.names.len(), 1);
        assert_eq!(imp.names[0].name, "Scheduler");
        assert_eq!(imp.names[0].asname.as_deref(), Some("S"));
        assert_eq!(imp.line, 1);
    }

    #[test]
    fn import_plain_and_relative() {
        let f = facts("import json\nfrom .scheduler import Scheduler\nfrom .. import foo\n");
        assert_eq!(f.imports.len(), 3, "imports={:?}", f.imports);
        assert_eq!(f.imports[0].form, "import");
        assert_eq!(f.imports[0].names[0].name, "json");
        assert_eq!(f.imports[1].form, "from");
        assert_eq!(f.imports[1].level, 1);
        assert_eq!(f.imports[1].module.as_deref(), Some("scheduler"));
        assert_eq!(f.imports[2].level, 2);
        assert!(
            f.imports[2].module.is_none(),
            "module={:?}",
            f.imports[2].module
        );
        assert_eq!(f.imports[2].names[0].name, "foo");
    }

    #[test]
    fn calls_name_and_attribute() {
        let f = facts("Foo()\nmod.Bar()\n");
        assert!(
            f.calls
                .iter()
                .any(|c| c.name == "Foo" && c.object.is_none()),
            "calls={:?}",
            f.calls
        );
        assert!(
            f.calls
                .iter()
                .any(|c| c.name == "Bar" && c.object.as_deref() == Some("mod")),
            "calls={:?}",
            f.calls
        );
    }

    #[test]
    fn nested_attribute_call_is_not_a_bound_name_call() {
        let f = facts("a.b.c()\n");
        assert!(
            f.calls.iter().find(|c| c.name == "c").is_none(),
            "chained attr must not be a bound-name call: {:?}",
            f.calls
        );
    }

    #[test]
    fn imported_name_used_as_attribute_object_is_weak_not_call() {
        let src = "from hcli.scheduler import Scheduler\nScheduler.from_workspace(ws)\n";
        let f = facts(src);
        assert!(
            !f.calls
                .iter()
                .any(|c| c.name == "Scheduler" && c.object.is_none()),
            "calls={:?}",
            f.calls
        );
        assert!(
            f.name_uses
                .iter()
                .any(|u| u.name == "Scheduler" && u.object.is_none()),
            "uses={:?}",
            f.name_uses
        );
    }

    #[test]
    fn method_call_on_non_name_is_not_bare_resolve() {
        let src = "from hcli.context_budget import resolve\nPath(__file__).resolve()\n";
        let f = facts(src);
        assert!(
            !f.calls
                .iter()
                .any(|c| c.name == "resolve" && c.object.is_none()),
            "calls={:?}",
            f.calls
        );
    }

    #[test]
    fn import_identifier_is_not_a_name_use() {
        let src = "from hcli.scheduler import Scheduler\n";
        let f = facts(src);
        assert!(
            !f.name_uses.iter().any(|u| u.name == "Scheduler"),
            "uses={:?}",
            f.name_uses
        );
    }

    #[test]
    fn subprocess_exact_string() {
        let f = facts("subprocess.run(['hcli/scheduler.py', '--help'])\n");
        assert!(
            f.subprocess_literals
                .iter()
                .any(|s| s.value == "hcli/scheduler.py"),
            "lits={:?}",
            f.subprocess_literals
        );
    }

    #[test]
    fn subprocess_does_not_count_fstring() {
        let f = facts("subprocess.run(f'hcli/{name}.py')\n");
        assert!(
            !f.subprocess_literals
                .iter()
                .any(|s| s.value.contains("hcli/")),
            "lits={:?}",
            f.subprocess_literals
        );
    }

    #[test]
    fn module_class_function_assignment() {
        let src = "class Scheduler:\n    def tick(self):\n        pass\n\ndef abort():\n    pass\n\nNO_PROGRESS = 3\n";
        let f = facts(src);
        let kinds: Vec<_> = f
            .definitions
            .iter()
            .map(|d| (d.name.as_str(), d.kind.as_str(), d.scope.as_str()))
            .collect();
        assert!(
            kinds.contains(&("Scheduler", "class", "module")),
            "{kinds:?}"
        );
        assert!(
            kinds.contains(&("abort", "function", "module")),
            "{kinds:?}"
        );
        assert!(kinds.contains(&("tick", "function", "nested")), "{kinds:?}");
        assert!(
            kinds.contains(&("NO_PROGRESS", "assignment", "module")),
            "{kinds:?}"
        );
    }

    #[test]
    fn name_use_of_imported_symbol_is_not_a_call() {
        let src = "from hcli.scheduler import NO_PROGRESS\nraise NO_PROGRESS()\nexcept NO_PROGRESS:\n    x = NO_PROGRESS\n";
        let f = facts(src);
        assert!(
            f.calls.iter().any(|c| c.name == "NO_PROGRESS"),
            "calls={:?}",
            f.calls
        );
        assert!(
            f.name_uses
                .iter()
                .any(|u| u.name == "NO_PROGRESS" && u.object.is_none()),
            "uses={:?}",
            f.name_uses
        );
    }

    #[test]
    fn schema_constant() {
        let dump = extract_python_facts_many([("a.py", "x = 1\n")]);
        assert_eq!(dump.schema, PYTHON_FACTS_SCHEMA);
        assert_eq!(dump.files.len(), 1);
        assert!(dump.commit.is_none());
        assert!(dump.files[0].commit.is_none());
    }

    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn scratch_dir() -> Scratch {
        let dir = std::env::temp_dir().join(format!(
            "hawking-index-query-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn git_in(repo: &Path, args: &[&str]) -> std::process::Output {
        Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .output()
            .unwrap_or_else(|e| panic!("git {args:?}: {e}"))
    }

    fn init_git_repo(dir: &Path) {
        let out = git_in(dir, &["init", "-b", "main"]);
        assert!(out.status.success(), "git init: {}", String::from_utf8_lossy(&out.stderr));
        for (k, v) in [
            ("user.email", "r5@test"),
            ("user.name", "r5"),
            ("commit.gpgsign", "false"),
        ] {
            let out = git_in(dir, &["config", k, v]);
            assert!(out.status.success(), "git config {k}");
        }
    }

    fn commit_all(dir: &Path, message: &str) {
        let add = git_in(dir, &["add", "-A"]);
        assert!(add.status.success(), "git add: {}", String::from_utf8_lossy(&add.stderr));
        let commit = git_in(dir, &["commit", "-m", message]);
        assert!(
            commit.status.success(),
            "git commit: {}",
            String::from_utf8_lossy(&commit.stderr)
        );
    }

    #[test]
    fn git_head_ignores_dirty_worktree_and_untracked() {
        let tmp = scratch_dir();
        let repo = tmp.0.as_path();
        init_git_repo(repo);
        std::fs::write(repo.join("mod.py"), "def alpha():\n    return 1\n").unwrap();
        commit_all(repo, "base");
        let sha = String::from_utf8_lossy(&git_in(repo, &["rev-parse", "HEAD"]).stdout)
            .trim()
            .to_string();

        // Dirtied committed file: extra def only exists on disk.
        std::fs::write(
            repo.join("mod.py"),
            "def alpha():\n    return 1\n\ndef extra_only_on_disk():\n    return 2\n",
        )
        .unwrap();
        // Untracked file.
        std::fs::write(repo.join("untracked.py"), "def ghost():\n    return 3\n").unwrap();
        // Staged-but-uncommitted new file.
        std::fs::write(repo.join("staged.py"), "def staged():\n    return 4\n").unwrap();
        let add = git_in(repo, &["add", "staged.py"]);
        assert!(add.status.success());

        let dump = dump_python_facts_at_commit(repo, "HEAD", &HashMap::new(), &HashSet::new())
            .expect("dump");
        assert_eq!(dump.commit.as_deref(), Some(sha.as_str()));
        let paths: Vec<&str> = dump.files.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"mod.py"), "paths={paths:?}");
        assert!(
            !paths.contains(&"untracked.py"),
            "untracked must be invisible: {paths:?}"
        );
        assert!(
            !paths.contains(&"staged.py"),
            "index-only file must be invisible: {paths:?}"
        );
        let modf = dump.files.iter().find(|f| f.path == "mod.py").unwrap();
        assert_eq!(modf.commit.as_deref(), Some(sha.as_str()));
        let names: Vec<&str> = modf.definitions.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"alpha"), "{names:?}");
        assert!(
            !names.contains(&"extra_only_on_disk"),
            "dirty worktree def leaked into dump: {names:?}"
        );
        let max_line = modf.definitions.iter().map(|d| d.line).max().unwrap_or(0);
        assert!(
            max_line <= 2,
            "citation line {max_line} is past HEAD blob (2 lines)"
        );
    }

    #[test]
    fn cat_file_does_not_deadlock_on_blob_larger_than_pipe() {
        let tmp = scratch_dir();
        let repo = tmp.0.as_path();
        init_git_repo(repo);
        // OS pipe buffers are typically 64KiB. A blob well above that
        // deadlocks if stdin is fully written before stdout is read.
        let mut src = String::from("def alpha():\n    return 1\n");
        src.push_str("# ");
        src.push_str(&"a".repeat(200_000));
        src.push('\n');
        std::fs::write(repo.join("mod.py"), &src).unwrap();
        commit_all(repo, "big");
        let dump = dump_python_facts_at_commit(repo, "HEAD", &HashMap::new(), &HashSet::new())
            .expect("dump");
        let modf = dump.files.iter().find(|f| f.path == "mod.py").unwrap();
        assert!(modf.definitions.iter().any(|d| d.name == "alpha"));
    }

    #[test]
    fn overlay_wins_and_does_not_inherit_commit() {
        let tmp = scratch_dir();
        let repo = tmp.0.as_path();
        init_git_repo(repo);
        std::fs::write(repo.join("mod.py"), "def alpha():\n    return 1\n").unwrap();
        commit_all(repo, "base");
        let mut overlay = HashMap::new();
        overlay.insert(
            "mod.py".to_string(),
            "def alpha():\n    return 1\n\ndef overlay_fn():\n    pass\n".to_string(),
        );
        let dump =
            dump_python_facts_at_commit(repo, "HEAD", &overlay, &HashSet::new()).expect("dump");
        assert!(dump.commit.is_some());
        let modf = dump.files.iter().find(|f| f.path == "mod.py").unwrap();
        assert!(
            modf.commit.is_none(),
            "overlay file must not carry the git commit"
        );
        let names: Vec<&str> = modf.definitions.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"overlay_fn"), "{names:?}");
    }
}
