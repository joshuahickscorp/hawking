//! CLAUDE.md tree discovery, HTML comment stripping, and `@import` resolution.
//!
//! Discovery walks the directory chain from the repo root down to the working
//! directory (root first, so more specific memory is injected last) and collects
//! `CLAUDE.md`, `.claude/CLAUDE.md`, and `CLAUDE.local.md` per directory. A user
//! global `~/.claude/CLAUDE.md` sorts ahead of the project chain. CLAUDE.md files
//! that live in subdirectories *below* the working directory are recorded but
//! marked lazy: they are meant to be injected only when a file under that subtree
//! is read, not at launch.
//!
//! `@path` imports inline another file's (comment-stripped) content. Imports may
//! be relative to the importing file, absolute, or `~/`-anchored, and recurse up
//! to depth 4. An `@` inside inline backticks or a fenced code block is left
//! alone. The first import that resolves outside the repo root is flagged as
//! requiring approval.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use globset::GlobSet;

use crate::layout::Layout;

/// Maximum `@import` recursion depth (levels below the launch-injected file).
pub const MAX_IMPORT_DEPTH: usize = 4;

/// Where a memory file sits in the scope hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    /// `~/.claude/CLAUDE.md`.
    UserGlobal,
    /// `<dir>/CLAUDE.md`.
    Project,
    /// `<dir>/.claude/CLAUDE.md`.
    DotClaude,
    /// `<dir>/CLAUDE.local.md`.
    Local,
    /// A `CLAUDE.md` in a subdirectory below the working directory.
    Subtree,
}

/// A single record of one `@import`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportRecord {
    /// The raw token as written (without the leading `@`).
    pub spec: String,
    /// The resolved absolute path.
    pub resolved: PathBuf,
    /// Depth at which this import was encountered (1 = imported by a launch file).
    pub depth: usize,
    /// Whether the target resolved outside the repo root.
    pub external: bool,
    /// Whether this import was flagged for user approval (the first external one).
    pub approval_required: bool,
    /// Whether the file existed and was inlined (false if missing or past cutoff).
    pub inlined: bool,
}

/// One discovered memory file.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub path: PathBuf,
    pub kind: MemoryKind,
    /// Lazy entries are not injected at launch; they attach to a subtree.
    pub lazy: bool,
    /// Raw file bytes as read.
    pub raw: String,
    /// Injection-ready content: HTML comments stripped, imports inlined.
    pub injected: String,
    /// Every import encountered while resolving this entry (any depth).
    pub imports: Vec<ImportRecord>,
}

/// The full discovered memory tree for a layout.
#[derive(Debug, Clone, Default)]
pub struct MemoryTree {
    pub entries: Vec<MemoryEntry>,
}

impl MemoryTree {
    /// Launch-injected entries in order (root first, more specific last). Lazy
    /// subtree entries are excluded.
    pub fn launch_entries(&self) -> Vec<&MemoryEntry> {
        self.entries.iter().filter(|e| !e.lazy).collect()
    }

    /// Paths of the launch-injected entries, in order. Handy for order assertions.
    pub fn launch_order(&self) -> Vec<PathBuf> {
        self.launch_entries().iter().map(|e| e.path.clone()).collect()
    }

    /// Lazy subtree entries (injected on demand when their subtree is read).
    pub fn lazy_entries(&self) -> Vec<&MemoryEntry> {
        self.entries.iter().filter(|e| e.lazy).collect()
    }

    /// Whether any import across the tree was flagged for approval.
    pub fn approval_required(&self) -> bool {
        self.entries
            .iter()
            .flat_map(|e| e.imports.iter())
            .any(|i| i.approval_required)
    }
}

/// Discover the CLAUDE.md tree. `excludes` is an optional compiled glob set from
/// `claudeMdExcludes` in settings; matching files are skipped entirely.
pub fn discover(layout: &Layout, excludes: Option<&GlobSet>) -> MemoryTree {
    let mut entries: Vec<MemoryEntry> = Vec::new();
    let mut approval_used = false;

    let excluded = |p: &Path| -> bool {
        match excludes {
            Some(set) => {
                // Match on both the absolute path and the path relative to the
                // repo root so simple globs like `packages/**/CLAUDE.md` work.
                let rel = p
                    .strip_prefix(&layout.repo_root)
                    .unwrap_or(p);
                set.is_match(p) || set.is_match(rel)
            }
            None => false,
        }
    };

    let push = |entries: &mut Vec<MemoryEntry>,
                    approval_used: &mut bool,
                    path: PathBuf,
                    kind: MemoryKind,
                    lazy: bool| {
        if excluded(&path) {
            return;
        }
        let raw = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => return,
        };
        let stripped = strip_html_comments(&raw);
        let mut imports = Vec::new();
        let injected = if lazy {
            // Lazy entries are not resolved until injected; keep stripped body.
            stripped.clone()
        } else {
            let mut visited = HashSet::new();
            visited.insert(canonical_key(&path));
            resolve_imports(
                &stripped,
                &path,
                layout,
                1,
                &mut visited,
                &mut imports,
                approval_used,
            )
        };
        entries.push(MemoryEntry {
            path,
            kind,
            lazy,
            raw,
            injected,
            imports,
        });
    };

    // 1. User global memory, least specific, injected first.
    let user_global = layout.home.join(".claude").join("CLAUDE.md");
    push(
        &mut entries,
        &mut approval_used,
        user_global,
        MemoryKind::UserGlobal,
        false,
    );

    // 2. The repo-root-down-to-cwd chain, root first.
    for dir in layout.dir_chain_root_first() {
        push(
            &mut entries,
            &mut approval_used,
            dir.join("CLAUDE.md"),
            MemoryKind::Project,
            false,
        );
        push(
            &mut entries,
            &mut approval_used,
            dir.join(".claude").join("CLAUDE.md"),
            MemoryKind::DotClaude,
            false,
        );
        push(
            &mut entries,
            &mut approval_used,
            dir.join("CLAUDE.local.md"),
            MemoryKind::Local,
            false,
        );
    }

    // 3. Subtree CLAUDE.md files below cwd, marked lazy. Sorted for determinism.
    let mut subtree: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(&layout.cwd)
        .min_depth(2)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() && entry.file_name() == "CLAUDE.md" {
            subtree.push(entry.into_path());
        }
    }
    subtree.sort();
    for path in subtree {
        push(
            &mut entries,
            &mut approval_used,
            path,
            MemoryKind::Subtree,
            true,
        );
    }

    MemoryTree { entries }
}

fn canonical_key(p: &Path) -> PathBuf {
    p.to_path_buf()
}

/// Resolve `@import` tokens in `content`, returning content with imports inlined.
#[allow(clippy::too_many_arguments)]
fn resolve_imports(
    content: &str,
    importing_file: &Path,
    layout: &Layout,
    depth: usize,
    visited: &mut HashSet<PathBuf>,
    records: &mut Vec<ImportRecord>,
    approval_used: &mut bool,
) -> String {
    let importing_dir = importing_file.parent().unwrap_or(Path::new("."));
    let mut out = String::with_capacity(content.len());
    let mut fence: Option<String> = None;

    for line in content.lines() {
        let trimmed = line.trim_start();
        // Fenced code block toggles: a run of >=3 backticks or tildes.
        if let Some(marker) = fence_marker(trimmed) {
            match &fence {
                None => fence = Some(marker),
                Some(open) if marker.starts_with(open.chars().next().unwrap()) => fence = None,
                Some(_) => {}
            }
            out.push_str(line);
            out.push('\n');
            continue;
        }
        if fence.is_some() {
            // Inside a code fence: imports are inert.
            out.push_str(line);
            out.push('\n');
            continue;
        }

        // Scan the line for import tokens outside inline backticks.
        let tokens = scan_line_imports(line);
        if tokens.is_empty() {
            out.push_str(line);
            out.push('\n');
            continue;
        }

        // The line has at least one import. Emit the line, then inline each
        // imported file after it (Claude Code injects imported memory alongside).
        out.push_str(line);
        out.push('\n');
        for spec in tokens {
            let resolved = resolve_import_path(&spec, importing_dir, layout);
            let external = !resolved.starts_with(&layout.repo_root);
            let approval_required = external && !*approval_used;
            if approval_required {
                *approval_used = true;
            }

            let mut inlined = false;
            if depth <= MAX_IMPORT_DEPTH {
                let key = canonical_key(&resolved);
                if !visited.contains(&key) {
                    if let Ok(raw) = std::fs::read_to_string(&resolved) {
                        visited.insert(key);
                        let stripped = strip_html_comments(&raw);
                        let nested = resolve_imports(
                            &stripped,
                            &resolved,
                            layout,
                            depth + 1,
                            visited,
                            records,
                            approval_used,
                        );
                        out.push_str(&nested);
                        if !nested.ends_with('\n') {
                            out.push('\n');
                        }
                        inlined = true;
                    }
                }
            }

            records.push(ImportRecord {
                spec,
                resolved,
                depth,
                external,
                approval_required,
                inlined,
            });
        }
    }

    out
}

/// Resolve one import spec to an absolute path.
fn resolve_import_path(spec: &str, importing_dir: &Path, layout: &Layout) -> PathBuf {
    if let Some(rest) = spec.strip_prefix("~/") {
        return layout.home.join(rest);
    }
    if spec == "~" {
        return layout.home.clone();
    }
    let p = Path::new(spec);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    importing_dir.join(spec)
}

/// Return the fence marker string if `trimmed` opens or closes a code fence.
fn fence_marker(trimmed: &str) -> Option<String> {
    let first = trimmed.chars().next()?;
    if first != '`' && first != '~' {
        return None;
    }
    let run: String = trimmed.chars().take_while(|&c| c == first).collect();
    if run.len() >= 3 {
        Some(run)
    } else {
        None
    }
}

/// Find `@import` tokens on a single line, skipping any inside inline backticks.
fn scan_line_imports(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    let mut in_code = false;
    let mut prev_boundary = true; // start of line is a boundary
    while i < chars.len() {
        let c = chars[i];
        if c == '`' {
            in_code = !in_code;
            prev_boundary = true;
            i += 1;
            continue;
        }
        if c == '@' && !in_code && prev_boundary {
            // Collect the path token: everything up to whitespace or backtick.
            let mut j = i + 1;
            while j < chars.len() {
                let cj = chars[j];
                if cj.is_whitespace() || cj == '`' {
                    break;
                }
                j += 1;
            }
            let mut token: String = chars[i + 1..j].iter().collect();
            // Trim trailing punctuation that is unlikely to be part of a path.
            while let Some(last) = token.chars().last() {
                if matches!(last, ',' | '.' | ';' | ':' | ')' | ']' | '!' | '?')
                    && !token.ends_with(".md")
                {
                    token.pop();
                } else {
                    break;
                }
            }
            if !token.is_empty() {
                out.push(token);
            }
            i = j;
            prev_boundary = true;
            continue;
        }
        prev_boundary = c.is_whitespace() || c == '(' || c == '[';
        i += 1;
    }
    out
}

/// Strip block-level HTML comments (`<!-- ... -->`) from markdown, preserving any
/// that appear inside fenced code blocks. Multi-line comments are supported.
pub fn strip_html_comments(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    let mut fence: Option<char> = None;
    let mut in_comment = false;

    for line in content.lines() {
        let trimmed = line.trim_start();

        // Fence handling first: never strip inside code.
        if fence.is_none() && !in_comment {
            if let Some(marker) = fence_marker(trimmed) {
                fence = Some(marker.chars().next().unwrap());
                out.push_str(line);
                out.push('\n');
                continue;
            }
        } else if let Some(f) = fence {
            if let Some(marker) = fence_marker(trimmed) {
                if marker.starts_with(f) {
                    fence = None;
                }
            }
            out.push_str(line);
            out.push('\n');
            continue;
        }

        if fence.is_some() {
            out.push_str(line);
            out.push('\n');
            continue;
        }

        // Outside code: strip comments, handling multi-line spans.
        let mut rest = line;
        let mut kept = String::new();
        loop {
            if in_comment {
                if let Some(end) = rest.find("-->") {
                    rest = &rest[end + 3..];
                    in_comment = false;
                } else {
                    // Comment continues past end of line; drop the remainder.
                    break;
                }
            } else if let Some(start) = rest.find("<!--") {
                kept.push_str(&rest[..start]);
                rest = &rest[start + 4..];
                in_comment = true;
            } else {
                kept.push_str(rest);
                break;
            }
        }

        // Only emit the line if it retained non-whitespace, or was originally
        // blank. This drops lines that were nothing but a comment.
        if !kept.trim().is_empty() || line.trim().is_empty() {
            out.push_str(&kept);
            out.push('\n');
        } else if !kept.is_empty() {
            // Kept only whitespace from a comment-only line: drop it.
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_block_comment_but_keeps_fenced() {
        let doc = "before\n<!-- secret -->\nafter\n```\n<!-- kept in code -->\n```\n";
        let out = strip_html_comments(doc);
        assert!(!out.contains("secret"));
        assert!(out.contains("before"));
        assert!(out.contains("after"));
        assert!(out.contains("<!-- kept in code -->"));
    }

    #[test]
    fn strips_multiline_comment() {
        let doc = "a\n<!-- line1\nline2 -->b\nc\n";
        let out = strip_html_comments(doc);
        assert!(!out.contains("line1"));
        assert!(!out.contains("line2"));
        assert!(out.contains('a'));
        assert!(out.contains('b'));
        assert!(out.contains('c'));
    }

    #[test]
    fn scan_skips_inline_backtick() {
        let tokens = scan_line_imports("see @real/file.md but not `@fake/file.md`");
        assert_eq!(tokens, vec!["real/file.md".to_string()]);
    }

    #[test]
    fn scan_ignores_mid_word_at() {
        let tokens = scan_line_imports("email me@example.com is not an import");
        assert!(tokens.is_empty());
    }

    #[test]
    fn fence_marker_detects_backticks_and_tildes() {
        assert_eq!(fence_marker("```rust"), Some("```".to_string()));
        assert_eq!(fence_marker("~~~"), Some("~~~".to_string()));
        assert_eq!(fence_marker("``inline"), None);
        assert_eq!(fence_marker("text"), None);
    }
}
