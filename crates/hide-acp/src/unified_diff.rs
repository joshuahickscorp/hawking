//! A small deterministic unified-diff reader used by the edit/diff projection.
//!
//! ACP `diff` content carries `path`, an optional `oldText`, and a `newText`.
//! HIDE items carry diffs two ways: a `Patch` (a unified-diff string over one or
//! more files) and a per-file `Diff` (structured hunks). This module turns both
//! into the text pair ACP wants.
//!
//! Reconstruction is over the HUNK WINDOWS only: it concatenates, in order, the
//! context and removed lines (the old side) and the context and added lines (the
//! new side). For a single-hunk change this is exact; for a multi-hunk file it
//! is the union of the changed windows, not the whole file. The whole-file text
//! would require reading the live workspace, which the model-free boundary does
//! not do. Nothing here runs a model.

/// One file's reconstructed change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedFile {
    pub path: String,
    /// `None` when the old side is a new file (`/dev/null`).
    pub old_text: Option<String>,
    pub new_text: String,
}

/// Reconstruct `(old_text, new_text)` from a single hide `DiffHunk` body. Each
/// line is prefixed ' ' (context), '+' (added), or '-' (removed).
pub fn reconstruct_hunk(body: &str) -> (String, String) {
    let mut old = String::new();
    let mut new = String::new();
    for line in body.lines() {
        let mut chars = line.chars();
        match chars.next() {
            Some(' ') => {
                let rest = &line[1..];
                push_line(&mut old, rest);
                push_line(&mut new, rest);
            }
            Some('+') => push_line(&mut new, &line[1..]),
            Some('-') => push_line(&mut old, &line[1..]),
            // "\ No newline at end of file" and blank lines carry no content.
            Some('\\') => {}
            None => {}
            // A bare line with no diff marker is treated as context.
            Some(_) => {
                push_line(&mut old, line);
                push_line(&mut new, line);
            }
        }
    }
    (old, new)
}

fn push_line(buf: &mut String, line: &str) {
    buf.push_str(line);
    buf.push('\n');
}

/// Parse a unified-diff string into a per-file change list. Recognizes the
/// `diff --git`, `--- `, `+++ `, and `@@` headers of standard unified diffs.
pub fn parse_unified_diff(diff: &str) -> Vec<ParsedFile> {
    let mut files: Vec<ParsedFile> = Vec::new();
    let mut cur_path: Option<String> = None;
    let mut old_is_dev_null = false;
    let mut in_hunk = false;
    let mut old = String::new();
    let mut new = String::new();

    // Commit whatever has accumulated for the current file into `files`.
    fn flush(
        files: &mut Vec<ParsedFile>,
        path: &Option<String>,
        old_is_dev_null: bool,
        old: &mut String,
        new: &mut String,
    ) {
        if let Some(p) = path {
            files.push(ParsedFile {
                path: p.clone(),
                old_text: if old_is_dev_null {
                    None
                } else {
                    Some(std::mem::take(old))
                },
                new_text: std::mem::take(new),
            });
        }
        old.clear();
        new.clear();
    }

    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("--- ") {
            // Start of a new file section: flush the previous one.
            flush(&mut files, &cur_path, old_is_dev_null, &mut old, &mut new);
            old_is_dev_null = rest.trim() == "/dev/null";
            in_hunk = false;
            continue;
        }
        if let Some(rest) = line.strip_prefix("+++ ") {
            cur_path = Some(normalize_path(rest.trim()));
            in_hunk = false;
            continue;
        }
        if line.starts_with("diff --git") || line.starts_with("index ") {
            // Header noise between files; ignore.
            continue;
        }
        if line.starts_with("@@") {
            in_hunk = true;
            continue;
        }
        if !in_hunk {
            continue;
        }
        match line.chars().next() {
            Some(' ') => {
                let rest = &line[1..];
                push_line(&mut old, rest);
                push_line(&mut new, rest);
            }
            Some('+') => push_line(&mut new, &line[1..]),
            Some('-') => push_line(&mut old, &line[1..]),
            Some('\\') => {}
            None => {}
            Some(_) => {
                push_line(&mut old, line);
                push_line(&mut new, line);
            }
        }
    }
    flush(&mut files, &cur_path, old_is_dev_null, &mut old, &mut new);
    files
}

/// Strip a `a/` or `b/` VCS prefix from a diff path.
fn normalize_path(path: &str) -> String {
    if path == "/dev/null" {
        return path.to_string();
    }
    path.strip_prefix("a/")
        .or_else(|| path.strip_prefix("b/"))
        .unwrap_or(path)
        .to_string()
}
