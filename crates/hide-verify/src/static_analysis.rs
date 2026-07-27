//! A real, deterministic static-analysis oracle over Rust source text.
//!
//! This is a genuine Tier1 deterministic check: it runs entirely in-process over
//! source strings (or a walked directory), with NO model and NO subprocess, and
//! the same input always yields the same findings. It is a lint, not a compiler:
//! it works over a light lexical model of the source (comments and string
//! literals are masked so braces and identifiers inside them do not count).
//!
//! Checks:
//! * `unwrap()` / `expect()` used outside `#[cfg(test)]` / `#[test]` code.
//! * `panic!` / `todo!` / `unimplemented!` / `unreachable!` marker macros.
//! * en dash (U+2013) / em dash (U+2014) presence: the house-rule lint.
//! * a very-long-function heuristic (body line count over a threshold).
//! * `TODO` / `FIXME` / `XXX` markers.

use std::path::Path;

use regex::Regex;
use walkdir::WalkDir;

use crate::error::{Result, VerifyError};
use crate::finding::{CheckKind, Finding, Severity};
use crate::oracle::{
    Evidence, Oracle, OracleClass, OracleOutcome, SourceFile, VerificationInput, Verdict,
};
use crate::tier::VerificationTier;

/// En dash. Referenced by codepoint so this source file never contains the
/// banned character itself.
const EN_DASH: char = '\u{2013}';
/// Em dash. Referenced by codepoint for the same reason.
const EM_DASH: char = '\u{2014}';

/// Default line-count threshold above which a function body is flagged as long.
pub const DEFAULT_LONG_FUNCTION_THRESHOLD: usize = 80;

/// The deterministic static-analysis oracle (Tier1, Deterministic class).
pub struct StaticAnalysisOracle {
    long_function_threshold: usize,
    unwrap_re: Regex,
    macro_re: Regex,
    todo_re: Regex,
}

impl Default for StaticAnalysisOracle {
    fn default() -> Self {
        Self {
            long_function_threshold: DEFAULT_LONG_FUNCTION_THRESHOLD,
            // A method call: a dot, optional space, then `unwrap`/`expect`, then
            // an open paren. `\b` keeps `unwrap_or`, `expect_err`, etc. clear.
            unwrap_re: Regex::new(r"\.\s*(unwrap|expect)\b\s*\(").expect("static unwrap regex"),
            // A marker macro invocation: name immediately followed by `!`.
            macro_re: Regex::new(r"\b(panic|todo|unimplemented|unreachable)\s*!")
                .expect("static macro regex"),
            // Uppercase-only so the lowercase `todo!` macro is not double-counted.
            todo_re: Regex::new(r"\b(TODO|FIXME|XXX)\b").expect("static todo regex"),
        }
    }
}

impl StaticAnalysisOracle {
    pub fn new() -> Self {
        Self::default()
    }

    /// An oracle with a custom long-function threshold.
    pub fn with_long_function_threshold(threshold: usize) -> Self {
        Self {
            long_function_threshold: threshold,
            ..Self::default()
        }
    }

    pub fn long_function_threshold(&self) -> usize {
        self.long_function_threshold
    }

    /// Analyze a single source file, returning findings sorted by (line, check).
    pub fn analyze_source(&self, file: &str, source: &str) -> Vec<Finding> {
        let mut out = Vec::new();

        // 1. En/em dash over the RAW text: the house rule bans them everywhere,
        //    including inside comments and strings.
        self.scan_dashes(file, source, &mut out);

        // 2. TODO/FIXME markers over the RAW text (they usually live in comments).
        self.scan_todo(file, source, &mut out);

        // 3. Structural checks over a comment- and string-masked view: unwrap /
        //    expect outside test code, marker macros, and long functions. The
        //    mask also yields, per line, whether we are inside test code.
        let masked = mask_comments_and_strings(source);
        let masked_lines: Vec<&str> = masked.lines().collect();
        let in_test = self.scan_structure(file, &masked_lines, &mut out);
        self.scan_unwrap(file, &masked_lines, &in_test, &mut out);
        self.scan_macros(file, &masked_lines, &mut out);

        out.sort_by(|a, b| {
            a.line
                .cmp(&b.line)
                .then(a.check.cmp(&b.check))
                .then(a.message.cmp(&b.message))
        });
        out
    }

    /// Analyze a set of in-memory source files.
    pub fn analyze_sources(&self, sources: &[SourceFile]) -> Vec<Finding> {
        sources
            .iter()
            .flat_map(|s| self.analyze_source(&s.path, &s.text))
            .collect()
    }

    /// Walk `root`, analyze every `*.rs` file, and return the combined findings.
    /// Deterministic: entries are visited in sorted order.
    pub fn analyze_dir(&self, root: &Path) -> Result<Vec<Finding>> {
        let mut out = Vec::new();
        for entry in WalkDir::new(root).sort_by_file_name() {
            let entry = entry.map_err(|e| VerifyError::Walk {
                root: root.display().to_string(),
                message: e.to_string(),
            })?;
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let text = std::fs::read_to_string(path).map_err(|e| VerifyError::Read {
                path: path.display().to_string(),
                message: e.to_string(),
            })?;
            out.extend(self.analyze_source(&path.display().to_string(), &text));
        }
        Ok(out)
    }

    fn scan_dashes(&self, file: &str, source: &str, out: &mut Vec<Finding>) {
        for (idx, line) in source.lines().enumerate() {
            for ch in line.chars() {
                if ch == EN_DASH || ch == EM_DASH {
                    let which = if ch == EN_DASH { "en" } else { "em" };
                    out.push(Finding::new(
                        CheckKind::EmDash,
                        file,
                        (idx + 1) as u32,
                        Severity::Error,
                        format!(
                            "house-rule violation: {which} dash (U+{:04X}) is banned; use a hyphen",
                            ch as u32
                        ),
                    ));
                }
            }
        }
    }

    fn scan_todo(&self, file: &str, source: &str, out: &mut Vec<Finding>) {
        for (idx, line) in source.lines().enumerate() {
            for m in self.todo_re.find_iter(line) {
                out.push(Finding::new(
                    CheckKind::TodoMarker,
                    file,
                    (idx + 1) as u32,
                    Severity::Info,
                    format!("`{}` marker found", m.as_str()),
                ));
            }
        }
    }

    fn scan_unwrap(
        &self,
        file: &str,
        lines: &[&str],
        in_test: &[bool],
        out: &mut Vec<Finding>,
    ) {
        for (idx, line) in lines.iter().enumerate() {
            if in_test.get(idx).copied().unwrap_or(false) {
                continue;
            }
            for caps in self.unwrap_re.captures_iter(line) {
                let which = &caps[1];
                out.push(Finding::new(
                    CheckKind::UnwrapOutsideTest,
                    file,
                    (idx + 1) as u32,
                    Severity::Warning,
                    format!("`{which}()` used outside test code; handle the error explicitly"),
                ));
            }
        }
    }

    fn scan_macros(&self, file: &str, lines: &[&str], out: &mut Vec<Finding>) {
        for (idx, line) in lines.iter().enumerate() {
            for caps in self.macro_re.captures_iter(line) {
                let name = &caps[1];
                let severity = match name {
                    "todo" | "unimplemented" => Severity::Error,
                    _ => Severity::Warning,
                };
                out.push(Finding::new(
                    CheckKind::PanicMarker,
                    file,
                    (idx + 1) as u32,
                    severity,
                    format!("`{name}!` marker macro"),
                ));
            }
        }
    }

    /// Single structural pass over masked lines. Tracks brace depth to (a) decide
    /// which lines are inside `#[cfg(test)]` / `#[test]` code and (b) measure
    /// function body lengths. Long-function findings are pushed to `out`; the
    /// per-line "in test" vector is returned for the unwrap check.
    fn scan_structure(&self, file: &str, lines: &[&str], out: &mut Vec<Finding>) -> Vec<bool> {
        let mut in_test = vec![false; lines.len()];

        let mut depth: i32 = 0;
        // Brace levels at which an active test region was opened.
        let mut test_stack: Vec<i32> = Vec::new();
        // A test attribute was seen and is waiting for the block it guards.
        let mut armed_test = false;
        // A `fn` token was seen and is waiting for its body's opening brace.
        let mut armed_fn_line: Option<u32> = None;
        // Open function bodies: (signature line, brace level of the enclosing block).
        let mut fn_stack: Vec<(u32, i32)> = Vec::new();

        for (idx, line) in lines.iter().enumerate() {
            in_test[idx] = !test_stack.is_empty();

            if line.contains("#[test]")
                || line.contains("#[tokio::test]")
                || line.contains("#[cfg(test)]")
            {
                armed_test = true;
            }

            let mut ident = String::new();
            for c in line.chars() {
                if c.is_alphanumeric() || c == '_' {
                    ident.push(c);
                    continue;
                }
                if ident == "fn" && armed_fn_line.is_none() {
                    armed_fn_line = Some((idx + 1) as u32);
                }
                ident.clear();

                match c {
                    '{' => {
                        let level = depth;
                        depth += 1;
                        if armed_test {
                            test_stack.push(level);
                            armed_test = false;
                        }
                        if let Some(sig) = armed_fn_line.take() {
                            fn_stack.push((sig, level));
                        }
                    }
                    '}' => {
                        depth -= 1;
                        if let Some(&(sig, level)) = fn_stack.last() {
                            if depth == level {
                                fn_stack.pop();
                                let body_len = (idx as i64 + 1) - sig as i64;
                                if body_len > self.long_function_threshold as i64 {
                                    out.push(Finding::new(
                                        CheckKind::LongFunction,
                                        file,
                                        sig,
                                        Severity::Warning,
                                        format!(
                                            "function body spans {body_len} lines (threshold {}); \
                                             consider splitting it",
                                            self.long_function_threshold
                                        ),
                                    ));
                                }
                            }
                        }
                        if let Some(&level) = test_stack.last() {
                            if depth == level {
                                test_stack.pop();
                            }
                        }
                    }
                    ';' => {
                        // A `fn foo();` declaration or `#[cfg(test)] mod m;`
                        // ended without opening a block: disarm.
                        armed_fn_line = None;
                        armed_test = false;
                    }
                    _ => {}
                }
            }
            // A trailing `fn` at end of line (no delimiter after it).
            if ident == "fn" && armed_fn_line.is_none() {
                armed_fn_line = Some((idx + 1) as u32);
            }
        }

        in_test
    }
}

impl Oracle for StaticAnalysisOracle {
    fn name(&self) -> &str {
        "static_analysis"
    }

    fn tier(&self) -> VerificationTier {
        VerificationTier::Tier1Deterministic
    }

    fn class(&self) -> OracleClass {
        OracleClass::Deterministic
    }

    fn evaluate(&self, input: &VerificationInput) -> OracleOutcome {
        let mut findings = self.analyze_sources(&input.sources);
        let mut notes = Vec::new();

        if let Some(root) = &input.root {
            match self.analyze_dir(root) {
                Ok(dir_findings) => findings.extend(dir_findings),
                Err(e) => notes.push(format!("directory scan skipped: {e}")),
            }
        }

        findings.sort_by(|a, b| {
            a.file
                .cmp(&b.file)
                .then(a.line.cmp(&b.line))
                .then(a.check.cmp(&b.check))
                .then(a.message.cmp(&b.message))
        });

        // A finding at or above Warning severity fails the gate. Info-only
        // findings (bare TODO markers) do not, but are still reported.
        let blocking: Vec<String> = findings
            .iter()
            .filter(|f| f.severity >= Severity::Warning)
            .map(|f| format!("{}:{} {}", f.file, f.line, f.message))
            .collect();

        let verdict = if blocking.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail { reasons: blocking }
        };

        OracleOutcome {
            verdict,
            evidence: Evidence { findings, notes },
        }
    }
}

/// Return a copy of `source` with line comments, block comments, string
/// literals, and char literals replaced by spaces, while preserving every
/// newline so line numbers are unchanged. Lifetimes (`'a`) are left intact so
/// they are not mistaken for char literals.
///
/// This is a lexical approximation, not a Rust parser: raw strings with embedded
/// quotes are not handled specially. That is acceptable for a lint whose job is
/// to keep braces and identifiers inside strings and comments from skewing the
/// structural pass.
fn mask_comments_and_strings(source: &str) -> String {
    let chars: Vec<char> = source.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(source.len());
    let mut i = 0;

    while i < n {
        let c = chars[i];

        // Line comment: to end of line.
        if c == '/' && i + 1 < n && chars[i + 1] == '/' {
            while i < n && chars[i] != '\n' {
                out.push(' ');
                i += 1;
            }
            continue;
        }

        // Block comment: to the closing `*/` (may span lines).
        if c == '/' && i + 1 < n && chars[i + 1] == '*' {
            out.push(' ');
            out.push(' ');
            i += 2;
            while i < n && !(chars[i] == '*' && i + 1 < n && chars[i + 1] == '/') {
                out.push(if chars[i] == '\n' { '\n' } else { ' ' });
                i += 1;
            }
            if i < n {
                out.push(' ');
                out.push(' ');
                i += 2;
            }
            continue;
        }

        // String literal.
        if c == '"' {
            out.push(' ');
            i += 1;
            while i < n {
                if chars[i] == '\\' {
                    out.push(' ');
                    i += 1;
                    if i < n {
                        out.push(if chars[i] == '\n' { '\n' } else { ' ' });
                        i += 1;
                    }
                    continue;
                }
                if chars[i] == '"' {
                    out.push(' ');
                    i += 1;
                    break;
                }
                out.push(if chars[i] == '\n' { '\n' } else { ' ' });
                i += 1;
            }
            continue;
        }

        // Char literal vs lifetime. `'\...` or `'x'` is a char literal; `'a`
        // followed by an identifier that does not immediately close is a lifetime.
        if c == '\'' {
            let is_char_lit = (i + 1 < n && chars[i + 1] == '\\')
                || (i + 2 < n && chars[i + 2] == '\'');
            if is_char_lit {
                out.push(' ');
                i += 1;
                while i < n {
                    if chars[i] == '\\' {
                        out.push(' ');
                        i += 1;
                        if i < n {
                            out.push(' ');
                            i += 1;
                        }
                        continue;
                    }
                    if chars[i] == '\'' {
                        out.push(' ');
                        i += 1;
                        break;
                    }
                    out.push(' ');
                    i += 1;
                }
                continue;
            }
            // Lifetime tick: keep it as ordinary punctuation.
            out.push(c);
            i += 1;
            continue;
        }

        out.push(c);
        i += 1;
    }

    out
}
