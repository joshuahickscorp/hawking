//! World model / project simulation (bible §11.8).
//!
//! **Tier 1 — lightweight STATIC simulation ([RESEARCH-PROVEN], build first).**
//! Before HIDE spends GPU tokens generating and applying an `edit_file(path,
//! diff)`, it can *statically* predict whether the post-edit text would build,
//! using cheap, subprocess-free checks over the resulting buffer:
//!
//!   * **balanced-delimiter** matching — every `(` `[` `{` is closed in the
//!     right order, with brace/bracket/paren-aware string and line-comment
//!     skipping so delimiters inside `"..."` / `// ...` don't count;
//!   * **parse-sanity** — when a tree-sitter grammar exists for the file's
//!     language, the post-edit text is parsed and the resulting tree is checked
//!     for `ERROR` / `MISSING` nodes (the real incremental-parse path the bible
//!     describes, minus the LSP type-check that lives behind the Tier-2 seam).
//!
//! This is the [`StaticSimulator`] implemented by [`StaticProjectSimulator`].
//! It is a *static analyzer*, not a learned forward model.
//!
//! **Tier 2 — dynamic learned world model ([MOONSHOT], documented seam).**
//! §11.8.3 proposes a learned `g(state, action) -> next_state` trained on the
//! user's git history (Dreamer-V3-for-projects). That is a 12–18-month research
//! project and is **not** implemented here. The key design constraint from the
//! bible is honored: the [`SimulationResult`] schema is shared between Tier 1
//! and Tier 2, so a future learned backend can implement the same
//! [`StaticSimulator`] trait and the Governor never learns which tier is active.
//! The Tier-2 backend is the [`LearnedForwardModel`] seam at the bottom of this
//! file — a typed placeholder that returns [`PredictedOutcome::Unknown`] until
//! the research matures.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use hawking_index::parse::grammars::{GrammarRegistry, LangId};

/// A request to simulate one edit (kept for serialized planning payloads).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationRequest {
    pub objective: String,
    pub changed_files: Vec<String>,
    pub assumptions: Vec<String>,
}

/// The §11.8.2 result schema — **shared between Tier 1 and Tier 2** so the
/// backend can be swapped without the Governor noticing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationResult {
    /// The simulator's best guess at the edit's outcome.
    pub predicted_outcome: PredictedOutcome,
    /// Confidence that the prediction is correct (0.0–1.0).
    pub confidence: f32,
    /// Specific issues detected (if any).
    pub issues: Vec<SimulatedIssue>,
}

impl SimulationResult {
    /// Convenience: a clean `Build` prediction with no issues.
    fn clean(confidence: f32) -> Self {
        Self {
            predicted_outcome: PredictedOutcome::Build,
            confidence,
            issues: Vec::new(),
        }
    }
}

/// The predicted outcome of applying an edit. Named per the Tier-1 task spec
/// (`Build | TestFail | TypeErr | Unknown`); the variants map onto the bible's
/// §11.8.2 `PredictedOutcome` (`Build` ≅ `LikelyClean`, `TypeErr` ≅
/// `PredictedTypeError` / `PredictedBuildBreak`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictedOutcome {
    /// The post-edit text is structurally sound — likely to build.
    Build,
    /// A test is predicted to fail (reserved for the dynamic Tier-2 backend,
    /// which can reason about behavioral regressions; the static tier does not
    /// emit this, since it cannot run tests).
    TestFail,
    /// A type / build error is predicted — unbalanced delimiters or a parse
    /// tree with `ERROR` / `MISSING` nodes.
    TypeErr,
    /// The simulator cannot reason about this edit (no grammar, empty buffer,
    /// or — for Tier 2 — an unseen state).
    Unknown,
}

/// A specific issue the simulator detected (§11.8.2 `SimulatedIssue`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimulatedIssue {
    pub kind: String,
    pub file: PathBuf,
    pub line: u32,
    pub message: String,
}

/// A snapshot of project state the simulator may consult. For Tier 1 this is
/// just the current on-disk text of the file being edited (so the diff can be
/// applied to it); the richer state (test results, type errors, dep graph) the
/// bible lists is what the Tier-2 forward model would consume.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProjectSnapshot {
    /// path -> current file text. The simulator looks up the edited file here.
    pub files: BTreeMap<PathBuf, String>,
}

impl ProjectSnapshot {
    pub fn with_file(path: impl Into<PathBuf>, text: impl Into<String>) -> Self {
        let mut files = BTreeMap::new();
        files.insert(path.into(), text.into());
        Self { files }
    }
}

/// §11.8.2 trait. Given a proposed edit, predict the outcome *before* GPU time
/// is spent generating it. The Governor calls this in the `SELECT_STEP -> ACT`
/// transition and re-plans if `predicted_outcome != Build && confidence > 0.7`.
pub trait StaticSimulator: Send + Sync {
    /// Predict whether applying `diff` to `path` would build cleanly.
    fn predict_edit(&self, path: &Path, diff: &str, context: &ProjectSnapshot) -> SimulationResult;
}

/// The Tier-1 STATIC simulator: balanced-delimiter + tree-sitter parse-sanity.
///
/// This is the "build first / research-proven" path. It runs no subprocess and
/// touches no filesystem — it applies the diff to an in-memory copy of the file
/// text and statically inspects the result.
#[derive(Debug, Default, Clone)]
pub struct StaticProjectSimulator;

impl StaticProjectSimulator {
    pub fn new() -> Self {
        Self
    }

    /// Apply a minimal unified diff to `base`, returning the post-edit text.
    ///
    /// Supports the two forms the agent's `edit_file` actually emits:
    ///   * a full-file replacement — a diff whose lines are all `+`-prefixed (no
    ///     context / removal lines) is treated as the new file body;
    ///   * a line-oriented unified diff — `+`/`-`/` ` lines (hunk `@@` headers
    ///     are ignored). Removed lines are dropped, added lines inserted, context
    ///     lines passed through, in diff order.
    ///
    /// This is deliberately lightweight: the goal is to reconstruct enough of the
    /// post-edit buffer to run *static* checks on it, not to be a conflict-aware
    /// patch engine.
    fn apply_diff(base: &str, diff: &str) -> String {
        let diff_lines: Vec<&str> = diff.lines().collect();
        let is_pure_add = !diff_lines.is_empty()
            && diff_lines
                .iter()
                .all(|l| l.starts_with('+') || l.is_empty());
        if is_pure_add {
            // Full-file replacement: strip the leading '+'.
            return diff_lines
                .iter()
                .map(|l| l.strip_prefix('+').unwrap_or(l))
                .collect::<Vec<_>>()
                .join("\n");
        }

        let mut out: Vec<String> = Vec::new();
        let mut used_diff = false;
        for line in &diff_lines {
            if line.starts_with("@@") || line.starts_with("+++") || line.starts_with("---") {
                continue;
            }
            if let Some(added) = line.strip_prefix('+') {
                out.push(added.to_string());
                used_diff = true;
            } else if line.starts_with('-') {
                used_diff = true; // removal — drop it
            } else if let Some(ctx) = line.strip_prefix(' ') {
                out.push(ctx.to_string());
                used_diff = true;
            } else {
                out.push((*line).to_string());
            }
        }
        if used_diff {
            out.join("\n")
        } else {
            // Nothing diff-shaped — treat the diff as a literal replacement body.
            base.to_string()
        }
    }

    /// Balanced-delimiter check over `text`, skipping `(){}[]` that appear inside
    /// double/single-quoted strings or `//` line comments. Returns the first
    /// imbalance as an issue.
    fn check_balanced(path: &Path, text: &str) -> Option<SimulatedIssue> {
        let mut stack: Vec<(char, u32)> = Vec::new();
        let mut in_string: Option<char> = None;
        let mut prev = '\0';

        for (lineno, line) in text.lines().enumerate() {
            let line_u32 = (lineno + 1) as u32;
            let mut chars = line.chars().peekable();
            while let Some(c) = chars.next() {
                if let Some(q) = in_string {
                    if c == q && prev != '\\' {
                        in_string = None;
                    }
                    prev = c;
                    continue;
                }
                match c {
                    '"' | '\'' => in_string = Some(c),
                    // `//` starts a line comment — ignore the rest of the line.
                    '/' if chars.peek() == Some(&'/') => break,
                    '(' | '[' | '{' => stack.push((c, line_u32)),
                    ')' | ']' | '}' => {
                        let expected = match c {
                            ')' => '(',
                            ']' => '[',
                            _ => '{',
                        };
                        match stack.pop() {
                            Some((open, _)) if open == expected => {}
                            Some((open, open_line)) => {
                                return Some(SimulatedIssue {
                                    kind: "unbalanced_delimiter".into(),
                                    file: path.to_path_buf(),
                                    line: line_u32,
                                    message: format!(
                                        "closing '{c}' on line {line_u32} does not match '{open}' opened on line {open_line}"
                                    ),
                                });
                            }
                            None => {
                                return Some(SimulatedIssue {
                                    kind: "unbalanced_delimiter".into(),
                                    file: path.to_path_buf(),
                                    line: line_u32,
                                    message: format!(
                                        "closing '{c}' on line {line_u32} has no matching opener"
                                    ),
                                });
                            }
                        }
                    }
                    _ => {}
                }
                prev = c;
            }
            prev = '\n';
        }

        if let Some((open, open_line)) = stack.last().copied() {
            return Some(SimulatedIssue {
                kind: "unbalanced_delimiter".into(),
                file: path.to_path_buf(),
                line: open_line,
                message: format!("'{open}' opened on line {open_line} is never closed"),
            });
        }
        None
    }

    /// Tree-sitter parse-sanity: parse `text` with the file's grammar and report
    /// the first `ERROR` / `MISSING` node as an issue. Returns `Ok(None)` if the
    /// tree is clean, `Ok(Some(issue))` if it has a syntax error, and `Err(())`
    /// if no grammar is registered for this language (caller treats that as
    /// "cannot reason" / Unknown territory rather than a failure).
    #[allow(clippy::result_unit_err)]
    fn check_parse(path: &Path, text: &str) -> std::result::Result<Option<SimulatedIssue>, ()> {
        let lang = LangId::from_path(path);
        if !lang.is_known() {
            return Err(());
        }
        let bundle = GrammarRegistry::bundle(lang).ok_or(())?;
        let mut parser = tree_sitter::Parser::new();
        if parser.set_language(&bundle.language).is_err() {
            return Err(());
        }
        let Some(tree) = parser.parse(text, None) else {
            return Err(());
        };
        let root = tree.root_node();
        if !root.has_error() {
            return Ok(None);
        }
        // Walk to the first ERROR / MISSING node for a precise line.
        let mut cursor = root.walk();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if node.is_error() || node.is_missing() {
                let line = (node.start_position().row + 1) as u32;
                return Ok(Some(SimulatedIssue {
                    kind: if node.is_missing() {
                        "missing_node".into()
                    } else {
                        "parse_error".into()
                    },
                    file: path.to_path_buf(),
                    line,
                    message: format!(
                        "{} parse {} at line {line}",
                        lang.as_str(),
                        if node.is_missing() {
                            "MISSING"
                        } else {
                            "ERROR"
                        }
                    ),
                }));
            }
            for child in node.children(&mut cursor) {
                stack.push(child);
            }
        }
        // has_error() was true but we couldn't localize it — still a failure.
        Ok(Some(SimulatedIssue {
            kind: "parse_error".into(),
            file: path.to_path_buf(),
            line: 1,
            message: format!("{} parse tree contains errors", lang.as_str()),
        }))
    }
}

impl StaticSimulator for StaticProjectSimulator {
    fn predict_edit(&self, path: &Path, diff: &str, context: &ProjectSnapshot) -> SimulationResult {
        let base = context.files.get(path).map(String::as_str).unwrap_or("");
        let post = Self::apply_diff(base, diff);

        if post.trim().is_empty() {
            return SimulationResult {
                predicted_outcome: PredictedOutcome::Unknown,
                confidence: 0.5,
                issues: Vec::new(),
            };
        }

        // 1) Cheap, language-agnostic balance check.
        if let Some(issue) = Self::check_balanced(path, &post) {
            return SimulationResult {
                predicted_outcome: PredictedOutcome::TypeErr,
                // Delimiter imbalance is a near-certain build break.
                confidence: 0.95,
                issues: vec![issue],
            };
        }

        // 2) Grammar parse-sanity (when a grammar exists).
        match Self::check_parse(path, &post) {
            Ok(Some(issue)) => SimulationResult {
                predicted_outcome: PredictedOutcome::TypeErr,
                confidence: 0.85,
                issues: vec![issue],
            },
            // Clean parse AND balanced delimiters → high-confidence Build.
            Ok(None) => SimulationResult::clean(0.9),
            // No grammar: balanced delimiters is the only signal we have. That is
            // weak on its own, so report Build with low confidence (below the
            // Governor's 0.7 re-plan gate it stays Build, but the low confidence
            // tells the planner not to over-trust it).
            Err(()) => SimulationResult::clean(0.6),
        }
    }
}

// ============================================================================
// Tier 2 — dynamic learned world model ([MOONSHOT], documented seam).
// ============================================================================

/// **POST-SHELL MOONSHOT SEAM (bible §11.8.3 Tier 2).**
///
/// A learned forward model `g(state, action) -> next_state` trained on the
/// user's git history (the "Dreamer V3 for software projects" idea). This is a
/// 12–18-month research project; it is **not implemented**. It exists here only
/// to lock in the shared [`SimulationResult`] schema and the [`StaticSimulator`]
/// interface so a future learned backend drops in transparently. Until the
/// research matures, [`predict_edit`](StaticSimulator::predict_edit) returns
/// [`PredictedOutcome::Unknown`].
#[derive(Debug, Default, Clone)]
pub struct LearnedForwardModel {
    _seam: (),
}

impl StaticSimulator for LearnedForwardModel {
    fn predict_edit(
        &self,
        _path: &Path,
        _diff: &str,
        _context: &ProjectSnapshot,
    ) -> SimulationResult {
        // Tier-2 moonshot: no learned model yet → defer to Unknown.
        SimulationResult {
            predicted_outcome: PredictedOutcome::Unknown,
            confidence: 0.0,
            issues: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_edit_predicts_build() {
        let sim = StaticProjectSimulator::new();
        let snap = ProjectSnapshot::with_file("src/lib.rs", "");
        let diff = "+pub fn add(a: i32, b: i32) -> i32 {\n+    a + b\n+}\n";
        let res = sim.predict_edit(Path::new("src/lib.rs"), diff, &snap);
        assert_eq!(res.predicted_outcome, PredictedOutcome::Build);
        assert!(res.issues.is_empty());
        assert!(res.confidence >= 0.7, "confidence was {}", res.confidence);
    }

    #[test]
    fn unbalanced_delimiters_predict_failure_with_issue() {
        let sim = StaticProjectSimulator::new();
        let snap = ProjectSnapshot::with_file("src/lib.rs", "");
        // Missing the closing brace of the function.
        let diff = "+pub fn broken() {\n+    let x = (1 + 2;\n";
        let res = sim.predict_edit(Path::new("src/lib.rs"), diff, &snap);
        assert_eq!(res.predicted_outcome, PredictedOutcome::TypeErr);
        assert!(
            !res.issues.is_empty(),
            "an unbalanced edit must report at least one issue"
        );
        assert_eq!(res.issues[0].kind, "unbalanced_delimiter");
    }

    #[test]
    fn broken_parse_predicts_failure() {
        // Balanced delimiters, but not valid Rust — tree-sitter flags it.
        let sim = StaticProjectSimulator::new();
        let snap = ProjectSnapshot::with_file("src/lib.rs", "");
        let diff = "+pub fn oops( {}\n";
        let res = sim.predict_edit(Path::new("src/lib.rs"), diff, &snap);
        assert_eq!(res.predicted_outcome, PredictedOutcome::TypeErr);
        assert!(!res.issues.is_empty());
    }

    #[test]
    fn delimiters_inside_strings_and_comments_are_ignored() {
        let sim = StaticProjectSimulator::new();
        let snap = ProjectSnapshot::with_file("src/lib.rs", "");
        // The unmatched ')' and '{' live inside a string and a comment.
        let diff = "+pub fn ok() {\n+    let s = \") not a real paren {\";\n+    // and a } here too\n+}\n";
        let res = sim.predict_edit(Path::new("src/lib.rs"), diff, &snap);
        assert_eq!(res.predicted_outcome, PredictedOutcome::Build);
    }

    #[test]
    fn unknown_language_still_balance_checks() {
        let sim = StaticProjectSimulator::new();
        let snap = ProjectSnapshot::default();
        // No grammar for `.txt` → balanced text is Build (low confidence)...
        let ok = sim.predict_edit(Path::new("notes.txt"), "+all good\n", &snap);
        assert_eq!(ok.predicted_outcome, PredictedOutcome::Build);
        assert!(ok.confidence < 0.7);
        // ...but an imbalance is still caught language-agnostically.
        let bad = sim.predict_edit(Path::new("notes.txt"), "+oops (unclosed\n", &snap);
        assert_eq!(bad.predicted_outcome, PredictedOutcome::TypeErr);
    }

    #[test]
    fn empty_edit_is_unknown() {
        let sim = StaticProjectSimulator::new();
        let snap = ProjectSnapshot::default();
        let res = sim.predict_edit(Path::new("src/lib.rs"), "+\n", &snap);
        assert_eq!(res.predicted_outcome, PredictedOutcome::Unknown);
    }

    #[test]
    fn tier2_seam_returns_unknown() {
        let model = LearnedForwardModel::default();
        let snap = ProjectSnapshot::default();
        let res = model.predict_edit(Path::new("src/lib.rs"), "+pub fn f() {}\n", &snap);
        assert_eq!(res.predicted_outcome, PredictedOutcome::Unknown);
        assert_eq!(res.confidence, 0.0);
    }
}
