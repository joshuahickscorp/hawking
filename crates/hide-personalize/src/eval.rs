//! Local eval harness + autonomous eval mining (bible §11.3).
//!
//! Two real things here:
//!
//!   * [`run_eval`] — actually **executes** an [`EvalOracle`]. A `Command`
//!     oracle is spawned as a real subprocess and its exit code checked; a
//!     `Regex` oracle compiles and matches against produced output; a
//!     `GoldenDiff` oracle blake3-hashes the produced diff and compares it to
//!     the recorded golden hash. The result is a real [`EvalResult`], not a
//!     hardcoded pass.
//!
//!   * [`EvalMiner`] — scans the codebase index for **functions that have no
//!     test linkage** and mints `WriteTest` eval candidates for them (§11.3.2,
//!     the "function has no test file linkage" heuristic). Wires the
//!     declared-but-previously-unused `hawking-index`.

use crate::records::Hash32;
use hide_core::ids::now_micros;
use hide_core::Result;
use hawking_index::{CodeIndex, SearchQuery};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalCase {
    pub id: String,
    pub task: String,
    pub oracle: EvalOracle,
    pub metadata: BTreeMap<String, String>,
}

/// How pass/fail is determined for a case (§11.3.2 `OracleSpec`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalOracle {
    /// Run `argv` (in `cwd` if set); pass iff the exit code equals
    /// `expected_exit`.
    Command {
        argv: Vec<String>,
        #[serde(default)]
        cwd: Option<PathBuf>,
        #[serde(default)]
        expected_exit: i32,
    },
    /// blake3 of the produced diff must equal this hex digest.
    GoldenDiff { diff_hash: String },
    /// The produced output must match this regular expression.
    Regex { pattern: String },
    /// Requires a human; [`run_eval`] returns an inconclusive result.
    Human,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalResult {
    pub case_id: String,
    pub passed: bool,
    pub score: f32,
    pub detail: String,
}

impl EvalResult {
    fn pass(case_id: &str, detail: impl Into<String>) -> Self {
        Self {
            case_id: case_id.to_string(),
            passed: true,
            score: 1.0,
            detail: detail.into(),
        }
    }
    fn fail(case_id: &str, detail: impl Into<String>) -> Self {
        Self {
            case_id: case_id.to_string(),
            passed: false,
            score: 0.0,
            detail: detail.into(),
        }
    }
}

/// Run a single case's oracle. `produced` is the agent's output (the produced
/// diff or text) — used by the `Regex` and `GoldenDiff` oracles; the `Command`
/// oracle ignores it and runs the process instead.
pub fn run_eval(case: &EvalCase, produced: &str) -> EvalResult {
    match &case.oracle {
        EvalOracle::Command {
            argv,
            cwd,
            expected_exit,
        } => run_command_oracle(&case.id, argv, cwd.as_deref(), *expected_exit),
        EvalOracle::GoldenDiff { diff_hash } => {
            let got = Hash32::of(produced).to_hex();
            if &got == diff_hash {
                EvalResult::pass(&case.id, "diff matches golden hash")
            } else {
                EvalResult::fail(&case.id, format!("golden mismatch: got {got}"))
            }
        }
        EvalOracle::Regex { pattern } => match Regex::new(pattern) {
            Ok(re) if re.is_match(produced) => {
                EvalResult::pass(&case.id, format!("matched /{pattern}/"))
            }
            Ok(_) => EvalResult::fail(&case.id, format!("no match for /{pattern}/")),
            Err(err) => EvalResult::fail(&case.id, format!("invalid regex: {err}")),
        },
        EvalOracle::Human => EvalResult {
            case_id: case.id.clone(),
            passed: false,
            score: 0.5,
            detail: "human oracle — pending review".to_string(),
        },
    }
}

fn run_command_oracle(
    case_id: &str,
    argv: &[String],
    cwd: Option<&std::path::Path>,
    expected_exit: i32,
) -> EvalResult {
    let Some((program, args)) = argv.split_first() else {
        return EvalResult::fail(case_id, "empty argv");
    };
    let mut cmd = Command::new(program);
    cmd.args(args);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    match cmd.output() {
        Ok(out) => {
            let code = out.status.code().unwrap_or(-1);
            if code == expected_exit {
                EvalResult::pass(case_id, format!("exit {code} == expected"))
            } else {
                let stderr = String::from_utf8_lossy(&out.stderr);
                let tail: String = stderr.chars().rev().take(200).collect::<String>();
                let tail: String = tail.chars().rev().collect();
                EvalResult::fail(
                    case_id,
                    format!("exit {code} != {expected_exit}; stderr tail: {tail}"),
                )
            }
        }
        Err(err) => EvalResult::fail(case_id, format!("spawn failed: {err}")),
    }
}

/// Run a whole suite, returning one result per case.
pub fn run_suite(cases: &[EvalCase], produced: &str) -> Vec<EvalResult> {
    cases.iter().map(|c| run_eval(c, produced)).collect()
}

// ============================================================================
// EvalMiner — autonomous eval generation (§11.3.2)
// ============================================================================

/// A mined eval-task candidate (§11.3.2 `EvalTaskCandidate`, trimmed to what the
/// miner can determine without the full Living Index daemon).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalTaskCandidate {
    pub id: String,
    pub task_description: String,
    pub oracle: EvalOracle,
    pub source_files: Vec<PathBuf>,
    /// 0.0–1.0; the "function has no test linkage" heuristic is 0.90.
    pub confidence: f32,
    pub mined_at_us: u64,
    pub status: CandidateStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStatus {
    PendingHuman,
    AutoAdded,
}

#[derive(Debug, Clone)]
pub struct EvalMinerConfig {
    /// Candidates at or above this confidence are auto-added (no human gate);
    /// below it they surface as suggestions. Default 0.95 (§11.3.2).
    pub auto_add_confidence_threshold: f32,
    /// Rate-limit on auto-adds per invocation (the daemon's per-day cap).
    pub max_auto_add: usize,
    /// Substrings that disqualify a function name from minting a candidate
    /// (generated code, test functions themselves, etc.).
    pub name_blocklist: Vec<String>,
}

impl Default for EvalMinerConfig {
    fn default() -> Self {
        Self {
            auto_add_confidence_threshold: 0.95,
            max_auto_add: 20,
            name_blocklist: vec!["test_".into(), "bench_".into()],
        }
    }
}

/// Mints `WriteTest` eval candidates for functions with no test linkage.
///
/// The miner queries the `CodeIndex` for definitions (the `pub fn ` / `fn `
/// symbols tree-sitter extracted) and for each one asks whether a *test*
/// references it. "Test reference" is approximated by: a search hit whose path
/// looks like a test (`tests/`, `_test`, `#[test]`-adjacent) and that mentions
/// the function name. A function with zero such hits is a candidate.
pub struct EvalMiner {
    config: EvalMinerConfig,
}

impl EvalMiner {
    pub fn new(config: EvalMinerConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(EvalMinerConfig::default())
    }

    /// Scan `index` for the given function names and mint candidates for those
    /// with no test linkage. `candidate_fns` is the set of public function names
    /// to consider (the caller supplies them from a symbol enumeration — the
    /// index's `search` is name-keyed, so we drive it per name).
    pub async fn mine<I: CodeIndex>(
        &self,
        index: &I,
        candidate_fns: &[String],
    ) -> Result<Vec<EvalTaskCandidate>> {
        let mut out = Vec::new();
        let mut auto_added = 0usize;

        for name in candidate_fns {
            if self
                .config
                .name_blocklist
                .iter()
                .any(|b| name.contains(b))
            {
                continue;
            }
            if self.has_test_linkage(index, name).await? {
                continue;
            }

            // Locate the function's defining file (best-effort) for source_files.
            let defs = index.definition(name).await?;
            let source_files: Vec<PathBuf> =
                defs.iter().map(|o| PathBuf::from(&o.file)).collect();

            let confidence = 0.90; // §11.3.2: "function has no test linkage".
            let status = if confidence >= self.config.auto_add_confidence_threshold
                && auto_added < self.config.max_auto_add
            {
                auto_added += 1;
                CandidateStatus::AutoAdded
            } else {
                CandidateStatus::PendingHuman
            };

            out.push(EvalTaskCandidate {
                id: format!("mined-{name}-{}", now_micros()),
                task_description: format!(
                    "Write a test for `{name}` — it currently has no test linkage."
                ),
                // The oracle is "the new test passes": a `cargo test` filtered to
                // the function name. The exact filter is the function name so the
                // harness runs only the relevant test.
                oracle: EvalOracle::Command {
                    argv: vec![
                        "cargo".into(),
                        "test".into(),
                        name.clone(),
                    ],
                    cwd: None,
                    expected_exit: 0,
                },
                source_files,
                confidence,
                mined_at_us: now_micros(),
                status,
            });
        }
        Ok(out)
    }

    /// Does any test reference this function? A hit in a path that looks like a
    /// test and mentions the name counts as linkage.
    async fn has_test_linkage<I: CodeIndex>(&self, index: &I, name: &str) -> Result<bool> {
        let hits = index
            .search(SearchQuery {
                text: name.to_string(),
                limit: 50,
                include_symbols: true,
                include_lexical: true,
                include_semantic: false,
            })
            .await?;
        Ok(hits.iter().any(|h| is_test_path(&h.span.path)))
    }
}

fn is_test_path(path: &std::path::Path) -> bool {
    let s = path.to_string_lossy();
    s.contains("/tests/")
        || s.starts_with("tests/")
        || s.contains("_test.")
        || s.ends_with("_test.rs")
        || s.contains("test_")
}

// ============================================================================
// The §11.1.4 accept-rate gate.
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterGateReport {
    pub base_accept_rate: f32,
    pub candidate_accept_rate: f32,
    pub min_delta: f32,
}

impl AdapterGateReport {
    pub fn passes(&self) -> bool {
        self.candidate_accept_rate >= self.base_accept_rate + self.min_delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hawking_index::InMemoryCodeIndex;

    #[test]
    fn regex_oracle_matches() {
        let case = EvalCase {
            id: "c1".into(),
            task: "t".into(),
            oracle: EvalOracle::Regex {
                pattern: r"fn \w+".into(),
            },
            metadata: BTreeMap::new(),
        };
        assert!(run_eval(&case, "pub fn helper() {}").passed);
        assert!(!run_eval(&case, "no functions here").passed);
    }

    #[test]
    fn golden_diff_oracle_compares_blake3() {
        let produced = "+added line\n";
        let case = EvalCase {
            id: "c2".into(),
            task: "t".into(),
            oracle: EvalOracle::GoldenDiff {
                diff_hash: Hash32::of(produced).to_hex(),
            },
            metadata: BTreeMap::new(),
        };
        assert!(run_eval(&case, produced).passed);
        assert!(!run_eval(&case, "different").passed);
    }

    #[test]
    fn command_oracle_runs_real_process() {
        // `true` exits 0; `false` exits 1 — real subprocess execution.
        let ok = EvalCase {
            id: "ok".into(),
            task: "t".into(),
            oracle: EvalOracle::Command {
                argv: vec!["true".into()],
                cwd: None,
                expected_exit: 0,
            },
            metadata: BTreeMap::new(),
        };
        assert!(run_eval(&ok, "").passed);

        let bad = EvalCase {
            id: "bad".into(),
            task: "t".into(),
            oracle: EvalOracle::Command {
                argv: vec!["false".into()],
                cwd: None,
                expected_exit: 0,
            },
            metadata: BTreeMap::new(),
        };
        assert!(!run_eval(&bad, "").passed);
    }

    #[tokio::test]
    async fn miner_flags_untested_function() {
        let index = InMemoryCodeIndex::default();
        // `helper` is defined in src/ and referenced from a test file → linked.
        index.add_text_file(
            "src/lib.rs",
            "pub fn helper() {}\npub fn lonely() {}\n",
            Some("h".into()),
        );
        index.add_text_file(
            "tests/helper_test.rs",
            "fn check() { helper(); }\n",
            Some("h2".into()),
        );

        let miner = EvalMiner::with_defaults();
        let candidates = miner
            .mine(&index, &["helper".into(), "lonely".into()])
            .await
            .unwrap();

        // `helper` has test linkage → no candidate; `lonely` does not → candidate.
        let names: Vec<&str> = candidates
            .iter()
            .map(|c| c.task_description.as_str())
            .collect();
        assert!(names.iter().any(|d| d.contains("lonely")));
        assert!(!names.iter().any(|d| d.contains("helper")));
    }
}
