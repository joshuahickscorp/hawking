//! Personalization and self-improvement backend (HIDE bible chapter 11).
//!
//! Real backend logic for the ch.11 bleeding-edge capabilities, staged after the
//! shell. What is REAL here (vs. a clean seam):
//!
//!   * **records** — the §11.1.1 capture record with blake3 `[u8;32]` hashes,
//!     microsecond timestamps, and constructors for all four outcomes.
//!   * **store / curate** — scrub-on-write secret redaction (real
//!     [`hide_security::Redactor`]), the `dataset/vNNN` layout, and the full
//!     §11.1.2 curation pipeline (p95×3 latency outliers + recency weighting).
//!   * **eval** — oracles that actually execute (a `Command` oracle spawns a
//!     real process; `Regex`/`GoldenDiff` evaluate real output) + an
//!     [`eval::EvalMiner`] that mines functions-without-tests from the code index.
//!   * **rlef** — reward **derived** from execution outcomes (not supplied),
//!     GRPO group-relative advantage, and a daemon + PPL-gate seam.
//!   * **retrieval** — the [`retrieval::MetaRouter`] trait + a real ε-greedy /
//!     online-SGD router over the code index.
//!   * **kv_handoff** — the §11.5 `KvShareGroup` protocol with a clean seam to
//!     the in-tree `copy_kv_prefix_to_slot`.
//!
//! Seams (post-shell): the actual LoRA gradient step (Hawking Condense), the PPL
//! forward pass, and the runtime KV block copy are trait seams, not faked.

#[rustfmt::skip]
pub mod curate {
    //! The nightly curation pass (bible §11.1.2).
    //!
    //! Turns raw, noisy [`PersonalizationRecord`]s into a clean, versioned SFT +
    //! DPO dataset. Implements all six §11.1.2 rules:
    //!
    //!   1. Keep `Accepted` as positive SFT examples.
    //!   2. Drop `Modified` whose rewrite ratio exceeds `max_rewrite_ratio`
    //!      (the model's proposal was mostly noise).
    //!   3. Pair `Accepted` vs `Rejected` on the same `prompt_hash` → DPO pair.
    //!   4. **Drop latency outliers (p95 × 3)** — timeout artifacts. Computed from
    //!      the actual record population, not a fixed threshold.
    //!   5. Cap at `max_records`, **recency-weighted** (newest first).
    //!   6. Deduplicate on `diff_accepted` content hash.

    use crate::records::{Hash32, Outcome, PersonalizationRecord};
    use crate::store::PersonalLayout;
    use hide_core::Result;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeSet;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct CurationPolicy {
        /// Rule 5 cap (recency-weighted).
        pub max_records: usize,
        /// Rule 2 rewrite-ratio gate.
        pub max_rewrite_ratio: f32,
        /// Rule 4 multiplier on the p95 latency. `3.0` per the bible. Set `None` to
        /// disable the outlier rule entirely.
        pub latency_outlier_p95_mult: Option<f32>,
        /// Fraction of positives withheld for the accept-rate gate (§11.1.4).
        pub held_out_frac: f32,
    }

    impl Default for CurationPolicy {
        fn default() -> Self {
            Self {
                max_records: 10_000,
                max_rewrite_ratio: 0.8,
                latency_outlier_p95_mult: Some(3.0),
                held_out_frac: 0.1,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct CuratedDataset {
        pub sft: Vec<SftExample>,
        pub preferences: Vec<PreferenceExample>,
        pub held_out: Vec<SftExample>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct SftExample {
        pub prompt_hash: Hash32,
        pub context_fingerprint: Hash32,
        pub target_diff: String,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PreferenceExample {
        pub prompt_hash: Hash32,
        pub chosen_diff: String,
        pub rejected_diff: String,
    }

    /// Compute the p95 of a latency population (nearest-rank). Returns `None` for an
    /// empty input.
    fn p95_latency(records: &[PersonalizationRecord]) -> Option<u32> {
        if records.is_empty() {
            return None;
        }
        let mut lat: Vec<u32> = records.iter().map(|r| r.latency_ms).collect();
        lat.sort_unstable();
        // nearest-rank p95: index = ceil(0.95 * n) - 1
        let rank = ((0.95 * lat.len() as f64).ceil() as usize).max(1) - 1;
        Some(lat[rank.min(lat.len() - 1)])
    }

    pub fn curate(records: &[PersonalizationRecord], policy: &CurationPolicy) -> CuratedDataset {
        // ── Rule 4: latency outlier cutoff, computed from the population ──────────
        let latency_cutoff: Option<u32> = policy
            .latency_outlier_p95_mult
            .and_then(|mult| p95_latency(records).map(|p95| ((p95 as f32) * mult).round() as u32));

        // ── Rule 5: recency-weighting. Process newest-first so the cap keeps the
        //    most recent examples (the user's current style), and apply it across
        //    both positives and the rejected pool. ────────────────────────────────
        let mut ordered: Vec<&PersonalizationRecord> = records.iter().collect();
        ordered.sort_by_key(|event| std::cmp::Reverse(event.observed_at_us));

        let mut seen = BTreeSet::new();
        let mut sft = Vec::new();
        let mut rejected_by_prompt = Vec::<&PersonalizationRecord>::new();

        for record in ordered {
            // Rule 4: drop timeout artifacts.
            if let Some(cutoff) = latency_cutoff {
                if record.latency_ms > cutoff {
                    continue;
                }
            }
            match &record.outcome {
                Outcome::Accepted => push_sft(record, policy, &mut seen, &mut sft),
                Outcome::Modified { edit_distance_chars } => {
                    // Rule 2: drop if the user rewrote too much.
                    let proposed_len = record.diff_proposed.chars().count().max(1) as f32;
                    let ratio = *edit_distance_chars as f32 / proposed_len;
                    if ratio <= policy.max_rewrite_ratio {
                        push_sft(record, policy, &mut seen, &mut sft);
                    }
                }
                Outcome::Rejected => rejected_by_prompt.push(record),
                Outcome::Abandoned => {}
            }
        }

        // ── Rule 3: DPO pairs on matching prompt_hash ────────────────────────────
        let mut preferences = Vec::new();
        for accepted in &sft {
            if let Some(rejected) =
                rejected_by_prompt.iter().find(|r| r.prompt_hash == accepted.prompt_hash && !r.diff_proposed.is_empty())
            {
                preferences.push(PreferenceExample {
                    prompt_hash: accepted.prompt_hash,
                    chosen_diff: accepted.target_diff.clone(),
                    rejected_diff: rejected.diff_proposed.clone(),
                });
            }
        }

        // ── Held-out split for the accept-rate gate. Since `sft` is newest-first,
        //    take the held-out slice from the *tail* (older examples) so the gate
        //    measures generalization, not memorized-recent. ───────────────────────
        let held_n = ((sft.len() as f32) * policy.held_out_frac).round() as usize;
        let held_n = held_n.min(sft.len());
        let split = sft.len() - held_n;
        let held_out = sft[split..].to_vec();
        let sft = sft[..split].to_vec();

        CuratedDataset { sft, preferences, held_out }
    }

    /// Rules 1/2 body + rule 6 dedup + rule 5 cap.
    fn push_sft(
        record: &PersonalizationRecord,
        policy: &CurationPolicy,
        seen: &mut BTreeSet<String>,
        sft: &mut Vec<SftExample>,
    ) {
        if record.diff_accepted.is_empty() || sft.len() >= policy.max_records {
            return;
        }
        // Rule 6: dedup on accepted-diff content.
        let key = Hash32::of(&record.diff_accepted).to_hex();
        if !seen.insert(key) {
            return;
        }
        sft.push(SftExample {
            prompt_hash: record.prompt_hash,
            context_fingerprint: record.context_fingerprint,
            target_diff: record.diff_accepted.clone(),
        });
    }

    /// Write a curated dataset to `dataset/vNNN/{train,pref,held_out}.jsonl`
    /// (§11.1.2 layout) and return the version that was written.
    pub fn write_dataset(layout: &PersonalLayout, dataset: &CuratedDataset) -> Result<u32> {
        let version = layout.next_dataset_version()?;
        let dir = layout.dataset_version_dir(version);
        std::fs::create_dir_all(&dir)?;
        write_jsonl(&dir.join("train.jsonl"), &dataset.sft)?;
        write_jsonl(&dir.join("pref.jsonl"), &dataset.preferences)?;
        write_jsonl(&dir.join("held_out.jsonl"), &dataset.held_out)?;
        Ok(version)
    }

    fn write_jsonl<T: Serialize>(path: &std::path::Path, rows: &[T]) -> Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        for row in rows {
            serde_json::to_writer(&mut file, row)?;
            file.write_all(b"\n")?;
        }
        file.sync_data()?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::records::TaskClass;

        fn rec_at(prompt: &str, diff: &str, latency: u32, age_us: u64) -> PersonalizationRecord {
            let mut r = PersonalizationRecord::accepted(TaskClass::EditCode, prompt, diff);
            r.latency_ms = latency;
            r.observed_at_us = age_us;
            r
        }

        #[test]
        fn curation_keeps_accepted_diffs() {
            let records = vec![rec_at("p1", "+hello", 10, 1), rec_at("p2", "+world", 10, 2)];
            let dataset = curate(&records, &CurationPolicy::default());
            // 2 positives, 10% held-out rounds to 0 → both in train.
            assert_eq!(dataset.sft.len(), 2);
            assert_eq!(dataset.held_out.len(), 0);
        }

        #[test]
        fn latency_p95x3_outlier_dropped() {
            // 20 fast records + 1 timeout. p95 of the fast pop is ~10; ×3 = 30; the
            // 5000ms record is dropped.
            let mut records: Vec<_> =
                (0..20).map(|i| rec_at(&format!("p{i}"), &format!("+d{i}"), 10, i as u64)).collect();
            records.push(rec_at("timeout", "+slow", 5000, 99));
            let dataset = curate(&records, &CurationPolicy::default());
            assert!(
                dataset.sft.iter().all(|e| e.target_diff != "+slow"),
                "timeout artifact must be dropped by the p95x3 rule"
            );
        }

        #[test]
        fn recency_cap_keeps_newest() {
            let policy = CurationPolicy { max_records: 1, held_out_frac: 0.0, ..Default::default() };
            let records = vec![rec_at("old", "+old", 10, 1), rec_at("new", "+new", 10, 100)];
            let dataset = curate(&records, &policy);
            assert_eq!(dataset.sft.len(), 1);
            assert_eq!(dataset.sft[0].target_diff, "+new");
        }

        #[test]
        fn dpo_pairs_on_matching_prompt() {
            let mut accepted = PersonalizationRecord::accepted(TaskClass::EditCode, "same", "+good");
            accepted.observed_at_us = 2;
            let rejected = PersonalizationRecord::rejected(TaskClass::EditCode, "same", "+bad", None);
            let dataset = curate(&[accepted, rejected], &CurationPolicy::default());
            assert_eq!(dataset.preferences.len(), 1);
            assert_eq!(dataset.preferences[0].chosen_diff, "+good");
            assert_eq!(dataset.preferences[0].rejected_diff, "+bad");
        }

        #[test]
        fn write_dataset_creates_versioned_layout() {
            let dir = tempfile::tempdir().unwrap();
            let layout = PersonalLayout::new(dir.path());
            layout.ensure().unwrap();
            let dataset = curate(&[rec_at("p", "+d", 10, 1)], &CurationPolicy::default());
            let v = write_dataset(&layout, &dataset).unwrap();
            assert_eq!(v, 1);
            assert!(layout.dataset_version_dir(1).join("train.jsonl").exists());
            assert!(layout.dataset_version_dir(1).join("pref.jsonl").exists());
            assert!(layout.dataset_version_dir(1).join("held_out.jsonl").exists());
        }
    }
}
#[rustfmt::skip]
pub mod eval {
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
    use hawking_index::{CodeIndex, Occurrence, SearchQuery};
    use hide_core::ids::now_micros;
    use hide_core::Result;
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
            Self { case_id: case_id.to_string(), passed: true, score: 1.0, detail: detail.into() }
        }
        fn fail(case_id: &str, detail: impl Into<String>) -> Self {
            Self { case_id: case_id.to_string(), passed: false, score: 0.0, detail: detail.into() }
        }
    }

    /// Run a single case's oracle. `produced` is the agent's output (the produced
    /// diff or text) — used by the `Regex` and `GoldenDiff` oracles; the `Command`
    /// oracle ignores it and runs the process instead.
    pub fn run_eval(case: &EvalCase, produced: &str) -> EvalResult {
        match &case.oracle {
            EvalOracle::Command { argv, cwd, expected_exit } => {
                run_command_oracle(&case.id, argv, cwd.as_deref(), *expected_exit)
            }
            EvalOracle::GoldenDiff { diff_hash } => {
                let got = Hash32::of(produced).to_hex();
                if &got == diff_hash {
                    EvalResult::pass(&case.id, "diff matches golden hash")
                } else {
                    EvalResult::fail(&case.id, format!("golden mismatch: got {got}"))
                }
            }
            EvalOracle::Regex { pattern } => match Regex::new(pattern) {
                Ok(re) if re.is_match(produced) => EvalResult::pass(&case.id, format!("matched /{pattern}/")),
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
                    EvalResult::fail(case_id, format!("exit {code} != {expected_exit}; stderr tail: {tail}"))
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
        /// 0.0–1.0. The base "function has no test linkage" heuristic scores 0.90
        /// (below the 0.95 auto-add gate → surfaces for review); the stronger
        /// "untested AND referenced from non-test source" heuristic scores 0.97
        /// (clears the gate → auto-added).
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
    ///
    /// # Implemented heuristics
    ///
    ///   * **no test linkage** (0.90) — the function is defined but no test path
    ///     references it.
    ///   * **untested AND load-bearing** (0.97) — additionally referenced from
    ///     non-test source, so it is product code with no test (auto-add tier).
    ///
    /// # POST-SHELL MOONSHOT SEAMS (bible §11.3.2 — NOT implemented)
    ///
    /// The bible's full `EvalMiner` also proposes richer, *learned* candidate
    /// sources that depend on infrastructure that does not exist until after the
    /// shell ships, and are deliberately **not** implemented here so this scaffold
    /// does not overclaim:
    ///
    ///   * **regression-from-failed-rollout** — mine a failing autonomous rollout
    ///     (RLEF) into a permanent eval. Requires the §11.1 capture pipeline running
    ///     in production with real rollout transcripts.
    ///   * **coverage-gap from telemetry** — diff executed code paths (runtime
    ///     coverage) against tested paths to find under-tested hot code. Requires
    ///     the runtime coverage daemon.
    ///   * **flaky-test detection** — repeated-run variance mining. Requires a
    ///     historical eval-result store with many runs per case.
    ///   * **LLM-proposed semantic evals** (DSPy/ADAS-style, see [`crate::prompts`])
    ///     — generate eval *tasks* from a description of recent changes. Requires the
    ///     self-improving prompt-optimization loop, itself a post-shell moonshot.
    ///
    /// These remain documented trait/heuristic seams; the only thing wired today is
    /// the static "no test linkage" path above.
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
        pub async fn mine<I: CodeIndex>(&self, index: &I, candidate_fns: &[String]) -> Result<Vec<EvalTaskCandidate>> {
            let mut out = Vec::new();
            let mut auto_added = 0usize;

            for name in candidate_fns {
                if self.config.name_blocklist.iter().any(|b| name.contains(b)) {
                    continue;
                }
                if self.has_test_linkage(index, name).await? {
                    continue;
                }

                // Locate the function's defining file (best-effort) for source_files.
                let defs = index.definition(name).await?;
                let source_files: Vec<PathBuf> = defs.iter().map(|o| PathBuf::from(&o.file)).collect();

                // Confidence heuristics (§11.3.2).
                //
                //   * Base "function has no test linkage" — 0.90. On its own this is
                //     below the default 0.95 auto-add gate, so it surfaces for human
                //     review (an untested-AND-unreferenced function may be dead code,
                //     where the right action is "delete", not "write a test").
                //
                //   * **Untested AND load-bearing** — 0.97. When the function has no
                //     test linkage BUT *is* referenced from non-test source, it is
                //     code that is actually used in the product yet exercised by no
                //     test. That is the highest-value, lowest-ambiguity "write a
                //     test" target, so it clears the auto-add gate.
                let source_refs = self.nontest_reference_count(index, name).await?;
                let confidence = if source_refs > 0 { 0.97 } else { 0.90 };
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
                    task_description: format!("Write a test for `{name}` — it currently has no test linkage."),
                    // The oracle is "the new test passes": a `cargo test` filtered to
                    // the function name. The exact filter is the function name so the
                    // harness runs only the relevant test.
                    oracle: EvalOracle::Command {
                        argv: vec!["cargo".into(), "test".into(), name.clone()],
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

        /// How many times is `name` referenced from **non-test** source? Uses the
        /// index's structural reference occurrences (tree-sitter extracted), filtered
        /// to drop the definition's own file and any test path. A positive count
        /// means the function is load-bearing — actually used by product code.
        async fn nontest_reference_count<I: CodeIndex>(&self, index: &I, name: &str) -> Result<usize> {
            let refs: Vec<Occurrence> = index.references(name).await?;
            let count = refs.iter().filter(|o| !is_test_path(std::path::Path::new(&o.file))).count();
            Ok(count)
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
                oracle: EvalOracle::Regex { pattern: r"fn \w+".into() },
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
                oracle: EvalOracle::GoldenDiff { diff_hash: Hash32::of(produced).to_hex() },
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
                oracle: EvalOracle::Command { argv: vec!["true".into()], cwd: None, expected_exit: 0 },
                metadata: BTreeMap::new(),
            };
            assert!(run_eval(&ok, "").passed);

            let bad = EvalCase {
                id: "bad".into(),
                task: "t".into(),
                oracle: EvalOracle::Command { argv: vec!["false".into()], cwd: None, expected_exit: 0 },
                metadata: BTreeMap::new(),
            };
            assert!(!run_eval(&bad, "").passed);
        }

        #[tokio::test]
        async fn miner_flags_untested_function() {
            let index = InMemoryCodeIndex::default();
            // `helper` is defined in src/ and referenced from a test file → linked.
            index.add_text_file("src/lib.rs", "pub fn helper() {}\npub fn lonely() {}\n", Some("h".into()));
            index.add_text_file("tests/helper_test.rs", "fn check() { helper(); }\n", Some("h2".into()));

            let miner = EvalMiner::with_defaults();
            let candidates = miner.mine(&index, &["helper".into(), "lonely".into()]).await.unwrap();

            // `helper` has test linkage → no candidate; `lonely` does not → candidate.
            let names: Vec<&str> = candidates.iter().map(|c| c.task_description.as_str()).collect();
            assert!(names.iter().any(|d| d.contains("lonely")));
            assert!(!names.iter().any(|d| d.contains("helper")));
        }

        #[tokio::test]
        async fn miner_auto_adds_untested_load_bearing_function() {
            let index = InMemoryCodeIndex::default();
            // `compute` is defined in one source file...
            index.add_text_file("src/math.rs", "pub fn compute(x: i32) -> i32 { x + 1 }\n", Some("m".into()));
            // ...and *used* from another (non-test) source file — load-bearing.
            index.add_text_file(
                "src/app.rs",
                "use crate::math::compute;\npub fn run() -> i32 { compute(41) }\n",
                Some("a".into()),
            );

            let miner = EvalMiner::with_defaults();
            let candidates = miner.mine(&index, &["compute".into()]).await.unwrap();

            let compute = candidates
                .iter()
                .find(|c| c.task_description.contains("compute"))
                .expect("compute should be a candidate (untested)");
            // Referenced from non-test source → 0.97 ≥ 0.95 gate → AutoAdded.
            assert_eq!(compute.confidence, 0.97);
            assert_eq!(compute.status, CandidateStatus::AutoAdded);
        }

        #[tokio::test]
        async fn miner_defers_untested_unreferenced_function() {
            let index = InMemoryCodeIndex::default();
            // Defined but referenced nowhere (possibly dead code) → human review.
            index.add_text_file("src/orphan.rs", "pub fn orphan() -> i32 { 0 }\n", Some("o".into()));

            let miner = EvalMiner::with_defaults();
            let candidates = miner.mine(&index, &["orphan".into()]).await.unwrap();

            let orphan = candidates
                .iter()
                .find(|c| c.task_description.contains("orphan"))
                .expect("orphan should be a candidate");
            assert_eq!(orphan.confidence, 0.90);
            assert_eq!(orphan.status, CandidateStatus::PendingHuman);
        }
    }
}
#[rustfmt::skip]
pub mod kv_handoff {
    //! KV-cache handoff between agents (bible §11.5 + §11.4.2 Approach 2).
    //!
    //! When a fan-out swarm launches, every worker receives the *same* prefix (the
    //! system prompt + repo-map + plan context). Re-prefilling that prefix N times
    //! is pure waste. This module defines the [`KvShareGroup`] protocol: the Planner
    //! checkpoints its prefix KV state under a [`KvKey`], registers a share group,
    //! and every worker's [`GenerateRequest`] carries a [`KvHandle`] seed so the
    //! runtime restores the shared prefix instead of re-prefilling it.
    //!
    //! ## Seam (important)
    //!
    //! This crate **does not invent KV state**. The actual block copy is performed
    //! by the in-tree `copy_kv_prefix_to_slot` primitive in the live runtime
    //! (`hawking-core`'s engine), exposed over `hawking-serve`'s HTTP surface. This
    //! module is the *protocol + lifecycle*: the keys, the fork position, the
    //! member set, the TTL/refcount bookkeeping, and the typed [`KvHandle`] that
    //! rides in the handoff. The [`KvPrefixCopier`] trait is the one-method seam the
    //! runtime fills in; [`copy_for_group`] drives it for every member. No fake KV
    //! map is constructed here.

    use serde::{Deserialize, Serialize};

    /// Identifier of an agent in a swarm (e.g. `"planner:0"`, `"worker:3"`).
    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct AgentId(pub String);

    impl AgentId {
        pub fn new(s: impl Into<String>) -> Self {
            Self(s.into())
        }
    }

    /// Key under which a prefix KV state is stored in the runtime's KV store. Opaque
    /// to this crate; the runtime maps it to its block table. Kept a newtype so it
    /// can't be confused with an arbitrary string handle.
    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct KvKey(pub String);

    impl KvKey {
        pub fn new(s: impl Into<String>) -> Self {
            Self(s.into())
        }
        pub fn as_str(&self) -> &str {
            &self.0
        }
    }

    /// A handle the receiver uses to restore a shared prefix (§11.4.2 / §11.5.2):
    /// the store key plus the token position at which divergence begins. Carried in
    /// the `kv_seed` field of a worker's generate request.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct KvHandle {
        pub store_key: KvKey,
        /// All members share tokens `0..fork_seq`; each diverges from `fork_seq` on.
        pub fork_seq: u64,
    }

    /// The §11.5.2 protocol object: registered with the Governor when a fan-out
    /// swarm launches. The prefix KV state lives once in the store; the N members
    /// share it.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct KvShareGroup {
        /// Key under which the prefix KV state is stored.
        pub prefix_key: KvKey,
        /// Agents that share this prefix.
        pub members: Vec<AgentId>,
        /// Token position at which each member diverges (shared region is `0..`).
        pub fork_seq: u64,
        /// The shared KV state is evicted after this many ms of non-use.
        pub ttl_ms: u64,
    }

    impl KvShareGroup {
        pub fn new(prefix_key: KvKey, fork_seq: u64, members: Vec<AgentId>) -> Self {
            Self { prefix_key, members, fork_seq, ttl_ms: 60_000 }
        }

        /// The seed every member's generate request carries (§11.5.2 step 3): the
        /// same store key + fork position for all of them.
        pub fn handle(&self) -> KvHandle {
            KvHandle { store_key: self.prefix_key.clone(), fork_seq: self.fork_seq }
        }

        /// Reference count = number of members still sharing the prefix. The runtime
        /// evicts the shared blocks only when this reaches zero (§11.5.2 step 4).
        pub fn refcount(&self) -> usize {
            self.members.len()
        }
    }

    /// The KV-seed extension to a generate request (§11.5.2). If `Some`, the runtime
    /// skips prefill of `0..fork_seq` and restores from the store.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct GenerateRequest {
        pub agent: AgentId,
        pub prompt_suffix: String,
        /// When set, restore the shared prefix instead of re-prefilling it.
        pub kv_seed: Option<KvHandle>,
    }

    /// The one-method seam to the in-tree `copy_kv_prefix_to_slot` primitive. The
    /// runtime (hawking-serve / hawking-core) implements this; this crate only
    /// orchestrates *which* copies happen and *when*. Returning a `Result<usize>` of
    /// tokens-restored lets the caller measure the prefill it saved.
    pub trait KvPrefixCopier {
        /// Copy the stored prefix at `handle.store_key` (length `handle.fork_seq`)
        /// into the decode slot for `agent`. Maps directly onto the in-tree
        /// `copy_kv_prefix_to_slot(store_key, fork_seq, slot)`.
        fn copy_kv_prefix_to_slot(&self, agent: &AgentId, handle: &KvHandle) -> std::result::Result<usize, String>;
    }

    /// Outcome of broadcasting a shared prefix to a group's members.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct BroadcastReport {
        /// Members whose prefix was restored, with tokens-restored each.
        pub restored: Vec<(AgentId, usize)>,
        /// Members that failed to restore (and must fall back to full prefill).
        pub failed: Vec<(AgentId, String)>,
    }

    impl BroadcastReport {
        /// Total tokens of prefill skipped across the swarm — the win this protocol
        /// exists to produce.
        pub fn tokens_saved(&self) -> usize {
            self.restored.iter().map(|(_, n)| *n).sum()
        }
    }

    /// Drive the seam for every member of a share group (§11.5.2 step 3). A member
    /// that fails to restore is reported, not panicked on, so the swarm can fall
    /// back to full prefill for that worker.
    pub fn copy_for_group(copier: &dyn KvPrefixCopier, group: &KvShareGroup) -> BroadcastReport {
        let handle = group.handle();
        let mut restored = Vec::new();
        let mut failed = Vec::new();
        for member in &group.members {
            match copier.copy_kv_prefix_to_slot(member, &handle) {
                Ok(n) => restored.push((member.clone(), n)),
                Err(e) => failed.push((member.clone(), e)),
            }
        }
        BroadcastReport { restored, failed }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A test double standing in for the runtime's `copy_kv_prefix_to_slot`.
        /// It records the calls and returns `fork_seq` tokens restored — it does NOT
        /// fabricate KV state, it only models the seam contract.
        struct FakeCopier {
            fail_for: Option<String>,
        }
        impl KvPrefixCopier for FakeCopier {
            fn copy_kv_prefix_to_slot(&self, agent: &AgentId, handle: &KvHandle) -> std::result::Result<usize, String> {
                if self.fail_for.as_deref() == Some(agent.0.as_str()) {
                    Err("slot busy".into())
                } else {
                    Ok(handle.fork_seq as usize)
                }
            }
        }

        #[test]
        fn share_group_handle_is_shared_across_members() {
            let group = KvShareGroup::new(
                KvKey::new("planner-0-fork"),
                2300,
                vec![AgentId::new("worker:0"), AgentId::new("worker:1")],
            );
            assert_eq!(group.refcount(), 2);
            let h = group.handle();
            assert_eq!(h.fork_seq, 2300);
            assert_eq!(h.store_key.as_str(), "planner-0-fork");
        }

        #[test]
        fn broadcast_sums_saved_prefill() {
            let group = KvShareGroup::new(
                KvKey::new("k"),
                2300,
                (0..16).map(|i| AgentId::new(format!("worker:{i}"))).collect(),
            );
            let copier = FakeCopier { fail_for: None };
            let report = copy_for_group(&copier, &group);
            assert_eq!(report.restored.len(), 16);
            assert!(report.failed.is_empty());
            // 16 workers × 2300-token prefix all skipped.
            assert_eq!(report.tokens_saved(), 16 * 2300);
        }

        #[test]
        fn broadcast_reports_failures_without_aborting() {
            let group =
                KvShareGroup::new(KvKey::new("k"), 100, vec![AgentId::new("worker:0"), AgentId::new("worker:1")]);
            let copier = FakeCopier { fail_for: Some("worker:1".into()) };
            let report = copy_for_group(&copier, &group);
            assert_eq!(report.restored.len(), 1);
            assert_eq!(report.failed.len(), 1);
            assert_eq!(report.tokens_saved(), 100);
        }

        #[test]
        fn generate_request_carries_seed() {
            let group = KvShareGroup::new(KvKey::new("k"), 42, vec![AgentId::new("worker:0")]);
            let req = GenerateRequest {
                agent: AgentId::new("worker:0"),
                prompt_suffix: "continue".into(),
                kv_seed: Some(group.handle()),
            };
            let json = serde_json::to_string(&req).unwrap();
            let back: GenerateRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(back.kv_seed.unwrap().fork_seq, 42);
        }
    }
}
#[rustfmt::skip]
pub mod prompts {
    //! Self-improving prompt modules (bible §11.2).
    //!
    //! # POST-SHELL MOONSHOT SEAM — the DSPy / ADAS / GEPA optimization loop is NOT implemented.
    //!
    //! What is REAL in this module is only the **typed data model and the promotion
    //! gate**: a versioned [`PromptModule`] with an immutable [`PromptVersion`]
    //! history, an [`OptimizationMetric`], a `min_eval_n` floor, and
    //! [`PromptModule::promote`] — which accepts a candidate version *only* if it was
    //! evaluated on enough cases (`eval_n >= min_eval_n`) **and** scores strictly
    //! higher than the current best. That gate is enough to store, audit, and
    //! roll back prompt versions safely.
    //!
    //! What is **deliberately deferred** (a documented seam, per the bible's
    //! moonshot tiering) is the actual *optimizer* that produces those candidate
    //! versions:
    //!
    //!   * **DSPy-style** compile/bootstrap of few-shot demonstrations and
    //!     instruction tuning from the eval set;
    //!   * **ADAS** (Automated Design of Agentic Systems) meta-search over module
    //!     graphs;
    //!   * **GEPA** reflective prompt evolution (LLM-in-the-loop mutate + select on
    //!     the [`OptimizationMetric`]).
    //!
    //! Each of those requires the production eval flywheel ([`crate::eval`]) plus a
    //! local-model optimization budget that only exists after the shell ships. The
    //! [`PromotedBy::AutoDspy`] and [`PromotedBy::Adas`] provenance variants exist so
    //! that, when the loop is built, a machine-promoted version is distinguishable
    //! from a [`PromotedBy::Human`] one — but nothing in this crate emits them yet.
    //! No optimizer is wired; do not read the presence of these types as a claim
    //! that the self-improving loop runs.

    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PromptModule {
        pub name: String,
        pub schema_version: u32,
        pub input_schema: Value,
        pub output_schema: Value,
        pub template: String,
        pub history: Vec<PromptVersion>,
        pub metric: OptimizationMetric,
        pub min_eval_n: u16,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PromptVersion {
        pub version: u32,
        pub template: String,
        pub score: f64,
        pub eval_n: u16,
        pub promoted_at_ms: u64,
        pub promoted_by: PromotedBy,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum OptimizationMetric {
        Accuracy,
        AcceptRate,
        LatencyMs,
        OraclePassRate,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum PromotedBy {
        Human,
        AutoDspy { optimizer_run_id: String },
        Adas { run_id: String },
    }

    impl PromptModule {
        pub fn promote(&mut self, version: PromptVersion) -> bool {
            if version.eval_n < self.min_eval_n {
                return false;
            }
            if self.history.last().map_or(true, |current| version.score > current.score) {
                self.template = version.template.clone();
                self.history.push(version);
                true
            } else {
                false
            }
        }
    }
}
#[rustfmt::skip]
pub mod records {
    //! The personalization record (bible §11.1.1).
    //!
    //! Reconciled with the normative schema: `prompt_hash` / `context_fingerprint`
    //! are **blake3 `[u8; 32]`** digests (not opaque strings), `observed_at_us` is
    //! microsecond wall-clock, `tok_s` is non-optional, and there is a constructor
    //! for every one of the four `Outcome` variants (Accepted / Rejected / Modified
    //! / Abandoned) so the capture layer can mint a record for whatever the user
    //! actually did.

    use hide_core::ids::{now_micros, RunId, SessionId};
    use serde::{Deserialize, Serialize};

    /// blake3-256 digest. Serialized as a lowercase hex string so the JSONL log
    /// stays human-inspectable (§11.1.5 "the user can find and inspect every
    /// record") while the in-memory form is the fixed 32-byte array the bible
    /// mandates for `prompt_hash` / `context_fingerprint`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Hash32(pub [u8; 32]);

    impl Hash32 {
        /// blake3 of arbitrary bytes — the canonical way to mint a `prompt_hash`
        /// (over the final system prompt + user message) or a `context_fingerprint`
        /// (over the set of file paths + their sizes).
        pub fn of(bytes: impl AsRef<[u8]>) -> Self {
            Self(*blake3::hash(bytes.as_ref()).as_bytes())
        }

        /// Fold an iterator of `(path, size)` pairs into a single fingerprint in a
        /// stable, order-independent way (the set of files, not their order).
        pub fn fingerprint_files<'a, I>(files: I) -> Self
        where
            I: IntoIterator<Item = (&'a str, u64)>,
        {
            let mut entries: Vec<(String, u64)> = files.into_iter().map(|(p, s)| (p.to_string(), s)).collect();
            entries.sort();
            let mut hasher = blake3::Hasher::new();
            for (path, size) in entries {
                hasher.update(path.as_bytes());
                hasher.update(&size.to_le_bytes());
            }
            Self(*hasher.finalize().as_bytes())
        }

        pub fn to_hex(self) -> String {
            let mut s = String::with_capacity(64);
            for byte in self.0 {
                s.push_str(&format!("{byte:02x}"));
            }
            s
        }

        pub fn from_hex(hex: &str) -> Option<Self> {
            if hex.len() != 64 {
                return None;
            }
            let mut out = [0u8; 32];
            for (i, byte) in out.iter_mut().enumerate() {
                *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
            }
            Some(Self(out))
        }
    }

    impl Serialize for Hash32 {
        fn serialize<S: serde::Serializer>(&self, ser: S) -> std::result::Result<S::Ok, S::Error> {
            ser.serialize_str(&self.to_hex())
        }
    }

    impl<'de> Deserialize<'de> for Hash32 {
        fn deserialize<D: serde::Deserializer<'de>>(de: D) -> std::result::Result<Self, D::Error> {
            let s = String::deserialize(de)?;
            Hash32::from_hex(&s).ok_or_else(|| serde::de::Error::custom("invalid blake3 hex digest"))
        }
    }

    /// One captured agent turn (§11.1.1). Written to the personal records log only
    /// **after** the outcome is known. Never leaves the device.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PersonalizationRecord {
        pub session_id: SessionId,
        pub run_id: Option<RunId>,
        /// Microsecond-precision wall-clock of the observation (was `_ms`).
        pub observed_at_us: u64,
        pub task_type: TaskClass,
        /// blake3 of the final system prompt + user message.
        pub prompt_hash: Hash32,
        /// blake3 of the set of file paths + their sizes at call time.
        pub context_fingerprint: Hash32,
        pub outcome: Outcome,
        pub diff_proposed: String,
        pub diff_accepted: String,
        pub latency_ms: u32,
        /// Decode throughput of the generation that produced `diff_proposed`.
        /// Non-optional (bible §11.1.1).
        pub tok_s: f32,
        pub reject_reason: Option<String>,
        pub model_role: String,
        pub active_adapters: Vec<String>,
    }

    impl PersonalizationRecord {
        /// The shared skeleton; the four public ctors fill `outcome` + the diffs.
        fn base(task_type: TaskClass, prompt: &str, outcome: Outcome) -> Self {
            Self {
                session_id: SessionId::new(),
                run_id: None,
                observed_at_us: now_micros(),
                task_type,
                prompt_hash: Hash32::of(prompt),
                context_fingerprint: Hash32::of(""),
                outcome,
                diff_proposed: String::new(),
                diff_accepted: String::new(),
                latency_ms: 0,
                tok_s: 0.0,
                reject_reason: None,
                model_role: "hero".to_string(),
                active_adapters: Vec::new(),
            }
        }

        /// User accepted the diff verbatim. `diff_accepted == diff_proposed`.
        pub fn accepted(task_type: TaskClass, prompt: &str, diff: impl Into<String>) -> Self {
            let diff = diff.into();
            let mut rec = Self::base(task_type, prompt, Outcome::Accepted);
            rec.diff_proposed = diff.clone();
            rec.diff_accepted = diff;
            rec
        }

        /// User accepted a manually-edited version. `edit_distance_chars` measures
        /// how much they rewrote (drives curate's rewrite-ratio gate, §11.1.2 rule 2).
        pub fn modified(
            task_type: TaskClass,
            prompt: &str,
            proposed: impl Into<String>,
            accepted: impl Into<String>,
            edit_distance_chars: u32,
        ) -> Self {
            let mut rec = Self::base(task_type, prompt, Outcome::Modified { edit_distance_chars });
            rec.diff_proposed = proposed.into();
            rec.diff_accepted = accepted.into();
            rec
        }

        /// User rejected the suggestion. `diff_accepted` stays empty; the proposed
        /// diff is retained as the negative half of a future DPO pair (§11.1.2 rule 3).
        pub fn rejected(
            task_type: TaskClass,
            prompt: &str,
            proposed: impl Into<String>,
            reason: Option<String>,
        ) -> Self {
            let mut rec = Self::base(task_type, prompt, Outcome::Rejected);
            rec.diff_proposed = proposed.into();
            rec.reject_reason = reason;
            rec
        }

        /// Session ended before an explicit accept/reject — an implicit partial
        /// signal (§11.1.1). Curation drops these.
        pub fn abandoned(task_type: TaskClass, prompt: &str, proposed: impl Into<String>) -> Self {
            let mut rec = Self::base(task_type, prompt, Outcome::Abandoned);
            rec.diff_proposed = proposed.into();
            rec
        }

        /// Builder helper for the capture layer to attach the real context manifest.
        pub fn with_context_fingerprint(mut self, fp: Hash32) -> Self {
            self.context_fingerprint = fp;
            self
        }

        /// Builder helper to record decode throughput + latency for curation rule 4.
        pub fn with_perf(mut self, tok_s: f32, latency_ms: u32) -> Self {
            self.tok_s = tok_s;
            self.latency_ms = latency_ms;
            self
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum TaskClass {
        EditCode,
        WriteTest,
        Refactor,
        ExplainCode,
        CommitMsg,
        Diagnose,
        Research,
        Other,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum Outcome {
        /// User accepted the diff without modification.
        Accepted,
        /// User accepted a manually-edited version of the diff.
        Modified { edit_distance_chars: u32 },
        /// User rejected (undo, explicit dismiss, or replaced entirely).
        Rejected,
        /// Session ended before explicit accept/reject.
        Abandoned,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn hash32_hex_roundtrip() {
            let h = Hash32::of("hello");
            let hex = h.to_hex();
            assert_eq!(hex.len(), 64);
            assert_eq!(Hash32::from_hex(&hex), Some(h));
            assert_eq!(Hash32::from_hex("nothex"), None);
        }

        #[test]
        fn fingerprint_is_order_independent() {
            let a = Hash32::fingerprint_files([("a.rs", 10), ("b.rs", 20)]);
            let b = Hash32::fingerprint_files([("b.rs", 20), ("a.rs", 10)]);
            assert_eq!(a, b);
            let c = Hash32::fingerprint_files([("a.rs", 11), ("b.rs", 20)]);
            assert_ne!(a, c);
        }

        #[test]
        fn all_four_outcome_ctors_exist() {
            let p = "system+user";
            assert_eq!(PersonalizationRecord::accepted(TaskClass::EditCode, p, "d").outcome, Outcome::Accepted);
            assert_eq!(
                PersonalizationRecord::modified(TaskClass::Refactor, p, "a", "b", 3).outcome,
                Outcome::Modified { edit_distance_chars: 3 }
            );
            assert_eq!(
                PersonalizationRecord::rejected(TaskClass::WriteTest, p, "d", Some("nope".into())).outcome,
                Outcome::Rejected
            );
            assert_eq!(PersonalizationRecord::abandoned(TaskClass::Diagnose, p, "d").outcome, Outcome::Abandoned);
            // prompt_hash is the same for the same prompt (DPO pairing depends on it).
            let r1 = PersonalizationRecord::accepted(TaskClass::EditCode, p, "x");
            let r2 = PersonalizationRecord::rejected(TaskClass::EditCode, p, "y", None);
            assert_eq!(r1.prompt_hash, r2.prompt_hash);
        }

        #[test]
        fn record_serde_roundtrips_with_hex_hashes() {
            let rec = PersonalizationRecord::accepted(TaskClass::EditCode, "p", "diff").with_perf(42.0, 100);
            let json = serde_json::to_string(&rec).unwrap();
            assert!(json.contains(&rec.prompt_hash.to_hex()));
            let back: PersonalizationRecord = serde_json::from_str(&json).unwrap();
            assert_eq!(back, rec);
        }
    }
}
#[rustfmt::skip]
pub mod retrieval {
    //! Learned retrieval — meta-learning over the codebase index (bible §11.6).
    //!
    //! Defines the [`MetaRouter`] trait (route a query to a retrieval strategy, then
    //! update online from the outcome) and a real implementation,
    //! [`EpsilonGreedyRouter`], that does **actual online learning**: an
    //! ε-greedy policy over per-`(query-kind, strategy)` value estimates updated by
    //! an incremental SGD step (a running mean of "did this strategy's span get
    //! used"). No retraining pipeline, no batching — one O(1) update per task, as
    //! §11.6.3 specifies.
    //!
    //! Wires the previously-unused `hawking-index`: [`route_and_search`] takes a
    //! routed [`RetrievalStrategy`] and drives the real `CodeIndex::search` with the
    //! matching leg toggles.

    use hawking_index::{CodeIndex, SearchQuery, SearchResult};
    use hide_core::Result;
    use rand::Rng;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    /// The retrieval strategies the index exposes (§11.6.2). These map onto the
    /// `CodeIndex::search` leg toggles in [`route_and_search`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum RetrievalStrategy {
        Bm25,
        EmbeddingCosine,
        CallGraphProximity,
        TestFileLinkage,
        Recency,
        Symbol,
    }

    impl RetrievalStrategy {
        pub const ALL: [RetrievalStrategy; 6] = [
            RetrievalStrategy::Bm25,
            RetrievalStrategy::EmbeddingCosine,
            RetrievalStrategy::CallGraphProximity,
            RetrievalStrategy::TestFileLinkage,
            RetrievalStrategy::Recency,
            RetrievalStrategy::Symbol,
        ];
    }

    /// The kind of query, used as the context key for the learned policy (§11.6.2).
    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct QueryType {
        pub kind: String,
        pub detected_language: Option<String>,
    }

    impl QueryType {
        pub fn new(kind: impl Into<String>) -> Self {
            Self { kind: kind.into(), detected_language: None }
        }
    }

    /// The supervision signal (§11.6.2): which strategy was chosen and whether its
    /// span actually appeared in the final accepted output.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct RetrievalOutcomeRecord {
        pub query_type: QueryType,
        pub strategy: RetrievalStrategy,
        /// Did a span this strategy returned appear in the final diff/plan?
        pub used_in_output: bool,
    }

    /// Per-strategy learned weights (kept for the cold-start prior + introspection).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct LearnedRetrievalWeights {
        pub bm25: f32,
        pub embedding_cosine: f32,
        pub call_graph: f32,
        pub test_linkage: f32,
        pub recency: f32,
        pub symbol: f32,
    }

    impl Default for LearnedRetrievalWeights {
        fn default() -> Self {
            // §11.6.3 cold-start priors: symbol/exact wins, then call-graph, then
            // embedding, with recency a weak tiebreak.
            Self { bm25: 1.0, embedding_cosine: 0.5, call_graph: 0.75, test_linkage: 0.5, recency: 0.25, symbol: 1.5 }
        }
    }

    impl LearnedRetrievalWeights {
        fn get(&self, s: RetrievalStrategy) -> f32 {
            match s {
                RetrievalStrategy::Bm25 => self.bm25,
                RetrievalStrategy::EmbeddingCosine => self.embedding_cosine,
                RetrievalStrategy::CallGraphProximity => self.call_graph,
                RetrievalStrategy::TestFileLinkage => self.test_linkage,
                RetrievalStrategy::Recency => self.recency,
                RetrievalStrategy::Symbol => self.symbol,
            }
        }
    }

    /// The §11.6.3 router contract: route a query, then learn from the outcome.
    pub trait MetaRouter: Send + Sync {
        /// Return the strategy to try first for this query. If learned confidence is
        /// below `confidence_min`, fall back to the static prior ordering.
        fn route(&self, query: &str, qtype: &QueryType, confidence_min: f32) -> RetrievalStrategy;

        /// One online SGD step from a completed task's outcome (§11.6.3).
        fn update(&mut self, record: &RetrievalOutcomeRecord);
    }

    /// Running value estimate for a `(query-kind, strategy)` cell: the SGD-updated
    /// mean usefulness and the number of observations (the confidence proxy).
    #[derive(Debug, Clone, Copy, Default)]
    struct ValueCell {
        /// EMA of `used_in_output` ∈ [0, 1].
        value: f32,
        /// Number of updates folded in (drives confidence).
        count: u32,
    }

    /// ε-greedy router with per-cell incremental SGD. Real online learning that
    /// improves monotonically with each completed task and never blocks.
    pub struct EpsilonGreedyRouter {
        /// Exploration rate (§11.6.3: 0.1 early, decaying to 0.02).
        epsilon: f64,
        /// SGD learning rate for the incremental mean update.
        lr: f32,
        /// Cold-start priors (used until a cell has observations).
        prior: LearnedRetrievalWeights,
        /// `(query_kind, strategy) → value`. Keyed by kind only (the bible buckets
        /// by query type + codebase fingerprint; kind alone is the live signal).
        cells: HashMap<(String, RetrievalStrategy), ValueCell>,
        rng: rand::rngs::StdRng,
    }

    impl EpsilonGreedyRouter {
        pub fn new(epsilon: f64, lr: f32) -> Self {
            use rand::SeedableRng;
            Self {
                epsilon,
                lr,
                prior: LearnedRetrievalWeights::default(),
                cells: HashMap::new(),
                // Deterministic seed so routing is reproducible in tests/replay; the
                // production caller can reseed from entropy via `with_seed`.
                rng: rand::rngs::StdRng::seed_from_u64(0x5217_3EF0),
            }
        }

        pub fn with_seed(epsilon: f64, lr: f32, seed: u64) -> Self {
            use rand::SeedableRng;
            Self {
                epsilon,
                lr,
                prior: LearnedRetrievalWeights::default(),
                cells: HashMap::new(),
                rng: rand::rngs::StdRng::seed_from_u64(seed),
            }
        }

        /// The learned value for a cell, or the prior if unobserved.
        fn value_of(&self, kind: &str, s: RetrievalStrategy) -> f32 {
            self.cells.get(&(kind.to_string(), s)).map(|c| c.value).unwrap_or_else(|| {
                // normalize the prior weight into a [0,1]-ish usefulness proxy.
                let w = self.prior.get(s);
                (w / 2.0).clamp(0.0, 1.0)
            })
        }

        /// Confidence in the best cell = observation count saturating toward 1.
        fn confidence(&self, kind: &str, s: RetrievalStrategy) -> f32 {
            let n = self.cells.get(&(kind.to_string(), s)).map(|c| c.count).unwrap_or(0) as f32;
            // 0 obs → 0 confidence; ~20 obs → ~0.9.
            n / (n + 2.0)
        }

        fn best_strategy(&self, kind: &str) -> RetrievalStrategy {
            RetrievalStrategy::ALL
                .into_iter()
                .max_by(|a, b| {
                    self.value_of(kind, *a).partial_cmp(&self.value_of(kind, *b)).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(RetrievalStrategy::Bm25)
        }
    }

    impl MetaRouter for EpsilonGreedyRouter {
        fn route(&self, _query: &str, qtype: &QueryType, confidence_min: f32) -> RetrievalStrategy {
            // Exploration is handled by route_explore; the immutable `route` is the
            // greedy/confident path (the trait method takes &self). Use the
            // confidence gate: if the best cell isn't confident enough, fall back to
            // the static prior ordering (highest prior weight).
            let best = self.best_strategy(&qtype.kind);
            if self.confidence(&qtype.kind, best) >= confidence_min {
                best
            } else {
                // static prior ordering: pick the strategy with the highest prior.
                RetrievalStrategy::ALL
                    .into_iter()
                    .max_by(|a, b| {
                        self.prior.get(*a).partial_cmp(&self.prior.get(*b)).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(RetrievalStrategy::Symbol)
            }
        }

        fn update(&mut self, record: &RetrievalOutcomeRecord) {
            let key = (record.query_type.kind.clone(), record.strategy);
            let cell = self.cells.entry(key).or_default();
            let target = if record.used_in_output { 1.0 } else { 0.0 };
            // Incremental SGD step toward the target (running EMA).
            cell.value += self.lr * (target - cell.value);
            cell.count += 1;
        }
    }

    impl EpsilonGreedyRouter {
        /// The exploratory route: with probability ε, return a random strategy to
        /// keep the signal fresh (§11.6.3). Mutable because it advances the RNG.
        pub fn route_explore(&mut self, query: &str, qtype: &QueryType, confidence_min: f32) -> RetrievalStrategy {
            if self.rng.gen::<f64>() < self.epsilon {
                let i = self.rng.gen_range(0..RetrievalStrategy::ALL.len());
                RetrievalStrategy::ALL[i]
            } else {
                self.route(query, qtype, confidence_min)
            }
        }
    }

    /// Drive the real `CodeIndex` with a routed strategy. Maps the abstract strategy
    /// onto the index's concrete search legs (the wiring of `hawking-index`).
    pub async fn route_and_search<I: CodeIndex>(
        index: &I,
        strategy: RetrievalStrategy,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let (include_symbols, include_lexical, include_semantic) = match strategy {
            RetrievalStrategy::Symbol | RetrievalStrategy::CallGraphProximity => (true, false, false),
            RetrievalStrategy::Bm25 => (false, true, false),
            RetrievalStrategy::EmbeddingCosine => (false, false, true),
            // Test-linkage / recency don't have a dedicated index leg yet; fall back
            // to symbol+lexical so the query still returns useful spans (a real,
            // documented degrade rather than a fake result).
            RetrievalStrategy::TestFileLinkage | RetrievalStrategy::Recency => (true, true, false),
        };
        index
            .search(SearchQuery { text: query.to_string(), limit, include_symbols, include_lexical, include_semantic })
            .await
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use hawking_index::InMemoryCodeIndex;

        #[test]
        fn online_update_shifts_policy() {
            let mut router = EpsilonGreedyRouter::new(0.0, 0.5); // ε=0 → pure greedy
            let qt = QueryType::new("find_callers");

            // Teach it that CallGraphProximity is useful for find_callers, Bm25 not.
            for _ in 0..10 {
                router.update(&RetrievalOutcomeRecord {
                    query_type: qt.clone(),
                    strategy: RetrievalStrategy::CallGraphProximity,
                    used_in_output: true,
                });
                router.update(&RetrievalOutcomeRecord {
                    query_type: qt.clone(),
                    strategy: RetrievalStrategy::Bm25,
                    used_in_output: false,
                });
            }
            // With enough observations, route picks the learned winner.
            let chosen = router.route("who calls foo", &qt, 0.5);
            assert_eq!(chosen, RetrievalStrategy::CallGraphProximity);
        }

        #[test]
        fn low_confidence_falls_back_to_prior() {
            let router = EpsilonGreedyRouter::new(0.0, 0.5);
            let qt = QueryType::new("novel_kind");
            // No observations → confidence 0 < 0.5 → fall back to highest prior
            // (Symbol, weight 1.5).
            assert_eq!(router.route("x", &qt, 0.5), RetrievalStrategy::Symbol);
        }

        #[test]
        fn epsilon_explores() {
            // ε=1 → always explore (random); just assert it returns a valid variant
            // and advances without panicking.
            let mut router = EpsilonGreedyRouter::with_seed(1.0, 0.5, 42);
            let qt = QueryType::new("k");
            let s = router.route_explore("q", &qt, 0.5);
            assert!(RetrievalStrategy::ALL.contains(&s));
        }

        #[tokio::test]
        async fn route_and_search_drives_real_index() {
            let index = InMemoryCodeIndex::default();
            index.add_text_file("src/lib.rs", "pub fn target_widget() {}\n", Some("h".into()));
            let hits = route_and_search(&index, RetrievalStrategy::Symbol, "target_widget", 5).await.unwrap();
            assert!(hits.iter().any(|h| h.title.contains("target_widget")));
        }
    }
}
#[rustfmt::skip]
pub mod rlef {
    //! RLEF — Reinforcement Learning from Execution Feedback (bible §11.7).
    //!
    //! The reward in RLEF comes from the **execution environment** (build/test/lint
    //! oracles), not from a human or a caller. This module makes that real:
    //!
    //!   * [`RewardConfig`] is the §11.7.2 reward-shaping table.
    //!   * [`reward_for`] maps an [`ExecutionOutcome`] → a scalar reward **derived**
    //!     from what the oracles reported (the previous code summed a
    //!     caller-supplied `reward` field; that is gone).
    //!   * [`assemble_dataset`] turns a set of attempts on the same task into the
    //!     `(context, response, reward)` tuples GRPO consumes, including the
    //!     group-relative advantage (§11.7.3) — real arithmetic, not a placeholder.
    //!   * [`RlefDaemon`] + [`ppl_gate`] are the documented training seam: the
    //!     gradient step itself is post-shell (Hawking Condense owns it), but the
    //!     reward derivation, group advantage, and PPL-gate decision are real.

    use hide_core::ids::RunId;
    use serde::{Deserialize, Serialize};

    /// What the oracles reported for one generation attempt (§11.7.2). This is the
    /// *cause* of the reward — the reward is computed from it, never supplied.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ExecutionOutcome {
        /// All oracles green.
        AllGreen,
        /// The build broke.
        BuildFail,
        /// Build ok, a test failed.
        TestFail,
        /// Build + tests ok, only a lint rule failed.
        LintOnly,
        /// The attempt timed out.
        Timeout,
    }

    /// A coarser per-signal feedback datum the harness can emit per oracle. Kept for
    /// compatibility with callers that report signals one at a time; folded into an
    /// [`ExecutionOutcome`] by [`ExecutionOutcome::from_signals`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum FeedbackSignal {
        BuildPassed,
        BuildFailed,
        TestPassed,
        TestFailed,
        LintFailed,
        Timeout,
    }

    impl ExecutionOutcome {
        /// Reduce a bag of per-oracle signals into the single worst outcome (the
        /// reward reflects the most severe failure, matching the §11.7.2 ladder:
        /// build break dominates a test fail dominates a lint-only fail).
        pub fn from_signals(signals: &[FeedbackSignal]) -> Self {
            if signals.iter().any(|s| matches!(s, FeedbackSignal::Timeout)) {
                return Self::Timeout;
            }
            if signals.iter().any(|s| matches!(s, FeedbackSignal::BuildFailed)) {
                return Self::BuildFail;
            }
            if signals.iter().any(|s| matches!(s, FeedbackSignal::TestFailed)) {
                return Self::TestFail;
            }
            if signals.iter().any(|s| matches!(s, FeedbackSignal::LintFailed)) {
                return Self::LintOnly;
            }
            Self::AllGreen
        }
    }

    /// The §11.7.2 reward-shaping table.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct RewardConfig {
        pub all_green: f32,
        pub build_fail: f32,
        pub test_fail: f32,
        pub lint_only: f32,
        pub timeout: f32,
    }

    impl Default for RewardConfig {
        fn default() -> Self {
            // The exact shaping the bible specifies (§11.7.2).
            Self { all_green: 1.0, build_fail: -1.0, test_fail: -0.5, lint_only: -0.25, timeout: -0.75 }
        }
    }

    /// Derive the scalar reward for an outcome from the shaping config. This is the
    /// load-bearing change: reward is a pure function of execution, not an input.
    pub fn reward_for(outcome: ExecutionOutcome, config: &RewardConfig) -> f32 {
        match outcome {
            ExecutionOutcome::AllGreen => config.all_green,
            ExecutionOutcome::BuildFail => config.build_fail,
            ExecutionOutcome::TestFail => config.test_fail,
            ExecutionOutcome::LintOnly => config.lint_only,
            ExecutionOutcome::Timeout => config.timeout,
        }
    }

    /// One generation attempt on a task: the prompt/context that produced it, the
    /// response, and the execution outcome it earned.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Attempt {
        pub context: String,
        pub response: String,
        pub outcome: ExecutionOutcome,
    }

    /// All attempts on a single task (a GRPO "group", §11.7.3).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct TaskGroup {
        pub run_id: RunId,
        pub task_id: String,
        pub attempts: Vec<Attempt>,
    }

    /// A training tuple with the GRPO group-relative advantage already computed.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct TrainingTuple {
        pub context: String,
        pub response: String,
        /// The raw execution reward (`reward_for`).
        pub reward: f32,
        /// `(reward - group_mean) / group_std` — the GRPO advantage (§11.7.3).
        pub advantage: f32,
    }

    /// Turn a task group into GRPO training tuples: derive each reward from its
    /// outcome, then normalize within the group to the group-relative advantage.
    pub fn assemble_group(group: &TaskGroup, config: &RewardConfig) -> Vec<TrainingTuple> {
        let rewards: Vec<f32> = group.attempts.iter().map(|a| reward_for(a.outcome, config)).collect();
        let (mean, std) = mean_std(&rewards);
        group
            .attempts
            .iter()
            .zip(&rewards)
            .map(|(a, &r)| TrainingTuple {
                context: a.context.clone(),
                response: a.response.clone(),
                reward: r,
                // std==0 (all attempts equal) → zero advantage, no gradient signal.
                advantage: if std > f32::EPSILON { (r - mean) / std } else { 0.0 },
            })
            .collect()
    }

    /// Assemble a whole batch of groups into one flat training set.
    pub fn assemble_dataset(groups: &[TaskGroup], config: &RewardConfig) -> Vec<TrainingTuple> {
        groups.iter().flat_map(|g| assemble_group(g, config)).collect()
    }

    fn mean_std(xs: &[f32]) -> (f32, f32) {
        if xs.is_empty() {
            return (0.0, 0.0);
        }
        let n = xs.len() as f32;
        let mean = xs.iter().sum::<f32>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        (mean, var.sqrt())
    }

    // ============================================================================
    // Daemon seam + PPL gate
    // ============================================================================

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct RlefConfig {
        pub tasks_per_batch: u32,
        pub attempts_per_task: u32,
        pub max_grad_steps: u32,
        /// Roll back if held-out PPL degrades by more than this many nats (§11.7.2).
        pub ppl_rollback_nats: f64,
        pub lora_rank: u32,
        pub learning_rate: f64,
        pub kl_penalty: f64,
        pub reward_shape: RewardConfig,
    }

    impl Default for RlefConfig {
        fn default() -> Self {
            Self {
                tasks_per_batch: 20,
                attempts_per_task: 4,
                max_grad_steps: 100,
                ppl_rollback_nats: 0.5,
                lora_rank: 16,
                learning_rate: 1e-5,
                kl_penalty: 0.02,
                reward_shape: RewardConfig::default(),
            }
        }
    }

    /// The §11.7.4 daemon. The gradient step is post-shell (a `GradientStepper` seam
    /// the trainer fills in); everything around it — dataset assembly, the PPL gate,
    /// the rollback decision — is real here.
    pub struct RlefDaemon {
        pub model_role: String,
        pub config: RlefConfig,
        pub ppl_baseline: f64,
    }

    impl RlefDaemon {
        pub fn new(model_role: impl Into<String>, ppl_baseline: f64) -> Self {
            Self { model_role: model_role.into(), config: RlefConfig::default(), ppl_baseline }
        }

        /// Assemble the training set for one overnight batch from the executed
        /// groups. Real work; the gradient step that consumes it is the seam below.
        pub fn prepare_batch(&self, groups: &[TaskGroup]) -> Vec<TrainingTuple> {
            assemble_dataset(groups, &self.config.reward_shape)
        }

        /// The PPL gate (§11.7.2): keep the new adapter iff held-out PPL did not
        /// degrade past the rollback threshold. `current_ppl` is measured by a
        /// forward pass (the [`PplEvaluator`] seam) — this is the *decision*, which
        /// is pure.
        pub fn ppl_gate_decision(&self, current_ppl: f64) -> GateOutcome {
            if current_ppl <= self.ppl_baseline + self.config.ppl_rollback_nats {
                GateOutcome::Keep { current_ppl }
            } else {
                GateOutcome::Rollback { current_ppl, baseline: self.ppl_baseline }
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum GateOutcome {
        Keep { current_ppl: f64 },
        Rollback { current_ppl: f64, baseline: f64 },
    }

    impl GateOutcome {
        pub fn keeps(&self) -> bool {
            matches!(self, GateOutcome::Keep { .. })
        }
    }

    /// Seam: the actual PPL measurement (a forward pass over held-out examples). The
    /// shell ships a stub; Hawking Condense provides the real implementation against
    /// a loaded adapter. Kept as a trait so the gate decision is testable without a
    /// model.
    pub trait PplEvaluator {
        fn held_out_ppl(&self, adapter_path: &std::path::Path) -> f64;
    }

    /// Convenience: evaluate PPL via the seam and apply the gate in one call.
    pub fn ppl_gate(daemon: &RlefDaemon, evaluator: &dyn PplEvaluator, adapter_path: &std::path::Path) -> GateOutcome {
        daemon.ppl_gate_decision(evaluator.held_out_ppl(adapter_path))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn group(outcomes: &[ExecutionOutcome]) -> TaskGroup {
            TaskGroup {
                run_id: RunId::new(),
                task_id: "t".into(),
                attempts: outcomes
                    .iter()
                    .enumerate()
                    .map(|(i, &o)| Attempt { context: format!("ctx{i}"), response: format!("resp{i}"), outcome: o })
                    .collect(),
            }
        }

        #[test]
        fn reward_is_derived_from_outcome() {
            let cfg = RewardConfig::default();
            assert_eq!(reward_for(ExecutionOutcome::AllGreen, &cfg), 1.0);
            assert_eq!(reward_for(ExecutionOutcome::BuildFail, &cfg), -1.0);
            assert_eq!(reward_for(ExecutionOutcome::TestFail, &cfg), -0.5);
            assert_eq!(reward_for(ExecutionOutcome::LintOnly, &cfg), -0.25);
            assert_eq!(reward_for(ExecutionOutcome::Timeout, &cfg), -0.75);
        }

        #[test]
        fn signals_fold_to_worst_outcome() {
            let s = [FeedbackSignal::BuildPassed, FeedbackSignal::TestFailed];
            assert_eq!(ExecutionOutcome::from_signals(&s), ExecutionOutcome::TestFail);
            let s2 = [FeedbackSignal::BuildFailed, FeedbackSignal::TestFailed];
            assert_eq!(ExecutionOutcome::from_signals(&s2), ExecutionOutcome::BuildFail);
            assert_eq!(ExecutionOutcome::from_signals(&[]), ExecutionOutcome::AllGreen);
        }

        #[test]
        fn group_advantage_is_zero_mean() {
            let g = group(&[
                ExecutionOutcome::AllGreen,
                ExecutionOutcome::BuildFail,
                ExecutionOutcome::AllGreen,
                ExecutionOutcome::TestFail,
            ]);
            let tuples = assemble_group(&g, &RewardConfig::default());
            let sum_adv: f32 = tuples.iter().map(|t| t.advantage).sum();
            assert!(sum_adv.abs() < 1e-4, "advantages should sum to ~0");
            // The all-green attempts have positive advantage; the failures negative.
            assert!(tuples[0].advantage > 0.0);
            assert!(tuples[1].advantage < 0.0);
        }

        #[test]
        fn identical_group_has_no_signal() {
            let g = group(&[ExecutionOutcome::AllGreen, ExecutionOutcome::AllGreen]);
            let tuples = assemble_group(&g, &RewardConfig::default());
            assert!(tuples.iter().all(|t| t.advantage == 0.0));
        }

        struct FixedPpl(f64);
        impl PplEvaluator for FixedPpl {
            fn held_out_ppl(&self, _: &std::path::Path) -> f64 {
                self.0
            }
        }

        #[test]
        fn ppl_gate_keeps_within_threshold_rolls_back_beyond() {
            let daemon = RlefDaemon::new("hero", 10.0);
            // +0.4 nats <= 0.5 threshold → keep.
            assert!(ppl_gate(&daemon, &FixedPpl(10.4), std::path::Path::new("a")).keeps());
            // +0.6 nats > 0.5 → rollback.
            assert!(!ppl_gate(&daemon, &FixedPpl(10.6), std::path::Path::new("a")).keeps());
        }
    }
}
#[rustfmt::skip]
pub mod store {
    //! Record persistence + the scrub-on-write + the curated `dataset/vNNN` layout
    //! (bible §11.1.1 / §11.1.2).
    //!
    //! Two real things beyond plain JSONL append:
    //!   * **Scrub-on-write** — every record's `diff_proposed` / `diff_accepted` is
    //!     run through the real [`hide_security::Redactor`] (the same detector suite
    //!     ch.10 applies to the event log) *before* it is persisted, so a secret in
    //!     a proposed diff never reaches disk (§11.1.1, the privacy invariant).
    //!   * **`PersonalLayout`** — the `~/.hawking/personal/{records,dataset/vNNN,
    //!     adapters,eval}` directory map (§11.1.2), with a real "next dataset
    //!     version" allocator so curate can write `dataset/v001`, `v002`, ….

    use crate::records::{PersonalizationRecord, TaskClass};
    use hide_core::{HideError, Result};
    use hide_security::Redactor;
    use parking_lot::Mutex;
    use std::fs::{File, OpenOptions};
    use std::io::{BufRead, BufReader, Write};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    pub type DynPersonalizationStore = Arc<dyn PersonalizationStore>;

    pub trait PersonalizationStore: Send + Sync {
        fn append(&self, record: &PersonalizationRecord) -> Result<()>;
        fn load_all(&self) -> Result<Vec<PersonalizationRecord>>;

        fn load_recent(&self, limit: usize) -> Result<Vec<PersonalizationRecord>> {
            let mut records = self.load_all()?;
            if records.len() > limit {
                records.drain(..records.len() - limit);
            }
            Ok(records)
        }

        fn load_by_task(&self, task_class: TaskClass, limit: usize) -> Result<Vec<PersonalizationRecord>> {
            let mut records: Vec<_> =
                self.load_all()?.into_iter().filter(|record| record.task_type == task_class).collect();
            if records.len() > limit {
                records.drain(..records.len() - limit);
            }
            Ok(records)
        }
    }

    /// Scrub a record's diffs in place using the supplied redactor. Returns the
    /// total number of redactions applied across both diffs (so the caller can emit
    /// a `security.redaction` event if it wants to, §4.8).
    pub fn scrub_record(redactor: &Redactor, record: &mut PersonalizationRecord) -> usize {
        let mut total = 0;
        let proposed = redactor.redact(&record.diff_proposed);
        total += proposed.redactions.iter().map(|r| r.occurrences).sum::<usize>();
        record.diff_proposed = proposed.text;
        let accepted = redactor.redact(&record.diff_accepted);
        total += accepted.redactions.iter().map(|r| r.occurrences).sum::<usize>();
        record.diff_accepted = accepted.text;
        total
    }

    #[derive(Debug, Default)]
    pub struct InMemoryPersonalizationStore {
        records: Mutex<Vec<PersonalizationRecord>>,
    }

    impl InMemoryPersonalizationStore {
        pub fn new() -> Self {
            Self::default()
        }
    }

    impl PersonalizationStore for InMemoryPersonalizationStore {
        fn append(&self, record: &PersonalizationRecord) -> Result<()> {
            self.records.lock().push(record.clone());
            Ok(())
        }

        fn load_all(&self) -> Result<Vec<PersonalizationRecord>> {
            Ok(self.records.lock().clone())
        }
    }

    /// JSONL store that **scrubs secrets on every write** (§11.1.1).
    pub struct JsonlPersonalizationStore {
        path: PathBuf,
        redactor: Redactor,
    }

    impl JsonlPersonalizationStore {
        /// Open (creating the file + parents) with the default redaction suite
        /// (pattern detectors + entropy catch-all).
        pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
            Self::open_with_redactor(path, Redactor::default())
        }

        pub fn open_with_redactor(path: impl Into<PathBuf>, redactor: Redactor) -> Result<Self> {
            let path = path.into();
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if !path.exists() {
                File::create(&path)?;
            }
            Ok(Self { path, redactor })
        }

        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    impl PersonalizationStore for JsonlPersonalizationStore {
        fn append(&self, record: &PersonalizationRecord) -> Result<()> {
            // Scrub-on-write: the secret never reaches disk.
            let mut scrubbed = record.clone();
            scrub_record(&self.redactor, &mut scrubbed);

            let mut file = OpenOptions::new().create(true).append(true).open(&self.path)?;
            serde_json::to_writer(&mut file, &scrubbed)?;
            file.write_all(b"\n")?;
            file.sync_data()?;
            Ok(())
        }

        fn load_all(&self) -> Result<Vec<PersonalizationRecord>> {
            read_records(&self.path)
        }
    }

    fn read_records(path: &Path) -> Result<Vec<PersonalizationRecord>> {
        if !path.exists() {
            return Ok(Vec::new());
        }
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();
        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record = serde_json::from_str(&line).map_err(|err| {
                HideError::Storage(format!(
                    "failed to parse personalization store {} line {}: {err}",
                    path.display(),
                    idx + 1
                ))
            })?;
            records.push(record);
        }
        Ok(records)
    }

    /// The `~/.hawking/personal/` directory map (§11.1.2).
    ///
    /// ```text
    /// <root>/
    ///   records/          # raw scrubbed JSONL, append-only, user-deletable
    ///   dataset/          # curated, versioned SFT records (v001, v002, …)
    ///   adapters/         # trained LoRA checkpoints (written by Hawking Condense)
    ///   eval/             # held-out accept-rate measurement
    /// ```
    #[derive(Debug, Clone)]
    pub struct PersonalLayout {
        root: PathBuf,
    }

    impl PersonalLayout {
        pub fn new(root: impl Into<PathBuf>) -> Self {
            Self { root: root.into() }
        }

        pub fn records_dir(&self) -> PathBuf {
            self.root.join("records")
        }

        pub fn dataset_dir(&self) -> PathBuf {
            self.root.join("dataset")
        }

        pub fn adapters_dir(&self) -> PathBuf {
            self.root.join("adapters")
        }

        pub fn eval_dir(&self) -> PathBuf {
            self.root.join("eval")
        }

        /// Create every directory in the map (idempotent).
        pub fn ensure(&self) -> Result<()> {
            for dir in [self.records_dir(), self.dataset_dir(), self.adapters_dir(), self.eval_dir()] {
                std::fs::create_dir_all(dir)?;
            }
            Ok(())
        }

        /// The directory for a specific dataset version, e.g. `dataset/v001`.
        pub fn dataset_version_dir(&self, version: u32) -> PathBuf {
            self.dataset_dir().join(format!("v{version:03}"))
        }

        /// Scan `dataset/` for `vNNN` directories and return the next free version
        /// (1 if none exist). Lets curate write a fresh `vNNN` without clobbering.
        pub fn next_dataset_version(&self) -> Result<u32> {
            let dir = self.dataset_dir();
            if !dir.exists() {
                return Ok(1);
            }
            let mut max = 0u32;
            for entry in std::fs::read_dir(&dir)? {
                let entry = entry?;
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if let Some(rest) = name.strip_prefix('v') {
                    if let Ok(n) = rest.parse::<u32>() {
                        max = max.max(n);
                    }
                }
            }
            Ok(max + 1)
        }

        /// Today's raw records file: `records/<date>.jsonl`. (Date is a coarse
        /// partition key; the bible's `<date>/<ulid>.jsonl` is a finer variant —
        /// either is replay-equivalent since records carry their own ids.)
        pub fn records_file_for_today(&self) -> PathBuf {
            // Avoid a chrono dep: derive a YYYYMMDD-ish key from the unix day.
            let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
            let day = secs / 86_400;
            self.records_dir().join(format!("day{day}.jsonl"))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::records::TaskClass;

        #[test]
        fn jsonl_personalization_store_roundtrips_records() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("records.jsonl");
            let store = JsonlPersonalizationStore::open(&path).unwrap();
            let first = PersonalizationRecord::accepted(TaskClass::EditCode, "prompt-a", "diff-a");
            let second = PersonalizationRecord::accepted(TaskClass::WriteTest, "prompt-b", "diff-b");

            store.append(&first).unwrap();
            store.append(&second).unwrap();

            let reopened = JsonlPersonalizationStore::open(&path).unwrap();
            let loaded = reopened.load_all().unwrap();
            assert_eq!(loaded.len(), 2);
            let recent = reopened.load_recent(1).unwrap();
            assert_eq!(recent[0].diff_accepted, "diff-b");
            let edit_records = reopened.load_by_task(TaskClass::EditCode, 10).unwrap();
            assert_eq!(edit_records.len(), 1);
        }

        #[test]
        fn scrub_on_write_removes_secret_from_disk() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("records.jsonl");
            let store = JsonlPersonalizationStore::open(&path).unwrap();
            // A GitHub PAT embedded in a proposed diff.
            let secret = "ghp_0123456789abcdefABCDEF0123456789abcdef";
            let rec =
                PersonalizationRecord::accepted(TaskClass::EditCode, "p", format!("+const TOKEN = \"{secret}\";"));
            store.append(&rec).unwrap();

            let raw = std::fs::read_to_string(&path).unwrap();
            assert!(!raw.contains(secret), "secret must not reach disk: {raw}");
            assert!(raw.contains("redacted"), "redaction marker expected");

            // The in-memory record we passed in is untouched (scrub is on the copy).
            assert!(rec.diff_proposed.contains(secret));
        }

        #[test]
        fn dataset_version_allocator() {
            let dir = tempfile::tempdir().unwrap();
            let layout = PersonalLayout::new(dir.path());
            layout.ensure().unwrap();
            assert_eq!(layout.next_dataset_version().unwrap(), 1);
            std::fs::create_dir_all(layout.dataset_version_dir(1)).unwrap();
            std::fs::create_dir_all(layout.dataset_version_dir(2)).unwrap();
            assert_eq!(layout.next_dataset_version().unwrap(), 3);
            assert!(layout.dataset_version_dir(7).ends_with("v007"));
        }
    }
}
#[rustfmt::skip]
pub mod world {
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
            Self { predicted_outcome: PredictedOutcome::Build, confidence, issues: Vec::new() }
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
            let is_pure_add = !diff_lines.is_empty() && diff_lines.iter().all(|l| l.starts_with('+') || l.is_empty());
            if is_pure_add {
                // Full-file replacement: strip the leading '+'.
                return diff_lines.iter().map(|l| l.strip_prefix('+').unwrap_or(l)).collect::<Vec<_>>().join("\n");
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
                                        message: format!("closing '{c}' on line {line_u32} has no matching opener"),
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
                        kind: if node.is_missing() { "missing_node".into() } else { "parse_error".into() },
                        file: path.to_path_buf(),
                        line,
                        message: format!(
                            "{} parse {} at line {line}",
                            lang.as_str(),
                            if node.is_missing() { "MISSING" } else { "ERROR" }
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
        fn predict_edit(&self, _path: &Path, _diff: &str, _context: &ProjectSnapshot) -> SimulationResult {
            // Tier-2 moonshot: no learned model yet → defer to Unknown.
            SimulationResult { predicted_outcome: PredictedOutcome::Unknown, confidence: 0.0, issues: Vec::new() }
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
            assert!(!res.issues.is_empty(), "an unbalanced edit must report at least one issue");
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
}

pub use records::{Hash32, Outcome, PersonalizationRecord, TaskClass};
pub use store::{
    scrub_record, DynPersonalizationStore, InMemoryPersonalizationStore, JsonlPersonalizationStore,
    PersonalLayout, PersonalizationStore,
};

pub use curate::{curate, write_dataset, CuratedDataset, CurationPolicy};
pub use eval::{
    run_eval, run_suite, AdapterGateReport, CandidateStatus, EvalCase, EvalMiner, EvalMinerConfig,
    EvalOracle, EvalResult, EvalTaskCandidate,
};
pub use kv_handoff::{
    copy_for_group, AgentId, BroadcastReport, GenerateRequest, KvHandle, KvKey, KvPrefixCopier,
    KvShareGroup,
};
pub use retrieval::{
    route_and_search, EpsilonGreedyRouter, LearnedRetrievalWeights, MetaRouter, QueryType,
    RetrievalOutcomeRecord, RetrievalStrategy,
};
pub use rlef::{
    assemble_dataset, assemble_group, ppl_gate, reward_for, Attempt, ExecutionOutcome,
    FeedbackSignal, GateOutcome, PplEvaluator, RewardConfig, RlefConfig, RlefDaemon, TaskGroup,
    TrainingTuple,
};
