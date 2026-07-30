//! Support-halo evaluation for Odyssey G5 / T7.
//!
//! Pure, offline, deterministic scoring over a frozen corpus. Generation against
//! a live `.gravity` serve path is intentionally out of this module: feed
//! pre-recorded completions, or use the Python gate to collect them later.
//!
//! A missing measurement is [`DimensionScore::NotMeasurable`], never a fabricated
//! pass rate.

use crate::{wilson_interval, EvalError, Z_95};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The seven support-halo dimensions G5 requires preserved.
pub const DIMENSIONS: [&str; 7] = [
    "technical_language",
    "general_reasoning",
    "coding",
    "retrieval",
    "tools",
    "long_context",
    "self_correction",
];

/// Minimum measurable dimensions before an aggregate score is honest enough for
/// a tournament (from SUPPORT_HALO_SCORING_RULES.json).
pub const MIN_MEASURABLE_DIMENSIONS: usize = 4;

/// Absolute aggregate drop vs baseline that counts as a support-halo regression.
pub const REGRESSION_AGGREGATE_EPS: f64 = 0.05;

/// Absolute per-dimension drop (with non-overlapping Wilson CIs) that counts as
/// a per-dimension regression.
pub const REGRESSION_DIMENSION_DROP: f64 = 0.15;

/// Consecutive identical whitespace-split tokens that trigger DEGENERATE_REPETITION.
pub const DEGENERATE_REPEAT_TOKENS: usize = 8;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Oracle {
    ExpectAll,
    Exact,
    Execution,
    ToolJson,
    Needle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HaloTask {
    pub id: String,
    pub dimension: String,
    pub oracle: Oracle,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub prompt_template: Option<String>,
    #[serde(default)]
    pub expect: Vec<String>,
    #[serde(default)]
    pub accept_any_of: Vec<String>,
    #[serde(default)]
    pub exact: Option<String>,
    #[serde(default)]
    pub lang: Option<String>,
    #[serde(default)]
    pub entry: Option<String>,
    #[serde(default)]
    pub test: Option<String>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub required_argument_keys: Vec<String>,
    #[serde(default)]
    pub argument_equals: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub needle: Option<String>,
    #[serde(default)]
    pub needle_answer: Option<String>,
    #[serde(default)]
    pub haystack_chars: Option<usize>,
    #[serde(default)]
    pub needle_offset_frac: Option<f64>,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
}

fn default_max_new_tokens() -> usize {
    128
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskScore {
    pub id: String,
    pub dimension: String,
    pub passed: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DimensionScore {
    Measured {
        dimension: String,
        passes: usize,
        total: usize,
        pass_rate: f64,
        ci_low: f64,
        ci_high: f64,
    },
    NotMeasurable {
        dimension: String,
        reason: String,
    },
}

impl DimensionScore {
    pub fn pass_rate(&self) -> Option<f64> {
        match self {
            Self::Measured { pass_rate, .. } => Some(*pass_rate),
            Self::NotMeasurable { .. } => None,
        }
    }

    pub fn dimension(&self) -> &str {
        match self {
            Self::Measured { dimension, .. } | Self::NotMeasurable { dimension, .. } => dimension,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AggregateScore {
    Measured {
        score: f64,
        measurable_dimensions: usize,
    },
    NotMeasurable {
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Disqualification {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HaloReport {
    pub schema: String,
    pub rules_sha256: String,
    pub corpus_sha256: String,
    pub artifact_sha256: String,
    pub seed: u64,
    pub no_hidden_fallback: bool,
    pub task_scores: Vec<TaskScore>,
    pub dimensions: Vec<DimensionScore>,
    pub aggregate: AggregateScore,
    pub disqualifications: Vec<Disqualification>,
    pub total_completion_chars: usize,
}

/// Load support-halo tasks from JSONL (one object per line).
pub fn load_halo_tasks_jsonl(text: &str) -> Result<Vec<HaloTask>, EvalError> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let task: HaloTask = serde_json::from_str(line)
            .map_err(|e| EvalError::Decode(format!("line {}: {e}", i + 1)))?;
        out.push(task);
    }
    Ok(out)
}

/// Expand a needle task's prompt_template into a concrete prompt. Pure and
/// deterministic for a fixed task definition.
pub fn expand_needle_prompt(task: &HaloTask) -> Result<String, EvalError> {
    let template = task
        .prompt_template
        .as_deref()
        .ok_or_else(|| EvalError::Decode(format!("task {}: missing prompt_template", task.id)))?;
    let needle = task
        .needle
        .as_deref()
        .ok_or_else(|| EvalError::Decode(format!("task {}: missing needle", task.id)))?;
    let haystack_chars = task.haystack_chars.unwrap_or(3500);
    let frac = task.needle_offset_frac.unwrap_or(0.25).clamp(0.0, 1.0);
    let filler = "lorem context pad. ";
    let mut hay = String::with_capacity(haystack_chars + needle.len() + 16);
    while hay.len() < haystack_chars {
        hay.push_str(filler);
    }
    hay.truncate(haystack_chars);
    let insert_at = ((hay.len() as f64) * frac) as usize;
    let insert_at = insert_at.min(hay.len());
    hay.insert_str(insert_at, &format!(" {needle} "));
    Ok(template.replace("{haystack}", &hay))
}

/// Concrete prompt text for a task (needle templates expanded).
pub fn task_prompt(task: &HaloTask) -> Result<String, EvalError> {
    match task.oracle {
        Oracle::Needle => expand_needle_prompt(task),
        _ => task
            .prompt
            .clone()
            .ok_or_else(|| EvalError::Decode(format!("task {}: missing prompt", task.id))),
    }
}

/// Score a non-execution, non-tool completion against a task oracle.
///
/// Execution and tool_json oracles that need structured parsing are handled by
/// [`score_task_completion`]; this helper is the pure string path.
pub fn score_expect_or_exact(task: &HaloTask, completion: &str) -> (bool, String) {
    let text = completion.trim();
    if !task.accept_any_of.is_empty() {
        let lower = text.to_ascii_lowercase();
        let hit = task
            .accept_any_of
            .iter()
            .any(|a| lower.contains(&a.to_ascii_lowercase()));
        return if hit {
            (true, "ok".into())
        } else {
            (false, "accept_any_of miss".into())
        };
    }
    match task.oracle {
        Oracle::Exact | Oracle::Needle => {
            let want = task
                .exact
                .as_deref()
                .or(task.needle_answer.as_deref())
                .unwrap_or("");
            // Accept exact match after trim, or first line exact, or contains as
            // sole token-ish answer (models sometimes prefix "Answer:").
            let first = text.lines().next().unwrap_or("").trim();
            let ok = first.eq_ignore_ascii_case(want)
                || text.eq_ignore_ascii_case(want)
                || first
                    .split_whitespace()
                    .any(|t| t.eq_ignore_ascii_case(want));
            if ok {
                (true, "ok".into())
            } else {
                (false, format!("expected exact {want:?}"))
            }
        }
        Oracle::ExpectAll => {
            let ok = task.expect.iter().all(|e| text.contains(e.as_str()));
            if ok {
                (true, "ok".into())
            } else {
                (false, "expect_all miss".into())
            }
        }
        Oracle::Execution | Oracle::ToolJson => {
            // Callers should use the specialized scorers; treat bare text as fail.
            (false, "use specialized scorer".into())
        }
    }
}

/// Score a tool_json completion: must be a JSON object (optionally fenced) with
/// the required name and argument keys/values.
pub fn score_tool_json(task: &HaloTask, completion: &str) -> (bool, String) {
    let raw = strip_code_fence(completion);
    let v: serde_json::Value = match serde_json::from_str(raw.trim()) {
        Ok(v) => v,
        Err(e) => return (false, format!("json: {e}")),
    };
    let obj = match v.as_object() {
        Some(o) => o,
        None => return (false, "not an object".into()),
    };
    let name = obj.get("name").and_then(|x| x.as_str()).unwrap_or("");
    let want_name = task.tool_name.as_deref().unwrap_or("");
    if name != want_name {
        return (false, format!("name {name:?} != {want_name:?}"));
    }
    let args = match obj.get("arguments") {
        Some(a) => a,
        None => return (false, "missing arguments".into()),
    };
    let args_obj = match args.as_object() {
        Some(o) => o,
        None => return (false, "arguments not object".into()),
    };
    for k in &task.required_argument_keys {
        if !args_obj.contains_key(k) {
            return (false, format!("missing arg {k}"));
        }
    }
    for (k, want) in &task.argument_equals {
        match args_obj.get(k) {
            Some(got) if json_values_equal(got, want) => {}
            Some(got) => return (false, format!("arg {k}: {got} != {want}")),
            None => return (false, format!("missing equals arg {k}")),
        }
    }
    (true, "ok".into())
}

fn json_values_equal(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    if a == b {
        return true;
    }
    // Allow "3" vs 3 for tool argument convenience.
    match (a, b) {
        (serde_json::Value::Number(n), serde_json::Value::String(s))
        | (serde_json::Value::String(s), serde_json::Value::Number(n)) => {
            s.parse::<f64>().ok() == n.as_f64()
        }
        _ => false,
    }
}

/// Pull the first fenced code block; fall back to body with fence lines stripped.
pub fn strip_code_fence(text: &str) -> &str {
    // Handled via owned String in extract_code for mutation; keep simple scan.
    text
}

/// Extract code from a fenced completion (mirrors tools/eval/thesis_gate.py).
pub fn extract_code(text: &str) -> String {
    let bytes = text.as_bytes();
    // Find opening fence of 2+ backticks.
    if let Some(start) = find_fence_open(bytes) {
        let after = &text[start..];
        // skip backticks + optional lang + newline
        let mut i = 0;
        while i < after.len() && after.as_bytes()[i] == b'`' {
            i += 1;
        }
        while i < after.len() && after.as_bytes()[i] != b'\n' {
            i += 1;
        }
        if i < after.len() && after.as_bytes()[i] == b'\n' {
            i += 1;
        }
        let body = &after[i..];
        if let Some(end) = find_fence_close(body.as_bytes()) {
            return body[..end].trim().to_string();
        }
    }
    text.lines()
        .filter(|ln| !ln.trim_start().starts_with('`'))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .trim_matches('`')
        .to_string()
}

fn find_fence_open(bytes: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'`' && bytes[i + 1] == b'`' {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn find_fence_close(bytes: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'`' && bytes[i + 1] == b'`' {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Detect degenerate repetition in a completion (screening DQ).
pub fn has_degenerate_repetition(text: &str, n: usize) -> bool {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.len() < n {
        return false;
    }
    let mut run = 1usize;
    for i in 1..tokens.len() {
        if tokens[i] == tokens[i - 1] {
            run += 1;
            if run >= n {
                return true;
            }
        } else {
            run = 1;
        }
    }
    false
}

/// Score one task given a completion. For `execution` oracles the caller must
/// supply `execution_passed` (subprocess result); pass `None` to mark the task
/// NOT measured at the task level (treated as fail with reason deferred).
pub fn score_task_completion(
    task: &HaloTask,
    completion: &str,
    execution_passed: Option<(bool, String)>,
) -> TaskScore {
    let (passed, reason) = match task.oracle {
        Oracle::Execution => match execution_passed {
            Some((p, r)) => (p, r),
            None => (false, "execution not run".into()),
        },
        Oracle::ToolJson => score_tool_json(task, completion),
        Oracle::ExpectAll | Oracle::Exact | Oracle::Needle => {
            score_expect_or_exact(task, completion)
        }
    };
    TaskScore {
        id: task.id.clone(),
        dimension: task.dimension.clone(),
        passed,
        reason,
    }
}

/// Aggregate task scores into per-dimension and overall report fields.
pub fn aggregate_scores(
    tasks: &[HaloTask],
    task_scores: &[TaskScore],
    rules_sha256: impl Into<String>,
    corpus_sha256: impl Into<String>,
    artifact_sha256: impl Into<String>,
    seed: u64,
    no_hidden_fallback: bool,
    completions: &BTreeMap<String, String>,
    extra_dq: Vec<Disqualification>,
) -> HaloReport {
    let mut by_dim: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for d in DIMENSIONS {
        by_dim.insert(d.to_string(), (0, 0));
    }
    for t in tasks {
        by_dim.entry(t.dimension.clone()).or_insert((0, 0));
    }
    for s in task_scores {
        let e = by_dim.entry(s.dimension.clone()).or_insert((0, 0));
        e.1 += 1;
        if s.passed {
            e.0 += 1;
        }
    }

    let mut dimensions = Vec::new();
    for d in DIMENSIONS {
        let (passes, total) = by_dim.get(d).copied().unwrap_or((0, 0));
        if total == 0 {
            dimensions.push(DimensionScore::NotMeasurable {
                dimension: d.to_string(),
                reason: "no tasks measured for this dimension".into(),
            });
        } else {
            let (ci_low, ci_high) = wilson_interval(passes, total, Z_95);
            dimensions.push(DimensionScore::Measured {
                dimension: d.to_string(),
                passes,
                total,
                pass_rate: passes as f64 / total as f64,
                ci_low,
                ci_high,
            });
        }
    }

    let measurable: Vec<f64> = dimensions.iter().filter_map(|d| d.pass_rate()).collect();
    let aggregate = if measurable.len() < MIN_MEASURABLE_DIMENSIONS {
        AggregateScore::NotMeasurable {
            reason: format!(
                "only {} measurable dimensions; need >= {}",
                measurable.len(),
                MIN_MEASURABLE_DIMENSIONS
            ),
        }
    } else {
        let score = measurable.iter().sum::<f64>() / measurable.len() as f64;
        AggregateScore::Measured {
            score,
            measurable_dimensions: measurable.len(),
        }
    };

    let mut disqualifications = extra_dq;
    if !no_hidden_fallback {
        disqualifications.push(Disqualification {
            code: "HIDDEN_FALLBACK".into(),
            detail: "receipt did not assert no_hidden_fallback=true".into(),
        });
    }

    let artifact_sha256 = artifact_sha256.into();
    if artifact_sha256.is_empty() {
        disqualifications.push(Disqualification {
            code: "ARTIFACT_MISMATCH".into(),
            detail: "artifact_sha256 is empty".into(),
        });
    }

    let mut empty = 0usize;
    let mut measured = 0usize;
    let mut total_chars = 0usize;
    for s in task_scores {
        if let Some(c) = completions.get(&s.id) {
            measured += 1;
            total_chars += c.len();
            if c.trim().is_empty() {
                empty += 1;
            }
            if has_degenerate_repetition(c, DEGENERATE_REPEAT_TOKENS) {
                disqualifications.push(Disqualification {
                    code: "DEGENERATE_REPETITION".into(),
                    detail: format!(
                        "task {} has >= {DEGENERATE_REPEAT_TOKENS} identical consecutive tokens",
                        s.id
                    ),
                });
            }
        }
    }
    if measured > 0 && (empty as f64) / (measured as f64) > 0.5 {
        disqualifications.push(Disqualification {
            code: "EMPTY_GENERATION".into(),
            detail: format!("{empty}/{measured} completions empty"),
        });
    }

    // De-dupe DQ codes (keep first detail).
    let mut seen = std::collections::BTreeSet::new();
    disqualifications.retain(|d| seen.insert(d.code.clone()));

    HaloReport {
        schema: "hawking.odyssey.support_halo.report.v0".into(),
        rules_sha256: rules_sha256.into(),
        corpus_sha256: corpus_sha256.into(),
        artifact_sha256,
        seed,
        no_hidden_fallback,
        task_scores: task_scores.to_vec(),
        dimensions,
        aggregate,
        disqualifications,
        total_completion_chars: total_chars,
    }
}

/// Tournament comparison: returns -1 if a ranks better, 1 if b ranks better, 0 if
/// still tied after all tie-breaks (should only happen if sha equal).
///
/// Disqualified checkpoints always lose to non-disqualified ones. Two
/// disqualified checkpoints compare by artifact_sha256 only.
pub fn compare_reports(a: &HaloReport, b: &HaloReport) -> i8 {
    let a_dq = !a.disqualifications.is_empty();
    let b_dq = !b.disqualifications.is_empty();
    match (a_dq, b_dq) {
        (true, false) => return 1,
        (false, true) => return -1,
        (true, true) => {
            return cmp_sha(&a.artifact_sha256, &b.artifact_sha256);
        }
        (false, false) => {}
    }

    // Rank 1: higher aggregate
    match (&a.aggregate, &b.aggregate) {
        (
            AggregateScore::Measured { score: sa, .. },
            AggregateScore::Measured { score: sb, .. },
        ) => {
            if (*sa - *sb).abs() > 1e-12 {
                return if sa > sb { -1 } else { 1 };
            }
        }
        (AggregateScore::Measured { .. }, AggregateScore::NotMeasurable { .. }) => return -1,
        (AggregateScore::NotMeasurable { .. }, AggregateScore::Measured { .. }) => return 1,
        _ => {}
    }

    // Rank 2: higher min measurable dimension
    let min_a = min_measurable(a);
    let min_b = min_measurable(b);
    match (min_a, min_b) {
        (Some(sa), Some(sb)) if (sa - sb).abs() > 1e-12 => {
            return if sa > sb { -1 } else { 1 };
        }
        (Some(_), None) => return -1,
        (None, Some(_)) => return 1,
        _ => {}
    }

    // Rank 3: higher coding pass_rate
    let ca = dim_rate(a, "coding");
    let cb = dim_rate(b, "coding");
    match (ca, cb) {
        (Some(sa), Some(sb)) if (sa - sb).abs() > 1e-12 => {
            return if sa > sb { -1 } else { 1 };
        }
        (Some(_), None) => return -1,
        (None, Some(_)) => return 1,
        _ => {}
    }

    // Rank 4: lower total_completion_chars
    if a.total_completion_chars != b.total_completion_chars {
        return if a.total_completion_chars < b.total_completion_chars {
            -1
        } else {
            1
        };
    }

    // Rank 5: smaller artifact sha
    cmp_sha(&a.artifact_sha256, &b.artifact_sha256)
}

fn cmp_sha(a: &str, b: &str) -> i8 {
    match a.cmp(b) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Equal => 0,
    }
}

fn min_measurable(r: &HaloReport) -> Option<f64> {
    r.dimensions
        .iter()
        .filter_map(|d| d.pass_rate())
        .reduce(f64::min)
}

fn dim_rate(r: &HaloReport, name: &str) -> Option<f64> {
    r.dimensions
        .iter()
        .find(|d| d.dimension() == name)
        .and_then(|d| d.pass_rate())
}

/// Regression check against a sealed baseline report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegressionVerdict {
    pub aggregate_regression: bool,
    pub dimension_regressions: Vec<String>,
    pub blocks_t7_winner: bool,
}

pub fn regression_vs_baseline(candidate: &HaloReport, baseline: &HaloReport) -> RegressionVerdict {
    let mut dimension_regressions = Vec::new();
    let mut aggregate_regression = false;

    if let (AggregateScore::Measured { score: c, .. }, AggregateScore::Measured { score: b, .. }) =
        (&candidate.aggregate, &baseline.aggregate)
    {
        if *c < *b - REGRESSION_AGGREGATE_EPS {
            aggregate_regression = true;
        }
    }

    for dim in DIMENSIONS {
        let c = candidate.dimensions.iter().find(|d| d.dimension() == dim);
        let b = baseline.dimensions.iter().find(|d| d.dimension() == dim);
        if let (
            Some(DimensionScore::Measured {
                pass_rate: cr,
                ci_low: c_lo,
                ci_high: c_hi,
                ..
            }),
            Some(DimensionScore::Measured {
                pass_rate: br,
                ci_low: b_lo,
                ci_high: b_hi,
                ..
            }),
        ) = (c, b)
        {
            let drop = *br - *cr;
            let non_overlap = *c_hi < *b_lo || *b_hi < *c_lo;
            if drop > REGRESSION_DIMENSION_DROP && non_overlap {
                dimension_regressions.push(dim.to_string());
            }
        }
    }

    let blocks_t7_winner = aggregate_regression || !dimension_regressions.is_empty();
    // Also: any DQ on candidate blocks winner.
    let blocks_t7_winner = blocks_t7_winner || !candidate.disqualifications.is_empty();

    RegressionVerdict {
        aggregate_regression,
        dimension_regressions,
        blocks_t7_winner,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample_tasks() -> Vec<HaloTask> {
        vec![
            HaloTask {
                id: "t1".into(),
                dimension: "technical_language".into(),
                oracle: Oracle::ExpectAll,
                prompt: Some("p".into()),
                prompt_template: None,
                expect: vec!["foo".into()],
                accept_any_of: vec![],
                exact: None,
                lang: None,
                entry: None,
                test: None,
                tool_name: None,
                required_argument_keys: vec![],
                argument_equals: BTreeMap::new(),
                needle: None,
                needle_answer: None,
                haystack_chars: None,
                needle_offset_frac: None,
                max_new_tokens: 32,
            },
            HaloTask {
                id: "t2".into(),
                dimension: "general_reasoning".into(),
                oracle: Oracle::Exact,
                prompt: Some("p".into()),
                prompt_template: None,
                expect: vec![],
                accept_any_of: vec![],
                exact: Some("YES".into()),
                lang: None,
                entry: None,
                test: None,
                tool_name: None,
                required_argument_keys: vec![],
                argument_equals: BTreeMap::new(),
                needle: None,
                needle_answer: None,
                haystack_chars: None,
                needle_offset_frac: None,
                max_new_tokens: 8,
            },
            HaloTask {
                id: "t3".into(),
                dimension: "coding".into(),
                oracle: Oracle::Execution,
                prompt: Some("p".into()),
                prompt_template: None,
                expect: vec![],
                accept_any_of: vec![],
                exact: None,
                lang: Some("python".into()),
                entry: Some("f".into()),
                test: Some("assert True".into()),
                tool_name: None,
                required_argument_keys: vec![],
                argument_equals: BTreeMap::new(),
                needle: None,
                needle_answer: None,
                haystack_chars: None,
                needle_offset_frac: None,
                max_new_tokens: 64,
            },
            HaloTask {
                id: "t4".into(),
                dimension: "retrieval".into(),
                oracle: Oracle::Exact,
                prompt: Some("p".into()),
                prompt_template: None,
                expect: vec![],
                accept_any_of: vec![],
                exact: Some("Z".into()),
                lang: None,
                entry: None,
                test: None,
                tool_name: None,
                required_argument_keys: vec![],
                argument_equals: BTreeMap::new(),
                needle: None,
                needle_answer: None,
                haystack_chars: None,
                needle_offset_frac: None,
                max_new_tokens: 8,
            },
            HaloTask {
                id: "t5".into(),
                dimension: "tools".into(),
                oracle: Oracle::ToolJson,
                prompt: Some("p".into()),
                prompt_template: None,
                expect: vec![],
                accept_any_of: vec![],
                exact: None,
                lang: None,
                entry: None,
                test: None,
                tool_name: Some("read_file".into()),
                required_argument_keys: vec!["path".into()],
                argument_equals: {
                    let mut m = BTreeMap::new();
                    m.insert("path".into(), serde_json::json!("x"));
                    m
                },
                needle: None,
                needle_answer: None,
                haystack_chars: None,
                needle_offset_frac: None,
                max_new_tokens: 64,
            },
            HaloTask {
                id: "t6".into(),
                dimension: "long_context".into(),
                oracle: Oracle::Needle,
                prompt: None,
                prompt_template: Some("Q: {haystack}\ncode?".into()),
                expect: vec![],
                accept_any_of: vec![],
                exact: None,
                lang: None,
                entry: None,
                test: None,
                tool_name: None,
                required_argument_keys: vec![],
                argument_equals: BTreeMap::new(),
                needle: Some("SECRET_PASSCODE=ABC".into()),
                needle_answer: Some("ABC".into()),
                haystack_chars: Some(200),
                needle_offset_frac: Some(0.3),
                max_new_tokens: 16,
            },
            HaloTask {
                id: "t7".into(),
                dimension: "self_correction".into(),
                oracle: Oracle::Exact,
                prompt: Some("p".into()),
                prompt_template: None,
                expect: vec![],
                accept_any_of: vec![],
                exact: Some("391".into()),
                lang: None,
                entry: None,
                test: None,
                tool_name: None,
                required_argument_keys: vec![],
                argument_equals: BTreeMap::new(),
                needle: None,
                needle_answer: None,
                haystack_chars: None,
                needle_offset_frac: None,
                max_new_tokens: 8,
            },
        ]
    }
    #[test]
    fn tool_json_scores_strict_object() {
        let tasks = sample_tasks();
        let t = tasks.iter().find(|t| t.id == "t5").unwrap();
        let (ok, _) = score_tool_json(t, r#"{"name":"read_file","arguments":{"path":"x"}}"#);
        assert!(ok);
        let (bad, _) = score_tool_json(t, r#"{"name":"read_file","arguments":{"path":"y"}}"#);
        assert!(!bad);
    }
    #[test]
    fn needle_prompt_is_deterministic() {
        let tasks = sample_tasks();
        let t = tasks.iter().find(|t| t.id == "t6").unwrap();
        let a = expand_needle_prompt(t).unwrap();
        let b = expand_needle_prompt(t).unwrap();
        assert_eq!(a, b);
        assert!(a.contains("SECRET_PASSCODE=ABC"));
    }
    #[test]
    fn degenerate_repetition_detects_loops() {
        assert!(has_degenerate_repetition(
            "settle settle settle settle settle settle settle settle",
            8
        ));
        assert!(!has_degenerate_repetition("a b c d e f g h", 8));
    }
    #[test]
    fn extract_code_handles_fence() {
        let c = extract_code("here\n```python\ndef f():\n  return 1\n```\n");
        assert!(c.contains("def f()"));
    }
    #[test]
    fn aggregate_and_tournament_are_deterministic() {
        let tasks = sample_tasks();
        let mut completions: BTreeMap<String, String> = BTreeMap::new();
        completions.insert("t1".to_string(), "has foo".to_string());
        completions.insert("t2".to_string(), "YES".to_string());
        completions.insert("t3".to_string(), "code".to_string());
        completions.insert("t4".to_string(), "Z".to_string());
        completions.insert(
            "t5".to_string(),
            r#"{"name":"read_file","arguments":{"path":"x"}}"#.to_string(),
        );
        completions.insert("t6".to_string(), "ABC".to_string());
        completions.insert("t7".to_string(), "391".to_string());
        let scores: Vec<TaskScore> = tasks
            .iter()
            .map(|t| {
                let exec = if t.id == "t3" {
                    Some((true, "ok".into()))
                } else {
                    None
                };
                score_task_completion(t, completions.get(&t.id).unwrap(), exec)
            })
            .collect();
        let r1 = aggregate_scores(
            &tasks,
            &scores,
            "rules",
            "corpus",
            "aaa",
            0,
            true,
            &completions,
            vec![],
        );
        let r2 = aggregate_scores(
            &tasks,
            &scores,
            "rules",
            "corpus",
            "aaa",
            0,
            true,
            &completions,
            vec![],
        );
        assert_eq!(r1, r2);
        assert!(matches!(r1.aggregate, AggregateScore::Measured { .. }));
        assert!(r1.disqualifications.is_empty());
        let mut worse = r1.clone();
        if let AggregateScore::Measured { score, .. } = &mut worse.aggregate {
            *score -= 0.5;
        }
        assert_eq!(compare_reports(&r1, &worse), -1);
        assert_eq!(compare_reports(&worse, &r1), 1);
        let mut other = r1.clone();
        other.artifact_sha256 = "bbb".into();
        assert_eq!(compare_reports(&r1, &other), -1);
    }
    #[test]
    fn hidden_fallback_disqualifies() {
        let tasks = sample_tasks();
        let scores: Vec<TaskScore> = tasks
            .iter()
            .map(|t| TaskScore {
                id: t.id.clone(),
                dimension: t.dimension.clone(),
                passed: true,
                reason: "ok".into(),
            })
            .collect();
        let comps = BTreeMap::new();
        let r = aggregate_scores(&tasks, &scores, "r", "c", "art", 0, false, &comps, vec![]);
        assert!(r
            .disqualifications
            .iter()
            .any(|d| d.code == "HIDDEN_FALLBACK"));
    }
    #[test]
    fn regression_blocks_t7_on_aggregate_drop() {
        let tasks = sample_tasks();
        let scores: Vec<TaskScore> = tasks
            .iter()
            .map(|t| TaskScore {
                id: t.id.clone(),
                dimension: t.dimension.clone(),
                passed: true,
                reason: "ok".into(),
            })
            .collect();
        let comps = BTreeMap::new();
        let baseline = aggregate_scores(&tasks, &scores, "r", "c", "base", 0, true, &comps, vec![]);
        let mut candidate = baseline.clone();
        candidate.artifact_sha256 = "cand".into();
        if let AggregateScore::Measured { score, .. } = &mut candidate.aggregate {
            *score -= 0.2;
        }
        let v = regression_vs_baseline(&candidate, &baseline);
        assert!(v.aggregate_regression);
        assert!(v.blocks_t7_winner);
    }
    #[test]
    fn load_corpus_parses_repo_fixture_shape() {
        let line = r#"{"id":"x","dimension":"coding","oracle":"execution","prompt":"p","lang":"python","entry":"f","test":"assert True","max_new_tokens":32}"#;
        let tasks = load_halo_tasks_jsonl(line).unwrap();
        assert_eq!(tasks.len(), 1);
        assert!(matches!(tasks[0].oracle, Oracle::Execution));
    }
    #[test]
    fn accept_any_of_works() {
        let t = HaloTask {
            id: "a".into(),
            dimension: "self_correction".into(),
            oracle: Oracle::ExpectAll,
            prompt: Some("p".into()),
            prompt_template: None,
            expect: vec![],
            accept_any_of: vec!["red".into(), "blue".into()],
            exact: None,
            lang: None,
            entry: None,
            test: None,
            tool_name: None,
            required_argument_keys: vec![],
            argument_equals: BTreeMap::new(),
            needle: None,
            needle_answer: None,
            haystack_chars: None,
            needle_offset_frac: None,
            max_new_tokens: 8,
        };
        let (ok, _) = score_expect_or_exact(&t, "I pick Red");
        assert!(ok);
        let (bad, _) = score_expect_or_exact(&t, "green");
        assert!(!bad);
    }
}
