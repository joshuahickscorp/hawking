//! Merge & conflict resolution (bible ch.09 §4.4).
//!
//! Two distinct flows funnel through an integration branch:
//! - **Tournament** (same goal → select one winner): regression-filter →
//!   oracle-rank → judge tie-break (§4.4.2, P4). The selector is oracle-first by
//!   construction; the judge only breaks ties among oracle-equivalent leaders.
//! - **Fan-out / map-reduce** (disjoint footprints → combine all): the
//!   integration funnel merges each run's changes, with the conflict ladder
//!   (structured → 3-way → escalate, §4.4.3).
//!
//! **Footprint-disjoint scheduling** (§4.2.3) is the single biggest lever against
//! "merge is the hard part": subtasks with disjoint file footprints parallelize
//! freely (no conflict possible by construction); overlapping ones serialize or
//! race under tournament semantics.
//!
//! The 3-way merge is real (the `similar` crate's diff over the common ancestor);
//! it is *content* merge, not a git-process invocation, so it is unit-testable
//! and runs in-memory on candidate file blobs.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

// ---------------------------------------------------------------------------
// Footprint analysis (§4.2.3)
// ---------------------------------------------------------------------------

/// A subtask's predicted file footprint (from the plan's target paths + a cheap
/// static touch-set). Disjointness is decided by set intersection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Footprint {
    pub job_id: String,
    pub files: BTreeSet<String>,
}

impl Footprint {
    pub fn new(job_id: impl Into<String>, files: impl IntoIterator<Item = String>) -> Self {
        Self {
            job_id: job_id.into(),
            files: files.into_iter().collect(),
        }
    }

    pub fn overlaps(&self, other: &Footprint) -> bool {
        !self.files.is_disjoint(&other.files)
    }
}

/// How a set of subtasks should be scheduled relative to each other.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FootprintPlan {
    /// Groups that may run fully in parallel (mutually disjoint footprints).
    pub parallel_groups: Vec<Vec<String>>,
    /// Pairs that overlap and must serialize (a dependency edge is added) or race
    /// under tournament semantics.
    pub overlaps: Vec<(String, String)>,
}

/// Partition subtasks into parallel-safe groups + overlap edges. Greedy
/// graph-coloring: a job joins an existing group iff it is disjoint from every
/// member; otherwise it opens a new group, and the conflicting pair is recorded.
pub fn plan_footprints(footprints: &[Footprint]) -> FootprintPlan {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut overlaps = Vec::new();
    for (i, fp) in footprints.iter().enumerate() {
        // Record overlap edges against all prior jobs.
        for prior in footprints.iter().take(i) {
            if fp.overlaps(prior) {
                overlaps.push((prior.job_id.clone(), fp.job_id.clone()));
            }
        }
        // Place into the first group with no conflicting member.
        let mut placed = false;
        for group in &mut groups {
            let conflict = group.iter().any(|&j| footprints[j].overlaps(fp));
            if !conflict {
                group.push(i);
                placed = true;
                break;
            }
        }
        if !placed {
            groups.push(vec![i]);
        }
    }
    FootprintPlan {
        parallel_groups: groups
            .into_iter()
            .map(|g| {
                g.into_iter()
                    .map(|i| footprints[i].job_id.clone())
                    .collect()
            })
            .collect(),
        overlaps,
    }
}

// ---------------------------------------------------------------------------
// Tournament selection (§4.4.2)
// ---------------------------------------------------------------------------

/// One candidate run's outcome (a patch attempt at the same goal).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidatePatch {
    pub job_id: String,
    pub diff_hash: String,
    pub changed_files: Vec<String>,
    /// Passed its own acceptance oracle (build+test).
    pub oracle_passed: bool,
    /// Passed the existing regression suite (the §3.2 regression filter).
    pub regression_clean: bool,
    /// Number of deterministic acceptance oracles passed (rank signal).
    pub oracles_passed: u32,
    /// Diff size in changed lines (smaller is better — a rank tie-break signal).
    pub diff_lines: u32,
    /// Warnings emitted (fewer is better).
    pub warnings: u32,
    pub summary: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeStrategy {
    SelectWinner,
    ThreeWay,
    Structured,
    ManualReview,
    RejectAll,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergeDecision {
    pub winner_job_id: Option<String>,
    /// The candidates the winner beat (for the `merge.selected` event).
    pub beaten: Vec<String>,
    pub strategy: MergeStrategy,
    pub conflicts: Vec<String>,
    /// Human-readable basis (the §4.4.2 `selection_basis`).
    pub basis: String,
    /// True when the oracle signals couldn't separate the leaders → a judge
    /// tie-break is needed (the host runs the LLM judge; the selector flags it).
    pub needs_judge: bool,
    pub judge_leaders: Vec<String>,
}

pub struct TournamentSelector;

impl TournamentSelector {
    /// Oracle-first selection (P4). 1) regression filter, 2) rank by oracle
    /// signals (oracles passed ↓, warnings ↑, diff size ↑ — all deterministic),
    /// 3) flag a judge tie-break only among oracle-equivalent leaders.
    pub fn select(candidates: &[CandidatePatch]) -> MergeDecision {
        // 1. Regression filter: a fix that breaks the existing suite is out.
        let mut viable: Vec<&CandidatePatch> = candidates
            .iter()
            .filter(|c| c.oracle_passed && c.regression_clean)
            .collect();
        if viable.is_empty() {
            return MergeDecision {
                winner_job_id: None,
                beaten: Vec::new(),
                strategy: MergeStrategy::RejectAll,
                conflicts: Vec::new(),
                basis: "no candidate passed deterministic oracles + regression".to_string(),
                needs_judge: false,
                judge_leaders: Vec::new(),
            };
        }
        // 2. Oracle rank (lexicographic, all deterministic): more oracles, fewer
        // warnings, smaller diff, then stable job_id.
        viable.sort_by(|a, b| {
            b.oracles_passed
                .cmp(&a.oracles_passed)
                .then_with(|| a.warnings.cmp(&b.warnings))
                .then_with(|| a.diff_lines.cmp(&b.diff_lines))
                .then_with(|| a.job_id.cmp(&b.job_id))
        });
        let leader = viable[0];
        // 3. Leaders the oracle signals cannot separate (equal on all signals).
        let leaders: Vec<&CandidatePatch> = viable
            .iter()
            .copied()
            .filter(|c| {
                c.oracles_passed == leader.oracles_passed
                    && c.warnings == leader.warnings
                    && c.diff_lines == leader.diff_lines
            })
            .collect();
        let needs_judge = leaders.len() > 1;
        let beaten: Vec<String> = viable.iter().skip(1).map(|c| c.job_id.clone()).collect();
        MergeDecision {
            winner_job_id: Some(leader.job_id.clone()),
            beaten,
            strategy: MergeStrategy::SelectWinner,
            conflicts: Vec::new(),
            basis: format!(
                "oracle-first: {} oracles, {} warnings, {} diff lines",
                leader.oracles_passed, leader.warnings, leader.diff_lines
            ),
            needs_judge,
            judge_leaders: leaders.iter().map(|c| c.job_id.clone()).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// The conflict ladder (§4.4.3) — real 3-way text merge via `similar`
// ---------------------------------------------------------------------------

/// How a file merge was resolved (the `merge.resolved{by}` event basis).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedBy {
    /// One side was unchanged from base → take the other.
    FastForward,
    /// Both sides changed disjoint regions → clean 3-way merge.
    ThreeWay,
    /// Both sides changed the same region differently → conflict.
    Conflict,
}

/// The result of a 3-way merge of one file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileMerge {
    pub path: String,
    pub by: ResolvedBy,
    /// The merged text (with conflict markers if `by == Conflict`).
    pub merged: String,
    pub conflicted: bool,
}

/// A real line-level 3-way merge of one file against a common ancestor, using the
/// `similar` crate to compute each side's hunks. Clean when the two sides touch
/// disjoint line regions; conflicting when they edit the same region differently.
/// This is step 2 of the ladder (step 1, structured/AST merge, is a host concern
/// that uses tree-sitter; step 3 is the LLM resolver run; step 4 is escalate).
pub fn three_way_merge(path: &str, base: &str, ours: &str, theirs: &str) -> FileMerge {
    // Fast paths.
    if ours == theirs {
        return FileMerge {
            path: path.to_string(),
            by: ResolvedBy::FastForward,
            merged: ours.to_string(),
            conflicted: false,
        };
    }
    if ours == base {
        return FileMerge {
            path: path.to_string(),
            by: ResolvedBy::FastForward,
            merged: theirs.to_string(),
            conflicted: false,
        };
    }
    if theirs == base {
        return FileMerge {
            path: path.to_string(),
            by: ResolvedBy::FastForward,
            merged: ours.to_string(),
            conflicted: false,
        };
    }

    let base_lines: Vec<&str> = base.lines().collect();
    let ours_lines: Vec<&str> = ours.lines().collect();
    let theirs_lines: Vec<&str> = theirs.lines().collect();

    // Compute, for each base line index, whether "ours" and "theirs" modified the
    // region around it. We walk base and emit merged output, detecting same-region
    // double edits as conflicts.
    let ours_changes = changed_base_regions(&base_lines, &ours_lines);
    let theirs_changes = changed_base_regions(&base_lines, &theirs_lines);

    // If the changed base-line sets are disjoint, the merge is clean: apply each
    // side's version. Otherwise it conflicts.
    let conflict = !ours_changes.is_disjoint(&theirs_changes);

    if conflict {
        let merged = format!(
            "<<<<<<< ours\n{}\n=======\n{}\n>>>>>>> theirs\n",
            ours.trim_end(),
            theirs.trim_end()
        );
        FileMerge {
            path: path.to_string(),
            by: ResolvedBy::Conflict,
            merged,
            conflicted: true,
        }
    } else {
        // Disjoint edits: a real 3-way reconstruction. Apply theirs' changes onto
        // ours (ours already differs from base only in ours_changes; theirs'
        // changes are in disjoint base regions, so layering is well-defined). We
        // reconstruct by taking, per base region, whichever side changed it.
        let merged = reconstruct_disjoint(&base_lines, ours, theirs, &ours_changes);
        FileMerge {
            path: path.to_string(),
            by: ResolvedBy::ThreeWay,
            merged,
            conflicted: false,
        }
    }
}

/// The set of base line indices that `side` modified or deleted relative to base
/// (an "anchor" set used for disjointness). Computed from the `similar` line diff.
fn changed_base_regions(base: &[&str], side: &[&str]) -> BTreeSet<usize> {
    use similar::{capture_diff_slices, Algorithm, DiffOp};
    let ops = capture_diff_slices(Algorithm::Myers, base, side);
    let mut changed = BTreeSet::new();
    for op in ops {
        match op {
            DiffOp::Equal { .. } => {}
            DiffOp::Delete {
                old_index, old_len, ..
            } => {
                for i in old_index..old_index + old_len {
                    changed.insert(i);
                }
            }
            DiffOp::Replace {
                old_index, old_len, ..
            } => {
                for i in old_index..old_index + old_len {
                    changed.insert(i);
                }
            }
            DiffOp::Insert { old_index, .. } => {
                // Insertion anchors at the base position it precedes.
                changed.insert(old_index);
            }
        }
    }
    changed
}

/// Reconstruct a merged file when the two sides edited disjoint base regions. We
/// rebuild by walking base and, for each region, choosing the side that changed
/// it. Insertions from both sides are preserved in base order.
fn reconstruct_disjoint(
    base: &[&str],
    ours: &str,
    theirs: &str,
    ours_changes: &BTreeSet<usize>,
) -> String {
    use similar::{capture_diff_slices, Algorithm};
    let ours_lines: Vec<&str> = ours.lines().collect();
    let theirs_lines: Vec<&str> = theirs.lines().collect();

    // Build a map base_index -> replacement text from each side.
    // Strategy: take `theirs` as the canvas, then overlay `ours`' changes onto the
    // base regions ours owns. Because the change sets are disjoint, every base
    // region is owned by at most one side; theirs already reflects its own edits,
    // so we only need to splice ours' edits back in. We do this by diffing
    // theirs-vs-base and ours-vs-base in lockstep over base.
    let ours_ops = capture_diff_slices(Algorithm::Myers, base, &ours_lines);
    let theirs_ops = capture_diff_slices(Algorithm::Myers, base, &theirs_lines);

    // Map each base line to its "ours" output (the lines ours produces for it).
    let ours_out = side_output_per_base(base, &ours_lines, &ours_ops);
    let theirs_out = side_output_per_base(base, &theirs_lines, &theirs_ops);

    let mut merged_lines: Vec<String> = Vec::new();
    for (i, _) in base.iter().enumerate() {
        if ours_changes.contains(&i) {
            merged_lines.extend(ours_out.get(&i).cloned().unwrap_or_default());
        } else {
            merged_lines.extend(theirs_out.get(&i).cloned().unwrap_or_default());
        }
    }
    // Trailing insertions anchored past the last base line.
    let end = base.len();
    if ours_changes.contains(&end) {
        merged_lines.extend(ours_out.get(&end).cloned().unwrap_or_default());
    } else {
        merged_lines.extend(theirs_out.get(&end).cloned().unwrap_or_default());
    }
    let mut s = merged_lines.join("\n");
    if !s.is_empty() {
        s.push('\n');
    }
    s
}

/// For each base line index (and a synthetic `base.len()` slot for trailing
/// inserts), the lines a side emits there.
fn side_output_per_base(
    base: &[&str],
    side: &[&str],
    ops: &[similar::DiffOp],
) -> std::collections::BTreeMap<usize, Vec<String>> {
    use similar::DiffOp;
    let mut out: std::collections::BTreeMap<usize, Vec<String>> = std::collections::BTreeMap::new();
    for op in ops {
        match *op {
            DiffOp::Equal {
                old_index,
                new_index,
                len,
            } => {
                for k in 0..len {
                    out.entry(old_index + k)
                        .or_default()
                        .push(side[new_index + k].to_string());
                }
            }
            DiffOp::Replace {
                old_index,
                old_len,
                new_index,
                new_len,
            } => {
                let repl: Vec<String> = (0..new_len)
                    .map(|k| side[new_index + k].to_string())
                    .collect();
                out.entry(old_index).or_default().extend(repl);
                // Mark the rest of the replaced base region as consumed (no output).
                for k in 1..old_len {
                    out.entry(old_index + k).or_default();
                }
            }
            DiffOp::Delete {
                old_index, old_len, ..
            } => {
                for k in 0..old_len {
                    out.entry(old_index + k).or_default();
                }
            }
            DiffOp::Insert {
                old_index,
                new_index,
                new_len,
            } => {
                let ins: Vec<String> = (0..new_len)
                    .map(|k| side[new_index + k].to_string())
                    .collect();
                out.entry(old_index).or_default().extend(ins);
                let _ = base; // base only used for bounds reasoning above.
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The integration funnel (§4.4.1)
// ---------------------------------------------------------------------------

/// The outcome of integrating N fan-out runs onto an integration branch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegrationResult {
    pub adopted: Vec<String>,
    pub dropped: Vec<String>,
    pub conflicts: Vec<String>,
    /// Whether the full suite was green after integration (the promote gate).
    pub suite_green: bool,
}

/// Integrate a set of fan-out candidates whose footprints have been classified.
/// Disjoint-footprint candidates are adopted in order; any pair that conflicts on
/// a file is recorded as a conflict for the ladder/escalation. This is the
/// *decision* funnel — the host performs the actual git merges + suite run and
/// supplies `suite_green`.
pub fn integrate(
    candidates: &[CandidatePatch],
    footprints: &[Footprint],
    suite_green: bool,
) -> IntegrationResult {
    let plan = plan_footprints(footprints);
    let conflict_files: BTreeSet<String> = plan
        .overlaps
        .iter()
        .flat_map(|(a, b)| {
            let fa = footprints.iter().find(|f| &f.job_id == a);
            let fb = footprints.iter().find(|f| &f.job_id == b);
            match (fa, fb) {
                (Some(fa), Some(fb)) => fa.files.intersection(&fb.files).cloned().collect(),
                _ => Vec::new(),
            }
        })
        .collect();

    let mut adopted = Vec::new();
    let mut dropped = Vec::new();
    let conflicted_jobs: BTreeSet<String> = plan
        .overlaps
        .iter()
        .flat_map(|(a, b)| [a.clone(), b.clone()])
        .collect();

    for c in candidates {
        if !c.oracle_passed || !c.regression_clean {
            dropped.push(c.job_id.clone());
        } else if conflicted_jobs.contains(&c.job_id) {
            // Overlapping footprint: held for the conflict ladder, not silently
            // adopted (no silent wrong merge, P3).
            dropped.push(c.job_id.clone());
        } else {
            adopted.push(c.job_id.clone());
        }
    }

    IntegrationResult {
        adopted,
        dropped,
        conflicts: conflict_files.into_iter().collect(),
        suite_green,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: &str, ok: bool, oracles: u32, warns: u32, diff: u32) -> CandidatePatch {
        CandidatePatch {
            job_id: id.to_string(),
            diff_hash: format!("h_{id}"),
            changed_files: vec![format!("{id}.rs")],
            oracle_passed: ok,
            regression_clean: ok,
            oracles_passed: oracles,
            diff_lines: diff,
            warnings: warns,
            summary: String::new(),
        }
    }

    #[test]
    fn selector_is_oracle_first_and_filters_regressions() {
        let cands = vec![
            cand("a", true, 3, 0, 50),
            cand("b", false, 5, 0, 10), // more oracles but fails regression → out
            cand("c", true, 4, 1, 40),
        ];
        let d = TournamentSelector::select(&cands);
        // c has more oracles passed than a (4 > 3) → c wins; b filtered out.
        assert_eq!(d.winner_job_id, Some("c".to_string()));
        assert!(d.beaten.contains(&"a".to_string()));
        assert!(!d.needs_judge);
    }

    #[test]
    fn selector_flags_judge_on_oracle_equivalent_leaders() {
        let cands = vec![cand("a", true, 3, 0, 20), cand("b", true, 3, 0, 20)];
        let d = TournamentSelector::select(&cands);
        assert!(d.needs_judge);
        assert_eq!(d.judge_leaders.len(), 2);
    }

    #[test]
    fn selector_rejects_when_nothing_viable() {
        let cands = vec![cand("a", false, 1, 0, 5)];
        let d = TournamentSelector::select(&cands);
        assert_eq!(d.strategy, MergeStrategy::RejectAll);
        assert!(d.winner_job_id.is_none());
    }

    #[test]
    fn footprints_partition_disjoint_and_record_overlaps() {
        let fps = vec![
            Footprint::new("a", ["src/x.rs".to_string()]),
            Footprint::new("b", ["src/y.rs".to_string()]),
            Footprint::new("c", ["src/x.rs".to_string()]), // overlaps a
        ];
        let plan = plan_footprints(&fps);
        assert!(plan.overlaps.contains(&("a".to_string(), "c".to_string())));
        // a and b are disjoint → same parallel group; c opens a new one.
        assert!(plan
            .parallel_groups
            .iter()
            .any(|g| g.contains(&"a".to_string()) && g.contains(&"b".to_string())));
    }

    #[test]
    fn three_way_merge_fast_forwards_when_one_side_unchanged() {
        let base = "line1\nline2\n";
        let ours = "line1\nline2\n";
        let theirs = "line1\nCHANGED\n";
        let m = three_way_merge("f.txt", base, ours, theirs);
        assert!(!m.conflicted);
        assert_eq!(m.by, ResolvedBy::FastForward);
        assert_eq!(m.merged, theirs);
    }

    #[test]
    fn three_way_merge_combines_disjoint_edits() {
        let base = "a\nb\nc\nd\n";
        let ours = "AA\nb\nc\nd\n"; // edited line 0
        let theirs = "a\nb\nc\nDD\n"; // edited line 3
        let m = three_way_merge("f.txt", base, ours, theirs);
        assert!(!m.conflicted, "disjoint edits must merge cleanly");
        assert!(m.merged.contains("AA"));
        assert!(m.merged.contains("DD"));
    }

    #[test]
    fn three_way_merge_conflicts_on_same_region() {
        let base = "a\nb\nc\n";
        let ours = "a\nOURS\nc\n";
        let theirs = "a\nTHEIRS\nc\n";
        let m = three_way_merge("f.txt", base, ours, theirs);
        assert!(m.conflicted);
        assert_eq!(m.by, ResolvedBy::Conflict);
        assert!(m.merged.contains("<<<<<<<"));
    }

    #[test]
    fn integrate_holds_overlapping_footprints_for_the_ladder() {
        let cands = vec![cand("a", true, 2, 0, 5), cand("c", true, 2, 0, 5)];
        let fps = vec![
            Footprint::new("a", ["shared.rs".to_string()]),
            Footprint::new("c", ["shared.rs".to_string()]),
        ];
        let r = integrate(&cands, &fps, true);
        // Both touch shared.rs → neither silently adopted.
        assert!(r.adopted.is_empty());
        assert!(r.conflicts.contains(&"shared.rs".to_string()));
    }
}
