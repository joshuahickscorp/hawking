//! Spine B — the RecallOracle: the measured "remember more, not half" guarantee.
//!
//! Compaction is only allowed to stand if it preserves recall. After a compaction
//! pass we re-ask a set of known facts ("needles") against the post-compaction
//! context and compute recall@k. If recall regresses past the threshold — or too
//! many importance-weighted tokens were dropped, or test coverage regressed, or
//! the compaction chain recursed past the depth cap — the compaction is rolled
//! back to the richer context. This module is the pure decision core (no I/O, no
//! model), so the thresholds are unit-tested and can never silently drift.

use serde::{Deserialize, Serialize};

/// Recall@k floor: below this the compaction is reverted.
pub const RECALL_FLOOR: f32 = 0.85;
/// Importance-weighted dropped-token ceiling: above this we revert even if the
/// needle recall looks fine (we dropped too much that mattered).
pub const DROPPED_IMPORTANT_CEIL: f32 = 0.10;
/// Recursion depth cap on a compaction chain; deeper => auto-revert to original.
pub const MAX_COMPACT_DEPTH: u8 = 2;

/// A known fact pinned from the PRE-compaction context, re-asked afterwards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecallProbe {
    pub id: String,
    /// The salient substring that must survive compaction (a file path, a symbol,
    /// a decision, a constraint, a test verdict).
    pub needle: String,
}

/// Fraction of needles still recoverable in `post_context`, 0..1.
///
/// A pragmatic, deterministic recall measure (no embedding model in the hot
/// path): a needle counts as recalled if its normalized form appears in the
/// post-compaction text. Empty probe set => 1.0 (nothing to lose).
pub fn recall_at_k(probes: &[RecallProbe], post_context: &str) -> f32 {
    if probes.is_empty() {
        return 1.0;
    }
    let hay = post_context.to_lowercase();
    let hit = probes
        .iter()
        .filter(|p| {
            let n = p.needle.trim().to_lowercase();
            !n.is_empty() && hay.contains(&n)
        })
        .count();
    hit as f32 / probes.len() as f32
}

/// Derive recall needles from a span's ORIGINAL text: the distinctive non-trivial
/// lines a faithful compaction must preserve. Deterministic, pure. Used by the
/// compiler to measure a degrade against the original before letting it stand.
pub fn needles_from(original: &str, max: usize) -> Vec<RecallProbe> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for line in original.lines() {
        let t = line.trim();
        // Skip trivia (short lines, lone braces) — keep lines with real content.
        if t.len() < 12 || !t.chars().any(|c| c.is_alphanumeric()) {
            continue;
        }
        if seen.insert(t.to_string()) {
            out.push(RecallProbe {
                id: format!("n{}", out.len()),
                needle: t.to_string(),
            });
            if out.len() >= max {
                break;
            }
        }
    }
    out
}

/// The verdict on whether a compaction may stand.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RollbackDecision {
    pub should_rollback: bool,
    pub recall: f32,
    pub reason: &'static str,
}

/// Decide whether a compaction must be rolled back. Pure; order of checks is the
/// order of severity so the `reason` is the most specific failure.
pub fn decide_rollback(
    recall: f32,
    dropped_important_frac: f32,
    coverage_regressed: bool,
    depth: u8,
) -> RollbackDecision {
    if depth > MAX_COMPACT_DEPTH {
        return RollbackDecision {
            should_rollback: true,
            recall,
            reason: "depth cap exceeded",
        };
    }
    if coverage_regressed {
        return RollbackDecision {
            should_rollback: true,
            recall,
            reason: "test coverage regressed",
        };
    }
    if recall < RECALL_FLOOR {
        return RollbackDecision {
            should_rollback: true,
            recall,
            reason: "recall below floor",
        };
    }
    if dropped_important_frac > DROPPED_IMPORTANT_CEIL {
        return RollbackDecision {
            should_rollback: true,
            recall,
            reason: "dropped too much that mattered",
        };
    }
    RollbackDecision {
        should_rollback: false,
        recall,
        reason: "ok",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(id: &str, needle: &str) -> RecallProbe {
        RecallProbe {
            id: id.into(),
            needle: needle.into(),
        }
    }

    #[test]
    fn recall_counts_surviving_needles() {
        let probes = vec![
            probe("f1", "guard.rs"),
            probe("d1", "drop permit past retry"),
            probe("c1", "never block on the semaphore"),
        ];
        // A summary that keeps two of three facts.
        let post = "Edited GUARD.RS to drop permit past retry; see notes.";
        let r = recall_at_k(&probes, post);
        assert!((r - 2.0 / 3.0).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn empty_probes_is_full_recall() {
        assert_eq!(recall_at_k(&[], "anything"), 1.0);
    }

    #[test]
    fn needles_skip_trivia_and_dedupe() {
        let src = "fn acquire() {\n        drop(permit); // release the slot back to the pool\n}\n        drop(permit); // release the slot back to the pool\n";
        let n = needles_from(src, 8);
        // The lone `}` / short lines are skipped; the duplicate content line is kept once.
        assert_eq!(n.len(), 2, "deduped + trivia-filtered: {n:?}");
        assert!(n.iter().any(|p| p.needle.contains("release the slot")));
        assert!(n.iter().all(|p| p.needle != "}"));
    }

    #[test]
    fn rollback_fires_on_each_condition() {
        assert!(
            decide_rollback(0.99, 0.0, false, 3).should_rollback,
            "depth"
        );
        assert!(
            decide_rollback(0.99, 0.0, true, 1).should_rollback,
            "coverage"
        );
        assert!(
            decide_rollback(0.50, 0.0, false, 1).should_rollback,
            "recall"
        );
        assert!(
            decide_rollback(0.99, 0.5, false, 1).should_rollback,
            "dropped"
        );
        assert!(
            !decide_rollback(0.99, 0.0, false, 1).should_rollback,
            "clean keeps"
        );
    }
}
