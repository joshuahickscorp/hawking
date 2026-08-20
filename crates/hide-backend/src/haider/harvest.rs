//! Deterministic harvest (section 8).
//!
//! Do NOT spend a giant fourth model call summarizing three giant answers.
//! Collect, normalize, deduplicate, detect agreements/contradictions, identify
//! changed files / test results / proposed actions, and hand the parent only the
//! compressed disagreement/frontier packet.

use std::collections::{BTreeMap, BTreeSet};

/// A structured test result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
}

/// Raw output from one lane.
#[derive(Clone, Debug, Default)]
pub struct LaneOutput {
    pub lane: String,
    pub role: String,
    pub text: String,
    pub changed_files: BTreeSet<String>,
    pub test_results: Vec<TestResult>,
    pub proposed_actions: Vec<String>,
}

/// A detected contradiction between lanes.
#[derive(Clone, Debug)]
pub struct Contradiction {
    pub topic: String,
    pub a: String,
    pub b: String,
}

/// The compressed packet the parent receives.
#[derive(Clone, Debug, Default)]
pub struct FrontierPacket {
    pub agreements: Vec<String>,
    pub contradictions: Vec<Contradiction>,
    pub changed_files: BTreeSet<String>,
    pub test_results: Vec<TestResult>,
    pub proposed_actions: Vec<String>,
}

/// The full harvest result.
#[derive(Clone, Debug, Default)]
pub struct Harvest {
    pub packet: FrontierPacket,
    pub per_lane_tokens: BTreeMap<String, usize>,
}

/// Deterministic harvest: no model call.
pub fn harvest(outputs: &[LaneOutput]) -> Harvest {
    let mut packet = FrontierPacket::default();
    let mut per_lane_tokens: BTreeMap<String, usize> = BTreeMap::new();

    // Collect + normalize + dedup.
    let mut all_files: BTreeSet<String> = BTreeSet::new();
    let mut all_tests: BTreeMap<String, bool> = BTreeMap::new();
    let mut all_actions: BTreeSet<String> = BTreeSet::new();

    for o in outputs {
        per_lane_tokens.insert(o.lane.clone(), o.text.split_whitespace().count());
        all_files.extend(o.changed_files.iter().cloned());
        for t in &o.test_results {
            all_tests.insert(t.name.clone(), t.passed);
        }
        all_actions.extend(o.proposed_actions.iter().cloned());
    }

    packet.changed_files = all_files;
    packet.test_results = all_tests
        .into_iter()
        .map(|(name, passed)| TestResult { name, passed })
        .collect();
    packet.proposed_actions = all_actions.into_iter().collect();

    // Detect agreements: actions that appear in >1 lane.
    let mut action_counts: BTreeMap<String, usize> = BTreeMap::new();
    for o in outputs {
        for a in &o.proposed_actions {
            *action_counts.entry(a.clone()).or_insert(0) += 1;
        }
    }
    packet.agreements = action_counts
        .into_iter()
        .filter(|(_, c)| *c > 1)
        .map(|(a, _)| a)
        .collect();

    // Detect contradictions: two lanes touch the same file but propose different
    // actions -> surface it for the parent.
    for i in 0..outputs.len() {
        for j in (i + 1)..outputs.len() {
            let (a, b) = (&outputs[i], &outputs[j]);
            if a.proposed_actions == b.proposed_actions {
                continue;
            }
            for fa in &a.changed_files {
                if b.changed_files.contains(fa) {
                    packet.contradictions.push(Contradiction {
                        topic: fa.clone(),
                        a: a.lane.clone(),
                        b: b.lane.clone(),
                    });
                }
            }
        }
    }

    Harvest {
        packet,
        per_lane_tokens,
    }
}
