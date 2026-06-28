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
    let latency_cutoff: Option<u32> = policy.latency_outlier_p95_mult.and_then(|mult| {
        p95_latency(records).map(|p95| ((p95 as f32) * mult).round() as u32)
    });

    // ── Rule 5: recency-weighting. Process newest-first so the cap keeps the
    //    most recent examples (the user's current style), and apply it across
    //    both positives and the rejected pool. ────────────────────────────────
    let mut ordered: Vec<&PersonalizationRecord> = records.iter().collect();
    ordered.sort_by(|a, b| b.observed_at_us.cmp(&a.observed_at_us));

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
            Outcome::Modified {
                edit_distance_chars,
            } => {
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
        if let Some(rejected) = rejected_by_prompt
            .iter()
            .find(|r| r.prompt_hash == accepted.prompt_hash && !r.diff_proposed.is_empty())
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

    CuratedDataset {
        sft,
        preferences,
        held_out,
    }
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
        let records = vec![
            rec_at("p1", "+hello", 10, 1),
            rec_at("p2", "+world", 10, 2),
        ];
        let dataset = curate(&records, &CurationPolicy::default());
        // 2 positives, 10% held-out rounds to 0 → both in train.
        assert_eq!(dataset.sft.len(), 2);
        assert_eq!(dataset.held_out.len(), 0);
    }

    #[test]
    fn latency_p95x3_outlier_dropped() {
        // 20 fast records + 1 timeout. p95 of the fast pop is ~10; ×3 = 30; the
        // 5000ms record is dropped.
        let mut records: Vec<_> = (0..20)
            .map(|i| rec_at(&format!("p{i}"), &format!("+d{i}"), 10, i as u64))
            .collect();
        records.push(rec_at("timeout", "+slow", 5000, 99));
        let dataset = curate(&records, &CurationPolicy::default());
        assert!(
            dataset.sft.iter().all(|e| e.target_diff != "+slow"),
            "timeout artifact must be dropped by the p95x3 rule"
        );
    }

    #[test]
    fn recency_cap_keeps_newest() {
        let policy = CurationPolicy {
            max_records: 1,
            held_out_frac: 0.0,
            ..Default::default()
        };
        let records = vec![
            rec_at("old", "+old", 10, 1),
            rec_at("new", "+new", 10, 100),
        ];
        let dataset = curate(&records, &policy);
        assert_eq!(dataset.sft.len(), 1);
        assert_eq!(dataset.sft[0].target_diff, "+new");
    }

    #[test]
    fn dpo_pairs_on_matching_prompt() {
        let mut accepted = PersonalizationRecord::accepted(TaskClass::EditCode, "same", "+good");
        accepted.observed_at_us = 2;
        let rejected =
            PersonalizationRecord::rejected(TaskClass::EditCode, "same", "+bad", None);
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
