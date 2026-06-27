use crate::records::{Outcome, PersonalizationRecord};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurationPolicy {
    pub max_records: usize,
    pub max_rewrite_ratio: f32,
    pub drop_latency_ms_above: Option<u32>,
}

impl Default for CurationPolicy {
    fn default() -> Self {
        Self {
            max_records: 10_000,
            max_rewrite_ratio: 0.8,
            drop_latency_ms_above: None,
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
    pub prompt_hash: String,
    pub context_fingerprint: String,
    pub target_diff: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreferenceExample {
    pub prompt_hash: String,
    pub chosen_diff: String,
    pub rejected_diff: String,
}

pub fn curate(records: &[PersonalizationRecord], policy: &CurationPolicy) -> CuratedDataset {
    let mut seen = BTreeSet::new();
    let mut sft = Vec::new();
    let mut rejected_by_prompt = Vec::<&PersonalizationRecord>::new();
    for record in records {
        if policy
            .drop_latency_ms_above
            .map_or(false, |max| record.latency_ms > max)
        {
            continue;
        }
        match &record.outcome {
            Outcome::Accepted => {
                if !record.diff_accepted.is_empty()
                    && seen.insert(record.diff_accepted.clone())
                    && sft.len() < policy.max_records
                {
                    sft.push(SftExample {
                        prompt_hash: record.prompt_hash.clone(),
                        context_fingerprint: record.context_fingerprint.clone(),
                        target_diff: record.diff_accepted.clone(),
                    });
                }
            }
            Outcome::Modified {
                edit_distance_chars,
            } => {
                let proposed_len = record.diff_proposed.chars().count().max(1) as f32;
                let ratio = *edit_distance_chars as f32 / proposed_len;
                if ratio <= policy.max_rewrite_ratio
                    && !record.diff_accepted.is_empty()
                    && seen.insert(record.diff_accepted.clone())
                    && sft.len() < policy.max_records
                {
                    sft.push(SftExample {
                        prompt_hash: record.prompt_hash.clone(),
                        context_fingerprint: record.context_fingerprint.clone(),
                        target_diff: record.diff_accepted.clone(),
                    });
                }
            }
            Outcome::Rejected => rejected_by_prompt.push(record),
            Outcome::Abandoned => {}
        }
    }
    let mut preferences = Vec::new();
    for accepted in &sft {
        if let Some(rejected) = rejected_by_prompt
            .iter()
            .find(|r| r.prompt_hash == accepted.prompt_hash)
        {
            preferences.push(PreferenceExample {
                prompt_hash: accepted.prompt_hash.clone(),
                chosen_diff: accepted.target_diff.clone(),
                rejected_diff: rejected.diff_proposed.clone(),
            });
        }
    }
    let split = sft.len().saturating_sub((sft.len() / 10).max(1));
    let held_out = sft[split..].to_vec();
    let sft = sft[..split].to_vec();
    CuratedDataset {
        sft,
        preferences,
        held_out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::records::{PersonalizationRecord, TaskClass};

    #[test]
    fn curation_keeps_accepted_diffs() {
        let records = vec![
            PersonalizationRecord::accepted(TaskClass::EditCode, "p1", "+hello"),
            PersonalizationRecord::accepted(TaskClass::EditCode, "p2", "+world"),
        ];
        let dataset = curate(&records, &CurationPolicy::default());
        assert_eq!(dataset.sft.len(), 1);
        assert_eq!(dataset.held_out.len(), 1);
    }
}
