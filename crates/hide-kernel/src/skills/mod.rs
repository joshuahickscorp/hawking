//! The persistent skill library (bible ch.02 §4.11 / Appendix A.6).
//!
//! Only EXECUTION-VALIDATED solutions become skills (Voyager): a skill is
//! captured *on success* (its oracles passed), retrieved by recency / importance
//! / relevance, and promoted or decayed by its track record.

use hide_core::ids::PluginId;
use hide_core::types::Provenance;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillKind {
    Procedure,
    Snippet,
    Recipe,
    Macro,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillRecord {
    pub id: PluginId,
    pub name: String,
    pub description: String,
    #[serde(default = "default_kind")]
    pub kind: SkillKind,
    /// When to apply this skill (a query/trigger string).
    #[serde(default)]
    pub trigger: String,
    pub body: String,
    /// Validation track record (A.6 `validation`).
    #[serde(default)]
    pub success_count: u32,
    #[serde(default)]
    pub fail_count: u32,
    /// Importance ∈ [0,1] (drives retrieval ranking + decay).
    #[serde(default = "default_importance")]
    pub importance: f32,
    #[serde(default)]
    pub access_count: u32,
    pub provenance: Provenance,
}

fn default_kind() -> SkillKind {
    SkillKind::Recipe
}
fn default_importance() -> f32 {
    0.5
}

impl SkillRecord {
    pub fn new(name: impl Into<String>, body: impl Into<String>, trigger: impl Into<String>) -> Self {
        Self {
            id: PluginId::new(),
            name: name.into(),
            description: String::new(),
            kind: SkillKind::Recipe,
            trigger: trigger.into(),
            body: body.into(),
            success_count: 1,
            fail_count: 0,
            importance: default_importance(),
            access_count: 0,
            provenance: Provenance::trusted("skill-capture"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillQuery {
    pub text: String,
    pub top_k: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedSkill {
    pub skill: SkillRecord,
    pub score: f32,
}

/// An in-memory skill store (the `.hide/memory/procedural/` file store is the
/// durable backing; this is the runtime index + the capture/retrieve/curate
/// logic). Retrieval ranks by lexical relevance × importance × success rate.
#[derive(Default)]
pub struct SkillStore {
    skills: BTreeMap<String, SkillRecord>,
}

impl SkillStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Capture-on-success: store (or reinforce) a validated skill. Re-capturing
    /// an existing skill by name increments its success count + importance.
    pub fn capture(&mut self, skill: SkillRecord) {
        match self.skills.get_mut(&skill.name) {
            Some(existing) => {
                existing.success_count += 1;
                existing.importance = (existing.importance + 0.1).min(1.0);
                existing.body = skill.body;
            }
            None => {
                self.skills.insert(skill.name.clone(), skill);
            }
        }
    }

    /// Record a failed application — decays importance; prune when it collapses.
    pub fn record_failure(&mut self, name: &str) {
        if let Some(s) = self.skills.get_mut(name) {
            s.fail_count += 1;
            s.importance = (s.importance - 0.15).max(0.0);
        }
        // Decay below the floor → forget the skill.
        if self
            .skills
            .get(name)
            .map(|s| s.importance <= 0.0 && s.fail_count > s.success_count)
            .unwrap_or(false)
        {
            self.skills.remove(name);
        }
    }

    /// Promote a skill (e.g. when a lesson proved a general recipe) by bumping
    /// its importance toward the pin ceiling.
    pub fn promote(&mut self, name: &str) {
        if let Some(s) = self.skills.get_mut(name) {
            s.importance = (s.importance + 0.25).min(1.0);
        }
    }

    /// Retrieve the top-k skills for a query by relevance × importance × success.
    pub fn retrieve(&mut self, query: &SkillQuery) -> Vec<RankedSkill> {
        let needle = query.text.to_lowercase();
        let mut ranked: Vec<RankedSkill> = self
            .skills
            .values()
            .map(|s| {
                let hay = format!("{} {} {}", s.name, s.description, s.trigger).to_lowercase();
                let relevance = lexical_overlap(&needle, &hay);
                let success_rate = if s.success_count + s.fail_count == 0 {
                    0.5
                } else {
                    s.success_count as f32 / (s.success_count + s.fail_count) as f32
                };
                let score = relevance * s.importance.max(0.05) * success_rate;
                RankedSkill {
                    skill: s.clone(),
                    score,
                }
            })
            .filter(|r| r.score > 0.0)
            .collect();
        ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        ranked.truncate(query.top_k);
        // Mark accesses (recency signal) on the returned skills.
        for r in &ranked {
            if let Some(s) = self.skills.get_mut(&r.skill.name) {
                s.access_count += 1;
            }
        }
        ranked
    }
}

fn lexical_overlap(query: &str, text: &str) -> f32 {
    let q: std::collections::BTreeSet<&str> = query.split_whitespace().collect();
    if q.is_empty() {
        return 0.0;
    }
    let hits = q.iter().filter(|w| text.contains(**w)).count();
    hits as f32 / q.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_reinforces_and_retrieve_ranks() {
        let mut store = SkillStore::new();
        store.capture(SkillRecord::new(
            "add-route",
            "register in router.rs::build",
            "add a route in this repo",
        ));
        store.capture(SkillRecord::new("add-route", "register in router.rs::build", "route"));
        // Reinforced: success_count went 1 → 2.
        let got = store.retrieve(&SkillQuery {
            text: "add a route".to_string(),
            top_k: 5,
        });
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].skill.success_count, 2);
    }

    #[test]
    fn decay_forgets_failing_skill() {
        let mut store = SkillStore::new();
        store.capture(SkillRecord::new("flaky", "body", "trigger"));
        for _ in 0..5 {
            store.record_failure("flaky");
        }
        assert!(store.is_empty(), "a repeatedly-failing skill is forgotten");
    }
}
