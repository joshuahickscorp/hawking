use hide_core::ids::PluginId;
use hide_core::types::Provenance;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillRecord {
    pub id: PluginId,
    pub name: String,
    pub description: String,
    pub body: String,
    pub tests: Vec<String>,
    pub provenance: Provenance,
    pub success_count: u32,
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
