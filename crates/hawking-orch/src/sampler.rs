use hide_core::runtime::SamplerProfile;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplerCatalog {
    pub edit: SamplerProfile,
    pub planning: SamplerProfile,
    pub brainstorm: SamplerProfile,
}

impl Default for SamplerCatalog {
    fn default() -> Self {
        Self {
            edit: SamplerProfile::deterministic_edit(),
            planning: SamplerProfile {
                temperature: 0.2,
                top_k: Some(40),
                top_p: Some(0.9),
                repetition_penalty: Some(1.05),
                seed: Some(0),
                deterministic: false,
            },
            brainstorm: SamplerProfile {
                temperature: 0.8,
                top_k: Some(80),
                top_p: Some(0.95),
                repetition_penalty: Some(1.05),
                seed: None,
                deterministic: false,
            },
        }
    }
}
