use hide_core::ids::ModelId;
use hide_core::ids::RoleId;
use hide_core::runtime::{
    ModelArchitecture, ModelDescriptor, ModelRole, ProviderCaps, RolePurpose, SamplerProfile,
};
use parking_lot::RwLock;
use std::collections::BTreeMap;

#[derive(Default)]
pub struct RoleRegistry {
    roles: RwLock<BTreeMap<RoleId, ModelRole>>,
}

impl RoleRegistry {
    pub fn with_default_local_roles() -> Self {
        let registry = Self::default();
        registry.register_default_local_roles();
        registry
    }

    pub fn register_default_local_roles(&self) {
        for role in default_hawking_local_roles() {
            self.register(role);
        }
    }

    pub fn register(&self, role: ModelRole) {
        self.roles.write().insert(role.id.clone(), role);
    }

    pub fn get(&self, id: &RoleId) -> Option<ModelRole> {
        self.roles.read().get(id).cloned()
    }

    pub fn by_purpose(&self, purpose: RolePurpose) -> Vec<ModelRole> {
        self.roles
            .read()
            .values()
            .filter(|role| role.purpose == purpose)
            .cloned()
            .collect()
    }

    pub fn all(&self) -> Vec<ModelRole> {
        self.roles.read().values().cloned().collect()
    }
}

pub fn default_hawking_local_roles() -> Vec<ModelRole> {
    vec![
        local_role(
            "hawking-fast-draft",
            RolePurpose::FastDraft,
            16_384,
            4_096,
            ProviderCaps::hawking_local_shell_today(),
            SamplerProfile {
                temperature: 0.2,
                top_k: Some(40),
                top_p: Some(0.95),
                repetition_penalty: None,
                seed: None,
                deterministic: false,
            },
        ),
        local_role(
            "hawking-hero-coder",
            RolePurpose::HeroCoder,
            32_768,
            8_192,
            ProviderCaps::hawking_local_shell_today(),
            SamplerProfile::deterministic_edit(),
        ),
        local_role(
            "hawking-embedder",
            RolePurpose::Embedder,
            8_192,
            1_024,
            ProviderCaps {
                streaming: false,
                embeddings: true,
                grammar: false,
                raw_logits: false,
                logprobs: false,
                lora: false,
                kv_handles: false,
                native_tokens_endpoint: false,
            },
            SamplerProfile::deterministic_edit(),
        ),
        local_role(
            "hawking-tool-planner",
            RolePurpose::ToolPlanner,
            16_384,
            4_096,
            ProviderCaps {
                grammar: true,
                ..ProviderCaps::hawking_local_shell_today()
            },
            SamplerProfile::deterministic_edit(),
        ),
    ]
}

fn local_role(
    name: &str,
    purpose: RolePurpose,
    context_tokens: usize,
    footprint_mb: u64,
    caps: ProviderCaps,
    default_sampler: SamplerProfile,
) -> ModelRole {
    ModelRole {
        id: RoleId::new(),
        name: name.to_string(),
        purpose,
        model: ModelDescriptor {
            id: ModelId::new(),
            name: name.to_string(),
            architecture: ModelArchitecture::Transformer,
            context_tokens,
            tokenizer_signature: "hawking-local".to_string(),
            footprint_mb,
        },
        caps,
        default_sampler,
        endpoint: None,
        cost: None,
        escalates_to: None,
        metadata: BTreeMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_contains_core_runtime_roles() {
        let registry = RoleRegistry::with_default_local_roles();
        assert!(!registry.by_purpose(RolePurpose::FastDraft).is_empty());
        assert!(!registry.by_purpose(RolePurpose::HeroCoder).is_empty());
        assert!(!registry.by_purpose(RolePurpose::Embedder).is_empty());
        assert!(registry
            .by_purpose(RolePurpose::ToolPlanner)
            .iter()
            .any(|role| role.caps.grammar));
    }
}
