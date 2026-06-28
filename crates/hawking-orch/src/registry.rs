use hide_core::error::{HideError, Result};
use hide_core::ids::ModelId;
use hide_core::ids::RoleId;
use hide_core::runtime::{
    ModelArchitecture, ModelDescriptor, ModelRole, ProviderCaps, RolePurpose, SamplerProfile,
};
use parking_lot::RwLock;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

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

    pub fn is_empty(&self) -> bool {
        self.roles.read().is_empty()
    }

    /// Resolve a role by its human name (the key used in `roles.toml`), since
    /// `escalates_to`/`draft_for` reference roles by name in the file.
    pub fn by_name(&self, name: &str) -> Option<ModelRole> {
        self.roles
            .read()
            .values()
            .find(|r| r.name == name)
            .cloned()
    }

    /// Load roles from a `.hide/roles.toml` (or any TOML) file if it exists,
    /// else fall back to the built-in defaults. `escalates_to` names are
    /// resolved to the registered roles' ids in a second pass.
    pub fn load_from_file_or_default(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::with_default_local_roles());
        }
        let text = std::fs::read_to_string(path)?;
        Self::from_roles_toml(&text)
    }

    /// Parse a `roles.toml` document into a registry (the §4.3 schema).
    pub fn from_roles_toml(text: &str) -> Result<Self> {
        let file: RolesFile = toml::from_str(text)
            .map_err(|e| HideError::Config(format!("invalid roles.toml: {e}")))?;
        let registry = Self::default();

        // First pass: register every role with a stable id == its name, so the
        // escalates_to wiring can resolve by name.
        for (name, spec) in &file.roles {
            registry.register(spec.to_model_role(name));
        }
        // Second pass: resolve escalates_to (a role *name*) to that role's id.
        let resolved: Vec<ModelRole> = registry
            .all()
            .into_iter()
            .map(|mut role| {
                if let Some(target_name) = file
                    .roles
                    .get(&role.name)
                    .and_then(|s| s.escalates_to.clone())
                {
                    role.escalates_to = registry.by_name(&target_name).map(|r| r.id);
                }
                role
            })
            .collect();
        for role in resolved {
            registry.register(role);
        }
        Ok(registry)
    }
}

// ---- roles.toml schema (ch.06 §4.3) ----------------------------------------

#[derive(Debug, Deserialize)]
struct RolesFile {
    #[serde(default)]
    roles: BTreeMap<String, RoleSpec>,
}

#[derive(Debug, Deserialize)]
struct RoleSpec {
    role_kind: String,
    model: ModelSpec,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    default_sampler: Option<String>,
    #[serde(default)]
    caps: CapsSpec,
    #[serde(default)]
    cost: Option<f32>,
    #[serde(default)]
    escalates_to: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelSpec {
    id: String,
    #[serde(default = "default_arch")]
    arch: String,
    #[serde(default = "default_ctx")]
    ctx_len_native: usize,
    #[serde(default)]
    tokenizer_sig: Option<String>,
    #[serde(default)]
    footprint_mb: u64,
}

fn default_arch() -> String {
    "transformer".to_string()
}
fn default_ctx() -> usize {
    8_192
}

#[derive(Debug, Default, Deserialize)]
struct CapsSpec {
    #[serde(default)]
    grammar: bool,
    #[serde(default)]
    logprobs: bool,
    #[serde(default)]
    embeddings: bool,
    #[serde(default)]
    lora: bool,
    #[serde(default = "default_true")]
    streaming: bool,
}

fn default_true() -> bool {
    true
}

impl RoleSpec {
    fn to_model_role(&self, name: &str) -> ModelRole {
        let purpose = parse_role_kind(&self.role_kind);
        let arch = match self.model.arch.as_str() {
            "ssm" => ModelArchitecture::Ssm,
            "hybrid" => ModelArchitecture::Hybrid,
            "transformer" => ModelArchitecture::Transformer,
            _ => ModelArchitecture::Unknown,
        };
        let sampler = match self.default_sampler.as_deref() {
            Some("greedy") => SamplerProfile {
                temperature: 0.0,
                top_k: None,
                top_p: None,
                repetition_penalty: None,
                seed: Some(0),
                deterministic: true,
            },
            Some("balanced") => SamplerProfile {
                temperature: 0.7,
                top_k: Some(40),
                top_p: Some(0.9),
                repetition_penalty: Some(1.05),
                seed: None,
                deterministic: false,
            },
            // "edit" and anything else → deterministic edit profile.
            _ => SamplerProfile::deterministic_edit(),
        };
        ModelRole {
            // Stable id == name so escalates_to can resolve before random ids.
            id: RoleId::from(name),
            name: name.to_string(),
            purpose,
            model: ModelDescriptor {
                id: ModelId::from(self.model.id.clone()),
                name: self.model.id.clone(),
                architecture: arch,
                context_tokens: self.model.ctx_len_native,
                tokenizer_signature: self
                    .model
                    .tokenizer_sig
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                footprint_mb: self.model.footprint_mb,
            },
            caps: ProviderCaps {
                streaming: self.caps.streaming,
                embeddings: self.caps.embeddings,
                grammar: self.caps.grammar,
                raw_logits: false,
                logprobs: self.caps.logprobs,
                lora: self.caps.lora,
                kv_handles: false,
                native_tokens_endpoint: true,
            },
            default_sampler: sampler,
            endpoint: self.endpoint.clone(),
            cost: self.cost,
            escalates_to: None, // resolved in the second pass
            metadata: BTreeMap::new(),
        }
    }
}

fn parse_role_kind(kind: &str) -> RolePurpose {
    match kind {
        "hero" | "hero_coder" => RolePurpose::HeroCoder,
        "fast_draft" => RolePurpose::FastDraft,
        "embedder" => RolePurpose::Embedder,
        "reranker" => RolePurpose::Reranker,
        "compactor" | "summarizer" => RolePurpose::Summarizer,
        "classifier" => RolePurpose::Classifier,
        "ssm_long" => RolePurpose::SsmLong,
        _ => RolePurpose::ToolPlanner,
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

    #[test]
    fn roles_toml_loads_and_resolves_escalation() {
        let toml = r#"
[roles.fast_draft]
role_kind = "fast_draft"
endpoint = "http://127.0.0.1:8082"
default_sampler = "greedy"
escalates_to = "hero"
[roles.fast_draft.model]
id = "qwen2.5-0.5b"
arch = "transformer"
ctx_len_native = 32768
footprint_mb = 500
[roles.fast_draft.caps]
grammar = true

[roles.hero]
role_kind = "hero"
endpoint = "http://127.0.0.1:8081"
default_sampler = "edit"
[roles.hero.model]
id = "qwen2.5-7b-instruct"
arch = "transformer"
ctx_len_native = 32768
footprint_mb = 4600
[roles.hero.caps]
grammar = true
"#;
        let registry = RoleRegistry::from_roles_toml(toml).unwrap();
        let draft = registry.by_name("fast_draft").unwrap();
        let hero = registry.by_name("hero").unwrap();
        assert_eq!(draft.purpose, RolePurpose::FastDraft);
        assert_eq!(draft.endpoint.as_deref(), Some("http://127.0.0.1:8082"));
        assert_eq!(draft.escalates_to, Some(hero.id.clone()));
        assert!(hero.escalates_to.is_none());
        assert_eq!(hero.model.footprint_mb, 4600);
    }

    #[test]
    fn missing_file_falls_back_to_defaults() {
        let registry =
            RoleRegistry::load_from_file_or_default("/nonexistent/path/roles.toml").unwrap();
        assert!(!registry.by_purpose(RolePurpose::HeroCoder).is_empty());
    }
}
