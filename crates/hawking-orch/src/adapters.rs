//! LoRA / adapter descriptors + selection (ch.06 §4.9).
//!
//! No LoRA *serving* exists in the runtime yet ([RUNTIME-SIDE — LATER]); this
//! module is the **selection policy** the router uses today and the clean seam a
//! runtime adapter-loader will implement. The policy is real: given a task, a
//! detected language, and config, it resolves the right adapter set
//! (`["rust","personal"]` for a Rust edit, `["commit-msg"]` for a commit, etc.)
//! and validates the choice against a role's `caps.lora`.

use hide_core::runtime::ModelRole;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A single resident-or-on-disk adapter the runtime can apply as a low-rank
/// delta over the base forward.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterDescriptor {
    pub id: String,
    pub kind: AdapterKind,
    /// Path/artifact id the runtime loader resolves (opaque to the shell).
    pub artifact: String,
    /// Default blend weight when selected.
    pub default_scale: f32,
}

/// What an adapter specializes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterKind {
    /// Per-language idioms (`rust`, `ts`, `python`).
    Language(String),
    /// Per-task (`commit-msg`, `sql`, `test-gen`).
    Task(String),
    /// The user's accepted-edit personal adapter (§4.10).
    Personal,
}

/// One selected adapter + its blend weight (the §4.9.1 `AdapterRef`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterRef {
    pub id: String,
    pub scale: f32,
}

/// The router's resolved adapter set for a request (the §4.9.1 `AdapterSelection`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AdapterSelection {
    /// Usually 0..2 adapters, e.g. `[language, personal]`.
    pub adapters: Vec<AdapterRef>,
}

impl AdapterSelection {
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }
}

/// The catalog of available adapters + the selection policy.
#[derive(Debug, Clone, Default)]
pub struct AdapterRegistry {
    by_id: BTreeMap<String, AdapterDescriptor>,
    /// Whether the user's personal adapter is enabled (opt-in, §4.10).
    personal_enabled: bool,
}

impl AdapterRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, descriptor: AdapterDescriptor) {
        self.by_id.insert(descriptor.id.clone(), descriptor);
    }

    pub fn enable_personal(&mut self, enabled: bool) {
        self.personal_enabled = enabled;
    }

    pub fn get(&self, id: &str) -> Option<&AdapterDescriptor> {
        self.by_id.get(id)
    }

    /// Pick adapters for a request (§4.4.2 `pick_adapter`): a language adapter
    /// (if one exists for the detected language) plus the personal adapter (if
    /// enabled), or a task adapter for task-shaped requests. The choice is
    /// validated against the role's `caps.lora` — a role without LoRA support
    /// gets an empty selection (it degrades to the base model).
    pub fn select(
        &self,
        role: &ModelRole,
        task_kind: &str,
        language: Option<&str>,
    ) -> AdapterSelection {
        if !role.caps.lora {
            return AdapterSelection::default();
        }
        let mut selection = AdapterSelection::default();

        // Task adapters take the slot for task-shaped requests (commit-msg, sql).
        if let Some(task_adapter) = self.task_adapter_for(task_kind) {
            selection.adapters.push(AdapterRef {
                id: task_adapter.id.clone(),
                scale: task_adapter.default_scale,
            });
        } else if let Some(lang) = language {
            if let Some(lang_adapter) = self.language_adapter_for(lang) {
                selection.adapters.push(AdapterRef {
                    id: lang_adapter.id.clone(),
                    scale: lang_adapter.default_scale,
                });
            }
        }

        // The personal adapter composes on top (when enabled and present).
        if self.personal_enabled {
            if let Some(personal) = self
                .by_id
                .values()
                .find(|d| d.kind == AdapterKind::Personal)
            {
                selection.adapters.push(AdapterRef {
                    id: personal.id.clone(),
                    scale: personal.default_scale,
                });
            }
        }
        selection
    }

    fn language_adapter_for(&self, language: &str) -> Option<&AdapterDescriptor> {
        let lang = language.to_lowercase();
        self.by_id.values().find(|d| {
            matches!(&d.kind, AdapterKind::Language(l) if l.eq_ignore_ascii_case(&lang))
        })
    }

    fn task_adapter_for(&self, task_kind: &str) -> Option<&AdapterDescriptor> {
        // Map a few task kinds onto their task adapters by id convention.
        let id = match task_kind {
            "commit_msg" | "commit-msg" => "commit-msg",
            "sql" => "sql",
            "test_gen" | "test-gen" => "test-gen",
            _ => return None,
        };
        self.by_id
            .get(id)
            .filter(|d| matches!(d.kind, AdapterKind::Task(_)))
    }
}

/// Map a file extension to the language adapter key the registry uses.
pub fn language_for_extension(ext: &str) -> Option<&'static str> {
    Some(match ext.trim_start_matches('.') {
        "rs" => "rust",
        "ts" | "tsx" => "ts",
        "js" | "jsx" => "js",
        "py" => "python",
        "go" => "go",
        "java" => "java",
        "c" | "h" => "c",
        "cpp" | "cc" | "hpp" => "cpp",
        _ => return None,
    })
}

/// The seam a runtime adapter-loader will implement ([RUNTIME-SIDE — LATER]).
/// Today only the no-op / not-loaded path exists; the shell-today fallback is a
/// separately-merged model served as its own role (§4.9.3).
pub trait AdapterServer: Send + Sync {
    /// Ensure the given adapters are resident; returns the ids actually loaded.
    fn ensure_resident(&self, adapters: &[AdapterRef]) -> hide_core::Result<Vec<String>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::{ModelId, RoleId};
    use hide_core::runtime::{
        ModelArchitecture, ModelDescriptor, ProviderCaps, RolePurpose, SamplerProfile,
    };

    fn role_with_lora(lora: bool) -> ModelRole {
        ModelRole {
            id: RoleId::new(),
            name: "hero".into(),
            purpose: RolePurpose::HeroCoder,
            model: ModelDescriptor {
                id: ModelId::new(),
                name: "hero".into(),
                architecture: ModelArchitecture::Transformer,
                context_tokens: 4096,
                tokenizer_signature: "tok".into(),
                footprint_mb: 4600,
            },
            caps: ProviderCaps {
                lora,
                ..ProviderCaps::hawking_local_shell_today()
            },
            default_sampler: SamplerProfile::deterministic_edit(),
            endpoint: None,
            cost: None,
            escalates_to: None,
            metadata: BTreeMap::new(),
        }
    }

    fn registry() -> AdapterRegistry {
        let mut r = AdapterRegistry::new();
        r.register(AdapterDescriptor {
            id: "rust".into(),
            kind: AdapterKind::Language("rust".into()),
            artifact: "rust.lora".into(),
            default_scale: 1.0,
        });
        r.register(AdapterDescriptor {
            id: "commit-msg".into(),
            kind: AdapterKind::Task("commit-msg".into()),
            artifact: "commit-msg.lora".into(),
            default_scale: 1.0,
        });
        r.register(AdapterDescriptor {
            id: "personal".into(),
            kind: AdapterKind::Personal,
            artifact: "personal.lora".into(),
            default_scale: 0.5,
        });
        r
    }

    #[test]
    fn no_lora_cap_means_empty_selection() {
        let r = registry();
        let sel = r.select(&role_with_lora(false), "edit_code", Some("rust"));
        assert!(sel.is_empty());
    }

    #[test]
    fn rust_edit_selects_language_and_personal() {
        let mut r = registry();
        r.enable_personal(true);
        let sel = r.select(&role_with_lora(true), "edit_code", Some("rust"));
        let ids: Vec<&str> = sel.adapters.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, vec!["rust", "personal"]);
        assert_eq!(sel.adapters[1].scale, 0.5);
    }

    #[test]
    fn task_adapter_takes_the_slot() {
        let r = registry();
        let sel = r.select(&role_with_lora(true), "commit-msg", Some("rust"));
        let ids: Vec<&str> = sel.adapters.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, vec!["commit-msg"]);
    }

    #[test]
    fn extension_mapping() {
        assert_eq!(language_for_extension("rs"), Some("rust"));
        assert_eq!(language_for_extension(".tsx"), Some("ts"));
        assert_eq!(language_for_extension("md"), None);
    }
}
