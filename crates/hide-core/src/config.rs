use crate::types::Decision;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HideConfig {
    pub user_root: PathBuf,
    pub workspace_root: PathBuf,
    pub runtime: RuntimeConfig,
    pub persistence: PersistenceConfig,
    pub security: SecurityConfig,
    pub context: ContextConfig,
    pub index: IndexConfig,
}

impl HideConfig {
    pub fn for_workspace(workspace_root: impl Into<PathBuf>) -> Self {
        let workspace_root = workspace_root.into();
        let user_root = std::env::var_os("HAWKING_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".hawking")))
            .unwrap_or_else(|| PathBuf::from(".hawking"));
        Self {
            user_root,
            workspace_root,
            runtime: RuntimeConfig::default(),
            persistence: PersistenceConfig::default(),
            security: SecurityConfig::default(),
            context: ContextConfig::default(),
            index: IndexConfig::default(),
        }
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub provider: String,
    pub base_url: String,
    pub spawn_sidecar: bool,
    pub health_timeout_ms: u64,
    pub restart_backoff_ms: Vec<u64>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            provider: "hawking-local".to_string(),
            base_url: "http://127.0.0.1:8080".to_string(),
            spawn_sidecar: true,
            health_timeout_ms: 60_000,
            restart_backoff_ms: vec![1_000, 2_000, 4_000, 8_000, 30_000],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistenceConfig {
    pub fsync_every_event: bool,
    pub segment_bytes: u64,
    pub snapshot_interval_events: u64,
    pub encryption_at_rest: bool,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            fsync_every_event: false,
            segment_bytes: 64 * 1024 * 1024,
            snapshot_interval_events: 250,
            encryption_at_rest: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub default_decision: Decision,
    pub network_default: Decision,
    pub shell_default: Decision,
    pub workspace_write_default: Decision,
    pub require_exact_effect_grants: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            default_decision: Decision::Ask,
            network_default: Decision::Deny,
            shell_default: Decision::Ask,
            workspace_write_default: Decision::Ask,
            require_exact_effect_grants: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextConfig {
    pub max_input_tokens: usize,
    pub reserve_output_tokens: usize,
    pub memory_top_k: usize,
    pub code_top_k: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_input_tokens: 16_384,
            reserve_output_tokens: 2_048,
            memory_top_k: 12,
            code_top_k: 40,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexConfig {
    pub enable_daemon: bool,
    pub lexical_first: bool,
    pub embedding_rerank: bool,
    pub max_file_bytes: u64,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            enable_daemon: true,
            lexical_first: true,
            embedding_rerank: true,
            max_file_bytes: 2 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigLayer {
    pub name: String,
    pub source: PathBuf,
    pub locked: bool,
    pub config: HideConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_roundtrips_as_json() {
        let dir = std::env::temp_dir().join(format!("hide_config_{}", crate::ids::now_ms()));
        let path = dir.join(".hide").join("config.json");
        let mut config = HideConfig::for_workspace(&dir);
        config.runtime.base_url = "http://127.0.0.1:9999".to_string();

        config.save_json(&path).unwrap();
        let loaded = HideConfig::load_json(&path).unwrap();

        assert_eq!(loaded.workspace_root, dir);
        assert_eq!(loaded.runtime.base_url, "http://127.0.0.1:9999");
        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }
}
