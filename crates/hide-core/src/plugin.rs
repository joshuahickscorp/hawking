use crate::ids::PluginId;
use crate::permission::Capability;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtensionManifest {
    pub id: PluginId,
    pub name: String,
    pub version: String,
    pub runtime: ExtensionRuntime,
    pub required_capabilities: Vec<Capability>,
    pub contributions: Vec<ExtensionContribution>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtensionRuntime {
    TrustedRust,
    WasmComponent,
    McpServer,
    Skill,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExtensionContribution {
    Tool { name: String },
    Panel { id: String },
    ModelProvider { id: String },
    Indexer { language: String },
    MemoryStore { id: String },
    Command { id: String },
    EventKind { event_kind: String },
}

#[derive(Debug, Default)]
pub struct ExtensionRegistry {
    manifests: BTreeMap<PluginId, ExtensionManifest>,
}

impl ExtensionRegistry {
    pub fn register(&mut self, manifest: ExtensionManifest) {
        self.manifests.insert(manifest.id.clone(), manifest);
    }

    pub fn get(&self, id: &PluginId) -> Option<&ExtensionManifest> {
        self.manifests.get(id)
    }

    pub fn len(&self) -> usize {
        self.manifests.len()
    }
}
