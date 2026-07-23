//! The **one source/artifact record**, valid across every pack. The Seed owns source and artifact
//! identity; packs only DECLARE what they require. This record unifies Hugging Face source references,
//! local source roots, immutable revisions, shard identities, tokenizer identities, and asset hashes into
//! a single sealed [`crate::record::Record`] of kind `source`. No pack invents an independent source
//! receipt or artifact store.

use crate::evidence::receipt;
use crate::record::Record;
use crate::Result;
use serde::{Deserialize, Serialize};

/// One shard of a sharded source (e.g. a safetensors shard), with its content identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shard {
    pub name: String,
    pub sha256: String,
    pub bytes: usize,
}

/// The unified source record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRecord {
    /// Hugging Face reference, e.g. `openai/gpt-oss-120b`, or empty for local-only.
    #[serde(default)]
    pub hf_repo: String,
    /// Immutable revision (commit hash) pinning the source.
    #[serde(default)]
    pub revision: String,
    /// Local source root (absolute path) when the source is materialized offline.
    #[serde(default)]
    pub local_root: String,
    /// Declared required source formats (e.g. `gguf`, `safetensors`).
    #[serde(default)]
    pub formats: Vec<String>,
    /// Declared required tensor dtypes (e.g. `Q4_K`, `MXFP4`, `BF16`).
    #[serde(default)]
    pub tensor_types: Vec<String>,
    /// Tokenizer identity (hash or name), if the pack needs a tokenizer/protocol asset.
    #[serde(default)]
    pub tokenizer_identity: String,
    /// The source shards with their content identities.
    #[serde(default)]
    pub shards: Vec<Shard>,
}

impl SourceRecord {
    pub fn local(local_root: &str) -> Self {
        SourceRecord {
            hf_repo: String::new(),
            revision: String::new(),
            local_root: local_root.into(),
            formats: Vec::new(),
            tensor_types: Vec::new(),
            tokenizer_identity: String::new(),
            shards: Vec::new(),
        }
    }

    pub fn hf(repo: &str, revision: &str) -> Self {
        let mut s = SourceRecord::local("");
        s.hf_repo = repo.into();
        s.revision = revision.into();
        s
    }

    pub fn with_format(mut self, f: &str) -> Self {
        self.formats.push(f.into());
        self
    }
    pub fn with_tensor_type(mut self, t: &str) -> Self {
        self.tensor_types.push(t.into());
        self
    }
    pub fn with_shard(mut self, name: &str, sha256: &str, bytes: usize) -> Self {
        self.shards.push(Shard { name: name.into(), sha256: sha256.into(), bytes });
        self
    }
    pub fn with_tokenizer(mut self, identity: &str) -> Self {
        self.tokenizer_identity = identity.into();
        self
    }

    /// The stable identity of this source, order-independent over its declared fields. Reuses the Seed's
    /// one canonical-JSON + sha256 engine directly (`Record::new`).
    pub fn identity(&self) -> Result<String> {
        Ok(Record::new("source", serde_json::to_value(self)?).identity)
    }

    /// Seal the source declaration as a Seed `source` receipt (the ONE evidence engine).
    pub fn seal(&self) -> Result<Record> {
        Ok(receipt("source", serde_json::to_value(self)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_record_seals_and_is_content_addressed() {
        let s = SourceRecord::hf("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a")
            .with_format("safetensors")
            .with_tensor_type("MXFP4")
            .with_tensor_type("BF16")
            .with_shard("model--00001-of-00007.safetensors", "abc123", 1234);
        let rec = s.seal().unwrap();
        assert!(rec.verify().is_ok() && rec.kind == "source");
        // identity is order-independent across a serde round-trip
        let s2: SourceRecord = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(s.identity().unwrap(), s2.identity().unwrap());
    }
}
