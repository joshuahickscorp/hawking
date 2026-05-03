//! Qwen3-MoE forward pass. Standard MHA attention, no MLA. Top-8 of
//! 128 routed experts; no shared expert. Validates that the MoE
//! kernel pack isn't DeepSeek-shaped.
//!
//! Phase 0: stub returning Unimplemented. Phase 3 fills this in
//! against the same kernel/runtime APIs as DeepSeek.

use crate::engine::{Engine, EngineConfig, GenStats, GenerateRequest, StreamEvent};
use crate::{Error, Result};
use std::path::Path;

pub struct QwenMoE {
    pub model_id: String,
}

impl Engine for QwenMoE {
    fn load(_weights: &Path, _config: EngineConfig) -> Result<Self> {
        Err(Error::Unimplemented(
            "qwen-moe: lands in Phase 3 (DeepSeek-V2-Lite ships first)",
        ))
    }

    fn generate(
        &mut self,
        _req: GenerateRequest,
        _sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<GenStats> {
        Err(Error::Unimplemented("qwen-moe forward"))
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_arch(&self) -> &str {
        "qwen2"
    }
}
