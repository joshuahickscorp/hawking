use crate::budget::{RegionBudget, TokenBudget};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextProfile {
    pub name: String,
    pub budget: TokenBudget,
    pub regions: Vec<RegionBudget>,
    pub preserve_document_order: bool,
    pub pin_system_to_head: bool,
    pub pin_recent_to_tail: bool,
}

impl ContextProfile {
    pub fn coding_default(max_input_tokens: usize) -> Self {
        Self {
            name: "coding_default".to_string(),
            budget: TokenBudget {
                max_input_tokens,
                reserve_output_tokens: 2_048.min(max_input_tokens / 4),
                hard_limit_tokens: max_input_tokens,
            },
            regions: vec![
                RegionBudget {
                    region: "system".to_string(),
                    target_tokens: 1_024,
                    max_tokens: 2_048,
                },
                RegionBudget {
                    region: "code".to_string(),
                    target_tokens: max_input_tokens / 2,
                    max_tokens: max_input_tokens,
                },
                RegionBudget {
                    region: "memory".to_string(),
                    target_tokens: max_input_tokens / 8,
                    max_tokens: max_input_tokens / 4,
                },
            ],
            preserve_document_order: true,
            pin_system_to_head: true,
            pin_recent_to_tail: true,
        }
    }
}
