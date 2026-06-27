use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenBudget {
    pub max_input_tokens: usize,
    pub reserve_output_tokens: usize,
    pub hard_limit_tokens: usize,
}

impl TokenBudget {
    pub fn available_input(&self) -> usize {
        self.max_input_tokens
            .saturating_sub(self.reserve_output_tokens)
            .min(self.hard_limit_tokens)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionBudget {
    pub region: String,
    pub target_tokens: usize,
    pub max_tokens: usize,
}

pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().saturating_add(3) / 4
}
