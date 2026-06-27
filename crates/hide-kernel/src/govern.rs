use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    pub max_steps: u32,
    pub max_repairs_per_step: u8,
    pub max_replans: u8,
    pub max_wallclock_ms: u64,
    pub max_tokens: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            max_steps: 128,
            max_repairs_per_step: 3,
            max_replans: 5,
            max_wallclock_ms: 30 * 60 * 1000,
            max_tokens: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BudgetLedger {
    pub steps: u32,
    pub replans: u8,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

impl BudgetLedger {
    pub fn consume_step(&mut self) {
        self.steps += 1;
    }

    pub fn within(&self, budget: &Budget) -> bool {
        self.steps < budget.max_steps
            && (budget.max_tokens == 0
                || self.input_tokens + self.output_tokens < budget.max_tokens)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interrupt {
    Abort,
    Pause,
    Steer { instruction: String },
}
