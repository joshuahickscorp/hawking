//! Model-aware context governor.
//!
//! Context is a cache, not memory. The governor measures and budgets actual
//! packed context components and protects the output reserve.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    Edit,
    Search,
    Test,
    Architecture,
    Benchmark,
}

impl Default for TaskType {
    fn default() -> Self {
        TaskType::Edit
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ContextBudgets {
    pub system: usize,
    pub stable_policy: usize,
    pub repo_map: usize,
    pub task_map: usize,
    pub source: usize,
    pub receipts: usize,
    pub history: usize,
    pub tool_results: usize,
    pub reasoning_reserve: usize,
    pub output_reserve: usize,
}

impl ContextBudgets {
    pub fn projected_context(&self) -> usize {
        self.system
            .saturating_add(self.stable_policy)
            .saturating_add(self.repo_map)
            .saturating_add(self.task_map)
            .saturating_add(self.source)
            .saturating_add(self.receipts)
            .saturating_add(self.history)
            .saturating_add(self.tool_results)
            .saturating_add(self.reasoning_reserve)
    }

    pub fn total(&self) -> usize {
        self.projected_context().saturating_add(self.output_reserve)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ContextGovernor {
    pub runtime_context: usize,
    pub output_max: usize,
}

impl ContextGovernor {
    pub fn new(runtime_context: usize, output_max: usize) -> Self {
        Self {
            runtime_context: runtime_context.max(1024),
            output_max: output_max.max(256),
        }
    }

    pub fn budgets_for(
        &self,
        task: TaskType,
        _repo_size_files: usize,
        _node_complexity: usize,
        pressure: f64,
    ) -> ContextBudgets {
        let pressure = pressure.clamp(0.0, 1.0);
        let scale = 1.0 - (pressure * 0.5);
        let map_frac = match task {
            TaskType::Search => 0.15,
            TaskType::Edit => 0.10,
            TaskType::Test => 0.08,
            TaskType::Architecture => 0.20,
            TaskType::Benchmark => 0.05,
        };

        let repo_map = ((self.runtime_context as f64 * map_frac * scale) as usize).clamp(512, 32768);
        let task_map = (repo_map / 2).clamp(256, 16384);
        let source = ((self.runtime_context as f64 * 0.25 * scale) as usize).clamp(1024, 32768);
        let receipts = ((self.runtime_context as f64 * 0.05 * scale) as usize).clamp(256, 8192);
        let history = ((self.runtime_context as f64 * 0.10 * scale) as usize).clamp(256, 16384);
        let tool_results = ((self.runtime_context as f64 * 0.10 * scale) as usize).clamp(256, 16384);
        let system = 512;
        let stable_policy = 512;
        let reasoning_reserve = (self.runtime_context as f64 * 0.10) as usize;
        let output_reserve = self.output_max;

        let mut b = ContextBudgets {
            system,
            stable_policy,
            repo_map,
            task_map,
            source,
            receipts,
            history,
            tool_results,
            reasoning_reserve,
            output_reserve,
        };

        let target = self.runtime_context.saturating_sub(1);
        let excess = b.total().saturating_sub(target);
        if excess > 0 {
            b.history = b.history.saturating_sub(excess / 3);
            b.tool_results = b.tool_results.saturating_sub(excess / 3);
            b.repo_map = b
                .repo_map
                .saturating_sub(excess.saturating_sub(excess / 3).saturating_sub(excess / 3));
        }
        b
    }

    pub fn invariant_ok(&self, b: &ContextBudgets) -> bool {
        b.projected_context().saturating_add(b.output_reserve) < self.runtime_context
    }

    pub fn compact(&self, b: ContextBudgets, pressure: f64) -> ContextBudgets {
        let factor = if pressure > 0.8 {
            0.5
        } else if pressure > 0.6 {
            0.75
        } else {
            1.0
        };
        ContextBudgets {
            history: (b.history as f64 * factor) as usize,
            tool_results: (b.tool_results as f64 * factor) as usize,
            receipts: (b.receipts as f64 * factor) as usize,
            repo_map: (b.repo_map as f64 * factor) as usize,
            ..b
        }
    }

    pub fn measure_tokens(text: &str) -> usize {
        text.split_whitespace().count()
    }
}
