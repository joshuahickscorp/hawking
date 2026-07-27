//! Resource limits and the meter that enforces them.
//!
//! Every dimension the sandbox bounds is a field on [`Limits`]. The [`Meter`]
//! holds the running counters and the virtual clock; each is checked at the
//! point it advances, so exhaustion is caught the instant it happens and turns
//! into a typed [`RuntimeError::LimitExceeded`]. Because time is virtual (there
//! is no real sleep or wall-clock read), a run is deterministic and the
//! wall-time budget is reproducible.

use crate::error::{LimitKind, Result, RuntimeError};

/// The bounds a program runs under. Construct with [`Limits::unbounded`] and
/// tighten, or with [`Limits::strict`] for a conservative default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Maximum AST nodes evaluated.
    pub instructions: u64,
    /// Virtual wall-clock budget, in milliseconds.
    pub wall_time_ms: u64,
    /// Peak single-value memory footprint, in bytes.
    pub memory_bytes: u64,
    /// Maximum serialized size of the program result, in bytes.
    pub output_bytes: u64,
    /// Maximum number of host handle calls.
    pub tool_calls: u32,
    /// Maximum concurrency a map operator may request.
    pub concurrency: u32,
    /// Maximum size of a single spilled artifact, in bytes.
    pub artifact_bytes: u64,
    /// Maximum evaluation nesting depth.
    pub recursion_depth: u32,
    /// Virtual milliseconds charged per handle call. Lets the wall-time budget
    /// be tripped by I/O-shaped work independently of instruction count.
    pub handle_latency_ms: u64,
}

impl Limits {
    /// Effectively no limits. Handy as a base to tighten one dimension at a time.
    pub fn unbounded() -> Self {
        Self {
            instructions: u64::MAX,
            wall_time_ms: u64::MAX,
            memory_bytes: u64::MAX,
            output_bytes: u64::MAX,
            tool_calls: u32::MAX,
            concurrency: u32::MAX,
            artifact_bytes: u64::MAX,
            recursion_depth: u32::MAX,
            handle_latency_ms: 1,
        }
    }

    /// A conservative default suited to a small analysis program.
    pub fn strict() -> Self {
        Self {
            instructions: 100_000,
            wall_time_ms: 5_000,
            memory_bytes: 8 * 1024 * 1024,
            output_bytes: 256 * 1024,
            tool_calls: 128,
            concurrency: 8,
            artifact_bytes: 1024 * 1024,
            recursion_depth: 256,
            handle_latency_ms: 5,
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        Limits::strict()
    }
}

/// Running counters plus the virtual clock. One meter lives per run.
#[derive(Debug)]
pub struct Meter {
    limits: Limits,
    instructions: u64,
    clock_ms: u64,
    tool_calls: u32,
    peak_memory: u64,
}

impl Meter {
    pub fn new(limits: Limits, clock_start_ms: u64) -> Self {
        Self {
            limits,
            instructions: 0,
            clock_ms: clock_start_ms,
            tool_calls: 0,
            peak_memory: 0,
        }
    }

    /// Charge one evaluated AST node. Called at the top of every eval step.
    pub fn tick_instruction(&mut self) -> Result<()> {
        self.instructions += 1;
        if self.instructions > self.limits.instructions {
            return Err(RuntimeError::limit(
                LimitKind::Instruction,
                self.limits.instructions,
                self.instructions,
            ));
        }
        Ok(())
    }

    /// Advance the virtual clock and check the wall-time budget. Used for handle
    /// latency and retry backoff.
    pub fn advance_clock(&mut self, ms: u64) -> Result<()> {
        self.clock_ms = self.clock_ms.saturating_add(ms);
        if self.clock_ms > self.limits.wall_time_ms {
            return Err(RuntimeError::limit(
                LimitKind::WallTime,
                self.limits.wall_time_ms,
                self.clock_ms,
            ));
        }
        Ok(())
    }

    /// Charge one host handle call (and its latency).
    pub fn charge_tool_call(&mut self) -> Result<()> {
        self.tool_calls += 1;
        if self.tool_calls > self.limits.tool_calls {
            return Err(RuntimeError::limit(
                LimitKind::ToolCall,
                self.limits.tool_calls as u64,
                self.tool_calls as u64,
            ));
        }
        let latency = self.limits.handle_latency_ms;
        self.advance_clock(latency)
    }

    /// Record that a value of `bytes` was produced and check the peak-memory
    /// budget.
    pub fn observe_value(&mut self, bytes: u64) -> Result<()> {
        if bytes > self.peak_memory {
            self.peak_memory = bytes;
        }
        if self.peak_memory > self.limits.memory_bytes {
            return Err(RuntimeError::limit(
                LimitKind::Memory,
                self.limits.memory_bytes,
                self.peak_memory,
            ));
        }
        Ok(())
    }

    /// Check that a requested map concurrency is within budget.
    pub fn check_concurrency(&self, requested: u32) -> Result<()> {
        if requested > self.limits.concurrency {
            return Err(RuntimeError::limit(
                LimitKind::Concurrency,
                self.limits.concurrency as u64,
                requested as u64,
            ));
        }
        Ok(())
    }

    /// Check a recursion depth against the budget.
    pub fn check_recursion(&self, depth: u32) -> Result<()> {
        if depth > self.limits.recursion_depth {
            return Err(RuntimeError::limit(
                LimitKind::Recursion,
                self.limits.recursion_depth as u64,
                depth as u64,
            ));
        }
        Ok(())
    }

    /// Check a serialized output size against the budget.
    pub fn check_output(&self, bytes: u64) -> Result<()> {
        if bytes > self.limits.output_bytes {
            return Err(RuntimeError::limit(
                LimitKind::OutputBytes,
                self.limits.output_bytes,
                bytes,
            ));
        }
        Ok(())
    }

    /// Check a spilled-artifact size against the budget.
    pub fn check_artifact(&self, bytes: u64) -> Result<()> {
        if bytes > self.limits.artifact_bytes {
            return Err(RuntimeError::limit(
                LimitKind::ArtifactByte,
                self.limits.artifact_bytes,
                bytes,
            ));
        }
        Ok(())
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// A deterministic snapshot of consumption, included in the run output.
    pub fn usage(&self) -> Usage {
        Usage {
            instructions: self.instructions,
            clock_ms: self.clock_ms,
            tool_calls: self.tool_calls,
            peak_memory_bytes: self.peak_memory,
        }
    }
}

/// What a run consumed. Deterministic for a given program + host, so it does not
/// perturb byte-identical output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Usage {
    pub instructions: u64,
    pub clock_ms: u64,
    pub tool_calls: u32,
    pub peak_memory_bytes: u64,
}
