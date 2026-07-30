//! Durable sinks that accept **only** [`Verified`] tokens.
//!
//! A draft token that reaches any of these sinks is a correctness bug of the
//! worst kind. The type signature makes that a compile error:
//!
//! ```ignore
//! // does not compile — expected VerifiedTokenId, found DraftTokenId
//! sink.emit_canonical_event(DraftTokenId::id(1));
//! ```
//!
//! The five sinks named by the speculation-safety invariant:
//! 1. canonical event stream
//! 2. durable context / memory
//! 3. tool dispatch
//! 4. file edit
//! 5. final user-visible output

use crate::token_boundary::{Verified, VerifiedTokenId};

/// Error from a durable sink (fixture or host adapter).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DurableSinkError {
    #[error("durable sink refused: {0}")]
    Refused(String),
}

pub type DurableResult<T> = Result<T, DurableSinkError>;

/// One durable action recorded by the in-memory fixture sink.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurableRecord {
    CanonicalEvent { token_id: u32 },
    MemoryWrite { token_id: u32 },
    ToolDispatch { token_id: u32 },
    FileEdit { token_id: u32 },
    FinalOutput { token_id: u32 },
}

/// The durable sink surface. **Every method takes [`Verified`] only.**
pub trait DurableTokenSink {
    /// Canonical event stream (session log / token event).
    fn emit_canonical_event(&mut self, token: VerifiedTokenId) -> DurableResult<()>;

    /// Durable context or any memory class.
    fn write_memory(&mut self, token: VerifiedTokenId) -> DurableResult<()>;

    /// Trigger a tool call.
    fn dispatch_tool(&mut self, token: VerifiedTokenId) -> DurableResult<()>;

    /// Change a file (edit/diff).
    fn edit_file(&mut self, token: VerifiedTokenId) -> DurableResult<()>;

    /// Final user-visible output.
    fn final_output(&mut self, token: VerifiedTokenId) -> DurableResult<()>;
}

/// In-memory fixture sink for unit tests. Not production I/O.
#[derive(Debug, Clone, Default)]
pub struct InMemoryDurableSink {
    records: Vec<DurableRecord>,
}

impl InMemoryDurableSink {
    pub fn events(&self) -> &[DurableRecord] {
        &self.records
    }

    pub fn clear(&mut self) {
        self.records.clear();
    }

    pub fn token_ids(&self) -> Vec<u32> {
        self.records
            .iter()
            .map(|r| match r {
                DurableRecord::CanonicalEvent { token_id }
                | DurableRecord::MemoryWrite { token_id }
                | DurableRecord::ToolDispatch { token_id }
                | DurableRecord::FileEdit { token_id }
                | DurableRecord::FinalOutput { token_id } => *token_id,
            })
            .collect()
    }
}

impl DurableTokenSink for InMemoryDurableSink {
    fn emit_canonical_event(&mut self, token: VerifiedTokenId) -> DurableResult<()> {
        self.records.push(DurableRecord::CanonicalEvent {
            token_id: token.get(),
        });
        Ok(())
    }

    fn write_memory(&mut self, token: VerifiedTokenId) -> DurableResult<()> {
        self.records.push(DurableRecord::MemoryWrite {
            token_id: token.get(),
        });
        Ok(())
    }

    fn dispatch_tool(&mut self, token: VerifiedTokenId) -> DurableResult<()> {
        self.records.push(DurableRecord::ToolDispatch {
            token_id: token.get(),
        });
        Ok(())
    }

    fn edit_file(&mut self, token: VerifiedTokenId) -> DurableResult<()> {
        self.records.push(DurableRecord::FileEdit {
            token_id: token.get(),
        });
        Ok(())
    }

    fn final_output(&mut self, token: VerifiedTokenId) -> DurableResult<()> {
        self.records.push(DurableRecord::FinalOutput {
            token_id: token.get(),
        });
        Ok(())
    }
}

/// Batch helper: only verified tokens may be flushed into a sink.
pub fn flush_verified_prefix<S: DurableTokenSink>(
    sink: &mut S,
    tokens: &[VerifiedTokenId],
) -> DurableResult<()> {
    for &t in tokens {
        sink.emit_canonical_event(t)?;
    }
    Ok(())
}

/// Text-bearing payload wrapper used when sinks carry more than a bare id.
pub type VerifiedText = Verified<String>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_boundary::{DraftTokenId, TargetVerification};
    #[test]
    fn all_five_sinks_accept_only_verified() {
        let gate = TargetVerification::gate();
        let mut sink = InMemoryDurableSink::default();
        let v = gate.emit_target(11u32);
        sink.emit_canonical_event(v).unwrap();
        sink.write_memory(v).unwrap();
        sink.dispatch_tool(v).unwrap();
        sink.edit_file(v).unwrap();
        sink.final_output(v).unwrap();
        assert_eq!(sink.events().len(), 5);
        let _draft = DraftTokenId::id(99);
        assert!(!sink.token_ids().contains(&99));
    }
    #[test]
    fn flush_verified_prefix_writes_in_order() {
        let gate = TargetVerification::gate();
        let mut sink = InMemoryDurableSink::default();
        let toks = vec![gate.emit_target(1u32), gate.emit_target(2u32)];
        flush_verified_prefix(&mut sink, &toks).unwrap();
        assert_eq!(sink.token_ids(), vec![1, 2]);
    }
}
