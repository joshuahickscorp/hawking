//! Metric separation for speculation and TG-ladder scoreboards.
//!
//! `BASE_TRUE_TPS`, `BLOCK_EXECUTED_TPS`, `ACCELERATED_ACCEPTED_TPS`,
//! `PREFILL_TPS`, and `TTFT` must never be averaged or mixed in a shared float
//! field. They are distinct newtypes with no conversion between them and no
//! shared "mean tps" helper.
//!
//! **Accepted TPS accounting rule:** wall time must include **full draft +
//! verify + rollback** cost. Draft-side throughput alone is not
//! `ACCELERATED_ACCEPTED_TPS`.
//!
//! Ascension bible §10 keeps the same separation for Self-TG gauntlets;
//! Python scaffold: `lab/operators/ascension_tg_gauntlet.py`.

use core::fmt;
use core::time::Duration;

/// Baseline true tokens/second of the target path with speculation **off**.
/// Scoreboard name: `BASE_TRUE_TPS`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct BaseTrueTps(f64);

/// Accelerated accepted tokens/second with **full** draft, verify, and rollback
/// cost in the denominator. Scoreboard name: `ACCELERATED_ACCEPTED_TPS`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct AcceleratedAcceptedTps(f64);

/// Tokens/second counted only while the GPU block/command graph is executing
/// (excludes host queue wait that is not device work). Scoreboard name:
/// `BLOCK_EXECUTED_TPS`. Never a substitute for `BASE_TRUE_TPS`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct BlockExecutedTps(f64);

/// Prefill-phase tokens/second. Scoreboard name: `PREFILL_TPS`.
/// Kept separate from decode `BASE_TRUE_TPS`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct PrefillTps(f64);

/// Time to first token (seconds). Scoreboard name: `TTFT`.
/// Not a tokens/second metric; never mixed into a TPS average.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TtftSeconds(f64);

impl BaseTrueTps {
    pub const SCOREBOARD: &'static str = "BASE_TRUE_TPS";

    pub fn new(tps: f64) -> Self {
        Self(tps.max(0.0))
    }

    pub fn value(self) -> f64 {
        self.0
    }

    /// From accepted (target) tokens and pure baseline wall time.
    pub fn from_counts(tokens: u64, wall: Duration) -> Self {
        Self::new(tps(tokens, wall))
    }
}

impl AcceleratedAcceptedTps {
    pub const SCOREBOARD: &'static str = "ACCELERATED_ACCEPTED_TPS";

    pub fn new(tps: f64) -> Self {
        Self(tps.max(0.0))
    }

    pub fn value(self) -> f64 {
        self.0
    }
}

impl BlockExecutedTps {
    pub const SCOREBOARD: &'static str = "BLOCK_EXECUTED_TPS";

    pub fn new(tps: f64) -> Self {
        Self(tps.max(0.0))
    }

    pub fn value(self) -> f64 {
        self.0
    }

    pub fn from_counts(tokens: u64, block_wall: Duration) -> Self {
        Self::new(tps(tokens, block_wall))
    }
}

impl PrefillTps {
    pub const SCOREBOARD: &'static str = "PREFILL_TPS";

    pub fn new(tps: f64) -> Self {
        Self(tps.max(0.0))
    }

    pub fn value(self) -> f64 {
        self.0
    }

    pub fn from_counts(tokens: u64, wall: Duration) -> Self {
        Self::new(tps(tokens, wall))
    }
}

impl TtftSeconds {
    pub const SCOREBOARD: &'static str = "TTFT";

    pub fn new(seconds: f64) -> Self {
        Self(seconds.max(0.0))
    }

    pub fn value(self) -> f64 {
        self.0
    }

    pub fn from_duration(wall: Duration) -> Self {
        Self::new(wall.as_secs_f64())
    }
}

impl fmt::Display for BaseTrueTps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={:.6}", Self::SCOREBOARD, self.0)
    }
}

impl fmt::Display for AcceleratedAcceptedTps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={:.6}", Self::SCOREBOARD, self.0)
    }
}

impl fmt::Display for BlockExecutedTps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={:.6}", Self::SCOREBOARD, self.0)
    }
}

impl fmt::Display for PrefillTps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={:.6}", Self::SCOREBOARD, self.0)
    }
}

impl fmt::Display for TtftSeconds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={:.6}s", Self::SCOREBOARD, self.0)
    }
}

fn tps(tokens: u64, wall: Duration) -> f64 {
    let secs = wall.as_secs_f64();
    if secs <= 0.0 || tokens == 0 {
        return 0.0;
    }
    tokens as f64 / secs
}

/// Cost ledger for one accelerated window. Every phase is mandatory so accepted
/// TPS cannot "forget" rollback.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AccelCostLedger {
    /// Wall nanoseconds spent drafting.
    pub draft_ns: u64,
    /// Wall nanoseconds spent in target verify.
    pub verify_ns: u64,
    /// Wall nanoseconds spent rolling back rejected provisional KV.
    pub rollback_ns: u64,
    /// Target-verified tokens that advanced committed state.
    pub accepted_tokens: u64,
    /// Draft tokens proposed (accepted + rejected).
    pub draft_tokens: u64,
    /// Draft tokens rejected (for diagnostics; not a scoreboard).
    pub rejected_tokens: u64,
}

impl AccelCostLedger {
    pub fn total_ns(self) -> u64 {
        self.draft_ns
            .saturating_add(self.verify_ns)
            .saturating_add(self.rollback_ns)
    }

    pub fn total_wall(self) -> Duration {
        Duration::from_nanos(self.total_ns())
    }

    /// `ACCELERATED_ACCEPTED_TPS` = accepted_tokens / (draft+verify+rollback).
    /// Draft-only time is **not** sufficient.
    pub fn accelerated_accepted_tps(self) -> AcceleratedAcceptedTps {
        AcceleratedAcceptedTps::new(tps(self.accepted_tokens, self.total_wall()))
    }

    /// Deliberately *not* the scoreboard: draft-side throughput excluding verify
    /// and rollback. Exposed only so tests can prove it differs from accepted TPS.
    pub fn draft_side_throughput_not_scoreboard(self) -> f64 {
        tps(
            self.draft_tokens,
            Duration::from_nanos(self.draft_ns.max(1)),
        )
    }

    pub fn record_draft(&mut self, ns: u64, drafted: u64) {
        self.draft_ns = self.draft_ns.saturating_add(ns);
        self.draft_tokens = self.draft_tokens.saturating_add(drafted);
    }

    pub fn record_verify(&mut self, ns: u64, accepted: u64, rejected: u64) {
        self.verify_ns = self.verify_ns.saturating_add(ns);
        self.accepted_tokens = self.accepted_tokens.saturating_add(accepted);
        self.rejected_tokens = self.rejected_tokens.saturating_add(rejected);
    }

    pub fn record_rollback(&mut self, ns: u64) {
        self.rollback_ns = self.rollback_ns.saturating_add(ns);
    }
}

/// Structurally separated scoreboard pair. There is **no** method that averages
/// the two fields together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeparatedTpsScoreboard {
    pub base_true: BaseTrueTps,
    pub accelerated_accepted: AcceleratedAcceptedTps,
}

impl SeparatedTpsScoreboard {
    pub fn new(base_true: BaseTrueTps, accelerated_accepted: AcceleratedAcceptedTps) -> Self {
        Self {
            base_true,
            accelerated_accepted,
        }
    }

    /// Speedup ratio for reporting only — not a blended TPS.
    pub fn speedup_ratio(self) -> Option<f64> {
        let b = self.base_true.value();
        if b <= 0.0 {
            return None;
        }
        Some(self.accelerated_accepted.value() / b)
    }
}

/// Full TG / complete-token scoreboard with every bible §10 metric kept as a
/// distinct typed field. Optional cells are `None` until eligible measurement.
/// There is **no** blended TPS method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeparatedTgScoreboard {
    pub base_true: Option<BaseTrueTps>,
    pub block_executed: Option<BlockExecutedTps>,
    pub accelerated_accepted: Option<AcceleratedAcceptedTps>,
    pub prefill: Option<PrefillTps>,
    pub ttft: Option<TtftSeconds>,
}

impl SeparatedTgScoreboard {
    pub fn empty() -> Self {
        Self {
            base_true: None,
            block_executed: None,
            accelerated_accepted: None,
            prefill: None,
            ttft: None,
        }
    }

    /// Reporting-only speedup; never a blended mean.
    pub fn accel_over_base_ratio(self) -> Option<f64> {
        let b = self.base_true?.value();
        let a = self.accelerated_accepted?.value();
        if b <= 0.0 {
            return None;
        }
        Some(a / b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn accepted_tps_includes_rollback_cost() {
        let mut ledger = AccelCostLedger::default();
        ledger.record_draft(10_000_000, 10);
        ledger.record_verify(10_000_000, 5, 5);
        ledger.record_rollback(10_000_000);
        let accepted = ledger.accelerated_accepted_tps();
        let expected = 5.0 / 0.030;
        assert!((accepted.value() - expected).abs() < 1e-6);
        let draft_side = ledger.draft_side_throughput_not_scoreboard();
        assert!((draft_side - accepted.value()).abs() > 1.0);
        let without_rollback = tps(
            ledger.accepted_tokens,
            Duration::from_nanos(ledger.draft_ns + ledger.verify_ns),
        );
        assert!(without_rollback > accepted.value());
        assert_eq!(ledger.total_ns(), 30_000_000);
        assert_eq!(
            AcceleratedAcceptedTps::SCOREBOARD,
            "ACCELERATED_ACCEPTED_TPS"
        );
        assert_eq!(BaseTrueTps::SCOREBOARD, "BASE_TRUE_TPS");
    }
    #[test]
    fn base_and_accelerated_are_separate_types() {
        let base = BaseTrueTps::from_counts(100, Duration::from_secs(1));
        let mut ledger = AccelCostLedger::default();
        ledger.record_draft(500_000_000, 80);
        ledger.record_verify(400_000_000, 70, 10);
        ledger.record_rollback(100_000_000);
        let accel = ledger.accelerated_accepted_tps();
        let board = SeparatedTpsScoreboard::new(base, accel);
        assert_eq!(board.base_true.value(), 100.0);
        assert!((board.accelerated_accepted.value() - 70.0).abs() < 1e-9);
        let ratio = board.speedup_ratio().unwrap();
        assert!((ratio - 0.7).abs() < 1e-9);
    }

    #[test]
    fn tg_scoreboard_keeps_block_prefill_ttft_separate() {
        let mut board = SeparatedTgScoreboard::empty();
        assert!(board.base_true.is_none());
        board.base_true = Some(BaseTrueTps::from_counts(50, Duration::from_secs(1)));
        board.block_executed = Some(BlockExecutedTps::from_counts(80, Duration::from_secs(1)));
        board.prefill = Some(PrefillTps::from_counts(200, Duration::from_secs(1)));
        board.ttft = Some(TtftSeconds::from_duration(Duration::from_millis(25)));
        board.accelerated_accepted = Some(AcceleratedAcceptedTps::new(70.0));
        assert_eq!(board.base_true.unwrap().value(), 50.0);
        assert_eq!(board.block_executed.unwrap().value(), 80.0);
        assert_eq!(board.prefill.unwrap().value(), 200.0);
        assert!((board.ttft.unwrap().value() - 0.025).abs() < 1e-9);
        // Block-executed must not silently stand in for base-true.
        assert!(board.block_executed.unwrap().value() > board.base_true.unwrap().value());
        assert_eq!(BlockExecutedTps::SCOREBOARD, "BLOCK_EXECUTED_TPS");
        assert_eq!(PrefillTps::SCOREBOARD, "PREFILL_TPS");
        assert_eq!(TtftSeconds::SCOREBOARD, "TTFT");
        let ratio = board.accel_over_base_ratio().unwrap();
        assert!((ratio - 1.4).abs() < 1e-9);
    }
}
