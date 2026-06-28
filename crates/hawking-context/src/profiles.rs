//! Per-task context profiles — the user's "dial" (bible §4.9, Appendix A.3).
//!
//! A profile bundles every knob (window, position policy, working-set mode,
//! eviction, KV precision, reservations, source weights, recency half-life,
//! ordering, retrieval_k, improve_iters) into a named preset. `Tight`,
//! `Standard`, `Long`, `Unbounded` are the reserved built-ins.

use crate::budget::{RegionBudget, TokenBudget};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Position-index scaling policy (bible §4.4.1). Recorded on the manifest; the
/// runtime applies it (shell records the choice today).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum PositionPolicy {
    Native,
    Pi { scale: f32 },
    Yarn { scale: f32, attn_temp: f32 },
    DynamicNtk { trained_len: usize },
}

impl PositionPolicy {
    /// YaRN attention temperature default: `0.1·ln(scale) + 1` (bible §4.4.1).
    pub fn yarn(scale: f32) -> Self {
        PositionPolicy::Yarn {
            scale,
            attn_temp: 0.1 * scale.max(1.0).ln() + 1.0,
        }
    }

    pub fn label(&self) -> String {
        match self {
            PositionPolicy::Native => "native".to_string(),
            PositionPolicy::Pi { scale } => format!("pi(scale={scale})"),
            PositionPolicy::Yarn { scale, attn_temp } => {
                format!("yarn(scale={scale},attn_temp={attn_temp:.2})")
            }
            PositionPolicy::DynamicNtk { trained_len } => {
                format!("dynamic_ntk(trained_len={trained_len})")
            }
        }
    }
}

/// The working-set mode (mirrors the in-tree `WorkingSetMode`, bible §4.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkingSetMode {
    Lossless,
    Bounded,
}

/// Eviction policy choice (bible §4.5.3). Profile-selectable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum EvictionChoice {
    Lossless,
    StreamingLlm { sinks: usize, recent: usize },
    SnapKv { keep: usize },
    H2o { recent: usize, heavy: usize },
}

/// KV precision codec (bible §4.5.3 / §4.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvPrecision {
    Native,
    F16,
    Int4,
}

/// Per-`SourceKind` scoring weights for the value blend (bible §4.2.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceWeights {
    pub w_band: f32,
    pub w_relevance: f32,
    pub w_recency: f32,
    pub w_importance: f32,
    pub w_redundancy: f32,
    /// Per-`ContextSourceKind` band multipliers ("debug profile weights
    /// diagnostics", "refactor weights symbols"); keyed by the kind's debug
    /// name. Missing kinds use 1.0.
    #[serde(default)]
    pub band_by_kind: BTreeMap<String, f32>,
}

impl Default for SourceWeights {
    fn default() -> Self {
        // Generative-Agents default α=1 for the three signals; band carries
        // the source's declared importance; redundancy is the anti-rot penalty.
        Self {
            w_band: 1.0,
            w_relevance: 1.0,
            w_recency: 0.5,
            w_importance: 0.75,
            w_redundancy: 1.0,
            band_by_kind: BTreeMap::new(),
        }
    }
}

/// Head/tail ordering policy to defeat lost-in-the-middle (bible §4.2.3, F3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderingPolicy {
    /// Spans in selection (value) order, no repositioning.
    AsScored,
    /// High-value spans pinned to the head and tail; filler buried in the
    /// middle. The anti-LITM default.
    HeadTail,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextProfile {
    pub name: String,
    pub budget: TokenBudget,
    pub regions: Vec<RegionBudget>,
    pub preserve_document_order: bool,
    pub pin_system_to_head: bool,
    pub pin_recent_to_tail: bool,
    // --- A.3 fields ---
    #[serde(default = "default_position_policy")]
    pub position_policy: PositionPolicy,
    #[serde(default = "default_working_set_mode")]
    pub working_set_mode: WorkingSetMode,
    #[serde(default = "default_eviction")]
    pub eviction: EvictionChoice,
    #[serde(default = "default_kv_precision")]
    pub kv_precision: KvPrecision,
    /// Fractions of the available input reserved for system/response/scratchpad.
    #[serde(default = "default_reservation_pcts")]
    pub reservation_pcts: ReservationPcts,
    #[serde(default)]
    pub source_weights: SourceWeights,
    #[serde(default = "default_recency_half_life_ms")]
    pub recency_half_life_ms: u64,
    #[serde(default = "default_ordering")]
    pub ordering: OrderingPolicy,
    #[serde(default = "default_retrieval_k")]
    pub retrieval_k: usize,
    #[serde(default = "default_improve_iters")]
    pub improve_iters: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReservationPcts {
    pub system: f32,
    pub response: f32,
    pub scratchpad: f32,
}

fn default_position_policy() -> PositionPolicy {
    PositionPolicy::Native
}
fn default_working_set_mode() -> WorkingSetMode {
    WorkingSetMode::Lossless
}
fn default_eviction() -> EvictionChoice {
    EvictionChoice::Lossless
}
fn default_kv_precision() -> KvPrecision {
    KvPrecision::Native
}
fn default_reservation_pcts() -> ReservationPcts {
    ReservationPcts {
        system: 0.08,
        response: 0.20,
        scratchpad: 0.06,
    }
}
fn default_recency_half_life_ms() -> u64 {
    // 30 minutes — tool outputs decay over a session (bible §4.2.2).
    30 * 60 * 1000
}
fn default_ordering() -> OrderingPolicy {
    OrderingPolicy::HeadTail
}
fn default_retrieval_k() -> usize {
    16
}
fn default_improve_iters() -> usize {
    8
}

impl ContextProfile {
    fn regions(max_input_tokens: usize) -> Vec<RegionBudget> {
        vec![
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
        ]
    }

    /// The day-to-day default (kept name-stable for siblings). Equivalent to
    /// the `Standard` preset sized to `max_input_tokens`.
    pub fn coding_default(max_input_tokens: usize) -> Self {
        Self::standard(max_input_tokens)
    }

    /// Tight: quick edits, single file, lowest latency. Native window, lossless.
    pub fn tight(max_input_tokens: usize) -> Self {
        Self {
            name: "tight".to_string(),
            budget: TokenBudget {
                max_input_tokens,
                reserve_output_tokens: 1_024.min(max_input_tokens / 4),
                hard_limit_tokens: max_input_tokens,
            },
            regions: Self::regions(max_input_tokens),
            preserve_document_order: true,
            pin_system_to_head: true,
            pin_recent_to_tail: true,
            position_policy: PositionPolicy::Native,
            working_set_mode: WorkingSetMode::Lossless,
            eviction: EvictionChoice::Lossless,
            kv_precision: KvPrecision::Native,
            reservation_pcts: default_reservation_pcts(),
            source_weights: SourceWeights::default(),
            recency_half_life_ms: 10 * 60 * 1000,
            ordering: OrderingPolicy::HeadTail,
            retrieval_k: 6,
            improve_iters: 4,
        }
    }

    /// Standard: day-to-day agentic coding (default).
    pub fn standard(max_input_tokens: usize) -> Self {
        Self {
            name: "standard".to_string(),
            budget: TokenBudget {
                max_input_tokens,
                reserve_output_tokens: 2_048.min(max_input_tokens / 4),
                hard_limit_tokens: max_input_tokens,
            },
            regions: Self::regions(max_input_tokens),
            preserve_document_order: true,
            pin_system_to_head: true,
            pin_recent_to_tail: true,
            position_policy: PositionPolicy::Native,
            working_set_mode: WorkingSetMode::Lossless,
            eviction: EvictionChoice::Lossless,
            kv_precision: KvPrecision::F16,
            reservation_pcts: default_reservation_pcts(),
            source_weights: SourceWeights::default(),
            recency_half_life_ms: default_recency_half_life_ms(),
            ordering: OrderingPolicy::HeadTail,
            retrieval_k: 16,
            improve_iters: 8,
        }
    }

    /// Long: multi-file refactor, large files, long sessions. Mild YaRN +
    /// SnapKV bounded working set.
    pub fn long(max_input_tokens: usize) -> Self {
        Self {
            name: "long".to_string(),
            budget: TokenBudget {
                max_input_tokens,
                reserve_output_tokens: 4_096.min(max_input_tokens / 4),
                hard_limit_tokens: max_input_tokens,
            },
            regions: Self::regions(max_input_tokens),
            preserve_document_order: true,
            pin_system_to_head: true,
            pin_recent_to_tail: true,
            position_policy: PositionPolicy::yarn(4.0),
            working_set_mode: WorkingSetMode::Bounded,
            eviction: EvictionChoice::SnapKv {
                keep: max_input_tokens / 2,
            },
            kv_precision: KvPrecision::F16,
            reservation_pcts: ReservationPcts {
                system: 0.06,
                response: 0.16,
                scratchpad: 0.06,
            },
            source_weights: SourceWeights::default(),
            recency_half_life_ms: 2 * 60 * 60 * 1000,
            ordering: OrderingPolicy::HeadTail,
            retrieval_k: 24,
            improve_iters: 12,
        }
    }

    /// Unbounded: "watch this stream / follow this long narrative". StreamingLLM
    /// sinks (or SSM route, recorded on the model descriptor).
    pub fn unbounded(max_input_tokens: usize) -> Self {
        Self {
            name: "unbounded".to_string(),
            budget: TokenBudget {
                max_input_tokens,
                reserve_output_tokens: 4_096.min(max_input_tokens / 4),
                hard_limit_tokens: max_input_tokens,
            },
            regions: Self::regions(max_input_tokens),
            preserve_document_order: true,
            pin_system_to_head: true,
            pin_recent_to_tail: true,
            position_policy: PositionPolicy::Native,
            working_set_mode: WorkingSetMode::Bounded,
            eviction: EvictionChoice::StreamingLlm {
                sinks: 4,
                recent: max_input_tokens / 2,
            },
            kv_precision: KvPrecision::Int4,
            reservation_pcts: ReservationPcts {
                system: 0.05,
                response: 0.16,
                scratchpad: 0.08,
            },
            source_weights: SourceWeights::default(),
            recency_half_life_ms: 4 * 60 * 60 * 1000,
            ordering: OrderingPolicy::HeadTail,
            retrieval_k: 32,
            improve_iters: 8,
        }
    }

    /// Look up a reserved preset by dial name; `None` for custom names.
    pub fn preset(name: &str, max_input_tokens: usize) -> Option<Self> {
        match name {
            "tight" => Some(Self::tight(max_input_tokens)),
            "standard" | "coding_default" => Some(Self::standard(max_input_tokens)),
            "long" => Some(Self::long(max_input_tokens)),
            "unbounded" => Some(Self::unbounded(max_input_tokens)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_resolve_and_differ() {
        let tight = ContextProfile::tight(8192);
        let unbounded = ContextProfile::unbounded(8192);
        assert_eq!(tight.eviction, EvictionChoice::Lossless);
        assert!(matches!(
            unbounded.eviction,
            EvictionChoice::StreamingLlm { sinks: 4, .. }
        ));
        assert!(ContextProfile::preset("long", 8192).is_some());
        assert!(ContextProfile::preset("nope", 8192).is_none());
    }

    #[test]
    fn yarn_temp_follows_formula() {
        if let PositionPolicy::Yarn { scale, attn_temp } = PositionPolicy::yarn(4.0) {
            assert_eq!(scale, 4.0);
            assert!((attn_temp - (0.1 * 4.0_f32.ln() + 1.0)).abs() < 1e-6);
        } else {
            panic!("expected yarn");
        }
    }
}
