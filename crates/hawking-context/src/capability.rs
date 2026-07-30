//! Honest context capability declaration (campaign §7.3).
//!
//! Retrieval and compaction extend *usable* context. They do not extend the
//! model's native window. Anything that blurs that line is a misreport.
//!
//! Numbers here are either **measured** (from the live engine / a calibration
//! probe) or explicitly **unmeasured** (`None`). An asserted constant is never
//! promoted to a "validated" claim.

use serde::{Deserialize, Serialize};

/// How a context number was obtained. The distinction is the whole point of
/// this module: a config default is not a measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumberSource {
    /// Read from the live engine / model config at inference time.
    Measured,
    /// Taken from a role/profile config (may be a default). Not a measurement.
    Config,
    /// Derived (e.g. native × `.tq` multiplier). May be estimated.
    Derived,
    /// Explicitly not claimed — the system refuses to invent a number.
    Unmeasured,
}

/// One declared context figure with its provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeclaredNumber {
    pub tokens: Option<usize>,
    pub source: NumberSource,
    /// True when the figure is an estimate (e.g. `.tq` expansion) rather than a
    /// hard cap the model guarantees.
    pub estimated: bool,
    /// Short human reason the figure is what it is (auditable meter).
    pub explanation: String,
}

impl DeclaredNumber {
    pub fn measured(tokens: usize, explanation: impl Into<String>) -> Self {
        Self {
            tokens: Some(tokens),
            source: NumberSource::Measured,
            estimated: false,
            explanation: explanation.into(),
        }
    }

    pub fn config(tokens: usize, explanation: impl Into<String>) -> Self {
        Self {
            tokens: Some(tokens),
            source: NumberSource::Config,
            estimated: false,
            explanation: explanation.into(),
        }
    }

    pub fn derived_estimated(tokens: usize, explanation: impl Into<String>) -> Self {
        Self {
            tokens: Some(tokens),
            source: NumberSource::Derived,
            estimated: true,
            explanation: explanation.into(),
        }
    }

    pub fn unmeasured(explanation: impl Into<String>) -> Self {
        Self {
            tokens: None,
            source: NumberSource::Unmeasured,
            estimated: false,
            explanation: explanation.into(),
        }
    }
}

/// Retrieval / compaction operating mode — declared as a mode, never as a
/// fake extension of the native window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalMode {
    /// Only what fits the native (or effective) window after packing.
    PackOnly,
    /// Lexical + semantic retrieval feeds the packer; still capped by the window.
    RetrieveThenPack,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionMode {
    /// No compaction; drop spans that do not fit.
    DropOnly,
    /// On-the-fly degrade with recall-gated rollback (Spine B).
    DegradeWithRecallGate,
    /// Bounded multi-step compaction chain (depth-capped).
    RecursiveBounded,
}

/// The full honest capability picture for one model / one turn.
///
/// **Law:** `usable` (retrieval + compaction) is never written into
/// `native_maximum`. Validated quality/agentic figures stay `None` until a
/// real measurement produces them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextCapability {
    pub native_maximum: DeclaredNumber,
    /// Effective ceiling after position scaling / `.tq` multiplier. Distinct
    /// from native; may be estimated.
    pub effective_ceiling: DeclaredNumber,
    /// Highest context length where quality was measured good enough. `None`
    /// until a calibration run fills it — never a copied native default.
    pub validated_quality: DeclaredNumber,
    /// Highest context length where agentic (tool-using) quality was measured.
    pub validated_agentic: DeclaredNumber,
    /// Prefill latency/throughput curve — declared only when measured.
    pub prefill_curve: Option<Vec<CurvePoint>>,
    /// KV memory curve — declared only when measured.
    pub kv_curve: Option<Vec<CurvePoint>>,
    pub retrieval_mode: RetrievalMode,
    pub compaction_mode: CompactionMode,
    /// Human-readable lines the context meter can show. Never invents a
    /// native claim from retrieval/compaction.
    pub explanations: Vec<String>,
}

/// One measured (or estimated) point on a prefill/KV curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    pub tokens: usize,
    /// Prefill ms, or KV bytes, depending on the curve.
    pub value: f64,
    pub unit: String,
    pub estimated: bool,
}

impl ContextCapability {
    /// Build from a role/config native and an optional live engine snapshot.
    ///
    /// * Live measured native wins over the config default.
    /// * Effective = live effective when present, else native (no invented expansion).
    /// * Validated quality/agentic and curves stay unmeasured unless supplied.
    /// * Retrieval/compaction modes are declared as modes, not token counts.
    pub fn declare(
        config_native: usize,
        live_native: Option<usize>,
        live_effective: Option<usize>,
        tq_multiplier: Option<f32>,
        tq_estimated: bool,
        retrieval: RetrievalMode,
        compaction: CompactionMode,
    ) -> Self {
        let (native, native_src_note) = match live_native {
            Some(n) if n > 0 => (
                DeclaredNumber::measured(n, "native maximum from live engine /v1/hawking/context"),
                "native is measured live",
            ),
            _ => (
                DeclaredNumber::config(
                    config_native,
                    "native maximum from role/config (not live-measured this turn)",
                ),
                "native is config (unmeasured this turn)",
            ),
        };

        let native_tokens = native.tokens.unwrap_or(config_native);
        let effective = match live_effective {
            Some(e) if e > 0 && e != native_tokens => {
                let mult = tq_multiplier.unwrap_or(1.0);
                DeclaredNumber {
                    tokens: Some(e),
                    source: NumberSource::Derived,
                    estimated: tq_estimated || mult > 1.0,
                    explanation: format!(
                        "effective ceiling = live effective (×{mult:.2} .tq); not a native-window claim"
                    ),
                }
            }
            Some(e) if e > 0 => DeclaredNumber::measured(
                e,
                "effective ceiling equals measured native (no expansion claimed)",
            ),
            _ => DeclaredNumber::config(
                native_tokens,
                "effective ceiling defaults to native; no .tq expansion applied",
            ),
        };

        let mut explanations =
            vec![
            format!(
                "native_maximum={} ({})",
                native.tokens.map(|t| t.to_string()).unwrap_or_else(|| "none".into()),
                native_src_note
            ),
            format!(
                "effective_ceiling={} (source={:?}, estimated={})",
                effective
                    .tokens
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "none".into()),
                effective.source,
                effective.estimated
            ),
            "retrieval and compaction extend usable context only; they do not raise native_maximum"
                .into(),
            format!("retrieval_mode={retrieval:?}"),
            format!("compaction_mode={compaction:?}"),
            "validated_quality and validated_agentic are unmeasured (no fake numbers)".into(),
            "kv_curve and prefill_curve are unmeasured (no fake numbers)".into(),
        ];

        if let (Some(n), Some(e)) = (native.tokens, effective.tokens) {
            if e > n {
                explanations.push(format!(
                    "effective ({e}) > native ({n}): the surplus is position-scaling/.tq, not a larger native window"
                ));
            }
        }

        Self {
            native_maximum: native,
            effective_ceiling: effective,
            validated_quality: DeclaredNumber::unmeasured(
                "no quality-context calibration has been recorded for this model",
            ),
            validated_agentic: DeclaredNumber::unmeasured(
                "no agentic-context calibration has been recorded for this model",
            ),
            prefill_curve: None,
            kv_curve: None,
            retrieval_mode: retrieval,
            compaction_mode: compaction,
            explanations,
        }
    }

    /// Token budget the packer may fill. Prefer measured native; never treat
    /// an unvalidated effective expansion as a free pass to overfill when the
    /// caller asks for a conservative pack.
    pub fn pack_budget_tokens(&self, prefer_effective: bool) -> usize {
        if prefer_effective {
            self.effective_ceiling
                .tokens
                .or(self.native_maximum.tokens)
                .unwrap_or(0)
        } else {
            self.native_maximum
                .tokens
                .or(self.effective_ceiling.tokens)
                .unwrap_or(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_native_beats_config_and_effective_is_not_native() {
        let cap = ContextCapability::declare(
            8192,
            Some(4096),
            Some(16384),
            Some(4.0),
            true,
            RetrievalMode::RetrieveThenPack,
            CompactionMode::DegradeWithRecallGate,
        );
        assert_eq!(cap.native_maximum.tokens, Some(4096));
        assert_eq!(cap.native_maximum.source, NumberSource::Measured);
        assert_eq!(cap.effective_ceiling.tokens, Some(16384));
        assert!(cap.effective_ceiling.estimated);
        assert_ne!(cap.native_maximum.tokens, cap.effective_ceiling.tokens);
        assert!(cap.validated_quality.tokens.is_none());
        assert!(cap.validated_agentic.tokens.is_none());
        assert!(cap.kv_curve.is_none());
        assert!(cap.prefill_curve.is_none());
        assert!(cap
            .explanations
            .iter()
            .any(|e| e.contains("do not raise native_maximum")));
    }
    #[test]
    fn unmeasured_validated_numbers_are_none_not_copied_native() {
        let cap = ContextCapability::declare(
            8192,
            None,
            None,
            None,
            false,
            RetrievalMode::PackOnly,
            CompactionMode::DropOnly,
        );
        assert_eq!(cap.native_maximum.tokens, Some(8192));
        assert_eq!(cap.native_maximum.source, NumberSource::Config);
        assert!(cap.validated_quality.tokens.is_none());
        assert!(cap.validated_agentic.tokens.is_none());
    }
    #[test]
    fn pack_budget_prefers_native_when_conservative() {
        let cap = ContextCapability::declare(
            8192,
            Some(4096),
            Some(16384),
            Some(4.0),
            true,
            RetrievalMode::RetrieveThenPack,
            CompactionMode::DegradeWithRecallGate,
        );
        assert_eq!(cap.pack_budget_tokens(false), 4096);
        assert_eq!(cap.pack_budget_tokens(true), 16384);
    }
}
