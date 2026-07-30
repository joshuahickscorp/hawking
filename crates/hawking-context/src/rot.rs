//! Context-rot detection (campaign §7.3).
//!
//! The agent loop needs a way to notice that its own context has degraded —
//! high occupancy, soft recall, aggressive compaction, redundant packing —
//! before the next turn silently answers from rot. This module is pure: it
//! reads a compiled / live manifest and returns a structured report the host
//! can publish and the loop can act on.

use crate::manifest::{CompactionEvent, ContextManifest, ContextSpan, WatermarkLevel};
use serde::{Deserialize, Serialize};

/// How badly the context has degraded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RotSeverity {
    /// No rot signals.
    Clean,
    /// Soft hints (watermark soft, mild redundancy).
    Watch,
    /// Material degradation; the loop should prefer compact-or-refresh.
    Degraded,
    /// Context is not trustworthy for further work without a reset/refresh.
    Critical,
}

/// One concrete reason the detector fired.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RotSignal {
    pub code: String,
    pub severity: RotSeverity,
    pub detail: String,
}

/// Full rot report: severity, signals, and meter-ready explanations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextRotReport {
    pub severity: RotSeverity,
    pub signals: Vec<RotSignal>,
    /// Human lines for the context meter (auditable).
    pub explanations: Vec<String>,
    /// True when the loop should treat further answers as at-risk.
    pub should_refresh: bool,
}

impl ContextRotReport {
    pub fn clean() -> Self {
        Self {
            severity: RotSeverity::Clean,
            signals: Vec::new(),
            explanations: vec!["context rot: clean (no degradation signals)".into()],
            should_refresh: false,
        }
    }
}

/// Thresholds for the pure detector. Defaults match Spine A watermarks and
/// Spine B recall floors so the two systems agree.
#[derive(Debug, Clone, Copy)]
pub struct RotThresholds {
    /// Mean redundancy across retained spans that trips Watch.
    pub redundancy_watch: f32,
    /// Mean redundancy that trips Degraded.
    pub redundancy_degraded: f32,
    /// Fraction of retained tokens that came from compaction.
    pub compacted_frac_watch: f32,
    /// Mean compaction recall below which we flag.
    pub compaction_recall_floor: f32,
    /// Live occupancy (or 1 - fidelity) that trips Critical.
    pub occupancy_critical: f32,
    /// Live occupancy that trips Degraded.
    pub occupancy_degraded: f32,
}

impl Default for RotThresholds {
    fn default() -> Self {
        Self {
            redundancy_watch: 0.45,
            redundancy_degraded: 0.70,
            compacted_frac_watch: 0.35,
            compaction_recall_floor: 0.85,
            occupancy_critical: 0.90,
            occupancy_degraded: 0.75,
        }
    }
}

/// Detect context rot from a compiled manifest and optional live occupancy.
///
/// Pure and deterministic. Empty retained + no live data => Clean.
pub fn detect_context_rot(
    manifest: &ContextManifest,
    live_occupancy: Option<f32>,
    live_watermark: Option<WatermarkLevel>,
    live_recall_fidelity: Option<f32>,
    thresholds: RotThresholds,
) -> ContextRotReport {
    let mut signals: Vec<RotSignal> = Vec::new();

    // --- Live occupancy / watermark / recall fidelity ---
    if let Some(occ) = live_occupancy {
        if occ >= thresholds.occupancy_critical {
            signals.push(RotSignal {
                code: "occupancy_critical".into(),
                severity: RotSeverity::Critical,
                detail: format!(
                    "live occupancy {occ:.2} ≥ {:.2} (headroom nearly gone)",
                    thresholds.occupancy_critical
                ),
            });
        } else if occ >= thresholds.occupancy_degraded {
            signals.push(RotSignal {
                code: "occupancy_high".into(),
                severity: RotSeverity::Degraded,
                detail: format!(
                    "live occupancy {occ:.2} ≥ {:.2} (context pressure)",
                    thresholds.occupancy_degraded
                ),
            });
        }
    }
    if let Some(wm) = live_watermark {
        let sev = match wm {
            WatermarkLevel::Critical => RotSeverity::Critical,
            WatermarkLevel::Warn => RotSeverity::Degraded,
            WatermarkLevel::Soft => RotSeverity::Watch,
            WatermarkLevel::Normal => RotSeverity::Clean,
        };
        if sev > RotSeverity::Clean {
            signals.push(RotSignal {
                code: format!("watermark_{}", format!("{wm:?}").to_lowercase()),
                severity: sev,
                detail: format!("live watermark band is {wm:?}"),
            });
        }
    }
    if let Some(fid) = live_recall_fidelity {
        if fid < 0.5 {
            signals.push(RotSignal {
                code: "recall_soft".into(),
                severity: RotSeverity::Critical,
                detail: format!("SSM recall fidelity {fid:.2} < 0.50 (state is soft)"),
            });
        } else if fid < 0.7 {
            signals.push(RotSignal {
                code: "recall_softening".into(),
                severity: RotSeverity::Degraded,
                detail: format!("SSM recall fidelity {fid:.2} < 0.70"),
            });
        }
    }

    // --- Packing quality: redundancy among retained ---
    if let Some(mean_red) = mean_redundancy(&manifest.retained) {
        if mean_red >= thresholds.redundancy_degraded {
            signals.push(RotSignal {
                code: "redundant_pack".into(),
                severity: RotSeverity::Degraded,
                detail: format!(
                    "mean retained redundancy {mean_red:.2} ≥ {:.2}",
                    thresholds.redundancy_degraded
                ),
            });
        } else if mean_red >= thresholds.redundancy_watch {
            signals.push(RotSignal {
                code: "redundant_pack_watch".into(),
                severity: RotSeverity::Watch,
                detail: format!(
                    "mean retained redundancy {mean_red:.2} ≥ {:.2}",
                    thresholds.redundancy_watch
                ),
            });
        }
    }

    // --- Compaction pressure ---
    let compacted_frac = compacted_token_fraction(&manifest.retained);
    if compacted_frac >= thresholds.compacted_frac_watch {
        signals.push(RotSignal {
            code: "heavy_compaction".into(),
            severity: RotSeverity::Watch,
            detail: format!(
                "{:.0}% of retained tokens came from compaction",
                compacted_frac * 100.0
            ),
        });
    }
    if let Some(mean_recall) = mean_compaction_recall(&manifest.compaction_events) {
        if mean_recall < thresholds.compaction_recall_floor {
            signals.push(RotSignal {
                code: "compaction_recall_low".into(),
                severity: RotSeverity::Degraded,
                detail: format!(
                    "mean post-compaction recall {mean_recall:.2} < {:.2}",
                    thresholds.compaction_recall_floor
                ),
            });
        }
    }
    let rolled_back = manifest
        .compaction_events
        .iter()
        .filter(|e| e.rolled_back)
        .count();
    if rolled_back > 0 {
        signals.push(RotSignal {
            code: "compaction_rollbacks".into(),
            severity: RotSeverity::Watch,
            detail: format!(
                "{rolled_back} compaction(s) rolled back by the recall gate (evidence preserved)"
            ),
        });
    }

    // --- Drop pressure: many important drops ---
    let dropped_n = manifest.dropped.len();
    let retained_n = manifest.retained.len();
    if dropped_n > 0 && retained_n > 0 && dropped_n as f32 / (dropped_n + retained_n) as f32 > 0.5 {
        signals.push(RotSignal {
            code: "heavy_drops".into(),
            severity: RotSeverity::Watch,
            detail: format!(
                "{dropped_n} spans dropped vs {retained_n} retained (>50% candidates lost)"
            ),
        });
    }

    let severity = signals
        .iter()
        .map(|s| s.severity)
        .max()
        .unwrap_or(RotSeverity::Clean);

    let mut explanations: Vec<String> = if signals.is_empty() {
        vec!["context rot: clean (no degradation signals)".into()]
    } else {
        signals
            .iter()
            .map(|s| format!("[{:?}] {}: {}", s.severity, s.code, s.detail))
            .collect()
    };
    explanations.push(format!("context rot severity: {severity:?}"));

    ContextRotReport {
        should_refresh: severity >= RotSeverity::Degraded,
        severity,
        signals,
        explanations,
    }
}

fn mean_redundancy(retained: &[ContextSpan]) -> Option<f32> {
    if retained.is_empty() {
        return None;
    }
    let sum: f32 = retained.iter().map(|s| s.signals.redundancy).sum();
    Some(sum / retained.len() as f32)
}

fn compacted_token_fraction(retained: &[ContextSpan]) -> f32 {
    let total: usize = retained.iter().map(|s| s.token_count).sum();
    if total == 0 {
        return 0.0;
    }
    let compacted: usize = retained
        .iter()
        .filter(|s| s.compacted_from.is_some())
        .map(|s| s.token_count)
        .sum();
    compacted as f32 / total as f32
}

fn mean_compaction_recall(events: &[CompactionEvent]) -> Option<f32> {
    let scores: Vec<f32> = events.iter().filter_map(|e| e.recall_score).collect();
    if scores.is_empty() {
        return None;
    }
    Some(scores.iter().sum::<f32>() / scores.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ContextManifest, SpanSignals};
    fn span(redundancy: f32, compacted: bool, tokens: usize) -> ContextSpan {
        use crate::manifest::{CompactedFrom, ContextSourceKind, PinState};
        use hide_core::types::Provenance;
        ContextSpan {
            id: "s".into(),
            source: ContextSourceKind::Code,
            title: "t".into(),
            text: "body".into(),
            order_index: 0,
            token_count: tokens,
            score: 0.5,
            signals: SpanSignals {
                recency: 0.5,
                importance: 0.5,
                relevance: 0.5,
                redundancy,
            },
            pin: PinState::Normal,
            banked: false,
            compacted_from: if compacted {
                Some(CompactedFrom {
                    original_id: "o".into(),
                    method: "degrade".into(),
                    ratio: 0.3,
                    depth: 1,
                })
            } else {
                None
            },
            provenance: Provenance::trusted("test"),
            blob_ref: None,
        }
    }
    #[test]
    fn clean_when_empty() {
        let m = ContextManifest::new(4096);
        let r = detect_context_rot(&m, None, None, None, RotThresholds::default());
        assert_eq!(r.severity, RotSeverity::Clean);
        assert!(!r.should_refresh);
    }
    #[test]
    fn critical_occupancy_forces_refresh() {
        let m = ContextManifest::new(4096);
        let r = detect_context_rot(
            &m,
            Some(0.95),
            Some(WatermarkLevel::Critical),
            None,
            RotThresholds::default(),
        );
        assert_eq!(r.severity, RotSeverity::Critical);
        assert!(r.should_refresh);
        assert!(r.explanations.iter().any(|e| e.contains("occupancy")));
    }
    #[test]
    fn redundant_pack_is_degraded() {
        let mut m = ContextManifest::new(4096);
        m.retained = vec![span(0.8, false, 100), span(0.75, false, 100)];
        let r = detect_context_rot(
            &m,
            Some(0.2),
            Some(WatermarkLevel::Normal),
            None,
            RotThresholds::default(),
        );
        assert!(r.severity >= RotSeverity::Degraded);
        assert!(r.signals.iter().any(|s| s.code.starts_with("redundant")));
    }
    #[test]
    fn soft_recall_is_critical() {
        let m = ContextManifest::new(4096);
        let r = detect_context_rot(&m, None, None, Some(0.4), RotThresholds::default());
        assert_eq!(r.severity, RotSeverity::Critical);
        assert!(r.should_refresh);
    }
}
