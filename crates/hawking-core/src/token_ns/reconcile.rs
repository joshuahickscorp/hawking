//! Lane-claim vs whole-token reconciler.
//!
//! `sum(ns saved by all lanes)` is not `(token_before − token_after)`.
//! A lane whose win does not appear in the whole token optimized overlapped
//! time, moved the cost sideways, measured off the critical path, or
//! measured something that is not a token.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscrepancyKind {
    /// Claimed delta is visible in the whole-token measurement.
    AppearsInToken,
    /// Stage got cheaper but that time was already hidden under a longer
    /// serial neighbour, so the token did not move.
    OverlappedTime,
    /// Stage went down and a neighbour went up by a similar amount.
    CostMovedSideways,
    /// The measured object is not on the token critical path.
    OffCriticalPath,
    /// A parallel-thread sum was treated as token latency.
    ParallelSumPresentedAsLatency,
    /// Cold start, unpaired run, wrong unit, or poisoned median.
    MeasurementArtifact,
    /// Isolated organ / microbench product labeled as ns/token.
    IsolatedMicrobenchNotToken,
    /// The lane named a cost but did not remove it.
    IdentifiedNotSaved,
    /// No whole-token after exists yet.
    NotMeasured,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LaneClaim {
    pub lane: String,
    pub model: String,
    pub claimed_ns_saved: i64,
    pub claim_source: String,
    pub claim_kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MeasuredToken {
    pub model: String,
    pub before_ns: u64,
    pub after_ns: Option<u64>,
    pub source: String,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LaneReconciliation {
    pub lane: String,
    pub model: String,
    pub claimed_ns_saved: i64,
    pub measured_token_delta_ns: Option<i64>,
    pub discrepancy_ns: Option<i64>,
    pub kind: DiscrepancyKind,
    pub why: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelReconciliation {
    pub model: String,
    pub token_before_ns: u64,
    pub token_after_ns: Option<u64>,
    pub measured_delta_ns: i64,
    pub sum_claimed_ns: i64,
    pub discrepancy_ns: i64,
    pub identity_note: &'static str,
    pub lanes: Vec<LaneReconciliation>,
    pub honesty: String,
}

pub fn reconcile(claims: &[LaneClaim], measured: &MeasuredToken) -> ModelReconciliation {
    let measured_delta_ns = match measured.after_ns {
        Some(after) => measured.before_ns as i64 - after as i64,
        None => 0,
    };
    let model_claims: Vec<&LaneClaim> = claims
        .iter()
        .filter(|c| c.model == measured.model)
        .collect();
    let sum_claimed_ns: i64 = model_claims.iter().map(|c| c.claimed_ns_saved).sum();
    let lanes = model_claims
        .iter()
        .map(|c| LaneReconciliation {
            lane: c.lane.clone(),
            model: c.model.clone(),
            claimed_ns_saved: c.claimed_ns_saved,
            measured_token_delta_ns: measured.after_ns.map(|_| measured_delta_ns),
            discrepancy_ns: Some(c.claimed_ns_saved - measured_delta_ns),
            kind: DiscrepancyKind::NotMeasured,
            why: c.claim_source.clone(),
        })
        .collect();
    ModelReconciliation {
        model: measured.model.clone(),
        token_before_ns: measured.before_ns,
        token_after_ns: measured.after_ns,
        measured_delta_ns,
        sum_claimed_ns,
        discrepancy_ns: sum_claimed_ns - measured_delta_ns,
        identity_note: "sum(claimed lane ns) is NOT (token_before - token_after)",
        lanes,
        honesty: format!(
            "sum_claimed={} ns, measured_token_delta={} ns, discrepancy={} ns",
            sum_claimed_ns, measured_delta_ns, sum_claimed_ns - measured_delta_ns
        ),
    }
}

/// Annotate a generic reconciliation with the per-lane verdict. The kind is
/// supplied by the caller — the library will not invent a story from a
/// number alone.
pub fn with_kinds(
    mut report: ModelReconciliation,
    kinds: &[(&str, DiscrepancyKind, &str)],
) -> ModelReconciliation {
    for lane in &mut report.lanes {
        if let Some((_, kind, why)) = kinds.iter().find(|(name, _, _)| *name == lane.lane) {
            lane.kind = *kind;
            lane.why = why.to_string();
        }
    }
    report
}

/// Standing 2026-08-16 claims taken from ASCENT_STATE.json plus the
/// receipt-level asserted deltas. Numbers are copied, not invented.
pub fn ascent_2026_08_16_claims() -> Vec<LaneClaim> {
    vec![
        LaneClaim {
            lane: "dsv-host-wall".into(),
            model: "dsv4f".into(),
            claimed_ns_saved: 500_000_000,
            claim_source: "ASCENT_STATE targets[].recoverable_ns_per_token".into(),
            claim_kind: "recoverable_hypothesis".into(),
        },
        LaneClaim {
            lane: "dsv-cb-collapse".into(),
            model: "dsv4f".into(),
            claimed_ns_saved: 300_000_000,
            claim_source: "ASCENT_STATE recoverable_ns_per_token; receipt also asserts 2264→1464 ms body".into(),
            claim_kind: "recoverable_hypothesis_and_receipt_result".into(),
        },
        LaneClaim {
            lane: "dsv-resident-gravity".into(),
            model: "dsv4f".into(),
            claimed_ns_saved: 900_000_000,
            claim_source: "ASCENT_STATE recoverable_ns_per_token".into(),
            claim_kind: "recoverable_hypothesis".into(),
        },
        LaneClaim {
            lane: "dsv-mla".into(),
            model: "dsv4f".into(),
            claimed_ns_saved: 150_000_000,
            claim_source: "ASCENT_STATE recoverable_ns_per_token; receipt attention GPU 198→128 ms".into(),
            claim_kind: "recoverable_hypothesis_and_component".into(),
        },
        LaneClaim {
            lane: "dsv-expert".into(),
            model: "dsv4f".into(),
            claimed_ns_saved: 200_000_000,
            claim_source: "ASCENT_STATE recoverable_ns_per_token; receipt slab_io 481→262 ms".into(),
            claim_kind: "recoverable_hypothesis_and_component".into(),
        },
        LaneClaim {
            lane: "dsv-expert-cache".into(),
            model: "dsv4f".into(),
            claimed_ns_saved: 415_126_416,
            claim_source: "ASCENT_STATE recoverable_ns_per_token (host.expert_slab_io)".into(),
            claim_kind: "recoverable_hypothesis".into(),
        },
        LaneClaim {
            lane: "dsv-admission-identity".into(),
            model: "dsv4f".into(),
            claimed_ns_saved: 80_000_000,
            claim_source: "ASCENT_STATE recoverable_ns_per_token".into(),
            claim_kind: "recoverable_hypothesis".into(),
        },
        LaneClaim {
            lane: "dsv-integrate-graph".into(),
            model: "dsv4f".into(),
            claimed_ns_saved: 289_000_000,
            claim_source: "ASCENT_STATE recoverable_ns_per_token (sum of other DSV lanes)".into(),
            claim_kind: "recoverable_hypothesis".into(),
        },
        LaneClaim {
            lane: "q80-runtime-residency".into(),
            model: "q80".into(),
            claimed_ns_saved: 269_000_000,
            claim_source: "ASCENT_STATE recoverable_ns_per_token".into(),
            claim_kind: "recoverable_hypothesis".into(),
        },
        LaneClaim {
            lane: "q80-deltanet-gqa2".into(),
            model: "q80".into(),
            claimed_ns_saved: 180_000_000,
            claim_source: "ASCENT_STATE recoverable_ns_per_token".into(),
            claim_kind: "recoverable_hypothesis".into(),
        },
        LaneClaim {
            lane: "q80-decode-throughput".into(),
            model: "q80".into(),
            claimed_ns_saved: 150_000_000,
            claim_source: "ASCENT_STATE recoverable_ns_per_token".into(),
            claim_kind: "recoverable_hypothesis".into(),
        },
        LaneClaim {
            lane: "q80-decode-kernels".into(),
            model: "q80".into(),
            claimed_ns_saved: 211_000_000,
            claim_source: "q80-decode-kernels.json result_ns_per_token=192ms vs baseline 403ms".into(),
            claim_kind: "receipt_result_not_a_token".into(),
        },
    ]
}

pub fn ascent_2026_08_16_measured() -> Vec<MeasuredToken> {
    vec![
        MeasuredToken {
            model: "dsv4f".into(),
            before_ns: 1_037_764_208,
            after_ns: Some(1_037_764_208),
            source: "DSV4F_HOST_WALL_BASELINE.json warm R2-R6 body median. No faster whole-token on main.".into(),
            label: "DIRTY_ENGINEERING".into(),
        },
        MeasuredToken {
            model: "q80".into(),
            before_ns: 559_171_655,
            after_ns: Some(559_171_655),
            source: "QWEN80_TOKEN_NS_LEDGER.json decode wall mean. Registered 403 ms incumbent is P1-invalid and is not used as the before.".into(),
            label: "DIRTY_ENGINEERING".into(),
        },
    ]
}

pub fn ascent_2026_08_16_kinds(model: &str) -> Vec<(&'static str, DiscrepancyKind, &'static str)> {
    match model {
        "dsv4f" => vec![
            (
                "dsv-host-wall",
                DiscrepancyKind::IdentifiedNotSaved,
                "Named host.expert_slab_io at 415 ms. Did not remove it. That 415 ms is still the token wall.",
            ),
            (
                "dsv-cb-collapse",
                DiscrepancyKind::MeasurementArtifact,
                "Receipt median 2265→1464 ms is cold-contaminated (pair1A 12.9 s). Warm B bodies 1447/1716/1447 ms are SLOWER than the 1024–1067 ms host-wall authority. The 35 ms GPU movement is real and does not appear in the authority token. Off the critical path relative to expert_slab_io.",
            ),
            (
                "dsv-resident-gravity",
                DiscrepancyKind::NotMeasured,
                "Architectural claim that residency deletes streaming. No packed artifact, no token measurement. Arithmetic only.",
            ),
            (
                "dsv-mla",
                DiscrepancyKind::OverlappedTime,
                "Attention GPU 198→128 ms on isolated/paired GPU. Host expert_slab_io (~415 ms, GPU-idle) is still the token wall. A GPU win under an IO wait does not move body_ns.",
            ),
            (
                "dsv-expert",
                DiscrepancyKind::CostMovedSideways,
                "slab_io 481→262 ms is a real component movement on the serial wall, but one of three body pairs flipped (1404>1278) and the branch is not on main. Whole-token authority is unchanged.",
            ),
            (
                "dsv-expert-cache",
                DiscrepancyKind::NotMeasured,
                "Would attack the actual critical path (415 ms expert_slab_io). No after-token exists yet.",
            ),
            (
                "dsv-admission-identity",
                DiscrepancyKind::ParallelSumPresentedAsLatency,
                "path_resolve 1318 ms and verify 2505 ms are parallel-thread sums against a 1038 ms token. Moving them behind a seal may help, but the 80 ms claim must be proven on body_ns, not on the parallel sum.",
            ),
            (
                "dsv-integrate-graph",
                DiscrepancyKind::NotMeasured,
                "289 ms is the sum of other lanes' claims. Summing claims is the error this reconciler exists to catch.",
            ),
        ],
        "q80" => vec![
            (
                "q80-runtime-residency",
                DiscrepancyKind::NotMeasured,
                "269 ms claimed against compact-slab pack. No paired whole-token after on this worktree.",
            ),
            (
                "q80-deltanet-gqa2",
                DiscrepancyKind::NotMeasured,
                "180 ms claimed on attention. Prefix CBs mix attention+shared+router; a component GPU win may be overlapped. No token after.",
            ),
            (
                "q80-decode-throughput",
                DiscrepancyKind::NotMeasured,
                "150 ms claimed on mixed-codec decode. The 192 ms organ product is not the current Q4 token.",
            ),
            (
                "q80-decode-kernels",
                DiscrepancyKind::IsolatedMicrobenchNotToken,
                "192 ms is 48*10*(gate+up+down) isolated-organ GPU, not a full token. CROSS P1-DECODE-192MS-NOT-A-TOKEN. Cannot sit beside the 403/559 ms incumbent.",
            ),
        ],
        _ => Vec::new(),
    }
}

pub fn ascent_2026_08_16_reports() -> Vec<ModelReconciliation> {
    let claims = ascent_2026_08_16_claims();
    ascent_2026_08_16_measured()
        .into_iter()
        .map(|m| {
            let kinds = ascent_2026_08_16_kinds(&m.model);
            let mut report = with_kinds(reconcile(&claims, &m), &kinds);
            report.honesty = match m.model.as_str() {
                "dsv4f" => format!(
                    "DSV4F authority token is 1,037,764,208 ns (DIRTY_ENGINEERING, host-wall warm median). \
                     Sum of ASCENT_STATE recoverable claims on this model is {} ns ({:.2}× the token). \
                     Measured whole-token delta on main is 0 ns. Discrepancy = {} ns. \
                     Lane wins do not add up to a token improvement because there is no token improvement.",
                    report.sum_claimed_ns,
                    report.sum_claimed_ns as f64 / report.token_before_ns as f64,
                    report.discrepancy_ns
                ),
                "q80" => format!(
                    "Q80 TOKEN_NS decode wall is 559,171,655 ns (DIRTY_ENGINEERING). \
                     The registered 403 ms incumbent is not used here (P1: unpaired, 1637 fallbacks, no GPU timestamps). \
                     Sum of claimed recoverables is {} ns. Measured whole-token delta is 0 ns. \
                     Discrepancy = {} ns. decode-kernels' 192 ms is not a token.",
                    report.sum_claimed_ns, report.discrepancy_ns
                ),
                _ => report.honesty.clone(),
            };
            report
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrepancy_is_claimed_minus_measured_delta() {
        let claims = vec![
            LaneClaim {
                lane: "a".into(),
                model: "m".into(),
                claimed_ns_saved: 300,
                claim_source: "t".into(),
                claim_kind: "h".into(),
            },
            LaneClaim {
                lane: "b".into(),
                model: "m".into(),
                claimed_ns_saved: 200,
                claim_source: "t".into(),
                claim_kind: "h".into(),
            },
        ];
        let measured = MeasuredToken {
            model: "m".into(),
            before_ns: 1000,
            after_ns: Some(965),
            source: "t".into(),
            label: "DIRTY_ENGINEERING".into(),
        };
        let r = reconcile(&claims, &measured);
        assert_eq!(r.measured_delta_ns, 35);
        assert_eq!(r.sum_claimed_ns, 500);
        assert_eq!(r.discrepancy_ns, 465);
    }

    #[test]
    fn dsv_claims_exceed_the_token() {
        let reports = ascent_2026_08_16_reports();
        let dsv = reports.iter().find(|r| r.model == "dsv4f").unwrap();
        assert_eq!(dsv.measured_delta_ns, 0);
        assert!(dsv.sum_claimed_ns > dsv.token_before_ns as i64);
        assert_eq!(dsv.discrepancy_ns, dsv.sum_claimed_ns);
        assert!(dsv.honesty.contains("do not add up"));
    }
}
