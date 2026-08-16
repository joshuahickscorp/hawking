//! Unified TOKEN_NS document. Both model ledgers adapt into this shape.
//!
//! Everything is nanoseconds. A stage that is a parallel-thread sum is
//! labeled `parallel_sum_not_latency` and does not enter the closure sum.
//! GPU time is `MTLCommandBuffer.GPUEndTime − GPUStartTime` after wait, or
//! it is marked missing — never a CPU-wait proxy.

use serde::{Deserialize, Serialize};

pub const TOKEN_NS_SCHEMA: &str = "hawking.ascent.token_ns.v1";

/// Fail the closure when unattributed residual exceeds this fraction of the
/// token. 5% is large enough for the known Q80 baseline hole (15.23 s of
/// named stages vs a 15.60 s run ≈ 2.3%) and small enough that a 200 ms
/// hole on a 1 s DSV4F token still fails loudly.
pub const DEFAULT_RESIDUAL_LIMIT: f64 = 0.05;

pub const GPU_TIMESTAMP_AUTHORITY: &str =
    "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ResourceClass {
    Cpu,
    Gpu,
    Dram,
    Io,
    Sync,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SerialOrOverlappable {
    Serial,
    Overlappable,
    ParallelSumNotLatency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RemovableOrNecessary {
    Removable,
    Necessary,
    Conditional,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    Measured,
    Derived,
    Estimated,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MeasurementLabel {
    DirtyEngineering,
    CleanCandidate,
    BaseTrue,
}

impl MeasurementLabel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DirtyEngineering => "DIRTY_ENGINEERING",
            Self::CleanCandidate => "CLEAN_CANDIDATE",
            Self::BaseTrue => "BASE_TRUE",
        }
    }
}

/// One comparable stage. `stage` is the cross-model name; `substage` keeps
/// the source ledger's original key so nothing is renamed out of existence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenNsStage {
    pub stage: String,
    pub substage: String,
    pub calls_per_token: f64,
    pub ns_per_call: f64,
    pub ns_per_token: f64,
    pub pct_of_token: f64,
    pub resource_class: ResourceClass,
    pub serial_or_overlappable: SerialOrOverlappable,
    pub removable_or_necessary: RemovableOrNecessary,
    pub confidence: Confidence,
    pub method: String,
    pub commit: String,
}

impl TokenNsStage {
    pub fn counts_toward_closure(&self) -> bool {
        self.serial_or_overlappable == SerialOrOverlappable::Serial
    }

    pub fn new(
        stage: impl Into<String>,
        substage: impl Into<String>,
        calls_per_token: f64,
        ns_per_token: f64,
        total_token_ns: u64,
        resource_class: ResourceClass,
        serial_or_overlappable: SerialOrOverlappable,
        removable_or_necessary: RemovableOrNecessary,
        confidence: Confidence,
        method: impl Into<String>,
        commit: impl Into<String>,
    ) -> Self {
        let ns_per_call = if calls_per_token > 0.0 {
            ns_per_token / calls_per_token
        } else {
            0.0
        };
        let pct_of_token = if total_token_ns == 0 {
            0.0
        } else {
            ns_per_token * 100.0 / total_token_ns as f64
        };
        Self {
            stage: stage.into(),
            substage: substage.into(),
            calls_per_token,
            ns_per_call,
            ns_per_token,
            pct_of_token,
            resource_class,
            serial_or_overlappable,
            removable_or_necessary,
            confidence,
            method: method.into(),
            commit: commit.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct TokenNsTotals {
    #[serde(rename = "TOTAL_TOKEN_NS")]
    pub total_token_ns: u64,
    #[serde(rename = "TOTAL_GPU_BUSY_NS")]
    pub total_gpu_busy_ns: u64,
    #[serde(rename = "TOTAL_GPU_IDLE_NS")]
    pub total_gpu_idle_ns: u64,
    #[serde(rename = "TOTAL_GPU_GAP_NS")]
    pub total_gpu_gap_ns: u64,
    #[serde(rename = "TOTAL_CPU_CRITICAL_NS")]
    pub total_cpu_critical_ns: u64,
    #[serde(rename = "TOTAL_DISPATCHES")]
    pub total_dispatches: u64,
    #[serde(rename = "TOTAL_COMMAND_BUFFERS")]
    pub total_command_buffers: u64,
    #[serde(rename = "TOTAL_SYNC_POINTS")]
    pub total_sync_points: u64,
    #[serde(rename = "TOTAL_READBACKS")]
    pub total_readbacks: u64,
    #[serde(rename = "TOTAL_BUFFER_CREATIONS")]
    pub total_buffer_creations: u64,
    #[serde(rename = "TOTAL_BUFFER_REBINDS")]
    pub total_buffer_rebinds: u64,
    #[serde(rename = "DRAM_BYTES_PER_TOKEN")]
    pub dram_bytes_per_token: u64,
    #[serde(rename = "TEMP_BYTES_PER_TOKEN")]
    pub temp_bytes_per_token: u64,
}

/// `sum(serial stage_ns) + residual_ns == TOTAL_TOKEN_NS` is required.
/// Residual is signed: negative means the exclusive set overcounted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClosureReport {
    pub identity: &'static str,
    pub sum_serial_stage_ns: i128,
    pub residual_ns: i128,
    pub total_token_ns: u64,
    pub identity_holds: bool,
    pub residual_fraction: f64,
    pub residual_limit_fraction: f64,
    pub residual_within_limit: bool,
    pub naive_all_stage_sum_ns: i128,
    pub naive_overcount_ns: i128,
    pub failed: bool,
    pub failure: Option<String>,
}

impl ClosureReport {
    pub fn compute(
        total_token_ns: u64,
        stages: &[TokenNsStage],
        residual_limit_fraction: f64,
    ) -> Self {
        let sum_serial_stage_ns: i128 = stages
            .iter()
            .filter(|s| s.counts_toward_closure())
            .map(|s| s.ns_per_token.round() as i128)
            .sum();
        let naive_all_stage_sum_ns: i128 = stages
            .iter()
            .map(|s| s.ns_per_token.round() as i128)
            .sum();
        let total = total_token_ns as i128;
        let residual_ns = total - sum_serial_stage_ns;
        let identity_holds = sum_serial_stage_ns + residual_ns == total;
        let residual_fraction = if total == 0 {
            0.0
        } else {
            residual_ns as f64 / total as f64
        };
        let residual_within_limit = residual_ns >= 0
            && residual_fraction.abs() <= residual_limit_fraction + f64::EPSILON;
        let naive_overcount_ns = naive_all_stage_sum_ns - total;
        let failure = if !identity_holds {
            Some(format!(
                "CLOSURE IDENTITY BROKEN: sum_serial_stage_ns ({sum_serial_stage_ns}) + residual_ns ({residual_ns}) != TOTAL_TOKEN_NS ({total})"
            ))
        } else if residual_ns < 0 {
            Some(format!(
                "CLOSURE FAIL: exclusive serial stages overcount the token by {} ns ({:.1}% of TOTAL_TOKEN_NS={}). Parallel-sum or nested rows leaked into the exclusive set.",
                -residual_ns,
                (-residual_ns as f64) * 100.0 / total.max(1) as f64,
                total
            ))
        } else if residual_fraction > residual_limit_fraction {
            Some(format!(
                "CLOSURE FAIL: residual_ns={residual_ns} is {:.1}% of TOTAL_TOKEN_NS={total}, exceeding the stated {:.1}% limit. Unattributed cost is hiding here.",
                residual_fraction * 100.0,
                residual_limit_fraction * 100.0
            ))
        } else {
            None
        };
        Self {
            identity: "sum(serial stage_ns) + residual_ns == TOTAL_TOKEN_NS",
            sum_serial_stage_ns,
            residual_ns,
            total_token_ns,
            identity_holds,
            residual_fraction,
            residual_limit_fraction,
            residual_within_limit,
            naive_all_stage_sum_ns,
            naive_overcount_ns,
            failed: failure.is_some(),
            failure,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CriticalPath {
    pub model: String,
    pub token_definition: String,
    pub serial_ns: u64,
    pub overlappable_ns: u64,
    pub parallel_sum_ns: u64,
    pub top_serial_stages: Vec<String>,
    pub statement: String,
    pub warning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenNsDocument {
    pub schema: &'static str,
    pub model: String,
    pub vehicle: String,
    pub source_schema: String,
    pub source_path: String,
    pub measurement_label: MeasurementLabel,
    pub commit: String,
    pub gpu_timestamp_authority: &'static str,
    pub residual_limit_fraction: f64,
    pub stages: Vec<TokenNsStage>,
    pub totals: TokenNsTotals,
    pub residual_ns: i128,
    pub closure: ClosureReport,
    pub critical_path: CriticalPath,
    pub notes: Vec<String>,
}

impl TokenNsDocument {
    pub fn seal(mut self) -> Self {
        self.stages
            .sort_by(|a, b| {
                b.ns_per_token
                    .partial_cmp(&a.ns_per_token)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.substage.cmp(&b.substage))
            });
        self.closure = ClosureReport::compute(
            self.totals.total_token_ns,
            &self.stages,
            self.residual_limit_fraction,
        );
        self.residual_ns = self.closure.residual_ns;
        self.critical_path = critical_path_from_stages(
            &self.model,
            &self.critical_path.token_definition,
            &self.stages,
            self.totals.total_token_ns,
            self.critical_path.statement.clone(),
            self.critical_path.warning.clone(),
        );
        self
    }
}

pub fn critical_path_from_stages(
    model: &str,
    token_definition: &str,
    stages: &[TokenNsStage],
    total_token_ns: u64,
    statement: String,
    warning: Option<String>,
) -> CriticalPath {
    let mut serial_ns = 0.0;
    let mut overlappable_ns = 0.0;
    let mut parallel_sum_ns = 0.0;
    let mut serial_names: Vec<(&str, f64)> = Vec::new();
    for s in stages {
        match s.serial_or_overlappable {
            SerialOrOverlappable::Serial => {
                serial_ns += s.ns_per_token;
                serial_names.push((s.substage.as_str(), s.ns_per_token));
            }
            SerialOrOverlappable::Overlappable => overlappable_ns += s.ns_per_token,
            SerialOrOverlappable::ParallelSumNotLatency => parallel_sum_ns += s.ns_per_token,
        }
    }
    serial_names.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_serial_stages = serial_names
        .into_iter()
        .take(6)
        .map(|(name, ns)| format!("{name} ({:.1} ms)", ns / 1e6))
        .collect();
    let _ = total_token_ns;
    CriticalPath {
        model: model.to_owned(),
        token_definition: token_definition.to_owned(),
        serial_ns: serial_ns.round() as u64,
        overlappable_ns: overlappable_ns.round() as u64,
        parallel_sum_ns: parallel_sum_ns.round() as u64,
        top_serial_stages,
        statement,
        warning,
    }
}

#[derive(Debug, Clone)]
pub struct EmitMeta {
    pub commit: String,
    pub source_path: String,
    pub measurement_label: MeasurementLabel,
    pub residual_limit_fraction: f64,
}

impl EmitMeta {
    pub fn new(commit: impl Into<String>, source_path: impl Into<String>) -> Self {
        Self {
            commit: commit.into(),
            source_path: source_path.into(),
            measurement_label: MeasurementLabel::DirtyEngineering,
            residual_limit_fraction: DEFAULT_RESIDUAL_LIMIT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(name: &str, ns: f64, kind: SerialOrOverlappable) -> TokenNsStage {
        TokenNsStage::new(
            "test",
            name,
            1.0,
            ns,
            1_000,
            ResourceClass::Cpu,
            kind,
            RemovableOrNecessary::Necessary,
            Confidence::Measured,
            "test",
            "commit",
        )
    }

    #[test]
    fn closure_identity_holds_by_construction() {
        let stages = vec![
            stage("a", 400.0, SerialOrOverlappable::Serial),
            stage("b", 500.0, SerialOrOverlappable::Serial),
            stage("parallel", 9_000.0, SerialOrOverlappable::ParallelSumNotLatency),
        ];
        let c = ClosureReport::compute(1_000, &stages, 0.15);
        assert!(c.identity_holds);
        assert_eq!(c.sum_serial_stage_ns, 900);
        assert_eq!(c.residual_ns, 100);
        assert!(!c.failed);
        assert_eq!(c.naive_overcount_ns, 8_900);
    }

    #[test]
    fn residual_over_limit_fails_loudly() {
        let stages = vec![stage("named", 700.0, SerialOrOverlappable::Serial)];
        let c = ClosureReport::compute(1_000, &stages, 0.05);
        assert!(c.failed);
        let msg = c.failure.expect("loud failure");
        assert!(msg.contains("CLOSURE FAIL"));
        assert!(msg.contains("residual_ns=300"));
    }

    #[test]
    fn overcount_fails_when_serial_set_leaks_overlap() {
        let stages = vec![
            stage("a", 800.0, SerialOrOverlappable::Serial),
            stage("b", 400.0, SerialOrOverlappable::Serial),
        ];
        let c = ClosureReport::compute(1_000, &stages, 0.05);
        assert!(c.failed);
        assert!(c.residual_ns < 0);
        assert!(c.failure.unwrap().contains("overcount"));
    }
}
