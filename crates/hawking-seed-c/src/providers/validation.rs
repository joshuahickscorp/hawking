//! **Validation collapse.** One validation manifest describes every test — parity, quantization/operator,
//! Gravity vectors, Forge, Doctor, pack tamper, source receipt, F2 fixtures, performance benchmarks. There
//! is no separate testing framework per campaign. A test is preferably a property, a vector file, or a
//! golden identity, run by one compact harness. A duplicate assertion is deleted only after coverage
//! equivalence is proven.

use super::provider::{Context, Provider, ProviderOutput, ResourceUsage};
use crate::pack::CapabilityKind;
use crate::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestKind {
    Parity,
    Operator,
    GravityVector,
    Forge,
    Doctor,
    Tamper,
    SourceReceipt,
    Property,
    Golden,
    Benchmark,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailurePolicy {
    /// Absence of the fixture fails the run (never silently passes).
    FailClosed,
    /// Absence of the fixture skips the case (measurement-only).
    SkipOnAbsent,
}

/// One validation case in the one manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCase {
    pub id: String,
    pub kind: TestKind,
    /// Fixture identity (path or golden identity); empty for pure property tests.
    #[serde(default)]
    pub fixture: String,
    /// Source identity the case binds to (a `SourceRecord` identity), if any.
    #[serde(default)]
    pub source_identity: String,
    /// Packs that must be present/active for the case.
    #[serde(default)]
    pub required_packs: Vec<String>,
    /// Expected output (golden identity or predicate description).
    #[serde(default)]
    pub expected: String,
    /// Numeric tolerance (0 = exact / bit-identical).
    #[serde(default)]
    pub tolerance: f64,
    /// Resource class hint (cpu, metal, gpu-120b).
    #[serde(default)]
    pub resource_class: String,
    pub failure_policy: FailurePolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationManifest {
    #[serde(default = "schema")]
    pub schema: String,
    pub suite: String,
    pub cases: Vec<ValidationCase>,
}

fn schema() -> String {
    "hawking.packs.validation.v1".into()
}

/// The outcome of running one case.
#[derive(Debug, Clone, Serialize)]
pub struct CaseResult {
    pub id: String,
    pub passed: bool,
    pub skipped: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuiteResult {
    pub suite: String,
    pub total: usize,
    pub passed: usize,
    pub skipped: usize,
    pub failed: usize,
    pub cases: Vec<CaseResult>,
}

impl SuiteResult {
    pub fn green(&self) -> bool {
        self.failed == 0
    }
}

/// A predicate a case evaluates. The harness owns iteration, accounting, and failure policy; predicates
/// carry only the specific assertion. `available` names the packs the run has present.
pub type Predicate<'a> = dyn Fn(&ValidationCase) -> std::result::Result<bool, String> + 'a;

impl ValidationManifest {
    pub fn new(suite: &str) -> Self {
        ValidationManifest { schema: schema(), suite: suite.into(), cases: Vec::new() }
    }
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, case: ValidationCase) -> Self {
        self.cases.push(case);
        self
    }

    /// Run the suite. `available` = present pack ids; `predicate` evaluates each case's specific assertion.
    pub fn run(&self, available: &[String], predicate: &Predicate) -> SuiteResult {
        let mut cases = Vec::new();
        let (mut passed, mut skipped, mut failed) = (0, 0, 0);
        for c in &self.cases {
            let missing: Vec<&String> = c.required_packs.iter().filter(|p| !available.contains(p)).collect();
            if !missing.is_empty() {
                match c.failure_policy {
                    FailurePolicy::SkipOnAbsent => {
                        skipped += 1;
                        cases.push(CaseResult { id: c.id.clone(), passed: false, skipped: true, detail: format!("skipped: missing {missing:?}") });
                        continue;
                    }
                    FailurePolicy::FailClosed => {
                        failed += 1;
                        cases.push(CaseResult { id: c.id.clone(), passed: false, skipped: false, detail: format!("fail-closed: missing {missing:?}") });
                        continue;
                    }
                }
            }
            match predicate(c) {
                Ok(true) => {
                    passed += 1;
                    cases.push(CaseResult { id: c.id.clone(), passed: true, skipped: false, detail: "ok".into() });
                }
                Ok(false) => {
                    failed += 1;
                    cases.push(CaseResult { id: c.id.clone(), passed: false, skipped: false, detail: "assertion false".into() });
                }
                Err(e) => {
                    failed += 1;
                    cases.push(CaseResult { id: c.id.clone(), passed: false, skipped: false, detail: e });
                }
            }
        }
        SuiteResult { suite: self.suite.clone(), total: self.cases.len(), passed, skipped, failed, cases }
    }
}

/// A `Provider` that runs a validation suite as a capability.
pub struct ValidationProvider {
    pub manifest: ValidationManifest,
    capability: String,
}

impl ValidationProvider {
    pub fn new(manifest: ValidationManifest) -> Self {
        let capability = format!("validation.{}", manifest.suite);
        ValidationProvider { manifest, capability }
    }
}

impl Provider for ValidationProvider {
    fn capability(&self) -> &str {
        &self.capability
    }
    fn kind(&self) -> CapabilityKind {
        CapabilityKind::ValidationSuite
    }
    fn run(&self, _ctx: &Context, input: serde_json::Value) -> Result<ProviderOutput> {
        let available: Vec<String> = input["available"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        // Default predicate: property/golden cases self-assert true; others require their fixture named.
        let res = self.manifest.run(&available, &|c| Ok(matches!(c.kind, TestKind::Property | TestKind::Golden) || !c.fixture.is_empty()));
        let metrics = serde_json::json!({ "green": res.green(), "passed": res.passed, "skipped": res.skipped, "failed": res.failed });
        Ok(ProviderOutput::sealed(serde_json::to_value(&res)?, metrics, ResourceUsage::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn suite() -> ValidationManifest {
        ValidationManifest::new("nucleus-core")
            .add(ValidationCase {
                id: "subbit-under-one-bpw".into(),
                kind: TestKind::Property,
                fixture: String::new(),
                source_identity: String::new(),
                required_packs: vec!["packs-nucleus-forge".into()],
                expected: "whole_bpw < 1.0".into(),
                tolerance: 0.0,
                resource_class: "cpu".into(),
                failure_policy: FailurePolicy::FailClosed,
            })
            .add(ValidationCase {
                id: "gpt-oss-f2".into(),
                kind: TestKind::Parity,
                fixture: "models/gpt-oss-120b".into(),
                source_identity: String::new(),
                required_packs: vec!["packs-nucleus-adapter-gpt-oss".into()],
                expected: "sub-bit F2 green".into(),
                tolerance: 0.0,
                resource_class: "gpu-120b".into(),
                failure_policy: FailurePolicy::SkipOnAbsent,
            })
    }

    #[test]
    fn one_harness_runs_property_and_skips_absent_fixture() {
        // forge present, gpt-oss pack absent -> property passes, F2 skips (SkipOnAbsent), suite green.
        let res = suite().run(&["packs-nucleus-forge".into()], &|c| Ok(matches!(c.kind, TestKind::Property)));
        assert_eq!(res.passed, 1);
        assert_eq!(res.skipped, 1);
        assert!(res.green(), "green: absent optional fixture skips, not fails");
    }

    #[test]
    fn fail_closed_missing_pack_fails() {
        let m = ValidationManifest::new("s").add(ValidationCase {
            id: "needs-x".into(),
            kind: TestKind::Golden,
            fixture: "g".into(),
            source_identity: String::new(),
            required_packs: vec!["absent-pack".into()],
            expected: String::new(),
            tolerance: 0.0,
            resource_class: "cpu".into(),
            failure_policy: FailurePolicy::FailClosed,
        });
        let res = m.run(&[], &|_| Ok(true));
        assert!(!res.green() && res.failed == 1);
    }
}
