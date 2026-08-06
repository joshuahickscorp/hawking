//! Skill Foundry (Ascension Bible §18).
//!
//! A repeated successful workflow may be **proposed** as a skill. A skill is
//! versioned, tested, source-bound, retrievable, composable, and retirable.
//!
//! **Authority boundary (non-negotiable):**
//! - The sandbox may propose.
//! - Only the protected controller may admit.
//!
//! Admission path:
//! ```text
//! propose → replay → hidden validation → compatibility test → protected admission
//! ```
//!
//! This module is a **scaffold**: real types + a deterministic stub pipeline.
//! Live model work, Qwen/Gravity, and frankenstein evidence are out of scope.
//! Procedural memory in [`crate::memory_classes`] already stores successful
//! recipes; Foundry elevates those into versioned, tested, admit-gated skills
//! that land in Memory OS L3.

use hide_core::ids::now_ms;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Skill schema (bible §18)
// ---------------------------------------------------------------------------

/// Full skill document. Fields match the Ascension Bible skill contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillSpec {
    pub name: String,
    pub purpose: String,
    pub scope: String,
    pub inputs: Vec<SkillIoField>,
    pub outputs: Vec<SkillIoField>,
    pub preconditions: Vec<String>,
    /// Ordered procedure steps (source-bound: each step may cite a receipt/path).
    pub procedure: Vec<SkillStep>,
    pub environment: SkillEnvironment,
    pub provenance: SkillProvenance,
    pub tests: Vec<SkillTest>,
    pub failure_modes: Vec<SkillFailureMode>,
    pub compatibility: SkillCompatibility,
    pub version: SkillVersion,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillIoField {
    pub name: String,
    pub description: String,
    /// JSON-schema-ish type hint (`string`, `path`, `json`, …).
    pub ty: String,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillStep {
    pub ordinal: u32,
    pub action: String,
    /// Optional source binding (receipt id, path, tool name).
    pub source_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillEnvironment {
    /// Host OS constraints (e.g. `macos`, `apple-silicon`).
    pub platforms: Vec<String>,
    /// Required tools / binaries.
    pub tools: Vec<String>,
    /// Required env vars (names only; never values/secrets).
    pub env_vars: Vec<String>,
    /// Workspace / path assumptions.
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillProvenance {
    /// Who proposed (sandbox role, human, distill path).
    pub proposed_by: String,
    pub proposed_at_ms: u64,
    /// Source workflow / receipt ids that motivated the proposal.
    pub source_workflow_ids: Vec<String>,
    /// Evidence paths (source-bound).
    pub evidence_refs: Vec<String>,
    /// Success count observed before proposal.
    pub success_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillTest {
    pub name: String,
    pub description: String,
    /// Fixture or input binding.
    pub input_ref: String,
    /// Expected outcome summary (deterministic check in stub).
    pub expect: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillFailureMode {
    pub name: String,
    pub description: String,
    pub recovery: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillCompatibility {
    /// Skill names this skill may compose with.
    pub composes_with: Vec<String>,
    /// Skill names / versions this conflicts with.
    pub conflicts_with: Vec<String>,
    /// Minimum Foundry schema version.
    pub min_foundry_schema: String,
    /// HCLI / agent OS surface compatibility tags.
    pub surfaces: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl SkillVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    pub fn as_string(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl std::fmt::Display for SkillVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_string())
    }
}

// ---------------------------------------------------------------------------
// Lifecycle + admission
// ---------------------------------------------------------------------------

/// Skill lifecycle. Only [`SkillStatus::Admitted`] is retrievable as L3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillStatus {
    /// Sandbox proposal; not yet admitted.
    Proposed,
    /// Replay stage running / complete.
    Replaying,
    /// Hidden validation stage.
    Validating,
    /// Compatibility test stage.
    CompatibilityTesting,
    /// Protected controller admitted; eligible for L3 Memory OS.
    Admitted,
    /// Explicitly retired (still inspectable, not default-retrievable).
    Retired,
    /// Failed a stage; not admitted.
    Rejected,
}

impl SkillStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Proposed => "proposed",
            Self::Replaying => "replaying",
            Self::Validating => "validating",
            Self::CompatibilityTesting => "compatibility_testing",
            Self::Admitted => "admitted",
            Self::Retired => "retired",
            Self::Rejected => "rejected",
        }
    }

    pub fn is_retrievable_as_skill(self) -> bool {
        matches!(self, Self::Admitted)
    }
}

/// Stages of the admission pipeline (ordered).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionStage {
    Propose,
    Replay,
    HiddenValidation,
    CompatibilityTest,
    ProtectedAdmission,
}

impl AdmissionStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Propose => "propose",
            Self::Replay => "replay",
            Self::HiddenValidation => "hidden_validation",
            Self::CompatibilityTest => "compatibility_test",
            Self::ProtectedAdmission => "protected_admission",
        }
    }

    pub fn pipeline() -> [AdmissionStage; 5] {
        [
            Self::Propose,
            Self::Replay,
            Self::HiddenValidation,
            Self::CompatibilityTest,
            Self::ProtectedAdmission,
        ]
    }

    pub fn next(self) -> Option<Self> {
        match self {
            Self::Propose => Some(Self::Replay),
            Self::Replay => Some(Self::HiddenValidation),
            Self::HiddenValidation => Some(Self::CompatibilityTest),
            Self::CompatibilityTest => Some(Self::ProtectedAdmission),
            Self::ProtectedAdmission => None,
        }
    }
}

/// Receipt for one completed admission stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageReceipt {
    pub stage: AdmissionStage,
    pub passed: bool,
    pub detail: String,
    pub at_ms: u64,
}

/// A skill under Foundry management.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillRecord {
    pub id: String,
    pub spec: SkillSpec,
    pub status: SkillStatus,
    pub stage_receipts: Vec<StageReceipt>,
    /// Set only when admitted under a protected controller.
    pub admitted_by: Option<String>,
    pub admitted_at_ms: Option<u64>,
    pub retired_at_ms: Option<u64>,
    pub reject_reason: Option<String>,
}

/// Capability: sandbox may propose (and only propose).
#[derive(Debug, Clone, Copy)]
pub struct SandboxProposeCap {
    _private: (),
}

impl SandboxProposeCap {
    pub fn mint() -> Self {
        Self { _private: () }
    }
}

/// Capability: protected controller may admit. Sandbox must not hold this type.
#[derive(Debug, Clone, Copy)]
pub struct ProtectedControllerCap {
    _private: (),
}

impl ProtectedControllerCap {
    /// Mint only at the protected-controller entry point — never on the sandbox
    /// model path. Requiring this type at [`SkillFoundry::admit`] makes a
    /// sandbox→admit write obvious in any diff (same pattern as VerifierWriteCap).
    pub fn mint() -> Self {
        Self { _private: () }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SkillFoundryError {
    #[error("skill not found: {0}")]
    NotFound(String),
    #[error("invalid skill foundry operation: {0}")]
    Invalid(String),
    #[error("admission denied: {0}")]
    AdmissionDenied(String),
}

// ---------------------------------------------------------------------------
// Foundry (stub pipeline)
// ---------------------------------------------------------------------------

const FOUNDRY_SCHEMA: &str = "hide.skill_foundry.v0";

/// Skill Foundry registry + admission pipeline stub.
#[derive(Debug, Default)]
pub struct SkillFoundry {
    skills: RwLock<BTreeMap<String, SkillRecord>>,
    next_id: AtomicU64,
    clock_ms: AtomicU64,
}

impl SkillFoundry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_clock_ms(&self, ms: u64) {
        self.clock_ms.store(ms, Ordering::Relaxed);
    }

    fn now(&self) -> u64 {
        let c = self.clock_ms.load(Ordering::Relaxed);
        if c == 0 {
            now_ms()
        } else {
            c
        }
    }

    fn mint_id(&self) -> String {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("skill-{n}")
    }

    pub fn schema_version() -> &'static str {
        FOUNDRY_SCHEMA
    }

    /// Validate minimal schema completeness before propose is accepted.
    pub fn validate_spec(spec: &SkillSpec) -> Result<(), SkillFoundryError> {
        if spec.name.trim().is_empty() {
            return Err(SkillFoundryError::Invalid("skill name required".into()));
        }
        if spec.purpose.trim().is_empty() {
            return Err(SkillFoundryError::Invalid("skill purpose required".into()));
        }
        if spec.procedure.is_empty() {
            return Err(SkillFoundryError::Invalid(
                "skill procedure must have at least one step".into(),
            ));
        }
        if spec.tests.is_empty() {
            return Err(SkillFoundryError::Invalid(
                "skill must declare at least one test (bible: tested)".into(),
            ));
        }
        if spec.provenance.source_workflow_ids.is_empty()
            && spec.provenance.evidence_refs.is_empty()
        {
            return Err(SkillFoundryError::Invalid(
                "skill must be source-bound (workflow ids or evidence refs)".into(),
            ));
        }
        Ok(())
    }

    /// Sandbox proposes a skill. Does **not** admit.
    pub fn propose(
        &self,
        _cap: &SandboxProposeCap,
        mut spec: SkillSpec,
    ) -> Result<SkillRecord, SkillFoundryError> {
        Self::validate_spec(&spec)?;
        let now = self.now();
        if spec.provenance.proposed_at_ms == 0 {
            spec.provenance.proposed_at_ms = now;
        }
        let id = self.mint_id();
        let rec = SkillRecord {
            id: id.clone(),
            spec,
            status: SkillStatus::Proposed,
            stage_receipts: vec![StageReceipt {
                stage: AdmissionStage::Propose,
                passed: true,
                detail: "sandbox proposal accepted into foundry queue".into(),
                at_ms: now,
            }],
            admitted_by: None,
            admitted_at_ms: None,
            retired_at_ms: None,
            reject_reason: None,
        };
        self.skills.write().insert(id, rec.clone());
        Ok(rec)
    }

    /// Replay stage: re-check procedure source bindings exist in the stub
    /// (non-empty source_ref or action). Deterministic; no live model.
    pub fn replay(&self, id: &str) -> Result<SkillRecord, SkillFoundryError> {
        self.advance_stage(id, AdmissionStage::Replay, |spec| {
            let ok = spec.procedure.iter().all(|s| !s.action.trim().is_empty());
            if ok {
                (true, "replay: all procedure steps have actions".into())
            } else {
                (false, "replay: empty procedure action".into())
            }
        })
    }

    /// Hidden validation: run declared tests against stub expect non-empty.
    pub fn hidden_validate(&self, id: &str) -> Result<SkillRecord, SkillFoundryError> {
        self.advance_stage(id, AdmissionStage::HiddenValidation, |spec| {
            let ok = spec
                .tests
                .iter()
                .all(|t| !t.name.is_empty() && !t.expect.is_empty());
            if ok {
                (
                    true,
                    format!("hidden_validation: {} tests declared well-formed", spec.tests.len()),
                )
            } else {
                (false, "hidden_validation: malformed test".into())
            }
        })
    }

    /// Compatibility test against already-admitted skills.
    pub fn compatibility_test(&self, id: &str) -> Result<SkillRecord, SkillFoundryError> {
        let admitted_names: Vec<String> = {
            let map = self.skills.read();
            map.values()
                .filter(|r| r.status == SkillStatus::Admitted)
                .map(|r| r.spec.name.clone())
                .collect()
        };
        self.advance_stage(id, AdmissionStage::CompatibilityTest, |spec| {
            for conflict in &spec.compatibility.conflicts_with {
                if admitted_names.iter().any(|n| n == conflict) {
                    return (
                        false,
                        format!("compatibility: conflicts with admitted skill {conflict}"),
                    );
                }
            }
            if spec.compatibility.min_foundry_schema != FOUNDRY_SCHEMA
                && !spec.compatibility.min_foundry_schema.is_empty()
                && spec.compatibility.min_foundry_schema != "hide.skill_foundry.v0"
            {
                // Allow empty or exact match; unknown future schemas fail closed.
                if !spec
                    .compatibility
                    .min_foundry_schema
                    .starts_with("hide.skill_foundry.")
                {
                    return (
                        false,
                        format!(
                            "compatibility: unknown foundry schema {}",
                            spec.compatibility.min_foundry_schema
                        ),
                    );
                }
            }
            (
                true,
                format!(
                    "compatibility: ok against {} admitted skills",
                    admitted_names.len()
                ),
            )
        })
    }

    /// Protected admission — **requires** [`ProtectedControllerCap`].
    /// Sandbox cannot call this without the cap type.
    pub fn admit(
        &self,
        cap: &ProtectedControllerCap,
        id: &str,
        controller_id: impl Into<String>,
    ) -> Result<SkillRecord, SkillFoundryError> {
        let _cap = cap; // type boundary is the security property
        let controller_id = controller_id.into();
        if controller_id.trim().is_empty() {
            return Err(SkillFoundryError::AdmissionDenied(
                "controller_id required".into(),
            ));
        }
        let now = self.now();
        let mut map = self.skills.write();
        let rec = map
            .get_mut(id)
            .ok_or_else(|| SkillFoundryError::NotFound(id.into()))?;

        if rec.status == SkillStatus::Rejected {
            return Err(SkillFoundryError::AdmissionDenied(
                "rejected skills cannot be admitted".into(),
            ));
        }
        if rec.status == SkillStatus::Admitted {
            return Ok(rec.clone());
        }
        if rec.status == SkillStatus::Retired {
            return Err(SkillFoundryError::AdmissionDenied(
                "retired skills cannot be re-admitted without a new version proposal".into(),
            ));
        }

        // Must have passed the three prior stages.
        let required = [
            AdmissionStage::Propose,
            AdmissionStage::Replay,
            AdmissionStage::HiddenValidation,
            AdmissionStage::CompatibilityTest,
        ];
        for stage in required {
            let passed = rec
                .stage_receipts
                .iter()
                .any(|r| r.stage == stage && r.passed);
            if !passed {
                return Err(SkillFoundryError::AdmissionDenied(format!(
                    "missing passing receipt for stage {}",
                    stage.as_str()
                )));
            }
        }

        rec.stage_receipts.push(StageReceipt {
            stage: AdmissionStage::ProtectedAdmission,
            passed: true,
            detail: format!("admitted by protected controller {controller_id}"),
            at_ms: now,
        });
        rec.status = SkillStatus::Admitted;
        rec.admitted_by = Some(controller_id);
        rec.admitted_at_ms = Some(now);
        Ok(rec.clone())
    }

    /// Run the full stub pipeline (propose already done) through compat, then
    /// admit under the protected controller. Convenience for tests / dry-runs.
    pub fn run_admission_pipeline(
        &self,
        cap: &ProtectedControllerCap,
        id: &str,
        controller_id: impl Into<String>,
    ) -> Result<SkillRecord, SkillFoundryError> {
        self.replay(id)?;
        self.hidden_validate(id)?;
        self.compatibility_test(id)?;
        self.admit(cap, id, controller_id)
    }

    pub fn retire(&self, id: &str, reason: &str) -> Result<SkillRecord, SkillFoundryError> {
        let now = self.now();
        let mut map = self.skills.write();
        let rec = map
            .get_mut(id)
            .ok_or_else(|| SkillFoundryError::NotFound(id.into()))?;
        if rec.status != SkillStatus::Admitted {
            return Err(SkillFoundryError::Invalid(
                "only admitted skills can be retired".into(),
            ));
        }
        rec.status = SkillStatus::Retired;
        rec.retired_at_ms = Some(now);
        rec.reject_reason = Some(reason.into());
        Ok(rec.clone())
    }

    pub fn get(&self, id: &str) -> Option<SkillRecord> {
        self.skills.read().get(id).cloned()
    }

    /// Retrievable skills = admitted only (bible: retrievable after admission).
    pub fn list_retrievable(&self) -> Vec<SkillRecord> {
        self.skills
            .read()
            .values()
            .filter(|r| r.status.is_retrievable_as_skill())
            .cloned()
            .collect()
    }

    pub fn list_all(&self) -> Vec<SkillRecord> {
        self.skills.read().values().cloned().collect()
    }

    fn advance_stage<F>(
        &self,
        id: &str,
        stage: AdmissionStage,
        check: F,
    ) -> Result<SkillRecord, SkillFoundryError>
    where
        F: FnOnce(&SkillSpec) -> (bool, String),
    {
        let now = self.now();
        let mut map = self.skills.write();
        let rec = map
            .get_mut(id)
            .ok_or_else(|| SkillFoundryError::NotFound(id.into()))?;
        if matches!(
            rec.status,
            SkillStatus::Rejected | SkillStatus::Retired | SkillStatus::Admitted
        ) {
            return Err(SkillFoundryError::Invalid(format!(
                "cannot run {} on skill in status {}",
                stage.as_str(),
                rec.status.as_str()
            )));
        }

        // Prior stage must have passed.
        if stage != AdmissionStage::Propose {
            let prior = match stage {
                AdmissionStage::Replay => AdmissionStage::Propose,
                AdmissionStage::HiddenValidation => AdmissionStage::Replay,
                AdmissionStage::CompatibilityTest => AdmissionStage::HiddenValidation,
                AdmissionStage::ProtectedAdmission => AdmissionStage::CompatibilityTest,
                AdmissionStage::Propose => AdmissionStage::Propose,
            };
            let prior_ok = rec
                .stage_receipts
                .iter()
                .any(|r| r.stage == prior && r.passed);
            if !prior_ok {
                return Err(SkillFoundryError::Invalid(format!(
                    "stage {} requires prior {} to pass",
                    stage.as_str(),
                    prior.as_str()
                )));
            }
        }

        let (passed, detail) = check(&rec.spec);
        rec.stage_receipts.push(StageReceipt {
            stage,
            passed,
            detail: detail.clone(),
            at_ms: now,
        });
        if !passed {
            rec.status = SkillStatus::Rejected;
            rec.reject_reason = Some(detail);
        } else {
            rec.status = match stage {
                AdmissionStage::Replay => SkillStatus::Replaying,
                AdmissionStage::HiddenValidation => SkillStatus::Validating,
                AdmissionStage::CompatibilityTest => SkillStatus::CompatibilityTesting,
                AdmissionStage::Propose => SkillStatus::Proposed,
                AdmissionStage::ProtectedAdmission => SkillStatus::Admitted,
            };
        }
        Ok(rec.clone())
    }
}

/// Helper to build a minimal valid skill for tests / scaffolding demos.
pub fn example_skill_spec(name: impl Into<String>) -> SkillSpec {
    let name = name.into();
    SkillSpec {
        name: name.clone(),
        purpose: format!("Example skill {name}"),
        scope: "workspace".into(),
        inputs: vec![SkillIoField {
            name: "target".into(),
            description: "path or symbol to act on".into(),
            ty: "string".into(),
            required: true,
        }],
        outputs: vec![SkillIoField {
            name: "receipt".into(),
            description: "execution receipt id".into(),
            ty: "string".into(),
            required: true,
        }],
        preconditions: vec!["workspace is a git repo".into()],
        procedure: vec![SkillStep {
            ordinal: 1,
            action: "run verified workflow steps".into(),
            source_ref: Some("receipt:example".into()),
        }],
        environment: SkillEnvironment {
            platforms: vec!["macos".into(), "apple-silicon".into()],
            tools: vec!["cargo".into()],
            env_vars: vec![],
            notes: "local only; no network".into(),
        },
        provenance: SkillProvenance {
            proposed_by: "sandbox".into(),
            proposed_at_ms: 0,
            source_workflow_ids: vec!["wf-example-1".into()],
            evidence_refs: vec!["receipts/example.json".into()],
            success_count: 3,
        },
        tests: vec![SkillTest {
            name: "smoke".into(),
            description: "procedure is non-empty and source-bound".into(),
            input_ref: "fixture:smoke".into(),
            expect: "ok".into(),
        }],
        failure_modes: vec![SkillFailureMode {
            name: "missing_tool".into(),
            description: "required tool not on PATH".into(),
            recovery: "install tool or skip skill".into(),
        }],
        compatibility: SkillCompatibility {
            composes_with: vec![],
            conflicts_with: vec![],
            min_foundry_schema: FOUNDRY_SCHEMA.into(),
            surfaces: vec!["hcli".into()],
        },
        version: SkillVersion::new(0, 1, 0),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandbox_can_propose_but_not_admit_without_cap() {
        let foundry = SkillFoundry::new();
        foundry.set_clock_ms(100);
        let sandbox = SandboxProposeCap::mint();
        let rec = foundry
            .propose(&sandbox, example_skill_spec("optimize_qwen_moe_projection_wave"))
            .unwrap();
        assert_eq!(rec.status, SkillStatus::Proposed);
        assert!(foundry.list_retrievable().is_empty());
        // admit requires ProtectedControllerCap at the type level — calling
        // without going through stages also fails.
        let ctrl = ProtectedControllerCap::mint();
        let err = foundry.admit(&ctrl, &rec.id, "controller").unwrap_err();
        assert!(matches!(err, SkillFoundryError::AdmissionDenied(_)));
    }

    #[test]
    fn full_pipeline_admits_under_protected_controller() {
        let foundry = SkillFoundry::new();
        foundry.set_clock_ms(1);
        let sandbox = SandboxProposeCap::mint();
        let ctrl = ProtectedControllerCap::mint();
        let rec = foundry
            .propose(&sandbox, example_skill_spec("optimize_qwen_moe_projection_wave"))
            .unwrap();
        let admitted = foundry
            .run_admission_pipeline(&ctrl, &rec.id, "protected-controller-1")
            .unwrap();
        assert_eq!(admitted.status, SkillStatus::Admitted);
        assert_eq!(
            admitted.admitted_by.as_deref(),
            Some("protected-controller-1")
        );
        assert!(admitted.admitted_at_ms.is_some());
        assert_eq!(foundry.list_retrievable().len(), 1);
        // Stage receipts cover full pipeline.
        let stages: Vec<_> = admitted
            .stage_receipts
            .iter()
            .map(|r| r.stage)
            .collect();
        for expected in AdmissionStage::pipeline() {
            assert!(
                stages.contains(&expected),
                "missing stage {}",
                expected.as_str()
            );
        }
    }

    #[test]
    fn reject_on_empty_procedure_at_propose() {
        let foundry = SkillFoundry::new();
        let mut spec = example_skill_spec("bad");
        spec.procedure.clear();
        let err = foundry
            .propose(&SandboxProposeCap::mint(), spec)
            .unwrap_err();
        assert!(matches!(err, SkillFoundryError::Invalid(_)));
    }

    #[test]
    fn reject_unbound_source_at_propose() {
        let foundry = SkillFoundry::new();
        let mut spec = example_skill_spec("unbound");
        spec.provenance.source_workflow_ids.clear();
        spec.provenance.evidence_refs.clear();
        let err = foundry
            .propose(&SandboxProposeCap::mint(), spec)
            .unwrap_err();
        assert!(matches!(err, SkillFoundryError::Invalid(_)));
    }

    #[test]
    fn compatibility_conflict_rejects() {
        let foundry = SkillFoundry::new();
        foundry.set_clock_ms(1);
        let sandbox = SandboxProposeCap::mint();
        let ctrl = ProtectedControllerCap::mint();
        let a = foundry
            .propose(&sandbox, example_skill_spec("skill_a"))
            .unwrap();
        foundry
            .run_admission_pipeline(&ctrl, &a.id, "ctrl")
            .unwrap();

        let mut conflict = example_skill_spec("skill_b");
        conflict.compatibility.conflicts_with = vec!["skill_a".into()];
        let b = foundry.propose(&sandbox, conflict).unwrap();
        foundry.replay(&b.id).unwrap();
        foundry.hidden_validate(&b.id).unwrap();
        let after = foundry.compatibility_test(&b.id).unwrap();
        assert_eq!(after.status, SkillStatus::Rejected);
        assert!(foundry.list_retrievable().iter().all(|r| r.spec.name != "skill_b"));
    }

    #[test]
    fn retire_removes_from_retrievable() {
        let foundry = SkillFoundry::new();
        foundry.set_clock_ms(1);
        let sandbox = SandboxProposeCap::mint();
        let ctrl = ProtectedControllerCap::mint();
        let rec = foundry
            .propose(&sandbox, example_skill_spec("retirable"))
            .unwrap();
        foundry
            .run_admission_pipeline(&ctrl, &rec.id, "ctrl")
            .unwrap();
        assert_eq!(foundry.list_retrievable().len(), 1);
        foundry.retire(&rec.id, "superseded by v2").unwrap();
        assert!(foundry.list_retrievable().is_empty());
        let got = foundry.get(&rec.id).unwrap();
        assert_eq!(got.status, SkillStatus::Retired);
    }

    #[test]
    fn skill_is_versioned_and_composable_fields_roundtrip() {
        let mut spec = example_skill_spec("compose_demo");
        spec.version = SkillVersion::new(1, 2, 3);
        spec.compatibility.composes_with = vec!["other_skill".into()];
        let json = serde_json::to_string(&spec).unwrap();
        let back: SkillSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.version.as_string(), "1.2.3");
        assert_eq!(back.compatibility.composes_with, vec!["other_skill"]);
        assert_eq!(SkillFoundry::schema_version(), "hide.skill_foundry.v0");
    }

    #[test]
    fn stages_must_run_in_order() {
        let foundry = SkillFoundry::new();
        foundry.set_clock_ms(1);
        let rec = foundry
            .propose(
                &SandboxProposeCap::mint(),
                example_skill_spec("ordered"),
            )
            .unwrap();
        // Skip replay → hidden_validate fails.
        let err = foundry.hidden_validate(&rec.id).unwrap_err();
        assert!(matches!(err, SkillFoundryError::Invalid(_)));
    }
}
