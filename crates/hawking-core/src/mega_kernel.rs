//! Deterministic, provider-neutral MegaKernel execution spine.
//!
//! This is the Rust-owned contract shared by model-specific lowerings.  It is
//! intentionally a plan and stage vocabulary, not a monolithic vendor kernel:
//! CPU, Metal GPU, public Core ML/ANE, and future backends can fill individual
//! stages without duplicating the model/runtime lifecycle.
//!
//! No type in this module claims device execution, source parity, capability,
//! or throughput.  Those remain downstream evidence gates.

use serde::{Deserialize, Serialize};

/// Stable cross-language schema also emitted by `hcli.physical_graph`.
pub const SCHEMA: &str = "hcli.physical_graph.mega_kernel.v1";

/// The common model-aware execution stages.
pub const STAGES: [Stage; 7] = [
    Stage::Load,
    Stage::Decode,
    Stage::Route,
    Stage::Project,
    Stage::Accumulate,
    Stage::StateUpdate,
    Stage::Sample,
];

/// Stages are deliberately broader than individual kernels.  A backend may
/// fuse adjacent stages only after it proves parity and complete useful work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Stage {
    Load,
    Decode,
    Route,
    Project,
    Accumulate,
    StateUpdate,
    Sample,
}

impl Stage {
    /// Stable wire name used in receipts and cross-language plans.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Load => "LOAD",
            Self::Decode => "DECODE",
            Self::Route => "ROUTE",
            Self::Project => "PROJECT",
            Self::Accumulate => "ACCUMULATE",
            Self::StateUpdate => "STATE_UPDATE",
            Self::Sample => "SAMPLE",
        }
    }
}

/// Backend slots are candidates, not placement results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BackendSlot {
    Cpu,
    Gpu,
    Ane,
    Remote,
}

/// Evidence state for a compiled native lowering.
///
/// A backend candidate and a native implementation are different facts.  This
/// keeps an incomplete diagnostic kernel from being mistaken for a selectable
/// executor merely because it targets the same device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NativeLoweringStatus {
    /// A real dispatch harness exists but does not establish full-model work.
    PassThroughPoc,
}

/// Exact contract for a compiled native lowering known to the Rust core.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeLowering {
    pub lowering_id: String,
    pub backend: BackendSlot,
    pub exact_model_contract: String,
    pub status: NativeLoweringStatus,
    pub source_parity_verified: bool,
    pub full_model_execution_verified: bool,
    pub promotion_eligible: bool,
    pub selection_rule: String,
}

/// Registered native lowerings at their earned evidence boundary.
///
/// The current Qwen entry is intentionally a pass-through Metal POC.  It is
/// discoverable for engineering work, but never a generic Qwen executor nor a
/// model-execution or throughput claim.
pub fn registered_native_lowerings() -> Vec<NativeLowering> {
    vec![NativeLowering {
        lowering_id: "qwen3b_metal_pass_through_poc".to_owned(),
        backend: BackendSlot::Gpu,
        exact_model_contract: "Qwen2.5-3B-style dense; hidden=2048; q=2048; kv=256; intermediate=11008; f16 prepared weights".to_owned(),
        status: NativeLoweringStatus::PassThroughPoc,
        source_parity_verified: false,
        full_model_execution_verified: false,
        promotion_eligible: false,
        selection_rule: "refuse automatic selection; reopen only after exact source parity and protected complete-model evidence".to_owned(),
    }]
}

/// A model-specific organ contract.  The strings remain opaque to the spine;
/// each model's anatomy/representation owner supplies their meaning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrganSlot {
    pub organ_id: String,
    pub operation: String,
    pub representation: String,
    pub state: String,
    pub route: String,
}

/// A persistent decode-state region owned by a model-aware MegaKernel plan.
///
/// This is deliberately a resource contract rather than a host-vector
/// implementation.  A lowering must bind it to a device allocation (or an
/// explicitly measured host allocation) before it can claim execution.  The
/// contract prevents a common false continuation: rebuilding a KV/recurrent
/// cache from the whole prefix and calling that a decode step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRegion {
    /// Stable, model-scoped identity.  It must survive backend replacement.
    pub region_id: String,
    /// `recurrent_state`, `kv_cache`, or `route_cache`.
    pub kind: String,
    /// Organ that owns the state; global cache state uses `runtime`.
    pub owner: String,
    /// Backend-native allocation remains unresolved until a lowering earns it.
    pub residency: String,
    /// State is preserved over accepted decode steps, never recreated per step.
    pub persistence: String,
}

/// Deterministic continuation contract shared by CPU, GPU, and ANE candidates.
///
/// `position` is an execution fact, not an estimate.  A backend may only
/// advance it after its state update and token acceptance path both complete.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeStateLayout {
    pub position: u64,
    pub regions: Vec<StateRegion>,
    pub continuation_rule: String,
    pub replay_is_not_decode: bool,
}

impl DecodeStateLayout {
    /// Build a conservative model-scoped layout from recognized organs.
    ///
    /// Recognition is intentionally vocabulary-level.  Counts, shapes, bytes,
    /// and device addresses are supplied only by an exact model lowering.
    pub fn from_organs(organs: &[OrganSlot]) -> Self {
        let mut regions = vec![StateRegion {
            region_id: "runtime.route_cache".to_owned(),
            kind: "route_cache".to_owned(),
            owner: "runtime".to_owned(),
            residency: "unresolved".to_owned(),
            persistence: "accepted_decode_session".to_owned(),
        }];
        for organ in organs {
            let operation = organ.operation.to_ascii_lowercase();
            let state = organ.state.to_ascii_lowercase();
            let identity = organ.organ_id.to_ascii_lowercase();
            if operation == "state_update"
                || state.contains("recurrent")
                || identity.contains("recurrent")
                || identity.contains("deltanet")
            {
                regions.push(StateRegion {
                    region_id: format!("{}.recurrent", organ.organ_id),
                    kind: "recurrent_state".to_owned(),
                    owner: organ.organ_id.clone(),
                    residency: "unresolved".to_owned(),
                    persistence: "accepted_decode_session".to_owned(),
                });
            }
            if operation == "sdpa"
                || state.contains("kv")
                || ((identity.contains("attention") || identity.contains("attn"))
                    && !identity.contains("linear_attention")
                    && !identity.contains("deltanet"))
            {
                regions.push(StateRegion {
                    region_id: format!("{}.kv", organ.organ_id),
                    kind: "kv_cache".to_owned(),
                    owner: organ.organ_id.clone(),
                    residency: "unresolved".to_owned(),
                    persistence: "accepted_decode_session".to_owned(),
                });
            }
        }
        regions.sort_by(|left, right| left.region_id.cmp(&right.region_id));
        regions.dedup_by(|left, right| left.region_id == right.region_id);
        Self {
            position: 0,
            regions,
            continuation_rule: "advance_position_only_after_state_update_and_accepted_token; retain_all_regions_across_steps".to_owned(),
            replay_is_not_decode: true,
        }
    }

    /// Advance a model session only after a token was accepted.
    pub fn record_accepted_token(&mut self) {
        self.position = self.position.saturating_add(1);
    }
}

/// Qualification is intentionally not an execution or promotion state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Qualification {
    PlanOnly,
}

/// Deterministic model-aware execution contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MegaKernelPlan {
    pub schema: String,
    pub model_id: String,
    pub stages: Vec<Stage>,
    pub backend_slots: Vec<BackendSlot>,
    pub organ_slots: Vec<OrganSlot>,
    pub dynamic_slots: Vec<String>,
    pub persistent_state: Vec<String>,
    pub decode_state_layout: DecodeStateLayout,
    pub registered_native_lowerings: Vec<NativeLowering>,
    pub qualification: Qualification,
    pub not_a_single_vendor_kernel_claim: bool,
}

impl MegaKernelPlan {
    /// Create an empty plan with the canonical stage/backend order.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            schema: SCHEMA.to_owned(),
            model_id: model_id.into(),
            stages: STAGES.to_vec(),
            backend_slots: vec![
                BackendSlot::Cpu,
                BackendSlot::Gpu,
                BackendSlot::Ane,
                BackendSlot::Remote,
            ],
            organ_slots: Vec::new(),
            dynamic_slots: [
                "token",
                "position",
                "route",
                "representation",
                "state_layout",
                "sampling",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            persistent_state: [
                "weights_or_representation",
                "route_cache",
                "kv_or_recurrent_state",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            decode_state_layout: DecodeStateLayout::from_organs(&[]),
            registered_native_lowerings: registered_native_lowerings(),
            qualification: Qualification::PlanOnly,
            not_a_single_vendor_kernel_claim: true,
        }
    }

    /// Add or replace an organ by identity, preserving deterministic order.
    pub fn upsert_organ(&mut self, organ: OrganSlot) {
        if let Some(existing) = self
            .organ_slots
            .iter_mut()
            .find(|existing| existing.organ_id == organ.organ_id)
        {
            *existing = organ;
        } else {
            self.organ_slots.push(organ);
        }
        self.organ_slots
            .sort_by(|left, right| left.organ_id.cmp(&right.organ_id));
        self.decode_state_layout = DecodeStateLayout::from_organs(&self.organ_slots);
    }

    /// Return the stable JSON representation used by receipts and adapters.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        registered_native_lowerings, BackendSlot, MegaKernelPlan, NativeLoweringStatus, OrganSlot,
        Qualification, Stage, SCHEMA, STAGES,
    };

    fn organ(id: &str) -> OrganSlot {
        OrganSlot {
            organ_id: id.to_owned(),
            operation: "matmul".to_owned(),
            representation: "model_declared".to_owned(),
            state: "model_declared".to_owned(),
            route: "dynamic_if_applicable".to_owned(),
        }
    }

    #[test]
    fn plan_has_stable_stages_and_candidate_backends() {
        let plan = MegaKernelPlan::new("flash-next");
        assert_eq!(plan.schema, SCHEMA);
        assert_eq!(plan.stages, STAGES);
        assert_eq!(
            plan.backend_slots,
            vec![
                BackendSlot::Cpu,
                BackendSlot::Gpu,
                BackendSlot::Ane,
                BackendSlot::Remote
            ]
        );
        assert_eq!(plan.qualification, Qualification::PlanOnly);
        assert!(plan.not_a_single_vendor_kernel_claim);
        assert_eq!(
            plan.registered_native_lowerings,
            registered_native_lowerings()
        );
    }

    #[test]
    fn organ_updates_are_model_scoped_and_sorted() {
        let mut plan = MegaKernelPlan::new("kimi-p0");
        plan.upsert_organ(organ("z"));
        plan.upsert_organ(organ("a"));
        let mut replacement = organ("z");
        replacement.operation = "state_update".to_owned();
        plan.upsert_organ(replacement);
        assert_eq!(
            plan.organ_slots
                .iter()
                .map(|row| row.organ_id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "z"]
        );
        assert_eq!(plan.organ_slots[1].operation, "state_update");
        assert_eq!(plan.decode_state_layout.position, 0);
        assert!(plan
            .decode_state_layout
            .regions
            .iter()
            .any(|region| region.region_id == "z.recurrent"));
    }

    #[test]
    fn state_layout_requires_persistent_state_before_a_decode_advance() {
        let mut plan = MegaKernelPlan::new("flash-next");
        let mut recurrent = organ("linear_attention_state");
        recurrent.operation = "state_update".to_owned();
        plan.upsert_organ(recurrent);
        let mut attention = organ("full_attention_sdpa");
        attention.operation = "sdpa".to_owned();
        plan.upsert_organ(attention);

        let regions = &plan.decode_state_layout.regions;
        assert!(regions
            .iter()
            .any(|region| region.kind == "recurrent_state"));
        assert!(regions.iter().any(|region| region.kind == "kv_cache"));
        assert!(!regions
            .iter()
            .any(|region| region.region_id == "linear_attention_state.kv"));
        assert!(plan.decode_state_layout.replay_is_not_decode);
        assert!(plan
            .decode_state_layout
            .continuation_rule
            .contains("accepted_token"));

        plan.decode_state_layout.record_accepted_token();
        assert_eq!(plan.decode_state_layout.position, 1);
    }

    #[test]
    fn json_wire_contract_is_stable() {
        let plan = MegaKernelPlan::new("flash-next");
        let json = plan.to_json().expect("plan serializes");
        assert!(json.contains("hcli.physical_graph.mega_kernel.v1"));
        assert!(json.contains("STATE_UPDATE"));
        assert!(json.contains("\"ANE\""));
        assert!(json.contains("PLAN_ONLY"));
        assert!(json.contains("replay_is_not_decode"));
        assert_eq!(Stage::StateUpdate.as_str(), "STATE_UPDATE");
    }

    #[test]
    fn qwen_poc_is_not_mistaken_for_a_promotable_model_lowering() {
        let lowering = registered_native_lowerings()
            .into_iter()
            .find(|entry| entry.lowering_id == "qwen3b_metal_pass_through_poc")
            .expect("Qwen POC remains discoverable for its exact contract");
        assert_eq!(lowering.backend, BackendSlot::Gpu);
        assert_eq!(lowering.status, NativeLoweringStatus::PassThroughPoc);
        assert!(!lowering.source_parity_verified);
        assert!(!lowering.full_model_execution_verified);
        assert!(!lowering.promotion_eligible);
        assert!(lowering.selection_rule.starts_with("refuse automatic"));
    }
}
