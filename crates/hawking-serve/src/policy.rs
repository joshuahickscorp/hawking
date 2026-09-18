//! Pure resolution of the Rust serving policy.
//!
//! The serving front door historically resolves profile, workload, explicit
//! flags, and inherited environment in separate places.  This module records
//! the exact legacy precedence in one typed result before behavior changes are
//! proposed.  The resolver is pure: process-environment reads and writes stay
//! in the small adapters at the boundary.

use crate::{BatchPolicy, EnergyMode, LeverPlan, RuntimeProfile, ServeOptions, WorkloadPack};
use std::collections::BTreeMap;

/// Rust-serving levers whose presence and values affect current policy.
pub const RUST_SERVE_LEVER_KEYS: &[&str] = &[
    "HAWKING_QWEN_Q4K_PREDEC",
    "HAWKING_QWEN_Q4K_LMHEAD",
    "HAWKING_QWEN_VOCAB_PRUNE",
    "HAWKING_QWEN_TCB",
    "HAWKING_QWEN_FFN_DOWN_Q4K",
    "HAWKING_QWEN_PREDEC_F16SCALES",
    "HAWKING_QWEN_F16_KV",
    "HAWKING_QWEN_CONCURRENT_QKV",
    "HAWKING_ENERGY_EFFICIENT",
];

/// Preserve the difference between an absent control and an explicit default.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Requested<T> {
    #[default]
    Omitted,
    Explicit(T),
    /// A deterministic selection adapter supplied the value after inspecting
    /// its declared facts. It is distinct from an operator CLI request.
    Automatic(T),
}

impl<T> Requested<T> {
    pub fn explicit(value: T) -> Self {
        Self::Explicit(value)
    }

    pub fn as_ref(&self) -> Requested<&T> {
        match self {
            Self::Omitted => Requested::Omitted,
            Self::Explicit(value) => Requested::Explicit(value),
            Self::Automatic(value) => Requested::Automatic(value),
        }
    }
}

/// The two Rust entry shapes whose historic policy must remain observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RustServeEntry {
    /// `hawking serve`: the root CLI applies its global profile before `run`.
    HawkingCliServe,
    /// A direct `ServeOptions` embedder: no root CLI prelude is applied.
    EmbeddedServe,
}

/// Presence-preserving representation of an inherited process environment.
///
/// A non-UTF-8 value remains present.  It must therefore prevent a
/// `set-if-unset` operation, just as `std::env::var_os` does in the existing
/// serving path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvironmentValue {
    Absent,
    Text(String),
    NonUtf8,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PolicyEnvironment {
    values: BTreeMap<String, EnvironmentValue>,
}

impl PolicyEnvironment {
    pub fn from_text<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let mut environment = Self::default();
        for (key, value) in pairs {
            environment.set_text(key, value);
        }
        environment
    }

    /// Snapshot only the levers this policy owns; unrelated environment state
    /// is intentionally outside this projection.
    pub fn from_process() -> Self {
        let mut environment = Self::default();
        for key in RUST_SERVE_LEVER_KEYS {
            let value = match std::env::var_os(key) {
                None => EnvironmentValue::Absent,
                Some(value) => match value.into_string() {
                    Ok(value) => EnvironmentValue::Text(value),
                    Err(_) => EnvironmentValue::NonUtf8,
                },
            };
            environment.values.insert((*key).to_owned(), value);
        }
        environment
    }

    pub fn value(&self, key: &str) -> EnvironmentValue {
        self.values
            .get(key)
            .cloned()
            .unwrap_or(EnvironmentValue::Absent)
    }

    pub fn values(&self) -> &BTreeMap<String, EnvironmentValue> {
        &self.values
    }

    fn is_absent(&self, key: &str) -> bool {
        matches!(self.value(key), EnvironmentValue::Absent)
    }

    fn set_text(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.values
            .insert(key.into(), EnvironmentValue::Text(value.into()));
    }
}

/// Why one resolved field has its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicySource {
    BuiltInDefault,
    Workload,
    Explicit,
    Automatic,
    FrontDoorOmittedProfile,
    Profile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPolicyValue<T> {
    pub value: T,
    pub source: PolicySource,
}

/// An operation applies in order.  SetIfUnset honors inherited values; Force
/// deliberately overwrites them for an Exact profile or unset-profile guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvironmentWriteMode {
    SetIfUnset,
    Force,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvironmentOperationSource {
    FrontDoorProfile,
    FrontDoorUnsetProfileGuard,
    ServeBaseline,
    ServeProfile,
    F16KvPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvironmentOperation {
    pub key: String,
    pub value: String,
    pub mode: EnvironmentWriteMode,
    pub source: EnvironmentOperationSource,
}

/// Typed input to the pure Rust-serving policy resolver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RustServePolicyInput {
    pub entry: RustServeEntry,
    /// `Some` when the Hawking CLI applied a valid global profile before serve.
    /// `None` covers direct embedders and an ignored unknown CLI spelling.
    pub front_door_profile: Option<Requested<RuntimeProfile>>,
    /// The profile request consumed by the serve/workload layer. It is separate
    /// from `front_door_profile` because `--auto` can refine serving behavior
    /// after the root CLI has already applied its omitted-profile guard.
    pub profile: Requested<RuntimeProfile>,
    pub workload: Requested<WorkloadPack>,
    pub energy_mode: Requested<EnergyMode>,
    pub batch_policy: Requested<BatchPolicy>,
    pub f16_kv: Requested<bool>,
    /// The optional metadata-only auto-selection that produced `Automatic`
    /// requests. It records facts already read at the CLI boundary; resolving
    /// policy neither opens the artifact nor probes the machine.
    pub auto_selection: Option<AutoPolicySelection>,
    pub inherited_environment: PolicyEnvironment,
}

/// Request data before the process-boundary environment snapshot. The public
/// Hawking CLI constructs this type; `run()` supplies inherited environment at
/// the last possible point and resolves the complete policy once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RustServePolicyRequest {
    pub entry: RustServeEntry,
    pub front_door_profile: Option<Requested<RuntimeProfile>>,
    pub profile: Requested<RuntimeProfile>,
    pub workload: Requested<WorkloadPack>,
    pub energy_mode: Requested<EnergyMode>,
    pub batch_policy: Requested<BatchPolicy>,
    pub f16_kv: Requested<bool>,
    pub auto_selection: Option<AutoPolicySelection>,
}

/// The flags as supplied to the Rust serving policy before legacy sentinel
/// compatibility is resolved. Keeping them in the result distinguishes an
/// absent control from an explicit `default` or `off` spelling even where the
/// current runtime deliberately gives both the same effective value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RustServeRequestedControls {
    pub front_door_profile: Option<Requested<RuntimeProfile>>,
    pub profile: Requested<RuntimeProfile>,
    pub workload: Requested<WorkloadPack>,
    pub energy_mode: Requested<EnergyMode>,
    pub batch_policy: Requested<BatchPolicy>,
    pub f16_kv: Requested<bool>,
}

impl RustServePolicyRequest {
    pub fn with_inherited_environment(
        self,
        inherited_environment: PolicyEnvironment,
    ) -> RustServePolicyInput {
        RustServePolicyInput {
            entry: self.entry,
            front_door_profile: self.front_door_profile,
            profile: self.profile,
            workload: self.workload,
            energy_mode: self.energy_mode,
            batch_policy: self.batch_policy,
            f16_kv: self.f16_kv,
            auto_selection: self.auto_selection,
            inherited_environment,
        }
    }
}

/// Metadata observed by the CLI's bounded `--auto` adapter. It describes a
/// local file selected for serving, not an admission grant or artifact hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoPolicyArtifactFacts {
    pub path: String,
    pub model_name: String,
    pub architecture: String,
    pub byte_len: u64,
    pub native_context_tokens: u64,
}

/// The machine facts that the bounded `--auto` adapter consulted. They are
/// reported as observed inputs rather than re-read by the pure resolver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoPolicyMachineFacts {
    pub name: String,
    pub total_memory_bytes: u64,
    pub os_version: String,
}

/// The adapter's chosen controls and explanation before normal policy
/// precedence applies. `context_tokens` is advisory until a serving capacity
/// contract consumes it; it is not a claim that `run()` changed KV capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoPolicyDecision {
    pub f16_kv: bool,
    pub fast_profile: bool,
    pub energy_efficient: bool,
    pub context_tokens: u64,
    pub rationale: String,
    pub safety_downgrade: Option<String>,
}

/// A complete provenance record for choices supplied by `--auto`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoPolicySelection {
    pub intent: String,
    pub artifact: AutoPolicyArtifactFacts,
    pub machine: AutoPolicyMachineFacts,
    pub decision: AutoPolicyDecision,
}

/// Inspectable policy result for the current Rust serve semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectiveRustServePolicy {
    pub entry: RustServeEntry,
    pub requested: RustServeRequestedControls,
    pub workload: ResolvedPolicyValue<WorkloadPack>,
    /// The global CLI profile action, if this run originated at that front door.
    /// It can differ from `runtime_profile`: legacy `hawking serve` applies it
    /// before the serve layer independently resolves workload defaults.
    pub front_door_profile: Option<ResolvedPolicyValue<RuntimeProfile>>,
    pub runtime_profile: ResolvedPolicyValue<RuntimeProfile>,
    pub energy_mode: ResolvedPolicyValue<EnergyMode>,
    pub batch_policy: ResolvedPolicyValue<BatchPolicy>,
    /// Auto-selection provenance, when the CLI's bounded metadata adapter
    /// participated in this request. It makes automatic values distinguishable
    /// from operator input without making the resolver perform I/O.
    pub auto_selection: Option<AutoPolicySelection>,
    /// The explicit-or-profile policy request, before inherited environment
    /// semantics are applied. `None` means the legacy code asks for no value.
    pub f16_kv_policy: Option<ResolvedPolicyValue<bool>>,
    /// The actual current `HAWKING_QWEN_F16_KV == "1"` result after ordered
    /// operations.  It deliberately exposes the legacy false-does-not-clear
    /// behavior rather than silently correcting it in this extraction.
    pub f16_kv_enabled: bool,
    pub concurrent_qkv: bool,
    pub environment_operations: Vec<EnvironmentOperation>,
    pub projected_environment: PolicyEnvironment,
}

/// The global `hawking --profile` prelude, shared by generate/bench and the
/// serving projection. It intentionally has no serving baseline or workload
/// logic; those remain the responsibility of `RustServePolicyInput`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrontDoorProfilePolicy {
    pub profile: ResolvedPolicyValue<RuntimeProfile>,
    pub environment_operations: Vec<EnvironmentOperation>,
}

const SERVE_BASELINE: &[(&str, &str)] = &[
    ("HAWKING_QWEN_Q4K_PREDEC", "1"),
    ("HAWKING_QWEN_Q4K_LMHEAD", "1"),
    ("HAWKING_QWEN_VOCAB_PRUNE", "32000"),
    ("HAWKING_QWEN_TCB", "1"),
    ("HAWKING_QWEN_FFN_DOWN_Q4K", "1"),
];

fn append_set_if_unset(
    operations: &mut Vec<EnvironmentOperation>,
    key: &str,
    value: &str,
    source: EnvironmentOperationSource,
) {
    operations.push(EnvironmentOperation {
        key: key.to_owned(),
        value: value.to_owned(),
        mode: EnvironmentWriteMode::SetIfUnset,
        source,
    });
}

fn append_force(
    operations: &mut Vec<EnvironmentOperation>,
    key: &str,
    value: &str,
    source: EnvironmentOperationSource,
) {
    operations.push(EnvironmentOperation {
        key: key.to_owned(),
        value: value.to_owned(),
        mode: EnvironmentWriteMode::Force,
        source,
    });
}

fn append_lever_plan(
    operations: &mut Vec<EnvironmentOperation>,
    plan: &LeverPlan,
    source: EnvironmentOperationSource,
) {
    for (key, value) in &plan.set_if_unset {
        append_set_if_unset(operations, key, value, source);
    }
    for key in &plan.force_off {
        append_force(operations, key, "0", source);
    }
}

fn apply_operations_to_snapshot(
    environment: &mut PolicyEnvironment,
    operations: &[EnvironmentOperation],
) {
    for operation in operations {
        match operation.mode {
            EnvironmentWriteMode::SetIfUnset if !environment.is_absent(&operation.key) => {}
            EnvironmentWriteMode::SetIfUnset | EnvironmentWriteMode::Force => {
                environment.set_text(&operation.key, &operation.value);
            }
        }
    }
}

fn environment_is_on(environment: &PolicyEnvironment, key: &str) -> bool {
    matches!(environment.value(key), EnvironmentValue::Text(value) if value == "1")
}

/// Resolve the global Hawking profile without touching the environment.
///
/// `Requested::Omitted` is intentionally Fast-minus-F16-scales, matching the
/// current CLI behavior rather than the stale historical comment that called
/// omission conservative. A caller that receives an unknown string should
/// retain the existing warning/refusal behavior before entering this typed
/// resolver.
pub fn resolve_front_door_profile(profile: Requested<RuntimeProfile>) -> FrontDoorProfilePolicy {
    let profile = match profile {
        Requested::Omitted => ResolvedPolicyValue {
            value: RuntimeProfile::default_when_unset(),
            source: PolicySource::FrontDoorOmittedProfile,
        },
        Requested::Explicit(value) => ResolvedPolicyValue {
            value,
            source: PolicySource::Explicit,
        },
        Requested::Automatic(value) => ResolvedPolicyValue {
            value,
            source: PolicySource::Automatic,
        },
    };
    let plan = profile.value.lever_plan();
    let mut environment_operations = Vec::new();
    append_lever_plan(
        &mut environment_operations,
        &plan,
        EnvironmentOperationSource::FrontDoorProfile,
    );
    if plan.f16_kv == Some(true) {
        append_set_if_unset(
            &mut environment_operations,
            "HAWKING_QWEN_F16_KV",
            "1",
            EnvironmentOperationSource::FrontDoorProfile,
        );
    }
    if plan.concurrent_qkv {
        append_set_if_unset(
            &mut environment_operations,
            "HAWKING_QWEN_CONCURRENT_QKV",
            "1",
            EnvironmentOperationSource::FrontDoorProfile,
        );
    }
    if profile.source == PolicySource::FrontDoorOmittedProfile {
        for key in RuntimeProfile::default_unset_force_off() {
            append_force(
                &mut environment_operations,
                key,
                "0",
                EnvironmentOperationSource::FrontDoorUnsetProfileGuard,
            );
        }
    }
    FrontDoorProfilePolicy {
        profile,
        environment_operations,
    }
}

fn resolve_workload(request: Requested<WorkloadPack>) -> ResolvedPolicyValue<WorkloadPack> {
    match request {
        Requested::Omitted => ResolvedPolicyValue {
            value: WorkloadPack::Default,
            source: PolicySource::BuiltInDefault,
        },
        Requested::Explicit(value) => ResolvedPolicyValue {
            value,
            source: PolicySource::Explicit,
        },
        Requested::Automatic(value) => ResolvedPolicyValue {
            value,
            source: PolicySource::Automatic,
        },
    }
}

fn legacy_requested<T: Clone + PartialEq>(value: &T, sentinel: &T) -> Requested<T> {
    if value == sentinel {
        Requested::Omitted
    } else {
        Requested::Explicit(value.clone())
    }
}

/// Convert the existing `ServeOptions` representation without changing its
/// sentinel behavior.  This is the compatibility adapter used by `run()`.
pub fn legacy_rust_serve_policy_input(
    options: &ServeOptions,
    inherited_environment: PolicyEnvironment,
) -> RustServePolicyInput {
    RustServePolicyInput {
        entry: RustServeEntry::EmbeddedServe,
        front_door_profile: None,
        profile: legacy_requested(&options.runtime_profile, &RuntimeProfile::Default),
        // Legacy ServeOptions has no workload-presence bit. Its Default pack
        // has the same result whether explicit or omitted, so retain its value.
        workload: Requested::Explicit(options.workload.clone()),
        energy_mode: legacy_requested(&options.energy_mode, &EnergyMode::Off),
        batch_policy: legacy_requested(&options.batch_policy, &BatchPolicy::Default),
        f16_kv: match options.f16_kv {
            Some(value) => Requested::Explicit(value),
            None => Requested::Omitted,
        },
        auto_selection: None,
        inherited_environment,
    }
}

/// Resolve the complete current Rust-serving policy without reading or writing
/// process state.  It intentionally preserves current sentinel behavior:
/// explicit `default`/`off` values remain indistinguishable from omission in
/// the legacy serve layer until the CLI presence migration lands.
pub fn resolve_rust_serve_policy(input: RustServePolicyInput) -> EffectiveRustServePolicy {
    let requested = RustServeRequestedControls {
        front_door_profile: input.front_door_profile.clone(),
        profile: input.profile.clone(),
        workload: input.workload.clone(),
        energy_mode: input.energy_mode.clone(),
        batch_policy: input.batch_policy.clone(),
        f16_kv: input.f16_kv.clone(),
    };
    let workload = resolve_workload(input.workload);
    let (workload_profile, workload_energy, workload_batch) = workload.value.defaults();

    // The entry is an authority boundary, not merely a label in the result.
    // A direct embedder cannot acquire the root CLI's global-profile side
    // effects by accidentally carrying a front-door request across a transport
    // adapter.  The public CLI is the sole caller that may apply this stage.
    let front_door = match input.entry {
        RustServeEntry::HawkingCliServe => input
            .front_door_profile
            .clone()
            .map(resolve_front_door_profile),
        RustServeEntry::EmbeddedServe => None,
    };
    let front_door_profile = front_door.as_ref().map(|policy| policy.profile.clone());

    // These three branches deliberately mirror the old `!= Default/Off`
    // sentinels. `requested` above preserves explicit-default intent even
    // where the current effective value still comes from the workload.
    let runtime_profile = match input.profile {
        Requested::Explicit(profile) if profile != RuntimeProfile::Default => ResolvedPolicyValue {
            value: profile,
            source: PolicySource::Explicit,
        },
        Requested::Automatic(profile) if profile != RuntimeProfile::Default => {
            ResolvedPolicyValue {
                value: profile,
                source: PolicySource::Automatic,
            }
        }
        Requested::Omitted | Requested::Explicit(_) | Requested::Automatic(_) => {
            ResolvedPolicyValue {
                value: workload_profile,
                source: PolicySource::Workload,
            }
        }
    };
    let energy_mode = match input.energy_mode {
        Requested::Explicit(energy) if energy != EnergyMode::Off => ResolvedPolicyValue {
            value: energy,
            source: PolicySource::Explicit,
        },
        Requested::Automatic(energy) if energy != EnergyMode::Off => ResolvedPolicyValue {
            value: energy,
            source: PolicySource::Automatic,
        },
        Requested::Omitted | Requested::Explicit(_) | Requested::Automatic(_) => {
            ResolvedPolicyValue {
                value: workload_energy,
                source: PolicySource::Workload,
            }
        }
    };
    let batch_policy = match input.batch_policy {
        Requested::Explicit(policy) if policy != BatchPolicy::Default => ResolvedPolicyValue {
            value: policy,
            source: PolicySource::Explicit,
        },
        Requested::Automatic(policy) if policy != BatchPolicy::Default => ResolvedPolicyValue {
            value: policy,
            source: PolicySource::Automatic,
        },
        Requested::Omitted | Requested::Explicit(_) | Requested::Automatic(_) => {
            ResolvedPolicyValue {
                value: workload_batch,
                source: PolicySource::Workload,
            }
        }
    };

    let serve_plan = runtime_profile.value.lever_plan();
    let f16_kv_policy = match input.f16_kv {
        Requested::Explicit(value) => Some(ResolvedPolicyValue {
            value,
            source: PolicySource::Explicit,
        }),
        Requested::Automatic(value) => Some(ResolvedPolicyValue {
            value,
            source: PolicySource::Automatic,
        }),
        Requested::Omitted => serve_plan.f16_kv.map(|value| ResolvedPolicyValue {
            value,
            source: PolicySource::Profile,
        }),
    };

    let mut environment_operations = front_door
        .as_ref()
        .map(|policy| policy.environment_operations.clone())
        .unwrap_or_default();

    for (key, value) in SERVE_BASELINE {
        append_set_if_unset(
            &mut environment_operations,
            key,
            value,
            EnvironmentOperationSource::ServeBaseline,
        );
    }
    append_lever_plan(
        &mut environment_operations,
        &serve_plan,
        EnvironmentOperationSource::ServeProfile,
    );
    if f16_kv_policy.as_ref().is_some_and(|policy| policy.value) {
        append_set_if_unset(
            &mut environment_operations,
            "HAWKING_QWEN_F16_KV",
            "1",
            EnvironmentOperationSource::F16KvPolicy,
        );
    }

    let mut projected_environment = input.inherited_environment;
    apply_operations_to_snapshot(&mut projected_environment, &environment_operations);
    let f16_kv_enabled = environment_is_on(&projected_environment, "HAWKING_QWEN_F16_KV");
    let concurrent_qkv = serve_plan.concurrent_qkv
        || environment_is_on(&projected_environment, "HAWKING_QWEN_CONCURRENT_QKV");

    EffectiveRustServePolicy {
        entry: input.entry,
        requested,
        workload,
        front_door_profile,
        runtime_profile,
        energy_mode,
        batch_policy,
        auto_selection: input.auto_selection,
        f16_kv_policy,
        f16_kv_enabled,
        concurrent_qkv,
        environment_operations,
        projected_environment,
    }
}

/// Apply already-resolved operations at the process boundary. The same
/// SetIfUnset/Force semantics were used by the pre-extraction serve path.
pub fn apply_environment_operations(operations: &[EnvironmentOperation]) {
    for operation in operations {
        match operation.mode {
            EnvironmentWriteMode::SetIfUnset if std::env::var_os(&operation.key).is_some() => {}
            EnvironmentWriteMode::SetIfUnset | EnvironmentWriteMode::Force => {
                std::env::set_var(&operation.key, &operation.value);
            }
        }
    }
}
