use hawking_serve::{
    legacy_rust_serve_policy_input, resolve_front_door_profile, resolve_rust_serve_policy,
    AutoPolicyArtifactFacts, AutoPolicyDecision, AutoPolicyMachineFacts, AutoPolicySelection,
    BatchPolicy, EnergyMode, EnvironmentOperationSource, EnvironmentValue, PolicyEnvironment,
    PolicySource, Requested, RuntimeProfile, RustServeEntry, RustServePolicyInput, ServeOptions,
    WorkloadPack,
};

fn input() -> RustServePolicyInput {
    RustServePolicyInput {
        entry: RustServeEntry::EmbeddedServe,
        front_door_profile: None,
        profile: Requested::Omitted,
        workload: Requested::Omitted,
        energy_mode: Requested::Omitted,
        batch_policy: Requested::Omitted,
        f16_kv: Requested::Omitted,
        auto_selection: None,
        inherited_environment: PolicyEnvironment::default(),
    }
}

fn auto_selection() -> AutoPolicySelection {
    AutoPolicySelection {
        intent: "max-speed".into(),
        artifact: AutoPolicyArtifactFacts {
            path: "/models/qwen.gguf".into(),
            model_name: "Qwen fixture".into(),
            architecture: "qwen2".into(),
            byte_len: 1_900_000_000,
            native_context_tokens: 32_768,
        },
        machine: AutoPolicyMachineFacts {
            name: "test mac".into(),
            total_memory_bytes: 18 << 30,
            os_version: "test os".into(),
        },
        decision: AutoPolicyDecision {
            f16_kv: true,
            fast_profile: true,
            energy_efficient: true,
            context_tokens: 8_192,
            rationale: "fixture selection".into(),
            safety_downgrade: None,
        },
    }
}

#[test]
fn embedded_default_retains_the_existing_conservative_serve_policy() {
    let policy = resolve_rust_serve_policy(input());

    assert!(policy.front_door_profile.is_none());
    assert_eq!(policy.workload.value, WorkloadPack::Default);
    assert_eq!(policy.workload.source, PolicySource::BuiltInDefault);
    assert_eq!(policy.runtime_profile.value, RuntimeProfile::Default);
    assert_eq!(policy.runtime_profile.source, PolicySource::Workload);
    assert_eq!(policy.energy_mode.value, EnergyMode::Off);
    assert_eq!(policy.batch_policy.value, BatchPolicy::Default);
    assert_eq!(policy.f16_kv_policy, None);
    assert!(!policy.f16_kv_enabled);
    assert!(!policy.concurrent_qkv);
    assert_eq!(
        policy
            .projected_environment
            .value("HAWKING_QWEN_Q4K_PREDEC"),
        EnvironmentValue::Text("1".into())
    );
}

#[test]
fn cli_omitted_profile_exposes_the_existing_two_stage_fast_default() {
    let mut request = input();
    request.entry = RustServeEntry::HawkingCliServe;
    request.front_door_profile = Some(Requested::Omitted);
    let policy = resolve_rust_serve_policy(request);

    assert_eq!(
        policy.front_door_profile.unwrap().value,
        RuntimeProfile::Fast,
        "the root CLI applies Fast when --profile is absent"
    );
    assert_eq!(
        policy.runtime_profile.value,
        RuntimeProfile::Default,
        "the service layer separately resolves its default workload profile"
    );
    assert!(policy.concurrent_qkv);
    assert_eq!(
        policy
            .projected_environment
            .value("HAWKING_QWEN_PREDEC_F16SCALES"),
        EnvironmentValue::Text("0".into()),
        "the omitted-profile quality guard wins after the Fast bundle"
    );
    assert_eq!(
        policy
            .projected_environment
            .value("HAWKING_QWEN_CONCURRENT_QKV"),
        EnvironmentValue::Text("1".into())
    );
    assert!(policy.environment_operations.iter().any(|operation| {
        operation.source == EnvironmentOperationSource::FrontDoorUnsetProfileGuard
            && operation.key == "HAWKING_QWEN_PREDEC_F16SCALES"
            && operation.value == "0"
    }));
}

#[test]
fn cli_serve_reuses_the_same_front_door_profile_projection_as_other_cli_actions() {
    let front_door = resolve_front_door_profile(Requested::Omitted);
    let mut request = input();
    request.entry = RustServeEntry::HawkingCliServe;
    request.front_door_profile = Some(Requested::Omitted);
    let policy = resolve_rust_serve_policy(request);

    assert_eq!(
        &policy.environment_operations[..front_door.environment_operations.len()],
        front_door.environment_operations.as_slice(),
        "the serving projection must not recreate the root CLI profile mapping"
    );
}

#[test]
fn embedded_callers_cannot_apply_a_cli_front_door_profile() {
    let mut request = input();
    request.front_door_profile = Some(Requested::Explicit(RuntimeProfile::Fast));

    let policy = resolve_rust_serve_policy(request);

    assert!(policy.front_door_profile.is_none());
    assert!(policy.environment_operations.iter().all(|operation| {
        operation.source != EnvironmentOperationSource::FrontDoorProfile
            && operation.source != EnvironmentOperationSource::FrontDoorUnsetProfileGuard
    }));
    assert!(!policy.concurrent_qkv);
}

#[test]
fn explicit_legacy_default_values_currently_lose_to_a_nondefault_workload() {
    let mut request = input();
    request.profile = Requested::Explicit(RuntimeProfile::Default);
    request.workload = Requested::Explicit(WorkloadPack::ChatSharedPrompt);
    request.energy_mode = Requested::Explicit(EnergyMode::Off);
    request.batch_policy = Requested::Explicit(BatchPolicy::Default);

    let policy = resolve_rust_serve_policy(request);

    assert_eq!(policy.runtime_profile.value, RuntimeProfile::Fast);
    assert_eq!(policy.runtime_profile.source, PolicySource::Workload);
    assert_eq!(
        policy.requested.profile,
        Requested::Explicit(RuntimeProfile::Default)
    );
    assert_eq!(policy.energy_mode.value, EnergyMode::Balanced);
    assert_eq!(policy.energy_mode.source, PolicySource::Workload);
    assert_eq!(
        policy.requested.energy_mode,
        Requested::Explicit(EnergyMode::Off)
    );
    assert_eq!(policy.batch_policy.value, BatchPolicy::PrefixGrouped);
    assert_eq!(policy.batch_policy.source, PolicySource::Workload);
    assert_eq!(
        policy.requested.batch_policy,
        Requested::Explicit(BatchPolicy::Default)
    );
}

#[test]
fn explicit_false_f16_kv_preserves_the_legacy_inherited_true_value() {
    let mut request = input();
    request.profile = Requested::Explicit(RuntimeProfile::Fast);
    request.f16_kv = Requested::Explicit(false);
    request.inherited_environment = PolicyEnvironment::from_text([("HAWKING_QWEN_F16_KV", "1")]);

    let policy = resolve_rust_serve_policy(request);

    assert_eq!(policy.f16_kv_policy.unwrap().value, false);
    assert!(
        policy.f16_kv_enabled,
        "this characterization guards against accidentally changing the false-does-not-clear behavior"
    );
    assert_eq!(
        policy.projected_environment.value("HAWKING_QWEN_F16_KV"),
        EnvironmentValue::Text("1".into())
    );
}

#[test]
fn automatic_choices_keep_machine_and_artifact_provenance() {
    let selection = auto_selection();
    let mut request = input();
    request.profile = Requested::Automatic(RuntimeProfile::Fast);
    request.energy_mode = Requested::Automatic(EnergyMode::Efficient);
    request.f16_kv = Requested::Automatic(true);
    request.auto_selection = Some(selection.clone());

    let policy = resolve_rust_serve_policy(request);

    assert_eq!(policy.runtime_profile.value, RuntimeProfile::Fast);
    assert_eq!(policy.runtime_profile.source, PolicySource::Automatic);
    assert_eq!(policy.energy_mode.value, EnergyMode::Efficient);
    assert_eq!(policy.energy_mode.source, PolicySource::Automatic);
    assert_eq!(
        policy.f16_kv_policy.unwrap().source,
        PolicySource::Automatic
    );
    assert_eq!(policy.auto_selection, Some(selection));
}

#[test]
fn inherited_concurrent_qkv_zero_cannot_override_a_fast_profile_today() {
    let mut request = input();
    request.profile = Requested::Explicit(RuntimeProfile::Fast);
    request.inherited_environment =
        PolicyEnvironment::from_text([("HAWKING_QWEN_CONCURRENT_QKV", "0")]);

    let policy = resolve_rust_serve_policy(request);

    assert!(policy.concurrent_qkv);
    assert_eq!(
        policy
            .projected_environment
            .value("HAWKING_QWEN_CONCURRENT_QKV"),
        EnvironmentValue::Text("0".into())
    );
}

#[test]
fn exact_forces_quality_trade_levers_off_after_inherited_values() {
    let mut request = input();
    request.profile = Requested::Explicit(RuntimeProfile::Exact);
    request.inherited_environment = PolicyEnvironment::from_text([
        ("HAWKING_QWEN_PREDEC_F16SCALES", "1"),
        ("HAWKING_QWEN_FFN_DOWN_Q4K", "1"),
        ("HAWKING_QWEN_VOCAB_PRUNE", "32000"),
    ]);

    let policy = resolve_rust_serve_policy(request);

    for key in [
        "HAWKING_QWEN_PREDEC_F16SCALES",
        "HAWKING_QWEN_FFN_DOWN_Q4K",
        "HAWKING_QWEN_VOCAB_PRUNE",
    ] {
        assert_eq!(
            policy.projected_environment.value(key),
            EnvironmentValue::Text("0".into()),
            "Exact must force {key} off"
        );
    }
}

#[test]
fn every_workload_uses_its_declared_triple_when_no_nondefault_override_exists() {
    for workload in [
        WorkloadPack::Default,
        WorkloadPack::CodeCompletion,
        WorkloadPack::ChatSharedPrompt,
        WorkloadPack::BatchSummarization,
        WorkloadPack::LocalAgentLoop,
    ] {
        let expected = workload.defaults();
        let mut request = input();
        request.workload = Requested::Explicit(workload.clone());

        let policy = resolve_rust_serve_policy(request);
        assert_eq!(policy.runtime_profile.value, expected.0, "{workload}");
        assert_eq!(policy.energy_mode.value, expected.1, "{workload}");
        assert_eq!(policy.batch_policy.value, expected.2, "{workload}");
    }
}

#[test]
fn legacy_options_adapter_characterizes_the_existing_sentinel_behavior() {
    let options = ServeOptions {
        workload: WorkloadPack::BatchSummarization,
        runtime_profile: RuntimeProfile::Default,
        energy_mode: EnergyMode::Off,
        batch_policy: BatchPolicy::Default,
        ..Default::default()
    };
    let policy = resolve_rust_serve_policy(legacy_rust_serve_policy_input(
        &options,
        PolicyEnvironment::default(),
    ));

    assert_eq!(policy.entry, RustServeEntry::EmbeddedServe);
    assert_eq!(policy.runtime_profile.value, RuntimeProfile::Efficient);
    assert_eq!(policy.energy_mode.value, EnergyMode::Efficient);
    assert_eq!(policy.batch_policy.value, BatchPolicy::GreedyFirst);
    assert!(policy.f16_kv_enabled);
}
