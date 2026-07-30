use hawking_serve::{BatchPolicy, EnergyMode, RuntimeProfile, WorkloadPack};
fn has(set: &[(&'static str, &'static str)], k: &str) -> bool {
    set.iter().any(|(kk, _)| *kk == k)
}
fn val<'a>(set: &'a [(&'static str, &'static str)], k: &str) -> Option<&'a str> {
    set.iter().find(|(kk, _)| *kk == k).map(|(_, v)| *v)
}
#[test]
fn workload_from_str_roundtrips_all_known() {
    for s in [
        "default",
        "code-completion",
        "chat-shared-prompt",
        "batch-summarization",
        "local-agent-loop",
    ] {
        assert_eq!(
            WorkloadPack::from_str(s).expect("known workload").as_str(),
            s
        );
    }
    assert!(WorkloadPack::from_str("nonsense-pack").is_none());
    assert!(WorkloadPack::from_str("fast").is_none());
}
#[test]
fn workload_defaults_match_documented_triples() {
    use BatchPolicy as BP;
    use EnergyMode as EM;
    use RuntimeProfile as RP;
    let cases: &[(WorkloadPack, RP, EM, BP)] = &[
        (WorkloadPack::Default, RP::Default, EM::Off, BP::Default),
        (
            WorkloadPack::CodeCompletion,
            RP::Race,
            EM::Off,
            BP::GreedyFirst,
        ),
        (
            WorkloadPack::ChatSharedPrompt,
            RP::Fast,
            EM::Balanced,
            BP::PrefixGrouped,
        ),
        (
            WorkloadPack::BatchSummarization,
            RP::Efficient,
            EM::Efficient,
            BP::GreedyFirst,
        ),
        (
            WorkloadPack::LocalAgentLoop,
            RP::Fast,
            EM::Off,
            BP::GreedyFirst,
        ),
    ];
    for (pack, want_profile, want_energy, want_policy) in cases {
        let (profile, energy, policy) = pack.defaults();
        assert_eq!(&profile, want_profile, "{pack} profile");
        assert_eq!(&energy, want_energy, "{pack} energy");
        assert_eq!(&policy, want_policy, "{pack} batch policy");
    }
}
#[test]
fn workload_expands_to_expected_profile_and_knobs() {
    {
        let (profile, _e, _p) = WorkloadPack::CodeCompletion.defaults();
        assert_eq!(profile, RuntimeProfile::Race);
        let plan = profile.lever_plan();
        for k in [
            "HAWKING_QWEN_Q4K_LMHEAD",
            "HAWKING_QWEN_Q4K_PREDEC",
            "HAWKING_QWEN_PREDEC_F16SCALES",
            "HAWKING_QWEN_VOCAB_PRUNE",
            "HAWKING_QWEN_FFN_DOWN_Q4K",
        ] {
            assert!(
                has(&plan.set_if_unset, k),
                "code-completion(race) must set {k}"
            );
        }
        assert_eq!(
            val(&plan.set_if_unset, "HAWKING_QWEN_VOCAB_PRUNE"),
            Some("32000")
        );
        assert_eq!(plan.f16_kv, Some(true), "race enables f16-KV");
        assert!(plan.concurrent_qkv);
        assert!(plan.force_off.is_empty());
        assert!(!has(&plan.set_if_unset, "HAWKING_ENERGY_EFFICIENT"));
    }
    {
        let (profile, _e, _p) = WorkloadPack::BatchSummarization.defaults();
        assert_eq!(profile, RuntimeProfile::Efficient);
        let plan = profile.lever_plan();
        assert!(has(&plan.set_if_unset, "HAWKING_ENERGY_EFFICIENT"));
        assert!(has(&plan.set_if_unset, "HAWKING_QWEN_Q4K_PREDEC"));
        assert_eq!(plan.f16_kv, Some(true));
    }
    {
        let (profile, energy, policy) = WorkloadPack::ChatSharedPrompt.defaults();
        assert_eq!(profile, RuntimeProfile::Fast);
        assert_eq!(energy, EnergyMode::Balanced);
        assert_eq!(policy, BatchPolicy::PrefixGrouped);
        let plan = profile.lever_plan();
        assert_eq!(plan.f16_kv, Some(false), "fast leaves f16-KV off");
        assert!(plan.force_off.is_empty());
        assert!(!has(&plan.set_if_unset, "HAWKING_ENERGY_EFFICIENT"));
    }
    {
        let (profile, energy, policy) = WorkloadPack::LocalAgentLoop.defaults();
        assert_eq!(profile, RuntimeProfile::Fast);
        assert_eq!(energy, EnergyMode::Off);
        assert_eq!(policy, BatchPolicy::GreedyFirst);
        let plan = profile.lever_plan();
        assert!(has(&plan.set_if_unset, "HAWKING_QWEN_Q4K_LMHEAD"));
        assert_eq!(plan.f16_kv, Some(false));
    }
    {
        let (profile, energy, policy) = WorkloadPack::Default.defaults();
        assert_eq!(profile, RuntimeProfile::Default);
        assert_eq!(energy, EnergyMode::Off);
        assert_eq!(policy, BatchPolicy::Default);
        let plan = profile.lever_plan();
        assert!(plan.set_if_unset.is_empty(), "default sets no lever");
        assert!(plan.force_off.is_empty());
        assert_eq!(plan.f16_kv, None);
        assert!(!plan.concurrent_qkv);
    }
}
#[test]
fn workload_energy_maps_to_gather_window_ms() {
    assert_eq!(WorkloadPack::Default.defaults().1.gather_window_ms(), 0);
    assert_eq!(
        WorkloadPack::CodeCompletion.defaults().1.gather_window_ms(),
        0
    );
    assert_eq!(
        WorkloadPack::ChatSharedPrompt
            .defaults()
            .1
            .gather_window_ms(),
        3
    );
    assert_eq!(
        WorkloadPack::BatchSummarization
            .defaults()
            .1
            .gather_window_ms(),
        8
    );
    assert_eq!(
        WorkloadPack::LocalAgentLoop.defaults().1.gather_window_ms(),
        0
    );
}
