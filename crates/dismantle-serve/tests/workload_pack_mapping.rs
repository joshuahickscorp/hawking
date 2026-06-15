//! Confirmation test for Track 9.3 `--workload` packs: lock in that each
//! workload string expands to the documented (profile, energy, batch-policy)
//! triple AND that the chosen profile expands to the expected env-lever knobs.
//!
//! This is the integration-test sibling of the inline `profile_lever_tests`
//! module in `dismantle-serve/src/lib.rs` (which covers RuntimeProfile alone).
//! Here we pin the WORKLOAD → (profile, energy, policy) layer plus the
//! workload → concrete knob set, so a silent drift in either
//! `WorkloadPack::defaults()` or `RuntimeProfile::lever_plan()` fails CI.
//!
//! Pure: builds data-only mappings, touches no process env, no model. Gates:
//!
//!   cargo test -p dismantle-serve --test workload_pack_mapping

use dismantle_serve::{BatchPolicy, EnergyMode, RuntimeProfile, WorkloadPack};

fn has(set: &[(&'static str, &'static str)], k: &str) -> bool {
    set.iter().any(|(kk, _)| *kk == k)
}

fn val<'a>(set: &'a [(&'static str, &'static str)], k: &str) -> Option<&'a str> {
    set.iter().find(|(kk, _)| *kk == k).map(|(_, v)| *v)
}

/// from_str round-trips every known workload name and rejects the unknown.
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
            s,
            "workload {s} must round-trip through from_str/as_str"
        );
    }
    assert!(WorkloadPack::from_str("nonsense-pack").is_none());
    // a runtime-profile name is NOT a workload pack name.
    assert!(WorkloadPack::from_str("fast").is_none());
}

/// Each workload expands to its documented (profile, energy, batch-policy)
/// triple (see the `WorkloadPack` doc comment in lib.rs).
#[test]
fn workload_defaults_match_documented_triples() {
    use BatchPolicy as BP;
    use EnergyMode as EM;
    use RuntimeProfile as RP;

    let cases: &[(WorkloadPack, RP, EM, BP)] = &[
        (WorkloadPack::Default, RP::Default, EM::Off, BP::Default),
        (WorkloadPack::CodeCompletion, RP::Race, EM::Off, BP::GreedyFirst),
        (WorkloadPack::ChatSharedPrompt, RP::Fast, EM::Balanced, BP::PrefixGrouped),
        (
            WorkloadPack::BatchSummarization,
            RP::Efficient,
            EM::Efficient,
            BP::GreedyFirst,
        ),
        (WorkloadPack::LocalAgentLoop, RP::Fast, EM::Off, BP::GreedyFirst),
    ];

    for (pack, want_profile, want_energy, want_policy) in cases {
        let (profile, energy, policy) = pack.defaults();
        assert_eq!(&profile, want_profile, "{pack} profile");
        assert_eq!(&energy, want_energy, "{pack} energy");
        assert_eq!(&policy, want_policy, "{pack} batch policy");
    }
}

/// End-to-end: workload → profile → concrete lever knobs. This is the layer the
/// server actually applies; pin the full knob set per workload so a regression
/// in either the pack mapping or the profile bundle is caught.
#[test]
fn workload_expands_to_expected_profile_and_knobs() {
    // code-completion ⇒ Race ⇒ fast bundle + f16-KV on + concurrent QKV.
    {
        let (profile, _e, _p) = WorkloadPack::CodeCompletion.defaults();
        assert_eq!(profile, RuntimeProfile::Race);
        let plan = profile.lever_plan();
        for k in [
            "DISMANTLE_QWEN_Q4K_LMHEAD",
            "DISMANTLE_QWEN_Q4K_PREDEC",
            "DISMANTLE_QWEN_PREDEC_F16SCALES",
            "DISMANTLE_QWEN_VOCAB_PRUNE",
            "DISMANTLE_QWEN_FFN_DOWN_Q4K",
        ] {
            assert!(has(&plan.set_if_unset, k), "code-completion(race) must set {k}");
        }
        assert_eq!(val(&plan.set_if_unset, "DISMANTLE_QWEN_VOCAB_PRUNE"), Some("32000"));
        assert_eq!(plan.f16_kv, Some(true), "race enables f16-KV");
        assert!(plan.concurrent_qkv);
        assert!(plan.force_off.is_empty());
        // race is NOT the energy profile.
        assert!(!has(&plan.set_if_unset, "DISMANTLE_ENERGY_EFFICIENT"));
    }

    // batch-summarization ⇒ Efficient ⇒ fast bundle + energy flag + f16-KV.
    {
        let (profile, _e, _p) = WorkloadPack::BatchSummarization.defaults();
        assert_eq!(profile, RuntimeProfile::Efficient);
        let plan = profile.lever_plan();
        assert!(
            has(&plan.set_if_unset, "DISMANTLE_ENERGY_EFFICIENT"),
            "efficient must set the energy lever"
        );
        assert!(has(&plan.set_if_unset, "DISMANTLE_QWEN_Q4K_PREDEC"));
        assert_eq!(plan.f16_kv, Some(true));
    }

    // chat-shared-prompt ⇒ Fast ⇒ fast bundle, f16-KV OFF (bit-identity to fast).
    {
        let (profile, energy, policy) = WorkloadPack::ChatSharedPrompt.defaults();
        assert_eq!(profile, RuntimeProfile::Fast);
        assert_eq!(energy, EnergyMode::Balanced);
        assert_eq!(policy, BatchPolicy::PrefixGrouped);
        let plan = profile.lever_plan();
        assert_eq!(plan.f16_kv, Some(false), "fast leaves f16-KV off");
        assert!(plan.force_off.is_empty());
        assert!(!has(&plan.set_if_unset, "DISMANTLE_ENERGY_EFFICIENT"));
    }

    // local-agent-loop ⇒ Fast ⇒ same fast bundle, energy OFF, greedy-first.
    {
        let (profile, energy, policy) = WorkloadPack::LocalAgentLoop.defaults();
        assert_eq!(profile, RuntimeProfile::Fast);
        assert_eq!(energy, EnergyMode::Off);
        assert_eq!(policy, BatchPolicy::GreedyFirst);
        let plan = profile.lever_plan();
        assert!(has(&plan.set_if_unset, "DISMANTLE_QWEN_Q4K_LMHEAD"));
        assert_eq!(plan.f16_kv, Some(false));
    }

    // default ⇒ Default ⇒ touches nothing (bit-identical golden path).
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

/// The energy mode each workload selects expands to its gather-window ms — the
/// number the serve loop actually sleeps. Pins the EnergyMode → ms contract.
#[test]
fn workload_energy_maps_to_gather_window_ms() {
    assert_eq!(WorkloadPack::Default.defaults().1.gather_window_ms(), 0);
    assert_eq!(WorkloadPack::CodeCompletion.defaults().1.gather_window_ms(), 0);
    assert_eq!(WorkloadPack::ChatSharedPrompt.defaults().1.gather_window_ms(), 3);
    assert_eq!(WorkloadPack::BatchSummarization.defaults().1.gather_window_ms(), 8);
    assert_eq!(WorkloadPack::LocalAgentLoop.defaults().1.gather_window_ms(), 0);
}
