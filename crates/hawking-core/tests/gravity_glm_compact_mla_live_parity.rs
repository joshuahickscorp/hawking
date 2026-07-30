#![cfg(target_os = "macos")]
use hawking_core::gravity_glm::gpu::GravityGlmGpu;
use hawking_core::gravity_glm::{
    GPU_COMPACT_ATTENTION_ICB_ENV, GPU_COMPACT_MLA_ENV, GPU_DEVICE_DSA_ENV, GPU_DEVICE_ROUTER_ENV,
    GPU_EXPERT_TABLE_HIT_ENV, GPU_EXPERT_TABLE_ICB_ENV, GPU_EXPERT_WAVE_CONCURRENT_ENV,
    GPU_EXPERT_WAVE_ENV, GPU_LM_HEAD_ENV, GPU_LM_HEAD_FULL_LOGITS_ENV, GPU_LM_HEAD_ICB_ENV,
};
use hawking_core::metal::MetalContext;
use hawking_core::numeric_parity::{score_pair, Bounds};
use std::path::PathBuf;
fn invalid_compact_geometry_fixture(source: &std::path::Path) -> tempfile::TempDir {
    let invalid = tempfile::tempdir().expect("temporary invalid compact fixture");
    for entry in std::fs::read_dir(source).expect("read compact fixture") {
        let entry = entry.expect("fixture directory entry");
        if entry.file_type().expect("fixture entry type").is_file() {
            std::fs::copy(entry.path(), invalid.path().join(entry.file_name()))
                .expect("copy compact fixture file");
        }
    }
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(invalid.path().join("model.gravity.index.json"))
            .expect("copied gravity index"),
    )
    .expect("parse copied gravity index");
    let kv_name = "model.layers.0.self_attn.kv_b_proj.weight";
    let shard_name = index["weight_map"][kv_name]
        .as_str()
        .expect("KV owning shard");
    let shard_path = invalid.path().join(shard_name);
    let mut shard = std::fs::read(&shard_path).expect("read copied compact shard");
    let header_len = u64::from_le_bytes(shard[12..20].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&shard[20..20 + header_len]).expect("parse shard header");
    let descriptor = header["tensors"]
        .as_array()
        .expect("shard tensors")
        .iter()
        .find(|tensor| tensor["name"].as_str() == Some(kv_name))
        .expect("KV descriptor");
    let payload =
        20 + header_len + descriptor["offset"].as_u64().expect("KV payload offset") as usize;
    shard[payload + 8..payload + 10].copy_from_slice(&16u16.to_le_bytes()); // D
    shard[payload + 12..payload + 14].copy_from_slice(&16u16.to_le_bytes()); // sub
    shard[payload + 24..payload + 28].copy_from_slice(&2u32.to_le_bytes()); // nchunk
    std::fs::write(shard_path, shard).expect("write invalid compact shard");
    invalid
}
#[test]
fn compact_mla_complete_tokens_match_expanded_v21_and_exact_decisions() {
    let Some(dir) = std::env::var_os("HAWKING_GLM_COMPACT_FIXTURE_DIR").map(PathBuf::from) else {
        eprintln!("skip: set HAWKING_GLM_COMPACT_FIXTURE_DIR to a bounded direct-u8 PQ fixture");
        return;
    };
    let Ok(expanded_ctx) = MetalContext::new() else {
        eprintln!("skip: no Metal device");
        return;
    };
    let compact_ctx = MetalContext::new().expect("second Metal context");
    let device_dsa_ctx = MetalContext::new().expect("device DSA Metal context");
    let device_router_ctx = MetalContext::new().expect("device router Metal context");
    let device_head_ctx = MetalContext::new().expect("device head Metal context");
    let device_table_cold_ctx = MetalContext::new().expect("cold expert-table Metal context");
    let invalid_ctx = MetalContext::new().expect("invalid-admission Metal context");
    let misconfigured_ctx = MetalContext::new().expect("misconfigured DSA Metal context");
    let misconfigured_router_ctx = MetalContext::new().expect("misconfigured router Metal context");
    let prior_compact = std::env::var_os(GPU_COMPACT_MLA_ENV);
    let prior_compact_attention_icb = std::env::var_os(GPU_COMPACT_ATTENTION_ICB_ENV);
    let prior_device_dsa = std::env::var_os(GPU_DEVICE_DSA_ENV);
    let prior_device_router = std::env::var_os(GPU_DEVICE_ROUTER_ENV);
    let prior_expert_wave = std::env::var_os(GPU_EXPERT_WAVE_ENV);
    let prior_expert_wave_concurrent = std::env::var_os(GPU_EXPERT_WAVE_CONCURRENT_ENV);
    let prior_expert_table = std::env::var_os(GPU_EXPERT_TABLE_HIT_ENV);
    let prior_expert_table_icb = std::env::var_os(GPU_EXPERT_TABLE_ICB_ENV);
    let prior_head = std::env::var_os(GPU_LM_HEAD_ENV);
    let prior_head_icb = std::env::var_os(GPU_LM_HEAD_ICB_ENV);
    let prior_full_logits = std::env::var_os(GPU_LM_HEAD_FULL_LOGITS_ENV);
    std::env::remove_var(GPU_DEVICE_DSA_ENV);
    std::env::remove_var(GPU_COMPACT_ATTENTION_ICB_ENV);
    std::env::remove_var(GPU_DEVICE_ROUTER_ENV);
    std::env::remove_var(GPU_EXPERT_WAVE_ENV);
    std::env::remove_var(GPU_EXPERT_WAVE_CONCURRENT_ENV);
    std::env::remove_var(GPU_EXPERT_TABLE_HIT_ENV);
    std::env::remove_var(GPU_EXPERT_TABLE_ICB_ENV);
    std::env::remove_var(GPU_LM_HEAD_ENV);
    std::env::remove_var(GPU_LM_HEAD_ICB_ENV);
    std::env::remove_var(GPU_LM_HEAD_FULL_LOGITS_ENV);
    std::env::set_var(GPU_DEVICE_DSA_ENV, "1");
    let mode_error = match GravityGlmGpu::open_dir_with_budget_resident(
        misconfigured_ctx,
        &dir,
        true,
        512 * 1024 * 1024,
        true,
    ) {
        Ok(_) => panic!("device DSA was admitted without compact MLA"),
        Err(error) => error,
    };
    assert!(
        mode_error
            .to_string()
            .contains("requires resident state and"),
        "device DSA mode coupling did not fail closed: {mode_error}"
    );
    std::env::remove_var(GPU_DEVICE_DSA_ENV);
    std::env::set_var(GPU_DEVICE_ROUTER_ENV, "1");
    let router_mode_error = match GravityGlmGpu::open_dir_with_budget_resident(
        misconfigured_router_ctx,
        &dir,
        true,
        512 * 1024 * 1024,
        false,
    ) {
        Ok(_) => panic!("device router was admitted without resident state"),
        Err(error) => error,
    };
    assert!(
        router_mode_error
            .to_string()
            .contains("requires resident state"),
        "device router mode coupling did not fail closed: {router_mode_error}"
    );
    std::env::remove_var(GPU_DEVICE_ROUTER_ENV);
    let invalid = invalid_compact_geometry_fixture(&dir);
    std::env::set_var(GPU_COMPACT_MLA_ENV, "1");
    let invalid_error = match GravityGlmGpu::open_dir_with_budget_resident(
        invalid_ctx,
        invalid.path(),
        true,
        512 * 1024 * 1024,
        true,
    ) {
        Ok(_) => panic!("D16 compact KV geometry was admitted"),
        Err(error) => error,
    };
    assert!(
        invalid_error.to_string().contains("dim=16")
            && invalid_error.to_string().contains("unsupported"),
        "invalid compact geometry did not fail in header preflight: {invalid_error}"
    );
    std::env::remove_var(GPU_COMPACT_MLA_ENV);
    let expanded = GravityGlmGpu::open_dir_with_budget_resident(
        expanded_ctx,
        &dir,
        true,
        512 * 1024 * 1024,
        true,
    )
    .expect("expanded resident fixture");
    std::env::set_var(GPU_COMPACT_MLA_ENV, "1");
    let compact = GravityGlmGpu::open_dir_with_budget_resident(
        compact_ctx,
        &dir,
        true,
        512 * 1024 * 1024,
        true,
    )
    .expect("compact resident fixture");
    std::env::remove_var(GPU_COMPACT_MLA_ENV);
    std::env::set_var(GPU_COMPACT_MLA_ENV, "1");
    std::env::set_var(GPU_DEVICE_DSA_ENV, "1");
    let compact_device_dsa = GravityGlmGpu::open_dir_with_budget_resident(
        device_dsa_ctx,
        &dir,
        true,
        512 * 1024 * 1024,
        true,
    )
    .expect("compact resident device DSA fixture");
    std::env::remove_var(GPU_DEVICE_DSA_ENV);
    std::env::remove_var(GPU_COMPACT_MLA_ENV);
    std::env::set_var(GPU_COMPACT_MLA_ENV, "1");
    std::env::set_var(GPU_DEVICE_DSA_ENV, "1");
    std::env::set_var(GPU_DEVICE_ROUTER_ENV, "1");
    let compact_device_router = GravityGlmGpu::open_dir_with_budget_resident(
        device_router_ctx,
        &dir,
        true,
        512 * 1024 * 1024,
        true,
    )
    .expect("compact resident device DSA plus router fixture");
    std::env::remove_var(GPU_DEVICE_ROUTER_ENV);
    std::env::remove_var(GPU_DEVICE_DSA_ENV);
    std::env::remove_var(GPU_COMPACT_MLA_ENV);
    std::env::set_var(GPU_COMPACT_MLA_ENV, "1");
    std::env::set_var(GPU_DEVICE_DSA_ENV, "1");
    std::env::set_var(GPU_DEVICE_ROUTER_ENV, "1");
    std::env::set_var(GPU_LM_HEAD_ENV, "1");
    std::env::set_var(GPU_LM_HEAD_FULL_LOGITS_ENV, "1");
    let compact_device_head = GravityGlmGpu::open_dir_with_budget_resident(
        device_head_ctx,
        &dir,
        true,
        512 * 1024 * 1024,
        true,
    )
    .expect("compact resident device DSA, router, and head fixture");
    std::env::remove_var(GPU_LM_HEAD_FULL_LOGITS_ENV);
    std::env::remove_var(GPU_LM_HEAD_ENV);
    std::env::remove_var(GPU_DEVICE_ROUTER_ENV);
    std::env::remove_var(GPU_DEVICE_DSA_ENV);
    std::env::remove_var(GPU_COMPACT_MLA_ENV);
    std::env::set_var(GPU_COMPACT_MLA_ENV, "1");
    std::env::set_var(GPU_DEVICE_DSA_ENV, "1");
    std::env::set_var(GPU_DEVICE_ROUTER_ENV, "1");
    std::env::set_var(GPU_EXPERT_WAVE_ENV, "1");
    std::env::set_var(GPU_EXPERT_TABLE_HIT_ENV, "1");
    std::env::set_var(GPU_EXPERT_TABLE_ICB_ENV, "1");
    std::env::set_var(GPU_LM_HEAD_ENV, "1");
    std::env::set_var(GPU_LM_HEAD_FULL_LOGITS_ENV, "1");
    let compact_device_table_cold = GravityGlmGpu::open_dir_with_budget_resident(
        device_table_cold_ctx,
        &dir,
        true,
        512 * 1024 * 1024,
        true,
    )
    .expect("cold cache-indexed expert-table fixture");
    std::env::remove_var(GPU_LM_HEAD_FULL_LOGITS_ENV);
    std::env::remove_var(GPU_LM_HEAD_ENV);
    std::env::remove_var(GPU_EXPERT_TABLE_HIT_ENV);
    std::env::remove_var(GPU_EXPERT_TABLE_ICB_ENV);
    std::env::remove_var(GPU_EXPERT_WAVE_ENV);
    std::env::remove_var(GPU_DEVICE_ROUTER_ENV);
    std::env::remove_var(GPU_DEVICE_DSA_ENV);
    std::env::remove_var(GPU_COMPACT_MLA_ENV);
    let receipt: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("compact_mla_fixture_receipt.json"))
            .expect("compact sparse fixture receipt"),
    )
    .expect("parse compact sparse fixture receipt");
    assert_eq!(receipt["production_artifact"], false);
    assert_eq!(receipt["runtime_default_enabled"], false);
    assert_eq!(receipt["layers"], 1);
    assert_eq!(receipt["mlp_schedule"], serde_json::json!(["sparse"]));
    let direct_u8 =
        serde_json::json!({"dim": 32, "subspaces": 1, "sub": 32, "cardinality": 256, "bits": 8});
    assert_eq!(receipt["physical_attention_codec"], direct_u8);
    for field in ["dim", "subspaces", "sub", "cardinality", "bits"] {
        assert_eq!(
            receipt["physical_routed_expert_codec"][field],
            direct_u8[field]
        );
    }
    assert_eq!(
        receipt["physical_routed_expert_codec"]["projection_tensors"],
        27
    );
    assert_eq!(receipt["direct_u8_validation"]["validated_tensors"], 29);
    assert_eq!(receipt["direct_u8_validation"]["status"], "PASS");
    assert_eq!(
        receipt["fp64_complete_token_authority"]["selection_patterns"],
        4
    );
    #[derive(serde::Deserialize)]
    struct Authority {
        tokens: Vec<u32>,
        logits: Vec<f64>,
        final_topk: Vec<usize>,
        expert_choices: Vec<Vec<usize>>,
    }
    let authorities: Vec<Authority> = serde_json::from_slice(
        &std::fs::read(dir.join("ref_logits_f64.json"))
            .expect("explicit FP64 complete-token authorities"),
    )
    .expect("parse FP64 complete-token authorities");
    let cold_authority = &authorities[0];
    std::env::set_var(GPU_COMPACT_MLA_ENV, "1");
    std::env::set_var(GPU_DEVICE_DSA_ENV, "1");
    std::env::set_var(GPU_DEVICE_ROUTER_ENV, "1");
    std::env::set_var(GPU_EXPERT_WAVE_ENV, "1");
    std::env::set_var(GPU_EXPERT_TABLE_HIT_ENV, "1");
    std::env::set_var(GPU_EXPERT_TABLE_ICB_ENV, "1");
    std::env::set_var(GPU_LM_HEAD_ENV, "1");
    std::env::set_var(GPU_LM_HEAD_FULL_LOGITS_ENV, "1");
    let (cold_table_logits, cold_table_trace) = compact_device_table_cold
        .forward(&cold_authority.tokens)
        .expect("cold cache-indexed expert-table miss fallback");
    std::env::remove_var(GPU_LM_HEAD_FULL_LOGITS_ENV);
    std::env::remove_var(GPU_LM_HEAD_ENV);
    std::env::remove_var(GPU_EXPERT_TABLE_HIT_ENV);
    std::env::remove_var(GPU_EXPERT_TABLE_ICB_ENV);
    std::env::remove_var(GPU_EXPERT_WAVE_ENV);
    std::env::remove_var(GPU_DEVICE_ROUTER_ENV);
    std::env::remove_var(GPU_DEVICE_DSA_ENV);
    std::env::remove_var(GPU_COMPACT_MLA_ENV);
    let cold_table_waits = compact_device_table_cold
        .last_resident_waits()
        .expect("cold cache-indexed expert-table resident wait count");
    let cold_table_pair = score_pair(
        &cold_table_logits,
        &cold_table_logits,
        &cold_authority.logits,
        &Bounds::logits(),
    );
    assert!(
        cold_table_pair.pass,
        "cold cache-indexed miss fallback failed complete-token V2.1: {cold_table_pair:#?}"
    );
    assert_eq!(
        cold_table_trace.final_topk, cold_authority.final_topk,
        "cold cache-indexed miss fallback changed exact DSA selection"
    );
    assert_eq!(
        cold_table_trace.expert_choices, cold_authority.expert_choices,
        "cold cache-indexed miss fallback changed exact expert choices"
    );
    for (case, authority) in authorities.iter().enumerate() {
        let prompt = &authority.tokens;
        let (expanded_logits, expanded_trace) = expanded.forward(prompt).expect("expanded forward");
        let (compact_logits, compact_trace) = compact.forward(prompt).expect("compact forward");
        let compact_waits = compact
            .last_resident_waits()
            .expect("compact resident wait count");
        let (device_dsa_direct_logits, device_dsa_direct_trace) = compact_device_dsa
            .forward(prompt)
            .expect("direct-encoded compact device DSA forward");
        let device_dsa_direct_waits = compact_device_dsa
            .last_resident_waits()
            .expect("direct-encoded device DSA resident wait count");
        std::env::set_var(GPU_COMPACT_MLA_ENV, "1");
        std::env::set_var(GPU_DEVICE_DSA_ENV, "1");
        std::env::set_var(GPU_COMPACT_ATTENTION_ICB_ENV, "1");
        let (device_dsa_logits, device_dsa_trace) = compact_device_dsa
            .forward(prompt)
            .expect("ICB compact device DSA forward");
        std::env::remove_var(GPU_COMPACT_ATTENTION_ICB_ENV);
        std::env::remove_var(GPU_DEVICE_DSA_ENV);
        std::env::remove_var(GPU_COMPACT_MLA_ENV);
        let device_dsa_waits = compact_device_dsa
            .last_resident_waits()
            .expect("ICB device DSA resident wait count");
        std::env::set_var(GPU_DEVICE_ROUTER_ENV, "1");
        let (device_router_logits, device_router_trace) = compact_device_router
            .forward(prompt)
            .expect("compact device DSA plus router forward");
        std::env::remove_var(GPU_DEVICE_ROUTER_ENV);
        let device_router_waits = compact_device_router
            .last_resident_waits()
            .expect("device router resident wait count");
        std::env::set_var(GPU_COMPACT_MLA_ENV, "1");
        std::env::set_var(GPU_DEVICE_DSA_ENV, "1");
        std::env::set_var(GPU_DEVICE_ROUTER_ENV, "1");
        std::env::set_var(GPU_LM_HEAD_ENV, "1");
        std::env::set_var(GPU_LM_HEAD_ICB_ENV, "1");
        std::env::set_var(GPU_LM_HEAD_FULL_LOGITS_ENV, "1");
        let ((device_head_logits, device_head_trace), device_head_report) = if prompt.len() == 1 {
            hawking_core::cost_ledger::set_enabled(true);
            let _ = hawking_core::cost_ledger::end_token();
            assert!(hawking_core::cost_ledger::begin_token());
            let result = compact_device_head
                .forward(prompt)
                .expect("profiled compact device final norm plus head forward");
            let report =
                hawking_core::cost_ledger::end_token().expect("device final-head ICB ledger");
            hawking_core::cost_ledger::set_enabled(false);
            (result, Some(report))
        } else {
            (
                compact_device_head
                    .forward(prompt)
                    .expect("compact device final norm plus head forward"),
                None,
            )
        };
        std::env::remove_var(GPU_LM_HEAD_FULL_LOGITS_ENV);
        std::env::remove_var(GPU_LM_HEAD_ICB_ENV);
        std::env::remove_var(GPU_LM_HEAD_ENV);
        std::env::remove_var(GPU_DEVICE_ROUTER_ENV);
        std::env::remove_var(GPU_DEVICE_DSA_ENV);
        std::env::remove_var(GPU_COMPACT_MLA_ENV);
        let device_head_waits = compact_device_head
            .last_resident_waits()
            .expect("device final norm plus head resident wait count");
        std::env::set_var(GPU_COMPACT_MLA_ENV, "1");
        std::env::set_var(GPU_DEVICE_DSA_ENV, "1");
        std::env::set_var(GPU_DEVICE_ROUTER_ENV, "1");
        std::env::set_var(GPU_EXPERT_WAVE_ENV, "1");
        std::env::set_var(GPU_LM_HEAD_ENV, "1");
        std::env::set_var(GPU_LM_HEAD_FULL_LOGITS_ENV, "1");
        let (expert_wave_logits, expert_wave_trace) = compact_device_head
            .forward(prompt)
            .expect("sequential projection expert-wave forward");
        std::env::remove_var(GPU_LM_HEAD_FULL_LOGITS_ENV);
        std::env::remove_var(GPU_LM_HEAD_ENV);
        std::env::remove_var(GPU_EXPERT_WAVE_ENV);
        std::env::remove_var(GPU_DEVICE_ROUTER_ENV);
        std::env::remove_var(GPU_DEVICE_DSA_ENV);
        std::env::remove_var(GPU_COMPACT_MLA_ENV);
        let expert_wave_waits = compact_device_head
            .last_resident_waits()
            .expect("sequential expert-wave resident wait count");
        std::env::set_var(GPU_COMPACT_MLA_ENV, "1");
        std::env::set_var(GPU_DEVICE_DSA_ENV, "1");
        std::env::set_var(GPU_DEVICE_ROUTER_ENV, "1");
        std::env::set_var(GPU_EXPERT_WAVE_ENV, "1");
        std::env::set_var(GPU_EXPERT_WAVE_CONCURRENT_ENV, "1");
        std::env::set_var(GPU_LM_HEAD_ENV, "1");
        std::env::set_var(GPU_LM_HEAD_FULL_LOGITS_ENV, "1");
        let (concurrent_wave_logits, concurrent_wave_trace) = compact_device_head
            .forward(prompt)
            .expect("concurrent projection expert-wave forward");
        std::env::remove_var(GPU_LM_HEAD_FULL_LOGITS_ENV);
        std::env::remove_var(GPU_LM_HEAD_ENV);
        std::env::remove_var(GPU_EXPERT_WAVE_CONCURRENT_ENV);
        std::env::remove_var(GPU_EXPERT_WAVE_ENV);
        std::env::remove_var(GPU_DEVICE_ROUTER_ENV);
        std::env::remove_var(GPU_DEVICE_DSA_ENV);
        std::env::remove_var(GPU_COMPACT_MLA_ENV);
        let concurrent_wave_waits = compact_device_head
            .last_resident_waits()
            .expect("concurrent expert-wave resident wait count");
        std::env::set_var(GPU_COMPACT_MLA_ENV, "1");
        std::env::set_var(GPU_DEVICE_DSA_ENV, "1");
        std::env::set_var(GPU_DEVICE_ROUTER_ENV, "1");
        std::env::set_var(GPU_EXPERT_WAVE_ENV, "1");
        std::env::set_var(GPU_EXPERT_TABLE_HIT_ENV, "1");
        std::env::set_var(GPU_EXPERT_TABLE_ICB_ENV, "1");
        std::env::set_var(GPU_LM_HEAD_ENV, "1");
        std::env::set_var(GPU_LM_HEAD_FULL_LOGITS_ENV, "1");
        let _ = compact_device_head
            .forward(prompt)
            .expect("persistent expert-table route prewarm");
        let ((table_wave_logits, table_wave_trace), table_hit_report) = if prompt.len() == 1 {
            hawking_core::cost_ledger::set_enabled(true);
            let _ = hawking_core::cost_ledger::end_token();
            assert!(hawking_core::cost_ledger::begin_token());
            let result = compact_device_head
                .forward(prompt)
                .expect("profiled persistent cache-indexed expert-table wave forward");
            let report =
                hawking_core::cost_ledger::end_token().expect("persistent table-hit ledger");
            hawking_core::cost_ledger::set_enabled(false);
            (result, Some(report))
        } else {
            (
                compact_device_head
                    .forward(prompt)
                    .expect("persistent cache-indexed expert-table wave forward"),
                None,
            )
        };
        std::env::remove_var(GPU_LM_HEAD_FULL_LOGITS_ENV);
        std::env::remove_var(GPU_LM_HEAD_ENV);
        std::env::remove_var(GPU_EXPERT_TABLE_HIT_ENV);
        std::env::remove_var(GPU_EXPERT_TABLE_ICB_ENV);
        std::env::remove_var(GPU_EXPERT_WAVE_ENV);
        std::env::remove_var(GPU_DEVICE_ROUTER_ENV);
        std::env::remove_var(GPU_DEVICE_DSA_ENV);
        std::env::remove_var(GPU_COMPACT_MLA_ENV);
        let table_wave_waits = compact_device_head
            .last_resident_waits()
            .expect("cache-indexed expert-table resident wait count");
        assert!(
            !authority.expert_choices.is_empty(),
            "prompt {prompt:?}: sparse router authority is vacuous"
        );
        let pair = score_pair(
            &expanded_logits,
            &compact_logits,
            &authority.logits,
            &Bounds::logits(),
        );
        let device_dsa_pair = score_pair(
            &expanded_logits,
            &device_dsa_logits,
            &authority.logits,
            &Bounds::logits(),
        );
        let device_router_pair = score_pair(
            &expanded_logits,
            &device_router_logits,
            &authority.logits,
            &Bounds::logits(),
        );
        let device_head_pair = score_pair(
            &expanded_logits,
            &device_head_logits,
            &authority.logits,
            &Bounds::logits(),
        );
        let expert_wave_pair = score_pair(
            &expanded_logits,
            &expert_wave_logits,
            &authority.logits,
            &Bounds::logits(),
        );
        let concurrent_wave_pair = score_pair(
            &expanded_logits,
            &concurrent_wave_logits,
            &authority.logits,
            &Bounds::logits(),
        );
        let table_wave_pair = score_pair(
            &expanded_logits,
            &table_wave_logits,
            &authority.logits,
            &Bounds::logits(),
        );
        assert!(
            pair.pass,
            "case {case} prompt {prompt:?}: compact complete-token V2.1 {pair:#?}"
        );
        assert!(
            device_dsa_pair.pass,
            "case {case} prompt {prompt:?}: device DSA complete-token V2.1 {device_dsa_pair:#?}"
        );
        assert_eq!(
            device_dsa_logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            device_dsa_direct_logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "case {case}: compact-attention ICB and direct device-DSA logits must be bit-exact"
        );
        assert_eq!(
            device_dsa_trace.final_topk, device_dsa_direct_trace.final_topk,
            "case {case}: compact-attention ICB cannot change exact DSA selection"
        );
        assert_eq!(
            device_dsa_trace.expert_choices, device_dsa_direct_trace.expert_choices,
            "case {case}: compact-attention ICB cannot change expert choices"
        );
        assert_eq!(
            device_dsa_waits, device_dsa_direct_waits,
            "case {case}: compact-attention ICB cannot change waits"
        );
        assert!(device_router_pair.pass, "case {case} prompt {prompt:?}: device router complete-token V2.1 {device_router_pair:#?}");
        assert!(
            device_head_pair.pass,
            "case {case} prompt {prompt:?}: device final norm + head complete-token V2.1 {device_head_pair:#?}"
        );
        assert!(expert_wave_pair.pass, "case {case} prompt {prompt:?}: sequential expert-wave complete-token V2.1 {expert_wave_pair:#?}");
        assert!(
            concurrent_wave_pair.pass,
            "case {case} prompt {prompt:?}: concurrent expert-wave complete-token V2.1 {concurrent_wave_pair:#?}"
        );
        assert!(
            table_wave_pair.pass,
            "case {case} prompt {prompt:?}: cache-indexed expert-table complete-token V2.1 {table_wave_pair:#?}"
        );
        assert_eq!(
            expanded_trace.final_topk, authority.final_topk,
            "case {case}: expanded exact DSA selection vs FP64 authority"
        );
        assert_eq!(
            compact_trace.final_topk, authority.final_topk,
            "case {case}: compact exact DSA selection vs FP64 authority"
        );
        assert_eq!(
            expanded_trace.expert_choices, authority.expert_choices,
            "case {case}: expanded exact expert choices vs FP64 authority"
        );
        assert_eq!(
            compact_trace.expert_choices, authority.expert_choices,
            "case {case}: compact exact expert choices vs FP64 authority"
        );
        assert_eq!(
            device_dsa_trace.final_topk, authority.final_topk,
            "case {case}: exact device DSA selection vs FP64 authority"
        );
        assert_eq!(
            device_dsa_trace.expert_choices, authority.expert_choices,
            "case {case}: exact device DSA expert choices vs FP64 authority"
        );
        assert_eq!(
            device_router_trace.final_topk, authority.final_topk,
            "case {case}: exact device-router DSA selection vs FP64 authority"
        );
        assert_eq!(
            device_router_trace.expert_choices, authority.expert_choices,
            "case {case}: exact device router expert choices vs FP64 authority"
        );
        assert_eq!(
            device_head_trace.final_topk, authority.final_topk,
            "case {case}: exact device-head DSA selection vs FP64 authority"
        );
        assert_eq!(
            device_head_trace.expert_choices, authority.expert_choices,
            "case {case}: exact device-head expert choices vs FP64 authority"
        );
        assert_eq!(
            device_head_trace.sample_token.map(|token| token as usize),
            device_head_pair.device.discrete.greedy_argmax_ref,
            "case {case}: device greedy readback must match FP64 authority"
        );
        assert_eq!(
            device_head_trace
                .head_topk_idx
                .iter()
                .map(|&index| index as usize)
                .collect::<Vec<_>>(),
            device_head_pair.device.discrete.top_k_ref,
            "case {case}: device top-k readback must match FP64 authority"
        );
        assert!(
            device_head_trace.head_full_logits_readback,
            "case {case}: full logits were requested for V2.1 scoring"
        );
        assert_eq!(
            concurrent_wave_logits, expert_wave_logits,
            "case {case}: independent concurrent projection groups must be bit-exact to the sequential wave"
        );
        assert_eq!(
            concurrent_wave_trace.final_topk, expert_wave_trace.final_topk,
            "case {case}: concurrent projection scheduling cannot change DSA"
        );
        assert_eq!(
            concurrent_wave_trace.expert_choices, expert_wave_trace.expert_choices,
            "case {case}: concurrent projection scheduling cannot change expert choices"
        );
        assert_eq!(
            concurrent_wave_trace.sample_token, expert_wave_trace.sample_token,
            "case {case}: concurrent projection scheduling cannot change greedy readback"
        );
        assert_eq!(
            concurrent_wave_trace.head_topk_idx, expert_wave_trace.head_topk_idx,
            "case {case}: concurrent projection scheduling cannot change head top-k"
        );
        assert_eq!(
            table_wave_trace.final_topk, expert_wave_trace.final_topk,
            "case {case}: cache-indexed expert table cannot change DSA"
        );
        assert_eq!(
            table_wave_trace.expert_choices, expert_wave_trace.expert_choices,
            "case {case}: deferred device trace must preserve exact expert choices"
        );
        assert_eq!(
            table_wave_trace.sample_token, expert_wave_trace.sample_token,
            "case {case}: cache-indexed expert table cannot change greedy readback"
        );
        assert_eq!(
            table_wave_trace.head_topk_idx, expert_wave_trace.head_topk_idx,
            "case {case}: cache-indexed expert table cannot change head top-k"
        );
        assert_eq!(
            compact_waits.saturating_sub(device_dsa_waits),
            (4 * prompt.len()) as u64,
            "case {case}: two attention-prelude and two full-indexer drains must be removed per token"
        );
        assert_eq!(
            device_router_waits, device_dsa_waits,
            "case {case}: device router selection must reuse the existing router commit"
        );
        assert_eq!(
            device_head_waits, device_router_waits,
            "case {case}: final RMSNorm must append to the existing device-head commit"
        );
        if let Some(report) = &device_head_report {
            let head_cb = report
                .device
                .command_buffers
                .iter()
                .find(|sample| sample.stage_key == "mixed:kv_and_norm+final_head+sampling")
                .expect("final-head ICB must retain exact mixed stage ownership");
            let composition: Vec<(&str, u64)> = head_cb
                .stage_composition
                .iter()
                .map(|entry| (entry.stage, entry.dispatches))
                .collect();
            assert_eq!(
                composition,
                vec![("kv_and_norm", 1), ("final_head", 1), ("sampling", 2)]
            );
            assert_eq!(head_cb.stage_dispatches_total, 4);
            assert!(head_cb.stage_dispatches_match_buffer);
        }
        assert_eq!(
            device_head_waits.saturating_sub(expert_wave_waits),
            (2 * prompt.len()) as u64,
            "case {case}: one expert wave must replace three expert batches per token"
        );
        assert_eq!(
            concurrent_wave_waits, expert_wave_waits,
            "case {case}: projection concurrency must not add command buffers or waits"
        );
        if case == 0 {
            assert!(
                cold_table_waits >= table_wave_waits && cold_table_waits <= expert_wave_waits,
                "cold table waits {cold_table_waits} must fall between persistent warm \
                 {table_wave_waits} and qualified fallback {expert_wave_waits}"
            );
        }
        assert!(
            table_wave_waits <= concurrent_wave_waits,
            "case {case}: persistent table routing cannot add waits after prewarm"
        );
        if prompt.len() == 1 {
            assert_eq!(
                concurrent_wave_waits.saturating_sub(table_wave_waits),
                1,
                "case {case}: a stable one-token route must merge router and expert wave"
            );
            let report = table_hit_report
                .as_ref()
                .expect("single-token persistent hit must be profiled");
            assert!(
                report
                    .transfers
                    .iter()
                    .all(|transfer| transfer.kind != "device_expert_table_snapshot_upload"),
                "case {case}: a persistent hit must not rebuild or upload its descriptor table"
            );
            assert_eq!(
                report.counters.routed_representations.r4_projection_touches, 6,
                "case {case}: two routed R4 triplets must remain visible to the profiler"
            );
        }
    }
    match prior_compact {
        Some(value) => std::env::set_var(GPU_COMPACT_MLA_ENV, value),
        None => std::env::remove_var(GPU_COMPACT_MLA_ENV),
    }
    match prior_compact_attention_icb {
        Some(value) => std::env::set_var(GPU_COMPACT_ATTENTION_ICB_ENV, value),
        None => std::env::remove_var(GPU_COMPACT_ATTENTION_ICB_ENV),
    }
    match prior_device_dsa {
        Some(value) => std::env::set_var(GPU_DEVICE_DSA_ENV, value),
        None => std::env::remove_var(GPU_DEVICE_DSA_ENV),
    }
    match prior_device_router {
        Some(value) => std::env::set_var(GPU_DEVICE_ROUTER_ENV, value),
        None => std::env::remove_var(GPU_DEVICE_ROUTER_ENV),
    }
    match prior_expert_wave {
        Some(value) => std::env::set_var(GPU_EXPERT_WAVE_ENV, value),
        None => std::env::remove_var(GPU_EXPERT_WAVE_ENV),
    }
    match prior_expert_wave_concurrent {
        Some(value) => std::env::set_var(GPU_EXPERT_WAVE_CONCURRENT_ENV, value),
        None => std::env::remove_var(GPU_EXPERT_WAVE_CONCURRENT_ENV),
    }
    match prior_expert_table {
        Some(value) => std::env::set_var(GPU_EXPERT_TABLE_HIT_ENV, value),
        None => std::env::remove_var(GPU_EXPERT_TABLE_HIT_ENV),
    }
    match prior_expert_table_icb {
        Some(value) => std::env::set_var(GPU_EXPERT_TABLE_ICB_ENV, value),
        None => std::env::remove_var(GPU_EXPERT_TABLE_ICB_ENV),
    }
    match prior_head {
        Some(value) => std::env::set_var(GPU_LM_HEAD_ENV, value),
        None => std::env::remove_var(GPU_LM_HEAD_ENV),
    }
    match prior_head_icb {
        Some(value) => std::env::set_var(GPU_LM_HEAD_ICB_ENV, value),
        None => std::env::remove_var(GPU_LM_HEAD_ICB_ENV),
    }
    match prior_full_logits {
        Some(value) => std::env::set_var(GPU_LM_HEAD_FULL_LOGITS_ENV, value),
        None => std::env::remove_var(GPU_LM_HEAD_FULL_LOGITS_ENV),
    }
}
