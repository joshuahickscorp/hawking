//! Track 8.2 gate — native `/v1/hawking/generate` request mapping + stats
//! object shape. Pure: tests the in-schema -> GenerateRequest mapping and the
//! GenStats-shaped final stats JSON, no engine/server/GPU. Runs despite the
//! #[ignore]'d generation-path integration tests.
//!   cargo test -p hawking-serve --test hawking_native_endpoint

use hawking_serve::http::{
    hawking_generate_stats_json, map_hawking_generate_req, HawkingGenerateReq,
};

fn parse(json: serde_json::Value) -> HawkingGenerateReq {
    serde_json::from_value(json).expect("parse HawkingGenerateReq")
}

#[test]
fn absent_temperature_maps_to_greedy() {
    let r = parse(serde_json::json!({"prompt": "hi", "max_tokens": 8}));
    let g = map_hawking_generate_req(&r);
    assert_eq!(g.prompt, "hi");
    assert_eq!(g.max_new_tokens, 8);
    assert_eq!(g.sampling.temperature, 0.0);
    assert_eq!(g.sampling.top_k, 0);
    assert_eq!(g.sampling.top_p, 1.0);
    assert!(g.stop.is_empty());
    assert!(g.abort.is_none());
}

#[test]
fn sampling_fields_and_stop_round_trip() {
    let r = parse(serde_json::json!({
        "prompt": "p", "max_tokens": 32, "temperature": 0.7,
        "top_p": 0.95, "seed": 123, "stop": ["</s>", "\n\n"]
    }));
    let g = map_hawking_generate_req(&r);
    assert_eq!(g.sampling.temperature, 0.7);
    assert!((g.sampling.top_p - 0.95).abs() < 1e-6);
    assert_eq!(g.sampling.top_k, 40);
    assert_eq!(g.sampling.seed, Some(123));
    assert_eq!(g.stop, vec!["</s>".to_string(), "\n\n".to_string()]);
}

#[test]
fn default_max_tokens_applied_when_absent() {
    let r = parse(serde_json::json!({"prompt": "x"}));
    let g = map_hawking_generate_req(&r);
    assert_eq!(g.max_new_tokens, 256);
}

#[test]
fn stats_json_has_genstats_field_names_and_dec_tps() {
    // 64 tokens / 2000 ms = 32 tok/s — mirrors GenStats::dec_tps formula.
    let j = hawking_generate_stats_json(14, 64, 2000.0, true, "q4k-predec-f16s");
    assert_eq!(j["prompt_tokens"], 14);
    assert_eq!(j["completion_tokens"], 64);
    assert_eq!(j["decode_ms"], 2000.0);
    assert_eq!(j["dec_tps"].as_f64().unwrap().round(), 32.0);
    assert_eq!(j["token_only_path_used"], true);
    assert_eq!(j["lm_head_path"], "q4k-predec-f16s");
    // parseable + compact (no heavy fields)
    let s = serde_json::to_string(&j).unwrap();
    assert!(serde_json::from_str::<serde_json::Value>(&s).is_ok());
}

#[test]
fn stats_dec_tps_zero_safe_when_no_decode() {
    let j = hawking_generate_stats_json(5, 0, 0.0, false, "f16");
    assert!(j["dec_tps"].as_f64().unwrap().is_finite());
}
