#![cfg(target_os = "macos")]

use std::path::PathBuf;

#[test]
fn residual_f16_is_rejected_clearly() {
    let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
    if !weights.exists() {
        eprintln!("skipping residual dtype rejection test: no weights at {weights:?}");
        return;
    }
    let cfg = dismantle_core::EngineConfig {
        residual_dtype: dismantle_core::ResidualDtype::F16,
        ..Default::default()
    };
    let err = match dismantle_core::model::load_engine(&weights, cfg) {
        Ok(_) => panic!("residual f16 must be rejected"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("residual_dtype=f16 is not supported"),
        "unexpected error: {msg}"
    );
}
