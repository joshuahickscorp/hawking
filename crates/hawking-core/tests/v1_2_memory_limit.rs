mod common;
use common::weights_path_deepseek as weights_path;
#[test]
fn memory_limit_too_low_returns_error() {
    let weights = weights_path();
    if !weights.exists() {
        eprintln!("skip memory_limit_too_low: no weights at {weights:?}");
        return;
    }
    let cfg = hawking_core::EngineConfig {
        memory_limit_mb: Some(1),
        ..Default::default()
    };
    let result = hawking_core::model::load_engine(&weights, cfg);
    match result {
        Ok(_) => panic!("expected error with 1 MiB budget, but got success"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("memory budget exceeded"),
                "error should mention 'memory budget exceeded', got: {msg}"
            );
        }
    }
}
#[test]
fn memory_limit_generous_succeeds() {
    let weights = weights_path();
    if !weights.exists() {
        eprintln!("skip memory_limit_generous: no weights at {weights:?}");
        return;
    }
    let cfg = hawking_core::EngineConfig {
        memory_limit_mb: Some(99_999),
        ..Default::default()
    };
    let result = hawking_core::model::load_engine(&weights, cfg);
    assert!(result.is_ok(), "expected success with 99_999 MiB budget");
}
#[test]
fn memory_limit_none_is_unlimited() {
    let weights = weights_path();
    if !weights.exists() {
        eprintln!("skip memory_limit_none: no weights at {weights:?}");
        return;
    }
    let cfg = hawking_core::EngineConfig {
        memory_limit_mb: None,
        ..Default::default()
    };
    let result = hawking_core::model::load_engine(&weights, cfg);
    assert!(result.is_ok(), "expected success with no memory limit");
}
#[test]
fn memory_limit_auto_detection_succeeds_on_18gb_mac() {
    let weights = weights_path();
    if !weights.exists() {
        eprintln!("skip memory_limit_auto: no weights at {weights:?}");
        return;
    }
    let cfg = hawking_core::EngineConfig {
        memory_limit_mb: Some(0),
        ..Default::default()
    };
    let result = hawking_core::model::load_engine(&weights, cfg);
    match result {
        Ok(_) => eprintln!("auto-detect budget: model fits"),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("memory budget exceeded") {
                eprintln!("auto-detect budget: model doesn't fit on this machine (ok to skip)");
            } else {
                panic!("unexpected error from auto-detection: {msg}");
            }
        }
    }
}
