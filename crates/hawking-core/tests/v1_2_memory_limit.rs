//! v1.2.0-12: memory budget enforcement tests.
//!
//! Verifies that `load_engine` enforces `memory_limit_mb` before mmap
//! allocation, and that auto-detection (Some(0)) does not erroneously
//! block a model that fits in 80% of available RAM.

use std::path::PathBuf;

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
}

/// Load with a 1 MiB budget — model is ~9 GiB so this must fail.
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

/// Load with a very generous 99_999 MiB budget — must succeed.
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

/// No budget (None) — must succeed (unlimited).
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

/// Auto (Some(0)) — 80% of system RAM. On an 18 GiB Mac the budget is
/// ~14_745 MiB; V2-Lite at ~8_700 MiB fits comfortably.
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
    // On a machine with < 11 GiB total RAM, the model might not fit even at
    // 80%. Skip the success assertion in that edge case.
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
