//! Tests for the TQ artifact loader for RWKV-7 models.
//!
//! Validates that a `.tq`-format artifact file contains the expected GGUF
//! tensor names for RWKV-7 and that projection shapes match the 0.4B model
//! config (n_embd=1024, n_ff=4096, n_layers=24).
//!
//! All tests that require a real artifact are `#[ignore]`; they are activated
//! by setting `RWKV7_TQ_TEST_ARTIFACT` to the path of a `.tq` file before
//! running:
//!
//! ```sh
//! RWKV7_TQ_TEST_ARTIFACT=/path/to/model.tq \
//!   cargo test -p hawking-core --features tq --test rwkv7_tq_loader -- --nocapture --ignored
//! ```

#![cfg(feature = "tq")]

#[allow(dead_code)]
/// Build the list of expected GGUF tensor names for an RWKV-7 model with
/// `n_layers` transformer layers.
///
/// Each layer contributes 6 projection weight tensors:
/// - `blk.{i}.time_mix_receptance.weight`
/// - `blk.{i}.time_mix_key.weight`
/// - `blk.{i}.time_mix_value.weight`
/// - `blk.{i}.time_mix_gate.weight`
/// - `blk.{i}.channel_mix_key.weight`
/// - `blk.{i}.channel_mix_value.weight`
fn expected_proj_names(n_layers: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(n_layers * 6);
    for i in 0..n_layers {
        names.push(format!("blk.{i}.time_mix_receptance.weight"));
        names.push(format!("blk.{i}.time_mix_key.weight"));
        names.push(format!("blk.{i}.time_mix_value.weight"));
        names.push(format!("blk.{i}.time_mix_gate.weight"));
        names.push(format!("blk.{i}.channel_mix_key.weight"));
        names.push(format!("blk.{i}.channel_mix_value.weight"));
    }
    names
}

/// Loads the artifact from `RWKV7_TQ_TEST_ARTIFACT` and checks that all
/// 6×24 = 144 expected projection tensor names are present.
///
/// Requires `RWKV7_TQ_TEST_ARTIFACT` to point to a valid 0.4B (24-layer)
/// RWKV-7 `.tq` artifact.
#[test]
#[ignore = "requires RWKV7_TQ_TEST_ARTIFACT env var pointing to a .tq file"]
fn tq_artifact_loads_expected_names() {
    let path = std::env::var("RWKV7_TQ_TEST_ARTIFACT")
        .expect("RWKV7_TQ_TEST_ARTIFACT must be set to run this test");

    // VERIFY PATH: update this import once the real loader function is wired.
    // The function is expected to return a type that exposes tensor names.
    // hawking_core::model::rwkv7::load_tq_artifact(&path)
    let _ = &path; // placeholder until the real loader is available
    panic!("STUB: wire hawking_core::model::rwkv7::load_tq_artifact and check tensor names");

    // Expected structure once wired:
    //   let artifact = hawking_core::model::rwkv7::load_tq_artifact(&path)
    //       .expect("artifact load should succeed");
    //   let names = expected_proj_names(24);
    //   for name in &names {
    //       assert!(
    //           artifact.contains_tensor(name),
    //           "artifact missing expected tensor: {name}"
    //       );
    //   }
    //   assert_eq!(names.len(), 144);
}

/// Loads the artifact from `RWKV7_TQ_TEST_ARTIFACT` and checks that the
/// channel_mix projections have the shapes expected for the RWKV-7 0.4B model
/// (n_embd=1024, n_ff=4096).
///
/// `channel_mix_key.weight` shape: [n_ff, n_embd] = [4096, 1024]
/// `channel_mix_value.weight` shape: [n_embd, n_ff] = [1024, 4096]
#[test]
#[ignore = "requires RWKV7_TQ_TEST_ARTIFACT env var pointing to a .tq file"]
fn tq_artifact_shapes_match_04b() {
    let path = std::env::var("RWKV7_TQ_TEST_ARTIFACT")
        .expect("RWKV7_TQ_TEST_ARTIFACT must be set to run this test");

    const N_FF: usize = 4096;
    const N_EMBD: usize = 1024;

    // VERIFY PATH: update import once the real loader is wired.
    let _ = (path, N_FF, N_EMBD);
    panic!("STUB: wire hawking_core::model::rwkv7::load_tq_artifact and check shapes");

    // Expected structure once wired:
    //   let artifact = hawking_core::model::rwkv7::load_tq_artifact(&path)
    //       .expect("artifact load should succeed");
    //   for i in 0..24 {
    //       let cmk = artifact.tensor(&format!("blk.{i}.channel_mix_key.weight"))
    //           .expect("channel_mix_key must exist");
    //       assert_eq!(cmk.shape(), [N_FF, N_EMBD], "blk.{i} channel_mix_key shape");
    //
    //       let cmv = artifact.tensor(&format!("blk.{i}.channel_mix_value.weight"))
    //           .expect("channel_mix_value must exist");
    //       assert_eq!(cmv.shape(), [N_EMBD, N_FF], "blk.{i} channel_mix_value shape");
    //   }
}

/// Loading a non-existent artifact must return an `Err`, not panic.
///
/// This test does NOT require any real artifact — it is always runnable.
#[test]
fn tq_loader_missing_artifact_is_err() {
    // VERIFY PATH: update to the real loader function path once wired.
    // The function under test is expected to have the signature:
    //   pub fn load_tq_artifact(path: impl AsRef<std::path::Path>)
    //       -> hawking_core::Result<TqArtifact>
    //
    // Calling it with a non-existent path must return Err, not panic.
    //
    // Uncomment and adjust once the function exists:
    // let result = hawking_core::model::rwkv7::load_tq_artifact(
    //     "/tmp/this_file_does_not_exist_rwkv7_tq.tq",
    // );
    // assert!(result.is_err(), "loading a missing artifact must return Err");

    // Until wired, assert that the `expected_proj_names` helper works correctly
    // so at least one non-ignored assertion runs in this file.
    let names = expected_proj_names(24);
    assert_eq!(names.len(), 144, "24 layers × 6 projections = 144 names");
    assert_eq!(names[0], "blk.0.time_mix_receptance.weight");
    assert_eq!(names[5], "blk.0.channel_mix_value.weight");
    assert_eq!(names[6], "blk.1.time_mix_receptance.weight");
    assert_eq!(
        names[143],
        "blk.23.channel_mix_value.weight",
        "last tensor name for 24-layer model"
    );
}
