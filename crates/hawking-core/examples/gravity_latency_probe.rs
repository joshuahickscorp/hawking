//! Measure the native Gravity admission/identification lane.
//!
//! Usage:
//! `cargo run -p hawking-core --example gravity_latency_probe -- /path/to/model [repeats]`
//! The probe reports integer wall-clock nanoseconds for deterministic operations. It
//! does not admit, transform, or execute a model.

use hawking_core::{
    gravity_manifest::{GravityManifest, RepresentationPart},
    gravity_policy::{select, GravityPolicy},
    safetensors_inventory::SafetensorsInventory,
};
use serde_json::json;
use std::{env, process::ExitCode, time::Instant};

fn summary(mut samples: Vec<u128>) -> serde_json::Value {
    samples.sort();
    let percentile = |numerator: usize, denominator: usize| -> u128 {
        samples[(samples.len() - 1) * numerator / denominator]
    };
    json!({
        "samples": samples.len(),
        "min_ns": samples[0],
        "median_ns": percentile(1, 2),
        "p95_ns": percentile(95, 100),
        "max_ns": samples[samples.len() - 1],
    })
}

fn elapsed_ns(operation: impl FnOnce()) -> u128 {
    let started = Instant::now();
    operation();
    started.elapsed().as_nanos()
}

fn main() -> ExitCode {
    let mut arguments = env::args_os().skip(1);
    let Some(root) = arguments.next() else {
        eprintln!("usage: gravity_latency_probe <model-directory> [repeats]");
        return ExitCode::from(2);
    };
    let repeats = arguments
        .next()
        .and_then(|value| value.into_string().ok())
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=100).contains(value))
        .unwrap_or(5);
    let root_display = root.to_string_lossy().into_owned();

    let inventory = match SafetensorsInventory::scan_dir(&root) {
        Ok(inventory) => inventory,
        Err(error) => {
            eprintln!("gravity latency probe refused source: {error}");
            return ExitCode::from(1);
        }
    };
    let names = inventory
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<Vec<_>>();
    let range_tensor = inventory
        .tensors
        .iter()
        .find(|tensor| tensor.payload_bytes >= 64)
        .expect("a valid safetensors inventory has a non-empty tensor")
        .name
        .clone();

    let scan_ns = (0..repeats)
        .map(|_| {
            let started = Instant::now();
            let rebuilt =
                SafetensorsInventory::scan_dir(&root).expect("source changed during probe");
            assert_eq!(rebuilt.tensor_count, inventory.tensor_count);
            started.elapsed().as_nanos()
        })
        .collect::<Vec<_>>();
    let lookup_ns = (0..repeats)
        .map(|_| {
            elapsed_ns(|| {
                for _ in 0..1000 {
                    for name in &names {
                        assert!(inventory.tensor(name).is_some());
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    let range_read_ns = (0..repeats)
        .map(|_| {
            elapsed_ns(|| {
                assert_eq!(
                    inventory
                        .read_tensor_range(&range_tensor, 0, 64)
                        .unwrap()
                        .len(),
                    64
                );
            })
        })
        .collect::<Vec<_>>();
    let parts = [
        RepresentationPart {
            label: "codes",
            persistent_bytes: 12,
            active_bytes_per_token: 4,
        },
        RepresentationPart {
            label: "metadata",
            persistent_bytes: 4,
            active_bytes_per_token: 1,
        },
    ];
    let manifest = GravityManifest {
        id: "probe",
        logical_weights: 128,
        parts: &parts,
        complete_accounting_attested: true,
        direct_execution: true,
        source_independent: true,
    };
    let policy_ns = (0..repeats)
        .map(|_| {
            elapsed_ns(|| {
                let candidate = manifest.candidate(1, true).unwrap();
                for _ in 0..100_000 {
                    assert!(select([candidate], GravityPolicy::default()).is_some());
                }
            })
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "hawking.gravity.native_latency_probe.v1",
            "scope": "admission_and_identification_only; not_model_execution",
            "root": root_display,
            "shards": inventory.shards.len(),
            "tensors": inventory.tensor_count,
            "repeats": repeats,
            "operations": {
                "header_inventory_scan": summary(scan_ns),
                "tensor_lookup_1000x_all_tensors": summary(lookup_ns),
                "bounded_64_byte_tensor_read": summary(range_read_ns),
                "manifest_plus_policy_select_100k": summary(policy_ns),
            }
        }))
        .expect("serializing primitive latency data cannot fail")
    );
    ExitCode::SUCCESS
}
