#![cfg(all(feature = "tq", target_os = "macos"))]

use hawking_core::metal::MetalContext;
use hawking_core::tq::read_strand;
use hawking_core::{TqDeviceHarness, TqRuntimePath};

/// Manual real-artifact probe. `TQ_LLAMA_PROBE_PATH` must name a one-tensor
/// archive made by `tq_bake`; it never silently substitutes synthetic weights.
#[test]
#[ignore = "requires an explicitly baked local TQ probe artifact"]
fn real_llama_ffn_gate_tq_bitslice_roofline() {
    let path = std::env::var("TQ_LLAMA_PROBE_PATH").expect("set TQ_LLAMA_PROBE_PATH");
    let bytes = std::fs::read(path).expect("read TQ artifact");
    let tensors = read_strand(&bytes).expect("parse TQ artifact");
    assert_eq!(tensors.len(), 1, "this probe must isolate exactly one projection");
    let tensor = &tensors[0];
    assert_eq!((tensor.out_features, tensor.in_features), (14_336, 4_096));
    let activation: Vec<f32> = (0..tensor.in_features)
        .map(|i| (i as f32 * 0.017).sin())
        .collect();
    let ctx = MetalContext::new().expect("Metal device");
    let harness = TqDeviceHarness::prepare(&ctx, tensor, TqRuntimePath::Stored, &activation)
        .expect("admitted resident TQ harness");
    let cpu = tensor.matvec(&activation);
    let (gpu, dispatches) = harness.run_gemv(&ctx).expect("resident GEMV");
    assert_eq!(dispatches, 2, "partials + row reduction are the production TQ graph");
    for (row, (actual, expected)) in gpu.iter().zip(cpu).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-3,
            "row {row}: TQ GPU {actual} != compact CPU {expected}"
        );
    }
    for _ in 0..4 { harness.run_gemv(&ctx).unwrap(); }
    let mut samples = Vec::new();
    for _ in 0..12 {
        let started = std::time::Instant::now();
        harness.run_gemv(&ctx).unwrap();
        samples.push(started.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    eprintln!(
        "TQ real Llama FFN gate k={} L={} blocks={} dispatches={} resident median_us={median:.3}",
        harness.k_bits, harness.l_bits, harness.blocks, dispatches
    );
}
