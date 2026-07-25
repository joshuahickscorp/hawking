//! Container-level integration test for `.gravity` shards: open a real
//! production artifact, read a `gravity-pq` tensor with hash verification,
//! run both matvec paths on it, and check the container's byte accounting.
//!
//! Exercises
//! `~/Library/Application Support/Hawking/CampaignS08/llama32-1b-R0.gravity`
//! (146 tensors, 113 `gravity-pq` + 33 `native.bf16`, complete_bpw
//! 0.877372) which lives outside the repo -- skips gracefully (prints why,
//! returns early) if that file isn't present on this machine.

#![cfg(target_os = "macos")]

use std::io::Read;
use std::path::PathBuf;

use hawking_core::gravity::{parse_pq_header, pq_matvec, pq_matvec_metal, GravityShard};
use hawking_core::metal::MetalContext;

const FIXTURE_PATH: &str =
    "/Users/scammermike/Library/Application Support/Hawking/CampaignS08/llama32-1b-R0.gravity";
const Q_PROJ_L0: &str = "model.layers.0.self_attn.q_proj.weight";

#[test]
fn gravity_container_real_artifact() {
    let path = PathBuf::from(FIXTURE_PATH);
    if !path.exists() {
        println!("skipping: fixture not present at {FIXTURE_PATH:?} (outside the repo)");
        return;
    }

    // 1. Open + header sanity: every billed tensor must actually be stored
    //    (a tensor billed but not stored is the exact defect this format
    //    exists to prevent).
    let shard = GravityShard::open(&path).expect("GravityShard::open");
    let names: Vec<&str> = shard.tensor_names().collect();
    assert_eq!(names.len(), 146, "tensor count");

    let mut total_bytes: u64 = 0;
    for name in &names {
        let d = shard
            .descriptor(name)
            .unwrap_or_else(|| panic!("descriptor for {name}"));
        assert!(d.bytes > 0, "tensor {name} billed with zero bytes");
        total_bytes += d.bytes;
    }

    // 2. Read one gravity-pq tensor WITH hash verification, check its header.
    let payload = shard
        .read_tensor(Q_PROJ_L0, true)
        .expect("read_tensor with hash verification");
    let h = parse_pq_header(&payload).expect("parse_pq_header");
    assert_eq!(h.rows, 2048, "rows");
    assert_eq!(h.cols, 2048, "cols");
    assert_eq!(h.d, 8, "d");
    assert_eq!(h.s, 1, "s");

    // 3. CPU matvec always; GPU matvec too when a Metal device is available.
    let x: Vec<f32> = (0..h.cols as usize)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();
    let y_cpu = pq_matvec(&payload, &x).expect("pq_matvec");
    println!("[{Q_PROJ_L0}] CPU first 3 outputs: {:?}", &y_cpu[..3]);

    match MetalContext::new() {
        Ok(ctx) => {
            let y_gpu = pq_matvec_metal(&ctx, &payload, &x).expect("pq_matvec_metal");
            println!("[{Q_PROJ_L0}] GPU first 3 outputs: {:?}", &y_gpu[..3]);
            assert_eq!(y_gpu.len(), y_cpu.len(), "GPU/CPU output length");

            let mut worst = (0usize, f32::NEG_INFINITY, 0f32, 0f32);
            for (i, (&a, &b)) in y_gpu.iter().zip(y_cpu.iter()).enumerate() {
                let diff = (a - b).abs();
                let tol = 1e-3 + 1e-3 * b.abs();
                let margin = diff - tol;
                if margin > worst.1 {
                    worst = (i, margin, diff, tol);
                }
            }
            let (idx, _, diff, tol) = worst;
            println!(
                "[{Q_PROJ_L0}] GPU vs CPU worst element {idx}: |diff|={diff:.3e} tol={tol:.3e}"
            );
            assert!(
                diff <= tol,
                "GPU vs CPU: element {idx}: got {}, want {}, |diff|={diff} > tol={tol}",
                y_gpu[idx],
                y_cpu[idx]
            );
        }
        Err(e) => {
            println!("skipping GPU comparison: no Metal device available: {e}");
        }
    }

    // 4. Byte accounting: header + all tensor bytes == file length exactly
    //    (no gaps, no overlap, no unaccounted bytes). Header size is
    //    re-derived from the raw prefix (magic 8B + version 4B + header_len
    //    8B, per the documented container layout) rather than trusting a
    //    second copy of the same arithmetic `GravityShard::open` already did.
    let mut prefix = [0u8; 20];
    std::fs::File::open(&path)
        .and_then(|mut f| f.read_exact(&mut prefix))
        .expect("read container prefix");
    let header_len = u64::from_le_bytes(prefix[12..20].try_into().unwrap());
    let header_size = 20u64 + header_len;
    let file_len = std::fs::metadata(&path).expect("metadata").len();
    assert_eq!(
        header_size + total_bytes,
        file_len,
        "header_size {header_size} + total tensor bytes {total_bytes} != file length {file_len}"
    );
}
