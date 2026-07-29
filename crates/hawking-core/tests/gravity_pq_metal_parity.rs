#![cfg(target_os = "macos")]
use hawking_core::gravity::{pq_matvec, pq_matvec_metal};
use hawking_core::metal::MetalContext;
use serde::Deserialize;
use std::path::{Path, PathBuf};
#[derive(Debug, Deserialize)]
struct ManifestFixture {
    fixture: String,
    rows: u64,
    cols: u64,
    blob_bytes: u64,
}
#[derive(Debug, Deserialize)]
struct Manifest {
    fixtures: Vec<ManifestFixture>,
}
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gravity_pq")
}
fn read_f32_le(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "{path:?}: byte length {} not a multiple of 4", bytes.len());
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn worst_margin(got: &[f32], want: &[f32]) -> (usize, f32, f32, f32) {
    let mut worst = (0usize, f32::NEG_INFINITY, 0f32, 0f32);
    for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (a - b).abs();
        let tol = 1e-3 + 1e-3 * b.abs();
        let margin = diff - tol;
        if margin > worst.1 {
            worst = (i, margin, diff, tol);
        }
    }
    worst
}
#[test]
fn pq_matvec_metal_matches_cpu_and_reference_for_all_fixtures() {
    let ctx = match MetalContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("skipping: no Metal device available: {e}");
            return;
        }
    };
    let dir = fixtures_dir();
    let manifest: Manifest = {
        let text = std::fs::read_to_string(dir.join("manifest.json")).expect("read manifest");
        serde_json::from_str(&text).expect("parse manifest")
    };
    assert_eq!(manifest.fixtures.len(), 3, "expected R0/R2/R4 fixtures");
    for row in &manifest.fixtures {
        let blob_path = dir.join(format!("{}.bin", row.fixture));
        let payload = std::fs::read(&blob_path).unwrap_or_else(|e| panic!("read {blob_path:?}: {e}"));
        assert_eq!(
            payload.len() as u64,
            row.blob_bytes,
            "{}: file length {} != manifest blob_bytes {}",
            row.fixture,
            payload.len(),
            row.blob_bytes
        );
        let x = read_f32_le(&dir.join(format!("{}.x.f32", row.fixture)));
        let y_ref = read_f32_le(&dir.join(format!("{}.y.f32", row.fixture)));
        assert_eq!(x.len() as u64, row.cols, "{}: x length", row.fixture);
        assert_eq!(y_ref.len() as u64, row.rows, "{}: y length", row.fixture);
        let y_cpu = pq_matvec(&payload, &x).unwrap_or_else(|e| panic!("{}: pq_matvec: {e}", row.fixture));
        let y_gpu = pq_matvec_metal(&ctx, &payload, &x).unwrap_or_else(|e| panic!("{}: pq_matvec_metal: {e}", row.fixture));
        assert_eq!(y_gpu.len(), row.rows as usize, "{}: GPU output length", row.fixture);
        let (idx_ref, margin_ref, diff_ref, tol_ref) = worst_margin(&y_gpu, &y_ref);
        assert!(
            diff_ref <= tol_ref,
            "{}: GPU vs fixture: element {idx_ref}: got {}, want {}, |diff|={diff_ref} > tol={tol_ref}",
            row.fixture,
            y_gpu[idx_ref],
            y_ref[idx_ref]
        );
        let (idx_cpu, margin_cpu, diff_cpu, tol_cpu) = worst_margin(&y_gpu, &y_cpu);
        assert!(
            diff_cpu <= tol_cpu,
            "{}: GPU vs CPU: element {idx_cpu}: got {}, want {}, |diff|={diff_cpu} > tol={tol_cpu}",
            row.fixture,
            y_gpu[idx_cpu],
            y_cpu[idx_cpu]
        );
    }
}
