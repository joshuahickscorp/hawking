//! AMX-routed f32 GEMV via Accelerate.framework's `cblas_sgemv`.
//!
//! Path-to-90 lever 5 (commit `d1d50fb`) introduced this for the Eagle4
//! head's 6 largest gemvs. Path-to-125 L4 extends the same pattern to
//! V2-Lite attention projections (`q_a_proj`, `kv_a_proj_with_mqa`,
//! `q_b_proj`, `kv_b_proj`) whose row/col shape fits AMX's sweet spot.
//!
//! `cblas_sgemv` routes through Apple's AMX matrix coprocessor — ~1790
//! GFLOPS at batch=1 for the row-major no-transpose shape used by these
//! projections. NOT Core ML (which has 8× dispatch overhead per the
//! deep-research Apple Silicon section); direct cblas via the framework
//! link.

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemv(
        order: i32,            // CblasRowMajor = 101
        trans: i32,            // CblasNoTrans  = 111
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
}

#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(target_os = "macos")]
const CBLAS_NO_TRANS: i32 = 111;

/// AMX-routed (rows × cols) f32 matrix-vector multiply: y = A·x.
/// Row-major; weight rows are contiguous, dot products per row.
#[cfg(target_os = "macos")]
#[inline]
pub fn amx_sgemv(rows: usize, cols: usize, a: &[f32], x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(a.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(y.len(), rows);
    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            rows as i32,
            cols as i32,
            1.0,
            a.as_ptr(),
            cols as i32,
            x.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }
}

/// Non-macOS fallback: CPU reference gemv. Same signature so callers can
/// be cfg-agnostic. (Not expected to be invoked on the production path,
/// but keeps `cargo check --target=x86_64-unknown-linux-gnu` honest.)
#[cfg(not(target_os = "macos"))]
#[inline]
pub fn amx_sgemv(rows: usize, cols: usize, a: &[f32], x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(a.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(y.len(), rows);
    for r in 0..rows {
        let mut acc = 0.0f32;
        let row = &a[r * cols..(r + 1) * cols];
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        y[r] = acc;
    }
}
