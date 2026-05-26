//! Smallest possible test of the Metal 3 `gpuAddress` + `useResource:`
//! pattern that the 2-layer megakernel dispatcher needs to scale up.
//!
//! Allocates two `f32` buffers, packs their `Buffer::gpu_address()` into
//! a `constant`-bound argbuf, calls `use_resource` on both, dispatches a
//! one-line kernel that does `out[i] = in[i] * 2.0`, and asserts the
//! arithmetic round-tripped.
//!
//! If this test fails — page fault on the gpu_address deref, garbage
//! readback, driver refusal of the bind — day-3 dispatch-harness
//! design must be revised before scaling to ~24 weight buffers/layer.
//! See `memory/build_megakernel_day2_2026_05_25.md` § "Day-3 entry
//! points" item 2.

#![cfg(target_os = "macos")]

use dismantle_core::metal::MetalContext;
use metal::MTLResourceUsage;

const N: usize = 4096;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct GpuAddrProbeArgs {
    in_ptr: u64,
    out_ptr: u64,
    n: u32,
    _pad: u32,
}

#[test]
fn gpu_address_probe_roundtrip() {
    let ctx = MetalContext::new().expect("MetalContext::new");

    // Input: [1.0, 2.0, ..., N]; output: zeroed.
    let input: Vec<f32> = (0..N).map(|i| (i + 1) as f32).collect();
    let zero: Vec<f32> = vec![0.0; N];

    let in_bytes = bytemuck::cast_slice::<f32, u8>(&input);
    let out_bytes = bytemuck::cast_slice::<f32, u8>(&zero);
    let in_buf = ctx.new_buffer_with_bytes(in_bytes);
    let out_buf = ctx.new_buffer_with_bytes(out_bytes);

    // gpu_address is the Metal-3 device virtual address. Stuffing it
    // into the argbuf as u64 makes Metal interpret the matching
    // `device const float*` field as a raw pointer.
    let args = GpuAddrProbeArgs {
        in_ptr: in_buf.gpu_address(),
        out_ptr: out_buf.gpu_address(),
        n: N as u32,
        _pad: 0,
    };
    let arg_bytes: [u8; std::mem::size_of::<GpuAddrProbeArgs>()] =
        unsafe { std::mem::transmute(args) };
    let arg_buf = ctx.new_buffer_with_bytes(&arg_bytes);

    let n = N as u32;
    ctx.dispatch_threads("gpu_address_probe", (n, 1, 1), (64, 1, 1), |enc| {
        enc.set_buffer(0, Some(&arg_buf), 0);
        // Driver must know in_buf/out_buf are live across the dispatch,
        // since the encoder only sees `arg_buf` directly.
        enc.use_resource(&in_buf, MTLResourceUsage::Read);
        enc.use_resource(&out_buf, MTLResourceUsage::Write);
    })
    .expect("dispatch_threads gpu_address_probe");

    // Readback.
    let out_ptr = out_buf.contents() as *const f32;
    let out: &[f32] = unsafe { std::slice::from_raw_parts(out_ptr, N) };
    for (i, (got, src)) in out.iter().zip(input.iter()).enumerate() {
        let want = src * 2.0;
        assert_eq!(
            *got, want,
            "gpu_address probe mismatch at i={i}: got {got}, want {want} (src={src})"
        );
    }
}
