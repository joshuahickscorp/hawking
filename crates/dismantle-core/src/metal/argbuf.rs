//! Argument-buffer abstraction for kernel scalar args.
//!
//! `KernelArgBuffer` packs all scalar arguments for one kernel invocation into
//! a sub-region of a persistent per-context bump arena. The host writes
//! values via `set_u32` / `set_f32` / `set_u64`; the encoder binds the
//! buffer at the first scalar arg index via `argbuf.bind(enc, slot)` —
//! the bind correctly applies the arena's base offset.
//!
//! The GPU shader must declare a packed constant struct at the corresponding
//! buffer index, e.g.:
//!
//! ```metal
//! struct ArgbufRowsCols { uint rows; uint cols; };
//! kernel void my_kernel(..., constant ArgbufRowsCols& args [[buffer(3)]], ...) { ... }
//! ```
//!
//! v2.3.0 A5: backed by a per-`MetalContext` bump arena
//! (`MetalContext::argbuf_alloc`) that resets at every
//! `TokenCommandBuffer::commit_and_wait`. Allocation is constant-time
//! after warmup; no per-dispatch `new_buffer` call. The Stage-0 path-to-90
//! attribution showed per-dispatch CPU cost ~117 µs against an expected
//! ~5-15 µs for Metal encoding — `new_buffer` (~50 µs each at ~80
//! argbuf-using calls/token = ~4 ms/tok) was the largest fixable
//! component. Reusing one shared MTLBuffer eliminates that cost.

/// Layout descriptor for one scalar field in a `KernelArgBuffer`.
#[derive(Clone, Copy, Debug)]
pub enum ArgLayout {
    U32,
    F32,
    U64,
}

#[cfg(target_os = "macos")]
mod imp {
    use super::ArgLayout;
    use crate::metal::MetalContext;
    use crate::Result;
    use metal::{Buffer, ComputeCommandEncoderRef};

    /// Sub-region of the per-context argbuf arena holding all scalar args
    /// for one kernel invocation.
    pub struct KernelArgBuffer {
        /// Clone of the arena's MTLBuffer. `Buffer` is reference-counted
        /// internally (Arc-equivalent), so cloning is constant time.
        buf: Buffer,
        /// Byte offset of this argbuf's region within `buf`.
        base_offset: usize,
        /// Field byte-offsets relative to `base_offset`.
        offsets: Vec<usize>,
    }

    impl KernelArgBuffer {
        /// Carve a region from the persistent argbuf arena, sized and laid
        /// out for the given field sequence. Fields are packed at their
        /// natural alignment (4 bytes for U32/F32, 8 bytes for U64) with
        /// no inter-field padding on Apple Silicon.
        ///
        /// v2.3.0 A5: no `new_buffer` call on the hot path — backed by
        /// `MetalContext::argbuf_alloc` which carves from a single
        /// pre-allocated MTLBuffer that resets per token.
        pub fn new(ctx: &MetalContext, layout: &[ArgLayout]) -> Result<Self> {
            let mut offsets = Vec::with_capacity(layout.len());
            let mut cur = 0usize;
            // Region alignment = strictest field alignment. Apple's MSL
            // requires constant-buffer offsets to be 4-byte aligned at
            // minimum (and 8 if the first field is U64). The arena's
            // alloc handles the base-offset alignment for us.
            let mut region_align = 4usize;
            for a in layout {
                let align = match a {
                    ArgLayout::U32 | ArgLayout::F32 => 4,
                    ArgLayout::U64 => 8,
                };
                if align > region_align {
                    region_align = align;
                }
                if cur % align != 0 {
                    cur += align - (cur % align);
                }
                offsets.push(cur);
                cur += align;
            }
            let size = cur.max(4);
            let (buf, base_offset) = ctx.argbuf_alloc(size, region_align);
            Ok(Self {
                buf,
                base_offset,
                offsets,
            })
        }

        /// Write a `u32` to field at position `idx` in the layout.
        ///
        /// # Panics
        /// Panics if `idx` is out of range.
        pub fn set_u32(&mut self, idx: usize, value: u32) {
            let off = self.base_offset + self.offsets[idx];
            let ptr = self.buf.contents() as *mut u8;
            unsafe { ptr.add(off).cast::<u32>().write(value) };
        }

        /// Write a `f32` to field at position `idx` in the layout.
        pub fn set_f32(&mut self, idx: usize, value: f32) {
            let off = self.base_offset + self.offsets[idx];
            let ptr = self.buf.contents() as *mut u8;
            unsafe { ptr.add(off).cast::<f32>().write(value) };
        }

        /// Write a `u64` to field at position `idx` in the layout.
        pub fn set_u64(&mut self, idx: usize, value: u64) {
            let off = self.base_offset + self.offsets[idx];
            let ptr = self.buf.contents() as *mut u8;
            unsafe { ptr.add(off).cast::<u64>().write(value) };
        }

        /// Bind this argbuf to a compute-encoder buffer slot. Equivalent
        /// to `enc.set_buffer(slot, Some(arena_buf), base_offset)`.
        pub fn bind(&self, enc: &ComputeCommandEncoderRef, slot: u64) {
            enc.set_buffer(slot, Some(&self.buf), self.base_offset as u64);
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::ArgLayout;
    use crate::metal::MetalContext;
    use crate::Result;
    use std::sync::Arc;

    /// Non-macOS stub. Never constructed; exists so the type resolves in
    /// cross-platform code.
    pub struct KernelArgBuffer {
        _priv: Arc<()>,
    }

    impl KernelArgBuffer {
        pub fn new(_ctx: &MetalContext, _layout: &[ArgLayout]) -> Result<Self> {
            Err(crate::Error::Metal(
                "KernelArgBuffer: Metal unavailable on this platform".into(),
            ))
        }
        pub fn set_u32(&mut self, _idx: usize, _value: u32) {}
        pub fn set_f32(&mut self, _idx: usize, _value: f32) {}
        pub fn set_u64(&mut self, _idx: usize, _value: u64) {}
        pub fn bind(&self, _enc: &(), _slot: u64) {}
    }
}

pub use imp::KernelArgBuffer;
