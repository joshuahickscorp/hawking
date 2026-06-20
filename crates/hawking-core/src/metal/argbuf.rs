//! Argument-buffer abstraction for ICB-compatible kernel scalar args.
//!
//! `KernelArgBuffer` packs all scalar arguments for one kernel invocation into
//! a single pre-allocated `MTLBuffer`.  The host writes values via `set_u32` /
//! `set_f32` / `set_u64`; the encoder binds the whole buffer at the first
//! scalar arg index via one `set_buffer` call instead of N `set_bytes` calls.
//!
//! The GPU shader must declare a packed constant struct at the corresponding
//! buffer index, e.g.:
//!
//! ```metal
//! struct ArgbufRowsCols { uint rows; uint cols; };
//! kernel void my_kernel(..., constant ArgbufRowsCols& args [[buffer(3)]], ...) { ... }
//! ```
//!
//! Phase 3 note: buffers are allocated per `new()` call (one-per-dispatch is
//! acceptable for correctness; Phase 5/ICB will pre-allocate persistent instances
//! at engine load time and only update per-token-changing fields).

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
    use metal::{Buffer, MTLResourceOptions};

    /// Pre-allocated MTLBuffer holding all scalar args for one kernel invocation.
    pub struct KernelArgBuffer {
        buf: Buffer,
        offsets: Vec<usize>,
    }

    impl KernelArgBuffer {
        /// Allocate a buffer sized and laid out for the given field sequence.
        /// Fields are packed at their natural alignment (4 bytes for U32/F32,
        /// 8 bytes for U64) with no inter-field padding on Apple Silicon.
        pub fn new(ctx: &MetalContext, layout: &[ArgLayout]) -> Result<Self> {
            let mut offsets = Vec::with_capacity(layout.len());
            let mut cur = 0usize;
            for a in layout {
                let align = match a {
                    ArgLayout::U32 | ArgLayout::F32 => 4,
                    ArgLayout::U64 => 8,
                };
                // align cur to field boundary
                if cur % align != 0 {
                    cur += align - (cur % align);
                }
                offsets.push(cur);
                cur += align;
            }
            let size = cur.max(4); // at least 4 bytes so handle() is always valid
            let buf = ctx
                .device()
                .new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
            Ok(Self { buf, offsets })
        }

        /// Write a `u32` to field at position `idx` in the layout.
        ///
        /// # Panics
        /// Panics if `idx` is out of range.
        pub fn set_u32(&mut self, idx: usize, value: u32) {
            let off = self.offsets[idx];
            let ptr = self.buf.contents() as *mut u8;
            unsafe { ptr.add(off).cast::<u32>().write(value) };
        }

        /// Write a `f32` to field at position `idx` in the layout.
        pub fn set_f32(&mut self, idx: usize, value: f32) {
            let off = self.offsets[idx];
            let ptr = self.buf.contents() as *mut u8;
            unsafe { ptr.add(off).cast::<f32>().write(value) };
        }

        /// Write a `u64` to field at position `idx` in the layout.
        pub fn set_u64(&mut self, idx: usize, value: u64) {
            let off = self.offsets[idx];
            let ptr = self.buf.contents() as *mut u8;
            unsafe { ptr.add(off).cast::<u64>().write(value) };
        }

        /// The underlying buffer for `enc.set_buffer(slot, Some(argbuf.handle()), 0)`.
        pub fn handle(&self) -> &Buffer {
            &self.buf
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
    }
}

pub use imp::KernelArgBuffer;
