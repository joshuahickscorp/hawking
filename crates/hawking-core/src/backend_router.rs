//! Per-op compute router with CPU fallback (Phase 3.2 op scheduler).
//!
//! macOS-only (it names the concrete Metal backend). Builds on the 3.1
//! seam ([`super::Backend`] + the concrete `MetalBackend` /
//! `MetalRecorder` in [`super::metal`]). The router IS the scheduler:
//! there is no IR/graph (hawking decode is imperative), so routing is a
//! single `supports()` check per op. hawking has no `ggml_backend_sched`
//! graph splitter and does not want one.
//!
//! # What it does
//!
//! For each fallback-capable op the router does exactly:
//!
//! ```text
//! if self.supports(op) { primary.<op>(rec, ..) }      // pure Metal
//! else { flush rec's in-flight TCB (commit_and_wait) -> run the CPU
//!        primitive (kernels::rmsnorm / rope_inplace) on the SHARED
//!        buffer in place -> continue on a FRESH TCB }                 }
//! ```
//!
//! gated by the string env [`FORCE_CPU_OP_ENV`] (`HAWKING_FORCE_CPU_OP`,
//! values `rmsnorm` | `rope`, DEFAULT-UNSET). When unset, `forced` is
//! `None`, every `supports()` returns `true`, the CPU branch is never
//! reached, and the router monomorphizes (concrete `MetalBackend`, no
//! `dyn`, `#[inline]` Metal arms) to the *identical* `kernels::*_tcb` call
//! the pre-router decode made — so the golden greedy-64 hash is unchanged
//! and the single-command-buffer-per-token invariant is intact.
//!
//! # Why the split command buffer is unavoidable here
//!
//! The op-trait methods receive `&mut Self::Recorder<'_>` (not ownership),
//! but [`super::CommandRecorder::commit_and_wait`] consumes the recorder /
//! its `TokenCommandBuffer` *by value* (`metal/mod.rs:1187`). To flush the
//! work queued so far (so the CPU reads a current buffer) we therefore
//! take the live `TokenCommandBuffer` *out* of `rec.tcb` with
//! [`std::mem::replace`], commit-and-wait that owned CB, run the CPU op on
//! the now-current shared buffer in place (unified memory — no blit), then
//! install a fresh `TokenCommandBuffer::new(ctx)` back into `rec.tcb` and
//! return so the caller keeps recording. The caller's `&mut MetalRecorder`
//! ends the call holding a fresh empty TCB. This split-CB is the one real
//! cost of a forced fallback (it shatters single-CB batching for that
//! token) and is acceptable precisely because fallback is the slow path —
//! see the stream `risks`. It is NEVER reached on the unset (Metal) path.
//!
//! # The lifetime contract on the flush (Wave-3 CRITICAL fix)
//!
//! `MetalRecorder<'a>` wraps `TokenCommandBuffer<'a>`, which holds a
//! `&'a MetalContext` (its `pub ctx` field, `metal/mod.rs:859`). The fresh
//! replacement TCB installed on a fallback MUST have that SAME lifetime
//! `'a`, or `std::mem::replace` cannot type-check: `TokenCommandBuffer` is
//! *covariant* in its context lifetime, so a TCB minted from a shorter or
//! unrelated borrow cannot substitute for `rec.tcb: TokenCommandBuffer<'a>`
//! (the substitution would require that other lifetime to outlive `'a`,
//! which the borrow checker cannot prove). In particular the router's OWN
//! `MetalContext` (an `Arc`-clone built by the decode hook via
//! `ctx.clone()`) is borrowed only for the transient `&self`, and is a
//! *different instance* from the one the recorder's TCB borrows — minting
//! the fresh TCB from `self.context()` is exactly the covariance error.
//!
//! The fix: [`Router::flush_and_reset`] takes an explicit
//! `ctx: &'a MetalContext` tied to the recorder's own lifetime, and the
//! callers pass the recorder's own borrowed context (`rec.tcb.ctx`, copied
//! into a local before the `&mut rec.tcb` borrow). The minted
//! `TokenCommandBuffer::new(ctx)` is then `TokenCommandBuffer<'a>` — the
//! exact type the slot expects. The `Arc`-backed contexts share the same
//! device/queue, so this is functionally identical to minting from the
//! router's clone; only the *type lifetime* differs, and that is what has
//! to line up.
//!
//! # Scope (held deliberately narrow per the 3.1 rule + scout)
//!
//! Only `rmsnorm` and `rope` are fallback-capable in 3.2. The GEMV moat
//! (86.7% GPU) and attention stay Metal-only — a CPU gemv is 3.3's job and
//! would tank decode if force-routed. The unfused `rmsnorm` site is the
//! flagship; `rope` proves the seam generalizes to a second, in-place op.
//! The fused `add_rmsnorm` site is intentionally NOT routed here: its CPU
//! path must also fold the residual add, which is a follow-up.

#![cfg(target_os = "macos")]

use crate::kernels;
use crate::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use crate::Result;

use super::metal::{MetalBackend, MetalRecorder};
use super::{Backend, BackendNorm, BackendRope, Op, RopeLayout};

/// String env lever selecting an op to force onto the CPU fallback.
/// DEFAULT-UNSET. Recognized values: `"rmsnorm"`, `"rope"`. Any other
/// value (or unset) forces nothing ⇒ pure Metal ⇒ golden hash unchanged.
pub const FORCE_CPU_OP_ENV: &str = "HAWKING_FORCE_CPU_OP";

/// Per-op compute router over a primary [`MetalBackend`] with a CPU
/// fallback for the ops the primary is forced to "not support".
///
/// `forced` is resolved ONCE (at construction, from [`FORCE_CPU_OP_ENV`]).
/// `supports(op)` returns `false` only for the single `forced` op, so the
/// fast path is bit-identical to the bare `MetalBackend` whenever the env
/// is unset.
pub struct Router {
    /// The complete Metal backend. Owned by value (its `MetalContext` is
    /// `Arc`-backed, so this is cheap and shares device/queue/pipeline
    /// cache with every other holder).
    primary: MetalBackend,
    /// The op (if any) forced onto the CPU fallback. `None` ⇒ everything
    /// runs on `primary` ⇒ the Metal path is untouched.
    forced: Option<Op>,
}

impl Router {
    /// Build a router over `primary`, reading [`FORCE_CPU_OP_ENV`] once to
    /// decide which op (if any) is forced onto the CPU fallback.
    #[inline]
    pub fn from_env(primary: MetalBackend) -> Self {
        let forced = parse_force_cpu_op();
        Self { primary, forced }
    }

    /// Build a router with an explicit forced op (bypasses the env read).
    /// `None` ⇒ pure Metal. Used by tests / call sites that want to drive
    /// the fallback deterministically.
    #[inline]
    pub fn with_forced(primary: MetalBackend, forced: Option<Op>) -> Self {
        Self { primary, forced }
    }

    /// Borrow the underlying [`MetalBackend`] (e.g. to drive Metal-only ops
    /// — gemv, attention, kv, embed, sample — that this router does not
    /// wrap because they stay Metal-only in 3.2).
    #[inline]
    pub fn primary(&self) -> &MetalBackend {
        &self.primary
    }

    /// The op currently forced onto the CPU fallback, if any.
    #[inline]
    pub fn forced_op(&self) -> Option<Op> {
        self.forced
    }

    /// Whether the *router* runs `op` on the primary (Metal) backend.
    /// `false` only for the single forced op. (Distinct from
    /// `MetalBackend::supports`, which is unconditionally `true`; the
    /// router overlays the forced-CPU policy on top.)
    #[inline]
    pub fn supports(&self, op: Op) -> bool {
        self.forced != Some(op)
    }

    /// Open a fresh per-token recorder on the primary backend. Identical to
    /// `MetalBackend::recorder()`; provided so call sites can obtain the
    /// recorder through the router.
    #[inline]
    pub fn recorder(&self) -> MetalRecorder<'_> {
        self.primary.recorder()
    }

    /// Flush the recorder's in-flight command buffer (commit + wait) so the
    /// shared buffers it wrote are current for a CPU read, then install a
    /// FRESH empty `TokenCommandBuffer` back into the recorder so the
    /// caller can keep recording. On return `rec.tcb` is a brand-new TCB
    /// and all previously queued GPU work has completed.
    ///
    /// This is the split-CB seam. It is only ever called on a forced
    /// fallback; the Metal path never touches it.
    ///
    /// # Lifetime (Wave-3 CRITICAL fix)
    ///
    /// `ctx` is taken as an explicit parameter bound to the recorder's own
    /// lifetime `'a` (the caller passes `rec.tcb.ctx`). The replacement
    /// `TokenCommandBuffer::new(ctx)` is therefore `TokenCommandBuffer<'a>`,
    /// the exact type of `rec.tcb`, so `std::mem::replace` type-checks. This
    /// is why the context is a PARAMETER and not `self.context()`: the
    /// router's owned (Arc-cloned) context is borrowed only for the
    /// transient `&self` and is a different instance, so a TCB minted from
    /// it has the wrong lifetime for the covariant `TokenCommandBuffer`
    /// (the borrow checker cannot prove the `&self` borrow outlives `'a`).
    fn flush_and_reset<'a>(&self, rec: &mut MetalRecorder<'a>, ctx: &'a MetalContext) -> Result<()> {
        // Take the live TCB out by value, swapping in a fresh CB on the
        // same queue. `ctx` has lifetime `'a` (the recorder's own context
        // borrow), so the new TCB is `TokenCommandBuffer<'a>` and the swap
        // is well-typed. Swapping first leaves `rec` in a valid state even
        // if the commit below errors. The old TCB is then committed-and-
        // waited (consumed by value — the only commit entry point).
        let live = std::mem::replace(&mut rec.tcb, TokenCommandBuffer::new(ctx));
        // `commit_and_wait` takes `mut self` and blocks until the GPU
        // finishes the queued work, so the shared buffer the CPU op reads
        // next is fully written.
        live.commit_and_wait()
    }

    // ── Fallback-capable ops ────────────────────────────────────────────

    /// RMS-norm: `out = rmsnorm(x, weight, eps)` over `hidden` elements.
    /// Metal fast path when not forced; otherwise flush → `kernels::rmsnorm`
    /// (f64-accumulated) on the shared buffers → fresh TCB.
    pub fn rmsnorm<'a>(&self, rec: &mut MetalRecorder<'a>, x: &PinnedBuffer, weight: &PinnedBuffer, out: &PinnedBuffer, eps: f32, hidden: usize) -> Result<()> {
        if self.supports(Op::RmsNorm) {
            // Bit-identical to the pre-router decode: this is the exact
            // `kernels::rmsnorm_metal_buf_tcb` dispatch, just behind the
            // trait. #[inline] on the impl method keeps it free.
            return self.primary.rmsnorm(rec, x, weight, out, eps, hidden);
        }
        // ── CPU fallback ───────────────────────────────────────────────
        // Copy the recorder's own context reference (a `&'a MetalContext`,
        // Copy) out FIRST so the shared borrow of `rec` ends before the
        // `&mut rec.tcb` taken inside flush_and_reset. This is what pins
        // the fresh TCB to lifetime `'a` (see flush_and_reset docs).
        let ctx: &'a MetalContext = rec.tcb.ctx;
        // Flush so `x` (written by the just-queued GPU ops) is current.
        self.flush_and_reset(rec, ctx)?;
        // SAFETY: x/weight/out are host-visible StorageModeShared buffers
        // (PinnedBuffer = ::metal::Buffer). After the flush above the GPU
        // is idle, so reading `x`/`weight` and writing `out` in place is a
        // plain pointer cast over unified memory — no blit. `hidden`
        // matches the element count the caller bound on every buffer.
        let x_ptr = x.contents() as *const f32;
        let w_ptr = weight.contents() as *const f32;
        let out_ptr = out.contents() as *mut f32;
        unsafe {
            let x_s = std::slice::from_raw_parts(x_ptr, hidden);
            let w_s = std::slice::from_raw_parts(w_ptr, hidden);
            let out_s = std::slice::from_raw_parts_mut(out_ptr, hidden);
            kernels::rmsnorm(x_s, w_s, eps, out_s);
        }
        // `rec.tcb` is already a fresh empty TCB (installed by
        // flush_and_reset); the caller keeps recording into it.
        Ok(())
    }

    /// RoPE in place on `buf` for position `pos`. Metal fast path when not
    /// forced; otherwise flush → CPU `rope_inplace` per head → fresh TCB.
    ///
    /// Only [`RopeLayout::Full`] is CPU-fallback-capable in 3.2 (the dense
    /// Qwen full-head case: `qk_nope_dim = 0`, rotate the whole head). The
    /// partial-slice (MLA) layout has no CPU fallback here and always runs
    /// on the primary — a forced `rope` simply leaves it on Metal.
    pub fn rope<'a>(&self, rec: &mut MetalRecorder<'a>, buf: &PinnedBuffer, layout: RopeLayout, pos: u32, base: f32) -> Result<()> {
        // Partial (MLA) layout: no CPU fallback in 3.2 — always Metal.
        let (n_heads, head_dim) = match layout {
            RopeLayout::Full { n_heads, head_dim } => (n_heads, head_dim),
            RopeLayout::Partial { .. } => return self.primary.rope(rec, buf, layout, pos, base),
        };
        if self.supports(Op::Rope) {
            return self.primary.rope(rec, buf, layout, pos, base);
        }
        // ── CPU fallback (full-head, interleaved pairs) ────────────────
        // GPU `rope_q_f32_inplace` (shaders/common.metal:471) rotates each
        // head's interleaved (2i, 2i+1) pairs with theta =
        // pos / base^(2*pair/head_dim); CPU `rope_inplace` (kernels/mod.rs:92)
        // does the SAME pairing/schedule on one head_dim slice. So the
        // fallback is `n_heads` calls to `rope_inplace`, one per contiguous
        // head slice. (For an unscaled rope this equals
        // `rope_inplace_scaled(.., None)`, which delegates here verbatim;
        // Qwen2/DeepSeek-V2 carry no rope scaling, so the plain form is the
        // exact match.)
        //
        // Pin the recorder's own context (lifetime `'a`) before the
        // `&mut rec.tcb` taken in flush_and_reset (see flush_and_reset docs).
        let ctx: &'a MetalContext = rec.tcb.ctx;
        self.flush_and_reset(rec, ctx)?;
        // SAFETY: `buf` is host-visible shared; after the flush the GPU is
        // idle so the in-place rotate over `n_heads * head_dim` floats is
        // safe. Matches the caller's buffer length (q_buf / k_token_buf).
        let ptr = buf.contents() as *mut f32;
        unsafe {
            let all = std::slice::from_raw_parts_mut(ptr, n_heads * head_dim);
            for h in 0..n_heads {
                let head = &mut all[h * head_dim..(h + 1) * head_dim];
                kernels::rope_inplace(head, pos, base);
            }
        }
        Ok(())
    }
}

/// Parse [`FORCE_CPU_OP_ENV`] into the op (if any) forced onto CPU.
/// Unset or unrecognized ⇒ `None` ⇒ pure Metal. Only `rmsnorm`/`rope` are
/// fallback-capable in 3.2; any other recognized-looking value returns
/// `None` (forcing gemv/attention to CPU is explicitly out of scope).
fn parse_force_cpu_op() -> Option<Op> {
    match std::env::var(FORCE_CPU_OP_ENV) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "rmsnorm" => Some(Op::RmsNorm),
            "rope" => Some(Op::Rope),
            _ => None,
        },
        Err(_) => None,
    }
}
