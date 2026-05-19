//! path-to-125 L5 — smoke gate for the secondary `MTLCommandQueue`.
//!
//! Validates that MetalContext exposes a secondary queue distinct from
//! the primary queue, and that both queues can produce and commit a
//! trivial command buffer back-to-back without deadlocking. This is the
//! minimum bar for the L5 scaffold; full Eagle4 chain-decode wiring
//! (head propose on secondary, verifier on primary, `MTLSharedEvent`
//! sync at the layer boundary) is deferred to a follow-up.

#![cfg(target_os = "macos")]

use dismantle_core::metal::{MetalContext, TokenCommandBuffer};

#[test]
fn secondary_queue_distinct_and_dispatchable() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => {
            // CI without a Metal-capable GPU — skip.
            return;
        }
    };

    let primary = ctx.queue();
    let secondary = ctx.secondary_queue();

    // Distinctness — the two accessors must NOT return clones of the same
    // queue. The metal crate's `CommandQueue` is `Send + Sync` and clones
    // are shallow references, so comparing `as *const _` is the cheapest
    // proof that the underlying ObjC objects differ.
    let p_ptr = primary as *const _ as usize;
    let s_ptr = secondary as *const _ as usize;
    assert_ne!(
        p_ptr, s_ptr,
        "secondary_queue() returned the same CommandQueue as queue()"
    );

    // Liveness — both queues must accept and run an empty command buffer.
    // An empty buffer commits in microseconds; if the queue is broken
    // this will hang or panic.
    let cb_primary = primary.new_command_buffer();
    cb_primary.commit();
    cb_primary.wait_until_completed();

    let cb_secondary = secondary.new_command_buffer();
    cb_secondary.commit();
    cb_secondary.wait_until_completed();
}

#[test]
fn tcb_on_secondary_constructs_and_commits() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    // path-to-125 L5w — TokenCommandBuffer::new_on_secondary should
    // produce a TCB whose underlying CommandBuffer comes from the
    // secondary queue. A trivial commit_and_wait must not deadlock
    // and must not interfere with a subsequent primary-queue TCB.
    {
        let tcb_sec = TokenCommandBuffer::new_on_secondary(&ctx);
        tcb_sec.commit_and_wait().expect("secondary TCB commit");
    }
    {
        let tcb_pri = TokenCommandBuffer::new(&ctx);
        tcb_pri.commit_and_wait().expect("primary TCB commit");
    }
}
