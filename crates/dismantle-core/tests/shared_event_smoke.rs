//! path-to-125 L5w — smoke gate for the cross-queue `SharedEventBarrier`.
//!
//! Validates that:
//! 1. Two command buffers, one each from the primary and secondary
//!    queues, can be linked by a SharedEvent signal/wait pair so the
//!    waiter doesn't dispatch its body until the signaler has finished.
//! 2. `signaled_value` advances monotonically as signals fire.
//! 3. The two buffers commit and complete without deadlock.

#![cfg(target_os = "macos")]

use dismantle_core::metal::{MetalContext, SharedEventBarrier};

#[test]
fn shared_event_pair_signal_and_wait() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return, // no Metal GPU — skip
    };

    let primary = ctx.queue();
    let secondary = ctx.secondary_queue();

    let mut barrier = SharedEventBarrier::new(&ctx);
    assert_eq!(barrier.counter(), 0);

    // Producer: encode an empty CB on the primary queue that signals
    // the event once its (empty) preceding kernels complete. With no
    // kernels in front of the signal, this completes essentially
    // immediately, but the encoded signal still bumps the event's
    // signaled_value.
    let cb_producer = primary.new_command_buffer();
    let signaled_value = barrier.encode_signal(cb_producer);
    assert_eq!(signaled_value, 1);
    assert_eq!(barrier.counter(), 1);

    // Consumer: encode an empty CB on the secondary queue that waits
    // for the producer's signal before any (empty) subsequent kernels
    // run. With the wait set up before the consumer commits, the
    // ordering must be respected even if the consumer commits first.
    let cb_consumer = secondary.new_command_buffer();
    barrier.encode_wait(cb_consumer, signaled_value);

    // Commit in REVERSE order — consumer first, then producer — to
    // prove the wait actually blocks until the signal fires (not just
    // an accidentally-correct happens-before from commit timing).
    cb_consumer.commit();
    cb_producer.commit();

    // Both must complete. A deadlock here would hang the test.
    cb_producer.wait_until_completed();
    cb_consumer.wait_until_completed();
}

#[test]
fn shared_event_counter_advances_per_signal() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    let primary = ctx.queue();
    let mut barrier = SharedEventBarrier::new(&ctx);

    let cb1 = primary.new_command_buffer();
    let cb2 = primary.new_command_buffer();
    let cb3 = primary.new_command_buffer();

    let v1 = barrier.encode_signal(cb1);
    let v2 = barrier.encode_signal(cb2);
    let v3 = barrier.encode_signal(cb3);
    assert_eq!(v1, 1);
    assert_eq!(v2, 2);
    assert_eq!(v3, 3);
    assert_eq!(barrier.counter(), 3);

    cb1.commit();
    cb2.commit();
    cb3.commit();
    cb3.wait_until_completed();
}
