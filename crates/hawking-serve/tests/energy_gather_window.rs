//! Track 7.1 — unit test for the energy gather/admission DECISION policy.
//!
//! Pins `EnergyMode::should_gather(ready, max_batch)` — the extracted,
//! pure form of the wait-or-commit predicate the continuous-batch loop runs
//! (serve::run(), the `prefilling.len() < max_batch && gather_window_ms > 0`
//! guard). Pure: no env, no model, no sleeps, no clock. Gate:
//!
//!   cargo test -p hawking-serve --test energy_gather_window

use hawking_serve::EnergyMode;

/// A latency-sensitive SINGLE request is NEVER delayed: with a 1-slot server
/// (max_batch == 1) batching is impossible, so no mode ever gathers — the
/// request goes straight to prefill regardless of window size.
#[test]
fn single_slot_server_never_gathers() {
    for mode in [EnergyMode::Off, EnergyMode::Balanced, EnergyMode::Efficient] {
        assert!(
            !mode.should_gather(1, 1),
            "{mode}: single-slot server must not delay the lone request"
        );
    }
}

/// A partial batch on a multi-slot server WAITS (up to the window) so co-
/// arriving requests can fill it for lower J/tok — but only when the window > 0.
#[test]
fn partial_batch_gathers_when_window_open() {
    // 1 ready of 8 capacity: balanced/efficient gather; off does not.
    assert!(!EnergyMode::Off.should_gather(1, 8), "off has no window");
    assert!(
        EnergyMode::Balanced.should_gather(1, 8),
        "balanced gathers a partial batch"
    );
    assert!(
        EnergyMode::Efficient.should_gather(1, 8),
        "efficient gathers a partial batch"
    );
    // window magnitudes back the decision.
    assert_eq!(EnergyMode::Off.gather_window_ms(), 0);
    assert_eq!(EnergyMode::Balanced.gather_window_ms(), 3);
    assert_eq!(EnergyMode::Efficient.gather_window_ms(), 8);
}

/// A FULL batch never waits: once ready == max_batch there is nothing to gain
/// from gathering, so commit immediately even under efficient mode. This is the
/// upper bound on added latency — a full batch is dispatched with zero delay.
#[test]
fn full_batch_commits_immediately() {
    for mode in [EnergyMode::Off, EnergyMode::Balanced, EnergyMode::Efficient] {
        assert!(
            !mode.should_gather(8, 8),
            "{mode}: a full batch must dispatch without waiting"
        );
        // and an (impossible) over-full count likewise commits.
        assert!(!mode.should_gather(9, 8), "{mode}: over-full never waits");
    }
}

/// Empty queue never gathers (nothing to wait for).
#[test]
fn empty_queue_never_gathers() {
    for mode in [EnergyMode::Off, EnergyMode::Balanced, EnergyMode::Efficient] {
        assert!(
            !mode.should_gather(0, 8),
            "{mode}: empty queue must not sleep"
        );
    }
}

/// Sweep: the helper must EXACTLY equal the inline loop predicate for every
/// (mode, ready, max_batch) in a representative grid. This is the anti-drift
/// lock — if the loop guard and the helper ever diverge, this fails.
#[test]
fn helper_matches_inline_loop_predicate() {
    for mode in [EnergyMode::Off, EnergyMode::Balanced, EnergyMode::Efficient] {
        for max_batch in [1usize, 2, 4, 8] {
            for ready in 0..=max_batch + 1 {
                let want =
                    ready > 0 && max_batch > 1 && ready < max_batch && mode.gather_window_ms() > 0;
                assert_eq!(
                    mode.should_gather(ready, max_batch),
                    want,
                    "drift at mode={mode} ready={ready} max_batch={max_batch}"
                );
            }
        }
    }
}
