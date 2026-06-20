//! Speculative-decode acceptance governor (pure logic, no GPU, no bench).
//!
//! Serial-verify speculation is **net-negative** on this engine when draft
//! acceptance is low: every rejected draft token costs a wasted slot in the
//! batched verify forward (`forward_tokens_verify`), so a stretch of misses
//! makes spec slower than plain greedy decode (see `docs/dead_levers.md`:
//! EAGLE-3 net-negative tau=0.877; the free n-gram draft tau~1.43 only wins when
//! the user's stream is actually predictable). This [`SpecGovernor`] is a
//! self-contained state machine that watches the live accept/reject stream and
//! flips a single `enabled` bit so spec can never hurt more than a small,
//! bounded amount before it is shut off -- and re-arms itself once acceptance
//! recovers.
//!
//! It is intentionally **pure logic**: one `bool` in, one `bool` out, no
//! tensors, no model handle, no timing. The live decode loop calls
//! [`SpecGovernor::step`] once per spec cycle with whether that cycle accepted
//! at least one draft token, and consults the returned `enabled` flag to decide
//! whether to *propose* on the next cycle. Wiring it into
//! `qwen_dense::forward_token_greedy_tcb` (the `'ud_loop` / `'udpf_loop` accept
//! sites at the `first_reject` / `na` signals) is a deliberate follow-up; this
//! module ships standalone with its transition behaviour fully unit-tested.
//!
//! # Hysteresis (why two thresholds, not one)
//!
//! A single threshold flaps: as soon as the rate dips below it spec disables,
//! and the very next lucky accept re-enables it, paying the warm-up/teardown
//! cost on every wobble. The governor instead uses **asymmetric thresholds plus
//! a cooldown dwell**:
//!
//!   * **disable** when EITHER `consecutive_rejections` reaches
//!     `max_consecutive_rejections` (fast bail on an obvious bad streak -- this
//!     is the "small bound" on how much spec may hurt) OR the rolling accept
//!     rate falls to/below `disable_below` once the window is full;
//!   * once disabled, stay disabled for at least `cooldown_steps` recorded
//!     steps (the dwell) -- during cooldown the loop runs plain greedy but the
//!     governor keeps observing the *would-be* draft outcomes so its window
//!     reflects current predictability;
//!   * **re-enable** only after the cooldown has elapsed AND the rolling accept
//!     rate has climbed to/above `enable_above` (strictly greater than
//!     `disable_below`, so the on/off bands do not touch).
//!
//! The gap `enable_above > disable_below` plus the cooldown dwell is the
//! hysteresis: a rate hovering in the dead-band leaves the current state
//! unchanged, so the enable bit cannot oscillate on noise.

use std::collections::VecDeque;

/// A spec-decode draft is governed in one of two regimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GovState {
    /// Spec is on: the loop should propose draft tokens.
    Enabled,
    /// Spec is off and cooling down: the loop runs plain greedy. The governor
    /// keeps recording observed outcomes; `remaining` counts how many more
    /// recorded steps the dwell lasts before re-enable becomes eligible.
    Cooldown { remaining: usize },
}

/// Rolling-acceptance state machine that auto-enables/disables spec-decode.
///
/// Pure logic: feed it one accept/reject boolean per spec cycle via
/// [`step`](Self::step); it returns whether spec should be active for the next
/// cycle. Construct with [`new`](Self::new) for sensible defaults, or
/// [`with_thresholds`](Self::with_thresholds) to tune.
#[derive(Debug, Clone)]
pub struct SpecGovernor {
    // -- configuration (immutable after construction) --
    /// Number of most-recent cycles the rolling accept rate averages over.
    window: usize,
    /// Re-enable once the rolling rate is `>=` this AND cooldown has elapsed.
    enable_above: f32,
    /// Disable once the rolling rate is `<=` this (window must be full).
    disable_below: f32,
    /// Disable immediately after this many back-to-back zero-accept cycles.
    max_consecutive_rejections: usize,
    /// Minimum number of recorded steps to remain disabled before a re-enable
    /// is even considered (the dwell that prevents flapping).
    cooldown_steps: usize,

    // -- rolling state (mutated by `step`) --
    /// Last `window` accept booleans (front = oldest).
    accepted: VecDeque<bool>,
    /// Running count of `true` entries in `accepted`, kept in sync so
    /// `accept_rate` is O(1).
    accepted_true: usize,
    /// Back-to-back zero-accept cycles, reset on any accept.
    consecutive_rejections: usize,
    /// Current regime.
    state: GovState,
}

impl SpecGovernor {
    /// Construct with default thresholds, tuned for the n-gram draft on code:
    /// `window` cycles averaged, re-enable at `>= min_accept_rate + 0.10`,
    /// disable at `<= min_accept_rate`, bail after 5 straight misses, dwell for
    /// one full window. Starts **enabled** with an optimistic empty-window rate
    /// of 1.0 so a fresh session speculates immediately.
    ///
    /// `window` is clamped to at least 1.
    pub fn new(window: usize, min_accept_rate: f32) -> Self {
        let disable_below = min_accept_rate;
        // Keep the bands strictly separated even if a caller passes a rate near
        // 1.0; +0.10 is the default dead-band width.
        let enable_above = (min_accept_rate + 0.10).min(1.0).max(min_accept_rate);
        Self::with_thresholds(
            window,
            enable_above,
            disable_below,
            5,
            window, // cooldown defaults to one full window of observations
        )
    }

    /// Fully explicit constructor. `enable_above` is clamped to be strictly at
    /// least `disable_below` (a zero-width band degenerates to a single
    /// threshold but is still permitted); `window` and `cooldown_steps` are
    /// clamped to sane minimums. Starts enabled.
    pub fn with_thresholds(
        window: usize,
        enable_above: f32,
        disable_below: f32,
        max_consecutive_rejections: usize,
        cooldown_steps: usize,
    ) -> Self {
        let window = window.max(1);
        let disable_below = disable_below.clamp(0.0, 1.0);
        let enable_above = enable_above.clamp(0.0, 1.0).max(disable_below);
        Self {
            window,
            enable_above,
            disable_below,
            max_consecutive_rejections: max_consecutive_rejections.max(1),
            cooldown_steps,
            accepted: VecDeque::with_capacity(window),
            accepted_true: 0,
            consecutive_rejections: 0,
            state: GovState::Enabled,
        }
    }

    /// Record one spec cycle's outcome and return whether spec should be
    /// **active for the next cycle**.
    ///
    /// `accepted` is `true` when the verify forward accepted at least one draft
    /// token this cycle (in the live loop: `first_reject > 0` / `na > 0`). Call
    /// this once per cycle *even while disabled* -- pass the would-be outcome of
    /// the draft you did not run if you can cheaply estimate it, or simply pass
    /// `false`; either way the window keeps tracking and the cooldown counts
    /// down. The return value is the post-update [`is_enabled`](Self::is_enabled).
    pub fn step(&mut self, accepted: bool) -> bool {
        // 1. roll the window, maintaining the O(1) true-count.
        if self.accepted.len() >= self.window {
            if let Some(old) = self.accepted.pop_front() {
                if old {
                    self.accepted_true -= 1;
                }
            }
        }
        self.accepted.push_back(accepted);
        if accepted {
            self.accepted_true += 1;
            self.consecutive_rejections = 0;
        } else {
            self.consecutive_rejections += 1;
        }

        // 2. advance the state machine.
        match self.state {
            GovState::Enabled => {
                let streak_bail = self.consecutive_rejections >= self.max_consecutive_rejections;
                // Rate-based disable only once the window is full, so a couple
                // of early misses cannot trip it on no evidence.
                let rate_bail =
                    self.accepted.len() >= self.window && self.accept_rate() <= self.disable_below;
                if streak_bail || rate_bail {
                    self.state = GovState::Cooldown {
                        remaining: self.cooldown_steps,
                    };
                }
            }
            GovState::Cooldown { remaining } => {
                let remaining = remaining.saturating_sub(1);
                // Re-enable only after the dwell elapses AND the rate has
                // climbed into the enable band -- the hysteresis gap.
                if remaining == 0 && self.accept_rate() >= self.enable_above {
                    self.state = GovState::Enabled;
                    self.consecutive_rejections = 0;
                } else {
                    self.state = GovState::Cooldown { remaining };
                }
            }
        }

        self.is_enabled()
    }

    /// Whether spec should be active for the next cycle (`true` iff
    /// [`GovState::Enabled`]).
    pub fn is_enabled(&self) -> bool {
        matches!(self.state, GovState::Enabled)
    }

    /// The current regime.
    pub fn state(&self) -> GovState {
        self.state
    }

    /// Rolling acceptance rate over the recorded window. Returns `1.0` on an
    /// empty window (optimistic prior so a fresh session starts speculating).
    pub fn accept_rate(&self) -> f32 {
        if self.accepted.is_empty() {
            return 1.0;
        }
        self.accepted_true as f32 / self.accepted.len() as f32
    }

    /// Number of cycles recorded into the current rolling window (0..=window).
    pub fn observed(&self) -> usize {
        self.accepted.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(gov: &mut SpecGovernor, outcomes: &[bool]) {
        for &o in outcomes {
            gov.step(o);
        }
    }

    #[test]
    fn starts_enabled_with_optimistic_rate() {
        let gov = SpecGovernor::new(20, 0.35);
        assert!(gov.is_enabled());
        assert_eq!(gov.accept_rate(), 1.0);
        assert_eq!(gov.observed(), 0);
    }

    #[test]
    fn rolling_window_is_bounded_and_rate_tracks() {
        let mut gov = SpecGovernor::new(4, 0.35);
        feed(&mut gov, &[true, true, true, true]);
        assert!((gov.accept_rate() - 1.0).abs() < 1e-6);
        assert_eq!(gov.observed(), 4);
        // Push 4 more rejects: the window holds only the last 4 (all false).
        feed(&mut gov, &[false, false, false, false]);
        assert_eq!(gov.observed(), 4, "window must not grow past its size");
        assert!((gov.accept_rate() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn consecutive_rejection_streak_disables_fast() {
        // Even with a wide window, 5 straight misses must bail (the small bound
        // on how much spec is allowed to hurt) without waiting for the rate.
        let mut gov = SpecGovernor::new(50, 0.35);
        assert!(gov.step(false)); // 1 miss, still enabled
        assert!(gov.step(false)); // 2
        assert!(gov.step(false)); // 3
        assert!(gov.step(false)); // 4
        assert!(!gov.step(false), "5th consecutive miss disables");
        assert!(matches!(gov.state(), GovState::Cooldown { .. }));
    }

    #[test]
    fn rate_floor_disables_even_without_a_clean_streak() {
        // No 5-in-a-row streak (an accept breaks it every few steps), but the
        // rolling rate sits at/below the floor once the window fills -> disable.
        // window=8, disable_below=0.35: pattern accept-once-every-4 -> rate 0.25.
        let mut gov = SpecGovernor::with_thresholds(8, 0.50, 0.35, 5, 8);
        // 8 cycles at 2/8 = 0.25 acceptance, max run of misses = 3 (< 5).
        feed(
            &mut gov,
            &[true, false, false, false, true, false, false, false],
        );
        assert!(
            gov.consecutive_rejections() < 5,
            "must not be a streak bail"
        );
        assert!(
            !gov.is_enabled(),
            "low rolling rate must disable via the floor"
        );
    }

    #[test]
    fn full_enable_disable_reenable_cycle() {
        // The headline transition the governor exists for.
        // window=4, disable_below=0.35, enable_above=0.45, cooldown=4.
        let mut gov = SpecGovernor::with_thresholds(4, 0.45, 0.35, 5, 4);

        // (1) ENABLE: starts on, a healthy accept streak keeps it on.
        assert!(gov.is_enabled());
        feed(&mut gov, &[true, true, true, true]);
        assert!(gov.is_enabled(), "high acceptance stays enabled");
        assert!((gov.accept_rate() - 1.0).abs() < 1e-6);

        // (2) DISABLE: a run of misses (>= max_consecutive=5) trips the bail.
        feed(&mut gov, &[false, false, false, false, false]);
        assert!(!gov.is_enabled(), "miss streak disabled spec");
        assert!(matches!(gov.state(), GovState::Cooldown { .. }));

        // (3) HYSTERESIS HOLD: a single recovering accept must NOT re-enable --
        // the cooldown dwell is still counting and the band is not yet cleared.
        assert!(
            !gov.step(true),
            "one lucky accept must not flap spec back on"
        );
        assert!(matches!(gov.state(), GovState::Cooldown { .. }));

        // (4) RE-ENABLE: sustained accepts clear both the cooldown dwell AND
        // push the rolling rate into the enable band (>= 0.45). The window is
        // 4; after enough accepts it reads [t,t,t,t] = 1.0 >= 0.45 and the
        // dwell has elapsed -> back to Enabled.
        feed(&mut gov, &[true, true, true, true]);
        assert!(
            gov.is_enabled(),
            "sustained recovery re-enables spec (rate {} state {:?})",
            gov.accept_rate(),
            gov.state()
        );
    }

    #[test]
    fn dead_band_does_not_flap() {
        // A rate parked strictly between disable_below and enable_above leaves
        // the current state untouched in BOTH directions.
        // band: disable_below=0.30, enable_above=0.60.
        let mut gov = SpecGovernor::with_thresholds(10, 0.60, 0.30, 5, 10);
        // Drive rate to ~0.5 (in the dead-band) without a 5-miss streak.
        feed(
            &mut gov,
            &[
                true, false, true, false, true, false, true, false, true, false,
            ],
        );
        assert!((gov.accept_rate() - 0.5).abs() < 1e-6);
        // 0.5 is below enable_above(0.6) and above disable_below(0.3): an
        // enabled governor stays enabled (no disable trigger fired).
        assert!(
            gov.is_enabled(),
            "dead-band rate must not disable from Enabled"
        );

        // Now force into cooldown via a streak, then hold the rate in the band:
        // it must NOT re-enable while the rate is below enable_above.
        feed(&mut gov, &[false, false, false, false, false]);
        assert!(!gov.is_enabled());
        // Long alternating run: cooldown elapses but rate ~0.5 < 0.6.
        feed(
            &mut gov,
            &[
                true, false, true, false, true, false, true, false, true, false,
            ],
        );
        assert!(
            !gov.is_enabled(),
            "dead-band rate must not re-enable from Cooldown (rate {})",
            gov.accept_rate()
        );
    }

    #[test]
    fn constructor_keeps_bands_separated_and_window_sane() {
        // Degenerate inputs are clamped, never panic.
        let g = SpecGovernor::with_thresholds(0, 0.2, 0.8, 0, 0);
        assert!(g.is_enabled());
        // window clamped to >=1, so one step fills it.
        let mut g = g;
        g.step(true);
        assert_eq!(g.observed(), 1);
    }
}

// Test-only accessor so tests can assert on the private streak counter without
// widening the public surface. Compiled out of the shipped library.
#[cfg(test)]
impl SpecGovernor {
    fn consecutive_rejections(&self) -> usize {
        self.consecutive_rejections
    }
}
