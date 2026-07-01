//! Spine A — recall fidelity for an SSM's constant-size recurrent state.
//!
//! RWKV-7 has no token cap: its ~6-16 MiB state is constant, and older context
//! decays in salience rather than falling off a hard edge. So "how full is the
//! window" is the wrong question — the right one is "how sharp is recall over
//! what the state has absorbed". This module models that as a 0..1 fidelity from
//! the state's age (tokens absorbed) against a native recall horizon. The
//! default is a conservative linear decay; a measured boot-needle calibration
//! replaces it later (the trait keeps that swap a one-liner).

/// A 0..1 recall-fidelity estimator for an aging recurrent state.
pub trait RecallFidelityProbe: Send + Sync {
    /// Fidelity in `0..=1` for a state that has absorbed `state_age_tokens`.
    fn fidelity(&self, state_age_tokens: usize) -> f32;
}

/// Conservative default: fidelity decays linearly from 1.0 toward `floor` across
/// `horizon_tokens`, then holds at `floor` (recall is soft, never zero — the
/// state still carries a rolling summary). Calibrated boot-needle values replace
/// this without changing the call site.
#[derive(Debug, Clone, Copy)]
pub struct LinearFidelity {
    pub horizon_tokens: usize,
    pub floor: f32,
}

impl LinearFidelity {
    /// `horizon_tokens` is typically the model's native recall horizon
    /// (`rwkv7.context_length`). Floor defaults to 0.3 (old context stays partly
    /// recoverable from the rolling state).
    pub fn new(horizon_tokens: usize) -> Self {
        Self { horizon_tokens, floor: 0.3 }
    }
}

impl RecallFidelityProbe for LinearFidelity {
    fn fidelity(&self, state_age_tokens: usize) -> f32 {
        if self.horizon_tokens == 0 {
            return 1.0;
        }
        let decayed = 1.0 - (state_age_tokens as f32 / self.horizon_tokens as f32);
        decayed.clamp(self.floor, 1.0)
    }
}

/// Measured-curve recall fidelity (W0-FID evaluator): piecewise-linear
/// interpolation over calibration knots `(state_age_tokens, fidelity)` produced
/// by a boot-needle probe on a real model. The CALIBRATION (running the needle
/// probe to fill the knots) is model-gated; this evaluator is the pure drop-in
/// for [`LinearFidelity`] that consumes those knots. Fidelity is clamped monotone
/// non-increasing (recall does not recover with age) and held flat outside the
/// measured range.
#[derive(Debug, Clone)]
pub struct SplineFidelity {
    knots: Vec<(usize, f32)>,
}

impl SplineFidelity {
    /// Build from calibration knots; sorts by age and enforces monotone
    /// non-increasing fidelity. Empty knots -> a degenerate full-fidelity curve.
    pub fn new(mut knots: Vec<(usize, f32)>) -> Self {
        knots.sort_by_key(|(age, _)| *age);
        let mut prev = 1.0f32;
        for (_, f) in knots.iter_mut() {
            *f = f.clamp(0.0, 1.0).min(prev);
            prev = *f;
        }
        Self { knots }
    }
}

impl RecallFidelityProbe for SplineFidelity {
    fn fidelity(&self, state_age_tokens: usize) -> f32 {
        match self.knots.as_slice() {
            [] => 1.0,
            [single] => single.1,
            knots => {
                let age = state_age_tokens;
                if age <= knots[0].0 {
                    return knots[0].1;
                }
                let last = knots[knots.len() - 1];
                if age >= last.0 {
                    return last.1;
                }
                for w in knots.windows(2) {
                    let (a0, f0) = w[0];
                    let (a1, f1) = w[1];
                    if age >= a0 && age <= a1 {
                        if a1 == a0 {
                            return f1;
                        }
                        let t = (age - a0) as f32 / (a1 - a0) as f32;
                        return f0 + t * (f1 - f0);
                    }
                }
                last.1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_state_is_sharp_and_decays_to_floor() {
        let p = LinearFidelity::new(1000);
        assert!((p.fidelity(0) - 1.0).abs() < 1e-6);
        assert!(p.fidelity(500) > 0.49 && p.fidelity(500) < 0.51);
        assert!((p.fidelity(100_000) - p.floor).abs() < 1e-6, "holds at floor, never 0");
    }

    #[test]
    fn zero_horizon_is_full() {
        assert_eq!(LinearFidelity { horizon_tokens: 0, floor: 0.3 }.fidelity(123), 1.0);
    }

    #[test]
    fn spline_interpolates_between_knots() {
        let p = SplineFidelity::new(vec![(0, 1.0), (1000, 0.5), (2000, 0.3)]);
        assert!((p.fidelity(0) - 1.0).abs() < 1e-6);
        assert!((p.fidelity(500) - 0.75).abs() < 1e-6, "midpoint of 1.0..0.5");
        assert!((p.fidelity(1500) - 0.4).abs() < 1e-6);
        assert!((p.fidelity(5000) - 0.3).abs() < 1e-6, "held flat past last knot");
    }

    #[test]
    fn spline_enforces_monotone_non_increasing() {
        // A noisy knot that rises is clamped down to the prior fidelity.
        let p = SplineFidelity::new(vec![(0, 1.0), (1000, 0.4), (2000, 0.9)]);
        assert!(p.fidelity(2000) <= 0.4 + 1e-6, "recall cannot recover with age");
    }

    #[test]
    fn spline_empty_is_full() {
        assert_eq!(SplineFidelity::new(vec![]).fidelity(123), 1.0);
    }
}
