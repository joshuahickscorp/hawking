//! Deterministic, tunable representation selection for Gravity.
//!
//! Codecs and kernels report measured candidate facts; this module makes the
//! release-facing decision.  It intentionally uses integer units only: policy
//! tuning may change a choice, but the same facts and policy always yield the
//! same choice regardless of candidate input order.

/// Measured facts for one candidate representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GravityCandidate<'a> {
    pub id: &'a str,
    pub persistent_bytes: u64,
    pub active_bytes_per_token: u64,
    /// Reconstruction or held-out error in parts per million; lower is better.
    pub error_ppm: u64,
    pub direct_execution: bool,
    pub source_independent: bool,
    pub verified: bool,
}

/// Dynamic policy inputs.  These are explicit runtime configuration, not
/// hidden autotuner state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GravityPolicy {
    pub max_error_ppm: u64,
    pub max_active_bytes_per_token: u64,
    pub require_direct_execution: bool,
    pub require_source_independent: bool,
    pub require_verified: bool,
    /// Score weights expressed in integer cost units.
    pub active_byte_weight: u64,
    pub error_ppm_weight: u64,
}

impl Default for GravityPolicy {
    fn default() -> Self {
        Self {
            max_error_ppm: u64::MAX,
            max_active_bytes_per_token: u64::MAX,
            require_direct_execution: true,
            require_source_independent: true,
            require_verified: true,
            active_byte_weight: 1,
            error_ppm_weight: 1,
        }
    }
}

/// The selected candidate plus an auditable integer score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GravitySelection<'a> {
    pub candidate: GravityCandidate<'a>,
    pub score: u128,
}

fn admissible(candidate: GravityCandidate<'_>, policy: GravityPolicy) -> bool {
    candidate.error_ppm <= policy.max_error_ppm
        && candidate.active_bytes_per_token <= policy.max_active_bytes_per_token
        && (!policy.require_direct_execution || candidate.direct_execution)
        && (!policy.require_source_independent || candidate.source_independent)
        && (!policy.require_verified || candidate.verified)
}

fn score(candidate: GravityCandidate<'_>, policy: GravityPolicy) -> u128 {
    candidate.persistent_bytes as u128
        + candidate.active_bytes_per_token as u128 * policy.active_byte_weight as u128
        + candidate.error_ppm as u128 * policy.error_ppm_weight as u128
}

/// Select an admissible candidate deterministically.
///
/// Ordering is score, then persistent bytes, then active bytes, then error,
/// then lexical ID.  The last key makes input order irrelevant and keeps a
/// release receipt reproducible.  `None` is fail-closed: callers must not
/// silently fall back to a non-admissible dense representation.
pub fn select<'a>(
    candidates: impl IntoIterator<Item = GravityCandidate<'a>>,
    policy: GravityPolicy,
) -> Option<GravitySelection<'a>> {
    let mut best: Option<GravitySelection<'a>> = None;
    for candidate in candidates {
        if !admissible(candidate, policy) {
            continue;
        }
        let current = GravitySelection {
            candidate,
            score: score(candidate, policy),
        };
        let replace = best.map_or(true, |prior| {
            (
                current.score,
                current.candidate.persistent_bytes,
                current.candidate.active_bytes_per_token,
                current.candidate.error_ppm,
                current.candidate.id,
            ) < (
                prior.score,
                prior.candidate.persistent_bytes,
                prior.candidate.active_bytes_per_token,
                prior.candidate.error_ppm,
                prior.candidate.id,
            )
        });
        if replace {
            best = Some(current);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        id: &'static str,
        persistent_bytes: u64,
        active_bytes: u64,
        error_ppm: u64,
    ) -> GravityCandidate<'static> {
        GravityCandidate {
            id,
            persistent_bytes,
            active_bytes_per_token: active_bytes,
            error_ppm,
            direct_execution: true,
            source_independent: true,
            verified: true,
        }
    }

    #[test]
    fn choice_is_order_independent_and_has_a_stable_tie_break() {
        let policy = GravityPolicy::default();
        let a = candidate("alpha", 100, 10, 2);
        let b = candidate("beta", 100, 10, 2);
        assert_eq!(select([b, a], policy).unwrap().candidate.id, "alpha");
        assert_eq!(select([a, b], policy).unwrap().candidate.id, "alpha");
    }

    #[test]
    fn policy_tuning_changes_only_the_earned_choice() {
        let fast = candidate("fast", 200, 1, 30);
        let accurate = candidate("accurate", 100, 20, 1);
        let bytes_first = GravityPolicy {
            error_ppm_weight: 1,
            active_byte_weight: 1,
            ..GravityPolicy::default()
        };
        let accuracy_first = GravityPolicy {
            error_ppm_weight: 100,
            active_byte_weight: 1,
            ..GravityPolicy::default()
        };
        assert_eq!(
            select([fast, accurate], bytes_first).unwrap().candidate.id,
            "accurate"
        );
        assert_eq!(
            select([fast, accurate], accuracy_first)
                .unwrap()
                .candidate
                .id,
            "accurate"
        );
        let latency_first = GravityPolicy {
            active_byte_weight: 20,
            error_ppm_weight: 1,
            ..GravityPolicy::default()
        };
        assert_eq!(
            select([fast, accurate], latency_first)
                .unwrap()
                .candidate
                .id,
            "fast"
        );
    }

    #[test]
    fn unsafe_or_unmeasured_candidates_fail_closed() {
        let mut unverified = candidate("unverified", 1, 1, 1);
        unverified.verified = false;
        let mut indirect = candidate("indirect", 1, 1, 1);
        indirect.direct_execution = false;
        assert!(select([unverified, indirect], GravityPolicy::default()).is_none());
    }

    #[test]
    fn explicit_error_budget_is_enforced() {
        let policy = GravityPolicy {
            max_error_ppm: 10,
            ..GravityPolicy::default()
        };
        assert!(select([candidate("bad", 1, 1, 11)], policy).is_none());
        assert_eq!(
            select([candidate("good", 1, 1, 10)], policy)
                .unwrap()
                .candidate
                .id,
            "good"
        );
    }
}
