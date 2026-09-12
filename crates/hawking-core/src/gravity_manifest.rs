//! Complete representation accounting bridge for Gravity policy.
//!
//! A candidate cannot enter release selection with a payload-only number.  This
//! manifest charges each persisted representation part and exposes the active
//! byte traffic separately.  It is intentionally codec-agnostic: low-rank,
//! PQ, generators, indices, scales, and sparse repairs are all just charged
//! parts until a concrete direct decoder proves otherwise.

use crate::{gravity_policy::GravityCandidate, Error, Result};

/// One persisted component required to instantiate a representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepresentationPart<'a> {
    pub label: &'a str,
    pub persistent_bytes: u64,
    /// Bytes touched per token on the measured direct execution path.
    pub active_bytes_per_token: u64,
}

/// Typed, complete accounting supplied by an artifact builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GravityManifest<'a> {
    pub id: &'a str,
    /// Number of original logical scalar weights in scope.
    pub logical_weights: u64,
    pub parts: &'a [RepresentationPart<'a>],
    /// Explicit assertion that no required persistent component was omitted.
    pub complete_accounting_attested: bool,
    pub direct_execution: bool,
    pub source_independent: bool,
}

/// Closed byte totals derived from a [`GravityManifest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GravityManifestTotals {
    pub persistent_bytes: u64,
    pub active_bytes_per_token: u64,
    /// Complete effective bits per weight in millionths of a bit.
    pub complete_ebpw_ppm: u64,
}

/// Complete, versioned accounting for a portable Noetic Representation (NR).
///
/// An NR is not merely a compiler intermediate.  It may be runnable directly
/// when its represented cognition, required decoder/runtime material, and
/// declared dependency closure are all present.  A later NX may specialize
/// the exact NR revision for one machine, but NX creation is not required for
/// this manifest to describe a runnable body.
///
/// Every persistent decoder, codebook, code payload, repair payload, metadata
/// table, and model-specific runtime artifact must appear in `parts`.  Values
/// in `external_dependencies` are deliberately *not* charged as free data:
/// their presence makes the manifest open and prevents a runnable-NR claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoeticRepresentationManifest<'a> {
    /// Stable semantic identity for this NR family.
    pub id: &'a str,
    /// Immutable revision within that family.
    pub revision: &'a str,
    /// Pinned parent/source identity retained for lineage, not execution.
    pub parent_source: &'a str,
    /// Number of original logical scalar weights in declared model scope.
    pub logical_weights: u64,
    /// Complete persistence closure of the NR itself.
    pub parts: &'a [RepresentationPart<'a>],
    pub complete_accounting_attested: bool,
    /// The declared representation path consumes the parts directly rather
    /// than reconstructing the dense parent before useful execution.
    pub direct_execution: bool,
    /// No source tensor is needed by the declared execution path.
    pub source_independent: bool,
    /// Mutable recurrent/KV/etc. state allocated for one declared instance.
    pub mutable_state_bytes: u64,
    /// Maximum transient working set needed by the declared direct path.
    pub scratch_bytes: u64,
    /// Named dependencies outside `parts`.  Any entry keeps the candidate
    /// inspectable and billable but deliberately prevents a runnable-NR claim.
    pub external_dependencies: &'a [&'a str],
}

/// Complete NR accounting plus the execution-closure result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoeticRepresentationTotals {
    pub representation: GravityManifestTotals,
    pub mutable_state_bytes: u64,
    pub scratch_bytes: u64,
    pub dependency_closed: bool,
    pub runnable: bool,
}

/// A byte budget for a declared complete EBPW target.
///
/// The budget is deliberately derived from the original logical-weight count,
/// not from a representation payload count.  This makes it useful before a
/// candidate closes: an open source control can expose how much storage a
/// target leaves for the next representation without pretending that it is a
/// runnable NR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoeticRateBudget {
    /// Requested complete EBPW in millionths of one bit per original weight.
    pub target_ebpw_ppm: u64,
    /// Largest integer number of persisted bytes compatible with that target.
    /// The calculation rounds up so a reported budget can never underbill a
    /// fractional final byte.
    pub target_persistent_bytes: u64,
    /// Complete persisted bytes currently charged by the manifest.
    pub billed_persistent_bytes: u64,
    /// Non-zero only when the current charged body fits the target.
    pub bytes_remaining: u64,
    /// Non-zero only when the current charged body exceeds the target.
    pub bytes_over_budget: u64,
}

impl<'a> GravityManifest<'a> {
    /// Validate complete accounting and calculate integer-only byte totals.
    pub fn totals(self) -> Result<GravityManifestTotals> {
        if self.id.is_empty() {
            return Err(Error::Gravity("manifest id must not be empty".into()));
        }
        if self.logical_weights == 0 {
            return Err(Error::Gravity(format!(
                "manifest {} has zero logical weights",
                self.id
            )));
        }
        if self.parts.is_empty() {
            return Err(Error::Gravity(format!(
                "manifest {} has no charged representation parts",
                self.id
            )));
        }
        if !self.complete_accounting_attested {
            return Err(Error::Gravity(format!(
                "manifest {} lacks complete-accounting attestation",
                self.id
            )));
        }
        let mut persistent_bytes = 0u64;
        let mut active_bytes_per_token = 0u64;
        for part in self.parts {
            if part.label.is_empty() {
                return Err(Error::Gravity(format!(
                    "manifest {} has an unnamed representation part",
                    self.id
                )));
            }
            persistent_bytes = persistent_bytes
                .checked_add(part.persistent_bytes)
                .ok_or_else(|| {
                    Error::Gravity(format!("manifest {} persistent bytes overflow", self.id))
                })?;
            active_bytes_per_token = active_bytes_per_token
                .checked_add(part.active_bytes_per_token)
                .ok_or_else(|| {
                    Error::Gravity(format!("manifest {} active bytes overflow", self.id))
                })?;
        }
        let complete_bits = persistent_bytes.checked_mul(8).ok_or_else(|| {
            Error::Gravity(format!("manifest {} complete bits overflow", self.id))
        })?;
        let complete_ebpw_ppm = complete_bits
            .checked_mul(1_000_000)
            .ok_or_else(|| Error::Gravity(format!("manifest {} EBPW ppm overflows", self.id)))?
            / self.logical_weights;
        Ok(GravityManifestTotals {
            persistent_bytes,
            active_bytes_per_token,
            complete_ebpw_ppm,
        })
    }

    /// Construct the policy input only from a closed manifest.
    pub fn candidate(self, error_ppm: u64, verified: bool) -> Result<GravityCandidate<'a>> {
        let totals = self.totals()?;
        Ok(GravityCandidate {
            id: self.id,
            persistent_bytes: totals.persistent_bytes,
            active_bytes_per_token: totals.active_bytes_per_token,
            error_ppm,
            direct_execution: self.direct_execution,
            source_independent: self.source_independent,
            verified,
        })
    }
}

impl<'a> NoeticRepresentationManifest<'a> {
    /// Calculate complete accounting without silently promoting an open
    /// candidate.  This lets Gravity compare an early collapsed substrate
    /// while keeping its external source/teacher dependency explicit.
    pub fn totals(self) -> Result<NoeticRepresentationTotals> {
        for (label, value) in [
            ("NR id", self.id),
            ("NR revision", self.revision),
            ("NR parent source", self.parent_source),
        ] {
            if value.is_empty() {
                return Err(Error::Gravity(format!("{label} must not be empty")));
            }
        }
        if self
            .external_dependencies
            .iter()
            .any(|dependency| dependency.is_empty())
        {
            return Err(Error::Gravity(format!(
                "NR {}:{} has an unnamed external dependency",
                self.id, self.revision
            )));
        }
        if self.external_dependencies.iter().enumerate().any(|(index, value)| {
            self.external_dependencies[..index].contains(value)
        }) {
            return Err(Error::Gravity(format!(
                "NR {}:{} declares an external dependency more than once",
                self.id, self.revision
            )));
        }
        if self.source_independent && !self.external_dependencies.is_empty() {
            return Err(Error::Gravity(format!(
                "NR {}:{} claims source independence while declaring external dependencies",
                self.id, self.revision
            )));
        }
        let representation = GravityManifest {
            id: self.id,
            logical_weights: self.logical_weights,
            parts: self.parts,
            complete_accounting_attested: self.complete_accounting_attested,
            direct_execution: self.direct_execution,
            source_independent: self.source_independent,
        }
        .totals()?;
        let dependency_closed = self.external_dependencies.is_empty();
        Ok(NoeticRepresentationTotals {
            representation,
            mutable_state_bytes: self.mutable_state_bytes,
            scratch_bytes: self.scratch_bytes,
            dependency_closed,
            runnable: dependency_closed && self.direct_execution && self.source_independent,
        })
    }

    /// Refuse an NR runnable claim unless the complete declared execution
    /// closure is direct, source-independent, and free of undeclared helpers.
    pub fn require_runnable(self) -> Result<NoeticRepresentationTotals> {
        let totals = self.totals()?;
        if !totals.dependency_closed {
            return Err(Error::Gravity(format!(
                "NR {}:{} is open: external dependencies prevent runnable status",
                self.id, self.revision
            )));
        }
        if !self.direct_execution {
            return Err(Error::Gravity(format!(
                "NR {}:{} does not execute its represented closure directly",
                self.id, self.revision
            )));
        }
        if !self.source_independent {
            return Err(Error::Gravity(format!(
                "NR {}:{} still requires source tensors",
                self.id, self.revision
            )));
        }
        Ok(totals)
    }

    /// Charge the current complete body against a target EBPW without changing
    /// its closure status.  This is the inexpensive discriminator used before
    /// a large representation experiment: if known retained parts already
    /// consume the target budget, a payload-only improvement elsewhere cannot
    /// produce the advertised complete rate.
    pub fn rate_budget(self, target_ebpw_ppm: u64) -> Result<NoeticRateBudget> {
        if target_ebpw_ppm == 0 {
            return Err(Error::Gravity(format!(
                "NR {}:{} has a zero EBPW target",
                self.id, self.revision
            )));
        }
        let totals = self.totals()?;
        let scaled_bits = self
            .logical_weights
            .checked_mul(target_ebpw_ppm)
            .ok_or_else(|| {
                Error::Gravity(format!(
                    "NR {}:{} target EBPW multiplication overflows",
                    self.id, self.revision
                ))
            })?;
        // `target_ebpw_ppm` is a millionth of a bit and eight bits make a
        // byte, so the denominator is exactly 8,000,000.  Ceiling division is
        // required: a budget may not silently discard a non-integral byte.
        const PPM_BITS_PER_BYTE: u64 = 8_000_000;
        let target_persistent_bytes = scaled_bits
            .checked_add(PPM_BITS_PER_BYTE - 1)
            .ok_or_else(|| {
                Error::Gravity(format!(
                    "NR {}:{} target EBPW ceiling overflows",
                    self.id, self.revision
                ))
            })?
            / PPM_BITS_PER_BYTE;
        Ok(NoeticRateBudget {
            target_ebpw_ppm,
            target_persistent_bytes,
            billed_persistent_bytes: totals.representation.persistent_bytes,
            bytes_remaining: target_persistent_bytes
                .saturating_sub(totals.representation.persistent_bytes),
            bytes_over_budget: totals
                .representation
                .persistent_bytes
                .saturating_sub(target_persistent_bytes),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity_policy::{select, GravityPolicy};

    #[test]
    fn charges_every_part_and_calculates_complete_ebpw() {
        let parts = [
            RepresentationPart {
                label: "codes",
                persistent_bytes: 10,
                active_bytes_per_token: 4,
            },
            RepresentationPart {
                label: "codebook",
                persistent_bytes: 5,
                active_bytes_per_token: 2,
            },
            RepresentationPart {
                label: "metadata",
                persistent_bytes: 1,
                active_bytes_per_token: 0,
            },
        ];
        let manifest = GravityManifest {
            id: "complete",
            logical_weights: 128,
            parts: &parts,
            complete_accounting_attested: true,
            direct_execution: true,
            source_independent: true,
        };
        assert_eq!(
            manifest.totals().unwrap(),
            GravityManifestTotals {
                persistent_bytes: 16,
                active_bytes_per_token: 6,
                complete_ebpw_ppm: 1_000_000,
            }
        );
        assert_eq!(
            select(
                [manifest.candidate(1, true).unwrap()],
                GravityPolicy::default()
            )
            .unwrap()
            .candidate
            .id,
            "complete"
        );
    }

    #[test]
    fn incomplete_or_empty_manifests_are_refused_before_policy() {
        let incomplete = GravityManifest {
            id: "payload-only",
            logical_weights: 1,
            parts: &[RepresentationPart {
                label: "payload",
                persistent_bytes: 1,
                active_bytes_per_token: 1,
            }],
            complete_accounting_attested: false,
            direct_execution: true,
            source_independent: true,
        };
        assert!(incomplete.candidate(1, true).is_err());
        let empty = GravityManifest {
            id: "empty",
            logical_weights: 1,
            parts: &[],
            complete_accounting_attested: true,
            direct_execution: true,
            source_independent: true,
        };
        assert!(empty.totals().is_err());
    }

    #[test]
    fn closed_direct_nr_is_runnable_and_bills_state_separately() {
        let parts = [
            RepresentationPart {
                label: "packed-weights",
                persistent_bytes: 12,
                active_bytes_per_token: 4,
            },
            RepresentationPart {
                label: "decoder-and-metadata",
                persistent_bytes: 4,
                active_bytes_per_token: 1,
            },
        ];
        let nr = NoeticRepresentationManifest {
            id: "flash",
            revision: "r1",
            parent_source: "Flash@pinned",
            logical_weights: 128,
            parts: &parts,
            complete_accounting_attested: true,
            direct_execution: true,
            source_independent: true,
            mutable_state_bytes: 64,
            scratch_bytes: 32,
            external_dependencies: &[],
        };
        assert_eq!(
            nr.require_runnable().unwrap(),
            NoeticRepresentationTotals {
                representation: GravityManifestTotals {
                    persistent_bytes: 16,
                    active_bytes_per_token: 5,
                    complete_ebpw_ppm: 1_000_000,
                },
                mutable_state_bytes: 64,
                scratch_bytes: 32,
                dependency_closed: true,
                runnable: true,
            }
        );
    }

    #[test]
    fn open_nr_is_billable_but_not_runnable() {
        let parts = [RepresentationPart {
            label: "packed-substrate",
            persistent_bytes: 8,
            active_bytes_per_token: 4,
        }];
        let nr = NoeticRepresentationManifest {
            id: "flash",
            revision: "open-r0",
            parent_source: "Flash@pinned",
            logical_weights: 64,
            parts: &parts,
            complete_accounting_attested: true,
            direct_execution: false,
            source_independent: false,
            mutable_state_bytes: 0,
            scratch_bytes: 0,
            external_dependencies: &["source-BF16 expert bank"],
        };
        let totals = nr.totals().unwrap();
        assert_eq!(totals.representation.complete_ebpw_ppm, 1_000_000);
        assert!(!totals.dependency_closed);
        assert!(!totals.runnable);
        assert!(nr.require_runnable().is_err());
    }

    #[test]
    fn rate_budget_uses_logical_weights_and_never_rounds_down_a_byte() {
        let parts = [RepresentationPart {
            label: "source-control",
            persistent_bytes: 100,
            active_bytes_per_token: 10,
        }];
        let nr = NoeticRepresentationManifest {
            id: "flash",
            revision: "budget-r0",
            parent_source: "Flash@pinned",
            logical_weights: 100,
            parts: &parts,
            complete_accounting_attested: true,
            direct_execution: false,
            source_independent: false,
            mutable_state_bytes: 0,
            scratch_bytes: 0,
            external_dependencies: &["source-control storage"],
        };
        // 100 weights * 0.5 bits = 50 bits, which must reserve seven bytes.
        assert_eq!(
            nr.rate_budget(500_000).unwrap(),
            NoeticRateBudget {
                target_ebpw_ppm: 500_000,
                target_persistent_bytes: 7,
                billed_persistent_bytes: 100,
                bytes_remaining: 0,
                bytes_over_budget: 93,
            }
        );
    }

    #[test]
    fn zero_rate_budget_is_refused() {
        let parts = [RepresentationPart {
            label: "codes",
            persistent_bytes: 1,
            active_bytes_per_token: 1,
        }];
        let nr = NoeticRepresentationManifest {
            id: "flash",
            revision: "budget-r1",
            parent_source: "Flash@pinned",
            logical_weights: 1,
            parts: &parts,
            complete_accounting_attested: true,
            direct_execution: false,
            source_independent: false,
            mutable_state_bytes: 0,
            scratch_bytes: 0,
            external_dependencies: &["source-control storage"],
        };
        assert!(nr.rate_budget(0).is_err());
    }
}
