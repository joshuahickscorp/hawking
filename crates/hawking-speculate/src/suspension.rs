//! Speculation suspension policy — one enumerable list, one place.
//!
//! Past each of these boundaries a wrong draft is not discarded work: it is an
//! observable effect (tool side-effect, schema commitment, permission grant,
//! file mutation, citation claim, user interrupt, compacted context, model
//! switch, or Fabric migration). Speculation must **suspend**, **constrain**,
//! or **flush-then-continue** before crossing them.
//!
//! Do not re-encode this table as scattered `if` statements in the decode loop.
//! Call [`policy_for`] / [`action_for`] at the boundary detector.

use core::fmt;

/// Every hard speculation boundary. Exhaustive: adding a variant forces the
/// policy table and tests to update.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SuspensionBoundary {
    /// Model is about to open a tool call (name/args). Wrong draft → real side effect.
    ToolCallStart,
    /// Structured emission / JSON schema boundary. Wrong draft opens a schema the
    /// target then disagrees with.
    JsonSchema,
    /// Permission / security gate. Draft must not drive an approve/deny.
    Permission,
    /// File edit / diff apply. Draft must not mutate the workspace.
    EditDiff,
    /// Citation or evidence claim that becomes user-visible provenance.
    CitationEvidence,
    /// User interrupt (cancel/pause/steer). Spec must not race the interrupt.
    UserInterrupt,
    /// Context compaction rewrites durable history; drafts are not history.
    ContextCompaction,
    /// Model or profile switch invalidates the draft proposer + KV assumptions.
    ModelProfileSwitch,
    /// Fabric / weight migration — different substrate, drafts are invalid.
    FabricMigration,
}

/// What speculation does at a boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SuspensionAction {
    /// Stop proposing; finish only with already-verified tokens; resume after.
    Suspend,
    /// Keep proposing but narrow the draft space (e.g. forbid tool-call tokens).
    Constrain,
    /// Verify+commit any open provisional KV, then continue under the new regime.
    FlushThenContinue,
}

/// One policy row: boundary + action + why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuspensionPoint {
    pub boundary: SuspensionBoundary,
    pub action: SuspensionAction,
    pub rationale: &'static str,
}

/// The single policy table. Order is stable for docs and tests.
pub static SUSPENSION_POLICY: &[SuspensionPoint] = &[
    SuspensionPoint {
        boundary: SuspensionBoundary::ToolCallStart,
        action: SuspensionAction::Suspend,
        rationale:
            "A draft tool-call start can invoke real tools; only target-verified tokens may open a call.",
    },
    SuspensionPoint {
        boundary: SuspensionBoundary::JsonSchema,
        action: SuspensionAction::Constrain,
        rationale:
            "A draft must not open structured emission the target later rejects; constrain to free text until verified.",
    },
    SuspensionPoint {
        boundary: SuspensionBoundary::Permission,
        action: SuspensionAction::Suspend,
        rationale:
            "Permission decisions are durable security state; drafts must not approve or deny.",
    },
    SuspensionPoint {
        boundary: SuspensionBoundary::EditDiff,
        action: SuspensionAction::Suspend,
        rationale:
            "File edits are durable workspace mutations; only verified tokens may drive diffs.",
    },
    SuspensionPoint {
        boundary: SuspensionBoundary::CitationEvidence,
        action: SuspensionAction::Suspend,
        rationale:
            "Citations become user-visible provenance; unverified draft claims must not be cited.",
    },
    SuspensionPoint {
        boundary: SuspensionBoundary::UserInterrupt,
        action: SuspensionAction::FlushThenContinue,
        rationale:
            "Flush accepted prefix so the interrupt sees a consistent committed state, then stop drafting.",
    },
    SuspensionPoint {
        boundary: SuspensionBoundary::ContextCompaction,
        action: SuspensionAction::FlushThenContinue,
        rationale:
            "Compaction rewrites durable context; provisional draft KV must not be compacted in.",
    },
    SuspensionPoint {
        boundary: SuspensionBoundary::ModelProfileSwitch,
        action: SuspensionAction::Suspend,
        rationale:
            "A new model/profile invalidates draft proposers and KV; suspend until re-anchored.",
    },
    SuspensionPoint {
        boundary: SuspensionBoundary::FabricMigration,
        action: SuspensionAction::Suspend,
        rationale:
            "Fabric migration changes the substrate; all provisional draft state is invalid.",
    },
];

/// Look up the policy row for a boundary. Panics only if the static table is
/// missing a variant (caught by `policy_covers_every_boundary` test).
pub fn policy_for(boundary: SuspensionBoundary) -> SuspensionPoint {
    SUSPENSION_POLICY
        .iter()
        .copied()
        .find(|p| p.boundary == boundary)
        .expect("SUSPENSION_POLICY must list every SuspensionBoundary variant")
}

#[inline]
pub fn action_for(boundary: SuspensionBoundary) -> SuspensionAction {
    policy_for(boundary).action
}

/// Runtime controller: given a detected boundary, returns whether speculation
/// may continue proposing drafts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculationPermit {
    /// May propose drafts (subject to any active constrain set).
    Continue,
    /// Must not propose; verified path only.
    Suspended {
        boundary: SuspensionBoundary,
        action: SuspensionAction,
    },
    /// May propose under a narrowed constraint set.
    Constrained { boundary: SuspensionBoundary },
}

impl SpeculationPermit {
    pub fn may_propose(self) -> bool {
        matches!(
            self,
            SpeculationPermit::Continue | SpeculationPermit::Constrained { .. }
        )
    }
}

/// Evaluate the policy for a detected boundary (or none).
pub fn evaluate(boundary: Option<SuspensionBoundary>) -> SpeculationPermit {
    match boundary {
        None => SpeculationPermit::Continue,
        Some(b) => {
            let action = action_for(b);
            match action {
                SuspensionAction::Suspend => SpeculationPermit::Suspended {
                    boundary: b,
                    action,
                },
                SuspensionAction::Constrain => SpeculationPermit::Constrained { boundary: b },
                SuspensionAction::FlushThenContinue => SpeculationPermit::Suspended {
                    boundary: b,
                    action,
                },
            }
        }
    }
}

/// Whether the action requires a KV flush of accepted provisional state first.
pub fn requires_flush(action: SuspensionAction) -> bool {
    matches!(action, SuspensionAction::FlushThenContinue)
}

impl fmt::Display for SuspensionBoundary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SuspensionBoundary::ToolCallStart => "tool_call_start",
            SuspensionBoundary::JsonSchema => "json_schema",
            SuspensionBoundary::Permission => "permission",
            SuspensionBoundary::EditDiff => "edit_diff",
            SuspensionBoundary::CitationEvidence => "citation_evidence",
            SuspensionBoundary::UserInterrupt => "user_interrupt",
            SuspensionBoundary::ContextCompaction => "context_compaction",
            SuspensionBoundary::ModelProfileSwitch => "model_profile_switch",
            SuspensionBoundary::FabricMigration => "fabric_migration",
        };
        f.write_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn policy_covers_every_boundary_exactly_once() {
        let all = [
            SuspensionBoundary::ToolCallStart,
            SuspensionBoundary::JsonSchema,
            SuspensionBoundary::Permission,
            SuspensionBoundary::EditDiff,
            SuspensionBoundary::CitationEvidence,
            SuspensionBoundary::UserInterrupt,
            SuspensionBoundary::ContextCompaction,
            SuspensionBoundary::ModelProfileSwitch,
            SuspensionBoundary::FabricMigration,
        ];
        assert_eq!(SUSPENSION_POLICY.len(), all.len());
        for b in all {
            let matches: Vec<_> = SUSPENSION_POLICY
                .iter()
                .filter(|p| p.boundary == b)
                .collect();
            assert_eq!(matches.len(), 1, "boundary {b} must appear exactly once");
            assert!(!matches[0].rationale.is_empty());
        }
    }
    fn assert_suspends(b: SuspensionBoundary) {
        let permit = evaluate(Some(b));
        assert!(!permit.may_propose() || matches!(permit, SpeculationPermit::Constrained { .. }));
        let point = policy_for(b);
        match point.action {
            SuspensionAction::Suspend | SuspensionAction::FlushThenContinue => {
                assert!(!evaluate(Some(b)).may_propose(), "{b} must not propose");
            }
            SuspensionAction::Constrain => {
                assert!(matches!(
                    evaluate(Some(b)),
                    SpeculationPermit::Constrained { boundary } if boundary == b
                ));
            }
        }
    }
    #[test]
    fn suspension_tool_call_start() {
        assert_eq!(
            action_for(SuspensionBoundary::ToolCallStart),
            SuspensionAction::Suspend
        );
        assert_suspends(SuspensionBoundary::ToolCallStart);
    }
    #[test]
    fn suspension_json_schema() {
        assert_eq!(
            action_for(SuspensionBoundary::JsonSchema),
            SuspensionAction::Constrain
        );
        assert_suspends(SuspensionBoundary::JsonSchema);
    }
    #[test]
    fn suspension_permission() {
        assert_eq!(
            action_for(SuspensionBoundary::Permission),
            SuspensionAction::Suspend
        );
        assert_suspends(SuspensionBoundary::Permission);
    }
    #[test]
    fn suspension_edit_diff() {
        assert_eq!(
            action_for(SuspensionBoundary::EditDiff),
            SuspensionAction::Suspend
        );
        assert_suspends(SuspensionBoundary::EditDiff);
    }
    #[test]
    fn suspension_citation_evidence() {
        assert_eq!(
            action_for(SuspensionBoundary::CitationEvidence),
            SuspensionAction::Suspend
        );
        assert_suspends(SuspensionBoundary::CitationEvidence);
    }
    #[test]
    fn suspension_user_interrupt() {
        assert_eq!(
            action_for(SuspensionBoundary::UserInterrupt),
            SuspensionAction::FlushThenContinue
        );
        assert!(requires_flush(action_for(
            SuspensionBoundary::UserInterrupt
        )));
        assert_suspends(SuspensionBoundary::UserInterrupt);
    }
    #[test]
    fn suspension_context_compaction() {
        assert_eq!(
            action_for(SuspensionBoundary::ContextCompaction),
            SuspensionAction::FlushThenContinue
        );
        assert!(requires_flush(action_for(
            SuspensionBoundary::ContextCompaction
        )));
        assert_suspends(SuspensionBoundary::ContextCompaction);
    }
    #[test]
    fn suspension_model_profile_switch() {
        assert_eq!(
            action_for(SuspensionBoundary::ModelProfileSwitch),
            SuspensionAction::Suspend
        );
        assert_suspends(SuspensionBoundary::ModelProfileSwitch);
    }
    #[test]
    fn suspension_fabric_migration() {
        assert_eq!(
            action_for(SuspensionBoundary::FabricMigration),
            SuspensionAction::Suspend
        );
        assert_suspends(SuspensionBoundary::FabricMigration);
    }
    #[test]
    fn no_boundary_allows_continue() {
        assert_eq!(evaluate(None), SpeculationPermit::Continue);
        assert!(evaluate(None).may_propose());
    }
}
