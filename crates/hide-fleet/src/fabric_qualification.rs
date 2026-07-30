//! Qualification labelling for fabric software vs hardware vs simulation.
//!
//! **Law:** never claim physical qualification from simulation or a local
//! multi-process fixture. The honest terminal hardware state on this single
//! M3 Ultra is `FABRIC_HARDWARE_QUALIFICATION_PENDING`.

use serde::{Deserialize, Serialize};

/// How a fabric result was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationKind {
    /// Multi-node physical fabric. Not available in this session.
    PhysicalHardware,
    /// Deterministic placement over an injected/simulated node set.
    Simulated,
    /// Local multi-process software fixture on one machine.
    SoftwareFixture,
}

impl QualificationKind {
    /// Physical hardware is the only kind that may omit
    /// `not_physical_qualification: true`.
    pub fn is_physical(&self) -> bool {
        matches!(self, Self::PhysicalHardware)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PhysicalHardware => "physical_hardware",
            Self::Simulated => "simulated",
            Self::SoftwareFixture => "software_fixture",
        }
    }
}

/// Schema id for qualification-bearing fabric artifacts.
pub const QUALIFICATION_SCHEMA: &str = "hawking.fabric.qualification.v1";

/// Human-readable hardware terminal state for this campaign session.
pub const HARDWARE_QUALIFICATION_PENDING: &str = "FABRIC_HARDWARE_QUALIFICATION_PENDING";

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simulated_is_not_physical() {
        assert!(!QualificationKind::Simulated.is_physical());
        assert!(!QualificationKind::SoftwareFixture.is_physical());
        assert!(QualificationKind::PhysicalHardware.is_physical());
    }
}
