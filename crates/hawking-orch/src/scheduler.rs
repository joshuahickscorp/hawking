//! Energy / thermal / RAM-aware admission control (ch.06 §4.11).
//!
//! On a laptop the model fleet is a *power budget*. Before a role is admitted to
//! run, the scheduler checks it against a [`ResourceSnapshot`] (free RAM, a
//! thermal-headroom proxy, in-flight count, battery/mode) and returns
//! [`Admission::Admit`] or [`Admission::Defer`] with a **structured reason**
//! (`Ram` / `Thermal` / `Concurrency` / `Energy`). The router (§4.4 step 4)
//! consults it to pick a smaller role or back off spec when the budget is tight.
//!
//! Pure policy — no OS probing here (that is the host's `ResourceProbe`); the
//! snapshot is fed in, so the predicates are fully testable.

use hide_core::runtime::ModelRole;
use serde::{Deserialize, Serialize};

/// User-facing power mode (the §4.11 dial).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerMode {
    /// Full fleet, full spec, biggest roles.
    PluggedPerf,
    /// Default.
    Balanced,
    /// Smaller roles, throttled concurrency, quieter fans.
    Quiet,
}

impl Default for PowerMode {
    fn default() -> Self {
        PowerMode::Balanced
    }
}

/// A point-in-time view of the machine's budget. Supplied by the host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    /// Free unified memory in MB (RAM == VRAM on Apple Silicon).
    pub ram_free_mb: u64,
    /// Thermal headroom proxy in [0,1]: 1.0 = cool, 0.0 = throttling.
    pub thermal_headroom: f32,
    /// Number of generations currently in flight across the fleet.
    pub in_flight: u32,
    /// Whether the machine is on battery.
    pub on_battery: bool,
    /// Active power mode.
    pub mode: PowerMode,
}

impl Default for ResourceSnapshot {
    fn default() -> Self {
        Self {
            ram_free_mb: u64::MAX,
            thermal_headroom: 1.0,
            in_flight: 0,
            on_battery: false,
            mode: PowerMode::Balanced,
        }
    }
}

/// Why a role was deferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeferReason {
    /// The role's footprint won't fit in free RAM (plus headroom).
    Ram,
    /// Thermal headroom too low for a role this heavy.
    Thermal,
    /// Too many generations already in flight.
    Concurrency,
    /// On battery in quiet mode: this role is too expensive.
    Energy,
}

/// The admission verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Admission {
    Admit,
    Defer { reason: DeferReason, detail: String },
}

impl Admission {
    pub fn is_admit(&self) -> bool {
        matches!(self, Admission::Admit)
    }
}

/// Admission policy thresholds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdmissionPolicy {
    /// RAM kept free as headroom beyond the role's footprint (MB).
    pub ram_headroom_mb: u64,
    /// Below this thermal headroom, only light roles are admitted.
    pub thermal_low: f32,
    /// A role footprint (MB) considered "heavy" for thermal/energy gating.
    pub heavy_footprint_mb: u64,
    /// Max concurrent generations (balanced/plugged).
    pub max_concurrency: u32,
    /// Max concurrent generations on battery / quiet.
    pub max_concurrency_battery: u32,
}

impl Default for AdmissionPolicy {
    fn default() -> Self {
        Self {
            ram_headroom_mb: 1_024,
            thermal_low: 0.2,
            heavy_footprint_mb: 4_000,
            max_concurrency: 6,
            max_concurrency_battery: 2,
        }
    }
}

/// The admission controller.
#[derive(Debug, Clone, Default)]
pub struct Scheduler {
    pub policy: AdmissionPolicy,
}

impl Scheduler {
    pub fn new(policy: AdmissionPolicy) -> Self {
        Self { policy }
    }

    /// Decide whether `role` may run under `snapshot`. Order matters: RAM is a
    /// hard wall, then concurrency, then thermal/energy (which prefer a smaller
    /// role rather than block outright — the router uses the reason to downgrade).
    pub fn admit(&self, role: &ModelRole, snapshot: &ResourceSnapshot) -> Admission {
        let footprint = role.model.footprint_mb;

        // 1. RAM: a shared-process role (footprint 0) always fits.
        if footprint > 0 {
            let needed = footprint.saturating_add(self.policy.ram_headroom_mb);
            if needed > snapshot.ram_free_mb {
                return Admission::Defer {
                    reason: DeferReason::Ram,
                    detail: format!(
                        "role '{}' needs {}MB (+{}MB headroom) but only {}MB free",
                        role.name, footprint, self.policy.ram_headroom_mb, snapshot.ram_free_mb
                    ),
                };
            }
        }

        // 2. Concurrency cap (battery/quiet lowers it).
        let cap = if snapshot.on_battery || snapshot.mode == PowerMode::Quiet {
            self.policy.max_concurrency_battery
        } else {
            self.policy.max_concurrency
        };
        if snapshot.in_flight >= cap {
            return Admission::Defer {
                reason: DeferReason::Concurrency,
                detail: format!("{} in flight ≥ cap {cap}", snapshot.in_flight),
            };
        }

        let heavy = footprint >= self.policy.heavy_footprint_mb;

        // 3. Thermal: throttling blocks heavy roles.
        if heavy && snapshot.thermal_headroom < self.policy.thermal_low {
            return Admission::Defer {
                reason: DeferReason::Thermal,
                detail: format!(
                    "thermal headroom {:.2} < {:.2}; defer heavy role '{}'",
                    snapshot.thermal_headroom, self.policy.thermal_low, role.name
                ),
            };
        }

        // 4. Energy: on battery in quiet mode, defer heavy roles to prefer a
        //    smaller one (the router downgrades on this reason).
        if heavy && snapshot.on_battery && snapshot.mode == PowerMode::Quiet {
            return Admission::Defer {
                reason: DeferReason::Energy,
                detail: format!(
                    "on battery + quiet: defer heavy role '{}' to a lighter one",
                    role.name
                ),
            };
        }

        Admission::Admit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::{ModelId, RoleId};
    use hide_core::runtime::{
        ModelArchitecture, ModelDescriptor, ProviderCaps, RolePurpose, SamplerProfile,
    };
    use std::collections::BTreeMap;

    fn role(footprint_mb: u64) -> ModelRole {
        ModelRole {
            id: RoleId::new(),
            name: "test".into(),
            purpose: RolePurpose::HeroCoder,
            model: ModelDescriptor {
                id: ModelId::new(),
                name: "test".into(),
                architecture: ModelArchitecture::Transformer,
                context_tokens: 4096,
                tokenizer_signature: "tok".into(),
                footprint_mb,
            },
            caps: ProviderCaps::hawking_local_shell_today(),
            default_sampler: SamplerProfile::deterministic_edit(),
            endpoint: None,
            cost: None,
            escalates_to: None,
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn admits_when_budget_is_ample() {
        let s = Scheduler::default();
        assert!(s.admit(&role(4_600), &ResourceSnapshot::default()).is_admit());
    }

    #[test]
    fn defers_on_insufficient_ram() {
        let s = Scheduler::default();
        let snap = ResourceSnapshot {
            ram_free_mb: 2_000,
            ..ResourceSnapshot::default()
        };
        match s.admit(&role(4_600), &snap) {
            Admission::Defer { reason, .. } => assert_eq!(reason, DeferReason::Ram),
            _ => panic!("expected RAM defer"),
        }
    }

    #[test]
    fn shared_process_role_ignores_ram() {
        let s = Scheduler::default();
        let snap = ResourceSnapshot {
            ram_free_mb: 0,
            ..ResourceSnapshot::default()
        };
        assert!(s.admit(&role(0), &snap).is_admit());
    }

    #[test]
    fn defers_heavy_role_when_throttling() {
        let s = Scheduler::default();
        let snap = ResourceSnapshot {
            thermal_headroom: 0.1,
            ..ResourceSnapshot::default()
        };
        match s.admit(&role(5_000), &snap) {
            Admission::Defer { reason, .. } => assert_eq!(reason, DeferReason::Thermal),
            _ => panic!("expected thermal defer"),
        }
        // A light role still admits when hot.
        assert!(s.admit(&role(500), &snap).is_admit());
    }

    #[test]
    fn defers_on_concurrency_cap() {
        let s = Scheduler::default();
        let snap = ResourceSnapshot {
            in_flight: 6,
            ..ResourceSnapshot::default()
        };
        match s.admit(&role(500), &snap) {
            Admission::Defer { reason, .. } => assert_eq!(reason, DeferReason::Concurrency),
            _ => panic!("expected concurrency defer"),
        }
    }

    #[test]
    fn battery_quiet_defers_heavy_for_energy() {
        let s = Scheduler::default();
        let snap = ResourceSnapshot {
            on_battery: true,
            mode: PowerMode::Quiet,
            in_flight: 0,
            ..ResourceSnapshot::default()
        };
        match s.admit(&role(5_000), &snap) {
            Admission::Defer { reason, .. } => assert_eq!(reason, DeferReason::Energy),
            _ => panic!("expected energy defer"),
        }
    }
}
