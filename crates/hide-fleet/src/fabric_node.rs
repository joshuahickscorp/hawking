//! Node discovery and capability reporting for the fabric plane.
//!
//! The fleet governor's [`crate::resources::ResourceProbe`] samples free RAM
//! for *agent-job admission*. This module reports a node's full capability
//! envelope for *distributed placement*: total memory, cores, bandwidth class,
//! and accelerator.
//!
//! Discovery is pluggable:
//! - [`OsNodeProbe`] — real OS reads on the local machine
//! - [`SimulatedNodeSet`] — obviously-named injected set for tests/simulation

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::qualification::QualificationKind;

/// Stable node identity. Opaque string; ABI does not assume co-location.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Coarse interconnect class. Localhost multi-process is **not** LAN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BandwidthClass {
    /// In-process / shared-memory (not used across OS processes).
    InProcess,
    /// Two processes on the same host (this session's only real interconnect).
    Localhost,
    Lan1g,
    Lan10g,
    Wan,
}

/// Accelerator presence. Reporting only — no kernel dispatch here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "class")]
pub enum AcceleratorClass {
    None,
    AppleSiliconGpu { name: String, gpu_cores: u32 },
    Other { name: String },
}

/// Where capability numbers came from. Simulated must be obvious in name + schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source")]
pub enum DiscoverySource {
    OsProbe,
    /// Injected test/sim profile. Name is required and must contain "sim".
    Simulated {
        profile_name: String,
    },
}

impl DiscoverySource {
    pub fn is_simulated(&self) -> bool {
        matches!(self, Self::Simulated { .. })
    }
}

/// Capability envelope a fabric agent advertises.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub node_id: NodeId,
    pub total_memory_bytes: u64,
    pub free_memory_bytes: u64,
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub bandwidth_class: BandwidthClass,
    pub accelerator: AcceleratorClass,
    pub discovery_source: DiscoverySource,
    /// Qualification of this capability report.
    pub qualification: QualificationKind,
    /// Must be true unless `qualification == PhysicalHardware` *and* multi-node
    /// hardware is real. Single-machine OS probes still set this true for
    /// placement plans that use them as a simulated multi-node set.
    pub not_physical_qualification: bool,
}

/// The canned 32 GIB value historically hard-coded into `FixedResourceProbe`
/// via `fleet_run`. Real probes must not equal this as total memory on this
/// 96 GIB M3 Ultra.
pub const FIXED_FAKE_MEMORY_BYTES: u64 = 32 * 1024 * 1024 * 1024;

/// Pluggable discovery: real OS probe or injected simulated node set.
#[async_trait]
pub trait NodeDiscovery: Send + Sync {
    async fn discover(&self) -> Vec<NodeCapabilities>;
}

/// Real local-machine probe. Reports this host only (no network discovery).
#[derive(Debug, Clone)]
pub struct OsNodeProbe {
    pub node_id: NodeId,
    /// Interconnect assumption for this agent process. Default: Localhost.
    pub bandwidth_class: BandwidthClass,
}

impl Default for OsNodeProbe {
    fn default() -> Self {
        Self {
            node_id: NodeId::new("local"),
            bandwidth_class: BandwidthClass::Localhost,
        }
    }
}

impl OsNodeProbe {
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: NodeId::new(node_id),
            bandwidth_class: BandwidthClass::Localhost,
        }
    }

    /// Sample once (sync). Used by the agent heartbeat path.
    pub fn probe_once(&self) -> NodeCapabilities {
        let total = read_total_memory_bytes().unwrap_or(0);
        let free_mb = crate::resources::read_free_memory_mb().unwrap_or(0);
        let free = free_mb.saturating_mul(1024 * 1024);
        let physical = read_physical_cores().unwrap_or(1);
        let logical = read_logical_cores().unwrap_or(physical);
        let accelerator = detect_accelerator();
        NodeCapabilities {
            node_id: self.node_id.clone(),
            total_memory_bytes: total,
            free_memory_bytes: free,
            physical_cores: physical,
            logical_cores: logical,
            bandwidth_class: self.bandwidth_class,
            accelerator,
            discovery_source: DiscoverySource::OsProbe,
            // A single-host OS probe is real hardware *measurement*, but it is
            // not multi-node fabric qualification. Placement that uses only this
            // host remains software-local.
            qualification: QualificationKind::SoftwareFixture,
            not_physical_qualification: true,
        }
    }
}

#[async_trait]
impl NodeDiscovery for OsNodeProbe {
    async fn discover(&self) -> Vec<NodeCapabilities> {
        vec![self.probe_once()]
    }
}

/// Obviously-named simulated node set for tests and placement simulation.
///
/// Profile names must contain `"sim"` (case-insensitive). This is intentional:
/// simulated results must be labelled simulated in their own name.
#[derive(Debug, Clone)]
pub struct SimulatedNodeSet {
    pub profile_name: String,
    pub nodes: Vec<NodeCapabilities>,
}

impl SimulatedNodeSet {
    /// Build a heterogeneous simulated set. Panics if `profile_name` does not
    /// contain `"sim"` — simulated things must say so in their name.
    pub fn heterogeneous_sim(profile_name: impl Into<String>) -> Self {
        let profile_name = profile_name.into();
        assert!(
            profile_name.to_ascii_lowercase().contains("sim"),
            "SimulatedNodeSet profile_name must contain 'sim' (got {profile_name})"
        );
        let nodes = vec![
            simulated_node(
                "sim-node-a",
                &profile_name,
                96 * GIB,
                28,
                BandwidthClass::Lan10g,
                AcceleratorClass::AppleSiliconGpu {
                    name: "sim-m3-ultra".into(),
                    gpu_cores: 60,
                },
            ),
            simulated_node(
                "sim-node-b",
                &profile_name,
                64 * GIB,
                16,
                BandwidthClass::Lan1g,
                AcceleratorClass::AppleSiliconGpu {
                    name: "sim-m2-ultra".into(),
                    gpu_cores: 76,
                },
            ),
            simulated_node(
                "sim-node-c",
                &profile_name,
                32 * GIB,
                12,
                BandwidthClass::Lan1g,
                AcceleratorClass::None,
            ),
        ];
        Self {
            profile_name,
            nodes,
        }
    }

    pub fn homogeneous_pair_sim(profile_name: impl Into<String>, mem: u64, cores: u32) -> Self {
        let profile_name = profile_name.into();
        assert!(
            profile_name.to_ascii_lowercase().contains("sim"),
            "SimulatedNodeSet profile_name must contain 'sim'"
        );
        let nodes = vec![
            simulated_node(
                "sim-homog-0",
                &profile_name,
                mem,
                cores,
                BandwidthClass::Lan10g,
                AcceleratorClass::None,
            ),
            simulated_node(
                "sim-homog-1",
                &profile_name,
                mem,
                cores,
                BandwidthClass::Lan10g,
                AcceleratorClass::None,
            ),
        ];
        Self {
            profile_name,
            nodes,
        }
    }
}

const GIB: u64 = 1024 * 1024 * 1024;

fn simulated_node(
    id: &str,
    profile: &str,
    total: u64,
    cores: u32,
    bw: BandwidthClass,
    accel: AcceleratorClass,
) -> NodeCapabilities {
    NodeCapabilities {
        node_id: NodeId::new(id),
        total_memory_bytes: total,
        free_memory_bytes: total / 2,
        physical_cores: cores,
        logical_cores: cores,
        bandwidth_class: bw,
        accelerator: accel,
        discovery_source: DiscoverySource::Simulated {
            profile_name: profile.to_string(),
        },
        qualification: QualificationKind::Simulated,
        not_physical_qualification: true,
    }
}

#[async_trait]
impl NodeDiscovery for SimulatedNodeSet {
    async fn discover(&self) -> Vec<NodeCapabilities> {
        self.nodes.clone()
    }
}

/// Composite discovery: real local probe plus optional simulated peers.
#[derive(Debug, Clone)]
pub struct CompositeDiscovery {
    pub local: OsNodeProbe,
    pub simulated_peers: Option<SimulatedNodeSet>,
}

#[async_trait]
impl NodeDiscovery for CompositeDiscovery {
    async fn discover(&self) -> Vec<NodeCapabilities> {
        let mut out = self.local.discover().await;
        if let Some(sim) = &self.simulated_peers {
            out.extend(sim.discover().await);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// OS readers (no heavy deps)
// ---------------------------------------------------------------------------

pub fn read_total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        read_sysctl_u64("hw.memsize")
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in text.lines() {
            if let Some(v) = line.strip_prefix("MemTotal:") {
                let kb: u64 = v.split_whitespace().next()?.parse().ok()?;
                return Some(kb.saturating_mul(1024));
            }
        }
        None
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

pub fn read_physical_cores() -> Option<u32> {
    #[cfg(target_os = "macos")]
    {
        read_sysctl_u64("hw.physicalcpu").map(|n| n as u32)
    }
    #[cfg(target_os = "linux")]
    {
        // Count unique physical id / core id pairs if possible; fall back to nproc.
        std::thread::available_parallelism()
            .ok()
            .map(|n| n.get() as u32)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        std::thread::available_parallelism()
            .ok()
            .map(|n| n.get() as u32)
    }
}

pub fn read_logical_cores() -> Option<u32> {
    #[cfg(target_os = "macos")]
    {
        read_sysctl_u64("hw.logicalcpu")
            .or_else(|| read_sysctl_u64("hw.ncpu"))
            .map(|n| n as u32)
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::thread::available_parallelism()
            .ok()
            .map(|n| n.get() as u32)
    }
}

#[cfg(target_os = "macos")]
fn read_sysctl_u64(name: &str) -> Option<u64> {
    use std::process::Command;
    let out = Command::new("sysctl").args(["-n", name]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.trim().parse().ok()
}

fn detect_accelerator() -> AcceleratorClass {
    #[cfg(target_os = "macos")]
    {
        // Prefer a cheap sysctl brand string over spawning system_profiler.
        let brand = read_sysctl_string("machdep.cpu.brand_string").unwrap_or_default();
        if brand.contains("Apple") {
            // GPU core count is best-effort; 0 means unknown.
            let gpu_cores = 0;
            return AcceleratorClass::AppleSiliconGpu {
                name: brand,
                gpu_cores,
            };
        }
        AcceleratorClass::None
    }
    #[cfg(not(target_os = "macos"))]
    {
        AcceleratorClass::None
    }
}

#[cfg(target_os = "macos")]
fn read_sysctl_string(name: &str) -> Option<String> {
    use std::process::Command;
    let out = Command::new("sysctl").args(["-n", name]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Index nodes by id for placement lookups.
pub fn index_by_id(nodes: &[NodeCapabilities]) -> BTreeMap<NodeId, NodeCapabilities> {
    nodes
        .iter()
        .map(|n| (n.node_id.clone(), n.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn os_probe_returns_real_memory_not_fake_32gib() {
        let probe = OsNodeProbe::new("test-local");
        let caps = probe.probe_once();
        assert_eq!(caps.discovery_source, DiscoverySource::OsProbe);
        assert_ne!(caps.total_memory_bytes, FIXED_FAKE_MEMORY_BYTES);
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            assert!(caps.total_memory_bytes > FIXED_FAKE_MEMORY_BYTES);
            assert!(caps.physical_cores >= 1);
            assert!(caps.logical_cores >= caps.physical_cores);
        }
        assert!(caps.not_physical_qualification);
    }
    #[tokio::test]
    async fn simulated_set_is_labelled_simulated() {
        let set = SimulatedNodeSet::heterogeneous_sim("sim-heterogeneous-v1");
        let nodes = set.discover().await;
        assert_eq!(nodes.len(), 3);
        for n in &nodes {
            assert!(n.discovery_source.is_simulated());
            assert_eq!(n.qualification, QualificationKind::Simulated);
            assert!(n.not_physical_qualification);
            assert!(n.node_id.as_str().contains("sim"));
        }
    }
    #[test]
    #[should_panic(expected = "must contain 'sim'")]
    fn simulated_profile_name_must_say_sim() {
        let _ = SimulatedNodeSet::heterogeneous_sim("production-nodes");
    }
}
