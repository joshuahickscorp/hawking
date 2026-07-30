//! Local two-process qualification fixture.
//!
//! Two Fabric Agents as two OS processes on this machine, a real placement
//! across them, a real request, a real injected failure, and a real replay
//! receipt. This qualifies the **software**, not the hardware.
//!
//! Status: `PASSED_SOFTWARE` with `not_physical_qualification: true`.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

use super::agent::fixture_receipt;
use super::failure::FailureReplayReceipt;
use super::node::{BandwidthClass, DiscoverySource, NodeCapabilities, NodeId, OsNodeProbe};
use super::placement::{
    KvOwnershipInvariant, ModelSection, PlacementRequest, PlacementSimulator, WorkloadClass,
};
use super::protocol::{AgentRequest, AgentResponse, PlacementAssignment};
use super::qualification::QualificationKind;

#[derive(Debug)]
pub struct AgentProcess {
    pub node_id: NodeId,
    pub addr: String,
    pub child: Child,
}

impl Drop for AgentProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Result of the two-process software qualification fixture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TwoProcessFixtureResult {
    pub schema: String,
    pub qualification: QualificationKind,
    pub not_physical_qualification: bool,
    pub artifact_label: String,
    pub plan_id: String,
    pub request_id: String,
    pub receipt: FailureReplayReceipt,
    pub replan_plan_id: String,
    pub hardware_status: String,
}

pub const FIXTURE_SCHEMA: &str = "hawking.fabric.two_process_fixture.v1";

/// Resolve the fabric-agent binary path (same target dir as tests).
pub fn fabric_agent_bin() -> PathBuf {
    // CARGO_BIN_EXE_fabric-agent is set when running integration tests for this crate.
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_fabric-agent") {
        return PathBuf::from(p);
    }
    // Fallback: target/{debug,release}/fabric-agent relative to workspace.
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // crates
    path.pop(); // workspace
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    path.push("target");
    path.push(profile);
    path.push("fabric-agent");
    path
}

fn wait_for_port(addr: &str, attempts: u32) -> std::io::Result<()> {
    for i in 0..attempts {
        if TcpStream::connect(addr).is_ok() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(20 + i as u64 * 10));
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        format!("port {addr} not ready"),
    ))
}

/// Spawn one fabric-agent process listening on `addr`.
pub fn spawn_agent(bin: &PathBuf, node_id: &str, addr: &str) -> std::io::Result<AgentProcess> {
    let child = Command::new(bin)
        .args(["serve", "--node-id", node_id, "--listen", addr])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    wait_for_port(addr, 100)?;
    Ok(AgentProcess {
        node_id: NodeId::new(node_id),
        addr: addr.to_string(),
        child,
    })
}

/// One request/response over JSON-lines TCP.
pub fn rpc(addr: &str, req: &AgentRequest) -> Result<AgentResponse, String> {
    let mut stream = TcpStream::connect(addr).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| e.to_string())?;
    let line = serde_json::to_string(req).map_err(|e| e.to_string())?;
    stream
        .write_all(line.as_bytes())
        .map_err(|e| e.to_string())?;
    stream.write_all(b"\n").map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(stream);
    let mut resp_line = String::new();
    reader
        .read_line(&mut resp_line)
        .map_err(|e| e.to_string())?;
    serde_json::from_str(resp_line.trim()).map_err(|e| format!("parse {e}: {resp_line}"))
}

/// Capabilities as seen from two real agent processes on this host.
pub fn probe_two_local_agents(
    a: &AgentProcess,
    b: &AgentProcess,
) -> Result<(NodeCapabilities, NodeCapabilities), String> {
    let ra = rpc(
        &a.addr,
        &AgentRequest::Register {
            capabilities: OsNodeProbe::new(a.node_id.as_str()).probe_once(),
        },
    )?;
    let rb = rpc(
        &b.addr,
        &AgentRequest::Register {
            capabilities: OsNodeProbe::new(b.node_id.as_str()).probe_once(),
        },
    )?;
    let ca = match ra {
        AgentResponse::Registered { capabilities } => capabilities,
        other => return Err(format!("agent a register: {other:?}")),
    };
    let cb = match rb {
        AgentResponse::Registered { capabilities } => capabilities,
        other => return Err(format!("agent b register: {other:?}")),
    };
    Ok((ca, cb))
}

/// Run the full two-process software qualification fixture.
pub fn run_two_process_fixture() -> Result<TwoProcessFixtureResult, String> {
    let bin = fabric_agent_bin();
    if !bin.exists() {
        return Err(format!(
            "fabric-agent binary not found at {} (build with cargo test -p hide-fleet --bins)",
            bin.display()
        ));
    }

    // Ephemeral ports: bind OS-chosen ports via helper agents... we pick high ports
    // that are unlikely busy; if bind fails, agent process exits and wait_for_port fails.
    let addr_a = "127.0.0.1:19701";
    let addr_b = "127.0.0.1:19702";

    let mut agent_a = spawn_agent(&bin, "fixture-node-a", addr_a).map_err(|e| e.to_string())?;
    let mut agent_b = spawn_agent(&bin, "fixture-node-b", addr_b).map_err(|e| e.to_string())?;

    let (mut cap_a, mut cap_b) = probe_two_local_agents(&agent_a, &agent_b)?;
    // Label as software fixture node set (real probes, one machine, two processes).
    cap_a.node_id = NodeId::new("fixture-node-a");
    cap_a.bandwidth_class = BandwidthClass::Localhost;
    cap_a.qualification = QualificationKind::SoftwareFixture;
    cap_a.not_physical_qualification = true;
    // discovery_source stays OsProbe — the *measurement* is real; multi-node fabric is not.
    cap_b.node_id = NodeId::new("fixture-node-b");
    cap_b.bandwidth_class = BandwidthClass::Localhost;
    cap_b.qualification = QualificationKind::SoftwareFixture;
    cap_b.not_physical_qualification = true;

    // Split each process's advertised capacity so placement spreads across both
    // (same host otherwise looks like 2× full RAM).
    let half_a = cap_a.total_memory_bytes / 2;
    let half_b = cap_b.total_memory_bytes / 2;
    cap_a.total_memory_bytes = half_a;
    cap_b.total_memory_bytes = half_b;

    let sections = vec![
        ModelSection::content_addressed("fixture-s0", 0, 2, half_a / 4, b"fixture-s0-v1"),
        ModelSection::content_addressed("fixture-s1", 2, 4, half_b / 4, b"fixture-s1-v1"),
    ];
    let workload = WorkloadClass {
        name: "fixture-request".into(),
        seq_len: 64,
        microbatch_size: 1,
        num_microbatches: 2,
    };
    let req = PlacementRequest {
        sections,
        nodes: vec![cap_a.clone(), cap_b.clone()],
        workload: workload.clone(),
        seed: 42,
        qualification: QualificationKind::SoftwareFixture,
    };
    let sim = PlacementSimulator::new();
    let plan = sim.place(&req).map_err(|e| e.to_string())?;
    KvOwnershipInvariant::assert_holds(&plan.kv_ownership, workload.seq_len)
        .map_err(|e| e.to_string())?;

    // Assign full plan to each agent; each keeps only its sections.
    for (agent, caps) in [(&agent_a, &cap_a), (&agent_b, &cap_b)] {
        let resp = rpc(
            &agent.addr,
            &AgentRequest::Assign {
                assignment: PlacementAssignment {
                    plan_id: plan.plan_id.clone(),
                    plan: plan.clone(),
                    assigned_node: caps.node_id.clone(),
                },
            },
        )?;
        match resp {
            AgentResponse::AssignmentAccepted { .. } => {}
            other => return Err(format!("assign {}: {other:?}", caps.node_id)),
        }
        let _ = rpc(
            &agent.addr,
            &AgentRequest::Heartbeat {
                node_id: caps.node_id.clone(),
                seq: 1,
            },
        )?;
    }

    let request_id = "fixture-req-1".to_string();
    // Real request against both agents.
    for agent in [&agent_a, &agent_b] {
        let resp = rpc(
            &agent.addr,
            &AgentRequest::RunRequest {
                request_id: request_id.clone(),
                plan_id: plan.plan_id.clone(),
            },
        )?;
        match resp {
            AgentResponse::RequestProgress { .. } => {}
            other => return Err(format!("run on {}: {other:?}", agent.node_id)),
        }
    }

    // Fail a node that actually owns stages so the receipt names real lost work.
    let failed = plan
        .stage_assignments
        .first()
        .map(|s| s.node_id.clone())
        .ok_or_else(|| "placement produced no stages".to_string())?;
    let (fail_agent, survivor) = if failed.as_str() == "fixture-node-a" {
        (&mut agent_a, &agent_b)
    } else {
        (&mut agent_b, &agent_a)
    };

    let fail_resp = rpc(
        &fail_agent.addr,
        &AgentRequest::InjectFailure {
            node_id: failed.clone(),
        },
    )?;
    match fail_resp {
        AgentResponse::PlaceholderEvent { .. } => {}
        other => return Err(format!("inject failure: {other:?}")),
    }

    let status = rpc(
        &fail_agent.addr,
        &AgentRequest::GetStatus {
            node_id: failed.clone(),
        },
    )?;
    match status {
        AgentResponse::Status { alive: false, .. } => {}
        other => return Err(format!("expected dead status, got {other:?}")),
    }

    let run_dead = rpc(
        &fail_agent.addr,
        &AgentRequest::RunRequest {
            request_id: request_id.clone(),
            plan_id: plan.plan_id.clone(),
        },
    )?;
    match run_dead {
        AgentResponse::Failed { .. } => {}
        other => return Err(format!("expected Failed after death, got {other:?}")),
    }

    let replan = sim
        .replan_after_failure(&req, &failed)
        .map_err(|e| e.to_string())?;
    KvOwnershipInvariant::assert_holds(&replan.kv_ownership, workload.seq_len)
        .map_err(|e| e.to_string())?;

    // Replay on survivor (and any remaining live node).
    let resp = rpc(
        &survivor.addr,
        &AgentRequest::Assign {
            assignment: PlacementAssignment {
                plan_id: replan.plan_id.clone(),
                plan: replan.clone(),
                assigned_node: survivor.node_id.clone(),
            },
        },
    )?;
    match resp {
        AgentResponse::AssignmentAccepted { .. } => {}
        other => return Err(format!("replan assign: {other:?}")),
    }
    let replay = rpc(
        &survivor.addr,
        &AgentRequest::RunRequest {
            request_id: format!("{request_id}-replay"),
            plan_id: replan.plan_id.clone(),
        },
    )?;
    match replay {
        AgentResponse::RequestProgress { .. } => {}
        other => return Err(format!("replay run: {other:?}")),
    }

    let receipt = fixture_receipt(&request_id, &failed, &plan, &replan, 1);
    if !receipt.not_physical_qualification {
        return Err("receipt must set not_physical_qualification".into());
    }
    if receipt.lost_work.stages.is_empty() && receipt.lost_work.kv_ranges.is_empty() {
        return Err("receipt must name lost work".into());
    }

    // Cleanup processes.
    let _ = rpc(&survivor.addr, &AgentRequest::Shutdown);
    let _ = agent_a.child.kill();
    let _ = agent_b.child.kill();

    Ok(TwoProcessFixtureResult {
        schema: FIXTURE_SCHEMA.into(),
        qualification: QualificationKind::SoftwareFixture,
        not_physical_qualification: true,
        artifact_label: "two_process_fixture_software_qualification".into(),
        plan_id: plan.plan_id,
        request_id,
        receipt,
        replan_plan_id: replan.plan_id,
        hardware_status: super::qualification::HARDWARE_QUALIFICATION_PENDING.into(),
    })
}

/// In-process software fixture (no OS child processes). Used when the binary
/// is unavailable; the integration test prefers the real two-process path.
pub fn run_inprocess_software_fixture() -> Result<TwoProcessFixtureResult, String> {
    use super::agent::{AgentConfig, FabricAgent};

    let probe = OsNodeProbe::new("local");
    let base = probe.probe_once();
    let mut cap_a = base.clone();
    cap_a.node_id = NodeId::new("fixture-node-a");
    cap_a.total_memory_bytes = base.total_memory_bytes / 2;
    cap_a.bandwidth_class = BandwidthClass::Localhost;
    cap_a.qualification = QualificationKind::SoftwareFixture;
    cap_a.not_physical_qualification = true;
    cap_a.discovery_source = DiscoverySource::OsProbe;

    let mut cap_b = cap_a.clone();
    cap_b.node_id = NodeId::new("fixture-node-b");

    let agent_a = FabricAgent::new(AgentConfig::new("fixture-node-a", "127.0.0.1:0"));
    let agent_b = FabricAgent::new(AgentConfig::new("fixture-node-b", "127.0.0.1:0"));

    let sections = vec![
        ModelSection::content_addressed("ip-s0", 0, 2, cap_a.total_memory_bytes / 4, b"ip-s0"),
        ModelSection::content_addressed("ip-s1", 2, 4, cap_b.total_memory_bytes / 4, b"ip-s1"),
    ];
    let workload = WorkloadClass {
        name: "inprocess-fixture".into(),
        seq_len: 32,
        microbatch_size: 1,
        num_microbatches: 2,
    };
    let req = PlacementRequest {
        sections,
        nodes: vec![cap_a, cap_b],
        workload: workload.clone(),
        seed: 7,
        qualification: QualificationKind::SoftwareFixture,
    };
    let sim = PlacementSimulator::new();
    let plan = sim.place(&req).map_err(|e| e.to_string())?;

    for (agent, nid) in [(&agent_a, "fixture-node-a"), (&agent_b, "fixture-node-b")] {
        agent.handle(AgentRequest::Assign {
            assignment: PlacementAssignment {
                plan_id: plan.plan_id.clone(),
                plan: plan.clone(),
                assigned_node: NodeId::new(nid),
            },
        });
        agent.handle(AgentRequest::RunRequest {
            request_id: "ip-req-1".into(),
            plan_id: plan.plan_id.clone(),
        });
    }
    agent_b.handle(AgentRequest::InjectFailure {
        node_id: NodeId::new("fixture-node-b"),
    });
    match agent_b.handle(AgentRequest::RunRequest {
        request_id: "ip-req-1".into(),
        plan_id: plan.plan_id.clone(),
    }) {
        AgentResponse::Failed { .. } => {}
        other => return Err(format!("expected fail, got {other:?}")),
    }
    let failed = NodeId::new("fixture-node-b");
    let replan = sim
        .replan_after_failure(&req, &failed)
        .map_err(|e| e.to_string())?;
    KvOwnershipInvariant::assert_holds(&replan.kv_ownership, workload.seq_len)
        .map_err(|e| e.to_string())?;
    agent_a.handle(AgentRequest::Assign {
        assignment: PlacementAssignment {
            plan_id: replan.plan_id.clone(),
            plan: replan.clone(),
            assigned_node: NodeId::new("fixture-node-a"),
        },
    });
    agent_a.handle(AgentRequest::RunRequest {
        request_id: "ip-req-1-replay".into(),
        plan_id: replan.plan_id.clone(),
    });
    let receipt = fixture_receipt("ip-req-1", &failed, &plan, &replan, 1);
    Ok(TwoProcessFixtureResult {
        schema: FIXTURE_SCHEMA.into(),
        qualification: QualificationKind::SoftwareFixture,
        not_physical_qualification: true,
        artifact_label: "two_process_fixture_software_qualification_inprocess".into(),
        plan_id: plan.plan_id,
        request_id: "ip-req-1".into(),
        receipt,
        replan_plan_id: replan.plan_id,
        hardware_status: super::qualification::HARDWARE_QUALIFICATION_PENDING.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inprocess_fixture_produces_labelled_receipt() {
        let result = run_inprocess_software_fixture().expect("fixture");
        assert!(result.not_physical_qualification);
        assert_eq!(result.qualification, QualificationKind::SoftwareFixture);
        assert!(result.receipt.not_physical_qualification);
        assert!(
            !result.receipt.lost_work.kv_ranges.is_empty()
                || !result.receipt.lost_work.stages.is_empty()
        );
        assert_eq!(
            result.hardware_status,
            super::super::qualification::HARDWARE_QUALIFICATION_PENDING
        );
    }
}
