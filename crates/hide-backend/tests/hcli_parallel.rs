//! Acceptance battery for HCLI P0.5 (section 15).

use hide_backend::hcli::{
    harvest, ContextBudget, Hcli, HcliDag, HcliNode, LaneRole, LaneOutput, MemGate, NodeId,
    NodeStatus, ResourceClass, Scope, SystemMemGate, TestResult, MemoryPressure,
};
use std::sync::Arc;

fn gate(ceiling: usize) -> Arc<SystemMemGate> {
    let g = Arc::new(SystemMemGate::new(ceiling));
    g.set_per_lane_bytes(1); // generous admission in tests
    g.set_pressure_override(MemoryPressure {
        total_physical_bytes: 16 << 30,
        available_bytes: 16 << 30,
        ..Default::default()
    });
    g
}

fn node(id: &str, role: LaneRole, write: &str, rc: ResourceClass) -> HcliNode {
    HcliNode {
        id: NodeId::new(id),
        parent: None,
        deps: vec![],
        role,
        objective: format!("task {id}"),
        read_scope: Scope::new(),
        write_scope: Scope::new().path(write),
        resource_class: rc,
        context_budget: ContextBudget { max_tokens: 4096 },
        worker_session: None,
        status: NodeStatus::Ready,
        result: None,
        receipt: None,
        acceptance: None,
    }
}

// A. one trivial task -> expected 1 lane.
#[test]
fn a_trivial_task_one_lane() {
    let mut h = Hcli::new(gate(3), 3);
    h.dag.insert(node("t1", LaneRole::Implementer, "src/a.rs", ResourceClass::Cpu));
    let admitted = h.admit_ready();
    assert_eq!(admitted.len(), 1, "a trivial task should admit exactly 1 lane");
}

// C. induced memory pressure -> expected reduced/refused concurrency.
#[test]
fn c_induced_memory_pressure_reduces_concurrency() {
    let g = gate(3);
    let mut h = Hcli::new(g.clone(), 3);
    g.set_pressure_override(MemoryPressure {
        total_physical_bytes: 16 << 30,
        available_bytes: 1 << 30,
        ..Default::default()
    });
    g.set_active_worker_bytes((1 << 30) - 1);

    h.dag.insert(node("t1", LaneRole::Architect, "src/a.rs", ResourceClass::Cpu));
    h.dag.insert(node("t2", LaneRole::Implementer, "src/b.rs", ResourceClass::Cpu));
    h.dag.insert(node("t3", LaneRole::Adversary, "src/c.rs", ResourceClass::Cpu));

    let admitted = h.admit_ready();
    assert!(
        admitted.len() <= 1,
        "pressure should reduce concurrency, got {}",
        admitted.len()
    );
}

// D. conflicting write scopes -> expected serialization.
#[test]
fn d_conflicting_write_scopes_serialize() {
    let mut dag = HcliDag::new();
    dag.insert(node("a", LaneRole::Implementer, "src/x.rs", ResourceClass::Cpu));
    dag.insert(node("b", LaneRole::Implementer, "src/x.rs", ResourceClass::Cpu));
    assert!(
        !dag.can_run_concurrently(&NodeId::new("a"), &NodeId::new("b")),
        "conflicting write scopes must serialize"
    );
}

// E. protected GPU benchmark -> expected exclusive timing lane.
#[test]
fn e_gpu_timing_is_exclusive() {
    let mut dag = HcliDag::new();
    dag.insert(node("bench", LaneRole::Adversary, "bench", ResourceClass::GpuTiming));
    dag.insert(node("infer", LaneRole::Implementer, "infer", ResourceClass::GpuInference));
    assert!(
        !dag.can_run_concurrently(&NodeId::new("bench"), &NodeId::new("infer")),
        "a protected GPU benchmark must not overlap an inference lane"
    );
}

// F. three independent CPU/static tasks -> expected overlap.
#[test]
fn f_three_independent_cpu_tasks_overlap() {
    let mut h = Hcli::new(gate(3), 3);
    h.dag.insert(node("t1", LaneRole::Architect, "src/a.rs", ResourceClass::Cpu));
    h.dag.insert(node("t2", LaneRole::Implementer, "src/b.rs", ResourceClass::Cpu));
    h.dag.insert(node("t3", LaneRole::Adversary, "src/c.rs", ResourceClass::Cpu));
    let admitted = h.admit_ready();
    assert_eq!(
        admitted.len(),
        3,
        "three independent CPU tasks should overlap (healthy machine)"
    );
}

// G. harvest -> expected compact synthesized evidence.
#[test]
fn g_harvest_compact_evidence() {
    let outputs = vec![
        LaneOutput {
            lane: "A".into(),
            role: "Architect".into(),
            text: "use approach X".into(),
            changed_files: ["src/a.rs".into()].into_iter().collect(),
            test_results: vec![TestResult {
                name: "build".into(),
                passed: true,
            }],
            proposed_actions: vec!["refactor a".into()],
        },
        LaneOutput {
            lane: "B".into(),
            role: "Implementer".into(),
            text: "use approach X too".into(),
            changed_files: ["src/a.rs".into()].into_iter().collect(),
            test_results: vec![],
            proposed_actions: vec!["refactor a".into()],
        },
    ];
    let h = harvest(&outputs);
    assert!(
        h.packet.agreements.contains(&"refactor a".to_string()),
        "harvest should detect the agreement"
    );
    assert!(h.packet.changed_files.contains("src/a.rs"));
    assert!(h.packet.test_results.iter().any(|t| t.name == "build" && t.passed));
}

// H. lanes are role-agnostic: same-role independent tasks can overlap.
#[test]
fn h_role_agnostic_lanes_admit_same_role() {
    let mut h = Hcli::new(gate(2), 2);
    h.dag.insert(node("a", LaneRole::Implementer, "src/a.rs", ResourceClass::Cpu));
    h.dag.insert(node("b", LaneRole::Implementer, "src/b.rs", ResourceClass::Cpu));
    let admitted = h.admit_ready();
    assert_eq!(
        admitted.len(),
        2,
        "independent same-role tasks should be able to overlap"
    );
}

// I. harvest detects test contradictions.
#[test]
fn i_harvest_detects_test_contradiction() {
    let outputs = vec![
        LaneOutput {
            lane: "A".into(),
            test_results: vec![TestResult {
                name: "build".into(),
                passed: true,
            }],
            ..Default::default()
        },
        LaneOutput {
            lane: "B".into(),
            test_results: vec![TestResult {
                name: "build".into(),
                passed: false,
            }],
            ..Default::default()
        },
    ];
    let h = harvest(&outputs);
    assert!(
        h.packet
            .contradictions
            .iter()
            .any(|c| c.topic == "test:build"),
        "harvest should detect test contradiction"
    );
    assert!(
        h.packet
            .test_results
            .iter()
            .any(|t| t.name == "build" && !t.passed),
        "conflicting test should be marked failed"
    );
}

// J. unknown MemGate pressure is permissive in bootstrap.
#[test]
fn j_memgate_unknown_pressure_is_permissive() {
    let g = SystemMemGate::new(3);
    g.set_per_lane_bytes(1 << 30);
    g.set_pressure_override(MemoryPressure::default());
    let d = g.admit(3);
    assert_eq!(d.admitted_lanes, 3);
}

// K. prefix write scopes serialize.
#[test]
fn k_scope_prefix_overlap_serializes() {
    let mut dag = HcliDag::new();
    dag.insert(node("a", LaneRole::Implementer, "src", ResourceClass::Cpu));
    dag.insert(node("b", LaneRole::Implementer, "src/a.rs", ResourceClass::Cpu));
    assert!(
        !dag.can_run_concurrently(&NodeId::new("a"), &NodeId::new("b")),
        "prefix write scopes must serialize"
    );
}
