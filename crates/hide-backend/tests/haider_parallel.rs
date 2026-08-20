//! Acceptance battery for HAIDER P0.5 (section 15).

use hide_backend::haider::{
    harvest, ContextBudget, Haider, HaiderDag, HaiderNode, LaneRole, LaneOutput, NodeId,
    NodeStatus, ResourceClass, Scope, SystemMemGate, TestResult,
};
use std::sync::Arc;

fn gate(ceiling: usize) -> Arc<SystemMemGate> {
    let g = Arc::new(SystemMemGate::new(ceiling));
    g.set_per_lane_bytes(1); // generous admission in tests
    g
}

fn node(id: &str, role: LaneRole, write: &str, rc: ResourceClass) -> HaiderNode {
    HaiderNode {
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
    }
}

// A. one trivial task -> expected 1 lane.
#[test]
fn a_trivial_task_one_lane() {
    let mut h = Haider::new(gate(3), 3);
    h.dag.insert(node("t1", LaneRole::Implementer, "src/a.rs", ResourceClass::Cpu));
    let admitted = h.admit_ready();
    assert_eq!(admitted.len(), 1, "a trivial task should admit exactly 1 lane");
}

// D. conflicting write scopes -> expected serialization.
#[test]
fn d_conflicting_write_scopes_serialize() {
    let mut dag = HaiderDag::new();
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
    let mut dag = HaiderDag::new();
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
    let mut h = Haider::new(gate(3), 3);
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
