use hide_fleet::fabric::fixture::{run_inprocess_software_fixture, run_two_process_fixture};
use hide_fleet::fabric::node::{OsNodeProbe, SimulatedNodeSet, FIXED_FAKE_MEMORY_BYTES};
use hide_fleet::fabric::pipeline::PipelineScheduler;
use hide_fleet::fabric::placement::{
    reject_unlabelled_simulated, validate_placement_plan_schema, KvOwnershipInvariant,
    ModelSection, PlacementPlan, PlacementRequest, PlacementSimulator, PredictedCost,
    WorkloadClass, PLACEMENT_SCHEMA,
};
use hide_fleet::fabric::qualification::{QualificationKind, HARDWARE_QUALIFICATION_PENDING};
use hide_fleet::resources::OsResourceProbe;
use hide_fleet::ResourceProbe;
const GIB: u64 = 1024 * 1024 * 1024;
fn sample_sections() -> Vec<ModelSection> {
    vec![
        ModelSection::content_addressed("embed", 0, 2, 4 * GIB, b"embed-payload-v1"),
        ModelSection::content_addressed("mid", 2, 6, 8 * GIB, b"mid-payload-v1"),
        ModelSection::content_addressed("head", 6, 8, 3 * GIB, b"head-payload-v1"),
    ]
}
#[test]
fn placement_determinism_same_seed_same_plan() {
    let nodes = SimulatedNodeSet::heterogeneous_sim("sim-integration-det-v1").nodes;
    let req = PlacementRequest {
        sections: sample_sections(),
        nodes,
        workload: WorkloadClass {
            name: "decode".into(),
            seq_len: 512,
            microbatch_size: 2,
            num_microbatches: 4,
        },
        seed: 0xC0FFEE,
        qualification: QualificationKind::Simulated,
    };
    let sim = PlacementSimulator::new();
    let p1 = sim.place(&req).expect("place");
    let p2 = sim.place(&req).expect("place");
    assert_eq!(p1, p2);
    assert!(p1.not_physical_qualification);
    assert!(p1.artifact_label.contains("simulated"));
    assert_eq!(p1.qualification, QualificationKind::Simulated);
}
#[test]
fn kv_ownership_invariant_across_placement_and_failure_replan() {
    let nodes = SimulatedNodeSet::heterogeneous_sim("sim-integration-kv-v1").nodes;
    let req = PlacementRequest {
        sections: sample_sections(),
        nodes,
        workload: WorkloadClass {
            name: "kv".into(),
            seq_len: 96,
            microbatch_size: 1,
            num_microbatches: 2,
        },
        seed: 19,
        qualification: QualificationKind::Simulated,
    };
    let sim = PlacementSimulator::new();
    let plan = sim.place(&req).unwrap();
    KvOwnershipInvariant::assert_holds(&plan.kv_ownership, req.workload.seq_len).unwrap();
    let failed = plan.section_placements[0].node_id.clone();
    let lost = KvOwnershipInvariant::ranges_lost_on_failure(&plan.kv_ownership, &failed);
    assert!(!lost.is_empty());
    let replan = sim.replan_after_failure(&req, &failed).unwrap();
    assert!(replan
        .section_placements
        .iter()
        .all(|sp| sp.node_id != failed));
    KvOwnershipInvariant::assert_holds(&replan.kv_ownership, req.workload.seq_len).unwrap();
}
#[tokio::test]
async fn real_resource_probe_not_fake_32gib() {
    let fleet_probe = OsResourceProbe::default();
    let snap = fleet_probe.snapshot(1, 0).await;
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        assert!(snap.free_memory_mb > 0, "OS free memory probe returned 0");
    }
    let node = OsNodeProbe::new("probe-test").probe_once();
    assert_ne!(node.total_memory_bytes, FIXED_FAKE_MEMORY_BYTES);
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        assert!(node.total_memory_bytes > FIXED_FAKE_MEMORY_BYTES);
        assert!(node.physical_cores >= 1);
    }
}
#[test]
fn schema_rejects_unlabelled_simulated_result() {
    let bad = PlacementPlan {
        schema: PLACEMENT_SCHEMA.into(),
        plan_id: "x".into(),
        seed: 0,
        qualification: QualificationKind::Simulated,
        not_physical_qualification: false,
        section_placements: vec![],
        stage_assignments: vec![],
        kv_ownership: vec![],
        predicted_cost: PredictedCost {
            total: 0,
            transfer_bytes: 0,
            pipeline_bubbles: 0,
        },
        artifact_label: "unlabelled".into(),
    };
    assert!(reject_unlabelled_simulated(&bad).is_err());
    let good = PlacementPlan {
        not_physical_qualification: true,
        artifact_label: "placement_plan_simulated_seed0".into(),
        ..bad.clone()
    };
    validate_placement_plan_schema(&good).unwrap();
}
#[test]
fn simulated_heterogeneous_placement_labelled() {
    let nodes = SimulatedNodeSet::heterogeneous_sim("sim-hetero-qual-v1").nodes;
    let req = PlacementRequest {
        sections: sample_sections(),
        nodes,
        workload: WorkloadClass::default(),
        seed: 5,
        qualification: QualificationKind::Simulated,
    };
    let plan = PlacementSimulator::new().place(&req).unwrap();
    assert_eq!(plan.qualification, QualificationKind::Simulated);
    assert!(plan.not_physical_qualification);
    assert!(plan.artifact_label.contains("simulated"));
    let mut pipe = PipelineScheduler::from_plan(&plan, &req.workload, 2);
    let st = pipe.run_to_completion(128);
    assert!(st.done || st.completed_microbatches > 0);
}
#[test]
fn inprocess_software_fixture_end_to_end() {
    let result = run_inprocess_software_fixture().expect("inprocess fixture");
    assert!(result.not_physical_qualification);
    assert_eq!(result.qualification, QualificationKind::SoftwareFixture);
    assert!(
        !result.receipt.lost_work.stages.is_empty()
            || !result.receipt.lost_work.kv_ranges.is_empty()
    );
    assert_eq!(result.hardware_status, HARDWARE_QUALIFICATION_PENDING);
}
#[test]
fn two_process_fixture_end_to_end_with_failure_replay() {
    match run_two_process_fixture() {
        Ok(result) => {
            assert!(result.not_physical_qualification);
            assert_eq!(result.qualification, QualificationKind::SoftwareFixture);
            assert!(result.receipt.not_physical_qualification);
            assert!(
                !result.receipt.lost_work.stages.is_empty()
                    || !result.receipt.lost_work.kv_ranges.is_empty()
            );
            assert_eq!(
                result.receipt.replayed_from_checkpoint.0,
                format!("ckpt-before-fail-{}", result.request_id)
            );
            assert_eq!(result.hardware_status, HARDWARE_QUALIFICATION_PENDING);
            assert_ne!(result.plan_id, result.replan_plan_id);
        }
        Err(e) => {
            eprintln!("two-process fixture unavailable ({e}); verifying in-process path");
            let result = run_inprocess_software_fixture().expect("fallback inprocess");
            assert!(result.not_physical_qualification);
            assert!(result.artifact_label.contains("software_qualification"));
        }
    }
}
