import json
from pathlib import Path

from tools.accelerator.scoreboard import build_scoreboard, normalize_receipt


def test_normalize_preserves_unmeasured_complete_work_as_none(tmp_path: Path):
    receipt = tmp_path / "fast.json"
    receipt.write_text(json.dumps({
        "schema": "example.v1",
        "status": "PASSED",
        "elapsed_wall_ns": 123,
        "execution": {"gpu_ns": 7, "dispatches": 2},
        "claim_boundary": "not a complete token",
    }))
    row = normalize_receipt(receipt, json.loads(receipt.read_text()), root=tmp_path)
    assert row["wall_ns"] == 123
    assert row["GPU_ns"] == 7
    assert row["complete_token_ns"] is None
    assert row["source_bytes_touched"] is None
    assert row["benchmark_class"] == "UNKNOWN"


def test_normalize_collects_nested_layer_costs_without_relabeling_them_complete(tmp_path: Path):
    receipt = tmp_path / "group.json"
    receipt.write_text(json.dumps({
        "schema": "group.v1",
        "elapsed_wall_ns": 200,
        "execution": {"device": "Apple M3 Ultra", "provider": "apple_metal"},
        "layers": [{
            "gpu_ns": [11, 13], "dispatches": 4, "command_buffers": 1,
            "source_bytes_read": 99,
        }],
    }))
    row = normalize_receipt(receipt, json.loads(receipt.read_text()), root=tmp_path)
    assert row["GPU_ns"] == 24
    assert row["dispatches"] == 4
    assert row["command_buffers"] == 1
    assert row["source_bytes_touched"] == 99
    assert row["complete_token_ns"] is None


def test_normalize_preserves_zero_cost_counters(tmp_path: Path):
    receipt = tmp_path / "resident.json"
    receipt.write_text(json.dumps({
        "schema": "resident.v1",
        "execution": {
            "host_activation_roundtrips": 0,
            "total_command_buffers": 0,
            "total_dispatches": 0,
            "synchronization_count": 0,
        },
    }))
    row = normalize_receipt(receipt, json.loads(receipt.read_text()), root=tmp_path)
    assert row["host_roundtrips"] == 0
    assert row["command_buffers"] == 0
    assert row["dispatches"] == 0
    assert row["synchronization_events"] == 0


def test_normalize_exposes_timing_denominator_without_inference(tmp_path: Path):
    receipt = tmp_path / "timed.json"
    receipt.write_text(json.dumps({
        "schema": "timed.v1",
        "timing": {
            "source_load_ns": 900,
            "device_prepare_ns": 40,
            "command_wait_ns": 7,
            "parity_ns": 3,
        },
        "layers": [{"graph_setup_ns": 5, "gpu_ns": [11], "parity_ns": 2}],
    }))
    row = normalize_receipt(receipt, json.loads(receipt.read_text()), root=tmp_path)
    assert row["timing_phases"] == {
        "root_canonicalize_ns": None,
        "manifest_ns": None,
        "config_ns": None,
        "index_context_ns": None,
        "input_load_ns": None,
        "state_read_ns": None,
        "cpu_readout_ns": None,
        "device_upload_ns": None,
        "source_load_ns": 900,
        "device_prepare_ns": 40,
        "graph_setup_ns": 5,
        "encode_ns": None,
        "command_submit_ns": None,
        "command_wait_ns": 7,
        "gpu_execution_ns": 11,
        "parity_ns": 3,
        "state_write_ns": None,
        "state_reload_ns": None,
        "receipt_write_ns": None,
        "experiment_turnaround_ns": None,
    }


def test_normalize_reads_terminal_timing_nested_under_execution(tmp_path: Path):
    receipt = tmp_path / "terminal.json"
    receipt.write_text(json.dumps({
        "schema": "terminal.v1",
        "execution": {
            "wall_ns": 200,
            "timing": {
                "state_read_ns": 4,
                "cpu_readout_ns": 5,
                "device_upload_ns": 6,
                "encode_wall_ns": 7,
                "command_submit_ns": 8,
                "command_wait_ns": 9,
                "gpu_execution_ns": 10,
                "parity_ns": 11,
            },
        },
    }))
    row = normalize_receipt(receipt, json.loads(receipt.read_text()), root=tmp_path)
    assert row["timing_phases"]["state_read_ns"] == 4
    assert row["timing_phases"]["cpu_readout_ns"] == 5
    assert row["timing_phases"]["device_upload_ns"] == 6
    assert row["timing_phases"]["encode_ns"] == 7
    assert row["timing_phases"]["command_submit_ns"] == 8
    assert row["timing_phases"]["command_wait_ns"] == 9
    assert row["timing_phases"]["gpu_execution_ns"] == 10
    assert row["timing_phases"]["parity_ns"] == 11


def test_scoreboard_uses_receipt_identity_and_keeps_diagnostic_rows_ineligible(tmp_path: Path):
    fast = tmp_path / "fast.json"
    fast.write_text(json.dumps({
        "schema": "fast.v1", "status": "PASSED", "elapsed_wall_ns": 100,
        "execution": {"gpu_ns": 4, "dispatches": 8},
        "model": "flash", "representation": "source_bf16",
        "backend": "apple_metal", "benchmark_class": "DIAGNOSTIC_RELATIVE",
        "capability_verified": True,
    }))
    protected = tmp_path / "protected.json"
    protected.write_text(json.dumps({
        "schema": "token.v1", "status": "PASSED", "complete_token_ns": 120,
        "model": "qwen27", "representation": "q4", "backend": "apple_metal",
        "benchmark_class": "PROTECTED_ABSOLUTE", "capability_verified": True,
    }))
    body = build_scoreboard([fast, protected], root=tmp_path)
    assert len(body["rows"]) == 2
    assert all(len(row["receipt_sha256"]) == 64 for row in body["rows"])
    assert body["physical_plan_score"]["winner"] == "protected.json"
    assert body["physical_plan_score"]["promotion_allowed"] is True
    assert body["claim_boundary"].startswith("Derived from source receipts")


def test_scoreboard_does_not_rank_nominal_utilization_when_work_is_missing(tmp_path: Path):
    high_util = tmp_path / "high.json"
    high_util.write_text(json.dumps({
        "schema": "x.v1", "nominal_utilization": 0.99,
        "benchmark_class": "PROTECTED_ABSOLUTE", "capability_verified": True,
    }))
    low_util = tmp_path / "low.json"
    low_util.write_text(json.dumps({
        "schema": "x.v1", "complete_token_ns": 50,
        "nominal_utilization": 0.01,
        "benchmark_class": "PROTECTED_ABSOLUTE", "capability_verified": True,
    }))
    body = build_scoreboard([high_util, low_util], root=tmp_path)
    assert body["physical_plan_score"]["winner"] == "low.json"
    assert any("complete_useful_work_unmeasured" in row["ineligibility_reasons"]
               for row in body["physical_plan_score"]["candidates"])


def test_normalize_reads_protected_hcli_aggregate_without_calling_it_a_new_run(tmp_path: Path):
    receipt = tmp_path / "protected.json"
    receipt.write_text(json.dumps({
        "schema": "protected.v1",
        "status": "PASSED",
        "benchmark_class": "QUALIFIED_PROTECTED",
        "resident_final": {
            "model_id": "qwen3.8-27b",
            "backend": "apple_metal",
            "binary_sha256_16": "abc123",
            "hot_bytes": 1024,
            "physical_ebpw": 3.2,
            "representation": {"kind": "native-packed"},
        },
        "complete_system_ebpw": 1.2,
        "aggregate": {
            "complete_wall_ns": {"median": 600},
            "complete_wall_ns_per_token": {"median": 100},
            "gpu_ns": {"median": 500},
            "gpu_ns_per_token": {"median": 80},
            "dispatches": {"median": 60},
            "dispatches_per_token": {"median": 10},
            "active_weight_bytes_per_generated_token": {"median": 256},
            "resident_weight_bytes": {"median": 1024},
            "workspace_resident_bytes": {"median": 128},
            "accepted_tps": {"median": 8.0},
        },
        "bytes": {
            "total_nx_bytes": 2048,
            "actual_read_bytes_per_token": 512,
            "transient_bytes_per_token": 128,
        },
        "measurements": [
            {"phase": "warmup", "complete_wall_ns_per_token": 900, "fallbacks": 0},
            {"phase": "measure", "complete_wall_ns_per_token": 110, "gpu_ns_per_token": 90, "fallbacks": 0},
            {"phase": "measure", "complete_wall_ns_per_token": 90, "gpu_ns_per_token": 70, "fallbacks": 0},
        ],
        "capability_verified": True,
    }))

    row = normalize_receipt(receipt, json.loads(receipt.read_text()), root=tmp_path)

    assert row["model"] == "qwen3.8-27b"
    assert row["backend"] == "apple_metal"
    assert row["representation"] == "native-packed"
    assert row["executable_id"] == "abc123"
    assert row["complete_ebpw"] == 1.2
    assert row["physical_ebpw"] == 3.2
    assert row["accepted_tps"] == 8.0
    assert row["resident_bytes"] == 1024
    assert row["active_weight_bytes_per_generated_token"] == 256
    assert row["resident_weight_bytes"] == 1024
    assert row["workspace_resident_bytes"] == 128
    assert row["total_nx_bytes"] == 2048
    assert row["actual_read_bytes_per_token"] == 512
    assert row["transient_bytes_per_token"] == 128
    assert row["complete_token_ns"] == 100
    assert row["wall_ns_per_token"] == 100
    assert row["GPU_ns"] == 500
    assert row["gpu_ns_per_token"] == 80
    assert row["dispatches"] == 60
    assert row["dispatches_per_token"] == 10
    assert row["fallback_count"] == 0
    assert row["wall_minus_gpu_ns_per_token"] is None
    accounting = row["latency_accounting"]
    assert accounting["scope"] == "per_token"
    assert accounting["denominator_metric"] == "wall_ns_per_token"
    assert accounting["denominator_ns"] == 100
    assert accounting["accounting_status"] == "ACCOUNTED"
    assert accounting["dominant_component"] == "gpu_execution"
    assert accounting["dominant_component_ns"] == 80
    assert accounting["dominant_component_fraction"] == 0.8
    assert accounting["components"][-1]["ns"] == 20
    assert accounting["components"][-1]["source"] == "derived_wall_minus_gpu"


def test_normalize_reads_session_totals_and_segments_but_does_not_infer_token_time(tmp_path: Path):
    receipt = tmp_path / "session.json"
    receipt.write_text(json.dumps({
        "schema": "session.v1",
        "totals": {
            "elapsed_wall_ns": 1000,
            "measured_gpu_ns": 600,
            "source_payload_bytes_read": 4000,
            "dispatches": 12,
            "command_buffers": 3,
        },
        "segments": [
            {"segment": 0, "source_payload_bytes_read": 1500, "gpu_ns": 250, "wall_ns": 400},
            {"segment": 1, "source_payload_bytes_read": 2500, "gpu_ns": 350, "wall_ns": 600},
        ],
        "execution": {"oracle_wall_ns": 999999},
    }))

    row = normalize_receipt(receipt, json.loads(receipt.read_text()), root=tmp_path)

    assert row["wall_ns"] == 1000
    assert row["GPU_ns"] == 600
    assert row["source_bytes_touched"] == 4000
    assert row["dispatches"] == 12
    assert row["command_buffers"] == 3
    assert row["timing_phases"]["gpu_execution_ns"] == 600
    assert row["complete_token_ns"] is None
    accounting = row["latency_accounting"]
    assert accounting["scope"] == "run"
    assert accounting["denominator_metric"] == "wall_ns"
    assert accounting["denominator_ns"] == 1000
    assert accounting["dominant_component"] == "gpu_execution"
    assert accounting["dominant_component_ns"] == 600
    assert accounting["components"][-1]["ns"] == 400
    assert accounting["components"][-1]["source"] == "derived_wall_minus_gpu"


def test_latency_accounting_does_not_mix_token_and_run_scopes(tmp_path: Path):
    receipt = tmp_path / "mixed-scope.json"
    receipt.write_text(json.dumps({
        "schema": "mixed.v1",
        "wall_ns_per_token": 100,
        "execution": {"gpu_ns": 80},
    }))

    row = normalize_receipt(receipt, json.loads(receipt.read_text()), root=tmp_path)

    accounting = row["latency_accounting"]
    assert row["gpu_ns_per_token"] is None
    assert accounting["scope"] == "per_token"
    assert accounting["accounting_status"] == "INCOMPLETE"
    assert accounting["dominant_component"] is None
    assert accounting["components"] == []


def test_latency_accounting_marks_negative_residual_inconsistent(tmp_path: Path):
    receipt = tmp_path / "inconsistent.json"
    receipt.write_text(json.dumps({
        "schema": "inconsistent.v1",
        "wall_ns_per_token": 100,
        "gpu_ns_per_token": 120,
    }))

    row = normalize_receipt(receipt, json.loads(receipt.read_text()), root=tmp_path)

    accounting = row["latency_accounting"]
    assert accounting["accounting_status"] == "INCONSISTENT"
    assert accounting["dominant_component"] == "gpu_execution"
    assert accounting["dominant_component_fraction"] == 1.2


def test_latency_accounting_does_not_use_narrower_reported_residual(tmp_path: Path):
    receipt = tmp_path / "narrower-residual.json"
    receipt.write_text(json.dumps({
        "schema": "narrower.v1",
        "wall_ns_per_token": 100,
        "gpu_ns_per_token": 80,
        "wall_minus_gpu_ns_per_token": 50,
    }))

    row = normalize_receipt(receipt, json.loads(receipt.read_text()), root=tmp_path)

    accounting = row["latency_accounting"]
    assert accounting["components"][-1]["ns"] == 20
    assert accounting["components"][-1]["source"] == "derived_wall_minus_gpu"
    assert accounting["reported_residual_ns"] == 50
    assert accounting["reported_residual_consistency"] == "mismatch"
    assert accounting["accounting_status"] == "INCONSISTENT"


def test_build_scoreboard_surfaces_existing_protected_and_session_receipt_shapes():
    root = Path(__file__).resolve().parents[2]
    protected = root / "receipts/headless/HCLI_PROTECTED_ACCELERATOR_BENCHMARK_AFTER_FLASH.json"
    session = root / "receipts/headless/FLASH_STATEFUL_COMPLETE_SESSION_TIMING.json"
    if not protected.is_file() or not session.is_file():
        return

    body = build_scoreboard([protected, session], root=root)
    rows = {row["receipt"]: row for row in body["rows"]}

    protected_row = rows["receipts/headless/HCLI_PROTECTED_ACCELERATOR_BENCHMARK_AFTER_FLASH.json"]
    assert protected_row["model"] == "qwen3.8-27b-sealed-3.14"
    assert protected_row["complete_token_ns"] is not None
    assert protected_row["GPU_ns"] is not None
    assert protected_row["dispatches"] is not None

    session_row = rows["receipts/headless/FLASH_STATEFUL_COMPLETE_SESSION_TIMING.json"]
    assert session_row["source_bytes_touched"] is not None
    assert session_row["complete_token_ns"] is None


def test_normalize_surfaces_diagnostic_arm_costs_without_promoting_them(tmp_path: Path):
    receipt = tmp_path / "diagnostic-ab.json"
    receipt.write_text(json.dumps({
        "schema": "diagnostic.v1",
        "benchmark_class": "DIAGNOSTIC_CONTAMINATED",
        "arms": [
            {"live": {"decode_metrics": {"wall_ns": 300, "gpu_ns": 260, "dispatches": 30}}},
            {"live": {"decode_metrics": {"wall_ns": 500, "gpu_ns": 460, "dispatches": 50}}},
        ],
        "execution": {"oracle_wall_ns": 999999},
    }))

    row = normalize_receipt(receipt, json.loads(receipt.read_text()), root=tmp_path)

    assert row["wall_ns"] == 400
    assert row["GPU_ns"] == 360
    assert row["dispatches"] == 40
    assert row["complete_token_ns"] is None
    assert row["evidence_mode"] == "DIAGNOSTIC_CONTAMINATED"


def test_scoreboard_derives_development_productivity_only_from_explicit_fields(tmp_path: Path):
    qualified = tmp_path / "qualified.json"
    qualified.write_text(json.dumps({
        "schema": "qualified.v1",
        "benchmark_class": "QUALIFIED_PROTECTED",
        "complete_token_ns": 100,
        "capability_verified": True,
        "fallback_count": 0,
        "transform_ns": 10,
        "compile_ns": 20,
        "load_ns": 30,
        "benchmark_ns": 40,
        "verification_ns": 50,
        "receipt_ns": 60,
        "total_experiment_turnaround_ns": 1_000_000_000,
        "experiment_verdict": "ACCEPT",
        "strong_model_turns": 2,
    }))
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({
        "schema": "failed.v1",
        "benchmark_class": "DIAGNOSTIC_RELATIVE",
        "complete_token_ns": 200,
        "capability_verified": True,
        "fallback_count": 0,
        "total_experiment_turnaround_ns": 2_000_000_000,
        "hypothesis_status": "REJECTED",
        "strong_model_turns": 0,
    }))

    body = build_scoreboard([qualified, failed], root=tmp_path)
    row = next(item for item in body["rows"] if item["receipt"] == "qualified.json")
    assert row["development_phases"] == {
        "transform_ns": 10,
        "compile_ns": 20,
        "load_ns": 30,
        "benchmark_ns": 40,
        "verification_ns": 50,
        "receipt_ns": 60,
        "total_experiment_turnaround_ns": 1_000_000_000,
    }
    assert row["hypothesis_outcome"] == "ACCEPT"
    productivity = body["development_productivity"]
    assert productivity["qualified_experiment_count"] == 1
    assert productivity["failed_hypothesis_count"] == 1
    assert productivity["strong_model_turns"] == 2
    assert productivity["total_experiment_turnaround_ns"] == 3_000_000_000
    assert productivity["qualified_experiments_per_hour"] == 1200.0
    assert productivity["failed_hypotheses_per_hour"] == 1200.0
    assert productivity["strong_model_turns_per_qualified_experiment"] == 2.0


def test_scoreboard_does_not_turn_missing_productivity_fields_into_rates(tmp_path: Path):
    receipt = tmp_path / "partial.json"
    receipt.write_text(json.dumps({"schema": "partial.v1", "complete_token_ns": 1}))

    body = build_scoreboard([receipt], root=tmp_path)
    productivity = body["development_productivity"]
    assert productivity["total_experiment_turnaround_ns"] is None
    assert productivity["qualified_experiments_per_hour"] is None
    assert productivity["failed_hypotheses_per_hour"] is None
    assert productivity["strong_model_turns_per_qualified_experiment"] is None
