"""Tests for the ceremony-free Frankenstein direct fusion harness."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import frankenstein_direct as direct  # noqa: E402
from lab.operators import frankenstein_fusion_op as fusion  # noqa: E402
from lab.receipts import verify  # noqa: E402


DURABLE_BRIDGE = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/child-baseline-v3/DSV4F_LATENT_BRIDGE_CONTRACT.json"
)
DURABLE_TRANSPLANT = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/child-baseline-v3/DSV4F_TRANSPLANT_POINTS.json"
)
DURABLE_BODY = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/full-43-layer-stream.gravity"
)


pytestmark = pytest.mark.skipif(
    not DURABLE_BRIDGE.is_file() or not DURABLE_TRANSPLANT.is_file(),
    reason="durable DSV4F bridge/transplant contracts not present on this host",
)


def _paths(tmp_path: Path) -> direct.HarnessPaths:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Place a tiny free-space anchor; real free space comes from the volume.
    campaign = tmp_path / "campaign"
    return direct.default_paths(
        workspace=workspace,
        out_dir=campaign,
        body_path=DURABLE_BODY,
        bridge_path=DURABLE_BRIDGE,
        transplant_path=DURABLE_TRANSPLANT,
    )


def test_shape_mismatches_forbid_weight_average() -> None:
    mismatches = fusion.shape_mismatches()
    fields = {row["field"] for row in mismatches}
    assert "hidden_size" in fields
    assert "num_hidden_layers" in fields
    hidden = next(row for row in mismatches if row["field"] == "hidden_size")
    assert hidden["values"]["deepseek_v4_flash"] == 4096
    assert hidden["values"]["kimi_k3"] == 7168
    assert hidden["values"]["glm_5_2"] == 6144
    assert hidden["compatible_for_elementwise_mean"] is False

    impossible = fusion.impossible_operations()
    names = {row["name"] for row in impossible}
    assert "stream_portion_and_average_weights" in names
    assert "direct_weight_transplant" in names
    avg = next(row for row in impossible if row["name"] == "stream_portion_and_average_weights")
    assert avg["verdict"] == "IMPOSSIBLE_AS_STATED"


def test_fusion_spec_shapes_and_loss() -> None:
    spec = fusion.fusion_operation_spec()
    assert spec["verdict"] == "REAL_AND_MINIMAL"
    assert spec["name"] == "block_wise_streaming_distillation_via_latent_bridge"
    assert spec["projections"]["kimi_k3"]["weight_shape"] == [7168, 4096]
    assert spec["projections"]["glm_5_2"]["weight_shape"] == [6144, 4096]
    assert spec["residual_adapter"]["weight_shape"] == [4096, 4096]
    loss = fusion.loss_target(transplant_point="post_moe_hidden_state")
    assert loss["forward_gate"] == fusion.FORWARD_GATE
    assert any(t["name"] == "mse_projected_donor" for t in loss["terms"])
    route_loss = fusion.loss_target(transplant_point="selected_expert_ids")
    assert any(t["name"] == "route_kl" for t in route_loss["terms"])


def test_layer_map_endpoints() -> None:
    assert fusion.layer_map(donor="kimi_k3", student_layer=0)["donor_layer"] == 0
    assert fusion.layer_map(donor="kimi_k3", student_layer=42)["donor_layer"] == 92
    assert fusion.layer_map(donor="glm_5_2", student_layer=0)["donor_layer"] == 0
    assert fusion.layer_map(donor="glm_5_2", student_layer=42)["donor_layer"] == 77


def test_disk_budget_preserves_floor() -> None:
    free = 309 * 1024**3
    archive = fusion.estimated_adapter_archive_bytes(
        layers=1, points=("post_moe_hidden_state",)
    )
    budget = direct.compute_disk_budget(
        free_now=free,
        body_budget_bytes=149 * 1024**3,
        donor_window_budget=32 * 1024**3,
        archive_upper_bound=int(archive["archive_upper_bound_bytes"]),
        body_already_on_volume=True,
    )
    assert budget["floor_preserved_with_working_set"] is True
    assert budget["verdict"] == "SAFE_UNDER_WORKING_SET"
    assert budget["working_set"]["total_bytes"] == (
        32 * 1024**3 + direct.DEFAULT_OUTPUT_BLOCK_BUDGET_BYTES + direct.DEFAULT_SCRATCH_BUDGET_BYTES
    )


def test_disk_budget_refuses_when_free_too_low() -> None:
    budget = direct.compute_disk_budget(
        free_now=20 * 1024**3,
        body_budget_bytes=149 * 1024**3,
        donor_window_budget=32 * 1024**3,
        archive_upper_bound=1 * 1024**3,
        body_already_on_volume=True,
    )
    assert budget["floor_preserved_with_working_set"] is False
    assert budget["verdict"] == "UNSAFE_WORKING_SET_WOULD_BREACH_FLOOR"


def test_assert_floor_ok(tmp_path: Path) -> None:
    check = direct.assert_floor(tmp_path, label="tmp")
    assert check["status"] == "FLOOR_OK"
    assert check["free_bytes"] >= direct.MIN_FREE_FLOOR_BYTES


def test_schedule_fixture_mode(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    plan = direct.build_schedule(paths=paths, mode="fixture")
    verify(plan, label="schedule")
    assert plan["schema"] == direct.SCHEDULE_SCHEMA
    assert plan["status"] == "DIRECT_SCHEDULE_FROZEN_CEREMONY_FREE"
    assert plan["ceremony"]["ramanujan_launcher"] is False
    assert plan["block_count"] == 1
    assert plan["disk"]["verdict"] == "SAFE_UNDER_WORKING_SET"
    assert plan["fusion_operation"]["forward_gate"] == fusion.FORWARD_GATE
    assert plan["storage_contract"]["gravity_compress_output"] is False
    assert plan["storage_contract"]["deepseek_body_eviction_authorized"] is False


def test_schedule_pilot_has_two_bridges(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    plan = direct.build_schedule(paths=paths, mode="pilot")
    assert plan["block_count"] == 2  # kimi + glm, one point, layer 0
    donors = {row["donor"] for row in plan["blocks"]}
    assert donors == {"kimi_k3", "glm_5_2"}


def test_first_step_fixture_stream_seal_evict(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    result = direct.run_first_step(paths=paths, mode="fixture")
    assert result["status"] == "DIRECT_HARNESS_FIRST_STEP_SEALED"
    assert result["forward_gate"] == fusion.FORWARD_GATE
    assert result["disk_verdict"] == "SAFE_UNDER_WORKING_SET"
    assert result["first_block"]["eviction_confirmed"] is True
    assert result["first_block"]["bytes_streamed"] > 0
    assert result["first_block"]["fit_status"] == fusion.FORWARD_GATE

    # Receipt sealed and on disk.
    receipt_path = Path(result["first_block"]["receipt_path"])
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verify(receipt, label="block receipt")
    assert receipt["status"] == "BLOCK_SEALED_FIT_PENDING_FORWARD"
    assert receipt["claim_boundary"]["weight_average"] is False
    assert receipt["claim_boundary"]["trained_adapter"] is False
    assert receipt["eviction"]["exact_eviction_confirmed"] is True
    assert receipt["disk"]["floor_preserved"] is True

    # Donor window gone; adapter block present and not gravity-named.
    assert not (paths.working_set_dir / f"donor-{result['first_block']['block_id']}").exists()
    adapter = Path(result["first_block"]["adapter_path"])
    assert adapter.is_file()
    assert adapter.name.endswith(".raw_adapter")
    assert b"FRNKADP1" == adapter.read_bytes()[:8]

    # Fusion spec + schedule sealed.
    for path_key in ("fusion_spec_path", "schedule_path", "run_manifest_path"):
        p = Path(result[path_key])
        assert p.is_file()
        doc = json.loads(p.read_text(encoding="utf-8"))
        verify(doc, label=path_key)


def test_refuses_live_kimi_full_stream(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    schedule = direct.build_schedule(paths=paths, mode="pilot")
    # First pilot block is KIMI_STRATEGIC_BRIDGE.
    with pytest.raises(direct.FrankensteinDirectError, match="Kimi-K3"):
        direct.run_block(paths=paths, schedule=schedule, order=0, fixture=False)


def test_idempotent_schedule_rewrite(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    a = direct.write_schedule(paths=paths, mode="fixture")
    b = direct.write_schedule(paths=paths, mode="fixture")
    assert a["seal_sha256"] == b["seal_sha256"]
    schedule_path = paths.schedule_path.parent / direct.schedule_filename("fixture")
    raw = schedule_path.read_bytes()
    assert json.loads(raw.decode("utf-8"))["seal_sha256"] == a["seal_sha256"]


def test_cli_first_step(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    campaign = tmp_path / "campaign"
    rc = direct.main(
        [
            "--workspace",
            str(workspace),
            "--out-dir",
            str(campaign),
            "--body-path",
            str(DURABLE_BODY),
            "--bridge-path",
            str(DURABLE_BRIDGE),
            "--transplant-path",
            str(DURABLE_TRANSPLANT),
            "first-step",
            "--mode",
            "fixture",
        ]
    )
    assert rc == 0
    frank = campaign / "evidence" / "models" / "frankenstein"
    assert (frank / direct.FUSION_SPEC_NAME).is_file()
    assert (frank / "FRANKENSTEIN_DIRECT_RUN_MANIFEST_FIXTURE.json").is_file()
    assert (frank / direct.schedule_filename("fixture")).is_file()
    assert (frank / "receipts").is_dir()
