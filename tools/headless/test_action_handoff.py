"""HAWKING_ACTION_HANDOFF: resume from disk; every field MEASURED or ABSENT.

pytest tools/headless -q must exit 0. The generator copies numbers from the
receipts that own them. These tests never pin a campaign figure as a literal.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from action_handoff import (  # noqa: E402
    ABSENT,
    MEASURED,
    RECEIPT,
    REQUIRED_KEYS,
    SCHEMA,
    assemble_and_write,
    build,
    canonical_dumps,
    parse_spec_peak_gb_s,
)

PARENT = REPO / "receipts" / "headless" / "NOETIC_PARENT_A.json"
CONTROL = REPO / "receipts" / "headless" / "CONVENTIONAL_CONTROL_SET.json"
GPU_LEDGER = REPO / "receipts" / "headless" / "GPU_LEDGER.json"
ROOF = REPO / "receipts" / "headless" / "BANDWIDTH_ROOF.json"
POLICY = REPO / "docs" / "ultragoals" / "ARTIFACT_STORAGE_POLICY.md"
GIT_LEDGER = REPO / "receipts" / "headless" / "GIT_STORAGE_LEDGER.json"
NEGSCI = REPO / "receipts" / "headless" / "NOETIC_NEGATIVE_SCIENCE.json"
GENESIS_REL = "receipts/ascent-2026-08-18/Genesis.m3ultra.nx"


@pytest.fixture(scope="session")
def handoff() -> dict:
    doc = assemble_and_write()
    assert RECEIPT.is_file(), f"missing {RECEIPT}"
    disk = json.loads(RECEIPT.read_text())
    assert disk == doc
    return doc


def _qty_ok(key: str, v: object) -> None:
    assert isinstance(v, dict), f"{key} is not a dict: {type(v)}"
    kind = v.get("kind")
    if kind == ABSENT:
        assert set(v.keys()) == {"kind", "reason"}, (
            f"{key} ABSENT must be exactly {{kind, reason}}, got {sorted(v)}"
        )
        reason = v.get("reason")
        assert isinstance(reason, str) and reason.strip(), f"{key} ABSENT has empty reason"
        return
    if kind == MEASURED:
        assert "value" in v, f"{key} MEASURED missing value"
        assert v.get("source"), f"{key} MEASURED missing source"
        return
    raise AssertionError(f"{key} kind must be MEASURED or ABSENT, got {kind!r}")


def test_schema_and_required_keys_present(handoff: dict):
    assert handoff["schema"] == SCHEMA
    for key in REQUIRED_KEYS:
        assert key in handoff, f"missing required key {key}"


def test_every_required_value_is_measured_or_absent(handoff: dict):
    for key in REQUIRED_KEYS:
        _qty_ok(key, handoff[key])


def test_two_builds_are_identical():
    a = build()
    b = build()
    assert a == b
    assert canonical_dumps(a) == canonical_dumps(b)


def test_write_is_idempotent():
    assemble_and_write()
    first = RECEIPT.read_text()
    assemble_and_write()
    second = RECEIPT.read_text()
    assert first == second
    json.loads(first)  # valid JSON


def test_cli_regenerates_the_same_bytes():
    proc1 = subprocess.run(
        [sys.executable, str(HERE / "action_handoff.py")],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc1.returncode == 0, proc1.stderr
    on_disk_1 = RECEIPT.read_text()
    proc2 = subprocess.run(
        [sys.executable, str(HERE / "action_handoff.py")],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc2.returncode == 0, proc2.stderr
    on_disk_2 = RECEIPT.read_text()
    assert on_disk_1 == on_disk_2
    assert proc1.stdout == proc2.stdout
    assert on_disk_1 == proc1.stdout


def test_git_head_matches_this_worktree(handoff: dict):
    git = handoff["git"]
    assert git["kind"] == MEASURED
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO), text=True
    ).strip()
    assert git["value"]["head"] == head
    assert git["value"]["branch"] == branch
    assert "state" in git["value"]["remote_sync"]


def test_leading_executable_copies_parent_a(handoff: dict):
    lead = handoff["leading_noetic_executable"]
    if not PARENT.is_file():
        assert lead["kind"] == ABSENT
        assert "NOETIC_PARENT_A.json" in lead["reason"]
        return
    parent = json.loads(PARENT.read_text())
    assert lead["kind"] == MEASURED
    v = lead["value"]
    assert v["closure_sha"] == parent["executable_closure"]["closure_sha256"]
    assert v["complete_ebpw"] == parent["RepresentationGenome"]["complete_ebpw"]
    assert v["sealed_path"] == parent["artifact"]["path"]
    assert v["dispatches"] == parent["dispatch_count"]
    if CONTROL.is_file():
        ctrl = json.loads(CONTROL.read_text())
        mlx = v["conventional_mlx_control"]
        archived = (
            ctrl.get("archived", {})
            .get("headline_vs_mlx", {})
            .get("value", {})
            .get("mlx_4bit_tps")
        )
        if archived is not None:
            assert mlx["archived_headline_tps"] == archived


def test_kernels_match_parent_a_and_spec_peak_is_not_forged_into_genesis(
    handoff: dict,
):
    km = handoff["kernels_runtime_machine"]
    assert km["kind"] == MEASURED
    v = km["value"]
    if PARENT.is_file():
        parent = json.loads(PARENT.read_text())
        kern = parent["KernelGenome"]
        pulled = v["kernels_from_parent_a"]
        assert pulled["production_kernel"] == kern["production_kernel"]
        assert pulled["fused_kernels"] == kern["fused_kernels"]
        metal = kern.get("metal_source_hashes") or {}
        exact = metal.get("exact_metal_source_hashes") or {
            k: val
            for k, val in metal.items()
            if isinstance(val, dict) and "sha256" in val
        }
        assert pulled["exact_metal_source_hashes"] == exact
        assert v["runtime_from_parent_a"]["example"] == parent["RuntimeGenome"]["example"]
        assert v["machine_from_parent_a"] == parent["MachineGenome"]
    peak, origin = parse_spec_peak_gb_s()
    if peak is not None:
        assert v["spec_peak_gb_s"] == peak
        assert origin and origin in v["spec_peak_gb_s_provenance"]
        assert GENESIS_REL in v["spec_peak_gb_s_provenance"]
    genesis = v.get("machine_from_genesis")
    if isinstance(genesis, dict) and genesis.get("kind") != ABSENT:
        assert "spec_peak_gb_s" not in genesis


def test_gpu_roof_tracks_bandwidth_roof_receipt(handoff: dict):
    roof = handoff["gpu_roof"]
    if ROOF.is_file():
        assert roof["kind"] == MEASURED
        disk = json.loads(ROOF.read_text())
        assert roof["value"]["body"] == disk
    else:
        assert roof["kind"] == ABSENT
        assert "BANDWIDTH_ROOF.json" in roof["reason"]
        assert set(roof.keys()) == {"kind", "reason"}


def test_bottleneck_copies_gpu_ledger(handoff: dict):
    bot = handoff["current_bottleneck"]
    if not GPU_LEDGER.is_file():
        assert bot["kind"] == ABSENT
        return
    ledger = json.loads(GPU_LEDGER.read_text())
    assert bot["kind"] == MEASURED
    v = bot["value"]
    assert v["verdict"] == ledger["q80_anchor"]["verdict"]
    assert v["reading"] == ledger["q80_anchor"]["reading"]
    assert v["q4_incumbent"] == ledger["q80_anchor"]["q4_incumbent"]
    assert v["ACTIVE_BYTES_PER_TOKEN"] == ledger["ACTIVE_BYTES_PER_TOKEN"]
    assert v["DRAM_READ_BYTES"] == ledger["fields"]["DRAM_READ_BYTES"]


def test_artifact_store_tracks_n019(handoff: dict):
    store = handoff["artifact_store"]
    policy_present = POLICY.is_file()
    ledger_present = GIT_LEDGER.is_file()
    if policy_present or ledger_present:
        assert store["kind"] == MEASURED
    else:
        assert store["kind"] == ABSENT
        assert "N019" in store["reason"]
        assert set(store.keys()) == {"kind", "reason"}


def test_git_storage_policy_points_at_gitignore(handoff: dict):
    pol = handoff["git_storage_policy"]
    assert pol["kind"] == MEASURED
    v = pol["value"]
    assert "gitignore" in v["enacted_pointer"]
    sha = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", ".gitignore"],
        cwd=str(REPO),
        text=True,
    ).strip()
    assert v["last_commit_touching_gitignore"] == sha


def test_next_workunits_are_the_unchecked_obligations(handoff: dict):
    nxt = handoff["next_workunits"]
    led = handoff["ledgers"]
    if led["kind"] == ABSENT:
        assert nxt["kind"] == ABSENT
        return
    assert nxt["kind"] == MEASURED
    pending_ids = [row["id"] for row in nxt["value"]]
    assert pending_ids == led["value"]["ids_pending"]
    assert led["value"]["counts"]["n_total"] == (
        led["value"]["counts"]["n_checked"] + led["value"]["counts"]["n_unchecked"]
    )


def test_active_grok_tasks_are_running_on_disk(handoff: dict):
    tasks = handoff["active_grok_tasks"]
    if tasks["kind"] == ABSENT:
        assert "tasks" in tasks["reason"].lower() or "not on disk" in tasks["reason"]
        return
    assert tasks["kind"] == MEASURED
    running = tasks["value"]["running"]
    assert tasks["value"]["n_running"] == len(running)
    ids = [t["id"] for t in running]
    assert ids == sorted(ids)
    root = Path(tasks["value"]["tasks_root"])
    for t in running:
        st = (root / t["id"] / "status").read_text().strip()
        assert st == "running"
        assert t["status"] == "running"


def test_negative_science_points_at_the_receipt(handoff: dict):
    ns = handoff["negative_science"]
    if not NEGSCI.is_file():
        assert ns["kind"] == ABSENT
        return
    disk = json.loads(NEGSCI.read_text())
    assert ns["kind"] == MEASURED
    assert ns["value"]["counts"] == disk.get("counts")
    assert ns["value"]["schema"] == disk.get("schema")
    assert "NOETIC_NEGATIVE_SCIENCE.json" in ns["value"]["path"]


def test_no_generated_at_clock_in_the_handoff(handoff: dict):
    assert "generated_at" not in handoff
    blob = canonical_dumps(handoff)
    rebuilt = json.loads(blob)
    assert rebuilt == handoff
