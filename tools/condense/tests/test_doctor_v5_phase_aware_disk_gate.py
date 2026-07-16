from __future__ import annotations

import copy
import datetime as dt
import hashlib
import inspect
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from unittest import mock

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import doctor_v5_phase_aware_disk_gate as gate
import doctor_v5_remaining_scratch_ledger as ledger


NOW = dt.datetime(2026, 7, 15, 12, 0, tzinfo=dt.timezone.utc)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def _tree(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


class Fixture:
    def __init__(self, root: Path, *, cell_id: str = "qwen-test__4bpw__codec-control",
                 packed_sizes: tuple[int, int] = (5, 7)) -> None:
        self.root = root
        self.cell_id = cell_id
        self.results = root / "reports/condense/doctor_v5_ultra/results"
        self.output = self.results / cell_id
        self.worker = self.output / "strand_ladder"
        self.policy_root = root / (
            "reports/condense/doctor_v5_ultra/staged_acceleration/"
            "controlled_swap_successor/generation-test")
        self.policy_root.mkdir(parents=True)
        self.plan_path = root / "reports/condense/doctor_v5_ultra/campaign_plan.json"
        self.spec_path = root / (
            f"reports/condense/doctor_v5_ultra/runtime_specs/{cell_id}.json")
        self.source = root / "scratch/staging/qwen-test.partial"
        self.source.mkdir(parents=True)
        (self.worker / "bundle/shards").mkdir(parents=True)
        (self.worker / "evaluation/reconstruction").mkdir(parents=True)
        self.identity = "1" * 64
        self.program_sha = "2" * 64
        self.scratch_bytes = 50
        self.frozen_projection = 10
        self.campaign = {
            "branch": "codec_control", "cell_id": cell_id,
            "cell_identity_sha256": self.identity, "label": "14B",
            "target_rate_bpw": 4.0, "target_rate_id": "4",
        }
        self.cell = {
            "adapter_id": "doctor-v5-strand-ladder-qwen25-dense",
            "admission": {
                "disk_reserve_bytes": ledger.DISK_RESERVE_BYTES,
                "recommended_scratch_bytes": self.scratch_bytes,
            },
            "backend": "apple-cpu-strand", "branch": "codec_control",
            "cell_id": cell_id, "cell_identity_sha256": self.identity,
            "command": "condense_control", "model_dir": str(self.source),
            "model_family": "qwen2.5-dense", "model_label": "14B",
            "parameter_manifest": {
                "path": str(root / "reports/parameter_manifest.json"),
                "file_sha256": "3" * 64,
                "source_manifest_sha256": "4" * 64,
            },
            "projected_output_bytes": self.frozen_projection, "rate_id": "4",
            "runtime_spec_path": str(self.spec_path),
            "runtime_spec_schema": "hawking.doctor_v5_strand_ladder_spec.v1",
            "source_census": {
                "path": str(root / "reports/source_census.json"),
                "file_sha256": "5" * 64,
            },
            "source_deletion_permitted": False,
        }
        plan = {"schema": "test-plan", "cells": [self.cell]}
        plan["plan_sha256"] = gate._hash_value(plan)
        self.plan = plan
        _write_json(self.plan_path, plan)

        resources = {
            "disk_reserve_bytes": ledger.DISK_RESERVE_BYTES,
            "scratch_budget_bytes": self.scratch_bytes, "threads": 20,
        }
        self.spec = {
            "schema": self.cell["runtime_spec_schema"],
            "adapter_id": self.cell["adapter_id"], "backend": self.cell["backend"],
            "campaign_binding": self.campaign,
            "codec": {"rate_id": "4", "symbol_bits": 4},
            "doctor_hook": {"method": "none"},
            "evaluation": {"mode": "resident", "retain_dense_reconstruction": False},
            "model_family": self.cell["model_family"],
            "operation": self.cell["command"],
            "program_spec_sha256": self.program_sha,
            "resource_admission_sha256": gate._hash_value(resources),
            "resources": resources, "source_deletion_permitted": False,
        }
        _write_json(self.spec_path, self.spec)

        source_rows = []
        for ordinal, size in enumerate((3, 4)):
            path = self.source / f"model-{ordinal:05d}.safetensors"
            path.write_bytes(bytes([ordinal + 1]) * size)
            source_rows.append({
                "bytes": size, "name": path.name, "ordinal": ordinal,
                "path": str(path), "sha256": _file_sha(path),
            })
        self.packed: list[Path] = []
        self.reconstruction: list[Path] = []
        for ordinal, (packed_size, reconstruction_size) in enumerate(
                zip(packed_sizes, (13, 17), strict=True)):
            packed = self.worker / f"bundle/shards/{ordinal:05d}.strand"
            packed.write_bytes(bytes([10 + ordinal]) * packed_size)
            reconstruction = self.worker / (
                f"evaluation/reconstruction/{ordinal:05d}.safetensors")
            reconstruction.write_bytes(bytes([20 + ordinal]) * reconstruction_size)
            self.packed.append(packed); self.reconstruction.append(reconstruction)

        self.request_path = self.worker / "request.json"
        self.checkpoint_path = self.worker / "checkpoint.json"
        self.request = {
            "schema": ledger.REQUEST_SCHEMA, "request_id": "phase-test",
            "label": "14B", "model_family": self.cell["model_family"],
            "campaign_binding": self.campaign, "codec": self.spec["codec"],
            "source": {
                "census_path": self.cell["source_census"]["path"],
                "census_sha256": self.cell["source_census"]["file_sha256"],
                "model_dir": str(self.source), "shards": source_rows,
                "source_manifest_sha256": self.cell["parameter_manifest"][
                    "source_manifest_sha256"],
            },
            "parameter_manifest": {
                "path": self.cell["parameter_manifest"]["path"],
                "sha256": self.cell["parameter_manifest"]["file_sha256"],
            },
            "execution": {}, "evaluation": self.spec["evaluation"],
            "doctor_hook": self.spec["doctor_hook"], "resources": {
                "disk_reserve_bytes": ledger.DISK_RESERVE_BYTES,
                "scratch_budget_bytes": self.scratch_bytes,
            },
            "output_root": str(self.worker), "evidence_policy": {},
        }
        _write_json(self.request_path, self.request)

        self.manifest_path = self.worker / "bundle/manifest.json"
        self.manifest = {
            "schema": "hawking.doctor_v5_strand_ladder_bundle.v1",
            "campaign_binding": self.campaign,
            "claims": {"packed_archive_roundtrip_validated": True,
                       "source_deletion": False},
            "shards": [{"ordinal": ordinal, "packed": _artifact(path)}
                       for ordinal, path in enumerate(self.packed)],
        }
        _write_json(self.manifest_path, self.manifest)
        plan_units = ["preflight", "metadata"]
        for ordinal in range(2):
            plan_units.extend([
                f"passthrough:{ordinal:05d}", f"encode:{ordinal:05d}",
                f"attest:{ordinal:05d}", f"decode:{ordinal:05d}",
            ])
        plan_units.extend(ledger.RESIDENT_SUFFIX)
        completed = plan_units[:plan_units.index("override_manifest")]
        units: dict[str, dict[str, object]] = {}
        for unit in completed:
            units[unit] = {"completed_at": "2026-07-15T00:00:00+00:00"}
            match = ledger.ORDINAL_UNIT_RE.fullmatch(unit)
            if match:
                phase, raw = match.groups(); ordinal = int(raw)
                if phase == "encode":
                    units[unit]["artifact"] = _artifact(self.packed[ordinal])
                elif phase == "attest":
                    units[unit]["archive"] = _artifact(self.packed[ordinal])
                elif phase == "decode":
                    units[unit]["artifact"] = _artifact(self.reconstruction[ordinal])
        units["bundle_manifest"]["artifact"] = _artifact(self.manifest_path)
        self.checkpoint = {
            "schema": ledger.CHECKPOINT_SCHEMA,
            "request_sha256": _file_sha(self.request_path),
            "created_at": "2026-07-15T00:00:00+00:00",
            "updated_at": "2026-07-15T00:01:00+00:00", "status": "running",
            "plan": plan_units, "completed_units": completed, "units": units,
            "stop_requested": False,
        }
        _write_json(self.checkpoint_path, self.checkpoint)
        self.refresh_binding()

    def refresh_binding(self) -> None:
        self.binding = gate.PhaseGateBinding(
            plan_path=self.plan_path, plan_file_sha256=_file_sha(self.plan_path),
            plan_sha256=self.plan["plan_sha256"],
            plan_cell_sha256=gate._hash_value(self.cell), cell_id=self.cell_id,
            cell_identity_sha256=self.identity, runtime_spec_path=self.spec_path,
            runtime_spec_file_sha256=_file_sha(self.spec_path),
            program_spec_sha256=self.program_sha,
            execution_output_root=self.output, policy_root=self.policy_root,
            disk_reserve_bytes=ledger.DISK_RESERVE_BYTES,
            declared_scratch_bytes=self.scratch_bytes,
            frozen_projected_output_bytes=self.frozen_projection,
            module_sha256=_file_sha(Path(gate.__file__)),
            ledger_module_sha256=_file_sha(Path(ledger.__file__)),
            ram_credit_bytes=0,
        )

    def original_gate(self, *, swap_blocker: bool = True) -> dict[str, object]:
        observed = sum(path.stat().st_size for path in self.packed)
        remaining = max(0, self.frozen_projection - observed)
        required = ledger.DISK_RESERVE_BYTES + self.scratch_bytes + remaining
        disk = f"disk free is below {required / 1e9:.3f} GB"
        blockers = (["swap exceeds tolerance or is unavailable"]
                    if swap_blocker else []) + [disk]
        return {
            "schema": "hawking.doctor_v5_ultra_resource_gate.v1",
            "sampled_at": "2026-07-15T00:00:00+00:00", "ok": False,
            "blockers": blockers, "disk_reserve_bytes": ledger.DISK_RESERVE_BYTES,
            "scratch_bytes": self.scratch_bytes,
            "required_free_bytes": required,
            "available_total_capacity_bytes": required - 1,
            "required_total_capacity_bytes": required,
            "resident_payload_bytes": 0, "resident_predecessor_bytes": 0,
            "projected_incremental_output_bytes": remaining,
            "capacity_ok": False, "resources": {"disk_free_gb": 1},
            "thermal": {"ok": True}, "cell_id": self.cell_id,
            "projected_whole_output_bytes": self.frozen_projection,
            "observed_current_output_bytes": observed,
        }

    def policy(self, *extra_bindings: gate.PhaseGateBinding) -> dict[str, object]:
        bindings = (self.binding, *extra_bindings)
        return {gate.PHASE_POLICY_KEY: {
            "schema": gate.PHASE_GATE_API_SCHEMA, "enabled": True,
            "module_sha256": _file_sha(Path(gate.__file__)),
            "ledger_module_sha256": _file_sha(Path(ledger.__file__)),
            "ram_credit_bytes": 0,
            "bindings": [{
                "plan_path": str(row.plan_path),
                "plan_file_sha256": row.plan_file_sha256,
                "plan_sha256": row.plan_sha256,
                "plan_cell_sha256": row.plan_cell_sha256,
                "cell_id": row.cell_id,
                "cell_identity_sha256": row.cell_identity_sha256,
                "runtime_spec_path": str(row.runtime_spec_path),
                "runtime_spec_file_sha256": row.runtime_spec_file_sha256,
                "program_spec_sha256": row.program_spec_sha256,
                "execution_output_root": str(row.execution_output_root),
                "disk_reserve_bytes": row.disk_reserve_bytes,
                "declared_scratch_bytes": row.declared_scratch_bytes,
                "frozen_projected_output_bytes": row.frozen_projected_output_bytes,
            } for row in bindings],
        }}

    def execution(self) -> dict[str, object]:
        return {
            "cell": self.cell, "output_dir": self.output,
            "scratch_bytes": self.scratch_bytes,
            "runtime": {"path": self.spec_path,
                        "sha256": _file_sha(self.spec_path),
                        "document": self.spec},
        }


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path)


def _statvfs(free: int) -> SimpleNamespace:
    return SimpleNamespace(f_bavail=free, f_frsize=1)


def _built_and_persisted(fixture: Fixture) -> tuple[dict[str, object], dict[str, object]]:
    original = fixture.original_gate()
    receipt = gate.build_phase_receipt(
        original, fixture.binding, execution_cell=fixture.cell,
        workspace_root=fixture.root, now=NOW)
    gate.persist_phase_receipt_atomic(
        receipt, fixture.binding, workspace_root=fixture.root)
    return original, receipt


def test_overrun_complete_proof_relaxes_only_disk_and_never_writes_results(
        fixture: Fixture) -> None:
    before = _tree(fixture.output)
    original, receipt = _built_and_persisted(fixture)
    required = receipt["evidence"]["required_free_bytes"]
    result = gate.evaluate_persisted_phase_receipt(
        original, fixture.binding, execution_cell=fixture.cell,
        persisted_receipt_path=fixture.binding.phase_receipt_path,
        workspace_root=fixture.root, now=NOW,
        _statvfs_fn=lambda _path: _statvfs(required),
    )
    assert result.applied is True
    assert result.gate["blockers"] == ["swap exceeds tolerance or is unavailable"]
    assert result.gate["ok"] is False
    assert result.gate["scratch_bytes"] == 20
    assert receipt["evidence"]["observed_durable_packed_bytes"] == 12
    assert receipt["evidence"]["projection_overrun_bytes"] == 2
    assert receipt["evidence"]["effective_whole_packed_output_bytes"] == 12
    assert result.gate["phase_aware_disk_gate"]["ram_credit_bytes"] == 0
    assert _tree(fixture.output) == before
    assert fixture.binding.phase_receipt_path.is_file()
    assert fixture.results not in fixture.binding.phase_receipt_path.parents


def test_only_disk_blocker_can_become_admitted(fixture: Fixture) -> None:
    original = fixture.original_gate(swap_blocker=False)
    receipt = gate.build_phase_receipt(
        original, fixture.binding, execution_cell=fixture.cell,
        workspace_root=fixture.root, now=NOW)
    gate.persist_phase_receipt_atomic(receipt, fixture.binding,
                                      workspace_root=fixture.root)
    result = gate.evaluate_persisted_phase_receipt(
        original, fixture.binding, execution_cell=fixture.cell,
        persisted_receipt_path=fixture.binding.phase_receipt_path,
        workspace_root=fixture.root, now=NOW,
        _statvfs_fn=lambda _path: _statvfs(receipt["evidence"][
            "required_free_bytes"]),
    )
    assert result.applied and result.gate["ok"] is True
    assert result.gate["blockers"] == []


@pytest.mark.parametrize("mutation", ["missing_bundle", "missing_attest",
                                       "observed_mismatch"])
def test_incomplete_or_inconsistent_output_cannot_build_credit(
        fixture: Fixture, mutation: str) -> None:
    original = fixture.original_gate()
    if mutation == "missing_bundle":
        fixture.checkpoint["completed_units"].remove("bundle_manifest")
        fixture.checkpoint["units"].pop("bundle_manifest")
        _write_json(fixture.checkpoint_path, fixture.checkpoint)
    elif mutation == "missing_attest":
        fixture.checkpoint["completed_units"].remove("attest:00001")
        fixture.checkpoint["units"].pop("attest:00001")
        _write_json(fixture.checkpoint_path, fixture.checkpoint)
    else:
        original["observed_current_output_bytes"] += 1
        original["projected_incremental_output_bytes"] = 0
    with pytest.raises(gate.PhaseGateError):
        gate.build_phase_receipt(
            original, fixture.binding, execution_cell=fixture.cell,
            workspace_root=fixture.root, now=NOW)


def test_absent_tampered_stale_or_low_disk_receipt_is_exact_fallback(
        fixture: Fixture) -> None:
    original, receipt = _built_and_persisted(fixture)
    absent = gate.evaluate_persisted_phase_receipt(
        original, fixture.binding, execution_cell=fixture.cell,
        persisted_receipt_path=fixture.policy_root / "wrong.json",
        workspace_root=fixture.root, now=NOW)
    assert not absent.applied and absent.gate == original

    tampered = copy.deepcopy(receipt)
    tampered["evidence"]["remaining_scratch_bytes"] += 1
    _write_json(fixture.binding.phase_receipt_path, tampered)
    bad = gate.evaluate_persisted_phase_receipt(
        original, fixture.binding, execution_cell=fixture.cell,
        persisted_receipt_path=fixture.binding.phase_receipt_path,
        workspace_root=fixture.root, now=NOW)
    assert not bad.applied and bad.gate == original

    gate.persist_phase_receipt_atomic(receipt, fixture.binding,
                                      workspace_root=fixture.root)
    stale = gate.evaluate_persisted_phase_receipt(
        original, fixture.binding, execution_cell=fixture.cell,
        persisted_receipt_path=fixture.binding.phase_receipt_path,
        workspace_root=fixture.root,
        now=NOW + dt.timedelta(seconds=gate.MAX_RECEIPT_AGE_SECONDS + 1))
    assert not stale.applied and stale.gate == original

    low = gate.evaluate_persisted_phase_receipt(
        original, fixture.binding, execution_cell=fixture.cell,
        persisted_receipt_path=fixture.binding.phase_receipt_path,
        workspace_root=fixture.root, now=NOW,
        _statvfs_fn=lambda _path: _statvfs(
            receipt["evidence"]["required_free_bytes"] - 1),
    )
    assert not low.applied and low.gate == original


def test_spec_checkpoint_manifest_and_path_tampering_fail_closed(
        fixture: Fixture) -> None:
    original, _ = _built_and_persisted(fixture)
    fixture.spec["campaign_binding"]["cell_id"] = "other"
    _write_json(fixture.spec_path, fixture.spec)
    result = gate.evaluate_persisted_phase_receipt(
        original, fixture.binding, execution_cell=fixture.cell,
        persisted_receipt_path=fixture.binding.phase_receipt_path,
        workspace_root=fixture.root, now=NOW)
    assert not result.applied and result.gate == original

    escaped = copy.copy(fixture.binding)
    object.__setattr__(escaped, "execution_output_root", Path("/tmp/escape"))
    with pytest.raises(gate.PhaseGateError):
        gate.build_phase_receipt(original, escaped, execution_cell=fixture.cell,
                                 workspace_root=fixture.root, now=NOW)


def test_checkpoint_monotonic_advancement_is_rebound_each_probe(
        fixture: Fixture) -> None:
    original, first = _built_and_persisted(fixture)
    first_checkpoint = first["evidence"]["worker_checkpoint"]["sha256"]
    next_unit = "override_manifest"
    fixture.checkpoint["completed_units"].append(next_unit)
    fixture.checkpoint["units"][next_unit] = {
        "completed_at": "2026-07-15T00:02:00+00:00"}
    fixture.checkpoint["updated_at"] = "2026-07-15T00:02:00+00:00"
    _write_json(fixture.checkpoint_path, fixture.checkpoint)
    second = gate.build_phase_receipt(
        original, fixture.binding, execution_cell=fixture.cell,
        workspace_root=fixture.root, now=NOW)
    assert second["evidence"]["worker_checkpoint"]["sha256"] != first_checkpoint
    gate.persist_phase_receipt_atomic(second, fixture.binding,
                                      workspace_root=fixture.root)
    result = gate.evaluate_persisted_phase_receipt(
        original, fixture.binding, execution_cell=fixture.cell,
        persisted_receipt_path=fixture.binding.phase_receipt_path,
        workspace_root=fixture.root, now=NOW,
        _statvfs_fn=lambda _path: _statvfs(second["evidence"][
            "required_free_bytes"]),
    )
    assert result.applied is True


def test_atomic_write_failure_never_relaxes_and_result_tree_is_untouched(
        fixture: Fixture) -> None:
    original = fixture.original_gate()
    before = _tree(fixture.output)

    def predecessor_gate(plan, state, execution):
        return copy.deepcopy(original)

    base = ModuleType("fake_base")
    base.ROOT = fixture.root
    wrapped = gate.install_phase_gate(
        base, predecessor_gate, fixture.policy(), fixture.policy_root)
    with mock.patch.object(gate.os, "replace", side_effect=OSError("injected")):
        observed = wrapped(fixture.plan, {}, fixture.execution())
    assert observed["ok"] == original["ok"]
    assert observed["blockers"] == original["blockers"]
    assert observed["required_free_bytes"] == original["required_free_bytes"]
    assert observed["phase_aware_disk_gate_diagnostic"]["nonpermissive"] is True
    assert _tree(fixture.output) == before


def test_installer_exact_signature_multi_cell_and_policy_hash_guards(
        fixture: Fixture, tmp_path: Path) -> None:
    original = fixture.original_gate()

    def predecessor_gate(plan, state, execution):
        return copy.deepcopy(original)

    base = ModuleType("fake_base")
    base.ROOT = fixture.root
    wrapped = gate.install_phase_gate(
        base, predecessor_gate, fixture.policy(), fixture.policy_root)
    assert str(inspect.signature(wrapped)) == str(inspect.signature(predecessor_gate))
    observed = wrapped(fixture.plan, {}, fixture.execution())
    assert "phase_aware_disk_gate" in observed
    assert fixture.binding.phase_receipt_path.is_file()

    unknown = copy.deepcopy(fixture.execution())
    unknown["cell"]["cell_id"] = "unbound-cell"
    assert wrapped(fixture.plan, {}, unknown) == original

    bad = fixture.policy()
    bad[gate.PHASE_POLICY_KEY]["module_sha256"] = "0" * 64
    with pytest.raises(gate.PhaseGateError):
        gate.install_phase_gate(base, predecessor_gate, bad, fixture.policy_root)
    bad = fixture.policy()
    bad[gate.PHASE_POLICY_KEY]["ram_credit_bytes"] = 1
    with pytest.raises(gate.PhaseGateError):
        gate.install_phase_gate(base, predecessor_gate, bad, fixture.policy_root)


def test_policy_root_must_be_staged_and_never_results(fixture: Fixture) -> None:
    with pytest.raises(gate.PhaseGateError):
        gate._bindings_from_policy(
            fixture.policy(), fixture.output, fixture.root)
