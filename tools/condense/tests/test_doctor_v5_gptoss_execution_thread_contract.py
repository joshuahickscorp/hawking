#!/usr/bin/env python3.12
"""No-model tests for exact GPT-OSS aggressive execution/thread contracts."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


CONDENSE = Path(__file__).resolve().parents[1]
ROOT = CONDENSE.parents[1]
sys.path.insert(0, str(CONDENSE))

import doctor_v5_adapter_abi as adapter_abi
import doctor_v5_gptoss_execution_thread_contract as contract
import doctor_v5_ultra_aggressive_queue as aggressive_queue
import doctor_v5_ultra_queue as queue_contract


class GptossExecutionThreadContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temporary.name)
        self.plan_path = self.root / "campaign_plan.json"
        self.registry_path = self.root / "adapter_registry.json"
        self.runtime_root = self.root / "runtime_specs"
        self.receipts_root = self.root / "physical_receipts"
        self.contracts_root = self.root / "contracts"
        self.source = self.root / "source-work-plan.json"
        self.source.write_text('{"source":"bound"}\n', encoding="utf-8")
        self.executables: dict[str, Path] = {}
        self.plan = self._make_plan()
        self.registry = self._make_registry()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _make_plan(self) -> dict:
        rows: list[dict] = []
        priority = 1
        for rate in contract.RATES:
            token = rate.replace(".", "p")
            for branch, (operation, adapter, schema) in contract.BRANCHES.items():
                suffix = branch.replace("_", "-")
                cell_id = f"gpt-oss-120b__{token}bpw__{suffix}"
                rows.append({
                    "cell_id": cell_id,
                    "cell_identity_sha256": hashlib.sha256(
                        f"identity:{cell_id}".encode()
                    ).hexdigest(),
                    "model_label": contract.MODEL_LABEL,
                    "model_family": contract.GPTOSS_FAMILY,
                    "hf_id": contract.HF_ID,
                    "rate_id": rate,
                    "rate_bpw": float(rate),
                    "branch": branch,
                    "command": operation,
                    "adapter_id": adapter,
                    "backend": "apple-cpu-strand",
                    "runtime_spec_schema": schema,
                    "runtime_spec_path": str(
                        (self.runtime_root / f"{cell_id}.json").relative_to(ROOT)
                    ),
                    "priority": priority,
                })
                priority += 1
        plan = {
            "schema": "hawking.doctor_v5_ultra_campaign_plan.v1",
            "cells": rows,
        }
        plan["plan_sha256"] = contract._hash_value(plan)
        self._write_json(self.plan_path, plan)
        return plan

    def _make_registry(self) -> dict:
        entries: list[dict] = []
        for _branch, (operation, adapter, _schema) in contract.BRANCHES.items():
            source = self.root / f"{adapter}.py"
            executable = self.root / f"{adapter}.bin"
            source.write_text(f"# reviewed {adapter}\n", encoding="utf-8")
            executable.write_text(f"physical {adapter}\n", encoding="utf-8")
            self.executables[adapter] = executable
            entries.append({
                "adapter_id": adapter, "adapter_version": "physical-v1",
                "source_path": str(source),
                "source_sha256": contract._hash_file(source)[0],
                "executable_path": str(executable),
                "executable_sha256": contract._hash_file(executable)[0],
                "entrypoint_argv": [str(executable), "{request_path}"],
                "operations": [operation],
                "model_families": [contract.GPTOSS_FAMILY],
                "backends": ["apple-cpu-strand"],
                "request_schema": adapter_abi.REQUEST_SCHEMA,
                "result_schema": adapter_abi.RESULT_SCHEMA,
                "checkpoint_schema": adapter_abi.CHECKPOINT_SCHEMA,
                "reviewed": True,
                "execution_only_not_quality_evidence": True,
            })
        registry = adapter_abi.build_registry(
            entries, created_at="2026-07-15T00:00:00+00:00"
        )
        self._write_json(self.registry_path, registry)
        return registry

    def _write_specs(self, *, threads: int = 7) -> None:
        source = {"role": "source_work_plan", **contract._artifact(self.source)}
        for cell in self.plan["cells"]:
            resources = {
                "disk_reserve_bytes": 150_000_000_000,
                "scratch_budget_bytes": 12_000_000_000,
                "threads": threads,
            }
            spec = {
                "schema": cell["runtime_spec_schema"],
                "label": contract.MODEL_LABEL,
                "model_family": contract.GPTOSS_FAMILY,
                "backend": cell["backend"],
                "adapter_id": cell["adapter_id"],
                "operation": cell["command"],
                "quality_claims_permitted": False,
                "source_deletion_permitted": False,
                "campaign_binding": {
                    "cell_id": cell["cell_id"],
                    "cell_identity_sha256": cell["cell_identity_sha256"],
                    "branch": cell["branch"],
                    "target_rate_id": cell["rate_id"],
                    "target_rate_bpw": cell["rate_bpw"],
                    "label": contract.MODEL_LABEL,
                },
                "resources": resources,
                "inputs": [source],
                "program_spec_sha256": "0" * 64,
                "resource_admission_sha256": contract._hash_value(resources),
            }
            spec["program_spec_sha256"] = contract._hash_value(
                queue_contract._runtime_program_payload(spec)
            )
            self._write_json(
                self.runtime_root / f"{cell['cell_id']}.json", spec
            )

    def _write_receipts(self, *, threads: int = 7,
                        omit: str | None = None,
                        simulated: bool = False) -> None:
        start = dt.datetime(2026, 7, 15, tzinfo=dt.timezone.utc)
        end = start + dt.timedelta(seconds=10)
        for cell in self.plan["cells"]:
            cell_id = cell["cell_id"]
            if cell_id == omit:
                continue
            spec_path = self.runtime_root / f"{cell_id}.json"
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            candidate = self.root / "physical_outputs" / f"{cell_id}.candidate"
            oracle = self.root / "physical_outputs" / f"{cell_id}.oracle"
            launch = self.root / "physical_outputs" / f"{cell_id}.launch.json"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(f"exact:{cell_id}\n".encode())
            oracle.write_bytes(f"exact:{cell_id}\n".encode())
            launch.write_text(
                json.dumps({"pid": 4000, "cell_id": cell_id}) + "\n",
                encoding="utf-8",
            )
            receipt = {
                "schema": contract.PHYSICAL_RECEIPT_SCHEMA,
                "version": contract.ABI_VERSION,
                "cell_id": cell_id,
                "model_family": contract.GPTOSS_FAMILY,
                "runtime_spec_sha256": contract._hash_file(spec_path)[0],
                "runtime_inputs_sha256": contract._hash_value(spec["inputs"]),
                "selected_threads": threads,
                "wall_seconds": 10.0,
                "review_status": "approved-source-bound-exact-output",
                "physical_execution": {
                    "mode": "physical-production-host",
                    "simulated": simulated, "fixture": False,
                    "host_id_sha256": "a" * 64,
                    "boot_id_sha256": "b" * 64,
                    "pid": 4000,
                    "started_at": start.isoformat(),
                    "completed_at": end.isoformat(),
                    "exit_code": 0,
                    "input_content_hashes_verified": True,
                    "input_count": len(spec["inputs"]),
                    "argv_sha256": "c" * 64,
                    "executable": contract._artifact(
                        self.executables[cell["adapter_id"]]
                    ),
                    "launch_receipt": contract._artifact(launch),
                },
                "exact_output": {
                    "status": "exact", "comparison": "byte-for-byte",
                    "candidate": contract._artifact(candidate),
                    "oracle": contract._artifact(oracle),
                    "tested_cases": 8, "skipped_cases": 0,
                    "mismatch_count": 0,
                },
            }
            receipt["receipt_sha256"] = contract._hash_value(receipt)
            self._write_json(self.receipts_root / f"{cell_id}.json", receipt)

    def _status(self) -> dict:
        return contract.status(
            plan_path=self.plan_path, registry_path=self.registry_path,
            runtime_root=self.runtime_root, receipts_root=self.receipts_root,
            contracts_root=self.contracts_root, pending_wiring_path=None,
        )

    def _stage(self) -> dict:
        return contract.stage(
            plan_path=self.plan_path, registry_path=self.registry_path,
            runtime_root=self.runtime_root, receipts_root=self.receipts_root,
            contracts_root=self.contracts_root, pending_wiring_path=None,
        )

    def test_abi_matches_aggressive_consumer_exactly(self) -> None:
        self.assertEqual(contract.CONTRACT_SCHEMA,
                         aggressive_queue.SOURCE_BOUND_CONTRACT_SCHEMA)
        self.assertEqual(contract.ABI_VERSION, aggressive_queue.VERSION)
        self.assertEqual(contract.CONTRACT_KEYS, {
            "schema", "version", "cell_id", "model_family",
            "runtime_spec_sha256", "runtime_inputs_sha256",
            "selected_threads", "projected_wall_seconds", "exclusive_cpu",
            "review_status", "exact_output_receipts", "contract_sha256",
        })

    def test_missing_physical_evidence_blocks_without_writing(self) -> None:
        self._write_specs()
        omitted = self.plan["cells"][0]["cell_id"]
        self._write_receipts(omit=omitted)
        status = self._status()
        self.assertEqual(status["reviewed_runtime_specs"], 40)
        self.assertEqual(status["physical_exact_output_receipts"], 39)
        self.assertEqual(status["missing_physical_evidence"], [omitted])
        self.assertFalse(status["stage_permitted"])
        with self.assertRaises(contract.ContractError):
            self._stage()
        self.assertFalse(self.contracts_root.exists())

    def test_exact_40_stage_verify_and_idempotence(self) -> None:
        # Seven threads proves this path consumes the reviewed GPT-OSS runtime
        # declaration rather than inventing a Qwen 8/12/16/20 profile.
        self._write_specs(threads=7)
        self._write_receipts(threads=7)
        before = self._status()
        self.assertTrue(before["stage_permitted"])
        self.assertEqual(before["reviewed_runtime_specs"], 40)
        self.assertEqual(before["physical_exact_output_receipts"], 40)
        manifest = self._stage()
        manifest_path = self.contracts_root / "manifest.json"
        first_mtime = manifest_path.stat().st_mtime_ns
        first = manifest_path.read_bytes()
        again = self._stage()
        self.assertEqual(manifest, again)
        self.assertEqual(first, manifest_path.read_bytes())
        self.assertEqual(first_mtime, manifest_path.stat().st_mtime_ns)
        errors = contract.verify_manifest(
            manifest_path, plan_path=self.plan_path,
            registry_path=self.registry_path, runtime_root=self.runtime_root,
            receipts_root=self.receipts_root, pending_wiring_path=None,
        )
        self.assertEqual(errors, [])
        for row in manifest["aggressive_runtime_rows"]:
            doc = json.loads(Path(
                row["source_bound_execution_contract"]["path"]
            ).read_text(encoding="utf-8"))
            self.assertEqual(set(doc), contract.CONTRACT_KEYS)
            self.assertEqual(doc["selected_threads"], 7)
            self.assertNotIn("thread_profile", doc)
            self.assertNotIn("thread_selection_sha256", doc)
        self.assertFalse(manifest["qwen_thread_profiles_used"])
        self.assertFalse(manifest["automatic_activation_permitted"])

    def test_simulated_receipt_is_rejected(self) -> None:
        self._write_specs()
        self._write_receipts(simulated=True)
        status = self._status()
        self.assertEqual(status["physical_exact_output_receipts"], 0)
        self.assertEqual(len(status["invalid_physical_evidence"]), 40)
        self.assertTrue(all(any("real source-verified physical run" in item
                                for item in errors)
                            for errors in status["invalid_physical_evidence"].values()))

    def test_symlinked_physical_evidence_is_rejected(self) -> None:
        self._write_specs()
        self._write_receipts()
        cell_id = self.plan["cells"][0]["cell_id"]
        target = self.receipts_root / f"{cell_id}.json"
        backing = self.receipts_root / f"{cell_id}.backing.json"
        target.rename(backing)
        target.symlink_to(backing)
        status = self._status()
        self.assertEqual(status["physical_exact_output_receipts"], 39)
        self.assertIn(cell_id, status["invalid_physical_evidence"])
        self.assertTrue(any("symlink" in item
                            for item in status["invalid_physical_evidence"][cell_id]))

    def test_runtime_or_receipt_drift_invalidates_manifest(self) -> None:
        self._write_specs()
        self._write_receipts()
        self._stage()
        cell_id = self.plan["cells"][0]["cell_id"]
        path = self.runtime_root / f"{cell_id}.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec["resources"]["threads"] = 8
        spec["resource_admission_sha256"] = contract._hash_value(spec["resources"])
        self._write_json(path, spec)
        errors = contract.verify_manifest(
            self.contracts_root / "manifest.json", plan_path=self.plan_path,
            registry_path=self.registry_path, runtime_root=self.runtime_root,
            receipts_root=self.receipts_root, pending_wiring_path=None,
        )
        self.assertTrue(errors)

    def test_cas_refuses_contract_replacement(self) -> None:
        self._write_specs()
        self._write_receipts()
        manifest = self._stage()
        target = Path(manifest["aggressive_runtime_rows"][0][
            "source_bound_execution_contract"
        ]["path"])
        changed = json.loads(target.read_text(encoding="utf-8"))
        changed["projected_wall_seconds"] = 999.0
        changed["contract_sha256"] = contract._hash_value(
            contract._without(changed, "contract_sha256")
        )
        self._write_json(target, changed)
        with self.assertRaises(contract.ContractError):
            self._stage()


if __name__ == "__main__":
    unittest.main()
