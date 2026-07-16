from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import doctor_v5_gc_runtime_transition as transition
import doctor_v5_qwen_treatment_block_parallel_adapter as wrapper


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def file_ref(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


class AuthorityFixture:
    def __init__(self) -> None:
        (ROOT / "scratch").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "scratch")
        self.root = Path(self.temp.name)
        self.wrapper_path = self.root / "tools/condense/wrapper.py"
        self.module_path = self.root / "tools/condense/transition.py"
        self.wrapper_path.parent.mkdir(parents=True)
        self.wrapper_path.write_text("# exact wrapper\n", encoding="utf-8")
        self.module_path.write_text("# exact transition module\n", encoding="utf-8")
        self.authority_path = self.root / "reports/authority.json"

        self.incidents: list[dict] = []
        self.old_specs: list[Path] = []
        self.current_paths: list[Path] = []
        self.receipt_paths: list[Path] = []
        self.consumer_specs: list[dict] = []
        cells = []
        state_rows = {}
        for index, (label, branch, rate_id) in enumerate((
            ("3B", "doctor_full", "3"),
            ("32B", "doctor_static", "4"),
        )):
            cell_id = f"consumer-{index}"
            cell_identity = f"{index + 1}" * 64
            predecessor_id = f"predecessor-{index}"
            predecessor_identity = f"{index + 3}" * 64
            old_rel = f"reports/rollback/old-{index}.json"
            current_rel = f"reports/runtime/current-{index}.json"
            receipt_rel = f"reports/results/{predecessor_id}/packed_gc_receipt.json"
            old_spec = {
                "schema": "runtime", "label": label,
                "campaign_binding": {
                    "cell_id": cell_id, "cell_identity_sha256": cell_identity,
                    "branch": branch, "label": label, "target_rate_id": rate_id,
                    "target_rate_bpw": float(rate_id),
                },
                "operation": branch, "semantic_nonce": index,
                "source_deletion_permitted": False,
                "quality_claims_permitted": False,
                "inputs": [{"role": "old", "path": "ignored", "sha256": "0" * 64,
                            "bytes": 1}],
                "resources": {"threads": 8},
                "resource_admission_sha256": "a" * 64,
                "program_spec_sha256": "0" * 64,
            }
            program = transition._hash_value(
                transition._runtime_program_payload(old_spec)
            )
            old_spec["program_spec_sha256"] = program
            old_path = self.root / old_rel
            write_json(old_path, old_spec)
            old_artifact = file_ref(old_path)
            successor = {
                "cell_id": cell_id, "cell_identity_sha256": cell_identity,
                "runtime_spec_path": current_rel,
                "runtime_spec_sha256": old_artifact["sha256"],
                "runtime_spec_bytes": old_artifact["bytes"],
                "program_spec_sha256": program,
            }
            receipt = {
                "schema": "hawking.doctor_v5_packed_gc_receipt.v2",
                "cell_id": predecessor_id,
                "cell_identity_sha256": predecessor_identity,
                "result_sha256": f"{index + 5}" * 64,
                "successor": successor, "parent_source_deleted": False,
                "receipt_sha256": "0" * 64,
            }
            receipt["receipt_sha256"] = transition._hash_value(
                transition._without(receipt, "receipt_sha256")
            )
            receipt_path = self.root / receipt_rel
            write_json(receipt_path, receipt)
            receipt_artifact = file_ref(receipt_path)
            consumer = {
                "cell_id": cell_id, "cell_identity_sha256": cell_identity,
                "program_spec_sha256": program, "branch": branch,
                "label": label, "rate_id": rate_id,
            }
            self.incidents.append({
                "incident_id": f"incident-{index}", "consumer": consumer,
                "predecessor": {
                    "cell_id": predecessor_id,
                    "cell_identity_sha256": predecessor_identity,
                    "result_sha256": receipt["result_sha256"],
                },
                "receipt_rel": receipt_rel,
                "receipt_file_sha256": receipt_artifact["sha256"],
                "receipt_bytes": receipt_artifact["bytes"],
                "receipt_sha256": receipt["receipt_sha256"],
                "old_spec_rel": old_rel, "successor": successor,
            })
            self.old_specs.append(old_path)
            self.current_paths.append(self.root / current_rel)
            self.receipt_paths.append(receipt_path)
            cells.append({
                "cell_id": cell_id, "cell_identity_sha256": cell_identity,
                "branch": branch, "model_label": label, "rate_id": rate_id,
                "source_deletion_permitted": False,
                "quality_claims_permitted": False,
            })
            state_rows[cell_id] = {
                "status": "blocked-execution", "attempts": index + 1,
                "result_sha256": None, "execution_receipt_sha256": None,
            }

        plan = {"schema": "test-plan", "cells": cells}
        plan["plan_sha256"] = transition._hash_value(plan)
        self.plan_sha256 = plan["plan_sha256"]
        write_json(self.root / transition.PLAN_REL, plan)
        state = {"schema": "test-state", "plan_sha256": self.plan_sha256,
                 "cells": state_rows}
        state["state_sha256"] = transition._hash_value(state)
        write_json(self.root / transition.STATE_REL, state)

        self.patch = mock.patch.multiple(
            transition, ROOT=self.root, MODULE_PATH=self.module_path,
            WRAPPER_PATH=self.wrapper_path, EXPECTED_PLAN_SHA256=self.plan_sha256,
            _INCIDENTS=tuple(self.incidents),
        )
        self.patch.start()
        transition.generate_authority(self.authority_path)

        wrapper_row = {"role": "adapter_source", **transition.artifact(
            self.wrapper_path)}
        module_row = transition.module_input_row()
        authority_row = transition.authority_input_row(self.authority_path)
        for old_path, current_path in zip(self.old_specs, self.current_paths):
            current = json.loads(old_path.read_text(encoding="utf-8"))
            current["resources"] = {"threads": 20, "scratch_budget_bytes": 1234}
            current["resource_admission_sha256"] = "b" * 64
            current["inputs"] = [wrapper_row, module_row, authority_row]
            write_json(current_path, current)
            self.consumer_specs.append(current)

    def close(self) -> None:
        self.patch.stop()
        self.temp.cleanup()

    def validate(self, index: int = 0) -> dict:
        return transition.validate_transition(
            authority_path=self.authority_path,
            receipt_path=self.receipt_paths[index],
            consumer_spec=self.consumer_specs[index],
        )


class GCRuntimeTransitionAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = AuthorityFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_production_allowlist_is_exactly_the_two_incidents(self) -> None:
        # Inspect the immutable production catalog, not the patched fixture.
        self.fixture.patch.stop()
        try:
            self.assertEqual(
                [row["consumer"]["cell_id"] for row in transition._INCIDENTS],
                ["qwen2-5-3b__3bpw__doctor-full",
                 "qwen2-5-32b__4bpw__doctor-static"],
            )
        finally:
            self.fixture.patch.start()

    def test_exact_two_transitions_pass(self) -> None:
        for index in range(2):
            proof = self.fixture.validate(index)
            self.assertEqual(proof["incident_id"], f"incident-{index}")
            self.assertFalse(proof["source_deletion_permitted"])
            self.assertFalse(proof["quality_claims_permitted"])

    def test_altered_program_fails(self) -> None:
        current = copy.deepcopy(self.fixture.consumer_specs[0])
        current["semantic_nonce"] = 99
        write_json(self.fixture.current_paths[0], current)
        with self.assertRaises(transition.GCRuntimeTransitionError):
            transition.validate_transition(
                authority_path=self.fixture.authority_path,
                receipt_path=self.fixture.receipt_paths[0], consumer_spec=current)

    def test_altered_cell_fails(self) -> None:
        current = copy.deepcopy(self.fixture.consumer_specs[0])
        current["campaign_binding"]["cell_identity_sha256"] = "f" * 64
        write_json(self.fixture.current_paths[0], current)
        with self.assertRaises(transition.GCRuntimeTransitionError):
            transition.validate_transition(
                authority_path=self.fixture.authority_path,
                receipt_path=self.fixture.receipt_paths[0], consumer_spec=current)

    def test_altered_receipt_fails(self) -> None:
        with self.fixture.receipt_paths[0].open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaises(transition.GCRuntimeTransitionError):
            self.fixture.validate()

    def test_altered_old_spec_fails(self) -> None:
        with self.fixture.old_specs[0].open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaises(transition.GCRuntimeTransitionError):
            self.fixture.validate()

    def test_altered_wrapper_fails(self) -> None:
        self.fixture.wrapper_path.write_text("# changed wrapper\n", encoding="utf-8")
        with self.assertRaises(transition.GCRuntimeTransitionError):
            self.fixture.validate()

    def test_altered_module_fails(self) -> None:
        self.fixture.module_path.write_text("# changed module\n", encoding="utf-8")
        with self.assertRaises(transition.GCRuntimeTransitionError):
            self.fixture.validate()

    def test_altered_authority_fails(self) -> None:
        authority = json.loads(self.fixture.authority_path.read_text(encoding="utf-8"))
        authority["policy"]["quality_claims_permitted"] = True
        write_json(self.fixture.authority_path, authority)
        with self.assertRaises(transition.GCRuntimeTransitionError):
            self.fixture.validate()

    def test_unlisted_transition_fails(self) -> None:
        unlisted = self.fixture.root / "reports/unlisted/packed_gc_receipt.json"
        unlisted.parent.mkdir(parents=True)
        unlisted.write_bytes(self.fixture.receipt_paths[0].read_bytes())
        with self.assertRaises(transition.GCRuntimeTransitionError):
            transition.validate_transition(
                authority_path=self.fixture.authority_path,
                receipt_path=unlisted,
                consumer_spec=self.fixture.consumer_specs[0],
            )


class BlockAdapterHookTests(unittest.TestCase):
    def test_original_valid_receipt_path_is_unchanged(self) -> None:
        sentinel = {"receipt_sha256": "a" * 64}
        with mock.patch.object(
            wrapper, "_ORIGINAL_VALIDATE_GC_RECEIPT", return_value=sentinel,
        ) as original, mock.patch.object(
            wrapper._gc_transition, "validate_transition",
        ) as authority:
            observed = wrapper._validate_gc_receipt(
                Path("unused"), binding={}, result={}, packed_rows=[],
                consumer_spec={},
            )
        self.assertIs(observed, sentinel)
        original.assert_called_once()
        authority.assert_not_called()

    def _hook_fixture(self):
        (ROOT / "scratch").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "scratch")
        successor = Path(temporary.name) / "successor.json"
        successor.write_text("{}\n", encoding="utf-8")
        current_sha, current_bytes = wrapper._BASE._hash_file(successor)
        proof = {
            "successor_path": str(successor),
            "current_runtime": {"path": str(successor), "sha256": current_sha,
                                "bytes": current_bytes},
            "historical_runtime": {"sha256": "9" * 64, "bytes": 999},
        }
        return temporary, successor, proof

    def test_exact_authorized_hash_is_virtualized_and_restored(self) -> None:
        temporary, successor, proof = self._hook_fixture()
        self.addCleanup(temporary.cleanup)
        original_hash = wrapper._BASE._hash_file
        calls = 0

        def frozen(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise wrapper._BASE.TreatmentError(wrapper._HISTORICAL_RUNTIME_ERROR)
            self.assertEqual(wrapper._BASE._hash_file(successor), ("9" * 64, 999))
            return {"receipt_sha256": "b" * 64}

        with mock.patch.object(wrapper, "_ORIGINAL_VALIDATE_GC_RECEIPT",
                               side_effect=frozen), mock.patch.object(
            wrapper._gc_transition, "validate_transition", return_value=proof,
        ):
            observed = wrapper._validate_gc_receipt(
                Path("receipt"), binding={}, result={}, packed_rows=[],
                consumer_spec={},
            )
        self.assertEqual(observed["receipt_sha256"], "b" * 64)
        self.assertIs(wrapper._BASE._hash_file, original_hash)

    def test_temporary_hash_hook_is_restored_after_failure(self) -> None:
        temporary, _successor, proof = self._hook_fixture()
        self.addCleanup(temporary.cleanup)
        original_hash = wrapper._BASE._hash_file
        calls = 0

        def frozen(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise wrapper._BASE.TreatmentError(wrapper._HISTORICAL_RUNTIME_ERROR)
            raise wrapper._BASE.TreatmentError("later canonical receipt failure")

        with mock.patch.object(wrapper, "_ORIGINAL_VALIDATE_GC_RECEIPT",
                               side_effect=frozen), mock.patch.object(
            wrapper._gc_transition, "validate_transition", return_value=proof,
        ):
            with self.assertRaises(wrapper._BASE.TreatmentError):
                wrapper._validate_gc_receipt(
                    Path("receipt"), binding={}, result={}, packed_rows=[],
                    consumer_spec={},
                )
        self.assertIs(wrapper._BASE._hash_file, original_hash)


if __name__ == "__main__":
    unittest.main()
