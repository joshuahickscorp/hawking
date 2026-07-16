#!/usr/bin/env python3.12
"""Cheap adversarial tests for the unbound shared-preprocessing cache."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doctor_v5_shared_preprocess_cache as C


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def artifact(label: str, *, content_sha: str | None = None,
             size: int = 17) -> dict[str, object]:
    return {"artifact_instance_id": label, "sha256": content_sha or sha(label),
            "bytes": size}


def serial_reference(candidate: dict[str, object], label: str,
                     input_identity_sha256: str) -> dict[str, object]:
    program = {"sha256": sha(f"{label}-serial-program"), "bytes": 123}
    invocation = {"mode": "independent-serial-oracle",
                  "program_sha256": program["sha256"],
                  "input_identity_sha256": input_identity_sha256}
    return {
        "artifact": artifact(f"{label}-serial", content_sha=str(candidate["sha256"]),
                             size=int(candidate["bytes"])),
        "program_artifact": program, "invocation": invocation,
        "invocation_sha256": C._hash_value(invocation),
        "semantic_receipt": artifact(f"{label}-serial-semantic"),
    }


class SharedPreprocessCacheTests(unittest.TestCase):
    maxDiff = 3000

    def setUp(self) -> None:
        vector_rates = {"0.8", "0.55", "0.5", "0.33", "0.25", "0.1"}
        recipes = []
        for rate_id in C.gptoss_parallel.RATES:
            for branch in C.QWEN_BRANCHES:
                if branch == "codec_control":
                    pct = "1.0"
                elif branch == "doctor_static":
                    pct = "0.0" if rate_id in vector_rates else "2.0"
                elif branch == "doctor_conditional":
                    pct = "0.25"
                else:
                    pct = "0.5" if rate_id in vector_rates else "3.0"
                recipes.append({
                    "rate_id": rate_id, "branch": branch, "status": "qualified",
                    "outlier_pct_decimal": pct, "use_rht": True, "rht_cols": True,
                    # Identical authorities are deliberate only inside this synthetic
                    # exact-preprocessing group; production inventories must hash real inputs.
                    "adapter_artifact_sha256": sha(f"adapter-{pct}"),
                    "runtime_spec_authority_sha256": sha(f"runtime-spec-{pct}"),
                    "recipe_authority_sha256": sha(f"recipe-{pct}"),
                })
        self.inventory = {
            "schema": C.QWEN_INVENTORY_SCHEMA,
            "created_at": "2026-07-14T00:00:00+00:00",
            "status": "unbound-input-only",
            "parent_bindings": {"inventory_parent_sha256": sha("parent")},
            "branch_preprocess_recipes": recipes,
            "source_units": [{
                "source_unit_id": "tensor-0001",
                "source_binding_sha256": sha("source"),
                "logical_parameters": 100,
                "estimated_source_read_bytes": 200,
                "estimated_decoded_bytes": 400,
                "preprocess_implementation_sha256": sha("preprocess-program"),
                "rht_seed_sha256": sha("rht-seed"),
                "decode_contract": {
                    "status": "qualified", "exact_serial_parity": True,
                    "implementation_sha256": sha("decoder-program"),
                    "input_dtype": "BF16", "output_dtype": "F32",
                    "layout_sha256": sha("layout"),
                },
            }],
            "source_deletion_permitted": False,
        }
        self.inventory["inventory_sha256"] = C._hash_value(self.inventory)
        self.manifest = C.build_qwen_consumer_manifest(
            self.inventory, created_at="2026-07-14T00:01:00+00:00")
        self.plan = C.build_cache_plan(
            self.manifest, process_budget_bytes=20_000,
            cache_unit_limit_bytes=5_000, disk_reserve_bytes=1_000,
            created_at="2026-07-14T00:02:00+00:00")

    def _resource_receipt(self, unit: dict[str, object]) -> dict[str, object]:
        observed_at = time.time()
        sample = {
            "active_cache_units": 0, "active_reserved_bytes": 0,
            "aggregate_process_tree_rss_bytes": 0,
            "free_disk_bytes": int(unit["estimated_disk_bytes"]) + 1_000,
            "memory_pressure": "normal", "thermal_state": "nominal",
            "observed_at_epoch": observed_at,
        }
        snapshot = {"pressure_level": 1, "swap_used_mb": 0.0}
        previous = C.aggressive_admission.initial_swap_state(
            snapshot, now_epoch=observed_at - 1)
        next_state, reviewed = C.aggressive_admission.advance_swap_state(
            previous, snapshot, now_epoch=observed_at,
            sealed_baseline_swap_mb=0.0)
        decision = {
            "controller_artifact": self.plan["resources"]["swap_controller_artifact"],
            "previous_state": previous, "snapshot": snapshot,
            "now_epoch": observed_at, "sealed_baseline_swap_mb": 0.0,
            "next_state": next_state, "decision": reviewed,
            "resource_sample_identity_sha256": C._hash_value(sample),
        }
        decision["receipt_sha256"] = C._hash_value(decision)
        receipt = C.admit_cache_unit(
            self.plan, self.manifest, cache_unit_id=str(unit["cache_unit_id"]),
            active_cache_units=0, active_reserved_bytes=0,
            aggregate_process_tree_rss_bytes=0,
            free_disk_bytes=sample["free_disk_bytes"], swap_decision=decision,
            memory_pressure="normal", thermal_state="nominal",
            observed_at_epoch=observed_at,
        )
        self.assertTrue(receipt["admitted"])
        return receipt

    def _cache_receipts(self) -> dict[str, dict[str, object]]:
        receipts: dict[str, dict[str, object]] = {}
        for unit_index, unit in enumerate(self.plan["cache_units"]):
            components = []
            input_authority = {
                "consumer_manifest_sha256": self.manifest["manifest_sha256"],
                "source_binding_sha256": unit["source_binding_sha256"],
                "cache_unit_id": unit["cache_unit_id"],
            }
            prior_candidate_sha256s = []
            for stage_index, stage in enumerate(unit["stages"]):
                label = f"cache-{unit_index}-stage-{stage_index}"
                candidate = artifact(f"{label}-candidate")
                stage_input = C._hash_value({
                    "cache_input_authority": input_authority, "stage": stage,
                    "prior_candidate_sha256s": prior_candidate_sha256s,
                })
                components.append({"stage": stage, "artifact": candidate,
                                   "serial_reference": serial_reference(
                                       candidate, label, stage_input),
                                   "byte_exact": True})
                prior_candidate_sha256s.append(candidate["sha256"])
            receipt = C.build_cache_receipt(
                self.plan, self.manifest, cache_unit_id=unit["cache_unit_id"],
                component_artifacts=components,
                cache_artifact=artifact(f"cache-{unit_index}-packed"),
                resource_receipt=self._resource_receipt(unit))
            receipts[unit["cache_unit_id"]] = receipt
        return receipts

    def _output_receipts(self, caches: dict[str, dict[str, object]]) \
            -> dict[str, dict[str, object]]:
        receipts: dict[str, dict[str, object]] = {}
        for index, consumer in enumerate(sorted(
                self.manifest["consumers"], key=lambda row: row["consumer_id"])):
            refs = [{"cache_unit_id": cache_id,
                     "receipt_sha256": caches[cache_id]["receipt_sha256"]}
                    for cache_id in C._consumer_route(self.manifest, consumer)]
            candidate = artifact(f"output-{index}-candidate")
            input_authority = {
                "consumer_id": consumer["consumer_id"],
                "source_binding_sha256": next(
                    row["source_binding_sha256"] for row in self.manifest["source_units"]
                    if row["source_unit_id"] == consumer["source_unit_id"]),
                "cache_receipt_refs": refs,
            }
            receipts[consumer["consumer_id"]] = C.build_output_receipt(
                self.plan, self.manifest, consumer_id=consumer["consumer_id"],
                cache_receipt_refs=refs, output_artifact=candidate,
                scientific_receipt=artifact(f"output-{index}-science"),
                serial_reference=serial_reference(
                    candidate, f"output-{index}", C._hash_value(input_authority)))
        return receipts

    def test_qwen_exact_matrix_and_signature_partition(self) -> None:
        self.assertEqual(C.validate_qwen_inventory(self.inventory), [])
        self.assertEqual(C.validate_consumer_manifest(self.manifest), [])
        self.assertEqual(C.validate_cache_plan(self.plan, self.manifest), [])
        self.assertEqual(self.manifest["coverage"]["consumer_count"], 40)
        self.assertEqual(self.plan["coverage"][
            "unique_evidence_namespace_count"], 40)
        # Six exact outlier signatures plus one base source/decode/statistics unit.
        self.assertEqual(len(self.plan["cache_units"]), 7)
        derived = [row for row in self.plan["cache_units"]
                   if row["kind"] == "exact-bulk-rht-group"]
        self.assertEqual({row["preprocess_key"]["outlier_pct_decimal"]
                          for row in derived}, {"0.0", "0.25", "0.5", "1.0", "2.0", "3.0"})
        for unit in derived:
            expected_ids = sorted(
                row["consumer_id"] for row in self.manifest["consumers"]
                if C._consumer_route(self.manifest, row)[-1] == unit["cache_unit_id"])
            self.assertEqual(unit["consumer_ids_sha256"], C._hash_value(expected_ids))
            self.assertIn("forward_rht", unit["stages"])

    def test_signature_tamper_cannot_coalesce_different_outlier_masks(self) -> None:
        tampered = copy.deepcopy(self.plan)
        derived = next(row for row in tampered["cache_units"]
                       if row["kind"] == "exact-bulk-rht-group")
        derived["preprocess_key"]["outlier_pct_decimal"] = "9.0"
        tampered["cache_plan_sha256"] = C._hash_value(
            C._without(tampered, "cache_plan_sha256"))
        self.assertIn("cache unit grouping differs from exact shareability signatures",
                      C.validate_cache_plan(tampered, self.manifest))
        gate_tamper = copy.deepcopy(self.plan)
        gate_tamper["serial_equivalence_gate"][
            "full_no_skip_coverage_required"] = False
        gate_tamper["cache_plan_sha256"] = C._hash_value(
            C._without(gate_tamper, "cache_plan_sha256"))
        self.assertIn("cache plan serial-equivalence gate differs",
                      C.validate_cache_plan(gate_tamper, self.manifest))

    def test_qwen_missing_matrix_and_authority_tamper_fail_closed(self) -> None:
        incomplete = copy.deepcopy(self.inventory)
        incomplete["branch_preprocess_recipes"].pop()
        incomplete["inventory_sha256"] = C._hash_value(
            C._without(incomplete, "inventory_sha256"))
        self.assertIn("Qwen preprocess authority must cover the exact canonical 10x4 matrix",
                      C.validate_qwen_inventory(incomplete))
        with self.assertRaisesRegex(C.SharedCacheError, "exact canonical 10x4"):
            C.build_qwen_consumer_manifest(incomplete)
        manifest = copy.deepcopy(self.manifest)
        manifest["consumers"][0]["preprocess_signature"][
            "recipe_authority_sha256"] = sha("tampered-recipe-authority")
        manifest["manifest_sha256"] = C._hash_value(
            C._without(manifest, "manifest_sha256"))
        tampered_plan = copy.deepcopy(self.plan)
        tampered_plan["consumer_manifest_sha256"] = manifest["manifest_sha256"]
        tampered_plan["cache_plan_sha256"] = C._hash_value(
            C._without(tampered_plan, "cache_plan_sha256"))
        self.assertIn("cache unit grouping differs from exact shareability signatures",
                      C.validate_cache_plan(tampered_plan, manifest))
        bad_manifest = copy.deepcopy(self.manifest)
        bad_manifest["consumers"][0]["preprocess_signature"][
            "adapter_artifact_sha256"] = "not-a-sha"
        bad_manifest["manifest_sha256"] = C._hash_value(
            C._without(bad_manifest, "manifest_sha256"))
        self.assertTrue(any("qualified preprocess signature" in error
                            for error in C.validate_consumer_manifest(bad_manifest)))

    def test_missing_decoder_fails_closed_to_source_read_only(self) -> None:
        inventory = copy.deepcopy(self.inventory)
        inventory["source_units"][0]["decode_contract"] = {
            "status": "missing", "exact_serial_parity": False,
            "blocker": "decoder not independently qualified",
        }
        inventory["inventory_sha256"] = C._hash_value(
            C._without(inventory, "inventory_sha256"))
        manifest = C.build_qwen_consumer_manifest(inventory)
        plan = C.build_cache_plan(manifest, process_budget_bytes=20_000,
                                  cache_unit_limit_bytes=5_000,
                                  disk_reserve_bytes=1_000)
        self.assertEqual(len(plan["cache_units"]), 1)
        self.assertEqual(plan["cache_units"][0]["stages"],
                         ["immutable_source_range_read"])
        self.assertEqual(plan["coverage"][
            "consumers_without_qualified_derived_preprocess"], 40)

    def test_admission_is_disk_ram_and_sealed_controller_state_bound(self) -> None:
        unit = self.plan["cache_units"][0]
        observed_at = time.time()
        sample = {
            "active_cache_units": 0, "active_reserved_bytes": 0,
            "aggregate_process_tree_rss_bytes": 0,
            "free_disk_bytes": unit["estimated_disk_bytes"] + 1_000,
            "memory_pressure": "normal", "thermal_state": "nominal",
            "observed_at_epoch": observed_at,
        }
        snapshot = {"pressure_level": 1, "swap_used_mb": 0.0}
        previous = C.aggressive_admission.initial_swap_state(
            snapshot, now_epoch=observed_at - 1)
        next_state, reviewed = C.aggressive_admission.advance_swap_state(
            previous, snapshot, now_epoch=observed_at, sealed_baseline_swap_mb=0.0)
        decision = {
            "controller_artifact": self.plan["resources"]["swap_controller_artifact"],
            "previous_state": previous, "snapshot": snapshot,
            "now_epoch": observed_at, "sealed_baseline_swap_mb": 0.0,
            "next_state": next_state, "decision": reviewed,
            "resource_sample_identity_sha256": C._hash_value(sample),
        }
        decision["receipt_sha256"] = C._hash_value(decision)
        args = dict(cache_unit_id=unit["cache_unit_id"],
                    active_cache_units=sample["active_cache_units"],
                    active_reserved_bytes=sample["active_reserved_bytes"],
                    aggregate_process_tree_rss_bytes=sample[
                        "aggregate_process_tree_rss_bytes"],
                    free_disk_bytes=sample["free_disk_bytes"],
                    swap_decision=decision, memory_pressure="normal",
                    thermal_state="nominal", observed_at_epoch=observed_at)
        self.assertTrue(C.admit_cache_unit(self.plan, self.manifest, **args)["admitted"])
        low_disk = {**args, "free_disk_bytes": unit["estimated_disk_bytes"] + 999}
        low_disk["swap_decision"] = copy.deepcopy(decision)
        low_sample = {key: low_disk[key] for key in (
            "active_cache_units", "active_reserved_bytes",
            "aggregate_process_tree_rss_bytes", "free_disk_bytes",
            "memory_pressure", "thermal_state", "observed_at_epoch")}
        low_disk["swap_decision"]["resource_sample_identity_sha256"] = C._hash_value(
            low_sample)
        low_disk["swap_decision"]["receipt_sha256"] = C._hash_value(
            C._without(low_disk["swap_decision"], "receipt_sha256"))
        self.assertFalse(C.admit_cache_unit(self.plan, self.manifest,
                                            **low_disk)["admitted"])
        unsealed = copy.deepcopy(args)
        unsealed["swap_decision"]["decision"]["allow_launch"] = False
        self.assertFalse(C.admit_cache_unit(self.plan, self.manifest,
                                            **unsealed)["admitted"])
        wrong_controller = copy.deepcopy(args)
        wrong_controller["swap_decision"]["controller_artifact"]["sha256"] = sha("wrong")
        wrong_controller["swap_decision"]["receipt_sha256"] = C._hash_value(
            C._without(wrong_controller["swap_decision"], "receipt_sha256"))
        self.assertFalse(C.admit_cache_unit(self.plan, self.manifest,
                                            **wrong_controller)["admitted"])

    def test_independent_serial_oracle_full_merge_and_resume(self) -> None:
        caches = self._cache_receipts()
        outputs = self._output_receipts(caches)
        merge = C.build_merge_manifest(
            self.plan, self.manifest, cache_receipts=caches.values(),
            output_receipts=outputs.values())
        self.assertEqual(merge["coverage"],
                         {"scheduled": 40, "executed": 40,
                          "validated": 40, "skipped": 0})
        self.assertEqual(merge["ephemeral_cache_gc_eligible"], sorted(caches))
        self.assertFalse(merge["parent_sources_deleted"])
        state = C.reconcile_resume_state(
            self.plan, self.manifest,
            C.build_resume_state(self.plan, self.manifest),
            cache_receipts=caches.values(), output_receipts=outputs.values())
        self.assertTrue(all(row["status"] == "complete"
                            for row in state["cache_units"].values()))
        self.assertTrue(all(row["status"] == "complete"
                            for row in state["outputs"].values()))
        first_cache_id = sorted(caches)[0]
        replacement_components = copy.deepcopy(
            caches[first_cache_id]["component_artifacts"])
        for index, component in enumerate(replacement_components):
            component["artifact"]["artifact_instance_id"] = f"replacement-candidate-{index}"
            component["serial_reference"]["artifact"][
                "artifact_instance_id"] = f"replacement-serial-{index}"
            component["serial_reference"]["semantic_receipt"][
                "artifact_instance_id"] = f"replacement-semantic-{index}"
        replacement = C.build_cache_receipt(
            self.plan, self.manifest, cache_unit_id=first_cache_id,
            component_artifacts=replacement_components,
            cache_artifact=artifact("replacement-cache-artifact"),
            resource_receipt=caches[first_cache_id]["resource_receipt"])
        replaced_caches = dict(caches); replaced_caches[first_cache_id] = replacement
        with self.assertRaisesRegex(C.SharedCacheError, "identical receipt"):
            C.reconcile_resume_state(
                self.plan, self.manifest, state,
                cache_receipts=replaced_caches.values(),
                output_receipts=outputs.values())

    def test_serial_self_assertion_alias_missing_and_refcount_fail_closed(self) -> None:
        caches = self._cache_receipts()
        outputs = self._output_receipts(caches)
        first_id, second_id = sorted(outputs)[:2]
        first = outputs[first_id]
        second_consumer = next(row for row in self.manifest["consumers"]
                               if row["consumer_id"] == second_id)
        refs = [{"cache_unit_id": cache_id,
                 "receipt_sha256": caches[cache_id]["receipt_sha256"]}
                for cache_id in C._consumer_route(self.manifest, second_consumer)]
        aliased_candidate = artifact(str(first["output_artifact"]["artifact_instance_id"]))
        alias_input = {"consumer_id": second_id,
                       "source_binding_sha256": self.manifest["source_units"][0][
                           "source_binding_sha256"], "cache_receipt_refs": refs}
        aliased = C.build_output_receipt(
            self.plan, self.manifest, consumer_id=second_id,
            cache_receipt_refs=refs, output_artifact=aliased_candidate,
            scientific_receipt=artifact("alias-science"),
            serial_reference=serial_reference(
                aliased_candidate, "alias", C._hash_value(alias_input)))
        alias_outputs = dict(outputs); alias_outputs[second_id] = aliased
        with self.assertRaisesRegex(C.SharedCacheError, "aliased output evidence"):
            C.build_merge_manifest(self.plan, self.manifest,
                                   cache_receipts=caches.values(),
                                   output_receipts=alias_outputs.values())
        cross_role_candidate = artifact("cross-role-output")
        cross_role_science = artifact(str(first["output_artifact"][
            "artifact_instance_id"]))
        cross_role = C.build_output_receipt(
            self.plan, self.manifest, consumer_id=second_id,
            cache_receipt_refs=refs, output_artifact=cross_role_candidate,
            scientific_receipt=cross_role_science,
            serial_reference=serial_reference(
                cross_role_candidate, "cross-role", C._hash_value(alias_input)))
        cross_outputs = dict(outputs); cross_outputs[second_id] = cross_role
        with self.assertRaisesRegex(C.SharedCacheError, "aliased output evidence"):
            C.build_merge_manifest(self.plan, self.manifest,
                                   cache_receipts=caches.values(),
                                   output_receipts=cross_outputs.values())
        missing = dict(outputs); missing.pop(first_id)
        with self.assertRaisesRegex(C.SharedCacheError, "missing/extra"):
            C.build_merge_manifest(self.plan, self.manifest,
                                   cache_receipts=caches.values(),
                                   output_receipts=missing.values())
        candidate = artifact("self-candidate")
        self_input = {"consumer_id": first_id,
                      "source_binding_sha256": self.manifest["source_units"][0][
                          "source_binding_sha256"],
                      "cache_receipt_refs": first["cache_receipt_refs"]}
        self_ref = serial_reference(candidate, "self", C._hash_value(self_input))
        self_ref["artifact"]["artifact_instance_id"] = candidate["artifact_instance_id"]
        with self.assertRaisesRegex(C.SharedCacheError, "serial artifact identity"):
            C.build_output_receipt(
                self.plan, self.manifest, consumer_id=first_id,
                cache_receipt_refs=first["cache_receipt_refs"],
                output_artifact=candidate, scientific_receipt=artifact("self-science"),
                serial_reference=self_ref)
        incomplete_output = copy.deepcopy(first)
        incomplete_output["status"] = "running"
        incomplete_output["receipt_sha256"] = C._hash_value(
            C._without(incomplete_output, "receipt_sha256"))
        self.assertIn("output receipt is not complete",
                      C.validate_output_receipt(self.plan, self.manifest,
                                                incomplete_output))
        first_cache = next(iter(caches.values()))
        incomplete_cache = copy.deepcopy(first_cache)
        incomplete_cache["status"] = "running"
        incomplete_cache["receipt_sha256"] = C._hash_value(
            C._without(incomplete_cache, "receipt_sha256"))
        self.assertIn("cache receipt is not complete",
                      C.validate_cache_receipt(self.plan, self.manifest,
                                               incomplete_cache))
        cache_ids = sorted(caches)
        first_cache, second_cache = caches[cache_ids[0]], caches[cache_ids[1]]
        cross_cache_components = copy.deepcopy(second_cache["component_artifacts"])
        cross_cache_components[0]["serial_reference"]["artifact"][
            "artifact_instance_id"] = first_cache["component_artifacts"][0][
                "artifact"]["artifact_instance_id"]
        cross_cache = C.build_cache_receipt(
            self.plan, self.manifest, cache_unit_id=cache_ids[1],
            component_artifacts=cross_cache_components,
            cache_artifact=second_cache["cache_artifact"],
            resource_receipt=second_cache["resource_receipt"])
        cross_caches = dict(caches); cross_caches[cache_ids[1]] = cross_cache
        with self.assertRaisesRegex(C.SharedCacheError,
                                    "invalid/duplicate cache receipt"):
            C.build_merge_manifest(self.plan, self.manifest,
                                   cache_receipts=cross_caches.values(),
                                   output_receipts=outputs.values())

    def test_malformed_component_and_cache_ref_elements_refuse_cleanly(self) -> None:
        caches = self._cache_receipts()
        outputs = self._output_receipts(caches)
        output_id = sorted(outputs)[0]
        cache_id = sorted(caches)[0]

        malformed_output = copy.deepcopy(outputs[output_id])
        malformed_output["cache_receipt_refs"].append(None)
        malformed_output["receipt_sha256"] = C._hash_value(
            C._without(malformed_output, "receipt_sha256"))
        output_errors = C.validate_output_receipt(
            self.plan, self.manifest, malformed_output)
        self.assertIn("output receipt cache route differs", output_errors)
        malformed_outputs = dict(outputs); malformed_outputs[output_id] = malformed_output
        with self.assertRaisesRegex(C.SharedCacheError,
                                    "invalid/aliased output evidence at merge"):
            C.build_merge_manifest(
                self.plan, self.manifest, cache_receipts=caches.values(),
                output_receipts=malformed_outputs.values())

        consumer = next(row for row in self.manifest["consumers"]
                        if row["consumer_id"] == output_id)
        valid_refs = outputs[output_id]["cache_receipt_refs"]
        candidate = artifact("malformed-ref-candidate")
        with self.assertRaisesRegex(C.SharedCacheError,
                                    "output cache receipt route differs"):
            C.build_output_receipt(
                self.plan, self.manifest, consumer_id=consumer["consumer_id"],
                cache_receipt_refs=[*valid_refs, None], output_artifact=candidate,
                scientific_receipt=artifact("malformed-ref-science"),
                serial_reference={})

        malformed_cache = copy.deepcopy(caches[cache_id])
        malformed_cache["component_artifacts"].append(None)
        malformed_cache["receipt_sha256"] = C._hash_value(
            C._without(malformed_cache, "receipt_sha256"))
        cache_errors = C.validate_cache_receipt(
            self.plan, self.manifest, malformed_cache)
        self.assertIn("cache receipt component coverage differs", cache_errors)
        malformed_caches = dict(caches); malformed_caches[cache_id] = malformed_cache
        with self.assertRaisesRegex(C.SharedCacheError,
                                    "invalid/duplicate cache receipt at merge"):
            C.build_merge_manifest(
                self.plan, self.manifest, cache_receipts=malformed_caches.values(),
                output_receipts=outputs.values())

        forged_resource = copy.deepcopy(caches[cache_id])
        forged_resource["resource_receipt"] = "a" * 64
        forged_resource["resource_receipt_sha256"] = "a" * 64
        forged_resource["receipt_sha256"] = C._hash_value(
            C._without(forged_resource, "receipt_sha256"))
        self.assertIn(
            "cache receipt artifact/resource/refcount differs",
            C.validate_cache_receipt(self.plan, self.manifest, forged_resource),
        )
        promoted = copy.deepcopy(caches[cache_id])
        promoted["production_activation_permitted"] = True
        promoted["resource_evidence_scope"] = C.PRODUCTION_RESOURCE_SCOPE
        promoted["receipt_sha256"] = C._hash_value(
            C._without(promoted, "receipt_sha256"))
        self.assertIn(
            "cache receipt artifact/resource/refcount differs",
            C.validate_cache_receipt(self.plan, self.manifest, promoted),
        )
        production_resource = copy.deepcopy(caches[cache_id]["resource_receipt"])
        production_resource["scope"] = C.PRODUCTION_RESOURCE_SCOPE
        production_resource["receipt_sha256"] = C._hash_value(
            C._without(production_resource, "receipt_sha256"))
        with self.assertRaisesRegex(C.SharedCacheError,
                                    "cache/resource artifact identity is invalid"):
            C.build_cache_receipt(
                self.plan, self.manifest, cache_unit_id=cache_id,
                component_artifacts=caches[cache_id]["component_artifacts"],
                cache_artifact=artifact("forged-production-resource"),
                resource_receipt=production_resource,
            )

        unit = next(row for row in self.plan["cache_units"]
                    if row["cache_unit_id"] == cache_id)
        with self.assertRaisesRegex(C.SharedCacheError,
                                    "cache component stage coverage differs"):
            C.build_cache_receipt(
                self.plan, self.manifest, cache_unit_id=cache_id,
                component_artifacts=[*caches[cache_id]["component_artifacts"], None],
                cache_artifact=artifact("malformed-component-cache"),
                resource_receipt=caches[cache_id]["resource_receipt"])

        extra_key_output = copy.deepcopy(outputs[output_id])
        extra_key_output["cache_receipt_refs"][0]["unexpected"] = True
        extra_key_output["receipt_sha256"] = C._hash_value(
            C._without(extra_key_output, "receipt_sha256"))
        self.assertIn(
            "output receipt cache route differs",
            C.validate_output_receipt(self.plan, self.manifest, extra_key_output))

    def test_unhashable_resource_labels_refuse_without_crashing(self) -> None:
        caches = self._cache_receipts()
        cache_id = sorted(caches)[0]
        malformed = copy.deepcopy(caches[cache_id])
        resource = malformed["resource_receipt"]
        resource["resource_sample"]["thermal_state"] = []
        resource["resource_sample_identity_sha256"] = C._hash_value(
            resource["resource_sample"])
        resource["receipt_sha256"] = C._hash_value(
            C._without(resource, "receipt_sha256"))
        malformed["resource_receipt_sha256"] = resource["receipt_sha256"]
        malformed["receipt_sha256"] = C._hash_value(
            C._without(malformed, "receipt_sha256"))
        self.assertIn(
            "cache receipt artifact/resource/refcount differs",
            C.validate_cache_receipt(self.plan, self.manifest, malformed),
        )

        unit = next(row for row in self.plan["cache_units"]
                    if row["cache_unit_id"] == cache_id)
        refused = C.admit_cache_unit(
            self.plan, self.manifest, cache_unit_id=cache_id,
            active_cache_units=0, active_reserved_bytes=0,
            aggregate_process_tree_rss_bytes=0,
            free_disk_bytes=int(unit["estimated_disk_bytes"]) + 1_000,
            swap_decision={}, memory_pressure="normal", thermal_state=[],
            observed_at_epoch=time.time(),
        )
        self.assertFalse(refused["admitted"])
        self.assertEqual(refused["reason"], "invalid plan/unit/resource sample")

    def test_requirements_and_symlink_hashing_are_fail_closed(self) -> None:
        packet = C.build_requirements_packet(created_at="2026-07-14T00:03:00+00:00")
        self.assertFalse(packet["cache_shape"]["whole_model_cache_permitted"])
        self.assertEqual(packet["cache_shape"]["maximum_active_cache_units_default"], 1)
        self.assertTrue(packet["benchmark_contract"][
            "exact_input_cache_output_scientific_receipt_identities"])
        self.assertFalse(packet["execution_permitted"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"; real.write_text("x", encoding="utf-8")
            link = root / "link"; link.symlink_to(real)
            with self.assertRaisesRegex(C.SharedCacheError, "non-symlink"):
                C._file_artifact(link)
            with self.assertRaisesRegex(C.SharedCacheError, "unbound root"):
                C._write_json(root / "escape.json", {"unsafe": True})


if __name__ == "__main__":
    unittest.main()
