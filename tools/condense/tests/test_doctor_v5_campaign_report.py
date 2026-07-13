#!/usr/bin/env python3.12
"""Contract tests for Doctor-v5 campaign evidence aggregation."""
from __future__ import annotations

import copy
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "doctor_v5_campaign_report.py"
SPEC = importlib.util.spec_from_file_location("doctor_v5_campaign_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha_value(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CampaignReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=reporter.ROOT)
        self.root = Path(self.temporary.name)
        self.campaign = self.root / "campaign.json"
        self.reporting = self.root / "reporting"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def campaign_document(self, *, ultra: bool = False) -> dict:
        schema = "hawking.doctor_v5_ultra_campaign.v1" if ultra else "test.campaign.v1"
        plan_sha = "a" * 64 if ultra else None
        document = {
            "schema": schema,
            "version": "2026-07-13.1",
            "plan_sha256": plan_sha,
            "generated_at": "2026-07-13T00:00:00+00:00",
            "reporting": {"frontier_threshold_b": 120, "max_parallel_cells": 2},
            "cells": [
                {
                    "cell_id": "seven-q4-control", "model_label": "7B",
                    "nominal_params_b": 7.0, "rate_bpw": 4.0,
                    "branch": "codec_control", "claim_track": "codec_fidelity",
                    "required_stages": ["L0", "L1"], "seed_plan": [11],
                    "status": "complete" if ultra else "pending",
                    "result_sha256": "b" * 64 if ultra else None,
                    "disposition_sha256": None,
                },
                {
                    "cell_id": "gptoss-q2-doctor", "model_label": "120B",
                    "nominal_params_b": 116.8, "rate_bpw": 2.0,
                    "branch": "doctor_full", "claim_track": "restorative_training",
                    "required_stages": ["L0"], "seed_plan": [13],
                    "status": "negative" if ultra else "pending",
                    "result_sha256": "c" * 64 if ultra else None,
                    "disposition_sha256": "d" * 64 if ultra else None,
                },
            ],
        }
        if not ultra:
            document.pop("plan_sha256")
        return document

    def test_normalizes_aliases_and_marketing_tier_cutoff(self) -> None:
        write_json(self.campaign, self.campaign_document())
        manifest = reporter.normalize_campaign(self.campaign)
        self.assertEqual(manifest["counts"]["sub_frontier_cells"], 1)
        self.assertEqual(manifest["counts"]["frontier_cells"], 1)
        self.assertEqual([group["group_id"] for group in manifest["report_groups"]],
                         ["sub-120B", "120B"])
        frontier = manifest["cells"][1]
        self.assertEqual(frontier["params_b"], 116.8)
        self.assertEqual(frontier["model_tier_b"], 120.0)
        self.assertEqual(frontier["rate_bpw"], 2.0)

    def test_eight_tier_ultra_matrix_counts_are_derived_as_320(self) -> None:
        models = (("0.5B", 0.5), ("1.5B", 1.5), ("3B", 3.0), ("7B", 7.0),
                  ("14B", 14.0), ("32B", 32.0), ("72B", 72.0), ("120B", 116.8))
        rates = (4.0, 3.0, 2.0, 1.0, 0.8, 0.55, 0.5, 0.33, 0.25, 0.1)
        branches = ("codec_control", "doctor_static", "doctor_conditional", "doctor_full")
        cells = []
        for model, params in models:
            for rate in rates:
                for branch in branches:
                    cells.append({
                        "cell_id": f"{model.lower()}-{str(rate).replace('.', 'p')}-{branch}",
                        "model_label": model, "nominal_params_b": params,
                        "rate_bpw": rate, "branch": branch,
                        "expected_replicates": 1,
                    })
        write_json(self.campaign, {
            "schema": "hawking.doctor_v5_ultra_campaign.v1",
            "version": "test", "plan_sha256": "a" * 64,
            "reporting": {"frontier_threshold_b": 120, "max_parallel_cells": 1},
            "cells": cells,
        })
        manifest = reporter.normalize_campaign(self.campaign)
        self.assertEqual(320, manifest["counts"]["cells"])
        self.assertEqual(280, manifest["counts"]["sub_frontier_cells"])
        self.assertEqual(40, manifest["counts"]["frontier_cells"])

    def test_dynamic_projection_may_change_but_matrix_may_not(self) -> None:
        document = self.campaign_document()
        write_json(self.campaign, document)
        original = reporter.initialize(self.campaign, self.reporting)
        document["generated_at"] = "2026-07-13T00:10:00+00:00"
        document["cells"][0]["status"] = "running"
        write_json(self.campaign, document)
        rebound = reporter._bound_manifest(self.campaign, self.reporting)
        self.assertEqual(rebound["manifest_sha256"], original["manifest_sha256"])
        self.assertNotEqual(rebound["_observed_campaign"]["sha256"],
                            original["campaign"]["sha256"])
        document["cells"][0]["branch"] = "changed-science"
        write_json(self.campaign, document)
        with self.assertRaises(reporter.ReportingError):
            reporter._bound_manifest(self.campaign, self.reporting)

    def test_zero_progress_parameter_tier_expansion_rebinds_additively(self) -> None:
        document = self.campaign_document()
        write_json(self.campaign, document)
        original = reporter.initialize(self.campaign, self.reporting)
        document["cells"].append({
            "cell_id": "three-q4-control", "model_label": "3B",
            "nominal_params_b": 3.0, "rate_bpw": 4.0,
            "branch": "codec_control", "claim_track": "codec_fidelity",
            "required_stages": ["L0"], "expected_replicates": 1,
            "status": "pending",
        })
        write_json(self.campaign, document)
        expanded = reporter._bound_manifest(self.campaign, self.reporting)
        self.assertEqual(3, expanded["counts"]["cells"])
        self.assertEqual(2, expanded["counts"]["sub_frontier_cells"])
        revision = self.reporting / "manifest_revisions" / (
            f"{original['manifest_sha256']}.json"
        )
        self.assertTrue(revision.is_file())

    def test_actual_280_to_320_rebind_accepts_regenerated_old_bindings(self) -> None:
        old_models = (("0.5B", 0.5), ("1.5B", 1.5), ("7B", 7.0),
                      ("14B", 14.0), ("32B", 32.0), ("72B", 72.0), ("120B", 116.8))
        rates = (4.0, 3.0, 2.0, 1.0, 0.8, 0.55, 0.5, 0.33, 0.25, 0.1)
        branches = ("codec_control", "doctor_static", "doctor_conditional", "doctor_full")

        def cells(models: tuple[tuple[str, float], ...], identity: str) -> list[dict]:
            return [{
                "cell_id": f"{model.lower()}-{str(rate).replace('.', 'p')}-{branch}",
                "model_label": model, "nominal_params_b": params,
                "rate_bpw": rate, "branch": branch,
                "cell_identity_sha256": identity * 64,
                "adapter_id": f"adapter-{identity}",
                "runtime_spec_schema": f"runtime-{identity}.v1",
                "expected_replicates": 1, "status": "pending", "attempts": 0,
            } for model, params in models for rate in rates for branch in branches]

        old_document = {
            "schema": "hawking.doctor_v5_ultra_campaign.v1",
            "version": "old", "plan_sha256": "a" * 64,
            "reporting": {"frontier_threshold_b": 120, "max_parallel_cells": 1},
            "cells": cells(old_models, "1"),
        }
        write_json(self.campaign, old_document)
        old_manifest = reporter.initialize(self.campaign, self.reporting)
        self.assertEqual(280, old_manifest["counts"]["cells"])

        new_document = copy.deepcopy(old_document)
        new_document["version"] = "new"
        new_document["plan_sha256"] = "b" * 64
        for row in new_document["cells"]:
            row["cell_identity_sha256"] = "2" * 64
            row["adapter_id"] = "adapter-2"
            row["runtime_spec_schema"] = "runtime-2.v1"
        new_document["cells"].extend(cells((("3B", 3.0),), "2"))
        write_json(self.campaign, new_document)
        rebound = reporter._bound_manifest(self.campaign, self.reporting)
        self.assertEqual(320, rebound["counts"]["cells"])
        self.assertEqual(280, rebound["counts"]["sub_frontier_cells"])
        self.assertEqual(40, rebound["counts"]["frontier_cells"])
        archived = self.reporting / "manifest_revisions" / (
            f"{old_manifest['manifest_sha256']}.json"
        )
        self.assertTrue(archived.is_file())

    def test_additive_rebind_never_overwrites_checkpointed_state(self) -> None:
        document = self.campaign_document()
        write_json(self.campaign, document)
        reporter.initialize(self.campaign, self.reporting)
        reporter.record_checkpoint(self.campaign, self.reporting, {
            "cell_id": "seven-q4-control", "status": "running",
            "completed_stages": [], "completed_replicates": 0,
        })
        document["cells"].append({
            "cell_id": "three-q4-control", "model_label": "3B",
            "nominal_params_b": 3.0, "rate_bpw": 4.0,
            "branch": "codec_control", "status": "pending", "attempts": 0,
        })
        write_json(self.campaign, document)
        with self.assertRaises(reporter.ReportingError):
            reporter._bound_manifest(self.campaign, self.reporting)

    def test_checkpoint_metrics_completeness_eta_and_report_split(self) -> None:
        write_json(self.campaign, self.campaign_document())
        evidence = self.root / "result.json"
        write_json(evidence, {
            "metrics": {
                "parameter_accounting": {
                    "stored_parameter_count": 7_000_000_000,
                    "quantized_parameter_count": 7_000_000_000,
                    "passthrough_parameter_count": 0,
                },
                "physical_accounting": {
                    "packed_bytes": 3_500_000_000,
                    "model_payload_bytes": 3_500_000_000,
                    "full_bundle_bytes": 3_500_000_123,
                },
                "quality_observation": {
                    "ppl": {"baseline": 10.0, "reconstruction": 11.0},
                    "capability": {"baseline": 0.8, "reconstruction": 0.75},
                },
            },
        })
        checkpoint = reporter.record_checkpoint(self.campaign, self.reporting, {
            "cell_id": "seven-q4-control", "status": "complete",
            "completed_stages": ["L0", "L1"], "completed_replicates": 1,
            "evidence_paths": [{"path": str(evidence), "role": "result"}],
            "timing": {"elapsed_s": 10.0},
            "resource_samples": [{
                "peak_rss_bytes": 100, "memory_pressure_level": 1,
                "swap_used_bytes": 0, "thermal_nominal": True,
                "disk_free_bytes": 999,
            }],
        })
        self.assertTrue(checkpoint["completeness"]["complete"])
        ledger = checkpoint["metric_ledger"]
        self.assertEqual(ledger["bytes"]["doctor"]["value"], None)
        self.assertEqual(ledger["bytes"]["doctor"]["reason"], "not_reported")
        self.assertEqual(ledger["parameters"]["pass_through"]["value"], 0)
        self.assertEqual(ledger["physical_bpw"]["packed"]["value"], 4.0)
        self.assertEqual(ledger["resources"]["swap_used_bytes"]["value"], 0)
        self.assertEqual(checkpoint["pareto"]["disposition"], "undetermined")

        index = reporter.aggregate(self.campaign, self.reporting)
        self.assertEqual(index["summary"]["complete_cells"], 1)
        self.assertEqual(index["summary"]["remaining_cells"], 1)
        self.assertEqual(len(index["reports"]), 2)
        self.assertEqual(reporter.verify(self.campaign, self.reporting, deep=True), [])
        eta_path = Path(index["snapshot_path"]) / index["eta_ledger"]["path"]
        eta = json.loads(eta_path.read_text())
        frontier = next(row for row in eta["cells"] if row["model_label"] == "120B")
        self.assertAlmostEqual(frontier["projected_remaining_s"], 10.0 / 7.0 * 116.8)
        self.assertEqual(frontier["confidence"], "low")

        report_path = Path(index["snapshot_path"]) / next(
            row["path"] for row in index["reports"] if row["group_id"] == "sub-120B"
        )
        report = json.loads(report_path.read_text())
        self.assertIn("retention_policy", report)
        self.assertEqual(report["retention_policy"]["gc_eligible_candidates"], [])
        self.assertFalse(report["retention_policy"]["unknown_defaults_to_retain"])
        self.assertEqual(report["retention_policy"]["retained_cell_ids"], [])

    def test_complete_without_evidence_is_not_complete(self) -> None:
        write_json(self.campaign, self.campaign_document())
        checkpoint = reporter.record_checkpoint(self.campaign, self.reporting, {
            "cell_id": "seven-q4-control", "status": "succeeded",
            "completed_stages": ["L0", "L1"], "completed_replicates": 1,
        })
        self.assertFalse(checkpoint["completeness"]["complete"])
        self.assertIn("hashed_evidence_missing", checkpoint["completeness"]["blockers"])
        self.assertIn("numeric_measurements_missing", checkpoint["completeness"]["blockers"])

    def test_negative_and_unsupported_close_without_fabricated_measurements(self) -> None:
        write_json(self.campaign, self.campaign_document())
        disposition = self.root / "disposition.json"
        write_json(disposition, {"status": "unsupported", "reason": "reviewed blocker"})
        negative = reporter.record_checkpoint(self.campaign, self.reporting, {
            "cell_id": "seven-q4-control", "status": "complete_negative",
            "completed_stages": ["L0", "L1"], "completed_replicates": 0,
            "evidence_paths": [{"path": str(disposition), "role": "disposition"}],
        })
        unsupported = reporter.record_checkpoint(self.campaign, self.reporting, {
            "cell_id": "gptoss-q2-doctor", "status": "unsupported",
            "completed_stages": [], "completed_replicates": 0,
            "evidence_paths": [{"path": str(disposition), "role": "disposition"}],
        })
        self.assertTrue(negative["completeness"]["complete"])
        self.assertTrue(unsupported["completeness"]["complete"])
        self.assertEqual(negative["numeric_measurements"], [])
        self.assertEqual(unsupported["numeric_measurements"], [])

    def test_ultra_chain_retention_keeps_at_most_one_reported_candidate(self) -> None:
        branches = ("codec_control", "doctor_static", "doctor_conditional", "doctor_full")
        document = {
            "schema": "hawking.doctor_v5_ultra_campaign.v1",
            "version": "2026-07-13.1", "plan_sha256": "a" * 64,
            "generated_at": "2026-07-13T00:00:00+00:00",
            "reporting": {"frontier_threshold_b": 120, "max_parallel_cells": 1},
            "cells": [],
        }
        packed_by_rate: dict[float, Path] = {}
        for rate, capability in ((4.0, 0.7), (2.0, 0.8)):
            packed = self.root / f"q{int(rate)}.strand"
            packed.write_bytes(f"packed-{rate}".encode())
            packed_sha, packed_bytes = reporter._hash_stable_file(packed)
            packed_by_rate[rate] = packed
            for branch in branches:
                cell_id = f"gptoss-q{int(rate)}-{branch}"
                result_path = self.root / f"{cell_id}.json"
                metrics = {"completed_replicates": 1}
                result: dict = {"status": "complete", "metrics": metrics}
                if branch == "codec_control":
                    metrics["physical_accounting"] = {
                        "all_in_model_payload_bpw": rate,
                        "model_payload_bytes": int(116_829_156_672 * rate / 8),
                    }
                    result["output_artifacts"] = [{
                        "role": "bundle_shard:00000", "path": str(packed),
                        "sha256": packed_sha, "bytes": packed_bytes,
                    }]
                if branch == "doctor_full":
                    metrics["quality_observation"] = {
                        "capability": {"baseline": 0.9, "reconstruction": capability}
                    }
                write_json(result_path, result)
                result_sha, _ = reporter._hash_stable_file(result_path)
                document["cells"].append({
                    "cell_id": cell_id, "model_label": "120B",
                    "nominal_params_b": 116.8, "rate_bpw": rate,
                    "branch": branch, "claim_track": "codec_fidelity",
                    "required_stages": ["run"], "expected_replicates": 1,
                    "status": "complete", "result_sha256": result_sha,
                    "disposition_sha256": None,
                    "result_paths": {"result": str(result_path)},
                })
        write_json(self.campaign, document)
        synced = reporter.sync_campaign(self.campaign, self.reporting)
        self.assertEqual({}, synced["errors"])
        index = json.loads((self.reporting / "report_index.json").read_text())
        self.assertEqual(2, len(index["retention_decisions"]))
        decisions = [json.loads(Path(row["live_path"]).read_text())
                     for row in index["retention_decisions"]]
        retained = [row for row in decisions if row["decision"]["action"] == "retain_one"]
        discarded = [row for row in decisions if row["decision"]["action"] == "retain_none"]
        self.assertEqual(1, len(retained))
        self.assertEqual(2.0, retained[0]["chain"]["rate_bpw"])
        self.assertEqual(1, len(discarded))
        self.assertEqual(
            str(packed_by_rate[4.0].resolve()),
            discarded[0]["decision"]["deletable_worker_owned_packed_artifacts"][0]["path"],
        )
        self.assertFalse(discarded[0]["gc_contract"]["unknown_defaults_to_retain"])
        self.assertEqual([], reporter.verify(self.campaign, self.reporting, deep=True))

    def test_sync_imports_live_terminal_result_without_inventing_values(self) -> None:
        document = self.campaign_document()
        result = self.root / "cell-result.json"
        write_json(result, {
            "metrics": {"completed_replicates": 1, "loss": 2.5},
            "status": "complete",
        })
        document["cells"][0].update({
            "status": "complete", "result_sha256": reporter._hash_stable_file(result)[0],
            "result_paths": {"result": str(result)},
            "started_at": "2026-07-13T00:00:00+00:00",
            "completed_at": "2026-07-13T00:00:12+00:00",
        })
        write_json(self.campaign, document)
        synced = reporter.sync_campaign(self.campaign, self.reporting)
        self.assertEqual(synced["errors"], {})
        self.assertEqual(synced["imported_cells"], 1)
        manifest = reporter._bound_manifest(self.campaign, self.reporting)
        cell = reporter._cell_map(manifest)["seven-q4-control"]
        checkpoint = json.loads(
            reporter._cell_checkpoint_path(self.reporting, cell).read_text()
        )
        self.assertEqual(checkpoint["status"], "succeeded")
        self.assertEqual(checkpoint["timing"]["elapsed_s"], 12.0)
        self.assertTrue(checkpoint["completeness"]["complete"])
        self.assertIsNone(checkpoint["metric_ledger"]["bytes"]["doctor"]["value"])

    def test_sync_cli_returns_nonzero_when_any_cell_import_fails(self) -> None:
        document = self.campaign_document()
        result = self.root / "invalid-progress-result.json"
        write_json(result, {"status": "complete", "metrics": {"completed_replicates": 1}})
        result_sha, _ = reporter._hash_stable_file(result)
        document["cells"][0].update({
            "status": "complete", "result_sha256": result_sha,
            "result_paths": {"result": str(result)},
            "completed_stages": ["not-the-declared-prefix"],
        })
        write_json(self.campaign, document)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = reporter.main([
                "sync", "--campaign", str(self.campaign),
                "--reporting-root", str(self.reporting),
            ])
        self.assertEqual(1, rc)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertIn("seven-q4-control", payload["errors"])

    def test_ultra_terminal_groups_emit_accept_report_receipts(self) -> None:
        write_json(self.campaign, self.campaign_document(ultra=True))
        index = reporter.aggregate(self.campaign, self.reporting)
        self.assertEqual(len(index["ultra_report_checkpoints"]), 2)
        for reference in index["ultra_report_checkpoints"]:
            path = Path(index["snapshot_path"]) / reference["path"]
            receipt = json.loads(path.read_text())
            self.assertEqual(
                set(receipt), {
                    "schema", "version", "plan_sha256", "group_id",
                    "covered_cells_sha256", "report_artifact", "verified",
                    "source_deletion_permitted", "checkpoint_sha256",
                },
            )
            self.assertTrue(receipt["verified"])
            self.assertFalse(receipt["source_deletion_permitted"])
            expected = copy.deepcopy(receipt)
            expected.pop("checkpoint_sha256")
            self.assertEqual(receipt["checkpoint_sha256"], sha_value(expected))
            artifact = reporter.ROOT / receipt["report_artifact"]["path"]
            digest, size = reporter._hash_stable_file(artifact)
            self.assertEqual((digest, size), (
                receipt["report_artifact"]["sha256"],
                receipt["report_artifact"]["bytes"],
            ))

    def test_missing_required_identity_fails_closed(self) -> None:
        document = self.campaign_document()
        del document["cells"][0]["branch"]
        write_json(self.campaign, document)
        with self.assertRaises(reporter.ReportingError):
            reporter.normalize_campaign(self.campaign)


if __name__ == "__main__":
    unittest.main()
