#!/usr/bin/env python3.12
"""Synthetic-fixture tests for the GPT-OSS-120B vulture harvest generator.

Everything runs against a temp evidence tree of tiny synthetic JSON blobs that mirror
the real general_frontier layout. No real 120B evidence is read, no model is loaded,
no forward is run. The tests prove:

  * generate + write emits ALL 8 harvest artifacts (7 JSON + 1 MD)
  * every salvaged prior carries the full salvage-confidence block
  * every result classification is a member of the 4-way (+REFERENCE) enum
  * the organ-sensitivity INVERSION is detected when mlp1-only hurts more on both probes
  * PROVISIONAL vs FINAL provenance is driven by campaign_final + G4 presence
  * classify_result covers CAPABILITY_PASS / HONEST_BOUNDARY / TRANSFERABLE_PARTIAL /
    INVALID / REFERENCE branches
  * missing evidence degrades gracefully (still emits, records the gap, stays PROVISIONAL)
"""
from __future__ import annotations

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import vulture_harvest as vh  # noqa: E402

SALVAGE_FIELDS = {
    "evidence_fidelity",
    "sample_scope",
    "parent_specificity",
    "architecture_specificity",
    "confidence",
    "expected_transfer_direction",
    "falsification_test",
}

DOCTOR_PARAMS = {
    "budget_frac": 0.01, "dim": 16, "doctor": "residual_codebook",
    "doctor_bpw": 0.15, "k": 64, "stages": 2, "strategy": "residual_energy", "subspaces": 2,
}


def _write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def _parent_ckpt(pid: str, ppl: float) -> dict:
    return {
        "kind": "parent", "variant": "parent", "prompt_id": pid,
        "row_id": f"{pid}__D0_parent", "forward_seconds": 300.0, "logits_finite": True,
        "quality": {"n_positions": 4, "nll": 1.0, "perplexity": ppl},
    }


def _diag_ckpt(pid: str, variant: str, agr: float, kl: float, verdict: str,
               per_expert_bpw: float, mlp1_fam: str, mlp2_fam: str) -> dict:
    return {
        "kind": "diagnosis", "variant": variant, "prompt_id": pid,
        "row_id": f"{pid}__{variant}", "forward_seconds": 400.0, "logits_finite": True,
        "verdict": verdict,
        "divergence_vs_parent": {
            "mean_logit_cosine": 0.95, "mean_sym_kl": kl,
            "mean_top5_overlap": 0.4, "next_token_argmax_agreement": agr,
        },
        "budget": {
            "mlp1_class": {"family": mlp1_fam, "n_weights": 16588800, "whole_bpw": 0.75},
            "mlp2_class": {"family": mlp2_fam, "n_weights": 8294400, "whole_bpw": 0.75},
            "per_expert_whole_bpw": per_expert_bpw,
        },
        "quality": {"n_positions": 4, "nll": 4.0, "perplexity": 100.0},
    }


def _d4_ckpt(pid: str, agr: float, kl: float) -> dict:
    return {
        "kind": "candidate", "variant": "D4_pq_doctor", "prompt_id": pid,
        "row_id": f"{pid}__D4_pq_doctor", "forward_seconds": 1500.0, "logits_finite": True,
        "verdict": "collapse",
        "mapping": {"mlp1": {"family": "pq_doctor_lowrank", "params": DOCTOR_PARAMS},
                    "mlp2": {"family": "pq_protected_islands", "params": DOCTOR_PARAMS}},
        "params": DOCTOR_PARAMS,
        "divergence_vs_parent": {
            "mean_logit_cosine": 0.9, "mean_sym_kl": kl,
            "mean_top5_overlap": 0.2, "next_token_argmax_agreement": agr,
        },
        "budget": {
            "mlp1_class": {"family": "pq_doctor_lowrank", "n_weights": 16588800, "whole_bpw": 0.87608},
            "mlp2_class": {"family": "pq_protected_islands", "n_weights": 8294400, "whole_bpw": 0.91319},
            "per_expert_whole_bpw": 0.88845,
        },
        "quality": {"n_positions": 4, "nll": 5.0, "perplexity": 365.0},
    }


def build_synthetic_tree(root: str, *, campaign_final: bool = False,
                         include_g4: bool = True, inversion: bool = True) -> dict:
    """Create a general_frontier evidence tree + sibling second_light dir. Returns paths."""
    er = os.path.join(root, "general_frontier")
    sl = os.path.join(root, "second_light", "GPT_OSS_120B_SECOND_LIGHT_BASELINE.json")
    dc = os.path.join(er, "DOCTOR_CAMPAIGN")
    ck = os.path.join(dc, "checkpoints")

    # Two probes, each with parent + both diagnosis variants.
    for pid, ppl in (("gen_paris", 27.4325), ("reason_syllogism", 3.4641)):
        _write(os.path.join(ck, f"{pid}__D0_parent.json"), _parent_ckpt(pid, ppl))
        if inversion:
            # mlp1_only hurts MORE (lower agreement, higher KL) than mlp2_only.
            _write(os.path.join(ck, f"{pid}__diag_mlp1_only.json"),
                   _diag_ckpt(pid, "diag_mlp1_only", 0.40, 1.37, "degraded", 5.83401,
                              "product_quant", "kept_original"))
            _write(os.path.join(ck, f"{pid}__diag_mlp2_only.json"),
                   _diag_ckpt(pid, "diag_mlp2_only", 0.60, 0.63, "degraded", 10.91735,
                              "kept_original", "product_quant"))
        else:
            # proxy direction: mlp2_only hurts more.
            _write(os.path.join(ck, f"{pid}__diag_mlp1_only.json"),
                   _diag_ckpt(pid, "diag_mlp1_only", 0.70, 0.50, "degraded", 5.83401,
                              "product_quant", "kept_original"))
            _write(os.path.join(ck, f"{pid}__diag_mlp2_only.json"),
                   _diag_ckpt(pid, "diag_mlp2_only", 0.40, 1.50, "degraded", 10.91735,
                              "kept_original", "product_quant"))
        _write(os.path.join(ck, f"{pid}__D4_pq_doctor.json"), _d4_ckpt(pid, 0.2, 2.5))

    _write(os.path.join(dc, "DOCTOR_CAMPAIGN_STATE.json"), {
        "schema": "hawking.doctor_campaign.state.v1",
        "final": campaign_final, "rows_done": 28 if campaign_final else 14, "rows_total": 28,
        "params": DOCTOR_PARAMS,
    })
    _write(os.path.join(dc, "GPT_OSS_120B_D3_D5_NON_ADMISSION.json"), {
        "schema": "hawking.gpt_oss_120b.doctor_ladder_non_admission.v1",
        "D3_non_admission": {"verdict": "DOMINATED / REDUNDANT"},
        "D5_non_admission": {"verdict": "DOMINATED"},
        "coverage_conclusion": "D3 and D5 are dominated and not admitted.",
    })

    control_rows = [
        {"row": "gen_paris__rvq1.0", "mean_sym_kl": 1.84, "next_token_agreement": 0.20, "ppl": 179.9, "verdict": "degraded"},
        {"row": "gen_science__rvq1.0", "mean_sym_kl": 3.49, "next_token_agreement": 0.11, "ppl": 196.8, "verdict": "collapse"},
        {"row": "code_py__rvq1.0", "mean_sym_kl": 1.86, "next_token_agreement": 0.63, "ppl": 6.5, "verdict": "degraded"},
    ]
    parent_ppl = {"gen_paris": 27.4325, "reason_syllogism": 3.4641, "code_py": 1.9202}

    if include_g4:
        _write(os.path.join(er, "GPT_OSS_120B_G4_RESULT.json"), {
            "schema": "hawking.gpt_oss_120b.g4_result.v1", "status": "COMPLETE",
            "verdict": "NEGATIVE at real fidelity: uniform sub-bit RVQ@1.0 preserves no capability.",
            "parent_real_perplexity": parent_ppl,
            "packed_control_rvq_1bpw": control_rows,
            "promote_thresholds": {"mean_sym_kl_max": 0.1, "next_token_argmax_agreement_min": 0.95},
        })
        _write(os.path.join(er, "GPT_OSS_120B_G4_UNTREATED_CONTROL.json"), {
            "schema": "hawking.gpt_oss_120b.g4_untreated_control.v1",
            "parent_real_perplexity": parent_ppl,
            "untreated_rvq_1bpw": control_rows,
            "source_revision": "openai/gpt-oss-120b @ b5c939de",
        })

    _write(os.path.join(er, "G3", "G3_TRANSFER.json"), {
        "schema": "hawking.frontier_g3.transfer.v1",
        "transfer_by_tensor_class": {
            "expert_mlp1": {"fully_transfers": False, "winner_family_per_layer":
                            {"0": "pq_doctor_lowrank", "18": "pq_doctor_lowrank", "35": "pq_protected_islands"}},
            "expert_mlp2": {"fully_transfers": True, "winner_family_per_layer":
                            {"0": "pq_protected_islands", "18": "pq_protected_islands", "35": "pq_protected_islands"}},
        },
    })
    _write(os.path.join(er, "GENERAL_FRONTIER_RESULTS", "GATE_F_G1_RESULT.json"), {
        "schema": "hawking.general_frontier.g1_result.v1",
        "winners": {"mlp1": {"candidate": "A3_pq_islands", "validation_div": 0.00534},
                    "mlp2": {"candidate": "B1_pq_islands", "validation_div": 0.14919}},
    })
    _write(os.path.join(er, "GENERAL_FRONTIER_RESULTS", "GATE_F_G0_RESULT.json"), {
        "schema": "hawking.general_frontier.g0_result.v1", "verdict": "G0 PASS",
    })
    _write(os.path.join(er, "GPT_OSS_120B_SOURCE_RELEASE_READINESS.json"), {
        "schema": "hawking.gpt_oss_120b.source_release_readiness.v1",
        "release_authorized": False, "release_decision": "DENIED",
        "source_root": "/models/gpt-oss-120b",
        "disk": {"free_gib": 539.7, "release_reclaims_gib": 60.8, "free_after_release_gib": 600.5},
        "gates": {
            "01_exact_root_identified": {"source_root": "/models/gpt-oss-120b"},
            "02_immutable_revision_sealed": {"immutable_revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"},
            "04_tokenizer_config_index_retained": {"retained_present": ["tokenizer.json", "config.json"]},
            "14_exact_deletion_paths_listed": {
                "delete_only_these": [
                    {"abs_path": f"/models/gpt-oss-120b/original/model--0000{i}-of-00007.safetensors",
                     "sealed_bytes": 10000000000 + i, "sealed_sha256": f"hash{i:02d}", "exists": True}
                    for i in range(1, 8)
                ],
                "explicitly_retain_do_not_delete": ["/models/gpt-oss-120b/original/config.json"],
            },
            "15_post_release_verification_plan": {"plan": ["re-fetch", "hash-verify", "revalidate"]},
        },
    })

    _write(sl, {
        "schema": "hawking.gpt_oss_120b.second_light_baseline.v1",
        "quality": {"capability_pass": False, "true_residual_output_divergence_mean": 0.68792,
                    "weight_rel_error_mean": 0.554},
        "result": {"budget_bpw": 0.92788, "realized_whole_artifact_bpw": 0.76976,
                   "output_gib": 10.469, "source_gib": 60.77, "rows_sealed": 183, "rows_total": 183},
    })

    return {"evidence_root": er, "second_light": sl, "checkpoints": ck}


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def test_emits_all_eight_artifacts(tmp_path):
    paths = build_synthetic_tree(str(tmp_path))
    out_dir = os.path.join(str(tmp_path), "out")
    objs = vh.generate(paths["evidence_root"], paths["second_light"])
    written = vh.write_artifacts(objs, out_dir)

    assert len(written) == 8
    expected = set(vh.ARTIFACT_FILENAMES.values())
    assert {os.path.basename(w) for w in written} == expected
    for w in written:
        assert os.path.isfile(w) and os.path.getsize(w) > 0
    # 7 JSON parse cleanly, 1 MD is non-empty text.
    n_json = 0
    for w in written:
        if w.endswith(".json"):
            with open(w) as fh:
                json.load(fh)
            n_json += 1
        else:
            assert open(w).read().startswith("# GPT-OSS-120B Vulture Harvest")
    assert n_json == 7


def test_salvage_confidence_on_every_prior(tmp_path):
    paths = build_synthetic_tree(str(tmp_path))
    objs = vh.generate(paths["evidence_root"], paths["second_light"])

    for prior in objs["transfer_priors"]["priors"]:
        assert "salvage_confidence" in prior, prior.get("prior_id")
        assert set(prior["salvage_confidence"].keys()) == SALVAGE_FIELDS
        assert prior["classification"] in vh.VALID_CLASSES

    for finding in objs["doctor_atlas"]["findings"]:
        assert set(finding["salvage_confidence"].keys()) == SALVAGE_FIELDS
        assert finding["classification"] in vh.VALID_CLASSES

    org = objs["failure_atlas"]["organ_sensitivity"]
    assert set(org["salvage_confidence"].keys()) == SALVAGE_FIELDS
    assert org["classification"] in vh.VALID_CLASSES


def test_evidence_classification_valid_and_present(tmp_path):
    paths = build_synthetic_tree(str(tmp_path))
    objs = vh.generate(paths["evidence_root"], paths["second_light"])
    harvest = objs["harvest_json"]

    assert harvest["results_index"], "results_index must be non-empty"
    for r in harvest["results_index"]:
        assert r["classification"] in vh.VALID_CLASSES, r
    # Counts reconcile with the index.
    counts = harvest["evidence_classification_counts"]
    assert sum(counts.values()) == len(harvest["results_index"])
    # Parent rows are REFERENCE; the uniform control rows are HONEST_BOUNDARY.
    assert counts.get(vh.CLASS_REFERENCE, 0) >= 2
    assert counts.get(vh.CLASS_BOUNDARY, 0) >= 1
    assert counts.get(vh.CLASS_PARTIAL, 0) >= 1


def test_organ_sensitivity_inversion_detected(tmp_path):
    paths = build_synthetic_tree(str(tmp_path), inversion=True)
    objs = vh.generate(paths["evidence_root"], paths["second_light"])
    org = objs["failure_atlas"]["organ_sensitivity"]

    assert org["inversion_confirmed"] is True
    assert org["proxy_prior_sensitive_class"] == "mlp2"
    assert org["real_forward_sensitive_class"] == "mlp1"
    assert org["probes_compared"] == 2
    assert org["probes_where_mlp1_only_hurts_more"] == 2
    assert "INVERSION" in org["headline"]
    for pair in org["pairs"]:
        assert pair["mlp1_only_hurts_more"] is True


def test_no_inversion_when_proxy_direction_holds(tmp_path):
    paths = build_synthetic_tree(str(tmp_path), inversion=False)
    objs = vh.generate(paths["evidence_root"], paths["second_light"])
    org = objs["failure_atlas"]["organ_sensitivity"]

    assert org["inversion_confirmed"] is False
    assert org["real_forward_sensitive_class"] == "inconclusive"
    assert org["classification"] == vh.CLASS_INVALID


def test_provisional_when_campaign_not_final(tmp_path):
    paths = build_synthetic_tree(str(tmp_path), campaign_final=False)
    objs = vh.generate(paths["evidence_root"], paths["second_light"])
    prov = objs["harvest_json"]["provenance"]

    assert prov["provenance_status"] == "PROVISIONAL"
    assert prov["campaign_final"] is False
    assert prov["harvest_is_complete"] is False
    assert any("final" in r.lower() for r in prov["reasons"])


def test_final_when_campaign_sealed_and_g4_present(tmp_path):
    paths = build_synthetic_tree(str(tmp_path), campaign_final=True, include_g4=True)
    objs = vh.generate(paths["evidence_root"], paths["second_light"])
    prov = objs["harvest_json"]["provenance"]

    assert prov["provenance_status"] == "FINAL"
    assert prov["campaign_final"] is True
    assert prov["harvest_is_complete"] is True


def test_missing_g4_degrades_gracefully(tmp_path):
    paths = build_synthetic_tree(str(tmp_path), campaign_final=True, include_g4=False)
    out_dir = os.path.join(str(tmp_path), "out")
    objs = vh.generate(paths["evidence_root"], paths["second_light"])
    written = vh.write_artifacts(objs, out_dir)

    # Still emits all 8 artifacts, but provenance is PROVISIONAL and records the gap.
    assert len(written) == 8
    prov = objs["harvest_json"]["provenance"]
    assert prov["provenance_status"] == "PROVISIONAL"
    assert "g4_result" in prov["missing_evidence"]
    assert "g4_control" in prov["missing_evidence"]


def test_classify_result_branches():
    # CAPABILITY_PASS: meets both gate thresholds.
    assert vh.classify_result(kind="candidate", role="treated_candidate",
                              mean_sym_kl=0.05, agreement=0.97, verdict="pass") == vh.CLASS_PASS
    # HONEST_BOUNDARY: negative control below the gate.
    assert vh.classify_result(kind="control", role="negative_control",
                              mean_sym_kl=1.8, agreement=0.2, verdict="collapse") == vh.CLASS_BOUNDARY
    # TRANSFERABLE_PARTIAL: degraded diagnosis, not a control.
    assert vh.classify_result(kind="diagnosis", role="organ_isolation",
                              mean_sym_kl=1.0, agreement=0.6, verdict="degraded") == vh.CLASS_PARTIAL
    # INVALID: not admitted (dominated) regardless of metrics.
    assert vh.classify_result(kind="candidate", role="treated_candidate",
                              mean_sym_kl=0.05, agreement=0.99, verdict="pass",
                              admitted=False) == vh.CLASS_INVALID
    # INVALID: no divergence evidence at all.
    assert vh.classify_result(kind="candidate", role=None,
                              mean_sym_kl=None, agreement=None, verdict=None) == vh.CLASS_INVALID
    # REFERENCE: parent baseline.
    assert vh.classify_result(kind="parent", role=None,
                              mean_sym_kl=None, agreement=None, verdict=None) == vh.CLASS_REFERENCE


def test_rehydration_receipt_has_seven_shards_and_route(tmp_path):
    paths = build_synthetic_tree(str(tmp_path))
    objs = vh.generate(paths["evidence_root"], paths["second_light"])
    rr = objs["rehydration_receipt"]

    assert rr["shard_count"] == 7
    assert len(rr["shards"]) == 7
    assert all(s["abs_path"] and s["sha256"] for s in rr["shards"])
    assert rr["rehydrate_route"].startswith("openai/gpt-oss-120b @ b5c939de")
    assert rr["release_decision"] == "DENIED"
    assert rr["release_authorized"] is False


def test_resource_and_runtime_atlases(tmp_path):
    paths = build_synthetic_tree(str(tmp_path))
    objs = vh.generate(paths["evidence_root"], paths["second_light"])

    res = objs["resource_atlas"]
    assert res["shard_count"] == 7
    assert res["source_gib"] == 60.8
    assert res["byte_budgets"]["d4_per_expert_whole_bpw"] == 0.88845

    rt = objs["runtime_lessons"]
    assert rt["cache_env"]["HAWKING_CACHE_MAX_GB"] == 48
    assert rt["cache_env"]["HAWKING_CACHE_FLOOR_GB"] == 12
    assert rt["cache_env"]["HAWKING_CACHE_DISK_RESERVE_GB"] == 40
    lesson_ids = {l["lesson"] for l in rt["lessons"]}
    assert {"byte_budget_pressure_aware_cache", "available_floor_into_swap_oom",
            "mps_packing_hoard", "per_expert_subbit_bpw", "treated_row_wall_clock"} <= lesson_ids
    # 3 OOM/jetsam crashes recorded.
    oom = next(l for l in rt["lessons"] if l["lesson"] == "available_floor_into_swap_oom")
    assert oom["crash_count"] == 3


def test_cli_dry_run(tmp_path, capsys):
    paths = build_synthetic_tree(str(tmp_path))
    rc = vh.main(["--evidence-root", paths["evidence_root"],
                  "--second-light-baseline", paths["second_light"],
                  "--dry-run", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    summary = json.loads(out)
    assert summary["dry_run"] is True
    assert summary["written_paths"] == []
    assert summary["provenance_status"] == "PROVISIONAL"
    assert len(summary["artifacts"]) == 8


def test_cli_writes_to_out_dir(tmp_path):
    paths = build_synthetic_tree(str(tmp_path))
    out_dir = os.path.join(str(tmp_path), "cli_out")
    rc = vh.main(["--evidence-root", paths["evidence_root"],
                  "--second-light-baseline", paths["second_light"],
                  "--out-dir", out_dir])
    assert rc == 0
    for name in vh.ARTIFACT_FILENAMES.values():
        assert os.path.isfile(os.path.join(out_dir, name))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
