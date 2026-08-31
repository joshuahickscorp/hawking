"""Negative-science index: ingest, query, and a refusal that actually fires."""
import json

from tools.future import negative_index as ni
from tools.future._common import HARDWARE_FIELDS, RECEIPTS


def test_build_emits_sealed_receipt():
    out = ni.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "NEGATIVE_SCIENCE_INDEX.json"
    assert doc["schema"] == "hawking.future.negative_index.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["coverage"]["n_scars"] >= 50
    assert doc["coverage"]["n_sources"] >= 8
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert all("source_path" in s for s in doc["scars"])


def test_selftest_runs():
    out = ni.selftest()
    doc = json.loads(out.read_text())
    assert doc["schema"] == ni.SCHEMA
    assert doc["seal_sha256"]


def test_unparsed_sources_are_not_dropped():
    scars = ni.ingest()
    unparsed = [s for s in scars if s.parse_status == ni.UNPARSED]
    assert unparsed, "expected at least one UNPARSED source (empty lock files, python implementations)"
    paths = {s.source_path for s in unparsed}
    assert any(p.endswith(".lock") or "graveyard" in p.lower() or p.endswith(".py") for p in paths)
    # Coverage accounts for them instead of silently omitting the path.
    cov = ni.coverage(scars)
    assert cov["n_unparsed"] == len(unparsed)
    covered = {row["path"] for row in cov["by_source"]}
    assert paths <= covered


def test_query_returns_source_path_and_ranks_by_specificity():
    hits = ni.query(
        model="qwen3-235b-a22b",
        organ="gate",
        hypothesis_family="cross_expert_structure",
    )
    assert hits, "cross-expert structure on qwen3-235b must be in the real corpus"
    assert all("source_path" in h for h in hits)
    scores = [h["match_score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    top = hits[0]
    assert top["hypothesis_family"] == "cross_expert_structure"
    assert "qwen3-235b-a22b" in top["models"]
    # A family-only query is a superset and its top score cannot beat a
    # model+organ+family hit that also matched those extra keys.
    family_only = ni.query(hypothesis_family="cross_expert_structure")
    assert len(family_only) >= len(hits)
    assert family_only[0]["match_score"] <= top["match_score"] or "qwen3-235b-a22b" in family_only[0]["models"]


def test_alias_normalization():
    assert ni.canon_model("q80") == "qwen3-80b"
    assert ni.canon_model("Qwen3-Coder-30B-A3B-Instruct") == "qwen3-30b-a3b"
    assert ni.canon_model("qwen3-235b-a22b:F1") == "qwen3-235b-a22b"
    assert ni.canon_model("qwen38") == "qwen3.8-27b"
    assert ni.canon_organ("gate_proj") == "gate"
    assert ni.canon_organ("mlp_down") == "down"
    assert ni.canon_family("trivial global expert sharing") == "cross_expert_structure"
    assert ni.canon_family("cross_expert_and_cross_layer_tying") == "cross_expert_structure"
    assert ni.canon_family("expert templates + deltas") == "cross_expert_structure"
    assert ni.canon_family("hwir_node_types") == "hwir_node_types"
    a = ni.query(model="q80", hypothesis_family="trivial global expert sharing")
    b = ni.query(model="qwen3-80b", hypothesis_family="cross_expert_structure")
    assert a and b
    assert {h["scar_id"] for h in a} & {h["scar_id"] for h in b}


def test_refuse_if_dead_negative_control():
    """The guard must actually fire on a known-dead hypothesis from the corpus."""
    refusal = ni.refuse_if_dead(
        {
            "model": "qwen3-235b-a22b",
            "organ": "gate",
            "hypothesis_family": "cross_expert_structure",
        }
    )
    assert refusal is not None, "refuse_if_dead did not fire on cross-expert structure"
    assert refusal["refused"] is True
    assert refusal["source_path"]
    assert refusal["scar_id"]
    assert refusal["hypothesis_family"] == "cross_expert_structure"
    # Real corpus, not a fixture: one of the known settling paths.
    known = (
        "NEGATIVE_TRANSFER_ATLAS.json",
        "NEGATIVE_SCIENCE.json",
        "NOETIC_NEGATIVE_SCIENCE.json",
        "QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json",
        "DOCTOR_NEGATIVE_TRANSFER_ATLAS.json",
    )
    assert any(k in refusal["source_path"] for k in known), refusal["source_path"]

    # Alias form used in the lane contract.
    also = ni.refuse_if_dead(
        {
            "model": "qwen3-80b",
            "hypothesis_family": "trivial global expert sharing",
        }
    )
    assert also is not None, "refuse_if_dead missed the Qwen80 cross-expert measurement"


def test_refuse_if_dead_is_targeted_not_blanket():
    dead = ni.refuse_if_dead(
        {
            "model": "qwen3-235b-a22b",
            "organ": "gate",
            "hypothesis_family": "cross_expert_structure",
        }
    )
    assert dead is not None

    allowed = ni.refuse_if_dead(
        {
            "model": "qwen3-235b-a22b",
            "organ": "gate",
            "hypothesis_family": "hwir_node_types",
        }
    )
    assert allowed is None, f"structurally different proposal was refused: {allowed}"

    # A different named parent is not pruned by a MODEL_SPECIFIC scar.
    other_parent = ni.refuse_if_dead(
        {
            "model": "brand_new_unmeasured_parent",
            "hypothesis_family": "cross_expert_structure",
        }
    )
    assert other_parent is None, f"blanket refuse across parents: {other_parent}"

    # Missing family is not a refuse.
    assert ni.refuse_if_dead({"model": "qwen3-235b-a22b", "organ": "gate"}) is None


def test_coverage_is_honest_about_gaps():
    cov = ni.coverage()
    assert cov["n_scars"] == cov["n_parsed"] + cov["n_unparsed"]
    assert cov["does_not_cover"], "coverage must say what the index does not cover"
    joined = " ".join(cov["does_not_cover"]).lower()
    assert "protected" in joined or "static_only" in joined
    assert any("haider" in x.lower() or "worktree" in x.lower() for x in cov["does_not_cover"])


def test_receipt_has_no_hardware_numeric_claims():
    out = ni.build()
    doc = json.loads(out.read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                    raise AssertionError(f"hardware field {here}={v!r}")
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)
    assert doc["evidence_class"] == "STATIC_ONLY"


def test_general_physical_scars_refuse_whatever_model_is_named():
    """A law about method does not stop being true because a parent is named.

    The seven campaign scars are process defects, not model findings. Gating
    them on an exact model match made every one of them unreachable from a
    model-specific proposal -- silently, which is the same narrow-probe defect
    they record.
    """
    from tools.future.negative_index import refuse_if_dead
    for family in (
        "prefill_over_generated_token_denominator",
        "adjacency_is_not_overlap",
        "priority_zero_falsy_or_default",
        "event_timestamp_unit_mismatch",
        "source_instrumented_runtime_binary_stale",
        "environment_mismatch_unfused_vs_sealed",
        "shared_index_bare_commit_sweeps_foreign_stage",
    ):
        for model in ("qwen3.8-27b", "glm-5.2", "deepseek-v4-flash", None):
            r = refuse_if_dead({"model": model, "hypothesis_family": family})
            assert r and r.get("refused"), f"{family} allowed for model={model}"


def test_model_specific_scars_still_do_not_prune_a_different_parent():
    """The widening must not have broken the rule it was carved out of."""
    from tools.future.negative_index import refuse_if_dead
    family = "catalog_addressing_not_primary_703_530_cause"   # qwen3.8-27b only
    assert refuse_if_dead({"model": "qwen3.8-27b", "hypothesis_family": family})
    assert refuse_if_dead({"model": "glm-5.2", "hypothesis_family": family}) is None
    assert refuse_if_dead({"model": "gpt-oss-120b", "hypothesis_family": family}) is None
