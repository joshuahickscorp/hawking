"""G011 pins."""
import json
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[2] / "receipts/headless/QWEN_RETIREMENT_GATE.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="G011 receipt not built")


def rec():
    return json.load(open(R))


def test_retirement_requires_the_transfer_substrate_first():
    """§17: only AFTER the transfer substrate succeeds."""
    p = rec()["prerequisite_transfer_substrate"]
    assert p["ok"] is True
    for name, v in p["receipts"].items():
        assert v["present"] and v["pass"], name


def test_the_substrate_rests_on_a_cross_architecture_law():
    """A law proven on one family is not evidence the science outlives the model."""
    p = rec()["prerequisite_transfer_substrate"]
    assert p["architecture_general_laws"]
    assert p["law_levels"].get("ARCHITECTURE_GENERAL", 0) >= 1


def test_every_section_17_category_is_checked_and_preserved():
    d = rec()["preservation"]
    assert set(d) == {"final_executable", "exact_source_identity", "genomes", "kernels",
                      "recipes", "receipts", "negative_science"}
    for k, v in d.items():
        assert v["ok"] is True, k
    assert rec()["all_preserved"] is True


def test_the_gate_verified_a_self_contained_executable_exists():
    e = rec()["preservation"]["final_executable"]["detail"]
    sc = [x for x in e if x.get("self_contained")]
    assert sc, "no executable is self-contained"
    for x in sc:
        assert x["closure_files"] == x["closure_required"]
        assert x["hardlinked_files"] == 0
        assert Path(x["root"]).is_dir()


def test_source_identity_is_hashes_not_a_52gb_copy():
    d = rec()["preservation"]["exact_source_identity"]["detail"]
    assert d["config_sha256"] and d["tensor_index_sha256"]
    assert d["architectures"]


def test_the_q4_recipe_metadata_is_preserved_outside_the_bulky_artifact():
    d = rec()["preservation"]["recipes"]["detail"]
    root = Path(__file__).resolve().parents[2]
    p = root / d["q4_manifest_preserved"]
    assert p.is_file()
    assert json.load(open(p))["tensors"]
    assert d["q4_manifest_sha256"]


def test_genome_check_accepts_both_schemas():
    """The sealed body predates the `genome` key; requiring it reported a preserved
    genome as missing."""
    g = rec()["preservation"]["genomes"]
    for name, v in g["detail"].items():
        assert v["schema"] in ("genome", "legacy"), name
    assert "two schemas" in g["note"]


def test_relegation_never_deletes():
    r = rec()["relegation"]
    assert "Nothing is deleted" in r["policy"]
    assert "operator's call" in r["operator_action_required"]


def test_the_q4_incumbent_is_not_relegated_while_the_resident_needs_it():
    """It was first classified RELEGATE on evidence that only held for variantB."""
    items = {Path(i["path"]).name: i for i in rec()["relegation"]["items"]}
    q4 = items["qwen38-gravity-uniform-q4-v1"]
    assert q4["class"] == "INDISPENSABLE_FOR_NOW"
    assert "210" in q4["why"]
    assert q4["path"] not in rec()["relegation"]["relegatable_now"]


def test_the_shippability_asymmetry_is_recorded():
    """The grand candidate is self-contained; the selected resident is not."""
    s = rec()["shippability"]
    assert "GRAND CANDIDATE is self-contained" in s["finding"]
    assert "SELECTED RESIDENT is not" in s["finding"]
    assert s["not_a_reopening_of_G040"]


def test_indispensable_artifacts_include_the_negative_evidence():
    items = {Path(i["path"]).name: i["class"] for i in rec()["relegation"]["items"]}
    assert items["VARIANT_A_MLP_ONLY"] == "PRESERVE_AS_NEGATIVE"
    assert items["CLEAN_REBUILD_A"] == "PRESERVE_AS_NEGATIVE"


def test_negative_science_is_preserved_with_this_campaigns_refutations():
    d = rec()["preservation"]["negative_science"]["detail"]
    assert d["n_entries"] > 0
    assert len(d["campaign_refutations_also_preserved"]) >= 3


def test_retirement_ready_is_the_conjunction_it_claims_to_be():
    d = rec()
    assert d["RETIREMENT_READY"] is (
        d["prerequisite_transfer_substrate"]["ok"] and d["all_preserved"])
    assert d["pass"] is d["RETIREMENT_READY"]
