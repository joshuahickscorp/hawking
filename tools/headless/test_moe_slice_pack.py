"""G023 step-1c pins."""
import json
from pathlib import Path

import pytest

RH = Path(__file__).resolve().parents[2] / "receipts/headless"
R = RH / "MODEL2_SLICE_PACK.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="slice not packed")


def rec():
    return json.load(open(R))


def test_a_real_catalog_and_segments_exist_on_disk():
    d = rec()
    root = Path(d["artifact_root"])
    assert (root / "catalog.hq38m20").is_file()
    segs = list((root / "segments").iterdir())
    assert len(segs) == d["n_segments_written"] == d["n_tensors"]


def test_it_is_driven_by_the_specimen_index_not_the_q4_manifest():
    """compile_mix requires the q4 incumbent manifest, which model #2 does not have."""
    d = rec()
    assert "safetensors index" in d["driven_by"]
    assert "q4 incumbent" in d["driven_by"]


def test_representations_come_from_the_planner_not_another_models_genome():
    d = rec()
    assert "KERNEL_PLANNER_MODEL2" in d["representations_from"]
    kp = json.load(open(RH / "KERNEL_PLANNER_MODEL2.json"))
    sel = {r["organ"]: r["selected_representation"] for r in kp["organ_plan"]}
    assert d["planned"]["moe_expert"] == sel["moe_expert"]
    assert d["planned"]["attention_gqa"] == sel["gqa_attention"]


def test_the_organ_name_alias_is_applied():
    """The planner says gqa_attention, the packer says attention_gqa. The mismatch made
    the lookup miss silently and fall back to qwen38's q3, decoding at 0.930."""
    d = rec()
    assert "attention_gqa" in d["planned"], "the alias is not being applied"


def test_every_segment_reads_back_byte_identical():
    assert rec()["all_segments_read_back_identical"] is True


def test_an_independent_decoder_verified_the_quantized_segments():
    """Not the packer's own round-trip helper: a separate reader in the tool."""
    d = rec()["decode"]
    assert "INDEPENDENT" in d["decoder"]
    assert d["n_quantized"] >= 20
    assert d["worst_cosine"] > 0.95
    assert d["median_cosine"] > 0.98


def test_the_run_has_no_failures():
    d = rec()
    assert d["failures"] == []
    assert d["pass"] is True


def test_all_three_organ_kinds_are_present_in_the_slice():
    r = rec()["roles"]
    assert r["moe_expert"] > 0
    assert r["attention_gqa"] > 0
    assert r["leftover"] > 0          # router and norms, f32 by plan


def test_the_scope_is_declared_a_slice():
    s = rec()["scope"]
    assert s["is_a_slice_not_a_model"] is True
    assert s["layers_packed"] < s["layers_in_model"]
    assert s["experts_packed"] < s["experts_per_layer"]


def test_the_composition_caveat_is_carried():
    assert "does not compose" in rec()["what_this_does_not_show"]
