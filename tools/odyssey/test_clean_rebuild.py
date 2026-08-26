"""The reproduction is only real if the closure is closed and the parent is not needed."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import clean_rebuild as cr

REPO = Path(__file__).resolve().parents[2]
R = REPO / "receipts/headless/QWEN_CLEAN_REBUILD.json"
ROOT = Path.home() / "noetic/CLEAN_REBUILD_A" / cr.MIX_ID


def test_rebuild_landed_and_matches_the_target_density():
    d = json.load(open(R))
    assert d["rebuild"]["exit_code"] == 0
    assert d["compare"]["present"] and d["compare"]["n_segments"] == 755


def test_no_segment_shares_an_inode_with_an_incumbent():
    d = json.load(open(R))
    assert d["compare"]["n_shared_inodes"] == 0
    assert d["compare"]["hardlink_smuggling"] is False
    live = [f for f in (ROOT / "segments").iterdir() if f.is_file() and f.stat().st_nlink > 1]
    assert live == [], live[:3]


def test_the_incumbent_holds_no_unique_weight_bytes():
    d = json.load(open(R))["dehardlink"]
    assert d["n_mismatched"] == 0 and d["n_unmapped"] == 0
    assert d["incumbent_holds_unique_weight_bytes"] is False
    assert d["n_regenerated_byte_identical"] > 300


def test_accounting_reconciles_and_the_canary_refuses():
    a = json.load(open(R))["accounting"]
    assert a["reconciles"] and a["n_unclassified_files"] == 0
    assert a["canary"]["refused"] is True and a["canary"]["restored"] is True
    assert a["total_bytes_in_classes"] == a["total_bytes_on_disk"]


def test_tokenizer_state_is_inside_the_closure():
    for name in ("tokenizer.json", "vocab.json", "merges.txt", "config.json"):
        assert (ROOT / name).is_file(), name


def test_zero_parent_was_proven_by_absence_not_by_a_counter():
    z = json.load(open(R))["zero_parent"]
    assert z["QWEN_ZERO_PARENT_RUNTIME_DEPENDENCY"] == "PASS"
    probes = {p["probe"]: p for p in z["adversarial_probes"]}
    assert probes["parent_path_absent"]["exists"] is False
    assert probes["parent_tokenizer_absent"]["exists"] is False
    assert probes["hf_cache_holds_no_qwen38_parent"]["holds_parent"] is False
    assert z["coherent"] and z["n_unique_ids"] > 2
    assert z["parent_restored"] is True


def test_the_parent_is_back_where_it_belongs():
    assert (Path.home() / "models/qwen3.8-27b-abliterated-bf16/tokenizer.json").is_file()
    assert not list((Path.home() / "models").glob("*MOVED_FOR_ZERO_PARENT_TEST*"))


def test_no_closure_gap_remains_open():
    d = json.load(open(R))
    assert d["n_closure_gaps_open"] == 0
    assert all(g.get("resolved") for g in d["closure_gaps"])
