"""Seeding must not recommend a representation whose capability already failed."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import representation_library as rl

REPO = Path(__file__).resolve().parents[2]
R = REPO / "receipts/headless/REPRESENTATION_LIBRARY.json"


def test_v1_values_all_survive():
    d = json.load(open(R))
    assert d["v1_preservation"]["lost"] == []
    assert d["v1_preservation"]["n_preserved"] == d["v1_preservation"]["n_v1_families"]


def test_binary_is_not_a_success_on_the_mlp():
    """v1 listed binary as successful on the MLP; it is generation-injured."""
    d = json.load(open(R))
    b = next(f for f in d["families"] if f["family"] == "binary")
    a = b["per_architecture"]["qwen3.8-27b-abliterated"]
    assert a["successful_organs"] == []
    assert set(a["capability_failed_organs"]) == {"mlp_gate_up", "mlp_down"}
    assert set(a["applies_to_organs"]) == {"mlp_gate_up", "mlp_down"}


def test_seed_ranks_the_coherent_family_first():
    fams = rl.build()
    r = rl.seed(fams, "moe_expert", "qwen3_moe")
    assert r["ranked"][0]["family"] == "q2_affine"
    binary = next(x for x in r["ranked"] if x["family"] == "binary")
    assert binary["score"] < r["ranked"][0]["score"]


def test_model_specific_failure_warns_but_never_prunes():
    fams = rl.build()
    r = rl.seed(fams, "mlp_gate_up", "some_other_arch")
    assert not any(e["family"] == "binary" for e in r["excluded"]), \
        "a MODEL_SPECIFIC Qwen failure must not exclude the family for another architecture"


def test_untested_families_are_present_not_invented():
    d = json.load(open(R))
    for f in d["families"]:
        if f["status"] == "UNTESTED":
            assert f["untested_reason"]
            assert f["per_architecture"] == {}


def test_bad_citation_is_refused():
    for rel, jp in (("receipts/headless/NO_SUCH.json", None),
                    ("receipts/headless/BYTES_FRONTIER.json", "nope.nope")):
        try:
            rl._cite(rel, jp)
        except rl.Refused:
            continue
        raise AssertionError(f"accepted bad citation {rel}#{jp}")
