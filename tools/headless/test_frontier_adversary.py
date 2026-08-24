"""N014 frontier adversary: attacks must have been RUN, not answered.

Cheap attacks re-execute in this process. The receipt must record a RUN
against every frontier claim and every one of the six questions, with at
least one physical refutation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from frontier_adversary import (  # noqa: E402
    CLAIM_IDS,
    ONEBIT_RULE,
    ORGAN_ELEMS,
    PARENT_A,
    PARENT_PARAMS,
    QUESTION_IDS,
    RECEIPT,
    SCHEMA,
    SEALED_CLOSURE,
    attack_cache_artifact,
    attack_capability,
    attack_dense_state,
    attack_hidden_bytes,
    attack_hidden_parent,
    attack_wrong_benchmark_path,
    onebit_local_survives,
    packed_affine2_codes,
    plant_integer_metallib,
    run,
    run_plant,
    score_text_against_suite,
    whole_model_floor,
)
from noetic_dispatch_fusion import theoretical_after  # noqa: E402
from noetic_information_accounting import embedded_weightlike_bytes  # noqa: E402
from noetic_parent_a import RECORDED_TOKEN_IDS  # noqa: E402


def _doc() -> dict:
    if RECEIPT.is_file():
        d = json.loads(RECEIPT.read_text())
        if d.get("schema") == SCHEMA and d.get("counts", {}).get("questions_ran") == 6:
            return d
    live = os.environ.get("FRONTIER_ADVERSARY_LIVE", "0") == "1"
    return run(live_gpu=live)


def test_integer_coded_metallib_is_missed_by_the_accountant():
    """The named remaining gap. A hide the 5/5 canary never planted."""
    payload = packed_affine2_codes()
    assert len(payload) == 256 * 4
    assert embedded_weightlike_bytes(payload) == 0
    row = run_plant(plant_integer_metallib)
    assert row["content"]["status"] == "MISSED", row["content"]
    assert row["content"]["delta"]["MODEL_SPECIFIC_BYTES"] == 0
    assert row["content"]["delta"]["MACHINE_SPECIFIC_BYTES"] >= row["plant"]["payload_bytes"]


def test_hidden_bytes_attack_function_runs_and_refutes():
    q = attack_hidden_bytes()
    assert q["ran"] is True
    assert q["id"] == "hidden_bytes"
    assert q["known_five"]["COUNTED"] == 5
    assert q["known_five"]["MISSED"] == 0
    assert q["new_hides"]["integer_coded_metallib"]["status"] == "MISSED"
    assert q["new_hides"]["geometry_codebook_n30"]["status"] == "MISSED"
    assert q["new_hides"]["geometry_codebook_n66"]["status"] == "COUNTED"
    assert q["verdict"] == "REFUTED"


def test_dense_counter_is_a_literal_zero_not_an_increment():
    q = attack_dense_state()
    assert q["ran"] is True
    dec = q["decoder"]
    assert dec["present"] is True
    assert dec["increments"] == 0
    greedy = q["greedy_example"]
    assert greedy["print_zero"] + greedy["json_zero"] + dec["literal_zero"] >= 1
    assert q["metal"]["dequant_kernel_only_in_parity_example"] is True
    probe = q["numpy_probe"]
    assert probe["detector_ok"] is True
    assert probe["reconstruct_zero_dense"] is False
    assert probe["fused_zero_dense"] is True
    assert q["verdict"] == "REFUTED"


def test_hidden_parent_attack_runs_without_loading_a_second_27b():
    q = attack_hidden_parent(live_gpu=False)
    assert q["ran"] is True
    assert q["did_not_load_second_27b"] is True
    assert q["did_not_modify_parent_a"] is True
    assert q["python_opens"] == []
    weight_opens = [
        h for h in q["source_hits"]
        if h.get("needle") == ".safetensors" and "tokenizer" not in (h.get("text") or "").lower()
    ]
    assert weight_opens == [], weight_opens


def test_bad_control_is_supplied_and_formula_is_blind_to_bytes_moved():
    q = attack_wrong_benchmark_path()
    assert q["ran"] is True
    bad = q["dispatch_fusion_bad_control"]
    assert bad["full_fusion"] == theoretical_after("swiglu", True, True) == 756
    assert bad["full_plus_dummy_per_layer"] == 820
    assert bad["dummy_is_worse"] is True
    gpu = q["gpu_ledger_bad_control"]
    assert gpu["dram_absent"] is True
    assert gpu["formula_blind_to_bytes_moved"] is True
    assert gpu["bad_2x_gpu_ns_implied_gb_s"] == gpu["implied_gb_s"] / 2.0
    assert q["multisession_bad_control"]["four_copies_rejected"] is True


def test_parent_a_tok_s_missed_its_own_reproduction_band():
    q = attack_cache_artifact()
    assert q["ran"] is True
    p = q["parent_a"]
    assert p["inside_recorded_band"] is False
    assert abs(p["recorded_tok_s"] - 34.87340648509909) < 1e-9
    assert p["measured_tok_s"] < p["recorded_band"][0]
    assert q["gpu_ledger"]["OS_PAGE_CACHE_COLD_GPU_NS_kind"] == "ABSENT"
    assert q["did_not_modify_parent_a"] is True
    assert q["verdict"] == "REFUTED"


def test_sixteen_token_sample_fails_every_capability_item():
    q = attack_capability(live=None)
    assert q["ran"] is True
    sealed = q["sealed_sample"]
    assert sealed["n_tokens"] == 16
    assert sealed["ids"] == RECORDED_TOKEN_IDS
    assert sealed["think_preamble"] is True
    assert sealed["suite"]["n_pass"] == 0
    assert sealed["suite"]["n_fail"] == sealed["suite"]["n_items"]
    assert q["parent_a_capability_suite_ran"] is False
    assert q["verdict"] == "REFUTED"
    # The predicate itself, not just the wrapper.
    score = score_text_against_suite(sealed["text_head"])
    assert score["n_pass"] == 0


def test_organ_floor_implication_charges_lm_head_at_the_embed_floor():
    quoted = 2.9398
    with_4125 = whole_model_floor(2.25, 4.125)
    with_325 = whole_model_floor(2.25, 3.25)
    assert abs(with_4125 - quoted) < 1e-4
    assert with_325 < quoted - 0.03
    assert sum(ORGAN_ELEMS.values()) == PARENT_PARAMS


def test_onebit_rule_reapplied_b3_survives_b6_unhealthy():
    doc = json.loads((REPO / "receipts" / "headless" / "ONEBIT_FAMILIES.json").read_text())
    null = 0.40992871175209683
    by = {f["family_id"]: f for f in doc["families"]}
    def tensors_ok(fam):
        ok_all = True
        n_bad = 0
        for t in fam.get("per_tensor") or fam.get("tensors") or []:
            rel = float(t.get("rel_fro"))
            gain = float(t.get("gain"))
            cos = float(t.get("cosine") or fam["mean_cosine"])
            n = float(t.get("null") or null)
            ok = onebit_local_survives(rel, gain, cos, n)
            if not ok:
                ok_all = False
                n_bad += 1
        return ok_all, n_bad
    b3_ok, b3_bad = tensors_ok(by["B3"])
    b6_ok, b6_bad = tensors_ok(by["B6"])
    assert b3_ok is True and b3_bad == 0
    assert b6_ok is False and b6_bad >= 1
    assert by["B3"]["storage_bpw"] == 1.85
    assert by["B6"]["mean_rel_fro"] < by["B3"]["mean_rel_fro"]
    assert ONEBIT_RULE["gain_min"] == 0.5


def test_receipt_records_a_run_against_every_claim_and_question():
    d = _doc()
    assert d["schema"] == SCHEMA
    assert d["did_not_load_second_27b"] is True
    assert d["did_not_modify_parent_a"] is True
    assert d["counts"]["questions_ran"] == 6
    assert d["counts"]["claims_ran"] == 6
    qids = [q["id"] for q in d["questions"]]
    cids = [c["id"] for c in d["claims"]]
    assert tuple(qids) == QUESTION_IDS
    assert tuple(cids) == CLAIM_IDS
    for q in d["questions"]:
        assert q["ran"] is True, q["id"]
        assert q["verdict"] in {"REFUTED", "SURVIVED", "WEAKENED"}, q
        assert q.get("why"), q["id"]
        assert q.get("commands"), q["id"]
    for c in d["claims"]:
        assert c["ran"] is True, c["id"]
        assert c["verdict"] in {"REFUTED", "SURVIVED", "WEAKENED"}, c
        assert c.get("why"), c["id"]
        assert c.get("attacks_run"), c["id"]
    assert d["at_least_one_claim_physically_refuted"] is True
    assert len(d["refuted_claim_ids"]) >= 1
    # These four are the physical hits this lane actually landed.
    for cid in ("NOETIC_PARENT_A", "GPU_ledger", "organ_floors", "onebit_families"):
        row = next(c for c in d["claims"] if c["id"] == cid)
        if cid == "onebit_families":
            assert row["verdict"] in {"SURVIVED", "REFUTED"}
        else:
            assert row["verdict"] == "REFUTED", (cid, row["verdict"], row["why"][:200])
    assert PARENT_A.is_dir()
    assert d["claims"][0]["evidence"]["closure_sha256_receipt"] == SEALED_CLOSURE


def test_sealed_parent_bytes_were_not_written():
    """Adversary may hash the parent. It may not mutate it."""
    d = _doc()
    assert d["did_not_modify_parent_a"] is True
    # Live reseal in the receipt must match the sealed hash if it ran.
    ev = next(c for c in d["claims"] if c["id"] == "NOETIC_PARENT_A")["evidence"]
    if ev.get("closure_sha256_live"):
        assert ev["closure_sha256_live"] == SEALED_CLOSURE
        assert ev["n_affine_live"] == 192


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
            print(f"ok  {n}")
    print("passed")
