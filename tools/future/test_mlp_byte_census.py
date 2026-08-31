"""Tests for the MLP byte census.

A guard nobody has watched fail is not a guard. The load-bearing refusal:
an irreconcilable per-organ sum raises, it does not silently pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import mlp_byte_census as mbc
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    HardwareClaimError,
    _assert_no_hardware_claims,
)


def test_build_emits_sealed_receipt():
    out = mbc.build()
    assert out.parent == RECEIPTS
    assert out.name == "MLP_BYTE_CENSUS.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == mbc.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    _assert_no_hardware_claims(doc)
    assert doc["census"]["active_weight_bytes_per_token"] == (
        mbc.RECORDED_ACTIVE_WEIGHT_BYTES_PER_TOKEN
    )
    assert doc["census"]["reconciliation"]["active_equals_recorded"] is True
    assert "tensors" not in doc["census"]
    assert doc["mlp_active_bytes"] == doc["census"]["mlp"]["active_bytes"]
    assert doc["mlp_share_of_active"] == doc["census"]["mlp"]["share_of_active"]


def test_module_entrypoint_runs_and_emits_sealed_receipt():
    rc = mbc.main(["--build"])
    assert rc == 0
    doc = json.loads((RECEIPTS / mbc.RECEIPT).read_text())
    assert doc["schema"] == mbc.SCHEMA
    assert doc["seal_sha256"]


def test_selftest_aliases_build():
    assert mbc.selftest is mbc.build or mbc.selftest().name == mbc.RECEIPT


def _snap() -> dict:
    return mbc.census()


def test_active_sum_reconciles_to_recorded_total():
    snap = _snap()
    recorded = mbc.RECORDED_ACTIVE_WEIGHT_BYTES_PER_TOKEN
    assert snap["active_weight_bytes_per_token"] == recorded
    assert snap["recorded_active_weight_bytes_per_token"] == recorded
    organ_sum = sum(row["active_bytes"] for row in snap["by_organ"])
    assert organ_sum == recorded
    family_sum = sum(row["active_bytes"] for row in snap["by_organ_family"])
    assert family_sum == recorded
    layer_sum = sum(row["active_bytes"] for row in snap["per_layer"])
    global_sum = sum(row["active_bytes"] for row in snap["globals"])
    assert layer_sum + global_sum == recorded
    assert snap["reconciliation"]["active_equals_recorded"] is True
    assert snap["reconciliation"]["unclassified_tensors"] == 0


def test_mlp_share_is_rederived_from_tensors_not_copied():
    src = Path(mbc.__file__).read_text()
    # The historically quoted percentage is not a source constant.
    assert "0.54" not in src
    assert "54%" not in src
    snap = _snap()
    mlp = next(row for row in snap["by_organ_family"] if row["family"] == "mlp")
    gate = next(row for row in snap["by_organ"] if row["organ"] == "mlp.gate")
    up = next(row for row in snap["by_organ"] if row["organ"] == "mlp.up")
    down = next(row for row in snap["by_organ"] if row["organ"] == "mlp.down")
    derived = gate["active_bytes"] + up["active_bytes"] + down["active_bytes"]
    assert mlp["active_bytes"] == derived
    assert snap["mlp"]["active_bytes"] == derived
    share = derived / snap["active_weight_bytes_per_token"]
    assert snap["mlp"]["share_of_active"] == share
    assert mlp["share_of_active"] == share
    # 64 layers * 3 organs, each a whole tensor.
    assert gate["n_tensors"] == 64
    assert up["n_tensors"] == 64
    assert down["n_tensors"] == 64
    assert gate["active_bytes"] == up["active_bytes"] == down["active_bytes"]


def test_embedding_is_one_row_not_the_table():
    snap = _snap()
    embed = next(row for row in snap["by_organ"] if row["organ"] == "embedding")
    hidden = snap["identity"]["geometry"]["hidden_size"]
    row_bytes = mbc.q4_group64_row_bytes(hidden)
    assert embed["active_bytes"] == row_bytes
    assert embed["storage_bytes"] > embed["active_bytes"]
    assert snap["unread_embedding_table_bytes"] == (
        embed["storage_bytes"] - embed["active_bytes"]
    )
    assert (
        snap["active_weight_bytes_per_token"] + snap["unread_embedding_table_bytes"]
        == snap["catalog_total_bytes"]
    )


def test_per_layer_organs_and_hybrid_attention():
    snap = _snap()
    kinds = {}
    for layer in snap["per_layer"]:
        kinds[layer["kind"]] = kinds.get(layer["kind"], 0) + 1
        organs = {row["organ"] for row in layer["organs"]}
        assert "mlp.gate" in organs and "mlp.up" in organs and "mlp.down" in organs
        if layer["kind"] == "full_attention":
            assert {"attention.q", "attention.k", "attention.v", "attention.o"} <= organs
            assert "attention.linear_qkvz" not in organs
        elif layer["kind"] == "linear_attention":
            assert "attention.linear_qkvz" in organs
            assert "attention.q" not in organs
    assert kinds.get("full_attention") == 16
    assert kinds.get("linear_attention") == 48


def test_state_kv_is_unknown_not_invented():
    snap = _snap()
    unknown = snap["state_not_in_catalog"]
    assert unknown["kv_cache_bytes_per_token"] == "UNKNOWN"
    assert unknown["deltanet_recurrent_state_bytes_per_token"] == "UNKNOWN"
    assert unknown["activation_bytes_per_token"] == "UNKNOWN"
    state = next(row for row in snap["by_organ_family"] if row["family"] == "state")
    # A_log + dt_bias are catalogued parameters and must be counted, not guessed.
    assert state["active_bytes"] > 0
    assert state["active_bytes"] == 19200


def test_unreconciled_sum_raises_and_does_not_silently_pass():
    """NEGATIVE CONTROL: a broken per-organ sum must refuse."""
    root = mbc.resolve_artifact_root()
    records = mbc.parse_catalog(root / mbc.CATALOG_NAME)
    geo = mbc.load_geometry(root)
    sealed = mbc.load_sealed()
    # Control: the real catalog reconciles.
    ok = mbc.census(
        root=root,
        catalog_records=records,
        geometry=geo,
        sealed=sealed,
        recorded_total=mbc.RECORDED_ACTIVE_WEIGHT_BYTES_PER_TOKEN,
    )
    assert ok["active_weight_bytes_per_token"] == mbc.RECORDED_ACTIVE_WEIGHT_BYTES_PER_TOKEN

    # Embedding storage is not per-token active (one Q4 row is). Mutate a
    # whole-tensor organ so the active sum actually moves.
    mutated = []
    flipped = False
    for name, nbytes in records:
        if (not flipped) and "mlp.gate_proj" in name:
            mutated.append((name, nbytes + 1))
            flipped = True
        else:
            mutated.append((name, nbytes))
    assert flipped, "catalog had no mlp.gate_proj to mutate"
    with pytest.raises(mbc.UnreconciledCensus) as caught:
        mbc.census(
            root=root,
            catalog_records=mutated,
            geometry=geo,
            sealed=sealed,
            recorded_total=mbc.RECORDED_ACTIVE_WEIGHT_BYTES_PER_TOKEN,
        )
    assert caught.value.active != caught.value.recorded
    assert "REFUSED" in str(caught.value)
    assert str(mbc.RECORDED_ACTIVE_WEIGHT_BYTES_PER_TOKEN) in str(caught.value)

    with pytest.raises(mbc.UnreconciledCensus):
        mbc.census(
            root=root,
            catalog_records=records,
            geometry=geo,
            sealed=sealed,
            recorded_total=1,
        )

    with pytest.raises(mbc.UnreconciledCensus):
        mbc.reconcile_active(0, mbc.RECORDED_ACTIVE_WEIGHT_BYTES_PER_TOKEN)


def test_unclassified_tensor_is_a_refusal():
    with pytest.raises(mbc.UnclassifiedTensor):
        mbc.classify_tensor("language_model.model.layers.0.not_an_organ.weight")


def test_required_families_cover_the_contract_list():
    families = mbc.representation_families(_snap(), consult_index=False)
    ids = [row["id"] for row in families]
    assert ids == list(mbc.REQUIRED_FAMILY_IDS)
    for row in families:
        assert row["mechanism"]
        assert row["byte_model"]
        assert row["cheapest_falsifier"]
        assert row["dense_rematerialization"] in {
            mbc.DIRECT_CONSUME,
            mbc.REJECTED_DENSE_REMAT,
            mbc.DEPENDS_ON_LOWERING,
        }
        assert row["status"] in {mbc.ALREADY_FALSIFIED, mbc.OPEN}
        assert row["evidence_class"] == "STATIC_ONLY"
        assert row["gpu_authority"] is False


def test_already_falsified_families_cite_this_surface_scars():
    families = mbc.representation_families(_snap(), consult_index=False)
    by_id = {row["id"]: row for row in families}
    assert by_id["lower_bit"]["status"] == mbc.ALREADY_FALSIFIED
    assert by_id["shared_bases"]["status"] == mbc.ALREADY_FALSIFIED
    assert by_id["factorized_programs"]["status"] == mbc.ALREADY_FALSIFIED
    assert by_id["dictionary_programs"]["status"] == mbc.ALREADY_FALSIFIED
    assert by_id["product_codebook_programs"]["status"] == mbc.ALREADY_FALSIFIED
    assert by_id["sparse_residuals"]["status"] == mbc.ALREADY_FALSIFIED
    assert by_id["latent_accumulation"]["status"] == mbc.ALREADY_FALSIFIED
    assert by_id["capability_sensitive_literal_islands"]["status"] == mbc.ALREADY_FALSIFIED
    assert by_id["routed_subprograms"]["status"] == mbc.ALREADY_FALSIFIED

    def _ids(family_id: str) -> set[str]:
        return {c["scar_id"] for c in by_id[family_id]["citations"]}

    assert "NNS-029" in _ids("lower_bit")
    assert "QN-BINARY-INJURY" in _ids("lower_bit")
    assert "QN-SHARED-BASIS-DENSITY" in _ids("shared_bases")
    assert "NNS-014" in _ids("factorized_programs")
    assert "NNS-016" in _ids("factorized_programs")
    assert "NNS-017" in _ids("dictionary_programs")
    assert "NNS-017" in _ids("product_codebook_programs")
    assert "NNS-013" in _ids("latent_accumulation")
    assert "QN-BINARY-HEALING" in _ids("capability_sensitive_literal_islands")
    for row in families:
        if row["status"] == mbc.ALREADY_FALSIFIED:
            assert row["citations"], row["id"]
            for cite in row["citations"]:
                assert cite["source_path"]
                assert cite["scar_id"]


def test_dense_remat_families_are_tagged():
    families = mbc.representation_families(_snap(), consult_index=False)
    remat = {
        row["id"] for row in families if row["dense_rematerialization"] == mbc.REJECTED_DENSE_REMAT
    }
    assert "generated_weights" in remat
    assert "dictionary_programs" in remat
    assert "product_codebook_programs" in remat
    assert "cross_layer_prediction" in remat
    direct = {
        row["id"] for row in families if row["dense_rematerialization"] == mbc.DIRECT_CONSUME
    }
    assert "shared_bases" in direct
    assert "factorized_programs" in direct
    assert "function_replacement" in direct
    assert "lower_bit" in direct


def test_flash_rival_is_not_laundered_as_this_surface():
    families = mbc.representation_families(_snap(), consult_index=False)
    shared = next(row for row in families if row["id"] == "shared_input_transforms")
    assert shared["status"] == mbc.OPEN
    cousins = shared["cousin_not_this_surface"]
    assert cousins
    assert cousins[0]["not_this_specimen"] is True
    assert "Flash" in cousins[0]["specimen"] or "Flash" in cousins[0]["use"]
    assert "shared_input_latent_plus_expert_local_output_readout" in (
        cousins[0].get("killed_family_ids") or []
    ) or "routed_experts" in cousins[0]["specimen"]
    # A Flash MoE scar must not flip this dense MLP family to ALREADY_FALSIFIED.
    dead_ids = {
        c["scar_id"]
        for row in families
        for c in row["citations"]
        if row["status"] == mbc.ALREADY_FALSIFIED
    }
    assert "shared_input_latent_plus_expert_local_output_readout" not in dead_ids


def test_function_replacement_stays_open_for_full_width():
    families = mbc.representation_families(_snap(), consult_index=False)
    row = next(r for r in families if r["id"] == "function_replacement")
    assert row["status"] == mbc.OPEN
    assert row["dense_rematerialization"] == mbc.DIRECT_CONSUME
    assert "NNS-013" in {c["scar_id"] for c in row["citations"]}
    assert "m<17408" in row["cheapest_falsifier"] or "NNS-013" in row["cheapest_falsifier"]


def test_incumbent_packing_splits_codes_from_overhead():
    snap = _snap()
    pack = snap["mlp"]["incumbent_packing"]
    assert pack["code_bytes"] + pack["scale_bias_and_header_bytes"] == snap["mlp"]["active_bytes"]
    assert pack["code_bits"] == 2
    assert pack["derived_bpw"] == pytest.approx(2.5, abs=1e-4)
    assert pack["code_share_of_mlp"] > pack["overhead_share_of_mlp"]


def test_hardware_fields_stay_non_numeric_on_the_receipt():
    out = mbc.build()
    doc = json.loads(out.read_text())
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        # Presence as a word in prose is fine; a numeric claim is not.
        def walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    here = f"{path}.{k}" if path else k
                    if k in HARDWARE_FIELDS:
                        assert not isinstance(v, (int, float)) or isinstance(v, bool), here
                    walk(v, here)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")

        walk(doc)


def test_catalog_absent_refuses():
    with pytest.raises(mbc.CatalogAbsent):
        mbc.parse_catalog(Path("/no/such/catalog.hq38m20"))
    missing = Path("/tmp/mlp-census-empty-root-does-not-exist")
    with pytest.raises(mbc.CatalogAbsent):
        mbc.load_geometry(missing)


def test_bad_catalog_magic_refuses(tmp_path: Path):
    bogus = tmp_path / "catalog.hq38m20"
    bogus.write_bytes(b"NOTAMAGIC" + b"\x00" * 64)
    with pytest.raises(mbc.CensusRefuse, match="bad catalog magic"):
        mbc.parse_catalog(bogus)


def test_noetic_citations_are_not_empty_strings():
    families = mbc.representation_families(_snap(), consult_index=False)
    nns013 = None
    for row in families:
        for cite in row["citations"]:
            if cite["scar_id"] == "NNS-013":
                nns013 = cite
    assert nns013 is not None
    assert "narrow" in nns013["claim_refuted"].lower() or "SwiGLU" in nns013["claim_refuted"]
    assert nns013["source_path"].endswith("NOETIC_NEGATIVE_SCIENCE.json")


def test_cli_entrypoint_actually_runs():
    """20 unit tests passed while `--record` raised NameError on sys.argv.

    Importing a module does not exercise its __main__ block, so a broken
    entrypoint hides behind a green suite. Run it as a subprocess.
    """
    import subprocess as _sp
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[2]
    p = _sp.run(["python3", "tools/future/mlp_byte_census.py", "--build"],
                cwd=root, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-800:]
    assert (root / "receipts/future/MLP_BYTE_CENSUS.json").exists()
