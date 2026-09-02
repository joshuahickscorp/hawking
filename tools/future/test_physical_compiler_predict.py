"""Unlearned physical-compiler prediction API + historical observations."""
from __future__ import annotations

import pytest

from tools.future import complete_ebpw as ce
from tools.future import lpc_baselines as lb
from tools.future import physical_compiler_predict as pc


def test_load_real_physical_observations_is_nonempty():
    bundle = pc.load_physical_observations()
    assert bundle["n_observations"] == len(bundle["observations"])
    assert bundle["n_observations"] > 0
    loaded = bundle["named_receipts_loaded"]
    assert "lpc_dataset.ingest_from_disk" in loaded
    assert bundle["lpc_ingest"]["n"] > 0
    assert bundle["lpc_ingest"]["call_site"] == "tools.future.lpc_dataset.ingest_from_disk"
    assert bundle["n_with_graph_features"] > 0
    assert bundle["n_with_backend_cost"] > 0
    assert bundle["n_with_actual_outcome"] > 0
    for rel in (
        "receipts/future/ORGAN_BANDWIDTH.json",
        "receipts/future/ECONOMICS_CALIBRATION.json",
        "receipts/future/DEVICE_COMPILER.json",
    ):
        assert rel in loaded, rel


def test_graph_features_backend_costs_and_actual_outcomes_are_bound():
    bundle = pc.load_physical_observations()
    organs = [
        o
        for o in bundle["observations"]
        if o["source_receipt"].endswith("ORGAN_BANDWIDTH.json")
    ]
    assert organs
    mlp = next(o for o in organs if o["graph_features"]["organ"] == "mlp")
    assert mlp["actual_outcome"]["active_bytes"] is not None
    assert mlp["actual_outcome"]["organ_gpu_ms"] is not None
    assert mlp["evidence_tier"] == "STATIC"
    assert mlp["source_evidence_class"] == "DIAGNOSTIC_RELATIVE"

    costs = [o for o in bundle["observations"] if o.get("backend_cost")]
    assert costs
    assert all(c["evidence_tier"] == "COST_MODEL" for c in costs)
    assert {c["backend_cost"]["stream_class"] for c in costs} >= {
        "weight_codes",
        "broadcast_aux",
    }

    compiled = [
        o
        for o in bundle["observations"]
        if o["source_receipt"].endswith("DEVICE_COMPILER.json")
    ]
    assert compiled
    assert compiled[0]["graph_features"]["compile_time_science_only"] is True


def test_train_eval_split_has_no_leak_and_calls_lpc_baselines():
    bundle = pc.load_physical_observations()
    split = pc.train_eval_split(bundle["observations"])
    assert split["n_train"] + split["n_eval"] == bundle["n_observations"]
    assert split["n_train"] > 0
    assert split["n_eval"] > 0
    assert split["leak"] == []
    assert set(split["train_ids"]).isdisjoint(set(split["eval_ids"]))
    assert split["call_site"] == "tools.future.lpc_baselines.held_out_splits"
    assert split["held_out_by_organ"]
    assert all(row["no_leak"] for row in split["held_out_by_organ"])


def test_predict_refuses_without_trained_model():
    pred = pc.predict({"organ": "mlp", "backend": "metal"})
    assert pred["status"] == pc.UNLEARNED
    assert pred["trained"] is False
    assert pred["value"] is None
    assert pred["confidence"] is None
    assert pred["uncertainty"]["kind"] == "UNDEFINED"
    assert pred["schema_is_not_a_learned_compiler"] is True
    assert "no model is trained" in pred["statement"].lower() or "not a learned compiler" in pred["statement"].lower()


def test_predict_does_not_return_a_confident_number_even_with_a_fake_model():
    class Fake:
        trained = True
        value = 12.5

    pred = pc.predict({"organ": "mlp"}, model=Fake())
    assert pred["status"] == pc.UNLEARNED
    assert pred["trained"] is False
    assert pred["value"] is None
    assert pred["confidence"] is None
    assert pred["model_supplied"] is True
    assert pred["model_ignored"] is True
    assert not isinstance(pred["value"], (int, float))


def test_require_numeric_prediction_says_no_trained_model():
    with pytest.raises(pc.UnlearnedCompilerError, match="no trained model"):
        pc.require_numeric_prediction({"organ": "mlp"})


def test_schema_is_not_a_learned_compiler():
    pred = pc.predict()
    assert pred["schema_is_not_a_learned_compiler"] is True
    assert pred["status"] == pc.UNLEARNED
    bundle = pc.load_physical_observations()
    assert all(o["schema_is_not_a_learned_compiler"] for o in bundle["observations"])


def test_complete_ebpw_build_calls_unlearned_predict_and_does_not_use_the_value():
    """Production call site: complete_ebpw.build invokes predict() and ignores value."""
    doc = ce.build()
    block = doc["science_dataset"]
    pred = block["learned_compiler_prediction"]
    assert pred["status"] == pc.UNLEARNED
    assert pred["value"] is None
    assert block["learned_compiler_value_used"] is False
    assert "unlearned predictor is not the billed figure" in block["cost_authority"]
    assert (
        "tools.future.physical_compiler_predict.predict" in block["call_sites"]
    )
    assert (
        "tools.future.science_corpus.measurement_from_ebpw_bill"
        in block["call_sites"]
    )
    assert doc["incumbent"]["complete_ebpw"] == pytest.approx(3.1393, abs=0.001)


def test_baseline_predict_is_labelled_not_learned_and_calls_lpc_baselines():
    row = lb._complete_fixture(row_id="t0", organ_fingerprint="mlp", latency=10.0)
    out = pc.baseline_predict(row, [row], method="nearest")
    assert out["learned"] is False
    assert out["call_site"] == "tools.future.lpc_baselines.predict"
    assert out["schema_is_not_a_learned_compiler"] is True
    assert out["status"] in (lb.PREDICTED, lb.ABSTAIN)
