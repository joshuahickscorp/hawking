"""Tests for the functional MLP teacher corpus.

Load-bearing negatives: a held-out split that shares a prompt id with train
must be REFUSED, and a Gaussian / synthetic row must be REFUSED. A guard
nobody has watched fail is not a guard.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from tools.future import mlp_teacher_corpus as mtc
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    _assert_no_hardware_claims,
)


def test_build_emits_sealed_receipt():
    out = mtc.build(capture=False)
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "MLP_TEACHER_CORPUS.json"
    assert doc["schema"] == mtc.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "DIAGNOSTIC_RELATIVE"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["gpu_authority"] is False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    _assert_no_hardware_claims(doc)
    for field in HARDWARE_FIELDS:
        assert field not in doc
    assert doc["selftest"]["held_out_leak_refused"] is True
    assert "HELD_OUT_PROMPT_LEAK" in doc["selftest"]["held_out_leak_codes"]
    assert doc["selftest"]["synthetic_refused"] is True
    assert "SYNTHETIC_ROW" in doc["selftest"]["synthetic_codes"]
    assert doc["fingerprint"]["n_layers"] == 64
    assert doc["fingerprint"]["representatives"]["layer0"]["typical"] is False


def test_held_out_split_refuses_shared_prompt_id():
    """NEGATIVE CONTROL: a prompt that sits on both sides must not emit."""
    rows = mtc.make_diverse_fixture_corpus(4, 3)
    split = mtc.split_by_prompt(rows)
    assert not (set(split["train_prompt_ids"]) & set(split["hold_prompt_ids"]))
    ok = mtc.emit_manifest(rows, split, allow_fixture=True, require_sizing=False)
    assert ok["accepted"] is True
    assert ok["split"]["disjoint"] is True

    leaked = dict(split)
    leaked["train_prompt_ids"] = list(split["train_prompt_ids"]) + [
        split["hold_prompt_ids"][0]
    ]
    with pytest.raises(mtc.CorpusRefused) as caught:
        mtc.emit_manifest(rows, leaked, allow_fixture=True, require_sizing=False)
    assert "HELD_OUT_PROMPT_LEAK" in caught.value.codes
    assert "REFUSED" in str(caught.value)
    assert split["hold_prompt_ids"][0] in caught.value.result["leaked_prompt_ids"]


def test_synthetic_row_refused():
    """NEGATIVE CONTROL: Gaussian / synthetic X cannot close the corpus (NNS-001)."""
    rows = mtc.make_diverse_fixture_corpus(4, 3)
    split = mtc.split_by_prompt(rows)
    poisoned = list(rows)
    poisoned[3] = mtc.make_gaussian_row(rows[3])
    with pytest.raises(mtc.CorpusRefused) as caught:
        mtc.emit_manifest(poisoned, split, allow_fixture=True, require_sizing=False)
    assert "SYNTHETIC_ROW" in caught.value.codes
    assert "REFUSED" in str(caught.value)
    assert caught.value.result["n_synthetic"] >= 1

    # The un-poisoned corpus of the same size must still emit.
    ok = mtc.emit_manifest(rows, split, allow_fixture=True, require_sizing=False)
    assert ok["accepted"] is True
    assert ok["n_rows"] == len(rows)


def test_synthetic_flag_alone_is_enough_to_refuse():
    rows = mtc.make_diverse_fixture_corpus(4, 3)
    split = mtc.split_by_prompt(rows)
    rows[1] = dict(rows[1])
    rows[1]["synthetic"] = True
    rows[1] = mtc.annotate_row(rows[1])
    with pytest.raises(mtc.CorpusRefused) as caught:
        mtc.emit_manifest(rows, split, allow_fixture=True, require_sizing=False)
    assert "SYNTHETIC_ROW" in caught.value.codes


def test_fingerprint_covers_64_layers_and_does_not_treat_layer0_as_typical():
    fps = mtc.fingerprint_layers()
    assert len(fps) == 64
    assert [r["layer"] for r in fps] == list(range(64))
    reps = mtc.pick_representatives(fps)
    assert reps["layer0"]["typical"] is False
    typical = next(p for p in reps["chosen"] if p["role"] == "typical")
    assert typical["layer"] != 0
    assert 0 not in [p["layer"] for p in reps["chosen"] if p["role"] == "typical"]
    mixers = {p["mixer"] for p in reps["chosen"]}
    assert "linear_attention" in mixers
    assert "full_attention" in mixers
    assert min(reps["chosen_layers"]) < 8
    assert max(reps["chosen_layers"]) == 63
    # Layer 0 is the high-entropy early outlier, not the mean.
    assert fps[0]["delta_from_global_mean"] > 0.0
    assert typical["layer"] == min(
        (r for r in fps if r["layer"] != 0),
        key=lambda r: abs(r["H_q_mean"] - reps["global_H_q_mean"]),
    )["layer"]


def test_position_bands_cover_early_middle_late():
    rows = mtc.make_diverse_fixture_corpus(4, 3)
    bands = {r["position_band"] for r in rows}
    assert bands == {"early", "middle", "late"}
    assert mtc.position_band(0, 9) == "early"
    assert mtc.position_band(4, 9) == "middle"
    assert mtc.position_band(8, 9) == "late"


def test_content_hash_excludes_row_id_and_provenance():
    rows = mtc.make_diverse_fixture_corpus(2, 3)
    a = dict(rows[0])
    b = dict(rows[0])
    b["row_id"] = "other-id"
    b["provenance"] = dict(a["provenance"])
    b["provenance"]["note"] = "copied envelope"
    assert mtc.content_sha256_of(a) == mtc.content_sha256_of(b)
    assert mtc.envelope_sha256_of(a) != mtc.envelope_sha256_of(mtc.annotate_row(b))


def test_duplicate_rows_above_threshold_refused():
    rows = mtc.make_diverse_fixture_corpus(4, 3)
    split = mtc.split_by_prompt(rows)
    # Copy the first unique row until copies dominate.
    pad = []
    for i in range(len(rows)):
        copy = dict(rows[0])
        copy["row_id"] = f"dup-{i}"
        pad.append(mtc.annotate_row(copy))
    padded = rows + pad
    with pytest.raises(mtc.CorpusRefused) as caught:
        mtc.emit_manifest(padded, split, allow_fixture=True, require_sizing=False)
    assert "DUPLICATE_ROWS_ABOVE_THRESHOLD" in caught.value.codes
    assert caught.value.result["collision_rate"] > mtc.DUP_RATE_MAX


def test_rank32_sizing_beats_nns007_scar_definition():
    assert mtc.min_train_rows_for_rank32() == 64
    assert mtc.RANK32_PARAMS == 32 * (2 * 5120 - 32)
    assert mtc.rows_per_dimension(mtc.NNS007_SCAR_ROWS, mtc.NNS007_SCAR_DIM) == pytest.approx(
        92 / 2048
    )
    # Rank-32 is determined at n >= 64. The scar (92 rows / 2048 dim) is n << dim
    # for a full-rank score. Per-layer n >= hidden is the ambient-dim floor.
    assert mtc.rows_per_dimension(64, 5120) == pytest.approx(64 / 5120)
    assert mtc.rows_per_dimension(5120) == pytest.approx(1.0)
    capture = mtc.load_existing_capture()
    if capture and capture.get("status") == "captured":
        rpd = float(capture["rows_per_dimension_train"])
        assert rpd > mtc.NNS007_SCAR_ROWS_PER_DIM
        per = capture.get("n_train_rows_per_layer") or {}
        assert per
        assert min(int(v) for v in per.values()) >= mtc.MIN_TRAIN_ROWS_DETERMINED


def test_selftest_function_proves_both_halves():
    result = mtc.selftest()
    assert result["held_out_leak_refused"] is True
    assert "HELD_OUT_PROMPT_LEAK" in result["held_out_leak_codes"]
    assert result["synthetic_refused"] is True
    assert "SYNTHETIC_ROW" in result["synthetic_codes"]
    assert result["fingerprint_n_layers"] == 64
    assert result["layer0_typical"] is False
    typical = mtc.pick_representatives(mtc.fingerprint_layers())
    assert next(p["layer"] for p in typical["chosen"] if p["role"] == "typical") != 0


def test_catalog_spans_five_required_domains():
    domains = {d for d, _ in mtc.CAPTURE_PROMPTS}
    assert set(mtc.CAPABILITY_DOMAINS) <= domains
    counts = {d: sum(1 for x, _ in mtc.CAPTURE_PROMPTS if x == d) for d in mtc.CAPABILITY_DOMAINS}
    assert all(n >= 8 for n in counts.values())


def test_family_mapping_hits_every_required_domain():
    mapped = set(mtc.FAMILY_TO_DOMAIN.values())
    assert set(mtc.CAPABILITY_DOMAINS) <= mapped


def test_is_synthetic_detects_nns001_kinds():
    row = mtc.make_diverse_fixture_corpus(2, 3)[0]
    assert mtc.is_synthetic_row(row) is False
    gauss = mtc.make_gaussian_row(row)
    assert mtc.is_synthetic_row(gauss) is True
    proxy = dict(row)
    proxy["provenance"] = dict(row["provenance"])
    proxy["provenance"]["kind"] = "gaussian_proxy"
    assert mtc.is_synthetic_row(proxy) is True


def test_fixture_rows_refused_on_real_emit_path():
    rows = mtc.make_diverse_fixture_corpus(4, 3)
    split = mtc.split_by_prompt(rows)
    with pytest.raises(mtc.CorpusRefused) as caught:
        mtc.emit_manifest(rows, split, allow_fixture=False, require_sizing=False)
    assert "SYNTHETIC_ROW" in caught.value.codes
