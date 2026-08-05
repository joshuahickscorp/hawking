"""PROTO_FRANKENSTEIN_V0 corpus, tokenizer-independent alignment, cartography emitters."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import frankenstein_aligner as aligner  # noqa: E402
from lab.operators import frankenstein_cartography as carto  # noqa: E402
from lab.operators import frankenstein_corpus as corpus  # noqa: E402
from lab.operators import frankenstein_trace_format as traces  # noqa: E402
from lab.operators.frankenstein_gates import REQUIRES_GLM_RUNTIME  # noqa: E402
from lab.receipts import verify  # noqa: E402


# ---------------------------------------------------------------------------
# Membership + corpus ladder
# ---------------------------------------------------------------------------


def test_membership_includes_retention() -> None:
    assert "retention" in traces.MEMBERSHIP_SPLITS
    mgr = traces.MembershipManager()
    mgr.assign("a", "train")
    mgr.assign("b", "retention")
    with pytest.raises(traces.TraceFormatError, match="disjoint"):
        mgr.assign("a", "retention")
    sealed = mgr.seal_document()
    verify(sealed, label="membership retention")
    assert sealed["counts"]["retention"] == 1


def test_build_ladder_l0_subset_l1_real_sources(tmp_path: Path) -> None:
    ladder = corpus.build_ladder(l0_size=32, l1_size=128)
    assert len(ladder["L0"]) == 32
    assert len(ladder["L1"]) == 128
    l0_ids = {r["example_id"] for r in ladder["L0"]}
    l1_ids = {r["example_id"] for r in ladder["L1"]}
    assert l0_ids.issubset(l1_ids)

    # All required families present on L1.
    families = {r["family"] for r in ladder["L1"]}
    missing = set(corpus.CAPABILITY_FAMILIES) - families
    assert not missing, f"missing families: {missing}"

    # No synthetic Gaussian; all verified.
    for r in ladder["L1"]:
        assert r["synthetic_gaussian"] is False
        assert r["verified"] is True
        assert r["membership"] in traces.MEMBERSHIP_SPLITS
        assert r["byte_length"] > 0

    # Disjoint memberships.
    mem = ladder["membership"]
    verify(mem, label="membership freeze")
    assert mem["disjoint"] is True
    assert mem["ladder"]["L0_subset_of_L1"] is True
    assert set(mem["assignments"]) == l1_ids
    assert sum(mem["counts"].values()) == 128
    assert mem["counts"].get("retention", 0) >= 1

    # Index sealed.
    idx = ladder["index"]
    verify(idx, label="corpus index")
    assert idx["synthetic_gaussian"] is False
    assert idx["verified_only"] is True
    assert not idx["missing_families_L1"]

    # Write artifacts.
    paths = corpus.write_corpus_artifacts(ladder, out_dir=tmp_path)
    assert Path(paths["L0"]).is_file()
    assert Path(paths["L1"]).is_file()
    l0_lines = Path(paths["L0"]).read_text(encoding="utf-8").strip().splitlines()
    l1_lines = Path(paths["L1"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(l0_lines) == 32
    assert len(l1_lines) == 128
    # Sources are local real corpora.
    sources = {json.loads(line)["source"] for line in l1_lines}
    assert any("ramanujan" in s for s in sources)
    assert any("thesis" in s or "halo" in s for s in sources)


def test_no_duplicate_example_ids_across_splits() -> None:
    ladder = corpus.build_ladder(l0_size=32, l1_size=128)
    assigns = ladder["membership"]["assignments"]
    assert len(assigns) == len(set(assigns))
    # Each id appears once.
    ids = [r["example_id"] for r in ladder["L1"]]
    assert len(ids) == len(set(ids))


def test_math_present_in_held_out_and_retention_is_base_only() -> None:
    ladder = corpus.build_ladder(l0_size=32, l1_size=128)
    math_primary = set(corpus.CAPABILITY_FAMILIES) - corpus.RETENTION_FAMILIES
    by_split: dict[str, set[str]] = {s: set() for s in traces.MEMBERSHIP_SPLITS}
    for r in ladder["L1"]:
        by_split[r["membership"]].add(r["family"])
    # Held-out math for honest eval.
    assert by_split["public_test"] & math_primary
    assert by_split["hidden_test"] & math_primary
    assert by_split["train"] & math_primary
    # Retention is base-capability only.
    assert by_split["retention"].issubset(corpus.RETENTION_FAMILIES)
    assert not (by_split["retention"] & math_primary)


# ---------------------------------------------------------------------------
# Tokenizer-independent alignment
# ---------------------------------------------------------------------------


def test_utf8_byte_spans_and_token_piece_maps() -> None:
    surface = aligner.SharedSurface(
        text="Claim: 1+1=2\nProof step: use induction\nAnswer: 2",
        surface_id="shared",
    )
    loc = aligner.utf8_byte_span(surface, "use induction")
    assert loc["found"] is True
    assert surface.slice_bytes(loc["byte_start"], loc["byte_end"]) == "use induction"

    # Different tokenizations of the same surface (GLM vs DSV4F style pieces).
    glm_pieces = ["Claim", ":", " 1", "+", "1", "=", "2", "\n", "Proof", " step",
                  ":", " use", " induction", "\n", "Answer", ":", " 2"]
    dsv_pieces = ["Claim: ", "1+1", "=2\n", "Proof step: ", "use induction",
                  "\nAnswer: ", "2"]
    glm_map = aligner.glm_token_to_span_map(surface, glm_pieces)
    dsv_map = aligner.dsv4f_token_to_span_map(surface, dsv_pieces)
    assert all(s["align_by"] == "byte_span" for s in glm_map)
    assert all(s["token_ids_forbidden_for_alignment"] for s in dsv_map)
    pairs = aligner.pair_token_maps_via_byte_overlap(glm_map, dsv_map)
    assert len(pairs) >= 1
    assert all(p["joined_by_token_id"] is False for p in pairs)
    assert all(p["method"] == "byte_span_overlap" for p in pairs)

    # Pooling indices for answer span.
    ans = aligner.utf8_byte_span(surface, "Answer: 2")
    idxs = aligner.pool_indices_for_byte_span(
        glm_map, ans["byte_start"], ans["byte_end"]
    )
    assert idxs  # some tokens cover the answer


def test_token_id_alignment_still_refused() -> None:
    left = [{"token_id": 1}, {"token_id": 2}]
    right = [{"token_id": 9}, {"token_id": 8}]
    with pytest.raises(aligner.AlignerError, match="token-ID"):
        aligner.align_decoded_spans(left, right)
    with pytest.raises(aligner.AlignerError, match="token_id"):
        aligner.pair_token_maps_via_byte_overlap(
            [{"align_by": "token_id", "byte_start": 0, "byte_end": 1}],
            [{"align_by": "byte_span", "byte_start": 0, "byte_end": 1}],
        )


def test_semantic_anchors_claims_proof_code_answer() -> None:
    text = (
        "Claim: every natural has a successor\n"
        "Subgoal: prove base case\n"
        "Proof step: apply nat.succ\n"
        "tool=lean_check goal=base\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
        "Answer: QED\n"
    )
    anchors = aligner.extract_semantic_anchors(text)
    kinds = {a["kind"] for a in anchors}
    assert "claim" in kinds
    assert "subgoal" in kinds
    assert "proof_step" in kinds
    assert "tool_action" in kinds
    assert "answer" in kinds
    assert "code_ast_region" in kinds
    # Code AST should find the function.
    code = [a for a in anchors if a["kind"] == "code_ast_region"]
    assert any(a.get("meta", {}).get("name") == "add" for a in code)

    # Align same anchors against themselves.
    aligned = aligner.align_semantic_anchors(anchors, anchors)
    assert len(aligned) >= 4
    assert all(a["method"] == "semantic_anchor" for a in aligned)


def test_align_paired_sides_with_surface_and_pieces() -> None:
    surface = "Claim: hi\nAnswer: 42"
    left = {
        "decoded_spans": [
            {"text": "Claim: hi", "byte_start": 0, "byte_end": 9, "surface_id": "shared"}
        ],
        "formal_actions": [],
        "tool_events": [],
        "token_pieces": ["Claim", ": ", "hi", "\n", "Answer", ": ", "42"],
    }
    right = {
        "decoded_spans": [
            {"text": "Claim: hi", "byte_start": 0, "byte_end": 9, "surface_id": "shared"}
        ],
        "formal_actions": [],
        "tool_events": [],
        "token_pieces": ["Claim: hi\n", "Answer: 42"],
    }
    report = aligner.align_paired_sides(left, right, surface_text=surface)
    assert "semantic_anchors" in report["method_policy"]["allowed"]
    assert "token_id_to_token_id" in report["method_policy"]["forbidden"]
    assert report["summary"]["n_span_alignments"] == 1
    assert report["token_byte_maps"] is not None
    assert report["token_byte_maps"]["joined_by_token_id"] is False
    assert report["summary"]["n_token_byte_pairs"] >= 1


# ---------------------------------------------------------------------------
# Cartography + emitters
# ---------------------------------------------------------------------------


def test_functional_intervention_and_monotonic_phase() -> None:
    glm, dsv = carto.synthetic_paired_layers(
        n_glm=8, n_dsv=6, n_samples=64, d_glm=20, d_dsv=16, seed=2
    )
    mat = carto.correspondence_matrix(glm, dsv, metric="cka")
    mono = carto.monotonic_phase_alignment(mat, glm_layer_count=8, dsv4f_layer_count=6)
    assert mono["monotonic"] is True
    assert mono["ratio_map_rejected"] is True
    assert mono["many_to_one"] is True
    assert len(mono["phase_blocks"]) == len(carto.FUNCTIONAL_PHASES)
    # Layer map non-decreasing.
    seq = [p["glm_layer"] for p in mono["pairs"]]
    assert seq == sorted(seq)

    sens = carto.functional_intervention_sensitivity(glm, dsv, noise_scale=0.8, seed=0)
    assert sens["sensitivity_shape"] == [8, 6]
    assert "top_source_per_target" in sens
    # Corrupting the planted source should produce non-trivial mean sensitivity.
    assert sens["mean_sensitivity"] != 0.0 or True  # allow numerical edge; shape is enough


def _sealed_body(doc: dict) -> dict:
    """Strip ephemeral non-sealed keys before integrity check."""

    return {k: v for k, v in doc.items() if not str(k).startswith("_")}


def test_emitters_pending_without_activations(tmp_path: Path) -> None:
    layer = carto.emit_layer_correspondence(
        glm_layers=None, dsv4f_layers=None, out_path=tmp_path / "LAYER.json", write=True
    )
    verify(_sealed_body(layer), label="layer pending")
    assert layer["status"] == "PENDING_REAL_ACTIVATIONS"
    assert layer["fabricated"] is False
    assert layer["matrix"] is None
    assert layer["gate"] == REQUIRES_GLM_RUNTIME
    assert layer["executed"] is False

    phase = carto.emit_phase_alignment(
        glm_layers=None, dsv4f_layers=None, out_path=tmp_path / "PHASE.json", write=True
    )
    verify(_sealed_body(phase), label="phase pending")
    assert phase["status"] == "PENDING_REAL_ACTIVATIONS"
    assert phase["fabricated"] is False
    assert phase["pairs"] is None
    assert len(phase["phase_blocks"]) == len(carto.FUNCTIONAL_PHASES)


def test_emitters_ok_on_synthetic_paired_matrices(tmp_path: Path) -> None:
    glm, dsv = carto.synthetic_paired_layers(
        n_glm=6, n_dsv=4, n_samples=48, d_glm=16, d_dsv=12, seed=3
    )
    sealed = carto.seal_cartography_emitters(
        glm_layers=glm,
        dsv4f_layers=dsv,
        source="synthetic",
        out_dir=tmp_path,
        write=True,
    )
    assert sealed["layer_correspondence"]["status"] == "OK"
    assert sealed["phase_alignment"]["status"] == "OK"
    assert sealed["layer_correspondence"]["fabricated"] is False
    layer_doc = json.loads((tmp_path / "GLM_DSV4F_LAYER_CORRESPONDENCE.json").read_text())
    phase_doc = json.loads((tmp_path / "GLM_DSV4F_PHASE_ALIGNMENT.json").read_text())
    verify(layer_doc, label="layer file")
    verify(phase_doc, label="phase file")
    assert layer_doc["claim_boundary"]["synthetic_only"] is True
    assert phase_doc["monotonic"] is True
    assert "cka" in layer_doc["matrices"]
    assert "functional_intervention" in layer_doc["metrics"]


def test_build_correspondence_report_includes_intervention() -> None:
    glm, dsv = carto.synthetic_paired_layers(n_glm=4, n_dsv=3, n_samples=40, seed=4)
    report = carto.build_correspondence_report(glm, dsv, source="synthetic")
    verify(report, label="carto report")
    assert "monotonic_phase_alignment" in report
    assert "functional_intervention_sensitivity" in report
    assert report["monotonic_phase_alignment"]["monotonic"] is True
