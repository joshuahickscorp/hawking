#!/usr/bin/env python3.12
"""Adversarial audit of the REAL full-model GPT-OSS-120B forward (gptoss_real_forward.py).

This is the Part II/2 adversarial suite for the G4 instrument. It is deliberately split into two
groups so it is SAFE to run while the durable G4 controller holds the single heavy Apple lease:

  (1) CHEAP group - runs now, no full forward, no touching the real 61 GB source. It pins the
      correctness of the parity-critical pure functions and the reader/tokenizer contracts using
      SYNTHETIC tiny tensors and metadata-only manifest lookups. Every test here is milliseconds.

  (2) HEAVY group - each test needs a full 36-block forward AND the heavy lease, so each is guarded
      with @HEAVY (skipif HAWKING_RUN_HEAVY != "1"). They encode the intended end-to-end audits and
      are meant to be run LATER, after G4 finishes, with:
          HAWKING_RUN_HEAVY=1 python3.12 -m pytest <thisfile> -q -k heavy

Honesty boundary: the cheap group proves the activation math and the plumbing contracts; it does NOT
prove HF numerical parity of a real forward (that is what the heavy group is for). The heavy group's
coherence test asserts the intended semantic outcome (capital-of-France -> " Paris") and will FAIL
loudly if a RoPE / interleave convention is wrong, which is the correct adversarial signal.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# tools/condense on sys.path (parent of this tests/ dir), matching sibling test modules.
_COND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COND not in sys.path:
    sys.path.insert(0, _COND)
_REPO = Path(_COND).resolve().parents[1]                       # tools/condense -> tools -> repo root

import gptoss_real_forward as rf          # noqa: E402  (apply_gate, block_n_attention, RealForward, ALPHA, LIMIT)
import gptoss_moe_runtime as rt           # noqa: E402  (ProvenanceReader, _swiglu split-half reference)
import eco_common                         # noqa: E402  (EcoError)

TOKENIZER_PATH = _REPO / "models" / "gpt-oss-120b" / "tokenizer.json"
MANIFEST_PATH = _REPO / "reports" / "condense" / "subbit_frontier" / "GRAVITY_120B_PROVENANCE.json"

# Heavy guard: these need a full forward + the single heavy Apple lease. Do NOT enable while G4 runs.
HEAVY = pytest.mark.skipif(
    os.environ.get("HAWKING_RUN_HEAVY") != "1",
    reason="heavy full-forward test; set HAWKING_RUN_HEAVY=1 to run (needs the heavy Apple lease; G4 must be idle)",
)


# ── synthetic helpers (no real source is ever read by the cheap group) ────────────────────
def _bf16_bytes(arr: np.ndarray) -> bytes:
    """Encode fp32 values as BF16 (top 16 bits) the same way the source shards store them."""
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    return (u32 >> 16).astype("<u2").tobytes()


def _build_synthetic_manifest(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """Write a tiny shard + provenance manifest describing two BF16 tensors. Returns (manifest_path,
    ground_truth). This exercises ProvenanceReader's real byte path on synthetic data only."""
    w = (np.arange(6, dtype=np.float32) - 2.0).reshape(2, 3)   # [2,3] tiny "weight"
    b = np.array([0.5, -1.25], dtype=np.float32)               # [2]   tiny "bias"
    wb, bb = _bf16_bytes(w), _bf16_bytes(b)
    shard = tmp_path / "synthetic_shard.safetensors"
    shard.write_bytes(wb + bb)
    manifest = {
        "schema": "test.synthetic.v1",
        "tensors": [
            {"tensor": "tiny.weight", "shape": [2, 3], "dtype": "bf16",
             "byte_range": [0, len(wb)], "shard_path": str(shard)},
            {"tensor": "tiny.bias", "shape": [2], "dtype": "bf16",
             "byte_range": [len(wb), len(wb) + len(bb)], "shard_path": str(shard)},
        ],
    }
    mp = tmp_path / "synthetic_manifest.json"
    mp.write_text(json.dumps(manifest))
    # ground truth is the bf16-roundtripped value (lossy: bf16 has 8 mantissa bits)
    gt_w = (np.frombuffer(wb, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(2, 3)
    gt_b = (np.frombuffer(bb, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(2)
    return mp, {"tiny.weight": gt_w, "tiny.bias": gt_b}


def _split_half_plain_silu(h: np.ndarray) -> np.ndarray:
    """The OLD, non-parity activation the module docstring warns about: split-half + plain SiLU,
    no interleave, no clamp, no (up+1), no alpha=1.702. Identical structure to rt._swiglu."""
    gate, up = np.split(h, 2, axis=-1)
    return (gate * (1.0 / (1.0 + np.exp(-gate)))) * up


# ==========================================================================================
# (1) CHEAP GROUP - runs now, safe while G4 holds the heavy lease. No full forward.
# ==========================================================================================

# ── apply_gate correctness (the parity fix) ───────────────────────────────────────────────
def test_apply_gate_matches_handcomputed_tiny_vector():
    """Hardcoded hand-computed expectation for a tiny interleaved vector. gate=[::2], up=[1::2],
    clamp gate<=7 / up in [-7,7], glu=gate*sigmoid(1.702*gate), out=(up+1)*glu."""
    v = np.array([-4., -3., -2., -1., 0., 1., 2., 3.], dtype=np.float64)
    got = rf.apply_gate(v)
    expected = np.array([0.00882945, 0.0, 0.0, 7.74263449])   # computed by hand off the spec
    assert got.shape == (4,)
    assert np.allclose(got, expected, atol=1e-6), (got, expected)


def test_apply_gate_constants_are_the_parity_constants():
    """The module must carry the transformers gpt-oss constants, not placeholders."""
    assert rf.ALPHA == 1.702
    assert rf.LIMIT == 7.0


def test_apply_gate_clamp_is_live():
    """A gate value far above the limit must be clamped to 7 (glu saturates), not passed through.
    Unclamped, (up+1)*gate*sigmoid(alpha*gate) would be ~300; clamped it is ~21."""
    v = np.array([100.0, 2.0], dtype=np.float64)              # gate=100 (clamps), up=2
    got = float(rf.apply_gate(v)[0])
    clamped = (2.0 + 1.0) * (7.0 * 1.0 / (1.0 + np.exp(-rf.ALPHA * 7.0)))
    assert abs(got - clamped) < 1e-4
    assert got < 25.0                                          # nowhere near the unclamped ~300


def test_apply_gate_uses_interleave_not_split_half():
    """Proves the parity fix is LIVE: the correct interleaved+clamped activation must DIFFER from the
    old split-half plain-SiLU on the same input (they only agree by accident, never generally)."""
    rng = np.random.default_rng(7)
    h = rng.standard_normal(5760).astype(np.float32) * 2.0    # real intermediate width, with clamp-range values
    correct = rf.apply_gate(h)
    wrong = _split_half_plain_silu(h)
    assert correct.shape == wrong.shape == (2880,)
    assert not np.allclose(correct, wrong, atol=1e-3), "apply_gate must not equal split-half plain SiLU"
    # and it must equal the module's own OLD reference structure only via the wrong path, confirming
    # rt._swiglu is the split-half variant (guards against a silent swap back to the buggy activation)
    assert np.allclose(wrong, rt._swiglu(h))


def test_apply_gate_is_pure_and_deterministic():
    """Same input -> bit-identical output; input is not mutated (pure function)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 8)).astype(np.float32)
    x_copy = x.copy()
    a = rf.apply_gate(x)
    b = rf.apply_gate(x)
    assert a.shape == (3, 4)
    assert np.array_equal(a, b)                               # deterministic, bit-identical
    assert np.array_equal(x, x_copy)                          # no in-place mutation


# ── tokenizer identity ────────────────────────────────────────────────────────────────────
def _load_tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    if not TOKENIZER_PATH.exists():
        pytest.skip(f"tokenizer.json absent at {TOKENIZER_PATH}")
    return tokenizers.Tokenizer.from_file(str(TOKENIZER_PATH))


def test_tokenizer_distinct_prompts_distinct_ids():
    tk = _load_tokenizer()
    a = tk.encode("The capital of France is").ids
    b = tk.encode("def fibonacci(n): return n").ids
    assert a and b
    assert a != b, "distinct prompts must tokenize to distinct id sequences"


def test_tokenizer_is_deterministic():
    tk = _load_tokenizer()
    s = "The capital of France is"
    assert tk.encode(s).ids == tk.encode(s).ids


def test_tokenizer_corrupted_path_raises(tmp_path):
    """A corrupted / nonexistent tokenizer file must fail loudly, not silently return junk ids."""
    tokenizers = pytest.importorskip("tokenizers")
    bad = tmp_path / "corrupt_tokenizer.json"
    bad.write_text("{ this is not valid tokenizer json ")
    with pytest.raises(Exception):
        tokenizers.Tokenizer.from_file(str(bad))
    with pytest.raises(Exception):
        tokenizers.Tokenizer.from_file(str(tmp_path / "does_not_exist.json"))


# ── ProvenanceReader / config-dimension guard (metadata + synthetic bytes only) ───────────
def test_reader_reads_synthetic_bf16_roundtrip(tmp_path):
    """ProvenanceReader.bf16 must decode a tensor from its byte range with the correct shape/values,
    exercised entirely on a synthetic shard (never the real 61 GB source)."""
    mp, gt = _build_synthetic_manifest(tmp_path)
    reader = rt.ProvenanceReader(str(mp))
    w = reader.bf16("tiny.weight")
    b = reader.bf16("tiny.bias")
    assert w.shape == (2, 3) and b.shape == (2,)
    assert np.array_equal(w, gt["tiny.weight"])
    assert np.array_equal(b, gt["tiny.bias"])


def test_reader_missing_tensor_name_is_detectable(tmp_path):
    """A wrong / missing tensor name must be detectable: raw() raises KeyError, by_name.get is None."""
    mp, _ = _build_synthetic_manifest(tmp_path)
    reader = rt.ProvenanceReader(str(mp))
    assert reader.by_name.get("block.999.mlp.gate.weight") is None
    with pytest.raises(KeyError):
        reader.raw("block.999.mlp.gate.weight")
    with pytest.raises(KeyError):
        reader.bf16("no.such.tensor")


def test_reader_wrong_expected_shape_is_detectable(tmp_path):
    """Config-dimension guard: a caller asserting an expected shape catches a mismatch from metadata
    alone (no bytes read). The real geometry constant HIDDEN=2880 is what such a guard protects."""
    mp, _ = _build_synthetic_manifest(tmp_path)
    reader = rt.ProvenanceReader(str(mp))

    def expect_shape(name: str, expected: tuple[int, ...]) -> None:
        actual = tuple(reader.by_name[name]["shape"])
        if actual != expected:
            raise ValueError(f"shape mismatch for {name}: {actual} != {expected}")

    expect_shape("tiny.weight", (2, 3))                        # correct -> no raise
    with pytest.raises(ValueError):
        expect_shape("tiny.weight", (2, 4))                    # wrong expected dim -> detected


def test_reader_corrupted_manifest_path_raises(tmp_path):
    """A nonexistent or malformed manifest must fail closed at construction, never load blindly."""
    with pytest.raises(eco_common.EcoError):
        rt.ProvenanceReader(str(tmp_path / "no_such_manifest.json"))
    bad = tmp_path / "malformed_manifest.json"
    bad.write_text("{ not json ")
    with pytest.raises(eco_common.EcoError):
        rt.ProvenanceReader(str(bad))


def test_real_manifest_geometry_matches_constants():
    """If the real provenance manifest is present, its declared geometry must match the module's
    hardcoded constants (metadata only; no tensor bytes are read, so this is safe while G4 runs).
    This is the config-dimension guard against a manifest/constant drift."""
    if not MANIFEST_PATH.exists():
        pytest.skip(f"real manifest absent at {MANIFEST_PATH}")
    reader = rt.ProvenanceReader(str(MANIFEST_PATH))
    assert tuple(reader.by_name["block.0.mlp.gate.weight"]["shape"]) == (128, rf.HIDDEN)
    assert tuple(reader.by_name["embedding.weight"]["shape"])[1] == rf.HIDDEN
    assert tuple(reader.by_name["unembedding.weight"]["shape"])[1] == rf.HIDDEN
    assert tuple(reader.by_name["norm.scale"]["shape"]) == (rf.HIDDEN,)
    # mlp1 up/gate width is 2*HIDDEN (interleaved gate/up) -> what apply_gate halves back to HIDDEN.
    assert tuple(reader.by_name["block.0.mlp.mlp1_weight.blocks"]["shape"])[1] == 2 * rf.HIDDEN


# ==========================================================================================
# (2) HEAVY GROUP - full forward + heavy lease required. GUARDED: skipped unless HAWKING_RUN_HEAVY=1.
#     Run later, after G4 finishes:  HAWKING_RUN_HEAVY=1 python3.12 -m pytest <thisfile> -q -k heavy
# ==========================================================================================

def _heavy_forward_and_tokenizer():
    """Build a RealForward on the real manifest + tokenizer. Skips (not fails) if the source shards
    are absent, so a heavy run on a box without the 61 GB source degrades gracefully."""
    tk = _load_tokenizer()
    if not MANIFEST_PATH.exists():
        pytest.skip(f"real manifest absent at {MANIFEST_PATH}")
    fwd = rf.RealForward(str(MANIFEST_PATH))
    if not fwd.source_present():
        pytest.skip("120B source shards absent; cannot run a real forward")
    return fwd, tk


def _top1(logits: np.ndarray) -> int:
    return int(np.argmax(logits[-1]))


@HEAVY
def test_heavy_capital_of_france_is_paris():
    """Coherence anchor: a correct real forward predicts ' Paris' as the top-1 next token for
    'The capital of France is'. If this fails, a RoPE / gate-up-interleave convention is wrong."""
    fwd, tk = _heavy_forward_and_tokenizer()
    ids = tk.encode("The capital of France is").ids
    logits = fwd.logits_for(ids, positions="last")
    top = _top1(logits)
    assert tk.decode([top]).strip().lower() == "paris", (top, repr(tk.decode([top])))


@HEAVY
def test_heavy_different_prompts_give_different_top1():
    """Two semantically different prompts must not collapse to the same top-1 token (a forward that
    ignores its input would)."""
    fwd, tk = _heavy_forward_and_tokenizer()
    a = fwd.logits_for(tk.encode("The capital of France is").ids, positions="last")
    b = fwd.logits_for(tk.encode("def quicksort(arr):\n    if len(arr)").ids, positions="last")
    assert _top1(a) != _top1(b)


@HEAVY
def test_heavy_corrupting_one_bounded_tensor_changes_logits():
    """Corrupting a single bounded tensor read (block.0 attention norm scale, ~2880 floats) must
    perturb the final logits. Proves every read is load-bearing, not silently dropped."""
    fwd, tk = _heavy_forward_and_tokenizer()
    ids = tk.encode("The capital of France is").ids
    base = fwd.logits_for(ids, positions="last").copy()

    corrupt = rf.RealForward(str(MANIFEST_PATH))
    orig_bf16 = corrupt.reader.bf16

    def perturbed(name: str) -> np.ndarray:
        a = orig_bf16(name)
        if name == "block.0.attn.norm.scale":
            a = a * 1.5 + 0.1                                  # bounded corruption of ONE tensor
        return a

    corrupt.reader.bf16 = perturbed                            # type: ignore[method-assign]
    got = corrupt.logits_for(ids, positions="last")
    assert not np.allclose(base, got, atol=1e-3), "corrupting a bounded tensor must change the logits"


@HEAVY
def test_heavy_missing_shard_raises():
    """If a required shard file is missing, the forward must fail closed (raise), never fabricate.
    We point the FIRST-read tensor (embedding) at a nonexistent shard so it fails immediately,
    without loading any real weights."""
    _, tk = _heavy_forward_and_tokenizer()
    fwd = rf.RealForward(str(MANIFEST_PATH))
    fwd.reader.by_name["embedding.weight"]["shard_path"] = "/nonexistent/missing_shard.safetensors"
    ids = tk.encode("The capital of France is").ids
    with pytest.raises((FileNotFoundError, OSError)):
        fwd.logits_for(ids, positions="last")


@HEAVY
def test_heavy_wrong_activation_ordering_degrades_coherence(monkeypatch):
    """Swapping the correct interleaved activation for the old split-half plain-SiLU must degrade
    coherence: 'The capital of France is' no longer predicts ' Paris' as top-1. This is the
    end-to-end proof that the activation parity fix is what makes the forward coherent."""
    fwd, tk = _heavy_forward_and_tokenizer()
    ids = tk.encode("The capital of France is").ids

    good = fwd.logits_for(ids, positions="last")
    assert tk.decode([_top1(good)]).strip().lower() == "paris", "baseline must be coherent first"

    # _moe_block looks up apply_gate as a module global at call time, so this monkeypatch takes effect.
    monkeypatch.setattr(rf, "apply_gate", _split_half_plain_silu)
    bad = rf.RealForward(str(MANIFEST_PATH)).logits_for(ids, positions="last")
    assert tk.decode([_top1(bad)]).strip().lower() != "paris", "wrong activation should break coherence"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-k", "not heavy"]))
