"""G028: the anatomy must be repeatable, and its verdicts must be falsifiable."""
import json
import pathlib
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "future"))
from representational_anatomy import (  # noqa: E402
    AnatomyUnavailable,
    break_even_rank,
    FLOAT_DTYPES,
    GRAM_BLOCK_BYTES,
    LOWRANK_LIVE_BELOW,
    LOWRANK_LIVE_DEFICIT_PCT,
    SHARING_LIVE_BELOW,
    SHARING_LIVE_DEFICIT_PCT,
    _gram,
    _spectrum,
    anatomy_from_safetensors,
    hypotheses,
    main,
    parse_expert_key,
    read_header,
    resolve_layer,
    scan_expert_index,
)

_REPO = Path(__file__).resolve().parent.parent
_RECEIPT = _REPO / "receipts/future/OI_ANATOMY_Qwen3-30B-A3B.json"
_PINNED = Path(
    "/Users/scammermike/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-30B-A3B-Instruct-2507/"
    "snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
)


def test_o003_numbers_produce_the_verdicts_actually_measured():
    """Verdicts now read a DEFICIT against a matched null, not a constant."""
    a = {"cross_expert": [{"ratio": 0.9825, "n": 64, "null_ratio": 0.9825,
                           "deficit_pct": 0.00, "noise_floor_ratio": 0.9844}],
         "cross_layer": {"ratio": 0.9569, "n": 26, "deficit_pct": 0.01},
         "within_expert": {"ratio": 0.6162, "rank_90": 794, "shape": [4096, 1024],
                           "null_ratio": 0.6220, "deficit_pct": 0.93}}
    h = " ".join(hypotheses(a))
    assert "cross-expert sharing is DEAD" in h, h
    assert "low-rank structure is ABSENT" in h, h
    assert "NO LINEAR STRUCTURE AVAILABLE" in h, h


def test_the_second_specimen_reaches_the_same_verdict():
    a = {"cross_expert": [{"ratio": 0.9796, "n": 128, "null_ratio": 0.9796,
                           "deficit_pct": 0.002, "noise_floor_ratio": 0.9922}],
         "within_expert": {"ratio": 0.6085, "rank_90": 476, "shape": [2048, 768],
                           "null_ratio": 0.6090, "deficit_pct": 0.08}}
    assert "cross-expert sharing is DEAD" in " ".join(hypotheses(a))


def test_a_specimen_that_CONTRADICTS_the_prior_is_reported_as_live():
    """The prior must be able to die. A real deficit has to flip the verdict."""
    a = {"cross_expert": [{"ratio": 0.62, "n": 64, "null_ratio": 0.98,
                           "deficit_pct": 36.7, "noise_floor_ratio": 0.9844}],
         "within_expert": {"ratio": 0.21, "rank_90": 90, "shape": [1024, 1024],
                           "null_ratio": 0.606, "deficit_pct": 65.3}}
    h = " ".join(hypotheses(a))
    assert "cross-expert sharing is LIVE" in h, h
    assert "contradicts the family prior" in h, h
    assert "low-rank structure is REAL" in h, h


def test_a_ratio_below_the_old_threshold_is_NOT_live_when_the_null_agrees():
    """The defect the deficit rule exists to fix.

    Centred independent rows score exactly (n-1)/n, so at n=16 pure noise gives
    0.9375 and the old rule -- ratio < 0.95 means LIVE -- called it shared
    structure. The deficit reads zero and the verdict is DEAD.
    """
    a = {"cross_expert": [{"ratio": 0.9375, "n": 16, "null_ratio": 0.9375,
                           "deficit_pct": 0.0, "noise_floor_ratio": 0.9375}]}
    h = " ".join(hypotheses(a))
    assert a["cross_expert"][0]["ratio"] < SHARING_LIVE_BELOW, "premise broken"
    assert "DEAD" in h, h
    assert "LIVE" not in h, h


def test_the_false_cost_claim_is_gone():
    """A rank-495 factor of a 2048x768 tensor SAVES 11.4%; the old text said otherwise."""
    assert round(break_even_rank([2048, 768])) == 559
    a = {"within_expert": {"ratio": 0.7171, "rank_90": 495, "shape": [2048, 768],
                           "null_ratio": 0.8289, "deficit_pct": 13.49}}
    h = " ".join(hypotheses(a))
    assert "0.886x dense" in h, h
    assert "UNDER the 559 break-even rank" in h, h
    assert "costs more than dense" not in h, h
    assert "not of behaviour" in h, h
    assert "low-rank structure is REAL" in h, h
    assert "NO LINEAR STRUCTURE" not in h


def test_thresholds_are_the_published_ones():
    """Both rules, so the superseded pair stays visible next to what replaced it.

    0.95 was the noise floor at n=20 and called pure noise LIVE below it; 0.40
    was compared against a null that moves with aspect ratio. They are kept as
    named constants so the record shows what changed, and nothing reads them.
    """
    assert SHARING_LIVE_BELOW == 0.95 and LOWRANK_LIVE_BELOW == 0.40
    assert SHARING_LIVE_DEFICIT_PCT == 2.0 and LOWRANK_LIVE_DEFICIT_PCT == 10.0
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "tools" / "future" / "representational_anatomy.py").read_text()
    body = src[src.index("def hypotheses("):src.index("def anatomy_from_safetensors(")]
    assert "SHARING_LIVE_BELOW" not in body, "hypotheses() still reads the superseded constant"
    assert "LOWRANK_LIVE_BELOW" not in body, "hypotheses() still reads the superseded constant"


def test_no_hypotheses_without_measurements():
    assert hypotheses({}) == []


def test_float_dtypes_are_the_only_measurable_payloads():
    assert FLOAT_DTYPES == frozenset({"BF16", "F16", "F32", "F64"})
    assert GRAM_BLOCK_BYTES == 256 * 1024 * 1024


def _mlx_usable() -> bool:
    try:
        return _mlx_usable.__cache  # type: ignore[attr-defined]
    except AttributeError:
        pass
    try:
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        x = mx.zeros((2, 2))
        mx.eval(x)
        ok = True
    except Exception:
        ok = False
    _mlx_usable.__cache = ok  # type: ignore[attr-defined]
    return ok


def _payload(dtype: str, shape: list[int], seed: int = 1) -> bytes:
    n = 1
    for d in shape:
        n *= d
    if dtype == "F32":
        return b"".join(struct.pack("<f", float(seed + i) * 0.01) for i in range(n))
    if dtype in ("I8", "U8", "F8_E4M3"):
        return bytes((seed + i) % 256 for i in range(n))
    if dtype == "BF16":
        return b"".join(struct.pack("<H", (seed + i) & 0xFFFF) for i in range(n))
    raise ValueError(dtype)


def _write_st(path: Path, tensors: dict[str, tuple[str, list[int]]], seed: int = 1) -> None:
    """Write a safetensors file. Prefers mx.save_safetensors when Metal works;
    otherwise a spec-faithful header+payload so header-only tests still run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _mlx_usable():
        import mlx.core as mx
        mx_dt = {"F32": mx.float32, "I8": mx.int8, "U8": mx.uint8, "BF16": mx.bfloat16}
        arrays = {}
        ok = True
        for name, (dtype, shape) in tensors.items():
            if dtype not in mx_dt:
                ok = False
                break
            n = 1
            for d in shape:
                n *= d
            if dtype in ("I8", "U8"):
                arrays[name] = (mx.arange(n, dtype=mx_dt[dtype]) + seed).reshape(shape)
            else:
                arrays[name] = mx.random.normal(shape, key=mx.random.key(seed)).astype(mx_dt[dtype])
            seed += 1
        if ok and arrays:
            mx.save_safetensors(str(path), arrays)
            return
    items = {}
    for i, (name, (dtype, shape)) in enumerate(tensors.items()):
        items[name] = (dtype, shape, _payload(dtype, shape, seed=seed + i * 17))
    header: dict = {}
    blobs: list[bytes] = []
    off = 0
    for name, (dtype, shape, raw) in items.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - len(blob) % 8) % 8
    blob = blob + b" " * pad
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for b in blobs:
            fh.write(b)


def _index_dir(d: Path) -> dict:
    index = {}
    for f in sorted(d.glob("*.safetensors")):
        for k, e in read_header(str(f)).items():
            index[k] = (str(f), e)
    return index


def test_no_safetensors_in_dir_names_the_extensions_found(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"not-safetensors")
    (tmp_path / "model.pt").write_bytes(b"nope")
    with pytest.raises(AnatomyUnavailable, match=r"no \.safetensors") as ei:
        anatomy_from_safetensors(str(tmp_path))
    msg = str(ei.value)
    assert ".bin" in msg and ".pt" in msg, msg


def test_safetensors_present_zero_experts_looks_dense(tmp_path):
    _write_st(tmp_path / "model.safetensors", {
        "model.layers.0.mlp.down_proj.weight": ("F32", [4, 4]),
        "model.layers.0.self_attn.q_proj.weight": ("F32", [4, 4]),
    })
    with pytest.raises(AnatomyUnavailable, match="looks dense") as ei:
        anatomy_from_safetensors(str(tmp_path), layer=0)
    assert "expert" in str(ei.value).lower()


def test_experts_exist_but_not_at_requested_layer(tmp_path):
    tensors = {
        f"model.layers.2.mlp.experts.{i}.down_proj.weight": ("F32", [8, 4])
        for i in range(4)
    }
    _write_st(tmp_path / "model.safetensors", tensors)
    with pytest.raises(AnatomyUnavailable) as ei:
        anatomy_from_safetensors(str(tmp_path), layer=0)
    msg = str(ei.value)
    assert "layers 2..2, not 0" in msg, msg


def test_schemes_a_b_c_parse_to_layer_expert_projection():
    assert parse_expert_key("model.layers.1.mlp.experts.0.down_proj.weight") == (
        1, 0, "down_proj", "A")
    assert parse_expert_key("model.layers.1.mlp.experts.10.gate_proj.weight") == (
        1, 10, "gate_proj", "A")
    assert parse_expert_key("model.language_model.layers.0.mlp.experts.gate_up_proj") == (
        0, None, "gate_up_proj", "B")
    assert parse_expert_key("model.language_model.layers.0.mlp.experts.down_proj") == (
        0, None, "down_proj", "B")
    assert parse_expert_key("model.llm.layers.2.mlp.experts.w13_weight") == (
        2, None, "w13_weight", "C")
    assert parse_expert_key("model.llm.layers.2.mlp.experts.w2_weight") == (
        2, None, "w2_weight", "C")
    # lake-measured neighbours of A/B/C -- still the table, not new branches
    assert parse_expert_key("layers.0.ffn.experts.0.w1.weight") == (0, 0, "w1", "D")
    assert parse_expert_key(
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed"
    ) == (1, 0, "w1", "E")
    assert parse_expert_key(
        "model.language_model.layers.45.mlp.experts.0.down_proj.weight"
    ) == (45, 0, "down_proj", "A")  # F is A at the key; dtype refuses it later
    assert parse_expert_key("model.layers.10.feed_forward.expert_bias") is None
    assert parse_expert_key(
        "language_model.model.layers.1.block_sparse_moe.routed_expert_down_proj.weight"
    ) is None


def test_schemes_d_e_f_refused_as_prequantized_naming_the_scale(tmp_path):
    ddir = tmp_path / "scheme_d"
    d_tensors = {}
    for i in range(4):
        d_tensors[f"layers.0.ffn.experts.{i}.w1.weight"] = ("I8", [4, 4])
        d_tensors[f"layers.0.ffn.experts.{i}.w1.scale"] = ("F32", [4, 1])
    _write_st(ddir / "model.safetensors", d_tensors)
    with pytest.raises(AnatomyUnavailable, match="PRE-QUANTIZED") as ei:
        anatomy_from_safetensors(str(ddir), layer=0)
    msg = str(ei.value)
    assert "I8" in msg and "w1" in msg, msg
    assert "layers.0.ffn.experts.0.w1.scale" in msg, msg

    edir = tmp_path / "scheme_e"
    e_tensors = {}
    for i in range(4):
        e_tensors[f"layers.1.block_sparse_moe.experts.{i}.w1.weight_packed"] = ("U8", [4, 4])
        e_tensors[f"layers.1.block_sparse_moe.experts.{i}.w1.weight_scale"] = ("U8", [4, 1])
    _write_st(edir / "model.safetensors", e_tensors)
    with pytest.raises(AnatomyUnavailable, match="PRE-QUANTIZED") as ei:
        anatomy_from_safetensors(str(edir), layer=1)
    msg = str(ei.value)
    assert "U8" in msg and "w1" in msg, msg
    assert "weight_scale" in msg, msg

    fdir = tmp_path / "scheme_f"
    raw = {}
    for i in range(2):
        raw[f"model.layers.45.mlp.experts.{i}.down_proj.weight"] = ("F8_E4M3", [4, 4])
        raw[f"model.layers.45.mlp.experts.{i}.down_proj.weight_scale_inv"] = ("F32", [1, 1])
    _write_st(fdir / "model.safetensors", raw)
    with pytest.raises(AnatomyUnavailable, match="PRE-QUANTIZED") as ei:
        anatomy_from_safetensors(str(fdir), layer=45)
    msg = str(ei.value)
    assert "F8_E4M3" in msg and "down_proj" in msg, msg
    assert "weight_scale_inv" in msg, msg


def test_shared_expert_keys_are_excluded_from_the_expert_group(tmp_path):
    tensors = {
        f"model.layers.0.mlp.experts.{i}.down_proj.weight": ("F32", [8, 4])
        for i in range(4)
    }
    tensors["model.layers.0.mlp.shared_experts.0.down_proj.weight"] = ("F32", [8, 4])
    tensors["model.layers.0.mlp.shared_expert_gate.weight"] = ("F32", [1, 4])
    _write_st(tmp_path / "model.safetensors", tensors)
    index = _index_dir(tmp_path)
    parsed = [k for k in index if parse_expert_key(k) is not None]
    assert len(parsed) == 4, parsed
    assert all("shared_expert" not in k for k in parsed)
    assert parse_expert_key("model.layers.0.mlp.shared_experts.0.down_proj.weight") is None
    assert parse_expert_key("model.layers.0.mlp.shared_expert.down_proj.weight") is None
    by_layer, _, shared = scan_expert_index(index)
    assert shared is True
    assert set(by_layer[0]["down_proj"]) == {0, 1, 2, 3}, by_layer[0]["down_proj"]
    assert "shared_expert" not in by_layer[0]["down_proj"][0]
    out = anatomy_from_safetensors(str(tmp_path), layer=0)
    assert out["shared_expert_present"] is True
    assert out["cross_expert"][0]["n"] == 4, out["cross_expert"]
    assert out["cross_expert"][0]["n_experts_total"] == 4


def test_layer_none_picks_lowest_layer_that_has_experts(tmp_path):
    tensors = {
        "model.layers.0.mlp.down_proj.weight": ("F32", [4, 4]),
    }
    for i in range(4):
        tensors[f"model.layers.2.mlp.experts.{i}.down_proj.weight"] = ("F32", [8, 4])
        tensors[f"model.layers.5.mlp.experts.{i}.down_proj.weight"] = ("F32", [8, 4])
    _write_st(tmp_path / "model.safetensors", tensors)
    by_layer, _, _ = scan_expert_index(_index_dir(tmp_path))
    assert resolve_layer(by_layer, None) == 2
    assert sorted(by_layer) == [2, 5]
    out = anatomy_from_safetensors(str(tmp_path), layer=None)
    assert out["layer"] == 2, out
    assert out["layers_with_experts"] == [2, 5], out


def test_chunked_gram_matches_unchunked_to_1e_6():
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 20000), dtype=np.float32)
    # 64 * 1024 * 4 B = 256 KiB per block => ~20 feature blocks, so this is
    # actually chunked rather than a single matmul that happens to fit.
    block = 64 * 1024 * 4
    r_chunk = _spectrum(X, gram_block_bytes=block)
    r_full = _spectrum(X, gram_block_bytes=None)
    rel = abs(r_chunk["ratio"] - r_full["ratio"]) / max(abs(r_full["ratio"]), 1e-30)
    assert rel < 1e-6, (r_chunk, r_full, rel)
    Xc = X - X.mean(axis=0, keepdims=True)
    G_chunk = _gram(Xc, block)
    G_full = _gram(Xc, None)
    num = float(np.max(np.abs(G_chunk - G_full)))
    den = max(float(np.max(np.abs(G_full))), 1e-30)
    assert num / den < 1e-5, (num, den)


@pytest.mark.slow
def test_pinned_qwen3_30b_a3b_baseline_still_reproduces():
    if not _PINNED.is_dir() or not any(_PINNED.glob("*.safetensors")):
        pytest.skip(
            "VISIBLE SKIP: pinned Qwen3-30B-A3B snapshot is ABSENT at "
            f"{_PINNED}. This is the regression test for "
            "receipts/future/OI_ANATOMY_Qwen3-30B-A3B.json; it did NOT run."
        )
    receipt = json.loads(_RECEIPT.read_text())
    out = anatomy_from_safetensors(str(_PINNED), layer=0)
    got = {x["tensor"]: x for x in out["cross_expert"]}
    assert got["down_proj"]["ratio"] == 0.9796 and got["down_proj"]["rank_90"] == 110, got
    assert got["gate_proj"]["ratio"] == 0.9644 and got["gate_proj"]["rank_90"] == 108, got
    assert got["up_proj"]["ratio"] == 0.9559 and got["up_proj"]["rank_90"] == 106, got
    we = out["within_expert"]
    assert we["ratio"] == 0.7171 and we["rank_90"] == 495 and we["shape"] == [2048, 768], we
    assert out["hypotheses"] == receipt["hypotheses"]


def test_cli_named_refusal_is_exit_2_not_crash(tmp_path, capsys):
    (tmp_path / "weights.bin").write_bytes(b"x")
    rc = main([str(tmp_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no .safetensors" in err and ".bin" in err


def test_cli_auto_layer_emits_json_anatomy(tmp_path, capsys):
    tensors = {f"model.layers.2.mlp.experts.{i}.down_proj.weight": ("F32", [8, 4])
               for i in range(4)}
    _write_st(tmp_path / "model.safetensors", tensors)
    rc = main([str(tmp_path), "--layer", "auto"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    out = json.loads(captured.out)
    assert out["layer"] == 2
    assert out["cross_expert"][0]["n"] == 4
    assert out["hypotheses"]
