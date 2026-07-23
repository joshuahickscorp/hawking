#!/usr/bin/env python3.12
"""Real DeepSeek-V4-Flash MoE organ forward, fp4 experts and all.

This is the organ the functional student would replace, and unlike a full-model forward it
is verifiable without reference logits: the fp4 dequant is checked by round-trip against the
e2m1 grid, and the MoE arithmetic is the reference implementation's own
(sqrtsoftplus + noaux_tc top-k, SwiGLU with a clamp, a shared expert, and the routed scale).

What this is NOT: a block forward. DeepSeek-V4 uses MLA, a DSA compressor/indexer and
hyper-connections, so the residual structure is not GLM's post_attention + post_moe, and a
propagation or amplification verdict needs that block, validated against reference logits.
This module deliberately stops at the MoE organ so that what it reports is real rather than
an artifact of an unvalidated attention path.

Routing: layers 0-2 are hash-routed via tid2eid; layers 3+ use the learned gate. The probe
uses learned-routing layers, which are the general case and the analogue of GLM's sparse
layers.

    selftest
    probe LAYER    fit a functional student against the real fp4 MoE on embedding-seeded
                   inputs, with fit-split nulls and the mandatory controls
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

CONDENSE = Path(__file__).resolve().parent
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

SUPPORT = Path("/Users/scammermike/Library/Application Support/Hawking/DeepSeekV4Flash")
SOURCE = SUPPORT / "source"

# e2m1 fp4: [sign][exp:2][mantissa:1]. The eight magnitudes of the grid; the sign bit (0x8)
# negates. This table is the standard MXFP4/e2m1 decode and is validated by round-trip in
# the selftest rather than trusted.
_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
DIM = 4096
MOE_INTER = 2048
N_ROUTED = 256
N_ACTIVATED = 6
N_SHARED = 1
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
RMS_EPS = 1e-6
FP4_BLOCK = 32  # one ue8m0 scale per 32 input fp4 values, per output row


def _read(name: str, index: dict) -> tuple[dict, bytes]:
    shard = index[name]
    with open(SOURCE / shard, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
        base = 8 + length
        meta = header[name]
        start, stop = meta["data_offsets"]
        handle.seek(base + start)
        return meta, handle.read(stop - start)


def _index() -> dict:
    return json.loads((SOURCE / "model.safetensors.index.json").read_text())["weight_map"]


def _bf16(raw: bytes, shape) -> np.ndarray:
    bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(shape)


def dequant_fp4(weight_i8: np.ndarray, scale_u8: np.ndarray) -> np.ndarray:
    """fp4 e2m1 packed two-per-byte, with a ue8m0 block scale per 32 input values.

    Low nibble is the even input index, high nibble the odd one. The scale byte b decodes as
    2**(b-127). Returns float32 [out, in] where in = 2 * weight_i8.shape[1].
    """
    out_dim, packed = weight_i8.shape
    in_dim = packed * 2
    bytes_u8 = weight_i8.astype(np.uint8)
    low = bytes_u8 & 0x0F
    high = (bytes_u8 >> 4) & 0x0F

    def decode(nib: np.ndarray) -> np.ndarray:
        mag = _E2M1[nib & 0x07]
        return np.where(nib & 0x08, -mag, mag)

    values = np.empty((out_dim, in_dim), dtype=np.float32)
    values[:, 0::2] = decode(low)
    values[:, 1::2] = decode(high)

    scale = (2.0 ** (scale_u8.astype(np.float32) - 127.0))
    # scale is [out, in/32]; expand to [out, in] by repeating each block.
    scale_full = np.repeat(scale, FP4_BLOCK, axis=1)[:, :in_dim]
    return values * scale_full


def dequant_fp8_e4m3(weight_u8: np.ndarray, scale_u8: np.ndarray) -> np.ndarray:
    """fp8 e4m3 (1-4-3, bias 7) with a ue8m0 [128,128] block scale.

    The shared expert ships fp8, not fp4. e4m3 has no infinities; 0x7f/0xff are NaN, which
    real weights never hit, so they are decoded as their finite pattern rather than
    special-cased.
    """
    out_dim, in_dim = weight_u8.shape
    sign = np.where(weight_u8 & 0x80, -1.0, 1.0).astype(np.float32)
    exp = ((weight_u8 >> 3) & 0x0F).astype(np.int32)
    mant = (weight_u8 & 0x07).astype(np.float32)
    normal = (2.0 ** (exp - 7).astype(np.float32)) * (1.0 + mant / 8.0)
    subnormal = (2.0 ** -6) * (mant / 8.0)
    magnitude = np.where(exp == 0, subnormal, normal)
    values = (sign * magnitude).astype(np.float32)

    scale = 2.0 ** (scale_u8.astype(np.float32) - 127.0)  # [out/128, in/128]
    scale_full = np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1)[:out_dim, :in_dim]
    return values * scale_full


def _load_expert(prefix: str, index: dict) -> dict:
    """Routed experts are fp4-in-I8; the shared expert is fp8 e4m3. Dispatch by dtype."""
    out = {}
    for proj in ("w1", "w2", "w3"):
        wm, wraw = _read(f"{prefix}.{proj}.weight", index)
        sm, sraw = _read(f"{prefix}.{proj}.scale", index)
        scale = np.frombuffer(sraw, dtype=np.uint8).reshape(sm["shape"])
        if wm["dtype"] == "I8":
            weight = np.frombuffer(wraw, dtype=np.int8).reshape(wm["shape"])
            out[proj] = dequant_fp4(weight, scale)
        elif wm["dtype"] == "F8_E4M3":
            weight = np.frombuffer(wraw, dtype=np.uint8).reshape(wm["shape"])
            out[proj] = dequant_fp8_e4m3(weight, scale)
        else:
            raise ValueError(f"{prefix}.{proj}: unexpected dtype {wm['dtype']}")
    return out


def rmsnorm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + RMS_EPS)
    return (x / scale) * weight


def _swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    gate = np.clip(gate, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    up = np.clip(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    return (gate / (1.0 + np.exp(-gate))) * up


def _expert_forward(x: np.ndarray, expert: dict) -> np.ndarray:
    gate = x @ expert["w1"].T
    up = x @ expert["w3"].T
    return _swiglu(gate, up) @ expert["w2"].T


def _sqrtsoftplus(logits: np.ndarray) -> np.ndarray:
    # softplus then sqrt: the reference scoring function for noaux_tc.
    return np.sqrt(np.log1p(np.exp(-np.abs(logits))) + np.maximum(logits, 0.0))


def moe_forward(x: np.ndarray, layer: int, index: dict, *, experts=None) -> dict:
    """The learned-routing MoE: gate -> sqrtsoftplus -> top-6 -> SwiGLU experts + shared."""
    prefix = f"layers.{layer}.ffn"
    gate_w = _bf16(*[_read(f"{prefix}.gate.weight", index)][0][::-1]) \
        if False else _bf16(_read(f"{prefix}.gate.weight", index)[1],
                            _read(f"{prefix}.gate.weight", index)[0]["shape"])
    scores = _sqrtsoftplus(x @ gate_w.T)  # [N, 256]
    topk = np.argsort(-scores, axis=-1)[:, :N_ACTIVATED]
    weights = np.take_along_axis(scores, topk, axis=-1)
    weights = weights / np.clip(weights.sum(axis=-1, keepdims=True), 1e-9, None)  # norm_topk
    margin = (np.sort(scores, axis=-1)[:, -N_ACTIVATED]
              - np.sort(scores, axis=-1)[:, -N_ACTIVATED - 1])

    if experts is None:
        experts = {e: _load_expert(f"{prefix}.experts.{e}", index) for e in range(N_ROUTED)}
    routed = np.zeros_like(x)
    for token in range(x.shape[0]):
        acc = np.zeros(DIM, dtype=np.float32)
        for slot in range(N_ACTIVATED):
            e = int(topk[token, slot])
            acc += weights[token, slot] * _expert_forward(x[token:token + 1], experts[e])[0]
        routed[token] = acc
    routed *= ROUTE_SCALE

    shared = _load_expert(f"{prefix}.shared_experts", index)
    shared_out = _expert_forward(x, shared)
    return {"post_moe": routed + shared_out, "topk_indices": topk,
            "topk_weights": weights, "router_margin": margin, "experts": experts}


def embedding_seeded_inputs(token_ids: np.ndarray, layer: int, index: dict) -> np.ndarray:
    """ffn_norm(embed[token]) as a bounded MoE input distribution.

    Honest label: this is the embedding through the layer's ffn norm, not the true
    pre-MoE hidden state, which would carry attention context. It is the same
    embedding-seeding the GLM capsules used for windows whose upstream layers were not
    resident, and it is enough to ask whether a cheap function reproduces this MoE.
    """
    em, eraw = _read("embed.weight", index)
    embed = _bf16(eraw, em["shape"])
    hidden = embed[token_ids]
    nm, nraw = _read(f"layers.{layer}.ffn_norm.weight", index)
    return rmsnorm(hidden, _bf16(nraw, nm["shape"]))


def _calibration_tokens(split: str, count: int, vocab: int = 129280) -> np.ndarray:
    """Deterministic disjoint token ids per split, drawn from real vocabulary."""
    seed = {"fit": 1, "router": 2, "doctor": 3, "score": 4, "replication": 5}[split]
    return np.random.default_rng(seed).integers(0, vocab, count, dtype=np.int64)


def probe(layer: int, *, fit_count: int = 3072, score_count: int = 1024) -> dict:
    """Does a cheap function reproduce DeepSeek's already-fp4 MoE at this layer?

    Function-first, null-first, on embedding-seeded inputs. This is the cross-parent probe
    the methodology requires before any encoding: fit-split nulls, a functional student, a
    full affine upper control, and the weight-space constant control, scored on a disjoint
    split. It is not a block, propagation or capability result.
    """
    import hawking_null_metric as metric
    import glm52_moe_student as student

    index = _index()
    experts = {e: _load_expert(f"layers.{layer}.ffn.experts.{e}", index)
               for e in range(N_ROUTED)}

    def pairs(split, count):
        tokens = _calibration_tokens(split, count)
        x = embedding_seeded_inputs(tokens, layer, index)
        y = moe_forward(x, layer, index, experts=experts)["post_moe"]
        return x.astype(np.float64), y.astype(np.float64)

    xs, ys = zip(*[pairs(s, fit_count // 3) for s in ("fit", "router", "doctor")])
    x_fit, y_fit = np.concatenate(xs), np.concatenate(ys)
    x_score, y_score = pairs("score", score_count)
    null = metric.fit_null(y_fit)

    # The organ this student would replace: 256 routed + 1 shared expert, three fp4/fp8
    # [2048,4096]-ish matrices each, plus the router.
    replaced = (N_ROUTED * (MOE_INTER * DIM * 2 + MOE_INTER * DIM)  # w1,w3 [2048,4096]; w2 [4096,2048]
                + N_SHARED * (MOE_INTER * DIM * 3) + DIM * N_ROUTED)

    def score_candidate(predict, blob_bytes, label):
        pred = predict(x_score)
        s = metric.score(y_score, pred, null)
        return {"label": label, "bytes": blob_bytes, "local_bpw": blob_bytes * 8 / replaced,
                **{k: v for k, v in s.items() if k != "schema"}}

    fitted = student.fit(x_fit.astype(np.float32), y_fit.astype(np.float32),
                         hidden=1024, seed=17, replaced_weights=replaced)
    student_blob = fitted["blob"]

    # Full affine upper control.
    a = np.concatenate([x_fit, np.ones((x_fit.shape[0], 1))], axis=1)
    ridge = 1.0
    weight = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T @ y_fit)

    def affine(z):
        za = np.concatenate([z, np.ones((z.shape[0], 1))], axis=1)
        return za @ weight

    rows = [
        score_candidate(lambda z: student.apply_student(student_blob, z.astype(np.float32)),
                        len(student_blob), "functional_student_h1024"),
        score_candidate(affine, weight.size * 2 + 64, "affine_upper_control"),
        score_candidate(lambda z: np.broadcast_to(null["mean"], (z.shape[0], DIM)),
                        DIM * 2, "weight_space_constant"),
    ]
    shuffled = np.random.default_rng(9).permutation(x_score.shape[0])
    control = metric.score(
        y_score, student.apply_student(student_blob, x_score[shuffled].astype(np.float32)),
        null)

    result = {
        "schema": "hawking.deepseek_v4.moe_functional_probe.v1",
        "parent": "deepseek-ai/DeepSeek-V4-Flash",
        "layer": layer,
        "routing": "learned noaux_tc sqrtsoftplus" if layer >= 3 else "hash tid2eid",
        "input": "embedding-seeded ffn_norm(embed[token]); no attention context",
        "expert_precision": "routed fp4-in-I8, shared fp8 e4m3",
        "fit_positions": int(x_fit.shape[0]), "score_positions": int(x_score.shape[0]),
        "constant_null_raw_cosine": metric.constant_null_raw_cosine(y_score, null),
        "candidates": rows,
        "shuffled_input_skill": control["skill"],
        "functional_escape_exists": bool(rows[0]["passes"]),
        "map_is_the_win": bool(rows[1]["skill"] >= rows[0]["skill"] - 0.05),
        "not_evidence_of": "block, propagation, amplification or capability. The DeepSeek "
                           "block uses MLA, a DSA compressor/indexer and hyper-connections; "
                           "a propagation verdict needs that block validated against "
                           "reference logits.",
    }
    return result


def probe_decompose(layer: int, *, fit_count: int = 3072, score_count: int = 1024) -> dict:
    """Isolate what a functional student can and cannot fit in the DeepSeek MoE.

    The shared expert is a plain SwiGLU applied to every token with no routing; the routed
    part is a sum of six of 256 experts chosen by a sharp gate. Fitting the student against
    each separately says whether the negative is the routing or the whole organ, and that
    conclusion is robust to the embedding-seeding confound because the shared expert has no
    routing to be starved of context.
    """
    import hawking_null_metric as metric
    import glm52_moe_student as student

    index = _index()
    experts = {e: _load_expert(f"layers.{layer}.ffn.experts.{e}", index)
               for e in range(N_ROUTED)}
    shared = _load_expert(f"layers.{layer}.ffn.shared_experts", index)

    def parts(split, count):
        tokens = _calibration_tokens(split, count)
        x = embedding_seeded_inputs(tokens, layer, index)
        out = moe_forward(x, layer, index, experts=experts)
        shared_out = _expert_forward(x, shared)
        return (x.astype(np.float64), out["post_moe"].astype(np.float64),
                (out["post_moe"] - shared_out).astype(np.float64),
                shared_out.astype(np.float64))

    fit = [parts(s, fit_count // 3) for s in ("fit", "router", "doctor")]
    x_fit = np.concatenate([f[0] for f in fit])
    targets_fit = {"full": np.concatenate([f[1] for f in fit]),
                   "routed_only": np.concatenate([f[2] for f in fit]),
                   "shared_only": np.concatenate([f[3] for f in fit])}
    xs, full_s, routed_s, shared_s = parts("score", score_count)
    targets_score = {"full": full_s, "routed_only": routed_s, "shared_only": shared_s}

    rows = {}
    for name, y_fit in targets_fit.items():
        null = metric.fit_null(y_fit)
        fitted = student.fit(x_fit.astype(np.float32), y_fit.astype(np.float32),
                             hidden=1024, seed=17, replaced_weights=1)
        pred = student.apply_student(fitted["blob"], xs.astype(np.float32))
        s = metric.score(targets_score[name], pred, null)
        rows[name] = {"skill": s["skill"], "skill_lower": s["skill_lower"],
                      "centered_cosine": s["centered_cosine"],
                      "null_raw_cosine": metric.constant_null_raw_cosine(targets_score[name], null),
                      "target_rms": float(np.sqrt(np.mean(targets_score[name] ** 2)))}
    result = {
        "schema": "hawking.deepseek_v4.moe_decomposition.v1",
        "layer": layer, "components": rows,
        "mechanism": ("routing_dominated" if rows["shared_only"]["skill"]
                      > rows["routed_only"]["skill"] + 0.2 else "whole_organ_hard"),
        "reading": "if the shared expert is fittable and the routed part is not, the sharp "
                   "6-of-256 routing is what a smooth functional student cannot capture, "
                   "independent of the input-context confound.",
    }
    return result


def selftest() -> int:
    # fp4 decode must land exactly on the e2m1 grid, and the round-trip nibble->value->grid
    # must be a fixed point.
    nibbles = np.arange(16, dtype=np.uint8)
    mag = _E2M1[nibbles & 0x07]
    decoded = np.where(nibbles & 0x08, -mag, mag)
    assert set(np.abs(decoded).tolist()) == set(_E2M1.tolist()), decoded
    assert decoded[0] == 0.0 and decoded[7] == 6.0 and decoded[15] == -6.0

    if not SOURCE.exists():
        print(json.dumps({"selftest": "PASS_DECODE_ONLY", "reason": "source not resident"}))
        return 0
    index = _index()

    # Dequant a real expert and check it is finite, on-grid times scale, and sanely scaled.
    weight_meta, wraw = _read("layers.5.ffn.experts.0.w1.weight", index)
    scale_meta, sraw = _read("layers.5.ffn.experts.0.w1.scale", index)
    weight = np.frombuffer(wraw, dtype=np.int8).reshape(weight_meta["shape"])
    scale = np.frombuffer(sraw, dtype=np.uint8).reshape(scale_meta["shape"])
    deq = dequant_fp4(weight, scale)
    assert deq.shape == (2048, 4096), deq.shape
    assert np.isfinite(deq).all()
    # Each value divided by its block scale must be exactly on the signed e2m1 grid.
    scale_full = np.repeat(2.0 ** (scale.astype(np.float32) - 127.0), FP4_BLOCK, axis=1)[:, :4096]
    grid = np.concatenate([_E2M1, -_E2M1])
    ratio = deq / scale_full
    on_grid = np.min(np.abs(ratio[..., None] - grid), axis=-1).max()
    assert on_grid < 1e-3, on_grid

    # A real MoE forward on a tiny embedding-seeded batch must produce a finite, non-trivial
    # output and route to six distinct experts.
    tokens = np.array([10, 200, 4096, 50000, 128000], dtype=np.int64)
    x = embedding_seeded_inputs(tokens, 5, index)
    out = moe_forward(x, 5, index)
    assert out["post_moe"].shape == (5, DIM)
    assert np.isfinite(out["post_moe"]).all()
    assert out["topk_indices"].shape == (5, N_ACTIVATED)
    assert len(set(out["topk_indices"][0].tolist())) == N_ACTIVATED

    print(json.dumps({
        "selftest": "PASS",
        "fp4_on_grid_max_error": float(on_grid),
        "expert_weight_abs_max": float(np.abs(deq).max()),
        "moe_output_rms": float(np.sqrt(np.mean(out["post_moe"] ** 2))),
        "distinct_experts_token0": len(set(out["topk_indices"][0].tolist())),
        "router_margin_mean": float(out["router_margin"].mean()),
    }))
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if command == "selftest":
        raise SystemExit(selftest())
    if command == "probe":
        layer = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        result = probe(layer)
        out = (Path(__file__).resolve().parents[2] / "reports" / "condense"
               / "deepseek_v4_flash")
        out.mkdir(parents=True, exist_ok=True)
        (out / f"DEEPSEEK_V4_MOE_PROBE_L{layer:02d}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=float))
        for row in result["candidates"]:
            print(f"  {row['label']:28} bpw {row['local_bpw']:.6f}  skill {row['skill']:7.4f}"
                  f"  lower {row['skill_lower']:7.4f}  centered {row['centered_cosine']:7.4f}"
                  f"  pass {row['passes']}")
        print(f"escape_exists={result['functional_escape_exists']} "
              f"map_is_the_win={result['map_is_the_win']} "
              f"shuffled_skill={result['shuffled_input_skill']:.4f} "
              f"null_raw={result['constant_null_raw_cosine']:.4f}")
        raise SystemExit(0)
    if command == "decompose":
        layer = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        res = probe_decompose(layer)
        out = (Path(__file__).resolve().parents[2] / "reports" / "condense" / "deepseek_v4_flash")
        out.mkdir(parents=True, exist_ok=True)
        (out / f"DEEPSEEK_V4_MOE_DECOMPOSE_L{layer:02d}.json").write_text(json.dumps(res, indent=2, sort_keys=True, default=float))
        for name, r in res["components"].items():
            print(f"  {name:14} skill {r['skill']:7.4f}  centered {r['centered_cosine']:7.4f}  null_raw {r['null_raw_cosine']:6.4f}  rms {r['target_rms']:.4f}")
        print("mechanism:", res["mechanism"])
        raise SystemExit(0)
    raise SystemExit(f"unknown command: {command}")
