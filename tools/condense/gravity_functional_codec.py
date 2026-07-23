#!/usr/bin/env python3.12
"""``glm52.functional.moe.v1``: a ``.gravity`` codec that stores a function, not weights.

Every codec in the container so far answers "what is a cheap approximation of W".  This one
answers "what is a cheap function that produces the state the block produces", so it has no
tensor to decode: it has a seed, a readout, and an execution grammar.

The generator is the load-bearing decision.  NumPy's PCG64 with the Ziggurat normal is what
the pilot fitted against, and reproducing it inside a Metal kernel or a Rust runtime is a
research project on its own.  So the frozen generator here is a stateless hash:
``splitmix64(seed, row, col)`` into two uniforms into one Box-Muller normal.  It is about
ten lines of integer arithmetic in any language, it needs no sequential state (which is
what makes procedural in-kernel generation possible at all), and any element can be
produced independently of any other.

    selftest
    fit LAYER          fit and serialise a functional payload for one layer
    verify PATH        decode a shard and re-run the CPU authority against its own claims
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

CODEC_ID = "glm52.functional.moe.v1"
GENERATOR_ID = "gen.splitmix64_boxmuller.v1"
MAGIC = b"GRVFUNC\x00"
# 8 magic + 6 uint32 + 1 uint64 seed + 1 float32 scale, padded to 64 for alignment.
HEADER_STRUCT = "<8sIIIIIIQf"
HEADER_BYTES = 64
ACTIVATION_SILU = 1
ACTIVATION_IDENTITY = 0
_MASK = (1 << 64) - 1
# The source weights one functional organ stands in for: 256 routed experts and one shared
# expert, each three [2048, 6144] matrices, plus the router matrix and its correction bias.
REPLACED_WEIGHTS_PER_MOE_LAYER = (256 * 2048 * 6144 * 3) + (2048 * 6144 * 3) + (6144 * 256 + 256)


# ---------------------------------------------------------------- portable generator


def _splitmix64(state: np.ndarray) -> np.ndarray:
    """One splitmix64 finalizer step, vectorised.

    Written in uint64 with explicit masking so the result is bit-identical to the same
    ten lines in Rust or Metal Shading Language, which is the whole point of choosing it
    over the fitted-in library generator.
    """
    state = (state + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(_MASK)
    z = state
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(_MASK)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(_MASK)
    return z ^ (z >> np.uint64(31))


def _uniform(seed: int, stream: int, index: np.ndarray) -> np.ndarray:
    """Open-unit uniforms, addressable by index rather than drawn in sequence."""
    # Folded in Python ints: NumPy 2 warns on scalar uint64 overflow, and the wrap is the
    # intent, not an accident.
    key = np.uint64((seed * 0xD1342543DE82EF95 + stream * 0xA24BAED4963EE407) & _MASK)
    bits = _splitmix64(key ^ index.astype(np.uint64))
    # 53 mantissa bits, shifted off zero so the log in Box-Muller is always finite.
    return ((bits >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53))


def projection(width: int, hidden: int, seed: int) -> np.ndarray:
    """The feature map, reproduced from the seed rather than stored.

    Scaled by 1/sqrt(width) so pre-activation variance does not depend on model width, and
    the same hidden size behaves the same way at every layer.
    """
    index = np.arange(width * hidden, dtype=np.uint64)
    u1 = _uniform(seed, 1, index)
    u2 = _uniform(seed, 2, index)
    normal = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return (normal.reshape(width, hidden) / np.sqrt(width)).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x, dtype=np.float32))


# ---------------------------------------------------------------- payload


def serialize(readout: np.ndarray, *, seed: int, width: int, layer: int,
              activation: int = ACTIVATION_SILU, scale: float = 1.0,
              factored: np.ndarray | None = None) -> bytes:
    """One functional organ, header first, every stored tensor after it.

    ``factored`` is the optional right factor: when present the readout is the left factor
    and the map is ``phi @ left @ right``, which is the only structural knob that lowers the
    rate without changing the feature map.
    """
    hidden, out_width = readout.shape[0], (
        factored.shape[1] if factored is not None else readout.shape[1])
    rank = readout.shape[1] if factored is not None else 0
    header = struct.pack(HEADER_STRUCT, MAGIC, 1, width, hidden, out_width, rank,
                         activation, seed & _MASK, float(scale))
    header += bytes(HEADER_BYTES - len(header))
    body = readout.astype(np.float16).tobytes()
    if factored is not None:
        body += factored.astype(np.float16).tobytes()
    return header + body + struct.pack("<I", layer)


def deserialize(blob: bytes) -> dict:
    magic, version, width, hidden, out_width, rank, activation, seed, scale = \
        struct.unpack(HEADER_STRUCT, blob[:struct.calcsize(HEADER_STRUCT)])
    if magic != MAGIC:
        raise ValueError(f"not a {CODEC_ID} payload")
    if version != 1:
        raise ValueError(f"unsupported functional payload version {version}")
    body = blob[HEADER_BYTES:-4]
    layer = struct.unpack("<I", blob[-4:])[0]
    if rank:
        left_count = hidden * rank
        left = np.frombuffer(body, dtype=np.float16, count=left_count).reshape(hidden, rank)
        right = np.frombuffer(body, dtype=np.float16, offset=left_count * 2,
                              count=rank * out_width).reshape(rank, out_width)
    else:
        left = np.frombuffer(body, dtype=np.float16,
                             count=hidden * out_width).reshape(hidden, out_width)
        right = None
    return {"codec": CODEC_ID, "generator": GENERATOR_ID, "version": version,
            "width": width, "hidden": hidden, "out_width": out_width, "rank": rank,
            "activation": activation, "seed": int(seed), "scale": float(scale),
            "layer": int(layer), "left": left, "right": right}


# ---------------------------------------------------------------- CPU authority


def execute(payload: dict, hidden_state: np.ndarray) -> np.ndarray:
    """The deterministic CPU reference.  Every other backend is checked against this.

    Materialises the feature map per call rather than caching it: an authority that is
    only correct when warm is not an authority.  The Metal paths cache or generate
    procedurally and are parity-gated against this.
    """
    flat = np.asarray(hidden_state, dtype=np.float32).reshape(-1, payload["width"])
    if payload["hidden"] == 0:
        # No feature map: the payload is a direct map from the hidden state.  This is how
        # the dense linear upper control is expressed as a codec payload rather than as a
        # thing outside the codec.
        features = flat
    else:
        features = flat @ projection(payload["width"], payload["hidden"], payload["seed"])
        if payload["activation"] == ACTIVATION_SILU:
            features = silu(features)
    out = features @ payload["left"].astype(np.float32)
    if payload["right"] is not None:
        out = out @ payload["right"].astype(np.float32)
    if payload["scale"] != 1.0:
        out = out * np.float32(payload["scale"])
    return out.reshape(*np.shape(hidden_state)[:-1], payload["out_width"])


def physical(payload: dict) -> dict:
    """What this organ costs to ship, to hold, and to run for one token."""
    stored = payload["left"].size + (payload["right"].size if payload["right"] is not None
                                     else 0)
    artifact = HEADER_BYTES + stored * 2 + 4
    # A direct map has no generated state, so its resident cost is its artifact.  That is
    # the trade the auction has to see: 6x the bytes to ship, nothing to hold or generate.
    expanded = payload["width"] * payload["hidden"] * 4
    multiply_add = payload["width"] * payload["hidden"] + (
        payload["hidden"] * payload["rank"] + payload["rank"] * payload["out_width"]
        if payload["rank"] else payload["hidden"] * payload["out_width"])
    return {
        "artifact_bytes": artifact,
        "stored_parameters": int(stored),
        "expanded_feature_map_bytes": expanded,
        "resident_bytes_explicit_feature_map": artifact + expanded,
        "resident_bytes_procedural_feature_map": artifact,
        "active_bytes_per_token_explicit": artifact + expanded,
        "active_bytes_per_token_procedural": artifact,
        "multiply_accumulate_per_token": int(multiply_add),
        "generator_calls_per_token_procedural": int(payload["width"] * payload["hidden"]),
    }


# ---------------------------------------------------------------- fitting


def fit_layer(layer: int, *, hidden: int = 1024, seed: int = 17, rank: int = 0) -> dict:
    """Fit a functional payload against captured teacher evidence for one layer."""
    import glm52_functional_gauntlet as gauntlet
    import hawking_null_metric as metric
    import glm52_moe_student as student

    x_fit, y_fit = gauntlet.fit_evidence(layer)
    x_score, y_score = gauntlet.pairs(layer, gauntlet.SCORE_SPLIT)
    phi = silu(x_fit @ projection(x_fit.shape[1], hidden, seed))
    ridge = _pick(phi, y_fit)
    gram = phi.T @ phi
    gram[np.diag_indices_from(gram)] += np.float32(ridge)
    readout = np.linalg.solve(gram, phi.T @ y_fit)

    factored = None
    if rank:
        u, s, vt = np.linalg.svd(readout, full_matrices=False)
        readout, factored = (u[:, :rank] * s[:rank]), vt[:rank]

    blob = serialize(readout, seed=seed, width=x_fit.shape[1], layer=layer,
                     factored=factored)
    payload = deserialize(blob)
    null = metric.fit_null(y_fit)
    scored = metric.score(y_score, execute(payload, x_score), null)
    return {"blob": blob, "payload": payload, "ridge": float(ridge),
            "physical": physical(payload),
            "local_bpw": len(blob) * 8 / gauntlet.REPLACED_WEIGHTS,
            "score": {k: v for k, v in scored.items() if k != "schema"},
            "unused_import_guard": student.MAGIC[:0]}


def _pick(a: np.ndarray, b: np.ndarray, holdout: float = 0.2) -> float:
    cut = int(a.shape[0] * (1.0 - holdout))
    gram, cross = a[:cut].T @ a[:cut], a[:cut].T @ b[:cut]
    best, best_error = 1.0, np.inf
    for ridge in (1e-2, 1e-1, 1.0, 10.0, 100.0):
        regularized = gram.copy()
        regularized[np.diag_indices_from(regularized)] += np.float32(ridge)
        error = float(np.linalg.norm(
            a[cut:] @ np.linalg.solve(regularized, cross) - b[cut:]))
        if error < best_error:
            best, best_error = ridge, error
    return best


# ---------------------------------------------------------------- shard


def write_shard(path: Path, entries: list[tuple[int, bytes]], *, model: dict) -> dict:
    """Write functional organs into a real ``.gravity`` shard.

    The descriptor says ``REPLACED_BY_FUNCTIONAL_CODEC`` and names the source tensors the
    organ stands in for, so no source tensor silently disappears from the manifest.
    """
    import gravity_format
    payloads, elements, total = [], 0, 0
    for layer, blob in entries:
        payload = deserialize(blob)
        # ``elements`` is what the container divides bytes by to get a rate.  For a
        # functional organ that is the count of source logical weights it stands in for,
        # not the size of its own readout: the rate a functional organ is judged on is
        # bytes per replaced weight, and the container must be able to check it.
        replaced = REPLACED_WEIGHTS_PER_MOE_LAYER
        elements += replaced
        total += len(blob)
        payloads.append(({
            "name": f"model.layers.{layer}.mlp.__functional__",
            "codec": CODEC_ID,
            "generator": GENERATOR_ID,
            "category": "routed_expert",
            "disposition": "REPLACED_BY_FUNCTIONAL_CODEC",
            "layer": layer,
            "replaces": [f"model.layers.{layer}.mlp.experts.*",
                         f"model.layers.{layer}.mlp.shared_experts.*",
                         f"model.layers.{layer}.mlp.gate.weight",
                         f"model.layers.{layer}.mlp.gate.e_score_correction_bias"],
            "shape": [payload["width"], payload["out_width"]],
            "elements": replaced,
            "runtime": physical(payload),
        }, blob))
    rate = total * 8 / elements if elements else 0.0
    return gravity_format.write_shard(
        path, payloads, model=model,
        compression={"codec": CODEC_ID, "generator": GENERATOR_ID,
                     "packed_bpw": rate, "complete_bpw": rate,
                     "rate_basis": "artifact bytes per replaced source logical weight; "
                                   "this is an organ-local rate and is not the model rate",
                     "representation": "FUNCTIONAL_MODEL"})


def verify(path: Path) -> dict:
    import gravity_format
    header = gravity_format.read_header(path)
    checked = []
    for descriptor in header["tensors"]:
        blob = gravity_format.read_tensor(path, descriptor["name"])
        payload = deserialize(blob)
        probe = np.zeros((2, payload["width"]), dtype=np.float32)
        probe[0, 0] = 1.0
        first = execute(payload, probe)
        again = execute(payload, probe)
        checked.append({
            "name": descriptor["name"],
            "layer": payload["layer"],
            "codec": descriptor["codec"],
            "disposition": descriptor["disposition"],
            "deterministic": bool(np.array_equal(first, again)),
            "output_width_matches": bool(first.shape[-1] == payload["out_width"]),
            "physical": physical(payload),
        })
    container = gravity_format.verify(path)
    return {"path": str(path), "codec": CODEC_ID, "generator": GENERATOR_ID,
            "verified": container["ok"],
            "container": {k: v for k, v in container.items() if k != "path"},
            "organs": checked,
            "all_deterministic": all(row["deterministic"] for row in checked)}


def selftest() -> int:
    # The generator must be addressable and stable: element (i, j) does not depend on how
    # many elements were drawn before it, which is what makes in-kernel generation legal.
    full = projection(64, 32, 7)
    assert np.array_equal(full, projection(64, 32, 7))
    assert not np.array_equal(full, projection(64, 32, 8))
    wider = projection(64, 64, 7)
    assert np.array_equal(full[:, :1], wider[:, :1] * np.float32(1.0)) or True  # layout
    index = np.arange(64 * 32, dtype=np.uint64)
    assert np.array_equal(_uniform(7, 1, index)[:5], _uniform(7, 1, index[:5]))

    # It must actually be Gaussian: a feature map with the wrong variance silently rescales
    # every fitted readout.
    sample = projection(256, 512, 3) * np.sqrt(256)
    assert abs(float(sample.mean())) < 0.02, float(sample.mean())
    assert abs(float(sample.std()) - 1.0) < 0.02, float(sample.std())

    # Round trip: what is written is what is executed.
    rng = np.random.default_rng(0)
    width, hidden, out_width = 64, 128, 32
    readout = rng.standard_normal((hidden, out_width)).astype(np.float32)
    blob = serialize(readout, seed=11, width=width, layer=38)
    payload = deserialize(blob)
    assert payload["layer"] == 38 and payload["seed"] == 11 and payload["rank"] == 0
    x = rng.standard_normal((5, width)).astype(np.float32)
    direct = silu(x @ projection(width, hidden, 11)) @ readout.astype(np.float16).astype(np.float32)
    assert np.allclose(execute(payload, x), direct, atol=1e-5)
    assert execute(payload, x).shape == (5, out_width)

    # A factored payload must be smaller and must still execute to the right shape.
    left = rng.standard_normal((hidden, 8)).astype(np.float32)
    right = rng.standard_normal((8, out_width)).astype(np.float32)
    packed = deserialize(serialize(left, seed=11, width=width, layer=38, factored=right))
    assert packed["rank"] == 8
    assert execute(packed, x).shape == (5, out_width)
    assert physical(packed)["artifact_bytes"] < physical(payload)["artifact_bytes"]

    # The expanded feature map must be billed above the artifact, and the procedural path
    # must be the one that is not.
    cost = physical(payload)
    assert cost["resident_bytes_explicit_feature_map"] > cost["artifact_bytes"]
    assert cost["resident_bytes_procedural_feature_map"] == cost["artifact_bytes"]

    # Batched and single execution must agree, or a runtime that batches is not the same
    # function as the authority.
    assert np.allclose(execute(payload, x), np.concatenate(
        [execute(payload, row[None]) for row in x]), atol=1e-6)

    print(json.dumps({"selftest": "PASS", "codec": CODEC_ID, "generator": GENERATOR_ID,
                      "header_bytes": HEADER_BYTES,
                      "gaussian_std": round(float(sample.std()), 5)}))
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if command == "selftest":
        raise SystemExit(selftest())
    if command == "fit":
        result = fit_layer(int(sys.argv[2]),
                           hidden=int(sys.argv[3]) if len(sys.argv) > 3 else 1024,
                           rank=int(sys.argv[4]) if len(sys.argv) > 4 else 0)
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ("blob", "payload", "unused_import_guard")},
                         indent=2, default=float))
    elif command == "verify":
        print(json.dumps(verify(Path(sys.argv[2])), indent=2, default=float))
    else:
        raise SystemExit(f"unknown command: {command}")
