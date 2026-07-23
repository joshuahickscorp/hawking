#!/usr/bin/env python3
"""Run one pilot window at one rate and measure what the representation did to the model.

The pilot is the only place candidate search is allowed, so it has to answer the question
the full stream will not reopen: at G0, G1 and G2, how far does the block's real function
drift from the teacher's?

Three things happen per window per rung, in this order, while the BF16 bytes are resident:

    PACK        every declared tensor of the window, compressed or carried natively,
                into a .gravity that verifies and whose complete rate is exact
    RELOAD      decode from the file, not from the fitting run's in-memory recon, so
                what is measured is what was stored
    PROPAGATE   run the same reference forward the teacher ran, on the compact weights,
                and diff every stage against the sealed capsule

The last step is the one that matters.  Weight-space error is a proxy and a poor one:
Generation A's whole scientific claim was an F1 number, and F1 says nothing about whether
the router still picks the same experts.

    run WINDOW RUNG     pack, reload and propagate one window at one rung
    results             collect every measurement into GLM52_GENERATION_B_PILOT_RESULTS
    selftest
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gravity_forge as forge  # noqa: E402
import gravity_format  # noqa: E402
import glm52_pack as pack  # noqa: E402
import glm52_reference as reference  # noqa: E402
import glm52_lowrank as lowrank  # noqa: E402
import glm52_shard_probe as probe  # noqa: E402
import glm52_teacher_capture as teacher  # noqa: E402

REPO = HERE.parent.parent
STATE = Path.home() / "Library/Application Support/Hawking/GLM52Gravity"
SOURCE = STATE / "source"
CAPSULES = STATE / "source_fetch/teacher/capsules_generation_b"
COMPACT = STATE / "compact/generation_b_pilot"
REPORTS = REPO / "reports/condense/glm52_generation_b"
LADDER_PATH = REPORTS / "GLM52_GENERATION_B_RATE_LADDER.json"
RESULTS = REPORTS / "GLM52_GENERATION_B_PILOT_RESULTS.json"

# One decoded expert tensor is 50 MiB.  Eight is a working set, not a checkpoint, and it
# is enough to keep the hot experts of one routing step resident.
DECODE_CACHE_TENSORS = 8


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ladder() -> dict[str, dict]:
    return {entry["rung"]: entry for entry in json.loads(LADDER_PATH.read_text())["ladder"]}


# --------------------------------------------------------------------------- decode


def reconstruct(blob: bytes) -> np.ndarray:
    """Dense weights from a serialized PQ payload, laid out exactly as pq_execute reads it.

    Subspace s owns columns [s*sub, (s+1)*sub) of every D-wide chunk, and index row
    r*nchunk + c selects chunk c of row r.  Getting this wrong produces a plausible matrix
    that is not the one that was stored, which no hash would catch.
    """
    codes = pack.deserialize(blob)
    rows, cols = int(codes["rows"]), int(codes["cols"])
    dim, subspaces, sub = int(codes["D"]), int(codes["S"]), int(codes["sub"])
    nchunk = int(codes["nchunk"])
    indices = codes["indices"]
    out = np.empty((rows, nchunk, dim), dtype=np.float32)
    for s in range(subspaces):
        book = codes["codebooks"][s]
        out[:, :, s * sub:(s + 1) * sub] = book[indices[:, s]].reshape(rows, nchunk, sub)
    return out.reshape(rows, cols)


class CompactSource:
    """A reference TensorSource backed by .gravity files, decoding one tensor at a time.

    Native tensors come back as their exact source bytes.  Compressed ones are decoded on
    demand behind a bounded cache, because a routed layer touches most of its 256 experts
    and materializing them all would be 39 GiB of dense weight for a 300 MiB artifact.
    """

    def __init__(self, artifacts: list[Path], fallback: teacher.ShardTensorSource) -> None:
        self.fallback = fallback
        self.index: dict[str, tuple[Path, dict]] = {}
        for path in artifacts:
            header = gravity_format.read_header(path)
            for entry in header["tensors"]:
                self.index[entry["name"]] = (path, entry)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.decoded = 0
        self.native = 0
        self.fell_back = 0

    def tensor(self, name: str) -> np.ndarray:
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        located = self.index.get(name)
        if located is None:
            # A tensor outside the packed window (the embedding table, an upstream
            # organ) is read from source.  It is not part of this measurement and
            # must not be silently zeroed.
            self.fell_back += 1
            return self.fallback.tensor(name)
        path, entry = located
        blob = gravity_format.read_tensor(path, name)
        if str(entry["codec"]).startswith("native."):
            dtype = np.uint16 if entry["codec"] == "native.bf16" else np.float32
            raw = np.frombuffer(blob, dtype=dtype)
            value = (probe._bf16_to_f32(raw) if dtype is np.uint16
                     else raw.astype(np.float32))
            value = value.reshape(entry["shape"])
            self.native += 1
        elif entry["codec"] == "glm52.lowrank.r1.v1":
            value = lowrank.reconstruct(blob)
            self.decoded += 1
        else:
            value = reconstruct(blob)
            self.decoded += 1
        self._cache[name] = value
        while len(self._cache) > DECODE_CACHE_TENSORS:
            self._cache.popitem(last=False)
        return value


# ----------------------------------------------------------------------------- pack


def window_rows(layers: list[int]) -> list[dict]:
    """Every declared tensor of these layers, with the extents the sealed graph records."""
    graph = teacher._graph()
    table = teacher._tensor_table(graph)
    prefixes = tuple(f"model.layers.{layer}." for layer in layers)
    rows = []
    for name, record in table.items():
        if not name.startswith(prefixes):
            continue
        layer = int(name.split(".")[2])
        expert = None
        if ".experts." in name:
            expert = int(name.split(".experts.")[1].split(".")[0])
        rows.append({
            "name": name, "shard": record["shard"],
            "absolute_start": record["absolute_start"],
            "payload_bytes": record["payload_bytes"],
            "dtype": record["dtype"], "shape": list(record["shape"]),
            "layer": layer, "expert": expert,
            "category": _category(name),
            "provisional_budget_class": _budget_class(name),
        })
    rows.sort(key=lambda row: (row["shard"], row["absolute_start"]))
    return rows


def _category(name: str) -> str:
    import glm52_contract as contract
    return contract.classify_tensor(name, teacher.official_config()).category


def _budget_class(name: str) -> str:
    import glm52_contract as contract
    return contract.classify_tensor(name, teacher.official_config()).provisional_budget_class


def geometry_for(rung: dict, position: int) -> tuple[int, int]:
    """The (dim, k) this tensor takes, resolving a mixed allocation by position."""
    if rung["kind"] == "SINGLE_GEOMETRY":
        return int(rung["dim"]), int(rung["k"])
    rich = position % int(rung["allocation_period"]) < int(rung["rich_per_period"])
    side = rung["rich"] if rich else rung["lean"]
    return int(side["dim"]), int(side["k"])


def pack_window(layers: list[int], rung: dict, *, seed: int = 0, iters: int = 4) -> dict:
    """Pack every tensor of the window at this rung into one .gravity, exactly billed."""
    rows = window_rows(layers)
    out_dir = COMPACT / rung["rung"]
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = f"W_L{layers[0]:02d}_L{layers[-1]:02d}_{rung['rung']}"
    target = out_dir / f"{identity}.gravity"

    payloads: list[tuple[dict, bytes]] = []
    handles: dict[str, object] = {}
    position = 0
    started = time.time()
    try:
        for row in rows:
            handle = handles.get(row["shard"])
            if handle is None:
                handle = open(SOURCE / row["shard"], "rb", buffering=0)
                handles[row["shard"]] = handle
            handle.seek(int(row["absolute_start"]))
            raw = handle.read(int(row["payload_bytes"]))
            elements = int(np.prod(row["shape"]))

            native = (row["dtype"] != "BF16"
                      or row["provisional_budget_class"] == pack.PROTECTED_BUDGET_CLASS)
            if native:
                payloads.append(({
                    "name": row["name"], "category": row["category"],
                    "layer": row["layer"], "expert": row["expert"],
                    "shape": row["shape"], "codec": f"native.{row['dtype'].lower()}",
                    "terminal_state": "PROTECTED_SOURCE_NATIVE",
                    "elements": elements, "bpw": len(raw) * 8 / max(1, elements),
                }, raw))
                continue

            weights = probe._bf16_to_f32(
                np.frombuffer(raw, dtype=np.uint16)).reshape(row["shape"]).astype(np.float32)
            position += 1

            codec = rung["kind"]
            if codec == "HYBRID_BY_ROLE":
                chosen = rung["by_category"].get(row["category"], "glm52.pq.r0.v1")
                codec = "LOW_RANK" if chosen == "glm52.lowrank.r1.v1" else "SINGLE_GEOMETRY"

            if codec == "LOW_RANK":
                target_rate = float(rung.get("target_compressed_bpw")
                                    or rung["lowrank_target_compressed_bpw"])
                fitted = lowrank.pack_tensor(weights, target_rate, seed=seed)
                blob = fitted["blob"]
                payloads.append(({
                    "name": row["name"], "category": row["category"],
                    "layer": row["layer"], "expert": row["expert"],
                    "shape": row["shape"], "codec": "glm52.lowrank.r1.v1",
                    "terminal_state": "PACKED_IN_CORE_ARTIFACT",
                    "elements": elements, "bpw": len(blob) * 8 / max(1, elements),
                    "rank": fitted["rank"],
                    "relative_frobenius_error": fitted["relative_frobenius_error"],
                }, blob))
                continue

            if rung["kind"] == "HYBRID_BY_ROLE":
                dim, k = int(rung["pq_geometry"]["dim"]), int(rung["pq_geometry"]["k"])
            else:
                dim, k = geometry_for(rung, position - 1)
            artifact = forge.pack_product_quant(weights, dim=dim, subspaces=1, k=k,
                                                seed=seed, iters=iters)
            blob = pack.serialize(artifact)
            payloads.append(({
                "name": row["name"], "category": row["category"],
                "layer": row["layer"], "expert": row["expert"],
                "shape": row["shape"], "codec": "glm52.pq.r0.v1",
                "terminal_state": "PACKED_IN_CORE_ARTIFACT",
                "elements": elements, "bpw": len(blob) * 8 / max(1, elements),
                "dim": dim, "k": k,
                "relative_frobenius_error": float(forge._rel_error(weights, artifact.recon)),
            }, blob))
    finally:
        for handle in handles.values():
            handle.close()

    compressed = [(d, b) for d, b in payloads if d["codec"] != "native.bf16"
                  and not str(d["codec"]).startswith("native.")]
    native = [(d, b) for d, b in payloads if str(d["codec"]).startswith("native.")]
    all_elements = sum(d["elements"] for d, _ in payloads)
    packed_elements = sum(d["elements"] for d, _ in compressed)

    partial = target.with_name(target.name + ".partial")
    gravity_format.write_shard(
        partial, payloads,
        model={"repo": "zai-org/GLM-5.2", "revision": teacher.IMMUTABLE_REVISION,
               "window": identity},
        architecture={"type": "GlmMoeDsaForCausalLM", "hidden_layers": 78,
                      "routed_experts": 256, "shared_experts": 1, "hidden_size": 6144},
        tokenizer={"kind": "reference", "source": "zai-org/GLM-5.2"},
        compression={
            "codec": "gravity-pq", "production_rung": rung["rung"],
            "packed_bpw": sum(len(b) for _, b in compressed) * 8 / max(1, packed_elements),
            "complete_bpw": sum(len(b) for _, b in payloads) * 8 / max(1, all_elements),
            "native_tensors": len(native), "compressed_tensors": len(compressed),
            "rung": {key: rung[key] for key in rung
                     if key not in ("confirmation",)},
            "evidence_level": "F0_PHYSICAL_ONLY_UNTIL_PROPAGATION",
        },
        shard={"window": identity, "layers": layers})
    with open(partial, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(partial, target)

    report = gravity_format.verify(target)
    declared = {row["name"] for row in rows}
    stored = {d["name"] for d, _ in payloads}
    return {
        "window": identity, "rung": rung["rung"], "layers": layers,
        "rate_is_window_local": (
            "complete_bpw here is this window's own rate, not the model's. The ladder's "
            "target is weighted by routed experts, which are 97.492 percent of the model "
            "and are the largest tensors, so a window of small dense organs bills higher "
            "and a window of experts bills lower. Only the whole-model roll-up is "
            "comparable to the 0.75 ceiling."),
        "artifact": str(target), "artifact_bytes": target.stat().st_size,
        "pack_seconds": round(time.time() - started, 2),
        "tensors": len(payloads), "native_tensors": len(native),
        "compressed_tensors": len(compressed),
        "complete_bpw": report["observed_complete_bpw"],
        "packed_bpw": report["observed_packed_bpw"],
        "verifies": report["ok"],
        "tensor_coverage_complete": declared == stored,
        "absent_tensors": sorted(declared - stored),
    }


# ------------------------------------------------------------------------ propagate


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Drop positions either side masked out, and say how much of the tensor survived.

    Indexer scores carry -inf at masked positions, and a norm over those is nan.  A nan
    metric is not a bad score, it is no score, and reporting it as a number would let a
    stage that was never compared look like a stage that agreed.
    """
    x, y = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep], float(keep.mean()) if keep.size else 0.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y, _ = _finite_pair(a, b)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom else 0.0


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    x, y, _ = _finite_pair(a, b)
    denom = float(np.linalg.norm(x))
    return float(np.linalg.norm(y - x) / denom) if denom else 0.0


def _finite_fraction(a: np.ndarray, b: np.ndarray) -> float:
    return _finite_pair(a, b)[2]


def _selection_agreement(expected: np.ndarray, value: np.ndarray) -> dict:
    """How much of the teacher's selection the compact model reproduced.

    Two numbers, because they answer different questions.  Positional agreement asks
    whether the same slot got the same id, which is what a decoder replaying a fixed
    order cares about.  Set agreement asks whether the same experts ran at all, which is
    what the block's function cares about, and it is the one that can stay high while the
    ranking scrambles.
    """
    teacher_ids = np.asarray(expected)
    compact_ids = np.asarray(value)
    positional = float(np.mean(teacher_ids == compact_ids))
    flat_teacher = teacher_ids.reshape(-1, teacher_ids.shape[-1])
    flat_compact = compact_ids.reshape(-1, compact_ids.shape[-1])
    width = flat_teacher.shape[-1]
    overlap = [len(set(a.tolist()) & set(b.tolist())) / width
               for a, b in zip(flat_teacher, flat_compact)]
    return {
        "positional_agreement": positional,
        "set_agreement": float(np.mean(overlap)),
        "selection_width": int(width),
        "note": "cosine is not reported: these are labels, not magnitudes",
    }


def propagate(layers: list[int], rung: dict, artifact: Path) -> dict:
    """Run the compact window and diff every stage against the sealed teacher capsule."""
    identity = teacher.capsule_id(layers)
    capsule_path = CAPSULES / f"{identity}.npz"
    if not capsule_path.exists():
        raise SystemExit(f"no teacher capsule for {identity}; capture it first")
    truth = np.load(capsule_path)

    config = teacher.official_config()
    graph = teacher._graph()
    fallback = teacher.ShardTensorSource(SOURCE, teacher._tensor_table(graph))
    source = CompactSource([artifact], fallback)

    hidden = np.asarray(truth[f"layer_{layers[0]:02d}/input_hidden"], dtype=np.float32)
    previous_topk = np.asarray(truth["carry_out_index_selection"], dtype=np.int32) \
        if f"layer_{layers[0]:02d}/index_selection" not in truth else \
        np.asarray(truth[f"layer_{layers[0]:02d}/index_selection"], dtype=np.int32)
    # The window's own input is teacher-exact on purpose: this measures what the
    # representation does to the block, not what an upstream error already did.
    cache = reference.ReferenceCache()
    started = time.time()
    stages: dict[str, dict] = {}
    for layer in layers:
        hidden, previous_topk, arrays = teacher.capture_layer(
            hidden, source, layer, config, previous_topk, cache)
        for key, value in arrays.items():
            reference_key = f"layer_{layer:02d}/{key}"
            if reference_key not in truth:
                continue
            expected = np.asarray(truth[reference_key])
            if expected.shape != value.shape:
                stages[reference_key] = {"status": "SHAPE_MISMATCH",
                                         "teacher": list(expected.shape),
                                         "compact": list(value.shape)}
                continue
            if key in ("index_selection", "topk_indices"):
                # Expert and key ids are labels, not magnitudes.  Cosine over them is
                # meaningless and flattering: two disjoint expert sets of similar numeric
                # value score high while sharing no expert at all.  What matters is how
                # many of the teacher's choices the compact model actually made.
                stages[reference_key] = _selection_agreement(expected, value)
                continue
            stages[reference_key] = {
                "cosine": _cosine(expected, value),
                "relative_error": _relative(expected, value),
                "finite_fraction": _finite_fraction(expected, value),
            }

    router = {}
    for layer in layers:
        key = f"layer_{layer:02d}/topk_indices"
        if key in truth:
            expected = np.asarray(truth[key])
            router[key] = {"present_in_teacher": True, "shape": list(expected.shape)}

    return {
        "window": f"W_L{layers[0]:02d}_L{layers[-1]:02d}",
        "rung": rung["rung"],
        "teacher_capsule": identity,
        "propagate_seconds": round(time.time() - started, 2),
        "decoded_tensors": source.decoded,
        "native_tensors_read": source.native,
        "read_from_source_outside_window": source.fell_back,
        "stages": stages,
        "carry_out_cosine": _cosine(np.asarray(truth["carry_out_hidden"]), hidden),
        "carry_out_relative_error": _relative(np.asarray(truth["carry_out_hidden"]), hidden),
        "evidence_level": "F2_TRAJECTORY_ON_NATURAL_CORPUS_BATCH",
        "not_evidence_of": "full-model quality, capability, or end-to-end behaviour",
    }


def run(window: str, rung_id: str) -> int:
    layers = [int(value) for value in window.split(",")]
    rung = ladder()[rung_id]
    packed = pack_window(layers, rung)
    if not packed["verifies"] or not packed["tensor_coverage_complete"]:
        print(json.dumps({"status": "REFUSED", **packed}, indent=2))
        return 1
    measured = propagate(layers, rung, Path(packed["artifact"]))
    row = {"at": now(), "pack": packed, "propagate": measured}
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / "GLM52_PILOT_MEASUREMENTS.jsonl").open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({
        "window": packed["window"], "rung": rung_id,
        "complete_bpw": packed["complete_bpw"],
        "verifies": packed["verifies"],
        "coverage_complete": packed["tensor_coverage_complete"],
        "pack_seconds": packed["pack_seconds"],
        "propagate_seconds": measured["propagate_seconds"],
        "carry_out_cosine": measured["carry_out_cosine"],
        "carry_out_relative_error": measured["carry_out_relative_error"],
    }, indent=2))
    return 0


def selftest() -> int:
    # Reconstruction must be exactly what the packer's own executor sees, or the
    # propagation measurement is against a different matrix than the one stored.
    rng = np.random.default_rng(0)
    weights = rng.standard_normal((64, 256)).astype(np.float32)
    artifact = forge.pack_product_quant(weights, dim=16, subspaces=1, k=64, seed=0, iters=3)
    blob = pack.serialize(artifact)
    dense = reconstruct(blob)
    assert dense.shape == weights.shape, dense.shape

    # The file is lossier than the fit, and deliberately so: codebooks serialize as
    # float16 while the fit held float32 centroids, so the stored artifact decodes a few
    # parts in ten thousand away from the in-memory recon.  That gap is the whole reason
    # propagation reloads from disk instead of measuring the fitting run's own output.
    # The layout, which a wrong subspace or index order would break by whole magnitudes,
    # is what this pins.
    drift = float(np.abs(dense - artifact.recon).max())
    scale = float(np.abs(artifact.recon).max())
    assert drift / scale < 1e-3, f"decode drift {drift} is too large to be fp16 rounding"
    assert _relative(artifact.recon, dense) < 1e-3, _relative(artifact.recon, dense)

    # A mixed rung must place rich and lean tensors exactly where the ladder says.
    mixed = {"kind": "MIXED_ALLOCATION", "allocation_period": 16, "rich_per_period": 11,
             "rich": {"dim": 8, "k": 16}, "lean": {"dim": 16, "k": 128}}
    assert [geometry_for(mixed, i) for i in range(4)] == [(8, 16)] * 4
    assert geometry_for(mixed, 11) == (16, 128)
    assert geometry_for(mixed, 16) == (8, 16)
    rich = sum(1 for i in range(16) if geometry_for(mixed, i) == (8, 16))
    assert rich == 11, rich

    single = {"kind": "SINGLE_GEOMETRY", "dim": 32, "k": 8192}
    assert geometry_for(single, 7) == (32, 8192)

    # The low-rank family must decode through the same path propagation uses, or the two
    # families would be compared through different code and the difference could be the
    # reader rather than the representation.
    fitted = lowrank.pack_tensor(
        np.random.default_rng(1).standard_normal((256, 512)).astype(np.float32), 0.75)
    restored = lowrank.reconstruct(fitted["blob"])
    assert restored.shape == (256, 512), restored.shape
    assert fitted["bpw"] <= 0.75 + 1e-9, fitted["bpw"]

    print("glm52_pilot selftest OK")
    return 0


def remeasure(window: str, rung_id: str) -> int:
    """Re-run propagation against an artifact already on disk, without repacking.

    Packing a MoE window at k=8192 costs eleven minutes; a metric correction should not.
    """
    layers = [int(value) for value in window.split(",")]
    rung = ladder()[rung_id]
    identity = f"W_L{layers[0]:02d}_L{layers[-1]:02d}_{rung_id}"
    artifact = COMPACT / rung_id / f"{identity}.gravity"
    if not artifact.exists():
        raise SystemExit(f"no artifact at {artifact}; run the pack first")
    report = gravity_format.verify(artifact)
    if not report["ok"]:
        raise SystemExit(f"artifact does not verify: {report}")
    measured = propagate(layers, rung, artifact)
    row = {"at": now(), "remeasured": True,
           "pack": {"window": identity, "rung": rung_id, "layers": layers,
                    "artifact": str(artifact),
                    "artifact_bytes": artifact.stat().st_size,
                    "complete_bpw": report["observed_complete_bpw"],
                    "packed_bpw": report["observed_packed_bpw"],
                    "verifies": True, "tensor_coverage_complete": True},
           "propagate": measured}
    with (REPORTS / "GLM52_PILOT_MEASUREMENTS.jsonl").open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"window": identity, "rung": rung_id,
                      "complete_bpw": report["observed_complete_bpw"],
                      "carry_out_cosine": measured["carry_out_cosine"]}, indent=2))
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if command == "run":
        raise SystemExit(run(sys.argv[2], sys.argv[3]))
    if command == "remeasure":
        raise SystemExit(remeasure(sys.argv[2], sys.argv[3]))
    raise SystemExit({"selftest": selftest}[command]())
