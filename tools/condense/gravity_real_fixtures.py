#!/usr/bin/env python3.12
"""Real packed GLM-5.2 tensors as fixtures, taken from a live campaign without disturbing it.

Mandate section 6: a kernel does not promote on synthetic-only evidence.  Every later GPU
phase has to run against tensors that came out of the production packer, at the production
rung, with the production geometry -- not a ``standard_normal`` that happens to have the
right shape.  This module is the only sanctioned way to reach them.

The artifact directory is the LIVE campaign's active output.  Interference is prevented
structurally rather than by care:

  * nothing here opens a file for writing, and nothing here imports ``shutil`` or calls
    ``unlink``/``rename``/``replace``/``mkdir`` against the artifact root.  Reads go through
    :mod:`gravity_format`, which is read-only by construction.  ``test_gravity_real_fixtures``
    asserts that property against this file's own source, so it cannot rot.
  * a shard is only eligible once its mtime is older than :data:`SAFETY_AGE_SECONDS`
    (default two hours).  A shard the packer is still writing is younger than that, so an
    in-flight body can never be read half-written.
  * ``.tmp`` files -- what ``gravity_format.write_shard`` writes before its atomic rename --
    are never enumerated at all.
  * eligibility is not trust: before any tensor is decoded, the whole shard is put through
    ``gravity_format.verify`` (body sha256, every tensor sha256, rate self-consistency) and
    that verdict is memoized per (path, size, mtime), so a 293 MB body is hashed once and
    not once per tensor.

What this module does NOT provide is real activations.  The teacher capsules that would
carry them live under the live campaign's support root, so they are reported by path and
marked UNAVAILABLE rather than read.  The substitute is synthetic, and :class:`Fixture`
makes the distinction impossible to drop: ``activation_source`` is a required field with
exactly two legal values, so a benchmark that consumed a synthetic activation cannot
serialize itself as though it had not.

CPU only.  Nothing here touches Metal, MPS, or a GPU.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_pack  # noqa: E402
import gravity_format  # noqa: E402
from bounded_cache import PressureAwareCache  # noqa: E402

SCHEMA = "hawking.glm52.real_artifact_survey.v1"
ARTIFACT_DIR = Path(os.environ.get(
    "GLM52_GRAVITY_ARTIFACT_DIR", "/Users/scammermike/Desktop/GLM52-Gravity-SubBit"))
# Two hours.  The packer emits one ~293 MB shard every few minutes, so anything this old
# has been closed and renamed long since.
SAFETY_AGE_SECONDS = float(os.environ.get("GLM52_FIXTURE_SAFETY_AGE_S", 2 * 3600))

# Where glm52_teacher_capture.py seals its capsules.  Live-campaign path: reported, never read.
TEACHER_CAPSULE_DIR = Path(os.environ.get(
    "GLM52_SUPPORT_ROOT",
    "/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity",
)) / "source_fetch" / "teacher" / "capsules"

_EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
_SHARED = re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.(gate|up|down)_proj\.weight$")
_ROUTER = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.weight$")
# first_k_dense_replace=3: layers 0-2 are dense MLP, not MoE
_DENSE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$")
_ATTENTION = re.compile(r"^model\.layers\.(\d+)\.self_attn\.")

PROJECTIONS = ("gate", "up", "down")
# num_experts_per_tok: one token routes to this many experts, so this is what a complete-layer
# MoE benchmark has to be able to execute for real.
EXPERTS_PER_TOKEN = 8


class FixtureError(Exception):
    """A real fixture cannot be produced safely, or would misdescribe itself."""


# ---------------------------------------------------------------------------- eligibility

def shard_paths(root: Path = ARTIFACT_DIR) -> list[Path]:
    """Every candidate shard.  ``.tmp`` is excluded here, not filtered downstream."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.gravity") if not p.name.endswith(".tmp"))


def shard_age(path: Path, *, now: float | None = None) -> float:
    return (time.time() if now is None else now) - path.stat().st_mtime


def is_safe(path: Path, *, min_age: float = SAFETY_AGE_SECONDS,
            now: float | None = None) -> bool:
    """Old enough that the packer has certainly finished with it."""
    return not path.name.endswith(".tmp") and shard_age(path, now=now) >= min_age


def safe_shards(root: Path = ARTIFACT_DIR, *, min_age: float = SAFETY_AGE_SECONDS,
                now: float | None = None) -> list[Path]:
    return [p for p in shard_paths(root) if is_safe(p, min_age=min_age, now=now)]


_VERDICTS: dict[tuple[str, int, float], dict[str, Any]] = {}


def verified(path: Path, *, min_age: float = SAFETY_AGE_SECONDS) -> dict[str, Any]:
    """Full integrity verdict for one shard, computed once per (path, size, mtime).

    Refuses an in-flight shard before reading a byte of its body: the age gate is the
    cheaper check and the one that protects the live campaign, so it runs first.
    """
    if not is_safe(path, min_age=min_age):
        raise FixtureError(
            f"{path.name}: mtime is {shard_age(path):.0f}s old, under the {min_age:.0f}s "
            "safety age -- the campaign may still be writing it")
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime)
    verdict = _VERDICTS.get(key)
    if verdict is None:
        verdict = gravity_format.verify(path)
        _VERDICTS[key] = verdict
    if not verdict["ok"]:
        raise FixtureError(f"{path.name}: failed integrity verify: {verdict}")
    return verdict


# ---------------------------------------------------------------------------- survey

def classify(name: str) -> tuple[str, int | None, int | None, str | None]:
    """(kind, layer, expert, projection) from a tensor name."""
    match = _EXPERT.match(name)
    if match:
        return "routed_expert", int(match.group(1)), int(match.group(2)), match.group(3)
    match = _SHARED.match(name)
    if match:
        return "shared_expert", int(match.group(1)), None, match.group(2)
    match = _ROUTER.match(name)
    if match:
        return "router", int(match.group(1)), None, None
    match = _ATTENTION.match(name)
    if match:
        return "attention", int(match.group(1)), None, None
    match = _DENSE.match(name)
    if match:
        return "dense_mlp", int(match.group(1)), None, match.group(2)
    return "other", None, None, None


def survey(root: Path = ARTIFACT_DIR, *, min_age: float = SAFETY_AGE_SECONDS,
           now: float | None = None) -> dict[str, Any]:
    """Enumerate every shard: size, age, safety, rung, rate, and tensor inventory.

    Headers only.  Reading 90 headers is milliseconds; verifying 90 bodies is ~26 GB, so
    verification is deferred to the shards a fixture actually loads from.
    """
    now = time.time() if now is None else now
    shards: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    for path in shard_paths(root):
        stat = path.stat()
        row: dict[str, Any] = {
            "shard": path.name, "bytes": stat.st_size,
            "mtime": stat.st_mtime, "age_s": now - stat.st_mtime,
            "safe": is_safe(path, min_age=min_age, now=now),
        }
        if row["safe"]:
            header = gravity_format.read_header(path)
            inventory: dict[str, int] = {}
            for tensor in header["tensors"]:
                kind, _, _, _ = classify(tensor["name"])
                inventory[kind] = inventory.get(kind, 0) + 1
                totals[kind] = totals.get(kind, 0) + 1
            row.update({
                "production_rung": header["compression"].get("production_rung"),
                "packed_bpw": header["compression"].get("packed_bpw"),
                "tensor_count": header["integrity"]["tensor_count"],
                "inventory": inventory,
            })
        shards.append(row)

    safe = [s for s in shards if s["safe"]]
    return {
        "root": str(root), "safety_age_s": min_age,
        "shards_total": len(shards), "shards_safe": len(safe),
        "shards_in_flight": len(shards) - len(safe),
        "bytes_total": sum(s["bytes"] for s in shards),
        "bytes_safe": sum(s["bytes"] for s in safe),
        "tensors_safe": sum(s.get("tensor_count", 0) for s in safe),
        "inventory_totals": totals,
        "rungs": sorted({s.get("production_rung") for s in safe if s.get("production_rung")}),
        "shards": shards,
    }


# ---------------------------------------------------------------------------- layer index

def layer_index(root: Path = ARTIFACT_DIR, *, min_age: float = SAFETY_AGE_SECONDS,
                now: float | None = None) -> dict[int, dict[str, Any]]:
    """Per layer: which experts are complete (gate+up+down) and in which shard.

    The router is reported as present/absent because ``pack_shard`` carries control tensors
    at source precision OUTSIDE the .gravity payload set.  A complete-layer benchmark that
    cannot find ``mlp.gate`` is running a fixed expert list, not a routed one, and it has to
    say so.
    """
    layers: dict[int, dict[str, Any]] = {}

    def slot(layer: int) -> dict[str, Any]:
        return layers.setdefault(layer, {
            "layer": layer, "experts": {}, "shared_expert": {},
            "router_shard": None, "attention": {}, "shards": set(),
        })

    for path in safe_shards(root, min_age=min_age, now=now):
        header = gravity_format.read_header(path)
        for tensor in header["tensors"]:
            kind, layer, expert, projection = classify(tensor["name"])
            if layer is None:
                continue
            entry = slot(layer)
            entry["shards"].add(path.name)
            if kind == "routed_expert":
                entry["experts"].setdefault(expert, {})[projection] = path.name
            elif kind == "shared_expert":
                entry["shared_expert"][projection] = path.name
            elif kind == "router":
                entry["router_shard"] = path.name
            elif kind == "attention":
                entry["attention"][tensor["name"]] = path.name

    for entry in layers.values():
        complete = sorted(e for e, p in entry["experts"].items()
                          if all(k in p for k in PROJECTIONS))
        entry["complete_experts"] = complete
        entry["complete_expert_count"] = len(complete)
        entry["partial_experts"] = sorted(set(entry["experts"]) - set(complete))
        entry["shared_expert_complete"] = all(k in entry["shared_expert"] for k in PROJECTIONS)
        entry["router_present"] = entry["router_shard"] is not None
        entry["all_256_experts"] = len(complete) == 256
        # what a complete-layer MoE executor can honestly claim from this layer
        entry["moe_layer_executable"] = (len(complete) >= EXPERTS_PER_TOKEN
                                         and entry["shared_expert_complete"])
        entry["routing_executable"] = entry["moe_layer_executable"] and entry["router_present"]
        entry["shards"] = sorted(entry["shards"])
        entry["experts"] = {str(k): v for k, v in sorted(entry["experts"].items())}
    return dict(sorted(layers.items()))


def executable_layers(index: dict[int, dict[str, Any]] | None = None,
                      **kwargs: Any) -> list[int]:
    """Layers where 8 complete experts plus a complete shared expert are all present."""
    index = layer_index(**kwargs) if index is None else index
    return [layer for layer, entry in index.items() if entry["moe_layer_executable"]]


# ---------------------------------------------------------------------------- loaders

_TENSORS = PressureAwareCache("gravity_real_fixtures", disk_path=str(HERE))


def descriptor_of(path: Path, name: str) -> dict[str, Any]:
    entry = next((t for t in gravity_format.read_header(path)["tensors"]
                  if t["name"] == name), None)
    if entry is None:
        raise FixtureError(f"{path.name}: no tensor named {name!r}")
    return entry


def cache_key(path: Path, descriptor: dict[str, Any]) -> str:
    """Stable across processes and immune to id() reuse.

    ``gravity_metal.py:231`` keys its decoded-index cache on ``id(codes)``, which CPython
    recycles after a GC, so two different tensors can collide and silently serve each
    other's indices.  The content hash cannot: it is what the shard itself recorded.
    """
    return f"{path.name}::{descriptor['name']}::{descriptor['sha256']}"


def load_codes(path: Path, name: str, *, min_age: float = SAFETY_AGE_SECONDS,
               use_cache: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decoded ``pq_codes`` plus the header descriptor for one real tensor."""
    verified(path, min_age=min_age)
    descriptor = descriptor_of(path, name)
    key = cache_key(path, descriptor)
    if use_cache:
        hit = _TENSORS.get(key)
        if hit is not None:
            return hit, descriptor
    codes = glm52_pack.deserialize(gravity_format.read_tensor(path, name))
    if use_cache:
        _TENSORS.put(key, codes)
    return codes, descriptor


# ---------------------------------------------------------------------------- fixtures

REAL = "REAL"
SYNTHETIC = "SYNTHETIC"


@dataclass(frozen=True)
class Fixture:
    """One real packed tensor with full provenance and an inescapable activation label."""

    shard: str
    shard_path: str
    tensor: str
    sha256: str
    descriptor: dict[str, Any]
    activation_source: str
    activation_provenance: str
    cache_key: str
    layer: int | None = None
    expert: int | None = None
    projection: str | None = None
    _codes: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.activation_source not in (REAL, SYNTHETIC):
            raise FixtureError(
                f"activation_source must be {REAL!r} or {SYNTHETIC!r}, "
                f"got {self.activation_source!r}")

    @property
    def codes(self) -> dict[str, Any]:
        if self._codes is None:  # pragma: no cover - fixtures are built with codes
            raise FixtureError(f"{self.tensor}: codes were not loaded")
        return self._codes

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.descriptor["shape"])  # type: ignore[return-value]

    def activation(self, *, seed: int = 0) -> np.ndarray:
        """The input vector for a matvec against this tensor, of the labelled source."""
        if self.activation_source == REAL:  # pragma: no cover - no real source exists yet
            raise FixtureError("REAL activations are not wired; see teacher_activation_status()")
        return synthetic_activation(self.shape[1], seed=seed)

    def as_json(self) -> dict[str, Any]:
        return {
            "shard": self.shard, "shard_path": self.shard_path, "tensor": self.tensor,
            "sha256": self.sha256, "cache_key": self.cache_key,
            "layer": self.layer, "expert": self.expert, "projection": self.projection,
            "shape": list(self.shape), "rung": self.descriptor.get("rung"),
            "bpw": self.descriptor.get("bpw"), "elements": self.descriptor.get("elements"),
            "activation_source": self.activation_source,
            "activation_provenance": self.activation_provenance,
        }


def synthetic_activation(cols: int, *, seed: int = 0) -> np.ndarray:
    """A stand-in activation.  NOT a real hidden state and not evidence of one.

    Unit-variance gaussian: it exercises the arithmetic but has none of a real GLM-5.2
    hidden state's outlier structure, so numerical-accuracy claims made against it are
    bounded by that.  Callers never see this unlabelled -- it reaches a benchmark only
    through :meth:`Fixture.activation`, and the fixture carries ``SYNTHETIC``.
    """
    return np.random.default_rng(seed).standard_normal(cols).astype(np.float32)


def teacher_activation_status() -> dict[str, Any]:
    """Whether real activations are reachable.  They are not, and this says why."""
    live = TEACHER_CAPSULE_DIR.exists()
    return {
        "source": "glm52_teacher_capture.py capsules",
        "path": str(TEACHER_CAPSULE_DIR),
        "directory_exists": live,
        "status": "UNAVAILABLE",
        "reason": (
            "capsule directory is inside the live campaign's support root; reading it "
            "could race the running teacher capture, so it is reported, not opened"
            if live else "no capsule directory on this machine"),
        "consequence": "every fixture below carries activation_source=SYNTHETIC",
    }


def _fixture(path: Path, name: str, *, min_age: float = SAFETY_AGE_SECONDS) -> Fixture:
    codes, descriptor = load_codes(path, name, min_age=min_age)
    _, layer, expert, projection = classify(name)
    status = teacher_activation_status()
    return Fixture(
        shard=path.name, shard_path=str(path), tensor=name,
        sha256=descriptor["sha256"], descriptor=descriptor,
        activation_source=REAL if status["status"] == "AVAILABLE" else SYNTHETIC,
        activation_provenance=f"{status['source']}: {status['status']}",
        cache_key=cache_key(path, descriptor),
        layer=layer, expert=expert, projection=projection, _codes=codes,
    )


def fixture_set(root: Path = ARTIFACT_DIR, *, min_age: float = SAFETY_AGE_SECONDS,
                layer: int | None = None, experts: int = EXPERTS_PER_TOKEN,
                index: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The curated fixtures the GPU phases ask for by name.

    ``one_expert`` is a single expert's three projections (the matvec triple), ``expert_set``
    is ``experts`` complete experts from ONE layer (the MoE-layer executor's working set),
    ``attention`` is one real attention tensor.  All from safe, verified shards.
    """
    index = layer_index(root, min_age=min_age) if index is None else index
    candidates = [layer] if layer is not None else executable_layers(index)
    if not candidates:
        raise FixtureError("no safe layer carries 8 complete experts plus a shared expert")
    chosen = candidates[0]
    entry = index[chosen]
    if len(entry["complete_experts"]) < experts:
        raise FixtureError(
            f"layer {chosen} has {len(entry['complete_experts'])} complete experts, need {experts}")

    picked = entry["complete_experts"][:experts]
    fixtures: dict[str, Any] = {"layer": chosen}

    def name_of(expert: int, projection: str) -> str:
        return f"model.layers.{chosen}.mlp.experts.{expert}.{projection}_proj.weight"

    first = picked[0]
    fixtures["one_expert"] = {
        projection: _fixture(root / entry["experts"][str(first)][projection],
                             name_of(first, projection), min_age=min_age)
        for projection in PROJECTIONS
    }
    fixtures["expert_set"] = [
        {projection: _fixture(root / entry["experts"][str(expert)][projection],
                              name_of(expert, projection), min_age=min_age)
         for projection in PROJECTIONS}
        for expert in picked
    ]
    fixtures["shared_expert"] = {
        projection: _fixture(
            root / entry["shared_expert"][projection],
            f"model.layers.{chosen}.mlp.shared_experts.{projection}_proj.weight",
            min_age=min_age)
        for projection in PROJECTIONS
    } if entry["shared_expert_complete"] else None
    attention_name = next(iter(entry["attention"]), None)
    fixtures["attention"] = (_fixture(root / entry["attention"][attention_name],
                                      attention_name, min_age=min_age)
                             if attention_name else None)
    fixtures["router_present"] = entry["router_present"]
    fixtures["activation"] = teacher_activation_status()
    return fixtures


def manifest(fixtures: dict[str, Any]) -> dict[str, Any]:
    """JSON view of a fixture set: provenance only, no arrays."""
    def dump(value: Any) -> Any:
        if isinstance(value, Fixture):
            return value.as_json()
        if isinstance(value, dict):
            return {k: dump(v) for k, v in value.items()}
        if isinstance(value, list):
            return [dump(v) for v in value]
        return value
    return dump(fixtures)


# ---------------------------------------------------------------------------- index stats

def index_distribution(codes: dict[str, Any], *, hot_mass: float = 0.5) -> dict[str, Any]:
    """Codeword usage histogram for one packed tensor.

    Mandate 6 wants this because a lookup-linear gather kernel's cache behaviour is a
    function of index skew, not of index count: a uniform stream touches the whole codebook
    on every threadgroup, a skewed one keeps a hot subset resident.  Uniform is the
    pessimistic case for a cached gather and the optimistic case for a bandwidth model.
    """
    indices = np.asarray(codes["indices"]).ravel()
    cardinality = int(codes["codebooks"][0].shape[0])
    counts = np.bincount(indices, minlength=cardinality).astype(np.int64)
    total = int(counts.sum())
    probability = counts / max(1, total)
    nonzero = probability[probability > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    uniform = float(np.log2(cardinality))
    order = np.argsort(counts)[::-1]
    cumulative = np.cumsum(counts[order]) / max(1, total)
    hot = int(np.searchsorted(cumulative, hot_mass) + 1)
    ratio = entropy / uniform if uniform else 1.0
    return {
        "codewords": cardinality, "indices": total,
        "entropy_bits": entropy, "uniform_entropy_bits": uniform,
        "entropy_ratio": ratio,
        "unused_codewords": int((counts == 0).sum()),
        "max_count": int(counts.max()), "min_count": int(counts.min()),
        "mean_count": total / max(1, cardinality),
        "top1_share": float(counts.max() / max(1, total)),
        f"codewords_covering_{hot_mass:g}_mass": hot,
        "hot_fraction": hot / cardinality,
        # A perfectly uniform stream would still wobble by sampling noise; 0.995 of the
        # uniform entropy is well inside that, so anything below it is real structure.
        "verdict": "NEAR_UNIFORM" if ratio >= 0.995 else "SKEWED",
        "histogram": counts.tolist(),
    }


# ---------------------------------------------------------------------------- report

def build_report(root: Path = ARTIFACT_DIR, *, min_age: float = SAFETY_AGE_SECONDS,
                 distribution_samples: int = 3) -> dict[str, Any]:
    """Survey + layer index + fixture manifest + measured index distributions."""
    survey_rows = survey(root, min_age=min_age)
    index = layer_index(root, min_age=min_age)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu_only": True,
        "survey": survey_rows,
        "activation": teacher_activation_status(),
        "layers": {
            str(layer): {k: v for k, v in entry.items() if k != "experts"}
            for layer, entry in index.items()
        },
        "executable_layers": executable_layers(index),
        "all_256_expert_layers": [l for l, e in index.items() if e["all_256_experts"]],
        "router_present_layers": [l for l, e in index.items() if e["router_present"]],
    }
    if not survey_rows["shards_safe"]:
        report["fixtures"] = None
        report["index_distribution"] = []
        return report

    fixtures = fixture_set(root, min_age=min_age, index=index)
    report["fixtures"] = manifest(fixtures)

    sampled = [fixtures["one_expert"][p] for p in PROJECTIONS][:distribution_samples]
    if fixtures["attention"] is not None and len(sampled) < distribution_samples + 1:
        sampled.append(fixtures["attention"])
    report["index_distribution"] = [
        {**fixture.as_json(),
         **{k: v for k, v in index_distribution(fixture.codes).items() if k != "histogram"}}
        for fixture in sampled
    ]
    return report


REPORT_PATH = HERE.parents[1] / "reports" / "condense" / "breakthrough" / \
    "GLM52_REAL_ARTIFACT_SURVEY.json"


def selftest() -> int:
    """Exercise the safety gate and the labelling on synthetic shards -- no live reads."""
    import tempfile

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        import gravity_forge as forge
        payloads = []
        for expert in range(2):
            for projection in PROJECTIONS:
                weights = rng.standard_normal((64, 128)).astype(np.float32)
                artifact = forge.pack_product_quant(weights, dim=8, subspaces=1, k=128,
                                                    seed=0, iters=2)
                payloads.append((
                    {"name": f"model.layers.7.mlp.experts.{expert}.{projection}_proj.weight",
                     "category": "routed_expert", "layer": 7, "expert": expert,
                     "shape": [64, 128], "elements": weights.size, "rung": "R0",
                     "bpw": artifact.whole_artifact_bpw},
                    glm52_pack.serialize(artifact)))
        path = root / "model-00001-of-00002.gravity"
        gravity_format.write_shard(
            path, payloads, model={"repo": "zai-org/GLM-5.2", "revision": "x"},
            compression={"codec": "gravity-pq", "production_rung": "R0",
                         "packed_bpw": (sum(len(b) for _, b in payloads) * 8
                                        / sum(d["elements"] for d, _ in payloads))})
        os.utime(path, (0, time.time() - 4 * 3600))

        assert is_safe(path)
        rows = survey(root)
        assert rows["shards_safe"] == 1, rows
        assert rows["inventory_totals"]["routed_expert"] == 6, rows

        # a freshly-touched shard is refused rather than read
        fresh = root / "model-00002-of-00002.gravity"
        fresh.write_bytes(path.read_bytes())
        assert not is_safe(fresh)
        try:
            verified(fresh)
        except FixtureError:
            pass
        else:  # pragma: no cover
            raise AssertionError("an in-flight shard was accepted")

        name = "model.layers.7.mlp.experts.0.gate_proj.weight"
        codes, descriptor = load_codes(path, name)
        assert cache_key(path, descriptor) == f"{path.name}::{name}::{descriptor['sha256']}"
        again, _ = load_codes(path, name)
        assert again is codes, "cache key is not stable"

        stats = index_distribution(codes)
        assert stats["codewords"] == 128 and stats["indices"] == codes["indices"].size

        fixture = _fixture(path, name)
        assert fixture.activation_source == SYNTHETIC
        assert fixture.as_json()["activation_source"] == SYNTHETIC
        assert fixture.activation().shape == (128,)

    print(json.dumps({"selftest": "PASS", "schema": SCHEMA,
                      "safety_age_s": SAFETY_AGE_SECONDS,
                      "in_flight_refused": True, "cache_key_content_addressed": True,
                      "activation_labelled": SYNTHETIC}, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(selftest())
    if len(sys.argv) > 1 and sys.argv[1] == "survey":
        print(json.dumps({k: v for k, v in survey().items() if k != "shards"}, indent=2))
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = build_report()
        REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"wrote": str(REPORT_PATH),
                          "shards_safe": payload["survey"]["shards_safe"],
                          "executable_layers": len(payload["executable_layers"])}, indent=2))
        raise SystemExit(0)
    sys.stderr.write("usage: gravity_real_fixtures.py [selftest|survey|report]\n")
    raise SystemExit(2)
