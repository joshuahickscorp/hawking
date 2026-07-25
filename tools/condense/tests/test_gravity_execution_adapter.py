from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

CONDENSE = Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_pack  # noqa: E402
import gravity_execution_adapter as gea  # noqa: E402
import gravity_format  # noqa: E402
import gravity_forge as forge  # noqa: E402
from glm52_common import Glm52Error  # noqa: E402


REAL_SHARD = Path(
    "/Users/scammermike/Desktop/GLM52-Gravity-SubBit/model-00002-of-00282.gravity"
)
REAL_TENSOR = "model.layers.10.mlp.experts.0.gate_proj.weight"
# offset of the packed `rotate` flag inside glm52_pack's fixed 64-byte tensor header:
# MAGIC(8) + 4H(8) + 4I(16) + H(2) == 34.
ROTATE_FLAG_OFFSET = 34
BUDGET = 4_000_000


def _pack(rows: int, cols: int, *, dim: int = 8, subspaces: int = 1, k: int = 16, seed: int = 0):
    weights = np.random.default_rng(seed).standard_normal((rows, cols)).astype(np.float32)
    artifact = forge.pack_product_quant(
        weights, dim=dim, subspaces=subspaces, k=k, seed=0, iters=3)
    return weights, glm52_pack.serialize(artifact)


def _descriptor(name: str, blob: bytes, shape, **overrides):
    elements = int(shape[0]) * int(shape[1])
    entry = {
        "name": name, "shape": list(shape), "elements": elements,
        "category": "routed_expert", "codec": "gravity-pq", "rung": "R0",
        "expert": 0, "layer": 10, "bpw": len(blob) * 8 / elements,
    }
    entry.update(overrides)
    return entry


def _shard(tmp_path: Path, payloads, name: str = "model-00001-of-00001.gravity") -> Path:
    path = tmp_path / name
    gravity_format.write_shard(
        path, payloads,
        model={"repo": "zai-org/GLM-5.2", "revision": "b" * 40},
        compression={"codec": "gravity-pq", "packed_bpw": 0.876, "production_rung": "R0"})
    return path


def _one_tensor_shard(tmp_path: Path, **overrides) -> tuple[Path, np.ndarray]:
    weights, blob = _pack(16, 32)
    descriptor = _descriptor("model.layers.0.mlp.experts.0.gate_proj.weight", blob,
                             (16, 32), **overrides)
    return _shard(tmp_path, [(descriptor, blob)]), weights


def _recording_backend() -> dict:
    seen: list[tuple] = []
    released: list[str] = []

    def can_execute(geometry):
        return None

    def matvec(codes, x, key):
        seen.append((dict(codes), np.array(x), key))
        return np.zeros(int(codes["rows"]), dtype=np.float32)

    def release(key):
        released.append(key)

    return {"name": "recording", "protocol": gea.BACKEND_PROTOCOL, "device": "none",
            "can_execute": can_execute, "matvec": matvec, "release": release,
            "seen": seen, "released": released}


# --------------------------------------------------------------------------------------
# Registry admission: the adapter must satisfy the gate it wants to be registered behind.
# --------------------------------------------------------------------------------------
def test_source_passes_the_workers_own_registry_admission_inspector() -> None:
    import glm52_worker as worker

    source = CONDENSE / "gravity_execution_adapter.py"
    worker._inspect_registered_adapter_source(
        source.read_bytes(),
        source_path="tools/condense/gravity_execution_adapter.py",
        adapter_id=gea.ADAPTER_ID,
        entry_points={gea.ADAPTER_ROLE: "execute_tensor"},
    )
    assert gea.ADAPTER_INTERFACE == worker.EXECUTION_ADAPTER_INTERFACE
    assert gea.ADAPTER_ROLE in worker.REQUIRED_EXECUTION_ROLES
    # registering this one adapter is necessary but NOT sufficient: seven roles remain
    assert set(worker.REQUIRED_EXECUTION_ROLES) - {gea.ADAPTER_ROLE}


def test_declared_capabilities_name_every_refusal_the_module_can_raise() -> None:
    declared = set(gea.declare_capabilities()["refusals"])
    implemented = {
        value for name, value in vars(gea).items()
        if name.startswith("REFUSAL_") and isinstance(value, str)
    }
    assert declared == implemented
    assert gea.declare_capabilities()["bindings"]["metal_grammar"] == "UNSELECTED_IN_THIS_PHASE"


# --------------------------------------------------------------------------------------
# Refusals.
# --------------------------------------------------------------------------------------
def test_refuses_a_shard_whose_format_version_is_newer_than_the_reader(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gravity_format, "FORMAT_VERSION", gea.SUPPORTED_FORMAT_VERSION + 1)
    path, _ = _one_tensor_shard(tmp_path)
    monkeypatch.undo()
    with pytest.raises(Glm52Error, match=gea.REFUSAL_FORMAT_VERSION):
        gea.open_adapter(path, gea.cpu_reference_backend(), BUDGET)


def test_refuses_a_file_that_is_not_a_gravity_shard(tmp_path) -> None:
    path = tmp_path / "not.gravity"
    path.write_bytes(b"NOTGRAVITY" + b"\x00" * 64)
    with pytest.raises(Glm52Error, match=gea.REFUSAL_NOT_A_SHARD):
        gea.open_adapter(path, gea.cpu_reference_backend(), BUDGET)

    short = tmp_path / "short.gravity"
    short.write_bytes(b"GRAV")
    with pytest.raises(Glm52Error, match=gea.REFUSAL_NOT_A_SHARD):
        gea.open_adapter(short, gea.cpu_reference_backend(), BUDGET)


def test_refuses_a_rung_that_is_not_production_r0(tmp_path) -> None:
    path, _ = _one_tensor_shard(tmp_path, rung="R2")
    adapter = gea.open_adapter(path, gea.cpu_reference_backend(), BUDGET)
    with pytest.raises(Glm52Error, match=gea.REFUSAL_RUNG):
        gea.load_tensor(adapter, "model.layers.0.mlp.experts.0.gate_proj.weight")


def test_refuses_a_control_tensor_that_carries_no_packed_payload(tmp_path) -> None:
    path, _ = _one_tensor_shard(tmp_path, category="router")
    adapter = gea.open_adapter(path, gea.cpu_reference_backend(), BUDGET)
    with pytest.raises(Glm52Error, match=gea.REFUSAL_CATEGORY):
        gea.load_tensor(adapter, "model.layers.0.mlp.experts.0.gate_proj.weight")


def test_refuses_a_rotated_geometry(tmp_path) -> None:
    _, blob = _pack(16, 32)
    raw = bytearray(blob)
    struct.pack_into("?", raw, ROTATE_FLAG_OFFSET, True)
    blob = bytes(raw)
    assert glm52_pack.deserialize(blob)["rotate"] is True
    path = _shard(tmp_path, [(_descriptor("t", blob, (16, 32)), blob)])
    adapter = gea.open_adapter(path, gea.cpu_reference_backend(), BUDGET)
    with pytest.raises(Glm52Error, match=gea.REFUSAL_ROTATE):
        gea.load_tensor(adapter, "t")


def test_refuses_more_than_one_subspace(tmp_path) -> None:
    _, blob = _pack(16, 32, subspaces=2)
    path = _shard(tmp_path, [(_descriptor("t", blob, (16, 32)), blob)])
    adapter = gea.open_adapter(path, gea.cpu_reference_backend(), BUDGET)
    with pytest.raises(Glm52Error, match=gea.REFUSAL_SUBSPACES):
        gea.load_tensor(adapter, "t")


@pytest.mark.parametrize("lie", [
    {"shape": [32, 16]},
    {"shape": [16, 64]},
])
def test_refuses_when_the_descriptor_geometry_disagrees_with_the_payload(tmp_path, lie) -> None:
    _, blob = _pack(16, 32)
    descriptor = _descriptor("t", blob, (16, 32))
    descriptor.update(lie)
    path = _shard(tmp_path, [(descriptor, blob)])
    adapter = gea.open_adapter(path, gea.cpu_reference_backend(), BUDGET)
    with pytest.raises(Glm52Error, match=gea.REFUSAL_GEOMETRY):
        gea.load_tensor(adapter, "t")


def test_refuses_when_the_descriptor_element_count_disagrees_with_the_payload(tmp_path) -> None:
    _, blob = _pack(16, 32)
    descriptor = _descriptor("t", blob, (16, 32))
    descriptor["elements"] = 999
    descriptor["bpw"] = len(blob) * 8 / 999
    path = _shard(tmp_path, [(descriptor, blob)])
    adapter = gea.open_adapter(path, gea.cpu_reference_backend(), BUDGET)
    with pytest.raises(Glm52Error, match=gea.REFUSAL_GEOMETRY):
        gea.load_tensor(adapter, "t")


def test_refuses_a_rate_claim_the_payload_bytes_do_not_support(tmp_path) -> None:
    _, blob = _pack(16, 32)
    descriptor = _descriptor("t", blob, (16, 32), bpw=0.5)
    path = _shard(tmp_path, [(descriptor, blob)])
    adapter = gea.open_adapter(path, gea.cpu_reference_backend(), BUDGET)
    with pytest.raises(Glm52Error, match=gea.REFUSAL_RATE):
        gea.load_tensor(adapter, "t")


@pytest.mark.parametrize("broken", [
    {"protocol": "hawking.gravity.execution_backend.v0"},
    {"name": ""},
    {"can_execute": None},
    {"matvec": "not-callable"},
    {"release": 7},
])
def test_refuses_a_backend_that_does_not_declare_the_protocol(tmp_path, broken) -> None:
    path, _ = _one_tensor_shard(tmp_path)
    backend = dict(_recording_backend())
    backend.update(broken)
    with pytest.raises(Glm52Error, match=gea.REFUSAL_BACKEND_PROTOCOL):
        gea.open_adapter(path, backend, BUDGET)
    with pytest.raises(Glm52Error, match=gea.REFUSAL_BACKEND_PROTOCOL):
        gea.open_adapter(path, "not-a-mapping", BUDGET)


def test_refuses_when_the_backend_vetoes_the_geometry(tmp_path) -> None:
    path, _ = _one_tensor_shard(tmp_path)
    backend = _recording_backend()
    backend["can_execute"] = lambda geometry: f"k={geometry['k']} exceeds my index width"
    adapter = gea.open_adapter(path, backend, BUDGET)
    x = np.ones(32, dtype=np.float32)
    with pytest.raises(Glm52Error, match=gea.REFUSAL_BACKEND_GEOMETRY):
        gea.execute_tensor(adapter, "model.layers.0.mlp.experts.0.gate_proj.weight", x)
    assert backend["seen"] == []  # the veto happens before the kernel is handed anything


def test_refuses_an_input_whose_length_disagrees_with_the_geometry(tmp_path) -> None:
    path, _ = _one_tensor_shard(tmp_path)
    backend = _recording_backend()
    adapter = gea.open_adapter(path, backend, BUDGET)
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    for bad in (np.ones(33, dtype=np.float32), np.ones((32, 2), dtype=np.float32)):
        with pytest.raises(Glm52Error, match=gea.REFUSAL_INPUT_SHAPE):
            gea.execute_tensor(adapter, name, bad)
    assert backend["seen"] == []


def test_refuses_a_backend_result_of_the_wrong_shape(tmp_path) -> None:
    path, _ = _one_tensor_shard(tmp_path)
    backend = _recording_backend()
    backend["matvec"] = lambda codes, x, key: np.zeros(3, dtype=np.float32)
    adapter = gea.open_adapter(path, backend, BUDGET)
    with pytest.raises(Glm52Error, match=gea.REFUSAL_OUTPUT_SHAPE):
        gea.execute_tensor(adapter, "model.layers.0.mlp.experts.0.gate_proj.weight",
                           np.ones(32, dtype=np.float32))


def test_refuses_a_tensor_that_cannot_fit_the_whole_byte_budget(tmp_path) -> None:
    path, _ = _one_tensor_shard(tmp_path)
    adapter = gea.open_adapter(path, gea.cpu_reference_backend(), 16)
    with pytest.raises(Glm52Error, match=gea.REFUSAL_BUDGET):
        gea.load_tensor(adapter, "model.layers.0.mlp.experts.0.gate_proj.weight")
    assert gea.resource_report(adapter)["resident_bytes"] == 0
    with pytest.raises(Glm52Error, match=gea.REFUSAL_BUDGET):
        gea.open_adapter(path, gea.cpu_reference_backend(), 0)


def test_refuses_an_absent_tensor_by_name(tmp_path) -> None:
    path, _ = _one_tensor_shard(tmp_path)
    adapter = gea.open_adapter(path, gea.cpu_reference_backend(), BUDGET)
    with pytest.raises(gravity_format.GravityFormatError, match="no tensor named"):
        gea.load_tensor(adapter, "model.layers.0.absent.weight")


# --------------------------------------------------------------------------------------
# Injected-backend protocol and resource policy.
# --------------------------------------------------------------------------------------
def test_the_backend_receives_the_decoded_codes_and_the_explicit_cache_key(tmp_path) -> None:
    path, weights = _one_tensor_shard(tmp_path)
    backend = _recording_backend()
    adapter = gea.open_adapter(path, backend, BUDGET)
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    x = np.random.default_rng(1).standard_normal(32).astype(np.float32)
    y = gea.execute_tensor(adapter, name, x)

    assert y.shape == (16,)
    codes, seen_x, key = backend["seen"][0]
    assert (codes["rows"], codes["cols"], codes["S"], codes["rotate"]) == (16, 32, 1, False)
    assert np.array_equal(seen_x, x)
    assert key == gea.cache_key(adapter, name)
    assert key.startswith(str(path)) and name in key and len(key.rsplit("|", 1)[1]) == 64
    assert hex(id(codes)) not in key


def test_the_cpu_reference_backend_matches_a_direct_pq_execute(tmp_path) -> None:
    weights, blob = _pack(16, 32)
    path = _shard(tmp_path, [(_descriptor("t", blob, (16, 32)), blob)])
    adapter = gea.open_adapter(path, gea.cpu_reference_backend(), BUDGET)
    x = np.random.default_rng(2).standard_normal(32).astype(np.float32)
    reference = forge.pq_execute(glm52_pack.load_artifact(blob), x)
    assert np.array_equal(gea.execute_tensor(adapter, "t", x), reference)


def test_the_byte_budget_evicts_least_recently_used_and_releases_the_backend_copy(tmp_path) -> None:
    payloads = []
    for index in range(3):
        _, blob = _pack(16, 32, seed=index)
        payloads.append((_descriptor(f"t{index}", blob, (16, 32)), blob))
    path = _shard(tmp_path, payloads)
    backend = _recording_backend()

    probe = gea.open_adapter(path, backend, BUDGET)
    per_entry = gea.load_tensor(probe, "t0")["bytes"]

    adapter = gea.open_adapter(path, backend, per_entry * 2)
    gea.load_tensor(adapter, "t0")
    gea.load_tensor(adapter, "t1")
    gea.load_tensor(adapter, "t0")          # t0 becomes most-recently-used, t1 is oldest
    assert gea.resource_report(adapter)["evictions"] == 0

    gea.load_tensor(adapter, "t2")
    report = gea.resource_report(adapter)
    assert report["entries"] == 2
    assert report["evictions"] == 1
    assert report["resident_bytes"] == per_entry * 2 <= report["byte_budget"]
    assert backend["released"] == [gea.cache_key(adapter, "t1")]
    assert [key.split("|")[1] for key in report["keys"]] == ["t0", "t2"]
    assert report["eviction_contract"] == gea.EVICTION_CONTRACT

    assert gea.evict_all(adapter) == 2
    assert gea.resource_report(adapter)["resident_bytes"] == 0
    assert gea.resource_report(adapter)["byte_budget"] == per_entry * 2
    assert backend["released"][-2:] == [gea.cache_key(adapter, "t0"),
                                        gea.cache_key(adapter, "t2")]


def test_a_cache_hit_re_reads_nothing_and_keeps_the_same_key(tmp_path) -> None:
    path, _ = _one_tensor_shard(tmp_path)
    adapter = gea.open_adapter(path, gea.cpu_reference_backend(), BUDGET)
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    first = gea.load_tensor(adapter, name)
    second = gea.load_tensor(adapter, name)
    assert first is second
    assert gea.resource_report(adapter)["loads"] == 1


# --------------------------------------------------------------------------------------
# Real sealed artifact, strictly read-only.
# --------------------------------------------------------------------------------------
@pytest.mark.skipif(not REAL_SHARD.exists(), reason=f"{REAL_SHARD} is absent")
def test_real_shard_routed_expert_executes_and_matches_a_direct_pq_execute() -> None:
    adapter = gea.open_adapter(REAL_SHARD, gea.cpu_reference_backend(), 2_000_000_000)
    entry = gea.load_tensor(adapter, REAL_TENSOR)
    geometry = entry["geometry"]
    assert geometry["rung"] == "R0"
    assert (geometry["D"], geometry["k"], geometry["S"]) == (8, 128, 1)
    assert (geometry["rows"], geometry["cols"], geometry["nchunk"]) == (2048, 6144, 768)
    assert geometry["rotate"] is False
    assert geometry["index_bits"] == 7
    assert geometry["category"] == "routed_expert"
    assert abs(geometry["bpw"] - 0.8763427734375) < 1e-12

    x = np.random.default_rng(0).standard_normal(6144).astype(np.float32)
    y = gea.execute_tensor(adapter, REAL_TENSOR, x)
    reference = forge.pq_execute(
        glm52_pack.load_artifact(gravity_format.read_tensor(REAL_SHARD, REAL_TENSOR)), x)
    assert y.shape == (2048,)
    assert np.isfinite(y).all()
    assert np.array_equal(y, reference)
