"""Metadata-only specimen open: a header parse must not load weight bytes.

Call sites, not imports: these tests invoke read_header, read_tensor,
read_specimen_headers, iter_shards, and (on a live lake) the production
wrapper device_profiles.metadata_open. A module import of specimen_open
is not a call of the gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.odyssey import device_profiles as dp
from tools.odyssey import specimen_open as so

WEIGHT_MARKER = b"WEIGHTS_MUST_NOT_BE_READ"


def _tiny(path: Path, *, payload: bytes = WEIGHT_MARKER + b"\x00" * 64) -> Path:
    """One shard: 4-byte first tensor, remainder is an obvious body marker."""
    header = {
        "tiny.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "body.weight": {
            "dtype": "U8",
            "shape": [len(payload) - 4],
            "data_offsets": [4, len(payload)],
        },
    }
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + payload)
    return path


def _specimen(root: Path, *, n_shards: int = 1) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for i in range(n_shards):
        name = f"model-{i+1:05d}-of-{n_shards:05d}.safetensors" if n_shards > 1 else "model.safetensors"
        _tiny(root / name, payload=WEIGHT_MARKER + bytes([i]) * 32)
        weight_map[f"t{i}.weight"] = name
        weight_map[f"t{i}.body"] = name
    if n_shards > 1:
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (root / "config.json").write_text(json.dumps({"model_type": "toy"}))
    return root


class _Spy:
    def __init__(self, fh):
        self._fh = fh
        self.buf = bytearray()
        self.reads: list[int] = []

    def read(self, n=-1):
        data = self._fh.read(n)
        self.buf.extend(data)
        self.reads.append(len(data))
        return data

    def seek(self, off, whence=0):
        return self._fh.seek(off, whence)

    def tell(self):
        return self._fh.tell()

    def fileno(self):
        return self._fh.fileno()

    def close(self):
        return self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_read_header_is_the_gate_symbol():
    """A revert that leaves the helper standing but stops calling it fails."""
    import inspect
    assert "read_header" in so.read_specimen_headers.__code__.co_names
    src = inspect.getsource(so.read_header)
    assert "WeightBytesRefused" in src
    assert "fh.cap = 8 + hl" in src
    assert so.read_header.__name__ == "read_header"


def test_read_header_does_not_consume_weight_bytes(tmp_path, monkeypatch):
    path = _tiny(tmp_path / "m.safetensors")
    spies: list[_Spy] = []
    real = so._open_binary

    def spy(p, *, nocache=False):
        s = _Spy(real(p, nocache=nocache))
        spies.append(s)
        return s

    monkeypatch.setattr(so, "_open_binary", spy)
    view = so.read_header(path, use_cache=False)
    assert spies, "read_header must open the shard (call _open_binary), not skip it"
    buf = bytes(spies[0].buf)
    assert WEIGHT_MARKER not in buf
    assert view["bytes_read"] == view["header_bytes"]
    assert view["bytes_read"] < view["file_bytes"]
    assert view["touched_weight_bytes"] is False
    assert view["n_tensors"] == 2
    assert view["tensors"]["tiny.weight"]["dtype"] == "F32"
    assert view["tensors"]["tiny.weight"]["shape"] == [1]


def test_a_whole_file_read_would_see_the_marker(tmp_path):
    """Negative control: the marker is in the file. The gate is what avoids it."""
    path = _tiny(tmp_path / "m.safetensors")
    raw = path.read_bytes()
    assert WEIGHT_MARKER in raw
    view = so.read_header(path, use_cache=False)
    assert view["bytes_read"] < len(raw)
    assert view["bytes_read"] == view["header_bytes"]


def test_capped_reader_refuses_a_body_read(tmp_path):
    path = _tiny(tmp_path / "m.safetensors")
    st = path.stat()
    with so._Counted(so._open_binary(path), cap=8) as fh:
        prefix = fh.read(8)
        assert len(prefix) == 8
        with pytest.raises(so.WeightBytesRefused, match="metadata-only cap"):
            fh.read(1)
    assert st.st_size > 8


def test_header_cache_hit_reads_zero_lake_bytes(tmp_path, monkeypatch):
    path = _tiny(tmp_path / "m.safetensors")
    cache = tmp_path / "cache"
    first = so.read_header(path, use_cache=True, cache_dir=cache)
    assert first["from_cache"] is False
    assert first["bytes_read"] == first["header_bytes"]

    opens = {"n": 0}
    real = so._open_binary

    def spy(p, *, nocache=False):
        opens["n"] += 1
        return real(p, nocache=nocache)

    monkeypatch.setattr(so, "_open_binary", spy)
    hit = so.read_header(path, use_cache=True, cache_dir=cache)
    assert hit["from_cache"] is True
    assert hit["bytes_read"] == 0
    assert hit["touched_weight_bytes"] is False
    assert hit["tensors"] == first["tensors"]
    assert opens["n"] == 0, "a cache hit must not open the specimen shard"


def test_read_tensor_range_reads_only_the_requested_body(tmp_path, monkeypatch):
    path = _tiny(tmp_path / "m.safetensors")
    header = so.read_header(path, use_cache=False)
    spies: list[_Spy] = []
    real = so._open_binary

    def spy(p, *, nocache=False):
        s = _Spy(real(p, nocache=nocache))
        spies.append(s)
        return s

    monkeypatch.setattr(so, "_open_binary", spy)
    row = so.read_tensor(path, "tiny.weight", header=header, use_mmap=False)
    assert row["bytes"] == 4
    assert row["payload_bytes"] == 4
    assert row["touched_other_tensors"] is False
    assert row["name"] == "tiny.weight"
    # The spy sees a seek + 4-byte read, not the WEIGHTS marker (that sits at offset 4
    # of the body, file offset header+4).
    assert WEIGHT_MARKER not in bytes(spies[0].buf)


def test_read_specimen_headers_on_sharded_layout_sums_headers_only(tmp_path):
    root = _specimen(tmp_path / "toy", n_shards=2)
    meta = so.read_specimen_headers(root, use_cache=False)
    assert meta["n_shards"] == 2
    assert meta["touched_weight_bytes"] is False
    assert meta["bytes_read"] == meta["header_bytes"]
    assert meta["bytes_read"] < meta["file_bytes"]
    assert so.iter_shards(root)[0].name.startswith("model-00001")


def test_metadata_open_calls_read_header(tmp_path, monkeypatch):
    """device_profiles.metadata_open is the production call site of the gate."""
    assert "read_header" in dp.metadata_open.__code__.co_names
    called = {"n": 0}

    def fake(path, **kwargs):
        called["n"] += 1
        return {
            "bytes_read": 16,
            "header_bytes": 16,
            "touched_weight_bytes": False,
            "n_tensors": 1,
            "tensors": {"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        }

    monkeypatch.setattr(so, "read_header", fake)
    path = tmp_path / "unused.safetensors"
    view = dp.metadata_open(path)
    assert called["n"] == 1
    assert view["bytes_read"] == 16

    def fat(path, **kwargs):
        return {"bytes_read": 99, "header_bytes": 16, "touched_weight_bytes": False}

    monkeypatch.setattr(so, "read_header", fat)
    with pytest.raises(so.WeightBytesRefused):
        dp.metadata_open(path)


def test_first_tensor_open_calls_read_tensor(monkeypatch):
    assert "read_tensor" in dp.first_tensor_open.__code__.co_names
    called = {"n": 0}

    def fake(path, name=None, **kwargs):
        called["n"] += 1
        return {"name": name or "tiny.weight", "payload": b"xxxx", "bytes": 4}

    monkeypatch.setattr(so, "read_tensor", fake)
    row = dp.first_tensor_open("unused", "tiny.weight")
    assert called["n"] == 1
    assert "payload" not in row


def test_header_cache_is_never_under_specimens():
    d = so.default_cache_dir()
    assert "specimens" not in Path(d).parts
    assert not str(d).startswith(str(so.TIER2))


RECEIPT = Path(__file__).resolve().parents[2] / "receipts/future/FAST_SPECIMEN_OPEN.json"
LIVE_SMALL = so.TIER2 / "Qwen--Qwen3-0.6B@c1899de289a0" / "model.safetensors"


@pytest.mark.skipif(not LIVE_SMALL.is_file(), reason="sealed Qwen3-0.6B not mounted")
def test_live_qwen_metadata_reads_no_weight_bytes():
    """Real sealed shard, still the default (not slow) selection: header is 35 KB."""
    view = dp.metadata_open(LIVE_SMALL, use_cache=False)
    assert view["bytes_read"] == view["header_bytes"]
    assert view["bytes_read"] < 1024 * 1024
    assert view["file_bytes"] > 1_000_000_000
    assert view["touched_weight_bytes"] is False
    assert "lm_head.weight" in view["tensors"]
    assert view["tensors"]["lm_head.weight"]["dtype"] == "BF16"


@pytest.mark.skipif(not RECEIPT.is_file(), reason="FAST_SPECIMEN_OPEN receipt not written")
def test_receipt_metadata_path_read_no_weight_bytes():
    d = json.loads(RECEIPT.read_text())
    assert d["schema"] == "hawking.odyssey.fast_specimen_open.v1"
    ids = {s["id"] for s in d["per_specimen"]}
    assert "Qwen--Qwen3-0.6B@c1899de289a0" in ids
    assert "tiiuae--Falcon-H1-7B-Instruct@41e72f27effb" in ids
    for s in d["per_specimen"]:
        m = s["metadata_only"]
        assert m["touched_weight_bytes"] is False
        assert m["bytes_read_cold"] == m["header_bytes"]
        assert m["bytes_read_cold"] < s["file_bytes"]
        assert m["bytes_read_cache_hit"] == 0
        assert m["evidence_tier"] == "HARDWARE_MEASURED"
        assert s["full_shards"]["evidence_tier"] == "HARDWARE_MEASURED"
        ba = s["before_after"]["metadata_common_path"]
        assert ba["evidence_tier"] == "HARDWARE_MEASURED"
        assert ba["speedup_cold_vs_full"] > 1000
    seq = d["sequential_rate"]
    assert seq["evidence_tier"] == "HARDWARE_MEASURED"
    # USB bus, not a spec (5 Gbps ≈ 0.625 GB/s) and not an internal-SSD roof.
    assert 0.05 < seq["median_cold_gb_s"] < 0.5
    assert "not a USB spec number" in seq["note"]
    assert "FUNCTIONAL_SIM" not in RECEIPT.read_text()
    assert d["tier1_already_staged"]["copied_the_lake"] is False
    assert d["tier1_already_staged"]["evidence_tier"] == "HARDWARE_MEASURED"
    assert d["tier1_already_staged"]["cold_speedup_vs_usb"] > 10
    assert any("measure --slug Qwen--Qwen3-0.6B" in c for c in d["commands"])


@pytest.mark.skipif(not RECEIPT.is_file(), reason="FAST_SPECIMEN_OPEN receipt not written")
def test_overlay_loads_the_real_receipt():
    out = dp.specimen_open_economics()
    assert out["status"] == "OK"
    assert out["called"] == "specimen_open.load_receipt"
    assert out["evidence_tier"] == "COST_MODEL"
    assert out["sequential_evidence_tier"] == "HARDWARE_MEASURED"
    assert out["specimens"][0]["metadata_touched_weight_bytes"] is False
