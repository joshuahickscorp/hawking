"""hcli.specimens: the disk-derived registry, never a hand-maintained list.

Built against a scratch lake under tmp_path for every test (never the real
ModelLake volume, so these pass whether or not /Volumes/corpdrive is
attached). One test at the bottom runs against the REAL 47-specimen lake,
skipped when it is not mounted, to prove the enumeration stays cheap on the
real thing -- stat and manifest reads, never a walk of 3.4 TiB of weights.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hcli.specimens import get, registry

REAL_SPECIMENS_DIR = Path("/Volumes/corpdrive/hawking-modellake/specimens")


def _lab(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """A scratch lake + manifest dir, wired in via the same env vars the
    module reads in production (HCLI_MODEL_LAKE_ROOT / _MANIFEST_DIR)."""
    lake = tmp_path / "lake"
    (lake / "specimens").mkdir(parents=True)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    monkeypatch.setenv("HCLI_MODEL_LAKE_ROOT", str(lake))
    monkeypatch.setenv("HCLI_SPECIMEN_MANIFEST_DIR", str(manifests))
    return lake, manifests


def _add_specimen(lake: Path, manifests: Path, name: str, *, config: dict | None = None,
                   files: dict[str, bytes] | None = None, seal: bool = False,
                   corrupt: bool = False) -> None:
    d = lake / "specimens" / name
    d.mkdir(parents=True)
    if config is not None:
        (d / "config.json").write_text(json.dumps(config))
    files = files or {"model.safetensors": b"x" * 128}
    for fname, blob in files.items():
        (d / fname).write_bytes(blob)
    if seal:
        sizes = {f: len(blob) for f, blob in files.items()}
        if config is not None:
            sizes["config.json"] = (d / "config.json").stat().st_size
        if corrupt:
            # Claim one file is bigger than it really is.
            first = next(iter(sizes))
            sizes[first] += 999
        manifest = {"repo": name.split("@")[0].replace("--", "/"),
                    "resolved_sha": name.split("@")[-1], "sizes": sizes,
                    "files": list(sizes), "expected": sum(sizes.values())}
        (manifests / f"{name}.json").write_text(json.dumps(manifest))


# --- identity, revision, architecture -----------------------------------


def test_identity_revision_and_path(tmp_path, monkeypatch):
    lake, manifests = _lab(tmp_path, monkeypatch)
    _add_specimen(lake, manifests, "Qwen--Qwen2.5-72B-Instruct@495f39366efe",
                  config={"model_type": "qwen2", "hidden_size": 8192, "num_hidden_layers": 80,
                          "architectures": ["Qwen2ForCausalLM"]})
    row = get("Qwen--Qwen2.5-72B-Instruct@495f39366efe")
    assert row is not None
    assert row["repo"] == "Qwen/Qwen2.5-72B-Instruct"
    assert row["revision"] == "495f39366efe"
    assert row["path"].endswith("Qwen--Qwen2.5-72B-Instruct@495f39366efe")


def test_architecture_falls_back_to_text_config_for_vl_models(tmp_path, monkeypatch):
    """Qwen3-VL nests the language-model shape under text_config; a top-level
    read alone reports hidden_size/num_hidden_layers as None even though the
    specimen is a real, fully-shaped model."""
    lake, manifests = _lab(tmp_path, monkeypatch)
    _add_specimen(lake, manifests, "Qwen--Qwen3-VL-8B-Instruct@0c351dd01ed8",
                  config={"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"],
                          "text_config": {"hidden_size": 4096, "num_hidden_layers": 36}})
    row = get("Qwen--Qwen3-VL-8B-Instruct@0c351dd01ed8")
    assert row["architecture"]["model_type"] == "qwen3_vl"
    assert row["architecture"]["hidden_size"] == 4096
    assert row["architecture"]["num_hidden_layers"] == 36


def test_missing_config_reports_unknown_architecture_not_an_error(tmp_path, monkeypatch):
    """Wan2.2, boltz-2 and moshika ship no config.json at all on the real lake."""
    lake, manifests = _lab(tmp_path, monkeypatch)
    _add_specimen(lake, manifests, "boltz-community--boltz-2@6fdef46d763f", config=None)
    row = get("boltz-community--boltz-2@6fdef46d763f")
    assert row is not None
    assert row["architecture"] == {"model_type": None, "hidden_size": None,
                                    "num_hidden_layers": None, "architectures": None}


# --- manifest presence and completeness ----------------------------------


def test_sealed_manifest_verifies_complete(tmp_path, monkeypatch):
    lake, manifests = _lab(tmp_path, monkeypatch)
    _add_specimen(lake, manifests, "ibm-granite--granite-4.0-h-tiny@791e0d3d28c8",
                  config={"model_type": "granitemoehybrid"}, seal=True)
    row = get("ibm-granite--granite-4.0-h-tiny@791e0d3d28c8")
    assert row["manifest_present"] is True
    assert row["verified_complete"] is True
    assert row["size_bytes"] > 0


def test_mismatched_file_size_is_verified_false_not_true(tmp_path, monkeypatch):
    lake, manifests = _lab(tmp_path, monkeypatch)
    _add_specimen(lake, manifests, "moonshotai--Kimi-Linear-48B-A3B-Instruct@e1df551a4471",
                  config={"model_type": "kimi_linear"}, seal=True, corrupt=True)
    row = get("moonshotai--Kimi-Linear-48B-A3B-Instruct@e1df551a4471")
    assert row["manifest_present"] is True
    assert row["verified_complete"] is False
    assert row["verified_complete_mismatches"]


def test_no_manifest_is_unknown_not_declared_incomplete(tmp_path, monkeypatch):
    """The honesty constraint: absence of a manifest must read as 'cannot
    verify', never silently coerced to True or False."""
    lake, manifests = _lab(tmp_path, monkeypatch)
    _add_specimen(lake, manifests, "tiiuae--Falcon-H1-7B-Instruct@41e72f27effb",
                  config={"model_type": "falcon_h1"}, seal=False)
    row = get("tiiuae--Falcon-H1-7B-Instruct@41e72f27effb")
    assert row["manifest_present"] is False
    assert row["verified_complete"] is None
    assert row["size_bytes"] > 0  # falls back to a stat-sum of the directory


# --- never a recommendation ------------------------------------------------


def _keys(obj) -> set[str]:
    """Every dict key anywhere in the structure -- prose is allowed to say
    'this is not a recommendation'; a FIELD offering one would not be."""
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k.lower())
            out |= _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _keys(v)
    return out


def test_registry_never_recommends_loading_anything(tmp_path, monkeypatch):
    lake, manifests = _lab(tmp_path, monkeypatch)
    _add_specimen(lake, manifests, "a--b@c", seal=True)
    doc = registry()
    keys = _keys(doc)
    for banned in ("should_load", "recommended", "recommendation", "load_now", "best_specimen", "rank", "ranked"):
        assert banned not in keys


# --- mid-flight join --------------------------------------------------------


def test_new_specimen_joins_without_restart(tmp_path, monkeypatch):
    """No cache anywhere: a specimen written to disk after the first call is
    present in the very next call, with nothing re-initialized in between."""
    lake, manifests = _lab(tmp_path, monkeypatch)
    _add_specimen(lake, manifests, "first--one@aaaa", seal=True)
    before = registry()
    assert before["n_specimens"] == 1

    _add_specimen(lake, manifests, "second--one@bbbb", seal=True)  # arrives mid-flight
    after = registry()
    assert after["n_specimens"] == 2
    assert {r["id"] for r in after["specimens"]} == {"first--one@aaaa", "second--one@bbbb"}


# --- absent volume is not zero specimens ------------------------------------


def test_unmounted_lake_is_not_reported_as_zero_specimens(tmp_path, monkeypatch):
    monkeypatch.setenv("HCLI_MODEL_LAKE_ROOT", str(tmp_path / "nothing-mounted-here"))
    monkeypatch.setenv("HCLI_SPECIMEN_MANIFEST_DIR", str(tmp_path / "manifests"))
    doc = registry()
    assert doc["mounted"] is False
    assert doc["n_specimens"] is None  # unknown, never coerced to 0
    assert doc["specimens"] == []


def test_get_returns_none_for_a_specimen_not_on_disk(tmp_path, monkeypatch):
    _lab(tmp_path, monkeypatch)
    assert get("nothing--here@0000") is None


# --- cost proof against the real lake ---------------------------------------


@pytest.mark.skipif(not REAL_SPECIMENS_DIR.is_dir(), reason="ModelLake volume not mounted")
def test_real_lake_enumeration_is_cheap(monkeypatch):
    """47 specimens holding a combined ~3.4 TiB must enumerate as a function
    of file COUNT (stat + small config/manifest reads), never of byte count.
    This is a real USB-attached APFS volume, not an SSD -- per-syscall
    latency alone measured ~35s here, so the bound below is generous, not
    tight. What it catches is the regression that matters: reading even a
    meaningful fraction of the weight bytes. The smallest specimen sealed
    today is 13 GB; reading that much content at a conservative 200 MB/s
    alone takes ~65s, and the library holds 3.4 TiB -- so 120s of pure
    directory/stat/manifest traffic is still unambiguously "did not read the
    weights", while comfortably clearing this drive's real seek latency."""
    monkeypatch.delenv("HCLI_MODEL_LAKE_ROOT", raising=False)
    monkeypatch.delenv("HCLI_SPECIMEN_MANIFEST_DIR", raising=False)
    start = time.perf_counter()
    doc = registry()
    elapsed = time.perf_counter() - start
    assert doc["mounted"] is True
    assert doc["n_specimens"] >= 40
    total_bytes = sum(r["size_bytes"] for r in doc["specimens"])
    print(f"enumerated {doc['n_specimens']} specimens ({total_bytes / 2**40:.2f} TiB) in {elapsed:.3f}s")
    assert elapsed < 120.0, f"enumeration took {elapsed:.3f}s -- looks like it started reading weight bytes, not stats"
