"""Durable ModelLake index: query without walking the lake.

Call sites, not imports: build / query_specimen / update_specimen / walk_specimen_files
are invoked. Lineage wrappers and modellake.main must CALL those symbols.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from tools.odyssey import modellake as ml
from tools.odyssey import modellake_index as mx
from tools.odyssey import modellake_lineage as lin

CANON = lin.CANONICAL_SPECIMEN


def _tiny_safetensors(path: Path, names: list[str]) -> None:
    header = {n: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for n in names}
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\x00\x00\x00\x00")


def _write_index_json(root: Path, shards: list[str], tensors: list[str]) -> None:
    wm = {t: shards[i % len(shards)] for i, t in enumerate(tensors)}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))


def _add_specimen(
    lake: Path,
    slug: str,
    *,
    model_type: str,
    architectures: list[str],
    n_shards: int = 2,
    extra_bytes: int = 0,
    watch: bool = True,
    lake_manifest: bool = True,
    location: str = "specimens",
    max_pos: int = 4096,
    num_experts: int | None = None,
) -> Path:
    body = lake / location / slug
    body.mkdir(parents=True)
    cfg = {
        "model_type": model_type,
        "architectures": architectures,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": max_pos,
        "vocab_size": 256,
    }
    if num_experts is not None:
        cfg["num_experts"] = num_experts
    (body / "config.json").write_text(json.dumps(cfg))
    tensors = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "lm_head.weight",
    ]
    shards = [f"model-{i:05d}-of-{n_shards:05d}.safetensors" for i in range(1, n_shards + 1)]
    _write_index_json(body, shards, tensors)
    for name in shards:
        _tiny_safetensors(body / name, tensors)
    if extra_bytes:
        (body / "blob.bin").write_bytes(b"\x00" * extra_bytes)
    files = ["config.json", "model.safetensors.index.json", *shards]
    if extra_bytes:
        files.append("blob.bin")
    sizes = {f: (body / f).stat().st_size for f in files}
    expected = sum(sizes.values())
    repo = slug.split("@", 1)[0].replace("--", "/")
    rev = slug.split("@", 1)[1] if "@" in slug else "0" * 12
    if watch:
        (lake / "watch" / f"{slug}.json").write_text(json.dumps({
            "repo": repo,
            "revision": rev,
            "resolved_sha": rev if len(rev) >= 40 else rev,
            "files": files,
            "sizes": sizes,
            "expected": expected,
        }))
    if lake_manifest:
        (lake / "manifests" / f"{slug}.json").write_text(json.dumps({
            "repo": repo,
            "revision": rev,
            "resolved_sha": rev,
            "bytes": expected,
            "n_files": len(files),
            "reacquisition": f"hf download {repo} --revision {rev} --local-dir <dest>",
        }))
    return body


def _lake(tmp: Path) -> Path:
    lake = tmp / "hawking-modellake"
    for d in ("specimens", "manifests", "watch", "logs"):
        (lake / d).mkdir(parents=True)
    (lake / "logs" / "acquisition-state.json").write_text("{}")
    return lake


def _build(lake: Path, **kw):
    return mx.build(
        lake=lake,
        index_dir=lake / "index",
        manifest_dir=lake / "watch",
        **kw,
    )


def test_scan_specimen_calls_lineage_pieces():
    names = mx.scan_specimen.__code__.co_names
    for needed in (
        "walk_specimen_files",
        "load_watch_manifest",
        "role_metadata",
        "architecture_fingerprint",
        "tensor_names_from_specimen",
        "derive_lifecycle",
        "storage_tier_for",
    ):
        assert needed in names, needed


def test_budget_constant_is_called_not_copied():
    assert "TIER2_BUDGET" in mx._tier2_budget.__code__.co_names
    assert mx._tier2_budget() == ml.TIER2_BUDGET == 3_500 * 10**9


def test_build_and_query_on_a_tiny_lake(tmp_path):
    lake = _lake(tmp_path)
    _add_specimen(
        lake, CANON, model_type="qwen3",
        architectures=["Qwen3ForCausalLM"], max_pos=40960,
    )
    cat = _build(lake, force=True)
    assert cat["n_specimens"] == 1
    assert cat["wrote_specimens"] is False
    assert cat["tier2_budget"] == ml.TIER2_BUDGET
    row = mx.query_specimen(CANON, index_dir=lake / "index", lake=lake)
    assert row["slug"] == CANON
    assert row["architecture_family"] == "qwen3"
    assert row["role"]["primary"] in lin.DIVERSITY_ROLES
    assert "dense decoder" in row["role"]["roles"]
    assert row["n_shards"] == 2
    assert row["n_files"] >= 4
    assert row["seal_status"] == "SEALED"
    assert row["loaded_weights"] is False
    assert row["wrote_specimen"] is False
    assert (lake / "index" / "catalog.json").is_file()
    assert (lake / "index" / "by-slug" / f"{CANON}.json").is_file()


def test_query_does_not_walk_the_lake(tmp_path, monkeypatch):
    lake = _lake(tmp_path)
    _add_specimen(lake, CANON, model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    _build(lake, force=True)

    def boom(*_a, **_k):
        raise AssertionError("query walked the lake")

    monkeypatch.setattr(mx, "walk_specimen_files", boom)
    row = mx.query_specimen(CANON, index_dir=lake / "index", lake=lake)
    assert row["slug"] == CANON


def test_query_is_under_50ms(tmp_path):
    lake = _lake(tmp_path)
    _add_specimen(lake, CANON, model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    _build(lake, force=True)
    # Warm.
    mx.query_specimen(CANON, index_dir=lake / "index", lake=lake)
    samples = []
    for _ in range(21):
        t0 = time.perf_counter()
        mx.query_specimen(CANON, index_dir=lake / "index", lake=lake)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    assert samples[len(samples) // 2] < 50


def test_incremental_update_walks_only_the_new_specimen(tmp_path, monkeypatch):
    lake = _lake(tmp_path)
    a = "fam--one@aaaaaaaaaaaa"
    b = "fam--two@bbbbbbbbbbbb"
    _add_specimen(lake, a, model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    _add_specimen(lake, b, model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    _build(lake, force=True)

    walked = []
    real = mx.walk_specimen_files

    def wrapped(root):
        walked.append(Path(root).name)
        return real(root)

    monkeypatch.setattr(mx, "walk_specimen_files", wrapped)
    new = "fam--three@cccccccccccc"
    _add_specimen(lake, new, model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    before = {
        p.name: p.stat().st_mtime_ns
        for p in (lake / "index" / "by-slug").glob("*.json")
    }
    out = mx.update_specimen(
        new, lake=lake, index_dir=lake / "index", manifest_dir=lake / "watch",
    )
    assert out["scanned_slugs"] == [new]
    assert out["n_scanned"] == 1
    assert walked == [new]
    after = {
        p.name: p.stat().st_mtime_ns
        for p in (lake / "index" / "by-slug").glob("*.json")
    }
    for name, ns in before.items():
        assert after[name] == ns, name
    assert f"{new}.json" in after
    cat = mx.load_catalog(index_dir=lake / "index", lake=lake)
    assert cat["n_specimens"] == 3


def test_rebuild_skips_unchanged_specimen_dirs(tmp_path, monkeypatch):
    lake = _lake(tmp_path)
    _add_specimen(lake, CANON, model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    cat1 = _build(lake, force=True)
    assert cat1["scanned_slugs"] == [CANON]
    walked = []
    real = mx.walk_specimen_files

    def wrapped(root):
        walked.append(Path(root).name)
        return real(root)

    monkeypatch.setattr(mx, "walk_specimen_files", wrapped)
    cat2 = _build(lake, force=False)
    assert walked == []
    assert cat2["skipped_slugs"] == [CANON]
    assert cat2["scanned_slugs"] == []


def test_overage_is_exact(tmp_path):
    lake = _lake(tmp_path)
    _add_specimen(
        lake, "big--a@aaaaaaaaaaaa", model_type="qwen3",
        architectures=["Qwen3ForCausalLM"], extra_bytes=5000,
    )
    _add_specimen(
        lake, "big--b@bbbbbbbbbbbb", model_type="qwen3",
        architectures=["Qwen3ForCausalLM"], extra_bytes=3000,
    )
    cat = _build(lake, force=True, budget=1000)
    used = cat["tier2_used_bytes"]
    assert used > 1000
    assert cat["tier2_budget"] == 1000
    assert cat["tier2_overage_bytes"] == used - 1000
    assert cat["tier2_used_minus_budget"] == used - 1000
    assert cat["over_budget"] is True
    rec = cat["retention_recommendation"]
    assert rec["operator_decision_only"] is True
    assert rec["does_not_retire"] is True
    assert rec["tier2_overage_bytes"] == used - 1000
    slugs = {r["slug"] for r in rec["ranked_redundant_bulk"]}
    assert "big--a@aaaaaaaaaaaa" in slugs
    assert "big--b@bbbbbbbbbbbb" in slugs


def test_unique_family_is_not_recommended_for_retention(tmp_path):
    lake = _lake(tmp_path)
    _add_specimen(
        lake, "qwen--a@aaaaaaaaaaaa", model_type="qwen3",
        architectures=["Qwen3ForCausalLM"], extra_bytes=100,
    )
    _add_specimen(
        lake, "qwen--b@bbbbbbbbbbbb", model_type="qwen3",
        architectures=["Qwen3ForCausalLM"], extra_bytes=100,
    )
    _add_specimen(
        lake, "moe--x@cccccccccccc", model_type="qwen3_moe",
        architectures=["Qwen3MoeForCausalLM"], extra_bytes=8000,
        num_experts=128,
    )
    cat = _build(lake, force=True, budget=1)
    rec = cat["retention_recommendation"]
    unique = {r["slug"] for r in rec["unique_families_not_recommended"]}
    redundant = {r["slug"] for r in rec["ranked_redundant_bulk"]}
    assert "moe--x@cccccccccccc" in unique
    assert "moe--x@cccccccccccc" not in redundant
    assert "qwen--a@aaaaaaaaaaaa" in redundant
    covering = {r["slug"] for r in rec["smallest_redundant_set_that_covers_overage"]}
    assert "moe--x@cccccccccccc" not in covering
    kept = {r["slug"] for r in rec["kept_as_family_representative"]}
    assert kept.isdisjoint(covering)
    assert len(kept) == 1  # the smaller qwen3 stays as the family representative


def test_partial_orphan_and_stale(tmp_path):
    lake = _lake(tmp_path)
    _add_specimen(
        lake, "ok--a@aaaaaaaaaaaa", model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
    )
    _add_specimen(
        lake, "part--b@bbbbbbbbbbbb", model_type="qwen3",
        architectures=["Qwen3ForCausalLM"], location="partial",
    )
    # Orphan watch: identity with no body.
    (lake / "watch" / "ghost--c@cccccccccccc.json").write_text(json.dumps({
        "repo": "ghost/c", "revision": "c" * 12, "files": ["config.json"],
        "sizes": {"config.json": 1}, "expected": 1,
    }))
    # Orphan lake manifest.
    (lake / "manifests" / "ghost--d@dddddddddddd.json").write_text(json.dumps({
        "repo": "ghost/d", "revision": "d" * 12, "bytes": 1, "n_files": 1,
    }))
    # Stale: watch size does not match disk.
    stale = "stale--e@eeeeeeeeeeee"
    _add_specimen(lake, stale, model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    watch = json.loads((lake / "watch" / f"{stale}.json").read_text())
    watch["sizes"]["config.json"] = 1
    (lake / "watch" / f"{stale}.json").write_text(json.dumps(watch))
    cat = _build(lake, force=True)
    an = cat["anomalies"]
    assert "part--b@bbbbbbbbbbbb" in an["partial"]
    assert "ghost--c@cccccccccccc" in an["orphaned_watch"]
    assert "ghost--d@dddddddddddd" in an["orphaned_manifests"]
    assert stale in an["stale_seals"]
    row = mx.query_specimen("part--b@bbbbbbbbbbbb", index_dir=lake / "index", lake=lake)
    assert row["seal_status"] == "PARTIAL"
    assert row["location"] == "partial"
    assert row["storage_role"] == "PARTIAL"


def test_index_refuses_to_write_under_specimens(tmp_path):
    lake = _lake(tmp_path)
    _add_specimen(lake, CANON, model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    with pytest.raises(PermissionError):
        mx.build(lake=lake, index_dir=lake / "specimens" / "index", force=True)
    with pytest.raises(PermissionError):
        mx.refuse_specimens_write(lake / "specimens" / CANON / "x.json", lake)


def test_build_does_not_touch_specimen_mtimes(tmp_path):
    lake = _lake(tmp_path)
    body = _add_specimen(lake, CANON, model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    before = {p: p.stat().st_mtime_ns for p in body.rglob("*") if p.is_file()}
    dir_before = body.stat().st_mtime_ns
    _build(lake, force=True)
    after = {p: p.stat().st_mtime_ns for p in body.rglob("*") if p.is_file()}
    assert after == before
    assert body.stat().st_mtime_ns == dir_before
    assert not (body / "MODEL_LAKE_SPECIMEN_SEAL.json").exists()


def test_layout_names_storage_roles():
    loc = mx.layout(lake="/Volumes/corpdrive/hawking-modellake")
    assert loc["roles"]["specimens"]["storage_role"] == "TIER2_COLD"
    assert loc["roles"]["partial"]["storage_role"] == "PARTIAL"
    assert loc["roles"]["logs"]["path"] == "logs/"
    # partial/ and claims/ exist only during an acquisition; an idle lake has
    # neither, so the layout must mark them transient rather than missing.
    assert loc["roles"]["partial"]["transient"] is True
    assert loc["roles"]["claims"]["transient"] is True
    assert "transient" not in loc["roles"]["specimens"]
    assert loc["roles"]["specimens"]["writable_by_index"] is False
    assert loc["roles"]["index"]["writable_by_index"] is True
    assert loc["index_never_writes_specimens"] is True


def test_lineage_wrappers_drive_the_index(tmp_path):
    lake = _lake(tmp_path)
    _add_specimen(lake, CANON, model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    cat = lin.build_lake_index(
        lake=lake, index_dir=lake / "index", manifest_dir=lake / "watch", force=True,
    )
    assert cat["n_specimens"] == 1
    row = lin.query_lake_specimen(
        CANON, index_dir=lake / "index", lake=lake,
    )
    assert row["slug"] == CANON
    listed = lin.lake_index(index_dir=lake / "index", lake=lake)
    assert listed["present"] is True
    assert listed["n_specimens"] == 1
    loc = lin.lake_layout(lake=lake)
    assert loc["roles"]["specimens"]["writable_by_index"] is False
    new = "extra--z@zzzzzzzzzzzz"
    _add_specimen(lake, new, model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    upd = lin.update_lake_specimen(
        new, lake=lake, index_dir=lake / "index", manifest_dir=lake / "watch",
    )
    assert upd["scanned_slugs"] == [new]


def test_list_watch_slugs_reads_git_objects_not_the_sparse_tree():
    slugs = mx.list_watch_slugs()
    assert CANON in slugs
    assert len(slugs) >= 40


def test_retired_watch_manifests_leave_the_live_queue(tmp_path):
    """A retirement must not read back as an orphan forever.

    git ls-tree -r is recursive, so watch-manifests/retired/<slug>.json would
    still be counted as queued and every retired slug would show up in
    anomalies.orphaned_watch on every build.
    """
    root = tmp_path / "repo"
    live = root / lin.WATCH_MANIFEST_REL
    (live / "retired").mkdir(parents=True)
    (live / "still--queued@aaaaaaaaaaaa.json").write_text("{}")
    (live / "retired" / "taken--out@bbbbbbbbbbbb.json").write_text("{}")
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "x"], check=True)

    slugs = mx.list_watch_slugs(git_root=root)
    assert slugs == ["still--queued@aaaaaaaaaaaa"], slugs


def test_unknown_family_is_not_clustered_as_redundant_bulk(tmp_path):
    lake = _lake(tmp_path)
    _add_specimen(
        lake, "wan--a@aaaaaaaaaaaa", model_type="",
        architectures=[], extra_bytes=5000,
    )
    # Empty model_type becomes UNKNOWN. A second UNKNOWN body is a different
    # architecture, not a duplicate.
    src = lake / "specimens" / "wan--a@aaaaaaaaaaaa" / "config.json"
    src.write_text(json.dumps({"architectures": ["SomethingElse"]}))
    _add_specimen(
        lake, "evo--b@bbbbbbbbbbbb", model_type="",
        architectures=[], extra_bytes=8000,
    )
    (lake / "specimens" / "evo--b@bbbbbbbbbbbb" / "config.json").write_text(
        json.dumps({"architectures": ["Evo"]})
    )
    cat = _build(lake, force=True, budget=1)
    rec = cat["retention_recommendation"]
    unique = {r["slug"] for r in rec["unique_families_not_recommended"]}
    assert "wan--a@aaaaaaaaaaaa" in unique
    assert "evo--b@bbbbbbbbbbbb" in unique
    redundant = {r["slug"] for r in rec["ranked_redundant_bulk"]}
    assert "wan--a@aaaaaaaaaaaa" not in redundant
    assert "evo--b@bbbbbbbbbbbb" not in redundant


def test_unsafe_slug_is_refused():
    with pytest.raises(mx.IndexError):
        mx._safe_slug("../escape")
    with pytest.raises(mx.IndexError):
        mx._safe_slug("a/b")
    with pytest.raises(mx.IndexError):
        mx._safe_slug("")


def test_receipt_records_real_lake_measurements():
    dest = Path("receipts/future/MODELLAKE_INDEX.json")
    if not dest.is_file():
        pytest.fail(
            f"{dest} does not exist. The index must be built over the real lake "
            "and the producer must write this receipt."
        )
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert doc["produced_by"] == "tools/odyssey/modellake_index.py"
    assert doc["n_specimens"] == 55
    assert doc["tier2_budget"] == ml.TIER2_BUDGET
    assert doc["tier2_overage_bytes"] == doc["tier2_used_bytes"] - ml.TIER2_BUDGET
    assert doc["query_ms"]["median_ms"] < 50
    assert doc["query_ms"]["pass"] is True
    assert doc["query_ms"]["evidence_tier"] == "HARDWARE_MEASURED"
    assert doc["incremental"]["proportional"] is True
    assert doc["incremental"]["n_scanned"] == 1
    assert doc["specimens_dir"]["unchanged"] is True
    assert doc["specimens_dir"]["bytes_written_under_specimens"] == 0
    assert doc["acceptance"]["query_median_ms_under_50"] is True
    assert doc["acceptance"]["incremental_proportional"] is True
    assert doc["acceptance"]["zero_bytes_under_specimens"] is True
    assert doc["gpu_authority"] is False
    assert "FPGA/U50" in doc["absent_hardware_not_measured"]
    rec = doc["retention_recommendation"]
    assert rec["operator_decision_only"] is True
    assert rec["does_not_retire"] is True
