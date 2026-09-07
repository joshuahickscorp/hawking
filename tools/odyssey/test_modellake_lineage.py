"""Roadmap-layer lineage: a named real specimen is expressible.

Call sites, not imports: these tests invoke express_lineage, role_metadata,
architecture_fingerprint, artifact_lineage, load_watch_manifest,
registry_index. architecture_fingerprint must CALL arch_recognizer.recognize.
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.odyssey import modellake as ml
from tools.odyssey import modellake_lineage as lin
from tools.odyssey.product_boundary import load_config, safe_defaults

CANON = lin.CANONICAL_SPECIMEN


def _tiny_safetensors(path: Path, names: list[str]) -> None:
    header = {
        n: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for n in names
    }
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\x00\x00\x00\x00")


def _boundary(tmp: Path, *, with_source: bool = True) -> dict:
    roots = tmp / "artifacts"
    (roots / "specimens").mkdir(parents=True)
    (roots / "partial").mkdir()
    (roots / "nr").mkdir()
    (roots / "nx").mkdir()
    (roots / "stage").mkdir()
    (roots / "manifests").mkdir()
    (roots / "watch").mkdir()
    slug = CANON
    if with_source:
        src = roots / "specimens" / slug
        src.mkdir()
        (src / "config.json").write_text(json.dumps({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "max_position_embeddings": 40960,
            "vocab_size": 151936,
        }))
        _tiny_safetensors(src / "model.safetensors", [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "lm_head.weight",
        ])
        (roots / "manifests" / f"{slug}.json").write_text(json.dumps({
            "repo": lin.CANONICAL_REPO,
            "revision": lin.CANONICAL_REVISION,
            "resolved_sha": lin.CANONICAL_REVISION,
            "bytes": 1519280128,
            "n_files": 10,
            "reacquisition": (
                f"hf download {lin.CANONICAL_REPO} --revision "
                f"{lin.CANONICAL_REVISION} --local-dir <dest>"
            ),
            "acquired_at": "2026-08-25T04:14:38Z",
        }))
    cfg_path = tmp / "hawking.json"
    cfg_path.write_text(json.dumps({
        "schema": "hawking.product.boundary.v1",
        "artifact_roots": {
            "specimens": str(roots / "specimens"),
            "partial": str(roots / "partial"),
            "nr": str(roots / "nr"),
            "nx": str(roots / "nx"),
            "stage": str(roots / "stage"),
            "lake_manifests": str(roots / "manifests"),
            "watch_manifests": str(roots / "watch"),
        },
    }))
    return load_config(cfg_path)


def test_main_calls_express_lineage():
    """A revert that leaves the helper standing but drops the CLI call fails."""
    import inspect
    src = inspect.getsource(ml.main)
    assert "out = express_lineage(" in src
    assert "out = resolve_artifact(" in src


def test_fingerprint_calls_recognizer_recognize():
    assert "recognize" in lin.architecture_fingerprint.__code__.co_names or \
        "_recognize" in lin.architecture_fingerprint.__code__.co_names
    assert "recognize" in lin._recognize.__code__.co_names


def test_express_lineage_calls_the_pieces():
    names = lin.express_lineage.__code__.co_names
    for needed in (
        "load_watch_manifest",
        "role_metadata",
        "architecture_fingerprint",
        "artifact_lineage",
        "storage_tier_for",
        "derive_lifecycle",
    ):
        assert needed in names, needed


def test_watch_manifest_of_the_named_real_specimen_loads_from_git():
    """The sparse worktree does not materialize watch-manifests/; git does."""
    doc = lin.load_watch_manifest(
        CANON, manifest_dir=Path("/no/such/watch-manifests")
    )
    assert doc is not None, "git show of the real watch-manifest must work"
    assert doc["repo"] == lin.CANONICAL_REPO
    assert doc["revision"].startswith("c1899de289a0")
    assert "model.safetensors" in doc["files"]
    assert doc["_manifest_source"].startswith("git:HEAD:")


def test_registry_index_names_the_real_specimen():
    idx = lin.registry_index(slugs=[CANON])
    assert idx["n"] == 1
    assert idx["specimens"][0]["id"] == CANON
    assert idx["specimens"][0]["repo"] == lin.CANONICAL_REPO


def test_role_metadata_qwen3_is_dense_decoder_and_long_context():
    role = lin.role_metadata(
        {"model_type": "qwen3", "max_position_embeddings": 40960,
         "architectures": ["Qwen3ForCausalLM"]},
        ["config.json", "model.safetensors"],
        repo=lin.CANONICAL_REPO, slug=CANON,
    )
    assert "dense decoder" in role["roles"]
    assert "long-context" in role["roles"]
    assert role["evidence_tier"] == "STATIC"


def test_role_metadata_moe_and_vl():
    moe = lin.role_metadata(
        {"model_type": "qwen3_moe", "num_experts": 128},
        ["model-00001-of-00016.safetensors"],
        repo="Qwen/Qwen3-30B-A3B", slug="Qwen--Qwen3-30B-A3B@ad44e777bcd1",
    )
    assert moe["primary"] == "MoE"
    assert "extreme expert count" in moe["roles"]
    vl = lin.role_metadata(
        {"model_type": "qwen3_vl"}, [],
        repo="Qwen/Qwen3-VL-8B-Instruct", slug="Qwen--Qwen3-VL-8B-Instruct@0c351dd01ed8",
    )
    assert "multimodal" in vl["roles"]


def test_express_lineage_for_the_named_real_specimen(tmp_path):
    cfg = _boundary(tmp_path)
    # Put the real watch-manifest next to the config so directory wins.
    watch = lin.load_watch_manifest(CANON)
    assert watch is not None
    (Path(cfg["artifact_roots"]["watch_manifests"]) / f"{CANON}.json").write_text(
        json.dumps({k: v for k, v in watch.items() if not str(k).startswith("_")})
    )
    out = lin.express_lineage(CANON, config=cfg)
    assert out["slug"] == CANON
    assert out["evidence_tier"] == "STATIC"
    assert out["loaded_weights"] is False
    assert out["wrote"] is False
    assert out["provenance"]["repo"] == lin.CANONICAL_REPO
    assert out["provenance"]["resolved_sha"].startswith("c1899de289a0")
    assert out["role"]["primary"] in lin.DIVERSITY_ROLES
    assert "long-context" in out["role"]["roles"]
    fp = out["architecture_fingerprint"]
    assert fp["loaded_weights"] is False
    assert fp["model_type"] == "qwen3"
    assert fp["strength"] == "ORGAN_FINGERPRINT"
    organs = {o["organ"] for o in fp["organs"]}
    assert "gqa_attention" in organs
    assert "mlp_gate_up" in organs
    stages = [s["stage"] for s in out["artifact_lineage"]]
    assert stages == ["SOURCE", "NR", "NX"]
    by = {s["stage"]: s for s in out["artifact_lineage"]}
    assert by["SOURCE"]["present"] is True
    assert by["NR"]["present"] is False
    assert "no NR" in by["NR"]["absent_because"]
    assert by["NX"]["present"] is False
    assert out["storage_tier"]["role"] == "TIER2_COLD"
    assert out["registry"]["lifecycle"] in lin.ROADMAP_LIFECYCLE
    assert out["registry"]["lifecycle"] in ("CENSUSED", "READY_COLD")


def test_lineage_without_source_stays_manifest_ready(tmp_path):
    cfg = _boundary(tmp_path, with_source=False)
    watch = lin.load_watch_manifest(CANON)
    (Path(cfg["artifact_roots"]["watch_manifests"]) / f"{CANON}.json").write_text(
        json.dumps({k: v for k, v in watch.items() if not str(k).startswith("_")})
    )
    out = lin.express_lineage(CANON, config=cfg)
    assert out["registry"]["lifecycle"] == "MANIFEST_READY"
    assert out["storage_tier"]["role"] == "GIT_METADATA"
    assert out["artifact_lineage"][0]["present"] is False


def test_nr_presence_advances_lifecycle_to_transfer_ready(tmp_path):
    cfg = _boundary(tmp_path)
    nr = Path(cfg["artifact_roots"]["nr"]) / CANON
    nr.mkdir()
    (nr / "nr.index.json").write_text("{}")
    out = lin.express_lineage(CANON, config=cfg)
    assert out["registry"]["lifecycle"] == "TRANSFER_READY"
    assert out["artifact_lineage"][1]["present"] is True


def test_empty_slug_is_a_lineage_error():
    try:
        lin.express_lineage("")
    except lin.LineageError:
        return
    raise AssertionError("empty slug must refuse")


def test_lineage_index_wrappers_call_the_index_module():
    """An import is not a call site. These wrappers must CALL the index."""
    assert "build" in lin.build_lake_index.__code__.co_names
    assert "query_specimen" in lin.query_lake_specimen.__code__.co_names
    assert "update_specimen" in lin.update_lake_specimen.__code__.co_names
    assert "load_catalog" in lin.lake_index.__code__.co_names
    assert "layout" in lin.lake_layout.__code__.co_names


def test_main_calls_index_symbols():
    import inspect
    src = inspect.getsource(ml.main)
    assert "out = build_lake_index(" in src
    assert "out = query_lake_specimen(" in src
    assert "out = update_lake_specimen(" in src
    assert "out = lake_index(" in src
    assert "out = express_lineage(" in src


# --- execution-class size tiering (operator decision 2026-09-05) ----------

def test_size_tier_for_boundaries():
    gib = 1024 ** 3
    assert lin.size_tier_for(0) == "A_TINY"
    assert lin.size_tier_for(8 * gib) == "A_TINY"
    assert lin.size_tier_for(8 * gib + 1) == "B_MID"
    assert lin.size_tier_for(40 * gib) == "B_MID"
    assert lin.size_tier_for(40 * gib + 1) == "C_LARGE"
    assert lin.size_tier_for(80 * gib) == "C_LARGE"
    assert lin.size_tier_for(80 * gib + 1) == "D_GIANT"
    assert lin.size_tier_for(2_000 * gib) == "D_GIANT"


def test_deferred_giants_are_named_and_ineligible():
    """Exactly the three operator-named giants, each with its own reason."""
    assert len(lin.DEFERRED_GIANTS) == 3
    for repo in (
        "moonshotai/Kimi-K3",
        "thinkingmachines/Inkling-Small",
        "windowsxp811203/Qwen3.8-Flash-Next-Abliterated",
    ):
        assert repo in lin.DEFERRED_GIANTS
        assert lin.DEFERRED_GIANTS[repo]
        assert lin.execution_eligible_for(repo) is False
        assert lin.deferred_reason_for(repo)


def test_non_deferred_specimen_is_eligible():
    assert lin.execution_eligible_for("Qwen/Qwen3-0.6B") is True
    assert lin.deferred_reason_for("Qwen/Qwen3-0.6B") is None


def test_deferred_does_not_mean_dropped():
    """The operator's distinction -- deferred, not dropped -- is in the code."""
    note = lin.DEFERRED_NOTE.lower()
    assert "not" in note and "dropped" in note
    assert "census" in note and "later pass" in note
