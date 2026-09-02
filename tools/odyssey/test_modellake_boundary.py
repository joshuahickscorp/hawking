"""Product boundary: artifacts resolve from config, not from cwd or checkout."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tools.odyssey import modellake as ml
from tools.odyssey import product_boundary as pb

MODELLAKE_PY = Path(ml.__file__).resolve()
REPO = Path(ml.REPO).resolve()


def _cfg(tmp: Path, *, relative_roots: bool = False) -> Path:
    arts = tmp / "arts"
    spec = arts / "specimens"
    spec.mkdir(parents=True)
    slug = "demo--toy@abcabcabcabc"
    (spec / slug).mkdir()
    (spec / slug / "config.json").write_text('{"model_type":"toy"}')
    (arts / "nr").mkdir()
    (arts / "nx").mkdir()
    (arts / "partial").mkdir()
    (arts / "stage").mkdir()
    (arts / "manifests").mkdir()
    (arts / "watch").mkdir()
    roots = {
        "specimens": "arts/specimens" if relative_roots else str(spec),
        "partial": "arts/partial" if relative_roots else str(arts / "partial"),
        "nr": "arts/nr" if relative_roots else str(arts / "nr"),
        "nx": "arts/nx" if relative_roots else str(arts / "nx"),
        "stage": "arts/stage" if relative_roots else str(arts / "stage"),
        "lake_manifests": "arts/manifests" if relative_roots else str(arts / "manifests"),
        "watch_manifests": "arts/watch" if relative_roots else str(arts / "watch"),
    }
    path = tmp / "hawking.json"
    path.write_text(json.dumps({
        "schema": pb.SCHEMA,
        "artifact_roots": roots,
    }))
    return path


def test_main_calls_resolve_artifact():
    names = ml.main.__code__.co_names
    assert "resolve_artifact" in names
    assert "load_config" in names
    assert "discover_machine" in names


def test_safe_defaults_are_not_the_developer_checkout():
    d = pb.safe_defaults()
    roots = d["artifact_roots"]
    for key, val in roots.items():
        assert str(REPO) not in val, (key, val)
    assert d["defaults"]["cwd_is_not_an_artifact_root"] is True
    assert d["evidence_tier"] == "STATIC"


def test_resolve_artifact_from_config_not_cwd(tmp_path, monkeypatch):
    cfg_path = _cfg(tmp_path)
    foreign = tmp_path / "foreign-cwd"
    foreign.mkdir()
    monkeypatch.chdir(foreign)
    cfg = pb.load_config(cfg_path)
    out = pb.resolve_artifact("demo--toy@abcabcabcabc", cfg)
    assert out["present"] is True
    assert out["cwd_independent"] is True
    assert out["checkout_independent"] is True
    assert out["path"] == str(tmp_path / "arts" / "specimens" / "demo--toy@abcabcabcabc")
    assert str(REPO) not in out["path"]
    assert out["cwd"] == str(foreign)
    assert out["resolved_from"] == "config.artifact_roots.specimens"
    assert out["evidence_tier"] == "STATIC"


def test_relative_roots_resolve_against_config_dir_not_cwd(tmp_path, monkeypatch):
    cfg_path = _cfg(tmp_path, relative_roots=True)
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    monkeypatch.chdir(foreign)
    cfg = pb.load_config(cfg_path)
    out = pb.resolve_artifact("demo--toy@abcabcabcabc", cfg)
    assert out["present"] is True
    assert out["path"].startswith(str(tmp_path / "arts"))
    assert not out["path"].startswith(str(foreign))


def test_discover_config_ignores_cwd(tmp_path, monkeypatch):
    decoy = tmp_path / "cwd"
    decoy.mkdir()
    (decoy / "config.json").write_text("{}")
    monkeypatch.chdir(decoy)
    found = pb.discover_config(env={"HAWKING_HOME": str(tmp_path / "no-home")})
    assert found is None


def test_discover_config_uses_hawking_config_env(tmp_path, monkeypatch):
    cfg_path = _cfg(tmp_path)
    monkeypatch.chdir(tmp_path / "arts")
    found = pb.discover_config(env={"HAWKING_CONFIG": str(cfg_path)})
    assert found == cfg_path.resolve()


def test_install_update_recover_do_not_write(tmp_path):
    cfg = pb.load_config(_cfg(tmp_path))
    slug = "demo--toy@abcabcabcabc"
    inst = pb.install_plan(slug, cfg)
    upd = pb.update_plan(cfg)
    rec = pb.recover_plan(slug, cfg, reacquisition="hf download demo/toy")
    assert inst["wrote"] is False
    assert upd["wrote"] is False and upd["fetched"] is False
    assert rec["wrote"] is False and rec["spawned"] is False
    assert rec["never_restart_healthy_worker"] is True
    assert inst["atomic_rename"] is True


def test_discover_machine_labels_static_and_absent_hardware():
    m = pb.discover_machine()
    assert m["evidence_tier"] == "STATIC"
    assert m["gpu_authority"] is False
    assert "FPGA/U50" in m["absent_as_model_not_measurement"]
    assert "DGX" in m["absent_as_model_not_measurement"]
    assert "eGPU" in m["absent_as_model_not_measurement"]
    assert m["present_domains"]["CPU"] is True
    assert m["cwd_is_not_used_for_artifacts"] is True


def test_cli_resolve_from_a_foreign_cwd(tmp_path):
    """The acceptance proof: different cwd + config => configured path."""
    cfg_path = _cfg(tmp_path)
    foreign = tmp_path / "not-the-repo"
    foreign.mkdir()
    r = subprocess.run(
        [sys.executable, str(MODELLAKE_PY), "resolve",
         "--config", str(cfg_path),
         "--artifact", "demo--toy@abcabcabcabc"],
        cwd=str(foreign),
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["present"] is True
    assert out["path"] == str(tmp_path / "arts" / "specimens" / "demo--toy@abcabcabcabc")
    assert out["cwd"] == str(foreign)
    assert str(REPO) not in out["path"]
    assert out["resolved_from"].startswith("config.artifact_roots.")
