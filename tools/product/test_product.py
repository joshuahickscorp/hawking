"""Era V product entry: config, not cwd; refuse unmeasured values."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.odyssey import product_boundary as pb
from tools.product.cli import (
    cmd_entry,
    cmd_resolve,
    cmd_scoreboard,
    main,
)
from tools.product.config import ConfigClosed, require_config
from tools.product.install import (
    InstallError,
    artifact_digest,
    assert_not_live_volume,
    install_artifact,
    staleness,
    update_artifact,
)
from tools.product.scoreboard import (
    UnmeasuredError,
    load_scoreboard,
    qualify,
    require_measured,
)

REPO = Path(__file__).resolve().parents[2]
CLI = Path(__file__).resolve().parent / "cli.py"
SCHEMA = pb.SCHEMA
SLUG = "demo--toy@abcabcabcabc"


def _env(tmp: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.pop("HAWKING_CONFIG", None)
    env["HAWKING_HOME"] = str(tmp / "no-home")
    if extra:
        env.update(extra)
    return env


def _cfg(tmp: Path, *, relative_roots: bool = False) -> Path:
    arts = tmp / "arts"
    spec = arts / "specimens"
    spec.mkdir(parents=True)
    (spec / SLUG).mkdir()
    (spec / SLUG / "config.json").write_text('{"model_type":"toy"}')
    partial = arts / "partial"
    (partial / SLUG).mkdir(parents=True)
    (partial / SLUG / "payload.txt").write_text("payload-v1")
    (arts / "install").mkdir()
    (arts / "nr").mkdir()
    (arts / "nx").mkdir()
    (arts / "stage").mkdir()
    (arts / "manifests").mkdir()
    (arts / "watch").mkdir()
    roots = {
        "specimens": "arts/specimens" if relative_roots else str(spec),
        "partial": "arts/partial" if relative_roots else str(partial),
        "install": "arts/install" if relative_roots else str(arts / "install"),
        "nr": "arts/nr" if relative_roots else str(arts / "nr"),
        "nx": "arts/nx" if relative_roots else str(arts / "nx"),
        "stage": "arts/stage" if relative_roots else str(arts / "stage"),
        "lake_manifests": "arts/manifests" if relative_roots else str(arts / "manifests"),
        "watch_manifests": "arts/watch" if relative_roots else str(arts / "watch"),
    }
    path = tmp / "hawking.json"
    path.write_text(json.dumps({
        "schema": SCHEMA,
        "artifact_roots": roots,
    }))
    return path


def _run(args: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )


def test_cmd_entry_calls_boundary_symbols():
    names = cmd_entry.__code__.co_names
    assert "require_config" in names
    assert "machine_inventory" in names
    assert "resolve_artifact" in names
    assert "install_plan" in names
    assert "update_plan" in names
    assert "recover_plan" in names
    assert "load_scoreboard" in names
    assert "require_measured" in names


def test_cmd_resolve_calls_require_config_and_resolve_artifact():
    names = cmd_resolve.__code__.co_names
    assert "require_config" in names
    assert "resolve_artifact" in names


def test_cmd_scoreboard_calls_require_measured():
    names = cmd_scoreboard.__code__.co_names
    assert "load_scoreboard" in names
    assert "require_measured" in names
    assert "qualify" in names


def test_require_config_missing_fails_closed_and_does_not_call_safe_defaults(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    called: list[bool] = []
    real = pb.safe_defaults

    def wrapped(*args, **kwargs):
        called.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(pb, "safe_defaults", wrapped)
    with pytest.raises(ConfigClosed, match="refusing to guess"):
        require_config(env={"HAWKING_HOME": str(tmp_path / "no-home")})
    assert called == []


def test_require_config_corrupt_fails_closed(tmp_path, monkeypatch):
    bad = tmp_path / "broken.json"
    bad.write_text("{not json")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigClosed, match="corrupt"):
        require_config(explicit=bad)
    with pytest.raises(ConfigClosed):
        require_config(explicit=bad)
    try:
        require_config(explicit=bad)
    except ConfigClosed as exc:
        assert exc.recovery["fail_closed"] is True
        assert exc.recovery["did_not_guess_artifact_roots"] is True
        assert exc.recovery["never_restart_healthy_worker"] is True


def test_require_config_wrong_schema_fails_closed(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({
        "schema": "not.the.product.schema",
        "artifact_roots": {"specimens": str(tmp_path)},
    }))
    with pytest.raises(ConfigClosed, match="unsupported or missing schema"):
        require_config(explicit=path)


def test_require_config_empty_roots_fails_closed(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"schema": SCHEMA, "artifact_roots": {}}))
    with pytest.raises(ConfigClosed, match="refusing to guess"):
        require_config(explicit=path)


def test_require_config_does_not_keep_undeclared_lake_defaults(tmp_path):
    cfg_path = _cfg(tmp_path)
    cfg = require_config(explicit=cfg_path)
    assert "lake" not in cfg["artifact_roots"]
    for value in cfg["artifact_roots"].values():
        assert "/Users/scammermike/Downloads/hawking" not in str(value)
        assert str(REPO) not in str(value) or str(tmp_path) in str(value)


def test_cli_resolve_from_a_foreign_cwd(tmp_path):
    cfg_path = _cfg(tmp_path)
    foreign = tmp_path / "not-the-repo"
    foreign.mkdir()
    r = _run(
        ["resolve", "--config", str(cfg_path), "--artifact", SLUG],
        cwd=foreign,
        env=_env(tmp_path),
    )
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["present"] is True
    assert out["path"] == str(tmp_path / "arts" / "specimens" / SLUG)
    assert out["cwd"] == str(foreign)
    assert str(REPO) not in out["path"]
    assert "/Users/scammermike/Downloads/hawking" not in out["path"]
    assert out["resolved_from"].startswith("config.artifact_roots.")
    assert out["cwd_independent"] is True
    assert out["checkout_independent"] is True


def test_cli_entry_from_a_foreign_cwd(tmp_path):
    cfg_path = _cfg(tmp_path)
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    r = _run(
        ["entry", "--config", str(cfg_path), "--artifact", SLUG],
        cwd=foreign,
        env=_env(tmp_path),
    )
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["ok"] is True
    assert out["fail_closed"] is True
    assert out["checkout_independent"] is True
    assert out["cwd"] == str(foreign)
    assert out["artifact"]["present"] is True
    assert out["artifact"]["path"] == str(tmp_path / "arts" / "specimens" / SLUG)
    assert out["machine"]["evidence_tier"] == "STATIC"
    assert out["machine"]["developer_checkout_not_assumed"] is True
    assert "FPGA/U50" in out["machine"]["absent_as_model_not_measurement"]
    assert out["machine"]["gpu_authority"] is False
    assert out["handoff"]["checkout_not_required"] is True
    assert out["install_plan"]["wrote"] is False
    assert out["updates"]["fetched"] is False
    assert out["recovery"]["never_restart_healthy_worker"] is True


def test_cli_corrupt_config_fails_closed_from_foreign_cwd(tmp_path):
    bad = tmp_path / "corrupt.json"
    bad.write_text("{")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    r = _run(
        ["entry", "--config", str(bad), "--artifact", SLUG],
        cwd=foreign,
        env=_env(tmp_path),
    )
    assert r.returncode == 2
    out = json.loads(r.stdout)
    assert out["ok"] is False
    assert out["fail_closed"] is True
    assert out["did_not_guess"] is True
    assert out["recovery"]["did_not_guess_artifact_roots"] is True
    assert "corrupt" in out["error"].lower() or "unreadable" in out["error"].lower()


def test_cli_missing_config_fails_closed_from_foreign_cwd(tmp_path):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    r = _run(
        ["resolve", "--artifact", SLUG],
        cwd=foreign,
        env=_env(tmp_path),
    )
    assert r.returncode == 2
    out = json.loads(r.stdout)
    assert out["ok"] is False
    assert out["fail_closed"] is True
    assert "guess" in out["error"].lower()


def test_scoreboard_refuses_unmeasured_value(tmp_path):
    receipt = tmp_path / "row.json"
    receipt.write_text(json.dumps({
        "schema": "hawking.product.measurement.v1",
        "metrics": {
            "accepted_tps": {
                "value": 12.5,
                "evidence_tier": "HARDWARE_MEASURED",
            },
            "energy": {
                "value": None,
                "reason": "no power instrumentation on this run",
            },
        },
    }))
    board = load_scoreboard([receipt])
    measured = require_measured(board, "accepted_tps")
    assert measured["value"] == 12.5
    assert measured["evidence_tier"] == "HARDWARE_MEASURED"
    energy = board["metrics"]["energy"]
    assert energy["state"] == "UNMEASURED"
    assert energy["value"] is None
    with pytest.raises(UnmeasuredError, match="refusing"):
        require_measured(board, "energy")
    with pytest.raises(UnmeasuredError, match="refusing"):
        require_measured(board, "ttft")


def test_scoreboard_unmeasured_cell_is_never_a_number(tmp_path):
    receipt = tmp_path / "empty.json"
    receipt.write_text(json.dumps({"metrics": {}}))
    board = load_scoreboard([receipt])
    for name, cell in board["metrics"].items():
        if cell["state"] == "UNMEASURED":
            assert cell["value"] is None, name
            assert cell["value"] != 0


def test_scoreboard_measured_zero_is_kept(tmp_path):
    receipt = tmp_path / "zero.json"
    receipt.write_text(json.dumps({
        "metrics": {
            "memory": {"value": 0, "evidence_tier": "HARDWARE_MEASURED"},
        },
    }))
    board = load_scoreboard([receipt])
    cell = require_measured(board, "memory")
    assert cell["value"] == 0


def test_scoreboard_refuses_number_without_tier(tmp_path):
    receipt = tmp_path / "bare.json"
    receipt.write_text(json.dumps({"metrics": {"accepted_tps": 99}}))
    board = load_scoreboard([receipt])
    assert board["metrics"]["accepted_tps"]["state"] == "UNMEASURED"
    assert board["metrics"]["accepted_tps"]["value"] is None
    with pytest.raises(UnmeasuredError, match="refusing"):
        require_measured(board, "accepted_tps")


def test_scoreboard_refuses_to_merge_conflicting_measurements(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({
        "metrics": {"accepted_tps": {"value": 1, "evidence_tier": "HARDWARE_MEASURED"}},
    }))
    b.write_text(json.dumps({
        "metrics": {"accepted_tps": {"value": 2, "evidence_tier": "HARDWARE_MEASURED"}},
    }))
    with pytest.raises(UnmeasuredError, match="refusing to merge"):
        load_scoreboard([a, b])


def test_qualify_refuses_unmeasured_required_metric(tmp_path):
    receipt = tmp_path / "row.json"
    receipt.write_text(json.dumps({
        "metrics": {
            "capability": {"value": "ok", "evidence_tier": "FUNCTIONAL_SIM"},
        },
    }))
    board = load_scoreboard([receipt])
    qualify(board, ["capability"])
    with pytest.raises(UnmeasuredError, match="qualification refused"):
        qualify(board, ["capability", "energy"])


def test_cli_scoreboard_require_unmeasured_fails_closed(tmp_path):
    receipt = tmp_path / "row.json"
    receipt.write_text(json.dumps({
        "metrics": {
            "accepted_tps": {"value": 3, "evidence_tier": "COST_MODEL"},
        },
    }))
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    r = _run(
        ["scoreboard", "--receipts", str(receipt), "--require", "energy"],
        cwd=foreign,
        env=_env(tmp_path),
    )
    assert r.returncode == 2
    out = json.loads(r.stdout)
    assert out["fail_closed"] is True
    assert out["error_type"] == "UnmeasuredError"
    assert "energy" in out["error"]


def test_install_and_staleness_roundtrip(tmp_path):
    source = tmp_path / "src" / SLUG
    source.mkdir(parents=True)
    (source / "payload.txt").write_text("v1")
    dest = tmp_path / "dst" / SLUG
    first = install_artifact(source, dest, slug=SLUG)
    assert first["wrote"] is True
    assert (dest / "install.json").is_file()
    assert (dest / "payload.txt").read_text() == "v1"
    check = staleness(source, dest)
    assert check["stale"] is False
    (source / "payload.txt").write_text("v2")
    check = staleness(source, dest)
    assert check["stale"] is True
    updated = update_artifact(source, dest, slug=SLUG)
    assert updated["action"] == "UPDATED"
    assert (dest / "payload.txt").read_text() == "v2"
    assert dest.with_name(dest.name + ".prev").is_dir()
    again = update_artifact(source, dest, slug=SLUG)
    assert again["action"] == "ALREADY_CURRENT"
    assert again["wrote"] is False


def test_cli_install_from_foreign_cwd(tmp_path):
    cfg_path = _cfg(tmp_path)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    r = _run(
        ["install", "--config", str(cfg_path), "--artifact", SLUG],
        cwd=foreign,
        env=_env(tmp_path),
    )
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["wrote"] is True
    dest = Path(out["destination"])
    assert dest.is_dir()
    assert (dest / "payload.txt").read_text() == "payload-v1"
    assert dest.is_relative_to(tmp_path)
    stale = _run(
        ["stale", "--config", str(cfg_path), "--artifact", SLUG],
        cwd=foreign,
        env=_env(tmp_path),
    )
    assert json.loads(stale.stdout)["stale"] is False


def test_install_refuses_live_volume():
    with pytest.raises(InstallError, match="live volume"):
        assert_not_live_volume(Path("/Volumes/corpdrive/hawking-modellake/specimens/x"), op="write")


def test_install_refuses_overwrite(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.txt").write_text("a")
    dest = tmp_path / "dst"
    dest.mkdir()
    (dest / "b.txt").write_text("b")
    with pytest.raises(InstallError, match="refusing overwrite"):
        install_artifact(source, dest, slug="x")


def test_digest_skips_stamp(tmp_path):
    tree = tmp_path / "t"
    tree.mkdir()
    (tree / "a.txt").write_text("a")
    d1 = artifact_digest(tree)
    (tree / "install.json").write_text("{}")
    d2 = artifact_digest(tree)
    assert d1 == d2


def test_main_discover_machine_does_not_need_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["discover-machine"])
    assert rc == 0
