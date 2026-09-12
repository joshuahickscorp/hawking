"""G025: a checkpoint is only worth writing if a cold process can read it back.

An in-process round trip proves nothing -- the object is already in memory. The
restart here is a real one: a separate interpreter, a fresh registry, no shared
state but the file.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import hcli.tool_registry as tr

REPO = Path(__file__).resolve().parents[2]


def _reg(root, perms=None):
    return tr.default_tool_registry(
        str(root), repo_root=str(root),
        permissions=perms if perms is not None else
        {tr.READ_ONLY, tr.RESEARCH, tr.WORKSPACE_WRITE})


def test_checkpoint_survives_a_real_restart(tmp_path):
    reg = _reg(tmp_path)
    out = reg.invoke("campaign.checkpoint", {
        "objective": "close the gpu axis",
        "active_specimen": "higgs-audio-v3-sens-a",
        "why_this_specimen": "only arm whose jam is unattributed",
        "hypothesis": "down_proj:2 and embed_tokens:2 are safe alone and unsafe together",
        "next_action": "build awq-b3g64 + down_proj:2 alone and render 8 units",
        "open_jobs": ["gpu_sweep2"],
        "resource_deps": ["GPU free"],
    }).to_dict()
    assert out["ok"], out
    assert (tmp_path / tr.CONTINUATION).is_file()

    # A real restart: new interpreter, nothing carried over but the file.
    code = (
        "import sys, json; sys.path.insert(0, %r);"
        "import hcli.tool_registry as tr;"
        "reg = tr.default_tool_registry(%r, repo_root=%r, permissions={tr.READ_ONLY});"
        "print(json.dumps(reg.invoke('campaign.state', {}).to_dict()['value']['continuation']))"
        % (str(REPO), str(tmp_path), str(tmp_path))
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    back = json.loads(p.stdout.strip().splitlines()[-1])
    assert back["next_action"].startswith("build awq-b3g64")
    assert back["hypothesis"].startswith("down_proj:2")
    assert back["active_specimen"] == "higgs-audio-v3-sens-a"


def test_state_says_so_when_there_is_no_checkpoint(tmp_path):
    reg = _reg(tmp_path, {tr.READ_ONLY})
    cont = reg.invoke("campaign.state", {}).to_dict()["value"]["continuation"]
    assert isinstance(cont, str) and "no checkpoint" in cont


def test_newer_sovereign_parent_is_not_hidden_by_an_old_child_checkpoint(tmp_path):
    cont_path = tmp_path / tr.CONTINUATION
    cont_path.parent.mkdir(parents=True)
    cont_path.write_text(json.dumps({
        "objective": "old child task", "hypothesis": "old",
        "next_action": "old action", "written_at": 1,
    }))
    goal = tmp_path / "civilization" / "sovereign-goal.txt"
    goal.parent.mkdir(parents=True)
    goal.write_text(
        "PRIMARY OBJECTIVE: move the current frontier\n\n"
        "### Current closeout\nEarn the live gates before transition.\n")

    state = _reg(tmp_path, {tr.READ_ONLY}).invoke(
        "campaign.state", {}).to_dict()["value"]

    assert state["authoritative_parent"]["path"] == (
        "civilization/sovereign-goal.txt")
    assert state["authoritative_parent"]["primary_objective"] == (
        "move the current frontier")
    assert state["authoritative_parent"]["latest_section"] == "Current closeout"
    assert state["continuation_older_than_parent"] is True


@pytest.mark.parametrize("drop", ["objective", "hypothesis", "next_action"])
def test_schema_rejects_a_checkpoint_missing_a_load_bearing_field(tmp_path, drop):
    full = {"objective": "o", "hypothesis": "h", "next_action": "n"}
    full.pop(drop)
    out = _reg(tmp_path).invoke("campaign.checkpoint", full).to_dict()
    assert not out["ok"] and "missing required" in out["error"]
    assert not (tmp_path / tr.CONTINUATION).is_file()


@pytest.mark.parametrize("blank", ["objective", "hypothesis", "next_action"])
def test_handler_rejects_a_blank_field_the_schema_accepts(tmp_path, blank):
    """`required` is satisfied by "". The handler check is what makes it mean
    something, so it is reachable exactly here and nowhere else."""
    full = {"objective": "o", "hypothesis": "h", "next_action": "n"}
    full[blank] = "   "
    out = _reg(tmp_path).invoke("campaign.checkpoint", full).to_dict()
    assert out["ok"], out
    assert blank in str(out["value"]["refused"])
    assert not (tmp_path / tr.CONTINUATION).is_file()


def test_write_needs_the_permission(tmp_path):
    out = _reg(tmp_path, {tr.READ_ONLY}).invoke("campaign.checkpoint", {
        "objective": "o", "hypothesis": "h", "next_action": "n"}).to_dict()
    assert not out["ok"] and "permission denied" in out["error"]
    assert not (tmp_path / tr.CONTINUATION).is_file()
