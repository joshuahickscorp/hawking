"""A tool whose actionable output does not survive the caller's budget does not work.

The engine's closed-turn compactor keeps 500 characters of each observation
(Engine.CLOSED_OBSERVATION_CHARS). json.dumps preserves insertion order, so a
tool that emits an 8 KB summary and puts the target names in the tail is a
broken tool even when every byte is true. odyssey.ledger already learned this;
this module is the same law applied to every model-facing tool.

Each test truncates the tool's serialized `value` to BUDGET characters and
asserts the decision-relevant names/ids/counts still appear in that head.
"""
from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path

import pytest

from hcli.tool_registry import (
    COSTLY,
    DESTRUCTIVE,
    READ_ONLY,
    RESEARCH,
    REVERSIBLE_REPO,
    REVERSIBLE_RUNTIME,
    default_tool_registry,
)

BUDGET = 500
REPO = Path(__file__).resolve().parents[2]

# Every name currently returned by default_tool_registry(...).discover().
# A name missing here means a new tool shipped without an actionable-first
# audit. "compliant" = already survives BUDGET; "fix" = this file's red tests.
AUDITED_TOOLS = {
    "accelerator.benchmark": "compliant",
    "accelerator.inspect": "compliant",
    "acquisition.propose": "fix",
    "architecture.inspect": "compliant",
    "benchmark.run": "compliant",
    "campaign.guard": "compliant",
    "context.recall": "compliant",
    "doctor.inspect": "compliant",
    "doctor.query": "compliant",
    "filesystem.write": "compliant",
    "forbidden_fruit.lab": "compliant",
    "frontier.decide": "compliant",
    "frontier.escalate": "compliant",
    "fs.list": "compliant",
    "fs.read": "compliant",
    "fs.search": "compliant",
    "git.checkout-safe": "compliant",
    "git.diff": "compliant",
    "git.land.propose": "compliant",
    "git.log": "fix",
    "git.status": "compliant",
    "github.fetch": "compliant",
    "github.search": "compliant",
    "gravity.experiment": "compliant",
    "gravity.inspect": "compliant",
    "grok.swarm.launch": "compliant",
    "grok.swarm.propose": "compliant",
    "huggingface.download": "compliant",
    "huggingface.fetch_file": "compliant",
    "huggingface.history": "compliant",
    "huggingface.resolve": "compliant",
    "lake.census": "fix",
    "modellake.status": "compliant",
    "odyssey.add_to_eligibility": "compliant",
    "odyssey.anatomy": "fix",
    "odyssey.completions": "compliant",
    "odyssey.create_adversarial_probe": "compliant",
    "odyssey.create_transfer_probe": "compliant",
    "odyssey.cycle": "compliant",
    "odyssey.economics": "compliant",
    "odyssey.gravity_gauntlet": "compliant",
    "odyssey.harvest": "compliant",
    "odyssey.ingest": "compliant",
    "odyssey.ledger": "compliant",
    "odyssey.park_specimen": "compliant",
    "odyssey.patient": "compliant",
    # Audited 2026-09-07 when it shipped: total output 234 chars, `recorded` first via
    # _lead_with, so the whole result survives BUDGET with room to spare. This entry
    # exists because the audit test CAUGHT the tool shipping without one.
    "odyssey.record_measurement": "compliant",
    "odyssey.queue": "compliant",
    "odyssey.record_law": "compliant",
    "odyssey.record_scar": "compliant",
    "odyssey.retire": "compliant",
    "odyssey.status": "compliant",
    "odyssey.value": "compliant",
    "odyssey.write_packet": "compliant",
    "processes.list": "fix",
    "processes.orphaned": "fix",
    "processes.summary": "compliant",
    "receipt.read": "compliant",
    "roadmap.read": "compliant",
    "shell.exec": "compliant",
    "shell.readonly": "compliant",
    "specimens.registry": "fix",
    "tests.list": "compliant",
    "tests.run": "fix",
    "tools.catalog": "fix",
    "vmcp.capabilities": "compliant",
    "vmcp.inspect": "compliant",
    "vmcp.query": "compliant",
    "web.fetch": "compliant",
    "web.search": "compliant",
}


def _registry(workspace, **kwargs):
    permissions = kwargs.pop(
        "permissions",
        {READ_ONLY, RESEARCH, REVERSIBLE_REPO, REVERSIBLE_RUNTIME, COSTLY, DESTRUCTIVE},
    )
    return default_tool_registry(workspace, repo_root=REPO, permissions=permissions, **kwargs)


def _dump(value) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def _head(value, n: int = BUDGET) -> str:
    return _dump(value)[:n]


def _write_safetensors(path: Path, header: dict) -> None:
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 32)


# ---------------------------------------------------------------------------
# Coverage: every live discover() name is named in this audit.
# ---------------------------------------------------------------------------


def test_every_discovered_tool_is_in_the_actionable_audit(tmp_path):
    names = {item["name"] for item in _registry(tmp_path).discover()}
    assert names == set(AUDITED_TOOLS), (
        f"unlisted: {sorted(names - set(AUDITED_TOOLS))}; "
        f"stale: {sorted(set(AUDITED_TOOLS) - names)}"
    )


# ---------------------------------------------------------------------------
# Already-compliant: the 500-char head already holds the decision.
# ---------------------------------------------------------------------------


def test_campaign_guard_actionable_state_survives_budget(tmp_path):
    result = _registry(tmp_path).invoke("campaign.guard", {"expected_gb": 1.0})
    assert result.ok, result.error
    head = _head(result.value)
    assert '"state"' in head
    assert result.value["state"] in head
    assert '"swapfiles"' in head


def test_odyssey_ledger_owed_names_survive_budget(tmp_path):
    axes = ("anatomy", "ebpw", "gpu", "cpu", "tps", "nr_candidate", "nx_disposition")
    owed = {
        a: {"state": "OWED", "value": None, "reason": None, "receipt": None} for a in axes
    }
    specimens = [
        {"slug": f"owed--body{i:02d}", "gib": float(i + 1), "family": "f", "class": "DENSE",
         "axes": dict(owed)}
        for i in range(16)
    ]
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"schema": "odyssey-ledger-1", "specimens": specimens}))
    result = _registry(tmp_path).invoke("odyssey.ledger", {"path": str(path), "owed_only": True})
    assert result.ok, result.error
    text = _dump(result.value)
    head = text[:BUDGET]
    assert '"owed"' in head
    assert head.count('"slug"') >= 2
    assert result.value["n_owed"] == 16
    assert result.value["shown"] <= 12
    assert len(text) < 4000


def test_fs_list_filenames_survive_budget(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    for i in range(30):
        (workspace / f"file_{i:02d}.txt").write_text("x")
    result = _registry(workspace).invoke("fs.list", {"path": str(workspace)})
    assert result.ok, result.error
    head = _head(result.value)
    assert "file_00.txt" in head
    assert "file_01.txt" in head


def test_tests_list_paths_and_count_survive_budget(tmp_path):
    result = _registry(tmp_path).invoke("tests.list", {})
    assert result.ok, result.error
    head = _head(result.value)
    assert '"count"' in head
    assert "test_" in head
    assert head.count("test_") >= 2


def test_frontier_decide_action_survives_budget(tmp_path):
    result = _registry(tmp_path).invoke("frontier.decide", {})
    assert result.ok, result.error
    head = _head(result.value)
    assert '"action"' in head
    assert result.value["action"] in head


def test_git_status_fits_or_stdout_survives(tmp_path):
    result = _registry(tmp_path).invoke("git.status", {})
    assert result.ok, result.error
    head = _head(result.value)
    assert '"stdout"' in head or '"returncode"' in head
    stdout = result.value.get("stdout") or ""
    if stdout.startswith("## "):
        branch = stdout.splitlines()[0].replace("## ", "").split()[0]
        assert branch[:40] in head


def test_git_checkout_safe_refusal_survives_budget(tmp_path):
    result = _registry(tmp_path).invoke("git.checkout-safe", {})
    assert result.ok, result.error
    head = _head(result.value)
    assert result.value["status"] in head
    assert "REFUSED" in head


def test_lake_census_single_slug_fits(tmp_path):
    body = tmp_path / "org--one@rev"
    body.mkdir()
    _write_safetensors(
        body / "m.safetensors",
        {"model.layers.0.self_attn.q_proj.weight":
         {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]}},
    )
    (body / "config.json").write_text('{"hidden_size": 4}')
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({
        "specimens": [{"slug": body.name, "path": str(body), "bytes": 100, "architecture_family": "qwen"}],
    }))
    result = _registry(tmp_path).invoke("lake.census", {"catalog": str(catalog), "slug": body.name})
    assert result.ok, result.error
    head = _head(result.value)
    assert '"klass"' in head
    assert result.value["klass"] in head
    assert body.name in _dump(result.value)


def test_odyssey_anatomy_refusal_survives_budget(tmp_path):
    snap = tmp_path / "nosft"
    snap.mkdir()
    (snap / "pytorch_model.bin").write_bytes(b"\0")
    result = _registry(tmp_path).invoke("odyssey.anatomy", {"snapshot": str(snap)})
    assert result.ok, result.error
    head = _head(result.value)
    assert result.value["anatomy"] is None
    assert "refused" in head
    assert "safetensors" in head


def test_processes_summary_rollup_survives_budget(tmp_path, monkeypatch):
    from hcli import processes

    monkeypatch.setattr(processes, "summary", lambda **_k: {
        "count": 15,
        "total_rss_bytes": 15 * 10**9,
        "by_class": {"model": 3, "tool": 12},
        "roles": {"resident": 1},
        "processes": [{"pid": 1000 + i, "command": "x" * 200} for i in range(15)],
    })
    result = _registry(tmp_path).invoke("processes.summary", {})
    assert result.ok, result.error
    head = _head(result.value)
    assert '"count"' in head
    assert "15" in head
    assert "total_rss_bytes" in head


# ---------------------------------------------------------------------------
# RED (before the fix): actionable names sit past the 500-char cut.
# ---------------------------------------------------------------------------


def _census_catalog(tmp_path: Path, n: int = 20) -> Path:
    specimens = []
    for i in range(n):
        slug = f"org--model{i:02d}@abcd{i:04d}"
        body = tmp_path / "lake" / slug
        body.mkdir(parents=True)
        _write_safetensors(
            body / "m.safetensors",
            {"model.layers.0.self_attn.q_proj.weight":
             {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]}},
        )
        (body / "config.json").write_text('{"hidden_size": 4}')
        specimens.append({
            "slug": slug, "path": str(body),
            "bytes": 10**9 * (n - i), "architecture_family": "qwen",
        })
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"specimens": specimens}))
    return catalog


def test_lake_census_slugs_survive_budget(tmp_path):
    """Full census put n/tally first and slug last inside each row, so a 500-char
    head named one body and hid the rest of the lake."""
    catalog = _census_catalog(tmp_path)
    result = _registry(tmp_path).invoke("lake.census", {"catalog": str(catalog)})
    assert result.ok, result.error
    head = _head(result.value)
    named = [f"org--model{i:02d}@abcd{i:04d}" for i in range(8) if f"org--model{i:02d}@abcd{i:04d}" in head]
    assert len(named) >= 2, f"only {named} specimen(s) survive the 500-char head: {head!r}"
    assert result.value["n"] == 20
    assert "shown" in result.value
    assert result.value["n"] >= result.value["shown"]
    assert len(_dump(result.value)) < 4000


def test_specimens_registry_ids_survive_budget(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    (lake / "specimens").mkdir(parents=True)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    monkeypatch.setenv("HCLI_MODEL_LAKE_ROOT", str(lake))
    monkeypatch.setenv("HCLI_SPECIMEN_MANIFEST_DIR", str(manifests))
    ids = []
    for i in range(16):
        name = f"Org--Model-{i:02d}@rev{i:04d}abcd"
        d = lake / "specimens" / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "model_type": "qwen2", "hidden_size": 4096, "num_hidden_layers": 32,
            "architectures": ["Qwen2ForCausalLM"],
        }))
        (d / "model.safetensors").write_bytes(b"x" * 64)
        ids.append(name)
    result = _registry(tmp_path).invoke("specimens.registry", {})
    assert result.ok, result.error
    head = _head(result.value)
    surviving = [name for name in ids[:6] if name in head]
    assert len(surviving) >= 2, f"only {surviving} id(s) in the 500-char head: {head!r}"
    assert result.value["n_specimens"] == 16
    assert result.value["shown"] <= 12
    assert result.value["n_specimens"] >= result.value["shown"]


def test_tools_catalog_names_survive_budget(tmp_path):
    result = _registry(tmp_path).invoke("tools.catalog", {"focus": "odyssey ledger"})
    assert result.ok, result.error
    names = [item["name"] for item in result.value["matches"][:6]]
    assert names, "catalog returned no matches"
    head = _head(result.value)
    surviving = [name for name in names if name in head]
    assert len(surviving) >= 2, f"only {surviving} tool name(s) in the 500-char head: {head!r}"
    assert "names" in result.value
    assert result.value["names"][0] in head


def test_processes_list_pids_survive_budget(tmp_path, monkeypatch):
    from hcli import processes

    class _Proc:
        def __init__(self, pid):
            self.pid = pid
            self.ppid = 1
            self.rss_bytes = 10**9
            self.cpu_percent = 1.2
            self.elapsed = "01:02:03"
            self.role = "resident"
            self.process_class = "model"
            self.safe_to_stop = False
            self.purpose = "resident body"
            self.command = "/usr/bin/python3 -m hcli.agentos.resident --workspace " + "x" * 220
            self.body = "qwen-long-body-name"
            self.memory_source = "rss"

        def to_dict(self):
            return {
                "pid": self.pid, "ppid": self.ppid, "rss_bytes": self.rss_bytes,
                "rss_gib": 0.93, "memory_source": self.memory_source,
                "cpu_percent": self.cpu_percent, "elapsed": self.elapsed,
                "role": self.role, "class": self.process_class,
                "safe_to_stop": self.safe_to_stop, "purpose": self.purpose,
                "command": self.command, "body": self.body,
            }

    procs = [_Proc(41000 + i) for i in range(15)]
    monkeypatch.setattr(processes, "live_processes", lambda **_k: procs)
    result = _registry(tmp_path).invoke("processes.list", {})
    assert result.ok, result.error
    head = _head(result.value)
    pids = [p.pid for p in procs[:6]]
    surviving = [pid for pid in pids if str(pid) in head]
    assert len(surviving) >= 2, f"only {surviving} pid(s) in the 500-char head: {head!r}"
    # Full rows stay present so existing shape tests keep matching to_dict().
    assert {row["pid"] for row in result.value["processes"]} == {p.pid for p in procs}


def test_processes_orphaned_pids_survive_budget(tmp_path, monkeypatch):
    from hcli import processes

    class _Proc:
        def __init__(self, pid):
            self.pid = pid
            self.ppid = 1
            self.rss_bytes = 10**9
            self.cpu_percent = 0.0
            self.elapsed = "3-00:00:00"
            self.role = "orphan"
            self.process_class = "model"
            self.safe_to_stop = True
            self.purpose = "orphaned resident body"
            self.command = "/opt/homebrew/bin/python3.12 -m mlx_lm.server --model " + "y" * 220
            self.body = "orphaned-qwen-body"
            self.memory_source = "rss"

        def to_dict(self):
            return {
                "pid": self.pid, "ppid": self.ppid, "rss_bytes": self.rss_bytes,
                "rss_gib": 0.93, "memory_source": self.memory_source,
                "cpu_percent": self.cpu_percent, "elapsed": self.elapsed,
                "role": self.role, "class": self.process_class,
                "safe_to_stop": self.safe_to_stop, "purpose": self.purpose,
                "command": self.command, "body": self.body,
            }

    orphans = [_Proc(52000 + i) for i in range(5)]
    monkeypatch.setattr(processes, "orphaned_resident_bodies", lambda **_k: orphans)
    result = _registry(tmp_path).invoke("processes.orphaned", {})
    assert result.ok, result.error
    head = _head(result.value)
    surviving = [p.pid for p in orphans if str(p.pid) in head]
    assert len(surviving) >= 2, f"only {surviving} pid(s) in the 500-char head: {head!r}"
    assert isinstance(result.value.get("orphaned"), list)


def test_acquisition_propose_recommended_survives_budget(tmp_path, monkeypatch):
    ranked = [
        {
            "oxx": f"O{i:03d}", "model": f"model-{i}", "est_gib": 10.0 + i,
            "fits_disk": True, "odyssey_value": 1.0, "partial_elsewhere": None,
            "role": {"arch_objective": "x" * 80},
        }
        for i in range(12)
    ]
    payload = {
        "schema": "hcli.acquisition.propose.v1",
        "disk": {"free_gib": 100, "mount": "/Volumes/corpdrive", "total_gib": 8000},
        "disk_safety_headroom_gib": 64,
        "network_slots_in_flight": {"count": 0, "commands": []},
        "curriculum": {"moe": 3, "dense": 4},
        "partials_in_progress": [
            {"tag": "partial-tag-" + "z" * 40, "bytes_on_disk": 1, "files": 2, "missing": 3}
        ] * 5,
        "already_acquired": [
            {"oxx": "O000", "model": "m", "repo": "a/b", "where": "modellake specimens",
             "tag": "t", "stale_claim": None, "_evidence": "MEASURED"}
        ] * 8,
        "blocked": [{"oxx": "O001", "reason": "RETIRED"}],
        "ranked": ranked,
        "recommended": ranked[2],
        "recommendation_reason": "O002 (model-2) ranks highest by Odyssey info-value",
        "list_order_pick": {"oxx": "O010", "repo": "z/z"},
        "list_order_would_redownload_sealed": True,
    }
    fake = types.ModuleType("hcli.acquisition")
    fake.propose = lambda: payload
    monkeypatch.setitem(sys.modules, "hcli.acquisition", fake)
    result = _registry(tmp_path).invoke("acquisition.propose", {})
    assert result.ok, result.error
    head = _head(result.value)
    assert "O002" in head, f"recommended oxx is past the 500-char cut: {head!r}"
    assert "recommended" in head


def test_git_log_commit_hashes_survive_budget(tmp_path):
    result = _registry(tmp_path).invoke("git.log", {"limit": 10})
    assert result.ok, result.error
    hashes = [
        line.split()[0]
        for line in (result.value.get("stdout") or "").splitlines()
        if line.strip()
    ][:6]
    assert len(hashes) >= 2, "fixture needs at least two commits"
    head = _head(result.value)
    surviving = [h for h in hashes if h in head]
    assert len(surviving) >= 2, f"only {surviving} hash(es) in the 500-char head: {head!r}"


def test_tests_run_verified_survives_budget(tmp_path, monkeypatch):
    import hcli.tool_registry as tr

    huge = "PASSED " + ("x" * 4000)

    def fake_run(argv, *, cwd, timeout=30.0):
        return {
            "argv": list(argv),
            "cwd": str(cwd),
            "returncode": 0,
            "stdout": huge,
            "stderr": "",
        }

    monkeypatch.setattr(tr, "_run_readonly", fake_run)
    result = _registry(tmp_path).invoke(
        "tests.run",
        {"runner": "pytest", "root": str(tmp_path), "paths": [], "timeout_s": 5},
    )
    assert result.ok, result.error
    head = _head(result.value)
    assert '"verified"' in head, f"verified is past the 500-char cut: {head!r}"
    assert "true" in head.lower() or result.value["verified"] is True


def test_odyssey_anatomy_hypotheses_survive_budget(tmp_path, monkeypatch):
    import hcli.tool_registry as tr

    class _Unavailable(Exception):
        pass

    def fake_anatomy(snapshot, layer=0):
        return {
            "snapshot": snapshot,
            "layer": 0,
            "layers_with_experts": [0, 30],
            "shared_expert_present": False,
            "scheme": "E",
            "n_shards": 40,
            "n_tensors": 8000,
            "storage": "per-expert",
            "cross_expert": [
                {"tensor": "w1", "spectrum": [0.01] * 80, "n_experts_total": 256},
                {"tensor": "w2", "spectrum": [0.02] * 80, "n_experts_total": 256},
                {"tensor": "w3", "spectrum": [0.03] * 80, "n_experts_total": 256},
            ],
            "within_expert": {"rank": 12, "detail": "y" * 400},
            "hypotheses": [
                {"name": "shared-basis-NR", "verdict": "LIVE"},
                {"name": "lowrank", "verdict": "DEAD"},
            ],
        }

    fake = types.SimpleNamespace(
        AnatomyUnavailable=_Unavailable,
        anatomy_from_safetensors=fake_anatomy,
    )
    monkeypatch.setattr(tr, "_future", lambda _name: fake)
    snap = tmp_path / "body"
    snap.mkdir()
    result = _registry(tmp_path).invoke("odyssey.anatomy", {"snapshot": str(snap)})
    assert result.ok, result.error
    head = _head(result.value)
    assert "shared-basis-NR" in head, f"hypotheses are past the 500-char cut: {head!r}"
    assert "hypotheses" in head
