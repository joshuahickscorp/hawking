"""The campaign tools must be REACHABLE from HCLI, not merely written.

Six modules under tools/future -- the lake census, the Odyssey ledger, the
spectral null, the autonomy ratio, the anatomy and the memory guard -- had ZERO
non-test references anywhere in hcli/. HCLI could select a specimen and report
its queue, but could not measure a single one of the seven axes G034 requires,
so every measurement had to come from the supervisor.

This asserts CALL SITES, not registrations. A tool that discover() lists but
invoke() cannot run is a catalogue entry.
"""
import json, pathlib, sys, tempfile
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hcli.tool_registry import default_tool_registry  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
LAKE = pathlib.Path("/Volumes/corpdrive/hawking-modellake/specimens")
CAMPAIGN_TOOLS = ("lake.census", "odyssey.ledger", "odyssey.anatomy", "campaign.guard")


@pytest.fixture
def registry():
    return default_tool_registry(pathlib.Path(tempfile.mkdtemp()), repo_root=REPO)


def test_every_campaign_tool_is_discoverable(registry):
    names = {t["name"] if isinstance(t, dict) else getattr(t, "name") for t in registry.discover()}
    missing = [t for t in CAMPAIGN_TOOLS if t not in names]
    assert not missing, f"not registered: {missing}"


def test_every_campaign_tool_actually_runs(registry):
    """Registration is not reachability. Invoke each one."""
    r = registry.invoke("campaign.guard", {"expected_gb": 1.0})
    assert r.ok, r.error
    assert r.value["state"] in {"OK", "WARN", "STOP"}
    assert r.value["swapfiles"] >= 0, "the guard is blind to swap again"

    r = registry.invoke("odyssey.ledger", {})
    assert r.ok, r.error
    p = r.value["progress"]
    assert p["specimens"] == 56
    assert p["axes_total"] == p["specimens"] * 7
    assert p["axes_resolved"] + p["axes_owed"] == p["axes_total"]


def test_the_ledger_can_name_what_is_still_owed(registry, tmp_path):
    """HCLI must be able to choose its next target from disk, unaided.

    Built on a synthetic ledger with ONE COMPLETE specimen, because every real
    specimen currently owes something -- so against live data a filter that
    returns everything is indistinguishable from one that filters correctly.
    """
    axes = ("anatomy", "ebpw", "gpu", "cpu", "tps", "nr_candidate", "nx_disposition")
    done = {"slug": "done--body", "gib": 1.0, "family": "f", "class": "DENSE",
            "axes": {a: {"state": "MEASURED", "value": 1, "reason": None,
                         "receipt": "r.json"} for a in axes}}
    partial = {"slug": "partial--body", "gib": 2.0, "family": "f", "class": "DENSE",
               "axes": {a: {"state": "OWED", "value": None, "reason": None,
                            "receipt": None} for a in axes}}
    partial["axes"]["ebpw"] = {"state": "MEASURED", "value": 16.0, "reason": None,
                               "receipt": "r.json"}
    led = tmp_path / "l.json"
    led.write_text(json.dumps({"schema": "odyssey-ledger-1",
                               "specimens": [done, partial]}))

    r = registry.invoke("odyssey.ledger", {"path": str(led), "owed_only": True})
    assert r.ok, r.error
    slugs = [x["slug"] for x in r.value["owed"]]
    assert slugs == ["partial--body"], f"a complete specimen was offered as work: {slugs}"
    assert "ebpw" not in r.value["owed"][0]["owed"], "a measured axis was reported as owed"
    assert r.value["progress"]["specimens_complete"] == 1

    live = registry.invoke("odyssey.ledger", {"owed_only": True})
    assert live.ok, live.error
    assert live.value["owed"], "nothing owed, yet no specimen is complete"


def test_anatomy_refuses_with_a_mechanism_rather_than_an_empty_result(registry):
    """The refusal must survive the tool boundary, not become a blank."""
    if not LAKE.is_dir():
        pytest.skip("lake not mounted -- THIS ASSERTION DID NOT RUN")
    body = LAKE / "arcinstitute--evo2_40b@d529aa57c307"
    if not body.is_dir():
        pytest.skip(f"{body.name} absent -- THIS ASSERTION DID NOT RUN")
    r = registry.invoke("odyssey.anatomy", {"snapshot": str(body)})
    assert r.ok, r.error
    assert r.value["anatomy"] is None
    assert "no .safetensors" in r.value["refused"]


def test_census_reads_a_prequantized_body_correctly(registry):
    if not LAKE.is_dir():
        pytest.skip("lake not mounted -- THIS ASSERTION DID NOT RUN")
    slug = "moonshotai--Kimi-K3@9f62e4e9fffb"
    if not (LAKE / slug).is_dir():
        pytest.skip(f"{slug} absent -- THIS ASSERTION DID NOT RUN")
    r = registry.invoke("lake.census", {"slug": slug})
    assert r.ok, r.error
    assert r.value["klass"] == "MOE-PREQUANTIZED"
    assert r.value["complete_ebpw"] < 5.0, r.value
    assert r.value["pack_factor"] == 2, "the mxfp4 packing factor was lost"


def test_a_missing_future_module_raises_rather_than_returning_nothing(registry):
    """A tool naming a module that does not exist must say so."""
    from hcli import tool_registry as tr
    with pytest.raises(FileNotFoundError, match="unreachable"):
        tr._future("a_module_that_does_not_exist")
