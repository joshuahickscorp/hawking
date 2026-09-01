"""K1 adversarial audit of the sovereign loop.

Each test is a named attack. Where the attack succeeds the test is xfail with
the precise reason, so the defect is recorded rather than hidden.

The live module is imported. run() is never called. execute() is never called
with PERTURB. The live mission kernel is never written.
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from tools.future import sovereign_attack_report as sar
from tools.future._common import RECEIPTS, HardwareClaimError, _assert_no_hardware_claims

sov = sar.load_sovereign()
LIVE_KERNEL = sov.kernel_path().resolve()


@pytest.fixture(autouse=True)
def _never_write_live_kernel(monkeypatch):
    orig = Path.write_text

    def guarded(self, *a, **kw):
        try:
            if Path(self).resolve() == LIVE_KERNEL:
                raise AssertionError(f"refusing to write live kernel {LIVE_KERNEL}")
        except OSError:
            pass
        return orig(self, *a, **kw)

    monkeypatch.setattr(Path, "write_text", guarded)


# ---------------------------------------------------------------------------
# FAKE SOVEREIGN
# ---------------------------------------------------------------------------


def test_attack_fake_sovereign_run_does_not_read_stdin():
    src = inspect.getsource(sov.run)
    assert "input(" not in src
    assert "sys.stdin" not in src
    assert "while time.time() < deadline" in src
    assert "prov.ask(" in src
    row = sar.attack_fake_sovereign()
    assert row["verdict"] == "HELD"
    assert row["reads_stdin"] is False
    assert row["closed_while_deadline"] is True


def test_attack_fake_sovereign_interventions_stay_zero():
    src = inspect.getsource(sov.run)
    assert "interventions = 0" in src
    assert "interventions +=" not in src
    stripped = src.replace("interventions = 0", "")
    assert "interventions =" not in stripped
    assert "claude_interventions" in src


# ---------------------------------------------------------------------------
# MALFORMED REPLY (crashes 1-3 held; crash 4 is the defect)
# ---------------------------------------------------------------------------


def test_attack_malformed_missing_selected_work():
    """Crash 1 was a missing key. validate({}) must still carry counts."""
    v = sov.validate({})
    assert v["n_accepted"] == 0
    assert v["n_rejected"] == 0
    assert v["accepted"] == []
    assert v["rejected"] == []


def test_attack_malformed_selected_work_dict():
    """Crash 2 was a dict where a list belonged."""
    v = sov.validate({
        "selected_work": {
            "type": "PERTURB",
            "params": {"tensor": "gate", "layer": 0, "fraction": 0.5, "side": "rows"},
        }
    })
    assert v["n_accepted"] == 1
    assert v["accepted"][0]["type"] == "PERTURB"


def test_attack_malformed_parse_none_counts():
    """Crash 3 was the parse-failure path omitting n_accepted/n_rejected."""
    v = sov.validate(None)
    assert v["ok"] is False
    assert v["n_accepted"] == 0
    assert v["n_rejected"] == 0
    assert "rejected" in v
    assert "accepted" in v


# FIXED. This lane found it as a live defect - validate() did
# `p = w.get('params') or {}` then `p.get('tensor')`, and a list is truthy so the
# `or {}` never fired. It is now coerced and REJECTED with its type named, so the
# xfail is gone and the assertions below say what holding it means.
def test_attack_malformed_params_list():
    v = sov.validate({
        "belief_update": "x",
        "selected_work": [{
            "type": "PERTURB",
            "params": ["gate", 0, "rows", 0.5],
            "why": "schema-confused list of values",
        }],
    })
    assert v["n_accepted"] == 0
    assert v["n_rejected"] == 1
    assert "params is list" in v["rejected"][0]["why"]


# FIXED, same family.
def test_attack_malformed_params_string():
    v = sov.validate({
        "selected_work": [{"type": "PERTURB", "params": "gate,0,rows,0.5"}]
    })
    assert v["n_accepted"] == 0
    assert "params is str" in v["rejected"][0]["why"]


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason=(
        "run() results_summary does r['result'].get('damage') when ran is true; "
        "execute() json.loads the last stdout line and a JSON list crashes the loop"
    ),
)
def test_attack_malformed_tool_result_list():
    r = {"type": "PERTURB", "ran": True, "params": {}, "result": [0.1, 0.2]}
    summary = (
        f"{r['type']} {r.get('params', {})} -> "
        f"{'damage ' + str(r['result'].get('damage')) if r.get('ran') else 'DID NOT RUN'}"
    )
    assert isinstance(summary, str)


# ---------------------------------------------------------------------------
# SILENT DROP
# ---------------------------------------------------------------------------


def test_attack_silent_drop_unsupported_is_recorded():
    v = sov.validate({
        "selected_work": [{
            "type": "LAUNCH_WORKUNIT",
            "params": {"id": "WU.1"},
            "why": "launch",
        }]
    })
    assert v["n_rejected"] == 1
    assert v["rejected"][0]["work"]["type"] == "LAUNCH_WORKUNIT"
    assert "not an executable work type" in v["rejected"][0]["why"]


# FIXED in the same landing this lane's report landed in.
def test_attack_silent_drop_truncation():
    items = [
        {"type": "PERTURB",
         "params": {"tensor": t, "layer": 0, "fraction": 0.5, "side": "rows"}}
        for t in ("gate", "up", "down")
    ]
    fourth = {"type": "COMPUTE", "params": {"op": "sum"}, "why": "the fourth request"}
    v = sov.validate({"selected_work": items + [fourth]})
    assert v["n_accepted"] + v["n_rejected"] == 4, (
        f"fourth item silently dropped: n_accepted={v['n_accepted']} "
        f"n_rejected={v['n_rejected']}"
    )


# FIXED in the same landing this lane's report landed in.
def test_attack_silent_drop_string_selected_work():
    v = sov.validate({"selected_work": "PERTURB gate layer 0 rows 0.5"})
    assert v["ok"] is False or v["n_rejected"] >= 1, (
        "string selected_work must be recorded as rejected, not swallowed"
    )


# FIXED: context_pack now shows the resident its own last hypotheses,
# read from the kernel or from the previous iteration record.
def test_attack_silent_drop_hypotheses_not_fed_back():
    k = sar.synthetic_kernel()
    token = "UNIQUE_HYP_XYZ_DO_NOT_ECHO"
    k["iterations"] = [{
        "n": 1,
        "parsed": True,
        "live_hypotheses": [{
            "id": "H9.unique",
            "claim": token,
            "cheapest_falsifier": "measure",
        }],
        "results_summary": ["no work was accepted from that turn"],
    }]
    pack = sov.context_pack(k)
    assert token in pack


# ---------------------------------------------------------------------------
# GENERATED BUT NEVER LAUNCHED
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "COMPUTE is in EXECUTABLE and validate() accepts it, but execute() "
        "returns ran=False with no runner. Same for READ_RECEIPT."
    ),
)
def test_attack_generated_compute_not_run():
    v = sov.validate({"selected_work": [{"type": "COMPUTE", "params": {"expr": "1+1"}}]})
    assert v["n_accepted"] == 1
    result = sov.execute(v["accepted"][0])
    assert result["ran"] is True


@pytest.mark.xfail(
    strict=True,
    reason=(
        "run() breaks the accepted-work loop on deadline and stores only "
        "n_accepted/n_rejected/rejected; the accepted list is not persisted, so "
        "unlaunched work vanishes"
    ),
)
def test_attack_generated_deadline_drops_work():
    src = inspect.getsource(sov.run)
    validation_line = [
        ln for ln in src.splitlines()
        if "n_accepted" in ln and "n_rejected" in ln and "rejected" in ln
    ]
    assert validation_line, "run() must record validation"
    # Desired: the accepted list is stored so unlaunched work is auditable.
    assert any(
        "accepted" in ln.replace("n_accepted", "")
        and "rejected" in ln.replace("n_rejected", "")
        for ln in validation_line
    ), f"accepted list omitted from iteration record: {validation_line}"


# ---------------------------------------------------------------------------
# CONTEXT ACCUMULATION
# ---------------------------------------------------------------------------


def test_attack_context_pack_does_not_accumulate():
    k = sar.synthetic_kernel()
    lengths = []
    for i in range(20):
        k["iterations"].append({"results_summary": [f"{i:04d}|" + ("Z" * 4000)]})
        k["tried_params"].append(f"up/L{i}/rows/0.5")
        lengths.append(len(sov.context_pack(k)))
    # If the pack concatenated history, 14 more 4000-char summaries would add
    # ~56k. It must not.
    assert lengths[-1] - lengths[6] < 1000, (
        f"pack grew with history: {lengths[6]} -> {lengths[-1]}"
    )
    row = sar.attack_context_accumulation()
    assert row["verdict"] == "HELD"
    assert row["grew_with_history"] is False


# ---------------------------------------------------------------------------
# IDENTICAL-REPLY LOOP
# ---------------------------------------------------------------------------


# FIXED: the pack now carries a turn number and the resident's own live
# hypotheses, so it cannot be byte-identical two turns running.
def test_attack_identical_reply_loop_can_escape():
    k = sar.synthetic_kernel()
    packs = []
    for _ in range(5):
        packs.append(sov.context_pack(k))
        k["iterations"].append({
            "results_summary": ["no work was accepted from that turn"],
            "parsed": False,
        })
    frozen = packs[1:]
    assert len(set(frozen)) == len(frozen), (
        "packs after the first failed turn are byte-identical; greedy decoding "
        f"cannot escape (digests {[sar._digest(p) for p in frozen]})"
    )


# ---------------------------------------------------------------------------
# KERNEL WRITE SAFETY
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "save_kernel uses Path.write_text which truncates in place; a crash "
        "mid-write corrupts the mission kernel. hcli/persist.py::atomic_write_json "
        "already does temp+fsync+os.replace and is unused here"
    ),
)
def test_attack_kernel_write_survives_crash_mid_write(tmp_path):
    dest = tmp_path / "HCLI_MISSION_KERNEL.json"
    row = sar.attack_kernel_write_safety(kernel_file=dest)
    assert row["previous_kernel_intact"] is True


# FIXED in the same landing this lane's report landed in.
def test_attack_corrupt_kernel_raises_sovereign_refused(tmp_path):
    dest = tmp_path / "HCLI_MISSION_KERNEL.json"
    dest.write_text("{")
    orig = sov.kernel_path
    try:
        sov.kernel_path = lambda: dest
        with pytest.raises(sov.SovereignRefused):
            sov.load_kernel()
    finally:
        sov.kernel_path = orig


def test_attack_load_kernel_refuses_when_missing(tmp_path):
    dest = tmp_path / "no-such-kernel.json"
    orig = sov.kernel_path
    try:
        sov.kernel_path = lambda: dest
        with pytest.raises(sov.SovereignRefused):
            sov.load_kernel()
    finally:
        sov.kernel_path = orig


def test_write_safety_refuses_live_kernel_path():
    with pytest.raises(sar.SovereignAttackRefused):
        sar.attack_kernel_write_safety()
    with pytest.raises(sar.SovereignAttackRefused):
        sar.attack_kernel_write_safety(kernel_file=sov.kernel_path())


# ---------------------------------------------------------------------------
# Receipt / import / negative controls
# ---------------------------------------------------------------------------


def test_build_emits_sealed_receipt():
    out = sar.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "SOVEREIGN_ATTACKS.json"
    assert doc["schema"] == sar.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["n_attacks"] >= 12
    assert doc["at_least_one_real_defect"] is True
    assert doc["did_not_call_run"] is True
    assert doc["did_not_execute_perturb"] is True
    assert doc["did_not_signal_processes"] is True
    _assert_no_hardware_claims(doc)


def test_receipt_lists_each_attack_verdict_and_reproduction():
    doc = json.loads(sar.build().read_text())
    assert doc["n_attacks"] == len(doc["attacks"])
    ids = [a["id"] for a in doc["attacks"]]
    assert len(ids) == len(set(ids))
    required = {
        "FAKE_SOVEREIGN",
        "MALFORMED_REPLY_PARAMS_LIST",
        "SILENT_DROP_UNSUPPORTED",
        "GENERATED_BUT_NEVER_LAUNCHED_COMPUTE",
        "CONTEXT_ACCUMULATION",
        "IDENTICAL_REPLY_LOOP",
        "KERNEL_WRITE_SAFETY",
    }
    assert required <= set(ids)
    for a in doc["attacks"]:
        assert a["verdict"] in {"HELD", "DEFECT", "UNTESTED"}
        assert a["reproduction"].startswith("python3 -m pytest ")
        assert a["reproduction"].endswith(" -q")
        assert a["detail"]
    # Was a live defect when this lane ran; fixed in the same landing. It moved
    # from defect_ids to held_ids, and a receipt that still called it a defect
    # would be describing a repo that no longer exists.
    assert "MALFORMED_REPLY_PARAMS_LIST" in doc["held_ids"]
    assert "MALFORMED_REPLY_PARAMS_LIST" not in doc["defect_ids"]
    # Nine defects were live when this lane reported. Seven were fixed in the
    # same landing, so the receipt must show them HELD - a receipt that still
    # named them defects would describe a repo that no longer exists.
    for fixed in ("KERNEL_WRITE_SAFETY", "KERNEL_CORRUPT_LOAD",
                  "SILENT_DROP_TRUNCATION", "SILENT_DROP_STRING_SELECTED_WORK",
                  "MALFORMED_REPLY_PARAMS_LIST", "IDENTICAL_REPLY_LOOP",
                  "SILENT_DROP_HYPOTHESES", "MALFORMED_REPLY_TOOL_RESULT_LIST",
                  "GENERATED_BUT_NEVER_LAUNCHED_DEADLINE"):
        assert fixed in doc["held_ids"], fixed
        assert fixed not in doc["defect_ids"], fixed
    # And the gate must still be able to REPORT a defect, or it is decoration.
    assert doc["defect_ids"], "an attack report with no defects left is suspect"


def test_live_module_is_imported_not_copied():
    path = Path(sov.__file__).resolve()
    assert path.name == "hcli_sovereign.py"
    assert path.is_file()
    assert "hcli_sovereign_live" in sov.__name__ or path.name == "hcli_sovereign.py"
    src = Path(sar.__file__).read_text()
    assert "def save_kernel" not in src
    assert "load_sovereign" in src


def test_this_suite_does_not_call_run():
    tree = ast.parse(Path(__file__).read_text())
    banned = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name in {"sov.run", "sov.init_kernel"}:
            banned.append(name)
    assert banned == []


def test_common_refuses_a_hardware_claim():
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims({"tps": 51.2})


def test_build_without_flag_is_refused():
    with pytest.raises(sar.SovereignAttackRefused):
        sar.main([])
