"""NOETIC_ASCENSION_HANDOFF: every field cites a receipt; the loop was run.

pytest tools/headless -q must exit 0. These tests do not import hcli at
collection time (this worktree is sparse). They read the receipts the loop
and builder wrote. A session fixture builds them if they are missing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HANDOFF = REPO / "receipts" / "headless" / "NOETIC_ASCENSION_HANDOFF.json"
LOOP = REPO / "receipts" / "headless" / "NOETIC_ASCENSION_LOOP.json"
SCIENCE = REPO / "receipts" / "headless" / "NOETIC_ASCENSION_SCIENCE_UPDATE.json"

REQUIRED_SECTIONS = (
    "parent",
    "strongest_candidate",
    "closure",
    "families_tested",
    "organ_results",
    "route_metrics",
    "density_metrics",
    "accounting_metrics",
    "kernels",
    "composition",
    "capability",
    "production_performance",
    "negative_science",
    "open_hypotheses",
    "grok_lane_results",
    "next_agentos_workunits",
    "rollback_command",
    "canonical_hcli_launch",
    "demonstration",
    "claims",
)

REQUIRED_STAGES = (
    "HCLI",
    "AgentOS",
    "resident",
    "Doctor",
    "Gravity experiment",
    "tools",
    "verifier",
    "accepted or rejected",
    "updated science",
)

SOURCE_RECEIPTS = (
    "NO_CANDIDATE_YET_BEATS_PARENT.json",
    "FIRST_NOETIC_EXECUTABLE.json",
    "NOETIC_Q3_MLP_Q4_ATTN.json",
    "AFFINE2_NATIVE_MLP.json",
    "NOETIC_COMPOSITION.json",
    "FRACTIONAL_BIT_CANON.json",
    "DENSE_SUBBIT_TRANSFER.json",
    "ATTENTION_FLOOR_REFIT.json",
    "DOCTOR_V2_PRESCRIPTION.json",
    "GRAVITY_COMPILER_SEARCH.json",
    "NOETIC_IR.json",
    "NOETIC_CLOSURE.json",
    "NOETIC_ZERO_PARENT.json",
    "NOETIC_NATIVE_OPERATOR.json",
    "CONVENTIONAL_CONTROL_SET.json",
    "CORE_AUTHORITIES.json",
    "CODE_ENTROPY.json",
    "NAMESPACE_MIGRATION.json",
    "ARCHITECTURE_CANON.json",
)


@pytest.fixture(scope="session")
def handoff() -> dict:
    if not LOOP.is_file():
        proc = subprocess.run(
            [sys.executable, str(HERE / "noetic_ascension_loop.py")],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    # Other headless tests rewrite live receipts (NOETIC_CLOSURE, NOETIC_IR,
    # GRAVITY_COMPILER_SEARCH, …) before this file is collected. Re-assemble
    # the handoff from the receipts as they stand now so reproducing commands
    # that pin a field value are not stale against a rewrite that already
    # happened in this pytest process.
    sys.path.insert(0, str(HERE))
    from noetic_ascension_handoff import assemble_and_write  # noqa: WPS433

    assemble_and_write()
    assert HANDOFF.is_file(), f"missing {HANDOFF}"
    doc = json.loads(HANDOFF.read_text())
    assert doc["schema"] == "hawking.headless.noetic_ascension_handoff.v1"
    return doc


@pytest.fixture(scope="session")
def loop_doc(handoff: dict) -> dict:  # noqa: ARG001
    assert LOOP.is_file(), f"missing {LOOP}"
    return json.loads(LOOP.read_text())


def test_required_sections_present(handoff: dict):
    for key in REQUIRED_SECTIONS:
        assert key in handoff, f"missing section {key}"
        assert handoff[key] not in (None, "", [], {}), key


def test_every_section_cites_a_receipt_and_a_command(handoff: dict):
    for key in REQUIRED_SECTIONS:
        if key in ("claims", "demonstration"):
            continue
        section = handoff[key]
        assert isinstance(section, dict), key
        assert section.get("evidence_path"), key
        cmd = section.get("reproducing_command")
        assert isinstance(cmd, list) and cmd and all(isinstance(x, str) for x in cmd), key


def test_every_claim_has_a_reproducing_command(handoff: dict):
    claims = handoff["claims"]
    assert len(claims) >= 10
    for c in claims:
        assert c.get("id")
        assert c.get("kind")
        assert c.get("statement")
        assert c.get("evidence_path")
        cmd = c.get("reproducing_command")
        assert isinstance(cmd, list) and cmd, c.get("id")
        assert all(isinstance(x, str) for x in cmd), c.get("id")


def test_builder_refuses_a_claim_with_no_command():
    proc = subprocess.run(
        [sys.executable, str(HERE / "noetic_ascension_handoff.py"), "--refuse-only"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "REFUSED (no reproducing_command)" in proc.stdout
    assert "VACUOUS" not in proc.stdout


def test_verdict_is_the_honest_negative(handoff: dict):
    # Pinning this to one literal makes the handoff lie the moment the evidence
    # moves. NOETIC_FUSED_SUBBIT fired the reopen condition the negative itself
    # named, so both states are legitimate — what must never happen is claiming
    # the flat negative while a candidate leads, or claiming a promotion.
    valid = {
        "NO_CANDIDATE_YET_BEATS_PARENT",
        "REOPENED_CANDIDATE_LEADS_BUT_QUALIFICATION_NOT_RUN",
    }
    assert handoff["verdict"] in valid, handoff["verdict"]
    assert handoff.get("did_not_promote") is not False
    assert handoff["gate"] == "REJECTED"
    assert handoff["did_not_promote"] is True
    parent = json.loads(
        (REPO / "receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json").read_text()
    )
    assert handoff["verdict"] == parent["verdict"]
    reading = parent["blocker"]["measured"]["reading"]
    assert reading == handoff["density_metrics"]["value"]["reading"]
    assert "44.7%" in reading
    assert "1.9%" in reading
    body = json.dumps(handoff).lower()
    assert "promoted" not in body or "no promotion" in json.dumps(handoff).lower() or "did_not_promote" in body


def test_strongest_candidate_is_not_a_win(handoff: dict):
    best = handoff["strongest_candidate"]["value"]
    fail = handoff["strongest_candidate"].get("qualification_failure") or {}
    assert best["id"]
    assert "no throughput win" in (fail.get("reason") or "")


def test_dispatch_count_is_964_either_way(handoff: dict):
    dens = handoff["density_metrics"]["value"]
    assert dens["source_dispatch_count"] == 964
    assert dens["executable_dispatch_count"] == 964
    kern = handoff["kernels"]["value"]
    assert kern["production_dispatches_per_token"] == 964


def test_next_family_is_non_matvec(handoff: dict):
    nxt = handoff["open_hypotheses"]["value"]
    fam = next(x for x in nxt if x["id"] == "H-NON-MATVEC")
    assert fam["value"]["family"] == "non-matvec operator that survives composition"


def test_loop_ran_every_required_stage(loop_doc: dict):
    assert loop_doc["schema"] == "hawking.headless.noetic_ascension_loop.v1"
    assert loop_doc["gate"] == "REJECTED"
    assert loop_doc["did_not_load_second_27b"] is True
    assert (loop_doc.get("mission") or {}).get("status") == "completed"
    stages = {s.get("stage") for s in loop_doc.get("steps") or []}
    for need in REQUIRED_STAGES:
        assert need in stages, (need, stages)
    for s in loop_doc.get("steps") or []:
        assert s.get("command_pretty") or s.get("command"), s.get("name")
        if s.get("status") == "NOT_RUN":
            assert s.get("reason"), f"NOT_RUN without reason: {s.get('name')}"


def test_science_was_updated():
    assert SCIENCE.is_file()
    d = json.loads(SCIENCE.read_text())
    assert d["schema"] == "hawking.headless.noetic_ascension_science_update.v1"
    assert d["gate"] == "REJECTED"
    assert d["did_not_promote"] is True
    assert d["verdict"] == "NO_CANDIDATE_YET_BEATS_PARENT"


def test_handoff_cites_the_named_source_receipts(handoff: dict):
    blob = json.dumps(handoff)
    for name in SOURCE_RECEIPTS:
        assert name in blob, name


def test_reproducing_commands_exit_zero(handoff: dict):
    """Cold-read: every claim's command still exits 0 from the repo root."""
    failures = []
    for c in handoff["claims"]:
        cmd = c["reproducing_command"]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO),
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            failures.append((c["id"], "timeout", ""))
            continue
        if proc.returncode != 0:
            failures.append(
                (c["id"], proc.returncode, (proc.stderr or proc.stdout or "")[-500:])
            )
    assert not failures, failures


def test_section_reproducing_commands_exit_zero(handoff: dict):
    failures = []
    for key in REQUIRED_SECTIONS:
        if key in ("claims", "demonstration"):
            continue
        cmd = handoff[key]["reproducing_command"]
        proc = subprocess.run(
            cmd, cwd=str(REPO), capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            failures.append(
                (key, proc.returncode, (proc.stderr or proc.stdout or "")[-500:])
            )
    assert not failures, failures


def test_canonical_launch_is_python3_m_hcli(handoff: dict):
    assert handoff["canonical_hcli_launch"]["value"] == "python3 -m hcli"


def test_rollback_does_not_delete_historical_receipts(handoff: dict):
    rb = handoff["rollback_command"]["value"]
    assert "NOETIC_ASCENSION_HANDOFF.json" in rb["command"]
    assert "receipts/" in (rb.get("never_delete") or {})
    assert "hcli/" in rb["does_not_touch"]


def test_did_not_load_second_27b(handoff: dict, loop_doc: dict):
    assert handoff["did_not_load_second_27b"] is True
    assert loop_doc["did_not_load_second_27b"] is True
    for s in loop_doc.get("steps") or []:
        produced = s.get("produced") or {}
        if isinstance(produced, dict) and "did_not_load_second_27b" in produced:
            assert produced["did_not_load_second_27b"] is True, s.get("name")


def test_honesty_is_not_a_success_story(handoff: dict):
    text = handoff.get("honesty") or ""
    assert "No Noetic candidate beats the parent" in text
    assert "success" in text.lower()  # the warning against a success story
    assert handoff["gate"] == "REJECTED"
