"""Resident-facing bridge into the live, mid-flight Odyssey-I driver.

Odyssey is real and running under ``tools/odyssey_ctl.py`` (~8k lines: queue,
patient packets, harvester, compiler-rule inference, run loop). O003 is
already SEALED. HCLI exposes a resident-facing Odyssey surface through this
module. This module is the connector, not a rewrite: every driver function
shells out to the existing driver's own subcommands. It adds nothing to the
curriculum and encodes none of it.

Two capability classes:

* **inspect** -- ``status``, ``queue``, ``value``, ``economics``,
  ``completions()``, ``patient()``. Read-only, run unconditionally, never
  write anything or spend a resource. ``admit_check`` was listed here on the
  ground that the CLI's ``admit`` verb only prints a memgate/worker_gate/disk
  decision. It does not: ``cmd_admit`` calls ``reclaim_if_tight()``, which
  runs ``tools/reclaim_safe.sh`` below ``DISK_WARN_GIB``, so it can delete
  build caches and finished lane directories. It is not registered, and the
  reason is recorded at its definition.

* **continue** -- ``harvest``, ``write_packet``, ``run``, ``cycle``,
  ``retire``, ``acquire_next``, ``completions(rebuild=True)``,
  ``gravity_gauntlet``. Every one of
  these can mutate persistent Odyssey state, spawn a subprocess, or start a
  download. Each requires ``confirm=True`` or raises ``PermissionError``,
  mirroring the gate ``benchmark.run``/``accelerator.benchmark`` already use
  in hcli/tool_registry.py (``if args.get("confirm") is not True: raise
  PermissionError(...)``). A ``dry_run=True`` path (the default, where the
  driver has one) needs no confirm because the driver itself launches
  nothing in that mode.

* **own** -- ``ingest``, ``add_to_eligibility``, ``park_specimen``,
  ``record_law``, ``record_scar``, ``create_transfer_probe``,
  ``create_adversarial_probe``. These give HCLI a campaign of its own
  instead of a read-only peek at ``tools/odyssey_ctl.py``'s. They do not
  encode a curriculum or choose which specimen matters -- they are the
  verbs HCLI calls once it has decided that for itself. All writes land in
  ``HCLI_LEDGER.json``, a file this module owns exclusively, alongside but
  never inside the driver's own ``ODYSSEY_STATE.json`` -- so nothing here
  can race the driver's own cycle/harvest/retire writers. ``ingest`` reads
  both (real on-disk state, never a fixture or a re-run) and is read-only;
  the other six mutate the ledger and gate on ``confirm=True`` like every
  verb in the *continue* class above, checked before the ledger is even
  loaded so a refusal touches no file. Odyssey II (transfer) and III
  (adversarial) probes require a law already recorded via ``record_law``
  -- that is the only ordering enforced, so I/II/III can overlap the
  moment a law is claimed, per the campaign's no-phase-barrier rule.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .persist import atomic_write_json

REPO = Path(__file__).resolve().parent.parent
ODYSSEY_CTL = REPO / "tools" / "odyssey_ctl.py"
ODYSSEY_DIR = REPO / "workspace" / "campaign" / "odyssey"
PATIENTS_DIR = ODYSSEY_DIR / "patients"
STATE_PATH = ODYSSEY_DIR / "ODYSSEY_STATE.json"  # driver-owned; read-only from here
LEDGER_PATH = ODYSSEY_DIR / "HCLI_LEDGER.json"  # HCLI-owned; this module is the only writer
_LEDGER_LISTS = ("eligible", "parked", "laws", "scars", "transfer_probes", "adversarial_probes")


def _run(args: list[str], timeout_s: float = 60.0) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(ODYSSEY_CTL), *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    stdout = proc.stdout
    if (
        args[:1] == ["acquire-next"]
        and "--dry-run" in args
        and proc.returncode == 1
        and proc.stdout.startswith("REFUSE")
    ):
        stdout += "DRY-RUN: admission refusal; no download launched.\n"
    return {
        "schema": "hcli.odyssey.ctl.v1",
        "argv": args,
        "exit_code": proc.returncode,
        # A dry-run acquisition can safely conclude REFUSE (disk/memory
        # admission) and still completed its requested planning operation.
        # Preserve the refusal in stdout/exit_code; ``ok`` describes the
        # read-only HCLI call, not whether admission was granted.
        "ok": proc.returncode == 0 or (
            args[:1] == ["acquire-next"]
            and "--dry-run" in args
            and proc.stdout.startswith("REFUSE")
        ),
        "stdout": stdout,
        "stderr": proc.stderr,
    }


def _require_confirm(confirm: bool, what: str) -> None:
    if confirm is not True:
        raise PermissionError(f"{what} mutates Odyssey state and requires confirm=True")


# --------------------------------------------------------------------------
# inspect -- read-only, no gate
# --------------------------------------------------------------------------

def status() -> dict:
    """Queue + current patient + compiler rules + research counters, in one shot."""
    return _run(["status"])


def queue() -> dict:
    return _run(["queue"])


def value() -> dict:
    """Ranked NEXT-work list with info-value proxies."""
    return _run(["value"])


def economics() -> dict:
    return _run(["economics"])


def completions(rebuild: bool = False, confirm: bool = False, completed_at: Optional[str] = None) -> dict:
    """List recorded completions; ``rebuild=True`` writes a backfill and needs confirm."""
    if not rebuild:
        return _run(["completions"])
    _require_confirm(confirm, "completions --rebuild")
    args = ["completions", "--rebuild"]
    if completed_at:
        args += ["--completed-at", completed_at]
    return _run(args)


def patient(oxx: str) -> dict:
    """Read a patient packet already on disk. Never writes one (see write_packet)."""
    path = PATIENTS_DIR / oxx / f"ODYSSEY_PATIENT_{oxx}.json"
    if not path.is_file():
        return {"schema": "hcli.odyssey.patient.v1", "oxx": oxx, "found": False, "path": str(path)}
    return {
        "schema": "hcli.odyssey.patient.v1",
        "oxx": oxx,
        "found": True,
        "path": str(path),
        "packet": json.loads(path.read_text()),
    }


# cli-only: NOT read-only despite the name -- odyssey_ctl's cmd_admit ends in reclaim_if_tight(), which shells out to tools/reclaim_safe.sh (rm -rf build caches, finished lane dirs, dead worktrees) whenever free disk is under DISK_WARN_GIB, so registering it as an inspect verb would advertise a lie.
def admit_check(slug: str, est_gib: float) -> dict:
    """Memgate/worker_gate/disk-floor GO-or-REFUSE check. Never touches queue
    state, but a REFUSE on the disk floor triggers a reclaim sweep on the way
    out -- see the cli-only note above."""
    return _run(["admit", slug, str(est_gib)])


# --------------------------------------------------------------------------
# continue -- mutating, confirm=True required (dry-run paths excepted)
# --------------------------------------------------------------------------

def harvest(dry_run: bool = True, confirm: bool = False) -> dict:
    if dry_run:
        return _run(["harvest", "--dry-run"])
    _require_confirm(confirm, "harvest (non-dry-run)")
    return _run(["harvest"])


def write_packet(oxx: str, confirm: bool = False) -> dict:
    _require_confirm(confirm, f"packet {oxx} (writes a patient packet file)")
    return _run(["packet", oxx])


# cli-only: odyssey_ctl's cycle_tick() already calls run_loop() with the same lane caps, and odyssey.cycle is registered, so a second registered lane-spawner is duplicate COSTLY surface with no reachable behaviour of its own.
def run(confirm: bool = False, dry_run: bool = True, max_lanes: int = 2, grok_lanes: int = 0) -> dict:
    args = ["run", "--max-lanes", str(max_lanes), "--grok-lanes", str(grok_lanes)]
    if dry_run:
        return _run([*args, "--dry-run"])
    _require_confirm(confirm, "run --go (spawns grok-run lanes)")
    return _run([*args, "--go"], timeout_s=600.0)


def cycle(
    confirm: bool = False,
    dry_run: bool = True,
    max_lanes: int = 2,
    grok_lanes: int = 0,
    loop_secs: Optional[float] = None,
    inner_sleep: float = 3.0,
) -> dict:
    args = ["cycle", "--max-lanes", str(max_lanes), "--grok-lanes", str(grok_lanes)]
    if loop_secs is not None:
        args += ["--loop-secs", str(loop_secs), "--inner-sleep", str(inner_sleep)]
    if dry_run:
        return _run([*args, "--dry-run"])
    _require_confirm(confirm, "cycle --go (harvest/retire/acquire/launch for real)")
    return _run([*args, "--go"], timeout_s=(loop_secs or 60.0) + 120.0)


def retire(oxx: str, confirm: bool = False) -> dict:
    _require_confirm(confirm, f"retire {oxx} (removes a patient from the queue)")
    return _run(["retire", oxx])


# cli-only: --go starts a multi-hundred-GiB Hugging Face download; cycle_tick() already acquires when no obligation is ready, and acquisition.propose covers the read side, so nothing needs a bare registered download starter.
def acquire_next(confirm: bool = False, dry_run: bool = True) -> dict:
    if dry_run:
        return _run(["acquire-next", "--dry-run"])
    _require_confirm(confirm, "acquire-next --go (starts a Hugging Face download)")
    return _run(["acquire-next", "--go"])


def gravity_gauntlet(
    oxx: str,
    candidate_specs: list[str],
    budget: int = 2,
    state_path: Optional[str] = None,
    receipt_dir: Optional[str] = None,
    confirm: bool = False,
) -> dict:
    """Run one bounded, resumable Gravity search from existing patient receipts.

    HCLI owns the checkpoint and next-candidate decision; the existing patient
    runner remains the evaluator. This helper intentionally requires explicit
    confirmation because it writes the gauntlet checkpoint.
    """
    _require_confirm(confirm, "odyssey.gravity_gauntlet")
    if not candidate_specs:
        raise ValueError("candidate_specs must not be empty")
    from .gravity_gauntlet import run_from_receipts

    state = state_path or str(ODYSSEY_DIR / "gauntlets" / f"{oxx}.json")
    receipts = receipt_dir or str(REPO / "receipts" / "odyssey-i")
    return run_from_receipts(state, oxx, candidate_specs, budget, receipts)


# --------------------------------------------------------------------------
# own -- HCLI's own ledger on top of the driver's real state. ingest() reads
# both; everything else writes only HCLI_LEDGER.json and gates on confirm.
# --------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _writer_identity() -> dict:
    """Who actually appended this entry, stamped at append time.

    ``hcli_owned`` is true only when the writing process shares the live
    resident worker's session. The supervisor spawns that worker with
    ``start_new_session=True`` (hcli/agentos/resident.py), so the worker leads
    its own session and only its descendants share the sid -- a human shell
    importing this module sits in the terminal's session and stamps false.
    The recorded start token is matched too, so a recycled pid does not pass.

    This is written once, when the fact is knowable, and can never be
    back-filled: G011's producer reads it to decide whether the campaign was
    HCLI's own or a person's. A failure to determine ownership stamps false,
    never true -- the safe direction.
    """
    owned = False
    try:
        state = json.loads((REPO / ".hcli" / "resident" / "state.json").read_text())
        pid = state.get("worker_pid")
        if isinstance(pid, int) and pid > 0:
            from .resources import process_start_token

            if process_start_token(pid) == state.get("worker_start_token"):
                owned = os.getsid(0) == os.getsid(pid)
    except (OSError, ValueError, ImportError, ProcessLookupError):
        owned = False
    return {"writer_pid": os.getpid(), "hcli_owned": owned}


def _real_specimens() -> dict[str, dict]:
    """oxx -> patient, read straight off the driver's own on-disk state.

    Read-only, no fixture, no re-run: if ``ODYSSEY_STATE.json`` is absent or
    unparseable this returns ``{}`` rather than inventing specimens.
    """
    if not STATE_PATH.is_file():
        return {}
    try:
        state = json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {p["oxx"]: p for p in (state.get("patients") or []) if p.get("oxx")}


def _require_known_oxx(oxx: str) -> None:
    """Refuse to record a specimen HCLI invented. Skipped only when the
    driver's state file itself is missing -- nothing real to check against."""
    specimens = _real_specimens()
    if specimens and oxx not in specimens:
        raise ValueError(f"{oxx} is not a specimen in {STATE_PATH}")


def _empty_ledger() -> dict:
    return {"schema": "hcli.odyssey.ledger.v1", **{k: [] for k in _LEDGER_LISTS}}


def _load_ledger() -> dict:
    if not LEDGER_PATH.is_file():
        return _empty_ledger()
    try:
        data = json.loads(LEDGER_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return _empty_ledger()
    ledger = _empty_ledger()
    for key in _LEDGER_LISTS:
        if isinstance(data.get(key), list):
            ledger[key] = data[key]
    return ledger


def _save_ledger(ledger: dict) -> None:
    atomic_write_json(LEDGER_PATH, ledger)


def ingest() -> dict:
    """The real, mid-flight Odyssey state plus HCLI's own ledger, in one
    shape HCLI can reason over. Read-only: never seeds, re-runs, or touches
    a sealed patient -- O003 stays exactly as sealed as it was found."""
    specimens = _real_specimens()
    ledger = _load_ledger()
    return {
        "schema": "hcli.odyssey.ingest.v1",
        "state_path": str(STATE_PATH),
        "ledger_path": str(LEDGER_PATH),
        "specimen_count": len(specimens),
        "specimens": [
            {
                "oxx": oxx,
                "state": p.get("state"),
                "ledger": p.get("ledger"),
                "on_disk": bool(p.get("on_disk")),
                "class": p.get("class"),
            }
            for oxx, p in sorted(specimens.items())
        ],
        **{key: ledger[key] for key in _LEDGER_LISTS},
    }


def add_to_eligibility(oxx: str, note: str = "", confirm: bool = False) -> dict:
    """Mark a real specimen eligible for HCLI-driven work. HCLI decides what
    eligibility means and when to act on it; this only records the intent.
    Re-adding an already-eligible oxx updates its note rather than duplicating it.
    """
    _require_confirm(confirm, f"add {oxx} to eligibility")
    _require_known_oxx(oxx)
    ledger = _load_ledger()
    ledger["eligible"] = [e for e in ledger["eligible"] if e.get("oxx") != oxx]
    entry = {"oxx": oxx, "note": note, "recorded_at": _now(), **_writer_identity()}
    ledger["eligible"].append(entry)
    _save_ledger(ledger)
    return {"schema": "hcli.odyssey.ledger.v1", "list": "eligible", "entry": entry}


def park_specimen(oxx: str, reason: str, confirm: bool = False) -> dict:
    """Set a specimen aside without deleting anything it already earned.
    Re-parking replaces the prior reason rather than piling up duplicates."""
    _require_confirm(confirm, f"park {oxx}")
    if not reason:
        raise ValueError("park_specimen requires a reason")
    _require_known_oxx(oxx)
    ledger = _load_ledger()
    ledger["parked"] = [e for e in ledger["parked"] if e.get("oxx") != oxx]
    entry = {"oxx": oxx, "reason": reason, "recorded_at": _now(), **_writer_identity()}
    ledger["parked"].append(entry)
    _save_ledger(ledger)
    return {"schema": "hcli.odyssey.ledger.v1", "list": "parked", "entry": entry}


def record_law(text: str, evidence: str = "", source_oxx: Optional[str] = None, confirm: bool = False) -> dict:
    """Record a claimed compiler/architecture law. HCLI decides what counts
    as a law worth claiming; this only appends it so a transfer probe (II)
    or an adversarial probe (III) has something to target -- immediately,
    with no phase barrier."""
    _require_confirm(confirm, "record a law")
    if not text:
        raise ValueError("record_law requires text")
    if source_oxx:
        _require_known_oxx(source_oxx)
    ledger = _load_ledger()
    law_id = f"LAW{len(ledger['laws']) + 1:03d}"
    entry = {"id": law_id, "text": text, "evidence": evidence, "source_oxx": source_oxx,
             "recorded_at": _now(), **_writer_identity()}
    ledger["laws"].append(entry)
    _save_ledger(ledger)
    return {"schema": "hcli.odyssey.ledger.v1", "list": "laws", "entry": entry}


def record_scar(law_id: str, description: str, confirm: bool = False) -> dict:
    """Record that a claimed law took damage -- a failed transfer, a
    successful attack. Does not delete the law: a scar is evidence against
    it, not a retraction of it."""
    _require_confirm(confirm, f"record a scar on {law_id}")
    if not description:
        raise ValueError("record_scar requires a description")
    ledger = _load_ledger()
    if not any(law["id"] == law_id for law in ledger["laws"]):
        raise ValueError(f"{law_id} is not a recorded law")
    entry = {"law_id": law_id, "description": description, "recorded_at": _now(), **_writer_identity()}
    ledger["scars"].append(entry)
    _save_ledger(ledger)
    return {"schema": "hcli.odyssey.ledger.v1", "list": "scars", "entry": entry}


def create_transfer_probe(law_id: str, target_oxx: str, confirm: bool = False) -> dict:
    """Odyssey II: propose testing whether ``law_id`` transfers to
    ``target_oxx``. Requires only that the law has been claimed -- may run
    the moment I produces a plausible law, per the campaign's overlap rule.
    Records the proposal; running the probe is a separate, real job."""
    _require_confirm(confirm, f"create transfer probe {law_id} -> {target_oxx}")
    ledger = _load_ledger()
    if not any(law["id"] == law_id for law in ledger["laws"]):
        raise ValueError(f"{law_id} is not a recorded law")
    _require_known_oxx(target_oxx)
    probe_id = f"TP{len(ledger['transfer_probes']) + 1:03d}"
    entry = {"id": probe_id, "law_id": law_id, "target_oxx": target_oxx, "status": "PROPOSED",
             "recorded_at": _now(), **_writer_identity()}
    ledger["transfer_probes"].append(entry)
    _save_ledger(ledger)
    return {"schema": "hcli.odyssey.ledger.v1", "list": "transfer_probes", "entry": entry}


def create_adversarial_probe(law_id: str, attack: str, confirm: bool = False) -> dict:
    """Odyssey III: propose an attack against a claimed law. Requires only
    that the law has been claimed -- may run the moment it is, per the
    campaign's overlap rule. Records the proposal; running the attack is a
    separate, real job."""
    _require_confirm(confirm, f"create adversarial probe against {law_id}")
    if not attack:
        raise ValueError("create_adversarial_probe requires an attack description")
    ledger = _load_ledger()
    if not any(law["id"] == law_id for law in ledger["laws"]):
        raise ValueError(f"{law_id} is not a recorded law")
    probe_id = f"AP{len(ledger['adversarial_probes']) + 1:03d}"
    entry = {"id": probe_id, "law_id": law_id, "attack": attack, "status": "PROPOSED",
             "recorded_at": _now(), **_writer_identity()}
    ledger["adversarial_probes"].append(entry)
    _save_ledger(ledger)
    return {"schema": "hcli.odyssey.ledger.v1", "list": "adversarial_probes", "entry": entry}
