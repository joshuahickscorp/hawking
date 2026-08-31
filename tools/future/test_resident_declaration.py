"""A running resident must not be invisible to its own health probe.

The concurrency doctor (G009) sat SLEEPING with "resident presence is
UNDECLARED, not PRESENT; will not invent a pid from the largest RSS neighbour"
while the sealed resident was RUNNING at pid 88764 with 13.7 GB RSS. That
refusal is correct - the doctor must never guess - but the resident had already
DECLARED itself: hcli/agentos/resident.py persists worker_pid and
worker_start_token to .hcli/resident/state.json via resident_state_path().
resident_health.sample() simply never looked there. Two components that should
agree, did not, and the cost was an obligation blocked on a plumbing gap.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hcli.resources import process_start_token  # noqa: E402
from tools.future import resident_health as rh  # noqa: E402


def _store(ws: Path, doc: dict) -> None:
    p = ws / ".hcli" / "resident" / "state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))


def test_a_declared_live_resident_reads_present(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _store(tmp_path, {
            "worker_pid": proc.pid,
            "worker_start_token": process_start_token(proc.pid),
        })
        r = rh.sample(workspace=tmp_path)["resident"]
        assert r["presence"] == "PRESENT"
        assert r["pid"] == proc.pid
        assert r["rss_bytes"] is not None
    finally:
        proc.kill()
        proc.wait()


def test_a_stale_start_token_is_refused_not_honoured(tmp_path):
    """A recycled pid must never be reported as the resident."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _store(tmp_path, {"worker_pid": proc.pid, "worker_start_token": "0.0"})
        assert rh.sample(workspace=tmp_path)["resident"]["presence"] == "UNDECLARED"
    finally:
        proc.kill()
        proc.wait()


def test_a_dead_declared_pid_is_absent_not_present(tmp_path):
    _store(tmp_path, {"worker_pid": 999999, "worker_start_token": None})
    assert rh.sample(workspace=tmp_path)["resident"]["presence"] == "ABSENT"


def test_no_store_stays_undeclared_and_still_refuses_to_guess(tmp_path):
    """The doctor's refusal survives: absent a declaration, nothing is invented."""
    r = rh.sample(workspace=tmp_path)["resident"]
    assert r["presence"] == "UNDECLARED"
    assert "will not invent one from the largest RSS neighbour" in r["reason"]


def test_a_malformed_store_does_not_crash_the_probe(tmp_path):
    p = tmp_path / ".hcli" / "resident" / "state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json")
    assert rh.sample(workspace=tmp_path)["resident"]["presence"] == "UNDECLARED"


def test_an_explicit_pid_still_wins_over_the_store(tmp_path):
    """The declaration is a fallback, not an override."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _store(tmp_path, {"worker_pid": 999999, "worker_start_token": None})
        r = rh.sample(pid=proc.pid, workspace=tmp_path)["resident"]
        assert r["pid"] == proc.pid
        assert r["presence"] == "PRESENT"
    finally:
        proc.kill()
        proc.wait()
