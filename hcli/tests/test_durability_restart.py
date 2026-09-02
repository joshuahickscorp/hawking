"""Cross-process durability for ledger verification and sessions.

An in-process round-trip would pass on the broken code: mark_verified
mutates memory, and Controller holds a Session object. These tests write
in one Python process, exit, and read in a second process. That is the
restart the product claims to survive.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


_VERIFY_WRITER = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hcli.ledger import Ledger
goal = Path(sys.argv[2])
led = Ledger.parse(goal)
result = led.run_verify("G001")
if not result.passed:
    sys.stderr.write("verify did not pass:\n")
    sys.stderr.write(result.output)
    sys.stderr.write("\n")
    raise SystemExit(2)
led.mark_verified("G001", result)
# Deliberately do not call Ledger.save. Persistence is mark_verified's job.
raise SystemExit(0)
"""

_VERIFY_READER = r"""
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hcli.ledger import Ledger
goal = Path(sys.argv[2])
led = Ledger.parse(goal)
ob = led.get("G001")
receipt = goal.parent / ".hcli" / "verify-receipts" / "G001.json"
print("status=" + ob.status)
print("checked=" + str(ob.checked))
print("receipt=" + str(receipt.is_file()))
if receipt.is_file():
    rec = json.loads(receipt.read_text(encoding="utf-8"))
    print("receipt_passed=" + str(rec.get("passed")))
ok = ob.status == "VERIFIED" and ob.checked and receipt.is_file()
raise SystemExit(0 if ok else 1)
"""

_SESSION_WRITER = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hcli.controller import Controller
from hcli.events import EventBus
ws = sys.argv[2]
out = Path(sys.argv[3])
c = Controller(workspace=ws, runtime_count=1, bus=EventBus())
out.write_text(c.session.id, encoding="utf-8")
# Exit without shutdown. A session that only hits disk in shutdown is
# not a session that survives a crash.
raise SystemExit(0)
"""

_SESSION_READER = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hcli.controller import Controller
from hcli.events import EventBus
ws = sys.argv[2]
sid = Path(sys.argv[3]).read_text(encoding="utf-8").strip()
c = Controller(workspace=ws, runtime_count=1, bus=EventBus())
got = c.resume_session(sid)
print("sid=" + sid)
print("got=" + repr(got))
print("session_id=" + c.session.id)
raise SystemExit(0 if got == sid and c.session.id == sid else 1)
"""


def _run_child(script: str, *args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-c", script, str(REPO), *args]
    env = os.environ.copy()
    env["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
    return subprocess.run(
        cmd,
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _passing_goal_md(tmp: Path) -> Path:
    marker = tmp / "probe.txt"
    marker.write_text("ok\n", encoding="utf-8")
    cmd = (
        "python3 -c \"import pathlib,sys; "
        "sys.exit(0 if pathlib.Path(r'%s').read_text().strip()=='ok' else 1)\""
    ) % marker
    goal = tmp / "GOAL.md"
    goal.write_text(
        "- [ ] G001 — durable verify | status: PENDING | risk: high | tier: V2\n"
        "      acceptance: probe file reads ok\n"
        f"      verify: {cmd}\n"
        "      evidence: (none yet)\n",
        encoding="utf-8",
    )
    return goal


class TestVerifiedSurvivesProcessRestart(unittest.TestCase):
    def test_mark_verified_is_read_by_a_separate_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            goal = _passing_goal_md(root)
            writer = _run_child(_VERIFY_WRITER, str(goal))
            self.assertEqual(
                writer.returncode,
                0,
                msg=(
                    "writer process failed\n"
                    f"stdout={writer.stdout!r}\n"
                    f"stderr={writer.stderr!r}"
                ),
            )
            reader = _run_child(_VERIFY_READER, str(goal))
            self.assertEqual(
                reader.returncode,
                0,
                msg=(
                    "reader process did not see VERIFIED\n"
                    f"stdout={reader.stdout!r}\n"
                    f"stderr={reader.stderr!r}\n"
                    f"GOAL.md={goal.read_text(encoding='utf-8')!r}"
                ),
            )
            self.assertIn("status=VERIFIED", reader.stdout)
            self.assertIn("checked=True", reader.stdout)
            self.assertIn("receipt=True", reader.stdout)


class TestSessionSurvivesProcessRestart(unittest.TestCase):
    def test_session_created_in_one_process_resumes_in_another(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            sid_path = Path(tmp) / "sid.txt"
            writer = _run_child(_SESSION_WRITER, str(ws), str(sid_path))
            self.assertEqual(
                writer.returncode,
                0,
                msg=(
                    "writer process failed\n"
                    f"stdout={writer.stdout!r}\n"
                    f"stderr={writer.stderr!r}"
                ),
            )
            sid = sid_path.read_text(encoding="utf-8").strip()
            self.assertTrue(sid)
            session_files = list((ws / ".hcli" / "sessions").glob("*.json"))
            self.assertTrue(
                session_files,
                msg="writer exited with no session file on disk",
            )
            reader = _run_child(_SESSION_READER, str(ws), str(sid_path))
            self.assertEqual(
                reader.returncode,
                0,
                msg=(
                    "reader process did not resume the session\n"
                    f"sid={sid!r}\n"
                    f"stdout={reader.stdout!r}\n"
                    f"stderr={reader.stderr!r}\n"
                    f"session_files={[p.name for p in session_files]}"
                ),
            )
            self.assertIn(f"got='{sid}'", reader.stdout.replace('"', "'"))


if __name__ == "__main__":
    unittest.main()
