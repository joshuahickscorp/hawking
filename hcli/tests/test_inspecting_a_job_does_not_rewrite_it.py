"""Reading a background job must not write it back.

`inspect()` routes through `_refresh`, which ended with an unconditional
`self._write(job)`. So reading a job was a mutation of it, and the sequence
that matters is:

  reader: _read()  -> RUNNING
  supervisor:         writes COMPLETED into the receipt
  reader: _write() -> RUNNING, over the top of it

The detached supervisor owns the receipt and is the only writer of the terminal
record. Once its record was clobbered it was already gone -- and a dead
supervisor is a zombie, which `_pid_alive` calls alive because os.kill(pid, 0)
succeeds on one -- so nothing ever moved the job off RUNNING and a child that
had finished was reported running forever.

Measured: one run in 48 under load, and the job never left RUNNING no matter
how long the reader waited.
"""
from __future__ import annotations

import json
import os
import unittest

import tempfile
from pathlib import Path

from hcli.agentos.background import BackgroundJob, BackgroundJobStore


def _running_job(store: BackgroundJobStore, pid: int) -> str:
    """A job the reader will see as RUNNING, owned by a live pid."""
    job = BackgroundJob(
        job_id="job-reader-test",
        argv=["/bin/echo", "hi"],
        cwd=str(store.root),
        state="RUNNING",
        pid=pid,
        created_at=1.0,
        started_at=1.0,
    )
    store._write(job)
    return job.job_id


class TestInspectDoesNotRewrite(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def _receipt(self, store, job_id):
        return json.loads(store._path(job_id).read_text())

    def test_inspect_leaves_the_receipt_byte_identical(self):
        """A sentinel key survives only if nothing rewrote the file.

        `_write` serialises the dataclass, which has no place to keep an
        unknown key -- so the sentinel disappearing IS the write.
        """
        store = BackgroundJobStore(self.workspace)
        job_id = _running_job(store, os.getpid())  # this process is always alive

        path = store._path(job_id)
        record = json.loads(path.read_text())
        record["sentinel"] = "written-by-the-supervisor"
        path.write_text(json.dumps(record))

        reader = BackgroundJobStore(self.workspace)
        for _ in range(5):
            reader.inspect(job_id)

        after = json.loads(path.read_text())
        self.assertEqual(
            after.get("sentinel"), "written-by-the-supervisor",
            "inspect() rewrote a receipt it had not changed",
        )

    def test_a_terminal_record_survives_a_stale_reader(self):
        """The exact clobber, with the interleaving forced.

        The reader holds a RUNNING view; the supervisor writes COMPLETED
        underneath it; the reader must not put RUNNING back.
        """
        store = BackgroundJobStore(self.workspace)
        job_id = _running_job(store, os.getpid())

        reader = BackgroundJobStore(self.workspace)
        stale = reader._read(job_id)
        self.assertEqual(stale.state, "RUNNING")

        supervised = BackgroundJob.from_dict(self._receipt(store, job_id))
        supervised.state = "COMPLETED"
        supervised.returncode = 0
        supervised.finished_at = 2.0
        store._write(supervised)

        reader._refresh(stale)

        self.assertEqual(
            self._receipt(store, job_id)["state"], "COMPLETED",
            "a stale reader overwrote the supervisor's terminal record",
        )

    def test_a_real_change_is_still_persisted(self):
        """Negative control: the guard must not stop a genuine transition.

        A RUNNING job whose owner is gone has to be recorded INTERRUPTED, or
        the guard would trade one stuck state for another.
        """
        store = BackgroundJobStore(self.workspace)
        dead = 2_000_000_000  # far above any live pid on this box
        job_id = _running_job(store, dead)

        reader = BackgroundJobStore(self.workspace)
        self.assertEqual(reader.inspect(job_id)["state"], "INTERRUPTED")
        self.assertEqual(
            self._receipt(store, job_id)["state"], "INTERRUPTED",
            "a real state change was not written to the receipt",
        )


if __name__ == "__main__":
    unittest.main()
