"""A capability probe must outlive the process, and not outlive the binary.

`--help` on a model server costs about a second -- mlx_lm imports torch just to
print it -- and the in-memory cache only helps a process that asks twice. Every
HCLI invocation and every test shard paid it again: measured 1.14 s inside a
4.4 s sharded suite, and once per control-plane start.

The key includes the binary's mtime and size, so a rebuilt binary is re-probed.
A capability answer that outlives the thing it describes is worse than no
cache: it reports what the runtime USED to support.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from hcli.backends import _probe_cached, _probe_key, _probe_store


class TestProbeCache(unittest.TestCase):
    def test_a_stored_probe_is_returned(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            b = Path(tmp) / "fake-server"
            b.write_text("#!/bin/sh\n")
            _probe_store(str(b), "help", "usage: fake --flag")
            self.assertEqual(_probe_cached(str(b), "help"), "usage: fake --flag")

    def test_a_changed_binary_invalidates_the_probe(self):
        """The property that matters: a rebuild must not answer from cache."""
        import tempfile
        import time

        with tempfile.TemporaryDirectory() as tmp:
            b = Path(tmp) / "fake-server"
            b.write_text("v1")
            _probe_store(str(b), "help", "supports: nothing")
            before = _probe_key(str(b), "help")

            time.sleep(0.01)
            b.write_text("v2 is a different binary entirely")
            after = _probe_key(str(b), "help")

            self.assertNotEqual(before, after, "a changed binary reused its key")
            self.assertIsNone(_probe_cached(str(b), "help"))

    def test_kinds_do_not_collide(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            b = Path(tmp) / "s"
            b.write_text("x")
            _probe_store(str(b), "llama-help", "llama")
            _probe_store(str(b), "mlx-help", "mlx")
            self.assertEqual(_probe_cached(str(b), "llama-help"), "llama")
            self.assertEqual(_probe_cached(str(b), "mlx-help"), "mlx")

    def test_a_missing_binary_does_not_raise(self):
        self.assertIsNone(_probe_cached("/no/such/binary", "help"))
        _probe_store("/no/such/binary", "help", "x")

    def test_an_unwritable_cache_is_not_fatal(self):
        """Telemetry must never be the thing that ends a run."""
        old = os.environ.get("HCLI_PROBE_CACHE")
        try:
            os.environ["HCLI_PROBE_CACHE"] = "/dev/null/nope"
            _probe_store("/bin/sh", "help", "x")  # must not raise
        finally:
            if old is None:
                os.environ.pop("HCLI_PROBE_CACHE", None)
            else:
                os.environ["HCLI_PROBE_CACHE"] = old


if __name__ == "__main__":
    unittest.main()
