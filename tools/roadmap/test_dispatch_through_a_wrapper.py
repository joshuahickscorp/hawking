"""A tool called through a one-line local wrapper is still called.

capability_reachability.py exists to answer "is this capability reachable, or
just defined" -- its own docstring opens "A definition is not a capability".
Its dispatch matcher looked for `invoke("name"` or `"tool": "name"` on one line.

hcli/agentos/research.py and hcli/agentos/vmcp_gate.py both route every tool
call through a local

    def call(name, arguments):
        return registry.invoke(name, arguments)

so `call("web.search", ...)` matched nothing and NINE live tools -- web.search,
web.fetch, github.search, github.fetch, huggingface.resolve/.fetch_file/
.download, vmcp.inspect, vmcp.query -- were reported DEAD or TEST-ONLY while
being exercised on every `hcli agentos research-gate` run.

A reachability checker that reports live capabilities as dead is worse than no
checker: it is the instrument the campaign uses to decide what to delete.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capability_reachability import _extract_file_facts  # noqa: E402


def _scan(source: str):
        return _extract_file_facts("mod.py", source, frozenset())


class TestWrapperDispatchIsSeen(unittest.TestCase):
    def test_a_direct_invoke_is_seen(self):
        found = _scan('registry.invoke("web.search", {})')
        self.assertIn("web.search", [name for name, _ in found["literals"]])

    def test_a_call_through_a_local_wrapper_is_seen(self):
        source = (
            "def build():\n"
            "    registry = make()\n"
            "    def call(name, arguments):\n"
            "        return registry.invoke(name, arguments)\n"
            "    search = call('web.search', {'query': 'x'})\n"
            "    page = call('web.fetch', {'url': 'y'})\n"
            "    return search, page\n"
        )
        names = [name for name, _ in _scan(source)["literals"]]
        self.assertIn("web.search", names,
                      "a tool dispatched through a local wrapper reads as DEAD")
        self.assertIn("web.fetch", names)

    def test_an_unrelated_helper_is_not_treated_as_dispatch(self):
        # Precision: a function that merely takes a name must not turn every
        # string passed to it into a "reachable tool".
        source = (
            "def label(name, value):\n"
            "    return f'{name}={value}'\n"
            "label('not.a.tool', 1)\n"
        )
        names = [name for name, _ in _scan(source)["literals"]]
        self.assertNotIn("not.a.tool", names)

    def test_a_wrapper_that_invokes_something_else_is_not_dispatch(self):
        source = (
            "def call(name, arguments):\n"
            "    return other.run(name, arguments)\n"
            "call('not.a.tool', {})\n"
        )
        names = [name for name, _ in _scan(source)["literals"]]
        self.assertNotIn("not.a.tool", names)


if __name__ == "__main__":
    unittest.main()
