"""A typed-tool WorkUnit must survive a disk round-trip.

`WorkUnit.tool` is what routes a unit to the ToolRegistry instead of to
cognition -- the resident's only path to typed tools (`resident.py`
`_child_workunit` sets `tool=`, `executors.py` dispatches it to
`registry.invoke`). It was declared as a field but omitted from `to_dict()`
and from `dag_store._PERSISTED_EXTRAS`, so any `dag.json` write/reload
silently demoted the unit to cognition with no tool, no arguments and no
error. It worked in-memory in one process and lost the binding on restart.

`provider`, declared immediately above it, carries the comment "persisted so a
restart does not silently route a specialist unit through the current
resident". The same concern, handled there and missed here.

The failure mode is silent, so the test asserts the binding survives rather
than asserting that some call succeeded.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hcli.dag_store import _apply_disk_extras, _unit_to_disk
from hcli.workunit import WorkUnit


def _tool_unit() -> WorkUnit:
    return WorkUnit(
        id="u-tool-1",
        role="worker",
        description="read a file through the typed tool surface",
        tool="file.read",
        tool_arguments={"path": "README.md", "max_bytes": 4096},
    )


class TestToolSurvivesPersistence(unittest.TestCase):
    def test_tool_binding_is_written_to_disk(self) -> None:
        payload = _unit_to_disk(_tool_unit())
        self.assertIn("tool", payload, "tool binding never reached disk")
        self.assertEqual(payload["tool"], "file.read")
        self.assertIn("tool_arguments", payload, "tool arguments never reached disk")
        self.assertEqual(payload["tool_arguments"]["path"], "README.md")

    def test_tool_binding_survives_a_full_round_trip(self) -> None:
        payload = _unit_to_disk(_tool_unit())
        # Through real JSON, because that is what dag.json does -- an in-memory
        # dict round-trip would not catch a non-serializable value.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "dag.json"
            p.write_text(json.dumps({"units": [payload]}), encoding="utf-8")
            restored_payload = json.loads(p.read_text(encoding="utf-8"))["units"][0]

        restored = WorkUnit(
            id=restored_payload["id"],
            role=restored_payload["role"],
            description=restored_payload["description"],
        )
        _apply_disk_extras(restored, restored_payload)

        self.assertEqual(
            restored.tool,
            "file.read",
            "the unit was silently demoted to cognition on reload",
        )
        self.assertEqual(restored.tool_arguments, {"path": "README.md", "max_bytes": 4096})

    def test_a_unit_with_no_tool_stays_clean(self) -> None:
        """The negative control: persistence must not invent a tool binding."""
        payload = _unit_to_disk(
            WorkUnit(id="u-plain", role="worker", description="ordinary cognition")
        )
        self.assertIsNone(payload.get("tool"))
        restored = WorkUnit(id="u-plain", role="worker", description="ordinary cognition")
        _apply_disk_extras(restored, payload)
        self.assertIsNone(restored.tool, "persistence fabricated a tool binding")


if __name__ == "__main__":
    unittest.main()
