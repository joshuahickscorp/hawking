from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from hcli.engine import Engine
from hcli.events import EventBus
from hcli.workspace import Workspace


class EvidenceAdmissionTests(unittest.TestCase):

    def make_engine(self, root: Path) -> Engine:
        return Engine(
            workspace=Workspace(str(root)),
            event_bus=EventBus(),
            runtime_count=1,
            model_name="test",
        )

    def test_readme_does_not_expand_repository_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            refs = []

            for i in range(40):
                name = f"module_{i}.py"

                (root / name).write_text(
                    "x = " + repr("z" * 10000),
                    encoding="utf-8",
                )

                refs.append(name)

            (root / "README.md").write_text(
                "# Small README\n\n"
                + "\n".join(refs),
                encoding="utf-8",
            )

            engine = self.make_engine(root)

            evidence = engine._gather_evidence(
                "Read README.md and report its first heading."
            )

            self.assertEqual(
                [item["path"] for item in evidence],
                ["README.md"],
            )

    def test_explicit_task_document_can_expand_nested_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            (root / ".haider").mkdir()

            (root / "target.py").write_text(
                "VALUE = 1\n",
                encoding="utf-8",
            )

            spec = (
                root
                / ".haider"
                / "FULFILLMENT_TASK.md"
            )

            spec.write_text(
                "# Task\n\n"
                "Inspect target.py and execute the requested task.\n",
                encoding="utf-8",
            )

            engine = self.make_engine(root)

            evidence = engine._gather_evidence(
                "Read .haider/FULFILLMENT_TASK.md and execute it."
            )

            paths = [
                item["path"]
                for item in evidence
            ]

            self.assertIn(
                ".haider/FULFILLMENT_TASK.md",
                paths,
            )

            self.assertIn(
                "target.py",
                paths,
            )

    def test_evidence_respects_context_derived_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            big = root / "BIG_TASK.md"

            big.write_text(
                "x" * 200000,
                encoding="utf-8",
            )

            old_ctx = os.environ.get(
                "HCLI_CTX_SIZE"
            )

            old_out = os.environ.get(
                "HCLI_MODEL_TOKENS"
            )

            try:
                os.environ[
                    "HCLI_CTX_SIZE"
                ] = "8192"

                os.environ[
                    "HCLI_MODEL_TOKENS"
                ] = "1024"

                engine = self.make_engine(root)

                evidence = engine._gather_evidence(
                    "Read BIG_TASK.md."
                )

                total = sum(
                    len(item["content"])
                    for item in evidence
                )

                self.assertLessEqual(
                    total,
                    engine._evidence_char_budget(),
                )

            finally:
                if old_ctx is None:
                    os.environ.pop(
                        "HCLI_CTX_SIZE",
                        None,
                    )
                else:
                    os.environ[
                        "HCLI_CTX_SIZE"
                    ] = old_ctx

                if old_out is None:
                    os.environ.pop(
                        "HCLI_MODEL_TOKENS",
                        None,
                    )
                else:
                    os.environ[
                        "HCLI_MODEL_TOKENS"
                    ] = old_out


if __name__ == "__main__":
    unittest.main()
