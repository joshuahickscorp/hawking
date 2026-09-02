from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]

from hcli.cli import parse_hcli_args, resolve_resident_runtime_limit
from hcli.machine import live_machine_identity


class TestGrammar(unittest.TestCase):
    def test_empty_is_interactive(self):
        args = parse_hcli_args([])
        self.assertEqual(args.runtime_count, 1)
        self.assertIsNone(args.prompt)
        self.assertTrue(args.interactive)
        self.assertIsNone(args.max_source)

    def test_prompt_only(self):
        args = parse_hcli_args(["p"])
        self.assertEqual(args.runtime_count, 1)
        self.assertEqual(args.prompt, "p")
        self.assertFalse(args.interactive)
        self.assertIsNone(args.max_source)

    def test_n_and_prompt(self):
        args = parse_hcli_args(["3", "p"])
        self.assertEqual(args.runtime_count, 3)
        self.assertEqual(args.prompt, "p")
        self.assertFalse(args.interactive)
        self.assertIsNone(args.max_source)

    def test_zero_rejected(self):
        with self.assertRaises(SystemExit):
            parse_hcli_args(["0", "p"])

    def test_nine_rejected(self):
        with self.assertRaises(SystemExit):
            parse_hcli_args(["9", "p"])

    def test_max_resolves_and_sets_source(self):
        saved = os.environ.pop("HCLI_RESIDENT_RUNTIME_LIMIT", None)
        try:
            args = parse_hcli_args(["max", "p"])
        finally:
            if saved is not None:
                os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = saved
        self.assertEqual(args.prompt, "p")
        self.assertFalse(args.interactive)
        self.assertIsNotNone(args.max_source)
        self.assertIn(
            args.max_source,
            {
                "HCLI_RESIDENT_RUNTIME_LIMIT",
                "RESIDENT_RUNTIME_LIMIT",
                "machine_genome.json",
                "MACHINE_GENOME.json",
                "worker-equilibrium.json",
                "fallback",
            },
        )
        self.assertGreaterEqual(args.runtime_count, 1)
        self.assertLessEqual(args.runtime_count, 8)

    def test_max_env_wins(self):
        with patch.dict(os.environ, {"HCLI_RESIDENT_RUNTIME_LIMIT": "2"}, clear=False):
            args = parse_hcli_args(["max", "hello"])
        self.assertEqual(args.runtime_count, 2)
        self.assertEqual(args.max_source, "HCLI_RESIDENT_RUNTIME_LIMIT")
        self.assertEqual(args.prompt, "hello")

    def test_max_genome_then_equilibrium_then_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            home = tmp_path / "home"
            workspace = tmp_path / "ws"
            workspace.mkdir()
            genome_dir = home / ".config" / "hcli"
            genome_dir.mkdir(parents=True)
            (genome_dir / "machine_genome.json").write_text(
                json.dumps(
                    {
                        "schema": "hcli.machine_genome.v1",
                        "generated_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        "measured_by": "tools/headless/machine_probe.py",
                        "resident_runtime_limit": 4,
                        "machine": live_machine_identity(),
                    }
                )
            )
            eq_dir = workspace / ".hcli-legacy" / "bootstrap-director-v6"
            eq_dir.mkdir(parents=True)
            (eq_dir / "worker-equilibrium.json").write_text(
                json.dumps({"bootstrap_workers": 3})
            )

            with patch.dict(os.environ, {"HOME": str(home)}, clear=False):
                os.environ.pop("HCLI_RESIDENT_RUNTIME_LIMIT", None)
                os.environ.pop("RESIDENT_RUNTIME_LIMIT", None)
                n, source = resolve_resident_runtime_limit(str(workspace))
                self.assertEqual(n, 4)
                self.assertEqual(source, "machine_genome.json")

                (genome_dir / "machine_genome.json").unlink()
                n, source = resolve_resident_runtime_limit(str(workspace))
                self.assertEqual(n, 3)
                self.assertEqual(source, "worker-equilibrium.json")

                empty = tmp_path / "empty"
                empty.mkdir()
                n, source = resolve_resident_runtime_limit(str(empty))
                self.assertEqual(n, 1)
                self.assertEqual(source, "fallback")


if __name__ == "__main__":
    unittest.main()
