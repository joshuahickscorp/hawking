#!/usr/bin/env python3
"""Deterministic tests for HAIDER positional CLI grammar."""
from __future__ import annotations

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from haider import parse_haider_args


class TestHaiderCliGrammar(unittest.TestCase):
    def test_default_interactive(self):
        args = parse_haider_args([])
        self.assertEqual(args.runtime_count, 1)
        self.assertIsNone(args.prompt)
        self.assertTrue(args.interactive)

    def test_n_interactive(self):
        args = parse_haider_args(["3"])
        self.assertEqual(args.runtime_count, 3)
        self.assertIsNone(args.prompt)
        self.assertTrue(args.interactive)

    def test_prompt_only(self):
        args = parse_haider_args(["fix the failing tests"])
        self.assertEqual(args.runtime_count, 1)
        self.assertEqual(args.prompt, "fix the failing tests")
        self.assertFalse(args.interactive)

    def test_n_and_prompt(self):
        args = parse_haider_args(["3", "fix the failing tests"])
        self.assertEqual(args.runtime_count, 3)
        self.assertEqual(args.prompt, "fix the failing tests")
        self.assertFalse(args.interactive)

    def test_invalid_n(self):
        with self.assertRaises(SystemExit):
            parse_haider_args(["0"])

    def test_invalid_n_negative(self):
        with self.assertRaises(SystemExit):
            parse_haider_args(["-1"])

    def test_non_numeric_first_arg_is_prompt(self):
        args = parse_haider_args(["read ULTRAGOAL.md and execute it"])
        self.assertEqual(args.runtime_count, 1)
        self.assertEqual(args.prompt, "read ULTRAGOAL.md and execute it")

    def test_debug_flag(self):
        args = parse_haider_args(["--debug"])
        self.assertTrue(args.debug)
        self.assertEqual(args.runtime_count, 1)

    def test_debug_with_n(self):
        args = parse_haider_args(["2", "--debug"])
        self.assertEqual(args.runtime_count, 2)
        self.assertTrue(args.debug)

    def test_debug_with_prompt(self):
        args = parse_haider_args(["--debug", "do something"])
        self.assertTrue(args.debug)
        self.assertEqual(args.prompt, "do something")

    def test_model_flag(self):
        args = parse_haider_args(["--model", "/path/to/model.gguf"])
        self.assertEqual(args.model, "/path/to/model.gguf")

    def test_model_with_n(self):
        args = parse_haider_args(["2", "--model", "/path/to/model.gguf"])
        self.assertEqual(args.runtime_count, 2)
        self.assertEqual(args.model, "/path/to/model.gguf")

    def test_max_cycles_flag(self):
        args = parse_haider_args(["--max-cycles", "5"])
        self.assertEqual(args.max_cycles, 5)

    def test_workspace_flag(self):
        args = parse_haider_args(["--workspace", "/tmp/test"])
        self.assertEqual(args.workspace, "/tmp/test")

    def test_combined_n_prompt_debug_model(self):
        args = parse_haider_args(["3", "fix bug", "--debug", "--model", "/m.gguf"])
        self.assertEqual(args.runtime_count, 3)
        self.assertEqual(args.prompt, "fix bug")
        self.assertTrue(args.debug)
        self.assertEqual(args.model, "/m.gguf")

    def test_empty_args(self):
        args = parse_haider_args([])
        self.assertEqual(args.runtime_count, 1)
        self.assertIsNone(args.prompt)
        self.assertTrue(args.interactive)


if __name__ == "__main__":
    unittest.main()