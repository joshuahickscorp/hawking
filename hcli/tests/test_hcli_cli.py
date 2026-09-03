#!/usr/bin/env python3
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hcli.cli import parse_hcli_args

class TestCli(unittest.TestCase):
    def test_default(self):
        a = parse_hcli_args([])
        self.assertEqual(a.runtime_count, 1)
        self.assertIsNone(a.prompt)
        self.assertTrue(a.interactive)

    def test_n(self):
        a = parse_hcli_args(["3"])
        self.assertEqual(a.runtime_count, 3)
        self.assertTrue(a.interactive)

    def test_prompt(self):
        a = parse_hcli_args(["fix tests"])
        self.assertEqual(a.runtime_count, 1)
        self.assertEqual(a.prompt, "fix tests")
        self.assertFalse(a.interactive)

    def test_n_prompt(self):
        a = parse_hcli_args(["3", "fix tests"])
        self.assertEqual(a.runtime_count, 3)
        self.assertEqual(a.prompt, "fix tests")

    def test_invalid_n(self):
        with self.assertRaises(SystemExit):
            parse_hcli_args(["0"])

if __name__ == "__main__":
    unittest.main()
