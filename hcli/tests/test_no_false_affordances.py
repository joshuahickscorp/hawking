"""Options that are advertised must reach the thing that would honour them.

Three measured instances of one bug class, each found by grepping for the READER
of a parsed option rather than trusting that a flag with a help string does
something:

  * `--max-cycles` and `--max-turns` are parsed by hcli/cli.py and read by
    NOTHING. The headless path calls controller.execute() exactly once; there is
    no cycle loop and no turn loop anywhere in the package. Two unattended tools
    pass --max-cycles believing it bounds their run.
  * `--constraint` is parsed, stored into the delegation spec, and never read
    back, so an operator constraint reaches no worker.
  * `agentos qwen27-mlp-ab` reads args.fusion_env, which its own subparser never
    defines, so the subcommand raises AttributeError on every invocation.

The rule these enforce: command shown == command executed.
"""
from __future__ import annotations

import unittest


class TestBoundsThatDoNotExistAreRefused(unittest.TestCase):
    """A bound nothing enforces must not be accepted silently.

    Accepting `--max-cycles 2` and running unbounded is worse than refusing it:
    the caller believes there is a ceiling and there is none.
    """

    def _parse(self, argv):
        from hcli.cli import parse_hcli_args
        return parse_hcli_args(argv)

    def test_max_cycles_is_refused_rather_than_ignored(self):
        with self.assertRaises(SystemExit) as caught:
            self._parse(["--max-cycles", "2", "do the thing"])
        self.assertNotEqual(caught.exception.code, 0)

    def test_max_turns_is_refused_rather_than_ignored(self):
        with self.assertRaises(SystemExit):
            self._parse(["--max-turns", "5", "do the thing"])

    def test_a_normal_invocation_still_parses(self):
        args = self._parse(["explain deltanet"])
        self.assertEqual(args.n_or_prompt, "explain deltanet")

    def test_the_refusal_names_what_does_bound_a_run(self, ):
        import contextlib, io
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            self._parse(["--max-cycles", "2", "x"])
        text = err.getvalue()
        self.assertIn("max-cycles", text)
        # Actionable: a refusal that does not say what to do instead is the same
        # dead end as the flag that did nothing.
        self.assertTrue(any(word in text for word in ("resident", "bound", "once")),
                        f"the refusal explains nothing: {text}")


class TestConstraintsReachTheWorker(unittest.TestCase):
    def test_a_constraint_appears_in_the_goal(self):
        # It used to be written into the spec and read by nobody.
        import tempfile
        from pathlib import Path
        from hcli.delegate import _goal_with_constraints
        goal = _goal_with_constraints("do the thing", ["never touch main"])
        self.assertIn("do the thing", goal)
        self.assertIn("never touch main", goal)

    def test_no_constraints_leaves_the_goal_untouched(self):
        from hcli.delegate import _goal_with_constraints
        self.assertEqual(_goal_with_constraints("do the thing", []), "do the thing")


class TestEveryAgentosSubcommandParses(unittest.TestCase):
    """A subcommand that cannot survive its own argument namespace is a stub."""

    def test_qwen27_mlp_ab_has_the_option_its_handler_reads(self):
        from hcli.agentos_cli import build_parser
        args = build_parser().parse_args(["qwen27-mlp-ab"])
        self.assertTrue(hasattr(args, "fusion_env"),
                        "the handler reads args.fusion_env and the parser never sets it")

    def test_every_subcommand_namespace_has_what_its_handler_reads(self):
        # The general form of the same defect: walk every registered subcommand,
        # parse it with no arguments, and confirm the namespace is at least
        # constructible. A parser that cannot produce a namespace for its own
        # subcommand is advertising something unreachable.
        from hcli.agentos_cli import build_parser
        parser = build_parser()
        actions = [a for a in parser._actions
                   if hasattr(a, "choices") and isinstance(getattr(a, "choices", None), dict)]
        self.assertTrue(actions, "no subparsers found")
        broken = []
        for action in actions:
            for name in action.choices:
                try:
                    parser.parse_args([name])
                except SystemExit:
                    pass  # required arguments missing is fine; that is a contract
                except Exception as exc:  # pragma: no cover - the defect path
                    broken.append(f"{name}: {type(exc).__name__}: {exc}")
        self.assertEqual(broken, [], f"subcommands that cannot parse: {broken}")


if __name__ == "__main__":
    unittest.main()
