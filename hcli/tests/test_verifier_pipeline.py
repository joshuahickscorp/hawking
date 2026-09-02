from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.verifier_pipeline import (
    AGENT_ROLES,
    Obligation,
    PlanError,
    Verdict,
    execute,
    plan,
    run_pipeline,
    synthesize,
    verify,
)

import hcli.verifier_pipeline as _vp

MODULE = Path(_vp.__file__).resolve()


class FakeCaller:
    def __init__(self, handler):
        self.n = 0
        self.prompts = []
        self.handler = handler

    def __call__(self, prompt: str, *, schema=None):
        self.n += 1
        self.prompts.append(prompt)
        return self.handler(prompt, schema)


def _ob(**kwargs) -> Obligation:
    base = dict(
        id="o1",
        statement="foo.py exports add",
        angles=["read foo.py"],
        consequential=False,
        agent_role="read",
    )
    base.update(kwargs)
    return Obligation(**base)


class TestVerifierPipeline(unittest.TestCase):
    def test_module_does_not_import_engine(self):
        src = MODULE.read_text(encoding="utf-8")
        self.assertNotIn("hcli.engine", src)
        self.assertNotIn("from .engine", src)
        self.assertNotIn("import engine", src)

    def test_agent_roles_are_a_closed_set(self):
        expected = {"locate", "read", "test", "research", "reason", "settle", "general"}
        self.assertEqual(set(AGENT_ROLES), expected)

    def test_plan_retries_then_accepts_a_proposition(self):
        def handler(prompt, schema):
            if handler.n == 0:
                handler.n += 1
                return {
                    "obligations": [
                        {
                            "id": "retry",
                            "statement": "Implement the retry logic",
                            "angles": ["read x.py"],
                            "consequential": False,
                            "agent_role": "locate",
                        }
                    ]
                }
            handler.n += 1
            return {
                "obligations": [
                    {
                        "id": "retry",
                        "statement": "Retry logic is present in x.py",
                        "angles": ["read x.py"],
                        "consequential": False,
                        "agent_role": "locate",
                    }
                ]
            }

        handler.n = 0
        caller = FakeCaller(handler)
        obs = plan("add retries", caller)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].statement, "Retry logic is present in x.py")
        self.assertEqual(obs[0].agent_role, "locate")
        self.assertGreaterEqual(caller.n, 2)

    def test_plan_raises_when_retry_still_imperative(self):
        def handler(prompt, schema):
            return {
                "obligations": [
                    {
                        "id": "x",
                        "statement": "Fix the off-by-one",
                        "angles": ["read y.py"],
                        "consequential": True,
                        "agent_role": "settle",
                    }
                ]
            }

        with self.assertRaises(PlanError):
            plan("fix it", FakeCaller(handler))

    def test_imperative_heuristic_is_first_token_only(self):
        """Limits: only the first token is checked against a small verb list.

        'Implementing retry is complete' and 'Additional coverage exists'
        are NOT rejected — the heuristic is cheap, not a parser.
        """
        good = {
            "obligations": [
                {
                    "id": "a",
                    "statement": "Implementing retry is complete",
                    "angles": ["read a.py"],
                    "consequential": False,
                    "agent_role": "read",
                },
                {
                    "id": "b",
                    "statement": "Additional coverage exists",
                    "angles": ["read b.py"],
                    "consequential": False,
                    "agent_role": "read",
                },
            ]
        }
        obs = plan("g", FakeCaller(lambda p, s: good))
        self.assertEqual([o.id for o in obs], ["a", "b"])

    def test_plan_maps_sa_agent_types(self):
        raw = {
            "obligations": [
                {
                    "id": "g",
                    "statement": "foo is defined in bar.py",
                    "angles": ["rg foo"],
                    "consequential": False,
                    "agentType": "sa-grep",
                }
            ]
        }
        obs = plan("g", FakeCaller(lambda p, s: raw))
        self.assertEqual(obs[0].agent_role, "locate")

    def test_plan_empty_raises(self):
        with self.assertRaises(PlanError):
            plan("g", FakeCaller(lambda p, s: {"obligations": []}))

    def test_plan_respects_max_obligations(self):
        raw = {
            "obligations": [
                {
                    "id": f"o{i}",
                    "statement": f"claim {i} holds",
                    "angles": ["read x.py"],
                    "consequential": False,
                    "agent_role": "read",
                }
                for i in range(20)
            ]
        }
        obs = plan("g", FakeCaller(lambda p, s: raw), max_obligations=3)
        self.assertEqual(len(obs), 3)

    def test_execute_joins_fan_angles(self):
        def handler(prompt, schema):
            return "ref:" + prompt.split("ATTACK IT THIS WAY:")[-1].split("\n", 1)[0].strip()

        caller = FakeCaller(handler)
        ob = _ob(angles=["angle-a", "angle-b", "angle-c"], agent_role="read")
        out = execute(ob, caller, fan=2)
        self.assertIsInstance(out, str)
        self.assertIn("---", out)
        self.assertEqual(caller.n, 2)

    def test_execute_test_role_fast_path_with_runner(self):
        ran = []

        def run_command(cmd):
            ran.append(cmd)
            return 0, "seven"

        caller = FakeCaller(lambda p, s: "nope")
        ob = Obligation(
            id="t",
            statement="python3 prints 7",
            angles=["python3 -c 'print(7)'"],
            consequential=False,
            agent_role="test",
        )
        out = execute(ob, caller, run_command=run_command)
        self.assertEqual(caller.n, 0)
        self.assertEqual(ran, ["python3 -c 'print(7)'"])
        self.assertIn("seven", out)
        self.assertIn("exit_code=0", out)

    def test_verify_empty_command_skips_runner(self):
        ran = []
        v = verify(
            _ob(agent_role="reason"),
            "ev",
            FakeCaller(lambda p, s: {"command": "  "}),
            lambda cmd: ran.append(cmd) or (0, "x"),
        )
        self.assertEqual(v.verdict, "UNVERIFIABLE")
        self.assertEqual(ran, [])

    def test_verify_does_not_trust_model_output_field(self):
        def handler(prompt, schema):
            if "MECHANICALLY" in prompt:
                return {"command": "python3 -c 'print(42)'", "output": "INVENTED"}
            return {"verdict": "TRUE", "output": "INVENTED"}

        v = verify(
            _ob(agent_role="settle"),
            "ev",
            FakeCaller(handler),
            lambda cmd: (0, "forty-two-real"),
        )
        self.assertIn("forty-two-real", v.output)
        self.assertNotIn("INVENTED", v.output)

    def test_verify_override_true_on_nonzero(self):
        def handler(prompt, schema):
            if "MECHANICALLY" in prompt:
                # A real admissible command; `false` is refused as vacuous.
                return {"command": "python3 -c 'import sys; sys.exit(2)'"}
            return {"verdict": "TRUE"}

        v = verify(_ob(), "ev", FakeCaller(handler), lambda cmd: (2, "nope"))
        self.assertEqual(v.verdict, "FALSE")
        self.assertIn("nope", v.output)

    def test_verify_does_not_promote_false_on_zero_exit(self):
        """A passing unrelated command does not flip FALSE up to TRUE.

        Previously this used the tautology ``true``. That command is now
        refused before run; the property under test is promotion, so the
        command must be admissible and actually return exit 0.
        """

        def handler(prompt, schema):
            if "MECHANICALLY" in prompt:
                return {"command": "python3 -c 'print(0)'"}
            return {"verdict": "FALSE"}

        v = verify(_ob(), "ev", FakeCaller(handler), lambda cmd: (0, "ok"))
        self.assertEqual(v.verdict, "FALSE")
        self.assertIn("ok", v.output)

    def test_synthesize_signature_has_no_runner(self):
        sig = inspect.signature(synthesize)
        self.assertNotIn("run_command", sig.parameters)
        caller = FakeCaller(lambda p, s: "answer")
        text = synthesize(
            "goal",
            [
                Verdict(
                    obligation_id="a",
                    verdict="TRUE",
                    command="python3 -c 'print(1)'",
                    output="1",
                    evidence="ref",
                ),
                Verdict(
                    obligation_id="b",
                    verdict="UNVERIFIABLE",
                    command="",
                    output="no command",
                    evidence="",
                ),
            ],
            caller,
        )
        self.assertEqual(text, "answer")
        joined = "\n".join(caller.prompts)
        self.assertIn("launder a guess into a finding", joined)
        self.assertIn("real command", joined.lower())
        self.assertIn("judgement", joined.lower())

    def test_run_pipeline_carries_verify_failure(self):
        def handler(prompt, schema):
            if "Not a task list" in prompt or "Decompose this" in prompt:
                return {
                    "obligations": [
                        {
                            "id": "only",
                            "statement": "foo is exported",
                            "angles": ["read f.py"],
                            "consequential": False,
                            "agent_role": "read",
                        }
                    ]
                }
            if "concrete refs" in prompt.lower() or "ATTACK IT THIS WAY" in prompt:
                return "/abs/f.py:1: foo"
            if "MECHANICALLY" in prompt:
                raise RuntimeError("verify exploded")
            if "launder a guess" in prompt:
                return "synth"
            return "x"

        result = run_pipeline("g", FakeCaller(handler), lambda cmd: (0, "ok"))
        self.assertEqual(len(result["verdicts"]), 1)
        self.assertEqual(result["verdicts"][0].verdict, "UNVERIFIABLE")
        self.assertIn("explod", result["verdicts"][0].output.lower())

    def test_plan_prompt_quotes_source_shape(self):
        caller = FakeCaller(
            lambda p, s: {
                "obligations": [
                    {
                        "id": "z",
                        "statement": "the suite exits 0",
                        "angles": ["pytest -q"],
                        "consequential": True,
                        "agent_role": "test",
                    }
                ]
            }
        )
        plan("g", caller)
        joined = "\n".join(caller.prompts)
        self.assertIn("TRUE or FALSE", joined)
        self.assertIn("Not a task list", joined)
        self.assertIn("most decisive first", joined.lower())
