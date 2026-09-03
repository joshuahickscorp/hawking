"""The output contract and the verifier must agree that tests are mandatory.

They did not, and it cost the campaign every accepted goal.

`_validate` is unambiguous:

    if not test_list:
        result["ok"] = False
        result["reason"] = "NO_EVIDENCE"

and NO_EVIDENCE maps to status "unverified", which is terminal. So a mutation
that names no test is thrown away however good it is.

The system prompt told the model the opposite -- `"tests": ["optional safe
workspace-relative Python test paths"]`. The model believed it, sent no tests,
and every mutation landed unverified. Live state when this was found: seven
work units, zero completed, accepted=0, across a mission that had been running
for hours with a healthy resident and a working repair path.

This is a correlated verifier: the prompt and the validator encode the same
rule in two places, so either one can be relaxed without the other noticing.
This test is the thing that notices.
"""
from __future__ import annotations

import inspect
import unittest

from hcli.engine import _SYSTEM_PROMPT, Engine


class TestContractMatchesVerifier(unittest.TestCase):
    def test_the_verifier_still_refuses_an_empty_test_list(self):
        """The rule this test exists to protect. If this changes, so must the prompt."""
        src = inspect.getsource(Engine._validate)
        self.assertIn("if not test_list:", src)
        self.assertIn('result["ok"] = False', src)
        self.assertIn('"NO_EVIDENCE"', src)

    def test_the_prompt_does_not_call_tests_optional(self):
        """The exact wording that caused it."""
        self.assertNotIn(
            "optional safe workspace-relative", _SYSTEM_PROMPT,
            "the contract advertises tests as optional while the verifier requires them",
        )

    def test_the_prompt_states_the_consequence_of_an_empty_test_list(self):
        """Telling the model a rule without its consequence did not change behaviour."""
        self.assertIn("EMPTY", _SYSTEM_PROMPT.upper())
        self.assertIn("NO_EVIDENCE", _SYSTEM_PROMPT)
        self.assertIn("UNVERIFIED", _SYSTEM_PROMPT.upper())

    def test_the_mutation_example_names_a_real_test_path(self):
        """An example is what the model copies, so the example must be valid."""
        self.assertIn('"tests": ["hcli/', _SYSTEM_PROMPT)
        self.assertNotIn('"tests": []\n  ,', _SYSTEM_PROMPT)

    def test_read_only_answers_still_need_no_tests(self):
        """Negative control: only MUTATIONS require evidence.

        An `answer` has nothing to verify, and demanding a test for one would
        make every read-only question unanswerable.
        """
        answer_block = _SYSTEM_PROMPT.split('"kind": "answer"')[1].split("}")[0]
        self.assertIn('"tests": []', answer_block)


if __name__ == "__main__":
    unittest.main()
