"""A census unit must be impossible to pass by sounding confident.

This is the gate that decides whether HCLI's survey output can be trusted
without re-reading it. If a body can pass by emitting a fluent paragraph, or by
echoing a parameter count it made up, then the survey saves nobody any time and
the whole point of moving this work to the resident is lost.

Two failure modes are pinned, both with negative controls so the checks cannot
be vacuous:

  * PROSE. A reply with no JSON object fails, no matter how good the prose is.
  * INVENTION. A reply that echoes a measured number WRONG fails, even when
    every required key is present and the disposition is valid.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hcli.odyssey_census import (EXECUTION_CLASSES, FAMILIES, REQUIRED, Unit,
                                 parse_verdict, run_unit, verify, write_receipt)

FACTS = {"n_tensors": 311, "tensor_gib": 1.4, "on_disk_gib": 1.4,
         "matrix_params_b": 0.5959, "dtype_bytes": {"BF16": 1}, "distinct_roles": 9,
         "repeated_blocks": {"model.layers.<L>.mlp.gate_proj.weight": 28},
         "complete_ebpw_if_bf16": 16.0}


def _good(**over):
    v = {"architecture_family": "DENSE_DECODER",
         "execution_class": "AUTOREGRESSIVE_DECODE_WITH_KV",
         "dominant_organs": "mlp gate/up/down across 28 layers",
         "state_or_kv": "standard attention KV, grows with context",
         "likely_bottleneck": "memory bandwidth at decode",
         "relevant_laws": ["complete EBPW includes scales and zeros"],
         "disposition": "RESEARCH SPECIMEN",
         "next_experiment": "matched-EBPW grouped affine sweep at 4 and 3 bits"}
    v.update(over)
    return v


class TestProseCannotPass(unittest.TestCase):
    def test_a_fluent_paragraph_with_no_json_fails(self):
        text = ("This specimen is a dense causal decoder of roughly 0.6B "
                "parameters. Its dominant organs are the MLP projections and it "
                "should be bandwidth bound at decode. I would recommend a "
                "grouped affine sweep as the next experiment.")
        verdict, why = parse_verdict(text)
        self.assertIsNone(verdict)
        self.assertIn("prose does not satisfy", why)

    def test_json_wrapped_in_prose_or_a_fence_still_passes(self):
        # The negative control for the check above: refusing every dialect
        # would be measuring the wrapper rather than the judgement.
        for wrap in ("Here is my answer:\n```json\n{J}\n```",
                     "{J}", "<tool_call>{J}</tool_call>"):
            text = wrap.replace("{J}", json.dumps(_good()))
            verdict, why = parse_verdict(text)
            self.assertIsNotNone(verdict, f"a valid verdict was rejected: {why}")

    def test_a_unit_whose_body_only_talks_is_failed(self):
        u = Unit(specimen="x", facts=FACTS)
        run_unit(u, lambda _p: "I think it is probably a dense transformer.", attempts=2)
        self.assertEqual(u.status, "failed")
        self.assertIn("prose", u.reason)


class TestInventedNumbersCannotPass(unittest.TestCase):
    def test_a_wrong_echoed_param_count_fails(self):
        u = Unit(specimen="x", facts=FACTS)
        ok, why = verify(u, _good(matrix_params_b=7.0))
        self.assertFalse(ok, "a survey that invented a 7B parameter count passed")
        self.assertIn("invented a number", why)

    def test_the_correct_echoed_number_passes(self):
        # Negative control: if echoing the RIGHT number also failed, the check
        # would just be banning the field.
        u = Unit(specimen="x", facts=FACTS)
        ok, why = verify(u, _good(matrix_params_b=0.5959))
        self.assertTrue(ok, why)

    def test_not_echoing_the_number_at_all_is_allowed(self):
        u = Unit(specimen="x", facts=FACTS)
        ok, why = verify(u, _good())
        self.assertTrue(ok, why)

    def test_every_required_key_is_actually_required(self):
        for key in REQUIRED:
            v = _good()
            v.pop(key)
            ok, why = verify(Unit(specimen="x", facts=FACTS), v)
            self.assertFalse(ok, f"a verdict missing {key} was accepted")
            self.assertIn(key, why)

    def test_an_off_list_architecture_family_fails(self):
        # The whole reason the vocabulary was closed. Two agents answered with
        # the config's own class name -- "DreamModel", "ILLaDA" -- which is not
        # a family, and the open-vocabulary gate had no way to say so.
        ok, why = verify(Unit(specimen="x", facts=FACTS),
                         _good(architecture_family="DreamModel"))
        self.assertFalse(ok, "'DreamModel' was accepted as an architecture family")
        self.assertIn("not one of the", why)

    def test_an_off_list_execution_class_fails(self):
        ok, why = verify(Unit(specimen="x", facts=FACTS),
                         _good(execution_class="instruct"))
        self.assertFalse(ok, "'instruct' was accepted as an execution class")
        self.assertIn("not one of the", why)

    def test_every_listed_value_is_actually_accepted(self):
        # Negative control: a closed set that rejects its own members would be
        # a gate that never opens.
        for fam in FAMILIES:
            ok, why = verify(Unit(specimen="x", facts=FACTS),
                             _good(architecture_family=fam))
            self.assertTrue(ok, f"{fam} is on the list and was rejected: {why}")
        for ex in EXECUTION_CLASSES:
            ok, why = verify(Unit(specimen="x", facts=FACTS),
                             _good(execution_class=ex))
            self.assertTrue(ok, f"{ex} is on the list and was rejected: {why}")

    def test_an_invented_disposition_fails(self):
        ok, why = verify(Unit(specimen="x", facts=FACTS),
                         _good(disposition="LOOKS PROMISING"))
        self.assertFalse(ok)
        self.assertIn("not one of the accepted terms", why)

    def test_a_non_answer_next_experiment_fails(self):
        ok, _ = verify(Unit(specimen="x", facts=FACTS), _good(next_experiment="tbd"))
        self.assertFalse(ok, "'tbd' was accepted as the next experiment")


class TestAnAcceptedUnitIsRecordedAsHcliAuthored(unittest.TestCase):
    def test_the_receipt_says_who_wrote_it(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            u = Unit(specimen="Qwen--Qwen3-0.6B@abc", facts=FACTS,
                     verdict=_good(), status="accepted")
            p = write_receipt(repo, u)
            rec = json.loads(p.read_text())
            self.assertEqual(rec["author"], "hcli",
                             "a receipt HCLI produced must not be attributable to the supervisor")
            self.assertEqual(rec["measured_facts"], FACTS,
                             "the receipt must carry the facts the verdict was checked against")


if __name__ == "__main__":
    unittest.main()
