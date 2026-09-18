"""KRT001 -- an attention backend must PRESERVE THE VALUE HEAD GEOMETRY.

MLA (DeepSeek-V2/V3, Kimi-VL, and every compatible body) gives attention a value
head narrower than its query/key head: v_head_dim 128 against a qk_head_dim of
qk_nope 128 + qk_rope 64 = 192. The attention output must therefore be
num_heads * v_head_dim, and o_proj is built expecting exactly that.

Torch's SDPA on MPS does not honour it. Measured on this machine:

    cpu/fp32/sdpa   OK        mps/fp32/sdpa   WRONG (num_heads*192)
    cpu/fp32/eager  OK        mps/bf16/sdpa   WRONG
                              mps/*/eager     OK

HOW THIS TEST IS WRITTEN, AND WHY IT MATTERS
--------------------------------------------
The defect was FOUND because o_proj raised a shape error. That is luck, not a
check: it only raises because 16*192 happens to differ from o_proj's in_features.
A body whose v_head_dim equalled its qk_head_dim would take the same wrong path
and return WRONG NUMBERS in silence, and so would any backend that pads and
forgets to slice back.

So this asserts the GEOMETRY DIRECTLY -- a pre-hook on o_proj captures the tensor
actually handed to it and compares it against num_heads * v_head_dim. It never
relies on an exception being raised downstream, which is the whole point: the
same assertion works on a geometry where nothing would raise.

The negative control is a SYMMETRIC-head config. If sdpa failed there too, this
would be measuring "sdpa is broken" rather than "sdpa loses the value geometry",
and the selection rule derived from it would be wrong.

    python3 tools/future/test_attention_geometry_asymmetric_heads.py
    python3 -m pytest tools/future/test_attention_geometry_asymmetric_heads.py
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch
from transformers import DeepseekV3Config
from transformers.models.deepseek_v3 import DeepseekV3ForCausalLM

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "receipts" / "future" / "ATTENTION_GEOMETRY_SELECTION.json"

# Kimi-VL-A3B / DeepSeek-V3 geometry, shrunk everywhere the head dims are not
# involved so this needs no checkpoint and runs in seconds.
ASYMMETRIC = dict(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
                  num_attention_heads=16, kv_lora_rank=512, q_lora_rank=None)
# The control keeps rope INTACT and widens the value head to match, so the only
# difference from ASYMMETRIC is the property under test. Setting rope to 0
# instead would break the rotary split and crash before o_proj, which is a
# different failure wearing the control's name -- it did exactly that once.
SYMMETRIC = dict(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=192,
                 num_attention_heads=16, kv_lora_rank=512, q_lora_rank=None)


def _model(head_geometry, device, dtype):
    cfg = DeepseekV3Config(
        hidden_size=2048, intermediate_size=64, moe_intermediate_size=64,
        num_hidden_layers=1, first_k_dense_replace=1, n_routed_experts=4,
        num_experts_per_tok=2, n_shared_experts=1, vocab_size=256,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        num_key_value_heads=head_geometry["num_attention_heads"],
        **head_geometry)
    return cfg, DeepseekV3ForCausalLM(cfg).eval().to(device, dtype)


def observed_o_proj_input_dim(head_geometry, device, dtype, impl):
    """The tensor attention actually hands to o_proj. A pre-hook, so the answer
    does not depend on whether o_proj would have accepted it."""
    cfg, m = _model(head_geometry, device, dtype)
    seen: list[int] = []
    h = m.model.layers[0].self_attn.o_proj.register_forward_pre_hook(
        lambda _m, inp: seen.append(inp[0].shape[-1]))
    m.set_attn_implementation(impl)
    try:
        with torch.no_grad():
            m(torch.randint(3, 256, (1, 5)).to(device))
    except RuntimeError:
        pass          # a raise is one way to fail; the hook already recorded why
    finally:
        h.remove()
    return (seen[0] if seen else None), cfg


def _devices():
    yield "cpu", torch.float32
    if torch.backends.mps.is_available():
        yield "mps", torch.float32
        yield "mps", torch.bfloat16


class TestValueHeadGeometryIsPreserved(unittest.TestCase):
    def test_every_implementation_is_measured_not_assumed(self):
        """The law: never assume a backend preserves V geometry because the call
        returned. Every (device, dtype, impl) is recorded, safe or not."""
        rows = []
        for device, dtype in _devices():
            for impl in ("eager", "sdpa"):
                got, cfg = observed_o_proj_input_dim(ASYMMETRIC, device, dtype, impl)
                want = cfg.num_attention_heads * cfg.v_head_dim
                rows.append({"device": device, "dtype": str(dtype).split(".")[-1],
                             "implementation": impl, "expected_dim": want,
                             "observed_dim": got, "safe": got == want})
        self.assertTrue(rows, "no implementation was measured at all")

        # eager must be safe everywhere, or the fallback the loader relies on is
        # itself unsound and the whole selection rule collapses.
        for r in rows:
            if r["implementation"] == "eager":
                self.assertTrue(r["safe"],
                                f"eager lost the value geometry on {r['device']}/"
                                f"{r['dtype']}: got {r['observed_dim']}, "
                                f"expected {r['expected_dim']}")

        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.write_text(json.dumps({
            "schema": "hawking.future.attention_geometry.v1",
            "obligation": "KRT001", "steer": "S014",
            "law": "an attention backend must return num_heads * v_head_dim. "
                   "Measured per implementation, never assumed from a successful call.",
            "geometry": {"qk_head_dim": ASYMMETRIC["qk_nope_head_dim"] + ASYMMETRIC["qk_rope_head_dim"],
                         "v_head_dim": ASYMMETRIC["v_head_dim"],
                         "num_heads": ASYMMETRIC["num_attention_heads"]},
            "affects": ["DeepSeek-V2 family", "DeepSeek-V3 family", "Kimi-VL MLA bodies",
                        "any architecture with asymmetric QK/V head dims"],
            "measured": rows,
            # Per DEVICE. A union across devices would read "sdpa is safe"
            # because cpu/sdpa is, which is the opposite of the finding.
            "safe_implementations_by_device": {
                d: sorted({r["implementation"] for r in rows
                           if r["device"] == d and r["safe"]})
                for d in sorted({r["device"] for r in rows})},
            "unsafe": [f"{r['device']}/{r['dtype']}/{r['implementation']}"
                       for r in rows if not r["safe"]],
            "note": "This is measured directly from the tensor handed to o_proj. It does "
                    "NOT depend on a shape error being raised -- on a body whose "
                    "v_head_dim equalled its qk_head_dim the same defect would return "
                    "wrong numbers in silence and no exception would appear.",
        }, indent=1))

    def test_mps_sdpa_is_the_one_that_loses_it(self):
        """The specific finding, pinned so a torch upgrade that fixes it is
        NOTICED rather than silently changing which path Hawking selects."""
        if not torch.backends.mps.is_available():
            self.skipTest("no MPS on this host")
        got, cfg = observed_o_proj_input_dim(ASYMMETRIC, "mps", torch.float32, "sdpa")
        want = cfg.num_attention_heads * cfg.v_head_dim
        qk = cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)
        if got == want:
            self.skipTest(f"MPS sdpa now preserves the value geometry (torch "
                          f"{torch.__version__}); the loader's eager fallback can be "
                          f"re-examined rather than assumed still necessary")
        self.assertEqual(got, qk,
                         "MPS sdpa returned neither the value geometry nor the query "
                         f"geometry ({got}); the diagnosis on record is wrong")


class TestGeometryIsNotCorrectness(unittest.TestCase):
    """A matching shape is not a matching answer, and "the shapes lined up" is
    precisely the assumption this law exists to forbid. On the SYMMETRIC config
    MPS sdpa returns the right DIMENSION -- but only because the query head and
    the value head happen to be the same width there, which is the geometry where
    the defect would be invisible. So the values are compared numerically against
    a CPU eager reference, which is the arm that could catch a backend that
    silently attends over the wrong tensor."""

    def _logits(self, geometry, device, dtype, impl, seed=0):
        torch.manual_seed(seed)
        cfg, m = _model(geometry, device, dtype)
        m.set_attn_implementation(impl)
        with torch.no_grad():
            return m(torch.arange(3, 11).unsqueeze(0).to(device)).logits.float().cpu()

    def test_mps_sdpa_agrees_numerically_where_the_dims_match(self):
        if not torch.backends.mps.is_available():
            self.skipTest("no MPS on this host")
        ref = self._logits(SYMMETRIC, "cpu", torch.float32, "eager")
        got = self._logits(SYMMETRIC, "mps", torch.float32, "sdpa")
        delta = float((got - ref).abs().max())
        scale = float(ref.abs().max()) or 1.0
        self.assertLess(delta / scale, 5e-3,
                        f"MPS sdpa produced the right SHAPE on symmetric heads but "
                        f"different NUMBERS (max rel {delta/scale:.5f}). The defect is "
                        "then wider than a geometry bug and eager must be forced for "
                        "this family regardless of head widths.")

    def test_the_numeric_check_can_actually_fail(self):
        # Negative control for the comparison itself: two different seeds must
        # disagree, or the comparison is insensitive and proves nothing.
        a = self._logits(SYMMETRIC, "cpu", torch.float32, "eager", seed=0)
        b = self._logits(SYMMETRIC, "cpu", torch.float32, "eager", seed=1)
        scale = float(a.abs().max()) or 1.0
        self.assertGreater(float((a - b).abs().max()) / scale, 5e-3,
                           "two differently-seeded models produced the same logits, so "
                           "the numeric comparison cannot detect a wrong answer")


class TestTheCheckDiscriminates(unittest.TestCase):
    """Negative control. Without it, the rule derived above could be 'sdpa is
    broken' rather than 'sdpa loses the VALUE geometry'."""

    def test_symmetric_heads_are_safe_on_every_implementation(self):
        if not torch.backends.mps.is_available():
            self.skipTest("no MPS on this host")
        for impl in ("eager", "sdpa"):
            got, cfg = observed_o_proj_input_dim(SYMMETRIC, "mps", torch.float32, impl)
            want = cfg.num_attention_heads * cfg.v_head_dim
            self.assertIsNotNone(got, f"{impl} never reached o_proj on the SYMMETRIC "
                                      "config, so this control measured nothing")
            self.assertEqual(got, want,
                             f"{impl} failed on SYMMETRIC heads too, so the asymmetric "
                             "result is not evidence about value geometry")

    def test_the_probe_would_notice_a_wrong_dimension(self):
        """A hook that never fires would make every assertion above vacuous."""
        got, cfg = observed_o_proj_input_dim(ASYMMETRIC, "cpu", torch.float32, "eager")
        self.assertIsNotNone(got, "the pre-hook never fired; nothing was measured")
        self.assertNotEqual(cfg.v_head_dim,
                            cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
                            "the ASYMMETRIC config is not actually asymmetric, so this "
                            "whole file tests nothing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
