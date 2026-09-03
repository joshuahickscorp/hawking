"""The sealed profile's capabilities must survive a directory launch.

The resident is launched against an artifact DIRECTORY:

    ascension_qwen38_resident --artifact-root /Users/.../NOETIC_PARENT_A

`config_for_model_path` handled that branch with `HawkingNativeConfig.defaults()`,
which calls itself resident_identity "sealed-3.14" and carries capabilities={}.
So `supports("grammar")` was False for the very resident whose profile declares
grammar "supported", and the grammar field was stripped before the request left
Python.

The channel is real: the resident parses `grammar: "json"`, masks logits through
json_constrain, and returns `grammar_enforced`. It was unreachable in the only
configuration that ships. Every receipt reported grammar_enforced as None, and
replies that could not have broken JSON did:

    the reply is NOT valid JSON -- the outermost object failed to decode
    (Expecting ',' delimiter: line 22 column 7)

response_format stays UNSUPPORTED and that is deliberate: the resident
constrains syntax, not schema, so the structured-output contract must keep
validating shape.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from hcli.hawking_native import HawkingNativeConfig, config_for_model_path

ARTIFACT_DIR = "/Users/scammermike/noetic/NOETIC_PARENT_A"
PROFILE = Path(__file__).resolve().parents[1] / "hawking-native.sealed-3.14.json"


class TestSealedCapabilities(unittest.TestCase):
    def test_the_profile_declares_the_grammar_channel(self):
        cfg = HawkingNativeConfig.from_file(str(PROFILE))
        features = (cfg.capabilities or {}).get("features") or {}
        self.assertEqual(features.get("grammar"), "supported")
        self.assertEqual(features.get("response_format"), "unsupported")

    @unittest.skipUnless(Path(ARTIFACT_DIR).is_dir(), "sealed artifact not on this box")
    def test_a_directory_launch_keeps_them(self):
        cfg = config_for_model_path(ARTIFACT_DIR)
        features = (cfg.capabilities or {}).get("features") or {}
        self.assertEqual(
            features.get("grammar"), "supported",
            "a directory launch dropped the sealed profile's capabilities",
        )

    @unittest.skipUnless(Path(ARTIFACT_DIR).is_dir(), "sealed artifact not on this box")
    def test_grammar_actually_reaches_the_wire(self):
        """The property that failed: not what is declared, what is SENT."""
        from hcli.backends import NoeticNativeBackend

        backend = NoeticNativeBackend(model_path=ARTIFACT_DIR)
        self.assertTrue(backend.supports("grammar"))
        prepared, degraded = backend._prepare_payload(
            {"messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 64, "grammar": "json"}
        )
        self.assertEqual(prepared.get("grammar"), "json")
        self.assertNotIn("grammar", degraded)

    @unittest.skipUnless(Path(ARTIFACT_DIR).is_dir(), "sealed artifact not on this box")
    def test_response_format_is_still_withheld(self):
        """Negative control: syntax masking is not schema enforcement.

        Declaring response_format supported would make the contract stop
        validating replies whose SHAPE the resident never checks.
        """
        from hcli.backends import NoeticNativeBackend

        backend = NoeticNativeBackend(model_path=ARTIFACT_DIR)
        self.assertFalse(backend.supports("response_format"))
        prepared, degraded = backend._prepare_payload(
            {"messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 64, "response_format": {"type": "json_object"}}
        )
        self.assertNotIn("response_format", prepared)
        self.assertIn("response_format", degraded)


if __name__ == "__main__":
    unittest.main()
