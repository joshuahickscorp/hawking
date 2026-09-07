"""The context budget must not overstate a Hawking-native artifact's window.

THE DEFECT: `_discover_ceiling` knows a GGUF header and a llama-server /props
port. A Hawking native artifact DIRECTORY is neither, so resolve() fell through
to `fallback:DEFAULT_PER_SLOT_CTX` and told the engine it had 32768 tokens with
24576 usable -- while the resident it was about to post to enforces 8192. A
15,545-token packet passed preflight and the runtime rejected it with "prompt is
15545 tokens and native max_seq_len is 8192; no generation token fits".

`native_profile_limits` already closed exactly this hole for the `.json` profile
form; its own docstring describes this failure. The directory form stayed open.

The authority was never missing. `config_for_model_path(artifact_dir)
.max_seq_len` returns 8192 for the same path the budget is handed. The budget
simply never asked.

Per the READY BANNER LAW: a live runtime's own limit outranks any file guess,
so it CAPS whatever else wins rather than merely competing with it -- an
override set above the runtime's limit would still be rejected by the runtime.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
import tempfile

from hcli.context_budget import resolve


def _native_artifact(root: Path) -> str:
    """A minimal Hawking-native artifact DIRECTORY, the form that regressed.

    Uses the REAL recogniser's marker -- `is_hawking_native_path` accepts a
    directory holding MIX_REPORT.json and catalog.hq38m20 -- so this exercises
    the production path rather than a stand-in for it. The window then comes
    from the native config's own default, which is the resident's 8192.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "MIX_REPORT.json").write_text(json.dumps({"architecture": "qwen3_next"}), encoding="utf-8")
    (root / "catalog.hq38m20").write_bytes(b"\0" * 16)
    (root / "tokenizer.json").write_text(json.dumps({"model": {}}), encoding="utf-8")
    return str(root)


class TestNativeArtifactCeiling(unittest.TestCase):
    def test_directory_artifact_does_not_fall_back_to_32768(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = _native_artifact(Path(d) / "ARTIFACT")
            budget = resolve(model_path=path, n_parallel=1)
            self.assertNotEqual(
                budget.source,
                "fallback:DEFAULT_PER_SLOT_CTX",
                "the budget went blind on a native artifact directory and guessed",
            )
            self.assertLessEqual(
                budget.total_ctx,
                8192,
                f"budget claimed {budget.total_ctx} against a resident that enforces 8192",
            )

    def test_usable_input_never_exceeds_what_the_runtime_accepts(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = _native_artifact(Path(d) / "ARTIFACT")
            budget = resolve(model_path=path, n_parallel=1)
            self.assertLess(
                budget.usable_input_tokens,
                8192,
                "usable input must leave room for generation inside the real window",
            )
            self.assertGreater(
                budget.usable_input_tokens,
                0,
                "a positive window must remain usable; refusing everything is not a fix",
            )
            # Getting the ceiling right and the reserves wrong trades one wrong
            # answer for another: 8192 - 4096 framing - 4096 generation is ZERO
            # usable, which refuses every packet rather than only oversized ones.
            self.assertEqual(
                budget.framing_reserve,
                512,
                "the native framing reserve was not applied to an artifact directory",
            )

    def test_a_non_native_path_is_left_alone(self) -> None:
        """Negative control: this must not hijack paths it knows nothing about."""
        with tempfile.TemporaryDirectory() as d:
            plain = Path(d) / "not-an-artifact"
            plain.mkdir()
            budget = resolve(model_path=str(plain), n_parallel=1)
            self.assertEqual(
                budget.source,
                "fallback:DEFAULT_PER_SLOT_CTX",
                "a directory with no native profile must still take the documented fallback",
            )


if __name__ == "__main__":
    unittest.main()
