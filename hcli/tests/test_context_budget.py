"""Canonical context-budget authority.

These tests were watched FAILING against the unmodified HCLI tree
(no context_budget.py, five independent 32768 literals, no preflight).
"""
from __future__ import annotations

import json
import os
import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]

from hcli.config import Config
from hcli.context_budget import (
    DEFAULT_FRAMING_RESERVE,
    DEFAULT_PER_SLOT_CTX,
    PacketBudgetError,
    estimate_tokens,
    fit_or_refuse,
    gguf_context_length,
    per_seq_context,
    preflight,
    preflight_packet,
    resolve,
    solve_parallel,
)
from hcli.engine import (
    ContextPreflightError,
    Engine,
    EventBus,
)
from hcli.workspace import Workspace

KNOWN_GGUF = Path(
    "/Users/scammermike/models/qwen3.8-27b-abliterated/"
    "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"
)

_CTX_ENV = ("HCLI_CTX_SIZE", "HCLI_MODEL_TOKENS", "HCLI_EVIDENCE_CHAR_BUDGET")


def _write_min_gguf(
    path: Path,
    context_length: int = 262144,
    key: str = "llama.context_length",
) -> None:
    def enc_str(text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    kv = enc_str(key) + struct.pack("<I", 4) + struct.pack("<I", int(context_length))
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack(
        "<Q", 1
    )
    path.write_bytes(header + kv)


class _CleanCtxEnv:
    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in _CTX_ENV}
        self._home = os.environ.get("HOME")
        for key in _CTX_ENV:
            os.environ.pop(key, None)
        return self

    def __exit__(self, *exc):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if self._home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self._home


def _engine(root: Path, runtime_count: int = 1) -> Engine:
    cfg = Config(str(root), global_path=str(root / "global-config.json"))
    runtime = type("Runtime", (), {})()
    runtime.index = 0
    runtime.pid = 1
    runtime.port = 18765
    runtime.active = True
    pool = type("Pool", (), {})()
    pool.runtimes = [runtime]
    return Engine(
        workspace=Workspace(str(root)),
        event_bus=EventBus(),
        runtime_provider=lambda: pool,
        runtime_state_provider=lambda: pool,
        runtime_count=runtime_count,
        model_name="local",
        config=cfg,
    )


class TestPerSeqContext(unittest.TestCase):
    def test_per_seq_context_regression(self):
        self.assertEqual(per_seq_context(32768, 3), 11008)
        self.assertEqual(per_seq_context(32768, 1), 32768)
        self.assertEqual(per_seq_context(32768, 2), 16384)
        self.assertEqual(per_seq_context(32768, 4), 8192)


class TestSolveParallel(unittest.TestCase):
    def test_solve_parallel_recovers_divisor(self):
        self.assertEqual(solve_parallel(32768, 11008), 3)
        self.assertEqual(solve_parallel(32768, 32768), 1)
        self.assertEqual(solve_parallel(32768, 16384), 2)
        self.assertIsNone(solve_parallel(32768, 12345))


class TestPrecedence(unittest.TestCase):
    def test_precedence_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".config" / "hcli").mkdir(parents=True)
            (home / ".config" / "hcli" / "machine_genome.json").write_text(
                json.dumps({"context_ctx_size": 16384})
            )
            model = home / "model.gguf"
            _write_min_gguf(model, 262144)
            with _CleanCtxEnv():
                with patch.dict(os.environ, {"HOME": str(home)}, clear=False):
                    budget = resolve(
                        model_path=str(model),
                        n_parallel=2,
                        ctx_size=65536,
                    )
            self.assertTrue(budget.source.startswith("override"))
            self.assertEqual(budget.total_ctx, 65536)
            self.assertEqual(budget.n_parallel, 2)
            self.assertEqual(budget.per_request_ctx, per_seq_context(65536, 2))
            for rung in ("profile", "discovered", "fallback"):
                self.assertIn(rung, budget.provenance)
                entry = budget.provenance[rung]
                self.assertFalse(entry.get("won"))
                self.assertTrue(str(entry.get("reason") or ""))

    def test_precedence_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            genome_dir = home / ".config" / "hcli"
            genome_dir.mkdir(parents=True)
            (genome_dir / "machine_genome.json").write_text(
                json.dumps({"context_ctx_size": 16384})
            )
            model = home / "model.gguf"
            _write_min_gguf(model, 262144)
            with _CleanCtxEnv():
                with patch.dict(os.environ, {"HOME": str(home)}, clear=False):
                    budget = resolve(model_path=str(model), n_parallel=1)
            self.assertEqual(budget.source, "profile:context_ctx_size")
            self.assertEqual(budget.total_ctx, 16384)
            for rung in ("override", "discovered", "fallback"):
                self.assertIn(rung, budget.provenance)
                entry = budget.provenance[rung]
                self.assertFalse(entry.get("won"))
                self.assertTrue(str(entry.get("reason") or ""))

    def test_precedence_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".config" / "hcli").mkdir(parents=True)
            model = home / "model.gguf"
            _write_min_gguf(model, 262144)
            with _CleanCtxEnv():
                with patch.dict(os.environ, {"HOME": str(home)}, clear=False):
                    budget = resolve(model_path=str(model), n_parallel=1)
            self.assertTrue(budget.source.startswith("discovered"))
            self.assertEqual(budget.model_ceiling, 262144)
            self.assertEqual(budget.total_ctx, DEFAULT_PER_SLOT_CTX)
            for rung in ("override", "profile", "fallback"):
                self.assertIn(rung, budget.provenance)
                entry = budget.provenance[rung]
                self.assertFalse(entry.get("won"))
                self.assertTrue(str(entry.get("reason") or ""))

    def test_precedence_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".config" / "hcli").mkdir(parents=True)
            with _CleanCtxEnv():
                with patch.dict(os.environ, {"HOME": str(home)}, clear=False):
                    budget = resolve(n_parallel=3)
            self.assertEqual(budget.source, "fallback:DEFAULT_PER_SLOT_CTX")
            self.assertEqual(budget.total_ctx, DEFAULT_PER_SLOT_CTX)
            self.assertEqual(budget.per_request_ctx, 11008)
            self.assertIsNone(budget.model_ceiling)
            for rung in ("override", "profile", "discovered"):
                self.assertIn(rung, budget.provenance)
                entry = budget.provenance[rung]
                self.assertFalse(entry.get("won"))
                self.assertTrue(str(entry.get("reason") or ""))


class TestOverrideDoesNotLeak(unittest.TestCase):
    def test_override_does_not_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".config" / "hcli").mkdir(parents=True)
            with _CleanCtxEnv():
                with patch.dict(
                    os.environ,
                    {"HOME": str(home), "HCLI_CTX_SIZE": "65536"},
                    clear=False,
                ):
                    overridden = resolve(n_parallel=2)
                self.assertTrue(overridden.source.startswith("override"))
                self.assertEqual(overridden.total_ctx, 65536)
                self.assertEqual(
                    overridden.per_request_ctx, per_seq_context(65536, 2)
                )
                self.assertEqual(DEFAULT_PER_SLOT_CTX, 32768)
                with patch.dict(os.environ, {"HOME": str(home)}, clear=False):
                    os.environ.pop("HCLI_CTX_SIZE", None)
                    after = resolve(n_parallel=2)
                self.assertEqual(after.source, "fallback:DEFAULT_PER_SLOT_CTX")
                self.assertEqual(after.total_ctx, DEFAULT_PER_SLOT_CTX)
                self.assertEqual(
                    after.per_request_ctx,
                    per_seq_context(DEFAULT_PER_SLOT_CTX, 2),
                )
                self.assertNotEqual(after.source, overridden.source)


class TestPreflightThreeSlot(unittest.TestCase):
    def test_preflight_fails_three_slot_32768(self):
        budget = resolve(
            ctx_size=32768,
            n_parallel=3,
            generation_reserve=DEFAULT_FRAMING_RESERVE,
        )
        result = preflight(budget, 23532, kind="root")
        self.assertFalse(result.ok)
        self.assertGreater(result.shortfall, 0)
        self.assertEqual(result.per_request_ctx, 11008)
        self.assertEqual(result.kind, "root")
        lever = result.remedy.lower()
        self.assertTrue(
            any(
                token in lever
                for token in (
                    "hcli_ctx_size",
                    "n_parallel",
                    "--parallel",
                    "--ctx-size",
                    "evidence",
                )
            ),
            result.remedy,
        )


class TestPreflightOneSlot(unittest.TestCase):
    def test_preflight_passes_one_slot(self):
        budget = resolve(
            ctx_size=32768,
            n_parallel=1,
            generation_reserve=DEFAULT_FRAMING_RESERVE,
        )
        result = preflight(budget, 23532, kind="root")
        self.assertTrue(result.ok)
        self.assertEqual(result.shortfall, 0)
        self.assertEqual(result.per_request_ctx, 32768)


class TestNoHttpOnPreflightFailure(unittest.TestCase):
    def test_no_http_on_preflight_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _CleanCtxEnv():
                with patch.dict(
                    os.environ,
                    {"HOME": str(tmp), "HCLI_CTX_SIZE": "4096"},
                    clear=False,
                ):
                    engine = _engine(Path(tmp), runtime_count=1)

                    def boom(*_args, **_kwargs):
                        raise AssertionError("HTTP must not be reached")

                    engine._post_completion = boom
                    with self.assertRaises(ContextPreflightError) as ctx:
                        engine._call_model("x" * 200000)
            result = ctx.exception.result
            self.assertFalse(result.ok)
            self.assertGreater(result.shortfall, 0)
            self.assertTrue(result.remedy)


class TestSingleAuthorityGuard(unittest.TestCase):
    def test_single_authority_guard(self):
        import hcli as hcli_pkg

        hcli_dir = Path(hcli_pkg.__file__).resolve().parent
        modules = ("config.py", "backends.py", "engine.py", "runtime.py")
        env_default = re.compile(
            r"""HCLI_CTX_SIZE["']\s*,\s*["']?\d+["']?"""
            r"""|os\.environ\.get\(\s*["']HCLI_CTX_SIZE["']"""
        )
        ctx_default = re.compile(
            r"""ctx_size\s*\([^)]*default\s*=\s*32768"""
            r"""|["']32768["']"""
        )
        bare_window = re.compile(r"\b32768\b")
        offenders = []
        for name in modules:
            text = (hcli_dir / name).read_text(encoding="utf-8")
            if env_default.search(text):
                offenders.append(f"{name}: HCLI_CTX_SIZE default/read")
            if ctx_default.search(text):
                offenders.append(f"{name}: ctx_size default/literal 32768")
            if bare_window.search(text):
                offenders.append(f"{name}: independent 32768 literal")
        self.assertEqual(offenders, [])


class TestGgufContextLength(unittest.TestCase):
    def test_gguf_context_length_real_model(self):
        if not KNOWN_GGUF.is_file():
            self.skipTest(f"GGUF absent: {KNOWN_GGUF}")
        self.assertEqual(gguf_context_length(str(KNOWN_GGUF)), 262144)

    def test_gguf_context_length_corrupt_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nope.gguf"
            path.write_bytes(b"x" * 64)
            self.assertIsNone(gguf_context_length(str(path)))


class TestPacketPreflight(unittest.TestCase):
    def test_preflight_packet_refuses_over_budget_worker(self):
        budget = resolve(
            ctx_size=32768,
            n_parallel=3,
            generation_reserve=DEFAULT_FRAMING_RESERVE,
        )
        result = preflight_packet(
            budget,
            "PHASE: running\nWORKUNIT: u1",
            evidence_tokens=50000,
            kind="worker",
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.kind, "worker")
        self.assertGreater(result.shortfall, 0)
        self.assertIn("demand", result.remedy.lower())

    def test_fit_or_refuse_is_the_caller_visible_signal(self):
        budget = resolve(
            ctx_size=32768,
            n_parallel=3,
            generation_reserve=DEFAULT_FRAMING_RESERVE,
        )
        with self.assertRaises(PacketBudgetError) as ctx:
            fit_or_refuse(
                budget,
                "tiny packet",
                evidence_tokens=50000,
                kind="worker",
            )
        exc = ctx.exception
        self.assertIsNotNone(exc.result)
        self.assertFalse(exc.result.ok)
        self.assertIn("refused", str(exc).lower())
        self.assertGreater(exc.shortfall, 0)

    def test_fit_or_refuse_passes_a_fitting_packet(self):
        budget = resolve(
            ctx_size=32768,
            n_parallel=1,
            generation_reserve=DEFAULT_FRAMING_RESERVE,
        )
        result = fit_or_refuse(budget, "PHASE: running", kind="worker")
        self.assertTrue(result.ok)
        self.assertEqual(result.shortfall, 0)

    def test_estimate_tokens_empty_is_zero(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertGreaterEqual(estimate_tokens("abcd"), 1)


if __name__ == "__main__":
    unittest.main()
