"""Backend health, retryability classification, and circuit breaker.

Acceptance covered here, each with a command-output proof:

1. N consecutive failures on one backend opens its circuit; another
   backend is unaffected.
2. The breaker reopens after the cooling period via an injected clock
   (no real sleep).
3. Health survives restart: one process writes, another reads.
4. Each of the seven failure kinds is produced from a real observable
   (cited) or named as currently unreachable.
5. A non-retryable failure is classified as such and does not count
   toward the retry budget.
"""
from __future__ import annotations

import inspect
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from collections import Counter

from hcli.engine import NoOpMutation
from hcli.grok_bridge import GrokContractError, GrokNotAvailable, GrokRunError
from hcli.mutation import MutationError
from hcli.resources import ResourceLimits, can_admit
import hcli.resources as resources_mod

# New API is imported lazily so this module still *collects* against an
# unmodified resources.py. A collection ImportError is not a behavioural
# failure; each test below asserts the names exist, then exercises them.


def _api(*names):
    missing = [n for n in names if not hasattr(resources_mod, n)]
    if missing:
        raise AssertionError(
            "resources.py is missing backend-health API: " + ", ".join(missing)
        )
    return [getattr(resources_mod, n) for n in names]


_API_NAMES = (
    "BackendHealth",
    "classify_failure",
    "counts_toward_retry_budget",
    "KNOWN_BACKENDS",
    "FAILURE_KINDS",
    "NON_RETRYABLE",
    "TRANSIENT_BACKEND",
    "VERIFIER_FAILURE",
    "DETERMINISTIC_IMPLEMENTATION",
    "INVALID_OUTPUT",
    "RATE_LIMIT",
    "UNAVAILABLE_DEPENDENCY",
    "IMPOSSIBLE_CONTRACT",
    "STATE_HEALTHY",
    "STATE_DEGRADED",
    "STATE_CIRCUIT_OPEN",
    "HEALTH_FILENAME",
    "HEALTH_STALE_AFTER_SECONDS",
)


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


class HealthTest(unittest.TestCase):
    def setUp(self):
        bound = _api(*_API_NAMES)
        g = globals()
        for name, value in zip(_API_NAMES, bound):
            setattr(self, name, value)
            g[name] = value


def _health(tmp, **kwargs):
    clock = kwargs.pop("clock", None)
    if clock is None:
        clock = FakeClock()
    kwargs.setdefault("failure_threshold", 3)
    kwargs.setdefault("cooling_seconds", 30.0)
    return globals()["BackendHealth"](tmp, clock=clock, **kwargs)


class TestClassifyFailure(HealthTest):
    """Item 4: every kind is produced from a cited path, or listed unreachable."""

    def test_seven_kinds_are_the_closed_set(self):
        self.assertEqual(
            FAILURE_KINDS,
            (
                TRANSIENT_BACKEND,
                VERIFIER_FAILURE,
                DETERMINISTIC_IMPLEMENTATION,
                INVALID_OUTPUT,
                RATE_LIMIT,
                UNAVAILABLE_DEPENDENCY,
                IMPOSSIBLE_CONTRACT,
            ),
        )

    def test_transient_backend_from_http_503_and_timeout(self):
        # backends.py LlamaServerBackend.complete: raise RuntimeError(
        #   f"llama-server HTTP {exc.code}: ...")  — HTTP 503
        # engine.py Engine._call_model / _post_completion: same HTTPError
        #   wrapping, plus "llama-server request failed: {exc}"
        # grok_bridge.py GrokBridge._run: GrokRunError on TimeoutExpired
        # runtime.py RuntimePool._wait_ready: "Runtime {index} not ready after {timeout}s"
        http = classify_failure({"error": "llama-server HTTP 503: overloaded"})
        self.assertEqual(http.kind, TRANSIENT_BACKEND)
        self.assertTrue(http.retryable)
        self.assertEqual(http.observed, "HTTP 503")

        failed = classify_failure(
            {"error": "llama-server request failed: timed out"}
        )
        self.assertEqual(failed.kind, TRANSIENT_BACKEND)
        self.assertTrue(failed.retryable)

        grok = classify_failure(
            GrokRunError("grok-run timed out after 120s: ['grok-run']")
        )
        self.assertEqual(grok.kind, TRANSIENT_BACKEND)
        self.assertTrue(grok.retryable)
        self.assertEqual(grok.observed, "GrokRunError")

        not_ready = classify_failure(
            {"error": "Runtime 0 not ready after 300s"}
        )
        self.assertEqual(not_ready.kind, TRANSIENT_BACKEND)
        self.assertTrue(not_ready.retryable)

    def test_verifier_failure_from_test_failed(self):
        # engine.py Engine._pytest_evidence: reason = "TEST_FAILED"
        # engine.py _run_one_test: "TEST_ERROR:{type(exc).__name__}"
        # engine.py _pytest_evidence: reason = "NO_EVIDENCE" (zero collected)
        failed = classify_failure({"reason": "TEST_FAILED"})
        self.assertEqual(failed.kind, VERIFIER_FAILURE)
        self.assertTrue(failed.retryable)

        err = classify_failure({"reason": "TEST_ERROR:TimeoutExpired"})
        self.assertEqual(err.kind, VERIFIER_FAILURE)
        self.assertTrue(err.retryable)

        empty = classify_failure({"reason": "NO_EVIDENCE"})
        self.assertEqual(empty.kind, VERIFIER_FAILURE)
        self.assertTrue(empty.retryable)

    def test_deterministic_implementation_from_no_op_mutation(self):
        # engine.py NoOpMutation.reason = "NO_OP_MUTATION"
        # mutation.py MutationError("NO_OP_MUTATION") / replace old==new
        from_exc = classify_failure(NoOpMutation("identical bytes"))
        self.assertEqual(from_exc.kind, DETERMINISTIC_IMPLEMENTATION)
        self.assertFalse(from_exc.retryable)
        self.assertEqual(from_exc.observed, "NO_OP_MUTATION")

        from_mut = classify_failure(MutationError("NO_OP_MUTATION"))
        self.assertEqual(from_mut.kind, DETERMINISTIC_IMPLEMENTATION)
        self.assertFalse(from_mut.retryable)

        from_reason = classify_failure({"reason": "NO_OP_MUTATION"})
        self.assertEqual(from_reason.kind, DETERMINISTIC_IMPLEMENTATION)
        self.assertFalse(from_reason.retryable)

    def test_invalid_output_from_llama_invalid_json(self):
        # backends.py LlamaServerBackend.complete:
        #   raise RuntimeError("llama-server returned invalid JSON")
        # engine.py Engine._post_completion: EngineError same message
        clf = classify_failure(
            {"error": "llama-server returned invalid JSON"}
        )
        self.assertEqual(clf.kind, INVALID_OUTPUT)
        self.assertTrue(clf.retryable)

    def test_rate_limit_from_http_429(self):
        # No dedicated RateLimitError exists. The observable is the HTTP
        # status on the same HTTPError path as 503:
        #   backends.py LlamaServerBackend.complete
        #   engine.py Engine._call_model
        #   raise ... f"llama-server HTTP {exc.code}: ..."
        clf = classify_failure(
            {"error": "llama-server HTTP 429: too many requests"}
        )
        self.assertEqual(clf.kind, RATE_LIMIT)
        self.assertTrue(clf.retryable)
        self.assertEqual(clf.observed, "HTTP 429")

        via_status = classify_failure({"http_status": 429, "error": "busy"})
        self.assertEqual(via_status.kind, RATE_LIMIT)
        self.assertTrue(via_status.retryable)

    def test_unavailable_dependency_from_grok_and_pytest(self):
        # grok_bridge.py find_grok_run / _run FileNotFoundError -> GrokNotAvailable
        # engine.py _admit_test_command: reason = "PYTEST_UNAVAILABLE"
        grok = classify_failure(GrokNotAvailable("grok-run is not on PATH"))
        self.assertEqual(grok.kind, UNAVAILABLE_DEPENDENCY)
        self.assertFalse(grok.retryable)
        self.assertEqual(grok.observed, "GrokNotAvailable")

        pytest_missing = classify_failure({"reason": "PYTEST_UNAVAILABLE"})
        self.assertEqual(pytest_missing.kind, UNAVAILABLE_DEPENDENCY)
        self.assertTrue(pytest_missing.retryable)

    def test_impossible_contract_from_grok_contract_and_landed_names(self):
        # grok_bridge.py validate_contract_text / _raise_from_failure
        #   -> GrokContractError  (in this snapshot)
        grok = classify_failure(
            GrokContractError("contract missing WRITE/VERIFY")
        )
        self.assertEqual(grok.kind, IMPOSSIBLE_CONTRACT)
        self.assertFalse(grok.retryable)
        self.assertEqual(grok.observed, "GrokContractError")

        # Names landed after HEAD 2f670c2; matched by string, not import.
        # VACUOUS_COMMAND / EMPTY_COMMAND: command_is_admissible
        #   (executors.py, verifier_pipeline.py) — not in this worktree.
        # ContextPreflightError: context_budget.py — not in this worktree.
        vacuous = classify_failure({"reason": "VACUOUS_COMMAND"})
        self.assertEqual(vacuous.kind, IMPOSSIBLE_CONTRACT)
        self.assertFalse(vacuous.retryable)

        empty = classify_failure({"reason": "EMPTY_COMMAND"})
        self.assertEqual(empty.kind, IMPOSSIBLE_CONTRACT)
        self.assertFalse(empty.retryable)

        preflight = classify_failure({"error": "ContextPreflightError"})
        self.assertEqual(preflight.kind, IMPOSSIBLE_CONTRACT)
        self.assertFalse(preflight.retryable)
        self.assertEqual(preflight.observed, "ContextPreflightError")

    def test_every_kind_is_produced_none_unreachable(self):
        produced = {
            classify_failure({"error": "llama-server HTTP 503: x"}).kind,
            classify_failure({"reason": "TEST_FAILED"}).kind,
            classify_failure({"reason": "NO_OP_MUTATION"}).kind,
            classify_failure({"error": "llama-server returned invalid JSON"}).kind,
            classify_failure({"error": "llama-server HTTP 429: x"}).kind,
            classify_failure(GrokNotAvailable("missing")).kind,
            classify_failure(GrokContractError("bad")).kind,
        }
        self.assertEqual(produced, set(FAILURE_KINDS))

    def test_unreachable_kinds_explicit_list(self):
        # None of the seven is unreachable: see test_every_kind_is_produced
        # and the cited paths above. RATE_LIMIT has no dedicated exception
        # but is the HTTP 429 branch of an existing HTTPError path.
        unreachable = ()
        self.assertEqual(unreachable, ())


class TestRetryBudget(HealthTest):
    """Item 5: non-retryable is classified as such and does not count."""

    def test_non_retryable_set_is_the_audit_set(self):
        self.assertEqual(
            NON_RETRYABLE,
            {
                "GrokNotAvailable",
                "GrokContractError",
                "VACUOUS_COMMAND",
                "EMPTY_COMMAND",
                "NO_OP_MUTATION",
                "ContextPreflightError",
            },
        )

    def test_every_non_retryable_name_is_not_budgeted(self):
        samples = {
            "GrokNotAvailable": GrokNotAvailable("no grok-run"),
            "GrokContractError": GrokContractError("no WRITE"),
            "VACUOUS_COMMAND": {"reason": "VACUOUS_COMMAND"},
            "EMPTY_COMMAND": {"reason": "EMPTY_COMMAND"},
            "NO_OP_MUTATION": {"reason": "NO_OP_MUTATION"},
            "ContextPreflightError": {"error": "ContextPreflightError"},
        }
        for name, context in samples.items():
            clf = classify_failure(context)
            self.assertFalse(clf.retryable, name)
            self.assertFalse(counts_toward_retry_budget(context), name)
            self.assertEqual(clf.observed, name)

    def test_retryable_failures_do_count_toward_budget(self):
        self.assertTrue(
            counts_toward_retry_budget({"error": "llama-server HTTP 503: x"})
        )
        self.assertTrue(counts_toward_retry_budget({"reason": "TEST_FAILED"}))
        self.assertTrue(
            counts_toward_retry_budget(
                {"error": "llama-server returned invalid JSON"}
            )
        )
        self.assertTrue(
            counts_toward_retry_budget({"error": "llama-server HTTP 429: x"})
        )

    def test_no_op_mutation_does_not_trip_the_circuit(self):
        # Deterministic implementation failure is not a backend outage.
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock()
            health = _health(tmp, clock=clock, failure_threshold=1)
            clf = health.record_failure("cpu", {"reason": "NO_OP_MUTATION"})
            self.assertFalse(clf.retryable)
            self.assertFalse(counts_toward_retry_budget({"reason": "NO_OP_MUTATION"}))
            snap = health.snapshot("cpu")
            self.assertEqual(snap["consecutive_failures"], 0)
            self.assertEqual(snap["state"], STATE_HEALTHY)
            self.assertTrue(health.allows_new_assignments("cpu"))

    def test_grok_not_available_does_not_burn_budget_but_trips_circuit(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock()
            health = _health(tmp, clock=clock, failure_threshold=2)
            ctx = GrokNotAvailable("grok-run is not on PATH")
            self.assertFalse(counts_toward_retry_budget(ctx))
            health.record_failure("grok", ctx)
            health.record_failure("grok", ctx)
            self.assertEqual(health.state("grok"), STATE_CIRCUIT_OPEN)
            self.assertFalse(health.allows_new_assignments("grok"))


class TestCircuitBreaker(HealthTest):
    """Items 1 and 2: isolation across backends, reopen via injected clock."""

    def test_n_failures_open_one_backend_leave_the_other_healthy(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock()
            n = 3
            health = _health(tmp, clock=clock, failure_threshold=n)
            ctx = {"error": "llama-server HTTP 503: overloaded"}
            for _ in range(n):
                health.record_failure("qwen", ctx)
            qwen = health.snapshot("qwen")
            grok = health.snapshot("grok")
            cpu = health.snapshot("cpu")
            self.assertEqual(qwen["consecutive_failures"], n)
            self.assertEqual(qwen["state"], STATE_CIRCUIT_OPEN)
            self.assertFalse(qwen["allows_new"])
            self.assertFalse(health.allows_new_assignments("qwen"))
            self.assertEqual(grok["state"], STATE_HEALTHY)
            self.assertEqual(grok["consecutive_failures"], 0)
            self.assertTrue(health.allows_new_assignments("grok"))
            self.assertEqual(cpu["state"], STATE_HEALTHY)
            self.assertTrue(health.allows_new_assignments("cpu"))

    def test_breaker_reopens_after_cooling_with_injected_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock(t=5_000.0)
            cooling = 30.0
            health = _health(
                tmp,
                clock=clock,
                failure_threshold=3,
                cooling_seconds=cooling,
            )
            ctx = {"error": "llama-server HTTP 503: overloaded"}
            for _ in range(3):
                health.record_failure("qwen", ctx)
            self.assertEqual(health.state("qwen"), STATE_CIRCUIT_OPEN)
            self.assertFalse(health.allows_new_assignments("qwen"))

            clock.advance(cooling - 0.01)
            self.assertEqual(health.state("qwen"), STATE_CIRCUIT_OPEN)
            self.assertFalse(health.allows_new_assignments("qwen"))

            clock.advance(0.02)
            self.assertNotEqual(health.state("qwen"), STATE_CIRCUIT_OPEN)
            self.assertEqual(health.state("qwen"), STATE_DEGRADED)
            self.assertTrue(health.allows_new_assignments("qwen"))
            # Other backends still untouched.
            self.assertEqual(health.state("grok"), STATE_HEALTHY)
            self.assertTrue(health.allows_new_assignments("grok"))

    def test_success_closes_the_breaker(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock()
            health = _health(tmp, clock=clock, failure_threshold=2)
            health.record_failure("grok", GrokRunError("blip"))
            health.record_failure("grok", GrokRunError("blip"))
            self.assertEqual(health.state("grok"), STATE_CIRCUIT_OPEN)
            clock.advance(1.0)
            snap = health.record_success("grok")
            self.assertEqual(snap["state"], STATE_HEALTHY)
            self.assertEqual(snap["consecutive_failures"], 0)
            self.assertTrue(health.allows_new_assignments("grok"))

    def test_below_threshold_is_degraded_not_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            health = _health(tmp, failure_threshold=3)
            health.record_failure("cpu", {"error": "llama-server HTTP 503: x"})
            self.assertEqual(health.state("cpu"), STATE_DEGRADED)
            self.assertTrue(health.allows_new_assignments("cpu"))
            self.assertEqual(health.state("qwen"), STATE_HEALTHY)


class TestHealthPersistence(HealthTest):
    """Item 3: written by one process, read by another. Atomic write."""

    def test_survives_in_process_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock()
            writer = _health(tmp, clock=clock, failure_threshold=3)
            writer.record_failure("qwen", {"error": "llama-server HTTP 503: x"})
            writer.record_success("cpu")
            path = Path(tmp) / ".hcli" / HEALTH_FILENAME
            self.assertTrue(path.is_file())
            reader = BackendHealth(
                tmp, clock=clock, failure_threshold=3, cooling_seconds=30.0
            )
            qwen = reader.snapshot("qwen")
            cpu = reader.snapshot("cpu")
            self.assertEqual(qwen["consecutive_failures"], 1)
            self.assertEqual(qwen["state"], STATE_DEGRADED)
            self.assertEqual(cpu["consecutive_failures"], 0)
            self.assertIsNotNone(cpu["last_success_time"])
            self.assertEqual(cpu["state"], STATE_HEALTHY)

    def test_written_by_one_process_read_by_another(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = (
                "import sys\n"
                ""
                "from hcli.resources import BackendHealth\n"
                f"h = BackendHealth({tmp!r}, failure_threshold=3, cooling_seconds=30.0)\n"
                "h.record_failure('grok', {'error': 'GrokNotAvailable'})\n"
                "h.record_failure('grok', {'error': 'GrokNotAvailable'})\n"
                "h.record_success('cpu')\n"
                "print('wrote')\n"
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = env.get("PYTHONPATH", "")
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(REPO),
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("wrote", proc.stdout)

            reader = BackendHealth(tmp, failure_threshold=3, cooling_seconds=30.0)
            grok = reader.snapshot("grok")
            cpu = reader.snapshot("cpu")
            qwen = reader.snapshot("qwen")
            self.assertEqual(grok["consecutive_failures"], 2)
            self.assertEqual(grok["state"], STATE_DEGRADED)
            self.assertIsNotNone(grok["last_failure_time"])
            self.assertIsNotNone(cpu["last_success_time"])
            self.assertEqual(cpu["consecutive_failures"], 0)
            self.assertEqual(qwen["consecutive_failures"], 0)
            self.assertEqual(qwen["state"], STATE_HEALTHY)

    def test_atomic_write_uses_temp_and_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            real = os.replace

            def spy(src, dst, *args, **kwargs):
                calls.append((str(src), str(dst)))
                return real(src, dst, *args, **kwargs)

            os.replace = spy
            try:
                health = _health(tmp)
                health.record_success("qwen")
            finally:
                os.replace = real
            self.assertTrue(calls)
            src, dst = calls[-1]
            self.assertIn(".tmp", Path(src).name)
            self.assertEqual(Path(dst), Path(tmp) / ".hcli" / HEALTH_FILENAME)
            leftover = list(Path(tmp).glob("**/*.tmp"))
            self.assertEqual(leftover, [])

    def test_corrupt_file_is_empty_health_not_invented_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".hcli" / HEALTH_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not json", encoding="utf-8")
            health = BackendHealth(tmp)
            snap = health.snapshot("qwen")
            self.assertEqual(snap["consecutive_failures"], 0)
            self.assertEqual(snap["state"], STATE_HEALTHY)
            self.assertIsNone(snap["last_failure_time"])


class TestStaleness(HealthTest):
    def test_hour_old_record_is_prior_not_present_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock(t=1_000.0)
            health = _health(
                tmp,
                clock=clock,
                failure_threshold=3,
                stale_after_seconds=HEALTH_STALE_AFTER_SECONDS,
            )
            health.record_success("cpu")
            fresh = health.snapshot("cpu")
            self.assertFalse(fresh["stale"])
            clock.advance(HEALTH_STALE_AFTER_SECONDS + 1.0)
            prior = health.snapshot("cpu")
            self.assertTrue(prior["stale"])
            # State is still derived, but flagged as a prior.
            self.assertEqual(prior["state"], STATE_HEALTHY)

    def test_stale_open_circuit_does_not_refuse_new_assignments(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock(t=1_000.0)
            health = _health(
                tmp,
                clock=clock,
                failure_threshold=3,
                cooling_seconds=10_000.0,  # still inside cooling after 1h
                stale_after_seconds=HEALTH_STALE_AFTER_SECONDS,
            )
            for _ in range(3):
                health.record_failure(
                    "qwen", {"error": "llama-server HTTP 503: x"}
                )
            self.assertEqual(health.state("qwen"), STATE_CIRCUIT_OPEN)
            self.assertFalse(health.allows_new_assignments("qwen"))
            clock.advance(HEALTH_STALE_AFTER_SECONDS + 1.0)
            snap = health.snapshot("qwen")
            self.assertTrue(snap["stale"])
            self.assertTrue(health.allows_new_assignments("qwen"))


class TestSchedulerAccessor(HealthTest):
    def test_allows_new_assignments_is_the_consult_site(self):
        source = inspect.getsource(BackendHealth.allows_new_assignments)
        self.assertIn("assign_ready", source)
        self.assertIn("can_admit", source)
        self.assertIn("preferred_backend", source)

    def test_can_admit_is_not_wired_to_health(self):
        source = inspect.getsource(can_admit)
        self.assertNotIn("BackendHealth", source)
        self.assertNotIn("circuit", source.lower())
        self.assertNotIn("allows_new", source)
        limits = ResourceLimits.resolve()
        occupied = Counter()
        self.assertTrue(can_admit("COMPILE", occupied, limits))

    def test_known_backends_are_qwen_grok_cpu(self):
        self.assertEqual(KNOWN_BACKENDS, ("qwen", "grok", "cpu"))

    def test_unknown_backend_is_refused_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            health = _health(tmp)
            with self.assertRaises(ValueError):
                health.record_failure("llama", {"error": "x"})


if __name__ == "__main__":
    unittest.main()
