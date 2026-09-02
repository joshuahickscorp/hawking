"""The context budget must know the native resident's real ceiling.

LIVE FAILURE this closes: a 12,456-token sovereign ultragoal passed preflight
with `ok=True, shortfall=0` and was then rejected by the runtime with
"prompt is 12456 tokens and native max_seq_len is 8192; no generation token
fits". `_discover_ceiling` knew a GGUF header and a llama-server /props port,
and a Hawking native profile is neither, so `resolve()` fell through to
`fallback:DEFAULT_PER_SLOT_CTX` and reported 32768 total / 24576 usable for a
resident that accepts 8192.
"""
from __future__ import annotations

import json

import pytest

from hcli.context_budget import (
    DEFAULT_PER_SLOT_CTX,
    NATIVE_FRAMING_RESERVE,
    native_profile_limits,
    preflight,
    resolve,
)

PROFILE = "hcli/hawking-native.sealed-3.14.json"


def _profile(tmp_path, **over):
    doc = {
        "runtime": "hawking-native",
        "provider": "native",
        "max_seq_len": 8192,
        "generation": {"max_new_tokens": 2048},
    }
    doc.update(over)
    p = tmp_path / "profile.json"
    p.write_text(json.dumps(doc))
    return str(p)


def test_the_shipped_profile_ceiling_is_read_not_guessed():
    ceiling, new_tokens, meta = native_profile_limits(PROFILE)
    assert ceiling == 8192, meta
    assert new_tokens == 2048, meta


def test_resolve_uses_the_native_ceiling_instead_of_the_llama_fallback(tmp_path):
    budget = resolve(model_path=_profile(tmp_path), n_parallel=1)
    assert budget.source == "discovered:hawking_native_profile"
    assert budget.total_ctx == 8192
    assert budget.model_ceiling == 8192
    assert budget.total_ctx != DEFAULT_PER_SLOT_CTX, "fell back to the llama default"


def test_native_reserves_leave_real_input_room(tmp_path):
    """4096 framing + 4096 generation on an 8192 window is ZERO usable input."""
    budget = resolve(model_path=_profile(tmp_path), n_parallel=1)
    assert budget.generation_reserve == 2048, "generation headroom is the profile's"
    assert budget.framing_reserve == NATIVE_FRAMING_RESERVE
    assert budget.usable_input_tokens == 8192 - 2048 - NATIVE_FRAMING_RESERVE
    assert budget.usable_input_tokens > 0


def test_the_exact_failing_demand_is_now_refused_before_the_runtime(tmp_path):
    """The mutation test the brief asks for: raw source must be REJECTED."""
    budget = resolve(model_path=_profile(tmp_path), n_parallel=1)
    result = preflight(budget, 12456, kind="root")
    assert result.ok is False, "the raw 12,456-token ultragoal must not pass"
    assert result.shortfall > 0
    assert result.remedy, "a refusal must name a lever"


def test_a_compiled_packet_still_fits_with_generation_headroom(tmp_path):
    budget = resolve(model_path=_profile(tmp_path), n_parallel=1)
    assert preflight(budget, 4000, kind="root").ok is True
    assert preflight(budget, budget.usable_input_tokens, kind="root").ok is True
    assert preflight(budget, budget.usable_input_tokens + 1, kind="root").ok is False


def test_a_non_native_model_path_is_untouched(tmp_path):
    """Nothing here may change llama.cpp/GGUF or remote budget resolution."""
    ceiling, new_tokens, _ = native_profile_limits(str(tmp_path / "model.gguf"))
    assert ceiling is None and new_tokens is None
    plain = tmp_path / "other.json"
    plain.write_text(json.dumps({"runtime": "something-else", "max_seq_len": 999}))
    assert native_profile_limits(str(plain))[0] is None
    budget = resolve(model_path=None, n_parallel=1)
    assert budget.total_ctx == DEFAULT_PER_SLOT_CTX
    assert budget.framing_reserve != NATIVE_FRAMING_RESERVE


def test_an_explicit_override_still_wins(tmp_path):
    budget = resolve(model_path=_profile(tmp_path), n_parallel=1, generation_reserve=99)
    assert budget.generation_reserve == 99


def test_a_native_profile_declares_its_window_and_is_not_clamped(tmp_path):
    """DEFAULT_PER_SLOT_CTX is a llama.cpp spawn guess, not a native ceiling.

    A profile asking for 131072 was served 32768 with no diagnostic, because
    `_safe_policy_total` did `min(ceiling, DEFAULT_PER_SLOT_CTX)`. For a native
    resident the profile IS the authority: the window is declared, not guessed.
    """
    long_profile = _profile(tmp_path, max_seq_len=131072)
    budget = resolve(model_path=long_profile, n_parallel=1)
    assert budget.total_ctx == 131072, "the declared window was clamped"
    assert budget.usable_input_tokens == 131072 - 2048 - NATIVE_FRAMING_RESERVE
    assert preflight(budget, 12456, kind="root").ok is True


def test_below_the_default_nothing_changes(tmp_path):
    """min(8192, 32768) == 8192 either way: only a bigger profile is affected."""
    budget = resolve(model_path=_profile(tmp_path), n_parallel=1)
    assert budget.total_ctx == 8192


def test_kv_cost_of_a_declared_window_is_arithmetic_the_operator_can_check():
    """16 of 64 layers keep KV (full_attention_interval=4): 65,536 B/token."""
    kv_per_token = 2 * 16 * 4 * 256 * 2   # K+V, full layers, kv_heads, head_dim, bf16
    assert kv_per_token == 65_536
    assert kv_per_token * 8192 / 1024**3 == pytest.approx(0.5, abs=0.01)
    assert kv_per_token * 131072 / 1024**3 == pytest.approx(8.0, abs=0.01)
    assert kv_per_token * 262144 / 1024**3 == pytest.approx(16.0, abs=0.01)


# --- the LIVE path, not resolve() in isolation -----------------------------
# The first version of this fix was verified by calling resolve() directly. That
# is not what the engine does: `Engine._context_budget` asks the RuntimePool for
# its budget first and returns it verbatim if present. A fix that only works
# when resolve() is called by hand is not on the live request path.

class _StubPool:
    """Shaped like RuntimePool for the two attributes _context_budget reads."""

    def __init__(self, model_path, budget=None):
        self.model_path = model_path
        self.topology = "process"
        self.requested_n = 1
        self.admitted_n = 1
        self.repo_root = "."
        if budget is not None:
            self.context_budget = budget


def _engine_budget(pool):
    from hcli.engine import Engine

    eng = Engine.__new__(Engine)
    eng.runtime_provider = lambda: pool
    eng.runtime_count = 1

    class _Cfg:
        def model_tokens(self):
            return (None, None)

    eng.config = _Cfg()
    return eng._context_budget()


def test_the_engine_budget_sees_the_native_ceiling_through_the_pool():
    """This is the assertion that would have caught the live failure."""
    budget = _engine_budget(_StubPool(PROFILE))
    assert budget.total_ctx == 8192, (
        f"engine preflights against {budget.total_ctx}, resident accepts 8192"
    )
    assert budget.source == "discovered:hawking_native_profile"


def test_the_exact_live_prompt_is_refused_on_the_engine_path():
    """12,456 tokens: the real sovereign ultragoal that failed at the runtime."""
    from hcli.context_budget import preflight

    budget = _engine_budget(_StubPool(PROFILE))
    result = preflight(budget, 12456, kind="root")
    assert result.ok is False, "the engine path must refuse before the runtime"
    assert result.shortfall > 0


def test_a_pool_that_already_carries_a_budget_still_wins():
    """Don't silently override a pool that measured its own slot."""
    from hcli.context_budget import resolve

    measured = resolve(model_path=PROFILE, n_parallel=1)
    assert _engine_budget(_StubPool(PROFILE, budget=measured)) is measured
