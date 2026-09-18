from __future__ import annotations

import pytest

from hawking.auto_mode import (
    AUTO_MODEL_ID,
    AUTO_POLICY_REVISION,
    AutoRouteUnavailable,
    REMOTE_AUTO_ROSTER,
    choose_remote_route,
    is_auto_model,
)


def test_auto_roster_contains_core_crossover_and_escalation_pool():
    assert REMOTE_AUTO_ROSTER == (
        "deepseek/deepseek-v4.1-flash",
        "moonshotai/kimi-k3",
        "z-ai/glm-5.3",
        "qwen/qwen3.8-2.4t-a95b",
        "nvidia/nemotron-3-ultra-550b-a55b",
    )
    assert "qwen/qwen3.8-flash" not in REMOTE_AUTO_ROSTER
    assert AUTO_POLICY_REVISION == "hawking-auto-aggressive-v2-measured-diversity-unbounded-14"


def test_auto_aliases_are_hawking_owned():
    assert is_auto_model(AUTO_MODEL_ID)
    assert is_auto_model("hawking/auto")
    assert is_auto_model("auto")
    assert not is_auto_model("z-ai/glm-5.3-flash")


def test_execution_requests_prefer_deepseek_flash():
    route = choose_remote_route({
        "messages": [{"role": "user", "content": "run the focused git test"}],
        "tools": [{"type": "function", "function": {"name": "tests.run"}}],
    })
    assert route.model_id == "deepseek/deepseek-v4.1-flash"
    assert route.task_class == "execution"


def test_alternate_route_remains_inside_the_default_roster():
    default = choose_remote_route({"messages": [{"role": "user", "content": "hello"}]})
    premium = choose_remote_route({"hawking_auto_profile": "premium"})
    assert default.model_id == "deepseek/deepseek-v4.1-flash"
    assert premium.model_id == "moonshotai/kimi-k3"


def test_diversification_profiles_select_glm_or_escalation_models():
    glm = choose_remote_route({"hawking_auto_profile": "glm"})
    qwen = choose_remote_route({"hawking_auto_profile": "qwen"})
    nemotron = choose_remote_route({"hawking_auto_profile": "nemotron"})

    assert glm.model_id == "z-ai/glm-5.3"
    assert qwen.model_id == "qwen/qwen3.8-2.4t-a95b"
    assert nemotron.model_id == "nvidia/nemotron-3-ultra-550b-a55b"


def test_escalation_models_are_not_used_by_routine_auto_routes():
    route = choose_remote_route({"messages": [{"role": "user", "content": "map the source owners"}]})
    assert route.model_id not in {
        "qwen/qwen3.8-2.4t-a95b",
        "nvidia/nemotron-3-ultra-550b-a55b",
    }


def test_live_catalog_and_policy_bound_auto_selection():
    route = choose_remote_route(
        {"hawking_auto_profile": "quality"},
        allowed_model_ids=("moonshotai/kimi-k3",),
        available_model_ids=("moonshotai/kimi-k3",),
    )
    assert route.model_id == "moonshotai/kimi-k3"
    with pytest.raises(AutoRouteUnavailable):
        choose_remote_route(
            {"hawking_auto_profile": "premium"},
            allowed_model_ids=("provider/not-in-hawking-roster",),
            available_model_ids=("provider/not-in-hawking-roster",),
        )
