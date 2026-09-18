"""`--model` was advertised and never arrived.

hawking/controller.py selects a model from self.model_info.path, else
HAWKING_MODEL_PATH, else HAWKING_NATIVE_CONFIG. `hawking resident start --model
<path>` set neither, so three start attempts died on "No model selected. Use
/models or /model <number|path>." with a valid path sitting in the config the
whole time. A flag the help text promises and the worker never sees is a defect.

The resident admits only validated Hawking-native profiles, so an MLX directory
cannot become a second model owner under a Hawking daemon.
"""
import inspect

import pytest

import hawking.resident as resident


def test_the_worker_env_carries_the_configured_model():
    src = inspect.getsource(resident.ResidentSupervisor._spawn_worker)
    assert 'env["HAWKING_MODEL_PATH"]' in src, (
        "_spawn_worker must put config.model into the env the controller reads, "
        "or --model is decorative")
    assert "config.model" in src


def test_an_explicit_environment_value_is_not_overridden():
    """An operator who exported HAWKING_MODEL_PATH made a deliberate choice."""
    src = inspect.getsource(resident.ResidentSupervisor._spawn_worker)
    assert 'not env.get("HAWKING_MODEL_PATH")' in src, (
        "the flag must not silently clobber an explicit export")


def test_a_bad_model_path_is_refused_at_start_not_at_first_completion(tmp_path):
    """BEHAVIOURAL, not a source grep.

    The two tests above read _spawn_worker's SOURCE for a string. That is the
    same source-text testing this campaign criticises elsewhere: it passes if
    the line exists and says nothing about what happens at runtime.

    A typo or a path relative to the caller's shell must fail before the
    detached supervisor is created. Refuse at the door.
    """
    from hawking.resident import start_resident

    with pytest.raises(SystemExit) as e:
        start_resident(tmp_path, goal="anything", model="/nope/not/a/model")
    assert "native Hawking profile" in str(e.value)


def test_legacy_mlx_model_path_is_refused_before_any_supervisor_spawn(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy-mlx"
    legacy.mkdir()
    (legacy / "config.json").write_text("{}\n")
    (legacy / "model.safetensors").write_bytes(b"fixture")

    def forbidden_spawn(*_args, **_kwargs):
        raise AssertionError("legacy MLX path reached supervisor spawn")

    monkeypatch.setattr(resident.subprocess, "Popen", forbidden_spawn)
    from hawking.resident import start_resident

    with pytest.raises(SystemExit) as exc:
        start_resident(tmp_path, goal="anything", model=str(legacy))
    assert "refuses legacy MLX" in str(exc.value)


def test_the_controller_still_reads_that_variable():
    """If the controller stops reading it, the fix above becomes a no-op."""
    import hawking.controller as controller
    src = inspect.getsource(controller)
    assert "HAWKING_MODEL_PATH" in src, (
        "the controller no longer reads HAWKING_MODEL_PATH; the resident's env "
        "hand-off is now pointing at nothing")
