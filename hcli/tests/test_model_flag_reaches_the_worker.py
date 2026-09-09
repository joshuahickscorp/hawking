"""`--model` was advertised and never arrived.

hcli/controller.py selects a model from self.model_info.path, else
HCLI_MODEL_PATH, else HCLI_HAWKING_NATIVE_CONFIG. `hcli resident start --model
<path>` set neither, so three start attempts died on "No model selected. Use
/models or /model <number|path>." with a valid path sitting in the config the
whole time. A flag the help text promises and the worker never sees is a defect.
"""
import inspect

import pytest

from hcli.agentos import resident


def test_the_worker_env_carries_the_configured_model():
    src = inspect.getsource(resident.ResidentSupervisor._spawn_worker)
    assert 'env["HCLI_MODEL_PATH"]' in src, (
        "_spawn_worker must put config.model into the env the controller reads, "
        "or --model is decorative")
    assert "config.model" in src


def test_an_explicit_environment_value_is_not_overridden():
    """An operator who exported HCLI_MODEL_PATH made a deliberate choice."""
    src = inspect.getsource(resident.ResidentSupervisor._spawn_worker)
    assert 'not env.get("HCLI_MODEL_PATH")' in src, (
        "the flag must not silently clobber an explicit export")


def test_a_bad_model_path_is_refused_at_START_not_at_first_completion(tmp_path):
    """BEHAVIOURAL, not a source grep.

    The two tests above read _spawn_worker's SOURCE for a string. That is the
    same source-text testing this campaign criticises elsewhere: it passes if
    the line exists and says nothing about what happens at runtime.

    resolve_model() returns whatever _info_from_path gives back, including
    None, so a typo or a path relative to the caller's shell (the worker runs
    with cwd=workspace) silently yields no model and the failure surfaces much
    later from inside a backend. Refuse at the door.
    """
    from hcli.agentos.resident import start_resident

    with pytest.raises(SystemExit) as e:
        start_resident(tmp_path, goal="anything", model="/nope/not/a/model")
    assert "does not resolve" in str(e.value)


def test_a_real_model_path_is_not_refused(tmp_path):
    """Negative control: the check must not reject a valid path."""
    from hcli.models import resolve_model
    real = ("/Volumes/corpdrive/hawking-modellake/specimens/"
            "Qwen--Qwen3-0.6B@c1899de289a0")
    if resolve_model(real) is None:
        pytest.skip("the reference model is not mounted on this host")
    from hcli.agentos.resident import start_resident
    try:
        start_resident(tmp_path, goal="anything", model=real)
    except SystemExit as exc:
        pytest.fail(f"a valid model path was refused: {exc}")
    except Exception:
        pass  # anything past validation is out of scope here


def test_the_controller_still_reads_that_variable():
    """If the controller stops reading it, the fix above becomes a no-op."""
    import hcli.controller as controller
    src = inspect.getsource(controller)
    assert "HCLI_MODEL_PATH" in src, (
        "the controller no longer reads HCLI_MODEL_PATH; the resident's env "
        "hand-off is now pointing at nothing")
