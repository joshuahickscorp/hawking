"""`--model` was advertised and never arrived.

hcli/controller.py selects a model from self.model_info.path, else
HCLI_MODEL_PATH, else HCLI_HAWKING_NATIVE_CONFIG. `hcli resident start --model
<path>` set neither, so three start attempts died on "No model selected. Use
/models or /model <number|path>." with a valid path sitting in the config the
whole time. A flag the help text promises and the worker never sees is a defect.
"""
import inspect

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


def test_the_controller_still_reads_that_variable():
    """If the controller stops reading it, the fix above becomes a no-op."""
    import hcli.controller as controller
    src = inspect.getsource(controller)
    assert "HCLI_MODEL_PATH" in src, (
        "the controller no longer reads HCLI_MODEL_PATH; the resident's env "
        "hand-off is now pointing at nothing")
