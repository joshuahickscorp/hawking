"""HCLI product package — command surface / UI / status rendering.

Canonical import name: ``hcli``. There is no second HCLI package.
Ownership packages: ``hcli.agentos``, ``hcli.genomes``,
``hcli.doctor``, ``hcli.gravity``, ``hcli.vmcp``. Runtime lives
as ``hcli.runtime`` / ``hcli.engine`` / ``hcli.backends``.

``Controller``, ``Workspace``, ``Event``, and ``EventBus`` are PEP 562
lazy so ``python3 -m hcli --help`` does not import the runtime graph.
``from hcli import Controller`` still works.
"""
from .cli import parse_hcli_args, main

__all__ = ["parse_hcli_args", "main", "Workspace", "Controller", "Event", "EventBus"]

_LAZY_ATTRS = {
    "Workspace": (".workspace", "Workspace"),
    "Controller": (".controller", "Controller"),
    "Event": (".events", "Event"),
    "EventBus": (".events", "EventBus"),
}


def __getattr__(name: str):
    spec = _LAZY_ATTRS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    mod_name, attr = spec
    value = getattr(import_module(mod_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
