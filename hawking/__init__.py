"""Hawking product package — command surface / UI / status rendering.

Canonical import name: ``hawking``. There is no second public control-plane
package.  The former AgentOS implementation is a lazy Hawking leg: its public
symbols resolve from this root without making ordinary imports initialize the
runtime graph.  Runtime lives as ``hawking.runtime`` / ``hawking.engine`` /
``hawking.backends``.

``Controller``, ``Workspace``, ``Event``, ``EventBus``, and the core
perception helpers are PEP 562
lazy so ``python3 -m hawking --help`` does not import the runtime graph.
``from hawking import Controller`` still works.
"""
from .cli import parse_hawking_args, main

__all__ = [
    "parse_hawking_args",
    "main",
    "Workspace",
    "Controller",
    "Event",
    "EventBus",
    "Hawking",
    "AgentOS",
    "observe",
    "classify_bytes",
    "create_perception_server",
]

_LAZY_ATTRS = {
    "Workspace": (".workspace", "Workspace"),
    "Controller": (".controller", "Controller"),
    "Event": (".events", "Event"),
    "EventBus": (".events", "EventBus"),
    "observe": (".perception", "observe"),
    "classify_bytes": (".perception", "classify_bytes"),
    "create_perception_server": (".perception", "create_server"),
    # Hawking is the public name for the durable orchestration class.  Keep
    # AgentOS as an import-compatible spelling while no public command or
    # receipt requires callers to know that implementation leg.
    "Hawking": (".agentos.runtime", "Hawking"),
    "AgentOS": (".agentos.runtime", "AgentOS"),
}


def __getattr__(name: str):
    spec = _LAZY_ATTRS.get(name)
    from importlib import import_module

    if spec is not None:
        mod_name, attr = spec
        value = getattr(import_module(mod_name, __name__), attr)
    else:
        # One public root, many lazy implementation legs.  This deliberately
        # does not import AgentOS during ``import hawking`` or ``--help``;
        # callers that request an operational symbol pay only for that leg.
        legacy_leg = import_module(".agentos", __name__)
        if name not in legacy_leg.__all__:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(legacy_leg, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
