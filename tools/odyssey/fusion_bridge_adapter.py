"""Small Odyssey adapter for the canonical heterogeneous Fusion Bridge.

The accelerator package owns plan construction and validation, while
``physical_graph_compiler`` owns the compiler-receipt annotation.  This
module only preserves the Odyssey import boundary used by callers that load
the adapter from ``tools/odyssey`` directly; it does not create another plan
or validation implementation.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _plan_dict(plan: Any) -> dict[str, Any]:
    """Return the existing plan's serialized form without mutating it."""
    serializer = getattr(plan, "to_dict", None)
    if callable(serializer):
        value = serializer()
    else:
        value = plan
    if not isinstance(value, Mapping):
        raise TypeError("Fusion Bridge plan must provide to_dict() or be a mapping")
    return dict(value)


def annotate_compiler_output(
    compiler_output: Mapping[str, Any], plan: Any
) -> dict[str, Any]:
    """Attach a serialized Fusion Bridge plan through the compiler owner."""
    from physical_graph_compiler import attach_heterogeneous_plan

    return attach_heterogeneous_plan(dict(compiler_output), _plan_dict(plan))


def plan_is_valid() -> bool:
    """Check the canonical two-domain plan through its own validator."""
    from fusion_bridge import two_domain_plan, validate

    return bool(validate(two_domain_plan()).ok)


__all__ = ["annotate_compiler_output", "plan_is_valid"]
