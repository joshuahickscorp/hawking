"""Share ONE resident child process across successive HCLI engines.

MEASURED PROBLEM. The resident's 9.9 GB weight load is lazy inside the first
complete_payload and costs 211.3 s (call 1 = 241.1 s, call 2 = 29.8 s in the same
process). ResidentProcess.start() already guards correctly --

    if self._alive() and self._ready_payload is not None: return

-- but a NEW ResidentProcess is constructed per connector, per backend, per
engine, per cli_main call, so the guard never fires. Two cli_main calls in one
Python process measured two ResidentProcess.start calls and paid the load twice
(147.5 s and 144.2 s).

This memoises the ResidentProcess by the identity that decides which weights get
loaded, so successive engines in one process share the child.

Not landed in hcli/ yet: it changes process lifetime, which is exactly the kind
of thing that should be measured before it is made default.
"""
from __future__ import annotations

from typing import Any

_CACHE: dict[tuple, Any] = {}
_ORIG = None


def _key(config: Any) -> tuple:
    return (
        str(getattr(config, "selected_binary", lambda: "")() or ""),
        str(getattr(config, "artifact_root", "") or ""),
        str(getattr(config, "resident_binary", "") or ""),
        tuple(sorted((getattr(config, "fusion_env", None) or {}).items())),
        tuple(sorted((getattr(config, "runtime_env", None) or {}).items())),
    )


def apply() -> bool:
    """Memoise ResidentProcess construction. Returns False if already applied."""
    global _ORIG
    from hcli import hawking_native as N

    if _ORIG is not None:
        return False
    _ORIG = N.ResidentProcess

    class _Shared(N.ResidentProcess):  # type: ignore[misc,valid-type]
        def __new__(cls, config, *a, **k):
            k2 = _key(config)
            got = _CACHE.get(k2)
            if got is not None and got._alive():
                return got
            obj = super().__new__(cls)
            # NOT a dunder name: `obj.__x` inside the class body mangles to
            # `_Shared__x`, while getattr(self, "__x") does NOT mangle, so the
            # flag was written under one name and read under another and
            # __init__ never ran.
            obj._hawking_needs_init = True
            _CACHE[k2] = obj
            return obj

        def __init__(self, config, *a, **k):
            if getattr(self, "_hawking_needs_init", False):
                self._hawking_needs_init = False
                super().__init__(config, *a, **k)

    N.ResidentProcess = _Shared
    return True


def revert() -> bool:
    global _ORIG
    from hcli import hawking_native as N

    if _ORIG is None:
        return False
    N.ResidentProcess = _ORIG
    _ORIG = None
    return True


def live_children() -> int:
    return sum(1 for p in _CACHE.values() if getattr(p, "_alive", lambda: False)())
