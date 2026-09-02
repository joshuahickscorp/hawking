"""Audit tools. Classification and inventory only; they do not modify classified modules.

The reachability inventory also exposes a compact HCLI-consumable surface
(capability.discover / capability.inspect / capability.invoke) so dormant
modules can be called without editing hcli/. An import is not a call site.
"""
