"""Era V product surface: leave the developer checkout.

Hawking still has entry paths that assume they run inside this git tree.
`tools.odyssey.product_boundary` already resolves artifacts from a config file
and inventories the host; `hcli.cli.install_shims` already stamps a digest and
warns when a deployed copy drifts. This package is the fail-closed product
entry those two do not provide:

- a config that is missing or corrupt does not fall back to the checkout
- artifacts install/update under the configured product home, with a stamp
- the dominance scoreboard refuses to report an unmeasured value

It does not package, sign, or install a distribution. It does not write the
live ModelLake volume. It does not restart a worker.
"""
from __future__ import annotations

from tools.product.config import (
    ConfigClosed,
    machine_inventory,
    recovery_document,
    require_config,
)
from tools.product.install import (
    InstallError,
    artifact_digest,
    install_artifact,
    staleness,
    update_artifact,
)
from tools.product.scoreboard import (
    UnmeasuredError,
    load_scoreboard,
    qualify,
    require_measured,
)

__all__ = [
    "ConfigClosed",
    "InstallError",
    "UnmeasuredError",
    "artifact_digest",
    "install_artifact",
    "load_scoreboard",
    "machine_inventory",
    "qualify",
    "recovery_document",
    "require_config",
    "require_measured",
    "staleness",
    "update_artifact",
]
