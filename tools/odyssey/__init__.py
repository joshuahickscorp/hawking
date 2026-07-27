"""Odyssey machinery: real reproductions, contract closures, and data membership.

This package is deliberately independent of the launch fence. Training stages
still refuse without authorization; baseline reproduction, contract tests, data
inventory and the contamination barrier must all be runnable while the fence
stays false.

It also inventories on-disk corpora, checks declared membership against reality,
ingests raw corpora into content-addressed training sets, and mechanically
rejects train/eval contamination. It does not download data.
"""

__all__ = [
    "SCHEMA_INVENTORY",
    "SCHEMA_MEMBERSHIP",
    "SCHEMA_BARRIER",
]

SCHEMA_INVENTORY = "hawking.odyssey.data_inventory.v1"
SCHEMA_MEMBERSHIP = "hawking.odyssey.membership_record.v1"
SCHEMA_BARRIER = "hawking.odyssey.contamination_barrier.v1"
