# Hawking completion dependency

Ramanujan is not an independent launch lane.  Its current state is
`BLOCKED_ON_HAWKING_COMPLETION`, declared in
`governance/boundary/HAWKING_COMPLETION_GATE.json` and corroborated by the
existing non-production governance and green-light records.

The existing [handoff contract](../../workspace/campaign/evidence/systems/ramanujan/RAMANUJAN_HANDOFF_CONTRACT.json)
names the exact trigger: `HAWKING_EVOLUTION_COMPLETE`.  It is currently
`PREPARED_NOT_EXECUTED` with `may_execute_now: false`; Ramanujan's status and
parent-restream guard therefore fail closed before any launch evaluation.

Before Hawking completion, it is appropriate to improve deterministic local
scaffolding, fixture tests, data-validation code, and audit readers.  Those
activities do not produce research authorization, production authority, or a
parent-restream right.

The dependency is satisfied only by validated Hawking completion evidence plus
the owner and production evidence required by the active Ramanujan contracts.
It cannot be satisfied by changing a Ramanujan JSON file, passing local tests,
or running a controller.  Until then, live research, teacher traces, parent
restream, and production launch remain blocked.

The Q0 clean-container material is retained as historical, offline evidence.
It proves only the bound offline replay it records; it is not a Hawking
completion certificate and it grants no present authority.
