# Specimen-events disposition

`tools/future/specimen_events.py` was a one-shot, static event/workgraph
producer. Its `SPECIMEN_EVENTS.json` receipt is retained as the historical
record. No executable path imported it: Doctor keeps the small
`fingerprint_from_config` rule locally, while ModelLake/HCLI own current
watching, admission, and work-unit state.

The producer is therefore retired. Git preserves its full implementation and
the sealed receipt preserves the transition, replay, and bounded-workgraph
semantics without keeping a second future authority alive.
