# Hawking perception contract

**Status:** current local contract

Hawking owns one local perception surface. It is implemented under
`hawking.perception` and exposed through `hawking.perception_adapter`; there
is no second visual-runtime package, proxy process, or public command tree.

## Surface and ownership

The read-only tool allowlist is deliberately small:

- `system.doctor`, `project.status`
- `vision.capabilities`, `vision.observe`, `vision.query`, `vision.verify`
- `vision.progress`, `vision.list_artifacts`, `vision.get_artifact`

`hawking.perception.acts` is the compact human/API surface. Its connected acts
are `see`, `hold`, `know`, `check`, and `prove`; `open`, `make`, `fix`, and
`keep` remain explicitly PARKED until Hawking owns and verifies a local
implementation. A parked act returns its reopening condition and never a
synthetic success.

The organs are intentionally narrow and lazy:

| Owner | Local behavior |
|---|---|
| `hawking.perception.file_eye` | magic/header classification and content identity |
| `hawking.perception.terminal` | real PTY capture, or an honest blocked result |
| `hawking.perception.doctor` | bounded subprocess diagnosis with network and dangerous-command refusal |
| `hawking.perception.store` | project metadata and content-addressed local artifacts |
| `hawking.behavior` | isolated BHV-01 through BHV-23 evidence matrix |

The normal `import hawking` and `python3 -m hawking --help` paths do not load
these organs. A caller pays only for the requested Hawking leg.

## Evidence boundary

Perception can observe local files, run bounded local diagnostics, and retain
receipts. It cannot certify model quality, hardware performance, browser
access, 3D validation, repair, or artifact generation merely because a name is
present. Those capacities remain PARKED until their owner, test, and receipt
exist.

Current inspection commands:

```bash
python3 -m hawking.perception.acts --selftest
python3 -m hawking.perception.acts --disposition
python3 -m hawking perception-gate --help
```

The perception gate writes an explicit result only when requested and uses the
Hawking-local source and tool registry as evidence. It does not start a model
runtime or make a GPU claim.

## Verification

The focused checks are:

```bash
python3 -m pytest -q \
  hawking/tests/test_perception_file.py \
  hawking/tests/test_perception_doctor.py \
  hawking/tests/test_perception_organs.py \
  hawking/tests/test_perception_terminal.py \
  hawking/tests/test_perception_no_network.py
```

Historical receipts and source hashes retain their original spellings as
provenance. They are read-only evidence, not a second runtime or an invitation
to restore retired package names.

<!-- DOC_STATUS: CURRENT -->
