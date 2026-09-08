# Sublating VisionMCP into Agent OS

Goal: hawking stops importing a foreign package. `vmcp.*` becomes Agent OS
capability, not a proxy to a separate repo. Only after that is proven do the
traces get deleted.

## Why this is smaller than 299k lines

`hcli/vmcp_adapter.py` is the ONLY file in hawking that touches visionmcp.
Everything else goes through it:

    hcli/connectivity.py:131          inspect_vmcp
    hcli/tool_registry.py:1716        inspect_vmcp        (vmcp.inspect)
    hcli/tool_registry.py:1846        VMCP_READ_ONLY_TOOLS, _source_tools (vmcp.tools)
    hcli/tool_registry.py:1875        call_vmcp           (vmcp.query)

So the port does not have to reproduce visionmcp. It has to satisfy four
imports and one calling convention.

## The exact seam

    vmcp_adapter.py:108   visionmcp.api.public_api_versions()      -> list
    vmcp_adapter.py:112   visionmcp.profiles.list_profiles(...)    -> list
    vmcp_adapter.py:116   visionmcp.mcp.factory.create_server      -> callable
    vmcp_adapter.py:223   create_server(projects_root, profile=)   -> host
                          host._tool_manager.get_tool(name)        -> registered
                          await registered.run(arguments)          -> value

Replace those and hawking is free of the dependency.

## The 9 tools that must survive the port

    system.doctor          project.status         vision.capabilities
    vision.observe         vision.query           vision.verify
    vision.progress        vision.list_artifacts  vision.get_artifact

These are the current allowlist (`VMCP_READ_ONLY_TOOLS`). Anything outside it
is already refused by `call_vmcp`, so the port owes nothing for the other ~294
tools visionmcp ships.

## Acceptance -- the port is done when ALL of these hold

1. `grep -rn "visionmcp" hcli/ tools/ --include="*.py"` returns nothing outside
   comments and historical receipts.
2. `vmcp.tools` lists the same 9 names it lists today.
3. `vmcp.query` on each of the 9 returns a result of the same shape, with the
   `visionmcp/` directory RENAMED AWAY so the foreign package cannot be found.
   That rename is the real test -- an import that silently falls back to the
   still-present checkout proves nothing.
4. `hcli agentos vmcp-gate` passes with the directory still renamed away.
5. The full hcli suite is green.

## Order of operations -- do not reorder

1. Port, with `visionmcp/` still present.
2. Rename `visionmcp/` aside. Re-run acceptance. THIS is the proof.
3. Only then delete `hcli/agentos/vmcp/` (the dead local reimplementation).
4. The GitHub repo is the user's to delete, not this campaign's. Before that:
   47 commits exist ONLY in the local checkout (fidelity 7, fidelity-apply 1,
   several grok/* lanes), none of them pushed. Everything on the remote IS
   already local (origin/main 367, release 6, codex/world-engine 213 -- zero
   commits unreachable locally), so the remote is a backup, not a source of
   truth. Deleting it makes the 1.2 GB local checkout a single point of failure.
   The source is 19 MB; the other 1.18 GB is artifacts/ (624 MB), tools/ (92 MB)
   and benchmarks/ (29 MB).

## What must NOT happen

Deleting `hcli/agentos/vmcp/` BEFORE the port. That subsystem is the only local
implementation of file classification, PTY capture and subprocess profiling in
the repo. It is dead, but it is also the head start. Salvage first, delete second.

## Salvage map (measured, not estimated)

An earlier read of this subsystem said "delete all 9,564 lines". That was
wrong. Half of it has no foreign dependency at all and is the head start for
the port.

### KEEP -- pure, stdlib + siblings only, 2,951 lines

| module | lines | capability |
|---|---|---|
| receipt.py | 169 | shared primitives: digests, tool receipts, the network/dangerous refusal lists. Everything below depends on it. |
| pty_eye.py | 469 | real POSIX PTY capture (openpty / posix_openpt), honest PARKED on EPERM instead of a silent pipe fallback |
| tool_doctor.py | 369 | subprocess profiling into a signed receipt, refusing network and dangerous commands BEFORE exec |
| file_eye.py | 811 | magic-byte classification across 12 formats with no PIL and no python-magic, plus a geometry cross-check against an independent tool |
| disposition.py | 1,133 | the nine-act dispatcher routing to the above; coupled to hawking (tools.future._common), NOT to visionmcp |

### DELETE -- dies with the foreign package, 5,019 lines

capability_probe.py (2,176) adversarially tests visionmcp's own MCP tool
surface; forgery_canary.py (1,174) implements a visionmcp SensorAdapter and
attacks its CaptureBus tamper detection; lattice_disposition.py (1,669) proves
visionmcp's internal evidence_graph / worldir / memory / repair models by name.
None has meaning once visionmcp is gone.

### KEEP-with-caveat / REWRITE

behavior_lab.py (780) runs a real 23-fixture behavioural matrix over the pure
organs, but hard-depends on tools.future.tabula (2,077 lines, pulls numpy and
hcli.workunit). Keep only if that scorer comes too.

hcli_integration.py (364) should not be ported. The idea worth keeping is its
SUBJECT_MISMATCH check: a verifier must bind evidence to a specific subject
path, not just compare digests, or a replay of identical-content-different-file
evidence is silently accepted. disposition.py's know() already carries the
successor logic (same_bytes vs same_path).

### The tests come with the code

25 tests, 424 lines, all passing, and genuinely behavioural rather than shallow:
test_file_eye flips one magic byte and requires the classified kind to change;
test_image_geometry monkeypatches the eye to report a height one pixel off and
requires the cross-check to flip to DISAGREE; test_no_network AST-walks every
module and fails on any socket/urllib/requests import. Passing behavioural tests
for salvaged code are worth more than the code.

### Receipts: the orphaning already happened once here

tools/acceptance/vmcp/ (2,917 lines) was deleted in 6a567369c, leaving 12
receipts under receipts/acceptance/VMCP_*.json with no producer. Deleting the
three visionmcp-coupled modules would orphan four more the same way
(VMCP_CAPABILITY_SURFACE, VMCP_FORGERY_CANARY, VMCP_LATTICE_DISPOSITION,
VMCP_AGENTOS_INTEGRATION). Delete or archive those alongside the code rather
than leaving free-floating evidence for gates that no longer run.
disposition.py's own two receipts stay live, because disposition.py is a KEEP.
