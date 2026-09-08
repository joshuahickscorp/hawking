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
