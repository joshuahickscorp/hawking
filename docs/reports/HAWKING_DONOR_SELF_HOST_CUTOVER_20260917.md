# Hawking donor / self-host cutover — 2026-09-17

## Scope and authority

This record covers the bounded graft of
`HAWKING_DONOR_SELF_HOST_2026-09-17.zip` into the current dirty checkout at
`/Users/scammermike/Downloads/hawking`. The donor was treated as a tested
source corpus, not as a production release. Current Hawking source, the live
daemon, focused tests, and the donor's own rollback warnings remain the
acceptance authority.

Existing dirty work, worktrees, receipts, Flash/Pulsar/ModelLake lanes, and
the running daemon were preserved. No reset, merge, worktree deletion, daemon
replacement, credential migration, or donor wholesale replacement was done.

## Adopted cutover

The donor's G6 public-entry change is adopted in the canonical Python CLI:

- bare `h`/`hawking` enters `hawking.web_launcher`;
- `h chat` enters the same read-mode launcher;
- `h web` remains a compatibility spelling for that same launcher;
- `h build` enters the same launcher with a server-owned Build authority
  profile and short-lived handoff;
- no public Web path invokes `hcli serve`, starts resident Kimi P0, revives
  MLX, or creates another daemon/scheduler.

The existing control-plane `h status` route remains available for local
Hawking status, while delegated mission status keeps its existing argumented
form. The change is routing-only; `hawkingd`, Web release ownership, model
catalog, credential resolution, Goal state, and mutation/evidence owners remain
where they were.

## Explicitly deferred donor layers

The donor's G1/G2/G3/G4/G5 code was not copied into the live tree.

- G1's donor H-NOTES path imports the donor-only `native_core` reader and
  assumes `native/libhawking`; the live checkout has neither as an accepted
  owner. The live Python H-NOTES/ProjectStore/FTS implementation remains the
  current owner.
- G2's full registry surface depends on donor-only evidence/runtime owners and
  would publish registrations without a live owner if copied piecemeal. The
  live source/tool registry remains authoritative.
- G3 native physical core, G4 machine-graph/evidence migration, and G5
  Gravity/science/NX candidates remain explicitly partial in the donor and
  were not promoted into production.

The isolated donor verifier was run before adoption: 60 new Python contracts,
native CTest (2,061 assertions), and the synthetic demos passed after the
donor native library was built. That proves the donor snapshot in isolation;
it does not qualify those partial layers on this macOS Hawking runtime.

## Live evidence

With `OPENROUTER_API_KEY` unset, all four public entry forms were exercised:

```text
h                         PASS (browser opener suppressed for shell check)
h chat --no-browser       PASS
h build --no-browser      PASS
h web --no-browser        PASS
h status                  PASS (canonical control-plane JSON)
```

All reused the existing `hawkingd` owner at PID `50440`, selected current Web
release `web-6347550e08ae`, resolved the OpenRouter credential through the
canonical Keychain resolver, and reported Auto availability. No credential or
Authorization header entered the output.

The current health owner reports `hawkingd`, `single_surface=true`, and the
qualified H-Web capability/builder revisions. The current Web release is
`web-6347550e08ae`; no candidate is active. The daemon currently reports
`waiting_for_native_runtime`, which is the existing honest local-runtime
boundary and was not changed by this graft.

## Verification

- Focused launcher, H-Web control, and OpenAI-compatible surface tests:
- Focused launcher, H-Web live contract, capability projection,
  attachments/memory, controls, and OpenAI-compatible surface tests:
  **76 passed**.
- `python3 -m compileall -q hawking`: passed.
- `git diff --check`: passed.
- Live Keychain-only aliases and health/release probes: passed.

The repository is intentionally dirty from prior user work; the full dirty
diff is not attributable to this graft. The targeted graft diff is limited to
the canonical CLI routing and its launcher regression coverage, plus this
status record.

## Next boundary

The current Hawking Web/self-host surface is the only donor capability
promoted by this cutover. G1/G2 may be evaluated later as separate bounded
adoptions only after their missing live owners and rollback tests are present.
G3/G4/G5 remain candidate research infrastructure, not production Hawking
authority. No further donor promotion is implied by this record.

`HAWKING_DONOR_SELF_HOST_CUTOVER=G6_ONLY`
