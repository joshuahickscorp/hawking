# Doctor V5 physical adapter authority

This document describes the source-reviewed issuer/verifier boundary. It does
not assert that any physical adapter or physical result exists.

## Current state

- The issuer contract is ready to consume concrete release artifacts.
- The independent verifier is implemented.
- The production envelope is absent, so verified production descriptors are
  **0/10** and the physical score remains **0/10**.
- No registry grants execution, requests activation, changes a runtime default,
  opens a model, or contributes a physical point.

The cheap status command is:

```sh
python3.12 tools/condense/doctor_v5_physical_adapter_registry.py status
```

`issuer_ready_for_concrete_inputs=true` means only that the pinned trust root,
SSHSIG tool, schemas, and issuer/verifier code are available. It must never be
reported as production-descriptor or physical-evidence readiness.

## Exact production population

The first production registry must contain exactly one source-bound descriptor
for each of these sub-120B Doctor facets:

1. `release_authority`
2. `thread_profiles`
3. `block_parallel`
4. `ordered_overlap`
5. `bounded_reuse`
6. `ram_swap_recovery`
7. `native_io_pgo`
8. `disk_lifecycle`
9. `full_stack_parity_ab`
10. `post120_appendix_bindings`

Every descriptor binds exact baseline and candidate executable bytes, unique
role-specific argv manifests, a unique facet execution scope, a unique launch
contract, the current controller plan/source hashes, the scientific receipt
schema, and the reviewed validator ABI. Programs may be shared only when their
bytes really are identical. Argv, scope, and launch documents may not be
silently reused within a segment/model group; scope and launch documents may
not be reused across groups.

The sub-120B scope is exactly `sub-120b-doctor / Doctor-V5 /
3B-through-72B`. GPT-OSS is exactly `gpt-oss-120b / GPT-OSS / 120B`.
Higher-tier descriptors require a numeric tier strictly greater than 120B and
an exact model identity; a display name or generic `>120B` tier is insufficient.

## Owner-free release sequence

Do not perform this sequence until Doctor is final-ready, the inherited heavy
lease proves zero owners, and the release handoff authorizes concrete builds.

1. Build and independently validate the concrete programs, argv manifests,
   source-unit manifests, execution scopes, launch contracts, input manifests,
   and counter authority.
2. Write a ten-entry request list using schema
   `hawking.doctor_v5_physical_adapter_entry_request.v2`.
3. Read the current `plan_sha256` and source `manifest_sha256` from
   `doctor_v5_physical_ab_controller.py --plan`.
4. Draft an inert registry:

   ```sh
   python3.12 tools/condense/doctor_v5_physical_adapter_registry.py draft \
     --entry-requests <EXACT_REQUESTS.json> \
     --plan-sha256 <CURRENT_PLAN_SHA256> \
     --source-manifest-sha256 <CURRENT_SOURCE_MANIFEST_SHA256> \
     --valid-seconds 86400 \
     --output <IMMUTABLE_DRAFT.json>
   ```

5. Independently review the draft and every bound file. Signing revalidates all
   files, derives the current controller bindings independently, proves that
   the private key matches the compiled signer, signs canonical bytes, and
   verifies the resulting SSHSIG before sealing the envelope:

   ```sh
   python3.12 tools/condense/doctor_v5_physical_adapter_registry.py sign \
     --registry <IMMUTABLE_DRAFT.json> \
     --private-key <OUT_OF_REPOSITORY_RELEASE_KEY> \
     --signature-output <IMMUTABLE_SIGNATURE> \
     --envelope-output <IMMUTABLE_ENVELOPE.json>
   ```

6. Verify using current source rather than caller-provided hashes:

   ```sh
   python3.12 tools/condense/doctor_v5_physical_adapter_registry.py verify \
     --envelope <IMMUTABLE_ENVELOPE.json>
   ```

7. Only a verification receipt with `signature_verified=true`,
   `exact_ten_facet_sub120_verified=true`, ten verified adapters, no errors,
   `execution_granted=false`, and `physical_execution_claimed=false` may be
   considered by the separately gated executor. The executor still requires
   final-ready observer authority, zero owners, inherited lease, direct
   counters, resource admission, and a concrete launch bundle.

## Fail-closed properties

Verification rejects missing, partial, unsigned, expired, future-issued,
overlong, reordered, stale, mutated, symlinked, duplicate-ID, duplicate-scope,
duplicate-launch, wrong-signer, wrong-namespace, wrong-plan, wrong-source,
cross-facet, cross-segment, self-hash-forged, shell-enabled, or activation-
requesting registries. Executor adapter lookup never falls back from an unknown
or post-120B scope to the sub-120B registry.

Focused structural verification (no model, GPU, Metal, Cargo, or corpus use):

```sh
python3.12 -m pytest -q \
  tools/condense/tests/test_doctor_v5_physical_adapter_registry.py \
  tools/condense/tests/test_doctor_v5_physical_ab_controller.py \
  tools/condense/tests/test_doctor_v5_physical_ab_executor.py
```
