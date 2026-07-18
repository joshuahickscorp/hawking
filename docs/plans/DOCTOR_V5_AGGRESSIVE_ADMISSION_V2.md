# Doctor V5 aggressive admission v2 (unbound scaffold)

Status: implemented and cheap-testable, but deliberately **not imported by the
active supervisor**. The current queue, autoresume entrypoint, activation marker,
runtime specifications, campaign state, and evidence remain untouched.

## What changes in the future generation

The v2 policy replaces whole-parent RAM guesses for pending cells with an exact
profile process-tree high-water. A sample is accepted only when its plan SHA,
current request SHA, 78 GB process envelope, root PID/PGID membership, process
count, and summed member RSS all agree. Accepted canonical sample hashes are
sealed. Profiles are exact across model tier, branch, and residency mode, so a
measurement from a lighter branch cannot silently underwrite a heavier branch.

A profile becomes calibrated after at least five authenticated samples spanning
24 seconds. Its reservation is:

`round_up_512MB(high_water + max(2GB, 15% of high_water))`

An uncalibrated profile gets the full 66 GB admission ceiling and therefore runs
as an exclusive canary. There is no optimistic fallback to the old whole-parent
projection. This makes the speed gain evidence-driven while remaining fail
closed.

The pending generation consumes and hash-binds
[`thread_profile_contract.py`](../../vendor/strand-quant/tools/thread_profile_contract.py),
one qualified profile file, its exact runtime binary, and every production
receipt. For every exact tier/rate entry the vendor contract requires all four
candidates—8, 12, 16, and 20 threads—then re-hashes the receipts and selects the
measured fastest RSS-admissible candidate. Admission uses that exact
`selected_threads`; there is no nominal-tier default, nearest-tier/rate fallback,
or scheduler override. Existing runtime specs are not rewritten in place.

The packer searches exact subsets under 66 GB RAM, 24 admitted CPU cores, and
eight lanes. Its primary objective is measured projected throughput,
`sum(1 / selected_wall_seconds)`, followed only by deterministic tie-breakers.
Thus 16+8 and 12+12 remain available when selected by their exact tier/rate
profiles, while a 20-thread lane remains exclusive and wins only when its
measured projected throughput beats every valid multi-lane pack.

## Controlled swap envelope

The swap baseline is sealed at quiescent promotion and can never ratchet upward
after a crash. Relative growth is treated as a bounded shock absorber:

- +512 MB or +256 MB/min: soft throttle, at most one reduced launch;
- +1,536 MB, +1,024 MB/min, or warning pressure: hard launch stop;
- +3,072 MB, 4,096 MB absolute, or critical pressure: stop launches and request
  one checkpoint/receipt-preserving largest-RSS shed;
- unknown probes: hard launch stop without blindly killing running evidence.

Recovery requires consecutive green samples plus 60/180-second cooldowns.
Emergency shedding is rate-limited to one lane per 60 seconds. Invalid or
tampered controller state self-heals from the separately sealed baseline into a
hard-stop cooldown; it does not re-baseline and does not invalidate evidence.
The existing aggregate RSS/OOM guard remains authoritative.

## Promotion boundary

The implementation is
[doctor_v5_aggressive_admission_policy.py](../../tools/condense/doctor_v5_aggressive_admission_policy.py).
Its `status` command is read-only. Its `stage` command writes only under
`reports/condense/doctor_v5_ultra/staged_acceleration/aggressive_v2/`.

Promotion must remain a single quiescent transaction:

1. Pause or drain and prove zero active children and free singleton/heavy leases.
2. Re-stage against the exact checkpoint state and full process-tree log.
3. Produce a qualified vendor tier/rate profile with exact production receipts
   for all 8/12/16/20 candidates and bind its exact runtime binary.
4. Generate pending-only runtime specs and a new queue/autoresume generation;
   bind every source and artifact hash.
5. Run adversarial parity, admission, swap, rollback, and crash-resume gates.
6. Atomically promote under the existing transaction journal. Never alter
   terminal evidence or completed results.

Rollback restores the pre-promotion pending runtime/queue generation and removes
the new activation keys. It never deletes parent sources or rewrites completed
evidence.

Cheap tests are in
[test_doctor_v5_aggressive_admission_policy.py](../../tools/condense/tests/test_doctor_v5_aggressive_admission_policy.py).
