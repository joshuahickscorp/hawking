## HAWKING ASCENT — standing laws for every lane

Repo: /Users/scammermike/Downloads/hawking . Build dir is `workspace/ops/build/rust`
(set by .cargo/config.toml). NEVER use `target/` or `target-parallel/` — stale
binaries there still run and have produced false results before.

Build: `cargo build --profile release-fast -p hawking-core --example <name>`
Binaries: `workspace/ops/build/rust/release-fast/examples/<name>`

### Resource discipline (MANDATORY)
This machine runs several lanes at once. Any command that touches the GPU or
allocates more than ~8 GiB MUST be wrapped:

    ./tools/gpu_lane_lock.sh <your-lane-name> <your command...>

It is a mutex; it blocks until free (90 min cap). Compiling, reading, static
analysis and unit tests do NOT need it. Never bypass it — an unlocked benchmark
run silently corrupts another lane's timing.

### Measurement law
- A single Metal run is page-cache confounded. Any timing claim needs >= 3
  alternating paired reps (A,B,A,B,A,B) and you must report the full spread,
  not just the median.
- GPU time means `MTLCommandBuffer.GPUEndTime - GPUStartTime` after wait.
  A CPU wall-clock wait is NOT GPU time; never report it as such.
- Label every number DIRTY_ENGINEERING (other lanes running), CLEAN_CANDIDATE,
  or BASE_TRUE. Do not launder a dirty number into a clean claim.
- Report ns/token, not just tok/s.

### Correctness law
- Bit-identity or a stated numeric-equivalence gate is required for every
  optimization. "Looks close enough" is a rejected result.
- 0 fallbacks. If a fast path silently falls back, that run is invalid.
- Never weaken an existing gate, assertion, or seal to make something pass.
  If a gate blocks you, report it as a finding — do not edit it away.

### Negative science — do NOT re-pay for these
- Q80 cross-expert shared-basis: REFUTED (experts mutually orthogonal, cos 0.004).
- Q80 "simply bandwidth-bound": REFUTED. Measured 0.79% of the 700-800 GB/s
  ceiling with ~51% GPU idle. It is dispatch/host bound, not bandwidth bound.
- DSV4F route-ID readback serializer hypothesis: REFUTED.
- Shader compile as the primary current wall: REFUTED / deprioritized.
- Single-family Q80 representation: INSUFFICIENT. gate_proj/up_proj/down_proj
  each prefer a different codec family; down_proj inverts the ranking and needs
  post-SwiGLU X, not the layer hidden.
- Q30 static <=1.5 coherence: FAILED. Do not copy the Q30 approach.
- Immutable-identity recomputation (SHA, st_dev, geometry parse, manifest scan)
  per token has repeatedly been the real latency. Suspect it early.
- Giant JSON indexes are a real iteration wall (1.38 GB capture-result.json).
  Do not add one.

### Reporting
End your final message with:

    LANE: <name>
    STATUS: SHIPPED | PARTIAL | BLOCKED
    BASELINE_NS_PER_TOKEN: <n> (label)
    RESULT_NS_PER_TOKEN: <n> (label)
    REPS: <the actual paired numbers>
    CORRECTNESS: <bit-identical | numeric gate + measured drift | N/A>
    FILES: <paths touched>
    RECEIPT: <path to json receipt you wrote under receipts/ascent-2026-08-16/>
    NEXT_BOTTLENECK: <what is now the top cost, with its measured ns>

Commit your work on your branch before finishing. Uncommitted lanes have been
lost here before.

---
# LANE: frontier-fs-per-weight
## Class: COMPILE / LIGHT_CONTROL, one GPU run under the lock.

## The unit change

tok/s is model-size-dependent and useless for comparing architectures or for
seeing the density/runtime joint frontier. The physically meaningful units are
**per served weight** and **per moved bit**.

Computed floor for Q80 (receipts/ascent-2026-08-16/PHYSICAL_FLOOR.json):

    active weights/token         3,562,274,816
    819 GB/s = 6.55 Tbit/s  ->   0.153 ps per bit moved (FIXED by the hardware)

    BPW      floor/token     fs/weight    ceiling
    4.259241   2315.7 us       650.07      432 tok/s
    1.392467    757.1 us       212.53    1,321 tok/s
    1.000000    543.7 us       152.63    1,839 tok/s
    0.500000    271.8 us        76.31    3,679 tok/s
    0.250000    135.9 us        38.16    7,357 tok/s

Q80 currently serves a weight every ~113 PICOseconds (403 ms / 3.562 G). The floor
at 1.392467 BPW is 212 FEMTOseconds. That is the 532x gap, expressed in the unit
that makes it comparable across models.

## Deliverable
Emit, for every timed run of BOTH models, alongside the existing ns figures:

    fs_per_weight_served        = token_ns * 1e6 / active_weights_per_token
    ps_per_bit_moved            = token_ns * 1e3 / bits_moved_per_token
    fs_per_weight_floor         = bytes_per_token / bandwidth / active_weights * 1e15
    distance_from_floor         = achieved / floor
    pJ_per_weight_served        = joules_per_token * 1e12 / active_weights   (see below)

Wire it into the joint TOKEN_NS schema that `joint-token-ns` just landed
(`crates/hawking-core/src/token_ns/`, merged e3068e69) rather than a parallel path.
Derive `active_weights_per_token` from the model geometry, not a constant, so DSV4F
gets a correct value too (43 layers, top-6 of 256, MLA, shared expert).

## HONESTY REQUIREMENT — this is the part that must not be got wrong
212 fs/weight is **amortized service time under concurrency**, NOT latency. A single
weight's real DRAM round trip is ~100 ns. We reach femtoseconds only because
thousands of weights are in flight at once.

Label the field `fs_per_weight_served (amortized throughput-derived, NOT latency)`
and put that caveat in the receipt itself. A reader must not be able to mistake it
for a claim of femtosecond latency, which would be false. This lane is worthless if
it produces a number that invites that misreading.

## Energy
`powermetrics` requires root and this session cannot run it. So:
  1. Implement `pJ_per_weight_served` behind an optional joules input, and emit
     `energy_available: false` with the reason when it is absent.
  2. Investigate whether IOReport / IOKit exposes SoC or DRAM energy counters
     WITHOUT root on this machine. If a non-root path exists, use it and say so.
  3. If not, document the exact command a human must run
     (e.g. `sudo powermetrics --samplers gpu_power -i <ms> -n <count>`), so the
     number can be filled in later without re-deriving anything.
Do not fabricate an energy figure or copy a datasheet number and present it as
measured.
