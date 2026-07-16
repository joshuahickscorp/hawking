# Doctor V5 post-120B handoff

The live Ultra campaign remains the only execution authority. The post-120B
observer is read-only: it consumes the queue's campaign projection and the
reporter's immutable snapshots, but cannot launch cells, rewrite results, accept
report checkpoints, garbage-collect payloads, or delete sources.

Intermediate reporter generations are useful samples. They may motivate safety,
scheduling, and instrumentation improvements, provided completed evidence and the
source-bound scientific recipe remain immutable. Intermediate samples are never
final quality or dominance claims.

Final interpretation unlocks only when all 320 cells have a terminal status, both
the `sub-120B` and `120B` reports are complete, and both reporter checkpoints have
been accepted back into queue state with exactly matching paths, file hashes,
receipt hashes, report hashes, and covered-cell hashes. The queue must also be
quiescent with no active cells or children. Reporter and queue projections are
asynchronous by design, so their exact bridge is the queue-accepted reporter
receipt plus a recomputed covered-cell hash, not an impossible raw-file equality.
The observer records raw alignment separately as provenance. At that point it writes
`reports/condense/doctor_v5_ultra/post_120b/final_interpretation_packet.json`,
binding the exact campaign, reporter generation, group reports, accepted
checkpoints, and claim guards. It also writes
`reports/condense/doctor_v5_ultra/post_120b/CLAUDE_INTERPRETATION_HANDOFF.md`, a
fully injected interpretation contract that is safe to hand to Claude, Codex, or
another analysis runtime without manually assembling paths or relaxing gates.
Exact final plan, campaign, and reporter-index bytes are copied into immutable
`post_120b/final_inputs/` before the packet is sealed.

The current GPT-OSS 120B adapter is contract-only. Its typed builder and bounded
MXFP4 inspection are real, but live execution stays blocked until the adapter's
capability receipt reports reviewed STR2 campaign execution. The observer therefore
publishes two distinct forecasts: arrival at the 120B boundary and completion
through 120B. The former is a provisional dependency-aware throughput projection
using observed per-rate/per-branch seconds per billion and RAM reservations; its
range is heuristic, not a calibrated confidence interval. A separate worst-case
sensitivity gives every 32B and 72B cell the full 50 GB reservation, preventing
same-tier and mixed-tier overlap. Completion through 120B stays
null until all four GPT-OSS operations have reviewed registry entries, all 40 typed
runtime specs validate, and codec, runtime, and quality capabilities are true.
Structural GPT-OSS readiness is lifecycle-stable. Current disk admissibility is
reported separately for dependency-ready, unstarted heads using remaining output
bytes; output produced by an already-running cell cannot revoke structural readiness
or erase its ETA. Full source-content hashes remain a fail-closed worker-prelaunch
gate rather than being rehashed by the five-minute observer.

The local LaunchAgent runs the read-only observer every five minutes. A six-hour
task heartbeat refines ETA and health observations, advances only unbound scaffold
work, and consumes the final handoff when its gate becomes true. The production
auto-resumer stops relaunching once queue state is complete, so the frozen terminal
campaign cannot churn after completion.

Models beyond 120B are separate admission campaigns, not silent extensions of the
current 320-cell matrix. The scaffold records DeepSeek-V4-Flash, Kimi-K2.6, and
DeepSeek-V4-Pro as gated horizons; none is downloaded or executed without its own
architecture, source, disk, and streamed-receipt authority.

The separate default-off acceleration contract is documented in
`docs/plans/DOCTOR_V5_POST120_ACCELERATION.md`. It binds the exact GPT-OSS 10x4
matrix and the complete post-120B single-device profile/parallel/overlap/reuse/
RAM/swap/lifecycle/native/Metal/quality/rollback ideology without importing or
mutating the live campaign. Named higher-tier cells remain templates until exact
architecture, parameter, source, and admission authorities exist.
