# Restart-proof next-goal prompt

Use the following prompt verbatim in the next Codex goal if this task loses
state after a reboot:

> Continue the existing Doctor V5 physical-evidence and post-120B goal in
> the Hawking repository root without redefining success. Read
> `docs/plans/APPENDIX_HANDOFF.md`,
> `docs/plans/DOCTOR_V5_POST120_ACCELERATION.md`, and the current sealed
> unbound handoffs before acting.
>
> First run `tools/condense/doctor_v5_post_120b.py sync`, read the current
> observer state, and verify the detached accelerated supervisor by PID,
> process-start time, command hash, and pinned source hashes. Verify the live
> RAM/swap/thermal/disk guard and inventory every heavy owner. If Doctor
> `final_interpretation_ready` is false or any heavy owner exists, do not
> mutate the live queue, worker, adapters, campaign plan, adapter registry,
> runtime specs, completed evidence, results, Appendix corpus, staged release
> state, or runtime defaults. During that period run only cheap, unbound work
> with no model, GPU, Metal, Cargo build, or corpus bandwidth.
>
> Completion has two ordered parts. Part 1 is not complete until the Doctor
> physical controller reports a genuine `10/10`: ten real, hash-bound,
> operator-signed physical receipts from the exact release host and artifacts.
> Structural tests, fixtures, simulations, estimates, dry runs, and synthetic
> receipts always count as zero physical points. Require the operator-signed
> sealed-evidence chain, result draft -> sign -> seal verification, pinned
> signer and expiry, immutable final references, inherited lease proof,
> corpus/source/build/rollback CAS, trusted direct `proc_pid_rusage` process
> joules, exact Metal GPU-time/traffic evidence, and all parity/zero-skip gates.
> Require process-joule snapshots to bracket each operation while operation
> timing excludes the snapshot syscalls. Never use `powermetrics`
> energy-impact as joules.
>
> Treat the raw Instruments `.trace` package as a first-class sealed artifact,
> never as JSON and never as an archive supplied to the normalizer. Freeze its
> single-link files and directory tree read-only after recording. Before
> normalizing it, require the pinned xctrace export adapter, a real immutable
> operator-reviewed export profile, exact full-Xcode binary/build/template,
> TOC/schema/XPath/column/unit fingerprints, trace-tree identity, probe PID,
> run nonce, argv hash, Metal registry ID, and exact phase-marker joins. Require
> the public static-name signpost shim, one collision-free predeclared signpost
> ID, an exact begin/end pair, and exact command-buffer/encoder/counter joins for
> every raw interval. Seal the trace tree, TOC, exports, profile, canonical JSON,
> and adapter receipt. Pass only the receipt-verified canonical JSON to the
> trusted normalizer; never pass the trace tree or a tar archive as its
> `--xctrace` input.
> Reject guessed aliases, overlap-only attribution, interpolation,
> apportionment, cross-interval events, skipped markers, and logical bytes
> mislabeled as physical traffic. Synthetic fixtures count as zero physical
> evidence. If full Xcode or a real reviewed profile is absent, remain
> fail-closed; do not weaken the gate or provision software while heavy owners
> exist.
> Require the final Doctor aggregate itself to pass
> `doctor_v5_physical_result_authority.py` draft -> SSHSIG sign -> seal ->
> verify. A raw or merely self-hashed aggregate must remain incapable of
> scoring any physical point.
>
> Open physical execution only at Doctor final-ready plus zero heavy owners.
> Follow `APPENDIX_HANDOFF.md` in order: freeze and verify the corpus including
> negative evidence; establish one common release boundary and rollback CAS;
> build fresh receipt-bound release probes; issue and verify ten exact signed
> physical program adapters; collect physical counters; prove stored, compact,
> hashed, and computed TQ device parity including two-pass residual parity; run
> the non-skipping B=1-8 speculative curve against the exact greedy oracle;
> seal the packet; independently verify every hash; require the controller to
> report `10/10`. Do not promote defaults automatically.
>
> Only after that sealed `10/10` packet is independently verified, complete
> part 2. Rebuild and verify the unbound post-120B handoff, preserve all prior
> evidence, take a quiescent-generation CAS, review and bind all four GPT-OSS
> live adapters, and materialize the exact 10-rate x 4-branch matrix. Bind all
> 615 source units and 24,600 isolated jobs to thread calibration, block
> parallelism, ordered overlap, bounded preprocessing reuse, RAM lane packing,
> controlled recoverable swap, disk lifecycle, native I/O/PGO, Metal
> preprocessing, exact-quality receipts, and rollback CAS. Keep each path
> pinned and default-off until exact parity and physical qualification pass,
> then resume detached under the fail-closed guarded supervisor.
>
> For every approved model above 120B, require an immutable exact model
> artifact, logical/stored parameter authority, architecture adapter, source
> corpus/range manifest, tokenizer/chat-template binding, memory admission,
> transport, and lifecycle plan before generating or activating its 10x4
> matrix. Never execute a display-name-only horizon template and never invent
> missing architecture or source facts.
> Raw/self-hashed/`reviewed=true` manifests are not exact-plan authority. Require
> `doctor_v5_higher_tier_authority.py` draft -> SSHSIG sign -> seal -> verify,
> and recompute every logical parameter from a recognized dtype, shape, and
> storage encoding with exact non-overlapping full source-byte coverage before
> generating any 10x4 matrix.
>
> Use parallel agents for implementation and independent adversarial review.
> Regenerate and reseal cheap reports and handoffs only after source contracts
> stop changing. The generic single-device maximization/scaffolding phase is
> closed: do not reopen a broad speed audit, add speculative optimizations, or
> spend release time polishing structure. Fix only a concrete failing gate in
> this ordered execution path. Report structural readiness and physical
> evidence separately, and do not mark the goal complete until current hashes,
> signed receipts, the `10/10` scorecard, exact live adapters, source-bound
> runtime specs, detached resume, and rollback point all verify.
