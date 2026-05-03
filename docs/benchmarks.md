# Benchmarks

(Auto-generated from `dismantle bench --suite all --json` output.
Until Phase 1 lands, this file is intentionally empty.)

## Methodology

See [m3_audit.md](./m3_audit.md) for hardware and software lockfile.

Every published number references:

- a dismantle commit hash
- a llama.cpp commit hash (for `wax-vs-llama-cpp` rows)
- the exact `dismantle bench` invocation that produced it
- first-trial and median-of-three timing

## Suites

| Suite | What it measures | First lands in |
|---|---|---|
| `decode-tps` | tokens/sec at batch=1, 256-token completion | Phase 0 |
| `prefill-tps` | tokens/sec on a 1024-token prompt | Phase 1 |
| `ttft` | time-to-first-token | Phase 1 |
| `throughput-vs-batch` | aggregate tok/s at batch sizes 1, 2, 4, 8, 16 | Phase 4 |
| `bandwidth-utilization` | measured GB/s ÷ theoretical peak | Phase 1 |
| `wax-vs-llama-cpp` | side-by-side, per-wedge ratio table | Phase 1, expanded each phase |
