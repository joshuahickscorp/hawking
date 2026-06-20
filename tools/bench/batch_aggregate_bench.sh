#!/usr/bin/env bash
# Aggregate-tps for continuous-batching multi-seq decode (B=1/4/8).
# The SPEEDUP RATIO is contamination-robust (valid with Claude open); the
# ABSOLUTE tps needs a clean room. Auto-detected by clean_bench_queue.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."
exec cargo test --release -p hawking-core --test multiseq_aggregate_bench -- --ignored --nocapture
