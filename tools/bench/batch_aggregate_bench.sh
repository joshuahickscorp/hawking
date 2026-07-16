#!/usr/bin/env bash
# Aggregate-tps for continuous-batching multi-seq decode (B=1/4/8).
# The SPEEDUP RATIO is contamination-robust (valid with the agent open); the
# ABSOLUTE tps needs a clean room. Run through `tools/ops.py bench run batch`.
set -uo pipefail
cd "$(dirname "$0")/../.."
exec cargo test --release -p hawking-core --test multiseq_aggregate_bench -- --ignored --nocapture
