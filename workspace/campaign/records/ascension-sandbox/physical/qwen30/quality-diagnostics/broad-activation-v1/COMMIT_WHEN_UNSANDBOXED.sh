#!/usr/bin/env bash
set -euo pipefail
cd /Users/scammermike/.claude-grok/worktrees/q30-broader-capture-20260809-154213
git add \
  crates/hawking-core/examples/ascension_qwen30_current_hcli_layer0_route_capture.rs \
  lab/operators/q30_activation_null_first_report.py \
  lab/operators/q30_broad_activation_after_capture.sh \
  lab/operators/q30_broad_activation_route_capture_prepare.py \
  workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-diagnostics/broad-activation-v1/
git commit -m "$(cat <<'MSG'
Prepare broader Q30 L0 activation capture and report baseline null first

Extend the existing L0 route-capture binary with an additive broad-activation
schema (min 12 probes, diagnostic claim boundary) while leaving the three-probe
HCLI path unchanged. Build a 32-prompt source-tokenized corpus across code,
prose, structured/JSON, multi-turn, and long-context domains, and add null-first
plus post-capture drivers that reuse the existing family probe.

Baseline three-prompt high-hit constant-mean null remains ~0.957 (null trap).
Metal capture from this sandboxed executor is blocked (no GPU device); the exact
serialized capture command is recorded for an unsandboxed gate run.
MSG
)"
git log -1 --oneline
