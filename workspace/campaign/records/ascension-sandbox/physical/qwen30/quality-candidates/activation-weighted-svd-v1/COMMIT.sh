#!/bin/zsh
# Run outside the sandboxed agent profile (gitdir lives under Downloads/hawking).
set -euo pipefail
cd "$(dirname "$0")/../../../../../../../../.."
# Resolve to worktree root from this script's known relative layout.
ROOT="$(cd "$(dirname "$0")/../../../../../../.." && pwd)"
# More reliable: walk up to the worktree that contains lab/operators.
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../../../../../.." && pwd)"
if [[ ! -d "$WT/lab/operators" ]]; then
  WT="/Users/scammermike/.claude-grok/worktrees/q30-activation-repack-20260809-160326"
fi
cd "$WT"
git add .gitignore \
  lab/operators/ascension_dual_gravity_worker.py \
  lab/operators/ascension_qwen30_activation_weighted_svd_repack.py \
  lab/tests/test_activation_weighted_svd_pack_path.py \
  workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates/activation-weighted-svd-v1/QWEN30_ACTIVATION_WEIGHTED_SVD_V1_HANDOFF.json \
  workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates/activation-weighted-svd-v1/QWEN30_ACTIVATION_WEIGHTED_SVD_V1_COMPLETE_GRAVITY_TERMINAL_RECEIPT.json \
  workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates/activation-weighted-svd-v1/QWEN30_ACTIVATION_WEIGHTED_SVD_V1_COMPLETE_GRAVITY_STATUS.json \
  workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates/activation-weighted-svd-v1/QWEN30_ACTIVATION_WEIGHTED_SVD_V1_SOURCE_BINDING_SNAPSHOT.json
git commit -m "$(cat <<'EOF'
Wire activation_weighted_svd into Q30 pack path and seal a surplus-first candidate.

Add the real-capture activation-weighted SVD low-rank family as a first-class
dual-gravity codec (explicit pack/encode; not remapping the pinned v2 schedule),
select L0 expert organs by surplus-over-null under the 1.5 BPW ceiling, and seal
a complete 18867-tensor candidate that hard-links the admitted binary baseline
for unchanged tensors. Capture identity is bound by path+sha256. No promotion
and no live server repoint.
EOF
)"
git status -sb
git log -1 --oneline
git rev-parse HEAD
