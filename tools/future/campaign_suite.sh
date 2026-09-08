#!/bin/bash
# THE canonical regression entrance. [S010 25]
#
# A whole session was certified against the wrong surface. `python3` on this host
# has pytest and NO mlx, so five MLX-dependent modules never ran and every
# regression check reported green while the campaign suite was red. The two
# interpreters are not interchangeable and nothing said so:
#
#   python3                      pytest yes, mlx NO,  numpy yes
#   /usr/local/bin/python3.12    pytest yes, mlx yes, numpy yes   <- the campaign
#
# LAW: A TEST ENVIRONMENT THAT CANNOT IMPORT THE CAMPAIGN'S REAL DEPENDENCIES
# CANNOT CERTIFY THE CAMPAIGN.
#
# So this refuses rather than skipping. A suite that silently omits what it
# cannot import produces exactly the false green this exists to prevent.
#
#   tools/future/campaign_suite.sh                 the whole hcli suite
#   tools/future/campaign_suite.sh hcli/tests/x.py  a subset, same interpreter
#   BASELINE=<sha> tools/future/campaign_suite.sh   run it, then report the DELTA
set -uo pipefail

PY=/usr/local/bin/python3.12
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

for mod in pytest mlx.core numpy; do
  "$PY" -c "import $mod" 2>/dev/null || {
    echo "REFUSING: $PY cannot import $mod." >&2
    echo "A test environment that cannot import the campaign's real dependencies cannot" >&2
    echo "certify the campaign. Skipping those modules is how 68 failures hid behind a" >&2
    echo "green narrow run for an entire session." >&2
    exit 2
  }
done

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(hcli/)

run_and_count() {  # $1 = output file
  "$PY" -m pytest "${TARGETS[@]}" -q --tb=no > "$1" 2>&1
  grep -c '^FAILED' "$1" || true
}

OUT="$(mktemp -t campaign-suite)"
echo "campaign interpreter: $PY"
echo "targets: ${TARGETS[*]}"
HEAD_N=$(run_and_count "$OUT")
tail -1 "$OUT"
echo "campaign suite: ${HEAD_N} failing"

if [ -n "${BASELINE:-}" ]; then
  WT="$REPO/.worktrees/suite-baseline"
  git worktree add -q --detach "$WT" "$BASELINE" 2>/dev/null || git -C "$WT" checkout -q "$BASELINE"
  BOUT="$(mktemp -t campaign-suite-base)"
  ( cd "$WT" && "$PY" -m pytest "${TARGETS[@]}" -q --tb=no > "$BOUT" 2>&1 )
  BASE_N=$(grep -c '^FAILED' "$BOUT" || true)
  echo "baseline $BASELINE: ${BASE_N} failing"
  echo "baseline delta: $(( HEAD_N - BASE_N ))"
  echo "--- introduced by HEAD ---"
  comm -13 <(grep '^FAILED' "$BOUT" | sed 's/^FAILED //' | sort) \
           <(grep '^FAILED' "$OUT"  | sed 's/^FAILED //' | sort)
  rm -f "$BOUT"
fi
rm -f "$OUT"
[ "$HEAD_N" -eq 0 ] || exit 1
