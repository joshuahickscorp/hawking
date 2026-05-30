#!/usr/bin/env bash
# 0.1 — xctrace EXPORT half (the Stage-2 unblock). mst_capture.sh records a
# "Metal System Trace" .trace bundle; this exports its Metal GPU interval table
# to XML for mst_analyze.py to turn into a per-kernel GPU-occupancy / achieved-
# bandwidth breakdown.
#
# Why: the homemade TCB trace is split-CB-distorted (the §1 gate rejects it),
# so kernel levers have been guesses (the `_4r` dead lever). A real MST gives
# busy-time per kernel so the next lever targets the actual stall.
#
# Usage:
#   tools/bench/mst_export.sh <trace-bundle>            # discover + export
#   tools/bench/mst_export.sh <trace-bundle> --toc      # just list schemas
#   tools/bench/mst_export.sh <trace-bundle> --schema NAME   # export one table
#   OUT_DIR=foo tools/bench/mst_export.sh <trace-bundle>
#
# Then: tools/bench/mst_analyze.py <OUT_DIR>/<schema>.xml --inspect
#       tools/bench/mst_analyze.py <OUT_DIR>/<schema>.xml --decode-wall-ms N
set -uo pipefail

TRACE="${1:-}"
[[ -n "$TRACE" && -e "$TRACE" ]] || { echo "usage: $0 <trace-bundle> [--toc|--schema NAME]" >&2; exit 64; }
shift || true
command -v xcrun >/dev/null 2>&1 || { echo "error: xcrun not found (need Xcode CLT)." >&2; exit 1; }

OUT_DIR="${OUT_DIR:-${TRACE%.trace}_export}"
mkdir -p "$OUT_DIR"
TOC="$OUT_DIR/toc.xml"

# 1. Always dump the table of contents so the real schema names are visible.
xcrun xctrace export --input "$TRACE" --toc --output "$TOC" 2>/dev/null \
  || { echo "error: xctrace export --toc failed on $TRACE" >&2; exit 1; }

# Schema names that carry Metal GPU work intervals vary by Instruments version;
# surface every candidate rather than hardcoding one.
mapfile -t SCHEMAS < <(grep -oE 'schema="[^"]+"' "$TOC" | sed -E 's/schema="([^"]+)"/\1/' | sort -u)
echo "=== schemas in $TRACE ==="
printf '  %s\n' "${SCHEMAS[@]}"
GPU_CANDIDATES=$(printf '%s\n' "${SCHEMAS[@]}" | grep -iE 'gpu|metal|shader|kernel|channel|interval' || true)
echo "=== GPU-interval candidates (export these) ==="
printf '  %s\n' $GPU_CANDIDATES

MODE="all"; ONE=""
while [[ $# -gt 0 ]]; do case "$1" in
  --toc) MODE="toc";;
  --schema) ONE="$2"; shift;;
  *) echo "unknown arg: $1" >&2; exit 64;;
esac; shift; done

[[ "$MODE" == "toc" ]] && { echo "toc written: $TOC"; exit 0; }

export_one() {
  local schema="$1" out="$OUT_DIR/${1//[^a-zA-Z0-9_-]/_}.xml"
  xcrun xctrace export --input "$TRACE" \
    --xpath "/trace-toc/run[1]/data/table[@schema=\"$schema\"]" \
    --output "$out" 2>/dev/null \
    && echo "  exported $schema -> $out ($(wc -l < "$out") lines)" \
    || echo "  FAILED $schema"
}

echo "=== exporting ==="
if [[ -n "$ONE" ]]; then export_one "$ONE"; else
  for s in $GPU_CANDIDATES; do export_one "$s"; done
  [[ -z "$GPU_CANDIDATES" ]] && echo "  (no GPU-interval schema matched; inspect $TOC and pass --schema NAME)"
fi
echo "next: tools/bench/mst_analyze.py $OUT_DIR/<schema>.xml --inspect"
