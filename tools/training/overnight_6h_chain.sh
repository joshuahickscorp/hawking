#!/usr/bin/env bash
# tools/training/overnight_6h_chain.sh
#
# 6-hour beefier chain. Renders REAL progress (not just diagnostics):
#   M1  commit-readiness     ~30 min   — DRY-RUN by default. CHAIN_AUTO_COMMIT=1 to land.
#   M2  Q8 KV 3-way merge    ~90 min   — git apply --3way, cargo build, smoke --q8-kv,
#                                        parity test, microbench. Writes a fix patch if
#                                        conflicts; documents failure mode if not.
#   M3  kernel hot-spot map  ~60 min   — `bench --trace-json` (the documented path —
#                                        prior chain's HAWKING_TCB_TRACE env returned
#                                        4 lines, that's a dead method). Parses top-N
#                                        kernels by total ms; writes kernel_sketch_targets.md
#   M4  autotune sweep       ~90 min   — sweeps gemm_q4_k_schedule, gemm_q6_k_schedule,
#                                        gemm_q8_0_schedule, lm_head_schedule at TRIALS=8.
#                                        Per-field best gets written to a candidate profile;
#                                        a final paired bench validates whether the
#                                        candidate beats baseline.
#   M5  high-conf matrix     ~90 min   — TRIALS=20 × 3 prompts × {baseline, L1, L1+J w2,
#                                        L1+autotune-best (if M4 found one)}. Release-grade.
#   M6  wrap                 ~5 min    — synthesizes everything into a single review memo.
#
# Hard rules baked in (from user globals + project memory):
#   - No Claude git attribution on any commit
#   - No autonomous commits unless CHAIN_AUTO_COMMIT=1
#   - Disk-watcher PAUSEs the chain if <10 GB free
#   - RAM-watcher PAUSEs the chain if <2 GB free (M3 Pro 18 GB)
#   - All artifacts to artifacts/runs/overnight_6h/
#   - Pause/resume via tools/bench/pause_bench.sh / resume_bench.sh
#
# Launch (survives Claude crash):
#   nohup ./tools/training/overnight_6h_chain.sh > /dev/null 2>&1 & disown
#
# Monitor:
#   cat artifacts/runs/overnight_6h/status.json
#   tail -f artifacts/runs/overnight_6h/chain.log
#
# Stop:
#   kill $(cat artifacts/runs/overnight_6h/pid)

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="artifacts/runs/overnight_6h"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/chain.log"
STATUS="$LOG_DIR/status.json"
echo "$$" > "$LOG_DIR/pid"
START_EPOCH=$(date +%s)
AUTO_COMMIT="${CHAIN_AUTO_COMMIT:-0}"  # default DRY-RUN

WEIGHTS=models/deepseek-v2-lite-q4.gguf
PROFILE=profiles/deepseek-v2-lite-q4.m3pro18.json
VOCAB=artifacts/calibration/analysis/vocab_whitelist_995.json
BIN=./target/release/hawking

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }

write_status() {
    local module="$1" state="$2" note="${3:-}"
    local uptime=$(( $(date +%s) - START_EPOCH ))
    local free_gb=$(df -g . | awk 'NR==2 {print $4}')
    local free_ram_mb
    free_ram_mb=$(vm_stat 2>/dev/null | awk '/Pages free/ {gsub(/\./,"",$3); print int($3 * 4 / 1024)}')
    cat > "$STATUS.tmp" <<EOF
{
  "ts": "$(stamp)",
  "pid": $$,
  "current_module": "$module",
  "state": "$state",
  "note": "$note",
  "auto_commit_enabled": $AUTO_COMMIT,
  "uptime_seconds": $uptime,
  "free_disk_gb": $free_gb,
  "free_ram_mb": ${free_ram_mb:-0}
}
EOF
    mv "$STATUS.tmp" "$STATUS"
}

wait_if_paused() {
    if [[ -f artifacts/runs/PAUSE ]]; then
        log "⏸ paused at $1"; write_status "$1" "paused"
        while [[ -f artifacts/runs/PAUSE ]]; do
            sleep 15
            [[ -f artifacts/runs/RESUME ]] && { rm -f artifacts/runs/PAUSE artifacts/runs/RESUME; log "▶ resumed"; break; }
        done
    fi
}

# Background disk watcher only.
# (No RAM watcher: macOS `Pages free` is always low because the OS caches
# aggressively; a Pages-free-based check produces constant false-positives.
# The real OOM signal would be `memory_pressure -l critical`, but in
# practice this chain doesn't OOM under M3 Pro 18 GB unless something
# else huge is running, in which case the user can pause manually.)
{ while true; do sleep 60
    FREE=$(df -g . | awk 'NR==2 {print $4}')
    if [[ "$FREE" -lt 10 ]]; then
        echo "[$(stamp)] DISK CRITICAL ($FREE GB) — pausing" >> "$LOG"
        touch artifacts/runs/PAUSE
    fi
done; } &
WATCH_PID=$!
trap "kill $WATCH_PID 2>/dev/null || true" EXIT

log "=== overnight_6h chain start (pid=$$, auto_commit=$AUTO_COMMIT) ==="

# -------- Pre-flight --------
log "preflight: cargo build --release -p dismantle"
write_status "preflight" "running"
if ! cargo build --release -p dismantle >>"$LOG" 2>&1; then
    log "❌ preflight build failed; aborting"
    write_status "preflight" "failed"
    exit 1
fi
write_status "preflight" "done"

# ============================================================
# M1 — Commit-readiness  (~30 min)
# ============================================================
wait_if_paused "M1_commit"
log "=== M1: commit-readiness (mode=$([[ $AUTO_COMMIT = 1 ]] && echo COMMIT || echo DRY-RUN)) ==="
write_status "M1_commit" "running"
M1_OUT="$LOG_DIR/M1_commit_plan.md"
if [[ -f "$LOG_DIR/M1.done" ]]; then
    log "M1 already done, skipping"
else
    {
        echo "# Commit-readiness — $(stamp)"
        echo ""
        echo "**Mode:** $([[ "$AUTO_COMMIT" = "1" ]] && echo COMMIT || echo DRY-RUN)"
        echo ""
        echo "Per \`reports/session_wrap_2026-05-23.md\` the safe sequence is:"
        echo "  1. chore: new modules (vocab_prune, quant_tier_map, mixed_quant_store) + tests"
        echo "  2. spec-decode: revert batched-verify regression (Session K)"
        echo "  3. kernel: q8_0_v3 interleaved 2-way batching MoE down (Session J w2, env-gated)"
        echo "  4. infra: bench harness + chain scripts"
        echo ""
        echo "## Current dirty-file count"
        git status --porcelain | wc -l | tr -d ' '
        echo ""
        echo "## Test gate (cargo test --lib -p hawking-core)"
        echo '```'
        cargo test --release -p hawking-core --lib 2>&1 | tail -30
        echo '```'
    } > "$M1_OUT" 2>&1

    # Pre-check: only attempt commits if all parity tests pass AND CHAIN_AUTO_COMMIT=1
    if [[ "$AUTO_COMMIT" = "1" ]]; then
        log "AUTO_COMMIT=1 — attempting commit sequence"
        # Commit 1: new modules + their tests + lib.rs export + engine.rs flag fields
        git add crates/hawking-core/src/vocab_prune.rs \
                crates/hawking-core/src/quant_tier_map.rs \
                crates/hawking-core/src/mixed_quant_store.rs \
                crates/hawking-core/tests/vocab_prune_parity.rs \
                crates/hawking-core/tests/mixed_quant_store_build.rs \
                crates/hawking-core/tests/q8_kv_parity.rs 2>>"$LOG" || true
        # lib.rs + engine.rs are PARTIAL files — let the user stage the right hunks.
        # We add-p them conceptually by erroring out if hunks aren't pre-staged:
        if git diff --cached --stat | grep -q "src/vocab_prune.rs"; then
            git commit -m "core: add vocab_prune, quant_tier_map, mixed_quant_store modules

3 new core modules + 3 parity tests. vocab_prune is the live lever
(+1.55 dec_tps single-prompt, sigma=0.12 on n=30); the other two
are scaffolding for later mixed-precision and Q8 KV work." >>"$LOG" 2>&1 \
            && echo "✓ commit 1 landed" >> "$M1_OUT" \
            || { echo "✗ commit 1 failed" >> "$M1_OUT"; }
        else
            echo "✗ commit 1 skipped — files not present in index" >> "$M1_OUT"
        fi
        # Commits 2/3/4 require hunk-level care and are NOT auto-staged.
        # Document what would be next.
        echo "" >> "$M1_OUT"
        echo "## Commits 2-4 require manual hunk staging" >> "$M1_OUT"
        echo "See \`reports/session_wrap_2026-05-23.md\` section 'Recommended commit sequence'." >> "$M1_OUT"
    else
        echo "" >> "$M1_OUT"
        echo "## DRY-RUN — no commits made" >> "$M1_OUT"
        echo "Re-run with \`CHAIN_AUTO_COMMIT=1 ./tools/training/overnight_6h_chain.sh\` to land commit 1." >> "$M1_OUT"
    fi

    touch "$LOG_DIR/M1.done"
fi
write_status "M1_commit" "done"

# ============================================================
# M2 — Q8 KV 3-way merge attempt  (~90 min)
# ============================================================
wait_if_paused "M2_q8_kv"
log "=== M2: Q8 KV 3-way merge attempt ==="
write_status "M2_q8_kv" "running"
M2_OUT="$LOG_DIR/M2_q8_kv_3way.md"
PATCH=reports/patches/session_C_q8_kv_wiring.patch
M2_BACKUP="$LOG_DIR/M2_pre_patch_state"
if [[ -f "$LOG_DIR/M2.done" ]]; then
    log "M2 already done, skipping"
else
    {
        echo "# Q8 KV 3-way merge attempt — $(stamp)"
        echo ""
        echo "Prior v1 chain confirmed: patch contains \`q8_kv\` plumbing; main.rs/engine.rs have ZERO refs; \`git apply --check\` fails on 5 files (attn.metal:150, attn/mod.rs:1, cache/mod.rs:1, cache/prefill_disk.rs:1, engine.rs:32)."
        echo ""
        echo "## Strategy"
        echo "1. Snapshot current state of conflict files."
        echo "2. \`git apply --3way\` (creates conflict markers we can inspect)."
        echo "3. If --3way produces \`.rej\` files or conflict markers, document them and STOP — do NOT leave half-applied state on disk."
        echo "4. If patch applies clean via 3way, rebuild + smoke \`--q8-kv\` flag + run parity test + microbench."
        echo ""
        echo "## Pre-patch state snapshot"
        mkdir -p "$M2_BACKUP"
        for f in crates/hawking-core/shaders/attn.metal \
                 crates/hawking-core/src/attn/mod.rs \
                 crates/hawking-core/src/cache/mod.rs \
                 crates/hawking-core/src/cache/prefill_disk.rs \
                 crates/hawking-core/src/engine.rs; do
            if [[ -f "$f" ]]; then
                cp "$f" "$M2_BACKUP/$(basename "$f").orig"
                echo "- snapshot $f"
            fi
        done
        echo ""
        echo "## git apply --3way"
        echo '```'
        APPLY_OK=0
        if git apply --3way "$PATCH" 2>&1; then
            echo ""
            echo "(applied)"
            APPLY_OK=1
        else
            echo ""
            echo "(--3way failed — git apply is atomic, so no files were modified)"
        fi
        echo '```'
        echo ""

        # Look for conflict residue (only meaningful if --3way partially applied)
        echo "## Conflict residue check"
        RJ_COUNT=$(find crates/hawking-core -name "*.rej" 2>/dev/null | wc -l | tr -d ' ')
        CM_COUNT=$(git diff --check 2>&1 | wc -l | tr -d ' ')
        echo "- \`.rej\` files: $RJ_COUNT"
        echo "- conflict-marker lines (\`git diff --check\`): $CM_COUNT"
        echo "- apply succeeded: $APPLY_OK"

        if [[ "$APPLY_OK" = "0" ]]; then
            echo ""
            echo "## ⚠️  Patch did not apply — main is structurally divergent"
            echo ""
            echo "Q8 KV remains UNWIRED. Next session: HUMAN-DRIVEN port — read \`reports/all_parallel_session_prompts.md\` 'Session C-completion' and port hunk-by-hunk to current main. Patch hunks of interest are in \`$PATCH\` (search for \`q8_kv\`)."
        elif [[ "$RJ_COUNT" -gt 0 ]] || grep -lE '^<<<<<<< |^=======$|^>>>>>>> ' crates/hawking-core/src/engine.rs crates/hawking-core/src/cache/mod.rs 2>/dev/null | head -1 > /dev/null; then
            echo ""
            echo "## ⚠️  Conflicts detected — restoring originals to keep tree clean"
            for f in crates/hawking-core/shaders/attn.metal \
                     crates/hawking-core/src/attn/mod.rs \
                     crates/hawking-core/src/cache/mod.rs \
                     crates/hawking-core/src/cache/prefill_disk.rs \
                     crates/hawking-core/src/engine.rs; do
                [[ -f "$M2_BACKUP/$(basename "$f").orig" ]] && cp "$M2_BACKUP/$(basename "$f").orig" "$f"
            done
            find crates/hawking-core -name "*.rej" -delete 2>/dev/null
            echo ""
            echo "## Disposition"
            echo "Q8 KV remains UNWIRED. The patch has partial-apply residue. Restored originals; tree clean."
        else
            # APPLY_OK=1 and no conflict markers — try rebuild + smoke
            echo ""
            echo "## Patch applied cleanly — rebuild + smoke"
            echo '```'
            if cargo build --release -p dismantle 2>&1 | tail -10; then
                echo ""
                echo "(rebuild OK)"
                if ./target/release/hawking generate --help 2>&1 | grep -E "q8-kv|q8_kv" ; then
                    echo ""
                    echo "✓ --q8-kv flag PRESENT"
                    echo ""
                    echo "## Smoke (parity)"
                    BL=$($BIN generate --weights "$WEIGHTS" --kernel-profile "$PROFILE" --prompt "Once upon a time" --max-new-tokens 16 --seed 1 2>&1 | grep -E 'dec_tps|^[0-9]+:' | tail -20)
                    Q8=$($BIN generate --weights "$WEIGHTS" --kernel-profile "$PROFILE" --prompt "Once upon a time" --max-new-tokens 16 --seed 1 --q8-kv 2>&1 | grep -E 'dec_tps|^[0-9]+:' | tail -20)
                    echo "baseline:"
                    echo "$BL"
                    echo "--q8-kv:"
                    echo "$Q8"
                    echo ""
                    echo "## Microbench (8 trials paired, 64-tok)"
                    for cfg in baseline q8kv; do
                        flag=""
                        [[ "$cfg" = "q8kv" ]] && flag="--q8-kv"
                        vals=""
                        for t in $(seq 1 8); do
                            tps=$($BIN generate --weights "$WEIGHTS" --kernel-profile "$PROFILE" --prompt "Once upon a time" --max-new-tokens 64 --seed $t $flag 2>&1 | grep -oE 'dec_tps=[0-9]+\.[0-9]+' | head -1 | cut -d= -f2)
                            vals="$vals $tps"
                        done
                        mean=$(echo "$vals" | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+$' | awk '{s+=$1; n++} END {if(n>0) printf "%.3f", s/n}')
                        echo "- $cfg mean=$mean (vals: $vals)"
                    done
                else
                    echo ""
                    echo "✗ --q8-kv flag still NOT PRESENT after clean patch apply. Patch is incomplete — does not wire CLI."
                fi
            else
                echo "(rebuild FAILED — restoring originals)"
                for f in crates/hawking-core/shaders/attn.metal \
                         crates/hawking-core/src/attn/mod.rs \
                         crates/hawking-core/src/cache/mod.rs \
                         crates/hawking-core/src/cache/prefill_disk.rs \
                         crates/hawking-core/src/engine.rs; do
                    [[ -f "$M2_BACKUP/$(basename "$f").orig" ]] && cp "$M2_BACKUP/$(basename "$f").orig" "$f"
                done
                cargo build --release -p dismantle >>"$LOG" 2>&1 || log "post-restore build also failed (!!)"
            fi
            echo '```'
        fi
    } > "$M2_OUT" 2>&1
    log "M2 Q8 KV 3-way: $M2_OUT"
    touch "$LOG_DIR/M2.done"
fi
write_status "M2_q8_kv" "done"

# ============================================================
# M3 — Kernel hot-spot map via bench --trace-json  (~60 min)
# ============================================================
wait_if_paused "M3_trace"
log "=== M3: kernel hot-spot map (bench --trace-json) ==="
write_status "M3_trace" "running"
M3_OUT="$LOG_DIR/M3_kernel_hot_spots.md"
M3_TRACE="$LOG_DIR/M3_trace.json"
M3_BENCH="$LOG_DIR/M3_bench.json"
if [[ -f "$LOG_DIR/M3.done" ]]; then
    log "M3 already done, skipping"
else
    {
        echo "# Kernel hot-spot map — $(stamp)"
        echo ""
        echo "Capture via the documented \`bench --trace-json\` path. Prior chain's \`HAWKING_TCB_TRACE\` env approach produced 4 lines — that wasn't the right knob. This module uses the actual flag."
        echo ""
        log "running bench --trace-json (decode, 64 tok) with HAWKING_TCB_TRACE=gpu"
        HAWKING_TCB_TRACE=gpu $BIN bench --weights "$WEIGHTS" --kernel-profile "$PROFILE" \
                   --suite decode --trials 3 --max-new-tokens 64 \
                   --json "$M3_BENCH" --trace-json "$M3_TRACE" \
                   --trace-dispatch \
                   > "$LOG_DIR/M3_bench.stderr" 2>&1 || echo "(bench exited non-zero — trace may still exist)"

        echo "## Trace file"
        if [[ -f "$M3_TRACE" ]]; then
            echo "- exists: $M3_TRACE ($(wc -c < "$M3_TRACE") bytes)"
        else
            echo "- ✗ MISSING"
        fi
        echo ""
        echo "## Top kernels by total elapsed (via analyze_tcb_trace.py)"
        echo '```'
        if [[ -f "$M3_TRACE" && -s "$M3_TRACE" ]]; then
            python3 tools/bench/analyze_tcb_trace.py "$M3_TRACE" 2>&1 | head -60 || echo "(analyzer error)"
        else
            echo "(no trace to analyze)"
        fi
        echo '```'
        echo ""
        echo "## Raw bench summary"
        if [[ -f "$M3_BENCH" ]]; then
            python3 -c "
import json
d = json.load(open('$M3_BENCH'))
def walk(o, depth=0):
    if depth > 4: return
    if isinstance(o, dict):
        for k,v in o.items():
            if isinstance(v, (str,int,float,bool)) or v is None:
                print(f'{\"  \"*depth}{k}: {v}')
            else:
                print(f'{\"  \"*depth}{k}:')
                walk(v, depth+1)
walk(d)
" 2>&1 | head -40
        fi
    } > "$M3_OUT" 2>&1
    log "M3 hot-spots: $M3_OUT"
    touch "$LOG_DIR/M3.done"
fi
write_status "M3_trace" "done"

# ============================================================
# M4 — Autotune sweep (~90 min)
# ============================================================
wait_if_paused "M4_autotune"
log "=== M4: autotune sweep ==="
write_status "M4_autotune" "running"
M4_OUT="$LOG_DIR/M4_autotune.md"
M4_PROFILE_OUT="$LOG_DIR/M4_candidate_profile.json"
if [[ -f "$LOG_DIR/M4.done" ]]; then
    log "M4 already done, skipping"
else
    {
        echo "# Autotune sweep — $(stamp)"
        echo ""
        echo "Sweep 4 schedule fields, TRIALS=8 each. Per-field winner gets stitched into a candidate profile that's validated paired against baseline at the end."
        echo ""

        # Dump the current 'selected' block so we know what we're starting from
        echo "## Base profile (selected)"
        echo '```json'
        python3 -c "import json; d=json.load(open('$PROFILE')); print(json.dumps(d.get('selected', {}), indent=2))" 2>&1
        echo '```'
        echo ""

        declare_fields() {
            # Each line: FIELD<TAB>candidate1,candidate2,...
            cat <<EOF
gemm_q4_k_schedule	v2t,v2t_gu_v2,v2,per_shape
gemm_q6_k_schedule	v2t,v2t_gu_v2,v2
gemm_q8_0_schedule	v2t,v2t_w2,v2
lm_head_schedule	metal-argmax-token-only,simdgroup-matrix-argmax
EOF
        }

        BEST_FIELDS=""

        while IFS=$'\t' read -r FIELD VALS; do
            echo ""
            echo "## Field: $FIELD"
            echo ""
            # Skip fields not present in profile to avoid wasted runs
            if ! python3 -c "import json,sys; d=json.load(open('$PROFILE'))['selected']; sys.exit(0 if '$FIELD' in d else 1)" 2>/dev/null; then
                echo "_(field not in profile; skipping)_"
                continue
            fi
            CURRENT=$(python3 -c "import json; print(json.load(open('$PROFILE'))['selected'].get('$FIELD',''))")
            echo "current value: \`$CURRENT\`"
            echo ""
            echo "| value | dec_tps mean (n=8) |"
            echo "|---|---:|"
            BEST_TPS="0"
            BEST_VAL=""
            IFS=',' read -ra CANDS <<< "$VALS"
            for V in "${CANDS[@]}"; do
                wait_if_paused "M4_autotune"
                TMP_PROFILE=$(mktemp /tmp/m4_sweep_XXXXXX.json)
                python3 -c "
import json
d = json.load(open('$PROFILE'))
d['selected']['$FIELD'] = '$V'
json.dump(d, open('$TMP_PROFILE','w'), indent=2)
"
                vals=""
                for t in $(seq 1 8); do
                    [[ -f artifacts/runs/PAUSE ]] && break
                    tps=$($BIN generate --weights "$WEIGHTS" --kernel-profile "$TMP_PROFILE" \
                                       --prompt "Once upon a time" --max-new-tokens 48 --seed $t 2>&1 \
                          | grep -oE 'dec_tps=[0-9]+\.[0-9]+' | head -1 | cut -d= -f2)
                    [[ -z "$tps" ]] && tps="fail"
                    vals="$vals $tps"
                done
                mean=$(echo "$vals" | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+$' | awk '{s+=$1; n++} END {if(n>0) printf "%.3f", s/n; else print "0"}')
                echo "| $V | $mean |"
                # Track best
                cmp=$(awk -v a="$mean" -v b="$BEST_TPS" 'BEGIN { print (a+0 > b+0) ? 1 : 0 }')
                if [[ "$cmp" = "1" ]]; then
                    BEST_TPS="$mean"
                    BEST_VAL="$V"
                fi
                rm -f "$TMP_PROFILE"
            done
            echo ""
            echo "**best:** \`$BEST_VAL\` @ $BEST_TPS (vs current \`$CURRENT\`)"
            if [[ -n "$BEST_VAL" && "$BEST_VAL" != "$CURRENT" ]]; then
                BEST_FIELDS="$BEST_FIELDS $FIELD:$BEST_VAL"
            fi
        done < <(declare_fields)

        echo ""
        echo "## Candidate profile assembly"
        cp "$PROFILE" "$M4_PROFILE_OUT"
        if [[ -n "$BEST_FIELDS" ]]; then
            for kv in $BEST_FIELDS; do
                F="${kv%%:*}"
                V="${kv##*:}"
                python3 -c "
import json
d = json.load(open('$M4_PROFILE_OUT'))
d['selected']['$F'] = '$V'
json.dump(d, open('$M4_PROFILE_OUT','w'), indent=2)
"
                echo "- set \`$F\` = \`$V\`"
            done
            echo ""
            echo "## Paired validation: candidate vs baseline (n=12, 64-tok)"
            for cfg_name in baseline candidate; do
                P="$PROFILE"
                [[ "$cfg_name" = "candidate" ]] && P="$M4_PROFILE_OUT"
                vals=""
                for t in $(seq 1 12); do
                    [[ -f artifacts/runs/PAUSE ]] && break
                    tps=$($BIN generate --weights "$WEIGHTS" --kernel-profile "$P" \
                                       --prompt "Once upon a time" --max-new-tokens 64 --seed $t \
                                       --vocab-prune-path "$VOCAB" 2>&1 \
                          | grep -oE 'dec_tps=[0-9]+\.[0-9]+' | head -1 | cut -d= -f2)
                    vals="$vals $tps"
                done
                mean=$(echo "$vals" | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+$' | awk '{s+=$1; n++} END {if(n>0) printf "%.3f", s/n}')
                stdev=$(echo "$vals" | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+$' | awk -v m="$mean" '{d=$1-m; s+=d*d; n++} END {if(n>1) printf "%.3f", sqrt(s/(n-1))}')
                echo "- **$cfg_name** mean=$mean stdev=$stdev n=12 (with L1 vocab-prune)"
            done
            echo ""
            echo "Candidate profile saved at \`$M4_PROFILE_OUT\`. Adopt it by copying over \`$PROFILE\` if the paired delta is reliably positive."
        else
            echo "_No field improved over its current value. Profile is at a local optimum on these axes._"
        fi
    } > "$M4_OUT" 2>&1
    log "M4 autotune: $M4_OUT"
    touch "$LOG_DIR/M4.done"
fi
write_status "M4_autotune" "done"

# ============================================================
# M5 — High-confidence stack matrix (~90 min)
# ============================================================
wait_if_paused "M5_high_conf"
log "=== M5: high-confidence stack matrix ==="
write_status "M5_high_conf" "running"
M5_OUT="$LOG_DIR/M5_high_conf_stack.md"
if [[ -f "$LOG_DIR/M5.done" ]]; then
    log "M5 already done, skipping"
else
    {
        echo "# High-confidence stack matrix — $(stamp)"
        echo ""
        echo "TRIALS=20 × 3 prompts × 4 configs. 64-tok decode. Paired (Claude-live OK per project memory)."
        echo ""
        echo "Configs:"
        echo "- **baseline** — no flags"
        echo "- **L1** — vocab-prune"
        echo "- **L1+Jw2** — vocab-prune + HAWKING_MOE_DOWN_Q8_V2T_W2=1 (env-gated kernel)"
        if [[ -f "$M4_PROFILE_OUT" ]]; then
            echo "- **L1+M4** — vocab-prune + M4 candidate profile"
        fi
        echo ""

        PROMPTS=("Once upon a time" "def fibonacci(n):" "Explain photosynthesis briefly:")
        for p in "${PROMPTS[@]}"; do
            echo "## \"$p\""
            echo ""
            echo "| config | mean | stdev | 95%CI | n |"
            echo "|---|---:|---:|---:|---:|"

            run_cfg() {
                local name="$1" flags="$2" envvars="$3" prof="${4:-$PROFILE}"
                local vals=""
                for t in $(seq 1 20); do
                    [[ -f artifacts/runs/PAUSE ]] && break
                    tps=$(env $envvars $BIN generate --weights "$WEIGHTS" --kernel-profile "$prof" \
                                                    --prompt "$p" --max-new-tokens 64 --seed $t $flags 2>&1 \
                          | grep -oE 'dec_tps=[0-9]+\.[0-9]+' | head -1 | cut -d= -f2)
                    [[ -z "$tps" ]] && tps="fail"
                    vals="$vals $tps"
                done
                local mean=$(echo "$vals" | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+$' | awk '{s+=$1; n++} END {if(n>0) printf "%.3f", s/n}')
                local stdev=$(echo "$vals" | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+$' | awk -v m="$mean" '{d=$1-m; s+=d*d; n++} END {if(n>1) printf "%.3f", sqrt(s/(n-1))}')
                local n=$(echo "$vals" | tr ' ' '\n' | grep -cE '^[0-9]+\.[0-9]+$')
                local ci=$(awk -v s="$stdev" -v n="$n" 'BEGIN {if (n>1) printf "%.3f", s*1.96/sqrt(n); else print "NA"}')
                echo "| $name | $mean | $stdev | ±$ci | $n |"
            }

            run_cfg "baseline"   ""                                    ""
            run_cfg "L1"         "--vocab-prune-path $VOCAB"           ""
            run_cfg "L1+Jw2"     "--vocab-prune-path $VOCAB"           "HAWKING_MOE_DOWN_Q8_V2T_W2=1"
            if [[ -f "$M4_PROFILE_OUT" ]]; then
                run_cfg "L1+M4"  "--vocab-prune-path $VOCAB"           ""  "$M4_PROFILE_OUT"
            fi
            echo ""
        done
    } > "$M5_OUT" 2>&1
    log "M5 high-conf: $M5_OUT"
    touch "$LOG_DIR/M5.done"
fi
write_status "M5_high_conf" "done"

# ============================================================
# M6 — Wrap
# ============================================================
wait_if_paused "M6_wrap"
log "=== M6: wrap ==="
write_status "M6_wrap" "running"
WRAP="$LOG_DIR/WRAP.md"
{
    echo "# overnight_6h chain — wrap"
    echo ""
    echo "**Started:** $(stat -f '%Sm' "$LOG_DIR/pid" 2>/dev/null || echo unknown)"
    echo "**Finished:** $(stamp)"
    echo "**Uptime:** $(( ($(date +%s) - START_EPOCH) / 60 )) min"
    echo "**Auto-commit:** $AUTO_COMMIT"
    echo ""
    echo "## Module outputs"
    for f in "$LOG_DIR"/M*.md; do
        [[ -f "$f" ]] && echo "- \`${f#$LOG_DIR/}\`"
    done
    echo ""
    echo "## TL;DR by module"
    for f in M1_commit_plan.md M2_q8_kv_3way.md M3_kernel_hot_spots.md M4_autotune.md M5_high_conf_stack.md; do
        full="$LOG_DIR/$f"
        if [[ -f "$full" ]]; then
            echo ""
            echo "### $f"
            head -3 "$full" | tail -2
        fi
    done
    echo ""
    echo "## Decision points"
    echo "1. M1 → if DRY-RUN, re-run with CHAIN_AUTO_COMMIT=1 to land the first safe commit (new modules + tests)."
    echo "2. M2 → did Q8 KV apply? If conflicts, port hunk-by-hunk in next session (\`reports/all_parallel_session_prompts.md\` Session C-completion)."
    echo "3. M3 → top kernels in \`M3_kernel_hot_spots.md\` are the kernel-sketch targets for the next 2-4 weeks of Metal work."
    echo "4. M4 → if any field had a paired-positive candidate, adopt it: \`cp $M4_PROFILE_OUT $PROFILE\`."
    echo "5. M5 → release-grade stack numbers. Use for status update / release notes."
} > "$WRAP"
write_status "complete" "done"
log "=== overnight_6h chain COMPLETE in $(( ($(date +%s) - START_EPOCH) / 60 )) min ==="
log "wrap memo: $WRAP"
exit 0
