#!/usr/bin/env bash
# G1a checkpoint watcher — polls every 5 min, evals PPL at step 25 and final.
# Run: nohup bash tools/training/g1a_watcher.sh > artifacts/lowbit_rwkv7/g1a_watcher.log 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="$ROOT/artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8"
HF="$ROOT/models/rwkv7-g1-04-hf"
PPL_OUT="$ROOT/artifacts/lowbit_rwkv7/ppl"
VENV="$ROOT/.venv-rwkv"
LOG="$ROOT/artifacts/lowbit_rwkv7/g1a_watcher.log"

# Promote-ladder thresholds (1.2× / 1.35× / 1.5× × baseline 11.30)
GATE_G1B=13.56
GATE_SILVER=15.26
GATE_TUNE=16.95

mkdir -p "$PPL_OUT"

eval_ckpt() {
    local ckpt="$1"  # path to state_dict.pt
    local tag="$2"   # short tag like g1a_step25 or g1a_final
    echo "[watcher] evaluating PPL for $tag at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    if [ -f "$VENV/bin/activate" ]; then
        source "$VENV/bin/activate"
    fi
    python3 "$ROOT/tools/training/rwkv7_eval_ppl.py" \
        --model "$ckpt" \
        --hf-dir "$HF" \
        --corpus wikitext2 \
        --tokens 4096 \
        --stride 5000 \
        --device cpu \
        --run-id "$tag" \
        2>&1 | tee -a "$LOG"
    # Print last PPL line for quick gate check
    local ppl
    ppl=$(python3 -c "
import json, sys
for line in open('$PPL_OUT/${tag}.jsonl' if False else '/dev/stdin'):
    try:
        d = json.loads(line)
        if 'ppl' in d: print(d['ppl'])
    except: pass
" 2>/dev/null || echo "?")
    echo "[watcher] $tag PPL=${ppl}"
    echo "[watcher] Gates: G1b≤${GATE_G1B} Silver≤${GATE_SILVER} Tune≤${GATE_TUNE}"
}

announce_ppl() {
    local ppl_file="$1"
    local tag="$2"
    local ppl
    ppl=$(python3 -c "
import json
for line in open('$ppl_file'):
    try:
        d = json.loads(line)
        if 'ppl' in d and d.get('run_id','').startswith('$tag'):
            print(d['ppl'])
    except: pass
" 2>/dev/null | tail -1)
    if [ -z "$ppl" ]; then return; fi
    echo "[watcher] ===== RESULT: $tag PPL=$ppl ====="
    python3 -c "
ppl = float('$ppl')
g1b, silver, tune = $GATE_G1B, $GATE_SILVER, $GATE_TUNE
if ppl <= g1b:
    print(f'  PASS G1b gate ({ppl:.2f} <= {g1b}) → NO G1b needed, proceed to TQ export')
elif ppl <= silver:
    print(f'  PASS Silver gate ({ppl:.2f} <= {silver}) → LAUNCH G1b for more quant training')
elif ppl <= tune:
    print(f'  PASS Tune gate ({ppl:.2f} <= {tune}) → reduced quant quality, tune-only')
else:
    print(f'  FAIL all gates ({ppl:.2f} > {tune}) → QAT regression, investigate')
"
}

STEP25_DONE=0
FINAL_DONE=0

echo "[watcher] started at $(date -u '+%Y-%m-%dT%H:%M:%SZ') — watching $RUN"
echo "[watcher] step 25 checkpoint: $RUN/step_000025/state_dict.pt"
echo "[watcher] final checkpoint:   $RUN/final/state_dict.pt"
echo "[watcher] baseline PPL: 11.30 (canonical 4k single window)"

while true; do
    NOW=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

    # Check step 25
    if [ "$STEP25_DONE" -eq 0 ] && [ -f "$RUN/step_000025/state_dict.pt" ]; then
        STEP25_DONE=1
        echo "[watcher] [$NOW] step_000025 checkpoint FOUND"
        eval_ckpt "$RUN/step_000025/state_dict.pt" "g1a_step025"
        # Write PPL result to ppl/ dir by reading events
        python3 -c "
import json, pathlib
ev = pathlib.Path('$RUN/events.jsonl')
for line in ev.read_text().splitlines():
    try:
        d = json.loads(line)
        if d.get('eval_step') == 25 or (d.get('step') == 25 and 'ppl' in d):
            out = pathlib.Path('$PPL_OUT/g1a_step025.jsonl')
            out.write_text(json.dumps(d) + '\n')
    except: pass
" 2>/dev/null || true
        announce_ppl "$PPL_OUT/g1a_step025.jsonl" "g1a_step025"
    fi

    # Check final
    if [ "$FINAL_DONE" -eq 0 ] && [ -f "$RUN/final/state_dict.pt" ]; then
        FINAL_DONE=1
        echo "[watcher] [$NOW] FINAL checkpoint FOUND"
        eval_ckpt "$RUN/final/state_dict.pt" "g1a_final"
        announce_ppl "$PPL_OUT/g1a_final.jsonl" "g1a_final"

        echo ""
        echo "[watcher] ===== G1a COMPLETE ====="

        # Auto-generate completion report
        REPORT_OUT="$ROOT/docs/plans/g1a_completion_report_$(date -u '+%Y_%m_%d').md"
        FINAL_PPL=$(python3 -c "
import json, pathlib
for line in pathlib.Path('$PPL_OUT/g1a_final.jsonl').read_text().splitlines():
    try:
        d = json.loads(line)
        if 'ppl' in d: print(d['ppl'])
    except: pass
" 2>/dev/null | tail -1 || echo "unknown")
        STEP25_PPL=$(python3 -c "
import json, pathlib
for line in pathlib.Path('$PPL_OUT/g1a_step025.jsonl').read_text().splitlines():
    try:
        d = json.loads(line)
        if 'ppl' in d: print(d['ppl'])
    except: pass
" 2>/dev/null | tail -1 || echo "unknown")
        GATE_RESULT=$(python3 -c "
try:
    ppl = float('$FINAL_PPL')
    g1b, silver, tune = $GATE_G1B, $GATE_SILVER, $GATE_TUNE
    if ppl <= g1b:   print('PASS_G1B — no G1b needed, proceed to TQ export')
    elif ppl <= silver: print('PASS_SILVER — launch G1b for more quant training')
    elif ppl <= tune:   print('PASS_TUNE — reduced quality, tune-only')
    else:               print('FAIL — QAT regression, investigate')
except: print('unknown')
" 2>/dev/null)
        LAST_STEP=$(python3 -c "
import json
lines = open('$RUN/events.jsonl').readlines()
for line in reversed(lines):
    try:
        d = json.loads(line)
        if 'step' in d: print(d['step']); break
    except: pass
" 2>/dev/null || echo "?")
        FINAL_LOSS=$(python3 -c "
import json
lines = open('$RUN/events.jsonl').readlines()
for line in reversed(lines):
    try:
        d = json.loads(line)
        if 'loss' in d: print(f\"{d['loss']:.4f}\"); break
    except: pass
" 2>/dev/null || echo "?")
        cat > "$REPORT_OUT" << REPORT_EOF
# G1a QAT Completion Report
**Date:** $(date -u '+%Y-%m-%d %H:%M UTC')

## Results
| | |
|---|---|
| Run | g1a (FFN ternary, last-8-layers, 150 steps) |
| Final loss | $FINAL_LOSS (step $LAST_STEP) |
| Step 25 PPL | $STEP25_PPL |
| Final PPL | $FINAL_PPL |
| Baseline PPL | 11.30 (F32, wikitext2 4k single window) |
| Gate result | **$GATE_RESULT** |
| G1b gate | ≤$GATE_G1B (1.2×) |
| Silver gate | ≤$GATE_SILVER (1.35×) |
| Tune gate | ≤$GATE_TUNE (1.5×) |

## Next Steps
$(python3 -c "
try:
    ppl = float('$FINAL_PPL')
    g1b, silver = $GATE_G1B, $GATE_SILVER
    if ppl <= g1b:
        print('1. Run TQ export: \`python3 tools/training/rwkv7_export_strand.py --checkpoint artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8/final --out artifacts/lowbit_rwkv7/export/g1a --bits 2 --l 7 --strand-bin target/release/quantize-model\`')
        print('2. Wire Rust TQ dispatch (fill stubs in rwkv7_tq_loader.rs)')
        print('3. Run TQ parity gate')
        print('4. Commit and push')
    elif ppl <= silver:
        print('1. Assess G1b launch (all-layers ternary, ~26h more)')
        print('2. G1b command: python3 tools/training/rwkv7_qat.py --model artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8/final/state_dict.pt ...')
    else:
        print('1. Review loss curve — check for training instability')
        print('2. Consider requant_every reduction or LR decay')
except: print('PPL eval result not available')
" 2>/dev/null)

## Training Curve
$(python3 -c "
import json
for line in open('$RUN/events.jsonl').readlines():
    try:
        d = json.loads(line)
        if 'step' in d and 'loss' in d:
            print(f\"Step {d['step']:3d}: loss={d['loss']:.4f}  ema={d['loss_ema']:.4f}\")
    except: pass
" 2>/dev/null)
REPORT_EOF
        echo "[watcher] Completion report written to $REPORT_OUT"

        # ── Auto-launch Phase 2 chain ─────────────────────────────────────────
        CHAIN="$ROOT/tools/training/g1a_phase2_chain.sh"
        if [ -f "$CHAIN" ]; then
            echo "[watcher] Launching Phase 2 chain: $CHAIN"
            FINAL_PPL="$FINAL_PPL" GATE_RESULT="$GATE_RESULT" \
                bash "$CHAIN" >> "$ROOT/artifacts/lowbit_rwkv7/g1a_phase2_chain.log" 2>&1 &
            CHAIN_PID=$!
            echo "[watcher] Phase 2 chain started (PID $CHAIN_PID) — tail artifacts/lowbit_rwkv7/g1a_phase2_chain.log"
        else
            echo "[watcher] Phase 2 chain script not found at $CHAIN — skipping auto-launch"
        fi
        # ──────────────────────────────────────────────────────────────────────

        echo "[watcher] Done. Exiting."
        break
    fi

    if [ "$STEP25_DONE" -eq 1 ] && [ "$FINAL_DONE" -eq 1 ]; then
        break
    fi

    # Show live progress every 5 min
    LAST=$(tail -1 "$RUN/events.jsonl" 2>/dev/null || echo "{}")
    STEP=$(python3 -c "import json; d=json.loads('$LAST'); print(d.get('step','?'))" 2>/dev/null || echo "?")
    LOSS=$(python3 -c "import json; d=json.loads('$LAST'); print(f\"{d.get('loss',0):.4f}\")" 2>/dev/null || echo "?")
    echo "[watcher] [$NOW] step=$STEP loss=$LOSS (waiting for step 25 / final)"

    sleep 300  # 5 min
done
