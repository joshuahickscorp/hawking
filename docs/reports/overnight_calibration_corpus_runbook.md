# Autonomous runbook — calibration corpus build

**For the new Claude-Code session that supervises this overnight.**
Self-contained: everything you need is in this file. No user
intervention required after Step 1 hands off.

## Goal

Build the full **10,000-sequence** calibration corpus for V2-Lite. This
is the largest single unblocker in the
[execution plan](../plans/dismantle-execution-plan-enchanted-salamander.md):
mixed-precision quant calibration (+5–10 tps), eagle5 activation
predictor (+5–12 tps), vocab-prune (+2–5 tps), and Q8 KV
layer-differential refinement (+1–5 tps beyond the uniform Q8 prototype
landed in the working session). The corpus run is the single highest-EV
overnight task — and pause/resume is rock-solid, so we target the full
10k from the start.

## Architecture — fully autonomous

```
   ┌──────────────────────────────────────────────────────────┐
   │  new Claude-Code session (the monitor)                    │
   │                                                            │
   │  /loop 20m  →  corpus_monitor.sh  →  classify status      │
   │                       │                                    │
   │                       └─ stalled?  → kill build PID        │
   │                       └─ dead?     → relaunch watchdog     │
   │                       └─ done?     → exit loop, report     │
   └──────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌──────────────────────────────────────────────────────────┐
   │  run_corpus_autonomous.sh   (watchdog, daemon-style)      │
   │                                                            │
   │   while true:                                              │
   │      python3 build_corpus.py  --skip-existing             │
   │      (write heartbeat.json every 30 s)                    │
   │      on crash → exponential backoff → restart             │
   │      on STOP flag → exit cleanly                          │
   └──────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌──────────────────────────────────────────────────────────┐
   │  build_corpus.py  (the actual work)                       │
   │  Per-shard idempotent: --skip-existing default-on.        │
   │  Crashed mid-shard? Next run resumes from that shard.     │
   └──────────────────────────────────────────────────────────┘
```

**Three artifacts deliver this autonomy:**

1. [tools/training/run_corpus_autonomous.sh](../tools/training/run_corpus_autonomous.sh)
   — self-restarting watchdog. Loops forever calling `build_corpus.py`.
   Writes heartbeat JSON every 30 s. Backs off on crashes.
   Stops cleanly when `artifacts/calibration/STOP` is touched.
2. [tools/training/corpus_monitor.sh](../tools/training/corpus_monitor.sh)
   — one-shot status check. Returns JSON + an exit code that classifies
   the state: 0 healthy, 1 stalled, 2 dead-watchdog, 3 done.
3. [tools/training/build_corpus.py](../tools/training/build_corpus.py)
   — the actual capture script. **`--skip-existing` is on by default**,
   so re-runs resume from where they left off with zero data loss.

## Hardware reality + plan

- V2-Lite is 16B params (~31 GB in fp16) on 18 GB unified memory.
  `accelerate` will disk-offload the inactive experts; expect throughput
  in the 1–3 tps range.
- 10k sequences × ~512 tokens / 2 tps ≈ **710 hours = ~30 days**
  continuous if every layer is hooked at fp16.
- User's machine stays running at work; pause/resume is built in via
  per-shard `.parquet` checkpoints. So 30 days *of total compute* is
  achievable across many sessions.
- **If throughput holds at ≥ 2 tps**: ~30 days continuous.
- **If we trim** `--max-tokens-per-seq 256` and skip the heavy
  `intermediate` capture: roughly half the tokens and half the per-token
  cost = ~7–10 days.

The watchdog runs until either (a) `MAX_SEQUENCES` shards complete,
(b) the user touches `artifacts/calibration/STOP`, or (c) the monitor
session kills the watchdog. Pause and resume are free — kill anything,
restart the watchdog, work continues from the next unfinished shard.

## Step 0 — sanity (~2 min)

```bash
cd /Users/scammermike/Downloads/dismantle
df -h .                                                  # need ≥ 35 GB free
python3 --version                                        # 3.10+
ls tools/training/*.sh tools/training/build_corpus.py    # all 3 present
```

## Step 1 — one-time setup (~15 min)

```bash
cd /Users/scammermike/Downloads/dismantle

# Isolated venv keeps the heavy install out of the system Python.
python3 -m venv .venv-calibration
source .venv-calibration/bin/activate
pip install --upgrade pip
pip install -r tools/training/requirements.txt

# Verify MPS:
python3 -c "import torch; print('torch', torch.__version__, 'mps:', torch.backends.mps.is_available())"

# Confirm HF download works (V2-Lite-Chat is public; no auth needed):
python3 -c "from transformers import AutoTokenizer; \
    AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V2-Lite-Chat', trust_remote_code=True); \
    print('tokenizer OK')"
```

If the tokenizer call 401s, run `hf auth login` (or
`huggingface-cli login` on older CLI) and retry.

## Step 2 — smoke test (~30 min, GATE)

Validate the full pipeline with a tiny run before committing the
big one. **Skip this only if you've already smoke-tested in a prior
session.**

```bash
cd /Users/scammermike/Downloads/dismantle
source .venv-calibration/bin/activate

time python3 tools/training/build_corpus.py \
    --max-sequences 8 \
    --max-tokens-per-seq 256 \
    --shard-size 4 \
    --device mps \
    --dtype float16 \
    --quantize-intermediates int8 \
    --out artifacts/calibration/v2_lite_corpus_smoke \
    2>&1 | tee artifacts/calibration/smoke.log
```

Pass conditions:
- 2 shards produced under `v2_lite_corpus_smoke/`.
- Each shard parses cleanly:
  ```bash
  python3 -c "
  import pyarrow.parquet as pq
  t = pq.read_table('artifacts/calibration/v2_lite_corpus_smoke/shard_0000.parquet')
  print(t.schema); print(t.num_rows, 'rows', round(t.nbytes/1e6, 1), 'MB')
  "
  ```
- Observed tps ≥ 1.0 (compute from `time` output: 8 × 256 / wall_sec).

If any of those fail, **do not start the overnight run** — debug
first. Common fixes: drop to `--device cpu`, drop `--max-tokens-per-seq
128`, drop `--capture residual_in,expert_idx,routing_logits,h_high,output_logits`
to skip the heavy `intermediate` capture.

## Step 3 — kick off the autonomous run

```bash
cd /Users/scammermike/Downloads/dismantle

# Defaults are pre-set in the watchdog: MAX_SEQUENCES=10000,
# MAX_TOKENS_PER_SEQ=512, DEVICE=mps, DTYPE=float16. Override via env if needed.
nohup ./tools/training/run_corpus_autonomous.sh \
    > /dev/null 2>&1 &
disown
echo "Watchdog PID: $(cat artifacts/calibration/overnight.pid)"

# First heartbeat within 5 seconds:
sleep 6
cat artifacts/calibration/heartbeat.json
```

The watchdog now runs detached. Closing the terminal won't kill it
(`nohup` + `disown`). You can verify autonomy by:

```bash
ps -p $(cat artifacts/calibration/overnight.pid) -o pid,stat,etime,command
```

`STAT` will show `S` (sleeping) or `R` (running), `PPID` will be 1
(init/launchd — fully detached).

## Step 4 — set up the autonomous monitor (one /loop call)

Tell **this new Claude-Code session** to monitor on a schedule. Single
slash command:

```
/loop 20m bash tools/training/corpus_monitor.sh
```

That triggers the monitor every 20 minutes. The /loop hook checks the
exit code:
- **0 (healthy)**: do nothing, print one-line status.
- **1 (stalled)**: kill the BUILD pid — watchdog will auto-restart it:
  ```bash
  python3 -c "import json; print(json.load(open('artifacts/calibration/heartbeat.json'))['build_pid'])" \
      | xargs -I{} kill {}
  ```
- **2 (dead watchdog)**: check the action field in the JSON.
  - `relaunch_watchdog`:
    ```bash
    cd /Users/scammermike/Downloads/dismantle
    nohup ./tools/training/run_corpus_autonomous.sh > /dev/null 2>&1 &
    ```
  - `human_setup_required_step_1`: venv missing — go run Step 1 (pip
    install). Do NOT loop-relaunch; the watchdog can't help until the
    venv exists.
- **3 (done)**: exit the /loop, report final stats to the user.

Concise loop prompt for the monitor agent (paste into the loop's
command field):

```
bash -c '
cd /Users/scammermike/Downloads/dismantle
STATUS_JSON=$(bash tools/training/corpus_monitor.sh)
RC=$?
echo "$STATUS_JSON"
ACTION=$(echo "$STATUS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)[\"action\"])")
case $RC in
  0) echo "OK" ;;
  1) echo "STALLED → killing build PID";
     BUILD_PID=$(python3 -c "import json; print(json.load(open(\"artifacts/calibration/heartbeat.json\"))[\"build_pid\"])");
     if [ -n "$BUILD_PID" ] && [ "$BUILD_PID" != "-" ]; then kill "$BUILD_PID" 2>/dev/null || true; fi ;;
  2) if [ "$ACTION" = "human_setup_required_step_1" ]; then
       echo "BLOCKED: venv missing. Run Step 1 (pip install) before the loop can do anything.";
       touch artifacts/calibration/LOOP_DONE;
     else
       echo "DEAD watchdog → relaunching";
       nohup ./tools/training/run_corpus_autonomous.sh > /dev/null 2>&1 & disown;
     fi ;;
  3) echo "DONE — corpus complete"; touch artifacts/calibration/LOOP_DONE ;;
esac
'
```

If `LOOP_DONE` exists, the next /loop tick can see that and stop
itself (or you can manually stop the /loop).

## Step 5 — pause/resume / clean stop

To stop cleanly: `touch artifacts/calibration/STOP`. Watchdog notices
within 30 seconds and exits after gracefully ending the current shard.

To resume later: simply re-run Step 3. Existing shards are skipped via
`--skip-existing`, work continues from the next unfinished shard. No
state corruption possible — each shard is atomic.

To extend past 10k: set `MAX_SEQUENCES=20000` env before the Step 3
relaunch. Already-built 10k shards stay, only the additional 10k get
built.

## Step 6 — final verification (when status hits "done")

```bash
cd /Users/scammermike/Downloads/dismantle
ls artifacts/calibration/v2_lite_corpus/shard_*.parquet | wc -l
du -sh artifacts/calibration/v2_lite_corpus/

python3 -c "
import pyarrow.parquet as pq
import glob
shards = sorted(glob.glob('artifacts/calibration/v2_lite_corpus/shard_*.parquet'))
total_rows = 0
for s in shards:
    t = pq.read_table(s)
    total_rows += t.num_rows
print(f'{len(shards)} shards, {total_rows} rows total')
print(f'first shard schema: {pq.read_table(shards[0]).column_names[:6]}...')
"
```

Then report back to the working session — the four downstream levers
(mixed-precision quant, eagle5, vocab-prune, Q8 KV per-layer) are
all unblocked.

## Files used (read-only context for the monitor)

- `tools/training/build_corpus.py` — the actual capture script
- `tools/training/run_corpus_autonomous.sh` — watchdog (do not edit
  while running; if you must, touch STOP first)
- `tools/training/corpus_monitor.sh` — status check
- `tools/training/README.md` — script overview
- `plans/dismantle-execution-plan-enchanted-salamander.md` § Calibration

## Don'ts

- **Don't `git checkout`.** The working session has many uncommitted
  M-files; the dirty tree is authoritative.
- **Don't commit autonomously.** User's global rule: no Claude
  attribution on commits/PRs.
- **Don't auto-pip-install** outside the venv. Step 1 is the gate.
- **Don't reduce `--max-sequences` below 200** without checkpointing
  the existing shards — partial corpora aren't enough for eagle5
  training.
- **Don't kill the watchdog** without first touching STOP. Hard-kill
  in the middle of a shard wastes that shard's compute (the next start
  redoes it from scratch).
