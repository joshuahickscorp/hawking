#!/usr/bin/env bash
# conductor.sh — THE local orchestrator (v7, 2026-06-11). Consolidates the lane watcher,
# idle-bench watcher, pod-watch, the re-pass relaunch, and runbook §2 stall handling
# into one stateless-resumable loop. Every decision is derived from the filesystem
# each tick, so it survives any restart: just relaunch it.
#
# MECHANICAL actions (run inline, logged, pre-authorized):
#   - re-pass: night3 DONE but headline arm missing -> wipe DONE, relaunch (max 2)
#   - idle window after the full marathon -> definitive speed sweep (once)
#   - pod poll every 10 min: status line + mirror results/logs locally
#   - stall: qat python alive, no requant, newest log >35 min old -> pkill (runbook §2)
#   - replay (v7): quiet 6h window -> launch scripts/replay.sh (the idle invariant sweep)
# JUDGMENT events (print EVENT + exit -> the session wakes to interpret):
#   - VERDICT (any new/changed qat-*.json)  - FAILURE (new rc= in night3.log)
#   - POD_MILESTONE (new ppl json)          - POD_LADDER_EXITED / POD_UNREACHABLE
#   - SPEED_SWEEP_DONE                      - STALL_KILLED / REPASS_EXHAUSTED
#   - LEDGER_TELL (results-ledger check found 15-digit/contamination errors)
#   - POD_CHAIN_FAILURE / POD_MEM           - REPLAY_FAIL (bit-rot = max salience)
#
# ── v7 RULE CHANGES (2026-06-11) — habituation + salience + replay ──────────────
# All v6 behavior preserved; mechanical-before-judgment order unchanged. New:
#
# SALIENCE CLASSES (every event is now tagged; the full taxonomy table lives in
# ops/nervous-system.md):
#   S1  wake ALWAYS, never habituated:  REPASS_EXHAUSTED, POD_UNREACHABLE, POD_MEM,
#       REPLAY_FAIL. Emitted via evt1().
#   S2  wake, HABITUATABLE:             VERDICT, FAILURE, STALL_KILLED,
#       SPEED_SWEEP_DONE, POD_MILESTONE, POD_LADDER_EXITED, POD_CHAIN_FAILURE,
#       LEDGER_TELL. Emitted via evt2().
#   S3  log only, daily digest line:    REPLAY_PASS + every habituated suppression.
#       Emitted via evt3(). Never exits.
#
# HABITUATION (evt2): fingerprint = md5("CLASS|normalized key fields") (first 12 hex).
# Memory = scratch/.conductor-habituation, one line per fingerprint:
#   "<fp> <epoch_last> <suppress_count> <max_sev> <class> <key...>"
# (pruned to HAB_WINDOW on every write; persists across conductor exits — that is
# the point: the wake/relaunch cycle no longer resets the memory).
# Rules, in order:
#   1. first occurrence of a fingerprint -> WAKE (normal S2 behavior), record it.
#   2. repeat within HAB_WINDOW (6h) -> log "HABITUATED" + S3-count, do NOT exit,
#      UNLESS severity escalates:
#      2a. the event's numeric severity GREW past the recorded max (rc count, ledger
#          error count, pod json count, chain-failure count) -> WAKE "ESCALATED sev".
#      2b. safety valve: HAB_MAXSUP (10) suppressions of one fingerprint in a window
#          -> WAKE once "ESCALATED xN", fingerprint reset (nothing is silenced forever).
#   3. repeat OUTSIDE the window -> fresh cycle (wake again).
# Normalized keys per event (digits -> N where line text is the key, so timestamps/
# counters don't defeat the fingerprint): VERDICT=json name+content md5 (a genuinely
# new verdict is a new fingerprint and always wakes); FAILURE=last rc= line class;
# STALL_KILLED=log name; LEDGER_TELL=first ERROR line class; POD_MILESTONE=jsons=N;
# POD_CHAIN_FAILURE=last failure line class; POD_LADDER_EXITED/SPEED_SWEEP_DONE=
# constants (their old one-shot flag/file guards are kept too).
# Baseline counters (B_RC, B_POD, B_JSON, B_CF) are now updated BEFORE emitting, so a
# suppressed event does not re-fire every tick.
#
# REPLAY ("dreaming" = the immune system): at most once per quiet 6h window
# (stamp scratch/.replay-last at launch), box idle by the same pgrep discipline as
# the other idle jobs -> nohup scripts/replay.sh (it self-gates again + pid-locks; safe).
# Its verdict lands in scratch/.replay-verdict: PASS -> S3 log line (digest),
# FAIL -> S1 wake (bit-rot detected = maximum salience).
#
# DAILY DIGEST: once per calendar day one "DIGEST" line summarizes the last 24h of
# S3 events and the live habituation table.
cd /Users/scammermike/Downloads/strand || exit 1  # pinned root
M=scratch/qwen-05b
CL=scratch/conductor.log
SWEEP=research/speed-G0b-idle-bench.txt
HAB=scratch/.conductor-habituation
S3F=scratch/.conductor-s3
HAB_WINDOW=$((6*3600))   # repeats of one fingerprint within 6h are habituated
HAB_MAXSUP=10            # safety valve: Nth suppression in a window wakes anyway
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o IdentitiesOnly=yes -i $HOME/.ssh/id_ed25519 -p 40078 root@213.192.2.110"
log(){ echo "[cond $(date '+%d %H:%M:%S')] $*" >> "$CL"; }
wake(){ echo "EVENT: $*"; log "EVENT: $*"; [ -n "${CONDUCTOR_SELFTEST:-}" ] && return 0; exit 0; }

# ── salience emitters ───────────────────────────────────────────────────────────
# S1: wake always, never habituated (always-wake list)
evt1(){ wake "$* [S1]"; }

# S3: log only; counted into the daily digest; never exits
s3_note(){ echo "$(date +%s) $1" >> "$S3F"; }
evt3(){ local c=$1; shift; log "S3 $c :: $*"; s3_note "$c"; }

# habituation memory ops (file rewritten pruned-to-window on every write)
hab_put(){ # fp epoch count sev class key...
    local f=$1; shift
    { grep -v "^$f " "$HAB" 2>/dev/null \
        | awk -v n="$(date +%s)" -v w="$HAB_WINDOW" '($2 >= n-w)'; echo "$f $*"; } > "$HAB.tmp"
    mv "$HAB.tmp" "$HAB"
}
hab_del(){ grep -v "^$1 " "$HAB" 2>/dev/null > "$HAB.tmp"; mv "$HAB.tmp" "$HAB"; }

# S2: wake, habituatable.  evt2 CLASS KEY SEV detail...
evt2(){
    local class=$1 key=$2 sev=$3; shift 3
    local f now line last count psev
    f=$(printf '%s|%s' "$class" "$key" | md5 | cut -c1-12)
    now=$(date +%s)
    line=$(grep "^$f " "$HAB" 2>/dev/null | tail -1)
    if [ -n "$line" ]; then
        last=$(echo "$line" | awk '{print $2}')
        count=$(echo "$line" | awk '{print $3}')
        psev=$(echo "$line" | awk '{print $4}')
        if [ $((now - last)) -le "$HAB_WINDOW" ]; then
            count=$((count+1))
            if [ "${sev:-0}" -gt "${psev:-0}" ] 2>/dev/null; then
                hab_put "$f" "$now" "$count" "$sev" "$class" "$key"
                wake "$class ESCALATED sev $psev->$sev (x$count in window) :: $* [S2 fp=$f]"
                return 0
            fi
            if [ "$count" -ge "$HAB_MAXSUP" ]; then
                hab_del "$f"
                wake "$class ESCALATED x$count repeats in 6h window (safety valve) :: $* [S2 fp=$f]"
                return 0
            fi
            hab_put "$f" "$now" "$count" "$psev" "$class" "$key"
            log "HABITUATED $class x$count (fp=$f sev=$sev<=$psev) — suppressed: $*"
            s3_note "HABITUATED_$class"
            return 0
        fi
    fi
    hab_put "$f" "$now" 1 "${sev:-0}" "$class" "$key"
    wake "$class $* [S2 fp=$f]"
}

# daily digest (S3 channel + live habituation table), once per calendar day
digest(){
    local today; today=$(date +%F)
    [ "$(cat scratch/.conductor-digest-day 2>/dev/null)" = "$today" ] && return 0
    echo "$today" > scratch/.conductor-digest-day
    local cut s3sum habsum; cut=$(( $(date +%s) - 86400 ))
    s3sum=$(awk -v c="$cut" '$1>=c {print $2}' "$S3F" 2>/dev/null | sort | uniq -c \
            | awk '{printf "%s:x%s ", $2, $1}')
    [ -f "$S3F" ] && { awk -v c="$cut" '$1>=c' "$S3F" > "$S3F.tmp"; mv "$S3F.tmp" "$S3F"; }
    habsum=$(awk '{printf "%s:x%s ", $5, $3}' "$HAB" 2>/dev/null)
    log "DIGEST $today s3-24h[${s3sum:-none}] habituated-live[${habsum:-none}]"
}

# ── selftest (bash ops/conductor.sh selftest): exercises the habituation rules
# without touching the real memory files; prints PASS/FAIL lines and exits ──────
if [ "${1:-}" = "selftest" ]; then
    export CONDUCTOR_SELFTEST=1
    HAB=$(mktemp) S3F=$(mktemp) CL=/dev/null
    r1=$(evt2 T k1 1 first)                          # 1st -> wake
    r2=$(evt2 T k1 1 second)                         # repeat, same sev -> suppressed
    r3=$(evt2 T k1 5 third)                          # sev grew -> escalate wake
    r4=$(evt2 T k2 0 other)                          # different key -> wake
    for i in 1 2 3 4 5 6 7 8 9 10; do r5=$(evt2 T k3 0 "rep$i"); done  # valve at x10
    ok=1
    case "$r1" in EVENT:\ T\ first*) ;; *) echo "FAIL 1st-occurrence should wake: $r1"; ok=0;; esac
    [ -z "$r2" ] || { echo "FAIL repeat should suppress: $r2"; ok=0; }
    case "$r3" in *ESCALATED\ sev\ 1-\>5*) ;; *) echo "FAIL sev-growth should escalate: $r3"; ok=0;; esac
    case "$r4" in EVENT:\ T\ other*) ;; *) echo "FAIL new key should wake: $r4"; ok=0;; esac
    case "$r5" in *"safety valve"*) ;; *) echo "FAIL valve should fire at x$HAB_MAXSUP: $r5"; ok=0;; esac
    rm -f "$HAB" "$HAB.tmp" "$S3F" "$S3F.tmp"
    [ "$ok" = 1 ] && { echo "selftest PASS (5/5 habituation rules)"; exit 0; } || exit 1
fi

json_snap(){ ls -l $M/qat-*.json 2>/dev/null | md5; }
B_JSON=$(json_snap)
B_RC=$(grep -c 'rc=' $M/night3.log 2>/dev/null | head -1); B_RC=${B_RC:-0}
B_POD=$(ls scratch/pod-results/ppl_*.json 2>/dev/null | wc -l | tr -d ' ')
podfails=0; tick=0; memhits=0; B_CF=""
front_snap(){ ls -l research/pv-dp/*.json research/mp-frontier/*/ppl_*.json research/debias-*.json research/down-protect/*.json 2>/dev/null | md5; }
B_FRONT=$(front_snap)
touch scratch/.conductor-habituation 2>/dev/null
log "conductor up v7 (rc-base=$B_RC pod-base=$B_POD hab=$(wc -l < "$HAB" 2>/dev/null | tr -d ' ' || echo 0) entries)"

while :; do
    sleep 60; tick=$((tick+1))
    digest

    # 1) RE-PASS FIRST (mechanical — must run before judgment exits): DONE written but the headline arm never produced
    if [ -f scratch/.night3-done ] && [ ! -f $M/qat-pv.json ] \
       && ! pgrep -f 'strand-act2-night3.sh' >/dev/null 2>&1; then
        n=$(cat scratch/.repass-count 2>/dev/null || echo 0)
        if [ "$n" -ge 2 ]; then evt1 "REPASS_EXHAUSTED ($n attempts, pv.json still missing)"; fi
        echo $((n+1)) > scratch/.repass-count
        log "re-pass #$((n+1)): wiping DONE, relaunching night3 + guardian"
        rm -f scratch/.night3-done
        nohup caffeinate -dimsu ./scripts/strand-act2-night3.sh >> $M/night3.log 2>&1 &
        sleep 3
        pgrep -f 'scratch/guardian.sh' >/dev/null 2>&1 || { nohup ./scratch/guardian.sh > /dev/null 2>&1 & }
        log "re-pass launched"
    fi

    # 2) VERDICT — any new/changed arm json (S2; key = filename+content hash, so a
    #    genuinely new verdict always wakes; only snapshot-churn repeats habituate)
    snap=$(json_snap)
    if [ "$snap" != "$B_JSON" ]; then
        B_JSON=$snap
        new=$(ls -t $M/qat-*.json | head -1)
        evt2 VERDICT "$new|$(md5 -q "$new" 2>/dev/null)" 0 \
            "$new :: $(tr -d ' \n' < "$new" | head -c 400)"
    fi

    # 2b) FRONTIER PROMOTION — new/changed quality-density result (research/pv-dp,
    # mp-frontier, debias, down-protect) is run through the shared promote.py grammar:
    # loss-tax billed, promotion_state stamped into the json, grammar line surfaced.
    # The conductor now speaks gates — a PROMOTE_CLOUD verdict is what earns a pod
    # confirm; KILLED/INCOMPLETE never auto-scale. All these lanes are 0.5B (anchor
    # 12.536), so the model hint is fixed here.
    fsnap=$(front_snap)
    if [ "$fsnap" != "$B_FRONT" ]; then
        B_FRONT=$fsnap
        fnew=$(ls -t research/pv-dp/*.json research/mp-frontier/*/ppl_*.json research/debias-*.json research/down-protect/*.json 2>/dev/null | head -1)
        if [ -n "$fnew" ]; then
            grammar=$(/usr/local/bin/python3 scripts/promote.py "$fnew" --quiet --model qwen-05b 2>/dev/null | head -1)
            evt2 PROMOTE "$fnew|$(md5 -q "$fnew" 2>/dev/null)" 0 "${grammar:-promote.py failed on $fnew}"
        fi
    fi

    # 3) FAILURE — new rc= lines (S2; key = last rc line class digits->N; sev = total count)
    RC=$(grep -c 'rc=' $M/night3.log 2>/dev/null | head -1); RC=${RC:-0}
    if [ "$RC" -gt "$B_RC" ]; then
        B_RC=$RC
        fkey=$(grep 'rc=' $M/night3.log 2>/dev/null | tail -1 | sed 's/[0-9][0-9]*/N/g')
        evt2 FAILURE "$fkey" "$RC" "new rc in night3.log :: $(tail -3 $M/night3.log | tr '\n' '|')"
    fi

    # 4) IDLE WINDOW — full marathon quiet => definitive speed sweep (once)
    if [ ! -f "$SWEEP" ] && [ -f $M/qat-pv.json ] && [ -f $M/qat-pv3bit.json ] \
       && ! pgrep -f 'strand-qat.py|quantize-model' >/dev/null 2>&1; then
        sleep 180
        if ! pgrep -f 'strand-qat.py|quantize-model' >/dev/null 2>&1; then
            log "idle window — running the definitive speed sweep"
            ( mkdir -p research && {
                echo "== definitive idle sweep $(date '+%F %H:%M') =="
                ./target/release/gate-interleave
                echo; echo "== canonical baseline gate (same idle box) =="
                ./target/release/gate-decode-speed
            } > research/speed-G0b-idle-bench.txt 2>&1 )
            evt2 SPEED_SWEEP_DONE sweep 0 ":: $(grep -E 'verdict' "$SWEEP" | tr '\n' '|')"
        fi
    fi

    # 4b) TERNARY LOW-LR FILLER (owner-approved 2026-06-10): after the speed sweep,
    # box idle => one depth arm from the 2k checkpoint at the corrected restart LR
    # (3e-5 — the 3k arm's full-LR restart burned ~600 steps re-converging). Its json
    # landing wakes the session via the VERDICT event (#2). Runs at most once.
    if [ -f "$SWEEP" ] && [ ! -f $M/qat-ternary-lowlr.json ] && [ ! -f scratch/.ternary-lowlr-launched ] \
       && [ -d $M/qat-ternary-kd-2k-hf ] \
       && ! pgrep -f 'strand-qat.py|quantize-model' >/dev/null 2>&1; then
        touch scratch/.ternary-lowlr-launched
        log "idle filler: ternary low-LR depth arm (500 steps @3e-5 from 2k-hf)"
        PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.85 \
        nohup caffeinate -dimsu /usr/local/bin/python3 scripts/strand-qat.py \
            --ctx 512 --eval-chunks 64 --eval-ctx 2048 --grad-accum 4 --batch 1 \
            --device mps --grad-checkpoint --lr 3e-5 --kd \
            --model $M/qat-ternary-kd-2k-hf --quant ternary --steps 500 \
            --train-chunks 2000 --eval-every 250 \
            --save $M/qat-ternary-lowlr.pt --save-hf $M/qat-ternary-lowlr-hf \
            --out $M/qat-ternary-lowlr.json >> $M/qat-ternary-lowlr.log 2>&1 &
        log "ternary-lowlr launched"
    fi

    # 4c) GATE QUEUE (2026-06-12): the durable local dev schedule. run-next-gate.sh
    # self-gates on box-idle + a pid lock and runs the first pending scripts/gates/*.sh
    # (each self-guards on its output sentinel). This is where "all dev is scheduled
    # locally" lives — drop a numbered gate script in and it runs when the box frees.
    bash scripts/run-next-gate.sh 2>/dev/null

    # 5) STALL (runbook §2, pre-authorized): qat alive, no requant, log silent >35 min
    # Stall rule applies ONLY to marathon-context runs (agent smokes/profilers also run
    # strand-qat.py and must never be friendly-fired — learned 2026-06-11 22:0x).
    if pgrep -f 'strand-act2-night3' >/dev/null 2>&1 \
       && pgrep -f strand-qat.py >/dev/null 2>&1 && ! pgrep -f quantize-model >/dev/null 2>&1; then
        newest=$(ls -t $M/qat-*.log 2>/dev/null | head -1)
        if [ -n "$newest" ]; then
            age=$(( $(date +%s) - $(stat -f %m "$newest") ))
            if [ "$age" -gt 2100 ]; then
                log "STALL: $newest silent ${age}s — pkill strand-qat.py (crash-resume covers it)"
                pkill -f strand-qat.py
                evt2 STALL_KILLED "$newest" 0 "$newest silent ${age}s"
            fi
        fi
    fi

    # 5b) REPLAY (v7, mechanical launch): at most once per quiet 6h window. The box
    # must be idle by the same pgrep discipline as #4/#4b; replay.sh self-gates and
    # pid-locks again, so a race here is harmless.
    rl=$(cat scratch/.replay-last 2>/dev/null || echo 0)
    if [ $(( $(date +%s) - rl )) -ge $((6*3600)) ] \
       && ! pgrep -f 'strand-qat.py|quantize-model' >/dev/null 2>&1 \
       && ! pgrep -f 'scripts/repla[y].sh' >/dev/null 2>&1; then
        date +%s > scratch/.replay-last
        log "quiet 6h window — launching replay (idle invariant sweep)"
        nohup ./scripts/replay.sh >> scratch/replay.log 2>&1 &
    fi
    # 5c) REPLAY VERDICT pickup: PASS -> S3 (digest only); FAIL -> S1 (bit-rot = max salience)
    if [ -f scratch/.replay-verdict ]; then
        rv=$(cat scratch/.replay-verdict); rm -f scratch/.replay-verdict
        case "$rv" in
            PASS*) evt3 REPLAY_PASS "$rv" ;;
            *)     evt1 "REPLAY_FAIL bit-rot detected :: $rv" ;;
        esac
    fi

    # 6) POD — every 10th tick: status, mirror, milestone
    if [ $((tick % 10)) -eq 0 ]; then
        S=$($SSH 'echo "jsons=$(ls /workspace/strand-results/ 2>/dev/null | grep -c "^ppl_.*\.json$") ladder=$(pgrep -f "strand-ladder[.]sh" >/dev/null && echo up || echo DOWN) chain=$(pgrep -f "bash /workspace/pod-chai[n]" >/dev/null && echo up || echo idle) mem=$(awk "/^anon /{printf \"%.0f\", \$2/1e9}" /sys/fs/cgroup/memory.stat 2>/dev/null) cfg=$(grep -a -oE "== (START|DONE) [^ ]+" /workspace/strand-ladder.log 2>/dev/null | tail -1 | sed "s/== //;s/ /:/")"' 2>/dev/null)
        if [ -z "$S" ]; then
            podfails=$((podfails+1)); log "pod ssh FAIL #$podfails"
            [ "$podfails" -ge 3 ] && evt1 "POD_UNREACHABLE 3 consecutive polls (ask owner: echo \"TCP=\$RUNPOD_PUBLIC_IP:\$RUNPOD_TCP_PORT_22\")"
        else
            podfails=0; log "pod $S"
            scp -q -o BatchMode=yes -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -P 40078 \
                "root@213.192.2.110:/workspace/strand-results/*.json" scratch/pod-results/ 2>/dev/null
            for lg in chain gate ladder governor; do
                scp -q -o BatchMode=yes -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -P 40078 \
                    "root@213.192.2.110:/workspace/strand-$lg.log" scratch/pod-results/ 2>/dev/null
            done
            scp -q -o BatchMode=yes -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -P 40078 \
                "root@213.192.2.110:/workspace/pod-governor.log" scratch/pod-results/ 2>/dev/null
            # LEDGER HOOK (audit 3.2): pod-result discoveries -> results ledger
            # (idempotent ingest + the 15-digit tell / harness-key checks; torch-free)
            ./scripts/strand-eval ledger ingest scratch/pod-results --quiet >> "$CL" 2>&1 || true
            lc=$(./scripts/strand-eval ledger check 2>/dev/null | grep -c '^ERROR ') || lc=0
            if [ "${lc:-0}" -gt 0 ]; then
                lkey=$(./scripts/strand-eval ledger check 2>/dev/null | grep '^ERROR ' | head -1 | sed 's/[0-9][0-9]*/N/g')
                evt2 LEDGER_TELL "$lkey" "$lc" "$lc error(s) :: $(./scripts/strand-eval ledger check 2>/dev/null | grep '^ERROR ' | head -2 | tr '\n' '|')"
            fi
            # POD MEMORY GUARD (2026-06-10: cgroup OOM killed mp_light's quant at 121/125GB):
            # two consecutive polls >= 115GB anon+cache => wake for intervention.
            pm=$(echo "$S" | sed -n 's/.*mem=\([0-9]*\).*/\1/p')
            if [ -n "$pm" ] && [ "$pm" -ge 115 ]; then
                memhits=$((memhits+1))
                [ "$memhits" -ge 2 ] && evt1 "POD_MEM ${pm}GB/125 two consecutive polls :: $S"
            else
                memhits=0
            fi
            # POD CHAIN FAILURE SCAN (v6): new FAILED/rc= lines in the mirrored chain log
            cf=$(grep -ac "FAILED\|rc=[0-9]" scratch/pod-results/strand-chain.log 2>/dev/null | head -1); cf=${cf:-0}
            if [ -z "$B_CF" ]; then B_CF=$cf; fi
            if [ "$cf" -gt "$B_CF" ]; then
                B_CF=$cf
                ckey=$(grep -a "FAILED\|rc=[0-9]" scratch/pod-results/strand-chain.log 2>/dev/null | tail -1 | sed 's/[0-9][0-9]*/N/g')
                evt2 POD_CHAIN_FAILURE "$ckey" "$cf" "new failure lines :: $(grep -a "FAILED\|rc=[0-9]" scratch/pod-results/strand-chain.log | tail -2 | tr '\n' '|')"
            fi
            np=$(echo "$S" | sed -n 's/.*jsons=\([0-9]*\).*/\1/p')
            if [ -n "$np" ] && [ "$np" -gt "$B_POD" ]; then
                op=$B_POD; B_POD=$np
                evt2 POD_MILESTONE "jsons=$np" "$np" "jsons $op -> $np :: $S"
            fi
            if echo "$S" | grep -q 'ladder=DOWN' && [ ! -f scratch/.pod-ladder-down-noted ]; then
                touch scratch/.pod-ladder-down-noted
                evt2 POD_LADDER_EXITED ladder-down 0 ":: $S"
            fi
        fi
    fi
done
