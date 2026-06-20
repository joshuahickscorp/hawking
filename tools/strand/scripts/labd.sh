#!/usr/bin/env bash
# labd.sh — lab daemon: the conductor pattern, config-driven (Tier 3b prototype).
# Spec, provenance, example config, known limits: ops/labd-design.md. The contract:
#   1. Stateless-resumable: every decision re-derived from the filesystem each tick.
#      Relaunch is always safe; detector cursors persist in the spool, never in memory.
#   2. Pass 1 = MECHANICAL rules (type=action): pre-authorized recipes, run inline, logged.
#   3. Pass 2 = JUDGMENT rules (type=event): print "EVENT: <name> :: <detail>", exit 0.
#      The tracked launcher wakes an LLM session; relaunching labd acknowledges the event.
#   4. Pass 1 always completes before pass 2 may exit (mechanical-before-judgment, paid for
#      2026-06-10 — even exhaust escalations defer to the end of pass 1).
#   5. Config re-read every tick (hot-edit); the engine stays dumb, protocol lives in recipes.
# Usage: labd.sh <config> [--once|--check]   # --once = single tick; --check = parse + list
CONF="$1"; MODE="${2:-}"
[ -n "$CONF" ] && [ -f "$CONF" ] || { echo "usage: labd.sh <config> [--once|--check]" >&2; exit 2; }
TICK=60 SPOOL=labd-spool LOG=labd.log CWD="" tick=0 PEND=""

trim(){ local s="$*"; s="${s#"${s%%[![:space:]]*}"}"; printf '%s' "${s%"${s##*[![:space:]]}"}"; }
log(){ echo "[labd $(date '+%d %H:%M:%S')] $*" >> "$LOG"; }
mtime(){ stat -f %m "$1" 2>/dev/null || stat -c %Y "$1" 2>/dev/null; }

# ---- predicates usable in `if =` / `detail =` (evaluated in a subshell; $RULE is set) ----
alive(){ pgrep -f "$1" >/dev/null 2>&1; }  # caller's duty: bracket-trick the pattern (doc §9c)
quiet(){ local n; n=$(ls -t $1 2>/dev/null | head -1)  # quiet '<glob>' <secs>: newest mtime older
  [ -z "$n" ] && return 0; [ $(( $(date +%s) - $(mtime "$n") )) -gt "$2" ]; }
changed(){ # changed <key> <glob...> — listing differs from the committed cursor; stages the new one
  local cur="$SPOOL/cur.$RULE.$1" snap; shift
  snap=$(ls -l $* 2>/dev/null | cksum)
  [ -f "$cur" ] || { printf '%s' "$snap" > "$cur"; return 1; }   # first sight: seed, do not fire
  [ "$snap" != "$(cat "$cur")" ] || return 1
  printf '%s' "$snap" > "$cur.stage"
}
grew(){ # grew <key> <file> <pattern> — match count increased. stdout-only read of grep -c:
  local cur="$SPOOL/cur.$RULE.$1" n old                # immune to the `|| echo 0` 0\n0 trap (doc §12)
  n=$(grep -c -- "$3" "$2" 2>/dev/null); n=${n:-0}
  old=$(cat "$cur" 2>/dev/null)
  [ -n "$old" ] || { printf '%s' "$n" > "$cur"; return 1; }
  [ "$n" -gt "$old" ] 2>/dev/null || return 1
  printf '%s' "$n" > "$cur.stage"
}

# ---- config: full-line # comments ONLY; globals (tick/spool/log/cwd), then [rule] key = value ----
parse(){
  RNAME=() RTYPE=() RIF=() RRUN=() REVENT=() RDETAIL=() REVERY=() RMAX=() REXH=() RONESHOT=() RORDER=()
  NR=0; local line key val cur=-1
  while IFS= read -r line || [ -n "$line" ]; do
    line=$(trim "$line"); case "$line" in ''|'#'*) continue;; esac
    if [[ "$line" == \[*\] ]]; then
      cur=$NR; NR=$((NR+1)); RNAME[cur]="${line:1:${#line}-2}"
      RTYPE[cur]=action REVERY[cur]=1 RMAX[cur]=0 RONESHOT[cur]=0 RORDER[cur]=50
      RIF[cur]="" RRUN[cur]="" REVENT[cur]="" RDETAIL[cur]="" REXH[cur]=""
      continue
    fi
    case "$line" in *=*) ;; *) continue;; esac
    key=$(trim "${line%%=*}"); val=$(trim "${line#*=}")
    if [ "$cur" -lt 0 ]; then case "$key" in
      tick) TICK="$val";; spool) SPOOL="$val";; log) LOG="$val";; cwd) CWD="$val";; esac
    else case "$key" in
      type) RTYPE[cur]="$val";;    if) RIF[cur]="$val";;         run) RRUN[cur]="$val";;
      event) REVENT[cur]="$val";;  detail) RDETAIL[cur]="$val";; every) REVERY[cur]="$val";;
      max) RMAX[cur]="$val";;      exhaust_event) REXH[cur]="$val";;
      oneshot) if [ "$val" = true ]; then RONESHOT[cur]=1; else RONESHOT[cur]=0; fi;;
      order) RORDER[cur]="$val";; esac
    fi
  done < "$CONF"
}
by_order(){ local i; for ((i=0;i<NR;i++)); do printf '%03d %s\n' "${RORDER[i]}" "$i" 2>/dev/null; done | sort -n | cut -d' ' -f2; }
commit(){ local s; for s in "$SPOOL/cur.$1".*.stage; do [ -e "$s" ] && mv -f "$s" "${s%.stage}"; done; }
fire(){ commit "$1"; log "EVENT($1): $2"; echo "EVENT: $2"; exit 0; }

do_action(){ local i="$1" name="${RNAME[i]}" ev="${REVERY[i]:-1}" n
  [ "$ev" -ge 1 ] 2>/dev/null || ev=1; [ $(( tick % ev )) -eq 0 ] || return 0
  [ "${RONESHOT[i]}" = 1 ] && [ -f "$SPOOL/done.$name" ] && return 0
  [ -n "${RRUN[i]}" ] || return 0
  if [ -n "${RIF[i]}" ]; then ( RULE="$name"; eval "${RIF[i]}" ) || return 0; fi
  if [ "${RMAX[i]:-0}" -gt 0 ] 2>/dev/null; then
    n=$(cat "$SPOOL/count.$name" 2>/dev/null); n=${n:-0}
    if [ "$n" -ge "${RMAX[i]}" ]; then   # budget spent, condition still true -> escalate to judgment
      [ -n "${REXH[i]}" ] && [ -z "$PEND" ] && PEND="$name|${REXH[i]} (budget $n/${RMAX[i]} spent, condition still true)"
      return 0
    fi
    echo $((n+1)) > "$SPOOL/count.$name"   # reset = rm spool/count.<rule> (a one-touch gate)
  fi
  log "action $name: firing"
  ( RULE="$name"; eval "${RRUN[i]}" ) >> "$LOG" 2>&1
  log "action $name: rc=$?"
  [ "${RONESHOT[i]}" = 1 ] && touch "$SPOOL/done.$name"
  commit "$name"   # changed/grew-triggered actions are at-most-once per change, even on rc!=0
}
do_event(){ local i="$1" name="${RNAME[i]}" ev="${REVERY[i]:-1}" d=""
  [ "$ev" -ge 1 ] 2>/dev/null || ev=1; [ $(( tick % ev )) -eq 0 ] || return 0
  [ "${RONESHOT[i]}" = 1 ] && [ -f "$SPOOL/done.$name" ] && return 0
  [ -n "${REVENT[i]}" ] && [ -n "${RIF[i]}" ] || return 0
  ( RULE="$name"; eval "${RIF[i]}" ) || return 0
  [ -n "${RDETAIL[i]}" ] && d=$( ( RULE="$name"; eval "${RDETAIL[i]}" ) 2>&1 | tr '\n' ' ' | head -c 500 )
  [ "${RONESHOT[i]}" = 1 ] && touch "$SPOOL/done.$name"
  fire "$name" "${REVENT[i]}${d:+ :: $d}"
}

parse
[ -n "$CWD" ] && { cd "$CWD" || { echo "labd: cd $CWD failed" >&2; exit 2; }; }
if [ "$MODE" = --check ]; then
  for i in $(by_order); do printf '%3s %-7s %-22s %s\n' "${RORDER[i]}" "${RTYPE[i]}" "${RNAME[i]}" "${REVENT[i]:-${RRUN[i]:0:70}}"; done
  echo "OK: $NR rules, tick=${TICK}s, spool=$SPOOL, log=$LOG"; exit 0
fi
mkdir -p "$SPOOL"
if [ -d "$SPOOL/lock" ]; then   # stale-lock takeover is pid-based (kill -0): no pgrep, no self-match
  p=$(cat "$SPOOL/lock/pid" 2>/dev/null)
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { echo "labd already running (pid $p)" >&2; exit 1; }
  rm -rf "$SPOOL/lock"
fi
mkdir "$SPOOL/lock" 2>/dev/null && echo $$ > "$SPOOL/lock/pid" || exit 1
trap 'rm -rf "$SPOOL/lock"' EXIT
log "labd up: $NR rules, tick=${TICK}s, conf=$CONF, pid=$$"
while :; do
  parse; PEND=""                                       # hot-reload: config edits land next tick
  for i in $(by_order); do [ "${RTYPE[i]}" = action ] && do_action "$i"; done
  [ -n "$PEND" ] && fire "${PEND%%|*}" "${PEND#*|}"    # exhaust events fire only after pass 1 completes
  for i in $(by_order); do [ "${RTYPE[i]}" = event ] && do_event "$i"; done
  rm -f "$SPOOL"/cur.*.stage                           # uncommitted detector stages die with the tick
  [ $(( tick % 10 )) -eq 0 ] && log "tick $tick"       # heartbeat for the cron backstop's liveness read
  [ "$MODE" = --once ] && exit 0
  sleep "$TICK"; tick=$((tick+1))
done
