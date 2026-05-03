#!/usr/bin/env bash
# propose-patch.sh — auto-apply self-improvement edits, write the unified
# diff to an evidence path so the audit trail records what changed.
#
# Per-user policy (haul-2 plan): apply mode is the only mode. The
# patches are small, deterministic, RAM-light fixes whose entire diff
# fits on screen — review happens by reading the applied.patch artifact
# in evidence after the haul.
#
# Usage:
#   ./tools/haul/propose-patch.sh apply <kind> <patch-out> [args...]
#
# Kinds (each is a named handler below):
#   expand-baseline-probe-state
#       Fix the `degraded\nunknown` log artifact in
#       `tools/haul/expand-baseline.sh`'s probe_state(). Capture the
#       probe pipeline's stdout into a local var, fall back to "unknown"
#       only when the var is empty (vs. tacking it onto a successful
#       output via `|| echo unknown` under `set -o pipefail`).
#
#   cargo-fmt
#       Run `cargo fmt --all` against the workspace.
#
#   unsafe-doc-comment <relpath>
#       Add a `# Safety` doc-comment block to the unsafe fn at
#       <relpath>:122 if it lacks one (currently only:
#       crates/dismantle-core/src/metal/mod.rs's new_buffer_no_copy).
#
# Exit codes:
#   0  patch applied (or already in desired state — idempotent no-op)
#   1  patch logic failed (couldn't find the target lines, etc.)
#   2  unknown <kind>
#
# The handler always writes <patch-out>; an empty patch (no-op) is
# written as the literal string "no-op" with a trailing newline so the
# file's existence is meaningful (= "this gate ran and considered the
# change").

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[propose-patch %s] %s\n' "$(ts)" "$*" >&2; }

# Compute and write a unified diff for `target` vs the snapshot at
# `target.before-patch`. If files are identical, write "no-op".
emit_diff() {
    local target="$1" patch_out="$2" before="$1.before-patch"
    if ! [[ -f "$before" ]]; then
        log "missing snapshot $before"
        return 1
    fi
    if cmp -s "$target" "$before"; then
        printf 'no-op\n' >"$patch_out"
    else
        diff -u "$before" "$target" >"$patch_out" || true
    fi
    rm -f "$before"
    return 0
}

# ---------- handler: expand-baseline-probe-state ----------------------

handle_expand_baseline_probe_state() {
    local target="$REPO_ROOT/tools/haul/expand-baseline.sh"
    [[ -f "$target" ]] || { log "missing $target"; return 1; }
    cp "$target" "$target.before-patch"

    # The replacement is bash, not python — keep it self-contained.
    # We rewrite only the probe_state() function body. Use python3 for
    # robust regex-with-multiline replace.
    # Build the replacement function as a literal in Python (no embedded
    # bash quote escapes — those land in the file unchanged through
    # write_text). The bash `'\''`-style apostrophes are written as the
    # literal string r"'\''" so the file contains exactly four chars.
    PROPOSE_TARGET="$target" python3 - <<'PY'
import os, re, sys, pathlib
target = pathlib.Path(os.environ["PROPOSE_TARGET"])
src = target.read_text()

SQ = r"'\''"
new_fn = f'''probe_state() {{
    # Capture once; under `set -o pipefail` the probe's nonzero exit
    # would otherwise make `|| echo unknown` fire AND keep the python
    # output, producing a literal "degraded\\nunknown" log line.
    local s
    s=$("$COEXIST" probe --json 2>/dev/null \\
        | python3 -c {SQ}import json,sys
try: print(json.load(sys.stdin).get("state","unknown"))
except Exception: print("unknown"){SQ} 2>/dev/null) || true
    [[ -z "$s" ]] && s="unknown"
    printf {SQ}%s\\n{SQ} "$s"
}}'''

pat = re.compile(r'probe_state\(\) \{.*?\n\}\n', re.DOTALL)
m = pat.search(src)
if not m:
    print("propose-patch: probe_state() not found", file=sys.stderr)
    sys.exit(1)

existing = m.group(0)
if "local s" in existing and "|| true" in existing:
    # Already in the desired post-patch shape — idempotent no-op.
    sys.exit(0)

src = src[: m.start()] + new_fn + "\n" + src[m.end() :]
target.write_text(src)
PY
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "expand-baseline-probe-state: handler failed"
        rm -f "$target.before-patch"
        return 1
    fi
    emit_diff "$target" "$1"
}

# ---------- handler: cargo-fmt ----------------------------------------

handle_cargo_fmt() {
    cd "$REPO_ROOT" || { log "cd $REPO_ROOT failed"; return 1; }
    # Snapshot the files cargo-fmt would touch (the union of `--check`
    # diff). Cheap — a few files at most, per the audit findings.
    local mods
    mods=$(cargo fmt --all -- --check 2>&1 | grep -oE 'Diff in [^ ]+' | awk '{print $3}' || true)
    if [[ -z "$mods" ]]; then
        printf 'no-op\n' >"$1"
        return 0
    fi
    local snap_dir
    snap_dir=$(mktemp -d)
    while IFS= read -r f; do
        [[ -f "$f" ]] || continue
        local rel="${f#$REPO_ROOT/}"
        mkdir -p "$snap_dir/$(dirname "$rel")"
        cp "$f" "$snap_dir/$rel"
    done <<<"$mods"

    cargo fmt --all || { log "cargo fmt failed"; rm -rf "$snap_dir"; return 1; }

    # Build a multi-file unified diff.
    : >"$1"
    while IFS= read -r f; do
        [[ -f "$f" ]] || continue
        local rel="${f#$REPO_ROOT/}"
        diff -u "$snap_dir/$rel" "$f" >>"$1" || true
    done <<<"$mods"
    rm -rf "$snap_dir"
    [[ -s "$1" ]] || printf 'no-op\n' >"$1"
}

# ---------- handler: unsafe-doc-comment -------------------------------

handle_unsafe_doc_comment() {
    # Args: <relpath>
    local rel="${PATCH_EXTRA_ARGS[0]:-}"
    [[ -n "$rel" ]] || { log "unsafe-doc-comment: missing target relpath"; return 1; }
    local target="$REPO_ROOT/$rel"
    [[ -f "$target" ]] || { log "missing $target"; return 1; }
    cp "$target" "$target.before-patch"

    # Add a `# Safety` doc paragraph immediately above the existing
    # SAFETY-style comment block on `pub unsafe fn new_buffer_no_copy`.
    # Idempotent: skip if a `# Safety` markdown header already appears
    # in the doc comments above the function.
    PROPOSE_TARGET="$target" python3 - <<'PY'
import os, re, sys, pathlib
target = pathlib.Path(os.environ["PROPOSE_TARGET"])
src = target.read_text()
fn_pat = re.compile(
    r'(?P<doc>(?:^[ \t]*///.*\n)+)(?P<sig>[ \t]*pub unsafe fn new_buffer_no_copy)',
    re.MULTILINE,
)
m = fn_pat.search(src)
if not m:
    sys.exit(0)
doc = m.group("doc")
if "# Safety" in doc:
    sys.exit(0)  # already documented
indent_match = re.match(r'([ \t]*)///', doc)
indent = indent_match.group(1) if indent_match else ""
safety_block = (
    f"{indent}///\n"
    f"{indent}/// # Safety\n"
    f"{indent}///\n"
    f"{indent}/// `bytes` must outlive every Metal command buffer that\n"
    f"{indent}/// references the returned buffer. The GGUF mmap pins the\n"
    f"{indent}/// underlying memory for the engine's lifetime, which is\n"
    f"{indent}/// where this is currently called from.\n"
)
new_doc = doc + safety_block
src = src[: m.start("doc")] + new_doc + src[m.end("doc") :]
target.write_text(src)
PY
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "unsafe-doc-comment: handler failed"
        rm -f "$target.before-patch"
        return 1
    fi
    emit_diff "$target" "$1"
}

# ---------- dispatch --------------------------------------------------

if [[ "${1:-}" != "apply" ]]; then
    log "usage: $0 apply <kind> <patch-out> [args...]"
    exit 2
fi
shift

KIND="${1:-}"
PATCH_OUT="${2:-}"
shift 2 || true
PATCH_EXTRA_ARGS=("$@")

if [[ -z "$KIND" || -z "$PATCH_OUT" ]]; then
    log "usage: $0 apply <kind> <patch-out> [args...]"
    exit 2
fi

mkdir -p "$(dirname "$PATCH_OUT")"

case "$KIND" in
    expand-baseline-probe-state)
        handle_expand_baseline_probe_state "$PATCH_OUT"
        ;;
    cargo-fmt)
        handle_cargo_fmt "$PATCH_OUT"
        ;;
    unsafe-doc-comment)
        handle_unsafe_doc_comment "$PATCH_OUT"
        ;;
    *)
        log "unknown kind: $KIND"
        exit 2
        ;;
esac

rc=$?
if [[ $rc -eq 0 ]]; then
    log "applied $KIND → $PATCH_OUT ($(wc -l <"$PATCH_OUT" | tr -d ' ') lines)"
else
    log "failed $KIND (rc=$rc)"
fi
exit $rc
