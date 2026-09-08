set -u
# source only the helper, not the dispatch
eval "$(sed -n '/^grok_capacity_verdict() {/,/^}/p' ~/.claude-grok/bin/grok-run)"
fail=0
check() { got=$(grok_capacity_verdict "$1"); [[ "$got" == "$2" ]] || { echo "FAIL: [$1] -> $got, want $2"; fail=1; }; }
check 'Error: Internal error: { "message": "API error (status 402 Payment Required): Grok Build usage balance exhausted", "http_status": 402 }' EXHAUSTED
check 'API error (status 401 Unauthorized)' UNAUTHENTICATED
check 'status 429 rate limit exceeded' RATE_LIMITED
check '' OK
check 'some unrelated warning on stderr' UNKNOWN
[[ $fail -eq 0 ]] && echo "classifier PASS (5/5)" || { echo "classifier FAILED"; exit 1; }
