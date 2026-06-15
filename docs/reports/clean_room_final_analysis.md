# Clean-room final analysis — commands to run (Claude QUIT)

> Hand-off for the user's final bench. Every number below is an **absolute**
> metric, so it is only valid with **Claude Code fully quit** (a running session
> inflates dec_tps ~4–5×). Paired A/B (`ab_lever.sh`) is contamination-robust and
> can run anytime; the absolute batches below need the clean room.
>
> Branch `paradigm/exec` @ `92bcdab`. Default decode is bit-identical to the
> session start (`b480cc10faf9a8ec`) — so the baseline below is a no-regression
> check, and the levers are opt-in.

## 0. Pre-flight (anytime, even with Claude open)
```
tools/bench/clean_room_batch.sh --gates-only
```
Confirms the machine is quiet (warns on any >X% CPU hog). If it passes, quit
Claude and run the rest.

## 1. THE HEADLINE — absolute baseline tps + J/tok  (Claude QUIT)
```
tools/bench/clean_room_batch.sh
```
Prints the honest **clean decode tps** (anchor ≈ 30), **J/tok** (≈ 0.20 @ ~5 W
GPU), and the §A Q3-byte-cut proxy (re-confirms the QTIP Type-1 kill at ~24%
peak). This is the no-regression anchor for the whole session.

## 2. THE KEY OPEN QUESTION — does f16-KV win on ENERGY?  (Claude QUIT)
f16-KV is **tps-negative at depth** (paired A/B: −3 to −8% @1–2K tok — the f16
dequant costs more compute than the KV-bandwidth it saves). It is shipped as a
**footprint** lever (half the KV-cache → longer context fits in RAM). The open
question is whether the reduced DRAM traffic still wins on **J/tok**:
```
# baseline J/tok at long context
tools/bench/phase_joules.sh --tokens 1024
# f16-KV J/tok at long context
DISMANTLE_QWEN_F16_KV=1 tools/bench/phase_joules.sh --tokens 1024
```
Compare total J/tok. **f16-KV J/tok < baseline ⇒ it is a real energy lever**
(footprint + energy); **≥ baseline ⇒ footprint-only** (the dequant compute eats
the DRAM saving). Either verdict is fine — it is default-off regardless.

## 3. QUALITY gates for the non-bit-identical levers  (Claude QUIT preferred)
The Phase-1.2 quality gate (was queued; tooling now in place):
```
tools/bench/quality_oracle.sh --lever DISMANTLE_QWEN_PREDEC_F16SCALES --label f16scales
tools/bench/quality_oracle.sh --lever DISMANTLE_QWEN_F16_KV          --label f16kv --long
```
PASS ≈ high token-identical fraction + logit-cosine ≥ ~0.999 (gate envs
`PASS_IDENT_MIN` / `PASS_DRIFT_MAX` overridable). Confirms `--profile fast`
(f16-scales, the **measured +4.9%** tps win) and f16-KV stay within quality.

## 4. (Optional) paired A/B — works WITH Claude open (contamination cancels)
```
tools/bench/ab_lever.sh --cli-b "--profile fast"                       # the +4.9% win
tools/bench/ab_lever.sh --lever DISMANTLE_QWEN_F16_KV   --long-ctx      # confirms tps-neutral/neg
tools/bench/ab_lever.sh --lever DISMANTLE_QWEN_FLASH_ATTN --long-ctx    # capability (long-ctx hold)
```

## What each result decides
- **§1 baseline** ⇒ confirms no regression vs the ~30 tps / 0.20 J/tok anchor.
- **§2 f16-KV J/tok** ⇒ promotes f16-KV from "footprint-only" to "footprint + energy" (or not).
- **§3 quality** ⇒ formally clears `--profile fast` + f16-KV for opt-in shipping.
- **§4** ⇒ re-confirms the +4.9% and the long-ctx behavior anytime.
