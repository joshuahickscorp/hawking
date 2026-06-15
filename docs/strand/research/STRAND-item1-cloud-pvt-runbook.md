# STRAND-PVT cloud runbook — arm the 2-bit 7B selective-PV (down_proj-only)

_Ready-to-execute pod runbook. Authored 2026-06-13 on the Apple dev box (READ-ONLY analysis;
NO local compile — the box has a live PV run PID 28690 holding ~15/18 GB). Every command below
runs ON THE POD or FROM the dev box over SSH; **nothing here runs heavy compute on the local
box.** Target: drive the q2 7B loss-tax from **0.324 → ≤0.15** (PPL 6.629 bf16 anchor →
target ≤ **7.70**) while the determinism moat stays bit-identical._

---

## 0. What this experiment is (and how it differs from `cloud-selective-pv.sh`)

This runbook arms **STRAND-PVT down_proj-only PV** — the headline 2-bit 7B quality experiment from
`research/pv-scale-plan.md` (Part 2/3). It is a **PV-Tuning V-step on the trellis**: the trainable
subset (down_proj's 28 QuantLinears, 1.901 B params, 29.1 % of projection) re-learns through the
**real Rust STRAND encoder recon**; everything else freezes at its requant recon. Goal = land near
AQLM/QuIP# 2-bit (≈ 6.9 PPL / +35 %) on a **single A100-80, ~$6/arm**, with the determinism moat
intact.

**It deliberately diverges from `scripts/cloud-selective-pv.sh` on two axes — do not conflate them:**

| axis | `cloud-selective-pv.sh` (existing) | THIS runbook (Item 1, STRAND-PVT) |
|---|---|---|
| PV scope | class-level RED `v_proj\|up_proj\|gate_proj` (5.7 B trainable) | **`down_proj` only** (1.901 B trainable) |
| thesis | KL-output-damage classes ride at 4-bit + PV | **down_proj concentration** prior (3-bit `mp_light` lever) |
| deployment quant | RED@4 via `mp-kl-routed.json` (mixed precision) | **flat q2** everywhere (uniform 2-bit; the honest PVT number) |
| gate | requires `pv-kl-routed.json == PROMOTE_CLOUD` | **on-pod 0.5B down_proj-vs-full A/B** (§D), self-contained |
| shadow | fp32 (OOMs at 32B) | **bf16 frozen shadow + 8-bit Adam** (the §136 landmine fix) |

Because the scope and gate differ, **this runbook drives `scripts/strand-qat.py` directly** (the
`--pv-tensors down_proj` path that is already wired) rather than invoking `cloud-selective-pv.sh`.
Reuse `cloud-selective-pv.sh` only if/when you also want the RED@4 mixed-precision deployment number
as a second arm — it is a separate, gate-different experiment.

> NOTE on `strand-qat.py` as shipped: the bf16-frozen-shadow + 8-bit-Adam path is the documented
> §136 remedy but is **not yet a flag** in `strand-qat.py` (today the QuantLinear shadow is fp32 —
> `nn.Parameter(...).float()` at line 65, AdamW is stock fp32 at line 576). For 7B down_proj-only
> the **fp32 shadow FITS** (47.6 GB of 80 GB per `pv-scale-plan.md`), so the default path is safe
> for the headline arm. The bf16-shadow + bitsandbytes 8-bit-Adam wiring is **required only for the
> 32B leg** and is captured as a code task in §E note + §H. Run 7B with the stock fp32 shadow; do
> NOT block the 7B arm on the 8-bit-Adam wiring.

---

## Conventions (fill these in before running)

The pod was **migrated**; the owner will supply the new TCP endpoint. Set these once at the top of
your shell session (replace `<TCP_IP>` / `<TCP_PORT>`; the old pod was `213.192.2.110:40078` per
`conductor.sh` / `cloud-selective-pv.sh` — the IP/port rotate, so **do not assume**):

```bash
export POD=<TCP_IP>
export PORT=<TCP_PORT>
export KEY=~/.ssh/id_ed25519
export SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o IdentitiesOnly=yes -i $KEY -p $PORT root@$POD"
export SCP="scp -o BatchMode=yes -o IdentitiesOnly=yes -i $KEY -P $PORT"
```

All `$SSH ...` / `$SCP ...` lines below run **from the dev box** (network only, zero local compute).
`$SSH 'cmd'` lines execute `cmd` **on the pod**.

---

## (a) Verify pod reachable + GPU + /workspace free space

```bash
# reachability + identity (one round-trip)
$SSH 'echo POD_OK; hostname; uptime'

# GPU present, idle, and the right card (want an A100-80; >=80 GB for fp32-shadow 7B down-only)
$SSH 'nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv'

# /workspace volume free space — need headroom for: 7B bf16 weights (~15 GB) + recon dumps
# + tuned HF dir + recon dir. Want >= ~50 GB free (sprint reported ~68 GB free pre-flight).
$SSH 'df -h /workspace; echo "---"; du -sh /workspace/strand 2>/dev/null; du -sh /workspace/.hf 2>/dev/null'

# CUDA torch sanity (the harness needs torch+cuda + transformers + datasets + safetensors)
$SSH 'cd /workspace/strand && python3 -c "import torch,transformers,datasets,safetensors; print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available(),torch.cuda.get_device_name(0))"'
```

**GO criteria:** `POD_OK` prints; `nvidia-smi` shows an A100-80 (>=80 GB total) at ~0 % util;
`/workspace` has >= ~50 GB free; `torch.cuda.is_available()` is `True`. If the card is an A100-40
or smaller, you MUST use the 8-bit-Adam path (see §H) — do **not** launch fp32-shadow 7B on <80 GB.

---

## (b) Rebuild the on-pod `quantize-model` binary (it was broken at 937 bytes) + verify

The sprint pre-flight found `target/release/quantize-model` **broken at 937 bytes** (a truncated/
stale artifact). Rebuild it cleanly on the pod and prove it actually quantizes a tensor.

```bash
# 1) sync the checkout to the branch that has the PV + quant code (sub4bit-innovation per
#    runpod-bootstrap.sh BRANCH default). Confirm the branch with the owner if unsure.
$SSH 'cd /workspace/strand && git fetch -q origin && git status -sb | head -1'

# 2) toolchain on the volume (idempotent; runpod-bootstrap.sh installs here if missing)
$SSH 'export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup; \
      export PATH=$CARGO_HOME/bin:$PATH; cargo --version || \
      (curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path)'

# 3) FORCE a clean rebuild of the binary (delete the 937-byte artifact first so a stale
#    cache cannot resurrect it). This is the exact build line every script uses.
$SSH 'cd /workspace/strand && export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup; \
      export PATH=$CARGO_HOME/bin:$PATH; \
      rm -f target/release/quantize-model; \
      cargo build --release -p strand-quant --bin quantize-model 2>&1 | tail -4'

# 4) sanity the artifact: it must be a real ELF binary, MUCH larger than 937 bytes.
$SSH 'cd /workspace/strand && ls -l target/release/quantize-model && file target/release/quantize-model && target/release/quantize-model --help 2>&1 | head -5'
```

**Functional smoke on a TINY tensor (proves it encodes, not just links):**

```bash
$SSH 'cd /workspace/strand && export PATH=/workspace/.cargo/bin:$PATH && \
python3 - <<PY
import torch
from safetensors.torch import save_file
# one small projection-shaped tensor; q2 needs in-dim divisible by the RHT block — 512x512 is safe
w = torch.randn(512, 512, dtype=torch.bfloat16)
save_file({"model.layers.0.mlp.down_proj.weight": w}, "/workspace/_smoke_in.safetensors", metadata={"format":"pt"})
print("smoke tensor written")
PY
target/release/quantize-model --in /workspace/_smoke_in.safetensors --out /workspace/_smoke_out.safetensors \
  --bits 2 --l 12 --outlier-channel 1 --threads 8 2>&1 | tail -8
echo "--- sidecar ---"; python3 -c "import json;print(json.load(open(\"/workspace/_smoke_out.safetensors.json\"))[\"aggregate\"])"
ls -l /workspace/_smoke_out.safetensors*'
```

**GO criteria:** binary is a multi-MB ELF; `--help` prints flags; the smoke produces
`/workspace/_smoke_out.safetensors` + a `.json` sidecar whose `aggregate.effective_bpw` is
≈ 2.0–2.3. If the build fails, capture the last 40 lines of the cargo output and STOP — every
downstream step needs this binary. Clean up: `$SSH 'rm -f /workspace/_smoke_*'`.

---

## (c) Ready Qwen2.5-7B weight shards on /workspace (download + verify)

There were **0 weight shards** at `scratch/qwen-7b` pre-flight. Download the **base** model (NOT
-Instruct; the bf16 PPL anchor 6.629 and the whole PTQ canon are on `Qwen/Qwen2.5-7B` base).

```bash
# download to the canonical path the harness expects (scratch/qwen-7b under the repo).
# HF cache lives on the volume so a pod restart never re-downloads.
$SSH 'cd /workspace/strand && export HF_HOME=/workspace/.hf HF_HUB_DISABLE_XET=1; \
      [ -f /workspace/HF_TOKEN ] && export HF_TOKEN=$(cat /workspace/HF_TOKEN); \
      bash scripts/download-model.sh Qwen/Qwen2.5-7B --out scratch/qwen-7b'
```

**Verify completeness (all index-listed shards present, loadable, correct dtype):**

```bash
$SSH 'cd /workspace/strand && python3 - <<PY
import json, glob, os
d="scratch/qwen-7b"
# config sanity (the measured 7B geometry: hidden 3584, intermediate 18944, 28 layers)
cfg=json.load(open(f"{d}/config.json"))
print("layers",cfg["num_hidden_layers"],"hidden",cfg["hidden_size"],"intermediate",cfg["intermediate_size"])
assert cfg["num_hidden_layers"]==28 and cfg["hidden_size"]==3584, "NOT Qwen2.5-7B base geometry"
# all shards present per the index
idx=json.load(open(f"{d}/model.safetensors.index.json"))
want=set(idx["weight_map"].values()); have={os.path.basename(p) for p in glob.glob(f"{d}/*.safetensors")}
missing=want-have
print("shards want",len(want),"have",len(have),"missing",sorted(missing))
assert not missing, f"MISSING SHARDS {missing}"
# every shard opens and a down_proj tensor reads back
from safetensors import safe_open
n=0
for s in sorted(want):
    with safe_open(f"{d}/{s}",framework="pt") as f:
        for k in f.keys():
            if "down_proj.weight" in k: t=f.get_tensor(k); n+=1; break
print("down_proj tensors readable:",n,"(expect 28)")
assert n==28, "down_proj count != 28"
print("WEIGHTS_OK")
PY'
```

**GO criteria:** prints `WEIGHTS_OK`, 28 down_proj tensors, no missing shards, base geometry
(28 layers / hidden 3584). If a license wall blocks the download, set `/workspace/HF_TOKEN` on the
pod and re-run. (32B is deferred — do NOT also download it; volume is tight. `touch
/workspace/SKIP-32B` if you reuse `cloud-selective-pv.sh` later.)

---

## (d) CHEAP DE-RISK GATE FIRST — 0.5B down_proj-only vs full-PV A/B (run on the pod)

**Run this BEFORE paying for any 7B.** This is the `pv-scale-plan.md` Part-3 concentration test:
does training **down_proj-only** recover **≥ ~90 %** of the **full-PV** gain on the 0.5B? If yes,
the 7B down_proj-only buy is de-risked. If no, the down_proj concentration prior does NOT transfer
to 2-bit PV and the 7B path needs a wider (more expensive) scope — STOP and escalate.

Both arms are identical except `--pv-tensors` (down_proj-only) vs `--pv-tensors '.'` (full = all
wrapped). Same 300 steps, same q2 levers as the recipe in `pv-scale-plan.md` Part 1. Run on the pod
(0.5B is tiny on an A100; ~minutes/arm — this gate is cheap GPU time, not a rental tier of its own).

```bash
# 0.5B weights on the pod (small; download if absent)
$SSH 'cd /workspace/strand && export HF_HOME=/workspace/.hf HF_HUB_DISABLE_XET=1; \
      [ -f scratch/qwen-05b/config.json ] || bash scripts/download-model.sh Qwen/Qwen2.5-0.5B --out scratch/qwen-05b'

# ARM A — FULL PV (train all wrapped QuantLinears): the gain denominator
$SSH 'cd /workspace/strand && export PATH=/workspace/.cargo/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
  nohup python3 scripts/strand-qat.py --model scratch/qwen-05b --quant strand --bits 2 --l 12 \
    --steps 300 --lr 1e-4 --ctx 512 --requant-every 75 --kd --grad-checkpoint \
    --warmup-frac 0.05 --cooldown-frac 0.2 --device cuda \
    --strand-flags "--bits 2 --l 12 --outlier-channel 1 --threads 96" \
    --strand-dir scratch/qwen-05b/strand-pv-full \
    --eval-chunks 64 --eval-ctx 2048 \
    --arm-name pvt_gate_05b_full --lineage-label science \
    --out /workspace/strand-results/pvt_gate_05b_full.json \
    > /workspace/pvt_gate_full.log 2>&1 & echo "full-PV pid $!"'

# ARM B — down_proj-ONLY PV (the concentration arm). Run after A finishes (or on a 2nd GPU).
$SSH 'cd /workspace/strand && export PATH=/workspace/.cargo/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
  nohup python3 scripts/strand-qat.py --model scratch/qwen-05b --quant strand --bits 2 --l 12 \
    --steps 300 --lr 1e-4 --ctx 512 --requant-every 75 --kd --grad-checkpoint \
    --warmup-frac 0.05 --cooldown-frac 0.2 --device cuda \
    --pv-tensors down_proj \
    --strand-flags "--bits 2 --l 12 --outlier-channel 1 --threads 96" \
    --strand-dir scratch/qwen-05b/strand-pv-down \
    --eval-chunks 64 --eval-ctx 2048 \
    --arm-name pvt_gate_05b_down --lineage-label science \
    --out /workspace/strand-results/pvt_gate_05b_down.json \
    > /workspace/pvt_gate_down.log 2>&1 & echo "down-only pid $!"'

# watch
$SSH 'tail -n 30 /workspace/pvt_gate_full.log; echo "==="; tail -n 30 /workspace/pvt_gate_down.log'
```

**Compute the concentration ratio + the GO/NO-GO (run on the pod after both finish):**

```bash
$SSH 'cd /workspace/strand && python3 - <<PY
import json
F=json.load(open("/workspace/strand-results/pvt_gate_05b_full.json"))
D=json.load(open("/workspace/strand-results/pvt_gate_05b_down.json"))
# both report ppl_before (PTQ floor, identical) and ppl_after (post-PV). Gain = before-after.
b=F["ppl_before"]
gain_full=b-F["ppl_after"]; gain_down=b-D["ppl_after"]
ratio=gain_down/gain_full if gain_full>0 else 0.0
print(f"PTQ-floor before={b:.3f}")
print(f"full-PV  after={F[\"ppl_after\"]:.3f}  gain={gain_full:.3f}")
print(f"down-only after={D[\"ppl_after\"]:.3f}  gain={gain_down:.3f}")
print(f"CONCENTRATION RATIO = {ratio*100:.1f}% of full-PV gain  ->  {\"GO (>=90%)\" if ratio>=0.90 else \"NO-GO (<90%)\"}")
PY'
```

**GATE DECISION:**
- **ratio ≥ ~90 %** → **GO.** The down_proj concentration transfers to 2-bit PV. Proceed to §E
  (7B down_proj-only, ~$6).
- **ratio < ~90 %** → **NO-GO on down_proj-only.** Per `pv-scale-plan.md` §GO/NO-GO, the 7B path
  then needs FFN-only (~106 GB → 2× A100-80, the ~$300 tier) — **do not silently widen scope.**
  STOP, record the ratio, and escalate to the owner to justify the wider buy against the AQLM
  margin. (Optionally, the existing `cloud-selective-pv.sh` RED `v\|up\|gate` class recipe is the
  pre-built wider-scope path, but it carries its own `pv-kl-routed.json` gate — a separate decision.)

---

## (e) ON GO — launch the 7B down_proj-only PV (~$6, single A100-80)

Stock fp32 shadow (FITS in 47.6 GB on the A100-80 per `pv-scale-plan.md`), KD on (cost-sensitive +
**non-gating** here — KD improves the result and is not in the GO criterion), segmented requant via
`--requant-every`, WSD schedule. Deployment quant in the loop and at the end is **flat q2** (the
honest PVT number — NO RED@4 mixed precision for this arm).

```bash
# launch DETACHED on the pod (survives SSH drop; ~2-4 h, ~$6 on A100-80 @ $1.5/hr)
$SSH 'cd /workspace/strand && \
  export PATH=/workspace/.cargo/bin:$PATH \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1; \
  [ -f /workspace/HF_TOKEN ] && export HF_TOKEN=$(cat /workspace/HF_TOKEN); \
  nohup python3 scripts/strand-qat.py \
    --model scratch/qwen-7b --quant strand --bits 2 --l 12 \
    --steps 300 --lr 1e-4 --ctx 512 \
    --requant-every 75 \
    --pv-tensors down_proj \
    --kd --grad-checkpoint \
    --warmup-frac 0.05 --cooldown-frac 0.2 \
    --device cuda \
    --strand-flags "--bits 2 --l 12 --outlier-channel 1 --threads 96" \
    --strand-dir scratch/qwen-7b/strand-pvt-down \
    --eval-chunks 64 --eval-ctx 2048 \
    --arm-name pvt_selpv_7b_down --lineage-label science \
    --save-hf scratch/qwen-7b/pvt-down-hf \
    --out /workspace/strand-results/pv_selpv_7b.json \
    > /workspace/pvt_7b.log 2>&1 & echo "7B PVT pid $!"'

# WATCH (the loss trajectory + the requant bpw + the frozen-invariant asserts)
$SSH 'tail -n 50 /workspace/pvt_7b.log'
# the harness self-asserts the moat each requant boundary; confirm these lines appear:
$SSH 'grep -E "frozen-verify|selective-PV: captured|trained shadows moved|requant" /workspace/pvt_7b.log | tail -20'
```

**KD-cache (optional cost lever, non-gating):** the teacher is frozen and chunks repeat across the
4 requant segments, so caching the top-k teacher target skips the teacher forward on later passes.
It changes the KD loss to the bucketed top-k approximation (exact derivation in `strand-qat.py`
`kd_loss_sparse`), so it is a **science flag** — adopt only if you want the wall-clock/$ saving and
accept |dPPL| < 0.5 %. To enable, add `--kd-cache /workspace/strand-results/kdcache_7b`.

> **fp32 shadow is correct here — do NOT add 8-bit-Adam for 7B down-only.** It fits (47.6 GB) and
> the fp32 path is the proven one. The bf16-shadow + 8-bit-Adam path is the **32B** landmine fix
> (fp32 shadow → ~238 GB → OOM at 32B); that requires a code change to `strand-qat.py` (§H) and is
> out of scope for the 7B headline arm.

**The canonical-harness PPL (recommended second confirmation):** the in-run `ppl_after` uses the
harness's own eval. To get the **canon WikiText-2 PPL** (same shim as the bf16 6.629 anchor) of the
tuned-then-STRAND-quantized weights, re-quantize the saved HF dir flat-q2 and eval through the canon
path. The cleanest reuse is `strand-7b-ppl.sh` on the tuned dir (it forwards `--bits`/
`--outlier-channel`, which is all flat-q2 needs):

```bash
$SSH 'cd /workspace/strand && export PATH=/workspace/.cargo/bin:$PATH; \
  bash scripts/strand-7b-ppl.sh scratch/qwen-7b/pvt-down-hf \
    --bits 2 --outlier-channel 1 --label pvt_selpv_7b_canon \
    --device cuda --limit-chunks 64 --resume --out-dir scratch/qwen-7b/reopen/pvt_canon \
    > /workspace/pvt_7b_canon.log 2>&1 & echo "canon-eval pid $!"'
```

---

## (f) Where the result lands + how it is monitored after the wave

**Result artifacts (on the pod, `/workspace/strand-results/`):**
- `pv_selpv_7b.json` — the primary result. `strand-qat.py` writes `ppl_before` (PTQ floor),
  `ppl_after` (post-PV), `pv_tensors`, `pv_count`. **This is the headline file the success target
  names.**
- `scratch/qwen-7b/pvt-down-hf/` — the drop-in tuned HF dir (bf16 shadow weights), the deployment
  artifact the real STRAND encoder ships.
- `research/pv-lineage.jsonl` (append-only) — the full provenance record (every requant's bpw + secs,
  the frozen-invariant + trained-moved counts, the exact argv).
- `scratch/qwen-7b/reopen/pvt_canon/ppl_*.json` — the canon-harness confirmation PPL (if §E canon
  step run).

**Score it (loss-tax against the 6.629 bf16 anchor) — run on the pod or after mirroring down:**
```bash
$SSH 'cd /workspace/strand && python3 scripts/promote.py /workspace/strand-results/pv_selpv_7b.json --model qwen-7b'
```
`promote.py` computes `loss_tax_nats = ln(PPL / 6.629)` automatically (anchor at promote.py:37).
**SUCCESS = loss_tax ≤ 0.15** ⇔ **PPL ≤ 7.70** (from the q2 base tax 0.324 ⇔ PPL ~9.16).
Realistic landing per the sprint: 7B tax **~0.18–0.28** — i.e. this arm is expected to *move toward*
0.15 and may need the canon confirmation + a 2nd arm (lr sweep / cooldown on-off) to close it. The
A/B-clean knob is `--cooldown-frac` (0.2 vs 0.0); `--lr` (1e-4 vs e.g. 5e-5) is the other cheap arm.

**Mirror results down to the dev box (network only — safe; no local compute):**
```bash
mkdir -p scratch/pod-results
$SCP "root@$POD:/workspace/strand-results/pv_selpv_7b.json" scratch/pod-results/
$SCP "root@$POD:/workspace/strand-results/pvt_gate_05b_*.json" scratch/pod-results/
# bill locally is fine (promote.py is tiny / not memory-heavy):
python3 scripts/promote.py scratch/pod-results/pv_selpv_7b.json --model qwen-7b --dry-run
```

**Monitoring after the wave:** `scripts/conductor.sh` already polls the pod every 10th tick
(`POD_MILESTONE` on new `ppl_*.json`, `POD_MEM` guard, `POD_UNREACHABLE` after 3 misses) and mirrors
`/workspace/strand-results/*.json` to `scratch/pod-results/`. **Update conductor's hardcoded SSH
endpoint to the new `<TCP_IP>:<TCP_PORT>`** (currently `213.192.2.110:40078` at `conductor.sh:72`,
`:301-314`) so its pod tick reaches the migrated pod — otherwise it will emit `POD_UNREACHABLE`.
Once mirrored, the new `pv_selpv_7b.json` lands in `scratch/pod-results/` and the milestone fires.

---

## Determinism-moat invariants that MUST hold (the non-negotiables)

PV is a **pre-encode weight change only**. The decode path and the attestation must remain
bit-identical. Concretely:

1. **Frozen integer Q12 LUT decode stays bit-identical.** PV never touches the decoder. The
   deployment recon is produced by the SAME `quantize-model` encoder + the SAME frozen integer Q12
   LUT, so decode is byte-for-byte identical to any other STRAND archive at the same config. (The
   commit-log determinism moat: "decode integer LUT bit-exact everywhere.") The §B smoke and §E
   in-loop requant both use this one binary — there is exactly one encoder/decoder in play.

2. **PV changes weights PRE-ENCODE only.** The trainable down_proj shadows are *quantized through
   the real encoder* every requant; the shipped artifact is the encoder's output, not a float
   shadow. Nothing post-encode (rANS stream, side-info, decode kernel) is altered by training. PV
   is the V-step that picks *which* integers get coded; the coding itself is unchanged.

3. **Frozen-tensor invariant (harness-asserted every requant boundary).** `strand-qat.py`
   `pv_verify_frozen` asserts that for every NON-down_proj tensor, both the shadow-weight digest AND
   the delta-forward `base+w == recon` digest are hash-equal to their init-requant values — i.e. the
   28-of-168 trainable scope did not leak. **Watch for the `frozen-verify ... OK` line at each
   boundary; a `FROZEN-INVARIANT VIOLATED` assert means the freeze scope is buggy — abort the run.**

4. **Trained-set actually moved (no silent no-op).** The end-of-run assert
   `pv_moved == len(pv_trained0)` requires all 28 trained down_proj shadows to have changed. A run
   where they did not move is a freeze-scope bug, not a result.

5. **Deployment config is flat q2 for this arm — and it is the SAME config in-loop and at deploy.**
   `--strand-flags "--bits 2 --l 12 --outlier-channel 1"` is used by the in-loop requant; the canon
   eval (§E) re-quantizes the tuned dir at the SAME `--bits 2 --outlier-channel 1`. Train-through-
   what-you-ship: the forward IS the deployment recon (proxy-transfer is DEAD per will.md §4). Do
   NOT introduce `--rung-config` / RED@4 for this arm — that would be a different (mixed-precision)
   experiment and a different number.

6. **outlier-channel side-channel parity.** `--outlier-channel 1` is the only live PTQ lever and is
   identical in-loop and at deploy. The outlier wire is EOF-chained and part of the recon both
   times, so it does not perturb determinism — but it MUST be present in BOTH the training requant
   and the final deploy quant (it is, above).

7. **(If/when wired) attestation must seal every new stream.** If a later arm stacks DBIA (de-bias)
   or C2 side-info, those MUST be added to `descriptor_digest` + read in `verify_archive` (mirror
   OUTL) or a dropped/tampered correction passes SPRV verification silently and breaks the moat
   (sprint §138 / audit wave wxgehdvom). **This 7B flat-q2 down_proj arm does NOT use DBIA/C2**, so
   it is unaffected — but do not add them without the seal.

8. **No FMA / IEEE-754 f32 reproducibility caveat (inherited, not introduced by PV).** Cross-device
   bit-identity of the RHT rests on f32 no-FMA for odd-power block widths (commit 1e5c520:
   even-power-of-2 block h is bit-exact; odd-power ~1e-6 approximate). PV does not change this — it
   operates on the same RHT — but the canon eval should run on the **same device class** used to
   produce the shipped recon to keep the asserted (not proven) cross-device identity. The integer
   decode LUT is bit-exact everywhere regardless.

---

## §H — DEFERRED: the bf16-shadow + 8-bit-Adam wiring (REQUIRED before the 32B leg, NOT for 7B)

The §136 landmine: at 32B the fp32 frozen PV shadow (8 B/param) is ~238 GB → OOM. The fix is a
**bf16 frozen shadow + bitsandbytes 8-bit Adam** (drops trainable state from 18 → 12 B/param). This
is **a code change to `strand-qat.py`**, not just a flag — today the QuantLinear shadow is fp32
(`nn.Parameter(...).float()` at ~line 65) and the optimizer is stock `torch.optim.AdamW` (line 576).
**Pod code tasks (do these before any 32B arm; 7B does NOT need them):**
- add a `--shadow-dtype {fp32,bf16}` flag → cast the trainable `self.weight` accordingly (keep the
  STE math; the recon dump already casts to bf16 at requant);
- add a `--optim {adamw,adamw8bit}` flag → `import bitsandbytes as bnb; bnb.optim.AdamW8bit(...)`
  when selected (verify `bitsandbytes` is pip-installable on the pod's CUDA image);
- A/B the 8-bit-Adam path on the 0.5B (kill bar |dPPL| < 0.5 %) before trusting it at 32B.
Then 32B runs via `cloud-selective-pv.sh` (which has `SKIP-32B`/sentinel plumbing) OR a 32B
down_proj-only `strand-qat.py` invocation mirroring §E with `--shadow-dtype bf16 --optim adamw8bit`.

---

## One-screen execution order

1. Set `POD/PORT/KEY/SSH/SCP` with the owner's new endpoint.
2. §A reachability + GPU(A100-80) + `/workspace` >=50 GB + torch.cuda.
3. §B rebuild `quantize-model` (rm the 937-byte stale first) + tiny-tensor smoke → bpw ~2.0–2.3.
4. §C download + verify Qwen2.5-7B **base** → `WEIGHTS_OK` (28 down_proj).
5. §D 0.5B down-only vs full A/B → **GO iff ratio ≥ ~90 %**; else STOP/escalate.
6. §E on GO: launch 7B down_proj-only PV detached (fp32 shadow, KD on, segmented requant); watch the
   `frozen-verify ... OK` asserts; optional canon eval.
7. §F `promote.py --model qwen-7b` → **loss_tax ≤ 0.15 (PPL ≤ 7.70)**; mirror down; point
   `conductor.sh` at the new endpoint for ongoing monitoring.
8. Moat invariants 1–8 hold throughout (decode LUT bit-identical; PV is pre-encode only).
