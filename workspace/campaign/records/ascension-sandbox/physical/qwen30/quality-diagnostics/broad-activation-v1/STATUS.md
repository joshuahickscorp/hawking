# Broad activation capture — status

## Blocker (honest)

Metal L0 capture was **attempted** from this sandboxed executor and refused:

```
qwen30 current HCLI L0 route capture refused: metal: no Metal-capable GPU
```

Production server `:18430` (instance `qwen30-native-95951-…`) remained ready after the attempt.
No exclusive GPU lease was taken. Incomplete run directory was removed so a clean re-run works.

**This work needs an unsandboxed / `gate` Metal pass, serialized against the live server if dual residency is unsafe.**

## Exact capture command (owner-run, serialized)

See `RUN_CAPTURE.command.txt`. Reproduced:

```bash
/Users/scammermike/.claude-grok/worktrees/q30-broader-capture-20260809-154213/workspace/ops/build/rust/debug/examples/ascension_qwen30_current_hcli_layer0_route_capture \
  --manifest /Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen30/complete-gravity/QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json \
  --expected-manifest-seal-sha256 3321a99d719e70499663b7bfebe14dd6c732bfc533bb05b9277eb398e44d6357 \
  --expected-source-audit-seal-sha256 00ed3e495416c2cbafbcdb7800528e15f243b1a13f5f4af13240109c8fc69f7b \
  --expected-source-revision b2cff646eb4bb1d68355c01b18ae02e7cf42d120 \
  --input-json /Users/scammermike/.claude-grok/worktrees/q30-broader-capture-20260809-154213/workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-diagnostics/broad-activation-v1/requests/QWEN30_BROAD_ACTIVATION_L0_ROUTE_CAPTURE_INPUT_901a24bdcfc6c1d2.json \
  --output-dir /Users/scammermike/.claude-grok/worktrees/q30-broader-capture-20260809-154213/workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-diagnostics/broad-activation-v1/runs/8c5ff2d8490f716b_94c3f75f83dce25a \
  --max-seq-len 2048
```

After capture:

```bash
lab/operators/q30_broad_activation_after_capture.sh \
  /Users/scammermike/.claude-grok/worktrees/q30-broader-capture-20260809-154213/workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-diagnostics/broad-activation-v1/runs/8c5ff2d8490f716b_94c3f75f83dce25a
```

(Use the actual run directory name if binary sha differs after rebuild.)

## Prepared capture input (done)

| field | value |
|---|---|
| probes | **32** |
| total tokens | **3929** (vs ~1115 three-prompt HCLI) |
| domains | code 7, prose 5, structured 5, multi_turn 3, long_context 2, math 3, instruction 3, dialogue 1, list 1, mixed 2 |
| token range | 22–1459 (mean 122.8) |
| schema | `hawking.ascension.qwen30_broad_activation_layer0_route_capture_input.v1` |
| claim | diagnostic activation pricing only; not HCLI/coherence/TPS/capability |

## Null-first: baseline three-prompt capture (done BEFORE family re-price)

| metric | value |
|---|---:|
| mean null (high-hit ge 200) | **0.957** |
| min / max null | 0.891 / 0.993 |
| prior lane mean null | **0.942** |
| materially below 0.942? | **no** (this is the baseline) |

Source: `null-first/NULL_BASELINE_THREE_PROMPT.json`

## Null-first: broad capture

**Not yet measured** — blocked on Metal capture.

## Family re-price

**Not yet measured** — blocked on broad capture. Existing probe left unmodified:
`lab/operators/q30_activation_aware_family_probe.py`.

## Claim boundary

- Diagnostic capture preparation only until Metal run completes
- Live production server not contacted for capture; health checked only
- Three-probe HCLI quality schema still requires exactly three protected probes
- Broad schema is additive, min 12 probes, self-declares non-capability
