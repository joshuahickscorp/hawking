# generation_golden_v1

Model-free bit-exact generation golden.

- `logits.json` — fixed T×V logit matrix
- `seeded_sampler_greedy.json` — frozen greedy argmax token sequence

Verify: `python3.12 tools/verify/blackbox.py --only-runnable` includes BC-GENERATION-020.
