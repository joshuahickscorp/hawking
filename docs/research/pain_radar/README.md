# Pain Radar

This directory tracks public runtime pain that can become Dismantle/Hawking
benchmarks.

Files:

- `ledger.jsonl`: normalized public issue/link ledger.
- `clusters.md`: grouped view by pain class.

Tooling:

```bash
python tools/research/pain_radar.py seed
python tools/research/pain_radar.py summarize
python tools/research/pain_radar.py fetch-github
```

`fetch-github` only reads public issue metadata. Do not add private content or
private user data to this ledger.

Every useful row should eventually become one of:

- a benchmark workload;
- a skipped reproduction with a reason;
- a measured fix in `docs/reports/apple_serving_pain_fixes.md`.
