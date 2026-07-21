# Kimi K2.6 Doctor Prime

- State: **MONITOR** (RUNNING)
- Running: PID `80723`, process group `80723`
- Lease live: `True`; heartbeat age: `10.750636100769043` s
- Source: `KIMI_DOWNLOAD_COMPLETE`
- Progress: P0 sealed; P1/P5 F1 bracket preflight advancing
- Download throughput/state: complete
- ETA: done
- Free disk: 48.1 GiB
- Available memory estimate: 46.4 GiB
- Swap used: 1.0 GiB (16 GiB guard)
- Current file/layer: `P1/P5 F1 representation bracket preflight advancing`
- Sealed/failed checkpoints: 85/0
- Best candidate / complete BPW: `P0_OFFICIAL_PARENT_REFERENCE` / `None`
- Next action: monitor the next advancing experiment
- Exact blocker: `none`
- Last Telegram: `{'checkpoint_id': 'controller:started', 'delivered_at': '2026-07-21T05:53:32Z', 'seal_sha256': '6ecd3419688e71b92abad9f9272aa6e4ed814d7ced610fa718fe9d5baf656486'}`

Control from the Hawking repository:

```text
python3 tools/kimi_k26_campaign.py status
python3 tools/kimi_k26_campaign.py pause-after-checkpoint
python3 tools/kimi_k26_campaign.py resume
python3 tools/kimi_k26_campaign.py stop
```
