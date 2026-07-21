# Kimi K2.6 Doctor Prime

- State: **MONITOR** (RUNNING)
- Running: PID `76038`, process group `76038`
- Lease live: `True`; heartbeat age: `21.402403116226196` s
- Source: `KIMI_DOWNLOAD_COMPLETE`
- Progress: P0 sealed; P1/P5 F1 bracket preflight advancing
- Download throughput/state: complete
- ETA: done
- Free disk: 48.1 GiB
- Available memory estimate: 45.4 GiB
- Swap used: 1.0 GiB (16 GiB guard)
- Current file/layer: `P1/P5 F1 representation bracket preflight advancing`
- Sealed/failed checkpoints: 84/0
- Best candidate / complete BPW: `P0_OFFICIAL_PARENT_REFERENCE` / `None`
- Next action: monitor the next advancing experiment
- Exact blocker: `none`
- Last Telegram: `{'checkpoint_id': 'controller:started', 'delivered_at': '2026-07-21T05:51:21Z', 'seal_sha256': '7d82495bb361acbd70c28de3ce2c7eaa3d4496e285f33abc209e4336c0dfd1a6'}`

Control from the Hawking repository:

```text
python3 tools/kimi_k26_campaign.py status
python3 tools/kimi_k26_campaign.py pause-after-checkpoint
python3 tools/kimi_k26_campaign.py resume
python3 tools/kimi_k26_campaign.py stop
```
