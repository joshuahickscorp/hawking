# GLM-5.2 Xet autotune offline plan

Status: **PASS_OFFLINE_PLAN_BODY_NOT_READ**

- Immutable source: `zai-org/GLM-5.2@b4734de4facf877f85769a911abafc5283eab3d9`
- Body bytes read by planner: `0`
- Bounded ranges: `184` × `67108864` bytes
- Planned maximum network accounting: `23084754064` / `30133860738` bytes
- Preliminary-schedule maximum new-fetch shards (informational, not a selection cap): `23`
- Maximum resident shards in one window: `26`
- Maximum actual adjacent active+prefetch union: `45`
- Required file settings `8`, `16`, `24`, `32`, and `48` are all selectable; the winning profile is applied only during post-autotune schedule refreeze.
- A separate live executor exists at `tools/condense/glm52_xet_live.py`; this offline plan alone does not authorize execution, which remains controller/Telegram gated.
- The 10 GiB cache trial is skipped because planned refetches are zero and the pinned cache path is inert.

Plan seal: `52dd152cd603a6e094126ef7ff047383785058b1e013d664af475afb25a4a916`.
