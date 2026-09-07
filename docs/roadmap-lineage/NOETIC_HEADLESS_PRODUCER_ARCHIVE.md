# Noetic headless producer disposition

Three standalone headless producers were retired after their live authority
collapsed into the current Rust/HCLI path:

- `bandwidth_ascent.py` — historical bandwidth sweep;
- `noetic_dispatch_fusion.py` — historical dispatch-count and fusion sweep;
- `noetic_fused_subbit.py` — historical fused-subbit producer.

Their sealed receipts remain under `receipts/headless/`, including bandwidth,
dispatch-fusion, fused-subbit, frontier-adversary, and gate-adversary findings.
The frontier and gate runners were also retired because their Noetic/VisionMCP
dependencies are absent from the current HCLI path. No measured result or
negative control is promoted by this retirement; the deleted files were
parallel execution surfaces, not the evidence record.
