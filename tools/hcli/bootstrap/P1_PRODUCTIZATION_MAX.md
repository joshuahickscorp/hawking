# P1 HAIDER PRODUCTIZATION MAX

Governing specification for P1. P0 is frozen. Do not rebuild P0.

======================================================================
GATE ZERO — ACCEPTANCE CONTRACT
======================================================================

Command:

    python tools/haider/haider.py 1

Must produce, without Aider owning the model loop:

    1. HAIDER allocates a free TCP port.
    2. HAIDER spawns one llama-server subprocess on that port.
    3. HAIDER waits for server readiness (HTTP /health).
    4. HAIDER drives the P0 Session observation loop:
       model -> tool request -> local execution -> observation -> model.
    5. HAIDER sends a scoped-edit prompt to the model.
    6. Model returns a JSON edit object (path, old_text, new_text).
    7. HAIDER applies the edit to the working tree.
    8. HAIDER runs the P0 deterministic test suite as validation.
    9. HAIDER writes a machine-generated receipt to
       .haider/receipts/gate_zero_<timestamp>.json containing:
         - gate: "zero"
         - timestamp
         - runtime_pid
         - port
         - model
         - observation: {ok, model_turns, tool_calls, elapsed_s}
         - edit: {path, old_sha256, new_sha256, diff_lines}
         - validation: {command, exit_code, duration_ms, stdout, stderr}
         - usage: {prompt_tokens, completion_tokens, total_tokens}
         - status: "PASS" | "FAIL"
    10. HAIDER terminates the llama-server subprocess cleanly.
    11. Exit code 0 iff status == "PASS".

Evidence origin: every field in the receipt must come from a subprocess
result, filesystem hash, or HTTP response. No model-authored values.

Gate Zero is PASS when the above sequence completes with exit code 0
and the receipt file exists on disk.

======================================================================
GATE ONE — HAIDER N RUNTIME POOL
======================================================================

After Gate Zero:

    python tools/haider/haider.py N  (N in 1..4)

spawns up to N independently supervised llama-server processes, each with:
- distinct PID
- distinct port
- distinct context/KV/session domain
- distinct role prompt (PARENT, IMPLEMENTER, SCOUT, ADVERSARY, TESTER)
- independent lifecycle (start, health-check, restart, shutdown)

All runtimes may share the same model artifact on disk.

HAIDER owns:
- port allocation (no user-managed ports)
- process lifecycle
- health monitoring
- context sizing
- role assignment
- evidence exchange between runtimes
- graceful shutdown (SIGTERM, wait, SIGKILL fallback)

======================================================================
GATE TWO — SELF-HOST PROOF
======================================================================

HAIDER (running as `haider 1`) edits its own source files to implement
Gate One. The edit is validated by the deterministic test suite.
Aider is no longer in the model loop.

======================================================================
IMPLEMENTATION CONSTRAINTS
======================================================================

- Python 3.10+ for the HAIDER CLI (matches P0).
- No external dependencies beyond stdlib.
- llama-server is the only external process HAIDER spawns for inference.
- All state lives under .haider/ in the project root.
- Receipts are append-only JSON files.
- No global mutable state. Each `haider N` invocation is self-contained.
- Wall-clock budget for Gate Zero: < 120 s from invocation to receipt.

======================================================================
NON-GOALS FOR P1
======================================================================

- No browser UI.
- No virtualenv ritual.
- No manual port/context flags.
- No Aider in the model loop after Gate Zero.
- No multi-model federation.
- No distributed runtime pool (single machine only).
