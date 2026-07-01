import { describe, it, expect } from "vitest";
import { useStore } from "./store";

// Test fixtures: `apply` takes a UiEvent; these are loose literals (the test file is excluded from the
// production tsc, and vitest transpiles without typecheck).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const apply = (kind: any, session_id: string | null = "ses_x", seq = 1) =>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (useStore.getState().apply as any)({ seq, session_id, kind });

describe("store.apply", () => {
  it("tracks the active session and runtime status from events", () => {
    apply({ type: "runtime_status", data: { status: "ready", detail: null } }, "ses_live");
    expect(useStore.getState().sessionId).toBe("ses_live");
    expect(useStore.getState().runtimeStatus).toBe("ready");
  });

  it("coalesces streamed tokens into one assistant message", () => {
    apply({ type: "token_batch", data: { stream_id: "s1", text: "Hello " } }, "ses_live", 2);
    apply({ type: "token_batch", data: { stream_id: "s1", text: "world" } }, "ses_live", 3);
    const msgs = useStore.getState().messages;
    const last = msgs[msgs.length - 1];
    expect(last.role).toBe("assistant");
    expect(last.text).toBe("Hello world");
  });

  it("folds a context_manifest projection into the live manifest", () => {
    apply(
      { type: "projection_patch", data: { projection: "context_manifest", patch: { ctx_len_effective: 131072, tq_multiplier: 4 } } },
      "ses_live",
      4,
    );
    expect(useStore.getState().manifest?.ctx_len_effective).toBe(131072);
    expect(useStore.getState().manifest?.tq_multiplier).toBe(4);
  });
});
