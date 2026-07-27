/*
  state.test.ts: the Context Stack as a receipt and as a set of real commands.

  Two things a user can be lied to about here: what the panel says went into the context, and what
  its controls actually do. So this asserts the receipt is read from the REAL manifest the host
  serializes (crates/hawking-context ContextManifest), that every control names a catalog command id
  (snapshot -> checkpoint_create, fork -> fork_session, the memory controls -> the memory domain),
  and that the mock skill controls and the no-op pin_span path are gone from the surface.
*/
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";

// The transport seam, stubbed so each test reads exactly what went on the wire.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const { sent } = vi.hoisted(() => ({ sent: [] as any[] }));
vi.mock("../../ipc", () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  sendIntent: async (i: any) => {
    sent.push(i);
    return { accepted: true, event_seq: 1, message: null };
  },
  subscribeUi: () => () => {},
  callConnector: async () => null,
  TRANSPORT_KIND: "mock",
}));

import { COMMANDS } from "../../store";
import {
  asReceipt,
  excludedSources,
  includedSources,
  memoryDraft,
  memoryRows,
  modelSummary,
  notePlan,
  plan,
  receiptTotals,
  runPlan,
  sessionScope,
  sourceKindLabel,
  staleWarnings,
  type Receipt,
} from "./state";

/* A manifest in the shape hide-backend really serializes: retained spans with blake3 ids, a
   compat-instructions span carrying its CLAUDE.md derivation, drops with typed reasons, an open
   conflict, and the live ceiling block. */
const REAL: Receipt = {
  model: {
    id: "qwen3-coder",
    arch: "qwen",
    ctx_len_native: 32768,
    ctx_len_effective: 131072,
    tokenizer_sig: "tok_sig_1",
  },
  budget: { total: 16384, used: 9000, free: 7384 },
  model_context_tokens: 16384,
  used_tokens: 9000,
  kv: { prefix_reuse_tokens: 512, bank_hit: true },
  retained: [
    {
      id: "blake3:aaa",
      source: "system",
      title: "Repository instructions (Claude Code compat)",
      token_count: 400,
      score: 1,
      pin: "never_evict",
      provenance: {
        source: "compat_instructions",
        trust: "trusted",
        labels: ["instructions", "claude-md-compat"],
        derived_from: ["CLAUDE.md", "docs/AGENTS.md"],
      },
    },
    {
      id: "blake3:bbb",
      source: "code",
      title: "crates/pool/src/guard.rs",
      token_count: 1200,
      score: 0.82,
      banked: true,
      provenance: { source: "code_index", labels: [], derived_from: ["crates/pool/src/guard.rs"] },
      blob_ref: { id: "blob_1", hash: "h", size_bytes: 10, media_type: "text/plain" },
    },
    {
      id: "blake3:ccc",
      source: "memory",
      title: "the pool is created once per process",
      token_count: 20,
      score: 0.7,
      provenance: { source: "memory", labels: [], derived_from: [] },
    },
  ],
  dropped: [
    { id: "blake3:ddd", source: "tool_output", token_count: 4200, score: 0.1, reason: "stale" },
    { id: "blake3:eee", source: "code", token_count: 900, score: 0.2, reason: "low_score" },
  ],
  conflicts: [
    { between: ["blake3:aaa", "blake3:bbb"], note: "two answers for the pool size", resolved: false },
    { between: ["blake3:bbb"], note: "already settled", resolved: true },
  ],
  memory: [
    {
      memory_id: "mem_1",
      claim: "the pool guard drops in reverse order",
      outcome_score: 0.8,
      status: "active",
      citations: ["crates/pool/src/guard.rs"],
    },
  ],
  recurrent_state_bytes: 12_000_000,
  live: { effective_ceiling_tokens: 131072, occupancy: 0.1, watermark: "normal" },
};

/* The older flat shape (the mock transport and the retrieval patch) must keep rendering. */
const FLAT: Receipt = {
  retrieved: [{ path: "crates/pool/src/lib.rs", range: "12-30", relevance: 0.74 }],
  dropped: [{ title: "cargo build log", would_be_tokens: 4200, reason: "low relevance" }],
  memory: [{ fact: "DB uses sqlx", confidence: 1 }],
};

const last = () => sent[sent.length - 1];

beforeEach(() => {
  sent.length = 0;
});

describe("the receipt reads the real manifest", () => {
  it("lists every included source with its cost and its content hash", () => {
    const rows = includedSources(REAL);
    expect(rows.map((r) => r.title)).toEqual([
      "Repository instructions (Claude Code compat)",
      "crates/pool/src/guard.rs",
      "the pool is created once per process",
    ]);
    expect(rows[1].tokens).toBe(1200);
    expect(rows[1].hash).toBe("blake3:bbb");
  });

  it("says WHY each source is in the window, from the packer's own numbers", () => {
    const rows = includedSources(REAL);
    expect(rows[0].why).toContain("compat_instructions");
    expect(rows[0].why).toContain("never evicted");
    expect(rows[1].why).toContain("score 0.82");
  });

  it("marks the loaded instructions and the compatibility instructions with their files", () => {
    const rows = includedSources(REAL);
    expect(rows[0].instructions).toBe(true);
    expect(rows[0].compat).toBe(true);
    expect(rows[0].files).toEqual(["CLAUDE.md", "docs/AGENTS.md"]);
    expect(rows[1].instructions).toBe(false);
  });

  it("surfaces attachments and reused kv rather than hiding them", () => {
    const rows = includedSources(REAL);
    expect(rows[1].attachment).toBe("blob_1");
    expect(rows[1].banked).toBe(true);
  });

  it("totals token use, reuse and attachments without inventing a number", () => {
    const t = receiptTotals(REAL);
    expect(t.usedTokens).toBe(9000);
    expect(t.ceilingTokens).toBe(131072);
    expect(t.includedTokens).toBe(1620);
    expect(t.excludedTokens).toBe(5100);
    expect(t.reusedTokens).toBe(512);
    expect(t.stateBytes).toBe(12_000_000);
    expect(t.instructionFiles).toEqual(["CLAUDE.md", "docs/AGENTS.md"]);
    expect(t.compatFiles).toEqual(["CLAUDE.md", "docs/AGENTS.md"]);
    expect(t.attachments).toBe(1);
  });

  it("names the excluded sources and the reason each was left out", () => {
    const rows = excludedSources(REAL);
    expect(rows).toHaveLength(2);
    expect(rows[0].reason).toBe("stale");
    expect(rows[1].reason).toBe("low score");
    expect(rows[0].tokens).toBe(4200);
  });

  it("raises a warning for stale reads and for unresolved conflicts only", () => {
    const w = staleWarnings(REAL);
    expect(w.map((x) => x.kind)).toEqual(["stale", "conflict"]);
    expect(w[1].text).toBe("two answers for the pool size");
    expect(w.some((x) => x.text === "already settled")).toBe(false);
  });

  it("shows the memory the compiler drew from, projection first then window spans", () => {
    const rows = memoryRows(REAL);
    expect(rows[0]).toMatchObject({ id: "mem_1", claim: "the pool guard drops in reverse order", score: 0.8, status: "active" });
    expect(rows[1].claim).toBe("the pool is created once per process");
    // A window span carries a content hash, not a durable record id, so no outcome can be recorded.
    expect(rows[1].id).toBeUndefined();
  });

  it("reads the model from either shape the host publishes", () => {
    expect(modelSummary(REAL)).toMatchObject({ id: "qwen3-coder", arch: "qwen", native: 32768, effective: 131072 });
    // the live projection patch reports the model flat
    expect(modelSummary({ model_id: "rwkv7-2b9", arch: "rwkv7", ctx_len_effective: 262144 })).toMatchObject({
      id: "rwkv7-2b9",
      arch: "rwkv7",
      effective: 262144,
    });
    expect(modelSummary({})).toBeUndefined();
  });

  it("renders a custom source kind without crashing on the tagged form", () => {
    expect(sourceKindLabel("tool_output")).toBe("tool output");
    expect(sourceKindLabel({ custom: "notebook" })).toBe("notebook");
    expect(sourceKindLabel(undefined)).toBe("source");
  });

  it("still reads the older flat shape, so nothing regresses on the live retrieval patch", () => {
    expect(includedSources(FLAT)[0].title).toBe("crates/pool/src/lib.rs:12-30");
    expect(includedSources(FLAT)[0].path).toBe("crates/pool/src/lib.rs");
    expect(excludedSources(FLAT)[0]).toMatchObject({ title: "cargo build log", tokens: 4200 });
    expect(memoryRows(FLAT)[0].claim).toBe("DB uses sqlx");
  });

  it("is empty, not broken, with no manifest at all", () => {
    const empty = asReceipt(null);
    expect(includedSources(empty)).toEqual([]);
    expect(excludedSources(empty)).toEqual([]);
    expect(memoryRows(empty)).toEqual([]);
    expect(staleWarnings(empty)).toEqual([]);
    expect(receiptTotals(empty).includedTokens).toBe(0);
  });
});

describe("every control names a real command", () => {
  const scope = sessionScope("ses_test");

  it("snapshot seals an integrity-verified checkpoint", () => {
    expect(plan.snapshot("ses_test", "guard fix")).toEqual({
      id: "checkpoint_create",
      args: { session_id: "ses_test", label: "guard fix" },
    });
  });

  it("fork forks the session at a recorded event", () => {
    expect(plan.fork("ses_test", "call_9")).toEqual({
      id: "fork_session",
      args: { session_id: "ses_test", at_event: "call_9" },
    });
  });

  it("marking a memory wrong records a real negative outcome", () => {
    expect(plan.outcome("mem_1", false)).toEqual({
      id: "memory_record_outcome",
      args: { memory_id: "mem_1", success: false },
    });
    expect(plan.outcome("mem_1", true).args).toMatchObject({ success: true });
  });

  it("the note field adds a durable memory with the provenance the host requires", () => {
    const p = notePlan("the pool is per process", scope);
    expect(p.id).toBe("memory_add");
    expect(p.args).toMatchObject({
      scope: { kind: "session", id: "ses_test" },
      claim: "the pool is per process",
      source: "context_stack",
      author: "user",
    });
  });

  it("the same note supersedes the memory marked wrong, keeping its history", () => {
    const p = notePlan("the pool is per process", scope, "mem_1", ["crates/pool/src/lib.rs"]);
    expect(p.id).toBe("memory_supersede");
    expect(p.args).toMatchObject({
      old_id: "mem_1",
      replacement: memoryDraft(scope, "the pool is per process", ["crates/pool/src/lib.rs"]),
    });
  });

  it("the memory header revalidates this session's citations against disk", () => {
    expect(plan.revalidate(scope)).toEqual({
      id: "memory_revalidate",
      args: { scope: { kind: "session", id: "ses_test" } },
    });
  });

  it("resolves every plan in the generated command catalog, never a private verb", () => {
    const plans = [
      plan.snapshot("s", "l"),
      plan.fork("s", "e"),
      plan.remember(scope, "c"),
      plan.supersede("m", scope, "c"),
      plan.outcome("m", true),
      plan.revalidate(scope),
      plan.open("a.rs"),
    ];
    for (const p of plans) {
      expect(COMMANDS.find((c) => c.id === p.id), p.id).toBeTruthy();
      // and never the no-op names this panel used to fire
      expect(p.id).not.toBe("pin_span");
      expect(p.id).not.toBe("unpin_span");
      expect(p.id).not.toBe("fleet_run");
    }
  });

  it("keeps the memory controls on the memory domain the catalog assigns to this panel", () => {
    for (const id of ["memory_add", "memory_supersede", "memory_record_outcome", "memory_revalidate"]) {
      const spec = COMMANDS.find((c) => c.id === id);
      expect(spec, id).toBeTruthy();
      expect(spec?.category).toBe("memory");
      expect(spec?.available_surfaces).toContain("context_stack");
    }
  });
});

describe("dispatch goes through the one spine", () => {
  it("snapshot puts a real checkpoint_create on the wire", async () => {
    await runPlan(plan.snapshot("ses_test", "guard fix"));
    expect(last().type).toBe("custom");
    expect(last().data).toMatchObject({
      name: "checkpoint_create",
      payload: { session_id: "ses_test", label: "guard fix" },
    });
  });

  it("fork puts a real fork_session intent on the wire, not a fleet run", async () => {
    await runPlan(plan.fork("ses_test", "call_9"));
    expect(last().type).toBe("fork_session");
    expect(last().data).toMatchObject({ session_id: "ses_test", at_event: "call_9" });
    expect(sent.some((i) => i.type === "custom" && i.data.name === "fleet_run")).toBe(false);
  });

  it("reaches memory over the intent channel now that memory_* binds Custom, not Rpc", async () => {
    // host.rs handle_intent dispatches memory_revalidate over Intent::Custom, so the control is
    // live instead of throwing "needs the elevated rpc channel".
    await runPlan(plan.revalidate(sessionScope("ses_test")));
    expect(last().type).toBe("custom");
    expect(last().data.name).toBe("memory_revalidate");
  });

  it("still refuses honestly when the spine cannot carry a command's payload", async () => {
    // run_static_analysis is Custom-bound now, but the host arm refuses an empty payload, so the
    // spine refuses it first with the reason. The control surfaces that as a failed state, never a
    // fake success.
    await expect(runPlan({ id: "run_static_analysis", args: {} })).rejects.toThrow(/needs paths/);
  });
});

describe("the panel is MOUNTED", () => {
  // It was imported by no module at all, so every control on it rendered nowhere and the whole
  // surface was dead code wearing a receipt.
  const panel = readFileSync(join(__dirname, "../home/ChatPanel.tsx"), "utf8");

  it("renders as a face of the conversation side panel", () => {
    expect(panel).toContain('import { ContextStack } from "../ContextStack"');
    expect(panel).toContain('{panel === "context" ? <ContextStack /> : null}');
  });

  it("the panel bar and the palette both reach that face", () => {
    expect(readFileSync(join(__dirname, "../home/Home.tsx"), "utf8")).toContain('kind: "context"');
    expect(readFileSync(join(__dirname, "../../store.ts"), "utf8")).toContain('id: "panel.context"');
    expect(readFileSync(join(__dirname, "../../App.tsx"), "utf8")).toContain('"panel.context":');
  });

  it("reads the host's published manifest, never the connector write route", () => {
    // `context.compile` upserts the durable memory store, so it is refused on the read-only
    // connector route (connectors.rs CONNECTOR_READ_METHODS).
    const src = readFileSync(join(__dirname, "../ContextStack.tsx"), "utf8");
    expect(src).not.toContain("callConnector");
  });
});

describe("retired controls are gone", () => {
  const src = readFileSync(join(__dirname, "../ContextStack.tsx"), "utf8");
  const code = src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");

  it("drops the fake Skills store: no save-skill button, no hardcoded skill rows", () => {
    expect(code).not.toContain("SKILLS");
    expect(code).not.toContain("save skill");
    expect(code).not.toContain("load skill");
  });

  it("drops the no-op pin_span / unpin_span path entirely", () => {
    expect(code).not.toContain("pin_span");
    expect(code).not.toContain("unpin_span");
  });

  it("drops the fleet_run misuse and the memcpy claim from fork", () => {
    expect(code).not.toContain("fleet_run");
    expect(code.toLowerCase()).not.toContain("memcpy");
  });

  it("drops the duplicate model chooser button", () => {
    expect(code).not.toContain("switch_profile");
    expect(code).not.toContain("switch_model");
  });

  it("builds no intent of its own: the panel speaks command ids only", () => {
    expect(code).not.toContain("intent.custom");
    expect(code).not.toContain("sendIntent");
  });
});
