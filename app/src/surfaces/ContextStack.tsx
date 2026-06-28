/*
  ContextStack.tsx: the right rail, THE differentiator (doctrine C7 + D1.5). The persistent
  spine across all three surfaces: wherever you are, you are watching the agent's mind. It renders
  the live ContextManifest verbatim as a calm legible COLUMN of strata, top to bottom, and makes
  every stratum touchable (pin/unpin, mute, evict, inject). The current-action stratum is the NOW:
  it wears the breathing gold radiation and scrolls the agent's real moves (no spinner, no % bar).

  Strata (top -> bottom): Model, Budget, Retrieved files, Symbols/Tools-in-context, Memory,
  Tests & state, and the live Current action feed.

  Binds: context.compile -> {prompt, manifest} (callConnector, refreshed when a turn ends) and the
  projection_patch(context_manifest | retrieval | memory) updates folded into the store. Steering
  writes go out as Custom intents the host's compiler honors next turn; a local optimistic overlay
  makes each touch feel material immediately (useSteer).

  Consumes (read-only): store slices manifest, tools (live feed), projections.build/.test, runPhase,
  runtimeStatus/runtimeDetail. Sends: open_file; Custom{pin_span, unpin_span, switch_profile,
  toggle_confidence, dismiss}. Touches NO shared foundation file.
*/
import { useEffect, useState } from "react";
import { callConnector, sendIntent } from "../ipc";
import { useStore, type ContextManifest } from "../store";
import { intent } from "../wire";
import { SectionLabel } from "../ui";
import { HardwareToggle, Line, Mark, NoteField, Stratum } from "./contextstack/parts";
import { spanKey, useSteer } from "./contextstack/state";

// Budget segment colors derive from the token ramp + gold. No blue/purple (doctrine C15).
const SEGMENT_COLOR: Record<string, string> = {
  system: "var(--text-low)",
  code: "var(--radiation)",
  tools: "var(--warning)",
  memory: "var(--success)",
  history: "var(--text-mid)",
};

const fileName = (p: string) => p.split("/").pop() ?? p;

export function ContextStack() {
  const manifest = useStore((s) => s.manifest);
  const liveFeed = useStore((s) => s.tools); // tool_progress stream = the agent's real moves
  const runPhase = useStore((s) => s.runPhase);
  const runtimeDetail = useStore((s) => s.runtimeDetail);
  const build = useStore((s) => s.projections.build) as { ok?: boolean; summary?: string } | undefined;
  const test = useStore((s) => s.projections.test) as
    | { passed?: number; failed?: number; total?: number; summary?: string }
    | undefined;

  const live = runPhase === "executing" || runPhase === "planning";
  const steer = useSteer();

  // Bind to the context connector: compile the manifest on mount and whenever a turn settles,
  // so the rail is the verbatim window assembly even before the host pushes a projection patch.
  const [compiled, setCompiled] = useState<ContextManifest | null>(null);
  useEffect(() => {
    let alive = true;
    callConnector<{ prompt: string; manifest: ContextManifest }>("context", "compile", {})
      .then((r) => alive && r?.manifest && setCompiled(r.manifest))
      .catch(() => void 0); // failures surface via the store's transport notice path, not here
    return () => {
      alive = false;
    };
  }, [runPhase === "done"]); // recompile when a turn ends (the manifest is published per turn)

  // The store's folded manifest is truth-of-record; the compile result is the fallback before
  // the first projection patch arrives. Prefer the live store manifest when present.
  const m = manifest ?? compiled;

  return (
    <aside
      aria-label="Context Stack"
      style={{
        height: "100%",
        overflowY: "auto",
        padding: "var(--s3)",
        display: "flex",
        flexDirection: "column",
        gap: "var(--s2)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "var(--s2)" }}>
        <SectionLabel>Context Stack</SectionLabel>
        {live ? (
          <span
            title="agent active"
            style={{
              marginLeft: "auto",
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: "var(--radiation)",
              animation: "radiation-breathe 2.2s ease-in-out infinite",
            }}
          />
        ) : null}
      </div>

      {!m && liveFeed.length === 0 ? (
        <section className="panel" style={{ padding: "var(--s4)", color: "var(--text-low)", fontSize: "var(--text-sm)" }}>
          No manifest yet. It compiles when the agent assembles a turn's context.
        </section>
      ) : null}

      {/* MODEL: id/arch/ctx, profile, sampling; click cycles the profile (switch_profile). */}
      {m?.model ? (
        <Stratum
          label="Model"
          summary={`${m.model.id} ${(m.budget ? Math.round((m.budget.used / m.model.ctx) * 100) : 0)}%`}
        >
          <button
            onClick={() => void sendIntent(intent.custom("switch_profile", { profile: m.model?.profile }))}
            title="switch profile"
            style={{ textAlign: "left", width: "100%", color: "var(--text-hi)", fontSize: "var(--text-sm)" }}
          >
            {m.model.id} / {m.model.arch}
            <div style={{ color: "var(--text-low)", fontSize: "var(--text-xs)" }}>
              {m.model.profile} / {m.model.sampling} / ctx {m.model.ctx.toLocaleString()}
            </div>
          </button>
        </Stratum>
      ) : null}

      {/* BUDGET: the stacked bar colored by source, with the drop note. */}
      {m?.budget ? (
        <Stratum
          label="Budget"
          summary={`${m.budget.used.toLocaleString()} / ${m.budget.total.toLocaleString()}`}
          defaultOpen
        >
          <div style={{ fontSize: "var(--text-sm)", color: "var(--text-mid)", marginBottom: "var(--s2)" }}>
            {m.budget.used.toLocaleString()} used / {m.budget.free.toLocaleString()} free
          </div>
          <div
            style={{
              display: "flex",
              height: 8,
              borderRadius: 999,
              overflow: "hidden",
              boxShadow: "inset 0 0 0 1px var(--rim)",
            }}
          >
            {m.budget.segments.map((seg) => (
              <div
                key={seg.source}
                title={`${seg.source}: ${seg.tokens.toLocaleString()} tok`}
                style={{
                  width: `${(seg.tokens / m.budget!.total) * 100}%`,
                  background: SEGMENT_COLOR[seg.source] ?? "var(--text-low)",
                }}
              />
            ))}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--s2)", marginTop: "var(--s2)" }}>
            {m.budget.segments.map((seg) => (
              <span key={seg.source} style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: "var(--text-xs)", color: "var(--text-low)" }}>
                <span style={{ width: 7, height: 7, borderRadius: 2, background: SEGMENT_COLOR[seg.source] ?? "var(--text-low)" }} />
                {seg.source} {Math.round(seg.tokens / 100) / 10}k
              </span>
            ))}
          </div>
        </Stratum>
      ) : null}

      {/* RETRIEVED: files/spans. Click opens; pin/unpin holds the span into the next turn. */}
      {m?.retrieved?.length ? (
        <Stratum
          label="Retrieved"
          count={m.retrieved.length}
          summary={fileName(m.retrieved[0].path)}
          defaultOpen
        >
          {m.retrieved.map((r) => {
            const id = spanKey("file", r.path + r.range);
            const pinned = steer.on(id, "pin");
            return (
              <Line key={id}>
                <button
                  onClick={() => void sendIntent(intent.openFile(r.path))}
                  title={r.path}
                  style={{ flex: 1, minWidth: 0, textAlign: "left", color: "var(--text-mid)", fontSize: "var(--text-sm)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                >
                  {fileName(r.path)}:{r.range}
                </button>
                <span style={{ flex: "0 0 auto", color: "var(--text-low)", fontSize: "var(--text-xs)" }}>{r.relevance.toFixed(2)}</span>
                <HardwareToggle
                  label="pin"
                  on={pinned}
                  title={pinned ? "unpin from next turn" : "pin into next turn"}
                  onToggle={() => {
                    const now = steer.toggle(id, "pin");
                    void sendIntent(intent.custom(now ? "pin_span" : "unpin_span", { path: r.path, range: r.range }));
                  }}
                />
              </Line>
            );
          })}
        </Stratum>
      ) : null}

      {/* TOOLS in context: each can be muted (drop its output from the next assembly). */}
      {m?.tools?.length ? (
        <Stratum
          label="Tools"
          count={m.tools.length}
          summary={m.tools.filter((t) => t.ok).length + " ok"}
        >
          {m.tools.map((t) => {
            const id = spanKey("tool", t.name);
            const muted = steer.on(id, "mute");
            return (
              <Line key={id}>
                <Mark ok={t.ok} />
                <span style={{ flex: 1, minWidth: 0, color: muted ? "var(--text-low)" : "var(--text-mid)", textDecoration: muted ? "line-through" : "none", fontSize: "var(--text-sm)" }}>
                  {t.name}
                </span>
                <HardwareToggle
                  label="mute"
                  tone="mute"
                  on={muted}
                  title={muted ? "unmute tool output" : "mute this tool's output"}
                  onToggle={() => {
                    const now = steer.toggle(id, "mute");
                    void sendIntent(intent.custom(now ? "pin_span" : "unpin_span", { mute_tool: t.name }));
                  }}
                />
              </Line>
            );
          })}
        </Stratum>
      ) : null}

      {/* MEMORY: injected facts. Each can be evicted from context, and notes injected. */}
      {m?.memory?.length ? (
        <Stratum label="Memory" count={m.memory.length} summary={m.memory[0]?.fact}>
          {m.memory.map((mem) => {
            const id = spanKey("mem", mem.fact);
            const evicted = steer.on(id, "evict");
            return (
              <Line key={id}>
                <span style={{ flex: 1, minWidth: 0, color: evicted ? "var(--text-low)" : "var(--text-mid)", textDecoration: evicted ? "line-through" : "none", fontSize: "var(--text-sm)" }}>
                  {mem.fact}
                </span>
                <span style={{ flex: "0 0 auto", color: "var(--text-low)", fontSize: "var(--text-xs)" }}>{mem.confidence.toFixed(1)}</span>
                <HardwareToggle
                  label="evict"
                  tone="danger"
                  on={evicted}
                  title={evicted ? "restore this memory" : "evict this memory from context"}
                  onToggle={() => {
                    const now = steer.toggle(id, "evict");
                    void sendIntent(intent.custom(now ? "unpin_span" : "pin_span", { evict_memory: mem.fact }));
                  }}
                />
              </Line>
            );
          })}
          <div style={{ marginTop: "var(--s1)" }}>
            <NoteField
              value={steer.noteOn("memory")}
              placeholder="inject a note into memory"
              onCommit={(text) => {
                steer.setNote("memory", text);
                void sendIntent(intent.custom("pin_span", { note: text }));
              }}
            />
          </div>
        </Stratum>
      ) : null}

      {/* DROPPED: candidates cut to fit. One press pins it back into the next turn. */}
      {m?.dropped?.length ? (
        <Stratum label="Dropped" count={m.dropped.length} summary={m.dropped[0]?.title}>
          {m.dropped.map((d) => {
            const id = spanKey("drop", d.title);
            const pinned = steer.on(id, "pin");
            return (
              <Line key={id} title={d.reason}>
                <span style={{ flex: 1, minWidth: 0, color: "var(--text-low)", fontSize: "var(--text-sm)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {d.title}
                </span>
                <span style={{ flex: "0 0 auto", color: "var(--text-low)", fontSize: "var(--text-xs)" }}>{d.would_be_tokens.toLocaleString()}</span>
                <HardwareToggle
                  label="pin"
                  on={pinned}
                  title={`${d.reason} / pin back into next turn`}
                  onToggle={() => {
                    const now = steer.toggle(id, "pin");
                    void sendIntent(intent.custom(now ? "pin_span" : "unpin_span", { title: d.title }));
                  }}
                />
              </Line>
            );
          })}
        </Stratum>
      ) : null}

      {/* TESTS & STATE: build + test projection, calm and read-only, shape not spinner. */}
      {build || test ? (
        <Stratum
          label="Tests & state"
          summary={test ? `${test.passed ?? 0}/${test.total ?? "?"} pass` : build?.ok ? "build ok" : "build"}
        >
          {build ? (
            <Line>
              <Mark ok={build.ok !== false} />
              <span style={{ flex: 1, color: "var(--text-mid)", fontSize: "var(--text-sm)" }}>{build.summary ?? "build"}</span>
            </Line>
          ) : null}
          {test ? (
            <Line>
              <Mark ok={(test.failed ?? 0) === 0} />
              <span style={{ flex: 1, color: "var(--text-mid)", fontSize: "var(--text-sm)" }}>
                {test.summary ?? `${test.passed ?? 0} passed${test.failed ? `, ${test.failed} failed` : ""}`}
              </span>
            </Line>
          ) : null}
        </Stratum>
      ) : null}

      {/* CURRENT ACTION: the NOW. The live-feed stratum wears the breathing gold and scrolls
          the agent's real moves (no % bar). Empty when idle, calm and honest. */}
      <Stratum
        label="Current action"
        live={live}
        defaultOpen
        summary={live ? runtimeDetail ?? "working" : "idle"}
      >
        {liveFeed.length === 0 ? (
          <div style={{ color: "var(--text-low)", fontSize: "var(--text-xs)", padding: "var(--s1) var(--s2)" }}>
            {live ? "assembling the turn" : "no moves yet this session"}
          </div>
        ) : (
          liveFeed.slice(-8).map((t, i) => {
            const last = i === liveFeed.slice(-8).length - 1;
            return (
              <Line key={t.call_id + t.ts}>
                <span
                  style={{
                    flex: "0 0 auto",
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: last && live ? "var(--radiation)" : "var(--text-low)",
                    boxShadow: last && live ? "0 0 6px 0 var(--radiation-bloom)" : undefined,
                  }}
                />
                <span style={{ flex: 1, minWidth: 0, color: last ? "var(--text-hi)" : "var(--text-mid)", fontSize: "var(--text-sm)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {t.message}
                </span>
              </Line>
            );
          })
        )}
      </Stratum>
    </aside>
  );
}
