/*
  ContextStack.tsx: the LIGHT WELL, THE differentiator (Doctrine v3, Part II + the Part IV recipe).
  A narrow vertical shaft along the east wall, present in every chamber: wherever you are, you are
  watching the agent's mind. It renders the live ContextManifest verbatim as a calm legible COLUMN
  of strata, top to bottom, each a .volume ledge floating in the void, and makes every stratum
  touchable (pin/unpin, mute, evict, inject). The current-action stratum is the NOW: it wears the
  breathing LIGHT (useLight, never gold) and scrolls the agent's real moves (no spinner, no % bar).

  Strata (top -> bottom): Model, Budget, Retrieved files, Tools-in-context, Memory, Dropped,
  Tests & state, and the live Current action feed.

  Binds: context.compile -> {prompt, manifest} (callConnector, refreshed when a turn ends) and the
  projection_patch(context_manifest | retrieval | memory) updates folded into the store. Steering
  writes go out as Custom intents the host's compiler honors next turn; a local optimistic overlay
  makes each touch feel material immediately (useSteer).

  Consumes (read-only): store slices manifest, tools (live feed), projections.build/.test, runPhase,
  runtimeDetail. Sends: open_file; Custom{pin_span, unpin_span, switch_profile}. Touches NO shared
  foundation file.
*/
import { useEffect, useState } from "react";
import { callConnector, sendIntent } from "../ipc";
import { useStore, type ContextManifest } from "../store";
import { intent } from "../wire";
import { SectionLabel } from "../ui";
import { HardwareToggle, Line, NoteField, OkMark, Stratum } from "./contextstack/parts";
import { spanKey, useSteer } from "./contextstack/state";

/*
  Budget segments read by LIGHT, not color (Doctrine: state is read by light, never a colored badge).
  The bar is a calm monochrome ramp of concrete tiers brightening toward LIGHT for the heaviest
  source; there is no hue here. Only --ok/--bad ever appear in this surface, and only where genuinely
  semantic (a tool result, a test outcome), never to label a budget source.
*/
const SEGMENT_TONE: Record<string, string> = {
  system: "var(--concrete-4)",
  code: "var(--light)",
  tools: "var(--text-2)",
  memory: "var(--text-3)",
  history: "var(--mute)",
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
        padding: "var(--ma-6)",
        display: "flex",
        flexDirection: "column",
        gap: "var(--ma-4)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "var(--ma-3)", paddingBottom: "var(--ma-2)" }}>
        <SectionLabel>Context Stack</SectionLabel>
        {live ? (
          <span
            className="alive"
            title="agent active"
            aria-label="agent active"
            style={{
              marginLeft: "auto",
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: "var(--light)",
            }}
          />
        ) : null}
      </div>

      {!m && liveFeed.length === 0 ? (
        <section className="volume" style={{ color: "var(--text-3)" }}>
          <span className="t-body">No manifest yet. It compiles when the agent assembles a turn's context.</span>
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
            className="t-code"
            style={{ textAlign: "left", width: "100%", color: "var(--text-1)" }}
          >
            {m.model.id} / {m.model.arch}
            <div className="t-micro" style={{ color: "var(--text-3)", marginTop: "var(--ma-1)" }}>
              {m.model.profile} / {m.model.sampling} / ctx {m.model.ctx.toLocaleString()}
            </div>
          </button>
        </Stratum>
      ) : null}

      {/* BUDGET: the stacked bar read by light, framed as abundance, never an alarm. */}
      {m?.budget ? (
        <Stratum
          label="Budget"
          summary={`${m.budget.used.toLocaleString()} / ${m.budget.total.toLocaleString()}`}
          defaultOpen
        >
          <div className="t-code" style={{ color: "var(--text-2)", marginBottom: "var(--ma-3)" }}>
            {m.budget.used.toLocaleString()} used / {m.budget.free.toLocaleString()} free
          </div>
          <div
            style={{
              display: "flex",
              height: 8,
              borderRadius: "var(--radius-pill)",
              overflow: "hidden",
              boxShadow: "var(--hairline), var(--inner-glow)",
            }}
          >
            {m.budget.segments.map((seg) => (
              <div
                key={seg.source}
                title={`${seg.source}: ${seg.tokens.toLocaleString()} tok`}
                style={{
                  width: `${(seg.tokens / m.budget!.total) * 100}%`,
                  background: SEGMENT_TONE[seg.source] ?? "var(--concrete-4)",
                }}
              />
            ))}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--ma-3)", marginTop: "var(--ma-3)" }}>
            {m.budget.segments.map((seg) => (
              <span key={seg.source} className="t-micro" style={{ display: "inline-flex", alignItems: "center", gap: 6, color: "var(--text-3)" }}>
                <span style={{ width: 7, height: 7, borderRadius: 2, background: SEGMENT_TONE[seg.source] ?? "var(--concrete-4)" }} />
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
                  className="t-code"
                  style={{ flex: 1, minWidth: 0, textAlign: "left", color: "var(--text-2)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                >
                  {fileName(r.path)}:{r.range}
                </button>
                <span className="t-micro" style={{ flex: "0 0 auto", color: "var(--text-3)" }}>{r.relevance.toFixed(2)}</span>
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
                <OkMark ok={t.ok} />
                <span className="t-code" style={{ flex: 1, minWidth: 0, color: muted ? "var(--text-3)" : "var(--text-2)", textDecoration: muted ? "line-through" : "none" }}>
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
                <span className="t-code" style={{ flex: 1, minWidth: 0, color: evicted ? "var(--text-3)" : "var(--text-2)", textDecoration: evicted ? "line-through" : "none" }}>
                  {mem.fact}
                </span>
                <span className="t-micro" style={{ flex: "0 0 auto", color: "var(--text-3)" }}>{mem.confidence.toFixed(1)}</span>
                <HardwareToggle
                  label="evict"
                  tone="bad"
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
          <div style={{ marginTop: "var(--ma-2)" }}>
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
                <span className="t-code" style={{ flex: 1, minWidth: 0, color: "var(--text-3)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {d.title}
                </span>
                <span className="t-micro" style={{ flex: "0 0 auto", color: "var(--text-3)" }}>{d.would_be_tokens.toLocaleString()}</span>
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
              <OkMark ok={build.ok !== false} />
              <span className="t-code" style={{ flex: 1, color: "var(--text-2)" }}>{build.summary ?? "build"}</span>
            </Line>
          ) : null}
          {test ? (
            <Line>
              <OkMark ok={(test.failed ?? 0) === 0} />
              <span className="t-code" style={{ flex: 1, color: "var(--text-2)" }}>
                {test.summary ?? `${test.passed ?? 0} passed${test.failed ? `, ${test.failed} failed` : ""}`}
              </span>
            </Line>
          ) : null}
        </Stratum>
      ) : null}

      {/* CURRENT ACTION: the NOW. The live-feed stratum wears the breathing LIGHT and scrolls
          the agent's real moves (no % bar). Empty when idle, calm and honest. */}
      <Stratum
        label="Current action"
        live={live}
        defaultOpen
        summary={live ? runtimeDetail ?? "working" : "idle"}
      >
        {liveFeed.length === 0 ? (
          <div className="t-micro" style={{ color: "var(--text-3)", padding: "var(--ma-2) var(--ma-3)" }}>
            {live ? "assembling the turn" : "no moves yet this session"}
          </div>
        ) : (
          liveFeed.slice(-8).map((t, i) => {
            const last = i === liveFeed.slice(-8).length - 1;
            return (
              <Line key={t.call_id + t.ts}>
                <span
                  className={last && live ? "alive" : undefined}
                  style={{
                    flex: "0 0 auto",
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: last && live ? "var(--light)" : "var(--text-3)",
                  }}
                />
                <span className="t-code" style={{ flex: 1, minWidth: 0, color: last ? "var(--text-1)" : "var(--text-2)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
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
