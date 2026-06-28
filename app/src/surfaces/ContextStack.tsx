/*
  ContextStack.tsx: the right rail, THE differentiator (01-surfaces §E). Always visible.
  Renders the live ContextManifest verbatim: Model, a stacked budget bar, retrieved files,
  tools, memory, dropped candidates. The live stratum wears the gold radiation when a turn is active.
  Skeleton: real sections + store wiring; pin/unpin drag affordances and the manifest ring scrub
  coupling are fleshed out in the surface pass.
  Sends: Custom:{pin_span, unpin_span, switch_profile, ...}. Consumes: projection_patch(context_manifest).
*/
import { sendIntent } from "../ipc";
import { useStore } from "../store";
import { intent } from "../wire";
import { Panel, SectionLabel } from "../ui";

// Color budget segments by source, derived from the token ramp + gold (no blue/purple).
const SEGMENT_COLOR: Record<string, string> = {
  system: "var(--text-low)",
  code: "var(--radiation)",
  tools: "var(--warning)",
  memory: "var(--success)",
  history: "var(--text-mid)",
};

export function ContextStack() {
  const manifest = useStore((s) => s.manifest);
  const live = useStore((s) => s.runPhase === "executing" || s.runPhase === "planning");

  return (
    <aside
      aria-label="Context Stack"
      style={{ height: "100%", overflowY: "auto", padding: "var(--s3)", display: "flex", flexDirection: "column", gap: "var(--s3)" }}
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

      {!manifest ? (
        <Panel pad="var(--s4)" style={{ color: "var(--text-low)", fontSize: "var(--text-sm)" }}>
          No manifest yet. It compiles when the agent assembles a turn's context.
        </Panel>
      ) : (
        <>
          {/* MODEL */}
          {manifest.model ? (
            <Panel pad="var(--s3)">
              <SectionLabel>Model</SectionLabel>
              <button
                onClick={() => void sendIntent(intent.custom("switch_profile", { profile: manifest.model?.profile }))}
                style={{ color: "var(--text-hi)", fontSize: "var(--text-sm)", textAlign: "left", width: "100%" }}
              >
                {manifest.model.id} / {manifest.model.profile} / {manifest.model.sampling}
                <div style={{ color: "var(--text-low)", fontSize: "var(--text-xs)" }}>ctx {manifest.model.ctx.toLocaleString()}</div>
              </button>
            </Panel>
          ) : null}

          {/* BUDGET */}
          {manifest.budget ? (
            <Panel pad="var(--s3)">
              <SectionLabel>Budget</SectionLabel>
              <div style={{ fontSize: "var(--text-sm)", color: "var(--text-mid)", marginBottom: "var(--s2)" }}>
                {manifest.budget.used.toLocaleString()} / {manifest.budget.total.toLocaleString()}
              </div>
              <div style={{ display: "flex", height: 8, borderRadius: 999, overflow: "hidden", boxShadow: "inset 0 0 0 1px var(--rim)" }}>
                {manifest.budget.segments.map((seg) => (
                  <div
                    key={seg.source}
                    title={`${seg.source}: ${seg.tokens.toLocaleString()} tok`}
                    style={{
                      width: `${(seg.tokens / manifest.budget!.total) * 100}%`,
                      background: SEGMENT_COLOR[seg.source] ?? "var(--text-low)",
                    }}
                  />
                ))}
              </div>
            </Panel>
          ) : null}

          {/* RETRIEVED */}
          {manifest.retrieved?.length ? (
            <Panel pad="var(--s3)">
              <SectionLabel count={manifest.retrieved.length}>Retrieved</SectionLabel>
              {manifest.retrieved.map((r) => (
                <Row key={r.path + r.range}>
                  <button onClick={() => void sendIntent(intent.openFile(r.path))} style={rowBtn}>
                    {r.path.split("/").pop()}:{r.range}
                  </button>
                  <span style={{ color: "var(--text-low)", fontSize: "var(--text-xs)" }}>{r.relevance.toFixed(2)}</span>
                  <button
                    title="pin into next turn"
                    onClick={() => void sendIntent(intent.custom("pin_span", { path: r.path, range: r.range }))}
                    style={{ color: "var(--radiation)", fontSize: "var(--text-xs)" }}
                  >
                    pin
                  </button>
                </Row>
              ))}
            </Panel>
          ) : null}

          {/* TOOLS */}
          {manifest.tools?.length ? (
            <Panel pad="var(--s3)">
              <SectionLabel count={manifest.tools.length}>Tools</SectionLabel>
              {manifest.tools.map((t) => (
                <Row key={t.name}>
                  <span style={{ color: t.ok ? "var(--success)" : "var(--danger)" }}>{t.ok ? "ok" : "fail"}</span>
                  <span style={{ color: "var(--text-mid)" }}>{t.name}</span>
                </Row>
              ))}
            </Panel>
          ) : null}

          {/* MEMORY */}
          {manifest.memory?.length ? (
            <Panel pad="var(--s3)">
              <SectionLabel count={manifest.memory.length}>Memory</SectionLabel>
              {manifest.memory.map((m) => (
                <Row key={m.fact}>
                  <span style={{ color: "var(--text-mid)" }}>{m.fact}</span>
                  <span style={{ color: "var(--text-low)", fontSize: "var(--text-xs)" }}>{m.confidence.toFixed(1)}</span>
                </Row>
              ))}
            </Panel>
          ) : null}

          {/* DROPPED (one-click pin back) */}
          {manifest.dropped?.length ? (
            <Panel pad="var(--s3)">
              <SectionLabel count={manifest.dropped.length}>Dropped</SectionLabel>
              {manifest.dropped.map((d) => (
                <Row key={d.title}>
                  <span style={{ color: "var(--text-low)" }}>{d.title}</span>
                  <span style={{ color: "var(--text-low)", fontSize: "var(--text-xs)" }}>{d.would_be_tokens.toLocaleString()}</span>
                  <button
                    title={d.reason}
                    onClick={() => void sendIntent(intent.custom("pin_span", { title: d.title }))}
                    style={{ color: "var(--radiation)", fontSize: "var(--text-xs)" }}
                  >
                    pin
                  </button>
                </Row>
              ))}
            </Panel>
          ) : null}
        </>
      )}
    </aside>
  );
}

const rowBtn = {
  color: "var(--text-mid)",
  fontSize: "var(--text-sm)",
  textAlign: "left" as const,
  flex: 1,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap" as const,
};

function Row({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "var(--s2)", padding: "2px 0" }}>{children}</div>
  );
}
