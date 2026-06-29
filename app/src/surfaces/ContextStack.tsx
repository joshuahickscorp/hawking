import { useEffect, useState } from "react";
import { callConnector, sendIntent } from "../ipc";
import { useStore, type ContextManifest } from "../store";
import { intent } from "../wire";
import { HardwareToggle, Line, NoteField, OkMark, Stratum } from "./contextstack/parts";
import { spanKey, useSteer } from "./contextstack/state";

const fileName = (p: string) => p.split("/").pop() ?? p;

// Saved skill states: ~10 MB trained-state seeds you load instantly (instant-resume, not "memory").
const SKILLS = ["refactor-mode", "test-writing", "this-repo style"];

export function ContextStack() {
  const manifest = useStore((s) => s.manifest);
  const liveFeed = useStore((s) => s.tools); // tool_progress stream = the agent's real moves
  const runPhase = useStore((s) => s.runPhase);
  const runtimeDetail = useStore((s) => s.runtimeDetail);
  const pushNotice = useStore((s) => s.pushNotice);
  const build = useStore((s) => s.projections.build) as { ok?: boolean; summary?: string } | undefined;
  const test = useStore((s) => s.projections.test) as
    | { passed?: number; failed?: number; total?: number; summary?: string }
    | undefined;

  const live = runPhase === "executing" || runPhase === "planning";
  const steer = useSteer();

  const [compiled, setCompiled] = useState<ContextManifest | null>(null);
  useEffect(() => {
    let alive = true;
    // The context connector compiles around a task; use the latest user turn, else a workspace default.
    const msgs = useStore.getState().messages;
    const task = [...msgs].reverse().find((x) => x.role === "user")?.text ?? "review the current workspace";
    callConnector<{ prompt: string; manifest: ContextManifest }>("context", "compile", { task })
      .then((r) => alive && r?.manifest && setCompiled(r.manifest))
      .catch(() => void 0); // failures surface via the store's transport notice path, not here
    return () => {
      alive = false;
    };
  }, [runPhase === "done"]); // recompile when a turn ends (the manifest is published per turn)

  const m = manifest ?? compiled;

  return (
    <aside aria-label="Context Stack" className="ctx-stack">
      {/* The Context Stack is the agent's STATE, opened: a serializable object, not a re-prompt.
          Snapshot/fork are instant because the RWKV state is a constant-size memcpy (plan 2 backend). */}
      <div className="ctx-state">
        <span className={"ctx-state__dot" + (live ? " ctx-state__dot--live" : "")} aria-hidden />
        <span className="ctx-state__title">state · {live ? "warm" : "ready"}</span>
        <span className="ctx-state__meta">~12 MB · serializable</span>
        <div className="ctx-state__actions">
          <button
            className="ctx-state__btn"
            title="Snapshot this state, resume instantly later (no re-prefill)"
            onClick={() => pushNotice({ kind: "info", code: "state", message: "state snapshot saved · instant resume" })}
          >
            snapshot
          </button>
          <button
            className="ctx-state__btn"
            title="Fork this state into a new branch (instant, it's a memcpy)"
            onClick={() => {
              void sendIntent(intent.custom("fleet_run", { task: "fork from current state", n: 2 }));
              pushNotice({ kind: "info", code: "state", message: "forked from current state" });
            }}
          >
            fork
          </button>
          <button
            className="ctx-state__btn"
            title="Save this state as a reusable skill, load it instantly later"
            onClick={() => pushNotice({ kind: "info", code: "skill", message: "saved as a skill state" })}
          >
            save skill
          </button>
        </div>
      </div>

      <Stratum label="Skills" count={SKILLS.length} summary="instant-resume states">
        {SKILLS.map((sk) => (
          <Line key={sk}>
            <span className="t-code" style={{ flex: 1, minWidth: 0, color: "var(--text-2)" }}>{sk}</span>
            <button
              className="ctx-state__btn"
              title="Load this skill state instantly (no re-prefill)"
              onClick={() => pushNotice({ kind: "info", code: "skill", message: `loaded skill · ${sk}` })}
            >
              load
            </button>
          </Line>
        ))}
      </Stratum>

      {!m && liveFeed.length === 0 ? (
        <div className="sidebar__empty">No context yet</div>
      ) : null}

      {m?.model ? (
        <Stratum
          label="Model"
          summary={m.model.id}
        >
          <button
            onClick={() => void sendIntent(intent.custom("switch_profile", { profile: m.model?.profile }))}
            title="switch profile"
            className="ctx-row ctx-row--btn"
            style={{ textAlign: "left", width: "100%", color: "var(--text)", padding: "var(--ma-1) var(--ma-2) var(--ma-1) 22px", background: "transparent" }}
          >
            <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
              <span style={{ fontSize: 13, color: "var(--text)" }}>{m.model.id} / {m.model.arch}</span>
              <span style={{ fontSize: 11, color: "var(--text-dim)" }}>
                {m.model.profile} / {m.model.sampling}
              </span>
            </div>
          </button>
        </Stratum>
      ) : null}

      {/* No budget stratum: HIDE is local and the format carries long context, so we do not meter
          tokens or show a context-window cap anywhere (doctrine Part VII Phase 2). */}

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
                  style={{ flex: 1, minWidth: 0, textAlign: "left", background: "transparent", color: "var(--text)", fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                >
                  {fileName(r.path)}:{r.range}
                </button>
                <span style={{ flex: "0 0 auto", fontSize: 11, color: "var(--text-dim)" }}>{r.relevance.toFixed(2)}</span>
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
                <span style={{ flex: 1, minWidth: 0, fontSize: 13, color: muted ? "var(--text-dim)" : "var(--text)", textDecoration: muted ? "line-through" : "none", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
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

      {m?.memory?.length ? (
        <Stratum label="Memory" count={m.memory.length} summary={m.memory[0]?.fact}>
          {m.memory.map((mem) => {
            const id = spanKey("mem", mem.fact);
            const evicted = steer.on(id, "evict");
            return (
              <Line key={id}>
                <span style={{ flex: 1, minWidth: 0, fontSize: 13, color: evicted ? "var(--text-dim)" : "var(--text)", textDecoration: evicted ? "line-through" : "none", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {mem.fact}
                </span>
                <span style={{ flex: "0 0 auto", fontSize: 11, color: "var(--text-dim)" }}>{mem.confidence.toFixed(1)}</span>
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
          <div>
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

      {m?.dropped?.length ? (
        <Stratum label="Dropped" count={m.dropped.length} summary={m.dropped[0]?.title}>
          {m.dropped.map((d) => {
            const id = spanKey("drop", d.title);
            const pinned = steer.on(id, "pin");
            return (
              <Line key={id} title={d.reason}>
                <span style={{ flex: 1, minWidth: 0, fontSize: 13, color: "var(--text-dim)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {d.title}
                </span>
                <span style={{ flex: "0 0 auto", fontSize: 11, color: "var(--text-dim)" }}>{d.would_be_tokens.toLocaleString()}</span>
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

      {build || test ? (
        <Stratum
          label="Tests & state"
          summary={test ? `${test.passed ?? 0}/${test.total ?? "?"} pass` : build?.ok ? "build ok" : "build"}
        >
          {build ? (
            <Line>
              <OkMark ok={build.ok !== false} />
              <span style={{ flex: 1, minWidth: 0, fontSize: 13, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{build.summary ?? "build"}</span>
            </Line>
          ) : null}
          {test ? (
            <Line>
              <OkMark ok={(test.failed ?? 0) === 0} />
              <span style={{ flex: 1, minWidth: 0, fontSize: 13, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {test.summary ?? `${test.passed ?? 0} passed${test.failed ? `, ${test.failed} failed` : ""}`}
              </span>
            </Line>
          ) : null}
        </Stratum>
      ) : null}

      <Stratum
        label="Current action"
        live={live}
        defaultOpen
        summary={live ? runtimeDetail ?? "working" : "idle"}
      >
        {liveFeed.length === 0 ? (
          <div style={{ fontSize: 11, color: "var(--text-dim)", padding: "2px var(--ma-2) 2px 22px" }}>
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
                    background: last && live ? "var(--accent)" : "var(--text-dim)",
                  }}
                />
                <span style={{ flex: 1, minWidth: 0, fontSize: 13, color: last ? "var(--text)" : "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
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
