/*
  ContextStack.tsx: the agent's context, opened as a RECEIPT.

  Every row here is something the host actually reported: what went into the window and why, what it
  cost, what was left out and for what reason, which repo instructions loaded, which memory the
  compiler drew from, and the blake3 address of each span. The panel makes no claim about model
  quality; it only shows the manifest (crates/hawking-context ContextManifest, delivered over the
  projection_patch{context_manifest} stream the host publishes per turn).

  MOUNTED as the Context face of the conversation side panel (surfaces/home/ChatPanel.tsx). It was
  imported by no module at all, so every control on it rendered nowhere.

  Actions resolve through the ONE command spine (src/generated/command_catalog.json), so snapshot is
  a real integrity-verified checkpoint, fork is a real session fork, and the memory controls are the
  real outcome-governed memory domain. Nothing here toasts a success the host did not give.

  Retired in the consolidation pass (docs/hide-impl/consolidation/HIDE_CONSOLIDATION_DECISIONS.md):
    save skill + the three hardcoded load-skill rows -> no skill store exists in the catalog (3.2).
    pin / mute / evict via pin_span / unpin_span      -> the names have NO host handler at all; the
                                                         real durable write is the memory domain (4).
    the model row's switch_profile button             -> an empty-payload duplicate of the one model
                                                         chooser, with no host handler (3.4).
    the "constant-size memcpy" fork label             -> the host replays the prefix under a new
                                                         session id; it is not a memcpy (3.3).
*/
import { useStore } from "../store";
import { ActionMark, HardwareToggle, Line, NoteField, OkMark, Stratum } from "./contextstack/parts";
import {
  asReceipt,
  excludedSources,
  includedSources,
  memoryRows,
  modelSummary,
  notePlan,
  plan,
  receiptTotals,
  sessionScope,
  spanKey,
  staleWarnings,
  useActions,
  useSteer,
  type ActionState,
} from "./contextstack/state";

const fileName = (p: string) => p.split("/").pop() ?? p;

// Compact token count for the ambient context line (8192 -> "8K", 4_000_000 -> "4M").
const fmtTok = (n: number) =>
  n >= 1_000_000 ? `${(n / 1_000_000).toFixed(n % 1_000_000 ? 1 : 0)}M` : n >= 1000 ? `${Math.round(n / 1000)}K` : `${n}`;
const WATERMARK_WORD: Record<string, string> = {
  normal: "cached",
  soft: "ready to compact",
  warn: "recency decay",
  critical: "compacting",
};

/* A control in the state row / a section header: same chrome as before, plus honest state.
   The title used to append "(also in the command palette)" to every enabled button. This component
   does not know its command id, and the claim was false for the fork control: fork_session needs an
   event id, so it is not a palette row at all. The palette lists what the palette can run. */
function ActBtn({
  label,
  name,
  title,
  onClick,
  disabled = false,
  status = "idle",
  message,
}: {
  label: string;
  name: string;
  title: string;
  onClick: () => void;
  disabled?: boolean;
  status?: ActionState;
  message?: string;
}) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 2 }}>
      <ActionMark state={status} message={message} />
      <button
        className="ctx-state__btn"
        title={title}
        aria-label={name}
        aria-disabled={disabled}
        aria-busy={status === "pending"}
        disabled={disabled}
        onClick={onClick}
        style={{ opacity: disabled ? 0.45 : 1, cursor: disabled ? "not-allowed" : "pointer" }}
      >
        {label}
      </button>
    </span>
  );
}

export function ContextStack() {
  const manifest = useStore((s) => s.manifest);
  const liveFeed = useStore((s) => s.tools); // tool_progress stream = the agent's real moves
  const runPhase = useStore((s) => s.runPhase);
  const runtimeDetail = useStore((s) => s.runtimeDetail);
  const sessionId = useStore((s) => s.sessionId);
  const pushNotice = useStore((s) => s.pushNotice);
  const build = useStore((s) => s.projections.build) as { ok?: boolean; summary?: string } | undefined;
  const test = useStore((s) => s.projections.test) as
    | { passed?: number; failed?: number; total?: number; summary?: string }
    | undefined;

  const live = runPhase === "executing" || runPhase === "planning";
  const steer = useSteer();
  // Every refusal is surfaced, never swallowed: the row marks failed AND the status bar says why.
  const acts = useActions((message) => pushNotice({ kind: "error", code: "context", message }));

  // The manifest is the one the HOST publishes per turn (projection_patch{context_manifest}, folded
  // into store.manifest). This panel used to also call the context connector's `compile` for a
  // second copy: that arm upserts the durable memory store, so it is a WRITE on a read-only route
  // and the transport now refuses it (connectors.rs CONNECTOR_READ_METHODS). Asking the engine to
  // recompile just to draw a receipt was never this surface's job anyway.
  const m = manifest;
  const r = asReceipt(m);
  const included = includedSources(r);
  const excluded = excludedSources(r);
  const warnings = staleWarnings(r);
  const memories = memoryRows(r);
  const model = modelSummary(r);
  const totals = receiptTotals(r);
  const scope = sessionScope(sessionId);

  // The fork boundary: the newest recorded step, the same point StateTimeline's "fork from here"
  // uses, so the two gestures mean exactly one thing. With no recorded step there is no boundary
  // and the control says so instead of sending a fork the host cannot resolve.
  const atEvent = liveFeed.length ? liveFeed[liveFeed.length - 1].call_id : "";
  const lastUserText = [...useStore.getState().messages].reverse().find((x) => x.role === "user")?.text;
  // A memory marked wrong is the supersede target: mark it, then type the correction.
  const supersedeTarget = memories.find((x) => x.id && steer.on(spanKey("mem", x.key), "evict"))?.id;

  return (
    <aside aria-label="Context Stack" className="ctx-stack">
      {/* The Context Stack is the agent's STATE, opened: a serializable object, not a re-prompt.
          Snapshot seals an integrity-verified checkpoint; fork branches a new session from the last
          recorded step. Both are catalog commands, so the palette offers the same two actions. */}
      <div className="ctx-state">
        <span className={"ctx-state__dot" + (live ? " ctx-state__dot--live" : "")} aria-hidden />
        <span className="ctx-state__title">state / {live ? "warm" : "ready"}</span>
        <span className="ctx-state__meta">
          {totals.stateBytes ? `${Math.round(totals.stateBytes / 1e6)} MB serializable` : "size not reported"}
        </span>
        <div className="ctx-state__actions">
          <ActBtn
            label="snapshot"
            name="Create checkpoint"
            title="Seal an integrity-verified checkpoint (blake3) you can restore or compare against"
            status={acts.stateOf("snapshot")}
            message={acts.messageOf("snapshot")}
            onClick={() =>
              void acts.run("snapshot", plan.snapshot(sessionId, (lastUserText ?? "").trim().slice(0, 60) || "checkpoint"))
            }
          />
          <ActBtn
            label="fork"
            name="Fork session from the last recorded step"
            title={
              atEvent
                ? "Fork a new session: the host replays this session's history up to the last recorded step under a fresh session id"
                : "No recorded step yet, so there is no point to fork from"
            }
            disabled={!atEvent}
            status={acts.stateOf("fork")}
            message={acts.messageOf("fork")}
            onClick={() => void acts.run("fork", plan.fork(sessionId, atEvent))}
          />
        </div>
      </div>

      {/* Spine A: the live, measured context ceiling (native x the .tq multiplier, read live, never a
          constant). Ambient and quiet, no meter: abundance shown as a number that is simply large. */}
      {m && (m.ctx_len_effective || m.live) ? (
        <Line title="Live context ceiling: native x the measured .tq multiplier, read live from the engine. Local: never billed, never truncated.">
          <span style={{ color: "var(--text-3)" }}>context</span>
          <span style={{ color: "var(--text-2)", minWidth: 0 }}>
            {fmtTok((m.live?.effective_ceiling_tokens ?? m.ctx_len_effective) || 0)} effective
            {m.tq_multiplier && m.tq_multiplier > 1 ? `   ${m.tq_multiplier.toFixed(1)}x .tq` : ""}
            {totals.usedTokens ? `   ${fmtTok(totals.usedTokens)} used` : ""}
            {m.recurrent_state_bytes ? `   ${Math.round(m.recurrent_state_bytes / 1e6)} MB state` : ""}
            {typeof m.live?.recall_fidelity === "number" ? `   recall ${Math.round(m.live.recall_fidelity * 100)}%` : ""}
          </span>
          <span
            style={{
              marginLeft: "auto",
              color: m.live && m.live.watermark !== "normal" ? "var(--light)" : "var(--text-3)",
            }}
          >
            {WATERMARK_WORD[m.live?.watermark ?? "normal"]}
          </span>
        </Line>
      ) : null}

      {!m && liveFeed.length === 0 ? (
        // Says WHY, not just that it is empty: every publisher of the context_manifest projection is
        // inside the turn path, so on a host with no served model this panel has no data source at
        // all and a bare "No context yet" reads as a load that never finished.
        <div className="sidebar__empty">
          No context manifest yet. It is published by a running turn, so it stays empty until a model runs.
        </div>
      ) : null}

      {model ? (
        <Stratum label="Model" summary={model.id}>
          <Line
            title={[
              model.tokenizerSig ? `tokenizer ${model.tokenizerSig}` : null,
              model.native ? `native ${model.native.toLocaleString()} tokens` : null,
              model.effective ? `effective ${model.effective.toLocaleString()} tokens` : null,
            ]
              .filter(Boolean)
              .join(" / ")}
          >
            <span style={{ flex: 1, minWidth: 0, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {model.id}
              {model.arch ? ` / ${model.arch}` : ""}
            </span>
            {model.profile ? (
              <span style={{ flex: "0 0 auto", fontSize: "var(--fs-label)", color: "var(--text-dim)" }}>{model.profile}</span>
            ) : null}
          </Line>
        </Stratum>
      ) : null}

      {/* The receipt, part 1: what the window HELD, why it held it, and what it cost. Instruction
          spans (repo CLAUDE.md style config the backend loaded) read as ordinary included sources,
          because that is exactly what they are. */}
      {included.length ? (
        <Stratum
          label="Included"
          count={included.length}
          summary={`${totals.includedTokens.toLocaleString()} tokens`}
          defaultOpen
        >
          {totals.instructionFiles.length || totals.compatFiles.length ? (
            <Line
              title={[...new Set([...totals.instructionFiles, ...totals.compatFiles])].join("\n")}
            >
              <span style={{ flex: 1, minWidth: 0, fontSize: "var(--fs-label)", color: "var(--text-dim)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                instructions loaded: {totals.instructionFiles.map(fileName).join(", ") || "none"}
                {totals.compatFiles.length ? ` / compat: ${totals.compatFiles.map(fileName).join(", ")}` : ""}
              </span>
            </Line>
          ) : null}
          {included.map((row) => {
            const detail = [
              row.why,
              row.hash ? `hash ${row.hash}` : null,
              row.files.length ? `from ${row.files.join(", ")}` : null,
              row.attachment ? `attachment ${row.attachment}` : null,
              row.banked ? "kv reused, not re-prefilled" : null,
            ]
              .filter(Boolean)
              .join("\n");
            const body = (
              <>
                <span style={{ flex: 1, minWidth: 0, fontSize: "var(--fs-ui)", color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {row.title}
                </span>
                <span style={{ flex: "0 0 auto", fontSize: "var(--fs-label)", color: "var(--text-dim)" }}>
                  {row.instructions ? "instructions" : row.kind}
                  {row.attachment ? " / attached" : ""}
                  {row.banked ? " / reused" : ""}
                </span>
                {row.tokens ? (
                  <span style={{ flex: "0 0 auto", fontSize: "var(--fs-label)", color: "var(--text-dim)" }}>
                    {row.tokens.toLocaleString()}
                  </span>
                ) : null}
              </>
            );
            return row.path ? (
              <Line
                key={row.key}
                title={`open ${row.path}\n${detail}`}
                onClick={() => void acts.run(`open:${row.key}`, plan.open(row.path as string))}
              >
                {body}
              </Line>
            ) : (
              <Line key={row.key} title={detail}>
                {body}
              </Line>
            );
          })}
        </Stratum>
      ) : null}

      {r.tools?.length ? (
        <Stratum
          label="Tools"
          count={r.tools.length}
          summary={r.tools.filter((t) => t.ok).length + " ok"}
        >
          {r.tools.map((t, i) => (
            <Line key={`${t.name}:${i}`}>
              <OkMark ok={t.ok === true} />
              <span style={{ flex: 1, minWidth: 0, fontSize: "var(--fs-ui)", color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {t.name}
              </span>
            </Line>
          ))}
        </Stratum>
      ) : null}

      {/* The receipt, part 2: the durable, outcome-governed memory the compiler drew from. Marking a
          claim wrong records a real negative outcome (it self-quarantines); typing while one is
          marked replaces it and keeps its history; the header re-checks citations against disk. */}
      {memories.length ? (
        <Stratum
          label="Memory"
          count={memories.length}
          summary={memories[0]?.claim}
          trailing={
            <ActBtn
              label="revalidate"
              name="Revalidate memory for this session"
              title="Re-check every memory citation in this session against the repo on disk"
              status={acts.stateOf("revalidate")}
              message={acts.messageOf("revalidate")}
              onClick={() => void acts.run("revalidate", plan.revalidate(scope))}
            />
          }
        >
          {memories.map((mem) => {
            const id = spanKey("mem", mem.key);
            const wrong = steer.on(id, "evict");
            return (
              <Line
                key={mem.key}
                title={[
                  mem.id ? `record ${mem.id}` : "no durable record id in this projection",
                  mem.citations.length ? `cites ${mem.citations.join(", ")}` : null,
                  mem.status ? `status ${mem.status}` : null,
                ]
                  .filter(Boolean)
                  .join("\n")}
              >
                <span style={{ flex: 1, minWidth: 0, fontSize: "var(--fs-ui)", color: wrong ? "var(--text-dim)" : "var(--text)", textDecoration: wrong ? "line-through" : "none", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {mem.claim}
                </span>
                {mem.status ? (
                  <span style={{ flex: "0 0 auto", fontSize: "var(--fs-label)", color: "var(--text-dim)" }}>{mem.status}</span>
                ) : null}
                {typeof mem.score === "number" ? (
                  <span style={{ flex: "0 0 auto", fontSize: "var(--fs-label)", color: "var(--text-dim)" }}>{mem.score.toFixed(1)}</span>
                ) : null}
                <HardwareToggle
                  label="wrong"
                  name={`Mark the memory "${mem.claim}" wrong`}
                  tone="bad"
                  on={wrong}
                  disabled={!mem.id}
                  status={acts.stateOf(`memory:${mem.key}`)}
                  message={acts.messageOf(`memory:${mem.key}`)}
                  title={
                    !mem.id
                      ? "This memory has no durable record id in the projection, so no outcome can be recorded against it"
                      : wrong
                        ? "Record that this memory held after all"
                        : "Record that this memory was wrong, so it self-quarantines. Type a replacement below to supersede it"
                  }
                  onToggle={() => {
                    if (!mem.id) return;
                    const nowWrong = steer.toggle(id, "evict");
                    void acts.run(`memory:${mem.key}`, plan.outcome(mem.id, !nowWrong));
                  }}
                />
              </Line>
            );
          })}
          <div>
            <NoteField
              value={steer.noteOn("memory")}
              placeholder={supersedeTarget ? "replace the memory marked wrong" : "remember this (durable note)"}
              onCommit={(text) => {
                steer.setNote("memory", text);
                void acts.run("memory:note", notePlan(text, scope, supersedeTarget));
              }}
            />
          </div>
        </Stratum>
      ) : null}

      {/* The receipt, part 3: what was LEFT OUT and why, with the stale and conflicting reads called
          out by word (never pigment alone) so a thin context is never silently thin. */}
      {excluded.length || warnings.length ? (
        <Stratum
          label="Excluded"
          count={excluded.length}
          summary={warnings.length ? `${warnings.length} to check` : excluded[0]?.reason}
        >
          {warnings.map((w) => (
            <Line key={w.key} title={w.text}>
              <span aria-label={w.kind === "stale" ? "stale" : "conflict"} role="img" style={{ flex: "0 0 auto", width: 14, textAlign: "center", color: "var(--red)" }}>
                !
              </span>
              <span style={{ flex: 1, minWidth: 0, fontSize: "var(--fs-ui)", color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {w.kind}: {w.text}
              </span>
            </Line>
          ))}
          {excluded.map((d) => (
            <Line key={d.key} title={`${d.kind} / ${d.reason}`}>
              <span style={{ flex: 1, minWidth: 0, fontSize: "var(--fs-ui)", color: "var(--text-dim)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {d.title}
              </span>
              <span style={{ flex: "0 0 auto", fontSize: "var(--fs-label)", color: "var(--text-dim)" }}>{d.reason}</span>
              {d.tokens ? (
                <span style={{ flex: "0 0 auto", fontSize: "var(--fs-label)", color: "var(--text-dim)" }}>{d.tokens.toLocaleString()}</span>
              ) : null}
            </Line>
          ))}
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
              <span style={{ flex: 1, minWidth: 0, fontSize: "var(--fs-ui)", color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{build.summary ?? "build"}</span>
            </Line>
          ) : null}
          {test ? (
            <Line>
              <OkMark ok={(test.failed ?? 0) === 0} />
              <span style={{ flex: 1, minWidth: 0, fontSize: "var(--fs-ui)", color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
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
          <div style={{ fontSize: "var(--fs-label)", color: "var(--text-dim)", padding: "2px var(--ma-2) 2px 22px" }}>
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
                <span style={{ flex: 1, minWidth: 0, fontSize: "var(--fs-ui)", color: last ? "var(--text)" : "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
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
