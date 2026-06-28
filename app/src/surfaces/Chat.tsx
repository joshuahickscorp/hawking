/*
  Chat.tsx: the AI Chat surface (01-surfaces §D1.3), the conversation chamber. Watch and steer
  the agent's reasoning. The calmest, most spacious surface: a single readable column down the
  center of the void, content capped near 700px, generous air on either side, no chat-app chrome.

   1. Transcript: user + assistant turns in Geist Mono telemetry voice; the streaming assistant text
      carries a faint light cusp at its leading edge (the light entering the dark, no spinner). A
      render-rate governor flushes one React commit per animation frame so a fast stream never
      thrashes the paint.
   2. Composer: SubmitTurn on Enter, a .volume--raised input pinned at the bottom in its own air.
      While a run is active, the persistent SteerBar redirects mid-flight (Custom:redirect_run) and
      exposes Cancel/Pause/Resume (CancelRun/PauseRun/ResumeRun). The agent is interruptible.
   3. Inline structure in the stream: the PlanCard (ordered steps + status, approve/edit/reorder), calm
      ToolChips (tool_progress, no churn), DiffChips (a produced diff -> opens the hunk review), and the
      SecurityGate as a lit inline approval (the gate capsule).

  Harvest: the plan-act + per-step chat UX (Cline/OpenCode), re-housed into the v3 doctrine (grayscale
  concrete volumes floating in void, light as the only accent, Geist Mono, glyph+label markers,
  real-work-as-progress).

  Sends: SubmitTurn, PauseRun/ResumeRun/CancelRun, AcceptDiff/RejectDiff, Custom(redirect_run,
  approve_plan, edit_plan_step, reorder_plan, approve_gate). Consumes: token_batch, projection_patch
  (turn/plan/diff_chip), tool_progress, security_gate (all folded by the store).
*/
import { useEffect, useLayoutEffect, useReducer, useRef, useState } from "react";
import { sendIntent } from "../ipc";
import { useStore } from "../store";
import { intent } from "../wire";
import { Display, Volume } from "../ui";
import type { DiffChip, DiffChipPatch, PlanPatch, PlanStep } from "./chat/parts";
import { DiffChipRow, InlineGate, PlanCard, ToolChipRow } from "./chat/structure";
import { SteerBar } from "./chat/SteerBar";

// The conversational column: capped near 700px and centered, the doctrine's readable measure.
const COLUMN = 700;

const SESSION = "ses_mock0000000000000000000";

// A run is live (and so steerable) for any non-terminal phase.
const STEERABLE = new Set(["planning", "executing", "paused", "awaiting"]);

export function Chat() {
  const messages = useStore((s) => s.messages);
  const tools = useStore((s) => s.tools);
  const gate = useStore((s) => s.gate);
  const dismissGate = useStore((s) => s.dismissGate);
  const runtimeReady = useStore((s) => s.runtimeStatus === "ready");
  const runPhase = useStore((s) => s.runPhase);
  const activeRunId = useStore((s) => s.activeRunId);
  const pushUserMessage = useStore((s) => s.pushUserMessage);
  const pushNotice = useStore((s) => s.pushNotice);

  // Plan + diff-chip state arrive as projection patches the store folds into its generic bag.
  const plan = useStore((s) => s.projections.plan as PlanPatch | undefined);
  const diffPatch = useStore((s) => s.projections.diff_chip as DiffChipPatch | undefined);
  const chips: DiffChip[] = diffPatch?.chips ?? [];

  const [text, setText] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const live = STEERABLE.has(runPhase);
  // The run_id steer/diff intents target: the active run, or whatever a fresh patch carried.
  const runId = activeRunId ?? plan?.run_id ?? chips[0]?.run_id ?? "";

  // Render-rate governor (D1.3): coalesce store churn into one commit per animation frame so a
  // 120 tok/s token stream never thrashes React. The store is already the source of truth; this
  // only paces the paint. We tick a frame whenever the streamed text length changes.
  const streamingLen = messages.reduce((n, m) => n + (m.streaming ? m.text.length : 0), 0);
  useRafGovernor(streamingLen);

  // Keep the transcript pinned to the streaming edge.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, tools.length, chips.length, gate]);

  const submit = async () => {
    const t = text.trim();
    if (!t) return;
    pushUserMessage(t);
    setText("");
    const ack = await sendIntent(intent.submitTurn(SESSION, t));
    if (!ack.accepted) pushNotice({ kind: "error", code: "rejected", message: ack.message ?? "turn rejected" });
  };

  // ---- steer + approve verbs, each a single Intent up the seam ----
  const steer = (steerText: string) => void sendIntent(intent.custom("redirect_run", { run_id: runId, text: steerText }));
  const pause = () => void sendIntent(intent.pauseRun(runId));
  const resume = () => void sendIntent(intent.resumeRun(runId));
  const cancel = () => void sendIntent(intent.cancelRun(runId));
  const approvePlan = () => void sendIntent(intent.custom("approve_plan", { run_id: runId }));
  const editStep = (step: PlanStep, title: string) =>
    void sendIntent(intent.custom("edit_plan_step", { run_id: runId, step_id: step.id, title }));
  const reorder = (from: number, to: number) => void sendIntent(intent.custom("reorder_plan", { run_id: runId, from, to }));
  const openDiff = (c: DiffChip) => void sendIntent(intent.openFile(c.path));
  const acceptDiff = (c: DiffChip) => void sendIntent(intent.acceptDiff(c.run_id ?? runId, c.diff_id));
  const rejectDiff = (c: DiffChip) => void sendIntent(intent.rejectDiff(c.run_id ?? runId, c.diff_id));
  const approveGate = () => {
    if (gate) void sendIntent(intent.custom("approve_gate", { gate: gate.gate }));
    dismissGate();
  };

  const empty = messages.length === 0 && tools.length === 0 && !plan && chips.length === 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div ref={scrollRef} style={{ flex: 1, overflowY: "auto", padding: "0 var(--ma-8)" }}>
        {empty ? (
          // The empty chamber points to the first action, in the doctrine's flight-log voice.
          <div style={{ maxWidth: COLUMN, margin: "0 auto", paddingTop: "18vh", textAlign: "center" }}>
            <Display>Open the box.</Display>
            <p className="t-body" style={{ color: "var(--text-2)", marginTop: "var(--ma-6)" }}>
              Ask the agent to do work. You will see exactly what it reads and runs.
            </p>
          </div>
        ) : (
          // The conversation column: capped near 700px, centered, --ma-18 top, --ma-14 between turns.
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "var(--ma-14)",
              maxWidth: COLUMN,
              margin: "0 auto",
              paddingTop: "var(--ma-18)",
              paddingBottom: "var(--ma-10)",
            }}
          >
            {messages.map((m) => (
              <Message key={m.id} role={m.role} text={m.text} streaming={m.streaming} />
            ))}

            {/* Inline structure, threaded after the stream: the plan, the calm tool feed, the diffs. */}
            {plan ? (
              <PlanCard plan={plan} onApprove={approvePlan} onEditStep={editStep} onReorder={reorder} />
            ) : null}
            <ToolChipRow tools={tools} />
            <DiffChipRow chips={chips} onOpen={openDiff} onAccept={acceptDiff} onReject={rejectDiff} />

            {/* The SecurityGate as a lit inline approval (the doctrine wants gate visible in-flow). */}
            {gate ? (
              <InlineGate gate={gate.gate} message={gate.message} onApprove={approveGate} onDismiss={dismissGate} />
            ) : null}
          </div>
        )}
      </div>

      {/* The steer field floats at the bottom in its own --ma-8 air; nothing touches an edge, no top rule. */}
      <div style={{ padding: "var(--ma-4) var(--ma-8) var(--ma-8)" }}>
        {/* While a run is active: the persistent steer bar above the composer (interruptible agent). */}
        {live ? (
          <SteerBar phase={runPhase} onRedirect={steer} onPause={pause} onResume={resume} onCancel={cancel} />
        ) : null}
        <Composer text={text} onText={setText} onSubmit={submit} ready={runtimeReady} live={live} />
      </div>
    </div>
  );
}

// A frame-paced re-render: bump a tiny reducer once per animation frame while `signal` keeps changing.
// This is the paint governor, not a data path; the store already holds truth.
function useRafGovernor(signal: number) {
  const [, tick] = useReducer((n: number) => n + 1, 0);
  const last = useRef(signal);
  useEffect(() => {
    if (last.current === signal) return;
    last.current = signal;
    const id = requestAnimationFrame(() => tick());
    return () => cancelAnimationFrame(id);
  }, [signal]);
}

function Message({ role, text, streaming }: { role: "user" | "assistant"; text: string; streaming: boolean }) {
  const isUser = role === "user";

  // The speaker label: the quietest mark, in --mute, naming who is talking (flight-log voice).
  const speaker = (
    <div className="t-label" style={{ marginBottom: "var(--ma-2)" }}>
      {isUser ? "You" : "Agent"}
    </div>
  );

  // Agent prose: held in the open void as readable body copy, not boxed. The streaming leading edge
  // carries a faint --light-soft cusp (the light entering the dark), never a spinner.
  if (!isUser) {
    return (
      <div>
        {speaker}
        <div className="t-body" style={{ color: "var(--text-1)", whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
          {text}
          {streaming ? (
            <span
              aria-hidden
              style={{
                display: "inline-block",
                width: "0.5em",
                height: "1.05em",
                marginLeft: 2,
                verticalAlign: "-0.15em",
                borderRadius: 1,
                background: "var(--light-soft)",
                boxShadow: "var(--light-bloom)",
                animation: "breathe var(--breathe) var(--ease) infinite",
              }}
            />
          ) : null}
        </div>
      </div>
    );
  }

  // The user's turn: a quiet raised-concrete slab, set right, never edge to edge.
  return (
    <div style={{ display: "flex", justifyContent: "flex-end" }}>
      <Volume
        raised
        pad="var(--ma-3) var(--ma-4)"
        style={{ maxWidth: "82%", color: "var(--text-1)", whiteSpace: "pre-wrap" }}
      >
        {speaker}
        <div className="t-body" style={{ lineHeight: 1.6 }}>{text}</div>
      </Volume>
    </div>
  );
}

function Composer({
  text,
  onText,
  onSubmit,
  ready,
  live,
}: {
  text: string;
  onText: (v: string) => void;
  onSubmit: () => void;
  ready: boolean;
  live: boolean;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 180) + "px";
  }, [text]);

  // While a run is live, Enter queues the turn behind the active one (Custom:queue_turn semantics);
  // otherwise it submits a fresh turn. The copy on the control says exactly which.
  const placeholder = !ready ? "Runtime not ready" : live ? "Queue a turn (or steer above)" : "Message the agent";
  const armed = !!text.trim() && ready;

  return (
    <Volume
      raised
      pad="var(--ma-3) var(--ma-4)"
      style={{ maxWidth: COLUMN, margin: "0 auto", display: "flex", alignItems: "flex-end", gap: "var(--ma-3)" }}
    >
      <textarea
        ref={ref}
        value={text}
        onChange={(e) => onText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onSubmit();
          }
        }}
        rows={1}
        placeholder={placeholder}
        disabled={!ready}
        className="t-body"
        style={{
          flex: 1,
          resize: "none",
          background: "transparent",
          border: "none",
          outline: "none",
          color: "var(--text-1)",
          font: "inherit",
          lineHeight: 1.6,
          padding: "var(--ma-1) 0",
        }}
      />
      {/* The send affordance catches the light only when armed; otherwise it is quiet concrete. */}
      <button
        onClick={onSubmit}
        disabled={!armed}
        style={{
          padding: "var(--ma-2) var(--ma-4)",
          borderRadius: "var(--radius)",
          fontSize: "13px",
          fontWeight: 500,
          color: armed ? "var(--light)" : "var(--text-3)",
          background: "var(--concrete-4)",
          boxShadow: armed ? "var(--hairline-strong), var(--light-bloom)" : "var(--hairline)",
          transition: "color var(--dur) var(--ease), box-shadow var(--dur) var(--ease)",
        }}
      >
        {live ? "Queue" : "Send"}
      </button>
    </Volume>
  );
}
