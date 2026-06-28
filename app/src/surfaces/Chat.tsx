/*
  Chat.tsx: the AI Chat surface (01-surfaces §D1.3). Watch and steer the agent's reasoning.
  The full conversational surface over the live UiEvent stream:

   1. Transcript: user + assistant turns in Geist Mono telemetry voice; the streaming assistant text
      wears the gold leading-edge cursor (the radiation leaking out, no spinner). A render-rate
      governor flushes one React commit per animation frame so a fast stream never thrashes the paint.
   2. Composer: SubmitTurn on Enter. While a run is active, the persistent SteerBar redirects mid-flight
      (Custom:redirect_run) and exposes Cancel/Pause/Resume (CancelRun/PauseRun/ResumeRun). Interruptible.
   3. Inline structure in the stream: the PlanCard (ordered steps + status, approve/edit/reorder), calm
      ToolChips (tool_progress, no churn), DiffChips (a produced diff -> opens the hunk review), and the
      SecurityGate as a lit inline approval.

  Harvest: the plan-act + per-step chat UX (Cline/OpenCode), re-housed into the doctrine (near-black
  material panels, gold rim-light, Geist Mono, shape+label markers, real-work-as-progress).

  Sends: SubmitTurn, PauseRun/ResumeRun/CancelRun, AcceptDiff/RejectDiff, Custom(redirect_run,
  approve_plan, edit_plan_step, reorder_plan, approve_gate). Consumes: token_batch, projection_patch
  (turn/plan/diff_chip), tool_progress, security_gate (all folded by the store).
*/
import { useEffect, useLayoutEffect, useReducer, useRef, useState } from "react";
import { sendIntent } from "../ipc";
import { useStore } from "../store";
import { intent } from "../wire";
import { Display, Panel, RadiationEdge } from "../ui";
import type { DiffChip, DiffChipPatch, PlanPatch, PlanStep } from "./chat/parts";
import { DiffChipRow, InlineGate, PlanCard, ToolChipRow } from "./chat/structure";
import { SteerBar } from "./chat/SteerBar";

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
      <div ref={scrollRef} style={{ flex: 1, overflowY: "auto", padding: "var(--s5)" }}>
        {empty ? (
          <div style={{ maxWidth: 560, margin: "10vh auto 0", textAlign: "center" }}>
            <Display size={44}>Open the box.</Display>
            <p style={{ color: "var(--text-mid)", marginTop: "var(--s4)", fontSize: "var(--text-sm)" }}>
              Ask the agent to do work. You will see exactly what it reads and runs.
            </p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--s4)", maxWidth: 720, margin: "0 auto" }}>
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

      <div style={{ padding: "var(--s4)", borderTop: "1px solid var(--rim)" }}>
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
  return (
    <div style={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start" }}>
      <Panel
        active={streaming}
        pad="var(--s3) var(--s4)"
        style={{
          maxWidth: "84%",
          background: isUser ? "var(--surface-1)" : "var(--surface-0)",
          color: "var(--text-hi)",
          whiteSpace: "pre-wrap",
          lineHeight: "var(--leading-ui)",
        }}
      >
        <div style={{ fontSize: "var(--text-xs)", color: "var(--text-low)", marginBottom: "var(--s1)", letterSpacing: "0.04em" }}>
          {isUser ? "you" : "agent"}
        </div>
        {text}
        {/* the gold leading edge of the stream: the radiation leaking out (no spinner) */}
        {streaming ? (
          <span
            aria-hidden
            style={{
              display: "inline-block",
              width: 7,
              height: "1em",
              marginLeft: 3,
              verticalAlign: "-2px",
              background: "var(--radiation)",
              boxShadow: "0 0 8px 0 var(--radiation-bloom)",
              animation: "radiation-breathe 1.1s ease-in-out infinite",
            }}
          />
        ) : null}
      </Panel>
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

  return (
    <RadiationEdge mode="breathe" style={{ maxWidth: 720, margin: "0 auto" }}>
      <Panel pad="var(--s2) var(--s3)" style={{ display: "flex", alignItems: "flex-end", gap: "var(--s2)" }}>
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
          style={{
            flex: 1,
            resize: "none",
            background: "transparent",
            border: "none",
            outline: "none",
            color: "var(--text-hi)",
            font: "inherit",
            lineHeight: "var(--leading-ui)",
            padding: "var(--s2) 0",
          }}
        />
        <button
          onClick={onSubmit}
          disabled={!ready || !text.trim()}
          style={{
            padding: "6px 14px",
            borderRadius: "var(--radius)",
            color: text.trim() && ready ? "var(--void)" : "var(--text-low)",
            background: text.trim() && ready ? "var(--radiation)" : "var(--surface-2)",
            boxShadow: text.trim() && ready ? "0 0 14px -4px var(--radiation-bloom)" : "inset 0 0 0 1px var(--rim)",
            fontSize: "var(--text-sm)",
          }}
        >
          {live ? "Queue" : "Send"}
        </button>
      </Panel>
    </RadiationEdge>
  );
}
