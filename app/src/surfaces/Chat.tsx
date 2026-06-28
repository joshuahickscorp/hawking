import { useEffect, useLayoutEffect, useReducer, useRef, useState } from "react";
import { sendIntent } from "../ipc";
import { useStore } from "../store";
import { intent } from "../wire";
import { Display, Volume } from "../ui";
import type { DiffChip, DiffChipPatch, PlanPatch, PlanStep } from "./chat/parts";
import { DiffChipRow, InlineGate, PlanCard, ToolChipRow } from "./chat/structure";
import { SteerBar } from "./chat/SteerBar";

const SESSION = "ses_mock0000000000000000000";
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
  const plan = useStore((s) => s.projections.plan as PlanPatch | undefined);
  const diffPatch = useStore((s) => s.projections.diff_chip as DiffChipPatch | undefined);
  const chips: DiffChip[] = diffPatch?.chips ?? [];

  const [text, setText] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const live = STEERABLE.has(runPhase);
  const runId = activeRunId ?? plan?.run_id ?? chips[0]?.run_id ?? "";

  const streamingLen = messages.reduce((n, m) => n + (m.streaming ? m.text.length : 0), 0);
  useRafGovernor(streamingLen);

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
    <div className="chat-shell">
      <div ref={scrollRef} className="chat-scroll">
        {empty ? (
          <div className="chat-empty">
            <Display>Describe the work</Display>
            <p className="t-body" style={{ color: "var(--text-2)", marginTop: "var(--ma-6)" }}>
              Start with a task, file, or question
            </p>
          </div>
        ) : (
          <div className="message-list">
            {messages.map((m) => (
              <Message key={m.id} role={m.role} text={m.text} streaming={m.streaming} />
            ))}
            {plan ? <PlanCard plan={plan} onApprove={approvePlan} onEditStep={editStep} onReorder={reorder} /> : null}
            <ToolChipRow tools={tools} />
            <DiffChipRow chips={chips} onOpen={openDiff} onAccept={acceptDiff} onReject={rejectDiff} />
            {gate ? <InlineGate gate={gate.gate} message={gate.message} onApprove={approveGate} onDismiss={dismissGate} /> : null}
          </div>
        )}
      </div>

      <div className="composer-zone">
        {live ? <SteerBar phase={runPhase} onRedirect={steer} onPause={pause} onResume={resume} onCancel={cancel} /> : null}
        <Composer text={text} onText={setText} onSubmit={submit} ready={runtimeReady} live={live} />
      </div>
    </div>
  );
}

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
  const label = (
    <div className="t-label" style={{ marginBottom: "var(--ma-2)" }}>
      {isUser ? "You" : "Agent"}
    </div>
  );

  if (!isUser) {
    return (
      <div className="message">
        {label}
        <div className="t-body" style={{ color: "var(--text-1)", whiteSpace: "pre-wrap" }}>
          {text}
          {streaming ? <span aria-hidden className="stream-cursor alive" /> : null}
        </div>
      </div>
    );
  }

  return (
    <div className="message message--user">
      <Volume raised className="message__bubble">
        {label}
        <div className="t-body">{text}</div>
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

  const placeholder = !ready ? "Runtime not ready" : live ? "Queue a turn" : "Message the agent";
  const armed = !!text.trim() && ready;

  return (
    <Volume raised className="composer">
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
      />
      <button
        className="text-button"
        onClick={onSubmit}
        disabled={!armed}
        style={{
          color: armed ? "var(--light)" : "var(--text-3)",
          boxShadow: armed ? "var(--hairline-strong), var(--light-bloom), var(--inner-glow)" : undefined,
        }}
      >
        {live ? "Queue" : "Send"}
      </button>
    </Volume>
  );
}
