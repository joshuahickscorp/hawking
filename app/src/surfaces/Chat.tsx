import { useEffect, useLayoutEffect, useMemo, useReducer, useRef, useState } from "react";
import { marked } from "marked";
import DOMPurify from "dompurify";
import { sendIntent } from "../ipc";
import { useStore } from "../store";
import { intent } from "../wire";
import { Display } from "../ui";
import { Icon } from "../shell/icons";
import { Radiate } from "../shell/Radiate";
import type { DiffChip, DiffChipPatch, PlanPatch, PlanStep } from "./chat/parts";
import { DiffChipRow, InlineGate, PlanCard, ToolChipRow } from "./chat/structure";
import { SteerBar } from "./chat/SteerBar";

const STEERABLE = new Set(["planning", "executing", "paused", "awaiting"]);

// The empty-composer prompt. Default terse (flight-log voice); flip DREAM_BIG to restore "dream big".
const DREAM_BIG = false;
const IDLE_PLACEHOLDER = DREAM_BIG ? "dream big" : "Describe a task";

// Oracle ladder for the radiate ring: how many distinct verify stages have reported this run. Returns
// undefined when no laddered progress is present, so the ring degrades to an indeterminate sweep.
const LADDER = ["build", "typecheck", "test", "lint"];
function oracleStage(tools: { message: string }[]): number | undefined {
  const seen = new Set<number>();
  for (const t of tools) {
    const m = t.message.toLowerCase();
    LADDER.forEach((k, i) => {
      if (m.includes(k)) seen.add(i);
    });
  }
  return seen.size > 0 ? seen.size : undefined;
}

marked.setOptions({ gfm: true, breaks: true });

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
  const sessionId = useStore((s) => s.sessionId);
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
    const ack = await sendIntent(intent.submitTurn(sessionId, t));
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
  // The Executor proposes; it does not host accept/reject. Opening a diff routes to the editor, which
  // owns the gesture (Tab applies the whole diff, Esc rejects) along with the per-hunk panel.
  const openDiff = (c: DiffChip) => void sendIntent(intent.openFile(c.path));
  const approveGate = () => {
    if (gate) void sendIntent(intent.custom("approve_gate", { gate: gate.gate }));
    dismissGate();
  };
  const denyGate = () => {
    if (gate) void sendIntent(intent.custom("deny_gate", { gate: gate.gate }));
    dismissGate();
  };

  const empty = messages.length === 0 && tools.length === 0 && !plan && chips.length === 0;

  return (
    <div className="chat-shell">
      <div ref={scrollRef} className="chat-scroll">
        {empty ? (
          <div className="chat-empty">
            <Display>Describe the work</Display>
          </div>
        ) : (
          <div className="message-list">
            {messages.map((m) => (
              <Message key={m.id} role={m.role} text={m.text} streaming={m.streaming} />
            ))}
            {plan ? <PlanCard plan={plan} onApprove={approvePlan} onEditStep={editStep} onReorder={reorder} /> : null}
            <ToolChipRow tools={tools} />
            <DiffChipRow chips={chips} onOpen={openDiff} />
            {gate ? <InlineGate gate={gate.gate} message={gate.message} onApprove={approveGate} onDismiss={denyGate} /> : null}
          </div>
        )}
      </div>

      <div className="composer-zone">
        {live ? <SteerBar phase={runPhase} onRedirect={steer} onPause={pause} onResume={resume} onCancel={cancel} /> : null}
        <Composer text={text} onText={setText} onSubmit={submit} ready={runtimeReady} live={live} stage={oracleStage(tools)} />
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
  if (role === "user") {
    return (
      <div className="message message--user">
        <div className="message__bubble">{text}</div>
      </div>
    );
  }
  return <AssistantMessage text={text} streaming={streaming} />;
}

// Assistant turn: full-width markdown (no bubble, no role label). Parsed per render (the RAF
// governor throttles re-render during streaming), sanitized, then a Copy button is injected into
// each rendered <pre> via a post-render effect.
function AssistantMessage({ text, streaming }: { text: string; streaming: boolean }) {
  const ref = useRef<HTMLDivElement>(null);

  const html = useMemo(() => {
    const raw = marked.parse(text, { async: false }) as string;
    return DOMPurify.sanitize(raw);
  }, [text]);

  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    const pres = root.querySelectorAll<HTMLPreElement>("pre");
    pres.forEach((pre) => {
      if (pre.dataset.copyWired === "1") return;
      pre.dataset.copyWired = "1";
      pre.classList.add("md-pre");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "md-copy";
      btn.textContent = "Copy";
      btn.addEventListener("click", () => {
        const code = pre.querySelector("code")?.textContent ?? pre.textContent ?? "";
        void navigator.clipboard.writeText(code).then(() => {
          btn.textContent = "Copied";
          window.setTimeout(() => {
            btn.textContent = "Copy";
          }, 1200);
        });
      });
      pre.appendChild(btn);
    });
  }, [html]);

  return (
    <div className="message message--assistant">
      <div ref={ref} className="md" dangerouslySetInnerHTML={{ __html: html }} />
      {streaming ? <span aria-hidden className="stream-cursor" /> : null}
    </div>
  );
}

function Composer({
  text,
  onText,
  onSubmit,
  ready,
  live,
  stage,
}: {
  text: string;
  onText: (v: string) => void;
  onSubmit: () => void;
  ready: boolean;
  live: boolean;
  stage?: number;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const pushNotice = useStore((s) => s.pushNotice);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const rec = useRef<{ mr: MediaRecorder; stream: MediaStream; timer: ReturnType<typeof setInterval> } | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 180) + "px";
  }, [text]);

  // Tear down any live recording if the composer unmounts.
  useEffect(() => () => stopRec(false), []); // eslint-disable-line react-hooks/exhaustive-deps

  const fmt = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

  const stopRec = (transcribe: boolean) => {
    const r = rec.current;
    if (!r) return;
    clearInterval(r.timer);
    try { r.mr.stop(); } catch { /* already stopped */ }
    r.stream.getTracks().forEach((t) => t.stop());
    rec.current = null;
    setRecording(false);
    if (transcribe) {
      // Capture is local (no egress, no time cap); local Whisper transcription is the backend pass.
      pushNotice({ kind: "info", code: "voice", message: `voice ${fmt(elapsed)} captured · transcribing locally` });
    }
    setElapsed(0);
  };

  const toggleMic = async () => {
    if (recording) {
      stopRec(true);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      mr.start();
      const timer = setInterval(() => setElapsed((e) => e + 1), 1000);
      rec.current = { mr, stream, timer };
      setElapsed(0);
      setRecording(true);
    } catch {
      pushNotice({ kind: "error", code: "voice", message: "microphone unavailable" });
    }
  };

  const placeholder = recording
    ? `listening… ${fmt(elapsed)}`
    : !ready
      ? "Runtime not ready"
      : live
        ? "Queue a turn"
        : IDLE_PLACEHOLDER;
  const armed = !!text.trim() && ready;

  return (
    <div className={"composer" + (recording ? " composer--recording" : "")}>
      <button className="composer__attach" type="button" title="Attach" aria-label="Attach" disabled={!ready}>
        <Icon name="plus" size={18} />
      </button>
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
        disabled={!ready || recording}
        className="composer__input"
      />
      <button
        className={"composer__mic" + (recording ? " composer__mic--on" : "")}
        type="button"
        onClick={toggleMic}
        title={recording ? "Stop voice" : "Voice (local, no time limit)"}
        aria-label={recording ? "Stop voice" : "Voice"}
        aria-pressed={recording}
      >
        <Icon name="mic" size={16} />
      </button>
      <button
        className="composer__send"
        type="button"
        onClick={onSubmit}
        disabled={!armed}
        title={live ? "Queue turn" : "Send"}
        aria-label={live ? "Queue turn" : "Send"}
      >
        {live ? <Radiate size={16} active stage={stage} /> : <Icon name="send" size={16} />}
      </button>
    </div>
  );
}
