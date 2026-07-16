/*
  Chat.tsx — the Executor's content: the shared Conversation transcript plus the steering composer. This
  is the same conversation the full-page Chat surface shows (both render <Conversation/> from one store),
  so popping between them (picture-in-picture) never loses context.
*/
import { useEffect, useRef, useState } from "react";
import { sendIntent } from "../ipc";
import { useStore } from "../store";
import { intent } from "../wire";
import { Icon } from "../shell/icons";
import { Radiate } from "../shell/components";
import type { DiffChipPatch, PlanPatch } from "./ChatParts";
import { SteerBar } from "./ChatSteerBar";
import { Conversation } from "./ChatConversation";

const STEERABLE = new Set(["planning", "executing", "paused", "awaiting"]);

// The empty-composer prompt. Default terse (flight-log voice); flip DREAM_BIG to restore "dream big".
const DREAM_BIG = false;
const IDLE_PLACEHOLDER = DREAM_BIG ? "dream big" : "Describe a task";

// Oracle ladder for the radiate ring: how many distinct verify stages have reported this run.
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

export function Chat() {
  const tools = useStore((s) => s.tools);
  const runtimeReady = useStore((s) => s.runtimeStatus === "ready");
  const runPhase = useStore((s) => s.runPhase);
  const activeRunId = useStore((s) => s.activeRunId);
  const pushUserMessage = useStore((s) => s.pushUserMessage);
  const pushNotice = useStore((s) => s.pushNotice);
  const sessionId = useStore((s) => s.sessionId);
  const plan = useStore((s) => s.projections.plan as PlanPatch | undefined);
  const diffPatch = useStore((s) => s.projections.diff_chip as DiffChipPatch | undefined);

  const [text, setText] = useState("");
  const live = STEERABLE.has(runPhase);
  const runId = activeRunId ?? plan?.run_id ?? diffPatch?.chips?.[0]?.run_id ?? "";

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

  return (
    <div className="chat-shell">
      <Conversation />
      <div className="composer-zone">
        {live ? <SteerBar phase={runPhase} onRedirect={steer} onPause={pause} onResume={resume} onCancel={cancel} /> : null}
        <Composer text={text} onText={setText} onSubmit={submit} ready={runtimeReady} live={live} stage={oracleStage(tools)} />
      </div>
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

  useEffect(() => () => stopRec(false), []); // eslint-disable-line react-hooks/exhaustive-deps

  const fmt = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

  const stopRec = (transcribe: boolean) => {
    const r = rec.current;
    if (!r) return;
    clearInterval(r.timer);
    try {
      r.mr.stop();
    } catch {
      /* already stopped */
    }
    r.stream.getTracks().forEach((t) => t.stop());
    rec.current = null;
    setRecording(false);
    if (transcribe) {
      pushNotice({ kind: "info", code: "voice", message: `voice ${fmt(elapsed)} captured, transcribing locally` });
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
    ? `listening ${fmt(elapsed)}`
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
