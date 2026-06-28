/*
  Chat.tsx: the AI Chat surface frame. Streams the mock TokenBatch so the shell is visibly ALIVE.
  Skeleton: message list + streaming assistant message + composer. The steer bar, plan cards,
  tool chips, and diff chips are clean stubs for the surface pass (01-surfaces §C).
  Sends: SubmitTurn (composer). Consumes: token_batch + projection_patch(turn) via the store.
*/
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { sendIntent } from "../ipc";
import { useStore } from "../store";
import { intent } from "../wire";
import { Display, Panel, RadiationEdge } from "../ui";

const SESSION = "ses_mock0000000000000000000";

export function Chat() {
  const messages = useStore((s) => s.messages);
  const runtimeReady = useStore((s) => s.runtimeStatus === "ready");
  const pushUserMessage = useStore((s) => s.pushUserMessage);
  const pushNotice = useStore((s) => s.pushNotice);
  const [text, setText] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  // Keep the transcript pinned to the streaming edge.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const submit = async () => {
    const t = text.trim();
    if (!t) return;
    pushUserMessage(t);
    setText("");
    const ack = await sendIntent(intent.submitTurn(SESSION, t));
    if (!ack.accepted) pushNotice({ kind: "error", code: "rejected", message: ack.message ?? "turn rejected" });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div ref={scrollRef} style={{ flex: 1, overflowY: "auto", padding: "var(--s5)" }}>
        {messages.length === 0 ? (
          <div style={{ maxWidth: 560, margin: "10vh auto 0", textAlign: "center" }}>
            <Display size={44}>Open the box.</Display>
            <p style={{ color: "var(--text-low)", marginTop: "var(--s4)" }}>
              Ask the agent to do work. You will see exactly what it reads and runs.
            </p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--s4)", maxWidth: 720, margin: "0 auto" }}>
            {messages.map((m) => (
              <Message key={m.id} role={m.role} text={m.text} streaming={m.streaming} />
            ))}
          </div>
        )}
      </div>

      <div style={{ padding: "var(--s4)", borderTop: "1px solid var(--rim)" }}>
        <Composer
          text={text}
          onText={setText}
          onSubmit={submit}
          ready={runtimeReady}
        />
      </div>
    </div>
  );
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
        }}
      >
        <div style={{ fontSize: "var(--text-xs)", color: "var(--text-low)", marginBottom: "var(--s1)" }}>
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
}: {
  text: string;
  onText: (v: string) => void;
  onSubmit: () => void;
  ready: boolean;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 180) + "px";
  }, [text]);

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
          placeholder={ready ? "Message the agent" : "Runtime not ready"}
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
          Send
        </button>
      </Panel>
    </RadiationEdge>
  );
}
