/*
  Conversation.tsx — the shared transcript. One conversation, two homes: the full-page Chat surface and
  the docked/floating Executor both render THIS from the same store, so they are the same session with the
  same context. Renders the scrolling message column, the plan card, tool chips, diff chips, and the
  inline gate, and owns their intent handlers. The composer lives with each host, not here.
*/
import { useEffect, useLayoutEffect, useMemo, useReducer, useRef } from "react";
import { marked } from "marked";
import DOMPurify from "dompurify";
import { runCommand, useStore } from "../../store";
import { Display } from "../../ui";
import type { DiffChip, DiffChipPatch } from "./parts";
import { DiffChipRow, InlineGate, PlanCard, ToolChipRow, type PlanProjection } from "./structure";

marked.setOptions({ gfm: true, breaks: true });

export function Conversation({ onOpenDiff }: { onOpenDiff?: (path: string) => void }) {
  const messages = useStore((s) => s.messages);
  const tools = useStore((s) => s.tools);
  const gate = useStore((s) => s.gate);
  // The ONE gate handler pair lives in the store, so the shell overlay and this inline capsule
  // cannot drift (consolidation decision: MERGE the two handler pairs, KEEP both presentations).
  const approveGate = useStore((s) => s.approveGate);
  const denyGate = useStore((s) => s.denyGate);
  const sessionId = useStore((s) => s.sessionId);
  const plan = useStore((s) => s.projections.plan as PlanProjection | undefined);
  const diffPatch = useStore((s) => s.projections.diff_chip as DiffChipPatch | undefined);
  const chips: DiffChip[] = diffPatch?.chips ?? [];

  const scrollRef = useRef<HTMLDivElement>(null);

  const streamingLen = messages.reduce((n, m) => n + (m.streaming ? m.text.length : 0), 0);
  useRafGovernor(streamingLen);

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, tools.length, chips.length, gate]);

  // Opening a diff routes to the editor (which owns accept/reject); the host may also switch chambers.
  const openDiff = (c: DiffChip) => {
    void runCommand("open_file", { path: c.path });
    onOpenDiff?.(c.path);
  };

  const empty = messages.length === 0 && tools.length === 0 && !plan && chips.length === 0;

  return (
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
          {plan ? <PlanCard plan={plan} sessionId={sessionId} /> : null}
          <ToolChipRow tools={tools} />
          <DiffChipRow chips={chips} onOpen={openDiff} />
          {gate ? <InlineGate gate={gate.gate} message={gate.message} onApprove={approveGate} onDismiss={denyGate} /> : null}
        </div>
      )}
    </div>
  );
}

// Throttle re-render during streaming to one paint per frame (the store coalesces token batches).
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

// Assistant turn: full-width markdown (no bubble), sanitized, with a Copy button injected into each <pre>.
function AssistantMessage({ text, streaming }: { text: string; streaming: boolean }) {
  const ref = useRef<HTMLDivElement>(null);

  const html = useMemo(() => {
    const raw = marked.parse(text, { async: false }) as string;
    return DOMPurify.sanitize(raw);
  }, [text]);

  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    root.querySelectorAll<HTMLPreElement>("pre").forEach((pre) => {
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
