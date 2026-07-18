/*
  Conversation.tsx — the shared transcript. One conversation, two homes: the full-page Chat surface and
  the docked/floating Executor both render THIS from the same store, so they are the same session with the
  same context. Renders the scrolling message column, the plan card, tool chips, diff chips, and the
  inline gate, and owns their intent handlers. The composer lives with each host, not here.
*/
import { useEffect, useLayoutEffect, useMemo, useReducer, useRef } from "react";
import { marked } from "marked";
import DOMPurify from "dompurify";
import { sendIntent } from "../../ipc";
import { useStore } from "../../store";
import { intent } from "../../wire";
import { Display } from "../../ui";
import type { DiffChip, DiffChipPatch, PlanPatch, PlanStep } from "./parts";
import { DiffChipRow, InlineGate, PlanCard, ToolChipRow } from "./structure";

marked.setOptions({ gfm: true, breaks: true });

export function Conversation({ onOpenDiff }: { onOpenDiff?: (path: string) => void }) {
  const messages = useStore((s) => s.messages);
  const tools = useStore((s) => s.tools);
  const gate = useStore((s) => s.gate);
  const dismissGate = useStore((s) => s.dismissGate);
  const activeRunId = useStore((s) => s.activeRunId);
  const plan = useStore((s) => s.projections.plan as PlanPatch | undefined);
  const diffPatch = useStore((s) => s.projections.diff_chip as DiffChipPatch | undefined);
  const chips: DiffChip[] = diffPatch?.chips ?? [];

  const scrollRef = useRef<HTMLDivElement>(null);
  const runId = activeRunId ?? plan?.run_id ?? chips[0]?.run_id ?? "";

  const streamingLen = messages.reduce((n, m) => n + (m.streaming ? m.text.length : 0), 0);
  useRafGovernor(streamingLen);

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, tools.length, chips.length, gate]);

  const approvePlan = () => void sendIntent(intent.custom("approve_plan", { run_id: runId }));
  const editStep = (step: PlanStep, title: string) =>
    void sendIntent(intent.custom("edit_plan_step", { run_id: runId, step_id: step.id, title }));
  const reorder = (from: number, to: number) => void sendIntent(intent.custom("reorder_plan", { run_id: runId, from, to }));
  // Opening a diff routes to the editor (which owns accept/reject); the host may also switch chambers.
  const openDiff = (c: DiffChip) => {
    void sendIntent(intent.openFile(c.path));
    onOpenDiff?.(c.path);
  };
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
