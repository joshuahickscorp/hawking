/*
  Ide.tsx: the AI IDE surface (D1.2). Editor-centric body: Explorer | (Editor / Diff Review) over a
  resizable bottom panel (Terminal). The core gesture is the per-hunk diff review (HunkReview): the
  agent's proposed change arrives as projection_patch{diff}, reviewed by keyboard (j/k/a/r), accepted
  per hunk as AcceptDiff / rejected as RejectDiff. Monaco and xterm are mounted and fully re-skinned to
  the doctrine (near-black anodized material, gold rim-light, Geist Mono): they must not read as VS Code.

  Sends: OpenFile, AcceptDiff/RejectDiff, RunCommand, Custom:save_file (via the editor instance).
  Consumes: projection_patch{diff} (the generic projections bag -> parseDiff), projection_patch{editor},
  tool_progress (mirrored into the terminal). When the mock is the transport (it emits no diff patch),
  the surface seeds MOCK_DIFF so the signature gesture is ALIVE with no backend.
*/
import { useEffect, useMemo, useState } from "react";
import { useStore } from "../store";
import { TRANSPORT_KIND } from "../ipc";
import { Explorer } from "./ide/Explorer";
import { EditorGroup } from "./ide/Editor";
import { Terminal } from "./ide/Terminal";
import { MOCK_DIFF, parseDiff, type DiffDoc } from "./ide/types";

export function Ide() {
  // The host folds a real diff into the generic projection bag (store routeProjection default case).
  const diffPatch = useStore((s) => s.projections.diff);
  const editorPatch = useStore((s) => s.projections.editor);

  const [openPath, setOpenPath] = useState<string | null>("crates/pool/src/guard.rs");
  // The surface owns the live diff doc so per-hunk accept/reject flips render immediately (optimistic),
  // then reconcile when the host echoes its own status patch.
  const [diff, setDiff] = useState<DiffDoc | null>(null);

  // Fold the host's diff patch, or (mock transport, no diff emitted) seed the believable MOCK_DIFF
  // a moment after mount so the signature hunk-review gesture is alive in the demo.
  const hostDiff = useMemo(() => parseDiff(diffPatch as Record<string, unknown> | undefined), [diffPatch]);
  useEffect(() => {
    if (hostDiff) {
      setDiff(hostDiff);
      return;
    }
    if (TRANSPORT_KIND === "mock") {
      const t = setTimeout(() => setDiff((d) => d ?? MOCK_DIFF), 1500);
      return () => clearTimeout(t);
    }
  }, [hostDiff]);

  // Reflect an externally-opened file (provenance peek, Context-Stack click) the host streams as editor.
  useEffect(() => {
    const path = (editorPatch as { path?: string } | undefined)?.path;
    if (path) setOpenPath(path);
  }, [editorPatch]);

  const allDecided = diff != null && diff.hunks.every((h) => h.status !== "pending");

  return (
    <div style={{ display: "grid", gridTemplateColumns: "var(--sidebar-w) 1fr", height: "100%", minHeight: 0 }}>
      <LocalKeyframes />

      {/* Primary sidebar: Explorer + Search */}
      <aside style={{ borderRight: "1px solid var(--rim)", minHeight: 0, background: "var(--surface-0)" }}>
        <Explorer activePath={openPath} onOpen={setOpenPath} />
      </aside>

      {/* Editor group over the bottom panel (Terminal). */}
      <div style={{ display: "grid", gridTemplateRows: "1fr 30%", minHeight: 0 }}>
        <section style={{ minHeight: 0, position: "relative" }}>
          {diff && !allDecided ? <DiffBanner path={diff.path} count={diff.hunks.filter((h) => h.status === "pending").length} /> : null}
          <EditorGroup openPath={openPath} diff={diff} onDiffChange={setDiff} />
        </section>

        <section style={{ borderTop: "1px solid var(--rim)", display: "flex", flexDirection: "column", minHeight: 0, background: "var(--surface-0)" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "var(--s3)",
              padding: "4px var(--s3)",
              fontSize: "var(--text-xs)",
              color: "var(--text-low)",
              borderBottom: "1px solid var(--rim)",
            }}
          >
            <span style={{ color: "var(--text-mid)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Terminal</span>
            <span>shell.run mirrors agent commands</span>
          </div>
          <div style={{ flex: 1, minHeight: 0, padding: "var(--s2) var(--s3) var(--s3)" }}>
            <Terminal />
          </div>
        </section>
      </div>
    </div>
  );
}

// A calm gold-rim banner that the agent has a change waiting (no spinner; the rim is the aliveness).
function DiffBanner({ path, count }: { path: string; count: number }) {
  return (
    <div
      role="status"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 5,
        display: "flex",
        alignItems: "center",
        gap: "var(--s2)",
        padding: "3px var(--s4)",
        background: "var(--surface-1)",
        boxShadow: "inset 0 0 0 1px var(--radiation), 0 0 18px -8px var(--radiation-bloom)",
        color: "var(--text-mid)",
        fontSize: "var(--text-xs)",
      }}
    >
      <span style={{ color: "var(--radiation-bright)" }}>change proposed</span>
      <span>{path}</span>
      <span style={{ marginLeft: "auto", color: "var(--text-low)" }}>
        {count} hunk{count === 1 ? "" : "s"} to review
      </span>
    </div>
  );
}

/*
  The two one-shot diff-settle keyframes the HunkReview absorption uses (C11). theme.css owns the
  shared keyframes (radiation-breathe / radiation-travel) and is off-limits to edit, so these
  surface-local animations are injected here. They honor prefers-reduced-motion via theme.css's
  global reduce rule.
*/
function LocalKeyframes() {
  return (
    <style>{`
      @keyframes hunk-absorb {
        0%   { box-shadow: inset 0 0 0 1px var(--radiation-bright), 0 0 26px 0 var(--radiation-bloom); }
        100% { box-shadow: inset 0 0 0 1px var(--rim); }
      }
      @keyframes hunk-dissolve {
        0%   { opacity: 1; transform: translateY(0); }
        100% { opacity: 0.4; transform: translateY(-2px); }
      }
    `}</style>
  );
}
