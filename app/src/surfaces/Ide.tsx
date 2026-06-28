/*
  Ide.tsx: the AI IDE surface, the workshop chamber (v3). Three concrete volumes float in the void:
  the file tree (Explorer), the editor (largest), and the terminal, with --ma-4..6 between them. The
  core gesture is the per-hunk diff review (HunkReview): the agent's proposed change arrives as
  projection_patch{diff}, reviewed by keyboard (j/k/a/r), accepted per hunk as AcceptDiff / rejected as
  RejectDiff. Monaco and xterm are mounted and recast in v3 (grayscale concrete, light as the only
  accent, Geist Mono): they must not read as VS Code, and there is no gold anywhere.

  Sends: OpenFile, AcceptDiff/RejectDiff, RunCommand, Custom:save_file (via the editor instance).
  Consumes: projection_patch{diff} (the generic projections bag -> parseDiff), projection_patch{editor},
  tool_progress (mirrored into the terminal). When the mock is the transport (it emits no diff patch),
  the surface seeds MOCK_DIFF so the signature gesture is ALIVE with no backend.
*/
import { useEffect, useMemo, useState } from "react";
import { useStore } from "../store";
import { TRANSPORT_KIND } from "../ipc";
import { SectionLabel } from "../ui";
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
  const reviewing = diff != null && !allDecided;

  return (
    <div
      style={{
        // Three volumes: a quiet file tree, the large editor over the terminal. They float in the
        // chamber with generous void between them (the editor column is the subject).
        display: "grid",
        gridTemplateColumns: "clamp(240px, 22vw, 320px) 1fr",
        gap: "var(--ma-6)",
        height: "100%",
        minHeight: 0,
      }}
    >
      {/* FILE TREE volume: a calm airy list along the west of the chamber. */}
      <aside className="volume" style={{ padding: 0, minHeight: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        <Explorer activePath={openPath} onOpen={setOpenPath} />
      </aside>

      {/* EDITOR over TERMINAL: the two largest volumes, stacked with --ma-5 of void between. */}
      <div style={{ display: "grid", gridTemplateRows: "1fr clamp(180px, 30%, 360px)", gap: "var(--ma-5)", minHeight: 0, minWidth: 0 }}>
        {/* EDITOR volume: the largest, where the agent's change lands as reviewable hunks. */}
        <section
          className={"volume" + (reviewing ? " alive" : "")}
          style={{ padding: 0, minHeight: 0, position: "relative", overflow: "hidden", display: "flex", flexDirection: "column" }}
        >
          {reviewing ? <DiffBanner path={diff.path} count={diff.hunks.filter((h) => h.status === "pending").length} /> : null}
          <div style={{ flex: 1, minHeight: 0 }}>
            <EditorGroup openPath={openPath} diff={diff} onDiffChange={setDiff} />
          </div>
        </section>

        {/* TERMINAL volume: the mirrored shell, the agent's commands entering as light. */}
        <section className="volume" style={{ padding: 0, minHeight: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "var(--ma-3)",
              padding: "var(--ma-2) var(--ma-4)",
              boxShadow: "inset 0 -1px 0 0 var(--line)",
            }}
          >
            <SectionLabel>Terminal</SectionLabel>
            <span className="t-micro">shell.run mirrors agent commands</span>
          </div>
          <div style={{ flex: 1, minHeight: 0, padding: "var(--ma-3) var(--ma-4) var(--ma-4)" }}>
            <Terminal />
          </div>
        </section>
      </div>
    </div>
  );
}

// A calm banner that the agent has a change waiting. No spinner; the editor volume already breathes
// (.alive), so this only names the change in quiet light. The leading word reads in light, never gold.
function DiffBanner({ path, count }: { path: string; count: number }) {
  return (
    <div
      role="status"
      className="t-micro"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 5,
        display: "flex",
        alignItems: "center",
        gap: "var(--ma-2)",
        padding: "var(--ma-2) var(--ma-4)",
        background: "var(--concrete-3)",
        boxShadow: "inset 0 -1px 0 0 var(--line), var(--inner-glow)",
        color: "var(--text-2)",
      }}
    >
      <span style={{ color: "var(--light)" }}>change proposed</span>
      <span style={{ color: "var(--text-2)" }}>{path}</span>
      <span style={{ marginLeft: "auto", color: "var(--text-3)" }}>
        {count} hunk{count === 1 ? "" : "s"} to review
      </span>
    </div>
  );
}
