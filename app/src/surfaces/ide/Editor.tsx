/*
  Editor.tsx: the Monaco editor group (D1.2 / D4.4 #2-#3). Two states:
    - no pending diff: a plain Monaco editor on the open file (ProjectionPatch{editor}, OpenFile).
    - pending diff: Monaco's DiffEditor (the agent's proposed change) on the left, the per-hunk
      HunkReview gesture on the right. Inline-by-default (C10), with a side-by-side toggle for large
      diffs (D1.2 useInlineViewWhenSpaceIsLimited). Accept/reject route AcceptDiff/RejectDiff.

  Re-housed from Void's Monaco DiffEditor wrapper + hunk controls, re-skinned via the hide-observatory
  theme (monacoTheme.ts) so it never reads as VS Code. The diff-accept gesture is the HunkReview
  component, identical to the one the Workstation merge-review reuses (C9).
*/
import { useCallback, useMemo, useRef, useState, type ReactNode } from "react";
import { DiffEditor, Editor as MonacoEditor, type Monaco } from "@monaco-editor/react";
import type { editor as MEditor } from "monaco-editor";
import { sendIntent } from "../../ipc";
import { intent } from "../../wire";
import { HIDE_EDITOR_OPTIONS, HIDE_THEME, configureMonacoLoader, registerHideTheme } from "./monacoTheme";
import { HunkReview } from "./HunkReview";
import { applyHunkStatus, MOCK_FILE_BODY, type DiffDoc } from "./types";

// Point the loader at bundled monaco once (air-gap: no CDN fetch).
configureMonacoLoader();

export function EditorGroup({
  openPath,
  diff,
  onDiffChange,
}: {
  openPath: string | null;
  diff: DiffDoc | null;
  // The surface owns the diff doc so accept/reject status flips re-render locally before the host
  // echoes its own status patch (optimistic, then reconciled by ProjectionPatch{diff}).
  onDiffChange: (next: DiffDoc | null) => void;
}) {
  const [sideBySide, setSideBySide] = useState(false);

  const beforeMount = useCallback((monaco: Monaco) => registerHideTheme(monaco), []);

  if (diff) {
    return (
      <div style={{ display: "grid", gridTemplateColumns: "1fr minmax(300px, 38%)", height: "100%", minHeight: 0 }}>
        <DiffPane diff={diff} sideBySide={sideBySide} onToggle={() => setSideBySide((v) => !v)} beforeMount={beforeMount} />
        <div style={{ borderLeft: "1px solid var(--rim)", minHeight: 0 }}>
          <HunkReview
            doc={diff}
            onAct={(hunk, action) => {
              // Optimistic local flip + the real intent (D2.2: AcceptDiff/RejectDiff{run_id,diff_id}).
              onDiffChange(applyHunkStatus(diff, hunk.id, action === "accept" ? "accepted" : "rejected"));
              const i = action === "accept" ? intent.acceptDiff(diff.run_id, diff.diff_id) : intent.rejectDiff(diff.run_id, diff.diff_id);
              void sendIntent(i);
            }}
          />
        </div>
      </div>
    );
  }

  return <FilePane openPath={openPath} beforeMount={beforeMount} />;
}

function DiffPane({
  diff,
  sideBySide,
  onToggle,
  beforeMount,
}: {
  diff: DiffDoc;
  sideBySide: boolean;
  onToggle: () => void;
  beforeMount: (m: Monaco) => void;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
      <TabRow path={diff.path} suffix="diff">
        <ViewToggle sideBySide={sideBySide} onToggle={onToggle} />
      </TabRow>
      <div style={{ flex: 1, minHeight: 0 }}>
        <DiffEditor
          original={diff.before}
          modified={diff.after}
          language={diff.lang}
          theme={HIDE_THEME}
          beforeMount={beforeMount}
          loading={<Loading />}
          options={{
            ...HIDE_EDITOR_OPTIONS,
            renderSideBySide: sideBySide,
            readOnly: false, // the modified side is editable before accept (D1.2)
            renderMarginRevertIcon: false,
            renderOverviewRuler: false,
            diffWordWrap: "off",
            enableSplitViewResizing: true,
          }}
        />
      </div>
    </div>
  );
}

function FilePane({ openPath, beforeMount }: { openPath: string | null; beforeMount: (m: Monaco) => void }) {
  const editorRef = useRef<MEditor.IStandaloneCodeEditor | null>(null);
  // Resolve the open file's body. Live host streams it as ProjectionPatch{editor}; the mock has stubs.
  const body = useMemo(() => (openPath ? MOCK_FILE_BODY[openPath] : null), [openPath]);

  if (!openPath) {
    return (
      <div style={{ display: "grid", placeItems: "center", height: "100%", color: "var(--text-low)", textAlign: "center" }}>
        <div style={{ maxWidth: 360 }}>
          <div style={{ fontSize: "var(--text-sm)" }}>Open a file from the Explorer.</div>
          <div style={{ marginTop: "var(--s2)", color: "var(--text-low)", fontSize: "var(--text-xs)" }}>
            The agent's edits arrive here as reviewable diff hunks. Move with j and k, take with a, drop with r.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <TabRow path={openPath} suffix={null} />
      <div style={{ flex: 1, minHeight: 0 }}>
        <MonacoEditor
          path={openPath}
          language={body?.lang ?? "plaintext"}
          value={body?.text ?? `// ${openPath}\n// (host streams this buffer as projection_patch{editor})\n`}
          theme={HIDE_THEME}
          beforeMount={beforeMount}
          onMount={(ed, monaco) => {
            editorRef.current = ed;
            ed.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () =>
              void sendIntent(intent.custom("save_file", { path: openPath })),
            );
          }}
          loading={<Loading />}
          options={{ ...HIDE_EDITOR_OPTIONS, readOnly: false }}
        />
      </div>
    </div>
  );
}

function TabRow({ path, suffix, children }: { path: string; suffix: string | null; children?: ReactNode }) {
  const name = path.split("/").pop() ?? path;
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--s2)",
        height: 30,
        padding: "0 var(--s3)",
        borderBottom: "1px solid var(--rim)",
        background: "var(--surface-0)",
        backgroundImage: "var(--panel-grad)",
        fontSize: "var(--text-xs)",
      }}
    >
      <span style={{ color: "var(--text-hi)" }}>{name}</span>
      {suffix ? (
        <span style={{ color: "var(--radiation)", textTransform: "uppercase", letterSpacing: "0.06em" }}>{suffix}</span>
      ) : null}
      <span style={{ color: "var(--text-low)" }}>{path}</span>
      <div style={{ marginLeft: "auto" }}>{children}</div>
    </div>
  );
}

function ViewToggle({ sideBySide, onToggle }: { sideBySide: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      title="Toggle inline / side-by-side"
      style={{
        fontSize: "var(--text-xs)",
        color: "var(--text-mid)",
        padding: "2px 8px",
        borderRadius: "var(--radius)",
        boxShadow: "inset 0 0 0 1px var(--rim)",
        background: "var(--surface-1)",
      }}
    >
      {sideBySide ? "side by side" : "inline"}
    </button>
  );
}

// No spinner (C14): aliveness is the breathing gold, not a wheel.
function Loading() {
  return (
    <div style={{ display: "grid", placeItems: "center", height: "100%" }}>
      <span
        aria-label="loading editor"
        style={{
          width: 10,
          height: 10,
          borderRadius: "50%",
          background: "var(--radiation)",
          boxShadow: "0 0 12px 0 var(--radiation-bloom)",
          animation: "radiation-breathe 1.8s ease-in-out infinite",
        }}
      />
    </div>
  );
}
