/*
  Editor.tsx: the Monaco editor group (v3). Two states:
    - no pending diff: a plain Monaco editor on the open file (ProjectionPatch{editor}, OpenFile).
      A selection resolves to a stable SourceRef (path, line range, content hash) and opens the
      CodeActions menu; the ref is re-hashed from the live buffer so a stale selection is caught.
    - pending diff: Monaco's DiffEditor (the agent's proposed change) on the left, the per-hunk
      HunkReview gesture on the right. Inline-by-default, with a side-by-side toggle for large diffs.

  The diff bar has two phases and never more than its two controls: while hunks are pending it
  accepts the WHOLE diff (accept_diff with no hunk_id, which is how the host reads "all of it",
  walking its own record so every hunk it applies keeps its provenance) or reverts it; once every
  hunk is decided the same two slots become that same whole-diff revert and closing the review.
  ONE revert command (`revert_diff`) serves both phases, because it is one host effect and it is
  approval-gated: a second button for it was a way around the gate. There is no provenance-free
  accept-all.

  Re-housed from a Monaco DiffEditor wrapper + hunk controls, recast in v3 via the hide-observatory
  theme (monacoTheme.ts): grayscale concrete, light as the only accent, Geist Mono. It must never read
  as VS Code. The diff-accept gesture is the HunkReview component, the single source of per-hunk review
  logic in the IDE.
*/
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { DiffEditor, Editor as MonacoEditor, type Monaco } from "@monaco-editor/react";
import type { editor as MEditor } from "monaco-editor";
import { callConnector } from "../../ipc";
import { runCommand, useStore } from "../../store";
import { HIDE_EDITOR_OPTIONS, HIDE_THEME, configureMonacoLoader, registerHideTheme } from "./monacoTheme";
import { HunkReview, diffActionSpec, runDiffAction, type DiffActionId } from "./HunkReview";
import { CodeActions, sourceRef, type SourceRef } from "./CodeActions";
import { Radiate } from "../../shell/Radiate";
import { applyHunkStatus, MOCK_FILE_BODY, mockOnly, type DiffDoc } from "./types";
import { ackState, heldNote } from "../../wire";

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
    return <DiffReview diff={diff} sideBySide={sideBySide} onToggle={() => setSideBySide((v) => !v)} beforeMount={beforeMount} onDiffChange={onDiffChange} />;
  }

  return <FilePane openPath={openPath} beforeMount={beforeMount} />;
}

// DiffReview: the inline diff the Executor proposes, with the per-hunk HunkReview beside it for
// granular control. NO bare key acts on the whole diff. The two whole-diff verbs are the two
// buttons in the diff bar, and the per-hunk keys (a, r, j, k, m, d) are scoped to this region by
// HunkReview.reviewKeysActive and touch one hunk at a time.
function DiffReview({
  diff,
  sideBySide,
  onToggle,
  beforeMount,
  onDiffChange,
}: {
  diff: DiffDoc;
  sideBySide: boolean;
  onToggle: () => void;
  beforeMount: (m: Monaco) => void;
  onDiffChange: (next: DiffDoc | null) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [note, setNote] = useState<string | null>(null);
  useEffect(() => {
    ref.current?.focus();
  }, [diff.diff_id]);

  const pending = diff.hunks.filter((h) => h.status === "pending").length;
  const decided = diff.hunks.length > 0 && pending === 0;

  // One whole-diff dispatch point. The status flip stays local (optimistic) until the host echoes
  // its own diff record, and every hunk keeps the provenance it was recorded with.
  const runWhole = useCallback(
    (id: DiffActionId) => {
      const spec = diffActionSpec(id);
      setNote(`${spec.label}: working`);
      void runDiffAction(id, { diffId: diff.diff_id, runId: diff.run_id })
        .then((ack) => {
          const state = ackState(ack);
          if (state === "refused") {
            setNote(ack.message ?? "The host refused that action");
            return;
          }
          // HELD: the host recorded the request and parked the effect at the approval gate. Nothing
          // was written, so the note must not say done and no hunk status may flip.
          if (state === "held") {
            setNote(heldNote(spec.label));
            return;
          }
          setNote(`${spec.label}: done`);
          const status = id === "accept_all" ? "accepted" : "rejected";
          onDiffChange(
            diff.hunks.reduce(
              (doc, h) => (h.status === "pending" || id !== "accept_all" ? applyHunkStatus(doc, h.id, status) : doc),
              diff,
            ),
          );
        })
        .catch((e) => setNote(e instanceof Error ? e.message : String(e)));
      ref.current?.focus();
    },
    [diff, onDiffChange],
  );

  // NEITHER Escape NOR Tab is bound here, deliberately.
  //   Escape used to reject the whole diff (reverting every changed file) while the shell was using
  //   the same key to close overlays, so one keystroke meant two things and one of them destroyed
  //   work with no confirmation.
  //   Tab used to accept the whole diff to disk, with preventDefault, from any focused control in
  //   this region: the first Tab a keyboard user pressed to reach the accept button wrote every
  //   hunk, and focus could never leave the region at all. Both defects were the same key.
  // The two whole-diff verbs are the two buttons below. Tab moves focus and nothing else.

  return (
    <div ref={ref} tabIndex={-1} className="diffreview" role="region" aria-label="Proposed edit" style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0, minWidth: 0, outline: "none" }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr minmax(300px, 38%)", flex: 1, minHeight: 0, minWidth: 0 }}>
        <DiffPane diff={diff} sideBySide={sideBySide} onToggle={onToggle} beforeMount={beforeMount} />
        <div style={{ borderLeft: "1px solid var(--border)", minHeight: 0, minWidth: 0, overflow: "hidden" }}>
          <HunkReview doc={diff} onStatus={(hunkId, status) => onDiffChange(applyHunkStatus(diff, hunkId, status))} />
        </div>
      </div>
      <div className="diffbar">
        <span className="diffbar__hint">
          {decided ? (
            <>every hunk decided<span className="diffbar__sep">/</span>revert the whole diff, or close the review</>
          ) : (
            <>
              {pending} pending<span className="diffbar__sep">/</span>accept all and revert all are buttons, so no stray key writes or reverts your files<span className="diffbar__sep">/</span>or review each hunk at right
            </>
          )}
        </span>
        <span role="status" aria-live="polite" className="diffbar__hint">{note}</span>
        <div className="diffbar__actions">
          {decided ? (
            <>
              <button className="diffbar__btn" title={diffActionSpec("revert_all").label} onClick={() => runWhole("revert_all")}>
                revert all
              </button>
              <button className="diffbar__btn diffbar__btn--accent" title="Close this review" onClick={() => onDiffChange(null)}>
                close review
              </button>
            </>
          ) : (
            <>
              <button className="diffbar__btn" title={diffActionSpec("revert_all").label} onClick={() => runWhole("revert_all")}>
                revert all
              </button>
              <button className="diffbar__btn diffbar__btn--accent" title={diffActionSpec("accept_all").label} onClick={() => runWhole("accept_all")}>
                accept all
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
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
            readOnly: false, // the modified side is editable before accept
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

/** Read a ref's line range back out of the live buffer, so the selection's content hash can be
 *  re-checked. Returns null when the buffer is gone or no longer reaches those lines. */
function readRangeFrom(ed: MEditor.IStandaloneCodeEditor | null, ref: SourceRef): string | null {
  const model = ed?.getModel();
  if (!model || model.getLineCount() < ref.endLine) return null;
  return model.getValueInRange({
    startLineNumber: ref.startLine,
    startColumn: 1,
    endLineNumber: ref.endLine,
    endColumn: model.getLineMaxColumn(ref.endLine),
  });
}

function FilePane({ openPath, beforeMount }: { openPath: string | null; beforeMount: (m: Monaco) => void }) {
  const editorRef = useRef<MEditor.IStandaloneCodeEditor | null>(null);
  const [sel, setSel] = useState<{ ref: SourceRef; top: number; left: number } | null>(null);
  const pushNotice = useStore((s) => s.pushNotice);
  // The open file's real body from the fs connector. The mock stub stands in ONLY on the mock
  // transport. There is no invented placeholder buffer any more: a body that was never really read
  // is an error state, not an editable buffer, because Cmd+S below writes the buffer to disk.
  // `hash` is the host's blake3 of the text that was read. It rides back out on save as `base_hash`
  // so a file an agent changed since this buffer was opened CONFLICTS instead of being clobbered.
  const [body, setBody] = useState<{ text: string; lang: string; hash?: string } | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  useEffect(() => {
    setBody(null);
    setLoadError(null);
    if (!openPath) return;
    let alive = true;
    const fallback = (why: string) => {
      const stub = mockOnly(MOCK_FILE_BODY)?.[openPath];
      if (stub) setBody(stub);
      else setLoadError(why);
    };
    callConnector<{ text?: string; lang?: string; hash?: string }>("fs", "read_file", { path: openPath })
      .then((r) => {
        if (!alive) return;
        if (typeof r?.text === "string") setBody({ text: r.text, lang: r.lang ?? "plaintext", hash: r.hash });
        else fallback("The host returned no content for this file");
      })
      .catch((e) => {
        if (alive) fallback(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, [openPath]);

  if (!openPath) {
    return (
      <div style={{ display: "grid", placeItems: "center", height: "100%", color: "var(--text-3)", textAlign: "center" }}>
        <div style={{ maxWidth: 380 }}>
          <div className="t-body" style={{ color: "var(--text-2)" }}>Open a file</div>
        </div>
      </div>
    );
  }

  // Never mount an editable buffer over a file whose body did not really load: saving it would
  // write this app's invention over the real file.
  if (loadError || !body) {
    return (
      <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
        <TabRow path={openPath} suffix={null} />
        <div style={{ display: "grid", placeItems: "center", flex: 1, minHeight: 0, color: "var(--text-3)", textAlign: "center" }}>
          {loadError ? (
            <div role="alert" style={{ maxWidth: 420 }} className="t-body">
              Could not read {openPath}. {loadError}
            </div>
          ) : (
            <Loading />
          )}
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <TabRow path={openPath} suffix={null} />
      <div style={{ flex: 1, minHeight: 0, position: "relative" }}>
        <MonacoEditor
          path={openPath}
          language={body.lang}
          value={body.text}
          theme={HIDE_THEME}
          beforeMount={beforeMount}
          onMount={(ed, monaco) => {
            editorRef.current = ed;
            // The save. It is the catalog's `save_file` (Mod+S, listed in the Settings keyboard
            // table), dispatched through the ONE spine, so the host runs it on the permission-gated
            // applier: a write the policy refuses comes back HELD at the approval gate with the
            // policy's own reason, never as a bare "save failed".
            ed.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
              const content = ed.getModel()?.getValue() ?? "";
              void runCommand("save_file", { path: openPath, content, base_hash: body.hash ?? null })
                .then((ack) => {
                  const state = ackState(ack);
                  if (state === "accepted") {
                    pushNotice({ kind: "info", code: "save_file", message: `saved ${openPath}` });
                    return;
                  }
                  pushNotice({
                    kind: state === "held" ? "info" : "error",
                    code: "save_file",
                    message: ack.message ?? `save refused ${openPath}`,
                  });
                })
                .catch((e) =>
                  pushNotice({
                    kind: "error",
                    code: "save_file",
                    message: e instanceof Error ? e.message : String(e),
                  }),
                );
            });
            // A non-empty selection resolves to a stable SourceRef (path, whole-line range, content
            // hash) and the contextual action menu follows it. Whole lines, so the same range can be
            // read back later and compared: that is what makes a stale selection detectable.
            ed.onDidChangeCursorSelection((e) => {
              const model = ed.getModel();
              if (!model || model.getValueInRange(e.selection).trim().length <= 1) {
                setSel(null);
                return;
              }
              const startLine = e.selection.startLineNumber;
              const endLine = e.selection.endLineNumber;
              const text = model.getValueInRange({
                startLineNumber: startLine,
                startColumn: 1,
                endLineNumber: endLine,
                endColumn: model.getLineMaxColumn(endLine),
              });
              const vp = ed.getScrolledVisiblePosition(e.selection.getStartPosition());
              if (vp)
                setSel({
                  ref: sourceRef(openPath, startLine, endLine, text),
                  top: Math.max(4, vp.top - 40),
                  left: Math.min(Math.max(8, vp.left), 360),
                });
            });
            ed.onDidScrollChange(() => setSel(null));
            // Focusing the menu blurs the editor widget, so the dismissal checks where focus landed.
            // Without this the menu closed on its own mount focus and could never be clicked.
            ed.onDidBlurEditorWidget(() => {
              setTimeout(() => {
                if (!document.activeElement?.closest(".codeactions")) setSel(null);
              }, 0);
            });
          }}
          loading={<Loading />}
          options={{ ...HIDE_EDITOR_OPTIONS, readOnly: false }}
        />
        {sel ? (
          <CodeActions
            sel={sel.ref}
            top={sel.top}
            left={sel.left}
            readRange={(r) => readRangeFrom(editorRef.current, r)}
            onDone={() => setSel(null)}
          />
        ) : null}
      </div>
    </div>
  );
}

// VS Code breadcrumb bar: thin, editor-bg, the path split into segments joined by a "›" separator.
function TabRow({ path, suffix, children }: { path: string; suffix: string | null; children?: ReactNode }) {
  const segments = path.split("/").filter(Boolean);
  return (
    <div className="vsc-breadcrumb">
      <div className="vsc-breadcrumb__trail">
        {segments.map((seg, i) => (
          <span key={i} className="vsc-breadcrumb__seg">
            {i > 0 ? <span className="vsc-breadcrumb__sep" aria-hidden>›</span> : null}
            <span className={i === segments.length - 1 ? "vsc-breadcrumb__leaf" : undefined}>{seg}</span>
          </span>
        ))}
        {suffix ? <span className="vsc-breadcrumb__suffix">{suffix}</span> : null}
      </div>
      {children ? <div className="vsc-breadcrumb__actions">{children}</div> : null}
    </div>
  );
}

function ViewToggle({ sideBySide, onToggle }: { sideBySide: boolean; onToggle: () => void }) {
  return (
    <button className="text-button vsc-view-toggle" onClick={onToggle} title="Toggle inline / side-by-side">
      {sideBySide ? "Side by side" : "Inline"}
    </button>
  );
}

// The radiate ring is the one loading indicator in the app, here too (no stock spinner).
function Loading() {
  return (
    <div style={{ display: "grid", placeItems: "center", height: "100%", gap: "var(--ma-3)" }}>
      <Radiate size={20} active title="loading" />
    </div>
  );
}
