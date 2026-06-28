/*
  Ide.tsx: the AI IDE surface frame (01-surfaces §B). Six-region body: Explorer + Editor + Bottom.
  Skeleton structure is real (regions, store wiring points, intents named) but the editor, diff
  review, file tree, and terminal get fully fleshed out in the surface pass. Monaco and xterm are
  installed deps; we mount a lightweight placeholder rather than pull workers into the foundation build.
  Sends: OpenFile, RunCommand, AcceptDiff/RejectDiff. Consumes: projection_patch(editor/diff/file_external).
*/
import { sendIntent } from "../ipc";
import { useStore } from "../store";
import { intent } from "../wire";
import { Panel, SectionLabel } from "../ui";

// A small stub tree until the file_tree projection + code_index connector are wired.
const STUB_TREE = [
  "crates/pool/src/guard.rs",
  "crates/pool/src/lib.rs",
  "crates/hide-core/src/api.rs",
];

export function Ide() {
  const editor = useStore((s) => s.projections.editor);
  const diff = useStore((s) => s.projections.diff);

  const open = (path: string) => void sendIntent(intent.openFile(path));

  return (
    <div style={{ display: "grid", gridTemplateColumns: "var(--sidebar-w) 1fr", height: "100%", minHeight: 0 }}>
      {/* Primary sidebar: Explorer (stub) */}
      <aside style={{ borderRight: "1px solid var(--rim)", padding: "var(--s3)", overflowY: "auto" }}>
        <SectionLabel>Explorer</SectionLabel>
        <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
          {STUB_TREE.map((p) => (
            <li key={p}>
              <button
                onClick={() => open(p)}
                style={{
                  width: "100%",
                  textAlign: "left",
                  padding: "3px var(--s2)",
                  borderRadius: "var(--radius)",
                  color: "var(--text-mid)",
                  fontSize: "var(--text-sm)",
                }}
              >
                {p.split("/").pop()}
                <span style={{ color: "var(--text-low)", marginLeft: "var(--s2)" }}>{p}</span>
              </button>
            </li>
          ))}
        </ul>
      </aside>

      {/* Editor group + bottom panel */}
      <div style={{ display: "grid", gridTemplateRows: "1fr 30%", minHeight: 0 }}>
        <section style={{ minHeight: 0, padding: "var(--s3)" }}>
          <Panel pad="var(--s4)" style={{ height: "100%", display: "grid", placeItems: "center" }}>
            <div style={{ textAlign: "center", color: "var(--text-low)" }}>
              <SectionLabel>Editor</SectionLabel>
              {editor || diff ? (
                <pre style={{ margin: 0, color: "var(--text-mid)", fontSize: "var(--text-sm)" }}>
                  {JSON.stringify(editor ?? diff, null, 2)}
                </pre>
              ) : (
                <p style={{ maxWidth: 320 }}>
                  Open a file from the Explorer. The agent's edits arrive here as reviewable diff hunks.
                </p>
              )}
            </div>
          </Panel>
        </section>

        <section style={{ borderTop: "1px solid var(--rim)", padding: "var(--s3)", minHeight: 0 }}>
          <SectionLabel>Terminal</SectionLabel>
          <Panel
            pad="var(--s3)"
            style={{
              height: "calc(100% - 28px)",
              overflowY: "auto",
              fontSize: "var(--text-sm)",
              color: "var(--text-mid)",
              background: "var(--void)",
            }}
          >
            <span style={{ color: "var(--success)" }}>hide</span> ready. The agent's shell runs mirror here
            (xterm over a dedicated PTY socket in the surface pass).
          </Panel>
        </section>
      </div>
    </div>
  );
}
