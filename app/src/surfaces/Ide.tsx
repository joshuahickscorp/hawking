import { useEffect, useMemo, useState } from "react";
import { TRANSPORT_KIND } from "../ipc";
import { useStore } from "../store";
import { SectionLabel } from "../ui";
import { EditorGroup } from "./ide/Editor";
import { Explorer } from "./ide/Explorer";
import { Terminal } from "./ide/Terminal";
import { MOCK_DIFF, parseDiff, type DiffDoc } from "./ide/types";

export function Ide() {
  const diffPatch = useStore((s) => s.projections.diff);
  const editorPatch = useStore((s) => s.projections.editor);
  const [openPath, setOpenPath] = useState<string | null>("crates/pool/src/guard.rs");
  const [diff, setDiff] = useState<DiffDoc | null>(null);

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

  useEffect(() => {
    const path = (editorPatch as { path?: string } | undefined)?.path;
    if (path) setOpenPath(path);
  }, [editorPatch]);

  const reviewing = diff != null && diff.hunks.some((h) => h.status === "pending");

  return (
    <div className="ide-grid">
      <aside className="volume ide-panel">
        <Explorer activePath={openPath} onOpen={setOpenPath} />
      </aside>

      <div className="editor-stack">
        <section
          className={["volume", "ide-panel", reviewing && "alive"].filter(Boolean).join(" ")}
          style={{ position: "relative", display: "flex", flexDirection: "column" }}
        >
          {reviewing ? <DiffBanner path={diff.path} count={diff.hunks.filter((h) => h.status === "pending").length} /> : null}
          <div style={{ flex: 1, minHeight: 0, paddingTop: reviewing ? 34 : 0 }}>
            <EditorGroup openPath={openPath} diff={diff} onDiffChange={setDiff} />
          </div>
        </section>

        <section className="volume ide-panel" style={{ display: "flex", flexDirection: "column" }}>
          <div className="panel-bar">
            <SectionLabel>Terminal</SectionLabel>
            <span className="t-micro">agent commands mirror here</span>
          </div>
          <div style={{ flex: 1, minHeight: 0, padding: "var(--ma-3) var(--ma-4) var(--ma-4)" }}>
            <Terminal />
          </div>
        </section>
      </div>
    </div>
  );
}

function DiffBanner({ path, count }: { path: string; count: number }) {
  return (
    <div role="status" className="diff-banner t-micro">
      <span style={{ color: "var(--light)" }}>Change proposed</span>
      <span style={{ color: "var(--text-2)" }}>{path}</span>
      <span style={{ marginLeft: "auto", color: "var(--text-3)" }}>
        {count} hunk{count === 1 ? "" : "s"}
      </span>
    </div>
  );
}
