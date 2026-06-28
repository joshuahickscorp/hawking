/*
  Explorer.tsx: the file tree + search box (D1.2 / D4.4 #4). Re-housed from the Void/VS-Code
  explorer pattern, re-skinned to near-black mono with gold accents (no VS Code chrome, no twisties
  in blue). Click a file -> OpenFile{path} intent. Search runs the code_index connector
  (callConnector("code_index","search",{q})); the mock returns [], so we fall back to a local
  name-filter over the seeded tree so the surface is ALIVE with no backend.

  Touched-by-run files wear a small gold dot (the agent's footprint), git badges wear a +/M letter
  (shape, not color alone). Selected row wears the gold rim, the one lit thing.
*/
import { useCallback, useEffect, useState, type CSSProperties, type ReactNode } from "react";
import { callConnector, sendIntent } from "../../ipc";
import { intent } from "../../wire";
import { MOCK_TREE, type FileNode } from "./types";

interface SearchHit {
  path: string;
  line: number;
  preview: string;
}

export function Explorer({
  activePath,
  onOpen,
}: {
  activePath: string | null;
  onOpen: (path: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const open = useCallback(
    (path: string) => {
      void sendIntent(intent.openFile(path));
      onOpen(path);
    },
    [onOpen],
  );

  // Search the code_index connector; if it returns nothing (mock), fall back to a local name filter.
  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setHits(null);
      setSearching(false);
      return;
    }
    let live = true;
    setSearching(true);
    const t = setTimeout(async () => {
      let result: SearchHit[] = [];
      try {
        const raw = await callConnector<unknown>("code_index", "search", { q, limit: 40 });
        if (Array.isArray(raw)) result = raw as SearchHit[];
      } catch {
        result = [];
      }
      if (result.length === 0) result = localSearch(MOCK_TREE, q);
      if (live) {
        setHits(result);
        setSearching(false);
      }
    }, 140);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [query]);

  const toggle = (path: string) => setCollapsed((c) => ({ ...c, [path]: !c[path] }));

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div style={{ padding: "var(--s3) var(--s3) var(--s2)" }}>
        <Label>Explorer</Label>
        <div style={{ position: "relative", marginTop: "var(--s2)" }}>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search the repo"
            spellCheck={false}
            style={{
              width: "100%",
              padding: "5px var(--s3)",
              borderRadius: "var(--radius)",
              background: "var(--void)",
              color: "var(--text-hi)",
              fontFamily: "var(--font-mono)",
              fontSize: "var(--text-sm)",
              border: "none",
              boxShadow: "inset 0 0 0 1px var(--rim)",
              outline: "none",
            }}
          />
          {searching ? (
            <span
              aria-hidden
              style={{
                position: "absolute",
                right: 8,
                top: 7,
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: "var(--radiation)",
                animation: "radiation-breathe 1.4s ease-in-out infinite",
              }}
            />
          ) : null}
        </div>
      </div>

      <div style={{ overflowY: "auto", padding: "0 var(--s2) var(--s3)", minHeight: 0, flex: 1 }}>
        {hits ? (
          <SearchResults hits={hits} query={query} onOpen={(p, line) => { void sendIntent(intent.openFile(p, line)); onOpen(p); }} />
        ) : (
          <ul style={list}>
            {MOCK_TREE.map((n) => (
              <TreeRow key={n.path} node={n} depth={0} activePath={activePath} collapsed={collapsed} onToggle={toggle} onOpen={open} />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function TreeRow({
  node,
  depth,
  activePath,
  collapsed,
  onToggle,
  onOpen,
}: {
  node: FileNode;
  depth: number;
  activePath: string | null;
  collapsed: Record<string, boolean>;
  onToggle: (path: string) => void;
  onOpen: (path: string) => void;
}) {
  const isOpen = !collapsed[node.path];
  const selected = !node.dir && node.path === activePath;
  return (
    <li>
      <button
        onClick={() => (node.dir ? onToggle(node.path) : onOpen(node.path))}
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--s2)",
          width: "100%",
          textAlign: "left",
          padding: "3px var(--s2)",
          paddingLeft: 8 + depth * 14,
          borderRadius: "var(--radius)",
          fontSize: "var(--text-sm)",
          color: selected ? "var(--text-hi)" : node.dir ? "var(--text-mid)" : "var(--text-mid)",
          background: selected ? "var(--surface-2)" : "transparent",
          boxShadow: selected ? "inset 0 0 0 1px var(--radiation)" : undefined,
        }}
      >
        <span style={{ width: 10, color: "var(--text-low)", userSelect: "none" }}>
          {node.dir ? (isOpen ? "▾" : "▸") : ""}
        </span>
        <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{node.name}</span>
        {node.badge ? <Badge kind={node.badge} /> : null}
      </button>
      {node.dir && isOpen && node.children ? (
        <ul style={list}>
          {node.children.map((c) => (
            <TreeRow key={c.path} node={c} depth={depth + 1} activePath={activePath} collapsed={collapsed} onToggle={onToggle} onOpen={onOpen} />
          ))}
        </ul>
      ) : null}
    </li>
  );
}

// Git/agent badge: a letter (shape) + token color, never color alone (C14).
function Badge({ kind }: { kind: NonNullable<FileNode["badge"]> }) {
  const map = {
    added: { ch: "+", color: "var(--diff-add-fg)" },
    modified: { ch: "M", color: "var(--warning)" },
    touched: { ch: "▪", color: "var(--radiation)" },
  } as const;
  const m = map[kind];
  return (
    <span title={kind} style={{ color: m.color, fontSize: "var(--text-xs)", width: 12, textAlign: "center" }}>
      {m.ch}
    </span>
  );
}

function SearchResults({
  hits,
  query,
  onOpen,
}: {
  hits: SearchHit[];
  query: string;
  onOpen: (path: string, line: number) => void;
}) {
  if (hits.length === 0) {
    return <div style={{ color: "var(--text-low)", fontSize: "var(--text-sm)", padding: "var(--s3)" }}>No matches for "{query}".</div>;
  }
  return (
    <ul style={list}>
      {hits.map((h, i) => (
        <li key={i}>
          <button
            onClick={() => onOpen(h.path, h.line)}
            style={{
              display: "block",
              width: "100%",
              textAlign: "left",
              padding: "4px var(--s2)",
              borderRadius: "var(--radius)",
              fontSize: "var(--text-sm)",
              color: "var(--text-mid)",
            }}
          >
            <span style={{ color: "var(--text-low)", fontSize: "var(--text-xs)" }}>
              {h.path}
              {h.line ? <span style={{ color: "var(--radiation)" }}>:{h.line}</span> : null}
            </span>
            <div style={{ color: "var(--text-mid)", whiteSpace: "pre", overflow: "hidden", textOverflow: "ellipsis" }}>{h.preview}</div>
          </button>
        </li>
      ))}
    </ul>
  );
}

// Local fallback so search works against the seeded tree when the connector is the mock.
function localSearch(nodes: FileNode[], q: string): SearchHit[] {
  const out: SearchHit[] = [];
  const lc = q.toLowerCase();
  const walk = (ns: FileNode[]) => {
    for (const n of ns) {
      if (!n.dir && (n.name.toLowerCase().includes(lc) || n.path.toLowerCase().includes(lc))) {
        out.push({ path: n.path, line: 0, preview: n.path });
      }
      if (n.children) walk(n.children);
    }
  };
  walk(nodes);
  return out;
}

const list: CSSProperties = { listStyle: "none", margin: 0, padding: 0 };

function Label({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        fontSize: "var(--text-xs)",
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        color: "var(--text-low)",
      }}
    >
      {children}
    </div>
  );
}
