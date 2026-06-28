/*
  Explorer.tsx: the file tree + search box (the workshop's quiet west list). Re-housed from the
  VS-Code explorer pattern, recast in v3: a calm airy list (14px, --text-2, generous row height), no
  VS Code chrome, no blue twisties. Click a file -> OpenFile{path} intent. Search runs the code_index
  connector (callConnector("code_index","search",{q})); the mock returns [], so we fall back to a local
  name-filter over the seeded tree so the surface is ALIVE with no backend.

  Touched-by-run files wear a faint light glyph (the agent's footprint); git badges wear a +/M letter
  (shape, not color alone). The active file rests at --text-1 with a faint --inner-glow row, the one
  thing the light catches.
*/
import { useCallback, useEffect, useState, type CSSProperties } from "react";
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
      <div style={{ padding: "var(--ma-4) var(--ma-4) var(--ma-3)" }}>
        <div className="t-label">Explorer</div>
        <div style={{ position: "relative", marginTop: "var(--ma-3)" }}>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search the repo"
            spellCheck={false}
            className="t-body"
            style={{
              width: "100%",
              padding: "var(--ma-2) var(--ma-3)",
              borderRadius: "var(--radius)",
              background: "var(--void)",
              color: "var(--text-1)",
              fontFamily: "var(--font)",
              border: "none",
              boxShadow: "var(--hairline)",
              outline: "none",
            }}
          />
          {searching ? (
            <span
              aria-hidden
              className="alive"
              style={{
                position: "absolute",
                right: 10,
                top: 11,
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: "var(--light)",
              }}
            />
          ) : null}
        </div>
      </div>

      <div style={{ overflowY: "auto", padding: "0 var(--ma-3) var(--ma-4)", minHeight: 0, flex: 1 }}>
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
          gap: "var(--ma-2)",
          width: "100%",
          textAlign: "left",
          padding: "5px var(--ma-2)",
          paddingLeft: 10 + depth * 16,
          borderRadius: "var(--radius)",
          fontSize: "14px",
          lineHeight: 1.8,
          // the active file rests at --text-1 with a faint inner-glow catch; everything else is quiet.
          color: selected ? "var(--text-1)" : "var(--text-2)",
          background: selected ? "var(--concrete-3)" : "transparent",
          boxShadow: selected ? "var(--inner-glow)" : undefined,
        }}
      >
        <span style={{ width: 12, color: "var(--text-3)", userSelect: "none" }}>
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

// Git/agent badge: a glyph (shape) + token color, never color alone. The only two colors are --ok and
// --bad; a 'modified' state has no hue (there is no third color), so it reads as a neutral glyph.
function Badge({ kind }: { kind: NonNullable<FileNode["badge"]> }) {
  const map = {
    added: { ch: "+", color: "var(--ok)" },
    modified: { ch: "M", color: "var(--text-2)" },
    touched: { ch: "▪", color: "var(--light)" },
  } as const;
  const m = map[kind];
  return (
    <span title={kind} style={{ color: m.color, fontSize: "12px", width: 14, textAlign: "center" }}>
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
    return <div className="t-body" style={{ color: "var(--text-3)", padding: "var(--ma-4)" }}>No matches for "{query}"</div>;
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
              padding: "var(--ma-2)",
              borderRadius: "var(--radius)",
              fontSize: "14px",
              color: "var(--text-2)",
            }}
          >
            <span style={{ color: "var(--text-3)", fontSize: "12px" }}>
              {h.path}
              {h.line ? <span style={{ color: "var(--text-2)" }}>:{h.line}</span> : null}
            </span>
            <div className="t-code" style={{ color: "var(--text-2)", whiteSpace: "pre", overflow: "hidden", textOverflow: "ellipsis" }}>{h.preview}</div>
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
