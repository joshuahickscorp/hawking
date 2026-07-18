/*
  Explorer.tsx: the navigator. A filter/search field sits at the top (Search folded in, Xcode-style):
  empty -> the file tree; a query -> inline code_index.search results (local fallback over MOCK_TREE).
  Click a file -> OpenFile{path}. Rows are quiet 22px lines; file glyphs are neutral (color is never
  identity); the active file rests on the selected row.
*/
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { callConnector, sendIntent } from "../../ipc";
import { intent } from "../../wire";
import { flattenVisible, treeKeyTarget, type TreeKey } from "../../shell/a11y";
import { MOCK_TREE, type FileNode } from "./types";

const TREE_KEYS = new Set(["ArrowDown", "ArrowUp", "ArrowRight", "ArrowLeft", "Home", "End"]);

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
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  // The real workspace tree from the fs connector; falls back to the mock tree when no backend (dev).
  const [tree, setTree] = useState<FileNode[]>(MOCK_TREE);

  useEffect(() => {
    let alive = true;
    callConnector<{ tree?: FileNode[] }>("fs", "tree", {})
      .then((r) => {
        if (alive && Array.isArray(r?.tree) && r.tree.length) setTree(r.tree);
      })
      .catch(() => void 0); // keep the mock fallback
    return () => {
      alive = false;
    };
  }, []);

  const open = useCallback(
    (path: string, line?: number) => {
      void sendIntent(intent.openFile(path, line));
      onOpen(path);
    },
    [onOpen],
  );

  const toggle = (path: string) => setCollapsed((c) => ({ ...c, [path]: !c[path] }));

  // Roving-tabindex focus for the ARIA tree: one row is tabbable; arrows move between visible rows
  // and expand/collapse dirs. `flattenVisible`/`treeKeyTarget` carry the (tested) navigation logic.
  const treeRef = useRef<HTMLDivElement>(null);
  const [focusPath, setFocusPath] = useState<string | null>(null);
  const rows = useMemo(() => flattenVisible(tree, collapsed), [tree, collapsed]);
  const activeFocus = focusPath ?? activePath ?? rows[0]?.path ?? null;

  const focusRow = (path: string) => {
    setFocusPath(path);
    treeRef.current?.querySelector<HTMLElement>(`[data-treepath="${CSS.escape(path)}"]`)?.focus();
  };

  const onTreeKeyDown = (e: ReactKeyboardEvent) => {
    if (!activeFocus || !TREE_KEYS.has(e.key)) return;
    const target = treeKeyTarget(rows, activeFocus, e.key as TreeKey);
    if (!target) return;
    e.preventDefault();
    if (target.kind === "focus") focusRow(target.path);
    else {
      setCollapsed((c) => ({ ...c, [target.path]: !target.expand }));
      setFocusPath(target.path);
    }
  };

  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setHits(null);
      return;
    }
    let live = true;
    const t = setTimeout(async () => {
      let result: SearchHit[] = [];
      try {
        const raw = await callConnector<unknown>("code_index", "search", { q, limit: 40 });
        if (Array.isArray(raw)) result = raw as SearchHit[];
      } catch {
        result = [];
      }
      if (result.length === 0) result = localSearch(tree, q);
      if (live) setHits(result);
    }, 140);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [query]);

  return (
    <div className="vsc-tree">
      <div className="nav-filter">
        <span className="nav-filter__glyph" aria-hidden>⌕</span>
        <input
          className="nav-filter__input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Filter or search"
          spellCheck={false}
          aria-label="Filter or search files"
        />
        {query ? (
          <button className="nav-filter__clear" title="Clear" aria-label="Clear" onClick={() => setQuery("")}>
            ×
          </button>
        ) : null}
      </div>

      {hits ? (
        <SearchResults hits={hits} query={query} onOpen={open} />
      ) : (
        <div ref={treeRef}>
          <ul style={list} role="tree" aria-label="Files" onKeyDown={onTreeKeyDown}>
            {tree.map((n) => (
              <TreeRow
                key={n.path}
                node={n}
                depth={0}
                activePath={activePath}
                focusPath={activeFocus}
                collapsed={collapsed}
                onToggle={(p) => {
                  toggle(p);
                  setFocusPath(p);
                }}
                onOpen={(p) => {
                  setFocusPath(p);
                  open(p);
                }}
              />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function TreeRow({
  node,
  depth,
  activePath,
  focusPath,
  collapsed,
  onToggle,
  onOpen,
}: {
  node: FileNode;
  depth: number;
  activePath: string | null;
  focusPath: string | null;
  collapsed: Record<string, boolean>;
  onToggle: (path: string) => void;
  onOpen: (path: string) => void;
}) {
  const isOpen = !collapsed[node.path];
  const selected = !node.dir && node.path === activePath;
  return (
    <li role="treeitem" aria-level={depth + 1} aria-selected={selected} aria-expanded={node.dir ? isOpen : undefined}>
      <button
        className={"vsc-row" + (selected ? " vsc-row--selected" : "")}
        data-treepath={node.path}
        tabIndex={node.path === focusPath ? 0 : -1}
        onClick={() => (node.dir ? onToggle(node.path) : onOpen(node.path))}
        style={{ paddingLeft: 8 + depth * 12 }}
      >
        {Array.from({ length: depth }).map((_, i) => (
          <span key={i} className="vsc-guide" style={{ left: 8 + i * 12 + 6 }} aria-hidden />
        ))}
        <span className="vsc-twisty" aria-hidden>
          {node.dir ? (isOpen ? "▾" : "▸") : ""}
        </span>
        <FileGlyph node={node} open={isOpen} />
        <span className="vsc-row__name">{node.name}</span>
        {node.badge ? <Badge kind={node.badge} /> : null}
      </button>
      {node.dir && isOpen && node.children ? (
        <ul style={list} role="group">
          {node.children.map((c) => (
            <TreeRow
              key={c.path}
              node={c}
              depth={depth + 1}
              activePath={activePath}
              focusPath={focusPath}
              collapsed={collapsed}
              onToggle={onToggle}
              onOpen={onOpen}
            />
          ))}
        </ul>
      ) : null}
    </li>
  );
}

// File glyph: folder twistie for dirs, a neutral extension letter-tile for files. No per-type hue
// (color is never identity); the selected file brightens via the row, not a colored dot.
function FileGlyph({ node, open }: { node: FileNode; open: boolean }) {
  if (node.dir) {
    return (
      <span className="vsc-glyph vsc-glyph--folder" aria-hidden>
        {open ? "▾" : "▸"}
      </span>
    );
  }
  const ext = node.name.split(".").pop() ?? "";
  return (
    <span className="vsc-glyph" aria-hidden>
      {ext ? ext[0].toUpperCase() : "#"}
    </span>
  );
}

// Git/agent badge: a glyph + (the only) two pigments, never color alone.
function Badge({ kind }: { kind: NonNullable<FileNode["badge"]> }) {
  const map = {
    added: { ch: "U", color: "var(--git-add)" },
    modified: { ch: "M", color: "var(--git-mod)" },
    touched: { ch: "●", color: "var(--text-dim)" },
  } as const;
  const m = map[kind];
  return (
    <span className="vsc-badge" title={kind} style={{ color: m.color }}>
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
  onOpen: (path: string, line?: number) => void;
}) {
  if (hits.length === 0) {
    return <div className="sidebar__empty">No results for "{query}"</div>;
  }
  return (
    <ul className="search-view__list">
      {hits.map((h, i) => (
        <li key={i}>
          <button className="ghost-button search-hit" onClick={() => onOpen(h.path, h.line || undefined)}>
            <span className="search-hit__path">
              {h.path}
              {h.line ? <span className="search-hit__line">:{h.line}</span> : null}
            </span>
            {h.preview && h.preview !== h.path ? <span className="search-hit__preview t-code">{h.preview}</span> : null}
          </button>
        </li>
      ))}
    </ul>
  );
}

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
