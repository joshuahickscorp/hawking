/*
  Explorer.tsx: the navigator. A filter/search field sits at the top (Search folded in, Xcode-style):
  empty -> the file tree; a query -> results from THE one search engine (src/ui.tsx), scoped to the
  editor origin (files, symbols, references). Click a file -> the `open_file` command, and the tab
  opens here in place. Rows are quiet 22px lines; file glyphs are neutral (color is never identity);
  the active file rests on the selected row.

  RETIRED with this stage: the private search this file used to run. It dialed code_index.search with
  `{ q, limit }`, a shape the connector cannot deserialize (it takes `{ query: SearchQuery }` and
  answers `{ results }`, not a bare array), so every keystroke fell through to the local tree walk and
  the panel had never once shown a real index hit. One engine now serves this field and the palette,
  so a result means the same thing in both, and the local tree walk survives only as an explicitly
  labelled fallback when the index answers nothing.
*/
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { callConnector } from "../../ipc";
import { flattenVisible, treeKeyTarget, type TreeKey } from "../../shell/a11y";
import {
  HIT_ACTION_HINT,
  HitRow,
  defaultScopes,
  hitActionFor,
  nextIndex,
  runHitAction,
  searchAll,
  setSearchOrigin,
  type HitAction,
  type SearchHit,
} from "../../ui";
import { runCommand, useStore } from "../../store";
import { MOCK_TREE, mockOnly, type FileNode } from "./types";

const TREE_KEYS = new Set(["ArrowDown", "ArrowUp", "ArrowRight", "ArrowLeft", "Home", "End"]);
const LIST_KEYS = new Set(["ArrowDown", "ArrowUp", "Home", "End"]);

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
  const [errors, setErrors] = useState<string[]>([]);
  const [sel, setSel] = useState(0);
  const [status, setStatus] = useState<string | null>(null);
  // The real workspace tree from the fs connector. The mock tree stands in ONLY on the mock
  // transport: on a live host a failed read shows why it failed, never invented paths.
  const [tree, setTree] = useState<FileNode[]>(() => mockOnly(MOCK_TREE) ?? []);
  const [treeNote, setTreeNote] = useState<string | null>(null);
  const sessionId = useStore((s) => s.sessionId);
  const runId = useStore((s) => s.activeRunId);

  useEffect(() => {
    let alive = true;
    callConnector<{ tree?: FileNode[] }>("fs", "tree", {})
      .then((r) => {
        if (!alive) return;
        if (Array.isArray(r?.tree) && r.tree.length) setTree(r.tree);
        else setTreeNote("The host listed no files for this workspace");
      })
      .catch((e) => {
        if (alive) setTreeNote(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, []);

  // While the navigator holds the search focus, the global palette defaults to the editor scopes.
  useEffect(() => {
    setSearchOrigin("editor");
    return () => setSearchOrigin("global");
  }, []);

  const open = useCallback(
    (path: string, line?: number) => {
      void runCommand("open_file", { path, line: line ?? null });
      onOpen(path); // open in the CURRENT tab set, not a new surface
    },
    [onOpen],
  );

  // One activation path for every result row, shared with the palette: Enter opens, Mod+Enter
  // attaches the result to the turn, Mod+Shift+Enter starts a side chat from it.
  const activateHit = useCallback(
    (action: HitAction, hit: SearchHit) => {
      if (action === "open" && hit.path) {
        open(hit.path, hit.line);
        setStatus(null);
        return;
      }
      void runHitAction(action, hit, { sessionId, runId: runId ?? "" })
        .then((ack) =>
          setStatus(
            !ack.accepted
              ? ack.message ?? "The host refused that action"
              : action === "side_chat"
                ? "Side chat started from this result"
                : runId
                  ? "Attached to the running turn"
                  : "Sent as a new turn",
          ),
        )
        .catch((e) => setStatus(e instanceof Error ? e.message : String(e)));
    },
    [open, runId, sessionId],
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
      setErrors([]);
      return;
    }
    let live = true;
    const t = setTimeout(async () => {
      const r = await searchAll(q, defaultScopes("editor"), { sessionId, runId: runId ?? "" });
      if (!live) return;
      setSel(0);
      setErrors(r.errors);
      // Nothing indexed yet (or no backend): fall back to filtering the tree already on screen, and
      // say so, so a local name match is never mistaken for an index hit.
      setHits(r.hits.length ? r.hits : localSearch(tree, q));
    }, 140);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [query, tree, sessionId, runId]);

  return (
    <div className="vsc-tree">
      <div className="nav-filter">
        <span className="nav-filter__glyph" aria-hidden>⌕</span>
        <input
          className="nav-filter__input"
          value={query}
          role="combobox"
          aria-expanded={hits != null}
          aria-controls="explorer-results"
          aria-activedescendant={hits?.length ? `explorer-result-${Math.min(sel, hits.length - 1)}` : undefined}
          onChange={(e) => {
            setQuery(e.target.value);
            setStatus(null);
          }}
          onKeyDown={(e) => {
            if (!hits?.length) return;
            if (LIST_KEYS.has(e.key)) {
              e.preventDefault();
              setSel((i) => nextIndex(hits.length, i, e.key));
            } else if (e.key === "Enter") {
              e.preventDefault();
              activateHit(hitActionFor(e), hits[Math.min(sel, hits.length - 1)]);
            }
          }}
          placeholder="Filter or search"
          spellCheck={false}
          title={HIT_ACTION_HINT}
          aria-label="Filter or search files, symbols and references"
        />
        {query ? (
          <button className="nav-filter__clear" title="Clear" aria-label="Clear" onClick={() => setQuery("")}>
            ×
          </button>
        ) : null}
      </div>

      {hits ? (
        <SearchResults
          hits={hits}
          query={query}
          sel={sel}
          notes={[status, ...errors].filter(Boolean) as string[]}
          onSelect={setSel}
          onActivate={activateHit}
        />
      ) : tree.length === 0 ? (
        <div className="sidebar__empty" role="status">
          {treeNote ?? "Reading the workspace"}
        </div>
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
  sel,
  notes,
  onSelect,
  onActivate,
}: {
  hits: SearchHit[];
  query: string;
  sel: number;
  notes: string[];
  onSelect: (i: number) => void;
  onActivate: (action: HitAction, hit: SearchHit) => void;
}) {
  if (hits.length === 0) {
    return <div className="sidebar__empty">No results for "{query}"</div>;
  }
  return (
    <>
      <ul className="search-view__list" id="explorer-results" role="listbox" aria-label="Search results">
        {hits.map((h, i) => (
          <li key={h.key} role="presentation">
            <HitRow
              id={`explorer-result-${i}`}
              hit={h}
              selected={i === sel}
              onHover={() => onSelect(i)}
              onActivate={onActivate}
            />
          </li>
        ))}
      </ul>
      <div role="status" aria-live="polite" className="t-micro" style={note}>
        {notes.join(" . ")}
      </div>
    </>
  );
}

/** Fallback when the index answers nothing: filter the tree already on screen. Labelled as such
 *  (provenance says "workspace tree") so a local name match never poses as an index hit. */
function localSearch(nodes: FileNode[], q: string): SearchHit[] {
  const out: SearchHit[] = [];
  const lc = q.toLowerCase();
  const walk = (ns: FileNode[]) => {
    for (const n of ns) {
      if (!n.dir && (n.name.toLowerCase().includes(lc) || n.path.toLowerCase().includes(lc))) {
        out.push({ key: `tree:${n.path}`, scope: "files", title: n.path, preview: "workspace tree", path: n.path });
      }
      if (n.children) walk(n.children);
    }
  };
  walk(nodes);
  return out;
}

const note: CSSProperties = { padding: "0 var(--ma-2) var(--ma-2)", color: "var(--text-dim)" };

const list: CSSProperties = { listStyle: "none", margin: 0, padding: 0 };
