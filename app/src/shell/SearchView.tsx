/*
  SearchView.tsx — the Search sidebar view (VS Code's search panel). Queries the code_index connector
  and falls back to a local name filter over the seeded tree so it works with the mock transport.
*/
import { useEffect, useState } from "react";
import { callConnector, sendIntent } from "../ipc";
import { intent } from "../wire";
import { MOCK_TREE, type FileNode } from "../surfaces/ide/types";

interface SearchHit {
  path: string;
  line: number;
  preview: string;
}

export function SearchView({ onOpen }: { onOpen: (path: string) => void }) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[] | null>(null);

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
      if (result.length === 0) result = localSearch(MOCK_TREE, q);
      if (live) setHits(result);
    }, 140);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [query]);

  return (
    <div className="search-view">
      <div className="search-view__field">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search"
          spellCheck={false}
          className="search-view__input"
        />
      </div>
      <div className="search-view__results">
        {hits == null ? null : hits.length === 0 ? (
          <div className="sidebar__empty">No results</div>
        ) : (
          <ul className="search-view__list">
            {hits.map((h, i) => (
              <li key={i}>
                <button
                  className="ghost-button search-hit"
                  onClick={() => {
                    void sendIntent(intent.openFile(h.path, h.line || undefined));
                    onOpen(h.path);
                  }}
                >
                  <span className="search-hit__path">
                    {h.path}
                    {h.line ? <span className="search-hit__line">:{h.line}</span> : null}
                  </span>
                  {h.preview && h.preview !== h.path ? (
                    <span className="search-hit__preview t-code">{h.preview}</span>
                  ) : null}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
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
