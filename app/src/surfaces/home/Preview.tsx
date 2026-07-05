/*
  Preview.tsx — the live preview panel (Claude Code's localhost view, recast). A minimal browser: a URL
  bar, reload, and an iframe onto a local dev server the agent is running. Local only by intent; there is
  no chrome beyond what you need to see the work. Empty until you point it at a server.
*/
import { useState } from "react";
import { Icon } from "../../shell/icons";

// Prepend a scheme if the user typed a bare host:port. Local dev servers only, so http is the default.
function normalize(raw: string): string {
  const t = raw.trim();
  if (!t) return "";
  if (/^https?:\/\//i.test(t)) return t;
  return "http://" + t;
}

export function Preview({ initialUrl = "" }: { initialUrl?: string }) {
  const [url, setUrl] = useState(initialUrl);
  const [live, setLive] = useState(normalize(initialUrl));
  const [nonce, setNonce] = useState(0); // bump to force the iframe to reload

  const go = () => setLive(normalize(url));

  return (
    <div className="preview">
      <div className="preview__bar">
        <button className="preview__btn" title="Reload" aria-label="Reload" onClick={() => setNonce((n) => n + 1)} disabled={!live}>
          <Icon name="reload" size={14} />
        </button>
        <div className="preview__url">
          <Icon name="globe" size={13} />
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") go();
            }}
            placeholder="localhost:3000"
            spellCheck={false}
            aria-label="Preview URL"
          />
        </div>
        <button className="preview__btn preview__btn--go" title="Load" aria-label="Load" onClick={go}>
          Go
        </button>
      </div>
      {live ? (
        <iframe key={nonce} src={live} className="preview__frame" title="Preview" sandbox="allow-scripts allow-same-origin allow-forms" />
      ) : (
        <div className="preview__empty">
          <span className="preview__mark" aria-hidden>
            <Icon name="globe" size={22} />
          </span>
          <div className="t-body">Point at a local dev server</div>
          <div className="t-micro">Type a host, e.g. localhost:3000, and the agent's running app renders here.</div>
        </div>
      )}
    </div>
  );
}
