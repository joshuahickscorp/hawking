import { Component, type ErrorInfo, type ReactNode } from "react";

// --- Mark.tsx ---
/*
  Mark.tsx — the HIDE logo, serialized to vector from the PSD. The 'h' is the Geist-Black glyph outline
  (y-flipped for SVG); the ball is the Comfortaa round dot. currentColor, so it takes the brand color.
  The app wordmark uses just the h; the full mark (ball + h on the diagonal) is here for reuse.
*/
const H_PATH =
  "M56 710V0H250V272C250 348 270 391 319 391C371 391 380 348 380 272V0H575V341C575 459 508 542 397 542C335 542 281 523 250 467V710Z";

export function LogoH({ size = 18 }: { size?: number }) {
  return (
    <svg height={size} viewBox="56 0 519 710" fill="currentColor" role="img" aria-label="HIDE">
      <g transform="translate(0,710) scale(1,-1)">
        <path d={H_PATH} />
      </g>
    </svg>
  );
}

export function LogoMark({ size = 28 }: { size?: number }) {
  // Coordinates derived from the raster generator (ball diameter 0.40 of the art, h = 0.66 of the
  // ball, 45-degree gap 0.70 of the ball, group centered) so the vector matches the PNG exactly.
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="currentColor" role="img" aria-label="HIDE">
      <circle cx="42.41" cy="59.12" r="17.2" />
      <g transform="translate(56.40,46.39) scale(0.03197,-0.03197)">
        <path d={H_PATH} />
      </g>
    </svg>
  );
}

// --- ErrorBoundary.tsx ---
/*
  ErrorBoundary.tsx — a crash never white-screens HIDE. Catches a render error anywhere below, logs it,
  and shows a concrete recover panel (flight-log voice) instead of a blank page.
*/

export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Local crash log (no egress). A crash-report file is the second-plan hardening item.
    console.error("HIDE render error:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="crash" role="alert">
          <div className="crash__box glass">
            <div className="crash__title">HIDE hit a render error</div>
            <pre className="crash__detail">{this.state.error.message}</pre>
            <button className="crash__btn" onClick={() => location.reload()}>reload</button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// --- glass.ts ---
/*
  glass.ts — engine-aware glass. Feature-detect (never UA sniff) whether the runtime can drive an SVG
  filter through backdrop-filter. Chromium-class accepts url() in backdrop-filter; WebKit-class accepts
  only filter functions (blur, saturate) and silently no-ops a url(), so we must not emit one there.

  Chromium-class: mark the root data-glass="refract" and inject the edge-refraction filter; the CSS
  upgrades only the FIXED chrome (toolbar, status strip) to a bent-light backdrop. Resizable panes
  (Executor, popovers, palette) stay on rim plus frost plus grain, since one displacement map assumes
  fixed dimensions and would break on a resizing element.

  WebKit-class (or anything ambiguous): data-glass="frost", rim plus frost plus grain everywhere. That
  path works on every engine, so it is the safe default.
*/
export function initGlass(): void {
  const root = document.documentElement;
  let refract = false;
  try {
    refract = typeof CSS !== "undefined" && !!CSS.supports && CSS.supports("backdrop-filter", "url(#p)");
  } catch {
    refract = false;
  }
  root.dataset.glass = refract ? "refract" : "frost";
  if (!refract || document.getElementById("hide-glass-defs")) return;

  // A gentle, low-frequency displacement plus a faint specular highlight: the void's light bending
  // through the pane. Small scale on purpose (correctness over flash). Only the fixed chrome uses it.
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.id = "hide-glass-defs";
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("style", "position:absolute;width:0;height:0;pointer-events:none");
  svg.innerHTML =
    '<defs><filter id="hide-glass-refract" x="-2%" y="-2%" width="104%" height="104%">' +
    '<feTurbulence type="fractalNoise" baseFrequency="0.012 0.018" numOctaves="1" seed="7" result="n"/>' +
    '<feGaussianBlur in="n" stdDeviation="1.4" result="nb"/>' +
    '<feDisplacementMap in="SourceGraphic" in2="nb" scale="5" xChannelSelector="R" yChannelSelector="G"/>' +
    "</filter></defs>";
  document.body.appendChild(svg);
}

// --- Radiate.tsx ---
/*
  Radiate.tsx — HIDE's one progress signature, the event-horizon ring: a dark disc with a thin arc of
  LIGHT on the rim (the agent radiating, the cross of light entering the dark). This is the only loading
  indicator in the app. Two modes:
    - indeterminate: the arc sweeps (a run is in flight, no laddered progress yet).
    - laddered: pass `stage` (0..stages) and the arc length encodes the oracle ladder (build, typecheck,
      test, lint). It sharpens as work lands and does not spin.
  Honors prefers-reduced-motion (static lit arc).
*/
export function Radiate({
  size = 16,
  active = true,
  stage,
  stages = 4,
  title,
}: {
  size?: number;
  active?: boolean;
  stage?: number;
  stages?: number;
  title?: string;
}) {
  const sw = Math.max(1.25, size / 12);
  const r = size / 2 - sw;
  const c = 2 * Math.PI * r;
  const laddered = typeof stage === "number";
  const clamped = laddered ? Math.min(Math.max(stage as number, 0), stages) : 0;
  // indeterminate arc is a fixed wedge that sweeps; laddered arc grows from a stub toward full.
  const frac = laddered ? 0.12 + 0.8 * (clamped / stages) : 0.26;
  const label = title ?? (laddered ? `verifying ${clamped} of ${stages}` : active ? "working" : "idle");
  return (
    <span
      className={["radiate", active && !laddered && "radiate--active", laddered && "radiate--laddered"].filter(Boolean).join(" ")}
      style={{ width: size, height: size }}
      role="img"
      aria-label={label}
      title={title}
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} fill="none">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="var(--line-strong)" strokeWidth={sw} />
        <circle
          className="radiate__arc"
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke="var(--light)"
          strokeWidth={sw}
          strokeLinecap="round"
          strokeDasharray={`${c * frac} ${c}`}
        />
      </svg>
    </span>
  );
}
