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
