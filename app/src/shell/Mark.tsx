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
