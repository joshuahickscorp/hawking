/*
  SideBar.tsx — the one side page: the Explorer, with a one-line workcard on top. The model name is
  the affordance: click it for a context popover (live ceiling, .tq multiplier, recall, state) and the
  switch-model action. Nothing else is parked here — forking/fleets auto-invoke, context detail is
  summoned, not displayed. (Invisible until summoned.)
*/
import { useEffect, useRef, useState } from "react";
import { sendIntent } from "../ipc";
import { useStore, type ContextManifest } from "../store";
import { intent } from "../wire";
import { Explorer } from "../surfaces/ide/Explorer";

const fmtTok = (n: number) =>
  n >= 1_000_000 ? `${(n / 1_000_000).toFixed(n % 1_000_000 ? 1 : 0)}M` : n >= 1000 ? `${Math.round(n / 1000)}K` : `${n}`;

export function SideBar({ openPath, onOpen }: { openPath: string | null; onOpen: (path: string) => void }) {
  const manifest = useStore((s) => s.manifest);
  const runtimeStatus = useStore((s) => s.runtimeStatus);
  const pushNotice = useStore((s) => s.pushNotice);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const model = manifest?.model?.id ?? "qwen2.5";
  const ready = runtimeStatus === "ready";

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onEsc);
    };
  }, [open]);

  const switchModel = () => {
    void sendIntent(intent.custom("switch_model", {}));
    pushNotice({ kind: "info", code: "model", message: "switch model" });
    setOpen(false);
  };

  return (
    <nav className="sidebar" aria-label="Explorer">
      <div className="workcard" ref={ref}>
        <div className="workcard__row">
          <span className={"workcard__dot" + (ready ? " workcard__dot--ready" : "")} aria-hidden />
          <button
            className="workcard__model"
            title="Model and live context, click for detail"
            aria-haspopup="dialog"
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            {model}
          </button>
          <span className="workcard__local">local</span>
        </div>
        {open ? <ModelPopover manifest={manifest} ready={ready} onSwitch={switchModel} /> : null}
      </div>
      <div className="sidebar__body">
        <Explorer activePath={openPath} onOpen={onOpen} />
      </div>
    </nav>
  );
}

// The context popover summoned from the model name: the live, measured window (ceiling = native x the
// .tq multiplier, read live) plus recall/state, and the switch-model action. Abundance, not a meter.
function ModelPopover({
  manifest,
  ready,
  onSwitch,
}: {
  manifest: ContextManifest | null;
  ready: boolean;
  onSwitch: () => void;
}) {
  const native = manifest?.ctx_len_native ?? manifest?.model?.ctx;
  const ceiling = manifest?.live?.effective_ceiling_tokens ?? manifest?.ctx_len_effective;
  const mult = manifest?.tq_multiplier;
  const occ = Math.min(Math.max(manifest?.live?.occupancy ?? 0, 0), 1);
  const state = manifest?.recurrent_state_bytes;
  const fidelity = manifest?.live?.recall_fidelity;

  return (
    <div className="modelpop glass" role="dialog" aria-label="Model and context">
      <div className="modelpop__head">
        <span className={"workcard__dot" + (ready ? " workcard__dot--ready" : "")} aria-hidden />
        <span className="modelpop__title">{manifest?.model?.id ?? "qwen2.5"}</span>
        <span className="modelpop__arch">{manifest?.arch ?? manifest?.model?.arch ?? "local"}</span>
      </div>
      <Row label="context" value={ceiling ? `${fmtTok(ceiling)}${mult && mult > 1 ? `   ${mult.toFixed(1)}x .tq` : ""}` : "loaded, cached"} />
      {native ? <Row label="native" value={fmtTok(native)} /> : null}
      {ceiling ? (
        <div className="modelpop__bar" aria-hidden>
          <span className="modelpop__fill" style={{ width: `${Math.round(occ * 100)}%` }} />
        </div>
      ) : null}
      {typeof fidelity === "number" ? <Row label="recall" value={`${Math.round(fidelity * 100)}%`} /> : null}
      {state ? <Row label="state" value={`${Math.round(state / 1e6)} MB`} /> : null}
      <button className="modelpop__switch" onClick={onSwitch}>switch model</button>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="modelpop__row">
      <span className="modelpop__label">{label}</span>
      <span className="modelpop__value">{value}</span>
    </div>
  );
}
