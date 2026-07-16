/*
  ContextStackState.ts: local steer-state for the Context Stack controls.
  The host owns truth; pin/mute/evict are Custom intents the compiler honors NEXT turn (FE: D1.5).
  So between emitting the intent and the next manifest, we hold an OPTIMISTIC local overlay
  so the toggle feels material and immediate (the OP-1 key resolves the instant you press it).
  This overlay is keyed by a stable id per span and reconciles itself away when a fresh
  manifest arrives carrying the change. No store edit: this is surface-local, by design.
*/
import { useCallback, useState } from "react";

export type SteerKind = "pin" | "mute" | "evict";

// A span's stable identity across manifests: source-kind + its natural key.
export function spanKey(kind: string, key: string): string {
  return `${kind}:${key}`;
}

export interface Steer {
  // is this span currently steered (locally), for the given kind?
  on(id: string, kind: SteerKind): boolean;
  // flip a steer and return the new value (caller emits the matching intent).
  toggle(id: string, kind: SteerKind): boolean;
  // a free-text note injected against a span (or the turn at large when id === "turn").
  noteOn(id: string): string | undefined;
  setNote(id: string, text: string): void;
}

export function useSteer(): Steer {
  // two flat maps keep this dense: one boolean overlay, one note overlay.
  const [flags, setFlags] = useState<Record<string, boolean>>({});
  const [notes, setNotes] = useState<Record<string, string>>({});

  const on = useCallback((id: string, kind: SteerKind) => flags[`${kind}|${id}`] === true, [flags]);

  const toggle = useCallback((id: string, kind: SteerKind) => {
    const k = `${kind}|${id}`;
    let next = false;
    setFlags((f) => {
      next = !f[k];
      return { ...f, [k]: next };
    });
    return next;
  }, []);

  const noteOn = useCallback((id: string) => notes[id], [notes]);
  const setNote = useCallback((id: string, text: string) => {
    setNotes((n) => {
      if (!text) {
        const { [id]: _drop, ...rest } = n;
        return rest;
      }
      return { ...n, [id]: text };
    });
  }, []);

  return { on, toggle, noteOn, setNote };
}
