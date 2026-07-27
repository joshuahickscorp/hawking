/*
  ModelChooser.tsx - the ONE place the model choice is presented (consolidation decision 3.4).

  `switch_model` had three entry points (SideBar popover, HomeComposer instrument row, Settings), all
  firing `Intent::Custom{switch_model, {}}` with an empty payload, and the host treats that name as
  LOG-ONLY: there is no model-switch capability anywhere in the catalog or on the host. Three buttons
  therefore promised the same thing three times and delivered it zero times.

  The decision is MERGE onto one chooser and, with no host capability, label it honestly rather than
  pretend. So this component states what is loaded and why there is nothing to switch to, the SideBar
  popover and Settings both render it, and the composer shows the same id as a plain label. When a
  real model-switch capability lands it gets wired HERE, once, and all three surfaces follow.
*/
import type { ContextManifest } from "../store";

/** What the app says when it does not yet know. Four surfaces used to print a hardcoded id here
 *  ("qwen2.5-7b" in two, "qwen2.5" in two, disagreeing with each other), which rendered an invented
 *  model as the live loaded one. The manifest is the ONLY source; absent means unknown, the way
 *  StatusBar's `branchLabel` says "no branch" instead of inventing "main". */
export const MODEL_ID_UNKNOWN = "no model reported";

/** Why there is nothing to switch to, told against what is ACTUALLY loaded. This was a bare const
 *  asserting "One local model is loaded by the runtime", rendered unconditionally on three surfaces,
 *  so a model-free host printed it beside this file's own MODEL_ID_UNKNOWN and said both at once. The
 *  manifest stays the single source, exactly as `modelId` reads it. */
export function modelSwitchNote(manifest: ContextManifest | null): string {
  const loaded = manifest?.model?.id
    ? "One local model is loaded by the runtime."
    : "No model is loaded by the runtime.";
  return `${loaded} This build has no model-switch capability, so there is nothing to switch to yet.`;
}

/** THE model label. Every surface reads it here, so there is one answer and it is never invented. */
export function modelId(manifest: ContextManifest | null): string {
  return manifest?.model?.id ?? MODEL_ID_UNKNOWN;
}

/** `popover` sits in the SideBar context popover (which already titles the model); `panel` sits in
 *  the Settings Model section and carries its own labelled row. */
export function ModelChooser({
  manifest,
  tone,
}: {
  manifest: ContextManifest | null;
  tone: "popover" | "panel";
}) {
  const id = modelId(manifest);
  const note = modelSwitchNote(manifest);
  return (
    <>
      {tone === "panel" ? (
        <div className="settings__row">
          <span className="settings__k">model</span>
          <span className="settings__v">{id}</span>
        </div>
      ) : null}
      <p className="settings__note" role="note" aria-label={`Model ${id}. ${note}`}>
        {note}
      </p>
    </>
  );
}
