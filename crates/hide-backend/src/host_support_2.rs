use serde_json::json;

/// Attach capability + rot + meter to a compiled manifest so the durable
/// `context.compiled` event and any projection carry auditable numbers.
///
/// `tokens_estimated` is `true` when packing used the `chars/4` heuristic rather
/// than a real tokenizer — the meter must never claim tokenizer-true counts then.
pub(crate) fn seal_compiled_manifest(
    manifest: &mut hawking_context::ContextManifest,
    capability: hawking_context::ContextCapability,
    live: Option<&hawking_context::ManifestLive>,
    tokens_estimated: bool,
) {
    use hawking_context::{detect_context_rot, ContextMeter, RotThresholds};
    let occupancy = live.map(|l| l.occupancy);
    let watermark = live.map(|l| l.watermark);
    let fidelity = live.and_then(|l| l.recall_fidelity);
    let rot = detect_context_rot(
        manifest,
        occupancy,
        watermark,
        fidelity,
        RotThresholds::default(),
    );
    let meter = ContextMeter::from_parts(
        &capability,
        manifest.used_tokens,
        tokens_estimated,
        live,
        Some(&rot),
    );
    manifest.capability = Some(capability);
    manifest.rot = Some(rot);
    manifest.meter = Some(meter);
}

/// JSON payload for the durable `context.compiled` marker: compile stats plus
/// the honest capability / rot / meter picture (so a later audit never has to
/// re-infer whether a number was measured).
pub(crate) fn context_compiled_payload(
    manifest: &hawking_context::ContextManifest,
    out_budget: Option<usize>,
    path: &str,
    run_id: Option<&str>,
) -> serde_json::Value {
    let mut body = json!({
        "used_tokens": manifest.used_tokens,
        "retained": manifest.retained.len(),
        "dropped": manifest.dropped.len(),
        "path": path,
        "capability": manifest.capability,
        "rot": manifest.rot,
        "meter": manifest.meter,
        // Hard rule, restated on every durable record.
        "native_is_not_usable": true,
    });
    if let Some(b) = out_budget {
        body["budget"] = json!(b);
    }
    if let Some(id) = run_id {
        body["run_id"] = json!(id);
    }
    body
}
