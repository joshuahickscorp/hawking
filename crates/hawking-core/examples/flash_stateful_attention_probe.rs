//! Bounded persistent-KV Flash full-attention organ probe.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash attention probe requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
#[path = "flash_full_attention_layer3.rs"]
mod attention;

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    let argv: Vec<String> = env::args().collect();
    let value = |flag: &str| argv.windows(2).find(|p| p[0] == flag).map(|p| p[1].clone());
    let root = PathBuf::from(value("--root").unwrap_or_else(|| {
        "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc".to_owned()
    }));
    let layer = value("--layer").unwrap_or_else(|| "3".to_owned()).parse::<usize>()?;
    let out = PathBuf::from(value("--out").unwrap_or_else(|| {
        "receipts/headless/FLASH_STATEFUL_ATTENTION_ORGAN.json".to_owned()
    }));
    let token_ids = value("--tokens").unwrap_or_else(|| "248044,248044".to_owned())
        .split(',').map(str::parse::<usize>).collect::<Result<Vec<_>, _>>()?;
    if let Some(union_from) = value("--union-from") {
        let doc: serde_json::Value = serde_json::from_slice(&fs::read(union_from)?)?;
        let mut route_ids = doc.get("steps").and_then(serde_json::Value::as_array)
            .into_iter().flatten()
            .flat_map(|step| step.get("route_ids").and_then(serde_json::Value::as_array)
                .into_iter().flatten())
            .filter_map(serde_json::Value::as_u64)
            .map(|id| u32::try_from(id).map_err(|_| std::io::Error::other("route id exceeds u32")))
            .collect::<Result<Vec<_>, _>>()?;
        route_ids.sort_unstable();
        route_ids.dedup();
        attention::run_stateful_attention_probe_route_union(root, layer, &token_ids, route_ids, out)
    } else {
        attention::run_stateful_attention_probe(root, layer, &token_ids, out)
    }
}
