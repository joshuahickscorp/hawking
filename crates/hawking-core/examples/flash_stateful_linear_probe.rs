//! Bounded stateful Flash linear-organ probe.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash stateful probe requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
#[path = "flash_noetic_complete_layer0.rs"]
mod linear;

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;
    use std::path::PathBuf;

    let argv: Vec<String> = env::args().collect();
    let value = |flag: &str| argv.windows(2).find(|p| p[0] == flag).map(|p| p[1].clone());
    let root = PathBuf::from(value("--root").unwrap_or_else(|| {
        "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc".to_owned()
    }));
    let out = PathBuf::from(value("--out").unwrap_or_else(|| {
        "receipts/headless/FLASH_STATEFUL_LINEAR_ORGAN.json".to_owned()
    }));
    let token_ids = value("--tokens")
        .unwrap_or_else(|| "248044,248044".to_owned())
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    linear::run_stateful_token_probe(root, &token_ids, out)
}
