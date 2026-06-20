
#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use strand_quant::codebook::codebook_lut;
use strand_quant::decode::{decode_tensor_fixed, decode_tensor_fixed_with_lut};
use strand_quant::encode::{
    encode_tensor_with_lut, f32_metric_from_env, f32_search_from_env, vector_lut_from_scalar,
    EncodedTensor,
};
use strand_quant::encode_cache::{self, Manifest};
use strand_quant::format::{self, PackedTensor, PackedTensorV2};
use strand_quant::outlier_wire::{append_outl, append_outl_c2f, OutlierWire};
use strand_quant::provenance_io::append_sprv_computed;
use strand_quant::sideinfo_wire::append_sdsq;
use strand_quant::safetensor_io::{extract_str_field, SafeTensors};
use strand_quant::sha256;
use strand_quant::learned_codebook::train_state_vector_lut;
use strand_quant::rht::{
    rht_forward_cols, rht_forward_rows, rht_inverse_cols_inplace, rht_inverse_rows_inplace,
    RhtConfig,
};
use strand_quant::gate_utils::{is_quantizable_linear, rht_seed_for};
use strand_quant::{encode_tensor_with, EncodeOpts, TrellisConfig};

struct MpRule {
    pattern: String,
    bits: u32,
}

fn parse_rung_config(path: &str) -> Vec<MpRule> {
    let s = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read rung-config '{}': {}", path, e));
    
    let mut rules = Vec::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        
        if b[i] == b'"' {
            let key_start = i + 1;
            i += 1;
            while i < b.len() && b[i] != b'"' {
                i += 1;
            }
            let key = s[key_start..i].to_string();
            i += 1; 
            
            while i < b.len() && (b[i].is_ascii_whitespace() || b[i] == b':') {
                i += 1;
            }
            
            if i < b.len() && b[i].is_ascii_digit() {
                let num_start = i;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                if let Ok(bits) = s[num_start..i].parse::<u32>() {
                    if !key.is_empty() {
                        rules.push(MpRule { pattern: key, bits });
                    }
                }
            }
        } else {
            i += 1;
        }
    }
    assert!(!rules.is_empty(), "rung-config '{}' parsed 0 rules — check JSON syntax", path);
    rules
}

fn parse_mp_config(path: &str) -> Vec<MpRule> {
    let s = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read mp-config '{}': {}", path, e));

    fn extract_str(obj: &str, field: &str) -> Option<String> {
        let pat = format!("\"{}\"", field);
        let p = obj.find(&pat)? + pat.len();
        let rest = &obj[p..];
        let q1 = rest.find('"')?;
        let after = &rest[q1 + 1..];
        let q2 = after.find('"')?;
        Some(after[..q2].to_string())
    }
    fn extract_uint(obj: &str, field: &str) -> Option<u32> {
        let pat = format!("\"{}\"", field);
        let p = obj.find(&pat)? + pat.len();
        let rest = &obj[p..];
        let rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ':');
        rest.split(|c: char| !c.is_ascii_digit())
            .next()
            .and_then(|t| t.parse().ok())
    }

    let b = s.as_bytes();
    let mut rules = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'{' {
            let start = i;
            let mut depth = 0usize;
            while i < b.len() {
                match b[i] {
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            i += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            let obj = &s[start..i];
            if let (Some(pattern), Some(bits)) =
                (extract_str(obj, "pattern"), extract_uint(obj, "bits"))
            {
                rules.push(MpRule { pattern, bits });
            }
        } else {
            i += 1;
        }
    }
    assert!(!rules.is_empty(), "mp-config '{}' parsed 0 rules — check JSON syntax", path);
    rules
}

fn mp_bits(rules: &[MpRule], name: &str, default_bits: u32) -> u32 {
    for r in rules {
        if r.pattern == "*" || name.contains(r.pattern.as_str()) {
            return r.bits;
        }
    }
    default_bits
}

fn extract_f64_field(obj: &str, field: &str) -> Option<f64> {
    let pat = format!("\"{field}\"");
    let p = obj.find(&pat)? + pat.len();
    let rest = obj[p..].trim_start_matches(|c: char| c.is_whitespace() || c == ':');
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn load_prior_sidecar(path: &str) -> HashMap<String, (f64, f64)> {
    let Ok(s) = fs::read_to_string(path) else { return HashMap::new() };
    let b = s.as_bytes();
    let mut map = HashMap::new();
    let mut i = 1usize; 
    while i < b.len() {
        if b[i] == b'{' {
            let start = i;
            let mut depth = 0usize;
            while i < b.len() {
                match b[i] {
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            i += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            let obj = &s[start..i];
            if let (Some(name), Some(bpw), Some(rel)) = (
                extract_str_field(obj, "name"),
                extract_f64_field(obj, "bpw"),
                extract_f64_field(obj, "rel_rms_pct"),
            ) {
                map.insert(name, (bpw, rel));
            }
        } else {
            i += 1;
        }
    }
    map
}

struct OutTensor {
    name: String,
    shape: Vec<u64>,
    data: Vec<f32>,
}

fn write_safetensors(path: &str, tensors: &[OutTensor]) -> std::io::Result<()> {
    let mut header = String::from("{");
    let mut offset = 0usize;
    for (i, t) in tensors.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let nbytes = t.data.len() * 4;
        let shape = t
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            t.name,
            shape,
            offset,
            offset + nbytes
        ));
        offset += nbytes;
    }
    header.push('}');
    while (8 + header.len()) % 8 != 0 {
        header.push(' ');
    }

    let mut f = std::io::BufWriter::new(fs::File::create(path)?);
    f.write_all(&(header.len() as u64).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for t in tensors {
        let mut buf = Vec::with_capacity(t.data.len() * 4);
        for &v in &t.data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&buf)?;
    }
    f.flush()?;
    Ok(())
}

struct Args {
    input: String,
    output: String,
    bits: u32,
    l: u32,
    tail_biting: bool,
    affine_mode: AffineMode,
    rht: bool,
    // --rht-cols: opt-in. Use per-COLUMN-sign RHT (rht_forward_cols / rht_inverse_cols)
    // instead of the default per-row-sign RHT. The encode still applies a per-row FWHT, but
    // the random sign is shared across output rows (sign_at(seed, col)). This is the
    // serveability unlock: at inference y = decode(W_col_rht)·T(x) where the activation
    // transform T(x) is computed ONCE per tensor and reused for every output row, vs the
    // per-row variant that needs out_features distinct activation transforms per token.
    // It is a DIFFERENT quantization grid (sign pattern differs), so it must clear the
    // quality gate; the recon weight it writes is a standard dense weight (RHT is fully
    // inverted on the encode side), so the loaded model and eval-gate need no RHT awareness.
    // Default OFF; ignored when --no-rht. Mutually independent of --vec-dim.
    rht_cols: bool,
    threads: usize,
    
    only: Option<String>,
    
    measure_only: bool,
    
    quality: bool,
    
    packed_out: Option<String>,
    
    packed_v2_out: Option<String>,
    
    strict_v2: bool,
    
    mp_config: Option<Vec<MpRule>>,
    
    rung_config: Option<Vec<MpRule>>,
    
    dump_indices: Option<String>,
    
    vec_dim: u32,
    
    learned_codebook: bool,
    
    no_sprv: bool,
    
    outlier_pct: f64,
    outlier_bits: u32,

    // --sdsq-sideinfo: opt-in. After writing the v2 archive (and OUTL, before SPRV),
    // entropy-code the per-block scale_q stream via sideinfo_rans into an EOF-chained
    // SDSQ section (sprint Lever 1). Decode overwrites BlockMeta.scale_q losslessly —
    // byte-identical reconstruct — so it is zero quality cost. Default OFF; when OFF
    // the output bytes are byte-identical to today.
    sdsq_sideinfo: bool,

    // --c2f-outl: opt-in. Gap-code the per-tensor outlier POSITIONS with the C2F
    // entropy coder (OUTL_FLAG_POS_RANS) instead of bit-packing them inline. The
    // largest measured side-info channel (~0.148 bpw on 0.5B q2). Container-only:
    // read_outl_bytes reconstructs byte-identical OutlierWire.entries, so decode /
    // MAC / SPRV are unchanged. Default OFF; requires --outlier-channel > 0.
    c2f_outl: bool,

    skip_manifest: Option<String>,
    
    reuse_from: Option<String>,

    // --actmean <calib.json>: per-module activation feature means from
    // scripts/calib-actmean.py. When set, the encoder computes the output de-bias
    // correction c = -(recon - orig) @ mu per tensor row (the de-bias gate that
    // measured -28.7% PPL on dp_d4_r2 for ~0.014 bpw) and persists it to
    // <output>.debias.json. Deploy applies c in the MAC epilogue; the base model
    // is never needed after encode.
    actmean: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AffineMode {
    Auto,
    On,
    Off,
}

fn parse_args() -> Args {
    let mut input = String::new();
    let mut output = String::new();
    let mut bits = 3u32;
    let mut l = 0u32;
    let mut tail_biting = false;
    let mut affine_mode = AffineMode::Auto;
    let mut rht = true;
    let mut rht_cols = false;
    let mut threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut only = None;
    let mut measure_only = false;
    let mut quality = false;
    let mut l_explicit = false;
    let mut packed_out = None;
    let mut packed_v2_out = None;
    let mut strict_v2 = true;
    let mut mp_config: Option<Vec<MpRule>> = None;
    let mut rung_config: Option<Vec<MpRule>> = None;
    let mut dump_indices: Option<String> = None;
    let mut vec_dim = 1u32;
    let mut learned_codebook = false;
    let mut no_sprv = false;
    let mut outlier_pct = 0.0f64;
    let mut outlier_bits = 8u32;
    let mut sdsq_sideinfo = false;
    let mut c2f_outl = false;
    let mut skip_manifest: Option<String> = None;
    let mut reuse_from: Option<String> = None;
    let mut actmean: Option<String> = None;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--in" | "--input" => input = it.next().expect("--in/--input needs a path"),
            "--out" | "--output" => output = it.next().expect("--out/--output needs a path"),
            "--bits" => bits = it.next().expect("--bits").parse().expect("bits int"),
            "--l" => {
                l = it.next().expect("--l").parse().expect("l int");
                l_explicit = true;
            }
            "--tail-biting" => tail_biting = true,
            "--no-tail-biting" => tail_biting = false,
            "--affine-min" => {
                affine_mode = match it.next().as_deref() {
                    Some("on") => AffineMode::On,
                    Some("off") => AffineMode::Off,
                    Some("auto") | None => AffineMode::Auto,
                    Some(x) => panic!("--affine-min expects on|off|auto, got {x}"),
                }
            }
            "--no-rht" => rht = false,
            "--rht-cols" => rht_cols = true,
            "--rht-rows" => rht_cols = false,
            "--threads" => threads = it.next().expect("--threads").parse().expect("threads int"),
            "--only" => only = Some(it.next().expect("--only needs a substring")),
            "--measure-only" => measure_only = true,
            "--quality" => quality = true,
            "--packed-out" => packed_out = Some(it.next().expect("--packed-out needs a path")),
            "--packed-v2-out" => {
                packed_v2_out = Some(it.next().expect("--packed-v2-out needs a path"))
            }
            "--ragged-v2" | "--no-strict-v2" => strict_v2 = false,
            "--strict-v2" => strict_v2 = true,
            "--dump-indices" => dump_indices = Some(it.next().expect("--dump-indices needs a path")),
            "--actmean" => actmean = Some(it.next().expect("--actmean needs a calib json path")),
            "--vec-dim" => {
                vec_dim = it.next().expect("--vec-dim needs N").parse().expect("vec-dim int")
            }
            "--learned-codebook" => learned_codebook = true,
            "--no-sprv" => no_sprv = true,
            "--outlier-channel" => {
                outlier_pct = it.next().expect("--outlier-channel needs PCT").parse().expect("outlier-channel f64")
            }
            "--outlier-bits" => {
                outlier_bits = it.next().expect("--outlier-bits needs N").parse().expect("outlier-bits int")
            }
            "--sdsq-sideinfo" => sdsq_sideinfo = true,
            "--c2f-outl" => c2f_outl = true,
            "--skip-manifest" => {
                skip_manifest = Some(it.next().expect("--skip-manifest needs a path"))
            }
            "--reuse-from" => {
                reuse_from = Some(it.next().expect("--reuse-from needs a prior recon path"))
            }
            "--mp-config" => {
                let path = it.next().expect("--mp-config needs a path");
                let rules = parse_mp_config(&path);
                eprintln!("[quantize-model] mp-config '{}': {} rules loaded", path, rules.len());
                mp_config = Some(rules);
            }
            "--rung-config" => {
                let path = it.next().expect("--rung-config needs a path");
                let rules = parse_rung_config(&path);
                eprintln!("[quantize-model] rung-config '{}': {} rules loaded", path, rules.len());
                rung_config = Some(rules);
            }
            "-h" | "--help" => {
                eprintln!(
                    "quantize-model --input/--in <safetensors> --output/--out <safetensors> \
                     --bits <1|2|3|4|5|6> [--l <4..14>] \
                     [--tail-biting] [--affine-min auto|on|off] [--no-rht] [--rht-cols] [--threads N] \
                     [--only <substr>] [--measure-only] [--quality]\n\
                     [--mp-config <json>] [--rung-config <json>] [--dump-indices <path>] \
                     [--vec-dim N] [--learned-codebook]\n\
                     [--outlier-channel PCT] [--outlier-bits N]\n\
                     [--skip-manifest <manifest.json> [--reuse-from <prior-recon.safetensors>]]\n\
                     [--packed-out <strand>] [--packed-v2-out <strand> [--ragged-v2]]\n\
                     \n\
                     Default: L=k+4 (fast mode). Metal is auto-selected on macOS when available.\n\
                     --quality:    use L=k+6 (4× slower on CPU; ~1-2× on Metal; lower PPL).\n\
                     --mp-config:  JSON array of {{pattern, bits}} rules for per-layer bit depths.\n\
                     \t      E.g. scripts/mp-4a3f.json (4-bit attn, 3-bit FFN).\n\
                     \t      --bits sets the fallback for tensors matching no rule.\n\
                     --rung-config: flat JSON dict of substring-pattern → bits overrides.\n\
                     \t      E.g. {{\"q_proj\": 4, \"k_proj\": 4, \"down_proj\": 3}}.\n\
                     \t      Lower priority than --mp-config (mp-config rules checked first);\n\
                     \t      --bits is the global fallback after both.\n\
                     --dump-indices <path>: write the trellis input-symbol stream per\n\
                     \t      processed tensor to <path>.<name>.bin (entropy-headroom\n\
                     \t      probe; works with --measure-only). See scripts/entropy-probe.py.\n\
                     --vec-dim N:  d-dim vector trellis (LEVER B1; default 1 = scalar).\n\
                     \t      Each step emits d weights ⇒ payload k/d bpw. Forces the\n\
                     \t      CPU explicit-LUT vector encode/decode pair (1<=N<=8).\n\
                     --learned-codebook: with --vec-dim>1, learn the [2^L*d] LUT from\n\
                     \t      each tensor's own post-RHT weights (deterministic, seeded)\n\
                     \t      instead of broadcasting the frozen Gaussian table.\n\
                     --outlier-channel PCT: remove the top PCT% weights by |w| pre-RHT,\n\
                     \t      store them as a sparse integer channel at --outlier-bits\n\
                     \t      (default 8), and patch them back at recon.\n\
                     --sdsq-sideinfo: (v2 only, opt-in) entropy-code the per-block\n\
                     \t      scale_q stream into an EOF-chained SDSQ section (rANS;\n\
                     \t      sprint Lever 1). Appended after OUTL, before SPRV. Decode\n\
                     \t      overwrites scale_q losslessly (byte-identical reconstruct);\n\
                     \t      ~0.084 bpw ship-density recovery, seek table untouched.\n\
                     \t      OFF = output bytes byte-identical to today.\n\
                     --c2f-outl: (v2 only, opt-in; needs --outlier-channel) gap-code\n\
                     \t      outlier POSITIONS with the C2F entropy coder in OUTL\n\
                     \t      (OUTL_FLAG_POS_RANS) instead of inline idx_bits. The biggest\n\
                     \t      side-info channel (~0.148 bpw on 0.5B q2). Container-only:\n\
                     \t      OutlierWire reconstruct is byte-identical, decode unchanged.\n\
                     --skip-manifest <manifest.json>: per-tensor input-hash manifest\n\
                     \t      (identity skip for selective-PV requant). With --reuse-from,\n\
                     \t      tensors whose input hash + bits + invocation config (incl.\n\
                     \t      STRAND_F32_METRIC/STRAND_F32_SEARCH) match the prior run are\n\
                     \t      copied from the prior recon instead of re-encoded. The\n\
                     \t      manifest is rewritten for this run. Incompatible with the\n\
                     \t      packed outputs. Default OFF.\n\
                     --packed-out <path>:    write the compact .strand v1 (STRQ) archive\n\
                     \t      (reference/round-trip format) and stop.\n\
                     --packed-v2-out <path>: write the .strand v2 (STR2) DEPLOY archive\n\
                     \t      (per-tensor page-aligned block-offset table + source_sha256;\n\
                     \t      the format the Metal GEMV kernel mmap-seeks) and stop.\n\
                     \t      With --outlier-channel the OUTL section (the integer\n\
                     \t      (idx,val) side-channel) is appended automatically, and the\n\
                     \t      SPRV self-verifying provenance trailer is appended by\n\
                     \t      default (skip with --no-sprv).\n\
                     --ragged-v2 / --no-strict-v2: allow v2 tensors whose in_features is\n\
                     \t      not a multiple of block_len (clears the ALL_STRICT flag).\n\
                     \t      Default is STRICT: fail fast naming the first violator."
                );
                std::process::exit(0);
            }
            other => panic!("unknown arg {other}"),
        }
    }
    assert!(!input.is_empty(), "--in is required");
    assert!(measure_only || !output.is_empty() || packed_out.is_some() || packed_v2_out.is_some(), "--out is required (unless --measure-only, --packed-out, or --packed-v2-out)");
    assert!((1..=6).contains(&bits), "--bits must be 1, 2, 3, 4, 5, or 6 (used as mp-config fallback when --mp-config is set)");
    // DETERMINISM GUARD: the v2 archive now records the col-RHT mode (flag bit 3) and the
    // decoder (strand-decode-kernel::outlier_mac) serves it, so --rht-cols + --packed-v2-out
    // is supported. The v1 .strand writer has no col-RHT flag, so --packed-out stays blocked.
    assert!(
        !(rht_cols && packed_out.is_some()),
        "--rht-cols cannot be combined with --packed-out: the v1 .strand format has no col-RHT \
         flag, so the decoder would invert with row-sign RHT and decode incorrectly. Use \
         --packed-v2-out (v2 records the col-RHT mode) or --out (dense safetensors recon)."
    );
    if quality && !l_explicit {
        l = bits + 6;
    }
    Args {
        input,
        output,
        bits,
        l,
        tail_biting,
        affine_mode,
        rht,
        rht_cols,
        threads,
        only,
        measure_only,
        quality,
        packed_out,
        packed_v2_out,
        strict_v2,
        mp_config,
        rung_config,
        dump_indices,
        vec_dim,
        learned_codebook,
        no_sprv,
        outlier_pct,
        outlier_bits,
        sdsq_sideinfo,
        c2f_outl,
        skip_manifest,
        reuse_from,
        actmean,
    }
}

fn cache_config_key(args: &Args, f32_metric: bool, f32_search: bool) -> String {
    format!(
        "v1;l={};quality={};tail_biting={};affine={};rht={};vec_dim={};learned={};out={}@{};f32_metric={};f32_search={}",
        args.l,
        args.quality,
        args.tail_biting,
        match args.affine_mode {
            AffineMode::Auto => "auto",
            AffineMode::On => "on",
            AffineMode::Off => "off",
        },
        args.rht,
        args.vec_dim,
        args.learned_codebook,
        args.outlier_pct,
        args.outlier_bits,
        f32_metric,
        f32_search,
    )
}

fn sanitize_for_filename(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') { c } else { '_' })
        .collect()
}

fn dump_index_stream(prefix: &str, name: &str, syms: &[u8], k: u32, l: u32) -> std::io::Result<String> {
    let path = format!("{}.{}.bin", prefix, sanitize_for_filename(name));
    let mut buf: Vec<u8> = Vec::with_capacity(24 + syms.len());
    buf.extend_from_slice(b"STRX");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&k.to_le_bytes());
    buf.extend_from_slice(&l.to_le_bytes());
    buf.extend_from_slice(&(syms.len() as u64).to_le_bytes());
    buf.extend_from_slice(syms);
    fs::write(&path, &buf)?;
    Ok(path)
}

/// Parse scripts/calib-actmean.py output: tensor name -> activation feature-mean
/// vector mu (one f32 per in_feature). Hand-rolled like the other sidecar parsers:
/// scan for `"tensor": "<name>"`, then the next `"feature_mean": [ ... ]` array.
fn load_actmean(path: &str) -> HashMap<String, Vec<f32>> {
    let s = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read actmean '{}': {}", path, e));
    let mut map = HashMap::new();
    let mut pos = 0usize;
    while let Some(t) = s[pos..].find("\"tensor\"") {
        let tstart = pos + t + "\"tensor\"".len();
        let rest = &s[tstart..];
        let Some(q1) = rest.find('"') else { break };
        let after = &rest[q1 + 1..];
        let Some(q2) = after.find('"') else { break };
        let name = after[..q2].to_string();
        pos = tstart + q1 + 1 + q2;
        // feature_mean must appear before the next "tensor" key (same module object)
        let scope_end = s[pos..].find("\"tensor\"").map(|n| pos + n).unwrap_or(s.len());
        if let Some(f) = s[pos..scope_end].find("\"feature_mean\"") {
            let fstart = pos + f;
            if let Some(br) = s[fstart..scope_end].find('[') {
                let astart = fstart + br + 1;
                if let Some(close) = s[astart..].find(']') {
                    let mu: Vec<f32> = s[astart..astart + close]
                        .split(',')
                        .filter_map(|t| t.trim().parse::<f32>().ok())
                        .collect();
                    if !mu.is_empty() {
                        map.insert(name, mu);
                    }
                    pos = astart + close;
                }
            }
        }
    }
    assert!(
        !map.is_empty(),
        "actmean '{}' parsed 0 feature_mean vectors — was calib-actmean.py run with feature means enabled?",
        path
    );
    map
}

/// De-bias correction (frontier doc §4.C/§8.3): c_i = -sum_j (recon_ij - orig_ij) * mu_j.
/// Computed at encode time where orig+recon+mu coexist; persisted with the artifact
/// so deploy never needs the base weights. Applied as a per-output-row bias.
fn debias_correction(orig: &[f32], recon: &[f32], mu: &[f32], in_features: usize) -> Vec<f32> {
    let rows = recon.len() / in_features;
    let mut c = vec![0f32; rows];
    for r in 0..rows {
        let o = &orig[r * in_features..(r + 1) * in_features];
        let q = &recon[r * in_features..(r + 1) * in_features];
        let mut acc = 0f64;
        for j in 0..in_features {
            acc += f64::from(q[j] - o[j]) * f64::from(mu[j]);
        }
        c[r] = (-acc) as f32;
    }
    c
}

struct TensorJob {
    name: String,
    shape: Vec<u64>,
    gt: Vec<f32>,
    in_features: usize,
    bits: u32,
}

struct TensorResult {
    name: String,
    shape: Vec<u64>,
    recon: Vec<f32>,
    bpw: f64,
    rel_rms_pct: f64,
    bits: u32,
    enc: Option<EncodedTensor>,
    l_bits: u8,
    k_bits: u8,
    rht_seed: u64,
    block_len: u32,
    outlier: Option<OutlierWire>,
}

#[allow(clippy::too_many_arguments)]
fn quantize_one(
    job: &TensorJob,
    cfg: &TrellisConfig,
    use_rht: bool,
    rht_cols: bool,
    tail_biting: bool,
    affine_min: bool,
    dump_indices: Option<&str>,
    learned_codebook: bool,
    outlier_pct: f64,
    outlier_bits: u32,
    want_packed: bool,
) -> TensorResult {
    
    let outliers: Option<(Vec<usize>, Vec<f32>, Vec<i32>, f32)> = if outlier_pct > 0.0 {
        let n = job.gt.len();
        let k = ((outlier_pct / 100.0) * n as f64).round() as usize;
        if k == 0 || n == 0 {
            None
        } else {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by(|&a, &b| {
                job.gt[b].abs().partial_cmp(&job.gt[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
            });
            let idx: Vec<usize> = order[..k].to_vec();
            let omax = idx.iter().fold(0f32, |m, &i| m.max(job.gt[i].abs())).max(1e-12);
            let ob = outlier_bits.clamp(2, 16);
            let levels = ((1i64 << (ob - 1)) - 1) as f32;
            let vals: Vec<f32> = idx
                .iter()
                .map(|&i| (job.gt[i] / omax * levels).round() / levels * omax)
                .collect();
            let codes: Vec<i32> = idx
                .iter()
                .map(|&i| (job.gt[i] / omax * levels).round() as i32)
                .collect();
            Some((idx, vals, codes, omax))
        }
    } else {
        None
    };
    let bulk_gt: Vec<f32>;
    let job_gt: &[f32] = if let Some((idx, ..)) = &outliers {
        let mut b = job.gt.clone();
        for &i in idx {
            b[i] = 0.0;
        }
        bulk_gt = b;
        &bulk_gt
    } else {
        &job.gt
    };
    
    let (work, rcfg) = if use_rht {
        let rcfg = RhtConfig::from_seed(rht_seed_for(&job.name));
        let fwd = if rht_cols {
            rht_forward_cols(job_gt, &rcfg, job.in_features)
        } else {
            rht_forward_rows(job_gt, &rcfg, job.in_features)
        };
        (fwd, Some(rcfg))
    } else {
        (job_gt.to_vec(), None)
    };

    let opts = EncodeOpts { adaptive: true, tail_biting, affine_min, ..EncodeOpts::default() };
    
    let d = cfg.vec_dim();
    let custom_lut: Option<Vec<i32>> = if d > 1 {
        if learned_codebook {
            let seed = rht_seed_for(&job.name);
            let learned = train_state_vector_lut(&work, cfg.l_bits, d, seed, 50);
            
            let scalar: Vec<i32> = codebook_lut(cfg.l_bits).to_vec();
            let broadcast = vector_lut_from_scalar(&scalar, d);
            let se_of = |lut: &[i32]| -> f64 {
                let e = encode_tensor_with_lut(&work, cfg, &opts, lut);
                let q12 = decode_tensor_fixed_with_lut(&e, cfg, lut);
                let mut r: Vec<f32> = q12.iter().map(|&q| (q as f32) * (1.0 / 4096.0)).collect();
                if let Some(rc) = rcfg.as_ref() {
                    if rht_cols {
                        rht_inverse_cols_inplace(&mut r, rc, job.in_features);
                    } else {
                        rht_inverse_rows_inplace(&mut r, rc, job.in_features);
                    }
                }
                job_gt.iter().zip(&r).map(|(&g, &a)| { let e = g as f64 - a as f64; e * e }).sum::<f64>()
            };
            if se_of(&learned) <= se_of(&broadcast) {
                Some(learned)
            } else {
                eprintln!("[quantize-model] {}: learned codebook worse than scalar — kept scalar (guard)", job.name);
                Some(broadcast)
            }
        } else {
            let scalar: Vec<i32> = codebook_lut(cfg.l_bits).to_vec();
            Some(vector_lut_from_scalar(&scalar, d))
        }
    } else {
        None
    };
    let mut enc = match &custom_lut {
        None => encode_tensor_with(&work, cfg, &opts),
        Some(lut) => encode_tensor_with_lut(&work, cfg, &opts, lut),
    };
    enc.has_rht_seed = use_rht;

    if let Some(prefix) = dump_indices {
        let syms = enc.index_symbols(cfg);
        match dump_index_stream(prefix, &job.name, &syms, cfg.k_bits, cfg.l_bits) {
            Ok(path) => eprintln!(
                "[quantize-model] dumped {} index symbols (k={}, L={}) -> {}",
                syms.len(), cfg.k_bits, cfg.l_bits, path
            ),
            Err(e) => eprintln!(
                "[quantize-model] WARNING: --dump-indices write failed for '{}': {}",
                job.name, e
            ),
        }
    }

    let q12 = match &custom_lut {
        None => decode_tensor_fixed(&enc, cfg),
        Some(lut) => decode_tensor_fixed_with_lut(&enc, cfg, lut),
    };
    let mut recon: Vec<f32> = q12.iter().map(|&q| (q as f32) * (1.0 / 4096.0)).collect();
    if let Some(rcfg) = rcfg {
        if rht_cols {
            rht_inverse_cols_inplace(&mut recon, &rcfg, job.in_features);
        } else {
            rht_inverse_rows_inplace(&mut recon, &rcfg, job.in_features);
        }
    }
    let mut eff_bpw = enc.total_bpw(cfg);
    if let Some((idx, vals, ..)) = &outliers {
        for (&i, &v) in idx.iter().zip(vals.iter()) {
            recon[i] = v;
        }
        let n = recon.len().max(1);
        let f = idx.len() as f64 / n as f64;
        let idx_bits = (n as f64).log2().ceil();
        eff_bpw += f * (idx_bits + outlier_bits as f64);
    }

    let mut se = 0.0f64;
    let mut pw = 0.0f64;
    for (&r, &a) in job.gt.iter().zip(&recon) {
        let d = r as f64 - a as f64;
        se += d * d;
        pw += (r as f64) * (r as f64);
    }
    let rel_rms_pct = if pw > 0.0 { (se / pw).sqrt() * 100.0 } else { 0.0 };

    let packed_enc = if want_packed && cfg.vec_dim() == 1 {
        Some(enc)
    } else {
        None
    };
    let outlier_wire = if packed_enc.is_some() {
        outliers.as_ref().map(|(idx, _vals, codes, omax)| {
            OutlierWire::from_selection(
                job.gt.len(),
                idx.clone(),
                codes.clone(),
                *omax,
                outlier_bits.clamp(2, 16),
            )
        })
    } else {
        None
    };
    TensorResult {
        name: job.name.clone(),
        shape: job.shape.clone(),
        recon,
        bpw: eff_bpw,
        rel_rms_pct,
        bits: job.bits,
        enc: packed_enc,
        l_bits: cfg.l_bits as u8,
        k_bits: cfg.k_bits as u8,
        rht_seed: if use_rht { rht_seed_for(&job.name) } else { 0 },
        block_len: cfg.block_len as u32,
        outlier: outlier_wire,
    }
}

fn main() {
    let args = parse_args();
    let t0 = Instant::now();

    let st = SafeTensors::open(&args.input).expect("open input safetensors");

    let resolve_cfg = |bits: u32| -> (TrellisConfig, bool, bool) {
        let cfg = if args.l != 0 {
            TrellisConfig::for_bpw_l(bits as f64, args.l)
        } else if args.quality {
            TrellisConfig::for_bpw_quality(bits as f64)
        } else {
            TrellisConfig::for_bpw(bits as f64)
        };
        let cfg = cfg.with_vec_dim(args.vec_dim);
        let affine_min = match args.affine_mode {
            AffineMode::On => true,
            AffineMode::Off => false,
            AffineMode::Auto => bits >= 4,
        };
        let tail_biting = args.tail_biting || affine_min;
        (cfg, affine_min, tail_biting)
    };

    if args.mp_config.is_some() || args.rung_config.is_some() {
        let mp_n = args.mp_config.as_ref().map(|r| r.len()).unwrap_or(0);
        let rc_n = args.rung_config.as_ref().map(|r| r.len()).unwrap_or(0);
        eprintln!(
            "[quantize-model] in={} MIXED-PRECISION (mp-config:{} rules, rung-config:{} rules) rht={} threads={} quality={}",
            args.input, mp_n, rc_n, args.rht, args.threads, args.quality,
        );
        if let Some(rules) = &args.mp_config {
            for r in rules {
                let (cfg, affine_min, tail_biting) = resolve_cfg(r.bits);
                eprintln!(
                    "[quantize-model]   mp-config  pattern={:<20} bits={} L={} affine_min={} tail_biting={}",
                    r.pattern, r.bits, cfg.l_bits, affine_min, tail_biting
                );
            }
        }
        if let Some(rules) = &args.rung_config {
            for r in rules {
                let (cfg, affine_min, tail_biting) = resolve_cfg(r.bits);
                eprintln!(
                    "[quantize-model]   rung-config pattern={:<20} bits={} L={} affine_min={} tail_biting={}",
                    r.pattern, r.bits, cfg.l_bits, affine_min, tail_biting
                );
            }
        }
        let (cfg, affine_min, tail_biting) = resolve_cfg(args.bits);
        eprintln!(
            "[quantize-model]   fallback (no match)   bits={} L={} affine_min={} tail_biting={}",
            args.bits, cfg.l_bits, affine_min, tail_biting
        );
    } else {
        let (cfg, affine_min, tail_biting) = resolve_cfg(args.bits);
        eprintln!(
            "[quantize-model] in={} bits={} L={} k={} rht={} tail_biting={} affine_min={} threads={} quality={} vec_dim={} learned_codebook={}",
            args.input, args.bits, cfg.l_bits, cfg.k_bits, args.rht,
            tail_biting, affine_min, args.threads, args.quality,
            cfg.vec_dim(), args.learned_codebook,
        );
    }

    let mut jobs: Vec<TensorJob> = Vec::new();
    let mut passthrough: Vec<OutTensor> = Vec::new();
    let only_match =
        |name: &str| args.only.as_ref().map(|s| name.contains(s.as_str())).unwrap_or(true);
    for name in &st.order {
        let t = &st.tensors[name];
        if is_quantizable_linear(name, &t.shape) && only_match(name) {
            let in_features = *t.shape.last().unwrap() as usize;
            
            let rung_fallback = match &args.rung_config {
                Some(rules) => mp_bits(rules, name, args.bits),
                None => args.bits,
            };
            let tensor_bits = match &args.mp_config {
                Some(rules) => mp_bits(rules, name, rung_fallback),
                None => rung_fallback,
            };
            jobs.push(TensorJob {
                name: name.clone(),
                shape: t.shape.clone(),
                gt: st.to_f32(t),
                in_features,
                bits: tensor_bits,
            });
        } else if !args.measure_only {
            passthrough.push(OutTensor {
                name: name.clone(),
                shape: t.shape.clone(),
                data: st.to_f32(t),
            });
        }
    }
    eprintln!(
        "[quantize-model] {} quantizable linear tensors, {} pass-through tensors",
        jobs.len(),
        passthrough.len()
    );

    let skip_key = cache_config_key(&args, f32_metric_from_env(), f32_search_from_env());
    if args.skip_manifest.is_some() {
        assert!(
            args.packed_out.is_none() && args.packed_v2_out.is_none(),
            "--skip-manifest cannot combine with --packed-out/--packed-v2-out \
             (reused tensors carry no in-memory EncodedTensor)"
        );
    } else {
        assert!(
            args.reuse_from.is_none(),
            "--reuse-from requires --skip-manifest (the manifest is the skip predicate)"
        );
    }
    let prior_manifest: Option<Manifest> = match (&args.skip_manifest, &args.reuse_from) {
        (Some(p), Some(_)) => Manifest::load(p).ok().flatten(),
        _ => None,
    };
    let prior_recon: Option<SafeTensors> = if prior_manifest.is_some() {
        args.reuse_from.as_ref().and_then(|p| SafeTensors::open(p).ok())
    } else {
        None
    };
    let prior_stats: HashMap<String, (f64, f64)> = if prior_recon.is_some() {
        args.reuse_from
            .as_ref()
            .map(|p| load_prior_sidecar(&format!("{p}.json")))
            .unwrap_or_default()
    } else {
        HashMap::new()
    };
    if args.skip_manifest.is_some() {
        eprintln!(
            "[quantize-model] identity-skip: key=\"{skip_key}\"; prior manifest {}",
            if prior_manifest.is_some() { "loaded" } else { "absent (cold: full encode)" }
        );
    }
    let new_manifest: Mutex<Manifest> = Mutex::new(Manifest::new(skip_key.clone()));
    let skipped_count = AtomicUsize::new(0);

    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<TensorResult>> = Mutex::new(Vec::new());
    let actmean_map: Option<HashMap<String, Vec<f32>>> = args.actmean.as_deref().map(|p| {
        let m = load_actmean(p);
        eprintln!("[quantize-model] actmean '{}': {} feature-mean vectors loaded (de-bias lane on)", p, m.len());
        m
    });
    let debias_out: Mutex<Vec<(String, Vec<f32>)>> = Mutex::new(Vec::new());
    let nthreads = args.threads.max(1);
    std::thread::scope(|scope| {
        for _ in 0..nthreads {
            scope.spawn(|| loop {
                let idx = next.fetch_add(1, Ordering::Relaxed);
                if idx >= jobs.len() {
                    break;
                }
                let job = &jobs[idx];
                let (tensor_cfg, tensor_affine_min, tensor_tail_biting) = resolve_cfg(job.bits);
                
                let mut reused: Option<TensorResult> = None;
                if args.skip_manifest.is_some() {
                    let h = encode_cache::hash_f32(&job.gt);
                    if encode_cache::should_skip(
                        &job.name, &h, job.bits, &skip_key, prior_manifest.as_ref(),
                    ) {
                        if let (Some(pr), Some(&(bpw, rel))) =
                            (prior_recon.as_ref(), prior_stats.get(job.name.as_str()))
                        {
                            if let Some(t) = pr.tensors.get(&job.name) {
                                reused = Some(TensorResult {
                                    name: job.name.clone(),
                                    shape: job.shape.clone(),
                                    recon: pr.to_f32(t),
                                    bpw,
                                    rel_rms_pct: rel,
                                    bits: job.bits,
                                    enc: None,
                                    l_bits: tensor_cfg.l_bits as u8,
                                    k_bits: tensor_cfg.k_bits as u8,
                                    rht_seed: if args.rht { rht_seed_for(&job.name) } else { 0 },
                                    block_len: tensor_cfg.block_len as u32,
                                    outlier: None,
                                });
                            }
                        }
                    }
                    
                    new_manifest.lock().unwrap().record(&job.name, &h, job.bits);
                }
                let r = match reused {
                    Some(r) => {
                        skipped_count.fetch_add(1, Ordering::Relaxed);
                        eprintln!(
                            "[skip {}/{}] {:<40} identity hit (prior recon reused, billed at prior bpw={:.3})",
                            idx + 1,
                            jobs.len(),
                            r.name,
                            r.bpw,
                        );
                        results.lock().unwrap().push(r);
                        continue;
                    }
                    None => quantize_one(
                        job,
                        &tensor_cfg,
                        args.rht,
                        args.rht_cols,
                        tensor_tail_biting,
                        tensor_affine_min,
                        args.dump_indices.as_deref(),
                        args.learned_codebook,
                        args.outlier_pct,
                        args.outlier_bits,
                        args.packed_out.is_some() || args.packed_v2_out.is_some(),
                    ),
                };
                eprintln!(
                    "[done {}/{}] {:<40} bits={} bpw={:.3} rel-RMS={:.2}%",
                    idx + 1,
                    jobs.len(),
                    r.name,
                    r.bits,
                    r.bpw,
                    r.rel_rms_pct
                );
                if let Some(map) = &actmean_map {
                    if let Some(mu) = map.get(&r.name) {
                        if mu.len() == job.in_features {
                            let c = debias_correction(&job.gt, &r.recon, mu, job.in_features);
                            debias_out.lock().unwrap().push((r.name.clone(), c));
                        } else {
                            eprintln!(
                                "[quantize-model] WARNING: actmean dim {} != in_features {} for '{}' — de-bias skipped",
                                mu.len(), job.in_features, r.name
                            );
                        }
                    }
                }
                results.lock().unwrap().push(r);
            });
        }
    });

    if let Some(p) = &args.actmean {
        let mut rows = debias_out.into_inner().unwrap();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        let bias_params: usize = rows.iter().map(|(_, c)| c.len()).sum();
        let mut js = String::from("{\n  \"schema\": \"strand_debias_v1\",\n");
        js.push_str(&format!("  \"actmean\": {:?},\n", p));
        js.push_str(&format!("  \"bias_rows\": {},\n  \"tensors\": {{\n", bias_params));
        for (i, (name, c)) in rows.iter().enumerate() {
            if i > 0 {
                js.push_str(",\n");
            }
            js.push_str(&format!("    {:?}: [", name));
            for (j, v) in c.iter().enumerate() {
                if j > 0 {
                    js.push(',');
                }
                js.push_str(&format!("{:.7e}", v));
            }
            js.push(']');
        }
        js.push_str("\n  }\n}\n");
        let dpath = format!("{}.debias.json", args.output);
        fs::write(&dpath, &js).expect("write debias sidecar");
        // billed mass: rows * 16 bits (bf16 deploy) over the quantized weights
        let nw: usize = results.lock().unwrap().iter().map(|r| r.recon.len()).sum();
        eprintln!(
            "[quantize-model] de-bias sidecar: {} tensors, {} bias rows ({:.4} bpw at bf16) -> {}",
            rows.len(),
            bias_params,
            if nw > 0 { bias_params as f64 * 16.0 / nw as f64 } else { 0.0 },
            dpath
        );
    }

    if let Some(p) = &args.skip_manifest {
        let m = new_manifest.into_inner().unwrap();
        match m.save(p) {
            Ok(()) => eprintln!(
                "[quantize-model] identity-skip: {} skipped / {} total; manifest -> {}",
                skipped_count.load(Ordering::Relaxed),
                jobs.len(),
                p
            ),
            Err(e) => eprintln!("[quantize-model] WARNING: identity manifest save failed: {e}"),
        }
    }

    let mut quant_results = results.into_inner().unwrap();
    let order_idx: HashMap<&str, usize> =
        st.order.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
    quant_results.sort_by_key(|r| order_idx[r.name.as_str()]);

    if let Some(packed_path) = &args.packed_out {
        let mut pts: Vec<PackedTensor> = Vec::new();
        let mut skipped = 0usize;
        for r in &quant_results {
            match &r.enc {
                Some(enc) => pts.push(PackedTensor {
                    name: &r.name,
                    shape: &r.shape,
                    rht_seed: r.rht_seed,
                    l_bits: r.l_bits,
                    k_bits: r.k_bits,
                    vec_dim: 1,
                    enc,
                }),
                None => skipped += 1,
            }
        }
        let bytes = format::write_strand(&pts);
        fs::write(packed_path, &bytes).expect("write .strand archive");
        let nw: usize = quant_results.iter().map(|r| r.recon.len()).sum();
        eprintln!(
            "[quantize-model] wrote {} tensors -> {} ({:.1} MB, {:.4} bytes/weight){}",
            pts.len(),
            packed_path,
            bytes.len() as f64 / 1e6,
            bytes.len() as f64 / nw.max(1) as f64,
            if skipped > 0 {
                format!("; skipped {skipped} non-scalar tensors (vector LUT not in .strand v1)")
            } else {
                String::new()
            }
        );
        return;
    }

    if let Some(packed_path) = &args.packed_v2_out {
        let mut v2s: Vec<PackedTensorV2> = Vec::new();
        // Per-tensor column-sign RHT flag, aligned to v2s. Only set when the tensor
        // actually carries RHT (has_rht_seed) AND --rht-cols was requested, so the
        // decoder's `has_rht_seed && rht_cols` col path stays self-consistent.
        let mut rht_cols_mask: Vec<bool> = Vec::new();
        let mut skipped = 0usize;
        for r in &quant_results {
            match &r.enc {
                Some(enc) => {
                    rht_cols_mask.push(args.rht_cols && enc.has_rht_seed);
                    v2s.push(PackedTensorV2 {
                        base: PackedTensor {
                            name: &r.name,
                            shape: &r.shape,
                            rht_seed: r.rht_seed,
                            l_bits: r.l_bits,
                            k_bits: r.k_bits,
                            vec_dim: 1,
                            enc,
                        },
                        block_len: r.block_len,
                    });
                }
                None => skipped += 1,
            }
        }
        assert!(
            !v2s.is_empty(),
            "--packed-v2-out: no scalar tensors to write (all {skipped} were vector \
             and carry no in-format LUT). Re-run on the clean scalar path (no --vec-dim)."
        );
        let source_sha256 = sha256::sha256(st.bytes());
        // When --sdsq-sideinfo is on, write the PACKED 12-byte seek-table records
        // (SCALEQ_IN_SDSQ) so scale_q is not also stored inline — this is what makes
        // the appended SDSQ section a net win instead of pure overhead. The loader
        // sources scale_q from SDSQ at load (byte-identical reconstruct).
        let write_res = format::write_strand_v2_rht(
            &v2s,
            source_sha256,
            args.strict_v2,
            args.sdsq_sideinfo,
            &rht_cols_mask,
        );
        let bytes = match write_res {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[quantize-model] ERROR writing .strand v2: {e}");
                eprintln!(
                    "[quantize-model] hint: pass --ragged-v2 (or --no-strict-v2) to emit a \
                     RAGGED v2 archive for odd-dim models (clears the ALL_STRICT file flag)."
                );
                std::process::exit(1);
            }
        };
        fs::write(packed_path, &bytes).expect("write .strand v2 archive");

        if args.outlier_pct > 0.0 {
            let wires: Vec<Option<OutlierWire>> = quant_results
                .iter()
                .filter(|r| r.enc.is_some())
                .map(|r| r.outlier.clone())
                .collect();
            let n_chan = wires.iter().filter(|w| w.is_some()).count();
            // --c2f-outl: gap-code positions (OUTL_FLAG_POS_RANS). Container-only —
            // reconstruct is byte-identical, so decode/MAC/SPRV are unchanged.
            if args.c2f_outl {
                append_outl_c2f(packed_path, &wires)
                    .unwrap_or_else(|e| panic!("append C2F OUTL section: {e}"));
            } else {
                append_outl(packed_path, &wires)
                    .unwrap_or_else(|e| panic!("append OUTL section: {e}"));
            }
            eprintln!(
                "[quantize-model] appended OUTL section ({}): {n_chan}/{} tensors carry the \
                 sparse-outlier channel",
                if args.c2f_outl { "C2F pos-rANS" } else { "inline idx" },
                wires.len()
            );
        }

        // SDSQ side-info (sprint Lever 1, opt-in): entropy-code the per-block scale_q
        // stream into an EOF-chained section. Appended AFTER OUTL and BEFORE SPRV so the
        // SPRV seal stays outermost; the v2 seek table is left byte-untouched (decode
        // overwrites scale_q losslessly). Collect scale_q in the SAME tensor/block order
        // the v2 writer used (filtered to encoded tensors), so it indexes 1:1 against the
        // archive's BlockOffsetRecord table.
        if args.sdsq_sideinfo {
            let scale_q: Vec<i32> = quant_results
                .iter()
                .filter(|r| r.enc.is_some())
                .flat_map(|r| r.enc.as_ref().unwrap().blocks.iter().map(|b| b.scale_q))
                .collect();
            let n_blocks = scale_q.len();
            let raw_inline_bits = (n_blocks as u64) * 32; // bare 32-bit/block ship cost
            append_sdsq(packed_path, &scale_q)
                .unwrap_or_else(|e| panic!("append SDSQ section: {e}"));
            // Report the rANS stream cost vs the would-be inline 32-bit footprint.
            let sec = strand_quant::sideinfo_wire::read_sdsq(packed_path)
                .ok()
                .flatten()
                .map(|s| s.scale_q.len())
                .unwrap_or(n_blocks);
            let nw: usize = quant_results.iter().map(|r| r.recon.len()).sum();
            let coded = strand_quant::sideinfo_rans::encode_scale_q(&scale_q).len() as u64 * 8;
            eprintln!(
                "[quantize-model] appended SDSQ section: {n_blocks} blocks ({sec} scale_q), \
                 rANS={:.0} bits ({:.4} bpw) vs inline 32-bit={raw_inline_bits} bits \
                 ({:.4} bpw) -> recover ~{:.4} bpw (ship projection; seek table untouched)",
                coded as f64,
                coded as f64 / nw.max(1) as f64,
                raw_inline_bits as f64 / nw.max(1) as f64,
                (raw_inline_bits as f64 - coded as f64).max(0.0) / nw.max(1) as f64,
            );
        }

        if !args.no_sprv {
            let sprv = append_sprv_computed(packed_path, false)
                .unwrap_or_else(|e| panic!("append SPRV section: {e}"));
            let mut root_hex = String::with_capacity(64);
            for b in &sprv.model_root {
                root_hex.push_str(&format!("{b:02x}"));
            }
            eprintln!("[quantize-model] appended SPRV trailer (v2): model_root={root_hex}");
        }

        let final_len = fs::metadata(packed_path).map(|m| m.len()).unwrap_or(bytes.len() as u64);
        let nw: usize = quant_results.iter().map(|r| r.recon.len()).sum();
        let mut sha_hex = String::with_capacity(16);
        for b in &source_sha256[0..8] {
            sha_hex.push_str(&format!("{b:02x}"));
        }
        eprintln!(
            "[quantize-model] wrote {} tensors -> {} (v2/STR2, strict={}, {:.1} MB, {:.4} bytes/weight FULL FILE incl. OUTL+SPRV, src_sha256[0..8]={}){}",
            v2s.len(),
            packed_path,
            args.strict_v2,
            final_len as f64 / 1e6,
            final_len as f64 / nw.max(1) as f64,
            sha_hex,
            if skipped > 0 {
                format!("; skipped {skipped} non-scalar tensors (vector LUT not in .strand v2)")
            } else {
                String::new()
            }
        );
        return;
    }

    let mut by_name: HashMap<String, OutTensor> = HashMap::new();
    let mut agg_num = 0.0f64;
    let mut agg_den = 0usize;
    let mut weighted_rel = 0.0f64;
    let mut sidecar = String::from("{\n  \"tensors\": [\n");
    let mut first = true;
    for r in &quant_results {
        let n = r.recon.len();
        agg_num += r.bpw * n as f64;
        agg_den += n;
        weighted_rel += r.rel_rms_pct * n as f64;
        if !first {
            sidecar.push_str(",\n");
        }
        first = false;
        sidecar.push_str(&format!(
            "    {{\"name\": \"{}\", \"n\": {}, \"bits\": {}, \"bpw\": {:.6}, \"rel_rms_pct\": {:.6}}}",
            r.name, n, r.bits, r.bpw, r.rel_rms_pct
        ));
        if !args.measure_only {
            by_name.insert(
                r.name.clone(),
                OutTensor {
                    name: r.name.clone(),
                    shape: r.shape.clone(),
                    data: r.recon.clone(),
                },
            );
        }
    }

    let agg_bpw = if agg_den > 0 { agg_num / agg_den as f64 } else { 0.0 };
    let agg_rel = if agg_den > 0 { weighted_rel / agg_den as f64 } else { 0.0 };
    sidecar.push_str("\n  ],\n");
    sidecar.push_str(&format!(
        "  \"aggregate\": {{\"quantized_weights\": {}, \"effective_bpw\": {:.6}, \"weighted_rel_rms_pct\": {:.6}}},\n",
        agg_den, agg_bpw, agg_rel
    ));
    let (cfg_for_log, affine_min_log, tail_biting_log) = resolve_cfg(args.bits);
    let mp_label = if args.mp_config.is_some() { "true" } else { "false" };
    
    sidecar.push_str(&format!(
        "  \"config\": {{\"bits\": {}, \"l\": {}, \"k\": {}, \"rht\": {}, \"tail_biting\": {}, \"affine_min\": {}, \"calibrated\": false, \"block_hessian\": false, \"mixed_precision\": {}, \"vec_dim\": {}, \"learned_codebook\": {}}}\n}}\n",
        args.bits, cfg_for_log.l_bits, cfg_for_log.k_bits, args.rht,
        tail_biting_log, affine_min_log, mp_label,
        cfg_for_log.vec_dim(), args.learned_codebook
    ));

    if args.measure_only {
        let json_path = format!("{}.json", args.output);
        if !args.output.is_empty() {
            fs::write(&json_path, &sidecar).expect("write sidecar json");
        }
        eprintln!(
            "[quantize-model] MEASURE-ONLY: {} tensors quantized in {:.1}s (no model written)",
            quant_results.len(),
            t0.elapsed().as_secs_f64()
        );
        eprintln!(
            "[quantize-model] AGGREGATE effective bpw = {:.4} over {} quantized weights ; weighted rel-RMS = {:.2}%",
            agg_bpw, agg_den, agg_rel
        );
        return;
    }

    for p in passthrough {
        by_name.insert(p.name.clone(), p);
    }
    let out_tensors: Vec<OutTensor> = st
        .order
        .iter()
        .map(|n| by_name.remove(n).expect("every input tensor accounted for"))
        .collect();

    write_safetensors(&args.output, &out_tensors).expect("write output safetensors");
    let json_path = format!("{}.json", args.output);
    fs::write(&json_path, &sidecar).expect("write sidecar json");

    eprintln!(
        "[quantize-model] wrote {} ({} tensors) + {} in {:.1}s",
        args.output,
        out_tensors.len(),
        json_path,
        t0.elapsed().as_secs_f64()
    );
    eprintln!(
        "[quantize-model] AGGREGATE effective bpw = {:.4} over {} quantized weights ; weighted rel-RMS = {:.2}%",
        agg_bpw, agg_den, agg_rel
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args() -> Args {
        Args {
            input: String::new(),
            output: String::new(),
            bits: 2,
            l: 12,
            tail_biting: false,
            affine_mode: AffineMode::Auto,
            rht: true,
            rht_cols: false,
            threads: 1,
            only: None,
            measure_only: false,
            quality: false,
            packed_out: None,
            packed_v2_out: None,
            strict_v2: true,
            sdsq_sideinfo: false,
            c2f_outl: false,
            mp_config: None,
            rung_config: None,
            dump_indices: None,
            vec_dim: 1,
            learned_codebook: false,
            no_sprv: false,
            outlier_pct: 1.0,
            outlier_bits: 8,
            skip_manifest: None,
            reuse_from: None,
            actmean: None,
        }
    }

    #[test]
    fn identity_skip_pins_the_f32_lanes() {
        let args = test_args();
        let key = cache_config_key(&args, false, false);
        let key_metric = cache_config_key(&args, true, false);
        let key_search = cache_config_key(&args, true, true);
        assert_ne!(key, key_metric, "STRAND_F32_METRIC must be pinned in the config key");
        assert_ne!(key_metric, key_search, "STRAND_F32_SEARCH must be pinned in the config key");
        
        assert_eq!(key, cache_config_key(&test_args(), false, false));

        let gt: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.013).sin()).collect();
        let h = encode_cache::hash_f32(&gt);
        let mut m = Manifest::new(key.clone());
        m.record("model.layers.0.mlp.down_proj.weight", &h, 2);

        assert!(encode_cache::should_skip(
            "model.layers.0.mlp.down_proj.weight", &h, 2, &key, Some(&m)
        ));
        
        assert!(!encode_cache::should_skip(
            "model.layers.0.mlp.down_proj.weight", &h, 2, &key_metric, Some(&m)
        ));
        assert!(!encode_cache::should_skip(
            "model.layers.0.mlp.down_proj.weight", &h, 2, &key_search, Some(&m)
        ));
        
        let mut gt2 = gt.clone();
        gt2[7] = f32::from_bits(gt2[7].to_bits() ^ 1); 
        assert!(!encode_cache::should_skip(
            "model.layers.0.mlp.down_proj.weight",
            &encode_cache::hash_f32(&gt2),
            2,
            &key,
            Some(&m)
        ));
        
        assert!(!encode_cache::should_skip(
            "model.layers.0.mlp.down_proj.weight", &h, 3, &key, Some(&m)
        ));
    }

    #[test]
    fn config_key_covers_every_encode_lever() {
        let base = cache_config_key(&test_args(), false, false);
        let variants: Vec<Args> = vec![
            { let mut a = test_args(); a.l = 10; a },
            { let mut a = test_args(); a.quality = true; a },
            { let mut a = test_args(); a.tail_biting = true; a },
            { let mut a = test_args(); a.affine_mode = AffineMode::On; a },
            { let mut a = test_args(); a.rht = false; a },
            { let mut a = test_args(); a.vec_dim = 2; a },
            { let mut a = test_args(); a.learned_codebook = true; a },
            { let mut a = test_args(); a.outlier_pct = 0.0; a },
            { let mut a = test_args(); a.outlier_bits = 6; a },
        ];
        for (i, v) in variants.iter().enumerate() {
            assert_ne!(
                base,
                cache_config_key(v, false, false),
                "lever variant #{i} did not change the config key"
            );
        }
    }

    #[test]
    fn prior_sidecar_parses_billing_rows() {
        let dir = std::env::temp_dir()
            .join(format!("qm-skip-sidecar-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("recon.safetensors.json");
        fs::write(
            &p,
            "{\n  \"tensors\": [\n    {\"name\": \"a.weight\", \"n\": 10, \"bits\": 2, \
             \"bpw\": 2.336914, \"rel_rms_pct\": 27.74}\n  ],\n  \
             \"aggregate\": {\"quantized_weights\": 10, \"effective_bpw\": 2.336914, \
             \"weighted_rel_rms_pct\": 27.74}\n}\n",
        )
        .unwrap();
        let m = load_prior_sidecar(p.to_str().unwrap());
        assert_eq!(m.len(), 1);
        let (bpw, rel) = m["a.weight"];
        assert!((bpw - 2.336914).abs() < 1e-9);
        assert!((rel - 27.74).abs() < 1e-9);
        
        assert!(load_prior_sidecar("/nonexistent/sidecar.json").is_empty());
        fs::remove_dir_all(&dir).ok();
    }
}
