
#![forbid(unsafe_code)]

use std::fs;
use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use strand_quant::encode::{BlockMeta, EncodedTensor};
use strand_quant::outlier_wire::OutlierWire;
use strand_quant::rht::{rht_forward_rows, RhtConfig};
use strand_quant::safetensor_io::SafeTensors;
use strand_quant::trellis::read_bits;
use strand_quant::gate_utils::{is_quantizable_linear, rht_seed_for};
use strand_quant::{encode_tensor_with, EncodeOpts, TrellisConfig};

fn write_safetensors_f32(path: &str, tensors: &[(String, Vec<u64>, Vec<f32>)]) -> std::io::Result<()> {
    let mut header = String::from("{");
    let mut offset = 0usize;
    for (i, (name, shape, data)) in tensors.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let nbytes = data.len() * 4;
        let shape_s = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            name, shape_s, offset, offset + nbytes
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
    for (_, _, data) in tensors {
        let mut buf = Vec::with_capacity(data.len() * 4);
        for &v in data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&buf)?;
    }
    f.flush()?;
    Ok(())
}

struct WireTensor {
    enc: EncodedTensor,
    outlier: Option<OutlierWire>,
}

#[allow(clippy::too_many_arguments)]
fn encode_for_wire(
    name: &str,
    gt: &[f32],
    in_features: usize,
    cfg: &TrellisConfig,
    opts: &EncodeOpts,
    use_rht: bool,
    outlier_pct: f64,
    outlier_bits: u32,
) -> WireTensor {
    let outliers: Option<(Vec<usize>, Vec<i32>, f32)> = if outlier_pct > 0.0 {
        let n = gt.len();
        let k = ((outlier_pct / 100.0) * n as f64).round() as usize;
        if k == 0 || n == 0 {
            None
        } else {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by(|&a, &b| {
                gt[b].abs().partial_cmp(&gt[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
            });
            let idx: Vec<usize> = order[..k].to_vec();
            let omax = idx.iter().fold(0f32, |m, &i| m.max(gt[i].abs())).max(1e-12);
            let ob = outlier_bits.clamp(2, 16);
            let levels = ((1i64 << (ob - 1)) - 1) as f32;
            let codes: Vec<i32> =
                idx.iter().map(|&i| (gt[i] / omax * levels).round() as i32).collect();
            Some((idx, codes, omax))
        }
    } else {
        None
    };

    let bulk_gt: Vec<f32>;
    let work_src: &[f32] = if let Some((idx, _, _)) = &outliers {
        let mut b = gt.to_vec();
        for &i in idx {
            b[i] = 0.0;
        }
        bulk_gt = b;
        &bulk_gt
    } else {
        gt
    };

    let work = if use_rht {
        let rcfg = RhtConfig::from_seed(rht_seed_for(name));
        rht_forward_rows(work_src, &rcfg, in_features)
    } else {
        work_src.to_vec()
    };

    let mut enc = encode_tensor_with(&work, cfg, opts);
    enc.has_rht_seed = use_rht;

    let outlier = outliers.map(|(idx, codes, omax)| {
        let n = gt.len();
        let idx_bits = if n <= 1 { 1 } else { usize::BITS - (n - 1).leading_zeros() };
        let mut entries: Vec<(u32, i32)> =
            idx.into_iter().map(|i| i as u32).zip(codes).collect();
        entries.sort_unstable_by_key(|&(i, _)| i);
        OutlierWire {
            omax_bits: omax.to_bits(),
            entries,
            idx_bits,
            val_bits: outlier_bits.clamp(2, 16),
        }
    });

    WireTensor { enc, outlier }
}

fn bits_eq(a: &[u8], a_start: usize, b: &[u8], b_start: usize, nbits: usize) -> bool {
    if nbits == 0 {
        return true;
    }
    if a_start % 8 == 0 && b_start % 8 == 0 {
        let full = nbits / 8;
        let (ab, bb) = (a_start / 8, b_start / 8);
        if a[ab..ab + full] != b[bb..bb + full] {
            return false;
        }
        let rem = nbits % 8;
        if rem == 0 {
            return true;
        }
        let mask = (1u16 << rem) - 1;
        let ga = a.get(ab + full).copied().unwrap_or(0) as u16 & mask;
        let gb = b.get(bb + full).copied().unwrap_or(0) as u16 & mask;
        return ga == gb;
    }
    let mut done = 0usize;
    while done < nbits {
        let take = (nbits - done).min(16) as u32;
        if read_bits(a, a_start + done, take) != read_bits(b, b_start + done, take) {
            return false;
        }
        done += take as usize;
    }
    true
}

const FULL_FILE_HEADER_BYTES: u64 = 12;

const PATCH_FILE_HEADER_BYTES: u64 = 84;

fn v1_tensor_header_bytes(name: &str, ndim: usize) -> u64 {
    40 + name.len() as u64 + 8 * ndim as u64
}

fn block_record_bytes(b: &BlockMeta) -> u64 {
    24 + b.sub_scales.len() as u64 + b.mins.len() as u64
}

fn outlier_wire_bytes(o: &OutlierWire) -> u64 {
    12 + (o.entries.len() as u64 * (o.idx_bits + o.val_bits) as u64).div_ceil(8)
}

fn full_tensor_bytes(name: &str, ndim: usize, w: &WireTensor) -> u64 {
    v1_tensor_header_bytes(name, ndim)
        + w.enc.bits.len() as u64
        + w.enc.blocks.iter().map(block_record_bytes).sum::<u64>()
        + w.outlier.as_ref().map(outlier_wire_bytes).unwrap_or(0)
}

fn patch_record_prefix_bytes(name: &str) -> u64 {
    5 + name.len() as u64
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    
    Diffed,
    
    Added,
    
    Removed,
    
    Replaced,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Diffed => "diffed",
            Status::Added => "added",
            Status::Removed => "removed",
            Status::Replaced => "replaced",
        }
    }
}

struct Row {
    name: String,
    status: Status,
    n_blocks: usize,
    n_changed: usize,
    changed_idx: Vec<u32>,
    outlier_changed: bool,
    full_bytes: u64,
    delta_bytes: u64,
}

fn diff_pair(name: &str, ndim: usize, k_bits: u32, a: &WireTensor, b: &WireTensor) -> Row {
    assert_eq!(
        a.enc.blocks.len(),
        b.enc.blocks.len(),
        "{name}: same shape must partition into the same block count"
    );
    assert_eq!(a.enc.total, b.enc.total, "{name}: total mismatch at equal shape");

    let k = k_bits as usize;
    let mut a_cursor = 0usize;
    let mut b_cursor = 0usize;
    let mut changed_idx: Vec<u32> = Vec::new();
    let mut delta_blocks_bytes: u64 = 0;

    for (i, (ba, bb)) in a.enc.blocks.iter().zip(&b.enc.blocks).enumerate() {
        let na = ba.n as usize * k;
        let nb = bb.n as usize * k;
        let identical = ba == bb && na == nb && bits_eq(&a.enc.bits, a_cursor, &b.enc.bits, b_cursor, na);
        if !identical {
            changed_idx.push(i as u32);
            delta_blocks_bytes += 4 + block_record_bytes(bb) + (nb as u64).div_ceil(8);
        }
        a_cursor += na;
        b_cursor += nb;
    }

    let outlier_changed = a.outlier != b.outlier;
    let n_changed = changed_idx.len();
    let full_bytes = full_tensor_bytes(name, ndim, b);
    let delta_bytes = if n_changed == 0 && !outlier_changed {
        
        patch_record_prefix_bytes(name) + 4
    } else {
        let patched = patch_record_prefix_bytes(name)
            + 8
            + delta_blocks_bytes
            + 1
            + if outlier_changed {
                b.outlier.as_ref().map(outlier_wire_bytes).unwrap_or(0)
            } else {
                0
            };
        patched.min(1 + full_bytes)
    };

    Row {
        name: name.to_string(),
        status: Status::Diffed,
        n_blocks: b.enc.blocks.len(),
        n_changed,
        changed_idx,
        outlier_changed,
        full_bytes,
        delta_bytes,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AffineMode {
    Auto,
    On,
    Off,
}

struct DiffArgs {
    a: String,
    b: String,
    bits: u32,
    l: u32,
    quality: bool,
    tail_biting: bool,
    affine_mode: AffineMode,
    rht: bool,
    threads: usize,
    only: Option<String>,
    outlier_pct: f64,
    outlier_bits: u32,
    json: Option<String>,
    gpu_encode: bool,
}

impl Default for DiffArgs {
    fn default() -> Self {
        DiffArgs {
            a: String::new(),
            b: String::new(),
            bits: 3,
            l: 0,
            quality: false,
            tail_biting: false,
            affine_mode: AffineMode::Auto,
            rht: true,
            threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
            only: None,
            outlier_pct: 0.0,
            outlier_bits: 8,
            json: None,
            gpu_encode: false,
        }
    }
}

fn resolve_cfg(args: &DiffArgs) -> (TrellisConfig, bool, bool) {
    let cfg = if args.l != 0 {
        TrellisConfig::for_bpw_l(args.bits as f64, args.l)
    } else if args.quality {
        TrellisConfig::for_bpw_quality(args.bits as f64)
    } else {
        TrellisConfig::for_bpw(args.bits as f64)
    };
    let affine_min = match args.affine_mode {
        AffineMode::On => true,
        AffineMode::Off => false,
        AffineMode::Auto => args.bits >= 4,
    };
    let tail_biting = args.tail_biting || affine_min;
    (cfg, affine_min, tail_biting)
}

fn usage() -> ! {
    eprintln!(
        "strand-delta — block-level diff between two checkpoints' STRAND encodings\n\
         \n\
         strand-delta diff <a.safetensors> <b.safetensors> --bits <2|3|4|5|6> [--l <4..14>]\n\
         \t[--outlier-channel PCT] [--outlier-bits N] [--no-rht] [--tail-biting]\n\
         \t[--affine-min auto|on|off] [--quality] [--threads N] [--only SUBSTR]\n\
         \t[--json PATH] [--gpu-encode]\n\
         strand-delta smoke\n\
         \n\
         diff:  encode the matching linear tensors of both files through the exact\n\
         \t   quantize-model path and report identical vs changed 256-weight blocks\n\
         \t   (identity = byte equality of the block's bit slice + BlockMeta), plus\n\
         \t   the bytes a changed-blocks-only patch ships vs the full archive.\n\
         \t   Flags mirror quantize-model. Scalar trellis only (no --vec-dim: the\n\
         \t   vector path's per-tensor learned LUT needs its own diff treatment).\n\
         \t   CPU encode is pinned (STRAND_NO_GPU=1) unless --gpu-encode.\n\
         smoke: synthesize two tiny checkpoints differing in known rows; assert the\n\
         \t   changed-block sets (A-vs-A all-identical, A-vs-B exact rows, outlier\n\
         \t   churn superset). Exits non-zero on any mismatch."
    );
    std::process::exit(2);
}

fn parse_diff_args(mut it: std::env::Args) -> DiffArgs {
    let mut a = DiffArgs::default();
    let mut pos: Vec<String> = Vec::new();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--bits" => a.bits = it.next().expect("--bits").parse().expect("bits int"),
            "--l" => a.l = it.next().expect("--l").parse().expect("l int"),
            "--quality" => a.quality = true,
            "--tail-biting" => a.tail_biting = true,
            "--no-tail-biting" => a.tail_biting = false,
            "--affine-min" => {
                a.affine_mode = match it.next().as_deref() {
                    Some("on") => AffineMode::On,
                    Some("off") => AffineMode::Off,
                    Some("auto") | None => AffineMode::Auto,
                    Some(x) => panic!("--affine-min expects on|off|auto, got {x}"),
                }
            }
            "--no-rht" => a.rht = false,
            "--threads" => a.threads = it.next().expect("--threads").parse().expect("threads int"),
            "--only" => a.only = Some(it.next().expect("--only needs a substring")),
            "--outlier-channel" => {
                a.outlier_pct =
                    it.next().expect("--outlier-channel needs PCT").parse().expect("outlier-channel f64")
            }
            "--outlier-bits" => {
                a.outlier_bits =
                    it.next().expect("--outlier-bits needs N").parse().expect("outlier-bits int")
            }
            "--json" => a.json = Some(it.next().expect("--json needs a path")),
            "--gpu-encode" => a.gpu_encode = true,
            "--vec-dim" | "--learned-codebook" => panic!(
                "{arg}: the vector trellis carries a per-tensor learned LUT as side \
                 info; diffing it is not wired — scalar path only"
            ),
            "-h" | "--help" => usage(),
            other if !other.starts_with("--") => pos.push(other.to_string()),
            other => panic!("unknown arg {other}"),
        }
    }
    assert!(pos.len() == 2, "diff needs exactly two positional paths: <a.safetensors> <b.safetensors>");
    a.a = pos.remove(0);
    a.b = pos.remove(0);
    assert!(
        (2..=6).contains(&a.bits),
        "--bits must be 2, 3, 4, 5, or 6 (same range as quantize-model)"
    );
    a
}

struct Report {
    rows: Vec<Row>,
    total_blocks: usize,
    total_identical: usize,
    total_changed: usize,
    total_full: u64,
    total_delta: u64,
}

fn run_diff(args: &DiffArgs) -> Report {
    if !args.gpu_encode {
        std::env::set_var("STRAND_NO_GPU", "1");
    }
    let t0 = Instant::now();
    let (cfg, affine_min, tail_biting) = resolve_cfg(args);
    let opts = EncodeOpts { adaptive: true, tail_biting, affine_min, silence_bonus: 0.0, entropy_bonus_scale: 0.0, entropy_bonus_two_pass: false };

    eprintln!(
        "[strand-delta] a={} b={} bits={} L={} k={} rht={} tail_biting={} affine_min={} \
         outlier={}%/{}b threads={} cpu_pin={}",
        args.a, args.b, args.bits, cfg.l_bits, cfg.k_bits, args.rht, tail_biting, affine_min,
        args.outlier_pct, args.outlier_bits, args.threads, !args.gpu_encode
    );

    let st_a = SafeTensors::open(&args.a).expect("open checkpoint A");
    let st_b = SafeTensors::open(&args.b).expect("open checkpoint B");

    let only_match =
        |name: &str| args.only.as_ref().map(|s| name.contains(s.as_str())).unwrap_or(true);

    struct Job {
        name: String,
        status: Status,
    }
    let mut jobs: Vec<Job> = Vec::new();
    for name in &st_a.order {
        if !only_match(name) {
            continue;
        }
        let ta = &st_a.tensors[name];
        let a_q = is_quantizable_linear(name, &ta.shape);
        let (b_q, b_shape) = match st_b.tensors.get(name) {
            Some(tb) => (is_quantizable_linear(name, &tb.shape), Some(tb.shape.clone())),
            None => (false, None),
        };
        match (a_q, b_q) {
            (true, true) if b_shape.as_deref() == Some(&ta.shape[..]) => {
                jobs.push(Job { name: name.clone(), status: Status::Diffed })
            }
            (true, true) => jobs.push(Job { name: name.clone(), status: Status::Replaced }),
            (true, false) => jobs.push(Job { name: name.clone(), status: Status::Removed }),
            _ => {}
        }
    }
    for name in &st_b.order {
        if !only_match(name) {
            continue;
        }
        let tb = &st_b.tensors[name];
        if is_quantizable_linear(name, &tb.shape)
            && !st_a.tensors.get(name).map(|ta| is_quantizable_linear(name, &ta.shape)).unwrap_or(false)
        {
            jobs.push(Job { name: name.clone(), status: Status::Added });
        }
    }
    eprintln!(
        "[strand-delta] {} matched tensors ({} diffed, {} added/removed/replaced)",
        jobs.len(),
        jobs.iter().filter(|j| j.status == Status::Diffed).count(),
        jobs.iter().filter(|j| j.status != Status::Diffed).count(),
    );

    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, Row)>> = Mutex::new(Vec::new());
    let nthreads = args.threads.max(1);
    std::thread::scope(|scope| {
        for _ in 0..nthreads {
            scope.spawn(|| loop {
                let ji = next.fetch_add(1, Ordering::Relaxed);
                if ji >= jobs.len() {
                    break;
                }
                let job = &jobs[ji];
                let row = match job.status {
                    Status::Diffed => {
                        let ta = &st_a.tensors[&job.name];
                        let tb = &st_b.tensors[&job.name];
                        let in_features = *ta.shape.last().unwrap() as usize;
                        let wa = encode_for_wire(
                            &job.name, &st_a.to_f32(ta), in_features, &cfg, &opts, args.rht,
                            args.outlier_pct, args.outlier_bits,
                        );
                        let wb = encode_for_wire(
                            &job.name, &st_b.to_f32(tb), in_features, &cfg, &opts, args.rht,
                            args.outlier_pct, args.outlier_bits,
                        );
                        diff_pair(&job.name, ta.shape.len(), cfg.k_bits, &wa, &wb)
                    }
                    Status::Added | Status::Replaced => {
                        let tb = &st_b.tensors[&job.name];
                        let in_features = *tb.shape.last().unwrap() as usize;
                        let wb = encode_for_wire(
                            &job.name, &st_b.to_f32(tb), in_features, &cfg, &opts, args.rht,
                            args.outlier_pct, args.outlier_bits,
                        );
                        let full = full_tensor_bytes(&job.name, tb.shape.len(), &wb);
                        Row {
                            name: job.name.clone(),
                            status: job.status,
                            n_blocks: wb.enc.blocks.len(),
                            n_changed: wb.enc.blocks.len(),
                            changed_idx: Vec::new(),
                            outlier_changed: wb.outlier.is_some(),
                            full_bytes: full,
                            delta_bytes: 1 + full,
                        }
                    }
                    Status::Removed => Row {
                        name: job.name.clone(),
                        status: Status::Removed,
                        n_blocks: 0,
                        n_changed: 0,
                        changed_idx: Vec::new(),
                        outlier_changed: false,
                        full_bytes: 0,
                        delta_bytes: patch_record_prefix_bytes(&job.name),
                    },
                };
                eprintln!(
                    "[done {}/{}] {:<44} blocks={} changed={}{}",
                    ji + 1,
                    jobs.len(),
                    row.name,
                    row.n_blocks,
                    row.n_changed,
                    if row.outlier_changed { " outlier-changed" } else { "" }
                );
                results.lock().unwrap().push((ji, row));
            });
        }
    });

    let mut rows: Vec<(usize, Row)> = results.into_inner().unwrap();
    rows.sort_by_key(|(i, _)| *i);
    let rows: Vec<Row> = rows.into_iter().map(|(_, r)| r).collect();

    let total_blocks: usize = rows.iter().map(|r| r.n_blocks).sum();
    let total_changed: usize = rows.iter().map(|r| r.n_changed).sum();
    let total_full: u64 =
        rows.iter().map(|r| r.full_bytes).sum::<u64>() + FULL_FILE_HEADER_BYTES;
    let total_delta: u64 =
        rows.iter().map(|r| r.delta_bytes).sum::<u64>() + PATCH_FILE_HEADER_BYTES;

    eprintln!("[strand-delta] diff complete in {:.1}s", t0.elapsed().as_secs_f64());
    Report {
        rows,
        total_blocks,
        total_identical: total_blocks - total_changed,
        total_changed,
        total_full,
        total_delta,
    }
}

fn print_report(r: &Report) {
    println!(
        "{:<44} {:>8} {:>8} {:>8} {:>5} {:>12} {:>12} {:>7}",
        "tensor", "blocks", "ident", "chang", "outl", "full(B)", "delta(B)", "ratio"
    );
    for row in &r.rows {
        let ratio = if row.full_bytes > 0 {
            format!("{:>6.1}%", 100.0 * row.delta_bytes as f64 / row.full_bytes as f64)
        } else {
            "      -".to_string()
        };
        println!(
            "{:<44} {:>8} {:>8} {:>8} {:>5} {:>12} {:>12} {}",
            row.name,
            row.n_blocks,
            row.n_blocks - row.n_changed,
            row.n_changed,
            if row.status == Status::Diffed {
                if row.outlier_changed { "yes" } else { "no" }
            } else {
                row.status.as_str()
            },
            row.full_bytes,
            row.delta_bytes,
            ratio
        );
    }
    let pct_ident = if r.total_blocks > 0 {
        100.0 * r.total_identical as f64 / r.total_blocks as f64
    } else {
        0.0
    };
    let pct_delta = if r.total_full > 0 {
        100.0 * r.total_delta as f64 / r.total_full as f64
    } else {
        0.0
    };
    println!(
        "{:<44} {:>8} {:>8} {:>8} {:>5} {:>12} {:>12} {:>6.1}%",
        "TOTAL", r.total_blocks, r.total_identical, r.total_changed, "-", r.total_full,
        r.total_delta, pct_delta
    );
    let win = if r.total_delta > 0 { r.total_full as f64 / r.total_delta as f64 } else { f64::INFINITY };
    println!(
        "identical blocks: {}/{} ({:.1}%); patch {} B vs full {} B = {:.1}%{}",
        r.total_identical,
        r.total_blocks,
        pct_ident,
        r.total_delta,
        r.total_full,
        pct_delta,
        if win >= 1.05 {
            format!(" ({win:.1}x smaller)")
        } else {
            " (no win — nearly everything changed and/or the fixed patch header dominates at this tensor count)".to_string()
        }
    );
}

fn write_json(path: &str, args: &DiffArgs, r: &Report) {
    let (cfg, affine_min, tail_biting) = resolve_cfg(args);
    let mut s = String::from("{\n  \"tensors\": [\n");
    for (i, row) in r.rows.iter().enumerate() {
        if i > 0 {
            s.push_str(",\n");
        }
        s.push_str(&format!(
            "    {{\"name\": \"{}\", \"status\": \"{}\", \"blocks\": {}, \"identical\": {}, \
             \"changed\": {}, \"outlier_changed\": {}, \"full_bytes\": {}, \"delta_bytes\": {}}}",
            row.name,
            row.status.as_str(),
            row.n_blocks,
            row.n_blocks - row.n_changed,
            row.n_changed,
            row.outlier_changed,
            row.full_bytes,
            row.delta_bytes
        ));
    }
    s.push_str(&format!(
        "\n  ],\n  \"total\": {{\"blocks\": {}, \"identical\": {}, \"changed\": {}, \
         \"full_bytes\": {}, \"delta_bytes\": {}, \"delta_pct\": {:.4}}},\n",
        r.total_blocks,
        r.total_identical,
        r.total_changed,
        r.total_full,
        r.total_delta,
        if r.total_full > 0 { 100.0 * r.total_delta as f64 / r.total_full as f64 } else { 0.0 }
    ));
    s.push_str(&format!(
        "  \"config\": {{\"a\": \"{}\", \"b\": \"{}\", \"bits\": {}, \"l\": {}, \"k\": {}, \
         \"rht\": {}, \"tail_biting\": {}, \"affine_min\": {}, \"outlier_pct\": {}, \
         \"outlier_bits\": {}}}\n}}\n",
        args.a, args.b, args.bits, cfg.l_bits, cfg.k_bits, args.rht, tail_biting, affine_min,
        args.outlier_pct, args.outlier_bits
    ));
    fs::write(path, s).expect("write --json report");
    eprintln!("[strand-delta] wrote {path}");
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn rng_f32(state: &mut u64) -> f32 {
    ((xorshift64(state) >> 40) as f32) / (1u64 << 24) as f32 - 0.5
}

fn gen_rows(rows: usize, cols: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed;
    (0..rows * cols).map(|_| rng_f32(&mut s) * amp).collect()
}

fn row_set(changed: &[u32]) -> Vec<u32> {
    let mut v = changed.to_vec();
    v.sort_unstable();
    v
}

fn run_smoke() -> Result<(), String> {
    std::env::set_var("STRAND_NO_GPU", "1");

    let dir = std::env::temp_dir().join(format!("strand-delta-smoke-{}", std::process::id()));
    fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let a_path = dir.join("a.safetensors");
    let b_path = dir.join("b.safetensors");

    const Q: &str = "model.layers.0.self_attn.q_proj.weight";
    const D: &str = "model.layers.0.mlp.down_proj.weight";
    let mut q_a = gen_rows(8, 256, 0x5EED_0001, 1.0);
    let mut d_a = gen_rows(4, 512, 0x5EED_0002, 1.0);
    let mut q_b = q_a.clone();
    let mut d_b = d_a.clone();
    q_b[3 * 256..4 * 256].copy_from_slice(&gen_rows(1, 256, 0xB00B_0003, 1.5));
    d_b[512..3 * 512].copy_from_slice(&gen_rows(2, 512, 0xB00B_0004, 1.5));

    let write = |path: &std::path::Path, q: &mut Vec<f32>, d: &mut Vec<f32>| {
        write_safetensors_f32(
            path.to_str().unwrap(),
            &[
                (Q.to_string(), vec![8, 256], std::mem::take(q)),
                (D.to_string(), vec![4, 512], std::mem::take(d)),
            ],
        )
    };
    write(&a_path, &mut q_a, &mut d_a).map_err(|e| format!("write A: {e}"))?;
    write(&b_path, &mut q_b, &mut d_b).map_err(|e| format!("write B: {e}"))?;

    let base_args = |a: &std::path::Path, b: &std::path::Path, outlier_pct: f64| DiffArgs {
        a: a.to_str().unwrap().to_string(),
        b: b.to_str().unwrap().to_string(),
        bits: 2,
        l: 12,
        outlier_pct,
        threads: 2,
        ..DiffArgs::default()
    };
    let find = |r: &Report, name: &str| -> Row {
        let row = r.rows.iter().find(|x| x.name == name).expect("tensor present");
        Row {
            name: row.name.clone(),
            status: row.status,
            n_blocks: row.n_blocks,
            n_changed: row.n_changed,
            changed_idx: row.changed_idx.clone(),
            outlier_changed: row.outlier_changed,
            full_bytes: row.full_bytes,
            delta_bytes: row.delta_bytes,
        }
    };

    eprintln!("[smoke] case 0: A vs A (determinism)");
    let r0 = run_diff(&base_args(&a_path, &a_path, 0.0));
    print_report(&r0);
    if r0.total_changed != 0 {
        return Err(format!("case 0: expected 0 changed blocks, got {}", r0.total_changed));
    }
    if r0.total_blocks != 16 {
        return Err(format!("case 0: expected 16 blocks total, got {}", r0.total_blocks));
    }

    eprintln!("[smoke] case 1: A vs B, exact changed-block sets");
    let r1 = run_diff(&base_args(&a_path, &b_path, 0.0));
    print_report(&r1);
    let q1 = find(&r1, Q);
    let d1 = find(&r1, D);
    if row_set(&q1.changed_idx) != vec![3] {
        return Err(format!("case 1: q_proj changed blocks {:?}, expected [3]", q1.changed_idx));
    }
    if row_set(&d1.changed_idx) != vec![2, 3, 4, 5] {
        return Err(format!(
            "case 1: down_proj changed blocks {:?}, expected [2,3,4,5]",
            d1.changed_idx
        ));
    }
    if r1.total_identical != 11 || r1.total_changed != 5 {
        return Err(format!(
            "case 1: identical/changed = {}/{}, expected 11/5",
            r1.total_identical, r1.total_changed
        ));
    }
    if r1.total_delta >= r1.total_full {
        return Err(format!(
            "case 1: delta ({}) must undercut full ({})",
            r1.total_delta, r1.total_full
        ));
    }

    eprintln!("[smoke] case 2: A vs B + outlier channel (superset + churn report)");
    let r2 = run_diff(&base_args(&a_path, &b_path, 1.0));
    print_report(&r2);
    let q2 = find(&r2, Q);
    let d2 = find(&r2, D);
    for must in [3u32] {
        if !q2.changed_idx.contains(&must) {
            return Err(format!("case 2: q_proj changed {:?} lost block {must}", q2.changed_idx));
        }
    }
    for must in [2u32, 3, 4, 5] {
        if !d2.changed_idx.contains(&must) {
            return Err(format!("case 2: down_proj changed {:?} lost block {must}", d2.changed_idx));
        }
    }
    let churn = r2.total_changed.saturating_sub(r1.total_changed);
    eprintln!(
        "[smoke] outlier-channel churn on this fixture: +{churn} blocks over case 1 \
         ({} -> {} changed; the fixture's 1.5x re-rolled rows steal most of the global \
         top-1% — real PV drift churns far less; measure there)",
        r1.total_changed, r2.total_changed
    );

    let _ = fs::remove_dir_all(&dir);
    eprintln!(
        "[smoke] PASS — case0 0/16 changed; case1 changed q={:?} d={:?} (5/16, delta {} B vs full {} B); \
         case2 superset held (+{churn} churn)",
        q1.changed_idx, d1.changed_idx, r1.total_delta, r1.total_full
    );
    Ok(())
}

fn main() {
    let mut it = std::env::args();
    let _bin = it.next();
    match it.next().as_deref() {
        Some("diff") => {
            let args = parse_diff_args(it);
            let report = run_diff(&args);
            print_report(&report);
            if let Some(path) = &args.json {
                write_json(path, &args, &report);
            }
        }
        Some("smoke") => {
            if let Err(e) = run_smoke() {
                eprintln!("[smoke] FAIL: {e}");
                std::process::exit(1);
            }
        }
        _ => usage(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_changed_block_sets() {
        run_smoke().expect("smoke must pass");
    }

    #[test]
    fn bits_eq_paths() {
        let a = vec![0b1010_1100u8, 0b0111_0001, 0b1111_0000];
        let mut b = a.clone();
        assert!(bits_eq(&a, 0, &b, 0, 24));
        assert!(bits_eq(&a, 0, &b, 0, 21)); 
        assert!(bits_eq(&a, 3, &b, 3, 17)); 
        b[1] ^= 0b0001_0000; 
        assert!(!bits_eq(&a, 0, &b, 0, 24));
        assert!(!bits_eq(&a, 8, &b, 8, 8));
        assert!(bits_eq(&a, 0, &b, 0, 12)); 
        assert!(bits_eq(&a, 13, &b, 13, 11)); 
    }
}
