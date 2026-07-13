use std::path::Path;
use strand_decode_kernel::loader::StrandModel;
use strand_decode_kernel::outlier_mac::patched_weights;
use strand_quant::decode::{decode_lean_with_lut, decode_tensor_fixed_with_lut};
use strand_quant::provenance::{model_root_from_tensor_roots, tensor_root};
use strand_quant::provenance_io::{read_sprv, verify_archive_with, VerifyDepth};

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

struct ReconFile {
    bytes: Vec<u8>,
    header: String,
    data_start: usize,
}

impl ReconFile {
    fn open(path: &str) -> ReconFile {
        let bytes = std::fs::read(path).expect("recon-check: read recon safetensors");
        let hdr_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header = String::from_utf8(bytes[8..8 + hdr_len].to_vec()).expect("header utf8");
        ReconFile {
            bytes,
            header,
            data_start: 8 + hdr_len,
        }
    }

    fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let key = format!("\"{name}\":{{");
        let kpos = self.header.find(&key)?;
        let obj = &self.header[kpos + key.len()..];
        let end = obj.find('}')?;
        let obj = &obj[..end];
        let dpos = obj.find("\"data_offsets\"")?;
        let rest = &obj[dpos..];
        let lb = rest.find('[')?;
        let rb = rest.find(']')?;
        let mut it = rest[lb + 1..rb]
            .split(',')
            .map(|t| t.trim().parse::<usize>().ok());
        let a = it.next()??;
        let b = it.next()??;
        self.bytes.get(self.data_start + a..self.data_start + b)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut recon_check: Option<String> = None;
    let mut positional: Vec<&String> = Vec::new();
    let mut print_roots = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--roots" => print_roots = true,
            "--recon-check" => {
                i += 1;
                recon_check = Some(
                    args.get(i)
                        .expect("--recon-check needs a recon.safetensors path")
                        .clone(),
                );
            }
            _ => positional.push(&args[i]),
        }
        i += 1;
    }
    let path = positional.first().expect(
        "usage: attest-strand <archive.strand> [--roots] [--recon-check <recon.safetensors>]",
    );

    let model = StrandModel::open(Path::new(path)).expect("attest-strand: open .strand v2 archive");
    let hdr = model.header();
    let n = hdr.tensors.len();
    let file_len = std::fs::metadata(path.as_str())
        .expect("stat archive")
        .len();
    println!("archive          {path}");
    println!(
        "tensors          {n}   strict={}   source_sha256={}",
        hdr.all_strict(),
        hex(&hdr.source_sha256)
    );
    assert!(n > 0, "attest-strand: archive contains no tensors");

    let n_weights: u64 = hdr.tensors.iter().map(|t| t.total as u64).sum();
    println!(
        "BILLING          file_len={file_len} B  n_weights={n_weights}  bytes/weight={:.4}  (whole file incl. OUTL+SPRV sections)",
        file_len as f64 / n_weights.max(1) as f64
    );

    match model.outl_section() {
        Some(o) => println!(
            "outliers         OUTL section present: {}/{} tensors carry a channel, {} entries total",
            o.n_with_channel(),
            n,
            o.total_entries()
        ),
        None => println!(
            "outliers         NO OUTL section — bulk-only archive (quality-bearing for outlier-channel configs ONLY when re-baked)"
        ),
    }

    match read_sprv(path.as_str()) {
        Ok(Some(sprv)) => {
            let lut_for = |tensor: &strand_quant::format::OwnedTensorV2| {
                model.lut_for(&tensor.base.name).map(|lut| lut.to_vec())
            };
            verify_archive_with(path.as_str(), VerifyDepth::Vectors, &lut_for)
                .expect("attest-strand: SPRV self-verification FAILED");
            println!(
                "sprv             trailer present (v2, R2 descriptor binding): self-verify (Vectors) PASS; stored model_root={}",
                hex(&sprv.model_root)
            );
        }
        Ok(None) => println!(
            "sprv             NO SPRV trailer — roots below are computed, not attested in-file"
        ),
        Err(e) => panic!("attest-strand: SPRV trailer present but corrupt: {e}"),
    }

    let mut spot_indices = vec![0usize, n / 2, n - 1];
    spot_indices.sort_unstable();
    spot_indices.dedup();
    for i in spot_indices {
        let label = if i == 0 {
            "first"
        } else if i == n - 1 {
            "last"
        } else {
            "middle"
        };
        let th = &hdr.tensors[i];
        let name = th.name.clone();
        let enc = model
            .encoded_tensor_checked(&name)
            .unwrap_or_else(|e| panic!("attest-strand: reconstruct {name}: {e}"));
        let cfg = model.config_for(th);
        let lut = model
            .lut_for(&name)
            .unwrap_or_else(|e| panic!("attest-strand: resolve LUT for {name}: {e}"));
        let q_fixed = decode_tensor_fixed_with_lut(&enc, &cfg, lut);
        let q_lean = decode_lean_with_lut(&enc, &cfg, lut);
        assert_eq!(
            q_fixed, q_lean,
            "DETERMINISM VIOLATION: fixed vs lean decode differ on {name}"
        );
        assert_eq!(
            q_fixed.len(),
            th.total,
            "decoded count != header total on {name}"
        );
        let nz = q_fixed.iter().filter(|&&q| q != 0).count();
        assert!(nz > 0, "degenerate all-zero Q12 stream in {name}");
        let mn = q_fixed.iter().copied().min().unwrap();
        let mx = q_fixed.iter().copied().max().unwrap();
        println!(
            "spot[{label:>6}] {name}  shape={:?} l={} k={} blocks={}  n={}  nonzero={:.2}%  q12range=[{mn},{mx}]  fixed==lean OK (i32: finite by construction)",
            th.shape,
            th.l_bits,
            th.k_bits,
            th.n_blocks,
            th.total,
            100.0 * nz as f64 / q_fixed.len() as f64,
        );
    }

    if let Some(recon_path) = &recon_check {
        let recon = ReconFile::open(recon_path);
        let mut checked = 0usize;
        for th in &hdr.tensors {
            let got = patched_weights(&model, &th.name)
                .unwrap_or_else(|e| panic!("recon-check: patched decode {}: {e}", th.name));
            let want = recon.tensor_bytes(&th.name).unwrap_or_else(|| {
                panic!("recon-check: tensor {} missing in {recon_path}", th.name)
            });
            if want.len() != got.len() * 4 {
                eprintln!(
                    "RECON-CHECK FAIL {}: {} archive weights vs {} recon bytes",
                    th.name,
                    got.len(),
                    want.len()
                );
                std::process::exit(1);
            }
            let mut got_bytes = Vec::with_capacity(want.len());
            for v in &got {
                got_bytes.extend_from_slice(&v.to_le_bytes());
            }
            if got_bytes != want {
                let first_bad = got_bytes
                    .iter()
                    .zip(want.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                eprintln!(
                    "RECON-CHECK FAIL {}: byte mismatch at offset {first_bad} (weight {})",
                    th.name,
                    first_bad / 4
                );
                std::process::exit(1);
            }
            checked += 1;
        }
        println!(
            "RECON-CHECK      PASS — all {checked} archive tensors byte-equal the recon path \
             (bulk Q12 + inverse RHT + outlier patch == quantize-model recon)"
        );
    }

    let mut roots: Vec<(String, [u8; 32])> = Vec::with_capacity(n);
    for th in &hdr.tensors {
        let enc = model
            .encoded_tensor_checked(&th.name)
            .unwrap_or_else(|e| panic!("attest-strand: reconstruct {}: {e}", th.name));
        let cfg = model.config_for(th);
        let lut = model
            .lut_for(&th.name)
            .unwrap_or_else(|e| panic!("attest-strand: resolve LUT for {}: {e}", th.name));
        let r = tensor_root(&enc, &cfg, lut);
        if print_roots {
            println!("tensor_root {}  {}", hex(&r), th.name);
        }
        roots.push((th.name.clone(), r));
    }
    let mroot = model_root_from_tensor_roots(roots.iter().map(|(name, r)| (name.as_str(), *r)));

    if let Ok(Some(sprv)) = read_sprv(path.as_str()) {
        assert_eq!(
            mroot, sprv.model_root,
            "attest-strand: recomputed model_root differs from the SPRV-stored root"
        );
        println!("model_root {}  (matches SPRV stored root)", hex(&mroot));
    } else {
        println!("model_root {}", hex(&mroot));
    }
}
