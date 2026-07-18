use std::io::Write;
use std::path::Path;

use strand_decode_kernel::loader::StrandModel;
use strand_decode_kernel::outlier_mac::patched_weights;

#[derive(Clone, Copy)]
enum OutputDtype {
    F32,
    Bf16,
}

impl OutputDtype {
    fn safetensors_name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Bf16 => "BF16",
        }
    }

    fn bytes_per_weight(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0x7f80_0000 && mantissa != 0 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut dtype = OutputDtype::F32;
    let mut positional: Vec<&str> = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--dtype" => {
                i += 1;
                dtype = match args.get(i).map(String::as_str) {
                    Some("f32") => OutputDtype::F32,
                    Some("bf16") => OutputDtype::Bf16,
                    Some(other) => panic!("unsupported --dtype {other}; expected f32 or bf16"),
                    None => panic!("--dtype requires f32 or bf16"),
                };
            }
            arg if arg.starts_with('-') => panic!("unknown option {arg}"),
            arg => positional.push(arg),
        }
        i += 1;
    }
    let [archive, out] = positional.as_slice() else {
        eprintln!(
            "usage: archive-to-safetensors <archive.strand> <out.safetensors> [--dtype f32|bf16]"
        );
        std::process::exit(2);
    };

    let model = StrandModel::open(Path::new(archive)).expect("open .strand archive");
    let names: Vec<String> = model.tensor_names().map(|s| s.to_string()).collect();
    eprintln!("[a2s] {} tensors in {archive}", names.len());

    let mut header = String::from("{");
    let mut payload_len = 0u64;
    for (i, name) in names.iter().enumerate() {
        let th = model.tensor_header(name).expect("tensor header");
        let shape: Vec<u64> = th.shape.clone();
        let start = payload_len;
        payload_len = payload_len
            .checked_add((th.total as u64) * dtype.bytes_per_weight())
            .expect("payload length overflow");
        if i > 0 {
            header.push(',');
        }
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            name,
            dtype.safetensors_name(),
            dims.join(","),
            start,
            payload_len
        ));
    }
    header.push('}');

    let mut f = std::fs::File::create(out).expect("create out safetensors");
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    for (i, name) in names.iter().enumerate() {
        let weights =
            patched_weights(&model, name).unwrap_or_else(|e| panic!("patched decode {name}: {e}"));
        for chunk in weights.chunks(1 << 20) {
            let mut bytes = Vec::with_capacity(chunk.len() * dtype.bytes_per_weight() as usize);
            match dtype {
                OutputDtype::F32 => {
                    for value in chunk {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
                OutputDtype::Bf16 => {
                    for value in chunk {
                        bytes.extend_from_slice(&f32_to_bf16_bits(*value).to_le_bytes());
                    }
                }
            }
            f.write_all(&bytes).expect("write decoded tensor payload");
        }
        if (i + 1) % 32 == 0 || i + 1 == names.len() {
            eprintln!("[a2s] decoded {}/{}", i + 1, names.len());
        }
    }
    f.sync_all().ok();
    eprintln!(
        "[a2s] wrote {out}: {} tensors, {} payload bytes ({}, patched decode)",
        names.len(),
        payload_len,
        dtype.safetensors_name()
    );
}
